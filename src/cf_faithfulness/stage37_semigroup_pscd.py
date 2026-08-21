"""Numerical core for Stage 37 semigroup-regularized PSCD.

Stage 37 keeps the Stage 36 architecture fixed and changes the training
criterion.  It applies the learned transition from every eligible native
history anchor and penalizes disagreement with the directly encoded future at
registered horizons.  The module is independent of PushT and JEPA-WM so the
training, controls, planning metrics, and decision semantics can be exercised
locally on synthetic systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
import torch

from cf_faithfulness.stage36_predictive_state_closure import (
    PredictiveStateClosureModel,
    _artifact_model,
    _mean_scale,
    history_tensor,
    next_history_tensor,
    rollout_predictive_state_closure,
    sequence_source_states,
)


FloatArray = NDArray[np.float64]


def registered_semigroup_horizons(
    mask: ArrayLike,
    history_length: int,
    horizons: Sequence[int],
) -> list[tuple[int, int]]:
    """Return every anchor/horizon pair with at least one complete path."""

    valid = np.asarray(mask, dtype=bool)
    if valid.ndim != 2:
        raise ValueError("mask must be a sequence matrix")
    history = int(history_length)
    ordered = sorted({int(value) for value in horizons})
    if history < 1 or not ordered or ordered[0] < 2:
        raise ValueError("history must be positive and semigroup horizons at least two")
    pairs: list[tuple[int, int]] = []
    # Anchor zero is a legitimate cold start: its finite history is the
    # current carrier left-padded into all history slots.  Including it is
    # required for planning, where no candidate-specific future prefix exists.
    start = 0
    for anchor in range(start, valid.shape[1]):
        for horizon in ordered:
            stop = anchor + horizon
            if stop > valid.shape[1]:
                continue
            complete = np.all(valid[:, anchor:stop], axis=1)
            if np.any(complete):
                pairs.append((anchor, horizon))
    if not pairs:
        raise ValueError("no registered semigroup horizon fits the sequence panel")
    return pairs


def fit_semigroup_predictive_state_closure(
    initial: ArrayLike,
    actions: ArrayLike,
    carrier_targets: ArrayLike,
    physical_targets: ArrayLike,
    mask: ArrayLike,
    *,
    history_length: int,
    latent_dim: int,
    dynamics: str,
    epochs: int,
    learning_rate: float,
    seed: int,
    semigroup_horizons: Sequence[int] = (2, 4, 8),
    semigroup_weight: float = 1.0,
    free_weight: float = 1.0,
    consistency_weight: float = 0.25,
    device: str | None = None,
    histories_override: ArrayLike | None = None,
) -> dict[str, Any]:
    """Fit PSCD with explicit multi-anchor direct/composed agreement.

    The target encoder is stop-gradient.  The transition is rolled from every
    eligible native-history anchor at the registered horizons, so the added
    term cannot be satisfied only by one rollout beginning at a single global
    warmup boundary.
    """

    first = np.asarray(initial, dtype=np.float32)
    action = np.asarray(actions, dtype=np.float32)
    carrier = np.asarray(carrier_targets, dtype=np.float32)
    physical = np.asarray(physical_targets, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool)
    if first.ndim != 2 or action.ndim != 3 or carrier.ndim != 3 or physical.ndim != 3:
        raise ValueError("S-PSCD arrays have invalid ranks")
    if action.shape[:2] != valid.shape or carrier.shape[:2] != valid.shape:
        raise ValueError("S-PSCD sequence arrays are not aligned")
    if physical.shape[:2] != valid.shape or len(first) != len(action):
        raise ValueError("S-PSCD target arrays are not aligned")
    if float(semigroup_weight) < 0 or float(free_weight) < 0:
        raise ValueError("loss weights must be nonnegative")

    history = history_tensor(first, carrier, valid, history_length).astype(np.float32)
    if histories_override is not None:
        override = np.asarray(histories_override, dtype=np.float32)
        if override.shape != history.shape:
            raise ValueError("history override has the wrong shape")
        history = override
    next_history = next_history_tensor(history, carrier).astype(np.float32)
    source = sequence_source_states(first, carrier).astype(np.float32)
    active = valid
    carrier_mean, carrier_scale = _mean_scale(
        np.concatenate([source[active], carrier[active]], axis=0)
    )
    action_mean, action_scale = _mean_scale(action[active])
    physical_mean, physical_scale = _mean_scale(physical[active])
    history_n = (history - carrier_mean) / carrier_scale
    next_history_n = (next_history - carrier_mean) / carrier_scale
    carrier_n = (carrier - carrier_mean) / carrier_scale
    action_n = (action - action_mean) / action_scale
    physical_n = (physical - physical_mean) / physical_scale

    pairs: list[tuple[int, int]] = []
    if float(semigroup_weight) > 0:
        pairs = registered_semigroup_horizons(valid, history_length, semigroup_horizons)
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model = PredictiveStateClosureModel(
        first.shape[1], int(history_length), action.shape[2], physical.shape[2],
        int(latent_dim), str(dynamics),
    ).to(selected_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=1e-4
    )
    tensors = {
        "history": torch.as_tensor(history_n, device=selected_device),
        "next_history": torch.as_tensor(next_history_n, device=selected_device),
        "action": torch.as_tensor(action_n, device=selected_device),
        "carrier": torch.as_tensor(carrier_n, device=selected_device),
        "physical": torch.as_tensor(physical_n, device=selected_device),
        "mask": torch.as_tensor(valid, device=selected_device),
    }
    pair_masks = {
        (anchor, horizon): torch.as_tensor(
            np.all(valid[:, anchor : anchor + horizon], axis=1),
            device=selected_device,
        )
        for anchor, horizon in pairs
    }
    start = int(history_length) - 1
    losses: list[float] = []
    final_components: dict[str, float] = {}
    model.train()
    for _epoch in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        source_state = model.encode(tensors["history"])
        with torch.no_grad():
            target_state = model.encode(tensors["next_history"])
        one_state = model.transition(source_state, tensors["action"])
        one_carrier, one_physical = model.decode(one_state)
        observed = tensors["mask"]
        carrier_loss = torch.mean(
            (one_carrier[observed] - tensors["carrier"][observed]) ** 2
        )
        physical_loss = torch.mean(
            (one_physical[observed] - tensors["physical"][observed]) ** 2
        )
        consistency = torch.mean((one_state[observed] - target_state[observed]) ** 2)

        recursive_terms = {"carrier": [], "physical": [], "state": []}
        state = source_state[:, start]
        for step in range(start, action.shape[1]):
            step_valid = observed[:, step]
            if not bool(torch.any(step_valid)):
                continue
            updated = model.transition(state[step_valid], tensors["action"][step_valid, step])
            state = state.clone()
            state[step_valid] = updated
            decoded_carrier, decoded_physical = model.decode(updated)
            recursive_terms["carrier"].append(torch.mean(
                (decoded_carrier - tensors["carrier"][step_valid, step]) ** 2
            ))
            recursive_terms["physical"].append(torch.mean(
                (decoded_physical - tensors["physical"][step_valid, step]) ** 2
            ))
            recursive_terms["state"].append(torch.mean(
                (updated - target_state[step_valid, step]) ** 2
            ))

        zero = torch.zeros((), device=selected_device)
        recursive = {
            key: torch.stack(values).mean() if values else zero
            for key, values in recursive_terms.items()
        }
        semigroup_terms = {"carrier": [], "physical": [], "state": []}
        if pairs:
            pairs_by_anchor: dict[int, list[int]] = {}
            for anchor, horizon in pairs:
                pairs_by_anchor.setdefault(anchor, []).append(horizon)
            for anchor, anchor_horizons in pairs_by_anchor.items():
                composed = source_state[:, anchor]
                maximum = max(anchor_horizons)
                for offset in range(maximum):
                    step = anchor + offset
                    active_path = tensors["mask"][:, anchor : step + 1].all(dim=1)
                    if not bool(torch.any(active_path)):
                        continue
                    update = model.transition(
                        composed[active_path], tensors["action"][active_path, step]
                    )
                    composed = composed.clone()
                    composed[active_path] = update
                    horizon = offset + 1
                    if horizon not in anchor_horizons:
                        continue
                    eligible = pair_masks[(anchor, horizon)]
                    endpoint = anchor + horizon - 1
                    direct = target_state[eligible, endpoint]
                    predicted = composed[eligible]
                    decoded_carrier, decoded_physical = model.decode(predicted)
                    semigroup_terms["state"].append(torch.mean((predicted - direct) ** 2))
                    semigroup_terms["carrier"].append(torch.mean(
                        (decoded_carrier - tensors["carrier"][eligible, endpoint]) ** 2
                    ))
                    semigroup_terms["physical"].append(torch.mean(
                        (decoded_physical - tensors["physical"][eligible, endpoint]) ** 2
                    ))
        semigroup = {
            key: torch.stack(values).mean() if values else zero
            for key, values in semigroup_terms.items()
        }
        free_loss = (
            0.45 * recursive["carrier"]
            + 0.25 * recursive["physical"]
            + 0.20 * recursive["state"]
        )
        semigroup_loss = (
            0.35 * semigroup["carrier"]
            + 0.20 * semigroup["physical"]
            + 0.45 * semigroup["state"]
        )
        loss = (
            0.45 * carrier_loss
            + 0.25 * physical_loss
            + float(consistency_weight) * consistency
            + float(free_weight) * free_loss
            + float(semigroup_weight) * semigroup_loss
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("S-PSCD training became nonfinite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        final_components = {
            "one_carrier": float(carrier_loss.detach().cpu()),
            "one_physical": float(physical_loss.detach().cpu()),
            "one_state": float(consistency.detach().cpu()),
            "free": float(free_loss.detach().cpu()),
            "semigroup": float(semigroup_loss.detach().cpu()),
        }
    state_dict = {
        key: value.detach().cpu().numpy().astype(np.float32)
        for key, value in model.state_dict().items()
    }
    return {
        "config": {
            "carrier_dim": int(first.shape[1]),
            "history_length": int(history_length),
            "action_dim": int(action.shape[2]),
            "physical_dim": int(physical.shape[2]),
            "latent_dim": int(latent_dim),
            "dynamics": str(dynamics),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "free_weight": float(free_weight),
            "consistency_weight": float(consistency_weight),
            "semigroup_weight": float(semigroup_weight),
            "semigroup_horizons": [int(value) for value in semigroup_horizons],
            "semigroup_anchor_pairs": len(pairs),
            "seed": int(seed),
        },
        "normalization": {
            "carrier_mean": carrier_mean,
            "carrier_scale": carrier_scale,
            "action_mean": action_mean,
            "action_scale": action_scale,
            "physical_mean": physical_mean,
            "physical_scale": physical_scale,
        },
        "state_dict": state_dict,
        "loss_initial": float(losses[0]),
        "loss_final": float(losses[-1]),
        "loss_components_final": final_components,
    }


def rollout_predictive_state_from_initial(
    artifact: Mapping[str, Any],
    initial: ArrayLike,
    actions: ArrayLike,
    mask: ArrayLike,
    *,
    device: str | None = None,
) -> dict[str, FloatArray]:
    """Recurse from the current carrier without future teacher warmup.

    The current carrier is repeated across the registered history slots.  This
    is the only valid Stage 37 rollout for ranking action candidates before any
    candidate action has been executed.
    """

    config = artifact["config"]
    normalization = artifact["normalization"]
    first = np.asarray(initial, dtype=np.float32)
    action = np.asarray(actions, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool)
    if first.ndim != 2 or action.ndim != 3 or action.shape[:2] != valid.shape:
        raise ValueError("cold-start rollout arrays are not aligned")
    if len(first) != len(action) or first.shape[1] != int(config["carrier_dim"]):
        raise ValueError("cold-start carrier shape changed")
    carrier_mean = np.asarray(normalization["carrier_mean"], dtype=np.float32)
    carrier_scale = np.asarray(normalization["carrier_scale"], dtype=np.float32)
    action_mean = np.asarray(normalization["action_mean"], dtype=np.float32)
    action_scale = np.asarray(normalization["action_scale"], dtype=np.float32)
    physical_mean = np.asarray(normalization["physical_mean"], dtype=np.float32)
    physical_scale = np.asarray(normalization["physical_scale"], dtype=np.float32)
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = _artifact_model(artifact, selected_device)
    cold_history = np.repeat(
        first[:, None, :], int(config["history_length"]), axis=1
    )
    history_n = torch.as_tensor(
        (cold_history - carrier_mean) / carrier_scale, device=selected_device
    )
    action_n = torch.as_tensor(
        (action - action_mean) / action_scale, device=selected_device
    )
    output_carrier = np.zeros(
        (*action.shape[:2], int(config["carrier_dim"])), dtype=np.float32
    )
    output_physical = np.zeros(
        (*action.shape[:2], int(config["physical_dim"])), dtype=np.float32
    )
    output_state = np.zeros(
        (*action.shape[:2], int(config["latent_dim"])), dtype=np.float32
    )
    with torch.inference_mode():
        state = model.encode(history_n)
        for step in range(action.shape[1]):
            active = valid[:, step]
            if not np.any(active):
                continue
            active_tensor = torch.as_tensor(active, device=selected_device)
            updated = model.transition(state[active_tensor], action_n[active_tensor, step])
            state = state.clone()
            state[active_tensor] = updated
            decoded_carrier, decoded_physical = model.decode(updated)
            output_carrier[active, step] = (
                decoded_carrier.cpu().numpy() * carrier_scale + carrier_mean
            )
            output_physical[active, step] = (
                decoded_physical.cpu().numpy() * physical_scale + physical_mean
            )
            output_state[active, step] = updated.cpu().numpy()
    return {
        "carrier": output_carrier.astype(np.float64),
        "physical": output_physical.astype(np.float64),
        "state": output_state.astype(np.float64),
        "evaluation_mask": valid.copy(),
    }


def select_semigroup_candidate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("semigroup selection requires at least one candidate")
    required = {
        "semigroup_weight", "validation_score", "recursive_physical_nmse",
        "semigroup_nmse",
    }
    candidates = []
    for row in rows:
        if not required.issubset(row):
            raise ValueError("semigroup candidate row is incomplete")
        if not np.isfinite(float(row["validation_score"])):
            raise ValueError("semigroup candidate score is nonfinite")
        candidates.append(dict(row))
    candidates.sort(key=lambda row: (
        float(row["validation_score"]),
        float(row["semigroup_nmse"]),
        float(row["semigroup_weight"]),
    ))
    return candidates[0]


def terminal_values(path: ArrayLike, mask: ArrayLike) -> FloatArray:
    values = np.asarray(path, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if values.ndim != 3 or values.shape[:2] != valid.shape:
        raise ValueError("terminal path inputs are not aligned")
    indices = np.max(
        np.where(valid, np.arange(valid.shape[1])[None, :], -1), axis=1
    )
    if np.any(indices < 0):
        raise ValueError("every path needs a valid endpoint")
    return values[np.arange(len(values)), indices]


def goal_cost(
    endpoints: ArrayLike,
    goals: ArrayLike,
    scale: ArrayLike,
    dimensions: Sequence[int] = (2, 3, 4, 5),
) -> FloatArray:
    predicted = np.asarray(endpoints, dtype=np.float64)
    target = np.asarray(goals, dtype=np.float64)
    width_scale = np.asarray(scale, dtype=np.float64)
    dims = np.asarray([int(value) for value in dimensions], dtype=np.int64)
    if predicted.ndim != 2 or target.shape != predicted.shape:
        raise ValueError("planning endpoints and goals must be aligned matrices")
    if width_scale.ndim != 1 or predicted.shape[1] != len(width_scale):
        raise ValueError("planning scale does not align with endpoints")
    if np.any(dims < 0) or np.any(dims >= predicted.shape[1]):
        raise ValueError("planning dimension is outside the endpoint schema")
    delta = (predicted[:, dims] - target[:, dims]) / np.maximum(width_scale[dims], 1e-8)
    return np.mean(delta**2, axis=1)


def grouped_planner_metrics(
    predicted_cost: ArrayLike,
    true_cost: ArrayLike,
    groups: ArrayLike,
) -> dict[str, np.ndarray]:
    predicted = np.asarray(predicted_cost, dtype=np.float64)
    truth = np.asarray(true_cost, dtype=np.float64)
    group = np.asarray(groups)
    if predicted.ndim != 1 or predicted.shape != truth.shape or len(group) != len(truth):
        raise ValueError("planner cost inputs must be aligned vectors")
    regret, success, pairwise, selected = [], [], [], []
    ordered_groups = np.unique(group)
    for value in ordered_groups:
        rows = np.flatnonzero(group == value)
        if len(rows) < 2:
            raise ValueError("each planning group needs at least two candidates")
        local_selected = int(np.argmin(predicted[rows]))
        local_oracle = int(np.argmin(truth[rows]))
        selected.append(int(rows[local_selected]))
        regret.append(float(truth[rows[local_selected]] - truth[rows[local_oracle]]))
        success.append(float(local_selected == local_oracle))
        agreements = []
        for left in range(len(rows)):
            for right in range(left + 1, len(rows)):
                true_delta = truth[rows[left]] - truth[rows[right]]
                predicted_delta = predicted[rows[left]] - predicted[rows[right]]
                if abs(true_delta) <= 1e-12:
                    continue
                agreements.append(float(np.sign(true_delta) == np.sign(predicted_delta)))
        pairwise.append(float(np.mean(agreements)) if agreements else 1.0)
    return {
        "groups": np.asarray(ordered_groups),
        "regret": np.asarray(regret, dtype=np.float64),
        "success": np.asarray(success, dtype=np.float64),
        "pairwise_accuracy": np.asarray(pairwise, dtype=np.float64),
        "selected_rows": np.asarray(selected, dtype=np.int64),
    }


@dataclass(frozen=True)
class Stage37Gates:
    source_and_split_binding: bool
    simulator_positive_control: bool
    native_physical_fidelity: bool
    semigroup_regularization_advantage: bool
    recursive_closure: bool
    planning_value: bool
    history_specificity: bool
    family_consistency: bool


def derive_stage37_decision(gates: Stage37Gates, *, run_mode: str) -> dict[str, Any]:
    checks = {
        "source_and_split_binding": bool(gates.source_and_split_binding),
        "simulator_positive_control": bool(gates.simulator_positive_control),
        "native_physical_fidelity": bool(gates.native_physical_fidelity),
        "semigroup_regularization_advantage": bool(
            gates.semigroup_regularization_advantage
        ),
        "recursive_closure": bool(gates.recursive_closure),
        "planning_value": bool(gates.planning_value),
        "history_specificity": bool(gates.history_specificity),
        "family_consistency": bool(gates.family_consistency),
    }
    first_failed = next((name for name, passed in checks.items() if not passed), None)
    if str(run_mode) == "smoke":
        status, passed = "smoke_complete_not_evidence", False
    elif first_failed is None:
        status, passed = "semigroup_pscd_closure_and_planning_value_observed", True
    else:
        labels = {
            "source_and_split_binding": "invalid_source_or_split_binding",
            "simulator_positive_control": "operator_class_failed_positive_control",
            "native_physical_fidelity": "native_jepa_not_physically_faithful",
            "semigroup_regularization_advantage": "semigroup_objective_not_specific",
            "recursive_closure": "recursive_state_closure_not_observed",
            "planning_value": "closure_gain_did_not_improve_open_loop_planning",
            "history_specificity": "history_not_predictively_specific",
            "family_consistency": "result_not_family_consistent",
        }
        status, passed = labels[first_failed], False
    return {
        "status": status,
        "passed": passed,
        "first_failed_gate": first_failed,
        "gates": checks,
        "causal_evidence": False,
        "closed_loop_planning_claimed": False,
        "original_jepa_state_claimed_closed": False,
        "adapter_repair_only": True,
    }
