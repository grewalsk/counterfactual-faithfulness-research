"""Numerical core for Stage 38 cross-model PSCD confirmation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
import torch

from cf_faithfulness.stage36_predictive_state_closure import (
    PredictiveStateClosureModel,
    _mean_scale,
    history_tensor,
    next_history_tensor,
    sequence_source_states,
)
from cf_faithfulness.stage37_semigroup_pscd import registered_semigroup_horizons


FloatArray = NDArray[np.float64]


def fit_weighted_semigroup_predictive_state_closure(
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
    semigroup_component_weights: Sequence[float] = (0.35, 0.20, 0.45),
    free_weight: float = 1.0,
    consistency_weight: float = 0.25,
    device: str | None = None,
    histories_override: ArrayLike | None = None,
) -> dict[str, Any]:
    """Fit a recurrent closure model with registered semigroup components.

    Component order is carrier, grounded physical state, and latent state.  A
    state-only setting is the deterministic analogue of latent overshooting;
    the full setting is S-PSCD.
    """

    first = np.asarray(initial, dtype=np.float32)
    action = np.asarray(actions, dtype=np.float32)
    carrier = np.asarray(carrier_targets, dtype=np.float32)
    physical = np.asarray(physical_targets, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool)
    if first.ndim != 2 or action.ndim != 3 or carrier.ndim != 3 or physical.ndim != 3:
        raise ValueError("Stage 38 closure arrays have invalid ranks")
    if action.shape[:2] != valid.shape or carrier.shape[:2] != valid.shape:
        raise ValueError("Stage 38 sequence arrays are not aligned")
    if physical.shape[:2] != valid.shape or len(first) != len(action):
        raise ValueError("Stage 38 physical arrays are not aligned")
    weights = np.asarray(semigroup_component_weights, dtype=np.float64)
    if weights.shape != (3,) or np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("semigroup component weights must be three nonnegative values")
    if float(semigroup_weight) > 0 and float(np.sum(weights)) <= 0:
        raise ValueError("a positive semigroup objective requires a component")
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
                for offset in range(max(anchor_horizons)):
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
        weight_sum = max(float(np.sum(weights)), 1e-12)
        semigroup_loss = (
            float(weights[0]) * semigroup["carrier"]
            + float(weights[1]) * semigroup["physical"]
            + float(weights[2]) * semigroup["state"]
        ) / weight_sum
        loss = (
            0.45 * carrier_loss
            + 0.25 * physical_loss
            + float(consistency_weight) * consistency
            + float(free_weight) * free_loss
            + float(semigroup_weight) * semigroup_loss
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Stage 38 closure training became nonfinite")
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
            "semigroup_component_weights": weights.tolist(),
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


def select_stage38_semigroup_candidate(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Stage 38 selection requires candidate rows")
    required = {"semigroup_weight", "physical_nmse", "semigroup_nmse", "score"}
    candidates = []
    for row in rows:
        if not required.issubset(row):
            raise ValueError("Stage 38 candidate row is incomplete")
        values = [float(row[key]) for key in ["physical_nmse", "semigroup_nmse", "score"]]
        if not np.all(np.isfinite(values)):
            raise ValueError("Stage 38 candidate row is nonfinite")
        candidates.append(dict(row))
    candidates.sort(key=lambda row: (float(row["score"]), float(row["semigroup_weight"])))
    return candidates[0]


def tail_risk_summary(errors: ArrayLike) -> dict[str, float]:
    values = np.asarray(errors, dtype=np.float64).reshape(-1)
    if not len(values) or np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("tail errors must be finite and nonnegative")
    cutoff = float(np.quantile(values, 0.95))
    tail = values[values >= cutoff]
    return {
        "median": float(np.median(values)),
        "p95": cutoff,
        "cvar95": float(np.mean(tail)),
        "catastrophic_rate_gt_1": float(np.mean(values > 1.0)),
        "maximum": float(np.max(values)),
    }


def hierarchical_seed_trajectory_interval(
    values: ArrayLike,
    groups: ArrayLike,
    *,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap training seeds and physical trajectory clusters."""

    matrix = np.asarray(values, dtype=np.float64)
    labels = np.asarray(groups)
    if matrix.ndim != 2 or matrix.shape[1] != len(labels):
        raise ValueError("hierarchical bootstrap arrays are not aligned")
    unique = np.unique(labels)
    if matrix.shape[0] < 2 or len(unique) < 2 or int(draws) < 2:
        raise ValueError("hierarchical bootstrap needs multiple seeds, groups, and draws")
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(int(draws), dtype=np.float64)
    group_rows = {group: np.flatnonzero(labels == group) for group in unique}
    for draw in range(int(draws)):
        sampled_seeds = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        sampled_groups = rng.choice(unique, size=len(unique), replace=True)
        pieces = [
            matrix[seed_index, group_rows[group]]
            for seed_index in sampled_seeds for group in sampled_groups
        ]
        estimates[draw] = float(np.mean(np.concatenate(pieces)))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


@dataclass(frozen=True)
class Stage38ModelGates:
    native_fidelity: bool
    absolute_recursive_closure: bool
    repair_advantage: bool
    semigroup_specificity: bool
    overshooting_noninferiority: bool
    recursion_and_history_specificity: bool
    seed_consistency: bool
    horizon_mode_tail_consistency: bool


@dataclass(frozen=True)
class Stage38Gates:
    source_and_split_binding: bool
    simulator_positive_control: bool
    jepa_confirmation: bool
    dino_replication: bool
    planning_value: bool


def derive_stage38_model_decision(gates: Stage38ModelGates) -> dict[str, Any]:
    checks = {name: bool(getattr(gates, name)) for name in gates.__dataclass_fields__}
    first_failed = next((name for name, passed in checks.items() if not passed), None)
    return {"passed": first_failed is None, "first_failed_gate": first_failed, "gates": checks}


def derive_stage38_decision(gates: Stage38Gates, *, run_mode: str) -> dict[str, Any]:
    checks = {name: bool(getattr(gates, name)) for name in gates.__dataclass_fields__}
    first_failed = next((name for name, passed in checks.items() if not passed), None)
    closure_passed = all(checks[name] for name in [
        "source_and_split_binding", "simulator_positive_control",
        "jepa_confirmation", "dino_replication",
    ])
    if str(run_mode) == "smoke":
        status, passed = "smoke_complete_not_evidence", False
    elif not closure_passed:
        labels = {
            "source_and_split_binding": "invalid_source_or_split_binding",
            "simulator_positive_control": "simulator_operator_failed",
            "jepa_confirmation": "jepa_pscd_confirmation_failed",
            "dino_replication": "dino_pscd_replication_failed",
        }
        status, passed = labels[first_failed], False
    elif checks["planning_value"]:
        status, passed = "cross_model_pscd_closure_and_planning_confirmed", True
    else:
        status, passed = "cross_model_pscd_closure_confirmed_without_planning_value", True
    return {
        "status": status,
        "passed": passed,
        "closure_confirmed": bool(closure_passed),
        "planning_confirmed": bool(closure_passed and checks["planning_value"]),
        "first_failed_gate": first_failed,
        "gates": checks,
        "causal_claimed": False,
        "native_jepa_closure_claimed": False,
        "cross_environment_claimed": False,
    }
