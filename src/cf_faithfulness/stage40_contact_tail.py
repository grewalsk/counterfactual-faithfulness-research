"""Numerical core for Stage 40 contact-tail risk distillation.

The repair changes only training weights.  Simulator-derived contact labels are
permitted on construction, model-selection, and calibration splits and are
never required at inference.  Evaluation labels are opened only after model
selection and final fitting are frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike
import torch

from cf_faithfulness.stage36_predictive_state_closure import (
    PredictiveStateClosureModel,
    _mean_scale,
    history_tensor,
    next_history_tensor,
    sequence_source_states,
)
from cf_faithfulness.stage37_semigroup_pscd import registered_semigroup_horizons


def contact_transition_weights(
    initial_modes: ArrayLike,
    target_modes: ArrayLike,
    mask: ArrayLike,
    *,
    contact_multiplier: float,
) -> np.ndarray:
    """Return mean-one training weights for contact and contact re-entry.

    A contact endpoint receives ``contact_multiplier``.  A
    post-contact-to-contact re-entry receives ``2 * multiplier - 1``.  Thus a
    multiplier of one is exactly the uniform control.
    """

    initial = np.asarray(initial_modes).astype(str)
    target = np.asarray(target_modes).astype(str)
    valid = np.asarray(mask, dtype=bool)
    multiplier = float(contact_multiplier)
    if initial.ndim != 1 or target.ndim != 2 or valid.shape != target.shape:
        raise ValueError("contact mode arrays are not aligned")
    if len(initial) != len(target) or not np.isfinite(multiplier) or multiplier < 1:
        raise ValueError("contact multiplier must be finite and at least one")
    source = np.empty_like(target)
    source[:, 0] = initial
    source[:, 1:] = target[:, :-1]
    contact = valid & (target == "contact")
    reentry = contact & (source == "post_contact")
    weights = np.ones(valid.shape, dtype=np.float32)
    weights[contact] += multiplier - 1.0
    weights[reentry] += multiplier - 1.0
    weights[~valid] = 0.0
    mean = float(np.mean(weights[valid]))
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError("contact weights have invalid support")
    weights[valid] /= mean
    return weights


def _weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.shape[0] != weights.shape[0]:
        raise ValueError("weighted loss tensors are not aligned")
    value = (prediction - target) ** 2
    weight = weights
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(-1)
    feature_count = int(np.prod(value.shape[weights.ndim :])) or 1
    denominator = torch.clamp(weights.sum() * feature_count, min=1e-12)
    return torch.sum(value * weight) / denominator


def fit_contact_risk_predictive_state_closure(
    initial: ArrayLike,
    actions: ArrayLike,
    carrier_targets: ArrayLike,
    physical_targets: ArrayLike,
    mask: ArrayLike,
    risk_weights: ArrayLike,
    *,
    history_length: int,
    latent_dim: int,
    dynamics: str,
    epochs: int,
    learning_rate: float,
    seed: int,
    semigroup_horizons: Sequence[int] = (2, 4, 8),
    semigroup_weight: float = 1.0,
    semigroup_component_weights: Sequence[float] = (0.0, 0.0, 1.0),
    free_weight: float = 1.0,
    consistency_weight: float = 0.25,
    device: str | None = None,
) -> dict[str, Any]:
    """Fit the Stage 39 recursive adapter with a locked contact-risk measure."""

    first = np.asarray(initial, dtype=np.float32)
    action = np.asarray(actions, dtype=np.float32)
    carrier = np.asarray(carrier_targets, dtype=np.float32)
    physical = np.asarray(physical_targets, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool)
    risk = np.asarray(risk_weights, dtype=np.float32)
    if first.ndim != 2 or action.ndim != 3 or carrier.ndim != 3 or physical.ndim != 3:
        raise ValueError("Stage 40 closure arrays have invalid ranks")
    if action.shape[:2] != valid.shape or carrier.shape[:2] != valid.shape:
        raise ValueError("Stage 40 sequence arrays are not aligned")
    if physical.shape[:2] != valid.shape or risk.shape != valid.shape:
        raise ValueError("Stage 40 physical or risk arrays are not aligned")
    if len(first) != len(action) or np.any(~np.isfinite(risk[valid])):
        raise ValueError("Stage 40 risk array is invalid")
    if np.any(risk[valid] <= 0) or np.any(risk[~valid] != 0):
        raise ValueError("Stage 40 risk weights need positive valid and zero padded entries")
    weights = np.asarray(semigroup_component_weights, dtype=np.float64)
    if weights.shape != (3,) or np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("semigroup component weights must be three nonnegative values")
    if float(semigroup_weight) > 0 and float(np.sum(weights)) <= 0:
        raise ValueError("a positive semigroup objective requires a component")
    if float(semigroup_weight) < 0 or float(free_weight) < 0:
        raise ValueError("loss weights must be nonnegative")

    history = history_tensor(first, carrier, valid, history_length).astype(np.float32)
    next_history = next_history_tensor(history, carrier).astype(np.float32)
    source = sequence_source_states(first, carrier).astype(np.float32)
    carrier_mean, carrier_scale = _mean_scale(
        np.concatenate([source[valid], carrier[valid]], axis=0)
    )
    action_mean, action_scale = _mean_scale(action[valid])
    physical_mean, physical_scale = _mean_scale(physical[valid])
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
        "risk": torch.as_tensor(risk, device=selected_device),
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
        observed_risk = tensors["risk"][observed]
        carrier_loss = _weighted_mse(
            one_carrier[observed], tensors["carrier"][observed], observed_risk
        )
        physical_loss = _weighted_mse(
            one_physical[observed], tensors["physical"][observed], observed_risk
        )
        consistency = _weighted_mse(
            one_state[observed], target_state[observed], observed_risk
        )

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
            step_risk = tensors["risk"][step_valid, step]
            recursive_terms["carrier"].append(_weighted_mse(
                decoded_carrier, tensors["carrier"][step_valid, step], step_risk
            ))
            recursive_terms["physical"].append(_weighted_mse(
                decoded_physical, tensors["physical"][step_valid, step], step_risk
            ))
            recursive_terms["state"].append(_weighted_mse(
                updated, target_state[step_valid, step], step_risk
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
                    predicted = composed[eligible]
                    direct = target_state[eligible, endpoint]
                    endpoint_risk = tensors["risk"][eligible, endpoint]
                    decoded_carrier, decoded_physical = model.decode(predicted)
                    semigroup_terms["state"].append(
                        _weighted_mse(predicted, direct, endpoint_risk)
                    )
                    semigroup_terms["carrier"].append(_weighted_mse(
                        decoded_carrier, tensors["carrier"][eligible, endpoint], endpoint_risk
                    ))
                    semigroup_terms["physical"].append(_weighted_mse(
                        decoded_physical, tensors["physical"][eligible, endpoint], endpoint_risk
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
            raise FloatingPointError("Stage 40 contact-risk training became nonfinite")
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
            "risk_weight_mean": float(np.mean(risk[valid])),
            "risk_weight_max": float(np.max(risk[valid])),
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


def select_contact_risk_candidate(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_mean_ratio: float = 1.05,
) -> dict[str, Any]:
    """Select the lowest validation p95 among mean-noninferior candidates."""

    required = {
        "contact_multiplier", "mean_nmse", "p95_nmse",
        "terminal_contact_nmse", "catastrophic_rate",
    }
    candidates = [dict(row) for row in rows]
    if not candidates or any(not required.issubset(row) for row in candidates):
        raise ValueError("Stage 40 selection rows are incomplete")
    baseline_rows = [row for row in candidates if float(row["contact_multiplier"]) == 1.0]
    if len(baseline_rows) != 1:
        raise ValueError("Stage 40 selection needs exactly one uniform baseline")
    baseline = float(baseline_rows[0]["mean_nmse"])
    if not np.isfinite(baseline) or baseline <= 0 or float(max_mean_ratio) < 1:
        raise ValueError("Stage 40 selection baseline is invalid")
    eligible = []
    for row in candidates:
        values = [float(row[key]) for key in required]
        if not np.all(np.isfinite(values)) or any(value < 0 for value in values):
            raise ValueError("Stage 40 selection contains invalid metrics")
        row["mean_noninferior_to_uniform"] = bool(
            float(row["mean_nmse"]) <= baseline * float(max_mean_ratio)
        )
        if row["mean_noninferior_to_uniform"]:
            eligible.append(row)
    eligible.sort(key=lambda row: (
        float(row["p95_nmse"]),
        float(row["terminal_contact_nmse"]),
        float(row["mean_nmse"]),
        float(row["contact_multiplier"]),
    ))
    return eligible[0]


@dataclass(frozen=True)
class Stage40PanelDecision:
    mean_gain: float
    interval90: tuple[float, float]
    mean_noninferiority: bool
    p95_improvement: bool
    contact_improvement: bool
    absolute_tail_qualified: bool
    quality_control_passed: bool
    classification: str


def derive_stage40_panel_decision(
    mean_gain: float,
    interval90: Sequence[float],
    *,
    p95_improvement: bool,
    contact_improvement: bool,
    absolute_tail_qualified: bool,
    quality_control_passed: bool = True,
    noninferiority_margin: float = 0.05,
) -> Stage40PanelDecision:
    interval = tuple(float(value) for value in interval90)
    mean = float(mean_gain)
    if len(interval) != 2 or interval[0] > interval[1]:
        raise ValueError("Stage 40 interval is invalid")
    if not np.all(np.isfinite([mean, *interval])) or float(noninferiority_margin) <= 0:
        raise ValueError("Stage 40 decision inputs are invalid")
    noninferior = bool(interval[0] >= -float(noninferiority_margin))
    if not quality_control_passed:
        classification = "invalid_quality_control"
    elif noninferior and p95_improvement and contact_improvement and absolute_tail_qualified:
        classification = "contact_tail_repair_confirmed"
    elif noninferior and p95_improvement and contact_improvement:
        classification = "tail_improved_but_not_absolutely_qualified"
    elif noninferior:
        classification = "mean_safe_without_tail_repair"
    else:
        classification = "repair_harms_mean_or_inconclusive"
    return Stage40PanelDecision(
        mean_gain=mean,
        interval90=interval,
        mean_noninferiority=noninferior,
        p95_improvement=bool(p95_improvement),
        contact_improvement=bool(contact_improvement),
        absolute_tail_qualified=bool(absolute_tail_qualified),
        quality_control_passed=bool(quality_control_passed),
        classification=classification,
    )


def derive_stage40_decision(
    panels: Mapping[str, Stage40PanelDecision],
) -> dict[str, Any]:
    if set(panels) != {"jepa", "dino"}:
        raise ValueError("Stage 40 requires separate JEPA and DINO panels")
    classifications = {key: panels[key].classification for key in sorted(panels)}
    if any(value == "invalid_quality_control" for value in classifications.values()):
        status = "invalid_quality_control"
    elif all(value == "contact_tail_repair_confirmed" for value in classifications.values()):
        status = "cross_model_contact_tail_repair_confirmed"
    elif all(value in {
        "contact_tail_repair_confirmed", "tail_improved_but_not_absolutely_qualified",
    } for value in classifications.values()):
        status = "cross_model_tail_improved_but_not_qualified"
    else:
        status = "heterogeneous_or_no_contact_tail_repair"
    return {
        "status": status,
        "passed": status == "cross_model_contact_tail_repair_confirmed",
        "panel_classifications": classifications,
        "panels_pooled": False,
    }
