"""Numerical core for the Stage 38.1 coefficient and hybrid audit.

The module is deliberately independent of PushT, JEPA, DINO, and Colab.  It
contains only the auditable numerical operations needed after the frozen
Stage 38 carrier shards have been loaded: coefficient matching, component
gradient diagnostics, the capacity-matched event/reset intervention, paired
cluster inference, event calibration metrics, and sequential decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
import torch
from torch import nn
import torch.nn.functional as F

from cf_faithfulness.stage36_predictive_state_closure import (
    PredictiveStateClosureModel,
    _mean_scale,
    history_tensor,
    next_history_tensor,
    rollout_evaluation_mask,
    sequence_source_states,
)
from cf_faithfulness.stage37_semigroup_pscd import registered_semigroup_horizons


FloatArray = NDArray[np.float64]


def coefficient_matched_outer_weight(
    full_outer_weight: float,
    full_component_weights: Sequence[float] = (0.35, 0.20, 0.45),
    latent_index: int = 2,
) -> float:
    """Return the latent-only outer weight with identical latent pressure."""

    weights = np.asarray(full_component_weights, dtype=np.float64)
    if weights.ndim != 1 or not len(weights):
        raise ValueError("component weights must be a nonempty vector")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0) or np.sum(weights) <= 0:
        raise ValueError("component weights must be finite, nonnegative, and nonzero")
    index = int(latent_index)
    if not 0 <= index < len(weights):
        raise ValueError("latent index is outside the component vector")
    outer = float(full_outer_weight)
    if not np.isfinite(outer) or outer < 0:
        raise ValueError("outer weight must be finite and nonnegative")
    return outer * float(weights[index] / np.sum(weights))


def macro_contact_events(source_modes: ArrayLike, target_modes: ArrayLike) -> NDArray[np.bool_]:
    """Mark macro-steps whose simulator-derived target mode is contact.

    These labels may supervise construction-only diagnostics.  They are never
    accepted by the label-free rollout path.
    """

    source = np.asarray(source_modes).astype(str)
    target = np.asarray(target_modes).astype(str)
    if source.shape != target.shape or source.ndim != 2:
        raise ValueError("mode paths must be aligned matrices")
    allowed = {"", "free", "pre_contact", "contact", "post_contact"}
    if not set(np.unique(source)).issubset(allowed) or not set(np.unique(target)).issubset(allowed):
        raise ValueError("mode paths contain an unknown label")
    return target == "contact"


def shuffled_event_labels(events: ArrayLike, mask: ArrayLike, *, seed: int) -> NDArray[np.bool_]:
    """Permute valid event labels while preserving prevalence exactly."""

    values = np.asarray(events, dtype=bool)
    valid = np.asarray(mask, dtype=bool)
    if values.shape != valid.shape or values.ndim != 2:
        raise ValueError("event labels and mask must be aligned matrices")
    result = values.copy()
    rng = np.random.default_rng(int(seed))
    result[valid] = rng.permutation(values[valid])
    return result


def _model_from_artifact(artifact: Mapping[str, Any], device: str | torch.device):
    config = artifact["config"]
    kind = str(config.get("transition_kind", "base"))
    if kind == "base":
        model: nn.Module = PredictiveStateClosureModel(
            config["carrier_dim"], config["history_length"], config["action_dim"],
            config["physical_dim"], config["latent_dim"], config["dynamics"],
        )
    else:
        model = EventFactorizedPredictiveStateClosureModel(
            config["carrier_dim"], config["history_length"], config["action_dim"],
            config["physical_dim"], config["latent_dim"], config["dynamics"],
            transition_kind=kind, event_hidden=int(config["event_hidden"]),
        )
    model = model.to(device)
    model.load_state_dict({
        key: torch.as_tensor(value, device=device)
        for key, value in artifact["state_dict"].items()
    }, strict=True)
    model.eval()
    return model


def _gradient_norm(loss: torch.Tensor, model: nn.Module) -> float:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True,
    )
    squared = torch.zeros((), device=loss.device)
    for gradient in gradients:
        if gradient is not None:
            squared = squared + torch.sum(gradient.detach() ** 2)
    return float(torch.sqrt(squared).cpu())


def semigroup_component_diagnostics(
    artifact: Mapping[str, Any],
    initial: ArrayLike,
    actions: ArrayLike,
    carrier_targets: ArrayLike,
    physical_targets: ArrayLike,
    mask: ArrayLike,
    *,
    device: str | None = None,
) -> dict[str, Any]:
    """Measure unweighted component losses and their gradient norms.

    The calculation uses the frozen trained artifact and the same registered
    direct-versus-composed pairs as training.  It does not update parameters.
    """

    config = artifact["config"]
    first = np.asarray(initial, dtype=np.float32)
    action = np.asarray(actions, dtype=np.float32)
    carrier = np.asarray(carrier_targets, dtype=np.float32)
    physical = np.asarray(physical_targets, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool)
    if (
        action.shape[:2] != valid.shape or carrier.shape[:2] != valid.shape
        or physical.shape[:2] != valid.shape
    ):
        raise ValueError("diagnostic arrays are not aligned")
    history = history_tensor(first, carrier, valid, config["history_length"]).astype(np.float32)
    next_history = next_history_tensor(history, carrier).astype(np.float32)
    normalization = artifact["normalization"]
    carrier_mean = np.asarray(normalization["carrier_mean"], dtype=np.float32)
    carrier_scale = np.asarray(normalization["carrier_scale"], dtype=np.float32)
    action_mean = np.asarray(normalization["action_mean"], dtype=np.float32)
    action_scale = np.asarray(normalization["action_scale"], dtype=np.float32)
    physical_mean = np.asarray(normalization["physical_mean"], dtype=np.float32)
    physical_scale = np.asarray(normalization["physical_scale"], dtype=np.float32)
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = _model_from_artifact(artifact, selected_device)
    model.train()
    history_t = torch.as_tensor((history - carrier_mean) / carrier_scale, device=selected_device)
    next_t = torch.as_tensor((next_history - carrier_mean) / carrier_scale, device=selected_device)
    action_t = torch.as_tensor((action - action_mean) / action_scale, device=selected_device)
    carrier_t = torch.as_tensor((carrier - carrier_mean) / carrier_scale, device=selected_device)
    physical_t = torch.as_tensor(
        (physical - physical_mean) / physical_scale, device=selected_device
    )
    valid_t = torch.as_tensor(valid, device=selected_device)
    source_state = model.encode(history_t)
    with torch.no_grad():
        target_state = model.encode(next_t)
    pairs = registered_semigroup_horizons(
        valid, int(config["history_length"]), config.get("semigroup_horizons", (2, 4, 8))
    )
    terms: dict[str, list[torch.Tensor]] = {"carrier": [], "physical": [], "state": []}
    by_anchor: dict[int, list[int]] = {}
    for anchor, horizon in pairs:
        by_anchor.setdefault(anchor, []).append(horizon)
    for anchor, horizons in by_anchor.items():
        composed = source_state[:, anchor]
        for offset in range(max(horizons)):
            step = anchor + offset
            active = valid_t[:, anchor : step + 1].all(dim=1)
            if not bool(torch.any(active)):
                continue
            updated = model.transition(composed[active], action_t[active, step])
            composed = composed.clone()
            composed[active] = updated
            horizon = offset + 1
            if horizon not in horizons:
                continue
            eligible_np = np.all(valid[:, anchor : anchor + horizon], axis=1)
            eligible = torch.as_tensor(eligible_np, device=selected_device)
            endpoint = anchor + horizon - 1
            predicted = composed[eligible]
            direct = target_state[eligible, endpoint]
            decoded_carrier, decoded_physical = model.decode(predicted)
            terms["state"].append(torch.mean((predicted - direct) ** 2))
            terms["carrier"].append(torch.mean(
                (decoded_carrier - carrier_t[eligible, endpoint]) ** 2
            ))
            terms["physical"].append(torch.mean(
                (decoded_physical - physical_t[eligible, endpoint]) ** 2
            ))
    zero = torch.zeros((), device=selected_device, requires_grad=True)
    losses = {
        key: torch.stack(values).mean() if values else zero
        for key, values in terms.items()
    }
    weights = np.asarray(config.get("semigroup_component_weights", [0.0, 0.0, 0.0]))
    outer = float(config.get("semigroup_weight", 0.0))
    coefficients = outer * weights / max(float(np.sum(weights)), 1e-12)
    return {
        "raw_losses": {key: float(value.detach().cpu()) for key, value in losses.items()},
        "raw_gradient_norms": {key: _gradient_norm(value, model) for key, value in losses.items()},
        "normalized_component_weights": weights.astype(float).tolist(),
        "effective_coefficients": coefficients.astype(float).tolist(),
        "outer_weight": outer,
        "registered_anchor_pairs": int(len(pairs)),
        "valid_transitions": int(np.sum(valid)),
    }


class EventFactorizedPredictiveStateClosureModel(PredictiveStateClosureModel):
    """Base predictive state plus a small capacity-matched jump branch."""

    def __init__(
        self,
        carrier_dim: int,
        history_length: int,
        action_dim: int,
        physical_dim: int,
        latent_dim: int,
        dynamics: str,
        *,
        transition_kind: str,
        event_hidden: int = 32,
    ) -> None:
        super().__init__(
            carrier_dim, history_length, action_dim, physical_dim, latent_dim, dynamics
        )
        if transition_kind not in {"hybrid", "smooth"}:
            raise ValueError("transition_kind must be 'hybrid' or 'smooth'")
        self.transition_kind = str(transition_kind)
        context_dim = self.latent_dim + self.action_dim
        self.event_head = nn.Sequential(
            nn.Linear(context_dim, int(event_hidden)), nn.SiLU(), nn.Linear(int(event_hidden), 1)
        )
        self.jump = nn.Sequential(
            nn.Linear(context_dim, int(event_hidden)), nn.SiLU(),
            nn.Linear(int(event_hidden), self.latent_dim),
        )
        nn.init.zeros_(self.jump[-1].weight)
        nn.init.zeros_(self.jump[-1].bias)

    def transition_with_event(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        oracle_gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        smooth = super().transition(state, action)
        context = torch.cat([state, action], dim=-1)
        logits = self.event_head(context).squeeze(-1)
        if self.transition_kind == "hybrid":
            gate = torch.sigmoid(logits) if oracle_gate is None else oracle_gate.to(state.dtype)
        else:
            gate = torch.ones_like(logits)
        return smooth + gate[..., None] * self.jump(context), logits

    def transition(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.transition_with_event(state, action)[0]


def _validate_training_arrays(
    initial: ArrayLike,
    actions: ArrayLike,
    carrier_targets: ArrayLike,
    physical_targets: ArrayLike,
    mask: ArrayLike,
    events: ArrayLike,
    groups: ArrayLike | None,
) -> tuple[np.ndarray, ...]:
    first = np.asarray(initial, dtype=np.float32)
    action = np.asarray(actions, dtype=np.float32)
    carrier = np.asarray(carrier_targets, dtype=np.float32)
    physical = np.asarray(physical_targets, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool)
    event = np.asarray(events, dtype=bool)
    if first.ndim != 2 or action.ndim != 3 or carrier.ndim != 3 or physical.ndim != 3:
        raise ValueError("Stage 38.1 training arrays have invalid ranks")
    if action.shape[:2] != valid.shape or carrier.shape[:2] != valid.shape:
        raise ValueError("Stage 38.1 sequence arrays are not aligned")
    if physical.shape[:2] != valid.shape or event.shape != valid.shape or len(first) != len(action):
        raise ValueError("Stage 38.1 targets are not aligned")
    group = np.arange(len(first), dtype=np.int64) if groups is None else np.asarray(groups)
    if group.ndim != 1 or len(group) != len(first):
        raise ValueError("family groups are not aligned")
    if not np.any(event[valid]) or np.all(event[valid]):
        raise ValueError("event supervision requires both classes")
    return first, action, carrier, physical, valid, event, group


def fit_event_factorized_pscd(
    initial: ArrayLike,
    actions: ArrayLike,
    carrier_targets: ArrayLike,
    physical_targets: ArrayLike,
    mask: ArrayLike,
    events: ArrayLike,
    *,
    groups: ArrayLike | None = None,
    history_length: int,
    latent_dim: int,
    dynamics: str,
    epochs: int,
    learning_rate: float,
    seed: int,
    semigroup_horizons: Sequence[int] = (2, 4, 8),
    semigroup_weight: float = 1.0,
    semigroup_component_weights: Sequence[float] = (0.35, 0.20, 0.45),
    event_weight: float = 0.10,
    transition_kind: str = "hybrid",
    event_hidden: int = 32,
    risk_weight: float = 0.0,
    risk_alpha: float = 0.90,
    device: str | None = None,
) -> dict[str, Any]:
    """Fit the no-tail hybrid or its parameter-matched smooth control."""

    first, action, carrier, physical, valid, event, group = _validate_training_arrays(
        initial, actions, carrier_targets, physical_targets, mask, events, groups
    )
    if transition_kind not in {"hybrid", "smooth"}:
        raise ValueError("unknown transition kind")
    weights = np.asarray(semigroup_component_weights, dtype=np.float64)
    if weights.shape != (3,) or np.any(weights < 0) or np.sum(weights) <= 0:
        raise ValueError("semigroup weights must be three nonnegative nonzero values")
    if not 0 < float(risk_alpha) < 1 or min(event_weight, risk_weight) < 0:
        raise ValueError("event/risk settings are invalid")
    history = history_tensor(first, carrier, valid, history_length).astype(np.float32)
    next_history = next_history_tensor(history, carrier).astype(np.float32)
    source = sequence_source_states(first, carrier).astype(np.float32)
    carrier_mean, carrier_scale = _mean_scale(np.concatenate([source[valid], carrier[valid]]))
    action_mean, action_scale = _mean_scale(action[valid])
    physical_mean, physical_scale = _mean_scale(physical[valid])
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model = EventFactorizedPredictiveStateClosureModel(
        first.shape[1], int(history_length), action.shape[2], physical.shape[2],
        int(latent_dim), str(dynamics), transition_kind=transition_kind,
        event_hidden=int(event_hidden),
    ).to(selected_device)
    tau = nn.Parameter(torch.zeros((), device=selected_device))
    parameters = list(model.parameters()) + ([tau] if float(risk_weight) > 0 else [])
    optimizer = torch.optim.AdamW(parameters, lr=float(learning_rate), weight_decay=1e-4)
    tensors = {
        "history": torch.as_tensor((history - carrier_mean) / carrier_scale, device=selected_device),
        "next_history": torch.as_tensor((next_history - carrier_mean) / carrier_scale, device=selected_device),
        "action": torch.as_tensor((action - action_mean) / action_scale, device=selected_device),
        "carrier": torch.as_tensor((carrier - carrier_mean) / carrier_scale, device=selected_device),
        "physical": torch.as_tensor((physical - physical_mean) / physical_scale, device=selected_device),
        "mask": torch.as_tensor(valid, device=selected_device),
        "event": torch.as_tensor(event, device=selected_device),
    }
    pairs = registered_semigroup_horizons(valid, int(history_length), semigroup_horizons)
    pair_masks = {
        (anchor, horizon): torch.as_tensor(
            np.all(valid[:, anchor : anchor + horizon], axis=1), device=selected_device
        )
        for anchor, horizon in pairs
    }
    unique_groups = np.unique(group)
    group_rows = {
        value: torch.as_tensor(group == value, device=selected_device)
        for value in unique_groups
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
        one_state, logits = model.transition_with_event(source_state, tensors["action"])
        one_carrier, one_physical = model.decode(one_state)
        observed = tensors["mask"]
        one = {
            "carrier": torch.mean((one_carrier[observed] - tensors["carrier"][observed]) ** 2),
            "physical": torch.mean((one_physical[observed] - tensors["physical"][observed]) ** 2),
            "state": torch.mean((one_state[observed] - target_state[observed]) ** 2),
        }
        labels = tensors["event"][observed].to(torch.float32)
        positive = torch.clamp(torch.sum(labels), min=1.0)
        negative = torch.clamp(torch.sum(1.0 - labels), min=1.0)
        sample_weights = 0.5 * labels / positive + 0.5 * (1.0 - labels) / negative
        event_loss = torch.sum(
            sample_weights * F.binary_cross_entropy_with_logits(logits[observed], labels, reduction="none")
        )

        recursive_terms: dict[str, list[torch.Tensor]] = {
            "carrier": [], "physical": [], "state": []
        }
        sequence_physical = torch.zeros(len(first), device=selected_device)
        sequence_counts = torch.zeros(len(first), device=selected_device)
        state = source_state[:, start]
        for step in range(start, action.shape[1]):
            step_valid = observed[:, step]
            if not bool(torch.any(step_valid)):
                continue
            updated = model.transition(state[step_valid], tensors["action"][step_valid, step])
            state = state.clone()
            state[step_valid] = updated
            decoded_carrier, decoded_physical = model.decode(updated)
            physical_squared = torch.mean(
                (decoded_physical - tensors["physical"][step_valid, step]) ** 2, dim=-1
            )
            recursive_terms["carrier"].append(torch.mean(
                (decoded_carrier - tensors["carrier"][step_valid, step]) ** 2
            ))
            recursive_terms["physical"].append(torch.mean(physical_squared))
            recursive_terms["state"].append(torch.mean(
                (updated - target_state[step_valid, step]) ** 2
            ))
            sequence_physical[step_valid] = sequence_physical[step_valid] + physical_squared
            sequence_counts[step_valid] = sequence_counts[step_valid] + 1.0
        zero = torch.zeros((), device=selected_device)
        recursive = {
            key: torch.stack(values).mean() if values else zero
            for key, values in recursive_terms.items()
        }

        semigroup_terms: dict[str, list[torch.Tensor]] = {
            "carrier": [], "physical": [], "state": []
        }
        by_anchor: dict[int, list[int]] = {}
        for anchor, horizon in pairs:
            by_anchor.setdefault(anchor, []).append(horizon)
        for anchor, horizons in by_anchor.items():
            composed = source_state[:, anchor]
            for offset in range(max(horizons)):
                step = anchor + offset
                active_path = observed[:, anchor : step + 1].all(dim=1)
                if not bool(torch.any(active_path)):
                    continue
                update = model.transition(composed[active_path], tensors["action"][active_path, step])
                composed = composed.clone()
                composed[active_path] = update
                horizon = offset + 1
                if horizon not in horizons:
                    continue
                eligible = pair_masks[(anchor, horizon)]
                endpoint = anchor + horizon - 1
                predicted = composed[eligible]
                decoded_carrier, decoded_physical = model.decode(predicted)
                semigroup_terms["state"].append(torch.mean(
                    (predicted - target_state[eligible, endpoint]) ** 2
                ))
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
        one_loss = 0.45 * one["carrier"] + 0.25 * one["physical"] + 0.25 * one["state"]
        free_loss = (
            0.45 * recursive["carrier"] + 0.25 * recursive["physical"]
            + 0.20 * recursive["state"]
        )
        semigroup_loss = sum(
            float(weights[index]) * semigroup[key]
            for index, key in enumerate(["carrier", "physical", "state"])
        ) / float(np.sum(weights))
        family_losses = torch.stack([
            torch.mean(sequence_physical[rows] / torch.clamp(sequence_counts[rows], min=1.0))
            for rows in group_rows.values()
        ])
        risk_loss = tau + torch.mean(torch.relu(family_losses - tau)) / (1.0 - float(risk_alpha))
        total = (
            one_loss + free_loss + float(semigroup_weight) * semigroup_loss
            + float(event_weight) * event_loss + float(risk_weight) * risk_loss
        )
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("Stage 38.1 hybrid training became nonfinite")
        total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        optimizer.step()
        losses.append(float(total.detach().cpu()))
        final_components = {
            "one_carrier": float(one["carrier"].detach().cpu()),
            "one_physical": float(one["physical"].detach().cpu()),
            "one_state": float(one["state"].detach().cpu()),
            "free_carrier": float(recursive["carrier"].detach().cpu()),
            "free_physical": float(recursive["physical"].detach().cpu()),
            "free_state": float(recursive["state"].detach().cpu()),
            "semigroup_carrier": float(semigroup["carrier"].detach().cpu()),
            "semigroup_physical": float(semigroup["physical"].detach().cpu()),
            "semigroup_state": float(semigroup["state"].detach().cpu()),
            "event": float(event_loss.detach().cpu()),
            "family_risk": float(risk_loss.detach().cpu()),
            "total_gradient_norm": float(gradient_norm.detach().cpu()),
        }
    return {
        "config": {
            "carrier_dim": int(first.shape[1]), "history_length": int(history_length),
            "action_dim": int(action.shape[2]), "physical_dim": int(physical.shape[2]),
            "latent_dim": int(latent_dim), "dynamics": str(dynamics),
            "transition_kind": str(transition_kind), "event_hidden": int(event_hidden),
            "epochs": int(epochs), "learning_rate": float(learning_rate), "seed": int(seed),
            "free_weight": 1.0, "semigroup_horizons": list(map(int, semigroup_horizons)),
            "semigroup_weight": float(semigroup_weight),
            "semigroup_component_weights": weights.tolist(),
            "event_weight": float(event_weight), "risk_weight": float(risk_weight),
            "risk_alpha": float(risk_alpha), "family_count": int(len(unique_groups)),
            "trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        },
        "normalization": {
            "carrier_mean": carrier_mean, "carrier_scale": carrier_scale,
            "action_mean": action_mean, "action_scale": action_scale,
            "physical_mean": physical_mean, "physical_scale": physical_scale,
        },
        "state_dict": {
            key: value.detach().cpu().numpy().astype(np.float32)
            for key, value in model.state_dict().items()
        },
        "loss_initial": float(losses[0]), "loss_final": float(losses[-1]),
        "loss_components_final": final_components,
        "risk_tau_final": float(tau.detach().cpu()),
    }


def rollout_event_factorized_pscd(
    artifact: Mapping[str, Any],
    initial: ArrayLike,
    actions: ArrayLike,
    native_carrier_path: ArrayLike,
    mask: ArrayLike,
    *,
    oracle_events: ArrayLike | None = None,
    histories_override: ArrayLike | None = None,
    device: str | None = None,
) -> dict[str, FloatArray]:
    """Warm up on native history, then recurse with predicted or oracle gates."""

    config, normalization = artifact["config"], artifact["normalization"]
    first = np.asarray(initial, dtype=np.float32)
    action = np.asarray(actions, dtype=np.float32)
    carrier = np.asarray(native_carrier_path, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool)
    history = history_tensor(first, carrier, valid, config["history_length"]).astype(np.float32)
    if histories_override is not None:
        override = np.asarray(histories_override, dtype=np.float32)
        if override.shape != history.shape:
            raise ValueError("history override has the wrong shape")
        history = override
    next_history = next_history_tensor(history, carrier).astype(np.float32)
    oracle = None if oracle_events is None else np.asarray(oracle_events, dtype=bool)
    if oracle is not None and oracle.shape != valid.shape:
        raise ValueError("oracle event path does not align with rollout")
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = _model_from_artifact(artifact, selected_device)
    cm = np.asarray(normalization["carrier_mean"], dtype=np.float32)
    cs = np.asarray(normalization["carrier_scale"], dtype=np.float32)
    am = np.asarray(normalization["action_mean"], dtype=np.float32)
    ass = np.asarray(normalization["action_scale"], dtype=np.float32)
    pm = np.asarray(normalization["physical_mean"], dtype=np.float32)
    ps = np.asarray(normalization["physical_scale"], dtype=np.float32)
    history_t = torch.as_tensor((history - cm) / cs, device=selected_device)
    next_t = torch.as_tensor((next_history - cm) / cs, device=selected_device)
    action_t = torch.as_tensor((action - am) / ass, device=selected_device)
    output_carrier = carrier.copy()
    output_physical = np.zeros((*valid.shape, int(config["physical_dim"])), dtype=np.float32)
    output_state = np.zeros((*valid.shape, int(config["latent_dim"])), dtype=np.float32)
    direct_state = np.zeros_like(output_state)
    event_probability = np.full(valid.shape, np.nan, dtype=np.float32)
    start = int(config["history_length"]) - 1
    with torch.inference_mode():
        teacher = model.encode(history_t)
        direct_state[:] = model.encode(next_t).cpu().numpy()
        state = teacher[:, start]
        for step in range(start, action.shape[1]):
            active = valid[:, step]
            if not np.any(active):
                continue
            active_t = torch.as_tensor(active, device=selected_device)
            gate = None
            if oracle is not None:
                gate = torch.as_tensor(oracle[active, step], device=selected_device)
            updated, logits = model.transition_with_event(
                state[active_t], action_t[active_t, step], oracle_gate=gate
            )
            state = state.clone()
            state[active_t] = updated
            decoded_carrier, decoded_physical = model.decode(updated)
            output_carrier[active, step] = decoded_carrier.cpu().numpy() * cs + cm
            output_physical[active, step] = decoded_physical.cpu().numpy() * ps + pm
            output_state[active, step] = updated.cpu().numpy()
            event_probability[active, step] = torch.sigmoid(logits).cpu().numpy()
    return {
        "carrier": output_carrier.astype(np.float64),
        "physical": output_physical.astype(np.float64),
        "state": output_state.astype(np.float64),
        "direct_state": direct_state.astype(np.float64),
        "event_probability": event_probability.astype(np.float64),
        "evaluation_mask": rollout_evaluation_mask(valid, config["history_length"]),
    }


def hierarchical_relative_gain_interval(
    primary: ArrayLike,
    comparator: ArrayLike,
    groups: ArrayLike,
    *,
    draws: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Paired seed/family bootstrap for a relative gain of means."""

    first = np.asarray(primary, dtype=np.float64)
    second = np.asarray(comparator, dtype=np.float64)
    labels = np.asarray(groups)
    if first.shape != second.shape or first.ndim != 2 or first.shape[1] != len(labels):
        raise ValueError("paired bootstrap arrays are not aligned")
    unique = np.unique(labels)
    if first.shape[0] < 2 or len(unique) < 2 or int(draws) < 2:
        raise ValueError("paired bootstrap needs multiple seeds, groups, and draws")
    rng = np.random.default_rng(int(seed))
    rows = {value: np.flatnonzero(labels == value) for value in unique}
    estimates = np.empty(int(draws), dtype=np.float64)
    for draw in range(int(draws)):
        seeds = rng.integers(0, first.shape[0], size=first.shape[0])
        sampled = rng.choice(unique, size=len(unique), replace=True)
        first_draw = np.concatenate([first[index, rows[value]] for index in seeds for value in sampled])
        second_draw = np.concatenate([second[index, rows[value]] for index in seeds for value in sampled])
        estimates[draw] = (np.mean(second_draw) - np.mean(first_draw)) / max(np.mean(second_draw), 1e-12)
    tail = (1.0 - float(confidence)) / 2.0
    low, high = np.quantile(estimates, [tail, 1.0 - tail])
    return float(low), float(high)


def hierarchical_statistic_interval(
    values: ArrayLike,
    groups: ArrayLike,
    *,
    statistic: str,
    draws: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Seed/family bootstrap interval for an absolute tail or mean statistic."""

    matrix = np.asarray(values, dtype=np.float64)
    labels = np.asarray(groups)
    if matrix.ndim != 2 or matrix.shape[1] != len(labels):
        raise ValueError("hierarchical statistic arrays are not aligned")
    unique = np.unique(labels)
    if matrix.shape[0] < 2 or len(unique) < 2 or int(draws) < 2:
        raise ValueError("hierarchical statistic needs multiple seeds, groups, and draws")
    if statistic not in {"mean", "p95", "cvar95", "catastrophic_rate_gt_1"}:
        raise ValueError("unknown hierarchical statistic")

    def evaluate(sample: np.ndarray) -> float:
        if statistic == "mean":
            return float(np.mean(sample))
        if statistic == "p95":
            return float(np.quantile(sample, 0.95))
        if statistic == "cvar95":
            cutoff = float(np.quantile(sample, 0.95))
            return float(np.mean(sample[sample >= cutoff]))
        return float(np.mean(sample > 1.0))

    rng = np.random.default_rng(int(seed))
    rows = {value: np.flatnonzero(labels == value) for value in unique}
    estimates = np.empty(int(draws), dtype=np.float64)
    for draw in range(int(draws)):
        seeds = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        sampled = rng.choice(unique, size=len(unique), replace=True)
        sample = np.concatenate([
            matrix[index, rows[value]] for index in seeds for value in sampled
        ])
        estimates[draw] = evaluate(sample)
    tail = (1.0 - float(confidence)) / 2.0
    low, high = np.quantile(estimates, [tail, 1.0 - tail])
    return float(low), float(high)


def leave_one_family_out_relative_gain(
    primary: ArrayLike, comparator: ArrayLike, groups: ArrayLike
) -> dict[str, float]:
    first = np.asarray(primary, dtype=np.float64)
    second = np.asarray(comparator, dtype=np.float64)
    labels = np.asarray(groups)
    if first.shape != second.shape or first.ndim != 2 or first.shape[1] != len(labels):
        raise ValueError("leave-one-family-out arrays are not aligned")
    output = {}
    for value in np.unique(labels):
        keep = labels != value
        output[str(value)] = float(
            (np.mean(second[:, keep]) - np.mean(first[:, keep]))
            / max(np.mean(second[:, keep]), 1e-12)
        )
    return output


def event_classification_metrics(
    probability: ArrayLike, labels: ArrayLike, mask: ArrayLike, *, bins: int = 10
) -> dict[str, float]:
    """Return tie-correct AUROC, Brier skill, and fixed-bin ECE."""

    score = np.asarray(probability, dtype=np.float64)
    truth = np.asarray(labels, dtype=bool)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(score)
    if score.shape != truth.shape or score.shape != valid.shape:
        raise ValueError("event metric arrays are not aligned")
    y, p = truth[valid].astype(np.float64), np.clip(score[valid], 0.0, 1.0)
    positives, negatives = int(np.sum(y)), int(np.sum(1.0 - y))
    if not positives or not negatives:
        raise ValueError("event metrics require both classes")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    cursor = 0
    while cursor < len(p):
        end = cursor + 1
        while end < len(p) and p[order[end]] == p[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = 0.5 * (cursor + 1 + end)
        cursor = end
    auc = (np.sum(ranks[y == 1]) - positives * (positives + 1) / 2) / (positives * negatives)
    brier = float(np.mean((p - y) ** 2))
    base = float(np.mean((np.mean(y) - y) ** 2))
    ece = 0.0
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    for index in range(int(bins)):
        selected = (p >= edges[index]) & (p < edges[index + 1] if index + 1 < bins else p <= 1.0)
        if np.any(selected):
            ece += float(np.mean(selected)) * abs(float(np.mean(p[selected]) - np.mean(y[selected])))
    return {
        "auroc": float(auc), "brier": brier,
        "brier_skill": float(1.0 - brier / max(base, 1e-12)),
        "ece": float(ece), "prevalence": float(np.mean(y)), "observations": int(len(y)),
    }


@dataclass(frozen=True)
class TierAGates:
    coefficient_specificity: bool
    tail_noninferiority: bool
    correct_history_specificity: bool
    absolute_viability: bool
    three_seed_stability: bool


@dataclass(frozen=True)
class TierBGates:
    oracle_headroom: bool
    event_tail_repair: bool
    overall_tail_repair: bool
    catastrophic_control: bool
    event_identifiability: bool
    shuffled_supervision_specificity: bool
    label_free_inference: bool
    seed_and_model_stability: bool


def sequential_decision(gates: TierAGates | TierBGates, *, passed_status: str) -> dict[str, Any]:
    checks = {name: bool(getattr(gates, name)) for name in gates.__dataclass_fields__}
    failed = next((name for name, value in checks.items() if not value), None)
    return {
        "passed": failed is None,
        "status": str(passed_status) if failed is None else f"failed_{failed}",
        "first_failed_gate": failed,
        "gates": checks,
    }
