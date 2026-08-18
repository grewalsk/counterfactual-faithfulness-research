"""Numerical core for Stage 36 predictive-state closure distillation.

The module contains the finite-history construction, a compact recurrent
predictive-state adapter, deterministic training and rollout helpers, and the
strict decision semantics used by the generated Colab notebook.  It does not
load JEPA-WM, PushT, or any external checkpoint, so its core behavior can be
tested locally on synthetic systems.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
import torch
from torch import nn


FloatArray = NDArray[np.float64]


def stable_seed(root: int, *parts: object) -> int:
    payload = ":".join([str(int(root)), *map(str, parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def sequence_source_states(initial: ArrayLike, targets: ArrayLike) -> FloatArray:
    first = np.asarray(initial, dtype=np.float64)
    path = np.asarray(targets, dtype=np.float64)
    if first.ndim != 2 or path.ndim != 3:
        raise ValueError("initial and targets must be a matrix and sequence tensor")
    if path.shape[0] != len(first) or path.shape[2] != first.shape[1]:
        raise ValueError("initial and targets are not aligned")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(path)):
        raise ValueError("state arrays contain nonfinite values")
    source = np.empty_like(path)
    source[:, 0] = first
    source[:, 1:] = path[:, :-1]
    return source


def history_tensor(
    initial: ArrayLike,
    targets: ArrayLike,
    mask: ArrayLike,
    history_length: int,
) -> FloatArray:
    """Return left-padded native histories ending at each transition source."""

    source = sequence_source_states(initial, targets)
    valid = np.asarray(mask, dtype=bool)
    history = int(history_length)
    if history < 1:
        raise ValueError("history_length must be positive")
    if valid.shape != source.shape[:2]:
        raise ValueError("mask does not align with state paths")
    output = np.empty((*source.shape[:2], history, source.shape[2]), dtype=np.float64)
    for step in range(source.shape[1]):
        for slot in range(history):
            source_step = max(0, step - history + slot + 1)
            output[:, step, slot] = source[:, source_step]
    return output


def next_history_tensor(histories: ArrayLike, targets: ArrayLike) -> FloatArray:
    history = np.asarray(histories, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if history.ndim != 4 or target.ndim != 3:
        raise ValueError("history and target tensors have invalid ranks")
    if history.shape[:2] != target.shape[:2] or history.shape[3] != target.shape[2]:
        raise ValueError("history and targets are not aligned")
    return np.concatenate([history[:, :, 1:], target[:, :, None, :]], axis=2)


def rollout_evaluation_mask(mask: ArrayLike, history_length: int) -> NDArray[np.bool_]:
    valid = np.asarray(mask, dtype=bool)
    if valid.ndim != 2:
        raise ValueError("mask must be a sequence matrix")
    history = int(history_length)
    if history < 1:
        raise ValueError("history_length must be positive")
    result = valid.copy()
    result[:, : history - 1] = False
    if np.any(np.sum(result, axis=1) < 1):
        raise ValueError("every sequence must extend beyond the history warmup")
    return result


def permute_past_history(
    histories: ArrayLike,
    groups: ArrayLike,
    mask: ArrayLike,
    *,
    seed: int,
) -> FloatArray:
    """Destroy past-slot identity while preserving the current carrier slot."""

    history = np.asarray(histories, dtype=np.float64)
    group = np.asarray(groups)
    valid = np.asarray(mask, dtype=bool)
    if history.ndim != 4 or valid.shape != history.shape[:2] or len(group) != len(history):
        raise ValueError("history-control inputs are not aligned")
    result = history.copy()
    if history.shape[2] == 1:
        return result
    rng = np.random.default_rng(int(seed))
    for value in np.unique(group):
        rows, steps = np.where(valid & (group[:, None] == value))
        if len(rows) < 2:
            continue
        order = rng.permutation(len(rows))
        for slot in range(history.shape[2] - 1):
            result[rows, steps, slot] = history[rows[order], steps[order], slot]
    return result


def _mean_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0)
    scale = np.maximum(np.std(values, axis=0, ddof=1), 1e-6)
    return mean.astype(np.float32), scale.astype(np.float32)


class PredictiveStateClosureModel(nn.Module):
    """Finite-history encoder with an action-conditioned recurrent state."""

    def __init__(
        self,
        carrier_dim: int,
        history_length: int,
        action_dim: int,
        physical_dim: int,
        latent_dim: int,
        dynamics: str = "single",
    ) -> None:
        super().__init__()
        self.carrier_dim = int(carrier_dim)
        self.history_length = int(history_length)
        self.action_dim = int(action_dim)
        self.physical_dim = int(physical_dim)
        self.latent_dim = int(latent_dim)
        self.dynamics = str(dynamics)
        if self.dynamics not in {"single", "mixture"}:
            raise ValueError("dynamics must be 'single' or 'mixture'")
        hidden = max(64, 2 * self.latent_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.carrier_dim * self.history_length, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_dim, self.latent_dim), nn.Tanh()
        )
        experts = 1 if self.dynamics == "single" else 3
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * self.latent_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.latent_dim),
            )
            for _ in range(experts)
        ])
        self.gate = None if experts == 1 else nn.Linear(2 * self.latent_dim, experts)
        self.transition_norm = nn.LayerNorm(self.latent_dim)
        self.carrier_decoder = nn.Linear(self.latent_dim, self.carrier_dim)
        self.physical_decoder = nn.Linear(self.latent_dim, self.physical_dim)

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        return self.encoder(history.flatten(start_dim=-2))

    def transition(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        encoded_action = self.action_encoder(action)
        features = torch.cat([state, encoded_action], dim=-1)
        updates = torch.stack([expert(features) for expert in self.experts], dim=-2)
        if self.gate is None:
            update = updates[..., 0, :]
        else:
            probability = torch.softmax(self.gate(features), dim=-1)
            update = torch.sum(updates * probability[..., :, None], dim=-2)
        return self.transition_norm(state + 0.25 * update)

    def decode(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.carrier_decoder(state), self.physical_decoder(state)


def _artifact_model(artifact: Mapping[str, Any], device: str | torch.device):
    config = artifact["config"]
    model = PredictiveStateClosureModel(
        config["carrier_dim"], config["history_length"], config["action_dim"],
        config["physical_dim"], config["latent_dim"], config["dynamics"],
    ).to(device)
    state = {
        key: torch.as_tensor(value, device=device)
        for key, value in artifact["state_dict"].items()
    }
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def fit_predictive_state_closure(
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
    device: str | None = None,
    free_weight: float = 1.0,
    consistency_weight: float = 0.25,
    histories_override: ArrayLike | None = None,
) -> dict[str, Any]:
    """Fit PSCD without ever updating the frozen world model."""

    first = np.asarray(initial, dtype=np.float32)
    action = np.asarray(actions, dtype=np.float32)
    carrier = np.asarray(carrier_targets, dtype=np.float32)
    physical = np.asarray(physical_targets, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool)
    if first.ndim != 2 or action.ndim != 3 or carrier.ndim != 3 or physical.ndim != 3:
        raise ValueError("PSCD arrays have invalid ranks")
    if action.shape[:2] != valid.shape or carrier.shape[:2] != valid.shape:
        raise ValueError("PSCD sequence arrays are not aligned")
    if physical.shape[:2] != valid.shape or len(first) != len(action):
        raise ValueError("PSCD target arrays are not aligned")
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
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model = PredictiveStateClosureModel(
        first.shape[1], int(history_length), action.shape[2], physical.shape[2],
        int(latent_dim), str(dynamics),
    ).to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    tensors = {
        "history": torch.as_tensor(history_n, device=selected_device),
        "next_history": torch.as_tensor(next_history_n, device=selected_device),
        "action": torch.as_tensor(action_n, device=selected_device),
        "carrier": torch.as_tensor(carrier_n, device=selected_device),
        "physical": torch.as_tensor(physical_n, device=selected_device),
        "mask": torch.as_tensor(valid, device=selected_device),
    }
    start = int(history_length) - 1
    losses: list[float] = []
    model.train()
    for _epoch in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        source_state = model.encode(tensors["history"])
        with torch.no_grad():
            target_state = model.encode(tensors["next_history"])
        one_state = model.transition(source_state, tensors["action"])
        one_carrier, one_physical = model.decode(one_state)
        observed = tensors["mask"]
        carrier_loss = torch.mean((one_carrier[observed] - tensors["carrier"][observed]) ** 2)
        physical_loss = torch.mean((one_physical[observed] - tensors["physical"][observed]) ** 2)
        consistency = torch.mean((one_state[observed] - target_state[observed]) ** 2)
        recursive_carrier = torch.zeros((), device=selected_device)
        recursive_physical = torch.zeros((), device=selected_device)
        recursive_state = torch.zeros((), device=selected_device)
        count = torch.zeros((), device=selected_device)
        state = source_state[:, start]
        for step in range(start, action.shape[1]):
            step_valid = observed[:, step]
            if not bool(torch.any(step_valid)):
                continue
            updated = model.transition(state[step_valid], tensors["action"][step_valid, step])
            state = state.clone()
            state[step_valid] = updated
            decoded_carrier, decoded_physical = model.decode(updated)
            recursive_carrier = recursive_carrier + torch.sum(
                (decoded_carrier - tensors["carrier"][step_valid, step]) ** 2
            )
            recursive_physical = recursive_physical + torch.sum(
                (decoded_physical - tensors["physical"][step_valid, step]) ** 2
            )
            recursive_state = recursive_state + torch.sum(
                (updated - target_state[step_valid, step]) ** 2
            )
            count = count + torch.sum(step_valid)
        recursive_carrier = recursive_carrier / torch.clamp(count * carrier.shape[2], min=1)
        recursive_physical = recursive_physical / torch.clamp(count * physical.shape[2], min=1)
        recursive_state = recursive_state / torch.clamp(count * int(latent_dim), min=1)
        loss = (
            0.45 * carrier_loss + 0.25 * physical_loss
            + float(consistency_weight) * consistency
            + float(free_weight) * (
                0.45 * recursive_carrier + 0.25 * recursive_physical
                + 0.20 * recursive_state
            )
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("PSCD training became nonfinite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
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
    }


def rollout_predictive_state_closure(
    artifact: Mapping[str, Any],
    initial: ArrayLike,
    actions: ArrayLike,
    native_carrier_path: ArrayLike,
    mask: ArrayLike,
    *,
    device: str | None = None,
    histories_override: ArrayLike | None = None,
) -> dict[str, FloatArray]:
    """Warm up on registered native history, then recurse without teacher forcing."""

    config = artifact["config"]
    normalization = artifact["normalization"]
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
    carrier_mean = np.asarray(normalization["carrier_mean"], dtype=np.float32)
    carrier_scale = np.asarray(normalization["carrier_scale"], dtype=np.float32)
    action_mean = np.asarray(normalization["action_mean"], dtype=np.float32)
    action_scale = np.asarray(normalization["action_scale"], dtype=np.float32)
    physical_mean = np.asarray(normalization["physical_mean"], dtype=np.float32)
    physical_scale = np.asarray(normalization["physical_scale"], dtype=np.float32)
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = _artifact_model(artifact, selected_device)
    history_n = torch.as_tensor((history - carrier_mean) / carrier_scale, device=selected_device)
    next_history_n = torch.as_tensor(
        (next_history - carrier_mean) / carrier_scale, device=selected_device
    )
    action_n = torch.as_tensor((action - action_mean) / action_scale, device=selected_device)
    output_carrier = carrier.copy()
    output_physical = np.zeros(
        (*carrier.shape[:2], int(config["physical_dim"])), dtype=np.float32
    )
    output_state = np.zeros(
        (*carrier.shape[:2], int(config["latent_dim"])), dtype=np.float32
    )
    direct_state = np.zeros_like(output_state)
    start = int(config["history_length"]) - 1
    with torch.inference_mode():
        teacher = model.encode(history_n)
        direct_state[:] = model.encode(next_history_n).cpu().numpy()
        state = teacher[:, start]
        for step in range(start, action.shape[1]):
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
        "direct_state": direct_state.astype(np.float64),
        "evaluation_mask": rollout_evaluation_mask(valid, config["history_length"]),
    }


def scaled_path_mse(
    prediction: ArrayLike,
    target: ArrayLike,
    mask: ArrayLike,
    scale: ArrayLike,
    *,
    final_only: bool = True,
) -> FloatArray:
    predicted = np.asarray(prediction, dtype=np.float64)
    observed = np.asarray(target, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    width_scale = np.maximum(np.asarray(scale, dtype=np.float64), 1e-8)
    if predicted.shape != observed.shape or predicted.shape[:2] != valid.shape:
        raise ValueError("path metric inputs are not aligned")
    squared = np.mean(((predicted - observed) / width_scale) ** 2, axis=2)
    if final_only:
        lengths = np.sum(valid, axis=1)
        if np.any(lengths < 1):
            raise ValueError("each sequence needs an evaluated step")
        # Evaluation masks are suffixes, so the final valid index is the final
        # path index even when a finite native-history warmup is excluded.
        index = np.max(np.where(valid, np.arange(valid.shape[1])[None, :], -1), axis=1)
        return squared[np.arange(len(squared)), index]
    return np.sum(squared * valid, axis=1) / np.sum(valid, axis=1)


def relative_gain(primary_error: ArrayLike, comparator_error: ArrayLike) -> FloatArray:
    primary = np.asarray(primary_error, dtype=np.float64)
    comparator = np.asarray(comparator_error, dtype=np.float64)
    if primary.shape != comparator.shape or primary.ndim != 1:
        raise ValueError("relative-gain inputs must be aligned vectors")
    return (comparator - primary) / np.maximum(comparator, 1e-12)


def select_pscd_candidate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("candidate selection requires at least one row")
    required = {
        "carrier_dim", "history_length", "latent_dim", "dynamics",
        "validation_score", "recursive_physical_nmse", "semigroup_nmse",
    }
    candidates = []
    for row in rows:
        if not required.issubset(row):
            raise ValueError("candidate row is incomplete")
        if not np.isfinite(float(row["validation_score"])):
            raise ValueError("candidate score is nonfinite")
        candidates.append(dict(row))
    candidates.sort(key=lambda row: (
        float(row["validation_score"]), int(row["carrier_dim"]),
        int(row["history_length"]), int(row["latent_dim"]), str(row["dynamics"]),
    ))
    return candidates[0]


@dataclass(frozen=True)
class Stage36Gates:
    source_and_split_binding: bool
    simulator_positive_control: bool
    native_physical_fidelity: bool
    distilled_state_recovery: bool
    closure_improvement: bool
    recursive_closure: bool
    semigroup_consistency: bool
    family_consistency: bool


def derive_stage36_decision(gates: Stage36Gates, *, run_mode: str) -> dict[str, Any]:
    checks = {
        "source_and_split_binding": bool(gates.source_and_split_binding),
        "simulator_positive_control": bool(gates.simulator_positive_control),
        "native_physical_fidelity": bool(gates.native_physical_fidelity),
        "distilled_state_recovery": bool(gates.distilled_state_recovery),
        "closure_improvement": bool(gates.closure_improvement),
        "recursive_closure": bool(gates.recursive_closure),
        "semigroup_consistency": bool(gates.semigroup_consistency),
        "family_consistency": bool(gates.family_consistency),
    }
    order = list(checks)
    first_failed = next((name for name in order if not checks[name]), None)
    if str(run_mode) == "smoke":
        status = "smoke_complete_not_evidence"
        passed = False
    elif first_failed is None:
        status = "bounded_predictive_state_closure_distilled"
        passed = True
    else:
        status_by_gate = {
            "source_and_split_binding": "invalid_source_or_split_binding",
            "simulator_positive_control": "operator_class_failed_positive_control",
            "native_physical_fidelity": "native_jepa_not_physically_faithful",
            "distilled_state_recovery": "predictive_state_did_not_recover_teacher",
            "closure_improvement": "adapter_did_not_improve_closure",
            "recursive_closure": "recursive_state_closure_not_observed",
            "semigroup_consistency": "composition_law_not_observed",
            "family_consistency": "closure_not_family_consistent",
        }
        status = status_by_gate[first_failed]
        passed = False
    return {
        "status": status,
        "passed": passed,
        "first_failed_gate": first_failed,
        "gates": checks,
        "causal_evidence": False,
        "original_jepa_state_claimed_closed": False,
        "minimal_predictive_state_claimed": False,
        "adapter_distillation_only": True,
    }
