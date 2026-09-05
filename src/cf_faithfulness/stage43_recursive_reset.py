"""Numerical core for Stage 43 recursive reset sufficiency experiments.

Stage 42 used an affine, post-hoc correction of decoded physical outputs.  The
functions here define the stricter Stage 43 question: can an event-conditioned
operator reset the predictive carrier itself and thereby improve all later
recursive predictions?  The module is NumPy-only so its feature construction,
recursion semantics, risk gates, and decision tree can be tested locally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
from numpy.typing import ArrayLike


def _finite_matrix(value: ArrayLike, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite matrix")
    return result


def mean_scale(value: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Return stable columnwise centering and sample scales."""

    matrix = _finite_matrix(value, "value")
    if len(matrix) < 2:
        raise ValueError("normalization requires at least two rows")
    return np.mean(matrix, axis=0), np.maximum(np.std(matrix, axis=0, ddof=1), 1e-8)


def lagged_state_features(
    states: ArrayLike,
    mask: ArrayLike,
    *,
    lags: int,
) -> np.ndarray:
    """Stack the current and previous valid states without crossing a row boundary."""

    values = np.asarray(states, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if values.ndim != 3 or valid.shape != values.shape[:2]:
        raise ValueError("states and mask are not sequence-aligned")
    if not np.all(np.isfinite(values)) or int(lags) < 1:
        raise ValueError("states must be finite and lags must be positive")
    result = np.empty((*values.shape[:2], int(lags) * values.shape[2]), dtype=np.float64)
    for row in range(values.shape[0]):
        for step in range(values.shape[1]):
            pieces = []
            for offset in range(int(lags)):
                source = max(0, step - offset)
                while source < step and not valid[row, source]:
                    source += 1
                if not valid[row, source]:
                    source = step
                pieces.append(values[row, source])
            result[row, step] = np.concatenate(pieces)
    return result


def preceding_physical_state(
    initial: ArrayLike,
    path: ArrayLike,
    mask: ArrayLike,
) -> np.ndarray:
    """Return the exact pre-macro physical state for each path position."""

    first = _finite_matrix(initial, "initial")
    values = np.asarray(path, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if values.ndim != 3 or valid.shape != values.shape[:2]:
        raise ValueError("path and mask are not sequence-aligned")
    if values.shape[0] != len(first) or values.shape[2] != first.shape[1]:
        raise ValueError("initial physical state does not match path")
    if not np.all(np.isfinite(values)):
        raise ValueError("physical path must be finite")
    result = np.repeat(first[:, None, :], values.shape[1], axis=1)
    if values.shape[1] > 1:
        result[:, 1:] = values[:, :-1]
    return result


def reset_base_tensor(
    proposed_state: ArrayLike,
    actions: ArrayLike,
    mask: ArrayLike,
    *,
    representation: str,
    history_lags: int = 3,
    initial_physical: ArrayLike | None = None,
    physical_path: ArrayLike | None = None,
) -> np.ndarray:
    """Construct one nested state representation for reset identification."""

    state = np.asarray(proposed_state, dtype=np.float64)
    action = np.asarray(actions, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if state.ndim != 3 or action.ndim != 3 or state.shape[:2] != action.shape[:2]:
        raise ValueError("state and actions are not sequence-aligned")
    if valid.shape != state.shape[:2] or not np.all(np.isfinite(state)):
        raise ValueError("reset mask or state is invalid")
    if not np.all(np.isfinite(action)):
        raise ValueError("actions must be finite")
    name = str(representation)
    if name == "current":
        pieces = [state, action]
    elif name == "history":
        pieces = [lagged_state_features(state, valid, lags=int(history_lags)), action]
    elif name == "physical_oracle":
        if initial_physical is None or physical_path is None:
            raise ValueError("physical oracle representation requires exact states")
        previous = preceding_physical_state(initial_physical, physical_path, valid)
        pieces = [state, action, previous]
    else:
        raise KeyError(f"unknown reset representation {name!r}")
    return np.concatenate(pieces, axis=-1)


def fixed_sham_projection(width: int, slots: int, seed: int) -> dict[str, np.ndarray]:
    """Create frozen outcome-independent metadata replacements."""

    if int(width) < 1 or int(slots) < 1:
        raise ValueError("projection dimensions must be positive")
    rng = np.random.default_rng(int(seed))
    return {
        "weight": rng.normal(size=(int(width), int(slots))) / np.sqrt(float(width)),
        "bias": rng.uniform(-0.5, 0.5, size=int(slots)),
    }


def fixed_state_projection(width: int, rank: int, seed: int) -> dict[str, np.ndarray]:
    """Create a frozen low-rank lift for parameter-matched tensor operators."""

    if int(width) < 1 or int(rank) < 1 or int(rank) > int(width):
        raise ValueError("state projection rank must lie between one and width")
    rng = np.random.default_rng(int(seed))
    weight = rng.normal(size=(int(width), int(rank)))
    weight, _ = np.linalg.qr(weight, mode="reduced")
    return {"weight": weight, "bias": np.zeros(int(rank), dtype=np.float64)}


def tensor_reset_design(
    base: ArrayLike,
    metadata: ArrayLike,
    *,
    base_mean: ArrayLike,
    base_scale: ArrayLike,
    metadata_mean: ArrayLike,
    metadata_scale: ArrayLike,
    metadata_mode: str = "oracle",
    sham_projection: Mapping[str, ArrayLike] | None = None,
    state_projection: Mapping[str, ArrayLike] | None = None,
) -> np.ndarray:
    """Return normalized main effects plus state-by-event tensor products."""

    x = _finite_matrix(base, "base")
    eta = _finite_matrix(metadata, "metadata")
    if len(x) != len(eta):
        raise ValueError("base and metadata rows do not align")
    x_mean = np.asarray(base_mean, dtype=np.float64)
    x_scale = np.asarray(base_scale, dtype=np.float64)
    e_mean = np.asarray(metadata_mean, dtype=np.float64)
    e_scale = np.asarray(metadata_scale, dtype=np.float64)
    if x_mean.shape != (x.shape[1],) or x_scale.shape != x_mean.shape:
        raise ValueError("base normalization has the wrong shape")
    if e_mean.shape != (eta.shape[1],) or e_scale.shape != e_mean.shape:
        raise ValueError("metadata normalization has the wrong shape")
    if np.any(x_scale <= 0) or np.any(e_scale <= 0):
        raise ValueError("normalization scales must be positive")
    xn = (x - x_mean) / x_scale
    if state_projection is None:
        lifted = xn
    else:
        state_weight = _finite_matrix(state_projection["weight"], "state weight")
        state_bias = np.asarray(state_projection["bias"], dtype=np.float64)
        if state_weight.shape[0] != x.shape[1] or state_bias.shape != (state_weight.shape[1],):
            raise ValueError("state projection has the wrong shape")
        lifted = xn @ state_weight + state_bias
    mode = str(metadata_mode)
    if mode == "oracle":
        slots = (eta - e_mean) / e_scale
    elif mode == "sham":
        if sham_projection is None:
            raise ValueError("sham metadata requires a frozen projection")
        weight = _finite_matrix(sham_projection["weight"], "sham weight")
        bias = np.asarray(sham_projection["bias"], dtype=np.float64)
        if weight.shape != (x.shape[1], eta.shape[1]) or bias.shape != (eta.shape[1],):
            raise ValueError("sham projection has the wrong shape")
        slots = np.tanh(xn @ weight + bias)
    else:
        raise KeyError(f"unknown metadata mode {mode!r}")
    interaction = np.einsum("ni,nj->nij", lifted, slots).reshape(len(xn), -1)
    result = np.concatenate([lifted, slots, interaction], axis=1)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("reset design contains nonfinite values")
    return result


def fit_ridge(design: ArrayLike, target: ArrayLike, penalty: float) -> dict[str, np.ndarray]:
    """Fit multi-output ridge with an unpenalized intercept via centering."""

    x = _finite_matrix(design, "design")
    y = _finite_matrix(target, "target")
    if len(x) != len(y) or float(penalty) < 0 or not np.isfinite(penalty):
        raise ValueError("ridge inputs are invalid")
    x_mean, y_mean = np.mean(x, axis=0), np.mean(y, axis=0)
    xc, yc = x - x_mean, y - y_mean
    if x.shape[1] <= x.shape[0]:
        gram = xc.T @ xc + float(penalty) * np.eye(x.shape[1], dtype=np.float64)
        weight = np.linalg.solve(gram, xc.T @ yc)
    else:
        gram = xc @ xc.T + float(penalty) * np.eye(x.shape[0], dtype=np.float64)
        weight = xc.T @ np.linalg.solve(gram, yc)
    return {"weight": weight, "intercept": y_mean - x_mean @ weight}


def ridge_predict(artifact: Mapping[str, ArrayLike], design: ArrayLike) -> np.ndarray:
    x = _finite_matrix(design, "design")
    weight = _finite_matrix(artifact["weight"], "weight")
    intercept = np.asarray(artifact["intercept"], dtype=np.float64)
    if x.shape[1] != weight.shape[0] or intercept.shape != (weight.shape[1],):
        raise ValueError("ridge artifact and design do not align")
    return x @ weight + intercept


def clip_row_norms(value: ArrayLike, maximum_norm: float) -> np.ndarray:
    """Clip each finite row to a prespecified Euclidean trust region."""

    matrix = _finite_matrix(value, "value")
    limit = float(maximum_norm)
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("maximum norm must be finite and positive")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    result = matrix * np.minimum(1.0, limit / np.maximum(norms, 1e-12))
    if not np.all(np.isfinite(result)):
        raise RuntimeError("norm clipping produced nonfinite values")
    return result


def select_ridge_penalty(
    train_design: ArrayLike,
    train_target: ArrayLike,
    validation_design: ArrayLike,
    validation_target: ArrayLike,
    penalties: list[float],
) -> dict[str, Any]:
    """Select a ridge penalty using validation data only."""

    x_train = _finite_matrix(train_design, "train design")
    y_train = _finite_matrix(train_target, "train target")
    x_validation = _finite_matrix(validation_design, "validation design")
    y_validation = _finite_matrix(validation_target, "validation target")
    if len(x_train) != len(y_train) or len(x_validation) != len(y_validation):
        raise ValueError("selection inputs are not aligned")
    candidates = sorted(set(float(value) for value in penalties))
    if not candidates or any(value < 0 or not np.isfinite(value) for value in candidates):
        raise ValueError("ridge penalty grid is invalid")
    rows = []
    for penalty in candidates:
        artifact = fit_ridge(x_train, y_train, penalty)
        residual = ridge_predict(artifact, x_validation) - y_validation
        rows.append({"penalty": penalty, "validation_mse": float(np.mean(residual**2))})
    selected = min(rows, key=lambda row: (row["validation_mse"], row["penalty"]))
    return {"selected_penalty": selected["penalty"], "candidate_rows": rows}


def recursive_reset_rollout_numpy(
    initial_state: ArrayLike,
    actions: ArrayLike,
    metadata: ArrayLike,
    mask: ArrayLike,
    *,
    transition: Callable[[np.ndarray, np.ndarray], np.ndarray],
    correction: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    history_lags: int = 3,
) -> np.ndarray:
    """Reference recursion: correct the carrier at events before the next step."""

    state = _finite_matrix(initial_state, "initial state").copy()
    action = np.asarray(actions, dtype=np.float64)
    eta = np.asarray(metadata, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if action.ndim != 3 or eta.ndim != 3 or action.shape[:2] != eta.shape[:2]:
        raise ValueError("actions and metadata are not sequence-aligned")
    if valid.shape != action.shape[:2] or len(state) != action.shape[0]:
        raise ValueError("rollout inputs are not aligned")
    output = np.zeros((len(state), action.shape[1], state.shape[1]), dtype=np.float64)
    history = np.repeat(state[:, None, :], int(history_lags), axis=1)
    for step in range(action.shape[1]):
        active = valid[:, step]
        if not np.any(active):
            continue
        proposed = _finite_matrix(transition(state[active], action[active, step]), "transition")
        updated = proposed.copy()
        event = eta[active, step, 0] > 0.5
        if np.any(event):
            delta = _finite_matrix(
                correction(
                    proposed[event], action[active, step][event],
                    eta[active, step][event], history[active][event],
                ),
                "correction",
            )
            if delta.shape != updated[event].shape:
                raise ValueError("correction has the wrong state width")
            updated[event] += delta
        state[active] = updated
        output[active, step] = updated
        active_rows = np.flatnonzero(active)
        history[active_rows, 1:] = history[active_rows, :-1]
        history[active_rows, 0] = updated
    return output


def upper_tail_mean(values: ArrayLike, mass: float = 0.25) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError("tail values must be nonempty and finite")
    if not 0 < float(mass) <= 1:
        raise ValueError("tail mass must lie in (0, 1]")
    count = max(1, int(np.ceil(float(mass) * len(array))))
    return float(np.mean(np.sort(array)[-count:]))


def reset_risk_metrics(
    candidate_error: ArrayLike,
    baseline_error: ArrayLike,
    reentry_mask: ArrayLike,
    groups: ArrayLike,
    *,
    tail_mass: float = 0.25,
    minimum_rows: int = 32,
) -> dict[str, float | int]:
    """Compute the registered mean, p95, tail, and family-robust gains."""

    candidate = np.asarray(candidate_error, dtype=np.float64).reshape(-1)
    baseline = np.asarray(baseline_error, dtype=np.float64).reshape(-1)
    reentry = np.asarray(reentry_mask, dtype=bool).reshape(-1)
    family = np.asarray(groups).reshape(-1)
    if not (candidate.shape == baseline.shape == reentry.shape == family.shape):
        raise ValueError("risk rows are not aligned")
    if np.any(candidate < 0) or np.any(baseline < 0):
        raise ValueError("risk values must be nonnegative")
    if int(np.sum(reentry)) < int(minimum_rows):
        raise ValueError("insufficient re-entry support")
    baseline_tail = upper_tail_mean(baseline[reentry], tail_mass)
    candidate_tail = upper_tail_mean(candidate[reentry], tail_mass)
    baseline_p95 = float(np.quantile(baseline, 0.95))
    candidate_p95 = float(np.quantile(candidate, 0.95))
    leave_one = []
    for value in np.unique(family):
        selected = reentry & (family != value)
        if int(np.sum(selected)) < int(minimum_rows):
            continue
        reference = upper_tail_mean(baseline[selected], tail_mass)
        proposed = upper_tail_mean(candidate[selected], tail_mass)
        leave_one.append((reference - proposed) / max(reference, 1e-12))
    return {
        "rows": int(len(candidate)),
        "reentry_rows": int(np.sum(reentry)),
        "mean_ratio": float(np.mean(candidate) / max(float(np.mean(baseline)), 1e-12)),
        "p95_relative_gain": float((baseline_p95 - candidate_p95) / max(baseline_p95, 1e-12)),
        "tail_relative_gain": float((baseline_tail - candidate_tail) / max(baseline_tail, 1e-12)),
        "minimum_leave_one_family_tail_gain": float(min(leave_one)) if leave_one else float("-inf"),
    }


def passes_registered_reset_gates(
    metrics: Mapping[str, float | int],
    *,
    minimum_tail_gain: float = 0.25,
    minimum_p95_gain: float = 0.10,
    maximum_mean_ratio: float = 1.02,
    minimum_leave_one_gain: float = 0.10,
) -> bool:
    return bool(
        float(metrics["tail_relative_gain"]) >= float(minimum_tail_gain)
        and float(metrics["p95_relative_gain"]) >= float(minimum_p95_gain)
        and float(metrics["mean_ratio"]) <= float(maximum_mean_ratio)
        and float(metrics["minimum_leave_one_family_tail_gain"])
        >= float(minimum_leave_one_gain)
    )


@dataclass(frozen=True)
class Stage43Decision:
    passed: bool
    classification: str
    support_certified: bool
    current_tensor_headroom: bool
    current_nonlinear_headroom: bool
    history_headroom: bool
    physical_oracle_headroom: bool
    learned_recursive_reset_authorized: bool


def derive_stage43_decision(
    *,
    support_certified: bool,
    current_tensor_headroom: bool,
    current_nonlinear_headroom: bool,
    history_headroom: bool,
    physical_oracle_headroom: bool,
) -> Stage43Decision:
    """Locate failure in operator class, predictive state, or reset hypothesis."""

    support = bool(support_certified)
    tensor = bool(current_tensor_headroom)
    nonlinear = bool(current_nonlinear_headroom)
    history = bool(history_headroom)
    physical = bool(physical_oracle_headroom)
    authorized = bool(support and (tensor or nonlinear))
    if not support:
        classification = "event_support_not_certified"
    elif tensor:
        classification = "recursive_tensor_reset_headroom_confirmed"
    elif nonlinear:
        classification = "nonlinear_recursive_reset_required"
    elif history:
        classification = "short_history_state_completion_required"
    elif physical:
        classification = "frozen_carrier_state_insufficient"
    else:
        classification = "reset_hypothesis_not_supported"
    return Stage43Decision(
        passed=authorized,
        classification=classification,
        support_certified=support,
        current_tensor_headroom=tensor,
        current_nonlinear_headroom=nonlinear,
        history_headroom=history,
        physical_oracle_headroom=physical,
        learned_recursive_reset_authorized=authorized,
    )
