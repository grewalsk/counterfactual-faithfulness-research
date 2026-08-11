"""Numerical core for the Stage 34.2 split-path continuation.

The continuation has two deliberately separate arms.  A low-capacity,
calibration-only diagonal affine map diagnoses whether DINO's Stage 34.1
failure is mostly grounded scale/bias error.  JEPA alone proceeds through the
previously unopened predictive-sufficiency and causal-use gates.

All routines here are NumPy-only.  Native checkpoint intervention code remains
in the generated Colab notebook, while its matching and scoring operations are
tested through these helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .stage34_predictive_fiber_abstraction import grouped_folds


FloatArray = NDArray[np.float64]


def _finite_matrix(values: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not len(array):
        raise ValueError(f"{name} must be a nonempty matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains nonfinite values")
    return array


def _fit_diagonal_affine(
    predictors: FloatArray,
    targets: FloatArray,
    penalty: float,
) -> tuple[FloatArray, FloatArray]:
    if predictors.shape != targets.shape:
        raise ValueError("predictors and targets must have identical shape")
    mean_x = np.mean(predictors, axis=0)
    mean_y = np.mean(targets, axis=0)
    centered_x = predictors - mean_x
    centered_y = targets - mean_y
    numerator = np.sum(centered_x * centered_y, axis=0)
    denominator = np.sum(centered_x**2, axis=0) + float(penalty)
    scale = numerator / np.maximum(denominator, 1e-12)
    intercept = mean_y - scale * mean_x
    return scale, intercept


def fit_grouped_diagonal_affine(
    predictors: ArrayLike,
    targets: ArrayLike,
    groups: ArrayLike,
    *,
    penalties: Iterable[float] = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
    folds: int = 4,
    seed: int = 0,
) -> dict[str, Any]:
    """Select a diagonal scale/bias map using grouped out-of-fold MSE."""

    x = _finite_matrix(predictors, "predictors")
    y = _finite_matrix(targets, "targets")
    labels = np.asarray(groups).reshape(-1)
    if x.shape != y.shape or len(labels) != len(x):
        raise ValueError("predictors, targets, and groups must align")
    candidates = [float(value) for value in penalties]
    if not candidates or any(value < 0 or not np.isfinite(value) for value in candidates):
        raise ValueError("penalties must be finite and nonnegative")
    masks = grouped_folds(labels, folds, seed)
    losses: list[float] = []
    for penalty in candidates:
        prediction = np.empty_like(y)
        for held_out in masks:
            scale, intercept = _fit_diagonal_affine(x[~held_out], y[~held_out], penalty)
            prediction[held_out] = x[held_out] * scale + intercept
        losses.append(float(np.mean((prediction - y) ** 2)))
    selected = int(np.argmin(losses))
    scale, intercept = _fit_diagonal_affine(x, y, candidates[selected])
    return {
        "scale": scale,
        "intercept": intercept,
        "penalty": candidates[selected],
        "oof_mse": losses[selected],
        "all_oof_mse": losses,
        "parameter_count": int(2 * x.shape[1]),
    }


def predict_diagonal_affine(model: Mapping[str, Any], predictors: ArrayLike) -> FloatArray:
    x = _finite_matrix(predictors, "predictors")
    scale = np.asarray(model["scale"], dtype=np.float64).reshape(-1)
    intercept = np.asarray(model["intercept"], dtype=np.float64).reshape(-1)
    if x.shape[1] != len(scale) or len(scale) != len(intercept):
        raise ValueError("diagonal affine dimensions do not agree")
    return x * scale + intercept


def row_cosine(left: ArrayLike, right: ArrayLike) -> FloatArray:
    first = _finite_matrix(left, "left")
    second = _finite_matrix(right, "right")
    if first.shape != second.shape:
        raise ValueError("cosine inputs must have identical shape")
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    return np.divide(
        np.sum(first * second, axis=1),
        denominator,
        out=np.zeros(len(first), dtype=np.float64),
        where=denominator > 1e-12,
    )


def row_norm_ratio(prediction: ArrayLike, target: ArrayLike) -> FloatArray:
    first = _finite_matrix(prediction, "prediction")
    second = _finite_matrix(target, "target")
    if first.shape != second.shape:
        raise ValueError("norm-ratio inputs must have identical shape")
    return np.linalg.norm(first, axis=1) / np.maximum(
        np.linalg.norm(second, axis=1), 1e-12
    )


def fit_matched_control_basis(
    carriers: ArrayLike,
    primary_subspace: Mapping[str, ArrayLike],
    rank: int,
) -> FloatArray:
    values = _finite_matrix(carriers, "carriers")
    mean = np.asarray(primary_subspace["mean"], dtype=np.float64).reshape(-1)
    scale = np.asarray(primary_subspace["scale"], dtype=np.float64).reshape(-1)
    primary = np.asarray(primary_subspace["basis"], dtype=np.float64)
    if values.shape[1] != len(mean) or len(mean) != len(scale):
        raise ValueError("carrier subspace dimensions do not agree")
    white = (values - mean) / scale
    residual = white - (white @ primary) @ primary.T
    _, singular, right = np.linalg.svd(residual, full_matrices=False)
    selected_rank = int(rank)
    threshold = max(float(singular[0]) * 1e-8, 1e-10)
    keep = min(selected_rank, int(np.sum(singular > threshold)))
    if keep < selected_rank:
        raise ValueError("matched carrier control is rank deficient")
    return right[:selected_rank].T


def project_delta_to_basis(
    delta: ArrayLike,
    subspace: Mapping[str, ArrayLike],
    basis: ArrayLike,
) -> FloatArray:
    value = np.asarray(delta, dtype=np.float64).reshape(-1)
    scale = np.asarray(subspace["scale"], dtype=np.float64).reshape(-1)
    directions = np.asarray(basis, dtype=np.float64)
    if len(value) != len(scale) or directions.shape[0] != len(value):
        raise ValueError("delta and basis dimensions do not agree")
    white = value / scale
    return (directions @ (directions.T @ white)) * scale


def summarize_causal_rows(
    rows: list[Mapping[str, Any]],
    mode_labels: Iterable[str],
    *,
    minimum_retention: float,
    minimum_cosine: float,
    minimum_control_advantage: float,
    maximum_fiber_ratio: float,
    maximum_ood_rate: float,
) -> dict[str, Any]:
    """Apply the frozen Stage 34 causal-use gate to one model's row table."""

    state_primary = [
        row for row in rows if row["kind"] == "state" and row["condition"] == "primary"
    ]
    state_positive = [
        row for row in rows
        if row["kind"] == "state" and row["condition"] == "full_swap_positive"
    ]
    state_random = [
        row for row in rows
        if row["kind"] == "state" and row["condition"] == "random_matched_subspace"
    ]
    fiber_primary = [
        row for row in rows if row["kind"] == "fiber" and row["condition"] == "primary"
    ]
    if not all([state_primary, state_positive, state_random, fiber_primary]):
        raise ValueError("causal row table is incomplete")
    primary_gain = np.asarray([row["error_gain"] for row in state_primary], dtype=np.float64)
    positive_gain = np.asarray([row["error_gain"] for row in state_positive], dtype=np.float64)
    random_gain = np.asarray([row["error_gain"] for row in state_random], dtype=np.float64)
    mean_positive = float(np.mean(positive_gain))
    retention = float(np.mean(primary_gain) / max(mean_positive, 1e-12))
    control_advantage = float(np.mean(primary_gain - random_gain))
    fiber_ratio = float(np.mean([row["fiber_effect_ratio"] for row in fiber_primary]))
    ood_rate = float(np.mean([
        float(row["ood_ratio"] > 1.0) for row in state_primary + fiber_primary
    ]))
    mode_retention: dict[str, float] = {}
    mode_positive: dict[str, float] = {}
    for mode in map(str, mode_labels):
        primary_values = [row["error_gain"] for row in state_primary if row["mode"] == mode]
        positive_values = [row["error_gain"] for row in state_positive if row["mode"] == mode]
        if not primary_values or not positive_values:
            raise ValueError(f"causal rows omit mode {mode!r}")
        mode_positive[mode] = float(np.mean(positive_values))
        mode_retention[mode] = float(
            np.mean(primary_values) / max(mode_positive[mode], 1e-12)
        )
    mean_effect_cosine = float(np.mean([row["effect_cosine"] for row in state_primary]))
    passed = bool(
        mean_positive > 0
        and retention >= float(minimum_retention)
        and mean_effect_cosine >= float(minimum_cosine)
        and control_advantage >= float(minimum_control_advantage)
        and fiber_ratio <= float(maximum_fiber_ratio)
        and ood_rate <= float(maximum_ood_rate)
        and all(value > 0 for value in mode_positive.values())
        and all(value > 0 for value in mode_retention.values())
    )
    return {
        "state_rows": len(state_primary),
        "fiber_rows": len(fiber_primary),
        "mean_state_effect_retention": retention,
        "mean_full_swap_positive_gain": mean_positive,
        "mean_state_effect_cosine": mean_effect_cosine,
        "mean_control_advantage": control_advantage,
        "mean_fiber_effect_ratio": fiber_ratio,
        "intervention_ood_rate": ood_rate,
        "mode_state_retention": mode_retention,
        "mode_full_swap_positive_gain": mode_positive,
        "passed": passed,
    }


@dataclass(frozen=True)
class Stage342Gates:
    upstream_binding: bool
    stage341_binding: bool
    jepa_action_specificity: bool
    jepa_predictive_sufficiency: bool
    jepa_causal_evaluated: bool
    jepa_causal_use: bool
    dino_diagonal_recoverability: bool


def derive_stage342_decision(
    gates: Stage342Gates | Mapping[str, bool],
    *,
    run_mode: str = "pilot",
) -> dict[str, Any]:
    """Return a split-path diagnostic decision without reviving a shared claim."""

    values = gates.__dict__ if isinstance(gates, Stage342Gates) else dict(gates)
    required = [
        "upstream_binding",
        "stage341_binding",
        "jepa_action_specificity",
        "jepa_predictive_sufficiency",
        "jepa_causal_evaluated",
        "jepa_causal_use",
        "dino_diagonal_recoverability",
    ]
    missing = [name for name in required if name not in values]
    if missing:
        raise ValueError(f"missing Stage 34.2 gates: {missing}")
    checks = {name: bool(values[name]) for name in required}
    if run_mode == "smoke":
        status = "smoke_only"
    elif not checks["upstream_binding"] or not checks["stage341_binding"]:
        status = "inconclusive_upstream_binding_failure"
    elif not checks["jepa_action_specificity"]:
        status = "inconclusive_jepa_specificity_binding_failure"
    elif not checks["jepa_predictive_sufficiency"]:
        status = "jepa_response_state_insufficient"
    elif not checks["jepa_causal_evaluated"]:
        status = "inconclusive_jepa_causal_gate_not_evaluated"
    elif not checks["jepa_causal_use"]:
        status = "jepa_response_state_not_causally_used"
    elif checks["dino_diagonal_recoverability"]:
        status = "jepa_causal_state_dino_calibration_limited"
    else:
        status = "jepa_only_causal_state_dino_not_calibration_recoverable"
    passed = bool(
        run_mode == "pilot"
        and checks["upstream_binding"]
        and checks["stage341_binding"]
        and checks["jepa_action_specificity"]
        and checks["jepa_predictive_sufficiency"]
        and checks["jepa_causal_evaluated"]
        and checks["jepa_causal_use"]
    )
    return {
        "status": status,
        "passed": passed,
        "checks": checks,
        "failed_checks": [name for name in required if not checks[name]],
        "run_mode": str(run_mode),
        "confirmation_eligible": False,
        "claim_scope": "post_outcome_split_path_diagnostic_not_shared_abstraction",
    }
