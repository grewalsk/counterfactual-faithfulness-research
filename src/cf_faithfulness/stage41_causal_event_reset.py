"""Numerical helpers for the Stage 41 causal event/reset headroom audit.

Stage 41 is deliberately diagnostic.  A frozen recursive predictor is held
fixed while equal-width ridge heads receive an increasing amount of oracle
simulator event information.  The helpers in this module contain no simulator
or checkpoint code, which makes the registered comparisons independently
testable with NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike


CAUSAL_VARIANTS = (
    "smooth_matched",
    "shuffled_event",
    "oracle_event",
    "oracle_time",
    "oracle_geometry",
    "oracle_reset_ceiling",
)


def _as_matrix(value: ArrayLike, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite matrix")
    return result


def mean_scale(value: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Return stable columnwise centering and scaling statistics."""

    matrix = _as_matrix(value, "value")
    if len(matrix) < 2:
        raise ValueError("at least two rows are required")
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0, ddof=1)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    return mean, scale


def storage_equivalent(
    reference: ArrayLike,
    stored: ArrayLike,
    *,
    storage_dtype: Any = np.float32,
) -> bool:
    """Check exact equality at a declared serialization precision boundary.

    Stage 41 truth shards retain float64 simulator values, while the compact
    model-path shards intentionally store the same grounded paths as float32.
    Comparing those arrays at float64 precision creates a false mismatch from
    ordinary float32 rounding.  This helper makes the storage contract explicit
    and still requires bit-exact equality after the registered cast.
    """

    expected = np.asarray(reference)
    observed = np.asarray(stored)
    if expected.shape != observed.shape:
        return False
    if not np.all(np.isfinite(expected)) or not np.all(np.isfinite(observed)):
        return False
    dtype = np.dtype(storage_dtype)
    return bool(np.array_equal(expected.astype(dtype), observed.astype(dtype)))


def fixed_sham_projection(input_dim: int, width: int, seed: int) -> dict[str, np.ndarray]:
    """Create a frozen outcome-independent smooth feature projection."""

    if int(input_dim) < 1 or int(width) < 1:
        raise ValueError("projection dimensions must be positive")
    rng = np.random.default_rng(int(seed))
    weight = rng.normal(size=(int(input_dim), int(width))) / np.sqrt(input_dim)
    bias = rng.uniform(-0.5, 0.5, size=int(width))
    return {"weight": weight.astype(np.float64), "bias": bias.astype(np.float64)}


def deterministic_permutation(length: int, seed: int) -> np.ndarray:
    """Return a registered permutation for the shuffled-event control."""

    if int(length) < 2:
        raise ValueError("a shuffled control needs at least two rows")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(int(length))
    if np.array_equal(order, np.arange(int(length))):
        order = np.roll(order, 1)
    return order.astype(np.int64)


def causal_design_matrix(
    base_features: ArrayLike,
    metadata: ArrayLike,
    *,
    variant: str,
    base_mean: ArrayLike,
    base_scale: ArrayLike,
    metadata_mean: ArrayLike,
    metadata_scale: ArrayLike,
    sham_projection: Mapping[str, ArrayLike],
    permutation: ArrayLike | None = None,
) -> np.ndarray:
    """Build an equal-width design for one rung of the oracle ladder.

    Metadata columns are registered as ``event, time, normal_x, normal_y,
    log_normal_impulse, tangent_impulse``.  Oracle rungs replace progressively
    more smooth sham columns, so every fitted head has the same nominal width.
    """

    base = _as_matrix(base_features, "base_features")
    meta = _as_matrix(metadata, "metadata")
    if len(base) != len(meta) or meta.shape[1] != 6:
        raise ValueError("base features and six-column metadata are not aligned")
    if str(variant) not in CAUSAL_VARIANTS:
        raise KeyError(f"unknown Stage 41 variant {variant!r}")
    bmean = np.asarray(base_mean, dtype=np.float64)
    bscale = np.asarray(base_scale, dtype=np.float64)
    mmean = np.asarray(metadata_mean, dtype=np.float64)
    mscale = np.asarray(metadata_scale, dtype=np.float64)
    if bmean.shape != (base.shape[1],) or bscale.shape != bmean.shape:
        raise ValueError("base normalization does not match feature width")
    if mmean.shape != (6,) or mscale.shape != (6,):
        raise ValueError("metadata normalization must have width six")
    if np.any(bscale <= 0) or np.any(mscale <= 0):
        raise ValueError("normalization scales must be positive")
    normalized_base = (base - bmean) / bscale
    normalized_meta = (meta - mmean) / mscale
    projection = _as_matrix(sham_projection["weight"], "sham weight")
    bias = np.asarray(sham_projection["bias"], dtype=np.float64)
    if projection.shape != (base.shape[1], 6) or bias.shape != (6,):
        raise ValueError("sham projection has the wrong shape")
    sham = np.tanh(normalized_base @ projection + bias)
    slots = sham.copy()
    revealed = {
        "smooth_matched": 0,
        "shuffled_event": 6,
        "oracle_event": 1,
        "oracle_time": 2,
        "oracle_geometry": 4,
        "oracle_reset_ceiling": 6,
    }[str(variant)]
    if variant == "shuffled_event":
        if permutation is None:
            raise ValueError("shuffled_event requires a registered permutation")
        order = np.asarray(permutation, dtype=np.int64)
        if order.shape != (len(meta),) or set(order.tolist()) != set(range(len(meta))):
            raise ValueError("invalid shuffled-event permutation")
        slots = normalized_meta[order]
    elif revealed:
        slots[:, :revealed] = normalized_meta[:, :revealed]
    result = np.concatenate([normalized_base, slots], axis=1)
    if not np.all(np.isfinite(result)):
        raise ValueError("causal design contains nonfinite values")
    return result


def fit_ridge(x: ArrayLike, y: ArrayLike, penalty: float) -> dict[str, np.ndarray | float]:
    """Fit a multi-output ridge with an unpenalized intercept."""

    design = _as_matrix(x, "x")
    target = _as_matrix(y, "y")
    if len(design) != len(target) or len(design) < 2:
        raise ValueError("ridge inputs are not aligned")
    value = float(penalty)
    if not np.isfinite(value) or value < 0:
        raise ValueError("ridge penalty must be finite and nonnegative")
    x_mean, y_mean = np.mean(design, axis=0), np.mean(target, axis=0)
    centered_x, centered_y = design - x_mean, target - y_mean
    gram = centered_x.T @ centered_x
    weight = np.linalg.solve(
        gram + value * np.eye(gram.shape[0], dtype=np.float64),
        centered_x.T @ centered_y,
    )
    intercept = y_mean - x_mean @ weight
    return {
        "weight": weight,
        "intercept": intercept,
        "penalty": value,
    }


def ridge_predict(artifact: Mapping[str, Any], x: ArrayLike) -> np.ndarray:
    design = _as_matrix(x, "x")
    weight = _as_matrix(artifact["weight"], "weight")
    intercept = np.asarray(artifact["intercept"], dtype=np.float64)
    if design.shape[1] != weight.shape[0] or intercept.shape != (weight.shape[1],):
        raise ValueError("ridge artifact and design are not aligned")
    return design @ weight + intercept


def select_ridge_penalty(
    train_x: ArrayLike,
    train_y: ArrayLike,
    validation_x: ArrayLike,
    validation_y: ArrayLike,
    penalties: Sequence[float],
) -> dict[str, Any]:
    """Select a ridge penalty on a separate development split."""

    x_train = _as_matrix(train_x, "train_x")
    y_train = _as_matrix(train_y, "train_y")
    x_validation = _as_matrix(validation_x, "validation_x")
    y_validation = _as_matrix(validation_y, "validation_y")
    if len(x_train) != len(y_train) or len(x_validation) != len(y_validation):
        raise ValueError("selection inputs are not aligned")
    candidates = sorted(set(float(value) for value in penalties))
    if not candidates or any(value < 0 or not np.isfinite(value) for value in candidates):
        raise ValueError("ridge penalty grid is invalid")
    rows = []
    for penalty in candidates:
        fitted = fit_ridge(x_train, y_train, penalty)
        residual = ridge_predict(fitted, x_validation) - y_validation
        rows.append({"penalty": penalty, "validation_mse": float(np.mean(residual**2))})
    selected = min(rows, key=lambda row: (row["validation_mse"], row["penalty"]))
    return {"selected_penalty": selected["penalty"], "candidate_rows": rows}


def upper_tail_mean(values: ArrayLike, mass: float = 0.10) -> float:
    """Return empirical upper-tail CVaR with at least one included row."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError("tail values must be nonempty and finite")
    fraction = float(mass)
    if not 0 < fraction <= 1:
        raise ValueError("tail mass must lie in (0, 1]")
    count = max(1, int(np.ceil(fraction * len(array))))
    return float(np.mean(np.sort(array)[-count:]))


def causal_effect_metrics(
    predicted_delta: ArrayLike,
    physical_delta: ArrayLike,
    physical_scale: ArrayLike,
    event_mask: ArrayLike,
) -> dict[str, float | int]:
    """Score a model-side intervention against the exact ordinary-minus-ghost effect."""

    predicted = _as_matrix(predicted_delta, "predicted_delta")
    physical = _as_matrix(physical_delta, "physical_delta")
    scale = np.asarray(physical_scale, dtype=np.float64)
    selected = np.asarray(event_mask, dtype=bool).reshape(-1)
    if predicted.shape != physical.shape or len(selected) != len(predicted):
        raise ValueError("causal effect arrays are not aligned")
    if scale.shape != (predicted.shape[1],) or np.any(scale <= 0):
        raise ValueError("physical scale does not match causal effect width")
    selected &= np.linalg.norm(physical / scale, axis=1) > 1e-8
    if not np.any(selected):
        return {
            "rows": 0, "effect_nmse": float("inf"), "zero_effect_nmse": float("inf"),
            "relative_gain_over_zero": float("-inf"), "mean_cosine": float("nan"),
            "median_magnitude_ratio": float("nan"),
        }
    p = predicted[selected] / scale
    t = physical[selected] / scale
    error = np.mean((p - t) ** 2, axis=1)
    zero = np.mean(t**2, axis=1)
    denominator = np.linalg.norm(p, axis=1) * np.linalg.norm(t, axis=1)
    cosine = np.divide(
        np.sum(p * t, axis=1), denominator,
        out=np.zeros_like(denominator), where=denominator > 1e-12,
    )
    magnitude = np.divide(
        np.linalg.norm(p, axis=1), np.linalg.norm(t, axis=1),
        out=np.zeros_like(denominator), where=np.linalg.norm(t, axis=1) > 1e-12,
    )
    mean_error, mean_zero = float(np.mean(error)), float(np.mean(zero))
    return {
        "rows": int(np.sum(selected)),
        "effect_nmse": mean_error,
        "zero_effect_nmse": mean_zero,
        "relative_gain_over_zero": float(1.0 - mean_error / max(mean_zero, 1e-12)),
        "mean_cosine": float(np.mean(cosine)),
        "median_magnitude_ratio": float(np.median(magnitude)),
    }


@dataclass(frozen=True)
class Stage41PanelDecision:
    model: str
    passed: bool
    classification: str
    all_seed_tail_improvement: bool
    all_seed_p95_improvement: bool
    all_seed_mean_noninferiority: bool
    all_seed_control_dominance: bool
    all_seed_causal_alignment: bool


def derive_stage41_panel_decision(
    model: str,
    *,
    all_seed_tail_improvement: bool,
    all_seed_p95_improvement: bool,
    all_seed_mean_noninferiority: bool,
    all_seed_control_dominance: bool,
    all_seed_causal_alignment: bool,
) -> Stage41PanelDecision:
    gates = [
        bool(all_seed_tail_improvement), bool(all_seed_p95_improvement),
        bool(all_seed_mean_noninferiority), bool(all_seed_control_dominance),
        bool(all_seed_causal_alignment),
    ]
    passed = bool(all(gates))
    if passed:
        classification = "oracle_event_reset_headroom_confirmed"
    elif not all_seed_causal_alignment:
        classification = "oracle_correction_not_causally_aligned"
    elif not all_seed_control_dominance:
        classification = "oracle_does_not_beat_matched_controls"
    elif not all_seed_tail_improvement or not all_seed_p95_improvement:
        classification = "oracle_headroom_below_registered_tail_target"
    else:
        classification = "oracle_headroom_costs_excess_mean_error"
    return Stage41PanelDecision(
        model=str(model), passed=passed, classification=classification,
        all_seed_tail_improvement=bool(all_seed_tail_improvement),
        all_seed_p95_improvement=bool(all_seed_p95_improvement),
        all_seed_mean_noninferiority=bool(all_seed_mean_noninferiority),
        all_seed_control_dominance=bool(all_seed_control_dominance),
        all_seed_causal_alignment=bool(all_seed_causal_alignment),
    )


def derive_stage41_decision(
    panels: Mapping[str, Stage41PanelDecision],
) -> dict[str, Any]:
    """Return the conjunctive cross-model Stage 41 development decision."""

    if set(panels) != {"jepa", "dino"}:
        raise ValueError("Stage 41 requires separate JEPA and DINO panels")
    if all(panel.passed for panel in panels.values()):
        status = "cross_model_oracle_event_reset_headroom_confirmed"
        next_step = "build_label_free_event_state_identifiability_test"
        passed = True
    else:
        status = "oracle_event_reset_headroom_not_confirmed"
        next_step = "stop_event_reset_adapter_and_revisit_state_abstraction"
        passed = False
    return {
        "status": status,
        "passed": passed,
        "next_step": next_step,
        "panels": {key: value.classification for key, value in panels.items()},
        "causal_claim_authorized": False,
        "learned_deployment_claim_authorized": False,
    }
