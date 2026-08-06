"""Numerical primitives for Stage 29 grounded causal closure.

Stage 29 compares a JEPA-WM prediction directly with the frozen encoder's
representation of the exact simulator future.  Both tensors therefore live in
the same latent coordinates.  The functions here deliberately operate on
arbitrary trailing dimensions so the same definitions can be used for exact
token-space metrics and small synthetic validation tests.
"""

from __future__ import annotations

import math

import numpy as np

from .stage28_hybrid_control_area import (
    area_antisymmetric_component,
    area_reversal_permutation,
    magnitude_center,
)


def vector_alignment(source, target):
    """Return coefficient, cosine, and normalized error toward ``target``.

    The coefficient is the least-squares scalar multiplying the target.  No
    centering is performed here; callers select the scientifically relevant
    contrast before calling this function.
    """

    left = np.asarray(source, dtype=np.float64)
    right = np.asarray(target, dtype=np.float64)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("source and target must have the same nonempty shape")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("source and target must be finite")
    left = left.reshape(-1)
    right = right.reshape(-1)
    source_energy = float(np.dot(left, left))
    target_energy = float(np.dot(right, right))
    dot = float(np.dot(left, right))
    if target_energy <= 1e-20:
        return {
            "source_energy": source_energy,
            "target_energy": target_energy,
            "dot": dot,
            "coefficient": math.nan,
            "cosine": math.nan,
            "normalized_rmse": math.nan,
        }
    denominator = math.sqrt(max(source_energy * target_energy, 0.0))
    return {
        "source_energy": source_energy,
        "target_energy": target_energy,
        "dot": dot,
        "coefficient": float(dot / target_energy),
        "cosine": float(dot / denominator) if denominator > 1e-20 else 0.0,
        "normalized_rmse": float(
            math.sqrt(np.dot(left - right, left - right) / target_energy)
        ),
    }


def latent_closure_metrics(predicted, target, magnitude_count):
    """Compare predicted and encoded-true futures at three resolutions."""

    prediction = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if prediction.shape != truth.shape or prediction.ndim < 2:
        raise ValueError("predicted and target latents must have equal row shapes")
    total = vector_alignment(prediction, truth)
    centered_prediction = magnitude_center(prediction, magnitude_count)
    centered_truth = magnitude_center(truth, magnitude_count)
    centered = vector_alignment(centered_prediction, centered_truth)
    area_prediction = area_antisymmetric_component(prediction, magnitude_count)
    area_truth = area_antisymmetric_component(truth, magnitude_count)
    area = vector_alignment(area_prediction, area_truth)
    return {
        **{f"total_{key}": value for key, value in total.items()},
        **{f"centered_{key}": value for key, value in centered.items()},
        **{f"area_{key}": value for key, value in area.items()},
    }


def ideal_contrast_effect(values, magnitude_count, mode="swap"):
    """Return the ideal branchwise edit under reversal or area ablation."""

    array = np.asarray(values, dtype=np.float64)
    permutation = area_reversal_permutation(magnitude_count)
    if array.ndim < 2 or array.shape[0] != len(permutation):
        raise ValueError("values do not match the magnitude/schedule design")
    if mode == "swap":
        return array[permutation] - array
    if mode == "ablation":
        return -area_antisymmetric_component(array, magnitude_count)
    raise ValueError("mode must be 'swap' or 'ablation'")


def ideal_absolute_target(values, magnitude_count, mode="swap"):
    """Return the absolute target corresponding to an ideal contrast edit."""

    array = np.asarray(values, dtype=np.float64)
    if mode == "swap":
        return array[area_reversal_permutation(magnitude_count)]
    if mode == "ablation":
        return array - area_antisymmetric_component(array, magnitude_count)
    raise ValueError("mode must be 'swap' or 'ablation'")


def grounded_intervention_metrics(
    baseline,
    patched,
    encoded_target,
    magnitude_count,
    mode="swap",
):
    """Score one edit against self-consistent and simulator-grounded targets.

    ``self`` asks whether the intervention follows the model's own donor
    prediction.  ``grounded`` asks whether the same edit follows the contrast
    between encoder representations of the exact simulator futures.
    """

    base = np.asarray(baseline, dtype=np.float64)
    edit = np.asarray(patched, dtype=np.float64)
    truth = np.asarray(encoded_target, dtype=np.float64)
    if base.shape != edit.shape or base.shape != truth.shape:
        raise ValueError("baseline, patched, and encoded_target shapes must match")
    effect = edit - base
    self_ideal = ideal_contrast_effect(base, magnitude_count, mode=mode)
    grounded_ideal = ideal_contrast_effect(truth, magnitude_count, mode=mode)
    self_metrics = vector_alignment(effect, self_ideal)
    grounded_metrics = vector_alignment(effect, grounded_ideal)
    absolute_target = ideal_absolute_target(truth, magnitude_count, mode=mode)
    before_error = float(np.sum((base - absolute_target) ** 2))
    after_error = float(np.sum((edit - absolute_target) ** 2))
    return {
        "effect_energy": float(np.sum(effect**2)),
        **{f"self_{key}": value for key, value in self_metrics.items()},
        **{f"grounded_{key}": value for key, value in grounded_metrics.items()},
        "self_minus_grounded_cosine": float(
            self_metrics["cosine"] - grounded_metrics["cosine"]
        )
        if np.isfinite(self_metrics["cosine"])
        and np.isfinite(grounded_metrics["cosine"])
        else math.nan,
        "absolute_target_error_before": before_error,
        "absolute_target_error_after": after_error,
        "absolute_target_error_reduction": float(1.0 - after_error / before_error)
        if before_error > 1e-20
        else math.nan,
    }

