"""Numerical primitives for the Stage 27 finite-action commutator test.

The experiment compares two action sequences containing the same pulses in
opposite temporal order.  The resulting finite commutator isolates order-
sensitive dynamics without differentiating the model or simulator.  This
module is deliberately NumPy-only so that the action design and causal
estimands can be tested on CPU independently of the Colab runtime.
"""

from __future__ import annotations

import math

import numpy as np

from .stage17_action_contrast import action_swap_delta, donor_transfer_metrics


CONTACT_PAIR_ANGLES_DEGREES = (
    (-30.0, 30.0),
    (-45.0, 15.0),
    (-15.0, 45.0),
    (-60.0, 60.0),
)
FREE_PAIR_ANGLES_DEGREES = (
    (150.0, 210.0),
    (135.0, 225.0),
)
PAIR_ANGLES_DEGREES = CONTACT_PAIR_ANGLES_DEGREES + FREE_PAIR_ANGLES_DEGREES
PAIR_DESIGN_LABELS = (
    "contact_symmetric_30",
    "contact_left_biased",
    "contact_right_biased",
    "contact_symmetric_60",
    "free_symmetric_30",
    "free_symmetric_45",
)


def rotate_vector(vector, angle):
    """Rotate a finite two-vector by ``angle`` radians."""

    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (2,) or not np.all(np.isfinite(value)):
        raise ValueError("vector must be a finite two-vector")
    cosine, sine = np.cos(float(angle)), np.sin(float(angle))
    return np.asarray(
        [
            cosine * value[0] - sine * value[1],
            sine * value[0] + cosine * value[1],
        ],
        dtype=np.float64,
    )


def pair_swap_permutation(pair_count):
    """Return the permutation that exchanges the two orders in every pair."""

    pair_count = int(pair_count)
    if pair_count < 1:
        raise ValueError("pair_count must be positive")
    permutation = np.arange(2 * pair_count, dtype=np.int64)
    permutation[0::2] += 1
    permutation[1::2] -= 1
    return permutation


def ordered_pulse_bank(
    toward_block,
    steps=15,
    pulse_steps=5,
    magnitude=0.18,
    angle_pairs_degrees=PAIR_ANGLES_DEGREES,
):
    """Construct same-pulse action pairs in opposite temporal order.

    Every unordered direction pair ``(u, v)`` yields two sequences:
    ``u`` for five steps then ``v`` for five steps, and the reverse.  The last
    five steps are zero.  Thus paired sequences have exactly equal integrated
    impulse, control energy, and active duration; only temporal order differs.
    """

    direction = np.asarray(toward_block, dtype=np.float64)
    if direction.shape != (2,) or not np.all(np.isfinite(direction)):
        raise ValueError("toward_block must be a finite two-vector")
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("toward_block is degenerate")
    direction /= norm
    steps = int(steps)
    pulse_steps = int(pulse_steps)
    magnitude = float(magnitude)
    if steps != 15 or pulse_steps != 5:
        raise ValueError("Stage 27 is frozen to two five-step pulses in 15 steps")
    if not np.isfinite(magnitude) or magnitude <= 0:
        raise ValueError("magnitude must be positive")

    branches = []
    for left_degrees, right_degrees in angle_pairs_degrees:
        left = rotate_vector(direction, np.deg2rad(float(left_degrees)))
        right = rotate_vector(direction, np.deg2rad(float(right_degrees)))
        left_then_right = np.zeros((steps, 2), dtype=np.float64)
        right_then_left = np.zeros((steps, 2), dtype=np.float64)
        left_then_right[:pulse_steps] = magnitude * left
        left_then_right[pulse_steps : 2 * pulse_steps] = magnitude * right
        right_then_left[:pulse_steps] = magnitude * right
        right_then_left[pulse_steps : 2 * pulse_steps] = magnitude * left
        branches.extend([left_then_right, right_then_left])

    actions = np.stack(branches).astype(np.float32)
    expected_shape = (2 * len(angle_pairs_degrees), steps, 2)
    if actions.shape != expected_shape:
        raise RuntimeError(f"bad Stage 27 action-bank shape {actions.shape}")
    for pair_index in range(len(angle_pairs_degrees)):
        first, second = actions[2 * pair_index : 2 * pair_index + 2]
        if not np.allclose(np.sum(first, axis=0), np.sum(second, axis=0), atol=1e-7):
            raise RuntimeError("paired sequences lost equal integrated impulse")
        if not np.isclose(np.sum(first**2), np.sum(second**2), atol=1e-7):
            raise RuntimeError("paired sequences lost equal control energy")
        if not np.allclose(first[2 * pulse_steps :], 0.0, atol=0.0):
            raise RuntimeError("action tail must be exactly zero")
    return actions


def paired_antisymmetric_component(values):
    """Return the order-sensitive half-difference inside every action pair."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 2 or array.shape[0] < 2 or array.shape[0] % 2:
        raise ValueError("values must contain an even number of candidate rows")
    original_shape = array.shape
    flat = array.reshape(array.shape[0], -1)
    pairs = flat.reshape(-1, 2, flat.shape[1])
    half_difference = 0.5 * (pairs[:, 0] - pairs[:, 1])
    antisymmetric = np.empty_like(pairs)
    antisymmetric[:, 0] = half_difference
    antisymmetric[:, 1] = -half_difference
    return antisymmetric.reshape(original_shape)


def paired_swap_delta(values, basis=None, dose=1.0):
    """Swap the two temporal orders, optionally within a frozen subspace."""

    array = np.asarray(values, dtype=np.float64)
    if array.shape[0] % 2:
        raise ValueError("paired swap requires an even number of rows")
    return action_swap_delta(
        array,
        pair_swap_permutation(array.shape[0] // 2),
        basis=basis,
        dose=dose,
    )


def paired_ablation_delta(values, basis, dose=1.0):
    """Erase only the projected within-pair order contrast.

    Unlike ordinary candidate-centered ablation, this leaves every pair mean
    and all pair-to-pair differences unchanged.  It therefore tests necessity
    of the order-sensitive component rather than generic action information.
    """

    array = np.asarray(values, dtype=np.float64)
    original_shape = array.shape
    flat = paired_antisymmetric_component(array).reshape(array.shape[0], -1)
    directions = np.asarray(basis, dtype=np.float64)
    if directions.ndim != 2 or directions.shape[0] != flat.shape[1]:
        raise ValueError("basis does not match flattened activation width")
    if not np.allclose(
        directions.T @ directions,
        np.eye(directions.shape[1]),
        atol=1e-6,
        rtol=1e-6,
    ):
        raise ValueError("basis columns must be orthonormal")
    projected = (flat @ directions) @ directions.T
    return (-float(dose) * projected).reshape(original_shape)


def _pair_rows(pair_mask, pair_count):
    if pair_mask is None:
        mask = np.ones(int(pair_count), dtype=bool)
    else:
        mask = np.asarray(pair_mask, dtype=bool)
        if mask.shape != (int(pair_count),):
            raise ValueError("pair mask has the wrong shape")
    pair_ids = np.flatnonzero(mask)
    if not len(pair_ids):
        return np.asarray([], dtype=np.int64)
    return np.stack([2 * pair_ids, 2 * pair_ids + 1], axis=1).reshape(-1)


def paired_transfer_metrics(baseline, patched, pair_mask=None):
    """Measure causal transfer toward the opposite temporal order."""

    base = np.asarray(baseline, dtype=np.float64)
    edit = np.asarray(patched, dtype=np.float64)
    if base.shape != edit.shape or base.shape[0] % 2:
        raise ValueError("paired transfer inputs have inconsistent shapes")
    rows = _pair_rows(pair_mask, base.shape[0] // 2)
    if not len(rows):
        return {
            "target_energy": 0.0,
            "coefficient": math.nan,
            "cosine": math.nan,
            "reconstruction": math.nan,
            "mean_shift_ratio": math.nan,
        }
    selected_base = base[rows]
    selected_edit = edit[rows]
    return donor_transfer_metrics(
        selected_base,
        selected_edit,
        pair_swap_permutation(len(rows) // 2),
    )


def paired_energy_metrics(baseline, patched, pair_mask=None):
    """Measure retention of the order-sensitive output contrast."""

    base = np.asarray(baseline, dtype=np.float64)
    edit = np.asarray(patched, dtype=np.float64)
    if base.shape != edit.shape or base.shape[0] % 2:
        raise ValueError("paired energy inputs have inconsistent shapes")
    rows = _pair_rows(pair_mask, base.shape[0] // 2)
    if not len(rows):
        return {
            "baseline_energy": 0.0,
            "patched_energy": 0.0,
            "energy_retention": math.nan,
            "energy_reduction": math.nan,
            "contrast_cosine": math.nan,
        }
    base_component = paired_antisymmetric_component(base[rows]).reshape(len(rows), -1)
    edit_component = paired_antisymmetric_component(edit[rows]).reshape(len(rows), -1)
    baseline_energy = float(np.sum(base_component**2))
    patched_energy = float(np.sum(edit_component**2))
    if baseline_energy <= 1e-12:
        return {
            "baseline_energy": baseline_energy,
            "patched_energy": patched_energy,
            "energy_retention": math.nan,
            "energy_reduction": math.nan,
            "contrast_cosine": math.nan,
        }
    denominator = math.sqrt(baseline_energy * patched_energy)
    cosine = (
        float(np.sum(base_component * edit_component) / denominator)
        if denominator > 1e-12
        else 0.0
    )
    return {
        "baseline_energy": baseline_energy,
        "patched_energy": patched_energy,
        "energy_retention": float(patched_energy / baseline_energy),
        "energy_reduction": float(1.0 - patched_energy / baseline_energy),
        "contrast_cosine": cosine,
    }


def pair_contact_masks(interaction_counts):
    """Classify action-order pairs by simulator contact in either ordering."""

    counts = np.asarray(interaction_counts, dtype=np.int64)
    if counts.ndim != 1 or len(counts) < 2 or len(counts) % 2:
        raise ValueError("interaction_counts must have an even number of rows")
    pairs = counts.reshape(-1, 2)
    first = pairs[:, 0] > 0
    second = pairs[:, 1] > 0
    return {
        "contact": first | second,
        "both_contact": first & second,
        "one_contact": first ^ second,
        "free": ~(first | second),
    }


def commutator_contrasts(values):
    """Return first-order-minus-second-order contrasts for every pair."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 2 or array.shape[0] < 2 or array.shape[0] % 2:
        raise ValueError("values must contain paired candidate rows")
    flat = array.reshape(array.shape[0], -1).reshape(-1, 2, int(np.prod(array.shape[1:])))
    return flat[:, 0] - flat[:, 1]


def commutator_norms(values):
    """Return one Euclidean finite-commutator norm per pair."""

    return np.linalg.norm(commutator_contrasts(values), axis=1)


def commutator_alignment_metrics(predicted, truth, pair_mask=None):
    """Align predicted finite commutators with exact simulator commutators."""

    predicted_contrast = commutator_contrasts(predicted)
    true_contrast = commutator_contrasts(truth)
    if predicted_contrast.shape != true_contrast.shape:
        raise ValueError("predicted and true commutators must have equal shape")
    if pair_mask is None:
        mask = np.ones(len(true_contrast), dtype=bool)
    else:
        mask = np.asarray(pair_mask, dtype=bool)
        if mask.shape != (len(true_contrast),):
            raise ValueError("pair mask has the wrong shape")
    predicted_selected = predicted_contrast[mask].reshape(-1)
    true_selected = true_contrast[mask].reshape(-1)
    target_energy = float(np.sum(true_selected**2))
    observed_energy = float(np.sum(predicted_selected**2))
    if target_energy <= 1e-12:
        return {
            "target_energy": target_energy,
            "predicted_energy": observed_energy,
            "coefficient": math.nan,
            "cosine": math.nan,
            "normalized_rmse": math.nan,
            "pairs": int(np.sum(mask)),
        }
    dot = float(np.sum(predicted_selected * true_selected))
    cosine_denominator = math.sqrt(target_energy * observed_energy)
    return {
        "target_energy": target_energy,
        "predicted_energy": observed_energy,
        "coefficient": float(dot / target_energy),
        "cosine": float(dot / cosine_denominator) if cosine_denominator > 1e-12 else 0.0,
        "normalized_rmse": float(
            math.sqrt(np.sum((predicted_selected - true_selected) ** 2) / target_energy)
        ),
        "pairs": int(np.sum(mask)),
    }
