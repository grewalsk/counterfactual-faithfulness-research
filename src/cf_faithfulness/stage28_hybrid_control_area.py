"""Numerical primitives for the Stage 28 hybrid control-area experiment.

The experiment holds the multiset of five ``u`` pulses, five ``v`` pulses,
and five zero actions fixed while changing their temporal ordering.  Six
frozen schedules span symmetric signed discrete-control-area levels.  Repeating
the design at several magnitudes separates the second-order smooth-control law
from contact-boundary switching without differentiating either the simulator
or the world model.
"""

from __future__ import annotations

import math

import numpy as np

from .stage17_action_contrast import action_swap_delta, donor_transfer_metrics
from .stage27_action_commutator import rotate_vector


SCHEDULE_STRINGS = (
    "uuuuuvvvvv",
    "uuvuuvvuvv",
    "uvuvuvuvuv",
    "vuvuvuvuvu",
    "vvuvvuuvuu",
    "vvvvvuuuuu",
)
SCHEDULE_INVERSION_COUNTS = (0, 5, 10, 15, 20, 25)
SIGNED_AREA_LEVELS = (25, 15, 5, -5, -15, -25)


def schedule_inversion_count(schedule):
    """Count ``v``-before-``u`` pairs in a two-letter schedule."""

    tokens = tuple(str(value).lower() for value in schedule)
    if len(tokens) != 10 or tokens.count("u") != 5 or tokens.count("v") != 5:
        raise ValueError("a schedule must contain exactly five u and five v tokens")
    if any(value not in {"u", "v"} for value in tokens):
        raise ValueError("a schedule may contain only u and v tokens")
    return int(
        sum(
            left == "v" and right == "u"
            for index, left in enumerate(tokens)
            for right in tokens[index + 1 :]
        )
    )


def signed_control_area(actions):
    """Return half the pairwise determinant sum of a two-dimensional path."""

    path = np.asarray(actions, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or not np.all(np.isfinite(path)):
        raise ValueError("actions must be a finite [steps, 2] array")
    cumulative = np.zeros(2, dtype=np.float64)
    twice_area = 0.0
    for action in path:
        twice_area += cumulative[0] * action[1] - cumulative[1] * action[0]
        cumulative += action
    return float(0.5 * twice_area)


def area_reversal_permutation(magnitude_count, schedule_count=6):
    """Map every schedule to the schedule with opposite signed area."""

    magnitude_count = int(magnitude_count)
    schedule_count = int(schedule_count)
    if magnitude_count < 1 or schedule_count != len(SCHEDULE_STRINGS):
        raise ValueError("Stage 28 requires at least one magnitude and six schedules")
    rows = np.arange(magnitude_count * schedule_count, dtype=np.int64).reshape(
        magnitude_count, schedule_count
    )
    return rows[:, ::-1].reshape(-1)


def area_action_bank(
    toward_block,
    magnitudes,
    steps=15,
    angle_pair_degrees=(-30.0, 30.0),
    schedules=SCHEDULE_STRINGS,
):
    """Construct a fixed-multiset action bank spanning area and magnitude.

    Within each magnitude, every row contains identical counts of the same
    ``u``, ``v``, and zero pulses.  Consequently impulse, energy, duration,
    and action histogram are exact controls; only schedule area changes.
    """

    direction = np.asarray(toward_block, dtype=np.float64)
    if direction.shape != (2,) or not np.all(np.isfinite(direction)):
        raise ValueError("toward_block must be a finite two-vector")
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("toward_block is degenerate")
    direction /= norm
    if int(steps) != 15:
        raise ValueError("Stage 28 is frozen to fifteen action steps")
    magnitudes = np.asarray(magnitudes, dtype=np.float64)
    if (
        magnitudes.ndim != 1
        or len(magnitudes) < 2
        or not np.all(np.isfinite(magnitudes))
        or np.any(magnitudes <= 0)
        or np.any(np.diff(magnitudes) <= 0)
    ):
        raise ValueError("magnitudes must be a strictly increasing positive vector")
    if len(schedules) != 6:
        raise ValueError("Stage 28 requires six frozen schedules")

    left_degrees, right_degrees = map(float, angle_pair_degrees)
    u_direction = rotate_vector(direction, np.deg2rad(left_degrees))
    v_direction = rotate_vector(direction, np.deg2rad(right_degrees))
    branches = []
    for magnitude in magnitudes:
        u = float(magnitude) * u_direction
        v = float(magnitude) * v_direction
        for schedule in schedules:
            if schedule_inversion_count(schedule) not in SCHEDULE_INVERSION_COUNTS:
                raise ValueError("schedule is outside the frozen inversion design")
            path = np.zeros((steps, 2), dtype=np.float64)
            for index, token in enumerate(schedule):
                path[index] = u if token == "u" else v
            branches.append(path)
    actions = np.stack(branches).astype(np.float32)

    schedule_count = len(schedules)
    for magnitude_index in range(len(magnitudes)):
        group = actions[
            magnitude_index * schedule_count : (magnitude_index + 1) * schedule_count
        ]
        impulses = np.sum(group, axis=1)
        energies = np.sum(group**2, axis=(1, 2))
        active = np.sum(np.linalg.norm(group, axis=2) > 0, axis=1)
        if not np.allclose(impulses, impulses[0], atol=5e-7, rtol=0):
            raise RuntimeError("within-magnitude schedules lost equal impulse")
        if not np.allclose(energies, energies[0], atol=5e-7, rtol=0):
            raise RuntimeError("within-magnitude schedules lost equal energy")
        if not np.array_equal(active, np.full(schedule_count, 10)):
            raise RuntimeError("within-magnitude schedules lost equal active duration")
        if not np.allclose(group[:, 10:], 0.0, atol=0.0, rtol=0):
            raise RuntimeError("the final five actions must be zero")
        areas = np.asarray([signed_control_area(value) for value in group])
        if not np.all(np.diff(areas) < 0):
            raise RuntimeError("signed control areas must decrease across schedules")
        if not np.allclose(areas, -areas[::-1], atol=1e-8, rtol=1e-6):
            raise RuntimeError("signed control-area levels are not antisymmetric")
    return actions


def area_antisymmetric_component(values, magnitude_count):
    """Return the component that reverses sign under schedule-area reversal."""

    array = np.asarray(values, dtype=np.float64)
    permutation = area_reversal_permutation(magnitude_count)
    if array.ndim < 2 or array.shape[0] != len(permutation):
        raise ValueError("values do not match the magnitude/schedule design")
    return 0.5 * (array - array[permutation])


def area_swap_delta(values, magnitude_count, basis=None, dose=1.0):
    """Reverse schedule area, optionally only inside a frozen subspace."""

    array = np.asarray(values, dtype=np.float64)
    permutation = area_reversal_permutation(magnitude_count)
    if array.shape[0] != len(permutation):
        raise ValueError("values do not match the magnitude/schedule design")
    return action_swap_delta(array, permutation, basis=basis, dose=dose)


def area_ablation_delta(values, magnitude_count, basis, dose=1.0):
    """Erase only the subspace-projected area-antisymmetric component."""

    array = np.asarray(values, dtype=np.float64)
    flat = area_antisymmetric_component(array, magnitude_count).reshape(
        array.shape[0], -1
    )
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
    return (-float(dose) * projected).reshape(array.shape)


def area_transfer_metrics(baseline, patched, magnitude_count):
    """Measure causal transfer toward outcomes with opposite schedule area."""

    permutation = area_reversal_permutation(magnitude_count)
    return donor_transfer_metrics(baseline, patched, permutation)


def area_energy_metrics(baseline, patched, magnitude_count):
    """Measure retention of the area-antisymmetric output component."""

    base = np.asarray(baseline, dtype=np.float64)
    edit = np.asarray(patched, dtype=np.float64)
    if base.shape != edit.shape:
        raise ValueError("baseline and patched values must have equal shape")
    base_component = area_antisymmetric_component(base, magnitude_count).reshape(
        len(base), -1
    )
    edit_component = area_antisymmetric_component(edit, magnitude_count).reshape(
        len(edit), -1
    )
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
    return {
        "baseline_energy": baseline_energy,
        "patched_energy": patched_energy,
        "energy_retention": float(patched_energy / baseline_energy),
        "energy_reduction": float(1.0 - patched_energy / baseline_energy),
        "contrast_cosine": (
            float(np.sum(base_component * edit_component) / denominator)
            if denominator > 1e-12
            else 0.0
        ),
    }


def contact_regime(interaction_counts):
    """Classify one state as persistent-contact, boundary-switching, or free."""

    counts = np.asarray(interaction_counts, dtype=np.int64)
    if counts.ndim != 2 or counts.shape[1] != len(SCHEDULE_STRINGS):
        raise ValueError("interaction counts must be [magnitudes, six schedules]")
    contact = counts > 0
    if np.all(contact):
        return "persistent_contact"
    if not np.any(contact):
        return "free"
    return "boundary_switching"


def magnitude_center(values, magnitude_count):
    """Remove one schedule mean independently at every magnitude."""

    array = np.asarray(values, dtype=np.float64)
    magnitude_count = int(magnitude_count)
    if array.shape[0] != magnitude_count * len(SCHEDULE_STRINGS):
        raise ValueError("values do not match the magnitude/schedule design")
    grouped = array.reshape(magnitude_count, len(SCHEDULE_STRINGS), -1)
    centered = grouped - np.mean(grouped, axis=1, keepdims=True)
    return centered.reshape(array.shape)


def max_area_contrasts(values, magnitude_count):
    """Return maximum-positive minus maximum-negative area per magnitude."""

    array = np.asarray(values, dtype=np.float64)
    grouped = array.reshape(int(magnitude_count), len(SCHEDULE_STRINGS), -1)
    return grouped[:, 0] - grouped[:, -1]


def _cosine(left, right):
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else math.nan


def area_law_metrics(values, actions, magnitudes):
    """Summarize signed-area linearity and magnitude scaling for one state."""

    outputs = np.asarray(values, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    magnitudes = np.asarray(magnitudes, dtype=np.float64)
    magnitude_count = len(magnitudes)
    schedule_count = len(SCHEDULE_STRINGS)
    if outputs.shape[0] != magnitude_count * schedule_count:
        raise ValueError("outputs do not match the area design")
    if actions.shape[:2] != (len(outputs), 15):
        raise ValueError("actions do not match the area design")
    flat = outputs.reshape(len(outputs), -1)
    grouped = flat.reshape(magnitude_count, schedule_count, -1)
    areas = np.asarray([signed_control_area(value) for value in actions]).reshape(
        magnitude_count, schedule_count
    )
    centered_outputs = grouped - np.mean(grouped, axis=1, keepdims=True)
    centered_areas = areas - np.mean(areas, axis=1, keepdims=True)
    slope_numerator = np.einsum("ms,msd->md", centered_areas, centered_outputs)
    slope_denominator = np.sum(centered_areas**2, axis=1)
    slopes = slope_numerator / np.maximum(slope_denominator[:, None], 1e-12)
    fitted = centered_areas[:, :, None] * slopes[:, None, :]
    residual_energy = float(np.sum((centered_outputs - fitted) ** 2))
    total_energy = float(np.sum(centered_outputs**2))
    r_squared = 1.0 - residual_energy / total_energy if total_energy > 1e-12 else math.nan

    pairwise_cosines = []
    for left in range(magnitude_count):
        for right in range(left + 1, magnitude_count):
            value = _cosine(slopes[left], slopes[right])
            if np.isfinite(value):
                pairwise_cosines.append(value)
    contrasts = max_area_contrasts(flat, magnitude_count)
    contrast_norms = np.linalg.norm(contrasts, axis=1)
    valid = contrast_norms > 1e-12
    exponent = (
        float(np.polyfit(np.log(magnitudes[valid]), np.log(contrast_norms[valid]), 1)[0])
        if np.sum(valid) >= 3
        else math.nan
    )
    normalized = contrasts / np.maximum(magnitudes[:, None] ** 2, 1e-12)
    normalized_mean = np.mean(normalized, axis=0)
    collapse_error = float(
        np.sqrt(np.mean((normalized - normalized_mean) ** 2))
        / max(np.sqrt(np.mean(normalized_mean**2)), 1e-12)
    )
    return {
        "area_r_squared": float(r_squared),
        "mean_slope_direction_cosine": (
            float(np.mean(pairwise_cosines)) if pairwise_cosines else math.nan
        ),
        "magnitude_exponent": exponent,
        "epsilon_squared_collapse_error": collapse_error,
        "max_area_contrast_norms": contrast_norms.tolist(),
        "slope_norms": np.linalg.norm(slopes, axis=1).tolist(),
    }


def model_physics_area_metrics(predicted, truth, magnitude_count):
    """Align model and simulator schedule effects after magnitude centering."""

    predicted_centered = magnitude_center(predicted, magnitude_count).reshape(-1)
    truth_centered = magnitude_center(truth, magnitude_count).reshape(-1)
    truth_energy = float(np.sum(truth_centered**2))
    predicted_energy = float(np.sum(predicted_centered**2))
    if truth_energy <= 1e-12:
        return {
            "target_energy": truth_energy,
            "predicted_energy": predicted_energy,
            "coefficient": math.nan,
            "cosine": math.nan,
            "normalized_rmse": math.nan,
        }
    dot = float(np.dot(predicted_centered, truth_centered))
    return {
        "target_energy": truth_energy,
        "predicted_energy": predicted_energy,
        "coefficient": float(dot / truth_energy),
        "cosine": (
            float(dot / math.sqrt(truth_energy * predicted_energy))
            if predicted_energy > 1e-12
            else 0.0
        ),
        "normalized_rmse": float(
            math.sqrt(np.sum((predicted_centered - truth_centered) ** 2) / truth_energy)
        ),
    }


for _schedule, _inversions, _area_level in zip(
    SCHEDULE_STRINGS, SCHEDULE_INVERSION_COUNTS, SIGNED_AREA_LEVELS
):
    if schedule_inversion_count(_schedule) != _inversions:
        raise RuntimeError("frozen schedule inversion count is inconsistent")
    if 25 - 2 * _inversions != _area_level:
        raise RuntimeError("frozen signed-area level is inconsistent")
