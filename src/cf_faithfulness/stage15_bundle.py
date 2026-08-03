"""Numerical primitives for the Stage 15 longitudinal bundle pilot.

The model-facing notebook owns simulator execution, JVP/VJP extraction, and
causal hooks.  This module deliberately stays NumPy-only so the geometry,
reader fitting, action coordinates, and transport statistics can be tested on
CPU before any accelerator is used.
"""

from __future__ import annotations

import math

import numpy as np


PHYSICAL_READER_LABELS = (
    "agent_x",
    "agent_y",
    "block_x",
    "block_y",
    "block_sin",
    "block_cos",
)


def physical_reader_targets(states):
    """Map PushT states to six fixed, goal-independent physical coordinates."""
    values = np.asarray(states, dtype=np.float64)
    if values.shape[-1] < 5:
        raise ValueError("PushT states must contain at least five coordinates")
    angle = values[..., 4]
    return np.stack(
        [
            values[..., 0] / 512.0,
            values[..., 1] / 512.0,
            values[..., 2] / 512.0,
            values[..., 3] / 512.0,
            np.sin(angle),
            np.cos(angle),
        ],
        axis=-1,
    )


def temporal_action_basis(steps, profiles=3):
    """Return one fixed orthonormal DCT-like action basis for every state.

    Actions are flattened in ``[time, xy]`` order.  The same columns can be
    used at every physical state because they describe perturbations in the
    normalized executable-action coordinates, not state-relative endpoints.
    """
    steps = int(steps)
    profiles = int(profiles)
    if steps < 1 or profiles < 1:
        raise ValueError("steps and profiles must be positive")
    profiles = min(profiles, steps)
    time = np.arange(steps, dtype=np.float64)
    temporal = []
    for frequency in range(profiles):
        column = np.cos(math.pi * (time + 0.5) * frequency / steps)
        column /= np.linalg.norm(column)
        temporal.append(column)
    columns = []
    for column in temporal:
        for axis in range(2):
            value = np.zeros((steps, 2), dtype=np.float64)
            value[:, axis] = column
            columns.append(value.reshape(-1))
    basis = np.stack(columns, axis=1)
    gram = basis.T @ basis
    if not np.allclose(gram, np.eye(len(columns)), atol=1e-12, rtol=0):
        raise AssertionError("temporal action basis is not orthonormal")
    return basis


def fit_ridge(features, targets, ridge):
    """Fit a standardized multi-output ridge model with an explicit intercept."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
        raise ValueError("features and targets must be aligned nontrivial matrices")
    if float(ridge) < 0:
        raise ValueError("ridge must be nonnegative")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    standardized = (x - mean) / scale
    intercept = y.mean(axis=0)
    centered = y - intercept
    gram = standardized.T @ standardized
    coefficient = np.linalg.solve(
        gram + float(ridge) * np.eye(gram.shape[0]),
        standardized.T @ centered,
    )
    return {
        "feature_mean": mean,
        "feature_scale": scale,
        "intercept": intercept,
        "coefficient": coefficient,
        "ridge": float(ridge),
    }


def predict_ridge(model, features):
    x = np.asarray(features, dtype=np.float64)
    standardized = (
        x - np.asarray(model["feature_mean"], dtype=np.float64)
    ) / np.asarray(model["feature_scale"], dtype=np.float64)
    return (
        standardized @ np.asarray(model["coefficient"], dtype=np.float64)
        + np.asarray(model["intercept"], dtype=np.float64)
    )


def grouped_ridge_cv(features, targets, groups, ridges):
    """Select ridge strength by leave-one-group-out construction-only CV."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    group_values = np.asarray(groups)
    ridge_values = tuple(float(value) for value in ridges)
    unique = np.unique(group_values)
    if len(x) != len(y) or len(x) != len(group_values):
        raise ValueError("features, targets, and groups must align")
    if len(unique) < 2 or not ridge_values:
        raise ValueError("grouped CV needs at least two groups and one ridge")
    rows = []
    for ridge in ridge_values:
        losses = []
        for held_out in unique:
            train = group_values != held_out
            validation = ~train
            model = fit_ridge(x[train], y[train], ridge)
            prediction = predict_ridge(model, x[validation])
            loss = float(np.mean((prediction - y[validation]) ** 2))
            losses.append(loss)
            rows.append(
                {
                    "ridge": ridge,
                    "held_out_group": int(held_out),
                    "mse": loss,
                }
            )
        rows.append(
            {
                "ridge": ridge,
                "held_out_group": "mean",
                "mse": float(np.mean(losses)),
            }
        )
    means = {
        row["ridge"]: row["mse"]
        for row in rows
        if row["held_out_group"] == "mean"
    }
    selected = min(ridge_values, key=lambda value: (means[value], value))
    return {
        "model": fit_ridge(x, y, selected),
        "selected_ridge": selected,
        "cv_rows": rows,
    }


def r2_per_output(target, prediction):
    truth = np.asarray(target, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    if truth.shape != estimate.shape or truth.ndim != 2:
        raise ValueError("target and prediction must be matching matrices")
    residual = np.sum((truth - estimate) ** 2, axis=0)
    centered = np.sum((truth - truth.mean(axis=0)) ** 2, axis=0)
    return 1.0 - residual / np.maximum(centered, 1e-12)


def orthonormal_columns(values, tolerance=1e-9, rank=None):
    """Return a deterministic orthonormal basis for a matrix's column span."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("values must be a matrix")
    if not matrix.size:
        return np.empty((matrix.shape[0], 0), dtype=np.float64)
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    keep = singular > max(float(singular[0]) * float(tolerance), 1e-12)
    available = int(np.sum(keep))
    if rank is not None:
        available = min(available, int(rank))
    return u[:, :available]


def principal_angle_cosines(left, right, rank=None):
    a = orthonormal_columns(left, rank=rank)
    b = orthonormal_columns(right, rank=rank)
    retained = min(a.shape[1], b.shape[1])
    if retained == 0:
        return np.empty(0, dtype=np.float64)
    return np.linalg.svd(a.T @ b, compute_uv=False)[:retained]


def chordal_subspace_distance(left, right, rank=None):
    cosines = principal_angle_cosines(left, right, rank=rank)
    if not len(cosines):
        return math.nan
    return float(np.sqrt(np.mean(np.maximum(1.0 - cosines**2, 0.0))))


def procrustes_align(reference, candidate):
    """Rotate ``candidate`` coordinates to align with ``reference``.

    Both matrices are ambient-by-mode bases with the same number of columns.
    The returned rotation changes only the mode coordinates; it cannot improve
    the underlying subspace overlap.
    """
    target = np.asarray(reference, dtype=np.float64)
    source = np.asarray(candidate, dtype=np.float64)
    if target.ndim != 2 or source.shape != target.shape or not target.shape[1]:
        raise ValueError("reference and candidate must be equal nonempty bases")
    cross = source.T @ target
    u, _, vh = np.linalg.svd(cross, full_matrices=False)
    rotation = u @ vh
    aligned = source @ rotation
    return {
        "aligned": aligned,
        "rotation": rotation,
        "relative_error": float(
            np.linalg.norm(aligned - target)
            / max(np.linalg.norm(target), 1e-12)
        ),
    }


def align_basis_sequence(bases, rank=None):
    """Align a sequence recursively without changing any sampled subspace."""
    if not bases:
        return []
    orthonormal = [orthonormal_columns(value, rank=rank) for value in bases]
    retained = min(value.shape[1] for value in orthonormal)
    if retained == 0:
        return [value[:, :0] for value in orthonormal]
    aligned = [orthonormal[0][:, :retained]]
    for value in orthonormal[1:]:
        result = procrustes_align(aligned[-1], value[:, :retained])
        aligned.append(result["aligned"])
    return aligned


def matrix_cosine(left, right):
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / max(float(denominator), 1e-12))


def normalized_matrix_distance(left, right):
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("matrices must have matching shapes")
    return float(
        np.linalg.norm(a - b)
        / max(0.5 * (np.linalg.norm(a) + np.linalg.norm(b)), 1e-12)
    )


def _average_ranks(values):
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman_correlation(left, right):
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2:
        raise ValueError("inputs must be aligned nontrivial vectors")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("inputs must be finite")
    ra = _average_ranks(a)
    rb = _average_ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denominator = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(np.dot(ra, rb) / max(float(denominator), 1e-12))


def grouped_label_permutation(labels, groups, seed):
    """Permute time labels independently within each trajectory group."""
    values = np.asarray(labels).copy()
    group_values = np.asarray(groups)
    if len(values) != len(group_values):
        raise ValueError("labels and groups must align")
    rng = np.random.default_rng(int(seed))
    output = values.copy()
    for group in np.unique(group_values):
        indices = np.flatnonzero(group_values == group)
        output[indices] = rng.permutation(values[indices])
    return output


def energy_map(values, tokens=256, channels=400):
    """Return normalized per-token energy for one or more carrier vectors."""
    array = np.asarray(values, dtype=np.float64)
    reshaped = array.reshape(-1, int(tokens), int(channels))
    energy = np.sum(reshaped**2, axis=(0, 2))
    return energy / max(float(np.sum(energy)), 1e-12)


def support_matched_random(template, seed, tokens=256, channels=400):
    """Randomize channel orientation while preserving exact token energies."""
    value = np.asarray(template, dtype=np.float64).reshape(
        int(tokens), int(channels)
    )
    rng = np.random.default_rng(int(seed))
    random = rng.normal(size=value.shape)
    random /= np.maximum(np.linalg.norm(random, axis=1, keepdims=True), 1e-12)
    random *= np.linalg.norm(value, axis=1, keepdims=True)
    result = random.reshape(-1)
    if not np.allclose(
        energy_map(result, tokens=tokens, channels=channels),
        energy_map(value, tokens=tokens, channels=channels),
        atol=1e-12,
        rtol=1e-10,
    ):
        raise AssertionError("support-matched null changed token energies")
    return result
