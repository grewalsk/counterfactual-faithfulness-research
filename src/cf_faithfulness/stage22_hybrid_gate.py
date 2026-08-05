"""Numerical primitives for Stage 22 hybrid-gate discovery.

The Stage 22 notebook remains responsible for simulator execution, model
hooks, and GPU linear algebra.  This module contains the NumPy-only pieces
needed to discover a two-mode partition without physical labels and to score
finite gate-by-effect interchange interventions.
"""

from __future__ import annotations

import math

import numpy as np

from .stage17_action_contrast import candidate_center


def center_by_group(values, groups):
    """Center rows independently within each prespecified state group."""

    array = np.asarray(values, dtype=np.float64)
    labels = np.asarray(groups)
    if array.ndim != 2 or labels.ndim != 1 or len(array) != len(labels):
        raise ValueError("values and groups must be aligned row arrays")
    result = np.empty_like(array)
    for group in np.unique(labels):
        mask = labels == group
        if np.sum(mask) < 2:
            raise ValueError("each group must contain at least two candidates")
        result[mask] = candidate_center(array[mask])
    return result


def deterministic_two_means(values, seed, max_iterations=200):
    """Fit a deterministic two-means partition with a farthest-point start."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or len(array) < 4 or array.shape[1] < 1:
        raise ValueError("two-means requires at least four finite row vectors")
    if not np.all(np.isfinite(array)):
        raise ValueError("two-means input contains nonfinite values")
    rng = np.random.default_rng(int(seed))
    first = int(rng.integers(0, len(array)))
    distance = np.sum((array - array[first]) ** 2, axis=1)
    second = int(np.argmax(distance))
    if second == first or distance[second] <= 1e-12:
        raise ValueError("two-means input is degenerate")
    centroids = np.stack([array[first], array[second]])
    assignments = np.zeros(len(array), dtype=np.int64)
    for _ in range(int(max_iterations)):
        distances = np.sum(
            (array[:, None, :] - centroids[None, :, :]) ** 2, axis=2
        )
        updated = np.argmin(distances, axis=1).astype(np.int64)
        counts = np.bincount(updated, minlength=2)
        if np.any(counts == 0):
            raise ValueError("two-means produced an empty cluster")
        new_centroids = np.stack(
            [np.mean(array[updated == index], axis=0) for index in range(2)]
        )
        if np.array_equal(updated, assignments) and np.allclose(
            new_centroids, centroids, atol=1e-12, rtol=0
        ):
            centroids = new_centroids
            assignments = updated
            break
        centroids = new_centroids
        assignments = updated
    return assignments, centroids


def apply_mode_partition(values, mean, scale, components, centroids):
    """Apply one frozen standardized-PCA two-mode partition."""

    array = np.asarray(values, dtype=np.float64)
    location = np.asarray(mean, dtype=np.float64)
    spread = np.asarray(scale, dtype=np.float64)
    rotation = np.asarray(components, dtype=np.float64)
    centers = np.asarray(centroids, dtype=np.float64)
    if array.ndim != 2 or location.shape != (array.shape[1],):
        raise ValueError("partition mean does not match input width")
    if spread.shape != location.shape or rotation.ndim != 2:
        raise ValueError("partition transform is malformed")
    if rotation.shape[0] != array.shape[1] or centers.shape != (2, rotation.shape[1]):
        raise ValueError("partition components or centroids are malformed")
    transformed = ((array - location) / spread) @ rotation
    distances = np.sum(
        (transformed[:, None, :] - centers[None, :, :]) ** 2, axis=2
    )
    return np.argmin(distances, axis=1).astype(np.int64), transformed


def discover_mode_partition(
    activation_sketch,
    output_sketch,
    groups,
    *,
    seed,
    pca_rank=8,
):
    """Discover two internal modes without simulator labels.

    Activations and predicted futures are centered within state.  Clustering
    uses only activation sketches.  The cluster with greater mean predicted
    consequence energy is named ``mode_on``; no physical outcome or contact
    annotation participates in fitting or naming.
    """

    activations = center_by_group(activation_sketch, groups)
    outputs = center_by_group(output_sketch, groups)
    mean = np.mean(activations, axis=0)
    scale = np.std(activations, axis=0, ddof=1)
    positive = scale[scale > 1e-10]
    if not len(positive):
        raise ValueError("activation sketch has zero variance")
    floor = float(np.median(positive) * 1e-3)
    scale = np.maximum(scale, floor)
    standardized = (activations - mean) / scale
    _, singular, right = np.linalg.svd(standardized, full_matrices=False)
    rank = min(int(pca_rank), right.shape[0], int(np.sum(singular > 1e-9)))
    if rank < 2:
        raise ValueError("mode discovery needs at least two nonzero PCs")
    components = right[:rank].T
    reduced = standardized @ components
    assignments, centroids = deterministic_two_means(reduced, seed)

    output_energy = np.sum(outputs**2, axis=1)
    cluster_energy = np.asarray(
        [np.mean(output_energy[assignments == index]) for index in range(2)]
    )
    mode_on_cluster = int(np.argmax(cluster_energy))
    mode_on = assignments == mode_on_cluster
    counts = np.bincount(assignments, minlength=2)
    within = float(
        sum(
            np.sum((reduced[assignments == index] - centroids[index]) ** 2)
            for index in range(2)
        )
    )
    grand = np.mean(reduced, axis=0)
    between = float(
        sum(counts[index] * np.sum((centroids[index] - grand) ** 2) for index in range(2))
    )
    separation = between / max(within, 1e-12)
    balance = float(np.min(counts) / np.max(counts))
    return {
        "assignments": assignments,
        "mode_on": mode_on,
        "mode_on_cluster": mode_on_cluster,
        "mean": mean,
        "scale": scale,
        "components": components,
        "centroids": centroids,
        "singular_values": singular,
        "cluster_output_energy": cluster_energy,
        "separation": separation,
        "balance": balance,
        "counts": counts,
    }


def binary_alignment_metrics(predicted, truth):
    """Evaluate one frozen binary discovery against held-out physical labels."""

    estimate = np.asarray(predicted, dtype=bool)
    target = np.asarray(truth, dtype=bool)
    if estimate.shape != target.shape or estimate.ndim != 1 or len(estimate) < 2:
        raise ValueError("binary labels must be aligned vectors")
    tp = int(np.sum(estimate & target))
    tn = int(np.sum(~estimate & ~target))
    fp = int(np.sum(estimate & ~target))
    fn = int(np.sum(~estimate & target))
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    denominator = math.sqrt(
        max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1)
    )
    mcc = (tp * tn - fp * fn) / denominator
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float((tp + tn) / len(target)),
        "balanced_accuracy": float(0.5 * (sensitivity + specificity)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "matthews_correlation": float(mcc),
    }


def difference_of_means_direction(features, mode_on):
    """Return a unit direction from off-mode to on-mode mean activation."""

    array = np.asarray(features, dtype=np.float64)
    labels = np.asarray(mode_on, dtype=bool)
    if array.ndim != 2 or labels.shape != (len(array),):
        raise ValueError("features and mode labels must align")
    if not np.any(labels) or np.all(labels):
        raise ValueError("both modes are required")
    vector = np.mean(array[labels], axis=0) - np.mean(array[~labels], axis=0)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("mode direction is degenerate")
    return vector / norm


def orthogonalize_basis(basis, excluded):
    """Remove excluded directions and return an orthonormal basis."""

    values = np.asarray(basis, dtype=np.float64)
    removed = np.asarray(excluded, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("basis must be a matrix")
    if removed.ndim == 1:
        removed = removed[:, None]
    if removed.ndim != 2 or removed.shape[0] != values.shape[0]:
        raise ValueError("excluded directions do not match the basis")
    excluded_q, _ = np.linalg.qr(removed, mode="reduced")
    residual = values - excluded_q @ (excluded_q.T @ values)
    left, singular, _ = np.linalg.svd(residual, full_matrices=False)
    tolerance = max(float(singular.max(initial=0.0)) * 1e-9, 1e-12)
    keep = int(np.sum(singular > tolerance))
    if keep < 1:
        raise ValueError("basis is contained in excluded span")
    return left[:, :keep]


def projected_pair_delta(carrier, base_index, donor_index, basis):
    """Project one finite donor-minus-base activation difference."""

    values = np.asarray(carrier, dtype=np.float64)
    original_shape = values.shape
    flat = values.reshape(values.shape[0], -1)
    directions = np.asarray(basis, dtype=np.float64)
    if directions.ndim == 1:
        directions = directions[:, None]
    if directions.ndim != 2 or directions.shape[0] != flat.shape[1]:
        raise ValueError("basis does not match carrier width")
    base_index, donor_index = int(base_index), int(donor_index)
    if not (0 <= base_index < len(flat) and 0 <= donor_index < len(flat)):
        raise ValueError("pair index is out of range")
    difference = flat[donor_index] - flat[base_index]
    projected = directions @ (directions.T @ difference)
    delta = np.zeros_like(flat)
    delta[base_index] = projected
    return delta.reshape(original_shape)


def factorial_interaction_metrics(y00, y10, y01, y11, donor):
    """Score a gate-by-effect interchange against the donor consequence."""

    baseline = np.asarray(y00, dtype=np.float64).reshape(-1)
    gate = np.asarray(y10, dtype=np.float64).reshape(-1)
    effect = np.asarray(y01, dtype=np.float64).reshape(-1)
    both = np.asarray(y11, dtype=np.float64).reshape(-1)
    target_value = np.asarray(donor, dtype=np.float64).reshape(-1)
    if not (
        baseline.shape == gate.shape == effect.shape == both.shape == target_value.shape
    ):
        raise ValueError("factorial outputs must have identical shapes")
    target = target_value - baseline
    target_energy = float(np.sum(target**2))
    if target_energy <= 1e-12:
        raise ValueError("donor consequence is degenerate")
    interaction = both - gate - effect + baseline

    def coefficient(value):
        return float(np.sum((value - baseline) * target) / target_energy)

    interaction_energy = float(np.sum(interaction**2))
    cosine_denominator = math.sqrt(target_energy * interaction_energy)
    interaction_cosine = (
        float(np.sum(interaction * target) / cosine_denominator)
        if cosine_denominator > 1e-12
        else 0.0
    )
    return {
        "target_energy": target_energy,
        "gate_coefficient": coefficient(gate),
        "effect_coefficient": coefficient(effect),
        "both_coefficient": coefficient(both),
        "interaction_coefficient": float(np.sum(interaction * target) / target_energy),
        "interaction_cosine": interaction_cosine,
        "interaction_energy_ratio": interaction_energy / target_energy,
        "both_reconstruction": 1.0 - float(np.sum((both - target_value) ** 2)) / target_energy,
    }
