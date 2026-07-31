"""Pure NumPy primitives for the Stage 13b outcome-geometry diagnostic."""

from __future__ import annotations

import hashlib
import math

import numpy as np


ONE_SIDED_T95_DF7 = 1.894578605061305


def array_sha256(value):
    """Hash an array together with its dtype and shape."""
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def frozen_action_bank(primitive_steps=15, magnitude=0.14):
    """Return no-op plus six fixed antithetic PushT action pairs."""
    if primitive_steps != 15:
        raise ValueError("Stage 13b action design requires 15 primitive steps")
    diagonal = magnitude / np.sqrt(2.0)
    bases = []

    constant_x = np.tile([magnitude, 0.0], (primitive_steps, 1))
    constant_y = np.tile([0.0, magnitude], (primitive_steps, 1))
    diagonal_ne = np.tile([diagonal, diagonal], (primitive_steps, 1))
    diagonal_se = np.tile([diagonal, -diagonal], (primitive_steps, 1))

    turn_xy = np.empty((primitive_steps, 2), dtype=np.float64)
    turn_xy[:3] = [magnitude, 0.0]
    turn_xy[3:5] = [0.0, magnitude]
    turn_xy[5:] = [0.0, magnitude]

    three_phase = np.empty((primitive_steps, 2), dtype=np.float64)
    three_phase[:2] = [0.0, magnitude]
    three_phase[2:5] = [-magnitude, 0.0]
    three_phase[5:10] = [-magnitude, 0.0]
    three_phase[10:] = [0.0, -magnitude]

    bases.extend(
        [
            ("constant_x", constant_x),
            ("constant_y", constant_y),
            ("diagonal_ne", diagonal_ne),
            ("diagonal_se", diagonal_se),
            ("turn_xy", turn_xy),
            ("three_phase", three_phase),
        ]
    )
    actions = [np.zeros((primitive_steps, 2), dtype=np.float32)]
    labels = ["noop"]
    for label, base in bases:
        base = np.asarray(base, dtype=np.float32)
        actions.extend([base, -base])
        labels.extend([f"{label}_plus", f"{label}_minus"])
    bank = np.stack(actions)
    if bank.shape != (13, primitive_steps, 2):
        raise AssertionError(f"unexpected action shape {bank.shape}")
    for prefix in [5, 15]:
        keys = {
            np.ascontiguousarray(action[:prefix]).tobytes()
            for action in bank[1:]
        }
        if len(keys) != 12:
            raise AssertionError(f"non-noop prefix collision at step {prefix}")
    for pair_start in range(1, 13, 2):
        if not np.array_equal(bank[pair_start], -bank[pair_start + 1]):
            raise AssertionError("action pair is not exactly antithetic")
    return labels, bank


def hash_sorted_ids(ids, arrays, count, namespace):
    """Select IDs by a content-bound hash order."""
    if len(ids) != len(arrays):
        raise ValueError("ids and arrays differ in length")
    rows = []
    for identifier, value in zip(ids, arrays):
        digest = hashlib.sha256()
        digest.update(str(namespace).encode())
        digest.update(str(int(identifier)).encode())
        digest.update(array_sha256(value).encode())
        rows.append((digest.hexdigest(), int(identifier)))
    rows.sort()
    if count > len(rows):
        raise ValueError("selection count exceeds candidates")
    return [identifier for _, identifier in rows[:count]]


def effective_rank_from_gram(gram, tolerance=1e-12):
    """Compute spectral effective rank without a feature covariance."""
    eigenvalues = np.linalg.eigvalsh(np.asarray(gram, dtype=np.float64))
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = float(np.sum(eigenvalues))
    squared = float(np.sum(eigenvalues**2))
    return total**2 / squared if squared > tolerance else 0.0


def fit_dual_pca(gram, max_rank=32, relative_tolerance=1e-10):
    """Fit feature-space PCA axes through the sample Gram matrix.

    If ``X`` is the unmaterialized row-by-feature matrix, returned coefficients
    satisfy ``axes = X.T @ coefficients`` and have orthonormal columns.
    """
    gram = np.asarray(gram, dtype=np.float64)
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("gram must be square")
    gram = (gram + gram.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    threshold = (
        float(eigenvalues[0]) * relative_tolerance
        if len(eigenvalues)
        else 0.0
    )
    keep = eigenvalues > max(threshold, 1e-12)
    eigenvalues = eigenvalues[keep][:max_rank]
    eigenvectors = eigenvectors[:, keep][:, :max_rank]
    coefficients = eigenvectors / np.sqrt(eigenvalues)[None]
    return {
        "coefficients": coefficients,
        "eigenvalues": eigenvalues,
        "rank": int(len(eigenvalues)),
    }


def weighted_dual_pca(
    gram, row_weights, max_rank=32, relative_tolerance=1e-10
):
    """Fit PCA to rows weighted by a frozen query-dependent kernel."""
    gram = np.asarray(gram, dtype=np.float64)
    weights = np.asarray(row_weights, dtype=np.float64)
    if len(weights) != len(gram):
        raise ValueError("weight length does not match gram")
    if np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("weights must be nonnegative and nonzero")
    root = np.sqrt(weights)
    weighted_gram = root[:, None] * gram * root[None]
    fitted = fit_dual_pca(
        weighted_gram,
        max_rank=max_rank,
        relative_tolerance=relative_tolerance,
    )
    fitted["coefficients"] = root[:, None] * fitted["coefficients"]
    return fitted


def reconstruction_by_groups(
    cross_gram, test_gram, coefficients, ranks, group_rows
):
    """Return reconstruction fractions for each group and nested rank."""
    cross_gram = np.asarray(cross_gram, dtype=np.float64)
    test_gram = np.asarray(test_gram, dtype=np.float64)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    scores = cross_gram @ coefficients
    output = np.full((len(group_rows), len(ranks)), np.nan, dtype=np.float64)
    for group_index, rows in enumerate(group_rows):
        rows = np.asarray(rows, dtype=np.int64)
        denominator = float(np.trace(test_gram[np.ix_(rows, rows)]))
        if denominator <= 1e-12:
            continue
        for rank_index, rank in enumerate(ranks):
            used = min(int(rank), scores.shape[1])
            if used:
                output[group_index, rank_index] = float(
                    np.sum(scores[rows, :used] ** 2) / denominator
                )
    return output


def select_rank_one_se(task_scores, ranks):
    """Use the smallest rank within one SE of the best task-equal mean."""
    values = np.asarray(task_scores, dtype=np.float64)
    ranks = np.asarray(ranks, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != len(ranks):
        raise ValueError("task_scores must have one column per rank")
    means = np.nanmean(values, axis=0)
    counts = np.sum(np.isfinite(values), axis=0)
    standard_errors = np.nanstd(values, axis=0, ddof=1) / np.sqrt(counts)
    best_index = int(np.nanargmax(means))
    threshold = float(means[best_index] - standard_errors[best_index])
    eligible = np.flatnonzero(means >= threshold)
    selected_index = int(eligible[0])
    return {
        "selected_rank": int(ranks[selected_index]),
        "selected_index": selected_index,
        "best_rank": int(ranks[best_index]),
        "best_index": best_index,
        "best_mean": float(means[best_index]),
        "best_standard_error": float(standard_errors[best_index]),
        "one_se_threshold": threshold,
        "means": means,
        "standard_errors": standard_errors,
    }


def covariance_shaped_coordinates(eigenvalues, seed, max_rank):
    """Draw a covariance-shaped orthonormal null in PCA coordinates."""
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    positive = eigenvalues > 1e-12
    eigenvalues = eigenvalues[positive]
    if len(eigenvalues) < max_rank:
        raise ValueError("training covariance rank is below requested null rank")
    rng = np.random.default_rng(int(seed))
    raw = rng.normal(size=(max_rank, len(eigenvalues)))
    raw *= np.sqrt(eigenvalues)[None]
    coordinates = np.linalg.qr(raw.T, mode="reduced")[0]
    return coordinates


def basis_overlap(coefficients_a, cross_gram, coefficients_b, rank):
    """Compute normalized squared overlap of two feature-space subspaces."""
    left = np.asarray(coefficients_a, dtype=np.float64)[:, :rank]
    right = np.asarray(coefficients_b, dtype=np.float64)[:, :rank]
    cross = np.asarray(cross_gram, dtype=np.float64)
    overlap = left.T @ cross @ right
    return float(np.sum(overlap**2) / rank)


def rbf_weights(query, train_queries, bandwidth, minimum=1e-6):
    """Return normalized RBF weights based only on frozen query encodings."""
    query = np.asarray(query, dtype=np.float64)
    train = np.asarray(train_queries, dtype=np.float64)
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    squared_distance = np.sum((train - query[None]) ** 2, axis=1)
    weights = np.exp(-0.5 * squared_distance / bandwidth**2)
    weights = np.maximum(weights, minimum)
    return weights / np.mean(weights)


def one_sided_t_lower(values, critical=ONE_SIDED_T95_DF7):
    """One-sided 95% lower bound for the predeclared eight-task pilot."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) != 8:
        raise ValueError("the confirmatory t bound requires exactly eight tasks")
    return float(
        np.mean(values)
        - critical * np.std(values, ddof=1) / np.sqrt(len(values))
    )


def exact_positive_sign_test(values):
    """Return one-sided exact sign-test statistics under p=0.5."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values) & (values != 0)]
    positives = int(np.sum(values > 0))
    total = int(len(values))
    probability = sum(
        math.comb(total, successes) for successes in range(positives, total + 1)
    ) / (2**total if total else 1)
    return {
        "positive": positives,
        "nonzero": total,
        "one_sided_p": float(probability),
    }


def hierarchical_bootstrap_indices(
    task_count, states_per_task, draws, seed
):
    """Precompute task-then-state bootstrap indices."""
    rng = np.random.default_rng(int(seed))
    tasks = rng.integers(
        0, task_count, size=(draws, task_count), dtype=np.int16
    )
    states = rng.integers(
        0,
        states_per_task,
        size=(draws, task_count, states_per_task),
        dtype=np.int16,
    )
    return tasks, states


def hierarchical_bootstrap_means(values, task_indices, state_indices):
    """Apply precomputed hierarchical bootstrap indices to a balanced matrix."""
    values = np.asarray(values, dtype=np.float64)
    task_indices = np.asarray(task_indices, dtype=np.int64)
    state_indices = np.asarray(state_indices, dtype=np.int64)
    if values.ndim != 2:
        raise ValueError("values must be task by state")
    draws = np.empty(len(task_indices), dtype=np.float64)
    for draw in range(len(draws)):
        task_means = []
        for slot, task in enumerate(task_indices[draw]):
            sampled = values[task, state_indices[draw, slot]]
            task_means.append(float(np.mean(sampled)))
        draws[draw] = float(np.mean(task_means))
    return draws


def countsketch_numpy(values, output_dim, seed):
    """Apply the normalized CountSketch used in Stages 13 and 13b."""
    values = np.asarray(values, dtype=np.float32)
    flattened = values.reshape(len(values), -1)
    rng = np.random.default_rng(int(seed))
    bucket = rng.integers(
        0, output_dim, size=flattened.shape[1], dtype=np.int64
    )
    sign = rng.choice(
        np.asarray([-1.0, 1.0], dtype=np.float32), flattened.shape[1]
    )
    counts = np.bincount(bucket, minlength=output_dim).astype(np.float32)
    counts[counts == 0] = 1.0
    output = np.empty((len(flattened), output_dim), dtype=np.float32)
    for row, value in enumerate(flattened):
        output[row] = np.bincount(
            bucket,
            weights=value * sign,
            minlength=output_dim,
        )
    return output / np.sqrt(counts)[None]
