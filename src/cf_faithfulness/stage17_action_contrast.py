"""Numerical primitives for the Stage 17 finite action-contrast pilot.

The module is deliberately NumPy-only.  Model hooks and GPU execution live in
the generated Colab notebook, while the causal estimands and subspace algebra
remain independently testable on CPU.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np


def candidate_center(values):
    """Remove the within-state mean across candidate actions."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 2 or array.shape[0] < 2:
        raise ValueError("values must have candidate rows and at least two actions")
    return array - np.mean(array, axis=0, keepdims=True)


def stable_seed(*items):
    """Derive a deterministic unsigned 32-bit seed from structured labels."""
    payload = "|".join(str(item) for item in items).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def fixed_derangement(length, seed):
    """Return a deterministic permutation with no fixed points."""
    length = int(length)
    if length < 2:
        raise ValueError("a derangement requires at least two items")
    rng = np.random.default_rng(int(seed))
    identity = np.arange(length, dtype=np.int64)
    for _ in range(10_000):
        candidate = rng.permutation(length)
        if np.all(candidate != identity):
            return candidate
    # This path is effectively unreachable, but a fixed cyclic permutation is
    # a valid fail-closed fallback for every length greater than one.
    return np.roll(identity, 1)


def linear_cka(left, right):
    """Linear centered-kernel alignment between candidate-row geometries."""
    x = candidate_center(np.asarray(left, dtype=np.float64).reshape(len(left), -1))
    y = candidate_center(np.asarray(right, dtype=np.float64).reshape(len(right), -1))
    if len(x) != len(y):
        raise ValueError("left and right must contain the same candidate rows")
    gram_x = x @ x.T
    gram_y = y @ y.T
    denominator = np.linalg.norm(gram_x) * np.linalg.norm(gram_y)
    if denominator <= 1e-12:
        return math.nan
    return float(np.sum(gram_x * gram_y) / denominator)


def action_swap_delta(values, permutation, basis=None, dose=1.0):
    """Construct a finite donor-action residual edit.

    ``values`` has one activation row per candidate action.  ``basis`` contains
    orthonormal column directions.  With ``basis=None`` and ``dose=1``, adding
    the returned edit exactly permutes the complete candidate activations.
    """
    array = np.asarray(values, dtype=np.float64)
    original_shape = array.shape
    flat = array.reshape(array.shape[0], -1)
    permutation = np.asarray(permutation, dtype=np.int64)
    if permutation.shape != (len(flat),) or sorted(permutation.tolist()) != list(
        range(len(flat))
    ):
        raise ValueError("permutation is invalid")
    residual = candidate_center(flat)
    difference = residual[permutation] - residual
    if basis is not None:
        directions = np.asarray(basis, dtype=np.float64)
        if directions.ndim != 2 or directions.shape[0] != flat.shape[1]:
            raise ValueError("basis does not match flattened activation width")
        difference = (difference @ directions) @ directions.T
    return (float(dose) * difference).reshape(original_shape)


def matched_common_mode(template_delta, direction):
    """Repeat one direction across actions and exactly match Frobenius energy."""
    template = np.asarray(template_delta, dtype=np.float64)
    vector = np.asarray(direction, dtype=np.float64).reshape(-1)
    if template.ndim < 2 or vector.size != int(np.prod(template.shape[1:])):
        raise ValueError("common direction does not match the activation shape")
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        raise ValueError("common direction must be nonzero")
    repeated = np.broadcast_to(vector / norm, (template.shape[0], vector.size)).copy()
    repeated *= np.linalg.norm(template) / max(np.linalg.norm(repeated), 1e-12)
    return repeated.reshape(template.shape)


def donor_transfer_metrics(baseline, patched, permutation):
    """Measure directional and reconstructive transfer toward donor outcomes.

    All calculations occur after candidate centering, so shared output drift
    cannot masquerade as donor-specific transfer.  The no-edit baseline has
    coefficient/reconstruction zero; an exact donor permutation has one.
    """
    base = np.asarray(baseline, dtype=np.float64).reshape(len(baseline), -1)
    edit = np.asarray(patched, dtype=np.float64).reshape(len(patched), -1)
    permutation = np.asarray(permutation, dtype=np.int64)
    if base.shape != edit.shape or permutation.shape != (len(base),):
        raise ValueError("donor-transfer inputs have inconsistent shapes")
    centered_base = candidate_center(base)
    centered_edit = candidate_center(edit)
    target = centered_base[permutation] - centered_base
    observed = centered_edit - centered_base
    denominator = float(np.sum(target**2))
    observed_energy = float(np.sum(observed**2))
    if denominator <= 1e-12:
        return {
            "target_energy": denominator,
            "coefficient": math.nan,
            "cosine": math.nan,
            "reconstruction": math.nan,
            "mean_shift_ratio": math.nan,
        }
    coefficient = float(np.sum(observed * target) / denominator)
    cosine_denominator = math.sqrt(denominator * observed_energy)
    cosine = (
        float(np.sum(observed * target) / cosine_denominator)
        if cosine_denominator > 1e-12
        else 0.0
    )
    reconstruction = 1.0 - float(np.sum((observed - target) ** 2)) / denominator
    mean_shift = np.mean(edit, axis=0) - np.mean(base, axis=0)
    return {
        "target_energy": denominator,
        "coefficient": coefficient,
        "cosine": cosine,
        "reconstruction": reconstruction,
        "mean_shift_ratio": float(np.linalg.norm(mean_shift) / math.sqrt(denominator)),
    }


def grouped_kernel_ridge_cv(kernel, targets, groups, ridge_multipliers):
    """Select a dual kernel-ridge penalty without crossing group boundaries.

    The ridge grid is expressed relative to ``trace(kernel) / n`` so it remains
    meaningful after a change in activation width or overall scale.
    """
    gram = np.asarray(kernel, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    labels = np.asarray(groups)
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("kernel must be square")
    if y.ndim != 2 or len(y) != len(gram) or len(labels) != len(gram):
        raise ValueError("kernel, targets, and groups disagree")
    scale = float(np.trace(gram) / max(len(gram), 1))
    scale = max(scale, 1e-12)
    rows = []
    unique = np.unique(labels)
    for multiplier in ridge_multipliers:
        penalty = float(multiplier) * scale
        fold_losses = []
        for held_out in unique:
            test = labels == held_out
            train = ~test
            if np.sum(train) < 2 or not np.any(test):
                raise ValueError("each grouped fold needs train and test rows")
            k_train = gram[np.ix_(train, train)]
            alpha = np.linalg.solve(
                k_train + penalty * np.eye(len(k_train)), y[train]
            )
            prediction = gram[np.ix_(test, train)] @ alpha
            denominator = float(np.sum((y[test] - np.mean(y[train], axis=0)) ** 2))
            numerator = float(np.sum((y[test] - prediction) ** 2))
            fold_losses.append(numerator / max(denominator, 1e-12))
            rows.append(
                {
                    "ridge_multiplier": float(multiplier),
                    "held_out_group": int(held_out),
                    "normalized_mse": fold_losses[-1],
                }
            )
        rows.append(
            {
                "ridge_multiplier": float(multiplier),
                "held_out_group": "mean",
                "normalized_mse": float(np.mean(fold_losses)),
            }
        )
    means = [row for row in rows if row["held_out_group"] == "mean"]
    selected = min(means, key=lambda row: (row["normalized_mse"], row["ridge_multiplier"]))
    return {
        "selected_multiplier": float(selected["ridge_multiplier"]),
        "kernel_scale": scale,
        "penalty": float(selected["ridge_multiplier"] * scale),
        "rows": rows,
    }


def fit_dual_ridge_basis(features, targets, penalty, max_rank):
    """Fit the input directions of a finite contrast-to-outcome ridge map."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError("features and targets must be aligned row matrices")
    gram = x @ x.T
    alpha = np.linalg.solve(gram + float(penalty) * np.eye(len(gram)), y)
    weight = x.T @ alpha
    left, singular, _ = np.linalg.svd(weight, full_matrices=False)
    keep = min(int(max_rank), left.shape[1], int(np.sum(singular > 1e-10)))
    if keep < 1:
        raise ValueError("ridge map has no nonzero input direction")
    return {
        "basis": left[:, :keep],
        "singular_values": singular,
        "weight": weight,
        "dual_coefficients": alpha,
    }


def random_subspace_in_span(features, rank, seed, orthogonal_to=None):
    """Draw an orthonormal rank-``r`` basis in the empirical action span."""
    x = np.asarray(features, dtype=np.float64)
    rank = int(rank)
    if x.ndim != 2 or rank < 1:
        raise ValueError("features must be a row matrix and rank must be positive")
    rng = np.random.default_rng(int(seed))
    coefficients = rng.normal(size=(len(x), rank + 8))
    candidate = x.T @ coefficients
    if orthogonal_to is not None:
        excluded = np.asarray(orthogonal_to, dtype=np.float64)
        candidate -= excluded @ (excluded.T @ candidate)
    q, r = np.linalg.qr(candidate, mode="reduced")
    diagonal = np.abs(np.diag(r))
    valid = int(np.sum(diagonal > max(diagonal.max(initial=0.0) * 1e-9, 1e-12)))
    if valid < rank:
        raise ValueError("empirical action span is too small for requested random rank")
    return q[:, :rank]


def pair_indices(length):
    return np.triu_indices(int(length), k=1)


def ranking_metrics(true_cost, predicted_cost, tie=1e-9):
    """Return planning regret and pairwise-ranking outcomes for one state."""
    truth = np.asarray(true_cost, dtype=np.float64)
    prediction = np.asarray(predicted_cost, dtype=np.float64)
    if truth.shape != prediction.shape or truth.ndim != 1:
        raise ValueError("cost vectors must be aligned and one-dimensional")
    selected = int(np.argmin(prediction))
    oracle = int(np.argmin(truth))
    best = float(np.min(truth))
    chosen = float(truth[selected])
    spread = float(np.max(truth) - best)
    normalized_regret = (chosen - best) / spread if spread > tie else 0.0
    left, right = pair_indices(len(truth))
    true_margin = truth[left] - truth[right]
    predicted_margin = prediction[left] - prediction[right]
    valid = np.abs(true_margin) > tie
    credit = np.full(len(left), np.nan)
    same = np.sign(true_margin) == np.sign(predicted_margin)
    credit[valid & same] = 1.0
    credit[valid & (np.abs(predicted_margin) <= tie)] = 0.5
    credit[valid & np.isnan(credit)] = 0.0
    weights = np.abs(true_margin)
    weighted = (
        float(np.nansum(weights * credit) / np.sum(weights[valid]))
        if np.any(valid)
        else math.nan
    )
    return {
        "selected_action": selected,
        "oracle_action": oracle,
        "top1_correct": float(chosen <= best + tie),
        "normalized_regret": float(normalized_regret),
        "weighted_pairwise_accuracy": weighted,
    }


def pose_target(states):
    """Map PushT simulator states to normalized block pose coordinates."""
    states = np.asarray(states, dtype=np.float64)
    angle = states[..., 4]
    return np.stack(
        [states[..., 2] / 512.0, states[..., 3] / 512.0, np.sin(angle), np.cos(angle)],
        axis=-1,
    )


def decoded_task_cost(prediction, goal):
    """Stage 3/4 normalized PushT goal cost for decoded block poses."""
    prediction = np.asarray(prediction, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    angle = np.arctan2(prediction[..., 2], prediction[..., 3])
    angular = np.arctan2(np.sin(angle - goal[2]), np.cos(angle - goal[2]))
    return np.linalg.norm(
        np.concatenate(
            [prediction[..., :2] - goal[:2] / 512.0, (angular / np.pi)[..., None]],
            axis=-1,
        ),
        axis=-1,
    )


def exact_positive_sign_test(values):
    """One-sided exact sign-test p-value after dropping exact zeros."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array) & (array != 0)]
    positives = int(np.sum(array > 0))
    total = int(len(array))
    if total == 0:
        return {"positive": 0, "nonzero": 0, "p_value": math.nan}
    probability = sum(math.comb(total, k) for k in range(positives, total + 1)) / 2**total
    return {"positive": positives, "nonzero": total, "p_value": float(probability)}
