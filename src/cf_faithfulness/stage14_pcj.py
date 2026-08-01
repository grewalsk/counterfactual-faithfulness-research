"""Numerical primitives for the Stage 14 predictive-control J-bundle pilot.

The functions in this module are deliberately NumPy-only so the notebook can
test its linear algebra on CPU before loading the frozen JEPA-WM checkpoint.
Model execution, VJPs, JVPs, and causal hooks live in the generated notebook.
"""

from __future__ import annotations

import math

import numpy as np


def symmetric_channel_metric(
    samples, shrinkage=0.10, relative_floor=1e-6
):
    """Fit a stable channel covariance, square root, and inverse square root.

    ``samples`` has shape ``(observations, channels)``.  The returned metric
    is equivariant under orthogonal channel rotations and deliberately uses a
    fixed shrinkage coefficient rather than selecting it on evaluation data.
    """
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("samples must be a two-dimensional nontrivial array")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must lie in [0, 1]")
    centered = values - values.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    isotropic = np.trace(covariance) / covariance.shape[0]
    covariance = (1.0 - shrinkage) * covariance
    covariance += shrinkage * isotropic * np.eye(covariance.shape[0])
    covariance = (covariance + covariance.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    maximum = max(float(np.max(eigenvalues)), 1e-12)
    floor = maximum * float(relative_floor)
    eigenvalues = np.maximum(eigenvalues, floor)
    square_root = (eigenvectors * np.sqrt(eigenvalues)[None]) @ eigenvectors.T
    inverse_square_root = (
        eigenvectors * (1.0 / np.sqrt(eigenvalues))[None]
    ) @ eigenvectors.T
    return {
        "mean": values.mean(axis=0),
        "covariance": covariance,
        "eigenvalues": eigenvalues,
        "square_root": square_root,
        "inverse_square_root": inverse_square_root,
        "condition_number": float(eigenvalues.max() / eigenvalues.min()),
    }


def channel_metric_from_moments(
    count, total, cross, shrinkage=0.10, relative_floor=1e-6
):
    """Fit the same stable metric from streamed first/second moments."""
    count = int(count)
    total = np.asarray(total, dtype=np.float64)
    cross = np.asarray(cross, dtype=np.float64)
    if count < 2 or total.ndim != 1 or cross.shape != (len(total), len(total)):
        raise ValueError("invalid streamed channel moments")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must lie in [0, 1]")
    mean = total / count
    covariance = (cross - np.outer(total, total) / count) / (count - 1)
    covariance = (covariance + covariance.T) / 2.0
    isotropic = np.trace(covariance) / covariance.shape[0]
    covariance = (1.0 - shrinkage) * covariance
    covariance += shrinkage * isotropic * np.eye(covariance.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    maximum = max(float(np.max(eigenvalues)), 1e-12)
    floor = maximum * float(relative_floor)
    eigenvalues = np.maximum(eigenvalues, floor)
    square_root = (eigenvectors * np.sqrt(eigenvalues)[None]) @ eigenvectors.T
    inverse_square_root = (
        eigenvectors * (1.0 / np.sqrt(eigenvalues))[None]
    ) @ eigenvectors.T
    return {
        "mean": mean,
        "covariance": covariance,
        "eigenvalues": eigenvalues,
        "square_root": square_root,
        "inverse_square_root": inverse_square_root,
        "condition_number": float(eigenvalues.max() / eigenvalues.min()),
    }


def transform_primal_channels(values, inverse_square_root):
    """Whiten hidden-space vectors along their final channel dimension."""
    array = np.asarray(values, dtype=np.float64)
    inverse = np.asarray(inverse_square_root, dtype=np.float64)
    if array.shape[-1] != inverse.shape[0] or inverse.shape[0] != inverse.shape[1]:
        raise ValueError("channel metric does not match primal values")
    return np.einsum("...c,dc->...d", array, inverse, optimize=True)


def inverse_transform_primal_channels(values, square_root):
    """Map a whitened hidden-space vector back to native coordinates."""
    array = np.asarray(values, dtype=np.float64)
    root = np.asarray(square_root, dtype=np.float64)
    if array.shape[-1] != root.shape[0] or root.shape[0] != root.shape[1]:
        raise ValueError("channel metric does not match primal values")
    return np.einsum("...c,dc->...d", array, root, optimize=True)


def transform_dual_channels(values, square_root):
    """Transform covectors so dual-primal contractions remain invariant."""
    array = np.asarray(values, dtype=np.float64)
    root = np.asarray(square_root, dtype=np.float64)
    if array.shape[-1] != root.shape[0] or root.shape[0] != root.shape[1]:
        raise ValueError("channel metric does not match dual values")
    return np.einsum("...c,cd->...d", array, root, optimize=True)


def balanced_modes(observability, controllability, tolerance=1e-7):
    """Return empirical balanced primal/dual modes from a small Hankel SVD.

    Parameters
    ----------
    observability:
        Matrix ``G`` with one query-pullback covector per row, shape ``(q,d)``.
    controllability:
        Matrix ``B`` with one action-written direction per column, shape
        ``(d,p)``.

    The construction avoids any ``d x d`` matrix.  If ``H = G B`` and
    ``H = U S V^T``, the returned modes satisfy ``dual.T @ primal ~= I``.
    """
    g = np.asarray(observability, dtype=np.float64)
    b = np.asarray(controllability, dtype=np.float64)
    if g.ndim != 2 or b.ndim != 2 or g.shape[1] != b.shape[0]:
        raise ValueError("observability and controllability shapes disagree")
    hankel = g @ b
    u, singular, vh = np.linalg.svd(hankel, full_matrices=False)
    if not len(singular) or singular[0] <= 0:
        return {
            "hankel": hankel,
            "singular_values": singular,
            "primal": np.empty((b.shape[0], 0), dtype=np.float64),
            "dual": np.empty((b.shape[0], 0), dtype=np.float64),
            "biorthogonality_error": math.nan,
        }
    keep = singular > max(float(singular[0]) * tolerance, 1e-12)
    retained = singular[keep]
    inverse_root = np.diag(1.0 / np.sqrt(retained))
    primal = b @ vh.T[:, keep] @ inverse_root
    dual = g.T @ u[:, keep] @ inverse_root
    identity = np.eye(len(retained), dtype=np.float64)
    error = np.linalg.norm(dual.T @ primal - identity)
    error /= max(np.linalg.norm(identity), 1e-12)
    return {
        "hankel": hankel,
        "singular_values": singular,
        "primal": primal,
        "dual": dual,
        "biorthogonality_error": float(error),
    }


def canonical_mode_rows(primal, dual):
    """Stack unit primal and dual mode rows with deterministic signs."""
    rows = []
    labels = []
    for kind, matrix in [("primal", primal), ("dual", dual)]:
        values = np.asarray(matrix, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("mode matrix must be two-dimensional")
        for index in range(values.shape[1]):
            row = values[:, index].copy()
            norm = np.linalg.norm(row)
            if norm <= 1e-12:
                continue
            row /= norm
            pivot = int(np.argmax(np.abs(row)))
            if row[pivot] < 0:
                row *= -1.0
            rows.append(row)
            labels.append((kind, index))
    if not rows:
        return np.empty((0, np.asarray(primal).shape[0])), labels
    return np.stack(rows), labels


def training_span(values, relative_tolerance=1e-9):
    """Represent high-dimensional rows losslessly in their sample span."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or not len(x):
        raise ValueError("values must be a nonempty row matrix")
    gram = x @ x.T
    gram = (gram + gram.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    threshold = max(float(eigenvalues[0]) * relative_tolerance, 1e-12)
    keep = eigenvalues > threshold
    eigenvalues = eigenvalues[keep]
    eigenvectors = eigenvectors[:, keep]
    basis = x.T @ (eigenvectors / np.sqrt(eigenvalues)[None])
    coordinates = x @ basis
    return {
        "basis": basis,
        "coordinates": coordinates,
        "eigenvalues": eigenvalues,
        "gram": gram,
    }


def omp_codes(values, dictionary, sparsity):
    """Deterministic signed orthogonal matching pursuit.

    ``dictionary`` contains unit atom rows.  Coefficients are signed; this is
    equivalent to nonnegative coding over explicit ``+atom/-atom`` pairs.
    """
    x = np.asarray(values, dtype=np.float64)
    atoms = np.asarray(dictionary, dtype=np.float64)
    if x.ndim == 1:
        x = x[None]
    if x.ndim != 2 or atoms.ndim != 2 or x.shape[1] != atoms.shape[1]:
        raise ValueError("values and dictionary shapes disagree")
    if not 1 <= int(sparsity) <= len(atoms):
        raise ValueError("invalid sparsity")
    norms = np.linalg.norm(atoms, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("dictionary contains a zero atom")
    atoms = atoms / norms[:, None]
    codes = np.zeros((len(x), len(atoms)), dtype=np.float64)
    for row_index, target in enumerate(x):
        residual = target.copy()
        selected = []
        coefficient = np.empty(0, dtype=np.float64)
        for _ in range(int(sparsity)):
            correlations = atoms @ residual
            if selected:
                correlations[np.asarray(selected, dtype=np.int64)] = 0.0
            candidate = int(np.argmax(np.abs(correlations)))
            if abs(correlations[candidate]) <= 1e-12:
                break
            selected.append(candidate)
            design = atoms[selected].T
            coefficient = np.linalg.lstsq(design, target, rcond=None)[0]
            residual = target - design @ coefficient
        if selected:
            codes[row_index, selected] = coefficient
    return codes


def sparse_reconstruct(values, dictionary, sparsity):
    """Return OMP coefficients and reconstructions in one call."""
    atoms = np.asarray(dictionary, dtype=np.float64)
    norms = np.linalg.norm(atoms, axis=1)
    atoms = atoms / np.maximum(norms[:, None], 1e-12)
    codes = omp_codes(values, atoms, sparsity)
    return codes, codes @ atoms


def transfer_metrics(target, prediction):
    """Score a predicted local input-output transfer matrix."""
    truth = np.asarray(target, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    if truth.shape != estimate.shape:
        raise ValueError("target and prediction shapes differ")
    denominator = float(np.sum(truth**2))
    residual = float(np.sum((truth - estimate) ** 2))
    cosine_denominator = np.linalg.norm(truth) * np.linalg.norm(estimate)
    cosine = (
        float(np.sum(truth * estimate) / cosine_denominator)
        if cosine_denominator > 1e-12
        else math.nan
    )
    return {
        "energy": denominator,
        "reconstruction": 1.0 - residual / max(denominator, 1e-12),
        "cosine": cosine,
        "scale": float(np.sum(truth * estimate) / max(denominator, 1e-12)),
    }


def earliest_within_one_se(task_by_layer):
    """Choose the earliest layer within one standard error of the best."""
    values = np.asarray(task_by_layer, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("task_by_layer must contain at least two tasks")
    means = np.nanmean(values, axis=0)
    counts = np.sum(np.isfinite(values), axis=0)
    errors = np.nanstd(values, axis=0, ddof=1) / np.sqrt(counts)
    best = int(np.nanargmax(means))
    threshold = float(means[best] - errors[best])
    selected = int(np.flatnonzero(means >= threshold)[0])
    return {
        "selected_index": selected,
        "best_index": best,
        "threshold": threshold,
        "means": means,
        "standard_errors": errors,
    }


def exact_positive_sign_test(values):
    """One-sided exact sign test after discarding exact ties."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array) & (array != 0)]
    positives = int(np.sum(array > 0))
    count = int(len(array))
    probability = sum(
        math.comb(count, k) for k in range(positives, count + 1)
    ) / (2**count if count else 1)
    return {"positive": positives, "nonzero": count, "one_sided_p": probability}


def one_sided_t_lower(values, critical=None):
    """Task-level one-sided 95% lower confidence bound with the actual df."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) < 2:
        return math.nan
    if critical is None:
        one_sided_95 = (
            math.nan,
            6.313752, 2.919986, 2.353363, 2.131847, 2.015048,
            1.943180, 1.894579, 1.859548, 1.833113, 1.812461,
            1.795885, 1.782288, 1.770933, 1.761310, 1.753050,
            1.745884, 1.739607, 1.734064, 1.729133, 1.724718,
            1.720743, 1.717144, 1.713872, 1.710882, 1.708141,
            1.705618, 1.703288, 1.701131, 1.699127, 1.697261,
        )
        degrees = len(array) - 1
        critical = one_sided_95[degrees] if degrees <= 30 else 1.644854
    return float(
        np.mean(array)
        - critical * np.std(array, ddof=1) / np.sqrt(len(array))
    )


def hierarchical_bootstrap_means(
    values, task_ids, draws=10_000, seed=0
):
    """Resample tasks, then states within sampled tasks."""
    values = np.asarray(values, dtype=np.float64)
    task_ids = np.asarray(task_ids)
    if len(values) != len(task_ids):
        raise ValueError("values and task_ids differ in length")
    tasks = np.unique(task_ids)
    rows = {task: np.flatnonzero(task_ids == task) for task in tasks}
    rng = np.random.default_rng(int(seed))
    output = np.empty(int(draws), dtype=np.float64)
    for draw in range(int(draws)):
        sampled_tasks = rng.choice(tasks, size=len(tasks), replace=True)
        task_means = []
        for task in sampled_tasks:
            indices = rows[task]
            sampled = rng.choice(indices, size=len(indices), replace=True)
            task_means.append(np.nanmean(values[sampled]))
        output[draw] = np.nanmean(task_means)
    return output


def haar_rotation(dimension, seed):
    """Draw a deterministic proper orthogonal matrix."""
    rng = np.random.default_rng(int(seed))
    q, r = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    q *= signs[None]
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


def relative_error(observed, expected):
    """Normalized Frobenius error with a stable zero denominator."""
    left = np.asarray(observed, dtype=np.float64)
    right = np.asarray(expected, dtype=np.float64)
    return float(
        np.linalg.norm(left - right) / max(np.linalg.norm(right), 1e-12)
    )
