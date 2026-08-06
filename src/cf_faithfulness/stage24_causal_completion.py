"""Numerical primitives for Stage 24 causal completion rank.

Stage 24 decomposes a natural counterfactual activation difference into the
exact low-dimensional mode edit tested in Stage 23 and a remaining completion
residual.  These NumPy helpers define the decomposition, nested completion
edits, reconstruction scores, and the preregistered completion-rank decision.
Model hooks and large GPU factorizations remain in the Colab notebook.
"""

from __future__ import annotations

import numpy as np


def completion_residual(native_delta, mode_edit, mode_covectors, *, tolerance=1e-8):
    """Return the part of a natural edit not supplied by an exact mode edit.

    Both edits are flattened whitened-space vectors.  A valid decomposition
    requires the residual to lie in the null space of the frozen mode
    covectors; otherwise adding completion components would silently alter the
    already-certified mode coordinates.
    """

    native = np.asarray(native_delta, dtype=np.float64).reshape(-1)
    mode = np.asarray(mode_edit, dtype=np.float64).reshape(-1)
    covectors = np.asarray(mode_covectors, dtype=np.float64)
    if native.shape != mode.shape:
        raise ValueError("native and mode edits must align")
    if covectors.ndim != 2 or covectors.shape[0] != len(native):
        raise ValueError("mode covectors do not match edit width")
    residual = native - mode
    coordinate_residual = covectors.T @ residual
    norm = float(np.linalg.norm(coordinate_residual))
    relative = norm / max(float(np.linalg.norm(covectors.T @ native)), 1e-12)
    if relative > float(tolerance):
        raise ValueError(
            f"completion residual changes frozen mode coordinates: {relative}"
        )
    return {
        "residual": residual,
        "coordinate_residual_norm": norm,
        "relative_coordinate_residual": float(relative),
    }


def orthonormal_residual_basis(residuals, rank, excluded=None):
    """Fit an uncentered nested right-singular basis to residual rows."""

    matrix = np.asarray(residuals, dtype=np.float64)
    rank = int(rank)
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise ValueError("residuals must be a nonempty row matrix")
    if rank < 1 or rank > min(matrix.shape):
        raise ValueError("rank exceeds the residual matrix")
    _, singular, right = np.linalg.svd(matrix, full_matrices=False)
    basis = right[:rank].T
    if excluded is not None:
        removed = np.asarray(excluded, dtype=np.float64)
        if removed.ndim == 1:
            removed = removed[:, None]
        if removed.ndim != 2 or removed.shape[0] != basis.shape[0]:
            raise ValueError("excluded covectors do not match residual width")
        gram = removed.T @ removed
        basis -= removed @ np.linalg.solve(gram, removed.T @ basis)
        basis, _ = np.linalg.qr(basis, mode="reduced")
    if basis.shape[1] != rank:
        raise ValueError("residual basis lost requested rank")
    return basis, singular


def completion_edit(mode_edit, residual, basis, rank):
    """Add the rank-k nested projection of a natural residual to a mode edit."""

    mode = np.asarray(mode_edit, dtype=np.float64).reshape(-1)
    remainder = np.asarray(residual, dtype=np.float64).reshape(-1)
    directions = np.asarray(basis, dtype=np.float64)
    rank = int(rank)
    if mode.shape != remainder.shape:
        raise ValueError("mode edit and residual must align")
    if directions.ndim != 2 or directions.shape[0] != len(mode):
        raise ValueError("completion basis does not match edit width")
    if rank < 0 or rank > directions.shape[1]:
        raise ValueError("completion rank is out of range")
    if rank == 0:
        projected = np.zeros_like(remainder)
    else:
        active = directions[:, :rank]
        projected = active @ (active.T @ remainder)
    return mode + projected


def native_reconstruction_fraction(native_delta, completed_edit):
    """Measure how much native activation distance an edit closes."""

    native = np.asarray(native_delta, dtype=np.float64).reshape(-1)
    completed = np.asarray(completed_edit, dtype=np.float64).reshape(-1)
    if native.shape != completed.shape or not len(native):
        raise ValueError("native and completed edits must be aligned vectors")
    denominator = float(np.linalg.norm(native))
    if denominator <= 1e-12:
        raise ValueError("native edit is degenerate")
    return float(1.0 - np.linalg.norm(native - completed) / denominator)


def causal_completion_rank(rank_summaries, threshold=0.8):
    """Return the smallest rank whose transfer confidence bound clears a target."""

    rows = sorted(rank_summaries, key=lambda row: int(row["rank"]))
    threshold = float(threshold)
    if not rows or not 0.0 < threshold <= 1.0:
        raise ValueError("rank summaries and threshold must be valid")
    for row in rows:
        if float(row["lower"]) >= threshold:
            return int(row["rank"])
    return None


def paired_completion_gain(learned, control):
    """Return aligned learned-minus-control completion effects."""

    left = np.asarray(learned, dtype=np.float64).reshape(-1)
    right = np.asarray(control, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or not len(left):
        raise ValueError("learned and control effects must be aligned")
    return left - right
