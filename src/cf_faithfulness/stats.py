"""State-clustered uncertainty and held-out incremental-validity analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    low: float
    high: float
    confidence: float
    n_clusters: int
    n_bootstrap: int


def clustered_bootstrap_mean(
    values: ArrayLike,
    cluster: ArrayLike,
    *,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapInterval:
    """Percentile interval resampling independent initial-state clusters."""

    y = np.asarray(values, dtype=np.float64).reshape(-1)
    g = np.asarray(cluster).reshape(-1)
    if y.shape != g.shape:
        raise ValueError("values and cluster must have equal length")
    if not np.all(np.isfinite(y)):
        raise ValueError("values contains non-finite entries")
    unique = np.unique(g)
    if unique.size < 2:
        raise ValueError("at least two clusters are required")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")

    grouped = [y[g == group] for group in unique]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap, dtype=np.float64)
    for idx in range(n_bootstrap):
        sampled = rng.integers(0, unique.size, size=unique.size)
        draws[idx] = np.mean(np.concatenate([grouped[item] for item in sampled]))
    alpha = 1.0 - confidence
    low, high = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapInterval(
        estimate=float(np.mean(y)),
        low=float(low),
        high=float(high),
        confidence=confidence,
        n_clusters=int(unique.size),
        n_bootstrap=n_bootstrap,
    )


def _design(*columns: ArrayLike) -> FloatArray:
    parts = [np.asarray(column, dtype=np.float64).reshape(-1, 1) for column in columns]
    if not parts:
        raise ValueError("at least one predictor is required")
    n = parts[0].shape[0]
    if any(part.shape[0] != n for part in parts):
        raise ValueError("predictors must have equal length")
    matrix = np.column_stack(parts)
    return np.column_stack([np.ones(n), matrix])


def _ridge_predict(
    x_train: FloatArray,
    y_train: FloatArray,
    x_test: FloatArray,
    ridge: float,
) -> FloatArray:
    penalty = np.eye(x_train.shape[1]) * ridge
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ y_train)
    return x_test @ coef


@dataclass(frozen=True)
class IncrementalValidity:
    base_r2: float
    full_r2: float
    delta_r2: float
    base_rmse: float
    full_rmse: float
    fold_by_row: NDArray[np.int64]
    base_prediction: FloatArray
    full_prediction: FloatArray


def grouped_incremental_validity(
    outcome: ArrayLike,
    ordinary_error: ArrayLike,
    counterfactual_error: ArrayLike,
    initial_state_id: ArrayLike,
    *,
    nuisance: ArrayLike | None = None,
    n_splits: int = 5,
    ridge: float = 1e-6,
    seed: int = 0,
) -> IncrementalValidity:
    """Cross-validated incremental R² with whole initial states held out.

    The base model uses ordinary rollout error plus optional nuisance columns.
    The full model adds counterfactual error. This is a diagnostic association,
    not a causal effect estimate.
    """

    y = np.asarray(outcome, dtype=np.float64).reshape(-1)
    ordinary = np.asarray(ordinary_error, dtype=np.float64).reshape(-1)
    counterfactual = np.asarray(counterfactual_error, dtype=np.float64).reshape(-1)
    groups = np.asarray(initial_state_id).reshape(-1)
    if not (y.shape == ordinary.shape == counterfactual.shape == groups.shape):
        raise ValueError("outcome, errors, and state ids must have equal length")
    if not np.all(np.isfinite(np.column_stack([y, ordinary, counterfactual]))):
        raise ValueError("inputs contain non-finite values")

    if nuisance is None:
        nuisance_matrix = np.empty((y.size, 0), dtype=np.float64)
    else:
        nuisance_matrix = np.asarray(nuisance, dtype=np.float64)
        if nuisance_matrix.ndim == 1:
            nuisance_matrix = nuisance_matrix[:, None]
        if nuisance_matrix.shape[0] != y.size:
            raise ValueError("nuisance must have one row per outcome")

    unique = np.unique(groups)
    if not 2 <= n_splits <= unique.size:
        raise ValueError("n_splits must be between 2 and the number of states")
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    fold_groups = np.array_split(shuffled, n_splits)

    base_prediction = np.empty_like(y)
    full_prediction = np.empty_like(y)
    fold_by_row = np.empty(y.size, dtype=np.int64)
    for fold, held_out_groups in enumerate(fold_groups):
        test = np.isin(groups, held_out_groups)
        train = ~test
        x_base_train = _design(ordinary[train], *nuisance_matrix[train].T)
        x_base_test = _design(ordinary[test], *nuisance_matrix[test].T)
        x_full_train = _design(
            ordinary[train], counterfactual[train], *nuisance_matrix[train].T
        )
        x_full_test = _design(
            ordinary[test], counterfactual[test], *nuisance_matrix[test].T
        )
        base_prediction[test] = _ridge_predict(
            x_base_train, y[train], x_base_test, ridge
        )
        full_prediction[test] = _ridge_predict(
            x_full_train, y[train], x_full_test, ridge
        )
        fold_by_row[test] = fold

    total = np.sum((y - np.mean(y)) ** 2)
    base_sse = np.sum((y - base_prediction) ** 2)
    full_sse = np.sum((y - full_prediction) ** 2)
    base_r2 = 1.0 - base_sse / total
    full_r2 = 1.0 - full_sse / total
    return IncrementalValidity(
        base_r2=float(base_r2),
        full_r2=float(full_r2),
        delta_r2=float(full_r2 - base_r2),
        base_rmse=float(np.sqrt(np.mean((y - base_prediction) ** 2))),
        full_rmse=float(np.sqrt(np.mean((y - full_prediction) ** 2))),
        fold_by_row=fold_by_row,
        base_prediction=base_prediction,
        full_prediction=full_prediction,
    )

