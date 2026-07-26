"""Core metrics.

All outcome tensors use the canonical layout
``[initial_state, action_alternative, horizon, feature]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _outcomes(value: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 4:
        raise ValueError(
            f"{name} must have shape [state, action, horizon, feature], got {array.shape}"
        )
    if array.shape[1] < 2:
        raise ValueError(f"{name} needs at least two action alternatives")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _costs(value: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError(
            f"{name} must have shape [state, action, horizon], got {array.shape}"
        )
    if array.shape[1] < 2:
        raise ValueError(f"{name} needs at least two action alternatives")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


@dataclass(frozen=True)
class PairedMetrics:
    """Per-state, per-horizon prediction and intervention-effect metrics."""

    ordinary_rmse: FloatArray
    common_mode_rmse: FloatArray
    action_dependent_rmse: FloatArray
    paired_effect_rmse: FloatArray
    ground_truth_effect_rms: FloatArray
    normalized_paired_effect_rmse: FloatArray
    paired_effect_cosine: FloatArray
    identity_residual: FloatArray

    def summary(self) -> dict[str, float]:
        return {
            field: float(np.nanmean(getattr(self, field)))
            for field in self.__dataclass_fields__
        }


def paired_counterfactual_metrics(
    truth: ArrayLike,
    prediction: ArrayLike,
    *,
    eps: float = 1e-12,
) -> PairedMetrics:
    """Measure ordinary error and error in paired intervention effects.

    For a fixed initial state and horizon, let ``y_j`` be the simulator outcome
    under action sequence ``j`` and ``ŷ_j`` the model outcome. The paired error is

    ``RMSE_{j<k}[(ŷ_j-ŷ_k) - (y_j-y_k)]``.

    This removes action-invariant common error. It is exactly related to the
    centered prediction error:

    ``paired_MSE = 2A/(A-1) * mean_j ||e_j - mean(e)||² / D``,

    where ``A`` is the number of alternatives, ``D`` the feature dimension, and
    ``e_j = ŷ_j-y_j``. ``identity_residual`` numerically checks this equality.
    """

    y = _outcomes(truth, "truth")
    y_hat = _outcomes(prediction, "prediction")
    if y.shape != y_hat.shape:
        raise ValueError(f"truth and prediction shapes differ: {y.shape} != {y_hat.shape}")

    n_actions = y.shape[1]
    errors = y_hat - y
    ordinary_mse = np.mean(errors**2, axis=(1, 3))
    ordinary_rmse = np.sqrt(ordinary_mse)

    common = np.mean(errors, axis=1)
    common_mode_rmse = np.sqrt(np.mean(common**2, axis=-1))
    centered = errors - common[:, None, :, :]
    centered_mse = np.mean(centered**2, axis=(1, 3))
    action_dependent_rmse = np.sqrt(centered_mse)

    pair_truth: list[FloatArray] = []
    pair_prediction: list[FloatArray] = []
    for left in range(n_actions):
        for right in range(left + 1, n_actions):
            pair_truth.append(y[:, left] - y[:, right])
            pair_prediction.append(y_hat[:, left] - y_hat[:, right])

    dy = np.stack(pair_truth, axis=1)
    dy_hat = np.stack(pair_prediction, axis=1)
    effect_error = dy_hat - dy
    paired_effect_mse = np.mean(effect_error**2, axis=(1, 3))
    paired_effect_rmse = np.sqrt(paired_effect_mse)
    ground_truth_effect_rms = np.sqrt(np.mean(dy**2, axis=(1, 3)))
    normalized = paired_effect_rmse / np.maximum(ground_truth_effect_rms, eps)

    dot = np.sum(dy_hat * dy, axis=-1)
    denom = np.linalg.norm(dy_hat, axis=-1) * np.linalg.norm(dy, axis=-1)
    cosine_by_pair = np.divide(
        dot,
        denom,
        out=np.full_like(dot, np.nan),
        where=denom > eps,
    )
    paired_effect_cosine = np.nanmean(cosine_by_pair, axis=1)

    expected_pair_mse = (2.0 * n_actions / (n_actions - 1.0)) * centered_mse
    identity_residual = paired_effect_mse - expected_pair_mse

    return PairedMetrics(
        ordinary_rmse=ordinary_rmse,
        common_mode_rmse=common_mode_rmse,
        action_dependent_rmse=action_dependent_rmse,
        paired_effect_rmse=paired_effect_rmse,
        ground_truth_effect_rms=ground_truth_effect_rms,
        normalized_paired_effect_rmse=normalized,
        paired_effect_cosine=paired_effect_cosine,
        identity_residual=identity_residual,
    )


@dataclass(frozen=True)
class RankingMetrics:
    """Per-state, per-horizon action ranking and regret metrics."""

    selected_action: NDArray[np.int64]
    oracle_action: NDArray[np.int64]
    top1_correct: FloatArray
    regret: FloatArray
    normalized_regret: FloatArray
    pairwise_accuracy: FloatArray

    def summary(self) -> dict[str, float]:
        return {
            "top1_accuracy": float(np.mean(self.top1_correct)),
            "mean_regret": float(np.mean(self.regret)),
            "mean_normalized_regret": float(np.mean(self.normalized_regret)),
            "mean_pairwise_accuracy": float(np.nanmean(self.pairwise_accuracy)),
        }


def ranking_metrics(
    true_cost: ArrayLike,
    predicted_cost: ArrayLike,
    *,
    tie_tolerance: float = 1e-9,
    eps: float = 1e-12,
) -> RankingMetrics:
    """Evaluate whether the model ranks executable alternatives correctly.

    Costs are minimized. A selected action is counted as top-1 correct when its
    true cost is within ``tie_tolerance`` of the oracle minimum.
    """

    true = _costs(true_cost, "true_cost")
    predicted = _costs(predicted_cost, "predicted_cost")
    if true.shape != predicted.shape:
        raise ValueError(
            f"true_cost and predicted_cost shapes differ: {true.shape} != {predicted.shape}"
        )

    selected = np.argmin(predicted, axis=1).astype(np.int64)
    oracle = np.argmin(true, axis=1).astype(np.int64)
    state_idx, horizon_idx = np.indices(selected.shape)
    selected_true_cost = true[state_idx, selected, horizon_idx]
    oracle_cost = np.min(true, axis=1)
    top1 = (selected_true_cost <= oracle_cost + tie_tolerance).astype(np.float64)
    regret = selected_true_cost - oracle_cost
    cost_range = np.max(true, axis=1) - oracle_cost
    normalized_regret = np.divide(
        regret,
        np.maximum(cost_range, eps),
        out=np.zeros_like(regret),
        where=cost_range > eps,
    )

    concordant = np.zeros(selected.shape, dtype=np.float64)
    compared = np.zeros(selected.shape, dtype=np.float64)
    for left in range(true.shape[1]):
        for right in range(left + 1, true.shape[1]):
            true_delta = true[:, left] - true[:, right]
            predicted_delta = predicted[:, left] - predicted[:, right]
            valid = np.abs(true_delta) > tie_tolerance
            compared += valid
            concordant += valid & (
                (np.sign(true_delta) == np.sign(predicted_delta))
                | (np.abs(predicted_delta) <= tie_tolerance)
            )
    pairwise = np.divide(
        concordant,
        compared,
        out=np.full_like(concordant, np.nan),
        where=compared > 0,
    )

    return RankingMetrics(
        selected_action=selected,
        oracle_action=oracle,
        top1_correct=top1,
        regret=regret,
        normalized_regret=normalized_regret,
        pairwise_accuracy=pairwise,
    )

