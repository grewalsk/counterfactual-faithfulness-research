import numpy as np

from cf_faithfulness.metrics import (
    paired_counterfactual_metrics,
    ranking_metrics,
)


def test_perfect_prediction_has_zero_errors():
    rng = np.random.default_rng(2)
    truth = rng.normal(size=(4, 3, 2, 5))
    metrics = paired_counterfactual_metrics(truth, truth)
    assert np.allclose(metrics.ordinary_rmse, 0.0)
    assert np.allclose(metrics.paired_effect_rmse, 0.0)
    assert np.allclose(metrics.normalized_paired_effect_rmse, 0.0)
    assert np.allclose(metrics.identity_residual, 0.0, atol=1e-15)


def test_common_bias_is_removed_by_paired_metric():
    rng = np.random.default_rng(3)
    truth = rng.normal(size=(5, 4, 3, 7))
    common_bias = rng.normal(size=(5, 1, 3, 7))
    prediction = truth + common_bias
    metrics = paired_counterfactual_metrics(truth, prediction)
    assert np.all(metrics.ordinary_rmse > 0)
    assert np.allclose(metrics.paired_effect_rmse, 0.0, atol=1e-14)
    assert np.allclose(metrics.action_dependent_rmse, 0.0, atol=1e-14)


def test_action_dependent_error_satisfies_pairwise_identity():
    rng = np.random.default_rng(4)
    truth = rng.normal(size=(6, 5, 3, 11))
    prediction = truth + rng.normal(scale=0.3, size=truth.shape)
    metrics = paired_counterfactual_metrics(truth, prediction)
    assert np.max(np.abs(metrics.identity_residual)) < 1e-14


def test_ranking_and_regret():
    true = np.array([[[1.0], [0.0], [2.0]], [[0.0], [3.0], [2.0]]])
    predicted = np.array([[[2.0], [0.0], [1.0]], [[2.0], [0.0], [1.0]]])
    metrics = ranking_metrics(true, predicted)
    assert metrics.top1_correct[:, 0].tolist() == [1.0, 0.0]
    assert metrics.regret[:, 0].tolist() == [0.0, 3.0]
    assert np.all(metrics.normalized_regret >= 0)
    assert np.all(metrics.normalized_regret <= 1)

