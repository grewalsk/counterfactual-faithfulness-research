import numpy as np

from cf_faithfulness.stats import (
    clustered_bootstrap_mean,
    grouped_incremental_validity,
)


def test_clustered_bootstrap_reproducible():
    values = np.array([0.0, 1.0, 2.0, 3.0])
    groups = np.array([0, 0, 1, 1])
    first = clustered_bootstrap_mean(values, groups, n_bootstrap=200, seed=7)
    second = clustered_bootstrap_mean(values, groups, n_bootstrap=200, seed=7)
    assert first == second
    assert first.low <= first.estimate <= first.high


def test_counterfactual_error_adds_heldout_signal():
    rng = np.random.default_rng(5)
    n_states = 80
    repeats = 2
    groups = np.repeat(np.arange(n_states), repeats)
    ordinary = rng.normal(size=groups.size)
    counterfactual = rng.normal(size=groups.size)
    outcome = 0.2 * ordinary + 1.4 * counterfactual + rng.normal(
        scale=0.1, size=groups.size
    )
    result = grouped_incremental_validity(
        outcome,
        ordinary,
        counterfactual,
        groups,
        n_splits=5,
        seed=9,
    )
    assert result.delta_r2 > 0.7
    assert result.full_rmse < result.base_rmse

