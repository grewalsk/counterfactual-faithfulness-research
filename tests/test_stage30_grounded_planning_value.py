import numpy as np

from cf_faithfulness.stage30_grounded_planning_value import (
    cross_fitted_incremental_value,
    diagnostic_closure_rows,
    deterministic_group_folds,
    native_terminal_costs,
    physical_terminal_costs,
    terminal_planning_rows,
)


def _toy_states(magnitudes=2, schedules=6):
    states = np.zeros((magnitudes, schedules, 10), dtype=np.float64)
    for magnitude in range(magnitudes):
        states[magnitude, :, 2] = 240 + 8 * magnitude + np.arange(schedules)
        states[magnitude, :, 3] = 255 - np.arange(schedules)
        states[magnitude, :, 4] = 0.03 * np.arange(schedules)
    return states.reshape(magnitudes * schedules, 10)


def test_exact_prediction_selects_exact_encoded_goal():
    rng = np.random.default_rng(30)
    target = rng.normal(size=(12, 3, 4))
    states = _toy_states()
    costs = native_terminal_costs(target, target, 2, 6, goal_schedule=5)
    assert costs.shape == (2, 6)
    assert np.all(np.argmin(costs, axis=1) == 5)
    rows = terminal_planning_rows(target, target, states, 2, 6)
    assert len(rows) == 4
    assert all(row["top1_correct"] == 1.0 for row in rows)
    assert all(row["normalized_regret"] == 0.0 for row in rows)


def test_physical_goal_cost_is_zero_for_goal_branch():
    states = _toy_states()
    costs = physical_terminal_costs(states, 2, 6, goal_schedule=0)
    assert costs.shape == (2, 6)
    assert np.allclose(costs[:, 0], 0.0)
    assert np.all(costs[:, 1:] > 0.0)


def test_self_consistent_residual_has_lower_grounded_closure():
    target = np.zeros((6, 2), dtype=np.float64)
    target[:, 0] = np.asarray([3, 2, 1, -1, -2, -3])
    residual = np.zeros_like(target)
    residual[:, 1] = np.asarray([6, -5, 4, -4, 5, -6])
    baseline = target + residual
    reversal = np.arange(5, -1, -1)
    patched = baseline[reversal]
    row = diagnostic_closure_rows(
        baseline, patched, target, 1, 6, mode="swap"
    )[0]
    assert np.isclose(row["self_coefficient"], 1.0)
    assert np.isclose(row["self_cosine"], 1.0)
    assert row["grounded_cosine"] < 0.5


def test_exact_grounded_ablation_has_unit_closure():
    target = np.arange(24, dtype=np.float64).reshape(6, 4)
    reversal = np.arange(5, -1, -1)
    component = 0.5 * (target - target[reversal])
    patched = target - component
    row = diagnostic_closure_rows(
        target, patched, target, 1, 6, mode="ablation"
    )[0]
    assert np.isclose(row["self_coefficient"], 1.0)
    assert np.isclose(row["grounded_coefficient"], 1.0)
    assert np.isclose(row["grounded_normalized_rmse"], 0.0)


def test_group_folds_never_split_groups_and_are_deterministic():
    groups = np.repeat(np.arange(20), 3)
    left = deterministic_group_folds(groups, 5, 7)
    right = deterministic_group_folds(groups, 5, 7)
    assert np.array_equal(left, right)
    for group in np.unique(groups):
        assert len(np.unique(left[groups == group])) == 1
    assert set(left.tolist()) == set(range(5))


def test_cross_fitted_grounded_feature_adds_out_of_sample_value():
    rng = np.random.default_rng(31)
    groups = np.repeat(np.arange(80), 2)
    grounded = rng.normal(size=(len(groups), 1))
    base = rng.normal(size=(len(groups), 2))
    outcome = 1.7 * grounded[:, 0] + 0.05 * rng.normal(size=len(groups))
    result = cross_fitted_incremental_value(
        outcome, groups, base, grounded, folds=5, seed=11
    )
    assert result["relative_mse_improvement"] > 0.9
    assert result["grounded_oof_r_squared"] > 0.95
    assert len(result["group_rows"]) == 80
