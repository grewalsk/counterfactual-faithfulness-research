import numpy as np

from cf_faithfulness.stage31_cross_model_certificate import (
    official_native_terminal_costs,
    official_terminal_planning_rows,
    paired_model_difference_rows,
    planner_metric_features,
)


def _states(magnitudes=2, schedules=6):
    states = np.zeros((magnitudes, schedules, 10), dtype=np.float64)
    for magnitude in range(magnitudes):
        states[magnitude, :, 2] = 200 + 5 * magnitude + np.arange(schedules)
        states[magnitude, :, 3] = 260 - np.arange(schedules)
        states[magnitude, :, 4] = 0.02 * np.arange(schedules)
    return states.reshape(magnitudes * schedules, 10)


def test_joint_chart_reproduces_official_cost():
    rng = np.random.default_rng(31)
    visual = rng.normal(size=(12, 3, 4))
    proprio = rng.normal(size=(12, 2, 5))
    chart = planner_metric_features(visual, proprio, alpha=0.1)
    observed = np.sum((chart[0] - chart[1]) ** 2)
    expected = np.mean((visual[0] - visual[1]) ** 2) + 0.1 * np.mean(
        (proprio[0] - proprio[1]) ** 2
    )
    assert np.isclose(observed, expected)


def test_exact_joint_prediction_selects_goal_branch():
    rng = np.random.default_rng(32)
    visual = rng.normal(size=(12, 3, 4))
    proprio = rng.normal(size=(12, 2, 5))
    costs = official_native_terminal_costs(
        visual, visual, proprio, proprio, 2, 6, goal_schedule=5, alpha=0.1
    )
    assert np.all(np.argmin(costs, axis=1) == 5)
    rows = official_terminal_planning_rows(
        visual,
        visual,
        proprio,
        proprio,
        _states(),
        2,
        6,
        alpha=0.1,
    )
    assert all(row["top1_correct"] == 1.0 for row in rows)
    assert all(row["normalized_regret"] == 0.0 for row in rows)


def test_paired_rows_are_right_minus_left_and_require_exact_panel():
    left = [
        {
            "record_id": 1,
            "magnitude_index": 0,
            "regime": "persistent_contact",
            "outcome": 0.2,
            "grounded_coefficient": 0.1,
        }
    ]
    right = [
        {
            "record_id": 1,
            "magnitude_index": 0,
            "regime": "persistent_contact",
            "outcome": 0.5,
            "grounded_coefficient": 0.04,
        }
    ]
    rows = paired_model_difference_rows(
        left, right, ["grounded_coefficient"]
    )
    assert np.isclose(rows[0]["outcome"], 0.3)
    assert np.isclose(rows[0]["difference_grounded_coefficient"], -0.06)
