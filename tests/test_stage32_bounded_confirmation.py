import math

import numpy as np

from cf_faithfulness.stage32_bounded_confirmation import (
    bounded_cosine,
    bounded_swap_closure_rows,
    paired_model_difference_rows,
    state_placebo_advantage,
)


def test_bounded_cosine_fails_closed_and_is_bounded():
    ineligible = bounded_cosine([1.0, 0.0], [1e-5, 0.0], 1e-6)
    assert not ineligible["eligible"]
    assert math.isnan(ineligible["cosine"])
    eligible = bounded_cosine([2.0, 0.0], [1.0, 0.0], 1e-6)
    assert eligible["eligible"]
    assert eligible["cosine"] == 1.0


def test_exact_swap_has_unit_self_and_grounded_cosine():
    rng = np.random.default_rng(32)
    target = rng.normal(size=(24, 3, 4))
    reversal = np.arange(24).reshape(4, 6)[:, ::-1].reshape(-1)
    rows = bounded_swap_closure_rows(target, target[reversal], target, 4, 6)
    assert len(rows) == 4
    assert all(row["self_eligible"] and row["grounded_eligible"] for row in rows)
    assert all(np.isclose(row["self_cosine"], 1.0) for row in rows)
    assert all(np.isclose(row["grounded_cosine"], 1.0) for row in rows)


def test_paired_panel_and_placebo_advantage():
    left = [{
        "record_id": 1, "family_index": 0, "magnitude_index": 0,
        "outcome": 0.2, "grounded_cosine": 0.1,
    }]
    right = [{
        "record_id": 1, "family_index": 0, "magnitude_index": 0,
        "outcome": 0.5, "grounded_cosine": 0.04,
    }]
    paired = paired_model_difference_rows(left, right, ["grounded_cosine"])
    assert np.isclose(paired[0]["outcome"], 0.3)
    assert np.isclose(paired[0]["difference_grounded_cosine"], -0.06)
    rows = state_placebo_advantage(
        [0.4, 0.2], [[0.1, 0.2], [0.0, 0.1]], [7, 7]
    )
    assert len(rows) == 1
    assert np.isclose(rows[0]["primary_minus_median_placebo_improvement"], 0.2)
