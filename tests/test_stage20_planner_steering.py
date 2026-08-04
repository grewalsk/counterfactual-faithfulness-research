import numpy as np
import pytest

from cf_faithfulness.stage20_planner_steering import (
    planner_steering_metrics,
    select_near_frontier_targets,
    stable_action_rank,
    targeted_derangement,
)


def test_targeted_derangement_is_deterministic_and_assigns_donor():
    first = targeted_derangement(13, target=7, donor=2, seed=2001)
    second = targeted_derangement(13, target=7, donor=2, seed=2001)
    assert np.array_equal(first, second)
    assert first[7] == 2
    assert sorted(first.tolist()) == list(range(13))
    assert np.all(first != np.arange(13))


def test_targeted_derangement_rejects_target_equal_to_donor():
    with pytest.raises(ValueError, match="must differ"):
        targeted_derangement(13, target=3, donor=3, seed=2003)


def test_near_frontier_targets_use_only_baseline_ranking():
    scores = np.asarray([0.8, 0.1, 0.4, 0.2, 0.3])
    donor, targets = select_near_frontier_targets(scores, ranks=(1, 2, 3))
    assert donor == 1
    assert targets == [3, 4, 2]


def test_stable_action_rank_breaks_ties_by_index():
    assert stable_action_rank([0.0, 1.0, 1.0, 2.0], 1) == 1
    assert stable_action_rank([0.0, 1.0, 1.0, 2.0], 2) == 2


def test_complete_counterfactual_swap_has_exact_choice_and_physical_metrics():
    baseline = np.asarray([0.9, 0.1, 0.4, 0.7, 0.3])
    physical = np.asarray([0.8, 0.2, 0.5, 0.9, 0.4])
    target = 4
    permutation = targeted_derangement(5, target=target, donor=1, seed=2005)
    patched = baseline[permutation]
    result = planner_steering_metrics(
        baseline, patched, physical, permutation, target
    )
    assert result["target_is_expected_choice"]
    assert result["target_selected"]
    assert result["choice_matches_counterfactual"]
    assert result["score_transfer_coefficient"] > 1 - 1e-12
    assert result["score_counterfactual_normalized_rmse"] < 1e-12
    assert result["patched_selected_true_cost"] == physical[target]
