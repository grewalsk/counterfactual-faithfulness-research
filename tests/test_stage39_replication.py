import numpy as np

from cf_faithfulness.stage39_replication import (
    derive_stage39_decision,
    derive_stage39_panel_decision,
    hierarchical_seed_family_interval,
    paired_rowwise_relative_gain,
    pooled_ratio_of_means_gain,
)


def test_rowwise_and_pooled_estimands_are_distinct_and_exact():
    full = np.array([[1.0, 9.0], [1.0, 9.0]])
    matched = np.array([[2.0, 10.0], [2.0, 10.0]])
    gain = paired_rowwise_relative_gain(full, matched)
    np.testing.assert_allclose(gain, [[0.5, 0.1], [0.5, 0.1]])
    assert pooled_ratio_of_means_gain(full, matched) == 1.0 / 6.0


def test_hierarchical_interval_is_deterministic_and_contains_constant():
    values = np.full((3, 12), 0.0125)
    groups = np.repeat(np.arange(4), 3)
    first = hierarchical_seed_family_interval(values, groups, draws=64, seed=7)
    second = hierarchical_seed_family_interval(values, groups, draws=64, seed=7)
    assert first == second
    np.testing.assert_allclose(first, (0.0125, 0.0125))


def test_equivalence_requires_interval_inside_fixed_band():
    equivalent = derive_stage39_panel_decision(0.0, (-0.02, 0.03))
    crossing = derive_stage39_panel_decision(0.01, (-0.06, 0.03))
    specific = derive_stage39_panel_decision(0.08, (0.05, 0.11))
    assert equivalent.classification == "practically_equivalent"
    assert crossing.classification == "inconclusive"
    assert specific.classification == "full_objective_specificity"


def test_global_decision_is_conjunctive_and_never_pools():
    panels = {
        "jepa": derive_stage39_panel_decision(0.0, (-0.02, 0.03)),
        "dino": derive_stage39_panel_decision(0.01, (-0.01, 0.04)),
    }
    decision = derive_stage39_decision(panels)
    assert decision["status"] == "coefficient_matched_equivalence_replicated"
    assert decision["panels_pooled"] is False
