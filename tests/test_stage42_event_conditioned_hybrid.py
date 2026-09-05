import numpy as np
import pytest

from cf_faithfulness.stage42_event_conditioned_hybrid import (
    binary_auroc,
    derive_stage42_defect_decision,
    derive_stage42_support_decision,
    family_reentry_count,
    guard_probe_metrics,
    macro_contact_flags,
    partition_hybrid_defects,
    propagated_hybrid_error_bound,
    select_event_rich_families,
    select_guard_threshold,
    terminal_mode_pair_from_contacts,
)


def test_macro_contact_flags_and_terminal_reentry_pair():
    trace = np.asarray([1, 0, 0, 0, 0, 0, 1, 0], dtype=np.int64)
    np.testing.assert_array_equal(
        macro_contact_flags(trace, length=4, frameskip=2),
        [True, False, False, True],
    )
    assert terminal_mode_pair_from_contacts(
        trace, length=4, frameskip=2, initial_mode="free"
    ) == ("post_contact", "contact")


def test_family_reentry_count_aggregates_records_and_words():
    reentry = np.asarray([1, 0, 0, 0, 0, 0, 1, 0])
    ordinary = np.asarray([0, 0, 0, 0, 0, 0, 0, 0])
    panels = [np.stack([reentry, ordinary]), np.stack([ordinary, reentry])]
    assert family_reentry_count(
        panels, [4, 4], ["free", "post_contact"], frameskip=2
    ) == 2


def test_event_rich_selection_is_earliest_and_never_uses_outcomes():
    result = select_event_rich_families(
        [10, 11, 12, 13, 14], [0, 2, 1, 0, 4],
        target_families=3, minimum_total_rows=7,
    )
    assert result["trajectory_ids"] == [11, 12, 14]
    assert result["total_reentry_rows"] == 7
    decision = derive_stage42_support_decision(
        result, expected_families=3, minimum_reentry_rows=7
    )
    assert decision.passed


def test_event_rich_selection_fails_closed_on_insufficient_support():
    with pytest.raises(
        RuntimeError,
        match=r"found 1 of 2; total qualifying rows=1",
    ):
        select_event_rich_families(
            [1, 2, 3], [0, 1, 0], target_families=2, minimum_total_rows=2
        )


def test_guard_probe_metrics_and_development_threshold():
    target = np.asarray([0, 0, 1, 1])
    score = np.asarray([0.1, 0.3, 0.7, 0.9])
    assert binary_auroc(target, score) == pytest.approx(1.0)
    selected = select_guard_threshold(target, score, [0.25, 0.5, 0.75])
    assert selected["selected_threshold"] == pytest.approx(0.5)
    metrics = guard_probe_metrics(
        target, score, [0.0, 0.0, 0.25, 0.75], [0.0, 0.0, 0.20, 0.80],
        threshold=selected["selected_threshold"],
    )
    assert metrics["event_auroc"] == pytest.approx(1.0)
    assert metrics["event_balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["event_time_mae"] == pytest.approx(0.05)


def test_hybrid_defects_and_propagation_bound():
    result = partition_hybrid_defects(
        [0.1, 0.2, 0.8, 0.6], [0.2, 0.3, 1.0, 2.0],
        [0.2, 0.3, 0.5, 1.0], [False, False, True, True],
        [False, False, True, True],
    )
    assert result["flow_composition_nmse"] == pytest.approx(0.15)
    assert result["reset_oracle_relative_gain"] == pytest.approx(0.5)
    assert propagated_hybrid_error_bound([1.0, 2.0, 3.0], [9.0, 2.0, 4.0]) == 19.0


def test_stage42_decision_does_not_conflate_oracle_and_guard_evidence():
    blocked = derive_stage42_defect_decision(
        support_certified=True, oracle_reset_headroom=True,
        frozen_guard_identifiable=False,
    )
    assert not blocked.passed
    assert blocked.classification == "oracle_headroom_but_guard_information_insufficient"
    passed = derive_stage42_defect_decision(
        support_certified=True, oracle_reset_headroom=True,
        frozen_guard_identifiable=True,
    )
    assert passed.learned_event_reset_experiment_authorized
