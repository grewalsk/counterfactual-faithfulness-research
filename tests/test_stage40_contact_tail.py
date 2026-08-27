import numpy as np

from cf_faithfulness.stage40_contact_tail import (
    contact_transition_weights,
    derive_stage40_decision,
    derive_stage40_panel_decision,
    select_contact_risk_candidate,
)


def test_contact_weights_are_uniform_at_one_and_upweight_reentry():
    initial = np.array(["free", "post_contact"])
    target = np.array([
        ["pre_contact", "contact", "post_contact"],
        ["contact", "post_contact", ""],
    ])
    mask = np.array([[True, True, True], [True, True, False]])
    uniform = contact_transition_weights(
        initial, target, mask, contact_multiplier=1.0
    )
    np.testing.assert_allclose(uniform[mask], 1.0)
    assert np.all(uniform[~mask] == 0)
    weighted = contact_transition_weights(
        initial, target, mask, contact_multiplier=4.0
    )
    assert np.isclose(np.mean(weighted[mask]), 1.0)
    assert weighted[1, 0] > weighted[0, 1] > weighted[0, 0]


def test_selection_optimizes_p95_subject_to_mean_noninferiority():
    rows = [
        {
            "contact_multiplier": 1.0, "mean_nmse": 0.20,
            "p95_nmse": 0.60, "terminal_contact_nmse": 0.80,
            "catastrophic_rate": 0.01,
        },
        {
            "contact_multiplier": 2.0, "mean_nmse": 0.205,
            "p95_nmse": 0.45, "terminal_contact_nmse": 0.60,
            "catastrophic_rate": 0.01,
        },
        {
            "contact_multiplier": 4.0, "mean_nmse": 0.23,
            "p95_nmse": 0.30, "terminal_contact_nmse": 0.40,
            "catastrophic_rate": 0.01,
        },
    ]
    selected = select_contact_risk_candidate(rows, max_mean_ratio=1.05)
    assert selected["contact_multiplier"] == 2.0
    assert selected["mean_noninferior_to_uniform"] is True


def test_stage40_decision_separates_improvement_from_absolute_qualification():
    partial = derive_stage40_panel_decision(
        0.02, (-0.01, 0.05), p95_improvement=True,
        contact_improvement=True, absolute_tail_qualified=False,
    )
    complete = derive_stage40_panel_decision(
        0.03, (0.0, 0.06), p95_improvement=True,
        contact_improvement=True, absolute_tail_qualified=True,
    )
    assert partial.classification == "tail_improved_but_not_absolutely_qualified"
    assert complete.classification == "contact_tail_repair_confirmed"
    global_result = derive_stage40_decision({"jepa": complete, "dino": partial})
    assert global_result["status"] == "cross_model_tail_improved_but_not_qualified"
    assert global_result["passed"] is False
