import pytest

from cf_faithfulness.stage37_1_horizon_control import (
    Stage371Gates,
    derive_stage371_decision,
    select_horizon_control_candidate,
)


def test_candidate_selection_is_score_first_and_complexity_tiebroken():
    selected = select_horizon_control_candidate([
        {"latent_dim": 256, "dynamics": "mixture", "physical_nmse": 0.1,
         "semigroup_nmse": 0.1, "validation_score": 0.2},
        {"latent_dim": 128, "dynamics": "single", "physical_nmse": 0.1,
         "semigroup_nmse": 0.1, "validation_score": 0.2},
    ])
    assert selected["latent_dim"] == 128
    assert selected["dynamics"] == "single"
    with pytest.raises(ValueError, match="incomplete"):
        select_horizon_control_candidate([{"latent_dim": 1}])


def test_decision_requires_every_calibration_gate():
    passed = derive_stage371_decision(Stage371Gates(*(True,) * 7), run_mode="pilot")
    assert passed["passed"]
    assert passed["authorizes_fresh_jepa_confirmation"]
    assert not passed["jepa_loaded"]
    failed = derive_stage371_decision(
        Stage371Gates(True, True, True, False, True, True, True),
        run_mode="pilot",
    )
    assert failed["status"] == "operator_failed_locked_semigroup_closure"
    assert not failed["authorizes_fresh_jepa_confirmation"]
