import numpy as np
import pytest

from cf_faithfulness.stage36_predictive_state_closure import (
    rollout_predictive_state_closure,
)
from cf_faithfulness.stage38_cross_model_pscd import (
    Stage38Gates,
    Stage38ModelGates,
    derive_stage38_decision,
    derive_stage38_model_decision,
    fit_weighted_semigroup_predictive_state_closure,
    hierarchical_seed_trajectory_interval,
    select_stage38_semigroup_candidate,
    tail_risk_summary,
)


def synthetic_system(seed=38, sequences=24, steps=9):
    rng = np.random.default_rng(seed)
    actions = rng.normal(size=(sequences, steps, 3)).astype(np.float32)
    initial = rng.normal(scale=0.2, size=(sequences, 4)).astype(np.float32)
    targets = np.zeros((sequences, steps, 4), dtype=np.float32)
    state = initial.copy()
    for step in range(steps):
        action = actions[:, step]
        state = np.column_stack([
            0.8 * state[:, 0] + 0.2 * action[:, 0],
            0.7 * state[:, 1] - 0.2 * action[:, 1],
            0.6 * state[:, 2] + 0.1 * state[:, 0] + 0.1 * action[:, 2],
            0.75 * state[:, 3] + 0.1 * action[:, 0] * action[:, 1],
        ]).astype(np.float32)
        targets[:, step] = state
    return initial, actions, targets, np.ones((sequences, steps), dtype=bool)


def test_weighted_semigroup_fit_supports_full_and_latent_only_controls():
    initial, actions, targets, mask = synthetic_system()
    for weights in [(0.35, 0.20, 0.45), (0.0, 0.0, 1.0)]:
        artifact = fit_weighted_semigroup_predictive_state_closure(
            initial, actions, targets, targets, mask,
            history_length=2, latent_dim=12, dynamics="single", epochs=20,
            learning_rate=3e-3, seed=3801,
            semigroup_horizons=[2, 4, 8], semigroup_weight=0.5,
            semigroup_component_weights=weights, device="cpu",
        )
        assert artifact["loss_final"] < artifact["loss_initial"]
        assert artifact["config"]["semigroup_component_weights"] == list(weights)
        result = rollout_predictive_state_closure(
            artifact, initial, actions, targets, mask, device="cpu"
        )
        assert np.all(np.isfinite(result["physical"]))


def test_stage38_selection_tail_and_hierarchical_interval():
    selected = select_stage38_semigroup_candidate([
        {"semigroup_weight": 1.0, "physical_nmse": 0.2,
         "semigroup_nmse": 0.1, "score": 0.3},
        {"semigroup_weight": 0.5, "physical_nmse": 0.1,
         "semigroup_nmse": 0.1, "score": 0.2},
    ])
    assert selected["semigroup_weight"] == 0.5
    with pytest.raises(ValueError, match="incomplete"):
        select_stage38_semigroup_candidate([{"score": 1.0}])
    tail = tail_risk_summary([0.0, 0.1, 0.2, 2.0])
    assert tail["maximum"] == 2.0
    assert tail["catastrophic_rate_gt_1"] == 0.25
    values = np.asarray([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]])
    interval = hierarchical_seed_trajectory_interval(
        values, [0, 0, 1, 1], draws=100, seed=38
    )
    assert interval[0] <= np.mean(values) <= interval[1]


def test_stage38_decisions_preserve_claim_boundary():
    model = derive_stage38_model_decision(Stage38ModelGates(*(True,) * 8))
    assert model["passed"]
    failed_model = derive_stage38_model_decision(
        Stage38ModelGates(True, True, False, True, True, True, True, True)
    )
    assert failed_model["first_failed_gate"] == "repair_advantage"
    full = derive_stage38_decision(Stage38Gates(*(True,) * 5), run_mode="pilot")
    assert full["passed"] and full["planning_confirmed"]
    closure_only = derive_stage38_decision(
        Stage38Gates(True, True, True, True, False), run_mode="pilot"
    )
    assert closure_only["passed"] and closure_only["closure_confirmed"]
    assert not closure_only["native_jepa_closure_claimed"]
