import numpy as np

from cf_faithfulness.stage36_predictive_state_closure import (
    rollout_predictive_state_closure,
)
from cf_faithfulness.stage37_semigroup_pscd import (
    Stage37Gates,
    derive_stage37_decision,
    fit_semigroup_predictive_state_closure,
    goal_cost,
    grouped_planner_metrics,
    registered_semigroup_horizons,
    rollout_predictive_state_from_initial,
    select_semigroup_candidate,
    terminal_values,
)


def synthetic_system(seed=37, sequences=32, steps=8):
    rng = np.random.default_rng(seed)
    actions = rng.normal(size=(sequences, steps, 2)).astype(np.float32)
    initial = rng.normal(scale=0.3, size=(sequences, 3)).astype(np.float32)
    targets = np.zeros((sequences, steps, 3), dtype=np.float32)
    state = initial.copy()
    for step in range(steps):
        control = actions[:, step]
        state = np.column_stack([
            0.80 * state[:, 0] + 0.30 * control[:, 0],
            0.65 * state[:, 1] - 0.20 * control[:, 1] + 0.10 * state[:, 0],
            0.75 * state[:, 2] + 0.15 * control[:, 0] * control[:, 1],
        ]).astype(np.float32)
        targets[:, step] = state
    mask = np.ones((sequences, steps), dtype=bool)
    return initial, actions, targets, mask


def test_registered_semigroup_horizons_are_multi_anchor_and_mask_safe():
    _, _, _, mask = synthetic_system(sequences=4)
    mask[0, 6:] = False
    pairs = registered_semigroup_horizons(mask, 2, [2, 4])
    assert (0, 2) in pairs and (0, 4) in pairs
    assert (1, 2) in pairs and (1, 4) in pairs
    assert (6, 2) in pairs
    assert all(anchor >= 0 and horizon in {2, 4} for anchor, horizon in pairs)


def test_semigroup_pscd_trains_and_rolls_out_on_cpu():
    initial, actions, targets, mask = synthetic_system()
    artifact = fit_semigroup_predictive_state_closure(
        initial[:24], actions[:24], targets[:24], targets[:24], mask[:24],
        history_length=2, latent_dim=12, dynamics="single", epochs=30,
        learning_rate=3e-3, seed=3701, semigroup_horizons=[2, 4],
        semigroup_weight=0.5, device="cpu",
    )
    assert artifact["loss_final"] < artifact["loss_initial"]
    assert artifact["config"]["semigroup_anchor_pairs"] > 1
    result = rollout_predictive_state_closure(
        artifact, initial[24:], actions[24:], targets[24:], mask[24:], device="cpu"
    )
    assert result["physical"].shape == targets[24:].shape
    assert np.all(np.isfinite(result["state"]))
    cold = rollout_predictive_state_from_initial(
        artifact, initial[24:], actions[24:], mask[24:], device="cpu"
    )
    assert cold["physical"].shape == targets[24:].shape
    assert np.all(cold["evaluation_mask"] == mask[24:])


def test_semigroup_candidate_selection_is_deterministic():
    selected = select_semigroup_candidate([
        {"semigroup_weight": 2.0, "validation_score": 0.7,
         "recursive_physical_nmse": 0.4, "semigroup_nmse": 0.3},
        {"semigroup_weight": 0.5, "validation_score": 0.7,
         "recursive_physical_nmse": 0.4, "semigroup_nmse": 0.2},
    ])
    assert selected["semigroup_weight"] == 0.5


def test_open_loop_planning_metrics_recover_oracle_and_regret():
    paths = np.asarray([
        [[0.0, 0.0], [1.0, 0.0]],
        [[0.0, 0.0], [2.0, 0.0]],
        [[0.0, 0.0], [0.0, 1.0]],
        [[0.0, 0.0], [0.0, 2.0]],
    ])
    mask = np.ones((4, 2), dtype=bool)
    endpoints = terminal_values(paths, mask)
    goals = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    truth = goal_cost(endpoints, goals, np.ones(2), dimensions=(0, 1))
    metrics = grouped_planner_metrics(truth, truth, np.asarray([0, 0, 1, 1]))
    np.testing.assert_allclose(metrics["regret"], 0.0)
    np.testing.assert_allclose(metrics["success"], 1.0)
    np.testing.assert_allclose(metrics["pairwise_accuracy"], 1.0)


def test_stage37_decision_preserves_claim_boundary():
    passed = derive_stage37_decision(Stage37Gates(*(True,) * 8), run_mode="pilot")
    assert passed["passed"]
    assert passed["status"] == "semigroup_pscd_closure_and_planning_value_observed"
    assert not passed["causal_evidence"]
    assert not passed["closed_loop_planning_claimed"]
    failed = derive_stage37_decision(
        Stage37Gates(True, True, True, True, True, False, True, True),
        run_mode="pilot",
    )
    assert failed["status"] == "closure_gain_did_not_improve_open_loop_planning"
