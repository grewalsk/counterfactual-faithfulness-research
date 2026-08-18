import numpy as np

from cf_faithfulness.stage36_predictive_state_closure import (
    Stage36Gates,
    derive_stage36_decision,
    fit_predictive_state_closure,
    history_tensor,
    next_history_tensor,
    permute_past_history,
    rollout_evaluation_mask,
    rollout_predictive_state_closure,
    select_pscd_candidate,
)


def synthetic_delayed_system(seed=36, sequences=64, steps=7):
    rng = np.random.default_rng(seed)
    action = rng.normal(size=(sequences, steps, 1)).astype(np.float32)
    initial = rng.normal(scale=0.5, size=(sequences, 2)).astype(np.float32)
    carrier = np.zeros((sequences, steps, 2), dtype=np.float32)
    physical = np.zeros((sequences, steps, 2), dtype=np.float32)
    state = initial.copy()
    previous_action = np.zeros(sequences, dtype=np.float32)
    for step in range(steps):
        value = action[:, step, 0]
        state = np.column_stack([
            0.75 * state[:, 0] + 0.35 * value + 0.2 * previous_action,
            0.6 * state[:, 1] - 0.25 * value + 0.15 * state[:, 0],
        ]).astype(np.float32)
        carrier[:, step] = state
        physical[:, step] = state
        previous_action = value
    mask = np.ones((sequences, steps), dtype=bool)
    groups = np.repeat(np.arange(sequences // 4), 4)
    return initial, action, carrier, physical, mask, groups


def test_history_construction_and_control_keep_current_slot():
    initial, _, carrier, _, mask, groups = synthetic_delayed_system(sequences=16)
    history = history_tensor(initial, carrier, mask, 3)
    future = next_history_tensor(history, carrier)
    assert history.shape == (16, 7, 3, 2)
    np.testing.assert_allclose(history[:, 0], np.repeat(initial[:, None, :], 3, axis=1))
    np.testing.assert_allclose(future[:, :, -1], carrier)
    permuted = permute_past_history(history, groups, mask, seed=7)
    np.testing.assert_allclose(permuted[:, :, -1], history[:, :, -1])
    assert np.any(permuted[:, :, 0] != history[:, :, 0])
    evaluation = rollout_evaluation_mask(mask, 3)
    assert not np.any(evaluation[:, :2]) and np.all(evaluation[:, 2:])


def test_small_predictive_state_model_trains_and_rolls_out_on_cpu():
    initial, action, carrier, physical, mask, _ = synthetic_delayed_system()
    artifact = fit_predictive_state_closure(
        initial[:48], action[:48], carrier[:48], physical[:48], mask[:48],
        history_length=2, latent_dim=12, dynamics="single", epochs=45,
        learning_rate=3e-3, seed=36, device="cpu",
    )
    assert artifact["loss_final"] < artifact["loss_initial"]
    result = rollout_predictive_state_closure(
        artifact, initial[48:], action[48:], carrier[48:], mask[48:], device="cpu"
    )
    assert result["carrier"].shape == carrier[48:].shape
    assert result["physical"].shape == physical[48:].shape
    assert np.all(np.isfinite(result["carrier"]))
    assert np.all(result["evaluation_mask"][:, 1:])


def test_candidate_selection_is_deterministic_and_complexity_tiebroken():
    rows = [
        {"carrier_dim": 1024, "history_length": 4, "latent_dim": 128,
         "dynamics": "mixture", "validation_score": 1.0,
         "recursive_physical_nmse": 0.5, "semigroup_nmse": 0.2},
        {"carrier_dim": 256, "history_length": 2, "latent_dim": 64,
         "dynamics": "single", "validation_score": 1.0,
         "recursive_physical_nmse": 0.5, "semigroup_nmse": 0.2},
    ]
    selected = select_pscd_candidate(rows)
    assert selected["carrier_dim"] == 256
    assert selected["history_length"] == 2


def test_stage36_decision_has_strict_adapter_only_boundary():
    passed = derive_stage36_decision(Stage36Gates(*(True,) * 8), run_mode="pilot")
    assert passed["status"] == "bounded_predictive_state_closure_distilled"
    assert passed["passed"] and passed["adapter_distillation_only"]
    assert not passed["causal_evidence"]
    assert not passed["original_jepa_state_claimed_closed"]
    failed = derive_stage36_decision(
        Stage36Gates(True, True, True, True, False, True, True, True),
        run_mode="pilot",
    )
    assert failed["status"] == "adapter_did_not_improve_closure"
    assert not failed["passed"]
