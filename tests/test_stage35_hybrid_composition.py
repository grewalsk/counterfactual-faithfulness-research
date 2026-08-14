import numpy as np

from cf_faithfulness.stage35_hybrid_composition import (
    Stage35Gates,
    derive_stage35_decision,
    fit_family_from_sequences,
    fit_rff_ridge,
    permuted_sequence_labels,
    predict_rff_ridge,
    recursive_rollout,
    scaled_sequence_mse,
    select_guard_hyperparameters,
    time_shifted_sequence_labels,
    transition_labels,
)


def synthetic_hybrid_sequences(seed=3501, sequences=120, steps=4):
    rng = np.random.default_rng(seed)
    actions = rng.choice([-1.0, 1.0], size=(sequences, steps, 1))
    initial = rng.normal(scale=0.4, size=(sequences, 2))
    mask = np.ones((sequences, steps), dtype=bool)
    source_modes = np.empty((sequences, steps), dtype="<U16")
    target_modes = np.empty((sequences, steps), dtype="<U16")
    targets = np.empty((sequences, steps, 2), dtype=np.float64)
    state = initial.copy()
    mode = np.where(state[:, 0] > 0.2, "contact", "free")
    for step in range(steps):
        source_modes[:, step] = mode
        crossing = (state[:, 0] + 0.55 * actions[:, step, 0]) > 0.2
        next_mode = np.where(crossing, "contact", "free")
        sign = np.where(next_mode == "contact", -1.0, 1.0)
        state = np.column_stack([
            state[:, 0] + 0.55 * actions[:, step, 0],
            0.75 * sign * state[:, 1] + 0.25 * actions[:, step, 0],
        ])
        state[:, 1] += np.where((mode == "free") & (next_mode == "contact"), 1.2, 0.0)
        targets[:, step] = state
        target_modes[:, step] = next_mode
        mode = next_mode
    return initial, actions, targets, mask, source_modes, target_modes


def test_transition_controls_preserve_shapes_and_multisets():
    _, _, _, mask, source, target = synthetic_hybrid_sequences(sequences=12)
    labels = transition_labels(source, target)
    shifted = time_shifted_sequence_labels(labels, mask)
    groups = np.repeat(np.arange(3), 4)
    permuted = permuted_sequence_labels(labels, mask, groups, seed=3)
    assert shifted.shape == labels.shape == permuted.shape
    for group in np.unique(groups):
        selected = groups == group
        assert sorted(permuted[selected][mask[selected]].tolist()) == sorted(
            labels[selected][mask[selected]].tolist()
        )


def test_rff_ridge_fits_smooth_nonlinear_map():
    rng = np.random.default_rng(35)
    x = rng.normal(size=(500, 3))
    y = np.column_stack([np.sin(x[:, 0]) + x[:, 1], x[:, 2] ** 2])
    model = fit_rff_ridge(x[:400], y[:400], width=256, penalty=1e-3, seed=9)
    prediction = predict_rff_ridge(model, x[400:])
    assert np.mean((prediction - y[400:]) ** 2) < 0.12


def test_predicted_guard_recursively_tracks_hybrid_sequences():
    initial, actions, targets, mask, source, target = synthetic_hybrid_sequences()
    family = fit_family_from_sequences(
        initial[:90], actions[:90], targets[:90], mask[:90], source[:90], target[:90],
        width=256, penalty=1e-3, seed=17,
    )
    prediction = recursive_rollout(
        family, initial[90:], actions[90:], mask[90:], strategy="predicted_guard"
    )
    fixed = recursive_rollout(
        family, initial[90:], actions[90:], mask[90:], strategy="fixed_source",
        source_modes=source[90:],
    )
    scale = np.std(targets[:90].reshape(-1, 2), axis=0, ddof=1)
    primary_error = scaled_sequence_mse(prediction, targets[90:], mask[90:], scale)
    fixed_error = scaled_sequence_mse(fixed, targets[90:], mask[90:], scale)
    assert np.mean(primary_error) < np.mean(fixed_error)


def test_selection_never_needs_evaluation_data():
    initial, actions, targets, mask, source, target = synthetic_hybrid_sequences(
        sequences=48, steps=3
    )
    groups = np.repeat(np.arange(12), 4)
    selected, rows = select_guard_hyperparameters(
        initial, actions, targets, mask, source, target, groups,
        widths=[32, 64], penalties=[1e-3, 1e-1], folds=3, seed=51,
    )
    assert selected["width"] in {32, 64}
    assert selected["penalty"] in {1e-3, 1e-1}
    assert len(rows) == 4
    assert all(np.isfinite(row["oof_recursive_mse"]) for row in rows)


def test_stage35_decision_has_strict_noncausal_boundary():
    passed = derive_stage35_decision(Stage35Gates(*(True,) * 7), run_mode="pilot")
    assert passed["status"] == "bounded_distributed_hybrid_closure_supported"
    assert passed["passed"]
    assert not passed["causal_evidence"]
    assert not passed["shared_cross_model_mechanism_claimed"]
    assert not passed["low_dimensional_state_claimed"]

    failed = derive_stage35_decision(
        Stage35Gates(True, True, True, False, True, True, True), run_mode="pilot"
    )
    assert failed["status"] == "guard_reset_structure_did_not_transfer"
    assert not failed["passed"]
