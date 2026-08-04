import numpy as np

from cf_faithfulness.stage17_action_contrast import (
    action_swap_delta,
    candidate_center,
    decoded_task_cost,
    donor_transfer_metrics,
    exact_positive_sign_test,
    fit_dual_ridge_basis,
    fixed_derangement,
    grouped_kernel_ridge_cv,
    linear_cka,
    matched_common_mode,
    pose_target,
    random_subspace_in_span,
)


def test_candidate_quotient_and_full_swap_identity():
    rng = np.random.default_rng(17)
    values = rng.normal(size=(7, 3, 5))
    permutation = fixed_derangement(len(values), 19)
    assert np.all(permutation != np.arange(len(values)))
    residual = candidate_center(values)
    assert np.allclose(np.mean(residual, axis=0), 0.0, atol=1e-12)
    edited = values + action_swap_delta(values, permutation)
    assert np.allclose(edited, values[permutation], atol=1e-12)


def test_partial_swap_and_common_mode_are_energy_matched():
    rng = np.random.default_rng(23)
    values = rng.normal(size=(6, 4))
    basis, _ = np.linalg.qr(rng.normal(size=(4, 2)))
    permutation = fixed_derangement(len(values), 29)
    delta = action_swap_delta(values, permutation, basis=basis, dose=0.5)
    common = matched_common_mode(delta, rng.normal(size=4))
    assert np.isclose(np.linalg.norm(delta), np.linalg.norm(common))
    assert np.allclose(common, common[:1])


def test_donor_metrics_have_zero_and_one_anchors():
    rng = np.random.default_rng(31)
    baseline = rng.normal(size=(8, 12))
    permutation = fixed_derangement(len(baseline), 37)
    zero = donor_transfer_metrics(baseline, baseline, permutation)
    perfect = donor_transfer_metrics(baseline, baseline[permutation], permutation)
    assert zero["coefficient"] == 0.0
    assert zero["reconstruction"] == 0.0
    assert np.isclose(perfect["coefficient"], 1.0)
    assert np.isclose(perfect["cosine"], 1.0)
    assert np.isclose(perfect["reconstruction"], 1.0)


def test_cka_is_rotation_invariant_and_detects_shuffle():
    rng = np.random.default_rng(41)
    values = rng.normal(size=(13, 9))
    rotation, _ = np.linalg.qr(rng.normal(size=(9, 9)))
    assert np.isclose(linear_cka(values, values @ rotation), 1.0)
    shuffled = values[fixed_derangement(len(values), 43)]
    assert linear_cka(values, shuffled) < 0.8


def test_grouped_kernel_ridge_and_input_basis_recover_signal():
    rng = np.random.default_rng(47)
    features = rng.normal(size=(80, 20))
    latent = features[:, :3]
    targets = latent @ rng.normal(size=(3, 6)) + 0.01 * rng.normal(size=(80, 6))
    groups = np.repeat(np.arange(8), 10)
    kernel = features @ features.T
    selection = grouped_kernel_ridge_cv(kernel, targets, groups, [1e-6, 1e-3, 1.0])
    fitted = fit_dual_ridge_basis(
        features, targets, selection["penalty"], max_rank=6
    )
    assert fitted["basis"].shape == (20, 6)
    assert np.allclose(fitted["basis"].T @ fitted["basis"], np.eye(6), atol=1e-10)
    assert selection["selected_multiplier"] != 1.0


def test_random_control_is_in_span_and_orthogonal_to_primary():
    rng = np.random.default_rng(53)
    features = rng.normal(size=(30, 50))
    primary, _ = np.linalg.qr(features.T @ rng.normal(size=(30, 4)))
    control = random_subspace_in_span(
        features, rank=5, seed=59, orthogonal_to=primary[:, :4]
    )
    projection_residual = control - features.T @ np.linalg.lstsq(
        features.T, control, rcond=None
    )[0]
    assert np.linalg.norm(projection_residual) < 1e-10
    assert np.linalg.norm(primary[:, :4].T @ control) < 1e-10
    assert np.allclose(control.T @ control, np.eye(5), atol=1e-10)


def test_pose_cost_and_sign_test():
    states = np.asarray(
        [[0.0, 0.0, 256.0, 128.0, 0.0], [0.0, 0.0, 512.0, 512.0, np.pi / 2]]
    )
    poses = pose_target(states)
    assert np.allclose(poses[0], [0.5, 0.25, 0.0, 1.0])
    assert decoded_task_cost(poses[:1], [256.0, 128.0, 0.0])[0] == 0.0
    result = exact_positive_sign_test([1, 2, 3, -1])
    assert result == {"positive": 3, "nonzero": 4, "p_value": 0.3125}


def test_synthetic_finite_contrast_mediator_recovers_donor_transfer():
    rng = np.random.default_rng(61)
    candidates, dimension, latent_dimension, output_dimension = 7, 40, 3, 8
    true_basis, _ = np.linalg.qr(rng.normal(size=(dimension, latent_dimension)))
    output_map = rng.normal(size=(latent_dimension, output_dimension))
    feature_rows, output_rows, groups = [], [], []
    for group in range(8):
        latent = candidate_center(rng.normal(size=(candidates, latent_dimension)))
        shared = rng.normal(size=(1, dimension))
        features = shared + latent @ true_basis.T
        outputs = latent @ output_map
        feature_rows.append(candidate_center(features))
        output_rows.append(candidate_center(outputs))
        groups.extend([group] * candidates)
    x = np.concatenate(feature_rows)
    y = np.concatenate(output_rows)
    selection = grouped_kernel_ridge_cv(
        x @ x.T, y, np.asarray(groups), [1e-8, 1e-5, 1e-2]
    )
    fitted = fit_dual_ridge_basis(x, y, selection["penalty"], max_rank=3)

    heldout_latent = candidate_center(
        rng.normal(size=(candidates, latent_dimension))
    )
    heldout_features = rng.normal(size=(1, dimension)) + heldout_latent @ true_basis.T
    baseline_output = heldout_latent @ output_map
    permutation = fixed_derangement(candidates, 67)
    delta = action_swap_delta(
        heldout_features, permutation, basis=fitted["basis"], dose=1.0
    )
    patched_output = candidate_center(heldout_features + delta) @ true_basis @ output_map
    transfer = donor_transfer_metrics(baseline_output, patched_output, permutation)
    assert transfer["coefficient"] > 0.99
    assert transfer["cosine"] > 0.99
