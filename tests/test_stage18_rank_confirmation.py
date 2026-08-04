import numpy as np

from cf_faithfulness.stage18_rank_confirmation import (
    action_contrast_energy_metrics,
    action_swap_delta,
    candidate_center,
    donor_transfer_metrics,
    fit_dual_ridge_basis,
    fixed_derangement,
    lower_triangle_principal_overlap,
    nested_orthonormalize_basis,
    physical_diversity_metrics,
    projection_ablation_delta,
)


def test_nested_orthonormalization_repairs_float_drift_and_preserves_prefixes():
    rng = np.random.default_rng(1800)
    orthonormal, _ = np.linalg.qr(rng.normal(size=(80, 12)))
    upper = np.eye(12) + np.triu(rng.normal(scale=2e-5, size=(12, 12)))
    drifted = orthonormal @ upper
    stored = drifted.astype(np.float32)
    repaired = nested_orthonormalize_basis(stored)
    assert np.max(np.abs(repaired.T @ repaired - np.eye(12))) < 1e-10
    for rank in [1, 4, 8, 12]:
        left, _ = np.linalg.qr(stored[:, :rank].astype(np.float64))
        assert np.allclose(
            repaired[:, :rank] @ repaired[:, :rank].T,
            left @ left.T,
            atol=1e-9,
            rtol=1e-9,
        )


def test_nested_orthonormalization_rejects_malformed_basis():
    malformed = np.eye(8)
    malformed[:, 1] = malformed[:, 0]
    try:
        nested_orthonormalize_basis(malformed)
    except ValueError as error:
        assert "small numerical perturbation" in str(error)
    else:
        raise AssertionError("rank-deficient basis was accepted")


def test_projection_ablation_removes_only_selected_action_contrast():
    rng = np.random.default_rng(1801)
    values = rng.normal(size=(13, 20))
    basis, _ = np.linalg.qr(rng.normal(size=(20, 6)))
    shared_before = np.mean(values, axis=0)
    edited = values + projection_ablation_delta(values, basis)
    assert np.allclose(np.mean(edited, axis=0), shared_before, atol=1e-12)
    assert np.linalg.norm(candidate_center(edited) @ basis) < 1e-10


def test_action_contrast_energy_has_zero_and_identity_anchors():
    rng = np.random.default_rng(1803)
    baseline = rng.normal(size=(13, 17))
    identity = action_contrast_energy_metrics(baseline, baseline)
    collapsed = np.broadcast_to(np.mean(baseline, axis=0), baseline.shape)
    removed = action_contrast_energy_metrics(baseline, collapsed)
    assert np.isclose(identity["energy_retention"], 1.0)
    assert np.isclose(identity["energy_reduction"], 0.0)
    assert np.isclose(removed["energy_retention"], 0.0)
    assert np.isclose(removed["energy_reduction"], 1.0)


def test_physical_diversity_counts_contacts_and_non_ties():
    result = physical_diversity_metrics(
        [0.0, 0.0, 0.2, 0.5], [0, 3, 0, 2], tie=1e-3
    )
    assert result["cost_spread"] == 0.5
    assert result["contact_branches"] == 2
    assert np.isclose(result["non_tied_pair_fraction"], 5 / 6)


def test_principal_overlap_extremes():
    identity = np.eye(8)
    assert np.isclose(
        lower_triangle_principal_overlap(identity[:, :3], identity[:, :3]), 1.0
    )
    assert np.isclose(
        lower_triangle_principal_overlap(identity[:, :3], identity[:, 3:6]), 0.0
    )


def test_synthetic_rank64_style_sufficiency_and_necessity():
    rng = np.random.default_rng(1805)
    candidates, dimension, latent_dimension, output_dimension = 13, 80, 8, 16
    true_basis, _ = np.linalg.qr(rng.normal(size=(dimension, latent_dimension)))
    output_map = rng.normal(size=(latent_dimension, output_dimension))
    feature_rows, output_rows = [], []
    for _ in range(24):
        latent = candidate_center(rng.normal(size=(candidates, latent_dimension)))
        features = rng.normal(size=(1, dimension)) + latent @ true_basis.T
        outputs = latent @ output_map
        feature_rows.append(candidate_center(features))
        output_rows.append(candidate_center(outputs))
    x = np.concatenate(feature_rows)
    y = np.concatenate(output_rows)
    fitted = fit_dual_ridge_basis(x, y, penalty=1e-8, max_rank=latent_dimension)

    latent = candidate_center(rng.normal(size=(candidates, latent_dimension)))
    heldout = rng.normal(size=(1, dimension)) + latent @ true_basis.T
    baseline = latent @ output_map
    permutation = fixed_derangement(candidates, 1807)
    donor_delta = action_swap_delta(heldout, permutation, fitted["basis"])
    donor_output = candidate_center(heldout + donor_delta) @ true_basis @ output_map
    transfer = donor_transfer_metrics(baseline, donor_output, permutation)
    assert transfer["coefficient"] > 0.99

    ablated = heldout + projection_ablation_delta(heldout, fitted["basis"])
    ablated_output = candidate_center(ablated) @ true_basis @ output_map
    necessity = action_contrast_energy_metrics(baseline, ablated_output)
    assert necessity["energy_reduction"] > 0.99
