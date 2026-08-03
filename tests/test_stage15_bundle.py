import numpy as np

from cf_faithfulness.stage15_bundle import (
    PHYSICAL_READER_LABELS,
    align_basis_sequence,
    chordal_subspace_distance,
    energy_map,
    grouped_label_permutation,
    grouped_ridge_cv,
    matrix_cosine,
    normalized_matrix_distance,
    orthonormal_columns,
    physical_reader_targets,
    predict_ridge,
    procrustes_align,
    r2_per_output,
    spearman_correlation,
    support_matched_random,
    temporal_action_basis,
)


def test_physical_reader_targets_and_action_basis():
    states = np.array(
        [
            [256.0, 128.0, 64.0, 384.0, 0.0],
            [0.0, 512.0, 512.0, 0.0, np.pi / 2],
        ]
    )
    targets = physical_reader_targets(states)
    assert targets.shape == (2, len(PHYSICAL_READER_LABELS))
    assert np.allclose(targets[0], [0.5, 0.25, 0.125, 0.75, 0.0, 1.0])
    assert np.allclose(targets[1], [0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    basis = temporal_action_basis(15, profiles=3)
    assert basis.shape == (30, 6)
    assert np.allclose(basis.T @ basis, np.eye(6), atol=1e-12)


def test_grouped_ridge_cv_recovers_linear_reader_without_group_leakage():
    rng = np.random.default_rng(3)
    features = rng.normal(size=(120, 8))
    coefficient = rng.normal(size=(8, 3))
    targets = features @ coefficient + np.array([0.2, -0.1, 0.4])
    groups = np.repeat(np.arange(6), 20)
    result = grouped_ridge_cv(
        features,
        targets,
        groups,
        ridges=[1e-8, 1e-3, 1.0],
    )
    prediction = predict_ridge(result["model"], features)
    assert result["selected_ridge"] == 1e-8
    assert np.min(r2_per_output(targets, prediction)) > 0.999999
    fold_rows = [
        row for row in result["cv_rows"] if row["held_out_group"] != "mean"
    ]
    assert len(fold_rows) == 18


def test_subspace_geometry_and_procrustes_are_coordinate_invariant():
    rng = np.random.default_rng(4)
    base = orthonormal_columns(rng.normal(size=(30, 4)))
    rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    rotated = base @ rotation
    assert chordal_subspace_distance(base, rotated) < 1e-7
    aligned = procrustes_align(base, rotated)
    assert aligned["relative_error"] < 1e-12
    assert np.allclose(aligned["rotation"].T @ aligned["rotation"], np.eye(4))
    sequence = align_basis_sequence([base, rotated, base @ rotation.T], rank=4)
    assert len(sequence) == 3
    assert all(matrix_cosine(sequence[0], value) > 0.999999 for value in sequence)


def test_distances_correlations_and_grouped_permutations():
    left = np.arange(10, dtype=float)
    right = left**3
    assert spearman_correlation(left, right) == 1.0
    assert normalized_matrix_distance(np.eye(2), np.eye(2)) == 0.0
    groups = np.repeat([10, 11], 5)
    labels = np.tile(np.arange(5), 2)
    shuffled = grouped_label_permutation(labels, groups, seed=99)
    for group in np.unique(groups):
        indices = groups == group
        assert sorted(shuffled[indices]) == sorted(labels[indices])


def test_support_matched_null_preserves_energy_not_direction():
    rng = np.random.default_rng(5)
    template = rng.normal(size=256 * 4)
    null = support_matched_random(template, seed=7, tokens=256, channels=4)
    assert np.allclose(
        energy_map(template, tokens=256, channels=4),
        energy_map(null, tokens=256, channels=4),
    )
    assert matrix_cosine(template, null) < 0.25
