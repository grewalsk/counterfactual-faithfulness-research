import numpy as np

from cf_faithfulness.stage13b_geometry import (
    array_sha256,
    basis_overlap,
    covariance_shaped_coordinates,
    exact_positive_sign_test,
    fit_dual_pca,
    frozen_action_bank,
    hierarchical_bootstrap_indices,
    hierarchical_bootstrap_means,
    one_sided_t_lower,
    reconstruction_by_groups,
    select_rank_one_se,
    weighted_dual_pca,
)


def test_frozen_actions_are_unique_antithetic_and_hash_stable():
    labels, actions = frozen_action_bank()
    assert len(labels) == 13
    assert actions.shape == (13, 15, 2)
    assert (
        array_sha256(actions)
        == "802129bd281fdd2d42a395429e5a0e00df2dc10032b339ecb8bdc8b2521d9fd2"
    )
    for prefix in [5, 15]:
        assert len({row[:prefix].tobytes() for row in actions[1:]}) == 12
    for left in range(1, 13, 2):
        np.testing.assert_array_equal(actions[left], -actions[left + 1])


def test_dual_pca_reconstruction_matches_primal_svd():
    rng = np.random.default_rng(4)
    train = rng.normal(size=(18, 11))
    test = rng.normal(size=(6, 11))
    fitted = fit_dual_pca(train @ train.T, max_rank=6)
    primal_axes = train.T @ fitted["coefficients"]
    np.testing.assert_allclose(
        primal_axes.T @ primal_axes, np.eye(6), atol=1e-10
    )
    observed = reconstruction_by_groups(
        test @ train.T,
        test @ test.T,
        fitted["coefficients"],
        [1, 3, 6],
        [np.arange(3), np.arange(3, 6)],
    )
    _, _, right = np.linalg.svd(train, full_matrices=False)
    expected = []
    for rows in [np.arange(3), np.arange(3, 6)]:
        denominator = np.sum(test[rows] ** 2)
        expected.append(
            [
                np.sum((test[rows] @ right[:rank].T) ** 2) / denominator
                for rank in [1, 3, 6]
            ]
        )
    np.testing.assert_allclose(observed, expected, atol=1e-10)


def test_weighted_dual_pca_matches_weighted_primal_svd():
    rng = np.random.default_rng(8)
    matrix = rng.normal(size=(14, 9))
    weights = np.linspace(0.2, 1.8, len(matrix))
    fitted = weighted_dual_pca(
        matrix @ matrix.T, weights, max_rank=5
    )
    axes = matrix.T @ fitted["coefficients"]
    _, _, right = np.linalg.svd(
        np.sqrt(weights)[:, None] * matrix, full_matrices=False
    )
    overlap = np.linalg.svd(axes.T @ right[:5].T)[1]
    np.testing.assert_allclose(overlap, np.ones(5), atol=1e-10)


def test_one_se_rule_selects_smallest_eligible_rank():
    scores = np.array(
        [
            [0.30, 0.50, 0.48],
            [0.31, 0.51, 0.56],
            [0.29, 0.50, 0.52],
            [0.30, 0.51, 0.50],
        ]
    )
    result = select_rank_one_se(scores, [2, 4, 8])
    assert result["best_rank"] == 8
    assert result["selected_rank"] == 4


def test_covariance_null_and_overlap_are_well_formed():
    coordinates = covariance_shaped_coordinates(
        np.linspace(20.0, 1.0, 12), seed=91, max_rank=5
    )
    np.testing.assert_allclose(
        coordinates.T @ coordinates, np.eye(5), atol=1e-10
    )
    coefficients = np.eye(8)
    assert basis_overlap(coefficients, np.eye(8), coefficients, 5) == 1.0


def test_task_level_inference_and_hierarchical_bootstrap():
    values = np.arange(32, dtype=float).reshape(8, 4)
    tasks, states = hierarchical_bootstrap_indices(8, 4, 100, 17)
    first = hierarchical_bootstrap_means(values, tasks, states)
    second = hierarchical_bootstrap_means(values, tasks, states)
    np.testing.assert_array_equal(first, second)
    assert len(first) == 100
    assert one_sided_t_lower(np.linspace(0.1, 0.8, 8)) > 0
    sign = exact_positive_sign_test([1, 1, 1, 1, 1, 1, 1, -1])
    assert sign == {"positive": 7, "nonzero": 8, "one_sided_p": 9 / 256}
