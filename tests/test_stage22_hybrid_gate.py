import numpy as np

from cf_faithfulness.stage22_hybrid_gate import (
    apply_mode_partition,
    binary_alignment_metrics,
    center_by_group,
    difference_of_means_direction,
    discover_mode_partition,
    factorial_interaction_metrics,
    orthogonalize_basis,
    projected_pair_delta,
)


def test_center_by_group_removes_each_state_mean():
    values = np.asarray([[1.0, 2.0], [3.0, 4.0], [8.0, 1.0], [4.0, 5.0]])
    groups = np.asarray([0, 0, 1, 1])
    centered = center_by_group(values, groups)
    assert np.allclose(centered[:2].mean(axis=0), 0.0)
    assert np.allclose(centered[2:].mean(axis=0), 0.0)


def test_mode_discovery_is_label_free_and_reapplicable():
    rng = np.random.default_rng(22)
    groups = np.repeat(np.arange(12), 6)
    latent = np.tile(np.asarray([0, 0, 0, 1, 1, 1]), 12)
    activations = rng.normal(scale=0.1, size=(len(groups), 5))
    activations += latent[:, None] * np.asarray([2.0, -1.0, 0.5, 0.0, 0.0])
    outputs = rng.normal(scale=0.05, size=(len(groups), 4))
    outputs += latent[:, None] * np.asarray([1.5, 0.2, 0.0, 0.0])
    found = discover_mode_partition(
        activations, outputs, groups, seed=2201, pca_rank=4
    )
    centered = center_by_group(activations, groups)
    assignments, _ = apply_mode_partition(
        centered,
        found["mean"],
        found["scale"],
        found["components"],
        found["centroids"],
    )
    assert np.array_equal(assignments, found["assignments"])
    metrics = binary_alignment_metrics(found["mode_on"], latent.astype(bool))
    assert metrics["balanced_accuracy"] > 0.95


def test_gate_direction_and_effect_basis_are_orthogonal():
    features = np.asarray(
        [[-2.0, 0.0, 1.0], [-1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]
    )
    labels = np.asarray([False, False, True, True])
    gate = difference_of_means_direction(features, labels)
    effect = orthogonalize_basis(np.eye(3), gate)
    assert np.allclose(gate @ effect, 0.0, atol=1e-12)
    assert np.allclose(effect.T @ effect, np.eye(effect.shape[1]), atol=1e-12)


def test_projected_pair_delta_changes_only_base_candidate():
    carrier = np.asarray([[1.0, 2.0], [4.0, 6.0], [7.0, 3.0]])
    delta = projected_pair_delta(carrier, 0, 1, np.asarray([1.0, 0.0]))
    assert np.allclose(delta[0], [3.0, 0.0])
    assert np.allclose(delta[1:], 0.0)


def test_factorial_interaction_recovers_multiplicative_gate():
    # y = background + gate * effect, with g0=0, g1=1 and two effects.
    background = np.asarray([0.2, -0.1, 0.0])
    effect0 = np.asarray([0.1, 0.0, 0.0])
    effect1 = np.asarray([1.1, 0.5, 0.0])
    y00 = background
    y10 = background + effect0
    y01 = background
    y11 = background + effect1
    result = factorial_interaction_metrics(y00, y10, y01, y11, y11)
    assert result["both_coefficient"] > 0.99
    assert result["interaction_coefficient"] > 0.8
    assert result["interaction_cosine"] > 0.99


def test_binary_alignment_metrics_handles_imperfect_classifier():
    result = binary_alignment_metrics(
        [True, True, False, False], [True, False, False, False]
    )
    assert result["tp"] == 1
    assert result["fp"] == 1
    assert 0.0 < result["balanced_accuracy"] < 1.0
