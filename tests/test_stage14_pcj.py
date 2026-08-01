import numpy as np

from cf_faithfulness.stage14_pcj import (
    balanced_modes,
    canonical_mode_rows,
    channel_metric_from_moments,
    earliest_within_one_se,
    haar_rotation,
    hierarchical_bootstrap_means,
    inverse_transform_primal_channels,
    one_sided_t_lower,
    omp_codes,
    relative_error,
    sparse_reconstruct,
    symmetric_channel_metric,
    training_span,
    transfer_metrics,
    transform_dual_channels,
    transform_primal_channels,
)


def test_channel_metric_preserves_dual_primal_contraction():
    rng = np.random.default_rng(4)
    samples = rng.normal(size=(200, 5))
    metric = symmetric_channel_metric(samples)
    streamed = channel_metric_from_moments(
        len(samples), samples.sum(axis=0), samples.T @ samples
    )
    assert np.allclose(metric["covariance"], streamed["covariance"])
    primal = rng.normal(size=(3, 7, 5))
    dual = rng.normal(size=(3, 7, 5))
    whitened = transform_primal_channels(
        primal, metric["inverse_square_root"]
    )
    transformed_dual = transform_dual_channels(
        dual, metric["square_root"]
    )
    assert np.allclose(
        np.sum(primal * dual, axis=(-2, -1)),
        np.sum(whitened * transformed_dual, axis=(-2, -1)),
        atol=1e-9,
    )
    restored = inverse_transform_primal_channels(
        whitened, metric["square_root"]
    )
    assert np.allclose(restored, primal, atol=1e-9)


def test_balanced_modes_are_biorthogonal():
    rng = np.random.default_rng(8)
    shared = rng.normal(size=(20, 4))
    g = rng.normal(size=(6, 4)) @ shared.T
    b = shared @ rng.normal(size=(4, 7))
    fitted = balanced_modes(g, b)
    identity = fitted["dual"].T @ fitted["primal"]
    assert relative_error(identity, np.eye(len(identity))) < 1e-10
    rows, labels = canonical_mode_rows(fitted["primal"], fitted["dual"])
    assert len(rows) == len(labels) == 2 * len(identity)
    assert np.allclose(np.linalg.norm(rows, axis=1), 1.0)


def test_training_span_is_lossless_and_omp_uses_fixed_support():
    rng = np.random.default_rng(11)
    values = rng.normal(size=(12, 30))
    fitted = training_span(values)
    recovered = fitted["coordinates"] @ fitted["basis"].T
    assert relative_error(recovered, values) < 1e-10

    dictionary = np.eye(fitted["coordinates"].shape[1])
    target = fitted["coordinates"][:3]
    codes = omp_codes(target, dictionary, sparsity=3)
    assert np.all(np.sum(codes != 0, axis=1) <= 3)
    codes_again, reconstructed = sparse_reconstruct(
        target, dictionary, sparsity=3
    )
    assert np.array_equal(codes, codes_again)
    assert reconstructed.shape == target.shape


def test_transfer_metrics_and_layer_rule():
    truth = np.array([[1.0, 2.0], [-1.0, 0.5]])
    metrics = transfer_metrics(truth, truth)
    assert metrics["reconstruction"] == 1.0
    assert np.isclose(metrics["cosine"], 1.0)
    task_scores = np.array(
        [
            [0.20, 0.40, 0.43],
            [0.22, 0.42, 0.47],
            [0.18, 0.38, 0.35],
            [0.21, 0.39, 0.38],
        ]
    )
    selected = earliest_within_one_se(task_scores)
    assert selected["selected_index"] == 1


def test_bootstrap_and_haar_are_reproducible():
    values = np.arange(8, dtype=np.float64)
    tasks = np.repeat(np.arange(4), 2)
    first = hierarchical_bootstrap_means(values, tasks, draws=100, seed=19)
    second = hierarchical_bootstrap_means(values, tasks, draws=100, seed=19)
    assert np.array_equal(first, second)
    rotation = haar_rotation(8, 23)
    assert np.allclose(rotation.T @ rotation, np.eye(8), atol=1e-10)
    assert np.linalg.det(rotation) > 0


def test_one_sided_bound_uses_actual_task_count():
    values = np.asarray([0.1, 0.2, 0.3, 0.4])
    expected = values.mean() - 2.353363 * values.std(ddof=1) / np.sqrt(4)
    assert np.isclose(one_sided_t_lower(values), expected)
