import numpy as np

from cf_faithfulness.stage23_mode_operator import (
    countsketch_mode_covectors,
    deterministic_signed_permutation,
    minimal_constrained_transport,
    native_to_whitened_covectors,
    operator_transfer_metrics,
    symmetric_finite_response,
)


def test_countsketch_covectors_match_explicit_linear_map():
    bucket = np.asarray([0, 1, 0, 2, 1])
    sign = np.asarray([1.0, -1.0, 1.0, -1.0, 1.0])
    counts = np.bincount(bucket, minlength=3)
    bucket_scale = np.sqrt(counts)
    partition_scale = np.asarray([2.0, 0.5, 4.0])
    components = np.asarray([[1.0, 0.0], [0.2, 0.8], [-0.5, 0.3]])
    covectors = countsketch_mode_covectors(
        bucket,
        sign,
        bucket_scale,
        partition_scale,
        components,
        center_factor=0.75,
    )
    projector = np.zeros((len(bucket), 3))
    projector[np.arange(len(bucket)), bucket] = sign / bucket_scale[bucket]
    explicit = 0.75 * projector @ np.diag(1.0 / partition_scale) @ components
    assert np.allclose(covectors, explicit)
    delta = np.asarray([0.5, -0.2, 1.0, 0.4, -0.7])
    assert np.allclose(delta @ covectors, delta @ explicit)


def test_whitened_covectors_preserve_linear_response():
    rng = np.random.default_rng(2301)
    root = np.asarray([[1.8, 0.2], [0.1, 0.9]])
    native_covectors = rng.normal(size=(6, 3))
    white_covectors = native_to_whitened_covectors(native_covectors, root, 2)
    delta_white = rng.normal(size=6)
    delta_native = (delta_white.reshape(3, 2) @ root.T).reshape(-1)
    assert np.allclose(delta_native @ native_covectors, delta_white @ white_covectors)


def test_constrained_transport_hits_mode_and_avoids_content():
    rng = np.random.default_rng(2302)
    raw = rng.normal(size=(30, 7))
    constraints, _ = np.linalg.qr(raw)
    mode = constraints[:, :3]
    protected = constraints[:, 3:]
    target = np.asarray([0.7, -1.1, 0.4])
    result = minimal_constrained_transport(mode, protected, target, ridge=0.0)
    assert np.allclose(mode.T @ result["delta"], target, atol=1e-11)
    assert np.allclose(protected.T @ result["delta"], 0.0, atol=1e-11)
    assert result["mode_residual_norm"] < 1e-10


def test_symmetric_response_recovers_linear_operator():
    matrix = np.asarray([[2.0, -1.0], [0.5, 3.0]])
    probe = np.asarray([0.3, -0.8])
    center = np.asarray([1.0, 2.0])
    dose = 0.25
    plus = center + matrix @ (dose * probe)
    minus = center - matrix @ (dose * probe)
    assert np.allclose(symmetric_finite_response(plus, minus, dose), matrix @ probe)


def test_operator_transfer_recovers_fraction_of_native_switch():
    off = np.asarray([0.0, 1.0, -2.0])
    native = np.asarray([2.0, -1.0, 0.0])
    context = off + 0.6 * (native - off)
    result = operator_transfer_metrics(off, context, native)
    assert np.isclose(result["transfer_coefficient"], 0.6)
    assert np.isclose(result["transfer_cosine"], 1.0)
    assert np.isclose(result["distance_to_native_ratio"], 0.4)


def test_signed_permutation_is_deterministic_and_nonidentity():
    values = np.arange(1.0, 9.0)
    first = deterministic_signed_permutation(values, 23)
    second = deterministic_signed_permutation(values, 23)
    assert np.array_equal(first[0], second[0])
    assert not np.array_equal(first[0], values)
