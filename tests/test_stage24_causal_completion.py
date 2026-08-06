import numpy as np

from cf_faithfulness.stage24_causal_completion import (
    causal_completion_rank,
    completion_edit,
    completion_residual,
    native_reconstruction_fraction,
    orthonormal_residual_basis,
    paired_completion_gain,
)


def test_exact_mode_decomposition_leaves_coordinate_null_residual():
    covectors = np.eye(6)[:, :2]
    native = np.asarray([1.0, -2.0, 3.0, 4.0, 0.0, 1.0])
    mode = np.asarray([1.0, -2.0, 0.0, 0.0, 0.0, 0.0])
    result = completion_residual(native, mode, covectors)
    assert np.allclose(covectors.T @ result["residual"], 0.0)
    assert result["relative_coordinate_residual"] == 0.0


def test_completion_residual_rejects_coordinate_drift():
    covectors = np.eye(4)[:, :1]
    with np.testing.assert_raises(ValueError):
        completion_residual([1.0, 2.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0], covectors)


def test_nested_completion_reconstructs_native_delta_monotonically():
    mode = np.asarray([1.0, 0.0, 0.0, 0.0])
    residual = np.asarray([0.0, 3.0, 2.0, 1.0])
    native = mode + residual
    basis = np.eye(4)[:, [1, 2, 3]]
    fractions = [
        native_reconstruction_fraction(native, completion_edit(mode, residual, basis, rank))
        for rank in range(4)
    ]
    assert np.all(np.diff(fractions) >= -1e-12)
    assert np.isclose(fractions[-1], 1.0)


def test_residual_basis_is_orthonormal_and_excludes_mode_covectors():
    rng = np.random.default_rng(24)
    excluded, _ = np.linalg.qr(rng.normal(size=(12, 2)))
    residuals = rng.normal(size=(8, 12))
    residuals -= (residuals @ excluded) @ excluded.T
    basis, singular = orthonormal_residual_basis(residuals, 5, excluded)
    assert len(singular) >= 5
    assert np.allclose(basis.T @ basis, np.eye(5), atol=1e-10)
    assert np.allclose(excluded.T @ basis, 0.0, atol=1e-10)


def test_causal_completion_rank_uses_lower_confidence_bound():
    rows = [
        {"rank": 4, "mean": 0.9, "lower": 0.7},
        {"rank": 8, "mean": 0.86, "lower": 0.81},
        {"rank": 16, "mean": 0.95, "lower": 0.9},
    ]
    assert causal_completion_rank(rows, 0.8) == 8
    assert causal_completion_rank(rows, 0.95) is None


def test_paired_completion_gain_preserves_pairing():
    gain = paired_completion_gain([0.8, 0.5, 0.1], [0.2, 0.4, 0.3])
    assert np.allclose(gain, [0.6, 0.1, -0.2])
