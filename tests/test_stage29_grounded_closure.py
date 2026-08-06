import numpy as np

from cf_faithfulness.stage29_grounded_closure import (
    grounded_intervention_metrics,
    latent_closure_metrics,
    vector_alignment,
)


def test_exact_prediction_has_unit_native_closure():
    rng = np.random.default_rng(2901)
    target = rng.normal(size=(24, 3, 5))
    result = latent_closure_metrics(target, target, magnitude_count=4)
    assert np.isclose(result["total_coefficient"], 1.0)
    assert np.isclose(result["centered_cosine"], 1.0)
    assert np.isclose(result["area_coefficient"], 1.0)
    assert np.isclose(result["area_normalized_rmse"], 0.0)


def test_exact_area_swap_closes_against_self_and_truth():
    rng = np.random.default_rng(2902)
    baseline = rng.normal(size=(24, 9))
    permutation = np.arange(24).reshape(4, 6)[:, ::-1].reshape(-1)
    patched = baseline[permutation]
    result = grounded_intervention_metrics(
        baseline, patched, baseline, magnitude_count=4, mode="swap"
    )
    assert np.isclose(result["self_coefficient"], 1.0)
    assert np.isclose(result["grounded_coefficient"], 1.0)
    assert np.isclose(result["self_cosine"], 1.0)
    assert np.isclose(result["grounded_cosine"], 1.0)
    assert np.isclose(result["absolute_target_error_after"], 0.0)


def test_self_consistent_orthogonal_hallucination_is_not_grounded():
    levels = np.asarray([25, 15, 5, -5, -15, -25], dtype=np.float64)
    baseline = np.zeros((24, 2), dtype=np.float64)
    target = np.zeros_like(baseline)
    baseline[:, 0] = np.tile(levels, 4)
    target[:, 1] = np.tile(levels, 4)
    permutation = np.arange(24).reshape(4, 6)[:, ::-1].reshape(-1)
    patched = baseline[permutation]
    result = grounded_intervention_metrics(
        baseline, patched, target, magnitude_count=4, mode="swap"
    )
    assert np.isclose(result["self_cosine"], 1.0)
    assert np.isclose(result["grounded_cosine"], 0.0)
    assert np.isclose(result["self_minus_grounded_cosine"], 1.0)


def test_zero_grounded_contrast_is_explicitly_undefined():
    source = np.ones((6, 2), dtype=np.float64)
    target = np.zeros_like(source)
    result = vector_alignment(source, target)
    assert result["target_energy"] == 0.0
    assert np.isnan(result["coefficient"])
    assert np.isnan(result["cosine"])
