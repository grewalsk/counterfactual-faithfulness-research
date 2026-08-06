import numpy as np

from cf_faithfulness.stage25_causal_kkt import (
    contact_projection_metrics,
    countsketch_adjoint,
    fit_standardized_ridge,
    intervention_transfer_metrics,
    minimum_norm_coordinate_edit,
    predict_standardized_ridge,
    stepwise_impulse_momentum_residual,
)


def test_contact_projection_endpoints():
    free = np.array([0.0, 0.0])
    contact = np.array([2.0, 0.0])
    assert np.isclose(
        contact_projection_metrics(free, contact, free)["contact_coefficient"], 0.0
    )
    result = contact_projection_metrics(contact, contact, free)
    assert np.isclose(result["contact_coefficient"], 1.0)
    assert np.isclose(result["contact_cosine"], 1.0)


def test_intervention_transfer_identity():
    baseline = np.array([1.0, -1.0])
    desired = np.array([2.0, 3.0])
    result = intervention_transfer_metrics(baseline, baseline + desired, desired)
    assert np.isclose(result["transfer_coefficient"], 1.0)
    assert np.isclose(result["distance_to_desired_ratio"], 0.0)


def test_stepwise_impulse_momentum_audit_groups_contact_arbiters():
    residuals = stepwise_impulse_momentum_residual(
        physics_steps=np.array([4, 5, 5]),
        block_impulses=np.array([[2.0, 0.0], [1.0, -0.5], [2.0, 0.5]]),
        block_velocities=np.array([[1.0, 0.0], [1.5, 0.0], [1.5, 0.0]]),
        block_masses=np.array([2.0, 2.0, 2.0]),
    )
    assert np.allclose(residuals, 0.0)


def test_countsketch_adjoint_identity():
    bucket = np.array([0, 1, 0, 1])
    sign = np.array([1.0, -1.0, -1.0, 1.0])
    scale = np.sqrt(np.array([2.0, 2.0]))
    weights = np.array([[2.0, 1.0], [-1.0, 3.0]])
    vector = np.array([0.3, -0.7, 1.1, 0.2])
    sketch = np.zeros(2)
    for index in range(len(vector)):
        sketch[bucket[index]] += sign[index] * vector[index]
    sketch /= scale
    adjoint = countsketch_adjoint(bucket, sign, scale, weights)
    assert np.allclose(vector @ adjoint, sketch @ weights)


def test_minimum_norm_edit_respects_protection():
    rng = np.random.default_rng(7)
    covectors = rng.normal(size=(20, 2))
    protected = rng.normal(size=(20, 3))
    target = np.array([0.4, -0.2])
    result = minimum_norm_coordinate_edit(covectors, target, protected)
    assert np.allclose(covectors.T @ result["edit"], target, atol=1e-9)
    assert np.allclose(protected.T @ result["edit"], 0.0, atol=1e-9)


def test_standardized_ridge_recovers_linear_map():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(200, 8))
    weight = rng.normal(size=(8, 2))
    y = x @ weight + np.array([0.5, -0.3])
    model = fit_standardized_ridge(x, y, penalty=1e-8)
    predicted = predict_standardized_ridge(model, x)[0]
    assert np.mean((predicted - y) ** 2) < 1e-12
