import numpy as np

from cf_faithfulness.stage26_contact_transport import (
    canonical_contact_features,
    contact_frame_basis,
    fit_response_fiber,
    grouped_ridge_oof,
    intervention_transfer_metrics,
    projected_donor_delta,
    response_coordinates,
    select_low_response_donor,
    token_centers,
    transport_contact_delta,
    wrap_angle,
)


def test_contact_basis_roundtrip_and_orthonormality():
    positions = token_centers()
    basis = contact_frame_basis(positions, [250.0, 230.0], [0.6, 0.8])
    assert basis.shape == (256, 3)
    assert np.allclose(basis.T @ basis, np.eye(3), atol=1e-10)
    rng = np.random.default_rng(7)
    feature = rng.normal(size=(3, 5))
    field = transport_contact_delta(feature, basis, channels=5)
    recovered = canonical_contact_features(field, basis)
    assert np.allclose(recovered, feature.reshape(-1), atol=1e-10)


def test_response_coordinates_use_contact_frame():
    ghost = np.zeros(10)
    contact = ghost.copy()
    contact[2:4] = [2.0, 1.0]
    contact[4] = 2.0 * np.pi - 0.2
    response = response_coordinates(contact, ghost, [3.0, 0.0], [1.0, 0.0])
    assert np.allclose(response[:3], [np.log1p(3.0), 2.0, 1.0])
    assert np.isclose(response[3], -0.2)
    assert np.isclose(wrap_angle(2.0 * np.pi + 0.3), 0.3)


def test_grouped_ridge_and_response_fiber_recover_signal():
    rng = np.random.default_rng(11)
    groups = np.repeat(np.arange(20), 8)
    features = rng.normal(size=(len(groups), 18))
    targets = features[:, :3] @ rng.normal(size=(3, 4))
    predictions = grouped_ridge_oof(features, targets, groups, penalty=0.01, folds=5)
    assert np.mean((predictions - targets) ** 2) < 0.02
    model = fit_response_fiber(features, targets, penalty=0.01, rank=4)
    assert model["fiber"].shape == (18, 3)
    assert np.allclose(model["fiber"].T @ model["fiber"], np.eye(3), atol=1e-10)


def test_projected_donor_delta_stays_inside_fiber():
    rng = np.random.default_rng(13)
    features = rng.normal(size=(100, 12))
    targets = features[:, :2]
    model = fit_response_fiber(features, targets, penalty=0.1, rank=2)
    delta = projected_donor_delta(model, features[0], features[1])
    standardized = delta / model["x_scale"]
    residual = standardized - model["fiber"] @ (model["fiber"].T @ standardized)
    assert np.linalg.norm(residual) < 1e-10


def test_low_response_donor_and_transfer_metric():
    descriptors = np.asarray([[0.0, 0.0], [0.1, 0.2], [3.0, 3.0]])
    responses = np.asarray([[0.1, 0.0], [0.9, 0.0], [0.2, 0.0]])
    chosen, distance = select_low_response_donor(
        [0.0, 0.1], [1.0, 0.0], descriptors, responses
    )
    assert chosen == 0
    assert distance >= 0.0
    result = intervention_transfer_metrics([0.0, 0.0], [1.0, 2.0], [1.0, 2.0])
    assert np.isclose(result["transfer_coefficient"], 1.0)
    assert np.isclose(result["transfer_cosine"], 1.0)
