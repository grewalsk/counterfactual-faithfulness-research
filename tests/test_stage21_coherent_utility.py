import numpy as np

from cf_faithfulness.stage21_coherent_utility import (
    apply_pose_correction,
    centered_pose_residual,
    counterfactual_interface_metrics,
    corrected_planning_metrics,
    select_ridge_on_calibration,
    subspace_coordinates,
)


def test_subspace_coordinates_remove_candidate_mean():
    carrier = np.asarray([[1.0, 2.0], [3.0, 5.0], [8.0, 4.0]])
    basis = np.eye(2)
    coordinates = subspace_coordinates(carrier, basis)
    assert np.allclose(coordinates.mean(axis=0), 0.0)
    assert np.allclose(coordinates, carrier - carrier.mean(axis=0))


def test_pose_correction_recovers_candidate_contrast_and_unit_orientation():
    decoded = np.asarray(
        [[0.1, 0.2, 0.0, 2.0], [0.2, 0.1, 2.0, 0.0], [0.3, 0.3, 1.0, 1.0]]
    )
    truth = np.asarray(
        [[0.2, 0.2, 0.0, 1.0], [0.1, 0.1, 1.0, 0.0], [0.3, 0.3, 2**-0.5, 2**-0.5]]
    )
    residual = centered_pose_residual(decoded, truth)
    corrected = apply_pose_correction(decoded, residual)
    assert np.allclose(np.linalg.norm(corrected[:, 2:4], axis=1), 1.0)
    assert np.allclose(residual.mean(axis=0), 0.0)


def test_calibration_selects_and_refits_one_ridge():
    rng = np.random.default_rng(2101)
    x = rng.normal(size=(40, 3))
    coefficient = np.asarray([[1.0, -0.2], [0.5, 0.3], [-0.4, 0.8]])
    y = x @ coefficient
    result = select_ridge_on_calibration(
        x[:24], y[:24], x[24:], y[24:], [1e-4, 1e-2, 1.0]
    )
    assert result["selected_ridge"] in {1e-4, 1e-2, 1.0}
    assert result["model"]["coefficient"].shape == (3, 2)
    assert len(result["calibration_rows"]) == 3


def test_corrected_planning_metrics_reports_true_cost_improvement():
    decoded = np.asarray(
        [[0.0, 0.0, 0.0, 1.0], [0.9, 0.0, 0.0, 1.0], [0.5, 0.0, 0.0, 1.0]]
    )
    truth = np.asarray(
        [[0.8, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.5, 0.0, 0.0, 1.0]]
    )
    correction = truth - decoded
    result = corrected_planning_metrics(decoded, correction, truth, [0.0, 0.0, 0.0])
    assert result["baseline"]["selected_action"] == 0
    assert result["corrected"]["selected_action"] == 1
    assert result["selected_true_cost_improvement"] > 0


def test_exact_interface_permutation_selects_target():
    baseline = np.asarray([0.6, 0.1, 0.4, 0.8])
    permutation = np.asarray([2, 3, 0, 1])
    target = 3
    result = counterfactual_interface_metrics(
        baseline, baseline[permutation], permutation, target
    )
    assert result["choice_matches_counterfactual"]
    assert result["target_selected"]
    assert result["score_transfer_coefficient"] > 1 - 1e-12
    assert result["score_counterfactual_normalized_rmse"] < 1e-12
