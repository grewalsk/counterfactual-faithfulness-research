"""Numerical primitives for Stage 21 coherent handoff and utility.

Stage 21 separates two questions that Stage 20 intentionally mixed: whether
the decoded planner is exactly controllable after the last action-conditioned
block, and whether frozen causal-subspace coordinates support a correction
that improves held-out physical action selection.  This module is NumPy-only
so the split, correction, and planning contracts can be tested without JEPA.
"""

from __future__ import annotations

import numpy as np

from .stage15_bundle import fit_ridge, predict_ridge
from .stage17_action_contrast import (
    candidate_center,
    decoded_task_cost,
    donor_transfer_metrics,
    ranking_metrics,
)


def subspace_coordinates(whitened_carrier, basis):
    """Project candidate-centered carrier rows onto one frozen basis."""

    values = np.asarray(whitened_carrier, dtype=np.float64)
    directions = np.asarray(basis, dtype=np.float64)
    if values.ndim < 2 or directions.ndim != 2:
        raise ValueError("carrier and basis must be matrices after flattening")
    flat = values.reshape(values.shape[0], -1)
    if flat.shape[1] != directions.shape[0]:
        raise ValueError("basis does not match flattened carrier width")
    if not np.all(np.isfinite(flat)) or not np.all(np.isfinite(directions)):
        raise ValueError("carrier coordinates contain nonfinite values")
    return candidate_center(flat) @ directions


def centered_pose_residual(decoded_pose, true_pose):
    """Return the within-state physical-pose error to be corrected."""

    decoded = np.asarray(decoded_pose, dtype=np.float64)
    truth = np.asarray(true_pose, dtype=np.float64)
    if decoded.shape != truth.shape or decoded.ndim != 2 or decoded.shape[1] != 4:
        raise ValueError("decoded and true poses must be aligned [actions,4]")
    return candidate_center(truth - decoded)


def normalize_pose_orientation(pose):
    """Normalize the predicted sine/cosine pair without changing xy."""

    values = np.asarray(pose, dtype=np.float64).copy()
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("pose must have shape [actions,4]")
    norm = np.linalg.norm(values[:, 2:4], axis=1, keepdims=True)
    fallback = np.zeros_like(values[:, 2:4])
    fallback[:, 1] = 1.0
    safe_norm = np.where(norm > 1e-8, norm, 1.0)
    values[:, 2:4] = np.where(
        norm > 1e-8, values[:, 2:4] / safe_norm, fallback
    )
    return values


def apply_pose_correction(decoded_pose, predicted_residual):
    """Add a candidate-contrast correction and restore valid orientation."""

    decoded = np.asarray(decoded_pose, dtype=np.float64)
    correction = np.asarray(predicted_residual, dtype=np.float64)
    if decoded.shape != correction.shape:
        raise ValueError("decoded pose and correction must align")
    return normalize_pose_orientation(decoded + candidate_center(correction))


def select_ridge_on_calibration(
    train_features,
    train_targets,
    calibration_features,
    calibration_targets,
    ridges,
):
    """Select one ridge using only a fixed construction/calibration split."""

    ridge_values = tuple(float(value) for value in ridges)
    if not ridge_values or any(value <= 0 for value in ridge_values):
        raise ValueError("ridges must be a nonempty positive grid")
    rows = []
    for ridge in ridge_values:
        model = fit_ridge(train_features, train_targets, ridge)
        prediction = predict_ridge(model, calibration_features)
        rows.append(
            {
                "ridge": ridge,
                "calibration_mse": float(
                    np.mean((prediction - np.asarray(calibration_targets)) ** 2)
                ),
            }
        )
    selected = min(rows, key=lambda row: (row["calibration_mse"], row["ridge"]))
    combined_features = np.concatenate(
        [np.asarray(train_features), np.asarray(calibration_features)], axis=0
    )
    combined_targets = np.concatenate(
        [np.asarray(train_targets), np.asarray(calibration_targets)], axis=0
    )
    return {
        "model": fit_ridge(combined_features, combined_targets, selected["ridge"]),
        "selected_ridge": float(selected["ridge"]),
        "calibration_rows": rows,
    }


def corrected_planning_metrics(decoded_pose, predicted_residual, true_pose, goal):
    """Evaluate a frozen correction against simulator truth for one state."""

    decoded = np.asarray(decoded_pose, dtype=np.float64)
    truth = np.asarray(true_pose, dtype=np.float64)
    corrected = apply_pose_correction(decoded, predicted_residual)
    baseline_cost = decoded_task_cost(decoded, goal)
    corrected_cost = decoded_task_cost(corrected, goal)
    true_cost = decoded_task_cost(truth, goal)
    baseline = ranking_metrics(true_cost, baseline_cost)
    treatment = ranking_metrics(true_cost, corrected_cost)
    return {
        "baseline": baseline,
        "corrected": treatment,
        "baseline_selected_true_cost": float(true_cost[baseline["selected_action"]]),
        "corrected_selected_true_cost": float(true_cost[treatment["selected_action"]]),
        "selected_true_cost_improvement": float(
            true_cost[baseline["selected_action"]]
            - true_cost[treatment["selected_action"]]
        ),
    }


def counterfactual_interface_metrics(
    baseline_scores, patched_scores, permutation, target_action
):
    """Score an intended candidate permutation without simulator outcomes."""

    baseline = np.asarray(baseline_scores, dtype=np.float64)
    patched = np.asarray(patched_scores, dtype=np.float64)
    permutation = np.asarray(permutation, dtype=np.int64)
    target_action = int(target_action)
    if baseline.ndim != 1 or patched.shape != baseline.shape:
        raise ValueError("score vectors must be aligned")
    if sorted(permutation.tolist()) != list(range(len(baseline))):
        raise ValueError("permutation is malformed")
    expected = baseline[permutation]
    transfer = donor_transfer_metrics(
        baseline[:, None], patched[:, None], permutation
    )
    denominator = float(np.sqrt(np.mean((expected - baseline) ** 2)))
    error = float(
        np.sqrt(np.mean((patched - expected) ** 2)) / max(denominator, 1e-12)
    )
    baseline_order = np.argsort(baseline, kind="stable")
    patched_order = np.argsort(patched, kind="stable")
    baseline_rank = int(np.flatnonzero(baseline_order == target_action)[0])
    patched_rank = int(np.flatnonzero(patched_order == target_action)[0])
    expected_choice = int(np.argmin(expected))
    patched_choice = int(np.argmin(patched))
    return {
        "score_transfer_coefficient": transfer["coefficient"],
        "score_transfer_cosine": transfer["cosine"],
        "score_counterfactual_normalized_rmse": error,
        "baseline_choice": int(np.argmin(baseline)),
        "expected_counterfactual_choice": expected_choice,
        "patched_choice": patched_choice,
        "target_action": target_action,
        "target_rank_baseline": baseline_rank,
        "target_rank_patched": patched_rank,
        "target_rank_gain": int(baseline_rank - patched_rank),
        "target_selected": bool(patched_choice == target_action),
        "choice_matches_counterfactual": bool(patched_choice == expected_choice),
    }
