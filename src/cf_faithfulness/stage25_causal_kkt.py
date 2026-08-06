"""Numerical primitives for Stage 25 causal KKT tomography.

The Stage 25 notebook asks whether a frozen action-conditioned world model
contains a causally used, state-dependent contact-impulse variable.  The
helpers in this module deliberately operate on ordinary NumPy arrays so their
algebra can be tested without loading the model or simulator.
"""

from __future__ import annotations

import numpy as np


def contact_projection_metrics(predicted, contact_target, free_target, *, floor=1e-12):
    """Locate predictions along the exact free-to-contact counterfactual chord.

    A coefficient of zero is the collision-disabled target and a coefficient
    of one is the ordinary-contact target.  The orthogonal residual prevents a
    large but misaligned prediction from looking like contact-law evidence.
    """

    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    contact = np.asarray(contact_target, dtype=np.float64).reshape(-1)
    free = np.asarray(free_target, dtype=np.float64).reshape(-1)
    if predicted.shape != contact.shape or contact.shape != free.shape:
        raise ValueError("prediction and counterfactual targets must align")
    target = contact - free
    estimate = predicted - free
    energy = float(target @ target)
    if energy <= float(floor):
        raise ValueError("contact counterfactual chord is degenerate")
    coefficient = float(estimate @ target / energy)
    parallel = coefficient * target
    orthogonal = estimate - parallel
    estimate_norm = float(np.linalg.norm(estimate))
    target_norm = float(np.sqrt(energy))
    cosine = float((estimate @ target) / max(estimate_norm * target_norm, floor))
    distance_contact = float(np.linalg.norm(predicted - contact))
    distance_free = float(np.linalg.norm(predicted - free))
    return {
        "contact_coefficient": coefficient,
        "contact_cosine": cosine,
        "orthogonal_residual_ratio": float(
            np.linalg.norm(orthogonal) / max(target_norm, floor)
        ),
        "distance_to_contact": distance_contact,
        "distance_to_free": distance_free,
        "contact_preference": float(
            (distance_free - distance_contact)
            / max(distance_free + distance_contact, floor)
        ),
        "target_energy": energy,
    }


def intervention_transfer_metrics(baseline, patched, desired, *, floor=1e-12):
    """Measure how much of a desired output displacement an intervention causes."""

    baseline = np.asarray(baseline, dtype=np.float64).reshape(-1)
    patched = np.asarray(patched, dtype=np.float64).reshape(-1)
    desired = np.asarray(desired, dtype=np.float64).reshape(-1)
    if baseline.shape != patched.shape or patched.shape != desired.shape:
        raise ValueError("baseline, patch, and desired output must align")
    moved = patched - baseline
    energy = float(desired @ desired)
    if energy <= float(floor):
        raise ValueError("desired intervention displacement is degenerate")
    coefficient = float(moved @ desired / energy)
    moved_norm = float(np.linalg.norm(moved))
    desired_norm = float(np.sqrt(energy))
    cosine = float((moved @ desired) / max(moved_norm * desired_norm, floor))
    orthogonal = moved - coefficient * desired
    return {
        "transfer_coefficient": coefficient,
        "transfer_cosine": cosine,
        "moved_energy": float(moved @ moved),
        "target_energy": energy,
        "orthogonal_residual_ratio": float(
            np.linalg.norm(orthogonal) / max(desired_norm, floor)
        ),
        "distance_to_desired_ratio": float(
            np.linalg.norm(moved - desired) / max(desired_norm, floor)
        ),
    }


def stepwise_impulse_momentum_residual(
    physics_steps,
    block_impulses,
    block_velocities,
    block_masses,
    *,
    floor=1e-12,
    velocity_tolerance=1e-9,
):
    """Audit per-step collision impulse against post-solve block momentum.

    PushT deliberately sets ``Space.damping=0``. Pymunk therefore clears the
    prior velocity at each physics step, and the post-solve momentum should
    equal the sum of impulses applied to the block during that step. Comparing
    the final velocity with impulses accumulated over the entire rollout would
    be invalid.
    """

    steps = np.asarray(physics_steps, dtype=np.int64).reshape(-1)
    impulses = np.asarray(block_impulses, dtype=np.float64)
    velocities = np.asarray(block_velocities, dtype=np.float64)
    masses = np.asarray(block_masses, dtype=np.float64).reshape(-1)
    if (
        impulses.ndim != 2
        or velocities.shape != impulses.shape
        or len(steps) != len(impulses)
        or len(masses) != len(impulses)
        or impulses.shape[1] != 2
        or not len(steps)
    ):
        raise ValueError("stepwise impulse audit arrays must be aligned nonempty 2-vectors")
    residuals = []
    for step in np.unique(steps):
        selected = np.flatnonzero(steps == step)
        reference_velocity = velocities[selected[0]]
        reference_mass = masses[selected[0]]
        if np.max(np.abs(velocities[selected] - reference_velocity)) > velocity_tolerance:
            raise ValueError("post-solve velocities disagree within one physics step")
        if np.max(np.abs(masses[selected] - reference_mass)) > velocity_tolerance:
            raise ValueError("block mass changed within one physics step")
        impulse = impulses[selected].sum(axis=0)
        momentum = reference_mass * reference_velocity
        denominator = max(float(np.linalg.norm(momentum)), float(np.linalg.norm(impulse)), floor)
        residuals.append(float(np.linalg.norm(momentum - impulse) / denominator))
    return np.asarray(residuals, dtype=np.float64)


def fit_standardized_ridge(features, targets, penalty, *, scale_floor=1e-8):
    """Fit a multi-output ridge map with frozen train-set standardization."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 3:
        raise ValueError("features and targets must be aligned row matrices")
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    x_scale = x.std(axis=0, ddof=1)
    y_scale = y.std(axis=0, ddof=1)
    x_positive = x_scale[x_scale > scale_floor]
    y_positive = y_scale[y_scale > scale_floor]
    if not len(x_positive) or not len(y_positive):
        raise ValueError("ridge inputs have no nondegenerate dimensions")
    x_scale = np.maximum(x_scale, np.median(x_positive) * 1e-3)
    y_scale = np.maximum(y_scale, np.median(y_positive) * 1e-3)
    x_standard = (x - x_mean) / x_scale
    y_standard = (y - y_mean) / y_scale
    gram = x_standard.T @ x_standard
    weight = np.linalg.solve(
        gram + float(penalty) * np.eye(gram.shape[0]),
        x_standard.T @ y_standard,
    )
    return {
        "x_mean": x_mean,
        "x_scale": x_scale,
        "y_mean": y_mean,
        "y_scale": y_scale,
        "weight": weight,
        "penalty": float(penalty),
    }


def predict_standardized_ridge(model, features):
    """Return physical and standardized predictions from ``fit_standardized_ridge``."""

    x = np.asarray(features, dtype=np.float64)
    if x.ndim == 1:
        x = x[None]
    standardized = (
        (x - np.asarray(model["x_mean"])) / np.asarray(model["x_scale"])
    ) @ np.asarray(model["weight"])
    physical = (
        standardized * np.asarray(model["y_scale"])
        + np.asarray(model["y_mean"])
    )
    return physical, standardized


def grouped_ridge_penalty_cv(features, targets, groups, penalties, folds=5):
    """Select a ridge penalty using deterministic trajectory-group folds."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    penalties = [float(value) for value in penalties]
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(groups) != len(x):
        raise ValueError("cross-validation arrays do not align")
    if len(unique) < 2 or not penalties:
        raise ValueError("cross-validation needs groups and penalties")
    folds = min(int(folds), len(unique))
    rows = []
    for penalty in penalties:
        predictions = np.full_like(y, np.nan, dtype=np.float64)
        for fold in range(folds):
            held_groups = unique[np.arange(len(unique)) % folds == fold]
            test = np.isin(groups, held_groups)
            train = ~test
            model = fit_standardized_ridge(x[train], y[train], penalty)
            predictions[test] = predict_standardized_ridge(model, x[test])[0]
        if not np.all(np.isfinite(predictions)):
            raise RuntimeError("grouped ridge predictions are incomplete")
        residual = np.sum((y - predictions) ** 2, axis=0)
        total = np.sum((y - y.mean(axis=0)) ** 2, axis=0)
        r2 = 1.0 - residual / np.maximum(total, 1e-12)
        rows.append(
            {
                "penalty": penalty,
                "mean_r2": float(np.mean(r2)),
                "component_r2": r2.tolist(),
            }
        )
    best = max(rows, key=lambda row: (row["mean_r2"], -row["penalty"]))
    return float(best["penalty"]), rows


def countsketch_adjoint(bucket, sign, scale, weights):
    """Map linear readout covectors from CountSketch space to input space."""

    bucket = np.asarray(bucket, dtype=np.int64).reshape(-1)
    sign = np.asarray(sign, dtype=np.float64).reshape(-1)
    scale = np.asarray(scale, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim == 1:
        weights = weights[:, None]
    if len(bucket) != len(sign) or weights.shape[0] != len(scale):
        raise ValueError("CountSketch metadata and weights do not align")
    return sign[:, None] * weights[bucket] / scale[bucket, None]


def minimum_norm_coordinate_edit(covectors, coordinate_delta, protected=None):
    """Solve the exact minimum-norm edit with optional protected coordinates."""

    covectors = np.asarray(covectors, dtype=np.float64)
    delta = np.asarray(coordinate_delta, dtype=np.float64).reshape(-1)
    if covectors.ndim != 2 or covectors.shape[1] != len(delta):
        raise ValueError("covectors and coordinate target do not align")
    if protected is None:
        protected = np.zeros((covectors.shape[0], 0), dtype=np.float64)
    protected = np.asarray(protected, dtype=np.float64)
    if protected.ndim == 1:
        protected = protected[:, None]
    if protected.ndim != 2 or protected.shape[0] != covectors.shape[0]:
        raise ValueError("protected covectors do not align")
    constraints = np.concatenate([covectors, protected], axis=1)
    target = np.concatenate([delta, np.zeros(protected.shape[1])])
    gram = constraints.T @ constraints
    coefficients = np.linalg.pinv(gram, rcond=1e-12) @ target
    edit = constraints @ coefficients
    achieved = covectors.T @ edit
    protected_drift = protected.T @ edit
    return {
        "edit": edit,
        "achieved": achieved,
        "coordinate_residual_norm": float(np.linalg.norm(achieved - delta)),
        "protected_drift_norm": float(np.linalg.norm(protected_drift)),
        "edit_norm": float(np.linalg.norm(edit)),
        "condition_number": float(np.linalg.cond(gram)),
    }


def orthogonal_random_control(target_norm, excluded, seed):
    """Draw a deterministic norm-matched vector orthogonal to all exclusions."""

    excluded = np.asarray(excluded, dtype=np.float64)
    if excluded.ndim == 1:
        excluded = excluded[:, None]
    if excluded.ndim != 2 or excluded.shape[0] < 1:
        raise ValueError("excluded directions must be a row-aligned matrix")
    rng = np.random.default_rng(int(seed))
    vector = rng.normal(size=excluded.shape[0])
    if excluded.shape[1]:
        vector -= excluded @ (np.linalg.pinv(excluded.T @ excluded) @ (excluded.T @ vector))
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("random control is degenerate")
    return vector * (float(target_norm) / norm)


def r2_components(targets, predictions):
    """Return per-component and mean held-out R-squared."""

    y = np.asarray(targets, dtype=np.float64)
    p = np.asarray(predictions, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 2:
        raise ValueError("targets and predictions must be aligned matrices")
    residual = np.sum((y - p) ** 2, axis=0)
    total = np.sum((y - y.mean(axis=0)) ** 2, axis=0)
    values = 1.0 - residual / np.maximum(total, 1e-12)
    return {"component_r2": values, "mean_r2": float(np.mean(values))}
