"""Numerical primitives for Stage 26 contact-frame causal transport.

The helpers here are deliberately NumPy-only.  The Colab notebook embeds the
same source so that contact-frame construction, grouped model selection, donor
matching, and causal transfer metrics can be unit-tested without a GPU.
"""

from __future__ import annotations

import numpy as np


def wrap_angle(value):
    """Wrap angles to ``[-pi, pi)`` while preserving array shape."""

    value = np.asarray(value, dtype=np.float64)
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def token_centers(grid_size=16, image_size=512.0):
    """Return row-major image-space centers for a square token grid."""

    grid_size = int(grid_size)
    image_size = float(image_size)
    if grid_size < 2 or image_size <= 0:
        raise ValueError("grid and image sizes must be positive")
    centers = (np.arange(grid_size, dtype=np.float64) + 0.5) * (
        image_size / grid_size
    )
    xx, yy = np.meshgrid(centers, centers, indexing="xy")
    return np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)


def contact_frame_basis(
    positions,
    contact_point,
    normal,
    *,
    radius=96.0,
    polynomial_degree=1,
):
    """Construct an orthonormal localized basis in contact coordinates.

    The returned columns span Gaussian-windowed contact-frame monomials.  QR
    signs are canonicalized against the raw monomials so coefficients are
    comparable across examples.
    """

    positions = np.asarray(positions, dtype=np.float64)
    point = np.asarray(contact_point, dtype=np.float64).reshape(-1)
    normal = np.asarray(normal, dtype=np.float64).reshape(-1)
    if positions.ndim != 2 or positions.shape[1] != 2 or point.shape != (2,):
        raise ValueError("positions and contact point must be planar")
    if normal.shape != (2,) or np.linalg.norm(normal) <= 1e-12:
        raise ValueError("contact normal must be a nonzero planar vector")
    if float(radius) <= 0 or int(polynomial_degree) not in {0, 1, 2}:
        raise ValueError("radius must be positive and degree must be 0, 1, or 2")
    normal = normal / np.linalg.norm(normal)
    tangent = np.asarray([-normal[1], normal[0]], dtype=np.float64)
    relative = positions - point[None]
    u = relative @ normal / float(radius)
    v = relative @ tangent / float(radius)
    window = np.exp(-0.5 * (u * u + v * v))
    columns = [window]
    if int(polynomial_degree) >= 1:
        columns.extend([window * u, window * v])
    if int(polynomial_degree) >= 2:
        columns.extend([window * u * u, window * u * v, window * v * v])
    raw = np.stack(columns, axis=1)
    basis, upper = np.linalg.qr(raw, mode="reduced")
    diagonal = np.diag(upper)
    signs = np.where(diagonal < 0.0, -1.0, 1.0)
    basis = basis * signs[None]
    if not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-10):
        raise RuntimeError("contact basis lost orthonormality")
    return basis


def canonical_contact_features(activations, basis):
    """Project a token-by-channel activation field into a contact frame."""

    values = np.asarray(activations, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    if values.ndim != 2 or basis.ndim != 2 or len(values) != len(basis):
        raise ValueError("activation field and contact basis do not align")
    return (basis.T @ values).reshape(-1)


def transport_contact_delta(feature_delta, basis, channels):
    """Map a canonical feature displacement back to the recipient token field."""

    basis = np.asarray(basis, dtype=np.float64)
    delta = np.asarray(feature_delta, dtype=np.float64).reshape(-1)
    channels = int(channels)
    if channels < 1 or len(delta) != basis.shape[1] * channels:
        raise ValueError("feature displacement has the wrong size")
    return basis @ delta.reshape(basis.shape[1], channels)


def response_coordinates(normal_state, ghost_state, impulse, normal):
    """Express the exact contact correction in a canonical physical frame."""

    contact = np.asarray(normal_state, dtype=np.float64).reshape(-1)
    ghost = np.asarray(ghost_state, dtype=np.float64).reshape(-1)
    impulse = np.asarray(impulse, dtype=np.float64).reshape(-1)
    normal = np.asarray(normal, dtype=np.float64).reshape(-1)
    if len(contact) < 5 or contact.shape != ghost.shape:
        raise ValueError("ordinary and ghost states must align")
    if impulse.shape != (2,) or normal.shape != (2,) or np.linalg.norm(normal) <= 1e-12:
        raise ValueError("impulse and normal must be planar")
    normal = normal / np.linalg.norm(normal)
    tangent = np.asarray([-normal[1], normal[0]], dtype=np.float64)
    displacement = contact[2:4] - ghost[2:4]
    lambda_normal = max(float(impulse @ normal), 0.0)
    return np.asarray(
        [
            np.log1p(lambda_normal),
            float(displacement @ normal),
            float(displacement @ tangent),
            float(wrap_angle(contact[4] - ghost[4])),
        ],
        dtype=np.float64,
    )


def fit_standardizer(values, floor=1e-8):
    """Fit a construction-only componentwise standardizer."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) < 3:
        raise ValueError("standardizer requires a row matrix")
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=1)
    positive = scale[scale > float(floor)]
    if not len(positive):
        raise ValueError("all standardizer dimensions are degenerate")
    scale = np.maximum(scale, np.median(positive) * 1e-3)
    return mean, scale


def grouped_ridge_oof(features, targets, groups, penalty, folds=5):
    """Deterministic grouped out-of-fold standardized ridge predictions."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    groups = np.asarray(groups)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(groups) != len(x):
        raise ValueError("grouped ridge arrays do not align")
    unique = np.unique(groups)
    folds = min(int(folds), len(unique))
    if folds < 2:
        raise ValueError("grouped ridge needs at least two groups")
    predictions = np.full_like(y, np.nan, dtype=np.float64)
    for fold in range(folds):
        held = unique[np.arange(len(unique)) % folds == fold]
        test = np.isin(groups, held)
        train = ~test
        x_mean, x_scale = fit_standardizer(x[train])
        y_mean, y_scale = fit_standardizer(y[train])
        xs = (x[train] - x_mean) / x_scale
        ys = (y[train] - y_mean) / y_scale
        if xs.shape[1] > xs.shape[0]:
            weight = xs.T @ np.linalg.solve(
                xs @ xs.T + float(penalty) * np.eye(xs.shape[0]), ys
            )
        else:
            gram = xs.T @ xs
            weight = np.linalg.solve(
                gram + float(penalty) * np.eye(gram.shape[0]), xs.T @ ys
            )
        predictions[test] = (
            ((x[test] - x_mean) / x_scale) @ weight
        ) * y_scale + y_mean
    if not np.all(np.isfinite(predictions)):
        raise RuntimeError("grouped ridge predictions are incomplete")
    return predictions


def r2_components(targets, predictions):
    """Return componentwise and mean R-squared."""

    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if targets.ndim != 2 or targets.shape != predictions.shape:
        raise ValueError("R-squared arrays do not align")
    residual = np.sum((targets - predictions) ** 2, axis=0)
    total = np.sum((targets - targets.mean(axis=0)) ** 2, axis=0)
    values = 1.0 - residual / np.maximum(total, 1e-12)
    return {"component_r2": values, "mean_r2": float(np.mean(values))}


def fit_response_fiber(features, targets, penalty, rank):
    """Fit a standardized response map and return its causal input fiber."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    x_mean, x_scale = fit_standardizer(x)
    y_mean, y_scale = fit_standardizer(y)
    xs = (x - x_mean) / x_scale
    ys = (y - y_mean) / y_scale
    if xs.shape[1] > xs.shape[0]:
        weight = xs.T @ np.linalg.solve(
            xs @ xs.T + float(penalty) * np.eye(xs.shape[0]), ys
        )
    else:
        weight = np.linalg.solve(
            xs.T @ xs + float(penalty) * np.eye(xs.shape[1]), xs.T @ ys
        )
    left, singular, _ = np.linalg.svd(weight, full_matrices=False)
    active_rank = min(int(rank), left.shape[1], int(np.sum(singular > 1e-10)))
    if active_rank < 1:
        raise ValueError("response fiber is degenerate")
    fiber = left[:, :active_rank]
    return {
        "x_mean": x_mean,
        "x_scale": x_scale,
        "y_mean": y_mean,
        "y_scale": y_scale,
        "weight": weight,
        "fiber": fiber,
        "singular_values": singular,
        "penalty": float(penalty),
        "rank": active_rank,
    }


def projected_donor_delta(model, recipient, donor):
    """Return a natural donor-minus-recipient displacement inside the fiber."""

    recipient = np.asarray(recipient, dtype=np.float64).reshape(-1)
    donor = np.asarray(donor, dtype=np.float64).reshape(-1)
    mean = np.asarray(model["x_mean"], dtype=np.float64)
    scale = np.asarray(model["x_scale"], dtype=np.float64)
    fiber = np.asarray(model["fiber"], dtype=np.float64)
    if recipient.shape != mean.shape or donor.shape != mean.shape:
        raise ValueError("donor features do not align with the frozen fiber")
    difference_standard = (donor - recipient) / scale
    projected_standard = fiber @ (fiber.T @ difference_standard)
    return projected_standard * scale


def select_low_response_donor(descriptor, response, donor_descriptors, donor_responses):
    """Choose the closest construction donor with materially lower impulse."""

    descriptor = np.asarray(descriptor, dtype=np.float64).reshape(-1)
    response = np.asarray(response, dtype=np.float64).reshape(-1)
    descriptors = np.asarray(donor_descriptors, dtype=np.float64)
    responses = np.asarray(donor_responses, dtype=np.float64)
    if descriptors.ndim != 2 or responses.ndim != 2 or len(descriptors) != len(responses):
        raise ValueError("donor bank arrays do not align")
    if descriptors.shape[1] != len(descriptor) or responses.shape[1] != len(response):
        raise ValueError("recipient and donor dimensions do not align")
    candidate = np.flatnonzero(responses[:, 0] <= max(0.0, 0.5 * response[0]))
    if not len(candidate):
        candidate = np.argsort(responses[:, 0])[: max(1, min(8, len(responses)))]
    center, scale = fit_standardizer(descriptors)
    distance = np.sum(((descriptors[candidate] - descriptor) / scale) ** 2, axis=1)
    chosen = int(candidate[int(np.argmin(distance))])
    return chosen, float(distance.min())


def intervention_transfer_metrics(baseline, patched, desired, floor=1e-12):
    """Measure signed transfer along a desired output displacement."""

    baseline = np.asarray(baseline, dtype=np.float64).reshape(-1)
    patched = np.asarray(patched, dtype=np.float64).reshape(-1)
    desired = np.asarray(desired, dtype=np.float64).reshape(-1)
    if baseline.shape != patched.shape or patched.shape != desired.shape:
        raise ValueError("intervention vectors do not align")
    moved = patched - baseline
    energy = float(desired @ desired)
    if energy <= float(floor):
        raise ValueError("desired displacement is degenerate")
    coefficient = float(moved @ desired / energy)
    cosine = float(
        moved @ desired
        / max(float(np.linalg.norm(moved) * np.linalg.norm(desired)), float(floor))
    )
    orthogonal = moved - coefficient * desired
    return {
        "transfer_coefficient": coefficient,
        "transfer_cosine": cosine,
        "orthogonal_residual_ratio": float(
            np.linalg.norm(orthogonal) / max(np.linalg.norm(desired), float(floor))
        ),
        "moved_norm": float(np.linalg.norm(moved)),
        "desired_norm": float(np.linalg.norm(desired)),
    }
