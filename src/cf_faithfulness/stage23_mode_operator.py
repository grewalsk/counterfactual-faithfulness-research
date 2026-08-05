"""Numerical primitives for the Stage 23 causal mode/operator-switch test.

Stage 23 asks whether moving a frozen JEPA activation across the label-free
mode boundary discovered in Stage 22 changes its downstream finite-response
operator.  The notebook owns model execution and hooks; this module keeps the
geometry and scoring independently testable with NumPy.
"""

from __future__ import annotations

import numpy as np


def countsketch_mode_covectors(
    bucket,
    sign,
    bucket_scale,
    partition_scale,
    components,
    *,
    center_factor=1.0,
):
    """Pull frozen PCA mode coordinates back through a CountSketch.

    Returns a matrix ``A`` such that ``delta @ A`` is the exact change in the
    frozen standardized-PCA coordinates caused by a flattened native-space
    carrier edit ``delta``.  ``center_factor`` accounts for candidate
    centering when only one candidate in an N-candidate state is edited.
    """

    buckets = np.asarray(bucket, dtype=np.int64).reshape(-1)
    signs = np.asarray(sign, dtype=np.float64).reshape(-1)
    sketch_scale = np.asarray(bucket_scale, dtype=np.float64).reshape(-1)
    feature_scale = np.asarray(partition_scale, dtype=np.float64).reshape(-1)
    rotation = np.asarray(components, dtype=np.float64)
    if buckets.shape != signs.shape or len(buckets) < 1:
        raise ValueError("CountSketch bucket and sign arrays must align")
    if rotation.ndim != 2 or rotation.shape[0] != len(feature_scale):
        raise ValueError("partition components do not match partition scale")
    if len(sketch_scale) != len(feature_scale):
        raise ValueError("CountSketch width does not match the partition")
    if np.any(buckets < 0) or np.any(buckets >= len(sketch_scale)):
        raise ValueError("CountSketch bucket index is out of range")
    if np.any(sketch_scale <= 0) or np.any(feature_scale <= 0):
        raise ValueError("all sketch and partition scales must be positive")
    if not np.isfinite(center_factor):
        raise ValueError("center_factor must be finite")
    standardized_rotation = rotation / feature_scale[:, None]
    weights = signs / sketch_scale[buckets]
    return (
        float(center_factor)
        * weights[:, None]
        * standardized_rotation[buckets]
    )


def native_to_whitened_covectors(native_covectors, channel_square_root, channels):
    """Express native activation covectors in channel-whitened coordinates.

    Stage 23 represents an edit by ``delta_native = delta_white @ S.T`` per
    token, where ``S`` is the Stage 14 channel metric square root.  Therefore
    the corresponding whitened covector is ``S.T @ a_native`` per token.
    """

    native = np.asarray(native_covectors, dtype=np.float64)
    root = np.asarray(channel_square_root, dtype=np.float64)
    channels = int(channels)
    if native.ndim != 2 or channels < 1 or native.shape[0] % channels:
        raise ValueError("native covectors must flatten whole channel vectors")
    if root.shape != (channels, channels):
        raise ValueError("channel square root has the wrong shape")
    tokens = native.shape[0] // channels
    shaped = native.reshape(tokens, channels, native.shape[1])
    # delta_native[t, d] = sum_c delta_white[t, c] * root[d, c]
    whitened = np.einsum("dc,tdr->tcr", root, shaped, optimize=True)
    return whitened.reshape(native.shape)


def minimal_constrained_transport(
    mode_covectors,
    protected_basis,
    coordinate_delta,
    *,
    ridge=1e-10,
):
    """Return the minimum-norm edit with exact mode and protection constraints.

    For mode covectors A, protected content directions U, and requested mode
    displacement dq, solve

        min ||delta||_2  subject to A.T delta = dq and U.T delta = 0.

    A small numerical ridge stabilizes the Gram solve.  Diagnostics expose the
    actual residuals, so notebook validity never rests on the ridge alone.
    """

    mode = np.asarray(mode_covectors, dtype=np.float64)
    protected = np.asarray(protected_basis, dtype=np.float64)
    target = np.asarray(coordinate_delta, dtype=np.float64).reshape(-1)
    if mode.ndim != 2 or mode.shape[1] != len(target):
        raise ValueError("mode covectors and target coordinates do not align")
    if protected.ndim == 1:
        protected = protected[:, None]
    if protected.ndim != 2 or protected.shape[0] != mode.shape[0]:
        raise ValueError("protected basis does not match activation width")
    if not np.all(np.isfinite(mode)) or not np.all(np.isfinite(protected)):
        raise ValueError("constraint matrices contain nonfinite values")
    constraints = np.concatenate([mode, protected], axis=1)
    rhs = np.concatenate([target, np.zeros(protected.shape[1])])
    gram = constraints.T @ constraints
    stabilizer = max(float(ridge), 0.0) * max(float(np.trace(gram)), 1.0)
    coefficients = np.linalg.solve(
        gram + stabilizer * np.eye(gram.shape[0]), rhs
    )
    delta = constraints @ coefficients
    achieved = mode.T @ delta
    protected_projection = protected.T @ delta
    return {
        "delta": delta,
        "achieved_coordinate_delta": achieved,
        "mode_residual_norm": float(np.linalg.norm(achieved - target)),
        "protected_projection_norm": float(np.linalg.norm(protected_projection)),
        "edit_norm": float(np.linalg.norm(delta)),
        "constraint_condition_number": float(np.linalg.cond(gram)),
    }


def symmetric_finite_response(plus, minus, dose):
    """Compute a symmetric finite intervention response at a fixed dose."""

    positive = np.asarray(plus, dtype=np.float64)
    negative = np.asarray(minus, dtype=np.float64)
    dose = float(dose)
    if positive.shape != negative.shape:
        raise ValueError("positive and negative responses must align")
    if not np.isfinite(dose) or dose <= 0:
        raise ValueError("dose must be finite and positive")
    return (positive - negative) / (2.0 * dose)


def operator_transfer_metrics(off_response, context_response, native_on_response):
    """Score how much a context edit moves a response toward the native-on one."""

    off = np.asarray(off_response, dtype=np.float64).reshape(-1)
    context = np.asarray(context_response, dtype=np.float64).reshape(-1)
    native = np.asarray(native_on_response, dtype=np.float64).reshape(-1)
    if off.shape != context.shape or off.shape != native.shape or not len(off):
        raise ValueError("operator responses must be aligned nonempty arrays")
    target = native - off
    moved = context - off
    target_energy = float(target @ target)
    moved_energy = float(moved @ moved)
    cross = float(moved @ target)
    coefficient = cross / max(target_energy, 1e-12)
    cosine = cross / max(np.sqrt(target_energy * moved_energy), 1e-12)
    residual = moved - coefficient * target
    return {
        "target_energy": target_energy,
        "moved_energy": moved_energy,
        "transfer_coefficient": float(coefficient),
        "transfer_cosine": float(cosine),
        "orthogonal_residual_ratio": float(
            np.linalg.norm(residual) / max(np.sqrt(target_energy), 1e-12)
        ),
        "distance_to_native_ratio": float(
            np.linalg.norm(context - native) / max(np.linalg.norm(target), 1e-12)
        ),
    }


def deterministic_signed_permutation(values, seed):
    """Apply a reproducible non-identity signed coordinate permutation."""

    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(vector) < 2:
        raise ValueError("signed permutation needs at least two coordinates")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(vector))
    if np.array_equal(order, np.arange(len(vector))):
        order = np.roll(order, 1)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(vector))
    return vector[order] * signs, order, signs
