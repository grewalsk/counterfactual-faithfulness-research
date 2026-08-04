"""Numerical primitives for Stage 19 unseen-action transfer.

Stage 19 does not fit a representation.  It imports the exact frozen Stage 18
projectors and asks whether their bidirectional causal effect transfers to
action banks that were absent from both Stage 18 construction and evaluation.
The helpers here keep the action-family contract and artifact checks testable
without a simulator, GPU, or model checkpoint.
"""

from __future__ import annotations

import numpy as np

from .stage18_rank_confirmation import (  # re-export frozen causal algebra
    action_contrast_energy_metrics,
    action_swap_delta,
    candidate_center,
    donor_transfer_metrics,
    exact_positive_sign_test,
    fixed_derangement,
    matched_common_mode,
    nested_orthonormalize_basis,
    physical_diversity_metrics,
    pose_target,
    projection_ablation_delta,
)
from .stage17_action_contrast import decoded_task_cost


TRANSFER_FAMILIES = (
    "rotated_direction",
    "magnitude_0p08",
    "magnitude_0p16",
    "delayed_equal_impulse",
    "pulsed_equal_impulse",
)


def rotate_vector(vector, angle):
    """Rotate a two-dimensional vector by ``angle`` radians."""

    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (2,):
        raise ValueError("vector must have shape (2,)")
    cosine, sine = np.cos(float(angle)), np.sin(float(angle))
    return np.asarray(
        [cosine * value[0] - sine * value[1],
         sine * value[0] + cosine * value[1]],
        dtype=np.float64,
    )


def unseen_action_bank(toward_block, family, steps=15):
    """Return the preregistered no-op plus twelve antithetic actions.

    The Stage 18 bank used twelve constant directions separated by 30 degrees
    at magnitude 0.12.  Stage 19 holds out either the angular midpoints, two
    new magnitudes, or two new equal-impulse temporal profiles.  Temporal
    profiles have ten active steps at magnitude 0.18, so their vector sum is
    equal to the Stage 18 constant profile (fifteen steps at 0.12).
    """

    direction = np.asarray(toward_block, dtype=np.float64)
    if direction.shape != (2,) or not np.all(np.isfinite(direction)):
        raise ValueError("toward_block must be a finite two-vector")
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("toward_block is degenerate")
    direction = direction / norm
    if family not in TRANSFER_FAMILIES:
        raise ValueError(f"unknown transfer family {family!r}")
    if int(steps) != 15:
        raise ValueError("Stage 19 is frozen to fifteen environment steps")

    branches = [np.zeros((steps, 2), dtype=np.float64)]
    for index in range(12):
        phase = 2.0 * np.pi * index / 12.0
        if family == "rotated_direction":
            phase += np.pi / 12.0
        radial = rotate_vector(direction, phase)
        if family == "magnitude_0p08":
            profile = np.full(steps, 0.08, dtype=np.float64)
        elif family == "magnitude_0p16":
            profile = np.full(steps, 0.16, dtype=np.float64)
        elif family == "delayed_equal_impulse":
            profile = np.r_[np.zeros(5), np.full(10, 0.18)]
        elif family == "pulsed_equal_impulse":
            profile = np.r_[np.full(5, 0.18), np.zeros(5), np.full(5, 0.18)]
        else:
            profile = np.full(steps, 0.12, dtype=np.float64)
        branches.append(profile[:, None] * radial[None, :])

    actions = np.stack(branches).astype(np.float32)
    if actions.shape != (13, steps, 2):
        raise RuntimeError(f"bad Stage 19 action-bank shape {actions.shape}")
    for index in range(1, 7):
        if not np.allclose(actions[index], -actions[index + 6], atol=1e-7):
            raise RuntimeError("Stage 19 action bank lost antithetic pairing")
    return actions


def validate_stage18_subspace_arrays(arrays, ambient=102400, max_rank=128):
    """Fail closed if the imported Stage 18 artifact violates its contract."""

    required = {
        "primary_basis",
        "shuffled_basis",
        "channel_square_root",
        "channel_inverse_square_root",
        *(f"random_basis_{draw:02d}" for draw in range(4)),
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"Stage 18 artifact is missing arrays: {missing}")
    basis_names = [
        "primary_basis",
        "shuffled_basis",
        *(f"random_basis_{draw:02d}" for draw in range(4)),
    ]
    errors = {}
    for name in basis_names:
        basis = np.asarray(arrays[name], dtype=np.float64)
        if basis.shape != (int(ambient), int(max_rank)):
            raise ValueError(f"{name} has shape {basis.shape}")
        error = float(np.max(np.abs(basis.T @ basis - np.eye(max_rank))))
        if not np.isfinite(error) or error > 1e-10:
            raise ValueError(f"{name} is not orthonormal: {error}")
        errors[name] = error
    for name in ["channel_square_root", "channel_inverse_square_root"]:
        if np.asarray(arrays[name]).shape != (400, 400):
            raise ValueError(f"{name} must have shape (400, 400)")
    return {"validated": True, "orthonormality_max_errors": errors}
