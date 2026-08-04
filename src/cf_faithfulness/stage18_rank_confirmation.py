"""Numerical primitives for Stage 18 rank-64 causal confirmation.

Stage 18 reuses the finite action-contrast algebra from Stage 17 and adds
necessity and simulator-diversity estimands.  Keeping these functions NumPy
only makes the central claims independently testable without a GPU or model
checkpoint.
"""

from __future__ import annotations

import math

import numpy as np

from .stage17_action_contrast import (  # re-export the frozen Stage 17 algebra
    action_swap_delta,
    candidate_center,
    decoded_task_cost,
    donor_transfer_metrics,
    exact_positive_sign_test,
    fit_dual_ridge_basis,
    fixed_derangement,
    grouped_kernel_ridge_cv,
    linear_cka,
    matched_common_mode,
    pose_target,
    random_subspace_in_span,
)


def projection_ablation_delta(values, basis, dose=1.0):
    """Remove an action-centered component lying in ``span(basis)``.

    At dose one, adding the returned delta to ``values`` leaves the shared
    candidate mean unchanged and deletes the projected action contrast.
    """

    array = np.asarray(values, dtype=np.float64)
    original_shape = array.shape
    flat = array.reshape(array.shape[0], -1)
    directions = np.asarray(basis, dtype=np.float64)
    if directions.ndim != 2 or directions.shape[0] != flat.shape[1]:
        raise ValueError("basis does not match flattened activation width")
    if not np.allclose(
        directions.T @ directions,
        np.eye(directions.shape[1]),
        atol=1e-6,
        rtol=1e-6,
    ):
        raise ValueError("basis columns must be orthonormal")
    residual = candidate_center(flat)
    projected = (residual @ directions) @ directions.T
    return (-float(dose) * projected).reshape(original_shape)


def action_contrast_energy_metrics(baseline, patched):
    """Measure how much candidate-specific output energy survives an edit."""

    base = np.asarray(baseline, dtype=np.float64).reshape(len(baseline), -1)
    edit = np.asarray(patched, dtype=np.float64).reshape(len(patched), -1)
    if base.shape != edit.shape:
        raise ValueError("baseline and patched outputs must have equal shape")
    centered_base = candidate_center(base)
    centered_edit = candidate_center(edit)
    baseline_energy = float(np.sum(centered_base**2))
    patched_energy = float(np.sum(centered_edit**2))
    if baseline_energy <= 1e-12:
        return {
            "baseline_energy": baseline_energy,
            "patched_energy": patched_energy,
            "energy_retention": math.nan,
            "energy_reduction": math.nan,
            "contrast_cosine": math.nan,
        }
    retention = patched_energy / baseline_energy
    cosine_denominator = math.sqrt(baseline_energy * patched_energy)
    cosine = (
        float(np.sum(centered_base * centered_edit) / cosine_denominator)
        if cosine_denominator > 1e-12
        else 0.0
    )
    return {
        "baseline_energy": baseline_energy,
        "patched_energy": patched_energy,
        "energy_retention": float(retention),
        "energy_reduction": float(1.0 - retention),
        "contrast_cosine": cosine,
    }


def physical_diversity_metrics(costs, interaction_counts, tie=1e-4):
    """Return model-blind action-bank eligibility statistics for one state."""

    values = np.asarray(costs, dtype=np.float64)
    contacts = np.asarray(interaction_counts)
    if values.ndim != 1 or contacts.shape != values.shape or len(values) < 2:
        raise ValueError("costs and contact counts must be aligned vectors")
    left, right = np.triu_indices(len(values), k=1)
    margins = np.abs(values[left] - values[right])
    return {
        "cost_min": float(np.min(values)),
        "cost_max": float(np.max(values)),
        "cost_spread": float(np.max(values) - np.min(values)),
        "non_tied_pair_fraction": float(np.mean(margins > float(tie))),
        "contact_branches": int(np.sum(contacts > 0)),
        "total_contacts": int(np.sum(contacts)),
    }


def lower_triangle_principal_overlap(left, right):
    """Average squared principal cosine between two orthonormal subspaces."""

    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]:
        raise ValueError("subspace bases must share an ambient dimension")
    denominator = max(min(a.shape[1], b.shape[1]), 1)
    return float(np.sum((a.T @ b) ** 2) / denominator)
