"""Numerical primitives for Stage 20 causal planner steering.

Stage 20 keeps the Stage 18 representation and Stage 19 transfer result frozen,
then tests a downstream causal chain from predicted consequence to action rank,
chosen action, and simulator outcome.  These helpers are NumPy-only so the
counterfactual choice contract can be tested independently of Colab and JEPA.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .stage18_rank_confirmation import (
    action_contrast_energy_metrics,
    action_swap_delta,
    candidate_center,
    donor_transfer_metrics,
    exact_positive_sign_test,
    matched_common_mode,
    physical_diversity_metrics,
    pose_target,
    projection_ablation_delta,
)
from .stage17_action_contrast import decoded_task_cost
from .stage19_unseen_action_transfer import (
    unseen_action_bank,
    validate_stage18_subspace_arrays,
)


def targeted_derangement(size, target, donor, seed):
    """Return a derangement whose target receives the donor's value.

    The remaining indices form one deterministic random cycle.  Consequently
    every candidate changes identity, ``permutation[target] == donor``, and a
    complete interchange makes the target inherit the donor's score.
    """

    size, target, donor = int(size), int(target), int(donor)
    if size < 3 or not (0 <= target < size) or not (0 <= donor < size):
        raise ValueError("invalid targeted-derangement arguments")
    if target == donor:
        raise ValueError("target and donor must differ")
    remaining = [value for value in range(size) if value not in {target, donor}]
    digest = hashlib.sha256(str(seed).encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    rng.shuffle(remaining)
    cycle = [target, donor, *remaining]
    permutation = np.empty(size, dtype=np.int64)
    for left, right in zip(cycle, cycle[1:] + cycle[:1]):
        permutation[left] = right
    if permutation[target] != donor or np.any(permutation == np.arange(size)):
        raise RuntimeError("failed to construct targeted derangement")
    return permutation


def stable_action_rank(values, action):
    """Zero-based stable ascending rank of one action (lower is better)."""

    scores = np.asarray(values, dtype=np.float64)
    action = int(action)
    if scores.ndim != 1 or not 0 <= action < len(scores):
        raise ValueError("scores and action are incompatible")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores contain nonfinite values")
    order = np.argsort(scores, kind="stable")
    return int(np.flatnonzero(order == action)[0])


def select_near_frontier_targets(scores, ranks=(1, 2, 3)):
    """Select fixed baseline rank positions without simulator-outcome access."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("scores must be a finite vector")
    order = np.argsort(values, kind="stable")
    ranks = tuple(int(value) for value in ranks)
    if len(set(ranks)) != len(ranks) or min(ranks) < 1 or max(ranks) >= len(values):
        raise ValueError("target ranks must be unique non-best positions")
    return int(order[0]), [int(order[value]) for value in ranks]


def planner_steering_metrics(
    baseline_scores,
    patched_scores,
    true_costs,
    permutation,
    target_action,
):
    """Score prediction, ranking, choice, and physical consequences of an edit."""

    baseline = np.asarray(baseline_scores, dtype=np.float64)
    patched = np.asarray(patched_scores, dtype=np.float64)
    physical = np.asarray(true_costs, dtype=np.float64)
    permutation = np.asarray(permutation, dtype=np.int64)
    target_action = int(target_action)
    if not (
        baseline.ndim == 1
        and patched.shape == baseline.shape
        and physical.shape == baseline.shape
        and permutation.shape == baseline.shape
    ):
        raise ValueError("planner metric vectors must be aligned")
    if sorted(permutation.tolist()) != list(range(len(baseline))):
        raise ValueError("permutation is malformed")
    if not all(np.all(np.isfinite(value)) for value in [baseline, patched, physical]):
        raise ValueError("planner metric vectors contain nonfinite values")

    expected = baseline[permutation]
    expected_choice = int(np.argmin(expected))
    baseline_choice = int(np.argmin(baseline))
    patched_choice = int(np.argmin(patched))
    transfer = donor_transfer_metrics(
        baseline[:, None], patched[:, None], permutation
    )
    denominator = float(np.sqrt(np.mean((expected - baseline) ** 2)))
    normalized_error = float(
        np.sqrt(np.mean((patched - expected) ** 2)) / max(denominator, 1e-12)
    )
    baseline_rank = stable_action_rank(baseline, target_action)
    patched_rank = stable_action_rank(patched, target_action)
    return {
        "score_transfer_coefficient": transfer["coefficient"],
        "score_transfer_cosine": transfer["cosine"],
        "score_counterfactual_normalized_rmse": normalized_error,
        "baseline_choice": baseline_choice,
        "expected_counterfactual_choice": expected_choice,
        "patched_choice": patched_choice,
        "target_action": target_action,
        "target_is_expected_choice": bool(expected_choice == target_action),
        "target_rank_baseline": baseline_rank,
        "target_rank_patched": patched_rank,
        "target_rank_gain": int(baseline_rank - patched_rank),
        "target_selected": bool(patched_choice == target_action),
        "choice_matches_counterfactual": bool(patched_choice == expected_choice),
        "choice_flipped": bool(patched_choice != baseline_choice),
        "baseline_selected_true_cost": float(physical[baseline_choice]),
        "patched_selected_true_cost": float(physical[patched_choice]),
        "selected_true_cost_change": float(
            physical[patched_choice] - physical[baseline_choice]
        ),
        "target_true_cost": float(physical[target_action]),
    }
