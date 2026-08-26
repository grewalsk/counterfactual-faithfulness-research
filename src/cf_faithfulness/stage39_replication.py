"""Numerical decisions for fresh coefficient-matched replications.

The Stage 39 notebooks embed these functions verbatim so that the Colab
artifact and the locally tested implementation share one source.  The primary
estimand is the mean paired row-wise relative gain of the full objective over
the coefficient-matched latent-only control.  Positive values favor the full
objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike


def paired_rowwise_relative_gain(
    full_errors: ArrayLike,
    matched_errors: ArrayLike,
    *,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Return ``(matched - full) / matched`` for exactly paired errors."""

    full = np.asarray(full_errors, dtype=np.float64)
    matched = np.asarray(matched_errors, dtype=np.float64)
    if full.shape != matched.shape or full.ndim != 2:
        raise ValueError("paired errors must be seed-by-row matrices of equal shape")
    if not np.all(np.isfinite(full)) or not np.all(np.isfinite(matched)):
        raise ValueError("paired errors must be finite")
    if np.any(full < 0) or np.any(matched < 0):
        raise ValueError("paired errors must be nonnegative")
    if float(epsilon) <= 0:
        raise ValueError("epsilon must be positive")
    return (matched - full) / np.maximum(matched, float(epsilon))


def pooled_ratio_of_means_gain(
    full_errors: ArrayLike,
    matched_errors: ArrayLike,
    *,
    epsilon: float = 1e-12,
) -> float:
    """Return the explicitly labeled pooled ratio-of-means sensitivity."""

    full = np.asarray(full_errors, dtype=np.float64)
    matched = np.asarray(matched_errors, dtype=np.float64)
    if full.shape != matched.shape or not full.size:
        raise ValueError("pooled errors must be nonempty and exactly paired")
    if not np.all(np.isfinite(full)) or not np.all(np.isfinite(matched)):
        raise ValueError("pooled errors must be finite")
    denominator = max(float(np.mean(matched)), float(epsilon))
    return float((np.mean(matched) - np.mean(full)) / denominator)


def hierarchical_seed_family_interval(
    values: ArrayLike,
    family_ids: ArrayLike,
    *,
    draws: int,
    seed: int,
    confidence: float = 0.90,
) -> tuple[float, float]:
    """Bootstrap seeds and trajectory families while preserving paired rows.

    ``values`` is seed-by-row.  Rows belonging to a sampled family are kept
    together, so repeated records and action words are not treated as
    independent experimental units.
    """

    matrix = np.asarray(values, dtype=np.float64)
    groups = np.asarray(family_ids)
    if matrix.ndim != 2 or matrix.shape[1] != len(groups):
        raise ValueError("values must be seed-by-row and align with family_ids")
    if matrix.shape[0] < 2 or len(np.unique(groups)) < 2:
        raise ValueError("hierarchical bootstrap requires two seeds and two families")
    if int(draws) < 32:
        raise ValueError("at least 32 bootstrap draws are required")
    if not 0 < float(confidence) < 1:
        raise ValueError("confidence must lie in (0, 1)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("bootstrap values must be finite")

    unique = np.unique(groups)
    rows = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(int(draws), dtype=np.float64)
    for draw in range(int(draws)):
        sampled_seeds = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        sampled_groups = rng.choice(unique, size=len(unique), replace=True)
        pieces = [
            matrix[np.ix_(sampled_seeds, rows[group])].reshape(-1)
            for group in sampled_groups
        ]
        estimates[draw] = float(np.mean(np.concatenate(pieces)))
    alpha = (1.0 - float(confidence)) / 2.0
    return (
        float(np.quantile(estimates, alpha)),
        float(np.quantile(estimates, 1.0 - alpha)),
    )


@dataclass(frozen=True)
class Stage39PanelDecision:
    mean_gain: float
    interval90: tuple[float, float]
    equivalence_margin: float
    quality_control_passed: bool
    classification: str


def derive_stage39_panel_decision(
    mean_gain: float,
    interval90: Sequence[float],
    *,
    equivalence_margin: float = 0.05,
    quality_control_passed: bool = True,
) -> Stage39PanelDecision:
    """Classify a panel using a fixed symmetric practical-equivalence band."""

    mean = float(mean_gain)
    interval = tuple(float(value) for value in interval90)
    margin = float(equivalence_margin)
    if len(interval) != 2 or not np.all(np.isfinite([mean, *interval])):
        raise ValueError("panel estimate and interval must be finite")
    if interval[0] > interval[1] or margin <= 0:
        raise ValueError("invalid interval or equivalence margin")
    if not bool(quality_control_passed):
        classification = "invalid_quality_control"
    elif interval[0] >= -margin and interval[1] <= margin:
        classification = "practically_equivalent"
    elif interval[0] >= margin:
        classification = "full_objective_specificity"
    elif interval[1] <= -margin:
        classification = "coefficient_matched_superiority"
    else:
        classification = "inconclusive"
    return Stage39PanelDecision(
        mean_gain=mean,
        interval90=interval,
        equivalence_margin=margin,
        quality_control_passed=bool(quality_control_passed),
        classification=classification,
    )


def derive_stage39_decision(
    panels: Mapping[str, Stage39PanelDecision],
) -> dict[str, Any]:
    """Make a conjunctive two-predictor decision without pooling panels."""

    required = {"jepa", "dino"}
    if set(panels) != required:
        raise ValueError(f"Stage 39 requires exactly {sorted(required)}")
    classifications = {key: panels[key].classification for key in sorted(panels)}
    if any(value == "invalid_quality_control" for value in classifications.values()):
        status = "invalid_quality_control"
    elif all(value == "practically_equivalent" for value in classifications.values()):
        status = "coefficient_matched_equivalence_replicated"
    elif all(value == "full_objective_specificity" for value in classifications.values()):
        status = "full_objective_specificity_confirmed"
    elif all(value == "coefficient_matched_superiority" for value in classifications.values()):
        status = "coefficient_matched_superiority_confirmed"
    else:
        status = "heterogeneous_or_inconclusive"
    return {
        "status": status,
        "passed": status in {
            "coefficient_matched_equivalence_replicated",
            "full_objective_specificity_confirmed",
            "coefficient_matched_superiority_confirmed",
        },
        "panels_pooled": False,
        "panel_classifications": classifications,
        "equivalence_replicated": status
        == "coefficient_matched_equivalence_replicated",
        "full_objective_specificity_confirmed": status
        == "full_objective_specificity_confirmed",
    }
