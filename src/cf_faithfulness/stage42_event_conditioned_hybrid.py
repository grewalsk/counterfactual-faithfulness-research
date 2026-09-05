"""Pure helpers for the Stage 42 event-conditioned hybrid-closure audit.

Stage 42 repairs a design-support failure in Stage 41.  Candidate evaluation
families may be screened using exact simulator contact incidence, but never
using a learned-model output, prediction error, or intervention magnitude.
The resulting estimand is explicitly conditional on an event-rich finite
bank.  The audit separates three prospective diagnostics: flow composition
away from events, guard information in frozen rollout features, and reset
headroom at contact.  These helpers keep the selection and decision rules
independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike


def macro_contact_flags(
    contact_counts: ArrayLike,
    *,
    length: int,
    frameskip: int,
) -> np.ndarray:
    """Reduce physics-step contact counts to one Boolean per macro action."""

    steps = int(length)
    skip = int(frameskip)
    if steps < 1 or skip < 1:
        raise ValueError("length and frameskip must be positive")
    contacts = np.asarray(contact_counts, dtype=np.int64).reshape(-1)
    if len(contacts) < steps * skip:
        raise ValueError("contact trace is shorter than the registered word")
    return np.asarray(
        [
            np.any(contacts[index * skip : (index + 1) * skip] > 0)
            for index in range(steps)
        ],
        dtype=bool,
    )


def terminal_mode_pair_from_contacts(
    contact_counts: ArrayLike,
    *,
    length: int,
    frameskip: int,
    initial_mode: str,
) -> tuple[str, str]:
    """Return the source/target mode pair at the final macro transition."""

    flags = macro_contact_flags(contact_counts, length=length, frameskip=frameskip)
    mode = str(initial_mode)
    if mode not in {"free", "pre_contact", "contact", "post_contact"}:
        raise ValueError(f"unknown initial mode {mode!r}")
    ever_contact = mode in {"contact", "post_contact"}
    source = [mode]
    for step in range(1, int(length)):
        previous_any = bool(flags[step - 1])
        future_any = bool(flags[step])
        ever_contact = ever_contact or previous_any
        if previous_any:
            label = "contact"
        elif not ever_contact and future_any:
            label = "pre_contact"
        elif ever_contact:
            label = "post_contact"
        else:
            label = "free"
        source.append(label)
    final = "contact" if flags[-1] else (
        "post_contact" if bool(np.any(flags[:-1])) or mode in {"contact", "post_contact"}
        else "free"
    )
    return str(source[-1]), str(final)


def family_reentry_count(
    contact_panels: Sequence[ArrayLike],
    word_lengths: Sequence[int],
    initial_modes: Sequence[str],
    *,
    frameskip: int,
) -> int:
    """Count post-contact-to-contact terminal transitions in one family."""

    if len(contact_panels) != len(initial_modes):
        raise ValueError("one contact panel is required per initial mode")
    lengths = [int(value) for value in word_lengths]
    total = 0
    for panel, mode in zip(contact_panels, initial_modes):
        matrix = np.asarray(panel, dtype=np.int64)
        if matrix.ndim != 2 or matrix.shape[0] != len(lengths):
            raise ValueError("contact panel does not align with registered words")
        for row, length in zip(matrix, lengths):
            pair = terminal_mode_pair_from_contacts(
                row, length=length, frameskip=frameskip, initial_mode=str(mode)
            )
            total += int(pair == ("post_contact", "contact"))
    return int(total)


def select_event_rich_families(
    candidate_ids: Sequence[int],
    reentry_counts: Sequence[int],
    *,
    target_families: int,
    minimum_total_rows: int,
) -> dict[str, Any]:
    """Apply the frozen earliest-event-rich-family selection rule."""

    ids = [int(value) for value in candidate_ids]
    counts = [int(value) for value in reentry_counts]
    target = int(target_families)
    minimum = int(minimum_total_rows)
    if len(ids) != len(counts) or len(set(ids)) != len(ids):
        raise ValueError("candidate identifiers and support counts must align uniquely")
    if target < 1 or minimum < 1 or any(value < 0 for value in counts):
        raise ValueError("support targets and counts must be nonnegative and nonzero")
    selected = [
        (trajectory_id, count)
        for trajectory_id, count in zip(ids, counts)
        if count > 0
    ][:target]
    if len(selected) != target:
        positive = int(sum(value > 0 for value in counts))
        support = int(sum(value for value in counts if value > 0))
        raise RuntimeError(
            "candidate pool contains too few event-rich families: "
            f"found {positive} of {target}; total qualifying rows={support}"
        )
    support = int(sum(value for _, value in selected))
    if support < minimum:
        raise RuntimeError("selected event-rich families do not meet row support")
    return {
        "trajectory_ids": [value for value, _ in selected],
        "family_reentry_counts": {str(key): value for key, value in selected},
        "total_reentry_rows": support,
        "target_families": target,
        "minimum_total_rows": minimum,
        "selection_rule": "earliest_complete_families_with_positive_reentry_incidence",
        "model_outputs_used": False,
        "prediction_errors_used": False,
        "effect_magnitudes_used": False,
    }


@dataclass(frozen=True)
class Stage42SupportDecision:
    passed: bool
    classification: str
    selected_families: int
    total_reentry_rows: int
    minimum_reentry_rows: int


def derive_stage42_support_decision(
    certificate: Mapping[str, Any],
    *,
    expected_families: int,
    minimum_reentry_rows: int,
) -> Stage42SupportDecision:
    """Fail closed unless the prospective support certificate is complete."""

    selected = len(certificate.get("trajectory_ids", []))
    rows = int(certificate.get("total_reentry_rows", -1))
    clean = all(
        certificate.get(key) is False
        for key in ["model_outputs_used", "prediction_errors_used", "effect_magnitudes_used"]
    )
    passed = bool(
        clean
        and selected == int(expected_families)
        and rows >= int(minimum_reentry_rows)
    )
    return Stage42SupportDecision(
        passed=passed,
        classification=(
            "prospective_event_support_certified"
            if passed else "prospective_event_support_not_certified"
        ),
        selected_families=selected,
        total_reentry_rows=rows,
        minimum_reentry_rows=int(minimum_reentry_rows),
    )


def binary_auroc(target: ArrayLike, score: ArrayLike) -> float:
    """Compute a tie-corrected empirical AUROC without external packages."""

    labels = np.asarray(target).reshape(-1)
    values = np.asarray(score, dtype=np.float64).reshape(-1)
    if labels.shape != values.shape or len(labels) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("binary AUROC inputs must be aligned, finite vectors")
    if not np.all(np.isin(labels, [0, 1, False, True])):
        raise ValueError("binary AUROC targets must contain only zero and one")
    labels = labels.astype(bool)
    positive, negative = int(np.sum(labels)), int(np.sum(~labels))
    if positive == 0 or negative == 0:
        raise ValueError("binary AUROC requires both target classes")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    cursor = 0
    while cursor < len(values):
        end = cursor + 1
        while end < len(values) and values[order[end]] == values[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = 0.5 * (cursor + 1 + end)
        cursor = end
    rank_sum = float(np.sum(ranks[labels]))
    return float((rank_sum - positive * (positive + 1) / 2) / (positive * negative))


def balanced_accuracy(
    target: ArrayLike,
    score: ArrayLike,
    *,
    threshold: float,
) -> float:
    """Return binary balanced accuracy at a frozen score threshold."""

    labels = np.asarray(target).reshape(-1)
    values = np.asarray(score, dtype=np.float64).reshape(-1)
    cutoff = float(threshold)
    if labels.shape != values.shape or not np.all(np.isfinite(values)):
        raise ValueError("balanced-accuracy inputs must be aligned and finite")
    if not np.all(np.isin(labels, [0, 1, False, True])) or not np.isfinite(cutoff):
        raise ValueError("balanced-accuracy target or threshold is invalid")
    labels = labels.astype(bool)
    if not np.any(labels) or not np.any(~labels):
        raise ValueError("balanced accuracy requires both target classes")
    predicted = values >= cutoff
    sensitivity = float(np.mean(predicted[labels]))
    specificity = float(np.mean(~predicted[~labels]))
    return float(0.5 * (sensitivity + specificity))


def select_guard_threshold(
    target: ArrayLike,
    score: ArrayLike,
    thresholds: Sequence[float],
) -> dict[str, Any]:
    """Select a guard threshold on a development split only."""

    candidates = sorted(set(float(value) for value in thresholds))
    if not candidates or any(not np.isfinite(value) for value in candidates):
        raise ValueError("guard threshold grid is invalid")
    rows = [
        {
            "threshold": value,
            "balanced_accuracy": balanced_accuracy(target, score, threshold=value),
        }
        for value in candidates
    ]
    selected = max(rows, key=lambda row: (row["balanced_accuracy"], -row["threshold"]))
    return {"selected_threshold": selected["threshold"], "candidate_rows": rows}


def guard_probe_metrics(
    event_target: ArrayLike,
    event_score: ArrayLike,
    event_time_target: ArrayLike,
    event_time_score: ArrayLike,
    *,
    threshold: float,
) -> dict[str, float | int]:
    """Score frozen-feature event occurrence and within-step event time."""

    target = np.asarray(event_target).reshape(-1)
    score = np.asarray(event_score, dtype=np.float64).reshape(-1)
    time_target = np.asarray(event_time_target, dtype=np.float64).reshape(-1)
    time_score = np.asarray(event_time_score, dtype=np.float64).reshape(-1)
    if not (target.shape == score.shape == time_target.shape == time_score.shape):
        raise ValueError("guard-probe arrays are not aligned")
    if not np.all(np.isfinite(np.concatenate([score, time_target, time_score]))):
        raise ValueError("guard-probe arrays must be finite")
    events = target.astype(bool)
    if not np.any(events) or not np.any(~events):
        raise ValueError("guard-probe scoring requires event and non-event rows")
    probability = np.clip(score, 0.0, 1.0)
    return {
        "rows": int(len(target)),
        "event_rows": int(np.sum(events)),
        "event_auroc": binary_auroc(target, score),
        "event_brier": float(np.mean((probability - target.astype(float)) ** 2)),
        "event_balanced_accuracy": balanced_accuracy(
            target, score, threshold=float(threshold)
        ),
        "event_time_mae": float(np.mean(np.abs(time_score[events] - time_target[events]))),
    }


def partition_hybrid_defects(
    composition_error: ArrayLike,
    baseline_physical_error: ArrayLike,
    oracle_reset_error: ArrayLike,
    event_mask: ArrayLike,
    reentry_mask: ArrayLike,
) -> dict[str, float | int]:
    """Partition flow discrepancy and event-conditioned reset headroom.

    The flow term is recursive-versus-direct latent discrepancy and is
    reported on event-free paths.  Guard quality is deliberately scored by a
    separate probe.  The reset term is the physical prediction error on
    re-entry paths together with its oracle-metadata residual.
    """

    composition = np.asarray(composition_error, dtype=np.float64).reshape(-1)
    baseline = np.asarray(baseline_physical_error, dtype=np.float64).reshape(-1)
    oracle = np.asarray(oracle_reset_error, dtype=np.float64).reshape(-1)
    event = np.asarray(event_mask, dtype=bool).reshape(-1)
    reentry = np.asarray(reentry_mask, dtype=bool).reshape(-1)
    if not (composition.shape == baseline.shape == oracle.shape == event.shape == reentry.shape):
        raise ValueError("hybrid-defect arrays are not aligned")
    if not np.all(np.isfinite(np.concatenate([composition, baseline, oracle]))):
        raise ValueError("hybrid-defect errors must be finite")
    smooth = ~event
    if not np.any(smooth) or not np.any(event) or not np.any(reentry):
        raise ValueError("hybrid-defect partition needs smooth, event, and re-entry rows")
    reset_baseline = float(np.mean(baseline[reentry]))
    reset_oracle = float(np.mean(oracle[reentry]))
    return {
        "smooth_rows": int(np.sum(smooth)),
        "event_rows": int(np.sum(event)),
        "reentry_rows": int(np.sum(reentry)),
        "flow_composition_nmse": float(np.mean(composition[smooth])),
        "event_composition_nmse": float(np.mean(composition[event])),
        "reset_baseline_nmse": reset_baseline,
        "reset_oracle_residual_nmse": reset_oracle,
        "reset_oracle_relative_gain": float(
            (reset_baseline - reset_oracle) / max(reset_baseline, 1e-12)
        ),
    }


def propagated_hybrid_error_bound(
    local_errors: ArrayLike,
    lipschitz_factors: ArrayLike,
) -> float:
    """Evaluate sum_k epsilon_k prod_{ell>k} L_ell for a fixed mode path."""

    errors = np.asarray(local_errors, dtype=np.float64).reshape(-1)
    factors = np.asarray(lipschitz_factors, dtype=np.float64).reshape(-1)
    if errors.shape != factors.shape or len(errors) < 1:
        raise ValueError("local errors and Lipschitz factors must align")
    if (
        not np.all(np.isfinite(errors))
        or not np.all(np.isfinite(factors))
        or np.any(errors < 0)
        or np.any(factors < 0)
    ):
        raise ValueError("hybrid error-bound inputs must be finite and nonnegative")
    suffix = 1.0
    total = 0.0
    for error, factor in zip(errors[::-1], factors[::-1]):
        total += float(error) * suffix
        suffix *= float(factor)
    return float(total)


@dataclass(frozen=True)
class Stage42DefectDecision:
    passed: bool
    classification: str
    support_certified: bool
    oracle_reset_headroom: bool
    frozen_guard_identifiable: bool
    learned_event_reset_experiment_authorized: bool


def derive_stage42_defect_decision(
    *,
    support_certified: bool,
    oracle_reset_headroom: bool,
    frozen_guard_identifiable: bool,
) -> Stage42DefectDecision:
    """Choose the next experiment without conflating oracle and learned evidence."""

    support = bool(support_certified)
    oracle = bool(oracle_reset_headroom)
    guard = bool(frozen_guard_identifiable)
    authorized = bool(support and oracle and guard)
    if not support:
        classification = "event_support_not_certified"
    elif not oracle:
        classification = "no_oracle_reset_headroom"
    elif not guard:
        classification = "oracle_headroom_but_guard_information_insufficient"
    else:
        classification = "learned_event_reset_experiment_authorized"
    return Stage42DefectDecision(
        passed=authorized,
        classification=classification,
        support_certified=support,
        oracle_reset_headroom=oracle,
        frozen_guard_identifiable=guard,
        learned_event_reset_experiment_authorized=authorized,
    )
