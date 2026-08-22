"""Decision semantics for Stage 37.1 horizon-matched operator calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


def select_horizon_control_candidate(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select validation score, then the simpler deterministic architecture."""

    if not rows:
        raise ValueError("horizon-control selection requires candidates")
    required = {
        "latent_dim", "dynamics", "physical_nmse", "semigroup_nmse",
        "validation_score",
    }
    candidates = []
    for row in rows:
        if not required.issubset(row):
            raise ValueError("horizon-control candidate row is incomplete")
        values = [
            float(row["physical_nmse"]), float(row["semigroup_nmse"]),
            float(row["validation_score"]),
        ]
        if not np.all(np.isfinite(values)):
            raise ValueError("horizon-control candidate is nonfinite")
        candidates.append(dict(row))
    candidates.sort(key=lambda row: (
        float(row["validation_score"]),
        int(row["latent_dim"]),
        str(row["dynamics"]),
    ))
    return candidates[0]


@dataclass(frozen=True)
class Stage371Gates:
    source_and_split_binding: bool
    development_preflight: bool
    locked_physical_closure: bool
    locked_semigroup_closure: bool
    objective_specificity: bool
    horizon_family_consistency: bool
    mode_family_consistency: bool


def derive_stage371_decision(
    gates: Stage371Gates, *, run_mode: str
) -> dict[str, Any]:
    checks = {
        "source_and_split_binding": bool(gates.source_and_split_binding),
        "development_preflight": bool(gates.development_preflight),
        "locked_physical_closure": bool(gates.locked_physical_closure),
        "locked_semigroup_closure": bool(gates.locked_semigroup_closure),
        "objective_specificity": bool(gates.objective_specificity),
        "horizon_family_consistency": bool(gates.horizon_family_consistency),
        "mode_family_consistency": bool(gates.mode_family_consistency),
    }
    first_failed = next((name for name, passed in checks.items() if not passed), None)
    if str(run_mode) == "smoke":
        status, passed = "smoke_complete_not_evidence", False
    elif first_failed is None:
        status, passed = "horizon_matched_operator_class_calibrated", True
    else:
        labels = {
            "source_and_split_binding": "invalid_source_or_split_binding",
            "development_preflight": "operator_failed_development_preflight",
            "locked_physical_closure": "operator_failed_locked_physical_closure",
            "locked_semigroup_closure": "operator_failed_locked_semigroup_closure",
            "objective_specificity": "semigroup_objective_not_specific",
            "horizon_family_consistency": "operator_not_horizon_consistent",
            "mode_family_consistency": "operator_not_mode_consistent",
        }
        status, passed = labels[first_failed], False
    return {
        "status": status,
        "passed": passed,
        "first_failed_gate": first_failed,
        "gates": checks,
        "jepa_loaded": False,
        "jepa_result_claimed": False,
        "authorizes_fresh_jepa_confirmation": bool(passed),
        "simulator_operator_calibration_only": True,
    }
