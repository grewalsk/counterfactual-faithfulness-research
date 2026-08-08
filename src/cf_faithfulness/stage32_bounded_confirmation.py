"""Numerical primitives for Stage 32 bounded cross-model confirmation.

Stage 32 removes the coefficient singularity exposed by Stage 31.  Grounded
closure is represented only by a bounded cosine and is defined only when the
physical target contrast has prespecified nontrivial energy.  The functions in
this module are NumPy-only so the confirmatory estimands can be tested without
loading a world model.
"""

from __future__ import annotations

import math

import numpy as np


def bounded_cosine(source, target, minimum_target_energy=1e-6):
    """Return a bounded cosine, failing closed below a target-energy floor."""

    left = np.asarray(source, dtype=np.float64)
    right = np.asarray(target, dtype=np.float64)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("source and target must have the same nonempty shape")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("source and target must be finite")
    floor = float(minimum_target_energy)
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("minimum_target_energy must be positive and finite")
    left = left.reshape(-1)
    right = right.reshape(-1)
    source_energy = float(np.dot(left, left))
    target_energy = float(np.dot(right, right))
    if target_energy < floor:
        return {
            "eligible": False,
            "source_energy": source_energy,
            "target_energy": target_energy,
            "cosine": math.nan,
        }
    denominator = math.sqrt(max(source_energy * target_energy, 0.0))
    cosine = float(np.dot(left, right) / denominator) if denominator > 1e-20 else 0.0
    return {
        "eligible": True,
        "source_energy": source_energy,
        "target_energy": target_energy,
        "cosine": float(np.clip(cosine, -1.0, 1.0)),
    }


def bounded_swap_closure_rows(
    baseline,
    patched,
    target,
    magnitude_count,
    schedule_count,
    diagnostic_schedules=(1, 2, 3, 4),
    minimum_target_energy=1e-6,
):
    """Measure bounded self and grounded swap closure per magnitude.

    Closure uses only the interior schedules.  Schedule reversal is defined by
    reversing the six frozen schedule positions; extreme schedules can remain
    reserved for planning goals.
    """

    magnitude_count = int(magnitude_count)
    schedule_count = int(schedule_count)
    expected = magnitude_count * schedule_count

    def layout(values, name):
        array = np.asarray(values, dtype=np.float64)
        if array.ndim < 2 or len(array) != expected:
            raise ValueError(f"{name} must contain {expected} action rows")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains nonfinite values")
        return array.reshape(magnitude_count, schedule_count, -1)

    base = layout(baseline, "baseline")
    edit = layout(patched, "patched")
    truth = layout(target, "target")
    if base.shape != edit.shape or base.shape != truth.shape:
        raise ValueError("baseline, patched, and target shapes differ")
    schedules = np.asarray(tuple(int(value) for value in diagnostic_schedules))
    if len(schedules) == 0 or len(np.unique(schedules)) != len(schedules):
        raise ValueError("diagnostic schedules must be unique and nonempty")
    reversal = np.arange(schedule_count - 1, -1, -1, dtype=np.int64)
    if set(schedules.tolist()) != set(reversal[schedules].tolist()):
        raise ValueError("diagnostic schedules must be closed under reversal")

    rows = []
    for magnitude_index in range(magnitude_count):
        observed = edit[magnitude_index, schedules] - base[magnitude_index, schedules]
        self_target = (
            base[magnitude_index, reversal[schedules]]
            - base[magnitude_index, schedules]
        )
        grounded_target = (
            truth[magnitude_index, reversal[schedules]]
            - truth[magnitude_index, schedules]
        )
        self_metrics = bounded_cosine(
            observed, self_target, minimum_target_energy=minimum_target_energy
        )
        grounded_metrics = bounded_cosine(
            observed, grounded_target, minimum_target_energy=minimum_target_energy
        )
        rows.append(
            {
                "magnitude_index": magnitude_index,
                "self_eligible": bool(self_metrics["eligible"]),
                "self_source_energy": self_metrics["source_energy"],
                "self_target_energy": self_metrics["target_energy"],
                "self_cosine": self_metrics["cosine"],
                "grounded_eligible": bool(grounded_metrics["eligible"]),
                "grounded_source_energy": grounded_metrics["source_energy"],
                "grounded_target_energy": grounded_metrics["target_energy"],
                "grounded_cosine": grounded_metrics["cosine"],
            }
        )
    return rows


def paired_model_difference_rows(left_rows, right_rows, feature_names):
    """Create an exact right-minus-left state/family/magnitude panel."""

    features = tuple(str(value) for value in feature_names)
    key_names = ("record_id", "family_index", "magnitude_index")

    def index(rows, label):
        result = {}
        for row in rows:
            key = tuple(int(row[name]) for name in key_names)
            if key in result:
                raise ValueError(f"duplicate {label} key {key}")
            result[key] = row
        return result

    left = index(left_rows, "left")
    right = index(right_rows, "right")
    if set(left) != set(right):
        raise ValueError("paired model panels differ")
    rows = []
    for key in sorted(left):
        first, second = left[key], right[key]
        values = {
            f"difference_{name}": float(second[name]) - float(first[name])
            for name in features
        }
        outcome = float(second["outcome"]) - float(first["outcome"])
        if not np.isfinite(outcome) or not all(np.isfinite(value) for value in values.values()):
            raise ValueError(f"paired row {key} contains a nonfinite value")
        rows.append(
            {
                "record_id": key[0],
                "family_index": key[1],
                "magnitude_index": key[2],
                "left_outcome": float(first["outcome"]),
                "right_outcome": float(second["outcome"]),
                "outcome": outcome,
                **values,
            }
        )
    return rows


def state_placebo_advantage(primary_improvement, control_improvements, record_ids):
    """Aggregate primary-minus-median-placebo MSE gains by physical state."""

    primary = np.asarray(primary_improvement, dtype=np.float64)
    controls = np.asarray(control_improvements, dtype=np.float64)
    groups = np.asarray(record_ids, dtype=np.int64)
    if primary.ndim != 1 or controls.ndim != 2 or controls.shape[0] != len(primary):
        raise ValueError("primary and control improvements are not aligned")
    if len(groups) != len(primary) or controls.shape[1] < 1:
        raise ValueError("record IDs or controls are malformed")
    if not np.all(np.isfinite(primary)) or not np.all(np.isfinite(controls)):
        raise ValueError("improvements must be finite")
    row_advantage = primary - np.median(controls, axis=1)
    rows = []
    for record_id in np.unique(groups):
        selected = groups == record_id
        rows.append(
            {
                "record_id": int(record_id),
                "primary_minus_median_placebo_improvement": float(
                    np.mean(row_advantage[selected])
                ),
            }
        )
    return rows
