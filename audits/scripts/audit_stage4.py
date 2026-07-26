#!/usr/bin/env python3
"""Independent integrity and decision audit for a Stage 4 result bundle."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


BUNDLE = Path(
    os.environ.get("STAGE4_BUNDLE", "stage4_result_bundle")
).expanduser()
OUTPUT = os.environ.get("STAGE4_AUDIT_OUTPUT", "")
SEED = 71
BOOTSTRAP_REPS = 2000


def load_json(name):
    return json.loads((BUNDLE / name).read_text())


def bootstrap_mean(values, groups, repetitions, seed):
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    finite = np.isfinite(values)
    values = values[finite]
    groups = groups[finite]
    unique = np.unique(groups)
    grouped = [values[groups == group] for group in unique]
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sampled = rng.integers(0, len(unique), size=len(unique))
        draws[index] = np.mean(
            np.concatenate([grouped[item] for item in sampled])
        )
    return {
        "estimate": float(np.mean(values)),
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
        "n_clusters": int(len(unique)),
        "n_bootstrap": int(repetitions),
    }


def close_interval(left, right, tolerance=1e-12):
    return all(
        abs(float(left[key]) - float(right[key])) <= tolerance
        for key in ["estimate", "low", "high"]
    ) and int(left["n_clusters"]) == int(right["n_clusters"])


failure = (BUNDLE / "FAILURE_TRACE.txt").read_text().strip()
unit = pd.read_csv(BUNDLE / "intervention_unit_metrics.csv")
summary = pd.read_csv(BUNDLE / "intervention_summary.csv")
full = pd.read_csv(BUNDLE / "full_severity_pairs.csv")
slopes = pd.read_csv(BUNDLE / "dose_response_slopes.csv")
subgroups = pd.read_csv(BUNDLE / "subgroup_specificity.csv")
integrity = load_json("matched_error_integrity.json")
decision = load_json("stage4_decision.json")
manifest = load_json("result_zip_manifest.json")

unit_key = [
    "environment",
    "state_id",
    "task_id",
    "model",
    "probe_seed",
    "horizon",
    "intervention_seed",
    "intervention",
    "severity",
]
matched_key = [
    "environment",
    "state_id",
    "task_id",
    "model",
    "probe_seed",
    "horizon",
    "intervention_seed",
    "severity",
]
matched = unit.pivot(
    index=matched_key,
    columns="intervention",
    values="pose_perturbation_rms",
)
maximum_match_difference = float(
    np.max(
        np.abs(
            matched["action_structure"].to_numpy()
            - matched["common_mode"].to_numpy()
        )
    )
)

finite_full = full.loc[
    np.isfinite(
        full[
            ["specific_regret_damage", "specific_ranking_damage"]
        ].to_numpy()
    ).all(axis=1)
].copy()

reproduced = {}
all_intervals_match = True
for environment_index, environment in enumerate(["PushT", "Wall"]):
    selected = finite_full.loc[
        finite_full["environment"].eq(environment)
    ]
    groups = selected["state_id"].to_numpy()
    reproduced[environment] = {
        "specific_regret_damage": bootstrap_mean(
            selected["specific_regret_damage"].to_numpy(),
            groups,
            BOOTSTRAP_REPS,
            SEED + 7000 + environment_index,
        ),
        "specific_ranking_damage": bootstrap_mean(
            selected["specific_ranking_damage"].to_numpy(),
            groups,
            BOOTSTRAP_REPS,
            SEED + 7100 + environment_index,
        ),
    }
    for metric, result in reproduced[environment].items():
        all_intervals_match &= close_interval(
            result,
            decision["environment_comparisons"][environment][metric],
        )

reproduced_slopes = {}
all_slope_intervals_match = True
for environment_index, environment in enumerate(["PushT", "Wall"]):
    selected = slopes.loc[slopes["environment"].eq(environment)]
    groups = selected["state_id"].to_numpy()
    reproduced_slopes[environment] = {
        "specific_regret_slope": bootstrap_mean(
            selected["specific_regret_slope"].to_numpy(),
            groups,
            BOOTSTRAP_REPS,
            SEED + 7400 + environment_index,
        ),
        "specific_ranking_slope": bootstrap_mean(
            selected["specific_ranking_slope"].to_numpy(),
            groups,
            BOOTSTRAP_REPS,
            SEED + 7500 + environment_index,
        ),
    }
    for metric, result in reproduced_slopes[environment].items():
        all_slope_intervals_match &= close_interval(
            result,
            decision["dose_response_slope_comparisons"][environment][
                metric
            ],
        )

positive_subgroups = {
    metric: int(np.sum(subgroups[f"{metric}_estimate"] > 0))
    for metric in [
        "specific_regret_damage",
        "specific_ranking_damage",
        "specific_top1_damage",
        "task_margin_specificity",
    ]
}
expected_files_present = all(
    (BUNDLE / name).is_file() for name in manifest["included"]
)

checks = {
    "failure_trace_none": failure == "NONE",
    "unit_rows_72000": len(unit) == 72_000,
    "unit_key_unique": not unit.duplicated(unit_key).any(),
    "environments_exact": set(unit["environment"]) == {"PushT", "Wall"},
    "state_clusters_40_each": (
        unit.groupby("environment")["state_id"].nunique().to_dict()
        == {"PushT": 40, "Wall": 40}
    ),
    "two_interventions": set(unit["intervention"])
    == {"action_structure", "common_mode"},
    "five_severities": sorted(unit["severity"].unique().tolist())
    == [0.0, 0.25, 0.5, 0.75, 1.0],
    "five_intervention_seeds": unit["intervention_seed"].nunique() == 5,
    "matched_pose_rms": maximum_match_difference <= 1e-10,
    "integrity_pass": bool(integrity["pass"]),
    "all_primary_intervals_reproduced": bool(all_intervals_match),
    "all_slope_intervals_reproduced": bool(all_slope_intervals_match),
    "both_environment_gates_pass": all(
        decision["environment_gate_pass"].values()
    ),
    "decision_cross_environment": (
        decision["status"]
        == "CROSS_ENV_ACTION_STRUCTURE_CAUSAL_SIGNAL"
    ),
    "primary_direction_positive_12_of_12": (
        positive_subgroups["specific_regret_damage"] == 12
        and positive_subgroups["specific_ranking_damage"] == 12
    ),
    "result_manifest_complete": expected_files_present,
    "summary_rows_20": len(summary) == 20,
}

payload = {
    "bundle": str(BUNDLE),
    "checks": checks,
    "all_checks_pass": all(checks.values()),
    "maximum_matched_pose_rms_difference": maximum_match_difference,
    "reproduced_primary_intervals": reproduced,
    "reproduced_slope_intervals": reproduced_slopes,
    "positive_subgroup_cells": positive_subgroups,
}
text = json.dumps(payload, indent=2) + "\n"
if OUTPUT:
    output_path = Path(OUTPUT)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text)
print(text)
if not payload["all_checks_pass"]:
    raise SystemExit(1)
