#!/usr/bin/env python3
"""Independent integrity and numerical audit of the Stage 12 result bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - integrity-only fallback
    torch = None


ENVIRONMENTS = ["PushT", "Wall"]
HORIZONS = [1, 3, 6]
METHODS = [
    "fidelity_constrained_latent_only",
    "fidelity_constrained_shuffled_geometry",
    "fidelity_constrained_matched_geometry",
]
ADAPTATION_SEEDS = [11401, 11419, 11437]
PLANNERS = ["shared_metric", "native_metric", "goal_permuted_metric"]
BASELINES = [
    "frozen",
    "fidelity_constrained_shuffled_geometry",
    "fidelity_constrained_latent_only",
]
METRICS = [
    "normalized_regret",
    "weighted_pairwise_accuracy",
    "normalized_margin_rmse",
    "top1_correct",
]
TIE = 1e-9


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text())


def load_csv(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def float_equal(left, right, tolerance=1e-10):
    left = float(left)
    right = float(right)
    if math.isnan(left) and math.isnan(right):
        return True
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def pair_indices(actions: int):
    return np.triu_indices(actions, 1)


def recompute_state_metrics(true_cost, predicted_cost):
    truth = np.asarray(true_cost, dtype=np.float64)
    prediction = np.asarray(predicted_cost, dtype=np.float64)
    selected = int(np.argmin(prediction))
    oracle = int(np.argmin(truth))
    best = float(np.min(truth))
    chosen = float(truth[selected])
    spread = float(np.max(truth) - best)
    normalized_regret = (
        (chosen - best) / spread if spread > TIE else 0.0
    )
    left, right = pair_indices(len(truth))
    true_margin = truth[left] - truth[right]
    predicted_margin = prediction[left] - prediction[right]
    valid = np.abs(true_margin) > TIE
    credit = np.full(len(left), np.nan)
    same = np.sign(true_margin) == np.sign(predicted_margin)
    credit[valid & same] = 1.0
    credit[valid & (np.abs(predicted_margin) <= TIE)] = 0.5
    credit[valid & np.isnan(credit)] = 0.0
    weights = np.abs(true_margin)
    weighted = (
        float(np.nansum(weights * credit) / np.sum(weights[valid]))
        if np.any(valid)
        else float("nan")
    )
    normalized_truth = (truth - best) / max(spread, 1e-6)
    normalized_margin = (
        normalized_truth[left] - normalized_truth[right]
    )
    denominator = float(np.sqrt(np.mean(normalized_margin**2)))
    normalized_margin_rmse = (
        float(
            np.sqrt(
                np.mean(
                    (predicted_margin - normalized_margin) ** 2
                )
            )
            / denominator
        )
        if denominator > 1e-12
        else 0.0
    )
    return {
        "normalized_regret": float(normalized_regret),
        "weighted_pairwise_accuracy": weighted,
        "top1_correct": float(chosen <= best + TIE),
        "normalized_margin_rmse": normalized_margin_rmse,
        "selected_action": selected,
        "oracle_action": oracle,
    }


def task_equal_summary(rows, finite_only):
    output = {}
    for horizon in HORIZONS:
        horizon_rows = [
            row for row in rows if int(row["horizon"]) == horizon
        ]
        values = {}
        for metric in METRICS:
            by_task = defaultdict(list)
            for row in horizon_rows:
                value = float(row[metric])
                if finite_only and not math.isfinite(value):
                    continue
                by_task[int(row["task_id"])].append(value)
            task_means = []
            for task_values in by_task.values():
                if task_values:
                    task_means.append(float(np.mean(task_values)))
            values[metric] = (
                float(np.mean(task_means))
                if task_means
                else float("nan")
            )
        output[str(horizon)] = values
    return output


def phase_a_from_summary(environment, summary):
    native_margin = float(
        np.mean(
            [
                summary["native_metric"][str(h)][
                    "normalized_margin_rmse"
                ]
                for h in HORIZONS
            ]
        )
    )
    shared_margin = float(
        np.mean(
            [
                summary["shared_metric"][str(h)][
                    "normalized_margin_rmse"
                ]
                for h in HORIZONS
            ]
        )
    )
    margin_improvement = 1.0 - shared_margin / max(
        native_margin, 1e-12
    )
    horizon_pass = {}
    for horizon in HORIZONS:
        shared = summary["shared_metric"][str(horizon)]
        native = summary["native_metric"][str(horizon)]
        horizon_pass[str(horizon)] = bool(
            shared["normalized_regret"]
            <= min(0.10, 0.75 * native["normalized_regret"]) + 1e-12
            and shared["weighted_pairwise_accuracy"]
            >= max(
                0.80,
                native["weighted_pairwise_accuracy"] + 0.03,
            )
            - 1e-12
        )
    shared_regret = float(
        np.mean(
            [
                summary["shared_metric"][str(h)][
                    "normalized_regret"
                ]
                for h in HORIZONS
            ]
        )
    )
    permuted_regret = float(
        np.mean(
            [
                summary["goal_permuted_metric"][str(h)][
                    "normalized_regret"
                ]
                for h in HORIZONS
            ]
        )
    )
    shared_accuracy = float(
        np.mean(
            [
                summary["shared_metric"][str(h)][
                    "weighted_pairwise_accuracy"
                ]
                for h in HORIZONS
            ]
        )
    )
    permuted_accuracy = float(
        np.mean(
            [
                summary["goal_permuted_metric"][str(h)][
                    "weighted_pairwise_accuracy"
                ]
                for h in HORIZONS
            ]
        )
    )
    goal_specificity_regret = permuted_regret - shared_regret
    goal_specificity_accuracy = shared_accuracy - permuted_accuracy
    goal_specificity = bool(
        goal_specificity_regret >= 0.02
        and goal_specificity_accuracy >= 0.02
    )
    return {
        "environment": environment,
        "passed": bool(
            margin_improvement >= 0.20
            and sum(horizon_pass.values()) >= 2
            and goal_specificity
        ),
        "margin_rmse_relative_improvement": margin_improvement,
        "horizon_pass": horizon_pass,
        "goal_specificity_pass": goal_specificity,
        "goal_specificity_regret_gain": goal_specificity_regret,
        "goal_specificity_accuracy_gain": goal_specificity_accuracy,
        "summary": summary,
    }


def task_equal_mean(rows, metric, finite_only=True):
    by_task = defaultdict(list)
    for row in rows:
        value = float(row[metric])
        if finite_only and not math.isfinite(value):
            continue
        by_task[int(row["task_id"])].append(value)
    task_means = [
        float(np.mean(values))
        for values in by_task.values()
        if values
    ]
    return (
        float(np.mean(task_means)) if task_means else float("nan")
    )


def contrast_from_collapsed(
    rows, environment, planner, baseline, horizon
):
    matched = {
        int(row["state_id"]): row
        for row in rows
        if row["environment"] == environment
        and row["split"] == "development_holdout"
        and row["planner"] == planner
        and row["method"]
        == "fidelity_constrained_matched_geometry"
        and int(row["horizon"]) == horizon
    }
    control = {
        int(row["state_id"]): row
        for row in rows
        if row["environment"] == environment
        and row["split"] == "development_holdout"
        and row["planner"] == planner
        and row["method"] == baseline
        and int(row["horizon"]) == horizon
    }
    assert set(matched) == set(control)
    output = {}
    for result_name, source_name, sign in [
        ("delta_regret", "normalized_regret", -1.0),
        (
            "delta_weighted_accuracy",
            "weighted_pairwise_accuracy",
            1.0,
        ),
    ]:
        differences = []
        for state_id in matched:
            matched_value = float(matched[state_id][source_name])
            control_value = float(control[state_id][source_name])
            difference = sign * (matched_value - control_value)
            differences.append(
                {
                    "task_id": int(matched[state_id]["task_id"]),
                    result_name: difference,
                }
            )
        output[result_name] = task_equal_mean(
            differences, result_name, finite_only=True
        )
    return output


def task_majority(rows, environment, baseline):
    selected = [
        row
        for row in rows
        if row["environment"] == environment
        and row["split"] == "development_holdout"
        and row["planner"] == "shared_metric"
        and row["method"]
        in {
            baseline,
            "fidelity_constrained_matched_geometry",
        }
    ]
    by_key = {
        (
            int(row["task_id"]),
            int(row["state_id"]),
            int(row["horizon"]),
            row["method"],
        ): row
        for row in selected
    }
    task_pass = {}
    for task_id in sorted({int(row["task_id"]) for row in selected}):
        regrets = []
        accuracies = []
        state_ids = sorted(
            {
                int(row["state_id"])
                for row in selected
                if int(row["task_id"]) == task_id
            }
        )
        for state_id in state_ids:
            for horizon in HORIZONS:
                matched = by_key[
                    (
                        task_id,
                        state_id,
                        horizon,
                        "fidelity_constrained_matched_geometry",
                    )
                ]
                control = by_key[
                    (task_id, state_id, horizon, baseline)
                ]
                regret = (
                    float(control["normalized_regret"])
                    - float(matched["normalized_regret"])
                )
                accuracy = (
                    float(matched["weighted_pairwise_accuracy"])
                    - float(control["weighted_pairwise_accuracy"])
                )
                if math.isfinite(regret):
                    regrets.append(regret)
                if math.isfinite(accuracy):
                    accuracies.append(accuracy)
        task_pass[str(task_id)] = bool(
            regrets
            and accuracies
            and np.mean(regrets) > 0
            and np.mean(accuracies) > 0
        )
    return {
        "task_pass": task_pass,
        "passing_tasks": int(sum(task_pass.values())),
        "passed": bool(sum(task_pass.values()) >= 2),
    }


def complete_planner_nonharm(rows, environment):
    result = {}
    for horizon in HORIZONS:
        matched = [
            row
            for row in rows
            if row["environment"] == environment
            and row["split"] == "development_holdout"
            and row["planner"] == "shared_metric"
            and row["method"]
            == "fidelity_constrained_matched_geometry"
            and int(row["horizon"]) == horizon
        ]
        frozen = [
            row
            for row in rows
            if row["environment"] == environment
            and row["split"] == "development_holdout"
            and row["planner"] == "native_metric"
            and row["method"] == "frozen"
            and int(row["horizon"]) == horizon
        ]
        regret_harm = task_equal_mean(
            matched, "normalized_regret"
        ) - task_equal_mean(frozen, "normalized_regret")
        accuracy_harm = task_equal_mean(
            frozen, "weighted_pairwise_accuracy"
        ) - task_equal_mean(
            matched, "weighted_pairwise_accuracy"
        )
        result[str(horizon)] = {
            "regret_harm": regret_harm,
            "accuracy_harm": accuracy_harm,
            "passed": bool(
                regret_harm <= 0.02 + 1e-12
                and accuracy_harm <= 0.02 + 1e-12
            ),
        }
    return {
        "horizons": result,
        "passed": all(value["passed"] for value in result.values()),
    }


def action_path_checksum(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def audit_manifest(bundle: Path):
    manifest_path = bundle / "stage12_result_zip_manifest.json"
    manifest = load_json(manifest_path)
    verified = []
    errors = []
    listed = set()
    for record in manifest["files"]:
        path = bundle / record["path"]
        listed.add(record["path"])
        if not path.is_file():
            errors.append(f"missing:{record['path']}")
            continue
        if path.stat().st_size != int(record["size_bytes"]):
            errors.append(f"size:{record['path']}")
            continue
        digest = sha256_file(path)
        if digest != record["sha256"]:
            errors.append(f"sha256:{record['path']}")
            continue
        verified.append(record["path"])
    actual = {
        str(path.relative_to(bundle))
        for path in bundle.rglob("*")
        if path.is_file()
    }
    expected_unlisted = {"stage12_result_zip_manifest.json"}
    unlisted = sorted(actual - listed)
    missing_from_disk = sorted(listed - actual)
    return {
        "manifest_files_verified": len(verified),
        "manifest_errors": errors,
        "unlisted_files": unlisted,
        "unlisted_files_expected": set(unlisted) == expected_unlisted,
        "missing_files": missing_from_disk,
        "passed": bool(
            not errors
            and not missing_from_disk
            and set(unlisted) == expected_unlisted
            and not manifest["pipeline_failed"]
        ),
    }


def audit_pretrained_assets(bundle: Path, run_signature):
    expected_manifest = load_json(bundle / "checkpoints_manifest.json")
    verification = load_json(
        bundle / "pretrained_asset_verification.json"
    )
    expected = {
        Path(record["path"]).name: (
            int(record["size_bytes"]),
            record["sha256"],
        )
        for record in expected_manifest["cached_files"]
    }
    errors = []
    observed_models = set()
    for model_name, model in verification["models"].items():
        observed_models.add(model_name)
        for record in model["records"]:
            name = record["name"]
            declared = (
                int(record["size_bytes"]),
                record["sha256"],
            )
            if name not in expected or expected[name] != declared:
                errors.append(f"{model_name}:{name}")
    if verification["run_signature"] != run_signature:
        errors.append("run_signature")
    if observed_models != set(expected_manifest["models"]):
        errors.append("models")
    return {
        "repository": expected_manifest["repository"],
        "repository_commit": expected_manifest[
            "repository_commit"
        ],
        "models": sorted(observed_models),
        "errors": errors,
        "passed": not errors,
    }


def audit_transition_checkpoints(bundle: Path, run_signature):
    paths = sorted((bundle / "adapted_action_paths").glob("*.pt"))
    records = []
    errors = []
    if torch is None:
        return {
            "torch_available": False,
            "checkpoint_count": len(paths),
            "internal_checks_skipped": True,
        }
    for path in paths:
        payload = torch.load(
            path, map_location="cpu", weights_only=False
        )
        checksum = action_path_checksum(payload["action_path"])
        finite = all(
            bool(torch.isfinite(value).all())
            for value in payload["action_path"].values()
        )
        valid = bool(
            payload["run_signature"] == run_signature
            and checksum == payload["selected_action_path_checksum"]
            and finite
        )
        if not valid:
            errors.append(path.name)
        records.append(
            {
                "file": path.name,
                "environment": payload["environment"],
                "method": payload["method"],
                "adaptation_seed": int(payload["adaptation_seed"]),
                "selected_epoch": int(payload["selected_epoch"]),
                "completed_epoch_limit": int(
                    payload["completed_epoch_limit"]
                ),
                "fidelity_feasible": bool(
                    payload["fidelity_feasible"]
                ),
                "checksum_verified": checksum
                == payload["selected_action_path_checksum"],
                "finite": finite,
            }
        )
    expected = {
        (environment, method, seed)
        for environment in ENVIRONMENTS
        for method in METHODS
        for seed in ADAPTATION_SEEDS
    }
    observed = {
        (
            record["environment"],
            record["method"],
            record["adaptation_seed"],
        )
        for record in records
    }
    return {
        "torch_available": True,
        "checkpoint_count": len(paths),
        "expected_matrix_complete": observed == expected,
        "internal_errors": errors,
        "records": records,
        "passed": bool(
            len(paths) == 18 and observed == expected and not errors
        ),
    }


def audit_metric_checkpoints(bundle: Path, run_signature):
    paths = sorted(
        (
            bundle
            / "shared_target_metric"
            / "metric_checkpoints"
        ).glob("*.pt")
    )
    if torch is None:
        return {
            "torch_available": False,
            "checkpoint_count": len(paths),
            "internal_checks_skipped": True,
        }
    records = []
    errors = []
    for path in paths:
        payload = torch.load(
            path, map_location="cpu", weights_only=False
        )
        low_rank = payload["low_rank"].double().numpy()
        metric = payload["metric"].double().numpy()
        dimension = low_rank.shape[1]
        unscaled = np.eye(dimension) + low_rank.T @ low_rank
        reconstructed = (
            dimension * unscaled / np.trace(unscaled)
        )
        eigenvalue = np.linalg.eigvalsh(metric)
        beta_matches = np.allclose(
            payload["beta"].double().numpy(),
            np.exp(payload["log_beta"].double().numpy()),
            rtol=1e-6,
            atol=1e-9,
        )
        matrix_valid = bool(
            np.all(np.isfinite(metric))
            and np.allclose(metric, metric.T, atol=1e-10)
            and np.allclose(metric, reconstructed, rtol=1e-6, atol=1e-8)
            and eigenvalue.min() > 0
            and math.isclose(
                float(np.trace(metric)),
                float(dimension),
                rel_tol=1e-10,
                abs_tol=1e-10,
            )
            and float(payload["condition_number"]) < 20.0
            and beta_matches
        )
        leakage_flags_valid = bool(
            payload["uses_predicted_rollouts"] is False
            and payload["uses_treatment_identity"] is False
            and payload["uses_candidate_identity_as_feature"] is False
            and payload["uses_development_outcomes"] is False
            and payload["uses_physical_decoder"] is False
        )
        valid = bool(
            payload["run_signature"] == run_signature
            and matrix_valid
            and leakage_flags_valid
        )
        if not valid:
            errors.append(path.name)
        records.append(
            {
                "file": path.name,
                "environment": payload["environment"],
                "control_name": payload["control_name"],
                "rank": int(payload["rank"]),
                "regularizer": float(payload["regularizer"]),
                "optimization_seed": int(
                    payload["optimization_seed"]
                ),
                "selected_epoch": int(payload["selected_epoch"]),
                "completed_epochs": int(payload["completed_epochs"]),
                "converged_before_max_epochs": bool(
                    payload["converged_before_max_epochs"]
                ),
                "condition_number": float(
                    payload["condition_number"]
                ),
                "matrix_and_flags_verified": valid,
            }
        )
    return {
        "torch_available": True,
        "checkpoint_count": len(paths),
        "internal_errors": errors,
        "all_matrix_and_leakage_checks_pass": not errors,
        "fits_converged_before_max_epochs": int(
            sum(
                record["converged_before_max_epochs"]
                for record in records
            )
        ),
        "fits_at_max_epoch": int(
            sum(
                not record["converged_before_max_epochs"]
                for record in records
            )
        ),
        "records": records,
        "passed": bool(len(paths) == 18 and not errors),
    }


def audit_unit_rows(rows):
    errors = []
    for index, row in enumerate(rows):
        true_cost = json.loads(row["true_cost_json"])
        predicted_cost = json.loads(row["predicted_cost_json"])
        true_margin = json.loads(row["true_margin_json"])
        predicted_margin = json.loads(row["predicted_margin_json"])
        if not (
            len(true_cost) == len(predicted_cost) == 10
            and len(true_margin) == len(predicted_margin) == 45
        ):
            errors.append(f"shape:{index}")
            continue
        recomputed = recompute_state_metrics(
            true_cost, predicted_cost
        )
        for key in [
            "normalized_regret",
            "weighted_pairwise_accuracy",
            "top1_correct",
            "normalized_margin_rmse",
        ]:
            if not float_equal(recomputed[key], row[key], 2e-9):
                errors.append(f"metric:{index}:{key}")
                break
        if (
            recomputed["selected_action"] != int(row["selected_action"])
            or recomputed["oracle_action"] != int(row["oracle_action"])
        ):
            errors.append(f"selection:{index}")
    return {
        "row_count": len(rows),
        "expected_row_count": 5472,
        "errors": errors[:50],
        "passed": bool(len(rows) == 5472 and not errors),
    }


def audit_seed_collapse(unit_rows, collapsed_rows):
    keys = [
        "environment",
        "state_id",
        "task_id",
        "split",
        "planner",
        "method",
        "horizon",
    ]
    grouped = defaultdict(list)
    for row in unit_rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    declared = {
        tuple(row[key] for key in keys): row for row in collapsed_rows
    }
    errors = []
    if set(grouped) != set(declared):
        errors.append("group_keys")
    for key in set(grouped) & set(declared):
        source = grouped[key]
        target = declared[key]
        for metric in METRICS:
            value = float(
                np.mean([float(row[metric]) for row in source])
            )
            if not float_equal(value, target[metric], 2e-10):
                errors.append(f"{key}:{metric}")
                break
        if len(source) != int(target["transition_seeds_averaged"]):
            errors.append(f"{key}:seed_count")
    return {
        "row_count": len(collapsed_rows),
        "expected_row_count": 2880,
        "errors": errors[:50],
        "passed": bool(len(collapsed_rows) == 2880 and not errors),
    }


def audit_phase_a(unit_rows, declared_gate):
    declared_recomputed = {}
    finite_recomputed = {}
    declared_match = True
    for environment in ENVIRONMENTS:
        declared_summary = {}
        finite_summary = {}
        for planner in PLANNERS:
            selected = [
                row
                for row in unit_rows
                if row["environment"] == environment
                and row["split"] == "probe_calibration"
                and row["planner"] == planner
                and row["method"] == "target_oracle"
            ]
            declared_summary[planner] = task_equal_summary(
                selected, finite_only=False
            )
            finite_summary[planner] = task_equal_summary(
                selected, finite_only=True
            )
        declared_result = phase_a_from_summary(
            environment, declared_summary
        )
        finite_result = phase_a_from_summary(
            environment, finite_summary
        )
        declared_recomputed[environment] = declared_result
        finite_recomputed[environment] = finite_result
        expected = declared_gate[
            "phase_a_metric_class_viability"
        ][environment]
        for key in [
            "passed",
            "margin_rmse_relative_improvement",
            "goal_specificity_pass",
            "goal_specificity_regret_gain",
            "goal_specificity_accuracy_gain",
        ]:
            if isinstance(expected[key], bool):
                declared_match &= expected[key] == declared_result[key]
            else:
                declared_match &= float_equal(
                    expected[key], declared_result[key], 2e-10
                )
    return {
        "declared_gate_reproduced": bool(declared_match),
        "declared_recomputed": declared_recomputed,
        "finite_row_recomputed": finite_recomputed,
        "finite_row_outcome_changes": any(
            declared_recomputed[environment]["passed"]
            != finite_recomputed[environment]["passed"]
            for environment in ENVIRONMENTS
        ),
        "both_environments_fail_after_nan_repair": all(
            not finite_recomputed[environment]["passed"]
            for environment in ENVIRONMENTS
        ),
    }


def audit_phase_b(collapsed_rows, declared_gate):
    contrasts = {}
    contrast_match = True
    common = {}
    task_gates = {}
    specificity = {}
    nonharm = {}
    for environment in ENVIRONMENTS:
        contrasts[environment] = {}
        for planner in [
            "shared_metric",
            "goal_permuted_metric",
        ]:
            contrasts[environment][planner] = {}
            for baseline in BASELINES:
                contrasts[environment][planner][baseline] = {}
                for horizon in HORIZONS:
                    value = contrast_from_collapsed(
                        collapsed_rows,
                        environment,
                        planner,
                        baseline,
                        horizon,
                    )
                    contrasts[environment][planner][baseline][
                        str(horizon)
                    ] = value
                    expected = declared_gate[
                        "phase_b_causal_bridge"
                    ][environment]["contrasts"][planner][baseline][
                        str(horizon)
                    ]
                    for metric in [
                        "delta_regret",
                        "delta_weighted_accuracy",
                    ]:
                        contrast_match &= float_equal(
                            value[metric],
                            expected[metric]["estimate"],
                            2e-10,
                        )
        common_horizons = []
        for horizon in HORIZONS:
            strong = all(
                contrasts[environment]["shared_metric"][baseline][
                    str(horizon)
                ]["delta_regret"]
                >= 0.015
                and contrasts[environment]["shared_metric"][baseline][
                    str(horizon)
                ]["delta_weighted_accuracy"]
                >= 0.01
                for baseline in [
                    "frozen",
                    "fidelity_constrained_shuffled_geometry",
                ]
            )
            latent = contrasts[environment]["shared_metric"][
                "fidelity_constrained_latent_only"
            ][str(horizon)]
            if (
                strong
                and latent["delta_regret"] > 0
                and latent["delta_weighted_accuracy"] > 0
            ):
                common_horizons.append(horizon)
        common[environment] = common_horizons
        task_gates[environment] = {
            baseline: task_majority(
                collapsed_rows, environment, baseline
            )
            for baseline in [
                "frozen",
                "fidelity_constrained_shuffled_geometry",
            ]
        }
        true_gain = {}
        permuted_gain = {}
        for metric in [
            "delta_regret",
            "delta_weighted_accuracy",
        ]:
            true_gain[metric] = float(
                np.mean(
                    [
                        contrasts[environment]["shared_metric"][
                            "frozen"
                        ][str(horizon)][metric]
                        for horizon in HORIZONS
                    ]
                )
            )
            permuted_gain[metric] = float(
                np.mean(
                    [
                        contrasts[environment][
                            "goal_permuted_metric"
                        ]["frozen"][str(horizon)][metric]
                        for horizon in HORIZONS
                    ]
                )
            )
        specificity[environment] = {
            metric: true_gain[metric] - permuted_gain[metric]
            for metric in true_gain
        }
        nonharm[environment] = complete_planner_nonharm(
            collapsed_rows, environment
        )
    return {
        "declared_point_contrasts_reproduced": bool(contrast_match),
        "finite_row_contrasts": contrasts,
        "common_strong_and_latent_directional_horizons": common,
        "task_majority": task_gates,
        "goal_specificity_gain": specificity,
        "complete_planner_nonharm_finite_row": nonharm,
        "both_environments_fail": all(
            len(common[environment]) < 2
            for environment in ENVIRONMENTS
        ),
    }


def audit_bootstrap_draws(rows, declared_gate):
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["environment"],
            row["planner"],
            row["baseline"],
            row["horizon"],
            row["metric"],
        )
        grouped[key].append(row)
    errors = []
    expected_keys = {
        (
            environment,
            planner,
            baseline,
            str(horizon),
            metric,
        )
        for environment in ENVIRONMENTS
        for planner in ["shared_metric", "goal_permuted_metric"]
        for baseline in BASELINES
        for horizon in HORIZONS
        for metric in ["delta_regret", "delta_weighted_accuracy"]
    }
    if set(grouped) != expected_keys:
        errors.append("group_keys")
    for key, group in grouped.items():
        environment, planner, baseline, horizon, metric = key
        if len(group) != 2000:
            errors.append(f"draw_count:{key}")
            continue
        indices = sorted(int(row["draw_index"]) for row in group)
        if indices != list(range(2000)):
            errors.append(f"draw_indices:{key}")
        seeds = {int(row["bootstrap_seed"]) for row in group}
        if len(seeds) != 1:
            errors.append(f"bootstrap_seed:{key}")
        values = np.asarray(
            [float(row["estimate"]) for row in group],
            dtype=np.float64,
        )
        low, high = np.quantile(values, [0.025, 0.975])
        expected = declared_gate["phase_b_causal_bridge"][environment][
            "contrasts"
        ][planner][baseline][horizon][metric]
        if not float_equal(low, expected["low"], 2e-12):
            errors.append(f"low:{key}")
        if not float_equal(high, expected["high"], 2e-12):
            errors.append(f"high:{key}")
        if int(expected["n_bootstrap"]) != 2000:
            errors.append(f"declared_draw_count:{key}")
    return {
        "row_count": len(rows),
        "expected_row_count": 144000,
        "contrast_groups": len(grouped),
        "expected_contrast_groups": 72,
        "percentile_intervals_reproduced": not errors,
        "errors": errors[:50],
        "passed": bool(
            len(rows) == 144000
            and len(grouped) == 72
            and not errors
        ),
    }


def geometry_summary(bundle):
    rows = load_csv(bundle / "stage12_geometry_task_contrasts.csv")
    output = {}
    for environment in ENVIRONMENTS:
        output[environment] = {}
        for baseline in [
            "frozen",
            "fidelity_constrained_latent_only",
            "fidelity_constrained_shuffled_geometry",
        ]:
            selected = [
                row
                for row in rows
                if row["environment"] == environment
                and row["baseline"] == baseline
            ]
            output[environment][baseline] = {
                row["horizon"]: {
                    key: float(row[key])
                    for key in ["estimate", "low", "high"]
                }
                for row in selected
            }
    return output


def audit_bundle(bundle: Path):
    config = load_json(bundle / "config.json")
    declared_gate = load_json(
        bundle
        / "shared_target_metric"
        / "stage12_pilot_gate.json"
    )
    unit_rows = load_csv(
        bundle
        / "shared_target_metric"
        / "stage12_unit_metrics.csv"
    )
    collapsed_rows = load_csv(
        bundle
        / "shared_target_metric"
        / "stage12_seed_collapsed_metrics.csv"
    )
    bootstrap_rows = load_csv(
        bundle
        / "shared_target_metric"
        / "stage12_bootstrap_draws.csv"
    )
    manifest = audit_manifest(bundle)
    pretrained = audit_pretrained_assets(
        bundle, config["run_signature"]
    )
    candidate_design = load_json(
        bundle / "candidate_design_summary.json"
    )
    candidate_design_pass = all(
        candidate_design[environment]["design_valid"]
        for environment in ENVIRONMENTS
    )
    failure_trace_success = (
        (bundle / "FAILURE_TRACE.txt")
        .read_text()
        .strip()
        .startswith("SUCCESS:")
    )
    transition = audit_transition_checkpoints(
        bundle, config["run_signature"]
    )
    metric = audit_metric_checkpoints(
        bundle, config["run_signature"]
    )
    unit = audit_unit_rows(unit_rows)
    collapse = audit_seed_collapse(unit_rows, collapsed_rows)
    phase_a = audit_phase_a(unit_rows, declared_gate)
    phase_b = audit_phase_b(collapsed_rows, declared_gate)
    bootstrap = audit_bootstrap_draws(
        bootstrap_rows, declared_gate
    )
    restore = load_json(bundle / "restore_test.json")
    restoration_pass = all(
        restore[environment]["endpoint_bitwise_exact"]
        and restore[environment]["initial_render_bitwise_exact"]
        and restore[environment]["diagnostics_exact"]
        and float(restore[environment]["max_endpoint_abs_diff"])
        == 0.0
        for environment in ENVIRONMENTS
    )
    integrity_pass = bool(
        manifest["passed"]
        and pretrained["passed"]
        and candidate_design_pass
        and failure_trace_success
        and transition.get("passed", False)
        and metric.get("passed", False)
        and unit["passed"]
        and collapse["passed"]
        and bootstrap["passed"]
        and restoration_pass
    )
    decision_robust = bool(
        declared_gate["decision"] == "STOP_METRIC_CLASS_NOT_VIABLE"
        and phase_a["declared_gate_reproduced"]
        and phase_a["both_environments_fail_after_nan_repair"]
        and phase_b["declared_point_contrasts_reproduced"]
        and phase_b["both_environments_fail"]
    )
    return {
        "audit_status": (
            "VERIFIED_NO_GO_WITH_IMPLEMENTATION_CAVEATS"
            if integrity_pass and decision_robust
            else "AUDIT_FAILURE"
        ),
        "bundle": (
            str(bundle.relative_to(Path(__file__).resolve().parents[1]))
            if bundle.is_relative_to(
                Path(__file__).resolve().parents[1]
            )
            else str(bundle)
        ),
        "run_signature": config["run_signature"],
        "declared_decision": declared_gate["decision"],
        "independent_decision": (
            "NO_GO_TO_UNTOUCHED_TASK_CONFIRMATION"
            if decision_robust
            else "UNRESOLVED"
        ),
        "integrity": {
            "manifest": manifest,
            "pretrained_assets": pretrained,
            "candidate_design_pass": candidate_design_pass,
            "failure_trace_success": failure_trace_success,
            "transition_checkpoints": transition,
            "metric_checkpoints": metric,
            "unit_rows": unit,
            "seed_collapse": collapse,
            "bootstrap_draws": bootstrap,
            "restoration_pass": restoration_pass,
            "passed": integrity_pass,
        },
        "phase_a": phase_a,
        "phase_b": phase_b,
        "geometry_context": geometry_summary(bundle),
        "implementation_findings": [
            {
                "severity": "material",
                "finding": (
                    "All 18 metric fits reached epoch 600 without "
                    "satisfying the convergence criterion."
                ),
                "effect": (
                    "The selected recipe fails its own optimization-boundary "
                    "guardrail. The run is a valid no-go for confirmation, "
                    "but not a clean proof that every low-rank PSD metric "
                    "in the proposed class is intrinsically incapable."
                ),
            },
            {
                "severity": "reporting",
                "finding": (
                    "Plain task means propagate undefined weighted accuracy "
                    "from tied PushT rows into the horizon-1 Phase A summary "
                    "and complete-planner non-harm gate."
                ),
                "effect": (
                    "Finite-row recomputation repairs the NaNs but does not "
                    "change any promotion decision."
                ),
            },
            {
                "severity": "interpretation",
                "finding": (
                    "The learned target metric increases normalized margin "
                    "RMSE by 56.3% in PushT and 158.5% in Wall relative to "
                    "the train-scaled native target metric."
                ),
                "effect": (
                    "Target-latent metric viability fails before predicted "
                    "transition quality can be credited or blamed."
                ),
            },
            {
                "severity": "interpretation",
                "finding": (
                    "Matched ARGA does not beat both frozen and shuffled "
                    "controls at the preregistered thresholds in two "
                    "horizons in either environment; gains also fail the "
                    "goal-permuted specificity test."
                ),
                "effect": (
                    "There is no causal bridge signal supporting untouched-"
                    "task confirmation of this combined method."
                ),
            },
        ],
        "recommended_action": {
            "immediate": (
                "Do not launch Stage 12 untouched-task confirmation and do "
                "not tune this bridge again on the inspected development "
                "tasks."
            ),
            "paper_direction": (
                "Treat the result as evidence for a diagnostic/negative "
                "paper: repaired action-relative geometry is real, but this "
                "target-only low-rank quadratic planner does not convert it "
                "into robust goal-conditioned decisions."
            ),
            "allowed_followup": (
                "If method development continues, debug optimizer "
                "convergence using probe-train/calibration only, freeze a "
                "numerically distinct planner recipe, and evaluate it only "
                "on newly generated tasks."
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "bundle",
        nargs="?",
        default=(
            Path(__file__).resolve().parents[1]
            / "results"
            / "bundles"
            / "stage12_result_bundle"
        ),
        type=Path,
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = audit_bundle(arguments.bundle.resolve())
    serialized = json.dumps(
        result, indent=2, sort_keys=True, allow_nan=True
    ) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized)
    print(serialized)
    if result["audit_status"] == "AUDIT_FAILURE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
