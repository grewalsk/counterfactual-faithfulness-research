#!/usr/bin/env python3
"""Verify Stage 11 bundle provenance and recompute headline aggregates."""

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "results" / "bundles" / "stage11_result_bundle"
DEFAULT_AUDIT = ROOT / "results" / "stage11_full_development_audit.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def task_equal_levels(rows, value_key):
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["environment"],
            row["method"],
            int(row["projection_seed"]),
            int(row["horizon"]),
            int(row["task_id"]),
        )
        grouped[key].append(float(row[value_key]))
    task_means = {
        key: statistics.fmean(values)
        for key, values in grouped.items()
    }
    collapsed = defaultdict(list)
    for (
        environment,
        method,
        _projection_seed,
        horizon,
        _task_id,
    ), value in task_means.items():
        collapsed[(environment, method, horizon)].append(value)
    return {
        key: statistics.fmean(values)
        for key, values in collapsed.items()
    }


def assert_close(observed, expected, tolerance=5e-6):
    if abs(float(observed) - float(expected)) > tolerance:
        raise AssertionError(
            f"value mismatch: observed={observed} expected={expected}"
        )


def verify_manifest(bundle):
    payload = json.loads(
        (bundle / "result_zip_manifest.json").read_text()
    )
    failures = []
    for row in payload["files"]:
        path = bundle / row["path"]
        if not path.exists():
            failures.append(f"missing:{row['path']}")
            continue
        if path.stat().st_size != int(row["size_bytes"]):
            failures.append(f"size:{row['path']}")
        if sha256_file(path) != row["sha256"]:
            failures.append(f"sha256:{row['path']}")
    if failures:
        raise AssertionError(
            f"Stage 11 manifest verification failed: {failures}"
        )
    return len(payload["files"])


def verify_source_hashes(bundle, audit):
    for relative, expected in audit["source_file_sha256"].items():
        observed = sha256_file(bundle / relative)
        if observed != expected:
            raise AssertionError(
                f"source hash mismatch for {relative}: "
                f"{observed} != {expected}"
            )


def verify_headline_levels(bundle, audit):
    geometry_rows = load_csv(
        bundle / "stage11_geometry_unit_metrics.csv"
    )
    planning_rows = load_csv(bundle / "stage11_unit_metrics.csv")
    geometry = task_equal_levels(
        geometry_rows, "whitened_geometry_rmse"
    )
    planning = task_equal_levels(
        planning_rows, "normalized_regret"
    )
    geometry_audit = audit["unseen_geometry"][
        "task_equal_mean_rmse"
    ]
    planning_audit = audit["fresh_readout_planning"][
        "task_equal_mean_normalized_regret"
    ]
    method = "fidelity_constrained_matched_geometry"
    for environment in ["PushT", "Wall"]:
        for horizon in [1, 3, 6]:
            label = f"h{horizon}"
            assert_close(
                geometry[(environment, "frozen", horizon)],
                geometry_audit[environment]["frozen"][label],
            )
            assert_close(
                geometry[(environment, method, horizon)],
                geometry_audit[environment]["matched"][label],
            )
            assert_close(
                planning[(environment, "frozen", horizon)],
                planning_audit[environment]["frozen"][label],
            )
            assert_close(
                planning[(environment, method, horizon)],
                planning_audit[environment]["matched"][label],
            )


def verify_decision_semantics(bundle):
    decision = json.loads(
        (bundle / "stage11_pilot_decision.json").read_text()
    )
    if decision["decision"] != "STOP_NATIVE_FIDELITY_FAILURE":
        raise AssertionError("unexpected raw Stage 11 decision")
    if not decision["matched_native_fidelity_pass"]:
        raise AssertionError(
            "audit assumption changed: matched latent fidelity did not pass"
        )
    if decision["native_planner_nondestruction_pass"]:
        raise AssertionError(
            "audit assumption changed: native planner non-harm passed"
        )
    if decision["fresh_readout_gate_pass"]:
        raise AssertionError(
            "audit assumption changed: fresh readout gate passed"
        )
    if decision["direct_geometry_projection_consensus"] != {
        "PushT": 5,
        "Wall": 5,
    }:
        raise AssertionError("unexpected geometry projection consensus")
    if decision["fresh_readout_projection_consensus"] != {
        "PushT": 1,
        "Wall": 2,
    }:
        raise AssertionError("unexpected readout projection consensus")


def verify_integrity_tables(bundle):
    probes = load_csv(bundle / "stage11_probe_selection.csv")
    if not probes or not all(
        row["ridge_optimum_interior"] == "True" for row in probes
    ):
        raise AssertionError("a fresh ridge optimum is not interior")
    certificates = load_csv(
        bundle / "stage11_certificate_metrics.csv"
    )
    if not certificates or not all(
        row["regret_bound_holds"] == "True"
        and (
            row["top1_certified"] != "True"
            or row["top1_correct"] == "True"
        )
        for row in certificates
    ):
        raise AssertionError("a Stage 11 certificate integrity check failed")
    restore = json.loads((bundle / "restore_test.json").read_text())
    if not all(
        restore[environment]["endpoint_bitwise_exact"]
        and restore[environment]["initial_render_bitwise_exact"]
        and restore[environment]["diagnostics_exact"]
        for environment in ["PushT", "Wall"]
    ):
        raise AssertionError("exact simulator restoration failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle", type=Path, default=DEFAULT_BUNDLE
    )
    parser.add_argument(
        "--audit", type=Path, default=DEFAULT_AUDIT
    )
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    audit = json.loads(args.audit.read_text())
    if (
        (bundle / "FAILURE_TRACE.txt").read_text().strip()
        != "SUCCESS: no captured pipeline failure"
    ):
        raise AssertionError("Stage 11 pipeline did not report success")
    config = json.loads((bundle / "config.json").read_text())
    if (
        config["run_mode"] != "full"
        or config["num_states"] != 96
        or config["adaptation_seeds"] != [11401, 11419, 11437]
        or len(config["evaluation_projection_seeds"]) != 5
        or config["bootstrap_reps"] != 2000
    ):
        raise AssertionError("Stage 11 full matrix is incomplete")

    manifest_count = verify_manifest(bundle)
    verify_source_hashes(bundle, audit)
    verify_headline_levels(bundle, audit)
    verify_decision_semantics(bundle)
    verify_integrity_tables(bundle)
    print(
        json.dumps(
            {
                "bundle": str(bundle),
                "manifest_files_verified": manifest_count,
                "run_signature": config["run_signature"],
                "geometry_consensus": {"PushT": 5, "Wall": 5},
                "fresh_readout_consensus": {
                    "PushT": 1,
                    "Wall": 2,
                },
                "status": "ok",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
