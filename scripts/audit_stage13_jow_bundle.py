#!/usr/bin/env python3
"""Independent integrity and numerical audit of a Stage 13 JOW bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


EXPECTED_STOP = "STOP_NO_COMPACT_OUTCOME_DICTIONARY"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text())


def close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return math.isclose(
        float(left), float(right), rel_tol=tolerance, abs_tol=tolerance
    )


def audit_bundle(bundle: Path) -> dict:
    manifest = load_json(bundle / "manifest.json")
    manifest_by_path = {row["path"]: row for row in manifest}
    actual_paths = {
        str(path.relative_to(bundle))
        for path in bundle.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    manifest_paths = set(manifest_by_path)
    hash_or_size_failures = []
    for relative, row in manifest_by_path.items():
        path = bundle / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(row["size_bytes"])
            or sha256_file(path) != row["sha256"]
        ):
            hash_or_size_failures.append(relative)

    config = load_json(bundle / "config.json")
    saved_signature = config["run_signature"]
    unsigned_config = {
        key: value for key, value in config.items() if key != "run_signature"
    }
    recomputed_signature = hashlib.sha256(
        json.dumps(unsigned_config, sort_keys=True).encode()
    ).hexdigest()

    gate = load_json(bundle / "dictionary_gate.json")
    decision = load_json(bundle / "stage13_jow_decision.json")
    gain = float(gate["pca_fraction"]) - float(gate["random_fraction"])
    ratio = float(gate["pca_fraction"]) / max(
        float(gate["random_fraction"]), 1e-12
    )
    gain_threshold = float(gate["thresholds"]["minimum_gain"])
    ratio_threshold = float(gate["thresholds"]["minimum_ratio"])
    recomputed_gate_pass = bool(
        gain >= gain_threshold and ratio >= ratio_threshold
    )

    with np.load(bundle / "outcome_dictionary.npz") as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    pca_axes = arrays["pca_axes"].astype(np.float64)
    random_axes = arrays["random_axes"].astype(np.float64)
    singular = arrays["singular_values"].astype(np.float64)
    pca_orthonormal_error = float(
        np.max(
            np.abs(
                pca_axes @ pca_axes.T - np.eye(pca_axes.shape[0])
            )
        )
    )
    random_orthonormal_error = float(
        np.max(
            np.abs(
                random_axes @ random_axes.T
                - np.eye(random_axes.shape[0])
            )
        )
    )
    singular_energy = singular**2
    construction_top_k_fraction = float(
        np.sum(singular_energy[: pca_axes.shape[0]])
        / np.sum(singular_energy)
    )
    effective_rank = float(
        np.sum(singular_energy) ** 2 / np.sum(singular_energy**2)
    )
    dictionary_arrays_finite = all(
        bool(np.isfinite(value).all()) for value in arrays.values()
    )

    expected_assets = config["EXPECTED_PRETRAINED_ASSET_SHA256"]
    verified_assets = {
        row["name"]: row["sha256"]
        for row in load_json(bundle / "pretrained_asset_verification.json")
    }
    restore = load_json(bundle / "restore_test.json")
    timings = load_json(bundle / "timings.json")
    failure_trace = (bundle / "FAILURE_TRACE.txt").read_text().strip()

    integrity_checks = {
        "manifest_hashes_and_sizes_match": not hash_or_size_failures,
        "manifest_has_no_missing_or_extra_entries": (
            actual_paths == manifest_paths
        ),
        "config_signature_matches": saved_signature == recomputed_signature,
        "failure_trace_is_none": failure_trace == "NONE",
        "restore_is_bitwise_exact": bool(
            restore["endpoint_bitwise_exact"]
            and restore["initial_render_bitwise_exact"]
            and restore["diagnostics_exact"]
            and float(restore["max_endpoint_abs_diff"]) == 0.0
        ),
        "pretrained_hashes_match_config": expected_assets == verified_assets,
        "dictionary_arrays_are_finite": dictionary_arrays_finite,
        "dictionary_axes_are_orthonormal": bool(
            pca_orthonormal_error <= 1e-5
            and random_orthonormal_error <= 1e-5
        ),
        "singular_values_are_descending": bool(
            np.all(np.diff(singular) <= 0)
        ),
        "scale_is_strictly_positive": bool(np.all(arrays["scale"] > 0)),
        "reported_gate_arithmetic_matches": bool(
            close(gain, gate["gain"])
            and close(ratio, gate["ratio"])
            and recomputed_gate_pass == bool(gate["passed"])
        ),
        "decision_matches_gate": bool(
            decision["dictionary_gate"] == gate
            and decision["decision"] == EXPECTED_STOP
            and decision["workspace_decision"] == EXPECTED_STOP
            and decision["treatment_decision"] == "NOT_TESTED"
            and not recomputed_gate_pass
        ),
    }

    downstream_patterns = [
        "*_lens.pt",
        "*_coordinate_gate.json",
        "*_causal_gate.json",
        "*_causal_swaps.csv",
    ]
    downstream_artifacts = sorted(
        {
            str(path.relative_to(bundle))
            for pattern in downstream_patterns
            for path in bundle.rglob(pattern)
        }
    )

    return {
        "audit_version": 1,
        "bundle": str(bundle.resolve()),
        "audit_passed": bool(all(integrity_checks.values())),
        "integrity_checks": integrity_checks,
        "manifest": {
            "declared_file_count": len(manifest),
            "hash_or_size_failures": sorted(hash_or_size_failures),
            "missing_entries": sorted(actual_paths - manifest_paths),
            "nonexistent_entries": sorted(manifest_paths - actual_paths),
        },
        "execution": {
            "run_mode": config["RUN_MODE"],
            "run_signature": saved_signature,
            "gpu": load_json(bundle / "versions.json")["gpu"],
            "gpu_total_gib": load_json(bundle / "versions.json")[
                "gpu_total_gib"
            ],
            "measured_stage_seconds": timings,
            "measured_total_seconds": float(sum(timings.values())),
            "failure_trace": failure_trace,
        },
        "dictionary": {
            "axis_count": int(pca_axes.shape[0]),
            "projected_dimension": int(pca_axes.shape[1]),
            "construction_top_k_variance_fraction": (
                construction_top_k_fraction
            ),
            "calibration_pca_fraction": float(gate["pca_fraction"]),
            "calibration_random_fraction": float(gate["random_fraction"]),
            "calibration_to_construction_fraction_ratio": float(
                gate["pca_fraction"] / construction_top_k_fraction
            ),
            "effective_rank_from_construction_spectrum": effective_rank,
            "numerical_rank_singular_value_gt_1e_5": int(
                np.sum(singular > 1e-5)
            ),
            "pca_orthonormal_max_error": pca_orthonormal_error,
            "random_orthonormal_max_error": random_orthonormal_error,
            "pca_random_overlap_frobenius_squared": float(
                np.linalg.norm(pca_axes @ random_axes.T, "fro") ** 2
            ),
        },
        "gate": {
            "reported": gate,
            "recomputed_gain": gain,
            "recomputed_ratio": ratio,
            "gain_margin_to_threshold": gain - gain_threshold,
            "ratio_margin_to_threshold": ratio - ratio_threshold,
            "fraction_of_gain_threshold_reached": gain / gain_threshold,
            "recomputed_passed": recomputed_gate_pass,
        },
        "evidence_boundary": {
            "scientific_stage_reached": "outcome_dictionary_gate",
            "coordinate_screen_performed": False,
            "jacobian_lens_constructed": False,
            "causal_interventions_performed": False,
            "arga_treatment_tested": False,
            "downstream_artifacts": downstream_artifacts,
            "raw_calibration_effects_in_bundle": False,
            "exact_notebook_commit_or_hash_in_config": False,
        },
        "judgment": {
            "protocol_verdict": EXPECTED_STOP,
            "scientific_label": (
                "BORDERLINE_NEGATIVE_FIXED_DICTIONARY_FEASIBILITY_SCREEN"
            ),
            "jow_hypothesis_tested": False,
            "recommended_action": (
                "Do not continue or retune this run post hoc. Archive it as "
                "a failure of the fixed eight-axis global outcome dictionary; "
                "only a separately specified outcome-representation study "
                "should revisit the broader workspace idea."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = audit_bundle(arguments.bundle)
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized)
    print(serialized, end="")
    if not result["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
