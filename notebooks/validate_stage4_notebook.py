#!/usr/bin/env python3
"""Static, synthetic, and optional source-bundle validation for Stage 4."""

from __future__ import annotations

import ast
import json
import os
import tempfile
from pathlib import Path

import numpy as np


NOTEBOOK = Path(__file__).with_name(
    "04_matched_action_structure_intervention.ipynb"
)


def definitions_from_notebook(notebook, names):
    definitions = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                definitions[node.name] = node
    return definitions


def synthetic_checks(notebook):
    names = {
        "pair_indices",
        "decoded_task_cost",
        "ranking_metrics",
        "stable_seed",
        "fixed_derangement",
        "matched_pose_interventions",
        "bootstrap_mean",
    }
    definitions = definitions_from_notebook(notebook, names)
    assert set(definitions) == names
    namespace = {
        "np": np,
        "hashlib": __import__("hashlib"),
        "RANKING_TIE": 1e-9,
    }
    for name in [
        "pair_indices",
        "decoded_task_cost",
        "ranking_metrics",
        "stable_seed",
        "fixed_derangement",
        "matched_pose_interventions",
        "bootstrap_mean",
    ]:
        module = ast.Module(body=[definitions[name]], type_ignores=[])
        exec(compile(module, NOTEBOOK.name, "exec"), namespace)

    poses = np.array(
        [[0.05 * index, np.sin(index / 3)] for index in range(10)],
        dtype=float,
    )
    for severity in [0.0, 0.25, 0.5, 0.75, 1.0]:
        payload = namespace["matched_pose_interventions"](
            poses, severity, 19071
        )
        assert payload["fixed_points"] == 0
        assert np.array_equal(
            np.sort(payload["permutation"]), np.arange(10)
        )
        assert abs(payload["action_rms"] - payload["common_rms"]) < 1e-12
        if severity == 0:
            assert np.allclose(payload["action_structure"], poses)
            assert np.allclose(payload["common_mode"], poses)
        if severity == 1:
            intact_centroid = poses.mean(axis=0)
            corrupted_centroid = payload["action_structure"].mean(axis=0)
            assert np.allclose(intact_centroid, corrupted_centroid)

    wall_task = {"goal": [52.0, 48.0]}
    predicted_cost = namespace["decoded_task_cost"](
        "Wall", poses, wall_task
    )
    truth_cost = np.linspace(0.05, 0.5, 10)
    ranking = namespace["ranking_metrics"](truth_cost, predicted_cost)
    assert 0 <= ranking["normalized_regret"] <= 1
    assert 0 <= ranking["weighted_pairwise_accuracy"] <= 1

    estimate = namespace["bootstrap_mean"](
        np.array([1.0, 1.2, 0.8, 1.1]),
        np.array([0, 0, 1, 1]),
        100,
        71,
    )
    assert estimate["n_clusters"] == 2
    assert estimate["low"] > 0


def optional_bundle_smoke(notebook):
    bundle = os.environ.get("STAGE4_SOURCE_BUNDLE", "")
    if not bundle:
        return
    bundle_path = Path(bundle)
    assert bundle_path.is_file(), bundle_path
    with tempfile.TemporaryDirectory() as temporary:
        namespace = {}
        code_cells = [
            cell
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        exec(
            compile(
                "".join(code_cells[0]["source"]),
                f"{NOTEBOOK.name}:config",
                "exec",
            ),
            namespace,
        )
        namespace.update(
            {
                "RUN_MODE": "smoke",
                "LOCAL_SOURCE_BUNDLE": str(bundle_path),
                "OUTPUT_DIR": str(Path(temporary) / "stage4"),
                "BOOTSTRAP_REPS": 2000,
                "DOWNLOAD_RESULTS": False,
            }
        )
        for index, cell in enumerate(code_cells[1:], start=1):
            exec(
                compile(
                    "".join(cell["source"]),
                    f"{NOTEBOOK.name}:cell_{index}",
                    "exec",
                ),
                namespace,
            )
            if index == 1:
                namespace["RESULT_ZIP"] = (
                    Path(temporary) / "stage4_result_bundle.zip"
                )
        output = Path(namespace["OUTPUT_DIR"])
        assert (output / "FAILURE_TRACE.txt").read_text().strip() == "NONE"
        decision = json.loads((output / "stage4_decision.json").read_text())
        assert decision["status"] in {
            "CROSS_ENV_ACTION_STRUCTURE_CAUSAL_SIGNAL",
            "MIXED_ACTION_STRUCTURE_SIGNAL",
            "NO_ACTION_STRUCTURE_SPECIFICITY",
        }
        integrity = json.loads(
            (output / "matched_error_integrity.json").read_text()
        )
        assert integrity["pass"]
        assert Path(namespace["RESULT_ZIP"]).is_file()


def main() -> int:
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["colab"]["name"] == NOTEBOOK.name
    assert len(notebook["cells"]) == 11
    assert notebook["cells"][0]["cell_type"] == "code"
    config = "".join(notebook["cells"][0]["source"])
    for fragment in [
        'RUN_MODE = "full"',
        "SEVERITIES = [0.00, 0.25, 0.50, 0.75, 1.00]",
        "INTERVENTION_SEEDS = [1103, 2203, 3301, 4409, 5501]",
        "BOOTSTRAP_REPS = 2000",
        "EXPECTED_ACTIONS = 10",
        "MATCH_TOLERANCE = 1e-10",
    ]:
        assert fragment in config

    write_json_calls = 0
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse(
            "".join(cell["source"]),
            filename=f"{NOTEBOOK.name}:cell_{index}",
        )
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "write_json"
            ):
                write_json_calls += 1
                assert len(node.args) == 2
                assert not node.keywords
    assert write_json_calls >= 5

    source = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"]
    )
    for fragment in [
        "Stage 4: matched-error action-structure intervention",
        "final_test",
        "linear_pose",
        "fixed_derangement",
        "matched_pose_interventions",
        "action_structure",
        "common_mode",
        "maximum_matched_pose_rms_discrepancy",
        "specific_regret_damage",
        "specific_ranking_damage",
        "specific_regret_slope",
        "specific_ranking_slope",
        "dose_response_slopes.csv",
        "subgroup_specificity.csv",
        "state_id",
        "bootstrap_mean",
        "CROSS_ENV_ACTION_STRUCTURE_CAUSAL_SIGNAL",
        "matched_regret_dose_response.png",
        "matched_ranking_dose_response.png",
        "full_severity_specificity.png",
        "stage4_result_bundle.zip",
        "files.download(str(archive))",
    ]:
        assert fragment in source, f"missing safeguard or output: {fragment}"

    synthetic_checks(notebook)
    optional_bundle_smoke(notebook)
    print(
        f"PASS: {NOTEBOOK.name} has {len(notebook['cells'])} cells; "
        "all code parses, the frozen matched-error safeguards are present, "
        "and synthetic intervention checks pass."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
