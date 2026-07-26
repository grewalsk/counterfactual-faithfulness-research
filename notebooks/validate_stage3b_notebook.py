#!/usr/bin/env python3
"""Static and synthetic CPU validation for the Stage 3B Colab notebook."""

from __future__ import annotations

import ast
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np


NOTEBOOK = Path(__file__).with_name("03b_stage3_analysis_repair.ipynb")


def synthetic_checks(notebook):
    rows = [
        {
            "split": "regression_train",
            "normalized_regret": 0.2,
            "pose_error": 0.1,
            "physical_cost_rmse": 0.1,
            "ordinary_feature_rmse": 0.2,
            "common_mode_feature_rmse": 0.2,
            "interaction_fraction": 0.5,
            "normalized_paired_feature_rmse": 0.4,
            "normalized_margin_rmse": 0.3,
        },
        {
            "split": "regression_train",
            "normalized_regret": 0.0,
            "pose_error": 0.1,
            "physical_cost_rmse": 0.1,
            "ordinary_feature_rmse": 0.2,
            "common_mode_feature_rmse": 0.2,
            "interaction_fraction": 0.0,
            "normalized_paired_feature_rmse": 0.0,
            "normalized_margin_rmse": np.nan,
        },
    ]
    required = [
        "normalized_regret",
        "pose_error",
        "physical_cost_rmse",
        "ordinary_feature_rmse",
        "common_mode_feature_rmse",
        "interaction_fraction",
        "normalized_paired_feature_rmse",
        "normalized_margin_rmse",
    ]
    finite = [
        row
        for row in rows
        if all(np.isfinite(float(row[field])) for field in required)
    ]
    assert len(finite) == 1

    interaction_types = np.array(
        [["free", "door_cross"], ["collision", "door_cross"]]
    )
    active = interaction_types != "free"
    left = np.array([0])
    right = np.array([1])
    pair_count = (
        active[:, left].astype(int) + active[:, right].astype(int)
    )
    assert pair_count[:, 0].tolist() == [1, 2]

    function_names = {
        "write_json",
        "standardize_fit",
        "bootstrap_mean",
        "regression_design",
        "fit_ridge_regression",
        "predict_ridge_regression",
        "regression_metrics",
        "held_out_regression",
        "rank_values",
        "package_results",
    }
    definitions = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in function_names:
                definitions[node.name] = node
    assert set(definitions) == function_names
    namespace = {
        "np": np,
        "json": json,
        "Path": Path,
        "zipfile": zipfile,
        "ENVIRONMENT": ["PushT", "Wall"],
        "HORIZONS": [1, 3, 6],
        "REGRESSION_RIDGE": 1e-3,
        "BOOTSTRAP_REPS": 50,
        "SEED": 71,
    }
    for name in [
        "write_json",
        "standardize_fit",
        "bootstrap_mean",
        "regression_design",
        "fit_ridge_regression",
        "predict_ridge_regression",
        "regression_metrics",
        "held_out_regression",
        "rank_values",
        "package_results",
    ]:
        module = ast.Module(body=[definitions[name]], type_ignores=[])
        exec(compile(module, NOTEBOOK.name, "exec"), namespace)

    synthetic_rows = []
    for split_index, split_name in enumerate(
        ["regression_train", "final_test"]
    ):
        for environment_index, environment in enumerate(["PushT", "Wall"]):
            for horizon in [1, 3, 6]:
                for state_offset in range(6):
                    index = (
                        split_index * 36
                        + environment_index * 18
                        + (horizon - 1) * 2
                        + state_offset
                    )
                    task_margin = 0.2 + 0.01 * index
                    if (
                        environment == "PushT"
                        and horizon == 1
                        and state_offset == 0
                    ):
                        task_margin = np.nan
                    synthetic_rows.append(
                        {
                            "readout": "linear_pose",
                            "split": split_name,
                            "environment": environment,
                            "state_id": (
                                split_index * 100
                                + environment_index * 50
                                + horizon * 10
                                + state_offset
                            ),
                            "task_id": split_index * 2 + environment_index,
                            "model": (
                                "jepa"
                                if state_offset % 2
                                else "dino"
                            ),
                            "model_family": (
                                "JEPA-WM"
                                if state_offset % 2
                                else "DINO-WM"
                            ),
                            "probe_seed": 2071,
                            "horizon": horizon,
                            "normalized_regret": 0.05 + 0.003 * index,
                            "pose_error": 0.1 + 0.002 * index,
                            "physical_cost_rmse": 0.08 + 0.001 * index,
                            "ordinary_feature_rmse": 0.2 + 0.001 * index,
                            "common_mode_feature_rmse": 0.1 + 0.001 * index,
                            "interaction_fraction": (
                                environment_index + state_offset / 10
                            ),
                            "normalized_paired_feature_rmse": (
                                0.4 + 0.001 * index
                            ),
                            "normalized_margin_rmse": task_margin,
                        }
                    )
    payload, prediction_rows = namespace["held_out_regression"](
        synthetic_rows
    )
    assert payload["models"]["ordinary_only"]["num_train_rows"] == 35
    assert payload["models"]["ordinary_only"]["num_test_rows"] == 35
    assert len(prediction_rows) == 35
    assert all(
        np.isfinite(row["ordinary_plus_task_pair_prediction"])
        for row in prediction_rows
    )
    ranks = namespace["rank_values"]([0.2, np.nan, 0.1])
    assert ranks.tolist()[0] == 2
    assert np.isnan(ranks[1])
    assert ranks.tolist()[2] == 1

    with tempfile.TemporaryDirectory() as temporary:
        output_dir = Path(temporary) / "stage3b"
        probe_dir = output_dir / "probes"
        probe_dir.mkdir(parents=True)
        expected_files = {
            "FAILURE_TRACE.txt": "NONE\n",
            "metrics_summary.json": '{"status": "SUCCESS"}\n',
            "unit_metrics.csv": "metric\n1\n",
            "stage3b_decision.json": '{"status": "TEST"}\n',
            "stage3b_revision.json": '{"revision": "TEST"}\n',
        }
        for name, content in expected_files.items():
            (output_dir / name).write_text(content)
        namespace.update(
            {
                "OUT": output_dir,
                "PROBE_DIR": probe_dir,
                "MOUNT_DRIVE": True,
            }
        )
        archive = namespace["package_results"]()
        assert archive.exists()
        with zipfile.ZipFile(archive) as handle:
            included = set(handle.namelist())
        assert set(expected_files).issubset(included)
        assert "result_zip_manifest.json" in included
        manifest = json.loads(
            (output_dir / "result_zip_manifest.json").read_text()
        )
        assert set(expected_files).issubset(set(manifest["included"]))


def main() -> int:
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 10
    assert notebook["metadata"]["colab"]["name"] == NOTEBOOK.name
    assert notebook["cells"][0]["cell_type"] == "code"
    config = "".join(notebook["cells"][0]["source"])
    assert 'RUN_MODE = "full"' in config
    write_json_call_count = 0
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
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
                    write_json_call_count += 1
                    assert len(node.args) == 2, (
                        "write_json must receive exactly path and payload"
                    )
                    assert not node.keywords
    assert write_json_call_count >= 10

    source = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"]
    )
    required_fragments = [
        "Stage 3B: finite-row analysis repair",
        "finite_decision_row",
        "All specifications use the same rows",
        'interaction_type_vector != "free"',
        'interaction_types[:, left, :] != "free"',
        'interaction_types != "free", axis=(0, 1)',
        "finite_counterfactual",
        "counterfactual_num_rows",
        "stage3b_revision.json",
        "stage3b_decision.json",
        '"unit_metrics.csv"',
        "stage3b_result_bundle.zip",
        "files.download(str(RESULT_ZIP))",
    ]
    for fragment in required_fragments:
        assert fragment in source, f"missing repair safeguard: {fragment}"
    forbidden_fragments = [
        "np.mean(interaction_vector > 0)",
        "interaction_vector[left] > 0",
        "interaction_vector[right] > 0",
        "Path(\"/content/stage3_result_bundle.zip\")",
    ]
    for fragment in forbidden_fragments:
        assert fragment not in source, f"stale Stage 3 logic: {fragment}"

    synthetic_checks(notebook)
    print(
        f"PASS: {NOTEBOOK.name} has {len(notebook['cells'])} cells; "
        "all code parses, finite-row handling and door-crossing safeguards "
        "are present, every write_json call has valid arity, and the synthetic "
        "result bundle packages successfully."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
