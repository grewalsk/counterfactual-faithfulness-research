#!/usr/bin/env python3
"""Static and synthetic validation for the Stage 5 Colab notebook."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


NOTEBOOK = Path(__file__).with_name(
    "05_counterfactual_decision_readout_training.ipynb"
)


def nodes_from_notebook(notebook, names):
    found = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if node.name in names:
                    found[node.name] = node
    return found


def execute_nodes(notebook, names, namespace):
    nodes = nodes_from_notebook(notebook, set(names))
    assert set(nodes) == set(names), set(names) - set(nodes)
    for name in names:
        module = ast.Module(body=[nodes[name]], type_ignores=[])
        exec(compile(module, NOTEBOOK.name, "exec"), namespace)
    return namespace


def synthetic_checks(notebook):
    namespace = {
        "np": np,
        "torch": torch,
        "hashlib": hashlib,
        "PAIR_LEFT": np.triu_indices(10, k=1)[0],
        "PAIR_RIGHT": np.triu_indices(10, k=1)[1],
        "PAIR_WEIGHT": 1.0,
        "BOOTSTRAP_REPS": 300,
    }
    execute_nodes(
        notebook,
        [
            "stable_seed",
            "model_state_hash",
            "random_projection",
            "PoseReadout",
            "objective_loss",
            "bootstrap_mean",
            "bootstrap_ratio",
        ],
        namespace,
    )

    projection = namespace["random_projection"](17, 11, 991)
    assert projection.shape == (17, 11)
    assert projection.dtype == np.float32
    features = np.zeros((5, 17), dtype=np.float32)
    projected = torch.from_numpy(features) @ torch.from_numpy(projection)
    assert projected.dtype == torch.float32
    assert projected.shape == (5, 11)

    torch.manual_seed(17)
    target = torch.randn(4, 10, 3)
    perfect = target.clone()
    ordinary, absolute, ordinary_relation = namespace[
        "objective_loss"
    ]("ordinary_endpoint", perfect, target)
    independent, _, _ = namespace["objective_loss"](
        "independent_pair", perfect, target
    )
    counterfactual, _, _ = namespace["objective_loss"](
        "counterfactual_difference", perfect, target
    )
    shuffled, _, shuffled_relation = namespace["objective_loss"](
        "shuffled_pair", perfect, target
    )
    assert float(ordinary) == 0.0
    assert float(absolute) == 0.0
    assert float(ordinary_relation) == 0.0
    assert float(independent) == 0.0
    assert float(counterfactual) == 0.0
    assert float(shuffled) > 0
    assert float(shuffled_relation) > 0

    common_shift = target + torch.tensor([0.3, -0.2, 0.1])
    ordinary, absolute, relation = namespace["objective_loss"](
        "ordinary_endpoint", common_shift, target
    )
    counterfactual, _, counterfactual_relation = namespace[
        "objective_loss"
    ]("counterfactual_difference", common_shift, target)
    assert float(ordinary) == float(absolute)
    assert float(relation) < 1e-12
    assert float(counterfactual_relation) < 1e-12
    assert abs(float(counterfactual) - float(absolute)) < 1e-12

    corrupted = target.clone()
    corrupted[:, 0] += 0.5
    _, _, corrupted_relation = namespace["objective_loss"](
        "counterfactual_difference", corrupted, target
    )
    assert float(corrupted_relation) > 0

    torch.manual_seed(991)
    model_a = namespace["PoseReadout"](8, 12, 3)
    torch.manual_seed(991)
    model_b = namespace["PoseReadout"](8, 12, 3)
    assert namespace["model_state_hash"](
        model_a
    ) == namespace["model_state_hash"](model_b)

    initial_hashes = []
    completed_updates = []
    features = torch.randn(6, 10, 8)
    training_target = torch.randn(6, 10, 3)
    for objective in [
        "ordinary_endpoint",
        "independent_pair",
        "counterfactual_difference",
        "shuffled_pair",
    ]:
        torch.manual_seed(991)
        model = namespace["PoseReadout"](8, 12, 3)
        initial_hashes.append(namespace["model_state_hash"](model))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        updates = 0
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features)
            loss, _, _ = namespace["objective_loss"](
                objective, prediction, training_target
            )
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            updates += 1
        completed_updates.append(updates)
    assert len(set(initial_hashes)) == 1
    assert set(completed_updates) == {3}

    ratio = namespace["bootstrap_ratio"](
        np.array([1.0, 1.1, 0.9, 1.0]),
        np.ones(4),
        np.array([0, 0, 1, 1]),
        200,
        71,
    )
    assert ratio["n_clusters"] == 2
    assert np.isfinite(ratio["estimate"])
    assert abs(ratio["estimate"] - 1.0) < 1e-12

    assert namespace["stable_seed"]("a", 1) == namespace[
        "stable_seed"
    ]("a", 1)
    assert namespace["stable_seed"]("a", 1) != namespace[
        "stable_seed"
    ]("a", 2)


def main() -> int:
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["colab"]["name"] == NOTEBOOK.name
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert len(notebook["cells"]) == 11
    assert notebook["cells"][0]["cell_type"] == "code"

    config = "".join(notebook["cells"][0]["source"])
    for fragment in [
        'RUN_MODE = "full"',
        "SEED = 811",
        "TASK_SPLIT_COUNTS = [6, 3, 0, 3]",
        "EVALUATION_SEEDS = [811, 1231, 1699]",
        "PROBE_SEEDS = [5101, 7103, 9109]",
        "TRAINING_EPOCHS = 160",
        "PAIR_WEIGHT = 1.0",
        "POSE_ERROR_RATIO_MARGIN = 1.05",
        'TASK_FAMILY_ID = "stage5_prospective_tasks_v1"',
        '"ordinary_endpoint"',
        '"independent_pair"',
        '"counterfactual_difference"',
        '"shuffled_pair"',
    ]:
        assert fragment in config, fragment

    source = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"]
    )
    for fragment in [
        "Stage 5: counterfactual decision-readout training",
        "stage5_pusht_goal_",
        "stage5_wall_layout_goal_",
        "world_models_frozen",
        "calibration_used_for_selection",
        "final_used_for_selection",
        "same_architecture_data_updates",
        "initial_parameter_sha256",
        "completed_updates",
        "counterfactual_difference",
        "independent_pair",
        "shuffled_pair",
        "ordinary_error_noninferiority.csv",
        "objective_contrasts.csv",
        "CROSS_ENV_OBJECTIVE_SPECIFIC_FIX",
        "CROSS_ENV_COUNTERFACTUAL_TRAINING_FIX",
        "PLANNING_GAIN_WITH_ERROR_TRADEOFF",
        "MIXED_TRAINING_SIGNAL",
        "NO_TRAINING_FIX",
        "INCONCLUSIVE",
        "planning_by_objective.png",
        "ordinary_error_by_objective.png",
        "objective_contrasts.png",
        "stage5_result_bundle.zip",
        "files.download(str(RESULT_ZIP))",
        'device="cuda", dtype=torch.float32',
    ]:
        assert fragment in source, fragment

    for old_task in [
        "(210.0, 210.0, -0.75)",
        "(27.0, 18.0, 53.0, 46.0)",
    ]:
        assert old_task not in source

    task_namespace = {
        "np": np,
        "SEED": 811,
        "TASKS_PER_ENVIRONMENT": 12,
        "TASK_SPLIT_COUNTS": [6, 3, 0, 3],
        "SPLIT_NAMES": [
            "probe_train",
            "probe_calibration",
            "regression_train",
            "final_test",
        ],
    }
    execute_nodes(
        notebook,
        ["task_split_map", "pusht_tasks", "wall_tasks"],
        task_namespace,
    )
    for function_name in ["pusht_tasks", "wall_tasks"]:
        tasks = task_namespace[function_name]()
        assert len(tasks) == 12
        counts = {
            split_name: sum(
                task["split"] == split_name for task in tasks
            )
            for split_name in task_namespace["SPLIT_NAMES"]
        }
        assert counts == {
            "probe_train": 6,
            "probe_calibration": 3,
            "regression_train": 0,
            "final_test": 3,
        }

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        ast.parse(
            "".join(cell["source"]),
            filename=f"{NOTEBOOK.name}:cell_{index}",
        )

    synthetic_checks(notebook)
    print(
        f"PASS: {NOTEBOOK.name} has {len(notebook['cells'])} cells; "
        "all code parses, prospective-task and equal-update safeguards are "
        "present, and synthetic objective checks pass."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
