#!/usr/bin/env python3
"""Static and synthetic validation for the Stage 8 Colab notebook."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_functional


NOTEBOOK = Path(__file__).with_name(
    "08_counterfactual_decision_energy.ipynb"
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
    feature_slices = {
        "native": [0, 2],
        "goal": [2, 5],
        "audit": [5, 9],
        "action": [9, 11],
        "total": [0, 11],
    }
    methods = [
        "final_token_energy",
        "action_prior_control",
        "wrong_state_control",
        "counterfactual_energy",
    ]
    namespace = {
        "np": np,
        "torch": torch,
        "torch_functional": torch_functional,
        "hashlib": hashlib,
        "math": math,
        "random": random,
        "ACTIONS_PER_STATE": 10,
        "RANKING_TIE": 1e-9,
        "ENERGY_METHODS": methods,
        "FEATURE_SLICES": feature_slices,
        "DEVELOPMENT_SPLIT": "development_holdout",
        "PAIRWISE_WEIGHT": 1.0,
        "LISTWISE_WEIGHT": 1.0,
        "COST_SHAPE_WEIGHT": 0.25,
        "PAIRWISE_TEMPERATURE": 0.25,
        "LISTWISE_TEMPERATURE": 0.20,
    }
    execute_nodes(
        notebook,
        [
            "pair_indices",
            "ranking_metrics",
            "stable_seed",
            "canonical_split_name",
            "wrong_state_indices",
            "native_anchor",
            "method_inputs",
            "normalized_cost_sets",
            "EnergyHead",
            "energy_ranking_objective",
        ],
        namespace,
    )

    assert namespace["canonical_split_name"]("final_test") == (
        "development_holdout"
    )
    task_ids = np.asarray([1, 1, 1, 2, 2, 2])
    wrong = namespace["wrong_state_indices"](task_ids)
    assert np.all(wrong != np.arange(len(task_ids)))
    assert np.array_equal(task_ids[wrong], task_ids)

    rng = np.random.default_rng(17)
    bundle = {
        "features": rng.normal(size=(6, 3, 10, 11)).astype(np.float32),
        "task_id": task_ids,
    }
    for method in methods:
        values, anchor, stats = namespace["method_inputs"](
            bundle, method
        )
        assert values.shape == (6, 3, 10, 11)
        assert anchor.shape == (6, 3, 10)
        assert np.isfinite(values).all()
        assert np.isfinite(anchor).all()
        repeated, repeated_anchor, _ = namespace["method_inputs"](
            bundle, method, stats=stats
        )
        assert np.array_equal(values, repeated)
        assert np.array_equal(anchor, repeated_anchor)
    action_values, action_anchor, _ = namespace["method_inputs"](
        bundle, "action_prior_control"
    )
    assert np.all(action_anchor == 0.0)
    assert np.all(action_values[..., :9] == 0.0)
    proposed_values, _, _ = namespace["method_inputs"](
        bundle, "counterfactual_energy"
    )
    assert np.all(proposed_values[..., 9:] == 0.0)

    costs = np.stack(
        [
            np.linspace(0.0, 1.0, 10),
            np.linspace(1.0, 2.0, 10),
        ]
    ).astype(np.float32)
    normalized, valid = namespace["normalized_cost_sets"](costs)
    assert valid.tolist() == [True, True]
    assert np.allclose(normalized[0], normalized[1])

    torch.manual_seed(23)
    head = namespace["EnergyHead"](11, 8, 0.0)
    features = torch.randn(5, 10, 11)
    anchor = torch.randn(5, 10)
    initial = head(features, anchor)
    assert torch.equal(initial, anchor)

    target = torch.linspace(0.0, 1.0, 10)[None].repeat(5, 1)
    good, _ = namespace["energy_ranking_objective"](
        target.clone(), target
    )
    bad, _ = namespace["energy_ranking_objective"](
        -target.clone(), target
    )
    assert torch.isfinite(good)
    assert torch.isfinite(bad)
    assert good < bad

    true_cost = np.arange(10, dtype=np.float64)
    good_rank = namespace["ranking_metrics"](true_cost, true_cost)
    bad_rank = namespace["ranking_metrics"](true_cost, true_cost[::-1])
    assert good_rank["normalized_regret"] == 0.0
    assert good_rank["weighted_pairwise_accuracy"] > bad_rank[
        "weighted_pairwise_accuracy"
    ]


def main() -> int:
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["colab"]["name"] == NOTEBOOK.name
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["metadata"]["colab"]["gpuType"] == "A100"
    assert len(notebook["cells"]) == 11
    assert notebook["cells"][0]["cell_type"] == "code"

    config = "".join(notebook["cells"][0]["source"])
    for fragment in [
        'RUN_MODE = "full"',
        'OUTPUT_DIR = "/content/counterfactual_faithfulness_stage8"',
        "REUSE_STAGE7_CACHE = True",
        "SEED = 913",
        'MODEL_NAME = ["jepa_wm_pusht", "jepa_wm_wall"]',
        "HORIZONS = [1, 3, 6]",
        "NUM_STATES = 96",
        "AUDIT_PROJECTION_DIM = 128",
        "AUDIT_PROJECTION_SEEDS = [7101, 9101]",
        "ENERGY_HEAD_SEEDS = [8301, 10301]",
        'ENERGY_IMPLEMENTATION_ID = "set_centered_energy_v1"',
        "TRAINING_EPOCHS = 40",
        "SELECTION_EPOCHS = [10, 20, 30, 40]",
        'EVIDENCE_STATUS = "EXPLORATORY_DEVELOPMENT"',
        'DEVELOPMENT_SPLIT = "development_holdout"',
    ]:
        assert fragment in config, fragment

    source = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"]
    )
    for fragment in [
        "Stage 8: counterfactual decision-energy calibration",
        "stage7_cache_compatible",
        "world_model_predictions_shared_across_methods",
        "pooled_spatial_error",
        "wrong_state_indices",
        "EnergyHead",
        "energy_ranking_objective",
        "final_token_energy",
        "action_prior_control",
        "wrong_state_control",
        "counterfactual_energy",
        "development_holdout_used_for_selection",
        "energy_task_descriptives.csv",
        "stage8_development_decision.json",
        "DECISION_ENERGY_CANDIDATE_READY",
        "DECISION_ENERGY_GAIN_NOT_SPECIFIC",
        "MIXED_DECISION_ENERGY_SIGNAL",
        "NO_DECISION_ENERGY_GAIN",
        "stage8_result_bundle.zip",
        "colab_files.download(str(result_zip))",
    ]:
        assert fragment in source, fragment

    for forbidden in [
        '"evidence_status": "CONFIRMATORY"',
        "Stage 8 establishes",
        "untouched Stage 8 final",
        "ranking_gradient_into_world_model\": True",
        "RECURRENT_COUNTERFACTUAL_CANDIDATE_READY",
        "FEATURE_POOL_GRID",
    ]:
        assert forbidden not in source, forbidden

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        ast.parse(
            "".join(cell["source"]),
            filename=f"{NOTEBOOK.name}:cell_{index}",
        )

    definition_namespace = {
        "csv": csv,
        "hashlib": hashlib,
        "json": json,
        "math": math,
        "np": np,
        "Path": Path,
        "random": random,
        "torch": torch,
        "torch_functional": torch_functional,
    }
    for cell_index in [0, 4]:
        exec(
            compile(
                "".join(notebook["cells"][cell_index]["source"]),
                f"{NOTEBOOK.name}:cell_{cell_index}",
                "exec",
            ),
            definition_namespace,
        )

    synthetic_checks(notebook)
    print(
        f"PASS: {NOTEBOOK.name} has {len(notebook['cells'])} cells; "
        "all code parses, exploratory-evidence safeguards are present, "
        "and synthetic feature masking, wrong-state, zero-initialization, "
        "energy-loss, and ranking checks pass."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
