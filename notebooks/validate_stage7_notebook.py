#!/usr/bin/env python3
"""Static and synthetic validation for the Stage 7 Colab notebook."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_functional


NOTEBOOK = Path(__file__).with_name(
    "07_recurrent_counterfactual_transition_adapter.ipynb"
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
    methods = [
        "absolute_residual",
        "independent_delta_control",
        "counterfactual_recurrent",
    ]
    method_weights = {
        "absolute_residual": {
            "counterfactual": 0.0,
            "independent": 0.0,
        },
        "independent_delta_control": {
            "counterfactual": 0.0,
            "independent": 1.0,
        },
        "counterfactual_recurrent": {
            "counterfactual": 1.0,
            "independent": 0.0,
        },
    }
    namespace = {
        "np": np,
        "torch": torch,
        "torch_functional": torch_functional,
        "hashlib": hashlib,
        "math": math,
        "random": random,
        "METHOD_WEIGHTS": method_weights,
        "RANKING_TIE": 1e-9,
    }
    execute_nodes(
        notebook,
        [
            "pair_indices",
            "ranking_metrics",
            "CountSketchProjector",
            "stable_seed",
            "TokenResidualAdapter",
            "dynamic_token_weights",
            "weighted_token_huber",
            "recurrent_adapter_objective",
            "paired_latent_metrics",
        ],
        namespace,
    )

    projector = namespace["CountSketchProjector"](
        input_dim=16 * 12,
        output_dim=17,
        seed=41,
        device="cpu",
    )
    value = torch.randn(7, 16, 12)
    projected_a = projector(value)
    projected_b = projector(value)
    assert projected_a.shape == (7, 17)
    assert torch.equal(projected_a, projected_b)

    torch.manual_seed(17)
    adapter = namespace["TokenResidualAdapter"](
        token_dim=12,
        action_dim=4,
        hidden_dim=8,
        steps=3,
        tokens_per_frame=16,
    )
    base = torch.randn(20, 16, 12)
    delta = torch.randn(20, 16, 12)
    action = torch.randn(20, 4)
    step = torch.arange(20) % 3
    initially_corrected = adapter(base, delta, action, step)
    assert initially_corrected.shape == base.shape
    # The zero-initialized residual head makes the adapter initially identity.
    assert torch.equal(initially_corrected, base)

    target = torch.randn(2, 10, 3, 16, 12)
    prediction = (target + 0.05 * torch.randn_like(target)).requires_grad_()
    token_weight = namespace["dynamic_token_weights"](target)
    assert token_weight.shape == (2, 3, 16)
    assert torch.isfinite(token_weight).all()
    for method in methods:
        corrected = prediction.detach().clone().requires_grad_(True)
        total, components = namespace["recurrent_adapter_objective"](
            method, corrected, target
        )
        assert torch.isfinite(total)
        assert set(components) == {
            "absolute_loss",
            "counterfactual_delta_loss",
            "independent_delta_loss",
        }
        total.backward()
        assert corrected.grad is not None
        assert torch.isfinite(corrected.grad).all()

    truth = np.random.default_rng(7).normal(size=(10, 16, 12))
    ordinary, paired = namespace["paired_latent_metrics"](truth, truth)
    assert ordinary == 0.0
    assert paired == 0.0

    true_cost = np.arange(10, dtype=np.float64)
    good = namespace["ranking_metrics"](true_cost, true_cost)
    bad = namespace["ranking_metrics"](true_cost, true_cost[::-1])
    assert good["normalized_regret"] == 0.0
    assert good["weighted_pairwise_accuracy"] > bad[
        "weighted_pairwise_accuracy"
    ]
    assert namespace["stable_seed"]("x", 1) == namespace["stable_seed"](
        "x", 1
    )


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
        'OUTPUT_DIR = "/content/counterfactual_faithfulness_stage7"',
        "SEED = 913",
        'MODEL_NAME = ["jepa_wm_pusht", "jepa_wm_wall"]',
        "HORIZONS = [1, 3, 6]",
        "TARGET_STEPS = list(range(1, max(HORIZONS) + 1))",
        "NUM_STATES = 96",
        "AUDIT_PROJECTION_DIM = 128",
        "AUDIT_PROJECTION_SEEDS = [7101, 9101]",
        "ADAPTER_SEEDS = [7201, 9201]",
        'ADAPTER_IMPLEMENTATION_ID = "recurrent_token_delta_v1"',
        "TRAINING_EPOCHS = 30",
        "SELECTION_EPOCHS = [10, 20, 30]",
        "COUNTERFACTUAL_WEIGHT = 1.0",
        'EVIDENCE_STATUS = "EXPLORATORY_DEVELOPMENT"',
        'DEVELOPMENT_SPLIT = "development_holdout"',
    ]:
        assert fragment in config, fragment

    source = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"]
    )
    for fragment in [
        "Stage 7: recurrent counterfactual transition adapter",
        "former final split `development_holdout`",
        "wanted = set(TARGET_STEPS)",
        "all_future_visual",
        "all_future_proprio",
        "all_endpoint_states",
        '"base_proprio"',
        "VisionTransformerAdaLN",
        "CountSketchProjector",
        "TokenResidualAdapter",
        "counterfactual_recurrent",
        "independent_delta_control",
        "ranking_gradient_into_dynamics",
        "spatial_pooling_before_adapter",
        "predicted_cost += 0.1 * np.mean",
        "layerwise_audit.csv",
        "recurrent_unroll_with_adapter",
        "stage7_development_decision.json",
        "RECURRENT_COUNTERFACTUAL_CANDIDATE_READY",
        "RECURRENT_GAIN_NOT_SPECIFIC",
        "MIXED_RECURRENT_SIGNAL",
        "NO_RECURRENT_DEVELOPMENT_GAIN",
        "stage7_result_bundle.zip",
        "colab_files.download(str(result_zip))",
        "The expensive autoregressive",
    ]:
        assert fragment in source, fragment

    for forbidden in [
        '"evidence_status": "CONFIRMATORY"',
        "Stage 7 establishes",
        "untouched Stage 7 final",
        "prospective task-disjoint training/calibration/final-test",
        "FEATURE_POOL_GRID",
        "def pool_visual",
        "RANK_WEIGHT",
    ]:
        assert forbidden not in source, forbidden

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        ast.parse(
            "".join(cell["source"]),
            filename=f"{NOTEBOOK.name}:cell_{index}",
        )

    # Execute the configuration and shared-definition cells in a minimal
    # namespace.  AST parsing alone does not resolve names used in function
    # defaults, which is how the missing FEATURE_POOL_GRID regression escaped.
    definition_namespace = {
        "csv": __import__("csv"),
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
        "and synthetic sketch, residual, counterfactual-loss, and ranking "
        "checks pass."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
