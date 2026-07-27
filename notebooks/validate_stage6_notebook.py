#!/usr/bin/env python3
"""Static and synthetic validation for the Stage 6 Colab notebook."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


NOTEBOOK = Path(__file__).with_name("06_structured_action_effect_development.ipynb")


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
        "endpoint_only",
        "action_decode_only",
        "ranking_only",
        "counterfactual_effect_only",
        "independent_action_effect",
        "counterfactual_action_effect",
    ]
    method_weights = {
        "endpoint_only": {
            "effect": 0.0,
            "independent": 0.0,
            "action": 0.0,
            "ranking": 0.0,
        },
        "action_decode_only": {
            "effect": 0.0,
            "independent": 0.0,
            "action": 0.1,
            "ranking": 0.0,
        },
        "ranking_only": {
            "effect": 0.0,
            "independent": 0.0,
            "action": 0.0,
            "ranking": 0.5,
        },
        "counterfactual_effect_only": {
            "effect": 1.0,
            "independent": 0.0,
            "action": 0.0,
            "ranking": 0.0,
        },
        "independent_action_effect": {
            "effect": 0.0,
            "independent": 1.0,
            "action": 0.1,
            "ranking": 0.5,
        },
        "counterfactual_action_effect": {
            "effect": 1.0,
            "independent": 0.0,
            "action": 0.1,
            "ranking": 0.5,
        },
    }
    namespace = {
        "np": np,
        "torch": torch,
        "math": math,
        "hashlib": hashlib,
        "NUM_STATES": 36,
        "ACTIONS_PER_STATE": 10,
        "HORIZONS": [1, 3, 6],
        "FRAMESKIP": 5,
        "PAIR_LEFT": np.triu_indices(10, k=1)[0],
        "PAIR_RIGHT": np.triu_indices(10, k=1)[1],
        "METHODS": methods,
        "METHOD_WEIGHTS": method_weights,
        "RANK_TEMPERATURE": 0.05,
        "LOWER_IS_BETTER": {
            "pose_error",
            "physical_cost_rmse",
            "normalized_regret",
            "normalized_margin_rmse",
        },
        "DEVELOPMENT_SPLIT": "development_holdout",
    }
    execute_nodes(
        notebook,
        [
            "build_action_descriptors",
            "ActionEffectAdapter",
            "differentiable_task_cost",
            "weighted_pairwise_ranking_loss",
            "stage6_objective_loss",
            "percentile_interval",
            "state_metric_map",
            "clustered_method_contrast",
            "clustered_pose_ratio",
        ],
        namespace,
    )

    action_bank = np.zeros((2, 10, 30, 2), dtype=np.float32)
    action_bank[:, 1, :12, 0] = 0.2
    descriptors = namespace["build_action_descriptors"](action_bank)
    assert descriptors.shape == (2, 10, 3, 6)
    assert descriptors.dtype == np.float32
    assert np.all(descriptors[:, 0] == 0)
    assert np.all(descriptors[:, 1, :, 0] > 0)

    torch.manual_seed(17)
    adapter = namespace["ActionEffectAdapter"](
        input_dim=32,
        bottleneck_dim=16,
        hidden_dim=24,
        pose_dim=4,
        action_descriptor_dim=6,
        horizon_count=3,
    )
    features = torch.randn(4, 10, 32)
    action = torch.randn(4, 10, 6)
    horizon = torch.tensor([0, 1, 2, 0])
    pose, decoded_action, effect = adapter(features, action, horizon)
    assert pose.shape == (4, 10, 4)
    assert decoded_action.shape == (4, 10, 6)
    assert effect.shape == (4, 10, 4)
    assert torch.allclose(effect[:, 0], torch.zeros_like(effect[:, 0]))

    push_goal = torch.tensor([[0.5, 0.5, 0.0]] * 4, dtype=torch.float32)
    wall_goal = torch.tensor([[0.7, 0.3, 0.0]] * 4, dtype=torch.float32)
    push_cost = namespace["differentiable_task_cost"]("PushT", pose, push_goal)
    wall_cost = namespace["differentiable_task_cost"]("Wall", pose[..., :2], wall_goal)
    assert push_cost.shape == (4, 10)
    assert wall_cost.shape == (4, 10)
    assert torch.isfinite(push_cost).all()
    assert torch.isfinite(wall_cost).all()

    true_cost = torch.arange(10, dtype=torch.float32)[None].repeat(4, 1)
    good_cost = true_cost.clone().requires_grad_(True)
    bad_cost = torch.flip(true_cost, dims=[1]).requires_grad_(True)
    good_loss = namespace["weighted_pairwise_ranking_loss"](good_cost, true_cost, 0.05)
    bad_loss = namespace["weighted_pairwise_ranking_loss"](bad_cost, true_cost, 0.05)
    assert float(good_loss.detach()) < float(bad_loss.detach())

    target = torch.randn(4, 10, 4)
    standardized_action = torch.randn(4, 10, 6)
    target_mean = torch.zeros(4)
    target_scale = torch.ones(4)
    for method in methods:
        prediction = target.clone().requires_grad_(True)
        decoded = standardized_action.clone().requires_grad_(True)
        total, components = namespace["stage6_objective_loss"](
            method,
            "PushT",
            prediction,
            target,
            decoded,
            standardized_action,
            true_cost,
            push_goal,
            target_mean,
            target_scale,
        )
        assert torch.isfinite(total)
        assert set(components) == {
            "absolute_loss",
            "counterfactual_effect_loss",
            "independent_effect_loss",
            "action_decode_loss",
            "ranking_loss",
        }
        total.backward()
        assert prediction.grad is not None

    adapter.zero_grad(set_to_none=True)
    model_prediction, model_action, _ = adapter(features, action, horizon)
    total, _ = namespace["stage6_objective_loss"](
        "counterfactual_action_effect",
        "PushT",
        model_prediction,
        target,
        model_action,
        standardized_action,
        true_cost,
        push_goal,
        target_mean,
        target_scale,
    )
    total.backward()
    assert adapter.projector[0].weight.grad is not None
    assert torch.isfinite(adapter.projector[0].weight.grad).all()

    rows = []
    for state_id in range(8):
        for method, offset in [
            ("endpoint_only", 0.1),
            ("counterfactual_action_effect", 0.0),
        ]:
            rows.append(
                {
                    "environment": "PushT",
                    "state_id": state_id,
                    "split": "development_holdout",
                    "method": method,
                    "normalized_regret": state_id / 100 + offset,
                    "pose_error": 1.0 + offset,
                }
            )
    contrast = namespace["clustered_method_contrast"](
        rows,
        "PushT",
        "endpoint_only",
        "counterfactual_action_effect",
        "normalized_regret",
        200,
        71,
    )
    assert contrast["n_clusters"] == 8
    assert abs(contrast["estimate"] - 0.1) < 1e-12
    assert contrast["low"] > 0
    ratio = namespace["clustered_pose_ratio"](
        rows,
        "PushT",
        "counterfactual_action_effect",
        "endpoint_only",
        200,
        91,
    )
    assert ratio["n_clusters"] == 8
    assert ratio["high"] < 1.0


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
        'OUTPUT_DIR = "/content/counterfactual_faithfulness_stage6"',
        "SEED = 811",
        "TASK_SPLIT_COUNTS = [6, 3, 0, 3]",
        "PROBE_SEEDS = [6101, 8101]",
        "ADAPTER_BOTTLENECK_DIM = 128",
        "ADAPTER_HIDDEN_DIM = 192",
        'ADAPTER_IMPLEMENTATION_ID = "set_noop_effect_rank_v1"',
        "TRAINING_EPOCHS = 160",
        "SELECTION_EPOCHS = [80, 120, 160]",
        "EFFECT_WEIGHT = 1.0",
        "RANK_WEIGHT = 0.5",
        "ACTION_DECODE_WEIGHT = 0.1",
        "RANK_TEMPERATURE = 0.05",
        'EVIDENCE_STATUS = "EXPLORATORY_DEVELOPMENT"',
        '"counterfactual_action_effect"',
        '"independent_action_effect"',
    ]:
        assert fragment in config, fragment

    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    for fragment in [
        "Stage 6: structured action-effect adapter development",
        "former final partition is renamed `development_holdout`",
        "stage5_tasks_reused_for_stage6_development",
        "stage5_final_tasks_reused_and_not_confirmatory",
        "development_holdout_used_for_selection",
        "ActionEffectAdapter",
        "effect = raw_effect - raw_effect[:, :1]",
        "weighted_pairwise_ranking_loss",
        "calibration_selection_score",
        "counterfactual_effect_only",
        "independent_action_effect",
        "counterfactual_action_effect",
        "DEVELOPMENT_CANDIDATE_READY",
        "PROMISING_BUT_NOT_SPECIFIC",
        "MIXED_DEVELOPMENT_SIGNAL",
        "NO_DEVELOPMENT_GAIN",
        "INCONCLUSIVE",
        "stage6_development_decision.json",
        "method_contrasts.csv",
        "pose_error_noninferiority.csv",
        "stage6_result_bundle.zip",
        "files.download(str(RESULT_ZIP))",
    ]:
        assert fragment in source, fragment

    for forbidden in [
        '"evidence_status": "CONFIRMATORY"',
        "Stage 6 establishes",
        "untouched Stage 6 final",
    ]:
        assert forbidden not in source

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
        "all code parses, exploratory-evidence safeguards are present, "
        "and synthetic action-effect, ranking, and bootstrap checks pass."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
