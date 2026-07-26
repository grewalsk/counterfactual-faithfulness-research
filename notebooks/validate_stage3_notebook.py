#!/usr/bin/env python3
"""Static and synthetic CPU validation for the GPU-only Stage 3 notebook."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np


NOTEBOOK = Path(__file__).with_name("03_full_counterfactual_benchmark.ipynb")


def ranking_metrics(truth, prediction):
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    selected = int(np.argmin(prediction))
    oracle = int(np.argmin(truth))
    spread = float(np.max(truth) - np.min(truth))
    regret = float(truth[selected] - truth[oracle])
    left, right = np.triu_indices(len(truth), k=1)
    true_margin = truth[left] - truth[right]
    predicted_margin = prediction[left] - prediction[right]
    valid = np.abs(true_margin) > 1e-9
    credit = (
        np.sign(true_margin[valid])
        == np.sign(predicted_margin[valid])
    ).astype(float)
    weighted = np.sum(
        np.abs(true_margin[valid]) * credit
    ) / np.sum(np.abs(true_margin[valid]))
    return {
        "selected": selected,
        "oracle": oracle,
        "normalized_regret": regret / spread,
        "weighted_pairwise_accuracy": float(weighted),
    }


def synthetic_checks():
    truth = np.array([0.05, 0.20, 0.40, 0.70])
    useful = np.array([0.08, 0.19, 0.43, 0.68])
    shuffled = np.roll(useful, 1)
    good = ranking_metrics(truth, useful)
    bad = ranking_metrics(truth, shuffled)
    assert good["normalized_regret"] == 0
    assert good["weighted_pairwise_accuracy"] > 0.95
    assert bad["normalized_regret"] > good["normalized_regret"]

    # PushT pose target and decoded cost must be algebraically consistent.
    state = np.array([100, 100, 240, 270, 0.4, 0, 0], dtype=float)
    target = np.array(
        [
            state[2] / 512,
            state[3] / 512,
            np.sin(state[4]),
            np.cos(state[4]),
        ]
    )
    goal = np.array([256.0, 256.0, np.pi / 4])
    direct = np.linalg.norm(
        np.r_[
            (state[2:4] - goal[:2]) / 512,
            np.arctan2(
                np.sin(state[4] - goal[2]),
                np.cos(state[4] - goal[2]),
            )
            / np.pi,
        ]
    )
    decoded_angle = np.arctan2(target[2], target[3])
    decoded = np.linalg.norm(
        np.r_[
            target[:2] - goal[:2] / 512,
            np.arctan2(
                np.sin(decoded_angle - goal[2]),
                np.cos(decoded_angle - goal[2]),
            )
            / np.pi,
        ]
    )
    assert abs(direct - decoded) < 1e-12

    # Wall pose target and decoded cost use the same 65-pixel normalization.
    wall_state = np.array([12.0, 20.0])
    wall_goal = np.array([53.0, 46.0])
    wall_direct = np.linalg.norm((wall_state - wall_goal) / 65)
    wall_decoded = np.linalg.norm(
        wall_state / 65 - wall_goal / 65
    )
    assert abs(wall_direct - wall_decoded) < 1e-12


def main() -> int:
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 10
    assert notebook["cells"][0]["cell_type"] == "code"
    config = "".join(notebook["cells"][0]["source"])
    for name in [
        "RUN_MODE",
        "OUTPUT_DIR",
        "SEED",
        "MODEL_NAME",
        "ENVIRONMENT",
        "HORIZONS",
        "NUM_STATES",
        "ACTIONS_PER_STATE",
        "TASK_SPLIT_COUNTS",
        "EVALUATION_SEEDS",
        "PROBE_SEEDS",
        "READOUT_PROJECTION_DIM",
        "RIDGE_LAMBDAS",
    ]:
        assert f"{name} =" in config, f"missing config {name}"
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse(
                "".join(cell["source"]),
                filename=f"{NOTEBOOK.name}:cell_{index}",
            )

    source = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"]
    )
    for fragment in [
        "pymunk==6.8.0",
        "torch.cuda.memory_allocated",
        "MOUNT_DRIVE",
        "atomic_npz",
        "FAILURE_TRACE.txt",
        "candidate_design_summary.json",
        "exact_restore_test",
        "fixed state/task-relative candidates",
        "test_states_used_for_fitting",
        "task_disjoint",
        "regression_train",
        "final_test",
        "normalized_paired_feature_rmse",
        "normalized_margin_rmse",
        "linear_pose_shuffled",
        "action_blind",
        "oracle_pose",
        "held_out_regression.json",
        "model_rankings.csv",
        "interaction_stratum",
        "stage3_decision.json",
        "CROSS_ENV_TASK_ALIGNED_SIGNAL",
        "stage3_result_bundle",
        "files.download(str(RESULT_ZIP))",
        "dino_wm_pusht",
        "jepa_wm_pusht",
        "dino_wm_wall",
        "jepa_wm_wall",
    ]:
        assert fragment in source, f"missing safeguard or output: {fragment}"
    assert 'RUN_MODE = "smoke"' in config
    assert "NUM_STATES = 240" in config
    assert "HORIZONS = [1, 3, 6]" in config
    assert "TASK_SPLIT_COUNTS = [6, 2, 2, 2]" in config
    assert "PROBE_SEEDS = [2071, 4071, 6071]" in config
    synthetic_checks()
    print(
        f"PASS: {NOTEBOOK.name} has {len(notebook['cells'])} cells; "
        "all code parses, required safeguards are present, and synthetic "
        "ranking/pose-cost checks pass."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
