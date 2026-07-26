#!/usr/bin/env python3
"""Static validation for the GPU-only task-aligned Stage 2C notebook."""

from __future__ import annotations

import ast
from pathlib import Path

import nbformat


NOTEBOOK = Path(__file__).with_name("02c_task_aligned_readout.ipynb")


def main() -> int:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    assert len(notebook.cells) == 9
    assert notebook.cells[0].cell_type == "code"
    for name in [
        "RUN_MODE",
        "OUTPUT_DIR",
        "SEED",
        "MODEL_NAME",
        "ENVIRONMENT",
        "HORIZONS",
        "NUM_STATES",
        "ACTIONS_PER_STATE",
        "PROBE_SPLIT",
        "READOUT_PROJECTION_DIM",
        "RIDGE_LAMBDAS",
    ]:
        assert f"{name} =" in notebook.cells[0].source, f"missing config {name}"
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=f"{NOTEBOOK.name}:cell_{index}")

    source = "\n".join(cell.source for cell in notebook.cells)
    for fragment in [
        "pymunk==6.8.0",
        "torch.cuda.memory_allocated",
        "MOUNT_DRIVE",
        "atomic_npz",
        "simulator_progress.json",
        "FAILURE_TRACE.txt",
        "candidate_design_summary.json",
        "fixed_candidate_indices",
        "predicted_features=predicted_features.astype(np.float16)",
        "make_state_split",
        "pose_target",
        "task_cost_from_pose_target",
        "fit_linear_pose",
        "fit_mlp_pose",
        "weighted_pairwise_accuracy",
        "linear_pose_shuffled",
        "test_states_used_for_fitting",
        "state-clustered 95% bootstrap",
        "stage2c_decision.json",
        "probe_manifest.json",
        "action_predictions.csv",
        "stage2c_result_bundle",
        "files.download(str(RESULT_ZIP))",
        "dino_wm_pusht",
        "jepa_wm_pusht",
        "TASK_ALIGNED_SIGNAL",
        "NONLINEAR_TASK_ALIGNED_SIGNAL",
    ]:
        assert fragment in source, f"missing safeguard or output: {fragment}"
    assert 'RUN_MODE = "smoke"' in notebook.cells[0].source
    assert "NUM_STATES = 300" in notebook.cells[0].source
    assert "HORIZONS = [3, 6]" in notebook.cells[0].source
    assert "CV_REPEATS" not in notebook.cells[0].source
    assert "PAIR_EFFECT_SCALE_MIN" not in notebook.cells[0].source
    print(
        f"PASS: {NOTEBOOK.name} has {len(notebook.cells)} cells; "
        "all code parses and Stage 2C leakage/readout safeguards are present."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
