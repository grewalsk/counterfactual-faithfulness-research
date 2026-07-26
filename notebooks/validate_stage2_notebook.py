#!/usr/bin/env python3
"""Static validation for the GPU-only Stage 2 Colab notebook."""

from __future__ import annotations

import ast
from pathlib import Path

import nbformat


NOTEBOOK = Path(__file__).with_name("02_counterfactual_faithfulness_pilot.ipynb")


def main() -> int:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    assert notebook.cells and notebook.cells[0].cell_type == "code"
    required_config = [
        "RUN_MODE",
        "OUTPUT_DIR",
        "SEED",
        "MODEL_NAME",
        "ENVIRONMENT",
        "HORIZONS",
        "NUM_STATES",
        "ACTIONS_PER_STATE",
    ]
    for name in required_config:
        assert f"{name} =" in notebook.cells[0].source, f"missing config {name}"
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=f"{NOTEBOOK.name}:cell_{index}")
    source = "\n".join(cell.source for cell in notebook.cells)
    required_fragments = [
        "pymunk==6.8.0",
        "torch.cuda.memory_allocated",
        "MOUNT_DRIVE",
        "atomic_npz",
        "simulator_progress.json",
        "FAILURE_TRACE.txt",
        "pair_metrics.csv",
        "incremental_validity.csv",
        "stage2_decision.json",
        "checkpoints_manifest.json",
        "stage2_result_bundle",
        "files.download(str(RESULT_ZIP))",
        "dino_wm_pusht",
        "jepa_wm_pusht",
        "action_blind",
        "action_shuffled",
        "bootstrap_oof_improvement",
        "contact_stratum",
    ]
    for fragment in required_fragments:
        assert fragment in source, f"missing safeguard or output: {fragment}"
    print(
        f"PASS: {NOTEBOOK.name} has {len(notebook.cells)} cells; "
        "all code parses and required Stage 2 safeguards are present."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
