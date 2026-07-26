#!/usr/bin/env python3
"""Structural validation that does not execute the GPU cells."""

from __future__ import annotations

import ast
from pathlib import Path

import nbformat


NOTEBOOK = Path(__file__).with_name("01_model_and_environment_smoke_test.ipynb")


def main() -> int:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    assert notebook.cells, "notebook is empty"
    first = notebook.cells[0]
    assert first.cell_type == "code", "the notebook must begin with the config block"
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
        assert f"{name} =" in first.source, f"config block is missing {name}"
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=f"{NOTEBOOK.name}:cell_{index}")
    all_source = "\n".join(cell.source for cell in notebook.cells)
    required_fragments = [
        "pymunk==6.8.0",
        "torch.cuda.memory_allocated",
        "MOUNT_DRIVE",
        "np.savez_compressed",
        "progress.json",
        "FAILURE_TRACE.txt",
        "metrics_summary.csv",
        "ranking_metrics.csv",
        "checkpoints_manifest.json",
        "stage1_result_bundle",
        "endpoint_bitwise_exact",
    ]
    for fragment in required_fragments:
        assert fragment in all_source, f"notebook is missing {fragment}"
    print(
        f"PASS: {NOTEBOOK.name} has {len(notebook.cells)} cells; "
        "all code parses and required Colab safeguards are present."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
