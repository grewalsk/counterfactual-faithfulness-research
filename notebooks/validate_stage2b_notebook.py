#!/usr/bin/env python3
"""Static validation for the GPU-only confirmatory Stage 2B Colab notebook."""

from __future__ import annotations

import ast
from pathlib import Path

import nbformat


NOTEBOOK = Path(__file__).with_name(
    "02b_counterfactual_faithfulness_confirmatory.ipynb"
)


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
        "pair_metrics.csv",
        "incremental_validity.csv",
        "stage2_decision.json",
        "checkpoints_manifest.json",
        "stage2b_result_bundle",
        "files.download(str(RESULT_ZIP))",
        "dino_wm_pusht",
        "jepa_wm_pusht",
        "action_blind",
        "action_shuffled",
        "candidate_library",
        "fixed_candidate_indices",
        "state_index, pair_left",
        "PAIR_EFFECT_SCALE_MIN",
        "bootstrap_oof_improvement",
        "outcome_groups",
        "contact_stratum",
    ]:
        assert fragment in source, f"missing safeguard or output: {fragment}"
    assert 'RUN_MODE = "smoke"' in notebook.cells[0].source
    assert 'NUM_STATES = 250' in notebook.cells[0].source
    print(
        f"PASS: {NOTEBOOK.name} has {len(notebook.cells)} cells; "
        "all code parses and Stage 2B safeguards are present."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
