import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "13_jacobian_outcome_workspace_screen.ipynb"
BUILDER = ROOT / "build_stage13_jow_notebook.py"


def payload():
    return json.loads(NOTEBOOK.read_text())


def function_source(code_cells, function_name):
    for source in code_cells:
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                segment = ast.get_source_segment(source, node)
                if segment is None:
                    raise AssertionError(f"could not recover {function_name}")
                return segment
    raise AssertionError(f"missing function {function_name}")


def validate_structure():
    notebook = payload()
    if len(notebook["cells"]) != 11:
        raise AssertionError(
            f"expected 11 cells, found {len(notebook['cells'])}"
        )
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    if len(code_cells) != 10:
        raise AssertionError(
            f"expected 10 code cells, found {len(code_cells)}"
        )
    for index, source in enumerate(code_cells):
        try:
            ast.parse(source)
        except SyntaxError as error:
            raise AssertionError(
                f"invalid Python in code cell {index}: {error}"
            ) from error
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("id") != f"stage13-{index:02d}":
            raise AssertionError(f"bad cell id at {index}")
        if cell.get("outputs"):
            raise AssertionError(f"stale outputs in cell {index}")
        if (
            cell["cell_type"] == "code"
            and cell.get("execution_count") is not None
        ):
            raise AssertionError(f"stale execution count in cell {index}")
    if (
        notebook["metadata"]["colab"]["name"]
        != NOTEBOOK.name
    ):
        raise AssertionError("bad Colab notebook name")
    if notebook["metadata"].get("accelerator") != "GPU":
        raise AssertionError("notebook does not request a GPU")

    joined = "\n".join(code_cells)
    required = [
        'RUN_MODE = "screen"',
        "CONSTRUCTION_STATES = 8",
        "CALIBRATION_STATES = 4",
        "HORIZONS = [1, 3]",
        "BLOCKS = list(range(6))",
        "PROTOTYPE_AXES = 8",
        "torch.autograd.grad(",
        "def state_vjp_lens(",
        "def unroll_with_hooks(",
        "def coordinate_gate(",
        "def run_causal_swaps(",
        '["jow", "residual_swap", "random_orthogonal"]',
        "relative_edit_norm",
        "activation_z_ratio",
        "activation_variance_fraction",
        "total_activation_variance_fraction",
        "action_dependent_activation_variance_fraction",
        "sparse_coordinate_energy_fraction",
        "mean_state_lens_alignment",
        "construction_grouped_cv_r2",
        '"layer_selection_split":',
        "CAUSAL_EVALUATE_ALL_LAYERS",
        "decode_physical_pose",
        "STOP_NO_FROZEN_CAUSAL_JOW_SIGNAL",
        "PROMOTE_TO_PHASE0_EXPANSION",
        "PROMOTE_TO_BROADCAST_AND_NEW_TASK_DESIGN",
        "PROMOTE_ARGA_TREATMENT_TO_EXPANSION",
        '"primary_lens_source": "frozen"',
        "SHARED_FROZEN_LENS",
        'shutil.make_archive(str(OUT / "stage13_jow_result_bundle"), "zip"',
        'print(f"RUN_STATUS:',
        "ASSET_COMMIT",
        "EXPECTED_PRETRAINED_ASSET_SHA256",
    ]
    missing = [needle for needle in required if needle not in joined]
    if missing:
        raise AssertionError(f"missing Stage 13 elements: {missing}")
    prohibited = [
        "torch.autograd.functional.jacobian",
        "torch.func.jacrev",
        "optimizer.step(",
        ".backward(",
        "NUM_STATES = 96",
    ]
    present = [needle for needle in prohibited if needle in joined]
    if present:
        raise AssertionError(
            f"compute-minimal notebook contains prohibited work: {present}"
        )

    vjp = function_source(code_cells, "state_vjp_lens")
    if "torch.autograd.grad(" not in vjp:
        raise AssertionError("lens does not use vector-Jacobian products")
    if "gc.collect()" not in vjp or "torch.cuda.empty_cache()" not in vjp:
        raise AssertionError("per-state graph cleanup is absent")

    swaps = function_source(code_cells, "run_causal_swaps")
    for required_control in [
        '"jow"',
        '"residual_swap"',
        '"random_orthogonal"',
    ]:
        if required_control not in swaps:
            raise AssertionError(
                f"causal swaps omit {required_control} control"
            )
    if "STATE_SELECTION[\"calibration\"]" not in swaps:
        raise AssertionError("causal swaps do not stay on calibration states")
    run_condition = function_source(code_cells, "run_condition")
    if "build_condition_lens(" in run_condition:
        raise AssertionError("adapted conditions refit the primary lens")
    causal_gate = function_source(code_cells, "causal_gate")
    if "max(layer_means" in causal_gate or "best_block" in causal_gate:
        raise AssertionError("causal outcomes select their own primary layer")
    if "primary_layer" not in causal_gate:
        raise AssertionError("causal gate lacks a preselected primary layer")
    return code_cells


def validate_builder_reproducibility():
    before = hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()
    subprocess.run(
        [sys.executable, str(BUILDER)],
        cwd=ROOT.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    after = hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()
    if before != after:
        raise AssertionError("Stage 13 builder is not deterministic")


if __name__ == "__main__":
    validate_structure()
    validate_builder_reproducibility()
    print("Stage 13 JOW notebook validation passed")
