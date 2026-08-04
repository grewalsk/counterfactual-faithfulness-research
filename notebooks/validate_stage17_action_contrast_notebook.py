import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "17_finite_action_contrast_interchange.ipynb"
BUILDER = ROOT / "build_stage17_action_contrast_notebook.py"


def payload():
    return json.loads(NOTEBOOK.read_text())


def function_source(code_cells, name):
    for source in code_cells:
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
                value = ast.get_source_segment(source, node)
                if value is None:
                    raise AssertionError(f"could not recover {name}")
                return value
    raise AssertionError(f"missing function {name}")


def validate_structure():
    notebook = payload()
    if len(notebook["cells"]) != 15:
        raise AssertionError(f"expected 15 cells, found {len(notebook['cells'])}")
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    if len(code_cells) != 14:
        raise AssertionError(f"expected 14 code cells, found {len(code_cells)}")
    for index, source in enumerate(code_cells):
        try:
            ast.parse(source)
        except SyntaxError as error:
            raise AssertionError(f"invalid Python in code cell {index}: {error}") from error
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("id") != f"stage17-{index:02d}":
            raise AssertionError(f"bad Stage 17 cell id at {index}")
        if cell.get("outputs"):
            raise AssertionError(f"stale output in cell {index}")
        if cell["cell_type"] == "code" and cell.get("execution_count") is not None:
            raise AssertionError(f"stale execution count in cell {index}")
    if notebook["metadata"].get("accelerator") != "GPU":
        raise AssertionError("notebook does not request a GPU")
    if notebook["metadata"]["colab"]["name"] != NOTEBOOK.name:
        raise AssertionError("Colab notebook name mismatch")

    joined = "\n".join(code_cells)
    required = [
        'RUN_MODE = "smoke"',
        'STAGE17_RUN_MODE',
        'STAGE17_SOURCE_COMMIT',
        'PROTOCOL_ID = "stage17-finite-action-contrast-interchange-v1"',
        'TRAIN_OUTPUT_SKETCH_SEED = 17161',
        'EVAL_OUTPUT_SKETCH_SEED = 17183',
        'PRIMARY_RANK = 32',
        'SENSITIVITY_RANKS = [4, 8, 16, 32, 64]',
        'CONSTRUCTION_TRAJECTORIES = [200, 202, 204, 206, 208, 210, 212, 214]',
        'EVALUATION_TRAJECTORIES = [201, 203, 205, 207, 209, 211, 213, 215]',
        'def candidate_center(',
        'def action_swap_delta(',
        'def donor_transfer_metrics(',
        'def grouped_kernel_ridge_cv(',
        'def trajectory_specs(',
        'def exact_dynamic_restore_test(',
        'def hook_identity_test(',
        'def construction_geometry_rows(',
        'def fit_and_freeze_subspaces(',
        'def intervention_specs(',
        'def run_record_interventions(',
        '"shuffled_fit"',
        '"wrong_state_donor"',
        '"common_mode"',
        '"full_activation_swap"',
        '"positive_control_only"',
        'FINITE_ACTION_CONTRAST_CAUSAL_MEDIATION',
        'FULL_SWAP_ONLY_NO_COMPRESSED_MEDIATION',
        'STOP_NO_CONSTRUCTION_ACTION_GEOMETRY',
        'stage17_action_contrast_result_bundle_',
    ]
    missing = [value for value in required if value not in joined]
    if missing:
        raise AssertionError(f"missing Stage 17 elements: {missing}")

    prohibited = [
        "torch.autograd",
        ".backward(",
        "functional.jvp",
        "register_full_backward_hook",
        "coordinate_moments",
        "fit_fixed_readers",
    ]
    present = [value for value in prohibited if value in joined]
    if present:
        raise AssertionError(f"prohibited Jacobian/reader machinery leaked in: {present}")

    source_identity = function_source(code_cells, "source_identity")
    for value in [
        '("numerical", EXPERIMENT_NUMERICAL_PATH)',
        "len(source_ref) != 40",
        'SOURCE_IDENTITY["confirmation_eligible"] = bool(matched)',
    ]:
        if value not in source_identity and value not in joined:
            raise AssertionError(f"source binding safeguard missing: {value}")

    model_loader = function_source(code_cells, "load_frozen_model")
    if "return model, preprocessor, predictor, blocks" not in model_loader:
        raise AssertionError("frozen model loader contract changed")
    carrier_forward = function_source(code_cells, "forward_with_carriers")
    for value in [
        "PREDICTOR_BLOCK_MODULES[block_index].register_forward_hook(hook)",
        'context["step"] != horizon - 1',
        'intervention["delta"]',
    ]:
        if value not in carrier_forward:
            raise AssertionError(f"finite carrier hook safeguard missing: {value}")

    swap = function_source(code_cells, "action_swap_delta")
    for value in [
        "residual[permutation] - residual",
        "(difference @ directions) @ directions.T",
    ]:
        if value not in swap:
            raise AssertionError(f"finite swap algebra missing: {value}")

    transfer = function_source(code_cells, "donor_transfer_metrics")
    for value in [
        "centered_edit - centered_base",
        "centered_base[permutation] - centered_base",
        '"mean_shift_ratio"',
    ]:
        if value not in transfer:
            raise AssertionError(f"donor-transfer estimand missing: {value}")

    subspace = function_source(code_cells, "fit_and_freeze_subspaces")
    for value in [
        "grouped_kernel_ridge_cv(",
        "shuffled_y[start:stop]",
        '"evaluation_trajectories_seen": []',
        '"full_activation_swap_is_positive_control_only": True',
        '"jacobians_computed": False',
    ]:
        if value not in subspace:
            raise AssertionError(f"construction-only subspace safeguard missing: {value}")

    specs = function_source(code_cells, "intervention_specs")
    for value in [
        "action_swap_delta(white, permutation, basis=primary_basis",
        "norm_match(shuffled, primary_full)",
        "norm_match(candidate, primary_full)",
        "matched_common_mode(primary_full",
        '"positive_control_only"',
    ]:
        if value not in specs:
            raise AssertionError(f"matched intervention control missing: {value}")

    evaluation_cell = next(
        source
        for source in code_cells
        if source.startswith(
            "# Open evaluation trajectories only after the layer and subspaces are frozen."
        )
    )
    ordered = [
        evaluation_cell.index("verify_executed_notebook_through("),
        evaluation_cell.index('SUBSPACE_DIR / "subspace_freeze.json"'),
        evaluation_cell.index("EVALUATION_RECORDS = realize_records"),
        evaluation_cell.index("extract_baselines(EVALUATION_RECORDS"),
    ]
    if ordered != sorted(ordered):
        raise AssertionError("evaluation data open before source/subspace freeze")

    decision = function_source(code_cells, "evaluate_causal_gate")
    for value in [
        "gain_random = primary - random_values",
        "gain_shuffled = primary - shuffled",
        "exact_positive_sign_test(gain_random)",
        "REQUIRED_POSITIVE_EVALUATION_TRAJECTORIES",
        '"dose_direction_pass"',
    ]:
        if value not in decision:
            raise AssertionError(f"causal decision safeguard missing: {value}")


def validate_builder_determinism():
    before = hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()
    subprocess.run([sys.executable, str(BUILDER)], check=True, capture_output=True)
    after = hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()
    if before != after:
        raise AssertionError("Stage 17 builder is not deterministic")


if __name__ == "__main__":
    validate_structure()
    validate_builder_determinism()
    print("Stage 17 finite action-contrast notebook validation passed")
