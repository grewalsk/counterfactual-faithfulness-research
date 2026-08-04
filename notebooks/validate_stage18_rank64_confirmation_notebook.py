import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "18_rank64_action_contrast_confirmation.ipynb"
BUILDER = ROOT / "build_stage18_rank64_confirmation_notebook.py"


def payload():
    return json.loads(NOTEBOOK.read_text())


def function_source(code_cells, name):
    for source in code_cells:
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
                recovered = ast.get_source_segment(source, node)
                if recovered is None:
                    raise AssertionError(f"could not recover {name}")
                return recovered
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
        if cell.get("id") != f"stage18-{index:02d}":
            raise AssertionError(f"bad Stage 18 cell id at {index}")
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
        "STAGE18_RUN_MODE",
        "STAGE18_SOURCE_COMMIT",
        "STAGE18_RUN_NONCE",
        ").strip().lower()",
        'f"received {RUN_MODE!r}"',
        'PROTOCOL_ID = "stage18-rank64-action-contrast-confirmation-v1"',
        "FRESH_RUN_REQUIRED = True",
        "FIXED_BLOCK = 4",
        "CONSTRUCTION_POOL_TRAJECTORIES = list(range(300, 348))",
        "EVALUATION_POOL_TRAJECTORIES = list(range(400, 464))",
        "CONSTRUCTION_TRAJECTORY_TARGET = 24",
        "EVALUATION_TRAJECTORY_TARGET = 32",
        "OUTPUT_SKETCH_DIM = 256",
        "PRIMARY_RANK = 64",
        "SENSITIVITY_RANKS = [16, 32, 64, 96, 128]",
        "MAX_SUBSPACE_RANK = 128",
        "CAUSAL_RANDOM_DRAWS = 4",
        "CAUSAL_DOSES = [-0.5, 0.25, 0.5, 1.0]",
        "INTERVENTION_FORWARDS_PER_RECORD = 42",
        "if len(PREDICTOR_BLOCK_MODULES) != 6:",
        "def candidate_action_bank(",
        "def physical_diversity_metrics(",
        "def projection_ablation_delta(",
        "def action_contrast_energy_metrics(",
        "def exact_dynamic_restore_test(",
        "def truth_eligibility(",
        "def select_records(",
        "def fixed_block_geometry_rows(",
        "def fit_and_freeze_subspaces(",
        "def intervention_specs(",
        "def evaluate_confirmation_gate(",
        "def fresh_run_certificate(",
        '"matched_shuffled_control"',
        '"empirical_span_random_control"',
        '"state_specificity_control"',
        '"matched_common_mode_control"',
        '"necessity"',
        '"positive_control_only"',
        "CONFIRMED_BIDIRECTIONAL_RANK64_MEDIATOR",
        "SUFFICIENCY_ONLY_RANK64_TRANSFER",
        "FULL_SWAP_ONLY_NO_CONFIRMED_RANK64_MEDIATOR",
        "NO_ACTION_CONTRAST_CAUSAL_SIGNAL",
        "STOP_NO_FIXED_BLOCK_ACTION_GEOMETRY",
        "raw_shard_manifest.json",
        "stage18_truth_montage.png",
        "stage18_rank64_result_bundle_",
    ]
    missing = [value for value in required if value not in joined]
    if missing:
        raise AssertionError(f"missing Stage 18 elements: {missing}")

    prohibited = [
        "torch.autograd",
        ".backward(",
        "functional.jvp",
        "register_full_backward_hook",
        "coordinate_moments",
        "fit_fixed_readers",
        "SELECTED_BLOCK",
        "len(PREDICTOR_BLOCKS)",
    ]
    present = [value for value in prohibited if value in joined]
    if present:
        raise AssertionError(f"prohibited Jacobian/reader/layer-selection machinery leaked in: {present}")

    setup = code_cells[2]
    for value in [
        "OUT_PREEXISTED = OUT.exists()",
        'RUN_MODE == "pilot" and FRESH_RUN_REQUIRED and OUT_PREEXISTED',
        "fresh pilot output already exists",
        'PROVENANCE_COUNTS = {"truth_generated": 0',
    ]:
        if value not in setup:
            raise AssertionError(f"fresh-run setup safeguard missing: {value}")

    source_identity = function_source(code_cells, "source_identity")
    for value in [
        '("numerical", EXPERIMENT_NUMERICAL_PATH)',
        "len(source_ref) != 40",
        'SOURCE_IDENTITY["confirmation_eligible"] = bool(matched)',
    ]:
        if value not in source_identity and value not in joined:
            raise AssertionError(f"source binding safeguard missing: {value}")

    eligibility = function_source(code_cells, "truth_eligibility")
    for value in [
        "decoded_task_cost",
        "MIN_ELIGIBLE_COST_SPREAD",
        "MIN_ELIGIBLE_NON_TIED_PAIR_FRACTION",
        "MIN_ELIGIBLE_CONTACT_BRANCHES",
        '"action_sha256"',
        '"endpoint_state_sha256"',
    ]:
        if value not in eligibility:
            raise AssertionError(f"model-blind physical eligibility safeguard missing: {value}")

    physical_cell = next(
        source for source in code_cells if source.startswith(
            "# Generate and select physical truth before loading any model or encoder."
        )
    )
    for value in [
        'if "MODEL" in globals()',
        '"selection_completed_before_model_load": True',
        '"selection_used_only_simulator_truth": True',
        "CONSTRUCTION_POOL_SPECS, ACTIVE_CONSTRUCTION_TARGET",
        "EVALUATION_POOL_SPECS, ACTIVE_EVALUATION_TARGET",
    ]:
        if value not in physical_cell:
            raise AssertionError(f"pre-model selection safeguard missing: {value}")
    if "load_frozen_model(" in physical_cell:
        raise AssertionError("model loader appears in physical selection cell")

    subspace = function_source(code_cells, "fit_and_freeze_subspaces")
    for value in [
        "theoretical_rank_ceiling",
        "grouped_kernel_ridge_cv(",
        "shuffled_y[start:stop]",
        "random_basis_gpu(",
        '"evaluation_model_activations_seen": []',
        '"evaluation_simulator_truth_used_only_for_frozen_eligibility": True',
        '"full_activation_swap_is_positive_control_only": True',
        '"jacobians_computed": False',
    ]:
        if value not in subspace:
            raise AssertionError(f"construction-only subspace safeguard missing: {value}")

    evaluation_cell = next(
        source for source in code_cells if source.startswith(
            "# Open evaluation model activations only after the fixed hypothesis and subspaces are frozen."
        )
    )
    ordered = [
        evaluation_cell.index("verify_executed_notebook_through("),
        evaluation_cell.index('SUBSPACE_DIR / "subspace_freeze.json"'),
        evaluation_cell.index("extract_baselines(EVALUATION_RECORDS"),
    ]
    if ordered != sorted(ordered):
        raise AssertionError("evaluation model activations open before source/subspace freeze")

    ablation = function_source(code_cells, "projection_ablation_delta")
    for value in [
        "residual = candidate_center(flat)",
        "projected = (residual @ directions) @ directions.T",
        "-float(dose) * projected",
    ]:
        if value not in ablation:
            raise AssertionError(f"necessity algebra missing: {value}")

    specs = function_source(code_cells, "intervention_specs")
    for value in [
        "basis=primary_basis, dose=1.0",
        "projection_ablation_delta(white, primary_basis, dose=1.0)",
        "norm_match(shuffled, learned)",
        "norm_match(random_delta, learned)",
        "norm_match(shuffled_ablation, primary_ablation)",
        "norm_match(random_ablation, primary_ablation)",
        'f"wrong_state_r{ACTIVE_PRIMARY_RANK:03d}"',
        '"positive_control_only"',
        "INTERVENTION_FORWARDS_PER_RECORD",
    ]:
        if value not in specs:
            raise AssertionError(f"matched bidirectional intervention missing: {value}")

    gate = function_source(code_cells, "evaluate_confirmation_gate")
    for value in [
        "gain_random = primary - random_values",
        "gain_shuffled = primary - shuffled",
        "necessity_gain_random = necessity - necessity_random",
        "necessity_gain_shuffled = necessity - necessity_shuffled",
        "exact_positive_sign_test(gain_random)",
        "exact_positive_sign_test(necessity_gain_random)",
        "REQUIRED_POSITIVE_EVALUATION_TRAJECTORIES",
        "REQUIRED_POSITIVE_NECESSITY_TRAJECTORIES",
        '"bidirectional_gate_pass"',
    ]:
        if value not in gate:
            raise AssertionError(f"confirmation gate safeguard missing: {value}")

    certificate = function_source(code_cells, "fresh_run_certificate")
    for value in [
        '"truth_generated"',
        '"baseline_generated"',
        '"intervention_generated"',
        '"cache_hits": 0',
        "not OUT_PREEXISTED and PROVENANCE_COUNTS == expected",
    ]:
        if value not in certificate:
            raise AssertionError(f"fresh provenance certificate missing: {value}")


def validate_builder_determinism():
    before = hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()
    subprocess.run([sys.executable, str(BUILDER)], check=True, capture_output=True)
    after = hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()
    if before != after:
        raise AssertionError("Stage 18 builder is not deterministic")


if __name__ == "__main__":
    validate_structure()
    validate_builder_determinism()
    print("Stage 18 rank-64 confirmation notebook validation passed")
