import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "15_longitudinal_predictive_control_bundle.ipynb"
BUILDER = ROOT / "build_stage15_longitudinal_bundle_notebook.py"


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
        if cell.get("id") != f"stage15-{index:02d}":
            raise AssertionError(f"bad Stage 15 cell id at {index}")
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
        'STAGE15_RUN_MODE=pilot',
        'STAGE15_SOURCE_COMMIT=<full 40-hex commit>',
        'PROTOCOL_ID = "stage15-fixed-reader-longitudinal-bundle-v1"',
        'READER_LABELS = [',
        '"agent_x", "agent_y", "block_x", "block_y", "block_sin", "block_cos"',
        'READER_PROJECTION_SEEDS = [15161, 15173, 15187]',
        'CONSTRUCTION_TRAJECTORIES = [0, 2, 4, 6]',
        'EVALUATION_TRAJECTORIES = [1, 3, 5, 7]',
        'LONGITUDINAL_SAVE_STEPS = [0, 5, 10, 15, 20]',
        'PREDICTOR_BLOCKS = [0, 1, 2, 3, 4, 5]',
        'def physical_reader_targets(',
        'def temporal_action_basis(',
        'def grouped_ridge_cv(',
        'def support_matched_random(',
        'def realize_trajectory(',
        'def dynamic_state_from_environment(',
        'def reset_dynamic_environment(',
        'def rollout_dynamic_branch(',
        'def exact_dynamic_restore_test(',
        'def candidate_action_bank(',
        'def fit_fixed_readers(',
        'def decode_fixed_physical(',
        'def common_action_basis(',
        'def fit_channel_metrics(',
        'def fixed_reader_gradients_all_blocks(',
        'def exact_action_tangent_jacobians_all_blocks(',
        'def build_operator_shard(',
        'def permutation_geometry_summary(',
        'def causal_transport_rows(',
        '"support_matched_null"',
        '"covariance_shaped_null"',
        '"time_shuffled"',
        'REQUIRED_POSITIVE_CAUSAL_TRAJECTORIES = 4',
        'MAX_ESTIMATED_TOTAL_MINUTES = 150.0',
        'automatic credit guard stopped Stage 15',
        'positive_trajectories >= required_positive',
        '"local_positive"',
        '"transported_neighbor"',
        'LONGITUDINAL_CAUSAL_BUNDLE_SUPPORTED',
        'SMOOTH_NONCAUSAL_FIELD',
        'LOCAL_CAUSAL_ONLY',
        'NO_LONGITUDINAL_BUNDLE_EVIDENCE',
        'stage15_longitudinal_bundle_result_bundle',
    ]
    missing = [value for value in required if value not in joined]
    if missing:
        raise AssertionError(f"missing Stage 15 elements: {missing}")
    prohibited = [
        "TRAIN_QUERY_PAIRS",
        "TEST_QUERY_PAIRS",
        "query_direction",
        "query_separation",
        "fit_sparse_frame",
        "state-specific oracle target-aligned assay",
        "global outcome dictionary",
        "workspace_claim_authorized",
        "j_space_claim_authorized",
    ]
    present = [value for value in prohibited if value in joined]
    if present:
        raise AssertionError(f"prohibited Stage 14 machinery leaked in: {present}")

    dynamic_restore = function_source(code_cells, "reset_dynamic_environment")
    for value in [
        "environment.block.velocity",
        "environment.block.angular_velocity",
        "dynamic_state_from_environment(environment)",
        "atol=1e-12",
    ]:
        if value not in dynamic_restore:
            raise AssertionError(f"full dynamic restore safeguard missing: {value}")
    restore_test = function_source(code_cells, "exact_dynamic_restore_test")
    if '"one_step_continuation_exact"' not in restore_test:
        raise AssertionError("dynamic restore does not test one-step continuation")

    source_identity = function_source(code_cells, "source_identity")
    if 'len(source_ref) != 40' not in source_identity:
        raise AssertionError("source binding does not require a full commit")
    verifier = function_source(code_cells, "verify_executed_notebook_through")
    for value in [
        'history[-len(expected) :] == expected',
        'source.startswith(cell_header)',
    ]:
        if value not in verifier:
            raise AssertionError(f"source-prefix safeguard missing: {value}")

    fit_readers = function_source(code_cells, "fit_fixed_readers")
    for value in [
        "grouped_ridge_cv(",
        "target_mean",
        "target_scale",
        '"evaluation_trajectories_seen": []',
        '"frozen_before_evaluation": True',
    ]:
        if value not in fit_readers:
            raise AssertionError(f"reader-freeze safeguard missing: {value}")
    model_loader = function_source(code_cells, "load_frozen_model")
    if "return model, preprocessor, predictor, blocks" not in model_loader:
        raise AssertionError("frozen-model loader return contract changed")
    reader_cell = next(
        source
        for source in code_cells
        if source.startswith(
            "# Fit and freeze fixed physical readers before opening evaluation trajectories."
        )
    )
    if (
        "MODEL, PREPROCESSOR, PREDICTOR, PREDICTOR_BLOCK_MODULES = "
        "load_frozen_model()"
    ) not in reader_cell:
        raise AssertionError("frozen-model loader return contract is not unpacked exactly")
    ordered = [
        reader_cell.index("READER_FREEZE = fit_fixed_readers"),
        reader_cell.index("verify_executed_notebook_through("),
        reader_cell.index("EVALUATION_RECORDS = realize_records(EVALUATION_SPECS)"),
        reader_cell.index("evaluate_fixed_readers(\n            EVALUATION_RECORDS"),
    ]
    if ordered != sorted(ordered):
        raise AssertionError("evaluation is opened before reader/source freeze")

    action_basis = function_source(code_cells, "common_action_basis")
    for value in ["np.linalg.qr", "plus_indices", "ACTION_BASIS_DIM"]:
        if value not in action_basis:
            raise AssertionError(f"common action-basis safeguard missing: {value}")
    extraction = function_source(
        code_cells, "exact_action_tangent_jacobians_all_blocks"
    )
    for value in [
        "capture_blocks=ACTIVE_BLOCKS",
        "torch.autograd.functional.jvp(",
        "for block in ACTIVE_BLOCKS",
        "for direction_index in range(ACTION_BASIS_DIM)",
        "action_basis[:, direction_index]",
    ]:
        if value not in extraction:
            raise AssertionError(f"all-layer JVP extraction missing: {value}")
    shard = function_source(code_cells, "build_operator_shard")
    for value in [
        "g @ b",
        "native_k",
        "MAX_METRIC_INVARIANCE_ERROR",
        "G=np.stack(all_g).astype(np.float32)",
        "B_action=np.stack(all_b).astype(np.float32)",
        "K=np.stack(all_k).astype(np.float64)",
    ]:
        if value not in shard:
            raise AssertionError(f"operator shard safeguard missing: {value}")

    causal = function_source(code_cells, "causal_transport_rows")
    for value in [
        "native_norm_match_whitened(",
        "support_matched_random(",
        '"support_matched_null"',
        '"covariance_shaped_null"',
        '"time_shuffled"',
        "zero_edit_check(destination)",
    ]:
        if value not in causal:
            raise AssertionError(f"causal control missing: {value}")
    response = function_source(code_cells, "causal_direction_responses")
    for value in [
        "CAUSAL_DOSE * delta",
        "-CAUSAL_DOSE * delta",
        "decode_fixed_physical",
        "linearity_cosine",
        "batched_actions",
    ]:
        if value not in response:
            raise AssertionError(f"causal response safeguard missing: {value}")

    decision_cell = next(
        source
        for source in code_cells
        if source.startswith(
            "# Apply the frozen Stage 15 claim ladder and render compact diagnostics."
        )
    )
    if 'elif RUN_MODE == "smoke":\n    DECISION = "SMOKE_ONLY"' not in decision_cell:
        raise AssertionError("smoke can authorize a scientific decision")
    for value in [
        '"global_j_space_authorized": False',
        '"cross_model_generality_authorized": False',
    ]:
        if value not in decision_cell:
            raise AssertionError(f"claim boundary missing: {value}")


def validate_builder_determinism():
    before = hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()
    subprocess.run([sys.executable, str(BUILDER)], check=True, capture_output=True)
    after = hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()
    if before != after:
        raise AssertionError("Stage 15 builder is not deterministic")


if __name__ == "__main__":
    validate_structure()
    validate_builder_determinism()
    print("Stage 15 longitudinal bundle notebook validation passed")
