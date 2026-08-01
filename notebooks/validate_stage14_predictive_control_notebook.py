import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "14_predictive_control_j_bundle_pilot.ipynb"
BUILDER = ROOT / "build_stage14_predictive_control_notebook.py"


def notebook_payload():
    return json.loads(NOTEBOOK.read_text())


def function_source(code_cells, name):
    for source in code_cells:
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
                segment = ast.get_source_segment(source, node)
                if segment is None:
                    raise AssertionError(f"could not recover {name}")
                return segment
    raise AssertionError(f"missing function {name}")


def validate_structure():
    payload = notebook_payload()
    if len(payload["cells"]) != 16:
        raise AssertionError(f"expected 16 cells, found {len(payload['cells'])}")
    code_cells = [
        "".join(cell.get("source", []))
        for cell in payload["cells"]
        if cell["cell_type"] == "code"
    ]
    if len(code_cells) != 15:
        raise AssertionError(f"expected 15 code cells, found {len(code_cells)}")
    for index, source in enumerate(code_cells):
        try:
            ast.parse(source)
        except SyntaxError as error:
            raise AssertionError(f"invalid Python in code cell {index}: {error}") from error
    for index, cell in enumerate(payload["cells"]):
        if cell.get("id") != f"stage14-{index:02d}":
            raise AssertionError(f"bad cell id at {index}")
        if cell.get("outputs"):
            raise AssertionError(f"stale outputs in cell {index}")
        if cell["cell_type"] == "code" and cell.get("execution_count") is not None:
            raise AssertionError(f"stale execution count in cell {index}")
    if payload["metadata"].get("accelerator") != "GPU":
        raise AssertionError("notebook does not request a GPU")
    if payload["metadata"]["colab"]["name"] != NOTEBOOK.name:
        raise AssertionError("Colab name does not match notebook")

    joined = "\n".join(code_cells)
    required = [
        'RUN_MODE = "smoke"',
        'NOTEBOOK_PROTOCOL_SHA256 = "',
        "TARGET_STEPS = HORIZONS",
        "EXPECTED_CARRIER_CHANNELS = 400",
        "TRAIN_QUERY_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8)]",
        "TEST_QUERY_PAIRS = [(9, 10), (11, 12)]",
        "CONSTRUCTION_CLUSTER_INDICES",
        "EVALUATION_CLUSTER_INDICES",
        "def scan_state_carrier(",
        "output.detach().requires_grad_(True)",
        "torch.autograd.grad(",
        "def exact_action_jacobian(",
        "torch.autograd.functional.jvp(",
        "def balanced_modes(",
        "def fit_sparse_frame(",
        'transform_algorithm="omp"',
        "def null_components(",
        "def adjoint_chain_smoke(",
        "def jvp_epsilon_linearity_check(",
        "def run_causal_mediation(",
        '"sufficiency"',
        '"necessity"',
        "def norm_match_native(",
        "def calibrate_nondegeneracy_floors(",
        "def aggregate_null_test(",
        "NULL_DRAWS = 256",
        "CAUSAL_NULL_DRAWS = 32",
        "def freeze_before_evaluation(",
        "def open_evaluation_after_freeze(",
        "def verify_executed_notebook_through(",
        '"# Apply the frozen claim ladder mechanically and plot compact diagnostics."',
        '"# Package compact audit evidence separately from recomputable large shards."',
        '"hankel_energy_per_cell"',
        '"train_query_separations"',
        '"train_train_reconstruction"',
        '"positive_task_gains"',
        'env.set_task_goal(requested_goal.copy())',
        "def gauge_identity_checks(",
        "def temporal_extension(",
        "interaction_types",
        'REQUIRED_INTERACTION_STRATA = ("free", "contact")',
        "set(qualified_interactions) == set(REQUIRED_INTERACTION_STRATA)",
        "for task_id in sorted(stable_task_ids)",
        'CALIBRATED_FLOORS["causal_denominator_abs"]',
        "JEPA_NATIVE_PREDICTIVE_CONTROL_BUNDLE_CANDIDATE",
        '"workspace_claim_authorized": False',
        '"j_space_claim_authorized": False',
        '"reader_scope": "state-specific oracle target-aligned assay"',
        'EVIDENCE_DIR / "evidence_manifest.json"',
        '"full_manifest.json"',
        '"files": manifest_rows(compact_root)',
        '"files describes every archive member except compact_manifest.json itself"',
        "stage14_predictive_control_result_bundle",
    ]
    missing = [value for value in required if value not in joined]
    if missing:
        raise AssertionError(f"missing Stage 14 elements: {missing}")
    prohibited = [
        "fit_outcome_dictionary",
        "pca_axes",
        "PROTOTYPE_AXES",
        "torch.autograd.functional.jacobian",
        "torch.func.jacrev",
        "optimizer.step(",
        ".backward(",
        "STOP_NO_COMPACT_OUTCOME_DICTIONARY",
        "query_mediation",
        "DIRTY_PATCH",
        "NO_REUSABLE_PREDICTIVE_CONTROL_INTERFACE",
        "SPARSE_PREDICTIVE_FRAME_NOT_CAUSAL",
        "manifest_rows(OUT, excluded_roots=compact_exclusions)",
    ]
    present = [value for value in prohibited if value in joined]
    if present:
        raise AssertionError(f"old/generic JOW machinery leaked in: {present}")

    scan = function_source(code_cells, "scan_state_carrier")
    if "predictions[READ_BRANCH]" not in scan:
        raise AssertionError("query VJPs are not anchored at the frozen read branch")
    if ".mean(dim=0)" in scan or ".mean(axis=0)" in scan:
        raise AssertionError("scan silently averages state/action carrier axes")
    if "construction_branches = [READ_BRANCH, *action_indices]" not in scan:
        raise AssertionError("channel metric still sees held-out action branches")
    jvp = function_source(code_cells, "exact_action_jacobian")
    if "torch.autograd.functional.jvp(" not in jvp:
        raise AssertionError("exact write map is not computed by JVP")
    if "2.0 * JVP_EPSILON" not in jvp:
        raise AssertionError("centered finite-difference fallback is absent")
    carrier = function_source(code_cells, "layer_tokens_full")
    if ":384" in carrier or "predictor_embed_dim" in carrier:
        raise AssertionError("carrier still truncates the 400-channel block output")
    shard = function_source(code_cells, "build_write_read_shard")
    if "metric_invariance_error" not in shard:
        raise AssertionError("hidden metric does not verify G@B invariance")
    causal = function_source(code_cells, "run_causal_mediation")
    for control in [
        '"linear_full"',
        '"dense_balanced"',
        '"natural_activation"',
        '"sparse_complement"',
        'f"null_{index:02d}"',
    ]:
        if control not in causal:
            raise AssertionError(f"causal control missing: {control}")
    for safeguard in [
        '"sufficiency"',
        '"necessity"',
        "natural_write = native_to_whitened_patch",
        "norm_match_native",
        '"target_pair_separation"',
        '"output_displacement_cosine"',
    ]:
        if safeguard not in causal:
            raise AssertionError(f"causal safeguard missing: {safeguard}")
    gate = function_source(code_cells, "causal_gate")
    for safeguard in [
        "-0.75 * full_dose <= negative_dose <= -0.25 * full_dose",
        "len(null_statistics) != CAUSAL_NULL_DRAWS",
        '"all_causal_safeguards_tasks"',
        "safeguard_tasks >= required_positive",
    ]:
        if safeguard not in gate:
            raise AssertionError(f"causal promotion gate missing: {safeguard}")
    source = function_source(code_cells, "source_identity")
    if 'len(source_ref) != 40' not in source:
        raise AssertionError("source binding does not require an immutable commit")
    verifier = function_source(code_cells, "verify_executed_notebook_through")
    if 'history[-len(expected) :] == expected' not in verifier:
        raise AssertionError("executed notebook prefix is not source-bound")
    if "source.startswith(cell_header)" not in verifier:
        raise AssertionError("source binding uses an ambiguous marker search")
    sparse_indices = [
        index for index, source in enumerate(code_cells)
        if source.startswith(
            "# Learn one sparse frame on construction modes; evaluate unopened task clusters."
        )
    ]
    if sparse_indices != [10]:
        raise AssertionError(f"ambiguous sparse-frame cell marker: {sparse_indices}")
    goal = function_source(code_cells, "reset_environment")
    if 'env.set_task_goal(requested_goal.copy())' not in goal:
        raise AssertionError("PushT task goal is metadata-only")
    save_scan = function_source(code_cells, "save_scan")
    if "np.float16" in save_scan:
        raise AssertionError("carrier scan still serializes evidence in float16")
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
        raise AssertionError("Stage 14 builder is not deterministic")


if __name__ == "__main__":
    validate_structure()
    validate_builder_reproducibility()
    print("Stage 14 predictive-control notebook validation passed")
