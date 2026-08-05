import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "21_coherent_interface_and_heldout_utility.ipynb"
BUILDER = ROOT / "build_stage21_coherent_interface_utility_notebook.py"


def source(cell):
    return "".join(cell.get("source", []))


def assigned_value(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"missing assignment {name}")


def function_source(cells, name):
    for text in cells:
        tree = ast.parse(text)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
                return ast.get_source_segment(text, node)
    raise AssertionError(f"missing function {name}")


def validate():
    before = NOTEBOOK.read_bytes()
    subprocess.run([sys.executable, str(BUILDER)], check=True, capture_output=True)
    after = NOTEBOOK.read_bytes()
    assert before == after, "Stage 21 builder is not deterministic"

    notebook = json.loads(after)
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["metadata"]["colab"]["gpuType"] == "L4"
    assert len(notebook["cells"]) == 14
    code_cells = [source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 13
    for index, cell in enumerate(notebook["cells"]):
        assert cell["id"] == f"stage21-{index:02d}"
        assert not cell.get("outputs")
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            ast.parse(source(cell))

    config = code_cells[0]
    tree = ast.parse(config)
    assert assigned_value(tree, "PROTOCOL_ID") == (
        "stage21-coherent-handoff-heldout-causal-subspace-utility-v1"
    )
    assert assigned_value(tree, "RUN_MODE") == "smoke"
    assert assigned_value(tree, "SUBSPACE_BLOCK") == 4
    assert assigned_value(tree, "LAST_ACTION_CONDITIONED_BLOCK") == 5
    assert "ACTIVE_BLOCKS = [SUBSPACE_BLOCK, LAST_ACTION_CONDITIONED_BLOCK]" in config
    assert assigned_value(tree, "CORRECTION_RANK") == 128
    assert assigned_value(tree, "CORRECTION_RANDOM_DRAWS") == 4
    assert assigned_value(tree, "TARGET_BASELINE_RANKS") == [1, 2, 3]
    assert assigned_value(tree, "INTERFACE_PATCHED_FORWARDS_PER_EVALUATION_RECORD") == 9
    assert assigned_value(tree, "INTERFACE_ROWS_PER_EVALUATION_RECORD") == 15
    assert assigned_value(tree, "UTILITY_ROWS_PER_EVALUATION_RECORD") == 7
    assert assigned_value(tree, "SPLIT_TARGETS_PER_FAMILY") == {
        "construction": 32,
        "calibration": 16,
        "evaluation": 32,
    }
    assert assigned_value(tree, "EXPECTED_STAGE20_DECISION_SHA256") == (
        "57e6ec6ab60415d782bd37e773842b96a8fef596ead111df33b7b816c83d601e"
    )
    for secret in ["STAGE21_RUN_MODE", "STAGE21_SOURCE_COMMIT", "STAGE21_RUN_NONCE"]:
        assert secret in config

    joined = "\n".join(code_cells)
    for required in [
        "def subspace_coordinates(",
        "def select_ridge_on_calibration(",
        "def counterfactual_interface_metrics(",
        "def freeze_evaluation_predictions_and_choices(",
        "def run_interface_record(",
        "def run_heldout_utility(",
        "CONFIRMED_COHERENT_HANDOFF_AND_CAUSAL_SUBSPACE_UTILITY_BOTH_FAMILIES",
        "COHERENT_HANDOFF_WITHOUT_CAUSAL_SUBSPACE_UTILITY",
        "stage21_coherent_utility_result_bundle_",
    ]:
        assert required in joined, f"missing {required}"
    for prohibited in [
        "torch.autograd",
        ".backward(",
        "torch.linalg.svd",
        "fit_dual_ridge_basis",
        "fit_and_freeze_subspaces",
        "make_truth_montage",
        "make_plots",
        ".savefig(",
    ]:
        assert prohibited not in joined, f"prohibited Stage 21 machinery: {prohibited}"

    freeze_function = function_source(code_cells, "freeze_evaluation_predictions_and_choices")
    assert "true_pose_for_record" not in freeze_function
    assert "endpoint_states" not in freeze_function
    assert "score_sha256" in freeze_function
    assert "selected_action" in freeze_function
    truth_function = function_source(code_cells, "true_pose_for_record")
    assert "EVALUATION_ENDPOINT_TRUTH_OPENED" in truth_function
    assert "sealed until all evaluation choices are frozen" in truth_function

    correction_cell = code_cells[9]
    assert correction_cell.index("EVALUATION_ENDPOINT_TRUTH_OPENED = False") < correction_cell.index(
        "freeze_evaluation_predictions_and_choices()"
    )
    run_cell = code_cells[10]
    assert run_cell.index("INTERFACE_ROWS = run_all_interfaces()") < run_cell.index(
        "EVALUATION_ENDPOINT_TRUTH_OPENED = True"
    )
    assert run_cell.index("EVALUATION_ENDPOINT_TRUTH_OPENED = True") < run_cell.index(
        "UTILITY_ROWS = run_heldout_utility()"
    )
    artifact_cell = code_cells[7]
    assert "EXPECTED_STAGE18_SUBSPACE_SHA256" in artifact_cell
    assert "EXPECTED_STAGE19_DECISION_SHA256" in artifact_cell
    assert "EXPECTED_STAGE20_DECISION_SHA256" in artifact_cell
    assert "validated_before_stage21_model_activations" in artifact_cell
    decision_cell = code_cells[11]
    assert 'verify_executed_notebook_through(' in decision_cell
    assert '"# Apply Stage 21 coherent-handoff and held-out utility gates."' in decision_cell

    # Frozen pilot workload.
    assert 32 * 2 + 16 * 2 + 32 * 2 == 160
    assert 32 * 2 * 9 == 576
    assert 32 * 2 * 15 == 960
    assert 32 * 2 * 7 == 448
    assert (48 + 32 + 48) * 2 == 256

    observed_digest = assigned_value(tree, "NOTEBOOK_PROTOCOL_SHA256")
    sources = [source(notebook["cells"][0])]
    sources.append(config.replace(observed_digest, "__PROTOCOL_DIGEST__", 1))
    sources.extend(code_cells[1:])
    expected_digest = hashlib.sha256(
        json.dumps(sources, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    assert observed_digest == expected_digest
    print("Stage 21 notebook validation passed")


if __name__ == "__main__":
    validate()
