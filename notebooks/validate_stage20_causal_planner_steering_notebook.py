import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "20_causal_planner_steering.ipynb"
BUILDER = ROOT / "build_stage20_causal_planner_steering_notebook.py"


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
    assert before == after, "Stage 20 builder is not deterministic"

    notebook = json.loads(after)
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["metadata"]["colab"]["gpuType"] == "L4"
    assert len(notebook["cells"]) == 13
    code_cells = [source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 12
    for index, cell in enumerate(notebook["cells"]):
        assert cell["id"] == f"stage20-{index:02d}"
        assert not cell.get("outputs")
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            ast.parse(source(cell))

    config = code_cells[0]
    tree = ast.parse(config)
    assert assigned_value(tree, "PROTOCOL_ID") == "stage20-frozen-subspace-causal-planner-steering-v1"
    assert assigned_value(tree, "RUN_MODE") == "smoke"
    assert assigned_value(tree, "FIXED_BLOCK") == 4
    assert assigned_value(tree, "PRIMARY_STEERING_RANK") == 128
    assert assigned_value(tree, "SENSITIVITY_RANK") == 64
    assert assigned_value(tree, "TRANSFER_FAMILIES") == [
        "rotated_direction", "pulsed_equal_impulse"
    ]
    assert assigned_value(tree, "TARGET_BASELINE_RANKS") == [1, 2, 3]
    assert assigned_value(tree, "EVALUATION_TRAJECTORY_TARGET_PER_FAMILY") == 32
    assert assigned_value(tree, "INTERVENTION_FORWARDS_PER_RECORD") == 39
    assert assigned_value(tree, "RESULT_ROWS_PER_RECORD") == 54
    assert assigned_value(tree, "EXPECTED_STAGE18_SUBSPACE_SHA256") == (
        "2f9c496d54623a9062e465a18c70039acc18cb8a1cc2833a5f4ade162ca3f90b"
    )
    assert assigned_value(tree, "EXPECTED_STAGE19_DECISION_SHA256") == (
        "493fdf5c707189caea11043db7d208dbc38677dcf5881008e13bede87f40be9c"
    )
    for secret in ["STAGE20_RUN_MODE", "STAGE20_SOURCE_COMMIT", "STAGE20_RUN_NONCE"]:
        assert secret in config

    joined = "\n".join(code_cells)
    for required in [
        "def targeted_derangement(",
        "def select_near_frontier_targets(",
        "def planner_steering_metrics(",
        "def freeze_steering_targets(",
        "def intervention_specs(",
        "def evaluate_family(",
        "CONFIRMED_CAUSAL_PLANNER_STEERING_BOTH_FAMILIES",
        "PREDICTION_MEDIATOR_TRANSFER_WITHOUT_CONFIRMED_PLANNER_STEERING",
        "stage20_causal_planner_steering_result_bundle_",
        '"patched_forwards_generated"',
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
        assert prohibited not in joined, f"prohibited Stage 20 machinery: {prohibited}"

    target_freeze = function_source(code_cells, "freeze_steering_targets")
    for value in [
        "select_near_frontier_targets",
        "baseline_scores",
        '"uses_simulator_endpoint_costs": False',
        '"uses_intervention_outputs": False',
    ]:
        assert value in target_freeze
    for value in ["truth_costs", "endpoint_states", "intervention_path"]:
        assert value not in target_freeze

    model_cell = code_cells[8]
    target_freeze_call = "STEERING_TARGETS, STEERING_TARGET_FREEZE = freeze_steering_targets("
    assert model_cell.index("extract_baselines(ALL_EVALUATION_RECORDS") < model_cell.index(target_freeze_call)
    assert model_cell.index(target_freeze_call) < model_cell.index("EVALUATION_OPENED = True")
    artifact_cell = code_cells[7]
    assert "EXPECTED_STAGE18_SUBSPACE_SHA256" in artifact_cell
    assert "EXPECTED_STAGE19_DECISION_SHA256" in artifact_cell
    assert "validated_before_stage20_model_activations" in artifact_cell
    assert "stage20_subspace_refit\": False" in artifact_cell

    # Frozen pilot and smoke forward/row arithmetic.
    pilot_target_forwards = 2 + 1 + 1 + 4 + 1 + 1 + 1
    pilot_necessity_forwards = 1 + 1 + 4
    assert 3 * pilot_target_forwards + pilot_necessity_forwards == 39
    assert 3 + 3 * pilot_target_forwards + 3 * pilot_necessity_forwards == 54
    smoke_target_forwards = 1 + 1 + 1 + 1 + 1 + 1 + 1
    smoke_necessity_forwards = 1 + 1 + 1
    assert smoke_target_forwards + smoke_necessity_forwards == 10
    assert 1 + smoke_target_forwards + smoke_necessity_forwards == 11

    observed_digest = assigned_value(tree, "NOTEBOOK_PROTOCOL_SHA256")
    sources = [source(notebook["cells"][0])]
    sources.append(config.replace(observed_digest, "__PROTOCOL_DIGEST__", 1))
    sources.extend(code_cells[1:])
    expected_digest = hashlib.sha256(
        json.dumps(sources, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    assert observed_digest == expected_digest
    print("Stage 20 notebook validation passed")


if __name__ == "__main__":
    validate()
