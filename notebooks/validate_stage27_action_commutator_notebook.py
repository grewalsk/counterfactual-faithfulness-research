import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "27_causal_action_commutator.ipynb"
BUILDER = ROOT / "build_stage27_action_commutator_notebook.py"


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
    assert before == after, "Stage 27 builder is not deterministic"

    notebook = json.loads(after)
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["metadata"]["colab"]["gpuType"] == "L4"
    assert len(notebook["cells"]) == 13
    code_cells = [source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 12
    for index, cell in enumerate(notebook["cells"]):
        assert cell["id"] == f"stage27-{index:02d}"
        assert not cell.get("outputs")
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            ast.parse(source(cell))

    config = code_cells[0]
    tree = ast.parse(config)
    assert assigned_value(tree, "PROTOCOL_ID") == "stage27-causal-finite-action-commutator-v1"
    assert assigned_value(tree, "RUN_MODE") == "pilot"
    assert assigned_value(tree, "EXPERIMENT_SOURCE_REF") == "codex/stage27-causal-action-commutator"
    assert assigned_value(tree, "EVALUATION_TRAJECTORY_TARGET") == 40
    assert assigned_value(tree, "ACTION_PAIR_COUNT") == 6
    assert "ACTIONS_PER_STATE = 2 * ACTION_PAIR_COUNT" in config
    assert assigned_value(tree, "PRIMARY_RANK") == 128
    assert assigned_value(tree, "SENSITIVITY_RANKS") == [64, 128]
    assert assigned_value(tree, "INTERVENTION_FORWARDS_PER_RECORD") == 30
    assert "token_hex(4)" in config
    assert '_colab_userdata.get("HF_TOKEN")' in config
    for forbidden in ["STAGE27_RUN_MODE", "STAGE27_SOURCE_COMMIT", "STAGE27_RUN_NONCE"]:
        assert forbidden not in "\n".join(code_cells)

    joined = "\n".join(code_cells)
    for required in [
        "def ordered_pulse_bank(",
        "def paired_swap_delta(",
        "def paired_ablation_delta(",
        "def commutator_alignment_metrics(",
        "def exact_dynamic_restore_test(",
        "def evaluate_model_physical_commutators(",
        "def physical_commutator_gate(",
        "def model_commutator_gate(",
        "def causal_commutator_gate(",
        "CAUSAL_NONCOMMUTATIVE_ACTION_DYNAMICS_SUPPORTED",
        "stage27_action_commutator_result_bundle_",
    ]:
        assert required in joined, f"missing {required}"
    for prohibited in [
        "torch.autograd", ".backward(", "torch.func.jvp", "torch.func.vjp",
        "jacrev", "jacfwd",
    ]:
        assert prohibited not in joined, f"prohibited Stage 27 machinery: {prohibited}"

    action_design = function_source(code_cells, "ordered_pulse_bank")
    assert "left_then_right[:pulse_steps]" in action_design
    assert "right_then_left[:pulse_steps]" in action_design
    assert "equal integrated impulse" in action_design
    selection = code_cells[5]
    assert '"model_loaded": bool("MODEL" in globals())' in selection
    assert '"selection_uses_model_outputs": False' in selection
    assert '"physical_commutator_magnitude_used_for_selection": False' in selection
    truth = code_cells[6]
    assert "full dynamic restore test failed" in truth
    assert "MODEL" in truth and "physical eligibility selection" in truth

    upstream = code_cells[7]
    assert "unique_matching_path" in upstream
    assert "EXPECTED_STAGE18_SUBSPACE_SHA256" in upstream
    assert "EXPECTED_STAGE19_SOURCE_COMMIT" in upstream
    assert "auto_located_without_stage27_secret" in upstream
    assert "verify_executed_notebook_through(" in upstream

    edits = function_source(code_cells, "intervention_specs")
    for required in [
        "paired_swap_delta(", "paired_ablation_delta(",
        'subspaces["primary_basis"]', 'subspaces["shuffled_basis"]',
        'subspaces[f"random_basis_', "wrong_state_order_delta(",
        "matched_common_mode(", '"full_activation_swap"',
    ]:
        assert required in edits
    decision = code_cells[10]
    assert 'PHYSICAL_GATE["passed"]' in decision
    assert 'MODEL_GATE["passed"]' in decision
    assert 'CAUSAL_GATE["passed"]' in decision
    assert '"infinitesimal_lie_bracket_established": False' in decision

    observed_digest = assigned_value(tree, "NOTEBOOK_PROTOCOL_SHA256")
    sources = [source(notebook["cells"][0])]
    sources.append(config.replace(observed_digest, "__PROTOCOL_DIGEST__", 1))
    sources.extend(code_cells[1:])
    expected_digest = hashlib.sha256(
        json.dumps(sources, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    assert observed_digest == expected_digest
    print("Stage 27 notebook validation passed")


if __name__ == "__main__":
    validate()
