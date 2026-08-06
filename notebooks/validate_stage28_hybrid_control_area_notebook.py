import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "28_hybrid_control_area_law.ipynb"
BUILDER = ROOT / "build_stage28_hybrid_control_area_notebook.py"


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
    environment = dict(os.environ)
    subprocess.run(
        [sys.executable, str(BUILDER)], check=True, capture_output=True, env=environment
    )
    after = NOTEBOOK.read_bytes()
    assert before == after, "Stage 28 builder is not deterministic"

    notebook = json.loads(after)
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["metadata"]["colab"]["gpuType"] == "L4"
    assert len(notebook["cells"]) == 13
    code_cells = [source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 12
    for index, cell in enumerate(notebook["cells"]):
        assert cell["id"] == f"stage28-{index:02d}"
        assert not cell.get("outputs")
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            ast.parse(source(cell))

    config = code_cells[0]
    tree = ast.parse(config)
    assert assigned_value(tree, "PROTOCOL_ID") == "stage28-hybrid-control-area-law-v1"
    assert assigned_value(tree, "RUN_MODE") == "pilot"
    assert assigned_value(tree, "EXPERIMENT_SOURCE_REF") == "codex/stage28-hybrid-control-area"
    assert assigned_value(tree, "EXPECTED_STAGE27_SOURCE_COMMIT") == "78d241fce761babdc4c11c51bfc5758867ecea07"
    assert assigned_value(tree, "EVALUATION_TRAJECTORY_TARGET") == 36
    assert assigned_value(tree, "SCHEDULE_COUNT") == 6
    assert assigned_value(tree, "MAGNITUDE_COUNT") == 4
    assert "ACTIONS_PER_STATE = MAGNITUDE_COUNT * SCHEDULE_COUNT" in config
    assert assigned_value(tree, "PRIMARY_RANK") == 128
    assert assigned_value(tree, "SENSITIVITY_RANKS") == [64, 128]
    assert assigned_value(tree, "INTERVENTION_FORWARDS_PER_RECORD") == 30
    assert "token_hex(4)" in config
    assert '_colab_userdata.get("HF_TOKEN")' in config
    for forbidden in ["STAGE28_RUN_MODE", "STAGE28_SOURCE_COMMIT", "STAGE28_RUN_NONCE"]:
        assert forbidden not in "\n".join(code_cells)

    joined = "\n".join(code_cells)
    for required in [
        "def signed_control_area(",
        "def area_action_bank(",
        "def area_law_metrics(",
        "def model_physics_area_metrics(",
        "def area_swap_delta(",
        "def area_ablation_delta(",
        "def screen_development_panel(",
        "def physical_area_gate(",
        "def model_area_gate(",
        "def causal_area_gate(",
        "HYBRID_CONTROL_AREA_LAW_CAUSALLY_ENCODED",
        "stage28_hybrid_control_area_result_bundle_",
    ]:
        assert required in joined, f"missing {required}"
    for prohibited in [
        "torch.autograd", ".backward(", "torch.func.jvp", "torch.func.vjp",
        "jacrev", "jacfwd",
    ]:
        assert prohibited not in joined, f"prohibited Stage 28 machinery: {prohibited}"

    action_design = function_source(code_cells, "area_action_bank")
    assert "within-magnitude schedules lost equal impulse" in action_design
    assert "signed control-area levels are not antisymmetric" in action_design
    development = code_cells[6]
    assert "for panel_index, panel in enumerate(MAGNITUDE_PANELS)" in development
    assert '"model_outputs_used": False' in development
    assert '"physical_area_effect_magnitude_used": False' in development
    selection = code_cells[6]
    assert '"commutator_or_area_effect_magnitude_used_for_selection": False' in selection
    assert '"model_outputs_used_for_selection": False' in selection
    upstream = code_cells[7]
    assert "EXPECTED_STAGE18_SUBSPACE_SHA256" in upstream
    assert "EXPECTED_STAGE27_SOURCE_COMMIT" in upstream
    assert "repaired positive Stage 27" in upstream
    edits = function_source(code_cells, "intervention_specs")
    for required in [
        "area_swap_delta(", "area_ablation_delta(", 'subspaces["primary_basis"]',
        'subspaces["shuffled_basis"]', 'subspaces[f"random_basis_',
        "wrong_state_area_delta(", "matched_common_mode(", '"full_activation_swap"',
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
    print("Stage 28 notebook validation passed")


if __name__ == "__main__":
    validate()

