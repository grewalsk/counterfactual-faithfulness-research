import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "22_latent_hybrid_gate_interaction.ipynb"
BUILDER = ROOT / "build_stage22_hybrid_gate_notebook.py"


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
    assert before == after, "Stage 22 builder is not deterministic"

    notebook = json.loads(after)
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["metadata"]["colab"]["gpuType"] == "L4"
    assert len(notebook["cells"]) == 15
    code_cells = [source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 14
    for index, cell in enumerate(notebook["cells"]):
        assert cell["id"] == f"stage22-{index:02d}"
        assert not cell.get("outputs")
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            ast.parse(source(cell))

    config = code_cells[0]
    tree = ast.parse(config)
    assert assigned_value(tree, "PROTOCOL_ID") == (
        "stage22-label-free-mode-gate-effect-factorial-v1"
    )
    assert assigned_value(tree, "RUN_MODE") == "smoke"
    assert assigned_value(tree, "DISCOVERY_BLOCKS") == [0, 1, 2, 3, 4]
    assert assigned_value(tree, "EFFECT_RANK") == 32
    assert assigned_value(tree, "PAIRS_PER_STATE") == 2
    assert assigned_value(tree, "PATCHED_FORWARDS_PER_PAIR") == 8
    assert assigned_value(tree, "CONSTRUCTION_TRAJECTORY_TARGET") == 32
    assert assigned_value(tree, "EVALUATION_TRAJECTORY_TARGET") == 40
    for secret in ["STAGE22_RUN_MODE", "STAGE22_SOURCE_COMMIT", "STAGE22_RUN_NONCE"]:
        assert secret in config

    joined = "\n".join(code_cells)
    for required in [
        "def discover_mode_partition(",
        "def fit_and_freeze_factorization(",
        "def select_model_only_pairs(",
        "def pair_factorial_rows(",
        "def factorial_interaction_metrics(",
        "EVENT_GATED_CAUSAL_INTERACTION_CONFIRMED",
        "PHYSICAL_MODE_WITHOUT_CAUSAL_INTERACTION",
        "stage22_hybrid_gate_result_bundle_",
    ]:
        assert required in joined, f"missing {required}"
    for prohibited in [
        "torch.autograd",
        ".backward(",
        "torch.func.jvp",
        "torch.func.vjp",
        "jacrev",
        "jacfwd",
        "fit_logistic",
    ]:
        assert prohibited not in joined, f"prohibited Stage 22 machinery: {prohibited}"

    discovery = function_source(code_cells, "discover_mode_partition")
    assert "interaction_counts" not in discovery
    assert "physical_contact" not in discovery
    pairing = function_source(code_cells, "select_model_only_pairs")
    assert "branch_path" not in pairing
    assert "contact" not in pairing
    assert "truth" not in pairing

    evaluation_cell = code_cells[10]
    freeze_write = evaluation_cell.index(
        'write_json(DESIGN_DIR / "evaluation_model_pair_freeze.json", pair_freeze)'
    )
    contact_read = evaluation_cell.index('with np.load(branch_path(record["record_id"])) as truth:')
    assert freeze_write < contact_read
    factorization_cell = code_cells[9]
    assert '"frozen_before_evaluation_activations": True' in factorization_cell
    assert '"contact_labels_used": False' in factorization_cell
    intervention_cell = code_cells[11]
    assert "gate_delta + effect" in intervention_cell
    assert "shuffled_gate" in intervention_cell
    assert "random_gate" in intervention_cell
    assert "full_swap_coefficient" in intervention_cell
    decision_cell = code_cells[12]
    assert 'verify_executed_notebook_through(' in decision_cell
    assert '"# Apply Stage 22 frozen scientific gates."' in decision_cell

    # Frozen pilot workload: 40 states x 2 model-only pairs x 8 patched forwards.
    assert 40 * 2 * 8 == 640

    observed_digest = assigned_value(tree, "NOTEBOOK_PROTOCOL_SHA256")
    sources = [source(notebook["cells"][0])]
    sources.append(config.replace(observed_digest, "__PROTOCOL_DIGEST__", 1))
    sources.extend(code_cells[1:])
    expected_digest = hashlib.sha256(
        json.dumps(sources, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    assert observed_digest == expected_digest
    print("Stage 22 notebook validation passed")


if __name__ == "__main__":
    validate()
