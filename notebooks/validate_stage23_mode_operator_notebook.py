import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "23_causal_mode_manifold_operator_switch.ipynb"
BUILDER = ROOT / "build_stage23_mode_operator_notebook.py"


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
    assert before == after, "Stage 23 builder is not deterministic"

    notebook = json.loads(after)
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["metadata"]["colab"]["gpuType"] == "L4"
    assert len(notebook["cells"]) == 15
    code_cells = [source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 14
    for index, cell in enumerate(notebook["cells"]):
        assert cell["id"] == f"stage23-{index:02d}"
        assert not cell.get("outputs")
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            ast.parse(source(cell))

    config = code_cells[0]
    tree = ast.parse(config)
    assert assigned_value(tree, "PROTOCOL_ID") == (
        "stage23-causal-mode-manifold-operator-switch-v1"
    )
    assert assigned_value(tree, "RUN_MODE") == "smoke"
    assert assigned_value(tree, "SELECTED_BLOCK") == 1
    assert assigned_value(tree, "DOWNSTREAM_BLOCKS") == [2, 3, 4, 5]
    assert assigned_value(tree, "CONTENT_RANK") == 64
    assert assigned_value(tree, "PROBE_COUNT") == 8
    assert assigned_value(tree, "FINITE_PROBE_DOSE") == 0.5
    assert assigned_value(tree, "PAIRS_PER_STATE") == 1
    assert assigned_value(tree, "PATCHED_FORWARDS_PER_PAIR") == 81
    assert assigned_value(tree, "CONSTRUCTION_TRAJECTORY_TARGET") == 48
    assert assigned_value(tree, "EVALUATION_TRAJECTORY_TARGET") == 64
    assert assigned_value(tree, "UPSTREAM_STAGE22_RUN_SUFFIX") == "7b0be321cc7d"
    assert assigned_value(tree, "EXPECTED_STAGE22_PARTITION_SHA256") == (
        "0f9e376e9d874f1b2429e62b119cf56ea77e3458b8a44c1d4575ed73c988697f"
    )
    for secret in ["STAGE23_RUN_MODE", "STAGE23_SOURCE_COMMIT", "STAGE23_RUN_NONCE"]:
        assert secret in config

    joined = "\n".join(code_cells)
    for required in [
        "def countsketch_mode_covectors(",
        "def minimal_constrained_transport(",
        "def pair_contexts(",
        "def pair_operator_rows(",
        "def evaluate_mode_operator(",
        "CAUSAL_MODE_OPERATOR_SWITCH_CONFIRMED",
        "MODE_FLIPS_WITHOUT_OPERATOR_SWITCH",
        "MODE_TRANSPORT_INVALID",
        "stage23_mode_operator_result_bundle_",
    ]:
        assert required in joined, f"missing {required}"
    for prohibited in [
        "torch.autograd",
        ".backward(",
        "torch.func.jvp",
        "torch.func.vjp",
        "jacrev",
        "jacfwd",
    ]:
        assert prohibited not in joined, f"prohibited Stage 23 machinery: {prohibited}"

    upstream_cell = code_cells[5]
    assert "EXPECTED_STAGE22_SOURCE_COMMIT" in upstream_cell
    assert "EXPECTED_STAGE22_PARTITION_SHA256" in upstream_cell
    assert 'shutil.copy2(required["partition"], destination)' in upstream_cell

    factorization_cell = code_cells[9]
    assert '"frozen_before_evaluation_activations": True' in factorization_cell
    assert '"contact_labels_used": False' in factorization_cell
    assert "orthogonalize_basis(raw_basis, mode_covectors)" in factorization_cell

    pairing = function_source(code_cells, "select_model_only_pairs")
    for prohibited in ["branch_path", "contact", "truth", "interaction_counts"]:
        assert prohibited not in pairing
    evaluation_cell = code_cells[10]
    freeze_write = evaluation_cell.index(
        'write_json(DESIGN_DIR / "evaluation_model_pair_freeze.json", pair_freeze)'
    )
    contact_read = evaluation_cell.index(
        'with np.load(branch_path(record["record_id"])) as truth:'
    )
    assert freeze_write < contact_read

    contexts = assigned_value(tree, "CONTEXTS")
    assert contexts == [
        "off",
        "native_on",
        "mode_transport",
        "permuted_transport",
        "random_tangent",
    ]
    intervention_cell = code_cells[11]
    assert "base_q + mode_covectors.T @ native_on" in intervention_cell
    assert "content_basis" in intervention_cell
    assert "symmetric_finite_response" in intervention_cell
    assert 'intervention={"block": SELECTED_BLOCK, "delta": delta}' in intervention_cell
    assert "observed_forwards = 1 + 2 * len(probes) * len(ACTIVE_CONTEXTS)" in intervention_cell

    decision_cell = code_cells[12]
    assert "verify_executed_notebook_through(" in decision_cell
    assert '"# Apply the preregistered Stage 23 causal operator-switch gates."' in decision_cell
    assert 'learned_summary["lower"] > 0' in decision_cell
    assert 'random_summary["lower"] > 0' in decision_cell
    assert 'permuted_summary["lower"] > 0' in decision_cell

    # Frozen pilot workload: 64 pairs x (one center + 5 x 8 x two signs).
    assert 64 * 81 == 5184

    observed_digest = assigned_value(tree, "NOTEBOOK_PROTOCOL_SHA256")
    sources = [source(notebook["cells"][0])]
    sources.append(config.replace(observed_digest, "__PROTOCOL_DIGEST__", 1))
    sources.extend(code_cells[1:])
    expected_digest = hashlib.sha256(
        json.dumps(sources, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    assert observed_digest == expected_digest
    print("Stage 23 notebook validation passed")


if __name__ == "__main__":
    validate()
