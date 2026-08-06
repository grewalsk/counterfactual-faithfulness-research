import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "24_causal_completion_rank.ipynb"
BUILDER = ROOT / "build_stage24_causal_completion_notebook.py"


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
    assert before == after, "Stage 24 builder is not deterministic"

    notebook = json.loads(after)
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["metadata"]["colab"]["gpuType"] == "L4"
    assert len(notebook["cells"]) == 15
    code_cells = [source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 14
    for index, cell in enumerate(notebook["cells"]):
        assert cell["id"] == f"stage24-{index:02d}"
        assert not cell.get("outputs")
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            ast.parse(source(cell))

    config = code_cells[0]
    tree = ast.parse(config)
    assert assigned_value(tree, "PROTOCOL_ID") == "stage24-causal-completion-rank-v1"
    assert assigned_value(tree, "RUN_MODE") == "smoke"
    assert assigned_value(tree, "SELECTED_BLOCK") == 1
    assert assigned_value(tree, "DOWNSTREAM_BLOCKS") == [2, 3, 4, 5]
    assert assigned_value(tree, "COMPLETION_RANKS") == [0, 4, 8, 16, 32, 64]
    assert assigned_value(tree, "COMPLETION_BASIS_RANK") == 64
    assert assigned_value(tree, "CONSTRUCTION_PAIRS_PER_STATE") == 3
    assert assigned_value(tree, "PAIRS_PER_STATE") == 1
    assert assigned_value(tree, "FULL_CONTEXT_COUNT") == 18
    assert assigned_value(tree, "PATCHED_FORWARDS_PER_PAIR") == 289
    assert assigned_value(tree, "CONSTRUCTION_TRAJECTORY_TARGET") == 64
    assert assigned_value(tree, "EVALUATION_TRAJECTORY_TARGET") == 64
    assert assigned_value(tree, "COMPLETION_TRANSFER_THRESHOLD") == 0.8
    assert assigned_value(tree, "UPSTREAM_STAGE23_RUN_SUFFIX") == "d47dee8b6789"
    assert assigned_value(tree, "EXPECTED_STAGE23_GEOMETRY_SHA256") == (
        "3598cc57768077550913cc2008baf870c55288089b8570d5ae96c523956aa4ff"
    )
    for secret in ["STAGE24_RUN_MODE", "STAGE24_SOURCE_COMMIT", "STAGE24_RUN_NONCE"]:
        assert secret in config

    joined = "\n".join(code_cells)
    for required in [
        "def completion_residual(",
        "def completion_edit(",
        "def causal_completion_rank(",
        "def fit_and_freeze_completion_bases(",
        "def pair_completion_contexts(",
        "def evaluate_causal_completion(",
        "COMPACT_CAUSAL_COMPLETION_FOUND",
        "NO_RANK64_CAUSAL_COMPLETION",
        "GENERIC_RESIDUAL_COMPLETION_NOT_MODE_SPECIFIC",
        "stage24_causal_completion_result_bundle_",
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
        assert prohibited not in joined, f"prohibited Stage 24 machinery: {prohibited}"

    upstream_cell = code_cells[5]
    assert "EXPECTED_STAGE23_SOURCE_COMMIT" in upstream_cell
    assert "EXPECTED_STAGE23_GEOMETRY_SHA256" in upstream_cell
    assert '"operator_switch_rejected"' in upstream_cell
    assert 'shutil.copy2(required["geometry"], geometry_destination)' in upstream_cell

    construction_pairing = function_source(code_cells, "select_construction_pairs")
    for prohibited in ["branch_path", "contact", "truth", "interaction_counts"]:
        assert prohibited not in construction_pairing
    basis_cell = code_cells[9]
    assert '"frozen_before_evaluation_activations": True' in basis_cell
    assert '"physical_contact_labels_used": False' in basis_cell
    assert 'DESIGN_DIR / "construction_completion_pair_freeze.json"' in basis_cell
    assert "mode_null_nested_qr(completion_raw, mode_covectors)" in basis_cell
    assert "mode_null_nested_qr(same_raw, mode_covectors)" in basis_cell
    assert "mode_null_nested_qr(random_raw, mode_covectors)" in basis_cell

    evaluation_pairing = function_source(code_cells, "select_model_only_pairs")
    for prohibited in ["branch_path", "contact", "truth", "interaction_counts"]:
        assert prohibited not in evaluation_pairing
    evaluation_cell = code_cells[10]
    assert 'SUBSPACE_DIR / "causal_completion_basis_freeze.json"' in evaluation_cell
    freeze_write = evaluation_cell.index(
        'write_json(DESIGN_DIR / "evaluation_model_pair_freeze.json", pair_freeze)'
    )
    contact_read = evaluation_cell.index(
        'with np.load(branch_path(record["record_id"])) as truth:'
    )
    assert freeze_write < contact_read

    intervention_cell = code_cells[11]
    assert '"q_only": mode_edit' in intervention_cell
    assert 'contexts[f"{family}_rank_{rank}"] = completion_edit(' in intervention_cell
    assert "symmetric_finite_response" in intervention_cell
    assert "observed_forwards = 1 + 2 * len(probes) * len(ACTIVE_CONTEXTS)" in intervention_cell
    assert 'intervention={"block": SELECTED_BLOCK, "delta": delta}' in intervention_cell

    decision_cell = code_cells[12]
    assert "verify_executed_notebook_through(" in decision_cell
    assert '"# Apply the preregistered Stage 24 causal-completion-rank gates."' in decision_cell
    assert 'causal_completion_rank(' in decision_cell
    assert 'summary["lower"] > 0' in decision_cell
    assert 'summary["mean"] >= MIN_GAIN_OVER_CONTROL' in decision_cell
    assert '"compact_completion_claim_authorized": bool(' in decision_cell

    # Frozen pilot: 64 evaluation pairs × (one center + 18 contexts × 8 probes × two signs).
    assert 64 * 289 == 18496

    observed_digest = assigned_value(tree, "NOTEBOOK_PROTOCOL_SHA256")
    sources = [source(notebook["cells"][0])]
    sources.append(config.replace(observed_digest, "__PROTOCOL_DIGEST__", 1))
    sources.extend(code_cells[1:])
    expected_digest = hashlib.sha256(
        json.dumps(sources, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    assert observed_digest == expected_digest
    print("Stage 24 notebook validation passed")


if __name__ == "__main__":
    validate()
