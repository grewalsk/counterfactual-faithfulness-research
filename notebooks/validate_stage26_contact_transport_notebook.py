import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "26_contact_frame_causal_transport.ipynb"
BUILDER = ROOT / "build_stage26_contact_frame_transport_notebook.py"


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
    assert before == after, "Stage 26 builder is not deterministic"

    notebook = json.loads(after)
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["metadata"]["colab"]["gpuType"] == "L4"
    assert len(notebook["cells"]) == 14
    code_cells = [source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 13
    for index, cell in enumerate(notebook["cells"]):
        assert cell["id"] == f"stage26-{index:02d}"
        assert not cell.get("outputs")
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            ast.parse(source(cell))

    config = code_cells[0]
    tree = ast.parse(config)
    assert assigned_value(tree, "PROTOCOL_ID") == "stage26-contact-frame-causal-transport-v1"
    assert assigned_value(tree, "RUN_MODE") == "smoke"
    assert assigned_value(tree, "DISCOVERY_BLOCKS") == list(range(6))
    assert assigned_value(tree, "CONSTRUCTION_TRAJECTORY_TARGET") == 40
    assert assigned_value(tree, "EVALUATION_TRAJECTORY_TARGET") == 40
    assert assigned_value(tree, "CONTACT_POLYNOMIAL_DEGREE") == 1
    assert assigned_value(tree, "CONTACT_BASIS_DIM") == 3
    assert assigned_value(tree, "FIBER_RANK") == 4
    assert assigned_value(tree, "UPSTREAM_STAGE25_RUN_SUFFIX") == "0c557d94ceae"
    assert assigned_value(tree, "PATCH_CONDITIONS") == [
        "aligned_fiber", "world_axis_control", "donor_location_control",
        "random_local_control", "reverse_aligned", "full_local_swap",
    ]
    for secret in ["STAGE26_RUN_MODE", "STAGE26_SOURCE_COMMIT", "STAGE26_RUN_NONCE"]:
        assert secret in config

    joined = "\n".join(code_cells)
    for required in [
        "def contact_frame_basis(",
        "def canonical_contact_features(",
        "def transport_contact_delta(",
        "def fit_response_fiber(",
        "def select_low_response_donor(",
        "def install_contact_recorder(",
        "def disable_agent_block_collision(",
        "CONTACT_FRAME_CAUSAL_TRANSPORT_SUPPORTED",
        "CONTACT_FIELD_READABLE_BUT_NOT_CAUSALLY_TRANSPORTABLE",
        "stage26_contact_transport_result_bundle_",
    ]:
        assert required in joined, f"missing {required}"
    for prohibited in [
        "torch.autograd", ".backward(", "torch.func.jvp", "torch.func.vjp",
        "jacrev", "jacfwd",
    ]:
        assert prohibited not in joined, f"prohibited Stage 26 machinery: {prohibited}"

    upstream = code_cells[5]
    assert "EXPECTED_STAGE25_SOURCE_COMMIT" in upstream
    assert "EXPECTED_STAGE25_STATUS" in upstream
    assert 'decision.get("status") == EXPECTED_STAGE25_STATUS' in upstream
    assert 'not bool(\n            decision.get("gates", {}).get("causal_erasure", True)' in upstream

    design = code_cells[6]
    assert '"model_loaded": bool("MODEL" in globals())' in design
    assert '"contact_basis": "gaussian_degree1_normal_tangent"' in design
    truth = code_cells[7]
    assert 'contact_points=np.asarray(contact_points, dtype=np.float64)' in truth
    assert 'np.average(np.stack(contact_points)' in truth
    assert 'ghost agent–block contact callback fired' in truth

    construction = code_cells[8]
    freeze = code_cells[9]
    evaluation = code_cells[10]
    assert "capture_blocks=DISCOVERY_BLOCKS" in construction
    assert '"frozen_before_evaluation_activations": True' in freeze
    assert '"evaluation_activation_ids_seen": []' in freeze
    assert "grouped_ridge_oof(" in freeze
    assert 'selected_block=np.asarray(selected_block' in freeze
    assert code_cells.index(freeze) < code_cells.index(evaluation)

    edits = function_source(code_cells, "make_record_transport_edits")
    for required in [
        "projected_donor_delta(",
        'edits["aligned_fiber"]',
        'edits["world_axis_control"]',
        'edits["donor_location_control"]',
        'edits["random_local_control"]',
        'edits["reverse_aligned"]',
        'edits["full_local_swap"]',
        "MAX_CANONICAL_RECONSTRUCTION_ERROR",
    ]:
        assert required in edits

    decision = code_cells[11]
    assert "verify_executed_notebook_through(" in decision
    assert 'aligned_summary["lower"] >= MIN_ALIGNED_TRANSFER' in decision
    assert 'summary["mean"] >= MIN_GAIN_OVER_CONTROL and summary["lower"] > 0' in decision
    assert 'native_summary["median"] >= MIN_NATIVE_MEDIAN_CONTACT' in decision
    assert 'RUN_MODE != "pilot"' in decision

    observed_digest = assigned_value(tree, "NOTEBOOK_PROTOCOL_SHA256")
    sources = [source(notebook["cells"][0])]
    sources.append(config.replace(observed_digest, "__PROTOCOL_DIGEST__", 1))
    sources.extend(code_cells[1:])
    expected_digest = hashlib.sha256(
        json.dumps(sources, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    assert observed_digest == expected_digest
    print("Stage 26 notebook validation passed")


if __name__ == "__main__":
    validate()
