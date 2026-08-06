import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "25_causal_kkt_tomography.ipynb"
BUILDER = ROOT / "build_stage25_causal_kkt_tomography_notebook.py"


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
    assert before == after, "Stage 25 builder is not deterministic"

    notebook = json.loads(after)
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["metadata"]["colab"]["gpuType"] == "L4"
    assert len(notebook["cells"]) == 15
    code_cells = [source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 14
    for index, cell in enumerate(notebook["cells"]):
        assert cell["id"] == f"stage25-{index:02d}"
        assert not cell.get("outputs")
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            ast.parse(source(cell))

    config = code_cells[0]
    tree = ast.parse(config)
    assert assigned_value(tree, "PROTOCOL_ID") == "stage25-causal-kkt-tomography-v1"
    assert assigned_value(tree, "RUN_MODE") == "smoke"
    assert assigned_value(tree, "SELECTED_BLOCK") == 1
    assert assigned_value(tree, "CONSTRUCTION_TRAJECTORY_TARGET") == 48
    assert assigned_value(tree, "EVALUATION_TRAJECTORY_TARGET") == 48
    assert assigned_value(tree, "PATCH_CONDITIONS") == [
        "impulse_erase", "reverse_impulse", "random_matched"
    ]
    assert assigned_value(tree, "UPSTREAM_STAGE24_RUN_SUFFIX") == "b18ea7810677"
    for secret in ["STAGE25_RUN_MODE", "STAGE25_SOURCE_COMMIT", "STAGE25_RUN_NONCE"]:
        assert secret in config

    joined = "\n".join(code_cells)
    for required in [
        "def install_contact_recorder(",
        "def disable_agent_block_collision(",
        "def aggregate_contact_trace(",
        "def stepwise_impulse_momentum_residual(",
        "def fit_and_freeze_impulse_reader(",
        "def make_record_edits(",
        "minimum_norm_coordinate_edit(",
        "SMOKE_ONLY",
        "LATENT_CONTACT_IMPULSE_MECHANISM_SUPPORTED",
        "IMPULSE_READABLE_BUT_NOT_CAUSALLY_USED",
        "stage25_causal_kkt_result_bundle_",
    ]:
        assert required in joined, f"missing {required}"
    for prohibited in [
        "torch.autograd", ".backward(", "torch.func.jvp", "torch.func.vjp",
        "jacrev", "jacfwd",
    ]:
        assert prohibited not in joined, f"prohibited Stage 25 machinery: {prohibited}"

    upstream = code_cells[5]
    assert "EXPECTED_STAGE24_SOURCE_COMMIT" in upstream
    assert "EXPECTED_STAGE24_STATUS" in upstream
    assert 'decision.get("status") == EXPECTED_STAGE24_STATUS' in upstream
    assert 'shutil.copy2(required["geometry"], geometry_destination)' in upstream

    ghost = function_source(code_cells, "disable_agent_block_collision")
    assert "agent_category" in ghost and "block_category" in ghost
    assert "mask=all_masks ^ block_category" in ghost
    assert "mask=all_masks ^ agent_category" in ghost
    recorder = function_source(code_cells, "install_contact_recorder")
    for required in [
        "arbiter.total_impulse", "points.normal", "point.distance",
        "physics_step", "block_velocity", "block_mass",
    ]:
        assert required in recorder
    aggregate = function_source(code_cells, "aggregate_contact_trace")
    assert "stepwise_impulse_momentum_residual(" in aggregate
    assert "np.median(step_residuals)" in aggregate

    construction_cell = code_cells[8]
    reader_cell = code_cells[9]
    evaluation_cell = code_cells[10]
    assert "extract_baselines(CONSTRUCTION_RECORDS" in construction_cell
    assert '"frozen_before_evaluation_activations": True' in reader_cell
    assert '"evaluation_activation_ids_seen": []' in reader_cell
    assert "extract_baselines(EVALUATION_RECORDS" in evaluation_cell
    assert code_cells.index(reader_cell) < code_cells.index(evaluation_cell)

    edits = function_source(code_cells, "make_record_edits")
    assert "protected=mode_covectors" in edits
    assert "zero_standard - predicted_standard[action_index]" in edits
    assert "reverse[action_index] = -erase[action_index]" in edits
    assert "orthogonal_random_control(" in edits

    decision_cell = code_cells[12]
    assert "verify_executed_notebook_through(" in decision_cell
    assert 'summaries.get("impulse_erase", {}).get("lower"' in decision_cell
    assert 'RUN_MODE != "pilot"' in decision_cell
    assert 'gain_summary["lower"] > 0' in decision_cell
    assert 'max_mode_drift <= MAX_MODE_DRIFT' in decision_cell

    observed_digest = assigned_value(tree, "NOTEBOOK_PROTOCOL_SHA256")
    sources = [source(notebook["cells"][0])]
    sources.append(config.replace(observed_digest, "__PROTOCOL_DIGEST__", 1))
    sources.extend(code_cells[1:])
    expected_digest = hashlib.sha256(
        json.dumps(sources, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    assert observed_digest == expected_digest
    print("Stage 25 notebook validation passed")


if __name__ == "__main__":
    validate()
