import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
NOTEBOOK = ROOT / "29_grounded_causal_closure.ipynb"
BUILDER = ROOT / "build_stage29_grounded_causal_closure_notebook.py"
sys.path.insert(0, str(REPOSITORY / "src"))

from cf_faithfulness.stage29_grounded_closure import (  # noqa: E402
    grounded_intervention_metrics,
    latent_closure_metrics,
)


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


def validate_numerics():
    rng = np.random.default_rng(29)
    target = rng.normal(size=(24, 7))
    closure = latent_closure_metrics(target, target, magnitude_count=4)
    assert np.isclose(closure["area_coefficient"], 1.0)
    assert np.isclose(closure["area_cosine"], 1.0)
    permutation = np.arange(24).reshape(4, 6)[:, ::-1].reshape(-1)
    patched = target[permutation]
    grounded = grounded_intervention_metrics(
        target, patched, target, magnitude_count=4, mode="swap"
    )
    assert np.isclose(grounded["self_coefficient"], 1.0)
    assert np.isclose(grounded["grounded_coefficient"], 1.0)
    assert np.isclose(grounded["self_cosine"], 1.0)
    assert np.isclose(grounded["grounded_cosine"], 1.0)

    self_latent = np.zeros((24, 2), dtype=np.float64)
    true_latent = np.zeros_like(self_latent)
    levels = np.asarray([25, 15, 5, -5, -15, -25], dtype=np.float64)
    self_latent[:, 0] = np.tile(levels, 4)
    true_latent[:, 1] = np.tile(levels, 4)
    self_patch = self_latent[permutation]
    hallucination = grounded_intervention_metrics(
        self_latent, self_patch, true_latent, magnitude_count=4, mode="swap"
    )
    assert np.isclose(hallucination["self_cosine"], 1.0)
    assert np.isclose(hallucination["grounded_cosine"], 0.0)


def validate():
    before = NOTEBOOK.read_bytes()
    environment = dict(os.environ)
    subprocess.run(
        [sys.executable, str(BUILDER)],
        check=True,
        capture_output=True,
        env=environment,
    )
    after = NOTEBOOK.read_bytes()
    assert before == after, "Stage 29 builder is not deterministic"

    notebook = json.loads(after)
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["metadata"]["colab"]["gpuType"] == "L4"
    assert len(notebook["cells"]) == 11
    code_cells = [source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 10
    for index, cell in enumerate(notebook["cells"]):
        assert cell["id"] == f"stage29-{index:02d}"
        assert not cell.get("outputs")
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            ast.parse(source(cell))

    config = code_cells[0]
    tree = ast.parse(config)
    assert assigned_value(tree, "PROTOCOL_ID") == "stage29-grounded-causal-closure-v1"
    assert assigned_value(tree, "RUN_MODE") == "pilot"
    assert assigned_value(tree, "EXPERIMENT_SOURCE_REF") == "codex/stage29-grounded-causal-closure"
    assert assigned_value(tree, "EXPECTED_STAGE28_STATUS") == "MODEL_DOES_NOT_CAPTURE_PHYSICAL_CONTROL_AREA_LAW"
    assert assigned_value(tree, "EXPECTED_STAGE28_SOURCE_COMMIT") == "917228edb9e7143c58bdd9640afe08ead75fa34c"
    assert assigned_value(tree, "EXPECTED_STAGE28_RECORDS") == 36
    assert assigned_value(tree, "EXPECTED_STAGE28_MAGNITUDES") == [0.10, 0.14, 0.18, 0.22]
    assert assigned_value(tree, "PRIMARY_RANK") == 128
    assert assigned_value(tree, "PILOT_INTERVENTION_FORWARDS_PER_RECORD") == 9
    assert "token_hex(4)" in config
    assert '_colab_userdata.get("HF_TOKEN")' in config
    for forbidden in ["STAGE29_RUN_MODE", "STAGE29_SOURCE_COMMIT", "STAGE29_RUN_NONCE"]:
        assert forbidden not in "\n".join(code_cells)

    joined = "\n".join(code_cells)
    for required in [
        "def encode_true_tokens(",
        "def tensor_area_component(",
        "def tensor_grounded_metrics(",
        "def intervention_specs(",
        "def target_encoder_gate(",
        "def predictor_target_gate(",
        "def decoder_localization_gate(",
        "def intervention_gate(",
        "CAUSAL_SELF_CONSISTENCY_WITHOUT_GROUNDED_CLOSURE",
        "PHYSICAL_READOUT_LIMITATION_SUPPORTED",
        "GROUNDED_CAUSAL_CLOSURE_SUPPORTED",
        "stage29_grounded_closure_result_bundle_",
    ]:
        assert required in joined, f"missing {required}"
    for prohibited in [
        "torch.autograd", ".backward(", "torch.func.jvp", "torch.func.vjp",
        "jacrev", "jacfwd",
    ]:
        assert prohibited not in joined, f"prohibited Stage 29 machinery: {prohibited}"

    upstream = code_cells[5]
    for required in [
        "EXPECTED_STAGE28_SOURCE_COMMIT",
        'STAGE28_CANDIDATES["confirmation_specs"]',
        "selected_truth_shards_verified",
        "truth_reused_without_resimulation",
        "sha256_file(path) != expected",
        "EXPECTED_STAGE18_SUBSPACE_SHA256",
    ]:
        assert required in upstream
    target_encoding = function_source(code_cells, "encode_true_tokens")
    assert 'payload["endpoint_visuals"][:, None]' in target_encoding
    assert 'encoded["visual"][:, :, 0]' in target_encoding
    assert "(ACTIONS_PER_STATE, 256, 384)" in target_encoding
    intervention = function_source(code_cells, "intervention_specs")
    for required in [
        'subspaces["primary_basis"][:, :PRIMARY_RANK]',
        'subspaces["shuffled_basis"][:, :PRIMARY_RANK]',
        'subspaces[f"random_basis_{draw:02d}"][:, :PRIMARY_RANK]',
        "wrong_state_swap_delta(",
        "area_swap_delta(white, MAGNITUDE_COUNT, basis=None",
        "area_ablation_delta(",
    ]:
        assert required in intervention
    decision = code_cells[8]
    assert 'intervention_gate("self")' in decision
    assert 'intervention_gate("grounded")' in decision
    assert 'PREDICTOR_GATE["passed"] and not DECODER_GATE["passed"]' in decision
    assert 'not PREDICTOR_GATE["passed"] and SELF_GATE["passed"] and not GROUNDED_GATE["passed"]' in decision
    assert '"new_reader_fit": False' in decision
    assert '"full_native_256_by_384_token_metrics": True' in decision

    observed_digest = assigned_value(tree, "NOTEBOOK_PROTOCOL_SHA256")
    sources = [source(notebook["cells"][0])]
    sources.append(config.replace(observed_digest, "__PROTOCOL_DIGEST__", 1))
    sources.extend(code_cells[1:])
    expected_digest = hashlib.sha256(
        json.dumps(sources, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    assert observed_digest == expected_digest
    validate_numerics()
    print("Stage 29 notebook validation passed")


if __name__ == "__main__":
    validate()
