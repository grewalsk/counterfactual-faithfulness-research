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
NOTEBOOK = ROOT / "32_powered_bounded_cross_model_confirmation.ipynb"
BUILDER = ROOT / "build_stage32_powered_bounded_confirmation_notebook.py"
sys.path.insert(0, str(REPOSITORY / "src"))

from cf_faithfulness.stage32_bounded_confirmation import (  # noqa: E402
    bounded_cosine,
    bounded_swap_closure_rows,
    paired_model_difference_rows,
    state_placebo_advantage,
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
    ineligible = bounded_cosine([1.0, 0.0], [1e-5, 0.0], 1e-6)
    assert not ineligible["eligible"] and np.isnan(ineligible["cosine"])
    rng = np.random.default_rng(32)
    target = rng.normal(size=(24, 3, 4))
    reversal = np.arange(24).reshape(4, 6)[:, ::-1].reshape(-1)
    rows = bounded_swap_closure_rows(target, target[reversal], target, 4, 6)
    assert all(row["grounded_eligible"] for row in rows)
    assert all(np.isclose(row["grounded_cosine"], 1.0) for row in rows)

    left = [{
        "record_id": 1, "family_index": 0, "magnitude_index": 0,
        "outcome": 0.2, "grounded_cosine": 0.1,
    }]
    right = [{
        "record_id": 1, "family_index": 0, "magnitude_index": 0,
        "outcome": 0.5, "grounded_cosine": 0.04,
    }]
    paired = paired_model_difference_rows(left, right, ["grounded_cosine"])
    assert np.isclose(paired[0]["outcome"], 0.3)
    assert np.isclose(paired[0]["difference_grounded_cosine"], -0.06)
    placebo = state_placebo_advantage(
        [0.4, 0.2], [[0.1, 0.2], [0.0, 0.1]], [7, 7]
    )
    assert np.isclose(placebo[0]["primary_minus_median_placebo_improvement"], 0.2)


def validate():
    before = NOTEBOOK.read_bytes()
    subprocess.run(
        [sys.executable, str(BUILDER)],
        check=True,
        capture_output=True,
        env=dict(os.environ),
    )
    after = NOTEBOOK.read_bytes()
    assert before == after, "Stage 32 builder is not deterministic"

    notebook = json.loads(after)
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["metadata"]["colab"]["gpuType"] == "L4"
    assert len(notebook["cells"]) == 11
    code_cells = [
        source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    assert len(code_cells) == 10
    for index, cell in enumerate(notebook["cells"]):
        assert cell["id"] == f"stage32-{index:02d}"
        assert not cell.get("outputs")
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            ast.parse(source(cell))

    config = code_cells[0]
    tree = ast.parse(config)
    assert assigned_value(tree, "PROTOCOL_ID") == (
        "stage32-powered-bounded-cross-model-confirmation-v1"
    )
    assert assigned_value(tree, "RUN_MODE") == "pilot"
    assert assigned_value(tree, "EXPERIMENT_SOURCE_REF") == (
        "codex/stage32-powered-bounded-cross-model"
    )
    assert assigned_value(tree, "EXPECTED_STAGE31_SOURCE_COMMIT") == (
        "8b407c34117d1029ad4bd212e3fff03062f2b437"
    )
    assert assigned_value(tree, "EXPECTED_STAGE31_STATUS") == (
        "WITHIN_MODEL_REPLICATION_WITHOUT_PAIRED_CERTIFICATE"
    )
    assert assigned_value(tree, "EXPECTED_STAGE31_SUBSPACE_SHA256") == {
        "jepa": "762c9437a0bc5c8726359158394ae0fd10814aa63f59f6323a2c81c8bad6703c",
        "dino": "6e146353775f6f714f8401337f4241af3cf7d53111544c301949c34817a7499c",
    }
    assert assigned_value(tree, "EVALUATION_TARGET") == 160
    assert "POOL_TRAJECTORIES = list(range(5000, 5800))" in config
    assert assigned_value(tree, "ACTION_FAMILY_COUNT") == 3
    assert "TOTAL_ACTIONS_PER_STATE = ACTION_FAMILY_COUNT * ACTIONS_PER_FAMILY" in config
    assert "assert ACTIONS_PER_FAMILY == 24 and TOTAL_ACTIONS_PER_STATE == 72" in config
    assert assigned_value(tree, "MIN_GROUNDED_TARGET_ENERGY") == 1e-6
    assert assigned_value(tree, "MIN_PAIRED_RELATIVE_MSE_IMPROVEMENT") == 0.05
    assert assigned_value(tree, "MIN_ELIGIBLE_STATES") == 140
    assert assigned_value(tree, "ASSET_SPECS") == {}
    assert "token_hex(4)" in config
    assert '_colab_userdata.get("HF_TOKEN")' in config
    for forbidden in [
        "STAGE32_RUN_MODE", "STAGE32_SOURCE_COMMIT", "STAGE32_RUN_NONCE",
    ]:
        assert forbidden not in "\n".join(code_cells)

    joined = "\n".join(code_cells)
    for required in [
        "def bounded_cosine(", "def bounded_swap_closure_rows(",
        "def state_placebo_advantage(", "def all_family_action_banks(",
        "def intervention_specs_cuda(", "def paired_certificate_gate(",
        "BOUNDED_CROSS_MODEL_GROUNDED_CLOSURE_CERTIFICATE_CONFIRMED",
        "PAIRED_SIGNAL_WITHOUT_SUBSPACE_SPECIFICITY",
        "stage32_bounded_confirmation_result_bundle_",
    ]:
        assert required in joined, f"missing {required}"
    for prohibited in [
        "torch.autograd", ".backward(", "torch.func.jvp", "torch.func.vjp",
        "jacrev", "jacfwd",
    ]:
        assert prohibited not in joined, f"prohibited machinery: {prohibited}"

    bounded_source = function_source(code_cells, "bounded_cosine")
    assert "target_energy < floor" in bounded_source
    assert "np.clip(cosine, -1.0, 1.0)" in bounded_source
    closure_source = function_source(code_cells, "bounded_swap_closure_rows")
    assert "diagnostic schedules must be closed under reversal" in closure_source
    assert "minimum_target_energy=minimum_target_energy" in closure_source

    upstream = code_cells[5]
    for required in [
        "EXPECTED_STAGE31_DECISION_SHA256",
        "EXPECTED_STAGE31_SOURCE_SHA256",
        "EXPECTED_STAGE31_SUBSPACE_SHA256",
        'not decision.get("paired_cross_model_grounded_reliability_gate", {}).get("passed", True)',
        '"stage32_basis_refit_or_tuning": False',
    ]:
        assert required in upstream
    physical = code_cells[6]
    assert "persistent_all_72_branches" in physical
    assert '"selection_uses_contact_only": True' in physical
    evaluation = code_cells[7]
    assert "bounded_swap_closure_rows(" in evaluation
    assert "official_terminal_planning_rows(" in evaluation
    assert "MIN_GROUNDED_TARGET_ENERGY" in evaluation
    assert "grounded_coefficient" not in evaluation
    decision = code_cells[8]
    assert "state_placebo_advantage(" in decision
    assert "all_family_means_positive" in decision
    assert "primary_minus_median_placebo_ci95" in decision
    assert '"coefficient_ratio_computed_or_used": False' in decision
    assert "grounded_coefficient" not in decision

    observed_digest = assigned_value(tree, "NOTEBOOK_PROTOCOL_SHA256")
    sources = [source(notebook["cells"][0])]
    sources.append(config.replace(observed_digest, "__PROTOCOL_DIGEST__", 1))
    sources.extend(code_cells[1:])
    expected_digest = hashlib.sha256(
        json.dumps(sources, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    assert observed_digest == expected_digest
    validate_numerics()
    print("Stage 32 notebook validation passed")


if __name__ == "__main__":
    validate()
