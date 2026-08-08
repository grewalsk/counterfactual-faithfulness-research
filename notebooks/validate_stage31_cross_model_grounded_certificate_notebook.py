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
NOTEBOOK = ROOT / "31_cross_model_grounded_closure_certificate.ipynb"
BUILDER = ROOT / "build_stage31_cross_model_grounded_certificate_notebook.py"
sys.path.insert(0, str(REPOSITORY / "src"))

from cf_faithfulness.stage31_cross_model_certificate import (  # noqa: E402
    official_native_terminal_costs,
    paired_model_difference_rows,
    planner_metric_features,
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
    rng = np.random.default_rng(31)
    visual = rng.normal(size=(24, 256, 3))
    proprio = rng.normal(size=(24, 256, 4))
    chart = planner_metric_features(visual, proprio, alpha=0.1)
    expected = np.mean((visual[0] - visual[1]) ** 2) + 0.1 * np.mean(
        (proprio[0] - proprio[1]) ** 2
    )
    assert np.isclose(np.sum((chart[0] - chart[1]) ** 2), expected)
    costs = official_native_terminal_costs(
        visual, visual, proprio, proprio, 4, 6, goal_schedule=5, alpha=0.1
    )
    assert costs.shape == (4, 6)
    assert np.all(np.argmin(costs, axis=1) == 5)

    left = [{
        "record_id": 1, "magnitude_index": 0, "regime": "persistent_contact",
        "outcome": 0.2, "grounded_coefficient": 0.1,
    }]
    right = [{
        "record_id": 1, "magnitude_index": 0, "regime": "persistent_contact",
        "outcome": 0.5, "grounded_coefficient": 0.04,
    }]
    paired = paired_model_difference_rows(
        left, right, ["grounded_coefficient"]
    )
    assert np.isclose(paired[0]["outcome"], 0.3)
    assert np.isclose(paired[0]["difference_grounded_coefficient"], -0.06)


def validate():
    before = NOTEBOOK.read_bytes()
    subprocess.run(
        [sys.executable, str(BUILDER)],
        check=True,
        capture_output=True,
        env=dict(os.environ),
    )
    after = NOTEBOOK.read_bytes()
    assert before == after, "Stage 31 builder is not deterministic"

    notebook = json.loads(after)
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["metadata"]["colab"]["gpuType"] == "L4"
    assert len(notebook["cells"]) == 12
    code_cells = [
        source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    assert len(code_cells) == 11
    for index, cell in enumerate(notebook["cells"]):
        assert cell["id"] == f"stage31-{index:02d}"
        assert not cell.get("outputs")
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            ast.parse(source(cell))

    config = code_cells[0]
    tree = ast.parse(config)
    assert assigned_value(tree, "PROTOCOL_ID") == (
        "stage31-cross-model-grounded-closure-certificate-v1"
    )
    assert assigned_value(tree, "RUN_MODE") == "pilot"
    assert assigned_value(tree, "EXPERIMENT_SOURCE_REF") == (
        "codex/stage31-cross-model-grounded-certificate"
    )
    assert assigned_value(tree, "MODEL_NAMES") == [
        "jepa_wm_pusht", "dino_wm_pusht"
    ]
    assert assigned_value(tree, "REPO_COMMIT") == (
        "13cf1d9c7e476f53c17714d2e0f1dc239a883ce0"
    )
    assert assigned_value(tree, "EXPECTED_HF_REVISION") == (
        "9b9c41ef249466630dbf1a20e78391865d07b3b9"
    )
    assert assigned_value(tree, "EXPECTED_PRETRAINED_ASSET_SHA256") == {
        "jepa_wm_pusht.pth.tar": (
            "9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb"
        ),
        "dino_wm_pusht.pth.tar": (
            "8ec9cb05f22812d7f12e3c216b0637f41641055c0653e503e2746edb981b550f"
        ),
        "dinov2_vits14_pretrain.pth": (
            "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
        ),
    }
    assert assigned_value(tree, "CONSTRUCTION_TARGET") == 32
    assert assigned_value(tree, "EVALUATION_TARGET") == 120
    assert assigned_value(tree, "PRIMARY_RANK") == 128
    assert assigned_value(tree, "OFFICIAL_PROPRIO_ALPHA") == 0.1
    assert assigned_value(tree, "DIAGNOSTIC_SCHEDULES") == [1, 2, 3, 4]
    assert assigned_value(tree, "PLANNING_GOAL_SCHEDULES") == [0, 5]
    assert assigned_value(tree, "MIN_PAIRED_RELATIVE_MSE_IMPROVEMENT") == 0.05
    assert assigned_value(tree, "ASSET_SPECS") == {}
    assert "token_hex(4)" in config
    assert '_colab_userdata.get("HF_TOKEN")' in config
    for forbidden in [
        "STAGE31_RUN_MODE", "STAGE31_SOURCE_COMMIT", "STAGE31_RUN_NONCE",
    ]:
        assert forbidden not in "\n".join(code_cells)

    joined = "\n".join(code_cells)
    for required in [
        "def official_native_terminal_costs(",
        "def paired_model_difference_rows(",
        "def layer_screen(",
        "def fit_model_subspace(",
        "def intervention_specs_cuda(",
        "def paired_predictive_gate(",
        "CROSS_MODEL_GROUNDED_CLOSURE_CERTIFICATE_SUPPORTED",
        "NO_CROSS_MODEL_GROUNDED_CLOSURE_GENERALIZATION",
        "stage31_cross_model_certificate_result_bundle_",
    ]:
        assert required in joined, f"missing {required}"
    for prohibited in [
        "torch.autograd", ".backward(", "torch.func.jvp", "torch.func.vjp",
        "jacrev", "jacfwd",
    ]:
        assert prohibited not in joined, f"prohibited machinery: {prohibited}"

    validate_source = function_source(code_cells, "validate_world_model")
    assert 'expected_type == "dino_wm"' in validate_source
    assert 'blocks = [layer[1] for layer in layers]' in validate_source
    forward_source = function_source(code_cells, "forward_with_carriers")
    assert 'post_block = inputs[0] + output' in forward_source
    assert "modifying the branch output modifies the block" in forward_source

    physical = code_cells[6]
    assert "CONSTRUCTION_RECORDS" in physical
    assert "EVALUATION_RECORDS" in physical
    assert '"construction_evaluation_disjoint": True' in physical
    construction = code_cells[7]
    assert '"evaluation_rows_used": 0' in construction
    assert '"planning_labels_used": 0' in construction
    assert "both_subspaces_frozen_before_evaluation" in construction
    evaluation = code_cells[8]
    assert "official_terminal_planning_rows(" in evaluation
    assert "OFFICIAL_PROPRIO_ALPHA" in evaluation
    assert "diagnostic_schedules=DIAGNOSTIC_SCHEDULES" in evaluation
    decision = code_cells[9]
    assert "paired_model_difference_rows(" in decision
    assert '"primary_outcome": "DINO-WM minus JEPA-WM normalized physical regret"' in decision
    assert '"common_coordinate_system_between_carriers_claimed": False' in decision
    assert '"terminal_exhaustive_planner_not_full_closed_loop_mpc": True' in decision

    objective_source = function_source(code_cells, "official_native_terminal_costs")
    assert "visual_difference**2" in objective_source
    assert "proprio_difference**2" in objective_source
    paired_source = function_source(code_cells, "paired_model_difference_rows")
    assert "paired model panels have different" in paired_source
    assert 'float(right_row["outcome"]) - float(left_row["outcome"])' in paired_source

    observed_digest = assigned_value(tree, "NOTEBOOK_PROTOCOL_SHA256")
    sources = [source(notebook["cells"][0])]
    sources.append(config.replace(observed_digest, "__PROTOCOL_DIGEST__", 1))
    sources.extend(code_cells[1:])
    expected_digest = hashlib.sha256(
        json.dumps(sources, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    assert observed_digest == expected_digest
    validate_numerics()
    print("Stage 31 notebook validation passed")


if __name__ == "__main__":
    validate()
