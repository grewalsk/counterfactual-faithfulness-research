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
NOTEBOOK = ROOT / "30_grounded_causal_planning_value.ipynb"
BUILDER = ROOT / "build_stage30_grounded_planning_value_notebook.py"
sys.path.insert(0, str(REPOSITORY / "src"))

from cf_faithfulness.stage30_grounded_planning_value import (  # noqa: E402
    cross_fitted_incremental_value,
    diagnostic_closure_rows,
    terminal_planning_rows,
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
    rng = np.random.default_rng(30)
    target = rng.normal(size=(24, 3, 4))
    permutation = np.arange(24).reshape(4, 6)[:, ::-1].reshape(-1)
    closure = diagnostic_closure_rows(
        target, target[permutation], target, 4, 6, mode="swap"
    )
    assert len(closure) == 4
    assert all(np.isclose(row["self_coefficient"], 1.0) for row in closure)
    assert all(np.isclose(row["grounded_coefficient"], 1.0) for row in closure)

    states = np.zeros((24, 10), dtype=np.float64)
    for magnitude in range(4):
        block = slice(6 * magnitude, 6 * (magnitude + 1))
        states[block, 2] = 200 + magnitude + np.arange(6)
        states[block, 3] = 250 - np.arange(6)
        states[block, 4] = 0.02 * np.arange(6)
    planning = terminal_planning_rows(target, target, states, 4, 6)
    assert len(planning) == 8
    assert all(row["top1_correct"] == 1.0 for row in planning)

    groups = np.repeat(np.arange(50), 2)
    grounded = rng.normal(size=(len(groups), 1))
    base = rng.normal(size=(len(groups), 2))
    outcome = grounded[:, 0] + 0.02 * rng.normal(size=len(groups))
    crossfit = cross_fitted_incremental_value(
        outcome, groups, base, grounded, folds=5, seed=30
    )
    assert crossfit["relative_mse_improvement"] > 0.9


def validate():
    before = NOTEBOOK.read_bytes()
    subprocess.run(
        [sys.executable, str(BUILDER)],
        check=True,
        capture_output=True,
        env=dict(os.environ),
    )
    after = NOTEBOOK.read_bytes()
    assert before == after, "Stage 30 builder is not deterministic"

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
        assert cell["id"] == f"stage30-{index:02d}"
        assert not cell.get("outputs")
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            ast.parse(source(cell))

    config = code_cells[0]
    tree = ast.parse(config)
    assert assigned_value(tree, "PROTOCOL_ID") == (
        "stage30-grounded-causal-planning-value-v1"
    )
    assert assigned_value(tree, "RUN_MODE") == "pilot"
    assert assigned_value(tree, "EXPERIMENT_SOURCE_REF") == (
        "codex/stage30-grounded-planning-value"
    )
    assert assigned_value(tree, "EXPECTED_STAGE29_STATUS") == (
        "PHYSICAL_READOUT_LIMITATION_SUPPORTED"
    )
    assert assigned_value(tree, "EXPECTED_STAGE29_SOURCE_COMMIT") == (
        "c0fc29df0fd5a150762cbb918d5401624e780833"
    )
    assert assigned_value(tree, "EVALUATION_TRAJECTORY_TARGET") == 120
    assert assigned_value(tree, "EVALUATION_TARGET_PER_STRATUM") == 40
    assert assigned_value(tree, "SELECTED_MAGNITUDES") == [0.10, 0.14, 0.18, 0.22]
    assert assigned_value(tree, "DIAGNOSTIC_SCHEDULES") == [1, 2, 3, 4]
    assert assigned_value(tree, "PLANNING_GOAL_SCHEDULES") == [0, 5]
    assert assigned_value(tree, "SCHEDULE_INVERSION_COUNTS") == [0, 5, 10, 15, 20, 25]
    assert assigned_value(tree, "INTERVENTION_FORWARDS_PER_RECORD") == 9
    assert assigned_value(tree, "ASSET_SPECS") == {}
    assert "token_hex(4)" in config
    assert '_colab_userdata.get("HF_TOKEN")' in config
    for forbidden in [
        "STAGE30_RUN_MODE",
        "STAGE30_SOURCE_COMMIT",
        "STAGE30_RUN_NONCE",
    ]:
        assert forbidden not in "\n".join(code_cells)

    joined = "\n".join(code_cells)
    for required in [
        "def diagnostic_closure_rows(",
        "def terminal_planning_rows(",
        "def cross_fitted_incremental_value(",
        "def screen_pool(",
        "def encode_true_tokens(",
        "def intervention_specs(",
        "def predictive_gate(",
        "def ablation_control_gate(",
        "GROUNDED_CLOSURE_PREDICTS_CAUSAL_PLANNING_VALUE",
        "GROUNDING_GAP_REPLICATED_WITHOUT_PLANNING_VALUE",
        "stage30_grounded_planning_value_result_bundle_",
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
        assert prohibited not in joined, f"prohibited Stage 30 machinery: {prohibited}"

    upstream = code_cells[5]
    for required in [
        "EXPECTED_STAGE18_SUBSPACE_SHA256",
        "EXPECTED_STAGE29_DECISION_SHA256",
        "EXPECTED_STAGE29_SOURCE_SHA256",
        'decision.get("self_consistent_causal_closure_gate", {}).get("passed", False)',
        'not decision.get("grounded_causal_closure_gate", {}).get("passed", True)',
        '"stage30_states_reused_from_stage29": False',
    ]:
        assert required in upstream

    closure_source = function_source(code_cells, "diagnostic_closure_rows")
    assert "diagnostic schedules must be closed under reversal" in closure_source
    assert "ground_target" in closure_source
    planner_source = function_source(code_cells, "terminal_planning_rows")
    assert "native_terminal_costs(" in planner_source
    assert "physical_terminal_costs(" in planner_source
    selection_source = function_source(code_cells, "select_records")
    assert "ACTIVE_TARGET_PER_STRATUM" in selection_source
    evaluation = code_cells[8]
    assert "terminal_planning_rows(" in evaluation
    assert "diagnostic_schedules=DIAGNOSTIC_SCHEDULES" in evaluation
    assert '"primary_r128_ablation"' in evaluation
    assert '"shuffled_r128_ablation"' in evaluation
    decision = code_cells[9]
    assert 'predictive_gate("native_regret")' in decision
    assert 'predictive_gate("ablation_regret_change")' in decision
    assert '"closure_and_goal_contrasts_disjoint": True' in decision
    assert '"learned_decoder_used": False' in decision
    assert "DECODE_PHYSICAL_POSE" not in evaluation
    assert "physical_pose_decoder(" not in evaluation

    observed_digest = assigned_value(tree, "NOTEBOOK_PROTOCOL_SHA256")
    sources = [source(notebook["cells"][0])]
    sources.append(config.replace(observed_digest, "__PROTOCOL_DIGEST__", 1))
    sources.extend(code_cells[1:])
    expected_digest = hashlib.sha256(
        json.dumps(sources, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    assert observed_digest == expected_digest
    validate_numerics()
    print("Stage 30 notebook validation passed")


if __name__ == "__main__":
    validate()
