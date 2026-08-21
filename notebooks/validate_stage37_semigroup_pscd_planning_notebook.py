"""Static, protocol, and numerical validation for the Stage 37 notebook."""

from __future__ import annotations

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
NOTEBOOK = ROOT / "37_semigroup_pscd_planning_value.ipynb"
BUILDER = ROOT / "build_stage37_semigroup_pscd_planning_notebook.py"
NUMERICAL = REPOSITORY / "src/cf_faithfulness/stage37_semigroup_pscd.py"
TESTS = REPOSITORY / "tests/test_stage37_semigroup_pscd.py"
GUIDE = REPOSITORY / "STAGE37_RUN_GUIDE.md"


def source(cell):
    return "".join(cell.get("source", []))


def validate_protocol_digest(notebook, observed):
    sources = [source(cell).strip() for cell in notebook["cells"]]
    replaced = False
    for index, text in enumerate(sources):
        if observed in text and "NOTEBOOK_PROTOCOL_SHA256" in text:
            sources[index] = text.replace(observed, "__PROTOCOL_DIGEST__", 1)
            replaced = True
            break
    assert replaced, "could not reconstruct Stage 37 protocol digest"
    expected = hashlib.sha256(
        json.dumps(sources, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    assert observed == expected, "Stage 37 protocol digest is stale"


def provenance_calls(code_cells):
    calls = []
    for index, text in enumerate(code_cells):
        for node in ast.walk(ast.parse(text)):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "verify_executed_notebook_through"
            ):
                continue
            assert len(node.args) == 1 and isinstance(node.args[0], ast.Constant)
            calls.append((index, node.args[0].value))
    return calls


def validate():
    for path in [NOTEBOOK, BUILDER, NUMERICAL, TESTS, GUIDE]:
        assert path.is_file(), f"missing Stage 37 artifact: {path}"
    before = NOTEBOOK.read_bytes()
    subprocess.run(
        [sys.executable, str(BUILDER)], check=True, capture_output=True,
        cwd=REPOSITORY, env=dict(os.environ),
    )
    after = NOTEBOOK.read_bytes()
    assert before == after, "Stage 37 builder is not deterministic"

    notebook = json.loads(after)
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] == 5
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["metadata"]["colab"]["gpuType"] == "L4"
    assert len(notebook["cells"]) == 14
    assert source(notebook["cells"][0]).startswith(
        "# Stage 37: semigroup-regularized PSCD and open-loop planning value\n"
    )
    assert [cell["id"] for cell in notebook["cells"]] == [
        f"stage37-{index:02d}" for index in range(14)
    ]
    expected_headers = [
        "# SINGLE CONFIGURATION BLOCK — no Stage 37 secrets required.",
        "import subprocess",
        "import csv",
        "# Tested predictive-state adapter, controls, metrics, and decision gates.",
        "def to_model_observation(visual, proprio):",
        "# Freeze trajectory families and action compositions before simulator or model access.",
        "# Select complete physical trajectories and materialize exact multi-step truth without model access.",
        "# Prove the operator class on true Markov state before loading JEPA-WM.",
        "# Fit the construction-only grounded JEPA readout and save every native prefix carrier.",
        "# Load split-bound teacher sequences without opening evaluation statistics.",
        "# Freeze S-PSCD, capacity controls, and the physical control before evaluation.",
        "# Open fresh closure and planning evaluation once and derive every Stage 37 gate.",
        "# Package compact audit evidence while retaining the complete resumable Drive directory.",
    ]
    code_cells = [
        source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    assert [text.splitlines()[0] for text in code_cells] == expected_headers
    for cell in notebook["cells"]:
        assert not cell.get("outputs")
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            ast.parse(source(cell))

    assert provenance_calls(code_cells) == [
        (6, expected_headers[6]),
        (7, expected_headers[7]),
        (8, expected_headers[8]),
        (9, expected_headers[9]),
        (10, expected_headers[10]),
        (11, expected_headers[11]),
    ]

    namespace = {}
    exec(compile(code_cells[0], "<stage37-config>", "exec"), namespace)
    assert namespace["RUN_MODE"] == "pilot"
    assert namespace["PROTOCOL_ID"] == "stage37-semigroup-pscd-planning-v2"
    assert namespace["V2_PREFLIGHT_HELPER_ORDER_AMENDMENT"] is True
    assert namespace["V2_V1_SCIENTIFIC_OUTCOMES_OBSERVED"] is False
    assert namespace["V2_V1_JEPA_LOADED"] is False
    assert namespace["V2_V1_LOCKED_EVALUATION_OPENED"] is False
    assert namespace["FIXED_CARRIER_DIM"] == 256
    assert namespace["FIXED_HISTORY_LENGTH"] == 4
    assert namespace["FIXED_LATENT_DIM"] == 128
    assert namespace["FIXED_DYNAMICS"] == "mixture"
    assert namespace["SEMIGROUP_HORIZONS"] == [2, 4, 8]
    assert namespace["SEMIGROUP_WEIGHTS"] == [0.25, 1.0, 2.0]
    assert namespace["SIMULATOR_LATENT_DIMS"] == [128, 256]
    assert namespace["SIMULATOR_DYNAMICS"] == ["single", "mixture"]
    assert namespace["EVALUATION_TRAJECTORIES"] == 24
    pools = [
        namespace[name] for name in [
            "CONSTRUCTION_TRAJECTORY_POOL", "MODEL_SELECTION_TRAJECTORY_POOL",
            "CALIBRATION_TRAJECTORY_POOL", "EVALUATION_TRAJECTORY_POOL",
        ]
    ]
    assert min(pools[0]) == 40000 and max(pools[-1]) == 47999
    assert all(
        not set(pools[left]) & set(pools[right])
        for left in range(4) for right in range(left + 1, 4)
    )
    assert set(namespace["MODEL_SELECTION_WORD_NAMES"]).isdisjoint(
        namespace["CONSTRUCTION_WORD_NAMES"]
    )
    assert {len(value) for value in namespace["MODEL_SELECTION_WORD_NAMES"]} == {
        5, 6, 7, 8
    }
    assert {len(value) for value in namespace["CLOSURE_EVALUATION_WORD_NAMES"]} == {
        9, 10, 11, 12
    }
    assert len(namespace["PLANNING_WORD_NAMES"]) == 12
    assert {len(value) for value in namespace["PLANNING_WORD_NAMES"]} == {10}
    assert set().union(*map(set, namespace["EVALUATION_WORD_NAMES_REGISTERED"])) <= {
        "A", "B", "C", "D"
    }
    digest = namespace["NOTEBOOK_PROTOCOL_SHA256"]
    assert len(digest) == 64
    validate_protocol_digest(notebook, digest)

    # Execute the generated analysis cell in the same preceding namespace and
    # evaluate the exact seed call used by the next simulator-preflight cell.
    # V1's numerical tests passed while this ordered namespace was incomplete.
    analysis_namespace = {"np": np, "hashlib": hashlib}
    exec(compile(code_cells[3], "<stage37-analysis>", "exec"), analysis_namespace)
    assert "stable_seed" in analysis_namespace
    preflight_tree = ast.parse(code_cells[7])
    seed_call = next(
        node for node in ast.walk(preflight_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "stable_seed"
        and len(node.args) == 4
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "simulator"
    )
    ordered_namespace = {
        **analysis_namespace,
        "CONTROL_SEED": namespace["CONTROL_SEED"],
        "latent_dim": namespace["SIMULATOR_LATENT_DIMS"][0],
        "dynamics": namespace["SIMULATOR_DYNAMICS"][0],
    }
    observed_seed = eval(
        compile(ast.Expression(seed_call), "<stage37-preflight-seed>", "eval"),
        ordered_namespace,
    )
    expected_seed = int.from_bytes(
        hashlib.sha256(
            f'{namespace["CONTROL_SEED"]}:simulator:'
            f'{namespace["SIMULATOR_LATENT_DIMS"][0]}:'
            f'{namespace["SIMULATOR_DYNAMICS"][0]}'.encode()
        ).digest()[:4],
        "little",
    )
    assert observed_seed == expected_seed
    stable_definition_cells = [
        index for index, text in enumerate(code_cells)
        if any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "stable_seed"
            for node in ast.walk(ast.parse(text))
        )
    ]
    assert stable_definition_cells and min(stable_definition_cells) < 7

    all_code = "\n".join(code_cells)
    required = [
        'load_world_model("jepa_wm_pusht")',
        "registered_semigroup_horizons(",
        "fit_semigroup_predictive_state_closure(",
        "rollout_predictive_state_from_initial(",
        "semigroup_horizons=SEMIGROUP_HORIZONS",
        "semigroup_weight=0.0",
        'matched_objective_seed = stable_seed(CALIBRATION_SEED, "matched_objective")',
        "SIMULATOR_PREFLIGHT_PASSED",
        '"jepa_loaded_before_decision": False',
        'if not PIPELINE_FAILED and SIMULATOR_PREFLIGHT_PASSED:',
        'np.isin(all_evaluation["word"], CLOSURE_EVALUATION_WORD_NAMES)',
        'np.isin(all_evaluation["word"], PLANNING_WORD_NAMES)',
        'stable_seed(DESIGN_SEED, "planning_goal", int(record_id))',
        "grouped_planner_metrics(",
        "MAX_COMPOSITION_DISCREPANCY_NMSE",
        "MAX_SEMIGROUP_NMSE",
        '"closed_loop_planning_claimed": False',
        '"native_jepa_mechanism_claimed": False',
        '"minimal_state_claimed": False',
        "retry_drive_io(",
        "fetch_url_bytes(",
        "stage37_spscd_result_bundle_",
        "V2_PREFLIGHT_HELPER_ORDER_AMENDMENT = True",
    ]
    for fragment in required:
        assert fragment in all_code, f"missing Stage 37 fragment: {fragment}"
    forbidden = [
        'load_world_model("dino_wm_pusht")',
        '"closed_loop_planning_claimed": True',
        '"native_jepa_mechanism_claimed": True',
        '"causal_evidence_claimed": True',
        "USE_SYNTHETIC_FALLBACK = True",
        'planning["carrier"], planning["mask"]',
        'issubset({"A", "B"})',
        "stage37_truth_path(",
    ]
    for fragment in forbidden:
        assert fragment not in all_code, f"forbidden Stage 37 fragment: {fragment}"

    # The positive control must be evaluated in the code cell before the first
    # JEPA load. The only model loads are in construction and locked evaluation.
    assert 'load_world_model("jepa_wm_pusht")' not in code_cells[7]
    assert 'load_world_model("jepa_wm_pusht")' in code_cells[8]
    assert 'load_world_model("jepa_wm_pusht")' in code_cells[11]
    assert all(
        'load_world_model("jepa_wm_pusht")' not in code_cells[index]
        for index in range(8)
    )

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPOSITORY / "src")
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(TESTS)],
        check=True, cwd=REPOSITORY, env=environment,
    )
    print("Stage 37 notebook validation passed")


if __name__ == "__main__":
    validate()
