"""Static, protocol, and numerical validation for the Stage 37.1 notebook."""

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
NOTEBOOK = ROOT / "37_1_horizon_matched_operator_calibration.ipynb"
BUILDER = ROOT / "build_stage37_1_horizon_control_notebook.py"
NUMERICAL = REPOSITORY / "src/cf_faithfulness/stage37_1_horizon_control.py"
TESTS = REPOSITORY / "tests/test_stage37_1_horizon_control.py"
UPSTREAM_TESTS = REPOSITORY / "tests/test_stage37_semigroup_pscd.py"
GUIDE = REPOSITORY / "STAGE37_1_RUN_GUIDE.md"


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
    assert replaced, "could not reconstruct Stage 37.1 protocol digest"
    expected = hashlib.sha256(
        json.dumps(sources, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    assert observed == expected, "Stage 37.1 protocol digest is stale"


def direct_calls(code_cells, function_name):
    calls = []
    for index, text in enumerate(code_cells):
        for node in ast.walk(ast.parse(text)):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == function_name
            ):
                calls.append((index, node))
    return calls


def validate():
    for path in [NOTEBOOK, BUILDER, NUMERICAL, TESTS, UPSTREAM_TESTS, GUIDE]:
        assert path.is_file(), f"missing Stage 37.1 artifact: {path}"

    before = NOTEBOOK.read_bytes()
    subprocess.run(
        [sys.executable, str(BUILDER)], check=True, capture_output=True,
        cwd=REPOSITORY, env=dict(os.environ),
    )
    after = NOTEBOOK.read_bytes()
    assert before == after, "Stage 37.1 builder is not deterministic"

    notebook = json.loads(after)
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] == 5
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["metadata"]["colab"]["gpuType"] == "L4"
    assert len(notebook["cells"]) == 12
    assert source(notebook["cells"][0]).startswith(
        "# Stage 37.1: horizon-matched true-state operator calibration\n"
    )
    assert [cell["id"] for cell in notebook["cells"]] == [
        f"stage371-{index:02d}" for index in range(12)
    ]
    expected_headers = [
        "# SINGLE CONFIGURATION BLOCK — no Stage 37.1 secrets required.",
        "import subprocess",
        "import csv",
        "# Tested predictive-state adapter, controls, metrics, and decision gates.",
        "def to_model_observation(visual, proprio):",
        "# Freeze trajectory families and action compositions before simulator or model access.",
        "# Select complete physical trajectories and materialize exact multi-step truth without model access.",
        "# Select a horizon-matched true-state operator without loading JEPA-WM.",
        "# Freeze primary and matched horizon controls before locked evaluation.",
        "# Open fresh horizon-matched evaluation once and derive every Stage 37.1 gate.",
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

    provenance = direct_calls(code_cells, "verify_executed_notebook_through")
    assert [(index, node.args[0].value) for index, node in provenance] == [
        (6, expected_headers[6]),
        (7, expected_headers[7]),
        (8, expected_headers[8]),
        (9, expected_headers[9]),
    ]

    namespace = {}
    exec(compile(code_cells[0], "<stage37.1-config>", "exec"), namespace)
    assert namespace["RUN_MODE"] == "pilot"
    assert namespace["PROTOCOL_ID"] == (
        "stage37-1-horizon-matched-operator-calibration-v1"
    )
    assert namespace["SEMIGROUP_HORIZONS"] == [2, 4, 8]
    assert namespace["SIMULATOR_LATENT_DIMS"] == [128, 256]
    assert namespace["SIMULATOR_DYNAMICS"] == ["single", "mixture"]
    assert namespace["CONSTRUCTION_TRAJECTORIES"] == 16
    assert namespace["MODEL_SELECTION_TRAJECTORIES"] == 16
    assert namespace["CALIBRATION_TRAJECTORIES"] == 16
    assert namespace["EVALUATION_TRAJECTORIES"] == 32
    assert namespace["MAX_LOCKED_PHYSICAL_NMSE"] == 0.25
    assert namespace["MAX_LOCKED_SEMIGROUP_NMSE"] == 0.25
    assert namespace["MIN_CONTROL_GAIN"] == 0.50
    assert namespace["MIN_OBJECTIVE_ADVANTAGE"] == 0.05

    pools = [
        namespace[name] for name in [
            "CONSTRUCTION_TRAJECTORY_POOL", "MODEL_SELECTION_TRAJECTORY_POOL",
            "CALIBRATION_TRAJECTORY_POOL", "EVALUATION_TRAJECTORY_POOL",
        ]
    ]
    assert [(min(pool), max(pool)) for pool in pools] == [
        (48000, 49599), (49600, 51199),
        (51200, 52799), (52800, 55999),
    ]
    assert all(
        set(pools[left]).isdisjoint(pools[right])
        for left in range(4) for right in range(left + 1, 4)
    )
    word_banks = namespace["TASK_WORD_BANKS"]
    assert all(len(bank) == 8 for bank in word_banks)
    assert all({len(word) for word in bank} == {9, 10, 11, 12} for bank in word_banks)
    assert all(
        set(word_banks[left]).isdisjoint(word_banks[right])
        for left in range(4) for right in range(left + 1, 4)
    )
    assert set().union(*(set(word) for bank in word_banks for word in bank)) <= {
        "A", "B", "C", "D"
    }
    digest = namespace["NOTEBOOK_PROTOCOL_SHA256"]
    assert len(digest) == 64
    validate_protocol_digest(notebook, digest)

    analysis_namespace = {"np": np, "hashlib": hashlib}
    exec(compile(code_cells[3], "<stage37.1-analysis>", "exec"), analysis_namespace)
    assert "stable_seed" in analysis_namespace
    selection_tree = ast.parse(code_cells[7])
    seed_call = next(
        node for node in ast.walk(selection_tree)
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
        compile(ast.Expression(seed_call), "<stage37.1-selection-seed>", "eval"),
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

    all_code = "\n".join(code_cells)
    required = [
        "select_horizon_control_candidate(CONTROL_SELECTION_ROWS)",
        "fit_semigroup_predictive_state_closure(",
        "semigroup_horizons=SEMIGROUP_HORIZONS",
        "semigroup_weight=SEMIGROUP_WEIGHT",
        "semigroup_weight=0.0, free_weight=1.0",
        "semigroup_weight=0.0, free_weight=0.0",
        'matched_seed = stable_seed(CALIBRATION_SEED, "matched_horizon_controls")',
        'load_stage37_1_physical_sequences("evaluation")',
        "pretrained_forward_count == 0",
        'int(PROVENANCE_COUNTS["patched_forwards"]) == 0',
        '"same_horizon_distribution": True',
        '"jepa_result_claimed": False',
        '"planning_result_claimed": False',
        "stage37_1_hmoc_result_bundle_",
        "retry_drive_io(",
        "fetch_url_bytes(",
    ]
    for fragment in required:
        assert fragment in all_code, f"missing Stage 37.1 fragment: {fragment}"
    forbidden = [
        'load_world_model("jepa_wm_pusht")',
        'load_world_model("dino_wm_pusht")',
        '"jepa_result_claimed": True',
        '"planning_result_claimed": True',
        '"causal_evidence_claimed": True',
        "USE_SYNTHETIC_FALLBACK = True",
    ]
    for fragment in forbidden:
        assert fragment not in all_code, f"forbidden Stage 37.1 fragment: {fragment}"
    assert not direct_calls(code_cells, "load_world_model")
    assert len(direct_calls(code_cells, "fit_semigroup_predictive_state_closure")) == 4

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPOSITORY / "src")
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(TESTS), str(UPSTREAM_TESTS)],
        check=True, cwd=REPOSITORY, env=environment,
    )
    print("Stage 37.1 notebook validation passed")


if __name__ == "__main__":
    validate()
