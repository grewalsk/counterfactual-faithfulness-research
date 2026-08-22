"""Static, protocol, and numerical validation for the Stage 38 notebook."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
NOTEBOOK = ROOT / "38_cross_model_pscd_confirmation.ipynb"
BUILDER = ROOT / "build_stage38_cross_model_pscd_notebook.py"
NUMERICAL = REPOSITORY / "src/cf_faithfulness/stage38_cross_model_pscd.py"
TESTS = REPOSITORY / "tests/test_stage38_cross_model_pscd.py"
UPSTREAM_TESTS = [
    REPOSITORY / "tests/test_stage37_semigroup_pscd.py",
    REPOSITORY / "tests/test_stage36_predictive_state_closure.py",
]
GUIDE = REPOSITORY / "STAGE38_RUN_GUIDE.md"


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
    assert replaced, "could not reconstruct Stage 38 protocol digest"
    expected = hashlib.sha256(
        json.dumps(sources, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    assert observed == expected, "Stage 38 protocol digest is stale"


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
    for path in [NOTEBOOK, BUILDER, NUMERICAL, TESTS, GUIDE, *UPSTREAM_TESTS]:
        assert path.is_file(), f"missing Stage 38 artifact: {path}"

    before = NOTEBOOK.read_bytes()
    subprocess.run(
        [sys.executable, str(BUILDER)], check=True, capture_output=True,
        cwd=REPOSITORY, env=dict(os.environ),
    )
    after = NOTEBOOK.read_bytes()
    assert before == after, "Stage 38 builder is not deterministic"

    notebook = json.loads(after)
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] == 5
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["metadata"]["colab"]["gpuType"] == "L4"
    assert len(notebook["cells"]) == 14
    assert source(notebook["cells"][0]).startswith(
        "# Stage 38: cross-model predictive-state closure confirmation\n"
    )
    assert [cell["id"] for cell in notebook["cells"]] == [
        f"stage38-{index:02d}" for index in range(14)
    ]
    expected_headers = [
        "# SINGLE CONFIGURATION BLOCK — no Stage 38 secrets required.",
        "import subprocess",
        "import csv",
        "# Tested predictive-state adapter, controls, metrics, and decision gates.",
        "def to_model_observation(visual, proprio):",
        "# Freeze trajectory families and action compositions before simulator or model access.",
        "# Select complete physical trajectories and materialize exact multi-step truth without model access.",
        "# Reconfirm the horizon-matched true-state operator before loading either checkpoint.",
        "# Freeze both grounded readouts and materialize non-evaluation carrier paths.",
        "# Select semigroup strength independently for each representation.",
        "# Freeze all matched-seed models and scales before opening evaluation.",
        "# Open closure first; open planning only after both representation panels pass.",
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
        (10, expected_headers[10]),
        (11, expected_headers[11]),
    ]

    namespace = {}
    exec(compile(code_cells[0], "<stage38-config>", "exec"), namespace)
    assert namespace["RUN_MODE"] == "pilot"
    assert namespace["PROTOCOL_ID"] == "stage38-cross-model-pscd-confirmation-v1"
    assert namespace["MODEL_NAMES"] == ["jepa_wm_pusht", "dino_wm_pusht"]
    assert namespace["SEMIGROUP_HORIZONS"] == [2, 4, 8]
    assert namespace["SEMIGROUP_WEIGHTS"] == [0.5, 1.0, 2.0]
    assert namespace["FINAL_TRAINING_SEEDS"] == [3801, 3802, 3803]
    assert namespace["FIXED_CARRIER_DIM"] == 256
    assert namespace["FIXED_HISTORY_LENGTH"] == 4
    assert namespace["FIXED_LATENT_DIM"] == 256
    assert namespace["FIXED_DYNAMICS"] == "mixture"
    assert [
        namespace[name] for name in [
            "CONSTRUCTION_TRAJECTORIES", "MODEL_SELECTION_TRAJECTORIES",
            "CALIBRATION_TRAJECTORIES", "EVALUATION_TRAJECTORIES",
        ]
    ] == [24, 16, 24, 32]
    assert namespace["MAX_RECURSIVE_PHYSICAL_NMSE"] == 0.25
    assert namespace["MAX_P95_PHYSICAL_NMSE"] == 0.35
    assert namespace["MAX_CATASTROPHIC_RATE"] == 0.02

    pools = [
        namespace[name] for name in [
            "CONSTRUCTION_TRAJECTORY_POOL", "MODEL_SELECTION_TRAJECTORY_POOL",
            "CALIBRATION_TRAJECTORY_POOL", "EVALUATION_TRAJECTORY_POOL",
        ]
    ]
    assert [(min(pool), max(pool)) for pool in pools] == [
        (56000, 57999), (58000, 59999), (60000, 61999), (62000, 65999)
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
    assert set(namespace["PLANNING_WORD_NAMES"]).isdisjoint(
        set().union(*map(set, word_banks))
    )
    digest = namespace["NOTEBOOK_PROTOCOL_SHA256"]
    assert len(digest) == 64
    validate_protocol_digest(notebook, digest)

    all_code = "\n".join(code_cells)
    required = [
        'MODEL_NAMES = ["jepa_wm_pusht", "dino_wm_pusht"]',
        'train["native"], train["mask"], history_length=FIXED_HISTORY_LENGTH',
        'data["native"], data["mask"], history_length=FIXED_HISTORY_LENGTH',
        'result["physical"], data["native"], valid',
        'for variant in ["one_step", "pscd", "overshoot", "spscd"]',
        'FINAL_TRAINING_SEEDS = [3801, 3802, 3803]',
        '"matched_seed_across_variants": True',
        '"evaluation_statistics_read": False',
        '"checkpoint_parameters_updated": False',
        '"evaluation_closure": CLOSURE_EVALUATION_WORD_NAMES',
        '"evaluation_planning": PLANNING_WORD_NAMES',
        "if closure_panels_passed:",
        "hierarchical_seed_trajectory_interval(",
        "tail_risk_summary(",
        'PROVENANCE_COUNTS["patched_forwards"] == 0',
        "derive_stage38_model_decision(",
        "derive_stage38_decision(Stage38Gates(",
        "stage38_xmpscd_result_bundle_",
        "retry_drive_io(",
        "fetch_url_bytes(",
    ]
    for fragment in required:
        assert fragment in all_code, f"missing Stage 38 fragment: {fragment}"
    forbidden = [
        'train["simulator"], train["mask"], history_length=FIXED_HISTORY_LENGTH',
        'data["simulator"], data["mask"], history_length=FIXED_HISTORY_LENGTH',
        '"checkpoint_parameters_updated": True',
        '"native_checkpoint_closure_claimed": True',
        '"cross_environment_claimed": True',
        '"causal_evidence_claimed": True',
        "np.cross(",
        "USE_SYNTHETIC_FALLBACK = True",
    ]
    for fragment in forbidden:
        assert fragment not in all_code, f"forbidden Stage 38 fragment: {fragment}"

    load_calls = direct_calls(code_cells, "load_world_model")
    assert [index for index, _ in load_calls] == [8, 11, 11]
    assert all(index >= 8 for index, _ in load_calls)
    assert len(direct_calls(code_cells, "fit_weighted_semigroup_predictive_state_closure")) == 4

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPOSITORY / "src")
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(TESTS),
         *map(str, UPSTREAM_TESTS)],
        check=True, cwd=REPOSITORY, env=environment,
    )
    print("Stage 38 notebook validation passed")


if __name__ == "__main__":
    validate()
