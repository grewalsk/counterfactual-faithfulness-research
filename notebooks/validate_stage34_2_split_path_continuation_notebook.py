"""Static and numerical validation for the Stage 34.2 Colab notebook."""

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
NOTEBOOK = ROOT / "34_2_split_path_continuation.ipynb"
BUILDER = ROOT / "build_stage34_2_split_path_continuation_notebook.py"
NUMERICAL = REPOSITORY / "src/cf_faithfulness/stage34_2_split_path_continuation.py"
TESTS = REPOSITORY / "tests/test_stage34_2_split_path_continuation.py"


def source(cell):
    return "".join(cell.get("source", []))


def static_value(node):
    try:
        return ast.literal_eval(node)
    except (TypeError, ValueError):
        pass
    raise AssertionError(f"unsupported static expression: {ast.dump(node)}")


def assigned_node(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node.value
    raise AssertionError(f"missing assignment {name}")


def assigned_value(tree, name):
    return static_value(assigned_node(tree, name))


def validate_protocol_digest(notebook, observed):
    sources = [source(cell).strip() for cell in notebook["cells"]]
    replaced = False
    for index, text in enumerate(sources):
        if observed in text and "NOTEBOOK_PROTOCOL_SHA256" in text:
            sources[index] = text.replace(observed, "__PROTOCOL_DIGEST__", 1)
            replaced = True
            break
    assert replaced, "could not reconstruct Stage 34.2 protocol digest"
    expected = hashlib.sha256(
        json.dumps(sources, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    assert observed == expected, "Stage 34.2 protocol digest is stale"


def validate():
    for path in [NOTEBOOK, BUILDER, NUMERICAL, TESTS]:
        assert path.is_file(), f"missing Stage 34.2 artifact: {path}"
    before = NOTEBOOK.read_bytes()
    subprocess.run(
        [sys.executable, str(BUILDER)], check=True, capture_output=True,
        env=dict(os.environ),
    )
    after = NOTEBOOK.read_bytes()
    assert before == after, "Stage 34.2 builder is not deterministic"

    notebook = json.loads(after)
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] == 5
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["metadata"]["colab"]["gpuType"] == "L4"
    assert len(notebook["cells"]) == 13
    assert source(notebook["cells"][0]).startswith(
        "# Stage 34.2: split-path predictive and causal continuation\n"
    )
    assert [cell["id"] for cell in notebook["cells"]] == [
        f"stage342-{index:02d}" for index in range(13)
    ]
    code_cells = [source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 12
    expected_headers = [
        "# Frozen Stage 34.2 contract. Pilot is diagnostic evidence; smoke is plumbing only.",
        "# Mount Drive, resolve committed source, and initialize one resumable output directory.",
        "# Bind exact Stage 34 raw data and the exact Stage 34.1 diagnostic decision.",
        "# Load no-op-corrected response rows and frozen transition rows from the bound inputs.",
        "# Diagnose DINO with calibration-only diagonal scale/bias; full affine is descriptive only.",
        "# Apply the original Stage 34 predictive-sufficiency gate to JEPA only.",
        "# Install and configure the exact official JEPA runtime only when sufficiency passes.",
        "# Define the exact Stage 34 JEPA hooks and an upstream-truth input adapter.",
        "# Freeze JEPA matched fiber/state pairs and carrier subspaces before live checkpoint inference.",
        "# Run or resume native JEPA interventions, validating unpatched replay for every pair.",
        "# Derive the split-path decision without reviving the rejected shared-model claim.",
        "# Package all compact diagnostics and resumable causal shards.",
    ]
    assert [cell.splitlines()[0] for cell in code_cells] == expected_headers
    for cell in notebook["cells"]:
        assert not cell.get("outputs")
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            ast.parse(source(cell))

    tree = ast.parse(code_cells[0])
    expected = {
        "RUN_MODE": "pilot",
        "PROTOCOL_ID": "stage34.2-split-path-predictive-causal-continuation-v1",
        "EVIDENCE_STATUS": "POST_OUTCOME_SPLIT_PATH_DIAGNOSTIC_NOT_CONFIRMATION",
        "UPSTREAM_STAGE34_RUN_SIGNATURE": "d3f4f88426afff4d964bb4f1f1556c94ec3613b667edd9403ddfcd0fd78ded84",
        "UPSTREAM_STAGE341_RUN_SIGNATURE": "208b72f570749a48719ddf22d693cde5ee0fd2c8b525021506d356cd6242a8ac",
        "UPSTREAM_STAGE341_MANIFEST_SHA256": "fb1b90ca2ed8e3652fcc8f9e09282f6fa94351f923ad3b9d0a3094747e6aaa67",
        "CANONICAL_RANK": 5,
        "MAX_RESIDUAL_RELATIVE_IMPROVEMENT": 0.05,
        "MAX_RESIDUAL_CI_UPPER": 0.10,
        "MIN_DELETION_CONTROL_IMPROVEMENT": 0.10,
        "MIN_DINO_CALIBRATION_GAIN": 0.10,
        "MIN_DINO_CONTROL_ADVANTAGE": 0.10,
        "MAX_FIBER_EFFECT_RATIO": 1.25,
        "MIN_STATE_EFFECT_RETENTION": 0.50,
        "MIN_STATE_INTERVENTION_COSINE": 0.20,
        "MAX_INTERVENTION_OOD_RATE": 0.05,
        "MIN_CAUSAL_CONTROL_ADVANTAGE": 0.10,
        "MAX_REPLAY_ABS_ERROR": 5e-4,
    }
    for name, value in expected.items():
        assert assigned_value(tree, name) == value, f"unexpected {name}"
    for name, pilot_value in [
        ("BOOTSTRAP_DRAWS", 2000),
        ("STAGE34_BOOTSTRAP_DRAWS", 5000),
        ("TRANSITION_RANDOM_FEATURES", 256),
    ]:
        node = assigned_node(tree, name)
        assert isinstance(node, ast.IfExp) and static_value(node.body) == pilot_value

    digest = assigned_value(tree, "NOTEBOOK_PROTOCOL_SHA256")
    assert len(digest) == 64
    validate_protocol_digest(notebook, digest)

    all_code = "\n".join(code_cells)
    required = [
        'stack_dino_split("calibration", CORE_WORD_NAMES)',
        'stack_dino_split("evaluation", EVALUATION_WORD_NAMES)',
        "fit_grouped_diagonal_affine(",
        "DINO_DIAGONAL_RECOVERABILITY_GATE",
        'load_transition_rows("model_selection")',
        "JEPA_PREDICTIVE_SUFFICIENCY_GATE",
        "if JEPA_PREDICTIVE_SUFFICIENCY_GATE:",
        "matched_fiber_pairs(",
        "full_swap_positive",
        "random_matched_subspace",
        "MAX_REPLAY_ABS_ERROR",
        "jepa_causal_progress.json",
        '"confirmation_eligible": False',
        '"shared_abstraction_claimed": False',
    ]
    for fragment in required:
        assert fragment in all_code, f"missing Stage 34.2 contract fragment: {fragment}"
    forbidden = [
        'load_world_model("dino_wm_pusht")',
        'for short in ["jepa", "dino"]',
        "bounded_two_sided_causal_abstraction_supported",
        "planning_run\": true",
    ]
    for fragment in forbidden:
        assert fragment not in all_code, f"forbidden Stage 34.2 fragment: {fragment}"

    dino = code_cells[4]
    assert "DIAGONAL_MODEL" in dino and "FULL_AFFINE_MODEL" in dino
    assert "DINO_DIAGONAL_RECOVERABILITY_GATE" in dino
    assert "full_affine" not in dino.split("DINO_DIAGONAL_RECOVERABILITY_GATE = bool(", 1)[1].split(")\n", 1)[0]

    sufficiency = code_cells[5]
    assert "draws=STAGE34_BOOTSTRAP_DRAWS" in sufficiency
    assert "residual_ci[1] <= MAX_RESIDUAL_CI_UPPER" in sufficiency
    assert "deletion_ci[0] > 0" in sufficiency

    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(TESTS)],
        check=True,
        cwd=REPOSITORY,
        env={
            **os.environ,
            "PYTHONPATH": str(REPOSITORY / "src"),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        },
    )


if __name__ == "__main__":
    validate()
    print("Stage 34.2 notebook validation passed")
