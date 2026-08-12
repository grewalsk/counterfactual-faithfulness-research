"""Static and numerical validation for the Stage 34.3 Colab notebook."""

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
NOTEBOOK = ROOT / "34_3_regime_innovation_diagnostic.ipynb"
BUILDER = ROOT / "build_stage34_3_regime_innovation_notebook.py"
NUMERICAL = REPOSITORY / "src/cf_faithfulness/stage34_3_regime_innovation.py"
TESTS = REPOSITORY / "tests/test_stage34_3_regime_innovation.py"


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
    assert replaced, "could not reconstruct Stage 34.3 protocol digest"
    expected = hashlib.sha256(
        json.dumps(sources, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    assert observed == expected, "Stage 34.3 protocol digest is stale"


def validate():
    for path in [NOTEBOOK, BUILDER, NUMERICAL, TESTS]:
        assert path.is_file(), f"missing Stage 34.3 artifact: {path}"
    before = NOTEBOOK.read_bytes()
    subprocess.run(
        [sys.executable, str(BUILDER)], check=True, capture_output=True,
        env=dict(os.environ),
    )
    after = NOTEBOOK.read_bytes()
    assert before == after, "Stage 34.3 builder is not deterministic"

    notebook = json.loads(after)
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] == 5
    assert "accelerator" not in notebook["metadata"]
    assert len(notebook["cells"]) == 10
    assert source(notebook["cells"][0]).startswith(
        "# Stage 34.3: regime-aware JEPA innovation diagnostic\n"
    )
    assert [cell["id"] for cell in notebook["cells"]] == [
        f"stage343-{index:02d}" for index in range(10)
    ]
    code_cells = [source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 9
    expected_headers = [
        "# Frozen Stage 34.3 contract. Pilot is diagnostic evidence; smoke is plumbing only.",
        "# Mount Drive, resolve committed source, and initialize one deterministic output directory.",
        "# Hash-bind the exact Stage 34 raw run and the exact Stage 34.2 stopped decision.",
        "# Materialize the frozen JEPA transition table without model loading or simulator reruns.",
        "# Select rank, innovation width, mode structure, and ridge before evaluation is touched.",
        "# Refit the frozen candidate on calibration and evaluate once on long, unseen words.",
        "# Test residual sufficiency, every-coordinate necessity, and matched mode specificity.",
        "# Derive the bounded diagnostic decision; no branch can claim causality or confirmation.",
        "# Package compact evidence and immutable source identities.",
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
        "PROTOCOL_ID": "stage34.3-regime-aware-jepa-innovation-diagnostic-v1",
        "EVIDENCE_STATUS": "POST_OUTCOME_CPU_DIAGNOSTIC_NOT_CONFIRMATION",
        "UPSTREAM_STAGE34_RUN_SIGNATURE": "d3f4f88426afff4d964bb4f1f1556c94ec3613b667edd9403ddfcd0fd78ded84",
        "UPSTREAM_STAGE342_RUN_SIGNATURE": "9fcedcf036a83c2ed234a39c62b4f4a7c2535dca7b0ba56cb0dd4c848c89ddb6",
        "UPSTREAM_STAGE342_MANIFEST_SHA256": "8c203828867eb88b19dd6eb8af578f84e8a463dceb69ee8b469203d9d11de3b7",
        "UPSTREAM_STAGE342_DECISION_SHA256": "514231d1faba4ec113e0e7ff865ddb89fa721f95cda8a662819959d660d9e792",
        "CANDIDATE_STATE_RANKS": [4, 5],
        "CANDIDATE_REGIMES": ["universal", "physical_mode"],
        "SELECTION_RELATIVE_TOLERANCE": 0.02,
        "MIN_SELECTION_IMPROVEMENT": 0.05,
        "MIN_EVALUATION_IMPROVEMENT": 0.05,
        "MAX_EXTRA_RESIDUAL_IMPROVEMENT": 0.05,
        "MAX_EXTRA_RESIDUAL_CI_UPPER": 0.10,
        "MIN_COORDINATE_NECESSITY": 0.02,
        "MIN_MODE_CONTROL_ADVANTAGE": 0.05,
        "CANONICAL_RANK": 5,
        "STATE_CARRIER_SKETCH_DIM": 64,
    }
    for name, value in expected.items():
        assert assigned_value(tree, name) == value, f"unexpected {name}"
    for name, pilot_value in [
        ("BOOTSTRAP_DRAWS", 5000),
        ("FOLDS", 4),
        ("RFF_WIDTH", 128),
    ]:
        node = assigned_node(tree, name)
        assert isinstance(node, ast.IfExp) and static_value(node.body) == pilot_value
    innovation_node = assigned_node(tree, "CANDIDATE_INNOVATION_RANKS")
    assert isinstance(innovation_node, ast.IfExp)
    assert static_value(innovation_node.body) == [0, 1, 2, 3]

    digest = assigned_value(tree, "NOTEBOOK_PROTOCOL_SHA256")
    assert len(digest) == 64
    validate_protocol_digest(notebook, digest)

    all_code = "\n".join(code_cells)
    required = [
        'stage342_decision.get("status") != "jepa_response_state_insufficient"',
        "grouped_candidate_oof(",
        "select_simplest_candidate(",
        'SELECTION = DATA["model_selection"]',
        'CALIBRATION = DATA["calibration"]',
        'EVALUATION = DATA["evaluation"]',
        "candidate_state_features(",
        "within_group_permuted_labels(",
        "for coordinate_index, coordinate_name in enumerate(coordinate_names):",
        "EXTRA_RESIDUAL_IMPROVEMENT",
        '"native_checkpoint_loaded": False',
        '"recursive_closure_claimed": False',
        '"confirmation_eligible": False',
    ]
    for fragment in required:
        assert fragment in all_code, f"missing Stage 34.3 contract fragment: {fragment}"
    forbidden = [
        "import torch",
        "load_world_model(",
        "torch.hub.load",
        "dino_wm_pusht",
        "JEPA_CAUSAL",
        '"confirmation_eligible": True',
    ]
    for fragment in forbidden:
        assert fragment not in all_code, f"forbidden Stage 34.3 fragment: {fragment}"

    selection_cell = code_cells[4]
    assert 'DATA["evaluation"]' not in selection_cell
    assert "EVALUATION" not in selection_cell
    assert "CANDIDATE_ROWS" in selection_cell
    assert "frozen_candidate_selection.json.sha256" in selection_cell
    controls_cell = code_cells[6]
    assert "all(row[\"passed\"] for row in COORDINATE_ROWS)" in controls_cell
    assert "MAX_EXTRA_RESIDUAL_CI_UPPER" in controls_cell

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
    print("Stage 34.3 notebook validation passed")
