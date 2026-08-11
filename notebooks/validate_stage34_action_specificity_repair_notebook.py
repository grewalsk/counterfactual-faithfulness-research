"""Static and numerical validation for the Stage 34.1 Colab notebook."""

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
NOTEBOOK = ROOT / "34_1_action_specificity_repair.ipynb"
BUILDER = ROOT / "build_stage34_action_specificity_repair_notebook.py"
NUMERICAL = REPOSITORY / "src/cf_faithfulness/stage34_action_specificity_repair.py"
TESTS = REPOSITORY / "tests/test_stage34_action_specificity_repair.py"


def source(cell):
    return "".join(cell.get("source", []))


def static_value(node):
    try:
        return ast.literal_eval(node)
    except (TypeError, ValueError):
        pass
    if isinstance(node, ast.DictComp):
        raise AssertionError("dictionary comprehensions are not static protocol values")
    raise AssertionError(f"unsupported static expression: {ast.dump(node)}")


def assigned_value(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return static_value(node.value)
    raise AssertionError(f"missing literal assignment {name}")


def validate_protocol_digest(notebook, observed):
    sources = [source(cell).strip() for cell in notebook["cells"]]
    replaced = False
    for index, text in enumerate(sources):
        if observed in text and "NOTEBOOK_PROTOCOL_SHA256" in text:
            sources[index] = text.replace(observed, "__PROTOCOL_DIGEST__", 1)
            replaced = True
            break
    assert replaced, "could not reconstruct Stage 34.1 protocol digest"
    expected = hashlib.sha256(
        json.dumps(sources, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    assert observed == expected, "Stage 34.1 protocol digest is stale"


def validate():
    for path in [NOTEBOOK, BUILDER, NUMERICAL, TESTS]:
        assert path.is_file(), f"missing Stage 34.1 artifact: {path}"

    before = NOTEBOOK.read_bytes()
    subprocess.run(
        [sys.executable, str(BUILDER)],
        check=True,
        capture_output=True,
        env=dict(os.environ),
    )
    after = NOTEBOOK.read_bytes()
    assert before == after, "Stage 34.1 notebook builder is not deterministic"

    notebook = json.loads(after)
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] == 5
    assert notebook["metadata"]["kernelspec"] == {
        "display_name": "Python 3", "language": "python", "name": "python3",
    }
    assert len(notebook["cells"]) == 9
    assert source(notebook["cells"][0]).startswith(
        "# Stage 34.1: leakage-free action-specificity repair\n"
    )
    assert [cell["id"] for cell in notebook["cells"]] == [
        f"stage341-{index:02d}" for index in range(9)
    ]
    code_cells = [source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 8
    for cell in notebook["cells"]:
        assert not cell.get("outputs")
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            ast.parse(source(cell))

    config_tree = ast.parse(code_cells[0])
    expected = {
        "RUN_MODE": "pilot",
        "PROTOCOL_ID": "stage34.1-leakage-free-action-specificity-repair-v1",
        "EVIDENCE_STATUS": "POST_OUTCOME_DIAGNOSTIC_REPAIR_NOT_CONFIRMATION",
        "UPSTREAM_PROTOCOL_ID": "stage34-predictive-fiber-causal-abstraction-v1",
        "UPSTREAM_RUN_SIGNATURE": "d3f4f88426afff4d964bb4f1f1556c94ec3613b667edd9403ddfcd0fd78ded84",
        "UPSTREAM_SOURCE_COMMIT": "db130a3d25505b7fa69efbcd88009365cb266688",
        "UPSTREAM_RAW_MANIFEST_SHA256": "2d2cf86fdeae5cb1034535104782dc526b8203b2e567e081ed530dbd288cb47e",
        "RFF_WIDTH": 256,
        "MIN_ACTION_SHUFFLE_ADVANTAGE": 0.10,
        "MIN_ACTION_BLIND_ADVANTAGE": 0.00,
        "MIN_PHYSICAL_ACTION_NECESSITY": 0.10,
        "MAX_ACTION_BLIND_PREDICTION_SPREAD": 1e-12,
    }
    for name, value in expected.items():
        if name == "RFF_WIDTH":
            # Pilot value is the left branch of the explicit smoke/pilot expression.
            node = next(
                node.value for node in config_tree.body
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
            )
            assert isinstance(node, ast.IfExp) and static_value(node.body) == value
        else:
            assert assigned_value(config_tree, name) == value, f"unexpected {name}"

    digest = assigned_value(config_tree, "NOTEBOOK_PROTOCOL_SHA256")
    assert len(digest) == 64
    validate_protocol_digest(notebook, digest)

    all_code = "\n".join(code_cells)
    required = [
        "action_response_path_rows(",
        "action_blind_context_features(",
        "fit_grouped_rff_ridge(",
        "deranged_word_rows(",
        "MAX_OBSERVED_ACTION_BLIND_SPREAD",
        "UPSTREAM_RAW_MANIFEST_SHA256",
        "verify_upstream(relative)",
        "confirmation_eligible\": False",
        "model_forwards\": 0",
    ]
    for fragment in required:
        assert fragment in all_code, f"missing repair contract fragment: {fragment}"
    forbidden = [
        "state @ state_baseline[\"weight\"]",
        "calibration_truth = np.asarray",
        "JEPA-to-DINO",
        "BOUNDED_TWO_SIDED_CAUSAL_ABSTRACTION_SUPPORTED",
    ]
    for fragment in forbidden:
        assert fragment not in all_code, f"forbidden Stage 34.1 fragment: {fragment}"

    locked = code_cells[5]
    assert "state_interval[0] > 0" in locked
    assert "shuffle_interval[0] > 0" in locked
    assert "all(value > 0 for value in mode_state.values())" in locked
    assert "all(value > 0 for value in mode_shuffle.values())" in locked

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
    print("Stage 34.1 notebook validation passed")
