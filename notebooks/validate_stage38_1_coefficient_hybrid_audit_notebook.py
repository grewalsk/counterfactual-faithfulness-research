"""Static, protocol, and numerical validation for the Stage 38.1 notebook."""

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
NOTEBOOK = ROOT / "38_1_coefficient_matched_hybrid_audit.ipynb"
BUILDER = ROOT / "build_stage38_1_coefficient_hybrid_audit_notebook.py"
NUMERICAL = REPOSITORY / "src/cf_faithfulness/stage38_1_coefficient_hybrid_audit.py"
TESTS = REPOSITORY / "tests/test_stage38_1_coefficient_hybrid_audit.py"
UPSTREAM_TESTS = [REPOSITORY / "tests/test_stage38_cross_model_pscd.py"]
GUIDE = REPOSITORY / "STAGE38_1_RUN_GUIDE.md"
DECISION = REPOSITORY / "docs/STAGE38_TO_NEXT_MATHEMATICAL_DECISION.md"


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
    assert replaced, "could not reconstruct Stage 38.1 protocol digest"
    expected = hashlib.sha256(
        json.dumps(sources, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    assert observed == expected, "Stage 38.1 protocol digest is stale"


def direct_calls(code_cells, function_name):
    calls = []
    for index, text in enumerate(code_cells):
        tree = ast.parse(text)
        parents = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == function_name
            ):
                continue
            parent = parents.get(node)
            inside_definition = False
            while parent is not None:
                if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    inside_definition = True
                    break
                parent = parents.get(parent)
            if not inside_definition:
                calls.append((index, node))
    return calls


def validate():
    for path in [NOTEBOOK, BUILDER, NUMERICAL, TESTS, GUIDE, DECISION, *UPSTREAM_TESTS]:
        assert path.is_file(), f"missing Stage 38.1 artifact: {path}"

    before = NOTEBOOK.read_bytes()
    subprocess.run(
        [sys.executable, str(BUILDER)], check=True, capture_output=True,
        cwd=REPOSITORY, env=dict(os.environ),
    )
    after = NOTEBOOK.read_bytes()
    assert before == after, "Stage 38.1 builder is not deterministic"

    notebook = json.loads(after)
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] == 5
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["metadata"]["colab"]["gpuType"] == "L4"
    assert len(notebook["cells"]) == 12
    assert source(notebook["cells"][0]).startswith(
        "# Stage 38.1: coefficient-matched and hybrid closure audit\n"
    )
    assert [cell["id"] for cell in notebook["cells"]] == [
        f"stage381-{index:02d}" for index in range(12)
    ]
    expected_headers = [
        "# SINGLE CONFIGURATION BLOCK — no Stage 38.1 secrets required.",
        "import subprocess",
        "import csv",
        "# Tested coefficient audit, event/reset controls, metrics, and decisions.",
        "# Bind only the three Stage 38 development shard families.",
        "# Load construction, model-selection, and calibration arrays only.",
        "# Fit Tier A construction-only matched controls for two screening seeds.",
        "# Score untouched calibration, then conditionally run the third seed.",
        "# Conditionally test oracle event headroom before matched hybrid controls.",
        "# Evaluate the label-free hybrid and conditionally test family risk.",
        "# Emit the development decision and compact result bundle.",
    ]
    code_cells = [source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert [text.splitlines()[0] for text in code_cells] == expected_headers
    for cell in notebook["cells"]:
        assert not cell.get("outputs")
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            ast.parse(source(cell))

    namespace = {}
    exec(compile(code_cells[0], "<stage381-config>", "exec"), namespace)
    assert namespace["RUN_MODE"] == "pilot"
    assert namespace["PROTOCOL_ID"] == "stage38.1-coefficient-matched-hybrid-audit-v2"
    assert namespace["SOURCE_STAGE38_RUN_SIGNATURE"].startswith("ceb85af5b4b9")
    assert namespace["SOURCE_STAGE38_COMMIT"] == "a7ed07e2e79bc4da77e022f7765239b260bff35c"
    assert namespace["LEGACY_STAGE381_PROTOCOL_ID"] == "stage38.1-coefficient-matched-hybrid-audit-v1"
    assert namespace["LEGACY_STAGE381_RUN_SIGNATURE"].startswith("0b09871c37cc")
    assert namespace["LEGACY_STAGE381_SOURCE_COMMIT"] == "d570ce091cc4c22e7a76cb91fce7a782484ac616"
    assert namespace["DEVELOPMENT_SPLITS"] == ["construction", "model_selection", "calibration"]
    assert namespace["EVALUATION_ACCESS_PERMITTED"] is False
    assert namespace["PLANNING_ACCESS_PERMITTED"] is False
    assert namespace["FULL_OUTER_WEIGHTS"] == {"jepa": 2.0, "dino": 1.0}
    assert namespace["COEFFICIENT_MATCHED_OUTER_WEIGHTS"] == {"jepa": 0.90, "dino": 0.45}
    assert namespace["SCREENING_SEEDS"] == [38101, 38102]
    assert namespace["PRECOMMITTED_THIRD_SEED"] == 38103
    assert namespace["FINAL_EPOCHS"] == 320
    assert namespace["EVENT_HIDDEN"] == 32
    assert namespace["EVENT_LOSS_WEIGHT"] == 0.10
    assert namespace["RISK_ALPHA"] == 0.90
    assert namespace["RISK_LOSS_WEIGHT"] == 0.10
    assert namespace["MIN_COEFFICIENT_GAIN"] == 0.05
    assert namespace["MIN_EVENT_CONDITIONAL_GAIN"] == 0.25
    assert namespace["MIN_OVERALL_P95_GAIN"] == 0.10
    digest = namespace["NOTEBOOK_PROTOCOL_SHA256"]
    assert len(digest) == 64
    validate_protocol_digest(notebook, digest)

    # Execute the rendered helper cell and instantiate both gate classes.  The
    # v1 validator only parsed this cell, so stripped decorators went unnoticed.
    helper_namespace = {"np": np}
    exec(compile(code_cells[3], "<stage381-rendered-helpers>", "exec"), helper_namespace)
    tier_a_gate = helper_namespace["TierAGates"](*(True,) * 5)
    tier_b_gate = helper_namespace["TierBGates"](*(True,) * 8)
    assert list(tier_a_gate.__dataclass_fields__) == [
        "coefficient_specificity", "tail_noninferiority",
        "correct_history_specificity", "absolute_viability", "three_seed_stability",
    ]
    assert len(tier_b_gate.__dataclass_fields__) == 8
    assert "@dataclass(frozen=True)\nclass TierAGates:" in code_cells[3]
    assert "@dataclass(frozen=True)\nclass TierBGates:" in code_cells[3]

    provenance = direct_calls(code_cells, "verify_executed_notebook_through")
    assert [(index, node.args[0].value) for index, node in provenance] == [
        (4, expected_headers[4]), (5, expected_headers[5]),
        (6, expected_headers[6]), (7, expected_headers[7]),
        (8, expected_headers[8]), (9, expected_headers[9]),
    ]

    executable = "\n".join(code_cells[4:])
    required = [
        'if str(split) not in DEVELOPMENT_SPLITS:',
        '"evaluation_files_read": 0',
        '"planning_files_read": 0',
        'train = DEVELOPMENT_DATA[short]["construction"]',
        'data, scale = DEVELOPMENT_DATA[short]["calibration"], PHYSICAL_SCALES[short]',
        '"coefficient_overshoot"',
        'COEFFICIENT_MATCHED_OUTER_WEIGHTS[short]',
        'def validate_legacy_migration_root():',
        '"TypeError: TierAGates() takes no arguments"',
        'def stage381_migration_receipt_path(short, variant, seed):',
        'def record_legacy_import(row):',
        'def load_legacy_tier_a_artifact(short, variant, seed):',
        '"source_array_sha256": sha256_file(array_path)',
        '"current_array_sha256": sha256_file(current_array)',
        'write_digest_sidecar(receipt_path)',
        'if LEGACY_MIGRATION_VERIFIED and int(seed) in SCREENING_SEEDS:',
        '"calibration_outcomes_imported": 0',
        '"legacy_imported_artifacts": len(LEGACY_IMPORT_ROWS)',
        'for seed in SCREENING_SEEDS:',
        'if screen_passed:',
        'freeze_tier_a_seed(PRECOMMITTED_THIRD_SEED)',
        'if not PIPELINE_FAILED and TIER_A_PROMOTED:',
        'oracle_events=data["event"]',
        'if ORACLE_HEADROOM_PASSED:',
        'for variant in ["smooth", "shuffled"]:',
        'if TIER_B_PROMOTED:',
        'risk_weight=RISK_LOSS_WEIGHT',
        'stage381_cmha_result_bundle_',
        'stage381_cmha_bundle_staging_',
    ]
    for fragment in required:
        assert fragment in executable, f"missing Stage 38.1 fragment: {fragment}"
    forbidden = [
        "locked_closure_rows.csv", "locked_planning_rows.csv",
        "selected_evaluation_trajectories.json",
        "load_world_model(", "grouped_model_words(",
        'DEVELOPMENT_DATA[short]["evaluation"]',
        'source_stage38_path(short, "evaluation"',
        '"native_closure_claimed": True', '"planning_claimed": True',
        '"fresh_confirmation_claimed": True',
        'shutil.make_archive(str(zip_base), "zip", root_dir=OUT)',
    ]
    for fragment in forbidden:
        assert fragment not in executable, f"forbidden Stage 38.1 fragment: {fragment}"

    assert len(direct_calls(code_cells, "fit_weighted_semigroup_predictive_state_closure")) == 0
    assert len(direct_calls(code_cells, "fit_event_factorized_pscd")) == 0
    # Both fitters appear once inside source-bound fit-or-load functions.
    assert executable.count("artifact = fit_weighted_semigroup_predictive_state_closure(") == 1
    assert executable.count("artifact = fit_event_factorized_pscd(") == 1
    # One wrapper signature, one forwarding call, and one oracle-only caller.
    assert executable.count("oracle_events=") == 3
    assert executable.count('"evaluation_evidence/tier_a_calibration_rows.csv"') == 1
    assert executable.count('"evaluation_evidence/tier_a_final_decisions.json"') == 1

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPOSITORY / "src")
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(TESTS), *map(str, UPSTREAM_TESTS)],
        check=True, cwd=REPOSITORY, env=environment,
    )
    print("Stage 38.1 notebook validation passed")


if __name__ == "__main__":
    validate()
