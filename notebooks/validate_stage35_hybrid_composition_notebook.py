"""Static and numerical validation for the Stage 35 Colab notebook."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
NOTEBOOK = ROOT / "35_hybrid_predictive_composition_closure.ipynb"
BUILDER = ROOT / "build_stage35_hybrid_composition_notebook.py"
NUMERICAL = REPOSITORY / "src/cf_faithfulness/stage35_hybrid_composition.py"
TESTS = REPOSITORY / "tests/test_stage35_hybrid_composition.py"


def source(cell):
    return "".join(cell.get("source", []))


def assigned_node(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node.value
    raise AssertionError(f"missing assignment {name}")


def static_value(node):
    try:
        return ast.literal_eval(node)
    except (TypeError, ValueError):
        pass
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "list"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Call)
        and isinstance(node.args[0].func, ast.Name)
        and node.args[0].func.id == "range"
    ):
        arguments = [ast.literal_eval(value) for value in node.args[0].args]
        return list(range(*arguments))
    raise AssertionError(f"unsupported static expression: {ast.dump(node)}")


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
    assert replaced, "could not reconstruct Stage 35 protocol digest"
    expected = hashlib.sha256(
        json.dumps(sources, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    assert expected == observed, "Stage 35 protocol digest is stale"


def validate():
    for path in [NOTEBOOK, BUILDER, NUMERICAL, TESTS]:
        assert path.is_file(), f"missing Stage 35 artifact: {path}"
    before = NOTEBOOK.read_bytes()
    subprocess.run(
        [sys.executable, str(BUILDER)], check=True, capture_output=True,
        env=dict(os.environ),
    )
    after = NOTEBOOK.read_bytes()
    assert before == after, "Stage 35 builder is not deterministic"

    notebook = json.loads(after)
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] == 5
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["metadata"]["colab"]["gpuType"] == "L4"
    assert len(notebook["cells"]) == 13
    assert source(notebook["cells"][0]).startswith(
        "# Stage 35: JEPA hybrid predictive composition and closure\n"
    )
    assert [cell["id"] for cell in notebook["cells"]] == [
        f"stage35-{index:02d}" for index in range(13)
    ]
    code_cells = [source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 12
    expected_headers = [
        "# SINGLE CONFIGURATION BLOCK — no Stage 35 secrets required.",
        "import subprocess",
        "import csv",
        "# Tested NumPy implementation of hybrid local operators and recursive gates.",
        "def to_model_observation(visual, proprio):",
        "# Freeze trajectory families and action compositions before simulator or model access.",
        "# Select complete physical trajectories and materialize exact multi-step truth without model access.",
        "# Fit the construction-only grounded JEPA readout and save every native prefix carrier.",
        "# Select predicted-guard capacity using model-selection trajectories only.",
        "# Freeze calibration dynamics, label controls, bridge, and support before evaluation is opened.",
        "# Open fresh evaluation exactly once and derive every registered Stage 35 gate.",
        "# Package compact audit evidence while retaining the complete resumable Drive directory.",
    ]
    assert [cell.splitlines()[0] for cell in code_cells] == expected_headers
    for cell in notebook["cells"]:
        assert not cell.get("outputs")
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            ast.parse(source(cell))

    configuration_tree = ast.parse(code_cells[0])
    expected = {
        "RUN_MODE": "pilot",
        "PROTOCOL_ID": "stage35-jepa-hybrid-predictive-composition-closure-v1",
        "EVIDENCE_STATUS": "FRESH_PROSPECTIVE_JEPA_ONLY_OBSERVATIONAL_CLOSURE_TEST",
        "MODEL_NAMES": ["jepa_wm_pusht"],
        "MODEL_SELECTION_WORD_NAMES": ["A", "B", "AB", "BA", "AAB", "BBA", "ABBA", "BAAB"],
        "CALIBRATION_WORD_NAMES": ["A", "B", "AA", "BB", "ABA", "BAB", "AAAB", "BBBA", "AABB", "BBAA"],
        "PATH_CARRIER_SKETCH_DIM": 256,
        "MIN_CROSSING_GUARD_GAIN": 0.10,
        "MIN_GUARD_CONTROL_ADVANTAGE": 0.05,
        "MAX_RECURSIVE_TO_NATIVE_PHYSICAL_RATIO": 1.25,
        "MAX_RECURSIVE_SUPPORT_ESCAPE_RATE": 0.10,
        "CONSTRUCTION_TRAJECTORIES": 16,
        "MODEL_SELECTION_TRAJECTORIES": 16,
        "CALIBRATION_TRAJECTORIES": 16,
        "EVALUATION_TRAJECTORIES": 32,
    }
    for name, value in expected.items():
        assert assigned_value(configuration_tree, name) == value, f"unexpected {name}"
    pools = [
        assigned_value(configuration_tree, name)
        for name in [
            "CONSTRUCTION_TRAJECTORY_POOL", "MODEL_SELECTION_TRAJECTORY_POOL",
            "CALIBRATION_TRAJECTORY_POOL", "EVALUATION_TRAJECTORY_POOL",
        ]
    ]
    assert min(pools[0]) == 16000 and max(pools[-1]) == 23999
    assert all(
        not set(pools[left]) & set(pools[right])
        for left in range(len(pools)) for right in range(left + 1, len(pools))
    )
    digest = assigned_value(configuration_tree, "NOTEBOOK_PROTOCOL_SHA256")
    assert len(digest) == 64
    validate_protocol_digest(notebook, digest)

    setup_tree = ast.parse(code_cells[2])
    retry_node = next(
        node for node in setup_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "retry_drive_io"
    )
    namespace = {
        "time": type("NoWaitClock", (), {"sleep": staticmethod(lambda _delay: None)})()
    }
    exec(
        compile(ast.Module(body=[retry_node], type_ignores=[]), "<retry-test>", "exec"),
        namespace,
    )
    calls = {"count": 0}

    def transient():
        calls["count"] += 1
        if calls["count"] < 3:
            raise ConnectionAbortedError(103, "synthetic Drive disconnect")
        return "ready"

    assert namespace["retry_drive_io"]("synthetic", transient) == "ready"
    assert calls["count"] == 3

    all_code = "\n".join(code_cells)
    required = [
        'CONSTRUCTION_TRAJECTORY_POOL = list(range(16000, 17600))',
        'MODEL_SELECTION_TRAJECTORY_POOL = list(range(17600, 19200))',
        'CALIBRATION_TRAJECTORY_POOL = list(range(19200, 20800))',
        'EVALUATION_TRAJECTORY_POOL = list(range(20800, 24000))',
        'load_world_model("jepa_wm_pusht")',
        "stage35_carrier_sketch(value)",
        "source_modes=source_modes",
        'strategy="fixed_source"',
        'strategy="oracle_source"',
        'strategy="oracle_transition"',
        'strategy="predicted_guard"',
        "permuted_sequence_labels(",
        "time_shifted_sequence_labels(",
        "select_guard_hyperparameters(",
        'load_stage35_sequences("evaluation")',
        "MAX_RECURSIVE_TO_NATIVE_PHYSICAL_RATIO",
        '"causal_mechanism_claimed": False',
        '"minimal_state_claimed": False',
        '"dino_branch_paused": True',
        "retry_drive_io(",
        "def action_contrast_signature(",
        "def pool_spatial_proprio_features(",
        "def select_stable_rank(",
        "def fit_grouped_ridge(",
        "def fit_response_chart(",
    ]
    for fragment in required:
        assert fragment in all_code, f"missing Stage 35 fragment: {fragment}"
    forbidden = [
        'load_world_model("dino_wm_pusht")',
        "JEPA_TO_DINO",
        '"causal_evidence": True',
        '"low_dimensional_state_claimed": True',
        "USE_SYNTHETIC_FALLBACK = True",
    ]
    for fragment in forbidden:
        assert fragment not in all_code, f"forbidden Stage 35 fragment: {fragment}"

    # The simulator-only response chart runs before model loading and depends on
    # this Stage 34 contrast helper.  Execute the generated numerical-helper
    # cell directly so an omitted cross-cell dependency fails local validation
    # instead of a Colab run.
    analysis_namespace = {"np": np}
    exec(compile(code_cells[3], "<stage35-analysis-helpers>", "exec"), analysis_namespace)
    contrast = analysis_namespace["action_contrast_signature"](
        np.asarray([[[2.0]], [[0.5]]]),
        ["A", "zero1"],
        [1, 1],
        ["A"],
        {1: "zero1"},
    )
    np.testing.assert_allclose(contrast, [1.5])
    pooled = analysis_namespace["pool_spatial_proprio_features"](
        np.ones((256, 4), dtype=np.float64)
    )
    np.testing.assert_allclose(pooled, np.ones(4))
    rank = analysis_namespace["select_stable_rank"](
        np.asarray([[0.0], [1.0], [2.0], [3.0]]),
        np.arange(4),
        max_rank=1,
        n_bootstrap=2,
        n_permutations=2,
        stability_floor=0.0,
        seed=35,
    )
    assert rank["selected_rank"] in {0, 1}
    ridge = analysis_namespace["fit_grouped_ridge"](
        np.asarray([[0.0], [1.0], [2.0], [3.0]]),
        np.asarray([[0.0], [1.0], [2.0], [3.0]]),
        np.arange(4),
        penalties=[1e-3],
        folds=2,
        seed=35,
    )
    assert np.all(np.isfinite(ridge["weight"]))
    chart = analysis_namespace["fit_response_chart"](
        np.asarray([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]]), rank=1
    )
    assert chart["basis"].shape == (2, 1)

    physical_tree = ast.parse(code_cells[6])
    response_node = next(
        node for node in physical_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "response_signature_from_truth_path"
    )
    response_namespace = {
        **analysis_namespace,
        "ZERO_WORD_NAMES": {1: "zero1"},
    }
    exec(
        compile(ast.Module(body=[response_node], type_ignores=[]), "<response-path-test>", "exec"),
        response_namespace,
    )
    with tempfile.TemporaryDirectory(prefix="stage35-response-") as directory:
        path = Path(directory) / "truth.npz"
        np.savez(
            path,
            path_observables=np.asarray([[[2.0]], [[0.5]]]),
            word_names=np.asarray(["A", "zero1"]),
            word_lengths=np.asarray([1, 1]),
        )
        signature = response_namespace["response_signature_from_truth_path"](
            path, ["A"], []
        )
    np.testing.assert_allclose(signature, [1.5])

    selection_cell = code_cells[8]
    calibration_cell = code_cells[9]
    evaluation_cell = code_cells[10]
    construction_cell = code_cells[7]
    assert 'load_stage35_sequences("evaluation")' not in selection_cell
    assert 'load_stage35_sequences("evaluation")' not in calibration_cell
    assert 'SELECTED_RECORDS["evaluation"]' not in construction_cell
    assert "EVALUATION_OPENED = False" in calibration_cell
    assert 'load_stage35_sequences("evaluation")' in evaluation_cell
    assert 'generate_stage35_path_record(bundle, record, "evaluation", JEPA_DECODER)' in evaluation_cell
    assert evaluation_cell.index('validate_digest_sidecar(certificate_path)') < evaluation_cell.index(
        'generate_stage35_path_record(bundle, record, "evaluation", JEPA_DECODER)'
    )
    assert evaluation_cell.index(
        'generate_stage35_path_record(bundle, record, "evaluation", JEPA_DECODER)'
    ) < evaluation_cell.index("EVALUATION_OPENED = True")
    predicted_call = '''recursive_rollout(
            CARRIER_FAMILY, evaluation["initial_carrier"], evaluation["actions"], mask,
            strategy="predicted_guard",
        )'''
    assert predicted_call in evaluation_cell
    assert "source_modes=" not in predicted_call and "target_modes=" not in predicted_call

    construction_tree = ast.parse(construction_cell)
    mode_node = next(
        node for node in construction_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "stage35_mode_paths"
    )
    mode_namespace = {"np": np, "FRAMESKIP": 5}
    exec(
        compile(ast.Module(body=[mode_node], type_ignores=[]), "<mode-path-test>", "exec"),
        mode_namespace,
    )
    mode_paths = mode_namespace["stage35_mode_paths"]
    free_source, free_target = mode_paths(
        {"mode": "free"}, np.zeros(10, dtype=int), 2
    )
    assert free_source == ["free", "free"] and free_target == ["free", "free"]
    crossing_contacts = np.asarray([0] * 5 + [1] * 5)
    crossing_source, crossing_target = mode_paths(
        {"mode": "free"}, crossing_contacts, 2
    )
    assert crossing_source == ["free", "pre_contact"]
    assert crossing_target == ["pre_contact", "contact"]
    post_source, post_target = mode_paths(
        {"mode": "contact"}, np.zeros(10, dtype=int), 2
    )
    assert post_source == ["contact", "post_contact"]
    assert post_target == ["post_contact", "post_contact"]

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
    print("Stage 35 notebook validation passed")
