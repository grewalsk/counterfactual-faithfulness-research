"""Static and numerical validation for the Stage 36 Colab notebook."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
NOTEBOOK = ROOT / "36_predictive_state_closure_distillation.ipynb"
BUILDER = ROOT / "build_stage36_predictive_state_closure_notebook.py"
NUMERICAL = REPOSITORY / "src/cf_faithfulness/stage36_predictive_state_closure.py"
TESTS = REPOSITORY / "tests/test_stage36_predictive_state_closure.py"


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
    if isinstance(node, ast.List):
        return [static_value(value) for value in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(static_value(value) for value in node.elts)
    if isinstance(node, ast.Dict):
        return {
            static_value(key): static_value(value)
            for key, value in zip(node.keys, node.values)
        }
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return static_value(node.left) * static_value(node.right)
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
    assert replaced, "could not reconstruct Stage 36 protocol digest"
    expected = hashlib.sha256(
        json.dumps(sources, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    assert expected == observed, "Stage 36 protocol digest is stale"


def validate():
    for path in [NOTEBOOK, BUILDER, NUMERICAL, TESTS]:
        assert path.is_file(), f"missing Stage 36 artifact: {path}"
    before = NOTEBOOK.read_bytes()
    subprocess.run(
        [sys.executable, str(BUILDER)], check=True, capture_output=True,
        cwd=REPOSITORY, env=dict(os.environ),
    )
    after = NOTEBOOK.read_bytes()
    assert before == after, "Stage 36 builder is not deterministic"

    notebook = json.loads(after)
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] == 5
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["metadata"]["colab"]["gpuType"] == "L4"
    assert len(notebook["cells"]) == 13
    assert source(notebook["cells"][0]).startswith(
        "# Stage 36: predictive-state closure distillation\n"
    )
    assert [cell["id"] for cell in notebook["cells"]] == [
        f"stage36-{index:02d}" for index in range(13)
    ]
    code_cells = [source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    expected_headers = [
        "# SINGLE CONFIGURATION BLOCK — no Stage 36 secrets required.",
        "import subprocess",
        "import csv",
        "# Tested predictive-state adapter, controls, metrics, and decision gates.",
        "def to_model_observation(visual, proprio):",
        "# Freeze trajectory families and action compositions before simulator or model access.",
        "# Select complete physical trajectories and materialize exact multi-step truth without model access.",
        "# Fit the construction-only grounded JEPA readout and save every native prefix carrier.",
        "# Load split-bound teacher sequences without opening evaluation statistics.",
        "# Freeze the final PSCD adapter and capacity-matched controls before evaluation.",
        "# Open fresh evaluation once and derive every registered Stage 36 gate.",
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
        "PROTOCOL_ID": "stage36-predictive-state-closure-distillation-v5",
        "EVIDENCE_STATUS": "FRESH_PROSPECTIVE_JEPA_ONLY_ADAPTER_CLOSURE_TEST",
        "MODEL_NAMES": ["jepa_wm_pusht"],
        "MAX_WORD_LENGTH": 12,
        "MAX_CARRIER_PROJECTION_DIM": 1024,
        "CANDIDATE_EPOCHS": 80,
        "FINAL_EPOCHS": 240,
        "MIN_CLOSURE_CONTROL_GAIN": 0.05,
        "MAX_RECURSIVE_TO_NATIVE_PHYSICAL_RATIO": 1.25,
        "MAX_SEMIGROUP_NMSE": 0.25,
        "CONSTRUCTION_TRAJECTORIES": 16,
        "MODEL_SELECTION_TRAJECTORIES": 16,
        "CALIBRATION_TRAJECTORIES": 16,
        "EVALUATION_TRAJECTORIES": 32,
        "MODEL_SELECTION_WORD_NAMES": [
            "ABABA", "BABAB", "AABBAB", "BBAABA", "AAABBAB", "BBABAAB",
            "AABBABAB", "BBAABAAB",
        ],
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
    assert min(pools[0]) == 24000 and max(pools[-1]) == 31999
    assert all(
        not set(pools[left]) & set(pools[right])
        for left in range(len(pools)) for right in range(left + 1, len(pools))
    )
    digest = assigned_value(configuration_tree, "NOTEBOOK_PROTOCOL_SHA256")
    assert len(digest) == 64
    validate_protocol_digest(notebook, digest)

    configuration_namespace = {}
    exec(compile(code_cells[0], "<stage36-config>", "exec"), configuration_namespace)
    assert [len(row["name"]) for row in configuration_namespace["EVALUATION_WORD_SPECS"]] == [
        9, 9, 10, 10, 11, 11, 12, 12,
    ]
    assert configuration_namespace["CARRIER_PROJECTION_DIMS"] == [256, 1024]
    assert configuration_namespace["HISTORY_LENGTHS"] == [1, 2, 4]
    assert configuration_namespace["DYNAMICS_FAMILIES"] == ["single", "mixture"]

    all_code = "\n".join(code_cells)
    required = [
        'load_world_model("jepa_wm_pusht")',
        'for split in ["construction", "model_selection", "calibration"]',
        "MAX_CARRIER_PROJECTION_DIM",
        "fit_predictive_state_closure(",
        "rollout_predictive_state_closure(",
        "free_weight=0.0",
        "permute_past_history(",
        "fit_family_from_sequences(",
        'strategy="global"',
        'load_stage36_sequences("evaluation")',
        '"evaluation_word_lengths": [9, 10, 11, 12]',
        '"jepa_parameters_updated": False',
        '"original_jepa_carrier_claimed_closed": False',
        '"minimal_state_claimed": False',
        '"causal_mechanism_claimed": False',
        "MAX_SEMIGROUP_NMSE",
        "retry_drive_io(",
        "def action_contrast_signature(",
        "def fit_response_chart(",
        "def pool_spatial_proprio_features(",
        "def select_stable_rank(",
        "def fit_grouped_ridge(",
        "v3_complete_truth_consumer_coverage_no_scientific_change",
        '"v3_truth_consumer_coverage_amendment": True',
        "cached word contract changed",
        "def fetch_url_bytes(",
        "v4_retryable_exact_source_binding_no_scientific_change",
        '"v4_source_binding_retry_amendment": True',
        "v5_registered_action_vocabulary_no_model_outcome_change",
        '"v5_action_vocabulary_amendment": True',
        "Stage 36 preflight word is outside the registered construction bank",
        "unregistered Stage 36 transition words",
    ]
    for fragment in required:
        assert fragment in all_code, f"missing Stage 36 fragment: {fragment}"
    forbidden = [
        'load_world_model("dino_wm_pusht")',
        '"jepa_parameters_updated": True',
        '"original_jepa_carrier_claimed_closed": True',
        '"causal_evidence": True',
        "USE_SYNTHETIC_FALLBACK = True",
        'name = "L"',
        'names = ["L", "R", "S"]',
        '"L": (-30.0, 0.14)',
        '"a": (-20.0, 0.10)',
    ]
    for fragment in forbidden:
        assert fragment not in all_code, f"forbidden Stage 36 fragment: {fragment}"

    # Reproduce the transient HTTP 504 that stopped v3 and prove the exact-byte
    # fetch succeeds on retry without any unverified fallback.
    setup_tree = ast.parse(code_cells[2])
    fetch_node = next(
        node for node in setup_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "fetch_url_bytes"
    )
    attempts = []
    sleeps = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return b"committed-source-bytes"

    def flaky_urlopen(request, timeout):
        attempts.append((request.full_url, timeout))
        if len(attempts) == 1:
            raise urllib.error.HTTPError(
                request.full_url, 504, "Gateway Timeout", {}, None
            )
        return FakeResponse()

    class FakeRequestModule:
        Request = urllib.request.Request
        urlopen = staticmethod(flaky_urlopen)

    class FakeErrorModule:
        HTTPError = urllib.error.HTTPError
        URLError = urllib.error.URLError

    class FakeUrllib:
        request = FakeRequestModule
        error = FakeErrorModule

    class FakeTime:
        sleep = staticmethod(sleeps.append)

    fetch_namespace = {
        "urllib": FakeUrllib,
        "time": FakeTime,
        "RETRYABLE_HTTP_STATUS": {408, 425, 429, 500, 502, 503, 504},
    }
    exec(
        compile(
            ast.Module(body=[fetch_node], type_ignores=[]),
            "<stage36-http-retry-test>", "exec",
        ),
        fetch_namespace,
    )
    fetched = fetch_namespace["fetch_url_bytes"](
        "https://example.invalid/committed", "test committed source",
        attempts=2, timeout_seconds=3.0,
    )
    assert fetched == b"committed-source-bytes"
    assert len(attempts) == 2 and sleeps == [1.0]

    # Execute the exact first-model preflight with the real Stage 36 manifest
    # and mocked shape-correct outputs.  This would have caught the legacy `L`
    # word before the v4 Colab run.
    construction_tree = ast.parse(code_cells[7])
    construction_functions = {
        node.name: node for node in construction_tree.body
        if isinstance(node, ast.FunctionDef)
    }
    preflight_calls = []
    visual_tokens = configuration_namespace["EXPECTED_VISUAL_TOKENS"]
    visual_width = configuration_namespace["EXPECTED_VISUAL_WIDTH"]
    proprio_tokens = configuration_namespace["EXPECTED_PROPRIO_TOKENS"]
    proprio_width = configuration_namespace["EXPECTED_PROPRIO_FEATURE_WIDTHS"][
        "jepa_wm_pusht"
    ]
    carrier_width = configuration_namespace["EXPECTED_CARRIER_WIDTHS"][
        "jepa_wm_pusht"
    ]

    def fake_grouped_model_words(bundle, record, names):
        preflight_calls.append((bundle["name"], record["record_id"], list(names)))
        assert names == ["A"]
        return (
            {
                "A": (
                    np.zeros((1, visual_tokens, visual_width), dtype=np.float32),
                    np.zeros((1, proprio_tokens, proprio_width), dtype=np.float32),
                )
            },
            {"A": np.zeros((1, visual_tokens, carrier_width), dtype=np.float32)},
        )

    def fake_feature_tensor(outputs, names):
        assert names == ["A"] and set(outputs) == {"A"}
        return (
            np.zeros(
                (
                    1,
                    configuration_namespace["MAX_WORD_LENGTH"],
                    configuration_namespace["VISUAL_SKETCH_DIM"]
                    + configuration_namespace["PROPRIO_PAD_DIM"],
                ),
                dtype=np.float32,
            ),
            proprio_width,
        )

    preflight_namespace = {
        "np": np,
        "CANONICAL_RESPONSE_WORD_NAMES": configuration_namespace[
            "CANONICAL_RESPONSE_WORD_NAMES"
        ],
        "CONSTRUCTION_WORD_NAMES": configuration_namespace["CONSTRUCTION_WORD_NAMES"],
        "WORD_BY_NAME": {"A": {"length": 1}},
        "SELECTED_RECORDS": {"construction": [{"record_id": 3600001}]},
        "grouped_model_words": fake_grouped_model_words,
        "feature_tensor_from_outputs": fake_feature_tensor,
        "EXPECTED_PROPRIO_FEATURE_WIDTHS": configuration_namespace[
            "EXPECTED_PROPRIO_FEATURE_WIDTHS"
        ],
        "EXPECTED_VISUAL_TOKENS": visual_tokens,
        "EXPECTED_VISUAL_WIDTH": visual_width,
        "EXPECTED_PROPRIO_TOKENS": proprio_tokens,
        "PROPRIO_FEATURE_POOLING": configuration_namespace["PROPRIO_FEATURE_POOLING"],
        "PROPRIO_PAD_DIM": configuration_namespace["PROPRIO_PAD_DIM"],
        "VISUAL_SKETCH_DIM": configuration_namespace["VISUAL_SKETCH_DIM"],
        "MAX_WORD_LENGTH": configuration_namespace["MAX_WORD_LENGTH"],
        "OUT": Path("/tmp/stage36-preflight-validator"),
        "write_json": lambda *_: None,
        "PROVENANCE_COUNTS": {"model_output_contract_preflights": {"jepa": 0}},
    }
    exec(
        compile(
            ast.Module(
                body=[construction_functions["preflight_model_output_contract"]],
                type_ignores=[],
            ),
            "<stage36-model-preflight-test>", "exec",
        ),
        preflight_namespace,
    )
    contract = preflight_namespace["preflight_model_output_contract"]({
        "name": "jepa_wm_pusht", "short": "jepa", "pred_type": "AdaLN",
        "carrier_width": carrier_width,
    })
    assert contract["word"] == "A" and preflight_calls == [
        ("jepa_wm_pusht", 3600001, ["A"])
    ]

    transition_namespace = {
        "CALIBRATION_INTERCHANGE_PAIRS": configuration_namespace[
            "CALIBRATION_INTERCHANGE_PAIRS"
        ],
        "EVALUATION_INTERCHANGE_PAIRS": configuration_namespace[
            "EVALUATION_INTERCHANGE_PAIRS"
        ],
        "EVALUATION_WORD_NAMES": [
            row["name"] for row in configuration_namespace["EVALUATION_WORD_SPECS"]
        ],
    }
    exec(
        compile(
            ast.Module(
                body=[construction_functions["transition_prefixes"]], type_ignores=[]
            ),
            "<stage36-transition-vocabulary-test>", "exec",
        ),
        transition_namespace,
    )
    for split in ["model_selection", "calibration", "evaluation"]:
        names, prefixes = transition_namespace["transition_prefixes"](split)
        assert names and prefixes
        assert all(set(value).issubset({"A", "B"}) for value in names + prefixes)

    runtime_tree = ast.parse(code_cells[5])
    token_node = next(
        node for node in runtime_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "token_definition"
    )
    token_namespace = {}
    exec(
        compile(
            ast.Module(body=[token_node], type_ignores=[]),
            "<stage36-token-vocabulary-test>", "exec",
        ),
        token_namespace,
    )
    assert token_namespace["token_definition"]("A") == (-40.0, 0.18)
    assert token_namespace["token_definition"]("B") == (40.0, 0.18)
    try:
        token_namespace["token_definition"]("L")
    except KeyError:
        pass
    else:
        raise AssertionError("legacy Stage 35 token L remains executable")

    analysis_namespace = {"np": np}
    exec(compile(code_cells[3], "<stage36-analysis>", "exec"), analysis_namespace)
    contrast = analysis_namespace["action_contrast_signature"](
        np.asarray([[[2.0]], [[0.5]]]),
        ["A", "zero1"], [1, 1], ["A"], {1: "zero1"},
    )
    np.testing.assert_allclose(contrast, [1.5])
    chart = analysis_namespace["fit_response_chart"](
        np.asarray([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]]), rank=1,
    )
    assert chart["basis"].shape == (2, 1)
    pooled = analysis_namespace["pool_spatial_proprio_features"](
        np.ones((256, 4), dtype=np.float64)
    )
    np.testing.assert_allclose(pooled, np.ones(4))
    rank = analysis_namespace["select_stable_rank"](
        np.asarray([[0.0], [1.0], [2.0], [3.0]]), np.arange(4),
        max_rank=1, n_bootstrap=2, n_permutations=2,
        stability_floor=0.0, seed=36,
    )
    assert rank["selected_rank"] in {0, 1}
    ridge = analysis_namespace["fit_grouped_ridge"](
        np.asarray([[0.0], [1.0], [2.0], [3.0]]),
        np.asarray([[0.0], [1.0], [2.0], [3.0]]),
        np.arange(4), penalties=[1e-3], folds=2, seed=36,
    )
    assert np.all(np.isfinite(ridge["weight"]))
    initial = np.asarray([[1.0], [2.0]])
    target = np.asarray([[[1.5], [2.0]], [[2.5], [3.0]]])
    mask = np.ones((2, 2), dtype=bool)
    history = analysis_namespace["history_tensor"](initial, target, mask, 2)
    assert history.shape == (2, 2, 2, 1)
    evaluation_mask = analysis_namespace["rollout_evaluation_mask"](mask, 2)
    assert not np.any(evaluation_mask[:, 0]) and np.all(evaluation_mask[:, 1])

    # Execute the real split-specific truth coverage contract.  This checks
    # every active truth consumer and reproduces the v2 failure schema before a
    # Colab GPU is allocated.
    physical_tree = ast.parse(code_cells[6])
    physical_functions = {
        node.name: node for node in physical_tree.body
        if isinstance(node, ast.FunctionDef)
    }
    path_tree = ast.parse(code_cells[7])
    path_functions = {
        node.name: node for node in path_tree.body
        if isinstance(node, ast.FunctionDef)
    }
    all_word_names = (
        set(configuration_namespace["STAGE36_CORE_WORD_NAMES"])
        | {row["name"] for row in configuration_namespace["EVALUATION_WORD_SPECS"]}
        | set(configuration_namespace["ZERO_WORD_NAMES"].values())
    )
    word_by_name = {
        name: {
            "length": int(name[4:]) if name.startswith("zero") else len(name)
        }
        for name in all_word_names
    }
    response_namespace = {
        **analysis_namespace,
        **{
            name: configuration_namespace[name]
            for name in [
                "CONSTRUCTION_WORD_NAMES", "MODEL_SELECTION_WORD_NAMES",
                "CALIBRATION_WORD_NAMES", "CANONICAL_RESPONSE_WORD_NAMES",
                "CORE_ORDER_PAIRS", "ZERO_WORD_NAMES",
            ]
        },
        "EVALUATION_WORD_NAMES": [
            row["name"] for row in configuration_namespace["EVALUATION_WORD_SPECS"]
        ],
        "WORD_BY_NAME": word_by_name,
    }
    exec(
        compile(
            ast.Module(
                body=[
                    physical_functions["stage36_truth_word_names"],
                    physical_functions["response_signature_from_truth_path"],
                    path_functions["stage36_names_for_split"],
                ],
                type_ignores=[],
            ),
            "<stage36-response-path-test>", "exec",
        ),
        response_namespace,
    )
    truth_names = {
        split: response_namespace["stage36_truth_word_names"](split)
        for split in ["construction", "model_selection", "calibration", "evaluation"]
    }
    task_names = {
        split: set(response_namespace["stage36_names_for_split"](split))
        for split in truth_names
    }
    canonical_required = set(configuration_namespace["CANONICAL_RESPONSE_WORD_NAMES"])
    canonical_required.update(
        name for pair in configuration_namespace["CORE_ORDER_PAIRS"] for name in pair
    )
    canonical_required.update({
        configuration_namespace["ZERO_WORD_NAMES"][word_by_name[name]["length"]]
        for name in tuple(canonical_required)
    })
    for split in truth_names:
        expected_truth_names = set(task_names[split])
        if split in {"construction", "model_selection"}:
            expected_truth_names.update(canonical_required)
        expected_truth_names.update({
            configuration_namespace["ZERO_WORD_NAMES"][word_by_name[name]["length"]]
            for name in tuple(expected_truth_names)
        })
        assert set(truth_names[split]) == expected_truth_names
    for split in ["construction", "model_selection"]:
        assert canonical_required.issubset(set(truth_names[split]))
    assert set(configuration_namespace["MODEL_SELECTION_WORD_NAMES"]).isdisjoint(
        set(configuration_namespace["CONSTRUCTION_WORD_NAMES"])
    )

    with tempfile.TemporaryDirectory(prefix="stage36-response-") as directory:
        for split in ["construction", "model_selection"]:
            names = truth_names[split]
            lengths = np.asarray(
                [word_by_name[name]["length"] for name in names], dtype=np.int64
            )
            paths = np.zeros((len(names), int(np.max(lengths)), 2), dtype=np.float64)
            for index, length in enumerate(lengths):
                paths[index, :length] = float(index + 1)
            path = Path(directory) / f"{split}_truth.npz"
            np.savez(
                path, path_observables=paths, word_names=np.asarray(names),
                word_lengths=lengths,
            )
            signature = response_namespace["response_signature_from_truth_path"](
                path,
                configuration_namespace["CANONICAL_RESPONSE_WORD_NAMES"],
                configuration_namespace["CORE_ORDER_PAIRS"],
            )
            assert signature.ndim == 1 and np.all(np.isfinite(signature))

    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(TESTS)],
        check=True, cwd=REPOSITORY,
        env={
            **os.environ,
            "PYTHONPATH": str(REPOSITORY / "src"),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        },
    )


if __name__ == "__main__":
    validate()
    print("Stage 36 notebook validation passed")
