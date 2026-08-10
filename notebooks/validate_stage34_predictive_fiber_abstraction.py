"""Static and reproducibility checks for the Stage 34 Colab notebook.

The GPU experiment is intentionally not executed here.  This validator checks
that the notebook is a deterministic builder product, every code cell parses,
the sequential protocol is frozen, and the implementation preserves the
simulator-only chart and no-direct-cross-model-map claim boundary.
"""

import ast
import builtins
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
NOTEBOOK = ROOT / "34_predictive_fiber_causal_abstraction.ipynb"
BUILDER = ROOT / "build_stage34_predictive_fiber_abstraction_notebook.py"
NUMERICAL = REPOSITORY / "src/cf_faithfulness/stage34_predictive_fiber_abstraction.py"
TESTS = REPOSITORY / "tests/test_stage34_predictive_fiber_abstraction.py"


def source(cell):
    return "".join(cell.get("source", []))


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
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = static_value(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mult)):
        left, right = static_value(node.left), static_value(node.right)
        return left + right if isinstance(node.op, ast.Add) else left * right
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "range"
        and not node.keywords
    ):
        return range(*(static_value(argument) for argument in node.args))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "list"
        and len(node.args) == 1
        and not node.keywords
    ):
        return list(static_value(node.args[0]))
    raise AssertionError(f"unsupported static expression: {ast.dump(node)}")


def assigned_value(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return static_value(node.value)
    raise AssertionError(f"missing literal assignment {name}")


def function_source(cells, name):
    for text in cells:
        tree = ast.parse(text)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name == name:
                    return ast.get_source_segment(text, node)
    raise AssertionError(f"missing definition {name}")


def require_all(text, fragments, section):
    for fragment in fragments:
        assert fragment in text, f"{section}: missing {fragment!r}"


def defined_names(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Param)):
            names.add(node.id)
        elif isinstance(node, ast.Import):
            names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(
                alias.asname or alias.name for alias in node.names if alias.name != "*"
            )
    return names


def validate_obvious_names(code_cells):
    trees = [ast.parse(text) for text in code_cells]
    defined = set(dir(builtins))
    for tree in trees:
        defined.update(defined_names(tree))
    loaded = {
        node.id
        for tree in trees
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    missing = sorted(name for name in loaded if name.isupper() and name not in defined)
    assert not missing, f"undefined protocol names: {missing}"


def validate_protocol_digest(notebook, config, observed):
    protocol_sources = [source(cell).strip() for cell in notebook["cells"]]
    replaced = False
    for index, text in enumerate(protocol_sources):
        if observed in text and "NOTEBOOK_PROTOCOL_SHA256" in text:
            protocol_sources[index] = text.replace(observed, "__PROTOCOL_DIGEST__", 1)
            replaced = True
            break
    assert replaced, "could not reconstruct protocol digest"
    expected = hashlib.sha256(
        json.dumps(protocol_sources, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    assert observed == expected, "notebook protocol digest is stale"
    assert "__PROTOCOL_DIGEST__" not in config


def validate():
    for path in [BUILDER, NOTEBOOK, NUMERICAL, TESTS]:
        assert path.is_file(), f"missing Stage 34 artifact: {path}"

    before = NOTEBOOK.read_bytes()
    subprocess.run(
        [sys.executable, str(BUILDER)],
        check=True,
        capture_output=True,
        env=dict(os.environ),
    )
    after = NOTEBOOK.read_bytes()
    assert before == after, "Stage 34 builder is not deterministic"

    notebook = json.loads(after)
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] == 5
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["metadata"]["colab"]["gpuType"] == "L4"
    assert notebook["metadata"]["kernelspec"] == {
        "display_name": "Python 3", "language": "python", "name": "python3",
    }
    assert len(notebook["cells"]) == 14
    assert notebook["cells"][0]["cell_type"] == "markdown"
    assert source(notebook["cells"][0]).startswith(
        "# Stage 34: predictive-fiber causal abstraction\n"
    )
    code_cells = [
        source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    assert len(code_cells) == 13
    assert [cell.splitlines()[0] for cell in code_cells[5:]] == [
        "# Freeze trajectory families and action compositions before simulator or model access.",
        "# Select complete physical trajectories and materialize exact multi-step truth without model access.",
        "# Fit model-specific grounded readouts and carrier interfaces on construction trajectories only.",
        "# Open the locked evaluation once and test action specificity against state/target controls.",
        "# Test whether residual carrier information improves unseen transition prediction.",
        "# Test matched predictive fibers and response-state edits with on-manifold diagnostics.",
        "# Test two model-to-physical diagrams, apply sequential gates, and interpret.",
        "# Package compact audit evidence while retaining the complete resumable Drive directory.",
    ]
    for index, cell in enumerate(notebook["cells"]):
        assert cell["id"] == f"stage34-{index:02d}"
        assert not cell.get("outputs")
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            ast.parse(source(cell))
    validate_obvious_names(code_cells)

    config = code_cells[0]
    tree = ast.parse(config)
    expected_values = {
        "PROTOCOL_ID": "stage34-predictive-fiber-causal-abstraction-v1",
        "EVIDENCE_STATUS": "PILOT_V1_ONLY_IF_SOURCE_BOUND_SPLIT_LOCKED_ACTION_SPECIFIC_SUFFICIENT_AND_ON_MANIFOLD",
        "RUN_MODE": "pilot",
        "EXPERIMENT_SOURCE_REF": "codex/stage34-predictive-fiber-abstraction",
        "CONSTRUCTION_TRAJECTORIES": 16,
        "MODEL_SELECTION_TRAJECTORIES": 16,
        "CALIBRATION_TRAJECTORIES": 16,
        "EVALUATION_TRAJECTORIES": 32,
        "STATES_PER_TRAJECTORY": 4,
        "MAX_WORD_LENGTH": 8,
        "MAX_COMPOSED_LENGTH": 12,
        "MAX_ESTIMATED_TOTAL_MINUTES": 900,
        "CAUSAL_WORDS": ["zero1"],
        "MIN_ACTION_SHUFFLE_ADVANTAGE": 0.10,
        "MAX_RESIDUAL_RELATIVE_IMPROVEMENT": 0.05,
        "MAX_RESIDUAL_CI_UPPER": 0.10,
        "MIN_DELETION_CONTROL_IMPROVEMENT": 0.10,
        "MAX_FIBER_EFFECT_RATIO": 1.25,
        "MIN_STATE_EFFECT_RETENTION": 0.50,
        "MIN_STATE_INTERVENTION_COSINE": 0.20,
        "MAX_INTERVENTION_OOD_RATE": 0.05,
        "MAX_COMMUTATIVITY_REFERENCE_ERROR_RATIO": 1.25,
        "MIN_STAGE34_CONTROL_ADVANTAGE": 0.10,
    }
    for name, expected in expected_values.items():
        assert assigned_value(tree, name) == expected, f"unexpected {name}"

    pools = [
        assigned_value(tree, "CONSTRUCTION_TRAJECTORY_POOL"),
        assigned_value(tree, "MODEL_SELECTION_TRAJECTORY_POOL"),
        assigned_value(tree, "CALIBRATION_TRAJECTORY_POOL"),
        assigned_value(tree, "EVALUATION_TRAJECTORY_POOL"),
    ]
    assert pools == [
        list(range(10000, 11200)), list(range(11200, 12400)),
        list(range(12400, 13600)), list(range(13600, 16000)),
    ]
    assert all(
        not set(pools[left]) & set(pools[right])
        for left in range(len(pools)) for right in range(left + 1, len(pools))
    )
    core_words = assigned_value(tree, "CORE_WORD_SPECS")
    evaluation_words = assigned_value(tree, "EVALUATION_WORD_SPECS")
    assert {len(row["angles"]) for row in core_words} == {1, 2, 3, 4}
    assert {len(row["angles"]) for row in evaluation_words} == {5, 6, 7, 8}

    joined = "\n".join(code_cells)
    require_all(
        joined,
        [
            "def action_contrast_signature(", "def fit_response_chart(",
            "def response_coordinates(", "def grouped_ridge_oof(",
            "def fit_supervised_subspace(", "def split_carrier_delta(",
            "def matched_fiber_pairs(", "def intervention_ood_ratio(",
            "class Stage34Gates:", "def derive_stage34_decision(",
            '"model_outputs_used": False', '"cross_model_map_count": 0',
            "CANONICAL_RESPONSE_CHART", "deleted_selection",
            "same_model_full_swap_positive", "register_forward_hook",
            "intervention_lookup", "MAX_INTERVENTION_OOD_RATE",
            "stage34_pfca_result_bundle_", "planning_run\": False",
        ],
        "predictive-fiber implementation",
    )
    for prohibited in [
        "fit_whitened_similarity(", "CARRIER_MAPS", "frozen_carrier_map",
        "STATE_MAPS", "transported_planning", "DummyModel", "FakeModel",
        "mock_predictions", "synthetic_predictions", "torch.autograd",
        ".backward(", "torch.func.jvp", "torch.func.vjp",
    ]:
        assert prohibited not in joined, f"prohibited Stage 34 machinery: {prohibited}"

    causal_source = function_source(code_cells, "evaluation_record_arrays")
    require_all(
        causal_source,
        ["source_coordinates", "state_carrier_sketch", "state_carrier"],
        "matched causal source state",
    )
    hook_source = function_source(code_cells, "forward_with_trace")
    require_all(
        hook_source,
        ["register_forward_hook", "intervention_by_step.get", "handle.remove()"],
        "native recurrent intervention",
    )
    direct_source = function_source(code_cells, "direct_simulator_response_signature")
    assert "len(CORE_WORD_NAMES) + 4" in direct_source
    assert "len(ZERO_WORD_NAMES)" not in direct_source
    action_cell = code_cells[8]
    assert action_cell.index("EVALUATION_OPENED = True") < action_cell.index(
        "for short in [\"jepa\", \"dino\"]:"
    )

    observed = assigned_value(tree, "NOTEBOOK_PROTOCOL_SHA256")
    validate_protocol_digest(notebook, config, observed)
    print("Stage 34 notebook validation passed")


if __name__ == "__main__":
    validate()
