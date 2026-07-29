import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "12_shared_target_metric_bridge.ipynb"
BUILDER = ROOT / "build_stage12_notebook.py"


def payload():
    return json.loads(NOTEBOOK.read_text())


def function_source(code_cells, function_name):
    for source in code_cells:
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                segment = ast.get_source_segment(source, node)
                if segment is None:
                    raise AssertionError(
                        f"could not recover {function_name}"
                    )
                return segment
    raise AssertionError(f"missing function {function_name}")


def validate_structure():
    notebook = payload()
    if len(notebook["cells"]) != 13:
        raise AssertionError(
            f"expected 13 cells, got {len(notebook['cells'])}"
        )
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    if len(code_cells) != 11:
        raise AssertionError(
            f"expected 11 code cells, got {len(code_cells)}"
        )
    for index, source in enumerate(code_cells):
        try:
            ast.parse(source)
        except SyntaxError as error:
            raise AssertionError(
                f"invalid Python in code cell {index}: {error}"
            ) from error
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("id") != f"stage12-{index:02d}":
            raise AssertionError(f"bad cell id at {index}")
        if cell.get("outputs"):
            raise AssertionError(f"stale output in cell {index}")
        if (
            cell["cell_type"] == "code"
            and cell.get("execution_count") is not None
        ):
            raise AssertionError(
                f"stale execution count in cell {index}"
            )
    if (
        notebook["metadata"]["colab"]["name"]
        != NOTEBOOK.name
    ):
        raise AssertionError("bad Colab notebook name")

    joined = "\n".join(code_cells)
    required = [
        'RUN_MODE = "full"',
        "NUM_STATES = 96",
        "STAGE12_FORCE_ALL_TRANSITION_SEEDS = True",
        "ADAPTATION_SEEDS == [11401, 11419, 11437]",
        "BRIDGE_PROJECTION_DIM = 128",
        "BRIDGE_PROJECTION_SEED = 15001",
        "BRIDGE_RANKS = [2, 8]",
        "BRIDGE_REGULARIZERS = [1e-3, 1e-2]",
        "BRIDGE_OPTIMIZATION_SEEDS = [15101, 15119]",
        "def fit_bridge_whitener(",
        "def fit_one_shared_metric(",
        "uses_predicted_rollouts",
        "deterministic_goal_derangement",
        "def evaluate_phase_a(",
        "def evaluate_phase_b(",
        "PROMOTE_TO_UNTOUCHED_TASK_CONFIRMATION",
        "STOP_METRIC_CLASS_NOT_VIABLE",
        "STOP_NO_CAUSAL_BRIDGE_SIGNAL",
        "AMBIGUOUS_DO_NOT_TUNE_ON_DEVELOPMENT_TASKS",
        "stage12_result_bundle.zip",
        "INCLUDE_ALL_TRANSITION_CHECKPOINTS_IN_ZIP",
        'print("RUN_STATUS:"',
    ]
    missing = [needle for needle in required if needle not in joined]
    if missing:
        raise AssertionError(f"missing Stage 12 elements: {missing}")

    fitting = function_source(code_cells, "fit_one_shared_metric")
    prohibited = [
        "load_variant_features",
        "adaptation_seed",
        "development_holdout",
    ]
    present = [needle for needle in prohibited if needle in fitting]
    if present:
        raise AssertionError(
            f"metric fitting leaks treatment information: {present}"
        )
    required_exclusions = [
        '"uses_predicted_rollouts": False',
        '"uses_treatment_identity": False',
        '"uses_candidate_identity_as_feature": False',
        '"uses_development_outcomes": False',
        '"uses_physical_decoder": False',
    ]
    missing_exclusions = [
        item for item in required_exclusions if item not in fitting
    ]
    if missing_exclusions:
        raise AssertionError(
            f"metric fitting lacks leakage declarations: {missing_exclusions}"
        )
    return code_cells


def validate_metric_math(code_cells):
    namespace = {"np": np}
    for function_name in [
        "bridge_metric_matrix_numpy",
        "bridge_metric_costs_numpy",
    ]:
        exec(function_source(code_cells, function_name), namespace)
    matrix_function = namespace["bridge_metric_matrix_numpy"]
    cost_function = namespace["bridge_metric_costs_numpy"]
    rng = np.random.default_rng(17001)
    for rank in [2, 8]:
        low_rank = rng.normal(size=(rank, 128))
        matrix = matrix_function(low_rank)
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-10)
        eigenvalue = np.linalg.eigvalsh(matrix)
        assert eigenvalue.min() > 0
        np.testing.assert_allclose(np.trace(matrix), 128.0, atol=1e-9)
    features = rng.normal(size=(5, 10, 3, 128))
    goals = rng.normal(size=(5, 128))
    matrix = np.eye(128)
    beta = np.asarray([0.5, 1.0, 2.0])
    costs = cost_function(features, goals, matrix, beta)
    expected = np.sum(
        (features - goals[:, None, None]) ** 2, axis=-1
    ) * beta[None, None]
    np.testing.assert_allclose(costs, expected, rtol=1e-12, atol=1e-12)


def validate_builder_reproducibility():
    before = hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()
    subprocess.run(
        [sys.executable, str(BUILDER)],
        cwd=ROOT.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    after = hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()
    if before != after:
        raise AssertionError("Stage 12 builder is not deterministic")


if __name__ == "__main__":
    cells = validate_structure()
    validate_metric_math(cells)
    validate_builder_reproducibility()
    print("Stage 12 notebook validation passed")
