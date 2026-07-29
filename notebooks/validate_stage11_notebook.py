import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "11_action_response_geometry_pilot.ipynb"
BUILDER = ROOT / "build_stage11_notebook.py"


def notebook_payload():
    return json.loads(NOTEBOOK.read_text())


def function_source(code_cells, function_name):
    for source in code_cells:
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == function_name:
                    segment = ast.get_source_segment(source, node)
                    if segment is None:
                        raise AssertionError(
                            f"could not recover source for {function_name}"
                        )
                    return segment
    raise AssertionError(f"missing function {function_name}")


def validate_structure():
    payload = notebook_payload()
    if len(payload["cells"]) != 11:
        raise AssertionError(
            f"expected 11 cells, found {len(payload['cells'])}"
        )
    code_cells = [
        "".join(cell.get("source", []))
        for cell in payload["cells"]
        if cell["cell_type"] == "code"
    ]
    if len(code_cells) != 10:
        raise AssertionError(
            f"expected 10 code cells, found {len(code_cells)}"
        )
    for index, source in enumerate(code_cells):
        try:
            ast.parse(source)
        except SyntaxError as error:
            raise AssertionError(
                f"code cell {index} has invalid Python: {error}"
            ) from error
    for index, cell in enumerate(payload["cells"]):
        if cell.get("id") != f"stage11-{index:02d}":
            raise AssertionError(
                f"cell {index} has a stale or missing id"
            )
        if cell.get("outputs"):
            raise AssertionError(
                f"cell {index} contains stale outputs"
            )
        if (
            cell["cell_type"] == "code"
            and cell.get("execution_count") is not None
        ):
            raise AssertionError(
                f"cell {index} contains a stale execution count"
            )
    if (
        payload.get("metadata", {})
        .get("colab", {})
        .get("name")
        != NOTEBOOK.name
    ):
        raise AssertionError("Colab metadata has the wrong name")

    joined = "\n".join(code_cells)
    required = [
        'RUN_MODE = "pilot"',
        "NUM_STATES = 36",
        "MOUNT_DRIVE = False",
        "DOWNLOAD_PHASE_C_RESCUE = True",
        "DOWNLOAD_RESULTS = True",
        "stage11_arga_whitened_centered_v1",
        "SCREENING_ADAPTATION_SEEDS = [11401]",
        "CONFIRMATION_ADAPTATION_SEEDS = [11419]",
        "INITIAL_EPOCH_LIMIT = 6",
        "EXTENSION_EPOCH_LIMITS = [8, 10]",
        "EARLY_STOPPING_CHECKPOINT_PATIENCE = 2",
        "TRAINING_GEOMETRY_PROJECTION_SEEDS = [13001, 13019]",
        "EVALUATION_PROJECTION_SEEDS = [14011, 14029, 14047]",
        "set(TRAINING_GEOMETRY_PROJECTION_SEEDS).isdisjoint",
        "fidelity_constrained_latent_only",
        "fidelity_constrained_shuffled_geometry",
        "fidelity_constrained_matched_geometry",
        "def projected_centered(",
        "def fit_geometry_reference(",
        "def geometry_loss_by_horizon(",
        "torch.einsum(",
        "uses_physical_pose_goal_cost_or_readout",
        "training_decoders_used_by_objective",
        "def screening_gate(",
        "skipping confirmation seed",
        "stage11_phase_c_checkpoint_rescue.zip",
        "colab_files.download",
        "def evaluate_unseen_geometry(",
        "stage11_geometry_task_contrasts.csv",
        "PROMOTE_TO_FULL_RUN",
        "GEOMETRY_ONLY_DIAGNOSIS",
        "STOP_NO_DIRECT_GEOMETRY_SIGNAL",
        "compute_stage11_cache_binding",
        "atomic_torch_save",
        "latest checkpoint state machine",
        "optimizer.load_state_dict",
        "current_action_path_checksum",
        "stage11_result_bundle.zip",
        'print("RUN_STATUS:"',
        "COMPUTE_GATED_EXPLORATORY_PILOT",
    ]
    missing = [needle for needle in required if needle not in joined]
    if missing:
        raise AssertionError(
            f"missing required Stage 11 elements: {missing}"
        )

    training = function_source(
        code_cells, "train_one_geometry_path"
    )
    geometry_loss = function_source(
        code_cells, "geometry_loss_by_horizon"
    )
    prohibited_training = [
        "decoded_costs_from_tokens",
        "physical_cost",
        "torch_decoded_task_cost",
        'task["goal"]',
        "ranking_metrics",
    ]
    present = [
        needle
        for needle in prohibited_training
        if needle in training or needle in geometry_loss
    ]
    if present:
        raise AssertionError(
            "readout-free training path contains forbidden decision "
            f"supervision: {present}"
        )
    if "deterministic_non_null_derangement" not in training:
        raise AssertionError("shuffled correspondence control is missing")
    if "native_horizon_loss" not in training:
        raise AssertionError("native fidelity anchor is missing")
    if "project_action_path_trust_region" not in training:
        raise AssertionError("parameter trust region is missing")
    if "latest_path.exists()" not in training:
        raise AssertionError("atomic checkpoint resume is missing")

    phase_d = code_cells[7]
    if "unconstrained_matched_fpma" in phase_d:
        raise AssertionError("Stage 10-only treatment leaked into Stage 11")
    if "EXECUTED_ADAPTATION_SEEDS" not in phase_d:
        raise AssertionError("evaluation ignores the sequential seed gate")
    return payload, code_cells


def validate_centered_geometry():
    rng = np.random.default_rng(16001)
    for _ in range(200):
        actions = 10
        dimension = 17
        prediction = rng.normal(size=(actions, dimension))
        target = rng.normal(size=(actions, dimension))
        centered_prediction = prediction - prediction.mean(
            axis=0, keepdims=True
        )
        centered_target = target - target.mean(
            axis=0, keepdims=True
        )
        baseline = centered_prediction - centered_target

        prediction_offset = rng.normal(size=(1, dimension))
        target_offset = rng.normal(size=(1, dimension))
        shifted = (
            prediction
            + prediction_offset
            - (prediction + prediction_offset).mean(
                axis=0, keepdims=True
            )
            - target
            - target_offset
            + (target + target_offset).mean(
                axis=0, keepdims=True
            )
        )
        np.testing.assert_allclose(
            shifted, baseline, rtol=1e-12, atol=1e-12
        )

        pairwise = []
        for left in range(actions):
            for right in range(left + 1, actions):
                pairwise.append(
                    (prediction[left] - prediction[right])
                    - (target[left] - target[right])
                )
        pairwise = np.asarray(pairwise)
        lhs = float(np.sum(pairwise**2))
        rhs = float(actions * np.sum(baseline**2))
        np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-12)


def validate_linear_readout_bound():
    rng = np.random.default_rng(16019)
    for _ in range(500):
        dimension = 12
        matrix = rng.normal(size=(dimension, dimension))
        covariance = matrix @ matrix.T + 0.2 * np.eye(dimension)
        eigenvalue, eigenvector = np.linalg.eigh(covariance)
        square_root = (
            eigenvector * np.sqrt(eigenvalue)[None]
        ) @ eigenvector.T
        inverse_square_root = (
            eigenvector * (1.0 / np.sqrt(eigenvalue))[None]
        ) @ eigenvector.T
        readout = rng.normal(size=dimension)
        action_pair_error = rng.normal(size=dimension)
        margin_error = abs(float(readout @ action_pair_error))
        bound = float(
            np.linalg.norm(square_root @ readout)
            * np.linalg.norm(
                inverse_square_root @ action_pair_error
            )
        )
        if margin_error > bound + 1e-10:
            raise AssertionError(
                f"linear readout bound failed: {margin_error} > {bound}"
            )


def validate_derangement_and_gate():
    for state_id in range(50):
        rng = np.random.default_rng(11401 + 1009 * state_id)
        original = np.arange(1, 10, dtype=np.int64)
        permuted = original.copy()
        for _ in range(1000):
            rng.shuffle(permuted)
            if np.all(permuted != original):
                break
        result = np.concatenate(
            [np.asarray([0], dtype=np.int64), permuted]
        )
        assert result[0] == 0
        assert np.all(result[1:] != np.arange(1, 10))
        assert sorted(result.tolist()) == list(range(10))

    def passes(environment_ratios):
        strong = [
            sum(
                value["matched_over_frozen"] <= 0.97
                and value["matched_over_shuffled"] <= 0.99
                for value in ratios
            )
            >= 2
            for ratios in environment_ratios
        ]
        catastrophic = any(
            value["matched_over_frozen"] > 1.05
            for ratios in environment_ratios
            for value in ratios
        )
        return any(strong) and not catastrophic

    good = [
        [
            {
                "matched_over_frozen": 0.95,
                "matched_over_shuffled": 0.98,
            },
            {
                "matched_over_frozen": 0.96,
                "matched_over_shuffled": 0.97,
            },
            {
                "matched_over_frozen": 1.01,
                "matched_over_shuffled": 1.00,
            },
        ],
        [
            {
                "matched_over_frozen": 1.01,
                "matched_over_shuffled": 1.00,
            }
        ]
        * 3,
    ]
    assert passes(good)
    harmed = json.loads(json.dumps(good))
    harmed[1][0]["matched_over_frozen"] = 1.06
    assert not passes(harmed)
    null = [
        [
            {
                "matched_over_frozen": 1.0,
                "matched_over_shuffled": 1.0,
            }
            for _ in range(3)
        ]
        for _ in range(2)
    ]
    assert not passes(null)


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
        raise AssertionError(
            "builder output is not deterministic relative to committed notebook"
        )


if __name__ == "__main__":
    validate_structure()
    validate_centered_geometry()
    validate_linear_readout_bound()
    validate_derangement_and_gate()
    validate_builder_reproducibility()
    print("Stage 11 notebook validation passed")
