import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from cf_faithfulness.stage13b_geometry import array_sha256, frozen_action_bank


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
NOTEBOOK = ROOT / "13b_outcome_geometry_diagnostic.ipynb"
BUILDER = ROOT / "build_stage13b_outcome_geometry_notebook.py"
DESIGN = (
    REPOSITORY
    / "results/bundles/stage12_result_bundle/pusht_design.npz"
)


def payload():
    return json.loads(NOTEBOOK.read_text())


def validate_structure():
    notebook = payload()
    if len(notebook["cells"]) != 12:
        raise AssertionError(
            f"expected 12 cells, found {len(notebook['cells'])}"
        )
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    if len(code_cells) != 11:
        raise AssertionError(
            f"expected 11 code cells, found {len(code_cells)}"
        )
    for index, source in enumerate(code_cells):
        try:
            ast.parse(source)
        except SyntaxError as error:
            raise AssertionError(
                f"invalid Python in code cell {index}: {error}"
            ) from error
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("id") != f"stage13b-{index:02d}":
            raise AssertionError(f"bad cell id at {index}")
        if cell.get("outputs"):
            raise AssertionError(f"stale outputs in cell {index}")
        if (
            cell["cell_type"] == "code"
            and cell.get("execution_count") is not None
        ):
            raise AssertionError(f"stale execution count in cell {index}")
    if notebook["metadata"]["colab"]["name"] != NOTEBOOK.name:
        raise AssertionError("bad Colab notebook name")
    if notebook["metadata"].get("accelerator") != "GPU":
        raise AssertionError("notebook does not request a GPU")

    joined = "\n".join(code_cells)
    required = [
        'RUN_MODE = "full"',
        "NUM_STATES = 80",
        "ACTIONS_PER_STATE = 13",
        "802129bd281fdd2d42a395429e5a0e00df2dc10032b339ecb8bdc8b2521d9fd2",
        "E1_TASKS = 12",
        "E1_STATES_PER_TASK = 4",
        "CONFIRMATION_TASKS = 8",
        "CONFIRMATION_STATES_PER_TASK = 4",
        "RANKS = [1, 2, 4, 6, 8, 12, 16, 24, 32]",
        "len(SKETCH_SEEDS) == 33",
        "NULL_DRAWS = 1024",
        "LOCAL_PERMUTATIONS = 1024",
        "BOOTSTRAP_DRAWS = 10000",
        'EXPERIMENT_SOURCE_REF = "stage13b-v1"',
        "def fit_dual_pca(",
        "def weighted_dual_pca(",
        "def native_loto(",
        "def local_loto(",
        "def task_split_overlap(",
        "def hierarchical_bootstrap_indices(",
        "def exact_positive_sign_test(",
        "def one_sided_t_lower(",
        "covariance_shaped_coordinates(",
        "analytic_haar_expectation.json",
        "e1_task_learning_curve.csv",
        "e1_state_learning_curve.csv",
        "frozen_confirmation_preregistration.json",
        "confirmation_freeze_certificate.json",
        "confirmation_target_shards_existing_at_freeze",
        'generate_truth("untouched_confirmation"',
        'encode_targets("untouched_confirmation"',
        "PROMOTE_GLOBAL_OUTCOME_VOCABULARY_TO_J_LENS",
        "PROMOTE_SEPARATE_HORIZON_J_LENSES",
        "PROMOTE_STATE_CONDITIONED_OUTCOME_TANGENT_BUNDLE",
        "STOP_DISTRIBUTED_HIGHER_RANK_OUTCOME_SPACE",
        "STOP_NO_REPLICABLE_GLOBAL_OR_LOCAL_JOW_GEOMETRY",
        '"jacobians_performed": False',
        '"causal_interventions_performed": False',
        "stage13b_full_evidence_bundle.zip",
        "target_token_manifest.json",
        "source_identity.json",
    ]
    missing = [needle for needle in required if needle not in joined]
    if missing:
        raise AssertionError(f"missing Stage 13b elements: {missing}")
    prohibited = [
        "torch.autograd.grad(",
        "torch.autograd.functional.jacobian",
        "torch.func.jacrev",
        "optimizer.step(",
        ".backward(",
        "matched.pt",
        "shuffled.pt",
    ]
    present = [needle for needle in prohibited if needle in joined]
    if present:
        raise AssertionError(
            f"representation-only notebook contains prohibited work: {present}"
        )

    freeze_cell = "".join(notebook["cells"][9]["source"])
    confirmation_cell = "".join(notebook["cells"][10]["source"])
    if "frozen_confirmation_preregistration.json" not in freeze_cell:
        raise AssertionError("E1 cell does not write the freeze")
    if "encode_targets(\"untouched_confirmation\"" in freeze_cell:
        raise AssertionError("confirmation is encoded before the freeze")
    if "freeze_is_valid()" not in confirmation_cell:
        raise AssertionError("confirmation does not fail closed on freeze hash")
    return code_cells


def validate_design_assets():
    labels, actions = frozen_action_bank()
    if len(labels) != 13 or actions.shape != (13, 15, 2):
        raise AssertionError("bad frozen action design")
    if (
        array_sha256(actions)
        != "802129bd281fdd2d42a395429e5a0e00df2dc10032b339ecb8bdc8b2521d9fd2"
    ):
        raise AssertionError("frozen action hash changed")
    for prefix in [5, 15]:
        unique = {row[:prefix].tobytes() for row in actions[1:]}
        if len(unique) != 12:
            raise AssertionError(f"action collision at prefix {prefix}")
    with np.load(DESIGN) as design:
        task_ids = design["task_ids"]
    e0 = {5, 28, 45, 68, 77, 94, 0, 30, 34, 63, 80, 39}
    selected_total = 0
    for task_id in range(12):
        available = [
            state_id
            for state_id, value in enumerate(task_ids)
            if int(value) == task_id and state_id not in e0
        ]
        if len(available) < 4:
            raise AssertionError(
                f"task {task_id} has only {len(available)} unused states"
            )
        selected_total += 4
    if selected_total != 48:
        raise AssertionError("E1 does not contain 48 states")


def validate_builder_reproducibility():
    before = hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()
    subprocess.run(
        [sys.executable, str(BUILDER)],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    )
    after = hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()
    if before != after:
        raise AssertionError("Stage 13b builder is not deterministic")


if __name__ == "__main__":
    validate_structure()
    validate_design_assets()
    validate_builder_reproducibility()
    print("Stage 13b outcome-geometry notebook validation passed")
