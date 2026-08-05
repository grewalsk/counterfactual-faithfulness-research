import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "21_coherent_interface_and_heldout_utility.ipynb"
BASE = json.loads((ROOT / "20_causal_planner_steering.ipynb").read_text())
NUMERICAL = ROOT.parent / "src/cf_faithfulness/stage21_coherent_utility.py"
RIDGE_NUMERICAL = ROOT.parent / "src/cf_faithfulness/stage15_bundle.py"


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(keepends=True),
    }


def markdown(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
    }


def base_source(index):
    return "".join(BASE["cells"][index]["source"])


def assigned_uppercase_names(source):
    tree = ast.parse(source)
    names = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id.isupper():
                names.append(target.id)
    return list(dict.fromkeys(names))


def function_sources(source, names):
    tree = ast.parse(source)
    by_name = {
        node.name: ast.get_source_segment(source, node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    missing = [name for name in names if name not in by_name]
    if missing:
        raise RuntimeError(f"missing source definitions: {missing}")
    return "\n\n\n".join(by_name[name] for name in names)


introduction = r'''# Stage 21: coherent handoff and held-out causal-subspace utility

Stage 21 resolves the ambiguity exposed by Stage 20 with two explicitly
separate tests.

**Phase A — coherent handoff.**  The same prespecified rank-2/3/4 targets are
used, but complete candidate swaps are compared after predictor block 4 and
after block 5, the last action-conditioned block.  A direct final-output
permutation is the exact oracle.  This determines whether Stage 20 produced a
hybrid because block 5 continued to receive the recipient action.  Phase A
uses no simulator outcome in target definition or scoring.

**Phase B — useful held-out action selection.**  The frozen Stage 18 rank-128
coordinates are used as inputs to one goal-independent linear correction of
the frozen physical decoder's candidate-centered pose error.  The correction
is fitted on construction states, its ridge is selected on calibration
states, and it is refit on construction plus calibration.  The shuffled basis
and four frozen empirical-span random bases receive exactly the same targets,
ridge grid, and sample budget.  On evaluation states all predicted score
vectors and chosen actions are cryptographically frozen before evaluation
simulator truth can be opened.  Physical regret is then compared against the
untouched planner and matched controls.

There is no subspace refit, goal-conditioned training, evaluation-oracle
choice, visual scoring, Jacobian, JVP, VJP, or model-weight update.  A utility
pass would show that coordinates from an already causal representation can
support a calibrated model-based action correction; it would not be a native
policy, multi-step control, or cross-environment result.

Return `stage21_coherent_utility_result_bundle_<signature>.zip`.
'''


configuration = r'''# SINGLE CONFIGURATION BLOCK
# Required Colab secrets for a claim-eligible pilot:
# STAGE21_RUN_MODE=pilot
# STAGE21_SOURCE_COMMIT=<full 40-hex commit shown in the handoff>
# STAGE21_RUN_NONCE=<new unique label, for example coherent_utility_20260804_a>
RUN_MODE = "smoke"
EXPERIMENT_SOURCE_REF = ""
RUN_NONCE = "smoke"
STAGE18_SUBSPACE_PATH = (
    "/content/drive/MyDrive/counterfactual_faithfulness_stage18_rank64/"
    "pilot_f1b34beffcac/subspaces/frozen_rank64_confirmation_subspaces.npz"
)
STAGE19_DECISION_PATH = (
    "/content/drive/MyDrive/counterfactual_faithfulness_stage19_transfer/"
    "pilot_b7f2b6cef37f/stage19_decision.json"
)
STAGE20_DECISION_PATH = (
    "/content/drive/MyDrive/counterfactual_faithfulness_stage20_steering/"
    "pilot_234bd6f0f2ae/stage20_decision.json"
)
try:
    from google.colab import userdata as _colab_userdata

    RUN_MODE = str(_colab_userdata.get("STAGE21_RUN_MODE") or RUN_MODE).strip().lower()
    EXPERIMENT_SOURCE_REF = str(
        _colab_userdata.get("STAGE21_SOURCE_COMMIT") or EXPERIMENT_SOURCE_REF
    ).strip()
    RUN_NONCE = str(_colab_userdata.get("STAGE21_RUN_NONCE") or RUN_NONCE).strip()
    STAGE18_SUBSPACE_PATH = str(
        _colab_userdata.get("STAGE21_STAGE18_SUBSPACE_PATH") or STAGE18_SUBSPACE_PATH
    ).strip()
    STAGE19_DECISION_PATH = str(
        _colab_userdata.get("STAGE21_STAGE19_DECISION_PATH") or STAGE19_DECISION_PATH
    ).strip()
    STAGE20_DECISION_PATH = str(
        _colab_userdata.get("STAGE21_STAGE20_DECISION_PATH") or STAGE20_DECISION_PATH
    ).strip()
except Exception:
    pass

if RUN_MODE == "pilot":
    if RUN_NONCE in {"", "smoke"}:
        raise ValueError("pilot mode requires a unique STAGE21_RUN_NONCE")
    if not all(value.isalnum() or value in "-_" for value in RUN_NONCE):
        raise ValueError("STAGE21_RUN_NONCE may contain only letters, numbers, '-' and '_'")

MOUNT_DRIVE = True
DOWNLOAD_RESULTS = True
CONTINUE_AFTER_BENCHMARK = True
MAX_ESTIMATED_TOTAL_MINUTES = 90.0
FRESH_RUN_REQUIRED = True

OUTPUT_DIR = "/content/counterfactual_faithfulness_stage21_coherent_utility"
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/counterfactual_faithfulness_stage21_coherent_utility"

PROTOCOL_ID = "stage21-coherent-handoff-heldout-causal-subspace-utility-v1"
NOTEBOOK_PROTOCOL_SHA256 = "__PROTOCOL_DIGEST__"
EVIDENCE_STATUS = "CONFIRMATORY_ONLY_IF_SOURCE_PRIORS_SPLITS_AND_FRESHNESS_BOUND"
EXPERIMENT_REPOSITORY = "grewalsk/counterfactual-faithfulness-research"
EXPERIMENT_NOTEBOOK_PATH = "notebooks/21_coherent_interface_and_heldout_utility.ipynb"
EXPERIMENT_BUILDER_PATH = "notebooks/build_stage21_coherent_interface_utility_notebook.py"
EXPERIMENT_NUMERICAL_PATH = "src/cf_faithfulness/stage21_coherent_utility.py"

SEED = 21101
DESIGN_SEED = 21137
MODEL_NAME = "jepa_wm_pusht"
ENVIRONMENT = "PushT"
FRAMESKIP = 5
PRIMARY_HORIZON = 3
TARGET_STEPS = [PRIMARY_HORIZON]
SUBSPACE_BLOCK = 4
LAST_ACTION_CONDITIONED_BLOCK = 5
ACTIVE_BLOCKS = [SUBSPACE_BLOCK, LAST_ACTION_CONDITIONED_BLOCK]
EXPECTED_CARRIER_CHANNELS = 400

EXPECTED_STAGE18_SUBSPACE_SHA256 = "2f9c496d54623a9062e465a18c70039acc18cb8a1cc2833a5f4ade162ca3f90b"
EXPECTED_STAGE18_SOURCE_COMMIT = "16edd247cddcb1aa121340eb5fa42bd9e07004c3"
EXPECTED_STAGE18_STATUS = "CONFIRMED_BIDIRECTIONAL_RANK64_MEDIATOR"
EXPECTED_STAGE18_AMBIENT_DIMENSION = 102400
EXPECTED_STAGE18_MAX_RANK = 128
EXPECTED_STAGE19_DECISION_SHA256 = "493fdf5c707189caea11043db7d208dbc38677dcf5881008e13bede87f40be9c"
EXPECTED_STAGE19_SOURCE_IDENTITY_SHA256 = "6fad7d1ee14efa0898125faaa4500a2ab7b62d81591412364b4379c43ec9ffcf"
EXPECTED_STAGE19_SOURCE_COMMIT = "bf8c3950fc1112b38baa2453e39793592537ec47"
EXPECTED_STAGE19_STATUS = "CONFIRMED_TRANSFER_ALL_UNSEEN_ACTION_FAMILIES"
EXPECTED_STAGE20_DECISION_SHA256 = "57e6ec6ab60415d782bd37e773842b96a8fef596ead111df33b7b816c83d601e"
EXPECTED_STAGE20_SOURCE_IDENTITY_SHA256 = "145887eb9ab4828452c38b3a3e689ba66facc255e226cd5458332029929ddb95"
EXPECTED_STAGE20_SOURCE_COMMIT = "ffac9271564c9da7894e7e8a1936df4f4af3ad32"
EXPECTED_STAGE20_STATUS = "PREDICTION_MEDIATOR_TRANSFER_WITHOUT_CONFIRMED_PLANNER_STEERING"

TRANSFER_FAMILIES = ["rotated_direction", "pulsed_equal_impulse"]
SPLIT_POOLS = {
    "construction": list(range(700, 748)),
    "calibration": list(range(748, 780)),
    "evaluation": list(range(780, 828)),
}
SPLIT_TARGETS_PER_FAMILY = {
    "construction": 32,
    "calibration": 16,
    "evaluation": 32,
}
TARGET_BASELINE_RANKS = [1, 2, 3]
TASK_ID_OFFSET = 3000
ACTIONS_PER_STATE = 13
ACTION_STEPS = PRIMARY_HORIZON * FRAMESKIP
APPROACH_DISTANCE = 80.0
MIN_ELIGIBLE_COST_SPREAD = 0.02
MIN_ELIGIBLE_NON_TIED_PAIR_FRACTION = 0.20
MIN_ELIGIBLE_CONTACT_BRANCHES = 2
PHYSICAL_COST_TIE = 1e-4

OUTPUT_SKETCH_DIM = 256
TRAIN_OUTPUT_SKETCH_SEED = 18161
EVAL_OUTPUT_SKETCH_SEED = 18183
CORRECTION_RANK = 128
CORRECTION_RANDOM_DRAWS = 4
RIDGE_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
PERMUTATION_SEED = 21251
BOOTSTRAP_SEED = 21269
BOOTSTRAP_DRAWS = 10000
INTERFACE_PATCHED_FORWARDS_PER_EVALUATION_RECORD = 9
INTERFACE_ROWS_PER_EVALUATION_RECORD = 15
UTILITY_ROWS_PER_EVALUATION_RECORD = 7
MAX_ZERO_EDIT_ERROR = 1e-6

MIN_FINAL_ORACLE_CHOICE_RATE = 1.0
MIN_LAST_BLOCK_OUTPUT_COEFFICIENT = 0.995
MIN_LAST_BLOCK_SCORE_COEFFICIENT = 0.995
MIN_LAST_BLOCK_TARGET_CHOICE_RATE = 0.99
MAX_LAST_BLOCK_SCORE_NRMSE = 0.02
MIN_LAST_VS_PENULTIMATE_CHOICE_GAIN = 0.05
MIN_LEARNED_BLOCK4_OUTPUT_COEFFICIENT = 0.25

MIN_REGRET_IMPROVEMENT_VS_BASELINE = 0.03
MIN_REGRET_GAIN_OVER_RANDOM = 0.02
MIN_REGRET_GAIN_OVER_SHUFFLED = 0.0
MIN_REGRET_GAIN_OVER_WRONG_STATE = 0.02
MIN_PAIRWISE_ACCURACY_GAIN = 0.02
MIN_PAIRWISE_GAIN_OVER_RANDOM = 0.01
MAX_TOP1_ACCURACY_HARM = 0.02

if RUN_MODE == "smoke":
    ACTIVE_SPLIT_POOLS = {name: values[:6] for name, values in SPLIT_POOLS.items()}
    ACTIVE_SPLIT_TARGETS = {name: 2 for name in SPLIT_POOLS}
    ACTIVE_TARGET_BASELINE_RANKS = [1]
    ACTIVE_CORRECTION_RANDOM_DRAWS = 1
    ACTIVE_BOOTSTRAP_DRAWS = 64
    ACTIVE_INTERFACE_PATCHED_FORWARDS_PER_RECORD = 3
    ACTIVE_INTERFACE_ROWS_PER_RECORD = 5
    ACTIVE_UTILITY_ROWS_PER_RECORD = 4
elif RUN_MODE == "pilot":
    ACTIVE_SPLIT_POOLS = SPLIT_POOLS
    ACTIVE_SPLIT_TARGETS = SPLIT_TARGETS_PER_FAMILY
    ACTIVE_TARGET_BASELINE_RANKS = TARGET_BASELINE_RANKS
    ACTIVE_CORRECTION_RANDOM_DRAWS = CORRECTION_RANDOM_DRAWS
    ACTIVE_BOOTSTRAP_DRAWS = BOOTSTRAP_DRAWS
    ACTIVE_INTERFACE_PATCHED_FORWARDS_PER_RECORD = INTERFACE_PATCHED_FORWARDS_PER_EVALUATION_RECORD
    ACTIVE_INTERFACE_ROWS_PER_RECORD = INTERFACE_ROWS_PER_EVALUATION_RECORD
    ACTIVE_UTILITY_ROWS_PER_RECORD = UTILITY_ROWS_PER_EVALUATION_RECORD
else:
    raise ValueError(
        "STAGE21_RUN_MODE must contain only smoke or pilot; "
        f"received {RUN_MODE!r}"
    )

REPO_URL = "https://github.com/facebookresearch/jepa-wms.git"
REPO_COMMIT = "13cf1d9c7e476f53c17714d2e0f1dc239a883ce0"
EXPECTED_HF_REVISION = "9b9c41ef249466630dbf1a20e78391865d07b3b9"
EXPECTED_PRETRAINED_ASSET_SHA256 = {
    "jepa_wm_pusht.pth.tar": "9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb",
    "dinov2_vits14_pretrain.pth": "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9",
}
ASSET_REPOSITORY = "grewalsk/counterfactual-faithfulness-research"
ASSET_COMMIT = "2326e74556f6f81db2560e4396f4cc52c16a28f4"
ASSET_SPECS = {
    "physical_decoders.pt": {
        "path": "results/bundles/stage12_result_bundle/frozen_training_decoders/jepa_wm_pusht_f975a0a746e7_training_decoders.pt",
        "sha256": "51b2dbb0a81df432a2db5b941de83717e9979e761d57365f47d93d2dd0c0c694",
    },
}

assert ACTIVE_BLOCKS == [4, 5]
assert ACTIONS_PER_STATE == 13 and ACTION_STEPS == 15
assert CORRECTION_RANK == 128
assert TARGET_BASELINE_RANKS == [1, 2, 3]
'''
configuration += "\n\nPROTOCOL_CONFIG_KEYS = " + repr(assigned_uppercase_names(configuration)) + "\n"


installation = base_source(2)


setup = base_source(3)
setup = setup.replace("Stage 20", "Stage 21").replace("STAGE20", "STAGE21")
setup = setup.replace("stage20_steering", "stage21_coherent_utility")
setup = setup.replace(
    'PROVENANCE_COUNTS = {"truth_generated": 0, "baseline_generated": 0, "intervention_generated": 0, "patched_forwards_generated": 0, "cache_hits": 0}',
    'PROVENANCE_COUNTS = {"truth_generated": 0, "baseline_generated": 0, '
    '"interface_shards_generated": 0, "patched_forwards_generated": 0, "cache_hits": 0}',
)


analysis_helpers = base_source(4)
analysis_helpers += "\n\n\n" + function_sources(
    RIDGE_NUMERICAL.read_text(), ["fit_ridge", "predict_ridge"]
)
analysis_helpers += "\n\n\n" + function_sources(
    NUMERICAL.read_text(),
    [
        "subspace_coordinates",
        "centered_pose_residual",
        "normalize_pose_orientation",
        "apply_pose_correction",
        "select_ridge_on_calibration",
        "corrected_planning_metrics",
        "counterfactual_interface_metrics",
    ],
)


model_helpers = base_source(5).replace("stage20-jepa-wms", "stage21-jepa-wms")


design = r'''# Freeze all state pools and split membership before simulator or model data.


def trajectory_specs():
    specs = []
    center = np.asarray([256.0, 256.0])
    all_trajectories = [value for values in SPLIT_POOLS.values() for value in values]
    total = len(all_trajectories)
    split_by_trajectory = {
        int(trajectory): split
        for split, values in SPLIT_POOLS.items()
        for trajectory in values
    }
    for design_index, trajectory_id in enumerate(all_trajectories):
        phase = 0.37 + 2.0 * np.pi * design_index / total
        block = center + 44.0 * np.asarray([np.cos(phase), np.sin(phase)])
        block_angle = ((2.07 * phase + np.pi) % (2.0 * np.pi)) - np.pi
        offsets = [np.pi / 3, 2 * np.pi / 3, 4 * np.pi / 3, 5 * np.pi / 3]
        approach = phase + offsets[design_index % 4] + 0.13 * np.sin(3 * design_index)
        agent = block + APPROACH_DISTANCE * np.asarray([np.cos(approach), np.sin(approach)])
        goal_index = (19 * design_index + 7) % total
        goal_phase = 0.83 + 2.0 * np.pi * goal_index / total
        goal_xy = center + 73.0 * np.asarray([np.cos(goal_phase), np.sin(goal_phase)])
        split = split_by_trajectory[int(trajectory_id)]
        common = {
            "design_index": int(design_index),
            "trajectory_id": int(trajectory_id),
            "time_index": 0,
            "physical_step": 0,
            "split": split,
            "evaluation_seed": int(DESIGN_SEED + 1013 * design_index),
            "goal": np.asarray(
                [goal_xy[0], goal_xy[1], ((1.19 * goal_phase + np.pi) % (2.0 * np.pi)) - np.pi],
                dtype=np.float64,
            ),
            "state": np.asarray(
                [agent[0], agent[1], block[0], block[1], block_angle, 0.0, 0.0, 0.0, 0.0, 0.0],
                dtype=np.float64,
            ),
        }
        for family_index, family in enumerate(TRANSFER_FAMILIES):
            specs.append(
                {
                    **common,
                    "record_id": int(700000 + 1000 * family_index + trajectory_id),
                    "task_id": int(TASK_ID_OFFSET + design_index),
                    "action_family": family,
                    "family_index": int(family_index),
                }
            )
    return specs


ALL_POOL_SPECS = trajectory_specs()
POOL_SPECS = [
    row for row in ALL_POOL_SPECS
    if row["trajectory_id"] in {
        value for values in ACTIVE_SPLIT_POOLS.values() for value in values
    }
]


def candidate_action_bank(record):
    state = np.asarray(record["state"], dtype=np.float64)
    return unseen_action_bank(
        state[2:4] - state[:2], record["action_family"], ACTION_STEPS
    )


np.savez_compressed(
    DESIGN_DIR / "stage21_split_pool_design.npz",
    record_ids=np.asarray([row["record_id"] for row in ALL_POOL_SPECS]),
    trajectory_ids=np.asarray([row["trajectory_id"] for row in ALL_POOL_SPECS]),
    splits=np.asarray([row["split"] for row in ALL_POOL_SPECS]),
    action_families=np.asarray([row["action_family"] for row in ALL_POOL_SPECS]),
    initial_states=np.stack([row["state"] for row in ALL_POOL_SPECS]),
    goals=np.stack([row["goal"] for row in ALL_POOL_SPECS]),
)
POOL_MANIFEST = {
    "specs": [
        {
            **{key: value for key, value in row.items() if key not in {"state", "goal"}},
            "state": row["state"].tolist(),
            "goal": row["goal"].tolist(),
        }
        for row in ALL_POOL_SPECS
    ],
    "active_split_pools": ACTIVE_SPLIT_POOLS,
    "active_targets_per_split_family": ACTIVE_SPLIT_TARGETS,
    "split_membership_frozen_before_truth": True,
    "evaluation_choice_may_use_evaluation_truth": False,
    "visual_evaluation_used": False,
}
write_json(DESIGN_DIR / "candidate_pool_manifest.json", POOL_MANIFEST)
DESIGN_FREEZE = {
    "created_before_simulator_or_model_data": True,
    "protocol_id": PROTOCOL_ID,
    "run_signature": RUN_SIGNATURE,
    "source_identity": SOURCE_IDENTITY,
    "candidate_pool_sha256": sha256_file(DESIGN_DIR / "stage21_split_pool_design.npz"),
    "pool_manifest_sha256": sha256_file(DESIGN_DIR / "candidate_pool_manifest.json"),
    "subspace_refit_allowed": False,
    "goal_conditioned_correction_allowed": False,
    "model_loaded": bool("MODEL" in globals()),
}
if DESIGN_FREEZE["model_loaded"]:
    raise RuntimeError("model was loaded before Stage 21 design freeze")
write_json(DESIGN_DIR / "design_freeze.json", DESIGN_FREEZE)
'''


truth_generation = function_sources(
    base_source(7),
    [
        "record_task",
        "dynamic_state_from_environment",
        "reset_dynamic_environment",
        "rollout_dynamic_branch",
        "exact_dynamic_restore_test",
        "branch_path",
        "generate_truth",
        "truth_eligibility",
        "select_records",
    ],
)
truth_generation += r'''


if not PIPELINE_FAILED:
    try:
        REPO = configure_repo()
        RESTORE_TEST = exact_dynamic_restore_test(POOL_SPECS[0])
        write_json(OUT / "restore_test.json", RESTORE_TEST)
        generate_truth(POOL_SPECS, "truth_stage21_split_pool")
        if "MODEL" in globals():
            raise RuntimeError("model was loaded before physical split selection")
        SELECTED_RECORDS_BY_SPLIT_FAMILY = {}
        ELIGIBILITY_ROWS = []
        for split in ["construction", "calibration", "evaluation"]:
            SELECTED_RECORDS_BY_SPLIT_FAMILY[split] = {}
            for family in TRANSFER_FAMILIES:
                family_pool = [
                    row for row in POOL_SPECS
                    if row["split"] == split and row["action_family"] == family
                ]
                chosen, rows = select_records(
                    family_pool, ACTIVE_SPLIT_TARGETS[split]
                )
                SELECTED_RECORDS_BY_SPLIT_FAMILY[split][family] = chosen
                ELIGIBILITY_ROWS.extend(rows)
        SELECTED_RECORDS_BY_SPLIT = {
            split: [
                row for family in TRANSFER_FAMILIES
                for row in SELECTED_RECORDS_BY_SPLIT_FAMILY[split][family]
            ]
            for split in ["construction", "calibration", "evaluation"]
        }
        CONSTRUCTION_RECORDS = SELECTED_RECORDS_BY_SPLIT["construction"]
        CALIBRATION_RECORDS = SELECTED_RECORDS_BY_SPLIT["calibration"]
        EVALUATION_RECORDS = SELECTED_RECORDS_BY_SPLIT["evaluation"]
        ALL_SELECTED_RECORDS = (
            CONSTRUCTION_RECORDS + CALIBRATION_RECORDS + EVALUATION_RECORDS
        )
        split_trajectory_sets = {
            split: {int(row["trajectory_id"]) for row in records}
            for split, records in SELECTED_RECORDS_BY_SPLIT.items()
        }
        for left, right in [
            ("construction", "calibration"),
            ("construction", "evaluation"),
            ("calibration", "evaluation"),
        ]:
            if split_trajectory_sets[left].intersection(split_trajectory_sets[right]):
                raise RuntimeError(f"trajectory leakage between {left} and {right}")
        WRONG_EVALUATION_RECORD = {}
        for family in TRANSFER_FAMILIES:
            identifiers = [
                int(row["record_id"])
                for row in SELECTED_RECORDS_BY_SPLIT_FAMILY["evaluation"][family]
            ]
            for index, record_id in enumerate(identifiers):
                WRONG_EVALUATION_RECORD[str(record_id)] = identifiers[(index + 1) % len(identifiers)]
        write_csv(EVIDENCE_DIR / "physical_eligibility_rows.csv", ELIGIBILITY_ROWS)
        SELECTION_CERTIFICATE = {
            "selection_completed_before_model_load": True,
            "selection_used_only_simulator_truth": True,
            "split_trajectory_disjoint": True,
            "selected_record_ids_by_split_family": {
                split: {
                    family: [int(row["record_id"]) for row in records]
                    for family, records in family_map.items()
                }
                for split, family_map in SELECTED_RECORDS_BY_SPLIT_FAMILY.items()
            },
            "selected_counts_by_split_family": {
                split: {family: len(records) for family, records in family_map.items()}
                for split, family_map in SELECTED_RECORDS_BY_SPLIT_FAMILY.items()
            },
            "wrong_evaluation_record_map": WRONG_EVALUATION_RECORD,
            "eligibility_rows_sha256": sha256_file(
                EVIDENCE_DIR / "physical_eligibility_rows.csv"
            ),
        }
        write_json(DESIGN_DIR / "physical_split_selection_freeze.json", SELECTION_CERTIFICATE)
        memory_report("physical_truth_and_split_selection_complete")
    except Exception:
        record_failure("physical_truth_split_selection")
'''


artifact_import = r'''# Bind successful Stages 18–20 before any Stage 21 model activations.
PRIOR_ARTIFACTS_VALIDATED = False
if not PIPELINE_FAILED:
    try:
        verify_executed_notebook_through(
            "# Bind successful Stages 18–20 before any Stage 21 model activations."
        )
        frozen_subspace_path = Path(STAGE18_SUBSPACE_PATH)
        if not frozen_subspace_path.is_file():
            raise FileNotFoundError(f"Stage 18 raw subspace is missing: {frozen_subspace_path}")
        if sha256_file(frozen_subspace_path) != EXPECTED_STAGE18_SUBSPACE_SHA256:
            raise RuntimeError("Stage 18 subspace hash mismatch")
        stage18_run_dir = frozen_subspace_path.parent.parent
        stage18_decision_path = stage18_run_dir / "stage18_decision.json"
        stage18_manifest_path = stage18_run_dir / "subspaces/subspace_manifest.json"
        stage18_source_path = stage18_run_dir / "source_identity.json"
        for path in [stage18_decision_path, stage18_manifest_path, stage18_source_path]:
            if not path.is_file():
                raise FileNotFoundError(f"Stage 18 provenance file is missing: {path}")
        stage18_decision = json.loads(stage18_decision_path.read_text())
        stage18_manifest = json.loads(stage18_manifest_path.read_text())
        stage18_source = json.loads(stage18_source_path.read_text())
        if stage18_decision.get("status") != EXPECTED_STAGE18_STATUS:
            raise RuntimeError("Stage 18 successful decision mismatch")
        if not bool(stage18_decision.get("confirmation_eligible", False)):
            raise RuntimeError("Stage 18 decision was not claim eligible")
        if stage18_manifest.get("subspace_sha256") != EXPECTED_STAGE18_SUBSPACE_SHA256:
            raise RuntimeError("Stage 18 manifest does not bind the subspace")
        if stage18_source.get("resolved_commit") != EXPECTED_STAGE18_SOURCE_COMMIT:
            raise RuntimeError("Stage 18 source commit mismatch")
        with np.load(frozen_subspace_path) as payload:
            FROZEN_SUBSPACES = {name: payload[name].copy() for name in payload.files}
        artifact_contract = validate_stage18_subspace_arrays(
            FROZEN_SUBSPACES,
            ambient=EXPECTED_STAGE18_AMBIENT_DIMENSION,
            max_rank=EXPECTED_STAGE18_MAX_RANK,
        )

        stage19_decision_path = Path(STAGE19_DECISION_PATH)
        stage19_source_path = stage19_decision_path.parent / "source_identity.json"
        if sha256_file(stage19_decision_path) != EXPECTED_STAGE19_DECISION_SHA256:
            raise RuntimeError("Stage 19 decision hash mismatch")
        if sha256_file(stage19_source_path) != EXPECTED_STAGE19_SOURCE_IDENTITY_SHA256:
            raise RuntimeError("Stage 19 source-identity hash mismatch")
        stage19_decision = json.loads(stage19_decision_path.read_text())
        stage19_source = json.loads(stage19_source_path.read_text())
        if stage19_decision.get("status") != EXPECTED_STAGE19_STATUS:
            raise RuntimeError("Stage 19 status mismatch")
        if not bool(stage19_decision.get("confirmation_eligible", False)):
            raise RuntimeError("Stage 19 was not claim eligible")
        if stage19_source.get("resolved_commit") != EXPECTED_STAGE19_SOURCE_COMMIT:
            raise RuntimeError("Stage 19 source commit mismatch")

        stage20_decision_path = Path(STAGE20_DECISION_PATH)
        stage20_source_path = stage20_decision_path.parent / "source_identity.json"
        if sha256_file(stage20_decision_path) != EXPECTED_STAGE20_DECISION_SHA256:
            raise RuntimeError("Stage 20 decision hash mismatch")
        if sha256_file(stage20_source_path) != EXPECTED_STAGE20_SOURCE_IDENTITY_SHA256:
            raise RuntimeError("Stage 20 source-identity hash mismatch")
        stage20_decision = json.loads(stage20_decision_path.read_text())
        stage20_source = json.loads(stage20_source_path.read_text())
        if stage20_decision.get("status") != EXPECTED_STAGE20_STATUS:
            raise RuntimeError("Stage 20 diagnosis mismatch")
        if not bool(stage20_decision.get("confirmation_eligible", False)):
            raise RuntimeError("Stage 20 was not claim eligible")
        if stage20_source.get("resolved_commit") != EXPECTED_STAGE20_SOURCE_COMMIT:
            raise RuntimeError("Stage 20 source commit mismatch")

        PRIOR_ARTIFACT_CERTIFICATE = {
            "validated_before_stage21_model_activations": True,
            "stage18_subspace_path": str(frozen_subspace_path),
            "stage18_subspace_sha256": sha256_file(frozen_subspace_path),
            "stage18_artifact_contract": artifact_contract,
            "stage19_decision_sha256": sha256_file(stage19_decision_path),
            "stage19_status": stage19_decision["status"],
            "stage20_decision_sha256": sha256_file(stage20_decision_path),
            "stage20_status": stage20_decision["status"],
            "stage21_subspace_refit": False,
            "stage21_model_weight_update": False,
        }
        write_json(OUT / "prior_artifact_certificate.json", PRIOR_ARTIFACT_CERTIFICATE)
        PRIOR_ARTIFACTS_VALIDATED = True
        memory_report("prior_artifacts_validated")
    except Exception:
        record_failure("prior_artifact_import")
'''


model_definitions = function_sources(
    base_source(9),
    [
        "state_model_inputs",
        "baseline_path",
        "load_baseline",
        "extract_baselines",
        "carrier_for_block",
        "hook_identity_test",
        "forward_benchmark",
    ],
)
model_definitions = model_definitions.replace(
    "ACTIVE_INTERVENTION_FORWARDS_PER_RECORD",
    "ACTIVE_INTERFACE_PATCHED_FORWARDS_PER_RECORD",
).replace("ALL_EVALUATION_RECORDS", "EVALUATION_RECORDS")
model_baselines = r'''# Load frozen JEPA-WM and extract split-bound block-4/block-5 baselines.
''' + model_definitions + r'''


MODEL_BASELINES_OPENED = False
if not PIPELINE_FAILED:
    try:
        if not PRIOR_ARTIFACTS_VALIDATED:
            raise RuntimeError("prior artifacts must validate before model loading")
        MODEL, PREPROCESSOR, PREDICTOR, PREDICTOR_BLOCK_MODULES = load_frozen_model()
        if len(PREDICTOR_BLOCK_MODULES) != 6:
            raise RuntimeError("predictor block count changed")
        TRAIN_OUTPUT_PROJECTOR = CountSketchProjector(
            256 * 384, OUTPUT_SKETCH_DIM, TRAIN_OUTPUT_SKETCH_SEED
        )
        EVAL_OUTPUT_PROJECTOR = CountSketchProjector(
            256 * 384, OUTPUT_SKETCH_DIM, EVAL_OUTPUT_SKETCH_SEED
        )
        DECODE_PHYSICAL_POSE = physical_pose_decoder()
        first_record_id = int(ALL_SELECTED_RECORDS[0]["record_id"])
        HOOK_IDENTITY = hook_identity_test(first_record_id)
        FORWARD_BENCHMARK = forward_benchmark(int(EVALUATION_RECORDS[0]["record_id"]))
        extract_baselines(CONSTRUCTION_RECORDS, [SUBSPACE_BLOCK])
        extract_baselines(CALIBRATION_RECORDS, [SUBSPACE_BLOCK])
        extract_baselines(EVALUATION_RECORDS, ACTIVE_BLOCKS)
        MODEL_BASELINES_OPENED = True
        write_json(
            OUT / "baseline_open_certificate.json",
            {
                "opened": True,
                "blocks_by_split": {
                    "construction": [SUBSPACE_BLOCK],
                    "calibration": [SUBSPACE_BLOCK],
                    "evaluation": ACTIVE_BLOCKS,
                },
                "records_by_split": {
                    split: len(records) for split, records in SELECTED_RECORDS_BY_SPLIT.items()
                },
                "evaluation_truth_used_for_model_baselines": False,
            },
        )
        memory_report("stage21_baselines_complete")
    except Exception:
        record_failure("stage21_model_baselines")
'''


correction_fit_and_choice_freeze = r'''# Fit goal-independent corrections, then freeze evaluation predictions and choices before endpoint truth opens.
EVALUATION_ENDPOINT_TRUTH_OPENED = False
EVALUATION_CHOICES_FROZEN = False


def true_pose_for_record(record):
    if record["split"] == "evaluation" and not EVALUATION_ENDPOINT_TRUTH_OPENED:
        raise RuntimeError(
            "evaluation endpoint truth is sealed until all evaluation choices are frozen"
        )
    with np.load(branch_path(record["record_id"])) as payload:
        endpoint_states = payload["endpoint_states"].astype(np.float64)
    return pose_target(endpoint_states)


def whiten_block4(record):
    payload = load_baseline(record["record_id"])
    carrier = carrier_for_block(payload, SUBSPACE_BLOCK)
    return transform_primal_channels(
        np.asarray(carrier, dtype=np.float64),
        FROZEN_SUBSPACES["channel_inverse_square_root"],
    )


def correction_condition_names():
    return [
        "learned_r128",
        "shuffled_r128",
        *[
            f"random_r128_{draw:02d}"
            for draw in range(ACTIVE_CORRECTION_RANDOM_DRAWS)
        ],
    ]


def basis_for_condition(condition):
    if condition == "learned_r128":
        return FROZEN_SUBSPACES["primary_basis"][:, :CORRECTION_RANK]
    if condition == "shuffled_r128":
        return FROZEN_SUBSPACES["shuffled_basis"][:, :CORRECTION_RANK]
    if condition.startswith("random_r128_"):
        draw = int(condition.rsplit("_", 1)[1])
        return FROZEN_SUBSPACES[f"random_basis_{draw:02d}"][:, :CORRECTION_RANK]
    raise KeyError(condition)


def correction_arrays(records, condition, include_truth):
    basis = basis_for_condition(condition)
    features = []
    targets = []
    groups = []
    for record in records:
        coordinates = subspace_coordinates(whiten_block4(record), basis)
        features.append(coordinates)
        groups.extend([int(record["trajectory_id"])] * ACTIONS_PER_STATE)
        if include_truth:
            payload = load_baseline(record["record_id"])
            decoded = payload["decoded_pose"].astype(np.float64)
            targets.append(centered_pose_residual(decoded, true_pose_for_record(record)))
    result = {
        "features": np.concatenate(features, axis=0),
        "groups": np.asarray(groups, dtype=np.int64),
    }
    if include_truth:
        result["targets"] = np.concatenate(targets, axis=0)
    return result


def save_correction_model(condition, result):
    model = result["model"]
    destination = SUBSPACE_DIR / f"correction_{condition}.npz"
    atomic_npz(
        destination,
        feature_mean=np.asarray(model["feature_mean"], dtype=np.float64),
        feature_scale=np.asarray(model["feature_scale"], dtype=np.float64),
        intercept=np.asarray(model["intercept"], dtype=np.float64),
        coefficient=np.asarray(model["coefficient"], dtype=np.float64),
        ridge=np.asarray(model["ridge"], dtype=np.float64),
    )
    return {
        "path": str(destination),
        "sha256": sha256_file(destination),
        "selected_ridge": float(result["selected_ridge"]),
        "calibration_rows": result["calibration_rows"],
    }


def correction_prediction(record, condition, coordinate_record=None):
    source_record = record if coordinate_record is None else coordinate_record
    basis = basis_for_condition(
        "learned_r128" if condition == "wrong_state_learned_r128" else condition
    )
    coordinates = subspace_coordinates(whiten_block4(source_record), basis)
    model_condition = (
        "learned_r128" if condition == "wrong_state_learned_r128" else condition
    )
    residual = predict_ridge(CORRECTION_MODELS[model_condition], coordinates)
    baseline = load_baseline(record["record_id"])
    decoded = baseline["decoded_pose"].astype(np.float64)
    corrected = apply_pose_correction(decoded, residual)
    scores = decoded_task_cost(corrected, np.asarray(record["goal"], dtype=np.float64))
    return residual, corrected, scores


def freeze_evaluation_predictions_and_choices():
    record_by_id = {
        int(record["record_id"]): record for record in EVALUATION_RECORDS
    }
    choices = {}
    choice_rows = []
    targets = {}
    target_rows = []
    for record in EVALUATION_RECORDS:
        record_id = int(record["record_id"])
        payload = load_baseline(record_id)
        decoded = payload["decoded_pose"].astype(np.float64)
        baseline_scores = decoded_task_cost(decoded, np.asarray(record["goal"]))
        condition_map = {
            "baseline": {
                "selected_action": int(np.argmin(baseline_scores)),
                "score_sha256": array_sha256(baseline_scores),
                "coordinate_record_id": record_id,
            }
        }
        for condition in correction_condition_names():
            _, _, scores = correction_prediction(record, condition)
            condition_map[condition] = {
                "selected_action": int(np.argmin(scores)),
                "score_sha256": array_sha256(scores),
                "coordinate_record_id": record_id,
            }
        wrong_id = int(WRONG_EVALUATION_RECORD[str(record_id)])
        _, _, scores = correction_prediction(
            record,
            "wrong_state_learned_r128",
            coordinate_record=record_by_id[wrong_id],
        )
        condition_map["wrong_state_learned_r128"] = {
            "selected_action": int(np.argmin(scores)),
            "score_sha256": array_sha256(scores),
            "coordinate_record_id": wrong_id,
        }
        choices[str(record_id)] = condition_map
        for condition, entry in condition_map.items():
            choice_rows.append(
                {
                    "record_id": record_id,
                    "trajectory_id": int(record["trajectory_id"]),
                    "action_family": record["action_family"],
                    "condition": condition,
                    **entry,
                }
            )

        donor, selected_targets = select_near_frontier_targets(
            baseline_scores, ACTIVE_TARGET_BASELINE_RANKS
        )
        entries = []
        for slot, (rank, target) in enumerate(
            zip(ACTIVE_TARGET_BASELINE_RANKS, selected_targets)
        ):
            permutation = targeted_derangement(
                ACTIONS_PER_STATE,
                target,
                donor,
                stable_seed(PERMUTATION_SEED, record_id, slot, "stage21_interface"),
            )
            if int(np.argmin(baseline_scores[permutation])) != int(target):
                raise RuntimeError("interface oracle does not select the frozen target")
            entry = {
                "target_slot": int(slot),
                "target_baseline_rank": int(rank),
                "target_action": int(target),
                "donor_action": int(donor),
                "permutation": permutation.tolist(),
            }
            entries.append(entry)
            target_rows.append(
                {
                    "record_id": record_id,
                    "trajectory_id": int(record["trajectory_id"]),
                    "action_family": record["action_family"],
                    **{key: value for key, value in entry.items() if key != "permutation"},
                    "permutation": " ".join(str(value) for value in permutation),
                }
            )
        targets[str(record_id)] = entries
    choice_schema = {
        "record_id",
        "trajectory_id",
        "action_family",
        "condition",
        "selected_action",
        "score_sha256",
        "coordinate_record_id",
    }
    if any(set(row) != choice_schema for row in choice_rows):
        raise RuntimeError("evaluation choice rows do not share one frozen schema")
    write_csv(DESIGN_DIR / "evaluation_choice_rows.csv", choice_rows)
    write_csv(DESIGN_DIR / "interface_target_rows.csv", target_rows)
    freeze = {
        "created_before_evaluation_endpoint_truth_opened": True,
        "evaluation_endpoint_truth_values_used": False,
        "evaluation_goals_used_only_to_score_model_predictions": True,
        "correction_is_goal_independent": True,
        "choices": choices,
        "interface_targets": targets,
        "choice_rows_sha256": sha256_file(DESIGN_DIR / "evaluation_choice_rows.csv"),
        "target_rows_sha256": sha256_file(DESIGN_DIR / "interface_target_rows.csv"),
    }
    write_json(DESIGN_DIR / "evaluation_choice_freeze.json", freeze)
    return choices, targets, freeze


if not PIPELINE_FAILED and MODEL_BASELINES_OPENED:
    try:
        CORRECTION_MODELS = {}
        CORRECTION_MANIFEST = {}
        CALIBRATION_ROWS = []
        for condition in correction_condition_names():
            construction = correction_arrays(
                CONSTRUCTION_RECORDS, condition, include_truth=True
            )
            calibration = correction_arrays(
                CALIBRATION_RECORDS, condition, include_truth=True
            )
            result = select_ridge_on_calibration(
                construction["features"],
                construction["targets"],
                calibration["features"],
                calibration["targets"],
                RIDGE_GRID,
            )
            CORRECTION_MODELS[condition] = result["model"]
            CORRECTION_MANIFEST[condition] = save_correction_model(condition, result)
            CALIBRATION_ROWS.extend(
                {"condition": condition, **row}
                for row in result["calibration_rows"]
            )
        write_csv(EVIDENCE_DIR / "correction_calibration_rows.csv", CALIBRATION_ROWS)
        CORRECTION_FIT_FREEZE = {
            "target": "candidate-centered true pose minus frozen decoded pose",
            "goal_used_in_fit_or_ridge_selection": False,
            "construction_records": len(CONSTRUCTION_RECORDS),
            "calibration_records": len(CALIBRATION_RECORDS),
            "evaluation_records_used": 0,
            "rank": CORRECTION_RANK,
            "ridge_grid": RIDGE_GRID,
            "conditions": CORRECTION_MANIFEST,
            "calibration_rows_sha256": sha256_file(
                EVIDENCE_DIR / "correction_calibration_rows.csv"
            ),
        }
        write_json(DESIGN_DIR / "correction_fit_freeze.json", CORRECTION_FIT_FREEZE)
        (
            EVALUATION_CHOICE_MAP,
            INTERFACE_TARGETS,
            EVALUATION_CHOICE_FREEZE,
        ) = freeze_evaluation_predictions_and_choices()
        EVALUATION_CHOICES_FROZEN = True
        memory_report("corrections_and_evaluation_choices_frozen")
    except Exception:
        record_failure("correction_fit_and_choice_freeze")
'''


interface_and_utility = r'''# Run the outcome-sealed coherent-interface test, then open truth for held-out utility scoring.


def interface_path(record_id):
    return INTERVENTION_DIR / f"evaluation_{int(record_id):06d}.json"


def interface_result_row(
    record,
    target_entry,
    condition,
    intervention_block,
    baseline_output,
    patched_output,
    baseline_pose,
    patched_pose,
):
    permutation = np.asarray(target_entry["permutation"], dtype=np.int64)
    target = int(target_entry["target_action"])
    output = donor_transfer_metrics(baseline_output, patched_output, permutation)
    pose = donor_transfer_metrics(baseline_pose, patched_pose, permutation)
    goal = np.asarray(record["goal"], dtype=np.float64)
    score = counterfactual_interface_metrics(
        decoded_task_cost(baseline_pose, goal),
        decoded_task_cost(patched_pose, goal),
        permutation,
        target,
    )
    return {
        "record_id": int(record["record_id"]),
        "trajectory_id": int(record["trajectory_id"]),
        "action_family": record["action_family"],
        "target_slot": int(target_entry["target_slot"]),
        "target_baseline_rank_frozen": int(target_entry["target_baseline_rank"]),
        "condition": condition,
        "intervention_block": int(intervention_block),
        "output_coefficient": output["coefficient"],
        "output_cosine": output["cosine"],
        "pose_coefficient": pose["coefficient"],
        "pose_cosine": pose["cosine"],
        **score,
    }


def run_interface_record(record):
    destination = interface_path(record["record_id"])
    if destination.exists():
        PROVENANCE_COUNTS["cache_hits"] += 1
        raise RuntimeError(f"fresh interface shard already exists: {destination}")
    payload = load_baseline(record["record_id"])
    baseline_output = payload["output_eval_sketch"].astype(np.float64)
    baseline_pose = payload["decoded_pose"].astype(np.float64)
    carrier4 = carrier_for_block(payload, SUBSPACE_BLOCK).astype(np.float64)
    carrier5 = carrier_for_block(payload, LAST_ACTION_CONDITIONED_BLOCK).astype(np.float64)
    white4 = transform_primal_channels(
        carrier4, FROZEN_SUBSPACES["channel_inverse_square_root"]
    )
    learned_basis = FROZEN_SUBSPACES["primary_basis"][:, :CORRECTION_RANK]
    rows = []
    specifications = []
    for target_entry in INTERFACE_TARGETS[str(int(record["record_id"]))]:
        permutation = np.asarray(target_entry["permutation"], dtype=np.int64)
        rows.append(
            interface_result_row(
                record,
                target_entry,
                "no_edit",
                -1,
                baseline_output,
                baseline_output,
                baseline_pose,
                baseline_pose,
            )
        )
        rows.append(
            interface_result_row(
                record,
                target_entry,
                "final_output_permutation_oracle",
                6,
                baseline_output,
                baseline_output[permutation],
                baseline_pose,
                baseline_pose[permutation],
            )
        )
        learned_white = action_swap_delta(
            white4, permutation, learned_basis, dose=1.0
        )
        learned_native = inverse_transform_primal_channels(
            learned_white, FROZEN_SUBSPACES["channel_square_root"]
        )
        specifications.extend(
            [
                {
                    "target_entry": target_entry,
                    "condition": "learned_r128_after_block4",
                    "block": SUBSPACE_BLOCK,
                    "delta": learned_native,
                },
                {
                    "target_entry": target_entry,
                    "condition": "full_swap_after_block4",
                    "block": SUBSPACE_BLOCK,
                    "delta": action_swap_delta(carrier4, permutation, basis=None),
                },
                {
                    "target_entry": target_entry,
                    "condition": "full_swap_after_block5",
                    "block": LAST_ACTION_CONDITIONED_BLOCK,
                    "delta": action_swap_delta(carrier5, permutation, basis=None),
                },
            ]
        )
    if len(specifications) != ACTIVE_INTERFACE_PATCHED_FORWARDS_PER_RECORD:
        raise RuntimeError(
            f"expected {ACTIVE_INTERFACE_PATCHED_FORWARDS_PER_RECORD} interface forwards, "
            f"found {len(specifications)}"
        )
    initial, actions = state_model_inputs(record["record_id"])
    for specification in specifications:
        delta = torch.as_tensor(
            specification["delta"], device="cuda", dtype=torch.float32
        )
        with torch.inference_mode():
            patched, _, _ = forward_with_carriers(
                initial,
                actions,
                PRIMARY_HORIZON,
                capture_blocks=[int(specification["block"])],
                intervention={"block": int(specification["block"]), "delta": delta},
            )
            patched_output = EVAL_OUTPUT_PROJECTOR(patched).cpu().numpy()
            patched_pose = DECODE_PHYSICAL_POSE(patched).cpu().numpy()
        rows.append(
            interface_result_row(
                record,
                specification["target_entry"],
                specification["condition"],
                specification["block"],
                baseline_output,
                patched_output,
                baseline_pose,
                patched_pose,
            )
        )
        del patched, patched_output, patched_pose, delta
    if len(rows) != ACTIVE_INTERFACE_ROWS_PER_RECORD:
        raise RuntimeError(
            f"expected {ACTIVE_INTERFACE_ROWS_PER_RECORD} interface rows, found {len(rows)}"
        )
    write_json(destination, rows)
    PROVENANCE_COUNTS["interface_shards_generated"] += 1
    PROVENANCE_COUNTS["patched_forwards_generated"] += len(specifications)
    del initial, actions
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def run_all_interfaces():
    started = time.perf_counter()
    rows = []
    for index, record in enumerate(EVALUATION_RECORDS):
        rows.extend(run_interface_record(record))
        write_json(
            OUT / "interface_progress.json",
            {
                "completed": index + 1,
                "total": len(EVALUATION_RECORDS),
                "patched_forwards_generated": PROVENANCE_COUNTS["patched_forwards_generated"],
            },
        )
    TIMINGS["coherent_interface_seconds"] = time.perf_counter() - started
    write_csv(EVIDENCE_DIR / "coherent_interface_rows.csv", rows)
    return rows


def utility_row(record, condition, coordinate_record=None):
    residual, corrected, scores = correction_prediction(
        record, condition, coordinate_record=coordinate_record
    )
    frozen = EVALUATION_CHOICE_MAP[str(int(record["record_id"]))][condition]
    if array_sha256(scores) != frozen["score_sha256"]:
        raise RuntimeError("evaluation score changed after choice freeze")
    if int(np.argmin(scores)) != int(frozen["selected_action"]):
        raise RuntimeError("evaluation choice changed after choice freeze")
    payload = load_baseline(record["record_id"])
    decoded = payload["decoded_pose"].astype(np.float64)
    truth = true_pose_for_record(record)
    result = corrected_planning_metrics(
        decoded, residual, truth, np.asarray(record["goal"], dtype=np.float64)
    )
    true_residual = centered_pose_residual(decoded, truth)
    return {
        "record_id": int(record["record_id"]),
        "trajectory_id": int(record["trajectory_id"]),
        "action_family": record["action_family"],
        "condition": condition,
        "coordinate_record_id": int(
            record["record_id"] if coordinate_record is None else coordinate_record["record_id"]
        ),
        "baseline_selected_action": int(result["baseline"]["selected_action"]),
        "corrected_selected_action": int(result["corrected"]["selected_action"]),
        "oracle_action": int(result["corrected"]["oracle_action"]),
        "baseline_normalized_regret": result["baseline"]["normalized_regret"],
        "corrected_normalized_regret": result["corrected"]["normalized_regret"],
        "normalized_regret_improvement": float(
            result["baseline"]["normalized_regret"]
            - result["corrected"]["normalized_regret"]
        ),
        "baseline_weighted_pairwise_accuracy": result["baseline"]["weighted_pairwise_accuracy"],
        "corrected_weighted_pairwise_accuracy": result["corrected"]["weighted_pairwise_accuracy"],
        "weighted_pairwise_accuracy_gain": float(
            result["corrected"]["weighted_pairwise_accuracy"]
            - result["baseline"]["weighted_pairwise_accuracy"]
        ),
        "baseline_top1_correct": result["baseline"]["top1_correct"],
        "corrected_top1_correct": result["corrected"]["top1_correct"],
        "top1_accuracy_gain": float(
            result["corrected"]["top1_correct"] - result["baseline"]["top1_correct"]
        ),
        "baseline_selected_true_cost": result["baseline_selected_true_cost"],
        "corrected_selected_true_cost": result["corrected_selected_true_cost"],
        "selected_true_cost_improvement": result["selected_true_cost_improvement"],
        "pose_residual_mse": float(np.mean((residual - true_residual) ** 2)),
    }


def run_heldout_utility():
    record_by_id = {
        int(record["record_id"]): record for record in EVALUATION_RECORDS
    }
    rows = []
    for record in EVALUATION_RECORDS:
        for condition in correction_condition_names():
            rows.append(utility_row(record, condition))
        wrong_id = int(WRONG_EVALUATION_RECORD[str(int(record["record_id"]))])
        rows.append(
            utility_row(
                record,
                "wrong_state_learned_r128",
                coordinate_record=record_by_id[wrong_id],
            )
        )
    expected = len(EVALUATION_RECORDS) * ACTIVE_UTILITY_ROWS_PER_RECORD
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} utility rows, found {len(rows)}")
    write_csv(EVIDENCE_DIR / "heldout_utility_rows.csv", rows)
    return rows


INTERFACE_COMPLETE = False
UTILITY_COMPLETE = False
if not PIPELINE_FAILED and EVALUATION_CHOICES_FROZEN:
    try:
        INTERFACE_ROWS = run_all_interfaces()
        INTERFACE_COMPLETE = True
        # This is the only transition that authorizes evaluation endpoint use.
        EVALUATION_ENDPOINT_TRUTH_OPENED = True
        started = time.perf_counter()
        UTILITY_ROWS = run_heldout_utility()
        TIMINGS["heldout_utility_seconds"] = time.perf_counter() - started
        UTILITY_COMPLETE = True
        write_json(
            OUT / "evaluation_truth_open_certificate.json",
            {
                "opened_after_choices_frozen": True,
                "choice_freeze_sha256": sha256_file(
                    DESIGN_DIR / "evaluation_choice_freeze.json"
                ),
                "interface_completed_before_truth_open": True,
                "evaluation_records": len(EVALUATION_RECORDS),
            },
        )
        memory_report("stage21_interface_and_utility_complete")
    except Exception:
        record_failure("stage21_interface_and_utility")
'''


decision = r'''# Apply Stage 21 coherent-handoff and held-out utility gates.


def rows_for(rows, family, condition):
    return [
        row for row in rows
        if row["action_family"] == family and row["condition"] == condition
    ]


def mean_key(rows, key):
    return float(np.mean([float(row[key]) for row in rows]))


def utility_map(family, condition, key):
    return {
        int(row["record_id"]): float(row[key])
        for row in rows_for(UTILITY_ROWS, family, condition)
    }


def bootstrap_interval(values, trajectories, family, label):
    draws = clustered_bootstrap_mean(
        np.asarray(values, dtype=np.float64),
        np.asarray(trajectories, dtype=np.int64),
        ACTIVE_BOOTSTRAP_DRAWS,
        stable_seed(BOOTSTRAP_SEED, family, label),
    )
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def evaluate_interface_family(family):
    oracle = rows_for(INTERFACE_ROWS, family, "final_output_permutation_oracle")
    block5 = rows_for(INTERFACE_ROWS, family, "full_swap_after_block5")
    block4 = rows_for(INTERFACE_ROWS, family, "full_swap_after_block4")
    learned = rows_for(INTERFACE_ROWS, family, "learned_r128_after_block4")
    required = [oracle, block5, block4, learned]
    finite = bool(
        all(
            len(values) > 0
            and all(
                np.isfinite(float(row[key]))
                for row in values
                for key in [
                    "output_coefficient",
                    "score_transfer_coefficient",
                    "score_counterfactual_normalized_rmse",
                ]
            )
            for values in required
        )
    )
    oracle_choice = mean_key(oracle, "choice_matches_counterfactual")
    block5_output = mean_key(block5, "output_coefficient")
    block5_score = mean_key(block5, "score_transfer_coefficient")
    block5_choice = mean_key(block5, "choice_matches_counterfactual")
    block5_nrmse = mean_key(block5, "score_counterfactual_normalized_rmse")
    block4_choice = mean_key(block4, "choice_matches_counterfactual")
    learned_output = mean_key(learned, "output_coefficient")
    passed = bool(
        finite
        and oracle_choice >= MIN_FINAL_ORACLE_CHOICE_RATE
        and block5_output >= MIN_LAST_BLOCK_OUTPUT_COEFFICIENT
        and block5_score >= MIN_LAST_BLOCK_SCORE_COEFFICIENT
        and block5_choice >= MIN_LAST_BLOCK_TARGET_CHOICE_RATE
        and block5_nrmse <= MAX_LAST_BLOCK_SCORE_NRMSE
        and learned_output >= MIN_LEARNED_BLOCK4_OUTPUT_COEFFICIENT
    )
    return {
        "action_family": family,
        "attempts": len(oracle),
        "all_required_metrics_finite": finite,
        "final_output_oracle_choice_rate": oracle_choice,
        "block5_full_output_coefficient": block5_output,
        "block5_full_score_coefficient": block5_score,
        "block5_full_choice_rate": block5_choice,
        "block5_full_score_nrmse": block5_nrmse,
        "block4_full_choice_rate": block4_choice,
        "block5_choice_gain_over_block4": block5_choice - block4_choice,
        "downstream_hybrid_localization_supported": bool(
            block5_choice - block4_choice >= MIN_LAST_VS_PENULTIMATE_CHOICE_GAIN
        ),
        "learned_block4_output_coefficient": learned_output,
        "coherent_handoff_gate_pass": passed,
    }


def evaluate_utility_family(family):
    learned_rows = rows_for(UTILITY_ROWS, family, "learned_r128")
    record_ids = [int(row["record_id"]) for row in learned_rows]
    trajectories = [int(row["trajectory_id"]) for row in learned_rows]
    learned_regret = np.asarray(
        [float(row["normalized_regret_improvement"]) for row in learned_rows]
    )
    learned_pairwise = np.asarray(
        [float(row["weighted_pairwise_accuracy_gain"]) for row in learned_rows]
    )
    learned_top1 = np.asarray(
        [float(row["top1_accuracy_gain"]) for row in learned_rows]
    )
    learned_true_cost = np.asarray(
        [float(row["selected_true_cost_improvement"]) for row in learned_rows]
    )
    learned_pose_mse = np.asarray(
        [float(row["pose_residual_mse"]) for row in learned_rows]
    )
    shuffled_regret_map = utility_map(
        family, "shuffled_r128", "normalized_regret_improvement"
    )
    wrong_regret_map = utility_map(
        family, "wrong_state_learned_r128", "normalized_regret_improvement"
    )
    random_regret_maps = [
        utility_map(
            family,
            f"random_r128_{draw:02d}",
            "normalized_regret_improvement",
        )
        for draw in range(ACTIVE_CORRECTION_RANDOM_DRAWS)
    ]
    random_pair_maps = [
        utility_map(
            family,
            f"random_r128_{draw:02d}",
            "weighted_pairwise_accuracy_gain",
        )
        for draw in range(ACTIVE_CORRECTION_RANDOM_DRAWS)
    ]
    random_regret = np.asarray([
        np.median([mapping[record_id] for mapping in random_regret_maps])
        for record_id in record_ids
    ])
    random_pairwise = np.asarray([
        np.median([mapping[record_id] for mapping in random_pair_maps])
        for record_id in record_ids
    ])
    shuffled_regret = np.asarray([shuffled_regret_map[value] for value in record_ids])
    wrong_regret = np.asarray([wrong_regret_map[value] for value in record_ids])
    gain_random = learned_regret - random_regret
    gain_shuffled = learned_regret - shuffled_regret
    gain_wrong = learned_regret - wrong_regret
    pair_gain_random = learned_pairwise - random_pairwise
    finite_arrays = [
        learned_regret, learned_pairwise, learned_top1, learned_true_cost,
        learned_pose_mse, random_regret, shuffled_regret, wrong_regret,
        gain_random, gain_shuffled, gain_wrong, pair_gain_random,
    ]
    finite = bool(all(np.all(np.isfinite(value)) for value in finite_arrays))
    regret_ci = bootstrap_interval(
        learned_regret, trajectories, family, "regret_vs_baseline"
    )
    random_ci = bootstrap_interval(
        gain_random, trajectories, family, "regret_gain_random"
    )
    pair_ci = bootstrap_interval(
        learned_pairwise, trajectories, family, "pairwise_vs_baseline"
    )
    utility_pass = bool(
        finite
        and np.mean(learned_regret) >= MIN_REGRET_IMPROVEMENT_VS_BASELINE
        and np.mean(gain_random) >= MIN_REGRET_GAIN_OVER_RANDOM
        and np.mean(gain_shuffled) > MIN_REGRET_GAIN_OVER_SHUFFLED
        and np.mean(gain_wrong) >= MIN_REGRET_GAIN_OVER_WRONG_STATE
        and np.mean(learned_pairwise) >= MIN_PAIRWISE_ACCURACY_GAIN
        and np.mean(pair_gain_random) >= MIN_PAIRWISE_GAIN_OVER_RANDOM
        and np.mean(learned_top1) >= -MAX_TOP1_ACCURACY_HARM
        and (regret_ci[0] > 0 if RUN_MODE == "pilot" else True)
        and (random_ci[0] > 0 if RUN_MODE == "pilot" else True)
        and (pair_ci[0] > 0 if RUN_MODE == "pilot" else True)
        and (
            exact_positive_sign_test(gain_random)["p_value"] <= 0.05
            if RUN_MODE == "pilot" else True
        )
    )
    return {
        "action_family": family,
        "evaluation_trajectories": len(record_ids),
        "all_required_metrics_finite": finite,
        "mean_learned_normalized_regret_improvement": float(np.mean(learned_regret)),
        "learned_regret_improvement_ci95": regret_ci,
        "mean_random_normalized_regret_improvement": float(np.mean(random_regret)),
        "mean_shuffled_normalized_regret_improvement": float(np.mean(shuffled_regret)),
        "mean_wrong_state_normalized_regret_improvement": float(np.mean(wrong_regret)),
        "mean_regret_gain_over_random": float(np.mean(gain_random)),
        "regret_gain_over_random_ci95": random_ci,
        "regret_gain_over_random_sign_test": exact_positive_sign_test(gain_random),
        "mean_regret_gain_over_shuffled": float(np.mean(gain_shuffled)),
        "mean_regret_gain_over_wrong_state": float(np.mean(gain_wrong)),
        "mean_learned_weighted_pairwise_accuracy_gain": float(np.mean(learned_pairwise)),
        "pairwise_accuracy_gain_ci95": pair_ci,
        "mean_pairwise_gain_over_random": float(np.mean(pair_gain_random)),
        "mean_top1_accuracy_gain": float(np.mean(learned_top1)),
        "mean_selected_true_cost_improvement": float(np.mean(learned_true_cost)),
        "mean_pose_residual_mse": float(np.mean(learned_pose_mse)),
        "heldout_utility_gate_pass": utility_pass,
    }


def fresh_run_certificate():
    expected = {
        "truth_generated": len(POOL_SPECS),
        "baseline_generated": len(ALL_SELECTED_RECORDS),
        "interface_shards_generated": len(EVALUATION_RECORDS),
        "patched_forwards_generated": len(EVALUATION_RECORDS)
        * ACTIVE_INTERFACE_PATCHED_FORWARDS_PER_RECORD,
        "cache_hits": 0,
    }
    passed = bool(not OUT_PREEXISTED and PROVENANCE_COUNTS == expected)
    payload = {
        "out_preexisted": bool(OUT_PREEXISTED),
        "observed_counts": dict(PROVENANCE_COUNTS),
        "expected_counts": expected,
        "passed": passed,
    }
    write_json(OUT / "fresh_run_certificate.json", payload)
    return payload


if PIPELINE_FAILED or not (INTERFACE_COMPLETE and UTILITY_COMPLETE):
    DECISION_PAYLOAD = {"status": "INCONCLUSIVE", "failure": FAILURE_MESSAGE}
else:
    try:
        # Verify the complete experimental execution, not only the pre-model prefix.
        verify_executed_notebook_through(
            "# Apply Stage 21 coherent-handoff and held-out utility gates."
        )
        INTERFACE_RESULTS = {
            family: evaluate_interface_family(family) for family in TRANSFER_FAMILIES
        }
        UTILITY_RESULTS = {
            family: evaluate_utility_family(family) for family in TRANSFER_FAMILIES
        }
        FRESH_CERTIFICATE = fresh_run_certificate()
        handoff_families = [
            family for family in TRANSFER_FAMILIES
            if INTERFACE_RESULTS[family]["coherent_handoff_gate_pass"]
        ]
        utility_families = [
            family for family in TRANSFER_FAMILIES
            if UTILITY_RESULTS[family]["heldout_utility_gate_pass"]
        ]
        localization_families = [
            family for family in TRANSFER_FAMILIES
            if INTERFACE_RESULTS[family]["downstream_hybrid_localization_supported"]
        ]
        if RUN_MODE == "smoke":
            candidate_status = "SMOKE_ONLY"
        elif len(handoff_families) < len(TRANSFER_FAMILIES):
            candidate_status = "HANDOFF_NOT_COHERENT_DIAGNOSTIC_FAILED"
        elif len(utility_families) == len(TRANSFER_FAMILIES):
            candidate_status = "CONFIRMED_COHERENT_HANDOFF_AND_CAUSAL_SUBSPACE_UTILITY_BOTH_FAMILIES"
        elif utility_families:
            candidate_status = "COHERENT_HANDOFF_WITH_PARTIAL_UTILITY"
        else:
            candidate_status = "COHERENT_HANDOFF_WITHOUT_CAUSAL_SUBSPACE_UTILITY"
        source_eligible = bool(SOURCE_IDENTITY.get("confirmation_eligible", False))
        confirmation_eligible = bool(
            source_eligible and PRIOR_ARTIFACTS_VALIDATED and FRESH_CERTIFICATE["passed"]
        )
        status = (
            candidate_status
            if RUN_MODE == "smoke" or confirmation_eligible
            else "UNBOUND_NONFRESH_OR_WRONG_PRIOR_EXPLORATORY_RESULT"
        )
        DECISION_PAYLOAD = {
            "status": status,
            "candidate_status": candidate_status,
            "source_bound_claim_eligible": source_eligible,
            "prior_artifacts_claim_eligible": bool(PRIOR_ARTIFACTS_VALIDATED),
            "fresh_run_claim_eligible": FRESH_CERTIFICATE["passed"],
            "confirmation_eligible": confirmation_eligible,
            "coherent_handoff_passed_families": handoff_families,
            "downstream_hybrid_localization_passed_families": localization_families,
            "heldout_utility_passed_families": utility_families,
            "interface_results": INTERFACE_RESULTS,
            "utility_results": UTILITY_RESULTS,
            "claim_boundary": {
                "evaluation_choices_frozen_before_endpoint_truth": True,
                "correction_fit_is_goal_independent": True,
                "causal_subspace_was_not_refit": True,
                "physical_decoder_is_external_and_frozen": True,
                "native_policy_or_planner_tested": False,
                "multi_step_closed_loop_control_tested": False,
                "visual_evaluation_used": False,
                "other_environment_checkpoint_or_architecture_generalized": False,
            },
        }
        write_json(OUT / "stage21_decision.json", DECISION_PAYLOAD)
        print(json.dumps(DECISION_PAYLOAD, indent=2))
    except Exception:
        record_failure("stage21_decision")
        DECISION_PAYLOAD = {"status": "INCONCLUSIVE", "failure": FAILURE_MESSAGE}

if not (OUT / "stage21_decision.json").exists():
    write_json(OUT / "stage21_decision.json", DECISION_PAYLOAD)
'''


packaging = base_source(12)
packaging = packaging.replace(
    "stage20_causal_planner_steering_result_bundle_",
    "stage21_coherent_utility_result_bundle_",
)
packaging = packaging.replace(
    '''raw_files += [
    path for path in sorted(SUBSPACE_DIR.rglob("*.npz")) if path.is_file()
]
''',
    "",
)
packaging = packaging.replace(
    '''    if SUBSPACE_DIR in path.parents and path.suffix == ".npz":
        continue
''',
    "",
)


protocol_sources = [
    introduction,
    configuration,
    installation,
    setup,
    analysis_helpers,
    model_helpers,
    design,
    truth_generation,
    artifact_import,
    model_baselines,
    correction_fit_and_choice_freeze,
    interface_and_utility,
    decision,
    packaging,
]
protocol_sources = [value.strip() for value in protocol_sources]
protocol_digest = hashlib.sha256(
    json.dumps(protocol_sources, ensure_ascii=False, separators=(",", ":")).encode()
).hexdigest()
configuration = configuration.replace("__PROTOCOL_DIGEST__", protocol_digest)
if "__PROTOCOL_DIGEST__" in configuration:
    raise RuntimeError("protocol digest placeholder was not replaced")

cells = [
    markdown(introduction),
    code(configuration),
    code(installation),
    code(setup),
    code(analysis_helpers),
    code(model_helpers),
    code(design),
    code(truth_generation),
    code(artifact_import),
    code(model_baselines),
    code(correction_fit_and_choice_freeze),
    code(interface_and_utility),
    code(decision),
    code(packaging),
]
for index, cell in enumerate(cells):
    cell["id"] = f"stage21-{index:02d}"

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "L4", "name": TARGET.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
TARGET.write_text(json.dumps(notebook, indent=1) + "\n")
print(f"Wrote {TARGET}")
