import ast
import hashlib
import json
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "19_unseen_action_family_transfer.ipynb"
BASE = json.loads((ROOT / "18_rank64_action_contrast_confirmation.ipynb").read_text())
NUMERICAL = ROOT.parent / "src/cf_faithfulness/stage19_unseen_action_transfer.py"


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


def checked_replace(source, old, new):
    if old not in source:
        raise RuntimeError(f"Stage 18 template changed; missing {old[:100]!r}")
    return source.replace(old, new)


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


def without_definitions(source, names):
    """Remove unused Stage 18 fitting helpers from the Stage 19 runtime."""

    tree = ast.parse(source)
    nodes = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    missing = sorted(set(names).difference(nodes))
    if missing:
        raise RuntimeError(f"missing removable definitions: {missing}")
    removed = set()
    for name in names:
        node = nodes[name]
        removed.update(range(node.lineno - 1, node.end_lineno))
    lines = source.splitlines(keepends=True)
    return "".join(line for index, line in enumerate(lines) if index not in removed)


introduction = r'''# Stage 19: frozen-subspace transfer to unseen action families

Stage 19 tests the most important boundary left by Stage 18: did the causal
block-4 subspace capture a reusable action-consequence mechanism, or only the
specific 12-direction, magnitude-0.12, constant-control candidate bank on
which it was constructed?

This notebook **does not fit, rotate, tune, or reselect a subspace**.  It
requires the exact successful Stage 18 artifact, identified by SHA-256
`2f9c496d...ca3f90b`, and fails closed on any mismatch.  Predictor block 4,
rank 64, the rank-128 sensitivity analysis, whitening metric, shuffled fit,
and four empirical-span random controls all come from that artifact.

Five action families were fixed before new simulator or model data:

1. directions rotated 15 degrees between every Stage 18 radial direction;
2. constant magnitude 0.08;
3. constant magnitude 0.16;
4. a five-step delay followed by ten steps at magnitude 0.18; and
5. two five-step pulses at magnitude 0.18 separated by five zero steps.

The two temporal families have the same integrated impulse as fifteen Stage
18 steps at magnitude 0.12.  Every family contains a no-op plus twelve
antithetic radial actions.  Each receives an independent, model-blind physical
eligibility screen on 64 fresh state/goal specifications; the first 24
eligible states are evaluated in pilot mode.

For every family, sufficiency transplants a fixed donor permutation inside
the frozen Stage 18 projector, and necessity removes the naturally occurring
projected action contrast.  Rank-matched shuffled and random projectors,
wrong-state, common-mode, zero-edit, full-swap, dose, and rank-128 sensitivity
checks remain frozen.  Planning changes are secondary and cannot rescue a
failed representation-transfer gate.

This is forward-pass-only: no coordinate readers, Jacobians, JVPs, VJPs,
gradients, model training, layer selection, or test-family subspace fitting.
Return `stage19_unseen_action_transfer_result_bundle_<signature>.zip`.
'''


configuration = r'''# SINGLE CONFIGURATION BLOCK
# Create these Colab secrets for a source-bound pilot:
# STAGE19_RUN_MODE=pilot
# STAGE19_SOURCE_COMMIT=<full 40-hex commit shown in the run guide>
# STAGE19_RUN_NONCE=<a new unique label, for example transfer_20260804_a>
# STAGE19_STAGE18_SUBSPACE_PATH is optional; the successful-run Drive path is the default.
RUN_MODE = "smoke"
EXPERIMENT_SOURCE_REF = ""
RUN_NONCE = "smoke"
STAGE18_SUBSPACE_PATH = (
    "/content/drive/MyDrive/counterfactual_faithfulness_stage18_rank64/"
    "pilot_f1b34beffcac/subspaces/frozen_rank64_confirmation_subspaces.npz"
)
try:
    from google.colab import userdata as _colab_userdata

    RUN_MODE = str(_colab_userdata.get("STAGE19_RUN_MODE") or RUN_MODE).strip().lower()
    EXPERIMENT_SOURCE_REF = str(
        _colab_userdata.get("STAGE19_SOURCE_COMMIT") or EXPERIMENT_SOURCE_REF
    ).strip()
    RUN_NONCE = str(_colab_userdata.get("STAGE19_RUN_NONCE") or RUN_NONCE).strip()
    STAGE18_SUBSPACE_PATH = str(
        _colab_userdata.get("STAGE19_STAGE18_SUBSPACE_PATH") or STAGE18_SUBSPACE_PATH
    ).strip()
except Exception:
    pass

if RUN_MODE == "pilot":
    if RUN_NONCE in {"", "smoke"}:
        raise ValueError("pilot mode requires a unique STAGE19_RUN_NONCE")
    if not all(value.isalnum() or value in "-_" for value in RUN_NONCE):
        raise ValueError("STAGE19_RUN_NONCE may contain only letters, numbers, '-' and '_'")

MOUNT_DRIVE = True
DOWNLOAD_RESULTS = True
CONTINUE_AFTER_BENCHMARK = True
MAX_ESTIMATED_TOTAL_MINUTES = 90.0
FRESH_RUN_REQUIRED = True

OUTPUT_DIR = "/content/counterfactual_faithfulness_stage19_transfer"
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/counterfactual_faithfulness_stage19_transfer"

PROTOCOL_ID = "stage19-frozen-subspace-unseen-action-transfer-v1"
NOTEBOOK_PROTOCOL_SHA256 = "__PROTOCOL_DIGEST__"
EVIDENCE_STATUS = "CONFIRMATORY_ONLY_IF_SOURCE_BOUND_FRESH_AND_STAGE18_HASH_BOUND"
EXPERIMENT_REPOSITORY = "grewalsk/counterfactual-faithfulness-research"
EXPERIMENT_NOTEBOOK_PATH = "notebooks/19_unseen_action_family_transfer.ipynb"
EXPERIMENT_BUILDER_PATH = "notebooks/build_stage19_unseen_action_transfer_notebook.py"
EXPERIMENT_NUMERICAL_PATH = "src/cf_faithfulness/stage19_unseen_action_transfer.py"

SEED = 19101
DESIGN_SEED = 19137
MODEL_NAME = "jepa_wm_pusht"
ENVIRONMENT = "PushT"
FRAMESKIP = 5
PRIMARY_HORIZON = 3
TARGET_STEPS = [PRIMARY_HORIZON]
FIXED_BLOCK = 4
ACTIVE_BLOCKS = [FIXED_BLOCK]
EXPECTED_CARRIER_CHANNELS = 400

EXPECTED_STAGE18_SUBSPACE_SHA256 = "2f9c496d54623a9062e465a18c70039acc18cb8a1cc2833a5f4ade162ca3f90b"
EXPECTED_STAGE18_SOURCE_COMMIT = "16edd247cddcb1aa121340eb5fa42bd9e07004c3"
EXPECTED_STAGE18_STATUS = "CONFIRMED_BIDIRECTIONAL_RANK64_MEDIATOR"
EXPECTED_STAGE18_AMBIENT_DIMENSION = 102400
EXPECTED_STAGE18_MAX_RANK = 128

TRANSFER_FAMILIES = [
    "rotated_direction",
    "magnitude_0p08",
    "magnitude_0p16",
    "delayed_equal_impulse",
    "pulsed_equal_impulse",
]
EVALUATION_POOL_TRAJECTORIES = list(range(500, 564))
EVALUATION_TRAJECTORY_TARGET_PER_FAMILY = 24
STATES_PER_TRAJECTORY = 1
TASK_ID_OFFSET = 1900
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
PRIMARY_RANK = 64
SENSITIVITY_RANKS = [64, 128]
MAX_SUBSPACE_RANK = 128
PERMUTATION_SEED = 19251
BOOTSTRAP_SEED = 19269
CAUSAL_RANDOM_DRAWS = 4
CAUSAL_DOSES = [-0.5, 0.25, 0.5, 1.0]
BOOTSTRAP_DRAWS = 10000
INTERVENTION_FORWARDS_PER_RECORD = 30

MIN_FULL_SWAP_COEFFICIENT = 0.75
MIN_PRIMARY_COEFFICIENT = 0.12
MIN_PRIMARY_COSINE = 0.15
MIN_PRIMARY_GAIN_OVER_RANDOM = 0.04
MIN_PRIMARY_GAIN_OVER_SHUFFLED = 0.04
MAX_PRIMARY_MEAN_SHIFT_RATIO = 0.25
REQUIRED_POSITIVE_EVALUATION_TRAJECTORIES = 18
MIN_NECESSITY_REDUCTION = 0.025
MIN_NECESSITY_GAIN_OVER_RANDOM = 0.015
MIN_NECESSITY_GAIN_OVER_SHUFFLED = 0.015
REQUIRED_POSITIVE_NECESSITY_TRAJECTORIES = 18
MAX_ZERO_EDIT_ERROR = 1e-6

if RUN_MODE == "smoke":
    ACTIVE_EVALUATION_POOL_TRAJECTORIES = EVALUATION_POOL_TRAJECTORIES[:6]
    ACTIVE_EVALUATION_TARGET_PER_FAMILY = 2
    ACTIVE_PRIMARY_RANK = PRIMARY_RANK
    ACTIVE_SENSITIVITY_RANKS = [PRIMARY_RANK]
    ACTIVE_CAUSAL_RANDOM_DRAWS = 1
    ACTIVE_CAUSAL_DOSES = [1.0]
    ACTIVE_BOOTSTRAP_DRAWS = 64
elif RUN_MODE == "pilot":
    ACTIVE_EVALUATION_POOL_TRAJECTORIES = EVALUATION_POOL_TRAJECTORIES
    ACTIVE_EVALUATION_TARGET_PER_FAMILY = EVALUATION_TRAJECTORY_TARGET_PER_FAMILY
    ACTIVE_PRIMARY_RANK = PRIMARY_RANK
    ACTIVE_SENSITIVITY_RANKS = SENSITIVITY_RANKS
    ACTIVE_CAUSAL_RANDOM_DRAWS = CAUSAL_RANDOM_DRAWS
    ACTIVE_CAUSAL_DOSES = CAUSAL_DOSES
    ACTIVE_BOOTSTRAP_DRAWS = BOOTSTRAP_DRAWS
else:
    raise ValueError(
        "STAGE19_RUN_MODE must contain only smoke or pilot; "
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

assert ACTIONS_PER_STATE == 13
assert ACTION_STEPS == 15
assert FIXED_BLOCK == 4
assert PRIMARY_RANK == 64
assert SENSITIVITY_RANKS == [64, 128]
assert MAX_SUBSPACE_RANK == EXPECTED_STAGE18_MAX_RANK
assert len(TRANSFER_FAMILIES) == 5 and len(set(TRANSFER_FAMILIES)) == 5
'''
configuration += "\n\nPROTOCOL_CONFIG_KEYS = " + repr(assigned_uppercase_names(configuration)) + "\n"


installation = base_source(2)


setup = base_source(3)
setup = setup.replace("Stage 18", "Stage 19").replace("STAGE18", "STAGE19")
setup = setup.replace('log = logging.getLogger("stage19_rank64")', 'log = logging.getLogger("stage19_transfer")')
setup = setup.replace("stage18_rank64_result_bundle_", "stage19_unseen_action_transfer_result_bundle_")
setup = setup.replace("stage18_rank64", "stage19_transfer")


analysis_helpers = without_definitions(
    base_source(4),
    [
        "channel_metric_from_moments",
        "earliest_within_one_se",
        "stable_cosine",
        "manifest_rows",
        "temporal_action_basis",
        "linear_cka",
        "grouped_kernel_ridge_cv",
        "fit_dual_ridge_basis",
        "random_subspace_in_span",
        "nested_orthonormalize_basis",
        "lower_triangle_principal_overlap",
    ],
)
analysis_helpers += "\n\n\n" + function_sources(
    NUMERICAL.read_text(),
    ["rotate_vector", "unseen_action_bank", "validate_stage18_subspace_arrays"],
)


model_helpers = base_source(5).replace("stage18-jepa-wms", "stage19-jepa-wms")


design = r'''# Freeze unseen action families and fresh state/goal pool before simulator or model data.


def trajectory_specs():
    specs = []
    center = np.asarray([256.0, 256.0])
    total = len(EVALUATION_POOL_TRAJECTORIES)
    for design_index, trajectory_id in enumerate(EVALUATION_POOL_TRAJECTORIES):
        phase = 0.41 + 2.0 * np.pi * design_index / total
        block = center + 46.0 * np.asarray([np.cos(phase), np.sin(phase)])
        block_angle = ((1.9 * phase + np.pi) % (2.0 * np.pi)) - np.pi
        approach = phase + [np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4][design_index % 4]
        approach += 0.13 * np.cos(3 * design_index)
        agent = block + APPROACH_DISTANCE * np.asarray([np.cos(approach), np.sin(approach)])
        goal_index = (13 * design_index + 7) % total
        goal_phase = 0.83 + 2.0 * np.pi * goal_index / total
        goal_xy = center + 74.0 * np.asarray([np.cos(goal_phase), np.sin(goal_phase)])
        common = {
            "design_index": int(design_index),
            "trajectory_id": int(trajectory_id),
            "time_index": 0,
            "physical_step": 0,
            "split": "evaluation",
            "evaluation_seed": int(DESIGN_SEED + 1009 * design_index),
            "goal": np.asarray(
                [goal_xy[0], goal_xy[1], ((1.1 * goal_phase + np.pi) % (2.0 * np.pi)) - np.pi],
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
                    "record_id": int(500000 + 1000 * family_index + trajectory_id),
                    "task_id": int(TASK_ID_OFFSET + design_index),
                    "action_family": family,
                    "family_index": int(family_index),
                }
            )
    return specs


ALL_POOL_SPECS = trajectory_specs()
POOL_SPECS = [
    row for row in ALL_POOL_SPECS
    if row["trajectory_id"] in ACTIVE_EVALUATION_POOL_TRAJECTORIES
]


def candidate_action_bank(record):
    state = np.asarray(record["state"], dtype=np.float64)
    if state.shape != (10,):
        raise ValueError("candidate state must be a ten-dimensional dynamic PushT state")
    return unseen_action_bank(state[2:4] - state[:2], record["action_family"], ACTION_STEPS)


np.savez_compressed(
    DESIGN_DIR / "stage19_unseen_action_pool_design.npz",
    record_ids=np.asarray([row["record_id"] for row in ALL_POOL_SPECS]),
    trajectory_ids=np.asarray([row["trajectory_id"] for row in ALL_POOL_SPECS]),
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
    "active_pool_trajectory_ids": ACTIVE_EVALUATION_POOL_TRAJECTORIES,
    "action_families": TRANSFER_FAMILIES,
    "target_per_family": ACTIVE_EVALUATION_TARGET_PER_FAMILY,
    "selection_uses_model_outputs": False,
    "eligibility": {
        "min_cost_spread": MIN_ELIGIBLE_COST_SPREAD,
        "min_non_tied_pair_fraction": MIN_ELIGIBLE_NON_TIED_PAIR_FRACTION,
        "min_contact_branches": MIN_ELIGIBLE_CONTACT_BRANCHES,
        "tie": PHYSICAL_COST_TIE,
    },
    "stage18_action_bank_excluded": True,
    "stage19_families_fixed_before_data": True,
}
write_json(DESIGN_DIR / "candidate_pool_manifest.json", POOL_MANIFEST)
DESIGN_FREEZE = {
    "created_before_simulator_or_model_data": True,
    "protocol_id": PROTOCOL_ID,
    "run_signature": RUN_SIGNATURE,
    "source_identity": SOURCE_IDENTITY,
    "candidate_pool_sha256": sha256_file(DESIGN_DIR / "stage19_unseen_action_pool_design.npz"),
    "pool_manifest_sha256": sha256_file(DESIGN_DIR / "candidate_pool_manifest.json"),
    "expected_stage18_subspace_sha256": EXPECTED_STAGE18_SUBSPACE_SHA256,
    "fixed_block": FIXED_BLOCK,
    "fixed_primary_rank": PRIMARY_RANK,
    "fixed_sensitivity_ranks": SENSITIVITY_RANKS,
    "subspace_refit_allowed": False,
    "coordinate_reader_used": False,
    "jacobian_used": False,
    "model_loaded": bool("MODEL" in globals()),
}
if DESIGN_FREEZE["model_loaded"]:
    raise RuntimeError("model was loaded before Stage 19 design freeze")
write_json(DESIGN_DIR / "design_freeze.json", DESIGN_FREEZE)
'''


truth_generation = base_source(7)
truth_generation = truth_generation.replace(
    "candidate_action_bank(record[\"state\"])", "candidate_action_bank(record)"
)
truth_generation = checked_replace(
    truth_generation,
    '            split=np.asarray(record["split"]),',
    '            split=np.asarray(record["split"]),\n'
    '            action_family=np.asarray(record["action_family"]),',
)
truth_generation = checked_replace(
    truth_generation,
    '        "split": record["split"],\n        **metrics,',
    '        "split": record["split"],\n'
    '        "action_family": record["action_family"],\n'
    '        **metrics,',
)
freeze_start = truth_generation.index("def freeze_maps(")
freeze_stop = truth_generation.index("\n\ndef make_truth_montage", freeze_start)
truth_generation = (
    truth_generation[:freeze_start]
    + r'''def freeze_maps_by_family(records_by_family):
    permutations = {}
    wrong = {}
    for family, records in records_by_family.items():
        identifiers = sorted(int(record["record_id"]) for record in records)
        if len(identifiers) < 2:
            raise RuntimeError(f"{family} needs at least two records for wrong-state control")
        for index, record_id in enumerate(identifiers):
            permutations[str(record_id)] = fixed_derangement(
                ACTIONS_PER_STATE,
                stable_seed(PERMUTATION_SEED, record_id, family, "donor"),
            ).tolist()
            wrong[str(record_id)] = identifiers[(index + 1) % len(identifiers)]
    return permutations, wrong
'''
    + truth_generation[freeze_stop:]
)
truth_generation = truth_generation.replace(
    "stage18_truth_montage.png", "stage19_unseen_action_truth_montage.png"
)
truth_generation = truth_generation.replace(
    'axes[row_index, 0].set_ylabel(f"{record[\'split\']} {record[\'record_id\']}")',
    'axes[row_index, 0].set_ylabel(f"{record[\'action_family\']}\\n{record[\'trajectory_id\']}")',
)
truth_final = truth_generation.index("if not PIPELINE_FAILED:")
truth_generation = truth_generation[:truth_final] + r'''if not PIPELINE_FAILED:
    try:
        REPO = configure_repo()
        RESTORE_TEST = exact_dynamic_restore_test(POOL_SPECS[0])
        write_json(OUT / "restore_test.json", RESTORE_TEST)
        generate_truth(POOL_SPECS, "truth_unseen_action_pool")
        if "MODEL" in globals():
            raise RuntimeError("model was loaded before physical eligibility selection")
        FAMILY_RECORDS = {}
        FAMILY_ELIGIBILITY_ROWS = {}
        for family in TRANSFER_FAMILIES:
            family_pool = [row for row in POOL_SPECS if row["action_family"] == family]
            chosen, rows = select_records(
                family_pool, ACTIVE_EVALUATION_TARGET_PER_FAMILY
            )
            FAMILY_RECORDS[family] = chosen
            FAMILY_ELIGIBILITY_ROWS[family] = rows
        ALL_EVALUATION_RECORDS = [
            record for family in TRANSFER_FAMILIES for record in FAMILY_RECORDS[family]
        ]
        ACTIVE_EVALUATION_TRAJECTORIES_BY_FAMILY = {
            family: [int(row["trajectory_id"]) for row in FAMILY_RECORDS[family]]
            for family in TRANSFER_FAMILIES
        }
        donor_permutations, wrong_state_map = freeze_maps_by_family(FAMILY_RECORDS)
        all_eligibility = [
            row for family in TRANSFER_FAMILIES for row in FAMILY_ELIGIBILITY_ROWS[family]
        ]
        write_csv(EVIDENCE_DIR / "physical_eligibility_rows.csv", all_eligibility)
        SELECTION_CERTIFICATE = {
            "selection_completed_before_model_load": True,
            "selection_used_only_simulator_truth": True,
            "selected_trajectory_ids_by_family": ACTIVE_EVALUATION_TRAJECTORIES_BY_FAMILY,
            "eligible_pool_count_by_family": {
                family: int(sum(row["eligible"] for row in FAMILY_ELIGIBILITY_ROWS[family]))
                for family in TRANSFER_FAMILIES
            },
            "donor_permutations": donor_permutations,
            "wrong_state_map": wrong_state_map,
            "wrong_state_within_action_family": True,
            "eligibility_rows_sha256": sha256_file(EVIDENCE_DIR / "physical_eligibility_rows.csv"),
        }
        write_json(DESIGN_DIR / "physical_selection_freeze.json", SELECTION_CERTIFICATE)
        make_truth_montage(ALL_EVALUATION_RECORDS)
        memory_report("physical_truth_and_selection_complete")
    except Exception:
        record_failure("physical_truth_selection")
'''


artifact_import = r'''# Import and hash-validate the exact successful Stage 18 subspace before model activations.
STAGE18_ARTIFACT_VALIDATED = False
if not PIPELINE_FAILED:
    try:
        verify_executed_notebook_through(
            "# Import and hash-validate the exact successful Stage 18 subspace before model activations."
        )
        FROZEN_SUBSPACE_PATH = Path(STAGE18_SUBSPACE_PATH)
        if not FROZEN_SUBSPACE_PATH.is_file():
            raise FileNotFoundError(
                "The exact Stage 18 raw subspace is missing. Expected it at "
                f"{FROZEN_SUBSPACE_PATH}. Keep the successful Stage 18 Drive run, or set "
                "the STAGE19_STAGE18_SUBSPACE_PATH Colab secret to its .npz path."
            )
        observed_subspace_sha256 = sha256_file(FROZEN_SUBSPACE_PATH)
        if observed_subspace_sha256 != EXPECTED_STAGE18_SUBSPACE_SHA256:
            raise RuntimeError(
                "Stage 18 subspace hash mismatch: expected "
                f"{EXPECTED_STAGE18_SUBSPACE_SHA256}, found {observed_subspace_sha256}"
            )
        stage18_run_dir = FROZEN_SUBSPACE_PATH.parent.parent
        stage18_decision_path = stage18_run_dir / "stage18_decision.json"
        stage18_manifest_path = stage18_run_dir / "subspaces/subspace_manifest.json"
        stage18_source_path = stage18_run_dir / "source_identity.json"
        for required_path in [stage18_decision_path, stage18_manifest_path, stage18_source_path]:
            if not required_path.is_file():
                raise FileNotFoundError(f"Stage 18 provenance file is missing: {required_path}")
        stage18_decision = json.loads(stage18_decision_path.read_text())
        stage18_manifest = json.loads(stage18_manifest_path.read_text())
        stage18_source = json.loads(stage18_source_path.read_text())
        if stage18_decision.get("status") != EXPECTED_STAGE18_STATUS:
            raise RuntimeError(f"Stage 18 decision is not confirmatory: {stage18_decision.get('status')}")
        if not bool(stage18_decision.get("confirmation_eligible", False)):
            raise RuntimeError("Stage 18 decision was not source-bound and fresh")
        if stage18_manifest.get("subspace_sha256") != EXPECTED_STAGE18_SUBSPACE_SHA256:
            raise RuntimeError("Stage 18 manifest does not bind the required subspace")
        if stage18_source.get("resolved_commit") != EXPECTED_STAGE18_SOURCE_COMMIT:
            raise RuntimeError("Stage 18 source commit does not match the successful frozen run")
        if not bool(stage18_source.get("confirmation_eligible", False)):
            raise RuntimeError("Stage 18 source execution was not verified")
        with np.load(FROZEN_SUBSPACE_PATH) as payload:
            FROZEN_SUBSPACES = {name: payload[name].copy() for name in payload.files}
        artifact_contract = validate_stage18_subspace_arrays(
            FROZEN_SUBSPACES,
            ambient=EXPECTED_STAGE18_AMBIENT_DIMENSION,
            max_rank=EXPECTED_STAGE18_MAX_RANK,
        )
        STAGE18_ARTIFACT_CERTIFICATE = {
            "validated_before_stage19_model_activations": True,
            "path": str(FROZEN_SUBSPACE_PATH),
            "bytes": int(FROZEN_SUBSPACE_PATH.stat().st_size),
            "sha256": observed_subspace_sha256,
            "expected_stage18_source_commit": EXPECTED_STAGE18_SOURCE_COMMIT,
            "stage18_decision_status": stage18_decision["status"],
            "stage18_confirmation_eligible": stage18_decision["confirmation_eligible"],
            "stage18_decision_sha256": sha256_file(stage18_decision_path),
            "stage18_manifest_sha256": sha256_file(stage18_manifest_path),
            "stage18_source_identity_sha256": sha256_file(stage18_source_path),
            "artifact_contract": artifact_contract,
            "stage19_subspace_refit": False,
            "stage19_basis_rotation_or_tuning": False,
        }
        write_json(OUT / "stage18_artifact_certificate.json", STAGE18_ARTIFACT_CERTIFICATE)
        STAGE18_ARTIFACT_VALIDATED = True
        memory_report("stage18_artifact_validated")
    except Exception:
        record_failure("stage18_artifact_import")
'''


model_and_baselines = base_source(8)
model_and_baselines = model_and_baselines.replace(
    "# Load frozen JEPA-WM and cache construction forward passes at all blocks.",
    "# Load frozen JEPA-WM and generate fresh unseen-family baselines at fixed block 4.",
)
model_and_baselines = checked_replace(
    model_and_baselines,
    "    total_eval_records = int(ACTIVE_EVALUATION_TARGET)",
    "    total_eval_records = len(ALL_EVALUATION_RECORDS)",
)
model_and_baselines = checked_replace(
    model_and_baselines,
    '    interventions_per_record = INTERVENTION_FORWARDS_PER_RECORD if RUN_MODE == "pilot" else 12',
    '    interventions_per_record = INTERVENTION_FORWARDS_PER_RECORD if RUN_MODE == "pilot" else 9',
)
model_final = model_and_baselines.index("if not PIPELINE_FAILED:")
model_and_baselines = model_and_baselines[:model_final] + r'''EVALUATION_OPENED = False
if not PIPELINE_FAILED:
    try:
        if not STAGE18_ARTIFACT_VALIDATED:
            raise RuntimeError("Stage 18 artifact must be validated before model loading")
        MODEL, PREPROCESSOR, PREDICTOR, PREDICTOR_BLOCK_MODULES = load_frozen_model()
        if len(PREDICTOR_BLOCK_MODULES) != 6:
            raise RuntimeError("predictor block count changed")
        for module in PREDICTOR_BLOCK_MODULES:
            if not isinstance(module, torch.nn.Module) or getattr(module, "register_forward_hook", None) is None:
                raise RuntimeError("predictor block does not support forward hooks")
        TRAIN_OUTPUT_PROJECTOR = CountSketchProjector(
            256 * 384, OUTPUT_SKETCH_DIM, TRAIN_OUTPUT_SKETCH_SEED
        )
        EVAL_OUTPUT_PROJECTOR = CountSketchProjector(
            256 * 384, OUTPUT_SKETCH_DIM, EVAL_OUTPUT_SKETCH_SEED
        )
        DECODE_PHYSICAL_POSE = physical_pose_decoder()
        first_record_id = ALL_EVALUATION_RECORDS[0]["record_id"]
        HOOK_IDENTITY = hook_identity_test(first_record_id)
        FORWARD_BENCHMARK = forward_benchmark(first_record_id)
        extract_baselines(ALL_EVALUATION_RECORDS, [FIXED_BLOCK])
        EVALUATION_OPENED = True
        write_json(
            OUT / "evaluation_open_certificate.json",
            {
                "opened": True,
                "source_identity": SOURCE_IDENTITY,
                "stage18_artifact_certificate_sha256": sha256_file(
                    OUT / "stage18_artifact_certificate.json"
                ),
                "physical_selection_freeze_sha256": sha256_file(
                    DESIGN_DIR / "physical_selection_freeze.json"
                ),
                "records_by_family": {
                    family: len(FAMILY_RECORDS[family]) for family in TRANSFER_FAMILIES
                },
                "stage18_artifact_validated_before_model_activations": True,
                "stage19_fit_or_selection_model_activations": [],
            },
        )
        memory_report("unseen_family_baselines_complete")
    except Exception:
        record_failure("unseen_family_model_baselines")
'''


causal_interchange = base_source(12)
causal_interchange = causal_interchange.replace(
    "# Run rank-matched sufficiency and necessity interventions on held-out trajectories.",
    "# Run frozen Stage 18 sufficiency and necessity interventions on unseen action families.",
)
causal_interchange = checked_replace(
    causal_interchange,
    '''def load_frozen_subspaces():
    with np.load(SUBSPACE_DIR / "frozen_rank64_confirmation_subspaces.npz") as payload:
        return {name: payload[name].copy() for name in payload.files}
''',
    '''def load_frozen_subspaces():
    if not STAGE18_ARTIFACT_VALIDATED:
        raise RuntimeError("Stage 18 artifact is not validated")
    return FROZEN_SUBSPACES
''',
)
causal_interchange = checked_replace(
    causal_interchange,
    '        "task_id": int(record["task_id"]),\n        "selected_block": FIXED_BLOCK,',
    '        "task_id": int(record["task_id"]),\n'
    '        "action_family": record["action_family"],\n'
    '        "selected_block": FIXED_BLOCK,',
)
necessity_start = causal_interchange.index(
    '    add(\n        f"ablate_primary_r{ACTIVE_PRIMARY_RANK:03d}"'
)
necessity_stop = causal_interchange.index(
    "\n\n    if RUN_MODE == \"pilot\"", necessity_start
)
causal_interchange = (
    causal_interchange[:necessity_start]
    + r'''    for rank in ACTIVE_SENSITIVITY_RANKS:
        learned_basis = subspaces["primary_basis"][:, :rank]
        learned_ablation = projection_ablation_delta(white, learned_basis, dose=1.0)
        learned_name = (
            f"ablate_primary_r{rank:03d}"
            if rank == ACTIVE_PRIMARY_RANK else f"ablate_learned_r{rank:03d}"
        )
        add(
            learned_name, "primary" if rank == ACTIVE_PRIMARY_RANK else "rank_sensitivity",
            "necessity", rank, 1.0, learned_ablation,
        )
        shuffled_ablation = projection_ablation_delta(
            white, subspaces["shuffled_basis"][:, :rank], dose=1.0
        )
        add(
            f"ablate_shuffled_r{rank:03d}", "matched_shuffled_control", "necessity",
            rank, 1.0, norm_match(shuffled_ablation, learned_ablation),
        )
        for draw in range(ACTIVE_CAUSAL_RANDOM_DRAWS):
            random_ablation = projection_ablation_delta(
                white, subspaces[f"random_basis_{draw:02d}"][:, :rank], dose=1.0
            )
            add(
                f"ablate_random_r{rank:03d}_{draw:02d}",
                "empirical_span_random_control", "necessity", rank, 1.0,
                norm_match(random_ablation, learned_ablation),
            )
'''
    + causal_interchange[necessity_stop:]
)
causal_interchange = causal_interchange.replace(
    "INTERVENTION_ROWS = run_all_interventions(EVALUATION_RECORDS)",
    "INTERVENTION_ROWS = run_all_interventions(ALL_EVALUATION_RECORDS)",
)


decision_and_plots = r'''# Aggregate by trajectory within family and apply frozen transfer gates.


def trajectory_summaries(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["action_family"], row["trajectory_id"], row["condition"], float(row["dose"]))].append(row)
    summaries = []
    metric_names = [
        "output_coefficient", "output_cosine", "output_reconstruction",
        "output_mean_shift_ratio", "output_contrast_energy_retention",
        "output_contrast_energy_reduction", "output_contrast_cosine",
        "pose_coefficient", "pose_cosine", "normalized_regret",
        "weighted_pairwise_accuracy", "output_rms_change",
        "edit_to_full_swap_ratio",
    ]
    for (action_family, trajectory_id, condition, dose), values in sorted(grouped.items()):
        row = {
            "action_family": action_family,
            "trajectory_id": int(trajectory_id),
            "condition": condition,
            "dose": float(dose),
            "records": len(values),
            "mode": values[0]["mode"],
            "rank": int(values[0]["rank"]),
        }
        for metric in metric_names:
            data = np.asarray([value[metric] for value in values], dtype=np.float64)
            row[metric] = float(np.nanmean(data)) if np.any(np.isfinite(data)) else math.nan
        summaries.append(row)
    write_csv(EVIDENCE_DIR / "trajectory_condition_summary.csv", summaries)
    return summaries


def lookup(summary, family, trajectory_id, condition, dose=1.0, key="output_coefficient"):
    values = [
        row[key]
        for row in summary
        if row["action_family"] == family
        and row["trajectory_id"] == trajectory_id
        and row["condition"] == condition
        and np.isclose(row["dose"], dose)
    ]
    return float(values[0]) if len(values) == 1 else np.nan


def bootstrap_interval(values, trajectories, family, label):
    seed = stable_seed(BOOTSTRAP_SEED, family, label) % (2**31 - 1)
    draws = clustered_bootstrap_mean(
        np.asarray(values), np.asarray(trajectories), ACTIVE_BOOTSTRAP_DRAWS, seed
    )
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def random_median(summary, family, trajectory_id, rank, key="output_coefficient", ablate=False):
    prefix = "ablate_random" if ablate else "random"
    values = [
        lookup(summary, family, trajectory_id, f"{prefix}_r{rank:03d}_{draw:02d}", key=key)
        for draw in range(ACTIVE_CAUSAL_RANDOM_DRAWS)
    ]
    return float(np.nanmedian(values))


def family_rank_curve(summary, family, trajectories):
    rows = []
    for rank in ACTIVE_SENSITIVITY_RANKS:
        learned_name = f"primary_r{rank:03d}" if rank == ACTIVE_PRIMARY_RANK else f"learned_r{rank:03d}"
        ablate_name = (
            f"ablate_primary_r{rank:03d}"
            if rank == ACTIVE_PRIMARY_RANK else f"ablate_learned_r{rank:03d}"
        )
        learned = np.asarray([lookup(summary, family, value, learned_name) for value in trajectories])
        shuffled = np.asarray([
            lookup(summary, family, value, f"shuffled_r{rank:03d}") for value in trajectories
        ])
        random_values = np.asarray([
            random_median(summary, family, value, rank) for value in trajectories
        ])
        necessity = np.asarray([
            lookup(
                summary, family, value, ablate_name,
                key="output_contrast_energy_reduction",
            )
            for value in trajectories
        ])
        rows.append(
            {
                "action_family": family,
                "rank": int(rank),
                "mean_learned_coefficient": float(np.nanmean(learned)),
                "mean_random_coefficient": float(np.nanmean(random_values)),
                "mean_shuffled_coefficient": float(np.nanmean(shuffled)),
                "mean_gain_over_random": float(np.nanmean(learned - random_values)),
                "mean_gain_over_shuffled": float(np.nanmean(learned - shuffled)),
                "mean_learned_necessity_reduction": float(np.nanmean(necessity)),
            }
        )
    return rows


def evaluate_family_gate(summary, family):
    trajectories = ACTIVE_EVALUATION_TRAJECTORIES_BY_FAMILY[family]
    primary_name = f"primary_r{ACTIVE_PRIMARY_RANK:03d}"
    primary = np.asarray([lookup(summary, family, value, primary_name) for value in trajectories])
    primary_cosine = np.asarray([
        lookup(summary, family, value, primary_name, key="output_cosine") for value in trajectories
    ])
    primary_shift = np.asarray([
        lookup(summary, family, value, primary_name, key="output_mean_shift_ratio") for value in trajectories
    ])
    full = np.asarray([lookup(summary, family, value, "full_activation_swap") for value in trajectories])
    shuffled = np.asarray([
        lookup(summary, family, value, f"shuffled_r{ACTIVE_PRIMARY_RANK:03d}")
        for value in trajectories
    ])
    random_values = np.asarray([
        random_median(summary, family, value, ACTIVE_PRIMARY_RANK) for value in trajectories
    ])
    gain_random = primary - random_values
    gain_shuffled = primary - shuffled
    sufficiency_sign = exact_positive_sign_test(gain_random)

    positive_doses = sorted(value for value in ACTIVE_CAUSAL_DOSES if value > 0)
    dose_slopes = []
    if len(positive_doses) >= 2:
        for trajectory_id in trajectories:
            values = np.asarray([
                lookup(summary, family, trajectory_id, primary_name, dose=value)
                for value in positive_doses
            ])
            dose_slopes.append(float(np.polyfit(positive_doses, values, 1)[0]))
    else:
        dose_slopes = [math.nan] * len(trajectories)
    negative = (
        np.asarray([
            lookup(summary, family, value, primary_name, dose=-0.5) for value in trajectories
        ])
        if -0.5 in ACTIVE_CAUSAL_DOSES else np.full(len(trajectories), np.nan)
    )

    ablate_primary = f"ablate_primary_r{ACTIVE_PRIMARY_RANK:03d}"
    necessity = np.asarray([
        lookup(
            summary, family, value, ablate_primary,
            key="output_contrast_energy_reduction",
        )
        for value in trajectories
    ])
    necessity_shuffled = np.asarray([
        lookup(
            summary, family, value, f"ablate_shuffled_r{ACTIVE_PRIMARY_RANK:03d}",
            key="output_contrast_energy_reduction",
        )
        for value in trajectories
    ])
    necessity_random = np.asarray([
        random_median(
            summary, family, value, ACTIVE_PRIMARY_RANK,
            key="output_contrast_energy_reduction", ablate=True,
        )
        for value in trajectories
    ])
    necessity_gain_random = necessity - necessity_random
    necessity_gain_shuffled = necessity - necessity_shuffled
    necessity_sign = exact_positive_sign_test(necessity_gain_random)
    gain_random_ci = bootstrap_interval(gain_random, trajectories, family, "sufficiency")
    necessity_gain_ci = bootstrap_interval(
        necessity_gain_random, trajectories, family, "necessity"
    )

    finite = bool(all(np.all(np.isfinite(value)) for value in [
        primary, primary_cosine, primary_shift, full, shuffled, random_values,
        necessity, necessity_shuffled, necessity_random,
    ]))
    sufficiency_pass = bool(
        finite
        and np.mean(full) >= MIN_FULL_SWAP_COEFFICIENT
        and np.mean(primary) >= MIN_PRIMARY_COEFFICIENT
        and np.mean(primary_cosine) >= MIN_PRIMARY_COSINE
        and np.mean(primary_shift) <= MAX_PRIMARY_MEAN_SHIFT_RATIO
        and np.mean(gain_random) >= MIN_PRIMARY_GAIN_OVER_RANDOM
        and np.mean(gain_shuffled) >= MIN_PRIMARY_GAIN_OVER_SHUFFLED
        and sufficiency_sign["positive"] >= min(REQUIRED_POSITIVE_EVALUATION_TRAJECTORIES, len(trajectories))
        and (sufficiency_sign["p_value"] <= 0.05 if RUN_MODE == "pilot" else True)
        and (gain_random_ci[0] > 0 if RUN_MODE == "pilot" else True)
        and (
            RUN_MODE == "smoke"
            or (
                np.sum(np.asarray(dose_slopes) > 0) >= REQUIRED_POSITIVE_EVALUATION_TRAJECTORIES
                and np.mean(negative) < 0
            )
        )
    )
    necessity_pass = bool(
        finite
        and np.mean(necessity) >= MIN_NECESSITY_REDUCTION
        and np.mean(necessity_gain_random) >= MIN_NECESSITY_GAIN_OVER_RANDOM
        and np.mean(necessity_gain_shuffled) >= MIN_NECESSITY_GAIN_OVER_SHUFFLED
        and necessity_sign["positive"] >= min(REQUIRED_POSITIVE_NECESSITY_TRAJECTORIES, len(trajectories))
        and (necessity_sign["p_value"] <= 0.05 if RUN_MODE == "pilot" else True)
        and (necessity_gain_ci[0] > 0 if RUN_MODE == "pilot" else True)
    )
    return {
        "action_family": family,
        "trajectories": len(trajectories),
        "all_required_metrics_finite": finite,
        "mean_primary_coefficient": float(np.mean(primary)),
        "mean_primary_cosine": float(np.mean(primary_cosine)),
        "mean_primary_mean_shift_ratio": float(np.mean(primary_shift)),
        "mean_full_swap_coefficient": float(np.mean(full)),
        "mean_random_coefficient": float(np.mean(random_values)),
        "mean_shuffled_coefficient": float(np.mean(shuffled)),
        "mean_gain_over_random": float(np.mean(gain_random)),
        "mean_gain_over_shuffled": float(np.mean(gain_shuffled)),
        "gain_over_random_ci95": gain_random_ci,
        "gain_over_random_sign_test": sufficiency_sign,
        "positive_dose_slope_trajectories": int(np.sum(np.asarray(dose_slopes) > 0)),
        "negative_dose_mean": float(np.mean(negative)) if np.any(np.isfinite(negative)) else None,
        "mean_necessity_reduction": float(np.mean(necessity)),
        "mean_necessity_random_reduction": float(np.mean(necessity_random)),
        "mean_necessity_shuffled_reduction": float(np.mean(necessity_shuffled)),
        "mean_necessity_gain_over_random": float(np.mean(necessity_gain_random)),
        "mean_necessity_gain_over_shuffled": float(np.mean(necessity_gain_shuffled)),
        "necessity_gain_over_random_ci95": necessity_gain_ci,
        "necessity_gain_over_random_sign_test": necessity_sign,
        "sufficiency_gate_pass": sufficiency_pass,
        "necessity_gate_pass": necessity_pass,
        "bidirectional_transfer_gate_pass": bool(sufficiency_pass and necessity_pass),
    }


def planning_secondary(summary, family):
    trajectories = ACTIVE_EVALUATION_TRAJECTORIES_BY_FAMILY[family]
    primary_name = f"primary_r{ACTIVE_PRIMARY_RANK:03d}"
    rows = {}
    for metric in ["normalized_regret", "weighted_pairwise_accuracy"]:
        baseline = np.asarray([
            lookup(summary, family, value, "no_edit", dose=0.0, key=metric)
            for value in trajectories
        ])
        primary = np.asarray([
            lookup(summary, family, value, primary_name, key=metric) for value in trajectories
        ])
        shuffled = np.asarray([
            lookup(summary, family, value, f"shuffled_r{ACTIVE_PRIMARY_RANK:03d}", key=metric)
            for value in trajectories
        ])
        random_shift = []
        for trajectory_id, base in zip(trajectories, baseline):
            values = [
                lookup(
                    summary, family, trajectory_id,
                    f"random_r{ACTIVE_PRIMARY_RANK:03d}_{draw:02d}", key=metric,
                )
                for draw in range(ACTIVE_CAUSAL_RANDOM_DRAWS)
            ]
            random_shift.append(float(np.nanmedian(np.abs(np.asarray(values) - base))))
        primary_shift = np.abs(primary - baseline)
        shuffled_shift = np.abs(shuffled - baseline)
        random_shift = np.asarray(random_shift)
        gain_random = primary_shift - random_shift
        gain_shuffled = primary_shift - shuffled_shift
        rows[metric] = {
            "mean_baseline": float(np.nanmean(baseline)),
            "mean_primary": float(np.nanmean(primary)),
            "mean_absolute_primary_change": float(np.nanmean(primary_shift)),
            "mean_absolute_random_change": float(np.nanmean(random_shift)),
            "mean_absolute_shuffled_change": float(np.nanmean(shuffled_shift)),
            "mean_change_gain_over_random": float(np.nanmean(gain_random)),
            "mean_change_gain_over_shuffled": float(np.nanmean(gain_shuffled)),
            "change_gain_over_random_ci95": bootstrap_interval(
                gain_random, trajectories, family, f"planning_{metric}"
            ),
        }
    rows["secondary_planning_signal"] = bool(any(
        value["change_gain_over_random_ci95"][0] > 0
        and value["mean_change_gain_over_shuffled"] > 0
        for key, value in rows.items() if key != "secondary_planning_signal"
    ))
    rows["claim_role"] = "secondary_only_does_not_enter_representation_transfer_gate"
    return rows


def fresh_run_certificate():
    expected = {
        "truth_generated": len(POOL_SPECS),
        "baseline_generated": len(ALL_EVALUATION_RECORDS),
        "intervention_generated": len(ALL_EVALUATION_RECORDS),
        "cache_hits": 0,
    }
    passed = bool(not OUT_PREEXISTED and PROVENANCE_COUNTS == expected)
    payload = {
        "out_preexisted": bool(OUT_PREEXISTED),
        "fresh_run_required": bool(FRESH_RUN_REQUIRED),
        "observed_counts": dict(PROVENANCE_COUNTS),
        "expected_counts": expected,
        "passed": passed,
    }
    write_json(OUT / "fresh_run_certificate.json", payload)
    return payload


def make_plots(family_gates, rank_rows):
    labels = [value.replace("_equal_impulse", "").replace("magnitude_", "mag ").replace("rotated_direction", "rotated") for value in TRANSFER_FAMILIES]
    figure, axes = plt.subplots(1, 4, figsize=(22, 4.8))
    x = np.arange(len(labels))
    width = 0.25
    axes[0].bar(x - width, [family_gates[f]["mean_primary_coefficient"] for f in TRANSFER_FAMILIES], width, label="learned")
    axes[0].bar(x, [family_gates[f]["mean_random_coefficient"] for f in TRANSFER_FAMILIES], width, label="random")
    axes[0].bar(x + width, [family_gates[f]["mean_shuffled_coefficient"] for f in TRANSFER_FAMILIES], width, label="shuffled")
    axes[0].set(ylabel="donor coefficient", title="Rank-64 sufficiency", xticks=x, xticklabels=labels)
    axes[0].legend()
    axes[1].bar(x, [family_gates[f]["mean_gain_over_random"] for f in TRANSFER_FAMILIES])
    axes[1].axhline(MIN_PRIMARY_GAIN_OVER_RANDOM, color="black", linestyle="--")
    axes[1].set(ylabel="learned - random", title="Sufficiency control gain", xticks=x, xticklabels=labels)
    axes[2].bar(x - width / 2, [family_gates[f]["mean_necessity_reduction"] for f in TRANSFER_FAMILIES], width, label="learned")
    axes[2].bar(x + width / 2, [family_gates[f]["mean_necessity_random_reduction"] for f in TRANSFER_FAMILIES], width, label="random")
    axes[2].set(ylabel="contrast-energy reduction", title="Rank-64 necessity", xticks=x, xticklabels=labels)
    axes[2].legend()
    rank128 = {row["action_family"]: row for row in rank_rows if row["rank"] == 128}
    if rank128:
        axes[3].bar(x - width / 2, [family_gates[f]["mean_primary_coefficient"] for f in TRANSFER_FAMILIES], width, label="r64")
        axes[3].bar(x + width / 2, [rank128[f]["mean_learned_coefficient"] for f in TRANSFER_FAMILIES], width, label="r128")
        axes[3].set(ylabel="donor coefficient", title="Frozen rank sensitivity", xticks=x, xticklabels=labels)
        axes[3].legend()
    else:
        axes[3].text(0.5, 0.5, "rank-128 omitted in smoke mode", ha="center", va="center")
        axes[3].axis("off")
    for axis in list(axes[:3]) + ([axes[3]] if rank128 else []):
        axis.tick_params(axis="x", rotation=28)
    figure.tight_layout()
    figure.savefig(PLOT_DIR / "stage19_unseen_action_transfer_summary.png", dpi=180)
    plt.close(figure)


if PIPELINE_FAILED:
    DECISION_PAYLOAD = {"status": "INCONCLUSIVE", "failure": FAILURE_MESSAGE}
elif not EVALUATION_OPENED:
    DECISION_PAYLOAD = {"status": "INCONCLUSIVE", "reason": "unseen-family model activations were not opened"}
else:
    try:
        TRAJECTORY_SUMMARY = trajectory_summaries(INTERVENTION_ROWS)
        FAMILY_GATES = {
            family: evaluate_family_gate(TRAJECTORY_SUMMARY, family)
            for family in TRANSFER_FAMILIES
        }
        RANK_CURVES = [
            row
            for family in TRANSFER_FAMILIES
            for row in family_rank_curve(
                TRAJECTORY_SUMMARY, family,
                ACTIVE_EVALUATION_TRAJECTORIES_BY_FAMILY[family],
            )
        ]
        write_csv(ANALYSIS_DIR / "family_rank_scaling_summary.csv", RANK_CURVES)
        PLANNING_SECONDARY = {
            family: planning_secondary(TRAJECTORY_SUMMARY, family)
            for family in TRANSFER_FAMILIES
        }
        write_json(ANALYSIS_DIR / "planning_secondary.json", PLANNING_SECONDARY)
        FRESH_CERTIFICATE = fresh_run_certificate()
        passed_families = [
            family for family in TRANSFER_FAMILIES
            if FAMILY_GATES[family]["bidirectional_transfer_gate_pass"]
        ]
        if RUN_MODE == "smoke":
            candidate_status = "SMOKE_ONLY"
        elif len(passed_families) == len(TRANSFER_FAMILIES):
            candidate_status = "CONFIRMED_TRANSFER_ALL_UNSEEN_ACTION_FAMILIES"
        elif "rotated_direction" in passed_families and len(passed_families) >= 3:
            candidate_status = "PARTIAL_TRANSFER_DIRECTIONS_AND_SOME_FAMILIES"
        elif passed_families:
            candidate_status = "LIMITED_ACTION_FAMILY_TRANSFER"
        else:
            candidate_status = "NO_CONFIRMED_UNSEEN_ACTION_TRANSFER"
        source_eligible = bool(SOURCE_IDENTITY.get("confirmation_eligible", False))
        artifact_eligible = bool(STAGE18_ARTIFACT_VALIDATED)
        confirmation_eligible = bool(
            source_eligible and artifact_eligible and FRESH_CERTIFICATE["passed"]
        )
        status = (
            candidate_status
            if RUN_MODE == "smoke" or confirmation_eligible
            else "UNBOUND_NONFRESH_OR_WRONG_ARTIFACT_EXPLORATORY_RESULT"
        )
        DECISION_PAYLOAD = {
            "status": status,
            "candidate_status": candidate_status,
            "source_bound_claim_eligible": source_eligible,
            "stage18_artifact_claim_eligible": artifact_eligible,
            "fresh_run_claim_eligible": FRESH_CERTIFICATE["passed"],
            "confirmation_eligible": confirmation_eligible,
            "passed_action_families": passed_families,
            "family_gates": FAMILY_GATES,
            "rank_curves": RANK_CURVES,
            "planning_secondary": PLANNING_SECONDARY,
            "claim_boundary": {
                "all_families_must_pass_for_broad_transfer_claim": True,
                "rank64_is_intrinsic_dimension": False,
                "coordinate_chart_authorized": False,
                "jacobian_claim_authorized": False,
                "planning_mediation_is_secondary": True,
                "other_models_or_environments_generalized": False,
                "causal_claim_is_frozen_subspace_transfer_across_prespecified_action_families": True,
            },
        }
        write_json(OUT / "stage19_decision.json", DECISION_PAYLOAD)
        make_plots(FAMILY_GATES, RANK_CURVES)
        print(json.dumps(DECISION_PAYLOAD, indent=2))
    except Exception:
        record_failure("decision_and_plots")
        DECISION_PAYLOAD = {"status": "INCONCLUSIVE", "failure": FAILURE_MESSAGE}

if not (OUT / "stage19_decision.json").exists():
    write_json(OUT / "stage19_decision.json", DECISION_PAYLOAD)
'''


packaging = base_source(14)
packaging = packaging.replace(
    "stage18_rank64_result_bundle_", "stage19_unseen_action_transfer_result_bundle_"
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
    model_and_baselines,
    causal_interchange,
    decision_and_plots,
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
    code(model_and_baselines),
    code(causal_interchange),
    code(decision_and_plots),
    code(packaging),
]
for index, cell in enumerate(cells):
    cell["id"] = f"stage19-{index:02d}"

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
