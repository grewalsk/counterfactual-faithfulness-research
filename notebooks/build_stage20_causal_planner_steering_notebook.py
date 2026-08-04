import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "20_causal_planner_steering.ipynb"
BASE = json.loads((ROOT / "19_unseen_action_family_transfer.ipynb").read_text())
NUMERICAL = ROOT.parent / "src/cf_faithfulness/stage20_planner_steering.py"


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
        raise RuntimeError(f"Stage 19 template changed; missing {old[:100]!r}")
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


introduction = r'''# Stage 20: frozen-subspace causal planner steering

Stage 20 tests whether the action-consequence mechanism confirmed in Stages
18 and 19 forms a causal interface to downstream planning behavior.  It is a
fresh, non-visual evaluation: all decisions use numerical predictions,
rankings, selected actions, and simulator costs.  No example selection, video
scoring, or human visual judgment enters the protocol.

The exact Stage 18 block-4 bases and whitening transform remain frozen.  Stage
20 also requires the exact successful Stage 19 decision before proceeding.
There is no training, subspace refit, layer selection, coordinate reader,
Jacobian, JVP, VJP, or gradient computation.

For each fresh state, the untouched model produces a planner cost vector
(q(a)).  Its actions at baseline ranks 2, 3, and 4 are fixed as steering
targets, using no simulator outcome.  For target (t) and baseline-best donor
(b), a deterministic derangement \(\pi_t\) is frozen with
(\pi_t(t)=b).  The high-level counterfactual is therefore exact:

\[
q^{\mathrm{cf}}_t(a)=q(\pi_t(a)),
\qquad \arg\min_a q^{\mathrm{cf}}_t(a)=t.
\]

The primary rank-128 edit replaces the current projected action contrast with
the corresponding permuted donor contrast.  Rank 64 is a sensitivity.  Four
rank-matched random bases, the frozen shuffled-fit basis, wrong-state,
common-mode, complete-swap, and necessity-ablation conditions remain controls.

Two action families are evaluated separately: interleaved directions and the
pulsed equal-impulse profile.  Simulator truth is generated and physically
screened before model loading.  Baseline predictions may define targets, but
no intervention output or simulator cost may do so.  The primary behavioral
question is whether the learned edit moves target rank and chosen action
toward the exact counterfactual more than all matched controls.

Return `stage20_causal_planner_steering_result_bundle_<signature>.zip`.
'''


configuration = r'''# SINGLE CONFIGURATION BLOCK
# Create these Colab secrets for a source-bound pilot:
# STAGE20_RUN_MODE=pilot
# STAGE20_SOURCE_COMMIT=<full 40-hex commit shown in the handoff>
# STAGE20_RUN_NONCE=<a new unique label, for example steering_20260804_a>
# The Stage 18 subspace and Stage 19 decision paths have successful-run defaults.
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
try:
    from google.colab import userdata as _colab_userdata

    RUN_MODE = str(_colab_userdata.get("STAGE20_RUN_MODE") or RUN_MODE).strip().lower()
    EXPERIMENT_SOURCE_REF = str(
        _colab_userdata.get("STAGE20_SOURCE_COMMIT") or EXPERIMENT_SOURCE_REF
    ).strip()
    RUN_NONCE = str(_colab_userdata.get("STAGE20_RUN_NONCE") or RUN_NONCE).strip()
    STAGE18_SUBSPACE_PATH = str(
        _colab_userdata.get("STAGE20_STAGE18_SUBSPACE_PATH") or STAGE18_SUBSPACE_PATH
    ).strip()
    STAGE19_DECISION_PATH = str(
        _colab_userdata.get("STAGE20_STAGE19_DECISION_PATH") or STAGE19_DECISION_PATH
    ).strip()
except Exception:
    pass

if RUN_MODE == "pilot":
    if RUN_NONCE in {"", "smoke"}:
        raise ValueError("pilot mode requires a unique STAGE20_RUN_NONCE")
    if not all(value.isalnum() or value in "-_" for value in RUN_NONCE):
        raise ValueError("STAGE20_RUN_NONCE may contain only letters, numbers, '-' and '_'")

MOUNT_DRIVE = True
DOWNLOAD_RESULTS = True
CONTINUE_AFTER_BENCHMARK = True
MAX_ESTIMATED_TOTAL_MINUTES = 90.0
FRESH_RUN_REQUIRED = True

OUTPUT_DIR = "/content/counterfactual_faithfulness_stage20_steering"
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/counterfactual_faithfulness_stage20_steering"

PROTOCOL_ID = "stage20-frozen-subspace-causal-planner-steering-v1"
NOTEBOOK_PROTOCOL_SHA256 = "__PROTOCOL_DIGEST__"
EVIDENCE_STATUS = "CONFIRMATORY_ONLY_IF_SOURCE_BOUND_FRESH_AND_PRIOR_ARTIFACTS_BOUND"
EXPERIMENT_REPOSITORY = "grewalsk/counterfactual-faithfulness-research"
EXPERIMENT_NOTEBOOK_PATH = "notebooks/20_causal_planner_steering.ipynb"
EXPERIMENT_BUILDER_PATH = "notebooks/build_stage20_causal_planner_steering_notebook.py"
EXPERIMENT_NUMERICAL_PATH = "src/cf_faithfulness/stage20_planner_steering.py"

SEED = 20101
DESIGN_SEED = 20137
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
EXPECTED_STAGE19_DECISION_SHA256 = "493fdf5c707189caea11043db7d208dbc38677dcf5881008e13bede87f40be9c"
EXPECTED_STAGE19_SOURCE_IDENTITY_SHA256 = "6fad7d1ee14efa0898125faaa4500a2ab7b62d81591412364b4379c43ec9ffcf"
EXPECTED_STAGE19_SOURCE_COMMIT = "bf8c3950fc1112b38baa2453e39793592537ec47"
EXPECTED_STAGE19_STATUS = "CONFIRMED_TRANSFER_ALL_UNSEEN_ACTION_FAMILIES"

TRANSFER_FAMILIES = ["rotated_direction", "pulsed_equal_impulse"]
EVALUATION_POOL_TRAJECTORIES = list(range(600, 680))
EVALUATION_TRAJECTORY_TARGET_PER_FAMILY = 32
TARGET_BASELINE_RANKS = [1, 2, 3]
STATES_PER_TRAJECTORY = 1
TASK_ID_OFFSET = 2000
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
PRIMARY_STEERING_RANK = 128
SENSITIVITY_RANK = 64
PERMUTATION_SEED = 20251
BOOTSTRAP_SEED = 20269
CAUSAL_RANDOM_DRAWS = 4
STEERING_DOSES = [0.5, 1.0]
BOOTSTRAP_DRAWS = 10000
INTERVENTION_FORWARDS_PER_RECORD = 39
RESULT_ROWS_PER_RECORD = 54

MIN_FULL_SWAP_COEFFICIENT = 0.80
MIN_FULL_TARGET_CHOICE_RATE = 0.90
MIN_PRIMARY_OUTPUT_COEFFICIENT = 0.25
MIN_OUTPUT_GAIN_OVER_RANDOM = 0.10
MIN_TARGET_RANK_GAIN_OVER_RANDOM = 0.25
MIN_CHOICE_MATCH_GAIN_OVER_RANDOM = 0.05
MIN_NECESSITY_REDUCTION = 0.03
MIN_NECESSITY_GAIN_OVER_RANDOM = 0.02
MIN_NECESSITY_GAIN_OVER_SHUFFLED = 0.02
MAX_ZERO_EDIT_ERROR = 1e-6

if RUN_MODE == "smoke":
    ACTIVE_EVALUATION_POOL_TRAJECTORIES = EVALUATION_POOL_TRAJECTORIES[:8]
    ACTIVE_EVALUATION_TARGET_PER_FAMILY = 2
    ACTIVE_TARGET_BASELINE_RANKS = [1]
    ACTIVE_CAUSAL_RANDOM_DRAWS = 1
    ACTIVE_STEERING_DOSES = [1.0]
    ACTIVE_BOOTSTRAP_DRAWS = 64
    ACTIVE_INTERVENTION_FORWARDS_PER_RECORD = 10
    ACTIVE_RESULT_ROWS_PER_RECORD = 11
elif RUN_MODE == "pilot":
    ACTIVE_EVALUATION_POOL_TRAJECTORIES = EVALUATION_POOL_TRAJECTORIES
    ACTIVE_EVALUATION_TARGET_PER_FAMILY = EVALUATION_TRAJECTORY_TARGET_PER_FAMILY
    ACTIVE_TARGET_BASELINE_RANKS = TARGET_BASELINE_RANKS
    ACTIVE_CAUSAL_RANDOM_DRAWS = CAUSAL_RANDOM_DRAWS
    ACTIVE_STEERING_DOSES = STEERING_DOSES
    ACTIVE_BOOTSTRAP_DRAWS = BOOTSTRAP_DRAWS
    ACTIVE_INTERVENTION_FORWARDS_PER_RECORD = INTERVENTION_FORWARDS_PER_RECORD
    ACTIVE_RESULT_ROWS_PER_RECORD = RESULT_ROWS_PER_RECORD
else:
    raise ValueError(
        "STAGE20_RUN_MODE must contain only smoke or pilot; "
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
assert PRIMARY_STEERING_RANK == 128
assert SENSITIVITY_RANK == 64
assert len(TRANSFER_FAMILIES) == 2
assert TARGET_BASELINE_RANKS == [1, 2, 3]
'''
configuration += "\n\nPROTOCOL_CONFIG_KEYS = " + repr(assigned_uppercase_names(configuration)) + "\n"


installation = base_source(2)


setup = base_source(3)
setup = setup.replace("Stage 19", "Stage 20").replace("STAGE19", "STAGE20")
setup = setup.replace('log = logging.getLogger("stage20_transfer")', 'log = logging.getLogger("stage20_steering")')
setup = setup.replace("stage19_transfer", "stage20_steering")
setup = checked_replace(
    setup,
    'PROVENANCE_COUNTS = {"truth_generated": 0, "baseline_generated": 0, "intervention_generated": 0, "cache_hits": 0}',
    'PROVENANCE_COUNTS = {"truth_generated": 0, "baseline_generated": 0, '
    '"intervention_generated": 0, "patched_forwards_generated": 0, "cache_hits": 0}',
)


analysis_helpers = base_source(4)
analysis_helpers += "\n\n\n" + function_sources(
    NUMERICAL.read_text(),
    [
        "targeted_derangement",
        "stable_action_rank",
        "select_near_frontier_targets",
        "planner_steering_metrics",
    ],
)


model_helpers = base_source(5).replace("stage19-jepa-wms", "stage20-jepa-wms")


design = r'''# Freeze two action families and fresh physical state pool before simulator or model data.


def trajectory_specs():
    specs = []
    center = np.asarray([256.0, 256.0])
    total = len(EVALUATION_POOL_TRAJECTORIES)
    for design_index, trajectory_id in enumerate(EVALUATION_POOL_TRAJECTORIES):
        phase = 0.29 + 2.0 * np.pi * design_index / total
        block = center + 44.0 * np.asarray([np.cos(phase), np.sin(phase)])
        block_angle = ((2.1 * phase + np.pi) % (2.0 * np.pi)) - np.pi
        approach = phase + [np.pi / 3, 2 * np.pi / 3, 4 * np.pi / 3, 5 * np.pi / 3][design_index % 4]
        approach += 0.11 * np.sin(2 * design_index)
        agent = block + APPROACH_DISTANCE * np.asarray([np.cos(approach), np.sin(approach)])
        goal_index = (17 * design_index + 5) % total
        goal_phase = 0.71 + 2.0 * np.pi * goal_index / total
        goal_xy = center + 73.0 * np.asarray([np.cos(goal_phase), np.sin(goal_phase)])
        common = {
            "design_index": int(design_index),
            "trajectory_id": int(trajectory_id),
            "time_index": 0,
            "physical_step": 0,
            "split": "evaluation",
            "evaluation_seed": int(DESIGN_SEED + 1013 * design_index),
            "goal": np.asarray(
                [goal_xy[0], goal_xy[1], ((1.2 * goal_phase + np.pi) % (2.0 * np.pi)) - np.pi],
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
                    "record_id": int(600000 + 1000 * family_index + trajectory_id),
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
    DESIGN_DIR / "stage20_steering_pool_design.npz",
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
    "target_baseline_ranks": ACTIVE_TARGET_BASELINE_RANKS,
    "physical_selection_uses_model_outputs": False,
    "steering_target_selection_uses_baseline_model_scores_only": True,
    "steering_target_selection_uses_simulator_costs": False,
    "eligibility": {
        "min_cost_spread": MIN_ELIGIBLE_COST_SPREAD,
        "min_non_tied_pair_fraction": MIN_ELIGIBLE_NON_TIED_PAIR_FRACTION,
        "min_contact_branches": MIN_ELIGIBLE_CONTACT_BRANCHES,
        "tie": PHYSICAL_COST_TIE,
    },
}
write_json(DESIGN_DIR / "candidate_pool_manifest.json", POOL_MANIFEST)
DESIGN_FREEZE = {
    "created_before_simulator_or_model_data": True,
    "protocol_id": PROTOCOL_ID,
    "run_signature": RUN_SIGNATURE,
    "source_identity": SOURCE_IDENTITY,
    "candidate_pool_sha256": sha256_file(DESIGN_DIR / "stage20_steering_pool_design.npz"),
    "pool_manifest_sha256": sha256_file(DESIGN_DIR / "candidate_pool_manifest.json"),
    "expected_stage18_subspace_sha256": EXPECTED_STAGE18_SUBSPACE_SHA256,
    "expected_stage19_decision_sha256": EXPECTED_STAGE19_DECISION_SHA256,
    "fixed_block": FIXED_BLOCK,
    "primary_steering_rank": PRIMARY_STEERING_RANK,
    "sensitivity_rank": SENSITIVITY_RANK,
    "subspace_refit_allowed": False,
    "visual_evaluation_used": False,
    "model_loaded": bool("MODEL" in globals()),
}
if DESIGN_FREEZE["model_loaded"]:
    raise RuntimeError("model was loaded before Stage 20 design freeze")
write_json(DESIGN_DIR / "design_freeze.json", DESIGN_FREEZE)
'''


truth_generation = without_definitions(base_source(7), ["make_truth_montage"])
truth_generation = truth_generation.replace(
    '        make_truth_montage(ALL_EVALUATION_RECORDS)\n', ''
)


artifact_import = r'''# Bind successful Stages 18/19 and load the exact frozen subspaces before model activations.
PRIOR_ARTIFACTS_VALIDATED = False
if not PIPELINE_FAILED:
    try:
        verify_executed_notebook_through(
            "# Bind successful Stages 18/19 and load the exact frozen subspaces before model activations."
        )
        frozen_subspace_path = Path(STAGE18_SUBSPACE_PATH)
        if not frozen_subspace_path.is_file():
            raise FileNotFoundError(f"Stage 18 raw subspace is missing: {frozen_subspace_path}")
        observed_subspace_sha256 = sha256_file(frozen_subspace_path)
        if observed_subspace_sha256 != EXPECTED_STAGE18_SUBSPACE_SHA256:
            raise RuntimeError(
                f"Stage 18 subspace hash mismatch: {observed_subspace_sha256}"
            )
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
            raise RuntimeError("Stage 18 decision is not the successful confirmation")
        if not bool(stage18_decision.get("confirmation_eligible", False)):
            raise RuntimeError("Stage 18 decision was not claim eligible")
        if stage18_manifest.get("subspace_sha256") != EXPECTED_STAGE18_SUBSPACE_SHA256:
            raise RuntimeError("Stage 18 manifest does not bind the required subspace")
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
        if not stage19_decision_path.is_file():
            raise FileNotFoundError(f"Stage 19 decision is missing: {stage19_decision_path}")
        if sha256_file(stage19_decision_path) != EXPECTED_STAGE19_DECISION_SHA256:
            raise RuntimeError("Stage 19 decision hash mismatch")
        stage19_source_path = stage19_decision_path.parent / "source_identity.json"
        if not stage19_source_path.is_file():
            raise FileNotFoundError(f"Stage 19 source identity is missing: {stage19_source_path}")
        if sha256_file(stage19_source_path) != EXPECTED_STAGE19_SOURCE_IDENTITY_SHA256:
            raise RuntimeError("Stage 19 source-identity hash mismatch")
        stage19_decision = json.loads(stage19_decision_path.read_text())
        stage19_source = json.loads(stage19_source_path.read_text())
        if stage19_decision.get("status") != EXPECTED_STAGE19_STATUS:
            raise RuntimeError("Stage 19 did not confirm all unseen action families")
        if not bool(stage19_decision.get("confirmation_eligible", False)):
            raise RuntimeError("Stage 19 decision was not claim eligible")
        if stage19_source.get("resolved_commit") != EXPECTED_STAGE19_SOURCE_COMMIT:
            raise RuntimeError("Stage 19 source commit mismatch")
        if not all(
            family in stage19_decision.get("passed_action_families", [])
            for family in TRANSFER_FAMILIES
        ):
            raise RuntimeError("Stage 19 did not pass both Stage 20 action families")

        PRIOR_ARTIFACT_CERTIFICATE = {
            "validated_before_stage20_model_activations": True,
            "stage18_subspace_path": str(frozen_subspace_path),
            "stage18_subspace_bytes": int(frozen_subspace_path.stat().st_size),
            "stage18_subspace_sha256": observed_subspace_sha256,
            "stage18_decision_status": stage18_decision["status"],
            "stage18_artifact_contract": artifact_contract,
            "stage19_decision_path": str(stage19_decision_path),
            "stage19_decision_sha256": sha256_file(stage19_decision_path),
            "stage19_source_identity_sha256": sha256_file(stage19_source_path),
            "stage19_decision_status": stage19_decision["status"],
            "stage19_passed_required_families": TRANSFER_FAMILIES,
            "stage20_subspace_refit": False,
            "stage20_basis_tuning": False,
        }
        write_json(OUT / "prior_artifact_certificate.json", PRIOR_ARTIFACT_CERTIFICATE)
        PRIOR_ARTIFACTS_VALIDATED = True
        memory_report("prior_artifacts_validated")
    except Exception:
        record_failure("prior_artifact_import")
'''


model_and_targets = base_source(9)
model_and_targets = model_and_targets.replace(
    "# Load frozen JEPA-WM and generate fresh unseen-family baselines at fixed block 4.",
    "# Load frozen JEPA-WM, generate baselines, and freeze near-frontier steering targets.",
)
model_and_targets = checked_replace(
    model_and_targets,
    '    interventions_per_record = INTERVENTION_FORWARDS_PER_RECORD if RUN_MODE == "pilot" else 9',
    '    interventions_per_record = ACTIVE_INTERVENTION_FORWARDS_PER_RECORD',
)
model_final = model_and_targets.index("EVALUATION_OPENED = False")
model_definitions = model_and_targets[:model_final]
model_and_targets = model_definitions + r'''

def freeze_steering_targets(records):
    target_map = {}
    rows = []
    for record in records:
        record_id = int(record["record_id"])
        payload = load_baseline(record_id)
        goal = np.asarray(record["goal"], dtype=np.float64)
        baseline_scores = decoded_task_cost(
            payload["decoded_pose"].astype(np.float64), goal
        )
        donor, targets = select_near_frontier_targets(
            baseline_scores, ACTIVE_TARGET_BASELINE_RANKS
        )
        entries = []
        for slot, (baseline_rank, target) in enumerate(
            zip(ACTIVE_TARGET_BASELINE_RANKS, targets)
        ):
            permutation = targeted_derangement(
                ACTIONS_PER_STATE,
                target,
                donor,
                stable_seed(PERMUTATION_SEED, record_id, slot, "planner_target"),
            )
            expected_scores = baseline_scores[permutation]
            expected_choice = int(np.argmin(expected_scores))
            if expected_choice != int(target):
                raise RuntimeError("complete targeted interchange does not select target")
            entry = {
                "target_slot": int(slot),
                "target_baseline_rank": int(baseline_rank),
                "target_action": int(target),
                "donor_action": int(donor),
                "permutation": permutation.tolist(),
                "expected_counterfactual_choice": expected_choice,
                "baseline_score_sha256": array_sha256(baseline_scores),
            }
            entries.append(entry)
            rows.append(
                {
                    "record_id": record_id,
                    "trajectory_id": int(record["trajectory_id"]),
                    "action_family": record["action_family"],
                    **{key: value for key, value in entry.items() if key != "permutation"},
                    "permutation": " ".join(str(value) for value in permutation),
                }
            )
        target_map[str(record_id)] = entries
    write_csv(DESIGN_DIR / "steering_target_rows.csv", rows)
    freeze = {
        "created_after_baseline_predictions_before_any_intervention": True,
        "selection_rule": "baseline planner ranks 2, 3, and 4; smoke uses rank 2 only",
        "uses_baseline_model_scores": True,
        "uses_simulator_endpoint_costs": False,
        "uses_intervention_outputs": False,
        "target_map": target_map,
        "target_rows_sha256": sha256_file(DESIGN_DIR / "steering_target_rows.csv"),
        "prior_artifact_certificate_sha256": sha256_file(
            OUT / "prior_artifact_certificate.json"
        ),
    }
    write_json(DESIGN_DIR / "steering_target_freeze.json", freeze)
    return target_map, freeze


EVALUATION_OPENED = False
if not PIPELINE_FAILED:
    try:
        if not PRIOR_ARTIFACTS_VALIDATED:
            raise RuntimeError("successful prior artifacts must be validated before model loading")
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
        STEERING_TARGETS, STEERING_TARGET_FREEZE = freeze_steering_targets(
            ALL_EVALUATION_RECORDS
        )
        EVALUATION_OPENED = True
        write_json(
            OUT / "evaluation_open_certificate.json",
            {
                "opened": True,
                "source_identity": SOURCE_IDENTITY,
                "prior_artifact_certificate_sha256": sha256_file(
                    OUT / "prior_artifact_certificate.json"
                ),
                "physical_selection_freeze_sha256": sha256_file(
                    DESIGN_DIR / "physical_selection_freeze.json"
                ),
                "steering_target_freeze_sha256": sha256_file(
                    DESIGN_DIR / "steering_target_freeze.json"
                ),
                "records_by_family": {
                    family: len(FAMILY_RECORDS[family]) for family in TRANSFER_FAMILIES
                },
                "targets_per_record": len(ACTIVE_TARGET_BASELINE_RANKS),
                "intervention_outputs_seen_during_target_selection": [],
                "simulator_endpoint_costs_seen_during_target_selection": [],
            },
        )
        memory_report("baselines_and_steering_targets_frozen")
    except Exception:
        record_failure("steering_model_baselines_and_targets")
'''


causal_steering = r'''# Intervene on the frozen subspaces and measure prediction, ranking, choice, and physical cost.


def load_frozen_subspaces():
    if not PRIOR_ARTIFACTS_VALIDATED:
        raise RuntimeError("prior artifacts are not validated")
    return FROZEN_SUBSPACES


def whiten_carrier(values, subspaces):
    return transform_primal_channels(
        np.asarray(values, dtype=np.float64),
        subspaces["channel_inverse_square_root"],
    )


def native_edit(values, subspaces):
    return inverse_transform_primal_channels(
        np.asarray(values, dtype=np.float64), subspaces["channel_square_root"]
    )


def truth_costs(record):
    with np.load(branch_path(record["record_id"])) as truth:
        endpoints = truth["endpoint_states"].astype(np.float64)
        goal = truth["goal"].astype(np.float64)
    return decoded_task_cost(pose_target(endpoints), goal), goal


def intervention_path(record_id):
    return INTERVENTION_DIR / f"state_{int(record_id):06d}.json"


def finite_json_rows(rows):
    return [
        {
            key: None
            if isinstance(value, (float, np.floating)) and not np.isfinite(value)
            else value
            for key, value in row.items()
        }
        for row in rows
    ]


def wrong_state_delta(current_white, wrong_white, permutation, basis):
    current_residual = candidate_center(current_white.reshape(ACTIONS_PER_STATE, -1))
    wrong_residual = candidate_center(wrong_white.reshape(ACTIONS_PER_STATE, -1))
    difference = wrong_residual[permutation] - current_residual
    return ((difference @ basis) @ basis.T).reshape(current_white.shape)


def make_result_row(
    record,
    target_entry,
    condition,
    control_family,
    mode,
    rank,
    dose,
    baseline_output,
    patched_output,
    baseline_pose,
    patched_pose,
    true_cost,
    goal,
    edit_norm,
    primary_swap_norm,
    full_swap_norm,
):
    permutation = np.asarray(target_entry["permutation"], dtype=np.int64)
    target_action = int(target_entry["target_action"])
    baseline_scores = decoded_task_cost(baseline_pose, goal)
    patched_scores = decoded_task_cost(patched_pose, goal)
    output = donor_transfer_metrics(baseline_output, patched_output, permutation)
    pose = donor_transfer_metrics(baseline_pose, patched_pose, permutation)
    energy = action_contrast_energy_metrics(baseline_output, patched_output)
    steering = planner_steering_metrics(
        baseline_scores,
        patched_scores,
        true_cost,
        permutation,
        target_action,
    )
    return {
        "record_id": int(record["record_id"]),
        "trajectory_id": int(record["trajectory_id"]),
        "task_id": int(record["task_id"]),
        "action_family": record["action_family"],
        "target_slot": int(target_entry["target_slot"]),
        "target_baseline_rank_frozen": int(target_entry["target_baseline_rank"]),
        "donor_action_frozen": int(target_entry["donor_action"]),
        "selected_block": FIXED_BLOCK,
        "condition": condition,
        "control_family": control_family,
        "mode": mode,
        "rank": int(rank),
        "dose": float(dose),
        "output_coefficient": output["coefficient"],
        "output_cosine": output["cosine"],
        "output_mean_shift_ratio": output["mean_shift_ratio"],
        "pose_coefficient": pose["coefficient"],
        "pose_cosine": pose["cosine"],
        "output_contrast_energy_reduction": energy["energy_reduction"],
        "output_contrast_cosine": energy["contrast_cosine"],
        **steering,
        "carrier_edit_whitened_norm": float(edit_norm),
        "primary_swap_norm": float(primary_swap_norm),
        "full_swap_norm": float(full_swap_norm),
        "edit_to_full_swap_ratio": float(edit_norm) / max(float(full_swap_norm), 1e-12),
    }


def intervention_specs(record, carrier, subspaces):
    record_id = int(record["record_id"])
    white = whiten_carrier(carrier, subspaces)
    primary_basis = subspaces["primary_basis"][:, :PRIMARY_STEERING_RANK]
    sensitivity_basis = subspaces["primary_basis"][:, :SENSITIVITY_RANK]
    specifications = []

    def add(target_entry, condition, family, mode, rank, dose, delta):
        specifications.append(
            {
                "target_entry": target_entry,
                "condition": condition,
                "control_family": family,
                "mode": mode,
                "rank": int(rank),
                "dose": float(dose),
                "delta_white": np.asarray(delta, dtype=np.float64),
            }
        )

    for target_entry in STEERING_TARGETS[str(record_id)]:
        permutation = np.asarray(target_entry["permutation"], dtype=np.int64)
        primary = action_swap_delta(white, permutation, primary_basis, dose=1.0)
        sensitivity = action_swap_delta(
            white, permutation, sensitivity_basis, dose=1.0
        )
        full_swap = action_swap_delta(white, permutation, basis=None, dose=1.0)
        if min(np.linalg.norm(primary), np.linalg.norm(full_swap)) <= 1e-12:
            raise RuntimeError("targeted steering edit is degenerate")
        for dose in ACTIVE_STEERING_DOSES:
            add(
                target_entry,
                f"learned_r{PRIMARY_STEERING_RANK:03d}",
                "primary",
                "targeted_replacement",
                PRIMARY_STEERING_RANK,
                dose,
                float(dose) * primary,
            )
        add(
            target_entry,
            f"learned_r{SENSITIVITY_RANK:03d}",
            "rank_sensitivity",
            "targeted_replacement",
            SENSITIVITY_RANK,
            1.0,
            sensitivity,
        )
        shuffled = action_swap_delta(
            white,
            permutation,
            subspaces["shuffled_basis"][:, :PRIMARY_STEERING_RANK],
            dose=1.0,
        )
        add(
            target_entry,
            f"shuffled_r{PRIMARY_STEERING_RANK:03d}",
            "matched_shuffled_control",
            "targeted_replacement",
            PRIMARY_STEERING_RANK,
            1.0,
            norm_match(shuffled, primary),
        )
        for draw in range(ACTIVE_CAUSAL_RANDOM_DRAWS):
            random_delta = action_swap_delta(
                white,
                permutation,
                subspaces[f"random_basis_{draw:02d}"][:, :PRIMARY_STEERING_RANK],
                dose=1.0,
            )
            add(
                target_entry,
                f"random_r{PRIMARY_STEERING_RANK:03d}_{draw:02d}",
                "empirical_span_random_control",
                "targeted_replacement",
                PRIMARY_STEERING_RANK,
                1.0,
                norm_match(random_delta, primary),
            )
        wrong_id = int(wrong_state_map[str(record_id)])
        wrong_carrier = carrier_for_block(load_baseline(wrong_id), FIXED_BLOCK)
        wrong = wrong_state_delta(
            white,
            whiten_carrier(wrong_carrier, subspaces),
            permutation,
            primary_basis,
        )
        add(
            target_entry,
            f"wrong_state_r{PRIMARY_STEERING_RANK:03d}",
            "state_specificity_control",
            "targeted_replacement",
            PRIMARY_STEERING_RANK,
            1.0,
            norm_match(wrong, primary),
        )
        add(
            target_entry,
            f"common_mode_r{PRIMARY_STEERING_RANK:03d}",
            "matched_common_mode_control",
            "targeted_replacement",
            PRIMARY_STEERING_RANK,
            1.0,
            matched_common_mode(primary, primary_basis[:, 0]),
        )
        add(
            target_entry,
            "full_activation_swap",
            "positive_control_only",
            "targeted_replacement",
            -1,
            1.0,
            full_swap,
        )

    primary_ablation = projection_ablation_delta(white, primary_basis, dose=1.0)
    add(
        None,
        f"ablate_primary_r{PRIMARY_STEERING_RANK:03d}",
        "primary",
        "necessity",
        PRIMARY_STEERING_RANK,
        1.0,
        primary_ablation,
    )
    shuffled_ablation = projection_ablation_delta(
        white,
        subspaces["shuffled_basis"][:, :PRIMARY_STEERING_RANK],
        dose=1.0,
    )
    add(
        None,
        f"ablate_shuffled_r{PRIMARY_STEERING_RANK:03d}",
        "matched_shuffled_control",
        "necessity",
        PRIMARY_STEERING_RANK,
        1.0,
        norm_match(shuffled_ablation, primary_ablation),
    )
    for draw in range(ACTIVE_CAUSAL_RANDOM_DRAWS):
        random_ablation = projection_ablation_delta(
            white,
            subspaces[f"random_basis_{draw:02d}"][:, :PRIMARY_STEERING_RANK],
            dose=1.0,
        )
        add(
            None,
            f"ablate_random_r{PRIMARY_STEERING_RANK:03d}_{draw:02d}",
            "empirical_span_random_control",
            "necessity",
            PRIMARY_STEERING_RANK,
            1.0,
            norm_match(random_ablation, primary_ablation),
        )

    if len(specifications) != ACTIVE_INTERVENTION_FORWARDS_PER_RECORD:
        raise RuntimeError(
            f"expected {ACTIVE_INTERVENTION_FORWARDS_PER_RECORD} patched forwards, "
            f"found {len(specifications)}"
        )
    return white, specifications


def run_record_interventions(record, subspaces):
    destination = intervention_path(record["record_id"])
    if destination.exists():
        PROVENANCE_COUNTS["cache_hits"] += 1
        raise RuntimeError(f"fresh-run intervention shard already exists: {destination}")
    payload = load_baseline(record["record_id"])
    carrier = carrier_for_block(payload, FIXED_BLOCK)
    baseline_output = payload["output_eval_sketch"].astype(np.float64)
    baseline_pose = payload["decoded_pose"].astype(np.float64)
    true_cost, goal = truth_costs(record)
    white, specifications = intervention_specs(record, carrier, subspaces)
    target_entries = STEERING_TARGETS[str(int(record["record_id"]))]
    rows = [
        make_result_row(
            record,
            target_entry,
            "no_edit",
            "baseline",
            "baseline",
            0,
            0.0,
            baseline_output,
            baseline_output,
            baseline_pose,
            baseline_pose,
            true_cost,
            goal,
            0.0,
            0.0,
            0.0,
        )
        for target_entry in target_entries
    ]
    initial, actions = state_model_inputs(record["record_id"])
    for specification in specifications:
        delta_native = native_edit(specification["delta_white"], subspaces)
        delta_tensor = torch.as_tensor(delta_native, device="cuda", dtype=torch.float32)
        with torch.inference_mode():
            patched, _, _ = forward_with_carriers(
                initial,
                actions,
                PRIMARY_HORIZON,
                capture_blocks=[FIXED_BLOCK],
                intervention={"block": FIXED_BLOCK, "delta": delta_tensor},
            )
            patched_output = EVAL_OUTPUT_PROJECTOR(patched).cpu().numpy()
            patched_pose = DECODE_PHYSICAL_POSE(patched).cpu().numpy()
        entries = (
            [specification["target_entry"]]
            if specification["target_entry"] is not None
            else target_entries
        )
        for target_entry in entries:
            permutation = np.asarray(target_entry["permutation"], dtype=np.int64)
            primary_swap = action_swap_delta(
                white,
                permutation,
                subspaces["primary_basis"][:, :PRIMARY_STEERING_RANK],
                dose=1.0,
            )
            full_swap = action_swap_delta(white, permutation, basis=None, dose=1.0)
            rows.append(
                make_result_row(
                    record,
                    target_entry,
                    specification["condition"],
                    specification["control_family"],
                    specification["mode"],
                    specification["rank"],
                    specification["dose"],
                    baseline_output,
                    patched_output,
                    baseline_pose,
                    patched_pose,
                    true_cost,
                    goal,
                    np.linalg.norm(specification["delta_white"]),
                    np.linalg.norm(primary_swap),
                    np.linalg.norm(full_swap),
                )
            )
        del patched, patched_output, patched_pose, delta_tensor
    if len(rows) != ACTIVE_RESULT_ROWS_PER_RECORD:
        raise RuntimeError(
            f"expected {ACTIVE_RESULT_ROWS_PER_RECORD} result rows, found {len(rows)}"
        )
    write_json(destination, finite_json_rows(rows))
    PROVENANCE_COUNTS["intervention_generated"] += 1
    PROVENANCE_COUNTS["patched_forwards_generated"] += len(specifications)
    del initial, actions
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def run_all_interventions(records):
    started = time.perf_counter()
    subspaces = load_frozen_subspaces()
    rows = []
    for index, record in enumerate(records):
        rows.extend(run_record_interventions(record, subspaces))
        write_json(
            OUT / "intervention_progress.json",
            {
                "completed": index + 1,
                "total": len(records),
                "last_record_id": int(record["record_id"]),
                "patched_forwards_generated": PROVENANCE_COUNTS["patched_forwards_generated"],
            },
        )
    TIMINGS["causal_steering_seconds"] = time.perf_counter() - started
    write_csv(EVIDENCE_DIR / "steering_state_rows.csv", rows)
    return rows


if not PIPELINE_FAILED and EVALUATION_OPENED:
    try:
        STEERING_ROWS = run_all_interventions(ALL_EVALUATION_RECORDS)
        memory_report("causal_planner_steering_complete")
    except Exception:
        record_failure("causal_planner_steering")
'''


decision = r'''# Apply frozen family-level representation and planner-steering gates.


def result_lookup(rows, family, record_id, target_slot, condition, key, dose=1.0):
    values = [
        row[key]
        for row in rows
        if row["action_family"] == family
        and row["record_id"] == record_id
        and row["target_slot"] == target_slot
        and row["condition"] == condition
        and np.isclose(row["dose"], dose)
    ]
    return float(values[0]) if len(values) == 1 else np.nan


def family_attempts(family):
    return [
        (int(record["record_id"]), int(record["trajectory_id"]), int(entry["target_slot"]))
        for record in FAMILY_RECORDS[family]
        for entry in STEERING_TARGETS[str(int(record["record_id"]))]
    ]


def random_median(rows, family, record_id, target_slot, key, ablate=False):
    prefix = "ablate_random" if ablate else "random"
    values = [
        result_lookup(
            rows,
            family,
            record_id,
            target_slot,
            f"{prefix}_r{PRIMARY_STEERING_RANK:03d}_{draw:02d}",
            key,
        )
        for draw in range(ACTIVE_CAUSAL_RANDOM_DRAWS)
    ]
    return float(np.nanmedian(values))


def bootstrap_interval(values, clusters, family, label):
    seed = stable_seed(BOOTSTRAP_SEED, family, label) % (2**31 - 1)
    draws = clustered_bootstrap_mean(
        np.asarray(values, dtype=np.float64),
        np.asarray(clusters, dtype=np.int64),
        ACTIVE_BOOTSTRAP_DRAWS,
        seed,
    )
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def trajectory_means(values, trajectories):
    values = np.asarray(values, dtype=np.float64)
    trajectories = np.asarray(trajectories, dtype=np.int64)
    return np.asarray([
        np.mean(values[trajectories == trajectory])
        for trajectory in np.unique(trajectories)
    ])


def evaluate_family(rows, family):
    attempts = family_attempts(family)
    record_ids = [value[0] for value in attempts]
    trajectories = [value[1] for value in attempts]
    slots = [value[2] for value in attempts]

    def values(condition, key, dose=1.0):
        return np.asarray([
            result_lookup(rows, family, record_id, slot, condition, key, dose)
            for record_id, slot in zip(record_ids, slots)
        ])

    learned_name = f"learned_r{PRIMARY_STEERING_RANK:03d}"
    learned_output = values(learned_name, "output_coefficient")
    half_output = (
        values(learned_name, "output_coefficient", 0.5)
        if 0.5 in ACTIVE_STEERING_DOSES else np.full(len(attempts), np.nan)
    )
    random_output = np.asarray([
        random_median(rows, family, record_id, slot, "output_coefficient")
        for record_id, slot in zip(record_ids, slots)
    ])
    shuffled_output = values(
        f"shuffled_r{PRIMARY_STEERING_RANK:03d}", "output_coefficient"
    )
    full_output = values("full_activation_swap", "output_coefficient")
    output_gain_random = learned_output - random_output
    output_gain_shuffled = learned_output - shuffled_output

    learned_rank_gain = values(learned_name, "target_rank_gain")
    random_rank_gain = np.asarray([
        random_median(rows, family, record_id, slot, "target_rank_gain")
        for record_id, slot in zip(record_ids, slots)
    ])
    shuffled_rank_gain = values(
        f"shuffled_r{PRIMARY_STEERING_RANK:03d}", "target_rank_gain"
    )
    rank_gain_random = learned_rank_gain - random_rank_gain
    rank_gain_shuffled = learned_rank_gain - shuffled_rank_gain

    learned_choice = values(learned_name, "choice_matches_counterfactual")
    random_choice = np.asarray([
        random_median(
            rows, family, record_id, slot, "choice_matches_counterfactual"
        )
        for record_id, slot in zip(record_ids, slots)
    ])
    shuffled_choice = values(
        f"shuffled_r{PRIMARY_STEERING_RANK:03d}",
        "choice_matches_counterfactual",
    )
    full_choice = values("full_activation_swap", "choice_matches_counterfactual")
    choice_gain_random = learned_choice - random_choice
    choice_gain_shuffled = learned_choice - shuffled_choice

    learned_score = values(learned_name, "score_transfer_coefficient")
    learned_flip = values(learned_name, "choice_flipped")
    learned_physical_change = values(learned_name, "selected_true_cost_change")
    sensitivity_output = values(
        f"learned_r{SENSITIVITY_RANK:03d}", "output_coefficient"
    )

    # Necessity energy is target-independent. Use one row per record.
    unique_records = [(int(record["record_id"]), int(record["trajectory_id"])) for record in FAMILY_RECORDS[family]]
    ablate_name = f"ablate_primary_r{PRIMARY_STEERING_RANK:03d}"
    ablate_shuffled_name = f"ablate_shuffled_r{PRIMARY_STEERING_RANK:03d}"
    necessity = np.asarray([
        result_lookup(
            rows, family, record_id, 0, ablate_name,
            "output_contrast_energy_reduction",
        )
        for record_id, _ in unique_records
    ])
    necessity_shuffled = np.asarray([
        result_lookup(
            rows, family, record_id, 0, ablate_shuffled_name,
            "output_contrast_energy_reduction",
        )
        for record_id, _ in unique_records
    ])
    necessity_random = np.asarray([
        random_median(
            rows,
            family,
            record_id,
            0,
            "output_contrast_energy_reduction",
            ablate=True,
        )
        for record_id, _ in unique_records
    ])
    necessity_gain_random = necessity - necessity_random
    necessity_gain_shuffled = necessity - necessity_shuffled
    necessity_clusters = [value[1] for value in unique_records]

    finite_arrays = [
        learned_output, random_output, shuffled_output, full_output,
        learned_rank_gain, random_rank_gain, shuffled_rank_gain,
        learned_choice, random_choice, shuffled_choice, full_choice,
        learned_score, learned_flip, learned_physical_change,
        sensitivity_output, necessity, necessity_random, necessity_shuffled,
    ]
    finite = bool(all(np.all(np.isfinite(value)) for value in finite_arrays))
    output_ci = bootstrap_interval(
        output_gain_random, trajectories, family, "output_gain_random"
    )
    rank_ci = bootstrap_interval(
        rank_gain_random, trajectories, family, "rank_gain_random"
    )
    choice_ci = bootstrap_interval(
        choice_gain_random, trajectories, family, "choice_gain_random"
    )
    necessity_ci = bootstrap_interval(
        necessity_gain_random,
        necessity_clusters,
        family,
        "necessity_gain_random",
    )
    trajectory_output_gain = trajectory_means(output_gain_random, trajectories)
    trajectory_rank_gain = trajectory_means(rank_gain_random, trajectories)

    representation_pass = bool(
        finite
        and np.mean(full_output) >= MIN_FULL_SWAP_COEFFICIENT
        and np.mean(learned_output) >= MIN_PRIMARY_OUTPUT_COEFFICIENT
        and np.mean(output_gain_random) >= MIN_OUTPUT_GAIN_OVER_RANDOM
        and np.mean(output_gain_shuffled) > 0
        and np.mean(necessity) >= MIN_NECESSITY_REDUCTION
        and np.mean(necessity_gain_random) >= MIN_NECESSITY_GAIN_OVER_RANDOM
        and np.mean(necessity_gain_shuffled) >= MIN_NECESSITY_GAIN_OVER_SHUFFLED
        and (output_ci[0] > 0 if RUN_MODE == "pilot" else True)
        and (necessity_ci[0] > 0 if RUN_MODE == "pilot" else True)
        and (
            RUN_MODE == "smoke"
            or (
                np.mean(learned_output - half_output) > 0
                and exact_positive_sign_test(trajectory_output_gain)["p_value"] <= 0.05
            )
        )
    )
    planner_pass = bool(
        finite
        and np.mean(full_choice) >= MIN_FULL_TARGET_CHOICE_RATE
        and np.mean(rank_gain_random) >= MIN_TARGET_RANK_GAIN_OVER_RANDOM
        and np.mean(rank_gain_shuffled) > 0
        and np.mean(choice_gain_random) >= MIN_CHOICE_MATCH_GAIN_OVER_RANDOM
        and np.mean(choice_gain_shuffled) > 0
        and (rank_ci[0] > 0 if RUN_MODE == "pilot" else True)
        and (choice_ci[0] > 0 if RUN_MODE == "pilot" else True)
        and (
            exact_positive_sign_test(trajectory_rank_gain)["p_value"] <= 0.05
            if RUN_MODE == "pilot" else True
        )
    )
    return {
        "action_family": family,
        "trajectories": len(unique_records),
        "target_attempts": len(attempts),
        "all_required_metrics_finite": finite,
        "mean_full_output_coefficient": float(np.mean(full_output)),
        "mean_learned_output_coefficient_r128": float(np.mean(learned_output)),
        "mean_learned_output_coefficient_r64": float(np.mean(sensitivity_output)),
        "mean_random_output_coefficient": float(np.mean(random_output)),
        "mean_shuffled_output_coefficient": float(np.mean(shuffled_output)),
        "mean_output_gain_over_random": float(np.mean(output_gain_random)),
        "output_gain_over_random_ci95": output_ci,
        "output_gain_over_random_sign_test_by_trajectory": exact_positive_sign_test(trajectory_output_gain),
        "mean_full_counterfactual_choice_rate": float(np.mean(full_choice)),
        "mean_learned_counterfactual_choice_rate": float(np.mean(learned_choice)),
        "mean_random_counterfactual_choice_rate": float(np.mean(random_choice)),
        "mean_shuffled_counterfactual_choice_rate": float(np.mean(shuffled_choice)),
        "mean_choice_match_gain_over_random": float(np.mean(choice_gain_random)),
        "choice_match_gain_over_random_ci95": choice_ci,
        "mean_learned_target_rank_gain": float(np.mean(learned_rank_gain)),
        "mean_random_target_rank_gain": float(np.mean(random_rank_gain)),
        "mean_shuffled_target_rank_gain": float(np.mean(shuffled_rank_gain)),
        "mean_target_rank_gain_over_random": float(np.mean(rank_gain_random)),
        "target_rank_gain_over_random_ci95": rank_ci,
        "target_rank_gain_over_random_sign_test_by_trajectory": exact_positive_sign_test(trajectory_rank_gain),
        "mean_planner_score_transfer_coefficient": float(np.mean(learned_score)),
        "mean_learned_choice_flip_rate": float(np.mean(learned_flip)),
        "mean_selected_true_cost_change": float(np.mean(learned_physical_change)),
        "mean_necessity_energy_reduction": float(np.mean(necessity)),
        "mean_necessity_random_reduction": float(np.mean(necessity_random)),
        "mean_necessity_shuffled_reduction": float(np.mean(necessity_shuffled)),
        "mean_necessity_gain_over_random": float(np.mean(necessity_gain_random)),
        "necessity_gain_over_random_ci95": necessity_ci,
        "representation_gate_pass": representation_pass,
        "planner_steering_gate_pass": planner_pass,
        "causal_planner_chain_gate_pass": bool(representation_pass and planner_pass),
    }


def fresh_run_certificate():
    expected = {
        "truth_generated": len(POOL_SPECS),
        "baseline_generated": len(ALL_EVALUATION_RECORDS),
        "intervention_generated": len(ALL_EVALUATION_RECORDS),
        "patched_forwards_generated": len(ALL_EVALUATION_RECORDS)
        * ACTIVE_INTERVENTION_FORWARDS_PER_RECORD,
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


if PIPELINE_FAILED:
    DECISION_PAYLOAD = {"status": "INCONCLUSIVE", "failure": FAILURE_MESSAGE}
elif not EVALUATION_OPENED:
    DECISION_PAYLOAD = {"status": "INCONCLUSIVE", "reason": "steering targets were not frozen"}
else:
    try:
        FAMILY_RESULTS = {
            family: evaluate_family(STEERING_ROWS, family)
            for family in TRANSFER_FAMILIES
        }
        FRESH_CERTIFICATE = fresh_run_certificate()
        representation_families = [
            family for family in TRANSFER_FAMILIES
            if FAMILY_RESULTS[family]["representation_gate_pass"]
        ]
        planner_families = [
            family for family in TRANSFER_FAMILIES
            if FAMILY_RESULTS[family]["causal_planner_chain_gate_pass"]
        ]
        if RUN_MODE == "smoke":
            candidate_status = "SMOKE_ONLY"
        elif len(planner_families) == len(TRANSFER_FAMILIES):
            candidate_status = "CONFIRMED_CAUSAL_PLANNER_STEERING_BOTH_FAMILIES"
        elif planner_families:
            candidate_status = "PARTIAL_CAUSAL_PLANNER_STEERING"
        elif len(representation_families) == len(TRANSFER_FAMILIES):
            candidate_status = "PREDICTION_MEDIATOR_TRANSFER_WITHOUT_CONFIRMED_PLANNER_STEERING"
        else:
            candidate_status = "NO_CONFIRMED_STAGE20_CAUSAL_CHAIN"
        source_eligible = bool(SOURCE_IDENTITY.get("confirmation_eligible", False))
        prior_eligible = bool(PRIOR_ARTIFACTS_VALIDATED)
        confirmation_eligible = bool(
            source_eligible and prior_eligible and FRESH_CERTIFICATE["passed"]
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
            "prior_artifacts_claim_eligible": prior_eligible,
            "fresh_run_claim_eligible": FRESH_CERTIFICATE["passed"],
            "confirmation_eligible": confirmation_eligible,
            "representation_passed_families": representation_families,
            "planner_chain_passed_families": planner_families,
            "family_results": FAMILY_RESULTS,
            "claim_boundary": {
                "nonvisual_evaluation_only": True,
                "steering_targets_are_near_frontier_baseline_actions": True,
                "targets_selected_without_simulator_outcomes": True,
                "physical_improvement_claim_authorized": False,
                "multi_step_closed_loop_control_tested": False,
                "other_models_or_environments_generalized": False,
                "causal_claim_is_prediction_to_ranking_to_choice_mediation": True,
            },
        }
        write_json(OUT / "stage20_decision.json", DECISION_PAYLOAD)
        print(json.dumps(DECISION_PAYLOAD, indent=2))
    except Exception:
        record_failure("decision")
        DECISION_PAYLOAD = {"status": "INCONCLUSIVE", "failure": FAILURE_MESSAGE}

if not (OUT / "stage20_decision.json").exists():
    write_json(OUT / "stage20_decision.json", DECISION_PAYLOAD)
'''


packaging = base_source(12)
packaging = packaging.replace(
    "stage19_unseen_action_transfer_result_bundle_",
    "stage20_causal_planner_steering_result_bundle_",
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
    model_and_targets,
    causal_steering,
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
    code(model_and_targets),
    code(causal_steering),
    code(decision),
    code(packaging),
]
for index, cell in enumerate(cells):
    cell["id"] = f"stage20-{index:02d}"

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
