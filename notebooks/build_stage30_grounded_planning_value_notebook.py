import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "30_grounded_causal_planning_value.ipynb"
NUMERICAL = ROOT.parent / "src/cf_faithfulness/stage30_grounded_planning_value.py"

spec = importlib.util.spec_from_file_location(
    "stage29_builder", ROOT / "build_stage29_grounded_causal_closure_notebook.py"
)
STAGE29 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(STAGE29)

code = STAGE29.code
markdown = STAGE29.markdown
assigned_uppercase_names = STAGE29.assigned_uppercase_names
function_sources = STAGE29.function_sources


introduction = r'''# Stage 30: grounded causal closure predicts planning value

Stage 29 established a prospective mathematical gap.  The frozen Stage 18
rank-128 carrier strongly followed the JEPA-WM predictor's own counterfactual
action contrast, but followed the exact simulator counterfactual only weakly.
The gap was quantitatively consistent with a carrier that mediates the
predictor's internal dynamics while the predictor itself contains only a
partial component of the true physical contrast.

Stage 30 asks the missing field-level question: **does that distinction matter
for decisions?**  It creates a completely fresh, model-blind pool of PushT
states, selects persistent-contact, boundary-switching, and free-motion states
using contact counts alone, and generates exact futures for the same frozen
24-action signed-area bank.  No Stage 29 state is reused.

For every selected state and magnitude, the terminal native JEPA planner ranks
the six histogram-matched schedules using

\[
c_{\mathrm{JEPA}}(a;g)=
\left\|\hat z_a-z^*_g\right\|_2,
\qquad
\hat z_a=P(E(o_t),a),\quad z^*_g=E(o^{\mathrm{sim}}_{t+H,g}).
\]

The exact physical ranking uses normalized simulator block-pose distance to
the same goal endpoint.  Goals are the two extreme signed-area schedules.
Crucially, self and grounded causal closure are estimated only on the four
*interior* schedules.  Thus the primary planning target is not reused as the
closure target.

The notebook evaluates two preregistered consequences:

1. whether grounded closure improves state-grouped out-of-fold prediction of
   native planning regret beyond total latent prediction error and
   self-consistent causal closure;
2. whether grounded closure predicts the physical planning-value loss caused
   by ablating the carrier, with shuffled and empirical-span random ablations
   as matched controls.

Free-motion states provide an exact null: when schedule ordering produces no
true endpoint contrast, the notebook measures false latent planning margins
and intervention energy rather than inventing undefined grounded coefficients.

The frozen Stage 18 carrier is never refit or rotated.  There is no learned
decoder, new reader, goal-conditioned fit, Jacobian, JVP, VJP, gradient, or
visual scoring.  Statistical folds are grouped by initial state.  Return
`stage30_grounded_planning_value_result_bundle_<signature>.zip`.
'''


configuration = r'''# SINGLE CONFIGURATION BLOCK — no Stage 30 secrets required.
# Run all on a GPU. A fresh nonce is generated automatically. HF_TOKEN is read
# only if the pinned public checkpoint is not already cached.
import secrets as _secrets
import time as _time

RUN_MODE = "pilot"
EXPERIMENT_SOURCE_REF = "codex/stage30-grounded-planning-value"
RUN_NONCE = f"auto_{_time.strftime('%Y%m%d_%H%M%S')}_{_secrets.token_hex(4)}"

try:
    import os as _os
    from google.colab import userdata as _colab_userdata

    _hf_token = str(_colab_userdata.get("HF_TOKEN") or "").strip()
    if _hf_token:
        _os.environ["HF_TOKEN"] = _hf_token
        _os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_token
except Exception:
    _hf_token = ""

if not all(value.isalnum() or value in "-_" for value in RUN_NONCE):
    raise ValueError("automatic RUN_NONCE contains an invalid character")

MOUNT_DRIVE = True
DOWNLOAD_RESULTS = True
CONTINUE_AFTER_BENCHMARK = True
MAX_ESTIMATED_TOTAL_MINUTES = 180.0
FRESH_RUN_REQUIRED = True

OUTPUT_DIR = "/content/counterfactual_faithfulness_stage30_grounded_planning"
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/counterfactual_faithfulness_stage30_grounded_planning"
STAGE18_SEARCH_ROOT = "/content/drive/MyDrive"
STAGE29_SEARCH_ROOT = "/content/drive/MyDrive"

PROTOCOL_ID = "stage30-grounded-causal-planning-value-v1"
NOTEBOOK_PROTOCOL_SHA256 = "__PROTOCOL_DIGEST__"
EVIDENCE_STATUS = "CONFIRMATORY_ONLY_IF_SOURCE_BOUND_FRESH_STAGE18_FROZEN_AND_EXACT_STAGE29_BOUND"
EXPERIMENT_REPOSITORY = "grewalsk/counterfactual-faithfulness-research"
EXPERIMENT_NOTEBOOK_PATH = "notebooks/30_grounded_causal_planning_value.ipynb"
EXPERIMENT_BUILDER_PATH = "notebooks/build_stage30_grounded_planning_value_notebook.py"
EXPERIMENT_NUMERICAL_PATH = "src/cf_faithfulness/stage30_grounded_planning_value.py"

EXPECTED_STAGE18_SUBSPACE_SHA256 = "2f9c496d54623a9062e465a18c70039acc18cb8a1cc2833a5f4ade162ca3f90b"
EXPECTED_STAGE18_SOURCE_COMMIT = "16edd247cddcb1aa121340eb5fa42bd9e07004c3"
EXPECTED_STAGE18_STATUS = "CONFIRMED_BIDIRECTIONAL_RANK64_MEDIATOR"
EXPECTED_STAGE18_AMBIENT_DIMENSION = 102400
EXPECTED_STAGE18_MAX_RANK = 128
EXPECTED_STAGE29_STATUS = "PHYSICAL_READOUT_LIMITATION_SUPPORTED"
EXPECTED_STAGE29_SOURCE_COMMIT = "c0fc29df0fd5a150762cbb918d5401624e780833"
EXPECTED_STAGE29_PROTOCOL_ID = "stage29-grounded-causal-closure-v1"
EXPECTED_STAGE29_DECISION_SHA256 = "1e1dd615f8a082870ca0b797e4bd597da628d958c33f1a590ea92600835fee28"
EXPECTED_STAGE29_SOURCE_SHA256 = "001f365d97926c6db83cf5b85d62ec8610ba77455d15ee98f74685b846cf60b6"

SEED = 30101
DESIGN_SEED = 30137
BOOTSTRAP_SEED = 30269
CROSSFIT_SEED = 30319
MODEL_NAME = "jepa_wm_pusht"
ENVIRONMENT = "PushT"
FRAMESKIP = 5
PRIMARY_HORIZON = 3
TARGET_STEPS = [PRIMARY_HORIZON]
ACTION_STEPS = PRIMARY_HORIZON * FRAMESKIP
FIXED_BLOCK = 4
ACTIVE_BLOCKS = [FIXED_BLOCK]
EXPECTED_CARRIER_CHANNELS = 400

POOL_TRAJECTORIES = list(range(3500, 3900))
EVALUATION_TARGET_PER_STRATUM = 40
EVALUATION_TRAJECTORY_TARGET = 120
TASK_ID_OFFSET = 30000
DISTANCE_GRID = [55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0]
STRATUM_LABELS = ["persistent_contact", "boundary_switching", "free"]

SELECTED_MAGNITUDES = [0.10, 0.14, 0.18, 0.22]
MAGNITUDE_COUNT = 4
SCHEDULE_STRINGS = [
    "uuuuuvvvvv", "uuvuuvvuvv", "uvuvuvuvuv",
    "vuvuvuvuvu", "vvuvvuuvuu", "vvvvvuuuuu",
]
SCHEDULE_COUNT = 6
SCHEDULE_INVERSION_COUNTS = [0, 5, 10, 15, 20, 25]
SIGNED_AREA_LEVELS = [25, 15, 5, -5, -15, -25]
ANGLE_PAIR_DEGREES = [-30.0, 30.0]
ACTIONS_PER_STATE = MAGNITUDE_COUNT * SCHEDULE_COUNT
DIAGNOSTIC_SCHEDULES = [1, 2, 3, 4]
PLANNING_GOAL_SCHEDULES = [0, 5]

PRIMARY_RANK = 128
CAUSAL_RANDOM_DRAWS = 2
INTERVENTION_FORWARDS_PER_RECORD = 9
BOOTSTRAP_DRAWS = 10000
CROSSFIT_FOLDS = 5
MAX_ZERO_EDIT_ERROR = 1e-6
MIN_PLANNING_TRUE_COST_SPREAD = 1e-5
MAX_FREE_TRUE_COST_SPREAD = 1e-6
MIN_SELF_GROUND_COEFFICIENT_GAP = 0.10
MIN_SELF_CLOSURE_COEFFICIENT = 0.15
MIN_OOF_RELATIVE_MSE_IMPROVEMENT = 0.01
MIN_ABLATION_CONTROL_ADVANTAGE = 0.0
MIN_ELIGIBLE_CONTACT_STATES = 50

if RUN_MODE == "smoke":
    ACTIVE_POOL_TRAJECTORIES = POOL_TRAJECTORIES[:30]
    ACTIVE_TARGET_PER_STRATUM = 1
    ACTIVE_EVALUATION_TARGET = 3
    ACTIVE_CAUSAL_RANDOM_DRAWS = 1
    ACTIVE_BOOTSTRAP_DRAWS = 64
    ACTIVE_CROSSFIT_FOLDS = 2
elif RUN_MODE == "pilot":
    ACTIVE_POOL_TRAJECTORIES = POOL_TRAJECTORIES
    ACTIVE_TARGET_PER_STRATUM = EVALUATION_TARGET_PER_STRATUM
    ACTIVE_EVALUATION_TARGET = EVALUATION_TRAJECTORY_TARGET
    ACTIVE_CAUSAL_RANDOM_DRAWS = CAUSAL_RANDOM_DRAWS
    ACTIVE_BOOTSTRAP_DRAWS = BOOTSTRAP_DRAWS
    ACTIVE_CROSSFIT_FOLDS = CROSSFIT_FOLDS
else:
    raise ValueError("RUN_MODE must be 'smoke' or 'pilot'")

ACTIVE_INTERVENTION_FORWARDS_PER_RECORD = (
    5 + 2 * ACTIVE_CAUSAL_RANDOM_DRAWS
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
ASSET_SPECS = {}
PINNED = [
    "model_weights", "stage18_frozen_subspace", "stage29_exact_decision",
    "fresh_model_blind_contact_strata", "frozen_signed_area_bank",
    "interior_schedule_closure", "heldout_extreme_schedule_goals",
    "native_terminal_latent_planner", "exact_simulator_regret",
    "state_grouped_cross_fitting", "matched_ablation_controls",
]

assert ACTION_STEPS == 15 and ACTIONS_PER_STATE == 24
assert len(SCHEDULE_STRINGS) == SCHEDULE_COUNT == 6
assert len(SCHEDULE_INVERSION_COUNTS) == len(SIGNED_AREA_LEVELS) == SCHEDULE_COUNT
assert SELECTED_MAGNITUDES == [0.10, 0.14, 0.18, 0.22]
assert DIAGNOSTIC_SCHEDULES == [1, 2, 3, 4]
assert PLANNING_GOAL_SCHEDULES == [0, 5]
assert not set(DIAGNOSTIC_SCHEDULES) & set(PLANNING_GOAL_SCHEDULES)
assert EVALUATION_TRAJECTORY_TARGET == 3 * EVALUATION_TARGET_PER_STRATUM
assert FIXED_BLOCK == 4 and PRIMARY_RANK == 128
assert INTERVENTION_FORWARDS_PER_RECORD == 5 + 2 * CAUSAL_RANDOM_DRAWS
'''
configuration += "\n\nPROTOCOL_CONFIG_KEYS = " + repr(
    assigned_uppercase_names(configuration)
) + "\n"


installation = STAGE29.installation


setup = STAGE29.setup
setup = setup.replace("Stage 29", "Stage 30").replace("STAGE29", "STAGE30")
setup = setup.replace("stage29_grounded_closure", "stage30_grounded_planning")
setup = setup.replace("stage29-source-binder", "stage30-source-binder")
setup = setup.replace(
    "stage29_grounded_closure_result_bundle_",
    "stage30_grounded_planning_value_result_bundle_",
)


analysis_helpers = STAGE29.analysis_helpers + "\n\n\n" + function_sources(
    NUMERICAL.read_text(),
    [
        "_validated_layout",
        "schedule_reversal",
        "diagnostic_closure_rows",
        "native_terminal_costs",
        "physical_terminal_costs",
        "terminal_planning_rows",
        "deterministic_group_folds",
        "_ridge_prediction",
        "cross_fitted_incremental_value",
    ],
)


model_helpers = STAGE29.model_helpers.replace(
    "stage29-jepa-wms", "stage30-jepa-wms"
).replace("Stage 29 supports PushT only", "Stage 30 supports PushT only")


design_and_upstream = r'''# Freeze the fresh state pool, action bank, and exact upstream identities before model use.
STAGE18_ARTIFACT_VALIDATED = False
STAGE29_UPSTREAM_BOUND = False


def unique_matching_path(candidates, expected_hash=None):
    existing = sorted({Path(value) for value in candidates if Path(value).is_file()})
    if expected_hash is not None:
        existing = [value for value in existing if sha256_file(value) == expected_hash]
    if not existing:
        raise FileNotFoundError("no matching frozen upstream artifact was found in MyDrive")
    existing.sort(key=lambda value: (len(str(value)), str(value)))
    return existing[0]


def make_specs(trajectory_ids):
    specs = []
    center = np.asarray([256.0, 256.0])
    total = len(POOL_TRAJECTORIES)
    for design_index, trajectory_id in enumerate(trajectory_ids):
        global_index = POOL_TRAJECTORIES.index(int(trajectory_id))
        phase = 0.413 + 2.0 * np.pi * global_index / total
        block = center + 43.0 * np.asarray([np.cos(phase), np.sin(phase)])
        block_angle = ((1.73 * phase + np.pi) % (2.0 * np.pi)) - np.pi
        offsets = [np.pi / 6, 5 * np.pi / 6, 7 * np.pi / 6, 11 * np.pi / 6]
        approach = phase + offsets[global_index % 4] + 0.11 * np.sin(5 * global_index)
        distance = float(DISTANCE_GRID[(5 * global_index + global_index // len(DISTANCE_GRID)) % len(DISTANCE_GRID)])
        agent = block + distance * np.asarray([np.cos(approach), np.sin(approach)])
        goal_index = (23 * global_index + 11) % total
        goal_phase = 0.917 + 2.0 * np.pi * goal_index / total
        goal_xy = center + 72.0 * np.asarray([np.cos(goal_phase), np.sin(goal_phase)])
        specs.append({
            "design_index": int(global_index),
            "record_id": int(340000 + trajectory_id),
            "trajectory_id": int(trajectory_id),
            "task_id": int(TASK_ID_OFFSET + global_index),
            "split": "stage30_evaluation_pool",
            "evaluation_seed": int(DESIGN_SEED + 1013 * global_index),
            "approach_distance": distance,
            "goal": np.asarray([
                goal_xy[0], goal_xy[1],
                ((1.31 * goal_phase + np.pi) % (2.0 * np.pi)) - np.pi,
            ], dtype=np.float64),
            "state": np.asarray([
                agent[0], agent[1], block[0], block[1], block_angle,
                0.0, 0.0, 0.0, 0.0, 0.0,
            ], dtype=np.float64),
        })
    return specs


POOL_SPECS = make_specs(ACTIVE_POOL_TRAJECTORIES)


def candidate_action_bank(record):
    state = np.asarray(record["state"], dtype=np.float64)
    if state.shape != (10,):
        raise ValueError("candidate state must be a ten-dimensional PushT state")
    return area_action_bank(
        state[2:4] - state[:2], SELECTED_MAGNITUDES,
        steps=ACTION_STEPS, angle_pair_degrees=ANGLE_PAIR_DEGREES,
        schedules=SCHEDULE_STRINGS,
    )


np.savez_compressed(
    DESIGN_DIR / "stage30_grounded_planning_design.npz",
    record_ids=np.asarray([row["record_id"] for row in POOL_SPECS], dtype=np.int64),
    initial_states=np.stack([row["state"] for row in POOL_SPECS]),
    goals=np.stack([row["goal"] for row in POOL_SPECS]),
    magnitudes=np.asarray(SELECTED_MAGNITUDES, dtype=np.float64),
    schedules=np.asarray(SCHEDULE_STRINGS),
    signed_area_levels=np.asarray(SIGNED_AREA_LEVELS, dtype=np.int64),
    diagnostic_schedules=np.asarray(DIAGNOSTIC_SCHEDULES, dtype=np.int64),
    planning_goal_schedules=np.asarray(PLANNING_GOAL_SCHEDULES, dtype=np.int64),
)
write_json(DESIGN_DIR / "candidate_pool_manifest.json", {
    "pool_specs": [
        {**{key: value for key, value in row.items() if key not in {"state", "goal"}},
         "state": row["state"].tolist(), "goal": row["goal"].tolist()}
        for row in POOL_SPECS
    ],
    "selection_rule": "first states in each contact stratum; no effect magnitude or model output used",
    "target_per_stratum": ACTIVE_TARGET_PER_STRATUM,
    "diagnostic_schedules": DIAGNOSTIC_SCHEDULES,
    "planning_goal_schedules": PLANNING_GOAL_SCHEDULES,
    "closure_and_goal_schedule_sets_disjoint": True,
})
DESIGN_FREEZE = {
    "created_before_simulator_or_model_data": True,
    "protocol_id": PROTOCOL_ID,
    "run_signature": RUN_SIGNATURE,
    "source_identity": SOURCE_IDENTITY,
    "design_sha256": sha256_file(DESIGN_DIR / "stage30_grounded_planning_design.npz"),
    "pool_manifest_sha256": sha256_file(DESIGN_DIR / "candidate_pool_manifest.json"),
    "stage18_subspace_refit_allowed": False,
    "learned_decoder_or_reader_allowed": False,
    "goal_schedule_contrast_used_for_closure": False,
    "jacobian_or_gradient_allowed": False,
    "model_loaded": bool("MODEL" in globals()),
}
if DESIGN_FREEZE["model_loaded"]:
    raise RuntimeError("model was loaded before the Stage 30 design freeze")
write_json(DESIGN_DIR / "design_freeze.json", DESIGN_FREEZE)


if not PIPELINE_FAILED:
    try:
        verify_executed_notebook_through(
            "# Freeze the fresh state pool, action bank, and exact upstream identities before model use."
        )
        stage18_root = Path(STAGE18_SEARCH_ROOT)
        stage18_candidates = list(stage18_root.glob(
            "counterfactual_faithfulness_stage18_rank64/pilot_*/subspaces/frozen_rank64_confirmation_subspaces.npz"
        ))
        FROZEN_SUBSPACE_PATH = unique_matching_path(
            stage18_candidates, EXPECTED_STAGE18_SUBSPACE_SHA256
        )
        stage18_run_dir = FROZEN_SUBSPACE_PATH.parent.parent
        stage18_decision_path = stage18_run_dir / "stage18_decision.json"
        stage18_manifest_path = stage18_run_dir / "subspaces/subspace_manifest.json"
        stage18_source_path = stage18_run_dir / "source_identity.json"
        for required in [stage18_decision_path, stage18_manifest_path, stage18_source_path]:
            if not required.is_file():
                raise FileNotFoundError(f"Stage 18 provenance file is missing: {required}")
        stage18_decision = json.loads(stage18_decision_path.read_text())
        stage18_manifest = json.loads(stage18_manifest_path.read_text())
        stage18_source = json.loads(stage18_source_path.read_text())
        if (
            stage18_decision.get("status") != EXPECTED_STAGE18_STATUS
            or not stage18_decision.get("confirmation_eligible", False)
        ):
            raise RuntimeError("Stage 18 decision is not the frozen confirmation")
        if stage18_manifest.get("subspace_sha256") != EXPECTED_STAGE18_SUBSPACE_SHA256:
            raise RuntimeError("Stage 18 manifest does not bind the required subspace")
        if (
            stage18_source.get("resolved_commit") != EXPECTED_STAGE18_SOURCE_COMMIT
            or not stage18_source.get("confirmation_eligible", False)
        ):
            raise RuntimeError("Stage 18 source binding mismatch")
        with np.load(FROZEN_SUBSPACE_PATH) as payload:
            FROZEN_SUBSPACES = {name: payload[name].copy() for name in payload.files}
        artifact_contract = validate_stage18_subspace_arrays(
            FROZEN_SUBSPACES,
            ambient=EXPECTED_STAGE18_AMBIENT_DIMENSION,
            max_rank=EXPECTED_STAGE18_MAX_RANK,
        )
        STAGE18_ARTIFACT_CERTIFICATE = {
            "validated_before_stage30_model_activations": True,
            "path": str(FROZEN_SUBSPACE_PATH),
            "bytes": int(FROZEN_SUBSPACE_PATH.stat().st_size),
            "sha256": sha256_file(FROZEN_SUBSPACE_PATH),
            "artifact_contract": artifact_contract,
            "stage30_subspace_refit": False,
            "stage30_basis_rotation_or_tuning": False,
        }
        write_json(OUT / "stage18_artifact_certificate.json", STAGE18_ARTIFACT_CERTIFICATE)
        STAGE18_ARTIFACT_VALIDATED = True

        stage29_root = Path(STAGE29_SEARCH_ROOT)
        candidates = list(stage29_root.glob(
            "counterfactual_faithfulness_stage29_grounded_closure/pilot_*/stage29_decision.json"
        ))
        valid = []
        for decision_path in candidates:
            source_path = decision_path.parent / "source_identity.json"
            if not source_path.is_file():
                continue
            if (
                sha256_file(decision_path) != EXPECTED_STAGE29_DECISION_SHA256
                or sha256_file(source_path) != EXPECTED_STAGE29_SOURCE_SHA256
            ):
                continue
            decision = json.loads(decision_path.read_text())
            source = json.loads(source_path.read_text())
            if (
                decision.get("status") == EXPECTED_STAGE29_STATUS
                and decision.get("confirmation_eligible", False)
                and decision.get("self_consistent_causal_closure_gate", {}).get("passed", False)
                and not decision.get("grounded_causal_closure_gate", {}).get("passed", True)
                and source.get("protocol_id") == EXPECTED_STAGE29_PROTOCOL_ID
                and source.get("resolved_commit") == EXPECTED_STAGE29_SOURCE_COMMIT
                and source.get("confirmation_eligible", False)
            ):
                valid.append((decision_path.parent, decision, source))
        if not valid:
            raise FileNotFoundError(
                "No exact source-bound Stage 29 result was found in MyDrive. "
                "Keep the complete Stage 29 Drive directory."
            )
        valid.sort(key=lambda row: str(row[0]))
        STAGE29_RUN_DIR, STAGE29_DECISION, STAGE29_SOURCE = valid[-1]
        STAGE29_CERTIFICATE = {
            "validated_before_stage30_model_activations": True,
            "run_dir": str(STAGE29_RUN_DIR),
            "decision_status": STAGE29_DECISION["status"],
            "resolved_commit": STAGE29_SOURCE["resolved_commit"],
            "decision_sha256": EXPECTED_STAGE29_DECISION_SHA256,
            "source_identity_sha256": EXPECTED_STAGE29_SOURCE_SHA256,
            "stage29_self_gate_passed": True,
            "stage29_grounded_gate_passed": False,
            "stage30_states_reused_from_stage29": False,
        }
        write_json(OUT / "stage29_upstream_certificate.json", STAGE29_CERTIFICATE)
        STAGE29_UPSTREAM_BOUND = True
        memory_report("stage30_design_and_upstream_bound")
    except Exception:
        record_failure("stage30_design_or_upstream_binding")
'''


physical_truth = r'''# Screen contacts without model access, then regenerate full truth only for selected states.
PROVENANCE_COUNTS = {
    "screened_states": 0,
    "truth_generated": 0,
    "baseline_generated": 0,
    "intervention_generated": 0,
    "cache_hits": 0,
}


def record_task(record):
    return {"goal": np.asarray(record["goal"], dtype=np.float64).tolist()}


def dynamic_state_from_environment(environment):
    return np.asarray([
        *environment.agent.position, *environment.block.position,
        float(environment.block.angle), *environment.agent.velocity,
        *environment.block.velocity, float(environment.block.angular_velocity),
    ], dtype=np.float64)


def reset_dynamic_environment(dynamic_state, task, seed):
    state = np.asarray(dynamic_state, dtype=np.float64)
    if state.shape != (10,):
        raise ValueError(f"expected ten-dimensional dynamic state, found {state.shape}")
    environment = make_environment(REPO, ENVIRONMENT)
    environment.seed(int(seed))
    environment.reset_to_state = np.asarray([*state[:5], 0.0, 0.0], dtype=np.float64)
    environment.reset()
    environment.agent.position = tuple(state[:2])
    environment.block.angle = float(state[4])
    environment.block.position = tuple(state[2:4])
    environment.agent.velocity = tuple(state[5:7])
    environment.block.velocity = tuple(state[7:9])
    environment.block.angular_velocity = float(state[9])
    environment.set_task_goal(np.asarray(task["goal"], dtype=np.float64))
    restored = dynamic_state_from_environment(environment)
    if not np.allclose(restored, state, atol=1e-12, rtol=0):
        raise RuntimeError(
            f"full dynamic restoration drifted: {np.max(np.abs(restored - state))}"
        )
    observation = {
        "visual": np.asarray(environment.render("rgb_array")).copy(),
        "proprio": np.asarray(
            [*environment.agent.position, *environment.agent.velocity],
            dtype=np.float32,
        ),
    }
    return environment, observation


def rollout_dynamic_branch(record, actions, retain_visual):
    environment, initial = reset_dynamic_environment(
        record["state"], record_task(record), record["evaluation_seed"]
    )
    cumulative = 0
    endpoint_observation = None
    endpoint_state = None
    try:
        for step, action in enumerate(actions, start=1):
            observation, _, _, info = environment.step(action)
            cumulative += int(info.get("n_contacts", 0))
            if step == ACTION_STEPS:
                endpoint_observation = {
                    "visual": (
                        np.asarray(observation["visual"]).copy()
                        if retain_visual else None
                    ),
                    "proprio": np.asarray(observation["proprio"]).copy(),
                }
                endpoint_state = dynamic_state_from_environment(environment)
    finally:
        environment.close()
    if endpoint_observation is None or endpoint_state is None:
        raise RuntimeError("dynamic rollout missed the primary horizon")
    return initial, endpoint_observation, endpoint_state, cumulative


def screen_pool(records):
    rows = []
    started = time.perf_counter()
    for index, record in enumerate(records):
        contacts = []
        actions = candidate_action_bank(record)
        for branch in actions:
            _, _, _, count = rollout_dynamic_branch(record, branch, retain_visual=False)
            contacts.append(count)
        contact_array = np.asarray(contacts, dtype=np.int64).reshape(
            MAGNITUDE_COUNT, SCHEDULE_COUNT
        )
        rows.append({
            "record_id": int(record["record_id"]),
            "trajectory_id": int(record["trajectory_id"]),
            "approach_distance": float(record["approach_distance"]),
            "regime": contact_regime(contact_array),
            "contact_fraction": float(np.mean(contact_array > 0)),
            "total_contacts": int(np.sum(contact_array)),
        })
        PROVENANCE_COUNTS["screened_states"] += 1
        write_json(OUT / "physical_screen_progress.json", {
            "completed": index + 1,
            "total": len(records),
            "last_record_id": int(record["record_id"]),
        })
    TIMINGS["physical_screen_seconds"] = time.perf_counter() - started
    write_csv(EVIDENCE_DIR / "physical_screen_rows.csv", rows)
    return rows


def select_records(records, screen_rows):
    lookup = {int(row["record_id"]): row for row in screen_rows}
    selected = []
    for label in STRATUM_LABELS:
        candidates = [
            row for row in records
            if lookup[int(row["record_id"])]["regime"] == label
        ]
        if len(candidates) < ACTIVE_TARGET_PER_STRATUM:
            raise RuntimeError(
                f"fresh pool has {len(candidates)} {label} states; "
                f"requires {ACTIVE_TARGET_PER_STRATUM}"
            )
        selected.extend(candidates[:ACTIVE_TARGET_PER_STRATUM])
    for record in selected:
        record["regime"] = lookup[int(record["record_id"])]["regime"]
    return selected


def truth_path(record_id):
    return TRUTH_DIR / f"state_{int(record_id):06d}.npz"


def generate_selected_truth(records):
    started = time.perf_counter()
    for index, record in enumerate(records):
        destination = truth_path(record["record_id"])
        if destination.exists():
            PROVENANCE_COUNTS["cache_hits"] += 1
            raise RuntimeError(f"fresh-run truth shard already exists: {destination}")
        action_bank = candidate_action_bank(record)
        initials, initial_proprios = [], []
        endpoint_visuals, endpoint_states, contacts = [], [], []
        for branch in action_bank:
            initial, endpoint, state, count = rollout_dynamic_branch(
                record, branch, retain_visual=True
            )
            initials.append(initial["visual"])
            initial_proprios.append(initial["proprio"])
            endpoint_visuals.append(endpoint["visual"])
            endpoint_states.append(state)
            contacts.append(count)
        if not all(np.array_equal(initials[0], value) for value in initials[1:]):
            raise RuntimeError("initial visual drift across exact branches")
        if not all(
            np.array_equal(initial_proprios[0], value)
            for value in initial_proprios[1:]
        ):
            raise RuntimeError("initial proprio drift across exact branches")
        observed_regime = contact_regime(
            np.asarray(contacts).reshape(MAGNITUDE_COUNT, SCHEDULE_COUNT)
        )
        if observed_regime != record["regime"]:
            raise RuntimeError("contact regime changed between screening and truth pass")
        atomic_npz(
            destination,
            record_id=np.asarray(record["record_id"], dtype=np.int64),
            trajectory_id=np.asarray(record["trajectory_id"], dtype=np.int64),
            state=np.asarray(record["state"], dtype=np.float64),
            goal=np.asarray(record["goal"], dtype=np.float64),
            regime=np.asarray(record["regime"]),
            initial_visual=np.asarray(initials[0], dtype=np.uint8),
            initial_proprio=np.asarray(initial_proprios[0], dtype=np.float32),
            selected_actions=action_bank.astype(np.float32),
            endpoint_visuals=np.asarray(endpoint_visuals, dtype=np.uint8),
            endpoint_states=np.asarray(endpoint_states, dtype=np.float64),
            interaction_counts=np.asarray(contacts, dtype=np.int32),
        )
        PROVENANCE_COUNTS["truth_generated"] += 1
        write_json(OUT / "selected_truth_progress.json", {
            "completed": index + 1,
            "total": len(records),
            "last_record_id": int(record["record_id"]),
        })
    TIMINGS["selected_truth_seconds"] = time.perf_counter() - started


if not PIPELINE_FAILED:
    try:
        if not STAGE18_ARTIFACT_VALIDATED or not STAGE29_UPSTREAM_BOUND:
            raise RuntimeError("frozen Stage 18 and exact Stage 29 must be bound first")
        REPO = configure_repo()
        SCREEN_ROWS = screen_pool(POOL_SPECS)
        ALL_EVALUATION_RECORDS = select_records(POOL_SPECS, SCREEN_ROWS)
        if len(ALL_EVALUATION_RECORDS) != ACTIVE_EVALUATION_TARGET:
            raise RuntimeError("fresh stratified selection returned the wrong count")
        generate_selected_truth(ALL_EVALUATION_RECORDS)
        SELECTION_CERTIFICATE = {
            "created_before_model_loading": True,
            "selected_record_ids": [
                int(row["record_id"]) for row in ALL_EVALUATION_RECORDS
            ],
            "selected_counts": {
                label: sum(row["regime"] == label for row in ALL_EVALUATION_RECORDS)
                for label in STRATUM_LABELS
            },
            "selection_uses_contact_regime_only": True,
            "effect_magnitude_used_for_selection": False,
            "model_outputs_used_for_selection": False,
            "stage29_states_reused": False,
            "screen_rows_sha256": sha256_file(
                EVIDENCE_DIR / "physical_screen_rows.csv"
            ),
        }
        write_json(DESIGN_DIR / "physical_selection_freeze.json", SELECTION_CERTIFICATE)
        memory_report("fresh_physical_truth_selected")
    except Exception:
        record_failure("fresh_physical_screen_or_truth")
'''


model_initialization = r'''# Load the frozen JEPA-WM only after physical selection; verify native-token and hook contracts.


def state_model_inputs(record_id, horizon=PRIMARY_HORIZON):
    with np.load(truth_path(record_id)) as truth:
        initial_visual = truth["initial_visual"]
        initial_proprio = truth["initial_proprio"]
        selected_actions = truth["selected_actions"]
    with torch.inference_mode():
        initial = MODEL.encode(to_model_observation(initial_visual, initial_proprio))
    initial = {name: value.detach() for name, value in initial.items()}
    actions = model_action_tensor(PREPROCESSOR, selected_actions, horizon)
    return initial, actions


def encode_true_tokens(record_id):
    with np.load(truth_path(record_id)) as payload:
        visual = payload["endpoint_visuals"][:, None]
        states = payload["endpoint_states"].astype(np.float32)
    proprio = np.concatenate([states[:, :2], states[:, 5:7]], axis=1)[:, None]
    with torch.inference_mode():
        encoded = MODEL.encode(to_model_observation(visual, proprio))
    tokens = encoded["visual"][:, :, 0]
    tokens = tokens.reshape(ACTIONS_PER_STATE, 256, tokens.shape[-1])
    if tokens.shape != (ACTIONS_PER_STATE, 256, 384):
        raise RuntimeError(f"unexpected true target-token shape {tuple(tokens.shape)}")
    return tokens.detach()


def hook_identity_test(record_id):
    initial, actions = state_model_inputs(record_id)
    with torch.inference_mode():
        baseline, _, _ = forward_with_carriers(
            initial, actions, PRIMARY_HORIZON, capture_blocks=[FIXED_BLOCK]
        )
        patched, _, _ = forward_with_carriers(
            initial, actions, PRIMARY_HORIZON, capture_blocks=[FIXED_BLOCK],
            intervention={
                "block": FIXED_BLOCK,
                "delta": torch.zeros(
                    ACTIONS_PER_STATE, 256, EXPECTED_CARRIER_CHANNELS,
                    device="cuda", dtype=torch.float32,
                ),
            },
        )
    error = float(torch.max(torch.abs(patched - baseline)).cpu())
    result = {
        "record_id": int(record_id),
        "max_abs_error": error,
        "passed": error <= MAX_ZERO_EDIT_ERROR,
    }
    write_json(OUT / "hook_identity_test.json", result)
    if not result["passed"]:
        raise RuntimeError(f"zero hook identity failed: {result}")
    return result


MODEL_READY = False
if not PIPELINE_FAILED:
    try:
        MODEL, PREPROCESSOR, PREDICTOR, PREDICTOR_BLOCK_MODULES = load_frozen_model()
        if len(PREDICTOR_BLOCK_MODULES) != 6:
            raise RuntimeError("predictor block count changed")
        if not all(
            isinstance(module, torch.nn.Module)
            and callable(getattr(module, "register_forward_hook", None))
            for module in PREDICTOR_BLOCK_MODULES
        ):
            raise RuntimeError("predictor block hook contract changed")
        probe = ALL_EVALUATION_RECORDS[0]
        probe_target = encode_true_tokens(probe["record_id"])
        if probe_target.requires_grad:
            raise RuntimeError("true target tokens unexpectedly require gradients")
        del probe_target
        HOOK_IDENTITY = hook_identity_test(probe["record_id"])
        initial, actions = state_model_inputs(probe["record_id"])
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            _, _, _ = forward_with_carriers(
                initial, actions, PRIMARY_HORIZON, capture_blocks=[FIXED_BLOCK]
            )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        estimated_batches = len(ALL_EVALUATION_RECORDS) * (
            ACTIVE_INTERVENTION_FORWARDS_PER_RECORD + 1
        )
        FORWARD_BENCHMARK = {
            "seconds_per_24_branch_predictor_batch": seconds,
            "predictor_batches_per_record": ACTIVE_INTERVENTION_FORWARDS_PER_RECORD + 1,
            "target_encoder_batches_per_record": 1,
            "evaluation_records": len(ALL_EVALUATION_RECORDS),
            "estimated_predictor_minutes": seconds * estimated_batches / 60.0,
            "physical_screen_states": len(POOL_SPECS),
            "warning_threshold_minutes": MAX_ESTIMATED_TOTAL_MINUTES,
        }
        write_json(OUT / "forward_benchmark.json", FORWARD_BENCHMARK)
        if (
            FORWARD_BENCHMARK["estimated_predictor_minutes"]
            > MAX_ESTIMATED_TOTAL_MINUTES
            and not CONTINUE_AFTER_BENCHMARK
        ):
            raise RuntimeError("measured predictor estimate exceeds the credit guard")
        del initial, actions
        MODEL_READY = True
        write_json(OUT / "evaluation_open_certificate.json", {
            "opened": True,
            "source_identity": SOURCE_IDENTITY,
            "stage18_artifact_certificate_sha256": sha256_file(
                OUT / "stage18_artifact_certificate.json"
            ),
            "stage29_upstream_certificate_sha256": sha256_file(
                OUT / "stage29_upstream_certificate.json"
            ),
            "physical_selection_freeze_sha256": sha256_file(
                DESIGN_DIR / "physical_selection_freeze.json"
            ),
            "learned_decoder_loaded": False,
            "new_reader_fit": False,
            "subspace_refit": False,
        })
        memory_report("stage30_model_contracts_verified")
    except Exception:
        record_failure("stage30_model_initialization")
'''


grounded_planning_evaluation = r'''# Measure heldout-schedule closure, native planning, and causal ablation value.


def whiten_carrier(values, subspaces):
    return transform_primal_channels(
        np.asarray(values, dtype=np.float64),
        subspaces["channel_inverse_square_root"],
    )


def native_edit(values, subspaces):
    return inverse_transform_primal_channels(
        np.asarray(values, dtype=np.float64), subspaces["channel_square_root"]
    )


def matched_norm(value, reference):
    array = np.asarray(value, dtype=np.float64)
    target = float(np.linalg.norm(reference))
    norm = float(np.linalg.norm(array))
    if norm <= 1e-12 or target <= 1e-12:
        raise RuntimeError("cannot norm-match a degenerate intervention")
    return array * (target / norm)


def intervention_specs(carrier, subspaces):
    white = whiten_carrier(carrier, subspaces)
    primary_basis = subspaces["primary_basis"][:, :PRIMARY_RANK]
    shuffled_basis = subspaces["shuffled_basis"][:, :PRIMARY_RANK]
    primary_swap = area_swap_delta(
        white, MAGNITUDE_COUNT, basis=primary_basis, dose=1.0
    )
    primary_ablation = area_ablation_delta(
        white, MAGNITUDE_COUNT, primary_basis, dose=1.0
    )
    specs = [
        {"condition": "primary_r128_swap", "family": "primary", "mode": "swap", "delta_white": primary_swap},
        {"condition": "shuffled_r128_swap", "family": "matched_shuffled_control", "mode": "swap", "delta_white": matched_norm(area_swap_delta(white, MAGNITUDE_COUNT, basis=shuffled_basis, dose=1.0), primary_swap)},
    ]
    for draw in range(ACTIVE_CAUSAL_RANDOM_DRAWS):
        random_basis = subspaces[f"random_basis_{draw:02d}"][:, :PRIMARY_RANK]
        specs.append({
            "condition": f"random_r128_{draw:02d}_swap",
            "family": "empirical_span_random_control",
            "mode": "swap",
            "delta_white": matched_norm(
                area_swap_delta(white, MAGNITUDE_COUNT, basis=random_basis, dose=1.0),
                primary_swap,
            ),
        })
    specs.append({
        "condition": "full_activation_swap",
        "family": "positive_control_only",
        "mode": "swap",
        "delta_white": area_swap_delta(white, MAGNITUDE_COUNT, basis=None, dose=1.0),
    })
    specs.extend([
        {"condition": "primary_r128_ablation", "family": "primary", "mode": "ablation", "delta_white": primary_ablation},
        {"condition": "shuffled_r128_ablation", "family": "matched_shuffled_control", "mode": "ablation", "delta_white": matched_norm(area_ablation_delta(white, MAGNITUDE_COUNT, shuffled_basis, dose=1.0), primary_ablation)},
    ])
    for draw in range(ACTIVE_CAUSAL_RANDOM_DRAWS):
        random_basis = subspaces[f"random_basis_{draw:02d}"][:, :PRIMARY_RANK]
        specs.append({
            "condition": f"random_r128_{draw:02d}_ablation",
            "family": "empirical_span_random_control",
            "mode": "ablation",
            "delta_white": matched_norm(
                area_ablation_delta(white, MAGNITUDE_COUNT, random_basis, dose=1.0),
                primary_ablation,
            ),
        })
    if len(specs) != ACTIVE_INTERVENTION_FORWARDS_PER_RECORD:
        raise RuntimeError(f"unexpected active intervention count: {len(specs)}")
    for spec in specs:
        spec["edit_norm"] = float(np.linalg.norm(spec["delta_white"]))
        spec["primary_swap_norm"] = float(np.linalg.norm(primary_swap))
        spec["primary_ablation_norm"] = float(np.linalg.norm(primary_ablation))
    return specs


def baseline_alignment_rows(record, predicted, target):
    prediction = predicted.reshape(
        MAGNITUDE_COUNT, SCHEDULE_COUNT, *predicted.shape[1:]
    )
    truth = target.reshape(
        MAGNITUDE_COUNT, SCHEDULE_COUNT, *target.shape[1:]
    )
    rows = []
    for magnitude_index in range(MAGNITUDE_COUNT):
        total = vector_alignment(prediction[magnitude_index], truth[magnitude_index])
        centered_prediction = prediction[magnitude_index] - prediction[magnitude_index].mean(axis=0, keepdims=True)
        centered_truth = truth[magnitude_index] - truth[magnitude_index].mean(axis=0, keepdims=True)
        centered = vector_alignment(centered_prediction, centered_truth)
        rows.append({
            "record_id": int(record["record_id"]),
            "trajectory_id": int(record["trajectory_id"]),
            "regime": record["regime"],
            "magnitude_index": int(magnitude_index),
            "magnitude": float(SELECTED_MAGNITUDES[magnitude_index]),
            "target_centered_mean_square": float(np.mean(centered_truth**2)),
            **{f"native_total_{key}": value for key, value in total.items()},
            **{f"native_centered_{key}": value for key, value in centered.items()},
        })
    return rows


def attach_planning_identity(rows, record, condition, family):
    return [
        {
            "record_id": int(record["record_id"]),
            "trajectory_id": int(record["trajectory_id"]),
            "regime": record["regime"],
            "condition": condition,
            "family": family,
            "magnitude": float(SELECTED_MAGNITUDES[row["magnitude_index"]]),
            **row,
        }
        for row in rows
    ]


def run_record(record, subspaces):
    record_id = int(record["record_id"])
    initial, actions = state_model_inputs(record_id)
    target_tensor = encode_true_tokens(record_id)
    with np.load(truth_path(record_id)) as payload:
        endpoint_states = payload["endpoint_states"].astype(np.float64)
    with torch.inference_mode():
        predicted_tensor, _, captures = forward_with_carriers(
            initial, actions, PRIMARY_HORIZON, capture_blocks=[FIXED_BLOCK]
        )
    predicted = predicted_tensor.detach().float().cpu().numpy()
    target = target_tensor.detach().float().cpu().numpy()
    carrier = layer_tokens_full(
        captures[FIXED_BLOCK]
    ).detach().float().cpu().numpy()
    alignment_rows = baseline_alignment_rows(record, predicted, target)
    planning_rows = attach_planning_identity(
        terminal_planning_rows(
            predicted, target, endpoint_states,
            MAGNITUDE_COUNT, SCHEDULE_COUNT,
            goal_schedules=PLANNING_GOAL_SCHEDULES,
        ),
        record, "baseline", "native_unedited",
    )
    closure_rows = []
    specs = intervention_specs(carrier, subspaces)
    for spec in specs:
        delta_native = native_edit(spec["delta_white"], subspaces)
        delta_tensor = torch.as_tensor(
            delta_native, device="cuda", dtype=torch.float32
        )
        with torch.inference_mode():
            patched_tensor, _, _ = forward_with_carriers(
                initial, actions, PRIMARY_HORIZON,
                capture_blocks=[FIXED_BLOCK],
                intervention={"block": FIXED_BLOCK, "delta": delta_tensor},
            )
        patched = patched_tensor.detach().float().cpu().numpy()
        for row in diagnostic_closure_rows(
            predicted, patched, target,
            MAGNITUDE_COUNT, SCHEDULE_COUNT,
            diagnostic_schedules=DIAGNOSTIC_SCHEDULES,
            mode=spec["mode"],
        ):
            closure_rows.append({
                "record_id": record_id,
                "trajectory_id": int(record["trajectory_id"]),
                "regime": record["regime"],
                "condition": spec["condition"],
                "family": spec["family"],
                "rank": -1 if spec["condition"] == "full_activation_swap" else PRIMARY_RANK,
                "magnitude": float(SELECTED_MAGNITUDES[row["magnitude_index"]]),
                "carrier_edit_whitened_norm": spec["edit_norm"],
                "primary_swap_whitened_norm": spec["primary_swap_norm"],
                "primary_ablation_whitened_norm": spec["primary_ablation_norm"],
                **row,
            })
        planning_rows.extend(attach_planning_identity(
            terminal_planning_rows(
                patched, target, endpoint_states,
                MAGNITUDE_COUNT, SCHEDULE_COUNT,
                goal_schedules=PLANNING_GOAL_SCHEDULES,
            ),
            record, spec["condition"], spec["family"],
        ))
        del patched_tensor, patched, delta_tensor
    PROVENANCE_COUNTS["baseline_generated"] += 1
    PROVENANCE_COUNTS["intervention_generated"] += 1
    del initial, actions, target_tensor, predicted_tensor, captures, carrier
    gc.collect()
    torch.cuda.empty_cache()
    return alignment_rows, closure_rows, planning_rows


BASELINE_ALIGNMENT_ROWS = []
CLOSURE_ROWS = []
PLANNING_ROWS = []
if not PIPELINE_FAILED and MODEL_READY:
    try:
        started = time.perf_counter()
        for index, record in enumerate(ALL_EVALUATION_RECORDS):
            alignment, closure, planning = run_record(record, FROZEN_SUBSPACES)
            BASELINE_ALIGNMENT_ROWS.extend(alignment)
            CLOSURE_ROWS.extend(closure)
            PLANNING_ROWS.extend(planning)
            write_json(OUT / "grounded_planning_progress.json", {
                "completed": index + 1,
                "total": len(ALL_EVALUATION_RECORDS),
                "last_record_id": int(record["record_id"]),
            })
        TIMINGS["grounded_planning_evaluation_seconds"] = time.perf_counter() - started
        write_csv(EVIDENCE_DIR / "baseline_native_alignment_rows.csv", BASELINE_ALIGNMENT_ROWS)
        write_csv(EVIDENCE_DIR / "diagnostic_closure_rows.csv", CLOSURE_ROWS)
        write_csv(EVIDENCE_DIR / "terminal_planning_rows.csv", PLANNING_ROWS)
        memory_report("stage30_grounded_planning_complete")
    except Exception:
        record_failure("stage30_grounded_planning_evaluation")
'''


decision_and_plots = r'''# Apply frozen replication, planning-reliability, causal-value, and null gates.


def bootstrap_state_mean(values, record_ids, label):
    draws = clustered_bootstrap_mean(
        np.asarray(values, dtype=np.float64),
        np.asarray(record_ids, dtype=np.int64),
        ACTIVE_BOOTSTRAP_DRAWS,
        stable_seed(BOOTSTRAP_SEED, label) % (2**31 - 1),
    )
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def row_map(rows, keys):
    result = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        if key in result:
            raise RuntimeError(f"duplicate row key: {key}")
        result[key] = row
    return result


def replicated_gap_gate():
    primary = [
        row for row in CLOSURE_ROWS
        if row["condition"] == "primary_r128_swap"
        and row["regime"] == "persistent_contact"
        and np.isfinite(row["grounded_coefficient"])
    ]
    gaps = np.asarray([
        row["self_coefficient"] - row["grounded_coefficient"]
        for row in primary
    ], dtype=np.float64)
    record_ids = np.asarray([row["record_id"] for row in primary], dtype=np.int64)
    ci = bootstrap_state_mean(gaps, record_ids, "replicated_self_ground_gap")
    self_mean = float(np.mean([row["self_coefficient"] for row in primary]))
    ground_mean = float(np.mean([row["grounded_coefficient"] for row in primary]))
    return {
        "persistent_states": len(np.unique(record_ids)),
        "persistent_state_magnitude_rows": len(primary),
        "mean_self_coefficient": self_mean,
        "mean_grounded_coefficient": ground_mean,
        "mean_self_minus_grounded_coefficient": float(np.mean(gaps)),
        "self_minus_grounded_ci95": ci,
        "minimum_self_coefficient": MIN_SELF_CLOSURE_COEFFICIENT,
        "minimum_gap": MIN_SELF_GROUND_COEFFICIENT_GAP,
        "passed": bool(
            self_mean >= MIN_SELF_CLOSURE_COEFFICIENT
            and np.mean(gaps) >= MIN_SELF_GROUND_COEFFICIENT_GAP
            and (ci[0] > 0 if RUN_MODE == "pilot" else True)
        ),
    }


def joined_planning_dataset(outcome_kind):
    alignment = row_map(
        BASELINE_ALIGNMENT_ROWS, ["record_id", "magnitude_index"]
    )
    closure_condition = (
        "primary_r128_swap"
        if outcome_kind == "native_regret"
        else "primary_r128_ablation"
    )
    closure = row_map(
        [row for row in CLOSURE_ROWS if row["condition"] == closure_condition],
        ["record_id", "magnitude_index"],
    )
    planning = row_map(
        PLANNING_ROWS,
        ["record_id", "magnitude_index", "goal_schedule", "condition"],
    )
    task_rows = []
    for key, base_row in planning.items():
        record_id, magnitude_index, goal_schedule, condition = key
        if condition != "baseline":
            continue
        if base_row["regime"] not in {"persistent_contact", "boundary_switching"}:
            continue
        if base_row["true_cost_spread"] < MIN_PLANNING_TRUE_COST_SPREAD:
            continue
        closure_row = closure[(record_id, magnitude_index)]
        alignment_row = alignment[(record_id, magnitude_index)]
        if outcome_kind == "native_regret":
            outcome = float(base_row["normalized_regret"])
        else:
            ablated = planning[
                (record_id, magnitude_index, goal_schedule, "primary_r128_ablation")
            ]
            outcome = float(
                ablated["normalized_regret"] - base_row["normalized_regret"]
            )
        candidate = {
            "record_id": int(record_id),
            "regime": base_row["regime"],
            "magnitude_index": int(magnitude_index),
            "goal_schedule": int(goal_schedule),
            "outcome": outcome,
            "native_total_normalized_rmse": float(
                alignment_row["native_total_normalized_rmse"]
            ),
            "self_coefficient": float(closure_row["self_coefficient"]),
            "self_cosine": float(closure_row["self_cosine"]),
            "grounded_coefficient": float(closure_row["grounded_coefficient"]),
            "grounded_cosine": float(closure_row["grounded_cosine"]),
        }
        if all(np.isfinite(value) for name, value in candidate.items() if name not in {"regime"}):
            task_rows.append(candidate)
    grouped = {}
    for row in task_rows:
        key = (row["record_id"], row["magnitude_index"])
        grouped.setdefault(key, []).append(row)
    rows = []
    for key, values in sorted(grouped.items()):
        first = values[0]
        if len(values) != len(PLANNING_GOAL_SCHEDULES):
            raise RuntimeError(f"incomplete planning-goal panel for {key}")
        rows.append({
            **{name: value for name, value in first.items() if name not in {"goal_schedule", "outcome"}},
            "goal_tasks": len(values),
            "outcome": float(np.mean([value["outcome"] for value in values])),
        })
    return rows


def predictive_gate(outcome_kind):
    rows = joined_planning_dataset(outcome_kind)
    groups = np.asarray([row["record_id"] for row in rows], dtype=np.int64)
    base_features = np.asarray([
        [
            row["magnitude_index"] / max(MAGNITUDE_COUNT - 1, 1),
            float(row["regime"] == "boundary_switching"),
            row["native_total_normalized_rmse"],
            row["self_coefficient"],
            row["self_cosine"],
        ]
        for row in rows
    ], dtype=np.float64)
    grounded_features = np.asarray([
        [row["grounded_coefficient"], row["grounded_cosine"]]
        for row in rows
    ], dtype=np.float64)
    outcome = np.asarray([row["outcome"] for row in rows], dtype=np.float64)
    result = cross_fitted_incremental_value(
        outcome,
        groups,
        base_features,
        grounded_features,
        folds=ACTIVE_CROSSFIT_FOLDS,
        seed=stable_seed(CROSSFIT_SEED, outcome_kind),
        ridge=1e-6,
    )
    group_values = np.asarray([
        row["mse_improvement"] for row in result["group_rows"]
    ], dtype=np.float64)
    group_ids = np.asarray([
        int(row["group"]) for row in result["group_rows"]
    ], dtype=np.int64)
    ci = bootstrap_state_mean(
        group_values, group_ids, f"{outcome_kind}_oof_mse_improvement"
    )
    evidence_rows = []
    for index, row in enumerate(rows):
        evidence_rows.append({
            **row,
            "fold": int(result["fold_id"][index]),
            "base_prediction": float(result["base_prediction"][index]),
            "grounded_prediction": float(result["grounded_prediction"][index]),
            "base_squared_error": float(
                (row["outcome"] - result["base_prediction"][index]) ** 2
            ),
            "grounded_squared_error": float(
                (row["outcome"] - result["grounded_prediction"][index]) ** 2
            ),
        })
    write_csv(
        EVIDENCE_DIR / f"{outcome_kind}_crossfit_rows.csv", evidence_rows
    )
    eligible_states = len(np.unique(groups))
    required_states = min(MIN_ELIGIBLE_CONTACT_STATES, 2 * ACTIVE_TARGET_PER_STRATUM)
    return {
        "outcome": outcome_kind,
        "eligible_states": eligible_states,
        "task_rows": len(rows),
        "crossfit_folds": ACTIVE_CROSSFIT_FOLDS,
        "base_features": [
            "magnitude", "boundary_indicator", "native_total_nrmse",
            "self_coefficient", "self_cosine",
        ],
        "added_grounded_features": [
            "grounded_coefficient", "grounded_cosine",
        ],
        "base_oof_mse": result["base_mse"],
        "grounded_oof_mse": result["grounded_mse"],
        "relative_oof_mse_improvement": result["relative_mse_improvement"],
        "base_oof_r_squared": result["base_oof_r_squared"],
        "grounded_oof_r_squared": result["grounded_oof_r_squared"],
        "state_mean_mse_improvement_ci95": ci,
        "minimum_relative_improvement": MIN_OOF_RELATIVE_MSE_IMPROVEMENT,
        "required_states": required_states,
        "passed": bool(
            eligible_states >= required_states
            and result["relative_mse_improvement"]
            >= MIN_OOF_RELATIVE_MSE_IMPROVEMENT
            and (ci[0] > 0 if RUN_MODE == "pilot" else True)
        ),
    }


def ablation_control_gate():
    planning = row_map(
        PLANNING_ROWS,
        ["record_id", "magnitude_index", "goal_schedule", "condition"],
    )
    by_state = {}
    for key, baseline in planning.items():
        record_id, magnitude_index, goal_schedule, condition = key
        if condition != "baseline" or baseline["regime"] not in {
            "persistent_contact", "boundary_switching"
        }:
            continue
        if baseline["true_cost_spread"] < MIN_PLANNING_TRUE_COST_SPREAD:
            continue
        primary = planning[
            (record_id, magnitude_index, goal_schedule, "primary_r128_ablation")
        ]
        control_conditions = [
            candidate[3] for candidate in planning
            if candidate[:3] == (record_id, magnitude_index, goal_schedule)
            and (
                candidate[3] == "shuffled_r128_ablation"
                or candidate[3].startswith("random_r128_")
                and candidate[3].endswith("_ablation")
            )
        ]
        control_changes = [
            planning[(record_id, magnitude_index, goal_schedule, name)]["normalized_regret"]
            - baseline["normalized_regret"]
            for name in control_conditions
        ]
        advantage = (
            primary["normalized_regret"] - baseline["normalized_regret"]
            - float(np.median(control_changes))
        )
        by_state.setdefault(int(record_id), []).append(float(advantage))
    state_rows = [
        {"record_id": record_id, "primary_minus_median_control_regret_change": float(np.mean(values))}
        for record_id, values in sorted(by_state.items())
    ]
    write_csv(EVIDENCE_DIR / "ablation_control_state_rows.csv", state_rows)
    values = np.asarray([
        row["primary_minus_median_control_regret_change"] for row in state_rows
    ], dtype=np.float64)
    ids = np.asarray([row["record_id"] for row in state_rows], dtype=np.int64)
    ci = bootstrap_state_mean(values, ids, "ablation_control_advantage")
    return {
        "eligible_states": len(state_rows),
        "mean_primary_minus_median_control_regret_change": float(np.mean(values)),
        "ci95": ci,
        "minimum_advantage": MIN_ABLATION_CONTROL_ADVANTAGE,
        "passed": bool(
            np.mean(values) > MIN_ABLATION_CONTROL_ADVANTAGE
            and (ci[0] > 0 if RUN_MODE == "pilot" else True)
        ),
    }


def free_null_summary():
    baseline = [
        row for row in PLANNING_ROWS
        if row["condition"] == "baseline" and row["regime"] == "free"
    ]
    closure = [
        row for row in CLOSURE_ROWS
        if row["condition"] == "primary_r128_swap" and row["regime"] == "free"
    ]
    exact_null = [
        row for row in baseline
        if row["true_cost_spread"] <= MAX_FREE_TRUE_COST_SPREAD
    ]
    return {
        "free_states": len({row["record_id"] for row in baseline}),
        "free_planning_tasks": len(baseline),
        "exact_physical_null_tasks": len(exact_null),
        "median_true_cost_spread": float(np.median([
            row["true_cost_spread"] for row in baseline
        ])),
        "median_model_cost_spread_on_exact_null": (
            float(np.median([row["model_cost_spread"] for row in exact_null]))
            if exact_null else None
        ),
        "median_model_extreme_margin_on_exact_null": (
            float(np.median([
                row["model_extreme_preference_margin"] for row in exact_null
            ]))
            if exact_null else None
        ),
        "median_primary_intervention_energy": float(np.median([
            row["effect_energy"] for row in closure
        ])),
        "grounded_coefficients_defined": int(sum(
            np.isfinite(row["grounded_coefficient"]) for row in closure
        )),
    }


def fresh_run_certificate():
    expected = {
        "screened_states": len(POOL_SPECS),
        "truth_generated": len(ALL_EVALUATION_RECORDS),
        "baseline_generated": len(ALL_EVALUATION_RECORDS),
        "intervention_generated": len(ALL_EVALUATION_RECORDS),
        "cache_hits": 0,
    }
    passed = bool(
        not OUT_PREEXISTED
        and PROVENANCE_COUNTS == expected
        and SOURCE_IDENTITY.get("confirmation_eligible", False)
        and STAGE18_ARTIFACT_VALIDATED
        and STAGE29_UPSTREAM_BOUND
    )
    payload = {
        "out_preexisted": bool(OUT_PREEXISTED),
        "observed_counts": dict(PROVENANCE_COUNTS),
        "expected_counts": expected,
        "source_execution_verified": bool(
            SOURCE_IDENTITY.get("confirmation_eligible", False)
        ),
        "stage18_artifact_validated": bool(STAGE18_ARTIFACT_VALIDATED),
        "exact_stage29_bound": bool(STAGE29_UPSTREAM_BOUND),
        "passed": passed,
    }
    write_json(OUT / "fresh_run_certificate.json", payload)
    return payload


def finite_xy(rows, x_name, y_name):
    pairs = [
        (float(row[x_name]), float(row[y_name])) for row in rows
        if np.isfinite(row[x_name]) and np.isfinite(row[y_name])
    ]
    if not pairs:
        return np.asarray([]), np.asarray([])
    return np.asarray([value[0] for value in pairs]), np.asarray([value[1] for value in pairs])


def make_plots(replicated, reliability, causal_value, ablation_control):
    figure, axes = plt.subplots(1, 4, figsize=(20, 4.6))
    primary = [
        row for row in CLOSURE_ROWS
        if row["condition"] == "primary_r128_swap"
        and row["regime"] != "free"
    ]
    x, y = finite_xy(primary, "self_coefficient", "grounded_coefficient")
    if len(x):
        axes[0].scatter(x, y, alpha=0.45, color="#4c78a8")
    else:
        axes[0].text(0.5, 0.5, "no finite rows", ha="center", va="center")
    axes[0].axline((0, 0), slope=1, color="black", linestyle="--", linewidth=0.8)
    axes[0].set(xlabel="self coefficient", ylabel="grounded coefficient", title="Fresh causal-grounding gap")

    regret_rows = joined_planning_dataset("native_regret")
    x, y = finite_xy(regret_rows, "grounded_coefficient", "outcome")
    if len(x):
        axes[1].scatter(x, y, alpha=0.35, color="#f58518")
    else:
        axes[1].text(0.5, 0.5, "no eligible rows", ha="center", va="center")
    axes[1].set(xlabel="grounded closure coefficient", ylabel="native normalized regret", title="Closure and planning")

    labels = ["native regret", "ablation value"]
    base = [reliability["base_oof_mse"], causal_value["base_oof_mse"]]
    grounded = [reliability["grounded_oof_mse"], causal_value["grounded_oof_mse"]]
    positions = np.arange(len(labels))
    axes[2].bar(positions - 0.18, base, width=0.36, label="self + total")
    axes[2].bar(positions + 0.18, grounded, width=0.36, label="+ grounded")
    axes[2].set_xticks(positions, labels, rotation=15)
    axes[2].set(ylabel="out-of-fold MSE", title="Incremental predictive value")
    axes[2].legend()

    axes[3].bar(
        ["primary - controls"],
        [ablation_control["mean_primary_minus_median_control_regret_change"]],
        color="#54a24b",
    )
    axes[3].axhline(0, color="black", linewidth=0.8)
    axes[3].set(ylabel="normalized-regret change", title="Causal planning necessity")
    figure.tight_layout()
    figure.savefig(PLOT_DIR / "stage30_grounded_planning_value_summary.png", dpi=180)
    plt.close(figure)


DECISION_PAYLOAD = {"status": "INCONCLUSIVE"}
if not PIPELINE_FAILED:
    try:
        REPLICATION_GATE = replicated_gap_gate()
        RELIABILITY_GATE = predictive_gate("native_regret")
        CAUSAL_VALUE_PREDICTION_GATE = predictive_gate("ablation_regret_change")
        ABLATION_CONTROL_GATE = ablation_control_gate()
        FREE_NULL = free_null_summary()
        FRESH_CERTIFICATE = fresh_run_certificate()
        causal_value_pass = bool(
            CAUSAL_VALUE_PREDICTION_GATE["passed"]
            and ABLATION_CONTROL_GATE["passed"]
        )
        if RUN_MODE == "smoke":
            candidate_status = "SMOKE_ONLY"
        elif not REPLICATION_GATE["passed"]:
            candidate_status = "CAUSAL_GROUNDING_GAP_NOT_REPLICATED"
        elif RELIABILITY_GATE["passed"] and causal_value_pass:
            candidate_status = "GROUNDED_CLOSURE_PREDICTS_CAUSAL_PLANNING_VALUE"
        elif RELIABILITY_GATE["passed"]:
            candidate_status = "GROUNDED_CLOSURE_PREDICTS_PLANNING_RELIABILITY_ONLY"
        elif causal_value_pass:
            candidate_status = "GROUNDED_CLOSURE_PREDICTS_ABLATION_VALUE_ONLY"
        else:
            candidate_status = "GROUNDING_GAP_REPLICATED_WITHOUT_PLANNING_VALUE"
        confirmation_eligible = bool(
            SOURCE_IDENTITY.get("confirmation_eligible", False)
            and STAGE18_ARTIFACT_VALIDATED
            and STAGE29_UPSTREAM_BOUND
            and FRESH_CERTIFICATE["passed"]
        )
        status = (
            candidate_status
            if RUN_MODE == "smoke" or confirmation_eligible
            else "UNBOUND_NONFRESH_OR_WRONG_UPSTREAM_EXPLORATORY_RESULT"
        )
        DECISION_PAYLOAD = {
            "status": status,
            "candidate_status": candidate_status,
            "confirmation_eligible": confirmation_eligible,
            "source_bound_claim_eligible": bool(
                SOURCE_IDENTITY.get("confirmation_eligible", False)
            ),
            "stage18_artifact_claim_eligible": bool(STAGE18_ARTIFACT_VALIDATED),
            "exact_stage29_upstream_claim_eligible": bool(STAGE29_UPSTREAM_BOUND),
            "fresh_run_claim_eligible": FRESH_CERTIFICATE["passed"],
            "causal_grounding_gap_replication_gate": REPLICATION_GATE,
            "grounded_closure_planning_reliability_gate": RELIABILITY_GATE,
            "grounded_closure_ablation_value_prediction_gate": CAUSAL_VALUE_PREDICTION_GATE,
            "carrier_ablation_control_gate": ABLATION_CONTROL_GATE,
            "free_motion_null_summary": FREE_NULL,
            "estimand_contract": {
                "closure_schedules": DIAGNOSTIC_SCHEDULES,
                "planning_goal_schedules": PLANNING_GOAL_SCHEDULES,
                "closure_and_goal_contrasts_disjoint": True,
                "planner": "exhaustive six-schedule terminal native latent L2",
                "physical_outcome": "exact simulator normalized block-pose cost",
                "inference_unit": "initial state",
                "crossfit_group": "initial state",
            },
            "claim_boundary": {
                "one_model_checkpoint": True,
                "one_environment": True,
                "fresh_stage30_states": True,
                "learned_decoder_used": False,
                "new_reader_fit": False,
                "goal_conditioned_fit": False,
                "stage18_subspace_refit_or_tuning": False,
                "jacobian_jvp_vjp_or_gradient_used": False,
                "terminal_discrete_planner_not_full_mpc": True,
                "generalization_to_other_models_or_environments": False,
            },
            "prespecified_next_step_if_positive": (
                "freeze Grounded Causal Closure and replicate its planning-value advantage "
                "across a second checkpoint and contact-rich world model"
            ),
        }
        write_json(OUT / "stage30_decision.json", DECISION_PAYLOAD)
        make_plots(
            REPLICATION_GATE,
            RELIABILITY_GATE,
            CAUSAL_VALUE_PREDICTION_GATE,
            ABLATION_CONTROL_GATE,
        )
        print(json.dumps(DECISION_PAYLOAD, indent=2))
    except Exception:
        record_failure("stage30_decision_and_plots")
        DECISION_PAYLOAD = {"status": "INCONCLUSIVE", "failure": FAILURE_MESSAGE}

if not (OUT / "stage30_decision.json").exists():
    write_json(OUT / "stage30_decision.json", DECISION_PAYLOAD)
'''


packaging = STAGE29.packaging.replace(
    "stage29_grounded_closure_result_bundle_",
    "stage30_grounded_planning_value_result_bundle_",
)


protocol_sources = [
    introduction,
    configuration,
    installation,
    setup,
    analysis_helpers,
    model_helpers,
    design_and_upstream,
    physical_truth,
    model_initialization,
    grounded_planning_evaluation,
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
    code(design_and_upstream),
    code(physical_truth),
    code(model_initialization),
    code(grounded_planning_evaluation),
    code(decision_and_plots),
    code(packaging),
]
for index, cell in enumerate(cells):
    cell["id"] = f"stage30-{index:02d}"

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "L4", "name": TARGET.name, "provenance": []},
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
TARGET.write_text(json.dumps(notebook, indent=1) + "\n")
print(f"Wrote {TARGET}")
