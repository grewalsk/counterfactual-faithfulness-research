import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "28_hybrid_control_area_law.ipynb"
NUMERICAL = ROOT.parent / "src/cf_faithfulness/stage28_hybrid_control_area.py"

spec = importlib.util.spec_from_file_location(
    "stage27_builder", ROOT / "build_stage27_action_commutator_notebook.py"
)
STAGE27 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(STAGE27)

code = STAGE27.code
markdown = STAGE27.markdown
assigned_uppercase_names = STAGE27.assigned_uppercase_names
function_sources = STAGE27.function_sources


introduction = r'''# Stage 28: hybrid control-area law

Stage 27 found a contact-amplified **finite action commutator**: action paths
containing the same two pulses reached different physical and predicted
endpoints when their order was reversed, and the frozen Stage 18 block-4
carrier was causally sufficient and necessary for part of that difference.
That establishes noncommutativity, but a single pulse magnitude and two extreme
orders cannot distinguish a genuine control-composition law from an isolated
contact-threshold effect.

Stage 28 makes the stronger test.  At each of four magnitudes, every path
contains exactly five copies of direction (u), five copies of direction (v),
and five zero actions.  Six frozen schedules preserve the complete action
histogram, integrated impulse, energy, active duration, and horizon while
spanning symmetric signed discrete-control-area levels

\[
A \propto (25,15,5,-5,-15,-25).
\]

For smooth control-affine dynamics, schedule-dependent endpoint displacement
is linear in signed control area to leading order and its maximum reversed-area
contrast scales as (\epsilon^2).  Contact dynamics are hybrid and nonsmooth, so
the notebook prespecifies three simulator-defined strata: persistent contact,
boundary switching, and free motion.  A disjoint, model-blind development pool
chooses the first feasible four-magnitude panel from three frozen panels using
only contact coverage.  All physical, predictive, and causal gates are then
computed on fresh confirmation states.

The exact Stage 18 rank-128 carrier remains frozen (rank 64 is sensitivity).
Area reversal and area-specific ablation are compared against shuffled,
empirical-span random, wrong-state, common-mode, reverse-dose, and full-swap
controls.  No representation, reader, layer, magnitude, schedule, or threshold
is fit to confirmation model outputs; no Jacobian, JVP, VJP, or gradient is
used.

This is a finite hybrid-control test, not a proof of an infinitesimal Lie
bracket.  A positive result would support a much sharper claim: within one
contact-rich world model, a previously frozen action carrier mediates a latent
approximation to signed control composition, while contact-boundary states
mark departures from the smooth second-order law.

Return `stage28_hybrid_control_area_result_bundle_<signature>.zip`.
'''


configuration = r'''# SINGLE CONFIGURATION BLOCK — no Stage 28 secrets required.
# Run all on a GPU. A fresh nonce is generated automatically and this committed
# branch is resolved to an exact source commit before model activations open.
import secrets as _secrets
import time as _time

RUN_MODE = "pilot"
EXPERIMENT_SOURCE_REF = "codex/stage28-hybrid-control-area"
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

OUTPUT_DIR = "/content/counterfactual_faithfulness_stage28_control_area"
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/counterfactual_faithfulness_stage28_control_area"
STAGE18_SEARCH_ROOT = "/content/drive/MyDrive"
STAGE27_SEARCH_ROOT = "/content/drive/MyDrive"

PROTOCOL_ID = "stage28-hybrid-control-area-law-v1"
NOTEBOOK_PROTOCOL_SHA256 = "__PROTOCOL_DIGEST__"
EVIDENCE_STATUS = "CONFIRMATORY_ONLY_IF_SOURCE_BOUND_FRESH_STAGE18_FROZEN_AND_REPAIRED_STAGE27_BOUND"
EXPERIMENT_REPOSITORY = "grewalsk/counterfactual-faithfulness-research"
EXPERIMENT_NOTEBOOK_PATH = "notebooks/28_hybrid_control_area_law.ipynb"
EXPERIMENT_BUILDER_PATH = "notebooks/build_stage28_hybrid_control_area_notebook.py"
EXPERIMENT_NUMERICAL_PATH = "src/cf_faithfulness/stage28_hybrid_control_area.py"

EXPECTED_STAGE18_SUBSPACE_SHA256 = "2f9c496d54623a9062e465a18c70039acc18cb8a1cc2833a5f4ade162ca3f90b"
EXPECTED_STAGE18_SOURCE_COMMIT = "16edd247cddcb1aa121340eb5fa42bd9e07004c3"
EXPECTED_STAGE18_STATUS = "CONFIRMED_BIDIRECTIONAL_RANK64_MEDIATOR"
EXPECTED_STAGE18_AMBIENT_DIMENSION = 102400
EXPECTED_STAGE18_MAX_RANK = 128
EXPECTED_STAGE27_STATUS = "CAUSAL_NONCOMMUTATIVE_ACTION_DYNAMICS_SUPPORTED"
EXPECTED_STAGE27_SOURCE_COMMIT = "78d241fce761babdc4c11c51bfc5758867ecea07"

SEED = 28101
DESIGN_SEED = 28137
MODEL_NAME = "jepa_wm_pusht"
ENVIRONMENT = "PushT"
FRAMESKIP = 5
PRIMARY_HORIZON = 3
TARGET_STEPS = [PRIMARY_HORIZON]
FIXED_BLOCK = 4
ACTIVE_BLOCKS = [FIXED_BLOCK]
EXPECTED_CARRIER_CHANNELS = 400

DEVELOPMENT_TRAJECTORIES = list(range(3200, 3230))
EVALUATION_POOL_TRAJECTORIES = list(range(3300, 3420))
EVALUATION_TARGET_PER_STRATUM = 12
EVALUATION_TRAJECTORY_TARGET = 36
TASK_ID_OFFSET = 14000
STATES_PER_TRAJECTORY = 1
ACTION_STEPS = PRIMARY_HORIZON * FRAMESKIP
SCHEDULE_STRINGS = [
    "uuuuuvvvvv", "uuvuuvvuvv", "uvuvuvuvuv",
    "vuvuvuvuvu", "vvuvvuuvuu", "vvvvvuuuuu",
]
SCHEDULE_INVERSION_COUNTS = [0, 5, 10, 15, 20, 25]
SIGNED_AREA_LEVELS = [25, 15, 5, -5, -15, -25]
SCHEDULE_COUNT = 6
MAGNITUDE_PANELS = [
    [0.08, 0.12, 0.16, 0.20],
    [0.10, 0.14, 0.18, 0.22],
    [0.12, 0.16, 0.20, 0.24],
]
MAGNITUDE_COUNT = 4
ACTIONS_PER_STATE = MAGNITUDE_COUNT * SCHEDULE_COUNT
ANGLE_PAIR_DEGREES = [-30.0, 30.0]
DEVELOPMENT_MIN_PER_STRATUM = 3
DISTANCE_GRID = [55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0]
STRATUM_LABELS = ["persistent_contact", "boundary_switching", "free"]

OUTPUT_SKETCH_DIM = 256
TRAIN_OUTPUT_SKETCH_SEED = 18161
EVAL_OUTPUT_SKETCH_SEED = 18183
PRIMARY_RANK = 128
SENSITIVITY_RANKS = [64, 128]
MAX_SUBSPACE_RANK = 128
BOOTSTRAP_SEED = 28269
CAUSAL_RANDOM_DRAWS = 4
CAUSAL_DOSES = [-0.5, 0.25, 0.5, 1.0]
BOOTSTRAP_DRAWS = 10000
INTERVENTION_FORWARDS_PER_RECORD = 30
MAX_ZERO_EDIT_ERROR = 1e-6

MIN_PERSISTENT_MAX_AREA_NORM = 1e-3
MIN_PERSISTENT_AREA_R2 = 0.45
MIN_PERSISTENT_SLOPE_COSINE = 0.20
MIN_MAGNITUDE_EXPONENT = 0.75
MAX_MAGNITUDE_EXPONENT = 3.75
MAX_EPSILON_SQUARED_COLLAPSE_ERROR = 1.50
MIN_PERSISTENT_TO_FREE_NORM_RATIO = 2.0
MIN_MODEL_PERSISTENT_AREA_COSINE = 0.10
MIN_MODEL_ALIGNMENT_GAIN_OVER_WRONG_STATE = 0.05
MAX_MODEL_EXPONENT_ABSOLUTE_ERROR = 1.50
MIN_FULL_SWAP_COEFFICIENT = 0.75
MIN_PRIMARY_AREA_COEFFICIENT = 0.15
MIN_PRIMARY_AREA_COSINE = 0.15
MIN_PRIMARY_GAIN_OVER_RANDOM = 0.05
MIN_PRIMARY_GAIN_OVER_SHUFFLED = 0.05
REQUIRED_POSITIVE_EVALUATION_TRAJECTORIES = 25
MIN_NECESSITY_REDUCTION = 0.05
MIN_NECESSITY_GAIN_OVER_RANDOM = 0.025
MIN_NECESSITY_GAIN_OVER_SHUFFLED = 0.025
REQUIRED_POSITIVE_NECESSITY_TRAJECTORIES = 25

if RUN_MODE == "smoke":
    ACTIVE_DEVELOPMENT_TRAJECTORIES = DEVELOPMENT_TRAJECTORIES[:9]
    ACTIVE_EVALUATION_POOL_TRAJECTORIES = EVALUATION_POOL_TRAJECTORIES[:18]
    ACTIVE_TARGET_PER_STRATUM = 1
    ACTIVE_EVALUATION_TARGET = 3
    ACTIVE_SENSITIVITY_RANKS = [PRIMARY_RANK]
    ACTIVE_CAUSAL_RANDOM_DRAWS = 1
    ACTIVE_CAUSAL_DOSES = [1.0]
    ACTIVE_BOOTSTRAP_DRAWS = 64
    ACTIVE_DEVELOPMENT_MIN_PER_STRATUM = 1
elif RUN_MODE == "pilot":
    ACTIVE_DEVELOPMENT_TRAJECTORIES = DEVELOPMENT_TRAJECTORIES
    ACTIVE_EVALUATION_POOL_TRAJECTORIES = EVALUATION_POOL_TRAJECTORIES
    ACTIVE_TARGET_PER_STRATUM = EVALUATION_TARGET_PER_STRATUM
    ACTIVE_EVALUATION_TARGET = EVALUATION_TRAJECTORY_TARGET
    ACTIVE_SENSITIVITY_RANKS = SENSITIVITY_RANKS
    ACTIVE_CAUSAL_RANDOM_DRAWS = CAUSAL_RANDOM_DRAWS
    ACTIVE_CAUSAL_DOSES = CAUSAL_DOSES
    ACTIVE_BOOTSTRAP_DRAWS = BOOTSTRAP_DRAWS
    ACTIVE_DEVELOPMENT_MIN_PER_STRATUM = DEVELOPMENT_MIN_PER_STRATUM
else:
    raise ValueError("RUN_MODE must be 'smoke' or 'pilot'")

REPO_URL = "https://github.com/facebookresearch/jepa-wms.git"
REPO_COMMIT = "13cf1d9c7e476f53c17714d2e0f1dc239a883ce0"
EXPECTED_HF_REVISION = "9b9c41ef249466630dbf1a20e78391865d07b3b9"
EXPECTED_PRETRAINED_ASSET_SHA256 = {
    "jepa_wm_pusht.pth.tar": "5bd5da68d7198d79a589026e6fe3980ef72e4420d0c00199d90919bb71d9f743",
    "jepa_wm_pusht_decoder.pth.tar": "a28d210fa75a0ea7350fe5842664d5437ca37f92ca343fd0a6305410c6c5ea42",
}
ASSET_REPOSITORY = "grewalsk/counterfactual-faithfulness-research"
ASSET_COMMIT = "2326e74556f6f81db2560e4396f4cc52c16a28f4"
ASSET_SPECS = {
    "stage3b_training_decoders.pt": {
        "path": "artifacts/stage3b_training_decoders.pt",
        "sha256": "1942b9391bc86f65d73622e764981e04ae2c5aec89db1b779615ece310ecb03b",
    },
    "stage3b_pusht_decoder_manifest.json": {
        "path": "artifacts/stage3b_pusht_decoder_manifest.json",
        "sha256": "1f3ca2ad75db81cac65ac4db7f1cc88b97921fb7b08bf789e7611756cad905eb",
    },
}
PINNED = [
    "model_weights", "decoder_weights", "stage3b_training_decoders",
    "stage18_frozen_subspace", "stage27_repaired_positive_certificate",
    "development_only_magnitude_panel_rule", "confirmation_state_pool",
    "signed_area_schedules", "physical_predictive_causal_gates",
]

assert ACTION_STEPS == 15
assert SCHEDULE_COUNT == len(SCHEDULE_STRINGS) == 6
assert MAGNITUDE_COUNT == 4 and ACTIONS_PER_STATE == 24
assert EVALUATION_TRAJECTORY_TARGET == 3 * EVALUATION_TARGET_PER_STRATUM
assert FIXED_BLOCK == 4
assert PRIMARY_RANK == 128 and SENSITIVITY_RANKS == [64, 128]
assert MAX_SUBSPACE_RANK == EXPECTED_STAGE18_MAX_RANK
'''
configuration += "\n\nPROTOCOL_CONFIG_KEYS = " + repr(
    assigned_uppercase_names(configuration)
) + "\n"


installation = STAGE27.installation


setup = STAGE27.setup
setup = setup.replace("Stage 27", "Stage 28").replace("STAGE27", "STAGE28")
setup = setup.replace("stage27_commutator", "stage28_control_area")
setup = setup.replace("stage27-source-binder", "stage28-source-binder")
setup = setup.replace(
    "stage27_action_commutator_result_bundle_",
    "stage28_hybrid_control_area_result_bundle_",
)


analysis_helpers = STAGE27.without_definitions(
    STAGE27.analysis_helpers,
    [
        "pair_swap_permutation",
        "ordered_pulse_bank",
        "paired_antisymmetric_component",
        "paired_swap_delta",
        "paired_ablation_delta",
        "_pair_rows",
        "paired_transfer_metrics",
        "paired_energy_metrics",
        "pair_contact_masks",
        "commutator_contrasts",
        "commutator_norms",
        "commutator_alignment_metrics",
    ],
) + "\n\n\n" + function_sources(
    NUMERICAL.read_text(),
    [
        "schedule_inversion_count",
        "signed_control_area",
        "area_reversal_permutation",
        "area_action_bank",
        "area_antisymmetric_component",
        "area_swap_delta",
        "area_ablation_delta",
        "area_transfer_metrics",
        "area_energy_metrics",
        "contact_regime",
        "magnitude_center",
        "max_area_contrasts",
        "_cosine",
        "area_law_metrics",
        "model_physics_area_metrics",
    ],
)


model_helpers = STAGE27.model_helpers.replace("stage27-jepa-wms", "stage28-jepa-wms")
model_helpers = model_helpers.replace("Stage 27 supports PushT only", "Stage 28 supports PushT only")


design = r'''# Freeze development and confirmation designs before simulator or model data.


def make_specs(trajectory_ids, split):
    specs = []
    center = np.asarray([256.0, 256.0])
    total = len(trajectory_ids)
    for design_index, trajectory_id in enumerate(trajectory_ids):
        phase = 0.29 + 2.0 * np.pi * design_index / total
        block = center + 42.0 * np.asarray([np.cos(phase), np.sin(phase)])
        block_angle = ((1.61 * phase + np.pi) % (2.0 * np.pi)) - np.pi
        approach = phase + [np.pi / 6, 5 * np.pi / 6, 7 * np.pi / 6, 11 * np.pi / 6][design_index % 4]
        approach += 0.09 * np.cos(7 * design_index)
        distance = float(DISTANCE_GRID[(3 * design_index + design_index // len(DISTANCE_GRID)) % len(DISTANCE_GRID)])
        agent = block + distance * np.asarray([np.cos(approach), np.sin(approach)])
        goal_index = (19 * design_index + 7) % total
        goal_phase = 0.73 + 2.0 * np.pi * goal_index / total
        goal_xy = center + 70.0 * np.asarray([np.cos(goal_phase), np.sin(goal_phase)])
        offset = 320000 if split == "development" else 330000
        specs.append({
            "design_index": int(design_index),
            "record_id": int(offset + trajectory_id),
            "trajectory_id": int(trajectory_id),
            "task_id": int(TASK_ID_OFFSET + design_index + (0 if split == "development" else 1000)),
            "time_index": 0,
            "physical_step": 0,
            "split": split,
            "evaluation_seed": int(DESIGN_SEED + 1009 * design_index + (0 if split == "development" else 100000)),
            "approach_distance": distance,
            "goal": np.asarray([
                goal_xy[0], goal_xy[1],
                ((1.29 * goal_phase + np.pi) % (2.0 * np.pi)) - np.pi,
            ], dtype=np.float64),
            "state": np.asarray([
                agent[0], agent[1], block[0], block[1], block_angle,
                0.0, 0.0, 0.0, 0.0, 0.0,
            ], dtype=np.float64),
        })
    return specs


ALL_DEVELOPMENT_SPECS = make_specs(DEVELOPMENT_TRAJECTORIES, "development")
ALL_POOL_SPECS = make_specs(EVALUATION_POOL_TRAJECTORIES, "evaluation")
DEVELOPMENT_SPECS = [
    row for row in ALL_DEVELOPMENT_SPECS
    if row["trajectory_id"] in ACTIVE_DEVELOPMENT_TRAJECTORIES
]
POOL_SPECS = [
    row for row in ALL_POOL_SPECS
    if row["trajectory_id"] in ACTIVE_EVALUATION_POOL_TRAJECTORIES
]


def candidate_action_bank(record, magnitudes):
    state = np.asarray(record["state"], dtype=np.float64)
    if state.shape != (10,):
        raise ValueError("candidate state must be a ten-dimensional dynamic PushT state")
    return area_action_bank(
        state[2:4] - state[:2], magnitudes,
        steps=ACTION_STEPS, angle_pair_degrees=ANGLE_PAIR_DEGREES,
        schedules=SCHEDULE_STRINGS,
    )


np.savez_compressed(
    DESIGN_DIR / "stage28_hybrid_control_area_design.npz",
    development_record_ids=np.asarray([row["record_id"] for row in ALL_DEVELOPMENT_SPECS]),
    evaluation_record_ids=np.asarray([row["record_id"] for row in ALL_POOL_SPECS]),
    evaluation_initial_states=np.stack([row["state"] for row in ALL_POOL_SPECS]),
    evaluation_goals=np.stack([row["goal"] for row in ALL_POOL_SPECS]),
    magnitude_panels=np.asarray(MAGNITUDE_PANELS, dtype=np.float64),
    schedules=np.asarray(SCHEDULE_STRINGS),
    signed_area_levels=np.asarray(SIGNED_AREA_LEVELS, dtype=np.int64),
    reversal_permutation=area_reversal_permutation(MAGNITUDE_COUNT),
)
POOL_MANIFEST = {
    "development_specs": [
        {**{key: value for key, value in row.items() if key not in {"state", "goal"}},
         "state": row["state"].tolist(), "goal": row["goal"].tolist()}
        for row in ALL_DEVELOPMENT_SPECS
    ],
    "confirmation_specs": [
        {**{key: value for key, value in row.items() if key not in {"state", "goal"}},
         "state": row["state"].tolist(), "goal": row["goal"].tolist()}
        for row in ALL_POOL_SPECS
    ],
    "magnitude_panels_in_priority_order": MAGNITUDE_PANELS,
    "panel_selection_rule": "first panel with development contact coverage in all three strata",
    "target_per_confirmation_stratum": ACTIVE_TARGET_PER_STRATUM,
    "schedule_strings": SCHEDULE_STRINGS,
    "signed_area_levels": SIGNED_AREA_LEVELS,
    "complete_action_histogram_matched_within_magnitude": True,
    "selection_uses_model_outputs": False,
    "confirmation_selection_uses_commutator_or_area_effect_magnitude": False,
}
write_json(DESIGN_DIR / "candidate_pool_manifest.json", POOL_MANIFEST)
DESIGN_FREEZE = {
    "created_before_simulator_or_model_data": True,
    "protocol_id": PROTOCOL_ID,
    "run_signature": RUN_SIGNATURE,
    "source_identity": SOURCE_IDENTITY,
    "design_sha256": sha256_file(DESIGN_DIR / "stage28_hybrid_control_area_design.npz"),
    "pool_manifest_sha256": sha256_file(DESIGN_DIR / "candidate_pool_manifest.json"),
    "expected_stage18_subspace_sha256": EXPECTED_STAGE18_SUBSPACE_SHA256,
    "expected_repaired_stage27_source_commit": EXPECTED_STAGE27_SOURCE_COMMIT,
    "fixed_block": FIXED_BLOCK,
    "fixed_primary_rank": PRIMARY_RANK,
    "subspace_refit_allowed": False,
    "coordinate_reader_used": False,
    "jacobian_used": False,
    "model_loaded": bool("MODEL" in globals()),
}
if DESIGN_FREEZE["model_loaded"]:
    raise RuntimeError("model was loaded before Stage 28 design freeze")
write_json(DESIGN_DIR / "design_freeze.json", DESIGN_FREEZE)
'''


truth_generation = r'''# Select magnitudes on development contacts, then freeze confirmation truth.


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
        raise RuntimeError(f"full dynamic restoration drifted: {np.max(np.abs(restored - state))}")
    observation = {
        "visual": np.asarray(environment.render("rgb_array")).copy(),
        "proprio": np.asarray([*environment.agent.position, *environment.agent.velocity], dtype=np.float32),
    }
    return environment, observation


def rollout_dynamic_branch(record, actions, retain_visual=True):
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
                    "visual": np.asarray(observation["visual"]).copy() if retain_visual else None,
                    "proprio": np.asarray(observation["proprio"]).copy(),
                }
                endpoint_state = dynamic_state_from_environment(environment)
    finally:
        environment.close()
    if endpoint_observation is None or endpoint_state is None:
        raise RuntimeError("dynamic rollout missed the primary horizon")
    return initial, endpoint_observation, endpoint_state, cumulative


def exact_dynamic_restore_test(record, magnitudes):
    first, first_observation = reset_dynamic_environment(record["state"], record_task(record), record["evaluation_seed"])
    second, second_observation = reset_dynamic_environment(record["state"], record_task(record), record["evaluation_seed"])
    test_action = candidate_action_bank(record, magnitudes)[0, 0]
    first.step(test_action)
    second.step(test_action)
    first_next, second_next = dynamic_state_from_environment(first), dynamic_state_from_environment(second)
    first.close(); second.close()
    result = {
        "visual_exact": bool(np.array_equal(first_observation["visual"], second_observation["visual"])),
        "proprio_exact": bool(np.array_equal(first_observation["proprio"], second_observation["proprio"])),
        "one_step_continuation_exact": bool(np.allclose(first_next, second_next, atol=1e-12, rtol=0)),
    }
    result["passed"] = bool(all(result.values()))
    if not result["passed"]:
        raise RuntimeError(f"full dynamic restore test failed: {result}")
    return result


def screen_development_panel(panel):
    rows = []
    for record in DEVELOPMENT_SPECS:
        counts = []
        for action in candidate_action_bank(record, panel):
            _, _, _, contacts = rollout_dynamic_branch(record, action, retain_visual=False)
            counts.append(contacts)
        matrix = np.asarray(counts, dtype=np.int64).reshape(MAGNITUDE_COUNT, SCHEDULE_COUNT)
        rows.append({
            "record_id": int(record["record_id"]),
            "trajectory_id": int(record["trajectory_id"]),
            "approach_distance": float(record["approach_distance"]),
            "regime": contact_regime(matrix),
            "contact_fraction": float(np.mean(matrix > 0)),
        })
    coverage = {label: sum(row["regime"] == label for row in rows) for label in STRATUM_LABELS}
    return rows, coverage


if not PIPELINE_FAILED:
    try:
        REPO = configure_repo()
        RESTORE_TEST = exact_dynamic_restore_test(DEVELOPMENT_SPECS[0], MAGNITUDE_PANELS[0])
        DEVELOPMENT_PANEL_ROWS = []
        SELECTED_MAGNITUDES = None
        PANEL_COVERAGE = None
        for panel_index, panel in enumerate(MAGNITUDE_PANELS):
            rows, coverage = screen_development_panel(panel)
            for row in rows:
                DEVELOPMENT_PANEL_ROWS.append({"panel_index": panel_index, "panel": str(panel), **row})
            if all(coverage[label] >= ACTIVE_DEVELOPMENT_MIN_PER_STRATUM for label in STRATUM_LABELS):
                SELECTED_MAGNITUDES = list(map(float, panel))
                PANEL_COVERAGE = coverage
                break
        if SELECTED_MAGNITUDES is None:
            raise RuntimeError("no frozen magnitude panel covered all three development contact strata")
        write_csv(EVIDENCE_DIR / "development_panel_screen.csv", DEVELOPMENT_PANEL_ROWS)
        MAGNITUDE_SELECTION = {
            "selected_panel": SELECTED_MAGNITUDES,
            "selected_panel_index": int(MAGNITUDE_PANELS.index(SELECTED_MAGNITUDES)),
            "development_coverage": PANEL_COVERAGE,
            "first_feasible_panel_rule": True,
            "development_and_confirmation_trajectories_disjoint": True,
            "model_outputs_used": False,
            "physical_area_effect_magnitude_used": False,
        }
        write_json(DESIGN_DIR / "magnitude_selection_freeze.json", MAGNITUDE_SELECTION)
    except Exception:
        record_failure("development_magnitude_panel_selection")


def branch_path(record_id):
    return TRUTH_DIR / f"state_{int(record_id):06d}.npz"


def generate_truth(records):
    started = time.perf_counter()
    for index, record in enumerate(records):
        destination = branch_path(record["record_id"])
        if destination.exists():
            PROVENANCE_COUNTS["cache_hits"] += 1
            raise RuntimeError(f"fresh-run truth shard already exists: {destination}")
        action_bank = candidate_action_bank(record, SELECTED_MAGNITUDES)
        initials, initial_proprios = [], []
        endpoint_visuals, endpoint_states, interaction_counts = [], [], []
        for action in action_bank:
            initial, endpoint, state, contacts = rollout_dynamic_branch(record, action)
            initials.append(initial["visual"]); initial_proprios.append(initial["proprio"])
            endpoint_visuals.append(endpoint["visual"]); endpoint_states.append(state)
            interaction_counts.append(contacts)
        if not all(np.array_equal(initials[0], value) for value in initials[1:]):
            raise AssertionError("initial visual drift across schedule branches")
        if not all(np.array_equal(initial_proprios[0], value) for value in initial_proprios[1:]):
            raise AssertionError("initial proprio drift across schedule branches")
        atomic_npz(
            destination,
            record_id=np.asarray(record["record_id"], dtype=np.int64),
            trajectory_id=np.asarray(record["trajectory_id"], dtype=np.int64),
            task_id=np.asarray(record["task_id"], dtype=np.int64),
            split=np.asarray(record["split"]), state=np.asarray(record["state"], dtype=np.float64),
            goal=np.asarray(record["goal"], dtype=np.float64),
            initial_visual=np.asarray(initials[0], dtype=np.uint8),
            initial_proprio=np.asarray(initial_proprios[0], dtype=np.float32),
            selected_actions=action_bank.astype(np.float32),
            endpoint_visuals=np.asarray(endpoint_visuals, dtype=np.uint8),
            endpoint_states=np.asarray(endpoint_states, dtype=np.float64),
            interaction_counts=np.asarray(interaction_counts, dtype=np.int32),
        )
        PROVENANCE_COUNTS["truth_generated"] += 1
        write_json(OUT / "truth_control_area_pool_progress.json", {
            "completed": index + 1, "total": len(records), "last_record_id": int(record["record_id"]),
        })
    TIMINGS["truth_control_area_pool_seconds"] = time.perf_counter() - started


def physical_record_row(record):
    with np.load(branch_path(record["record_id"])) as payload:
        endpoints = payload["endpoint_states"].astype(np.float64)
        contacts = payload["interaction_counts"].astype(np.int64).reshape(MAGNITUDE_COUNT, SCHEDULE_COUNT)
        actions = payload["selected_actions"].astype(np.float64)
    poses = pose_target(endpoints)
    law = area_law_metrics(poses, actions, SELECTED_MAGNITUDES)
    grouped_actions = actions.reshape(MAGNITUDE_COUNT, SCHEDULE_COUNT, ACTION_STEPS, 2)
    controls_match = True
    for group in grouped_actions:
        controls_match &= bool(np.allclose(np.sum(group, axis=1), np.sum(group[0], axis=0), atol=5e-7))
        controls_match &= bool(np.allclose(np.sum(group**2, axis=(1, 2)), np.sum(group[0] ** 2), atol=5e-7))
    return {
        "record_id": int(record["record_id"]), "trajectory_id": int(record["trajectory_id"]),
        "approach_distance": float(record["approach_distance"]),
        "regime": contact_regime(contacts), "contact_fraction": float(np.mean(contacts > 0)),
        "controls_match_within_magnitude": controls_match,
        "median_max_area_contrast_norm": float(np.median(law["max_area_contrast_norms"])),
        **{key: value for key, value in law.items() if key not in {"max_area_contrast_norms", "slope_norms"}},
        **{f"max_area_norm_m{index}": float(value) for index, value in enumerate(law["max_area_contrast_norms"])},
    }


def select_records(records, target_per_stratum):
    rows = [physical_record_row(record) for record in records]
    selected_ids = []
    for label in STRATUM_LABELS:
        ids = [row["record_id"] for row in rows if row["regime"] == label]
        if len(ids) < int(target_per_stratum):
            raise RuntimeError(f"confirmation pool has {len(ids)} {label} states; requires {target_per_stratum}")
        selected_ids.extend(ids[: int(target_per_stratum)])
    selected = [record for record in records if record["record_id"] in selected_ids]
    return selected, rows


def make_truth_montage(records):
    sample = []
    for label in STRATUM_LABELS:
        sample.extend([record for record in records if record["regime"] == label][:2])
    figure, axes = plt.subplots(len(sample), 3, figsize=(9, 2.7 * len(sample)))
    for row_index, record in enumerate(sample):
        with np.load(branch_path(record["record_id"])) as payload:
            initial = payload["initial_visual"]
            endpoints = payload["endpoint_visuals"]
        for column, (image, title) in enumerate([
            (initial, f"{record['regime']} initial"),
            (endpoints[0], "+max area"), (endpoints[-1], "-max area"),
        ]):
            axes[row_index, column].imshow(image); axes[row_index, column].axis("off")
            axes[row_index, column].set_title(title)
    figure.tight_layout()
    figure.savefig(PLOT_DIR / "stage28_physical_area_montage.png", dpi=160)
    plt.close(figure)


if not PIPELINE_FAILED:
    try:
        generate_truth(POOL_SPECS)
        ALL_EVALUATION_RECORDS, PHYSICAL_RECORD_ROWS = select_records(POOL_SPECS, ACTIVE_TARGET_PER_STRATUM)
        row_lookup = {int(row["record_id"]): row for row in PHYSICAL_RECORD_ROWS}
        for record in ALL_EVALUATION_RECORDS:
            record["regime"] = row_lookup[int(record["record_id"])]["regime"]
        if len(ALL_EVALUATION_RECORDS) != ACTIVE_EVALUATION_TARGET:
            raise RuntimeError("stratified confirmation selection returned the wrong record count")
        selected_ids = [int(record["record_id"]) for record in ALL_EVALUATION_RECORDS]
        wrong_state_map = {}
        for label in STRATUM_LABELS:
            ids = [int(record["record_id"]) for record in ALL_EVALUATION_RECORDS if record["regime"] == label]
            permutation = fixed_derangement(len(ids), stable_seed(DESIGN_SEED, label)) if len(ids) > 1 else np.asarray([0])
            wrong_state_map.update({str(ids[index]): int(ids[permutation[index]]) for index in range(len(ids))})
        write_csv(EVIDENCE_DIR / "physical_control_area_record_rows.csv", PHYSICAL_RECORD_ROWS)
        SELECTION_CERTIFICATE = {
            "selected_record_ids": selected_ids,
            "selected_counts": {label: sum(record["regime"] == label for record in ALL_EVALUATION_RECORDS) for label in STRATUM_LABELS},
            "wrong_state_map": wrong_state_map,
            "selection_uses_only_contact_regime": True,
            "commutator_or_area_effect_magnitude_used_for_selection": False,
            "model_outputs_used_for_selection": False,
            "magnitude_selection_freeze_sha256": sha256_file(DESIGN_DIR / "magnitude_selection_freeze.json"),
            "physical_rows_sha256": sha256_file(EVIDENCE_DIR / "physical_control_area_record_rows.csv"),
        }
        write_json(DESIGN_DIR / "physical_selection_freeze.json", SELECTION_CERTIFICATE)
        make_truth_montage(ALL_EVALUATION_RECORDS)
        memory_report("physical_truth_and_selection_complete")
    except Exception:
        record_failure("physical_truth_selection")
'''


artifact_import = r'''# Bind the frozen Stage 18 carrier and repaired positive Stage 27 result.
STAGE18_ARTIFACT_VALIDATED = False
STAGE27_UPSTREAM_BOUND = False


def unique_matching_path(candidates, expected_hash=None):
    existing = sorted({Path(value) for value in candidates if Path(value).is_file()})
    if expected_hash is not None:
        existing = [value for value in existing if sha256_file(value) == expected_hash]
    if not existing:
        raise FileNotFoundError("no matching frozen upstream artifact was found in MyDrive")
    existing.sort(key=lambda value: (len(str(value)), str(value)))
    return existing[0]


if not PIPELINE_FAILED:
    try:
        verify_executed_notebook_through("# Bind the frozen Stage 18 carrier and repaired positive Stage 27 result.")
        stage18_root = Path(STAGE18_SEARCH_ROOT)
        stage18_candidates = list(stage18_root.glob(
            "counterfactual_faithfulness_stage18_rank64/pilot_*/subspaces/frozen_rank64_confirmation_subspaces.npz"
        ))
        FROZEN_SUBSPACE_PATH = unique_matching_path(stage18_candidates, EXPECTED_STAGE18_SUBSPACE_SHA256)
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
        if stage18_decision.get("status") != EXPECTED_STAGE18_STATUS or not stage18_decision.get("confirmation_eligible", False):
            raise RuntimeError("Stage 18 decision is not the frozen confirmation")
        if stage18_manifest.get("subspace_sha256") != EXPECTED_STAGE18_SUBSPACE_SHA256:
            raise RuntimeError("Stage 18 manifest does not bind the required subspace")
        if stage18_source.get("resolved_commit") != EXPECTED_STAGE18_SOURCE_COMMIT or not stage18_source.get("confirmation_eligible", False):
            raise RuntimeError("Stage 18 source binding mismatch")
        with np.load(FROZEN_SUBSPACE_PATH) as payload:
            FROZEN_SUBSPACES = {name: payload[name].copy() for name in payload.files}
        artifact_contract = validate_stage18_subspace_arrays(
            FROZEN_SUBSPACES, ambient=EXPECTED_STAGE18_AMBIENT_DIMENSION,
            max_rank=EXPECTED_STAGE18_MAX_RANK,
        )
        STAGE18_ARTIFACT_CERTIFICATE = {
            "validated_before_stage28_model_activations": True,
            "path": str(FROZEN_SUBSPACE_PATH), "bytes": int(FROZEN_SUBSPACE_PATH.stat().st_size),
            "sha256": sha256_file(FROZEN_SUBSPACE_PATH), "artifact_contract": artifact_contract,
            "stage28_subspace_refit": False, "stage28_basis_rotation_or_tuning": False,
        }
        write_json(OUT / "stage18_artifact_certificate.json", STAGE18_ARTIFACT_CERTIFICATE)
        STAGE18_ARTIFACT_VALIDATED = True

        stage27_root = Path(STAGE27_SEARCH_ROOT)
        candidates = list(stage27_root.glob(
            "counterfactual_faithfulness_stage27_commutator/pilot_*/stage27_decision.json"
        ))
        valid = []
        for decision_path in candidates:
            source_path = decision_path.parent / "source_identity.json"
            if not source_path.is_file():
                continue
            decision = json.loads(decision_path.read_text())
            source = json.loads(source_path.read_text())
            if (
                decision.get("status") == EXPECTED_STAGE27_STATUS
                and decision.get("confirmation_eligible", False)
                and source.get("resolved_commit") == EXPECTED_STAGE27_SOURCE_COMMIT
                and source.get("confirmation_eligible", False)
            ):
                valid.append((decision_path, source_path, decision, source))
        if not valid:
            raise FileNotFoundError(
                "No clean repaired Stage 27 positive run was found. Run the repaired Stage 27 notebook at "
                f"commit {EXPECTED_STAGE27_SOURCE_COMMIT} first."
            )
        valid.sort(key=lambda row: str(row[0]))
        stage27_decision_path, stage27_source_path, stage27_decision, stage27_source = valid[-1]
        STAGE27_CERTIFICATE = {
            "validated_before_stage28_model_activations": True,
            "path": str(stage27_decision_path), "decision_status": stage27_decision["status"],
            "confirmation_eligible": stage27_decision["confirmation_eligible"],
            "resolved_commit": stage27_source["resolved_commit"],
            "decision_sha256": sha256_file(stage27_decision_path),
            "source_identity_sha256": sha256_file(stage27_source_path),
        }
        write_json(OUT / "stage27_upstream_certificate.json", STAGE27_CERTIFICATE)
        STAGE27_UPSTREAM_BOUND = True
        memory_report("upstream_artifacts_validated")
    except Exception:
        record_failure("upstream_artifact_import")
'''


model_and_baselines = STAGE27.model_and_baselines
model_and_baselines = model_and_baselines.replace(
    "# Load frozen JEPA-WM and generate Stage 27 paired-order baselines at block 4.",
    "# Load frozen JEPA-WM and generate Stage 28 area-schedule baselines at block 4.",
)
model_and_baselines = model_and_baselines.replace(
    'interventions_per_record = INTERVENTION_FORWARDS_PER_RECORD if RUN_MODE == "pilot" else 5',
    'interventions_per_record = INTERVENTION_FORWARDS_PER_RECORD if RUN_MODE == "pilot" else 5',
)
model_final = model_and_baselines.index("def evaluate_model_physical_commutators")
model_and_baselines = model_and_baselines[:model_final] + r'''def evaluate_model_physical_area(records):
    rows = []
    for record in records:
        payload = load_baseline(record["record_id"])
        predicted_pose = payload["decoded_pose"].astype(np.float64)
        with np.load(branch_path(record["record_id"])) as truth_payload:
            truth_pose = pose_target(truth_payload["endpoint_states"].astype(np.float64))
            actions = truth_payload["selected_actions"].astype(np.float64)
        alignment = model_physics_area_metrics(predicted_pose, truth_pose, MAGNITUDE_COUNT)
        predicted_law = area_law_metrics(predicted_pose, actions, SELECTED_MAGNITUDES)
        truth_law = area_law_metrics(truth_pose, actions, SELECTED_MAGNITUDES)
        rows.append({
            "record_id": int(record["record_id"]), "trajectory_id": int(record["trajectory_id"]),
            "regime": record["regime"], **alignment,
            "predicted_area_r_squared": predicted_law["area_r_squared"],
            "truth_area_r_squared": truth_law["area_r_squared"],
            "predicted_magnitude_exponent": predicted_law["magnitude_exponent"],
            "truth_magnitude_exponent": truth_law["magnitude_exponent"],
            "exponent_absolute_error": abs(predicted_law["magnitude_exponent"] - truth_law["magnitude_exponent"]),
        })
    write_csv(EVIDENCE_DIR / "model_physical_area_record_rows.csv", rows)
    return rows


EVALUATION_OPENED = False
if not PIPELINE_FAILED:
    try:
        if not STAGE18_ARTIFACT_VALIDATED or not STAGE27_UPSTREAM_BOUND:
            raise RuntimeError("frozen Stage 18 and repaired positive Stage 27 evidence must be bound first")
        MODEL, PREPROCESSOR, PREDICTOR, PREDICTOR_BLOCK_MODULES = load_frozen_model()
        if len(PREDICTOR_BLOCK_MODULES) != 6:
            raise RuntimeError("predictor block count changed")
        for module in PREDICTOR_BLOCK_MODULES:
            if not isinstance(module, torch.nn.Module) or getattr(module, "register_forward_hook", None) is None:
                raise RuntimeError("predictor block does not support forward hooks")
        TRAIN_OUTPUT_PROJECTOR = CountSketchProjector(256 * 384, OUTPUT_SKETCH_DIM, TRAIN_OUTPUT_SKETCH_SEED)
        EVAL_OUTPUT_PROJECTOR = CountSketchProjector(256 * 384, OUTPUT_SKETCH_DIM, EVAL_OUTPUT_SKETCH_SEED)
        DECODE_PHYSICAL_POSE = physical_pose_decoder()
        first_record_id = ALL_EVALUATION_RECORDS[0]["record_id"]
        HOOK_IDENTITY = hook_identity_test(first_record_id)
        FORWARD_BENCHMARK = forward_benchmark(first_record_id)
        extract_baselines(ALL_EVALUATION_RECORDS, [FIXED_BLOCK])
        MODEL_PHYSICAL_RECORD_ROWS = evaluate_model_physical_area(ALL_EVALUATION_RECORDS)
        EVALUATION_OPENED = True
        write_json(OUT / "evaluation_open_certificate.json", {
            "opened": True, "source_identity": SOURCE_IDENTITY,
            "stage18_artifact_certificate_sha256": sha256_file(OUT / "stage18_artifact_certificate.json"),
            "stage27_upstream_certificate_sha256": sha256_file(OUT / "stage27_upstream_certificate.json"),
            "physical_selection_freeze_sha256": sha256_file(DESIGN_DIR / "physical_selection_freeze.json"),
            "evaluation_records": len(ALL_EVALUATION_RECORDS),
            "fit_or_selection_model_activations": [], "stage28_subspace_refit": False,
        })
        memory_report("stage28_model_baselines_complete")
    except Exception:
        record_failure("stage28_model_baselines")
'''


causal_interchange = r'''# Reverse or erase only the area-antisymmetric frozen-carrier component.


def load_frozen_subspaces():
    if not STAGE18_ARTIFACT_VALIDATED:
        raise RuntimeError("Stage 18 artifact is not validated")
    return FROZEN_SUBSPACES


def whiten_carrier(values, subspaces):
    return transform_primal_channels(np.asarray(values, dtype=np.float64), subspaces["channel_inverse_square_root"])


def native_edit(values, subspaces):
    return inverse_transform_primal_channels(np.asarray(values, dtype=np.float64), subspaces["channel_square_root"])


def intervention_path(record_id):
    return INTERVENTION_DIR / f"state_{int(record_id):06d}.json"


def finite_json_rows(rows):
    return [{key: None if isinstance(value, (float, np.floating)) and not np.isfinite(value) else value for key, value in row.items()} for row in rows]


def wrong_state_area_delta(wrong_white, basis):
    component = area_antisymmetric_component(wrong_white, MAGNITUDE_COUNT).reshape(ACTIONS_PER_STATE, -1)
    projected = (component @ basis) @ basis.T
    return (-2.0 * projected).reshape(wrong_white.shape)


def intervention_specs(record, carrier, subspaces):
    record_id = int(record["record_id"])
    white = whiten_carrier(carrier, subspaces)
    primary_basis = subspaces["primary_basis"][:, :PRIMARY_RANK]
    primary_swap = area_swap_delta(white, MAGNITUDE_COUNT, basis=primary_basis)
    primary_ablation = area_ablation_delta(white, MAGNITUDE_COUNT, primary_basis)
    full_swap = area_swap_delta(white, MAGNITUDE_COUNT, basis=None)
    if min(np.linalg.norm(primary_swap), np.linalg.norm(primary_ablation)) <= 1e-12:
        raise RuntimeError("primary area intervention is degenerate")
    specs = []

    def add(condition, family, mode, rank, dose, delta):
        specs.append({"condition": condition, "family": family, "mode": mode, "rank": int(rank),
                      "dose": float(dose), "delta_white": np.asarray(delta, dtype=np.float64)})

    for dose in ACTIVE_CAUSAL_DOSES:
        add(f"primary_r{PRIMARY_RANK:03d}", "primary", "sufficiency", PRIMARY_RANK, dose, float(dose) * primary_swap)
    for rank in ACTIVE_SENSITIVITY_RANKS:
        learned = area_swap_delta(white, MAGNITUDE_COUNT, basis=subspaces["primary_basis"][:, :rank])
        if rank != PRIMARY_RANK:
            add(f"learned_r{rank:03d}", "rank_sensitivity", "sufficiency", rank, 1.0, learned)
        shuffled = area_swap_delta(white, MAGNITUDE_COUNT, basis=subspaces["shuffled_basis"][:, :rank])
        add(f"shuffled_r{rank:03d}", "matched_shuffled_control", "sufficiency", rank, 1.0, norm_match(shuffled, learned))
        for draw in range(ACTIVE_CAUSAL_RANDOM_DRAWS):
            random_delta = area_swap_delta(white, MAGNITUDE_COUNT, basis=subspaces[f"random_basis_{draw:02d}"][:, :rank])
            add(f"random_r{rank:03d}_{draw:02d}", "empirical_span_random_control", "sufficiency", rank, 1.0, norm_match(random_delta, learned))

    wrong_id = int(SELECTION_CERTIFICATE["wrong_state_map"][str(record_id)])
    wrong_carrier = carrier_for_block(load_baseline(wrong_id), FIXED_BLOCK)
    wrong_delta = wrong_state_area_delta(whiten_carrier(wrong_carrier, subspaces), primary_basis)
    add(f"wrong_state_r{PRIMARY_RANK:03d}", "state_specificity_control", "sufficiency", PRIMARY_RANK, 1.0, norm_match(wrong_delta, primary_swap))
    add(f"common_mode_r{PRIMARY_RANK:03d}", "matched_common_mode_control", "sufficiency", PRIMARY_RANK, 1.0, matched_common_mode(primary_swap, primary_basis[:, 0]))
    add("full_activation_swap", "positive_control_only", "sufficiency", -1, 1.0, full_swap)

    for rank in ACTIVE_SENSITIVITY_RANKS:
        learned_basis = subspaces["primary_basis"][:, :rank]
        learned = area_ablation_delta(white, MAGNITUDE_COUNT, learned_basis)
        learned_name = f"ablate_primary_r{rank:03d}" if rank == PRIMARY_RANK else f"ablate_learned_r{rank:03d}"
        add(learned_name, "primary" if rank == PRIMARY_RANK else "rank_sensitivity", "necessity", rank, 1.0, learned)
        shuffled = area_ablation_delta(white, MAGNITUDE_COUNT, subspaces["shuffled_basis"][:, :rank])
        add(f"ablate_shuffled_r{rank:03d}", "matched_shuffled_control", "necessity", rank, 1.0, norm_match(shuffled, learned))
        for draw in range(ACTIVE_CAUSAL_RANDOM_DRAWS):
            random_delta = area_ablation_delta(white, MAGNITUDE_COUNT, subspaces[f"random_basis_{draw:02d}"][:, :rank])
            add(f"ablate_random_r{rank:03d}_{draw:02d}", "empirical_span_random_control", "necessity", rank, 1.0, norm_match(random_delta, learned))
    if RUN_MODE == "pilot" and len(specs) != INTERVENTION_FORWARDS_PER_RECORD:
        raise RuntimeError(f"expected {INTERVENTION_FORWARDS_PER_RECORD} interventions, found {len(specs)}")
    for specification in specs:
        specification["edit_norm"] = float(np.linalg.norm(specification["delta_white"]))
        specification["primary_swap_norm"] = float(np.linalg.norm(primary_swap))
        specification["primary_ablation_norm"] = float(np.linalg.norm(primary_ablation))
        specification["full_swap_norm"] = float(np.linalg.norm(full_swap))
    return specs


def area_metrics(baseline_output, patched_output, baseline_pose, patched_pose):
    row = {}
    for prefix, baseline, patched in [
        ("output", baseline_output, patched_output), ("pose", baseline_pose, patched_pose),
    ]:
        transfer = area_transfer_metrics(baseline, patched, MAGNITUDE_COUNT)
        energy = area_energy_metrics(baseline, patched, MAGNITUDE_COUNT)
        for name, value in transfer.items(): row[f"{prefix}_{name}"] = value
        for name, value in energy.items(): row[f"{prefix}_area_{name}"] = value
    return row


def make_result_row(record, condition, family, mode, rank, dose,
                    baseline_output, patched_output, baseline_pose, patched_pose,
                    edit_norm, primary_swap_norm, primary_ablation_norm, full_swap_norm):
    return {
        "record_id": int(record["record_id"]), "trajectory_id": int(record["trajectory_id"]),
        "task_id": int(record["task_id"]), "regime": record["regime"], "selected_block": FIXED_BLOCK,
        "condition": condition, "family": family, "mode": mode, "rank": int(rank), "dose": float(dose),
        "output_rms_change": float(np.sqrt(np.mean((patched_output - baseline_output) ** 2))),
        "carrier_edit_whitened_norm": float(edit_norm), "primary_swap_norm": float(primary_swap_norm),
        "primary_ablation_norm": float(primary_ablation_norm), "full_swap_norm": float(full_swap_norm),
        "edit_to_full_swap_ratio": float(edit_norm) / max(float(full_swap_norm), 1e-12),
        **area_metrics(baseline_output, patched_output, baseline_pose, patched_pose),
    }


def run_record_interventions(record, subspaces):
    destination = intervention_path(record["record_id"])
    if destination.exists():
        PROVENANCE_COUNTS["cache_hits"] += 1
        raise RuntimeError(f"fresh-run intervention shard already exists: {destination}")
    payload = load_baseline(record["record_id"])
    carrier = carrier_for_block(payload, FIXED_BLOCK)
    baseline_output = payload["output_eval_sketch"].astype(np.float64)
    baseline_pose = payload["decoded_pose"].astype(np.float64)
    specifications = intervention_specs(record, carrier, subspaces)
    rows = [make_result_row(record, "no_edit", "baseline", "baseline", 0, 0.0,
                            baseline_output, baseline_output, baseline_pose, baseline_pose, 0, 0, 0, 0)]
    initial, actions = state_model_inputs(record["record_id"])
    for specification in specifications:
        delta_native = native_edit(specification["delta_white"], subspaces)
        delta_tensor = torch.as_tensor(delta_native, device="cuda", dtype=torch.float32)
        with torch.inference_mode():
            patched, _, _ = forward_with_carriers(
                initial, actions, PRIMARY_HORIZON, capture_blocks=[FIXED_BLOCK],
                intervention={"block": FIXED_BLOCK, "delta": delta_tensor},
            )
            patched_output = EVAL_OUTPUT_PROJECTOR(patched).cpu().numpy()
            patched_pose = DECODE_PHYSICAL_POSE(patched).cpu().numpy()
        rows.append(make_result_row(
            record, specification["condition"], specification["family"], specification["mode"],
            specification["rank"], specification["dose"], baseline_output, patched_output,
            baseline_pose, patched_pose, specification["edit_norm"], specification["primary_swap_norm"],
            specification["primary_ablation_norm"], specification["full_swap_norm"],
        ))
        del patched, patched_output, patched_pose, delta_tensor
    write_json(destination, finite_json_rows(rows))
    PROVENANCE_COUNTS["intervention_generated"] += 1
    del initial, actions
    gc.collect(); torch.cuda.empty_cache()
    return rows


def run_all_interventions(records):
    started = time.perf_counter(); subspaces = load_frozen_subspaces(); rows = []
    for index, record in enumerate(records):
        rows.extend(run_record_interventions(record, subspaces))
        write_json(OUT / "intervention_progress.json", {
            "completed": index + 1, "total": len(records), "last_record_id": int(record["record_id"]),
        })
    TIMINGS["causal_intervention_seconds"] = time.perf_counter() - started
    write_csv(EVIDENCE_DIR / "control_area_intervention_rows.csv", rows)
    return rows


if not PIPELINE_FAILED and EVALUATION_OPENED:
    try:
        INTERVENTION_ROWS = run_all_interventions(ALL_EVALUATION_RECORDS)
        memory_report("stage28_causal_interventions_complete")
    except Exception:
        record_failure("stage28_causal_interchange")
'''


decision_and_plots = r'''# Apply physical, predictive, and causal hybrid control-area gates.


def lookup(rows, trajectory_id, condition, dose=1.0, key="output_coefficient"):
    values = [row[key] for row in rows if int(row["trajectory_id"]) == int(trajectory_id)
              and row["condition"] == condition and np.isclose(float(row["dose"]), float(dose))]
    return float(values[0]) if len(values) == 1 else math.nan


def random_median(rows, trajectory_id, rank, key, ablate=False):
    prefix = "ablate_random" if ablate else "random"
    values = [lookup(rows, trajectory_id, f"{prefix}_r{rank:03d}_{draw:02d}", key=key)
              for draw in range(ACTIVE_CAUSAL_RANDOM_DRAWS)]
    return float(np.nanmedian(values))


def bootstrap_interval(values, trajectories, label):
    draws = clustered_bootstrap_mean(
        np.asarray(values, dtype=np.float64), np.asarray(trajectories), ACTIVE_BOOTSTRAP_DRAWS,
        stable_seed(BOOTSTRAP_SEED, label) % (2**31 - 1),
    )
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def physical_area_gate():
    selected_ids = {int(record["record_id"]) for record in ALL_EVALUATION_RECORDS}
    rows = [row for row in PHYSICAL_RECORD_ROWS if int(row["record_id"]) in selected_ids]
    by_regime = {label: [row for row in rows if row["regime"] == label] for label in STRATUM_LABELS}
    persistent = by_regime["persistent_contact"]; free = by_regime["free"]; boundary = by_regime["boundary_switching"]
    persistent_norm = float(np.median([row["median_max_area_contrast_norm"] for row in persistent]))
    free_norm = float(np.median([row["median_max_area_contrast_norm"] for row in free]))
    persistent_r2 = float(np.median([row["area_r_squared"] for row in persistent]))
    persistent_cosine = float(np.median([row["mean_slope_direction_cosine"] for row in persistent]))
    persistent_exponent = float(np.median([row["magnitude_exponent"] for row in persistent]))
    persistent_collapse = float(np.median([row["epsilon_squared_collapse_error"] for row in persistent]))
    ratio = persistent_norm / max(free_norm, 1e-12)
    passed = bool(
        all(len(by_regime[label]) == ACTIVE_TARGET_PER_STRATUM for label in STRATUM_LABELS)
        and all(row["controls_match_within_magnitude"] for row in rows)
        and persistent_norm >= MIN_PERSISTENT_MAX_AREA_NORM
        and persistent_r2 >= MIN_PERSISTENT_AREA_R2
        and persistent_cosine >= MIN_PERSISTENT_SLOPE_COSINE
        and MIN_MAGNITUDE_EXPONENT <= persistent_exponent <= MAX_MAGNITUDE_EXPONENT
        and persistent_collapse <= MAX_EPSILON_SQUARED_COLLAPSE_ERROR
        and ratio >= MIN_PERSISTENT_TO_FREE_NORM_RATIO
        and all(0.0 < row["contact_fraction"] < 1.0 for row in boundary)
    )
    return {
        "selected_counts": {label: len(by_regime[label]) for label in STRATUM_LABELS},
        "median_persistent_max_area_contrast_norm": persistent_norm,
        "median_free_max_area_contrast_norm": free_norm,
        "persistent_to_free_norm_ratio": ratio,
        "median_persistent_area_r_squared": persistent_r2,
        "median_persistent_slope_direction_cosine": persistent_cosine,
        "median_persistent_magnitude_exponent": persistent_exponent,
        "median_persistent_epsilon_squared_collapse_error": persistent_collapse,
        "boundary_contact_fraction_range": [float(min(row["contact_fraction"] for row in boundary)),
                                            float(max(row["contact_fraction"] for row in boundary))],
        "complete_action_histogram_impulse_energy_duration_matched": bool(all(row["controls_match_within_magnitude"] for row in rows)),
        "passed": passed,
    }


def aggregate_alignment(rows):
    truth_energy = float(sum(row["target_energy"] for row in rows))
    predicted_energy = float(sum(row["predicted_energy"] for row in rows))
    dot = float(sum(row["coefficient"] * row["target_energy"] for row in rows))
    return dot / max(truth_energy, 1e-12), dot / max(math.sqrt(truth_energy * predicted_energy), 1e-12)


def model_area_gate():
    persistent = [row for row in MODEL_PHYSICAL_RECORD_ROWS if row["regime"] == "persistent_contact"]
    coefficient, cosine = aggregate_alignment(persistent)
    lookup_truth = {int(row["record_id"]): row for row in MODEL_PHYSICAL_RECORD_ROWS}
    wrong_rows = []
    for row in persistent:
        wrong = lookup_truth[int(SELECTION_CERTIFICATE["wrong_state_map"][str(int(row["record_id"]))])]
        wrong_rows.append({**row, "coefficient": 0.0, "target_energy": wrong["target_energy"]})
    # Exact wrong-state cosine is reconstructed from saved decoded poses below.
    matched_predicted, matched_truth, wrong_truth = [], [], []
    for record in [value for value in ALL_EVALUATION_RECORDS if value["regime"] == "persistent_contact"]:
        payload = load_baseline(record["record_id"])
        predicted = magnitude_center(payload["decoded_pose"].astype(np.float64), MAGNITUDE_COUNT).reshape(-1)
        with np.load(branch_path(record["record_id"])) as truth_payload:
            truth = magnitude_center(pose_target(truth_payload["endpoint_states"].astype(np.float64)), MAGNITUDE_COUNT).reshape(-1)
        wrong_id = int(SELECTION_CERTIFICATE["wrong_state_map"][str(int(record["record_id"]))])
        with np.load(branch_path(wrong_id)) as wrong_payload:
            wrong = magnitude_center(pose_target(wrong_payload["endpoint_states"].astype(np.float64)), MAGNITUDE_COUNT).reshape(-1)
        matched_predicted.append(predicted); matched_truth.append(truth); wrong_truth.append(wrong)
    predicted = np.concatenate(matched_predicted); truth = np.concatenate(matched_truth); wrong = np.concatenate(wrong_truth)
    wrong_cosine = float(np.dot(predicted, wrong) / max(np.linalg.norm(predicted) * np.linalg.norm(wrong), 1e-12))
    exponent_errors = np.asarray([row["exponent_absolute_error"] for row in persistent], dtype=np.float64)
    record_cosines = np.asarray([row["cosine"] for row in persistent], dtype=np.float64)
    trajectories = np.asarray([row["trajectory_id"] for row in persistent])
    cosine_ci = bootstrap_interval(record_cosines, trajectories, "model_persistent_area_cosine")
    passed = bool(
        np.all(np.isfinite(record_cosines)) and cosine >= MIN_MODEL_PERSISTENT_AREA_COSINE
        and cosine - wrong_cosine >= MIN_MODEL_ALIGNMENT_GAIN_OVER_WRONG_STATE
        and np.median(exponent_errors) <= MAX_MODEL_EXPONENT_ABSOLUTE_ERROR
        and (cosine_ci[0] > 0 if RUN_MODE == "pilot" else True)
    )
    return {
        "persistent_records": len(persistent), "aggregate_coefficient": float(coefficient),
        "aggregate_cosine": float(cosine), "wrong_state_truth_cosine": wrong_cosine,
        "alignment_gain_over_wrong_state": float(cosine - wrong_cosine),
        "median_exponent_absolute_error": float(np.median(exponent_errors)),
        "mean_record_cosine": float(np.mean(record_cosines)), "record_cosine_ci95": cosine_ci,
        "passed": passed,
    }


def causal_area_gate(rows):
    trajectories = [int(record["trajectory_id"]) for record in ALL_EVALUATION_RECORDS]
    primary_name = f"primary_r{PRIMARY_RANK:03d}"
    primary = np.asarray([lookup(rows, value, primary_name, key="output_coefficient") for value in trajectories])
    primary_cosine = np.asarray([lookup(rows, value, primary_name, key="output_cosine") for value in trajectories])
    full = np.asarray([lookup(rows, value, "full_activation_swap", key="output_coefficient") for value in trajectories])
    shuffled = np.asarray([lookup(rows, value, f"shuffled_r{PRIMARY_RANK:03d}", key="output_coefficient") for value in trajectories])
    random_values = np.asarray([random_median(rows, value, PRIMARY_RANK, "output_coefficient") for value in trajectories])
    wrong = np.asarray([lookup(rows, value, f"wrong_state_r{PRIMARY_RANK:03d}", key="output_coefficient") for value in trajectories])
    gain_random = primary - random_values; gain_shuffled = primary - shuffled
    gain_ci = bootstrap_interval(gain_random, trajectories, "causal_area_sufficiency")
    gain_sign = exact_positive_sign_test(gain_random)
    positive_doses = sorted(value for value in ACTIVE_CAUSAL_DOSES if value > 0)
    dose_slopes = []
    for trajectory_id in trajectories:
        if len(positive_doses) < 2: dose_slopes.append(math.nan); continue
        values = np.asarray([lookup(rows, trajectory_id, primary_name, dose=value, key="output_coefficient") for value in positive_doses])
        dose_slopes.append(float(np.polyfit(positive_doses, values, 1)[0]))
    negative = np.asarray([lookup(rows, value, primary_name, dose=-0.5, key="output_coefficient") for value in trajectories]) if -0.5 in ACTIVE_CAUSAL_DOSES else np.full(len(trajectories), np.nan)
    necessity_key = "output_area_energy_reduction"
    necessity = np.asarray([lookup(rows, value, f"ablate_primary_r{PRIMARY_RANK:03d}", key=necessity_key) for value in trajectories])
    necessity_shuffled = np.asarray([lookup(rows, value, f"ablate_shuffled_r{PRIMARY_RANK:03d}", key=necessity_key) for value in trajectories])
    necessity_random = np.asarray([random_median(rows, value, PRIMARY_RANK, necessity_key, ablate=True) for value in trajectories])
    necessity_gain_random = necessity - necessity_random; necessity_gain_shuffled = necessity - necessity_shuffled
    necessity_ci = bootstrap_interval(necessity_gain_random, trajectories, "causal_area_necessity")
    necessity_sign = exact_positive_sign_test(necessity_gain_random)
    finite = bool(all(np.all(np.isfinite(value)) for value in [primary, primary_cosine, full, shuffled, random_values, wrong, necessity, necessity_shuffled, necessity_random]))
    sufficiency_pass = bool(
        finite and np.mean(full) >= MIN_FULL_SWAP_COEFFICIENT
        and np.mean(primary) >= MIN_PRIMARY_AREA_COEFFICIENT and np.mean(primary_cosine) >= MIN_PRIMARY_AREA_COSINE
        and np.mean(gain_random) >= MIN_PRIMARY_GAIN_OVER_RANDOM and np.mean(gain_shuffled) >= MIN_PRIMARY_GAIN_OVER_SHUFFLED
        and gain_sign["positive"] >= min(REQUIRED_POSITIVE_EVALUATION_TRAJECTORIES, len(trajectories))
        and (gain_sign["p_value"] <= 0.05 and gain_ci[0] > 0 if RUN_MODE == "pilot" else True)
        and (RUN_MODE == "smoke" or (np.sum(np.asarray(dose_slopes) > 0) >= REQUIRED_POSITIVE_EVALUATION_TRAJECTORIES and np.mean(negative) < 0))
    )
    necessity_pass = bool(
        finite and np.mean(necessity) >= MIN_NECESSITY_REDUCTION
        and np.mean(necessity_gain_random) >= MIN_NECESSITY_GAIN_OVER_RANDOM
        and np.mean(necessity_gain_shuffled) >= MIN_NECESSITY_GAIN_OVER_SHUFFLED
        and necessity_sign["positive"] >= min(REQUIRED_POSITIVE_NECESSITY_TRAJECTORIES, len(trajectories))
        and (necessity_sign["p_value"] <= 0.05 and necessity_ci[0] > 0 if RUN_MODE == "pilot" else True)
    )
    regime_means = {}
    for label in STRATUM_LABELS:
        ids = [int(record["trajectory_id"]) for record in ALL_EVALUATION_RECORDS if record["regime"] == label]
        regime_means[label] = {
            "primary_coefficient": float(np.mean([lookup(rows, value, primary_name, key="output_coefficient") for value in ids])),
            "necessity_reduction": float(np.mean([lookup(rows, value, f"ablate_primary_r{PRIMARY_RANK:03d}", key=necessity_key) for value in ids])),
        }
    return {
        "trajectories": len(trajectories), "all_required_metrics_finite": finite,
        "mean_primary_area_coefficient": float(np.mean(primary)), "mean_primary_area_cosine": float(np.mean(primary_cosine)),
        "mean_full_swap_coefficient": float(np.mean(full)), "mean_random_coefficient": float(np.mean(random_values)),
        "mean_shuffled_coefficient": float(np.mean(shuffled)), "mean_wrong_state_coefficient": float(np.mean(wrong)),
        "mean_gain_over_random": float(np.mean(gain_random)), "mean_gain_over_shuffled": float(np.mean(gain_shuffled)),
        "gain_over_random_ci95": gain_ci, "gain_over_random_sign_test": gain_sign,
        "positive_dose_slope_trajectories": int(np.sum(np.asarray(dose_slopes) > 0)),
        "negative_dose_mean": float(np.nanmean(negative)) if np.any(np.isfinite(negative)) else None,
        "mean_necessity_reduction": float(np.mean(necessity)), "mean_necessity_random_reduction": float(np.mean(necessity_random)),
        "mean_necessity_shuffled_reduction": float(np.mean(necessity_shuffled)),
        "mean_necessity_gain_over_random": float(np.mean(necessity_gain_random)),
        "mean_necessity_gain_over_shuffled": float(np.mean(necessity_gain_shuffled)),
        "necessity_gain_over_random_ci95": necessity_ci, "necessity_gain_over_random_sign_test": necessity_sign,
        "regime_means": regime_means, "sufficiency_gate_pass": sufficiency_pass,
        "necessity_gate_pass": necessity_pass, "passed": bool(sufficiency_pass and necessity_pass),
    }


def rank_sensitivity(rows):
    trajectories = [int(record["trajectory_id"]) for record in ALL_EVALUATION_RECORDS]
    results = []
    for rank in ACTIVE_SENSITIVITY_RANKS:
        name = f"primary_r{rank:03d}" if rank == PRIMARY_RANK else f"learned_r{rank:03d}"
        ablate = f"ablate_primary_r{rank:03d}" if rank == PRIMARY_RANK else f"ablate_learned_r{rank:03d}"
        results.append({
            "rank": int(rank),
            "mean_area_coefficient": float(np.mean([lookup(rows, value, name, key="output_coefficient") for value in trajectories])),
            "mean_area_necessity_reduction": float(np.mean([lookup(rows, value, ablate, key="output_area_energy_reduction") for value in trajectories])),
        })
    write_csv(ANALYSIS_DIR / "rank_sensitivity.csv", results)
    return results


def fresh_run_certificate():
    expected = {"truth_generated": len(POOL_SPECS), "baseline_generated": len(ALL_EVALUATION_RECORDS),
                "intervention_generated": len(ALL_EVALUATION_RECORDS), "cache_hits": 0}
    passed = bool(not OUT_PREEXISTED and PROVENANCE_COUNTS == expected)
    payload = {"out_preexisted": bool(OUT_PREEXISTED), "fresh_run_required": bool(FRESH_RUN_REQUIRED),
               "observed_counts": dict(PROVENANCE_COUNTS), "expected_counts": expected, "passed": passed}
    write_json(OUT / "fresh_run_certificate.json", payload)
    return payload


def make_plots(physical, model, causal, ranks):
    figure, axes = plt.subplots(1, 4, figsize=(20, 4.6))
    axes[0].bar(["persistent", "free"], [physical["median_persistent_max_area_contrast_norm"], physical["median_free_max_area_contrast_norm"]])
    axes[0].set(title="Physical signed-area effect", ylabel="median max-area contrast norm")
    axes[1].bar(["matched", "wrong state"], [model["aggregate_cosine"], model["wrong_state_truth_cosine"]])
    axes[1].axhline(MIN_MODEL_PERSISTENT_AREA_COSINE, color="black", linestyle="--")
    axes[1].set(title="Model–physics area alignment", ylabel="cosine")
    axes[2].bar(["learned", "random", "shuffled"], [causal["mean_primary_area_coefficient"], causal["mean_random_coefficient"], causal["mean_shuffled_coefficient"]])
    axes[2].set(title="Causal area reversal", ylabel="opposite-area coefficient")
    axes[3].bar([f"r{row['rank']}" for row in ranks], [row["mean_area_necessity_reduction"] for row in ranks])
    axes[3].set(title="Area-specific necessity", ylabel="antisymmetric energy reduction")
    figure.tight_layout(); figure.savefig(PLOT_DIR / "stage28_hybrid_control_area_summary.png", dpi=180); plt.close(figure)


if PIPELINE_FAILED:
    DECISION_PAYLOAD = {"status": "INCONCLUSIVE", "failure": FAILURE_MESSAGE}
elif not EVALUATION_OPENED:
    DECISION_PAYLOAD = {"status": "INCONCLUSIVE", "reason": "model evidence was not opened"}
else:
    try:
        PHYSICAL_GATE = physical_area_gate(); MODEL_GATE = model_area_gate()
        CAUSAL_GATE = causal_area_gate(INTERVENTION_ROWS); RANK_SENSITIVITY = rank_sensitivity(INTERVENTION_ROWS)
        FRESH_CERTIFICATE = fresh_run_certificate()
        if RUN_MODE == "smoke": candidate_status = "SMOKE_ONLY"
        elif not PHYSICAL_GATE["passed"]: candidate_status = "NO_PERSISTENT_CONTACT_CONTROL_AREA_LAW"
        elif not MODEL_GATE["passed"]: candidate_status = "MODEL_DOES_NOT_CAPTURE_PHYSICAL_CONTROL_AREA_LAW"
        elif not CAUSAL_GATE["passed"]: candidate_status = "CONTROL_AREA_EFFECT_NOT_MEDIATED_BY_FROZEN_ACTION_CARRIER"
        else: candidate_status = "HYBRID_CONTROL_AREA_LAW_CAUSALLY_ENCODED"
        confirmation_eligible = bool(
            SOURCE_IDENTITY.get("confirmation_eligible", False) and STAGE18_ARTIFACT_VALIDATED
            and STAGE27_UPSTREAM_BOUND and FRESH_CERTIFICATE["passed"]
        )
        status = candidate_status if RUN_MODE == "smoke" or confirmation_eligible else "UNBOUND_NONFRESH_OR_WRONG_UPSTREAM_EXPLORATORY_RESULT"
        DECISION_PAYLOAD = {
            "status": status, "candidate_status": candidate_status, "confirmation_eligible": confirmation_eligible,
            "source_bound_claim_eligible": bool(SOURCE_IDENTITY.get("confirmation_eligible", False)),
            "stage18_artifact_claim_eligible": bool(STAGE18_ARTIFACT_VALIDATED),
            "repaired_stage27_upstream_claim_eligible": bool(STAGE27_UPSTREAM_BOUND),
            "fresh_run_claim_eligible": FRESH_CERTIFICATE["passed"],
            "selected_magnitudes": SELECTED_MAGNITUDES, "magnitude_development_selection": MAGNITUDE_SELECTION,
            "physical_control_area_gate": PHYSICAL_GATE, "model_physical_control_area_gate": MODEL_GATE,
            "causal_carrier_control_area_gate": CAUSAL_GATE, "rank_sensitivity": RANK_SENSITIVITY,
            "claim_boundary": {
                "finite_signed_control_area_law_only": True, "infinitesimal_lie_bracket_established": False,
                "complete_action_histogram_impulse_energy_duration_matched_within_magnitude": True,
                "development_magnitude_selection_disjoint_from_confirmation": True,
                "contact_strata_selected_without_area_effect_magnitude_or_model_outputs": True,
                "stage18_subspace_refit_or_tuning": False, "coordinate_reader_used": False,
                "jacobian_jvp_vjp_or_gradient_used": False, "one_model_checkpoint": True,
                "one_environment": True, "generalization_to_other_models_or_environments": False,
                "causal_claim": "the frozen block-4 action carrier mediates the finite signed-area schedule effect only if all frozen gates pass",
            },
            "prespecified_next_step_if_positive": "replicate the signed-area exponent, contact-boundary departure, and frozen-carrier mediation across checkpoints and a second contact-rich environment",
        }
        write_json(OUT / "stage28_decision.json", DECISION_PAYLOAD)
        make_plots(PHYSICAL_GATE, MODEL_GATE, CAUSAL_GATE, RANK_SENSITIVITY)
        print(json.dumps(DECISION_PAYLOAD, indent=2))
    except Exception:
        record_failure("stage28_decision_and_plots")
        DECISION_PAYLOAD = {"status": "INCONCLUSIVE", "failure": FAILURE_MESSAGE}

if not (OUT / "stage28_decision.json").exists():
    write_json(OUT / "stage28_decision.json", DECISION_PAYLOAD)
'''


packaging = STAGE27.packaging
packaging = packaging.replace("stage27_action_commutator_result_bundle_", "stage28_hybrid_control_area_result_bundle_")


protocol_sources = [
    introduction, configuration, installation, setup, analysis_helpers,
    model_helpers, design, truth_generation, artifact_import,
    model_and_baselines, causal_interchange, decision_and_plots, packaging,
]
protocol_sources = [value.strip() for value in protocol_sources]
protocol_digest = hashlib.sha256(
    json.dumps(protocol_sources, ensure_ascii=False, separators=(",", ":")).encode()
).hexdigest()
configuration = configuration.replace("__PROTOCOL_DIGEST__", protocol_digest)
if "__PROTOCOL_DIGEST__" in configuration:
    raise RuntimeError("protocol digest placeholder was not replaced")

cells = [
    markdown(introduction), code(configuration), code(installation), code(setup),
    code(analysis_helpers), code(model_helpers), code(design), code(truth_generation),
    code(artifact_import), code(model_and_baselines), code(causal_interchange),
    code(decision_and_plots), code(packaging),
]
for index, cell in enumerate(cells):
    cell["id"] = f"stage28-{index:02d}"

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
