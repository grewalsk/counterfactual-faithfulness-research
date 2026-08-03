"""Build the source-bound Stage 15 longitudinal predictive-control notebook."""

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STAGE14 = ROOT / "14_predictive_control_j_bundle_pilot.ipynb"
NUMERICAL = ROOT.parent / "src/cf_faithfulness/stage15_bundle.py"
TARGET = ROOT / "15_longitudinal_predictive_control_bundle.ipynb"


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def function_sources(source, names):
    tree = ast.parse(source)
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            found[node.name] = ast.get_source_segment(source, node)
    missing = sorted(set(names) - set(found))
    if missing:
        raise RuntimeError(f"missing source functions: {missing}")
    return "\n\n\n".join(found[name] for name in names)


def assigned_uppercase_names(source):
    tree = ast.parse(source)
    return tuple(
        sorted(
            {
                node.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Store)
                and node.id.isupper()
            }
        )
    )


stage14 = json.loads(STAGE14.read_text())
stage14_setup = "".join(stage14["cells"][3]["source"])
stage14_analysis = "".join(stage14["cells"][4]["source"])
stage14_model = "".join(stage14["cells"][5]["source"])


introduction = r'''# Stage 15: longitudinal predictive-control bundle pilot

This notebook tests the strong statement left open by Stage 14:

> JEPA's local predictive-control spaces transport smoothly along physical
> trajectories and their transported modes causally control fixed physical
> predictions.

It removes the two central Stage 14 ambiguities. First, every state uses the
same six semantic reader coordinates: agent x/y, block x/y, and block
sin/cos. The readers are learned only from construction trajectories and
frozen before evaluation is opened. Second, every state uses the same
six-dimensional DCT-like action-tangent basis in normalized executable-action
coordinates. Therefore

\[
K_{s,h,l}=G_{s,h,l}B_{s,h,l}A_h
\]

has identical physical-reader rows and action columns at every state `s`,
horizon `h`, and predictor block `l`. Raw K is now a legitimate longitudinal
estimand. Procrustes alignment is used only to diagnose carrier-mode coordinate
rotation; it is never allowed to rotate K's semantic axes.

The pilot saves five evenly spaced states on each of eight deterministic PushT
trajectories, extracts exact JVP/VJP operators at horizons 1/3 and all six
predictor blocks, and causally reuses each source state's dominant write mode
at the next state. Controls include the destination-local mode, no edit,
equal-energy covariance-shaped directions, exact token-support-matched
directions, and nonadjacent time-shuffled source modes.

`smoke` mode validates plumbing only. `pilot` mode is scientifically eligible
only when run from an exact committed notebook prefix frozen before evaluation.
A positive result is checkpoint-specific evidence, not yet a cross-model
generality claim.
'''


configuration = r'''# SINGLE CONFIGURATION BLOCK
# Do not edit generated source. For the source-bound pilot, create Colab secrets
# STAGE15_RUN_MODE=pilot and STAGE15_SOURCE_COMMIT=<full 40-hex commit>.
RUN_MODE = "smoke"
EXPERIMENT_SOURCE_REF = ""
try:
    from google.colab import userdata as _colab_userdata

    RUN_MODE = _colab_userdata.get("STAGE15_RUN_MODE") or RUN_MODE
    EXPERIMENT_SOURCE_REF = (
        _colab_userdata.get("STAGE15_SOURCE_COMMIT") or EXPERIMENT_SOURCE_REF
    )
except Exception:
    pass

MOUNT_DRIVE = True
DOWNLOAD_RESULTS = True
CONTINUE_AFTER_BENCHMARK = True

OUTPUT_DIR = "/content/counterfactual_faithfulness_stage15_bundle"
DRIVE_OUTPUT_DIR = (
    "/content/drive/MyDrive/counterfactual_faithfulness_stage15_bundle"
)

PROTOCOL_ID = "stage15-fixed-reader-longitudinal-bundle-v1"
NOTEBOOK_PROTOCOL_SHA256 = "__PROTOCOL_DIGEST__"
EVIDENCE_STATUS = "EXPLORATORY_UNTIL_SOURCE_BOUND_BEFORE_EVALUATION"
EXPERIMENT_REPOSITORY = "grewalsk/counterfactual-faithfulness-research"
EXPERIMENT_NOTEBOOK_PATH = "notebooks/15_longitudinal_predictive_control_bundle.ipynb"
EXPERIMENT_BUILDER_PATH = "notebooks/build_stage15_longitudinal_bundle_notebook.py"

SEED = 15101
DESIGN_SEED = 15137
MODEL_NAME = "jepa_wm_pusht"
ENVIRONMENT = "PushT"
FRAMESKIP = 5
HORIZONS = [1, 3]
TARGET_STEPS = HORIZONS
PRIMARY_HORIZON = 3
PRIMARY_BLOCK = 3
PREDICTOR_BLOCKS = [0, 1, 2, 3, 4, 5]
EXPECTED_CARRIER_CHANNELS = 400
READ_BRANCH = 0

TOTAL_TRAJECTORIES = 8
CONSTRUCTION_TRAJECTORIES = [0, 2, 4, 6]
EVALUATION_TRAJECTORIES = [1, 3, 5, 7]
STATES_PER_TRAJECTORY = 5
LONGITUDINAL_SAVE_STEPS = [0, 5, 10, 15, 20]
FIXED_GOAL = [256.0, 256.0, 0.0]
TASK_ID_OFFSET = 300

ACTION_PROFILES = 3
ACTION_BASIS_DIM = 6
ACTION_TANGENT_NORM = 0.35
ACTIONS_PER_STATE = 1 + 2 * ACTION_BASIS_DIM
READER_LABELS = [
    "agent_x", "agent_y", "block_x", "block_y", "block_sin", "block_cos"
]
READER_PROJECTION_SEEDS = [15161, 15173, 15187]
READER_PROJECTION_DIM = 192
READER_RIDGES = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

CHANNEL_SHRINKAGE = 0.10
CHANNEL_EIGEN_FLOOR = 1e-6
BALANCED_TOLERANCE = 1e-4
JVP_EPSILON = 1e-3
JVP_EPSILON_CHECK = [0.5e-3, 1e-3, 2e-3]
MAX_METRIC_INVARIANCE_ERROR = 2e-6
MAX_ADJOINT_RELATIVE_ERROR = 1e-3
MAX_ADJOINT_ABS_ERROR = 1e-5

PERMUTATION_SEED = 15211
NULL_ROOT_SEED = 15233
BOOTSTRAP_SEED = 15251
CAUSAL_DOSE = 0.5
CAUSAL_NULL_DRAWS = 8
PERMUTATION_DRAWS = 5000

# Frozen pilot gates. Smoke mode can never authorize a scientific claim.
MIN_READER_MEDIAN_R2 = 0.25
MIN_READER_SPATIAL_R2 = 0.10
MIN_GEOMETRY_RHO = 0.25
MAX_GEOMETRY_PERMUTATION_P = 0.05
MIN_ADJACENT_K_COSINE = 0.50
MIN_CAUSAL_LINEARITY_COSINE = 0.80
MIN_TRANSPORT_RECOVERY = 0.50
MIN_TRANSPORT_NULL_ADVANTAGE = 0.05
MIN_TRANSPORT_SEMANTIC_COSINE = 0.40
MAX_ZERO_EDIT_ERROR = 1e-6
REQUIRED_POSITIVE_CAUSAL_TRAJECTORIES = 4

if RUN_MODE == "smoke":
    ACTIVE_CONSTRUCTION_TRAJECTORIES = CONSTRUCTION_TRAJECTORIES[:2]
    ACTIVE_EVALUATION_TRAJECTORIES = EVALUATION_TRAJECTORIES[:2]
    ACTIVE_TIME_INDICES = [0, 2, 4]
    ACTIVE_HORIZONS = [3]
    ACTIVE_BLOCKS = [PRIMARY_BLOCK]
    ACTIVE_CAUSAL_NULL_DRAWS = 2
    ACTIVE_PERMUTATION_DRAWS = 64
elif RUN_MODE == "pilot":
    ACTIVE_CONSTRUCTION_TRAJECTORIES = CONSTRUCTION_TRAJECTORIES
    ACTIVE_EVALUATION_TRAJECTORIES = EVALUATION_TRAJECTORIES
    ACTIVE_TIME_INDICES = list(range(STATES_PER_TRAJECTORY))
    ACTIVE_HORIZONS = HORIZONS
    ACTIVE_BLOCKS = PREDICTOR_BLOCKS
    ACTIVE_CAUSAL_NULL_DRAWS = CAUSAL_NULL_DRAWS
    ACTIVE_PERMUTATION_DRAWS = PERMUTATION_DRAWS
else:
    raise ValueError("RUN_MODE must be 'smoke' or 'pilot'")

REPO_URL = "https://github.com/facebookresearch/jepa-wms.git"
REPO_COMMIT = "13cf1d9c7e476f53c17714d2e0f1dc239a883ce0"
EXPECTED_HF_REVISION = "9b9c41ef249466630dbf1a20e78391865d07b3b9"
EXPECTED_PRETRAINED_ASSET_SHA256 = {
    "jepa_wm_pusht.pth.tar": (
        "9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb"
    ),
    "dinov2_vits14_pretrain.pth": (
        "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
    ),
}
ASSET_SPECS = {}

assert PRIMARY_HORIZON in ACTIVE_HORIZONS
assert PRIMARY_BLOCK in ACTIVE_BLOCKS
assert len(READER_LABELS) == 6
assert ACTIONS_PER_STATE == 13
assert not set(CONSTRUCTION_TRAJECTORIES) & set(EVALUATION_TRAJECTORIES)
'''
configuration_keys = assigned_uppercase_names(configuration)
configuration = (
    configuration.rstrip()
    + "\n\nPROTOCOL_CONFIG_KEYS = "
    + repr(configuration_keys)
    + "\n"
)


installation = "".join(stage14["cells"][2]["source"])
installation = installation.replace(
    '    "scikit-learn==1.6.1",',
    '    "scikit-learn==1.6.1",\n    "scikit-image==0.24.0",',
)


setup = (
    stage14_setup.replace("Stage 14", "Stage 15")
    .replace("STAGE14", "STAGE15")
    .replace("stage14_pcj", "stage15_bundle")
    .replace("stage14", "stage15")
)


analysis_helpers = function_sources(
    stage14_analysis,
    [
        "array_sha256",
        "channel_metric_from_moments",
        "transform_primal_channels",
        "inverse_transform_primal_channels",
        "transform_dual_channels",
        "balanced_modes",
        "relative_error",
        "CountSketchProjector",
        "stable_cosine",
        "manifest_rows",
    ],
)
analysis_helpers += "\n\n\n" + function_sources(
    NUMERICAL.read_text(),
    [
        "physical_reader_targets",
        "temporal_action_basis",
        "fit_ridge",
        "predict_ridge",
        "grouped_ridge_cv",
        "r2_per_output",
        "orthonormal_columns",
        "principal_angle_cosines",
        "chordal_subspace_distance",
        "procrustes_align",
        "align_basis_sequence",
        "matrix_cosine",
        "normalized_matrix_distance",
        "_average_ranks",
        "spearman_correlation",
        "grouped_label_permutation",
        "energy_map",
        "support_matched_random",
    ],
)
analysis_helpers += r'''


def norm_match(candidate, reference):
    value = np.asarray(candidate, dtype=np.float64)
    target = np.asarray(reference, dtype=np.float64)
    return value * (np.linalg.norm(target) / max(np.linalg.norm(value), 1e-12))


# Pure CPU identities execute before model or simulator access.
_rng = np.random.default_rng(151)
_basis = temporal_action_basis(15, ACTION_PROFILES)
if not np.allclose(_basis.T @ _basis, np.eye(ACTION_BASIS_DIM), atol=1e-12):
    raise AssertionError("fixed action basis CPU identity failed")
_template = _rng.normal(size=256 * EXPECTED_CARRIER_CHANNELS)
_matched = support_matched_random(
    _template, 153, channels=EXPECTED_CARRIER_CHANNELS
)
if not np.allclose(
    energy_map(_template, channels=EXPECTED_CARRIER_CHANNELS),
    energy_map(_matched, channels=EXPECTED_CARRIER_CHANNELS),
):
    raise AssertionError("support-matched null CPU identity failed")
'''


model_helpers = function_sources(
    stage14_model,
    [
        "to_model_observation",
        "configure_repo",
        "pose_target",
        "make_environment",
        "wall_visual",
        "reset_environment",
        "rollout_branch",
        "exact_restore_test",
        "verify_pretrained_assets",
        "validate_jepa_predictor",
        "load_frozen_model",
        "model_action_tensor",
        "layer_tokens_full",
        "forward_with_carriers",
    ],
)
model_helpers = model_helpers.replace("stage14-jepa-wms", "stage15-jepa-wms")


design = r'''# Freeze trajectory paths, common action coordinates, and all null schedules.


def trajectory_specs():
    specs = []
    center = np.asarray([256.0, 256.0])
    for trajectory_id in range(TOTAL_TRAJECTORIES):
        block_angle = 0.17 + 2.0 * np.pi * trajectory_id / TOTAL_TRAJECTORIES
        block = center + 52.0 * np.asarray(
            [np.cos(block_angle), np.sin(block_angle)]
        )
        agent_index = (3 * trajectory_id + 1) % TOTAL_TRAJECTORIES
        agent_angle = 0.41 + 2.0 * np.pi * agent_index / TOTAL_TRAJECTORIES
        agent = center + 125.0 * np.asarray(
            [np.cos(agent_angle), np.sin(agent_angle)]
        )
        direction = block - agent
        direction /= max(np.linalg.norm(direction), 1e-12)
        perpendicular = np.asarray([-direction[1], direction[0]])
        controls = []
        for step in range(1, LONGITUDINAL_SAVE_STEPS[-1] + 1):
            fraction = step / LONGITUDINAL_SAVE_STEPS[-1]
            curve = (
                (1.0 if trajectory_id % 2 == 0 else -1.0)
                * 0.22
                * np.sin(2.0 * np.pi * fraction)
                * perpendicular
            )
            action = 0.09 * (direction + curve)
            if np.linalg.norm(action) > 0.14:
                raise RuntimeError("longitudinal relative action is too large")
            controls.append(action)
        split = (
            "construction"
            if trajectory_id in CONSTRUCTION_TRAJECTORIES
            else "evaluation"
        )
        specs.append(
            {
                "trajectory_id": trajectory_id,
                "task_id": TASK_ID_OFFSET + trajectory_id,
                "split": split,
                "seed": DESIGN_SEED + 1009 * trajectory_id,
                "initial_state": np.asarray(
                    [
                        agent[0], agent[1], block[0], block[1],
                        ((1.3 * block_angle + np.pi) % (2 * np.pi)) - np.pi,
                        0.0, 0.0,  # agent velocity
                        0.0, 0.0,  # block velocity
                        0.0,       # block angular velocity
                    ],
                    dtype=np.float64,
                ),
                "controls": np.asarray(controls, dtype=np.float64),
            }
        )
    return specs


ALL_TRAJECTORY_SPECS = trajectory_specs()
CONSTRUCTION_SPECS = [
    row for row in ALL_TRAJECTORY_SPECS
    if row["trajectory_id"] in ACTIVE_CONSTRUCTION_TRAJECTORIES
]
EVALUATION_SPECS = [
    row for row in ALL_TRAJECTORY_SPECS
    if row["trajectory_id"] in ACTIVE_EVALUATION_TRAJECTORIES
]

initials = np.stack([row["initial_state"] for row in ALL_TRAJECTORY_SPECS])
if abs(np.corrcoef(initials[:, 0], initials[:, 2])[0, 1]) > 0.05:
    raise AssertionError("agent/block x sampling is unexpectedly coupled")
if abs(np.corrcoef(initials[:, 1], initials[:, 3])[0, 1]) > 0.05:
    raise AssertionError("agent/block y sampling is unexpectedly coupled")

NULL_RNG = np.random.default_rng(NULL_ROOT_SEED)
NULL_SEEDS = NULL_RNG.integers(
    0,
    np.iinfo(np.uint32).max,
    size=(
        TOTAL_TRAJECTORIES,
        STATES_PER_TRAJECTORY - 1,
        max(CAUSAL_NULL_DRAWS, ACTIVE_CAUSAL_NULL_DRAWS),
        2,
    ),
    dtype=np.uint32,
)

np.savez_compressed(
    DESIGN_DIR / "stage15_design.npz",
    trajectory_ids=np.asarray([row["trajectory_id"] for row in ALL_TRAJECTORY_SPECS]),
    splits=np.asarray([row["split"] for row in ALL_TRAJECTORY_SPECS]),
    initial_states=initials,
    controls=np.stack([row["controls"] for row in ALL_TRAJECTORY_SPECS]),
    save_steps=np.asarray(LONGITUDINAL_SAVE_STEPS),
    null_seeds=NULL_SEEDS,
)
write_json(
    DESIGN_DIR / "trajectory_design_manifest.json",
    {
        "specs": [
            {
                **{key: value for key, value in row.items() if key not in {"initial_state", "controls"}},
                "initial_state": row["initial_state"].tolist(),
                "controls": row["controls"].tolist(),
            }
            for row in ALL_TRAJECTORY_SPECS
        ],
        "active_construction_trajectories": ACTIVE_CONSTRUCTION_TRAJECTORIES,
        "active_evaluation_trajectories": ACTIVE_EVALUATION_TRAJECTORIES,
        "active_time_indices": ACTIVE_TIME_INDICES,
        "active_horizons": ACTIVE_HORIZONS,
        "active_blocks": ACTIVE_BLOCKS,
        "reader_labels": READER_LABELS,
        "reader_projection_seeds": READER_PROJECTION_SEEDS,
        "reader_ridges": READER_RIDGES,
        "fixed_goal": FIXED_GOAL,
    },
)
DESIGN_FREEZE = {
    "created_before_simulator_or_model_data": True,
    "protocol_id": PROTOCOL_ID,
    "run_signature": RUN_SIGNATURE,
    "source_identity": SOURCE_IDENTITY,
    "design_sha256": sha256_file(DESIGN_DIR / "stage15_design.npz"),
    "manifest_sha256": sha256_file(
        DESIGN_DIR / "trajectory_design_manifest.json"
    ),
    "construction_trajectories": ACTIVE_CONSTRUCTION_TRAJECTORIES,
    "evaluation_trajectories": ACTIVE_EVALUATION_TRAJECTORIES,
    "state_specific_reader_forbidden": True,
    "global_frame_search_forbidden": True,
}
freeze_path = DESIGN_DIR / "design_freeze.json"
if freeze_path.exists() and json.loads(freeze_path.read_text()) != DESIGN_FREEZE:
    raise RuntimeError("existing Stage 15 design freeze differs")
write_json(freeze_path, DESIGN_FREEZE)
print(json.dumps(DESIGN_FREEZE, indent=2))
'''


truth_generation = r'''# Realize construction trajectories and exact future-action branches.


def record_task(record_or_spec):
    return {
        "environment": ENVIRONMENT,
        "task_id": int(record_or_spec["task_id"]),
        "goal": list(FIXED_GOAL),
    }


def dynamic_state_from_environment(environment):
    return np.asarray(
        [
            *environment.agent.position,
            *environment.block.position,
            float(environment.block.angle),
            *environment.agent.velocity,
            *environment.block.velocity,
            float(environment.block.angular_velocity),
        ],
        dtype=np.float64,
    )


def reset_dynamic_environment(dynamic_state, task, seed):
    state = np.asarray(dynamic_state, dtype=np.float64)
    if state.shape != (10,):
        raise ValueError(f"expected ten-dimensional dynamic state, found {state.shape}")
    environment = make_environment(REPO, ENVIRONMENT)
    environment.seed(int(seed))
    environment.reset_to_state = np.asarray(
        [*state[:5], 0.0, 0.0], dtype=np.float64
    )
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


def rollout_dynamic_branch(record, actions):
    environment, initial = reset_dynamic_environment(
        record["state"], record_task(record), record["evaluation_seed"]
    )
    observations = {}
    states = {}
    counts = {}
    kinds = {}
    cumulative = 0
    try:
        for step, action in enumerate(actions, start=1):
            observation, _, _, info = environment.step(action)
            cumulative += int(info.get("n_contacts", 0))
            if step % FRAMESKIP == 0:
                horizon = step // FRAMESKIP
                if horizon in TARGET_STEPS:
                    observations[horizon] = {
                        "visual": np.asarray(observation["visual"]).copy(),
                        "proprio": np.asarray(observation["proprio"]).copy(),
                    }
                    states[horizon] = dynamic_state_from_environment(environment)
                    counts[horizon] = cumulative
                    kinds[horizon] = "contact" if cumulative > 0 else "free"
    finally:
        environment.close()
    if set(observations) != set(TARGET_STEPS):
        raise RuntimeError("dynamic rollout missed a target horizon")
    return initial, observations, states, counts, kinds


def exact_dynamic_restore_test(record):
    first, first_observation = reset_dynamic_environment(
        record["state"], record_task(record), record["evaluation_seed"]
    )
    first_state = dynamic_state_from_environment(first)
    second, second_observation = reset_dynamic_environment(
        record["state"], record_task(record), record["evaluation_seed"]
    )
    second_state = dynamic_state_from_environment(second)
    test_action = candidate_action_bank(record["state"])[1, 0]
    first.step(test_action)
    second.step(test_action)
    first_next = dynamic_state_from_environment(first)
    second_next = dynamic_state_from_environment(second)
    first.close()
    second.close()
    result = {
        "state_exact": bool(np.allclose(first_state, second_state, atol=1e-12, rtol=0)),
        "visual_exact": bool(
            np.array_equal(first_observation["visual"], second_observation["visual"])
        ),
        "proprio_exact": bool(
            np.array_equal(first_observation["proprio"], second_observation["proprio"])
        ),
        "one_step_continuation_exact": bool(
            np.allclose(first_next, second_next, atol=1e-12, rtol=0)
        ),
    }
    result["passed"] = bool(all(result.values()))
    if not result["passed"]:
        raise RuntimeError(f"full dynamic restore test failed: {result}")
    return result


def trajectory_path(spec):
    return TRUTH_DIR / f"trajectory_{int(spec['trajectory_id']):02d}.npz"


def realize_trajectory(spec):
    destination = trajectory_path(spec)
    if destination.exists():
        with np.load(destination) as payload:
            return [
                {
                    "record_id": int(spec["trajectory_id"]) * STATES_PER_TRAJECTORY + index,
                    "trajectory_id": int(spec["trajectory_id"]),
                    "time_index": index,
                    "physical_step": int(LONGITUDINAL_SAVE_STEPS[index]),
                    "task_id": int(spec["task_id"]),
                    "split": spec["split"],
                    "evaluation_seed": int(spec["seed"]),
                    "state": payload["states"][index].astype(np.float64),
                    "path_contacts": int(payload["cumulative_contacts"][index]),
                }
                for index in ACTIVE_TIME_INDICES
            ]
    environment, _ = reset_dynamic_environment(
        spec["initial_state"], record_task(spec), int(spec["seed"])
    )
    states = [dynamic_state_from_environment(environment)]
    contacts = [0]
    cumulative = 0
    try:
        for step, action in enumerate(spec["controls"], start=1):
            result = environment.step(action)
            if len(result) == 4:
                _, _, _, info = result
            elif len(result) == 5:
                _, _, _, _, info = result
            else:
                raise RuntimeError(f"unexpected PushT step result of length {len(result)}")
            cumulative += int(info.get("n_contacts", 0))
            if step in LONGITUDINAL_SAVE_STEPS[1:]:
                states.append(dynamic_state_from_environment(environment))
                contacts.append(cumulative)
    finally:
        environment.close()
    if len(states) != STATES_PER_TRAJECTORY:
        raise RuntimeError("longitudinal trajectory missed a frozen save point")
    atomic_npz(
        destination,
        trajectory_id=np.asarray(spec["trajectory_id"], dtype=np.int64),
        split=np.asarray(spec["split"]),
        states=np.stack(states).astype(np.float64),
        controls=np.asarray(spec["controls"], dtype=np.float64),
        save_steps=np.asarray(LONGITUDINAL_SAVE_STEPS, dtype=np.int64),
        cumulative_contacts=np.asarray(contacts, dtype=np.int64),
    )
    return realize_trajectory(spec)


def realize_records(specs):
    records = []
    for spec in specs:
        records.extend(realize_trajectory(spec))
    return records


RAW_ACTION_BASIS = temporal_action_basis(
    max(HORIZONS) * FRAMESKIP, ACTION_PROFILES
)


def candidate_action_bank(state):
    state = np.asarray(state, dtype=np.float64)
    if state.shape[-1] < 5:
        raise ValueError("candidate state is not a PushT state")
    baseline = np.zeros((max(HORIZONS) * FRAMESKIP, 2), dtype=np.float64)
    branches = [baseline]
    for column in range(ACTION_BASIS_DIM):
        delta = (
            ACTION_TANGENT_NORM
            * RAW_ACTION_BASIS[:, column].reshape(-1, 2)
        )
        branches.extend([baseline + delta, baseline - delta])
    actions = np.stack(branches)
    if actions.shape != (
        ACTIONS_PER_STATE,
        max(HORIZONS) * FRAMESKIP,
        2,
    ):
        raise RuntimeError(f"bad fixed action bank shape {actions.shape}")
    if np.max(np.abs(actions)) > 0.14:
        raise RuntimeError("fixed relative action perturbation exceeds safe magnitude")
    return actions.astype(np.float32)


def branch_path(record_id):
    return TRUTH_DIR / f"state_{int(record_id):04d}.npz"


def generate_truth(records):
    started = time.perf_counter()
    for record_index, record in enumerate(records):
        destination = branch_path(record["record_id"])
        if destination.exists():
            continue
        selected_actions = candidate_action_bank(record["state"])
        initials = []
        initial_proprios = []
        future_visual = []
        future_proprio = []
        endpoint_states = []
        interaction_counts = []
        interaction_types = []
        for action in selected_actions:
            initial, observations, states, counts, kinds = rollout_dynamic_branch(
                record, action
            )
            initials.append(initial["visual"])
            initial_proprios.append(initial["proprio"])
            future_visual.append([observations[h]["visual"] for h in HORIZONS])
            future_proprio.append([observations[h]["proprio"] for h in HORIZONS])
            endpoint_states.append([states[h] for h in HORIZONS])
            interaction_counts.append([counts[h] for h in HORIZONS])
            interaction_types.append([kinds[h] for h in HORIZONS])
        if not all(np.array_equal(initials[0], value) for value in initials[1:]):
            raise AssertionError("initial visual drift across action branches")
        if not all(
            np.array_equal(initial_proprios[0], value)
            for value in initial_proprios[1:]
        ):
            raise AssertionError("initial proprio drift across action branches")
        atomic_npz(
            destination,
            record_id=np.asarray(record["record_id"], dtype=np.int64),
            trajectory_id=np.asarray(record["trajectory_id"], dtype=np.int64),
            time_index=np.asarray(record["time_index"], dtype=np.int64),
            physical_step=np.asarray(record["physical_step"], dtype=np.int64),
            split=np.asarray(record["split"]),
            initial_state=np.asarray(record["state"], dtype=np.float64),
            initial_visual=np.asarray(initials[0], dtype=np.uint8),
            initial_proprio=np.asarray(initial_proprios[0], dtype=np.float32),
            selected_actions=selected_actions,
            future_visual=np.asarray(future_visual, dtype=np.uint8),
            future_proprio=np.asarray(future_proprio, dtype=np.float32),
            endpoint_states=np.asarray(endpoint_states, dtype=np.float32),
            interaction_counts=np.asarray(interaction_counts, dtype=np.int32),
            interaction_types=np.asarray(interaction_types),
        )
        write_json(
            OUT / f"{record['split']}_truth_progress.json",
            {
                "completed": record_index + 1,
                "total": len(records),
                "last_record_id": record["record_id"],
            },
        )
    TIMINGS[f"{records[0]['split']}_truth_seconds"] = (
        time.perf_counter() - started
    )


CONSTRUCTION_RECORDS = []
EVALUATION_RECORDS = []
ALL_RECORDS = []
RECORD_BY_ID = {}
if not PIPELINE_FAILED:
    try:
        REPO = configure_repo()
        CONSTRUCTION_RECORDS = realize_records(CONSTRUCTION_SPECS)
        restore_record = CONSTRUCTION_RECORDS[0]
        RESTORE_TEST = exact_dynamic_restore_test(restore_record)
        write_json(OUT / "restore_test.json", RESTORE_TEST)
        generate_truth(CONSTRUCTION_RECORDS)
        memory_report("construction_truth_complete")
    except Exception:
        record_failure("construction_truth_generation")
'''


fixed_readers = r'''# Fit and freeze fixed physical readers before opening evaluation trajectories.


def target_path(record_id):
    return TARGET_DIR / f"state_{int(record_id):04d}.npz"


def encode_target_cache(records):
    started = time.perf_counter()
    for index, record in enumerate(records):
        destination = target_path(record["record_id"])
        if destination.exists():
            continue
        with np.load(branch_path(record["record_id"])) as truth:
            visual = truth["future_visual"]
            proprio = truth["future_proprio"]
        with torch.inference_mode():
            encoded = MODEL.encode(to_model_observation(visual, proprio))
        tokens = (
            encoded["visual"][:, :, 0]
            .reshape(
                ACTIONS_PER_STATE,
                len(HORIZONS),
                256,
                encoded["visual"].shape[-1],
            )
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        if tokens.shape != (ACTIONS_PER_STATE, len(HORIZONS), 256, 384):
            raise RuntimeError(f"unexpected target-token shape {tokens.shape}")
        atomic_npz(destination, true_tokens=tokens.astype(np.float32))
        write_json(
            OUT / f"{record['split']}_target_progress.json",
            {
                "completed": index + 1,
                "total": len(records),
                "last_record_id": record["record_id"],
            },
        )
    TIMINGS[f"{records[0]['split']}_target_encoding_seconds"] = (
        time.perf_counter() - started
    )


def load_target_tokens(record_id):
    with np.load(target_path(record_id)) as payload:
        return payload["true_tokens"].astype(np.float32)


def state_model_inputs(record_id, horizon):
    with np.load(branch_path(record_id)) as truth:
        initial_visual = truth["initial_visual"]
        initial_proprio = truth["initial_proprio"]
        selected_actions = truth["selected_actions"]
    with torch.inference_mode():
        initial = MODEL.encode(to_model_observation(initial_visual, initial_proprio))
    initial = {name: value.detach() for name, value in initial.items()}
    actions = model_action_tensor(PREPROCESSOR, selected_actions, horizon)
    return initial, actions


def reader_feature_chunks(records, projectors):
    features = [[] for _ in projectors]
    targets = []
    groups = []
    for record in records:
        tokens = load_target_tokens(record["record_id"])
        with np.load(branch_path(record["record_id"])) as truth:
            endpoints = truth["endpoint_states"].astype(np.float64)
        flattened = torch.as_tensor(
            tokens.reshape(-1, 256, 384), device="cuda", dtype=torch.float32
        )
        with torch.inference_mode():
            for projector_index, projector in enumerate(projectors):
                features[projector_index].append(
                    projector(flattened).detach().cpu().numpy()
                )
        targets.append(physical_reader_targets(endpoints).reshape(-1, 6))
        groups.append(
            np.full(
                ACTIONS_PER_STATE * len(HORIZONS),
                record["trajectory_id"],
                dtype=np.int64,
            )
        )
    return (
        [np.concatenate(values, axis=0) for values in features],
        np.concatenate(targets, axis=0),
        np.concatenate(groups, axis=0),
    )


def fit_fixed_readers(records):
    projectors = [
        CountSketchProjector(256 * 384, READER_PROJECTION_DIM, seed)
        for seed in READER_PROJECTION_SEEDS
    ]
    feature_sets, physical, groups = reader_feature_chunks(records, projectors)
    target_mean = physical.mean(axis=0)
    target_scale = physical.std(axis=0)
    if np.any(target_scale <= 1e-5):
        raise RuntimeError(f"degenerate construction physical target: {target_scale}")
    standardized_target = (physical - target_mean) / target_scale
    models = []
    cv_rows = []
    for seed, features in zip(READER_PROJECTION_SEEDS, feature_sets):
        result = grouped_ridge_cv(
            features,
            standardized_target,
            groups,
            READER_RIDGES,
        )
        model = result["model"]
        models.append(model)
        for row in result["cv_rows"]:
            cv_rows.append({"projection_seed": seed, **row})
    destination = ANALYSIS_DIR / "fixed_physical_readers.npz"
    atomic_npz(
        destination,
        projection_seeds=np.asarray(READER_PROJECTION_SEEDS, dtype=np.int64),
        feature_mean=np.stack([row["feature_mean"] for row in models]),
        feature_scale=np.stack([row["feature_scale"] for row in models]),
        intercept=np.stack([row["intercept"] for row in models]),
        coefficient=np.stack([row["coefficient"] for row in models]),
        ridge=np.asarray([row["ridge"] for row in models]),
        target_mean=target_mean,
        target_scale=target_scale,
        reader_labels=np.asarray(READER_LABELS),
    )
    write_csv(ANALYSIS_DIR / "reader_construction_cv.csv", cv_rows)
    payload = {
        "path": str(destination),
        "sha256": sha256_file(destination),
        "fit_trajectories": sorted({int(value) for value in groups}),
        "evaluation_trajectories_seen": [],
        "selected_ridges": [float(row["ridge"]) for row in models],
        "reader_labels": READER_LABELS,
        "target_mean": target_mean.tolist(),
        "target_scale": target_scale.tolist(),
        "state_specific_inputs": False,
        "goal_task_action_inputs": "none",
        "frozen_before_evaluation": True,
    }
    write_json(ANALYSIS_DIR / "reader_freeze.json", payload)
    return payload


def load_fixed_readers():
    with np.load(ANALYSIS_DIR / "fixed_physical_readers.npz") as payload:
        arrays = {name: payload[name].copy() for name in payload.files}
    projectors = [
        CountSketchProjector(256 * 384, READER_PROJECTION_DIM, int(seed))
        for seed in arrays["projection_seeds"]
    ]
    return arrays, projectors


def decode_fixed_physical(tokens):
    outputs = []
    for index, projector in enumerate(READER_PROJECTORS):
        features = projector(tokens)
        mean = torch.as_tensor(
            READER_ARRAYS["feature_mean"][index],
            device=tokens.device,
            dtype=tokens.dtype,
        )
        scale = torch.as_tensor(
            READER_ARRAYS["feature_scale"][index],
            device=tokens.device,
            dtype=tokens.dtype,
        )
        intercept = torch.as_tensor(
            READER_ARRAYS["intercept"][index],
            device=tokens.device,
            dtype=tokens.dtype,
        )
        coefficient = torch.as_tensor(
            READER_ARRAYS["coefficient"][index],
            device=tokens.device,
            dtype=tokens.dtype,
        )
        outputs.append(intercept + ((features - mean) / scale) @ coefficient)
    return torch.stack(outputs, dim=0).mean(dim=0)


def evaluate_fixed_readers(records, split):
    feature_sets, physical, _ = reader_feature_chunks(records, READER_PROJECTORS)
    target = (
        physical - READER_ARRAYS["target_mean"]
    ) / READER_ARRAYS["target_scale"]
    estimates = []
    for index, features in enumerate(feature_sets):
        model = {
            "feature_mean": READER_ARRAYS["feature_mean"][index],
            "feature_scale": READER_ARRAYS["feature_scale"][index],
            "intercept": READER_ARRAYS["intercept"][index],
            "coefficient": READER_ARRAYS["coefficient"][index],
        }
        estimates.append(predict_ridge(model, features))
    prediction = np.mean(estimates, axis=0)
    r2 = r2_per_output(target, prediction)
    rows = [
        {"split": split, "reader": label, "r2": float(value)}
        for label, value in zip(READER_LABELS, r2)
    ]
    write_csv(ANALYSIS_DIR / f"reader_{split}_metrics.csv", rows)
    result = {
        "split": split,
        "r2": {label: float(value) for label, value in zip(READER_LABELS, r2)},
        "median_r2": float(np.median(r2)),
        "minimum_spatial_r2": float(np.min(r2[:4])),
    }
    write_json(ANALYSIS_DIR / f"reader_{split}_gate.json", result)
    return result


def common_action_basis(record_id, horizon):
    _, actions = state_model_inputs(record_id, horizon)
    plus_indices = [1 + 2 * index for index in range(ACTION_BASIS_DIM)]
    base = actions[:, READ_BRANCH].detach().cpu().numpy().reshape(-1)
    columns = []
    for action_index in plus_indices:
        value = actions[:, action_index].detach().cpu().numpy().reshape(-1) - base
        columns.append(value)
    raw = np.stack(columns, axis=1).astype(np.float64)
    basis, triangular = np.linalg.qr(raw, mode="reduced")
    signs = np.sign(np.diag(triangular))
    signs[signs == 0] = 1.0
    basis *= signs[None]
    if basis.shape != (horizon * FRAMESKIP * 2, ACTION_BASIS_DIM):
        raise RuntimeError(f"bad common action basis shape {basis.shape}")
    return basis


def hook_identity_test(record_id):
    initial, actions = state_model_inputs(record_id, PRIMARY_HORIZON)
    with torch.inference_mode():
        baseline, _, captures = forward_with_carriers(
            initial,
            actions[:, :1],
            PRIMARY_HORIZON,
            capture_blocks=[PRIMARY_BLOCK],
        )
        zero = torch.zeros_like(layer_tokens_full(captures[PRIMARY_BLOCK]))
        hooked, _, _ = forward_with_carriers(
            initial,
            actions[:, :1],
            PRIMARY_HORIZON,
            capture_blocks=[PRIMARY_BLOCK],
            intervention={"block": PRIMARY_BLOCK, "delta": zero},
        )
    error = float(torch.max(torch.abs(baseline - hooked)).cpu())
    result = {"zero_edit_max_abs_error": error, "passed": error <= MAX_ZERO_EDIT_ERROR}
    write_json(ANALYSIS_DIR / "hook_identity.json", result)
    if not result["passed"]:
        raise RuntimeError(f"hook identity failed: {result}")
    return result


READER_GATE = {}
COMMON_ACTION_BASES = {}
if not PIPELINE_FAILED:
    try:
        MODEL, PREPROCESSOR = load_frozen_model()
        encode_target_cache(CONSTRUCTION_RECORDS)
        READER_FREEZE = fit_fixed_readers(CONSTRUCTION_RECORDS)
        READER_ARRAYS, READER_PROJECTORS = load_fixed_readers()
        CONSTRUCTION_READER = evaluate_fixed_readers(
            CONSTRUCTION_RECORDS, "construction"
        )
        COMMON_ACTION_BASES = {
            horizon: common_action_basis(
                CONSTRUCTION_RECORDS[0]["record_id"], horizon
            )
            for horizon in ACTIVE_HORIZONS
        }
        HOOK_IDENTITY = hook_identity_test(CONSTRUCTION_RECORDS[0]["record_id"])
        verify_executed_notebook_through(
            "# Fit and freeze fixed physical readers before opening evaluation trajectories."
        )
        EVALUATION_RECORDS = realize_records(EVALUATION_SPECS)
        generate_truth(EVALUATION_RECORDS)
        encode_target_cache(EVALUATION_RECORDS)
        EVALUATION_READER = evaluate_fixed_readers(
            EVALUATION_RECORDS, "evaluation"
        )
        READER_GATE = {
            **EVALUATION_READER,
            "median_threshold": MIN_READER_MEDIAN_R2,
            "spatial_threshold": MIN_READER_SPATIAL_R2,
            "passed": bool(
                EVALUATION_READER["median_r2"] >= MIN_READER_MEDIAN_R2
                and EVALUATION_READER["minimum_spatial_r2"]
                >= MIN_READER_SPATIAL_R2
            ),
        }
        write_json(ANALYSIS_DIR / "reader_evaluation_gate.json", READER_GATE)
        if not READER_GATE["passed"]:
            raise RuntimeError(f"fixed physical reader gate failed: {READER_GATE}")
        ALL_RECORDS = CONSTRUCTION_RECORDS + EVALUATION_RECORDS
        RECORD_BY_ID = {int(row["record_id"]): row for row in ALL_RECORDS}
        for record in ALL_RECORDS:
            for horizon in ACTIVE_HORIZONS:
                observed = common_action_basis(record["record_id"], horizon)
                if not np.allclose(
                    observed, COMMON_ACTION_BASES[horizon], atol=1e-6, rtol=1e-6
                ):
                    raise RuntimeError("action coordinates changed across states")
        memory_report("fixed_reader_evaluation_complete")
    except Exception:
        record_failure("fixed_reader_freeze_or_evaluation")
'''


carrier_metrics = r'''# Fit one construction-only hidden metric per predictor block.


def scan_carrier_moments(record, horizon):
    initial, actions = state_model_inputs(record["record_id"], horizon)
    with torch.inference_mode():
        _, _, captures = forward_with_carriers(
            initial,
            actions,
            horizon,
            capture_blocks=ACTIVE_BLOCKS,
        )
    rows = {}
    for block in ACTIVE_BLOCKS:
        activation = layer_tokens_full(captures[block]).detach().double()
        channels = activation.reshape(-1, EXPECTED_CARRIER_CHANNELS)
        rows[block] = {
            "count": int(len(channels)),
            "sum": channels.sum(dim=0).cpu().numpy(),
            "cross": (channels.T @ channels).cpu().numpy(),
        }
    return rows


def fit_channel_metrics(records):
    aggregate = {
        block: {
            "count": 0,
            "sum": np.zeros(EXPECTED_CARRIER_CHANNELS, dtype=np.float64),
            "cross": np.zeros(
                (EXPECTED_CARRIER_CHANNELS, EXPECTED_CARRIER_CHANNELS),
                dtype=np.float64,
            ),
        }
        for block in ACTIVE_BLOCKS
    }
    started = time.perf_counter()
    for index, record in enumerate(records):
        for horizon in ACTIVE_HORIZONS:
            payload = scan_carrier_moments(record, horizon)
            for block, values in payload.items():
                aggregate[block]["count"] += values["count"]
                aggregate[block]["sum"] += values["sum"]
                aggregate[block]["cross"] += values["cross"]
        write_json(
            OUT / "carrier_metric_progress.json",
            {"completed": index + 1, "total": len(records)},
        )
    metrics = {}
    rows = []
    for block in ACTIVE_BLOCKS:
        moments = aggregate[block]
        metric = channel_metric_from_moments(
            moments["count"],
            moments["sum"],
            moments["cross"],
            shrinkage=CHANNEL_SHRINKAGE,
            relative_floor=CHANNEL_EIGEN_FLOOR,
        )
        destination = ANALYSIS_DIR / f"channel_metric_block_{block}.npz"
        atomic_npz(destination, **metric)
        metrics[block] = metric
        rows.append(
            {
                "block": block,
                "count": moments["count"],
                "condition_number": metric["condition_number"],
                "sha256": sha256_file(destination),
            }
        )
    write_csv(ANALYSIS_DIR / "channel_metric_manifest.csv", rows)
    write_json(
        ANALYSIS_DIR / "channel_metric_freeze.json",
        {
            "fit_split": "construction_only",
            "fit_trajectories": ACTIVE_CONSTRUCTION_TRAJECTORIES,
            "evaluation_activations_seen": False,
            "rows": rows,
        },
    )
    TIMINGS["channel_metric_seconds"] = time.perf_counter() - started
    return metrics


CHANNEL_METRICS = {}
if not PIPELINE_FAILED:
    try:
        CHANNEL_METRICS = fit_channel_metrics(CONSTRUCTION_RECORDS)
        memory_report("channel_metrics_frozen")
    except Exception:
        record_failure("channel_metric_fit")
'''


operator_extraction = r'''# Extract fixed-reader operators at every saved trajectory state and layer.


def operator_path(record_id, horizon):
    return JACOBIAN_DIR / f"state_{int(record_id):04d}_h{int(horizon)}.npz"


def fixed_reader_gradients_all_blocks(record_id, horizon):
    initial, actions = state_model_inputs(record_id, horizon)
    predictions, _, captures = forward_with_carriers(
        initial,
        actions[:, :1],
        horizon,
        capture_blocks=ACTIVE_BLOCKS,
        require_grad=True,
    )
    decoded = decode_fixed_physical(predictions)
    if decoded.shape != (1, len(READER_LABELS)):
        raise RuntimeError(f"bad fixed-reader output shape {tuple(decoded.shape)}")
    capture_values = [captures[block] for block in ACTIVE_BLOCKS]
    gradients = {block: [] for block in ACTIVE_BLOCKS}
    for reader_index in range(len(READER_LABELS)):
        values = torch.autograd.grad(
            decoded[0, reader_index],
            capture_values,
            retain_graph=reader_index + 1 < len(READER_LABELS),
            allow_unused=False,
            create_graph=False,
        )
        for block, value in zip(ACTIVE_BLOCKS, values):
            gradients[block].append(
                layer_tokens_full(value)[0].detach().float().cpu().numpy()
            )
    return {
        block: np.stack(values).reshape(len(READER_LABELS), -1)
        for block, values in gradients.items()
    }


def exact_action_tangent_jacobians_all_blocks(record_id, horizon):
    initial, actions = state_model_inputs(record_id, horizon)
    base = actions[:, :1].detach()
    shape = tuple(base.shape)
    carrier_size = 256 * EXPECTED_CARRIER_CHANNELS
    action_basis = COMMON_ACTION_BASES[horizon]

    def capture_function(flat_action):
        action = flat_action.reshape(shape)
        _, _, captures = forward_with_carriers(
            initial,
            action,
            horizon,
            capture_blocks=ACTIVE_BLOCKS,
            require_grad=True,
        )
        return torch.cat(
            [
                layer_tokens_full(captures[block])[0].reshape(-1)
                for block in ACTIVE_BLOCKS
            ]
        )

    flat = base.reshape(-1).detach().requires_grad_(True)
    columns = []
    used_fallback = False
    for direction_index in range(ACTION_BASIS_DIM):
        tangent = torch.as_tensor(
            action_basis[:, direction_index],
            device=flat.device,
            dtype=flat.dtype,
        )
        try:
            _, value = torch.autograd.functional.jvp(
                capture_function,
                (flat,),
                (tangent,),
                strict=True,
                create_graph=False,
            )
        except (RuntimeError, NotImplementedError):
            used_fallback = True
            plus = capture_function((flat + JVP_EPSILON * tangent).detach())
            minus = capture_function((flat - JVP_EPSILON * tangent).detach())
            value = (plus - minus) / (2.0 * JVP_EPSILON)
        columns.append(value.detach().float().cpu())
    combined = torch.stack(columns, dim=1).numpy()
    expected = len(ACTIVE_BLOCKS) * carrier_size
    if combined.shape != (expected, ACTION_BASIS_DIM):
        raise RuntimeError(
            f"bad all-block action-tangent Jacobian shape {combined.shape}"
        )
    matrices = {
        block: combined[
            index * carrier_size : (index + 1) * carrier_size
        ]
        for index, block in enumerate(ACTIVE_BLOCKS)
    }
    return matrices, used_fallback


def transform_block_operators(block, g_raw, b_raw):
    metric = CHANNEL_METRICS[block]
    g_values = np.asarray(g_raw, dtype=np.float64).reshape(
        len(READER_LABELS), 256, EXPECTED_CARRIER_CHANNELS
    )
    g = transform_dual_channels(
        g_values, metric["square_root"]
    ).reshape(len(READER_LABELS), -1)
    b_shape = b_raw.shape
    b_values = np.asarray(b_raw, dtype=np.float64).reshape(
        256, EXPECTED_CARRIER_CHANNELS, b_shape[1]
    ).transpose(0, 2, 1)
    b = transform_primal_channels(
        b_values, metric["inverse_square_root"]
    ).transpose(0, 2, 1).reshape(-1, b_shape[1])
    return g, b


def build_operator_shard(record, horizon):
    destination = operator_path(record["record_id"], horizon)
    if destination.exists():
        return
    g_raw = fixed_reader_gradients_all_blocks(record["record_id"], horizon)
    b_raw, used_fallback = exact_action_tangent_jacobians_all_blocks(
        record["record_id"], horizon
    )
    action_basis = COMMON_ACTION_BASES[horizon]
    all_g = []
    all_b = []
    all_k = []
    all_singular = []
    all_read_energy = []
    all_write_energy = []
    all_metric_error = []
    for block in ACTIVE_BLOCKS:
        g, b = transform_block_operators(block, g_raw[block], b_raw[block])
        k = g @ b
        native_k = (
            np.asarray(g_raw[block], dtype=np.float64)
            @ np.asarray(b_raw[block], dtype=np.float64)
        )
        error = relative_error(k, native_k)
        if error > MAX_METRIC_INVARIANCE_ERROR:
            raise RuntimeError(
                f"hidden metric changed fixed K for state {record['record_id']} "
                f"block {block}: {error}"
            )
        balanced = balanced_modes(g, b, tolerance=BALANCED_TOLERANCE)
        singular = np.zeros(ACTION_BASIS_DIM, dtype=np.float64)
        retained = min(ACTION_BASIS_DIM, len(balanced["singular_values"]))
        if retained:
            singular[:retained] = balanced["singular_values"][:retained]
        all_g.append(g)
        all_b.append(b)
        all_k.append(k)
        all_singular.append(singular)
        all_read_energy.append(
            energy_map(g, channels=EXPECTED_CARRIER_CHANNELS)
        )
        all_write_energy.append(
            energy_map(b.T, channels=EXPECTED_CARRIER_CHANNELS)
        )
        all_metric_error.append(error)
    atomic_npz(
        destination,
        record_id=np.asarray(record["record_id"], dtype=np.int64),
        trajectory_id=np.asarray(record["trajectory_id"], dtype=np.int64),
        time_index=np.asarray(record["time_index"], dtype=np.int64),
        physical_step=np.asarray(record["physical_step"], dtype=np.int64),
        split=np.asarray(record["split"]),
        horizon=np.asarray(horizon, dtype=np.int64),
        blocks=np.asarray(ACTIVE_BLOCKS, dtype=np.int64),
        reader_labels=np.asarray(READER_LABELS),
        action_basis=action_basis.astype(np.float32),
        G=np.stack(all_g).astype(np.float32),
        B_action=np.stack(all_b).astype(np.float32),
        K=np.stack(all_k).astype(np.float64),
        singular_values=np.stack(all_singular).astype(np.float64),
        read_energy=np.stack(all_read_energy).astype(np.float32),
        write_energy=np.stack(all_write_energy).astype(np.float32),
        metric_invariance_error=np.asarray(all_metric_error, dtype=np.float64),
        jvp_fallback=np.asarray(used_fallback),
    )


def load_operator_shard(record_id, horizon):
    with np.load(operator_path(record_id, horizon)) as payload:
        return {name: payload[name].copy() for name in payload.files}


def direct_chain_identity(record_id):
    horizon = PRIMARY_HORIZON
    initial, actions = state_model_inputs(record_id, horizon)
    base = actions[:, :1].detach()
    shape = tuple(base.shape)
    rng = np.random.default_rng(SEED + 91)
    coefficients = rng.normal(size=ACTION_BASIS_DIM)
    coefficients /= np.linalg.norm(coefficients)
    tangent_numpy = COMMON_ACTION_BASES[horizon] @ coefficients
    tangent = torch.as_tensor(
        tangent_numpy.reshape(shape), device="cuda", dtype=base.dtype
    )

    def decoded_function(action):
        prediction, _, _ = forward_with_carriers(
            initial, action, horizon, capture_blocks=[], require_grad=True
        )
        return decode_fixed_physical(prediction)[0]

    _, direct = torch.autograd.functional.jvp(
        decoded_function,
        (base,),
        (tangent,),
        strict=True,
        create_graph=False,
    )
    shard = load_operator_shard(record_id, horizon)
    block_index = list(shard["blocks"]).index(PRIMARY_BLOCK)
    chained = shard["K"][block_index] @ coefficients
    direct_numpy = direct.detach().double().cpu().numpy()
    absolute_error = float(np.linalg.norm(direct_numpy - chained))
    relative = relative_error(chained, direct_numpy)
    result = {
        "direct_reader_action_jvp": direct_numpy.tolist(),
        "g_b_a_chain": chained.tolist(),
        "absolute_error": absolute_error,
        "relative_error": relative,
        "passed": bool(
            relative <= MAX_ADJOINT_RELATIVE_ERROR
            or absolute_error <= MAX_ADJOINT_ABS_ERROR
        ),
    }
    write_json(ANALYSIS_DIR / "direct_chain_identity.json", result)
    if not result["passed"]:
        raise RuntimeError(f"direct fixed-reader chain identity failed: {result}")
    return result


def run_operator_extraction(records):
    started = time.perf_counter()
    for index, record in enumerate(records):
        for horizon in ACTIVE_HORIZONS:
            build_operator_shard(record, horizon)
        write_json(
            OUT / f"{record['split']}_operator_progress.json",
            {"completed": index + 1, "total": len(records)},
        )
    TIMINGS[f"{records[0]['split']}_operator_seconds"] = (
        time.perf_counter() - started
    )


if not PIPELINE_FAILED:
    try:
        benchmark_record = CONSTRUCTION_RECORDS[0]
        started = time.perf_counter()
        build_operator_shard(benchmark_record, PRIMARY_HORIZON)
        benchmark_seconds = time.perf_counter() - started
        BENCHMARK = {
            "one_state_horizon_all_active_blocks_seconds": benchmark_seconds,
            "estimated_total_minutes": (
                benchmark_seconds
                * len(ALL_RECORDS)
                * len(ACTIVE_HORIZONS)
                / 60.0
            ),
            "active_states": len(ALL_RECORDS),
            "active_horizons": ACTIVE_HORIZONS,
            "active_blocks": ACTIVE_BLOCKS,
            "reader_dimensions": len(READER_LABELS),
            "action_basis_dimensions": ACTION_BASIS_DIM,
            "jvp_directions_per_state_horizon": ACTION_BASIS_DIM,
            "continue_after_benchmark": CONTINUE_AFTER_BENCHMARK,
        }
        write_json(OUT / "benchmark.json", BENCHMARK)
        print(json.dumps(BENCHMARK, indent=2))
        if not CONTINUE_AFTER_BENCHMARK:
            raise RuntimeError(
                "Benchmark complete. Set CONTINUE_AFTER_BENCHMARK=True to continue."
            )
        run_operator_extraction(CONSTRUCTION_RECORDS)
        DIRECT_CHAIN_IDENTITY = direct_chain_identity(
            benchmark_record["record_id"]
        )
        run_operator_extraction(EVALUATION_RECORDS)
        memory_report("all_longitudinal_operators_complete")
    except Exception:
        record_failure("longitudinal_operator_extraction")
'''


longitudinal_analysis = r'''# Test longitudinal smoothness in fixed semantic and action coordinates.


def block_payload(shard, block):
    index = list(shard["blocks"]).index(block)
    g = shard["G"][index].astype(np.float64)
    b = shard["B_action"][index].astype(np.float64)
    balanced = balanced_modes(g, b, tolerance=BALANCED_TOLERANCE)
    return {
        "G": g,
        "B": b,
        "K": shard["K"][index].astype(np.float64),
        "primal": balanced["primal"],
        "dual": balanced["dual"],
        "singular": shard["singular_values"][index].astype(np.float64),
        "read_energy": shard["read_energy"][index].astype(np.float64),
        "write_energy": shard["write_energy"][index].astype(np.float64),
    }


def cosine_distance(left, right):
    return 1.0 - matrix_cosine(left, right)


def physical_state_coordinate(record):
    value = physical_reader_targets(record["state"])
    return (
        value - READER_ARRAYS["target_mean"]
    ) / READER_ARRAYS["target_scale"]


GEOMETRY_CACHE = {}
PAIR_ROWS = []
LAYER_CHAIN_ROWS = []
PROCRUSTES_ROWS = []


def build_geometry_cache():
    for split, records in [
        ("construction", CONSTRUCTION_RECORDS),
        ("evaluation", EVALUATION_RECORDS),
    ]:
        trajectory_ids = sorted({row["trajectory_id"] for row in records})
        for horizon in ACTIVE_HORIZONS:
            for trajectory_id in trajectory_ids:
                trajectory = sorted(
                    [row for row in records if row["trajectory_id"] == trajectory_id],
                    key=lambda row: row["time_index"],
                )
                shards = [
                    load_operator_shard(row["record_id"], horizon)
                    for row in trajectory
                ]
                physical = np.stack(
                    [physical_state_coordinate(row) for row in trajectory]
                )
                for state_index, (record, shard) in enumerate(zip(trajectory, shards)):
                    if len(ACTIVE_BLOCKS) > 1:
                        matrices = [block_payload(shard, block)["K"] for block in ACTIVE_BLOCKS]
                        reference = matrices[0]
                        LAYER_CHAIN_ROWS.append(
                            {
                                "split": split,
                                "trajectory_id": trajectory_id,
                                "time_index": record["time_index"],
                                "horizon": horizon,
                                "maximum_cross_block_k_relative_error": max(
                                    normalized_matrix_distance(reference, value)
                                    for value in matrices[1:]
                                ),
                            }
                        )
                for block in ACTIVE_BLOCKS:
                    payloads = [block_payload(shard, block) for shard in shards]
                    count = len(trajectory)
                    matrices = {
                        name: np.zeros((count, count), dtype=np.float64)
                        for name in [
                            "physical_distance",
                            "k_distance",
                            "read_subspace_distance",
                            "write_subspace_distance",
                            "read_energy_distance",
                            "write_energy_distance",
                        ]
                    }
                    for left in range(count):
                        for right in range(left + 1, count):
                            values = {
                                "physical_distance": float(
                                    np.linalg.norm(physical[left] - physical[right])
                                ),
                                "k_distance": normalized_matrix_distance(
                                    payloads[left]["K"], payloads[right]["K"]
                                ),
                                "read_subspace_distance": chordal_subspace_distance(
                                    payloads[left]["G"].T,
                                    payloads[right]["G"].T,
                                    rank=len(READER_LABELS),
                                ),
                                "write_subspace_distance": chordal_subspace_distance(
                                    payloads[left]["B"],
                                    payloads[right]["B"],
                                    rank=ACTION_BASIS_DIM,
                                ),
                                "read_energy_distance": cosine_distance(
                                    payloads[left]["read_energy"],
                                    payloads[right]["read_energy"],
                                ),
                                "write_energy_distance": cosine_distance(
                                    payloads[left]["write_energy"],
                                    payloads[right]["write_energy"],
                                ),
                            }
                            for name, value in values.items():
                                matrices[name][left, right] = value
                                matrices[name][right, left] = value
                            PAIR_ROWS.append(
                                {
                                    "split": split,
                                    "horizon": horizon,
                                    "block": block,
                                    "trajectory_id": trajectory_id,
                                    "left_time": trajectory[left]["time_index"],
                                    "right_time": trajectory[right]["time_index"],
                                    "temporal_gap": trajectory[right]["time_index"] - trajectory[left]["time_index"],
                                    "adjacent": int(right == left + 1),
                                    "k_cosine": matrix_cosine(
                                        payloads[left]["K"], payloads[right]["K"]
                                    ),
                                    **values,
                                }
                            )
                    GEOMETRY_CACHE[(split, horizon, block, trajectory_id)] = {
                        "records": trajectory,
                        "payloads": payloads,
                        "matrices": matrices,
                    }
                    if horizon == PRIMARY_HORIZON and block == PRIMARY_BLOCK:
                        retained = min(
                            int(np.sum(value["singular"] > max(value["singular"][0] * BALANCED_TOLERANCE, 1e-12)))
                            for value in payloads
                        )
                        retained = max(retained, 1)
                        for kind in ["primal", "dual"]:
                            bases = [value[kind][:, :retained] for value in payloads]
                            aligned = align_basis_sequence(bases, rank=retained)
                            for index in range(1, len(aligned)):
                                raw_left = orthonormal_columns(bases[index - 1], rank=retained)
                                raw_right = orthonormal_columns(bases[index], rank=retained)
                                PROCRUSTES_ROWS.append(
                                    {
                                        "split": split,
                                        "trajectory_id": trajectory_id,
                                        "source_time": trajectory[index - 1]["time_index"],
                                        "destination_time": trajectory[index]["time_index"],
                                        "kind": kind,
                                        "rank": retained,
                                        "raw_basis_cosine": matrix_cosine(raw_left, raw_right),
                                        "aligned_basis_cosine": matrix_cosine(
                                            aligned[index - 1], aligned[index]
                                        ),
                                        "subspace_distance": chordal_subspace_distance(
                                            bases[index - 1], bases[index], rank=retained
                                        ),
                                    }
                                )


def permutation_geometry_summary():
    rows = []
    metric_names = [
        "k_distance",
        "read_subspace_distance",
        "write_subspace_distance",
        "read_energy_distance",
        "write_energy_distance",
    ]
    for split, records in [
        ("construction", CONSTRUCTION_RECORDS),
        ("evaluation", EVALUATION_RECORDS),
    ]:
        trajectory_ids = sorted({row["trajectory_id"] for row in records})
        for horizon in ACTIVE_HORIZONS:
            for block in ACTIVE_BLOCKS:
                caches = [
                    GEOMETRY_CACHE[(split, horizon, block, trajectory_id)]
                    for trajectory_id in trajectory_ids
                ]
                upper = [
                    np.triu_indices(len(cache["records"]), k=1)
                    for cache in caches
                ]
                physical = np.concatenate(
                    [
                        cache["matrices"]["physical_distance"][indices]
                        for cache, indices in zip(caches, upper)
                    ]
                )
                for metric_index, metric_name in enumerate(metric_names):
                    observed_values = np.concatenate(
                        [
                            cache["matrices"][metric_name][indices]
                            for cache, indices in zip(caches, upper)
                        ]
                    )
                    observed = spearman_correlation(physical, observed_values)
                    rng = np.random.default_rng(
                        PERMUTATION_SEED
                        + 10000 * (0 if split == "construction" else 1)
                        + 1000 * horizon
                        + 100 * block
                        + metric_index
                    )
                    null = np.empty(ACTIVE_PERMUTATION_DRAWS, dtype=np.float64)
                    for draw in range(ACTIVE_PERMUTATION_DRAWS):
                        shuffled = []
                        for cache, indices in zip(caches, upper):
                            permutation = rng.permutation(len(cache["records"]))
                            matrix = cache["matrices"][metric_name]
                            permuted = matrix[np.ix_(permutation, permutation)]
                            shuffled.append(permuted[indices])
                        null[draw] = spearman_correlation(
                            physical, np.concatenate(shuffled)
                        )
                    p_value = float(
                        (1 + np.sum(null >= observed)) / (len(null) + 1)
                    )
                    rows.append(
                        {
                            "split": split,
                            "horizon": horizon,
                            "block": block,
                            "metric": metric_name,
                            "spearman_physical_distance": observed,
                            "permutation_p": p_value,
                            "null_95": float(np.quantile(null, 0.95)),
                            "draws": len(null),
                        }
                    )
    return rows


GEOMETRY_SUMMARY = []
GEOMETRY_GATE = {}
if not PIPELINE_FAILED:
    try:
        build_geometry_cache()
        GEOMETRY_SUMMARY = permutation_geometry_summary()
        write_csv(ANALYSIS_DIR / "longitudinal_pair_metrics.csv", PAIR_ROWS)
        write_csv(ANALYSIS_DIR / "longitudinal_geometry_summary.csv", GEOMETRY_SUMMARY)
        write_csv(ANALYSIS_DIR / "carrier_procrustes_diagnostics.csv", PROCRUSTES_ROWS)
        write_csv(ANALYSIS_DIR / "cross_block_chain_consistency.csv", LAYER_CHAIN_ROWS)
        primary = {
            row["metric"]: row
            for row in GEOMETRY_SUMMARY
            if row["split"] == "evaluation"
            and row["horizon"] == PRIMARY_HORIZON
            and row["block"] == PRIMARY_BLOCK
        }
        adjacent_k = [
            row["k_cosine"]
            for row in PAIR_ROWS
            if row["split"] == "evaluation"
            and row["horizon"] == PRIMARY_HORIZON
            and row["block"] == PRIMARY_BLOCK
            and row["adjacent"] == 1
        ]
        required = [
            "k_distance", "read_subspace_distance", "write_subspace_distance"
        ]
        GEOMETRY_GATE = {
            "primary_rows": {name: primary[name] for name in required},
            "median_adjacent_k_cosine": float(np.median(adjacent_k)),
            "minimum_rho": MIN_GEOMETRY_RHO,
            "maximum_permutation_p": MAX_GEOMETRY_PERMUTATION_P,
            "minimum_adjacent_k_cosine": MIN_ADJACENT_K_COSINE,
            "passed": bool(
                all(
                    primary[name]["spearman_physical_distance"] >= MIN_GEOMETRY_RHO
                    and primary[name]["permutation_p"] <= MAX_GEOMETRY_PERMUTATION_P
                    for name in required
                )
                and np.median(adjacent_k) >= MIN_ADJACENT_K_COSINE
            ),
        }
        write_json(ANALYSIS_DIR / "longitudinal_geometry_gate.json", GEOMETRY_GATE)
        memory_report("longitudinal_geometry_complete")
    except Exception:
        record_failure("longitudinal_geometry")
'''


causal_transport = r'''# Causally reuse source-state modes at neighboring held-out states.


def dominant_write(payload):
    _, singular, vh = np.linalg.svd(payload["K"], full_matrices=False)
    if not len(singular) or singular[0] <= 1e-10:
        raise RuntimeError("degenerate local fixed-coordinate K")
    action_coefficients = vh[0]
    direction = payload["B"] @ action_coefficients
    effect = payload["G"] @ direction
    return {
        "direction": direction,
        "effect": effect,
        "action_coefficients": action_coefficients,
        "singular_value": float(singular[0]),
    }


def whitened_to_native(block, direction):
    value = np.asarray(direction, dtype=np.float64).reshape(
        256, EXPECTED_CARRIER_CHANNELS
    )
    return inverse_transform_primal_channels(
        value, CHANNEL_METRICS[block]["square_root"]
    ).reshape(-1)


def native_norm_match_whitened(block, candidate, reference):
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    candidate_native = whitened_to_native(block, candidate)
    reference_native = whitened_to_native(block, reference)
    scale = np.linalg.norm(reference_native) / max(
        np.linalg.norm(candidate_native), 1e-12
    )
    return candidate * scale


def causal_direction_responses(record, payload, conditions):
    directions = np.stack(
        [np.asarray(direction, dtype=np.float64) for _, direction, _ in conditions]
    )
    predicted = directions @ payload["G"].T
    native = np.stack(
        [whitened_to_native(PRIMARY_BLOCK, direction) for direction in directions]
    ).reshape(len(conditions), 256, EXPECTED_CARRIER_CHANNELS)
    delta = torch.as_tensor(native, device="cuda", dtype=torch.float32)
    initial, actions = state_model_inputs(record["record_id"], PRIMARY_HORIZON)
    base_actions = actions[:, :1]
    batched_actions = base_actions.expand(
        -1, len(conditions), -1
    ).contiguous()
    with torch.inference_mode():
        baseline_tokens, _, _ = forward_with_carriers(
            initial,
            base_actions,
            PRIMARY_HORIZON,
            capture_blocks=[PRIMARY_BLOCK],
        )
        plus_tokens, _, _ = forward_with_carriers(
            initial,
            batched_actions,
            PRIMARY_HORIZON,
            capture_blocks=[PRIMARY_BLOCK],
            intervention={
                "block": PRIMARY_BLOCK,
                "delta": CAUSAL_DOSE * delta,
            },
        )
        minus_tokens, _, _ = forward_with_carriers(
            initial,
            batched_actions,
            PRIMARY_HORIZON,
            capture_blocks=[PRIMARY_BLOCK],
            intervention={
                "block": PRIMARY_BLOCK,
                "delta": -CAUSAL_DOSE * delta,
            },
        )
        baseline = decode_fixed_physical(baseline_tokens)[0]
        plus = decode_fixed_physical(plus_tokens)
        minus = decode_fixed_physical(minus_tokens)
    observed = ((plus - minus) / (2.0 * CAUSAL_DOSE)).cpu().numpy()
    nonlinear = torch.max(
        torch.abs((plus + minus) / 2.0 - baseline[None]), dim=1
    ).values.cpu().numpy()
    results = []
    for index, (label, _direction, null_draw) in enumerate(conditions):
        results.append(
            {
                "label": label,
                "null_draw": int(null_draw),
                "predicted_effect": predicted[index],
                "observed_effect": observed[index],
                "predicted_norm": float(np.linalg.norm(predicted[index])),
                "observed_norm": float(np.linalg.norm(observed[index])),
                "linearity_cosine": matrix_cosine(
                    predicted[index], observed[index]
                ),
                "linearity_relative_error": relative_error(
                    observed[index], predicted[index]
                ),
                "central_nonlinearity": float(nonlinear[index]),
                "native_patch_norm": float(np.linalg.norm(native[index])),
            }
        )
    return results


def zero_edit_check(record):
    initial, actions = state_model_inputs(record["record_id"], PRIMARY_HORIZON)
    zero = torch.zeros(
        (1, 256, EXPECTED_CARRIER_CHANNELS),
        device="cuda",
        dtype=torch.float32,
    )
    with torch.inference_mode():
        baseline, _, _ = forward_with_carriers(
            initial,
            actions[:, :1],
            PRIMARY_HORIZON,
            capture_blocks=[PRIMARY_BLOCK],
        )
        edited, _, _ = forward_with_carriers(
            initial,
            actions[:, :1],
            PRIMARY_HORIZON,
            capture_blocks=[PRIMARY_BLOCK],
            intervention={"block": PRIMARY_BLOCK, "delta": zero},
        )
    return float(torch.max(torch.abs(baseline - edited)).cpu())


def causal_transport_rows():
    rows = []
    transition_rows = []
    for trajectory_id in sorted(
        {row["trajectory_id"] for row in EVALUATION_RECORDS}
    ):
        trajectory = sorted(
            [
                row
                for row in EVALUATION_RECORDS
                if row["trajectory_id"] == trajectory_id
            ],
            key=lambda row: row["time_index"],
        )
        payloads = [
            block_payload(
                load_operator_shard(row["record_id"], PRIMARY_HORIZON),
                PRIMARY_BLOCK,
            )
            for row in trajectory
        ]
        modes = [dominant_write(payload) for payload in payloads]
        for source_index in range(len(trajectory) - 1):
            destination_index = source_index + 1
            source = trajectory[source_index]
            destination = trajectory[destination_index]
            destination_payload = payloads[destination_index]
            local = modes[destination_index]["direction"]
            transported = native_norm_match_whitened(
                PRIMARY_BLOCK, modes[source_index]["direction"], local
            )
            farthest_index = int(
                np.argmax(
                    [
                        abs(index - destination_index)
                        if index != source_index else -1
                        for index in range(len(trajectory))
                    ]
                )
            )
            time_shuffled = native_norm_match_whitened(
                PRIMARY_BLOCK, modes[farthest_index]["direction"], local
            )
            conditions = [
                ("local_positive", local, -1),
                ("transported_neighbor", transported, -1),
                ("time_shuffled", time_shuffled, -1),
            ]
            for draw in range(ACTIVE_CAUSAL_NULL_DRAWS):
                support_seed = int(
                    NULL_SEEDS[trajectory_id, source_index, draw, 0]
                )
                covariance_seed = int(
                    NULL_SEEDS[trajectory_id, source_index, draw, 1]
                )
                support = support_matched_random(
                    modes[source_index]["direction"],
                    support_seed,
                    channels=EXPECTED_CARRIER_CHANNELS,
                )
                support = native_norm_match_whitened(
                    PRIMARY_BLOCK, support, local
                )
                rng = np.random.default_rng(covariance_seed)
                covariance = rng.normal(size=len(local))
                covariance = native_norm_match_whitened(
                    PRIMARY_BLOCK, covariance, local
                )
                conditions.extend(
                    [
                        ("support_matched_null", support, draw),
                        ("covariance_shaped_null", covariance, draw),
                    ]
                )
            condition_results = causal_direction_responses(
                destination, destination_payload, conditions
            )
            for result in condition_results:
                label = result["label"]
                if label == "transported_neighbor":
                    source_effect = modes[source_index]["effect"]
                    result["source_destination_semantic_cosine"] = matrix_cosine(
                        source_effect, result["predicted_effect"]
                    )
                else:
                    result["source_destination_semantic_cosine"] = math.nan
                condition_results.append(result)
                rows.append(
                    {
                        "trajectory_id": trajectory_id,
                        "source_time": source["time_index"],
                        "destination_time": destination["time_index"],
                        "source_record_id": source["record_id"],
                        "destination_record_id": destination["record_id"],
                        **{
                            key: value
                            for key, value in result.items()
                            if key not in {"predicted_effect", "observed_effect"}
                        },
                        "predicted_effect": json.dumps(
                            result["predicted_effect"].tolist()
                        ),
                        "observed_effect": json.dumps(
                            result["observed_effect"].tolist()
                        ),
                    }
                )
            by_label = defaultdict(list)
            for result in condition_results:
                by_label[result["label"]].append(result)
            local_result = by_label["local_positive"][0]
            transport_result = by_label["transported_neighbor"][0]
            null_results = (
                by_label["support_matched_null"]
                + by_label["covariance_shaped_null"]
                + by_label["time_shuffled"]
            )
            local_predicted = max(local_result["predicted_norm"], 1e-12)
            local_observed = max(local_result["observed_norm"], 1e-12)
            null_predicted_95 = float(
                np.quantile(
                    [value["predicted_norm"] / local_predicted for value in null_results],
                    0.95,
                )
            )
            null_observed_95 = float(
                np.quantile(
                    [value["observed_norm"] / local_observed for value in null_results],
                    0.95,
                )
            )
            transition_rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "source_time": source["time_index"],
                    "destination_time": destination["time_index"],
                    "transport_predicted_recovery": (
                        transport_result["predicted_norm"] / local_predicted
                    ),
                    "transport_observed_recovery": (
                        transport_result["observed_norm"] / local_observed
                    ),
                    "predicted_null_95": null_predicted_95,
                    "observed_null_95": null_observed_95,
                    "predicted_null_advantage": (
                        transport_result["predicted_norm"] / local_predicted
                        - null_predicted_95
                    ),
                    "observed_null_advantage": (
                        transport_result["observed_norm"] / local_observed
                        - null_observed_95
                    ),
                    "linearity_cosine": transport_result["linearity_cosine"],
                    "semantic_cosine": transport_result[
                        "source_destination_semantic_cosine"
                    ],
                    "zero_edit_error": zero_edit_check(destination),
                }
            )
    return rows, transition_rows


CAUSAL_ROWS = []
CAUSAL_TRANSITIONS = []
CAUSAL_GATE = {}
if not PIPELINE_FAILED:
    try:
        CAUSAL_ROWS, CAUSAL_TRANSITIONS = causal_transport_rows()
        write_csv(CAUSAL_DIR / "causal_direction_rows.csv", CAUSAL_ROWS)
        write_csv(CAUSAL_DIR / "causal_transition_summary.csv", CAUSAL_TRANSITIONS)
        metrics = {
            key: float(np.median([row[key] for row in CAUSAL_TRANSITIONS]))
            for key in [
                "transport_predicted_recovery",
                "transport_observed_recovery",
                "predicted_null_advantage",
                "observed_null_advantage",
                "linearity_cosine",
                "semantic_cosine",
            ]
        }
        maximum_zero = max(row["zero_edit_error"] for row in CAUSAL_TRANSITIONS)
        trajectory_advantages = {
            int(trajectory_id): float(
                np.median(
                    [
                        row["observed_null_advantage"]
                        for row in CAUSAL_TRANSITIONS
                        if row["trajectory_id"] == trajectory_id
                    ]
                )
            )
            for trajectory_id in sorted(
                {row["trajectory_id"] for row in CAUSAL_TRANSITIONS}
            )
        }
        positive_trajectories = sum(
            value > 0 for value in trajectory_advantages.values()
        )
        required_positive = min(
            REQUIRED_POSITIVE_CAUSAL_TRAJECTORIES,
            len(trajectory_advantages),
        )
        CAUSAL_GATE = {
            "medians": metrics,
            "maximum_zero_edit_error": maximum_zero,
            "trajectory_observed_null_advantage": trajectory_advantages,
            "positive_trajectories": positive_trajectories,
            "required_positive_trajectories": required_positive,
            "thresholds": {
                "minimum_linearity_cosine": MIN_CAUSAL_LINEARITY_COSINE,
                "minimum_transport_recovery": MIN_TRANSPORT_RECOVERY,
                "minimum_null_advantage": MIN_TRANSPORT_NULL_ADVANTAGE,
                "minimum_semantic_cosine": MIN_TRANSPORT_SEMANTIC_COSINE,
                "maximum_zero_edit_error": MAX_ZERO_EDIT_ERROR,
            },
            "passed": bool(
                metrics["linearity_cosine"] >= MIN_CAUSAL_LINEARITY_COSINE
                and metrics["transport_observed_recovery"] >= MIN_TRANSPORT_RECOVERY
                and metrics["observed_null_advantage"] >= MIN_TRANSPORT_NULL_ADVANTAGE
                and metrics["semantic_cosine"] >= MIN_TRANSPORT_SEMANTIC_COSINE
                and maximum_zero <= MAX_ZERO_EDIT_ERROR
                and positive_trajectories >= required_positive
            ),
        }
        write_json(CAUSAL_DIR / "causal_transport_gate.json", CAUSAL_GATE)
        memory_report("causal_transport_complete")
    except Exception:
        record_failure("causal_transport")
'''


decision_and_plots = r'''# Apply the frozen Stage 15 claim ladder and render compact diagnostics.


def finalize_source_identity():
    if PIPELINE_FAILED:
        return False
    return verify_executed_notebook_through(
        "# Apply the frozen Stage 15 claim ladder and render compact diagnostics."
    )


FINAL_SOURCE_VERIFIED = False
try:
    FINAL_SOURCE_VERIFIED = finalize_source_identity()
except Exception:
    record_failure("final_source_verification")

local_rows = [
    row for row in CAUSAL_ROWS if row.get("label") == "local_positive"
]
LOCAL_CAUSAL_GATE = {
    "median_linearity_cosine": (
        float(np.median([row["linearity_cosine"] for row in local_rows]))
        if local_rows else None
    ),
    "median_observed_norm": (
        float(np.median([row["observed_norm"] for row in local_rows]))
        if local_rows else None
    ),
}
LOCAL_CAUSAL_GATE["passed"] = bool(
    local_rows
    and LOCAL_CAUSAL_GATE["median_linearity_cosine"]
    >= MIN_CAUSAL_LINEARITY_COSINE
    and LOCAL_CAUSAL_GATE["median_observed_norm"] > 1e-4
)

if PIPELINE_FAILED:
    DECISION = "PIPELINE_FAILURE"
elif RUN_MODE == "smoke":
    DECISION = "SMOKE_ONLY"
elif not SOURCE_IDENTITY.get("confirmation_eligible", False):
    DECISION = "UNBOUND_EXPLORATORY_RUN"
elif READER_GATE.get("passed") and GEOMETRY_GATE.get("passed") and CAUSAL_GATE.get("passed"):
    DECISION = "LONGITUDINAL_CAUSAL_BUNDLE_SUPPORTED"
elif GEOMETRY_GATE.get("passed"):
    DECISION = "SMOOTH_NONCAUSAL_FIELD"
elif LOCAL_CAUSAL_GATE.get("passed"):
    DECISION = "LOCAL_CAUSAL_ONLY"
else:
    DECISION = "NO_LONGITUDINAL_BUNDLE_EVIDENCE"

DECISION_PAYLOAD = {
    "decision": DECISION,
    "run_mode": RUN_MODE,
    "protocol_id": PROTOCOL_ID,
    "run_signature": RUN_SIGNATURE,
    "source_identity": SOURCE_IDENTITY,
    "reader_gate": READER_GATE,
    "geometry_gate": GEOMETRY_GATE,
    "local_causal_gate": LOCAL_CAUSAL_GATE,
    "causal_transport_gate": CAUSAL_GATE,
    "pipeline_failed": PIPELINE_FAILED,
    "failure_message": FAILURE_MESSAGE,
    "claims": {
        "global_j_space_authorized": False,
        "cross_model_generality_authorized": False,
        "longitudinal_smoothness_authorized": bool(
            DECISION == "LONGITUDINAL_CAUSAL_BUNDLE_SUPPORTED"
        ),
        "transported_causal_control_authorized": bool(
            DECISION == "LONGITUDINAL_CAUSAL_BUNDLE_SUPPORTED"
        ),
        "reader_scope": "construction-only fixed physical reader",
        "action_scope": "state-independent normalized DCT action basis",
    },
}
write_json(OUT / "stage15_decision.json", DECISION_PAYLOAD)
write_json(OUT / "timings.json", TIMINGS)
write_json(OUT / "memory.json", MEMORY)
if not PIPELINE_FAILED:
    (OUT / "FAILURE_TRACE.txt").write_text("NONE\n")

if not PIPELINE_FAILED:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    reader_rows = list(READER_GATE.get("r2", {}).items())
    axes[0].bar(
        np.arange(len(reader_rows)),
        [value for _, value in reader_rows],
        color="#3267a8",
    )
    axes[0].axhline(MIN_READER_SPATIAL_R2, color="black", linestyle="--")
    axes[0].set_xticks(
        np.arange(len(reader_rows)),
        [label.replace("_", "\n") for label, _ in reader_rows],
    )
    axes[0].set_ylabel("held-out true-token R²")
    axes[0].set_title("Fixed physical readers")

    primary_rows = [
        row
        for row in GEOMETRY_SUMMARY
        if row["split"] == "evaluation"
        and row["horizon"] == PRIMARY_HORIZON
        and row["block"] == PRIMARY_BLOCK
        and row["metric"] in {
            "k_distance", "read_subspace_distance", "write_subspace_distance"
        }
    ]
    axes[1].bar(
        np.arange(len(primary_rows)),
        [row["spearman_physical_distance"] for row in primary_rows],
        color=["#8b5fbf", "#2f9e73", "#e08b36"],
    )
    axes[1].axhline(MIN_GEOMETRY_RHO, color="black", linestyle="--")
    axes[1].set_xticks(
        np.arange(len(primary_rows)),
        [row["metric"].replace("_distance", "").replace("_", "\n") for row in primary_rows],
    )
    axes[1].set_ylabel("Spearman ρ vs physical distance")
    axes[1].set_title("Held-out longitudinal geometry")

    if CAUSAL_TRANSITIONS:
        categories = ["transport\nrecovery", "null\nadvantage", "semantic\ncosine"]
        values = [
            np.median([row["transport_observed_recovery"] for row in CAUSAL_TRANSITIONS]),
            np.median([row["observed_null_advantage"] for row in CAUSAL_TRANSITIONS]),
            np.median([row["semantic_cosine"] for row in CAUSAL_TRANSITIONS]),
        ]
        axes[2].bar(np.arange(3), values, color="#b84949")
        axes[2].set_xticks(np.arange(3), categories)
        axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_title("Transported-mode causal test")
    axes[2].set_ylabel("median normalized metric")
    figure.suptitle(DECISION)
    figure.tight_layout()
    figure.savefig(PLOT_DIR / "stage15_primary_diagnostics.png", dpi=180)
    plt.show()

print(json.dumps(DECISION_PAYLOAD, indent=2))
'''


packaging = r'''# Package compact evidence; retain recomputable raw shards separately.

(OUT / "full_manifest.json").unlink(missing_ok=True)
write_json(
    OUT / "full_manifest.json",
    {
        "root": str(OUT),
        "files": manifest_rows(OUT),
        "raw_operator_note": (
            "truth, target-token, and write/read shards remain in the durable "
            "run directory and are intentionally excluded from the compact ZIP"
        ),
    },
)

compact_root = OUT.parent / f"{OUT.name}_compact"
if compact_root.exists():
    shutil.rmtree(compact_root)
compact_root.mkdir(parents=True)

for name in [
    "config.json",
    "versions.json",
    "source_identity.json",
    "stage15_decision.json",
    "benchmark.json",
    "timings.json",
    "memory.json",
    "restore_test.json",
    "FAILURE_TRACE.txt",
    "full_manifest.json",
]:
    source = OUT / name
    if source.exists():
        shutil.copy2(source, compact_root / name)

for directory_name in ["analysis", "causal", "plots", "logs"]:
    source = OUT / directory_name
    if source.exists():
        shutil.copytree(source, compact_root / directory_name)

compact_design = compact_root / "design"
compact_design.mkdir(exist_ok=True)
for name in [
    "trajectory_design_manifest.json",
    "design_freeze.json",
]:
    source = DESIGN_DIR / name
    if source.exists():
        shutil.copy2(source, compact_design / name)

compact_manifest = {
    "protocol_id": PROTOCOL_ID,
    "decision": DECISION,
    "run_signature": RUN_SIGNATURE,
    "source_identity": SOURCE_IDENTITY,
    "files": manifest_rows(compact_root),
    "excluded_recomputable_directories": [
        "assets", "truth", "target_tokens", "carrier_scan", "write_read_shards"
    ],
    "raw_evidence_root": str(OUT),
}
write_json(compact_root / "compact_manifest.json", compact_manifest)

archive_base = OUT.parent / "stage15_longitudinal_bundle_result_bundle"
archive = shutil.make_archive(str(archive_base), "zip", compact_root)
print(f"Compact Stage 15 bundle: {archive}")
print(f"Raw evidence retained at: {OUT}")

if DOWNLOAD_RESULTS:
    try:
        from google.colab import files

        files.download(archive)
    except Exception as error:
        print(f"Automatic download unavailable: {error}")
'''


protocol_sources = [
    introduction,
    configuration,
    installation,
    setup,
    analysis_helpers,
    model_helpers,
    design,
    truth_generation,
    fixed_readers,
    carrier_metrics,
    operator_extraction,
    longitudinal_analysis,
    causal_transport,
    decision_and_plots,
    packaging,
]
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
    code(fixed_readers),
    code(carrier_metrics),
    code(operator_extraction),
    code(longitudinal_analysis),
    code(causal_transport),
    code(decision_and_plots),
    code(packaging),
]
for index, cell in enumerate(cells):
    cell["id"] = f"stage15-{index:02d}"

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "gpuType": "T4",
            "name": TARGET.name,
            "provenance": [],
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
payload = json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"
TARGET.write_text(payload)
print(TARGET)
print(hashlib.sha256(payload.encode()).hexdigest())
