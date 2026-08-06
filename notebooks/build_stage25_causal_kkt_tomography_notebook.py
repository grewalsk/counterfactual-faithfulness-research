import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "25_causal_kkt_tomography.ipynb"
BASE = json.loads((ROOT / "24_causal_completion_rank.ipynb").read_text())
NUMERICAL = ROOT.parent / "src/cf_faithfulness/stage25_causal_kkt.py"


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


introduction = r'''# Stage 25: causal KKT tomography of latent contact dynamics

Stages 23–24 found a contact-aligned internal coordinate but rejected two much
stronger claims: changing that coordinate did not switch the downstream action
operator, and adding a shared rank-64 completion recovered only 26.5% of the
native effect while a same-mode control recovered 22.8%.  Stage 25 tests a
different mathematical hypothesis.  In rigid contact, a binary active-set
indicator is only the support of a state- and action-dependent impulse:

\[
v^+=v_{\mathrm{free}}+M^{-1}J(x)^\top\lambda^*,
\qquad 0\leq\lambda_n\perp g(x^+)\geq0.
\]

The notebook therefore asks whether the frozen JEPA-WM contains a causally used
**latent contact-impulse coordinate**, not another global contact direction.
For every state and action, the simulator is run twice from bitwise-identical
initial conditions: once normally and once with only agent–block collisions
disabled.  Pymunk's post-solve callback records the actual impulse, contact
normal, contact points, and penetration distance.  The ordinary-minus-ghost
endpoint is the exact finite contact counterfactual.

Construction trajectories alone fit a two-coordinate readout for normal and
tangential impulse from predictor block 1.  Its full activation covectors are
recovered through the exact adjoint of the frozen CountSketch.  Held-out causal
interventions then erase the inferred impulse while exactly protecting the
eight Stage 23 mode coordinates.  A real latent impulse mechanism must satisfy
all of the following: held-out impulse prediction, a native model preference
for ordinary-contact over ghost targets, movement toward the ghost target when
the impulse is erased, superiority to a norm-matched random control, the
opposite sign under a reverse edit, and negligible mode-coordinate drift.

This is a falsification screen, not an assumption that JEPA implements a full
KKT solver.  It uses finite forward interventions only—no Jacobian, JVP, VJP,
gradient probe, or model-weight update.  Return
`stage25_causal_kkt_result_bundle_<signature>.zip`.
'''


configuration = r'''# SINGLE CONFIGURATION BLOCK
# Required Colab secrets for a source-bound pilot:
# STAGE25_RUN_MODE=pilot
# STAGE25_SOURCE_COMMIT=<full 40-hex commit from the Colab handoff>
# STAGE25_RUN_NONCE=<new unique label, e.g. latent_kkt_20260806_a>
RUN_MODE = "smoke"
EXPERIMENT_SOURCE_REF = ""
RUN_NONCE = "smoke"
try:
    from google.colab import userdata as _colab_userdata

    RUN_MODE = str(_colab_userdata.get("STAGE25_RUN_MODE") or RUN_MODE).strip().lower()
    EXPERIMENT_SOURCE_REF = str(
        _colab_userdata.get("STAGE25_SOURCE_COMMIT") or EXPERIMENT_SOURCE_REF
    ).strip()
    RUN_NONCE = str(
        _colab_userdata.get("STAGE25_RUN_NONCE") or RUN_NONCE
    ).strip()
except Exception:
    pass

if RUN_MODE == "pilot":
    if RUN_NONCE in {"", "smoke"}:
        raise ValueError("pilot mode requires a unique STAGE25_RUN_NONCE")
    if not all(value.isalnum() or value in "-_" for value in RUN_NONCE):
        raise ValueError("STAGE25_RUN_NONCE may contain only letters, numbers, '-' and '_'")

MOUNT_DRIVE = True
DOWNLOAD_RESULTS = True
CONTINUE_AFTER_BENCHMARK = True
MAX_ESTIMATED_TOTAL_MINUTES = 120.0
FRESH_RUN_REQUIRED = True

OUTPUT_DIR = "/content/counterfactual_faithfulness_stage25_causal_kkt"
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/counterfactual_faithfulness_stage25_causal_kkt"
UPSTREAM_STAGE24_DIR = "/content/drive/MyDrive/counterfactual_faithfulness_stage24_causal_completion"
UPSTREAM_STAGE24_RUN_SUFFIX = "b18ea7810677"

PROTOCOL_ID = "stage25-causal-kkt-tomography-v1"
NOTEBOOK_PROTOCOL_SHA256 = "__PROTOCOL_DIGEST__"
EVIDENCE_STATUS = "CONFIRMATORY_ONLY_IF_SOURCE_BOUND_FRESH_UPSTREAM_BOUND_AND_READER_FROZEN"
EXPERIMENT_REPOSITORY = "grewalsk/counterfactual-faithfulness-research"
EXPERIMENT_NOTEBOOK_PATH = "notebooks/25_causal_kkt_tomography.ipynb"
EXPERIMENT_BUILDER_PATH = "notebooks/build_stage25_causal_kkt_tomography_notebook.py"
EXPERIMENT_NUMERICAL_PATH = "src/cf_faithfulness/stage25_causal_kkt.py"

EXPECTED_STAGE24_SOURCE_COMMIT = "2ca9700beb935a964d7416381a9d91d1094b95a0"
EXPECTED_STAGE24_STATUS = "NO_RANK64_CAUSAL_COMPLETION"
EXPECTED_STAGE24_PROTOCOL_ID = "stage24-causal-completion-rank-v1"
EXPECTED_STAGE23_PARTITION_SHA256 = "0f9e376e9d874f1b2429e62b119cf56ea77e3458b8a44c1d4575ed73c988697f"
EXPECTED_STAGE23_GEOMETRY_SHA256 = "3598cc57768077550913cc2008baf870c55288089b8570d5ae96c523956aa4ff"

SEED = 25101
DESIGN_SEED = 25137
MODEL_NAME = "jepa_wm_pusht"
ENVIRONMENT = "PushT"
FRAMESKIP = 5
PRIMARY_HORIZON = 3
TARGET_STEPS = [PRIMARY_HORIZON]
SELECTED_BLOCK = 1
DISCOVERY_BLOCKS = [SELECTED_BLOCK]
ACTIVE_BLOCKS = DISCOVERY_BLOCKS
EXPECTED_CARRIER_CHANNELS = 400

CONSTRUCTION_POOL_TRAJECTORIES = list(range(2100, 2200))
EVALUATION_POOL_TRAJECTORIES = list(range(2300, 2400))
CONSTRUCTION_TRAJECTORY_TARGET = 48
EVALUATION_TRAJECTORY_TARGET = 48
TASK_ID_OFFSET = 10000

ACTIONS_PER_STATE = 13
ACTION_MAGNITUDE = 0.12
ACTION_STEPS = PRIMARY_HORIZON * FRAMESKIP
APPROACH_DISTANCE = 80.0
MIN_ELIGIBLE_CONTACT_BRANCHES = 2
MIN_ELIGIBLE_NONCONTACT_BRANCHES = 2
MIN_CONTACT_IMPULSE_NORM = 1e-4
MIN_CONTACT_POSE_CORRECTION = 1e-4

CARRIER_SKETCH_DIM = 256
OUTPUT_SKETCH_DIM = 256
CARRIER_SKETCH_SEED = 22197
EVAL_OUTPUT_SKETCH_SEED = 23183
RIDGE_PENALTIES = [0.01, 0.1, 1.0, 10.0, 100.0]
RIDGE_CV_FOLDS = 5
IMPULSE_COORDINATES = ["log1p_normal_impulse", "asinh_tangent_impulse"]
PATCH_CONDITIONS = ["impulse_erase", "reverse_impulse", "random_matched"]
ENCODER_BATCH_SIZE = 13
BOOTSTRAP_SEED = 25269
BOOTSTRAP_DRAWS = 10000
MAX_ZERO_EDIT_ERROR = 1e-6
MAX_COORDINATE_RESIDUAL = 1e-6
MAX_MODE_DRIFT = 1e-6
TARGET_ENERGY_FLOOR = 1e-10

MAX_MEDIAN_MOMENTUM_RESIDUAL = 0.35
MIN_CONSTRUCTION_CV_R2 = 0.10
MIN_CONSTRUCTION_CONTACT_AUC = 0.75
MIN_HELDOUT_IMPULSE_R2 = 0.10
MIN_HELDOUT_CONTACT_AUC = 0.75
MIN_NATIVE_CONTACT_COEFFICIENT = 0.10
MIN_ERASURE_TRANSFER = 0.10
MIN_GAIN_OVER_RANDOM = 0.05
MIN_VALID_CONTACT_BRANCHES = 40

if RUN_MODE == "smoke":
    ACTIVE_CONSTRUCTION_POOL_TRAJECTORIES = CONSTRUCTION_POOL_TRAJECTORIES[:12]
    ACTIVE_EVALUATION_POOL_TRAJECTORIES = EVALUATION_POOL_TRAJECTORIES[:12]
    ACTIVE_CONSTRUCTION_TARGET = 3
    ACTIVE_EVALUATION_TARGET = 2
    ACTIVE_BOOTSTRAP_DRAWS = 64
    ACTIVE_MIN_VALID_CONTACT_BRANCHES = 1
elif RUN_MODE == "pilot":
    ACTIVE_CONSTRUCTION_POOL_TRAJECTORIES = CONSTRUCTION_POOL_TRAJECTORIES
    ACTIVE_EVALUATION_POOL_TRAJECTORIES = EVALUATION_POOL_TRAJECTORIES
    ACTIVE_CONSTRUCTION_TARGET = CONSTRUCTION_TRAJECTORY_TARGET
    ACTIVE_EVALUATION_TARGET = EVALUATION_TRAJECTORY_TARGET
    ACTIVE_BOOTSTRAP_DRAWS = BOOTSTRAP_DRAWS
    ACTIVE_MIN_VALID_CONTACT_BRANCHES = MIN_VALID_CONTACT_BRANCHES
else:
    raise ValueError(
        "STAGE25_RUN_MODE must contain only smoke or pilot; "
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

assert ACTIONS_PER_STATE == 13 and ACTION_STEPS == 15
assert SELECTED_BLOCK == 1 and EXPECTED_CARRIER_CHANNELS == 400
assert not set(CONSTRUCTION_POOL_TRAJECTORIES) & set(EVALUATION_POOL_TRAJECTORIES)
assert len(IMPULSE_COORDINATES) == 2
'''
configuration += "\n\nPROTOCOL_CONFIG_KEYS = " + repr(
    assigned_uppercase_names(configuration)
) + "\n"


installation = base_source(2)


setup = base_source(3)
setup = setup.replace("Stage 24", "Stage 25").replace("STAGE24", "STAGE25")
setup = setup.replace("stage24_causal_completion", "stage25_causal_kkt")


analysis_helpers = base_source(4)
analysis_helpers += "\n\n\n" + function_sources(
    NUMERICAL.read_text(),
    [
        "contact_projection_metrics",
        "intervention_transfer_metrics",
        "stepwise_impulse_momentum_residual",
        "fit_standardized_ridge",
        "predict_standardized_ridge",
        "grouped_ridge_penalty_cv",
        "countsketch_adjoint",
        "minimum_norm_coordinate_edit",
        "orthogonal_random_control",
        "r2_components",
    ],
)


model_helpers = base_source(5)
model_helpers = model_helpers.replace("stage24-jepa-wms", "stage25-jepa-wms")
model_helpers = model_helpers.replace("Stage 14 supports PushT only", "Stage 25 supports PushT only")


upstream_import = r'''# Bind the exact Stage 24 negative and its inherited Stage 23 geometry.


def locate_and_verify_stage24():
    root = Path(UPSTREAM_STAGE24_DIR)
    candidate = root / f"pilot_{UPSTREAM_STAGE24_RUN_SUFFIX}"
    if not candidate.is_dir():
        raise RuntimeError(
            f"missing Stage 24 Drive run {candidate}; the compact ZIP is insufficient "
            "because Stage 25 requires the inherited geometry NPZ files"
        )
    required = {
        "source": candidate / "source_identity.json",
        "decision": candidate / "stage24_decision.json",
        "stage23_binding": candidate / "subspaces/stage23_upstream_binding.json",
        "partition": candidate / "subspaces/frozen_mode_partition.npz",
        "geometry": candidate / "subspaces/frozen_mode_operator_geometry.npz",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Stage 24 upstream artifacts are incomplete: {missing}")
    source = json.loads(required["source"].read_text())
    decision = json.loads(required["decision"].read_text())
    stage23_binding = json.loads(required["stage23_binding"].read_text())
    partition_sha = sha256_file(required["partition"])
    geometry_sha = sha256_file(required["geometry"])
    checks = {
        "source_commit": source.get("resolved_commit") == EXPECTED_STAGE24_SOURCE_COMMIT,
        "source_execution_verified": bool(source.get("confirmation_eligible", False)),
        "protocol": source.get("protocol_id") == EXPECTED_STAGE24_PROTOCOL_ID,
        "decision": decision.get("status") == EXPECTED_STAGE24_STATUS,
        "source_bound_claim": bool(decision.get("source_bound_claim_eligible", False)),
        "stage23_bound": bool(decision.get("stage23_upstream_bound", False)),
        "partition_sha": partition_sha == EXPECTED_STAGE23_PARTITION_SHA256,
        "geometry_sha": geometry_sha == EXPECTED_STAGE23_GEOMETRY_SHA256,
        "nested_stage23_checks": bool(all(stage23_binding.get("checks", {}).values())),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Stage 24 upstream binding failed: {checks}")
    partition_destination = SUBSPACE_DIR / "frozen_mode_partition.npz"
    geometry_destination = SUBSPACE_DIR / "frozen_mode_operator_geometry.npz"
    shutil.copy2(required["partition"], partition_destination)
    shutil.copy2(required["geometry"], geometry_destination)
    payload = {
        "upstream_run": str(candidate),
        "checks": checks,
        "source_identity": source,
        "decision_status": decision["status"],
        "partition_sha256": partition_sha,
        "geometry_sha256": geometry_sha,
        "local_partition_sha256": sha256_file(partition_destination),
        "local_geometry_sha256": sha256_file(geometry_destination),
    }
    write_json(SUBSPACE_DIR / "stage24_upstream_binding.json", payload)
    return payload


if not PIPELINE_FAILED:
    try:
        STAGE24_BINDING = locate_and_verify_stage24()
        print(json.dumps(STAGE24_BINDING, indent=2))
    except Exception:
        record_failure("stage24_upstream_binding")
'''


design = r'''# Freeze trajectory pools and action banks before simulator or model data exist.


def rotate_vector(vector, angle):
    vector = np.asarray(vector, dtype=np.float64)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(
        [cosine * vector[0] - sine * vector[1], sine * vector[0] + cosine * vector[1]],
        dtype=np.float64,
    )


def trajectory_specs():
    specs = []
    center = np.asarray([256.0, 256.0])
    all_ids = CONSTRUCTION_POOL_TRAJECTORIES + EVALUATION_POOL_TRAJECTORIES
    total = len(all_ids)
    for design_index, trajectory_id in enumerate(all_ids):
        phase = 0.17 + 2.0 * np.pi * design_index / total
        block = center + 42.0 * np.asarray([np.cos(phase), np.sin(phase)])
        block_angle = ((1.7 * phase + np.pi) % (2.0 * np.pi)) - np.pi
        approach = phase + [0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0][design_index % 4]
        approach += 0.19 * np.sin(design_index)
        agent = block + APPROACH_DISTANCE * np.asarray([np.cos(approach), np.sin(approach)])
        goal_index = (11 * design_index + 3) % total
        goal_phase = 0.61 + 2.0 * np.pi * goal_index / total
        goal_xy = center + 72.0 * np.asarray([np.cos(goal_phase), np.sin(goal_phase)])
        split = "construction" if trajectory_id in CONSTRUCTION_POOL_TRAJECTORIES else "evaluation"
        specs.append(
            {
                "design_index": design_index,
                "trajectory_id": int(trajectory_id),
                "record_id": int(trajectory_id),
                "task_id": int(TASK_ID_OFFSET + design_index),
                "split": split,
                "evaluation_seed": int(DESIGN_SEED + 1009 * design_index),
                "goal": np.asarray(
                    [goal_xy[0], goal_xy[1], ((1.3 * goal_phase + np.pi) % (2.0 * np.pi)) - np.pi],
                    dtype=np.float64,
                ),
                "state": np.asarray(
                    [agent[0], agent[1], block[0], block[1], block_angle, 0.0, 0.0, 0.0, 0.0, 0.0],
                    dtype=np.float64,
                ),
            }
        )
    return specs


def candidate_action_bank(state):
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (10,):
        raise ValueError("candidate state must be a ten-dimensional dynamic PushT state")
    toward_block = state[2:4] - state[:2]
    norm = np.linalg.norm(toward_block)
    if norm <= 1e-12:
        raise RuntimeError("agent-to-block direction is degenerate")
    toward_block /= norm
    branches = [np.zeros((ACTION_STEPS, 2), dtype=np.float64)]
    for index in range(12):
        direction = rotate_vector(toward_block, 2.0 * np.pi * index / 12.0)
        branches.append(np.broadcast_to(ACTION_MAGNITUDE * direction, (ACTION_STEPS, 2)).copy())
    actions = np.stack(branches)
    if actions.shape != (ACTIONS_PER_STATE, ACTION_STEPS, 2):
        raise RuntimeError(f"bad candidate action bank shape {actions.shape}")
    for index in range(1, 7):
        if not np.allclose(actions[index], -actions[index + 6], atol=1e-12):
            raise RuntimeError("radial candidate bank lost antithetic pairing")
    return actions.astype(np.float32)


ALL_POOL_SPECS = trajectory_specs()
CONSTRUCTION_POOL_SPECS = [
    row for row in ALL_POOL_SPECS
    if row["trajectory_id"] in ACTIVE_CONSTRUCTION_POOL_TRAJECTORIES
]
EVALUATION_POOL_SPECS = [
    row for row in ALL_POOL_SPECS
    if row["trajectory_id"] in ACTIVE_EVALUATION_POOL_TRAJECTORIES
]
atomic_npz(
    DESIGN_DIR / "stage25_candidate_pool_design.npz",
    trajectory_ids=np.asarray([row["trajectory_id"] for row in ALL_POOL_SPECS]),
    splits=np.asarray([row["split"] for row in ALL_POOL_SPECS]),
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
    "active_construction_pool_ids": ACTIVE_CONSTRUCTION_POOL_TRAJECTORIES,
    "active_evaluation_pool_ids": ACTIVE_EVALUATION_POOL_TRAJECTORIES,
    "construction_target": ACTIVE_CONSTRUCTION_TARGET,
    "evaluation_target": ACTIVE_EVALUATION_TARGET,
    "selection_uses_model_outputs": False,
}
write_json(DESIGN_DIR / "candidate_pool_manifest.json", POOL_MANIFEST)
DESIGN_FREEZE = {
    "created_before_simulator_or_model_data": True,
    "protocol_id": PROTOCOL_ID,
    "run_signature": RUN_SIGNATURE,
    "candidate_pool_sha256": sha256_file(DESIGN_DIR / "stage25_candidate_pool_design.npz"),
    "pool_manifest_sha256": sha256_file(DESIGN_DIR / "candidate_pool_manifest.json"),
    "paired_counterfactual": "ordinary_contact_vs_agent_block_collision_disabled",
    "impulse_coordinates": IMPULSE_COORDINATES,
    "patch_conditions": PATCH_CONDITIONS,
    "jacobian_used": False,
    "model_loaded": bool("MODEL" in globals()),
}
if DESIGN_FREEZE["model_loaded"]:
    raise RuntimeError("model was loaded before Stage 25 design freeze")
write_json(DESIGN_DIR / "design_freeze.json", DESIGN_FREEZE)
'''


truth_generation = r'''# Generate paired ordinary/ghost rollouts and record exact contact traces.


def record_task(record):
    return {"goal": np.asarray(record["goal"], dtype=np.float64).tolist()}


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


def install_contact_recorder(environment):
    events = []
    original = environment._handle_collision
    original_step = environment.space.step
    physics_step = {"value": 0}

    def counted_step(dt):
        physics_step["value"] += 1
        return original_step(dt)

    environment.space.step = counted_step

    def post_solve(arbiter, space, data):
        original(arbiter, space, data)
        shapes = list(arbiter.shapes)
        bodies = [shape.body for shape in shapes]
        if environment.agent not in bodies or environment.block not in bodies:
            return
        block_index = bodies.index(environment.block)
        raw_impulse = np.asarray(arbiter.total_impulse, dtype=np.float64)
        block_impulse = raw_impulse if block_index == 0 else -raw_impulse
        points = arbiter.contact_point_set
        block_normal = np.asarray(points.normal, dtype=np.float64)
        if block_index == 0:
            block_normal = -block_normal
        world_points = []
        distances = []
        for point in points.points:
            world_points.append(
                0.5 * (
                    np.asarray(point.point_a, dtype=np.float64)
                    + np.asarray(point.point_b, dtype=np.float64)
                )
            )
            distances.append(float(point.distance))
        events.append(
            {
                "impulse": block_impulse,
                "normal": block_normal,
                "world_points": world_points,
                "distances": distances,
                "physics_step": int(physics_step["value"]),
                "block_velocity": np.asarray(environment.block.velocity, dtype=np.float64),
                "block_mass": float(environment.block.mass),
            }
        )

    environment.collision_handeler.post_solve = post_solve
    return events


def disable_agent_block_collision(environment):
    import pymunk

    agent_category = 1 << 0
    block_category = 1 << 1
    all_masks = int(pymunk.ShapeFilter.ALL_MASKS())
    for shape in environment.agent.shapes:
        shape.filter = pymunk.ShapeFilter(
            categories=agent_category,
            mask=all_masks ^ block_category,
        )
    for shape in environment.block.shapes:
        shape.filter = pymunk.ShapeFilter(
            categories=block_category,
            mask=all_masks ^ agent_category,
        )
    return {
        "agent_category": agent_category,
        "block_category": block_category,
        "agent_mask": all_masks ^ block_category,
        "block_mask": all_masks ^ agent_category,
    }


def rollout_branch(record, actions, ghost=False):
    environment, initial = reset_dynamic_environment(
        record["state"], record_task(record), record["evaluation_seed"]
    )
    events = install_contact_recorder(environment)
    filter_audit = disable_agent_block_collision(environment) if ghost else None
    endpoint = None
    endpoint_state = None
    try:
        for step, action in enumerate(actions, start=1):
            observation, _, _, _ = environment.step(action)
            if step == ACTION_STEPS:
                endpoint = {
                    "visual": np.asarray(observation["visual"]).copy(),
                    "proprio": np.asarray(observation["proprio"]).copy(),
                }
                endpoint_state = dynamic_state_from_environment(environment)
    finally:
        environment.close()
    if endpoint is None or endpoint_state is None:
        raise RuntimeError("paired rollout missed the primary horizon")
    return initial, endpoint, endpoint_state, events, filter_audit


def aggregate_contact_trace(events, normal_state, ghost_state):
    if not events:
        return {
            "impulse": np.zeros(2),
            "normal": np.zeros(2),
            "coordinates": np.zeros(2),
            "event_count": 0,
            "contact_point_count": 0,
            "max_penetration": 0.0,
            "momentum_residual": 0.0,
        }
    impulse = np.sum([event["impulse"] for event in events], axis=0)
    weights = np.asarray(
        [max(float(np.linalg.norm(event["impulse"])), 1e-12) for event in events]
    )
    normal = np.average(np.stack([event["normal"] for event in events]), axis=0, weights=weights)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-12:
        normal = impulse.copy()
        normal_norm = float(np.linalg.norm(normal))
    if normal_norm > 1e-12:
        normal /= normal_norm
    if np.dot(impulse, normal) < 0:
        normal = -normal
    tangent = np.asarray([-normal[1], normal[0]], dtype=np.float64)
    lambda_normal = max(float(np.dot(impulse, normal)), 0.0)
    lambda_tangent = float(np.dot(impulse, tangent))
    step_residuals = stepwise_impulse_momentum_residual(
        [event["physics_step"] for event in events],
        [event["impulse"] for event in events],
        [event["block_velocity"] for event in events],
        [event["block_mass"] for event in events],
    )
    distances = [distance for event in events for distance in event["distances"]]
    return {
        "impulse": impulse,
        "normal": normal,
        "coordinates": np.asarray(
            [np.log1p(lambda_normal), np.arcsinh(lambda_tangent)], dtype=np.float64
        ),
        "event_count": len(events),
        "contact_point_count": len(distances),
        "max_penetration": max([max(-value, 0.0) for value in distances] or [0.0]),
        # PushT sets Space.damping=0, so velocity is reset before each solver
        # step.  The valid certificate is therefore per-step block momentum,
        # not the final velocity minus the sum over persistent-contact steps.
        "momentum_residual": float(np.median(step_residuals)),
    }


def branch_path(record_id):
    return TRUTH_DIR / f"state_{int(record_id):04d}.npz"


def generate_paired_truth(records, progress_name):
    started = time.perf_counter()
    for record_index, record in enumerate(records):
        destination = branch_path(record["record_id"])
        if destination.exists():
            raise RuntimeError(f"fresh-run truth shard already exists: {destination}")
        actions = candidate_action_bank(record["state"])
        normal_visuals, ghost_visuals = [], []
        normal_proprios, ghost_proprios = [], []
        normal_states, ghost_states = [], []
        impulses, normals, coordinates = [], [], []
        event_counts, point_counts, penetrations, momentum_residuals = [], [], [], []
        initial_visuals, initial_proprios = [], []
        filter_audit = None
        for action in actions:
            initial, normal_endpoint, normal_state, events, _ = rollout_branch(
                record, action, ghost=False
            )
            ghost_initial, ghost_endpoint, ghost_state, ghost_events, audit = rollout_branch(
                record, action, ghost=True
            )
            if ghost_events:
                raise RuntimeError("ghost agent–block contact callback fired")
            if not np.array_equal(initial["visual"], ghost_initial["visual"]):
                raise RuntimeError("ordinary and ghost initial visuals differ")
            trace = aggregate_contact_trace(events, normal_state, ghost_state)
            initial_visuals.append(initial["visual"])
            initial_proprios.append(initial["proprio"])
            normal_visuals.append(normal_endpoint["visual"])
            ghost_visuals.append(ghost_endpoint["visual"])
            normal_proprios.append(normal_endpoint["proprio"])
            ghost_proprios.append(ghost_endpoint["proprio"])
            normal_states.append(normal_state)
            ghost_states.append(ghost_state)
            impulses.append(trace["impulse"])
            normals.append(trace["normal"])
            coordinates.append(trace["coordinates"])
            event_counts.append(trace["event_count"])
            point_counts.append(trace["contact_point_count"])
            penetrations.append(trace["max_penetration"])
            momentum_residuals.append(trace["momentum_residual"])
            filter_audit = audit
        if not all(np.array_equal(initial_visuals[0], value) for value in initial_visuals[1:]):
            raise RuntimeError("initial visual drift across action branches")
        if not all(np.array_equal(initial_proprios[0], value) for value in initial_proprios[1:]):
            raise RuntimeError("initial proprio drift across action branches")
        atomic_npz(
            destination,
            record_id=np.asarray(record["record_id"], dtype=np.int64),
            split=np.asarray(record["split"]),
            state=np.asarray(record["state"], dtype=np.float64),
            goal=np.asarray(record["goal"], dtype=np.float64),
            initial_visual=np.asarray(initial_visuals[0], dtype=np.uint8),
            initial_proprio=np.asarray(initial_proprios[0], dtype=np.float32),
            selected_actions=actions.astype(np.float32),
            normal_endpoint_visuals=np.asarray(normal_visuals, dtype=np.uint8),
            ghost_endpoint_visuals=np.asarray(ghost_visuals, dtype=np.uint8),
            normal_endpoint_proprios=np.asarray(normal_proprios, dtype=np.float32),
            ghost_endpoint_proprios=np.asarray(ghost_proprios, dtype=np.float32),
            normal_endpoint_states=np.asarray(normal_states, dtype=np.float64),
            ghost_endpoint_states=np.asarray(ghost_states, dtype=np.float64),
            contact_impulses=np.asarray(impulses, dtype=np.float64),
            contact_normals=np.asarray(normals, dtype=np.float64),
            impulse_coordinates=np.asarray(coordinates, dtype=np.float64),
            contact_event_counts=np.asarray(event_counts, dtype=np.int32),
            contact_point_counts=np.asarray(point_counts, dtype=np.int32),
            max_penetrations=np.asarray(penetrations, dtype=np.float64),
            momentum_residuals=np.asarray(momentum_residuals, dtype=np.float64),
        )
        PROVENANCE_COUNTS["truth_generated"] += 1
        write_json(
            OUT / f"{progress_name}_progress.json",
            {
                "completed": record_index + 1,
                "total": len(records),
                "last_record_id": int(record["record_id"]),
                "ghost_filter": filter_audit,
            },
        )
    TIMINGS[f"{progress_name}_seconds"] = time.perf_counter() - started


def paired_truth_eligibility(record):
    with np.load(branch_path(record["record_id"])) as payload:
        normal_states = payload["normal_endpoint_states"].astype(np.float64)
        ghost_states = payload["ghost_endpoint_states"].astype(np.float64)
        impulses = payload["contact_impulses"].astype(np.float64)
        events = payload["contact_event_counts"].astype(np.int64)
        residuals = payload["momentum_residuals"].astype(np.float64)
    pose_correction = np.linalg.norm(
        pose_target(normal_states) - pose_target(ghost_states), axis=1
    )
    impulse_norm = np.linalg.norm(impulses, axis=1)
    contact = (
        (events > 0)
        & (impulse_norm >= MIN_CONTACT_IMPULSE_NORM)
        & (pose_correction >= MIN_CONTACT_POSE_CORRECTION)
    )
    return {
        "record_id": int(record["record_id"]),
        "trajectory_id": int(record["trajectory_id"]),
        "split": record["split"],
        "contact_branches": int(np.sum(contact)),
        "noncontact_branches": int(np.sum(~contact)),
        "median_contact_momentum_residual": float(np.median(residuals[contact])) if np.any(contact) else 1.0,
        "max_contact_pose_correction": float(np.max(pose_correction)) if len(pose_correction) else 0.0,
        "max_contact_impulse_norm": float(np.max(impulse_norm)) if len(impulse_norm) else 0.0,
        "eligible": bool(
            np.sum(contact) >= MIN_ELIGIBLE_CONTACT_BRANCHES
            and np.sum(~contact) >= MIN_ELIGIBLE_NONCONTACT_BRANCHES
        ),
    }


def select_records(records, target):
    rows = [paired_truth_eligibility(record) for record in records]
    selected = [row["record_id"] for row in rows if row["eligible"]][: int(target)]
    if len(selected) != int(target):
        raise RuntimeError(f"paired truth produced {len(selected)} eligible records; need {target}")
    return [record for record in records if record["record_id"] in selected], rows


if not PIPELINE_FAILED:
    try:
        REPO = configure_repo()
        generate_paired_truth(CONSTRUCTION_POOL_SPECS, "truth_construction_pool")
        generate_paired_truth(EVALUATION_POOL_SPECS, "truth_evaluation_pool")
        if "MODEL" in globals():
            raise RuntimeError("model was loaded before paired physical selection")
        CONSTRUCTION_RECORDS, CONSTRUCTION_ELIGIBILITY = select_records(
            CONSTRUCTION_POOL_SPECS, ACTIVE_CONSTRUCTION_TARGET
        )
        EVALUATION_RECORDS, EVALUATION_ELIGIBILITY = select_records(
            EVALUATION_POOL_SPECS, ACTIVE_EVALUATION_TARGET
        )
        write_csv(
            EVIDENCE_DIR / "physical_eligibility_rows.csv",
            CONSTRUCTION_ELIGIBILITY + EVALUATION_ELIGIBILITY,
        )
        SELECTION_CERTIFICATE = {
            "selection_completed_before_model_load": True,
            "selection_used_only_paired_simulator_truth": True,
            "construction_selected_ids": [row["record_id"] for row in CONSTRUCTION_RECORDS],
            "evaluation_selected_ids": [row["record_id"] for row in EVALUATION_RECORDS],
            "eligibility_sha256": sha256_file(EVIDENCE_DIR / "physical_eligibility_rows.csv"),
        }
        write_json(DESIGN_DIR / "physical_selection_freeze.json", SELECTION_CERTIFICATE)
        memory_report("paired_truth_and_selection_complete")
    except Exception:
        record_failure("paired_truth_selection")
'''


construction_baselines = r'''# Load the frozen JEPA-WM and open construction activations only.


def state_model_inputs(record_id, horizon=PRIMARY_HORIZON):
    with np.load(branch_path(record_id)) as truth:
        initial_visual = truth["initial_visual"]
        initial_proprio = truth["initial_proprio"]
        selected_actions = truth["selected_actions"]
    with torch.inference_mode():
        initial = MODEL.encode(to_model_observation(initial_visual, initial_proprio))
    initial = {name: value.detach() for name, value in initial.items()}
    actions = model_action_tensor(PREPROCESSOR, selected_actions, horizon)
    return initial, actions


def carrier_path(record_id):
    return BASELINE_DIR / f"carrier_{int(record_id):04d}.npz"


def load_carrier(record_id):
    with np.load(carrier_path(record_id)) as payload:
        return {name: payload[name].copy() for name in payload.files}


def endpoint_target_sketches(record_id):
    with np.load(branch_path(record_id)) as truth:
        normal_visuals = truth["normal_endpoint_visuals"]
        ghost_visuals = truth["ghost_endpoint_visuals"]
        normal_proprios = truth["normal_endpoint_proprios"]
        ghost_proprios = truth["ghost_endpoint_proprios"]
    visuals = np.concatenate([normal_visuals, ghost_visuals], axis=0)
    proprios = np.concatenate([normal_proprios, ghost_proprios], axis=0)
    sketches = []
    for start in range(0, len(visuals), ENCODER_BATCH_SIZE):
        stop = min(start + ENCODER_BATCH_SIZE, len(visuals))
        observation = to_model_observation(
            visuals[start:stop, None], proprios[start:stop, None]
        )
        with torch.inference_mode():
            encoded = MODEL.encode(observation)
            tokens = encoded["visual"][:, -1, 0].flatten(1, 2)
            if tokens.shape[1:] != (256, 384):
                raise RuntimeError(f"unexpected encoded endpoint shape {tuple(tokens.shape)}")
            sketches.append(EVAL_OUTPUT_PROJECTOR(tokens).cpu().numpy())
    values = np.concatenate(sketches).astype(np.float32)
    return values[:ACTIONS_PER_STATE], values[ACTIONS_PER_STATE:]


def extract_baselines(records, progress_name):
    started = time.perf_counter()
    for index, record in enumerate(records):
        destination = carrier_path(record["record_id"])
        if destination.exists():
            raise RuntimeError(f"fresh baseline shard already exists: {destination}")
        initial, actions = state_model_inputs(record["record_id"])
        with torch.inference_mode():
            predicted, _, captures = forward_with_carriers(
                initial, actions, PRIMARY_HORIZON, capture_blocks=[SELECTED_BLOCK]
            )
            output = EVAL_OUTPUT_PROJECTOR(predicted).cpu().numpy()
            decoded_pose = PHYSICAL_POSE_DECODER(predicted).cpu().numpy()
            carrier = layer_tokens_full(captures[SELECTED_BLOCK]).float().cpu().numpy()
        normal_targets, ghost_targets = endpoint_target_sketches(record["record_id"])
        atomic_npz(
            destination,
            record_id=np.asarray(record["record_id"], dtype=np.int64),
            carrier=carrier.astype(np.float32),
            output_eval_sketch=output.astype(np.float32),
            decoded_pose=decoded_pose.astype(np.float32),
            normal_target_sketch=normal_targets.astype(np.float32),
            ghost_target_sketch=ghost_targets.astype(np.float32),
        )
        PROVENANCE_COUNTS["carrier_baseline_generated"] += 1
        write_json(
            OUT / f"{progress_name}_progress.json",
            {"completed": index + 1, "total": len(records), "last_record_id": int(record["record_id"])},
        )
        del initial, actions, predicted, captures, carrier
        gc.collect()
        torch.cuda.empty_cache()
    TIMINGS[f"{progress_name}_seconds"] = time.perf_counter() - started


def hook_identity_test(record_id):
    initial, actions = state_model_inputs(record_id)
    with torch.inference_mode():
        baseline, _, _ = forward_with_carriers(
            initial, actions, PRIMARY_HORIZON, capture_blocks=[SELECTED_BLOCK]
        )
        patched, _, _ = forward_with_carriers(
            initial,
            actions,
            PRIMARY_HORIZON,
            capture_blocks=[SELECTED_BLOCK],
            intervention={
                "block": SELECTED_BLOCK,
                "delta": torch.zeros(
                    ACTIONS_PER_STATE, 256, EXPECTED_CARRIER_CHANNELS,
                    device="cuda", dtype=torch.float32,
                ),
            },
        )
    error = float(torch.max(torch.abs(patched - baseline)).cpu())
    result = {"record_id": int(record_id), "max_abs_error": error, "passed": error <= MAX_ZERO_EDIT_ERROR}
    if not result["passed"]:
        raise RuntimeError(f"zero intervention changed output: {result}")
    write_json(OUT / "hook_identity_test.json", result)
    return result


def forward_benchmark(record_id):
    initial, actions = state_model_inputs(record_id)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        forward_with_carriers(initial, actions, PRIMARY_HORIZON, capture_blocks=[SELECTED_BLOCK])
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    estimated = seconds * ACTIVE_EVALUATION_TARGET * len(PATCH_CONDITIONS) / 60.0
    result = {
        "seconds_per_candidate_batch": seconds,
        "evaluation_records": ACTIVE_EVALUATION_TARGET,
        "patched_candidate_batches_per_record": len(PATCH_CONDITIONS),
        "estimated_intervention_minutes": estimated,
        "warning_threshold_minutes": MAX_ESTIMATED_TOTAL_MINUTES,
    }
    write_json(OUT / "forward_benchmark.json", result)
    if estimated > MAX_ESTIMATED_TOTAL_MINUTES and not CONTINUE_AFTER_BENCHMARK:
        raise RuntimeError("measured estimate exceeds configured credit guard")
    return result


if not PIPELINE_FAILED:
    try:
        MODEL, PREPROCESSOR, PREDICTOR, PREDICTOR_BLOCK_MODULES = load_frozen_model()
        CARRIER_PROJECTOR = CountSketchProjector(
            256 * EXPECTED_CARRIER_CHANNELS, CARRIER_SKETCH_DIM, CARRIER_SKETCH_SEED
        )
        EVAL_OUTPUT_PROJECTOR = CountSketchProjector(
            256 * 384, OUTPUT_SKETCH_DIM, EVAL_OUTPUT_SKETCH_SEED
        )
        PHYSICAL_POSE_DECODER = physical_pose_decoder()
        HOOK_IDENTITY = hook_identity_test(CONSTRUCTION_RECORDS[0]["record_id"])
        FORWARD_BENCHMARK = forward_benchmark(CONSTRUCTION_RECORDS[0]["record_id"])
        extract_baselines(CONSTRUCTION_RECORDS, "construction_baselines")
        memory_report("construction_activations_complete")
    except Exception:
        record_failure("construction_baselines")
'''


reader_fit = r'''# Fit and freeze the construction-only impulse readout before evaluation activations open.


def load_stage23_geometry():
    with np.load(SUBSPACE_DIR / "frozen_mode_operator_geometry.npz") as payload:
        return {name: payload[name].copy() for name in payload.files}


def whiten_carrier(values, geometry):
    return transform_primal_channels(
        np.asarray(values, dtype=np.float64),
        geometry["channel_inverse_square_root"],
    )


def carrier_sketch(values, geometry):
    white = whiten_carrier(values, geometry)
    tensor = torch.as_tensor(white, device="cuda", dtype=torch.float32)
    with torch.inference_mode():
        return CARRIER_PROJECTOR(tensor).cpu().numpy().astype(np.float64)


def reader_training_matrices(records, geometry):
    features, targets, groups, contacts = [], [], [], []
    for record in records:
        payload = load_carrier(record["record_id"])
        with np.load(branch_path(record["record_id"])) as truth:
            coordinates = truth["impulse_coordinates"].astype(np.float64)
            events = truth["contact_event_counts"].astype(np.int64)
        features.append(carrier_sketch(payload["carrier"], geometry))
        targets.append(coordinates)
        groups.extend([int(record["record_id"])] * ACTIONS_PER_STATE)
        contacts.extend((events > 0).tolist())
    return (
        np.concatenate(features), np.concatenate(targets),
        np.asarray(groups), np.asarray(contacts, dtype=bool),
    )


def grouped_oof_predictions(features, targets, groups, penalty, folds):
    unique = np.unique(groups)
    folds = min(int(folds), len(unique))
    predictions = np.full_like(targets, np.nan, dtype=np.float64)
    for fold in range(folds):
        held = unique[np.arange(len(unique)) % folds == fold]
        test = np.isin(groups, held)
        model = fit_standardized_ridge(features[~test], targets[~test], penalty)
        predictions[test] = predict_standardized_ridge(model, features[test])[0]
    if not np.all(np.isfinite(predictions)):
        raise RuntimeError("construction OOF impulse predictions are incomplete")
    return predictions


def rank_correlation(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left_rank = np.argsort(np.argsort(left, kind="stable"), kind="stable").astype(np.float64)
    right_rank = np.argsort(np.argsort(right, kind="stable"), kind="stable").astype(np.float64)
    if np.std(left_rank) <= 1e-12 or np.std(right_rank) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def fit_and_freeze_impulse_reader():
    from sklearn.metrics import roc_auc_score

    geometry = load_stage23_geometry()
    features, targets, groups, contacts = reader_training_matrices(
        CONSTRUCTION_RECORDS, geometry
    )
    penalty, cv_rows = grouped_ridge_penalty_cv(
        features, targets, groups, RIDGE_PENALTIES, RIDGE_CV_FOLDS
    )
    oof = grouped_oof_predictions(features, targets, groups, penalty, RIDGE_CV_FOLDS)
    metrics = r2_components(targets, oof)
    contact_auc = float(roc_auc_score(contacts.astype(int), oof[:, 0]))
    normal_spearman = rank_correlation(targets[:, 0], oof[:, 0])
    model = fit_standardized_ridge(features, targets, penalty)
    sketch_gradient = np.asarray(model["weight"]) / np.asarray(model["x_scale"])[:, None]
    white_covectors = countsketch_adjoint(
        CARRIER_PROJECTOR.bucket.detach().cpu().numpy(),
        CARRIER_PROJECTOR.sign.detach().cpu().numpy(),
        CARRIER_PROJECTOR.scale.detach().cpu().numpy(),
        sketch_gradient,
    )
    mode_covectors = geometry["mode_covectors_white"].astype(np.float64)
    if white_covectors.shape != (256 * EXPECTED_CARRIER_CHANNELS, 2):
        raise RuntimeError(f"bad impulse covector shape {white_covectors.shape}")
    overlap = np.linalg.norm(mode_covectors.T @ white_covectors, axis=0)
    atomic_npz(
        SUBSPACE_DIR / "frozen_impulse_reader.npz",
        x_mean=model["x_mean"], x_scale=model["x_scale"],
        y_mean=model["y_mean"], y_scale=model["y_scale"],
        weight=model["weight"], white_covectors=white_covectors,
        mode_covectors_white=mode_covectors,
        ridge_penalty=np.asarray(penalty),
    )
    summary = {
        "frozen_before_evaluation_activations": True,
        "construction_only": True,
        "evaluation_activation_ids_seen": [],
        "physical_contact_labels_used_for_reader_fit": True,
        "ridge_penalty": penalty,
        "cv_rows": cv_rows,
        "mean_cv_r2": metrics["mean_r2"],
        "component_cv_r2": metrics["component_r2"].tolist(),
        "normal_impulse_spearman": normal_spearman,
        "contact_auc": contact_auc,
        "mode_overlap_norms": overlap.tolist(),
        "reader_sha256": sha256_file(SUBSPACE_DIR / "frozen_impulse_reader.npz"),
    }
    summary["construction_gate_pass"] = bool(
        summary["mean_cv_r2"] >= MIN_CONSTRUCTION_CV_R2
        and summary["contact_auc"] >= MIN_CONSTRUCTION_CONTACT_AUC
    )
    write_json(SUBSPACE_DIR / "impulse_reader_freeze.json", summary)
    write_csv(EVIDENCE_DIR / "construction_reader_cv.csv", cv_rows)
    return summary


if not PIPELINE_FAILED:
    try:
        IMPULSE_READER_FREEZE = fit_and_freeze_impulse_reader()
        print(json.dumps(IMPULSE_READER_FREEZE, indent=2))
    except Exception:
        record_failure("construction_impulse_reader")
'''


evaluation_open = r'''# Open held-out activations only after the impulse reader is frozen.


def load_impulse_reader():
    with np.load(SUBSPACE_DIR / "frozen_impulse_reader.npz") as payload:
        return {name: payload[name].copy() for name in payload.files}


def reader_model_from_payload(payload):
    return {
        "x_mean": payload["x_mean"], "x_scale": payload["x_scale"],
        "y_mean": payload["y_mean"], "y_scale": payload["y_scale"],
        "weight": payload["weight"], "penalty": float(payload["ridge_penalty"]),
    }


if not PIPELINE_FAILED:
    try:
        if not json.loads((SUBSPACE_DIR / "impulse_reader_freeze.json").read_text()).get(
            "frozen_before_evaluation_activations", False
        ):
            raise RuntimeError("impulse reader was not frozen before evaluation")
        extract_baselines(EVALUATION_RECORDS, "evaluation_baselines")
        from sklearn.metrics import roc_auc_score

        geometry = load_stage23_geometry()
        reader_payload = load_impulse_reader()
        reader_model = reader_model_from_payload(reader_payload)
        features, targets, groups, contacts = reader_training_matrices(
            EVALUATION_RECORDS, geometry
        )
        predictions = predict_standardized_ridge(reader_model, features)[0]
        heldout = r2_components(targets, predictions)
        heldout_auc = float(roc_auc_score(contacts.astype(int), predictions[:, 0]))
        heldout_spearman = rank_correlation(targets[:, 0], predictions[:, 0])
        reader_rows = []
        cursor = 0
        for record in EVALUATION_RECORDS:
            for action_index in range(ACTIONS_PER_STATE):
                reader_rows.append(
                    {
                        "record_id": int(record["record_id"]),
                        "action_index": action_index,
                        "contact": bool(contacts[cursor]),
                        "true_log1p_normal": float(targets[cursor, 0]),
                        "true_asinh_tangent": float(targets[cursor, 1]),
                        "pred_log1p_normal": float(predictions[cursor, 0]),
                        "pred_asinh_tangent": float(predictions[cursor, 1]),
                    }
                )
                cursor += 1
        write_csv(EVIDENCE_DIR / "heldout_impulse_reader_rows.csv", reader_rows)
        HELDOUT_READER = {
            "mean_r2": heldout["mean_r2"],
            "component_r2": heldout["component_r2"].tolist(),
            "contact_auc": heldout_auc,
            "normal_impulse_spearman": heldout_spearman,
            "records": len(EVALUATION_RECORDS),
            "branches": len(reader_rows),
        }
        write_json(EVIDENCE_DIR / "heldout_impulse_reader.json", HELDOUT_READER)
        EVALUATION_OPENED = True
        memory_report("evaluation_activations_opened")
        print(json.dumps(HELDOUT_READER, indent=2))
    except Exception:
        EVALUATION_OPENED = False
        record_failure("evaluation_open")
'''


interventions = r'''# Erase latent impulse coordinates while protecting the frozen mode coordinates.


def native_from_white_edit(values, geometry):
    return inverse_transform_primal_channels(
        np.asarray(values, dtype=np.float64), geometry["channel_square_root"]
    )


def make_record_edits(record, geometry, reader_payload):
    payload = load_carrier(record["record_id"])
    white = whiten_carrier(payload["carrier"], geometry)
    sketches = carrier_sketch(payload["carrier"], geometry)
    reader_model = reader_model_from_payload(reader_payload)
    _, predicted_standard = predict_standardized_ridge(reader_model, sketches)
    zero_standard = (
        -np.asarray(reader_model["y_mean"]) / np.asarray(reader_model["y_scale"])
    )
    impulse_covectors = reader_payload["white_covectors"].astype(np.float64)
    mode_covectors = reader_payload["mode_covectors_white"].astype(np.float64)
    exclusions = np.concatenate([impulse_covectors, mode_covectors], axis=1)
    erase = np.zeros_like(white, dtype=np.float64)
    reverse = np.zeros_like(white, dtype=np.float64)
    random_control = np.zeros_like(white, dtype=np.float64)
    diagnostics = []
    with np.load(branch_path(record["record_id"])) as truth:
        event_counts = truth["contact_event_counts"].astype(np.int64)
        impulses = truth["contact_impulses"].astype(np.float64)
        normal_states = truth["normal_endpoint_states"].astype(np.float64)
        ghost_states = truth["ghost_endpoint_states"].astype(np.float64)
    pose_correction = np.linalg.norm(
        pose_target(normal_states) - pose_target(ghost_states), axis=1
    )
    contact_mask = (
        (event_counts > 0)
        & (np.linalg.norm(impulses, axis=1) >= MIN_CONTACT_IMPULSE_NORM)
        & (pose_correction >= MIN_CONTACT_POSE_CORRECTION)
    )
    for action_index in range(ACTIONS_PER_STATE):
        if not contact_mask[action_index]:
            continue
        coordinate_delta = zero_standard - predicted_standard[action_index]
        result = minimum_norm_coordinate_edit(
            impulse_covectors, coordinate_delta, protected=mode_covectors
        )
        if result["coordinate_residual_norm"] > MAX_COORDINATE_RESIDUAL:
            raise RuntimeError(f"impulse edit coordinate residual failed: {result}")
        if result["protected_drift_norm"] > MAX_MODE_DRIFT:
            raise RuntimeError(f"impulse edit changed protected mode coordinates: {result}")
        flat = result["edit"]
        erase[action_index] = flat.reshape(256, EXPECTED_CARRIER_CHANNELS)
        reverse[action_index] = -erase[action_index]
        random_flat = orthogonal_random_control(
            result["edit_norm"], exclusions,
            stable_seed(SEED, record["record_id"], action_index, "random_control"),
        )
        random_control[action_index] = random_flat.reshape(256, EXPECTED_CARRIER_CHANNELS)
        diagnostics.append(
            {
                "record_id": int(record["record_id"]),
                "action_index": action_index,
                "pred_log1p_normal": float(
                    predicted_standard[action_index, 0] * reader_model["y_scale"][0]
                    + reader_model["y_mean"][0]
                ),
                "pred_asinh_tangent": float(
                    predicted_standard[action_index, 1] * reader_model["y_scale"][1]
                    + reader_model["y_mean"][1]
                ),
                "coordinate_residual_norm": result["coordinate_residual_norm"],
                "mode_drift_norm": result["protected_drift_norm"],
                "edit_norm_white": result["edit_norm"],
                "constraint_condition_number": result["condition_number"],
            }
        )
    return {
        "impulse_erase": native_from_white_edit(erase, geometry),
        "reverse_impulse": native_from_white_edit(reverse, geometry),
        "random_matched": native_from_white_edit(random_control, geometry),
    }, diagnostics, contact_mask


def run_patched_context(initial, actions, delta):
    tensor = torch.as_tensor(delta, device="cuda", dtype=torch.float32)
    with torch.inference_mode():
        predicted, _, _ = forward_with_carriers(
            initial, actions, PRIMARY_HORIZON,
            capture_blocks=[SELECTED_BLOCK],
            intervention={"block": SELECTED_BLOCK, "delta": tensor},
        )
        output = EVAL_OUTPUT_PROJECTOR(predicted).cpu().numpy().astype(np.float64)
        pose = PHYSICAL_POSE_DECODER(predicted).cpu().numpy().astype(np.float64)
    PROVENANCE_COUNTS["patched_forwards_generated"] += 1
    return output, pose


def evaluate_record_interventions(record, geometry, reader_payload):
    baseline = load_carrier(record["record_id"])
    edits, diagnostics, contact_mask = make_record_edits(record, geometry, reader_payload)
    initial, actions = state_model_inputs(record["record_id"])
    patched = {}
    for condition in PATCH_CONDITIONS:
        patched[condition] = run_patched_context(initial, actions, edits[condition])
    with np.load(branch_path(record["record_id"])) as truth:
        normal_states = truth["normal_endpoint_states"].astype(np.float64)
        ghost_states = truth["ghost_endpoint_states"].astype(np.float64)
        momentum_residuals = truth["momentum_residuals"].astype(np.float64)
    normal_pose = pose_target(normal_states)
    ghost_pose = pose_target(ghost_states)
    rows = []
    for action_index in np.flatnonzero(contact_mask):
        baseline_output = baseline["output_eval_sketch"][action_index].astype(np.float64)
        contact_output = baseline["normal_target_sketch"][action_index].astype(np.float64)
        ghost_output = baseline["ghost_target_sketch"][action_index].astype(np.float64)
        baseline_pose = baseline["decoded_pose"][action_index].astype(np.float64)
        desired_output = ghost_output - baseline_output
        desired_pose = ghost_pose[action_index] - baseline_pose
        if (
            np.sum((contact_output - ghost_output) ** 2) <= TARGET_ENERGY_FLOOR
            or np.sum(desired_output**2) <= TARGET_ENERGY_FLOOR
            or np.sum(desired_pose**2) <= TARGET_ENERGY_FLOOR
        ):
            continue
        native = contact_projection_metrics(baseline_output, contact_output, ghost_output)
        for condition in PATCH_CONDITIONS:
            output, pose = patched[condition]
            output_metrics = intervention_transfer_metrics(
                baseline_output, output[action_index], desired_output
            )
            pose_metrics = intervention_transfer_metrics(
                baseline_pose, pose[action_index], desired_pose
            )
            rows.append(
                {
                    "record_id": int(record["record_id"]),
                    "trajectory_id": int(record["trajectory_id"]),
                    "action_index": int(action_index),
                    "condition": condition,
                    "native_contact_coefficient": native["contact_coefficient"],
                    "native_contact_cosine": native["contact_cosine"],
                    "native_contact_preference": native["contact_preference"],
                    "output_transfer_coefficient": output_metrics["transfer_coefficient"],
                    "output_transfer_cosine": output_metrics["transfer_cosine"],
                    "output_target_energy": output_metrics["target_energy"],
                    "output_orthogonal_residual_ratio": output_metrics["orthogonal_residual_ratio"],
                    "pose_transfer_coefficient": pose_metrics["transfer_coefficient"],
                    "pose_transfer_cosine": pose_metrics["transfer_cosine"],
                    "pose_target_energy": pose_metrics["target_energy"],
                    "momentum_residual": float(momentum_residuals[action_index]),
                }
            )
    return rows, diagnostics


if not PIPELINE_FAILED:
    try:
        geometry = load_stage23_geometry()
        reader_payload = load_impulse_reader()
        INTERVENTION_ROWS, EDIT_DIAGNOSTICS = [], []
        started = time.perf_counter()
        if IMPULSE_READER_FREEZE.get("construction_gate_pass", False):
            for index, record in enumerate(EVALUATION_RECORDS):
                rows, diagnostics = evaluate_record_interventions(
                    record, geometry, reader_payload
                )
                INTERVENTION_ROWS.extend(rows)
                EDIT_DIAGNOSTICS.extend(diagnostics)
                write_json(
                    OUT / "intervention_progress.json",
                    {"completed": index + 1, "total": len(EVALUATION_RECORDS), "last_record_id": int(record["record_id"])},
                )
        else:
            log.warning("Construction impulse-reader gate failed; causal interventions skipped")
        TIMINGS["causal_interventions_seconds"] = time.perf_counter() - started
        write_csv(EVIDENCE_DIR / "causal_impulse_intervention_rows.csv", INTERVENTION_ROWS)
        write_csv(EVIDENCE_DIR / "causal_impulse_edit_diagnostics.csv", EDIT_DIAGNOSTICS)
        memory_report("causal_impulse_interventions_complete")
    except Exception:
        record_failure("causal_impulse_interventions")
'''


decision = r'''# Apply preregistered Stage 25 latent-contact-impulse gates.


def bootstrap_summary(values, groups, seed):
    values = np.asarray(values, dtype=np.float64)
    draws = clustered_bootstrap_mean(values, groups, ACTIVE_BOOTSTRAP_DRAWS, seed)
    return {
        "mean": float(np.mean(values)),
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
        "n": int(len(values)),
        "clusters": int(len(np.unique(groups))),
    }


if not PIPELINE_FAILED:
    try:
        verify_executed_notebook_through(
            "# Apply preregistered Stage 25 latent-contact-impulse gates."
        )
    except Exception:
        record_failure("source_execution_verification")


if PIPELINE_FAILED:
    DECISION_PAYLOAD = {
        "status": "EXECUTION_ERROR",
        "failure": FAILURE_MESSAGE,
        "source_bound_claim_eligible": False,
    }
else:
    eligibility = CONSTRUCTION_ELIGIBILITY + EVALUATION_ELIGIBILITY
    physical_residuals = [
        row["median_contact_momentum_residual"] for row in eligibility if row["eligible"]
    ]
    instrumentation = {
        "median_momentum_residual": float(np.median(physical_residuals)),
        "pass": bool(np.median(physical_residuals) <= MAX_MEDIAN_MOMENTUM_RESIDUAL),
    }
    summaries = {}
    if INTERVENTION_ROWS:
        for condition in PATCH_CONDITIONS:
            selected = [row for row in INTERVENTION_ROWS if row["condition"] == condition]
            summaries[condition] = bootstrap_summary(
                [row["output_transfer_coefficient"] for row in selected],
                [row["record_id"] for row in selected],
                stable_seed(BOOTSTRAP_SEED, condition),
            )
        erase = [row for row in INTERVENTION_ROWS if row["condition"] == "impulse_erase"]
        random_rows = [row for row in INTERVENTION_ROWS if row["condition"] == "random_matched"]
        reverse_rows = [row for row in INTERVENTION_ROWS if row["condition"] == "reverse_impulse"]
        gain = np.asarray([row["output_transfer_coefficient"] for row in erase]) - np.asarray(
            [row["output_transfer_coefficient"] for row in random_rows]
        )
        gain_summary = bootstrap_summary(
            gain, [row["record_id"] for row in erase], stable_seed(BOOTSTRAP_SEED, "gain")
        )
        native_summary = bootstrap_summary(
            [row["native_contact_coefficient"] for row in erase],
            [row["record_id"] for row in erase],
            stable_seed(BOOTSTRAP_SEED, "native"),
        )
        reverse_mean = float(np.mean([row["output_transfer_coefficient"] for row in reverse_rows]))
        max_mode_drift = max([row["mode_drift_norm"] for row in EDIT_DIAGNOSTICS] or [float("inf")])
        max_coordinate_residual = max(
            [row["coordinate_residual_norm"] for row in EDIT_DIAGNOSTICS] or [float("inf")]
        )
    else:
        gain_summary = {"mean": 0.0, "lower": 0.0, "upper": 0.0, "n": 0, "clusters": 0}
        native_summary = {"mean": 0.0, "lower": 0.0, "upper": 0.0, "n": 0, "clusters": 0}
        reverse_mean = 0.0
        max_mode_drift = float("inf")
        max_coordinate_residual = float("inf")
    gates = {
        "instrumentation": instrumentation["pass"],
        "construction_reader": bool(IMPULSE_READER_FREEZE.get("construction_gate_pass", False)),
        "heldout_reader": bool(
            HELDOUT_READER["mean_r2"] >= MIN_HELDOUT_IMPULSE_R2
            and HELDOUT_READER["contact_auc"] >= MIN_HELDOUT_CONTACT_AUC
        ),
        "valid_contact_branches": bool(gain_summary["n"] >= ACTIVE_MIN_VALID_CONTACT_BRANCHES),
        "native_contact_signal": bool(native_summary["lower"] >= MIN_NATIVE_CONTACT_COEFFICIENT),
        "causal_erasure": bool(
            summaries.get("impulse_erase", {}).get("lower", -np.inf) >= MIN_ERASURE_TRANSFER
        ),
        "beats_random": bool(
            gain_summary["mean"] >= MIN_GAIN_OVER_RANDOM and gain_summary["lower"] > 0
        ),
        "reverse_sign_control": bool(
            reverse_mean < summaries.get("impulse_erase", {}).get("mean", 0.0)
        ),
        "mode_protected": bool(max_mode_drift <= MAX_MODE_DRIFT),
        "coordinate_edit_exact": bool(max_coordinate_residual <= MAX_COORDINATE_RESIDUAL),
    }
    if RUN_MODE != "pilot":
        status = "SMOKE_ONLY"
    elif all(gates.values()):
        status = "LATENT_CONTACT_IMPULSE_MECHANISM_SUPPORTED"
    elif not gates["instrumentation"]:
        status = "CONTACT_INSTRUMENTATION_INVALID"
    elif not gates["construction_reader"]:
        status = "NO_CONSTRUCTION_IMPULSE_REPRESENTATION"
    elif not gates["heldout_reader"]:
        status = "NO_HELDOUT_IMPULSE_REPRESENTATION"
    elif not gates["native_contact_signal"]:
        status = "MODEL_HAS_NO_MEASURABLE_CONTACT_CORRECTION"
    else:
        status = "IMPULSE_READABLE_BUT_NOT_CAUSALLY_USED"
    DECISION_PAYLOAD = {
        "status": status,
        "protocol_id": PROTOCOL_ID,
        "run_mode": RUN_MODE,
        "run_signature": RUN_SIGNATURE,
        "source_identity": SOURCE_IDENTITY,
        "source_bound_claim_eligible": bool(
            SOURCE_IDENTITY.get("confirmation_eligible", False)
            and all(STAGE24_BINDING.get("checks", {}).values())
        ),
        "stage24_upstream_bound": bool(all(STAGE24_BINDING.get("checks", {}).values())),
        "instrumentation": instrumentation,
        "construction_reader": IMPULSE_READER_FREEZE,
        "heldout_reader": HELDOUT_READER,
        "intervention_summaries": summaries,
        "erase_minus_random": gain_summary,
        "native_contact_signal": native_summary,
        "reverse_mean": reverse_mean,
        "max_mode_drift": max_mode_drift,
        "max_coordinate_residual": max_coordinate_residual,
        "gates": gates,
        "claim_scope": (
            "A positive result supports a causally used two-coordinate latent contact-impulse "
            "mechanism at predictor block 1 in this model/task. It does not by itself establish "
            "a complete KKT solver or cross-model universality."
        ),
    }
write_json(OUT / "stage25_decision.json", DECISION_PAYLOAD)
(OUT / "FAILURE_TRACE.txt").write_text("NONE\n" if not PIPELINE_FAILED else FAILURE_MESSAGE)
write_json(OUT / "timings.json", TIMINGS)
write_json(OUT / "memory.json", MEMORY)


if not PIPELINE_FAILED and INTERVENTION_ROWS:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    erase_rows = [row for row in INTERVENTION_ROWS if row["condition"] == "impulse_erase"]
    axes[0].scatter(
        [row["native_contact_coefficient"] for row in erase_rows],
        [row["output_transfer_coefficient"] for row in erase_rows],
        alpha=0.6,
    )
    axes[0].set_xlabel("native contact coefficient")
    axes[0].set_ylabel("erasure transfer to ghost")
    labels = PATCH_CONDITIONS
    axes[1].bar(labels, [summaries[name]["mean"] for name in labels])
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_ylabel("mean output transfer")
    axes[2].bar(["construction", "heldout"], [
        IMPULSE_READER_FREEZE["mean_cv_r2"], HELDOUT_READER["mean_r2"]
    ])
    axes[2].set_ylabel("impulse readout mean R²")
    figure.suptitle(DECISION_PAYLOAD["status"])
    figure.tight_layout()
    figure.savefig(PLOT_DIR / "stage25_causal_kkt_summary.png", dpi=160)
    plt.show()
print(json.dumps(DECISION_PAYLOAD, indent=2))
'''


packaging = r'''# Package compact audit evidence and download one result bundle.


INCLUDE_PATHS = [
    "config.json", "versions.json", "source_identity.json", "FAILURE_TRACE.txt",
    "stage25_decision.json", "timings.json", "memory.json", "forward_benchmark.json",
    "hook_identity_test.json", "design/design_freeze.json", "design/candidate_pool_manifest.json",
    "design/physical_selection_freeze.json", "subspaces/stage24_upstream_binding.json",
    "subspaces/impulse_reader_freeze.json", "evaluation_evidence/physical_eligibility_rows.csv",
    "evaluation_evidence/construction_reader_cv.csv", "evaluation_evidence/heldout_impulse_reader.json",
    "evaluation_evidence/heldout_impulse_reader_rows.csv",
    "evaluation_evidence/causal_impulse_intervention_rows.csv",
    "evaluation_evidence/causal_impulse_edit_diagnostics.csv",
    "plots/stage25_causal_kkt_summary.png", "logs/run.log",
    "truth_construction_pool_progress.json", "truth_evaluation_pool_progress.json",
    "construction_baselines_progress.json", "evaluation_baselines_progress.json",
    "intervention_progress.json",
]
staging = OUT / "result_bundle_staging"
staging.mkdir(parents=True, exist_ok=True)
for relative in INCLUDE_PATHS:
    path = OUT / relative
    if not path.exists():
        continue
    destination = staging / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)

archive_base = OUT / f"stage25_causal_kkt_result_bundle_{RUN_SIGNATURE[:12]}"
archive = Path(shutil.make_archive(str(archive_base), "zip", staging))
print(f"RUN_STATUS: {DECISION_PAYLOAD['status']}")
print(f"RESULT_BUNDLE: {archive}")
print(f"RESULT_BUNDLE_SHA256: {sha256_file(archive)}")
if DOWNLOAD_RESULTS:
    try:
        from google.colab import files

        files.download(str(archive))
    except Exception as error:
        print(f"Automatic download unavailable: {error}")
'''


protocol_sources = [source.strip() for source in [
    introduction,
    configuration,
    installation,
    setup,
    analysis_helpers,
    model_helpers,
    upstream_import,
    design,
    truth_generation,
    construction_baselines,
    reader_fit,
    evaluation_open,
    interventions,
    decision,
    packaging,
]]
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
    code(upstream_import),
    code(design),
    code(truth_generation),
    code(construction_baselines),
    code(reader_fit),
    code(evaluation_open),
    code(interventions),
    code(decision),
    code(packaging),
]
for index, cell in enumerate(cells):
    cell["id"] = f"stage25-{index:02d}"

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "L4", "name": TARGET.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
TARGET.write_text(json.dumps(notebook, indent=1) + "\n")
print(f"Wrote {TARGET}")
