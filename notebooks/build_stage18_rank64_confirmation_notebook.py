import ast
import hashlib
import json
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "18_rank64_action_contrast_confirmation.ipynb"
BASE = json.loads((ROOT / "17_finite_action_contrast_interchange.ipynb").read_text())
NUMERICAL = ROOT.parent / "src/cf_faithfulness/stage18_rank_confirmation.py"


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


def base_source(index):
    return "".join(BASE["cells"][index]["source"])


def checked_replace(source, old, new):
    if old not in source:
        raise RuntimeError(f"Stage 17 template changed; missing {old[:100]!r}")
    return source.replace(old, new)


introduction = r'''# Stage 18: preregistered rank-64 action-contrast confirmation

Stage 18 is a **fresh, forward-pass-only confirmation** of the distributed
finite action-contrast signal discovered in Stage 17.  Stage 17 is left
unchanged.  This notebook fixes predictor block 4 and rank 64 before seeing
new model activations; it does not use coordinate readers, Jacobians, JVPs,
VJPs, or intervention-optimized subspaces.

For one state and thirteen finite candidate action sequences,

\[
H(s)=[h(s,a_1)^\top;\ldots;h(s,a_{13})^\top],\qquad
C=I-\tfrac1{13}\mathbf1\mathbf1^\top.
\]

Construction-only ridge regression maps the whitened hidden action contrasts
\(CH\) to a frozen sketch of predicted-future contrasts.  If
\(\widehat W=U\Sigma V^\top\), the primary projector is
\(P_{64}=U_{:64}U_{:64}^\top\).  Sufficiency is tested with

\[
H'=H+\rho(\Pi-I)CHP_{64},
\]

and necessity with

\[
H^-=H-CHP_{64}.
\]

The first edit asks whether the learned space can transplant donor-action
identity.  The second asks whether removing the naturally occurring component
selectively destroys action-dependent predicted-future structure.  Equal-rank
empirical-span random and shuffled-fit controls are evaluated at every tested
rank.  A complete activation swap remains a positive control only.

The physical design uses one independent state per trajectory and a
model-blind simulator eligibility screen.  No-op plus twelve equal-norm radial
action branches are retained only when they generate prespecified contact and
true-cost diversity.  Both construction and evaluation simulator pools are
screened before the model is loaded; evaluation model activations remain
sealed until the subspace and exact executed notebook prefix are frozen.

Pilot mode requires a unique run nonce and refuses an existing output
directory.  The compact result bundle contains raw-shard hashes, generated
versus reused counts, physical eligibility rows, and a montage of rendered
initial/end frames.  Return the downloaded
`stage18_rank64_result_bundle_<signature>.zip`.
'''


configuration = r'''# SINGLE CONFIGURATION BLOCK
# For a source-bound confirmation, create Colab secrets:
# STAGE18_RUN_MODE=pilot
# STAGE18_SOURCE_COMMIT=<full 40 hex>
# STAGE18_RUN_NONCE=<a new unique label for this fresh run>
RUN_MODE = "smoke"
EXPERIMENT_SOURCE_REF = ""
RUN_NONCE = "smoke"
try:
    from google.colab import userdata as _colab_userdata

    RUN_MODE = str(
        _colab_userdata.get("STAGE18_RUN_MODE") or RUN_MODE
    ).strip().lower()
    EXPERIMENT_SOURCE_REF = (
        _colab_userdata.get("STAGE18_SOURCE_COMMIT") or EXPERIMENT_SOURCE_REF
    ).strip()
    RUN_NONCE = str(
        _colab_userdata.get("STAGE18_RUN_NONCE") or RUN_NONCE
    ).strip()
except Exception:
    pass

if RUN_MODE == "pilot":
    if RUN_NONCE in {"", "smoke"}:
        raise ValueError("pilot mode requires a unique STAGE18_RUN_NONCE")
    if not all(value.isalnum() or value in "-_" for value in RUN_NONCE):
        raise ValueError("STAGE18_RUN_NONCE may contain only letters, numbers, '-' and '_'")

MOUNT_DRIVE = True
DOWNLOAD_RESULTS = True
CONTINUE_AFTER_BENCHMARK = True
MAX_ESTIMATED_TOTAL_MINUTES = 180.0
FRESH_RUN_REQUIRED = True

OUTPUT_DIR = "/content/counterfactual_faithfulness_stage18_rank64"
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/counterfactual_faithfulness_stage18_rank64"

PROTOCOL_ID = "stage18-rank64-action-contrast-confirmation-v1"
NOTEBOOK_PROTOCOL_SHA256 = "__PROTOCOL_DIGEST__"
EVIDENCE_STATUS = "CONFIRMATORY_ONLY_IF_SOURCE_BOUND_AND_FRESH"
EXPERIMENT_REPOSITORY = "grewalsk/counterfactual-faithfulness-research"
EXPERIMENT_NOTEBOOK_PATH = "notebooks/18_rank64_action_contrast_confirmation.ipynb"
EXPERIMENT_BUILDER_PATH = "notebooks/build_stage18_rank64_confirmation_notebook.py"
EXPERIMENT_NUMERICAL_PATH = "src/cf_faithfulness/stage18_rank_confirmation.py"

SEED = 18101
DESIGN_SEED = 18137
MODEL_NAME = "jepa_wm_pusht"
ENVIRONMENT = "PushT"
FRAMESKIP = 5
PRIMARY_HORIZON = 3
TARGET_STEPS = [PRIMARY_HORIZON]
FIXED_BLOCK = 4
ACTIVE_BLOCKS = [FIXED_BLOCK]
EXPECTED_CARRIER_CHANNELS = 400

CONSTRUCTION_POOL_TRAJECTORIES = list(range(300, 348))
EVALUATION_POOL_TRAJECTORIES = list(range(400, 464))
CONSTRUCTION_TRAJECTORY_TARGET = 24
EVALUATION_TRAJECTORY_TARGET = 32
STATES_PER_TRAJECTORY = 1
TASK_ID_OFFSET = 800

ACTIONS_PER_STATE = 13
ACTION_MAGNITUDE = 0.12
ACTION_STEPS = PRIMARY_HORIZON * FRAMESKIP
APPROACH_DISTANCE = 80.0
MIN_ELIGIBLE_COST_SPREAD = 0.02
MIN_ELIGIBLE_NON_TIED_PAIR_FRACTION = 0.20
MIN_ELIGIBLE_CONTACT_BRANCHES = 2
PHYSICAL_COST_TIE = 1e-4

OUTPUT_SKETCH_DIM = 256
TRAIN_OUTPUT_SKETCH_SEED = 18161
EVAL_OUTPUT_SKETCH_SEED = 18183
CHANNEL_SHRINKAGE = 0.10
CHANNEL_EIGEN_FLOOR = 1e-6
RIDGE_MULTIPLIERS = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 100.0]
PRIMARY_RANK = 64
SENSITIVITY_RANKS = [16, 32, 64, 96, 128]
MAX_SUBSPACE_RANK = 128
NULL_ROOT_SEED = 18231
PERMUTATION_SEED = 18251
BOOTSTRAP_SEED = 18269
CONSTRUCTION_SHUFFLE_DRAWS = 64
CAUSAL_RANDOM_DRAWS = 4
CAUSAL_DOSES = [-0.5, 0.25, 0.5, 1.0]
BOOTSTRAP_DRAWS = 10000
INTERVENTION_FORWARDS_PER_RECORD = 42

MIN_CONSTRUCTION_CKA = 0.15
MIN_CONSTRUCTION_CKA_ADVANTAGE = 0.03
REQUIRED_POSITIVE_CONSTRUCTION_TRAJECTORIES = 18
MIN_FULL_SWAP_COEFFICIENT = 0.80
MIN_PRIMARY_COEFFICIENT = 0.15
MIN_PRIMARY_COSINE = 0.20
MIN_PRIMARY_GAIN_OVER_RANDOM = 0.05
MIN_PRIMARY_GAIN_OVER_SHUFFLED = 0.05
MAX_PRIMARY_MEAN_SHIFT_RATIO = 0.25
REQUIRED_POSITIVE_EVALUATION_TRAJECTORIES = 24
MIN_NECESSITY_REDUCTION = 0.03
MIN_NECESSITY_GAIN_OVER_RANDOM = 0.02
MIN_NECESSITY_GAIN_OVER_SHUFFLED = 0.02
REQUIRED_POSITIVE_NECESSITY_TRAJECTORIES = 24
MAX_ZERO_EDIT_ERROR = 1e-6

if RUN_MODE == "smoke":
    ACTIVE_CONSTRUCTION_POOL_TRAJECTORIES = CONSTRUCTION_POOL_TRAJECTORIES[:8]
    ACTIVE_EVALUATION_POOL_TRAJECTORIES = EVALUATION_POOL_TRAJECTORIES[:12]
    ACTIVE_CONSTRUCTION_TARGET = 2
    ACTIVE_EVALUATION_TARGET = 2
    ACTIVE_PRIMARY_RANK = 8
    ACTIVE_SENSITIVITY_RANKS = [4, 8]
    ACTIVE_MAX_SUBSPACE_RANK = 8
    ACTIVE_CONSTRUCTION_SHUFFLE_DRAWS = 4
    ACTIVE_CAUSAL_RANDOM_DRAWS = 1
    ACTIVE_CAUSAL_DOSES = [1.0]
    ACTIVE_BOOTSTRAP_DRAWS = 64
elif RUN_MODE == "pilot":
    ACTIVE_CONSTRUCTION_POOL_TRAJECTORIES = CONSTRUCTION_POOL_TRAJECTORIES
    ACTIVE_EVALUATION_POOL_TRAJECTORIES = EVALUATION_POOL_TRAJECTORIES
    ACTIVE_CONSTRUCTION_TARGET = CONSTRUCTION_TRAJECTORY_TARGET
    ACTIVE_EVALUATION_TARGET = EVALUATION_TRAJECTORY_TARGET
    ACTIVE_PRIMARY_RANK = PRIMARY_RANK
    ACTIVE_SENSITIVITY_RANKS = SENSITIVITY_RANKS
    ACTIVE_MAX_SUBSPACE_RANK = MAX_SUBSPACE_RANK
    ACTIVE_CONSTRUCTION_SHUFFLE_DRAWS = CONSTRUCTION_SHUFFLE_DRAWS
    ACTIVE_CAUSAL_RANDOM_DRAWS = CAUSAL_RANDOM_DRAWS
    ACTIVE_CAUSAL_DOSES = CAUSAL_DOSES
    ACTIVE_BOOTSTRAP_DRAWS = BOOTSTRAP_DRAWS
else:
    raise ValueError(
        "STAGE18_RUN_MODE must contain only smoke or pilot; "
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
assert FIXED_BLOCK == 4
assert PRIMARY_RANK in SENSITIVITY_RANKS
assert MAX_SUBSPACE_RANK <= OUTPUT_SKETCH_DIM
assert not set(CONSTRUCTION_POOL_TRAJECTORIES) & set(EVALUATION_POOL_TRAJECTORIES)
assert ACTIVE_PRIMARY_RANK <= ACTIVE_MAX_SUBSPACE_RANK
assert set(ACTIVE_SENSITIVITY_RANKS).issubset(set(range(1, ACTIVE_MAX_SUBSPACE_RANK + 1)))
'''
configuration += "\n\nPROTOCOL_CONFIG_KEYS = " + repr(assigned_uppercase_names(configuration)) + "\n"


installation = base_source(2)


setup = base_source(3)
setup = setup.replace("Stage 17", "Stage 18").replace("STAGE17", "STAGE18")
setup = setup.replace("stage17_action_contrast", "stage18_rank64")
setup = setup.replace("stage17-finite-action-contrast-interchange-v1", "stage18-rank64-action-contrast-confirmation-v1")
setup = setup.replace("notebooks/17_finite_action_contrast_interchange.ipynb", "notebooks/18_rank64_action_contrast_confirmation.ipynb")
setup = setup.replace("notebooks/build_stage17_action_contrast_notebook.py", "notebooks/build_stage18_rank64_confirmation_notebook.py")
setup = setup.replace("src/cf_faithfulness/stage17_action_contrast.py", "src/cf_faithfulness/stage18_rank_confirmation.py")
setup = setup.replace('log = logging.getLogger("stage17_action_contrast")', 'log = logging.getLogger("stage18_rank64")')
setup = checked_replace(
    setup,
    'OUT = Path(OUTPUT_DIR) / f"{RUN_MODE}_{RUN_SIGNATURE[:12]}"\nASSET_DIR = OUT / "assets"',
    'OUT = Path(OUTPUT_DIR) / f"{RUN_MODE}_{RUN_SIGNATURE[:12]}"\n'
    'OUT_PREEXISTED = OUT.exists()\n'
    'if RUN_MODE == "pilot" and FRESH_RUN_REQUIRED and OUT_PREEXISTED:\n'
    '    raise RuntimeError("fresh pilot output already exists; choose a new STAGE18_RUN_NONCE")\n'
    'ASSET_DIR = OUT / "assets"',
)
setup = checked_replace(
    setup,
    "TIMINGS = {}\nMEMORY = []",
    'TIMINGS = {}\nMEMORY = []\nPROVENANCE_COUNTS = {"truth_generated": 0, "baseline_generated": 0, "intervention_generated": 0, "cache_hits": 0}',
)


analysis_helpers = base_source(4)
analysis_helpers = analysis_helpers.replace("Stage 17", "Stage 18")
analysis_helpers += "\n\n\n" + function_sources(
    NUMERICAL.read_text(),
    [
        "projection_ablation_delta",
        "action_contrast_energy_metrics",
        "physical_diversity_metrics",
        "lower_triangle_principal_overlap",
    ],
)


model_helpers = base_source(5).replace("stage17-jepa-wms", "stage18-jepa-wms")
model_helpers = checked_replace(
    model_helpers,
    "    intervention=None,\n    require_grad=False,\n):",
    "    intervention=None,\n):",
)
model_helpers = checked_replace(
    model_helpers,
    "            # With frozen parameters and ordinary inputs, autograd would not\n"
    "            # create a graph. Anchor the requested carrier as a leaf so the\n"
    "            # suffix VJP is exact without retaining the upstream encoder.\n"
    "            if require_grad and not output.requires_grad:\n"
    "                output = output.detach().requires_grad_(True)\n",
    "",
)
model_helpers = checked_replace(
    model_helpers,
    "        with torch.set_grad_enabled(require_grad):",
    "        with torch.inference_mode():",
)


design = r'''# Freeze candidate pools, model-blind eligibility rules, actions, and null seeds.


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
                "time_index": 0,
                "physical_step": 0,
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


ALL_POOL_SPECS = trajectory_specs()
CONSTRUCTION_POOL_SPECS = [
    row for row in ALL_POOL_SPECS
    if row["trajectory_id"] in ACTIVE_CONSTRUCTION_POOL_TRAJECTORIES
]
EVALUATION_POOL_SPECS = [
    row for row in ALL_POOL_SPECS
    if row["trajectory_id"] in ACTIVE_EVALUATION_POOL_TRAJECTORIES
]


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
    if np.max(np.linalg.norm(actions, axis=-1)) > ACTION_MAGNITUDE + 1e-12:
        raise RuntimeError("candidate action bank exceeds frozen magnitude")
    for index in range(1, 7):
        if not np.allclose(actions[index], -actions[index + 6], atol=1e-12):
            raise RuntimeError("radial candidate bank lost antithetic pairing")
    return actions.astype(np.float32)


np.savez_compressed(
    DESIGN_DIR / "stage18_candidate_pool_design.npz",
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
    "candidate_pool_sha256": sha256_file(DESIGN_DIR / "stage18_candidate_pool_design.npz"),
    "pool_manifest_sha256": sha256_file(DESIGN_DIR / "candidate_pool_manifest.json"),
    "fixed_block": FIXED_BLOCK,
    "fixed_primary_rank": PRIMARY_RANK,
    "coordinate_reader_used": False,
    "jacobian_used": False,
    "model_loaded": bool("MODEL" in globals()),
}
if DESIGN_FREEZE["model_loaded"]:
    raise RuntimeError("model was loaded before design freeze")
write_json(DESIGN_DIR / "design_freeze.json", DESIGN_FREEZE)
'''


truth_generation = r'''# Generate and select physical truth before loading any model or encoder.


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


def rollout_dynamic_branch(record, actions):
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
                    "visual": np.asarray(observation["visual"]).copy(),
                    "proprio": np.asarray(observation["proprio"]).copy(),
                }
                endpoint_state = dynamic_state_from_environment(environment)
    finally:
        environment.close()
    if endpoint_observation is None or endpoint_state is None:
        raise RuntimeError("dynamic rollout missed the primary horizon")
    return initial, endpoint_observation, endpoint_state, cumulative


def exact_dynamic_restore_test(record):
    first, first_observation = reset_dynamic_environment(
        record["state"], record_task(record), record["evaluation_seed"]
    )
    second, second_observation = reset_dynamic_environment(
        record["state"], record_task(record), record["evaluation_seed"]
    )
    first_state = dynamic_state_from_environment(first)
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
        "visual_exact": bool(np.array_equal(first_observation["visual"], second_observation["visual"])),
        "proprio_exact": bool(np.array_equal(first_observation["proprio"], second_observation["proprio"])),
        "one_step_continuation_exact": bool(np.allclose(first_next, second_next, atol=1e-12, rtol=0)),
    }
    result["passed"] = bool(all(result.values()))
    if not result["passed"]:
        raise RuntimeError(f"full dynamic restore test failed: {result}")
    return result


def branch_path(record_id):
    return TRUTH_DIR / f"state_{int(record_id):04d}.npz"


def generate_truth(records, progress_name):
    started = time.perf_counter()
    for index, record in enumerate(records):
        destination = branch_path(record["record_id"])
        if destination.exists():
            PROVENANCE_COUNTS["cache_hits"] += 1
            raise RuntimeError(f"fresh-run truth shard already exists: {destination}")
        action_bank = candidate_action_bank(record["state"])
        initials, initial_proprios = [], []
        endpoint_visuals, endpoint_states, interaction_counts = [], [], []
        for action in action_bank:
            initial, endpoint, state, contacts = rollout_dynamic_branch(record, action)
            initials.append(initial["visual"])
            initial_proprios.append(initial["proprio"])
            endpoint_visuals.append(endpoint["visual"])
            endpoint_states.append(state)
            interaction_counts.append(contacts)
        if not all(np.array_equal(initials[0], value) for value in initials[1:]):
            raise AssertionError("initial visual drift across candidate branches")
        if not all(np.array_equal(initial_proprios[0], value) for value in initial_proprios[1:]):
            raise AssertionError("initial proprio drift across candidate branches")
        atomic_npz(
            destination,
            record_id=np.asarray(record["record_id"], dtype=np.int64),
            trajectory_id=np.asarray(record["trajectory_id"], dtype=np.int64),
            task_id=np.asarray(record["task_id"], dtype=np.int64),
            split=np.asarray(record["split"]),
            state=np.asarray(record["state"], dtype=np.float64),
            goal=np.asarray(record["goal"], dtype=np.float64),
            initial_visual=np.asarray(initials[0], dtype=np.uint8),
            initial_proprio=np.asarray(initial_proprios[0], dtype=np.float32),
            selected_actions=action_bank.astype(np.float32),
            endpoint_visuals=np.asarray(endpoint_visuals, dtype=np.uint8),
            endpoint_states=np.asarray(endpoint_states, dtype=np.float64),
            interaction_counts=np.asarray(interaction_counts, dtype=np.int32),
        )
        PROVENANCE_COUNTS["truth_generated"] += 1
        write_json(
            OUT / f"{progress_name}_progress.json",
            {"completed": index + 1, "total": len(records), "last_record_id": int(record["record_id"])},
        )
    TIMINGS[f"{progress_name}_seconds"] = time.perf_counter() - started


def truth_eligibility(record):
    with np.load(branch_path(record["record_id"])) as payload:
        endpoints = payload["endpoint_states"].astype(np.float64)
        contacts = payload["interaction_counts"].astype(np.int64)
        actions = payload["selected_actions"]
        initial_visual = payload["initial_visual"]
    costs = decoded_task_cost(pose_target(endpoints), np.asarray(record["goal"], dtype=np.float64))
    metrics = physical_diversity_metrics(costs, contacts, tie=PHYSICAL_COST_TIE)
    eligible = bool(
        metrics["cost_spread"] >= MIN_ELIGIBLE_COST_SPREAD
        and metrics["non_tied_pair_fraction"] >= MIN_ELIGIBLE_NON_TIED_PAIR_FRACTION
        and metrics["contact_branches"] >= MIN_ELIGIBLE_CONTACT_BRANCHES
    )
    return {
        "record_id": int(record["record_id"]),
        "trajectory_id": int(record["trajectory_id"]),
        "task_id": int(record["task_id"]),
        "split": record["split"],
        **metrics,
        "eligible": eligible,
        "action_sha256": array_sha256(actions),
        "endpoint_state_sha256": array_sha256(endpoints),
        "initial_visual_sha256": array_sha256(initial_visual),
    }


def select_records(records, target):
    rows = [truth_eligibility(record) for record in records]
    selected_ids = [row["record_id"] for row in rows if row["eligible"]][: int(target)]
    if len(selected_ids) != int(target):
        raise RuntimeError(
            f"physical eligibility produced {len(selected_ids)} records but requires {target}"
        )
    chosen = [record for record in records if record["record_id"] in selected_ids]
    return chosen, rows


def freeze_maps(construction_records, evaluation_records):
    permutations = {}
    wrong = {}
    for records in [construction_records, evaluation_records]:
        identifiers = sorted(int(record["record_id"]) for record in records)
        for index, record_id in enumerate(identifiers):
            permutations[str(record_id)] = fixed_derangement(
                ACTIONS_PER_STATE, stable_seed(PERMUTATION_SEED, record_id, "donor")
            ).tolist()
            wrong[str(record_id)] = identifiers[(index + 1) % len(identifiers)]
    return permutations, wrong


def make_truth_montage(records):
    sample = records[:4] + records[-4:]
    figure, axes = plt.subplots(len(sample), 3, figsize=(9, 2.8 * len(sample)))
    for row_index, record in enumerate(sample):
        with np.load(branch_path(record["record_id"])) as payload:
            initial = payload["initial_visual"]
            endpoint_visuals = payload["endpoint_visuals"]
            endpoints = payload["endpoint_states"].astype(np.float64)
        costs = decoded_task_cost(pose_target(endpoints), np.asarray(record["goal"], dtype=np.float64))
        best, worst = int(np.argmin(costs)), int(np.argmax(costs))
        for column, (image, title) in enumerate(
            [(initial, "initial"), (endpoint_visuals[best], f"best a={best}"), (endpoint_visuals[worst], f"worst a={worst}")]
        ):
            axes[row_index, column].imshow(image)
            axes[row_index, column].set_title(title)
            axes[row_index, column].axis("off")
        axes[row_index, 0].set_ylabel(f"{record['split']} {record['record_id']}")
    figure.tight_layout()
    figure.savefig(PLOT_DIR / "stage18_truth_montage.png", dpi=150)
    plt.close(figure)


if not PIPELINE_FAILED:
    try:
        REPO = configure_repo()
        RESTORE_TEST = exact_dynamic_restore_test(CONSTRUCTION_POOL_SPECS[0])
        write_json(OUT / "restore_test.json", RESTORE_TEST)
        generate_truth(CONSTRUCTION_POOL_SPECS, "truth_construction_pool")
        generate_truth(EVALUATION_POOL_SPECS, "truth_evaluation_pool")
        if "MODEL" in globals():
            raise RuntimeError("model was loaded before physical eligibility selection")
        CONSTRUCTION_RECORDS, CONSTRUCTION_ELIGIBILITY_ROWS = select_records(
            CONSTRUCTION_POOL_SPECS, ACTIVE_CONSTRUCTION_TARGET
        )
        EVALUATION_RECORDS, EVALUATION_ELIGIBILITY_ROWS = select_records(
            EVALUATION_POOL_SPECS, ACTIVE_EVALUATION_TARGET
        )
        ACTIVE_CONSTRUCTION_TRAJECTORIES = [row["trajectory_id"] for row in CONSTRUCTION_RECORDS]
        ACTIVE_EVALUATION_TRAJECTORIES = [row["trajectory_id"] for row in EVALUATION_RECORDS]
        donor_permutations, wrong_state_map = freeze_maps(CONSTRUCTION_RECORDS, EVALUATION_RECORDS)
        all_eligibility = CONSTRUCTION_ELIGIBILITY_ROWS + EVALUATION_ELIGIBILITY_ROWS
        write_csv(EVIDENCE_DIR / "physical_eligibility_rows.csv", all_eligibility)
        SELECTION_CERTIFICATE = {
            "selection_completed_before_model_load": True,
            "selection_used_only_simulator_truth": True,
            "construction_selected_ids": ACTIVE_CONSTRUCTION_TRAJECTORIES,
            "evaluation_selected_ids": ACTIVE_EVALUATION_TRAJECTORIES,
            "construction_eligible_pool_count": int(sum(row["eligible"] for row in CONSTRUCTION_ELIGIBILITY_ROWS)),
            "evaluation_eligible_pool_count": int(sum(row["eligible"] for row in EVALUATION_ELIGIBILITY_ROWS)),
            "donor_permutations": donor_permutations,
            "wrong_state_map": wrong_state_map,
            "eligibility_rows_sha256": sha256_file(EVIDENCE_DIR / "physical_eligibility_rows.csv"),
        }
        write_json(DESIGN_DIR / "physical_selection_freeze.json", SELECTION_CERTIFICATE)
        make_truth_montage(CONSTRUCTION_RECORDS + EVALUATION_RECORDS)
        memory_report("physical_truth_and_selection_complete")
    except Exception:
        record_failure("physical_truth_selection")
'''


model_and_construction = base_source(8)
model_and_construction = model_and_construction.replace("Stage 17", "Stage 18")
model_and_construction = checked_replace(
    model_and_construction,
    '        if destination.exists():\n            with np.load(destination) as existing:\n                saved_blocks = existing["blocks"].astype(int).tolist()\n            if saved_blocks == list(blocks):\n                continue\n            destination.unlink()',
    '        if destination.exists():\n            PROVENANCE_COUNTS["cache_hits"] += 1\n            raise RuntimeError(f"fresh-run baseline shard already exists: {destination}")',
)
model_and_construction = checked_replace(
    model_and_construction,
    '        write_json(\n            OUT / f"baseline_{record[\'split\']}_progress.json",',
    '        PROVENANCE_COUNTS["baseline_generated"] += 1\n        write_json(\n            OUT / f"baseline_{record[\'split\']}_progress.json",',
)
model_and_construction = checked_replace(
    model_and_construction,
    '    interventions_per_record = len(ACTIVE_CAUSAL_DOSES) + 3 + ACTIVE_CAUSAL_RANDOM_DRAWS + len(ACTIVE_SENSITIVITY_RANKS)',
    '    interventions_per_record = INTERVENTION_FORWARDS_PER_RECORD if RUN_MODE == "pilot" else 12',
)
model_and_construction = checked_replace(
    model_and_construction,
    '    total_eval_records = len(ACTIVE_EVALUATION_TRAJECTORIES) * len(ACTIVE_TIME_INDICES)',
    '    total_eval_records = int(ACTIVE_EVALUATION_TARGET)',
)
model_and_construction = checked_replace(
    model_and_construction,
    '        extract_baselines(CONSTRUCTION_RECORDS, ACTIVE_BLOCKS)',
    '        extract_baselines(CONSTRUCTION_RECORDS, [FIXED_BLOCK])',
)


construction_geometry = r'''# Verify the fixed block-4 construction geometry without selecting a layer.


def fixed_block_geometry_rows():
    rows = []
    output_by_record = {
        int(record["record_id"]): load_baseline(record["record_id"])["output_train_sketch"].astype(np.float64)
        for record in CONSTRUCTION_RECORDS
    }
    for record in CONSTRUCTION_RECORDS:
        record_id = int(record["record_id"])
        payload = load_baseline(record_id)
        carrier = carrier_for_block(payload, FIXED_BLOCK)
        output = output_by_record[record_id]
        wrong_output = output_by_record[int(wrong_state_map[str(record_id)])]
        shuffled = []
        for draw in range(ACTIVE_CONSTRUCTION_SHUFFLE_DRAWS):
            permutation = fixed_derangement(
                ACTIONS_PER_STATE,
                stable_seed(PERMUTATION_SEED, record_id, FIXED_BLOCK, draw, "geometry"),
            )
            shuffled.append(linear_cka(carrier, output[permutation]))
        observed = linear_cka(carrier, output)
        wrong = linear_cka(carrier, wrong_output)
        rows.append(
            {
                "record_id": record_id,
                "trajectory_id": int(record["trajectory_id"]),
                "block": FIXED_BLOCK,
                "observed_cka": float(observed),
                "shuffled_cka": float(np.mean(shuffled)),
                "wrong_state_cka": float(wrong),
                "shuffle_advantage": float(observed - np.mean(shuffled)),
                "wrong_state_advantage": float(observed - wrong),
            }
        )
    write_csv(ANALYSIS_DIR / "fixed_block_geometry_rows.csv", rows)
    return rows


def fixed_block_gate(rows):
    observed = np.asarray([row["observed_cka"] for row in rows])
    shuffle_advantage = np.asarray([row["shuffle_advantage"] for row in rows])
    payload = {
        "fixed_before_new_data": True,
        "fixed_block": FIXED_BLOCK,
        "selection_rule": "block 4 fixed from Stage 17; no Stage 18 layer selection",
        "trajectories": len(rows),
        "mean_cka": float(np.mean(observed)),
        "standard_error_cka": float(np.std(observed, ddof=1) / np.sqrt(len(observed))) if len(observed) > 1 else 0.0,
        "mean_shuffle_advantage": float(np.mean(shuffle_advantage)),
        "mean_wrong_state_advantage": float(np.mean([row["wrong_state_advantage"] for row in rows])),
        "positive_shuffle_advantage_trajectories": int(np.sum(shuffle_advantage > 0)),
        "evaluation_model_activations_seen": [],
    }
    payload["construction_gate_pass"] = bool(
        payload["mean_cka"] >= MIN_CONSTRUCTION_CKA
        and payload["mean_shuffle_advantage"] >= MIN_CONSTRUCTION_CKA_ADVANTAGE
        and payload["positive_shuffle_advantage_trajectories"]
        >= min(REQUIRED_POSITIVE_CONSTRUCTION_TRAJECTORIES, len(rows))
    )
    write_json(ANALYSIS_DIR / "fixed_block_construction_gate.json", payload)
    return payload


if not PIPELINE_FAILED:
    try:
        CONSTRUCTION_GEOMETRY_ROWS = fixed_block_geometry_rows()
        CONSTRUCTION_GATE = fixed_block_gate(CONSTRUCTION_GEOMETRY_ROWS)
        CONSTRUCTION_GATE_PASS = bool(CONSTRUCTION_GATE["construction_gate_pass"])
        print(json.dumps(CONSTRUCTION_GATE, indent=2))
    except Exception:
        record_failure("fixed_block_geometry")
'''


subspace_fit = r'''# Fit and freeze nested rank-16/32/64/96/128 subspaces on construction data only.


def construction_matrices():
    count = 0
    total = np.zeros(EXPECTED_CARRIER_CHANNELS, dtype=np.float64)
    cross = np.zeros((EXPECTED_CARRIER_CHANNELS, EXPECTED_CARRIER_CHANNELS), dtype=np.float64)
    native_residuals, output_residuals, trajectory_groups, record_slices = [], [], [], []
    offset = 0
    for record in CONSTRUCTION_RECORDS:
        payload = load_baseline(record["record_id"])
        carrier = carrier_for_block(payload, FIXED_BLOCK).astype(np.float64)
        channels = carrier.reshape(-1, carrier.shape[-1])
        count += len(channels)
        total += channels.sum(axis=0)
        cross += channels.T @ channels
        native_residuals.append(candidate_center(carrier))
        output_residuals.append(candidate_center(payload["output_train_sketch"]))
        trajectory_groups.extend([record["trajectory_id"]] * ACTIONS_PER_STATE)
        record_slices.append((int(record["record_id"]), offset, offset + ACTIONS_PER_STATE))
        offset += ACTIONS_PER_STATE
    metric = channel_metric_from_moments(
        count, total, cross, shrinkage=CHANNEL_SHRINKAGE, relative_floor=CHANNEL_EIGEN_FLOOR
    )
    whitened = [
        transform_primal_channels(value, metric["inverse_square_root"])
        for value in native_residuals
    ]
    x = np.concatenate([value.reshape(ACTIONS_PER_STATE, -1) for value in whitened])
    x /= np.sqrt(x.shape[1])
    y = np.concatenate(output_residuals).astype(np.float64)
    output_scale = np.std(y, axis=0, ddof=1)
    positive_scale = output_scale[output_scale > 1e-12]
    if not len(positive_scale):
        raise RuntimeError("construction output sketch has zero action variance")
    output_scale = np.maximum(output_scale, np.median(positive_scale) * 1e-3)
    y /= output_scale[None]
    return x.astype(np.float32), y.astype(np.float32), np.asarray(trajectory_groups), record_slices, metric, output_scale


def fit_basis_gpu(features, targets, penalty, max_rank):
    x = torch.as_tensor(features, device="cuda", dtype=torch.float32)
    y = torch.as_tensor(targets, device="cuda", dtype=torch.float32)
    gram = x @ x.T
    alpha = torch.linalg.solve(gram + float(penalty) * torch.eye(len(gram), device="cuda"), y)
    weight = x.T @ alpha
    left, singular, _ = torch.linalg.svd(weight, full_matrices=False)
    keep = min(int(max_rank), left.shape[1], int(torch.sum(singular > 1e-7).item()))
    if keep < int(ACTIVE_MAX_SUBSPACE_RANK):
        raise RuntimeError(f"ridge map rank {keep} is below required rank {ACTIVE_MAX_SUBSPACE_RANK}")
    result = left[:, :keep].detach().cpu().numpy().astype(np.float32)
    singular_values = singular.detach().cpu().numpy().astype(np.float64)
    del x, y, gram, alpha, weight, left, singular
    torch.cuda.empty_cache()
    return result, singular_values


def random_basis_gpu(features, rank, seed, excluded):
    x = torch.as_tensor(features, device="cuda", dtype=torch.float32)
    excluded_tensor = torch.as_tensor(excluded, device="cuda", dtype=torch.float32)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(int(seed))
    coefficients = torch.randn(
        len(x), int(rank) + 16, generator=generator, device="cuda", dtype=torch.float32
    )
    candidate = x.T @ coefficients
    candidate -= excluded_tensor @ (excluded_tensor.T @ candidate)
    basis, triangular = torch.linalg.qr(candidate, mode="reduced")
    diagonal = torch.abs(torch.diag(triangular))
    if int(torch.sum(diagonal > torch.max(diagonal) * 1e-6).item()) < int(rank):
        raise RuntimeError("empirical action span is too small for requested random rank")
    result = basis[:, : int(rank)].detach().cpu().numpy().astype(np.float32)
    del x, excluded_tensor, coefficients, candidate, basis, triangular, diagonal
    torch.cuda.empty_cache()
    return result


def fit_and_freeze_subspaces():
    x, y, groups, record_slices, metric, output_scale = construction_matrices()
    theoretical_rank_ceiling = min(
        len(np.unique(groups)) * (ACTIONS_PER_STATE - 1), y.shape[1], x.shape[1]
    )
    if ACTIVE_MAX_SUBSPACE_RANK > theoretical_rank_ceiling:
        raise RuntimeError(
            f"requested rank {ACTIVE_MAX_SUBSPACE_RANK} exceeds ceiling {theoretical_rank_ceiling}"
        )
    kernel = np.asarray(x, dtype=np.float64) @ np.asarray(x, dtype=np.float64).T
    ridge = grouped_kernel_ridge_cv(kernel, y, groups, RIDGE_MULTIPLIERS)
    write_csv(ANALYSIS_DIR / "ridge_group_cv.csv", ridge["rows"])
    primary_basis, singular_values = fit_basis_gpu(
        x, y, ridge["penalty"], ACTIVE_MAX_SUBSPACE_RANK
    )
    shuffled_y = y.copy()
    for record_id, start, stop in record_slices:
        permutation = fixed_derangement(
            ACTIONS_PER_STATE, stable_seed(PERMUTATION_SEED, record_id, "fit_shuffle")
        )
        shuffled_y[start:stop] = y[start:stop][permutation]
    shuffled_basis, shuffled_singular = fit_basis_gpu(
        x, shuffled_y, ridge["penalty"], ACTIVE_MAX_SUBSPACE_RANK
    )
    random_bases = []
    for draw in range(ACTIVE_CAUSAL_RANDOM_DRAWS):
        random_bases.append(
            random_basis_gpu(
                x,
                ACTIVE_MAX_SUBSPACE_RANK,
                stable_seed(NULL_ROOT_SEED, draw, "empirical_span"),
                primary_basis[:, :ACTIVE_MAX_SUBSPACE_RANK],
            )
        )
    destination = SUBSPACE_DIR / "frozen_rank64_confirmation_subspaces.npz"
    arrays = {
        "primary_basis": primary_basis,
        "shuffled_basis": shuffled_basis,
        "primary_singular_values": singular_values,
        "shuffled_singular_values": shuffled_singular,
        "channel_mean": metric["mean"],
        "channel_covariance": metric["covariance"],
        "channel_square_root": metric["square_root"],
        "channel_inverse_square_root": metric["inverse_square_root"],
        "output_scale": output_scale,
    }
    for draw, basis in enumerate(random_bases):
        arrays[f"random_basis_{draw:02d}"] = basis
    atomic_npz(destination, **arrays)
    manifest = {
        "fixed_block": FIXED_BLOCK,
        "primary_rank": ACTIVE_PRIMARY_RANK,
        "sensitivity_ranks": ACTIVE_SENSITIVITY_RANKS,
        "max_rank": ACTIVE_MAX_SUBSPACE_RANK,
        "output_sketch_dimension": OUTPUT_SKETCH_DIM,
        "theoretical_rank_ceiling": int(theoretical_rank_ceiling),
        "selected_ridge_multiplier": ridge["selected_multiplier"],
        "ridge_penalty": ridge["penalty"],
        "channel_condition_number": metric["condition_number"],
        "primary_basis_shape": list(primary_basis.shape),
        "shuffled_basis_shape": list(shuffled_basis.shape),
        "random_draws": len(random_bases),
        "random_bases_orthogonal_to_primary_max_rank": True,
        "shuffled_primary_overlap": lower_triangle_principal_overlap(
            primary_basis[:, :ACTIVE_PRIMARY_RANK], shuffled_basis[:, :ACTIVE_PRIMARY_RANK]
        ),
        "subspace_sha256": sha256_file(destination),
        "construction_trajectory_ids": sorted(set(groups.astype(int).tolist())),
        "evaluation_model_activations_seen": [],
        "evaluation_simulator_truth_used_only_for_frozen_eligibility": True,
        "full_activation_swap_is_positive_control_only": True,
        "jacobians_computed": False,
    }
    write_json(SUBSPACE_DIR / "subspace_manifest.json", manifest)
    freeze = {
        "frozen_before_evaluation_model_activations": True,
        "source_identity": SOURCE_IDENTITY,
        "design_freeze_sha256": sha256_file(DESIGN_DIR / "design_freeze.json"),
        "physical_selection_freeze_sha256": sha256_file(DESIGN_DIR / "physical_selection_freeze.json"),
        "fixed_block_gate_sha256": sha256_file(ANALYSIS_DIR / "fixed_block_construction_gate.json"),
        "subspace_manifest": manifest,
    }
    write_json(SUBSPACE_DIR / "subspace_freeze.json", freeze)
    return manifest


if not PIPELINE_FAILED:
    try:
        SUBSPACE_MANIFEST = fit_and_freeze_subspaces()
        print(json.dumps(SUBSPACE_MANIFEST, indent=2))
        memory_report("subspaces_frozen")
    except Exception:
        record_failure("subspace_fit")
'''


evaluation_open = r'''# Open evaluation model activations only after the fixed hypothesis and subspaces are frozen.
EVALUATION_OPENED = False
if not PIPELINE_FAILED:
    try:
        if RUN_MODE == "pilot" and not CONSTRUCTION_GATE_PASS:
            write_json(
                OUT / "evaluation_open_certificate.json",
                {"opened": False, "reason": "STOP_NO_FIXED_BLOCK_ACTION_GEOMETRY"},
            )
        else:
            verify_executed_notebook_through(
                "# Open evaluation model activations only after the fixed hypothesis and subspaces are frozen."
            )
            if not (SUBSPACE_DIR / "subspace_freeze.json").exists():
                raise RuntimeError("subspace freeze is absent")
            extract_baselines(EVALUATION_RECORDS, [FIXED_BLOCK])
            EVALUATION_OPENED = True
            write_json(
                OUT / "evaluation_open_certificate.json",
                {
                    "opened": True,
                    "source_identity": SOURCE_IDENTITY,
                    "subspace_freeze_sha256": sha256_file(SUBSPACE_DIR / "subspace_freeze.json"),
                    "physical_selection_freeze_sha256": sha256_file(DESIGN_DIR / "physical_selection_freeze.json"),
                    "evaluation_trajectory_ids": ACTIVE_EVALUATION_TRAJECTORIES,
                    "opened_after_subspace_freeze": True,
                    "evaluation_model_activations_seen_during_fit": [],
                },
            )
            memory_report("evaluation_baselines_complete")
    except Exception:
        record_failure("evaluation_open")
'''


causal_interchange = r'''# Run rank-matched sufficiency and necessity interventions on held-out trajectories.


def load_frozen_subspaces():
    with np.load(SUBSPACE_DIR / "frozen_rank64_confirmation_subspaces.npz") as payload:
        return {name: payload[name].copy() for name in payload.files}


def whiten_carrier(values, subspaces):
    return transform_primal_channels(
        np.asarray(values, dtype=np.float64), subspaces["channel_inverse_square_root"]
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
    return INTERVENTION_DIR / f"state_{int(record_id):04d}.json"


def finite_json_rows(rows):
    return [
        {
            key: None if isinstance(value, (float, np.floating)) and not np.isfinite(value) else value
            for key, value in row.items()
        }
        for row in rows
    ]


def wrong_state_delta(current_white, wrong_white, permutation, basis):
    current_residual = candidate_center(current_white.reshape(ACTIONS_PER_STATE, -1))
    wrong_residual = candidate_center(wrong_white.reshape(ACTIONS_PER_STATE, -1))
    difference = wrong_residual[permutation] - current_residual
    return ((difference @ basis) @ basis.T).reshape(current_white.shape)


def intervention_specs(record, carrier, subspaces):
    record_id = int(record["record_id"])
    permutation = np.asarray(donor_permutations[str(record_id)], dtype=np.int64)
    white = whiten_carrier(carrier, subspaces)
    full_swap = action_swap_delta(white, permutation, basis=None, dose=1.0)
    primary_basis = subspaces["primary_basis"][:, :ACTIVE_PRIMARY_RANK]
    primary_swap = action_swap_delta(white, permutation, basis=primary_basis, dose=1.0)
    primary_ablation = projection_ablation_delta(white, primary_basis, dose=1.0)
    if min(np.linalg.norm(primary_swap), np.linalg.norm(primary_ablation)) <= 1e-12:
        raise RuntimeError("primary rank intervention is degenerate")
    specs = []

    def add(condition, family, mode, rank, dose, delta):
        specs.append(
            {
                "condition": condition,
                "family": family,
                "mode": mode,
                "rank": int(rank),
                "dose": float(dose),
                "delta_white": np.asarray(delta, dtype=np.float64),
            }
        )

    for dose in ACTIVE_CAUSAL_DOSES:
        add(
            f"primary_r{ACTIVE_PRIMARY_RANK:03d}", "primary", "sufficiency",
            ACTIVE_PRIMARY_RANK, dose, float(dose) * primary_swap,
        )
    for rank in ACTIVE_SENSITIVITY_RANKS:
        learned = action_swap_delta(
            white, permutation, basis=subspaces["primary_basis"][:, :rank], dose=1.0
        )
        if rank != ACTIVE_PRIMARY_RANK:
            add(f"learned_r{rank:03d}", "rank_sensitivity", "sufficiency", rank, 1.0, learned)
        shuffled = action_swap_delta(
            white, permutation, basis=subspaces["shuffled_basis"][:, :rank], dose=1.0
        )
        add(
            f"shuffled_r{rank:03d}", "matched_shuffled_control", "sufficiency",
            rank, 1.0, norm_match(shuffled, learned),
        )
        for draw in range(ACTIVE_CAUSAL_RANDOM_DRAWS):
            random_delta = action_swap_delta(
                white, permutation, basis=subspaces[f"random_basis_{draw:02d}"][:, :rank], dose=1.0
            )
            add(
                f"random_r{rank:03d}_{draw:02d}", "empirical_span_random_control",
                "sufficiency", rank, 1.0, norm_match(random_delta, learned),
            )

    wrong_id = int(wrong_state_map[str(record_id)])
    wrong_carrier = carrier_for_block(load_baseline(wrong_id), FIXED_BLOCK)
    wrong = wrong_state_delta(
        white, whiten_carrier(wrong_carrier, subspaces), permutation, primary_basis
    )
    add(
        f"wrong_state_r{ACTIVE_PRIMARY_RANK:03d}", "state_specificity_control", "sufficiency",
        ACTIVE_PRIMARY_RANK, 1.0, norm_match(wrong, primary_swap),
    )
    add(
        f"common_mode_r{ACTIVE_PRIMARY_RANK:03d}", "matched_common_mode_control", "sufficiency",
        ACTIVE_PRIMARY_RANK, 1.0, matched_common_mode(primary_swap, primary_basis[:, 0]),
    )
    add("full_activation_swap", "positive_control_only", "sufficiency", -1, 1.0, full_swap)

    add(
        f"ablate_primary_r{ACTIVE_PRIMARY_RANK:03d}", "primary", "necessity",
        ACTIVE_PRIMARY_RANK, 1.0, primary_ablation,
    )
    shuffled_ablation = projection_ablation_delta(
        white, subspaces["shuffled_basis"][:, :ACTIVE_PRIMARY_RANK], dose=1.0
    )
    add(
        f"ablate_shuffled_r{ACTIVE_PRIMARY_RANK:03d}", "matched_shuffled_control", "necessity",
        ACTIVE_PRIMARY_RANK, 1.0, norm_match(shuffled_ablation, primary_ablation),
    )
    for draw in range(ACTIVE_CAUSAL_RANDOM_DRAWS):
        random_ablation = projection_ablation_delta(
            white, subspaces[f"random_basis_{draw:02d}"][:, :ACTIVE_PRIMARY_RANK], dose=1.0
        )
        add(
            f"ablate_random_r{ACTIVE_PRIMARY_RANK:03d}_{draw:02d}", "empirical_span_random_control",
            "necessity", ACTIVE_PRIMARY_RANK, 1.0,
            norm_match(random_ablation, primary_ablation),
        )

    if RUN_MODE == "pilot" and len(specs) != INTERVENTION_FORWARDS_PER_RECORD:
        raise RuntimeError(
            f"expected {INTERVENTION_FORWARDS_PER_RECORD} interventions, found {len(specs)}"
        )
    for specification in specs:
        specification["edit_norm"] = float(np.linalg.norm(specification["delta_white"]))
        specification["primary_swap_norm"] = float(np.linalg.norm(primary_swap))
        specification["primary_ablation_norm"] = float(np.linalg.norm(primary_ablation))
        specification["full_swap_norm"] = float(np.linalg.norm(full_swap))
    return permutation, specs


def make_result_row(record, condition, family, mode, rank, dose, permutation,
                    baseline_output, patched_output, baseline_pose, patched_pose,
                    true_cost, goal, edit_norm, primary_swap_norm,
                    primary_ablation_norm, full_swap_norm):
    output_metrics = donor_transfer_metrics(baseline_output, patched_output, permutation)
    pose_metrics = donor_transfer_metrics(baseline_pose, patched_pose, permutation)
    energy = action_contrast_energy_metrics(baseline_output, patched_output)
    planning = ranking_metrics(true_cost, decoded_task_cost(patched_pose, goal))
    return {
        "record_id": int(record["record_id"]),
        "trajectory_id": int(record["trajectory_id"]),
        "task_id": int(record["task_id"]),
        "selected_block": FIXED_BLOCK,
        "condition": condition,
        "family": family,
        "mode": mode,
        "rank": int(rank),
        "dose": float(dose),
        "output_coefficient": output_metrics["coefficient"],
        "output_cosine": output_metrics["cosine"],
        "output_reconstruction": output_metrics["reconstruction"],
        "output_mean_shift_ratio": output_metrics["mean_shift_ratio"],
        "output_contrast_energy_retention": energy["energy_retention"],
        "output_contrast_energy_reduction": energy["energy_reduction"],
        "output_contrast_cosine": energy["contrast_cosine"],
        "pose_coefficient": pose_metrics["coefficient"],
        "pose_cosine": pose_metrics["cosine"],
        "normalized_regret": planning["normalized_regret"],
        "weighted_pairwise_accuracy": planning["weighted_pairwise_accuracy"],
        "top1_correct": planning["top1_correct"],
        "selected_action": planning["selected_action"],
        "oracle_action": planning["oracle_action"],
        "output_rms_change": float(np.sqrt(np.mean((patched_output - baseline_output) ** 2))),
        "carrier_edit_whitened_norm": float(edit_norm),
        "primary_swap_norm": float(primary_swap_norm),
        "primary_ablation_norm": float(primary_ablation_norm),
        "full_swap_norm": float(full_swap_norm),
        "edit_to_full_swap_ratio": float(edit_norm) / max(float(full_swap_norm), 1e-12),
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
    true_cost, goal = truth_costs(record)
    permutation, specifications = intervention_specs(record, carrier, subspaces)
    rows = [
        make_result_row(
            record, "no_edit", "baseline", "baseline", 0, 0.0, permutation,
            baseline_output, baseline_output, baseline_pose, baseline_pose,
            true_cost, goal, 0.0, 0.0, 0.0, 0.0,
        )
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
        rows.append(
            make_result_row(
                record,
                specification["condition"],
                specification["family"],
                specification["mode"],
                specification["rank"],
                specification["dose"],
                permutation,
                baseline_output,
                patched_output,
                baseline_pose,
                patched_pose,
                true_cost,
                goal,
                specification["edit_norm"],
                specification["primary_swap_norm"],
                specification["primary_ablation_norm"],
                specification["full_swap_norm"],
            )
        )
        del patched, patched_output, patched_pose, delta_tensor
    write_json(destination, finite_json_rows(rows))
    PROVENANCE_COUNTS["intervention_generated"] += 1
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
            {"completed": index + 1, "total": len(records), "last_record_id": int(record["record_id"])},
        )
    TIMINGS["causal_intervention_seconds"] = time.perf_counter() - started
    write_csv(EVIDENCE_DIR / "intervention_state_rows.csv", rows)
    return rows


if not PIPELINE_FAILED and EVALUATION_OPENED:
    try:
        INTERVENTION_ROWS = run_all_interventions(EVALUATION_RECORDS)
        memory_report("causal_interventions_complete")
    except Exception:
        record_failure("causal_interchange")
'''


decision_and_plots = r'''# Aggregate by independent trajectory and apply frozen sufficiency/necessity gates.


def trajectory_summaries(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["trajectory_id"], row["condition"], float(row["dose"]))].append(row)
    summaries = []
    metric_names = [
        "output_coefficient", "output_cosine", "output_reconstruction",
        "output_mean_shift_ratio", "output_contrast_energy_retention",
        "output_contrast_energy_reduction", "output_contrast_cosine",
        "pose_coefficient", "pose_cosine", "normalized_regret",
        "weighted_pairwise_accuracy", "output_rms_change",
        "edit_to_full_swap_ratio",
    ]
    for (trajectory_id, condition, dose), values in sorted(grouped.items()):
        row = {
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


def lookup(summary, trajectory_id, condition, dose=1.0, key="output_coefficient"):
    values = [
        row[key]
        for row in summary
        if row["trajectory_id"] == trajectory_id
        and row["condition"] == condition
        and np.isclose(row["dose"], dose)
    ]
    return float(values[0]) if len(values) == 1 else np.nan


def bootstrap_interval(values, trajectories, seed_offset=0):
    draws = clustered_bootstrap_mean(
        np.asarray(values), np.asarray(trajectories), ACTIVE_BOOTSTRAP_DRAWS,
        BOOTSTRAP_SEED + int(seed_offset),
    )
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def rank_curve(summary, trajectories):
    rows = []
    for rank in ACTIVE_SENSITIVITY_RANKS:
        learned_name = (
            f"primary_r{rank:03d}" if rank == ACTIVE_PRIMARY_RANK else f"learned_r{rank:03d}"
        )
        learned = np.asarray([lookup(summary, value, learned_name) for value in trajectories])
        shuffled = np.asarray(
            [lookup(summary, value, f"shuffled_r{rank:03d}") for value in trajectories]
        )
        random = []
        for trajectory_id in trajectories:
            values = [
                lookup(summary, trajectory_id, f"random_r{rank:03d}_{draw:02d}")
                for draw in range(ACTIVE_CAUSAL_RANDOM_DRAWS)
            ]
            random.append(float(np.nanmedian(values)))
        rows.append(
            {
                "rank": int(rank),
                "mean_learned_coefficient": float(np.nanmean(learned)),
                "mean_random_coefficient": float(np.nanmean(random)),
                "mean_shuffled_coefficient": float(np.nanmean(shuffled)),
                "mean_gain_over_random": float(np.nanmean(learned - np.asarray(random))),
                "mean_gain_over_shuffled": float(np.nanmean(learned - shuffled)),
            }
        )
    write_csv(ANALYSIS_DIR / "rank_scaling_summary.csv", rows)
    return rows


def evaluate_confirmation_gate(summary):
    trajectories = sorted({row["trajectory_id"] for row in summary})
    primary_name = f"primary_r{ACTIVE_PRIMARY_RANK:03d}"
    primary = np.asarray([lookup(summary, value, primary_name) for value in trajectories])
    primary_cosine = np.asarray(
        [lookup(summary, value, primary_name, key="output_cosine") for value in trajectories]
    )
    primary_shift = np.asarray(
        [lookup(summary, value, primary_name, key="output_mean_shift_ratio") for value in trajectories]
    )
    full = np.asarray([lookup(summary, value, "full_activation_swap") for value in trajectories])
    shuffled = np.asarray(
        [lookup(summary, value, f"shuffled_r{ACTIVE_PRIMARY_RANK:03d}") for value in trajectories]
    )
    random_values = []
    for trajectory_id in trajectories:
        values = [
            lookup(summary, trajectory_id, f"random_r{ACTIVE_PRIMARY_RANK:03d}_{draw:02d}")
            for draw in range(ACTIVE_CAUSAL_RANDOM_DRAWS)
        ]
        random_values.append(float(np.nanmedian(values)))
    random_values = np.asarray(random_values)
    gain_random = primary - random_values
    gain_shuffled = primary - shuffled
    sufficiency_sign = exact_positive_sign_test(gain_random)

    positive_doses = sorted(value for value in ACTIVE_CAUSAL_DOSES if value > 0)
    dose_slopes = []
    if len(positive_doses) >= 2:
        for trajectory_id in trajectories:
            values = np.asarray(
                [lookup(summary, trajectory_id, primary_name, dose=value) for value in positive_doses]
            )
            dose_slopes.append(float(np.polyfit(positive_doses, values, 1)[0]))
    else:
        dose_slopes = [math.nan] * len(trajectories)
    negative = (
        np.asarray([lookup(summary, value, primary_name, dose=-0.5) for value in trajectories])
        if -0.5 in ACTIVE_CAUSAL_DOSES else np.full(len(trajectories), np.nan)
    )

    ablate_primary_name = f"ablate_primary_r{ACTIVE_PRIMARY_RANK:03d}"
    ablate_shuffled_name = f"ablate_shuffled_r{ACTIVE_PRIMARY_RANK:03d}"
    necessity = np.asarray([
        lookup(summary, value, ablate_primary_name, key="output_contrast_energy_reduction")
        for value in trajectories
    ])
    necessity_shuffled = np.asarray([
        lookup(summary, value, ablate_shuffled_name, key="output_contrast_energy_reduction")
        for value in trajectories
    ])
    necessity_random = []
    for trajectory_id in trajectories:
        values = [
            lookup(
                summary, trajectory_id,
                f"ablate_random_r{ACTIVE_PRIMARY_RANK:03d}_{draw:02d}",
                key="output_contrast_energy_reduction",
            )
            for draw in range(ACTIVE_CAUSAL_RANDOM_DRAWS)
        ]
        necessity_random.append(float(np.nanmedian(values)))
    necessity_random = np.asarray(necessity_random)
    necessity_gain_random = necessity - necessity_random
    necessity_gain_shuffled = necessity - necessity_shuffled
    necessity_sign = exact_positive_sign_test(necessity_gain_random)

    sufficiency_pass = bool(
        np.nanmean(full) >= MIN_FULL_SWAP_COEFFICIENT
        and np.nanmean(primary) >= MIN_PRIMARY_COEFFICIENT
        and np.nanmean(primary_cosine) >= MIN_PRIMARY_COSINE
        and np.nanmean(primary_shift) <= MAX_PRIMARY_MEAN_SHIFT_RATIO
        and np.nanmean(gain_random) >= MIN_PRIMARY_GAIN_OVER_RANDOM
        and np.nanmean(gain_shuffled) >= MIN_PRIMARY_GAIN_OVER_SHUFFLED
        and sufficiency_sign["positive"] >= min(REQUIRED_POSITIVE_EVALUATION_TRAJECTORIES, len(trajectories))
        and (sufficiency_sign["p_value"] <= 0.05 if RUN_MODE == "pilot" else True)
        and (bootstrap_interval(gain_random, trajectories)[0] > 0 if RUN_MODE == "pilot" else True)
        and (
            RUN_MODE == "smoke"
            or (
                np.sum(np.asarray(dose_slopes) > 0) >= REQUIRED_POSITIVE_EVALUATION_TRAJECTORIES
                and np.nanmean(negative) < 0
            )
        )
    )
    necessity_pass = bool(
        np.nanmean(necessity) >= MIN_NECESSITY_REDUCTION
        and np.nanmean(necessity_gain_random) >= MIN_NECESSITY_GAIN_OVER_RANDOM
        and np.nanmean(necessity_gain_shuffled) >= MIN_NECESSITY_GAIN_OVER_SHUFFLED
        and necessity_sign["positive"] >= min(REQUIRED_POSITIVE_NECESSITY_TRAJECTORIES, len(trajectories))
        and (necessity_sign["p_value"] <= 0.05 if RUN_MODE == "pilot" else True)
        and (bootstrap_interval(necessity_gain_random, trajectories, 97)[0] > 0 if RUN_MODE == "pilot" else True)
    )
    return {
        "primary_condition": primary_name,
        "trajectories": len(trajectories),
        "mean_primary_coefficient": float(np.nanmean(primary)),
        "mean_primary_cosine": float(np.nanmean(primary_cosine)),
        "mean_primary_mean_shift_ratio": float(np.nanmean(primary_shift)),
        "mean_full_swap_coefficient": float(np.nanmean(full)),
        "mean_random_coefficient": float(np.nanmean(random_values)),
        "mean_shuffled_coefficient": float(np.nanmean(shuffled)),
        "mean_gain_over_random": float(np.nanmean(gain_random)),
        "mean_gain_over_shuffled": float(np.nanmean(gain_shuffled)),
        "gain_over_random_ci95": bootstrap_interval(gain_random, trajectories),
        "gain_over_random_sign_test": sufficiency_sign,
        "positive_dose_slope_trajectories": int(np.sum(np.asarray(dose_slopes) > 0)),
        "negative_dose_mean": float(np.nanmean(negative)) if np.any(np.isfinite(negative)) else None,
        "mean_necessity_reduction": float(np.nanmean(necessity)),
        "mean_necessity_random_reduction": float(np.nanmean(necessity_random)),
        "mean_necessity_shuffled_reduction": float(np.nanmean(necessity_shuffled)),
        "mean_necessity_gain_over_random": float(np.nanmean(necessity_gain_random)),
        "mean_necessity_gain_over_shuffled": float(np.nanmean(necessity_gain_shuffled)),
        "necessity_gain_over_random_ci95": bootstrap_interval(necessity_gain_random, trajectories, 97),
        "necessity_gain_over_random_sign_test": necessity_sign,
        "sufficiency_gate_pass": sufficiency_pass,
        "necessity_gate_pass": necessity_pass,
        "bidirectional_gate_pass": bool(sufficiency_pass and necessity_pass),
    }


def fresh_run_certificate():
    expected = {
        "truth_generated": len(CONSTRUCTION_POOL_SPECS) + len(EVALUATION_POOL_SPECS),
        "baseline_generated": len(CONSTRUCTION_RECORDS) + len(EVALUATION_RECORDS),
        "intervention_generated": len(EVALUATION_RECORDS),
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


def make_plots(summary, rank_rows):
    figure, axes = plt.subplots(1, 4, figsize=(19, 4.5))
    geometry = json.loads((ANALYSIS_DIR / "fixed_block_construction_gate.json").read_text())
    axes[0].bar(["observed", "shuffled"], [
        geometry["mean_cka"], geometry["mean_cka"] - geometry["mean_shuffle_advantage"]
    ])
    axes[0].set(ylabel="linear CKA", title="Fixed block 4 geometry")
    primary_name = f"primary_r{ACTIVE_PRIMARY_RANK:03d}"
    doses = sorted({row["dose"] for row in summary if row["condition"] == primary_name})
    axes[1].plot(doses, [
        np.mean([row["output_coefficient"] for row in summary if row["condition"] == primary_name and np.isclose(row["dose"], dose)])
        for dose in doses
    ], marker="o")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set(xlabel="dose", ylabel="donor coefficient", title="Rank-64 sufficiency")
    ranks = [row["rank"] for row in rank_rows]
    axes[2].plot(ranks, [row["mean_learned_coefficient"] for row in rank_rows], marker="o", label="learned")
    axes[2].plot(ranks, [row["mean_random_coefficient"] for row in rank_rows], marker="o", label="random")
    axes[2].plot(ranks, [row["mean_shuffled_coefficient"] for row in rank_rows], marker="o", label="shuffled")
    axes[2].set(xlabel="rank", ylabel="donor coefficient", title="Rank scaling")
    axes[2].legend()
    conditions = [
        f"ablate_primary_r{ACTIVE_PRIMARY_RANK:03d}",
        f"ablate_shuffled_r{ACTIVE_PRIMARY_RANK:03d}",
    ]
    values = [
        np.mean([row["output_contrast_energy_reduction"] for row in summary if row["condition"] == condition])
        for condition in conditions
    ]
    random_values = [
        row["output_contrast_energy_reduction"] for row in summary
        if row["condition"].startswith(f"ablate_random_r{ACTIVE_PRIMARY_RANK:03d}_")
    ]
    axes[3].bar(["primary", "random", "shuffled"], [values[0], np.mean(random_values), values[1]])
    axes[3].set(ylabel="contrast-energy reduction", title="Rank-64 necessity")
    figure.tight_layout()
    figure.savefig(PLOT_DIR / "stage18_rank64_confirmation_summary.png", dpi=180)
    plt.close(figure)


if PIPELINE_FAILED:
    DECISION_PAYLOAD = {"status": "INCONCLUSIVE", "failure": FAILURE_MESSAGE}
elif RUN_MODE == "pilot" and not CONSTRUCTION_GATE_PASS:
    DECISION_PAYLOAD = {"status": "STOP_NO_FIXED_BLOCK_ACTION_GEOMETRY", "construction_gate": CONSTRUCTION_GATE}
elif not EVALUATION_OPENED:
    DECISION_PAYLOAD = {"status": "INCONCLUSIVE", "reason": "evaluation model activations were not opened"}
else:
    try:
        TRAJECTORY_SUMMARY = trajectory_summaries(INTERVENTION_ROWS)
        RANK_CURVE = rank_curve(TRAJECTORY_SUMMARY, ACTIVE_EVALUATION_TRAJECTORIES)
        CONFIRMATION_GATE = evaluate_confirmation_gate(TRAJECTORY_SUMMARY)
        FRESH_CERTIFICATE = fresh_run_certificate()
        if RUN_MODE == "smoke":
            candidate_status = "SMOKE_ONLY"
        elif CONFIRMATION_GATE["bidirectional_gate_pass"]:
            candidate_status = "CONFIRMED_BIDIRECTIONAL_RANK64_MEDIATOR"
        elif CONFIRMATION_GATE["sufficiency_gate_pass"]:
            candidate_status = "SUFFICIENCY_ONLY_RANK64_TRANSFER"
        elif CONFIRMATION_GATE["mean_full_swap_coefficient"] >= MIN_FULL_SWAP_COEFFICIENT:
            candidate_status = "FULL_SWAP_ONLY_NO_CONFIRMED_RANK64_MEDIATOR"
        else:
            candidate_status = "NO_ACTION_CONTRAST_CAUSAL_SIGNAL"
        source_eligible = bool(SOURCE_IDENTITY.get("confirmation_eligible", False))
        confirmation_eligible = bool(source_eligible and FRESH_CERTIFICATE["passed"])
        status = candidate_status if RUN_MODE == "smoke" or confirmation_eligible else "UNBOUND_OR_NONFRESH_EXPLORATORY_RESULT"
        DECISION_PAYLOAD = {
            "status": status,
            "candidate_status": candidate_status,
            "source_bound_claim_eligible": source_eligible,
            "fresh_run_claim_eligible": FRESH_CERTIFICATE["passed"],
            "confirmation_eligible": confirmation_eligible,
            "construction_gate": CONSTRUCTION_GATE,
            "confirmation_gate": CONFIRMATION_GATE,
            "rank_curve": RANK_CURVE,
            "claim_boundary": {
                "rank64_is_intrinsic_dimension": False,
                "coordinate_chart_authorized": False,
                "jacobian_claim_authorized": False,
                "full_swap_alone_is_mechanistic_evidence": False,
                "planning_mediation_requires_secondary_planning_results": True,
                "causal_claim_is_candidate_action_predicted_consequence_mediation": True,
            },
        }
        write_json(OUT / "stage18_decision.json", DECISION_PAYLOAD)
        make_plots(TRAJECTORY_SUMMARY, RANK_CURVE)
        print(json.dumps(DECISION_PAYLOAD, indent=2))
    except Exception:
        record_failure("decision_and_plots")
        DECISION_PAYLOAD = {"status": "INCONCLUSIVE", "failure": FAILURE_MESSAGE}

if not (OUT / "stage18_decision.json").exists():
    write_json(OUT / "stage18_decision.json", DECISION_PAYLOAD)
'''


packaging = r'''# Package compact audit evidence, including hashes for excluded raw shards.
write_json(OUT / "timings.json", TIMINGS)
memory_report("final")
if not PIPELINE_FAILED:
    (OUT / "FAILURE_TRACE.txt").write_text("NONE\n")

raw_roots = [TRUTH_DIR, BASELINE_DIR, INTERVENTION_DIR]
raw_files = [
    path for root in raw_roots for path in sorted(root.rglob("*")) if path.is_file()
]
raw_files += [
    path for path in sorted(SUBSPACE_DIR.rglob("*.npz")) if path.is_file()
]
raw_manifest = [
    {"path": str(path.relative_to(OUT)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    for path in raw_files
]
write_json(OUT / "raw_shard_manifest.json", raw_manifest)

excluded_roots = {ASSET_DIR, TRUTH_DIR, BASELINE_DIR, INTERVENTION_DIR}
compact_files = []
for path in sorted(OUT.rglob("*")):
    if not path.is_file():
        continue
    if any(root == path or root in path.parents for root in excluded_roots):
        continue
    if SUBSPACE_DIR in path.parents and path.suffix == ".npz":
        continue
    if path.name.startswith("stage18_rank64_result_bundle_"):
        continue
    compact_files.append(path)

manifest = [
    {"path": str(path.relative_to(OUT)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    for path in compact_files
]
write_json(OUT / "result_zip_manifest.json", manifest)
compact_files.append(OUT / "result_zip_manifest.json")

staging = OUT / "_result_staging"
if staging.exists():
    shutil.rmtree(staging)
staging.mkdir()
for path in compact_files:
    relative = path.relative_to(OUT)
    destination = staging / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)

archive_base = OUT / f"stage18_rank64_result_bundle_{RUN_SIGNATURE[:12]}"
archive = Path(shutil.make_archive(str(archive_base), "zip", staging))
shutil.rmtree(staging)
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


protocol_sources = [
    introduction,
    configuration,
    installation,
    setup,
    analysis_helpers,
    model_helpers,
    design,
    truth_generation,
    model_and_construction,
    construction_geometry,
    subspace_fit,
    evaluation_open,
    causal_interchange,
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
    code(model_and_construction),
    code(construction_geometry),
    code(subspace_fit),
    code(evaluation_open),
    code(causal_interchange),
    code(decision_and_plots),
    code(packaging),
]
for index, cell in enumerate(cells):
    cell["id"] = f"stage18-{index:02d}"

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
