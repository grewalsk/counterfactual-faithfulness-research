import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "17_finite_action_contrast_interchange.ipynb"
BASE14 = json.loads((ROOT / "14_predictive_control_j_bundle_pilot.ipynb").read_text())
BASE15 = json.loads((ROOT / "15_longitudinal_predictive_control_bundle.ipynb").read_text())
NUMERICAL = ROOT.parent / "src/cf_faithfulness/stage17_action_contrast.py"
STAGE15_NUMERICAL = ROOT.parent / "src/cf_faithfulness/stage15_bundle.py"


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


def assigned_uppercase_names(source):
    tree = ast.parse(source)
    names = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    names.append(target.id)
    return list(dict.fromkeys(names))


base14_analysis = "".join(BASE14["cells"][4]["source"])
base14_model = "".join(BASE14["cells"][5]["source"])
base15_model = "".join(BASE15["cells"][5]["source"])


introduction = r'''# Stage 17: finite action-contrast causal interchange

This is a **forward-pass-only** mechanistic pilot.  It tests the strongest
surviving causal hypothesis from Stage 4 without using coordinate moments,
physical-coordinate Jacobians, VJPs, JVPs, or an unconstrained nonlinear
probe.

For the candidate-action activation matrix at predictor block (l),

\[
H_l(s)=\begin{bmatrix}h_l(s,a_1)^\top\\\cdots\\h_l(s,a_K)^\top\end{bmatrix},
\qquad
C=I-\frac1K\mathbf 1\mathbf 1^\top,
\]

the object of interest is the action quotient (CH_l): information that
distinguishes candidate actions after their shared state component is removed.
Construction trajectories select one block and fit a proper reduced-rank
subspace from finite hidden-action contrasts to a frozen random sketch of the
model's predicted future-token contrasts.  Evaluation trajectories are opened
only after the block, ridge penalty, channel metric, subspaces, permutations,
and controls are frozen.

The primary intervention is

\[
H_l'=H_l+\rho(\Pi-I)CH_lP_l,
\]

where (P_l=U_lU_l^\top) is a rank-32 construction-fitted projection.  The
primary endpoint is donor-specific transfer in an **independent output sketch**
that was not used to fit (P_l).  A complete activation swap is retained only
as an on-manifold positive control because at dose one it is algebraically just
(H_l'=\Pi H_l).

Matched controls include a shuffled-correspondence fitted subspace, four
equal-rank empirical-span random subspaces, an exactly energy-matched common
mode, a wrong-state donor, negative dose, and the complete activation swap.
Frozen Stage 3/12 physical decoders provide secondary pose, action-ranking,
and regret outcomes; they never select the layer or subspace.

Run from a fresh GPU runtime.  Smoke mode validates execution only and cannot
authorize a scientific claim.  The full pilot is source-bound through Colab
secrets documented in the run guide.  Return the automatically downloaded
`stage17_action_contrast_result_bundle_<signature>.zip`.
'''


configuration = r'''# SINGLE CONFIGURATION BLOCK
# Leave this generated source unchanged.  For a source-bound full run, create
# Colab secrets STAGE17_RUN_MODE=pilot and STAGE17_SOURCE_COMMIT=<full 40 hex>.
RUN_MODE = "smoke"
EXPERIMENT_SOURCE_REF = ""
try:
    from google.colab import userdata as _colab_userdata

    RUN_MODE = _colab_userdata.get("STAGE17_RUN_MODE") or RUN_MODE
    EXPERIMENT_SOURCE_REF = (
        _colab_userdata.get("STAGE17_SOURCE_COMMIT") or EXPERIMENT_SOURCE_REF
    )
except Exception:
    pass

MOUNT_DRIVE = True
DOWNLOAD_RESULTS = True
CONTINUE_AFTER_BENCHMARK = True
MAX_ESTIMATED_TOTAL_MINUTES = 180.0

OUTPUT_DIR = "/content/counterfactual_faithfulness_stage17_action_contrast"
DRIVE_OUTPUT_DIR = (
    "/content/drive/MyDrive/counterfactual_faithfulness_stage17_action_contrast"
)

PROTOCOL_ID = "stage17-finite-action-contrast-interchange-v1"
NOTEBOOK_PROTOCOL_SHA256 = "__PROTOCOL_DIGEST__"
EVIDENCE_STATUS = "EXPLORATORY_UNTIL_SOURCE_BOUND_BEFORE_EVALUATION"
EXPERIMENT_REPOSITORY = "grewalsk/counterfactual-faithfulness-research"
EXPERIMENT_NOTEBOOK_PATH = "notebooks/17_finite_action_contrast_interchange.ipynb"
EXPERIMENT_BUILDER_PATH = "notebooks/build_stage17_action_contrast_notebook.py"
EXPERIMENT_NUMERICAL_PATH = "src/cf_faithfulness/stage17_action_contrast.py"

SEED = 17101
DESIGN_SEED = 17137
MODEL_NAME = "jepa_wm_pusht"
ENVIRONMENT = "PushT"
FRAMESKIP = 5
HORIZONS = [3]
TARGET_STEPS = HORIZONS
PRIMARY_HORIZON = 3
PREDICTOR_BLOCKS = [0, 1, 2, 3, 4, 5]
EXPECTED_CARRIER_CHANNELS = 400

TOTAL_TRAJECTORIES = 16
CONSTRUCTION_TRAJECTORIES = [200, 202, 204, 206, 208, 210, 212, 214]
EVALUATION_TRAJECTORIES = [201, 203, 205, 207, 209, 211, 213, 215]
STATES_PER_TRAJECTORY = 3
LONGITUDINAL_SAVE_STEPS = [0, 10, 20]
TASK_ID_OFFSET = 500

ACTION_PROFILES = 3
ACTION_BASIS_DIM = 6
ACTION_TANGENT_NORM = 0.35
ACTIONS_PER_STATE = 1 + 2 * ACTION_BASIS_DIM

OUTPUT_SKETCH_DIM = 128
TRAIN_OUTPUT_SKETCH_SEED = 17161
EVAL_OUTPUT_SKETCH_SEED = 17183
CHANNEL_SHRINKAGE = 0.10
CHANNEL_EIGEN_FLOOR = 1e-6
RIDGE_MULTIPLIERS = [1e-6, 1e-4, 1e-2, 1.0, 100.0]
PRIMARY_RANK = 32
SENSITIVITY_RANKS = [4, 8, 16, 32, 64]
MAX_SUBSPACE_RANK = 64
NULL_ROOT_SEED = 17231
PERMUTATION_SEED = 17251
BOOTSTRAP_SEED = 17269
CONSTRUCTION_SHUFFLE_DRAWS = 64
CAUSAL_RANDOM_DRAWS = 4
CAUSAL_DOSES = [-0.5, 0.25, 0.5, 1.0]
BOOTSTRAP_DRAWS = 5000

# Frozen pilot gates.  Smoke mode can never produce a scientific pass.
MIN_CONSTRUCTION_CKA = 0.15
MIN_CONSTRUCTION_CKA_ADVANTAGE = 0.03
REQUIRED_POSITIVE_CONSTRUCTION_TRAJECTORIES = 6
MIN_FULL_SWAP_COEFFICIENT = 0.35
MIN_PRIMARY_COEFFICIENT = 0.15
MIN_PRIMARY_COSINE = 0.20
MIN_PRIMARY_GAIN_OVER_RANDOM = 0.08
MIN_PRIMARY_GAIN_OVER_SHUFFLED = 0.05
MAX_PRIMARY_MEAN_SHIFT_RATIO = 0.25
REQUIRED_POSITIVE_EVALUATION_TRAJECTORIES = 7
MAX_ZERO_EDIT_ERROR = 1e-6

if RUN_MODE == "smoke":
    ACTIVE_CONSTRUCTION_TRAJECTORIES = CONSTRUCTION_TRAJECTORIES[:2]
    ACTIVE_EVALUATION_TRAJECTORIES = EVALUATION_TRAJECTORIES[:2]
    ACTIVE_TIME_INDICES = [0]
    ACTIVE_BLOCKS = [2, 3]
    ACTIVE_PRIMARY_RANK = 4
    ACTIVE_SENSITIVITY_RANKS = [2, 4]
    ACTIVE_MAX_SUBSPACE_RANK = 4
    ACTIVE_CONSTRUCTION_SHUFFLE_DRAWS = 4
    ACTIVE_CAUSAL_RANDOM_DRAWS = 1
    ACTIVE_CAUSAL_DOSES = [1.0]
    ACTIVE_BOOTSTRAP_DRAWS = 64
elif RUN_MODE == "pilot":
    ACTIVE_CONSTRUCTION_TRAJECTORIES = CONSTRUCTION_TRAJECTORIES
    ACTIVE_EVALUATION_TRAJECTORIES = EVALUATION_TRAJECTORIES
    ACTIVE_TIME_INDICES = list(range(STATES_PER_TRAJECTORY))
    ACTIVE_BLOCKS = PREDICTOR_BLOCKS
    ACTIVE_PRIMARY_RANK = PRIMARY_RANK
    ACTIVE_SENSITIVITY_RANKS = SENSITIVITY_RANKS
    ACTIVE_MAX_SUBSPACE_RANK = MAX_SUBSPACE_RANK
    ACTIVE_CONSTRUCTION_SHUFFLE_DRAWS = CONSTRUCTION_SHUFFLE_DRAWS
    ACTIVE_CAUSAL_RANDOM_DRAWS = CAUSAL_RANDOM_DRAWS
    ACTIVE_CAUSAL_DOSES = CAUSAL_DOSES
    ACTIVE_BOOTSTRAP_DRAWS = BOOTSTRAP_DRAWS
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
ASSET_REPOSITORY = "grewalsk/counterfactual-faithfulness-research"
ASSET_COMMIT = "2326e74556f6f81db2560e4396f4cc52c16a28f4"
ASSET_SPECS = {
    "physical_decoders.pt": {
        "path": (
            "results/bundles/stage12_result_bundle/frozen_training_decoders/"
            "jepa_wm_pusht_f975a0a746e7_training_decoders.pt"
        ),
        "sha256": (
            "51b2dbb0a81df432a2db5b941de83717e9979e761d57365f47d93d2dd0c0c694"
        ),
    },
}

assert PRIMARY_HORIZON in HORIZONS
assert ACTIONS_PER_STATE == 13
assert not set(CONSTRUCTION_TRAJECTORIES) & set(EVALUATION_TRAJECTORIES)
assert min(CONSTRUCTION_TRAJECTORIES + EVALUATION_TRAJECTORIES) >= 200
assert ACTIVE_PRIMARY_RANK <= ACTIVE_MAX_SUBSPACE_RANK
assert set(ACTIVE_SENSITIVITY_RANKS).issubset(set(range(1, ACTIVE_MAX_SUBSPACE_RANK + 1)))
'''

configuration_keys = assigned_uppercase_names(configuration)
configuration = configuration.rstrip() + "\n\nPROTOCOL_CONFIG_KEYS = " + repr(configuration_keys) + "\n"


installation = "".join(BASE14["cells"][2]["source"])


setup = "".join(BASE14["cells"][3]["source"])
setup = (
    setup.replace("Stage 14", "Stage 17")
    .replace("STAGE14", "STAGE17")
    .replace("stage14_pcj", "stage17_action_contrast")
)
old_directories = '''ASSET_DIR = OUT / "assets"
DESIGN_DIR = OUT / "design"
TRUTH_DIR = OUT / "truth"
TARGET_DIR = OUT / "target_tokens"
SCAN_DIR = OUT / "carrier_scan"
JACOBIAN_DIR = OUT / "write_read_shards"
ANALYSIS_DIR = OUT / "analysis"
CAUSAL_DIR = OUT / "causal"
EVIDENCE_DIR = OUT / "evaluation_evidence"
PLOT_DIR = OUT / "plots"
LOG_DIR = OUT / "logs"
for directory in [
    OUT, ASSET_DIR, DESIGN_DIR, TRUTH_DIR, TARGET_DIR, SCAN_DIR,
    JACOBIAN_DIR, ANALYSIS_DIR, CAUSAL_DIR, EVIDENCE_DIR, PLOT_DIR, LOG_DIR,
]:'''
new_directories = '''ASSET_DIR = OUT / "assets"
DESIGN_DIR = OUT / "design"
TRUTH_DIR = OUT / "truth"
BASELINE_DIR = OUT / "baseline_shards"
SUBSPACE_DIR = OUT / "subspaces"
ANALYSIS_DIR = OUT / "analysis"
INTERVENTION_DIR = OUT / "intervention_shards"
EVIDENCE_DIR = OUT / "evaluation_evidence"
PLOT_DIR = OUT / "plots"
LOG_DIR = OUT / "logs"
for directory in [
    OUT, ASSET_DIR, DESIGN_DIR, TRUTH_DIR, BASELINE_DIR, SUBSPACE_DIR,
    ANALYSIS_DIR, INTERVENTION_DIR, EVIDENCE_DIR, PLOT_DIR, LOG_DIR,
]:'''
if old_directories not in setup:
    raise RuntimeError("Stage 14 setup directory template changed")
setup = setup.replace(old_directories, new_directories)
old_sources = '''for label, relative in [
        ("notebook", EXPERIMENT_NOTEBOOK_PATH),
        ("builder", EXPERIMENT_BUILDER_PATH),
    ]:'''
new_sources = '''for label, relative in [
        ("notebook", EXPERIMENT_NOTEBOOK_PATH),
        ("builder", EXPERIMENT_BUILDER_PATH),
        ("numerical", EXPERIMENT_NUMERICAL_PATH),
    ]:'''
if old_sources not in setup:
    raise RuntimeError("Stage 14 source identity template changed")
setup = setup.replace(old_sources, new_sources)


analysis_helpers = function_sources(
    base14_analysis,
    [
        "array_sha256",
        "channel_metric_from_moments",
        "transform_primal_channels",
        "inverse_transform_primal_channels",
        "CountSketchProjector",
        "earliest_within_one_se",
        "stable_cosine",
        "manifest_rows",
    ],
)
analysis_helpers += "\n\n\n" + function_sources(
    STAGE15_NUMERICAL.read_text(), ["temporal_action_basis"]
)
analysis_helpers += "\n\n\n" + function_sources(
    NUMERICAL.read_text(),
    [
        "candidate_center",
        "stable_seed",
        "fixed_derangement",
        "linear_cka",
        "action_swap_delta",
        "matched_common_mode",
        "donor_transfer_metrics",
        "grouped_kernel_ridge_cv",
        "fit_dual_ridge_basis",
        "random_subspace_in_span",
        "pair_indices",
        "ranking_metrics",
        "pose_target",
        "decoded_task_cost",
        "exact_positive_sign_test",
    ],
)
analysis_helpers += r'''


def norm_match(candidate, reference):
    value = np.asarray(candidate, dtype=np.float64)
    target = np.asarray(reference, dtype=np.float64)
    norm = np.linalg.norm(value)
    if norm <= 1e-12:
        raise RuntimeError("cannot norm-match a zero control")
    return value * (np.linalg.norm(target) / norm)


def clustered_bootstrap_mean(values, groups, draws, seed):
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    by_group = {group: values[groups == group] for group in unique}
    rng = np.random.default_rng(int(seed))
    results = np.empty(int(draws), dtype=np.float64)
    for index in range(int(draws)):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        results[index] = np.mean([np.mean(by_group[group]) for group in sampled])
    return results


# Algebraic identities execute on CPU before any simulator or model access.
_rng = np.random.default_rng(1701)
_toy = _rng.normal(size=(7, 3, 5))
_permutation = fixed_derangement(len(_toy), 1703)
_full = _toy + action_swap_delta(_toy, _permutation)
if not np.allclose(_full, _toy[_permutation], atol=1e-12):
    raise AssertionError("full finite action swap identity failed")
_metrics = donor_transfer_metrics(_toy, _toy[_permutation], _permutation)
if not np.isclose(_metrics["coefficient"], 1.0, atol=1e-12):
    raise AssertionError("donor transfer coefficient identity failed")
'''


model_helpers = function_sources(
    base15_model,
    [
        "to_model_observation",
        "configure_repo",
        "make_environment",
        "verify_pretrained_assets",
        "validate_jepa_predictor",
        "load_frozen_model",
        "model_action_tensor",
        "layer_tokens_full",
        "forward_with_carriers",
    ],
)
model_helpers = model_helpers.replace("stage15-jepa-wms", "stage17-jepa-wms")
model_helpers += "\n\n\n" + function_sources(base14_model, ["physical_pose_decoder"])


design = r'''# Freeze new trajectory paths, tasks, action branches, donors, and null seeds.


def trajectory_specs():
    specs = []
    center = np.asarray([256.0, 256.0])
    trajectory_ids = CONSTRUCTION_TRAJECTORIES + EVALUATION_TRAJECTORIES
    for design_index, trajectory_id in enumerate(sorted(trajectory_ids)):
        block_angle = 0.29 + 2.0 * np.pi * design_index / TOTAL_TRAJECTORIES
        block = center + 60.0 * np.asarray([np.cos(block_angle), np.sin(block_angle)])
        agent_index = (7 * design_index + 3) % TOTAL_TRAJECTORIES
        agent_angle = 0.53 + 2.0 * np.pi * agent_index / TOTAL_TRAJECTORIES
        agent = center + 132.0 * np.asarray([np.cos(agent_angle), np.sin(agent_angle)])
        direction = block - agent
        direction /= max(np.linalg.norm(direction), 1e-12)
        perpendicular = np.asarray([-direction[1], direction[0]])
        controls = []
        for step in range(1, LONGITUDINAL_SAVE_STEPS[-1] + 1):
            fraction = step / LONGITUDINAL_SAVE_STEPS[-1]
            curve = (
                (1.0 if design_index % 2 == 0 else -1.0)
                * 0.18
                * np.sin(2.0 * np.pi * fraction)
                * perpendicular
            )
            action = 0.085 * (direction + curve)
            if np.linalg.norm(action) > 0.14:
                raise RuntimeError("longitudinal relative action exceeds safe magnitude")
            controls.append(action)
        goal_index = (5 * design_index + 1) % TOTAL_TRAJECTORIES
        goal_angle = 0.71 + 2.0 * np.pi * goal_index / TOTAL_TRAJECTORIES
        goal_xy = center + 62.0 * np.asarray([np.cos(goal_angle), np.sin(goal_angle)])
        split = "construction" if trajectory_id in CONSTRUCTION_TRAJECTORIES else "evaluation"
        specs.append(
            {
                "design_index": design_index,
                "trajectory_id": trajectory_id,
                "task_id": TASK_ID_OFFSET + design_index,
                "split": split,
                "seed": DESIGN_SEED + 1009 * design_index,
                "goal": np.asarray(
                    [goal_xy[0], goal_xy[1], ((1.4 * goal_angle + np.pi) % (2 * np.pi)) - np.pi],
                    dtype=np.float64,
                ),
                "initial_state": np.asarray(
                    [
                        agent[0], agent[1], block[0], block[1],
                        ((1.1 * block_angle + np.pi) % (2 * np.pi)) - np.pi,
                        0.0, 0.0, 0.0, 0.0, 0.0,
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
if {row["trajectory_id"] for row in ALL_TRAJECTORY_SPECS} & set(range(124)):
    raise AssertionError("Stage 17 trajectory IDs overlap inspected Stages 15/16 IDs")
if {row["task_id"] for row in CONSTRUCTION_SPECS} & {row["task_id"] for row in EVALUATION_SPECS}:
    raise AssertionError("construction and evaluation tasks overlap")

RAW_ACTION_BASIS = temporal_action_basis(max(HORIZONS) * FRAMESKIP, ACTION_PROFILES)


def candidate_action_bank(state):
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (10,):
        raise ValueError("candidate state must be a ten-dimensional dynamic PushT state")
    baseline = np.zeros((max(HORIZONS) * FRAMESKIP, 2), dtype=np.float64)
    branches = [baseline]
    for column in range(ACTION_BASIS_DIM):
        delta = ACTION_TANGENT_NORM * RAW_ACTION_BASIS[:, column].reshape(-1, 2)
        branches.extend([baseline + delta, baseline - delta])
    actions = np.stack(branches)
    if actions.shape != (ACTIONS_PER_STATE, max(HORIZONS) * FRAMESKIP, 2):
        raise RuntimeError(f"bad candidate action bank shape {actions.shape}")
    if np.max(np.abs(actions)) > 0.14:
        raise RuntimeError("candidate action bank exceeds safe relative-action magnitude")
    return actions.astype(np.float32)


NULL_RNG = np.random.default_rng(NULL_ROOT_SEED)
NULL_SEEDS = NULL_RNG.integers(
    0,
    np.iinfo(np.uint32).max,
    size=(TOTAL_TRAJECTORIES, STATES_PER_TRAJECTORY, CAUSAL_RANDOM_DRAWS + 4),
    dtype=np.uint32,
)
donor_permutations = {}
wrong_state_map = {}
for split_specs in [CONSTRUCTION_SPECS, EVALUATION_SPECS]:
    by_time = {}
    for spec in split_specs:
        for time_index in ACTIVE_TIME_INDICES:
            record_id = int(spec["trajectory_id"]) * STATES_PER_TRAJECTORY + time_index
            donor_permutations[str(record_id)] = fixed_derangement(
                ACTIONS_PER_STATE, stable_seed(PERMUTATION_SEED, record_id, "donor")
            ).tolist()
            by_time.setdefault(time_index, []).append(record_id)
    for time_index, identifiers in by_time.items():
        identifiers = sorted(identifiers)
        for index, record_id in enumerate(identifiers):
            wrong_state_map[str(record_id)] = identifiers[(index + 1) % len(identifiers)]

np.savez_compressed(
    DESIGN_DIR / "stage17_design.npz",
    trajectory_ids=np.asarray([row["trajectory_id"] for row in ALL_TRAJECTORY_SPECS]),
    splits=np.asarray([row["split"] for row in ALL_TRAJECTORY_SPECS]),
    initial_states=np.stack([row["initial_state"] for row in ALL_TRAJECTORY_SPECS]),
    goals=np.stack([row["goal"] for row in ALL_TRAJECTORY_SPECS]),
    controls=np.stack([row["controls"] for row in ALL_TRAJECTORY_SPECS]),
    save_steps=np.asarray(LONGITUDINAL_SAVE_STEPS),
    null_seeds=NULL_SEEDS,
)
DESIGN_MANIFEST = {
    "specs": [
        {
            **{
                key: value
                for key, value in row.items()
                if key not in {"initial_state", "controls", "goal"}
            },
            "initial_state": row["initial_state"].tolist(),
            "controls": row["controls"].tolist(),
            "goal": row["goal"].tolist(),
        }
        for row in ALL_TRAJECTORY_SPECS
    ],
    "active_construction_trajectories": ACTIVE_CONSTRUCTION_TRAJECTORIES,
    "active_evaluation_trajectories": ACTIVE_EVALUATION_TRAJECTORIES,
    "active_time_indices": ACTIVE_TIME_INDICES,
    "donor_permutations": donor_permutations,
    "wrong_state_map": wrong_state_map,
}
write_json(DESIGN_DIR / "trajectory_design_manifest.json", DESIGN_MANIFEST)
DESIGN_FREEZE = {
    "created_before_simulator_or_model_data": True,
    "protocol_id": PROTOCOL_ID,
    "run_signature": RUN_SIGNATURE,
    "source_identity": SOURCE_IDENTITY,
    "design_sha256": sha256_file(DESIGN_DIR / "stage17_design.npz"),
    "manifest_sha256": sha256_file(DESIGN_DIR / "trajectory_design_manifest.json"),
    "coordinate_reader_used_for_selection": False,
    "jacobian_used": False,
    "evaluation_trajectories_opened": [],
}
freeze_path = DESIGN_DIR / "design_freeze.json"
if freeze_path.exists() and json.loads(freeze_path.read_text()) != DESIGN_FREEZE:
    raise RuntimeError("existing Stage 17 design freeze differs")
write_json(freeze_path, DESIGN_FREEZE)
print(json.dumps(DESIGN_FREEZE, indent=2))
'''


truth_generation = r'''# Realize construction trajectories and exact finite action branches.


def record_task(record_or_spec):
    return {
        "environment": ENVIRONMENT,
        "task_id": int(record_or_spec["task_id"]),
        "goal": np.asarray(record_or_spec["goal"], dtype=np.float64).tolist(),
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
        "proprio": np.asarray(
            [*environment.agent.position, *environment.agent.velocity], dtype=np.float32
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
    finally:
        environment.close()
    if set(observations) != set(TARGET_STEPS):
        raise RuntimeError("dynamic rollout missed the primary horizon")
    return initial, observations, states, counts


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


def trajectory_path(spec):
    return TRUTH_DIR / f"trajectory_{int(spec['trajectory_id']):03d}.npz"


def realize_trajectory(spec):
    destination = trajectory_path(spec)
    if not destination.exists():
        environment, _ = reset_dynamic_environment(
            spec["initial_state"], record_task(spec), int(spec["seed"])
        )
        states = [dynamic_state_from_environment(environment)]
        contacts = [0]
        cumulative = 0
        try:
            for step, action in enumerate(spec["controls"], start=1):
                _, _, _, info = environment.step(action)
                cumulative += int(info.get("n_contacts", 0))
                if step in LONGITUDINAL_SAVE_STEPS[1:]:
                    states.append(dynamic_state_from_environment(environment))
                    contacts.append(cumulative)
        finally:
            environment.close()
        if len(states) != STATES_PER_TRAJECTORY:
            raise RuntimeError("trajectory missed a frozen save point")
        atomic_npz(
            destination,
            trajectory_id=np.asarray(spec["trajectory_id"], dtype=np.int64),
            split=np.asarray(spec["split"]),
            states=np.stack(states).astype(np.float64),
            goal=np.asarray(spec["goal"], dtype=np.float64),
            controls=np.asarray(spec["controls"], dtype=np.float64),
            save_steps=np.asarray(LONGITUDINAL_SAVE_STEPS, dtype=np.int64),
            cumulative_contacts=np.asarray(contacts, dtype=np.int64),
        )
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
                "goal": payload["goal"].astype(np.float64),
                "path_contacts": int(payload["cumulative_contacts"][index]),
            }
            for index in ACTIVE_TIME_INDICES
        ]


def realize_records(specs):
    records = []
    for spec in specs:
        records.extend(realize_trajectory(spec))
    return records


def branch_path(record_id):
    return TRUTH_DIR / f"state_{int(record_id):04d}.npz"


def generate_truth(records):
    started = time.perf_counter()
    for record_index, record in enumerate(records):
        destination = branch_path(record["record_id"])
        if destination.exists():
            continue
        action_bank = candidate_action_bank(record["state"])
        initials = []
        initial_proprios = []
        endpoint_states = []
        interaction_counts = []
        for action in action_bank:
            initial, _, states, counts = rollout_dynamic_branch(record, action)
            initials.append(initial["visual"])
            initial_proprios.append(initial["proprio"])
            endpoint_states.append(states[PRIMARY_HORIZON])
            interaction_counts.append(counts[PRIMARY_HORIZON])
        if not all(np.array_equal(initials[0], value) for value in initials[1:]):
            raise AssertionError("initial visual drift across candidate branches")
        if not all(np.array_equal(initial_proprios[0], value) for value in initial_proprios[1:]):
            raise AssertionError("initial proprio drift across candidate branches")
        atomic_npz(
            destination,
            record_id=np.asarray(record["record_id"], dtype=np.int64),
            trajectory_id=np.asarray(record["trajectory_id"], dtype=np.int64),
            time_index=np.asarray(record["time_index"], dtype=np.int64),
            task_id=np.asarray(record["task_id"], dtype=np.int64),
            split=np.asarray(record["split"]),
            state=np.asarray(record["state"], dtype=np.float64),
            goal=np.asarray(record["goal"], dtype=np.float64),
            initial_visual=np.asarray(initials[0], dtype=np.uint8),
            initial_proprio=np.asarray(initial_proprios[0], dtype=np.float32),
            selected_actions=action_bank.astype(np.float32),
            endpoint_states=np.asarray(endpoint_states, dtype=np.float32),
            interaction_counts=np.asarray(interaction_counts, dtype=np.int32),
        )
        write_json(
            OUT / "truth_progress.json",
            {"completed": record_index + 1, "total": len(records), "last_record_id": int(record["record_id"])},
        )
    TIMINGS[f"truth_{records[0]['split']}_seconds"] = time.perf_counter() - started


if not PIPELINE_FAILED:
    try:
        REPO = configure_repo()
        CONSTRUCTION_RECORDS = realize_records(CONSTRUCTION_SPECS)
        RESTORE_TEST = exact_dynamic_restore_test(CONSTRUCTION_RECORDS[0])
        write_json(OUT / "restore_test.json", RESTORE_TEST)
        generate_truth(CONSTRUCTION_RECORDS)
        memory_report("construction_truth_complete")
    except Exception:
        record_failure("construction_truth_generation")
'''


model_and_construction = r'''# Load frozen JEPA-WM and cache construction forward passes at all blocks.


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


def baseline_path(record_id):
    return BASELINE_DIR / f"state_{int(record_id):04d}.npz"


def load_baseline(record_id):
    with np.load(baseline_path(record_id)) as payload:
        return {name: payload[name].copy() for name in payload.files}


def extract_baselines(records, blocks):
    started = time.perf_counter()
    for index, record in enumerate(records):
        destination = baseline_path(record["record_id"])
        if destination.exists():
            with np.load(destination) as existing:
                saved_blocks = existing["blocks"].astype(int).tolist()
            if saved_blocks == list(blocks):
                continue
            destination.unlink()
        initial, actions = state_model_inputs(record["record_id"])
        with torch.inference_mode():
            predicted, predicted_proprio, captures = forward_with_carriers(
                initial,
                actions,
                PRIMARY_HORIZON,
                capture_blocks=blocks,
            )
            train_sketch = TRAIN_OUTPUT_PROJECTOR(predicted).cpu().numpy()
            eval_sketch = EVAL_OUTPUT_PROJECTOR(predicted).cpu().numpy()
            decoded_pose = DECODE_PHYSICAL_POSE(predicted).cpu().numpy()
        carriers = np.stack(
            [
                layer_tokens_full(captures[block]).detach().float().cpu().numpy()
                for block in blocks
            ]
        )
        atomic_npz(
            destination,
            record_id=np.asarray(record["record_id"], dtype=np.int64),
            trajectory_id=np.asarray(record["trajectory_id"], dtype=np.int64),
            time_index=np.asarray(record["time_index"], dtype=np.int64),
            blocks=np.asarray(blocks, dtype=np.int64),
            # Float32 is deliberate: the full-swap positive control must use
            # the exact cached carrier rather than a quantized approximation.
            carriers=carriers.astype(np.float32),
            output_train_sketch=train_sketch.astype(np.float32),
            output_eval_sketch=eval_sketch.astype(np.float32),
            decoded_pose=decoded_pose.astype(np.float32),
            predicted_proprio=predicted_proprio.detach().float().cpu().numpy(),
        )
        write_json(
            OUT / f"baseline_{record['split']}_progress.json",
            {"completed": index + 1, "total": len(records), "last_record_id": int(record["record_id"])},
        )
        del initial, actions, predicted, predicted_proprio, captures, carriers
        gc.collect()
        torch.cuda.empty_cache()
    TIMINGS[f"baseline_{records[0]['split']}_seconds"] = time.perf_counter() - started


def carrier_for_block(payload, block):
    blocks = payload["blocks"].astype(int).tolist()
    if int(block) not in blocks:
        raise RuntimeError(f"block {block} is absent from baseline shard")
    return payload["carriers"][blocks.index(int(block))].astype(np.float32)


def hook_identity_test(record_id):
    initial, actions = state_model_inputs(record_id)
    block = int(ACTIVE_BLOCKS[0])
    with torch.inference_mode():
        baseline, _, _ = forward_with_carriers(
            initial, actions, PRIMARY_HORIZON, capture_blocks=[block]
        )
        patched, _, _ = forward_with_carriers(
            initial,
            actions,
            PRIMARY_HORIZON,
            capture_blocks=[block],
            intervention={
                "block": block,
                "delta": torch.zeros(
                    ACTIONS_PER_STATE,
                    256,
                    EXPECTED_CARRIER_CHANNELS,
                    device="cuda",
                    dtype=torch.float32,
                ),
            },
        )
    error = float(torch.max(torch.abs(patched - baseline)).cpu())
    result = {"record_id": int(record_id), "block": block, "max_abs_error": error, "passed": error <= MAX_ZERO_EDIT_ERROR}
    if not result["passed"]:
        raise RuntimeError(f"zero intervention changed the model output: {result}")
    write_json(OUT / "hook_identity_test.json", result)
    return result


def forward_benchmark(record_id):
    initial, actions = state_model_inputs(record_id)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        _, _, _ = forward_with_carriers(
            initial, actions, PRIMARY_HORIZON, capture_blocks=[int(ACTIVE_BLOCKS[0])]
        )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    interventions_per_record = len(ACTIVE_CAUSAL_DOSES) + 3 + ACTIVE_CAUSAL_RANDOM_DRAWS + len(ACTIVE_SENSITIVITY_RANKS)
    total_eval_records = len(ACTIVE_EVALUATION_TRAJECTORIES) * len(ACTIVE_TIME_INDICES)
    estimate = seconds * interventions_per_record * total_eval_records / 60.0
    result = {
        "seconds_per_candidate_batch": seconds,
        "intervention_forwards_per_record": interventions_per_record,
        "evaluation_records": total_eval_records,
        "estimated_intervention_minutes": estimate,
        "warning_threshold_minutes": MAX_ESTIMATED_TOTAL_MINUTES,
    }
    write_json(OUT / "forward_benchmark.json", result)
    if estimate > MAX_ESTIMATED_TOTAL_MINUTES and not CONTINUE_AFTER_BENCHMARK:
        raise RuntimeError(
            "measured estimate exceeds the configured credit guard; set "
            "CONTINUE_AFTER_BENCHMARK=True only after reviewing forward_benchmark.json"
        )
    return result


if not PIPELINE_FAILED:
    try:
        MODEL, PREPROCESSOR, PREDICTOR, PREDICTOR_BLOCK_MODULES = load_frozen_model()
        if len(PREDICTOR_BLOCK_MODULES) != len(PREDICTOR_BLOCKS):
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
        HOOK_IDENTITY = hook_identity_test(CONSTRUCTION_RECORDS[0]["record_id"])
        FORWARD_BENCHMARK = forward_benchmark(CONSTRUCTION_RECORDS[0]["record_id"])
        extract_baselines(CONSTRUCTION_RECORDS, ACTIVE_BLOCKS)
        memory_report("construction_baselines_complete")
    except Exception:
        record_failure("construction_model_baselines")
'''


construction_geometry = r'''# Coordinate-free construction screen and immutable layer selection.


def construction_geometry_rows():
    rows = []
    output_by_record = {
        int(record["record_id"]): load_baseline(record["record_id"])["output_train_sketch"].astype(np.float64)
        for record in CONSTRUCTION_RECORDS
    }
    for record in CONSTRUCTION_RECORDS:
        record_id = int(record["record_id"])
        payload = load_baseline(record_id)
        output = output_by_record[record_id]
        wrong_output = output_by_record[int(wrong_state_map[str(record_id)])]
        for block in ACTIVE_BLOCKS:
            carrier = carrier_for_block(payload, block)
            observed = linear_cka(carrier, output)
            shuffled = []
            for draw in range(ACTIVE_CONSTRUCTION_SHUFFLE_DRAWS):
                permutation = fixed_derangement(
                    ACTIONS_PER_STATE,
                    stable_seed(PERMUTATION_SEED, record_id, block, draw, "geometry"),
                )
                shuffled.append(linear_cka(carrier, output[permutation]))
            rows.append(
                {
                    "record_id": record_id,
                    "trajectory_id": int(record["trajectory_id"]),
                    "time_index": int(record["time_index"]),
                    "block": int(block),
                    "observed_cka": float(observed),
                    "shuffled_cka": float(np.mean(shuffled)),
                    "wrong_state_cka": float(linear_cka(carrier, wrong_output)),
                    "shuffle_advantage": float(observed - np.mean(shuffled)),
                    "wrong_state_advantage": float(observed - linear_cka(carrier, wrong_output)),
                }
            )
    write_csv(ANALYSIS_DIR / "construction_geometry_rows.csv", rows)
    return rows


def select_construction_layer(rows):
    trajectory_ids = sorted({row["trajectory_id"] for row in rows})
    blocks = sorted({row["block"] for row in rows})
    matrix = np.empty((len(trajectory_ids), len(blocks)), dtype=np.float64)
    summary = []
    for trajectory_index, trajectory_id in enumerate(trajectory_ids):
        for block_index, block in enumerate(blocks):
            selected = [
                row for row in rows
                if row["trajectory_id"] == trajectory_id and row["block"] == block
            ]
            matrix[trajectory_index, block_index] = np.mean([row["observed_cka"] for row in selected])
    chosen = earliest_within_one_se(matrix)
    selected_block = int(blocks[int(chosen["selected_index"])])
    for block_index, block in enumerate(blocks):
        selected = [row for row in rows if row["block"] == block]
        by_trajectory = {
            trajectory_id: np.mean(
                [row["shuffle_advantage"] for row in selected if row["trajectory_id"] == trajectory_id]
            )
            for trajectory_id in trajectory_ids
        }
        summary.append(
            {
                "block": int(block),
                "mean_cka": float(np.mean([row["observed_cka"] for row in selected])),
                "mean_shuffle_advantage": float(np.mean([row["shuffle_advantage"] for row in selected])),
                "mean_wrong_state_advantage": float(np.mean([row["wrong_state_advantage"] for row in selected])),
                "positive_trajectories": int(np.sum(np.asarray(list(by_trajectory.values())) > 0)),
                "selected": bool(block == selected_block),
            }
        )
    selected = next(row for row in summary if row["selected"])
    gate = bool(
        selected["mean_cka"] >= MIN_CONSTRUCTION_CKA
        and selected["mean_shuffle_advantage"] >= MIN_CONSTRUCTION_CKA_ADVANTAGE
        and selected["positive_trajectories"] >= min(
            REQUIRED_POSITIVE_CONSTRUCTION_TRAJECTORIES, len(trajectory_ids)
        )
    )
    payload = {
        "selected_block": selected_block,
        "selection_rule": "earliest block within one trajectory-level SE of best mean CKA",
        "means": chosen["means"].tolist(),
        "standard_errors": chosen["standard_errors"].tolist(),
        "threshold": float(chosen["threshold"]),
        "construction_gate_pass": gate,
        "summary": summary,
        "evaluation_trajectories_seen": [],
    }
    write_json(ANALYSIS_DIR / "construction_layer_selection.json", payload)
    write_csv(ANALYSIS_DIR / "construction_layer_summary.csv", summary)
    return payload


if not PIPELINE_FAILED:
    try:
        CONSTRUCTION_GEOMETRY_ROWS = construction_geometry_rows()
        LAYER_SELECTION = select_construction_layer(CONSTRUCTION_GEOMETRY_ROWS)
        SELECTED_BLOCK = int(LAYER_SELECTION["selected_block"])
        CONSTRUCTION_GATE_PASS = bool(LAYER_SELECTION["construction_gate_pass"])
        print(json.dumps(LAYER_SELECTION, indent=2))
    except Exception:
        record_failure("construction_geometry")
'''


subspace_fit = r'''# Fit proper action-contrast subspaces using construction trajectories only.


def construction_matrices():
    count = 0
    total = np.zeros(EXPECTED_CARRIER_CHANNELS, dtype=np.float64)
    cross = np.zeros((EXPECTED_CARRIER_CHANNELS, EXPECTED_CARRIER_CHANNELS), dtype=np.float64)
    native_residuals = []
    output_residuals = []
    trajectory_groups = []
    record_slices = []
    offset = 0
    for record in CONSTRUCTION_RECORDS:
        payload = load_baseline(record["record_id"])
        carrier = carrier_for_block(payload, SELECTED_BLOCK).astype(np.float64)
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
        count,
        total,
        cross,
        shrinkage=CHANNEL_SHRINKAGE,
        relative_floor=CHANNEL_EIGEN_FLOOR,
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
    alpha = torch.linalg.solve(
        gram + float(penalty) * torch.eye(len(gram), device="cuda"), y
    )
    weight = x.T @ alpha
    left, singular, _ = torch.linalg.svd(weight, full_matrices=False)
    keep = min(int(max_rank), left.shape[1], int(torch.sum(singular > 1e-7).item()))
    if keep < int(ACTIVE_PRIMARY_RANK):
        raise RuntimeError(f"ridge map rank {keep} is below primary rank {ACTIVE_PRIMARY_RANK}")
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
        len(x), int(rank) + 8, generator=generator, device="cuda", dtype=torch.float32
    )
    candidate = x.T @ coefficients
    candidate -= excluded_tensor @ (excluded_tensor.T @ candidate)
    basis, triangular = torch.linalg.qr(candidate, mode="reduced")
    diagonal = torch.abs(torch.diag(triangular))
    if int(torch.sum(diagonal > torch.max(diagonal) * 1e-6).item()) < int(rank):
        raise RuntimeError("empirical action span is too small for random control")
    result = basis[:, : int(rank)].detach().cpu().numpy().astype(np.float32)
    del x, excluded_tensor, coefficients, candidate, basis, triangular, diagonal
    torch.cuda.empty_cache()
    return result


def fit_and_freeze_subspaces():
    x, y, groups, record_slices, metric, output_scale = construction_matrices()
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
                ACTIVE_PRIMARY_RANK,
                stable_seed(NULL_ROOT_SEED, draw, "empirical_span"),
                primary_basis[:, :ACTIVE_PRIMARY_RANK],
            )
        )
    destination = SUBSPACE_DIR / "frozen_action_contrast_subspaces.npz"
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
        "selected_block": SELECTED_BLOCK,
        "primary_rank": ACTIVE_PRIMARY_RANK,
        "sensitivity_ranks": ACTIVE_SENSITIVITY_RANKS,
        "max_rank": ACTIVE_MAX_SUBSPACE_RANK,
        "selected_ridge_multiplier": ridge["selected_multiplier"],
        "ridge_penalty": ridge["penalty"],
        "channel_condition_number": metric["condition_number"],
        "primary_basis_shape": list(primary_basis.shape),
        "shuffled_basis_shape": list(shuffled_basis.shape),
        "random_draws": len(random_bases),
        "subspace_sha256": sha256_file(destination),
        "construction_trajectory_ids": sorted(set(groups.astype(int).tolist())),
        "evaluation_trajectories_seen": [],
        "full_activation_swap_is_positive_control_only": True,
        "jacobians_computed": False,
    }
    write_json(SUBSPACE_DIR / "subspace_manifest.json", manifest)
    freeze = {
        "frozen_before_evaluation": True,
        "source_identity": SOURCE_IDENTITY,
        "design_freeze_sha256": sha256_file(DESIGN_DIR / "design_freeze.json"),
        "layer_selection_sha256": sha256_file(ANALYSIS_DIR / "construction_layer_selection.json"),
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


evaluation_open = r'''# Open evaluation trajectories only after the layer and subspaces are frozen.
EVALUATION_OPENED = False
if not PIPELINE_FAILED:
    try:
        if RUN_MODE == "pilot" and not CONSTRUCTION_GATE_PASS:
            EVALUATION_OPENED = False
            write_json(
                OUT / "evaluation_open_certificate.json",
                {
                    "opened": False,
                    "reason": "STOP_NO_CONSTRUCTION_ACTION_GEOMETRY",
                    "construction_gate_pass": False,
                },
            )
        else:
            verify_executed_notebook_through(
                "# Open evaluation trajectories only after the layer and subspaces are frozen."
            )
            if not (SUBSPACE_DIR / "subspace_freeze.json").exists():
                raise RuntimeError("subspace freeze is absent")
            EVALUATION_RECORDS = realize_records(EVALUATION_SPECS)
            generate_truth(EVALUATION_RECORDS)
            extract_baselines(EVALUATION_RECORDS, [SELECTED_BLOCK])
            EVALUATION_OPENED = True
            write_json(
                OUT / "evaluation_open_certificate.json",
                {
                    "opened": True,
                    "source_identity": SOURCE_IDENTITY,
                    "subspace_freeze_sha256": sha256_file(SUBSPACE_DIR / "subspace_freeze.json"),
                    "evaluation_trajectory_ids": ACTIVE_EVALUATION_TRAJECTORIES,
                    "opened_after_freeze": True,
                },
            )
            memory_report("evaluation_baselines_complete")
    except Exception:
        record_failure("evaluation_open")
'''


causal_interchange = r'''# Run finite partial-residual interchanges and matched controls.


def load_frozen_subspaces():
    with np.load(SUBSPACE_DIR / "frozen_action_contrast_subspaces.npz") as payload:
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
    true_pose = pose_target(endpoints)
    return decoded_task_cost(true_pose, goal), goal


def intervention_path(record_id):
    return INTERVENTION_DIR / f"state_{int(record_id):04d}.json"


def finite_json_rows(rows):
    cleaned = []
    for row in rows:
        cleaned.append(
            {
                key: (
                    None
                    if isinstance(value, (float, np.floating)) and not np.isfinite(value)
                    else value
                )
                for key, value in row.items()
            }
        )
    return cleaned


def restore_json_rows(rows):
    return [
        {key: (np.nan if value is None else value) for key, value in row.items()}
        for row in rows
    ]


def wrong_state_delta(current_white, wrong_white, permutation, basis):
    current_residual = candidate_center(current_white.reshape(ACTIONS_PER_STATE, -1))
    wrong_residual = candidate_center(wrong_white.reshape(ACTIONS_PER_STATE, -1))
    difference = wrong_residual[permutation] - current_residual
    projected = (difference @ basis) @ basis.T
    return projected.reshape(current_white.shape)


def intervention_specs(record, carrier, subspaces):
    record_id = int(record["record_id"])
    permutation = np.asarray(donor_permutations[str(record_id)], dtype=np.int64)
    white = whiten_carrier(carrier, subspaces)
    primary_basis = subspaces["primary_basis"][:, :ACTIVE_PRIMARY_RANK]
    primary_full = action_swap_delta(white, permutation, basis=primary_basis, dose=1.0)
    if np.linalg.norm(primary_full) <= 1e-12:
        raise RuntimeError("primary projected donor edit is degenerate")
    specs = []
    for dose in ACTIVE_CAUSAL_DOSES:
        specs.append(
            {
                "condition": f"primary_r{ACTIVE_PRIMARY_RANK:02d}",
                "family": "primary",
                "dose": float(dose),
                "delta_white": float(dose) * primary_full,
            }
        )
    for rank in ACTIVE_SENSITIVITY_RANKS:
        if int(rank) == int(ACTIVE_PRIMARY_RANK):
            continue
        basis = subspaces["primary_basis"][:, : int(rank)]
        specs.append(
            {
                "condition": f"rank_{int(rank):02d}",
                "family": "rank_sensitivity",
                "dose": 1.0,
                "delta_white": action_swap_delta(white, permutation, basis=basis, dose=1.0),
            }
        )
    shuffled = action_swap_delta(
        white,
        permutation,
        basis=subspaces["shuffled_basis"][:, :ACTIVE_PRIMARY_RANK],
        dose=1.0,
    )
    specs.append(
        {
            "condition": "shuffled_fit",
            "family": "matched_subspace_control",
            "dose": 1.0,
            "delta_white": norm_match(shuffled, primary_full),
        }
    )
    for draw in range(ACTIVE_CAUSAL_RANDOM_DRAWS):
        candidate = action_swap_delta(
            white,
            permutation,
            basis=subspaces[f"random_basis_{draw:02d}"],
            dose=1.0,
        )
        specs.append(
            {
                "condition": f"random_{draw:02d}",
                "family": "empirical_span_random_control",
                "dose": 1.0,
                "delta_white": norm_match(candidate, primary_full),
            }
        )
    wrong_id = int(wrong_state_map[str(record_id)])
    wrong_payload = load_baseline(wrong_id)
    wrong_carrier = carrier_for_block(wrong_payload, SELECTED_BLOCK)
    wrong_white = whiten_carrier(wrong_carrier, subspaces)
    wrong = wrong_state_delta(current_white=white, wrong_white=wrong_white, permutation=permutation, basis=primary_basis)
    specs.append(
        {
            "condition": "wrong_state_donor",
            "family": "state_specificity_control",
            "dose": 1.0,
            "delta_white": norm_match(wrong, primary_full),
        }
    )
    common = matched_common_mode(primary_full, primary_basis[:, 0])
    specs.append(
        {
            "condition": "common_mode",
            "family": "matched_common_mode_control",
            "dose": 1.0,
            "delta_white": common,
        }
    )
    specs.append(
        {
            "condition": "full_activation_swap",
            "family": "positive_control_only",
            "dose": 1.0,
            "delta_white": action_swap_delta(white, permutation, basis=None, dose=1.0),
        }
    )
    primary_norm = float(np.linalg.norm(primary_full))
    for specification in specs:
        specification["primary_reference_norm"] = primary_norm
        specification["edit_norm"] = float(np.linalg.norm(specification["delta_white"]))
        specification["full_swap_norm"] = float(
            np.linalg.norm(action_swap_delta(white, permutation, basis=None, dose=1.0))
        )
    return permutation, specs


def no_edit_row(record, permutation, baseline_output, baseline_pose, true_cost, goal):
    output_metrics = donor_transfer_metrics(baseline_output, baseline_output, permutation)
    pose_metrics = donor_transfer_metrics(baseline_pose, baseline_pose, permutation)
    planning = ranking_metrics(true_cost, decoded_task_cost(baseline_pose, goal))
    return {
        "record_id": int(record["record_id"]),
        "trajectory_id": int(record["trajectory_id"]),
        "time_index": int(record["time_index"]),
        "task_id": int(record["task_id"]),
        "selected_block": int(SELECTED_BLOCK),
        "condition": "no_edit",
        "family": "baseline",
        "dose": 0.0,
        "output_coefficient": output_metrics["coefficient"],
        "output_cosine": output_metrics["cosine"],
        "output_reconstruction": output_metrics["reconstruction"],
        "output_mean_shift_ratio": output_metrics["mean_shift_ratio"],
        "pose_coefficient": pose_metrics["coefficient"],
        "pose_cosine": pose_metrics["cosine"],
        "normalized_regret": planning["normalized_regret"],
        "weighted_pairwise_accuracy": planning["weighted_pairwise_accuracy"],
        "top1_correct": planning["top1_correct"],
        "selected_action": planning["selected_action"],
        "oracle_action": planning["oracle_action"],
        "output_rms_change": 0.0,
        "carrier_edit_whitened_norm": 0.0,
        "primary_reference_norm": 0.0,
        "full_swap_norm": 0.0,
        "edit_to_primary_ratio": 0.0,
        "edit_to_full_swap_ratio": 0.0,
    }


def run_record_interventions(record, subspaces):
    destination = intervention_path(record["record_id"])
    if destination.exists():
        return restore_json_rows(json.loads(destination.read_text()))
    payload = load_baseline(record["record_id"])
    carrier = carrier_for_block(payload, SELECTED_BLOCK)
    baseline_output = payload["output_eval_sketch"].astype(np.float64)
    baseline_pose = payload["decoded_pose"].astype(np.float64)
    true_cost, goal = truth_costs(record)
    permutation, specifications = intervention_specs(record, carrier, subspaces)
    rows = [
        no_edit_row(
            record, permutation, baseline_output, baseline_pose, true_cost, goal
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
                capture_blocks=[SELECTED_BLOCK],
                intervention={"block": SELECTED_BLOCK, "delta": delta_tensor},
            )
            patched_output = EVAL_OUTPUT_PROJECTOR(patched).cpu().numpy()
            patched_pose = DECODE_PHYSICAL_POSE(patched).cpu().numpy()
        output_metrics = donor_transfer_metrics(
            baseline_output, patched_output, permutation
        )
        pose_metrics = donor_transfer_metrics(baseline_pose, patched_pose, permutation)
        planning = ranking_metrics(true_cost, decoded_task_cost(patched_pose, goal))
        edit_norm = float(specification["edit_norm"])
        rows.append(
            {
                "record_id": int(record["record_id"]),
                "trajectory_id": int(record["trajectory_id"]),
                "time_index": int(record["time_index"]),
                "task_id": int(record["task_id"]),
                "selected_block": int(SELECTED_BLOCK),
                "condition": specification["condition"],
                "family": specification["family"],
                "dose": float(specification["dose"]),
                "output_coefficient": output_metrics["coefficient"],
                "output_cosine": output_metrics["cosine"],
                "output_reconstruction": output_metrics["reconstruction"],
                "output_mean_shift_ratio": output_metrics["mean_shift_ratio"],
                "pose_coefficient": pose_metrics["coefficient"],
                "pose_cosine": pose_metrics["cosine"],
                "normalized_regret": planning["normalized_regret"],
                "weighted_pairwise_accuracy": planning["weighted_pairwise_accuracy"],
                "top1_correct": planning["top1_correct"],
                "selected_action": planning["selected_action"],
                "oracle_action": planning["oracle_action"],
                "output_rms_change": float(np.sqrt(np.mean((patched_output - baseline_output) ** 2))),
                "carrier_edit_whitened_norm": edit_norm,
                "primary_reference_norm": float(specification["primary_reference_norm"]),
                "full_swap_norm": float(specification["full_swap_norm"]),
                "edit_to_primary_ratio": edit_norm / max(float(specification["primary_reference_norm"]), 1e-12),
                "edit_to_full_swap_ratio": edit_norm / max(float(specification["full_swap_norm"]), 1e-12),
            }
        )
        del patched, patched_output, patched_pose, delta_tensor
    write_json(destination, finite_json_rows(rows))
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


decision_and_plots = r'''# Aggregate by independent trajectory and apply the frozen scientific gate.


def trajectory_summaries(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["trajectory_id"], row["condition"], float(row["dose"]))].append(row)
    summaries = []
    for (trajectory_id, condition, dose), values in sorted(grouped.items()):
        summaries.append(
            {
                "trajectory_id": int(trajectory_id),
                "condition": condition,
                "dose": float(dose),
                "records": len(values),
                "output_coefficient": float(np.mean([row["output_coefficient"] for row in values])),
                "output_cosine": float(np.mean([row["output_cosine"] for row in values])),
                "output_reconstruction": float(np.mean([row["output_reconstruction"] for row in values])),
                "output_mean_shift_ratio": float(np.mean([row["output_mean_shift_ratio"] for row in values])),
                "pose_coefficient": float(np.mean([row["pose_coefficient"] for row in values])),
                "normalized_regret": float(np.mean([row["normalized_regret"] for row in values])),
                "weighted_pairwise_accuracy": float(np.mean([row["weighted_pairwise_accuracy"] for row in values])),
                "output_rms_change": float(np.mean([row["output_rms_change"] for row in values])),
            }
        )
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


def evaluate_causal_gate(summary):
    trajectories = sorted({row["trajectory_id"] for row in summary})
    primary_name = f"primary_r{ACTIVE_PRIMARY_RANK:02d}"
    primary = np.asarray([lookup(summary, value, primary_name) for value in trajectories])
    primary_cosine = np.asarray(
        [lookup(summary, value, primary_name, key="output_cosine") for value in trajectories]
    )
    primary_shift = np.asarray(
        [lookup(summary, value, primary_name, key="output_mean_shift_ratio") for value in trajectories]
    )
    full = np.asarray(
        [lookup(summary, value, "full_activation_swap") for value in trajectories]
    )
    shuffled = np.asarray([lookup(summary, value, "shuffled_fit") for value in trajectories])
    common = np.asarray([lookup(summary, value, "common_mode") for value in trajectories])
    random_values = []
    for trajectory_id in trajectories:
        values = [
            lookup(summary, trajectory_id, f"random_{draw:02d}")
            for draw in range(ACTIVE_CAUSAL_RANDOM_DRAWS)
        ]
        random_values.append(float(np.nanmedian(values)))
    random_values = np.asarray(random_values)
    gain_random = primary - random_values
    gain_shuffled = primary - shuffled
    sign = exact_positive_sign_test(gain_random)
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
        if -0.5 in ACTIVE_CAUSAL_DOSES
        else np.full(len(trajectories), np.nan)
    )
    bootstrap = clustered_bootstrap_mean(
        gain_random,
        np.asarray(trajectories),
        ACTIVE_BOOTSTRAP_DRAWS,
        BOOTSTRAP_SEED,
    )
    result = {
        "primary_condition": primary_name,
        "trajectories": len(trajectories),
        "mean_primary_coefficient": float(np.nanmean(primary)),
        "mean_primary_cosine": float(np.nanmean(primary_cosine)),
        "mean_primary_mean_shift_ratio": float(np.nanmean(primary_shift)),
        "mean_full_swap_coefficient": float(np.nanmean(full)),
        "mean_common_mode_coefficient": float(np.nanmean(common)),
        "mean_random_coefficient": float(np.nanmean(random_values)),
        "mean_shuffled_coefficient": float(np.nanmean(shuffled)),
        "mean_gain_over_random": float(np.nanmean(gain_random)),
        "mean_gain_over_shuffled": float(np.nanmean(gain_shuffled)),
        "gain_over_random_ci95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "gain_over_random_sign_test": sign,
        "positive_dose_slope_trajectories": int(np.sum(np.asarray(dose_slopes) > 0)),
        "negative_dose_mean": float(np.nanmean(negative)) if np.any(np.isfinite(negative)) else None,
        "full_swap_control_pass": bool(np.nanmean(full) >= MIN_FULL_SWAP_COEFFICIENT),
        "primary_absolute_pass": bool(
            np.nanmean(primary) >= MIN_PRIMARY_COEFFICIENT
            and np.nanmean(primary_cosine) >= MIN_PRIMARY_COSINE
            and np.nanmean(primary_shift) <= MAX_PRIMARY_MEAN_SHIFT_RATIO
        ),
        "specificity_pass": bool(
            np.nanmean(gain_random) >= MIN_PRIMARY_GAIN_OVER_RANDOM
            and np.nanmean(gain_shuffled) >= MIN_PRIMARY_GAIN_OVER_SHUFFLED
            and sign["positive"] >= min(REQUIRED_POSITIVE_EVALUATION_TRAJECTORIES, len(trajectories))
            and (sign["p_value"] <= 0.05 if RUN_MODE == "pilot" else True)
        ),
        "dose_direction_pass": bool(
            RUN_MODE == "smoke"
            or (
                np.sum(np.asarray(dose_slopes) > 0) >= REQUIRED_POSITIVE_EVALUATION_TRAJECTORIES
                and np.nanmean(negative) < 0
            )
        ),
    }
    result["causal_gate_pass"] = bool(
        result["full_swap_control_pass"]
        and result["primary_absolute_pass"]
        and result["specificity_pass"]
        and result["dose_direction_pass"]
    )
    return result


def make_plots(rows, summary):
    selection = json.loads((ANALYSIS_DIR / "construction_layer_selection.json").read_text())
    blocks = [row["block"] for row in selection["summary"]]
    observed = [row["mean_cka"] for row in selection["summary"]]
    shuffled = [row["mean_cka"] - row["mean_shuffle_advantage"] for row in selection["summary"]]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(blocks, observed, marker="o", label="observed")
    axes[0].plot(blocks, shuffled, marker="o", label="action-shuffled")
    axes[0].axvline(SELECTED_BLOCK, color="black", linestyle="--", alpha=0.5)
    axes[0].set(xlabel="predictor block", ylabel="linear CKA", title="Construction geometry")
    axes[0].legend()

    primary_name = f"primary_r{ACTIVE_PRIMARY_RANK:02d}"
    doses = sorted({row["dose"] for row in summary if row["condition"] == primary_name})
    means = [
        np.mean([row["output_coefficient"] for row in summary if row["condition"] == primary_name and np.isclose(row["dose"], dose)])
        for dose in doses
    ]
    axes[1].plot(doses, means, marker="o")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set(xlabel="intervention dose", ylabel="donor coefficient", title="Primary dose response")

    conditions = [primary_name, "shuffled_fit", "common_mode", "wrong_state_donor", "full_activation_swap"]
    labels = ["primary", "shuffled", "common", "wrong state", "full swap"]
    values = [
        np.mean([row["output_coefficient"] for row in summary if row["condition"] == condition and np.isclose(row["dose"], 1.0)])
        for condition in conditions
    ]
    axes[2].bar(np.arange(len(values)), values)
    axes[2].set_xticks(np.arange(len(values)), labels, rotation=35, ha="right")
    axes[2].set(ylabel="donor coefficient", title="Held-out causal controls")
    figure.tight_layout()
    figure.savefig(PLOT_DIR / "stage17_action_contrast_summary.png", dpi=180)
    plt.close(figure)


if PIPELINE_FAILED:
    DECISION_PAYLOAD = {"status": "INCONCLUSIVE", "failure": FAILURE_MESSAGE}
elif RUN_MODE == "pilot" and not CONSTRUCTION_GATE_PASS:
    DECISION_PAYLOAD = {
        "status": "STOP_NO_CONSTRUCTION_ACTION_GEOMETRY",
        "construction_gate": LAYER_SELECTION,
    }
elif not EVALUATION_OPENED:
    DECISION_PAYLOAD = {"status": "INCONCLUSIVE", "reason": "evaluation was not opened"}
else:
    try:
        TRAJECTORY_SUMMARY = trajectory_summaries(INTERVENTION_ROWS)
        CAUSAL_GATE = evaluate_causal_gate(TRAJECTORY_SUMMARY)
        if RUN_MODE == "smoke":
            candidate_status = "SMOKE_ONLY"
        elif CAUSAL_GATE["causal_gate_pass"]:
            candidate_status = "FINITE_ACTION_CONTRAST_CAUSAL_MEDIATION"
        elif CAUSAL_GATE["full_swap_control_pass"]:
            candidate_status = "FULL_SWAP_ONLY_NO_COMPRESSED_MEDIATION"
        else:
            candidate_status = "NO_INTERNAL_ACTION_CONTRAST_SIGNAL"
        source_eligible = bool(SOURCE_IDENTITY.get("confirmation_eligible", False))
        status = (
            candidate_status
            if RUN_MODE == "smoke" or source_eligible
            else "UNBOUND_EXPLORATORY_RESULT"
        )
        DECISION_PAYLOAD = {
            "status": status,
            "candidate_status": candidate_status,
            "source_bound_claim_eligible": source_eligible,
            "construction_gate": LAYER_SELECTION,
            "causal_gate": CAUSAL_GATE,
            "claim_boundary": {
                "coordinate_chart_authorized": False,
                "jacobian_claim_authorized": False,
                "koopman_or_transport_claim_authorized": False,
                "full_swap_alone_is_mechanistic_evidence": False,
                "causal_claim_is_internal_predicted_consequence_mediation_only": True,
            },
        }
        write_json(OUT / "stage17_decision.json", DECISION_PAYLOAD)
        make_plots(INTERVENTION_ROWS, TRAJECTORY_SUMMARY)
        print(json.dumps(DECISION_PAYLOAD, indent=2))
    except Exception:
        record_failure("decision_and_plots")
        DECISION_PAYLOAD = {"status": "INCONCLUSIVE", "failure": FAILURE_MESSAGE}

if not (OUT / "stage17_decision.json").exists():
    write_json(OUT / "stage17_decision.json", DECISION_PAYLOAD)
'''


packaging = r'''# Package compact audit evidence and download one result bundle.
write_json(OUT / "timings.json", TIMINGS)
memory_report("final")
if not PIPELINE_FAILED:
    (OUT / "FAILURE_TRACE.txt").write_text("NONE\n")

excluded_roots = {ASSET_DIR, TRUTH_DIR, BASELINE_DIR, INTERVENTION_DIR}
compact_files = []
for path in sorted(OUT.rglob("*")):
    if not path.is_file():
        continue
    if any(root == path or root in path.parents for root in excluded_roots):
        continue
    if SUBSPACE_DIR in path.parents and path.suffix == ".npz":
        continue
    if path.name.startswith("stage17_action_contrast_result_bundle_"):
        continue
    compact_files.append(path)

manifest = [
    {
        "path": str(path.relative_to(OUT)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
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

archive_base = OUT / f"stage17_action_contrast_result_bundle_{RUN_SIGNATURE[:12]}"
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
    cell["id"] = f"stage17-{index:02d}"

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "gpuType": "L4",
            "name": TARGET.name,
            "provenance": [],
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
TARGET.write_text(json.dumps(notebook, indent=1) + "\n")
print(f"Wrote {TARGET}")
