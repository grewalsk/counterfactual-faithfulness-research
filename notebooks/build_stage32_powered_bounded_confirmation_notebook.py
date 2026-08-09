import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "32_powered_bounded_cross_model_confirmation.ipynb"
NUMERICAL = ROOT.parent / "src/cf_faithfulness/stage32_bounded_confirmation.py"

spec = importlib.util.spec_from_file_location(
    "stage31_builder", ROOT / "build_stage31_cross_model_grounded_certificate_notebook.py"
)
STAGE31 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(STAGE31)

code = STAGE31.code
markdown = STAGE31.markdown
assigned_uppercase_names = STAGE31.assigned_uppercase_names
function_sources = STAGE31.function_sources


introduction = r'''# Stage 32: powered bounded cross-model confirmation

Stage 31 produced a clean but incomplete result.  Physically grounded closure
improved held-out planning-regret prediction inside both JEPA-WM and DINO-WM,
yet the paired DINO-minus-JEPA certificate missed its bootstrap gate.  It also
revealed a numerical failure mode: coefficient ratios become meaningless when
the physical target contrast has nearly zero energy.  Six such JEPA rows
created enormous coefficients even though the bounded cosine signal and the
paired panel were stable.

This notebook is the frozen confirmation of the scientifically defensible
signal.  It imports the exact Stage 31 model-specific rank-128 bases, selects
160 entirely new persistent-contact PushT states without model access, and
evaluates three prespecified action geometries: ±20°, ±30°, and ±40°.  The
public JEPA-WM and DINO-WM checkpoints see exactly the same physical branches.

Grounded closure is now represented **only** by cosine, bounded to [-1, 1], and
is undefined unless the exact encoded-physical target contrast has energy at
least `1e-6`.  No coefficient enters any gate or plot.  Closure still uses the
four interior schedules; planning uses the two excluded extreme schedules and
the exact public visual-MSE + 0.1 proprio-MSE objective.

The primary paired estimand remains

\[
Y=R_{\mathrm{DINO}}-R_{\mathrm{JEPA}}.
\]

A state-grouped out-of-fold base model uses action magnitude, geometry,
ordinary joint target error, and self-consistent causal cosine differences.
The confirmatory model adds only the difference in bounded physically grounded
cosine.  A full certificate requires at least 5% relative held-out MSE
improvement, a positive state-bootstrap interval, positive mean improvement in
all three action geometries, and a positive bootstrap advantage over the
median of shuffled-output and two empirical-span random-subspace placebos.

The official asset inventory contains only one public PushT checkpoint for
each family.  This notebook therefore does not manufacture pseudo-checkpoints;
it is a powered two-architecture, three-action-family confirmation.  It uses no
decoder, probe, reader, coefficient ratio, gradient, Jacobian, JVP, or VJP.
Return `stage32_bounded_confirmation_result_bundle_<signature>.zip`.
'''


configuration = r'''# SINGLE CONFIGURATION BLOCK — no Stage 32 secrets required.
import secrets as _secrets
import time as _time

RUN_MODE = "pilot"
EXPERIMENT_SOURCE_REF = "codex/stage32-powered-bounded-cross-model"
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
FRESH_RUN_REQUIRED = True
CONTINUE_AFTER_BENCHMARK = True
MAX_ESTIMATED_TOTAL_MINUTES = 180.0

OUTPUT_DIR = "/content/counterfactual_faithfulness_stage32_bounded_cross_model"
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/counterfactual_faithfulness_stage32_bounded_cross_model"

PROTOCOL_ID = "stage32-powered-bounded-cross-model-confirmation-v1"
NOTEBOOK_PROTOCOL_SHA256 = "__PROTOCOL_DIGEST__"
EVIDENCE_STATUS = "CONFIRMATORY_ONLY_IF_SOURCE_BOUND_FRESH_STAGE31_FROZEN_AND_ENERGY_ELIGIBLE"
EXPERIMENT_REPOSITORY = "grewalsk/counterfactual-faithfulness-research"
EXPERIMENT_NOTEBOOK_PATH = "notebooks/32_powered_bounded_cross_model_confirmation.ipynb"
EXPERIMENT_BUILDER_PATH = "notebooks/build_stage32_powered_bounded_confirmation_notebook.py"
EXPERIMENT_NUMERICAL_PATH = "src/cf_faithfulness/stage32_bounded_confirmation.py"

SEED = 32101
DESIGN_SEED = 32137
BOOTSTRAP_SEED = 32269
CROSSFIT_SEED = 32319
NULL_ROOT_SEED = 32411
ENVIRONMENT = "PushT"
MODEL_NAMES = ["jepa_wm_pusht", "dino_wm_pusht"]
MODEL_SHORT_NAMES = {"jepa_wm_pusht": "jepa", "dino_wm_pusht": "dino"}
EXPECTED_MODEL_TYPES = {"jepa_wm_pusht": "AdaLN", "dino_wm_pusht": "dino_wm"}
EXPECTED_CARRIER_WIDTHS = {"jepa_wm_pusht": 400, "dino_wm_pusht": 414}

EXPECTED_STAGE31_STATUS = "WITHIN_MODEL_REPLICATION_WITHOUT_PAIRED_CERTIFICATE"
EXPECTED_STAGE31_PROTOCOL_ID = "stage31-cross-model-grounded-closure-certificate-v1"
EXPECTED_STAGE31_SOURCE_COMMIT = "8b407c34117d1029ad4bd212e3fff03062f2b437"
EXPECTED_STAGE31_DECISION_SHA256 = "9d95d79f627aa62741a2c56fd9e10f4b257ca4d979393bcb950b523bda39c193"
EXPECTED_STAGE31_SOURCE_SHA256 = "98dce1a1109c61d6c045d1ffddbc482a4b77fc1dc898036751929c1cae114e5e"
EXPECTED_STAGE31_SUBSPACE_SHA256 = {
    "jepa": "762c9437a0bc5c8726359158394ae0fd10814aa63f59f6323a2c81c8bad6703c",
    "dino": "6e146353775f6f714f8401337f4241af3cf7d53111544c301949c34817a7499c",
}
EXPECTED_STAGE31_MANIFEST_SHA256 = {
    "jepa": "7559a5e4fd20a8366e309eaaebb6b45d48940a4ead099f3e6e289e082adeb04b",
    "dino": "e631cde0d12f83b68ad541766ac7680a3b6e7826d001c8aab00000d9d6390c8d",
}
EXPECTED_STAGE31_SELECTED_BLOCK = {"jepa": 5, "dino": 5}
EXPECTED_STAGE31_AMBIENT_DIMENSION = {"jepa": 102400, "dino": 105984}
EXPECTED_STAGE31_RANK = 128
STAGE31_SEARCH_ROOT = "/content/drive/MyDrive"

FRAMESKIP = 5
PRIMARY_HORIZON = 3
ACTION_STEPS = PRIMARY_HORIZON * FRAMESKIP
PRIMARY_RANK = 128
POOL_TRAJECTORIES = list(range(5000, 5800))
EVALUATION_TARGET = 160
TASK_ID_OFFSET = 32000
DISTANCE_GRID = [55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0]

SELECTED_MAGNITUDES = [0.10, 0.14, 0.18, 0.22]
MAGNITUDE_COUNT = 4
SCHEDULE_STRINGS = [
    "uuuuuvvvvv", "uuvuuvvuvv", "uvuvuvuvuv",
    "vuvuvuvuvu", "vvuvvuuvuu", "vvvvvuuuuu",
]
SCHEDULE_COUNT = 6
SCHEDULE_INVERSION_COUNTS = [0, 5, 10, 15, 20, 25]
SIGNED_AREA_LEVELS = [25, 15, 5, -5, -15, -25]
ACTION_FAMILIES = [
    {"name": "narrow", "angle_pair_degrees": [-20.0, 20.0]},
    {"name": "reference", "angle_pair_degrees": [-30.0, 30.0]},
    {"name": "wide", "angle_pair_degrees": [-40.0, 40.0]},
]
ACTION_FAMILY_COUNT = 3
ACTIONS_PER_FAMILY = MAGNITUDE_COUNT * SCHEDULE_COUNT
TOTAL_ACTIONS_PER_STATE = ACTION_FAMILY_COUNT * ACTIONS_PER_FAMILY
DIAGNOSTIC_SCHEDULES = [1, 2, 3, 4]
PLANNING_GOAL_SCHEDULES = [0, 5]
OFFICIAL_PROPRIO_ALPHA = 0.1

INTERVENTION_CONDITIONS = [
    "primary_r128_swap", "shuffled_r128_swap",
    "random_r128_00_swap", "random_r128_01_swap",
]
PLACEBO_CONDITIONS = [
    "shuffled_r128_swap", "random_r128_00_swap", "random_r128_01_swap",
]
INTERVENTION_FORWARDS_PER_FAMILY = 4
MIN_GROUNDED_TARGET_ENERGY = 1e-6
MAX_ZERO_EDIT_ERROR = 1e-6
MIN_PLANNING_TRUE_COST_SPREAD = 1e-5
BOOTSTRAP_DRAWS = 10000
CROSSFIT_FOLDS = 5
MIN_ELIGIBLE_STATES = 140
MIN_PAIRED_RELATIVE_MSE_IMPROVEMENT = 0.05
MIN_WITHIN_MODEL_RELATIVE_MSE_IMPROVEMENT = 0.05
REQUIRE_ALL_FAMILY_MEAN_IMPROVEMENTS_POSITIVE = True
REQUIRE_PLACEBO_ADVANTAGE_CI_POSITIVE = True

if RUN_MODE == "smoke":
    ACTIVE_POOL_TRAJECTORIES = POOL_TRAJECTORIES[:60]
    ACTIVE_EVALUATION_TARGET = 6
    ACTIVE_BOOTSTRAP_DRAWS = 64
    ACTIVE_CROSSFIT_FOLDS = 2
    ACTIVE_MIN_ELIGIBLE_STATES = 4
elif RUN_MODE == "pilot":
    ACTIVE_POOL_TRAJECTORIES = POOL_TRAJECTORIES
    ACTIVE_EVALUATION_TARGET = EVALUATION_TARGET
    ACTIVE_BOOTSTRAP_DRAWS = BOOTSTRAP_DRAWS
    ACTIVE_CROSSFIT_FOLDS = CROSSFIT_FOLDS
    ACTIVE_MIN_ELIGIBLE_STATES = MIN_ELIGIBLE_STATES
else:
    raise ValueError("RUN_MODE must be 'smoke' or 'pilot'")

REPO_URL = "https://github.com/facebookresearch/jepa-wms.git"
REPO_COMMIT = "13cf1d9c7e476f53c17714d2e0f1dc239a883ce0"
EXPECTED_HF_REVISION = "9b9c41ef249466630dbf1a20e78391865d07b3b9"
EXPECTED_PRETRAINED_ASSET_SHA256 = {
    "jepa_wm_pusht.pth.tar": "9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb",
    "dino_wm_pusht.pth.tar": "8ec9cb05f22812d7f12e3c216b0637f41641055c0653e503e2746edb981b550f",
    "dinov2_vits14_pretrain.pth": "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9",
}
ASSET_REPOSITORY = "grewalsk/counterfactual-faithfulness-research"
ASSET_COMMIT = "2326e74556f6f81db2560e4396f4cc52c16a28f4"
ASSET_SPECS = {}
PINNED = [
    "exact_stage31_model_specific_subspaces", "new_persistent_contact_states",
    "three_prespecified_action_geometries", "bounded_grounded_cosine_only",
    "absolute_target_energy_floor", "interior_schedule_closure",
    "disjoint_extreme_schedule_goals", "official_visual_plus_proprio_objective",
    "paired_dino_minus_jepa_estimand", "state_grouped_cross_fitting",
    "shuffled_and_empirical_span_placebos", "no_required_colab_secrets",
]

assert ACTION_STEPS == 15
assert len(ACTION_FAMILIES) == ACTION_FAMILY_COUNT == 3
assert ACTIONS_PER_FAMILY == 24 and TOTAL_ACTIONS_PER_STATE == 72
assert len(INTERVENTION_CONDITIONS) == INTERVENTION_FORWARDS_PER_FAMILY
assert set(PLACEBO_CONDITIONS) < set(INTERVENTION_CONDITIONS)
assert not set(DIAGNOSTIC_SCHEDULES) & set(PLANNING_GOAL_SCHEDULES)
assert MIN_GROUNDED_TARGET_ENERGY > 0
'''
configuration += "\n\nPROTOCOL_CONFIG_KEYS = " + repr(
    assigned_uppercase_names(configuration)
) + "\n"


installation = STAGE31.installation


setup = STAGE31.setup
setup = setup.replace("Stage 31", "Stage 32").replace("STAGE31", "STAGE32")
setup = setup.replace("stage31_cross_model", "stage32_bounded_cross_model")
setup = setup.replace("stage31-source-binder", "stage32-source-binder")
setup = setup.replace(
    "stage31_cross_model_certificate_result_bundle_",
    "stage32_bounded_confirmation_result_bundle_",
)


analysis_helpers = STAGE31.analysis_helpers + "\n\n\n" + function_sources(
    NUMERICAL.read_text(),
    [
        "bounded_cosine",
        "bounded_swap_closure_rows",
        "paired_model_difference_rows",
        "state_placebo_advantage",
    ],
)

# The simulator consumes float32 actions, but the equal-impulse/energy checks are
# mathematical invariants of an identical pulse multiset.  Accumulating those
# diagnostics in float32 makes their result depend on schedule order and caused
# a false failure at a 5.96e-7 discrepancy.  Preserve the actions exactly while
# evaluating the invariants with order-stable float64 accumulation.
_float32_invariant_check = """        impulses = np.sum(group, axis=1)
        energies = np.sum(group**2, axis=(1, 2))"""
_float64_invariant_check = """        diagnostic_group = group.astype(np.float64)
        impulses = np.sum(diagnostic_group, axis=1)
        energies = np.sum(diagnostic_group**2, axis=(1, 2))"""
if analysis_helpers.count(_float32_invariant_check) != 1:
    raise RuntimeError("could not locate the inherited action-invariant check")
analysis_helpers = analysis_helpers.replace(
    _float32_invariant_check, _float64_invariant_check
)


model_helpers = STAGE31.model_helpers
model_helpers = model_helpers.replace("stage31-jepa-wms", "stage32-jepa-wms")
model_helpers = model_helpers.replace("Stage 31 supports PushT only", "Stage 32 supports PushT only")


design_and_upstream = r'''# Bind the exact Stage 31 bases and freeze all new tasks before simulator or model use.
STAGE31_ARTIFACTS_VALIDATED = False
UPSTREAM_SUBSPACE_PATHS = {}


def make_specs(trajectory_ids):
    center = np.asarray([256.0, 256.0], dtype=np.float64)
    specs = []
    for trajectory_id in trajectory_ids:
        global_index = POOL_TRAJECTORIES.index(int(trajectory_id))
        phase = 0.619 + 2.0 * np.pi * global_index / len(POOL_TRAJECTORIES)
        block = center + 43.0 * np.asarray([np.cos(phase), np.sin(phase)])
        block_angle = ((1.73 * phase + np.pi) % (2.0 * np.pi)) - np.pi
        offsets = [np.pi / 6, 5 * np.pi / 6, 7 * np.pi / 6, 11 * np.pi / 6]
        approach = phase + offsets[global_index % 4] + 0.11 * np.sin(5 * global_index)
        distance = float(
            DISTANCE_GRID[
                (5 * global_index + global_index // len(DISTANCE_GRID))
                % len(DISTANCE_GRID)
            ]
        )
        agent = block + distance * np.asarray([np.cos(approach), np.sin(approach)])
        goal_index = (23 * global_index + 11) % len(POOL_TRAJECTORIES)
        goal_phase = 1.117 + 2.0 * np.pi * goal_index / len(POOL_TRAJECTORIES)
        goal_xy = center + 72.0 * np.asarray([np.cos(goal_phase), np.sin(goal_phase)])
        specs.append({
            "design_index": int(global_index),
            "record_id": int(530000 + trajectory_id),
            "trajectory_id": int(trajectory_id),
            "task_id": int(TASK_ID_OFFSET + global_index),
            "split": "stage32_fresh_evaluation_pool",
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


def family_action_bank(record, family):
    return area_action_bank(
        np.asarray(record["state"][2:4] - record["state"][:2], dtype=np.float64),
        SELECTED_MAGNITUDES,
        steps=ACTION_STEPS,
        angle_pair_degrees=family["angle_pair_degrees"],
        schedules=SCHEDULE_STRINGS,
    ).astype(np.float32)


def all_family_action_banks(record):
    return np.stack([family_action_bank(record, family) for family in ACTION_FAMILIES])


np.savez_compressed(
    DESIGN_DIR / "stage32_bounded_confirmation_design.npz",
    record_ids=np.asarray([row["record_id"] for row in POOL_SPECS], dtype=np.int64),
    initial_states=np.stack([row["state"] for row in POOL_SPECS]),
    goals=np.stack([row["goal"] for row in POOL_SPECS]),
    magnitudes=np.asarray(SELECTED_MAGNITUDES, dtype=np.float64),
    schedules=np.asarray(SCHEDULE_STRINGS),
    family_names=np.asarray([family["name"] for family in ACTION_FAMILIES]),
    family_angles=np.asarray([family["angle_pair_degrees"] for family in ACTION_FAMILIES]),
    diagnostic_schedules=np.asarray(DIAGNOSTIC_SCHEDULES, dtype=np.int64),
    planning_goal_schedules=np.asarray(PLANNING_GOAL_SCHEDULES, dtype=np.int64),
)
write_json(DESIGN_DIR / "candidate_pool_manifest.json", {
    "pool_specs": [
        {**{key: value for key, value in row.items() if key not in {"state", "goal"}},
         "state": row["state"].tolist(), "goal": row["goal"].tolist()}
        for row in POOL_SPECS
    ],
    "selection_rule": "first states with contact on all 72 branches; no model output or effect magnitude used",
    "target_persistent_states": ACTIVE_EVALUATION_TARGET,
    "action_families": ACTION_FAMILIES,
    "stage31_state_ids_reused": False,
    "grounded_coefficient_allowed": False,
    "minimum_grounded_target_energy": MIN_GROUNDED_TARGET_ENERGY,
})
DESIGN_FREEZE = {
    "created_before_simulator_or_model_data": True,
    "protocol_id": PROTOCOL_ID,
    "run_signature": RUN_SIGNATURE,
    "source_identity": SOURCE_IDENTITY,
    "design_sha256": sha256_file(DESIGN_DIR / "stage32_bounded_confirmation_design.npz"),
    "pool_manifest_sha256": sha256_file(DESIGN_DIR / "candidate_pool_manifest.json"),
    "model_loaded": bool("MODEL_BUNDLE" in globals()),
    "stage31_basis_refit_or_tuning_allowed": False,
    "coefficient_ratio_allowed": False,
    "evaluation_data_allowed_in_basis_fit": False,
}
if DESIGN_FREEZE["model_loaded"]:
    raise RuntimeError("a model was loaded before the Stage 32 design freeze")
write_json(DESIGN_DIR / "design_freeze.json", DESIGN_FREEZE)


if not PIPELINE_FAILED:
    try:
        verify_executed_notebook_through(
            "# Bind the exact Stage 31 bases and freeze all new tasks before simulator or model use."
        )
        stage31_root = Path(STAGE31_SEARCH_ROOT)
        candidates = sorted(stage31_root.glob(
            "counterfactual_faithfulness_stage31_cross_model/pilot_*/stage31_decision.json"
        ))
        valid = []
        for decision_path in candidates:
            run_dir = decision_path.parent
            source_path = run_dir / "source_identity.json"
            if not source_path.is_file():
                continue
            if (
                sha256_file(decision_path) != EXPECTED_STAGE31_DECISION_SHA256
                or sha256_file(source_path) != EXPECTED_STAGE31_SOURCE_SHA256
            ):
                continue
            decision = json.loads(decision_path.read_text())
            source = json.loads(source_path.read_text())
            if (
                decision.get("status") == EXPECTED_STAGE31_STATUS
                and decision.get("confirmation_eligible", False)
                and all(
                    value.get("passed", False)
                    for value in decision.get("within_model_grounded_reliability_gates", {}).values()
                )
                and not decision.get("paired_cross_model_grounded_reliability_gate", {}).get("passed", True)
                and source.get("protocol_id") == EXPECTED_STAGE31_PROTOCOL_ID
                and source.get("resolved_commit") == EXPECTED_STAGE31_SOURCE_COMMIT
                and source.get("confirmation_eligible", False)
            ):
                valid.append((run_dir, decision, source))
        if not valid:
            raise FileNotFoundError(
                "No exact source-bound Stage 31 run was found in MyDrive. Keep the complete Stage 31 Drive directory."
            )
        STAGE31_RUN_DIR, STAGE31_DECISION, STAGE31_SOURCE = valid[-1]
        artifact_rows = []
        for short in ["jepa", "dino"]:
            manifest_path = STAGE31_RUN_DIR / f"subspaces/subspace_manifest_{short}.json"
            subspace_path = STAGE31_RUN_DIR / f"subspaces/frozen_{short}_rank128_subspaces.npz"
            if not manifest_path.is_file() or not subspace_path.is_file():
                raise FileNotFoundError(f"complete Stage 31 {short} subspace is missing")
            if sha256_file(manifest_path) != EXPECTED_STAGE31_MANIFEST_SHA256[short]:
                raise RuntimeError(f"Stage 31 {short} manifest hash mismatch")
            if sha256_file(subspace_path) != EXPECTED_STAGE31_SUBSPACE_SHA256[short]:
                raise RuntimeError(f"Stage 31 {short} subspace hash mismatch")
            manifest = json.loads(manifest_path.read_text())
            if (
                manifest.get("subspace_sha256") != EXPECTED_STAGE31_SUBSPACE_SHA256[short]
                or int(manifest.get("selected_block", -1)) != EXPECTED_STAGE31_SELECTED_BLOCK[short]
                or int(manifest.get("rank", -1)) != EXPECTED_STAGE31_RANK
                or int(manifest.get("ambient_dimension", -1))
                != EXPECTED_STAGE31_AMBIENT_DIMENSION[short]
                or int(manifest.get("evaluation_rows_used", -1)) != 0
            ):
                raise RuntimeError(f"Stage 31 {short} manifest contract changed")
            with np.load(subspace_path) as payload:
                required = {
                    "primary_basis", "shuffled_basis", "random_basis_00", "random_basis_01",
                    "channel_square_root", "channel_inverse_square_root", "selected_block",
                }
                if not required <= set(payload.files):
                    raise RuntimeError(f"Stage 31 {short} subspace arrays are incomplete")
                for name in ["primary_basis", "shuffled_basis", "random_basis_00", "random_basis_01"]:
                    basis = payload[name].astype(np.float64)
                    if basis.shape != (
                        EXPECTED_STAGE31_AMBIENT_DIMENSION[short], EXPECTED_STAGE31_RANK
                    ):
                        raise RuntimeError(f"Stage 31 {short} {name} shape changed")
                    error = float(np.max(np.abs(basis.T @ basis - np.eye(EXPECTED_STAGE31_RANK))))
                    if error > 2e-5:
                        raise RuntimeError(f"Stage 31 {short} {name} lost orthogonality")
            UPSTREAM_SUBSPACE_PATHS[short] = subspace_path
            artifact_rows.append({
                "model": short, "manifest_path": str(manifest_path),
                "manifest_sha256": EXPECTED_STAGE31_MANIFEST_SHA256[short],
                "subspace_path": str(subspace_path),
                "subspace_sha256": EXPECTED_STAGE31_SUBSPACE_SHA256[short],
            })
        STAGE31_CERTIFICATE = {
            "validated_before_stage32_simulator_or_model_data": True,
            "run_dir": str(STAGE31_RUN_DIR),
            "decision_status": STAGE31_DECISION["status"],
            "source_commit": STAGE31_SOURCE["resolved_commit"],
            "decision_sha256": EXPECTED_STAGE31_DECISION_SHA256,
            "source_identity_sha256": EXPECTED_STAGE31_SOURCE_SHA256,
            "artifacts": artifact_rows,
            "stage31_states_reused": False,
            "stage32_basis_refit_or_tuning": False,
        }
        write_json(OUT / "stage31_upstream_certificate.json", STAGE31_CERTIFICATE)
        STAGE31_ARTIFACTS_VALIDATED = True
        memory_report("stage32_design_and_stage31_artifacts_bound")
    except Exception:
        record_failure("stage32_design_or_stage31_binding")
'''


physical_truth = r'''# Select and materialize 160 new all-family persistent-contact states without model access.
PROVENANCE_COUNTS = {
    "screened_states": 0,
    "truth_generated": 0,
    "jepa_state_family_evaluations": 0,
    "dino_state_family_evaluations": 0,
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
            [*environment.agent.position, *environment.agent.velocity], dtype=np.float32
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
                    "visual": np.asarray(observation["visual"]).copy() if retain_visual else None,
                    "proprio": np.asarray(observation["proprio"]).copy(),
                }
                endpoint_state = dynamic_state_from_environment(environment)
    finally:
        environment.close()
    if endpoint_observation is None or endpoint_state is None:
        raise RuntimeError("dynamic rollout missed the primary horizon")
    return initial, endpoint_observation, endpoint_state, cumulative


def truth_path(record_or_id):
    record_id = int(
        record_or_id["record_id"] if isinstance(record_or_id, dict) else record_or_id
    )
    return TRUTH_DIR / f"state_{record_id:06d}.npz"


def screen_pool(records):
    rows = []
    started = time.perf_counter()
    for index, record in enumerate(records):
        contact_counts = np.zeros(
            (ACTION_FAMILY_COUNT, MAGNITUDE_COUNT, SCHEDULE_COUNT), dtype=np.int64
        )
        for family_index, action_bank in enumerate(all_family_action_banks(record)):
            for action_index, branch in enumerate(action_bank):
                _, _, _, count = rollout_dynamic_branch(
                    record, branch, retain_visual=False
                )
                magnitude_index, schedule_index = divmod(action_index, SCHEDULE_COUNT)
                contact_counts[family_index, magnitude_index, schedule_index] = count
        persistent = bool(np.all(contact_counts > 0))
        rows.append({
            "record_id": int(record["record_id"]),
            "trajectory_id": int(record["trajectory_id"]),
            "approach_distance": float(record["approach_distance"]),
            "persistent_all_72_branches": persistent,
            "contact_fraction": float(np.mean(contact_counts > 0)),
            "minimum_branch_contacts": int(np.min(contact_counts)),
            "total_contacts": int(np.sum(contact_counts)),
        })
        PROVENANCE_COUNTS["screened_states"] += 1
        write_json(OUT / "physical_screen_progress.json", {
            "completed": index + 1, "total": len(records),
            "last_record_id": int(record["record_id"]),
        })
    TIMINGS["physical_screen_seconds"] = time.perf_counter() - started
    write_csv(EVIDENCE_DIR / "physical_screen_rows.csv", rows)
    return rows


def select_persistent_records(records, screen_rows):
    lookup = {int(row["record_id"]): row for row in screen_rows}
    selected = [
        record for record in records
        if lookup[int(record["record_id"])]["persistent_all_72_branches"]
    ]
    if len(selected) < ACTIVE_EVALUATION_TARGET:
        raise RuntimeError(
            f"fresh pool has {len(selected)} all-family persistent states; requires {ACTIVE_EVALUATION_TARGET}"
        )
    selected = selected[:ACTIVE_EVALUATION_TARGET]
    for record in selected:
        record["regime"] = "persistent_contact"
    return selected


def generate_truth(records):
    started = time.perf_counter()
    for index, record in enumerate(records):
        destination = truth_path(record)
        if destination.exists():
            PROVENANCE_COUNTS["cache_hits"] += 1
            raise RuntimeError(f"fresh-run truth shard already exists: {destination}")
        action_banks = all_family_action_banks(record)
        initials, initial_proprios = [], []
        endpoint_visuals, endpoint_proprios, endpoint_states, contacts = [], [], [], []
        for action_bank in action_banks:
            family_visuals, family_proprios, family_states, family_contacts = [], [], [], []
            for branch in action_bank:
                initial, endpoint, state, count = rollout_dynamic_branch(
                    record, branch, retain_visual=True
                )
                initials.append(initial["visual"])
                initial_proprios.append(initial["proprio"])
                family_visuals.append(endpoint["visual"])
                family_proprios.append(endpoint["proprio"])
                family_states.append(state)
                family_contacts.append(count)
            endpoint_visuals.append(family_visuals)
            endpoint_proprios.append(family_proprios)
            endpoint_states.append(family_states)
            contacts.append(family_contacts)
        if not all(np.array_equal(initials[0], value) for value in initials[1:]):
            raise RuntimeError("initial visual drift across exact branches")
        if not all(np.array_equal(initial_proprios[0], value) for value in initial_proprios[1:]):
            raise RuntimeError("initial proprio drift across exact branches")
        contacts = np.asarray(contacts, dtype=np.int32).reshape(
            ACTION_FAMILY_COUNT, MAGNITUDE_COUNT, SCHEDULE_COUNT
        )
        if not np.all(contacts > 0):
            raise RuntimeError("persistent-contact state changed during truth regeneration")
        atomic_npz(
            destination,
            record_id=np.asarray(record["record_id"], dtype=np.int64),
            trajectory_id=np.asarray(record["trajectory_id"], dtype=np.int64),
            regime=np.asarray("persistent_contact"),
            state=np.asarray(record["state"], dtype=np.float64),
            goal=np.asarray(record["goal"], dtype=np.float64),
            initial_visual=np.asarray(initials[0], dtype=np.uint8),
            initial_proprio=np.asarray(initial_proprios[0], dtype=np.float32),
            selected_actions=action_banks.astype(np.float32),
            endpoint_visuals=np.asarray(endpoint_visuals, dtype=np.uint8),
            endpoint_proprios=np.asarray(endpoint_proprios, dtype=np.float32),
            endpoint_states=np.asarray(endpoint_states, dtype=np.float64),
            interaction_counts=contacts,
        )
        PROVENANCE_COUNTS["truth_generated"] += 1
        write_json(OUT / "selected_truth_progress.json", {
            "completed": index + 1, "total": len(records),
            "last_record_id": int(record["record_id"]),
        })
    TIMINGS["selected_truth_seconds"] = time.perf_counter() - started


if not PIPELINE_FAILED:
    try:
        verify_executed_notebook_through(
            "# Select and materialize 160 new all-family persistent-contact states without model access."
        )
        if not STAGE31_ARTIFACTS_VALIDATED:
            raise RuntimeError("exact Stage 31 artifacts must be bound first")
        REPO = configure_repo()
        SCREEN_ROWS = screen_pool(POOL_SPECS)
        EVALUATION_RECORDS = select_persistent_records(POOL_SPECS, SCREEN_ROWS)
        generate_truth(EVALUATION_RECORDS)
        SELECTION_CERTIFICATE = {
            "created_before_stage32_model_loading": True,
            "selected_record_ids": [int(row["record_id"]) for row in EVALUATION_RECORDS],
            "selected_states": len(EVALUATION_RECORDS),
            "all_selected_states_contact_on_all_72_branches": True,
            "selection_uses_contact_only": True,
            "model_outputs_used_for_selection": False,
            "effect_magnitude_used_for_selection": False,
            "stage31_record_ids_reused": False,
            "screen_rows_sha256": sha256_file(EVIDENCE_DIR / "physical_screen_rows.csv"),
        }
        write_json(DESIGN_DIR / "physical_selection_freeze.json", SELECTION_CERTIFICATE)
        memory_report("stage32_fresh_persistent_truth_selected")
    except Exception:
        record_failure("stage32_fresh_physical_screen_or_truth")
'''


model_evaluation = r'''# Evaluate the exact frozen Stage 31 bases with bounded cosine and four controlled swaps.


def load_upstream_subspace_cuda(short):
    with np.load(UPSTREAM_SUBSPACE_PATHS[short]) as payload:
        arrays = {name: payload[name].copy() for name in payload.files}
    if int(arrays["selected_block"]) != EXPECTED_STAGE31_SELECTED_BLOCK[short]:
        raise RuntimeError(f"{short} selected block changed")
    cuda = {
        name: torch.as_tensor(value, device="cuda", dtype=torch.float32)
        for name, value in arrays.items()
        if name.endswith("basis") or name.startswith("random_basis_")
        or name in {"channel_square_root", "channel_inverse_square_root"}
    }
    return arrays, cuda


def state_model_inputs(bundle, record, family_index):
    with np.load(truth_path(record)) as payload:
        initial_visual = payload["initial_visual"]
        initial_proprio = payload["initial_proprio"]
        selected_actions = payload["selected_actions"][int(family_index)]
    with torch.inference_mode():
        initial = bundle["model"].encode(
            to_model_observation(initial_visual, initial_proprio)
        )
    initial = {name: value.detach() for name, value in initial.items()}
    actions = model_action_tensor(
        bundle["preprocessor"], selected_actions, PRIMARY_HORIZON
    )
    return initial, actions


def encode_true_state(bundle, record, family_index):
    with np.load(truth_path(record)) as payload:
        visual = payload["endpoint_visuals"][int(family_index), :, None]
        states = payload["endpoint_states"][int(family_index)].astype(np.float32)
    proprio = np.concatenate([states[:, :2], states[:, 5:7]], axis=1)[:, None]
    with torch.inference_mode():
        encoded = bundle["model"].encode(to_model_observation(visual, proprio))
    visual_tokens = encoded["visual"][:, :, 0]
    visual_tokens = visual_tokens.reshape(ACTIONS_PER_FAMILY, 256, visual_tokens.shape[-1])
    proprio_tokens = encoded["proprio"][:, 0]
    if visual_tokens.shape != (ACTIONS_PER_FAMILY, 256, 384):
        raise RuntimeError(f"unexpected true visual-token shape {visual_tokens.shape}")
    if proprio_tokens.shape[0] != ACTIONS_PER_FAMILY:
        raise RuntimeError("true proprio-token action axis changed")
    return visual_tokens.detach(), proprio_tokens.detach()


def matched_tensor_norm(value, reference):
    norm = torch.linalg.vector_norm(value)
    target = torch.linalg.vector_norm(reference)
    if float(norm) <= 1e-12 or float(target) <= 1e-12:
        raise RuntimeError("cannot norm-match a degenerate intervention")
    return value * (target / norm)


def intervention_specs_cuda(carrier, subspace_cuda):
    width = carrier.shape[-1]
    white = carrier.float() @ subspace_cuda["channel_inverse_square_root"].T
    flat = white.reshape(ACTIONS_PER_FAMILY, -1)
    grouped = flat.reshape(MAGNITUDE_COUNT, SCHEDULE_COUNT, -1)
    swap_target = torch.flip(grouped, dims=[1]).reshape_as(flat) - flat

    def projected(values, basis):
        return (values @ basis) @ basis.T

    basis_names = {
        "primary_r128_swap": "primary_basis",
        "shuffled_r128_swap": "shuffled_basis",
        "random_r128_00_swap": "random_basis_00",
        "random_r128_01_swap": "random_basis_01",
    }
    primary_basis = subspace_cuda["primary_basis"][:, :PRIMARY_RANK]
    primary = projected(swap_target, primary_basis)
    root = subspace_cuda["channel_square_root"]
    specs = []
    for condition in INTERVENTION_CONDITIONS:
        basis = subspace_cuda[basis_names[condition]][:, :PRIMARY_RANK]
        delta = projected(swap_target, basis)
        if condition != "primary_r128_swap":
            delta = matched_tensor_norm(delta, primary)
        specs.append({
            "condition": condition,
            "family": (
                "primary" if condition == "primary_r128_swap"
                else "matched_shuffled_control" if condition == "shuffled_r128_swap"
                else "empirical_span_random_control"
            ),
            "edit_norm": float(torch.linalg.vector_norm(delta).cpu()),
            "delta_native": delta.reshape(ACTIONS_PER_FAMILY, 256, width) @ root.T,
        })
    return specs


def baseline_alignment_rows(
    record, short, family_index,
    predicted_visual, target_visual, predicted_proprio, target_proprio,
):
    prediction = planner_metric_features(
        predicted_visual, predicted_proprio, OFFICIAL_PROPRIO_ALPHA
    ).reshape(MAGNITUDE_COUNT, SCHEDULE_COUNT, -1)
    target = planner_metric_features(
        target_visual, target_proprio, OFFICIAL_PROPRIO_ALPHA
    ).reshape(MAGNITUDE_COUNT, SCHEDULE_COUNT, -1)
    rows = []
    for magnitude_index in range(MAGNITUDE_COUNT):
        metrics = vector_alignment(
            prediction[magnitude_index], target[magnitude_index]
        )
        rows.append({
            "model": short, "record_id": int(record["record_id"]),
            "trajectory_id": int(record["trajectory_id"]),
            "family_index": int(family_index),
            "family_name": ACTION_FAMILIES[family_index]["name"],
            "family_angle": float(ACTION_FAMILIES[family_index]["angle_pair_degrees"][1]),
            "magnitude_index": int(magnitude_index),
            "magnitude": float(SELECTED_MAGNITUDES[magnitude_index]),
            **{f"native_joint_{key}": value for key, value in metrics.items()},
        })
    return rows


def attach_planning_identity(rows, record, short, family_index):
    return [{
        "model": short, "record_id": int(record["record_id"]),
        "trajectory_id": int(record["trajectory_id"]),
        "family_index": int(family_index),
        "family_name": ACTION_FAMILIES[family_index]["name"],
        "family_angle": float(ACTION_FAMILIES[family_index]["angle_pair_degrees"][1]),
        "condition": "baseline", "objective": "official_visual_plus_proprio",
        "magnitude": float(SELECTED_MAGNITUDES[row["magnitude_index"]]),
        **row,
    } for row in rows]


def run_state_family(bundle, record, family_index, subspace_cuda):
    short = bundle["short"]
    selected_block = EXPECTED_STAGE31_SELECTED_BLOCK[short]
    initial, actions = state_model_inputs(bundle, record, family_index)
    target_visual_tensor, target_proprio_tensor = encode_true_state(
        bundle, record, family_index
    )
    with np.load(truth_path(record)) as payload:
        endpoint_states = payload["endpoint_states"][family_index].astype(np.float64)
    with torch.inference_mode():
        predicted_visual_tensor, predicted_proprio_tensor, captures = forward_with_carriers(
            bundle, initial, actions, PRIMARY_HORIZON,
            capture_blocks=[selected_block],
        )
    predicted_visual = predicted_visual_tensor.detach().float().cpu().numpy()
    predicted_proprio = predicted_proprio_tensor.detach().float().cpu().numpy()
    target_visual = target_visual_tensor.detach().float().cpu().numpy()
    target_proprio = target_proprio_tensor.detach().float().cpu().numpy()
    carrier = layer_tokens_full(
        captures[selected_block], ACTIONS_PER_FAMILY, bundle["carrier_width"]
    ).detach()
    alignment = baseline_alignment_rows(
        record, short, family_index,
        predicted_visual, target_visual, predicted_proprio, target_proprio,
    )
    planning = attach_planning_identity(
        official_terminal_planning_rows(
            predicted_visual, target_visual, predicted_proprio, target_proprio,
            endpoint_states, MAGNITUDE_COUNT, SCHEDULE_COUNT,
            goal_schedules=PLANNING_GOAL_SCHEDULES,
            alpha=OFFICIAL_PROPRIO_ALPHA,
        ),
        record, short, family_index,
    )
    baseline_joint = planner_metric_features(
        predicted_visual, predicted_proprio, OFFICIAL_PROPRIO_ALPHA
    )
    target_joint = planner_metric_features(
        target_visual, target_proprio, OFFICIAL_PROPRIO_ALPHA
    )
    closure = []
    for spec in intervention_specs_cuda(carrier, subspace_cuda):
        with torch.inference_mode():
            patched_visual_tensor, patched_proprio_tensor, _ = forward_with_carriers(
                bundle, initial, actions, PRIMARY_HORIZON,
                capture_blocks=[selected_block],
                intervention={"block": selected_block, "delta": spec["delta_native"]},
            )
        patched_joint = planner_metric_features(
            patched_visual_tensor.detach().float().cpu().numpy(),
            patched_proprio_tensor.detach().float().cpu().numpy(),
            OFFICIAL_PROPRIO_ALPHA,
        )
        rows = bounded_swap_closure_rows(
            baseline_joint, patched_joint, target_joint,
            MAGNITUDE_COUNT, SCHEDULE_COUNT,
            diagnostic_schedules=DIAGNOSTIC_SCHEDULES,
            minimum_target_energy=MIN_GROUNDED_TARGET_ENERGY,
        )
        for row in rows:
            closure.append({
                "model": short, "record_id": int(record["record_id"]),
                "trajectory_id": int(record["trajectory_id"]),
                "family_index": int(family_index),
                "family_name": ACTION_FAMILIES[family_index]["name"],
                "family_angle": float(ACTION_FAMILIES[family_index]["angle_pair_degrees"][1]),
                "condition": spec["condition"], "control_family": spec["family"],
                "carrier_edit_whitened_norm": spec["edit_norm"],
                "magnitude": float(SELECTED_MAGNITUDES[row["magnitude_index"]]),
                **row,
            })
        del patched_visual_tensor, patched_proprio_tensor, patched_joint
    PROVENANCE_COUNTS[f"{short}_state_family_evaluations"] += 1
    del initial, actions, target_visual_tensor, target_proprio_tensor
    del predicted_visual_tensor, predicted_proprio_tensor, captures, carrier
    gc.collect()
    torch.cuda.empty_cache()
    return alignment, closure, planning


def hook_identity_test(bundle, record):
    selected_block = EXPECTED_STAGE31_SELECTED_BLOCK[bundle["short"]]
    initial, actions = state_model_inputs(bundle, record, 1)
    with torch.inference_mode():
        baseline, _, _ = forward_with_carriers(
            bundle, initial, actions, PRIMARY_HORIZON,
            capture_blocks=[selected_block],
        )
        patched, _, _ = forward_with_carriers(
            bundle, initial, actions, PRIMARY_HORIZON,
            capture_blocks=[selected_block],
            intervention={
                "block": selected_block,
                "delta": torch.zeros(
                    ACTIONS_PER_FAMILY, 256, bundle["carrier_width"],
                    device="cuda", dtype=torch.float32,
                ),
            },
        )
    error = float(torch.max(torch.abs(patched - baseline)).cpu())
    result = {
        "model": bundle["short"], "record_id": int(record["record_id"]),
        "max_abs_error": error, "passed": error <= MAX_ZERO_EDIT_ERROR,
    }
    write_json(OUT / f"hook_identity_test_{bundle['short']}.json", result)
    if not result["passed"]:
        raise RuntimeError(f"zero hook identity failed: {result}")
    return result


BASELINE_ALIGNMENT_ROWS = []
BOUNDED_CLOSURE_ROWS = []
PLANNING_ROWS = []
FORWARD_BENCHMARKS = {}
if not PIPELINE_FAILED:
    try:
        verify_executed_notebook_through(
            "# Evaluate the exact frozen Stage 31 bases with bounded cosine and four controlled swaps."
        )
        started = time.perf_counter()
        for model_name in MODEL_NAMES:
            MODEL_BUNDLE = load_world_model(model_name)
            short = MODEL_BUNDLE["short"]
            _, SUBSPACE_CUDA = load_upstream_subspace_cuda(short)
            hook_identity_test(MODEL_BUNDLE, EVALUATION_RECORDS[0])
            initial, actions = state_model_inputs(MODEL_BUNDLE, EVALUATION_RECORDS[0], 1)
            torch.cuda.synchronize()
            benchmark_started = time.perf_counter()
            with torch.inference_mode():
                forward_with_carriers(
                    MODEL_BUNDLE, initial, actions, PRIMARY_HORIZON,
                    capture_blocks=[EXPECTED_STAGE31_SELECTED_BLOCK[short]],
                )
            torch.cuda.synchronize()
            seconds = time.perf_counter() - benchmark_started
            FORWARD_BENCHMARKS[short] = {
                "seconds_per_24_branch_predictor_batch": seconds,
                "predictor_batches_per_state_family": 1 + INTERVENTION_FORWARDS_PER_FAMILY,
                "state_families": len(EVALUATION_RECORDS) * ACTION_FAMILY_COUNT,
                "estimated_predictor_minutes": seconds * len(EVALUATION_RECORDS)
                * ACTION_FAMILY_COUNT * (1 + INTERVENTION_FORWARDS_PER_FAMILY) / 60.0,
            }
            write_json(OUT / f"forward_benchmark_{short}.json", FORWARD_BENCHMARKS[short])
            del initial, actions
            for record_index, record in enumerate(EVALUATION_RECORDS):
                for family_index in range(ACTION_FAMILY_COUNT):
                    alignment, closure, planning = run_state_family(
                        MODEL_BUNDLE, record, family_index, SUBSPACE_CUDA
                    )
                    BASELINE_ALIGNMENT_ROWS.extend(alignment)
                    BOUNDED_CLOSURE_ROWS.extend(closure)
                    PLANNING_ROWS.extend(planning)
                write_json(OUT / f"evaluation_{short}_progress.json", {
                    "completed_states": record_index + 1,
                    "completed_state_families": (record_index + 1) * ACTION_FAMILY_COUNT,
                    "total_states": len(EVALUATION_RECORDS),
                    "last_record_id": int(record["record_id"]),
                })
            del SUBSPACE_CUDA
            unload_world_model(MODEL_BUNDLE)
            MODEL_BUNDLE = None
        TIMINGS["bounded_cross_model_evaluation_seconds"] = time.perf_counter() - started
        write_csv(EVIDENCE_DIR / "baseline_native_alignment_rows.csv", BASELINE_ALIGNMENT_ROWS)
        write_csv(EVIDENCE_DIR / "bounded_diagnostic_closure_rows.csv", BOUNDED_CLOSURE_ROWS)
        write_csv(EVIDENCE_DIR / "official_terminal_planning_rows.csv", PLANNING_ROWS)
        write_json(OUT / "evaluation_open_certificate.json", {
            "opened": True, "source_identity": SOURCE_IDENTITY,
            "stage31_upstream_certificate_sha256": sha256_file(
                OUT / "stage31_upstream_certificate.json"
            ),
            "fresh_persistent_states": len(EVALUATION_RECORDS),
            "action_families": ACTION_FAMILIES,
            "official_objective": "visual_mse_plus_0.1_proprio_mse",
            "grounded_metric": "bounded_cosine_only",
            "minimum_grounded_target_energy": MIN_GROUNDED_TARGET_ENERGY,
            "coefficient_computed_or_used": False,
            "decoder_reader_gradient_used": False,
        })
        memory_report("stage32_bounded_cross_model_evaluation_complete")
    except Exception:
        if "MODEL_BUNDLE" in globals() and MODEL_BUNDLE is not None:
            unload_world_model(MODEL_BUNDLE)
        record_failure("stage32_bounded_cross_model_evaluation")
'''


decision = r'''# Apply bounded within-model, paired, family-consistency, and placebo-specificity gates.


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


def joined_model_dataset(short):
    alignment = row_map(
        [row for row in BASELINE_ALIGNMENT_ROWS if row["model"] == short],
        ["record_id", "family_index", "magnitude_index"],
    )
    closure = row_map(
        [row for row in BOUNDED_CLOSURE_ROWS if row["model"] == short],
        ["record_id", "family_index", "magnitude_index", "condition"],
    )
    planning = row_map(
        [row for row in PLANNING_ROWS if row["model"] == short],
        ["record_id", "family_index", "magnitude_index", "goal_schedule"],
    )
    task_rows = []
    for key, plan in planning.items():
        record_id, family_index, magnitude_index, goal_schedule = key
        if float(plan["true_cost_spread"]) < MIN_PLANNING_TRUE_COST_SPREAD:
            continue
        primary = closure[(record_id, family_index, magnitude_index, "primary_r128_swap")]
        condition_rows = {
            condition: closure[(record_id, family_index, magnitude_index, condition)]
            for condition in INTERVENTION_CONDITIONS
        }
        if not primary["self_eligible"]:
            continue
        if not all(row["grounded_eligible"] for row in condition_rows.values()):
            continue
        align = alignment[(record_id, family_index, magnitude_index)]
        candidate = {
            "record_id": int(record_id), "family_index": int(family_index),
            "family_name": ACTION_FAMILIES[int(family_index)]["name"],
            "magnitude_index": int(magnitude_index), "goal_schedule": int(goal_schedule),
            "outcome": float(plan["normalized_regret"]),
            "native_joint_normalized_rmse": float(
                align["native_joint_normalized_rmse"]
            ),
            "self_cosine": float(primary["self_cosine"]),
            **{
                f"{condition}_grounded_cosine": float(row["grounded_cosine"])
                for condition, row in condition_rows.items()
            },
        }
        if all(
            np.isfinite(value) for name, value in candidate.items()
            if name not in {"family_name"}
        ):
            task_rows.append(candidate)
    grouped = defaultdict(list)
    for row in task_rows:
        grouped[(row["record_id"], row["family_index"], row["magnitude_index"])].append(row)
    rows = []
    for key, values in sorted(grouped.items()):
        if len(values) != len(PLANNING_GOAL_SCHEDULES):
            continue
        first = values[0]
        rows.append({
            **{name: value for name, value in first.items()
               if name not in {"goal_schedule", "outcome"}},
            "goal_tasks": len(values),
            "outcome": float(np.mean([value["outcome"] for value in values])),
        })
    return rows


def common_model_panels(model_rows):
    key_names = ("record_id", "family_index", "magnitude_index")
    indices = {
        short: {tuple(row[name] for name in key_names): row for row in rows}
        for short, rows in model_rows.items()
    }
    common = sorted(set(indices["jepa"]) & set(indices["dino"]))
    return {
        short: [indices[short][key] for key in common]
        for short in ["jepa", "dino"]
    }, {
        "common_rows": len(common),
        "jepa_rows_dropped": len(indices["jepa"]) - len(common),
        "dino_rows_dropped": len(indices["dino"]) - len(common),
    }


def crossfit_condition(rows, condition, label, paired):
    groups = np.asarray([row["record_id"] for row in rows], dtype=np.int64)
    if paired:
        base = np.asarray([[
            float(row["magnitude_index"] == 1),
            float(row["magnitude_index"] == 2),
            float(row["magnitude_index"] == 3),
            float(row["family_index"] == 0),
            float(row["family_index"] == 2),
            row["difference_native_joint_normalized_rmse"],
            row["difference_self_cosine"],
        ] for row in rows], dtype=np.float64)
        added = np.asarray([[
            row[f"difference_{condition}_grounded_cosine"]
        ] for row in rows], dtype=np.float64)
    else:
        base = np.asarray([[
            float(row["magnitude_index"] == 1),
            float(row["magnitude_index"] == 2),
            float(row["magnitude_index"] == 3),
            float(row["family_index"] == 0),
            float(row["family_index"] == 2),
            row["native_joint_normalized_rmse"],
            row["self_cosine"],
        ] for row in rows], dtype=np.float64)
        added = np.asarray([[
            row[f"{condition}_grounded_cosine"]
        ] for row in rows], dtype=np.float64)
    outcome = np.asarray([row["outcome"] for row in rows], dtype=np.float64)
    result = cross_fitted_incremental_value(
        outcome, groups, base, added,
        folds=ACTIVE_CROSSFIT_FOLDS,
        seed=stable_seed(CROSSFIT_SEED, label), ridge=1e-6,
    )
    improvement = (
        (outcome - result["base_prediction"]) ** 2
        - (outcome - result["grounded_prediction"]) ** 2
    )
    evidence = [{
        **row, "condition_tested": condition,
        "fold": int(result["fold_id"][index]),
        "base_prediction": float(result["base_prediction"][index]),
        "added_prediction": float(result["grounded_prediction"][index]),
        "mse_improvement": float(improvement[index]),
    } for index, row in enumerate(rows)]
    write_csv(EVIDENCE_DIR / f"{label}_{condition}_crossfit_rows.csv", evidence)
    group_rows = [{
        "record_id": int(row["group"]),
        "mse_improvement": float(row["mse_improvement"]),
    } for row in result["group_rows"]]
    ci = bootstrap_state_mean(
        [row["mse_improvement"] for row in group_rows],
        [row["record_id"] for row in group_rows],
        f"{label}_{condition}_improvement",
    )
    return {
        "condition": condition,
        "eligible_states": int(len(np.unique(groups))),
        "state_family_magnitude_rows": len(rows),
        "base_oof_mse": float(result["base_mse"]),
        "added_oof_mse": float(result["grounded_mse"]),
        "relative_oof_mse_improvement": float(result["relative_mse_improvement"]),
        "base_oof_r_squared": float(result["base_oof_r_squared"]),
        "added_oof_r_squared": float(result["grounded_oof_r_squared"]),
        "state_mean_mse_improvement_ci95": ci,
    }, improvement, group_rows


def within_model_gate(short, rows):
    summary, improvement, group_rows = crossfit_condition(
        rows, "primary_r128_swap", f"within_{short}", paired=False
    )
    family_means = {
        ACTION_FAMILIES[family_index]["name"]: float(np.mean(
            improvement[np.asarray([row["family_index"] for row in rows]) == family_index]
        ))
        for family_index in range(ACTION_FAMILY_COUNT)
    }
    summary.update({
        "minimum_relative_improvement": MIN_WITHIN_MODEL_RELATIVE_MSE_IMPROVEMENT,
        "required_states": ACTIVE_MIN_ELIGIBLE_STATES,
        "family_mean_mse_improvements": family_means,
        "passed": bool(
            summary["eligible_states"] >= ACTIVE_MIN_ELIGIBLE_STATES
            and summary["relative_oof_mse_improvement"]
            >= MIN_WITHIN_MODEL_RELATIVE_MSE_IMPROVEMENT
            and (summary["state_mean_mse_improvement_ci95"][0] > 0 if RUN_MODE == "pilot" else True)
        ),
    })
    return summary


def paired_certificate_gate(model_rows):
    feature_names = ["native_joint_normalized_rmse", "self_cosine"] + [
        f"{condition}_grounded_cosine" for condition in INTERVENTION_CONDITIONS
    ]
    paired_rows = paired_model_difference_rows(
        model_rows["jepa"], model_rows["dino"], feature_names
    )
    summaries, improvements = {}, {}
    for condition in INTERVENTION_CONDITIONS:
        summary, improvement, _ = crossfit_condition(
            paired_rows, condition, "paired_dino_minus_jepa", paired=True
        )
        summaries[condition] = summary
        improvements[condition] = improvement
    primary = summaries["primary_r128_swap"]
    groups = np.asarray([row["record_id"] for row in paired_rows], dtype=np.int64)
    control_matrix = np.column_stack([
        improvements[condition] for condition in PLACEBO_CONDITIONS
    ])
    placebo_state_rows = state_placebo_advantage(
        improvements["primary_r128_swap"], control_matrix, groups
    )
    placebo_values = np.asarray([
        row["primary_minus_median_placebo_improvement"]
        for row in placebo_state_rows
    ], dtype=np.float64)
    placebo_ids = np.asarray([
        row["record_id"] for row in placebo_state_rows
    ], dtype=np.int64)
    placebo_ci = bootstrap_state_mean(
        placebo_values, placebo_ids, "paired_primary_minus_placebo"
    )
    write_csv(EVIDENCE_DIR / "paired_placebo_advantage_state_rows.csv", placebo_state_rows)
    family_indices = np.asarray([row["family_index"] for row in paired_rows], dtype=np.int64)
    family_means = {
        ACTION_FAMILIES[index]["name"]: float(np.mean(
            improvements["primary_r128_swap"][family_indices == index]
        ))
        for index in range(ACTION_FAMILY_COUNT)
    }
    effect_gate = bool(
        primary["eligible_states"] >= ACTIVE_MIN_ELIGIBLE_STATES
        and primary["relative_oof_mse_improvement"]
        >= MIN_PAIRED_RELATIVE_MSE_IMPROVEMENT
        and (primary["state_mean_mse_improvement_ci95"][0] > 0 if RUN_MODE == "pilot" else True)
    )
    family_gate = bool(
        all(value > 0 for value in family_means.values())
        if REQUIRE_ALL_FAMILY_MEAN_IMPROVEMENTS_POSITIVE else True
    )
    placebo_gate = bool(
        placebo_ci[0] > 0
        if RUN_MODE == "pilot" and REQUIRE_PLACEBO_ADVANTAGE_CI_POSITIVE
        else np.mean(placebo_values) > 0
    )
    gate = {
        "estimand": "DINO-WM normalized regret minus JEPA-WM normalized regret",
        "grounded_feature": "bounded physically grounded cosine difference",
        "minimum_grounded_target_energy": MIN_GROUNDED_TARGET_ENERGY,
        "coefficient_ratio_used": False,
        "primary_crossfit": primary,
        "placebo_crossfits": {
            condition: summaries[condition] for condition in PLACEBO_CONDITIONS
        },
        "family_mean_primary_mse_improvements": family_means,
        "all_family_means_positive": family_gate,
        "mean_primary_minus_median_placebo_improvement": float(np.mean(placebo_values)),
        "primary_minus_median_placebo_ci95": placebo_ci,
        "effect_gate_passed": effect_gate,
        "family_consistency_gate_passed": family_gate,
        "placebo_specificity_gate_passed": placebo_gate,
        "passed": bool(effect_gate and family_gate and placebo_gate),
    }
    return gate, paired_rows, improvements


def baseline_model_summary(model_rows):
    result = {}
    for short, rows in model_rows.items():
        values = np.asarray([row["outcome"] for row in rows], dtype=np.float64)
        result[short] = {
            "rows": len(rows), "states": len({row["record_id"] for row in rows}),
            "mean_normalized_regret": float(np.mean(values)),
            "median_normalized_regret": float(np.median(values)),
        }
    return result


def energy_eligibility_summary():
    primary = [
        row for row in BOUNDED_CLOSURE_ROWS
        if row["condition"] == "primary_r128_swap"
    ]
    result = {}
    for short in ["jepa", "dino"]:
        rows = [row for row in primary if row["model"] == short]
        result[short] = {
            "rows": len(rows),
            "eligible_rows": int(sum(row["grounded_eligible"] for row in rows)),
            "ineligible_rows": int(sum(not row["grounded_eligible"] for row in rows)),
            "minimum_eligible_target_energy": float(min(
                row["grounded_target_energy"] for row in rows if row["grounded_eligible"]
            )),
            "grounded_cosine_min": float(min(
                row["grounded_cosine"] for row in rows if row["grounded_eligible"]
            )),
            "grounded_cosine_max": float(max(
                row["grounded_cosine"] for row in rows if row["grounded_eligible"]
            )),
        }
    return result


def fresh_run_certificate():
    expected = {
        "screened_states": len(POOL_SPECS),
        "truth_generated": len(EVALUATION_RECORDS),
        "jepa_state_family_evaluations": len(EVALUATION_RECORDS) * ACTION_FAMILY_COUNT,
        "dino_state_family_evaluations": len(EVALUATION_RECORDS) * ACTION_FAMILY_COUNT,
        "cache_hits": 0,
    }
    passed = bool(
        not OUT_PREEXISTED and PROVENANCE_COUNTS == expected
        and SOURCE_IDENTITY.get("confirmation_eligible", False)
        and STAGE31_ARTIFACTS_VALIDATED
    )
    payload = {
        "out_preexisted": bool(OUT_PREEXISTED),
        "observed_counts": dict(PROVENANCE_COUNTS), "expected_counts": expected,
        "source_execution_verified": bool(SOURCE_IDENTITY.get("confirmation_eligible", False)),
        "stage31_artifacts_validated": bool(STAGE31_ARTIFACTS_VALIDATED),
        "passed": passed,
    }
    write_json(OUT / "fresh_run_certificate.json", payload)
    return payload


def make_plots(within, paired_gate, paired_rows, improvements):
    figure, axes = plt.subplots(1, 4, figsize=(20, 4.7))
    primary_rows = [
        row for row in BOUNDED_CLOSURE_ROWS
        if row["condition"] == "primary_r128_swap" and row["grounded_eligible"]
    ]
    for short, color in [("jepa", "#4c78a8"), ("dino", "#f58518")]:
        values = [row["grounded_cosine"] for row in primary_rows if row["model"] == short]
        axes[0].hist(values, bins=30, alpha=0.55, density=True, label=short.upper(), color=color)
    axes[0].set(xlabel="bounded grounded cosine", ylabel="density", title="Energy-eligible closure")
    axes[0].legend()

    axes[1].scatter(
        [row["difference_primary_r128_swap_grounded_cosine"] for row in paired_rows],
        [row["outcome"] for row in paired_rows], alpha=0.22, color="#54a24b",
    )
    axes[1].set(
        xlabel="DINO − JEPA grounded cosine", ylabel="DINO − JEPA regret",
        title="Fresh paired panel",
    )

    labels = ["base", "primary", "shuffled", "random 0", "random 1"]
    primary = paired_gate["primary_crossfit"]
    placebo = paired_gate["placebo_crossfits"]
    values = [
        primary["base_oof_mse"], primary["added_oof_mse"],
        placebo["shuffled_r128_swap"]["added_oof_mse"],
        placebo["random_r128_00_swap"]["added_oof_mse"],
        placebo["random_r128_01_swap"]["added_oof_mse"],
    ]
    axes[2].bar(labels, values, color=["#9d9d9d", "#4c78a8", "#b279a2", "#bab0ac", "#bab0ac"])
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].set(ylabel="out-of-fold MSE", title="Primary versus placebos")

    family_indices = np.asarray([row["family_index"] for row in paired_rows])
    primary_improvement = improvements["primary_r128_swap"]
    axes[3].bar(
        [family["name"] for family in ACTION_FAMILIES],
        [float(np.mean(primary_improvement[family_indices == index]))
         for index in range(ACTION_FAMILY_COUNT)],
        color=["#72b7b2", "#4c78a8", "#e45756"],
    )
    axes[3].axhline(0, color="black", linewidth=0.8)
    axes[3].set(ylabel="mean OOF MSE improvement", title="Action-family consistency")
    figure.tight_layout()
    figure.savefig(PLOT_DIR / "stage32_bounded_confirmation_summary.png", dpi=180)
    plt.close(figure)


DECISION_PAYLOAD = {"status": "INCONCLUSIVE"}
if not PIPELINE_FAILED:
    try:
        verify_executed_notebook_through(
            "# Apply bounded within-model, paired, family-consistency, and placebo-specificity gates."
        )
        RAW_MODEL_DATASETS = {
            short: joined_model_dataset(short) for short in ["jepa", "dino"]
        }
        MODEL_DATASETS, PANEL_COUNTS = common_model_panels(RAW_MODEL_DATASETS)
        WITHIN_MODEL_GATES = {
            short: within_model_gate(short, MODEL_DATASETS[short])
            for short in ["jepa", "dino"]
        }
        PAIRED_GATE, PAIRED_ROWS, PAIRED_IMPROVEMENTS = paired_certificate_gate(
            MODEL_DATASETS
        )
        BASELINE_SUMMARY = baseline_model_summary(MODEL_DATASETS)
        ENERGY_ELIGIBILITY = energy_eligibility_summary()
        FRESH_CERTIFICATE = fresh_run_certificate()
        both_within = all(gate["passed"] for gate in WITHIN_MODEL_GATES.values())
        if RUN_MODE == "smoke":
            candidate_status = "SMOKE_ONLY"
        elif PAIRED_GATE["passed"] and both_within:
            candidate_status = "BOUNDED_CROSS_MODEL_GROUNDED_CLOSURE_CERTIFICATE_CONFIRMED"
        elif PAIRED_GATE["passed"]:
            candidate_status = "PAIRED_CERTIFICATE_WITHOUT_BOTH_WITHIN_MODEL_REPLICATIONS"
        elif PAIRED_GATE["effect_gate_passed"] and not PAIRED_GATE["placebo_specificity_gate_passed"]:
            candidate_status = "PAIRED_SIGNAL_WITHOUT_SUBSPACE_SPECIFICITY"
        elif PAIRED_GATE["effect_gate_passed"] and not PAIRED_GATE["family_consistency_gate_passed"]:
            candidate_status = "PAIRED_SIGNAL_WITHOUT_FULL_FAMILY_CONSISTENCY"
        elif PAIRED_GATE["effect_gate_passed"]:
            candidate_status = "PAIRED_SIGNAL_WITHOUT_FULL_CONFIRMATORY_GATE"
        elif both_within:
            candidate_status = "BOUNDED_WITHIN_MODEL_REPLICATION_ONLY"
        else:
            candidate_status = "BOUNDED_GROUNDED_CLOSURE_NOT_REPLICATED"
        confirmation_eligible = bool(
            SOURCE_IDENTITY.get("confirmation_eligible", False)
            and FRESH_CERTIFICATE["passed"]
        )
        status = (
            candidate_status if RUN_MODE == "smoke" or confirmation_eligible
            else "UNBOUND_NONFRESH_OR_WRONG_UPSTREAM_EXPLORATORY_RESULT"
        )
        DECISION_PAYLOAD = {
            "status": status, "candidate_status": candidate_status,
            "confirmation_eligible": confirmation_eligible,
            "panel_counts": PANEL_COUNTS,
            "within_model_bounded_reliability_gates": WITHIN_MODEL_GATES,
            "paired_bounded_cross_model_certificate_gate": PAIRED_GATE,
            "baseline_model_summary": BASELINE_SUMMARY,
            "energy_eligibility_summary": ENERGY_ELIGIBILITY,
            "fresh_run_certificate": FRESH_CERTIFICATE,
            "estimand_contract": {
                "primary_outcome": "DINO-WM minus JEPA-WM normalized physical regret",
                "grounded_feature": "bounded cosine only",
                "minimum_grounded_target_energy": MIN_GROUNDED_TARGET_ENERGY,
                "coefficient_ratio_computed_or_used": False,
                "action_families": ACTION_FAMILIES,
                "closure_schedules": DIAGNOSTIC_SCHEDULES,
                "planning_goal_schedules": PLANNING_GOAL_SCHEDULES,
                "planner_score": "visual MSE plus 0.1 proprio MSE",
                "inference_crossfit_and_bootstrap_unit": "initial physical state",
                "placebos": PLACEBO_CONDITIONS,
            },
            "claim_boundary": {
                "public_pusht_checkpoints": MODEL_NAMES,
                "one_checkpoint_per_model_family_available": True,
                "pseudo_checkpoints_manufactured": False,
                "one_environment": True,
                "fresh_persistent_contact_states": len(EVALUATION_RECORDS),
                "exact_stage31_bases_reused_without_tuning": True,
                "learned_decoder_probe_or_reader_used": False,
                "jacobian_jvp_vjp_or_gradient_used": False,
                "terminal_exhaustive_planner_not_full_closed_loop_mpc": True,
            },
            "prespecified_next_step_if_confirmed": (
                "closed-loop CEM intervention under the official planner, then environment generalization"
            ),
        }
        write_json(OUT / "stage32_decision.json", DECISION_PAYLOAD)
        make_plots(WITHIN_MODEL_GATES, PAIRED_GATE, PAIRED_ROWS, PAIRED_IMPROVEMENTS)
        print(json.dumps(DECISION_PAYLOAD, indent=2))
    except Exception:
        record_failure("stage32_decision_and_plots")
        DECISION_PAYLOAD = {"status": "INCONCLUSIVE", "failure": FAILURE_MESSAGE}

if not (OUT / "stage32_decision.json").exists():
    write_json(OUT / "stage32_decision.json", DECISION_PAYLOAD)
'''


packaging = STAGE31.packaging.replace(
    "stage31_cross_model_certificate_result_bundle_",
    "stage32_bounded_confirmation_result_bundle_",
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
    model_evaluation,
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
    code(design_and_upstream),
    code(physical_truth),
    code(model_evaluation),
    code(decision),
    code(packaging),
]
for index, cell in enumerate(cells):
    cell["id"] = f"stage32-{index:02d}"

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "L4", "name": TARGET.name, "provenance": []},
        "kernelspec": {
            "display_name": "Python 3", "language": "python", "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
TARGET.write_text(json.dumps(notebook, indent=1) + "\n")
print(f"Wrote {TARGET}")
