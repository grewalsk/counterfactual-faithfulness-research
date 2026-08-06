import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "26_contact_frame_causal_transport.ipynb"
BASE = json.loads((ROOT / "25_causal_kkt_tomography.ipynb").read_text())
NUMERICAL = ROOT.parent / "src/cf_faithfulness/stage26_contact_transport.py"


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


introduction = r'''# Stage 26: contact-frame causal transport

Stage 25 found a genuine held-out normal-impulse readout at predictor block 1,
but its exact minimum-norm erasure transferred only 0.024% of the model's
contact correction.  The tangent label was numerically degenerate.  Together
with the earlier failures of a fixed hybrid gate, fixed downstream operator,
and shared rank-64 completion, this rejects another fixed global contact
coordinate as the main mechanism.

Stage 26 tests a stronger spatial hypothesis.  Contact computation may occupy
a family of causal subspaces that moves with the contact point and normal.  For
contact geometry \(\xi=(c,n)\), the candidate carrier is

\[
\delta h_\ell = U_\ell(\xi)q,
\qquad U_\ell(g\xi)\simeq\rho_\ell(g)U_\ell(\xi).
\]

Every exact simulator contact is therefore mapped into a normal--tangent
coordinate frame.  Gaussian-windowed degree-1 token moments provide one fixed,
low-complexity local chart.  Construction trajectories alone compare this
contact-aligned chart with a world-axis chart across all predictor blocks,
select one layer and ridge penalty, and freeze a rank-at-most-four response
fiber for normal impulse and the ordinary-minus-ghost physical correction.

Sealed evaluation uses only finite forward interventions.  A natural
construction donor with a lower impulse supplies the coordinate value; the
donor-minus-recipient displacement is projected into the frozen fiber and
transported into the recipient's contact frame.  The identical coordinate
change is also applied in the world-axis frame, at the donor's world location,
with a random local direction, with reversed sign, and as a full local-chart
swap.  Evidence requires contact-frame alignment to beat every matched control,
a functioning full-swap intervention assay, the correct reverse sign, exact
ordinary-versus-collision-disabled targets, and fresh source-bound execution.

This notebook does not use Jacobians, JVPs, VJPs, gradient probes, model-weight
updates, or evaluation-selected layers.  It returns
`stage26_contact_transport_result_bundle_<signature>.zip`.
'''


configuration = r'''# SINGLE CONFIGURATION BLOCK
# Required Colab secrets for a source-bound pilot:
# STAGE26_RUN_MODE=pilot
# STAGE26_SOURCE_COMMIT=<full 40-hex commit from the Colab handoff>
# STAGE26_RUN_NONCE=<new unique label, e.g. contact_transport_20260806_a>
RUN_MODE = "smoke"
EXPERIMENT_SOURCE_REF = ""
RUN_NONCE = "smoke"
try:
    from google.colab import userdata as _colab_userdata

    RUN_MODE = str(_colab_userdata.get("STAGE26_RUN_MODE") or RUN_MODE).strip().lower()
    EXPERIMENT_SOURCE_REF = str(
        _colab_userdata.get("STAGE26_SOURCE_COMMIT") or EXPERIMENT_SOURCE_REF
    ).strip()
    RUN_NONCE = str(
        _colab_userdata.get("STAGE26_RUN_NONCE") or RUN_NONCE
    ).strip()
except Exception:
    pass

if RUN_MODE == "pilot":
    if RUN_NONCE in {"", "smoke"}:
        raise ValueError("pilot mode requires a unique STAGE26_RUN_NONCE")
    if not all(value.isalnum() or value in "-_" for value in RUN_NONCE):
        raise ValueError("STAGE26_RUN_NONCE may contain only letters, numbers, '-' and '_'")

MOUNT_DRIVE = True
DOWNLOAD_RESULTS = True
CONTINUE_AFTER_BENCHMARK = True
MAX_ESTIMATED_TOTAL_MINUTES = 120.0
FRESH_RUN_REQUIRED = True

OUTPUT_DIR = "/content/counterfactual_faithfulness_stage26_contact_transport"
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/counterfactual_faithfulness_stage26_contact_transport"
UPSTREAM_STAGE25_DIR = "/content/drive/MyDrive/counterfactual_faithfulness_stage25_causal_kkt"
UPSTREAM_STAGE25_RUN_SUFFIX = "0c557d94ceae"

PROTOCOL_ID = "stage26-contact-frame-causal-transport-v1"
NOTEBOOK_PROTOCOL_SHA256 = "__PROTOCOL_DIGEST__"
EVIDENCE_STATUS = "CONFIRMATORY_ONLY_IF_SOURCE_BOUND_FRESH_UPSTREAM_BOUND_AND_FIBER_FROZEN"
EXPERIMENT_REPOSITORY = "grewalsk/counterfactual-faithfulness-research"
EXPERIMENT_NOTEBOOK_PATH = "notebooks/26_contact_frame_causal_transport.ipynb"
EXPERIMENT_BUILDER_PATH = "notebooks/build_stage26_contact_frame_transport_notebook.py"
EXPERIMENT_NUMERICAL_PATH = "src/cf_faithfulness/stage26_contact_transport.py"

EXPECTED_STAGE25_SOURCE_COMMIT = "b9422f43d94b4fff35746db7e99dc1ca93afab71"
EXPECTED_STAGE25_STATUS = "IMPULSE_READABLE_BUT_NOT_CAUSALLY_USED"
EXPECTED_STAGE25_PROTOCOL_ID = "stage25-causal-kkt-tomography-v1"

SEED = 26101
DESIGN_SEED = 26137
MODEL_NAME = "jepa_wm_pusht"
ENVIRONMENT = "PushT"
FRAMESKIP = 5
PRIMARY_HORIZON = 3
TARGET_STEPS = [PRIMARY_HORIZON]
DISCOVERY_BLOCKS = [0, 1, 2, 3, 4, 5]
ACTIVE_BLOCKS = DISCOVERY_BLOCKS
EXPECTED_CARRIER_CHANNELS = 400

CONSTRUCTION_POOL_TRAJECTORIES = list(range(2500, 2600))
EVALUATION_POOL_TRAJECTORIES = list(range(2700, 2800))
CONSTRUCTION_TRAJECTORY_TARGET = 40
EVALUATION_TRAJECTORY_TARGET = 40
TASK_ID_OFFSET = 12000

ACTIONS_PER_STATE = 13
ACTION_MAGNITUDE = 0.12
ACTION_STEPS = PRIMARY_HORIZON * FRAMESKIP
APPROACH_DISTANCE = 80.0
MIN_ELIGIBLE_CONTACT_BRANCHES = 2
MIN_ELIGIBLE_NONCONTACT_BRANCHES = 2
MIN_CONTACT_IMPULSE_NORM = 1e-4
MIN_CONTACT_POSE_CORRECTION = 1e-4
MIN_VALID_CONTACT_BRANCHES = 40

TOKEN_GRID_SIZE = 16
IMAGE_SIZE = 512.0
CONTACT_RADIUS = 96.0
CONTACT_POLYNOMIAL_DEGREE = 1
CONTACT_BASIS_DIM = 3
FIBER_RANK = 4
RIDGE_PENALTIES = [0.1, 1.0, 10.0, 100.0, 1000.0]
RIDGE_CV_FOLDS = 5
RESPONSE_NAMES = [
    "log1p_normal_impulse", "normal_pose_correction",
    "tangent_pose_correction", "angular_pose_correction",
]
RESPONSE_STD_FLOORS = [1e-6, 1e-4, 1e-4, 1e-5]
PATCH_CONDITIONS = [
    "aligned_fiber", "world_axis_control", "donor_location_control",
    "random_local_control", "reverse_aligned", "full_local_swap",
]
OUTPUT_SKETCH_DIM = 256
EVAL_OUTPUT_SKETCH_SEED = 23183
ENCODER_BATCH_SIZE = 13
BOOTSTRAP_SEED = 26269
BOOTSTRAP_DRAWS = 10000
MAX_ZERO_EDIT_ERROR = 1e-6
MAX_CANONICAL_RECONSTRUCTION_ERROR = 1e-6
TARGET_ENERGY_FLOOR = 1e-10

MIN_CONSTRUCTION_ALIGNED_R2 = 0.10
MIN_ALIGNMENT_ADVANTAGE = 0.05
MIN_ALIGNED_TRANSFER = 0.05
MIN_GAIN_OVER_CONTROL = 0.02
MIN_FULL_SWAP_MOVED_RATIO = 0.05
MIN_NATIVE_MEDIAN_CONTACT = 0.10

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
        "STAGE26_RUN_MODE must contain only smoke or pilot; "
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
assert CONTACT_BASIS_DIM == 3 and CONTACT_POLYNOMIAL_DEGREE == 1
assert not set(CONSTRUCTION_POOL_TRAJECTORIES) & set(EVALUATION_POOL_TRAJECTORIES)
assert len(RESPONSE_NAMES) == len(RESPONSE_STD_FLOORS) == 4
'''
configuration += "\n\nPROTOCOL_CONFIG_KEYS = " + repr(
    assigned_uppercase_names(configuration)
) + "\n"


installation = base_source(2)


setup = base_source(3)
setup = setup.replace("Stage 25", "Stage 26").replace("STAGE25", "STAGE26")
setup = setup.replace("stage25_causal_kkt", "stage26_contact_transport")


analysis_helpers = base_source(4)
analysis_helpers += "\n\n\n" + function_sources(
    NUMERICAL.read_text(),
    [
        "wrap_angle",
        "token_centers",
        "contact_frame_basis",
        "canonical_contact_features",
        "transport_contact_delta",
        "response_coordinates",
        "fit_standardizer",
        "grouped_ridge_oof",
        "r2_components",
        "fit_response_fiber",
        "projected_donor_delta",
        "select_low_response_donor",
        "intervention_transfer_metrics",
    ],
)


model_helpers = base_source(5)
model_helpers = model_helpers.replace("stage25-jepa-wms", "stage26-jepa-wms")
model_helpers = model_helpers.replace("Stage 25 supports PushT only", "Stage 26 supports PushT only")


upstream_import = r'''# Bind the exact Stage 25 negative before opening new data.


def locate_and_verify_stage25():
    root = Path(UPSTREAM_STAGE25_DIR)
    candidate = root / f"pilot_{UPSTREAM_STAGE25_RUN_SUFFIX}"
    required = {
        "source": candidate / "source_identity.json",
        "decision": candidate / "stage25_decision.json",
        "reader": candidate / "subspaces/impulse_reader_freeze.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise RuntimeError(
            "missing complete Stage 25 Drive run; compact ZIP is insufficient: "
            f"{missing}"
        )
    source = json.loads(required["source"].read_text())
    decision = json.loads(required["decision"].read_text())
    reader = json.loads(required["reader"].read_text())
    tangent_std = float(reader.get("target_std", [np.nan, np.nan])[1]) if "target_std" in reader else None
    checks = {
        "source_commit": source.get("resolved_commit") == EXPECTED_STAGE25_SOURCE_COMMIT,
        "source_execution_verified": bool(source.get("confirmation_eligible", False)),
        "protocol": source.get("protocol_id") == EXPECTED_STAGE25_PROTOCOL_ID,
        "decision": decision.get("status") == EXPECTED_STAGE25_STATUS,
        "source_bound_claim": bool(decision.get("source_bound_claim_eligible", False)),
        "normal_reader_heldout": float(
            decision.get("heldout_reader", {}).get("component_r2", [-np.inf])[0]
        ) >= 0.10,
        "causal_erasure_failed": not bool(
            decision.get("gates", {}).get("causal_erasure", True)
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Stage 25 upstream binding failed: {checks}")
    payload = {
        "upstream_run": str(candidate),
        "checks": checks,
        "source_identity": source,
        "decision_status": decision["status"],
        "heldout_reader": decision.get("heldout_reader", {}),
        "erasure_summary": decision.get("intervention_summaries", {}).get("impulse_erase", {}),
        "normal_only_stage26": True,
        "tangent_coordinate_dropped": True,
        "reader_target_std_if_available": reader.get("target_std"),
        "reader_freeze_sha256": sha256_file(required["reader"]),
    }
    write_json(SUBSPACE_DIR / "stage25_upstream_binding.json", payload)
    return payload


if not PIPELINE_FAILED:
    try:
        STAGE25_BINDING = locate_and_verify_stage25()
        print(json.dumps(STAGE25_BINDING, indent=2))
    except Exception:
        record_failure("stage25_upstream_binding")
'''


design = base_source(7)
design = design.replace("Stage 25", "Stage 26").replace("stage25", "stage26")
design = design.replace(
    '"impulse_coordinates": IMPULSE_COORDINATES,\n    "patch_conditions": PATCH_CONDITIONS,',
    '"response_coordinates": RESPONSE_NAMES,\n    "contact_basis": "gaussian_degree1_normal_tangent",\n    "patch_conditions": PATCH_CONDITIONS,',
)


truth_generation = base_source(8)
truth_generation = truth_generation.replace(
    '"normal": np.zeros(2),\n            "coordinates": np.zeros(2),',
    '"normal": np.zeros(2),\n            "contact_point": np.zeros(2),\n            "coordinates": np.zeros(2),',
)
truth_generation = truth_generation.replace(
    'distances = [distance for event in events for distance in event["distances"]]\n    return {',
    '''distances = [distance for event in events for distance in event["distances"]]
    contact_points, contact_weights = [], []
    for event, weight in zip(events, weights):
        for point in event["world_points"]:
            contact_points.append(np.asarray(point, dtype=np.float64))
            contact_weights.append(float(weight))
    contact_point = (
        np.average(np.stack(contact_points), axis=0, weights=np.asarray(contact_weights))
        if contact_points else np.asarray(normal_state[2:4], dtype=np.float64)
    )
    return {''',
)
truth_generation = truth_generation.replace(
    '"normal": normal,\n        "coordinates":',
    '"normal": normal,\n        "contact_point": contact_point,\n        "coordinates":',
)
truth_generation = truth_generation.replace(
    'impulses, normals, coordinates = [], [], []',
    'impulses, normals, contact_points, coordinates = [], [], [], []',
)
truth_generation = truth_generation.replace(
    'normals.append(trace["normal"])\n            coordinates.append(trace["coordinates"])',
    'normals.append(trace["normal"])\n            contact_points.append(trace["contact_point"])\n            coordinates.append(trace["coordinates"])',
)
truth_generation = truth_generation.replace(
    'contact_normals=np.asarray(normals, dtype=np.float64),\n            impulse_coordinates=',
    'contact_normals=np.asarray(normals, dtype=np.float64),\n            contact_points=np.asarray(contact_points, dtype=np.float64),\n            impulse_coordinates=',
)
truth_generation = truth_generation.replace("Stage 25", "Stage 26").replace("stage25", "stage26")


construction_features = r'''# Load JEPA-WM and open construction activations at all six blocks only.


TOKEN_POSITIONS = token_centers(TOKEN_GRID_SIZE, IMAGE_SIZE)


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


def contact_mask_from_truth(truth):
    events = truth["contact_event_counts"].astype(np.int64)
    impulses = truth["contact_impulses"].astype(np.float64)
    normal_states = truth["normal_endpoint_states"].astype(np.float64)
    ghost_states = truth["ghost_endpoint_states"].astype(np.float64)
    pose_correction = np.linalg.norm(
        pose_target(normal_states) - pose_target(ghost_states), axis=1
    )
    return (
        (events > 0)
        & (np.linalg.norm(impulses, axis=1) >= MIN_CONTACT_IMPULSE_NORM)
        & (pose_correction >= MIN_CONTACT_POSE_CORRECTION)
    )


def branch_descriptor(truth, action_index):
    state = truth["state"].astype(np.float64)
    point = truth["contact_points"][action_index].astype(np.float64)
    normal = truth["contact_normals"][action_index].astype(np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    tangent = np.asarray([-normal[1], normal[0]], dtype=np.float64)
    action = truth["selected_actions"][action_index].astype(np.float64).mean(axis=0)
    ghost = truth["ghost_endpoint_states"][action_index].astype(np.float64)
    block = state[2:4]
    agent_relative = state[:2] - block
    contact_relative = point - block
    ghost_displacement = ghost[2:4] - block
    return np.asarray(
        [
            contact_relative @ normal / CONTACT_RADIUS,
            contact_relative @ tangent / CONTACT_RADIUS,
            agent_relative @ normal / APPROACH_DISTANCE,
            agent_relative @ tangent / APPROACH_DISTANCE,
            action @ normal / ACTION_MAGNITUDE,
            action @ tangent / ACTION_MAGNITUDE,
            ghost_displacement @ normal / CONTACT_RADIUS,
            ghost_displacement @ tangent / CONTACT_RADIUS,
            float(wrap_angle(ghost[4] - state[4])),
        ],
        dtype=np.float64,
    )


def physical_rows_from_truth(truth):
    contact_mask = contact_mask_from_truth(truth)
    responses = np.zeros((ACTIONS_PER_STATE, len(RESPONSE_NAMES)), dtype=np.float64)
    descriptors = np.zeros((ACTIONS_PER_STATE, 9), dtype=np.float64)
    for action_index in np.flatnonzero(contact_mask):
        responses[action_index] = response_coordinates(
            truth["normal_endpoint_states"][action_index],
            truth["ghost_endpoint_states"][action_index],
            truth["contact_impulses"][action_index],
            truth["contact_normals"][action_index],
        )
        descriptors[action_index] = branch_descriptor(truth, action_index)
    return contact_mask, responses, descriptors


def feature_path(record_id):
    return BASELINE_DIR / f"construction_features_{int(record_id):04d}.npz"


def capture_contact_features(captures, truth, contact_mask):
    feature_dim = CONTACT_BASIS_DIM * EXPECTED_CARRIER_CHANNELS
    canonical = np.zeros(
        (len(DISCOVERY_BLOCKS), ACTIONS_PER_STATE, feature_dim), dtype=np.float32
    )
    world_axis = np.zeros_like(canonical)
    for block_position, block_index in enumerate(DISCOVERY_BLOCKS):
        carrier = layer_tokens_full(captures[block_index]).float().cpu().numpy()
        for action_index in np.flatnonzero(contact_mask):
            point = truth["contact_points"][action_index]
            normal = truth["contact_normals"][action_index]
            aligned_basis = contact_frame_basis(
                TOKEN_POSITIONS, point, normal,
                radius=CONTACT_RADIUS,
                polynomial_degree=CONTACT_POLYNOMIAL_DEGREE,
            )
            world_basis = contact_frame_basis(
                TOKEN_POSITIONS, point, [1.0, 0.0],
                radius=CONTACT_RADIUS,
                polynomial_degree=CONTACT_POLYNOMIAL_DEGREE,
            )
            canonical[block_position, action_index] = canonical_contact_features(
                carrier[action_index], aligned_basis
            )
            world_axis[block_position, action_index] = canonical_contact_features(
                carrier[action_index], world_basis
            )
    return canonical, world_axis


def extract_construction_features(records):
    started = time.perf_counter()
    for index, record in enumerate(records):
        destination = feature_path(record["record_id"])
        if destination.exists():
            raise RuntimeError(f"fresh construction feature shard exists: {destination}")
        with np.load(branch_path(record["record_id"])) as payload:
            truth = {name: payload[name].copy() for name in payload.files}
        contact_mask, responses, descriptors = physical_rows_from_truth(truth)
        initial, actions = state_model_inputs(record["record_id"])
        with torch.inference_mode():
            _, _, captures = forward_with_carriers(
                initial, actions, PRIMARY_HORIZON, capture_blocks=DISCOVERY_BLOCKS
            )
        canonical, world_axis = capture_contact_features(captures, truth, contact_mask)
        atomic_npz(
            destination,
            record_id=np.asarray(record["record_id"], dtype=np.int64),
            contact_mask=contact_mask,
            canonical_features=canonical,
            world_axis_features=world_axis,
            responses=responses,
            descriptors=descriptors,
            contact_points=truth["contact_points"],
            contact_normals=truth["contact_normals"],
        )
        PROVENANCE_COUNTS["carrier_baseline_generated"] += 1
        write_json(
            OUT / "construction_features_progress.json",
            {"completed": index + 1, "total": len(records), "last_record_id": int(record["record_id"])},
        )
        del initial, actions, captures
        gc.collect()
        torch.cuda.empty_cache()
    TIMINGS["construction_features_seconds"] = time.perf_counter() - started


if not PIPELINE_FAILED:
    try:
        MODEL, PREPROCESSOR, PREDICTOR, PREDICTOR_BLOCK_MODULES = load_frozen_model()
        EVAL_OUTPUT_PROJECTOR = CountSketchProjector(
            256 * 384, OUTPUT_SKETCH_DIM, EVAL_OUTPUT_SKETCH_SEED
        )
        PHYSICAL_POSE_DECODER = physical_pose_decoder()
        extract_construction_features(CONSTRUCTION_RECORDS)
        memory_report("construction_contact_features_complete")
    except Exception:
        record_failure("construction_contact_features")
'''


fiber_freeze = r'''# Select one layer and freeze the construction-only contact response fiber.


def load_construction_feature_shard(record_id):
    with np.load(feature_path(record_id)) as payload:
        return {name: payload[name].copy() for name in payload.files}


def construction_matrices(block_position):
    canonical, world_axis, targets, groups = [], [], [], []
    descriptors, points, normals = [], [], []
    record_ids, action_indices = [], []
    for record in CONSTRUCTION_RECORDS:
        payload = load_construction_feature_shard(record["record_id"])
        for action_index in np.flatnonzero(payload["contact_mask"]):
            canonical.append(payload["canonical_features"][block_position, action_index])
            world_axis.append(payload["world_axis_features"][block_position, action_index])
            targets.append(payload["responses"][action_index])
            descriptors.append(payload["descriptors"][action_index])
            points.append(payload["contact_points"][action_index])
            normals.append(payload["contact_normals"][action_index])
            groups.append(int(record["record_id"]))
            record_ids.append(int(record["record_id"]))
            action_indices.append(int(action_index))
    return {
        "canonical": np.asarray(canonical, dtype=np.float64),
        "world_axis": np.asarray(world_axis, dtype=np.float64),
        "targets": np.asarray(targets, dtype=np.float64),
        "descriptors": np.asarray(descriptors, dtype=np.float64),
        "points": np.asarray(points, dtype=np.float64),
        "normals": np.asarray(normals, dtype=np.float64),
        "groups": np.asarray(groups, dtype=np.int64),
        "record_ids": np.asarray(record_ids, dtype=np.int64),
        "action_indices": np.asarray(action_indices, dtype=np.int64),
    }


def active_response_mask(targets):
    standard_deviation = np.std(np.asarray(targets), axis=0, ddof=1)
    active = standard_deviation > np.asarray(RESPONSE_STD_FLOORS)
    active[0] = True
    return active, standard_deviation


def selected_r2(targets, predictions, active):
    result = r2_components(targets[:, active], predictions[:, active])
    return result["mean_r2"], result["component_r2"]


def fit_and_freeze_contact_fiber():
    rows = []
    candidates = []
    matrices_by_block = {}
    for block_position, block_index in enumerate(DISCOVERY_BLOCKS):
        matrices = construction_matrices(block_position)
        matrices_by_block[block_index] = matrices
        active, response_std = active_response_mask(matrices["targets"])
        for penalty in RIDGE_PENALTIES:
            canonical_oof = grouped_ridge_oof(
                matrices["canonical"], matrices["targets"], matrices["groups"],
                penalty, RIDGE_CV_FOLDS,
            )
            world_oof = grouped_ridge_oof(
                matrices["world_axis"], matrices["targets"], matrices["groups"],
                penalty, RIDGE_CV_FOLDS,
            )
            canonical_r2, canonical_components = selected_r2(
                matrices["targets"], canonical_oof, active
            )
            world_r2, world_components = selected_r2(
                matrices["targets"], world_oof, active
            )
            row = {
                "block": int(block_index),
                "penalty": float(penalty),
                "canonical_mean_r2": canonical_r2,
                "world_axis_mean_r2": world_r2,
                "alignment_advantage": canonical_r2 - world_r2,
                "canonical_component_r2": canonical_components.tolist(),
                "world_axis_component_r2": world_components.tolist(),
                "active_response_mask": active.tolist(),
                "response_std": response_std.tolist(),
                "contact_rows": int(len(matrices["targets"])),
            }
            rows.append(row)
            candidates.append(row)
    chosen = max(
        candidates,
        key=lambda row: (
            min(
                row["canonical_mean_r2"] - MIN_CONSTRUCTION_ALIGNED_R2,
                row["alignment_advantage"] - MIN_ALIGNMENT_ADVANTAGE,
            ),
            row["canonical_mean_r2"], row["alignment_advantage"],
            -row["block"], -row["penalty"],
        ),
    )
    selected_block = int(chosen["block"])
    matrices = matrices_by_block[selected_block]
    active = np.asarray(chosen["active_response_mask"], dtype=bool)
    model = fit_response_fiber(
        matrices["canonical"], matrices["targets"][:, active],
        chosen["penalty"], min(FIBER_RANK, int(np.sum(active))),
    )
    descriptor_mean, descriptor_scale = fit_standardizer(matrices["descriptors"])
    atomic_npz(
        SUBSPACE_DIR / "frozen_contact_response_fiber.npz",
        selected_block=np.asarray(selected_block, dtype=np.int64),
        x_mean=model["x_mean"], x_scale=model["x_scale"],
        y_mean=model["y_mean"], y_scale=model["y_scale"],
        weight=model["weight"], fiber=model["fiber"],
        singular_values=model["singular_values"],
        ridge_penalty=np.asarray(model["penalty"]),
        active_response_mask=active,
        donor_features=matrices["canonical"],
        donor_responses=matrices["targets"],
        donor_descriptors=matrices["descriptors"],
        donor_points=matrices["points"], donor_normals=matrices["normals"],
        donor_record_ids=matrices["record_ids"],
        donor_action_indices=matrices["action_indices"],
        descriptor_mean=descriptor_mean, descriptor_scale=descriptor_scale,
    )
    summary = {
        "frozen_before_evaluation_activations": True,
        "construction_only": True,
        "evaluation_activation_ids_seen": [],
        "selected_block": selected_block,
        "selected_penalty": float(chosen["penalty"]),
        "fiber_rank": int(model["rank"]),
        "canonical_mean_r2": float(chosen["canonical_mean_r2"]),
        "world_axis_mean_r2": float(chosen["world_axis_mean_r2"]),
        "alignment_advantage": float(chosen["alignment_advantage"]),
        "active_response_mask": chosen["active_response_mask"],
        "response_std": chosen["response_std"],
        "normal_only_stage25_cleanup": True,
        "construction_gate_pass": bool(
            chosen["canonical_mean_r2"] >= MIN_CONSTRUCTION_ALIGNED_R2
            and chosen["alignment_advantage"] >= MIN_ALIGNMENT_ADVANTAGE
        ),
        "fiber_sha256": sha256_file(SUBSPACE_DIR / "frozen_contact_response_fiber.npz"),
    }
    write_csv(EVIDENCE_DIR / "construction_layer_selection.csv", rows)
    write_json(SUBSPACE_DIR / "contact_response_fiber_freeze.json", summary)
    return summary


if not PIPELINE_FAILED:
    try:
        CONTACT_FIBER_FREEZE = fit_and_freeze_contact_fiber()
        print(json.dumps(CONTACT_FIBER_FREEZE, indent=2))
    except Exception:
        record_failure("contact_response_fiber_freeze")
'''


evaluation_interventions = r'''# Open sealed evaluation activations and run contact-frame donor interventions.


def load_contact_fiber():
    with np.load(SUBSPACE_DIR / "frozen_contact_response_fiber.npz") as payload:
        return {name: payload[name].copy() for name in payload.files}


def fiber_model(payload):
    return {
        "x_mean": payload["x_mean"], "x_scale": payload["x_scale"],
        "y_mean": payload["y_mean"], "y_scale": payload["y_scale"],
        "weight": payload["weight"], "fiber": payload["fiber"],
        "singular_values": payload["singular_values"],
        "penalty": float(payload["ridge_penalty"]),
    }


def hook_identity_test(record_id, selected_block):
    initial, actions = state_model_inputs(record_id)
    with torch.inference_mode():
        baseline, _, _ = forward_with_carriers(
            initial, actions, PRIMARY_HORIZON, capture_blocks=[selected_block]
        )
        patched, _, _ = forward_with_carriers(
            initial, actions, PRIMARY_HORIZON, capture_blocks=[selected_block],
            intervention={
                "block": selected_block,
                "delta": torch.zeros(
                    ACTIONS_PER_STATE, 256, EXPECTED_CARRIER_CHANNELS,
                    device="cuda", dtype=torch.float32,
                ),
            },
        )
    error = float(torch.max(torch.abs(patched - baseline)).cpu())
    result = {
        "record_id": int(record_id), "selected_block": int(selected_block),
        "max_abs_error": error, "passed": error <= MAX_ZERO_EDIT_ERROR,
    }
    if not result["passed"]:
        raise RuntimeError(f"zero intervention changed output: {result}")
    write_json(OUT / "hook_identity_test.json", result)
    return result


def forward_benchmark(record_id, selected_block):
    initial, actions = state_model_inputs(record_id)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        forward_with_carriers(
            initial, actions, PRIMARY_HORIZON, capture_blocks=[selected_block]
        )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    estimated = seconds * ACTIVE_EVALUATION_TARGET * (1 + len(PATCH_CONDITIONS)) / 60.0
    result = {
        "seconds_per_candidate_batch": seconds,
        "evaluation_records": ACTIVE_EVALUATION_TARGET,
        "candidate_batches_per_record": 1 + len(PATCH_CONDITIONS),
        "estimated_evaluation_minutes": estimated,
        "warning_threshold_minutes": MAX_ESTIMATED_TOTAL_MINUTES,
    }
    write_json(OUT / "forward_benchmark.json", result)
    if estimated > MAX_ESTIMATED_TOTAL_MINUTES and not CONTINUE_AFTER_BENCHMARK:
        raise RuntimeError("measured estimate exceeds configured credit guard")
    return result


def norm_matched_random_local_delta(target_field, basis, model, seed):
    rng = np.random.default_rng(int(seed))
    fiber = np.asarray(model["fiber"], dtype=np.float64)
    standard = rng.normal(size=fiber.shape[0])
    standard -= fiber @ (fiber.T @ standard)
    feature = standard * np.asarray(model["x_scale"], dtype=np.float64)
    field = transport_contact_delta(feature, basis, EXPECTED_CARRIER_CHANNELS)
    field_norm = float(np.linalg.norm(field))
    target_norm = float(np.linalg.norm(target_field))
    if field_norm <= 1e-12:
        raise RuntimeError("random local control is degenerate")
    return field * (target_norm / field_norm)


def make_record_transport_edits(record, carrier, truth, payload):
    model = fiber_model(payload)
    contact_mask, responses, descriptors = physical_rows_from_truth(truth)
    edits = {
        condition: np.zeros_like(carrier, dtype=np.float64)
        for condition in PATCH_CONDITIONS
    }
    diagnostics = []
    donor_features = payload["donor_features"].astype(np.float64)
    donor_responses = payload["donor_responses"].astype(np.float64)
    donor_descriptors = payload["donor_descriptors"].astype(np.float64)
    donor_points = payload["donor_points"].astype(np.float64)
    donor_normals = payload["donor_normals"].astype(np.float64)
    for action_index in np.flatnonzero(contact_mask):
        point = truth["contact_points"][action_index].astype(np.float64)
        normal = truth["contact_normals"][action_index].astype(np.float64)
        aligned_basis = contact_frame_basis(
            TOKEN_POSITIONS, point, normal,
            radius=CONTACT_RADIUS, polynomial_degree=CONTACT_POLYNOMIAL_DEGREE,
        )
        world_basis = contact_frame_basis(
            TOKEN_POSITIONS, point, [1.0, 0.0],
            radius=CONTACT_RADIUS, polynomial_degree=CONTACT_POLYNOMIAL_DEGREE,
        )
        recipient_feature = canonical_contact_features(
            carrier[action_index], aligned_basis
        )
        recipient_matching_response = responses[action_index].copy()
        recipient_matching_response[0] = np.expm1(recipient_matching_response[0])
        donor_matching_responses = donor_responses.copy()
        donor_matching_responses[:, 0] = np.expm1(donor_matching_responses[:, 0])
        donor_index, donor_distance = select_low_response_donor(
            descriptors[action_index], recipient_matching_response,
            donor_descriptors, donor_matching_responses,
        )
        donor_feature = donor_features[donor_index]
        feature_delta = projected_donor_delta(model, recipient_feature, donor_feature)
        aligned = transport_contact_delta(
            feature_delta, aligned_basis, EXPECTED_CARRIER_CHANNELS
        )
        if np.linalg.norm(aligned) <= 1e-10:
            continue
        donor_basis = contact_frame_basis(
            TOKEN_POSITIONS, donor_points[donor_index], donor_normals[donor_index],
            radius=CONTACT_RADIUS, polynomial_degree=CONTACT_POLYNOMIAL_DEGREE,
        )
        edits["aligned_fiber"][action_index] = aligned
        edits["world_axis_control"][action_index] = transport_contact_delta(
            feature_delta, world_basis, EXPECTED_CARRIER_CHANNELS
        )
        edits["donor_location_control"][action_index] = transport_contact_delta(
            feature_delta, donor_basis, EXPECTED_CARRIER_CHANNELS
        )
        edits["random_local_control"][action_index] = norm_matched_random_local_delta(
            aligned, aligned_basis, model,
            stable_seed(SEED, record["record_id"], action_index, "random_local"),
        )
        edits["reverse_aligned"][action_index] = -aligned
        full_feature_delta = donor_feature - recipient_feature
        edits["full_local_swap"][action_index] = transport_contact_delta(
            full_feature_delta, aligned_basis, EXPECTED_CARRIER_CHANNELS
        )
        reconstruction_error = float(np.linalg.norm(
            canonical_contact_features(aligned, aligned_basis) - feature_delta
        ))
        if reconstruction_error > MAX_CANONICAL_RECONSTRUCTION_ERROR:
            raise RuntimeError(
                f"contact-frame edit reconstruction failed: {reconstruction_error}"
            )
        recipient_impulse = float(np.expm1(responses[action_index, 0]))
        donor_impulse = float(np.expm1(donor_responses[donor_index, 0]))
        dose = float(np.clip(
            1.0 - donor_impulse / max(recipient_impulse, 1e-12),
            0.0, 1.0,
        ))
        diagnostics.append(
            {
                "record_id": int(record["record_id"]),
                "action_index": int(action_index),
                "donor_index": int(donor_index),
                "donor_record_id": int(payload["donor_record_ids"][donor_index]),
                "donor_action_index": int(payload["donor_action_indices"][donor_index]),
                "recipient_log_impulse": float(responses[action_index, 0]),
                "donor_log_impulse": float(donor_responses[donor_index, 0]),
                "recipient_normal_impulse": recipient_impulse,
                "donor_normal_impulse": donor_impulse,
                "erasure_dose": dose,
                "donor_descriptor_distance": donor_distance,
                "aligned_edit_norm": float(np.linalg.norm(aligned)),
                "full_local_swap_norm": float(
                    np.linalg.norm(edits["full_local_swap"][action_index])
                ),
                "canonical_reconstruction_error": reconstruction_error,
            }
        )
    return edits, diagnostics, contact_mask


def patched_prediction(initial, actions, selected_block, delta):
    tensor = torch.as_tensor(delta, device="cuda", dtype=torch.float32)
    with torch.inference_mode():
        predicted, _, _ = forward_with_carriers(
            initial, actions, PRIMARY_HORIZON, capture_blocks=[selected_block],
            intervention={"block": selected_block, "delta": tensor},
        )
        output = EVAL_OUTPUT_PROJECTOR(predicted).cpu().numpy().astype(np.float64)
        pose = PHYSICAL_POSE_DECODER(predicted).cpu().numpy().astype(np.float64)
    PROVENANCE_COUNTS["patched_forwards_generated"] += 1
    return output, pose


def evaluate_transport_record(record, payload):
    selected_block = int(payload["selected_block"])
    with np.load(branch_path(record["record_id"])) as source:
        truth = {name: source[name].copy() for name in source.files}
    initial, actions = state_model_inputs(record["record_id"])
    with torch.inference_mode():
        predicted, _, captures = forward_with_carriers(
            initial, actions, PRIMARY_HORIZON, capture_blocks=[selected_block]
        )
        baseline_output = EVAL_OUTPUT_PROJECTOR(predicted).cpu().numpy().astype(np.float64)
        baseline_pose = PHYSICAL_POSE_DECODER(predicted).cpu().numpy().astype(np.float64)
        carrier = layer_tokens_full(captures[selected_block]).float().cpu().numpy()
    normal_targets, ghost_targets = endpoint_target_sketches(record["record_id"])
    normal_pose = pose_target(truth["normal_endpoint_states"])
    ghost_pose = pose_target(truth["ghost_endpoint_states"])
    edits, diagnostics, contact_mask = make_record_transport_edits(
        record, carrier, truth, payload
    )
    patched = {
        condition: patched_prediction(
            initial, actions, selected_block, edits[condition]
        )
        for condition in PATCH_CONDITIONS
    }
    diagnostic_by_action = {row["action_index"]: row for row in diagnostics}
    rows = []
    for action_index in np.flatnonzero(contact_mask):
        if action_index not in diagnostic_by_action:
            continue
        diagnostic = diagnostic_by_action[action_index]
        dose = diagnostic["erasure_dose"]
        desired_output = dose * (
            ghost_targets[action_index].astype(np.float64) - baseline_output[action_index]
        )
        desired_pose = dose * (
            ghost_pose[action_index].astype(np.float64) - baseline_pose[action_index]
        )
        if (
            np.sum((
                normal_targets[action_index].astype(np.float64)
                - ghost_targets[action_index].astype(np.float64)
            ) ** 2) <= TARGET_ENERGY_FLOOR
            or
            np.sum(desired_output**2) <= TARGET_ENERGY_FLOOR
            or np.sum(desired_pose**2) <= TARGET_ENERGY_FLOOR
        ):
            continue
        native = contact_projection_metrics(
            baseline_output[action_index],
            normal_targets[action_index].astype(np.float64),
            ghost_targets[action_index].astype(np.float64),
        )
        for condition in PATCH_CONDITIONS:
            output, pose = patched[condition]
            output_metrics = intervention_transfer_metrics(
                baseline_output[action_index], output[action_index], desired_output
            )
            pose_metrics = intervention_transfer_metrics(
                baseline_pose[action_index], pose[action_index], desired_pose
            )
            rows.append(
                {
                    "record_id": int(record["record_id"]),
                    "trajectory_id": int(record["trajectory_id"]),
                    "action_index": int(action_index),
                    "condition": condition,
                    "native_contact_coefficient": native["contact_coefficient"],
                    "native_contact_preference": native["contact_preference"],
                    "output_transfer_coefficient": output_metrics["transfer_coefficient"],
                    "output_transfer_cosine": output_metrics["transfer_cosine"],
                    "output_orthogonal_residual_ratio": output_metrics["orthogonal_residual_ratio"],
                    "output_moved_ratio": output_metrics["moved_norm"] / output_metrics["desired_norm"],
                    "pose_transfer_coefficient": pose_metrics["transfer_coefficient"],
                    "pose_transfer_cosine": pose_metrics["transfer_cosine"],
                    "pose_moved_ratio": pose_metrics["moved_norm"] / pose_metrics["desired_norm"],
                    "erasure_dose": dose,
                    "donor_descriptor_distance": diagnostic["donor_descriptor_distance"],
                }
            )
    return rows, diagnostics


if not PIPELINE_FAILED:
    try:
        freeze = json.loads((SUBSPACE_DIR / "contact_response_fiber_freeze.json").read_text())
        if not freeze.get("frozen_before_evaluation_activations", False):
            raise RuntimeError("contact fiber was not frozen before evaluation opened")
        payload = load_contact_fiber()
        selected_block = int(payload["selected_block"])
        HOOK_IDENTITY = hook_identity_test(
            EVALUATION_RECORDS[0]["record_id"], selected_block
        )
        FORWARD_BENCHMARK = forward_benchmark(
            EVALUATION_RECORDS[0]["record_id"], selected_block
        )
        INTERVENTION_ROWS, EDIT_DIAGNOSTICS = [], []
        started = time.perf_counter()
        if RUN_MODE == "smoke" or freeze.get("construction_gate_pass", False):
            for index, record in enumerate(EVALUATION_RECORDS):
                rows, diagnostics = evaluate_transport_record(record, payload)
                INTERVENTION_ROWS.extend(rows)
                EDIT_DIAGNOSTICS.extend(diagnostics)
                write_json(
                    OUT / "intervention_progress.json",
                    {"completed": index + 1, "total": len(EVALUATION_RECORDS), "last_record_id": int(record["record_id"])},
                )
        else:
            log.warning("Construction contact-frame gate failed; evaluation interventions skipped")
        TIMINGS["evaluation_interventions_seconds"] = time.perf_counter() - started
        write_csv(EVIDENCE_DIR / "contact_transport_intervention_rows.csv", INTERVENTION_ROWS)
        write_csv(EVIDENCE_DIR / "contact_transport_edit_diagnostics.csv", EDIT_DIAGNOSTICS)
        memory_report("contact_transport_evaluation_complete")
    except Exception:
        record_failure("contact_transport_evaluation")
'''


decision = r'''# Apply preregistered Stage 26 contact-frame causal-transport gates.


def bootstrap_summary(values, groups, seed):
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    group_means = np.asarray([
        np.mean(values[groups == group]) for group in unique_groups
    ], dtype=np.float64)
    draws = clustered_bootstrap_mean(
        group_means, np.arange(len(group_means)), ACTIVE_BOOTSTRAP_DRAWS, seed
    )
    ordered = np.sort(values)
    trim = int(np.floor(0.05 * len(ordered)))
    trimmed = ordered[trim:len(ordered) - trim] if trim and len(ordered) > 2 * trim else ordered
    return {
        "mean": float(np.mean(group_means)),
        "median": float(np.median(values)),
        "trimmed_mean_5pct": float(np.mean(trimmed)),
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
        "n": int(len(values)),
        "clusters": int(len(np.unique(groups))),
    }


def rows_for_condition(condition):
    return [row for row in INTERVENTION_ROWS if row["condition"] == condition]


def paired_condition_difference(left, right):
    left_rows = {
        (row["record_id"], row["action_index"]): row for row in rows_for_condition(left)
    }
    right_rows = {
        (row["record_id"], row["action_index"]): row for row in rows_for_condition(right)
    }
    keys = sorted(set(left_rows) & set(right_rows))
    values = np.asarray([
        left_rows[key]["output_transfer_coefficient"]
        - right_rows[key]["output_transfer_coefficient"]
        for key in keys
    ])
    groups = np.asarray([key[0] for key in keys])
    return values, groups


if not PIPELINE_FAILED:
    try:
        verify_executed_notebook_through(
            "# Apply preregistered Stage 26 contact-frame causal-transport gates."
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
    summaries = {}
    gains = {}
    if INTERVENTION_ROWS:
        for condition in PATCH_CONDITIONS:
            selected = rows_for_condition(condition)
            summaries[condition] = bootstrap_summary(
                [row["output_transfer_coefficient"] for row in selected],
                [row["record_id"] for row in selected],
                stable_seed(BOOTSTRAP_SEED, condition),
            )
        for control in [
            "world_axis_control", "donor_location_control", "random_local_control"
        ]:
            values, groups = paired_condition_difference("aligned_fiber", control)
            gains[control] = bootstrap_summary(
                values, groups, stable_seed(BOOTSTRAP_SEED, "gain", control)
            )
        aligned_rows = rows_for_condition("aligned_fiber")
        full_rows = rows_for_condition("full_local_swap")
        reverse_rows = rows_for_condition("reverse_aligned")
        native_values = [row["native_contact_coefficient"] for row in aligned_rows]
        native_summary = bootstrap_summary(
            native_values, [row["record_id"] for row in aligned_rows],
            stable_seed(BOOTSTRAP_SEED, "native"),
        )
        full_swap_moved_ratio = float(np.median([
            row["output_moved_ratio"] for row in full_rows
        ]))
        reverse_mean = float(np.mean([
            row["output_transfer_coefficient"] for row in reverse_rows
        ]))
        max_reconstruction_error = max(
            [row["canonical_reconstruction_error"] for row in EDIT_DIAGNOSTICS]
            or [float("inf")]
        )
    else:
        native_summary = {
            "mean": 0.0, "median": 0.0, "trimmed_mean_5pct": 0.0,
            "lower": 0.0, "upper": 0.0, "n": 0, "clusters": 0,
        }
        full_swap_moved_ratio = 0.0
        reverse_mean = 0.0
        max_reconstruction_error = float("inf")
    aligned_summary = summaries.get("aligned_fiber", {
        "mean": 0.0, "median": 0.0, "lower": -np.inf, "n": 0, "clusters": 0,
    })
    construction_pass = bool(CONTACT_FIBER_FREEZE.get("construction_gate_pass", False))
    controls_pass = bool(
        gains
        and all(
            summary["mean"] >= MIN_GAIN_OVER_CONTROL and summary["lower"] > 0
            for summary in gains.values()
        )
    )
    gates = {
        "stage25_upstream_bound": bool(all(STAGE25_BINDING.get("checks", {}).values())),
        "construction_contact_frame": construction_pass,
        "valid_contact_branches": bool(
            aligned_summary["n"] >= ACTIVE_MIN_VALID_CONTACT_BRANCHES
        ),
        "full_swap_assay_active": bool(
            full_swap_moved_ratio >= MIN_FULL_SWAP_MOVED_RATIO
        ),
        "native_contact_signal": bool(
            native_summary["median"] >= MIN_NATIVE_MEDIAN_CONTACT
        ),
        "aligned_transfer": bool(aligned_summary["lower"] >= MIN_ALIGNED_TRANSFER),
        "beats_all_matched_controls": controls_pass,
        "reverse_sign": bool(reverse_mean < 0.0 < aligned_summary["mean"]),
        "contact_frame_edit_exact": bool(
            max_reconstruction_error <= MAX_CANONICAL_RECONSTRUCTION_ERROR
        ),
        "normal_only_cleanup": True,
    }
    if RUN_MODE != "pilot":
        status = "SMOKE_ONLY"
    elif not gates["construction_contact_frame"]:
        status = "NO_CONTACT_FRAME_RESPONSE_FIELD"
    elif not gates["full_swap_assay_active"]:
        status = "INTERVENTION_ASSAY_INACTIVE"
    elif all(gates.values()):
        status = "CONTACT_FRAME_CAUSAL_TRANSPORT_SUPPORTED"
    else:
        status = "CONTACT_FIELD_READABLE_BUT_NOT_CAUSALLY_TRANSPORTABLE"
    DECISION_PAYLOAD = {
        "status": status,
        "protocol_id": PROTOCOL_ID,
        "run_mode": RUN_MODE,
        "run_signature": RUN_SIGNATURE,
        "source_identity": SOURCE_IDENTITY,
        "source_bound_claim_eligible": bool(
            SOURCE_IDENTITY.get("confirmation_eligible", False)
            and all(STAGE25_BINDING.get("checks", {}).values())
        ),
        "stage25_upstream_bound": bool(all(STAGE25_BINDING.get("checks", {}).values())),
        "contact_response_fiber": CONTACT_FIBER_FREEZE,
        "intervention_summaries": summaries,
        "aligned_minus_controls": gains,
        "native_contact_signal_robust": native_summary,
        "full_swap_median_moved_ratio": full_swap_moved_ratio,
        "reverse_mean": reverse_mean,
        "max_canonical_reconstruction_error": max_reconstruction_error,
        "gates": gates,
        "claim_scope": (
            "A positive result supports a state-conditioned, contact-frame-transportable "
            "causal response fiber in one frozen JEPA-WM/PushT checkpoint. It does not "
            "establish exact SE(2) equivariance, a universal contact solver, or cross-model "
            "generality without new environments and checkpoints."
        ),
        "failure_interpretation": (
            "If the full-swap assay is active but aligned transport fails, close the "
            "low-dimensional contact-mediator thesis and move to path-level distributed "
            "computation rather than another global subspace search."
        ),
    }
write_json(OUT / "stage26_decision.json", DECISION_PAYLOAD)
(OUT / "FAILURE_TRACE.txt").write_text("NONE\n" if not PIPELINE_FAILED else FAILURE_MESSAGE)
write_json(OUT / "timings.json", TIMINGS)
write_json(OUT / "memory.json", MEMORY)


if not PIPELINE_FAILED and INTERVENTION_ROWS:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].bar(
        ["aligned", "world", "location", "random", "reverse", "full"],
        [summaries[name]["mean"] for name in PATCH_CONDITIONS],
    )
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set_ylabel("output transfer toward ghost")
    selection_rows = list(csv.DictReader(
        (EVIDENCE_DIR / "construction_layer_selection.csv").open()
    ))
    for block in DISCOVERY_BLOCKS:
        block_rows = [row for row in selection_rows if int(row["block"]) == block]
        best = max(block_rows, key=lambda row: float(row["canonical_mean_r2"]))
        axes[1].scatter(
            int(block), float(best["canonical_mean_r2"]), color="tab:blue"
        )
        axes[1].scatter(
            int(block), float(best["world_axis_mean_r2"]), color="tab:orange"
        )
    axes[1].set_xlabel("predictor block")
    axes[1].set_ylabel("construction grouped OOF R²")
    axes[1].legend(["contact frame", "world axis"])
    aligned_rows = rows_for_condition("aligned_fiber")
    axes[2].scatter(
        [row["native_contact_coefficient"] for row in aligned_rows],
        [row["output_transfer_coefficient"] for row in aligned_rows],
        alpha=0.6,
    )
    axes[2].set_xlabel("native contact coefficient")
    axes[2].set_ylabel("aligned donor transfer")
    figure.suptitle(DECISION_PAYLOAD["status"])
    figure.tight_layout()
    figure.savefig(PLOT_DIR / "stage26_contact_transport_summary.png", dpi=160)
    plt.show()
print(json.dumps(DECISION_PAYLOAD, indent=2))
'''


packaging = r'''# Package compact audit evidence and download one result bundle.


INCLUDE_PATHS = [
    "config.json", "versions.json", "source_identity.json", "FAILURE_TRACE.txt",
    "stage26_decision.json", "timings.json", "memory.json", "forward_benchmark.json",
    "hook_identity_test.json", "design/design_freeze.json", "design/candidate_pool_manifest.json",
    "design/physical_selection_freeze.json", "subspaces/stage25_upstream_binding.json",
    "subspaces/contact_response_fiber_freeze.json",
    "evaluation_evidence/physical_eligibility_rows.csv",
    "evaluation_evidence/construction_layer_selection.csv",
    "evaluation_evidence/contact_transport_intervention_rows.csv",
    "evaluation_evidence/contact_transport_edit_diagnostics.csv",
    "plots/stage26_contact_transport_summary.png", "logs/run.log",
    "truth_construction_pool_progress.json", "truth_evaluation_pool_progress.json",
    "construction_features_progress.json", "intervention_progress.json",
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

archive_base = OUT / f"stage26_contact_transport_result_bundle_{RUN_SIGNATURE[:12]}"
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
    construction_features,
    fiber_freeze,
    evaluation_interventions,
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
    code(construction_features),
    code(fiber_freeze),
    code(evaluation_interventions),
    code(decision),
    code(packaging),
]
for index, cell in enumerate(cells):
    cell["id"] = f"stage26-{index:02d}"

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
