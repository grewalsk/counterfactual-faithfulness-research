import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "23_causal_mode_manifold_operator_switch.ipynb"
BASE = json.loads((ROOT / "22_latent_hybrid_gate_interaction.ipynb").read_text())
NUMERICAL = ROOT.parent / "src/cf_faithfulness/stage23_mode_operator.py"


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


introduction = r'''# Stage 23: causal mode-manifold operator switch

Stage 22 found a label-free internal state at predictor block 1 that aligned
strongly with physical contact, but a one-dimensional gate did not
multiplicatively modulate its rank-32 effect subspace.  This notebook tests the
stronger and more appropriate hybrid-dynamics hypothesis: **the distributed
mode state selects a different downstream finite transition operator**.

The exact Stage 22 eight-dimensional partition is imported and hash-verified.
On fresh construction states, an output-aligned rank-64 content subspace `U`
and its channel metric are frozen.  For every held-out off/on model-only pair,
the minimum-norm mode transport solves

\[
\min_\delta \|\delta\|_2\quad\text{s.t.}\quad
A^\top\delta=q_{\rm native\ on}-q_{\rm off},\qquad U^\top\delta=0,
\]

where `A` is the exact pullback of the frozen Stage 22 mode coordinates through
candidate centering and CountSketch.  The intervention is invalid unless the
frozen partition certifies an off-to-on reassignment with negligible protected
content leakage.

With the downstream action held fixed to the base candidate, eight frozen
rank-64 content probes measure symmetric finite responses

\[
R_c(u)=\frac{F(h_c+\epsilon u)-F(h_c-\epsilon u)}{2\epsilon}
\]

under five contexts: off baseline, exact native-on activation, learned mode
transport, signed-permuted mode transport, and a norm-matched random tangent.
The preregistered test asks whether the learned edit moves the response
operator toward the native-on operator and beyond both controls.  Independent
sketches at blocks 2--5 and the output localize where the switch emerges.

This is a source-bound, fresh-state, forward-pass-only pilot.  It computes no
Jacobian, JVP, VJP, gradient probe, or weight update.  A pass would establish a
causal mode-conditioned operator in this frozen PushT JEPA-WM—not a complete
hybrid automaton, universal contact variable, or planning algorithm.  Return
`stage23_mode_operator_result_bundle_<signature>.zip`.
'''


configuration = r'''# SINGLE CONFIGURATION BLOCK
# Required Colab secrets for a source-bound pilot:
# STAGE23_RUN_MODE=pilot
# STAGE23_SOURCE_COMMIT=<full 40-hex commit from the Colab handoff>
# STAGE23_RUN_NONCE=<new unique label, e.g. mode_operator_20260805_a>
RUN_MODE = "smoke"
EXPERIMENT_SOURCE_REF = ""
RUN_NONCE = "smoke"
try:
    from google.colab import userdata as _colab_userdata

    RUN_MODE = str(_colab_userdata.get("STAGE23_RUN_MODE") or RUN_MODE).strip().lower()
    EXPERIMENT_SOURCE_REF = str(
        _colab_userdata.get("STAGE23_SOURCE_COMMIT") or EXPERIMENT_SOURCE_REF
    ).strip()
    RUN_NONCE = str(
        _colab_userdata.get("STAGE23_RUN_NONCE") or RUN_NONCE
    ).strip()
except Exception:
    pass

if RUN_MODE == "pilot":
    if RUN_NONCE in {"", "smoke"}:
        raise ValueError("pilot mode requires a unique STAGE23_RUN_NONCE")
    if not all(value.isalnum() or value in "-_" for value in RUN_NONCE):
        raise ValueError("STAGE23_RUN_NONCE may contain only letters, numbers, '-' and '_'")

MOUNT_DRIVE = True
DOWNLOAD_RESULTS = True
CONTINUE_AFTER_BENCHMARK = True
MAX_ESTIMATED_TOTAL_MINUTES = 180.0
FRESH_RUN_REQUIRED = True

OUTPUT_DIR = "/content/counterfactual_faithfulness_stage23_mode_operator"
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/counterfactual_faithfulness_stage23_mode_operator"
UPSTREAM_STAGE22_DIR = "/content/drive/MyDrive/counterfactual_faithfulness_stage22_hybrid_gate"
UPSTREAM_STAGE22_RUN_SUFFIX = "7b0be321cc7d"

PROTOCOL_ID = "stage23-causal-mode-manifold-operator-switch-v1"
NOTEBOOK_PROTOCOL_SHA256 = "__PROTOCOL_DIGEST__"
EVIDENCE_STATUS = "CONFIRMATORY_ONLY_IF_SOURCE_BOUND_FRESH_UPSTREAM_BOUND_AND_EVALUATION_SEALED"
EXPERIMENT_REPOSITORY = "grewalsk/counterfactual-faithfulness-research"
EXPERIMENT_NOTEBOOK_PATH = "notebooks/23_causal_mode_manifold_operator_switch.ipynb"
EXPERIMENT_BUILDER_PATH = "notebooks/build_stage23_mode_operator_notebook.py"
EXPERIMENT_NUMERICAL_PATH = "src/cf_faithfulness/stage23_mode_operator.py"

EXPECTED_STAGE22_SOURCE_COMMIT = "06b4240875248bce5882637bfc3f4f670470bb03"
EXPECTED_STAGE22_STATUS = "PHYSICAL_MODE_WITHOUT_CAUSAL_INTERACTION"
EXPECTED_STAGE22_PROTOCOL_ID = "stage22-label-free-mode-gate-effect-factorial-v1"
EXPECTED_STAGE22_PARTITION_SHA256 = "0f9e376e9d874f1b2429e62b119cf56ea77e3458b8a44c1d4575ed73c988697f"
EXPECTED_STAGE22_SELECTED_BLOCK = 1

SEED = 23101
DESIGN_SEED = 23137
MODEL_NAME = "jepa_wm_pusht"
ENVIRONMENT = "PushT"
FRAMESKIP = 5
PRIMARY_HORIZON = 3
TARGET_STEPS = [PRIMARY_HORIZON]
SELECTED_BLOCK = 1
DISCOVERY_BLOCKS = [SELECTED_BLOCK]
ACTIVE_BLOCKS = DISCOVERY_BLOCKS
DOWNSTREAM_BLOCKS = [2, 3, 4, 5]
EXPECTED_CARRIER_CHANNELS = 400

CONSTRUCTION_POOL_TRAJECTORIES = list(range(1200, 1280))
EVALUATION_POOL_TRAJECTORIES = list(range(1300, 1396))
CONSTRUCTION_TRAJECTORY_TARGET = 48
EVALUATION_TRAJECTORY_TARGET = 64
TASK_ID_OFFSET = 7000

ACTIONS_PER_STATE = 13
ACTION_MAGNITUDE = 0.12
ACTION_STEPS = PRIMARY_HORIZON * FRAMESKIP
APPROACH_DISTANCE = 80.0
MIN_ELIGIBLE_COST_SPREAD = 0.02
MIN_ELIGIBLE_NON_TIED_PAIR_FRACTION = 0.20
MIN_ELIGIBLE_CONTACT_BRANCHES = 2
MIN_ELIGIBLE_NONCONTACT_BRANCHES = 2
PHYSICAL_COST_TIE = 1e-4

OUTPUT_SKETCH_DIM = 256
CARRIER_SKETCH_DIM = 256
RESPONSE_SKETCH_DIM = 128
TRAIN_OUTPUT_SKETCH_SEED = 23161
EVAL_OUTPUT_SKETCH_SEED = 23183
CARRIER_SKETCH_SEED = 22197
RESPONSE_SKETCH_SEED = 23203
CHANNEL_SHRINKAGE = 0.10
CHANNEL_EIGEN_FLOOR = 1e-6
RIDGE_MULTIPLIER = 1e-4
CONTENT_RANK = 64
PROBE_COUNT = 8
FINITE_PROBE_DOSE = 0.5
TRANSPORT_SOLVE_RIDGE = 0.0
PERMUTATION_SEED = 23231
RANDOM_TANGENT_SEED = 23251
BOOTSTRAP_SEED = 23269
BOOTSTRAP_DRAWS = 10000
PAIRS_PER_STATE = 1
CONTEXTS = ["off", "native_on", "mode_transport", "permuted_transport", "random_tangent"]
PATCHED_FORWARDS_PER_PAIR = 81
MAX_ZERO_EDIT_ERROR = 1e-6

MIN_CONTACT_BALANCED_ACCURACY = 0.65
MIN_CONTACT_MCC = 0.30
MIN_PHYSICALLY_ALIGNED_PAIRS = 40
MIN_NATIVE_FULL_SWAP_COEFFICIENT = 0.60
MIN_LEARNED_FLIP_RATE = 0.80
MAX_MODE_COORDINATE_RESIDUAL = 1e-5
MAX_PROTECTED_CONTENT_LEAKAGE = 1e-5
MIN_NONDEGENERATE_TARGET_FRACTION = 0.80
TARGET_ENERGY_FLOOR = 1e-10
MIN_OPERATOR_TRANSFER_COEFFICIENT = 0.35
MIN_GAIN_OVER_RANDOM = 0.15
MIN_GAIN_OVER_PERMUTED = 0.15
REQUIRED_POSITIVE_GAIN_FRACTION = 0.65

if RUN_MODE == "smoke":
    ACTIVE_CONSTRUCTION_POOL_TRAJECTORIES = CONSTRUCTION_POOL_TRAJECTORIES[:10]
    ACTIVE_EVALUATION_POOL_TRAJECTORIES = EVALUATION_POOL_TRAJECTORIES[:12]
    ACTIVE_CONSTRUCTION_TARGET = 3
    ACTIVE_EVALUATION_TARGET = 3
    ACTIVE_DISCOVERY_BLOCKS = DISCOVERY_BLOCKS
    ACTIVE_CONTENT_RANK = 8
    ACTIVE_PROBE_COUNT = 2
    ACTIVE_PAIRS_PER_STATE = 1
    ACTIVE_CONTEXTS = CONTEXTS
    ACTIVE_PATCHED_FORWARDS_PER_PAIR = 1 + 2 * ACTIVE_PROBE_COUNT * len(ACTIVE_CONTEXTS)
    ACTIVE_BOOTSTRAP_DRAWS = 64
    ACTIVE_MIN_PHYSICALLY_ALIGNED_PAIRS = 1
elif RUN_MODE == "pilot":
    ACTIVE_CONSTRUCTION_POOL_TRAJECTORIES = CONSTRUCTION_POOL_TRAJECTORIES
    ACTIVE_EVALUATION_POOL_TRAJECTORIES = EVALUATION_POOL_TRAJECTORIES
    ACTIVE_CONSTRUCTION_TARGET = CONSTRUCTION_TRAJECTORY_TARGET
    ACTIVE_EVALUATION_TARGET = EVALUATION_TRAJECTORY_TARGET
    ACTIVE_DISCOVERY_BLOCKS = DISCOVERY_BLOCKS
    ACTIVE_CONTENT_RANK = CONTENT_RANK
    ACTIVE_PROBE_COUNT = PROBE_COUNT
    ACTIVE_PAIRS_PER_STATE = PAIRS_PER_STATE
    ACTIVE_CONTEXTS = CONTEXTS
    ACTIVE_PATCHED_FORWARDS_PER_PAIR = PATCHED_FORWARDS_PER_PAIR
    ACTIVE_BOOTSTRAP_DRAWS = BOOTSTRAP_DRAWS
    ACTIVE_MIN_PHYSICALLY_ALIGNED_PAIRS = MIN_PHYSICALLY_ALIGNED_PAIRS
else:
    raise ValueError(
        "STAGE23_RUN_MODE must contain only smoke or pilot; "
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
assert SELECTED_BLOCK < min(DOWNSTREAM_BLOCKS)
assert PROBE_COUNT <= CONTENT_RANK < OUTPUT_SKETCH_DIM
assert PATCHED_FORWARDS_PER_PAIR == 81
assert CARRIER_SKETCH_SEED == 22197
assert not set(CONSTRUCTION_POOL_TRAJECTORIES) & set(EVALUATION_POOL_TRAJECTORIES)
'''
configuration += "\n\nPROTOCOL_CONFIG_KEYS = " + repr(
    assigned_uppercase_names(configuration)
) + "\n"


installation = base_source(2)


setup = base_source(3)
setup = setup.replace("Stage 22", "Stage 23").replace("STAGE22", "STAGE23")
setup = setup.replace("stage22_hybrid_gate", "stage23_mode_operator")
setup = setup.replace(
    'PROVENANCE_COUNTS = {"truth_generated": 0, "discovery_baseline_generated": 0, "carrier_baseline_generated": 0, "intervention_generated": 0, "patched_forwards_generated": 0, "cache_hits": 0}',
    'PROVENANCE_COUNTS = {"truth_generated": 0, "carrier_baseline_generated": 0, '
    '"intervention_generated": 0, "patched_forwards_generated": 0, "cache_hits": 0}',
)


analysis_helpers = base_source(4)
analysis_helpers += "\n\n\n" + function_sources(
    NUMERICAL.read_text(),
    [
        "countsketch_mode_covectors",
        "native_to_whitened_covectors",
        "minimal_constrained_transport",
        "symmetric_finite_response",
        "operator_transfer_metrics",
        "deterministic_signed_permutation",
    ],
)


model_helpers = base_source(5).replace("stage22-jepa-wms", "stage23-jepa-wms")


upstream_import = r'''# Bind the exact Stage 22 mode partition before any fresh model execution.


def locate_and_verify_stage22():
    root = Path(UPSTREAM_STAGE22_DIR)
    candidate = root / f"pilot_{UPSTREAM_STAGE22_RUN_SUFFIX}"
    if not candidate.is_dir():
        raise RuntimeError(
            f"missing Stage 22 Drive run {candidate}; the compact ZIP is insufficient "
            "because Stage 23 requires the frozen partition NPZ"
        )
    required = {
        "source": candidate / "source_identity.json",
        "decision": candidate / "stage22_decision.json",
        "selection": candidate / "analysis/construction_mode_selection.json",
        "partition": candidate / "subspaces/frozen_mode_partition.npz",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Stage 22 upstream artifacts are incomplete: {missing}")
    source = json.loads(required["source"].read_text())
    decision = json.loads(required["decision"].read_text())
    selection = json.loads(required["selection"].read_text())
    observed_partition_sha = sha256_file(required["partition"])
    checks = {
        "source_commit": source.get("resolved_commit") == EXPECTED_STAGE22_SOURCE_COMMIT,
        "source_execution_verified": bool(source.get("confirmation_eligible", False)),
        "protocol": source.get("protocol_id") == EXPECTED_STAGE22_PROTOCOL_ID,
        "decision": decision.get("status") == EXPECTED_STAGE22_STATUS,
        "selected_block": int(selection.get("selected_block", -1)) == SELECTED_BLOCK,
        "partition_sha": observed_partition_sha == EXPECTED_STAGE22_PARTITION_SHA256,
        "construction_gate": bool(selection.get("construction_gate_pass", False)),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Stage 22 upstream binding failed: {checks}")
    destination = SUBSPACE_DIR / "frozen_mode_partition.npz"
    shutil.copy2(required["partition"], destination)
    payload = {
        "upstream_run": str(candidate),
        "checks": checks,
        "source_identity": source,
        "decision_status": decision["status"],
        "selected_block": SELECTED_BLOCK,
        "partition_sha256": observed_partition_sha,
        "local_partition_sha256": sha256_file(destination),
    }
    write_json(SUBSPACE_DIR / "stage22_upstream_binding.json", payload)
    return payload


if not PIPELINE_FAILED:
    try:
        STAGE22_BINDING = locate_and_verify_stage22()
        print(json.dumps(STAGE22_BINDING, indent=2))
    except Exception:
        record_failure("stage22_upstream_binding")
'''


design = base_source(6)
design = design.replace("stage22_candidate_pool_design.npz", "stage23_candidate_pool_design.npz")
design = design.replace("Stage 22", "Stage 23").replace("stage22", "stage23")
design = design.replace(
    '"effect_rank": ACTIVE_EFFECT_RANK,',
    '"content_rank": ACTIVE_CONTENT_RANK,\n'
    '    "probe_count": ACTIVE_PROBE_COUNT,\n'
    '    "finite_response_contexts": ACTIVE_CONTEXTS,',
)


truth_generation = base_source(7)
truth_generation = truth_generation.replace("Stage 22", "Stage 23")
truth_generation = truth_generation.replace("stage22_truth_montage.png", "stage23_truth_montage.png")


baselines = r'''# Load the frozen JEPA-WM and cache fresh construction carriers.


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


def extract_full_carriers(records, progress_name):
    started = time.perf_counter()
    for index, record in enumerate(records):
        destination = carrier_path(record["record_id"])
        if destination.exists():
            PROVENANCE_COUNTS["cache_hits"] += 1
            raise RuntimeError(f"fresh carrier shard already exists: {destination}")
        initial, actions = state_model_inputs(record["record_id"])
        with torch.inference_mode():
            predicted, _, captures = forward_with_carriers(
                initial, actions, PRIMARY_HORIZON, capture_blocks=[SELECTED_BLOCK]
            )
            train_output = TRAIN_OUTPUT_PROJECTOR(predicted).cpu().numpy()
            eval_output = EVAL_OUTPUT_PROJECTOR(predicted).cpu().numpy()
            carrier = layer_tokens_full(captures[SELECTED_BLOCK]).detach().float().cpu().numpy()
        atomic_npz(
            destination,
            record_id=np.asarray(record["record_id"], dtype=np.int64),
            selected_block=np.asarray(SELECTED_BLOCK, dtype=np.int64),
            carrier=carrier.astype(np.float32),
            output_train_sketch=train_output.astype(np.float32),
            output_eval_sketch=eval_output.astype(np.float32),
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
        forward_with_carriers(
            initial, actions, PRIMARY_HORIZON,
            capture_blocks=[SELECTED_BLOCK, *DOWNSTREAM_BLOCKS],
        )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    total_pairs = ACTIVE_EVALUATION_TARGET * ACTIVE_PAIRS_PER_STATE
    estimate = seconds * ACTIVE_PATCHED_FORWARDS_PER_PAIR * total_pairs / 60.0
    result = {
        "seconds_per_candidate_batch": seconds,
        "pairs": int(total_pairs),
        "patched_forwards_per_pair": ACTIVE_PATCHED_FORWARDS_PER_PAIR,
        "estimated_intervention_minutes": estimate,
        "warning_threshold_minutes": MAX_ESTIMATED_TOTAL_MINUTES,
    }
    write_json(OUT / "forward_benchmark.json", result)
    if estimate > MAX_ESTIMATED_TOTAL_MINUTES and not CONTINUE_AFTER_BENCHMARK:
        raise RuntimeError("measured estimate exceeds configured credit guard")
    return result


if not PIPELINE_FAILED:
    try:
        MODEL, PREPROCESSOR, PREDICTOR, PREDICTOR_BLOCK_MODULES = load_frozen_model()
        if len(PREDICTOR_BLOCK_MODULES) != 6:
            raise RuntimeError("predictor block count changed")
        TRAIN_OUTPUT_PROJECTOR = CountSketchProjector(
            256 * 384, OUTPUT_SKETCH_DIM, TRAIN_OUTPUT_SKETCH_SEED
        )
        EVAL_OUTPUT_PROJECTOR = CountSketchProjector(
            256 * 384, OUTPUT_SKETCH_DIM, EVAL_OUTPUT_SKETCH_SEED
        )
        CARRIER_PROJECTOR = CountSketchProjector(
            256 * EXPECTED_CARRIER_CHANNELS, CARRIER_SKETCH_DIM, CARRIER_SKETCH_SEED
        )
        RESPONSE_PROJECTORS = {
            block: CountSketchProjector(
                256 * EXPECTED_CARRIER_CHANNELS,
                RESPONSE_SKETCH_DIM,
                stable_seed(RESPONSE_SKETCH_SEED, block),
            )
            for block in DOWNSTREAM_BLOCKS
        }
        HOOK_IDENTITY = hook_identity_test(CONSTRUCTION_RECORDS[0]["record_id"])
        FORWARD_BENCHMARK = forward_benchmark(CONSTRUCTION_RECORDS[0]["record_id"])
        extract_full_carriers(CONSTRUCTION_RECORDS, "construction_full_carriers")
        memory_report("construction_carriers_complete")
    except Exception:
        record_failure("construction_carriers")
'''


factorization = r'''# Freeze the fresh rank-64 content geometry before evaluation activations open.


def load_partition():
    with np.load(SUBSPACE_DIR / "frozen_mode_partition.npz") as payload:
        return {name: payload[name].copy() for name in payload.files}


def fit_basis_gpu(features, targets, penalty, rank):
    x = torch.as_tensor(features, device="cuda", dtype=torch.float32)
    y = torch.as_tensor(targets, device="cuda", dtype=torch.float32)
    gram = x @ x.T
    alpha = torch.linalg.solve(
        gram + float(penalty) * torch.eye(len(gram), device="cuda"), y
    )
    weight = x.T @ alpha
    left, singular, _ = torch.linalg.svd(weight, full_matrices=False)
    keep = min(int(rank), left.shape[1], int(torch.sum(singular > 1e-7).item()))
    if keep < int(rank):
        raise RuntimeError(f"content rank {keep} is below required rank {rank}")
    basis = left[:, : int(rank)].detach().cpu().numpy().astype(np.float64)
    singular_values = singular.detach().cpu().numpy().astype(np.float64)
    del x, y, gram, alpha, weight, left, singular
    torch.cuda.empty_cache()
    return basis, singular_values


def construction_matrices():
    count = 0
    total = np.zeros(EXPECTED_CARRIER_CHANNELS, dtype=np.float64)
    cross = np.zeros((EXPECTED_CARRIER_CHANNELS, EXPECTED_CARRIER_CHANNELS), dtype=np.float64)
    native_residuals, output_residuals = [], []
    for record in CONSTRUCTION_RECORDS:
        payload = load_carrier(record["record_id"])
        carrier = payload["carrier"].astype(np.float64)
        channels = carrier.reshape(-1, carrier.shape[-1])
        count += len(channels)
        total += channels.sum(axis=0)
        cross += channels.T @ channels
        native_residuals.append(candidate_center(carrier))
        output_residuals.append(candidate_center(payload["output_train_sketch"]))
    metric = channel_metric_from_moments(
        count, total, cross,
        shrinkage=CHANNEL_SHRINKAGE,
        relative_floor=CHANNEL_EIGEN_FLOOR,
    )
    whitened = [
        transform_primal_channels(value, metric["inverse_square_root"])
        for value in native_residuals
    ]
    x_native = np.concatenate(
        [value.reshape(ACTIONS_PER_STATE, -1) for value in whitened]
    )
    x = x_native / np.sqrt(x_native.shape[1])
    y = np.concatenate(output_residuals).astype(np.float64)
    output_scale = np.std(y, axis=0, ddof=1)
    positive = output_scale[output_scale > 1e-12]
    if not len(positive):
        raise RuntimeError("construction output sketch has zero variance")
    output_scale = np.maximum(output_scale, np.median(positive) * 1e-3)
    y /= output_scale[None]
    return x, x_native, y, metric, output_scale


def fit_and_freeze_geometry():
    partition = load_partition()
    x, x_native, y, metric, output_scale = construction_matrices()
    gram = x @ x.T
    penalty = float(RIDGE_MULTIPLIER * np.trace(gram) / len(gram))
    raw_basis, singular = fit_basis_gpu(x, y, penalty, ACTIVE_CONTENT_RANK)
    native_covectors = countsketch_mode_covectors(
        CARRIER_PROJECTOR.bucket.detach().cpu().numpy(),
        CARRIER_PROJECTOR.sign.detach().cpu().numpy(),
        CARRIER_PROJECTOR.scale.detach().cpu().numpy(),
        partition["scale"],
        partition["components"],
        center_factor=(ACTIONS_PER_STATE - 1) / ACTIONS_PER_STATE,
    )
    mode_covectors = native_to_whitened_covectors(
        native_covectors, metric["square_root"], EXPECTED_CARRIER_CHANNELS
    )
    content_basis = orthogonalize_basis(raw_basis, mode_covectors)
    if content_basis.shape[1] < ACTIVE_CONTENT_RANK:
        raise RuntimeError("mode removal reduced the requested content rank")
    content_basis = content_basis[:, :ACTIVE_CONTENT_RANK]
    orthogonality = float(np.max(np.abs(mode_covectors.T @ content_basis)))
    if orthogonality > 1e-7:
        raise RuntimeError(f"mode/content orthogonality failed: {orthogonality}")
    coefficients = x_native @ content_basis
    probe_scale = np.std(coefficients, axis=0, ddof=1)
    positive_probe = probe_scale[probe_scale > 1e-10]
    if not len(positive_probe):
        raise RuntimeError("frozen content probes have zero construction variance")
    floor = float(np.median(positive_probe) * 1e-3)
    probe_scale = np.maximum(probe_scale, floor)
    probe_vectors = content_basis[:, :ACTIVE_PROBE_COUNT] * probe_scale[:ACTIVE_PROBE_COUNT]
    atomic_npz(
        SUBSPACE_DIR / "frozen_mode_operator_geometry.npz",
        selected_block=np.asarray(SELECTED_BLOCK, dtype=np.int64),
        channel_square_root=metric["square_root"],
        channel_inverse_square_root=metric["inverse_square_root"],
        channel_eigenvalues=metric["eigenvalues"],
        mode_covectors_white=mode_covectors,
        content_basis=content_basis,
        probe_vectors=probe_vectors,
        probe_scale=probe_scale[:ACTIVE_PROBE_COUNT],
        output_scale=output_scale,
        ridge_penalty=np.asarray(penalty),
        singular_values=singular,
    )
    payload = {
        "frozen_before_evaluation_activations": True,
        "contact_labels_used": False,
        "upstream_partition_sha256": EXPECTED_STAGE22_PARTITION_SHA256,
        "selected_block": SELECTED_BLOCK,
        "content_rank": ACTIVE_CONTENT_RANK,
        "probe_count": ACTIVE_PROBE_COUNT,
        "ridge_penalty": penalty,
        "mode_content_max_abs_inner_product": orthogonality,
        "channel_condition_number": metric["condition_number"],
        "geometry_sha256": sha256_file(SUBSPACE_DIR / "frozen_mode_operator_geometry.npz"),
        "evaluation_activation_ids_seen": [],
    }
    write_json(SUBSPACE_DIR / "mode_operator_geometry_freeze.json", payload)
    return payload


if not PIPELINE_FAILED:
    try:
        GEOMETRY_FREEZE = fit_and_freeze_geometry()
        print(json.dumps(GEOMETRY_FREEZE, indent=2))
    except Exception:
        record_failure("construction_geometry_freeze")
'''


evaluation_open = r'''# Open held-out activations, freeze model-only pairs, then reveal contact labels.


def frozen_mode_assignments_from_carrier(carrier, partition):
    values = torch.as_tensor(carrier, device="cuda", dtype=torch.float32)
    with torch.inference_mode():
        sketch = CARRIER_PROJECTOR(values).cpu().numpy().astype(np.float64)
    centered = candidate_center(sketch)
    assignments, transformed = apply_mode_partition(
        centered,
        partition["mean"],
        partition["scale"],
        partition["components"],
        partition["centroids"],
    )
    return assignments, transformed


def select_model_only_pairs(record, assignments, transformed, partition):
    on_cluster = int(partition["mode_on_cluster"])
    off_cluster = 1 - on_cluster
    centroids = partition["centroids"].astype(np.float64)
    off = np.flatnonzero(assignments == off_cluster)
    on = np.flatnonzero(assignments == on_cluster)
    if len(off) < ACTIVE_PAIRS_PER_STATE or len(on) < ACTIVE_PAIRS_PER_STATE:
        return []
    off_distance = np.sum((transformed[off] - centroids[off_cluster]) ** 2, axis=1)
    on_distance = np.sum((transformed[on] - centroids[on_cluster]) ** 2, axis=1)
    off = off[np.argsort(off_distance, kind="stable")]
    on = on[np.argsort(on_distance, kind="stable")]
    return [
        {
            "pair_id": f"{int(record['record_id'])}:{slot}",
            "record_id": int(record["record_id"]),
            "trajectory_id": int(record["trajectory_id"]),
            "slot": int(slot),
            "base_index": int(off[slot]),
            "donor_index": int(on[slot]),
            "base_cluster": int(off_cluster),
            "donor_cluster": int(on_cluster),
            "base_mode_coordinates": transformed[off[slot]].tolist(),
            "donor_mode_coordinates": transformed[on[slot]].tolist(),
        }
        for slot in range(ACTIVE_PAIRS_PER_STATE)
    ]


if not PIPELINE_FAILED:
    try:
        if not (SUBSPACE_DIR / "mode_operator_geometry_freeze.json").exists():
            raise RuntimeError("construction geometry must freeze before evaluation opens")
        extract_full_carriers(EVALUATION_RECORDS, "evaluation_full_carriers")
        PARTITION = load_partition()
        EVALUATION_ASSIGNMENTS = {}
        EVALUATION_TRANSFORMS = {}
        EVALUATION_PAIRS = []
        for record in EVALUATION_RECORDS:
            payload = load_carrier(record["record_id"])
            assignments, transformed = frozen_mode_assignments_from_carrier(
                payload["carrier"], PARTITION
            )
            key = str(int(record["record_id"]))
            EVALUATION_ASSIGNMENTS[key] = assignments.tolist()
            EVALUATION_TRANSFORMS[key] = transformed.tolist()
            EVALUATION_PAIRS.extend(
                select_model_only_pairs(record, assignments, transformed, PARTITION)
            )
        pair_freeze = {
            "assignments": EVALUATION_ASSIGNMENTS,
            "mode_coordinates": EVALUATION_TRANSFORMS,
            "pairs": EVALUATION_PAIRS,
            "selection_inputs": ["fresh frozen activation", "imported frozen Stage 22 partition"],
            "simulator_contact_labels_used": False,
            "pair_count": len(EVALUATION_PAIRS),
        }
        write_json(DESIGN_DIR / "evaluation_model_pair_freeze.json", pair_freeze)
        PAIR_FREEZE_SHA256 = sha256_file(DESIGN_DIR / "evaluation_model_pair_freeze.json")

        # Action-level contact labels are opened only after every model-only
        # assignment, coordinate, and pair is durably frozen above.
        label_rows = []
        PAIR_TRUTH_MAP = {}
        for record in EVALUATION_RECORDS:
            with np.load(branch_path(record["record_id"])) as truth:
                contacts = truth["interaction_counts"].astype(np.int64)
            predicted = (
                np.asarray(EVALUATION_ASSIGNMENTS[str(int(record["record_id"]))])
                == int(PARTITION["mode_on_cluster"])
            )
            for action_index in range(ACTIONS_PER_STATE):
                label_rows.append(
                    {
                        "record_id": int(record["record_id"]),
                        "trajectory_id": int(record["trajectory_id"]),
                        "action_index": int(action_index),
                        "predicted_mode_on": bool(predicted[action_index]),
                        "physical_contact": bool(contacts[action_index] > 0),
                        "contact_count": int(contacts[action_index]),
                    }
                )
        for pair in EVALUATION_PAIRS:
            with np.load(branch_path(pair["record_id"])) as truth:
                contacts = truth["interaction_counts"].astype(np.int64)
            PAIR_TRUTH_MAP[pair["pair_id"]] = {
                "base_contact": bool(contacts[pair["base_index"]] > 0),
                "donor_contact": bool(contacts[pair["donor_index"]] > 0),
                "physically_aligned": bool(
                    contacts[pair["base_index"]] == 0
                    and contacts[pair["donor_index"]] > 0
                ),
            }
        CONTACT_ALIGNMENT = binary_alignment_metrics(
            [row["predicted_mode_on"] for row in label_rows],
            [row["physical_contact"] for row in label_rows],
        )
        CONTACT_ALIGNMENT["pair_freeze_sha256"] = PAIR_FREEZE_SHA256
        CONTACT_ALIGNMENT["model_only_pairs"] = len(EVALUATION_PAIRS)
        CONTACT_ALIGNMENT["physically_aligned_pairs"] = int(
            sum(value["physically_aligned"] for value in PAIR_TRUTH_MAP.values())
        )
        write_csv(EVIDENCE_DIR / "heldout_mode_contact_rows.csv", label_rows)
        write_json(EVIDENCE_DIR / "heldout_contact_alignment.json", CONTACT_ALIGNMENT)
        EVALUATION_OPENED = True
        memory_report("evaluation_pair_freeze_and_contact_reveal_complete")
        print(json.dumps(CONTACT_ALIGNMENT, indent=2))
    except Exception:
        EVALUATION_OPENED = False
        record_failure("evaluation_open_and_contact_reveal")
'''


interventions = r'''# Run certified mode transports and finite downstream operator responses.


def load_geometry():
    with np.load(SUBSPACE_DIR / "frozen_mode_operator_geometry.npz") as payload:
        return {name: payload[name].copy() for name in payload.files}


def whiten_carrier(values, geometry):
    return transform_primal_channels(
        np.asarray(values, dtype=np.float64),
        geometry["channel_inverse_square_root"],
    )


def native_edit(values, geometry):
    return inverse_transform_primal_channels(
        np.asarray(values, dtype=np.float64), geometry["channel_square_root"]
    )


def cluster_for_coordinate(coordinate, partition):
    distances = np.sum(
        (partition["centroids"].astype(np.float64) - coordinate[None]) ** 2,
        axis=1,
    )
    return int(np.argmin(distances))


def vector_transfer_coefficient(baseline, patched, donor):
    base = np.asarray(baseline, dtype=np.float64).reshape(-1)
    edit = np.asarray(patched, dtype=np.float64).reshape(-1)
    target = np.asarray(donor, dtype=np.float64).reshape(-1) - base
    return float(np.sum((edit - base) * target) / max(np.sum(target**2), 1e-12))


def norm_matched_random_tangent(mode_covectors, content_basis, target_norm, seed):
    rng = np.random.default_rng(int(seed))
    vector = rng.normal(size=mode_covectors.shape[0])
    vector -= content_basis @ (content_basis.T @ vector)
    gram = mode_covectors.T @ mode_covectors
    vector -= mode_covectors @ np.linalg.solve(gram, mode_covectors.T @ vector)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise RuntimeError("random tangent control is degenerate")
    return vector * (float(target_norm) / norm)


def pair_contexts(pair, payload, geometry, partition):
    base_index = int(pair["base_index"])
    donor_index = int(pair["donor_index"])
    white = whiten_carrier(payload["carrier"], geometry)
    flat = white.reshape(ACTIONS_PER_STATE, -1)
    base_q = np.asarray(pair["base_mode_coordinates"], dtype=np.float64)
    mode_covectors = geometry["mode_covectors_white"].astype(np.float64)
    content_basis = geometry["content_basis"].astype(np.float64)

    native_on = flat[donor_index] - flat[base_index]
    native_q = base_q + mode_covectors.T @ native_on
    coordinate_delta = native_q - base_q
    learned_result = minimal_constrained_transport(
        mode_covectors,
        content_basis,
        coordinate_delta,
        ridge=TRANSPORT_SOLVE_RIDGE,
    )
    learned = learned_result["delta"]

    permuted_target, order, signs = deterministic_signed_permutation(
        coordinate_delta, stable_seed(PERMUTATION_SEED, pair["pair_id"])
    )
    permuted_result = minimal_constrained_transport(
        mode_covectors,
        content_basis,
        permuted_target,
        ridge=TRANSPORT_SOLVE_RIDGE,
    )
    permuted = permuted_result["delta"]
    random_tangent = norm_matched_random_tangent(
        mode_covectors,
        content_basis,
        np.linalg.norm(learned),
        stable_seed(RANDOM_TANGENT_SEED, pair["pair_id"]),
    )
    contexts = {
        "off": np.zeros_like(learned),
        "native_on": native_on,
        "mode_transport": learned,
        "permuted_transport": permuted,
        "random_tangent": random_tangent,
    }
    on_cluster = int(partition["mode_on_cluster"])
    context_q = {
        name: base_q + mode_covectors.T @ value for name, value in contexts.items()
    }
    context_cluster = {
        name: cluster_for_coordinate(value, partition) for name, value in context_q.items()
    }
    learned_leakage = float(
        np.linalg.norm(content_basis.T @ learned) / max(np.linalg.norm(learned), 1e-12)
    )
    diagnostics = {
        "pair_id": pair["pair_id"],
        "record_id": int(pair["record_id"]),
        "trajectory_id": int(pair["trajectory_id"]),
        "base_index": base_index,
        "donor_index": donor_index,
        "base_cluster": int(pair["base_cluster"]),
        "on_cluster": on_cluster,
        "native_on_cluster": context_cluster["native_on"],
        "learned_cluster": context_cluster["mode_transport"],
        "permuted_cluster": context_cluster["permuted_transport"],
        "random_cluster": context_cluster["random_tangent"],
        "native_on_flip": context_cluster["native_on"] == on_cluster,
        "learned_flip": context_cluster["mode_transport"] == on_cluster,
        "permuted_flip": context_cluster["permuted_transport"] == on_cluster,
        "random_flip": context_cluster["random_tangent"] == on_cluster,
        "mode_coordinate_residual": learned_result["mode_residual_norm"],
        "protected_content_leakage": learned_leakage,
        "learned_edit_norm": float(np.linalg.norm(learned)),
        "native_edit_norm": float(np.linalg.norm(native_on)),
        "permuted_edit_norm": float(np.linalg.norm(permuted)),
        "random_edit_norm": float(np.linalg.norm(random_tangent)),
        "constraint_condition_number": learned_result["constraint_condition_number"],
        "permutation_order": order.tolist(),
        "permutation_signs": signs.tolist(),
        **PAIR_TRUTH_MAP[pair["pair_id"]],
    }
    return contexts, diagnostics


def run_flat_edit(initial, actions, base_index, flat_edit, geometry):
    shaped = np.zeros(
        (ACTIONS_PER_STATE, 256, EXPECTED_CARRIER_CHANNELS), dtype=np.float64
    )
    shaped[int(base_index)] = flat_edit.reshape(256, EXPECTED_CARRIER_CHANNELS)
    delta_native = native_edit(shaped, geometry)
    delta = torch.as_tensor(delta_native, device="cuda", dtype=torch.float32)
    with torch.inference_mode():
        predicted, _, captures = forward_with_carriers(
            initial,
            actions,
            PRIMARY_HORIZON,
            capture_blocks=[SELECTED_BLOCK, *DOWNSTREAM_BLOCKS],
            intervention={"block": SELECTED_BLOCK, "delta": delta},
        )
        result = {
            f"block_{block}": RESPONSE_PROJECTORS[block](
                layer_tokens_full(captures[block])[int(base_index) : int(base_index) + 1]
            )[0].cpu().numpy().astype(np.float64)
            for block in DOWNSTREAM_BLOCKS
        }
        result["output"] = EVAL_OUTPUT_PROJECTOR(
            predicted[int(base_index) : int(base_index) + 1]
        )[0].cpu().numpy().astype(np.float64)
    PROVENANCE_COUNTS["patched_forwards_generated"] += 1
    del predicted, captures, delta
    return result


def pair_operator_rows(record, pair, payload, initial, actions, geometry, partition):
    contexts, diagnostics = pair_contexts(pair, payload, geometry, partition)
    base_index = int(pair["base_index"])
    donor_index = int(pair["donor_index"])
    probes = geometry["probe_vectors"].astype(np.float64).T
    responses = {
        condition: {f"block_{block}": [] for block in DOWNSTREAM_BLOCKS} | {"output": []}
        for condition in ACTIVE_CONTEXTS
    }
    native_center = run_flat_edit(
        initial, actions, base_index, contexts["native_on"], geometry
    )
    for condition in ACTIVE_CONTEXTS:
        context = contexts[condition]
        for probe in probes:
            plus = run_flat_edit(
                initial,
                actions,
                base_index,
                context + FINITE_PROBE_DOSE * probe,
                geometry,
            )
            minus = run_flat_edit(
                initial,
                actions,
                base_index,
                context - FINITE_PROBE_DOSE * probe,
                geometry,
            )
            for stage in responses[condition]:
                responses[condition][stage].append(
                    symmetric_finite_response(plus[stage], minus[stage], FINITE_PROBE_DOSE)
                )
    rows = []
    stages = [f"block_{block}" for block in DOWNSTREAM_BLOCKS] + ["output"]
    for stage in stages:
        off = np.stack(responses["off"][stage]).reshape(-1)
        native = np.stack(responses["native_on"][stage]).reshape(-1)
        for condition in ACTIVE_CONTEXTS:
            context_response = np.stack(responses[condition][stage]).reshape(-1)
            rows.append(
                {
                    "pair_id": pair["pair_id"],
                    "record_id": int(record["record_id"]),
                    "trajectory_id": int(record["trajectory_id"]),
                    "base_index": base_index,
                    "donor_index": donor_index,
                    "condition": condition,
                    "stage": stage,
                    "physically_aligned": diagnostics["physically_aligned"],
                    "learned_flip": diagnostics["learned_flip"],
                    **operator_transfer_metrics(off, context_response, native),
                }
            )
    baseline = payload["output_eval_sketch"].astype(np.float64)
    diagnostics["native_full_swap_coefficient"] = vector_transfer_coefficient(
        baseline[base_index], native_center["output"], baseline[donor_index]
    )
    observed_forwards = 1 + 2 * len(probes) * len(ACTIVE_CONTEXTS)
    if observed_forwards != ACTIVE_PATCHED_FORWARDS_PER_PAIR:
        raise RuntimeError(
            f"operator forward contract changed: {observed_forwards} != "
            f"{ACTIVE_PATCHED_FORWARDS_PER_PAIR}"
        )
    return rows, diagnostics


def intervention_path(record_id):
    return INTERVENTION_DIR / f"evaluation_{int(record_id):04d}.json"


def run_all_operator_tests():
    started = time.perf_counter()
    geometry = load_geometry()
    partition = load_partition()
    by_record = defaultdict(list)
    for pair in EVALUATION_PAIRS:
        by_record[int(pair["record_id"])].append(pair)
    rows, diagnostics = [], []
    active_records = [
        record for record in EVALUATION_RECORDS
        if int(record["record_id"]) in by_record
    ]
    for index, record in enumerate(active_records):
        destination = intervention_path(record["record_id"])
        if destination.exists():
            PROVENANCE_COUNTS["cache_hits"] += 1
            raise RuntimeError(f"fresh intervention shard already exists: {destination}")
        payload = load_carrier(record["record_id"])
        initial, actions = state_model_inputs(record["record_id"])
        record_rows, record_diagnostics = [], []
        for pair in by_record[int(record["record_id"])]:
            pair_rows, pair_diagnostics = pair_operator_rows(
                record, pair, payload, initial, actions, geometry, partition
            )
            record_rows.extend(pair_rows)
            record_diagnostics.append(pair_diagnostics)
        write_json(destination, {"operator_rows": record_rows, "transport_diagnostics": record_diagnostics})
        rows.extend(record_rows)
        diagnostics.extend(record_diagnostics)
        PROVENANCE_COUNTS["intervention_generated"] += 1
        write_json(
            OUT / "operator_progress.json",
            {
                "completed_records": index + 1,
                "total_records": len(active_records),
                "completed_pairs": len(diagnostics),
                "patched_forwards_generated": PROVENANCE_COUNTS["patched_forwards_generated"],
            },
        )
        del initial, actions
        gc.collect()
        torch.cuda.empty_cache()
    TIMINGS["operator_interventions_seconds"] = time.perf_counter() - started
    write_csv(EVIDENCE_DIR / "operator_transfer_rows.csv", rows)
    write_csv(EVIDENCE_DIR / "mode_transport_diagnostics.csv", diagnostics)
    return rows, diagnostics


if not PIPELINE_FAILED and EVALUATION_OPENED:
    try:
        OPERATOR_ROWS, TRANSPORT_DIAGNOSTICS = run_all_operator_tests()
        memory_report("mode_operator_interventions_complete")
    except Exception:
        record_failure("mode_operator_interventions")
'''


decision = r'''# Apply the preregistered Stage 23 causal operator-switch gates.


def bootstrap_payload(values, groups, label):
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups, dtype=np.int64)
    if not len(values):
        raise RuntimeError(f"no values available for {label}")
    draws = clustered_bootstrap_mean(
        values,
        groups,
        draws=ACTIVE_BOOTSTRAP_DRAWS,
        seed=stable_seed(BOOTSTRAP_SEED, label),
    )
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "mean": float(np.mean(values)),
        "lower": float(lower),
        "upper": float(upper),
        "n_pairs": int(len(values)),
        "n_trajectories": int(len(np.unique(groups))),
    }


def output_row_map(rows, condition):
    return {
        row["pair_id"]: row for row in rows
        if row["stage"] == "output"
        and row["condition"] == condition
        and row["physically_aligned"]
        and row["learned_flip"]
    }


def paired_condition_gain(rows, comparator):
    learned = output_row_map(rows, "mode_transport")
    control = output_row_map(rows, comparator)
    identifiers = sorted(set(learned) & set(control))
    values = np.asarray(
        [
            learned[pair]["transfer_coefficient"]
            - control[pair]["transfer_coefficient"]
            for pair in identifiers
        ],
        dtype=np.float64,
    )
    groups = np.asarray(
        [learned[pair]["trajectory_id"] for pair in identifiers], dtype=np.int64
    )
    return identifiers, values, groups


def evaluate_mode_operator(rows, diagnostics):
    learned = output_row_map(rows, "mode_transport")
    identifiers = sorted(learned)
    flip_rate = float(np.mean([row["learned_flip"] for row in diagnostics])) if diagnostics else 0.0
    native_flip_rate = float(np.mean([row["native_on_flip"] for row in diagnostics])) if diagnostics else 0.0
    max_residual = float(max((row["mode_coordinate_residual"] for row in diagnostics), default=1e9))
    max_leakage = float(max((row["protected_content_leakage"] for row in diagnostics), default=1e9))
    aligned_diagnostics = [row for row in diagnostics if row["physically_aligned"]]
    full_swap = float(np.mean([row["native_full_swap_coefficient"] for row in aligned_diagnostics])) if aligned_diagnostics else 0.0
    if not identifiers:
        return {
            "physically_aligned_certified_pairs": 0,
            "learned_mode_flip_rate_all_model_pairs": flip_rate,
            "native_on_flip_rate_all_model_pairs": native_flip_rate,
            "max_mode_coordinate_residual": max_residual,
            "max_protected_content_leakage": max_leakage,
            "mean_native_full_swap_coefficient": full_swap,
            "nondegenerate_native_operator_target_fraction": 0.0,
            "learned_operator_transfer": None,
            "gain_over_random_tangent": None,
            "gain_over_permuted_transport": None,
            "positive_gain_over_random_pairs": 0,
            "positive_gain_over_permuted_pairs": 0,
            "required_positive_pairs": 0,
            "random_gain_sign_test_p": None,
            "permuted_gain_sign_test_p": None,
            "transport_certification_pass": False,
            "operator_switch_pass": False,
            "causal_mode_operator_switch_pass": False,
            "reason": "no physically aligned pair with a certified learned mode flip",
        }
    learned_values = np.asarray(
        [learned[pair]["transfer_coefficient"] for pair in identifiers], dtype=np.float64
    )
    learned_groups = np.asarray(
        [learned[pair]["trajectory_id"] for pair in identifiers], dtype=np.int64
    )
    target_energy = np.asarray(
        [learned[pair]["target_energy"] for pair in identifiers], dtype=np.float64
    )
    _, random_gain, random_groups = paired_condition_gain(rows, "random_tangent")
    _, permuted_gain, permuted_groups = paired_condition_gain(rows, "permuted_transport")
    learned_summary = bootstrap_payload(learned_values, learned_groups, "learned_transfer")
    random_summary = bootstrap_payload(random_gain, random_groups, "gain_over_random")
    permuted_summary = bootstrap_payload(permuted_gain, permuted_groups, "gain_over_permuted")
    required_positive = int(np.ceil(REQUIRED_POSITIVE_GAIN_FRACTION * len(identifiers)))

    nondegenerate_fraction = float(np.mean(target_energy > TARGET_ENERGY_FLOOR))
    payload = {
        "physically_aligned_certified_pairs": len(identifiers),
        "learned_mode_flip_rate_all_model_pairs": flip_rate,
        "native_on_flip_rate_all_model_pairs": native_flip_rate,
        "max_mode_coordinate_residual": max_residual,
        "max_protected_content_leakage": max_leakage,
        "mean_native_full_swap_coefficient": full_swap,
        "nondegenerate_native_operator_target_fraction": nondegenerate_fraction,
        "learned_operator_transfer": learned_summary,
        "gain_over_random_tangent": random_summary,
        "gain_over_permuted_transport": permuted_summary,
        "positive_gain_over_random_pairs": int(np.sum(random_gain > 0)),
        "positive_gain_over_permuted_pairs": int(np.sum(permuted_gain > 0)),
        "required_positive_pairs": required_positive,
        "random_gain_sign_test_p": exact_positive_sign_test(random_gain),
        "permuted_gain_sign_test_p": exact_positive_sign_test(permuted_gain),
    }
    payload["transport_certification_pass"] = bool(
        native_flip_rate >= MIN_LEARNED_FLIP_RATE
        and flip_rate >= MIN_LEARNED_FLIP_RATE
        and max_residual <= MAX_MODE_COORDINATE_RESIDUAL
        and max_leakage <= MAX_PROTECTED_CONTENT_LEAKAGE
        and full_swap >= MIN_NATIVE_FULL_SWAP_COEFFICIENT
    )
    payload["operator_switch_pass"] = bool(
        len(identifiers) >= ACTIVE_MIN_PHYSICALLY_ALIGNED_PAIRS
        and nondegenerate_fraction >= MIN_NONDEGENERATE_TARGET_FRACTION
        and learned_summary["mean"] >= MIN_OPERATOR_TRANSFER_COEFFICIENT
        and learned_summary["lower"] > 0
        and random_summary["mean"] >= MIN_GAIN_OVER_RANDOM
        and random_summary["lower"] > 0
        and permuted_summary["mean"] >= MIN_GAIN_OVER_PERMUTED
        and permuted_summary["lower"] > 0
        and payload["positive_gain_over_random_pairs"] >= required_positive
        and payload["positive_gain_over_permuted_pairs"] >= required_positive
    )
    payload["causal_mode_operator_switch_pass"] = bool(
        payload["transport_certification_pass"] and payload["operator_switch_pass"]
    )
    return payload


def downstream_curve(rows):
    result = []
    for stage in [f"block_{block}" for block in DOWNSTREAM_BLOCKS] + ["output"]:
        for condition in ["mode_transport", "permuted_transport", "random_tangent"]:
            values = [
                row["transfer_coefficient"] for row in rows
                if row["stage"] == stage
                and row["condition"] == condition
                and row["physically_aligned"]
                and row["learned_flip"]
            ]
            result.append(
                {
                    "stage": stage,
                    "condition": condition,
                    "mean_transfer_coefficient": float(np.mean(values)) if values else None,
                    "n_pairs": len(values),
                }
            )
    return result


if not PIPELINE_FAILED:
    try:
        SOURCE_EXECUTION_VERIFIED = verify_executed_notebook_through(
            "# Apply the preregistered Stage 23 causal operator-switch gates."
        )
        physical_mode_pass = bool(
            CONTACT_ALIGNMENT["balanced_accuracy"] >= MIN_CONTACT_BALANCED_ACCURACY
            and CONTACT_ALIGNMENT["matthews_correlation"] >= MIN_CONTACT_MCC
        )
        MODE_OPERATOR_RESULT = evaluate_mode_operator(
            OPERATOR_ROWS, TRANSPORT_DIAGNOSTICS
        )
        LAYERWISE_TRANSFER = downstream_curve(OPERATOR_ROWS)
        write_csv(EVIDENCE_DIR / "layerwise_operator_transfer.csv", LAYERWISE_TRANSFER)

        if RUN_MODE == "smoke":
            candidate_status = "SMOKE_ONLY"
        elif not physical_mode_pass:
            candidate_status = "STAGE22_MODE_NOT_REPLICATED_ON_FRESH_STATES"
        elif MODE_OPERATOR_RESULT["physically_aligned_certified_pairs"] == 0:
            candidate_status = "INSUFFICIENT_FRESH_MODE_PAIRS"
        elif not MODE_OPERATOR_RESULT["transport_certification_pass"]:
            candidate_status = "MODE_TRANSPORT_INVALID"
        elif MODE_OPERATOR_RESULT["causal_mode_operator_switch_pass"]:
            candidate_status = "CAUSAL_MODE_OPERATOR_SWITCH_CONFIRMED"
        elif (
            MODE_OPERATOR_RESULT["nondegenerate_native_operator_target_fraction"]
            < MIN_NONDEGENERATE_TARGET_FRACTION
        ):
            candidate_status = "NO_MEASURABLE_NATIVE_OPERATOR_DIFFERENCE"
        else:
            candidate_status = "MODE_FLIPS_WITHOUT_OPERATOR_SWITCH"

        upstream_eligible = bool(all(STAGE22_BINDING["checks"].values()))
        source_eligible = bool(
            SOURCE_IDENTITY.get("confirmation_eligible", False) and upstream_eligible
        )
        status = (
            candidate_status
            if RUN_MODE == "smoke" or source_eligible
            else "UNBOUND_EXPLORATORY_RESULT"
        )
        DECISION_PAYLOAD = {
            "status": status,
            "candidate_status": candidate_status,
            "source_bound_claim_eligible": source_eligible,
            "stage22_upstream_bound": upstream_eligible,
            "heldout_contact_alignment": CONTACT_ALIGNMENT,
            "causal_mode_operator_test": MODE_OPERATOR_RESULT,
            "layerwise_operator_transfer": LAYERWISE_TRANSFER,
            "claim_boundary": {
                "complete_hybrid_automaton_extracted": False,
                "universal_contact_variable_claim_authorized": False,
                "planning_utility_claim_authorized": False,
                "jacobian_or_infinitesimal_linearization_claim_authorized": False,
                "sublayer_circuit_localized": False,
                "authorized_claim": "one distributed mode state causally selects a distinct finite downstream response operator in one frozen JEPA-WM",
            },
        }
        write_json(OUT / "stage23_decision.json", DECISION_PAYLOAD)

        figure, axes = plt.subplots(1, 3, figsize=(14, 4))
        labels = ["balanced accuracy", "MCC"]
        axes[0].bar(labels, [CONTACT_ALIGNMENT["balanced_accuracy"], CONTACT_ALIGNMENT["matthews_correlation"]])
        axes[0].set_ylim(-0.1, 1.0)
        axes[0].set_title("Fresh physical alignment")
        axes[1].bar(
            ["native on", "mode transport"],
            [
                MODE_OPERATOR_RESULT["native_on_flip_rate_all_model_pairs"],
                MODE_OPERATOR_RESULT["learned_mode_flip_rate_all_model_pairs"],
            ],
        )
        axes[1].set_ylim(0, 1.05)
        axes[1].set_title("Certified off→on reassignment")
        stage_order = [f"block_{block}" for block in DOWNSTREAM_BLOCKS] + ["output"]
        for condition, label in [
            ("mode_transport", "mode transport"),
            ("permuted_transport", "permuted"),
            ("random_tangent", "random tangent"),
        ]:
            selected = [row for row in LAYERWISE_TRANSFER if row["condition"] == condition]
            lookup = {row["stage"]: row["mean_transfer_coefficient"] for row in selected}
            axes[2].plot(
                stage_order,
                [np.nan if lookup[stage] is None else lookup[stage] for stage in stage_order],
                marker="o",
                label=label,
            )
        axes[2].axhline(0, color="black", linewidth=0.8)
        axes[2].set(ylabel="native-operator transfer", title="Downstream emergence")
        axes[2].legend()
        axes[2].tick_params(axis="x", rotation=25)
        figure.tight_layout()
        figure.savefig(PLOT_DIR / "stage23_mode_operator_summary.png", dpi=180)
        plt.close(figure)
        print(json.dumps(DECISION_PAYLOAD, indent=2))
    except Exception:
        record_failure("decision_and_plots")
        DECISION_PAYLOAD = {"status": "INCONCLUSIVE", "failure": FAILURE_MESSAGE}
else:
    DECISION_PAYLOAD = {"status": "INCONCLUSIVE", "failure": FAILURE_MESSAGE}

if not (OUT / "stage23_decision.json").exists():
    write_json(OUT / "stage23_decision.json", DECISION_PAYLOAD)
'''


packaging = r'''# Package compact audit evidence and download one result bundle.
write_json(OUT / "timings.json", TIMINGS)
memory_report("final")
if not PIPELINE_FAILED:
    (OUT / "FAILURE_TRACE.txt").write_text("NONE\n")

raw_roots = [TRUTH_DIR, BASELINE_DIR, INTERVENTION_DIR]
excluded_roots = {ASSET_DIR, *raw_roots}
RAW_MANIFEST = manifest_rows(OUT, excluded_roots=())
write_json(OUT / "raw_manifest.json", RAW_MANIFEST)

compact_files = []
for path in sorted(OUT.rglob("*")):
    if not path.is_file():
        continue
    if any(root == path or root in path.parents for root in excluded_roots):
        continue
    if SUBSPACE_DIR in path.parents and path.suffix == ".npz":
        continue
    if path.name.startswith("stage23_mode_operator_result_bundle_"):
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

archive_base = OUT / f"stage23_mode_operator_result_bundle_{RUN_SIGNATURE[:12]}"
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
    baselines,
    factorization,
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
    code(baselines),
    code(factorization),
    code(evaluation_open),
    code(interventions),
    code(decision),
    code(packaging),
]
for index, cell in enumerate(cells):
    cell["id"] = f"stage23-{index:02d}"

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
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
TARGET.write_text(json.dumps(notebook, indent=1) + "\n")
print(f"Wrote {TARGET}")
