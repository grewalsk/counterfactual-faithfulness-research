import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "22_latent_hybrid_gate_interaction.ipynb"
BASE = json.loads((ROOT / "18_rank64_action_contrast_confirmation.ipynb").read_text())
NUMERICAL = ROOT.parent / "src/cf_faithfulness/stage22_hybrid_gate.py"


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


introduction = r'''# Stage 22: latent hybrid-gate causal interaction

This notebook tests whether a frozen action-conditioned JEPA world model uses
an **event-gated computation**, rather than merely storing one additive action
direction.  The proposed computation is

\[
\hat z^+=b(z,a)+g(z,a)u(z,a),
\]

where `g` selects an internal interaction mode and `u` carries a continuous
action effect.  Construction activations are candidate-centered, sketched, and
partitioned by label-free deterministic two-means.  Simulator contact counts
do not fit the partition, orient its labels, or select the predictor block.

At the construction-selected block, the notebook freezes a one-dimensional
mode direction and an orthogonal output-aligned effect subspace.  Held-out
base/donor pairs are selected from frozen model assignments.  Four finite
interchanges produce

\[
Y_{ij}=F_\theta(\operatorname{do}(G=G_i),
                 \operatorname{do}(U=U_j)),
\qquad
I=Y_{11}-Y_{10}-Y_{01}+Y_{00}.
\]

An additive representation has `I=0`; a multiplicative gate has a structured
nonzero interaction.  The primary interaction must align with the donor
consequence and exceed within-state shuffled and empirical-span random gate
controls.  Exact simulator contact labels are opened only after evaluation
assignments are frozen, allowing a separate test of whether the discovered
internal mode is physically interpretable as contact.

This is a forward-pass-only, falsification-first pilot.  It computes no
Jacobian, JVP, VJP, gradient probe, or model-weight update.  A pass establishes
one event-gated internal mechanism, not a complete hybrid automaton or planning
algorithm.  Return `stage22_hybrid_gate_result_bundle_<signature>.zip`.
'''


configuration = r'''# SINGLE CONFIGURATION BLOCK
# Required Colab secrets for a source-bound pilot:
# STAGE22_RUN_MODE=pilot
# STAGE22_SOURCE_COMMIT=<full 40-hex commit from the handoff>
# STAGE22_RUN_NONCE=<new unique label, e.g. hybrid_gate_20260804_a>
RUN_MODE = "smoke"
EXPERIMENT_SOURCE_REF = ""
RUN_NONCE = "smoke"
try:
    from google.colab import userdata as _colab_userdata

    RUN_MODE = str(_colab_userdata.get("STAGE22_RUN_MODE") or RUN_MODE).strip().lower()
    EXPERIMENT_SOURCE_REF = str(
        _colab_userdata.get("STAGE22_SOURCE_COMMIT") or EXPERIMENT_SOURCE_REF
    ).strip()
    RUN_NONCE = str(
        _colab_userdata.get("STAGE22_RUN_NONCE") or RUN_NONCE
    ).strip()
except Exception:
    pass

if RUN_MODE == "pilot":
    if RUN_NONCE in {"", "smoke"}:
        raise ValueError("pilot mode requires a unique STAGE22_RUN_NONCE")
    if not all(value.isalnum() or value in "-_" for value in RUN_NONCE):
        raise ValueError("STAGE22_RUN_NONCE may contain only letters, numbers, '-' and '_'")

MOUNT_DRIVE = True
DOWNLOAD_RESULTS = True
CONTINUE_AFTER_BENCHMARK = True
MAX_ESTIMATED_TOTAL_MINUTES = 180.0
FRESH_RUN_REQUIRED = True

OUTPUT_DIR = "/content/counterfactual_faithfulness_stage22_hybrid_gate"
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/counterfactual_faithfulness_stage22_hybrid_gate"

PROTOCOL_ID = "stage22-label-free-mode-gate-effect-factorial-v1"
NOTEBOOK_PROTOCOL_SHA256 = "__PROTOCOL_DIGEST__"
EVIDENCE_STATUS = "CONFIRMATORY_ONLY_IF_SOURCE_BOUND_FRESH_AND_EVALUATION_SEALED"
EXPERIMENT_REPOSITORY = "grewalsk/counterfactual-faithfulness-research"
EXPERIMENT_NOTEBOOK_PATH = "notebooks/22_latent_hybrid_gate_interaction.ipynb"
EXPERIMENT_BUILDER_PATH = "notebooks/build_stage22_hybrid_gate_notebook.py"
EXPERIMENT_NUMERICAL_PATH = "src/cf_faithfulness/stage22_hybrid_gate.py"

SEED = 22101
DESIGN_SEED = 22137
MODEL_NAME = "jepa_wm_pusht"
ENVIRONMENT = "PushT"
FRAMESKIP = 5
PRIMARY_HORIZON = 3
TARGET_STEPS = [PRIMARY_HORIZON]
DISCOVERY_BLOCKS = [0, 1, 2, 3, 4]
ACTIVE_BLOCKS = DISCOVERY_BLOCKS
EXPECTED_CARRIER_CHANNELS = 400

CONSTRUCTION_POOL_TRAJECTORIES = list(range(900, 956))
EVALUATION_POOL_TRAJECTORIES = list(range(1000, 1072))
CONSTRUCTION_TRAJECTORY_TARGET = 32
EVALUATION_TRAJECTORY_TARGET = 40
TASK_ID_OFFSET = 5000

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
TRAIN_OUTPUT_SKETCH_SEED = 22161
EVAL_OUTPUT_SKETCH_SEED = 22183
CARRIER_SKETCH_SEED = 22197
MODE_PCA_RANK = 8
MODE_CLUSTER_SEED = 22211
CHANNEL_SHRINKAGE = 0.10
CHANNEL_EIGEN_FLOOR = 1e-6
RIDGE_MULTIPLIERS = [1e-8, 1e-6, 1e-4, 1e-2, 1.0, 100.0]
EFFECT_RANK = 32
NULL_ROOT_SEED = 22231
SHUFFLE_SEED = 22251
PERMUTATION_SEED = 22253
BOOTSTRAP_SEED = 22269
BOOTSTRAP_DRAWS = 10000
PAIRS_PER_STATE = 2
PATCHED_FORWARDS_PER_PAIR = 8
MAX_ZERO_EDIT_ERROR = 1e-6

MIN_DISCOVERY_SEPARATION = 0.02
MIN_DISCOVERY_BALANCE = 0.20
MIN_DISCOVERY_STATE_COVERAGE = 0.50
MIN_DISCOVERY_OUTPUT_ENERGY_RATIO = 1.10
MIN_CONTACT_BALANCED_ACCURACY = 0.65
MIN_CONTACT_MCC = 0.30
MIN_PHYSICALLY_ALIGNED_PAIRS = 24
MIN_FULL_SWAP_COEFFICIENT = 0.60
MIN_INTERACTION_COEFFICIENT = 0.05
MIN_INTERACTION_COSINE = 0.10
MIN_INTERACTION_GAIN_OVER_RANDOM = 0.03
MIN_INTERACTION_GAIN_OVER_SHUFFLED = 0.03
REQUIRED_POSITIVE_GAIN_FRACTION = 0.65

if RUN_MODE == "smoke":
    ACTIVE_CONSTRUCTION_POOL_TRAJECTORIES = CONSTRUCTION_POOL_TRAJECTORIES[:10]
    ACTIVE_EVALUATION_POOL_TRAJECTORIES = EVALUATION_POOL_TRAJECTORIES[:12]
    ACTIVE_CONSTRUCTION_TARGET = 3
    ACTIVE_EVALUATION_TARGET = 3
    ACTIVE_DISCOVERY_BLOCKS = [2, 4]
    ACTIVE_EFFECT_RANK = 4
    ACTIVE_PAIRS_PER_STATE = 1
    ACTIVE_BOOTSTRAP_DRAWS = 64
    ACTIVE_MIN_PHYSICALLY_ALIGNED_PAIRS = 1
elif RUN_MODE == "pilot":
    ACTIVE_CONSTRUCTION_POOL_TRAJECTORIES = CONSTRUCTION_POOL_TRAJECTORIES
    ACTIVE_EVALUATION_POOL_TRAJECTORIES = EVALUATION_POOL_TRAJECTORIES
    ACTIVE_CONSTRUCTION_TARGET = CONSTRUCTION_TRAJECTORY_TARGET
    ACTIVE_EVALUATION_TARGET = EVALUATION_TRAJECTORY_TARGET
    ACTIVE_DISCOVERY_BLOCKS = DISCOVERY_BLOCKS
    ACTIVE_EFFECT_RANK = EFFECT_RANK
    ACTIVE_PAIRS_PER_STATE = PAIRS_PER_STATE
    ACTIVE_BOOTSTRAP_DRAWS = BOOTSTRAP_DRAWS
    ACTIVE_MIN_PHYSICALLY_ALIGNED_PAIRS = MIN_PHYSICALLY_ALIGNED_PAIRS
else:
    raise ValueError(
        "STAGE22_RUN_MODE must contain only smoke or pilot; "
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
assert max(DISCOVERY_BLOCKS) <= 4
assert EFFECT_RANK < OUTPUT_SKETCH_DIM
assert not set(CONSTRUCTION_POOL_TRAJECTORIES) & set(EVALUATION_POOL_TRAJECTORIES)
'''
configuration += "\n\nPROTOCOL_CONFIG_KEYS = " + repr(
    assigned_uppercase_names(configuration)
) + "\n"


installation = base_source(2)


setup = base_source(3)
setup = setup.replace("Stage 18", "Stage 22").replace("STAGE18", "STAGE22")
setup = setup.replace("stage18_rank64", "stage22_hybrid_gate")
setup = setup.replace(
    'PROVENANCE_COUNTS = {"truth_generated": 0, "baseline_generated": 0, "intervention_generated": 0, "cache_hits": 0}',
    'PROVENANCE_COUNTS = {"truth_generated": 0, "discovery_baseline_generated": 0, '
    '"carrier_baseline_generated": 0, "intervention_generated": 0, '
    '"patched_forwards_generated": 0, "cache_hits": 0}',
)


analysis_helpers = base_source(4)
analysis_helpers += "\n\n\n" + function_sources(
    NUMERICAL.read_text(),
    [
        "center_by_group",
        "deterministic_two_means",
        "apply_mode_partition",
        "discover_mode_partition",
        "binary_alignment_metrics",
        "difference_of_means_direction",
        "orthogonalize_basis",
        "projected_pair_delta",
        "factorial_interaction_metrics",
    ],
)


model_helpers = base_source(5).replace("stage18-jepa-wms", "stage22-jepa-wms")


design = base_source(6)
design = design.replace(
    "# Freeze candidate pools, model-blind eligibility rules, actions, and null seeds.",
    "# Freeze candidate pools, actions, physical eligibility, and all null seeds.",
)
design = design.replace("stage18_candidate_pool_design.npz", "stage22_candidate_pool_design.npz")
design = design.replace(
    '"min_contact_branches": MIN_ELIGIBLE_CONTACT_BRANCHES,',
    '"min_contact_branches": MIN_ELIGIBLE_CONTACT_BRANCHES,\n'
    '        "min_noncontact_branches": MIN_ELIGIBLE_NONCONTACT_BRANCHES,',
)
design = design.replace(
    '"fixed_block": FIXED_BLOCK,\n    "fixed_primary_rank": PRIMARY_RANK,',
    '"discovery_blocks": ACTIVE_DISCOVERY_BLOCKS,\n'
    '    "effect_rank": ACTIVE_EFFECT_RANK,',
)
design = design.replace(
    '"candidate_pool_sha256": sha256_file(DESIGN_DIR / "stage18_candidate_pool_design.npz")',
    '"candidate_pool_sha256": sha256_file(DESIGN_DIR / "stage22_candidate_pool_design.npz")',
)


truth_generation = base_source(7)
truth_generation = truth_generation.replace(
    "# Generate and select physical truth before loading any model or encoder.",
    "# Generate exact branches and apply model-blind physical eligibility.",
)
truth_generation = truth_generation.replace(
    "and metrics[\"contact_branches\"] >= MIN_ELIGIBLE_CONTACT_BRANCHES\n    )",
    "and metrics[\"contact_branches\"] >= MIN_ELIGIBLE_CONTACT_BRANCHES\n"
    "        and ACTIONS_PER_STATE - metrics[\"contact_branches\"] >= MIN_ELIGIBLE_NONCONTACT_BRANCHES\n"
    "    )",
)
truth_generation = truth_generation.replace(
    '"eligible": eligible,',
    '"noncontact_branches": int(ACTIONS_PER_STATE - metrics["contact_branches"]),\n'
    '        "eligible": eligible,',
)
truth_generation = truth_generation.replace(
    "stage18_truth_montage.png", "stage22_truth_montage.png"
)


discovery_baselines = r'''# Load the frozen JEPA-WM and cache label-free discovery sketches.


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


def discovery_path(record_id):
    return BASELINE_DIR / f"discovery_{int(record_id):04d}.npz"


def load_discovery(record_id):
    with np.load(discovery_path(record_id)) as payload:
        return {name: payload[name].copy() for name in payload.files}


def extract_discovery_baselines(records, progress_name):
    started = time.perf_counter()
    for index, record in enumerate(records):
        destination = discovery_path(record["record_id"])
        if destination.exists():
            PROVENANCE_COUNTS["cache_hits"] += 1
            raise RuntimeError(f"fresh discovery shard already exists: {destination}")
        initial, actions = state_model_inputs(record["record_id"])
        with torch.inference_mode():
            predicted, predicted_proprio, captures = forward_with_carriers(
                initial,
                actions,
                PRIMARY_HORIZON,
                capture_blocks=ACTIVE_DISCOVERY_BLOCKS,
            )
            train_output = TRAIN_OUTPUT_PROJECTOR(predicted).cpu().numpy()
            eval_output = EVAL_OUTPUT_PROJECTOR(predicted).cpu().numpy()
            carrier_sketches = np.stack(
                [
                    CARRIER_PROJECTOR(layer_tokens_full(captures[block])).cpu().numpy()
                    for block in ACTIVE_DISCOVERY_BLOCKS
                ]
            )
        atomic_npz(
            destination,
            record_id=np.asarray(record["record_id"], dtype=np.int64),
            trajectory_id=np.asarray(record["trajectory_id"], dtype=np.int64),
            split=np.asarray(record["split"]),
            blocks=np.asarray(ACTIVE_DISCOVERY_BLOCKS, dtype=np.int64),
            carrier_sketches=carrier_sketches.astype(np.float32),
            output_train_sketch=train_output.astype(np.float32),
            output_eval_sketch=eval_output.astype(np.float32),
            predicted_proprio=predicted_proprio.detach().float().cpu().numpy(),
        )
        PROVENANCE_COUNTS["discovery_baseline_generated"] += 1
        write_json(
            OUT / f"{progress_name}_progress.json",
            {
                "completed": index + 1,
                "total": len(records),
                "last_record_id": int(record["record_id"]),
            },
        )
        del initial, actions, predicted, predicted_proprio, captures, carrier_sketches
        gc.collect()
        torch.cuda.empty_cache()
    TIMINGS[f"{progress_name}_seconds"] = time.perf_counter() - started


def hook_identity_test(record_id):
    initial, actions = state_model_inputs(record_id)
    block = int(ACTIVE_DISCOVERY_BLOCKS[-1])
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
    result = {
        "record_id": int(record_id),
        "block": block,
        "max_abs_error": error,
        "passed": error <= MAX_ZERO_EDIT_ERROR,
    }
    if not result["passed"]:
        raise RuntimeError(f"zero intervention changed output: {result}")
    write_json(OUT / "hook_identity_test.json", result)
    return result


def forward_benchmark(record_id):
    initial, actions = state_model_inputs(record_id)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        _, _, _ = forward_with_carriers(
            initial,
            actions,
            PRIMARY_HORIZON,
            capture_blocks=[int(ACTIVE_DISCOVERY_BLOCKS[-1])],
        )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    total_pairs = ACTIVE_EVALUATION_TARGET * ACTIVE_PAIRS_PER_STATE
    estimate = seconds * PATCHED_FORWARDS_PER_PAIR * total_pairs / 60.0
    result = {
        "seconds_per_candidate_batch": seconds,
        "pairs": int(total_pairs),
        "patched_forwards_per_pair": PATCHED_FORWARDS_PER_PAIR,
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
            256 * EXPECTED_CARRIER_CHANNELS,
            CARRIER_SKETCH_DIM,
            CARRIER_SKETCH_SEED,
        )
        HOOK_IDENTITY = hook_identity_test(CONSTRUCTION_RECORDS[0]["record_id"])
        FORWARD_BENCHMARK = forward_benchmark(CONSTRUCTION_RECORDS[0]["record_id"])
        extract_discovery_baselines(CONSTRUCTION_RECORDS, "construction_discovery")
        memory_report("construction_discovery_complete")
    except Exception:
        record_failure("construction_discovery_baselines")
'''


mode_discovery = r'''# Discover two internal modes and select one block without contact labels.


def block_discovery_matrices(block):
    activations, outputs, groups = [], [], []
    for record in CONSTRUCTION_RECORDS:
        payload = load_discovery(record["record_id"])
        blocks = payload["blocks"].astype(int).tolist()
        index = blocks.index(int(block))
        activations.append(payload["carrier_sketches"][index].astype(np.float64))
        outputs.append(payload["output_train_sketch"].astype(np.float64))
        groups.extend([int(record["record_id"])] * ACTIONS_PER_STATE)
    return (
        np.concatenate(activations),
        np.concatenate(outputs),
        np.asarray(groups, dtype=np.int64),
    )


def state_mode_coverage(assignments, groups):
    rows = []
    for group in np.unique(groups):
        values = assignments[groups == group]
        rows.append(bool(np.any(values == 0) and np.any(values == 1)))
    return float(np.mean(rows))


def discover_all_blocks():
    rows = []
    fitted = {}
    for block in ACTIVE_DISCOVERY_BLOCKS:
        activations, outputs, groups = block_discovery_matrices(block)
        result = discover_mode_partition(
            activations,
            outputs,
            groups,
            seed=stable_seed(MODE_CLUSTER_SEED, block),
            pca_rank=MODE_PCA_RANK,
        )
        coverage = state_mode_coverage(result["assignments"], groups)
        energy = result["cluster_output_energy"]
        on = int(result["mode_on_cluster"])
        off = 1 - on
        energy_ratio = float(energy[on] / max(energy[off], 1e-12))
        score = float(
            result["separation"]
            * result["balance"]
            * coverage
            * max(np.log(max(energy_ratio, 1.0)), 0.0)
        )
        rows.append(
            {
                "block": int(block),
                "separation": float(result["separation"]),
                "balance": float(result["balance"]),
                "state_two_mode_coverage": coverage,
                "output_energy_ratio": energy_ratio,
                "cluster_0_count": int(result["counts"][0]),
                "cluster_1_count": int(result["counts"][1]),
                "selection_score": score,
                "contact_labels_seen": False,
            }
        )
        fitted[int(block)] = result
    selected_row = max(rows, key=lambda row: (row["selection_score"], -row["block"]))
    selected_block = int(selected_row["block"])
    selected = fitted[selected_block]
    atomic_npz(
        SUBSPACE_DIR / "frozen_mode_partition.npz",
        selected_block=np.asarray(selected_block, dtype=np.int64),
        mode_on_cluster=np.asarray(selected["mode_on_cluster"], dtype=np.int64),
        mean=selected["mean"],
        scale=selected["scale"],
        components=selected["components"],
        centroids=selected["centroids"],
        construction_assignments=selected["assignments"],
        construction_groups=block_discovery_matrices(selected_block)[2],
    )
    gate_pass = bool(
        selected_row["separation"] >= MIN_DISCOVERY_SEPARATION
        and selected_row["balance"] >= MIN_DISCOVERY_BALANCE
        and selected_row["state_two_mode_coverage"] >= MIN_DISCOVERY_STATE_COVERAGE
        and selected_row["output_energy_ratio"] >= MIN_DISCOVERY_OUTPUT_ENERGY_RATIO
    )
    payload = {
        "selected_block": selected_block,
        "selection_rule": "maximum frozen label-free separation-balance-coverage-energy score",
        "selected_row": selected_row,
        "rows": rows,
        "construction_gate_pass": gate_pass,
        "simulator_contact_labels_used": False,
        "evaluation_activations_seen": [],
        "partition_sha256": sha256_file(SUBSPACE_DIR / "frozen_mode_partition.npz"),
    }
    write_csv(ANALYSIS_DIR / "construction_mode_discovery_by_block.csv", rows)
    write_json(ANALYSIS_DIR / "construction_mode_selection.json", payload)
    return payload


if not PIPELINE_FAILED:
    try:
        MODE_SELECTION = discover_all_blocks()
        SELECTED_BLOCK = int(MODE_SELECTION["selected_block"])
        CONSTRUCTION_MODE_GATE_PASS = bool(MODE_SELECTION["construction_gate_pass"])
        print(json.dumps(MODE_SELECTION, indent=2))
    except Exception:
        record_failure("label_free_mode_discovery")
'''


factorization = r'''# Freeze the gate direction and an orthogonal effect subspace on construction data.


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
                initial,
                actions,
                PRIMARY_HORIZON,
                capture_blocks=[SELECTED_BLOCK],
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
            {
                "completed": index + 1,
                "total": len(records),
                "last_record_id": int(record["record_id"]),
            },
        )
        del initial, actions, predicted, captures, carrier
        gc.collect()
        torch.cuda.empty_cache()
    TIMINGS[f"{progress_name}_seconds"] = time.perf_counter() - started


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
        raise RuntimeError(f"ridge effect rank {keep} is below required rank {rank}")
    basis = left[:, : int(rank)].detach().cpu().numpy().astype(np.float64)
    singular_values = singular.detach().cpu().numpy().astype(np.float64)
    del x, y, gram, alpha, weight, left, singular
    torch.cuda.empty_cache()
    return basis, singular_values


def construction_factorization_matrices():
    count = 0
    total = np.zeros(EXPECTED_CARRIER_CHANNELS, dtype=np.float64)
    cross = np.zeros((EXPECTED_CARRIER_CHANNELS, EXPECTED_CARRIER_CHANNELS), dtype=np.float64)
    native_residuals, output_residuals, groups = [], [], []
    for record in CONSTRUCTION_RECORDS:
        payload = load_carrier(record["record_id"])
        carrier = payload["carrier"].astype(np.float64)
        channels = carrier.reshape(-1, carrier.shape[-1])
        count += len(channels)
        total += channels.sum(axis=0)
        cross += channels.T @ channels
        native_residuals.append(candidate_center(carrier))
        output_residuals.append(candidate_center(payload["output_train_sketch"]))
        groups.extend([int(record["record_id"])] * ACTIONS_PER_STATE)
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
    x_native_scale = np.concatenate(
        [value.reshape(ACTIONS_PER_STATE, -1) for value in whitened]
    )
    x = x_native_scale / np.sqrt(x_native_scale.shape[1])
    y = np.concatenate(output_residuals).astype(np.float64)
    output_scale = np.std(y, axis=0, ddof=1)
    positive = output_scale[output_scale > 1e-12]
    if not len(positive):
        raise RuntimeError("construction output sketch has zero variance")
    output_scale = np.maximum(output_scale, np.median(positive) * 1e-3)
    y /= output_scale[None]
    return x, x_native_scale, y, np.asarray(groups, dtype=np.int64), metric, output_scale


def shuffled_mode_labels(mode_on, groups):
    shuffled = np.asarray(mode_on, dtype=bool).copy()
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        permutation = fixed_derangement(
            len(indices), stable_seed(SHUFFLE_SEED, int(group), "mode_labels")
        )
        shuffled[indices] = shuffled[indices][permutation]
    return shuffled


def fit_and_freeze_factorization():
    x, x_native, y, groups, metric, output_scale = construction_factorization_matrices()
    with np.load(SUBSPACE_DIR / "frozen_mode_partition.npz") as partition:
        assignments = partition["construction_assignments"].astype(np.int64)
        mode_on_cluster = int(partition["mode_on_cluster"])
        partition_groups = partition["construction_groups"].astype(np.int64)
    if not np.array_equal(groups, partition_groups):
        raise RuntimeError("construction partition rows do not match full carriers")
    mode_on = assignments == mode_on_cluster
    gate = difference_of_means_direction(x_native, mode_on)

    kernel = x @ x.T
    ridge = grouped_kernel_ridge_cv(kernel, y, groups, RIDGE_MULTIPLIERS)
    raw_effect, singular = fit_basis_gpu(
        x, y, ridge["penalty"], ACTIVE_EFFECT_RANK + 1
    )
    effect = orthogonalize_basis(raw_effect, gate)
    if effect.shape[1] < ACTIVE_EFFECT_RANK:
        raise RuntimeError("gate removal left too few effect directions")
    effect = effect[:, :ACTIVE_EFFECT_RANK]

    shuffled_labels = shuffled_mode_labels(mode_on, groups)
    shuffled_raw = difference_of_means_direction(x_native, shuffled_labels)
    excluded = np.column_stack([gate, effect])
    shuffled_gate = orthogonalize_basis(shuffled_raw[:, None], excluded)[:, 0]
    random_gate = random_subspace_in_span(
        x,
        rank=1,
        seed=stable_seed(NULL_ROOT_SEED, "random_gate"),
        orthogonal_to=excluded,
    )[:, 0]

    # Float64 re-orthogonalization before serialization keeps the factorial
    # edits algebraically separate after the GPU/CPU round trip.
    effect = nested_orthonormalize_basis(effect)
    gate = orthogonalize_basis(gate[:, None], effect)[:, 0]
    control_excluded = np.column_stack([gate, effect])
    shuffled_gate = orthogonalize_basis(shuffled_gate[:, None], control_excluded)[:, 0]
    random_gate = orthogonalize_basis(random_gate[:, None], control_excluded)[:, 0]

    destination = SUBSPACE_DIR / "frozen_hybrid_factorization.npz"
    atomic_npz(
        destination,
        selected_block=np.asarray(SELECTED_BLOCK, dtype=np.int64),
        gate_direction=gate.astype(np.float32),
        effect_basis=effect.astype(np.float32),
        shuffled_gate_direction=shuffled_gate.astype(np.float32),
        random_gate_direction=random_gate.astype(np.float32),
        channel_mean=metric["mean"],
        channel_covariance=metric["covariance"],
        channel_square_root=metric["square_root"],
        channel_inverse_square_root=metric["inverse_square_root"],
        output_scale=output_scale,
        effect_singular_values=singular,
    )
    write_csv(ANALYSIS_DIR / "effect_ridge_group_cv.csv", ridge["rows"])
    orthogonality = {
        "gate_effect_max_abs": float(np.max(np.abs(gate @ effect))),
        "shuffled_effect_max_abs": float(np.max(np.abs(shuffled_gate @ effect))),
        "random_effect_max_abs": float(np.max(np.abs(random_gate @ effect))),
        "gate_shuffled_abs": float(abs(gate @ shuffled_gate)),
        "gate_random_abs": float(abs(gate @ random_gate)),
    }
    manifest = {
        "selected_block": int(SELECTED_BLOCK),
        "effect_rank": int(ACTIVE_EFFECT_RANK),
        "ambient_dimension": int(x.shape[1]),
        "ridge_multiplier": ridge["selected_multiplier"],
        "ridge_penalty": ridge["penalty"],
        "channel_condition_number": metric["condition_number"],
        "orthogonality": orthogonality,
        "factorization_sha256": sha256_file(destination),
        "partition_sha256": sha256_file(SUBSPACE_DIR / "frozen_mode_partition.npz"),
        "construction_record_ids": sorted(np.unique(groups).astype(int).tolist()),
        "evaluation_activations_seen": [],
        "contact_labels_used": False,
        "jacobians_computed": False,
    }
    if max(orthogonality.values()) > 1e-4:
        raise RuntimeError(f"factorization orthogonality failed: {orthogonality}")
    write_json(SUBSPACE_DIR / "factorization_manifest.json", manifest)
    write_json(
        SUBSPACE_DIR / "factorization_freeze.json",
        {
            "frozen_before_evaluation_activations": True,
            "source_identity": SOURCE_IDENTITY,
            "design_freeze_sha256": sha256_file(DESIGN_DIR / "design_freeze.json"),
            "mode_selection_sha256": sha256_file(ANALYSIS_DIR / "construction_mode_selection.json"),
            "manifest": manifest,
        },
    )
    return manifest


if not PIPELINE_FAILED:
    try:
        extract_full_carriers(CONSTRUCTION_RECORDS, "construction_full_carriers")
        FACTORIZATION_MANIFEST = fit_and_freeze_factorization()
        print(json.dumps(FACTORIZATION_MANIFEST, indent=2))
        memory_report("construction_factorization_frozen")
    except Exception:
        record_failure("construction_factorization")
'''


evaluation_open = r'''# Open evaluation activations, freeze model-only pairs, then reveal contact labels.


def load_partition():
    with np.load(SUBSPACE_DIR / "frozen_mode_partition.npz") as payload:
        return {name: payload[name].copy() for name in payload.files}


def frozen_mode_assignments(record, partition):
    discovery = load_discovery(record["record_id"])
    blocks = discovery["blocks"].astype(int).tolist()
    sketch = discovery["carrier_sketches"][blocks.index(SELECTED_BLOCK)].astype(np.float64)
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
        }
        for slot in range(ACTIVE_PAIRS_PER_STATE)
    ]


if not PIPELINE_FAILED:
    try:
        if not (SUBSPACE_DIR / "factorization_freeze.json").exists():
            raise RuntimeError("factorization must freeze before evaluation opens")
        extract_discovery_baselines(EVALUATION_RECORDS, "evaluation_discovery")
        extract_full_carriers(EVALUATION_RECORDS, "evaluation_full_carriers")
        PARTITION = load_partition()
        EVALUATION_ASSIGNMENTS = {}
        EVALUATION_PAIRS = []
        for record in EVALUATION_RECORDS:
            assignments, transformed = frozen_mode_assignments(record, PARTITION)
            EVALUATION_ASSIGNMENTS[str(int(record["record_id"]))] = assignments.tolist()
            EVALUATION_PAIRS.extend(
                select_model_only_pairs(record, assignments, transformed, PARTITION)
            )
        pair_freeze = {
            "assignments": EVALUATION_ASSIGNMENTS,
            "pairs": EVALUATION_PAIRS,
            "selection_inputs": ["frozen activation sketch", "frozen partition"],
            "simulator_contact_labels_used": False,
            "pair_count": len(EVALUATION_PAIRS),
        }
        write_json(DESIGN_DIR / "evaluation_model_pair_freeze.json", pair_freeze)
        PAIR_FREEZE_SHA256 = sha256_file(DESIGN_DIR / "evaluation_model_pair_freeze.json")

        # Only after the complete assignment file is frozen may contact labels
        # be read for physical interpretation and aligned-subset scoring.
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
        CONTACT_ALIGNMENT["physically_aligned_pairs"] = int(
            sum(value["physically_aligned"] for value in PAIR_TRUTH_MAP.values())
        )
        write_csv(EVIDENCE_DIR / "heldout_mode_contact_rows.csv", label_rows)
        write_json(EVIDENCE_DIR / "heldout_contact_alignment.json", CONTACT_ALIGNMENT)
        EVALUATION_OPENED = True
        memory_report("evaluation_assignments_and_contact_reveal_complete")
        print(json.dumps(CONTACT_ALIGNMENT, indent=2))
    except Exception:
        EVALUATION_OPENED = False
        record_failure("evaluation_open_and_contact_reveal")
'''


factorial_interventions = r'''# Run the gate-by-effect factorial and matched causal controls.


def load_factorization():
    with np.load(SUBSPACE_DIR / "frozen_hybrid_factorization.npz") as payload:
        return {name: payload[name].copy() for name in payload.files}


def whiten_carrier(values, factorization):
    return transform_primal_channels(
        np.asarray(values, dtype=np.float64),
        factorization["channel_inverse_square_root"],
    )


def native_edit(values, factorization):
    return inverse_transform_primal_channels(
        np.asarray(values, dtype=np.float64), factorization["channel_square_root"]
    )


def vector_transfer_coefficient(baseline, patched, donor):
    base = np.asarray(baseline, dtype=np.float64).reshape(-1)
    edit = np.asarray(patched, dtype=np.float64).reshape(-1)
    target_value = np.asarray(donor, dtype=np.float64).reshape(-1)
    target = target_value - base
    denominator = float(np.sum(target**2))
    if denominator <= 1e-12:
        return math.nan
    return float(np.sum((edit - base) * target) / denominator)


def intervention_path(record_id):
    return INTERVENTION_DIR / f"evaluation_{int(record_id):04d}.json"


def run_patch(initial, actions, delta_white, factorization):
    delta_native = native_edit(delta_white, factorization)
    delta = torch.as_tensor(delta_native, device="cuda", dtype=torch.float32)
    with torch.inference_mode():
        patched, _, _ = forward_with_carriers(
            initial,
            actions,
            PRIMARY_HORIZON,
            capture_blocks=[SELECTED_BLOCK],
            intervention={"block": SELECTED_BLOCK, "delta": delta},
        )
        result = EVAL_OUTPUT_PROJECTOR(patched).cpu().numpy()
    PROVENANCE_COUNTS["patched_forwards_generated"] += 1
    del patched, delta
    return result


def pair_factorial_rows(record, pair, payload, initial, actions, factorization):
    base_index = int(pair["base_index"])
    donor_index = int(pair["donor_index"])
    output = payload["output_eval_sketch"].astype(np.float64)
    white = whiten_carrier(payload["carrier"], factorization)
    baseline = output[base_index]
    donor = output[donor_index]

    learned_gate = projected_pair_delta(
        white, base_index, donor_index, factorization["gate_direction"]
    )
    effect = projected_pair_delta(
        white, base_index, donor_index, factorization["effect_basis"]
    )
    shuffled_gate = projected_pair_delta(
        white, base_index, donor_index, factorization["shuffled_gate_direction"]
    )
    random_gate = projected_pair_delta(
        white, base_index, donor_index, factorization["random_gate_direction"]
    )
    if np.linalg.norm(learned_gate) <= 1e-12 or np.linalg.norm(effect) <= 1e-12:
        raise RuntimeError(f"degenerate learned pair edit {pair['pair_id']}")
    shuffled_gate = norm_match(shuffled_gate, learned_gate)
    random_gate = norm_match(random_gate, learned_gate)

    effect_output = run_patch(initial, actions, effect, factorization)
    gate_deltas = {
        "learned_gate": learned_gate,
        "shuffled_gate": shuffled_gate,
        "random_gate": random_gate,
    }
    rows = []
    gate_outputs = {}
    both_outputs = {}
    for condition, gate_delta in gate_deltas.items():
        gate_outputs[condition] = run_patch(initial, actions, gate_delta, factorization)
        both_outputs[condition] = run_patch(
            initial, actions, gate_delta + effect, factorization
        )

    full = np.zeros_like(white)
    full[base_index] = white[donor_index] - white[base_index]
    full_output = run_patch(initial, actions, full, factorization)
    physical = PAIR_TRUTH_MAP[pair["pair_id"]]
    for condition, gate_delta in gate_deltas.items():
        metrics = factorial_interaction_metrics(
            baseline,
            gate_outputs[condition][base_index],
            effect_output[base_index],
            both_outputs[condition][base_index],
            donor,
        )
        rows.append(
            {
                "pair_id": pair["pair_id"],
                "record_id": int(record["record_id"]),
                "trajectory_id": int(record["trajectory_id"]),
                "slot": int(pair["slot"]),
                "base_index": base_index,
                "donor_index": donor_index,
                "condition": condition,
                "selected_block": int(SELECTED_BLOCK),
                "base_contact": physical["base_contact"],
                "donor_contact": physical["donor_contact"],
                "physically_aligned": physical["physically_aligned"],
                "gate_edit_whitened_norm": float(np.linalg.norm(gate_delta)),
                "effect_edit_whitened_norm": float(np.linalg.norm(effect)),
                "gate_effect_edit_cosine": float(
                    np.sum(gate_delta * effect)
                    / max(np.linalg.norm(gate_delta) * np.linalg.norm(effect), 1e-12)
                ),
                "full_swap_coefficient": vector_transfer_coefficient(
                    baseline, full_output[base_index], donor
                ),
                **metrics,
            }
        )
    expected_forwards = PATCHED_FORWARDS_PER_PAIR
    observed_forwards = 1 + 2 * len(gate_deltas) + 1
    if observed_forwards != expected_forwards:
        raise RuntimeError(
            f"factorial forward contract changed: {observed_forwards} != {expected_forwards}"
        )
    return rows


def run_record_factorials(record, pairs, factorization):
    destination = intervention_path(record["record_id"])
    if destination.exists():
        PROVENANCE_COUNTS["cache_hits"] += 1
        raise RuntimeError(f"fresh intervention shard already exists: {destination}")
    payload = load_carrier(record["record_id"])
    initial, actions = state_model_inputs(record["record_id"])
    rows = []
    for pair in pairs:
        rows.extend(
            pair_factorial_rows(
                record, pair, payload, initial, actions, factorization
            )
        )
    write_json(destination, rows)
    PROVENANCE_COUNTS["intervention_generated"] += 1
    del initial, actions
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def run_all_factorials():
    started = time.perf_counter()
    factorization = load_factorization()
    by_record = defaultdict(list)
    for pair in EVALUATION_PAIRS:
        by_record[int(pair["record_id"])].append(pair)
    rows = []
    active_records = [
        record for record in EVALUATION_RECORDS
        if int(record["record_id"]) in by_record
    ]
    for index, record in enumerate(active_records):
        rows.extend(
            run_record_factorials(
                record, by_record[int(record["record_id"])], factorization
            )
        )
        write_json(
            OUT / "factorial_progress.json",
            {
                "completed_records": index + 1,
                "total_records": len(active_records),
                "completed_pairs": int(len(rows) // 3),
                "patched_forwards_generated": PROVENANCE_COUNTS["patched_forwards_generated"],
            },
        )
    TIMINGS["factorial_interventions_seconds"] = time.perf_counter() - started
    write_csv(EVIDENCE_DIR / "factorial_interaction_rows.csv", rows)
    return rows


if not PIPELINE_FAILED and EVALUATION_OPENED:
    try:
        FACTORIAL_ROWS = run_all_factorials()
        memory_report("factorial_interventions_complete")
    except Exception:
        record_failure("factorial_interventions")
'''


decision = r'''# Apply Stage 22 frozen scientific gates.


def condition_values(rows, condition, key):
    return np.asarray(
        [
            row[key] for row in rows
            if row["condition"] == condition and row["physically_aligned"]
        ],
        dtype=np.float64,
    )


def condition_groups(rows, condition):
    return np.asarray(
        [
            row["trajectory_id"] for row in rows
            if row["condition"] == condition and row["physically_aligned"]
        ],
        dtype=np.int64,
    )


def paired_gain(rows, comparator, key="interaction_coefficient"):
    learned = {
        row["pair_id"]: row for row in rows
        if row["condition"] == "learned_gate" and row["physically_aligned"]
    }
    control = {
        row["pair_id"]: row for row in rows
        if row["condition"] == comparator and row["physically_aligned"]
    }
    identifiers = sorted(set(learned) & set(control))
    values = np.asarray(
        [learned[pair][key] - control[pair][key] for pair in identifiers],
        dtype=np.float64,
    )
    groups = np.asarray(
        [learned[pair]["trajectory_id"] for pair in identifiers], dtype=np.int64
    )
    return identifiers, values, groups


def bootstrap_payload(values, groups, label):
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


def evaluate_hybrid_gate(rows):
    learned = condition_values(rows, "learned_gate", "interaction_coefficient")
    learned_cosine = condition_values(rows, "learned_gate", "interaction_cosine")
    learned_full = condition_values(rows, "learned_gate", "full_swap_coefficient")
    learned_groups = condition_groups(rows, "learned_gate")
    if not len(learned):
        raise RuntimeError("no physically aligned learned-gate pairs")
    _, random_gain, random_groups = paired_gain(rows, "random_gate")
    _, shuffled_gain, shuffled_groups = paired_gain(rows, "shuffled_gate")
    random_summary = bootstrap_payload(random_gain, random_groups, "random_gain")
    shuffled_summary = bootstrap_payload(shuffled_gain, shuffled_groups, "shuffled_gain")
    learned_summary = bootstrap_payload(learned, learned_groups, "learned_interaction")
    required_positive = int(np.ceil(REQUIRED_POSITIVE_GAIN_FRACTION * len(learned)))
    payload = {
        "physically_aligned_pairs": int(len(learned)),
        "learned_interaction": learned_summary,
        "mean_learned_interaction_cosine": float(np.mean(learned_cosine)),
        "mean_full_swap_coefficient": float(np.mean(learned_full)),
        "gain_over_random": random_summary,
        "gain_over_shuffled": shuffled_summary,
        "positive_gain_over_random_pairs": int(np.sum(random_gain > 0)),
        "positive_gain_over_shuffled_pairs": int(np.sum(shuffled_gain > 0)),
        "required_positive_pairs": required_positive,
        "random_gain_sign_test_p": exact_positive_sign_test(random_gain),
        "shuffled_gain_sign_test_p": exact_positive_sign_test(shuffled_gain),
    }
    payload["causal_interaction_pass"] = bool(
        len(learned) >= ACTIVE_MIN_PHYSICALLY_ALIGNED_PAIRS
        and learned_summary["mean"] >= MIN_INTERACTION_COEFFICIENT
        and learned_summary["lower"] > 0
        and payload["mean_learned_interaction_cosine"] >= MIN_INTERACTION_COSINE
        and payload["mean_full_swap_coefficient"] >= MIN_FULL_SWAP_COEFFICIENT
        and random_summary["mean"] >= MIN_INTERACTION_GAIN_OVER_RANDOM
        and random_summary["lower"] > 0
        and shuffled_summary["mean"] >= MIN_INTERACTION_GAIN_OVER_SHUFFLED
        and shuffled_summary["lower"] > 0
        and payload["positive_gain_over_random_pairs"] >= required_positive
        and payload["positive_gain_over_shuffled_pairs"] >= required_positive
    )
    return payload


if not PIPELINE_FAILED:
    try:
        SOURCE_EXECUTION_VERIFIED = verify_executed_notebook_through(
            "# Apply Stage 22 frozen scientific gates."
        )
        physical_mode_pass = bool(
            CONTACT_ALIGNMENT["balanced_accuracy"] >= MIN_CONTACT_BALANCED_ACCURACY
            and CONTACT_ALIGNMENT["matthews_correlation"] >= MIN_CONTACT_MCC
        )
        aligned_pair_count = int(CONTACT_ALIGNMENT["physically_aligned_pairs"])
        if aligned_pair_count:
            HYBRID_GATE_RESULT = evaluate_hybrid_gate(FACTORIAL_ROWS)
        else:
            HYBRID_GATE_RESULT = {
                "physically_aligned_pairs": 0,
                "causal_interaction_pass": False,
                "reason": "model-only pair assignment produced no off-contact to contact pairs",
            }
        construction_pass = bool(MODE_SELECTION["construction_gate_pass"])
        if RUN_MODE == "smoke":
            candidate_status = "SMOKE_ONLY"
        elif not construction_pass:
            candidate_status = "NO_STABLE_INTERNAL_MODE_PARTITION"
        elif not physical_mode_pass:
            candidate_status = "INTERNAL_MODE_NOT_PHYSICAL_CONTACT"
        elif HYBRID_GATE_RESULT["causal_interaction_pass"]:
            candidate_status = "EVENT_GATED_CAUSAL_INTERACTION_CONFIRMED"
        else:
            candidate_status = "PHYSICAL_MODE_WITHOUT_CAUSAL_INTERACTION"
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
            "construction_mode_discovery": MODE_SELECTION,
            "heldout_contact_alignment": CONTACT_ALIGNMENT,
            "factorial_causal_interaction": HYBRID_GATE_RESULT,
            "claim_boundary": {
                "complete_hybrid_program_extracted": False,
                "symbolic_rule_claim_authorized": False,
                "physical_truth_claim_authorized": False,
                "planning_utility_claim_authorized": False,
                "jacobian_or_local_linearization_claim_authorized": False,
                "authorized_claim": "two-mode internal gate-by-effect interaction in one frozen JEPA-WM",
            },
        }
        write_json(OUT / "stage22_decision.json", DECISION_PAYLOAD)

        figure, axes = plt.subplots(1, 3, figsize=(13, 4))
        mode_rows = MODE_SELECTION["rows"]
        axes[0].plot(
            [row["block"] for row in mode_rows],
            [row["selection_score"] for row in mode_rows],
            marker="o",
        )
        axes[0].set(xlabel="predictor block", ylabel="label-free score", title="Mode discovery")
        labels = ["accuracy", "balanced_accuracy", "matthews_correlation"]
        axes[1].bar(labels, [CONTACT_ALIGNMENT[key] for key in labels])
        axes[1].tick_params(axis="x", rotation=25)
        axes[1].set_ylim(-0.1, 1.0)
        axes[1].set_title("Held-out physical alignment")
        interaction_means = [
            float(np.mean(condition_values(FACTORIAL_ROWS, condition, "interaction_coefficient")))
            for condition in ["learned_gate", "shuffled_gate", "random_gate"]
        ]
        axes[2].bar(["learned", "shuffled", "random"], interaction_means)
        axes[2].axhline(0, color="black", linewidth=0.8)
        axes[2].set(ylabel="interaction coefficient", title="Gate × effect causal test")
        figure.tight_layout()
        figure.savefig(PLOT_DIR / "stage22_hybrid_gate_summary.png", dpi=180)
        plt.close(figure)
        print(json.dumps(DECISION_PAYLOAD, indent=2))
    except Exception:
        record_failure("decision_and_plots")
        DECISION_PAYLOAD = {"status": "INCONCLUSIVE", "failure": FAILURE_MESSAGE}
else:
    DECISION_PAYLOAD = {"status": "INCONCLUSIVE", "failure": FAILURE_MESSAGE}

if not (OUT / "stage22_decision.json").exists():
    write_json(OUT / "stage22_decision.json", DECISION_PAYLOAD)
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
    if path.name.startswith("stage22_hybrid_gate_result_bundle_"):
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

archive_base = OUT / f"stage22_hybrid_gate_result_bundle_{RUN_SIGNATURE[:12]}"
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
    design,
    truth_generation,
    discovery_baselines,
    mode_discovery,
    factorization,
    evaluation_open,
    factorial_interventions,
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
    code(design),
    code(truth_generation),
    code(discovery_baselines),
    code(mode_discovery),
    code(factorization),
    code(evaluation_open),
    code(factorial_interventions),
    code(decision),
    code(packaging),
]
for index, cell in enumerate(cells):
    cell["id"] = f"stage22-{index:02d}"

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
