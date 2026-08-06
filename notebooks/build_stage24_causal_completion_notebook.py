import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "24_causal_completion_rank.ipynb"
BASE = json.loads((ROOT / "23_causal_mode_manifold_operator_switch.ipynb").read_text())
NUMERICAL = ROOT.parent / "src/cf_faithfulness/stage24_causal_completion.py"


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


introduction = r'''# Stage 24: causal completion rank

Stage 23 established a sharp readout–mechanism dissociation.  Its exact
eight-dimensional intervention moved every held-out activation into the frozen
contact-like cluster, yet reproduced only 0.025% of the native contact-operator
change.  The learned edit was only 2.8% of a natural off→on activation
difference, leaving open whether the mode coordinates are a diagnostic address
for a compact missing mechanism or merely a passive correlate.

This notebook defines and measures **Causal Completion Rank (CCR)**.  For each
natural construction pair at predictor block 1, write

\[
d_i=h_i^+-h_i^-,\qquad r_i=d_i-\delta_i^q,
\]

where `delta_q` is the exact Stage 23 minimum-norm mode edit.  Because
`A.T @ r_i = 0`, the residual changes no frozen mode coordinate.  An uncentered
construction-only SVD freezes a nested residual basis `V`.  On fresh held-out
pairs, rank-`k` completion applies

\[
\delta_i(k)=\delta_i^q+V_kV_k^\top r_i,
\qquad k\in\{0,4,8,16,32,64\}.
\]

The primary outcome is transfer toward the exact native-on symmetric finite
response operator.  `CCR_0.8` is the smallest rank whose trajectory-bootstrap
95% lower bound reaches 0.8.  Same-mode residual and random mode-null bases are
matched controls.  Every basis is frozen before evaluation activations open;
pair selection remains model-only; physical contact labels are revealed only
after the complete evaluation-pair file is sealed.

Possible outcomes are discriminating.  A small CCR identifies a compact
mechanism payload missing from the readable mode address.  A gradual curve
supports a distributed contact manifold.  Failure through rank 64 establishes
that no compact shared linear completion of the tested form recreates the
operator.  This is a source-bound, forward-pass-only experiment: no Jacobian,
JVP, VJP, gradient probe, or model-weight update is used.  Return
`stage24_causal_completion_result_bundle_<signature>.zip`.
'''


configuration = r'''# SINGLE CONFIGURATION BLOCK
# Required Colab secrets for a source-bound pilot:
# STAGE24_RUN_MODE=pilot
# STAGE24_SOURCE_COMMIT=<full 40-hex commit from the Colab handoff>
# STAGE24_RUN_NONCE=<new unique label, e.g. causal_completion_20260805_a>
RUN_MODE = "smoke"
EXPERIMENT_SOURCE_REF = ""
RUN_NONCE = "smoke"
try:
    from google.colab import userdata as _colab_userdata

    RUN_MODE = str(_colab_userdata.get("STAGE24_RUN_MODE") or RUN_MODE).strip().lower()
    EXPERIMENT_SOURCE_REF = str(
        _colab_userdata.get("STAGE24_SOURCE_COMMIT") or EXPERIMENT_SOURCE_REF
    ).strip()
    RUN_NONCE = str(
        _colab_userdata.get("STAGE24_RUN_NONCE") or RUN_NONCE
    ).strip()
except Exception:
    pass

if RUN_MODE == "pilot":
    if RUN_NONCE in {"", "smoke"}:
        raise ValueError("pilot mode requires a unique STAGE24_RUN_NONCE")
    if not all(value.isalnum() or value in "-_" for value in RUN_NONCE):
        raise ValueError("STAGE24_RUN_NONCE may contain only letters, numbers, '-' and '_'")

MOUNT_DRIVE = True
DOWNLOAD_RESULTS = True
CONTINUE_AFTER_BENCHMARK = True
MAX_ESTIMATED_TOTAL_MINUTES = 180.0
FRESH_RUN_REQUIRED = True

OUTPUT_DIR = "/content/counterfactual_faithfulness_stage24_causal_completion"
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/counterfactual_faithfulness_stage24_causal_completion"
UPSTREAM_STAGE23_DIR = "/content/drive/MyDrive/counterfactual_faithfulness_stage23_mode_operator"
UPSTREAM_STAGE23_RUN_SUFFIX = "d47dee8b6789"

PROTOCOL_ID = "stage24-causal-completion-rank-v1"
NOTEBOOK_PROTOCOL_SHA256 = "__PROTOCOL_DIGEST__"
EVIDENCE_STATUS = "CONFIRMATORY_ONLY_IF_SOURCE_BOUND_FRESH_UPSTREAM_BOUND_AND_EVALUATION_SEALED"
EXPERIMENT_REPOSITORY = "grewalsk/counterfactual-faithfulness-research"
EXPERIMENT_NOTEBOOK_PATH = "notebooks/24_causal_completion_rank.ipynb"
EXPERIMENT_BUILDER_PATH = "notebooks/build_stage24_causal_completion_notebook.py"
EXPERIMENT_NUMERICAL_PATH = "src/cf_faithfulness/stage24_causal_completion.py"

EXPECTED_STAGE23_SOURCE_COMMIT = "5b8a9a9cd360f08148a32ec6cfd6b8f39ab4e792"
EXPECTED_STAGE23_STATUS = "MODE_FLIPS_WITHOUT_OPERATOR_SWITCH"
EXPECTED_STAGE23_PROTOCOL_ID = "stage23-causal-mode-manifold-operator-switch-v1"
EXPECTED_STAGE23_PARTITION_SHA256 = "0f9e376e9d874f1b2429e62b119cf56ea77e3458b8a44c1d4575ed73c988697f"
EXPECTED_STAGE23_GEOMETRY_SHA256 = "3598cc57768077550913cc2008baf870c55288089b8570d5ae96c523956aa4ff"
EXPECTED_STAGE23_SELECTED_BLOCK = 1

SEED = 24101
DESIGN_SEED = 24137
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

CONSTRUCTION_POOL_TRAJECTORIES = list(range(1700, 1828))
EVALUATION_POOL_TRAJECTORIES = list(range(1900, 2028))
CONSTRUCTION_TRAJECTORY_TARGET = 64
EVALUATION_TRAJECTORY_TARGET = 64
TASK_ID_OFFSET = 8000

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
TRAIN_OUTPUT_SKETCH_SEED = 24161
EVAL_OUTPUT_SKETCH_SEED = 23183
CARRIER_SKETCH_SEED = 22197
RESPONSE_SKETCH_SEED = 23203
FINITE_PROBE_DOSE = 0.5
TRANSPORT_SOLVE_RIDGE = 0.0
COMPLETION_RANKS = [0, 4, 8, 16, 32, 64]
COMPLETION_BASIS_RANK = 64
CONSTRUCTION_PAIRS_PER_STATE = 3
PAIRS_PER_STATE = 1
BASIS_CONDITIONS = ["completion", "same_mode", "random"]
FULL_CONTEXT_COUNT = 18
PATCHED_FORWARDS_PER_PAIR = 289
RANDOM_BASIS_SEED = 24231
PERMUTATION_SEED = 24233
BOOTSTRAP_SEED = 24269
BOOTSTRAP_DRAWS = 10000
MAX_ZERO_EDIT_ERROR = 1e-6
MAX_RESIDUAL_MODE_DRIFT = 1e-7

MIN_CONTACT_BALANCED_ACCURACY = 0.65
MIN_CONTACT_MCC = 0.30
MIN_PHYSICALLY_ALIGNED_PAIRS = 40
MIN_NATIVE_FULL_SWAP_COEFFICIENT = 0.60
MIN_MODE_FLIP_RATE = 0.80
MAX_MODE_COORDINATE_RESIDUAL = 1e-5
MIN_NONDEGENERATE_TARGET_FRACTION = 0.80
TARGET_ENERGY_FLOOR = 1e-10
COMPLETION_TRANSFER_THRESHOLD = 0.80
MIN_GAIN_OVER_CONTROL = 0.20
REQUIRED_POSITIVE_GAIN_FRACTION = 0.65

if RUN_MODE == "smoke":
    ACTIVE_CONSTRUCTION_POOL_TRAJECTORIES = CONSTRUCTION_POOL_TRAJECTORIES[:12]
    ACTIVE_EVALUATION_POOL_TRAJECTORIES = EVALUATION_POOL_TRAJECTORIES[:12]
    ACTIVE_CONSTRUCTION_TARGET = 4
    ACTIVE_EVALUATION_TARGET = 3
    ACTIVE_DISCOVERY_BLOCKS = DISCOVERY_BLOCKS
    ACTIVE_CONTENT_RANK = 64
    ACTIVE_PROBE_COUNT = 2
    ACTIVE_COMPLETION_RANKS = [0, 4]
    ACTIVE_COMPLETION_BASIS_RANK = 4
    ACTIVE_CONSTRUCTION_PAIRS_PER_STATE = 1
    ACTIVE_PAIRS_PER_STATE = 1
    ACTIVE_CONTEXT_COUNT = 6
    ACTIVE_PATCHED_FORWARDS_PER_PAIR = 25
    ACTIVE_BOOTSTRAP_DRAWS = 64
    ACTIVE_MIN_PHYSICALLY_ALIGNED_PAIRS = 1
elif RUN_MODE == "pilot":
    ACTIVE_CONSTRUCTION_POOL_TRAJECTORIES = CONSTRUCTION_POOL_TRAJECTORIES
    ACTIVE_EVALUATION_POOL_TRAJECTORIES = EVALUATION_POOL_TRAJECTORIES
    ACTIVE_CONSTRUCTION_TARGET = CONSTRUCTION_TRAJECTORY_TARGET
    ACTIVE_EVALUATION_TARGET = EVALUATION_TRAJECTORY_TARGET
    ACTIVE_DISCOVERY_BLOCKS = DISCOVERY_BLOCKS
    ACTIVE_CONTENT_RANK = 64
    ACTIVE_PROBE_COUNT = 8
    ACTIVE_COMPLETION_RANKS = COMPLETION_RANKS
    ACTIVE_COMPLETION_BASIS_RANK = COMPLETION_BASIS_RANK
    ACTIVE_CONSTRUCTION_PAIRS_PER_STATE = CONSTRUCTION_PAIRS_PER_STATE
    ACTIVE_PAIRS_PER_STATE = PAIRS_PER_STATE
    ACTIVE_CONTEXT_COUNT = FULL_CONTEXT_COUNT
    ACTIVE_PATCHED_FORWARDS_PER_PAIR = PATCHED_FORWARDS_PER_PAIR
    ACTIVE_BOOTSTRAP_DRAWS = BOOTSTRAP_DRAWS
    ACTIVE_MIN_PHYSICALLY_ALIGNED_PAIRS = MIN_PHYSICALLY_ALIGNED_PAIRS
else:
    raise ValueError(
        "STAGE24_RUN_MODE must contain only smoke or pilot; "
        f"received {RUN_MODE!r}"
    )

ACTIVE_CONTEXTS = ["off", "native_on", "q_only"] + [
    f"{condition}_rank_{rank}"
    for rank in ACTIVE_COMPLETION_RANKS[1:]
    for condition in BASIS_CONDITIONS
]
if len(ACTIVE_CONTEXTS) != ACTIVE_CONTEXT_COUNT:
    raise RuntimeError("completion context count changed")

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
assert SELECTED_BLOCK == EXPECTED_STAGE23_SELECTED_BLOCK < min(DOWNSTREAM_BLOCKS)
assert COMPLETION_RANKS == sorted(set(COMPLETION_RANKS))
assert COMPLETION_RANKS[0] == 0 and COMPLETION_RANKS[-1] == COMPLETION_BASIS_RANK
assert FULL_CONTEXT_COUNT == 3 + 3 * (len(COMPLETION_RANKS) - 1)
assert PATCHED_FORWARDS_PER_PAIR == 1 + 2 * 8 * FULL_CONTEXT_COUNT
assert CARRIER_SKETCH_SEED == 22197
assert not set(CONSTRUCTION_POOL_TRAJECTORIES) & set(EVALUATION_POOL_TRAJECTORIES)
'''
configuration += "\n\nPROTOCOL_CONFIG_KEYS = " + repr(
    assigned_uppercase_names(configuration)
) + "\n"


installation = base_source(2)


setup = base_source(3)
setup = setup.replace("Stage 23", "Stage 24").replace("STAGE23", "STAGE24")
setup = setup.replace("stage23_mode_operator", "stage24_causal_completion")


analysis_helpers = base_source(4)
analysis_helpers += "\n\n\n" + function_sources(
    NUMERICAL.read_text(),
    [
        "completion_residual",
        "orthonormal_residual_basis",
        "completion_edit",
        "native_reconstruction_fraction",
        "causal_completion_rank",
        "paired_completion_gain",
    ],
)


model_helpers = base_source(5).replace("stage23-jepa-wms", "stage24-jepa-wms")


upstream_import = r'''# Bind the exact Stage 23 negative and its frozen geometry.


def locate_and_verify_stage23():
    root = Path(UPSTREAM_STAGE23_DIR)
    candidate = root / f"pilot_{UPSTREAM_STAGE23_RUN_SUFFIX}"
    if not candidate.is_dir():
        raise RuntimeError(
            f"missing Stage 23 Drive run {candidate}; the compact ZIP is insufficient "
            "because Stage 24 requires the frozen geometry NPZ"
        )
    required = {
        "source": candidate / "source_identity.json",
        "decision": candidate / "stage23_decision.json",
        "stage22_binding": candidate / "subspaces/stage22_upstream_binding.json",
        "partition": candidate / "subspaces/frozen_mode_partition.npz",
        "geometry": candidate / "subspaces/frozen_mode_operator_geometry.npz",
        "geometry_freeze": candidate / "subspaces/mode_operator_geometry_freeze.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Stage 23 upstream artifacts are incomplete: {missing}")
    source = json.loads(required["source"].read_text())
    decision = json.loads(required["decision"].read_text())
    stage22_binding = json.loads(required["stage22_binding"].read_text())
    geometry_freeze = json.loads(required["geometry_freeze"].read_text())
    partition_sha = sha256_file(required["partition"])
    geometry_sha = sha256_file(required["geometry"])
    causal_result = decision.get("causal_mode_operator_test", {})
    checks = {
        "source_commit": source.get("resolved_commit") == EXPECTED_STAGE23_SOURCE_COMMIT,
        "source_execution_verified": bool(source.get("confirmation_eligible", False)),
        "protocol": source.get("protocol_id") == EXPECTED_STAGE23_PROTOCOL_ID,
        "decision": decision.get("status") == EXPECTED_STAGE23_STATUS,
        "source_bound_claim": bool(decision.get("source_bound_claim_eligible", False)),
        "stage22_bound": bool(decision.get("stage22_upstream_bound", False)),
        "transport_certified": bool(causal_result.get("transport_certification_pass", False)),
        "operator_switch_rejected": not bool(causal_result.get("operator_switch_pass", True)),
        "selected_block": int(geometry_freeze.get("selected_block", -1)) == SELECTED_BLOCK,
        "partition_sha": partition_sha == EXPECTED_STAGE23_PARTITION_SHA256,
        "geometry_sha": geometry_sha == EXPECTED_STAGE23_GEOMETRY_SHA256,
        "nested_stage22_checks": bool(all(stage22_binding.get("checks", {}).values())),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Stage 23 upstream binding failed: {checks}")
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
    write_json(SUBSPACE_DIR / "stage23_upstream_binding.json", payload)
    return payload


if not PIPELINE_FAILED:
    try:
        STAGE23_BINDING = locate_and_verify_stage23()
        print(json.dumps(STAGE23_BINDING, indent=2))
    except Exception:
        record_failure("stage23_upstream_binding")
'''


design = base_source(7)
design = design.replace("stage23_candidate_pool_design.npz", "stage24_candidate_pool_design.npz")
design = design.replace("Stage 23", "Stage 24").replace("stage23", "stage24")
design = design.replace(
    '"content_rank": ACTIVE_CONTENT_RANK,\n'
    '    "probe_count": ACTIVE_PROBE_COUNT,\n'
    '    "finite_response_contexts": ACTIVE_CONTEXTS,',
    '"completion_ranks": ACTIVE_COMPLETION_RANKS,\n'
    '    "completion_basis_rank": ACTIVE_COMPLETION_BASIS_RANK,\n'
    '    "construction_pairs_per_state": ACTIVE_CONSTRUCTION_PAIRS_PER_STATE,\n'
    '    "probe_count": ACTIVE_PROBE_COUNT,\n'
    '    "finite_response_contexts": ACTIVE_CONTEXTS,',
)


truth_generation = base_source(8)
truth_generation = truth_generation.replace("Stage 23", "Stage 24")
truth_generation = truth_generation.replace("stage23_truth_montage.png", "stage24_truth_montage.png")


baselines = base_source(9)
baselines = baselines.replace(
    "# Load the frozen JEPA-WM and cache fresh construction carriers.",
    "# Load the frozen JEPA-WM and cache fresh Stage 24 construction carriers.",
)
baselines = baselines.replace("stage23", "stage24")


completion_basis = r'''# Learn nested completion and matched-control bases on construction activations only.


def load_partition():
    with np.load(SUBSPACE_DIR / "frozen_mode_partition.npz") as payload:
        return {name: payload[name].copy() for name in payload.files}


def load_stage23_geometry():
    with np.load(SUBSPACE_DIR / "frozen_mode_operator_geometry.npz") as payload:
        return {name: payload[name].copy() for name in payload.files}


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


def sorted_cluster_members(assignments, transformed, partition, cluster):
    indices = np.flatnonzero(assignments == int(cluster))
    centroid = partition["centroids"][int(cluster)].astype(np.float64)
    distance = np.sum((transformed[indices] - centroid) ** 2, axis=1)
    return indices[np.argsort(distance, kind="stable")]


def select_construction_pairs(record, assignments, transformed, partition):
    on_cluster = int(partition["mode_on_cluster"])
    off_cluster = 1 - on_cluster
    off = sorted_cluster_members(
        assignments, transformed, partition, off_cluster
    )
    on = sorted_cluster_members(
        assignments, transformed, partition, on_cluster
    )
    count = min(ACTIVE_CONSTRUCTION_PAIRS_PER_STATE, len(off), len(on))
    mode_pairs, same_pairs = [], []
    for slot in range(count):
        mode_pairs.append(
            {
                "pair_id": f"{int(record['record_id'])}:mode:{slot}",
                "record_id": int(record["record_id"]),
                "trajectory_id": int(record["trajectory_id"]),
                "slot": int(slot),
                "base_index": int(off[slot]),
                "donor_index": int(on[slot]),
                "base_mode_coordinates": transformed[off[slot]].tolist(),
                "donor_mode_coordinates": transformed[on[slot]].tolist(),
            }
        )
        if len(off) >= 2:
            first = int(off[slot % len(off)])
            second = int(off[(slot + 1) % len(off)])
        elif len(on) >= 2:
            first = int(on[slot % len(on)])
            second = int(on[(slot + 1) % len(on)])
        else:
            continue
        if first == second:
            continue
        same_pairs.append(
            {
                "pair_id": f"{int(record['record_id'])}:same:{slot}",
                "record_id": int(record["record_id"]),
                "trajectory_id": int(record["trajectory_id"]),
                "slot": int(slot),
                "base_index": first,
                "donor_index": second,
                "base_mode_coordinates": transformed[first].tolist(),
                "donor_mode_coordinates": transformed[second].tolist(),
            }
        )
    return mode_pairs, same_pairs


def whiten_carrier(values, geometry):
    return transform_primal_channels(
        np.asarray(values, dtype=np.float64),
        geometry["channel_inverse_square_root"],
    )


def pair_decomposition(payload, pair, geometry):
    white = whiten_carrier(payload["carrier"], geometry)
    flat = white.reshape(ACTIONS_PER_STATE, -1)
    base_index = int(pair["base_index"])
    donor_index = int(pair["donor_index"])
    native = flat[donor_index] - flat[base_index]
    mode_covectors = geometry["mode_covectors_white"].astype(np.float64)
    base_q = np.asarray(pair["base_mode_coordinates"], dtype=np.float64)
    native_q = base_q + mode_covectors.T @ native
    target = native_q - base_q
    transport = minimal_constrained_transport(
        mode_covectors,
        geometry["content_basis"].astype(np.float64),
        target,
        ridge=TRANSPORT_SOLVE_RIDGE,
    )
    decomposition = completion_residual(
        native,
        transport["delta"],
        mode_covectors,
        tolerance=MAX_RESIDUAL_MODE_DRIFT,
    )
    return {
        "native_delta": native,
        "mode_edit": transport["delta"],
        "residual": decomposition["residual"],
        "native_mode_coordinates": native_q,
        "mode_coordinate_residual": transport["mode_residual_norm"],
        "residual_mode_drift": decomposition["relative_coordinate_residual"],
    }


def dual_residual_basis_gpu(residuals, rank):
    matrix = torch.as_tensor(
        np.asarray(residuals), device="cuda", dtype=torch.float32
    )
    gram = matrix @ matrix.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    positive = int(torch.sum(eigenvalues > max(float(eigenvalues[0]) * 1e-9, 1e-8)).item())
    if positive < int(rank):
        raise RuntimeError(f"residual basis rank {positive} is below required {rank}")
    values = torch.clamp(eigenvalues[: int(rank)], min=1e-12)
    basis = matrix.T @ eigenvectors[:, : int(rank)]
    basis = basis / torch.sqrt(values)[None]
    result = basis.detach().cpu().numpy().astype(np.float64)
    spectrum = torch.sqrt(torch.clamp(eigenvalues, min=0)).detach().cpu().numpy().astype(np.float64)
    del matrix, gram, eigenvalues, eigenvectors, values, basis
    torch.cuda.empty_cache()
    return result, spectrum


def mode_null_nested_qr(basis, mode_covectors):
    values = np.asarray(basis, dtype=np.float64)
    mode = np.asarray(mode_covectors, dtype=np.float64)
    gram = mode.T @ mode
    residual = values - mode @ np.linalg.solve(gram, mode.T @ values)
    orthonormal, triangular = np.linalg.qr(residual, mode="reduced")
    if np.min(np.abs(np.diag(triangular))) <= 1e-10:
        raise RuntimeError("mode-null basis lost rank")
    return orthonormal


def fit_and_freeze_completion_bases():
    partition = load_partition()
    geometry = load_stage23_geometry()
    mode_pairs, same_pairs = [], []
    assignments_payload = {}
    for record in CONSTRUCTION_RECORDS:
        payload = load_carrier(record["record_id"])
        assignments, transformed = frozen_mode_assignments_from_carrier(
            payload["carrier"], partition
        )
        record_mode, record_same = select_construction_pairs(
            record, assignments, transformed, partition
        )
        mode_pairs.extend(record_mode)
        same_pairs.extend(record_same)
        assignments_payload[str(int(record["record_id"]))] = assignments.tolist()
    pair_freeze = {
        "assignments": assignments_payload,
        "mode_pairs": mode_pairs,
        "same_mode_pairs": same_pairs,
        "selection_inputs": ["construction activations", "frozen Stage 23 partition"],
        "physical_contact_labels_used": False,
    }
    write_json(DESIGN_DIR / "construction_completion_pair_freeze.json", pair_freeze)

    mode_residuals, same_residuals = [], []
    decomposition_rows = []
    for family, pairs, destination in [
        ("completion", mode_pairs, mode_residuals),
        ("same_mode", same_pairs, same_residuals),
    ]:
        for pair in pairs:
            payload = load_carrier(pair["record_id"])
            decomposition = pair_decomposition(payload, pair, geometry)
            destination.append(decomposition["residual"])
            decomposition_rows.append(
                {
                    "pair_id": pair["pair_id"],
                    "record_id": int(pair["record_id"]),
                    "family": family,
                    "native_edit_norm": float(np.linalg.norm(decomposition["native_delta"])),
                    "mode_edit_norm": float(np.linalg.norm(decomposition["mode_edit"])),
                    "residual_norm": float(np.linalg.norm(decomposition["residual"])),
                    "mode_coordinate_residual": decomposition["mode_coordinate_residual"],
                    "residual_mode_drift": decomposition["residual_mode_drift"],
                }
            )
    if len(mode_residuals) < ACTIVE_COMPLETION_BASIS_RANK:
        raise RuntimeError("insufficient construction mode-pair residuals")
    if len(same_residuals) < ACTIVE_COMPLETION_BASIS_RANK:
        raise RuntimeError("insufficient construction same-mode residuals")
    completion_raw, completion_singular = dual_residual_basis_gpu(
        np.stack(mode_residuals), ACTIVE_COMPLETION_BASIS_RANK
    )
    same_raw, same_singular = dual_residual_basis_gpu(
        np.stack(same_residuals), ACTIVE_COMPLETION_BASIS_RANK
    )
    mode_covectors = geometry["mode_covectors_white"].astype(np.float64)
    completion_basis = mode_null_nested_qr(completion_raw, mode_covectors)
    same_basis = mode_null_nested_qr(same_raw, mode_covectors)
    rng = np.random.default_rng(RANDOM_BASIS_SEED)
    random_raw = rng.normal(
        size=(mode_covectors.shape[0], ACTIVE_COMPLETION_BASIS_RANK)
    )
    random_basis = mode_null_nested_qr(random_raw, mode_covectors)
    basis_map = {
        "completion_basis": completion_basis,
        "same_mode_basis": same_basis,
        "random_basis": random_basis,
    }
    max_drift = max(
        float(np.max(np.abs(mode_covectors.T @ basis)))
        for basis in basis_map.values()
    )
    if max_drift > 1e-7:
        raise RuntimeError(f"completion bases change mode coordinates: {max_drift}")
    atomic_npz(
        SUBSPACE_DIR / "frozen_causal_completion_bases.npz",
        completion_basis=completion_basis,
        same_mode_basis=same_basis,
        random_basis=random_basis,
        completion_singular_values=completion_singular,
        same_mode_singular_values=same_singular,
        active_completion_ranks=np.asarray(ACTIVE_COMPLETION_RANKS, dtype=np.int64),
        construction_mode_pair_count=np.asarray(len(mode_residuals), dtype=np.int64),
        construction_same_pair_count=np.asarray(len(same_residuals), dtype=np.int64),
    )
    write_csv(ANALYSIS_DIR / "construction_residual_decompositions.csv", decomposition_rows)
    payload = {
        "frozen_before_evaluation_activations": True,
        "physical_contact_labels_used": False,
        "stage23_geometry_sha256": EXPECTED_STAGE23_GEOMETRY_SHA256,
        "completion_ranks": ACTIVE_COMPLETION_RANKS,
        "completion_basis_rank": ACTIVE_COMPLETION_BASIS_RANK,
        "construction_mode_pairs": len(mode_residuals),
        "construction_same_mode_pairs": len(same_residuals),
        "max_mode_covector_basis_inner_product": max_drift,
        "evaluation_activation_ids_seen": [],
        "basis_sha256": sha256_file(SUBSPACE_DIR / "frozen_causal_completion_bases.npz"),
        "construction_pair_freeze_sha256": sha256_file(
            DESIGN_DIR / "construction_completion_pair_freeze.json"
        ),
    }
    write_json(SUBSPACE_DIR / "causal_completion_basis_freeze.json", payload)
    return payload


if not PIPELINE_FAILED:
    try:
        COMPLETION_BASIS_FREEZE = fit_and_freeze_completion_bases()
        print(json.dumps(COMPLETION_BASIS_FREEZE, indent=2))
    except Exception:
        record_failure("construction_completion_basis_freeze")
'''


evaluation_open = base_source(11)
evaluation_open = evaluation_open.replace(
    function_sources(evaluation_open, ["frozen_mode_assignments_from_carrier"]),
    "",
)
evaluation_open = evaluation_open.replace(
    "# Open held-out activations, freeze model-only pairs, then reveal contact labels.",
    "# Open fresh Stage 24 activations, freeze model-only pairs, then reveal contact labels.",
)
evaluation_open = evaluation_open.replace(
    'if not (SUBSPACE_DIR / "mode_operator_geometry_freeze.json").exists():\n'
    '            raise RuntimeError("construction geometry must freeze before evaluation opens")',
    'if not (SUBSPACE_DIR / "causal_completion_basis_freeze.json").exists():\n'
    '            raise RuntimeError("completion bases must freeze before evaluation opens")',
)


interventions = r'''# Measure the nested causal-completion curve and matched controls.


def load_completion_bases():
    with np.load(SUBSPACE_DIR / "frozen_causal_completion_bases.npz") as payload:
        return {name: payload[name].copy() for name in payload.files}


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


def pair_completion_contexts(pair, payload, geometry, bases, partition):
    decomposition = pair_decomposition(payload, pair, geometry)
    native = decomposition["native_delta"]
    mode_edit = decomposition["mode_edit"]
    residual = decomposition["residual"]
    basis_map = {
        "completion": bases["completion_basis"].astype(np.float64),
        "same_mode": bases["same_mode_basis"].astype(np.float64),
        "random": bases["random_basis"].astype(np.float64),
    }
    contexts = {
        "off": np.zeros_like(native),
        "native_on": native,
        "q_only": mode_edit,
    }
    for rank in ACTIVE_COMPLETION_RANKS[1:]:
        for family, basis in basis_map.items():
            contexts[f"{family}_rank_{rank}"] = completion_edit(
                mode_edit, residual, basis, rank
            )
    if list(contexts) != ACTIVE_CONTEXTS:
        raise RuntimeError("completion context ordering changed")
    base_q = np.asarray(pair["base_mode_coordinates"], dtype=np.float64)
    native_q = decomposition["native_mode_coordinates"]
    mode_covectors = geometry["mode_covectors_white"].astype(np.float64)
    on_cluster = int(partition["mode_on_cluster"])
    context_rows = []
    for condition, edit in contexts.items():
        coordinate = base_q + mode_covectors.T @ edit
        cluster = cluster_for_coordinate(coordinate, partition)
        if condition == "off":
            family, rank = "off", -1
        elif condition == "native_on":
            family, rank = "native_on", -1
        elif condition == "q_only":
            family, rank = "completion", 0
        else:
            prefix, rank_text = condition.rsplit("_rank_", 1)
            family, rank = prefix, int(rank_text)
        reconstruction = (
            0.0
            if condition == "off"
            else native_reconstruction_fraction(native, edit)
        )
        context_rows.append(
            {
                "pair_id": pair["pair_id"],
                "record_id": int(pair["record_id"]),
                "trajectory_id": int(pair["trajectory_id"]),
                "condition": condition,
                "family": family,
                "rank": int(rank),
                "mode_cluster": cluster,
                "mode_flip": cluster == on_cluster,
                "mode_coordinate_distance_to_native": float(
                    np.linalg.norm(coordinate - native_q)
                ),
                "edit_norm": float(np.linalg.norm(edit)),
                "native_reconstruction_fraction": reconstruction,
                **PAIR_TRUTH_MAP[pair["pair_id"]],
            }
        )
    diagnostics = {
        "pair_id": pair["pair_id"],
        "record_id": int(pair["record_id"]),
        "trajectory_id": int(pair["trajectory_id"]),
        "base_index": int(pair["base_index"]),
        "donor_index": int(pair["donor_index"]),
        "native_on_flip": next(
            row["mode_flip"] for row in context_rows if row["condition"] == "native_on"
        ),
        "q_only_flip": next(
            row["mode_flip"] for row in context_rows if row["condition"] == "q_only"
        ),
        "mode_coordinate_residual": decomposition["mode_coordinate_residual"],
        "residual_mode_drift": decomposition["residual_mode_drift"],
        "native_edit_norm": float(np.linalg.norm(native)),
        "q_only_edit_norm": float(np.linalg.norm(mode_edit)),
        "completion_residual_norm": float(np.linalg.norm(residual)),
        **PAIR_TRUTH_MAP[pair["pair_id"]],
    }
    return contexts, context_rows, diagnostics


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


def pair_operator_rows(record, pair, payload, initial, actions, geometry, bases, partition):
    contexts, context_rows, diagnostics = pair_completion_contexts(
        pair, payload, geometry, bases, partition
    )
    base_index = int(pair["base_index"])
    donor_index = int(pair["donor_index"])
    probes = geometry["probe_vectors"].astype(np.float64).T[:ACTIVE_PROBE_COUNT]
    stages = [f"block_{block}" for block in DOWNSTREAM_BLOCKS] + ["output"]
    responses = {
        condition: {stage: [] for stage in stages}
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
            for stage in stages:
                responses[condition][stage].append(
                    symmetric_finite_response(plus[stage], minus[stage], FINITE_PROBE_DOSE)
                )
    context_lookup = {row["condition"]: row for row in context_rows}
    rows = []
    for stage in stages:
        off = np.stack(responses["off"][stage]).reshape(-1)
        native = np.stack(responses["native_on"][stage]).reshape(-1)
        for condition in ACTIVE_CONTEXTS:
            context_response = np.stack(responses[condition][stage]).reshape(-1)
            metadata = context_lookup[condition]
            rows.append(
                {
                    "pair_id": pair["pair_id"],
                    "record_id": int(record["record_id"]),
                    "trajectory_id": int(record["trajectory_id"]),
                    "base_index": base_index,
                    "donor_index": donor_index,
                    "condition": condition,
                    "family": metadata["family"],
                    "rank": metadata["rank"],
                    "stage": stage,
                    "physically_aligned": diagnostics["physically_aligned"],
                    "mode_flip": metadata["mode_flip"],
                    "native_reconstruction_fraction": metadata["native_reconstruction_fraction"],
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
            f"completion forward contract changed: {observed_forwards} != "
            f"{ACTIVE_PATCHED_FORWARDS_PER_PAIR}"
        )
    return rows, context_rows, diagnostics


def intervention_path(record_id):
    return INTERVENTION_DIR / f"evaluation_{int(record_id):04d}.json"


def run_all_completion_tests():
    started = time.perf_counter()
    geometry = load_stage23_geometry()
    bases = load_completion_bases()
    partition = load_partition()
    by_record = defaultdict(list)
    for pair in EVALUATION_PAIRS:
        by_record[int(pair["record_id"])].append(pair)
    rows, context_rows, diagnostics = [], [], []
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
        record_rows, record_contexts, record_diagnostics = [], [], []
        for pair in by_record[int(record["record_id"])]:
            pair_rows, pair_contexts, pair_diagnostics = pair_operator_rows(
                record, pair, payload, initial, actions, geometry, bases, partition
            )
            record_rows.extend(pair_rows)
            record_contexts.extend(pair_contexts)
            record_diagnostics.append(pair_diagnostics)
        write_json(
            destination,
            {
                "operator_rows": record_rows,
                "context_rows": record_contexts,
                "pair_diagnostics": record_diagnostics,
            },
        )
        rows.extend(record_rows)
        context_rows.extend(record_contexts)
        diagnostics.extend(record_diagnostics)
        PROVENANCE_COUNTS["intervention_generated"] += 1
        write_json(
            OUT / "completion_progress.json",
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
    TIMINGS["completion_interventions_seconds"] = time.perf_counter() - started
    write_csv(EVIDENCE_DIR / "causal_completion_operator_rows.csv", rows)
    write_csv(EVIDENCE_DIR / "completion_context_diagnostics.csv", context_rows)
    write_csv(EVIDENCE_DIR / "completion_pair_diagnostics.csv", diagnostics)
    return rows, context_rows, diagnostics


if not PIPELINE_FAILED and EVALUATION_OPENED:
    try:
        COMPLETION_ROWS, COMPLETION_CONTEXT_ROWS, COMPLETION_PAIR_DIAGNOSTICS = (
            run_all_completion_tests()
        )
        memory_report("causal_completion_interventions_complete")
    except Exception:
        record_failure("causal_completion_interventions")
'''


decision = r'''# Apply the preregistered Stage 24 causal-completion-rank gates.


def bootstrap_payload(values, groups, label):
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups, dtype=np.int64)
    if not len(values):
        return None
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


def condition_for_rank(family, rank):
    return "q_only" if int(rank) == 0 else f"{family}_rank_{int(rank)}"


def aligned_output_map(rows, family, rank):
    condition = condition_for_rank(family, rank)
    return {
        row["pair_id"]: row for row in rows
        if row["stage"] == "output"
        and row["condition"] == condition
        and row["physically_aligned"]
        and row["mode_flip"]
    }


def build_rank_curve(rows):
    curve = []
    for family in BASIS_CONDITIONS:
        for rank in ACTIVE_COMPLETION_RANKS:
            mapping = aligned_output_map(rows, family, rank)
            identifiers = sorted(mapping)
            values = np.asarray(
                [mapping[pair]["transfer_coefficient"] for pair in identifiers],
                dtype=np.float64,
            )
            groups = np.asarray(
                [mapping[pair]["trajectory_id"] for pair in identifiers],
                dtype=np.int64,
            )
            summary = bootstrap_payload(
                values, groups, f"curve:{family}:{rank}"
            )
            reconstruction = np.asarray(
                [mapping[pair]["native_reconstruction_fraction"] for pair in identifiers],
                dtype=np.float64,
            )
            curve.append(
                {
                    "family": family,
                    "rank": int(rank),
                    "mean": None if summary is None else summary["mean"],
                    "lower": None if summary is None else summary["lower"],
                    "upper": None if summary is None else summary["upper"],
                    "n_pairs": 0 if summary is None else summary["n_pairs"],
                    "n_trajectories": 0 if summary is None else summary["n_trajectories"],
                    "mean_native_reconstruction_fraction": (
                        None if not len(reconstruction) else float(np.mean(reconstruction))
                    ),
                }
            )
    return curve


def paired_rank_gain(rows, rank, control_family):
    learned = aligned_output_map(rows, "completion", rank)
    control = aligned_output_map(rows, control_family, rank)
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


def layerwise_completion_curve(rows):
    result = []
    for stage in [f"block_{block}" for block in DOWNSTREAM_BLOCKS] + ["output"]:
        for family in BASIS_CONDITIONS:
            for rank in ACTIVE_COMPLETION_RANKS:
                condition = condition_for_rank(family, rank)
                values = [
                    row["transfer_coefficient"] for row in rows
                    if row["stage"] == stage
                    and row["condition"] == condition
                    and row["physically_aligned"]
                    and row["mode_flip"]
                ]
                result.append(
                    {
                        "stage": stage,
                        "family": family,
                        "rank": int(rank),
                        "mean_transfer_coefficient": (
                            None if not values else float(np.mean(values))
                        ),
                        "n_pairs": len(values),
                    }
                )
    return result


def evaluate_causal_completion(rows, context_rows, diagnostics):
    rank_curve = build_rank_curve(rows)
    learned_curve = [
        row for row in rank_curve
        if row["family"] == "completion" and row["lower"] is not None
    ]
    ccr = causal_completion_rank(
        learned_curve, threshold=COMPLETION_TRANSFER_THRESHOLD
    ) if learned_curve else None
    q_flip_rate = float(np.mean([row["q_only_flip"] for row in diagnostics])) if diagnostics else 0.0
    native_flip_rate = float(np.mean([row["native_on_flip"] for row in diagnostics])) if diagnostics else 0.0
    max_coordinate_residual = float(max(
        (row["mode_coordinate_residual"] for row in diagnostics), default=1e9
    ))
    max_residual_drift = float(max(
        (row["residual_mode_drift"] for row in diagnostics), default=1e9
    ))
    aligned_diagnostics = [row for row in diagnostics if row["physically_aligned"]]
    full_swap = float(np.mean([
        row["native_full_swap_coefficient"] for row in aligned_diagnostics
    ])) if aligned_diagnostics else 0.0
    q_rows = aligned_output_map(rows, "completion", 0)
    target_energy = np.asarray(
        [row["target_energy"] for row in q_rows.values()], dtype=np.float64
    )
    nondegenerate = float(np.mean(target_energy > TARGET_ENERGY_FLOOR)) if len(target_energy) else 0.0

    control_payload = {}
    specificity_pass = False
    if ccr is not None:
        required_positive = int(
            np.ceil(REQUIRED_POSITIVE_GAIN_FRACTION * len(q_rows))
        )
        passes = []
        for control_family in ["same_mode", "random"]:
            identifiers, gain, groups = paired_rank_gain(
                rows, ccr, control_family
            )
            summary = bootstrap_payload(
                gain, groups, f"gain:{control_family}:{ccr}"
            )
            positive = int(np.sum(gain > 0))
            sign_payload = exact_positive_sign_test(gain) if len(gain) else None
            if (
                sign_payload is not None
                and not math.isfinite(sign_payload["p_value"])
            ):
                sign_payload["p_value"] = None
            payload = {
                "summary": summary,
                "positive_pairs": positive,
                "required_positive_pairs": required_positive,
                "sign_test_p": sign_payload,
            }
            control_payload[control_family] = payload
            passes.append(bool(
                summary is not None
                and summary["mean"] >= MIN_GAIN_OVER_CONTROL
                and summary["lower"] > 0
                and positive >= required_positive
            ))
        specificity_pass = bool(all(passes))

    certification_pass = bool(
        len(q_rows) >= ACTIVE_MIN_PHYSICALLY_ALIGNED_PAIRS
        and q_flip_rate >= MIN_MODE_FLIP_RATE
        and native_flip_rate >= MIN_MODE_FLIP_RATE
        and max_coordinate_residual <= MAX_MODE_COORDINATE_RESIDUAL
        and max_residual_drift <= MAX_RESIDUAL_MODE_DRIFT
        and full_swap >= MIN_NATIVE_FULL_SWAP_COEFFICIENT
        and nondegenerate >= MIN_NONDEGENERATE_TARGET_FRACTION
    )
    payload = {
        "physically_aligned_certified_pairs": len(q_rows),
        "q_only_flip_rate_all_model_pairs": q_flip_rate,
        "native_on_flip_rate_all_model_pairs": native_flip_rate,
        "max_mode_coordinate_residual": max_coordinate_residual,
        "max_completion_residual_mode_drift": max_residual_drift,
        "mean_native_full_swap_coefficient": full_swap,
        "nondegenerate_native_operator_target_fraction": nondegenerate,
        "rank_curve": rank_curve,
        "ccr_0_8": ccr,
        "control_specificity_at_ccr": control_payload,
        "certification_pass": certification_pass,
        "specificity_pass": specificity_pass,
        "compact_causal_completion_pass": bool(
            certification_pass and ccr is not None and specificity_pass
        ),
    }
    return payload


if not PIPELINE_FAILED:
    try:
        SOURCE_EXECUTION_VERIFIED = verify_executed_notebook_through(
            "# Apply the preregistered Stage 24 causal-completion-rank gates."
        )
        physical_mode_pass = bool(
            CONTACT_ALIGNMENT["balanced_accuracy"] >= MIN_CONTACT_BALANCED_ACCURACY
            and CONTACT_ALIGNMENT["matthews_correlation"] >= MIN_CONTACT_MCC
        )
        COMPLETION_RESULT = evaluate_causal_completion(
            COMPLETION_ROWS,
            COMPLETION_CONTEXT_ROWS,
            COMPLETION_PAIR_DIAGNOSTICS,
        )
        LAYERWISE_COMPLETION = layerwise_completion_curve(COMPLETION_ROWS)
        write_csv(EVIDENCE_DIR / "causal_completion_rank_curve.csv", COMPLETION_RESULT["rank_curve"])
        write_csv(EVIDENCE_DIR / "layerwise_completion_curve.csv", LAYERWISE_COMPLETION)

        if RUN_MODE == "smoke":
            candidate_status = "SMOKE_ONLY"
        elif not physical_mode_pass:
            candidate_status = "STAGE23_MODE_NOT_REPLICATED_ON_FRESH_STATES"
        elif not COMPLETION_RESULT["certification_pass"]:
            candidate_status = "COMPLETION_PROTOCOL_INVALID"
        elif COMPLETION_RESULT["compact_causal_completion_pass"]:
            candidate_status = "COMPACT_CAUSAL_COMPLETION_FOUND"
        elif COMPLETION_RESULT["ccr_0_8"] is not None:
            candidate_status = "GENERIC_RESIDUAL_COMPLETION_NOT_MODE_SPECIFIC"
        else:
            candidate_status = "NO_RANK64_CAUSAL_COMPLETION"

        upstream_eligible = bool(all(STAGE23_BINDING["checks"].values()))
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
            "stage23_upstream_bound": upstream_eligible,
            "heldout_contact_alignment": CONTACT_ALIGNMENT,
            "causal_completion_rank_test": COMPLETION_RESULT,
            "layerwise_completion": LAYERWISE_COMPLETION,
            "claim_boundary": {
                "compact_completion_claim_authorized": bool(
                    status == "COMPACT_CAUSAL_COMPLETION_FOUND"
                ),
                "nonlinear_manifold_ruled_out": False,
                "rank_above_64_ruled_out": False,
                "complete_hybrid_automaton_extracted": False,
                "planning_utility_claim_authorized": False,
                "jacobian_or_infinitesimal_linearization_claim_authorized": False,
                "authorized_claim_if_passed": "a shared low-rank residual subspace completes a readable contact mode into the native finite response operator",
            },
        }
        write_json(OUT / "stage24_decision.json", DECISION_PAYLOAD)

        curve = COMPLETION_RESULT["rank_curve"]
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        colors = {"completion": "tab:blue", "same_mode": "tab:orange", "random": "tab:green"}
        for family in BASIS_CONDITIONS:
            selected = [row for row in curve if row["family"] == family]
            ranks = [row["rank"] for row in selected]
            means = [np.nan if row["mean"] is None else row["mean"] for row in selected]
            lowers = [np.nan if row["lower"] is None else row["lower"] for row in selected]
            uppers = [np.nan if row["upper"] is None else row["upper"] for row in selected]
            axes[0].plot(ranks, means, marker="o", color=colors[family], label=family)
            axes[0].fill_between(ranks, lowers, uppers, color=colors[family], alpha=0.15)
            reconstruction = [
                np.nan if row["mean_native_reconstruction_fraction"] is None
                else row["mean_native_reconstruction_fraction"]
                for row in selected
            ]
            axes[1].plot(ranks, reconstruction, marker="o", color=colors[family], label=family)
        axes[0].axhline(COMPLETION_TRANSFER_THRESHOLD, color="black", linestyle="--", linewidth=1)
        axes[0].set(xlabel="completion rank", ylabel="native-operator transfer", title="Causal completion curve")
        axes[0].legend()
        axes[1].set(xlabel="completion rank", ylabel="native activation distance closed", title="Representational completion")
        axes[1].legend()
        selected_rank = (
            COMPLETION_RESULT["ccr_0_8"]
            if COMPLETION_RESULT["ccr_0_8"] is not None
            else max(ACTIVE_COMPLETION_RANKS)
        )
        stages = [f"block_{block}" for block in DOWNSTREAM_BLOCKS] + ["output"]
        for family in BASIS_CONDITIONS:
            selected = [
                row for row in LAYERWISE_COMPLETION
                if row["family"] == family and row["rank"] == selected_rank
            ]
            lookup = {row["stage"]: row["mean_transfer_coefficient"] for row in selected}
            axes[2].plot(
                stages,
                [np.nan if lookup[stage] is None else lookup[stage] for stage in stages],
                marker="o",
                color=colors[family],
                label=family,
            )
        axes[2].set(
            ylabel="native-operator transfer",
            title=f"Downstream completion at rank {selected_rank}",
        )
        axes[2].tick_params(axis="x", rotation=25)
        axes[2].legend()
        figure.tight_layout()
        figure.savefig(PLOT_DIR / "stage24_causal_completion_summary.png", dpi=180)
        plt.close(figure)
        print(json.dumps(DECISION_PAYLOAD, indent=2))
    except Exception:
        record_failure("decision_and_plots")
        DECISION_PAYLOAD = {"status": "INCONCLUSIVE", "failure": FAILURE_MESSAGE}
else:
    DECISION_PAYLOAD = {"status": "INCONCLUSIVE", "failure": FAILURE_MESSAGE}

if not (OUT / "stage24_decision.json").exists():
    write_json(OUT / "stage24_decision.json", DECISION_PAYLOAD)
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
    if path.name.startswith("stage24_causal_completion_result_bundle_"):
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

archive_base = OUT / f"stage24_causal_completion_result_bundle_{RUN_SIGNATURE[:12]}"
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
    completion_basis,
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
    code(completion_basis),
    code(evaluation_open),
    code(interventions),
    code(decision),
    code(packaging),
]
for index, cell in enumerate(cells):
    cell["id"] = f"stage24-{index:02d}"

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
