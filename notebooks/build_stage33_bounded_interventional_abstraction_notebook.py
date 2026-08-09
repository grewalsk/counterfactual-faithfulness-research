import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
TARGET = ROOT / "33_bounded_interventional_predictive_causal_abstraction.ipynb"
NUMERICAL = REPOSITORY / "src/cf_faithfulness/stage33_interventional_abstraction.py"

# Reuse the audited source-binding, official model loader, and hook semantics.
# The generated Stage 33 notebook is self-contained; this dependency exists
# only while deterministically building the JSON artifact.
spec = importlib.util.spec_from_file_location(
    "stage32_builder",
    ROOT / "build_stage32_powered_bounded_confirmation_notebook.py",
)
STAGE32 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(STAGE32)

code = STAGE32.code
markdown = STAGE32.markdown
assigned_uppercase_names = STAGE32.assigned_uppercase_names
function_sources = STAGE32.function_sources


introduction = r'''# Stage 33: bounded interventional predictive causal abstraction

## V2 model-free coverage amendment

The source-bound v1 pilot stopped before loading either world model because
its frozen evaluation candidate pool yielded 15 complete four-mode physical
trajectories rather than the required 16.  No activation, prediction, decoded
effect, planning score, or scientific gate was observed.  V2 therefore keeps
all targets, estimands, action banks, thresholds, and decision gates unchanged
while replacing pool-size-dependent candidate geometry with a stable
trajectory-ID design, enlarging the four disjoint model-free screening pools,
and writing complete screening evidence before any coverage exception.

## Verdict before computation

The proposed **minimal hybrid predictive realization** is not identifiable
from the proposed experiment.  Equality of terminal conditional means is too
weak to define a predictive state in a stochastic or partially observed
system; a finite action bank supports only a bounded claim; and two minimal
factorizations of the same grounded input-output matrix are similar by
realization theory even when the neural predictors compute in unrelated ways.

This notebook therefore tests a narrower and falsifiable object: a **bounded,
mode-conditioned interventional predictive causal abstraction** (BIPCA).  A
positive result requires all of the following, in increasing order:

1. grounded physical quantities are decodable from each model's real native
   predictions;
2. construction charts have stable effective rank on the separate
   model-selection split;
3. held-out predictive charts and model-native transition operators align;
4. one calibration-only map intertwines global and mode-conditioned operators
   on unseen states and unseen action compositions;
5. that same map transports reachable internal action-response interventions
   through the frozen DINO predictor with the intended multi-step grounded
   effect; and
6. transported native predictions preserve physical planning value.

Only levels 4--6 can support a shared-*abstract*-mechanism statement.  Even a
full pass does not identify the models' full neural algorithms, a universal
contact automaton, an exact Koopman lift, or an ontologically minimal physical
state.  The public JEPA-WM and DINO-WM PushT checkpoints also share the DINOv2
target family and there is only one checkpoint per family, so the final claim
is restricted to two predictor architectures with a shared output target.

### Bounded object

For a reachable history `H`, core test bank `B`, horizons at most four, and a
frozen decoder of grounded path features, the empirical signature is

\[
q_m^{\mathcal B}(H)=
\{\widehat{\mathbb E}_m[\phi(X_{t+1:t+k})\mid H,\operatorname{do}(w)]
 : w\in\mathcal B,\ k\le 4\}.
\]

Distances between finite signatures are pseudometrics.  The relation
`distance <= epsilon` is **not** treated as an equivalence relation.  The rank
reported below is an effective rank of this frozen finite response family,
not the rank of the infinite controlled Hankel operator.

The notebook uses real official model predictions, exact PushT restoration,
real simulator branches, four physical contact-timing strata, multi-length
action words, a trajectory-block bootstrap, construction/calibration/locked
evaluation separation, model-native activation transport, and exact physical
planning regret.  There is no synthetic fallback.  Return
`stage33_bipca_result_bundle_<signature>.zip` and retain the complete Drive
directory containing the resumable raw shards.
'''


configuration = r'''# SINGLE CONFIGURATION BLOCK — no Stage 33 secrets required.
import secrets as _secrets
import time as _time

# `pilot` never inherits smoke counts.
RUN_MODE = "pilot"
EXPERIMENT_SOURCE_REF = "codex/stage33-bounded-interventional-abstraction"
# Prefer the optional Drive request file; this value is never a secret.
MANUAL_RUN_NONCE = ""
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

MOUNT_DRIVE = True
DOWNLOAD_RESULTS = True
RESUME_INCOMPLETE = True
CONTINUE_AFTER_BENCHMARK = True
MAX_ESTIMATED_TOTAL_MINUTES = 420.0

OUTPUT_DIR = "/content/counterfactual_faithfulness_stage33_bipca"
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/counterfactual_faithfulness_stage33_bipca"
RUN_REQUEST_PATH = "/content/drive/MyDrive/counterfactual_faithfulness_stage33_bipca/stage33_run_request.json"

PROTOCOL_ID = "stage33-bounded-interventional-predictive-causal-abstraction-v2"
NOTEBOOK_PROTOCOL_SHA256 = "__PROTOCOL_DIGEST__"
EVIDENCE_STATUS = "CONFIRMATORY_V2_ONLY_IF_SOURCE_BOUND_SPLIT_LOCKED_AND_CAUSALLY_TRANSPORTED"
EXPERIMENT_REPOSITORY = "grewalsk/counterfactual-faithfulness-research"
EXPERIMENT_NOTEBOOK_PATH = "notebooks/33_bounded_interventional_predictive_causal_abstraction.ipynb"
EXPERIMENT_BUILDER_PATH = "notebooks/build_stage33_bounded_interventional_abstraction_notebook.py"
EXPERIMENT_NUMERICAL_PATH = "src/cf_faithfulness/stage33_interventional_abstraction.py"

SEED = 33101
DESIGN_SEED = 33137
DECODER_SEED = 33179
RANK_SEED = 33211
CALIBRATION_SEED = 33247
BOOTSTRAP_SEED = 33269
MAP_SEED = 33311
CONTROL_SEED = 33347

ENVIRONMENT = "PushT"
MODEL_NAMES = ["jepa_wm_pusht", "dino_wm_pusht"]
MODEL_SHORT_NAMES = {"jepa_wm_pusht": "jepa", "dino_wm_pusht": "dino"}
EXPECTED_MODEL_TYPES = {"jepa_wm_pusht": "AdaLN", "dino_wm_pusht": "dino_wm"}
EXPECTED_CARRIER_WIDTHS = {"jepa_wm_pusht": 400, "dino_wm_pusht": 414}
INTERVENTION_BLOCK = 4
FRAMESKIP = 5
MAX_WORD_LENGTH = 4
MODE_LABELS = ["free", "pre_contact", "contact", "post_contact"]
TRAJECTORY_GEOMETRY_VERSION = "absolute_golden_angle_v2"
TRAJECTORY_PHASE_INCREMENT = 0.6180339887498949

CONSTRUCTION_TRAJECTORY_POOL = list(range(6000, 6800))
MODEL_SELECTION_TRAJECTORY_POOL = list(range(6800, 7600))
CALIBRATION_TRAJECTORY_POOL = list(range(7600, 8400))
EVALUATION_TRAJECTORY_POOL = list(range(8400, 10000))
CONSTRUCTION_TRAJECTORIES = 8
MODEL_SELECTION_TRAJECTORIES = 8
CALIBRATION_TRAJECTORIES = 8
EVALUATION_TRAJECTORIES = 16
STATES_PER_TRAJECTORY = 4
TASK_ID_OFFSET = 33000
DISTANCE_GRID = [82.0, 86.0, 90.0, 94.0, 98.0]

CORE_WORD_SPECS = [
    {"name": "L", "angles": [-30.0], "magnitudes": [0.14]},
    {"name": "R", "angles": [30.0], "magnitudes": [0.14]},
    {"name": "S", "angles": [0.0], "magnitudes": [0.10]},
    {"name": "LR", "angles": [-30.0, 30.0], "magnitudes": [0.14, 0.14]},
    {"name": "RL", "angles": [30.0, -30.0], "magnitudes": [0.14, 0.14]},
    {"name": "LL", "angles": [-30.0, -30.0], "magnitudes": [0.14, 0.14]},
    {"name": "RR", "angles": [30.0, 30.0], "magnitudes": [0.14, 0.14]},
    {"name": "LRL", "angles": [-30.0, 30.0, -30.0], "magnitudes": [0.14, 0.14, 0.14]},
    {"name": "RLR", "angles": [30.0, -30.0, 30.0], "magnitudes": [0.14, 0.14, 0.14]},
    {"name": "LLR", "angles": [-30.0, -30.0, 30.0], "magnitudes": [0.14, 0.14, 0.14]},
    {"name": "RRL", "angles": [30.0, 30.0, -30.0], "magnitudes": [0.14, 0.14, 0.14]},
]
CALIBRATION_INTERCHANGE_PAIRS = [
    ["LR", "RL", 0], ["LLR", "RLL", 0],
    ["RRL", "LRR", 0], ["LRL", "RLR", 1],
]
EVALUATION_WORD_SPECS = [
    {"name": "a", "angles": [-20.0], "magnitudes": [0.10]},
    {"name": "b", "angles": [20.0], "magnitudes": [0.22]},
    {"name": "A", "angles": [-40.0], "magnitudes": [0.18]},
    {"name": "B", "angles": [40.0], "magnitudes": [0.18]},
    {"name": "ab", "angles": [-20.0, 20.0], "magnitudes": [0.10, 0.22]},
    {"name": "ba", "angles": [20.0, -20.0], "magnitudes": [0.22, 0.10]},
    {"name": "AAB", "angles": [-40.0, -40.0, 40.0], "magnitudes": [0.18, 0.18, 0.18]},
    {"name": "BAA", "angles": [40.0, -40.0, -40.0], "magnitudes": [0.18, 0.18, 0.18]},
    {"name": "ABB", "angles": [-40.0, 40.0, 40.0], "magnitudes": [0.18, 0.18, 0.18]},
    {"name": "BBA", "angles": [40.0, 40.0, -40.0], "magnitudes": [0.18, 0.18, 0.18]},
    {"name": "ABAB", "angles": [-40.0, 40.0, -40.0, 40.0], "magnitudes": [0.18] * 4},
    {"name": "BABA", "angles": [40.0, -40.0, 40.0, -40.0], "magnitudes": [0.18] * 4},
]
EVALUATION_INTERCHANGE_PAIRS = [
    ["ab", "ba", 0], ["AAB", "BAA", 0],
    ["ABB", "BBA", 0], ["ABAB", "BABA", 1],
]
ZERO_WORD_NAMES = {1: "zero1", 2: "zero2", 3: "zero3", 4: "zero4"}

GROUNDED_OBSERVABLES = [
    "agent_x", "agent_y", "block_x", "block_y", "block_sin", "block_cos",
    "agent_vx", "agent_vy", "block_vx", "block_vy", "block_angular_velocity",
]
VISUAL_SKETCH_DIM = 256
PROPRIO_PAD_DIM = 64
DECODER_RIDGES = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
OPERATOR_RIDGES = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
MAX_EFFECTIVE_RANK = 12
RANK_BOOTSTRAPS = 512
RANK_PERMUTATIONS = 256
RANK_STABILITY_FLOOR = 0.80
RANK_NULL_QUANTILE = 0.95
MIN_COMMON_RANK = 3
MAX_RANK_DIFFERENCE = 2
CARRIER_RANK = 16
CARRIER_MAP_MAX_CONDITION = 100.0
MIN_MAP_SINGULAR_VALUE = 1e-3
# Four modes times the affine-bilinear design width at rank cap 12:
# 4 * (1 intercept + 12 state + 3 action + 3*12 interactions) = 208.
# It is conservatively over-capacity if the selected common rank is smaller.
NONLINEAR_RANDOM_FEATURES = 208
BOOTSTRAP_DRAWS = 5000
HOLM_ALPHA = 0.05

MIN_DECODER_MEDIAN_R2 = 0.20
MIN_HYBRID_RELATIVE_GAIN = 0.05
MIN_LABEL_FREE_GAIN_RETENTION = 0.50
MAX_GLOBAL_TO_ACTION_SPECIFIC_ERROR_RATIO = 1.25
MAX_CONJUGACY_RELATIVE_ERROR = 0.35
MAX_SAME_MODEL_SPLIT_HALF_ERROR = 0.35
MIN_CONTROL_ADVANTAGE = 0.10
MIN_GROUNDED_INTERCHANGE_COSINE = 0.20
MIN_INTERCHANGE_RELATIVE_ERROR_GAIN = 0.10
MIN_GROUNDED_EFFECT_ENERGY = 1e-6
MAX_ZERO_EDIT_ERROR = 1e-6
MAX_PLANNING_REGRET_DEGRADATION = 0.02
PLANNING_GOALS_PER_RECORD = 4
MIN_EVALUATION_TRAJECTORIES = 12
MIN_EVALUATION_MODE_TRAJECTORIES = 10

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

if RUN_MODE == "smoke":
    ACTIVE_CONSTRUCTION_TRAJECTORY_POOL = CONSTRUCTION_TRAJECTORY_POOL[:256]
    ACTIVE_MODEL_SELECTION_TRAJECTORY_POOL = MODEL_SELECTION_TRAJECTORY_POOL[:256]
    ACTIVE_CALIBRATION_TRAJECTORY_POOL = CALIBRATION_TRAJECTORY_POOL[:256]
    ACTIVE_EVALUATION_TRAJECTORY_POOL = EVALUATION_TRAJECTORY_POOL[:512]
    ACTIVE_CONSTRUCTION_TRAJECTORIES = 1
    ACTIVE_MODEL_SELECTION_TRAJECTORIES = 1
    ACTIVE_CALIBRATION_TRAJECTORIES = 1
    ACTIVE_EVALUATION_TRAJECTORIES = 2
    ACTIVE_RANK_BOOTSTRAPS = 32
    ACTIVE_RANK_PERMUTATIONS = 16
    ACTIVE_BOOTSTRAP_DRAWS = 64
elif RUN_MODE == "pilot":
    ACTIVE_CONSTRUCTION_TRAJECTORY_POOL = CONSTRUCTION_TRAJECTORY_POOL
    ACTIVE_MODEL_SELECTION_TRAJECTORY_POOL = MODEL_SELECTION_TRAJECTORY_POOL
    ACTIVE_CALIBRATION_TRAJECTORY_POOL = CALIBRATION_TRAJECTORY_POOL
    ACTIVE_EVALUATION_TRAJECTORY_POOL = EVALUATION_TRAJECTORY_POOL
    ACTIVE_CONSTRUCTION_TRAJECTORIES = CONSTRUCTION_TRAJECTORIES
    ACTIVE_MODEL_SELECTION_TRAJECTORIES = MODEL_SELECTION_TRAJECTORIES
    ACTIVE_CALIBRATION_TRAJECTORIES = CALIBRATION_TRAJECTORIES
    ACTIVE_EVALUATION_TRAJECTORIES = EVALUATION_TRAJECTORIES
    ACTIVE_RANK_BOOTSTRAPS = RANK_BOOTSTRAPS
    ACTIVE_RANK_PERMUTATIONS = RANK_PERMUTATIONS
    ACTIVE_BOOTSTRAP_DRAWS = BOOTSTRAP_DRAWS
else:
    raise ValueError("RUN_MODE must be 'smoke' or 'pilot'")

PINNED = [
    "real_official_jepa_and_dino_predictions", "exact_pusht_restoration",
    "trajectory_state_family_and_action_composition_splits",
    "construction_rank_and_decoder_only", "calibration_operators_and_map_only",
    "model_selection_locks_rank_ridge_mode_and_capacity",
    "locked_evaluation", "finite_bounded_claim_only", "joint_multistep_paths",
    "free_pre_contact_contact_post_contact", "one_map_all_actions_modes_steps",
    "model_native_internal_interchange", "planning_transport",
    "shared_dinov2_target_is_a_declared_confound", "no_synthetic_fallback",
    "model_free_v1_coverage_amendment", "stable_trajectory_id_geometry",
    "hash_validated_resume", "no_required_colab_secret",
]

assert INTERVENTION_BLOCK in range(6)
assert MAX_WORD_LENGTH == 4 and STATES_PER_TRAJECTORY == len(MODE_LABELS)
_split_pools = [
    CONSTRUCTION_TRAJECTORY_POOL,
    MODEL_SELECTION_TRAJECTORY_POOL,
    CALIBRATION_TRAJECTORY_POOL,
    EVALUATION_TRAJECTORY_POOL,
]
assert all(
    not set(_split_pools[left]) & set(_split_pools[right])
    for left in range(len(_split_pools))
    for right in range(left + 1, len(_split_pools))
)
assert {len(row["angles"]) for row in CORE_WORD_SPECS} == {1, 2, 3}
assert {len(row["angles"]) for row in EVALUATION_WORD_SPECS} == {1, 2, 3, 4}
assert MIN_COMMON_RANK <= MAX_EFFECTIVE_RANK
'''
configuration += "\n\nPROTOCOL_CONFIG_KEYS = " + repr(
    assigned_uppercase_names(configuration)
) + "\n"


installation = STAGE32.installation


setup = STAGE32.setup
setup = setup.replace("Stage 32", "Stage 33").replace("STAGE32", "STAGE33")
setup = setup.replace("from collections import defaultdict\n", "")
setup = setup.replace("import torch.nn.functional as torch_functional\n", "")
setup = setup.replace("stage32_bounded_cross_model", "stage33_bipca")
setup = setup.replace("stage32-source-binder", "stage33-source-binder")
setup = setup.replace(
    "stage32_bounded_confirmation_result_bundle_",
    "stage33_bipca_result_bundle_",
)
setup = setup.replace(
    'CONFIG = build_protocol_config(globals(), PINNED)\nRUN_SIGNATURE = hashlib.sha256(\n    json.dumps(CONFIG, sort_keys=True, allow_nan=False).encode()\n).hexdigest()\nOUT = Path(OUTPUT_DIR) / f"{RUN_MODE}_{RUN_SIGNATURE[:12]}"\nOUT_PREEXISTED = OUT.exists()\nif RUN_MODE == "pilot" and FRESH_RUN_REQUIRED and OUT_PREEXISTED:\n    raise RuntimeError("fresh pilot output already exists; rerun the configuration cell for a fresh automatic nonce")',
    '''CONFIG = build_protocol_config(globals(), PINNED)
REQUESTED_NONCE = str(MANUAL_RUN_NONCE).strip()
request_path = Path(RUN_REQUEST_PATH)
if not REQUESTED_NONCE and request_path.is_file():
    request = json.loads(request_path.read_text())
    if request.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("Stage 33 run request has the wrong protocol_id")
    REQUESTED_NONCE = str(request.get("run_nonce", "")).strip()
if REQUESTED_NONCE:
    if not all(value.isalnum() or value in "-_" for value in REQUESTED_NONCE):
        raise ValueError("manual Stage 33 nonce has an invalid character")
    RUN_NONCE = REQUESTED_NONCE
    CONFIG["RUN_NONCE"] = RUN_NONCE

resume_config = {key: value for key, value in CONFIG.items() if key != "RUN_NONCE"}
RESUME_KEY = hashlib.sha256(
    json.dumps(resume_config, sort_keys=True, allow_nan=False).encode()
).hexdigest()
output_root = Path(OUTPUT_DIR)
output_root.mkdir(parents=True, exist_ok=True)
INCOMPLETE_POINTER = output_root / f".incomplete_{RUN_MODE}_{RESUME_KEY[:16]}.json"
if RESUME_INCOMPLETE and not REQUESTED_NONCE and INCOMPLETE_POINTER.is_file():
    pointer = json.loads(INCOMPLETE_POINTER.read_text())
    if (
        pointer.get("protocol_id") != PROTOCOL_ID
        or pointer.get("protocol_sha256") != NOTEBOOK_PROTOCOL_SHA256
        or pointer.get("resume_key") != RESUME_KEY
    ):
        raise RuntimeError("stale Stage 33 incomplete-run pointer")
    RUN_NONCE = str(pointer["run_nonce"])
    CONFIG["RUN_NONCE"] = RUN_NONCE

RUN_SIGNATURE = hashlib.sha256(
    json.dumps(CONFIG, sort_keys=True, allow_nan=False).encode()
).hexdigest()
OUT = Path(OUTPUT_DIR) / f"{RUN_MODE}_{RUN_SIGNATURE[:12]}"
OUT_PREEXISTED = OUT.exists()
RESUMED_RUN = bool(OUT_PREEXISTED)
if RESUMED_RUN and not RESUME_INCOMPLETE:
    raise RuntimeError("Stage 33 output exists but resume is disabled")''',
)
setup = setup.replace(
    'SUBSPACE_DIR = OUT / "subspaces"',
    'SUBSPACE_DIR = OUT / "predictive_charts"\nRANK_DIR = OUT / "rank"\nREALIZATION_DIR = OUT / "realizations"\nMAP_DIR = OUT / "cross_model_map"\nCHECKPOINT_DIR = OUT / "checkpoints"\nCAUSAL_DIR = OUT / "causal_transport"',
)
setup = setup.replace(
    'ANALYSIS_DIR, INTERVENTION_DIR, EVIDENCE_DIR, PLOT_DIR, LOG_DIR,',
    'ANALYSIS_DIR, INTERVENTION_DIR, EVIDENCE_DIR, PLOT_DIR, LOG_DIR,\n    RANK_DIR, REALIZATION_DIR, MAP_DIR, CHECKPOINT_DIR, CAUSAL_DIR,',
)
setup = setup.replace(
    '(OUT / "FAILURE_TRACE.txt").write_text("PENDING\\n")',
    '''(OUT / "FAILURE_TRACE.txt").write_text("PENDING\\n")
pointer_payload = {
    "protocol_id": PROTOCOL_ID,
    "protocol_sha256": NOTEBOOK_PROTOCOL_SHA256,
    "resume_key": RESUME_KEY,
    "run_nonce": RUN_NONCE,
    "run_signature": RUN_SIGNATURE,
    "out": str(OUT),
    "status": "INCOMPLETE",
}
write_json(INCOMPLETE_POINTER, pointer_payload)
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception as deterministic_error:
    log.warning("deterministic-algorithm request failed: %s", deterministic_error)''',
)
setup = setup.replace(
    '''def atomic_npz(path, **arrays):
    temporary = Path(str(path) + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)''',
    '''def atomic_npz(path, **arrays):
    """Atomically write an array shard and its independently verified digest."""
    path = Path(path)
    temporary = Path(str(path) + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    digest_path = Path(str(path) + ".sha256")
    digest_temporary = Path(str(digest_path) + ".tmp")
    digest_temporary.write_text(sha256_file(path) + "\\n")
    digest_temporary.replace(digest_path)''',
)
setup += r'''


def atomic_checkpoint(name, payload):
    """Write a phase checkpoint whose content hash is verified on reuse."""
    path = CHECKPOINT_DIR / f"{name}.json"
    wrapped = {
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": NOTEBOOK_PROTOCOL_SHA256,
        "run_signature": RUN_SIGNATURE,
        "name": str(name),
        "payload": payload,
    }
    write_json(path, wrapped)
    return path


def write_digest_sidecar(path):
    path = Path(path)
    digest_path = Path(str(path) + ".sha256")
    temporary = Path(str(digest_path) + ".tmp")
    temporary.write_text(sha256_file(path) + "\n")
    temporary.replace(digest_path)
    return digest_path


def validate_digest_sidecar(path):
    path = Path(path)
    digest_path = Path(str(path) + ".sha256")
    if not digest_path.is_file():
        raise RuntimeError(f"missing digest sidecar for {path}")
    if digest_path.read_text().strip() != sha256_file(path):
        raise RuntimeError(f"content hash mismatch for {path}")
    return True


def validated_checkpoint(name):
    path = CHECKPOINT_DIR / f"{name}.json"
    if not path.is_file():
        return None
    wrapped = json.loads(path.read_text())
    expected = {
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": NOTEBOOK_PROTOCOL_SHA256,
        "run_signature": RUN_SIGNATURE,
        "name": str(name),
    }
    if any(wrapped.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"stale checkpoint {path}")
    return wrapped["payload"]


def validate_npz_shard(path, required, identity):
    path = Path(path)
    if not path.is_file():
        return False
    digest_path = Path(str(path) + ".sha256")
    if not digest_path.is_file():
        raise RuntimeError(f"stale shard {path}: missing digest sidecar")
    expected_digest = digest_path.read_text().strip()
    observed_digest = sha256_file(path)
    if expected_digest != observed_digest:
        raise RuntimeError(f"stale shard {path}: content hash mismatch")
    with np.load(path, allow_pickle=False) as payload:
        missing = set(required) - set(payload.files)
        if missing:
            raise RuntimeError(f"stale shard {path}: missing {sorted(missing)}")
        observed = str(payload["identity"].item())
        if observed != str(identity):
            raise RuntimeError(f"stale shard {path}: identity mismatch")
        for key in required:
            value = np.asarray(payload[key])
            if value.dtype == object:
                raise RuntimeError(f"mock/object array in {path}:{key}")
            if value.size and value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
                raise RuntimeError(f"nonfinite shard {path}:{key}")
    return True


def manifest_rows(root, excluded_roots=()):
    root = Path(root)
    excluded = {Path(value) for value in excluded_roots}
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(value == path or value in path.parents for value in excluded):
            continue
        rows.append({
            "path": str(path.relative_to(root)),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        })
    return rows


RUN_STARTED_AT = time.time()
'''


analysis_helpers = function_sources(
    NUMERICAL.read_text(),
    [
        "signature_pseudometric",
        "effective_rank",
        "select_stable_rank",
        "fit_grouped_ridge",
        "fit_affine_bilinear_operator",
        "predict_affine_bilinear",
        "compose_affine_bilinear",
        "fit_whitened_similarity",
        "operator_intertwining_metrics",
        "reachability_observability_diagnostics",
        "interchange_metrics",
        "clustered_bootstrap_interval",
        "holm_adjust",
        "derive_decision",
    ],
)


model_helpers = STAGE32.model_helpers
model_helpers = model_helpers.replace("stage32-jepa-wms", "stage33-jepa-wms")
model_helpers = model_helpers.replace("Stage 32 supports PushT only", "Stage 33 supports PushT only")


design_and_runtime_helpers = r'''# Freeze trajectory families and action compositions before simulator or model access.


def rotate_vector(vector, degrees):
    angle = np.deg2rad(float(degrees))
    matrix = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float64,
    )
    return matrix @ np.asarray(vector, dtype=np.float64)


def normalized(vector):
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("cannot normalize a zero vector")
    return value / norm


def token_definition(symbol):
    table = {
        "L": (-30.0, 0.14), "R": (30.0, 0.14), "S": (0.0, 0.10),
        "a": (-20.0, 0.10), "b": (20.0, 0.22),
        "A": (-40.0, 0.18), "B": (40.0, 0.18),
        "0": (0.0, 0.0),
    }
    if symbol not in table:
        raise KeyError(f"unknown action token {symbol!r}")
    return table[symbol]


def spec_from_name(name):
    angles, magnitudes = [], []
    for symbol in str(name):
        angle, magnitude = token_definition(symbol)
        angles.append(angle)
        magnitudes.append(magnitude)
    return {"name": str(name), "angles": angles, "magnitudes": magnitudes}


def canonical_word_specs():
    by_name = {row["name"]: dict(row) for row in CORE_WORD_SPECS + EVALUATION_WORD_SPECS}
    requested = set()
    for base, donor, step in CALIBRATION_INTERCHANGE_PAIRS + EVALUATION_INTERCHANGE_PAIRS:
        requested.update([base, donor])
        if step >= min(len(base), len(donor)):
            raise ValueError("interchange step lies outside a word")
        requested.add(donor[: step + 1] + base[step + 1 :])
    for length, name in ZERO_WORD_NAMES.items():
        by_name[name] = {
            "name": name, "angles": [0.0] * int(length),
            "magnitudes": [0.0] * int(length), "zero": True,
        }
    for name in requested:
        if name not in by_name:
            by_name[name] = spec_from_name(name)
    rows = list(by_name.values())
    for row in rows:
        row["length"] = len(row["angles"])
        if row["length"] != len(row["magnitudes"]):
            raise RuntimeError(f"malformed action word {row['name']}")
        if not 1 <= row["length"] <= MAX_WORD_LENGTH:
            raise RuntimeError(f"word length outside protocol: {row['name']}")
    return sorted(rows, key=lambda row: (row["length"], row["name"]))


ALL_WORD_SPECS = canonical_word_specs()
WORD_BY_NAME = {row["name"]: row for row in ALL_WORD_SPECS}
CORE_WORD_NAMES = [row["name"] for row in CORE_WORD_SPECS]
EVALUATION_WORD_NAMES = [row["name"] for row in EVALUATION_WORD_SPECS]


def word_actions(record, specification):
    state = np.asarray(record["state"], dtype=np.float64)
    toward = normalized(state[2:4] - state[:2])
    macros = []
    for angle, magnitude in zip(specification["angles"], specification["magnitudes"]):
        pulse = rotate_vector(toward, angle) * float(magnitude)
        macros.append(np.repeat(pulse[None], FRAMESKIP, axis=0))
    actions = np.concatenate(macros, axis=0).astype(np.float32)
    if actions.shape != (int(specification["length"]) * FRAMESKIP, 2):
        raise RuntimeError("action word shape changed")
    # Accumulate invariants in float64.  Stage 32 exposed an order-dependent
    # float32 false failure at 5.96e-7.
    diagnostic = actions.astype(np.float64)
    impulse = np.sum(diagnostic, axis=0)
    energy = float(np.sum(diagnostic**2))
    signed_area = float(sum(
        np.cross(diagnostic[i], diagnostic[j])
        for i in range(len(diagnostic)) for j in range(i + 1, len(diagnostic))
    ))
    return actions, {"impulse": impulse, "energy": energy, "signed_area": signed_area}


def grounded_observables(state):
    value = np.asarray(state, dtype=np.float64)
    if value.shape != (10,):
        raise ValueError("grounded state must have ten dynamic coordinates")
    result = np.asarray([
        value[0] / 512.0, value[1] / 512.0,
        value[2] / 512.0, value[3] / 512.0,
        np.sin(value[4]), np.cos(value[4]),
        value[5] / 50.0, value[6] / 50.0,
        value[7] / 50.0, value[8] / 50.0, value[9] / 5.0,
    ], dtype=np.float64)
    if len(result) != len(GROUNDED_OBSERVABLES):
        raise RuntimeError("grounded observable schema changed")
    return result


def dynamic_state_from_environment(environment):
    return np.asarray([
        *environment.agent.position, *environment.block.position,
        float(environment.block.angle), *environment.agent.velocity,
        *environment.block.velocity, float(environment.block.angular_velocity),
    ], dtype=np.float64)


def reset_dynamic_environment(dynamic_state, goal, seed):
    state = np.asarray(dynamic_state, dtype=np.float64)
    environment = make_environment(REPO, ENVIRONMENT)
    environment.seed(int(seed))
    environment.reset_to_state = np.asarray([*state[:5], 0.0, 0.0], dtype=np.float64)
    environment.reset()
    environment.agent.position = tuple(state[:2])
    environment.block.position = tuple(state[2:4])
    environment.block.angle = float(state[4])
    environment.agent.velocity = tuple(state[5:7])
    environment.block.velocity = tuple(state[7:9])
    environment.block.angular_velocity = float(state[9])
    environment.set_task_goal(np.asarray(goal, dtype=np.float64))
    restored = dynamic_state_from_environment(environment)
    if not np.allclose(restored, state, atol=1e-10, rtol=0):
        environment.close()
        raise RuntimeError("full dynamic-state restoration drifted")
    return environment


def initial_trajectory_record(trajectory_id, split, pool):
    trajectory_id = int(trajectory_id)
    if trajectory_id not in pool:
        raise ValueError("trajectory_id lies outside its declared split pool")
    phase_fraction = float(
        np.mod((trajectory_id + 1) * TRAJECTORY_PHASE_INCREMENT, 1.0)
    )
    phase = 0.413 + 2.0 * np.pi * phase_fraction
    center = np.asarray([256.0, 256.0], dtype=np.float64)
    block = center + 38.0 * np.asarray([np.cos(phase), np.sin(phase)])
    distance_slot = (37 * trajectory_id + DESIGN_SEED) % len(DISTANCE_GRID)
    distance = float(DISTANCE_GRID[distance_slot])
    approach = phase + np.pi + 0.17 * np.sin(3.0 * phase)
    agent = block + distance * np.asarray([np.cos(approach), np.sin(approach)])
    goal_phase = phase + 1.9
    goal_xy = center + 72.0 * np.asarray([np.cos(goal_phase), np.sin(goal_phase)])
    return {
        "trajectory_id": trajectory_id, "split": str(split),
        "state_family_id": trajectory_id,
        "trajectory_geometry_version": TRAJECTORY_GEOMETRY_VERSION,
        "evaluation_seed": int(DESIGN_SEED + 1009 * trajectory_id),
        "task_id": int(TASK_ID_OFFSET + trajectory_id),
        "state": np.asarray([
            agent[0], agent[1], block[0], block[1],
            ((1.41 * phase + np.pi) % (2 * np.pi)) - np.pi,
            0.0, 0.0, 0.0, 0.0, 0.0,
        ], dtype=np.float64),
        "goal": np.asarray([
            goal_xy[0], goal_xy[1],
            ((0.83 * goal_phase + np.pi) % (2 * np.pi)) - np.pi,
        ], dtype=np.float64),
    }


def screening_policy(record):
    state = np.asarray(record["state"], dtype=np.float64)
    toward = normalized(state[2:4] - state[:2])
    tangent = rotate_vector(toward, 90.0)
    macros = (
        [0.20 * toward] * 12
        + [0.17 * normalized(toward + 0.55 * tangent)] * 8
        + [-0.22 * toward] * 10
    )
    return np.concatenate([
        np.repeat(np.asarray(action, dtype=np.float32)[None], FRAMESKIP, axis=0)
        for action in macros
    ], axis=0)


def trajectory_mode_snapshots(record):
    environment = reset_dynamic_environment(
        record["state"], record["goal"], record["evaluation_seed"]
    )
    actions = screening_policy(record)
    boundaries, contacts = [], []
    try:
        for step, action in enumerate(actions):
            if step % FRAMESKIP == 0:
                boundaries.append({
                    "step": int(step),
                    "state": dynamic_state_from_environment(environment),
                })
            _, _, _, info = environment.step(action)
            contacts.append(int(info.get("n_contacts", 0)) > 0)
        boundaries.append({
            "step": int(len(actions)),
            "state": dynamic_state_from_environment(environment),
        })
    finally:
        environment.close()
    contacts = np.asarray(contacts, dtype=bool)
    candidates = {label: [] for label in MODE_LABELS}
    ever_before = False
    for boundary in boundaries[:-1]:
        step = int(boundary["step"])
        previous = bool(contacts[step - 1]) if step > 0 else False
        future = contacts[step : min(step + FRAMESKIP, len(contacts))]
        next_any = bool(np.any(future))
        if not ever_before and not previous and not next_any:
            label = "free"
        elif not previous and next_any:
            label = "pre_contact"
        elif previous and next_any:
            label = "contact"
        elif ever_before and not previous and not next_any:
            label = "post_contact"
        else:
            label = None
        if label is not None:
            candidates[label].append(boundary)
        ever_before = ever_before or previous or next_any
    if not all(candidates[label] for label in MODE_LABELS):
        return None
    chosen = {
        "free": candidates["free"][0],
        "pre_contact": candidates["pre_contact"][-1],
        "contact": candidates["contact"][len(candidates["contact"]) // 2],
        "post_contact": candidates["post_contact"][0],
    }
    rows = []
    for mode_index, label in enumerate(MODE_LABELS):
        snapshot = chosen[label]
        rows.append({
            **{key: value for key, value in record.items() if key != "state"},
            "record_id": int(3300000 + 10 * record["trajectory_id"] + mode_index),
            "mode": label, "mode_index": int(mode_index),
            "trajectory_step": int(snapshot["step"]),
            "state": np.asarray(snapshot["state"], dtype=np.float64),
        })
    return rows


def truth_path(record):
    return TRUTH_DIR / f"truth_{int(record['record_id'])}.npz"


def model_path(short, record):
    return BASELINE_DIR / f"{short}_{int(record['record_id'])}.npz"


def rollout_word(record, specification, retain_visual=True):
    environment = reset_dynamic_environment(
        record["state"], record["goal"], record["evaluation_seed"]
    )
    actions, invariants = word_actions(record, specification)
    path_states, path_visuals, contacts = [], [], []
    initial_visual = np.asarray(environment.render("rgb_array")).copy()
    initial_proprio = np.asarray([
        *environment.agent.position, *environment.agent.velocity
    ], dtype=np.float32)
    try:
        for step, action in enumerate(actions, start=1):
            observation, _, _, info = environment.step(action)
            contacts.append(int(info.get("n_contacts", 0)))
            if step % FRAMESKIP == 0:
                path_states.append(dynamic_state_from_environment(environment))
                if retain_visual:
                    path_visuals.append(np.asarray(observation["visual"]).copy())
    finally:
        environment.close()
    first_contact = next((index for index, value in enumerate(contacts) if value > 0), -1)
    return {
        "initial_visual": initial_visual,
        "initial_proprio": initial_proprio,
        "path_states": np.asarray(path_states, dtype=np.float64),
        "path_visuals": np.asarray(path_visuals, dtype=np.uint8),
        "contacts": np.asarray(contacts, dtype=np.int64),
        "first_contact_step": int(first_contact),
        **invariants,
    }


def count_sketch(values, dimension, seed):
    array = np.asarray(values, dtype=np.float32).reshape(len(values), -1)
    rng = np.random.default_rng(int(seed))
    buckets = rng.integers(0, int(dimension), size=array.shape[1], dtype=np.int64)
    signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=array.shape[1])
    result = np.stack([
        np.bincount(buckets, weights=row * signs, minlength=int(dimension))
        for row in array
    ])
    return result.astype(np.float64) / math.sqrt(max(array.shape[1] / dimension, 1.0))


def forward_with_trace(bundle, initial, actions, horizon, intervention_by_step=None):
    """Run the real recurrent predictor and retain a carrier at every step."""
    selected_block = INTERVENTION_BLOCK
    module = bundle["blocks"][selected_block]
    hook_kind = bundle["hook_kinds"][selected_block]
    captures, context = [], {"step": -1}
    intervention_by_step = intervention_by_step or {}

    def hook(_module, inputs, output):
        post = output if hook_kind == "direct" else inputs[0] + output
        view = post.view(post.shape[0], post.shape[1] // 256, 256, post.shape[-1])
        captures.append(view[:, -1].detach().clone())
        delta = intervention_by_step.get(int(context["step"]))
        if delta is None:
            return output
        delta = delta.to(output.device, output.dtype)
        if tuple(delta.shape) != tuple(view[:, -1].shape):
            raise RuntimeError("step intervention does not match carrier shape")
        if hook_kind == "direct":
            changed = post.clone().view_as(view)
            changed[:, -1] = changed[:, -1] + delta
            return changed.reshape_as(output)
        changed = output.clone()
        changed_view = changed.view(
            changed.shape[0], changed.shape[1] // 256, 256, changed.shape[-1]
        )
        changed_view[:, -1] = changed_view[:, -1] + delta
        return changed_view.reshape_as(output)

    handle = module.register_forward_hook(hook)
    model = bundle["model"]
    try:
        batch = actions.shape[1]
        action_batch = actions[:horizon].permute(1, 0, 2).contiguous()
        with torch.inference_mode():
            action_features = model.model.encode_act(action_batch)
            visual_history = initial["visual"].expand(
                batch, *initial["visual"].shape[1:]
            ).detach().clone()
            proprio_history = initial["proprio"].expand(
                batch, *initial["proprio"].shape[1:]
            ).detach().clone()
            visual_paths, proprio_paths = [], []
            for step in range(horizon):
                context["step"] = step
                visual, _, proprio = model.model.forward_pred(
                    visual_history[:, -model.ctxt_window :],
                    action_features[:, : step + 1][:, -model.ctxt_window :],
                    proprio_history[:, -model.ctxt_window :],
                )
                next_visual, next_proprio = visual[:, -1:], proprio[:, -1:]
                tokens = next_visual[:, 0, 0].flatten(1, 2)
                if tuple(tokens.shape[1:]) != (256, 384):
                    raise RuntimeError("predicted visual grid changed")
                visual_paths.append(tokens)
                proprio_paths.append(next_proprio[:, 0])
                visual_history = torch.cat([visual_history, next_visual], dim=1)
                proprio_history = torch.cat([proprio_history, next_proprio], dim=1)
        if len(captures) != horizon:
            raise RuntimeError("carrier hook did not fire exactly once per recurrent step")
        if "PROVENANCE_COUNTS" in globals():
            short = bundle["short"]
            PROVENANCE_COUNTS["native_forward_pred_calls"][short] += int(horizon)
            PROVENANCE_COUNTS["native_predicted_word_sequences"][short] += int(batch)
        return (
            torch.stack(visual_paths, dim=1),
            torch.stack(proprio_paths, dim=1),
            torch.stack(captures, dim=1),
        )
    finally:
        handle.remove()


def encoded_initial(bundle, record, payload):
    observation = to_model_observation(
        np.asarray(payload["initial_visual"]),
        np.asarray(payload["initial_proprio"]),
    )
    with torch.inference_mode():
        encoded = bundle["model"].encode(observation)
    return {key: value.detach() for key, value in encoded.items()}


def encoded_initial_from_record(bundle, record):
    """Encode a restored state directly, including generated successor states."""
    environment = reset_dynamic_environment(
        record["state"], record["goal"], record["evaluation_seed"]
    )
    try:
        visual = np.asarray(environment.render("rgb_array")).copy()
        proprio = np.asarray([
            *environment.agent.position, *environment.agent.velocity
        ], dtype=np.float32)
    finally:
        environment.close()
    observation = to_model_observation(visual, proprio)
    with torch.inference_mode():
        encoded = bundle["model"].encode(observation)
    return {key: value.detach() for key, value in encoded.items()}


def grouped_model_words(bundle, record, names, intervention_lookup=None):
    # Direct restoration is intentionally used here: transition-prefix states
    # do not have pre-materialized truth shards, and must enter the native
    # encoder through the same pixels/proprioception as the original records.
    initial = encoded_initial_from_record(bundle, record)
    outputs = {}
    traces = {}
    for length in sorted({WORD_BY_NAME[name]["length"] for name in names}):
        selected = [name for name in names if WORD_BY_NAME[name]["length"] == length]
        actions = np.stack([
            word_actions(record, WORD_BY_NAME[name])[0] for name in selected
        ])
        action_tensor = model_action_tensor(bundle["preprocessor"], actions, length)
        step_edits = None
        if intervention_lookup:
            step_edits = {}
            for step in range(length):
                rows = []
                any_edit = False
                for name in selected:
                    delta = intervention_lookup.get((name, step))
                    if delta is None:
                        rows.append(np.zeros((256, bundle["carrier_width"]), dtype=np.float32))
                    else:
                        rows.append(np.asarray(delta, dtype=np.float32))
                        any_edit = True
                if any_edit:
                    step_edits[step] = torch.as_tensor(np.stack(rows), device="cuda")
        visual, proprio, carrier = forward_with_trace(
            bundle, initial, action_tensor, length, intervention_by_step=step_edits
        )
        visual = visual.detach().float().cpu().numpy()
        proprio = proprio.detach().float().cpu().numpy()
        carrier = carrier.detach().float().cpu().numpy()
        for index, name in enumerate(selected):
            outputs[name] = (visual[index], proprio[index])
            traces[name] = carrier[index]
        del action_tensor, visual, proprio, carrier
    return outputs, traces


def model_feature_rows(outputs, names):
    rows = []
    for name in names:
        visual, proprio = outputs[name]
        for step in range(len(visual)):
            sketch = count_sketch(
                visual[step : step + 1], VISUAL_SKETCH_DIM,
                stable_seed(DECODER_SEED, "visual_sketch"),
            )[0]
            rows.append(np.concatenate([sketch, proprio[step].reshape(-1)]))
    return np.asarray(rows, dtype=np.float64)


def carrier_delta_rows(traces, pairs):
    rows, metadata = [], []
    for base, donor, step in pairs:
        hybrid = donor[: step + 1] + base[step + 1 :]
        if base not in traces or donor not in traces or hybrid not in traces:
            raise RuntimeError(f"missing interchange trace for {base}/{donor}/{hybrid}")
        rows.append((traces[donor][step] - traces[base][step]).reshape(-1))
        metadata.append({
            "base": base, "donor": donor, "hybrid": hybrid, "step": int(step),
        })
    return np.asarray(rows, dtype=np.float32), metadata


def decoder_targets(record, names):
    with np.load(truth_path(record), allow_pickle=False) as truth:
        word_names = [str(value) for value in truth["word_names"]]
        lookup = {name: index for index, name in enumerate(word_names)}
        rows = []
        for name in names:
            index = lookup[name]
            length = int(truth["word_lengths"][index])
            rows.extend(truth["path_observables"][index, :length])
    return np.asarray(rows, dtype=np.float64)


def trajectory_split_manifest():
    return {
        "construction_pool": ACTIVE_CONSTRUCTION_TRAJECTORY_POOL,
        "model_selection_pool": ACTIVE_MODEL_SELECTION_TRAJECTORY_POOL,
        "calibration_pool": ACTIVE_CALIBRATION_TRAJECTORY_POOL,
        "evaluation_pool": ACTIVE_EVALUATION_TRAJECTORY_POOL,
        "targets": {
            "construction": ACTIVE_CONSTRUCTION_TRAJECTORIES,
            "model_selection": ACTIVE_MODEL_SELECTION_TRAJECTORIES,
            "calibration": ACTIVE_CALIBRATION_TRAJECTORIES,
            "evaluation": ACTIVE_EVALUATION_TRAJECTORIES,
        },
        "entire_trajectory_disjoint": True,
        "state_family_is_trajectory": True,
        "core_words": CORE_WORD_SPECS,
        "calibration_interchange_pairs": CALIBRATION_INTERCHANGE_PAIRS,
        "evaluation_words": EVALUATION_WORD_SPECS,
        "evaluation_interchange_pairs": EVALUATION_INTERCHANGE_PAIRS,
        "trajectory_geometry_version": TRAJECTORY_GEOMETRY_VERSION,
        "trajectory_phase_increment": TRAJECTORY_PHASE_INCREMENT,
        "v1_model_outputs_observed_before_amendment": False,
        "model_outputs_used": False,
        "physical_effect_magnitudes_used": False,
    }


write_json(DESIGN_DIR / "trajectory_action_split_manifest.json", trajectory_split_manifest())
write_json(DESIGN_DIR / "action_word_manifest.json", {
    "words": ALL_WORD_SPECS,
    "grounded_observables": GROUNDED_OBSERVABLES,
    "finite_bank_only": True,
    "universal_predictive_equivalence_claimed": False,
    "terminal_mean_sufficiency_claimed": False,
})
DESIGN_FREEZE = {
    "created_before_simulator_or_model_data": True,
    "protocol_id": PROTOCOL_ID,
    "run_signature": RUN_SIGNATURE,
    "split_manifest_sha256": sha256_file(DESIGN_DIR / "trajectory_action_split_manifest.json"),
    "word_manifest_sha256": sha256_file(DESIGN_DIR / "action_word_manifest.json"),
    "models_loaded": False,
    "trajectory_geometry_version": TRAJECTORY_GEOMETRY_VERSION,
    "v1_model_free_coverage_amendment": True,
    "evaluation_rank_map_or_mode_selection_allowed": False,
}
write_json(DESIGN_DIR / "design_freeze.json", DESIGN_FREEZE)
print(json.dumps({
    "run_mode": RUN_MODE,
    "trajectory_targets": trajectory_split_manifest()["targets"],
    "core_word_count": len(CORE_WORD_NAMES),
    "evaluation_word_count": len(EVALUATION_WORD_NAMES),
    "all_materialized_word_count": len(ALL_WORD_SPECS),
    "core_word_lengths": sorted({WORD_BY_NAME[name]["length"] for name in CORE_WORD_NAMES}),
    "evaluation_word_lengths": sorted({WORD_BY_NAME[name]["length"] for name in EVALUATION_WORD_NAMES}),
}, indent=2))
'''


physical_truth = r'''# Select complete physical trajectories and materialize exact multi-step truth without model access.

PROVENANCE_COUNTS = {
    "trajectory_candidates_screened": 0,
    "trajectory_families_selected": 0,
    "physical_state_records": 0,
    "truth_words_generated": 0,
    "model_record_forwards": {"jepa": 0, "dino": 0},
    "native_forward_pred_calls": {"jepa": 0, "dino": 0},
    "native_predicted_word_sequences": {"jepa": 0, "dino": 0},
    "patched_forwards": 0,
    "validated_cache_hits": 0,
}


def json_record(row):
    return {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in row.items()
    }


def restore_json_record(row):
    result = dict(row)
    result["state"] = np.asarray(result["state"], dtype=np.float64)
    result["goal"] = np.asarray(result["goal"], dtype=np.float64)
    return result


def select_complete_trajectories(split, pool, target):
    path = DESIGN_DIR / f"selected_{split}_trajectories.json"
    if path.is_file():
        validate_digest_sidecar(path)
        payload = json.loads(path.read_text())
        if (
            payload.get("protocol_id") != PROTOCOL_ID
            or payload.get("split") != split
            or payload.get("pool") != list(pool)
            or payload.get("target") != int(target)
            or payload.get("trajectory_geometry_version")
            != TRAJECTORY_GEOMETRY_VERSION
        ):
            raise RuntimeError(f"stale selected-trajectory manifest for {split}")
        records = [restore_json_record(row) for row in payload["records"]]
        if len({row["trajectory_id"] for row in records}) != int(target):
            raise RuntimeError(f"selected {split} trajectory count changed")
        PROVENANCE_COUNTS["validated_cache_hits"] += 1
        return records
    selected, screen_rows = [], []
    for trajectory_id in pool:
        base = initial_trajectory_record(trajectory_id, split, list(pool))
        snapshots = trajectory_mode_snapshots(base)
        PROVENANCE_COUNTS["trajectory_candidates_screened"] += 1
        screen_rows.append({
            "split": split, "trajectory_id": int(trajectory_id),
            "complete_four_mode_family": bool(snapshots is not None),
        })
        if snapshots is not None:
            selected.extend(snapshots)
        selected_count = len({row["trajectory_id"] for row in selected})
        write_json(OUT / f"physical_screen_{split}_progress.json", {
            "screened": len(screen_rows), "pool": len(pool),
            "selected": selected_count,
            "target": int(target), "last_trajectory_id": int(trajectory_id),
        })
        if selected_count == int(target):
            break
    write_csv(EVIDENCE_DIR / f"physical_screen_{split}_rows.csv", screen_rows)
    if len({row["trajectory_id"] for row in selected}) != int(target):
        raise RuntimeError(
            f"{split} produced fewer than {target} complete four-mode trajectories"
        )
    payload = {
        "protocol_id": PROTOCOL_ID, "split": split, "pool": list(pool),
        "target": int(target), "selection_uses_contact_timing_only": True,
        "trajectory_geometry_version": TRAJECTORY_GEOMETRY_VERSION,
        "model_outputs_used": False, "effect_magnitude_used": False,
        "records": [json_record(row) for row in selected],
    }
    write_json(path, payload)
    write_digest_sidecar(path)
    PROVENANCE_COUNTS["trajectory_families_selected"] += int(target)
    return selected


def generate_truth_record(record):
    path = truth_path(record)
    identity = f"{PROTOCOL_ID}:{RUN_SIGNATURE}:{record['record_id']}:truth-v1"
    required = {
        "identity", "word_names", "word_lengths", "actions", "action_mask",
        "path_states", "path_observables", "path_mask", "path_visuals",
        "contact_counts", "first_contact_steps", "impulses", "energies",
        "signed_areas", "initial_visual", "initial_proprio",
    }
    if validate_npz_shard(path, required, identity):
        PROVENANCE_COUNTS["validated_cache_hits"] += 1
        return path
    words = ALL_WORD_SPECS
    action_rows = np.zeros((len(words), MAX_WORD_LENGTH * FRAMESKIP, 2), dtype=np.float32)
    action_mask = np.zeros((len(words), MAX_WORD_LENGTH * FRAMESKIP), dtype=bool)
    path_states = np.zeros((len(words), MAX_WORD_LENGTH, 10), dtype=np.float64)
    path_observables = np.zeros(
        (len(words), MAX_WORD_LENGTH, len(GROUNDED_OBSERVABLES)), dtype=np.float64
    )
    path_mask = np.zeros((len(words), MAX_WORD_LENGTH), dtype=bool)
    path_visuals = []
    contacts = np.zeros((len(words), MAX_WORD_LENGTH * FRAMESKIP), dtype=np.int64)
    first_contact, impulses, energies, signed_areas = [], [], [], []
    initial_visual = initial_proprio = None
    for index, word in enumerate(words):
        result = rollout_word(record, word, retain_visual=True)
        length, action_steps = int(word["length"]), int(word["length"] * FRAMESKIP)
        actions, _ = word_actions(record, word)
        action_rows[index, :action_steps] = actions
        action_mask[index, :action_steps] = True
        path_states[index, :length] = result["path_states"]
        path_observables[index, :length] = np.stack([
            grounded_observables(value) for value in result["path_states"]
        ])
        path_mask[index, :length] = True
        padded_visual = np.zeros(
            (MAX_WORD_LENGTH, *result["path_visuals"].shape[1:]), dtype=np.uint8
        )
        padded_visual[:length] = result["path_visuals"]
        path_visuals.append(padded_visual)
        contacts[index, :action_steps] = result["contacts"]
        first_contact.append(result["first_contact_step"])
        impulses.append(result["impulse"])
        energies.append(result["energy"])
        signed_areas.append(result["signed_area"])
        initial_visual = result["initial_visual"]
        initial_proprio = result["initial_proprio"]
        PROVENANCE_COUNTS["truth_words_generated"] += 1
    atomic_npz(
        path,
        identity=np.asarray(identity),
        word_names=np.asarray([row["name"] for row in words]),
        word_lengths=np.asarray([row["length"] for row in words], dtype=np.int64),
        actions=action_rows,
        action_mask=action_mask,
        path_states=path_states,
        path_observables=path_observables,
        path_mask=path_mask,
        path_visuals=np.stack(path_visuals),
        contact_counts=contacts,
        first_contact_steps=np.asarray(first_contact, dtype=np.int64),
        impulses=np.asarray(impulses, dtype=np.float64),
        energies=np.asarray(energies, dtype=np.float64),
        signed_areas=np.asarray(signed_areas, dtype=np.float64),
        initial_visual=np.asarray(initial_visual, dtype=np.uint8),
        initial_proprio=np.asarray(initial_proprio, dtype=np.float32),
    )
    PROVENANCE_COUNTS["physical_state_records"] += 1
    return path


SELECTED_RECORDS = {}
ALL_RECORDS = []
if not PIPELINE_FAILED:
    try:
        verify_executed_notebook_through(
            "# Select complete physical trajectories and materialize exact multi-step truth without model access."
        )
        REPO = configure_repo()
        split_arguments = {
            "construction": (
                ACTIVE_CONSTRUCTION_TRAJECTORY_POOL, ACTIVE_CONSTRUCTION_TRAJECTORIES
            ),
            "model_selection": (
                ACTIVE_MODEL_SELECTION_TRAJECTORY_POOL,
                ACTIVE_MODEL_SELECTION_TRAJECTORIES,
            ),
            "calibration": (
                ACTIVE_CALIBRATION_TRAJECTORY_POOL, ACTIVE_CALIBRATION_TRAJECTORIES
            ),
            "evaluation": (
                ACTIVE_EVALUATION_TRAJECTORY_POOL, ACTIVE_EVALUATION_TRAJECTORIES
            ),
        }
        for split, (pool, target) in split_arguments.items():
            SELECTED_RECORDS[split] = select_complete_trajectories(split, pool, target)
            ALL_RECORDS.extend(SELECTED_RECORDS[split])
        trajectory_sets = {
            split: {row["trajectory_id"] for row in rows}
            for split, rows in SELECTED_RECORDS.items()
        }
        split_names = ["construction", "model_selection", "calibration", "evaluation"]
        if any(
            trajectory_sets[left] & trajectory_sets[right]
            for left_index, left in enumerate(split_names)
            for right in split_names[left_index + 1 :]
        ):
            raise RuntimeError("trajectory split leakage")
        for index, record in enumerate(ALL_RECORDS):
            generate_truth_record(record)
            write_json(OUT / "truth_progress.json", {
                "completed_records": index + 1, "total_records": len(ALL_RECORDS),
                "last_record_id": int(record["record_id"]),
            })
        selection = {
            "created_before_model_loading": True,
            "trajectory_ids": {
                split: sorted(values) for split, values in trajectory_sets.items()
            },
            "record_ids": {
                split: [int(row["record_id"]) for row in rows]
                for split, rows in SELECTED_RECORDS.items()
            },
            "mode_counts": {
                split: {
                    mode: sum(row["mode"] == mode for row in rows)
                    for mode in MODE_LABELS
                }
                for split, rows in SELECTED_RECORDS.items()
            },
            "entire_trajectory_disjoint": True,
            "selection_uses_contact_timing_only": True,
            "model_outputs_used": False,
        }
        write_json(DESIGN_DIR / "physical_selection_freeze.json", selection)
        print(json.dumps({
            "selected_trajectories": {
                split: len(values) for split, values in trajectory_sets.items()
            },
            "selected_state_records": {
                split: len(rows) for split, rows in SELECTED_RECORDS.items()
            },
            "state_family_counts": {
                split: len({row["state_family_id"] for row in rows})
                for split, rows in SELECTED_RECORDS.items()
            },
            "mode_counts": selection["mode_counts"],
            "cache_hits_so_far": PROVENANCE_COUNTS["validated_cache_hits"],
        }, indent=2))
        atomic_checkpoint("physical_truth_complete", selection)
        memory_report("stage33_physical_truth_complete")
    except Exception:
        record_failure("stage33_physical_trajectory_selection_or_truth")
'''


construction_and_models = r'''# Fit grounded readouts, predictive charts, and carrier bases on construction trajectories only.


def stable_seed(root, *parts):
    payload = ":".join([str(int(root)), *map(str, parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def names_for_split(split):
    if split in {"construction", "model_selection", "calibration"}:
        names = set(CORE_WORD_NAMES)
        pairs = CALIBRATION_INTERCHANGE_PAIRS
    else:
        names = set(CORE_WORD_NAMES + EVALUATION_WORD_NAMES)
        pairs = EVALUATION_INTERCHANGE_PAIRS
    for base, donor, step in pairs:
        names.update([base, donor, donor[: step + 1] + base[step + 1 :]])
    for name in list(names):
        names.add(ZERO_WORD_NAMES[len(name)])
    return sorted(names, key=lambda name: (WORD_BY_NAME[name]["length"], name))


def feature_tensor_from_outputs(outputs, names):
    tensor = np.zeros(
        (len(names), MAX_WORD_LENGTH, VISUAL_SKETCH_DIM + PROPRIO_PAD_DIM), dtype=np.float32
    )
    # The proprio width is checkpoint-defined and much smaller than the frozen pad.  A
    # fixed padded chart makes every saved shard schema-stable without mixing
    # the JEPA and DINO fitted decoders.
    widths = []
    for word_index, name in enumerate(names):
        visual, proprio = outputs[name]
        for step in range(len(visual)):
            sketch = count_sketch(
                visual[step : step + 1], VISUAL_SKETCH_DIM,
                stable_seed(DECODER_SEED, "visual_sketch"),
            )[0]
            flat_proprio = np.asarray(proprio[step], dtype=np.float32).reshape(-1)
            if len(flat_proprio) > PROPRIO_PAD_DIM:
                raise RuntimeError("proprio prediction exceeded frozen padding")
            tensor[word_index, step, :VISUAL_SKETCH_DIM] = sketch.astype(np.float32)
            tensor[
                word_index, step,
                VISUAL_SKETCH_DIM : VISUAL_SKETCH_DIM + len(flat_proprio),
            ] = flat_proprio
            widths.append(len(flat_proprio))
    if len(set(widths)) != 1:
        raise RuntimeError("proprio prediction width changed within a model")
    return tensor, int(widths[0])


def response_rows_from_feature_tensor(tensor, names):
    rows, metadata = [], []
    for word_index, name in enumerate(names):
        length = int(WORD_BY_NAME[name]["length"])
        for step in range(length):
            rows.append(tensor[word_index, step])
            metadata.append((name, step))
    return np.asarray(rows, dtype=np.float64), metadata


def truth_rows(record, names):
    with np.load(truth_path(record), allow_pickle=False) as truth:
        lookup = {str(name): index for index, name in enumerate(truth["word_names"])}
        rows, metadata = [], []
        for name in names:
            index = lookup[name]
            length = int(truth["word_lengths"][index])
            for step in range(length):
                rows.append(truth["path_observables"][index, step])
                metadata.append((name, step))
    return np.asarray(rows, dtype=np.float64), metadata


def channel_scale_from_deltas(deltas, width):
    values = np.asarray(deltas, dtype=np.float64).reshape(-1, 256, int(width))
    rms = np.sqrt(np.mean(values**2, axis=(0, 1)) + 1e-8)
    median = float(np.median(rms))
    floor = max(1e-4 * median, 1e-8)
    return np.maximum(rms, floor)


def fit_dual_basis(deltas, width, rank, seed):
    scale = channel_scale_from_deltas(deltas, width)
    white = (
        np.asarray(deltas, dtype=np.float64).reshape(-1, 256, int(width)) / scale
    ).reshape(len(deltas), -1)
    white -= np.mean(white, axis=0, keepdims=True)
    gram = white @ white.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    keep = min(int(rank), int(np.sum(eigenvalues > max(eigenvalues[0] * 1e-10, 1e-12))))
    if keep < MIN_COMMON_RANK:
        raise RuntimeError("construction carrier deltas are rank deficient")
    singular = np.sqrt(eigenvalues[:keep])
    basis = white.T @ (eigenvectors[:, :keep] / singular[None])
    basis, _ = np.linalg.qr(basis)
    # A dimension- and norm-matched empirical-span control.  It is sampled
    # before calibration/evaluation and orthogonalized against the primary
    # construction basis; it is not an ambient Gaussian straw man.
    rng = np.random.default_rng(int(seed))
    candidate = white.T @ rng.normal(size=(len(white), keep))
    candidate -= basis[:, :keep] @ (basis[:, :keep].T @ candidate)
    random_basis, random_r = np.linalg.qr(candidate)
    if np.min(np.abs(np.diag(random_r)[:keep])) <= 1e-10:
        raise RuntimeError("matched empirical-span control is rank deficient")
    return {
        "mean": np.mean(
            (np.asarray(deltas, dtype=np.float64).reshape(-1, 256, int(width)) / scale).reshape(len(deltas), -1),
            axis=0,
        ),
        "scale": scale,
        "basis": basis[:, :keep].astype(np.float32),
        "random_basis": random_basis[:, :keep].astype(np.float32),
        "singular_values": singular,
        "rank": int(keep),
    }


def project_carrier_deltas(deltas, carrier_fit, width):
    white = (
        np.asarray(deltas, dtype=np.float64).reshape(-1, 256, int(width))
        / carrier_fit["scale"]
    ).reshape(len(deltas), -1)
    white -= carrier_fit["mean"]
    return white @ np.asarray(carrier_fit["basis"], dtype=np.float64)


def reconstruct_carrier_delta(coordinates, carrier_fit, width):
    white = np.asarray(coordinates, dtype=np.float64) @ np.asarray(
        carrier_fit["basis"], dtype=np.float64
    ).T
    # Interventions are deltas, so do not add the sample mean back.
    return (
        white.reshape(-1, 256, int(width)) * carrier_fit["scale"]
    ).astype(np.float32)


def reconstruct_with_basis(coordinates, basis, carrier_fit, width):
    white = np.asarray(coordinates, dtype=np.float64) @ np.asarray(
        basis, dtype=np.float64
    ).T
    return (
        white.reshape(-1, 256, int(width)) * carrier_fit["scale"]
    ).astype(np.float32)


def core_signature_from_grounded(grounded, names):
    lookup = {name: index for index, name in enumerate(names)}
    pieces = []
    for name in CORE_WORD_NAMES:
        index = lookup[name]
        length = int(WORD_BY_NAME[name]["length"])
        pieces.append(np.asarray(grounded[index, :length], dtype=np.float64).reshape(-1))
    return np.concatenate(pieces)


def chart_coordinates(signature, chart, rank):
    standardized = (
        np.asarray(signature, dtype=np.float64) - np.asarray(chart["mean"])
    ) / np.asarray(chart["scale"])
    return standardized @ np.asarray(chart["basis"])[:, : int(rank)]


def record_after_word(record, name):
    specification = WORD_BY_NAME.get(str(name))
    if specification is None:
        specification = spec_from_name(str(name))
        specification["length"] = len(str(name))
    result = rollout_word(record, specification, retain_visual=False)
    successor = dict(record)
    successor["state"] = np.asarray(result["path_states"][-1], dtype=np.float64)
    successor["record_id"] = int(
        stable_seed(DESIGN_SEED, "successor", record["record_id"], name)
    )
    return successor, result


def transition_path(short, record, split):
    return BASELINE_DIR / f"transitions_{short}_{split}_{int(record['record_id'])}.npz"


def action_feature(symbol):
    angle, magnitude = token_definition(symbol)
    radians = np.deg2rad(float(angle))
    return np.asarray([
        float(magnitude) * np.cos(radians),
        float(magnitude) * np.sin(radians),
        float(magnitude),
    ], dtype=np.float64)


def transition_prefixes(split):
    if split == "model_selection":
        names = ["L", "R", "S"]
    elif split == "calibration":
        names = ["L", "R", "S"]
        for base, donor, step in CALIBRATION_INTERCHANGE_PAIRS:
            names.extend([base, donor, donor[: step + 1] + base[step + 1 :]])
    else:
        names = list(EVALUATION_WORD_NAMES)
        for base, donor, step in EVALUATION_INTERCHANGE_PAIRS:
            names.extend([base, donor, donor[: step + 1] + base[step + 1 :]])
    names = sorted(set(names), key=lambda value: (len(value), value))
    prefixes = sorted(
        {name[:step] for name in names for step in range(1, len(name) + 1)},
        key=lambda value: (len(value), value),
    )
    return names, prefixes


def source_mode_sequence(record, word, rollout):
    sequence = [str(record["mode"])]
    ever_contact = str(record["mode"]) in {"contact", "post_contact"}
    contacts = np.asarray(rollout["contacts"], dtype=np.int64)
    for step in range(1, len(word)):
        previous = contacts[(step - 1) * FRAMESKIP : step * FRAMESKIP]
        future = contacts[step * FRAMESKIP : (step + 1) * FRAMESKIP]
        previous_any = bool(np.any(previous > 0))
        future_any = bool(np.any(future > 0))
        ever_contact = ever_contact or previous_any
        if previous_any:
            label = "contact"
        elif not ever_contact and future_any:
            label = "pre_contact"
        elif ever_contact:
            label = "post_contact"
        else:
            label = "free"
        sequence.append(label)
    return sequence


def construction_model_data(bundle):
    names = names_for_split("construction")
    feature_rows, target_rows, groups = [], [], []
    signatures_by_record, carrier_delta_blocks = {}, []
    for index, record in enumerate(SELECTED_RECORDS["construction"]):
        outputs, traces = grouped_model_words(bundle, record, names)
        tensor, proprio_width = feature_tensor_from_outputs(outputs, names)
        x, x_meta = response_rows_from_feature_tensor(tensor, names)
        y, y_meta = truth_rows(record, names)
        if x_meta != y_meta:
            raise RuntimeError("construction prediction/truth row order mismatch")
        feature_rows.append(x)
        target_rows.append(y)
        groups.extend([int(record["trajectory_id"])] * len(x))
        deltas, _ = carrier_delta_rows(traces, CALIBRATION_INTERCHANGE_PAIRS)
        carrier_delta_blocks.append(deltas)
        signatures_by_record[int(record["record_id"])] = {
            "tensor": tensor, "names": names, "proprio_width": proprio_width,
        }
        PROVENANCE_COUNTS["model_record_forwards"][bundle["short"]] += 1
        write_json(OUT / f"construction_{bundle['short']}_progress.json", {
            "completed": index + 1,
            "total": len(SELECTED_RECORDS["construction"]),
            "last_record_id": int(record["record_id"]),
        })
    decoder = fit_grouped_ridge(
        np.concatenate(feature_rows), np.concatenate(target_rows),
        np.asarray(groups, dtype=np.int64), penalties=DECODER_RIDGES,
        folds=min(4, len(set(groups))), seed=stable_seed(DECODER_SEED, bundle["short"]),
    )
    carrier = fit_dual_basis(
        np.concatenate(carrier_delta_blocks), bundle["carrier_width"], CARRIER_RANK,
        stable_seed(CONTROL_SEED, bundle["short"], "carrier_control"),
    )
    core_signatures, signature_groups, signature_modes, record_ids = [], [], [], []
    for record in SELECTED_RECORDS["construction"]:
        saved = signatures_by_record[int(record["record_id"])]
        tensor, names = saved["tensor"], saved["names"]
        lookup = {name: index for index, name in enumerate(names)}
        pieces = []
        for name in CORE_WORD_NAMES:
            word_index = lookup[name]
            length = int(WORD_BY_NAME[name]["length"])
            prediction = tensor[word_index, :length].astype(np.float64)
            grounded = prediction @ decoder["weight"] + decoder["intercept"]
            pieces.append(grounded.reshape(-1))
        core_signatures.append(np.concatenate(pieces))
        signature_groups.append(int(record["trajectory_id"]))
        signature_modes.append(int(record["mode_index"]))
        record_ids.append(int(record["record_id"]))
    signatures = np.asarray(core_signatures, dtype=np.float64)
    mean = np.mean(signatures, axis=0)
    scale = np.std(signatures, axis=0, ddof=1)
    scale = np.maximum(scale, 1e-6)
    standardized = (signatures - mean) / scale
    rank_result = select_stable_rank(
        standardized, np.asarray(signature_groups, dtype=np.int64),
        max_rank=MAX_EFFECTIVE_RANK,
        n_bootstrap=ACTIVE_RANK_BOOTSTRAPS,
        n_permutations=ACTIVE_RANK_PERMUTATIONS,
        stability_floor=RANK_STABILITY_FLOOR,
        null_quantile=RANK_NULL_QUANTILE,
        seed=stable_seed(RANK_SEED, bundle["short"]),
    )
    _, singular, right = np.linalg.svd(standardized, full_matrices=False)
    chart_rank = min(MAX_EFFECTIVE_RANK, len(singular), right.shape[0])
    chart = {
        "mean": mean, "scale": scale,
        "basis": right[:chart_rank].T.astype(np.float64),
        "singular_values": singular,
        "rank_result": rank_result,
        "record_ids": np.asarray(record_ids, dtype=np.int64),
        "groups": np.asarray(signature_groups, dtype=np.int64),
        "modes": np.asarray(signature_modes, dtype=np.int64),
    }
    return decoder, carrier, chart


def save_frozen_model_artifacts(short, decoder, carrier, chart, width):
    decoder_path = SUBSPACE_DIR / f"decoder_{short}.npz"
    carrier_path = SUBSPACE_DIR / f"carrier_basis_{short}.npz"
    chart_path = SUBSPACE_DIR / f"predictive_chart_{short}.npz"
    atomic_npz(
        decoder_path, weight=np.asarray(decoder["weight"]),
        intercept=np.asarray(decoder["intercept"]), penalty=np.asarray(decoder["penalty"]),
    )
    atomic_npz(
        carrier_path, mean=np.asarray(carrier["mean"]), scale=np.asarray(carrier["scale"]),
        basis=np.asarray(carrier["basis"]), random_basis=np.asarray(carrier["random_basis"]),
        singular_values=np.asarray(carrier["singular_values"]),
        width=np.asarray(width, dtype=np.int64), rank=np.asarray(carrier["rank"], dtype=np.int64),
    )
    rank_result = chart["rank_result"]
    atomic_npz(
        chart_path, mean=chart["mean"], scale=chart["scale"], basis=chart["basis"],
        singular_values=chart["singular_values"], record_ids=chart["record_ids"],
        groups=chart["groups"], modes=chart["modes"],
    )
    write_json(RANK_DIR / f"rank_selection_{short}.json", {
        key: (np.asarray(value).tolist() if isinstance(value, np.ndarray) else value)
        for key, value in rank_result.items()
    })
    write_json(SUBSPACE_DIR / f"artifact_manifest_{short}.json", {
        "model": short, "decoder_sha256": sha256_file(decoder_path),
        "carrier_sha256": sha256_file(carrier_path), "chart_sha256": sha256_file(chart_path),
        "decoder_training_split": "construction",
        "rank_training_split": "construction",
        "carrier_training_split": "construction",
        "evaluation_rows_used": 0, "shared_physical_labels_used_by_decoder": True,
        "carrier_rank": int(carrier["rank"]), "ambient_carrier_dimension": int(256 * width),
    })


def load_frozen_artifacts(short):
    paths = {
        "decoder": SUBSPACE_DIR / f"decoder_{short}.npz",
        "carrier": SUBSPACE_DIR / f"carrier_basis_{short}.npz",
        "chart": SUBSPACE_DIR / f"predictive_chart_{short}.npz",
    }
    manifest = json.loads((SUBSPACE_DIR / f"artifact_manifest_{short}.json").read_text())
    for label, path in paths.items():
        digest_path = Path(str(path) + ".sha256")
        if not path.is_file() or not digest_path.is_file():
            raise RuntimeError(f"missing frozen {short} {label} artifact or digest")
        observed = sha256_file(path)
        if digest_path.read_text().strip() != observed:
            raise RuntimeError(f"frozen {short} {label} digest sidecar mismatch")
        if manifest[f"{label}_sha256"] != observed:
            raise RuntimeError(f"frozen {short} {label} manifest mismatch")
    with np.load(paths["decoder"], allow_pickle=False) as payload:
        decoder = {key: payload[key] for key in payload.files}
    with np.load(paths["carrier"], allow_pickle=False) as payload:
        carrier = {key: payload[key] for key in payload.files}
    with np.load(paths["chart"], allow_pickle=False) as payload:
        chart = {key: payload[key] for key in payload.files}
    chart["rank_result"] = json.loads((RANK_DIR / f"rank_selection_{short}.json").read_text())
    return decoder, carrier, chart


def projected_action_response_rows(traces, names, carrier, width):
    rows, metadata = [], []
    for name in names:
        zero = ZERO_WORD_NAMES[len(name)]
        if zero not in traces:
            continue
        for step in range(len(name)):
            rows.append((traces[name][step] - traces[zero][step]).reshape(-1))
            metadata.append({"word": name, "zero": zero, "step": int(step)})
    if not rows:
        return np.empty((0, int(carrier["rank"]))), []
    return project_carrier_deltas(np.asarray(rows), carrier, width), metadata


def generate_model_record(bundle, record, split, decoder, carrier):
    path = model_path(bundle["short"], record)
    names = names_for_split(split)
    identity = (
        f"{PROTOCOL_ID}:{RUN_SIGNATURE}:{bundle['short']}:{record['record_id']}:"
        f"{REPO_COMMIT}:{EXPECTED_PRETRAINED_ASSET_SHA256[bundle['name'] + '.pth.tar']}"
    )
    required = {
        "identity", "word_names", "word_lengths", "feature_tensor", "proprio_width",
        "grounded_predictions", "pair_coordinates", "pair_metadata",
        "response_coordinates", "response_metadata",
    }
    if validate_npz_shard(path, required, identity):
        PROVENANCE_COUNTS["validated_cache_hits"] += 1
        return path
    outputs, traces = grouped_model_words(bundle, record, names)
    tensor, proprio_width = feature_tensor_from_outputs(outputs, names)
    grounded = tensor.astype(np.float64) @ decoder["weight"] + decoder["intercept"]
    pairs = CALIBRATION_INTERCHANGE_PAIRS if split != "evaluation" else EVALUATION_INTERCHANGE_PAIRS
    pair_deltas, pair_meta = carrier_delta_rows(traces, pairs)
    pair_coordinates = project_carrier_deltas(
        pair_deltas, carrier, bundle["carrier_width"]
    )
    response_names = CORE_WORD_NAMES if split != "evaluation" else EVALUATION_WORD_NAMES
    response_coordinates, response_meta = projected_action_response_rows(
        traces, response_names, carrier, bundle["carrier_width"]
    )
    atomic_npz(
        path,
        identity=np.asarray(identity),
        word_names=np.asarray(names),
        word_lengths=np.asarray([WORD_BY_NAME[name]["length"] for name in names], dtype=np.int64),
        feature_tensor=tensor,
        proprio_width=np.asarray(proprio_width, dtype=np.int64),
        grounded_predictions=grounded.astype(np.float32),
        pair_coordinates=pair_coordinates.astype(np.float32),
        pair_metadata=np.asarray([json.dumps(row, sort_keys=True) for row in pair_meta]),
        response_coordinates=response_coordinates.astype(np.float32),
        response_metadata=np.asarray([json.dumps(row, sort_keys=True) for row in response_meta]),
    )
    PROVENANCE_COUNTS["model_record_forwards"][bundle["short"]] += 1
    return path


def generate_transition_record(bundle, record, split, decoder):
    """Materialize model predictive signatures at every unique word prefix."""
    if split not in {"model_selection", "calibration", "evaluation"}:
        raise ValueError("transition shards are model-selection/calibration/evaluation only")
    path = transition_path(bundle["short"], record, split)
    names, prefixes = transition_prefixes(split)
    identity = (
        f"{PROTOCOL_ID}:{RUN_SIGNATURE}:{bundle['short']}:{record['record_id']}:"
        f"{split}:transition-prefix-v1"
    )
    required = {
        "identity", "word_names", "word_lengths", "prefix_names", "signatures",
        "signature_mask", "action_features", "source_modes", "physical_paths",
        "physical_path_mask",
    }
    if validate_npz_shard(path, required, identity):
        PROVENANCE_COUNTS["validated_cache_hits"] += 1
        return path

    base_model_path = model_path(bundle["short"], record)
    with np.load(base_model_path, allow_pickle=False) as base:
        base_names = [str(value) for value in base["word_names"]]
        base_signature = core_signature_from_grounded(
            base["grounded_predictions"], base_names
        )
    prefix_signatures = {"": base_signature}
    for prefix in prefixes:
        successor, _ = record_after_word(record, prefix)
        outputs, _ = grouped_model_words(bundle, successor, CORE_WORD_NAMES)
        tensor, _ = feature_tensor_from_outputs(outputs, CORE_WORD_NAMES)
        grounded = tensor.astype(np.float64) @ decoder["weight"] + decoder["intercept"]
        prefix_signatures[prefix] = core_signature_from_grounded(
            grounded, CORE_WORD_NAMES
        )

    signature_dim = len(base_signature)
    signatures = np.zeros(
        (len(names), MAX_WORD_LENGTH + 1, signature_dim), dtype=np.float32
    )
    signature_mask = np.zeros((len(names), MAX_WORD_LENGTH + 1), dtype=bool)
    action_features = np.zeros((len(names), MAX_WORD_LENGTH, 3), dtype=np.float64)
    source_modes = np.full((len(names), MAX_WORD_LENGTH), "", dtype="<U16")
    physical_paths = np.zeros((len(names), MAX_WORD_LENGTH, 11), dtype=np.float64)
    physical_path_mask = np.zeros((len(names), MAX_WORD_LENGTH), dtype=bool)
    for index, name in enumerate(names):
        rollout = rollout_word(record, WORD_BY_NAME[name], retain_visual=False)
        length = len(name)
        signatures[index, 0] = prefix_signatures[""]
        signature_mask[index, : length + 1] = True
        modes = source_mode_sequence(record, name, rollout)
        for step, symbol in enumerate(name):
            prefix = name[: step + 1]
            signatures[index, step + 1] = prefix_signatures[prefix]
            action_features[index, step] = action_feature(symbol)
            source_modes[index, step] = modes[step]
            physical_paths[index, step] = grounded_observables(
                rollout["path_states"][step]
            )
            physical_path_mask[index, step] = True
    atomic_npz(
        path,
        identity=np.asarray(identity),
        word_names=np.asarray(names),
        word_lengths=np.asarray([len(name) for name in names], dtype=np.int64),
        prefix_names=np.asarray(prefixes),
        signatures=signatures,
        signature_mask=signature_mask,
        action_features=action_features,
        source_modes=source_modes,
        physical_paths=physical_paths,
        physical_path_mask=physical_path_mask,
    )
    PROVENANCE_COUNTS["model_record_forwards"][bundle["short"]] += len(prefixes)
    return path


MODEL_ARTIFACTS = {}
if not PIPELINE_FAILED:
    try:
        verify_executed_notebook_through(
            "# Fit grounded readouts, predictive charts, and carrier bases on construction trajectories only."
        )
        verify_pretrained_assets()
        for model_name in MODEL_NAMES:
            bundle = load_world_model(model_name)
            short = bundle["short"]
            try:
                if all((SUBSPACE_DIR / name).is_file() for name in [
                    f"decoder_{short}.npz", f"carrier_basis_{short}.npz",
                    f"predictive_chart_{short}.npz",
                ]):
                    decoder, carrier, chart = load_frozen_artifacts(short)
                    PROVENANCE_COUNTS["validated_cache_hits"] += 3
                else:
                    decoder, carrier, chart = construction_model_data(bundle)
                    save_frozen_model_artifacts(
                        short, decoder, carrier, chart, bundle["carrier_width"]
                    )
                MODEL_ARTIFACTS[short] = {
                    "decoder": decoder, "carrier": carrier, "chart": chart,
                    "carrier_width": int(bundle["carrier_width"]),
                }
                for split in ["construction", "model_selection", "calibration", "evaluation"]:
                    for index, record in enumerate(SELECTED_RECORDS[split]):
                        generate_model_record(bundle, record, split, decoder, carrier)
                        if split in {"model_selection", "calibration", "evaluation"}:
                            generate_transition_record(bundle, record, split, decoder)
                        write_json(OUT / f"model_{short}_{split}_progress.json", {
                            "completed": index + 1,
                            "total": len(SELECTED_RECORDS[split]),
                            "last_record_id": int(record["record_id"]),
                        })
                memory_report(f"stage33_{short}_model_shards_complete")
            finally:
                unload_world_model(bundle)
        atomic_checkpoint("construction_models_complete", {
            "models": list(MODEL_ARTIFACTS),
            "artifact_manifests": {
                short: sha256_file(SUBSPACE_DIR / f"artifact_manifest_{short}.json")
                for short in MODEL_ARTIFACTS
            },
        })
    except Exception:
        record_failure("stage33_construction_decoder_rank_carrier_or_model_shards")
'''


model_selection_and_calibration = r'''# Lock rank, operator class, and regularization on model-selection trajectories; fit maps on calibration only.


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def model_signature(short, record):
    with np.load(model_path(short, record), allow_pickle=False) as payload:
        names = [str(value) for value in payload["word_names"]]
        return core_signature_from_grounded(payload["grounded_predictions"], names)


def split_coordinates(short, split, rank):
    chart = MODEL_ARTIFACTS[short]["chart"]
    rows = []
    for record in SELECTED_RECORDS[split]:
        rows.append(chart_coordinates(model_signature(short, record), chart, rank))
    return np.asarray(rows, dtype=np.float64)


def model_selection_rank(short):
    chart = MODEL_ARTIFACTS[short]["chart"]
    signatures = np.asarray([
        model_signature(short, record) for record in SELECTED_RECORDS["model_selection"]
    ])
    standardized = (signatures - chart["mean"]) / chart["scale"]
    groups = np.asarray([
        record["trajectory_id"] for record in SELECTED_RECORDS["model_selection"]
    ], dtype=np.int64)
    return select_stable_rank(
        standardized, groups, max_rank=MAX_EFFECTIVE_RANK,
        n_bootstrap=ACTIVE_RANK_BOOTSTRAPS,
        n_permutations=ACTIVE_RANK_PERMUTATIONS,
        stability_floor=RANK_STABILITY_FLOOR,
        null_quantile=RANK_NULL_QUANTILE,
        seed=stable_seed(RANK_SEED, short, "model_selection"),
    )


def transition_rows(short, split, rank):
    chart = MODEL_ARTIFACTS[short]["chart"]
    states, actions, targets, physical_modes = [], [], [], []
    groups, records, words, steps = [], [], [], []
    paths = []
    for record in SELECTED_RECORDS[split]:
        with np.load(transition_path(short, record, split), allow_pickle=False) as payload:
            names = [str(value) for value in payload["word_names"]]
            lengths = payload["word_lengths"].astype(int)
            for word_index, name in enumerate(names):
                length = int(lengths[word_index])
                coordinates = np.asarray([
                    chart_coordinates(payload["signatures"][word_index, step], chart, rank)
                    for step in range(length + 1)
                ])
                paths.append({
                    "record": record, "word": name, "states": coordinates,
                    "actions": payload["action_features"][word_index, :length].astype(np.float64),
                    "modes": payload["source_modes"][word_index, :length].astype(str),
                    "physical_path": payload["physical_paths"][word_index, :length].astype(np.float64),
                })
                for step in range(length):
                    states.append(coordinates[step])
                    actions.append(payload["action_features"][word_index, step])
                    targets.append(coordinates[step + 1])
                    physical_modes.append(str(payload["source_modes"][word_index, step]))
                    groups.append(int(record["trajectory_id"]))
                    records.append(int(record["record_id"]))
                    words.append(name)
                    steps.append(step)
    return {
        "states": np.asarray(states), "actions": np.asarray(actions),
        "targets": np.asarray(targets), "physical_modes": np.asarray(physical_modes),
        "groups": np.asarray(groups, dtype=np.int64),
        "record_ids": np.asarray(records, dtype=np.int64),
        "words": np.asarray(words), "steps": np.asarray(steps, dtype=np.int64),
        "paths": paths,
    }


def deterministic_kmeans(values, clusters, seed, iterations=100):
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    centers = [x[int(rng.integers(0, len(x)))]]
    while len(centers) < int(clusters):
        distance = np.min(
            np.stack([np.sum((x - center) ** 2, axis=1) for center in centers]), axis=0
        )
        if np.sum(distance) <= 1e-12:
            centers.append(x[len(centers) % len(x)])
        else:
            centers.append(x[int(rng.choice(len(x), p=distance / np.sum(distance)))])
    centers = np.asarray(centers)
    for _ in range(int(iterations)):
        labels = np.argmin(np.sum((x[:, None] - centers[None]) ** 2, axis=2), axis=1)
        updated = np.stack([
            np.mean(x[labels == index], axis=0) if np.any(labels == index) else centers[index]
            for index in range(int(clusters))
        ])
        if np.allclose(updated, centers, atol=1e-10, rtol=0):
            break
        centers = updated
    labels = np.argmin(np.sum((x[:, None] - centers[None]) ** 2, axis=2), axis=1)
    return centers, labels


def assign_centers(values, centers):
    x = np.asarray(values, dtype=np.float64)
    return np.argmin(
        np.sum((x[:, None] - np.asarray(centers)[None]) ** 2, axis=2), axis=1
    )


def grouped_operator_oof(dataset, ridge, mode_kind, centers, seed):
    groups = np.asarray(list(dict.fromkeys(dataset["groups"].tolist())))
    rng = np.random.default_rng(int(seed))
    rng.shuffle(groups)
    folds = np.array_split(groups, min(4, len(groups)))
    predictions = np.zeros_like(dataset["targets"])
    for held_out in folds:
        test = np.isin(dataset["groups"], held_out)
        train = ~test
        if mode_kind == "global":
            train_modes = test_modes = None
        elif mode_kind == "physical":
            train_modes = dataset["physical_modes"][train]
            test_modes = dataset["physical_modes"][test]
        elif mode_kind == "label_free":
            train_modes = assign_centers(dataset["states"][train], centers)
            test_modes = assign_centers(dataset["states"][test], centers)
        else:
            raise ValueError(mode_kind)
        operator = fit_affine_bilinear_operator(
            dataset["states"][train], dataset["actions"][train],
            dataset["targets"][train], ridge=float(ridge), modes=train_modes,
        )
        if isinstance(operator, dict) and "operators" in operator:
            for mode in set(np.asarray(test_modes).tolist()):
                operator["operators"].setdefault(mode, operator["global_operator"])
        predictions[test] = predict_affine_bilinear(
            operator, dataset["states"][test], dataset["actions"][test], modes=test_modes
        )
    return float(np.mean((predictions - dataset["targets"]) ** 2))


def select_operator_ridges(short, dataset, centers):
    result = {}
    for mode_kind in ["global", "physical", "label_free"]:
        losses = []
        for ridge in OPERATOR_RIDGES:
            losses.append(grouped_operator_oof(
                dataset, ridge, mode_kind, centers,
                stable_seed(CALIBRATION_SEED, short, mode_kind, ridge),
            ))
        index = int(np.argmin(losses))
        result[mode_kind] = {
            "ridge": float(OPERATOR_RIDGES[index]), "oof_mse": float(losses[index]),
            "all_oof_mse": [float(value) for value in losses],
        }
    return result


def random_features(values, seed, width, parameters=None):
    x = np.asarray(values, dtype=np.float64)
    if parameters is None:
        rng = np.random.default_rng(int(seed))
        weight = rng.normal(size=(x.shape[1], int(width))) / np.sqrt(x.shape[1])
        bias = rng.uniform(-np.pi, np.pi, size=int(width))
    else:
        weight, bias = parameters
    return np.sqrt(2.0 / int(width)) * np.cos(x @ weight + bias), (weight, bias)


def fit_random_feature_predictor(dataset, groups, seed, width):
    inputs = np.column_stack([dataset["states"], dataset["actions"]])
    features, parameters = random_features(inputs, seed, width)
    fit = fit_grouped_ridge(
        features, dataset["targets"], groups,
        penalties=OPERATOR_RIDGES, folds=min(4, len(set(groups))), seed=seed,
    )
    return {"parameters": parameters, "fit": fit, "width": int(width)}


def apply_random_feature_predictor(model, states, actions):
    inputs = np.column_stack([
        np.atleast_2d(np.asarray(states, dtype=np.float64)),
        np.atleast_2d(np.asarray(actions, dtype=np.float64)),
    ])
    features, _ = random_features(
        inputs, 0, model["width"], parameters=model["parameters"]
    )
    return features @ model["fit"]["weight"] + model["fit"]["intercept"]


def fit_split_half_chart(short, parity, rank):
    trajectory_ids = sorted({
        int(record["trajectory_id"]) for record in SELECTED_RECORDS["construction"]
    })
    selected_ids = set(trajectory_ids[int(parity) :: 2])
    records = [
        record for record in SELECTED_RECORDS["construction"]
        if int(record["trajectory_id"]) in selected_ids
    ]
    signatures = np.asarray([model_signature(short, record) for record in records])
    mean = np.mean(signatures, axis=0)
    scale = np.maximum(np.std(signatures, axis=0, ddof=1), 1e-6)
    _, _, right = np.linalg.svd((signatures - mean) / scale, full_matrices=False)
    if len(right) < int(rank):
        raise RuntimeError("same-model split-half chart cannot support common rank")
    return {"mean": mean, "scale": scale, "basis": right[: int(rank)].T}


def coordinates_in_chart(signatures, chart):
    return ((np.asarray(signatures) - chart["mean"]) / chart["scale"]) @ chart["basis"]


def permuted_target_rows(target, records, seed):
    rng = np.random.default_rng(int(seed))
    trajectories = sorted({record["trajectory_id"] for record in records})
    permuted = np.asarray(trajectories)[rng.permutation(len(trajectories))]
    mapping = dict(zip(trajectories, permuted.tolist()))
    lookup = {
        (record["trajectory_id"], record["mode"]): index
        for index, record in enumerate(records)
    }
    return np.asarray([
        target[lookup[(mapping[record["trajectory_id"]], record["mode"])]]
        for record in records
    ])


def fit_similarity_with_fallback(source, target, max_condition, min_singular_value):
    """Keep a failed identifiability gate evaluable without calling it a pass."""
    try:
        result = fit_whitened_similarity(
            source, target, max_condition=max_condition,
            min_singular_value=min_singular_value,
        )
        result["strict_fit_passed"] = True
        return result
    except (ValueError, np.linalg.LinAlgError) as error:
        first = np.asarray(source, dtype=np.float64)
        second = np.asarray(target, dtype=np.float64)
        first_mean, second_mean = np.mean(first, axis=0), np.mean(second, axis=0)
        row_map = np.linalg.lstsq(first - first_mean, second - second_mean, rcond=1e-8)[0]
        matrix = row_map.T
        singular = np.linalg.svd(matrix, compute_uv=False)
        condition = float(np.linalg.cond(matrix))
        if not np.isfinite(condition):
            condition = 1e12
        offset = second_mean - matrix @ first_mean
        prediction = first @ matrix.T + offset
        return {
            "matrix": matrix, "inverse": np.linalg.pinv(matrix), "offset": offset,
            "source_mean": first_mean, "target_mean": second_mean,
            "condition_number": condition,
            "minimum_singular_value": float(singular[-1]),
            "calibration_rmse": float(np.sqrt(np.mean((prediction - second) ** 2))),
            "calibration_relative_rmse": float(
                np.sqrt(np.mean((prediction - second) ** 2))
                / max(np.sqrt(np.mean((second - second_mean) ** 2)), 1e-12)
            ),
            "strict_fit_passed": False, "strict_fit_error": str(error),
        }


def fit_carrier_predictive_bridges(short, calibration_data):
    """Fit only within-model response bridges; the sole cross-model map remains S."""
    path_lookup = {
        (int(row["record"]["record_id"]), str(row["word"])): row
        for row in calibration_data["paths"]
    }
    carrier_rows, predictive_effects, groups = [], [], []
    for record in SELECTED_RECORDS["calibration"]:
        with np.load(model_path(short, record), allow_pickle=False) as payload:
            coordinates = payload["pair_coordinates"].astype(np.float64)
            metadata = [json.loads(str(value)) for value in payload["pair_metadata"]]
        for coordinate, meta in zip(coordinates, metadata):
            base = path_lookup[(int(record["record_id"]), str(meta["base"]))]
            hybrid = path_lookup[(int(record["record_id"]), str(meta["hybrid"]))]
            carrier_rows.append(coordinate)
            predictive_effects.append(hybrid["states"][-1] - base["states"][-1])
            groups.append(int(record["trajectory_id"]))
    carrier_rows = np.asarray(carrier_rows, dtype=np.float64)
    predictive_effects = np.asarray(predictive_effects, dtype=np.float64)
    groups = np.asarray(groups, dtype=np.int64)
    # Antisymmetric augmentation enforces an effect map through the origin
    # while retaining grouped ridge selection and model-specific gauges.
    carrier_fit_rows = np.concatenate([carrier_rows, -carrier_rows])
    predictive_fit_rows = np.concatenate([predictive_effects, -predictive_effects])
    fit_groups = np.concatenate([groups, groups])
    forward = fit_grouped_ridge(
        carrier_fit_rows, predictive_fit_rows, fit_groups, penalties=DECODER_RIDGES,
        folds=min(4, len(set(groups))), seed=stable_seed(MAP_SEED, short, "bridge_forward"),
    )
    inverse = fit_grouped_ridge(
        predictive_fit_rows, carrier_fit_rows, fit_groups, penalties=DECODER_RIDGES,
        folds=min(4, len(set(groups))), seed=stable_seed(MAP_SEED, short, "bridge_inverse"),
    )
    return {
        "forward": forward, "inverse": inverse,
        "n_calibration_effects": int(len(groups)),
        "fit_split": "calibration", "cross_model_parameters": 0,
    }


RANK_SELECTION = {}
MODEL_SELECTION_DATA = {}
OPERATOR_SELECTION = {}
MODE_CENTERS = {}
MODE_DISCOVERY_COUNTS = {}
OPERATORS = {}
STATE_MAPS = {}
WITHIN_MODEL_BRIDGES = {}
SPLIT_HALF_CONTROLS = {}
if not PIPELINE_FAILED:
    try:
        verify_executed_notebook_through(
            "# Lock rank, operator class, and regularization on model-selection trajectories; fit maps on calibration only."
        )
        for short in ["jepa", "dino"]:
            RANK_SELECTION[short] = model_selection_rank(short)
        selected_ranks = {short: int(result["rank"]) for short, result in RANK_SELECTION.items()}
        RANK_STABILITY_LOCK_PASSED = bool(
            min(selected_ranks.values()) >= MIN_COMMON_RANK
            and abs(selected_ranks["jepa"] - selected_ranks["dino"])
                <= MAX_RANK_DIFFERENCE
        )
        # Continue a failed rank gate at the preregistered minimum diagnostic
        # rank so the run yields a scientific FAIL rather than a pipeline error.
        COMMON_RANK = max(MIN_COMMON_RANK, min(selected_ranks.values()))
        COMMON_RANK = min(COMMON_RANK, MAX_EFFECTIVE_RANK)
        write_json(RANK_DIR / "common_rank_lock.json", {
            "selected_ranks": selected_ranks, "common_rank": int(COMMON_RANK),
            "rank_stability_gate_passed": RANK_STABILITY_LOCK_PASSED,
            "selection_split": "model_selection", "evaluation_rows_used": 0,
            "results": jsonable(RANK_SELECTION),
        })
        write_digest_sidecar(RANK_DIR / "common_rank_lock.json")

        for short in ["jepa", "dino"]:
            dataset = transition_rows(short, "model_selection", COMMON_RANK)
            centers, _ = deterministic_kmeans(
                dataset["states"], len(MODE_LABELS),
                stable_seed(CALIBRATION_SEED, short, "label_free_modes"),
            )
            MODE_CENTERS[short] = centers
            discovered_labels = assign_centers(dataset["states"], centers)
            MODE_DISCOVERY_COUNTS[short] = {
                str(index): int(np.sum(discovered_labels == index))
                for index in range(len(MODE_LABELS))
            }
            MODEL_SELECTION_DATA[short] = dataset
            OPERATOR_SELECTION[short] = select_operator_ridges(short, dataset, centers)
        write_json(REALIZATION_DIR / "operator_model_selection_lock.json", {
            "split": "model_selection", "evaluation_rows_used": 0,
            "common_rank": int(COMMON_RANK), "choices": jsonable(OPERATOR_SELECTION),
            "mode_centers": jsonable(MODE_CENTERS),
            "mode_discovery_counts": MODE_DISCOVERY_COUNTS,
            "capacity_matched_random_features": int(NONLINEAR_RANDOM_FEATURES),
        })
        write_digest_sidecar(REALIZATION_DIR / "operator_model_selection_lock.json")
        print(json.dumps({
            "model_selection_ranks": selected_ranks,
            "common_rank": int(COMMON_RANK),
            "rank_gate_passed": RANK_STABILITY_LOCK_PASSED,
            "operator_choices": OPERATOR_SELECTION,
            "label_free_mode_counts": MODE_DISCOVERY_COUNTS,
        }, indent=2))

        CALIBRATION_DATA = {
            short: transition_rows(short, "calibration", COMMON_RANK)
            for short in ["jepa", "dino"]
        }
        for short in ["jepa", "dino"]:
            data = CALIBRATION_DATA[short]
            label_free_modes = assign_centers(data["states"], MODE_CENTERS[short])
            label_free_operator = fit_affine_bilinear_operator(
                data["states"], data["actions"], data["targets"],
                ridge=OPERATOR_SELECTION[short]["label_free"]["ridge"],
                modes=label_free_modes,
            )
            for mode_index in range(len(MODE_LABELS)):
                label_free_operator["operators"].setdefault(
                    mode_index, label_free_operator["global_operator"]
                )
            OPERATORS[short] = {
                "global": fit_affine_bilinear_operator(
                    data["states"], data["actions"], data["targets"],
                    ridge=OPERATOR_SELECTION[short]["global"]["ridge"],
                ),
                "physical": fit_affine_bilinear_operator(
                    data["states"], data["actions"], data["targets"],
                    ridge=OPERATOR_SELECTION[short]["physical"]["ridge"],
                    modes=data["physical_modes"],
                ),
                "label_free": label_free_operator,
                "nonlinear": fit_random_feature_predictor(
                    data, data["groups"],
                    stable_seed(CALIBRATION_SEED, short, "nonlinear"),
                    NONLINEAR_RANDOM_FEATURES,
                ),
            }
            operator_path = REALIZATION_DIR / f"frozen_operators_{short}.json"
            write_json(operator_path, jsonable(OPERATORS[short]))
            write_digest_sidecar(operator_path)

        calibration_coordinates = {
            short: split_coordinates(short, "calibration", COMMON_RANK)
            for short in ["jepa", "dino"]
        }
        STATE_MAPS["primary"] = fit_similarity_with_fallback(
            calibration_coordinates["jepa"], calibration_coordinates["dino"],
            max_condition=CARRIER_MAP_MAX_CONDITION,
            min_singular_value=MIN_MAP_SINGULAR_VALUE,
        )
        permuted_target = permuted_target_rows(
            calibration_coordinates["dino"], SELECTED_RECORDS["calibration"],
            stable_seed(CONTROL_SEED, "state_permutation"),
        )
        STATE_MAPS["state_permutation"] = fit_similarity_with_fallback(
            calibration_coordinates["jepa"], permuted_target,
            max_condition=1e6, min_singular_value=1e-6,
        )
        rng = np.random.default_rng(stable_seed(CONTROL_SEED, "orthogonal_map"))
        orthogonal, _ = np.linalg.qr(rng.normal(size=(COMMON_RANK, COMMON_RANK)))
        STATE_MAPS["random_orthogonal"] = {
            "matrix": orthogonal,
            "inverse": orthogonal.T,
            "offset": np.mean(calibration_coordinates["dino"], axis=0)
                - orthogonal @ np.mean(calibration_coordinates["jepa"], axis=0),
            "condition_number": 1.0, "minimum_singular_value": 1.0,
            "calibration_relative_rmse": -1.0,
        }
        for symbol, calibration_word in [("negative", "L"), ("center", "S"), ("positive", "R")]:
            j_rows, d_rows = [], []
            for short, destination in [("jepa", j_rows), ("dino", d_rows)]:
                data = CALIBRATION_DATA[short]
                selected = (data["words"] == calibration_word) & (data["steps"] == 0)
                destination.extend(data["targets"][selected])
            STATE_MAPS[f"action_{symbol}"] = fit_similarity_with_fallback(
                np.asarray(j_rows), np.asarray(d_rows), max_condition=1e6,
                min_singular_value=1e-6,
            )

        for short in ["jepa", "dino"]:
            WITHIN_MODEL_BRIDGES[short] = fit_carrier_predictive_bridges(
                short, CALIBRATION_DATA[short]
            )

        for short in ["jepa", "dino"]:
            chart_zero = fit_split_half_chart(short, 0, COMMON_RANK)
            chart_one = fit_split_half_chart(short, 1, COMMON_RANK)
            signatures = np.asarray([
                model_signature(short, record) for record in SELECTED_RECORDS["calibration"]
            ])
            SPLIT_HALF_CONTROLS[short] = {
                "chart_zero": chart_zero, "chart_one": chart_one,
                "map": fit_similarity_with_fallback(
                    coordinates_in_chart(signatures, chart_zero),
                    coordinates_in_chart(signatures, chart_one),
                    max_condition=1e6, min_singular_value=1e-6,
                ),
            }

        atomic_npz(
            MAP_DIR / "frozen_state_map.npz",
            **{f"primary_{key}": np.asarray(value) for key, value in STATE_MAPS["primary"].items() if isinstance(value, (np.ndarray, int, float))},
        )
        write_json(MAP_DIR / "calibration_freeze.json", {
            "fit_split": "calibration", "model_selection_split": "model_selection",
            "evaluation_opened": False, "common_rank": int(COMMON_RANK),
            "state_maps": jsonable(STATE_MAPS),
            "within_model_bridges": jsonable(WITHIN_MODEL_BRIDGES),
            "cross_model_map_count": 1,
            "transport_path": "carrier_J_to_delta_q_J_to_S_delta_q_J_to_delta_q_D_to_carrier_D",
            "operator_selection": jsonable(OPERATOR_SELECTION),
            "calibration_trajectory_ids": sorted({
                int(record["trajectory_id"]) for record in SELECTED_RECORDS["calibration"]
            }),
        })
        write_digest_sidecar(MAP_DIR / "calibration_freeze.json")
        atomic_checkpoint("model_selection_and_calibration_complete", {
            "common_rank": int(COMMON_RANK),
            "rank_lock_sha256": sha256_file(RANK_DIR / "common_rank_lock.json"),
            "map_lock_sha256": sha256_file(MAP_DIR / "calibration_freeze.json"),
            "operator_sha256": {
                short: sha256_file(REALIZATION_DIR / f"frozen_operators_{short}.json")
                for short in ["jepa", "dino"]
            },
        })
        print(json.dumps({
            "calibration_records": len(SELECTED_RECORDS["calibration"]),
            "sole_cross_model_map_condition": STATE_MAPS["primary"]["condition_number"],
            "sole_cross_model_map_strict_fit": STATE_MAPS["primary"].get("strict_fit_passed", False),
            "within_model_bridge_effects": {
                short: WITHIN_MODEL_BRIDGES[short]["n_calibration_effects"]
                for short in ["jepa", "dino"]
            },
            "evaluation_opened": False,
        }, indent=2))
        memory_report("stage33_model_selection_and_calibration_complete")
    except Exception:
        record_failure("stage33_model_selection_rank_operator_or_calibration_map")
'''


locked_evaluation = r'''# Open the locked evaluation once and score multi-step realization and conjugacy controls.


def apply_map(values, mapping, affine=True):
    rows = np.asarray(values, dtype=np.float64)
    result = rows @ np.asarray(mapping["matrix"], dtype=np.float64).T
    if affine:
        result = result + np.asarray(mapping.get("offset", 0.0), dtype=np.float64)
    return result


def relative_squared_error(prediction, target, reference_mean=None):
    predicted = np.asarray(prediction, dtype=np.float64)
    observed = np.asarray(target, dtype=np.float64)
    center = np.mean(observed, axis=0) if reference_mean is None else np.asarray(reference_mean)
    numerator = np.mean((predicted - observed) ** 2, axis=-1)
    denominator = np.mean((observed - center) ** 2, axis=-1)
    floor = max(float(np.median(denominator)) * 1e-6, 1e-12)
    return numerator / np.maximum(denominator, floor)


def predict_random_feature_word(model, initial, actions):
    state = np.asarray(initial, dtype=np.float64)[None]
    for action in np.asarray(actions, dtype=np.float64):
        state = apply_random_feature_predictor(model, state, action[None])
    return state[0]


def path_predictions(short, path):
    initial = path["states"][0]
    actions = path["actions"]
    modes = path["modes"]
    label_free_modes = assign_centers(path["states"][:-1], MODE_CENTERS[short])
    shuffled_actions = actions[::-1].copy()
    permutation = {label: MODE_LABELS[(index + 1) % len(MODE_LABELS)] for index, label in enumerate(MODE_LABELS)}
    permuted_modes = np.asarray([permutation[str(value)] for value in modes])
    return {
        "global": compose_affine_bilinear(OPERATORS[short]["global"], initial, actions),
        "hybrid_physical": compose_affine_bilinear(
            OPERATORS[short]["physical"], initial, actions, modes
        ),
        "hybrid_label_free": compose_affine_bilinear(
            OPERATORS[short]["label_free"], initial, actions, label_free_modes
        ),
        "nonlinear": predict_random_feature_word(
            OPERATORS[short]["nonlinear"], initial, actions
        ),
        "action_word_shuffled": compose_affine_bilinear(
            OPERATORS[short]["global"], initial, shuffled_actions
        ),
        "mode_label_permuted": compose_affine_bilinear(
            OPERATORS[short]["physical"], initial, actions, permuted_modes
        ),
    }


def word_family(name):
    if len(name) == 1:
        return "one_step"
    if name in {"ab", "ba", "ABAB", "BABA"}:
        return "order_and_signed_area"
    if name in {"AAB", "BAA", "ABB", "BBA"}:
        return "fixed_multiset_order"
    return f"length_{len(name)}"


def decoder_evaluation(short):
    truth_rows_all, predicted_rows_all = [], []
    for record in SELECTED_RECORDS["evaluation"]:
        with np.load(model_path(short, record), allow_pickle=False) as model_payload:
            names = [str(value) for value in model_payload["word_names"]]
            predicted = model_payload["grounded_predictions"]
        with np.load(truth_path(record), allow_pickle=False) as truth:
            truth_lookup = {str(value): index for index, value in enumerate(truth["word_names"])}
            for index, name in enumerate(names):
                length = int(WORD_BY_NAME[name]["length"])
                predicted_rows_all.extend(predicted[index, :length])
                truth_rows_all.extend(truth["path_observables"][truth_lookup[name], :length])
    target = np.asarray(truth_rows_all)
    prediction = np.asarray(predicted_rows_all)
    total = np.sum((target - np.mean(target, axis=0)) ** 2, axis=0)
    residual = np.sum((target - prediction) ** 2, axis=0)
    r2 = 1.0 - residual / np.maximum(total, 1e-12)
    return {
        "r2_by_observable": dict(zip(GROUNDED_OBSERVABLES, r2.tolist())),
        "median_r2": float(np.median(r2)), "rows": int(len(target)),
        "rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
    }


def same_model_split_half_evaluation(short):
    control = SPLIT_HALF_CONTROLS[short]
    signatures = np.asarray([
        model_signature(short, record) for record in SELECTED_RECORDS["evaluation"]
    ])
    left = coordinates_in_chart(signatures, control["chart_zero"])
    right = coordinates_in_chart(signatures, control["chart_one"])
    mapped = apply_map(left, control["map"])
    rmse = np.sqrt(np.mean((mapped - right) ** 2))
    scale = max(float(np.sqrt(np.mean((right - np.mean(right, axis=0)) ** 2))), 1e-12)
    return {"relative_rmse": float(rmse / scale), "rows": int(len(right))}


def empirical_simulator_noise_floor():
    differences = []
    probe_names = ["a", "ab", "AAB", "ABAB"]
    for record in SELECTED_RECORDS["evaluation"]:
        for name in probe_names:
            first = rollout_word(record, WORD_BY_NAME[name], retain_visual=False)
            second = rollout_word(record, WORD_BY_NAME[name], retain_visual=False)
            first_path = np.asarray([
                grounded_observables(state) for state in first["path_states"]
            ])
            second_path = np.asarray([
                grounded_observables(state) for state in second["path_states"]
            ])
            differences.append(first_path - second_path)
    flat = np.concatenate([value.reshape(-1) for value in differences])
    return {
        "repeated_branches": int(2 * len(differences)),
        "probe_words": probe_names,
        "max_absolute_difference": float(np.max(np.abs(flat))),
        "rmse": float(np.sqrt(np.mean(flat**2))),
    }


def one_sided_bootstrap_pvalue(values, groups, positive=True):
    array = np.asarray(values, dtype=np.float64)
    ci = clustered_bootstrap_interval(
        array, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
        seed=stable_seed(BOOTSTRAP_SEED, "pvalue", len(array), float(np.mean(array))),
        alpha=HOLM_ALPHA,
    )
    # A conservative normal-free sign summary used only for multiplicity
    # bookkeeping; the preregistered percentile interval remains the gate.
    units = [np.mean(array[np.asarray(groups) == group]) for group in sorted(set(groups))]
    contrary = sum(value <= 0 if positive else value >= 0 for value in units)
    pvalue = float((contrary + 1) / (len(units) + 1))
    return ci, pvalue


EVALUATION_DATA = {}
REALIZATION_ROWS = []
DECODER_METRICS = {}
SPLIT_HALF_METRICS = {}
MAP_ROWS = []
CONJUGACY_DIAGNOSTICS = {}
if not PIPELINE_FAILED:
    try:
        verify_executed_notebook_through(
            "# Open the locked evaluation once and score multi-step realization and conjugacy controls."
        )
        calibration_lock = json.loads((MAP_DIR / "calibration_freeze.json").read_text())
        validate_digest_sidecar(MAP_DIR / "calibration_freeze.json")
        if calibration_lock.get("evaluation_opened", True):
            raise RuntimeError("calibration was not frozen before evaluation")
        CALIBRATION_LOCK_SHA256 = sha256_file(MAP_DIR / "calibration_freeze.json")
        EMPIRICAL_NOISE_FLOOR = empirical_simulator_noise_floor()
        write_json(EVIDENCE_DIR / "empirical_simulator_noise_floor.json", EMPIRICAL_NOISE_FLOOR)
        for short in ["jepa", "dino"]:
            EVALUATION_DATA[short] = transition_rows(short, "evaluation", COMMON_RANK)
            DECODER_METRICS[short] = decoder_evaluation(short)
            SPLIT_HALF_METRICS[short] = same_model_split_half_evaluation(short)
            for path in EVALUATION_DATA[short]["paths"]:
                predictions = path_predictions(short, path)
                target = path["states"][-1]
                center = np.mean(EVALUATION_DATA[short]["targets"], axis=0)
                errors = {
                    condition: float(relative_squared_error(value[None], target[None], center)[0])
                    for condition, value in predictions.items()
                }
                decoder_only = float(relative_squared_error(
                    np.mean(EVALUATION_DATA[short]["targets"], axis=0)[None],
                    target[None], center,
                )[0])
                record = path["record"]
                REALIZATION_ROWS.append({
                    "model": short, "record_id": int(record["record_id"]),
                    "trajectory_id": int(record["trajectory_id"]),
                    "mode": str(record["mode"]), "word": str(path["word"]),
                    "word_length": int(len(path["word"])),
                    "word_family": word_family(str(path["word"])),
                    **{f"{key}_relative_squared_error": value for key, value in errors.items()},
                    "decoder_only_relative_squared_error": decoder_only,
                    "oracle_relative_squared_error": 0.0,
                    "empirical_deterministic_noise_floor": EMPIRICAL_NOISE_FLOOR["rmse"],
                })

        j_lookup = {
            (int(path["record"]["record_id"]), str(path["word"])): path
            for path in EVALUATION_DATA["jepa"]["paths"]
        }
        d_lookup = {
            (int(path["record"]["record_id"]), str(path["word"])): path
            for path in EVALUATION_DATA["dino"]["paths"]
        }
        for key in sorted(set(j_lookup) & set(d_lookup)):
            j_path, d_path = j_lookup[key], d_lookup[key]
            source, target = j_path["states"][-1], d_path["states"][-1]
            record, word = j_path["record"], j_path["word"]
            if word[0] in {"a", "A"}:
                action_map = STATE_MAPS["action_negative"]
            elif word[0] in {"b", "B"}:
                action_map = STATE_MAPS["action_positive"]
            else:
                action_map = STATE_MAPS["action_center"]
            mapped = {
                "primary": apply_map(source[None], STATE_MAPS["primary"])[0],
                "state_permutation": apply_map(source[None], STATE_MAPS["state_permutation"])[0],
                "random_orthogonal": apply_map(source[None], STATE_MAPS["random_orthogonal"])[0],
                "action_specific": apply_map(source[None], action_map)[0],
            }
            d_prediction = path_predictions("dino", d_path)["hybrid_physical"]
            j_prediction = path_predictions("jepa", j_path)["hybrid_physical"]
            mapped_operator_prediction = apply_map(
                j_prediction[None], STATE_MAPS["primary"]
            )[0]
            center = np.mean(EVALUATION_DATA["dino"]["targets"], axis=0)
            MAP_ROWS.append({
                "record_id": int(record["record_id"]),
                "trajectory_id": int(record["trajectory_id"]),
                "mode": str(record["mode"]), "word": str(word),
                "word_length": int(len(word)), "word_family": word_family(str(word)),
                **{
                    f"{condition}_relative_squared_error": float(
                        relative_squared_error(value[None], target[None], center)[0]
                    ) for condition, value in mapped.items()
                },
                "operator_conjugacy_relative_squared_error": float(
                    relative_squared_error(
                        mapped_operator_prediction[None], d_prediction[None], center
                    )[0]
                ),
            })

        direct = {
            "global": operator_intertwining_metrics(
                OPERATORS["jepa"]["global"], OPERATORS["dino"]["global"],
                STATE_MAPS["primary"],
            ),
            "physical_modes": {
                mode: operator_intertwining_metrics(
                    OPERATORS["jepa"]["physical"]["operators"][mode],
                    OPERATORS["dino"]["physical"]["operators"][mode],
                    STATE_MAPS["primary"],
                ) for mode in MODE_LABELS
            },
        }
        reachability = {
            short: {
                mode: reachability_observability_diagnostics(
                    operator["A"][None], operator["B"][None], np.eye(COMMON_RANK),
                    tolerance=1e-8,
                )
                for mode, operator in OPERATORS[short]["physical"]["operators"].items()
            } for short in ["jepa", "dino"]
        }
        CONJUGACY_DIAGNOSTICS = {
            "direct_operator_intertwining": direct,
            "reachability_observability": reachability,
            "same_model_split_half": SPLIT_HALF_METRICS,
        }
        write_csv(EVIDENCE_DIR / "locked_realization_rows.csv", REALIZATION_ROWS)
        write_csv(EVIDENCE_DIR / "locked_cross_model_map_rows.csv", MAP_ROWS)
        write_json(EVIDENCE_DIR / "decoder_metrics.json", DECODER_METRICS)
        write_json(EVIDENCE_DIR / "conjugacy_diagnostics.json", jsonable(CONJUGACY_DIAGNOSTICS))
        write_json(MAP_DIR / "evaluation_open_certificate.json", {
            "calibration_lock_sha256": CALIBRATION_LOCK_SHA256,
            "rank_lock_sha256": sha256_file(RANK_DIR / "common_rank_lock.json"),
            "evaluation_trajectory_ids": sorted({
                int(record["trajectory_id"]) for record in SELECTED_RECORDS["evaluation"]
            }),
            "unseen_action_words": EVALUATION_WORD_NAMES,
            "evaluation_map_refits": 0, "evaluation_rank_refits": 0,
            "evaluation_mode_refits": 0,
        })
        write_digest_sidecar(MAP_DIR / "evaluation_open_certificate.json")
        atomic_checkpoint("locked_evaluation_realization_complete", {
            "realization_rows": len(REALIZATION_ROWS), "map_rows": len(MAP_ROWS),
            "evaluation_open_sha256": sha256_file(MAP_DIR / "evaluation_open_certificate.json"),
        })
        memory_report("stage33_locked_realization_complete")
    except Exception:
        record_failure("stage33_locked_realization_or_conjugacy_evaluation")
'''


causal_transport = r'''# Transport reachable internal responses through the single frozen predictive map and test planning.


def bridge_predict(fit, values):
    rows = np.asarray(values, dtype=np.float64)
    return rows @ np.asarray(fit["weight"], dtype=np.float64) + np.asarray(fit["intercept"])


def transported_dino_coordinates(jepa_coordinates, map_name="primary"):
    source_effect = bridge_predict(
        WITHIN_MODEL_BRIDGES["jepa"]["forward"],
        np.atleast_2d(jepa_coordinates),
    )
    target_effect = apply_map(source_effect, STATE_MAPS[map_name], affine=False)
    return bridge_predict(WITHIN_MODEL_BRIDGES["dino"]["inverse"], target_effect)


def norm_match(delta, reference):
    value = np.asarray(delta, dtype=np.float32)
    target = np.asarray(reference, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    target_norm = float(np.linalg.norm(target))
    if norm <= 1e-12 or target_norm <= 1e-12:
        return value
    return value * (target_norm / norm)


def grounded_word_from_model_payload(payload, name):
    names = [str(value) for value in payload["word_names"]]
    index = names.index(str(name))
    length = int(payload["word_lengths"][index])
    return payload["grounded_predictions"][index, :length].astype(np.float64)


def grounded_word_from_truth_payload(payload, name):
    names = [str(value) for value in payload["word_names"]]
    index = names.index(str(name))
    length = int(payload["word_lengths"][index])
    return payload["path_observables"][index, :length].astype(np.float64)


def coordinate_lookup(payload, metadata_key, coordinate_key):
    metadata = [json.loads(str(value)) for value in payload[metadata_key]]
    return {
        tuple(sorted(row.items())): np.asarray(payload[coordinate_key][index], dtype=np.float64)
        for index, row in enumerate(metadata)
    }, metadata


def find_coordinate(payload, metadata_key, coordinate_key, **query):
    metadata = [json.loads(str(value)) for value in payload[metadata_key]]
    for index, row in enumerate(metadata):
        if all(row.get(key) == value for key, value in query.items()):
            return np.asarray(payload[coordinate_key][index], dtype=np.float64)
    raise KeyError(f"coordinate metadata not found: {query}")


def reconstruct_dino_intervention(coordinate, basis_kind="primary"):
    carrier = MODEL_ARTIFACTS["dino"]["carrier"]
    if basis_kind == "primary":
        basis = carrier["basis"]
    elif basis_kind == "random_matched_subspace":
        basis = carrier["random_basis"]
    else:
        raise ValueError(basis_kind)
    return reconstruct_with_basis(
        np.atleast_2d(coordinate), basis, carrier,
        MODEL_ARTIFACTS["dino"]["carrier_width"],
    )[0]


def patched_grounded_word(bundle, record, name, step_deltas, decoder):
    lookup = {(str(name), int(step)): delta for step, delta in step_deltas.items()}
    outputs, _ = grouped_model_words(bundle, record, [str(name)], lookup)
    tensor, _ = feature_tensor_from_outputs(outputs, [str(name)])
    grounded = tensor.astype(np.float64) @ decoder["weight"] + decoder["intercept"]
    return grounded[0, : int(WORD_BY_NAME[str(name)]["length"])]


def planning_cost(endpoints, goal):
    values = np.asarray(endpoints, dtype=np.float64)
    target = np.asarray(goal, dtype=np.float64)
    position = np.sum((values[:, 2:4] - target[2:4]) ** 2, axis=1)
    orientation = np.sum((values[:, 4:6] - target[4:6]) ** 2, axis=1)
    return position + 0.1 * orientation


def normalized_regret(true_costs, choice):
    values = np.asarray(true_costs, dtype=np.float64)
    denominator = max(float(np.max(values) - np.min(values)), 1e-12)
    return float((values[int(choice)] - np.min(values)) / denominator)


def causal_record_path(record):
    return CAUSAL_DIR / f"causal_{int(record['record_id'])}.npz"


def evaluate_causal_record(bundle, record):
    path = causal_record_path(record)
    identity = f"{PROTOCOL_ID}:{RUN_SIGNATURE}:causal:{record['record_id']}:v1"
    required = {"identity", "interchange_rows", "planning_rows", "forward_count"}
    if validate_npz_shard(path, required, identity):
        PROVENANCE_COUNTS["validated_cache_hits"] += 1
        with np.load(path, allow_pickle=False) as payload:
            return (
                [json.loads(str(value)) for value in payload["interchange_rows"]],
                [json.loads(str(value)) for value in payload["planning_rows"]],
                int(payload["forward_count"]),
            )

    with np.load(model_path("jepa", record), allow_pickle=False) as jepa_payload, np.load(
        model_path("dino", record), allow_pickle=False
    ) as dino_payload, np.load(truth_path(record), allow_pickle=False) as truth_payload:
        interchange_rows = []
        forward_count = 0
        zero_base, _, zero_step = EVALUATION_INTERCHANGE_PAIRS[0]
        zero_delta = np.zeros(
            (256, MODEL_ARTIFACTS["dino"]["carrier_width"]), dtype=np.float32
        )
        zero_patched = patched_grounded_word(
            bundle, record, zero_base, {int(zero_step): zero_delta},
            MODEL_ARTIFACTS["dino"]["decoder"],
        )
        zero_reference = grounded_word_from_model_payload(dino_payload, zero_base)
        zero_edit_max_abs = float(np.max(np.abs(zero_patched - zero_reference)))
        forward_count += 1
        for pair_index, (base, donor, step) in enumerate(EVALUATION_INTERCHANGE_PAIRS):
            hybrid = donor[: step + 1] + base[step + 1 :]
            source_coordinate = find_coordinate(
                jepa_payload, "pair_metadata", "pair_coordinates",
                base=base, donor=donor, hybrid=hybrid, step=int(step),
            )
            self_coordinate = find_coordinate(
                dino_payload, "pair_metadata", "pair_coordinates",
                base=base, donor=donor, hybrid=hybrid, step=int(step),
            )
            mapped_coordinates = {
                "primary": transported_dino_coordinates(source_coordinate, "primary")[0],
                "state_permutation": transported_dino_coordinates(
                    source_coordinate, "state_permutation"
                )[0],
                "random_orthogonal_map": transported_dino_coordinates(
                    source_coordinate, "random_orthogonal"
                )[0],
                "dino_self_positive": self_coordinate,
            }
            primary_delta = reconstruct_dino_intervention(mapped_coordinates["primary"])
            deltas = {
                condition: reconstruct_dino_intervention(coordinate)
                for condition, coordinate in mapped_coordinates.items()
            }
            deltas["random_matched_subspace"] = reconstruct_dino_intervention(
                mapped_coordinates["primary"], "random_matched_subspace"
            )
            for condition in ["state_permutation", "random_orthogonal_map", "random_matched_subspace"]:
                deltas[condition] = norm_match(deltas[condition], primary_delta)
            baseline = grounded_word_from_model_payload(dino_payload, base)
            natural_hybrid = grounded_word_from_model_payload(dino_payload, hybrid)
            physical_base = grounded_word_from_truth_payload(truth_payload, base)
            physical_hybrid = grounded_word_from_truth_payload(truth_payload, hybrid)
            physical_effect = physical_hybrid - physical_base
            natural_effect = natural_hybrid - baseline
            for condition, delta in deltas.items():
                patched = patched_grounded_word(
                    bundle, record, base, {int(step): delta},
                    MODEL_ARTIFACTS["dino"]["decoder"],
                )
                forward_count += 1
                observed_effect = patched - baseline
                grounded_metrics = interchange_metrics(
                    observed_effect, physical_effect,
                    minimum_effect_energy=MIN_GROUNDED_EFFECT_ENERGY,
                )
                self_metrics = interchange_metrics(
                    observed_effect, natural_effect,
                    minimum_effect_energy=MIN_GROUNDED_EFFECT_ENERGY,
                )
                baseline_error = float(np.mean((baseline - physical_hybrid) ** 2))
                patched_error = float(np.mean((patched - physical_hybrid) ** 2))
                interchange_rows.append({
                    "record_id": int(record["record_id"]),
                    "trajectory_id": int(record["trajectory_id"]),
                    "mode": str(record["mode"]), "pair_index": int(pair_index),
                    "base": base, "donor": donor, "hybrid": hybrid, "step": int(step),
                    "condition": condition,
                    "grounded_eligible": bool(grounded_metrics["eligible"]),
                    "grounded_cosine": float(grounded_metrics["cosine"]),
                    "grounded_relative_error": float(grounded_metrics["relative_error"]),
                    "grounded_error_gain": float(
                        (baseline_error - patched_error) / max(baseline_error, 1e-12)
                    ),
                    "self_cosine": float(self_metrics["cosine"])
                        if self_metrics["eligible"] else -1.0,
                    "self_relative_error": float(self_metrics["relative_error"])
                        if self_metrics["eligible"] else 1e9,
                    "intervention_norm": float(np.linalg.norm(delta)),
                    "zero_edit_max_abs": zero_edit_max_abs,
                })

        candidate_names = list(EVALUATION_WORD_NAMES)
        physical_endpoints = np.asarray([
            grounded_word_from_truth_payload(truth_payload, name)[-1]
            for name in candidate_names
        ])
        native_endpoints = np.asarray([
            grounded_word_from_model_payload(dino_payload, name)[-1]
            for name in candidate_names
        ])
        transported_endpoints = {
            condition: [] for condition in [
                "primary", "state_permutation", "random_orthogonal_map",
                "random_matched_subspace",
            ]
        }
        for name in candidate_names:
            zero = ZERO_WORD_NAMES[len(name)]
            source_steps = np.asarray([
                find_coordinate(
                    jepa_payload, "response_metadata", "response_coordinates",
                    word=name, zero=zero, step=int(step),
                ) for step in range(len(name))
            ])
            coordinate_sets = {
                "primary": transported_dino_coordinates(source_steps, "primary"),
                "state_permutation": transported_dino_coordinates(
                    source_steps, "state_permutation"
                ),
                "random_orthogonal_map": transported_dino_coordinates(
                    source_steps, "random_orthogonal"
                ),
            }
            primary_deltas = [
                reconstruct_dino_intervention(value) for value in coordinate_sets["primary"]
            ]
            delta_sets = {
                condition: [reconstruct_dino_intervention(value) for value in coordinates]
                for condition, coordinates in coordinate_sets.items()
            }
            delta_sets["random_matched_subspace"] = [
                reconstruct_dino_intervention(value, "random_matched_subspace")
                for value in coordinate_sets["primary"]
            ]
            for condition in ["state_permutation", "random_orthogonal_map", "random_matched_subspace"]:
                delta_sets[condition] = [
                    norm_match(delta, reference)
                    for delta, reference in zip(delta_sets[condition], primary_deltas)
                ]
            for condition, deltas_for_word in delta_sets.items():
                patched = patched_grounded_word(
                    bundle, record, zero,
                    {step: delta for step, delta in enumerate(deltas_for_word)},
                    MODEL_ARTIFACTS["dino"]["decoder"],
                )
                transported_endpoints[condition].append(patched[-1])
                forward_count += 1
        transported_endpoints = {
            key: np.asarray(value) for key, value in transported_endpoints.items()
        }
        goal_rng = np.random.default_rng(
            stable_seed(DESIGN_SEED, "planning_goals", record["record_id"])
        )
        goal_indices = goal_rng.choice(
            len(candidate_names), size=min(PLANNING_GOALS_PER_RECORD, len(candidate_names)),
            replace=False,
        )
        planning_rows = []
        for goal_index in goal_indices:
            goal = physical_endpoints[int(goal_index)]
            true_costs = planning_cost(physical_endpoints, goal)
            native_choice = int(np.argmin(planning_cost(native_endpoints, goal)))
            choices = {"dino_native": native_choice}
            choices.update({
                condition: int(np.argmin(planning_cost(endpoints, goal)))
                for condition, endpoints in transported_endpoints.items()
            })
            planning_rows.append({
                "record_id": int(record["record_id"]),
                "trajectory_id": int(record["trajectory_id"]),
                "mode": str(record["mode"]), "goal_candidate": candidate_names[int(goal_index)],
                **{f"{condition}_choice": int(choice) for condition, choice in choices.items()},
                **{
                    f"{condition}_normalized_regret": normalized_regret(true_costs, choice)
                    for condition, choice in choices.items()
                },
                "oracle_normalized_regret": 0.0,
            })
    atomic_npz(
        path, identity=np.asarray(identity),
        interchange_rows=np.asarray([json.dumps(row, sort_keys=True) for row in interchange_rows]),
        planning_rows=np.asarray([json.dumps(row, sort_keys=True) for row in planning_rows]),
        forward_count=np.asarray(forward_count, dtype=np.int64),
    )
    return interchange_rows, planning_rows, forward_count


INTERCHANGE_ROWS = []
PLANNING_ROWS = []
if not PIPELINE_FAILED:
    try:
        verify_executed_notebook_through(
            "# Transport reachable internal responses through the single frozen predictive map and test planning."
        )
        bundle = load_world_model("dino_wm_pusht")
        try:
            for index, record in enumerate(SELECTED_RECORDS["evaluation"]):
                interchange, planning, count = evaluate_causal_record(bundle, record)
                INTERCHANGE_ROWS.extend(interchange)
                PLANNING_ROWS.extend(planning)
                PROVENANCE_COUNTS["patched_forwards"] += int(count)
                write_json(OUT / "causal_transport_progress.json", {
                    "completed": index + 1, "total": len(SELECTED_RECORDS["evaluation"]),
                    "last_record_id": int(record["record_id"]),
                    "patched_forwards": int(PROVENANCE_COUNTS["patched_forwards"]),
                })
        finally:
            unload_world_model(bundle)
        write_csv(EVIDENCE_DIR / "cross_model_internal_interchange_rows.csv", INTERCHANGE_ROWS)
        write_csv(EVIDENCE_DIR / "transported_planning_rows.csv", PLANNING_ROWS)
        atomic_checkpoint("causal_transport_complete", {
            "interchange_rows": len(INTERCHANGE_ROWS),
            "planning_rows": len(PLANNING_ROWS),
            "patched_forwards": int(PROVENANCE_COUNTS["patched_forwards"]),
            "sole_cross_model_map_sha256": sha256_file(MAP_DIR / "frozen_state_map.npz"),
        })
        memory_report("stage33_causal_transport_complete")
    except Exception:
        record_failure("stage33_cross_model_internal_interchange_or_planning")
'''


decision_and_reporting = r'''# Apply preregistered cumulative gates, multiplicity correction, and automatic interpretation.


def rows_array(rows, key, condition=None):
    selected = rows
    if condition is not None:
        selected = [row for row in rows if row.get("condition") == condition]
    return np.asarray([row[key] for row in selected], dtype=np.float64)


def grouped_values(rows, key, condition=None):
    selected = rows if condition is None else [row for row in rows if row.get("condition") == condition]
    return (
        np.asarray([row[key] for row in selected], dtype=np.float64),
        np.asarray([row["trajectory_id"] for row in selected], dtype=np.int64),
    )


def sign_flip_pvalue(values, groups, alternative="greater"):
    array = np.asarray(values, dtype=np.float64)
    labels = np.asarray(groups)
    means = np.asarray([
        np.mean(array[labels == group]) for group in sorted(set(labels.tolist()))
    ])
    observed = float(np.mean(means))
    count = len(means)
    if count <= 20:
        indices = np.arange(1 << count, dtype=np.uint64)[:, None]
        bits = ((indices >> np.arange(count, dtype=np.uint64)) & 1).astype(np.float64)
        signs = 2.0 * bits - 1.0
        null = np.mean(signs * means[None], axis=1)
    else:
        rng = np.random.default_rng(stable_seed(BOOTSTRAP_SEED, "sign_flip", count))
        signs = rng.choice([-1.0, 1.0], size=(200000, count))
        null = np.mean(signs * means[None], axis=1)
    if alternative == "greater":
        return float((np.sum(null >= observed) + 1) / (len(null) + 1))
    if alternative == "less":
        return float((np.sum(null <= observed) + 1) / (len(null) + 1))
    raise ValueError(alternative)


def summarize_realization_model(short):
    rows = [
        row for row in REALIZATION_ROWS
        if row["model"] == short and int(row["word_length"]) >= 2
    ]
    groups = np.asarray([row["trajectory_id"] for row in rows], dtype=np.int64)
    global_error = rows_array(rows, "global_relative_squared_error")
    physical_error = rows_array(rows, "hybrid_physical_relative_squared_error")
    label_free_error = rows_array(rows, "hybrid_label_free_relative_squared_error")
    nonlinear_error = rows_array(rows, "nonlinear_relative_squared_error")
    decoder_error = rows_array(rows, "decoder_only_relative_squared_error")
    shuffled_error = rows_array(rows, "action_word_shuffled_relative_squared_error")
    permuted_mode_error = rows_array(rows, "mode_label_permuted_relative_squared_error")
    denominator = np.maximum(global_error, 1e-12)
    physical_gain = (global_error - physical_error) / denominator
    label_free_gain = (global_error - label_free_error) / denominator
    nonlinear_advantage = (nonlinear_error - physical_error) / np.maximum(nonlinear_error, 1e-12)
    decoder_advantage = (decoder_error - physical_error) / np.maximum(decoder_error, 1e-12)
    shuffle_advantage = (shuffled_error - physical_error) / np.maximum(shuffled_error, 1e-12)
    mode_advantage = (permuted_mode_error - physical_error) / np.maximum(permuted_mode_error, 1e-12)
    family_means = {
        family: float(np.mean(physical_gain[np.asarray([row["word_family"] for row in rows]) == family]))
        for family in sorted({row["word_family"] for row in rows})
    }
    mode_means = {
        mode: float(np.mean(physical_gain[np.asarray([row["mode"] for row in rows]) == mode]))
        for mode in MODE_LABELS
    }
    return {
        "rows": len(rows), "trajectories": len(set(groups.tolist())),
        "mean_hybrid_relative_gain": float(np.mean(physical_gain)),
        "hybrid_gain_ci95": clustered_bootstrap_interval(
            physical_gain, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
            seed=stable_seed(BOOTSTRAP_SEED, short, "hybrid"), alpha=HOLM_ALPHA,
        ),
        "mean_label_free_gain": float(np.mean(label_free_gain)),
        "label_free_retention": float(
            np.mean(label_free_gain) / max(np.mean(physical_gain), 1e-12)
        ),
        "mean_nonlinear_advantage": float(np.mean(nonlinear_advantage)),
        "mean_decoder_only_advantage": float(np.mean(decoder_advantage)),
        "mean_action_shuffle_advantage": float(np.mean(shuffle_advantage)),
        "mean_mode_permutation_advantage": float(np.mean(mode_advantage)),
        "family_mean_gains": family_means, "mode_mean_gains": mode_means,
        "p_hybrid_gain": sign_flip_pvalue(physical_gain, groups),
        "passed": bool(
            np.mean(physical_gain) >= MIN_HYBRID_RELATIVE_GAIN
            and clustered_bootstrap_interval(
                physical_gain, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
                seed=stable_seed(BOOTSTRAP_SEED, short, "hybrid_gate"), alpha=HOLM_ALPHA,
            )[0] > 0
            and np.mean(label_free_gain) >= MIN_LABEL_FREE_GAIN_RETENTION * np.mean(physical_gain)
            and all(value > 0 for value in MODE_DISCOVERY_COUNTS[short].values())
            and np.mean(nonlinear_advantage) > 0
            and np.mean(decoder_advantage) > 0
            and all(value > 0 for value in family_means.values())
            and all(value > 0 for value in mode_means.values())
        ),
    }


def summarize_cross_model_map():
    rows = [row for row in MAP_ROWS if int(row["word_length"]) >= 2]
    groups = np.asarray([row["trajectory_id"] for row in rows], dtype=np.int64)
    primary = np.sqrt(rows_array(rows, "primary_relative_squared_error"))
    state_control = np.sqrt(rows_array(rows, "state_permutation_relative_squared_error"))
    orthogonal_control = np.sqrt(rows_array(rows, "random_orthogonal_relative_squared_error"))
    action_specific = np.sqrt(rows_array(rows, "action_specific_relative_squared_error"))
    conjugacy = np.sqrt(rows_array(rows, "operator_conjugacy_relative_squared_error"))
    best_control = np.minimum(state_control, orthogonal_control)
    control_advantage = (best_control - primary) / np.maximum(best_control, 1e-12)
    family_primary = {
        family: float(np.mean(primary[np.asarray([row["word_family"] for row in rows]) == family]))
        for family in sorted({row["word_family"] for row in rows})
    }
    return {
        "rows": len(rows), "mean_primary_relative_error": float(np.mean(primary)),
        "primary_relative_error_ci95": clustered_bootstrap_interval(
            primary, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
            seed=stable_seed(BOOTSTRAP_SEED, "map_primary"), alpha=HOLM_ALPHA,
        ),
        "mean_operator_conjugacy_relative_error": float(np.mean(conjugacy)),
        "mean_control_advantage": float(np.mean(control_advantage)),
        "control_advantage_ci95": clustered_bootstrap_interval(
            control_advantage, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
            seed=stable_seed(BOOTSTRAP_SEED, "map_controls"), alpha=HOLM_ALPHA,
        ),
        "global_to_action_specific_error_ratio": float(
            np.mean(primary) / max(np.mean(action_specific), 1e-12)
        ),
        "family_primary_errors": family_primary,
        "p_control_advantage": sign_flip_pvalue(control_advantage, groups),
        "map_condition_number": float(STATE_MAPS["primary"]["condition_number"]),
        "map_minimum_singular_value": float(STATE_MAPS["primary"]["minimum_singular_value"]),
        "passed": bool(
            np.mean(conjugacy) <= MAX_CONJUGACY_RELATIVE_ERROR
            and np.mean(control_advantage) >= MIN_CONTROL_ADVANTAGE
            and clustered_bootstrap_interval(
                control_advantage, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
                seed=stable_seed(BOOTSTRAP_SEED, "map_control_gate"), alpha=HOLM_ALPHA,
            )[0] > 0
            and np.mean(primary) / max(np.mean(action_specific), 1e-12)
                <= MAX_GLOBAL_TO_ACTION_SPECIFIC_ERROR_RATIO
            and all(value <= 2 * MAX_CONJUGACY_RELATIVE_ERROR for value in family_primary.values())
        ),
    }


def summarize_interchange():
    eligible = [row for row in INTERCHANGE_ROWS if row["grounded_eligible"]]
    conditions = sorted({row["condition"] for row in INTERCHANGE_ROWS})
    summaries = {}
    for condition in conditions:
        rows = [row for row in eligible if row["condition"] == condition]
        summaries[condition] = {
            "rows": len(rows),
            "mean_grounded_cosine": float(np.mean(rows_array(rows, "grounded_cosine"))) if rows else -1.0,
            "mean_grounded_relative_error": float(np.mean(rows_array(rows, "grounded_relative_error"))) if rows else 1e9,
            "mean_grounded_error_gain": float(np.mean(rows_array(rows, "grounded_error_gain"))) if rows else -1.0,
            "mean_self_cosine": float(np.mean(rows_array(rows, "self_cosine"))) if rows else -1.0,
        }
    primary_rows = [row for row in eligible if row["condition"] == "primary"]
    if not primary_rows:
        return {
            "by_condition": summaries, "primary_cosine_ci95": [-1.0, -1.0],
            "primary_error_gain_ci95": [-1.0, -1.0],
            "control_advantage_ci95": [-1.0, -1.0],
            "mean_control_advantage": -1.0,
            "mode_mean_error_gains": {mode: -1.0 for mode in MODE_LABELS},
            "pair_mean_error_gains": {
                str(index): -1.0 for index in range(len(EVALUATION_INTERCHANGE_PAIRS))
            },
            "p_control_advantage": 1.0, "passed": False,
        }
    groups = np.asarray([row["trajectory_id"] for row in primary_rows], dtype=np.int64)
    primary_cosine = rows_array(primary_rows, "grounded_cosine")
    primary_gain = rows_array(primary_rows, "grounded_error_gain")
    control_conditions = [
        "state_permutation", "random_orthogonal_map", "random_matched_subspace"
    ]
    by_key = {
        (row["record_id"], row["pair_index"], row["condition"]): row
        for row in eligible
    }
    advantages = []
    for row in primary_rows:
        controls = [
            by_key[(row["record_id"], row["pair_index"], condition)]["grounded_error_gain"]
            for condition in control_conditions
            if (row["record_id"], row["pair_index"], condition) in by_key
        ]
        advantages.append(row["grounded_error_gain"] - max(controls))
    advantages = np.asarray(advantages, dtype=np.float64)
    mode_means = {
        mode: float(np.mean([
            row["grounded_error_gain"] for row in primary_rows if row["mode"] == mode
        ])) for mode in MODE_LABELS
    }
    pair_means = {
        str(index): float(np.mean([
            row["grounded_error_gain"] for row in primary_rows if row["pair_index"] == index
        ])) for index in range(len(EVALUATION_INTERCHANGE_PAIRS))
    }
    return {
        "by_condition": summaries,
        "primary_cosine_ci95": clustered_bootstrap_interval(
            primary_cosine, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
            seed=stable_seed(BOOTSTRAP_SEED, "interchange_cosine"), alpha=HOLM_ALPHA,
        ),
        "primary_error_gain_ci95": clustered_bootstrap_interval(
            primary_gain, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
            seed=stable_seed(BOOTSTRAP_SEED, "interchange_gain"), alpha=HOLM_ALPHA,
        ),
        "control_advantage_ci95": clustered_bootstrap_interval(
            advantages, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
            seed=stable_seed(BOOTSTRAP_SEED, "interchange_controls"), alpha=HOLM_ALPHA,
        ),
        "mean_control_advantage": float(np.mean(advantages)),
        "mode_mean_error_gains": mode_means, "pair_mean_error_gains": pair_means,
        "p_control_advantage": sign_flip_pvalue(advantages, groups),
        "passed": bool(
            np.mean(primary_cosine) >= MIN_GROUNDED_INTERCHANGE_COSINE
            and np.mean(primary_gain) >= MIN_INTERCHANGE_RELATIVE_ERROR_GAIN
            and clustered_bootstrap_interval(
                advantages, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
                seed=stable_seed(BOOTSTRAP_SEED, "interchange_control_gate"), alpha=HOLM_ALPHA,
            )[0] > 0
            and all(value > 0 for value in mode_means.values())
            and all(value > 0 for value in pair_means.values())
        ),
    }


def summarize_planning():
    groups = np.asarray([row["trajectory_id"] for row in PLANNING_ROWS], dtype=np.int64)
    native = rows_array(PLANNING_ROWS, "dino_native_normalized_regret")
    primary = rows_array(PLANNING_ROWS, "primary_normalized_regret")
    degradation = primary - native
    controls = np.column_stack([
        rows_array(PLANNING_ROWS, f"{condition}_normalized_regret")
        for condition in ["state_permutation", "random_orthogonal_map", "random_matched_subspace"]
    ])
    control_advantage = np.min(controls, axis=1) - primary
    mode_degradation = {
        mode: float(np.mean(degradation[np.asarray([row["mode"] for row in PLANNING_ROWS]) == mode]))
        for mode in MODE_LABELS
    }
    return {
        "rows": len(PLANNING_ROWS),
        "mean_native_regret": float(np.mean(native)),
        "mean_transported_regret": float(np.mean(primary)),
        "mean_regret_degradation": float(np.mean(degradation)),
        "degradation_ci95": clustered_bootstrap_interval(
            degradation, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
            seed=stable_seed(BOOTSTRAP_SEED, "planning_degradation"), alpha=HOLM_ALPHA,
        ),
        "mean_control_advantage": float(np.mean(control_advantage)),
        "control_advantage_ci95": clustered_bootstrap_interval(
            control_advantage, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
            seed=stable_seed(BOOTSTRAP_SEED, "planning_controls"), alpha=HOLM_ALPHA,
        ),
        "mode_mean_degradation": mode_degradation,
        "p_control_advantage": sign_flip_pvalue(control_advantage, groups),
        "passed": bool(
            clustered_bootstrap_interval(
                degradation, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
                seed=stable_seed(BOOTSTRAP_SEED, "planning_degradation_gate"), alpha=HOLM_ALPHA,
            )[1] <= MAX_PLANNING_REGRET_DEGRADATION
            and clustered_bootstrap_interval(
                control_advantage, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
                seed=stable_seed(BOOTSTRAP_SEED, "planning_control_gate"), alpha=HOLM_ALPHA,
            )[0] > 0
            and all(value <= 2 * MAX_PLANNING_REGRET_DEGRADATION for value in mode_degradation.values())
        ),
    }


def make_stage33_plots(realization, cross_map, interchange, planning):
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].bar(
        ["JEPA", "DINO"],
        [realization[name]["mean_hybrid_relative_gain"] for name in ["jepa", "dino"]],
        color=["#4c78a8", "#f58518"],
    )
    axes[0, 0].axhline(MIN_HYBRID_RELATIVE_GAIN, color="black", linestyle="--")
    axes[0, 0].set(title="Hybrid gain over one global operator", ylabel="relative gain")
    axes[0, 1].bar(
        ["primary", "state perm.", "random map"],
        [
            np.mean(np.sqrt(rows_array(MAP_ROWS, "primary_relative_squared_error"))),
            np.mean(np.sqrt(rows_array(MAP_ROWS, "state_permutation_relative_squared_error"))),
            np.mean(np.sqrt(rows_array(MAP_ROWS, "random_orthogonal_relative_squared_error"))),
        ], color=["#54a24b", "#b279a2", "#bab0ac"],
    )
    axes[0, 1].set(title="Held-out state transport", ylabel="relative error")
    condition_names = ["primary", "state_permutation", "random_orthogonal_map", "random_matched_subspace"]
    axes[1, 0].bar(
        condition_names,
        [interchange["by_condition"][name]["mean_grounded_error_gain"] for name in condition_names],
        color=["#4c78a8", "#b279a2", "#bab0ac", "#e45756"],
    )
    axes[1, 0].tick_params(axis="x", rotation=20)
    axes[1, 0].set(title="Grounded counterfactual interchange", ylabel="error gain")
    axes[1, 1].bar(
        ["DINO native", "transported", "best control"],
        [
            planning["mean_native_regret"], planning["mean_transported_regret"],
            planning["mean_transported_regret"] + planning["mean_control_advantage"],
        ], color=["#f58518", "#4c78a8", "#bab0ac"],
    )
    axes[1, 1].set(title="Physical planning", ylabel="normalized regret")
    figure.tight_layout()
    figure.savefig(PLOT_DIR / "stage33_bipca_summary.png", dpi=180)
    plt.close(figure)


DECISION_PAYLOAD = {"status": "INCONCLUSIVE_PIPELINE_FAILURE"}
if not PIPELINE_FAILED:
    try:
        verify_executed_notebook_through(
            "# Apply preregistered cumulative gates, multiplicity correction, and automatic interpretation."
        )
        REALIZATION_SUMMARY = {
            short: summarize_realization_model(short) for short in ["jepa", "dino"]
        }
        MAP_SUMMARY = summarize_cross_model_map()
        INTERCHANGE_SUMMARY = summarize_interchange()
        PLANNING_SUMMARY = summarize_planning()
        pvalues = np.asarray([
            REALIZATION_SUMMARY["jepa"]["p_hybrid_gain"],
            REALIZATION_SUMMARY["dino"]["p_hybrid_gain"],
            MAP_SUMMARY["p_control_advantage"],
            INTERCHANGE_SUMMARY["p_control_advantage"],
            PLANNING_SUMMARY["p_control_advantage"],
        ])
        multiplicity = holm_adjust(pvalues, alpha=HOLM_ALPHA)
        rank_gate = bool(
            RANK_STABILITY_LOCK_PASSED
        )
        decoder_gate = bool(
            all(DECODER_METRICS[short]["median_r2"] >= MIN_DECODER_MEDIAN_R2 for short in ["jepa", "dino"])
        )
        hybrid_gate = bool(
            all(REALIZATION_SUMMARY[short]["passed"] for short in ["jepa", "dino"])
            and np.all(multiplicity["reject"][:2])
        )
        fixed_map_gate = bool(
            STATE_MAPS["primary"].get("strict_fit_passed", False)
            and STATE_MAPS["primary"]["condition_number"] <= CARRIER_MAP_MAX_CONDITION
            and STATE_MAPS["primary"]["minimum_singular_value"] >= MIN_MAP_SINGULAR_VALUE
            and sha256_file(MAP_DIR / "calibration_freeze.json") == CALIBRATION_LOCK_SHA256
        )
        conjugacy_gate = bool(MAP_SUMMARY["passed"] and multiplicity["reject"][2])
        interchange_gate = bool(INTERCHANGE_SUMMARY["passed"] and multiplicity["reject"][3])
        planning_gate = bool(PLANNING_SUMMARY["passed"] and multiplicity["reject"][4])
        controls_gate = bool(
            MAP_SUMMARY["mean_control_advantage"] >= MIN_CONTROL_ADVANTAGE
            and INTERCHANGE_SUMMARY["mean_control_advantage"] > 0
            and PLANNING_SUMMARY["mean_control_advantage"] > 0
            and all(REALIZATION_SUMMARY[short]["mean_action_shuffle_advantage"] > 0 for short in ["jepa", "dino"])
            and all(REALIZATION_SUMMARY[short]["mean_mode_permutation_advantage"] > 0 for short in ["jepa", "dino"])
            and max(row["zero_edit_max_abs"] for row in INTERCHANGE_ROWS)
                <= MAX_ZERO_EDIT_ERROR
        )
        same_model_positive_gate = bool(
            all(
                SPLIT_HALF_METRICS[short]["relative_rmse"]
                    <= MAX_SAME_MODEL_SPLIT_HALF_ERROR
                for short in ["jepa", "dino"]
            )
            and INTERCHANGE_SUMMARY["by_condition"].get(
                "dino_self_positive", {"mean_self_cosine": -1.0}
            )["mean_self_cosine"] >= MIN_GROUNDED_INTERCHANGE_COSINE
        )
        family_gate = bool(
            all(
                all(value > 0 for value in REALIZATION_SUMMARY[short]["family_mean_gains"].values())
                for short in ["jepa", "dino"]
            )
            and all(value > 0 for value in INTERCHANGE_SUMMARY["pair_mean_error_gains"].values())
        )
        confirmation_eligible = bool(
            RUN_MODE == "pilot"
            and SOURCE_IDENTITY.get("confirmation_eligible", False)
            and len({row["trajectory_id"] for row in SELECTED_RECORDS["evaluation"]})
                >= MIN_EVALUATION_TRAJECTORIES
            and len(MODE_LABELS) == 4
            and all(
                len({
                    row["trajectory_id"] for row in SELECTED_RECORDS["evaluation"]
                    if row["mode"] == mode
                }) >= MIN_EVALUATION_MODE_TRAJECTORIES
                for mode in MODE_LABELS
            )
            and same_model_positive_gate
            and not PIPELINE_FAILED
        )
        raw_decision = derive_decision(
            {
                "stable_low_rank": bool(rank_gate and decoder_gate),
                "hybrid_improvement": hybrid_gate,
                "fixed_map_well_conditioned": fixed_map_gate,
                "heldout_conjugacy": conjugacy_gate,
                "internal_interchange": interchange_gate,
                "planning_value": planning_gate,
                "controls_rejected": controls_gate,
                "family_consistency": family_gate,
            },
            run_mode=RUN_MODE, confirmation_eligible=confirmation_eligible,
        )
        status_map = {
            "pass": "BOUNDED_INTERVENTIONAL_PREDICTIVE_CAUSAL_ABSTRACTION_SUPPORTED",
            "partial_pass": "OPERATOR_CONJUGACY_WITHOUT_FULL_CAUSAL_PLANNING_CERTIFICATE",
            "fail": "BOUNDED_SHARED_ABSTRACT_MECHANISM_NOT_SUPPORTED",
        }
        if RUN_MODE == "smoke":
            status = "SMOKE_ONLY"
        elif not same_model_positive_gate:
            status = "INCONCLUSIVE_SAME_MODEL_POSITIVE_CONTROL_FAILURE"
        else:
            status = status_map[raw_decision["status"]]
        DECISION_PAYLOAD = {
            "status": status, "protocol_decision": raw_decision,
            "confirmation_eligible": confirmation_eligible,
            "decoder_gate": decoder_gate, "rank_gate": rank_gate,
            "same_model_positive_control_gate": same_model_positive_gate,
            "realization_summary": REALIZATION_SUMMARY,
            "cross_model_map_summary": MAP_SUMMARY,
            "interchange_summary": INTERCHANGE_SUMMARY,
            "planning_summary": PLANNING_SUMMARY,
            "holm_family": {
                "hypotheses": [
                    "JEPA hybrid gain", "DINO hybrid gain", "map control advantage",
                    "interchange control advantage", "planning control advantage",
                ],
                "raw_pvalues": pvalues.tolist(),
                "adjusted_pvalues": multiplicity["adjusted_pvalues"].tolist(),
                "reject": multiplicity["reject"].tolist(),
            },
            "sole_cross_model_map": "predictive-coordinate S; within-model carrier bridges are not cross-model maps",
            "evidence_levels": {
                "1_common_decodability": decoder_gate,
                "2_similar_bounded_rank": rank_gate,
                "3_similar_predictive_subspaces": fixed_map_gate,
                "4_heldout_operator_conjugacy": conjugacy_gate,
                "5_cross_model_counterfactual_interchange": interchange_gate,
                "6_preserved_planning_value": planning_gate,
            },
            "claim_boundary": {
                "bounded_action_bank": True, "finite_horizon_at_most": MAX_WORD_LENGTH,
                "one_environment": ENVIRONMENT, "one_checkpoint_per_family": True,
                "shared_dinov2_target_family_confound": True,
                "minimal_neural_mechanism_identified": False,
                "universal_predictive_equivalence_claimed": False,
                "shared_abstract_causal_mechanism_claim_requires_levels_4_to_6": True,
            },
            "provenance_counts": PROVENANCE_COUNTS,
        }
        write_json(OUT / "run_provenance_certificate.json", {
            "protocol_id": PROTOCOL_ID,
            "run_signature": RUN_SIGNATURE,
            "run_nonce": RUN_NONCE,
            "resumed_run": bool(RESUMED_RUN),
            "source_bound": bool(SOURCE_IDENTITY.get("confirmation_eligible", False)),
            "split_trajectory_counts": {
                split: len({record["trajectory_id"] for record in records})
                for split, records in SELECTED_RECORDS.items()
            },
            "split_record_counts": {
                split: len(records) for split, records in SELECTED_RECORDS.items()
            },
            "calibration_lock_sha256": CALIBRATION_LOCK_SHA256,
            "evaluation_open_sha256": sha256_file(
                MAP_DIR / "evaluation_open_certificate.json"
            ),
            "provenance_counts": PROVENANCE_COUNTS,
            "confirmation_eligible": confirmation_eligible,
        })
        write_json(OUT / "stage33_decision.json", DECISION_PAYLOAD)
        make_stage33_plots(
            REALIZATION_SUMMARY, MAP_SUMMARY, INTERCHANGE_SUMMARY, PLANNING_SUMMARY
        )
        interpretation = f"""# Automatic Stage 33 interpretation

Status: **{status}**

This run tests a finite, horizon-{MAX_WORD_LENGTH} predictive response bank.  It does not
identify a universal minimal state.  Evidence levels 1--3 concern decodability,
rank, and coordinate alignment; only levels 4--6 concern held-out operator
conjugacy, transported internal counterfactuals, and planning value.

Common predictive rank: {COMMON_RANK}.  JEPA/DINO held-out decoder median R2:
{DECODER_METRICS['jepa']['median_r2']:.3f}/{DECODER_METRICS['dino']['median_r2']:.3f}.
The one predictive cross-model map has condition number
{STATE_MAPS['primary']['condition_number']:.3f}.  Mean held-out operator-conjugacy
relative error is {MAP_SUMMARY['mean_operator_conjugacy_relative_error']:.3f}.
Primary grounded interchange cosine is
{INTERCHANGE_SUMMARY['by_condition']['primary']['mean_grounded_cosine']:.3f};
transported-minus-native planning regret is
{PLANNING_SUMMARY['mean_regret_degradation']:.3f}.

The correct reading is bounded to these two public PushT checkpoints and their
shared DINOv2 target family.  A positive status supports a shared abstract,
causally used predictive response organization—not shared circuitry, a globally
minimal physical state, or a general theorem about JEPA-style models.
"""
        (OUT / "AUTOMATIC_INTERPRETATION.md").write_text(interpretation)
        print(json.dumps(DECISION_PAYLOAD, indent=2))
    except Exception:
        record_failure("stage33_decision_multiplicity_plots_or_interpretation")
        DECISION_PAYLOAD = {"status": "INCONCLUSIVE_PIPELINE_FAILURE", "failure": FAILURE_MESSAGE}

if not (OUT / "stage33_decision.json").exists():
    write_json(OUT / "stage33_decision.json", DECISION_PAYLOAD)
'''


packaging = r'''# Package compact audit evidence while retaining the complete resumable Drive directory.
TIMINGS["total_seconds"] = float(time.time() - RUN_STARTED_AT)
write_json(OUT / "timings.json", TIMINGS)
memory_report("final")
if not PIPELINE_FAILED:
    (OUT / "FAILURE_TRACE.txt").write_text("NONE\n")

raw_roots = [TRUTH_DIR, BASELINE_DIR, CAUSAL_DIR]
excluded_roots = {ASSET_DIR, *raw_roots}
RAW_MANIFEST = manifest_rows(OUT, excluded_roots=())
write_json(OUT / "raw_manifest.json", RAW_MANIFEST)

compact_files = []
for path in sorted(OUT.rglob("*")):
    if not path.is_file():
        continue
    if any(root == path or root in path.parents for root in excluded_roots):
        continue
    if path.name.startswith("stage33_bipca_result_bundle_"):
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
archive_base = OUT / f"stage33_bipca_result_bundle_{RUN_SIGNATURE[:12]}"
archive = Path(shutil.make_archive(str(archive_base), "zip", staging))
shutil.rmtree(staging)
if not PIPELINE_FAILED:
    INCOMPLETE_POINTER.unlink(missing_ok=True)
print(f"RUN_STATUS: {DECISION_PAYLOAD['status']}")
print(f"COMPLETE_RESUMABLE_DRIVE_DIRECTORY: {OUT}")
print(f"RESULT_BUNDLE: {archive}")
print(f"RESULT_BUNDLE_SHA256: {sha256_file(archive)}")
print(f"COUNTS: {json.dumps(PROVENANCE_COUNTS, sort_keys=True)}")
print(f"DEVICE: {VERSIONS['gpu']}; ELAPSED_MINUTES: {sum(TIMINGS.values()) / 60.0:.1f}")
if DOWNLOAD_RESULTS:
    try:
        from google.colab import files
        files.download(str(archive))
    except Exception as error:
        print(f"Automatic download unavailable: {error}")
'''


protocol_sources = [
    introduction, configuration, installation, setup, analysis_helpers,
    model_helpers, design_and_runtime_helpers, physical_truth,
    construction_and_models, model_selection_and_calibration, locked_evaluation,
    causal_transport, decision_and_reporting, packaging,
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
    code(design_and_runtime_helpers),
    code(physical_truth),
    code(construction_and_models),
    code(model_selection_and_calibration),
    code(locked_evaluation),
    code(causal_transport),
    code(decision_and_reporting),
    code(packaging),
]
for index, cell in enumerate(cells):
    cell["id"] = f"stage33-{index:02d}"

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
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
TARGET.write_text(json.dumps(notebook, indent=1) + "\n")
print(f"Wrote {TARGET}")
