import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STAGE11 = ROOT / "11_action_response_geometry_pilot.ipynb"
GEOMETRY = ROOT.parent / "src/cf_faithfulness/stage13b_geometry.py"
NUMERICAL = ROOT.parent / "src/cf_faithfulness/stage14_pcj.py"
TARGET = ROOT / "14_predictive_control_j_bundle_pilot.ipynb"


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
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if node.name in names:
                found[node.name] = ast.get_source_segment(source, node)
    missing = sorted(set(names) - set(found))
    if missing:
        raise RuntimeError(f"missing source functions: {missing}")
    return "\n\n\n".join(found[name] for name in names)


def assigned_uppercase_names(source):
    """Return the protocol constants assigned by one generated source cell."""
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


stage11 = json.loads(STAGE11.read_text())
stage11_helpers = "".join(stage11["cells"][4]["source"])
simulator_helpers = function_sources(
    stage11_helpers,
    [
        "to_model_observation",
        "configure_repo",
        "pose_target",
        "make_environment",
        "wall_visual",
        "reset_environment",
        "rollout_branch",
        "exact_restore_test",
    ],
)
simulator_helpers = simulator_helpers.replace(
    'task = TASKS["Wall"][0]',
    'raise RuntimeError("Stage 14 supports PushT only")',
)
simulator_helpers = simulator_helpers.replace(
    '''        observation, restored = env.reset()
        payload = {
            "visual": np.asarray(observation["visual"]).copy(),''',
    '''        observation, restored = env.reset()
        requested_goal = np.asarray(task["goal"], dtype=np.float64)
        env.set_task_goal(requested_goal.copy())
        if not np.array_equal(np.asarray(env.goal_pose), requested_goal):
            raise RuntimeError("PushT task goal was not installed exactly")
        payload = {
            "visual": np.asarray(env.render("rgb_array")).copy(),''',
)
cached_repo_setup = '''def configure_repo():
    repo = CACHE_ROOT / "jepa-wms"
    if not repo.exists():
        subprocess.run(["git", "clone", REPO_URL, str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "fetch", "origin", REPO_COMMIT],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "checkout", "--detach", REPO_COMMIT],
        check=True,
    )
    resolved = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()'''
ephemeral_repo_setup = r'''def configure_repo():
    # Google Drive is reliable for immutable model assets but not as a mutable
    # Git worktree. Use a fresh runtime-local checkout and keep HF/Torch assets
    # in the persistent cache configured by the setup cell.
    repo = Path("/content") / f"stage14-jepa-wms-{REPO_COMMIT[:12]}"
    if repo.exists():
        shutil.rmtree(repo)

    def run_git(command):
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "git repository setup failed\n"
                f"command: {command!r}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        return completed

    print(f"Preparing clean ephemeral JEPA-WM source at {repo}")
    run_git(["git", "clone", "--no-checkout", REPO_URL, str(repo)])
    run_git(["git", "-C", str(repo), "fetch", "origin", REPO_COMMIT])
    run_git(["git", "-C", str(repo), "checkout", "--detach", REPO_COMMIT])
    resolved = run_git(
        ["git", "-C", str(repo), "rev-parse", "HEAD"]
    ).stdout.strip()'''
if cached_repo_setup not in simulator_helpers:
    raise RuntimeError("Stage 11 repository setup source changed unexpectedly")
simulator_helpers = simulator_helpers.replace(
    cached_repo_setup,
    ephemeral_repo_setup,
)

geometry_helpers = function_sources(
    GEOMETRY.read_text(), ["array_sha256", "frozen_action_bank"]
)
numerical_helpers = function_sources(
    NUMERICAL.read_text(),
    [
        "channel_metric_from_moments",
        "transform_primal_channels",
        "inverse_transform_primal_channels",
        "transform_dual_channels",
        "balanced_modes",
        "canonical_mode_rows",
        "training_span",
        "omp_codes",
        "sparse_reconstruct",
        "transfer_metrics",
        "earliest_within_one_se",
        "exact_positive_sign_test",
        "one_sided_t_lower",
        "hierarchical_bootstrap_means",
        "haar_rotation",
        "relative_error",
    ],
)


introduction = r'''# Stage 14: predictive-control J-bundle pilot

This notebook tests a JEPA-native hypothesis rather than reviving the failed
global outcome dictionary from Stages 13/13b.

For a frozen PushT JEPA-WM it estimates, at an internal predictor carrier:

\[
G_s = D_H q(F_h), \qquad B_s = D_u H, \qquad K_s = G_s B_s,
\]

where `q` is a fixed native-energy contrast between two true future target
embeddings.  `G` therefore asks which internal directions a future query can
read; `B` asks which directions normalized executable action trajectories can
write.  The notebook forms local balanced modes from the small Hankel operator
`K`, learns one signed sparse frame on construction tasks, and freezes it
before evaluating:

- Stage-14-held-out action families (`turn_xy`, `three_phase`);
- Stage-14-held-out, state-specific oracle target-energy queries;
- crossed train-query/test-action and test-query/train-action cells;
- task-disjoint states;
- causal indirect-path patches and matched controls;
- horizon-1 reuse of a horizon-3 frame;
- free/contact interaction strata; and
- hidden-coordinate gauge identities.

The full 256-token × 400-channel AdaLN block output is the carrier.  The
primary reader is the native visual-token energy component; this is not the
model's complete visual-plus-proprio planning objective.  No PCA
outcome prototypes, goals, costs, rewards, value heads, or planner outcomes are
used to fit the frame. The action tangent design has only two independent
temporal trajectories per split, and the evaluation readers are target-aligned
oracle assays rather than deployable readouts. A pass is therefore called
`JEPA_NATIVE_PREDICTIVE_CONTROL_BUNDLE_CANDIDATE`, not “workspace” or
“J-space”; those names require independent fixed readers and a frozen native
context/action-to-code interface.

`smoke` mode validates plumbing and cannot produce scientific evidence.
`pilot` mode is a preregisterable new-data pilot only after the notebook is
source-bound and committed before execution; otherwise it remains exploratory.
'''


configuration = r'''# SINGLE CONFIGURATION BLOCK
# Leave this generated source untouched. For a source-bound pilot, add Colab
# secrets STAGE14_RUN_MODE=pilot and STAGE14_SOURCE_COMMIT=<40-hex commit>.
RUN_MODE = "smoke"
EXPERIMENT_SOURCE_REF = ""
try:
    from google.colab import userdata as _colab_userdata

    RUN_MODE = _colab_userdata.get("STAGE14_RUN_MODE") or RUN_MODE
    EXPERIMENT_SOURCE_REF = (
        _colab_userdata.get("STAGE14_SOURCE_COMMIT") or EXPERIMENT_SOURCE_REF
    )
except Exception:
    pass

MOUNT_DRIVE = True
DOWNLOAD_RESULTS = True
RUN_TEMPORAL_EXTENSION = True
CONTINUE_AFTER_BENCHMARK = True

OUTPUT_DIR = "/content/counterfactual_faithfulness_stage14_pcj"
DRIVE_OUTPUT_DIR = (
    "/content/drive/MyDrive/counterfactual_faithfulness_stage14_pcj"
)

PROTOCOL_ID = "stage14-predictive-control-j-bundle-v1"
NOTEBOOK_PROTOCOL_SHA256 = "__PROTOCOL_DIGEST__"
EVIDENCE_STATUS = "EXPLORATORY_UNTIL_SOURCE_BOUND_BEFORE_DATA"
EXPERIMENT_REPOSITORY = "grewalsk/counterfactual-faithfulness-research"
EXPERIMENT_NOTEBOOK_PATH = "notebooks/14_predictive_control_j_bundle_pilot.ipynb"
EXPERIMENT_BUILDER_PATH = "notebooks/build_stage14_predictive_control_notebook.py"

SEED = 14101
DESIGN_SEED = 14137
MODEL_NAME = "jepa_wm_pusht"
ENVIRONMENT = "PushT"
FRAMESKIP = 5
HORIZONS = [1, 3]
TARGET_STEPS = HORIZONS
PRIMARY_HORIZON = 3
READ_BRANCH = 0
EXPECTED_CARRIER_CHANNELS = 400
CANDIDATE_BLOCKS = [0, 1, 2, 3]  # exclude readout-adjacent blocks 5/6

# Six fixed antithetic future-query families from the Stage 13b action bank.
TRAIN_QUERY_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8)]
TEST_QUERY_PAIRS = [(9, 10), (11, 12)]
TRAIN_ACTION_INDICES = list(range(1, 9))
TEST_ACTION_INDICES = list(range(9, 13))
ACTIONS_PER_STATE = 13
EXPECTED_ACTION_SHA256 = (
    "802129bd281fdd2d42a395429e5a0e00df2dc10032b339ecb8bdc8b2521d9fd2"
)

STATE_CLUSTERS = 16
CONSTRUCTION_CLUSTER_INDICES = list(range(0, STATE_CLUSTERS, 2))
EVALUATION_CLUSTER_INDICES = list(range(1, STATE_CLUSTERS, 2))
STATES_PER_CLUSTER = 2
TASK_ID_OFFSET = 200

CHANNEL_SHRINKAGE = 0.10
CHANNEL_EIGEN_FLOOR = 1e-6
BALANCED_TOLERANCE = 1e-3
JVP_METHOD = "autograd_with_centered_finite_difference_fallback"
JVP_EPSILON = 1e-3
JVP_EPSILON_CHECK = [0.5e-3, 1e-3, 2e-3]

DICTIONARY_SEED = 14201
NULL_ROOT_SEED = 14231
BOOTSTRAP_SEED = 14251
CONTEXT_SKETCH_SEED = 14269
CONTEXT_SKETCH_DIM = 128

if RUN_MODE == "smoke":
    ACTIVE_CONSTRUCTION_CLUSTERS = CONSTRUCTION_CLUSTER_INDICES[:2]
    ACTIVE_EVALUATION_CLUSTERS = EVALUATION_CLUSTER_INDICES[:2]
    ACTIVE_STATES_PER_CLUSTER = 1
    FRAME_ATOMS = 8
    FRAME_SPARSITY = 2
    LOCAL_NEIGHBORS = 2
    NULL_DRAWS = 7
    CAUSAL_NULL_DRAWS = 3
    BOOTSTRAP_DRAWS = 128
elif RUN_MODE == "pilot":
    ACTIVE_CONSTRUCTION_CLUSTERS = CONSTRUCTION_CLUSTER_INDICES
    ACTIVE_EVALUATION_CLUSTERS = EVALUATION_CLUSTER_INDICES
    ACTIVE_STATES_PER_CLUSTER = STATES_PER_CLUSTER
    FRAME_ATOMS = 32
    FRAME_SPARSITY = 4
    LOCAL_NEIGHBORS = 4
    NULL_DRAWS = 256
    CAUSAL_NULL_DRAWS = 32
    BOOTSTRAP_DRAWS = 10_000
else:
    raise ValueError("RUN_MODE must be 'smoke' or 'pilot'")

# Pilot promotion thresholds. Smoke mode can never pass scientifically.
MIN_SCAN_COSINE = 0.10
MIN_HELDOUT_RECONSTRUCTION = 0.20
MIN_GAIN_OVER_NULL_95 = 0.05
MIN_POSITIVE_TASKS = 7
MIN_CAUSAL_MEDIATION = 0.25
MIN_CAUSAL_GAIN_OVER_NULL = 0.10
MIN_CAUSAL_GAIN_OVER_COMPLEMENT = 0.10
MIN_POSITIVE_CONTROL_RECOVERY = 0.10
MAX_POSITIVE_CONTROL_RECOVERY = 2.00
MIN_CAUSAL_DENOMINATOR_COHERENCE = 0.50
MIN_CAUSAL_OUTPUT_RECONSTRUCTION = 0.10
MIN_CAUSAL_OUTPUT_DISPLACEMENT_COSINE = 0.25
MAX_CAUSAL_RECOVERY = 1.50
MAX_PATCH_TO_NATURAL_NORM_RATIO = 1.25
MIN_INTERACTION_STRATUM_TASKS = 4
REQUIRED_INTERACTION_STRATA = ("free", "contact")
MAX_ADJOINT_RELATIVE_ERROR = 1e-3
MAX_ADJOINT_ABS_ERROR = 1e-5
MAX_JVP_RELATIVE_ERROR = 0.25
MAX_BIORTHOGONALITY_ERROR = 1e-4
MIN_TEMPORAL_RECONSTRUCTION = 0.10
CAUSAL_DOSES = [-0.5, 0.5, 1.0]
CALIBRATION_QUANTILE = 0.10
CALIBRATION_MULTIPLIER = 0.50
ABSOLUTE_NONDEGENERACY_FLOOR = 1e-4

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
assert set(CANDIDATE_BLOCKS).issubset(set(range(4)))
assert not set(TRAIN_ACTION_INDICES) & set(TEST_ACTION_INDICES)
assert not set(TRAIN_QUERY_PAIRS) & set(TEST_QUERY_PAIRS)
assert FRAME_SPARSITY <= FRAME_ATOMS
'''

# Freeze an explicit allowlist while building the notebook.  Colab may inject
# unrelated uppercase globals (including dictionaries containing ndarrays), so
# the runtime must never discover protocol configuration by sweeping globals().
configuration_keys = assigned_uppercase_names(configuration)
configuration = (
    configuration.rstrip()
    + "\n\nPROTOCOL_CONFIG_KEYS = "
    + repr(configuration_keys)
    + "\n"
)


installation = r'''import subprocess
import sys

# Preserve Colab's CUDA-matched torch and torchvision.
PINNED = [
    "einops==0.8.1",
    "tensordict==0.9.1",
    "timm==1.0.19",
    "omegaconf==2.3.0",
    "hydra-core==1.3.2",
    "PyYAML==6.0.2",
    "huggingface_hub==0.36.2",
    "hf-xet==1.5.1",
    "gym==0.23.1",
    "pygame==2.6.1",
    "pymunk==6.8.0",
    "opencv-python-headless==4.11.0.86",
    "shapely==2.1.2",
    "lpips==0.1.4",
    "ruamel.yaml==0.18.10",
    "scikit-learn==1.6.1",
]
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", *PINNED],
    check=True,
)
print("Installed pinned non-PyTorch dependencies.")
'''


setup = r'''import csv
import gc
import hashlib
import json
import logging
import math
import os
import platform
import random
import resource
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as torch_functional
import torchvision
import yaml


def ensure_colab_drive():
    from google.colab import drive

    mountpoint = "/content/drive"
    if Path(mountpoint, "MyDrive").is_dir():
        return
    drive.mount(mountpoint, timeout_ms=600_000)
    if not Path(mountpoint, "MyDrive").is_dir():
        raise RuntimeError("Google Drive mount did not produce MyDrive")


if MOUNT_DRIVE:
    ensure_colab_drive()
    OUTPUT_DIR = DRIVE_OUTPUT_DIR

def build_protocol_config(namespace, pinned):
    missing = [key for key in PROTOCOL_CONFIG_KEYS if key not in namespace]
    if missing:
        raise RuntimeError(f"missing frozen protocol configuration: {missing}")
    config = {key: namespace[key] for key in PROTOCOL_CONFIG_KEYS}
    config["PROTOCOL_CONFIG_KEYS"] = list(PROTOCOL_CONFIG_KEYS)
    config["PINNED"] = list(pinned)
    # Fail here with a protocol-specific error if a declared value ever ceases
    # to be JSON-safe.  Ambient notebook globals are deliberately unreachable.
    try:
        json.dumps(config, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "a declared Stage 14 protocol value is not JSON serializable"
        ) from error
    return config


CONFIG = build_protocol_config(globals(), PINNED)
RUN_SIGNATURE = hashlib.sha256(
    json.dumps(CONFIG, sort_keys=True, allow_nan=False).encode()
).hexdigest()
OUT = Path(OUTPUT_DIR) / f"{RUN_MODE}_{RUN_SIGNATURE[:12]}"
ASSET_DIR = OUT / "assets"
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
]:
    directory.mkdir(parents=True, exist_ok=True)

CACHE_ROOT = (
    Path("/content/drive/MyDrive/cf_faithfulness_cache")
    if MOUNT_DRIVE
    else Path("/content/cf_faithfulness_cache")
)
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["TORCH_HOME"] = str(CACHE_ROOT / "torch")
os.environ["HF_HOME"] = str(CACHE_ROOT / "huggingface")
os.environ["MPLCONFIGDIR"] = str(CACHE_ROOT / "matplotlib")
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["JEPAWM_LOGS"] = str(CACHE_ROOT / "unused_jepawm_logs")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
if not torch.cuda.is_available():
    raise RuntimeError("Select Runtime > Change runtime type > GPU")
if tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < (2, 7):
    raise RuntimeError(f"JEPA-WMs requires torch>=2.7; found {torch.__version__}")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "run.log"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)
log = logging.getLogger("stage14_pcj")


def write_json(path, payload):
    temporary = Path(path).with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    temporary = Path(path).with_suffix(".tmp.csv")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_npz(path, **arrays):
    temporary = Path(str(path) + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def sha256_file(path, chunk_bytes=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def download_asset(name):
    specification = ASSET_SPECS[name]
    destination = ASSET_DIR / name
    if destination.exists() and sha256_file(destination) == specification["sha256"]:
        return destination
    destination.unlink(missing_ok=True)
    url = (
        "https://raw.githubusercontent.com/"
        f"{ASSET_REPOSITORY}/{ASSET_COMMIT}/{specification['path']}"
    )
    temporary = destination.with_suffix(destination.suffix + ".part")
    urllib.request.urlretrieve(url, temporary)
    observed = sha256_file(temporary)
    if observed != specification["sha256"]:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"{name} hash mismatch: {observed}")
    temporary.replace(destination)
    return destination


REMOTE_NOTEBOOK_CODE_CELLS = []


def canonical_cell_source(value):
    return str(value).replace("\r\n", "\n").strip()


def source_identity():
    global REMOTE_NOTEBOOK_CODE_CELLS
    payload = {
        "protocol_id": PROTOCOL_ID,
        "notebook_protocol_sha256": NOTEBOOK_PROTOCOL_SHA256,
        "repository": EXPERIMENT_REPOSITORY,
        "source_ref": EXPERIMENT_SOURCE_REF,
        "execution_verified": False,
    }
    if not EXPERIMENT_SOURCE_REF:
        payload["status"] = "UNBOUND_EXPLORATORY_NOTEBOOK"
        payload["confirmation_eligible"] = False
        return payload
    source_ref = EXPERIMENT_SOURCE_REF.lower()
    if len(source_ref) != 40 or any(value not in "0123456789abcdef" for value in source_ref):
        raise RuntimeError("STAGE14_SOURCE_COMMIT must be a full 40-hex commit")
    payload["resolved_commit"] = source_ref
    base = (
        "https://raw.githubusercontent.com/"
        f"{EXPERIMENT_REPOSITORY}/{source_ref}/"
    )
    payload["files"] = {}
    for label, relative in [
        ("notebook", EXPERIMENT_NOTEBOOK_PATH),
        ("builder", EXPERIMENT_BUILDER_PATH),
    ]:
        with urllib.request.urlopen(base + relative) as response:
            content = response.read()
        payload["files"][label] = {
            "path": relative,
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
        }
        if label == "notebook":
            remote_notebook = json.loads(content.decode())
            REMOTE_NOTEBOOK_CODE_CELLS = [
                canonical_cell_source("".join(cell.get("source", [])))
                for cell in remote_notebook["cells"]
                if cell.get("cell_type") == "code"
            ]
            payload["remote_code_cells"] = len(REMOTE_NOTEBOOK_CODE_CELLS)
    payload["status"] = "SOURCE_BOUND_EXECUTION_UNVERIFIED"
    payload["confirmation_eligible"] = False
    return payload


def verify_executed_notebook_through(cell_header):
    if SOURCE_IDENTITY["status"] == "UNBOUND_EXPLORATORY_NOTEBOOK":
        return False
    expected_index = next(
        index
        for index, source in enumerate(REMOTE_NOTEBOOK_CODE_CELLS)
        if source.startswith(cell_header)
    )
    expected = REMOTE_NOTEBOOK_CODE_CELLS[: expected_index + 1]
    shell = get_ipython()
    history = [
        canonical_cell_source(value)
        for value in shell.user_ns.get("_ih", [])[1:]
        if canonical_cell_source(value)
    ]
    matched = len(history) >= len(expected) and history[-len(expected) :] == expected
    prefix_payload = json.dumps(expected, ensure_ascii=False, separators=(",", ":"))
    SOURCE_IDENTITY["execution_verified"] = bool(matched)
    SOURCE_IDENTITY["executed_through_code_cell"] = expected_index
    SOURCE_IDENTITY["executed_through_header"] = cell_header
    SOURCE_IDENTITY["executed_code_cells_verified"] = len(expected)
    SOURCE_IDENTITY["executed_cells_sha256"] = hashlib.sha256(
        prefix_payload.encode()
    ).hexdigest()
    SOURCE_IDENTITY["status"] = (
        "SOURCE_BOUND_EXECUTION_VERIFIED"
        if matched else "SOURCE_BOUND_EXECUTION_MISMATCH"
    )
    SOURCE_IDENTITY["confirmation_eligible"] = bool(matched)
    write_json(OUT / "source_identity.json", SOURCE_IDENTITY)
    if not matched:
        raise RuntimeError(
            "executed code is not the exact committed notebook prefix; "
            "restart and Run all from the source-bound artifact"
        )
    return True


VERSIONS = {
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "numpy": np.__version__,
    "cuda_runtime": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "gpu_total_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
}
SOURCE_IDENTITY = source_identity()
write_json(OUT / "config.json", {**CONFIG, "run_signature": RUN_SIGNATURE})
write_json(OUT / "versions.json", VERSIONS)
write_json(OUT / "source_identity.json", SOURCE_IDENTITY)
(OUT / "FAILURE_TRACE.txt").write_text("PENDING\n")

TIMINGS = {}
MEMORY = []
PIPELINE_FAILED = False
FAILURE_MESSAGE = ""


def record_failure(stage):
    global PIPELINE_FAILED, FAILURE_MESSAGE
    PIPELINE_FAILED = True
    FAILURE_MESSAGE = f"STAGE: {stage}\n{traceback.format_exc()}"
    (OUT / "FAILURE_TRACE.txt").write_text(FAILURE_MESSAGE)
    log.exception("Captured failure in %s", stage)


def memory_report(stage):
    maximum_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    row = {
        "stage": stage,
        "cpu_peak_gib": float(maximum_rss * 1024 / 2**30),
        "gpu_allocated_gib": float(torch.cuda.memory_allocated() / 2**30),
        "gpu_reserved_gib": float(torch.cuda.memory_reserved() / 2**30),
        "gpu_peak_allocated_gib": float(torch.cuda.max_memory_allocated() / 2**30),
    }
    MEMORY.append(row)
    write_json(OUT / "memory.json", MEMORY)
    return row


for asset_name in ASSET_SPECS:
    download_asset(asset_name)
print(json.dumps(VERSIONS, indent=2))
print(json.dumps(SOURCE_IDENTITY, indent=2))
print(f"Durable run directory: {OUT}")
memory_report("startup")
'''


analysis_helpers = (
    geometry_helpers
    + "\n\n\n"
    + numerical_helpers
    + r'''


class CountSketchProjector:
    """Deterministic norm-stabilized projection for context/decoder features."""

    def __init__(self, input_dim, output_dim, seed, device="cuda"):
        rng = np.random.default_rng(int(seed))
        bucket = rng.integers(0, output_dim, size=input_dim, dtype=np.int64)
        sign = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), input_dim)
        counts = np.bincount(bucket, minlength=output_dim).astype(np.float32)
        counts[counts == 0] = 1.0
        self.bucket = torch.as_tensor(bucket, device=device, dtype=torch.long)
        self.sign = torch.as_tensor(sign, device=device, dtype=torch.float32)
        self.scale = torch.as_tensor(np.sqrt(counts), device=device)
        self.output_dim = int(output_dim)

    def __call__(self, values):
        values = values.float().flatten(1)
        output = torch.zeros(
            values.shape[0], self.output_dim, device=values.device
        )
        output.scatter_add_(
            1,
            self.bucket[None].expand(values.shape[0], -1),
            values * self.sign[None],
        )
        return output / self.scale[None]


def task_equal_rows(rows, value_key):
    by_task = defaultdict(list)
    for row in rows:
        value = float(row[value_key])
        if np.isfinite(value):
            by_task[int(row["task_id"])].append(value)
    return {
        task_id: float(np.mean(values))
        for task_id, values in sorted(by_task.items())
        if values
    }


def summarize_task_metric(rows, value_key, bootstrap_seed):
    task_means = task_equal_rows(rows, value_key)
    if not task_means:
        return {
            "task_means": {},
            "task_equal_mean": None,
            "task_equal_median": None,
            "one_sided_95_lower": None,
            "sign_test": {"positive": 0, "nonzero": 0, "one_sided_p": 1.0},
            "hierarchical_bootstrap_95_interval": [None, None],
        }
    state_values = np.asarray([float(row[value_key]) for row in rows])
    task_ids = np.asarray([int(row["task_id"]) for row in rows])
    finite = np.isfinite(state_values)
    bootstrap = hierarchical_bootstrap_means(
        state_values[finite],
        task_ids[finite],
        draws=BOOTSTRAP_DRAWS,
        seed=bootstrap_seed,
    )
    values = np.asarray(list(task_means.values()), dtype=np.float64)
    lower = one_sided_t_lower(values)
    return {
        "task_means": {str(key): value for key, value in task_means.items()},
        "task_equal_mean": float(np.mean(values)) if len(values) else None,
        "task_equal_median": float(np.median(values)) if len(values) else None,
        "one_sided_95_lower": float(lower) if np.isfinite(lower) else None,
        "sign_test": exact_positive_sign_test(values),
        "hierarchical_bootstrap_95_interval": [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ],
    }


def fit_sparse_frame(mode_rows):
    from sklearn.decomposition import DictionaryLearning

    span = training_span(mode_rows)
    coordinates = span["coordinates"]
    atoms = min(FRAME_ATOMS, max(2, len(coordinates)))
    sparsity = min(FRAME_SPARSITY, atoms)
    learner = DictionaryLearning(
        n_components=atoms,
        alpha=0.10,
        max_iter=600,
        fit_algorithm="lars",
        transform_algorithm="omp",
        transform_n_nonzero_coefs=sparsity,
        random_state=DICTIONARY_SEED,
        positive_code=False,
        positive_dict=False,
    )
    learner.fit(coordinates)
    components = np.asarray(learner.components_, dtype=np.float64)
    norms = np.linalg.norm(components, axis=1)
    components /= np.maximum(norms[:, None], 1e-12)
    codes = omp_codes(coordinates, components, sparsity)
    hidden_atoms = components @ span["basis"].T
    hidden_atoms /= np.maximum(
        np.linalg.norm(hidden_atoms, axis=1, keepdims=True), 1e-12
    )
    return {
        "basis": span["basis"],
        "components": components,
        "hidden_atoms": hidden_atoms,
        "codes": codes,
        "eigenvalues": span["eigenvalues"],
        "atoms": atoms,
        "sparsity": sparsity,
    }


def frame_reconstruct(values, frame, components=None):
    vectors = np.asarray(values, dtype=np.float64)
    if vectors.ndim == 1:
        vectors = vectors[None]
    basis = np.asarray(frame["basis"], dtype=np.float64)
    dictionary = np.asarray(
        frame["components"] if components is None else components,
        dtype=np.float64,
    )
    coordinates = vectors @ basis
    codes, reconstructed_coordinates = sparse_reconstruct(
        coordinates, dictionary, frame["sparsity"]
    )
    reconstruction = reconstructed_coordinates @ basis.T
    return codes, reconstruction


def null_components(frame, seed):
    components = np.asarray(frame["components"], dtype=np.float64)
    rotation = haar_rotation(components.shape[1], seed)
    output = components @ rotation
    output /= np.maximum(np.linalg.norm(output, axis=1, keepdims=True), 1e-12)
    return output


def orthonormal_row_basis(rows, tolerance=1e-9):
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or not len(values):
        return np.empty((0, values.shape[-1]))
    _, singular, vh = np.linalg.svd(values, full_matrices=False)
    keep = singular > max(float(singular[0]) * tolerance, 1e-12)
    return vh[keep]


def project_rows(values, basis_rows):
    values = np.asarray(values, dtype=np.float64)
    basis = np.asarray(basis_rows, dtype=np.float64)
    if not len(basis):
        return np.zeros_like(values)
    return (values @ basis.T) @ basis


def stable_cosine(left, right):
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / max(denominator, 1e-12))


def normalized_action_directions(actions, action_indices):
    values = actions.detach().float().cpu().numpy()
    base = values[:, READ_BRANCH].reshape(-1)
    columns = []
    for action_index in action_indices:
        direction = values[:, action_index].reshape(-1) - base
        norm = np.linalg.norm(direction)
        if norm <= 1e-12:
            raise RuntimeError(f"zero action direction for action {action_index}")
        columns.append(direction / norm)
    return np.stack(columns, axis=1)


def manifest_rows(root, excluded_roots=()):
    excluded = {Path(value).resolve() for value in excluded_roots}
    rows = []
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file() or any(parent in path.resolve().parents for parent in excluded):
            continue
        if path.name.endswith(".part") or path.name.endswith(".tmp"):
            continue
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


# Pure numerical identities before any model/data access.
_rng = np.random.default_rng(17)
_g = _rng.normal(size=(5, 20))
_b = _rng.normal(size=(20, 6))
_balanced = balanced_modes(_g, _b)
if _balanced["biorthogonality_error"] > 1e-9:
    raise AssertionError("balanced-mode CPU identity failed")
_rows, _ = canonical_mode_rows(_balanced["primal"], _balanced["dual"])
_span = training_span(_rows)
if relative_error(_span["coordinates"] @ _span["basis"].T, _rows) > 1e-9:
    raise AssertionError("training-span CPU identity failed")
print("Pure numerical self-tests passed.")
'''
)


model_helpers = (
    simulator_helpers
    + r'''


def verify_pretrained_assets():
    rows = []
    for name, expected in EXPECTED_PRETRAINED_ASSET_SHA256.items():
        matching = [
            path for path in CACHE_ROOT.rglob(name)
            if sha256_file(path) == expected
        ]
        if not matching:
            raise RuntimeError(f"verified pretrained asset not found: {name}")
        rows.append({"name": name, "path": str(matching[0]), "sha256": expected})
    write_json(OUT / "pretrained_asset_verification.json", rows)
    return rows


def validate_jepa_predictor(model):
    predictor = model.model.predictor
    blocks = list(getattr(predictor, "predictor_blocks", []))
    if predictor.__class__.__name__ != "VisionTransformerAdaLN":
        raise RuntimeError(f"unexpected predictor class {predictor.__class__.__name__}")
    if len(blocks) != 6:
        raise RuntimeError(f"expected six predictor blocks, found {len(blocks)}")
    if not bool(getattr(predictor, "action_encoder_inpred", False)):
        raise RuntimeError("predictor does not internally encode actions")
    if bool(getattr(predictor, "use_activation_checkpointing", False)):
        raise RuntimeError("activation checkpointing must be disabled for hooks/JVPs")
    if int(getattr(model, "ctxt_window", -1)) != 2:
        raise RuntimeError(f"expected context window 2, found {model.ctxt_window}")
    return predictor, blocks


def load_frozen_model():
    model, preprocessor = torch.hub.load(
        str(REPO),
        MODEL_NAME,
        source="local",
        pretrained=True,
        device="cuda:0",
        trust_repo=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    predictor, blocks = validate_jepa_predictor(model)
    verify_pretrained_assets()
    return model, preprocessor, predictor, blocks


def model_action_tensor(preprocessor, selected_actions, horizon):
    selected_actions = np.asarray(selected_actions, dtype=np.float32)
    chunks = torch.from_numpy(
        selected_actions[:, : horizon * FRAMESKIP].reshape(
            ACTIONS_PER_STATE, horizon, FRAMESKIP, 2
        )
    ).float()
    normalized = preprocessor.normalize_actions(chunks)
    return (
        normalized.reshape(ACTIONS_PER_STATE, horizon, -1)
        .permute(1, 0, 2)
        .contiguous()
        .cuda()
    )


def layer_tokens_full(capture):
    if capture.ndim != 3 or capture.shape[1] % 256:
        raise ValueError(f"unexpected block output {tuple(capture.shape)}")
    if capture.shape[-1] != EXPECTED_CARRIER_CHANNELS:
        raise ValueError(
            f"expected carrier width {EXPECTED_CARRIER_CHANNELS}, "
            f"found {capture.shape[-1]}"
        )
    return capture.view(
        capture.shape[0], capture.shape[1] // 256, 256, capture.shape[-1]
    )[:, -1]


def forward_with_carriers(
    initial,
    actions,
    horizon,
    capture_blocks=(),
    intervention=None,
    require_grad=False,
):
    captures = {int(block): [] for block in capture_blocks}
    context = {"step": -1}
    handles = []
    for block_index in capture_blocks:
        def hook(_module, _inputs, output, block_index=int(block_index)):
            # With frozen parameters and ordinary inputs, autograd would not
            # create a graph. Anchor the requested carrier as a leaf so the
            # suffix VJP is exact without retaining the upstream encoder.
            if require_grad and not output.requires_grad:
                output = output.detach().requires_grad_(True)
            captures[block_index].append(output)
            if (
                intervention is None
                or block_index != int(intervention["block"])
                or context["step"] != horizon - 1
            ):
                return output
            value = output.clone()
            view = value.view(
                value.shape[0], value.shape[1] // 256, 256, value.shape[-1]
            )
            replacement = intervention.get("replacement")
            if replacement is not None:
                view[:, -1] = replacement.to(view.device, view.dtype)
            else:
                delta = intervention["delta"].to(view.device, view.dtype)
                view[:, -1] = view[:, -1] + delta
            return view.reshape_as(value)

        handles.append(PREDICTOR_BLOCKS[block_index].register_forward_hook(hook))

    try:
        batch = actions.shape[1]
        action_batch = actions[:horizon].permute(1, 0, 2).contiguous()
        with torch.set_grad_enabled(require_grad):
            action_features = MODEL.model.encode_act(action_batch)
            if action_features.shape[-1] != 10:
                raise RuntimeError(
                    f"expected encoded action width 10, found {action_features.shape[-1]}"
                )
            visual_history = initial["visual"].expand(
                batch, *initial["visual"].shape[1:]
            ).detach().clone()
            proprio_history = initial["proprio"].expand(
                batch, *initial["proprio"].shape[1:]
            ).detach().clone()
            predicted_tokens = None
            predicted_proprio = None
            for step_index in range(horizon):
                context["step"] = step_index
                predicted_visual, _, predicted_proprio = MODEL.model.forward_pred(
                    visual_history[:, -MODEL.ctxt_window :],
                    action_features[:, : step_index + 1][:, -MODEL.ctxt_window :],
                    proprio_history[:, -MODEL.ctxt_window :],
                )
                next_visual = predicted_visual[:, -1:]
                next_proprio = predicted_proprio[:, -1:]
                predicted_tokens = next_visual[:, 0, 0].flatten(1, 2)
                if predicted_tokens.shape[1:] != (256, 384):
                    raise RuntimeError(
                        f"expected visual grid [256,384], found {predicted_tokens.shape[1:]}"
                    )
                visual_history = torch.cat([visual_history, next_visual], dim=1)
                proprio_history = torch.cat([proprio_history, next_proprio], dim=1)
        final_captures = {
            block: captures[block][-1] for block in capture_blocks
        }
        return predicted_tokens, predicted_proprio[:, -1], final_captures
    finally:
        for handle in handles:
            handle.remove()


def state_model_inputs(record_id, horizon):
    with np.load(TRUTH_DIR / f"state_{record_id:04d}.npz") as truth:
        initial_visual = truth["initial_visual"]
        initial_proprio = truth["initial_proprio"]
        selected_actions = truth["selected_actions"]
    with torch.inference_mode():
        initial = MODEL.encode(to_model_observation(initial_visual, initial_proprio))
    initial = {name: value.detach() for name, value in initial.items()}
    actions = model_action_tensor(PREPROCESSOR, selected_actions, horizon)
    return initial, actions


def load_target_tokens(record_id):
    with np.load(TARGET_DIR / f"state_{record_id:04d}.npz") as payload:
        return payload["true_tokens"].astype(np.float32)


def query_separation(targets, horizon, pair):
    horizon_index = HORIZONS.index(horizon)
    direction = targets[pair[0], horizon_index] - targets[pair[1], horizon_index]
    norm = np.linalg.norm(direction)
    if norm <= 1e-12:
        raise RuntimeError(f"degenerate target query {pair}")
    return direction, float(norm)


def query_direction(targets, horizon, pair):
    direction, separation = query_separation(targets, horizon, pair)
    return direction / separation


def scan_state_carrier(record_id, horizon, blocks, query_pairs, action_indices):
    initial, actions = state_model_inputs(record_id, horizon)
    targets = load_target_tokens(record_id)
    predictions, _, captures = forward_with_carriers(
        initial, actions, horizon, capture_blocks=blocks, require_grad=True
    )
    if not predictions.requires_grad or not all(
        captures[block][-1].requires_grad for block in blocks
    ):
        raise RuntimeError("suffix VJP graph anchor is absent")
    capture_values = [captures[block] for block in blocks]
    gradients = {block: [] for block in blocks}
    direct = []
    for query_index, pair in enumerate(query_pairs):
        direction_numpy = query_direction(targets, horizon, pair)
        direction = torch.as_tensor(
            direction_numpy, device=predictions.device, dtype=predictions.dtype
        )
        scalar = torch.sum(predictions[READ_BRANCH] * direction)
        values = torch.autograd.grad(
            scalar,
            capture_values,
            retain_graph=query_index + 1 < len(query_pairs),
            allow_unused=False,
            create_graph=False,
        )
        for block, value in zip(blocks, values):
            gradients[block].append(
                layer_tokens_full(value)[READ_BRANCH].detach().float().cpu().numpy()
            )
        direct.append(
            [
                float(
                    torch.sum(
                        (predictions[action_index] - predictions[READ_BRANCH])
                        * direction
                    ).detach().cpu()
                )
                for action_index in action_indices
            ]
        )
    writes = {}
    moments = {}
    for block in blocks:
        activation = layer_tokens_full(captures[block]).detach().float()
        if action_indices:
            writes[block] = torch.stack(
                [
                    activation[action_index] - activation[READ_BRANCH]
                    for action_index in action_indices
                ]
            ).cpu().numpy()
        else:
            writes[block] = np.empty(
                (0, 256, EXPECTED_CARRIER_CHANNELS), dtype=np.float32
            )
        construction_branches = [READ_BRANCH, *action_indices]
        channels = activation[construction_branches].reshape(
            -1, activation.shape[-1]
        ).double()
        moments[block] = {
            "count": int(len(channels)),
            "sum": channels.sum(dim=0).cpu().numpy(),
            "cross": (channels.T @ channels).cpu().numpy(),
        }
    with torch.inference_mode():
        context_projector = CountSketchProjector(
            initial["visual"].numel(),
            CONTEXT_SKETCH_DIM,
            CONTEXT_SKETCH_SEED,
        )
        context = context_projector(initial["visual"].reshape(1, -1))[0]
        context = context / torch.clamp(torch.linalg.vector_norm(context), min=1e-12)
    payload = {
        "gradients": gradients,
        "writes": writes,
        "direct": np.asarray(direct, dtype=np.float64),
        "moments": moments,
        "context": context.detach().cpu().numpy(),
    }
    del predictions, captures, capture_values, initial, actions
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def exact_action_jacobian(record_id, horizon, block):
    initial, all_actions = state_model_inputs(record_id, horizon)
    base = all_actions[:, READ_BRANCH : READ_BRANCH + 1].detach()
    shape = tuple(base.shape)

    def capture_function(flat_action):
        action = flat_action.reshape(shape)
        _, _, captures = forward_with_carriers(
            initial,
            action,
            horizon,
            capture_blocks=[block],
            require_grad=True,
        )
        return layer_tokens_full(captures[block])[0].reshape(-1)

    flat = base.reshape(-1).detach().requires_grad_(True)
    columns = []
    used_fallback = False
    for coordinate in range(flat.numel()):
        tangent = torch.zeros_like(flat)
        tangent[coordinate] = 1.0
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
    matrix = torch.stack(columns, dim=1).numpy()
    return matrix, used_fallback


def physical_pose_decoder():
    payload = torch.load(
        ASSET_DIR / "physical_decoders.pt", map_location="cpu", weights_only=False
    )
    projectors = {}

    def decode(tokens):
        outputs = []
        for decoder in payload["decoders"]:
            seed = int(decoder["projection_seed"])
            if seed not in projectors:
                projectors[seed] = CountSketchProjector(
                    tokens.shape[-2] * tokens.shape[-1],
                    int(payload["projection_dim"]),
                    seed,
                )
            features = projectors[seed](tokens)
            mean = torch.as_tensor(
                decoder["mean"], device=tokens.device, dtype=torch.float32
            )
            scale = torch.as_tensor(
                decoder["scale"], device=tokens.device, dtype=torch.float32
            )
            coefficient = torch.as_tensor(
                decoder["coefficient"], device=tokens.device, dtype=torch.float32
            )
            intercept = torch.as_tensor(
                decoder["intercept"], device=tokens.device, dtype=torch.float32
            )
            outputs.append(intercept + ((features - mean) / scale) @ coefficient)
        return torch.stack(outputs).mean(dim=0)

    return decode
'''
)


design = r'''# Freeze state clusters, action/query families, and all random schedules.
ACTION_LABELS, ACTION_BANK = frozen_action_bank(
    primitive_steps=max(HORIZONS) * FRAMESKIP
)
ACTION_HASH = array_sha256(ACTION_BANK)
if ACTION_HASH != EXPECTED_ACTION_SHA256:
    raise AssertionError(
        f"action design changed: {ACTION_HASH} != {EXPECTED_ACTION_SHA256}"
    )


def cluster_anchors():
    anchors = []
    for index in range(STATE_CLUSTERS):
        angle = 0.13 + 2.0 * np.pi * index / STATE_CLUSTERS
        xy = np.asarray([256.0, 256.0]) + 58.0 * np.asarray(
            [np.cos(angle), np.sin(angle)]
        )
        orientation = float(((1.7 * angle + np.pi) % (2 * np.pi)) - np.pi)
        anchors.append(
            {
                "cluster_index": index,
                "task_id": TASK_ID_OFFSET + index,
                "anchor_xy": xy,
                "orientation": orientation,
                "split": (
                    "construction"
                    if index in CONSTRUCTION_CLUSTER_INDICES
                    else "evaluation"
                ),
            }
        )
    return anchors


def make_state_records():
    records = []
    record_id = 0
    for anchor in cluster_anchors():
        for within_cluster in range(STATES_PER_CLUSTER):
            rng = np.random.default_rng(
                DESIGN_SEED + anchor["cluster_index"] * 1000 + within_cluster
            )
            center = anchor["anchor_xy"]
            for _ in range(2000):
                radial = rng.uniform(82.0, 116.0)
                polar = rng.uniform(-np.pi, np.pi)
                block = center + radial * np.asarray(
                    [np.cos(polar), np.sin(polar)]
                )
                direction = (center - block) / max(
                    np.linalg.norm(center - block), 1e-12
                )
                agent = block - rng.uniform(58.0, 78.0) * direction
                if (
                    np.all(block > 90.0)
                    and np.all(block < 422.0)
                    and np.all(agent > 35.0)
                    and np.all(agent < 477.0)
                ):
                    break
            else:
                raise RuntimeError("could not construct a valid PushT state")
            state = np.asarray(
                [
                    agent[0], agent[1], block[0], block[1],
                    rng.uniform(-0.75, 0.75), 0.0, 0.0,
                ],
                dtype=np.float64,
            )
            records.append(
                {
                    "record_id": record_id,
                    "task_id": anchor["task_id"],
                    "cluster_index": anchor["cluster_index"],
                    "within_cluster": within_cluster,
                    "split": anchor["split"],
                    "evaluation_seed": int(
                        DESIGN_SEED
                        + anchor["cluster_index"] * 1000
                        + within_cluster
                    ),
                    "state": state,
                    "anchor_xy": anchor["anchor_xy"],
                    "orientation": anchor["orientation"],
                }
            )
            record_id += 1
    return records


ALL_RECORDS = make_state_records()
ACTIVE_RECORDS = [
    row
    for row in ALL_RECORDS
    if (
        row["cluster_index"] in ACTIVE_CONSTRUCTION_CLUSTERS
        or row["cluster_index"] in ACTIVE_EVALUATION_CLUSTERS
    )
    and row["within_cluster"] < ACTIVE_STATES_PER_CLUSTER
]
CONSTRUCTION_RECORDS = [row for row in ACTIVE_RECORDS if row["split"] == "construction"]
EVALUATION_RECORDS = [row for row in ACTIVE_RECORDS if row["split"] == "evaluation"]
RECORD_BY_ID = {int(row["record_id"]): row for row in ACTIVE_RECORDS}

if {row["task_id"] for row in CONSTRUCTION_RECORDS} & {
    row["task_id"] for row in EVALUATION_RECORDS
}:
    raise AssertionError("construction/evaluation task overlap")
state_hashes = [array_sha256(row["state"]) for row in ALL_RECORDS]
if len(state_hashes) != len(set(state_hashes)):
    raise AssertionError("duplicate Stage 14 state")

null_rng = np.random.default_rng(NULL_ROOT_SEED)
NULL_SEEDS = null_rng.integers(
    0, np.iinfo(np.uint32).max, size=max(NULL_DRAWS, CAUSAL_NULL_DRAWS), dtype=np.uint32
)
np.savez_compressed(
    DESIGN_DIR / "stage14_design.npz",
    action_bank=ACTION_BANK,
    action_labels=np.asarray(ACTION_LABELS),
    states=np.stack([row["state"] for row in ALL_RECORDS]),
    record_ids=np.asarray([row["record_id"] for row in ALL_RECORDS]),
    task_ids=np.asarray([row["task_id"] for row in ALL_RECORDS]),
    cluster_indices=np.asarray([row["cluster_index"] for row in ALL_RECORDS]),
    split=np.asarray([row["split"] for row in ALL_RECORDS]),
    null_seeds=NULL_SEEDS,
)
manifest_payload = {
    "records": [
        {
            **{key: value for key, value in row.items() if key not in {"state", "anchor_xy"}},
            "state": row["state"].tolist(),
            "anchor_xy": row["anchor_xy"].tolist(),
            "state_sha256": array_sha256(row["state"]),
        }
        for row in ALL_RECORDS
    ],
    "active_construction_record_ids": [row["record_id"] for row in CONSTRUCTION_RECORDS],
    "active_evaluation_record_ids": [row["record_id"] for row in EVALUATION_RECORDS],
    "train_query_pairs": TRAIN_QUERY_PAIRS,
    "test_query_pairs": TEST_QUERY_PAIRS,
    "train_action_indices": TRAIN_ACTION_INDICES,
    "test_action_indices": TEST_ACTION_INDICES,
}
write_json(DESIGN_DIR / "task_state_manifest.json", manifest_payload)

freeze = {
    "created_before_truth_or_target_generation": True,
    "protocol_id": PROTOCOL_ID,
    "run_signature": RUN_SIGNATURE,
    "source_identity": SOURCE_IDENTITY,
    "action_sha256": ACTION_HASH,
    "design_sha256": sha256_file(DESIGN_DIR / "stage14_design.npz"),
    "manifest_sha256": sha256_file(DESIGN_DIR / "task_state_manifest.json"),
    "construction_task_ids": sorted({row["task_id"] for row in CONSTRUCTION_RECORDS}),
    "evaluation_task_ids": sorted({row["task_id"] for row in EVALUATION_RECORDS}),
    "stage13b_states_confirmation_eligible": False,
}
freeze_path = DESIGN_DIR / "design_freeze.json"
if freeze_path.exists():
    existing = json.loads(freeze_path.read_text())
    if existing != freeze:
        raise RuntimeError("existing design freeze differs from this run")
else:
    write_json(freeze_path, freeze)
write_json(
    DESIGN_DIR / "design_freeze_certificate.json",
    {"sha256": sha256_file(freeze_path)},
)
print(json.dumps(freeze, indent=2))
'''


truth_generation = r'''# Regenerate only the new frozen PushT branches; save contact metadata.


def record_task(record):
    return {
        "environment": ENVIRONMENT,
        "task_id": int(record["task_id"]),
        "goal": [
            float(record["anchor_xy"][0]),
            float(record["anchor_xy"][1]),
            float(record["orientation"]),
        ],
    }


def generate_truth(records):
    started = time.perf_counter()
    for record_index, record in enumerate(records):
        record_id = int(record["record_id"])
        destination = TRUTH_DIR / f"state_{record_id:04d}.npz"
        if destination.exists():
            continue
        initials = []
        initial_proprios = []
        future_visual = []
        future_proprio = []
        endpoint_states = []
        interaction_counts = []
        interaction_types = []
        task = record_task(record)
        for action in ACTION_BANK:
            initial, _, observations, states, counts, kinds = rollout_branch(
                REPO,
                ENVIRONMENT,
                task,
                record["state"],
                action,
                int(record["evaluation_seed"]),
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
            record_id=np.asarray(record_id, dtype=np.int64),
            task_id=np.asarray(record["task_id"], dtype=np.int64),
            split=np.asarray(record["split"]),
            initial_state=np.asarray(record["state"], dtype=np.float64),
            initial_visual=np.asarray(initials[0], dtype=np.uint8),
            initial_proprio=np.asarray(initial_proprios[0], dtype=np.float32),
            selected_actions=np.asarray(ACTION_BANK, dtype=np.float32),
            future_visual=np.asarray(future_visual, dtype=np.uint8),
            future_proprio=np.asarray(future_proprio, dtype=np.float32),
            endpoint_states=np.asarray(endpoint_states, dtype=np.float32),
            interaction_counts=np.asarray(interaction_counts, dtype=np.int32),
            interaction_types=np.asarray(interaction_types),
        )
        write_json(
            OUT / "truth_progress.json",
            {
                "completed": record_index + 1,
                "total": len(records),
                "last_record_id": record_id,
            },
        )
    TIMINGS["truth_seconds"] = time.perf_counter() - started


def goal_binding_smoke():
    first = CONSTRUCTION_RECORDS[0]
    second = next(
        row
        for row in CONSTRUCTION_RECORDS
        if row["task_id"] != first["task_id"]
    )
    goals = []
    visuals = []
    for record in [first, second]:
        environment, payload, _ = reset_environment(
            REPO,
            ENVIRONMENT,
            record_task(record),
            first["state"],
            int(first["evaluation_seed"]),
        )
        goals.append(np.asarray(environment.goal_pose, dtype=np.float64))
        visuals.append(np.asarray(payload["visual"]))
    result = {
        "requested_goals": [record_task(row)["goal"] for row in [first, second]],
        "observed_goals": [value.tolist() for value in goals],
        "goals_differ": bool(not np.array_equal(goals[0], goals[1])),
        "renders_differ": bool(not np.array_equal(visuals[0], visuals[1])),
        "passed": bool(
            not np.array_equal(goals[0], goals[1])
            and not np.array_equal(visuals[0], visuals[1])
        ),
    }
    if not result["passed"]:
        raise RuntimeError(f"PushT goal binding failed: {result}")
    write_json(OUT / "goal_binding_smoke.json", result)
    return result


if not PIPELINE_FAILED:
    try:
        REPO = configure_repo()
        GOAL_BINDING = goal_binding_smoke()
        smoke_record = CONSTRUCTION_RECORDS[0]
        restore = exact_restore_test(
            REPO,
            ENVIRONMENT,
            record_task(smoke_record),
            smoke_record["state"],
            ACTION_BANK[1],
        )
        write_json(OUT / "restore_test.json", restore)
        # Evaluation design is frozen above, but no evaluation outcome is opened yet.
        generate_truth(CONSTRUCTION_RECORDS)
        memory_report("truth_complete")
    except Exception:
        record_failure("truth_generation")
'''


model_and_targets = r'''# Load the frozen model, encode float32 targets, and verify hook identity.


def encode_target_cache(records):
    started = time.perf_counter()
    for index, record in enumerate(records):
        record_id = int(record["record_id"])
        destination = TARGET_DIR / f"state_{record_id:04d}.npz"
        if destination.exists():
            continue
        with np.load(TRUTH_DIR / f"state_{record_id:04d}.npz") as truth:
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
            raise RuntimeError(f"unexpected target shape {tokens.shape}")
        atomic_npz(destination, true_tokens=tokens.astype(np.float32))
        write_json(
            OUT / "target_progress.json",
            {"completed": index + 1, "total": len(records), "last_record_id": record_id},
        )
    TIMINGS["target_encoding_seconds"] = time.perf_counter() - started


def hook_identity_test(record_id):
    initial, actions = state_model_inputs(record_id, PRIMARY_HORIZON)
    block = CANDIDATE_BLOCKS[0]
    with torch.inference_mode():
        baseline, _, captures = forward_with_carriers(
            initial, actions[:, :1], PRIMARY_HORIZON, capture_blocks=[block]
        )
        zero = torch.zeros_like(layer_tokens_full(captures[block]))
        hooked, _, _ = forward_with_carriers(
            initial,
            actions[:, :1],
            PRIMARY_HORIZON,
            capture_blocks=[block],
            intervention={"block": block, "delta": zero},
        )
        after, _, _ = forward_with_carriers(
            initial, actions[:, :1], PRIMARY_HORIZON, capture_blocks=[]
        )
    zero_error = float(torch.max(torch.abs(baseline - hooked)).cpu())
    removal_error = float(torch.max(torch.abs(baseline - after)).cpu())
    result = {
        "zero_dose_max_abs_error": zero_error,
        "hook_removal_max_abs_error": removal_error,
        "carrier_shape": list(layer_tokens_full(captures[block]).shape),
        "passed": bool(zero_error <= 1e-6 and removal_error <= 1e-6),
    }
    if not result["passed"]:
        raise RuntimeError(f"hook identity failed: {result}")
    write_json(OUT / "hook_identity.json", result)
    return result


if not PIPELINE_FAILED:
    try:
        MODEL, PREPROCESSOR, PREDICTOR, PREDICTOR_BLOCKS = load_frozen_model()
        encode_target_cache(CONSTRUCTION_RECORDS)
        HOOK_IDENTITY = hook_identity_test(CONSTRUCTION_RECORDS[0]["record_id"])
        DECODE_PHYSICAL_POSE = physical_pose_decoder()
        memory_report("model_targets_hooks_ready")
    except Exception:
        record_failure("model_and_targets")
'''


carrier_scan = r'''# Construction-only carrier scan; select an early reusable layer.


def scan_path(record_id):
    return SCAN_DIR / f"state_{record_id:04d}.npz"


def save_scan(record, payload):
    block_count = len(CANDIDATE_BLOCKS)
    gradients = np.stack(
        [np.stack(payload["gradients"][block]) for block in CANDIDATE_BLOCKS]
    )
    writes = np.stack([payload["writes"][block] for block in CANDIDATE_BLOCKS])
    moment_count = np.asarray(
        [payload["moments"][block]["count"] for block in CANDIDATE_BLOCKS],
        dtype=np.int64,
    )
    moment_sum = np.stack(
        [payload["moments"][block]["sum"] for block in CANDIDATE_BLOCKS]
    )
    moment_cross = np.stack(
        [payload["moments"][block]["cross"] for block in CANDIDATE_BLOCKS]
    )
    if gradients.shape[:2] != (block_count, len(TRAIN_QUERY_PAIRS)):
        raise RuntimeError(f"unexpected scan gradient shape {gradients.shape}")
    atomic_npz(
        scan_path(record["record_id"]),
        record_id=np.asarray(record["record_id"], dtype=np.int64),
        task_id=np.asarray(record["task_id"], dtype=np.int64),
        gradients=gradients.astype(np.float32),
        gradient_norms=np.linalg.norm(
            gradients.reshape(block_count, len(TRAIN_QUERY_PAIRS), -1), axis=2
        ).astype(np.float32),
        writes=writes.astype(np.float32),
        write_norms=np.linalg.norm(
            writes.reshape(block_count, len(TRAIN_ACTION_INDICES), -1), axis=2
        ).astype(np.float32),
        direct=payload["direct"].astype(np.float64),
        moment_count=moment_count,
        moment_sum=moment_sum.astype(np.float64),
        moment_cross=moment_cross.astype(np.float64),
        context=payload["context"].astype(np.float32),
    )


def run_carrier_scan():
    started = time.perf_counter()
    for index, record in enumerate(CONSTRUCTION_RECORDS):
        if not scan_path(record["record_id"]).exists():
            payload = scan_state_carrier(
                record["record_id"],
                PRIMARY_HORIZON,
                CANDIDATE_BLOCKS,
                TRAIN_QUERY_PAIRS,
                TRAIN_ACTION_INDICES,
            )
            save_scan(record, payload)
        write_json(
            OUT / "carrier_scan_progress.json",
            {
                "completed": index + 1,
                "total": len(CONSTRUCTION_RECORDS),
                "last_record_id": record["record_id"],
            },
        )
    TIMINGS["carrier_scan_seconds"] = time.perf_counter() - started


def select_carrier_layer():
    state_rows = []
    for record in CONSTRUCTION_RECORDS:
        with np.load(scan_path(record["record_id"])) as payload:
            gradients = payload["gradients"].astype(np.float64)
            writes = payload["writes"].astype(np.float64)
            direct = payload["direct"].astype(np.float64)
        for layer_index, block in enumerate(CANDIDATE_BLOCKS):
            g = gradients[layer_index].reshape(len(TRAIN_QUERY_PAIRS), -1)
            w = writes[layer_index].reshape(len(TRAIN_ACTION_INDICES), -1)
            local = g @ w.T
            metrics = transfer_metrics(direct, local)
            state_rows.append(
                {
                    "record_id": int(record["record_id"]),
                    "task_id": int(record["task_id"]),
                    "block": int(block),
                    "reconstruction": metrics["reconstruction"],
                    "cosine": metrics["cosine"],
                    "scale": metrics["scale"],
                    "direct_energy": metrics["energy"],
                }
            )
    write_csv(ANALYSIS_DIR / "carrier_scan_state_rows.csv", state_rows)
    task_ids = sorted({row["task_id"] for row in state_rows})
    task_by_layer = np.full(
        (len(task_ids), len(CANDIDATE_BLOCKS)), np.nan, dtype=np.float64
    )
    task_rows = []
    for task_index, task_id in enumerate(task_ids):
        for layer_index, block in enumerate(CANDIDATE_BLOCKS):
            values = [
                row["cosine"]
                for row in state_rows
                if row["task_id"] == task_id and row["block"] == block
            ]
            task_by_layer[task_index, layer_index] = np.nanmean(values)
            task_rows.append(
                {"task_id": task_id, "block": block, "mean_cosine": np.nanmean(values)}
            )
    selection = earliest_within_one_se(task_by_layer)
    selected_block = CANDIDATE_BLOCKS[selection["selected_index"]]
    result = {
        "selected_block": int(selected_block),
        "selection_rule": "earliest_within_one_standard_error_of_best_task_mean_cosine",
        "candidate_blocks": CANDIDATE_BLOCKS,
        "task_mean_cosines": task_by_layer.tolist(),
        "means": selection["means"].tolist(),
        "standard_errors": selection["standard_errors"].tolist(),
        "one_se_threshold": selection["threshold"],
        "best_block": CANDIDATE_BLOCKS[selection["best_index"]],
        "construction_only": True,
        "passed_feasibility": bool(
            selection["means"][selection["selected_index"]] >= MIN_SCAN_COSINE
        ),
    }
    write_csv(ANALYSIS_DIR / "carrier_scan_task_rows.csv", task_rows)
    write_json(ANALYSIS_DIR / "carrier_selection.json", result)
    return result


def fit_selected_channel_metric(selected_block):
    layer_index = CANDIDATE_BLOCKS.index(selected_block)
    count = 0
    total = np.zeros(EXPECTED_CARRIER_CHANNELS, dtype=np.float64)
    cross = np.zeros(
        (EXPECTED_CARRIER_CHANNELS, EXPECTED_CARRIER_CHANNELS), dtype=np.float64
    )
    for record in CONSTRUCTION_RECORDS:
        with np.load(scan_path(record["record_id"])) as payload:
            count += int(payload["moment_count"][layer_index])
            total += payload["moment_sum"][layer_index]
            cross += payload["moment_cross"][layer_index]
    metric = channel_metric_from_moments(
        count,
        total,
        cross,
        shrinkage=CHANNEL_SHRINKAGE,
        relative_floor=CHANNEL_EIGEN_FLOOR,
    )
    np.savez_compressed(
        ANALYSIS_DIR / "carrier_channel_metric.npz",
        count=np.asarray(count, dtype=np.int64),
        mean=metric["mean"].astype(np.float64),
        covariance=metric["covariance"].astype(np.float64),
        eigenvalues=metric["eigenvalues"].astype(np.float64),
        square_root=metric["square_root"].astype(np.float64),
        inverse_square_root=metric["inverse_square_root"].astype(np.float64),
        condition_number=np.asarray(metric["condition_number"]),
    )
    return metric


if not PIPELINE_FAILED:
    try:
        run_carrier_scan()
        CARRIER_SELECTION = select_carrier_layer()
        SELECTED_BLOCK = int(CARRIER_SELECTION["selected_block"])
        CHANNEL_METRIC = fit_selected_channel_metric(SELECTED_BLOCK)
        memory_report("carrier_selected")
    except Exception:
        record_failure("carrier_scan")
'''


exact_write_read = r'''# Exact normalized-action JVPs and target-energy VJPs at the frozen carrier.


def shard_path(record_id, horizon):
    return JACOBIAN_DIR / f"state_{record_id:04d}_h{horizon}.npz"


def query_gradients_at_selected_block(record_id, horizon, pairs):
    payload = scan_state_carrier(
        record_id,
        horizon,
        [SELECTED_BLOCK],
        pairs,
        [],
    )
    return np.stack(payload["gradients"][SELECTED_BLOCK]), payload["context"]


def action_direction_payload(record_id, horizon):
    _, actions = state_model_inputs(record_id, horizon)
    train = normalized_action_directions(actions, TRAIN_ACTION_INDICES)
    test = normalized_action_directions(actions, TEST_ACTION_INDICES)
    test_values = (
        actions[:, TEST_ACTION_INDICES]
        .permute(1, 0, 2)
        .detach()
        .cpu()
        .numpy()
        .reshape(len(TEST_ACTION_INDICES), -1)
        .T
    )
    base = (
        actions[:, READ_BRANCH]
        .detach()
        .cpu()
        .numpy()
        .reshape(-1, 1)
    )
    raw_test = test_values - base
    return train, test, raw_test


def transform_jacobians(g_raw, b_raw):
    g_shape = g_raw.shape
    b_shape = b_raw.shape
    g = transform_dual_channels(
        g_raw.reshape(g_shape[0], 256, EXPECTED_CARRIER_CHANNELS),
        CHANNEL_METRIC["square_root"],
    ).reshape(g_shape[0], -1)
    b = transform_primal_channels(
        b_raw.reshape(256, EXPECTED_CARRIER_CHANNELS, b_shape[1]).transpose(0, 2, 1),
        CHANNEL_METRIC["inverse_square_root"],
    ).transpose(0, 2, 1).reshape(-1, b_shape[1])
    return g, b


def build_write_read_shard(record, horizon):
    record_id = int(record["record_id"])
    destination = shard_path(record_id, horizon)
    if destination.exists():
        return
    pairs = TRAIN_QUERY_PAIRS if record["split"] == "construction" else TEST_QUERY_PAIRS
    gradient_pairs = (
        TRAIN_QUERY_PAIRS
        if record["split"] == "construction"
        else TRAIN_QUERY_PAIRS + TEST_QUERY_PAIRS
    )
    g_raw_all, context = query_gradients_at_selected_block(
        record_id, horizon, gradient_pairs
    )
    g_raw_all = g_raw_all.reshape(len(gradient_pairs), -1)
    targets = load_target_tokens(record_id)
    all_query_separations = np.asarray(
        [query_separation(targets, horizon, pair)[1] for pair in gradient_pairs],
        dtype=np.float64,
    )
    b_raw, used_fallback = exact_action_jacobian(record_id, horizon, SELECTED_BLOCK)
    g_all, b = transform_jacobians(g_raw_all, b_raw)
    if record["split"] == "construction":
        g = g_all
        g_train = g_all
        g_raw = g_raw_all
        query_separations = all_query_separations
        train_query_separations = all_query_separations
    else:
        split_index = len(TRAIN_QUERY_PAIRS)
        g_train = g_all[:split_index]
        g = g_all[split_index:]
        g_raw = g_raw_all[split_index:]
        train_query_separations = all_query_separations[:split_index]
        query_separations = all_query_separations[split_index:]
    train_directions, test_directions, raw_test_directions = action_direction_payload(
        record_id, horizon
    )
    selected_directions = (
        train_directions if record["split"] == "construction" else test_directions
    )
    written = b @ selected_directions
    written_train = b @ train_directions
    raw_written = b_raw @ selected_directions
    metric_invariance_error = relative_error(
        g @ written, g_raw @ raw_written
    )
    if metric_invariance_error > 1e-6:
        raise RuntimeError(
            f"hidden metric changed G@B for state {record_id}: "
            f"{metric_invariance_error}"
        )
    balanced = balanced_modes(g, written, tolerance=BALANCED_TOLERANCE)
    mode_rows, mode_labels = canonical_mode_rows(
        balanced["primal"], balanced["dual"]
    )
    if len(mode_rows):
        relative_singular = balanced["singular_values"] / max(
            float(balanced["singular_values"][0]), 1e-12
        )
        mode_weights = np.asarray(
            [math.sqrt(float(relative_singular[index])) for _kind, index in mode_labels]
        )
        mode_rows = mode_rows * mode_weights[:, None]
    else:
        mode_weights = np.empty(0, dtype=np.float64)
    if (
        len(balanced["singular_values"])
        and balanced["biorthogonality_error"] > MAX_BIORTHOGONALITY_ERROR
    ):
        raise RuntimeError(
            f"biorthogonality failed for state {record_id}: "
            f"{balanced['biorthogonality_error']}"
        )
    atomic_npz(
        destination,
        record_id=np.asarray(record_id, dtype=np.int64),
        task_id=np.asarray(record["task_id"], dtype=np.int64),
        split=np.asarray(record["split"]),
        horizon=np.asarray(horizon, dtype=np.int64),
        query_pairs=np.asarray(pairs, dtype=np.int64),
        train_query_pairs=np.asarray(TRAIN_QUERY_PAIRS, dtype=np.int64),
        query_separations=query_separations,
        train_query_separations=train_query_separations,
        G=g.astype(np.float32),
        G_train=g_train.astype(np.float32),
        G_norms=np.linalg.norm(g, axis=1).astype(np.float32),
        B=b.astype(np.float32),
        B_norms=np.linalg.norm(b, axis=0).astype(np.float32),
        selected_action_directions=selected_directions.astype(np.float32),
        train_action_directions=train_directions.astype(np.float32),
        raw_test_action_directions=raw_test_directions.astype(np.float32),
        written=written.astype(np.float32),
        written_train=written_train.astype(np.float32),
        hankel=balanced["hankel"].astype(np.float64),
        hankel_energy=np.asarray(np.sum(balanced["hankel"] ** 2)),
        hankel_energy_per_cell=np.asarray(np.mean(balanced["hankel"] ** 2)),
        singular_values=balanced["singular_values"].astype(np.float64),
        primal=balanced["primal"].T.astype(np.float32),
        dual=balanced["dual"].T.astype(np.float32),
        mode_rows=mode_rows.astype(np.float32),
        mode_weights=mode_weights.astype(np.float32),
        mode_labels=np.asarray([f"{kind}:{index}" for kind, index in mode_labels]),
        biorthogonality_error=np.asarray(balanced["biorthogonality_error"]),
        metric_invariance_error=np.asarray(metric_invariance_error),
        context=context.astype(np.float32),
        jvp_fallback=np.asarray(used_fallback),
    )


def adjoint_chain_smoke(record_id):
    horizon = PRIMARY_HORIZON
    initial, actions = state_model_inputs(record_id, horizon)
    base = actions[:, READ_BRANCH : READ_BRANCH + 1]
    targets = load_target_tokens(record_id)
    direction_numpy = query_direction(targets, horizon, TRAIN_QUERY_PAIRS[0])
    direction = torch.as_tensor(direction_numpy, device="cuda")
    predictions, _, captures = forward_with_carriers(
        initial,
        base,
        horizon,
        capture_blocks=[SELECTED_BLOCK],
        require_grad=True,
    )
    scalar = torch.sum(predictions[0] * direction)
    gradient, = torch.autograd.grad(scalar, captures[SELECTED_BLOCK])
    g = layer_tokens_full(gradient)[0].detach().float()

    b_raw, _ = exact_action_jacobian(record_id, horizon, SELECTED_BLOCK)
    rng = np.random.default_rng(SEED + 99)
    tangent = rng.normal(size=b_raw.shape[1])
    tangent /= np.linalg.norm(tangent)
    write = torch.as_tensor(
        (b_raw @ tangent).reshape(1, 256, EXPECTED_CARRIER_CHANNELS),
        device="cuda",
    )

    def suffix_scalar(delta):
        perturbed, _, _ = forward_with_carriers(
            initial,
            base,
            horizon,
            capture_blocks=[SELECTED_BLOCK],
            intervention={"block": SELECTED_BLOCK, "delta": delta},
            require_grad=True,
        )
        return torch.sum(perturbed[0] * direction)

    zero_delta = torch.zeros_like(write)
    _, exact_suffix = torch.autograd.functional.jvp(
        suffix_scalar,
        (zero_delta,),
        (write,),
        strict=True,
        create_graph=False,
    )
    epsilon = JVP_EPSILON
    with torch.inference_mode():
        plus, _, _ = forward_with_carriers(
            initial,
            base,
            horizon,
            capture_blocks=[SELECTED_BLOCK],
            intervention={"block": SELECTED_BLOCK, "delta": epsilon * write},
        )
        minus, _, _ = forward_with_carriers(
            initial,
            base,
            horizon,
            capture_blocks=[SELECTED_BLOCK],
            intervention={"block": SELECTED_BLOCK, "delta": -epsilon * write},
        )
    finite = torch.sum(((plus[0] - minus[0]) / (2 * epsilon)) * direction)
    adjoint = torch.sum(g * write[0])
    exact_absolute_error = float(torch.abs(exact_suffix - adjoint).cpu())
    exact_relative_error = float(
        torch.abs(exact_suffix - adjoint)
        / torch.clamp(
            torch.maximum(torch.abs(exact_suffix), torch.abs(adjoint)), min=1e-8
        )
    )
    finite_relative_error = float(
        torch.abs(finite - exact_suffix)
        / torch.clamp(
            torch.maximum(torch.abs(finite), torch.abs(exact_suffix)), min=1e-8
        )
    )
    result = {
        "finite_suffix_derivative": float(finite.cpu()),
        "exact_suffix_jvp": float(exact_suffix.cpu()),
        "vjp_jvp_contraction": float(adjoint.cpu()),
        "exact_absolute_error": exact_absolute_error,
        "exact_relative_error": exact_relative_error,
        "finite_vs_exact_relative_error": finite_relative_error,
        "finite_difference_epsilon": epsilon,
        "threshold": MAX_ADJOINT_RELATIVE_ERROR,
        "absolute_threshold": MAX_ADJOINT_ABS_ERROR,
        "passed": bool(
            exact_relative_error <= MAX_ADJOINT_RELATIVE_ERROR
            or exact_absolute_error <= MAX_ADJOINT_ABS_ERROR
        ),
    }
    write_json(ANALYSIS_DIR / "adjoint_chain_identity.json", result)
    if not result["passed"]:
        raise RuntimeError(f"adjoint chain identity failed: {result}")
    return result


def jvp_epsilon_linearity_check(record_id):
    horizon = PRIMARY_HORIZON
    initial, actions = state_model_inputs(record_id, horizon)
    base = actions[:, READ_BRANCH : READ_BRANCH + 1].detach()
    b_raw, used_fallback = exact_action_jacobian(
        record_id, horizon, SELECTED_BLOCK
    )
    rng = np.random.default_rng(SEED + 123)
    tangent = rng.normal(size=b_raw.shape[1])
    tangent /= np.linalg.norm(tangent)
    reference = b_raw @ tangent
    tangent_tensor = torch.as_tensor(
        tangent.reshape(base.shape), device="cuda", dtype=base.dtype
    )
    rows = []
    for epsilon in JVP_EPSILON_CHECK:
        with torch.inference_mode():
            _, _, plus_capture = forward_with_carriers(
                initial,
                base + epsilon * tangent_tensor,
                horizon,
                capture_blocks=[SELECTED_BLOCK],
            )
            _, _, minus_capture = forward_with_carriers(
                initial,
                base - epsilon * tangent_tensor,
                horizon,
                capture_blocks=[SELECTED_BLOCK],
            )
        plus = layer_tokens_full(plus_capture[SELECTED_BLOCK])[0]
        minus = layer_tokens_full(minus_capture[SELECTED_BLOCK])[0]
        finite = ((plus - minus) / (2.0 * epsilon)).float().cpu().numpy().reshape(-1)
        rows.append(
            {
                "epsilon": float(epsilon),
                "relative_error_to_jvp": relative_error(finite, reference),
                "cosine_to_jvp": stable_cosine(finite, reference),
            }
        )
    result = {
        "rows": rows,
        "autograd_used_finite_difference_fallback": used_fallback,
        "maximum_relative_error": max(
            row["relative_error_to_jvp"] for row in rows
        ),
        "minimum_cosine": min(row["cosine_to_jvp"] for row in rows),
        "relative_error_threshold": MAX_JVP_RELATIVE_ERROR,
        "passed": bool(
            min(row["cosine_to_jvp"] for row in rows) >= 0.99
            and max(row["relative_error_to_jvp"] for row in rows)
            <= MAX_JVP_RELATIVE_ERROR
        ),
    }
    write_json(ANALYSIS_DIR / "jvp_epsilon_linearity.json", result)
    if not result["passed"]:
        raise RuntimeError(f"JVP epsilon check failed: {result}")
    return result


def run_exact_extraction(records, horizon):
    started = time.perf_counter()
    for index, record in enumerate(records):
        build_write_read_shard(record, horizon)
        write_json(
            OUT / f"write_read_h{horizon}_progress.json",
            {
                "completed": index + 1,
                "total": len(records),
                "last_record_id": record["record_id"],
            },
        )
    key = f"write_read_h{horizon}_seconds"
    TIMINGS[key] = TIMINGS.get(key, 0.0) + time.perf_counter() - started


if not PIPELINE_FAILED:
    try:
        benchmark_record = CONSTRUCTION_RECORDS[0]
        started = time.perf_counter()
        build_write_read_shard(benchmark_record, PRIMARY_HORIZON)
        benchmark_seconds = time.perf_counter() - started
        benchmark = {
            "one_state_exact_write_read_seconds": benchmark_seconds,
            "estimated_primary_minutes": (
                benchmark_seconds * len(ACTIVE_RECORDS) / 60.0
            ),
            "selected_block": SELECTED_BLOCK,
            "carrier_dimensions": 256 * EXPECTED_CARRIER_CHANNELS,
            "action_dimensions_h3": PRIMARY_HORIZON * FRAMESKIP * 2,
            "continue_after_benchmark": CONTINUE_AFTER_BENCHMARK,
        }
        write_json(OUT / "benchmark.json", benchmark)
        print(json.dumps(benchmark, indent=2))
        if not CONTINUE_AFTER_BENCHMARK:
            raise RuntimeError(
                "Benchmark complete. Set CONTINUE_AFTER_BENCHMARK=True to continue."
            )
        run_exact_extraction(CONSTRUCTION_RECORDS, PRIMARY_HORIZON)
        ADJOINT_IDENTITY = adjoint_chain_smoke(benchmark_record["record_id"])
        JVP_LINEARITY = jvp_epsilon_linearity_check(
            benchmark_record["record_id"]
        )
        memory_report("primary_write_read_complete")
    except Exception:
        record_failure("exact_write_read")
'''


sparse_frame = r'''# Learn one sparse frame on construction modes; evaluate unopened task clusters.


def load_shard(record_id, horizon):
    with np.load(shard_path(record_id, horizon)) as payload:
        return {name: payload[name].copy() for name in payload.files}


def fit_construction_frame(horizon):
    mode_rows = []
    observability = []
    controllability = []
    for record in CONSTRUCTION_RECORDS:
        payload = load_shard(record["record_id"], horizon)
        rows = payload["mode_rows"].astype(np.float64)
        if len(rows):
            mode_rows.append(rows)
        observability.append(payload["G"].astype(np.float64))
        controllability.append(payload["written"].astype(np.float64).T)
    mode_rows = np.concatenate(mode_rows, axis=0)
    frame = fit_sparse_frame(mode_rows)

    # A dense controllable-observable upper-bound span, not the sparse claim.
    task_weight = 1.0 / math.sqrt(max(len(CONSTRUCTION_RECORDS), 1))
    g_global = np.concatenate(
        [task_weight * value for value in observability], axis=0
    )
    b_global = np.concatenate(
        [task_weight * value.T for value in controllability], axis=1
    )
    global_balanced = balanced_modes(
        g_global, b_global, tolerance=BALANCED_TOLERANCE
    )
    dense_basis = orthonormal_row_basis(global_balanced["primal"].T)
    np.savez_compressed(
        ANALYSIS_DIR / f"predictive_control_frame_h{horizon}.npz",
        basis=frame["basis"].astype(np.float32),
        components=frame["components"].astype(np.float64),
        hidden_atoms=frame["hidden_atoms"].astype(np.float32),
        construction_codes=frame["codes"].astype(np.float32),
        eigenvalues=frame["eigenvalues"].astype(np.float64),
        atoms=np.asarray(frame["atoms"], dtype=np.int64),
        sparsity=np.asarray(frame["sparsity"], dtype=np.int64),
        dense_basis=dense_basis.astype(np.float32),
        global_hankel_singular_values=global_balanced["singular_values"].astype(np.float64),
    )
    frame["dense_basis"] = dense_basis
    return frame


def fit_transported_local_frame(record, horizon):
    evaluation = load_shard(record["record_id"], horizon)
    query = evaluation["context"].astype(np.float64)
    candidates = []
    for construction in CONSTRUCTION_RECORDS:
        payload = load_shard(construction["record_id"], horizon)
        context = payload["context"].astype(np.float64)
        candidates.append(
            (
                float(np.sum((context - query) ** 2)),
                int(construction["record_id"]),
                payload["mode_rows"].astype(np.float64),
            )
        )
    candidates.sort(key=lambda value: (value[0], value[1]))
    selected = candidates[: min(LOCAL_NEIGHBORS, len(candidates))]
    rows = np.concatenate([value[2] for value in selected], axis=0)
    frame = fit_sparse_frame(rows)
    frame["neighbor_record_ids"] = [value[1] for value in selected]
    frame["neighbor_squared_distances"] = [value[0] for value in selected]
    return frame


def calibrated_positive_floor(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0)]
    if not len(values):
        raise RuntimeError("construction calibration has no positive values")
    return float(
        max(
            ABSOLUTE_NONDEGENERACY_FLOOR,
            CALIBRATION_MULTIPLIER * np.quantile(values, CALIBRATION_QUANTILE),
        )
    )


def calibrate_nondegeneracy_floors(horizon):
    query_separations = []
    hankel_energies_per_cell = []
    causal_target_separations = []
    causal_denominators = []
    for record in CONSTRUCTION_RECORDS:
        payload = load_shard(record["record_id"], horizon)
        query_separations.extend(payload["query_separations"].astype(np.float64))
        hankel_energies_per_cell.append(
            float(payload["hankel_energy_per_cell"])
        )
        targets = load_target_tokens(record["record_id"])
        initial, actions = state_model_inputs(record["record_id"], horizon)
        with torch.inference_mode():
            predictions, _, _ = forward_with_carriers(
                initial,
                actions[:, [READ_BRANCH, *TRAIN_ACTION_INDICES]],
                horizon,
            )
        baseline = predictions[0]
        for offset, action_index in enumerate(TRAIN_ACTION_INDICES, start=1):
            direction, separation = query_separation(
                targets, horizon, (action_index, READ_BRANCH)
            )
            direction_tensor = torch.as_tensor(
                direction / separation,
                device=predictions.device,
                dtype=predictions.dtype,
            )
            denominator = torch.sum(
                (predictions[offset] - baseline) * direction_tensor
            )
            causal_target_separations.append(separation)
            causal_denominators.append(abs(float(denominator.cpu())))
    result = {
        "construction_only": True,
        "rule": (
            "max(absolute_floor, multiplier * construction_quantile)"
        ),
        "quantile": CALIBRATION_QUANTILE,
        "multiplier": CALIBRATION_MULTIPLIER,
        "absolute_floor": ABSOLUTE_NONDEGENERACY_FLOOR,
        "query_separation": calibrated_positive_floor(query_separations),
        "hankel_energy_per_cell": calibrated_positive_floor(
            hankel_energies_per_cell
        ),
        "causal_target_separation": calibrated_positive_floor(
            causal_target_separations
        ),
        "causal_denominator_abs": calibrated_positive_floor(causal_denominators),
        "construction_counts": {
            "query_separations": len(query_separations),
            "hankel_energies_per_cell": len(hankel_energies_per_cell),
            "causal_denominators": len(causal_denominators),
        },
    }
    write_json(
        ANALYSIS_DIR / f"construction_nondegeneracy_floors_h{horizon}.json",
        result,
    )
    return result


def evaluation_transfer_rows(frame, horizon, include_transported=True):
    rows = []
    null_rows = []
    null_seed_values = [int(value) for value in NULL_SEEDS[:NULL_DRAWS]]
    null_dictionaries = [null_components(frame, seed) for seed in null_seed_values]
    for record in EVALUATION_RECORDS:
        payload = load_shard(record["record_id"], horizon)
        g = payload["G"].astype(np.float64)
        g_train = payload["G_train"].astype(np.float64)
        writes = payload["written"].astype(np.float64).T
        writes_train = payload["written_train"].astype(np.float64).T
        target = g @ writes.T
        codes, sparse_writes = frame_reconstruct(writes, frame)
        sparse = g @ sparse_writes.T
        _, sparse_writes_train = frame_reconstruct(writes_train, frame)
        action_holdout_target = g_train @ writes.T
        action_holdout_sparse = g_train @ sparse_writes.T
        query_holdout_target = g @ writes_train.T
        query_holdout_sparse = g @ sparse_writes_train.T
        train_train_target = g_train @ writes_train.T
        train_train_sparse = g_train @ sparse_writes_train.T
        dense_writes = project_rows(writes, frame["dense_basis"])
        dense = g @ dense_writes.T
        local = balanced_modes(g, writes.T, tolerance=BALANCED_TOLERANCE)
        local_basis = orthonormal_row_basis(local["primal"].T)
        local_oracle = g @ project_rows(writes, local_basis).T
        sparse_metrics = transfer_metrics(target, sparse)
        action_holdout_metrics = transfer_metrics(
            action_holdout_target, action_holdout_sparse
        )
        query_holdout_metrics = transfer_metrics(
            query_holdout_target, query_holdout_sparse
        )
        train_train_metrics = transfer_metrics(train_train_target, train_train_sparse)
        dense_metrics = transfer_metrics(target, dense)
        local_metrics = transfer_metrics(target, local_oracle)
        null_metrics = []
        for null_index, components in enumerate(null_dictionaries):
            _, reconstructed = frame_reconstruct(
                writes, frame, components=components
            )
            _, reconstructed_train = frame_reconstruct(
                writes_train, frame, components=components
            )
            metrics = transfer_metrics(target, g @ reconstructed.T)
            action_metrics = transfer_metrics(
                action_holdout_target, g_train @ reconstructed.T
            )
            query_metrics = transfer_metrics(
                query_holdout_target, g @ reconstructed_train.T
            )
            train_train_null_metrics = transfer_metrics(
                train_train_target, g_train @ reconstructed_train.T
            )
            null_metrics.append(metrics)
            null_rows.append(
                {
                    "record_id": int(record["record_id"]),
                    "task_id": int(record["task_id"]),
                    "horizon": int(horizon),
                    "null_index": null_index,
                    "seed": null_seed_values[null_index],
                    "null_family": "global_haar",
                    "reconstruction": metrics["reconstruction"],
                    "action_holdout_reconstruction": action_metrics["reconstruction"],
                    "query_holdout_reconstruction": query_metrics["reconstruction"],
                    "train_train_reconstruction": train_train_null_metrics[
                        "reconstruction"
                    ],
                    "cosine": metrics["cosine"],
                    "scale": metrics["scale"],
                }
            )
        transported_metrics = {
            "reconstruction": np.nan, "cosine": np.nan
        }
        transported_strongest = np.nan
        transported_neighbor_ids = []
        transported_neighbor_distances = []
        if include_transported:
            transported_frame = fit_transported_local_frame(record, horizon)
            _, transported_writes = frame_reconstruct(
                writes, transported_frame
            )
            transported_metrics = transfer_metrics(
                target, g @ transported_writes.T
            )
            transported_null_metrics = []
            for null_index, seed in enumerate(null_seed_values):
                components = null_components(transported_frame, seed + 100_000)
                _, reconstructed = frame_reconstruct(
                    writes, transported_frame, components=components
                )
                metrics = transfer_metrics(target, g @ reconstructed.T)
                transported_null_metrics.append(metrics)
                null_rows.append(
                    {
                        "record_id": int(record["record_id"]),
                        "task_id": int(record["task_id"]),
                        "horizon": int(horizon),
                        "null_index": null_index,
                        "seed": seed + 100_000,
                        "null_family": "transported_local_haar",
                        "reconstruction": metrics["reconstruction"],
                        "cosine": metrics["cosine"],
                        "scale": metrics["scale"],
                    }
                )
            transported_strongest = max(
                value["reconstruction"] for value in transported_null_metrics
            )
            transported_neighbor_ids = transported_frame["neighbor_record_ids"]
            transported_neighbor_distances = transported_frame[
                "neighbor_squared_distances"
            ]
        strongest_null_reconstruction = max(
            value["reconstruction"] for value in null_metrics
        )
        strongest_null_cosine = max(value["cosine"] for value in null_metrics)
        active_supports = [
            tuple(np.flatnonzero(np.abs(code) > 1e-10).tolist()) for code in codes
        ]
        rows.append(
            {
                "record_id": int(record["record_id"]),
                "task_id": int(record["task_id"]),
                "horizon": int(horizon),
                "transfer_energy": sparse_metrics["energy"],
                "joint_energy_per_cell": float(np.mean(target**2)),
                "action_holdout_energy_per_cell": float(
                    np.mean(action_holdout_target**2)
                ),
                "query_holdout_energy_per_cell": float(
                    np.mean(query_holdout_target**2)
                ),
                "train_train_energy_per_cell": float(
                    np.mean(train_train_target**2)
                ),
                "minimum_query_separation": float(
                    np.min(payload["query_separations"])
                ),
                "minimum_train_query_separation": float(
                    np.min(payload["train_query_separations"])
                ),
                "sparse_reconstruction": sparse_metrics["reconstruction"],
                "action_holdout_reconstruction": action_holdout_metrics[
                    "reconstruction"
                ],
                "query_holdout_reconstruction": query_holdout_metrics[
                    "reconstruction"
                ],
                "train_train_reconstruction": train_train_metrics[
                    "reconstruction"
                ],
                "sparse_cosine": sparse_metrics["cosine"],
                "sparse_scale": sparse_metrics["scale"],
                "dense_reconstruction": dense_metrics["reconstruction"],
                "local_oracle_reconstruction": local_metrics["reconstruction"],
                "transported_reconstruction": transported_metrics["reconstruction"],
                "transported_cosine": transported_metrics["cosine"],
                "transported_strongest_null_reconstruction": transported_strongest,
                "transported_gain_over_strongest_null": (
                    transported_metrics["reconstruction"] - transported_strongest
                ),
                "transport_neighbor_record_ids": json.dumps(
                    transported_neighbor_ids
                ),
                "transport_neighbor_squared_distances": json.dumps(
                    transported_neighbor_distances
                ),
                "strongest_null_reconstruction": strongest_null_reconstruction,
                "strongest_null_cosine": strongest_null_cosine,
                "gain_over_strongest_null": (
                    sparse_metrics["reconstruction"] - strongest_null_reconstruction
                ),
                "cosine_gain_over_strongest_null": (
                    sparse_metrics["cosine"] - strongest_null_cosine
                ),
                "mean_support_size": float(
                    np.mean([len(value) for value in active_supports])
                ),
                "supports": json.dumps(active_supports),
            }
        )
    write_csv(ANALYSIS_DIR / f"heldout_transfer_h{horizon}_state_rows.csv", rows)
    write_csv(ANALYSIS_DIR / f"heldout_transfer_h{horizon}_null_rows.csv", null_rows)
    return rows


def aggregate_null_test(rows, horizon, family, observed_metric):
    null_metric = {
        "sparse_reconstruction": "reconstruction",
        "transported_reconstruction": "reconstruction",
        "action_holdout_reconstruction": "action_holdout_reconstruction",
        "query_holdout_reconstruction": "query_holdout_reconstruction",
        "train_train_reconstruction": "train_train_reconstruction",
    }[observed_metric]
    observed_by_task = defaultdict(list)
    for row in rows:
        observed_by_task[int(row["task_id"])].append(float(row[observed_metric]))
    observed = float(np.median([
        np.mean(values) for values in observed_by_task.values()
    ]))
    with (ANALYSIS_DIR / f"heldout_transfer_h{horizon}_null_rows.csv").open() as handle:
        null_rows = [
            row for row in csv.DictReader(handle)
            if row["null_family"] == family
        ]
    by_draw_task = defaultdict(list)
    for row in null_rows:
        by_draw_task[(int(row["null_index"]), int(row["task_id"]))].append(
            float(row[null_metric])
        )
    draws = defaultdict(list)
    for (draw, _task_id), values in by_draw_task.items():
        draws[draw].append(float(np.mean(values)))
    null_statistics = np.asarray(
        [np.median(draws[index]) for index in sorted(draws)], dtype=np.float64
    )
    if len(null_statistics) != NULL_DRAWS or any(
        len(draws[index]) != len(observed_by_task) for index in draws
    ):
        raise RuntimeError("incomplete coherent representation null ensemble")
    empirical_p = float(
        (1 + np.sum(null_statistics >= observed)) / (len(null_statistics) + 1)
    )
    null_95 = float(np.quantile(null_statistics, 0.95))
    task_gains = {}
    for task_id, observed_values in sorted(observed_by_task.items()):
        task_null = np.asarray(
            [
                float(np.mean(by_draw_task[(draw, task_id)]))
                for draw in sorted(draws)
            ],
            dtype=np.float64,
        )
        task_gains[str(task_id)] = float(
            np.mean(observed_values) - np.quantile(task_null, 0.95)
        )
    return {
        "observed_task_equal_median": observed,
        "null_draws": int(len(null_statistics)),
        "null_task_equal_median_95th_percentile": null_95,
        "gain_over_null_95th_percentile": observed - null_95,
        "one_sided_empirical_p": empirical_p,
        "task_gains_over_task_null_95th_percentile": task_gains,
        "positive_task_gains": int(
            np.sum([value > 0 for value in task_gains.values()])
        ),
        "coherent_draw_across_tasks": True,
    }


def count_nondegenerate_tasks(rows, floors):
    count = 0
    for task_id in sorted({int(row["task_id"]) for row in rows}):
        selected = [row for row in rows if int(row["task_id"]) == task_id]
        state_passes = [
            row["minimum_query_separation"] >= floors["query_separation"]
            and row["minimum_train_query_separation"]
            >= floors["query_separation"]
            and min(
                row["joint_energy_per_cell"],
                row["action_holdout_energy_per_cell"],
                row["query_holdout_energy_per_cell"],
                row["train_train_energy_per_cell"],
            ) >= floors["hankel_energy_per_cell"]
            for row in selected
        ]
        if state_passes and all(state_passes):
            count += 1
    return count


def representation_gate(rows, horizon):
    reconstruction = summarize_task_metric(
        rows, "sparse_reconstruction", BOOTSTRAP_SEED + horizon
    )
    gain = summarize_task_metric(
        rows, "gain_over_strongest_null", BOOTSTRAP_SEED + 100 + horizon
    )
    local = summarize_task_metric(
        rows, "local_oracle_reconstruction", BOOTSTRAP_SEED + 200 + horizon
    )
    transported = summarize_task_metric(
        rows, "transported_reconstruction", BOOTSTRAP_SEED + 210 + horizon
    )
    transported_gain = summarize_task_metric(
        rows,
        "transported_gain_over_strongest_null",
        BOOTSTRAP_SEED + 220 + horizon,
    )
    required_positive = min(MIN_POSITIVE_TASKS, len(ACTIVE_EVALUATION_CLUSTERS))
    global_null_test = aggregate_null_test(
        rows, horizon, "global_haar", "sparse_reconstruction"
    )
    transported_null_test = aggregate_null_test(
        rows, horizon, "transported_local_haar", "transported_reconstruction"
    )
    action_holdout = summarize_task_metric(
        rows, "action_holdout_reconstruction", BOOTSTRAP_SEED + 230 + horizon
    )
    query_holdout = summarize_task_metric(
        rows, "query_holdout_reconstruction", BOOTSTRAP_SEED + 240 + horizon
    )
    train_train = summarize_task_metric(
        rows, "train_train_reconstruction", BOOTSTRAP_SEED + 250 + horizon
    )
    action_holdout_null_test = aggregate_null_test(
        rows, horizon, "global_haar", "action_holdout_reconstruction"
    )
    query_holdout_null_test = aggregate_null_test(
        rows, horizon, "global_haar", "query_holdout_reconstruction"
    )
    train_train_null_test = aggregate_null_test(
        rows, horizon, "global_haar", "train_train_reconstruction"
    )
    nondegenerate_tasks = count_nondegenerate_tasks(rows, CALIBRATED_FLOORS)
    passed = bool(
        RUN_MODE == "pilot"
        and CARRIER_SELECTION["passed_feasibility"]
        and reconstruction["task_equal_median"] >= MIN_HELDOUT_RECONSTRUCTION
        and global_null_test["gain_over_null_95th_percentile"]
        >= MIN_GAIN_OVER_NULL_95
        and global_null_test["one_sided_empirical_p"] <= 0.05
        and global_null_test["positive_task_gains"] >= required_positive
        and action_holdout["task_equal_median"] >= MIN_HELDOUT_RECONSTRUCTION
        and action_holdout_null_test["gain_over_null_95th_percentile"]
        >= MIN_GAIN_OVER_NULL_95
        and action_holdout_null_test["one_sided_empirical_p"] <= 0.05
        and action_holdout_null_test["positive_task_gains"] >= required_positive
        and query_holdout["task_equal_median"] >= MIN_HELDOUT_RECONSTRUCTION
        and query_holdout_null_test["gain_over_null_95th_percentile"]
        >= MIN_GAIN_OVER_NULL_95
        and query_holdout_null_test["one_sided_empirical_p"] <= 0.05
        and query_holdout_null_test["positive_task_gains"] >= required_positive
        and train_train["task_equal_median"] >= MIN_HELDOUT_RECONSTRUCTION
        and train_train_null_test["gain_over_null_95th_percentile"]
        >= MIN_GAIN_OVER_NULL_95
        and train_train_null_test["one_sided_empirical_p"] <= 0.05
        and train_train_null_test["positive_task_gains"] >= required_positive
        and nondegenerate_tasks >= required_positive
    )
    atlas_passed = bool(
        RUN_MODE == "pilot"
        and CARRIER_SELECTION["passed_feasibility"]
        and transported["task_equal_median"] >= MIN_HELDOUT_RECONSTRUCTION
        and transported_null_test["gain_over_null_95th_percentile"]
        >= MIN_GAIN_OVER_NULL_95
        and transported_null_test["one_sided_empirical_p"] <= 0.05
        and transported_null_test["positive_task_gains"] >= required_positive
        and nondegenerate_tasks >= required_positive
    )
    result = {
        "horizon": horizon,
        "reconstruction": reconstruction,
        "gain_over_strongest_null": gain,
        "local_oracle": local,
        "transported_local_frame": transported,
        "transported_gain_over_strongest_null": transported_gain,
        "global_coherent_null_test": global_null_test,
        "transported_coherent_null_test": transported_null_test,
        "action_holdout_train_query": action_holdout,
        "action_holdout_coherent_null_test": action_holdout_null_test,
        "query_holdout_train_actions": query_holdout,
        "query_holdout_coherent_null_test": query_holdout_null_test,
        "evaluation_train_train": train_train,
        "train_train_coherent_null_test": train_train_null_test,
        "nondegenerate_tasks": nondegenerate_tasks,
        "required_positive_tasks": required_positive,
        "thresholds": {
            "minimum_reconstruction": MIN_HELDOUT_RECONSTRUCTION,
            "minimum_gain_over_null_95th_percentile": MIN_GAIN_OVER_NULL_95,
        },
        "passed": passed,
        "atlas_passed": atlas_passed,
        "smoke_cannot_pass": RUN_MODE == "smoke",
    }
    write_json(ANALYSIS_DIR / f"representation_gate_h{horizon}.json", result)
    return result


def freeze_before_evaluation(frame, horizon):
    verify_executed_notebook_through(
        "# Learn one sparse frame on construction modes; evaluate unopened task clusters."
    )
    frame_path = ANALYSIS_DIR / f"predictive_control_frame_h{horizon}.npz"
    payload = {
        "created_after_construction_and_before_evaluation_outcomes": True,
        "protocol_id": PROTOCOL_ID,
        "run_signature": RUN_SIGNATURE,
        "source_identity": SOURCE_IDENTITY,
        "design_freeze_sha256": sha256_file(DESIGN_DIR / "design_freeze.json"),
        "selected_block": SELECTED_BLOCK,
        "carrier_selection_sha256": sha256_file(
            ANALYSIS_DIR / "carrier_selection.json"
        ),
        "carrier_metric_sha256": sha256_file(
            ANALYSIS_DIR / "carrier_channel_metric.npz"
        ),
        "frame_sha256": sha256_file(frame_path),
        "nondegeneracy_floors": CALIBRATED_FLOORS,
        "nondegeneracy_floors_sha256": sha256_file(
            ANALYSIS_DIR / f"construction_nondegeneracy_floors_h{horizon}.json"
        ),
        "temporal_nondegeneracy_floors": TEMPORAL_CALIBRATED_FLOORS,
        "temporal_nondegeneracy_floors_sha256": (
            sha256_file(ANALYSIS_DIR / "construction_nondegeneracy_floors_h1.json")
            if TEMPORAL_CALIBRATED_FLOORS is not None else None
        ),
        "train_query_pairs": TRAIN_QUERY_PAIRS,
        "test_query_pairs": TEST_QUERY_PAIRS,
        "train_action_indices": TRAIN_ACTION_INDICES,
        "test_action_indices": TEST_ACTION_INDICES,
        "frame_atoms": int(frame["atoms"]),
        "frame_sparsity": int(frame["sparsity"]),
        "causal_doses": CAUSAL_DOSES,
        "representation_null_draws": NULL_DRAWS,
        "causal_null_draws": CAUSAL_NULL_DRAWS,
        "null_seeds": [int(value) for value in NULL_SEEDS.tolist()],
        "thresholds": {
            "minimum_scan_cosine": MIN_SCAN_COSINE,
            "minimum_positive_tasks": MIN_POSITIVE_TASKS,
            "minimum_heldout_reconstruction": MIN_HELDOUT_RECONSTRUCTION,
            "minimum_gain_over_null_95th_percentile": MIN_GAIN_OVER_NULL_95,
            "minimum_causal_mediation": MIN_CAUSAL_MEDIATION,
            "minimum_causal_gain_over_null": MIN_CAUSAL_GAIN_OVER_NULL,
            "minimum_causal_gain_over_complement": MIN_CAUSAL_GAIN_OVER_COMPLEMENT,
            "minimum_positive_control_recovery": MIN_POSITIVE_CONTROL_RECOVERY,
            "maximum_positive_control_recovery": MAX_POSITIVE_CONTROL_RECOVERY,
            "minimum_causal_denominator_coherence": MIN_CAUSAL_DENOMINATOR_COHERENCE,
            "minimum_causal_output_reconstruction": MIN_CAUSAL_OUTPUT_RECONSTRUCTION,
            "minimum_causal_output_displacement_cosine": (
                MIN_CAUSAL_OUTPUT_DISPLACEMENT_COSINE
            ),
            "maximum_causal_recovery": MAX_CAUSAL_RECOVERY,
            "maximum_patch_to_natural_norm_ratio": (
                MAX_PATCH_TO_NATURAL_NORM_RATIO
            ),
            "minimum_interaction_stratum_tasks": MIN_INTERACTION_STRATUM_TASKS,
            "required_interaction_strata": list(REQUIRED_INTERACTION_STRATA),
            "minimum_temporal_reconstruction": MIN_TEMPORAL_RECONSTRUCTION,
        },
    }
    destination = DESIGN_DIR / "evaluation_preregistration.json"
    if destination.exists():
        existing = json.loads(destination.read_text())
        if existing != payload:
            raise RuntimeError("existing evaluation preregistration differs")
        return {"status": "RESUME_ALREADY_FROZEN", **payload}
    premature = []
    for record in EVALUATION_RECORDS:
        record_id = int(record["record_id"])
        for path in [
            TRUTH_DIR / f"state_{record_id:04d}.npz",
            TARGET_DIR / f"state_{record_id:04d}.npz",
            shard_path(record_id, horizon),
        ]:
            if path.exists():
                premature.append(str(path))
    if premature:
        raise RuntimeError(
            f"evaluation data existed before the freeze: {premature[:3]}"
        )
    write_json(destination, payload)
    write_json(
        DESIGN_DIR / "evaluation_freeze_certificate.json",
        {"sha256": sha256_file(destination)},
    )
    return {"status": "FROZEN_BEFORE_OPEN", **payload}


def open_evaluation_after_freeze():
    freeze_path = DESIGN_DIR / "evaluation_preregistration.json"
    certificate = DESIGN_DIR / "evaluation_freeze_certificate.json"
    if not freeze_path.exists() or not certificate.exists():
        raise RuntimeError("evaluation freeze/certificate is absent")
    expected = json.loads(certificate.read_text())["sha256"]
    if sha256_file(freeze_path) != expected:
        raise RuntimeError("evaluation freeze certificate mismatch")
    generate_truth(EVALUATION_RECORDS)
    encode_target_cache(EVALUATION_RECORDS)
    run_exact_extraction(EVALUATION_RECORDS, PRIMARY_HORIZON)


if not PIPELINE_FAILED:
    try:
        PRIMARY_FRAME = fit_construction_frame(PRIMARY_HORIZON)
        CALIBRATED_FLOORS = calibrate_nondegeneracy_floors(PRIMARY_HORIZON)
        if RUN_TEMPORAL_EXTENSION:
            run_exact_extraction(CONSTRUCTION_RECORDS, 1)
            TEMPORAL_CALIBRATED_FLOORS = calibrate_nondegeneracy_floors(1)
        else:
            TEMPORAL_CALIBRATED_FLOORS = None
        EVALUATION_FREEZE = freeze_before_evaluation(
            PRIMARY_FRAME, PRIMARY_HORIZON
        )
        open_evaluation_after_freeze()
        PRIMARY_TRANSFER_ROWS = evaluation_transfer_rows(
            PRIMARY_FRAME, PRIMARY_HORIZON
        )
        REPRESENTATION_GATE = representation_gate(
            PRIMARY_TRANSFER_ROWS, PRIMARY_HORIZON
        )
        print(json.dumps(REPRESENTATION_GATE, indent=2))
        memory_report("sparse_frame_evaluated")
    except Exception:
        record_failure("sparse_frame")
'''


causal_mediation = r'''# Causal indirect-path mediation on one frozen state per evaluation cluster.


def native_query_score(tokens, target_direction):
    return float(torch.sum(tokens * target_direction).detach().cpu())


def whitened_to_native_patch(vector):
    values = np.asarray(vector, dtype=np.float64).reshape(
        256, EXPECTED_CARRIER_CHANNELS
    )
    native = inverse_transform_primal_channels(
        values, CHANNEL_METRIC["square_root"]
    )
    return torch.as_tensor(native, device="cuda", dtype=torch.float32)[None]


def native_to_whitened_patch(vector):
    values = np.asarray(
        vector.detach().float().cpu().numpy(), dtype=np.float64
    ).reshape(256, EXPECTED_CARRIER_CHANNELS)
    whitened = transform_primal_channels(
        values, CHANNEL_METRIC["inverse_square_root"]
    )
    return whitened.reshape(-1)


def norm_match_native(candidate, reference):
    candidate_norm = torch.linalg.vector_norm(candidate)
    reference_norm = torch.linalg.vector_norm(reference)
    if candidate_norm <= 1e-12:
        raise RuntimeError("cannot norm-match a zero causal control")
    return candidate * (reference_norm / candidate_norm)


def output_displacement_metrics(base, counterfactual, patched):
    target = (counterfactual - base).detach().float().cpu().numpy()
    estimate = (patched - base).detach().float().cpu().numpy()
    return transfer_metrics(target, estimate)


def causal_records():
    by_cluster = {}
    for record in EVALUATION_RECORDS:
        by_cluster.setdefault(record["cluster_index"], record)
    return [by_cluster[key] for key in sorted(by_cluster)]


def run_causal_mediation(frame):
    started = time.perf_counter()
    rows = []
    null_seed_values = [
        int(value) for value in NULL_SEEDS[:CAUSAL_NULL_DRAWS]
    ]
    null_dictionaries = [null_components(frame, seed) for seed in null_seed_values]
    for record_index, record in enumerate(causal_records()):
        record_id = int(record["record_id"])
        shard = load_shard(record_id, PRIMARY_HORIZON)
        b = shard["B"].astype(np.float64)
        initial, actions = state_model_inputs(record_id, PRIMARY_HORIZON)
        targets = load_target_tokens(record_id)
        with torch.inference_mode():
            baseline, _, captures = forward_with_carriers(
                initial,
                actions[:, READ_BRANCH : READ_BRANCH + 1],
                PRIMARY_HORIZON,
                capture_blocks=[SELECTED_BLOCK],
            )
            baseline_pose = DECODE_PHYSICAL_POSE(baseline)
        base_capture = layer_tokens_full(captures[SELECTED_BLOCK])[0].detach()
        with np.load(TRUTH_DIR / f"state_{record_id:04d}.npz") as truth:
            interaction_types = truth["interaction_types"].astype(str)
        horizon_index = HORIZONS.index(PRIMARY_HORIZON)

        for action_offset, action_index in enumerate(TEST_ACTION_INDICES):
            raw_target_direction, target_separation = query_separation(
                targets, PRIMARY_HORIZON, (action_index, READ_BRANCH)
            )
            target_direction = torch.as_tensor(
                raw_target_direction / target_separation,
                device="cuda",
                dtype=torch.float32,
            )
            with torch.inference_mode():
                counterfactual, _, counter_captures = forward_with_carriers(
                    initial,
                    actions[:, action_index : action_index + 1],
                    PRIMARY_HORIZON,
                    capture_blocks=[SELECTED_BLOCK],
                )
                counter_pose = DECODE_PHYSICAL_POSE(counterfactual)
            counter_capture = layer_tokens_full(
                counter_captures[SELECTED_BLOCK]
            )[0].detach()
            action_direction = (
                actions[:, action_index].detach().cpu().numpy().reshape(-1)
                - actions[:, READ_BRANCH].detach().cpu().numpy().reshape(-1)
            )
            linear_write = b @ action_direction
            natural_delta = (counter_capture - base_capture)[None]
            natural_write = native_to_whitened_patch(natural_delta[0])
            _, sparse_write = frame_reconstruct(natural_write, frame)
            sparse_write = sparse_write[0]
            dense_write = project_rows(natural_write[None], frame["dense_basis"])[0]
            complement_write = natural_write - sparse_write
            null_writes = []
            for components in null_dictionaries:
                _, value = frame_reconstruct(
                    natural_write, frame, components=components
                )
                null_writes.append(value[0])
            native_linear = whitened_to_native_patch(linear_write)
            native_sparse = whitened_to_native_patch(sparse_write)
            native_dense = whitened_to_native_patch(dense_write)
            native_complement = whitened_to_native_patch(complement_write)
            native_nulls = []
            for value in null_writes:
                candidate = whitened_to_native_patch(value)
                scale = float(
                    torch.linalg.vector_norm(native_sparse).cpu()
                    / torch.clamp(torch.linalg.vector_norm(candidate).cpu(), min=1e-12)
                )
                native_nulls.append(
                    (
                        norm_match_native(candidate, native_sparse),
                        scale * float(np.linalg.norm(value)),
                    )
                )
            native_natural = natural_delta
            sparse_whitened_norm = float(np.linalg.norm(sparse_write))
            linear_whitened_norm = float(np.linalg.norm(linear_write))
            dense_whitened_norm = float(np.linalg.norm(dense_write))
            complement_whitened_norm = float(np.linalg.norm(complement_write))
            natural_whitened_norm = float(np.linalg.norm(natural_write))

            base_score = native_query_score(baseline[0], target_direction)
            counter_score = native_query_score(counterfactual[0], target_direction)
            denominator = counter_score - base_score
            conditions = []
            for dose in CAUSAL_DOSES:
                conditions.append(
                    (
                        "sparse", "sufficiency", dose, READ_BRANCH,
                        dose * native_sparse, abs(dose) * sparse_whitened_norm,
                    )
                )
            conditions.extend(
                [
                    ("sparse", "necessity", 1.0, action_index, -native_sparse, sparse_whitened_norm),
                    ("linear_full", "sufficiency", 1.0, READ_BRANCH, native_linear, linear_whitened_norm),
                    ("linear_full", "necessity", 1.0, action_index, -native_linear, linear_whitened_norm),
                    ("dense_balanced", "sufficiency", 1.0, READ_BRANCH, native_dense, dense_whitened_norm),
                    ("dense_balanced", "necessity", 1.0, action_index, -native_dense, dense_whitened_norm),
                    ("natural_activation", "sufficiency", 1.0, READ_BRANCH, native_natural, natural_whitened_norm),
                    ("natural_activation", "necessity", 1.0, action_index, -native_natural, natural_whitened_norm),
                    ("sparse_complement", "sufficiency", 1.0, READ_BRANCH, native_complement, complement_whitened_norm),
                    ("sparse_complement", "necessity", 1.0, action_index, -native_complement, complement_whitened_norm),
                ]
            )
            for index, (value, whitened_norm) in enumerate(native_nulls):
                conditions.extend(
                    [
                        (f"null_{index:02d}", "sufficiency", 1.0, READ_BRANCH, value, whitened_norm),
                        (f"null_{index:02d}", "necessity", 1.0, action_index, -value, whitened_norm),
                    ]
                )
            for (
                condition, path_test, dose, branch_index, delta, whitened_norm
            ) in conditions:
                with torch.inference_mode():
                    patched, _, _ = forward_with_carriers(
                        initial,
                        actions[:, branch_index : branch_index + 1],
                        PRIMARY_HORIZON,
                        capture_blocks=[SELECTED_BLOCK],
                        intervention={"block": SELECTED_BLOCK, "delta": delta},
                    )
                    patched_pose = DECODE_PHYSICAL_POSE(patched)
                score = native_query_score(patched[0], target_direction)
                if path_test == "sufficiency":
                    query_effect = score - base_score
                    output_estimate = patched[0] - baseline[0]
                    pose_estimate = patched_pose - baseline_pose
                    branch_reference = baseline[0]
                else:
                    query_effect = counter_score - score
                    output_estimate = counterfactual[0] - patched[0]
                    pose_estimate = counter_pose - patched_pose
                    branch_reference = counterfactual[0]
                output_target = counterfactual[0] - baseline[0]
                pose_target = counter_pose - baseline_pose
                displacement = transfer_metrics(
                    output_target.detach().float().cpu().numpy(),
                    output_estimate.detach().float().cpu().numpy(),
                )
                pose_displacement = transfer_metrics(
                    pose_target.cpu().numpy(),
                    pose_estimate.cpu().numpy(),
                )
                output_cosine = stable_cosine(
                    branch_reference.detach().cpu().numpy(),
                    patched[0].detach().cpu().numpy(),
                )
                native_norm = float(torch.linalg.vector_norm(delta).cpu())
                rows.append(
                    {
                        "record_id": record_id,
                        "task_id": int(record["task_id"]),
                        "action_index": int(action_index),
                        "action_label": ACTION_LABELS[action_index],
                        "interaction_type": interaction_types[action_index, horizon_index],
                        "condition": condition,
                        "path_test": path_test,
                        "dose": float(dose),
                        "target_pair_separation": float(target_separation),
                        "query_denominator": denominator,
                        "query_effect": query_effect,
                        "query_recovery": (
                            query_effect / denominator
                            if abs(denominator) > 1e-8 else np.nan
                        ),
                        "output_reconstruction": displacement["reconstruction"],
                        "output_displacement_cosine": displacement["cosine"],
                        "output_cosine_to_branch_reference": output_cosine,
                        "pose_reconstruction": pose_displacement["reconstruction"],
                        "pose_displacement_cosine": pose_displacement["cosine"],
                        "patch_native_norm": native_norm,
                        "patch_whitened_norm": whitened_norm,
                        "natural_native_norm": float(
                            torch.linalg.vector_norm(native_natural).cpu()
                        ),
                        "natural_whitened_norm": natural_whitened_norm,
                    }
                )
        write_json(
            OUT / "causal_progress.json",
            {
                "completed": record_index + 1,
                "total": len(causal_records()),
                "last_record_id": record_id,
            },
        )
    TIMINGS["causal_seconds"] = time.perf_counter() - started
    write_csv(CAUSAL_DIR / "causal_state_action_rows.csv", rows)
    return rows


def causal_gate(rows):
    def recovery(values, condition, path_test, dose=1.0):
        selected = [
            row for row in values
            if row["condition"] == condition
            and row["path_test"] == path_test
            and row["dose"] == dose
        ]
        denominator = float(np.sum([row["query_denominator"] for row in selected]))
        effect = float(np.sum([row["query_effect"] for row in selected]))
        return effect / denominator if abs(denominator) > 1e-12 else np.nan

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["record_id"], row["task_id"], row["action_index"])].append(row)
    state_rows = []
    for (record_id, task_id, action_index), values in grouped.items():
        sufficiency = recovery(values, "sparse", "sufficiency")
        necessity = recovery(values, "sparse", "necessity")
        null_mediations = []
        for condition in sorted(
            {row["condition"] for row in values if row["condition"].startswith("null_")}
        ):
            null_mediations.append(
                0.5 * (
                    recovery(values, condition, "sufficiency")
                    + recovery(values, condition, "necessity")
                )
            )
        state_rows.append(
            {
                "record_id": record_id,
                "task_id": task_id,
                "action_index": action_index,
                "sufficiency_recovery": sufficiency,
                "necessity_recovery": necessity,
                "symmetric_mediated_recovery": 0.5 * (sufficiency + necessity),
                "strongest_null_symmetric_recovery": float(np.max(null_mediations)),
            }
        )
    write_csv(CAUSAL_DIR / "causal_primary_rows.csv", state_rows)

    task_rows = []
    null_task_rows = []
    task_ids = sorted({int(row["task_id"]) for row in rows})
    for task_id in task_ids:
        selected = [row for row in rows if int(row["task_id"]) == task_id]
        primary = [
            row for row in selected
            if row["condition"] == "sparse"
            and row["path_test"] == "sufficiency"
            and row["dose"] == 1.0
        ]
        denominators = np.asarray(
            [row["query_denominator"] for row in primary], dtype=np.float64
        )
        separations = np.asarray(
            [row["target_pair_separation"] for row in primary], dtype=np.float64
        )
        denominator = float(np.sum(denominators))
        denominator_abs_sum = float(np.sum(np.abs(denominators)))
        coherence = abs(denominator) / max(denominator_abs_sum, 1e-12)
        positive_denominators = int(np.sum(denominators > 0))
        denominator_floor_pass = bool(
            np.all(
                np.abs(denominators)
                >= CALIBRATED_FLOORS["causal_denominator_abs"]
            )
        )
        separation_floor_pass = bool(
            np.all(
                separations
                >= CALIBRATED_FLOORS["causal_target_separation"]
            )
        )
        stable = bool(
            denominator_floor_pass
            and separation_floor_pass
            and coherence >= MIN_CAUSAL_DENOMINATOR_COHERENCE
            and positive_denominators >= len(TEST_ACTION_INDICES) - 1
        )
        sparse_sufficiency = recovery(selected, "sparse", "sufficiency")
        sparse_necessity = recovery(selected, "sparse", "necessity")
        mediation = 0.5 * (sparse_sufficiency + sparse_necessity)
        complement = 0.5 * (
            recovery(selected, "sparse_complement", "sufficiency")
            + recovery(selected, "sparse_complement", "necessity")
        )
        linear_control = 0.5 * (
            recovery(selected, "linear_full", "sufficiency")
            + recovery(selected, "linear_full", "necessity")
        )
        natural_control = 0.5 * (
            recovery(selected, "natural_activation", "sufficiency")
            + recovery(selected, "natural_activation", "necessity")
        )
        dense_control = 0.5 * (
            recovery(selected, "dense_balanced", "sufficiency")
            + recovery(selected, "dense_balanced", "necessity")
        )
        negative_dose = recovery(selected, "sparse", "sufficiency", -0.5)
        half_dose = recovery(selected, "sparse", "sufficiency", 0.5)
        full_dose = sparse_sufficiency
        dose_response_pass = bool(
            full_dose > 0
            and -0.75 * full_dose <= negative_dose <= -0.25 * full_dose
            and 0.25 * full_dose <= half_dose <= 0.75 * full_dose
        )
        null_ratios = []
        for condition in sorted(
            {row["condition"] for row in selected if row["condition"].startswith("null_")}
        ):
            null_mediation = 0.5 * (
                recovery(selected, condition, "sufficiency")
                + recovery(selected, condition, "necessity")
            )
            null_index = int(condition.split("_")[1])
            null_ratios.append(null_mediation)
            null_task_rows.append(
                {
                    "task_id": task_id,
                    "null_index": null_index,
                    "symmetric_mediated_recovery": null_mediation,
                    "stable_denominator": stable,
                }
            )
        strongest_null = float(np.nanmax(null_ratios)) if null_ratios else np.nan
        sparse_rows = [
            row for row in selected
            if row["condition"] == "sparse" and row["dose"] == 1.0
        ]
        patch_norm_ratio = max(
            row["patch_native_norm"] / max(row["natural_native_norm"], 1e-12)
            for row in sparse_rows
        )
        output_reconstruction = float(
            np.mean([row["output_reconstruction"] for row in sparse_rows])
        )
        output_displacement_cosine = float(
            np.mean([row["output_displacement_cosine"] for row in sparse_rows])
        )
        minimum_output_cosine = float(
            np.min([
                row["output_cosine_to_branch_reference"] for row in sparse_rows
            ])
        )
        overshoot_pass = bool(
            max(
                abs(sparse_sufficiency),
                abs(sparse_necessity),
                abs(mediation),
                abs(negative_dose),
                abs(half_dose),
            )
            <= MAX_CAUSAL_RECOVERY
        )
        positive_controls_pass = bool(
            MIN_POSITIVE_CONTROL_RECOVERY
            <= natural_control
            <= MAX_POSITIVE_CONTROL_RECOVERY
            and MIN_POSITIVE_CONTROL_RECOVERY
            <= linear_control
            <= MAX_POSITIVE_CONTROL_RECOVERY
        )
        alignment_pass = bool(
            output_reconstruction >= MIN_CAUSAL_OUTPUT_RECONSTRUCTION
            and output_displacement_cosine
            >= MIN_CAUSAL_OUTPUT_DISPLACEMENT_COSINE
        )
        task_rows.append(
            {
                "task_id": task_id,
                "query_denominator_sum": denominator,
                "query_denominator_abs_sum": denominator_abs_sum,
                "denominator_coherence": coherence,
                "positive_action_denominators": positive_denominators,
                "denominator_floor_pass": denominator_floor_pass,
                "target_separation_floor_pass": separation_floor_pass,
                "stable_denominator": stable,
                "sufficiency_recovery": sparse_sufficiency,
                "necessity_recovery": sparse_necessity,
                "symmetric_mediated_recovery": mediation,
                "strongest_null_mediation": strongest_null,
                "gain_over_strongest_null": mediation - strongest_null,
                "complement_mediated_recovery": complement,
                "gain_over_complement": mediation - complement,
                "linear_full_positive_control": linear_control,
                "natural_activation_positive_control": natural_control,
                "dense_balanced_control": dense_control,
                "negative_half_dose_recovery": negative_dose,
                "positive_half_dose_recovery": half_dose,
                "dose_response_pass": dose_response_pass,
                "overshoot_pass": overshoot_pass,
                "positive_controls_pass": positive_controls_pass,
                "mean_output_reconstruction": output_reconstruction,
                "mean_output_displacement_cosine": output_displacement_cosine,
                "minimum_output_cosine": minimum_output_cosine,
                "maximum_patch_to_natural_norm_ratio": patch_norm_ratio,
                "patch_norm_pass": bool(
                    patch_norm_ratio <= MAX_PATCH_TO_NATURAL_NORM_RATIO
                ),
                "output_alignment_pass": alignment_pass,
            }
        )
    write_csv(CAUSAL_DIR / "causal_task_rows.csv", task_rows)
    write_csv(CAUSAL_DIR / "causal_null_task_rows.csv", null_task_rows)
    stable_rows = [row for row in task_rows if row["stable_denominator"]]
    summary_rows = stable_rows if stable_rows else task_rows
    mediation = summarize_task_metric(
        summary_rows, "symmetric_mediated_recovery", BOOTSTRAP_SEED + 301
    )
    complement_gain = summarize_task_metric(
        summary_rows, "gain_over_complement", BOOTSTRAP_SEED + 302
    )
    stable_task_ids = {row["task_id"] for row in stable_rows}
    coherent_nulls = defaultdict(list)
    for row in null_task_rows:
        if row["task_id"] in stable_task_ids:
            coherent_nulls[row["null_index"]].append(
                row["symmetric_mediated_recovery"]
            )
    observed_null_statistic_value = float(
        np.mean([row["symmetric_mediated_recovery"] for row in stable_rows])
    ) if stable_rows else np.nan
    null_statistics = np.asarray(
        [np.mean(coherent_nulls[index]) for index in sorted(coherent_nulls)],
        dtype=np.float64,
    )
    if stable_rows and (
        len(null_statistics) != CAUSAL_NULL_DRAWS
        or any(
            len(coherent_nulls[index]) != len(stable_rows)
            for index in coherent_nulls
        )
    ):
        raise RuntimeError("incomplete coherent causal null ensemble")
    causal_task_null_gains = {}
    for task_row in stable_rows:
        task_id = task_row["task_id"]
        task_nulls = [
            row["symmetric_mediated_recovery"]
            for row in null_task_rows
            if row["task_id"] == task_id
        ]
        causal_task_null_gains[str(task_id)] = float(
            task_row["symmetric_mediated_recovery"]
            - np.quantile(task_nulls, 0.95)
        )
    causal_positive_null_tasks = int(
        np.sum([value > 0 for value in causal_task_null_gains.values()])
    )
    null_empirical_p = (
        float(
            (1 + np.sum(null_statistics >= observed_null_statistic_value))
            / (len(null_statistics) + 1)
        )
        if len(null_statistics) and np.isfinite(observed_null_statistic_value)
        else np.nan
    )
    null_95_value = (
        float(np.quantile(null_statistics, 0.95)) if len(null_statistics) else np.nan
    )
    null_gain_value = observed_null_statistic_value - null_95_value
    interaction_task_rows = []
    for task_id in sorted(stable_task_ids):
        task_values = [row for row in rows if int(row["task_id"]) == task_id]
        for label in sorted({row["interaction_type"] for row in task_values}):
            stratum = [row for row in task_values if row["interaction_type"] == label]
            stratum_primary = [
                row for row in stratum
                if row["condition"] == "sparse"
                and row["path_test"] == "sufficiency"
                and row["dose"] == 1.0
            ]
            denominators = np.asarray(
                [row["query_denominator"] for row in stratum_primary],
                dtype=np.float64,
            )
            separations = np.asarray(
                [row["target_pair_separation"] for row in stratum_primary],
                dtype=np.float64,
            )
            denominator_abs_sum = float(np.sum(np.abs(denominators)))
            denominator_coherence = (
                abs(float(np.sum(denominators)))
                / max(denominator_abs_sum, 1e-12)
            )
            stratum_stable = bool(
                len(stratum_primary) > 0
                and np.all(
                    np.abs(denominators)
                    >= CALIBRATED_FLOORS["causal_denominator_abs"]
                )
                and np.all(
                    separations
                    >= CALIBRATED_FLOORS["causal_target_separation"]
                )
                and denominator_coherence
                >= MIN_CAUSAL_DENOMINATOR_COHERENCE
            )
            if not stratum_stable:
                continue
            value = 0.5 * (
                recovery(stratum, "sparse", "sufficiency")
                + recovery(stratum, "sparse", "necessity")
            )
            if np.isfinite(value):
                interaction_task_rows.append(
                    {
                        "task_id": task_id,
                        "interaction_type": label,
                        "mediation": value,
                        "denominator_coherence": denominator_coherence,
                    }
                )
    interaction_summary = {}
    interaction_task_counts = {}
    for label in sorted({row["interaction_type"] for row in interaction_task_rows}):
        values = [
            row["mediation"] for row in interaction_task_rows
            if row["interaction_type"] == label and np.isfinite(row["mediation"])
        ]
        interaction_summary[label] = float(np.mean(values)) if values else None
        interaction_task_counts[label] = len(values)
    qualified_interactions = [
        label for label in REQUIRED_INTERACTION_STRATA
        if interaction_task_counts.get(label, 0) >= MIN_INTERACTION_STRATUM_TASKS
    ]
    interaction_reversal_pass = bool(
        set(qualified_interactions) == set(REQUIRED_INTERACTION_STRATA)
        and all(interaction_summary[label] >= 0 for label in qualified_interactions)
    )
    required_positive = min(MIN_POSITIVE_TASKS, len(ACTIVE_EVALUATION_CLUSTERS))
    dose_response_tasks = int(np.sum([row["dose_response_pass"] for row in stable_rows]))
    positive_control_tasks = int(
        np.sum([row["positive_controls_pass"] for row in stable_rows])
    )
    bounded_tasks = int(
        np.sum([
            row["overshoot_pass"]
            and row["patch_norm_pass"]
            and row["output_alignment_pass"]
            for row in stable_rows
        ])
    )
    safeguard_tasks = int(
        np.sum([
            row["dose_response_pass"]
            and row["positive_controls_pass"]
            and row["overshoot_pass"]
            and row["patch_norm_pass"]
            and row["output_alignment_pass"]
            for row in stable_rows
        ])
    )
    safety_values = [row["minimum_output_cosine"] for row in stable_rows]
    safety = min(safety_values) if safety_values else None
    observed_null_statistic = (
        observed_null_statistic_value
        if np.isfinite(observed_null_statistic_value) else None
    )
    null_95 = null_95_value if np.isfinite(null_95_value) else None
    null_gain = null_gain_value if np.isfinite(null_gain_value) else None
    null_p_value = null_empirical_p if np.isfinite(null_empirical_p) else None
    result = {
        "mediation": mediation,
        "estimand": (
            "mean of hidden-path sufficiency recovery and full-action necessity loss"
        ),
        "inference_conditioning": (
            "construction-threshold-stable denominator tasks; at least 7/8 required"
        ),
        "gain_over_complement": complement_gain,
        "coherent_null_test": {
            "observed_task_equal_mean": observed_null_statistic,
            "null_draws": int(len(null_statistics)),
            "null_95th_percentile": null_95,
            "gain_over_null_95th_percentile": null_gain,
            "one_sided_empirical_p": null_p_value,
            "task_gains_over_task_null_95th_percentile": (
                causal_task_null_gains
            ),
            "positive_task_gains": causal_positive_null_tasks,
        },
        "minimum_output_cosine": safety,
        "stable_denominator_tasks": len(stable_rows),
        "dose_response_tasks": dose_response_tasks,
        "positive_control_tasks": positive_control_tasks,
        "bounded_aligned_tasks": bounded_tasks,
        "all_causal_safeguards_tasks": safeguard_tasks,
        "interaction_task_equal_recovery": interaction_summary,
        "interaction_task_counts": interaction_task_counts,
        "required_interaction_strata": list(REQUIRED_INTERACTION_STRATA),
        "qualified_interaction_strata": qualified_interactions,
        "no_interaction_stratum_sign_reversal": interaction_reversal_pass,
        "required_positive_tasks": required_positive,
        "thresholds": {
            "minimum_mediation": MIN_CAUSAL_MEDIATION,
            "minimum_gain_over_null": MIN_CAUSAL_GAIN_OVER_NULL,
            "minimum_gain_over_complement": MIN_CAUSAL_GAIN_OVER_COMPLEMENT,
            "minimum_positive_control_recovery": MIN_POSITIVE_CONTROL_RECOVERY,
            "maximum_positive_control_recovery": MAX_POSITIVE_CONTROL_RECOVERY,
            "minimum_denominator_coherence": MIN_CAUSAL_DENOMINATOR_COHERENCE,
            "minimum_output_reconstruction": MIN_CAUSAL_OUTPUT_RECONSTRUCTION,
            "minimum_output_displacement_cosine": MIN_CAUSAL_OUTPUT_DISPLACEMENT_COSINE,
            "maximum_recovery": MAX_CAUSAL_RECOVERY,
            "maximum_patch_to_natural_norm_ratio": MAX_PATCH_TO_NATURAL_NORM_RATIO,
        },
        "passed": bool(
            RUN_MODE == "pilot"
            and len(stable_rows) >= required_positive
            and mediation["task_equal_mean"] >= MIN_CAUSAL_MEDIATION
            and mediation["one_sided_95_lower"] > 0
            and null_gain is not None
            and null_gain >= MIN_CAUSAL_GAIN_OVER_NULL
            and null_p_value is not None
            and null_p_value <= 0.05
            and causal_positive_null_tasks >= required_positive
            and complement_gain["task_equal_mean"]
            >= MIN_CAUSAL_GAIN_OVER_COMPLEMENT
            and complement_gain["one_sided_95_lower"] > 0
            and safeguard_tasks >= required_positive
            and interaction_reversal_pass
        ),
    }
    write_json(CAUSAL_DIR / "causal_gate.json", result)
    return result


if not PIPELINE_FAILED:
    try:
        CAUSAL_ROWS = run_causal_mediation(PRIMARY_FRAME)
        CAUSAL_GATE = causal_gate(CAUSAL_ROWS)
        print(json.dumps(CAUSAL_GATE, indent=2))
        memory_report("causal_complete")
    except Exception:
        record_failure("causal_mediation")
'''


robustness = r'''# Temporal reuse, gauge identities, and interaction-regime summaries.


def gauge_identity_checks():
    record = CONSTRUCTION_RECORDS[0]
    payload = load_shard(record["record_id"], PRIMARY_HORIZON)
    g = payload["G"].astype(np.float64).reshape(
        len(TRAIN_QUERY_PAIRS), 256, EXPECTED_CARRIER_CHANNELS
    )
    b = payload["B"].astype(np.float64).reshape(
        256, EXPECTED_CARRIER_CHANNELS, -1
    )
    reference = g.reshape(len(g), -1) @ b.reshape(-1, b.shape[-1])
    rows = []
    for draw, seed in enumerate(NULL_SEEDS[: min(4, len(NULL_SEEDS))]):
        rotation = haar_rotation(EXPECTED_CARRIER_CHANNELS, int(seed) + 5000)
        b_rotated = np.einsum("cd,tdp->tcp", rotation, b, optimize=True)
        g_rotated = np.einsum("qtc,dc->qtd", g, rotation, optimize=True)
        observed = g_rotated.reshape(len(g), -1) @ b_rotated.reshape(
            -1, b.shape[-1]
        )
        rows.append(
            {
                "draw": draw,
                "seed": int(seed) + 5000,
                "relative_hankel_error": relative_error(observed, reference),
            }
        )
    write_csv(ANALYSIS_DIR / "gauge_identity_rows.csv", rows)
    result = {
        "maximum_relative_hankel_error": max(
            row["relative_hankel_error"] for row in rows
        ),
        "passed": bool(
            max(row["relative_hankel_error"] for row in rows) <= 1e-8
        ),
        "identity": "G'=GQ^T, B'=QB, therefore G'B'=GB",
        "evidence_scope": (
            "algebraic orthogonal-coordinate identity only; not an end-to-end gauge test"
        ),
    }
    write_json(ANALYSIS_DIR / "gauge_identity_gate.json", result)
    if not result["passed"]:
        raise RuntimeError(f"gauge identity failed: {result}")
    return result


def temporal_extension():
    if not RUN_TEMPORAL_EXTENSION:
        return {"status": "SKIPPED_BY_CONFIGURATION", "passed": False}, []
    horizon = 1
    run_exact_extraction(EVALUATION_RECORDS, horizon)
    rows = evaluation_transfer_rows(
        PRIMARY_FRAME, horizon, include_transported=False
    )
    summary = summarize_task_metric(
        rows, "sparse_reconstruction", BOOTSTRAP_SEED + 401
    )
    gain = summarize_task_metric(
        rows, "gain_over_strongest_null", BOOTSTRAP_SEED + 402
    )
    null_test = aggregate_null_test(
        rows, horizon, "global_haar", "sparse_reconstruction"
    )
    required_positive = min(MIN_POSITIVE_TASKS, len(ACTIVE_EVALUATION_CLUSTERS))
    nondegenerate_tasks = count_nondegenerate_tasks(
        rows, TEMPORAL_CALIBRATED_FLOORS
    )
    result = {
        "status": "H3_FRAME_APPLIED_WITHOUT_REFITTING_AT_H1",
        "reconstruction": summary,
        "gain_over_strongest_null": gain,
        "coherent_null_test": null_test,
        "nondegenerate_tasks": nondegenerate_tasks,
        "required_positive_tasks": required_positive,
        "construction_calibrated_floors": TEMPORAL_CALIBRATED_FLOORS,
        "minimum_reconstruction": MIN_TEMPORAL_RECONSTRUCTION,
        "passed": bool(
            RUN_MODE == "pilot"
            and summary["task_equal_median"] >= MIN_TEMPORAL_RECONSTRUCTION
            and null_test["gain_over_null_95th_percentile"]
            >= MIN_GAIN_OVER_NULL_95
            and null_test["one_sided_empirical_p"] <= 0.05
            and null_test["positive_task_gains"] >= required_positive
            and nondegenerate_tasks >= required_positive
        ),
    }
    write_json(ANALYSIS_DIR / "temporal_reuse_gate.json", result)
    return result, rows


def interaction_strata(rows):
    output = {}
    primary = [row for row in rows if row["condition"] == "sparse" and row["dose"] == 1.0]
    for label in sorted({row["interaction_type"] for row in primary}):
        task_values = []
        for task_id in sorted({int(row["task_id"]) for row in primary}):
            selected = [
                row for row in primary
                if row["interaction_type"] == label and int(row["task_id"]) == task_id
            ]
            if not selected:
                continue
            path_values = []
            for path_test in ["sufficiency", "necessity"]:
                path_rows = [row for row in selected if row["path_test"] == path_test]
                denominator = float(np.sum([row["query_denominator"] for row in path_rows]))
                effect = float(np.sum([row["query_effect"] for row in path_rows]))
                if abs(denominator) > 1e-12:
                    path_values.append(effect / denominator)
            if len(path_values) == 2:
                task_values.append(float(np.mean(path_values)))
        output[label] = {
            "tasks": len(task_values),
            "task_equal_symmetric_mediated_recovery": (
                float(np.mean(task_values)) if task_values else None
            ),
        }
    result = {
        "strata": output,
        "no_preregistered_sign_reversal": CAUSAL_GATE[
            "no_interaction_stratum_sign_reversal"
        ],
        "task_equal": True,
        "note": "PushT labels come from cumulative simulator contact counts.",
    }
    write_json(ANALYSIS_DIR / "interaction_strata.json", result)
    return result


if not PIPELINE_FAILED:
    try:
        GAUGE_GATE = gauge_identity_checks()
        TEMPORAL_GATE, TEMPORAL_ROWS = temporal_extension()
        INTERACTION_STRATA = interaction_strata(CAUSAL_ROWS)
        memory_report("robustness_complete")
    except Exception:
        record_failure("robustness")
'''


decision_and_plots = r'''# Apply the frozen claim ladder mechanically and plot compact diagnostics.
verify_executed_notebook_through(
    "# Apply the frozen claim ladder mechanically and plot compact diagnostics."
)


def final_decision():
    def qualify(label):
        return (
            label
            if SOURCE_IDENTITY.get("confirmation_eligible", False)
            else f"EXPLORATORY_{label}"
        )

    if PIPELINE_FAILED:
        return "PIPELINE_FAILURE"
    if RUN_MODE == "smoke":
        return "SMOKE_COMPLETE_NO_SCIENTIFIC_DECISION"
    if not REPRESENTATION_GATE["passed"]:
        if REPRESENTATION_GATE.get("atlas_passed", False):
            return qualify("STATE_CONDITIONED_PREDICTIVE_ATLAS_CANDIDATE")
        return "NO_EVIDENCE_FOR_THIS_FROZEN_GLOBAL_FRAME"
    if not CAUSAL_GATE["passed"]:
        return "NO_CAUSAL_CONFIRMATION_FOR_THIS_FRAME"
    if RUN_TEMPORAL_EXTENSION and not TEMPORAL_GATE["passed"]:
        return qualify("H3_PREDICTIVE_CONTROL_BUNDLE_WITHOUT_H1_REUSE")
    return qualify("JEPA_NATIVE_PREDICTIVE_CONTROL_BUNDLE_CANDIDATE")


DECISION = final_decision()
DECISION_PAYLOAD = {
    "decision": DECISION,
    "run_mode": RUN_MODE,
    "evidence_status": (
        "SOURCE_BOUND_PREDECLARED_PILOT"
        if SOURCE_IDENTITY.get("confirmation_eligible", False)
        else EVIDENCE_STATUS
    ),
    "source_identity": SOURCE_IDENTITY,
    "selected_block": SELECTED_BLOCK if not PIPELINE_FAILED else None,
    "representation_gate": REPRESENTATION_GATE if not PIPELINE_FAILED else None,
    "causal_gate": CAUSAL_GATE if not PIPELINE_FAILED else None,
    "temporal_gate": TEMPORAL_GATE if not PIPELINE_FAILED else None,
    "workspace_claim_authorized": False,
    "j_space_claim_authorized": False,
    "reader_scope": "state-specific oracle target-aligned assay",
    "action_tangent_rank_per_split": 2,
    "next_if_pass": (
        "source-bound untouched multi-reader/native-planner confirmation"
        if DECISION in {
            "JEPA_NATIVE_PREDICTIVE_CONTROL_BUNDLE_CANDIDATE",
            "EXPLORATORY_JEPA_NATIVE_PREDICTIVE_CONTROL_BUNDLE_CANDIDATE",
        } else None
    ),
}
write_json(OUT / "stage14_decision.json", DECISION_PAYLOAD)

if not PIPELINE_FAILED:
    with (ANALYSIS_DIR / "carrier_scan_task_rows.csv").open() as handle:
        scan_rows = list(csv.DictReader(handle))
    blocks = CANDIDATE_BLOCKS
    scan_means = [
        np.mean(
            [float(row["mean_cosine"]) for row in scan_rows if int(row["block"]) == block]
        )
        for block in blocks
    ]
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot([block + 1 for block in blocks], scan_means, marker="o")
    axes[0].axvline(SELECTED_BLOCK + 1, color="black", linestyle="--", alpha=0.6)
    axes[0].set(title="Construction carrier scan", xlabel="block", ylabel="task-mean cosine")

    axes[1].scatter(
        [row["strongest_null_reconstruction"] for row in PRIMARY_TRANSFER_ROWS],
        [row["sparse_reconstruction"] for row in PRIMARY_TRANSFER_ROWS],
        alpha=0.8,
    )
    limits = axes[1].get_xlim()
    axes[1].plot(limits, limits, color="black", linestyle="--", alpha=0.5)
    axes[1].set(
        title="Held-out transfer",
        xlabel="strongest null reconstruction",
        ylabel="sparse frame reconstruction",
    )

    causal_primary = [
        row for row in CAUSAL_ROWS
        if row["condition"] == "sparse"
        and row["path_test"] == "sufficiency"
        and row["dose"] == 1.0
    ]
    axes[2].hist(
        [row["query_recovery"] for row in causal_primary], bins=12, alpha=0.8
    )
    axes[2].axvline(0, color="black", linestyle="--")
    axes[2].set(title="Hidden-path sufficiency", xlabel="recovery", ylabel="count")
    figure.tight_layout()
    figure.savefig(PLOT_DIR / "stage14_summary.png", dpi=180)
    plt.show()

print(json.dumps(DECISION_PAYLOAD, indent=2))
'''


packaging = r'''# Package compact audit evidence separately from recomputable large shards.
verify_executed_notebook_through(
    "# Package compact audit evidence separately from recomputable large shards."
)
DECISION_PAYLOAD["source_identity"] = SOURCE_IDENTITY
write_json(OUT / "stage14_decision.json", DECISION_PAYLOAD)
write_json(OUT / "timings.json", TIMINGS)
memory_report("final")
if not PIPELINE_FAILED:
    (OUT / "FAILURE_TRACE.txt").write_text("NONE\n")

compact_root = OUT / "compact_bundle"
archive_path = OUT / "stage14_predictive_control_result_bundle.zip"
for stale_file in [
    archive_path,
    OUT / "compact_manifest.json",
    OUT / "full_manifest.json",
]:
    stale_file.unlink(missing_ok=True)
if compact_root.exists():
    shutil.rmtree(compact_root)
if EVIDENCE_DIR.exists():
    shutil.rmtree(EVIDENCE_DIR)
EVIDENCE_DIR.mkdir()

if not PIPELINE_FAILED:
    evidence_arrays = [
        "record_id", "task_id", "split", "horizon", "query_pairs",
        "query_separations", "train_query_separations", "G", "G_train",
        "written", "written_train", "hankel", "hankel_energy",
        "hankel_energy_per_cell",
        "context", "metric_invariance_error",
    ]
    for record in EVALUATION_RECORDS:
        for horizon in HORIZONS:
            source = shard_path(record["record_id"], horizon)
            if not source.exists():
                continue
            with np.load(source) as payload:
                atomic_npz(
                    EVIDENCE_DIR / source.name,
                    **{name: payload[name] for name in evidence_arrays},
                )
    write_json(
        EVIDENCE_DIR / "evidence_manifest.json",
        manifest_rows(EVIDENCE_DIR),
    )

full_rows = manifest_rows(OUT, excluded_roots=[ASSET_DIR])
write_json(OUT / "full_manifest.json", full_rows)

compact_root.mkdir()
for name in [
    "config.json", "versions.json", "source_identity.json", "stage14_decision.json",
    "timings.json", "memory.json", "FAILURE_TRACE.txt", "hook_identity.json",
    "benchmark.json", "restore_test.json", "pretrained_asset_verification.json",
    "goal_binding_smoke.json", "full_manifest.json",
]:
    source = OUT / name
    if source.exists():
        shutil.copy2(source, compact_root / name)
for directory in [
    DESIGN_DIR, ANALYSIS_DIR, CAUSAL_DIR, EVIDENCE_DIR, PLOT_DIR, LOG_DIR
]:
    if directory.exists():
        shutil.copytree(directory, compact_root / directory.name)

compact_manifest_payload = {
    "convention": (
        "files describes every archive member except compact_manifest.json itself"
    ),
    "files": manifest_rows(compact_root),
}
write_json(compact_root / "compact_manifest.json", compact_manifest_payload)
shutil.copy2(compact_root / "compact_manifest.json", OUT / "compact_manifest.json")

archive = shutil.make_archive(
    str(OUT / "stage14_predictive_control_result_bundle"),
    "zip",
    root_dir=compact_root,
)
print(f"RUN_STATUS: {'FAILED' if PIPELINE_FAILED else DECISION}")
print(f"Compact result archive: {archive}")

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
    model_and_targets,
    carrier_scan,
    exact_write_read,
    sparse_frame,
    causal_mediation,
    robustness,
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
    code(model_and_targets),
    code(carrier_scan),
    code(exact_write_read),
    code(sparse_frame),
    code(causal_mediation),
    code(robustness),
    code(decision_and_plots),
    code(packaging),
]
for index, cell in enumerate(cells):
    cell["id"] = f"stage14-{index:02d}"

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
