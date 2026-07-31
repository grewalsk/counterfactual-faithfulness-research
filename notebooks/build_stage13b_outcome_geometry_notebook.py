import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STAGE11 = ROOT / "11_action_response_geometry_pilot.ipynb"
GEOMETRY = ROOT.parent / "src/cf_faithfulness/stage13b_geometry.py"
TARGET = ROOT / "13b_outcome_geometry_diagnostic.ipynb"


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


base = json.loads(STAGE11.read_text())
stage11_helpers = "".join(base["cells"][4]["source"])
simulator_helpers = function_sources(
    stage11_helpers,
    [
        "write_csv",
        "atomic_npz",
        "to_model_observation",
        "configure_repo",
        "make_environment",
        "wall_visual",
        "reset_environment",
        "rollout_branch",
        "exact_restore_test",
    ],
)
simulator_helpers = simulator_helpers.replace(
    'task = TASKS["Wall"][0]',
    'raise RuntimeError("Stage 13b supports PushT only")',
)

geometry_helpers = function_sources(
    GEOMETRY.read_text(),
    [
        "array_sha256",
        "frozen_action_bank",
        "hash_sorted_ids",
        "effective_rank_from_gram",
        "fit_dual_pca",
        "weighted_dual_pca",
        "reconstruction_by_groups",
        "select_rank_one_se",
        "covariance_shaped_coordinates",
        "basis_overlap",
        "rbf_weights",
        "one_sided_t_lower",
        "exact_positive_sign_test",
        "hierarchical_bootstrap_indices",
        "hierarchical_bootstrap_means",
        "countsketch_numpy",
    ],
)


configuration = r'''# SINGLE CONFIGURATION BLOCK — edit only RUN_MODE if needed.
RUN_MODE = "full"  # "smoke" validates plumbing; "full" is the Stage 13b study.
MOUNT_DRIVE = True
DOWNLOAD_RESULTS = True
OUTPUT_DIR = "/content/counterfactual_faithfulness_stage13b"
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/counterfactual_faithfulness_stage13b"

SEED = 13301
MODEL_NAME = "jepa_wm_pusht"
ENVIRONMENT = "PushT"
HORIZONS = [1, 3]
NUM_STATES = 80
ACTIONS_PER_STATE = 13
FRAMESKIP = 5
TARGET_STEPS = HORIZONS
EXPECTED_ACTION_SHA256 = (
    "802129bd281fdd2d42a395429e5a0e00df2dc10032b339ecb8bdc8b2521d9fd2"
)

E0_CONSTRUCTION_STATE_IDS = [5, 28, 45, 68, 77, 94, 0, 30]
E0_CALIBRATION_STATE_IDS = [34, 63, 80, 39]
E0_CACHE_DIR = (
    "/content/drive/MyDrive/counterfactual_faithfulness_stage13_jow/"
    "screen_042eeb57e3a6/target_tokens"
)
E1_TASKS = 12
E1_STATES_PER_TASK = 4
CONFIRMATION_TASKS = 8
CONFIRMATION_STATES_PER_TASK = 4
CONFIRMATION_TASK_ID_OFFSET = 100
CONFIRMATION_STATE_SEED = 13337
CONFIRMATION_GOALS = [
    [206.0, 218.0, -1.15],
    [258.0, 206.0, -0.65],
    [306.0, 210.0, -0.10],
    [326.0, 252.0, 0.35],
    [304.0, 312.0, 0.70],
    [264.0, 330.0, 1.10],
    [212.0, 304.0, 0.55],
    [194.0, 252.0, -0.20],
]

RANKS = [1, 2, 4, 6, 8, 12, 16, 24, 32]
LOCAL_BANDWIDTH_MULTIPLIERS = [0.5, 1.0, 2.0, 4.0]
ORACLE_TRAIN_ACTIONS = [0, 1, 3, 5, 7, 9, 11]
ORACLE_TEST_ACTIONS = [2, 4, 6, 8, 10, 12]
LEARNING_CURVE_TASK_COUNTS = [2, 4, 6, 8, 10]

OUTCOME_PROJECTION_DIM = 128
ORIGINAL_SKETCH_SEED = 13119
ADDITIONAL_SKETCH_SEEDS = [
    13217, 13331, 13441, 13553, 13669, 13781, 13901, 14011,
    14149, 14251, 14369, 14479, 14591, 14713, 14821, 14939,
    15053, 15173, 15287, 15401, 15511, 15629, 15739, 15859,
    15971, 16087, 16193, 16301, 16411, 16529, 16633, 16747,
]
SKETCH_SEEDS = [ORIGINAL_SKETCH_SEED] + ADDITIONAL_SKETCH_SEEDS
NULL_ROOT_SEED = 13379
LOCAL_PERMUTATION_SEED = 13411
BOOTSTRAP_SEED = 13441
NULL_DRAWS = 1024
LOCAL_PERMUTATIONS = 1024
BOOTSTRAP_DRAWS = 10000

MIN_NULL_GAIN = 0.05
MIN_NULL_RATIO = 1.25
MIN_TASK_LOWER_BOUND = 0.0
MIN_POSITIVE_TASKS = 7
MIN_RANK32_PRESERVATION = 0.80
MIN_TASK_SPLIT_OVERLAP = 0.50
NULL_OVERLAP_PERCENTILE = 97.5
MIN_LOCAL_OVER_GLOBAL_GAIN = 0.05
MAX_COMPACT_RANK = 8

EVIDENCE_STATUS = "PREREGISTERED_STAGE13B_REPRESENTATION_DIAGNOSTIC"
STOP_BEFORE_JACOBIANS = True
DIRTY_PATCH = ""

EXPERIMENT_REPOSITORY = "grewalsk/counterfactual-faithfulness-research"
EXPERIMENT_SOURCE_REF = "stage13b-v1"
EXPERIMENT_NOTEBOOK_PATH = "notebooks/13b_outcome_geometry_diagnostic.ipynb"
EXPERIMENT_BUILDER_PATH = (
    "notebooks/build_stage13b_outcome_geometry_notebook.py"
)

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
ASSET_COMMIT = "2326e74556f6f81db2560e4396f4cc52c16a28f4"
ASSET_REPOSITORY = EXPERIMENT_REPOSITORY
ASSET_SPECS = {
    "pusht_design.npz": {
        "path": "results/bundles/stage12_result_bundle/pusht_design.npz",
        "sha256": "bcacd9bf640c0ae779fba1a2e80d3bcb8bc9c77e5c53baf0a9f49878a24d65da",
    },
    "tasks.json": {
        "path": "results/bundles/stage12_result_bundle/tasks.json",
        "sha256": "90826f91eeaf34214a2aadafe816177f2730a8e2b22ac83c032a869d114e86c7",
    },
    "split_manifest.json": {
        "path": "results/bundles/stage12_result_bundle/split_manifest.json",
        "sha256": "d04d0db10a31970c311b307473fb1dc268816bf055f714ef4ac9be6c810284c6",
    },
}

if RUN_MODE == "smoke":
    ACTIVE_E1_TASKS = 4
    ACTIVE_E1_STATES_PER_TASK = 2
    ACTIVE_CONFIRMATION_TASKS = 2
    ACTIVE_CONFIRMATION_STATES_PER_TASK = 2
    ACTIVE_RANKS = [1, 2, 4]
    ACTIVE_SKETCH_SEEDS = SKETCH_SEEDS[:2]
    ACTIVE_NULL_DRAWS = 8
    ACTIVE_LOCAL_PERMUTATIONS = 8
    ACTIVE_BOOTSTRAP_DRAWS = 128
    NUM_STATES = 12
elif RUN_MODE == "full":
    ACTIVE_E1_TASKS = E1_TASKS
    ACTIVE_E1_STATES_PER_TASK = E1_STATES_PER_TASK
    ACTIVE_CONFIRMATION_TASKS = CONFIRMATION_TASKS
    ACTIVE_CONFIRMATION_STATES_PER_TASK = CONFIRMATION_STATES_PER_TASK
    ACTIVE_RANKS = RANKS
    ACTIVE_SKETCH_SEEDS = SKETCH_SEEDS
    ACTIVE_NULL_DRAWS = NULL_DRAWS
    ACTIVE_LOCAL_PERMUTATIONS = LOCAL_PERMUTATIONS
    ACTIVE_BOOTSTRAP_DRAWS = BOOTSTRAP_DRAWS
else:
    raise ValueError("RUN_MODE must be 'smoke' or 'full'")

assert HORIZONS == [1, 3]
assert ACTIONS_PER_STATE == 13
assert len(CONFIRMATION_GOALS) == CONFIRMATION_TASKS
assert len(SKETCH_SEEDS) == 33 and len(set(SKETCH_SEEDS)) == 33
assert STOP_BEFORE_JACOBIANS
'''


introduction = r'''# Stage 13b: outcome-geometry diagnostic

This notebook asks a narrower question than the failed Stage 13 JOW screen:
**is PushT outcome geometry globally compact, horizon-specific, predictably
state-local, higher-rank, or an artifact of the original projection?**

The full run freezes a goal-independent 13-action design, selects predictive
rank on 48 previously unused states from the 12 existing tasks, writes a hash-
bound preregistration, and only then encodes 32 untouched states from eight new
task clusters. Full `256 x 384` target-token effects are primary. CountSketch,
diagonal standardization, pooled horizons, an RBF local model, and an oracle
within-state model are diagnostics and cannot silently replace the primary
analysis.

There is no Jacobian or causal intervention code here. A failure or
inconclusive representation gate ends the JOW line before that compute.
`smoke` mode validates plumbing and can never issue a scientific promotion.
'''


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
]
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", *PINNED],
    check=True,
)
'''


setup = r'''import csv
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
import zipfile
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
        print("Google Drive already mounted.")
        return
    drive.mount(mountpoint, timeout_ms=600_000)
    if not Path(mountpoint, "MyDrive").is_dir():
        raise RuntimeError("Google Drive mount did not produce MyDrive")


if MOUNT_DRIVE:
    ensure_colab_drive()
    OUTPUT_DIR = DRIVE_OUTPUT_DIR

CONFIG = {
    key: value
    for key, value in globals().copy().items()
    if key.isupper()
    and isinstance(value, (str, int, float, bool, list, tuple, dict))
}
CONFIG["PINNED"] = PINNED
RUN_SIGNATURE = hashlib.sha256(
    json.dumps(CONFIG, sort_keys=True).encode()
).hexdigest()
OUT = Path(OUTPUT_DIR) / f"{RUN_MODE}_{RUN_SIGNATURE[:12]}"
ASSET_DIR = OUT / "assets"
DESIGN_DIR = OUT / "design"
TRUTH_DIR = OUT / "truth"
TARGET_DIR = OUT / "target_tokens"
MATRIX_DIR = OUT / "effect_matrices"
ANALYSIS_DIR = OUT / "analysis"
PLOT_DIR = OUT / "plots"
LOG_DIR = OUT / "logs"
for directory in [
    OUT, ASSET_DIR, DESIGN_DIR, TRUTH_DIR, TARGET_DIR, MATRIX_DIR,
    ANALYSIS_DIR, PLOT_DIR, LOG_DIR,
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
log = logging.getLogger("stage13b")


def write_json(path, payload):
    temporary = Path(path).with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256_file(path, chunk_bytes=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity():
    reference = EXPERIMENT_SOURCE_REF
    remote = f"https://github.com/{EXPERIMENT_REPOSITORY}.git"
    listing = subprocess.check_output(
        ["git", "ls-remote", remote, reference], text=True
    ).strip()
    if not listing:
        raise RuntimeError(f"could not resolve experiment source ref {reference}")
    resolved_commit = listing.split()[0]
    base = (
        "https://raw.githubusercontent.com/"
        f"{EXPERIMENT_REPOSITORY}/{reference}/"
    )
    rows = {}
    for name, relative in [
        ("notebook", EXPERIMENT_NOTEBOOK_PATH),
        ("builder", EXPERIMENT_BUILDER_PATH),
    ]:
        with urllib.request.urlopen(base + relative) as response:
            payload = response.read()
        rows[name] = {
            "path": relative,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    return {
        "repository": EXPERIMENT_REPOSITORY,
        "source_ref": reference,
        "resolved_commit": resolved_commit,
        "files": rows,
        "dirty_patch_declared": bool(DIRTY_PATCH),
        "dirty_patch_sha256": hashlib.sha256(DIRTY_PATCH.encode()).hexdigest(),
    }


VERSIONS = {
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "numpy": np.__version__,
    "cuda_runtime": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "gpu_total_gib": round(
        torch.cuda.get_device_properties(0).total_memory / 2**30, 2
    ),
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
        "gpu_peak_allocated_gib": float(
            torch.cuda.max_memory_allocated() / 2**30
        ),
    }
    MEMORY.append(row)
    write_json(OUT / "memory.json", MEMORY)
    log.info("memory %s", row)
    return row


def download_asset(name):
    specification = ASSET_SPECS[name]
    destination = ASSET_DIR / name
    if destination.exists() and sha256_file(destination) == specification["sha256"]:
        return destination
    if destination.exists():
        destination.unlink()
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


for core_asset in ASSET_SPECS:
    download_asset(core_asset)
print(json.dumps(VERSIONS, indent=2))
print(f"Durable run directory: {OUT}")
memory_report("startup")
'''


helpers = (
    "ONE_SIDED_T95_DF7 = 1.894578605061305\n\n"
    + geometry_helpers
    + "\n\n\n"
    + simulator_helpers
    + r'''


def verify_pretrained_assets():
    rows = []
    for name, expected in EXPECTED_PRETRAINED_ASSET_SHA256.items():
        matching = [
            path
            for path in CACHE_ROOT.rglob(name)
            if sha256_file(path) == expected
        ]
        if not matching:
            raise RuntimeError(f"verified pretrained asset not found: {name}")
        rows.append(
            {"name": name, "path": str(matching[0]), "sha256": expected}
        )
    write_json(OUT / "pretrained_asset_verification.json", rows)
    return rows


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
    verify_pretrained_assets()
    return model, preprocessor


def active_records(records, task_count, states_per_task):
    task_ids = sorted({int(row["task_id"]) for row in records})[:task_count]
    output = []
    for task_id in task_ids:
        selected = [row for row in records if int(row["task_id"]) == task_id]
        output.extend(selected[:states_per_task])
    return output


def record_manifest_row(row):
    return {
        key: (
            value.tolist() if isinstance(value, np.ndarray) else value
        )
        for key, value in row.items()
    }


def state_effect_rows(state_indices):
    return [
        np.arange(index * ACTIONS_PER_STATE, (index + 1) * ACTIONS_PER_STATE)
        for index in state_indices
    ]


def combine_seed(*values):
    digest = hashlib.sha256("|".join(str(x) for x in values).encode()).digest()
    return int.from_bytes(digest[:8], "little")
'''
)


design = r'''# Freeze all tasks, states, actions, random schedules, and split labels.
ACTION_LABELS, ACTION_BANK = frozen_action_bank(
    primitive_steps=max(HORIZONS) * FRAMESKIP
)
ACTION_HASH = array_sha256(ACTION_BANK)
if ACTION_HASH != EXPECTED_ACTION_SHA256:
    raise AssertionError(
        f"frozen action hash changed: {ACTION_HASH} != {EXPECTED_ACTION_SHA256}"
    )

with np.load(ASSET_DIR / "pusht_design.npz") as payload:
    LEGACY_DESIGN = {name: payload[name].copy() for name in payload.files}
LEGACY_TASKS = [
    row
    for row in json.loads((ASSET_DIR / "tasks.json").read_text())
    if row["environment"] == ENVIRONMENT
]
LEGACY_TASKS_BY_ID = {int(row["task_id"]): row for row in LEGACY_TASKS}
E0_STATE_IDS = E0_CONSTRUCTION_STATE_IDS + E0_CALIBRATION_STATE_IDS

E1_RECORDS = []
for task_id in sorted(LEGACY_TASKS_BY_ID):
    candidates = [
        state_id
        for state_id, value in enumerate(LEGACY_DESIGN["task_ids"])
        if int(value) == task_id and state_id not in E0_STATE_IDS
    ]
    selected = hash_sorted_ids(
        candidates,
        [LEGACY_DESIGN["states"][state_id] for state_id in candidates],
        E1_STATES_PER_TASK,
        f"stage13b-e1-task-{task_id}-seed-{SEED}",
    )
    for within_task, state_id in enumerate(selected):
        E1_RECORDS.append(
            {
                "split": "e1_method_selection",
                "record_id": int(state_id),
                "source_state_id": int(state_id),
                "task_id": int(task_id),
                "within_task": int(within_task),
                "evaluation_seed": int(SEED + 1000 + state_id),
                "state": LEGACY_DESIGN["states"][state_id].astype(np.float64),
            }
        )


def confirmation_tasks_and_states():
    tasks = []
    records = []
    record_id = 0
    for task_index, goal in enumerate(CONFIRMATION_GOALS):
        task_id = CONFIRMATION_TASK_ID_OFFSET + task_index
        task = {
            "environment": ENVIRONMENT,
            "task_id": task_id,
            "task_name": f"stage13b_untouched_pusht_{task_index:02d}",
            "goal": list(goal),
            "split": "untouched_confirmation",
        }
        tasks.append(task)
        goal_xy = np.asarray(goal[:2], dtype=np.float64)
        for within_task in range(CONFIRMATION_STATES_PER_TASK):
            rng = np.random.default_rng(
                CONFIRMATION_STATE_SEED + task_index * 1000 + within_task
            )
            for _ in range(1000):
                radial = rng.uniform(85.0, 120.0)
                polar = rng.uniform(-np.pi, np.pi)
                block = goal_xy + radial * np.array(
                    [np.cos(polar), np.sin(polar)]
                )
                direction = (goal_xy - block) / np.linalg.norm(goal_xy - block)
                agent_distance = rng.uniform(58.0, 80.0)
                agent = block - agent_distance * direction
                if (
                    np.all(block > 90.0)
                    and np.all(block < 422.0)
                    and np.all(agent > 35.0)
                    and np.all(agent < 477.0)
                ):
                    break
            else:
                raise RuntimeError("could not construct confirmation state")
            state = np.array(
                [
                    agent[0], agent[1], block[0], block[1],
                    rng.uniform(-0.65, 0.65), 0.0, 0.0,
                ],
                dtype=np.float64,
            )
            records.append(
                {
                    "split": "untouched_confirmation",
                    "record_id": record_id,
                    "source_state_id": None,
                    "task_id": task_id,
                    "within_task": within_task,
                    "evaluation_seed": int(
                        CONFIRMATION_STATE_SEED + task_index * 1000 + within_task
                    ),
                    "state": state,
                }
            )
            record_id += 1
    return tasks, records


CONFIRMATION_TASK_DEFINITIONS, CONFIRMATION_RECORDS = (
    confirmation_tasks_and_states()
)
CONFIRMATION_TASK_IDS = sorted(
    int(row["task_id"]) for row in CONFIRMATION_TASK_DEFINITIONS
)
CONFIRMATION_TASK_HALVES = [
    CONFIRMATION_TASK_IDS[: CONFIRMATION_TASKS // 2],
    CONFIRMATION_TASK_IDS[CONFIRMATION_TASKS // 2 :],
]
old_goals = {tuple(row["goal"]) for row in LEGACY_TASKS}
if any(tuple(row["goal"]) in old_goals for row in CONFIRMATION_TASK_DEFINITIONS):
    raise AssertionError("confirmation task duplicates an existing goal")

ACTIVE_E1_RECORDS = active_records(
    E1_RECORDS, ACTIVE_E1_TASKS, ACTIVE_E1_STATES_PER_TASK
)
ACTIVE_CONFIRMATION_RECORDS = active_records(
    CONFIRMATION_RECORDS,
    ACTIVE_CONFIRMATION_TASKS,
    ACTIVE_CONFIRMATION_STATES_PER_TASK,
)
TASKS_BY_ID = {
    **LEGACY_TASKS_BY_ID,
    **{int(row["task_id"]): row for row in CONFIRMATION_TASK_DEFINITIONS},
}

null_rng = np.random.default_rng(NULL_ROOT_SEED)
NULL_SEED_SCHEDULE = null_rng.integers(
    0, np.iinfo(np.uint64).max, size=NULL_DRAWS, dtype=np.uint64
)
permutation_rng = np.random.default_rng(LOCAL_PERMUTATION_SEED)
LOCAL_TASK_PERMUTATION_SCHEDULE = np.stack(
    [
        permutation_rng.permutation(CONFIRMATION_TASKS)
        for _ in range(LOCAL_PERMUTATIONS)
    ]
).astype(np.int16)
BOOTSTRAP_TASK_INDICES, BOOTSTRAP_STATE_INDICES = (
    hierarchical_bootstrap_indices(
        CONFIRMATION_TASKS,
        CONFIRMATION_STATES_PER_TASK,
        BOOTSTRAP_DRAWS,
        BOOTSTRAP_SEED,
    )
)

np.savez_compressed(
    DESIGN_DIR / "random_schedules.npz",
    null_seeds=NULL_SEED_SCHEDULE,
    local_task_permutations=LOCAL_TASK_PERMUTATION_SCHEDULE,
    bootstrap_task_indices=BOOTSTRAP_TASK_INDICES,
    bootstrap_state_indices=BOOTSTRAP_STATE_INDICES,
)
np.savez_compressed(
    DESIGN_DIR / "stage13b_design.npz",
    action_bank=ACTION_BANK,
    action_labels=np.asarray(ACTION_LABELS),
    e1_states=np.stack([row["state"] for row in E1_RECORDS]),
    e1_state_ids=np.asarray([row["record_id"] for row in E1_RECORDS]),
    e1_task_ids=np.asarray([row["task_id"] for row in E1_RECORDS]),
    confirmation_states=np.stack([row["state"] for row in CONFIRMATION_RECORDS]),
    confirmation_state_ids=np.asarray(
        [row["record_id"] for row in CONFIRMATION_RECORDS]
    ),
    confirmation_task_ids=np.asarray(
        [row["task_id"] for row in CONFIRMATION_RECORDS]
    ),
)
write_json(
    DESIGN_DIR / "task_state_manifest.json",
    {
        "e0_observed_state_ids": E0_STATE_IDS,
        "e1_records": [record_manifest_row(row) for row in E1_RECORDS],
        "confirmation_tasks": CONFIRMATION_TASK_DEFINITIONS,
        "confirmation_records": [
            record_manifest_row(row) for row in CONFIRMATION_RECORDS
        ],
        "active_e1_record_ids": [row["record_id"] for row in ACTIVE_E1_RECORDS],
        "active_confirmation_record_ids": [
            row["record_id"] for row in ACTIVE_CONFIRMATION_RECORDS
        ],
    },
)
DESIGN_FREEZE = {
    "created_before_any_stage13b_target_encoding": True,
    "action_sha256": ACTION_HASH,
    "action_labels": ACTION_LABELS,
    "unique_nonnoop_prefixes": {"5": 12, "15": 12},
    "design_npz_sha256": sha256_file(DESIGN_DIR / "stage13b_design.npz"),
    "task_state_manifest_sha256": sha256_file(
        DESIGN_DIR / "task_state_manifest.json"
    ),
    "random_schedules_sha256": sha256_file(
        DESIGN_DIR / "random_schedules.npz"
    ),
    "confirmation_overlap_task_halves": CONFIRMATION_TASK_HALVES,
    "source_identity": SOURCE_IDENTITY,
}
write_json(DESIGN_DIR / "design_freeze.json", DESIGN_FREEZE)
write_json(
    DESIGN_DIR / "design_freeze_certificate.json",
    {"sha256": sha256_file(DESIGN_DIR / "design_freeze.json")},
)
print(json.dumps(DESIGN_FREEZE, indent=2))
'''


e0_audit = r'''# E0: optional observed-cache audit. This can never confirm Stage 13b.
def e0_cache_audit():
    cache = Path(E0_CACHE_DIR)
    expected = E0_STATE_IDS
    paths = {state_id: cache / f"state_{state_id:04d}.npz" for state_id in expected}
    missing = [state_id for state_id, path in paths.items() if not path.exists()]
    if missing:
        result = {
            "status": "OBSERVED_CACHE_NOT_AVAILABLE",
            "missing_state_ids": missing,
            "confirmation_eligible": False,
        }
        write_json(ANALYSIS_DIR / "e0_observed_cache_audit.json", result)
        return result

    horizon_rows = {}
    duplicate_rows = []
    sketch_rows = []
    for horizon_index, horizon in enumerate(HORIZONS):
        matrices = []
        state_row_slices = {}
        cursor = 0
        for state_id in expected:
            actions = LEGACY_DESIGN["action_bank"][
                state_id, :, : horizon * FRAMESKIP
            ]
            groups = defaultdict(list)
            for action_index, action in enumerate(actions):
                groups[array_sha256(action)].append(action_index)
            representatives = [values[0] for values in groups.values()]
            with np.load(paths[state_id]) as payload:
                tokens = payload["true_tokens"].astype(np.float32)
            values = tokens[representatives, horizon_index].reshape(
                len(representatives), -1
            )
            values -= values.mean(axis=0, keepdims=True)
            matrices.append(values)
            state_row_slices[state_id] = np.arange(
                cursor, cursor + len(representatives)
            )
            cursor += len(representatives)
            duplicate_rows.append(
                {
                    "state_id": state_id,
                    "horizon": horizon,
                    "original_actions": int(len(actions)),
                    "unique_prefix_actions": int(len(representatives)),
                    "groups": [list(map(int, row)) for row in groups.values()],
                }
            )
        matrix = np.concatenate(matrices)
        construction_rows = np.concatenate(
            [state_row_slices[state_id] for state_id in E0_CONSTRUCTION_STATE_IDS]
        )
        calibration_rows = np.concatenate(
            [state_row_slices[state_id] for state_id in E0_CALIBRATION_STATE_IDS]
        )
        train = matrix[construction_rows]
        test = matrix[calibration_rows]
        fitted = fit_dual_pca(train @ train.T, max_rank=max(ACTIVE_RANKS))
        state_groups = []
        for state_id in E0_CALIBRATION_STATE_IDS:
            global_rows = state_row_slices[state_id]
            lookup = {value: index for index, value in enumerate(calibration_rows)}
            state_groups.append([lookup[int(value)] for value in global_rows])
        fractions = reconstruction_by_groups(
            test @ train.T,
            test @ test.T,
            fitted["coefficients"],
            ACTIVE_RANKS,
            state_groups,
        )
        horizon_rows[str(horizon)] = {
            "effective_rank": effective_rank_from_gram(train @ train.T),
            "rank_curve_task_unadjusted_mean": {
                str(rank): float(np.nanmean(fractions[:, index]))
                for index, rank in enumerate(ACTIVE_RANKS)
            },
        }
        for seed in ACTIVE_SKETCH_SEEDS:
            projected = countsketch_numpy(matrix, OUTCOME_PROJECTION_DIM, seed)
            projected_train = projected[construction_rows]
            projected_test = projected[calibration_rows]
            sketch_fit = fit_dual_pca(
                projected_train @ projected_train.T,
                max_rank=min(8, max(ACTIVE_RANKS)),
            )
            sketch_fraction = reconstruction_by_groups(
                projected_test @ projected_train.T,
                projected_test @ projected_test.T,
                sketch_fit["coefficients"],
                [min(8, max(ACTIVE_RANKS))],
                state_groups,
            )
            sketch_rows.append(
                {
                    "horizon": horizon,
                    "seed": int(seed),
                    "rank": int(min(8, max(ACTIVE_RANKS))),
                    "mean_fraction": float(np.nanmean(sketch_fraction)),
                }
            )
    write_csv(ANALYSIS_DIR / "e0_duplicate_action_groups.csv", duplicate_rows)
    write_csv(ANALYSIS_DIR / "e0_sketch_seed_rows.csv", sketch_rows)
    result = {
        "status": "EXPLORATORY_OBSERVED_CACHE_AUDITED",
        "confirmation_eligible": False,
        "horizons": horizon_rows,
        "cache_path": str(cache),
    }
    write_json(ANALYSIS_DIR / "e0_observed_cache_audit.json", result)
    return result


try:
    E0_AUDIT = e0_cache_audit()
except Exception:
    # E0 is optional and cannot block the untouched study.
    E0_AUDIT = {
        "status": "OBSERVED_CACHE_AUDIT_ERROR",
        "trace": traceback.format_exc(),
        "confirmation_eligible": False,
    }
    write_json(ANALYSIS_DIR / "e0_observed_cache_audit.json", E0_AUDIT)
print(json.dumps(E0_AUDIT, indent=2))
'''


data_and_encoding = r'''# Generate/encode E1 only. Confirmation remains unopened.
def truth_path(split, record_id):
    return TRUTH_DIR / split / f"state_{int(record_id):04d}.npz"


def target_path(split, record_id):
    return TARGET_DIR / split / f"state_{int(record_id):04d}.npz"


def generate_truth(split, records):
    started = time.perf_counter()
    destination_dir = TRUTH_DIR / split
    destination_dir.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records):
        destination = truth_path(split, record["record_id"])
        state_hash = array_sha256(record["state"])
        if destination.exists():
            with np.load(destination) as previous:
                if (
                    str(previous["state_sha256"]) == state_hash
                    and str(previous["action_sha256"]) == ACTION_HASH
                ):
                    log.info("%s truth resume: %s", split, destination.name)
                    continue
            raise RuntimeError(f"incompatible truth shard {destination}")
        task = TASKS_BY_ID[int(record["task_id"])]
        initials = []
        initial_proprios = []
        future_visual = []
        future_proprio = []
        endpoints = []
        for action_index, actions in enumerate(ACTION_BANK):
            initial, _, observations, states, _, _ = rollout_branch(
                REPO,
                ENVIRONMENT,
                task,
                record["state"],
                actions,
                record["evaluation_seed"] * 100 + action_index,
            )
            initials.append(initial["visual"])
            initial_proprios.append(initial["proprio"])
            future_visual.append(
                [observations[horizon]["visual"] for horizon in HORIZONS]
            )
            future_proprio.append(
                [observations[horizon]["proprio"] for horizon in HORIZONS]
            )
            endpoints.append([states[horizon] for horizon in HORIZONS])
        if not all(np.array_equal(initials[0], item) for item in initials[1:]):
            raise AssertionError("action branches changed the initial render")
        if not all(
            np.array_equal(initial_proprios[0], item)
            for item in initial_proprios[1:]
        ):
            raise AssertionError("action branches changed initial proprio")
        atomic_npz(
            destination,
            split=np.asarray(split),
            record_id=np.asarray(record["record_id"], dtype=np.int64),
            source_state_id=np.asarray(
                -1 if record["source_state_id"] is None else record["source_state_id"],
                dtype=np.int64,
            ),
            task_id=np.asarray(record["task_id"], dtype=np.int64),
            within_task=np.asarray(record["within_task"], dtype=np.int64),
            state_sha256=np.asarray(state_hash),
            action_sha256=np.asarray(ACTION_HASH),
            initial_state=np.asarray(record["state"], dtype=np.float64),
            initial_visual=np.asarray(initials[0], dtype=np.uint8),
            initial_proprio=np.asarray(initial_proprios[0], dtype=np.float32),
            selected_actions=ACTION_BANK,
            future_visual=np.asarray(future_visual, dtype=np.uint8),
            future_proprio=np.asarray(future_proprio, dtype=np.float32),
            endpoint_states=np.asarray(endpoints, dtype=np.float32),
        )
        write_json(
            OUT / f"{split}_truth_progress.json",
            {
                "completed": index + 1,
                "total": len(records),
                "last_record_id": int(record["record_id"]),
            },
        )
        log.info("%s simulator truth %d/%d", split, index + 1, len(records))
    TIMINGS[f"{split}_truth_seconds"] = time.perf_counter() - started


def encode_targets(split, records):
    started = time.perf_counter()
    destination_dir = TARGET_DIR / split
    destination_dir.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records):
        source = truth_path(split, record["record_id"])
        destination = target_path(split, record["record_id"])
        source_hash = sha256_file(source)
        if destination.exists():
            with np.load(destination) as previous:
                if str(previous["truth_sha256"]) == source_hash:
                    log.info("%s target resume: %s", split, destination.name)
                    continue
            raise RuntimeError(f"incompatible target shard {destination}")
        with np.load(source) as truth:
            visual = truth["future_visual"]
            proprio = truth["future_proprio"]
            initial_visual = truth["initial_visual"]
            initial_proprio = truth["initial_proprio"]
        with torch.inference_mode():
            encoded = MODEL.encode(to_model_observation(visual, proprio))
            initial_encoded = MODEL.encode(
                to_model_observation(initial_visual, initial_proprio)
            )
        tokens = (
            encoded["visual"][:, :, 0]
            .reshape(
                ACTIONS_PER_STATE,
                len(HORIZONS),
                -1,
                encoded["visual"].shape[-1],
            )
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        initial_tokens = (
            initial_encoded["visual"][:, :, 0]
            .reshape(-1, initial_encoded["visual"].shape[-1])
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        query = initial_tokens.mean(axis=0)
        atomic_npz(
            destination,
            true_tokens=tokens.astype(np.float16),
            initial_query=query.astype(np.float32),
            truth_sha256=np.asarray(source_hash),
            task_id=np.asarray(record["task_id"], dtype=np.int64),
            within_task=np.asarray(record["within_task"], dtype=np.int64),
            action_sha256=np.asarray(ACTION_HASH),
        )
        write_json(
            OUT / f"{split}_target_progress.json",
            {
                "completed": index + 1,
                "total": len(records),
                "last_record_id": int(record["record_id"]),
            },
        )
    TIMINGS[f"{split}_target_encoding_seconds"] = (
        time.perf_counter() - started
    )


if not PIPELINE_FAILED:
    try:
        REPO = configure_repo()
        first = ACTIVE_E1_RECORDS[0]
        write_json(
            OUT / "restore_test.json",
            exact_restore_test(
                REPO,
                ENVIRONMENT,
                TASKS_BY_ID[first["task_id"]],
                first["state"],
                ACTION_BANK[1],
            ),
        )
        generate_truth("e1_method_selection", ACTIVE_E1_RECORDS)
        MODEL, PREPROCESSOR = load_frozen_model()
        encode_targets("e1_method_selection", ACTIVE_E1_RECORDS)
        memory_report("e1_encoded")
    except Exception:
        record_failure("e1_data_and_encoding")
'''


analysis_helpers = r'''# Streamed native-token geometry and grouped statistics.
def split_records(split):
    if split == "e1_method_selection":
        return ACTIVE_E1_RECORDS
    if split == "untouched_confirmation":
        return ACTIVE_CONFIRMATION_RECORDS
    raise KeyError(split)


def effect_matrix_path(split, horizon):
    return MATRIX_DIR / f"{split}_h{horizon}_native_effects.npy"


def query_path(split):
    return MATRIX_DIR / f"{split}_initial_queries.npy"


def build_effect_matrices(split, records):
    marker = MATRIX_DIR / f"{split}_effects_complete.json"
    expected_shape = (
        len(records) * ACTIONS_PER_STATE,
        256 * 384,
    )
    if marker.exists():
        previous = json.loads(marker.read_text())
        valid = bool(
            previous.get("run_signature") == RUN_SIGNATURE
            and all(
                effect_matrix_path(split, horizon).exists()
                and tuple(
                    np.load(
                        effect_matrix_path(split, horizon), mmap_mode="r"
                    ).shape
                ) == expected_shape
                for horizon in HORIZONS
            )
            and query_path(split).exists()
        )
        if valid:
            return previous
        raise RuntimeError(f"incompatible effect-matrix cache for {split}")
    matrices = {
        horizon: np.lib.format.open_memmap(
            effect_matrix_path(split, horizon),
            mode="w+",
            dtype=np.float16,
            shape=expected_shape,
        )
        for horizon in HORIZONS
    }
    queries = np.empty((len(records), 384), dtype=np.float32)
    source_hashes = []
    for state_index, record in enumerate(records):
        source = target_path(split, record["record_id"])
        source_hashes.append(sha256_file(source))
        with np.load(source) as payload:
            tokens = payload["true_tokens"].astype(np.float32)
            queries[state_index] = payload["initial_query"]
        rows = slice(
            state_index * ACTIONS_PER_STATE,
            (state_index + 1) * ACTIONS_PER_STATE,
        )
        for horizon_index, horizon in enumerate(HORIZONS):
            values = tokens[:, horizon_index].reshape(ACTIONS_PER_STATE, -1)
            values -= values.mean(axis=0, keepdims=True)
            matrices[horizon][rows] = values.astype(np.float16)
    for matrix in matrices.values():
        matrix.flush()
    np.save(query_path(split), queries)
    result = {
        "run_signature": RUN_SIGNATURE,
        "shape": list(expected_shape),
        "dtype": "float16",
        "target_shard_hashes": source_hashes,
        "within_state_centered": True,
    }
    write_json(marker, result)
    return result


def feature_scale(path, chunk_features=2048):
    matrix = np.load(path, mmap_mode="r")
    dimension = matrix.shape[1]
    scale = np.empty(dimension, dtype=np.float32)
    for start in range(0, dimension, chunk_features):
        stop = min(start + chunk_features, dimension)
        block = np.asarray(matrix[:, start:stop], dtype=np.float32)
        block_scale = np.std(block, axis=0)
        floor = float(np.median(block_scale) * 1e-3 + 1e-6)
        scale[start:stop] = np.maximum(block_scale, floor)
    return scale


def streamed_gram(left_path, right_path=None, scale=None, chunk_features=2048):
    left = np.load(left_path, mmap_mode="r")
    right = left if right_path is None else np.load(right_path, mmap_mode="r")
    if left.shape[1] != right.shape[1]:
        raise ValueError("feature dimensions differ")
    output = np.zeros((left.shape[0], right.shape[0]), dtype=np.float64)
    for start in range(0, left.shape[1], chunk_features):
        stop = min(start + chunk_features, left.shape[1])
        a = np.asarray(left[:, start:stop], dtype=np.float32)
        b = a if right is left else np.asarray(
            right[:, start:stop], dtype=np.float32
        )
        if scale is not None:
            divisor = np.asarray(scale[start:stop], dtype=np.float32)
            a = a / divisor[None]
            b = a if right is left else b / divisor[None]
        output += a @ b.T
    return output


def load_or_build_gram(name, left_path, right_path=None, scale=None):
    destination = ANALYSIS_DIR / f"{name}.npy"
    if destination.exists():
        return np.load(destination)
    started = time.perf_counter()
    gram = streamed_gram(left_path, right_path, scale=scale)
    np.save(destination, gram)
    TIMINGS[f"gram_{name}_seconds"] = time.perf_counter() - started
    return gram


def task_state_indices(records):
    output = defaultdict(list)
    for index, row in enumerate(records):
        output[int(row["task_id"])].append(index)
    return dict(output)


def local_groups(state_count):
    return [
        np.arange(index * ACTIONS_PER_STATE, (index + 1) * ACTIONS_PER_STATE)
        for index in range(state_count)
    ]


def task_scores_from_state_scores(state_scores, records):
    by_task = task_state_indices(records)
    task_ids = sorted(by_task)
    scores = np.stack(
        [np.nanmean(state_scores[by_task[task_id]], axis=0) for task_id in task_ids]
    )
    return task_ids, scores


def native_loto(gram, records, horizon, label, include_null):
    started = time.perf_counter()
    by_task = task_state_indices(records)
    task_ids = sorted(by_task)
    state_scores = np.full(
        (len(records), len(ACTIVE_RANKS)), np.nan, dtype=np.float64
    )
    null_scores = (
        np.full(
            (
                len(records),
                len(ACTIVE_RANKS),
                ACTIVE_NULL_DRAWS,
            ),
            np.nan,
            dtype=np.float32,
        )
        if include_null
        else None
    )
    fold_rows = []
    for fold_index, heldout_task in enumerate(task_ids):
        test_states = by_task[heldout_task]
        train_states = [
            index for index in range(len(records)) if index not in test_states
        ]
        train_rows = np.concatenate(state_effect_rows(train_states))
        test_rows = np.concatenate(state_effect_rows(test_states))
        train_gram = gram[np.ix_(train_rows, train_rows)]
        test_train = gram[np.ix_(test_rows, train_rows)]
        test_gram = gram[np.ix_(test_rows, test_rows)]
        full_fit = fit_dual_pca(
            train_gram, max_rank=len(train_rows)
        )
        fractions = reconstruction_by_groups(
            test_train,
            test_gram,
            full_fit["coefficients"],
            ACTIVE_RANKS,
            local_groups(len(test_states)),
        )
        state_scores[test_states] = fractions
        fold_rows.append(
            {
                "fold": fold_index,
                "heldout_task": int(heldout_task),
                "train_states": len(train_states),
                "test_states": len(test_states),
                "effective_rank": effective_rank_from_gram(train_gram),
            }
        )
        if include_null:
            maximum_rank = max(ACTIVE_RANKS)
            eig_scores = test_train @ full_fit["coefficients"]
            for draw in range(ACTIVE_NULL_DRAWS):
                seed = combine_seed(
                    int(NULL_SEED_SCHEDULE[draw]), label, horizon, heldout_task
                )
                coordinates = covariance_shaped_coordinates(
                    full_fit["eigenvalues"], seed, maximum_rank
                )
                scores = eig_scores @ coordinates
                for local_state, global_state in enumerate(test_states):
                    rows = local_groups(len(test_states))[local_state]
                    denominator = float(np.trace(test_gram[np.ix_(rows, rows)]))
                    cumulative = np.cumsum(scores[rows] ** 2, axis=1).sum(axis=0)
                    for rank_index, rank in enumerate(ACTIVE_RANKS):
                        null_scores[global_state, rank_index, draw] = (
                            cumulative[rank - 1] / denominator
                        )
    selected_task_ids, task_scores = task_scores_from_state_scores(
        state_scores, records
    )
    selection = select_rank_one_se(task_scores, ACTIVE_RANKS)
    np.savez_compressed(
        ANALYSIS_DIR / f"e1_{label}_h{horizon}_loto.npz",
        state_scores=state_scores.astype(np.float32),
        task_ids=np.asarray(selected_task_ids),
        task_scores=task_scores.astype(np.float32),
        ranks=np.asarray(ACTIVE_RANKS),
        null_scores=null_scores if null_scores is not None else np.asarray([]),
    )
    write_csv(
        ANALYSIS_DIR / f"e1_{label}_h{horizon}_folds.csv", fold_rows
    )
    result = {
        "label": label,
        "horizon": horizon,
        "selected_rank": selection["selected_rank"],
        "best_rank": selection["best_rank"],
        "best_mean": selection["best_mean"],
        "best_standard_error": selection["best_standard_error"],
        "one_se_threshold": selection["one_se_threshold"],
        "rank_curve": [
            {
                "rank": int(rank),
                "task_equal_mean": float(selection["means"][index]),
                "task_standard_error": float(
                    selection["standard_errors"][index]
                ),
            }
            for index, rank in enumerate(ACTIVE_RANKS)
        ],
    }
    write_json(ANALYSIS_DIR / f"e1_{label}_h{horizon}_selection.json", result)
    TIMINGS[f"e1_{label}_h{horizon}_loto_seconds"] = (
        time.perf_counter() - started
    )
    return result, state_scores, null_scores


def standardize_queries(train_queries, test_queries):
    mean = np.mean(train_queries, axis=0)
    scale = np.std(train_queries, axis=0)
    floor = np.median(scale) * 1e-3 + 1e-6
    scale = np.maximum(scale, floor)
    return (train_queries - mean) / scale, (test_queries - mean) / scale, mean, scale


def median_nonzero_distance(queries):
    squared = np.sum(
        (queries[:, None] - queries[None]) ** 2, axis=-1
    )
    values = np.sqrt(squared[np.triu_indices(len(queries), 1)])
    values = values[values > 0]
    return float(np.median(values))


def local_loto(gram, queries, records, horizon):
    started = time.perf_counter()
    by_task = task_state_indices(records)
    task_ids = sorted(by_task)
    curves = np.full(
        (
            len(LOCAL_BANDWIDTH_MULTIPLIERS),
            len(records),
            len(ACTIVE_RANKS),
        ),
        np.nan,
        dtype=np.float32,
    )
    for heldout_task in task_ids:
        test_states = by_task[heldout_task]
        train_states = [
            index for index in range(len(records)) if index not in test_states
        ]
        train_rows = np.concatenate(state_effect_rows(train_states))
        train_gram = gram[np.ix_(train_rows, train_rows)]
        train_query, test_query, _, _ = standardize_queries(
            queries[train_states], queries[test_states]
        )
        base_bandwidth = median_nonzero_distance(train_query)
        for multiplier_index, multiplier in enumerate(
            LOCAL_BANDWIDTH_MULTIPLIERS
        ):
            bandwidth = base_bandwidth * multiplier
            for local_index, state_index in enumerate(test_states):
                state_rows = state_effect_rows([state_index])[0]
                state_train = gram[np.ix_(state_rows, train_rows)]
                state_gram = gram[np.ix_(state_rows, state_rows)]
                state_weights = rbf_weights(
                    test_query[local_index], train_query, bandwidth
                )
                row_weights = np.repeat(state_weights, ACTIONS_PER_STATE)
                fitted = weighted_dual_pca(
                    train_gram,
                    row_weights,
                    max_rank=max(ACTIVE_RANKS),
                )
                fraction = reconstruction_by_groups(
                    state_train,
                    state_gram,
                    fitted["coefficients"],
                    ACTIVE_RANKS,
                    [np.arange(ACTIONS_PER_STATE)],
                )[0]
                curves[multiplier_index, state_index] = fraction
    candidate_rows = []
    for multiplier_index, multiplier in enumerate(
        LOCAL_BANDWIDTH_MULTIPLIERS
    ):
        _, task_scores = task_scores_from_state_scores(
            curves[multiplier_index], records
        )
        means = np.nanmean(task_scores, axis=0)
        ses = np.nanstd(task_scores, axis=0, ddof=1) / np.sqrt(len(task_scores))
        for rank_index, rank in enumerate(ACTIVE_RANKS):
            candidate_rows.append(
                {
                    "multiplier": float(multiplier),
                    "rank": int(rank),
                    "mean": float(means[rank_index]),
                    "se": float(ses[rank_index]),
                    "multiplier_index": multiplier_index,
                    "rank_index": rank_index,
                }
            )
    best = max(candidate_rows, key=lambda row: row["mean"])
    threshold = best["mean"] - best["se"]
    eligible = [row for row in candidate_rows if row["mean"] >= threshold]
    selected = sorted(
        eligible, key=lambda row: (row["rank"], -row["multiplier"])
    )[0]
    result = {
        "horizon": horizon,
        "selected_rank": selected["rank"],
        "selected_bandwidth_multiplier": selected["multiplier"],
        "best_rank": best["rank"],
        "best_bandwidth_multiplier": best["multiplier"],
        "best_mean": best["mean"],
        "best_standard_error": best["se"],
        "one_se_threshold": threshold,
        "candidate_rows": [
            {key: value for key, value in row.items() if not key.endswith("_index")}
            for row in candidate_rows
        ],
    }
    np.savez_compressed(
        ANALYSIS_DIR / f"e1_local_h{horizon}_loto.npz",
        curves=curves,
        ranks=np.asarray(ACTIVE_RANKS),
        bandwidth_multipliers=np.asarray(LOCAL_BANDWIDTH_MULTIPLIERS),
    )
    write_json(ANALYSIS_DIR / f"e1_local_h{horizon}_selection.json", result)
    TIMINGS[f"e1_local_h{horizon}_loto_seconds"] = time.perf_counter() - started
    return result


def oracle_action_split(path, records, horizon):
    matrix = np.load(path, mmap_mode="r")
    rows = []
    for state_index, record in enumerate(records):
        block = np.asarray(
            matrix[
                state_index * ACTIONS_PER_STATE:
                (state_index + 1) * ACTIONS_PER_STATE
            ],
            dtype=np.float32,
        )
        train = block[ORACLE_TRAIN_ACTIONS]
        test = block[ORACLE_TEST_ACTIONS]
        fitted = fit_dual_pca(
            train @ train.T, max_rank=min(max(ACTIVE_RANKS), len(train))
        )
        valid_ranks = [rank for rank in ACTIVE_RANKS if rank <= fitted["rank"]]
        fraction = reconstruction_by_groups(
            test @ train.T,
            test @ test.T,
            fitted["coefficients"],
            valid_ranks,
            [np.arange(len(test))],
        )[0]
        for rank, value in zip(valid_ranks, fraction):
            rows.append(
                {
                    "state_index": state_index,
                    "record_id": int(record["record_id"]),
                    "task_id": int(record["task_id"]),
                    "horizon": horizon,
                    "rank": int(rank),
                    "reconstruction_fraction": float(value),
                }
            )
    write_csv(ANALYSIS_DIR / f"e1_oracle_h{horizon}.csv", rows)
    return rows


def build_sketch_cache(split, records):
    destination = ANALYSIS_DIR / f"{split}_sketch_effects.npz"
    if destination.exists():
        with np.load(destination) as payload:
            return payload["sketches"].astype(np.float32)
    sketches = np.empty(
        (
            len(HORIZONS),
            len(ACTIVE_SKETCH_SEEDS),
            len(records) * ACTIONS_PER_STATE,
            OUTCOME_PROJECTION_DIM,
        ),
        dtype=np.float16,
    )
    for horizon_index, horizon in enumerate(HORIZONS):
        matrix = np.load(effect_matrix_path(split, horizon), mmap_mode="r")
        for seed_index, seed in enumerate(ACTIVE_SKETCH_SEEDS):
            sketches[horizon_index, seed_index] = countsketch_numpy(
                matrix, OUTCOME_PROJECTION_DIM, seed
            ).astype(np.float16)
    np.savez_compressed(
        destination,
        sketches=sketches,
        seeds=np.asarray(ACTIVE_SKETCH_SEEDS),
        horizons=np.asarray(HORIZONS),
    )
    return sketches.astype(np.float32)


def primal_rank_curve_loto(matrix, records, standardize):
    by_task = task_state_indices(records)
    state_scores = np.full(
        (len(records), len(ACTIVE_RANKS)), np.nan, dtype=np.float64
    )
    for heldout_task, test_states in by_task.items():
        train_states = [
            index for index in range(len(records)) if index not in test_states
        ]
        train_rows = np.concatenate(state_effect_rows(train_states))
        test_rows = np.concatenate(state_effect_rows(test_states))
        train = matrix[train_rows].astype(np.float64)
        test = matrix[test_rows].astype(np.float64)
        if standardize:
            scale = np.std(train, axis=0)
            scale = np.maximum(scale, np.median(scale) * 1e-3 + 1e-6)
            train = train / scale
            test = test / scale
        covariance = train.T @ train
        eigenvalues, axes = np.linalg.eigh(covariance)
        axes = axes[:, np.argsort(eigenvalues)[::-1]]
        scores = test @ axes[:, : max(ACTIVE_RANKS)]
        for local_state, global_state in enumerate(test_states):
            rows = local_groups(len(test_states))[local_state]
            denominator = float(np.sum(test[rows] ** 2))
            cumulative = np.cumsum(scores[rows] ** 2, axis=1).sum(axis=0)
            for rank_index, rank in enumerate(ACTIVE_RANKS):
                state_scores[global_state, rank_index] = (
                    cumulative[rank - 1] / denominator
                )
    _, task_scores = task_scores_from_state_scores(state_scores, records)
    return state_scores, task_scores, select_rank_one_se(task_scores, ACTIVE_RANKS)


def e1_sketch_analysis(sketches, records):
    selections = []
    rows = []
    for horizon_index, horizon in enumerate(HORIZONS):
        for seed_index, seed in enumerate(ACTIVE_SKETCH_SEEDS):
            matrix = sketches[horizon_index, seed_index]
            for standardize in [False, True]:
                state_scores, task_scores, selection = primal_rank_curve_loto(
                    matrix, records, standardize
                )
                selections.append(
                    {
                        "horizon": horizon,
                        "seed": int(seed),
                        "preprocessing": (
                            "diagonal_standardized" if standardize else "raw"
                        ),
                        "selected_rank": selection["selected_rank"],
                        "best_rank": selection["best_rank"],
                    }
                )
                for rank_index, rank in enumerate(ACTIVE_RANKS):
                    rows.append(
                        {
                            "horizon": horizon,
                            "seed": int(seed),
                            "preprocessing": (
                                "diagonal_standardized" if standardize else "raw"
                            ),
                            "rank": int(rank),
                            "task_equal_mean": float(
                                np.nanmean(task_scores[:, rank_index])
                            ),
                            "task_standard_error": float(
                                np.nanstd(task_scores[:, rank_index], ddof=1)
                                / np.sqrt(len(task_scores))
                            ),
                        }
                    )
    write_csv(ANALYSIS_DIR / "e1_sketch_rank_curves.csv", rows)
    write_json(ANALYSIS_DIR / "e1_sketch_selections.json", selections)
    return selections


def task_learning_curve(gram, records, selected_rank, horizon):
    by_task = task_state_indices(records)
    ordered_tasks = sorted(
        by_task,
        key=lambda task: hashlib.sha256(
            f"stage13b-learning-{SEED}-{task}".encode()
        ).hexdigest(),
    )
    rows = []
    for count in LEARNING_CURVE_TASK_COUNTS:
        if count >= len(ordered_tasks):
            continue
        train_tasks = ordered_tasks[:count]
        test_tasks = ordered_tasks[count:]
        train_states = [index for task in train_tasks for index in by_task[task]]
        test_states = [index for task in test_tasks for index in by_task[task]]
        train_rows = np.concatenate(state_effect_rows(train_states))
        test_rows = np.concatenate(state_effect_rows(test_states))
        fitted = fit_dual_pca(
            gram[np.ix_(train_rows, train_rows)], max_rank=selected_rank
        )
        fractions = reconstruction_by_groups(
            gram[np.ix_(test_rows, train_rows)],
            gram[np.ix_(test_rows, test_rows)],
            fitted["coefficients"],
            [selected_rank],
            local_groups(len(test_states)),
        )[:, 0]
        state_lookup = {state: value for state, value in zip(test_states, fractions)}
        task_values = [
            np.mean([state_lookup[state] for state in by_task[task]])
            for task in test_tasks
        ]
        rows.append(
            {
                "horizon": horizon,
                "training_tasks": count,
                "training_states": len(train_states),
                "test_tasks": len(test_tasks),
                "rank": selected_rank,
                "task_equal_reconstruction": float(np.mean(task_values)),
            }
        )
    return rows


def state_learning_curve(gram, records, selected_rank, horizon):
    by_task = task_state_indices(records)
    ordered = {
        task: sorted(
            states,
            key=lambda state: hashlib.sha256(
                f"stage13b-state-learning-{SEED}-{records[state]['record_id']}".encode()
            ).hexdigest(),
        )
        for task, states in by_task.items()
    }
    rows = []
    maximum_training_states = min(len(states) for states in ordered.values()) - 1
    for states_per_task in range(1, maximum_training_states + 1):
        train_states = [
            state
            for states in ordered.values()
            for state in states[:states_per_task]
        ]
        test_by_task = {
            task: states[states_per_task:]
            for task, states in ordered.items()
        }
        test_states = [state for states in test_by_task.values() for state in states]
        train_rows = np.concatenate(state_effect_rows(train_states))
        test_rows = np.concatenate(state_effect_rows(test_states))
        fitted = fit_dual_pca(
            gram[np.ix_(train_rows, train_rows)], max_rank=selected_rank
        )
        fractions = reconstruction_by_groups(
            gram[np.ix_(test_rows, train_rows)],
            gram[np.ix_(test_rows, test_rows)],
            fitted["coefficients"],
            [selected_rank],
            local_groups(len(test_states)),
        )[:, 0]
        lookup = {state: value for state, value in zip(test_states, fractions)}
        task_values = [
            np.mean([lookup[state] for state in states])
            for states in test_by_task.values()
        ]
        rows.append(
            {
                "horizon": horizon,
                "training_states_per_task": states_per_task,
                "training_states": len(train_states),
                "test_states": len(test_states),
                "rank": selected_rank,
                "task_equal_reconstruction": float(np.mean(task_values)),
            }
        )
    return rows
'''


e1_analysis = r'''# E1 method selection, followed by a fail-closed confirmation freeze.
E1_SELECTIONS = {}
if not PIPELINE_FAILED:
    try:
        started = time.perf_counter()
        build_effect_matrices("e1_method_selection", ACTIVE_E1_RECORDS)
        E1_QUERIES = np.load(query_path("e1_method_selection"))
        E1_GRAMS = {}
        E1_SCALES = {}
        for horizon in HORIZONS:
            path = effect_matrix_path("e1_method_selection", horizon)
            E1_GRAMS[(horizon, "raw")] = load_or_build_gram(
                f"e1_h{horizon}_raw_gram", path
            )
            scale = feature_scale(path)
            E1_SCALES[horizon] = scale
            np.save(ANALYSIS_DIR / f"e1_h{horizon}_diagonal_scale.npy", scale)
            E1_GRAMS[(horizon, "standardized")] = load_or_build_gram(
                f"e1_h{horizon}_standardized_gram", path, scale=scale
            )
        E1_CROSS_HORIZON = load_or_build_gram(
            "e1_h1_h3_cross_gram",
            effect_matrix_path("e1_method_selection", 1),
            effect_matrix_path("e1_method_selection", 3),
        )
        learning_rows = []
        state_learning_rows = []
        for horizon in HORIZONS:
            raw_selection, _, _ = native_loto(
                E1_GRAMS[(horizon, "raw")],
                ACTIVE_E1_RECORDS,
                horizon,
                "native_raw",
                include_null=True,
            )
            standardized_selection, _, _ = native_loto(
                E1_GRAMS[(horizon, "standardized")],
                ACTIVE_E1_RECORDS,
                horizon,
                "native_standardized",
                include_null=False,
            )
            local_selection = local_loto(
                E1_GRAMS[(horizon, "raw")],
                E1_QUERIES,
                ACTIVE_E1_RECORDS,
                horizon,
            )
            oracle_action_split(
                effect_matrix_path("e1_method_selection", horizon),
                ACTIVE_E1_RECORDS,
                horizon,
            )
            learning_rows.extend(
                task_learning_curve(
                    E1_GRAMS[(horizon, "raw")],
                    ACTIVE_E1_RECORDS,
                    raw_selection["selected_rank"],
                    horizon,
                )
            )
            state_learning_rows.extend(
                state_learning_curve(
                    E1_GRAMS[(horizon, "raw")],
                    ACTIVE_E1_RECORDS,
                    raw_selection["selected_rank"],
                    horizon,
                )
            )
            E1_SELECTIONS[str(horizon)] = {
                "native_raw": raw_selection,
                "native_standardized": standardized_selection,
                "local": local_selection,
            }
        write_csv(
            ANALYSIS_DIR / "e1_task_learning_curve.csv",
            learning_rows,
            fieldnames=[
                "horizon", "training_tasks", "training_states", "test_tasks",
                "rank", "task_equal_reconstruction",
            ],
        )
        write_csv(
            ANALYSIS_DIR / "e1_state_learning_curve.csv",
            state_learning_rows,
            fieldnames=[
                "horizon", "training_states_per_task", "training_states",
                "test_states", "rank", "task_equal_reconstruction",
            ],
        )
        write_json(
            ANALYSIS_DIR / "analytic_haar_expectation.json",
            {
                str(rank): rank / float(256 * 384)
                for rank in ACTIVE_RANKS
            },
        )
        E1_SKETCHES = build_sketch_cache(
            "e1_method_selection", ACTIVE_E1_RECORDS
        )
        E1_SKETCH_SELECTIONS = e1_sketch_analysis(
            E1_SKETCHES, ACTIVE_E1_RECORDS
        )
        query_standardized, _, query_mean, query_scale = standardize_queries(
            E1_QUERIES, E1_QUERIES
        )
        full_query_base_bandwidth = median_nonzero_distance(query_standardized)
        np.savez_compressed(
            ANALYSIS_DIR / "e1_frozen_local_query_parameters.npz",
            mean=query_mean.astype(np.float32),
            scale=query_scale.astype(np.float32),
            base_bandwidth=np.asarray(full_query_base_bandwidth),
        )
        frozen_basis_hashes = {}
        for horizon in HORIZONS:
            local_selection = E1_SELECTIONS[str(horizon)]["local"]
            local_selection["full_e1_base_bandwidth"] = full_query_base_bandwidth
            local_selection["absolute_bandwidth"] = (
                full_query_base_bandwidth
                * local_selection["selected_bandwidth_multiplier"]
            )
            for preprocessing in ["raw", "standardized"]:
                selection_key = (
                    "native_raw"
                    if preprocessing == "raw"
                    else "native_standardized"
                )
                selected_rank = int(
                    E1_SELECTIONS[str(horizon)][selection_key]["selected_rank"]
                )
                fit = fit_dual_pca(
                    E1_GRAMS[(horizon, preprocessing)],
                    max_rank=max(selected_rank, max(ACTIVE_RANKS)),
                )
                basis_path = (
                    ANALYSIS_DIR
                    / f"e1_h{horizon}_{preprocessing}_frozen_dual_basis.npz"
                )
                np.savez_compressed(
                    basis_path,
                    coefficients=fit["coefficients"].astype(np.float32),
                    eigenvalues=fit["eigenvalues"].astype(np.float32),
                    selected_rank=np.asarray(selected_rank),
                    state_record_ids=np.asarray(
                        [row["record_id"] for row in ACTIVE_E1_RECORDS]
                    ),
                )
                frozen_basis_hashes[basis_path.name] = sha256_file(basis_path)
        TIMINGS["e1_geometry_seconds"] = time.perf_counter() - started

        FREEZE = {
            "stage": "before_untouched_confirmation_encoding",
            "run_mode": RUN_MODE,
            "scientific_confirmation_enabled": RUN_MODE == "full",
            "configuration": CONFIG,
            "source_identity": SOURCE_IDENTITY,
            "design_freeze_sha256": sha256_file(
                DESIGN_DIR / "design_freeze.json"
            ),
            "action_sha256": ACTION_HASH,
            "methods": [
                "native_raw_global_separate_horizons",
                "native_standardized_diagnostic",
                "native_pooled_equal_total_atom_budget",
                "query_conditioned_rbf_local_pca",
                "within_state_action_split_oracle",
                "countsketch_seed_distribution",
            ],
            "ranks": ACTIVE_RANKS,
            "e1_selections": E1_SELECTIONS,
            "sketch_selections": E1_SKETCH_SELECTIONS,
            "frozen_basis_hashes": frozen_basis_hashes,
            "local_query_parameters_sha256": sha256_file(
                ANALYSIS_DIR / "e1_frozen_local_query_parameters.npz"
            ),
            "confirmation_overlap_task_halves": CONFIRMATION_TASK_HALVES,
            "thresholds": {
                "maximum_compact_rank": MAX_COMPACT_RANK,
                "minimum_null_gain": MIN_NULL_GAIN,
                "minimum_null_ratio": MIN_NULL_RATIO,
                "minimum_task_lower_bound": MIN_TASK_LOWER_BOUND,
                "minimum_positive_tasks": MIN_POSITIVE_TASKS,
                "minimum_rank32_preservation": MIN_RANK32_PRESERVATION,
                "minimum_task_split_overlap": MIN_TASK_SPLIT_OVERLAP,
                "null_overlap_percentile": NULL_OVERLAP_PERCENTILE,
                "minimum_local_over_global_gain": MIN_LOCAL_OVER_GLOBAL_GAIN,
            },
            "null_seed_schedule_sha256": array_sha256(
                NULL_SEED_SCHEDULE
            ),
            "local_permutation_schedule_sha256": array_sha256(
                LOCAL_TASK_PERMUTATION_SCHEDULE
            ),
            "bootstrap_task_indices_sha256": array_sha256(
                BOOTSTRAP_TASK_INDICES
            ),
            "bootstrap_state_indices_sha256": array_sha256(
                BOOTSTRAP_STATE_INDICES
            ),
        }
        freeze_path = OUT / "frozen_confirmation_preregistration.json"
        certificate_path = OUT / "confirmation_freeze_certificate.json"
        if freeze_path.exists() or certificate_path.exists():
            if not (freeze_path.exists() and certificate_path.exists()):
                raise RuntimeError("partial confirmation freeze artifacts")
            previous_freeze = json.loads(freeze_path.read_text())
            if previous_freeze != FREEZE:
                raise RuntimeError("resumed E1 selections differ from frozen values")
            FREEZE_HASH = sha256_file(freeze_path)
            previous_certificate = json.loads(certificate_path.read_text())
            if previous_certificate["sha256"] != FREEZE_HASH:
                raise RuntimeError("resumed confirmation freeze hash mismatch")
        else:
            existing_confirmation = list(
                (TARGET_DIR / "untouched_confirmation").glob("*.npz")
            )
            if existing_confirmation:
                raise RuntimeError(
                    "confirmation target cache predates the freeze certificate"
                )
            write_json(freeze_path, FREEZE)
            FREEZE_HASH = sha256_file(freeze_path)
            write_json(
                certificate_path,
                {
                    "sha256": FREEZE_HASH,
                    "confirmation_target_shards_existing_at_freeze": 0,
                },
            )
        memory_report("e1_analysis_and_confirmation_freeze")
        print(f"CONFIRMATION_FREEZE_SHA256: {FREEZE_HASH}")
    except Exception:
        record_failure("e1_analysis_and_freeze")
'''


confirmation = r'''# Open C only after verifying the frozen preregistration.
CONFIRMATION_DECISION = {"decision": "NOT_RUN"}


def freeze_is_valid():
    certificate = json.loads(
        (OUT / "confirmation_freeze_certificate.json").read_text()
    )
    observed = sha256_file(OUT / "frozen_confirmation_preregistration.json")
    return bool(
        observed == certificate["sha256"]
        and certificate["confirmation_target_shards_existing_at_freeze"] == 0
    )


def equal_task_statistics(pca_state, null_state, rank32_state, records, label):
    by_task = task_state_indices(records)
    task_ids = sorted(by_task)
    task_pca = np.asarray(
        [np.mean(pca_state[by_task[task]]) for task in task_ids]
    )
    task_null = np.asarray(
        [np.mean(null_state[by_task[task]]) for task in task_ids]
    )
    task_rank32 = np.asarray(
        [np.mean(rank32_state[by_task[task]]) for task in task_ids]
    )
    task_gain = task_pca - task_null
    state_rows = []
    for state_index, record in enumerate(records):
        state_rows.append(
            {
                "state_index": state_index,
                "record_id": int(record["record_id"]),
                "task_id": int(record["task_id"]),
                "within_task": int(record["within_task"]),
                "pca_fraction": float(pca_state[state_index]),
                "null_mean_fraction": float(null_state[state_index]),
                "gain": float(pca_state[state_index] - null_state[state_index]),
                "rank32_fraction": float(rank32_state[state_index]),
            }
        )
    task_rows = [
        {
            "task_id": int(task),
            "pca_fraction": float(task_pca[index]),
            "null_mean_fraction": float(task_null[index]),
            "gain": float(task_gain[index]),
            "rank32_fraction": float(task_rank32[index]),
        }
        for index, task in enumerate(task_ids)
    ]
    write_csv(ANALYSIS_DIR / f"{label}_state_rows.csv", state_rows)
    write_csv(ANALYSIS_DIR / f"{label}_task_rows.csv", task_rows)
    mean_pca = float(np.mean(task_pca))
    mean_null = float(np.mean(task_null))
    mean_gain = float(np.mean(task_gain))
    if RUN_MODE == "full":
        lower = one_sided_t_lower(task_gain)
        sign = exact_positive_sign_test(task_gain)
        values = np.stack(
            [
                [
                    pca_state[state] - null_state[state]
                    for state in by_task[task]
                ]
                for task in task_ids
            ]
        )
        bootstrap = hierarchical_bootstrap_means(
            values,
            BOOTSTRAP_TASK_INDICES[:ACTIVE_BOOTSTRAP_DRAWS],
            BOOTSTRAP_STATE_INDICES[:ACTIVE_BOOTSTRAP_DRAWS],
        )
        np.savez_compressed(
            ANALYSIS_DIR / f"{label}_hierarchical_bootstrap.npz",
            draws=bootstrap.astype(np.float32),
            task_indices=BOOTSTRAP_TASK_INDICES[:ACTIVE_BOOTSTRAP_DRAWS],
            state_indices=BOOTSTRAP_STATE_INDICES[:ACTIVE_BOOTSTRAP_DRAWS],
        )
        bootstrap_interval = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    else:
        lower = None
        sign = exact_positive_sign_test(task_gain)
        bootstrap_interval = [None, None]
    return {
        "task_ids": task_ids,
        "task_pca": task_pca,
        "task_null": task_null,
        "task_rank32": task_rank32,
        "task_gain": task_gain,
        "mean_pca": mean_pca,
        "mean_null": mean_null,
        "mean_gain": mean_gain,
        "ratio": mean_pca / max(mean_null, 1e-12),
        "one_sided_95_lower": lower,
        "sign_test": sign,
        "rank32_preservation": mean_pca / max(float(np.mean(task_rank32)), 1e-12),
        "hierarchical_bootstrap_95_interval": bootstrap_interval,
    }


def task_split_overlap(gram, records, rank, horizon):
    by_task = task_state_indices(records)
    tasks = sorted(by_task)
    midpoint = len(tasks) // 2
    task_halves = [tasks[:midpoint], tasks[midpoint:]]
    half_states = [
        [state for task in half for state in by_task[task]]
        for half in task_halves
    ]
    half_rows = [np.concatenate(state_effect_rows(states)) for states in half_states]
    fits = [
        fit_dual_pca(gram[np.ix_(rows, rows)], max_rank=len(rows))
        for rows in half_rows
    ]
    cross = gram[np.ix_(half_rows[0], half_rows[1])]
    used_rank = min(rank, fits[0]["rank"], fits[1]["rank"])
    observed = basis_overlap(
        fits[0]["coefficients"],
        cross,
        fits[1]["coefficients"],
        used_rank,
    )
    singular = np.linalg.svd(
        fits[0]["coefficients"][:, :used_rank].T
        @ cross
        @ fits[1]["coefficients"][:, :used_rank],
        compute_uv=False,
    )
    null = np.empty(ACTIVE_NULL_DRAWS, dtype=np.float32)
    for draw in range(ACTIVE_NULL_DRAWS):
        left_coordinates = covariance_shaped_coordinates(
            fits[0]["eigenvalues"],
            combine_seed(int(NULL_SEED_SCHEDULE[draw]), "overlap-left", horizon),
            used_rank,
        )
        right_coordinates = covariance_shaped_coordinates(
            fits[1]["eigenvalues"],
            combine_seed(int(NULL_SEED_SCHEDULE[draw]), "overlap-right", horizon),
            used_rank,
        )
        left_coefficients = fits[0]["coefficients"] @ left_coordinates
        right_coefficients = fits[1]["coefficients"] @ right_coordinates
        null[draw] = basis_overlap(
            left_coefficients,
            cross,
            right_coefficients,
            used_rank,
        )
    percentile = float(np.percentile(null, NULL_OVERLAP_PERCENTILE))
    np.savez_compressed(
        ANALYSIS_DIR / f"confirmation_h{horizon}_task_split_overlap.npz",
        observed=np.asarray(observed),
        null=null,
        singular_values=singular.astype(np.float32),
        principal_angles_radians=np.arccos(np.clip(singular, 0, 1)).astype(
            np.float32
        ),
        left_tasks=np.asarray(task_halves[0]),
        right_tasks=np.asarray(task_halves[1]),
    )
    return {
        "rank": used_rank,
        "overlap": observed,
        "null_percentile_97_5": percentile,
        "exceeds_absolute_floor": observed >= MIN_TASK_SPLIT_OVERLAP,
        "exceeds_null_percentile": observed > percentile,
    }


def confirmation_global(
    train_gram,
    cross_gram,
    test_gram,
    records,
    selected_rank,
    horizon,
    label,
    compute_overlap=True,
):
    reference_rank = max(ACTIVE_RANKS)
    maximum_rank = max(selected_rank, reference_rank)
    fitted = fit_dual_pca(train_gram, max_rank=len(train_gram))
    actual = reconstruction_by_groups(
        cross_gram,
        test_gram,
        fitted["coefficients"],
        [selected_rank, reference_rank],
        local_groups(len(records)),
    )
    eig_scores = cross_gram @ fitted["coefficients"]
    null_curves = np.empty(
        (len(records), len(ACTIVE_RANKS), ACTIVE_NULL_DRAWS),
        dtype=np.float32,
    )
    groups = local_groups(len(records))
    denominators = np.asarray(
        [float(np.trace(test_gram[np.ix_(rows, rows)])) for rows in groups]
    )
    for draw in range(ACTIVE_NULL_DRAWS):
        coordinates = covariance_shaped_coordinates(
            fitted["eigenvalues"],
            combine_seed(int(NULL_SEED_SCHEDULE[draw]), label, horizon),
            max(ACTIVE_RANKS),
        )
        scores = eig_scores @ coordinates
        for state_index, rows in enumerate(groups):
            cumulative = np.cumsum(scores[rows] ** 2, axis=1).sum(axis=0)
            for rank_index, rank in enumerate(ACTIVE_RANKS):
                null_curves[state_index, rank_index, draw] = (
                    cumulative[rank - 1] / denominators[state_index]
                )
    selected_rank_index = ACTIVE_RANKS.index(selected_rank)
    null_state = np.mean(null_curves[:, selected_rank_index], axis=1)
    statistics = equal_task_statistics(
        actual[:, 0], null_state, actual[:, 1], records, label
    )
    overlap = (
        task_split_overlap(test_gram, records, selected_rank, horizon)
        if compute_overlap
        else None
    )
    checks = {
        "selected_rank_at_most_eight": selected_rank <= MAX_COMPACT_RANK,
        "mean_null_gain": statistics["mean_gain"] >= MIN_NULL_GAIN,
        "pca_null_ratio": statistics["ratio"] >= MIN_NULL_RATIO,
        "positive_task_lower_bound": (
            RUN_MODE == "full"
            and statistics["one_sided_95_lower"] > MIN_TASK_LOWER_BOUND
        ),
        "positive_tasks": (
            statistics["sign_test"]["positive"] >= MIN_POSITIVE_TASKS
        ),
        "rank32_preservation": (
            statistics["rank32_preservation"] >= MIN_RANK32_PRESERVATION
        ),
    }
    if overlap is not None:
        checks["task_split_overlap"] = bool(
            overlap["exceeds_absolute_floor"]
            and overlap["exceeds_null_percentile"]
        )
    result = {
        "label": label,
        "horizon": horizon,
        "selected_rank": selected_rank,
        "reference_rank": reference_rank,
        "statistics": {
            key: (
                value.tolist() if isinstance(value, np.ndarray) else value
            )
            for key, value in statistics.items()
        },
        "overlap": overlap,
        "checks": checks,
        "passed": bool(RUN_MODE == "full" and all(checks.values())),
        "all_effect_checks_except_compact_rank": bool(
            RUN_MODE == "full"
            and all(
                value
                for key, value in checks.items()
                if key != "selected_rank_at_most_eight"
            )
        ),
    }
    np.savez_compressed(
        ANALYSIS_DIR / f"{label}_null_draws.npz",
        null_curves=null_curves,
        ranks=np.asarray(ACTIVE_RANKS),
        actual=actual.astype(np.float32),
    )
    np.savez_compressed(
        ANALYSIS_DIR / f"{label}_dual_basis.npz",
        coefficients=fitted["coefficients"][:, :maximum_rank].astype(np.float32),
        eigenvalues=fitted["eigenvalues"].astype(np.float32),
        selected_rank=np.asarray(selected_rank),
        reference_rank=np.asarray(reference_rank),
    )
    write_json(ANALYSIS_DIR / f"{label}_gate.json", result)
    return result, {
        "actual_selected": actual[:, 0],
        "actual_reference": actual[:, 1],
        "null_state": null_state,
        "fit": fitted,
    }


def simple_fixed_confirmation(
    train_matrix, test_matrix, records, selected_rank, standardize
):
    train = np.asarray(train_matrix, dtype=np.float64)
    test = np.asarray(test_matrix, dtype=np.float64)
    if standardize:
        scale = np.std(train, axis=0)
        scale = np.maximum(scale, np.median(scale) * 1e-3 + 1e-6)
        train = train / scale
        test = test / scale
    covariance = train.T @ train
    eigenvalues, axes = np.linalg.eigh(covariance)
    axes = axes[:, np.argsort(eigenvalues)[::-1]]
    scores = test @ axes[:, :selected_rank]
    fractions = []
    for rows in local_groups(len(records)):
        fractions.append(
            float(np.sum(scores[rows] ** 2) / np.sum(test[rows] ** 2))
        )
    _, task_scores = task_scores_from_state_scores(
        np.asarray(fractions)[:, None], records
    )
    return float(np.mean(task_scores))


def confirmation_sketch_diagnostics(e1_sketches, c_sketches, records):
    lookup = {
        (row["horizon"], row["seed"], row["preprocessing"]): row
        for row in E1_SKETCH_SELECTIONS
    }
    rows = []
    for horizon_index, horizon in enumerate(HORIZONS):
        for seed_index, seed in enumerate(ACTIVE_SKETCH_SEEDS):
            for standardize in [False, True]:
                preprocessing = (
                    "diagonal_standardized" if standardize else "raw"
                )
                selection = lookup[(horizon, int(seed), preprocessing)]
                value = simple_fixed_confirmation(
                    e1_sketches[horizon_index, seed_index],
                    c_sketches[horizon_index, seed_index],
                    records,
                    selection["selected_rank"],
                    standardize,
                )
                rows.append(
                    {
                        "horizon": horizon,
                        "seed": int(seed),
                        "preprocessing": preprocessing,
                        "selected_rank": selection["selected_rank"],
                        "confirmation_task_equal_reconstruction": value,
                        "is_original_stage13_seed": seed == ORIGINAL_SKETCH_SEED,
                    }
                )
    write_csv(ANALYSIS_DIR / "confirmation_sketch_diagnostics.csv", rows)
    return rows


def local_confirmation(
    train_gram, cross_gram, test_gram, e1_queries, c_queries,
    records, selection, global_internal, horizon,
):
    selected_rank = int(selection["selected_rank"])
    reference_rank = max(ACTIVE_RANKS)
    train_query, test_query, query_mean, query_scale = standardize_queries(
        e1_queries, c_queries
    )
    base_bandwidth = median_nonzero_distance(train_query)
    bandwidth = base_bandwidth * float(
        selection["selected_bandwidth_multiplier"]
    )
    if not np.isclose(
        bandwidth,
        float(selection["absolute_bandwidth"]),
        rtol=1e-10,
        atol=1e-10,
    ):
        raise RuntimeError("confirmation local bandwidth differs from E1 freeze")
    coefficients = []
    selected_scores = np.empty(len(records), dtype=np.float64)
    reference_scores = np.empty(len(records), dtype=np.float64)
    denominator = np.asarray(
        [
            np.trace(test_gram[np.ix_(rows, rows)])
            for rows in local_groups(len(records))
        ]
    )
    for state_index, rows in enumerate(local_groups(len(records))):
        weights = rbf_weights(test_query[state_index], train_query, bandwidth)
        fitted = weighted_dual_pca(
            train_gram,
            np.repeat(weights, ACTIONS_PER_STATE),
            max_rank=reference_rank,
        )
        coefficients.append(fitted["coefficients"])
        score = cross_gram[rows] @ fitted["coefficients"]
        selected_scores[state_index] = (
            np.sum(score[:, :selected_rank] ** 2) / denominator[state_index]
        )
        reference_scores[state_index] = (
            np.sum(score[:, :reference_rank] ** 2) / denominator[state_index]
        )
    null_state = global_internal["null_state"]
    statistics = equal_task_statistics(
        selected_scores,
        null_state,
        reference_scores,
        records,
        f"confirmation_local_h{horizon}",
    )
    improvement = selected_scores - global_internal["actual_selected"]
    zeros = np.zeros_like(improvement)
    improvement_statistics = equal_task_statistics(
        improvement,
        zeros,
        np.ones_like(improvement),
        records,
        f"confirmation_local_over_global_h{horizon}",
    )
    coefficient_stack = np.stack(
        [value[:, :reference_rank] for value in coefficients]
    )
    cross_performance = np.empty(
        (len(records), len(records)), dtype=np.float32
    )
    for data_state, rows in enumerate(local_groups(len(records))):
        for basis_state, fitted_coefficients in enumerate(coefficients):
            score = cross_gram[rows] @ fitted_coefficients[:, :selected_rank]
            cross_performance[data_state, basis_state] = (
                np.sum(score**2) / denominator[data_state]
            )
    if RUN_MODE == "full":
        by_task = task_state_indices(records)
        task_ids = sorted(by_task)
        index_by_task_within = {
            (int(row["task_id"]), int(row["within_task"])): index
            for index, row in enumerate(records)
        }
        permutation_draws = np.empty(
            ACTIVE_LOCAL_PERMUTATIONS, dtype=np.float32
        )
        for draw in range(ACTIVE_LOCAL_PERMUTATIONS):
            permutation = LOCAL_TASK_PERMUTATION_SCHEDULE[draw]
            task_values = []
            for task_position, task_id in enumerate(task_ids):
                mapped_task = task_ids[int(permutation[task_position])]
                state_values = []
                for state_index in by_task[task_id]:
                    within = int(records[state_index]["within_task"])
                    mapped_state = index_by_task_within[(mapped_task, within)]
                    state_values.append(
                        cross_performance[state_index, mapped_state]
                    )
                task_values.append(float(np.mean(state_values)))
            permutation_draws[draw] = float(np.mean(task_values))
        observed = float(statistics["mean_pca"])
        permutation_p = float(
            (1 + np.sum(permutation_draws >= observed))
            / (1 + len(permutation_draws))
        )
    else:
        permutation_draws = np.asarray([], dtype=np.float32)
        permutation_p = None
    checks = {
        "selected_rank_at_most_eight": selected_rank <= MAX_COMPACT_RANK,
        "mean_null_gain": statistics["mean_gain"] >= MIN_NULL_GAIN,
        "pca_null_ratio": statistics["ratio"] >= MIN_NULL_RATIO,
        "positive_task_lower_bound": (
            RUN_MODE == "full"
            and statistics["one_sided_95_lower"] > MIN_TASK_LOWER_BOUND
        ),
        "positive_tasks": (
            statistics["sign_test"]["positive"] >= MIN_POSITIVE_TASKS
        ),
        "rank32_preservation": (
            statistics["rank32_preservation"] >= MIN_RANK32_PRESERVATION
        ),
        "local_over_global_gain": (
            improvement_statistics["mean_pca"] >= MIN_LOCAL_OVER_GLOBAL_GAIN
        ),
        "local_over_global_lower_bound": (
            RUN_MODE == "full"
            and improvement_statistics["one_sided_95_lower"] > 0
        ),
        "local_over_global_positive_tasks": (
            improvement_statistics["sign_test"]["positive"]
            >= MIN_POSITIVE_TASKS
        ),
    }
    result = {
        "horizon": horizon,
        "selected_rank": selected_rank,
        "bandwidth_multiplier": selection["selected_bandwidth_multiplier"],
        "absolute_bandwidth": bandwidth,
        "statistics": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in statistics.items()
        },
        "improvement_over_global": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in improvement_statistics.items()
        },
        "task_block_permutation_p": permutation_p,
        "checks": checks,
        "passed": bool(RUN_MODE == "full" and all(checks.values())),
    }
    np.savez_compressed(
        ANALYSIS_DIR / f"confirmation_local_h{horizon}_model.npz",
        coefficients=coefficient_stack.astype(np.float32),
        query_mean=query_mean.astype(np.float32),
        query_scale=query_scale.astype(np.float32),
        base_bandwidth=np.asarray(base_bandwidth),
        bandwidth=np.asarray(bandwidth),
        cross_performance=cross_performance,
        task_block_permutation_draws=permutation_draws,
    )
    write_json(ANALYSIS_DIR / f"confirmation_local_h{horizon}_gate.json", result)
    return result


def pooled_confirmation(
    train_gram, cross_gram, test_gram, records, total_rank, horizon
):
    fitted = fit_dual_pca(train_gram, max_rank=len(train_gram))
    actual = reconstruction_by_groups(
        cross_gram,
        test_gram,
        fitted["coefficients"],
        [total_rank],
        local_groups(len(records)),
    )[:, 0]
    eig_scores = cross_gram @ fitted["coefficients"]
    groups = local_groups(len(records))
    denominators = np.asarray(
        [np.trace(test_gram[np.ix_(rows, rows)]) for rows in groups]
    )
    null_draws = np.empty(
        (len(records), ACTIVE_NULL_DRAWS), dtype=np.float32
    )
    for draw in range(ACTIVE_NULL_DRAWS):
        coordinates = covariance_shaped_coordinates(
            fitted["eigenvalues"],
            combine_seed(int(NULL_SEED_SCHEDULE[draw]), "pooled", horizon),
            total_rank,
        )
        scores = eig_scores @ coordinates
        for state_index, rows in enumerate(groups):
            null_draws[state_index, draw] = (
                np.sum(scores[rows] ** 2) / denominators[state_index]
            )
    null_state = np.mean(null_draws, axis=1)
    statistics = equal_task_statistics(
        actual,
        null_state,
        actual,
        records,
        f"confirmation_pooled_h{horizon}",
    )
    checks = {
        "mean_null_gain": statistics["mean_gain"] >= MIN_NULL_GAIN,
        "pca_null_ratio": statistics["ratio"] >= MIN_NULL_RATIO,
        "positive_task_lower_bound": (
            RUN_MODE == "full" and statistics["one_sided_95_lower"] > 0
        ),
        "positive_tasks": (
            statistics["sign_test"]["positive"] >= MIN_POSITIVE_TASKS
        ),
    }
    result = {
        "horizon": horizon,
        "total_atom_budget": total_rank,
        "statistics": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in statistics.items()
        },
        "checks": checks,
        "passed": bool(RUN_MODE == "full" and all(checks.values())),
    }
    np.savez_compressed(
        ANALYSIS_DIR / f"confirmation_pooled_h{horizon}_null.npz",
        actual=actual.astype(np.float32),
        null_draws=null_draws,
    )
    write_json(ANALYSIS_DIR / f"confirmation_pooled_h{horizon}_gate.json", result)
    return result


if not PIPELINE_FAILED:
    try:
        if not freeze_is_valid():
            raise RuntimeError("confirmation freeze certificate mismatch")
        generate_truth("untouched_confirmation", ACTIVE_CONFIRMATION_RECORDS)
        encode_targets("untouched_confirmation", ACTIVE_CONFIRMATION_RECORDS)
        build_effect_matrices(
            "untouched_confirmation", ACTIVE_CONFIRMATION_RECORDS
        )
        C_QUERIES = np.load(query_path("untouched_confirmation"))
        C_GRAMS = {}
        C_TO_E1 = {}
        for horizon in HORIZONS:
            c_path = effect_matrix_path("untouched_confirmation", horizon)
            e1_path = effect_matrix_path("e1_method_selection", horizon)
            C_GRAMS[(horizon, "raw")] = load_or_build_gram(
                f"confirmation_h{horizon}_raw_gram", c_path
            )
            C_TO_E1[(horizon, horizon, "raw")] = load_or_build_gram(
                f"confirmation_h{horizon}_to_e1_h{horizon}_raw_cross",
                c_path,
                e1_path,
            )
            C_GRAMS[(horizon, "standardized")] = load_or_build_gram(
                f"confirmation_h{horizon}_standardized_gram",
                c_path,
                scale=E1_SCALES[horizon],
            )
            C_TO_E1[(horizon, horizon, "standardized")] = load_or_build_gram(
                f"confirmation_h{horizon}_to_e1_h{horizon}_standardized_cross",
                c_path,
                e1_path,
                scale=E1_SCALES[horizon],
            )
        for c_horizon in HORIZONS:
            for e1_horizon in HORIZONS:
                key = (c_horizon, e1_horizon, "raw")
                if key not in C_TO_E1:
                    C_TO_E1[key] = load_or_build_gram(
                        f"confirmation_h{c_horizon}_to_e1_h{e1_horizon}_raw_cross",
                        effect_matrix_path("untouched_confirmation", c_horizon),
                        effect_matrix_path("e1_method_selection", e1_horizon),
                    )

        GLOBAL_GATES = {}
        GLOBAL_INTERNAL = {}
        STANDARDIZED_DIAGNOSTICS = {}
        LOCAL_GATES = {}
        for horizon in HORIZONS:
            selected_rank = int(
                E1_SELECTIONS[str(horizon)]["native_raw"]["selected_rank"]
            )
            gate, internal = confirmation_global(
                E1_GRAMS[(horizon, "raw")],
                C_TO_E1[(horizon, horizon, "raw")],
                C_GRAMS[(horizon, "raw")],
                ACTIVE_CONFIRMATION_RECORDS,
                selected_rank,
                horizon,
                f"confirmation_native_raw_h{horizon}",
                compute_overlap=True,
            )
            GLOBAL_GATES[str(horizon)] = gate
            GLOBAL_INTERNAL[horizon] = internal
            standardized_rank = int(
                E1_SELECTIONS[str(horizon)]["native_standardized"][
                    "selected_rank"
                ]
            )
            standardized_gate, _ = confirmation_global(
                E1_GRAMS[(horizon, "standardized")],
                C_TO_E1[(horizon, horizon, "standardized")],
                C_GRAMS[(horizon, "standardized")],
                ACTIVE_CONFIRMATION_RECORDS,
                standardized_rank,
                horizon,
                f"confirmation_native_standardized_h{horizon}",
                compute_overlap=False,
            )
            STANDARDIZED_DIAGNOSTICS[str(horizon)] = standardized_gate
            LOCAL_GATES[str(horizon)] = local_confirmation(
                E1_GRAMS[(horizon, "raw")],
                C_TO_E1[(horizon, horizon, "raw")],
                C_GRAMS[(horizon, "raw")],
                E1_QUERIES,
                C_QUERIES,
                ACTIVE_CONFIRMATION_RECORDS,
                E1_SELECTIONS[str(horizon)]["local"],
                internal,
                horizon,
            )

        selected_h1 = int(
            E1_SELECTIONS["1"]["native_raw"]["selected_rank"]
        )
        selected_h3 = int(
            E1_SELECTIONS["3"]["native_raw"]["selected_rank"]
        )
        fit_h1 = fit_dual_pca(
            E1_GRAMS[(1, "raw")], max_rank=selected_h1
        )
        fit_h3 = fit_dual_pca(
            E1_GRAMS[(3, "raw")], max_rank=selected_h3
        )
        cross_rank = min(selected_h1, selected_h3)
        CROSS_HORIZON_OVERLAP = basis_overlap(
            fit_h1["coefficients"],
            E1_CROSS_HORIZON,
            fit_h3["coefficients"],
            cross_rank,
        )
        cross_singular = np.linalg.svd(
            fit_h1["coefficients"][:, :cross_rank].T
            @ E1_CROSS_HORIZON
            @ fit_h3["coefficients"][:, :cross_rank],
            compute_uv=False,
        )
        write_json(
            ANALYSIS_DIR / "cross_horizon_overlap.json",
            {
                "rank": cross_rank,
                "normalized_overlap": CROSS_HORIZON_OVERLAP,
                "singular_values": cross_singular.tolist(),
                "principal_angles_radians": np.arccos(
                    np.clip(cross_singular, 0, 1)
                ).tolist(),
            },
        )

        pooled_train = np.block(
            [
                [E1_GRAMS[(1, "raw")], E1_CROSS_HORIZON],
                [E1_CROSS_HORIZON.T, E1_GRAMS[(3, "raw")]],
            ]
        )
        total_atom_budget = selected_h1 + selected_h3
        POOLED_GATES = {}
        for horizon in HORIZONS:
            pooled_cross = np.concatenate(
                [
                    C_TO_E1[(horizon, 1, "raw")],
                    C_TO_E1[(horizon, 3, "raw")],
                ],
                axis=1,
            )
            POOLED_GATES[str(horizon)] = pooled_confirmation(
                pooled_train,
                pooled_cross,
                C_GRAMS[(horizon, "raw")],
                ACTIVE_CONFIRMATION_RECORDS,
                total_atom_budget,
                horizon,
            )

        C_SKETCHES = build_sketch_cache(
            "untouched_confirmation", ACTIVE_CONFIRMATION_RECORDS
        )
        SKETCH_DIAGNOSTICS = confirmation_sketch_diagnostics(
            E1_SKETCHES, C_SKETCHES, ACTIVE_CONFIRMATION_RECORDS
        )

        if RUN_MODE == "smoke":
            decision = "SMOKE_COMPLETE_NO_SCIENTIFIC_DECISION"
            strategic_choice = "NONE_SMOKE_ONLY"
        elif all(row["passed"] for row in GLOBAL_GATES.values()):
            if (
                CROSS_HORIZON_OVERLAP >= MIN_TASK_SPLIT_OVERLAP
                and all(row["passed"] for row in POOLED_GATES.values())
            ):
                decision = "PROMOTE_GLOBAL_OUTCOME_VOCABULARY_TO_J_LENS"
                strategic_choice = "GLOBAL_JOW_EARNED"
            else:
                decision = "PROMOTE_SEPARATE_HORIZON_J_LENSES"
                strategic_choice = "HORIZON_SPECIFIC_GEOMETRY"
        elif all(row["passed"] for row in LOCAL_GATES.values()):
            decision = "PROMOTE_STATE_CONDITIONED_OUTCOME_TANGENT_BUNDLE"
            strategic_choice = "C_STATE_CONDITIONED_LOCAL_WORKSPACE"
        elif (
            any(
                int(E1_SELECTIONS[str(h)]["native_raw"]["selected_rank"])
                > MAX_COMPACT_RANK
                for h in HORIZONS
            )
            and all(
                row["all_effect_checks_except_compact_rank"]
                for row in GLOBAL_GATES.values()
            )
        ):
            decision = "STOP_DISTRIBUTED_HIGHER_RANK_OUTCOME_SPACE"
            strategic_choice = "B_HIGHER_RANK_DISTRIBUTED"
        else:
            decision = "STOP_NO_REPLICABLE_GLOBAL_OR_LOCAL_JOW_GEOMETRY"
            strategic_choice = "A_ABANDON_JOW_AT_THIS_SCALE"

        CONFIRMATION_DECISION = {
            "decision": decision,
            "strategic_choice": strategic_choice,
            "run_mode": RUN_MODE,
            "stage13_original_decision_unchanged": (
                "STOP_NO_COMPACT_OUTCOME_DICTIONARY"
            ),
            "jacobians_performed": False,
            "causal_interventions_performed": False,
            "global_gates": GLOBAL_GATES,
            "local_gates": LOCAL_GATES,
            "standardized_diagnostics": STANDARDIZED_DIAGNOSTICS,
            "pooled_gates": POOLED_GATES,
            "cross_horizon_overlap": CROSS_HORIZON_OVERLAP,
            "sketch_seed_count": len(ACTIVE_SKETCH_SEEDS),
            "confirmation_freeze_sha256": FREEZE_HASH,
        }
        write_json(OUT / "stage13b_decision.json", CONFIRMATION_DECISION)
        memory_report("confirmation_complete")
    except Exception:
        record_failure("untouched_confirmation")

if PIPELINE_FAILED:
    CONFIRMATION_DECISION = {
        "decision": "PIPELINE_FAILURE",
        "failure": FAILURE_MESSAGE,
        "jacobians_performed": False,
        "causal_interventions_performed": False,
    }
    write_json(OUT / "stage13b_decision.json", CONFIRMATION_DECISION)

print(json.dumps(CONFIRMATION_DECISION, indent=2, allow_nan=True))
'''


packaging = r'''# Plot, manifest, and package compact plus full audit evidence.
def plot_rank_curves():
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for axis, horizon in zip(axes, HORIZONS):
        selection = E1_SELECTIONS[str(horizon)]["native_raw"]
        ranks = [row["rank"] for row in selection["rank_curve"]]
        means = [row["task_equal_mean"] for row in selection["rank_curve"]]
        errors = [row["task_standard_error"] for row in selection["rank_curve"]]
        axis.errorbar(ranks, means, yerr=errors, marker="o", label="E1 LOTO")
        axis.axvline(
            selection["selected_rank"], color="black", linestyle="--",
            label="one-SE rank",
        )
        axis.set_title(f"horizon {horizon}")
        axis.set_xlabel("rank")
        axis.set_ylabel("task-equal reconstruction")
        axis.set_xticks(ranks)
        axis.legend()
    figure.tight_layout()
    destination = PLOT_DIR / "stage13b_e1_rank_curves.png"
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def plot_confirmation_gains():
    if CONFIRMATION_DECISION.get("decision") in ["NOT_RUN", "PIPELINE_FAILURE"]:
        return
    labels = []
    values = []
    for horizon in HORIZONS:
        labels.append(f"global h{horizon}")
        values.append(
            CONFIRMATION_DECISION["global_gates"][str(horizon)]["statistics"][
                "mean_gain"
            ]
        )
        labels.append(f"local h{horizon}")
        values.append(
            CONFIRMATION_DECISION["local_gates"][str(horizon)]["statistics"][
                "mean_gain"
            ]
        )
    figure, axis = plt.subplots(figsize=(8, 4.2))
    axis.bar(labels, values)
    axis.axhline(MIN_NULL_GAIN, color="black", linestyle="--", label="gate")
    axis.set_ylabel("task-equal gain over covariance null")
    axis.legend()
    figure.tight_layout()
    figure.savefig(PLOT_DIR / "stage13b_confirmation_gains.png", dpi=180)
    plt.close(figure)


if E1_SELECTIONS:
    plot_rank_curves()
plot_confirmation_gains()
write_json(OUT / "timings.json", TIMINGS)
write_json(OUT / "memory.json", MEMORY)
if not PIPELINE_FAILED:
    (OUT / "FAILURE_TRACE.txt").write_text("NONE\n")

TARGET_MANIFEST = []
for path in sorted(TARGET_DIR.rglob("*.npz")):
    TARGET_MANIFEST.append(
        {
            "path": str(path.relative_to(OUT)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    )
write_json(OUT / "target_token_manifest.json", TARGET_MANIFEST)

COMPACT = OUT / "stage13b_result_bundle"
if COMPACT.exists():
    shutil.rmtree(COMPACT)
COMPACT.mkdir(parents=True)
for name in [
    "config.json",
    "versions.json",
    "source_identity.json",
    "frozen_confirmation_preregistration.json",
    "confirmation_freeze_certificate.json",
    "pretrained_asset_verification.json",
    "restore_test.json",
    "stage13b_decision.json",
    "timings.json",
    "memory.json",
    "target_token_manifest.json",
    "FAILURE_TRACE.txt",
]:
    source = OUT / name
    if source.exists():
        shutil.copy2(source, COMPACT / name)
for directory in [DESIGN_DIR, ANALYSIS_DIR, PLOT_DIR, LOG_DIR]:
    if directory.exists():
        shutil.copytree(directory, COMPACT / directory.name)

compact_manifest = []
for path in sorted(COMPACT.rglob("*")):
    if path.is_file():
        compact_manifest.append(
            {
                "path": str(path.relative_to(COMPACT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
write_json(COMPACT / "manifest.json", compact_manifest)
compact_zip = Path(
    shutil.make_archive(str(OUT / "stage13b_result_bundle"), "zip", COMPACT)
)

full_zip = OUT / "stage13b_full_evidence_bundle.zip"
with zipfile.ZipFile(
    full_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
) as archive:
    for path in sorted(COMPACT.rglob("*")):
        if path.is_file():
            archive.write(path, Path("stage13b_result_bundle") / path.relative_to(COMPACT))
    for path in sorted(TARGET_DIR.rglob("*.npz")):
        archive.write(path, path.relative_to(OUT))

print(f"RUN_STATUS: {CONFIRMATION_DECISION['decision']}")
print(f"Compact result ZIP: {compact_zip} ({compact_zip.stat().st_size / 2**20:.1f} MiB)")
print(f"Full evidence ZIP: {full_zip} ({full_zip.stat().st_size / 2**20:.1f} MiB)")
print("The full ZIP remains on Drive; it contains the target-token shards.")
if DOWNLOAD_RESULTS:
    try:
        from google.colab import files

        files.download(str(compact_zip))
    except Exception as error:
        print(f"Automatic compact download unavailable: {error}")
'''


cells = [
    code(configuration),
    markdown(introduction),
    code(installation),
    code(setup),
    code(helpers),
    code(design),
    code(e0_audit),
    code(data_and_encoding),
    code(analysis_helpers),
    code(e1_analysis),
    code(confirmation),
    code(packaging),
]
for index, cell in enumerate(cells):
    cell["id"] = f"stage13b-{index:02d}"

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "name": TARGET.name,
            "provenance": [],
        },
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
payload = json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"
TARGET.write_text(payload)
print(f"Wrote {TARGET} ({len(payload):,} bytes)")
