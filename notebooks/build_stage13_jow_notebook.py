import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "11_action_response_geometry_pilot.ipynb"
TARGET = ROOT / "13_jacobian_outcome_workspace_screen.ipynb"


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


base = json.loads(SOURCE.read_text())
stage11_helpers = "".join(base["cells"][4]["source"])
stage11_adaptation = "".join(base["cells"][7]["source"])

simulator_helpers = function_sources(
    stage11_helpers,
    [
        "write_csv",
        "atomic_npz",
        "to_model_observation",
        "configure_repo",
        "task_cost",
        "pose_target",
        "make_environment",
        "wall_visual",
        "reset_environment",
        "rollout_branch",
        "exact_restore_test",
    ],
)
action_path_helpers = function_sources(
    stage11_adaptation,
    [
        "action_path_named_parameters",
        "action_path_checksum",
        "extract_action_path_state",
        "validate_action_path_state",
        "load_action_path_state",
    ],
)


configuration = r'''# SINGLE CONFIGURATION BLOCK — the default is the cheap causal screen.
RUN_MODE = "screen"  # "screen" or "expand"; run screen first.
MOUNT_DRIVE = True
CONTINUE_AFTER_BENCHMARK = True
DOWNLOAD_RESULTS = True

OUTPUT_DIR = "/content/counterfactual_faithfulness_stage13_jow"
DRIVE_OUTPUT_DIR = (
    "/content/drive/MyDrive/counterfactual_faithfulness_stage13_jow"
)

SEED = 13101
ENVIRONMENT = "PushT"
MODEL_NAME = "jepa_wm_pusht"
ACTIONS_PER_STATE = 10
FRAMESKIP = 5

if RUN_MODE == "screen":
    CONSTRUCTION_STATES = 8
    CALIBRATION_STATES = 4
    HORIZONS = [1]
    BLOCKS = [1, 3, 5]  # zero indexed: blocks 2, 4, and 6
    PROTOTYPE_AXES = 8
    SWAP_DOSES = [0.5, 1.0]
    ACTION_PAIRS_PER_STATE = 2
elif RUN_MODE == "expand":
    CONSTRUCTION_STATES = 16
    CALIBRATION_STATES = 8
    HORIZONS = [1, 3]
    BLOCKS = list(range(6))
    PROTOTYPE_AXES = 16
    SWAP_DOSES = [0.25, 0.5, 0.75, 1.0]
    ACTION_PAIRS_PER_STATE = 4
else:
    raise ValueError("RUN_MODE must be 'screen' or 'expand'")

TARGET_STEPS = HORIZONS
SPARSITY = 2
OUTCOME_PROJECTION_DIM = 128
OUTCOME_PROJECTION_SEED = 13119
RANDOM_DICTIONARY_SEED = 13137
CONTROL_SEED = 13151
ADAPTATION_SEED = 11401

# Provisional feasibility gates, not confirmatory thresholds.
MIN_RECONSTRUCTION_GAIN = 0.05
MIN_RECONSTRUCTION_RATIO = 1.25
MIN_COORDINATE_R2 = 0.05
MIN_COORDINATE_R2_GAIN = 0.05
MIN_CAUSAL_TRANSFER = 0.05
MIN_CAUSAL_CONTROL_GAIN = 0.10
MIN_POSITIVE_STATE_FRACTION = 0.60
MAX_RELATIVE_EDIT_NORM = 1.50
MAX_OOD_FRACTION = 0.05
MIN_MATCHED_OVER_FROZEN_GAIN = 0.05
STOP_ON_FAILED_GATE = True

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

# This already-pushed commit contains the Stage 12 design and checkpoints.
ASSET_COMMIT = "2326e74556f6f81db2560e4396f4cc52c16a28f4"
ASSET_REPOSITORY = "grewalsk/counterfactual-faithfulness-research"
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
    "physical_decoders.pt": {
        "path": (
            "results/bundles/stage12_result_bundle/frozen_training_decoders/"
            "jepa_wm_pusht_f975a0a746e7_training_decoders.pt"
        ),
        "sha256": "51b2dbb0a81df432a2db5b941de83717e9979e761d57365f47d93d2dd0c0c694",
    },
    "matched.pt": {
        "path": (
            "results/bundles/stage12_result_bundle/adapted_action_paths/"
            "jepa_wm_pusht_fidelity_constrained_matched_geometry_"
            "seed11401_f975a0a746e7.pt"
        ),
        "sha256": "3ae22aa2130f866eea67d27b7681e8f9c38999322b1747f8233895086ffa35b6",
    },
    "shuffled.pt": {
        "path": (
            "results/bundles/stage12_result_bundle/adapted_action_paths/"
            "jepa_wm_pusht_fidelity_constrained_shuffled_geometry_"
            "seed11401_f975a0a746e7.pt"
        ),
        "sha256": "f5719a44b9465cd73002ba742fb12f6a988a462e381fa0a91eaf2c499fcbfce3",
    },
}

assert ENVIRONMENT == "PushT"
assert MODEL_NAME == "jepa_wm_pusht"
assert set(BLOCKS).issubset(set(range(6)))
assert PROTOTYPE_AXES <= OUTCOME_PROJECTION_DIM
assert SPARSITY <= PROTOTYPE_AXES
'''


introduction = r'''# Stage 13: compute-minimal Jacobian Outcome Workspace screen

This Colab performs a **training-free, sequentially gated causal diagnostic**.
It asks whether PushT JEPA-WM contains a small intermediate component whose
coordinates causally move a predicted outcome toward another action's outcome.

The default `screen` mode uses 8 construction states, 4 calibration states,
horizon 1, predictor blocks 2/4/6, and 8 outcome axes. It stops before adapted
checkpoints unless the frozen model passes necessary dictionary, coordinate,
and intervention gates. The 22 MB matched checkpoint is downloaded only after
the frozen causal gate passes; shuffled geometry is downloaded only if matched
beats frozen.

Run the cells in order on a GPU runtime. The notebook:

- restores only selected Stage 12 PushT states;
- performs no model training;
- uses vector-Jacobian products rather than a full Jacobian;
- keeps one state graph live at a time;
- benchmarks the real G4 before estimating remaining time;
- writes atomic, hash-bound progress to Google Drive; and
- packages compact evidence without simulator frames or activation caches.

This is a feasibility screen, not a confirmatory workspace claim.
'''


installation = r'''import subprocess
import sys

# Preserve Colab's CUDA-matched torch and torchvision installations.
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
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "-q",
        *PINNED,
    ],
    check=True,
)
print("Installed pinned non-PyTorch dependencies; no restart is expected.")
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
RUN_SIGNATURE = hashlib.sha256(
    json.dumps(CONFIG, sort_keys=True).encode()
).hexdigest()
OUT = Path(OUTPUT_DIR) / f"{RUN_MODE}_{RUN_SIGNATURE[:12]}"
OUT.mkdir(parents=True, exist_ok=True)
ASSET_DIR = OUT / "assets"
TRUTH_DIR = OUT / "truth"
TARGET_DIR = OUT / "target_tokens"
LENS_DIR = OUT / "lenses"
PLOT_DIR = OUT / "plots"
LOG_DIR = OUT / "logs"
for directory in [
    ASSET_DIR,
    TRUTH_DIR,
    TARGET_DIR,
    LENS_DIR,
    PLOT_DIR,
    LOG_DIR,
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
    raise RuntimeError("Select Runtime > Change runtime type > GPU.")
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
log = logging.getLogger("stage13_jow")

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
    digest = sha256_file(temporary)
    if digest != specification["sha256"]:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"{name} hash mismatch: {digest}")
    temporary.replace(destination)
    return destination


write_json(OUT / "config.json", {**CONFIG, "run_signature": RUN_SIGNATURE})
write_json(OUT / "versions.json", VERSIONS)
(OUT / "FAILURE_TRACE.txt").write_text("PENDING\n")
TIMINGS = {}
PIPELINE_FAILED = False
FAILURE_MESSAGE = ""


def record_failure(stage):
    global PIPELINE_FAILED, FAILURE_MESSAGE
    PIPELINE_FAILED = True
    FAILURE_MESSAGE = f"STAGE: {stage}\n{traceback.format_exc()}"
    (OUT / "FAILURE_TRACE.txt").write_text(FAILURE_MESSAGE)
    log.exception("Captured failure in %s", stage)


for core_asset in [
    "pusht_design.npz",
    "tasks.json",
    "split_manifest.json",
    "physical_decoders.pt",
]:
    download_asset(core_asset)

print(json.dumps(VERSIONS, indent=2))
print(f"Durable run directory: {OUT}")
'''


runtime_helpers = (
    simulator_helpers
    + "\n\n\n"
    + action_path_helpers
    + r'''


class CountSketchProjector:
    """Deterministic projection of flattened patch-token tensors."""

    def __init__(self, input_dim, output_dim, seed, device="cuda"):
        rng = np.random.default_rng(seed)
        bucket = rng.integers(0, output_dim, size=input_dim, dtype=np.int64)
        sign = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), input_dim)
        self.bucket_numpy = bucket
        self.sign_numpy = sign
        self.bucket = torch.as_tensor(bucket, device=device, dtype=torch.long)
        self.sign = torch.as_tensor(sign, device=device, dtype=torch.float32)
        counts = np.bincount(bucket, minlength=output_dim).astype(np.float32)
        counts[counts == 0] = 1.0
        self.scale_numpy = np.sqrt(counts)
        self.scale = torch.as_tensor(
            self.scale_numpy, device=device, dtype=torch.float32
        )
        self.output_dim = int(output_dim)

    def __call__(self, values):
        values = values.float().flatten(1)
        output = torch.zeros(
            values.shape[0],
            self.output_dim,
            device=values.device,
            dtype=torch.float32,
        )
        output.scatter_add_(
            1,
            self.bucket[None].expand(values.shape[0], -1),
            values * self.sign[None],
        )
        return output / self.scale[None]

    def numpy(self, values):
        values = np.asarray(values, dtype=np.float32).reshape(len(values), -1)
        output = np.zeros((len(values), self.output_dim), dtype=np.float32)
        for row in range(len(values)):
            np.add.at(
                output[row],
                self.bucket_numpy,
                values[row] * self.sign_numpy,
            )
        return output / self.scale_numpy[None]


def validate_jepa_predictor(model, model_name):
    predictor = model.model.predictor
    blocks = list(getattr(predictor, "predictor_blocks", []))
    if len(blocks) != 6:
        raise RuntimeError(
            f"{model_name} expected six AdaLN blocks, found {len(blocks)}"
        )
    if predictor.__class__.__name__ != "VisionTransformerAdaLN":
        raise RuntimeError(
            f"unexpected predictor class {predictor.__class__.__name__}"
        )
    if not bool(getattr(predictor, "action_encoder_inpred", False)):
        raise RuntimeError("predictor does not encode actions internally")
    return predictor, blocks


def model_action_tensor(preprocessor, selected_actions):
    horizon = max(HORIZONS)
    selected_actions = np.asarray(selected_actions, dtype=np.float32)
    chunks = torch.from_numpy(
        selected_actions.reshape(
            ACTIONS_PER_STATE,
            horizon,
            FRAMESKIP,
            2,
        )
    ).float()
    normalized = preprocessor.normalize_actions(chunks)
    return (
        normalized.reshape(ACTIONS_PER_STATE, horizon, -1)
        .permute(1, 0, 2)
        .contiguous()
        .cuda()
    )


def layer_tokens(capture, visual_dim):
    if capture.ndim != 3:
        raise ValueError(f"unexpected block output {tuple(capture.shape)}")
    if capture.shape[1] % 256:
        raise ValueError("block sequence is not divisible by 256 tokens")
    return capture.view(
        capture.shape[0],
        capture.shape[1] // 256,
        256,
        capture.shape[-1],
    )[:, -1, :, :visual_dim]


def verify_pretrained_assets():
    rows = []
    for name, expected in EXPECTED_PRETRAINED_ASSET_SHA256.items():
        candidates = list(CACHE_ROOT.rglob(name))
        matching = [
            path for path in candidates if sha256_file(path) == expected
        ]
        if not matching:
            raise RuntimeError(f"verified pretrained asset not found: {name}")
        rows.append(
            {
                "name": name,
                "path": str(matching[0]),
                "sha256": expected,
            }
        )
    write_json(OUT / "pretrained_asset_verification.json", rows)
    return rows
'''
)


truth_generation = r'''# Reconstruct only the selected Stage 12 PushT branches.


def balanced_state_selection(eligible, task_ids, count, seed):
    rng = np.random.default_rng(seed)
    by_task = defaultdict(list)
    for state_id in eligible:
        by_task[int(task_ids[state_id])].append(int(state_id))
    for values in by_task.values():
        rng.shuffle(values)
    task_order = sorted(by_task)
    selected = []
    while len(selected) < count:
        progressed = False
        for task_id in task_order:
            if by_task[task_id] and len(selected) < count:
                selected.append(by_task[task_id].pop())
                progressed = True
        if not progressed:
            break
    if len(selected) != count:
        raise RuntimeError(f"could select only {len(selected)} of {count} states")
    return selected


def load_design():
    with np.load(ASSET_DIR / "pusht_design.npz") as payload:
        design = {name: payload[name].copy() for name in payload.files}
    tasks = [
        task
        for task in json.loads((ASSET_DIR / "tasks.json").read_text())
        if task["environment"] == ENVIRONMENT
    ]
    splits = json.loads(
        (ASSET_DIR / "split_manifest.json").read_text()
    )["environments"][ENVIRONMENT]
    construction = balanced_state_selection(
        splits["probe_train"]["state_ids"],
        design["task_ids"],
        CONSTRUCTION_STATES,
        SEED + 1,
    )
    calibration = balanced_state_selection(
        splits["probe_calibration"]["state_ids"],
        design["task_ids"],
        CALIBRATION_STATES,
        SEED + 2,
    )
    if set(construction) & set(calibration):
        raise AssertionError("construction and calibration states overlap")
    selection = {
        "construction": construction,
        "calibration": calibration,
        "construction_task_ids": sorted(
            set(int(design["task_ids"][item]) for item in construction)
        ),
        "calibration_task_ids": sorted(
            set(int(design["task_ids"][item]) for item in calibration)
        ),
    }
    write_json(OUT / "state_selection.json", selection)
    return design, {int(task["task_id"]): task for task in tasks}, selection


DESIGN, TASKS_BY_ID, STATE_SELECTION = load_design()
SELECTED_STATE_IDS = (
    STATE_SELECTION["construction"] + STATE_SELECTION["calibration"]
)


def reconstruct_truth():
    started = time.perf_counter()
    repo = configure_repo()
    first_state_id = SELECTED_STATE_IDS[0]
    first_task = TASKS_BY_ID[int(DESIGN["task_ids"][first_state_id])]
    first_actions = DESIGN["action_bank"][
        first_state_id, 1, : max(HORIZONS) * FRAMESKIP
    ]
    restore = exact_restore_test(
        repo,
        ENVIRONMENT,
        first_task,
        DESIGN["states"][first_state_id],
        first_actions,
    )
    write_json(OUT / "restore_test.json", restore)

    for index, state_id in enumerate(SELECTED_STATE_IDS):
        destination = TRUTH_DIR / f"state_{state_id:04d}.npz"
        if destination.exists():
            with np.load(destination) as previous:
                if (
                    previous["future_visual"].shape[1] == len(HORIZONS)
                    and previous["selected_actions"].shape[1]
                    == max(HORIZONS) * FRAMESKIP
                ):
                    log.info("truth resume: keeping %s", destination.name)
                    continue
            destination.unlink()

        state = DESIGN["states"][state_id]
        task_id = int(DESIGN["task_ids"][state_id])
        task = TASKS_BY_ID[task_id]
        action_bank = DESIGN["action_bank"][
            state_id, :, : max(HORIZONS) * FRAMESKIP
        ]
        initials = []
        initial_proprios = []
        future_visual = []
        future_proprio = []
        endpoints = []
        for action_index, actions in enumerate(action_bank):
            initial, _, observations, states, _, _ = rollout_branch(
                repo,
                ENVIRONMENT,
                task,
                state,
                actions,
                SEED * 1000 + state_id * 20 + action_index,
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
        if not all(
            np.array_equal(initials[0], item) for item in initials[1:]
        ):
            raise AssertionError(f"initial render drift for state {state_id}")
        if not all(
            np.array_equal(initial_proprios[0], item)
            for item in initial_proprios[1:]
        ):
            raise AssertionError(f"initial proprio drift for state {state_id}")

        atomic_npz(
            destination,
            original_state_id=np.asarray(state_id, dtype=np.int64),
            split=np.asarray(
                "construction"
                if state_id in STATE_SELECTION["construction"]
                else "calibration"
            ),
            task_id=np.asarray(task_id, dtype=np.int64),
            initial_state=np.asarray(state, dtype=np.float64),
            initial_visual=np.asarray(initials[0], dtype=np.uint8),
            initial_proprio=np.asarray(initial_proprios[0], dtype=np.float32),
            selected_actions=np.asarray(action_bank, dtype=np.float32),
            future_visual=np.asarray(future_visual, dtype=np.uint8),
            future_proprio=np.asarray(future_proprio, dtype=np.float32),
            endpoint_states=np.asarray(endpoints, dtype=np.float32),
        )
        write_json(
            OUT / "truth_progress.json",
            {
                "completed": index + 1,
                "total": len(SELECTED_STATE_IDS),
                "last_state_id": state_id,
            },
        )
        log.info(
            "simulator truth %d/%d", index + 1, len(SELECTED_STATE_IDS)
        )
    TIMINGS["truth_seconds"] = time.perf_counter() - started
    return repo


if not PIPELINE_FAILED:
    try:
        REPO = reconstruct_truth()
    except Exception:
        record_failure("reconstruct_truth")
'''


dictionary = r'''# Load the frozen JEPA-WM, encode true outcomes, and freeze the dictionary.


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
    predictor, blocks = validate_jepa_predictor(model, MODEL_NAME)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    verify_pretrained_assets()
    return model, preprocessor, predictor, blocks


def encode_target_cache(model):
    started = time.perf_counter()
    for index, state_id in enumerate(SELECTED_STATE_IDS):
        destination = TARGET_DIR / f"state_{state_id:04d}.npz"
        if destination.exists():
            continue
        with np.load(TRUTH_DIR / f"state_{state_id:04d}.npz") as truth:
            visual = truth["future_visual"]
            proprio = truth["future_proprio"]
        with torch.inference_mode():
            encoded = model.encode(to_model_observation(visual, proprio))
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
        atomic_npz(destination, true_tokens=tokens.astype(np.float16))
        write_json(
            OUT / "target_progress.json",
            {
                "completed": index + 1,
                "total": len(SELECTED_STATE_IDS),
                "last_state_id": state_id,
            },
        )
    TIMINGS["target_encoding_seconds"] = time.perf_counter() - started


def load_true_tokens(state_id):
    with np.load(TARGET_DIR / f"state_{state_id:04d}.npz") as payload:
        return payload["true_tokens"].astype(np.float32)


def projected_true_effects(state_id):
    tokens = load_true_tokens(state_id)
    projected = []
    for horizon_index in range(len(HORIZONS)):
        projected.append(PROJECTOR.numpy(tokens[:, horizon_index]))
    projected = np.stack(projected, axis=1)
    return projected - projected.mean(axis=0, keepdims=True)


def fit_outcome_dictionary():
    started = time.perf_counter()
    train = np.concatenate(
        [
            projected_true_effects(state_id).reshape(
                ACTIONS_PER_STATE * len(HORIZONS),
                OUTCOME_PROJECTION_DIM,
            )
            for state_id in STATE_SELECTION["construction"]
        ],
        axis=0,
    )
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.maximum(scale, np.median(scale) * 1e-3 + 1e-6)
    standardized = (train - mean) / scale
    _, singular, right = np.linalg.svd(
        standardized, full_matrices=False
    )
    pca_axes = right[:PROTOTYPE_AXES].astype(np.float32)

    rng = np.random.default_rng(RANDOM_DICTIONARY_SEED)
    covariance = np.cov(standardized, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    covariance_square_root = (
        eigenvectors
        * np.sqrt(np.maximum(eigenvalues, 0.0))[None]
    ) @ eigenvectors.T
    covariance_random = (
        rng.normal(size=(PROTOTYPE_AXES, OUTCOME_PROJECTION_DIM))
        @ covariance_square_root
    )
    random_axes = np.linalg.qr(covariance_random.T)[0].T.astype(np.float32)

    calibration = np.concatenate(
        [
            projected_true_effects(state_id).reshape(
                ACTIONS_PER_STATE * len(HORIZONS),
                OUTCOME_PROJECTION_DIM,
            )
            for state_id in STATE_SELECTION["calibration"]
        ],
        axis=0,
    )
    calibration = (calibration - mean) / scale

    def fraction_reconstructed(values, axes):
        reconstruction = (values @ axes.T) @ axes
        return float(
            np.sum(reconstruction**2)
            / max(np.sum(values**2), 1e-12)
        )

    pca_fraction = fraction_reconstructed(calibration, pca_axes)
    random_fraction = fraction_reconstructed(calibration, random_axes)
    reconstruction_gain = pca_fraction - random_fraction
    reconstruction_ratio = pca_fraction / max(random_fraction, 1e-12)
    gate = {
        "pca_fraction": pca_fraction,
        "random_fraction": random_fraction,
        "gain": reconstruction_gain,
        "ratio": reconstruction_ratio,
        "thresholds": {
            "minimum_gain": MIN_RECONSTRUCTION_GAIN,
            "minimum_ratio": MIN_RECONSTRUCTION_RATIO,
        },
        "passed": bool(
            reconstruction_gain >= MIN_RECONSTRUCTION_GAIN
            and reconstruction_ratio >= MIN_RECONSTRUCTION_RATIO
        ),
    }
    np.savez_compressed(
        OUT / "outcome_dictionary.npz",
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        pca_axes=pca_axes,
        random_axes=random_axes,
        singular_values=singular.astype(np.float32),
        covariance_eigenvalues=eigenvalues.astype(np.float32),
    )
    write_json(OUT / "dictionary_gate.json", gate)
    TIMINGS["dictionary_seconds"] = time.perf_counter() - started
    return {
        "mean": mean.astype(np.float32),
        "scale": scale.astype(np.float32),
        "pca_axes": pca_axes,
        "random_axes": random_axes,
        "gate": gate,
    }


if not PIPELINE_FAILED:
    try:
        MODEL, PREPROCESSOR, PREDICTOR, PREDICTOR_BLOCKS = load_frozen_model()
        BASE_ACTION_PATH = extract_action_path_state(PREDICTOR)
        encode_target_cache(MODEL)
        sample_tokens = load_true_tokens(SELECTED_STATE_IDS[0])
        PROJECTOR = CountSketchProjector(
            int(np.prod(sample_tokens.shape[-2:])),
            OUTCOME_PROJECTION_DIM,
            OUTCOME_PROJECTION_SEED,
        )
        DICTIONARY = fit_outcome_dictionary()
        DICTIONARY_GATE_PASSED = DICTIONARY["gate"]["passed"]
        print(json.dumps(DICTIONARY["gate"], indent=2))
    except Exception:
        record_failure("target_dictionary")
'''


forward_and_lens = r'''# Differentiable unroll, VJP lens, and measured-GPU benchmark.


def state_model_inputs(state_id):
    with np.load(TRUTH_DIR / f"state_{state_id:04d}.npz") as truth:
        initial_visual = truth["initial_visual"]
        initial_proprio = truth["initial_proprio"]
        selected_actions = truth["selected_actions"]
    with torch.inference_mode():
        initial = MODEL.encode(
            to_model_observation(initial_visual, initial_proprio)
        )
    initial = {
        name: value.detach() for name, value in initial.items()
    }
    actions = model_action_tensor(PREPROCESSOR, selected_actions)
    return initial, actions


def unroll_with_hooks(
    initial,
    actions,
    horizon,
    capture_blocks=(),
    intervention=None,
    require_grad=False,
):
    capture_blocks = tuple(capture_blocks)
    captures = {block: [] for block in capture_blocks}
    context = {"step": -1}
    handles = []

    for block_index in capture_blocks:
        def hook(_module, _inputs, output, block_index=block_index):
            captures[block_index].append(output)
            if (
                intervention is not None
                and block_index == intervention["block"]
                and context["step"] == horizon - 1
            ):
                value = output.clone()
                view = value.view(
                    value.shape[0],
                    value.shape[1] // 256,
                    256,
                    value.shape[-1],
                )
                delta = intervention["delta"].to(
                    view.device, view.dtype
                )
                view[:, -1, :, : delta.shape[-1]] = (
                    view[:, -1, :, : delta.shape[-1]] + delta
                )
                return view.reshape_as(value)
            return output

        handles.append(
            PREDICTOR_BLOCKS[block_index].register_forward_hook(hook)
        )

    try:
        batch = actions.shape[1]
        action_batch = actions[:horizon].permute(1, 0, 2).contiguous()
        action_features = MODEL.model.encode_act(action_batch)
        visual_history = initial["visual"].expand(
            batch, *initial["visual"].shape[1:]
        ).detach().clone()
        proprio_history = initial["proprio"].expand(
            batch, *initial["proprio"].shape[1:]
        ).detach().clone()
        if require_grad:
            visual_history.requires_grad_(True)

        predicted_tokens = None
        time_steps = None
        with torch.set_grad_enabled(require_grad):
            for step_index in range(horizon):
                context["step"] = step_index
                predicted_visual, _, predicted_proprio = (
                    MODEL.model.forward_pred(
                        visual_history[:, -MODEL.ctxt_window :],
                        action_features[
                            :, : step_index + 1
                        ][:, -MODEL.ctxt_window :],
                        proprio_history[:, -MODEL.ctxt_window :],
                    )
                )
                next_visual = predicted_visual[:, -1:]
                next_proprio = predicted_proprio[:, -1:]
                predicted_tokens = next_visual[:, 0, 0].flatten(1, 2)
                time_steps = predicted_visual.shape[1]
                visual_history = torch.cat(
                    [visual_history, next_visual], dim=1
                )
                proprio_history = torch.cat(
                    [proprio_history, next_proprio], dim=1
                )
        final_captures = {
            block: captures[block][-1] for block in capture_blocks
        }
        return predicted_tokens, final_captures, time_steps
    finally:
        for handle in handles:
            handle.remove()


def standardized_projected_tokens(tokens):
    projected = PROJECTOR(tokens)
    baseline = projected.mean(dim=0, keepdim=True).detach()
    centered = projected - baseline
    mean = torch.as_tensor(
        DICTIONARY["mean"], device=tokens.device, dtype=torch.float32
    )
    scale = torch.as_tensor(
        DICTIONARY["scale"], device=tokens.device, dtype=torch.float32
    )
    return (centered - mean) / scale


def state_vjp_lens(state_id, horizon):
    initial, actions = state_model_inputs(state_id)
    predictions, captures, _ = unroll_with_hooks(
        initial,
        actions,
        horizon,
        capture_blocks=BLOCKS,
        require_grad=True,
    )
    standardized = standardized_projected_tokens(predictions)
    axes = torch.as_tensor(
        DICTIONARY["pca_axes"],
        device=predictions.device,
        dtype=torch.float32,
    )
    scores = standardized @ axes.T
    sums = {
        block: torch.zeros(
            PROTOTYPE_AXES,
            256,
            int(PREDICTOR.predictor_embed_dim),
            dtype=torch.float32,
        )
        for block in BLOCKS
    }
    capture_values = [captures[block] for block in BLOCKS]
    for prototype in range(PROTOTYPE_AXES):
        gradients = torch.autograd.grad(
            scores[:, prototype].sum(),
            capture_values,
            retain_graph=prototype < PROTOTYPE_AXES - 1,
            allow_unused=False,
        )
        for block, gradient in zip(BLOCKS, gradients):
            sums[block][prototype] = (
                layer_tokens(
                    gradient,
                    int(PREDICTOR.predictor_embed_dim),
                )
                .mean(dim=0)
                .detach()
                .float()
                .cpu()
            )
    del predictions, captures, scores
    gc.collect()
    torch.cuda.empty_cache()
    return sums


def orthonormal_lens(raw):
    matrix = raw.reshape(raw.shape[0], -1).double().T
    q, r = torch.linalg.qr(matrix, mode="reduced")
    diagonal = torch.abs(torch.diagonal(r))
    tolerance = max(float(diagonal.max()) * 1e-6, 1e-12)
    rank = int(torch.sum(diagonal > tolerance))
    return q.T.float().reshape_as(raw), rank, diagonal.float()


def benchmark_gpu():
    state_id = STATE_SELECTION["construction"][0]
    initial, actions = state_model_inputs(state_id)
    started = time.perf_counter()
    with torch.inference_mode():
        clean, _, _ = unroll_with_hooks(
            initial, actions, HORIZONS[0], capture_blocks=[BLOCKS[0]]
        )
    forward_seconds = time.perf_counter() - started

    zero = torch.zeros(
        1,
        256,
        int(PREDICTOR.predictor_embed_dim),
        device="cuda",
    )
    started = time.perf_counter()
    with torch.inference_mode():
        zero_edit, _, _ = unroll_with_hooks(
            initial,
            actions[:, :1],
            HORIZONS[0],
            capture_blocks=[BLOCKS[0]],
            intervention={"block": BLOCKS[0], "delta": zero},
        )
    intervention_seconds = time.perf_counter() - started
    zero_difference = float(
        torch.max(torch.abs(clean[:1] - zero_edit)).item()
    )
    if zero_difference > 1e-6:
        raise AssertionError(
            f"zero-dose hook changed output by {zero_difference}"
        )

    started = time.perf_counter()
    smoke_lens = state_vjp_lens(state_id, HORIZONS[0])
    vjp_seconds = time.perf_counter() - started
    finite = all(
        bool(torch.isfinite(value).all()) for value in smoke_lens.values()
    )
    nonzero = all(
        float(torch.linalg.vector_norm(value)) > 0
        for value in smoke_lens.values()
    )
    if not (finite and nonzero):
        raise AssertionError("non-finite or zero VJP smoke result")

    interventions = (
        CALIBRATION_STATES
        * len(HORIZONS)
        * len(BLOCKS)
        * ACTION_PAIRS_PER_STATE
        * 2
        * len(SWAP_DOSES)
        * 3
    )
    frozen_seconds = (
        CONSTRUCTION_STATES * len(HORIZONS) * vjp_seconds
        + interventions * intervention_seconds
    )
    payload = {
        "forward_seconds": forward_seconds,
        "single_action_intervention_seconds": intervention_seconds,
        "one_state_one_horizon_vjp_seconds": vjp_seconds,
        "zero_dose_max_abs_difference": zero_difference,
        "finite_vjp": finite,
        "nonzero_vjp": nonzero,
        "estimated_frozen_screen_minutes": frozen_seconds / 60,
        "estimated_successful_three_condition_minutes": (
            3 * frozen_seconds / 60
        ),
        "estimate_excludes_completed_setup_and_truth": True,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    write_json(OUT / "benchmark.json", payload)
    print(json.dumps(payload, indent=2))
    return payload


if not PIPELINE_FAILED and DICTIONARY_GATE_PASSED:
    try:
        BENCHMARK = benchmark_gpu()
        if not CONTINUE_AFTER_BENCHMARK:
            raise RuntimeError(
                "Benchmark complete. Set CONTINUE_AFTER_BENCHMARK=True "
                "to run the causal screen."
            )
    except Exception:
        record_failure("integrity_and_benchmark")
elif not PIPELINE_FAILED:
    print("Dictionary gate failed; expensive VJP screen was not started.")
'''


causal_diagnostics = r'''# Build the mean VJP lens, test held-out coordinates, and run swaps.


def set_model_condition(condition):
    load_action_path_state(PREDICTOR, BASE_ACTION_PATH)
    if condition == "frozen":
        return action_path_checksum(BASE_ACTION_PATH)
    if condition not in {"matched", "shuffled"}:
        raise ValueError(condition)
    checkpoint_path = download_asset(f"{condition}.pt")
    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    expected_method = {
        "matched": "fidelity_constrained_matched_geometry",
        "shuffled": "fidelity_constrained_shuffled_geometry",
    }[condition]
    if payload.get("method") != expected_method:
        raise RuntimeError(
            f"{condition} checkpoint method mismatch: {payload.get('method')}"
        )
    if int(payload.get("adaptation_seed")) != ADAPTATION_SEED:
        raise RuntimeError(f"{condition} checkpoint seed mismatch")
    load_action_path_state(PREDICTOR, payload["action_path"])
    checksum = action_path_checksum(payload["action_path"])
    if checksum != payload.get("selected_action_path_checksum"):
        raise RuntimeError(f"{condition} action-path checksum mismatch")
    return checksum


def build_condition_lens(condition):
    destination = LENS_DIR / f"{condition}_lens.pt"
    if destination.exists():
        payload = torch.load(
            destination, map_location="cpu", weights_only=False
        )
        current_checksum = action_path_checksum(
            extract_action_path_state(PREDICTOR)
        )
        if payload.get("condition_checksum") != current_checksum:
            raise RuntimeError(
                f"{condition} cached lens checkpoint binding mismatch"
            )
        return payload
    started = time.perf_counter()
    raw_sums = {
        (horizon, block): torch.zeros(
            PROTOTYPE_AXES,
            256,
            int(PREDICTOR.predictor_embed_dim),
            dtype=torch.float32,
        )
        for horizon in HORIZONS
        for block in BLOCKS
    }
    contribution_norms = defaultdict(list)
    for state_index, state_id in enumerate(
        STATE_SELECTION["construction"]
    ):
        for horizon in HORIZONS:
            state_lens = state_vjp_lens(state_id, horizon)
            for block in BLOCKS:
                raw_sums[(horizon, block)] += state_lens[block]
                contribution_norms[(horizon, block)].append(
                    float(torch.linalg.vector_norm(state_lens[block]))
                )
        write_json(
            OUT / f"{condition}_lens_progress.json",
            {
                "completed": state_index + 1,
                "total": CONSTRUCTION_STATES,
                "last_state_id": state_id,
            },
        )
    bases = {}
    ranks = {}
    diagonals = {}
    dominance = {}
    for key, value in raw_sums.items():
        mean_lens = value / CONSTRUCTION_STATES
        basis, rank, diagonal = orthonormal_lens(mean_lens)
        name = f"h{key[0]}_b{key[1] + 1}"
        bases[name] = basis.half()
        ranks[name] = rank
        diagonals[name] = diagonal.tolist()
        norms = np.asarray(contribution_norms[key], dtype=np.float64)
        dominance[name] = float(norms.max() / max(norms.sum(), 1e-12))
    payload = {
        "condition": condition,
        "condition_checksum": action_path_checksum(
            extract_action_path_state(PREDICTOR)
        ),
        "bases": bases,
        "ranks": ranks,
        "r_diagonals": diagonals,
        "maximum_state_norm_share": dominance,
        "elapsed_seconds": time.perf_counter() - started,
    }
    torch.save(payload, destination)
    return payload


def activation_random_basis(lens_basis, seed):
    lens = lens_basis.reshape(PROTOTYPE_AXES, -1).float().cuda()
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    random = torch.randn(
        lens.shape[1],
        PROTOTYPE_AXES,
        generator=generator,
        device="cuda",
    )
    random = random - lens.T @ (lens @ random)
    q, _ = torch.linalg.qr(random, mode="reduced")
    return q.T.reshape_as(lens_basis).float().cpu()


def coordinate_dataset(state_ids, lens_payload, random_bases):
    values = defaultdict(lambda: {"jow": [], "random": [], "target": []})
    axes = DICTIONARY["pca_axes"]
    for state_id in state_ids:
        initial, actions = state_model_inputs(state_id)
        true = projected_true_effects(state_id)
        true_standardized = (
            true - DICTIONARY["mean"][None, None]
        ) / DICTIONARY["scale"][None, None]
        target_scores = true_standardized @ axes.T
        for horizon_index, horizon in enumerate(HORIZONS):
            with torch.inference_mode():
                _, captures, _ = unroll_with_hooks(
                    initial,
                    actions,
                    horizon,
                    capture_blocks=BLOCKS,
                )
            for block in BLOCKS:
                name = f"h{horizon}_b{block + 1}"
                activation = layer_tokens(
                    captures[block],
                    int(PREDICTOR.predictor_embed_dim),
                ).float()
                activation = activation - activation.mean(
                    dim=0, keepdim=True
                )
                flat = activation.flatten(1)
                basis = lens_payload["bases"][name].float().cuda().flatten(1)
                random_basis = random_bases[name].float().cuda().flatten(1)
                values[name]["jow"].append(
                    (flat @ basis.T).cpu().numpy()
                )
                values[name]["random"].append(
                    (flat @ random_basis.T).cpu().numpy()
                )
                values[name]["target"].append(
                    target_scores[:, horizon_index]
                )
    return {
        name: {
            key: np.concatenate(parts, axis=0)
            for key, parts in payload.items()
        }
        for name, payload in values.items()
    }


def ridge_fit(features, targets, ridge=1e-3):
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-6)
    x = (features - mean) / scale
    augmented = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(augmented.shape[1]) * ridge
    penalty[0, 0] = 0.0
    coefficient = np.linalg.solve(
        augmented.T @ augmented + penalty,
        augmented.T @ targets,
    )
    return {
        "mean": mean,
        "scale": scale,
        "coefficient": coefficient,
        "target_mean": targets.mean(axis=0),
    }


def ridge_r2(model, features, targets):
    x = (features - model["mean"]) / model["scale"]
    prediction = np.column_stack([np.ones(len(x)), x]) @ model["coefficient"]
    residual = np.sum((targets - prediction) ** 2)
    baseline = np.sum((targets - model["target_mean"]) ** 2)
    return float(1.0 - residual / max(baseline, 1e-12))


def coordinate_gate(condition, lens_payload):
    random_bases = {
        name: activation_random_basis(
            basis,
            CONTROL_SEED + 1000 * int(name.split("_b")[0][1:])
            + int(name.split("_b")[1]),
        )
        for name, basis in lens_payload["bases"].items()
    }
    construction = coordinate_dataset(
        STATE_SELECTION["construction"], lens_payload, random_bases
    )
    calibration = coordinate_dataset(
        STATE_SELECTION["calibration"], lens_payload, random_bases
    )
    rows = []
    for name in sorted(construction):
        for basis_name in ["jow", "random"]:
            model = ridge_fit(
                construction[name][basis_name],
                construction[name]["target"],
            )
            rows.append(
                {
                    "layer": name,
                    "basis": basis_name,
                    "calibration_r2": ridge_r2(
                        model,
                        calibration[name][basis_name],
                        calibration[name]["target"],
                    ),
                }
            )
    by_layer = defaultdict(dict)
    for row in rows:
        by_layer[row["layer"]][row["basis"]] = row["calibration_r2"]
    layer_gains = {
        name: payload["jow"] - payload["random"]
        for name, payload in by_layer.items()
    }
    best_layer = max(
        layer_gains,
        key=lambda name: (layer_gains[name], by_layer[name]["jow"]),
    )
    full_rank = all(
        rank == PROTOTYPE_AXES for rank in lens_payload["ranks"].values()
    )
    gate = {
        "condition": condition,
        "rows": rows,
        "best_layer": best_layer,
        "best_jow_r2": by_layer[best_layer]["jow"],
        "best_random_r2": by_layer[best_layer]["random"],
        "best_gain": layer_gains[best_layer],
        "all_lenses_full_rank": full_rank,
        "maximum_single_state_norm_share": max(
            lens_payload["maximum_state_norm_share"].values()
        ),
        "passed": bool(
            full_rank
            and by_layer[best_layer]["jow"] >= MIN_COORDINATE_R2
            and layer_gains[best_layer] >= MIN_COORDINATE_R2_GAIN
        ),
    }
    write_json(OUT / f"{condition}_coordinate_gate.json", gate)
    return gate


DECODER_PAYLOAD = torch.load(
    ASSET_DIR / "physical_decoders.pt",
    map_location="cpu",
    weights_only=False,
)
DECODER_PROJECTORS = {}


def decode_physical_pose(tokens):
    outputs = []
    for decoder in DECODER_PAYLOAD["decoders"]:
        seed = int(decoder["projection_seed"])
        if seed not in DECODER_PROJECTORS:
            DECODER_PROJECTORS[seed] = CountSketchProjector(
                tokens.shape[-2] * tokens.shape[-1],
                int(DECODER_PAYLOAD["projection_dim"]),
                seed,
            )
        features = DECODER_PROJECTORS[seed](tokens)
        mean = torch.as_tensor(
            decoder["mean"], device=tokens.device, dtype=torch.float32
        )
        scale = torch.as_tensor(
            decoder["scale"], device=tokens.device, dtype=torch.float32
        )
        coefficient = torch.as_tensor(
            decoder["coefficient"],
            device=tokens.device,
            dtype=torch.float32,
        )
        intercept = torch.as_tensor(
            decoder["intercept"],
            device=tokens.device,
            dtype=torch.float32,
        )
        outputs.append(
            intercept + ((features - mean) / scale) @ coefficient
        )
    return torch.stack(outputs, dim=0).mean(dim=0)


def top_effect_pairs(true_scores, count):
    candidates = []
    for left in range(ACTIONS_PER_STATE):
        for right in range(left + 1, ACTIONS_PER_STATE):
            distance = float(
                np.linalg.norm(true_scores[left] - true_scores[right])
            )
            candidates.append((distance, left, right))
    candidates.sort(reverse=True)
    return [(left, right) for _, left, right in candidates[:count]]


def sparse_coordinates(coordinates):
    if SPARSITY >= coordinates.shape[1]:
        return coordinates
    indices = torch.topk(
        torch.abs(coordinates), SPARSITY, dim=1
    ).indices
    output = torch.zeros_like(coordinates)
    output.scatter_(1, indices, coordinates.gather(1, indices))
    return output


def control_direction(kind, jow_delta, basis, seed):
    if kind == "jow":
        return jow_delta
    generator = torch.Generator(device=jow_delta.device)
    generator.manual_seed(seed)
    random = torch.randn(
        jow_delta.numel(),
        generator=generator,
        device=jow_delta.device,
    )
    if kind == "orthogonal":
        basis_flat = basis.flatten(1)
        random = random - basis_flat.T @ (basis_flat @ random)
    random_norm = torch.linalg.vector_norm(random)
    target_norm = torch.linalg.vector_norm(jow_delta)
    if random_norm <= 1e-12:
        raise RuntimeError("degenerate control direction")
    return (random * target_norm / random_norm).reshape_as(jow_delta)


def normalized_transfer(move, donor_gap):
    numerator = torch.sum(move * donor_gap)
    denominator = torch.sum(donor_gap * donor_gap)
    return float((numerator / torch.clamp(denominator, min=1e-12)).item())


def run_causal_swaps(condition, lens_payload):
    rows = []
    axes = DICTIONARY["pca_axes"]
    for state_id in STATE_SELECTION["calibration"]:
        initial, actions = state_model_inputs(state_id)
        with np.load(TRUTH_DIR / f"state_{state_id:04d}.npz") as truth:
            endpoint_states = truth["endpoint_states"].astype(np.float32)
            task_id = int(truth["task_id"])
        true_projected = projected_true_effects(state_id)
        true_standardized = (
            true_projected - DICTIONARY["mean"][None, None]
        ) / DICTIONARY["scale"][None, None]
        for horizon_index, horizon in enumerate(HORIZONS):
            with torch.inference_mode():
                clean_tokens, captures, _ = unroll_with_hooks(
                    initial,
                    actions,
                    horizon,
                    capture_blocks=BLOCKS,
                )
                clean_projected = PROJECTOR(clean_tokens)
                clean_baseline = clean_projected.mean(dim=0, keepdim=True)
                dictionary_mean = torch.as_tensor(
                    DICTIONARY["mean"], device="cuda"
                )
                dictionary_scale = torch.as_tensor(
                    DICTIONARY["scale"], device="cuda"
                )
                clean_standardized = (
                    clean_projected
                    - clean_baseline
                    - dictionary_mean
                ) / dictionary_scale
                clean_pose = decode_physical_pose(clean_tokens)
            pairs = top_effect_pairs(
                true_standardized[:, horizon_index],
                ACTION_PAIRS_PER_STATE,
            )
            for block in BLOCKS:
                name = f"h{horizon}_b{block + 1}"
                basis = lens_payload["bases"][name].float().cuda()
                activation = layer_tokens(
                    captures[block],
                    int(PREDICTOR.predictor_embed_dim),
                ).float()
                centered = activation - activation.mean(
                    dim=0, keepdim=True
                )
                coordinates = centered.flatten(1) @ basis.flatten(1).T
                sparse = sparse_coordinates(coordinates)
                natural_distances = []
                for left in range(ACTIONS_PER_STATE):
                    for right in range(left + 1, ACTIONS_PER_STATE):
                        natural_distances.append(
                            torch.linalg.vector_norm(
                                activation[left] - activation[right]
                            )
                        )
                natural_scale = float(
                    torch.median(torch.stack(natural_distances)).item()
                )

                for left, right in pairs:
                    for donor, recipient in [(left, right), (right, left)]:
                        coordinate_delta = sparse[donor] - sparse[recipient]
                        jow_delta = (
                            coordinate_delta @ basis.flatten(1)
                        ).reshape(256, -1)
                        if float(torch.linalg.vector_norm(jow_delta)) <= 1e-10:
                            continue
                        true_donor_gap = torch.as_tensor(
                            true_standardized[donor, horizon_index]
                            - true_standardized[recipient, horizon_index],
                            device="cuda",
                            dtype=torch.float32,
                        )
                        model_donor_gap = (
                            clean_standardized[donor]
                            - clean_standardized[recipient]
                        )
                        true_pose = torch.as_tensor(
                            pose_target(
                                ENVIRONMENT,
                                endpoint_states[:, horizon_index],
                            ),
                            device="cuda",
                            dtype=torch.float32,
                        )
                        true_pose_gap = (
                            true_pose[donor] - true_pose[recipient]
                        )

                        for kind_index, kind in enumerate(
                            ["jow", "orthogonal", "random"]
                        ):
                            delta = control_direction(
                                kind,
                                jow_delta,
                                basis,
                                CONTROL_SEED
                                + state_id * 100000
                                + horizon * 10000
                                + block * 1000
                                + donor * 100
                                + recipient * 10
                                + kind_index,
                            )
                            for dose in SWAP_DOSES:
                                edit = (dose * delta)[None]
                                with torch.inference_mode():
                                    intervention_tokens, _, _ = (
                                        unroll_with_hooks(
                                            initial,
                                            actions[
                                                :, recipient : recipient + 1
                                            ],
                                            horizon,
                                            capture_blocks=[block],
                                            intervention={
                                                "block": block,
                                                "delta": edit,
                                            },
                                        )
                                    )
                                    intervention_projected = PROJECTOR(
                                        intervention_tokens
                                    )
                                    intervention_standardized = (
                                        intervention_projected
                                        - clean_baseline
                                        - dictionary_mean
                                    ) / dictionary_scale
                                    intervention_pose = decode_physical_pose(
                                        intervention_tokens
                                    )[0]
                                latent_move = (
                                    intervention_standardized[0]
                                    - clean_standardized[recipient]
                                )
                                pose_move = (
                                    intervention_pose
                                    - clean_pose[recipient]
                                )
                                rows.append(
                                    {
                                        "condition": condition,
                                        "state_id": state_id,
                                        "task_id": task_id,
                                        "horizon": horizon,
                                        "block": block + 1,
                                        "donor": donor,
                                        "recipient": recipient,
                                        "dose": float(dose),
                                        "control": kind,
                                        "true_latent_transfer": normalized_transfer(
                                            latent_move, true_donor_gap
                                        ),
                                        "model_latent_transfer": normalized_transfer(
                                            latent_move, model_donor_gap
                                        ),
                                        "physical_transfer": normalized_transfer(
                                            pose_move, true_pose_gap
                                        ),
                                        "edit_norm": float(
                                            torch.linalg.vector_norm(edit).item()
                                        ),
                                        "relative_edit_norm": float(
                                            torch.linalg.vector_norm(edit).item()
                                            / max(natural_scale, 1e-12)
                                        ),
                                    }
                                )
    path = OUT / f"{condition}_causal_swaps.csv"
    write_csv(path, rows)
    return rows


def causal_gate(condition, rows):
    maximum_dose = max(SWAP_DOSES)
    minimum_dose = min(SWAP_DOSES)
    jow_max = [
        row
        for row in rows
        if row["control"] == "jow" and row["dose"] == maximum_dose
    ]
    if not jow_max:
        gate = {"condition": condition, "passed": False, "reason": "no_rows"}
        write_json(OUT / f"{condition}_causal_gate.json", gate)
        return gate
    block_means = {
        block: float(
            np.mean(
                [
                    row["true_latent_transfer"]
                    for row in jow_max
                    if row["block"] == block
                ]
            )
        )
        for block in sorted(set(row["block"] for row in jow_max))
    }
    best_block = max(block_means, key=block_means.get)

    def mean_for(control, dose):
        values = [
            row["true_latent_transfer"]
            for row in rows
            if row["block"] == best_block
            and row["control"] == control
            and row["dose"] == dose
        ]
        return float(np.mean(values)) if values else float("nan")

    jow_mean = mean_for("jow", maximum_dose)
    control_means = {
        control: mean_for(control, maximum_dose)
        for control in ["orthogonal", "random"]
    }
    control_gain = jow_mean - max(control_means.values())
    minimum_dose_mean = mean_for("jow", minimum_dose)
    state_means = defaultdict(list)
    for row in rows:
        if (
            row["block"] == best_block
            and row["control"] == "jow"
            and row["dose"] == maximum_dose
        ):
            state_means[row["state_id"]].append(
                row["true_latent_transfer"]
            )
    positive_fraction = float(
        np.mean(
            [
                np.mean(values) > 0
                for values in state_means.values()
            ]
        )
    )
    ood_fraction = float(
        np.mean(
            [
                row["relative_edit_norm"] > MAX_RELATIVE_EDIT_NORM
                for row in rows
                if row["block"] == best_block
                and row["control"] == "jow"
            ]
        )
    )
    physical_mean = float(
        np.mean(
            [
                row["physical_transfer"]
                for row in rows
                if row["block"] == best_block
                and row["control"] == "jow"
                and row["dose"] == maximum_dose
            ]
        )
    )
    gate = {
        "condition": condition,
        "best_block": best_block,
        "block_means": block_means,
        "jow_true_latent_transfer": jow_mean,
        "control_means": control_means,
        "control_gain": control_gain,
        "minimum_dose_mean": minimum_dose_mean,
        "dose_response_passed": bool(
            jow_mean >= minimum_dose_mean - 0.02
        ),
        "positive_state_fraction": positive_fraction,
        "ood_fraction": ood_fraction,
        "physical_transfer_secondary": physical_mean,
        "passed": bool(
            jow_mean >= MIN_CAUSAL_TRANSFER
            and control_gain >= MIN_CAUSAL_CONTROL_GAIN
            and jow_mean >= minimum_dose_mean - 0.02
            and positive_fraction >= MIN_POSITIVE_STATE_FRACTION
            and ood_fraction <= MAX_OOD_FRACTION
        ),
    }
    write_json(OUT / f"{condition}_causal_gate.json", gate)
    return gate


def run_condition(condition):
    checksum = set_model_condition(condition)
    log.info("running condition %s checksum=%s", condition, checksum)
    lens = build_condition_lens(condition)
    coordinates = coordinate_gate(condition, lens)
    if not coordinates["passed"] and STOP_ON_FAILED_GATE:
        return {
            "condition": condition,
            "coordinate_gate": coordinates,
            "causal_gate": {
                "condition": condition,
                "passed": False,
                "reason": "coordinate_gate_failed",
            },
        }
    rows = run_causal_swaps(condition, lens)
    causal = causal_gate(condition, rows)
    return {
        "condition": condition,
        "coordinate_gate": coordinates,
        "causal_gate": causal,
    }
'''


condition_sequence = r'''# Sequential compute gate: frozen -> matched -> conditional shuffled.

CONDITION_RESULTS = {}
FINAL_DECISION = {
    "decision": "NOT_RUN",
    "run_mode": RUN_MODE,
    "dictionary_gate": (
        DICTIONARY["gate"] if not PIPELINE_FAILED else None
    ),
}

if not PIPELINE_FAILED and DICTIONARY_GATE_PASSED:
    try:
        CONDITION_RESULTS["frozen"] = run_condition("frozen")
        frozen_causal = CONDITION_RESULTS["frozen"]["causal_gate"]
        if not frozen_causal["passed"]:
            FINAL_DECISION["decision"] = "STOP_NO_FROZEN_CAUSAL_JOW_SIGNAL"
        else:
            CONDITION_RESULTS["matched"] = run_condition("matched")
            matched_causal = CONDITION_RESULTS["matched"]["causal_gate"]
            if not matched_causal["passed"]:
                FINAL_DECISION["decision"] = "STOP_MATCHED_HAS_NO_CAUSAL_SIGNAL"
            else:
                treatment_gain = (
                    matched_causal["jow_true_latent_transfer"]
                    - frozen_causal["jow_true_latent_transfer"]
                )
                FINAL_DECISION["matched_over_frozen_gain"] = treatment_gain
                if treatment_gain < MIN_MATCHED_OVER_FROZEN_GAIN:
                    FINAL_DECISION["decision"] = "STOP_NO_MATCHED_OVER_FROZEN_GAIN"
                else:
                    CONDITION_RESULTS["shuffled"] = run_condition("shuffled")
                    shuffled_causal = CONDITION_RESULTS["shuffled"]["causal_gate"]
                    shuffled_value = shuffled_causal.get(
                        "jow_true_latent_transfer", 0.0
                    )
                    matched_over_shuffled = (
                        matched_causal["jow_true_latent_transfer"]
                        - shuffled_value
                    )
                    FINAL_DECISION[
                        "matched_over_shuffled_gain"
                    ] = matched_over_shuffled
                    if matched_over_shuffled <= 0:
                        FINAL_DECISION[
                            "decision"
                        ] = "STOP_NO_TREATMENT_SPECIFICITY"
                    elif RUN_MODE == "screen":
                        FINAL_DECISION[
                            "decision"
                        ] = "PROMOTE_TO_PHASE0_EXPANSION"
                    else:
                        FINAL_DECISION[
                            "decision"
                        ] = "PROMOTE_TO_BROADCAST_AND_NEW_TASK_DESIGN"
        FINAL_DECISION["conditions"] = CONDITION_RESULTS
    except Exception:
        record_failure("condition_sequence")
elif not PIPELINE_FAILED:
    FINAL_DECISION["decision"] = "STOP_NO_COMPACT_OUTCOME_DICTIONARY"

if PIPELINE_FAILED:
    FINAL_DECISION["decision"] = "PIPELINE_FAILURE"
    FINAL_DECISION["failure"] = FAILURE_MESSAGE

write_json(OUT / "stage13_jow_decision.json", FINAL_DECISION)
print(json.dumps(FINAL_DECISION, indent=2, allow_nan=True))
'''


packaging = r'''# Plot, package compact evidence, and optionally download one ZIP.


def plot_causal_summary():
    summaries = []
    for condition, payload in CONDITION_RESULTS.items():
        gate = payload.get("causal_gate", {})
        if "jow_true_latent_transfer" not in gate:
            continue
        summaries.append(
            (
                condition,
                gate["jow_true_latent_transfer"],
                gate["control_means"].get("orthogonal", np.nan),
                gate["control_means"].get("random", np.nan),
            )
        )
    if not summaries:
        return None
    labels = [item[0] for item in summaries]
    x = np.arange(len(labels))
    width = 0.24
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(
        x - width,
        [item[1] for item in summaries],
        width,
        label="JOW",
    )
    axis.bar(
        x,
        [item[2] for item in summaries],
        width,
        label="orthogonal",
    )
    axis.bar(
        x + width,
        [item[3] for item in summaries],
        width,
        label="random",
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x, labels)
    axis.set_ylabel("normalized true-latent donor transfer")
    axis.set_title("Stage 13 JOW causal screen")
    axis.legend()
    figure.tight_layout()
    destination = PLOT_DIR / "stage13_jow_causal_summary.png"
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


plot_causal_summary()
write_json(OUT / "timings.json", TIMINGS)
if not PIPELINE_FAILED:
    (OUT / "FAILURE_TRACE.txt").write_text("NONE\n")

BUNDLE = OUT / "stage13_jow_result_bundle"
if BUNDLE.exists():
    shutil.rmtree(BUNDLE)
BUNDLE.mkdir(parents=True)

for name in [
    "config.json",
    "versions.json",
    "pretrained_asset_verification.json",
    "state_selection.json",
    "restore_test.json",
    "dictionary_gate.json",
    "outcome_dictionary.npz",
    "benchmark.json",
    "stage13_jow_decision.json",
    "timings.json",
    "FAILURE_TRACE.txt",
]:
    source = OUT / name
    if source.exists():
        shutil.copy2(source, BUNDLE / source.name)

for pattern in [
    "*_coordinate_gate.json",
    "*_causal_gate.json",
    "*_causal_swaps.csv",
]:
    for source in OUT.glob(pattern):
        shutil.copy2(source, BUNDLE / source.name)

for source in LENS_DIR.glob("*_lens.pt"):
    shutil.copy2(source, BUNDLE / source.name)
if PLOT_DIR.exists():
    shutil.copytree(PLOT_DIR, BUNDLE / "plots")
if LOG_DIR.exists():
    shutil.copytree(LOG_DIR, BUNDLE / "logs")

manifest = []
for path in sorted(BUNDLE.rglob("*")):
    if path.is_file():
        manifest.append(
            {
                "path": str(path.relative_to(BUNDLE)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
write_json(BUNDLE / "manifest.json", manifest)
archive = Path(
    shutil.make_archive(str(OUT / "stage13_jow_result_bundle"), "zip", BUNDLE)
)
print(f"RUN_STATUS: {FINAL_DECISION['decision']}")
print(f"Result ZIP: {archive} ({archive.stat().st_size / 2**20:.1f} MB)")

if DOWNLOAD_RESULTS:
    try:
        from google.colab import files

        files.download(str(archive))
    except Exception as error:
        print(f"Automatic download unavailable: {error}")
'''


cells = [
    code(configuration),
    markdown(introduction),
    code(installation),
    code(setup),
    code(runtime_helpers),
    code(truth_generation),
    code(dictionary),
    code(forward_and_lens),
    code(causal_diagnostics),
    code(condition_sequence),
    code(packaging),
]

for index, cell in enumerate(cells):
    cell["id"] = f"stage13-{index:02d}"

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
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

payload = json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"
TARGET.write_text(payload)
print(
    f"Wrote {TARGET} with {len(cells)} cells; "
    f"sha256={hashlib.sha256(payload.encode()).hexdigest()}"
)
