"""Build the deterministic Stage 34.2 split-path continuation notebook."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
TARGET = ROOT / "34_2_split_path_continuation.ipynb"


def load_builder(path: Path):
    specification = importlib.util.spec_from_file_location("stage34_builder_for_342", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


STAGE34 = load_builder(ROOT / "build_stage34_predictive_fiber_abstraction_notebook.py")


def function_sources(text: str, names: list[str]) -> str:
    tree = ast.parse(text)
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in names:
                found[node.name] = ast.get_source_segment(text, node)
    missing = [name for name in names if name not in found]
    if missing:
        raise RuntimeError(f"missing extracted functions: {missing}")
    return "\n\n\n".join(found[name] for name in names)


def markdown(text: str, cell_id: str) -> dict:
    return {
        "cell_type": "markdown", "id": cell_id, "metadata": {},
        "source": text.strip() + "\n",
    }


def code(text: str, cell_id: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "id": cell_id,
        "metadata": {}, "outputs": [], "source": text.strip() + "\n",
    }


introduction = r'''
# Stage 34.2: split-path predictive and causal continuation

## Decision before computation

Stage 34.1 removed the action-indexed-column leak from the static comparator.
JEPA then beat both the leakage-free state prior and a same-length action
derangement on every trajectory average. DINO beat the derangement but remained
worse than the action-blind physical prior in every trajectory average.

This post-outcome continuation therefore separates the models instead of
reopening a shared gate:

1. **DINO calibration diagnosis.** A 22-parameter diagonal scale/bias map is
   fitted only on short-word calibration rows and applied without refitting to
   length-5--8 evaluation responses. It must beat raw DINO, the leakage-free
   state prior, and a calibrated same-length derangement to call the failure
   calibration-limited. A full 11-by-11 affine map is descriptive only.
2. **JEPA predictive sufficiency.** The exact unopened Stage 34 transition
   shards are scored under the original nested residual-carrier and real-
   coordinate deletion controls.
3. **JEPA native causal use.** This runs only if predictive sufficiency passes.
   The original block-4 matched fiber/state interventions, equal-energy
   residual-subspace control, full-swap positive control, and natural-support
   check are retained. One digest-validated shard is written after every pair.

## Claim boundary

Stage 34.2 is diagnostic and confirmation-ineligible because its split path was
chosen after observing Stages 34 and 34.1. A JEPA pass would establish a strong
single-checkpoint candidate causal response state on this panel, not a shared
JEPA--DINO abstraction. DINO calibration recoverability is not causal evidence.
Fresh trajectories remain necessary for a confirmatory claim.
'''


configuration = r'''
# Frozen Stage 34.2 contract. Pilot is diagnostic evidence; smoke is plumbing only.
RUN_MODE = "pilot"
MOUNT_DRIVE = True
DOWNLOAD_RESULTS = True
RESUME_INCOMPLETE = True

PROTOCOL_ID = "stage34.2-split-path-predictive-causal-continuation-v1"
NOTEBOOK_PROTOCOL_SHA256 = "__PROTOCOL_DIGEST__"
EVIDENCE_STATUS = "POST_OUTCOME_SPLIT_PATH_DIAGNOSTIC_NOT_CONFIRMATION"

EXPERIMENT_REPOSITORY = "grewalsk/counterfactual-faithfulness-research"
EXPERIMENT_SOURCE_REF = "codex/stage34-predictive-fiber-abstraction"
EXPERIMENT_NOTEBOOK_PATH = "notebooks/34_2_split_path_continuation.ipynb"
EXPERIMENT_BUILDER_PATH = "notebooks/build_stage34_2_split_path_continuation_notebook.py"
EXPERIMENT_NUMERICAL_PATH = "src/cf_faithfulness/stage34_2_split_path_continuation.py"

UPSTREAM_STAGE34_PROTOCOL_ID = "stage34-predictive-fiber-causal-abstraction-v1"
UPSTREAM_STAGE34_RUN_SIGNATURE = "d3f4f88426afff4d964bb4f1f1556c94ec3613b667edd9403ddfcd0fd78ded84"
UPSTREAM_STAGE34_SOURCE_COMMIT = "db130a3d25505b7fa69efbcd88009365cb266688"
UPSTREAM_STAGE34_RAW_MANIFEST_SHA256 = "2d2cf86fdeae5cb1034535104782dc526b8203b2e567e081ed530dbd288cb47e"
UPSTREAM_STAGE341_PROTOCOL_ID = "stage34.1-leakage-free-action-specificity-repair-v1"
UPSTREAM_STAGE341_RUN_SIGNATURE = "208b72f570749a48719ddf22d693cde5ee0fd2c8b525021506d356cd6242a8ac"
UPSTREAM_STAGE341_SOURCE_COMMIT = "7033ade938656d4c0902662aaa29d25b93e6b931"
UPSTREAM_STAGE341_MANIFEST_SHA256 = "fb1b90ca2ed8e3652fcc8f9e09282f6fa94351f923ad3b9d0a3094747e6aaa67"
UPSTREAM_STAGE341_DECISION_SHA256 = "1c800e08c60b5b430b5b0e73fafebc7cc10e517f869b7312dd184e2ac0b20979"

DRIVE_STAGE34_ROOT = "/content/drive/MyDrive/counterfactual_faithfulness_stage34_pfca"
DRIVE_STAGE341_ROOT = "/content/drive/MyDrive/counterfactual_faithfulness_stage34_1_action_specificity"
DRIVE_OUTPUT_ROOT = "/content/drive/MyDrive/counterfactual_faithfulness_stage34_2_split_path"

SEED = 342101
CALIBRATION_SEED = 342137
BOOTSTRAP_SEED = 342179
CONTROL_SEED = 342211
BOOTSTRAP_DRAWS = 2000 if RUN_MODE == "pilot" else 100
HOLM_ALPHA = 0.05

STAGE34_DECODER_SEED = 34183
STAGE34_CALIBRATION_SEED = 34253
STAGE34_BOOTSTRAP_SEED = 34283
STAGE34_CONTROL_SEED = 34351
STAGE34_BOOTSTRAP_DRAWS = 5000 if RUN_MODE == "pilot" else 64
TRANSITION_RANDOM_FEATURES = 256 if RUN_MODE == "pilot" else 32
OPERATOR_RIDGES = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
DINO_DIAGONAL_RIDGES = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
DINO_FULL_AFFINE_RIDGES = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

MAX_RESIDUAL_RELATIVE_IMPROVEMENT = 0.05
MAX_RESIDUAL_CI_UPPER = 0.10
MIN_DELETION_CONTROL_IMPROVEMENT = 0.10
MIN_DINO_CALIBRATION_GAIN = 0.10
MIN_DINO_CONTROL_ADVANTAGE = 0.10

MAX_FIBER_EFFECT_RATIO = 1.25
MIN_STATE_EFFECT_RETENTION = 0.50
MIN_STATE_INTERVENTION_COSINE = 0.20
MAX_INTERVENTION_OOD_RATE = 0.05
MIN_CAUSAL_CONTROL_ADVANTAGE = 0.10
CAUSAL_PAIRS_PER_MODE = 8
ACTIVE_CAUSAL_PAIRS_PER_MODE = CAUSAL_PAIRS_PER_MODE if RUN_MODE == "pilot" else 1
MAX_REPLAY_ABS_ERROR = 5e-4

FRAMESKIP = 5
MAX_WORD_LENGTH = 8
MODE_LABELS = ["free", "pre_contact", "contact", "post_contact"]
CORE_WORD_NAMES = [
    "L", "R", "S", "LR", "RL", "LL", "RR", "LRL", "RLR", "LLR", "RRL",
    "LRLR", "RLRL", "LLRR", "RRLL",
]
EVALUATION_WORD_NAMES = [
    "AABAB", "BABAA", "AABBAB", "BABAAB",
    "AAABBAB", "BABAAAB", "AABBABAB", "BABAABBA",
]
ZERO_WORD_NAMES = {length: f"zero{length}" for length in range(1, 9)}
CAUSAL_WORD = "zero1"
CANONICAL_RANK = 5
STATE_CARRIER_SKETCH_DIM = 64

REPO_URL = "https://github.com/facebookresearch/jepa-wms.git"
REPO_COMMIT = "13cf1d9c7e476f53c17714d2e0f1dc239a883ce0"
EXPECTED_HF_REVISION = "9b9c41ef249466630dbf1a20e78391865d07b3b9"
EXPECTED_PRETRAINED_ASSET_SHA256 = {
    "jepa_wm_pusht.pth.tar": "9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb",
    "dinov2_vits14_pretrain.pth": "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9",
}
MODEL_NAMES = ["jepa_wm_pusht"]
MODEL_SHORT_NAMES = {"jepa_wm_pusht": "jepa"}
EXPECTED_MODEL_TYPES = {"jepa_wm_pusht": "AdaLN"}
EXPECTED_CARRIER_WIDTHS = {"jepa_wm_pusht": 400}
INTERVENTION_BLOCK = 4
EXPECTED_VISUAL_TOKENS = 256
EXPECTED_VISUAL_WIDTH = 384
EXPECTED_PROPRIO_TOKENS = 256
VISUAL_SKETCH_DIM = 256
PROPRIO_PAD_DIM = 64

assert RUN_MODE in {"pilot", "smoke"}
assert CANONICAL_RANK == 5
assert len(EVALUATION_WORD_NAMES) == 8
assert ACTIVE_CAUSAL_PAIRS_PER_MODE in {1, 8}
'''


setup = r'''
# Mount Drive, resolve committed source, and initialize one resumable output directory.
import csv
import gc
import hashlib
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if MOUNT_DRIVE:
    from google.colab import drive
    if not Path("/content/drive/MyDrive").is_dir():
        drive.mount("/content/drive", timeout_ms=600_000)

SOURCE_REPOSITORY = Path("/content/counterfactual-faithfulness-stage342")
REMOTE = f"https://github.com/{EXPERIMENT_REPOSITORY}.git"
if (SOURCE_REPOSITORY / ".git").is_dir():
    subprocess.run(["git", "-C", str(SOURCE_REPOSITORY), "fetch", "origin", EXPERIMENT_SOURCE_REF], check=True)
    subprocess.run(["git", "-C", str(SOURCE_REPOSITORY), "checkout", "--detach", "FETCH_HEAD"], check=True)
else:
    subprocess.run([
        "git", "clone", "--depth", "1", "--branch", EXPERIMENT_SOURCE_REF,
        REMOTE, str(SOURCE_REPOSITORY),
    ], check=True)
SOURCE_COMMIT = subprocess.check_output(
    ["git", "-C", str(SOURCE_REPOSITORY), "rev-parse", "HEAD"], text=True
).strip()
sys.path.insert(0, str(SOURCE_REPOSITORY / "src"))

stage33_module = importlib.import_module("cf_faithfulness.stage33_interventional_abstraction")
stage341_module = importlib.import_module("cf_faithfulness.stage34_action_specificity_repair")
stage34_module = importlib.import_module("cf_faithfulness.stage34_predictive_fiber_abstraction")
stage342_module = importlib.import_module("cf_faithfulness.stage34_2_split_path_continuation")

pool_spatial_proprio_features = stage33_module.pool_spatial_proprio_features
action_response_path_rows = stage341_module.action_response_path_rows
clustered_bootstrap_interval = stage341_module.clustered_bootstrap_interval
deranged_word_rows = stage341_module.deranged_word_rows
grouped_record_mse = stage341_module.grouped_record_mse
relative_advantage = stage341_module.relative_advantage
_ridge_fit = stage34_module._ridge_fit
cosine_rows = stage34_module.cosine_rows
fit_supervised_subspace = stage34_module.fit_supervised_subspace
grouped_ridge_oof = stage34_module.grouped_ridge_oof
intervention_ood_ratio = stage34_module.intervention_ood_ratio
matched_fiber_pairs = stage34_module.matched_fiber_pairs
split_carrier_delta = stage34_module.split_carrier_delta
Stage342Gates = stage342_module.Stage342Gates
derive_stage342_decision = stage342_module.derive_stage342_decision
fit_grouped_diagonal_affine = stage342_module.fit_grouped_diagonal_affine
fit_matched_control_basis = stage342_module.fit_matched_control_basis
predict_diagonal_affine = stage342_module.predict_diagonal_affine
project_delta_to_basis = stage342_module.project_delta_to_basis
row_cosine = stage342_module.row_cosine
row_norm_ratio = stage342_module.row_norm_ratio
summarize_causal_rows = stage342_module.summarize_causal_rows


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_npz(path, **arrays):
    path = Path(path)
    temporary = Path(str(path) + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    Path(str(path) + ".sha256").write_text(sha256_file(path) + "\n")


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return jsonable(value.item())
    return value


def stable_seed(root, *parts):
    payload = ":".join([str(int(root)), *map(str, parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


CONFIG_NAMES = [
    "RUN_MODE", "PROTOCOL_ID", "NOTEBOOK_PROTOCOL_SHA256", "EVIDENCE_STATUS",
    "EXPERIMENT_REPOSITORY", "EXPERIMENT_SOURCE_REF", "EXPERIMENT_NOTEBOOK_PATH",
    "EXPERIMENT_BUILDER_PATH", "EXPERIMENT_NUMERICAL_PATH",
    "UPSTREAM_STAGE34_PROTOCOL_ID", "UPSTREAM_STAGE34_RUN_SIGNATURE",
    "UPSTREAM_STAGE34_SOURCE_COMMIT", "UPSTREAM_STAGE34_RAW_MANIFEST_SHA256",
    "UPSTREAM_STAGE341_PROTOCOL_ID", "UPSTREAM_STAGE341_RUN_SIGNATURE",
    "UPSTREAM_STAGE341_SOURCE_COMMIT", "UPSTREAM_STAGE341_MANIFEST_SHA256",
    "UPSTREAM_STAGE341_DECISION_SHA256", "SEED", "CALIBRATION_SEED",
    "BOOTSTRAP_SEED", "CONTROL_SEED", "BOOTSTRAP_DRAWS", "HOLM_ALPHA",
    "STAGE34_DECODER_SEED", "STAGE34_CALIBRATION_SEED", "STAGE34_BOOTSTRAP_SEED",
    "STAGE34_CONTROL_SEED", "STAGE34_BOOTSTRAP_DRAWS",
    "TRANSITION_RANDOM_FEATURES", "OPERATOR_RIDGES",
    "DINO_DIAGONAL_RIDGES", "DINO_FULL_AFFINE_RIDGES",
    "MAX_RESIDUAL_RELATIVE_IMPROVEMENT", "MAX_RESIDUAL_CI_UPPER",
    "MIN_DELETION_CONTROL_IMPROVEMENT", "MIN_DINO_CALIBRATION_GAIN",
    "MIN_DINO_CONTROL_ADVANTAGE", "MAX_FIBER_EFFECT_RATIO",
    "MIN_STATE_EFFECT_RETENTION", "MIN_STATE_INTERVENTION_COSINE",
    "MAX_INTERVENTION_OOD_RATE", "MIN_CAUSAL_CONTROL_ADVANTAGE",
    "ACTIVE_CAUSAL_PAIRS_PER_MODE", "MAX_REPLAY_ABS_ERROR", "FRAMESKIP",
    "MAX_WORD_LENGTH", "MODE_LABELS", "CORE_WORD_NAMES", "EVALUATION_WORD_NAMES",
    "ZERO_WORD_NAMES", "CAUSAL_WORD", "CANONICAL_RANK", "STATE_CARRIER_SKETCH_DIM",
    "REPO_URL", "REPO_COMMIT", "EXPECTED_HF_REVISION",
    "EXPECTED_PRETRAINED_ASSET_SHA256", "MODEL_NAMES", "MODEL_SHORT_NAMES",
    "EXPECTED_MODEL_TYPES", "EXPECTED_CARRIER_WIDTHS", "INTERVENTION_BLOCK",
]
CONFIG = {name: globals()[name] for name in CONFIG_NAMES}
CONFIG["SOURCE_COMMIT"] = SOURCE_COMMIT
RUN_SIGNATURE = hashlib.sha256(
    json.dumps(CONFIG, sort_keys=True, allow_nan=False).encode()
).hexdigest()
OUT = Path(DRIVE_OUTPUT_ROOT) / f"{RUN_MODE}_{RUN_SIGNATURE[:12]}"
EVIDENCE_DIR = OUT / "evaluation_evidence"
PLOT_DIR = OUT / "plots"
CAUSAL_SHARD_DIR = OUT / "causal_shards"
CHECKPOINT_DIR = OUT / "checkpoints"
for directory in [OUT, EVIDENCE_DIR, PLOT_DIR, CAUSAL_SHARD_DIR, CHECKPOINT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
RUN_STARTED_AT = time.time()
(OUT / "FAILURE_TRACE.txt").write_text("PENDING\n")
write_json(OUT / "config.json", {**CONFIG, "run_signature": RUN_SIGNATURE, "source_commit": SOURCE_COMMIT})
print(json.dumps({
    "protocol": PROTOCOL_ID, "run_signature": RUN_SIGNATURE,
    "source_commit": SOURCE_COMMIT, "output": str(OUT),
}, indent=2))
'''


binding = r'''
# Bind exact Stage 34 raw data and the exact Stage 34.1 diagnostic decision.
STAGE34_ROOT = Path(DRIVE_STAGE34_ROOT) / f"pilot_{UPSTREAM_STAGE34_RUN_SIGNATURE[:12]}"
STAGE341_ROOT = Path(DRIVE_STAGE341_ROOT) / f"pilot_{UPSTREAM_STAGE341_RUN_SIGNATURE[:12]}"
if not STAGE34_ROOT.is_dir():
    raise FileNotFoundError(f"missing complete Stage 34 Drive directory: {STAGE34_ROOT}")
if not STAGE341_ROOT.is_dir():
    raise FileNotFoundError(f"missing complete Stage 34.1 Drive directory: {STAGE341_ROOT}")

stage34_manifest_path = STAGE34_ROOT / "raw_manifest.json"
if sha256_file(stage34_manifest_path) != UPSTREAM_STAGE34_RAW_MANIFEST_SHA256:
    raise RuntimeError("Stage 34 raw manifest does not match the audited run")
stage34_manifest = json.loads(stage34_manifest_path.read_text())
stage34_by_path = {row["path"]: row for row in stage34_manifest}


def verify_stage34(relative):
    relative = str(relative)
    if relative not in stage34_by_path:
        raise RuntimeError(f"unmanifested Stage 34 input: {relative}")
    path = STAGE34_ROOT / relative
    row = stage34_by_path[relative]
    if not path.is_file() or path.stat().st_size != int(row["bytes"]):
        raise RuntimeError(f"missing or wrong-sized Stage 34 input: {relative}")
    if sha256_file(path) != row["sha256"]:
        raise RuntimeError(f"Stage 34 hash mismatch: {relative}")
    return path


stage341_manifest_path = STAGE341_ROOT / "result_zip_manifest.json"
if sha256_file(stage341_manifest_path) != UPSTREAM_STAGE341_MANIFEST_SHA256:
    raise RuntimeError("Stage 34.1 result manifest does not match the audited bundle")
stage341_manifest = json.loads(stage341_manifest_path.read_text())
stage341_by_path = {row["path"]: row for row in stage341_manifest}


def verify_stage341(relative):
    relative = str(relative)
    if relative not in stage341_by_path:
        raise RuntimeError(f"unmanifested Stage 34.1 input: {relative}")
    path = STAGE341_ROOT / relative
    row = stage341_by_path[relative]
    if not path.is_file() or path.stat().st_size != int(row["bytes"]):
        raise RuntimeError(f"missing or wrong-sized Stage 34.1 input: {relative}")
    if sha256_file(path) != row["sha256"]:
        raise RuntimeError(f"Stage 34.1 hash mismatch: {relative}")
    return path


for relative in [
    "config.json", "source_identity.json", "run_provenance_certificate.json",
    "stage34_decision.json", "FAILURE_TRACE.txt", "physical_response_chart/rank_lock.json",
    "predictive_charts/decoder_jepa.npz", "predictive_charts/artifact_manifest_jepa.json",
    "design/selected_model_selection_trajectories.json",
    "design/selected_calibration_trajectories.json",
    "design/selected_evaluation_trajectories.json",
]:
    verify_stage34(relative)
for relative in stage341_by_path:
    verify_stage341(relative)

stage34_config = json.loads((STAGE34_ROOT / "config.json").read_text())
stage34_source = json.loads((STAGE34_ROOT / "source_identity.json").read_text())
stage34_provenance = json.loads((STAGE34_ROOT / "run_provenance_certificate.json").read_text())
stage341_config = json.loads((STAGE341_ROOT / "config.json").read_text())
stage341_source = json.loads((STAGE341_ROOT / "source_identity.json").read_text())
stage341_decision = json.loads((STAGE341_ROOT / "stage34_1_decision.json").read_text())
if stage34_config.get("PROTOCOL_ID") != UPSTREAM_STAGE34_PROTOCOL_ID:
    raise RuntimeError("wrong Stage 34 protocol")
if stage34_source.get("resolved_commit") != UPSTREAM_STAGE34_SOURCE_COMMIT:
    raise RuntimeError("wrong Stage 34 source commit")
if stage34_provenance.get("run_signature") != UPSTREAM_STAGE34_RUN_SIGNATURE:
    raise RuntimeError("wrong Stage 34 run signature")
if not stage34_provenance.get("confirmation_eligible", False):
    raise RuntimeError("Stage 34 was not source/split eligible")
if (STAGE34_ROOT / "FAILURE_TRACE.txt").read_text().strip() != "NONE":
    raise RuntimeError("Stage 34 contains an execution failure")
if stage341_config.get("PROTOCOL_ID") != UPSTREAM_STAGE341_PROTOCOL_ID:
    raise RuntimeError("wrong Stage 34.1 protocol")
if stage341_source.get("commit") != UPSTREAM_STAGE341_SOURCE_COMMIT:
    raise RuntimeError("wrong Stage 34.1 source commit")
if stage341_config.get("run_signature") != UPSTREAM_STAGE341_RUN_SIGNATURE:
    raise RuntimeError("wrong Stage 34.1 run signature")
if sha256_file(STAGE341_ROOT / "stage34_1_decision.json") != UPSTREAM_STAGE341_DECISION_SHA256:
    raise RuntimeError("wrong Stage 34.1 decision content")
if not stage341_decision.get("checks", {}).get("jepa_action_specificity", False):
    raise RuntimeError("Stage 34.1 did not clear JEPA action specificity")
if stage341_decision.get("checks", {}).get("dino_action_specificity", True):
    raise RuntimeError("Stage 34.1 DINO branch is not the registered failed branch")


def load_split(split):
    payload = json.loads(
        (STAGE34_ROOT / f"design/selected_{split}_trajectories.json").read_text()
    )
    if payload.get("protocol_id") != UPSTREAM_STAGE34_PROTOCOL_ID:
        raise RuntimeError(f"wrong Stage 34 {split} selection")
    return payload["records"]


SELECTED_RECORDS = {
    split: load_split(split)
    for split in ["model_selection", "calibration", "evaluation"]
}
expected_counts = {"model_selection": 64, "calibration": 64, "evaluation": 128}
if {key: len(value) for key, value in SELECTED_RECORDS.items()} != expected_counts:
    raise RuntimeError("Stage 34 record counts changed")
trajectory_sets = {
    split: {int(row["trajectory_id"]) for row in records}
    for split, records in SELECTED_RECORDS.items()
}
if any(
    trajectory_sets[left] & trajectory_sets[right]
    for left in trajectory_sets for right in trajectory_sets if left < right
):
    raise RuntimeError("Stage 34 trajectory splits overlap")

consumed_stage34 = set()
for split in ["calibration", "evaluation"]:
    for record in SELECTED_RECORDS[split]:
        record_id = int(record["record_id"])
        consumed_stage34.add(f"truth/truth_{record_id}.npz")
        consumed_stage34.add(f"baseline_shards/dino_{record_id}.npz")
        consumed_stage34.add(f"baseline_shards/jepa_{record_id}.npz")
for split in ["model_selection", "calibration", "evaluation"]:
    for record in SELECTED_RECORDS[split]:
        consumed_stage34.add(
            f"baseline_shards/transitions_jepa_{split}_{int(record['record_id'])}.npz"
        )
for relative in sorted(consumed_stage34):
    verify_stage34(relative)

rank_lock = json.loads((STAGE34_ROOT / "physical_response_chart/rank_lock.json").read_text())
if int(rank_lock.get("diagnostic_rank", -1)) != CANONICAL_RANK:
    raise RuntimeError("Stage 34 canonical rank changed")
UPSTREAM_BINDING_GATE = True
STAGE341_BINDING_GATE = True
JEPA_ACTION_SPECIFICITY_GATE = True
write_json(OUT / "upstream_binding_certificate.json", {
    "stage34_run_signature": UPSTREAM_STAGE34_RUN_SIGNATURE,
    "stage341_run_signature": UPSTREAM_STAGE341_RUN_SIGNATURE,
    "stage34_consumed_shards": len(consumed_stage34),
    "stage341_verified_files": len(stage341_by_path),
    "trajectory_split_counts": {key: len(value) // 4 for key, value in SELECTED_RECORDS.items()},
    "trajectory_disjoint": True,
    "canonical_rank": CANONICAL_RANK,
    "jepa_action_specificity_bound": True,
    "dino_failed_branch_bound": True,
    "binding_passed": True,
})
print(f"Verified {len(consumed_stage34)} Stage 34 shards and {len(stage341_by_path)} Stage 34.1 files")
'''


data_helpers = r'''
# Load no-op-corrected response rows and frozen transition rows from the bound inputs.
def truth_path(record):
    return STAGE34_ROOT / f"truth/truth_{int(record['record_id'])}.npz"


def model_path(short, record):
    return STAGE34_ROOT / f"baseline_shards/{short}_{int(record['record_id'])}.npz"


def transition_path(record, split):
    return STAGE34_ROOT / (
        f"baseline_shards/transitions_jepa_{split}_{int(record['record_id'])}.npz"
    )


def response_rows(path, response_words):
    with np.load(path, allow_pickle=False) as payload:
        return action_response_path_rows(
            payload["path_observables"] if "path_observables" in payload.files
            else payload["grounded_predictions"],
            [str(value) for value in payload["word_names"]],
            payload["word_lengths"], response_words, ZERO_WORD_NAMES,
        )


def stack_dino_split(split, response_words):
    model_rows, truth_rows, groups, records, modes, blocks = [], [], [], [], [], []
    for record in SELECTED_RECORDS[split]:
        predicted, metadata = response_rows(model_path("dino", record), response_words)
        target, truth_metadata = response_rows(truth_path(record), response_words)
        if not all(np.array_equal(metadata[key], truth_metadata[key]) for key in metadata):
            raise RuntimeError("DINO response-row metadata drifted")
        count = len(predicted)
        model_rows.append(predicted)
        truth_rows.append(target)
        groups.extend([int(record["trajectory_id"])] * count)
        records.extend([int(record["record_id"])] * count)
        modes.extend([str(record["mode"])] * count)
        blocks.append({"prediction": predicted, "target": target, "metadata": metadata})
    return {
        "prediction": np.concatenate(model_rows),
        "target": np.concatenate(truth_rows),
        "group": np.asarray(groups, dtype=np.int64),
        "record": np.asarray(records, dtype=np.int64),
        "mode": np.asarray(modes),
        "blocks": blocks,
    }


def load_transition_rows(split):
    rows = {key: [] for key in [
        "state", "action", "target", "residual", "group", "mode",
        "word", "length", "record_id",
    ]}
    for record in SELECTED_RECORDS[split]:
        with np.load(transition_path(record, split), allow_pickle=False) as payload:
            count = len(payload["words"])
            rows["state"].extend(payload["source_coordinates"])
            rows["action"].extend(payload["actions"])
            rows["target"].extend(payload["target_coordinates"])
            rows["residual"].extend(payload["state_carrier_sketch"])
            rows["group"].extend([int(record["trajectory_id"])] * count)
            rows["mode"].extend([str(value) for value in payload["source_mode"]])
            rows["word"].extend([str(value) for value in payload["words"]])
            rows["length"].extend(payload["word_lengths"].astype(int).tolist())
            rows["record_id"].extend([int(record["record_id"])] * count)
    for key in ["state", "action", "target", "residual"]:
        rows[key] = np.asarray(rows[key], dtype=np.float64)
    for key in ["group", "length", "record_id"]:
        rows[key] = np.asarray(rows[key], dtype=np.int64)
    rows["mode"] = np.asarray(rows["mode"])
    rows["word"] = np.asarray(rows["word"])
    return rows


with (STAGE341_ROOT / "evaluation_evidence/locked_action_specificity_repair_rows.csv").open() as handle:
    stage341_rows = list(csv.DictReader(handle))
ACTION_BLIND_ERROR_BY_RECORD = {
    int(row["record_id"]): float(row["action_blind_state_mse"])
    for row in stage341_rows if row["model"] == "dino"
}
if len(ACTION_BLIND_ERROR_BY_RECORD) != 128:
    raise RuntimeError("Stage 34.1 DINO state-error table changed")
'''


dino_diagnostic = r'''
# Diagnose DINO with calibration-only diagonal scale/bias; full affine is descriptive only.
DINO_CALIBRATION = stack_dino_split("calibration", CORE_WORD_NAMES)
DINO_EVALUATION = stack_dino_split("evaluation", EVALUATION_WORD_NAMES)
DIAGONAL_MODEL = fit_grouped_diagonal_affine(
    DINO_CALIBRATION["prediction"], DINO_CALIBRATION["target"],
    DINO_CALIBRATION["group"], penalties=DINO_DIAGONAL_RIDGES,
    folds=4, seed=CALIBRATION_SEED,
)
FULL_AFFINE_MODEL = grouped_ridge_oof(
    DINO_CALIBRATION["prediction"], DINO_CALIBRATION["target"],
    DINO_CALIBRATION["group"], penalties=DINO_FULL_AFFINE_RIDGES,
    folds=4, seed=CALIBRATION_SEED + 1,
)
diagonal_prediction = predict_diagonal_affine(
    DIAGONAL_MODEL, DINO_EVALUATION["prediction"]
)
full_affine_prediction = (
    DINO_EVALUATION["prediction"] @ FULL_AFFINE_MODEL["weight"]
    + FULL_AFFINE_MODEL["intercept"]
)

raw_error, record_order = grouped_record_mse(
    DINO_EVALUATION["prediction"], DINO_EVALUATION["target"], DINO_EVALUATION["record"]
)
diagonal_error, diagonal_order = grouped_record_mse(
    diagonal_prediction, DINO_EVALUATION["target"], DINO_EVALUATION["record"]
)
full_error, full_order = grouped_record_mse(
    full_affine_prediction, DINO_EVALUATION["target"], DINO_EVALUATION["record"]
)
if not np.array_equal(record_order, diagonal_order) or not np.array_equal(record_order, full_order):
    raise RuntimeError("DINO calibration record order changed")
record_by_id = {
    int(record["record_id"]): record for record in SELECTED_RECORDS["evaluation"]
}
record_groups = np.asarray([int(record_by_id[int(value)]["trajectory_id"]) for value in record_order])
record_modes = np.asarray([str(record_by_id[int(value)]["mode"]) for value in record_order])
state_error = np.asarray([ACTION_BLIND_ERROR_BY_RECORD[int(value)] for value in record_order])

deranged_predictions = []
offset = 0
for index, block in enumerate(DINO_EVALUATION["blocks"]):
    count = len(block["prediction"])
    calibrated_block = diagonal_prediction[offset:offset + count]
    deranged_predictions.append(deranged_word_rows(
        calibrated_block, block["metadata"],
        seed=stable_seed(CONTROL_SEED, "dino", index, "calibrated_derangement"),
    ))
    offset += count
deranged_prediction = np.concatenate(deranged_predictions)
deranged_error, deranged_order = grouped_record_mse(
    deranged_prediction, DINO_EVALUATION["target"], DINO_EVALUATION["record"]
)
if not np.array_equal(record_order, deranged_order):
    raise RuntimeError("DINO derangement record order changed")

calibration_gain = relative_advantage(diagonal_error, raw_error)
state_advantage = relative_advantage(diagonal_error, state_error)
derangement_advantage = relative_advantage(diagonal_error, deranged_error)
calibration_ci = clustered_bootstrap_interval(
    calibration_gain, record_groups, draws=BOOTSTRAP_DRAWS,
    seed=stable_seed(BOOTSTRAP_SEED, "dino_calibration"), alpha=HOLM_ALPHA,
)
state_ci = clustered_bootstrap_interval(
    state_advantage, record_groups, draws=BOOTSTRAP_DRAWS,
    seed=stable_seed(BOOTSTRAP_SEED, "dino_state"), alpha=HOLM_ALPHA,
)
derangement_ci = clustered_bootstrap_interval(
    derangement_advantage, record_groups, draws=BOOTSTRAP_DRAWS,
    seed=stable_seed(BOOTSTRAP_SEED, "dino_derangement"), alpha=HOLM_ALPHA,
)
mode_state = {
    mode: float(np.mean(state_advantage[record_modes == mode])) for mode in MODE_LABELS
}
mode_derangement = {
    mode: float(np.mean(derangement_advantage[record_modes == mode])) for mode in MODE_LABELS
}
DINO_DIAGONAL_RECOVERABILITY_GATE = bool(
    np.mean(calibration_gain) >= MIN_DINO_CALIBRATION_GAIN
    and calibration_ci[0] > 0
    and np.mean(state_advantage) > 0
    and state_ci[0] > 0
    and all(value > 0 for value in mode_state.values())
    and np.mean(derangement_advantage) >= MIN_DINO_CONTROL_ADVANTAGE
    and derangement_ci[0] > 0
    and all(value > 0 for value in mode_derangement.values())
)

DINO_ROWS = []
for index, record_id in enumerate(record_order):
    DINO_ROWS.append({
        "record_id": int(record_id),
        "trajectory_id": int(record_groups[index]),
        "mode": str(record_modes[index]),
        "raw_mse": float(raw_error[index]),
        "diagonal_affine_mse": float(diagonal_error[index]),
        "full_affine_mse": float(full_error[index]),
        "action_blind_state_mse": float(state_error[index]),
        "calibrated_derangement_mse": float(deranged_error[index]),
        "diagonal_calibration_gain": float(calibration_gain[index]),
        "diagonal_state_advantage": float(state_advantage[index]),
        "diagonal_derangement_advantage": float(derangement_advantage[index]),
    })
DINO_SUMMARY = {
    "calibration_rows": len(DINO_CALIBRATION["prediction"]),
    "evaluation_records": len(record_order),
    "diagonal_parameter_count": DIAGONAL_MODEL["parameter_count"],
    "diagonal_penalty": DIAGONAL_MODEL["penalty"],
    "diagonal_calibration_oof_mse": DIAGONAL_MODEL["oof_mse"],
    "full_affine_parameter_count": 11 * 11 + 11,
    "full_affine_penalty": FULL_AFFINE_MODEL["penalty"],
    "full_affine_calibration_oof_mse": FULL_AFFINE_MODEL["oof_mse"],
    "raw_evaluation_mse": float(np.mean(raw_error)),
    "diagonal_evaluation_mse": float(np.mean(diagonal_error)),
    "full_affine_evaluation_mse": float(np.mean(full_error)),
    "action_blind_state_mse": float(np.mean(state_error)),
    "mean_diagonal_calibration_gain": float(np.mean(calibration_gain)),
    "diagonal_calibration_gain_ci95": calibration_ci,
    "mean_diagonal_state_advantage": float(np.mean(state_advantage)),
    "diagonal_state_advantage_ci95": state_ci,
    "mean_diagonal_derangement_advantage": float(np.mean(derangement_advantage)),
    "diagonal_derangement_advantage_ci95": derangement_ci,
    "mode_diagonal_state_advantage": mode_state,
    "mode_diagonal_derangement_advantage": mode_derangement,
    "raw_mean_row_cosine": float(np.mean(row_cosine(
        DINO_EVALUATION["prediction"], DINO_EVALUATION["target"]
    ))),
    "diagonal_mean_row_cosine": float(np.mean(row_cosine(
        diagonal_prediction, DINO_EVALUATION["target"]
    ))),
    "raw_median_norm_ratio": float(np.median(row_norm_ratio(
        DINO_EVALUATION["prediction"], DINO_EVALUATION["target"]
    ))),
    "diagonal_median_norm_ratio": float(np.median(row_norm_ratio(
        diagonal_prediction, DINO_EVALUATION["target"]
    ))),
    "diagonal_recoverability_passed": DINO_DIAGONAL_RECOVERABILITY_GATE,
}
write_csv(EVIDENCE_DIR / "dino_calibration_diagnostic_rows.csv", DINO_ROWS)
write_json(EVIDENCE_DIR / "dino_calibration_diagnostic_summary.json", jsonable(DINO_SUMMARY))
atomic_npz(
    OUT / "dino_diagonal_calibration.npz",
    scale=np.asarray(DIAGONAL_MODEL["scale"]),
    intercept=np.asarray(DIAGONAL_MODEL["intercept"]),
    penalty=np.asarray(DIAGONAL_MODEL["penalty"]),
)
print(json.dumps(DINO_SUMMARY, indent=2))
'''


jepa_sufficiency = r'''
# Apply the original Stage 34 predictive-sufficiency gate to JEPA only.
def rff_parameters(inputs, seed, width):
    x = np.asarray(inputs, dtype=np.float64)
    mean = np.mean(x, axis=0)
    scale = np.maximum(np.std(x, axis=0, ddof=1), 1e-8)
    rng = np.random.default_rng(int(seed))
    weight = rng.normal(size=(x.shape[1], int(width))) / np.sqrt(x.shape[1])
    bias = rng.uniform(-np.pi, np.pi, size=int(width))
    return {"mean": mean, "scale": scale, "weight": weight, "bias": bias}


def rff_apply(inputs, parameters):
    x = (np.asarray(inputs, dtype=np.float64) - parameters["mean"]) / parameters["scale"]
    return np.sqrt(2.0 / parameters["weight"].shape[1]) * np.cos(
        x @ parameters["weight"] + parameters["bias"]
    )


def fit_locked_rff(selection_x, selection_y, selection_groups, calibration_x, calibration_y, seed):
    parameters = rff_parameters(calibration_x, seed, TRANSITION_RANDOM_FEATURES)
    selected = grouped_ridge_oof(
        rff_apply(selection_x, parameters), selection_y, selection_groups,
        penalties=OPERATOR_RIDGES, folds=4, seed=seed,
    )
    weight, intercept = _ridge_fit(
        rff_apply(calibration_x, parameters), np.asarray(calibration_y), selected["penalty"]
    )
    return {
        "parameters": parameters, "weight": weight, "intercept": intercept,
        "penalty": selected["penalty"], "selection_oof_mse": selected["oof_mse"],
    }


def apply_locked_rff(model, inputs):
    return rff_apply(inputs, model["parameters"]) @ model["weight"] + model["intercept"]


selection = load_transition_rows("model_selection")
calibration = load_transition_rows("calibration")
evaluation = load_transition_rows("evaluation")
base_selection = np.column_stack([selection["state"], selection["action"]])
base_calibration = np.column_stack([calibration["state"], calibration["action"]])
base_evaluation = np.column_stack([evaluation["state"], evaluation["action"]])
enriched_selection = np.column_stack([selection["state"], selection["action"], selection["residual"]])
enriched_calibration = np.column_stack([calibration["state"], calibration["action"], calibration["residual"]])
enriched_evaluation = np.column_stack([evaluation["state"], evaluation["action"], evaluation["residual"]])
deleted_selection = np.column_stack([selection["state"][:, :-1], selection["action"]])
deleted_calibration = np.column_stack([calibration["state"][:, :-1], calibration["action"]])
deleted_evaluation = np.column_stack([evaluation["state"][:, :-1], evaluation["action"]])

base_model = fit_locked_rff(
    base_selection, selection["target"], selection["group"],
    base_calibration, calibration["target"],
    stable_seed(STAGE34_CALIBRATION_SEED, "jepa", "base_transition"),
)
enriched_model = fit_locked_rff(
    enriched_selection, selection["target"], selection["group"],
    enriched_calibration, calibration["target"],
    stable_seed(STAGE34_CALIBRATION_SEED, "jepa", "enriched_transition"),
)
deleted_model = fit_locked_rff(
    deleted_selection, selection["target"], selection["group"],
    deleted_calibration, calibration["target"],
    stable_seed(STAGE34_CALIBRATION_SEED, "jepa", "deleted_transition"),
)
base_prediction = apply_locked_rff(base_model, base_evaluation)
enriched_prediction = apply_locked_rff(enriched_model, enriched_evaluation)
deleted_prediction = apply_locked_rff(deleted_model, deleted_evaluation)
base_error = np.mean((base_prediction - evaluation["target"]) ** 2, axis=1)
enriched_error = np.mean((enriched_prediction - evaluation["target"]) ** 2, axis=1)
deleted_error = np.mean((deleted_prediction - evaluation["target"]) ** 2, axis=1)
residual_gain = relative_advantage(enriched_error, base_error)
deletion_gain = relative_advantage(base_error, deleted_error)
residual_ci = clustered_bootstrap_interval(
    residual_gain, evaluation["group"], draws=STAGE34_BOOTSTRAP_DRAWS,
    seed=stable_seed(STAGE34_BOOTSTRAP_SEED, "jepa", "residual_sufficiency"),
    alpha=HOLM_ALPHA,
)
deletion_ci = clustered_bootstrap_interval(
    deletion_gain, evaluation["group"], draws=STAGE34_BOOTSTRAP_DRAWS,
    seed=stable_seed(STAGE34_BOOTSTRAP_SEED, "jepa", "deletion_control"),
    alpha=HOLM_ALPHA,
)
mode_residual = {
    mode: float(np.mean(residual_gain[evaluation["mode"] == mode])) for mode in MODE_LABELS
}
JEPA_PREDICTIVE_SUFFICIENCY_GATE = bool(
    np.mean(residual_gain) <= MAX_RESIDUAL_RELATIVE_IMPROVEMENT
    and residual_ci[1] <= MAX_RESIDUAL_CI_UPPER
    and np.mean(deletion_gain) >= MIN_DELETION_CONTROL_IMPROVEMENT
    and deletion_ci[0] > 0
    and all(value <= MAX_RESIDUAL_CI_UPPER for value in mode_residual.values())
)
JEPA_SUFFICIENCY_SUMMARY = {
    "rows": len(base_error),
    "mean_residual_relative_improvement": float(np.mean(residual_gain)),
    "residual_improvement_ci95": residual_ci,
    "mean_deletion_control_improvement": float(np.mean(deletion_gain)),
    "deletion_control_ci95": deletion_ci,
    "mode_residual_improvements": mode_residual,
    "base_mse": float(np.mean(base_error)),
    "enriched_mse": float(np.mean(enriched_error)),
    "deleted_coordinate_mse": float(np.mean(deleted_error)),
    "passed": JEPA_PREDICTIVE_SUFFICIENCY_GATE,
}
JEPA_SUFFICIENCY_ROWS = [{
    "record_id": int(evaluation["record_id"][index]),
    "trajectory_id": int(evaluation["group"][index]),
    "mode": str(evaluation["mode"][index]),
    "word": str(evaluation["word"][index]),
    "word_length": int(evaluation["length"][index]),
    "base_mse": float(base_error[index]),
    "enriched_mse": float(enriched_error[index]),
    "deleted_coordinate_mse": float(deleted_error[index]),
    "residual_relative_improvement": float(residual_gain[index]),
    "deletion_control_improvement": float(deletion_gain[index]),
} for index in range(len(base_error))]
write_csv(EVIDENCE_DIR / "jepa_predictive_sufficiency_rows.csv", JEPA_SUFFICIENCY_ROWS)
write_json(EVIDENCE_DIR / "jepa_predictive_sufficiency_summary.json", jsonable(JEPA_SUFFICIENCY_SUMMARY))
write_json(CHECKPOINT_DIR / "jepa_predictive_sufficiency_complete.json", {
    "gate_passed": JEPA_PREDICTIVE_SUFFICIENCY_GATE, "rows": len(base_error),
})
print(json.dumps(JEPA_SUFFICIENCY_SUMMARY, indent=2))
'''


runtime_install = r'''
# Install and configure the exact official JEPA runtime only when sufficiency passes.
MODEL_RUNTIME_READY = False
if JEPA_PREDICTIVE_SUFFICIENCY_GATE:
    PINNED_RUNTIME = [
        "einops==0.8.1", "tensordict==0.9.1", "timm==1.0.19",
        "omegaconf==2.3.0", "hydra-core==1.3.2", "PyYAML==6.0.2",
        "huggingface_hub==0.36.2", "hf-xet==1.5.1", "gym==0.23.1",
        "pygame==2.6.1", "pymunk==6.8.0", "opencv-python-headless==4.11.0.86",
        "shapely==2.1.2", "lpips==0.1.4", "ruamel.yaml==0.18.10",
        "scikit-learn==1.6.1",
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PINNED_RUNTIME], check=True)
    import torch
    import yaml
    if not torch.cuda.is_available():
        raise RuntimeError("JEPA causal-use gate passed its precursor; select a GPU runtime and Run all")
    if tuple(int(value) for value in torch.__version__.split("+")[0].split(".")[:2]) < (2, 7):
        raise RuntimeError(f"JEPA-WMs requires torch>=2.7; found {torch.__version__}")
    CACHE_ROOT = Path("/content/drive/MyDrive/cf_faithfulness_cache")
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(CACHE_ROOT / "torch")
    os.environ["HF_HOME"] = str(CACHE_ROOT / "huggingface")
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    MODEL_RUNTIME_READY = True
    print({"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__})
else:
    print("Skipping JEPA runtime installation: predictive-sufficiency gate failed.")
'''


model_function_names = [
    "to_model_observation", "configure_repo", "verify_pretrained_assets",
    "cached_verified_asset", "validate_world_model", "load_world_model",
    "unload_world_model", "model_action_tensor", "layer_tokens_full",
    "forward_with_carriers",
]
runtime_functions = function_sources(STAGE34.model_helpers, model_function_names)
runtime_functions += "\n\n\n" + function_sources(
    STAGE34.design_and_runtime_helpers,
    ["count_sketch", "forward_with_trace"],
)
runtime_functions += "\n\n\n" + function_sources(
    STAGE34.construction_and_models,
    ["feature_tensor_from_outputs"],
)


model_runtime = r'''# Define the exact Stage 34 JEPA hooks and an upstream-truth input adapter.
DECODER_SEED = STAGE34_DECODER_SEED
PROVENANCE_COUNTS = {
    "native_forward_pred_calls": {"jepa": 0},
    "native_predicted_word_sequences": {"jepa": 0},
}
''' + runtime_functions + r'''


WORD_BY_NAME = {
    "zero1": {"name": "zero1", "angles": [0.0], "magnitudes": [0.0], "length": 1}
}


def word_actions(record, specification):
    state = np.asarray(record["state"], dtype=np.float64)
    toward = state[2:4] - state[:2]
    toward = toward / max(float(np.linalg.norm(toward)), 1e-12)
    angle = np.deg2rad(float(specification["angles"][0]))
    rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    pulse = rotation @ toward * float(specification["magnitudes"][0])
    return np.repeat(pulse[None], FRAMESKIP, axis=0).astype(np.float32), {}


def encoded_upstream_initial(bundle, record):
    with np.load(truth_path(record), allow_pickle=False) as payload:
        observation = to_model_observation(payload["initial_visual"], payload["initial_proprio"])
    with torch.inference_mode():
        encoded = bundle["model"].encode(observation)
    return {key: value.detach() for key, value in encoded.items()}


def grouped_model_words_upstream(bundle, record, names, intervention_lookup=None):
    initial = encoded_upstream_initial(bundle, record)
    outputs, traces = {}, {}
    for length in sorted({WORD_BY_NAME[name]["length"] for name in names}):
        selected = [name for name in names if WORD_BY_NAME[name]["length"] == length]
        actions = np.stack([word_actions(record, WORD_BY_NAME[name])[0] for name in selected])
        action_tensor = model_action_tensor(bundle["preprocessor"], actions, length)
        step_edits = None
        if intervention_lookup:
            step_edits = {}
            for step in range(length):
                values, any_edit = [], False
                for name in selected:
                    delta = intervention_lookup.get((name, step))
                    if delta is None:
                        values.append(np.zeros((256, bundle["carrier_width"]), dtype=np.float32))
                    else:
                        values.append(np.asarray(delta, dtype=np.float32))
                        any_edit = True
                if any_edit:
                    step_edits[step] = torch.as_tensor(np.stack(values), device="cuda")
        visual, proprio, carrier = forward_with_trace(
            bundle, initial, action_tensor, length, intervention_by_step=step_edits
        )
        visual = visual.detach().float().cpu().numpy()
        proprio = proprio.detach().float().cpu().numpy()
        carrier = carrier.detach().float().cpu().numpy()
        for index, name in enumerate(selected):
            outputs[name] = (visual[index], proprio[index])
            traces[name] = carrier[index]
    return outputs, traces


def carrier_state_sketch(carrier):
    return count_sketch(
        np.asarray(carrier).reshape(1, -1), STATE_CARRIER_SKETCH_DIM,
        stable_seed(STAGE34_CONTROL_SEED, "jepa", "state_carrier_sketch"),
    )[0]


with np.load(STAGE34_ROOT / "predictive_charts/decoder_jepa.npz", allow_pickle=False) as payload:
    JEPA_DECODER = {key: payload[key] for key in payload.files}


def final_grounded_prediction(outputs, name):
    tensor, _ = feature_tensor_from_outputs(outputs, [name])
    grounded = tensor.astype(np.float64) @ JEPA_DECODER["weight"] + JEPA_DECODER["intercept"]
    return grounded[0, WORD_BY_NAME[name]["length"] - 1]


def model_payload(record):
    with np.load(model_path("jepa", record), allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}
'''


causal_design = r'''
# Freeze JEPA matched fiber/state pairs and carrier subspaces before live checkpoint inference.
JEPA_CAUSAL_EVALUATED = False
JEPA_CAUSAL_GATE = False
JEPA_CAUSAL_SUMMARY = {}
PAIR_MANIFEST = []
NATURAL_SKETCHES = None
PRIMARY_SUBSPACE = None
CONTROL_BASIS = None
if JEPA_PREDICTIVE_SUFFICIENCY_GATE:
    calibration_carriers, calibration_q = [], []
    for record in SELECTED_RECORDS["calibration"]:
        payload = model_payload(record)
        calibration_carriers.append(payload["state_carrier"].reshape(-1))
        with np.load(transition_path(record, "calibration"), allow_pickle=False) as transition:
            calibration_q.append(transition["source_coordinates"][0])
    calibration_carriers = np.asarray(calibration_carriers, dtype=np.float64)
    calibration_q = np.asarray(calibration_q, dtype=np.float64)
    PRIMARY_SUBSPACE = fit_supervised_subspace(
        calibration_carriers, calibration_q, rank=CANONICAL_RANK, ridge=1e-3
    )
    CONTROL_BASIS = fit_matched_control_basis(
        calibration_carriers, PRIMARY_SUBSPACE, CANONICAL_RANK
    )
    atomic_npz(
        OUT / "jepa_carrier_alignment.npz",
        mean=np.asarray(PRIMARY_SUBSPACE["mean"]),
        scale=np.asarray(PRIMARY_SUBSPACE["scale"]),
        basis=np.asarray(PRIMARY_SUBSPACE["basis"]),
        control_basis=np.asarray(CONTROL_BASIS),
        singular_values=np.asarray(PRIMARY_SUBSPACE["singular_values"]),
    )

    q, residual = [], []
    for record in SELECTED_RECORDS["evaluation"]:
        payload = model_payload(record)
        residual.append(payload["state_carrier_sketch"])
        with np.load(transition_path(record, "evaluation"), allow_pickle=False) as transition:
            q.append(transition["source_coordinates"][0])
    q = np.asarray(q, dtype=np.float64)
    NATURAL_SKETCHES = np.asarray(residual, dtype=np.float64)
    records = SELECTED_RECORDS["evaluation"]
    modes = np.asarray([record["mode"] for record in records])
    trajectories = np.asarray([record["trajectory_id"] for record in records])

    def select_pairs(pairs, kind):
        selected, used = [], set()
        for mode in MODE_LABELS:
            candidates = [pair for pair in pairs if records[int(pair[0])]["mode"] == mode]
            for pair in candidates:
                trajectory = int(records[int(pair[0])]["trajectory_id"])
                if trajectory in used:
                    continue
                selected.append(pair)
                used.add(trajectory)
                if sum(records[int(value[0])]["mode"] == mode for value in selected) >= ACTIVE_CAUSAL_PAIRS_PER_MODE:
                    break
        expected = len(MODE_LABELS) * ACTIVE_CAUSAL_PAIRS_PER_MODE
        if len(selected) != expected:
            raise RuntimeError(f"{kind} pair panel has {len(selected)} rows; expected {expected}")
        return np.asarray(selected, dtype=np.int64)

    for kind in ["fiber", "state"]:
        pairs = matched_fiber_pairs(q, NATURAL_SKETCHES, modes, trajectories, kind=kind)
        for pair_index, (base_index, donor_index) in enumerate(select_pairs(pairs, kind)):
            base, donor = records[int(base_index)], records[int(donor_index)]
            PAIR_MANIFEST.append({
                "kind": kind, "pair_index": pair_index,
                "base_index": int(base_index), "donor_index": int(donor_index),
                "base_record_id": int(base["record_id"]),
                "donor_record_id": int(donor["record_id"]),
                "base_trajectory_id": int(base["trajectory_id"]),
                "donor_trajectory_id": int(donor["trajectory_id"]),
                "mode": str(base["mode"]),
            })
    write_json(OUT / "jepa_causal_pair_manifest.json", PAIR_MANIFEST)
    print({"frozen_causal_pairs": len(PAIR_MANIFEST), "rank": CANONICAL_RANK})
else:
    print("Skipping causal pair construction: JEPA predictive sufficiency failed.")
'''


causal_inference = r'''
# Run or resume native JEPA interventions, validating unpatched replay for every pair.
JEPA_CAUSAL_ROWS = []
if JEPA_PREDICTIVE_SUFFICIENCY_GATE:
    if not MODEL_RUNTIME_READY:
        raise RuntimeError("JEPA model runtime was not initialized")
    REPO = configure_repo()
    bundle = load_world_model("jepa_wm_pusht")
    try:
        verify_pretrained_assets()
        width = int(bundle["carrier_width"])
        records = SELECTED_RECORDS["evaluation"]
        for completed, pair in enumerate(PAIR_MANIFEST, start=1):
            shard = CAUSAL_SHARD_DIR / (
                f"{pair['kind']}_{pair['pair_index']:02d}_"
                f"{pair['base_record_id']}_{pair['donor_record_id']}.json"
            )
            identity = (
                f"{PROTOCOL_ID}:{RUN_SIGNATURE}:{SOURCE_COMMIT}:jepa:"
                f"{pair['kind']}:{pair['base_record_id']}:{pair['donor_record_id']}"
            )
            digest = Path(str(shard) + ".sha256")
            if shard.is_file() and digest.is_file() and digest.read_text().strip() == sha256_file(shard):
                cached = json.loads(shard.read_text())
                if cached.get("identity") != identity:
                    raise RuntimeError(f"stale causal shard identity: {shard}")
                JEPA_CAUSAL_ROWS.extend(cached["rows"])
                continue

            base = records[pair["base_index"]]
            donor = records[pair["donor_index"]]
            base_outputs, base_traces = grouped_model_words_upstream(bundle, base, [CAUSAL_WORD])
            donor_outputs, donor_traces = grouped_model_words_upstream(bundle, donor, [CAUSAL_WORD])
            base_prediction = final_grounded_prediction(base_outputs, CAUSAL_WORD)
            donor_prediction = final_grounded_prediction(donor_outputs, CAUSAL_WORD)
            base_cached = model_payload(base)
            donor_cached = model_payload(donor)
            base_lookup = {str(value): index for index, value in enumerate(base_cached["word_names"])}
            donor_lookup = {str(value): index for index, value in enumerate(donor_cached["word_names"])}
            replay_errors = {
                "base_prediction": float(np.max(np.abs(
                    base_prediction - base_cached["grounded_predictions"][base_lookup[CAUSAL_WORD], 0]
                ))),
                "donor_prediction": float(np.max(np.abs(
                    donor_prediction - donor_cached["grounded_predictions"][donor_lookup[CAUSAL_WORD], 0]
                ))),
                "base_carrier": float(np.max(np.abs(
                    base_traces[CAUSAL_WORD][0] - base_cached["state_carrier"]
                ))),
                "donor_carrier": float(np.max(np.abs(
                    donor_traces[CAUSAL_WORD][0] - donor_cached["state_carrier"]
                ))),
            }
            if max(replay_errors.values()) > MAX_REPLAY_ABS_ERROR:
                raise RuntimeError(f"live JEPA replay exceeded tolerance: {replay_errors}")

            delta = (
                donor_traces[CAUSAL_WORD][0] - base_traces[CAUSAL_WORD][0]
            ).reshape(-1).astype(np.float64)
            aligned, fiber = split_carrier_delta(delta, PRIMARY_SUBSPACE)
            primary_delta = fiber if pair["kind"] == "fiber" else aligned
            random_delta = project_delta_to_basis(delta, PRIMARY_SUBSPACE, CONTROL_BASIS)
            primary_norm = float(np.linalg.norm(primary_delta))
            random_norm = float(np.linalg.norm(random_delta))
            if primary_norm > 1e-12 and random_norm > 1e-12:
                random_delta *= primary_norm / random_norm
            conditions = {
                "primary": primary_delta,
                "random_matched_subspace": random_delta,
                "full_swap_positive": delta,
            }
            condition_predictions, condition_sketches = {}, {}
            for condition, edit in conditions.items():
                edit_field = edit.reshape(256, width).astype(np.float32)
                outputs, traces = grouped_model_words_upstream(
                    bundle, base, [CAUSAL_WORD],
                    intervention_lookup={(CAUSAL_WORD, 0): edit_field},
                )
                condition_predictions[condition] = final_grounded_prediction(outputs, CAUSAL_WORD)
                condition_sketches[condition] = carrier_state_sketch(
                    base_traces[CAUSAL_WORD][0] + edit_field
                )
            intended = donor_prediction - base_prediction
            intended_norm = float(np.linalg.norm(intended))
            baseline_error = float(np.mean((base_prediction - donor_prediction) ** 2))
            shard_rows = []
            for condition, prediction in condition_predictions.items():
                observed = prediction - base_prediction
                patched_error = float(np.mean((prediction - donor_prediction) ** 2))
                shard_rows.append({
                    "model": "jepa", "kind": pair["kind"], "condition": condition,
                    "base_record_id": pair["base_record_id"],
                    "donor_record_id": pair["donor_record_id"],
                    "trajectory_id": pair["base_trajectory_id"],
                    "mode": pair["mode"], "word": CAUSAL_WORD,
                    "effect_cosine": float(cosine_rows(
                        np.asarray([observed]), np.asarray([intended])
                    )[0]),
                    "error_gain": float(
                        (baseline_error - patched_error) / max(baseline_error, 1e-12)
                    ),
                    "effect_norm": float(np.linalg.norm(observed)),
                    "intended_norm": intended_norm,
                    "fiber_effect_ratio": float(
                        np.linalg.norm(observed) / max(intended_norm, 1e-6)
                    ),
                    "ood_ratio": float(intervention_ood_ratio(
                        np.asarray([condition_sketches[condition]]), NATURAL_SKETCHES
                    )[0]),
                })
            write_json(shard, {
                "identity": identity, "pair": pair,
                "replay_max_abs_errors": replay_errors, "rows": shard_rows,
            })
            digest.write_text(sha256_file(shard) + "\n")
            JEPA_CAUSAL_ROWS.extend(shard_rows)
            write_json(CHECKPOINT_DIR / "jepa_causal_progress.json", {
                "completed_pairs": completed, "total_pairs": len(PAIR_MANIFEST),
                "last_pair": pair, "resumable": True,
            })
            print(f"completed causal pair {completed}/{len(PAIR_MANIFEST)}")
    finally:
        unload_world_model(bundle)

    expected_rows = len(PAIR_MANIFEST) * 3
    if len(JEPA_CAUSAL_ROWS) != expected_rows:
        raise RuntimeError(f"causal row count {len(JEPA_CAUSAL_ROWS)} != {expected_rows}")
    JEPA_CAUSAL_SUMMARY = summarize_causal_rows(
        JEPA_CAUSAL_ROWS, MODE_LABELS,
        minimum_retention=MIN_STATE_EFFECT_RETENTION,
        minimum_cosine=MIN_STATE_INTERVENTION_COSINE,
        minimum_control_advantage=MIN_CAUSAL_CONTROL_ADVANTAGE,
        maximum_fiber_ratio=MAX_FIBER_EFFECT_RATIO,
        maximum_ood_rate=MAX_INTERVENTION_OOD_RATE,
    )
    JEPA_CAUSAL_EVALUATED = True
    JEPA_CAUSAL_GATE = bool(JEPA_CAUSAL_SUMMARY["passed"])
    write_csv(EVIDENCE_DIR / "jepa_predictive_fiber_intervention_rows.csv", JEPA_CAUSAL_ROWS)
    write_json(EVIDENCE_DIR / "jepa_predictive_fiber_causal_summary.json", jsonable(JEPA_CAUSAL_SUMMARY))
    write_json(CHECKPOINT_DIR / "jepa_causal_use_complete.json", {
        "gate_passed": JEPA_CAUSAL_GATE, "rows": len(JEPA_CAUSAL_ROWS),
    })
    print(json.dumps(JEPA_CAUSAL_SUMMARY, indent=2))
else:
    print("Skipping native JEPA causal use: predictive-sufficiency gate failed.")
'''


decision = r'''
# Derive the split-path decision without reviving the rejected shared-model claim.
DECISION = derive_stage342_decision(
    Stage342Gates(
        upstream_binding=UPSTREAM_BINDING_GATE,
        stage341_binding=STAGE341_BINDING_GATE,
        jepa_action_specificity=JEPA_ACTION_SPECIFICITY_GATE,
        jepa_predictive_sufficiency=JEPA_PREDICTIVE_SUFFICIENCY_GATE,
        jepa_causal_evaluated=JEPA_CAUSAL_EVALUATED,
        jepa_causal_use=JEPA_CAUSAL_GATE,
        dino_diagonal_recoverability=DINO_DIAGONAL_RECOVERABILITY_GATE,
    ),
    run_mode=RUN_MODE,
)
DECISION.update({
    "protocol_id": PROTOCOL_ID,
    "protocol_sha256": NOTEBOOK_PROTOCOL_SHA256,
    "run_signature": RUN_SIGNATURE,
    "source_commit": SOURCE_COMMIT,
    "upstream_stage34_run_signature": UPSTREAM_STAGE34_RUN_SIGNATURE,
    "upstream_stage341_run_signature": UPSTREAM_STAGE341_RUN_SIGNATURE,
    "dino_calibration_summary": DINO_SUMMARY,
    "jepa_predictive_sufficiency_summary": JEPA_SUFFICIENCY_SUMMARY,
    "jepa_causal_summary": JEPA_CAUSAL_SUMMARY,
    "shared_abstraction_claimed": False,
    "post_outcome_diagnostic": True,
    "confirmation_eligible": False,
})
write_json(OUT / "stage34_2_decision.json", jsonable(DECISION))

figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
axes[0].bar(
    ["raw", "diagonal", "full affine", "state-only"],
    [DINO_SUMMARY["raw_evaluation_mse"], DINO_SUMMARY["diagonal_evaluation_mse"],
     DINO_SUMMARY["full_affine_evaluation_mse"], DINO_SUMMARY["action_blind_state_mse"]],
    color=["#fb923c", "#ea580c", "#9a3412", "#64748b"],
)
axes[0].set_title("DINO grounded response MSE")
axes[0].tick_params(axis="x", rotation=25)
axes[1].bar(
    ["residual gain", "deletion control"],
    [JEPA_SUFFICIENCY_SUMMARY["mean_residual_relative_improvement"],
     JEPA_SUFFICIENCY_SUMMARY["mean_deletion_control_improvement"]],
    color=["#60a5fa", "#2563eb"],
)
axes[1].axhline(MAX_RESIDUAL_RELATIVE_IMPROVEMENT, color="black", linestyle="--")
axes[1].set_title("JEPA predictive sufficiency")
if JEPA_CAUSAL_EVALUATED:
    axes[2].bar(
        ["retention", "cosine", "control", "fiber ratio"],
        [JEPA_CAUSAL_SUMMARY["mean_state_effect_retention"],
         JEPA_CAUSAL_SUMMARY["mean_state_effect_cosine"],
         JEPA_CAUSAL_SUMMARY["mean_control_advantage"],
         JEPA_CAUSAL_SUMMARY["mean_fiber_effect_ratio"]],
        color="#2563eb",
    )
    axes[2].set_title("JEPA native causal use")
else:
    axes[2].text(0.5, 0.5, "Causal gate skipped\n(sufficiency failed)", ha="center", va="center")
    axes[2].set_axis_off()
figure.suptitle(f"Stage 34.2: {DECISION['status']}")
figure.tight_layout()
figure.savefig(PLOT_DIR / "stage34_2_split_path_summary.png", dpi=180)
plt.show()

if DECISION["status"].startswith("jepa_response_state_insufficient"):
    interpretation = "JEPA's action-specific response state is not predictively sufficient; native causal testing was correctly skipped."
elif DECISION["status"] == "jepa_response_state_not_causally_used":
    interpretation = "JEPA passed observational sufficiency but failed the native causal-use gate."
elif DECISION["passed"]:
    interpretation = "JEPA passed action specificity, predictive sufficiency, and native causal use on the reused panel. This remains diagnostic and single-checkpoint evidence."
else:
    interpretation = "The split-path diagnostic did not support continuation to a confirmatory claim."
if DINO_DIAGONAL_RECOVERABILITY_GATE:
    interpretation += " DINO's failure is recoverable by a calibration-only diagonal map, suggesting scale/bias limitation rather than absence of action signal."
else:
    interpretation += " DINO was not recovered by the registered low-capacity calibration, so its defect is not merely per-observable scale/bias."
(OUT / "AUTOMATIC_INTERPRETATION.md").write_text(
    f"# Automatic Stage 34.2 interpretation\n\nStatus: **{DECISION['status'].upper()}**\n\n{interpretation}\n\n"
    "No shared JEPA--DINO abstraction or confirmation claim is made.\n"
)
(OUT / "FAILURE_TRACE.txt").write_text("NONE\n")
print(json.dumps({"status": DECISION["status"], "passed": DECISION["passed"]}, indent=2))
'''


package = r'''
# Package all compact diagnostics and resumable causal shards.
source_identity = {
    "repository": EXPERIMENT_REPOSITORY,
    "source_ref": EXPERIMENT_SOURCE_REF,
    "commit": SOURCE_COMMIT,
    "protocol_sha256": NOTEBOOK_PROTOCOL_SHA256,
    "files": {},
}
for relative in [EXPERIMENT_NOTEBOOK_PATH, EXPERIMENT_BUILDER_PATH, EXPERIMENT_NUMERICAL_PATH]:
    path = SOURCE_REPOSITORY / relative
    if not path.is_file():
        raise FileNotFoundError(f"missing committed Stage 34.2 source: {relative}")
    source_identity["files"][relative] = sha256_file(path)
write_json(OUT / "source_identity.json", source_identity)
write_json(OUT / "timings.json", {
    "elapsed_seconds": time.time() - RUN_STARTED_AT,
    "gpu_causal_gate_required": bool(JEPA_PREDICTIVE_SUFFICIENCY_GATE),
    "jepa_causal_pairs": len(PAIR_MANIFEST),
    "jepa_causal_rows": len(JEPA_CAUSAL_ROWS),
    "stage34_shards_reused": len(consumed_stage34),
    "resumable_pair_shards": len(list(CAUSAL_SHARD_DIR.glob("*.json"))),
})


def manifest_rows(root, excluded=()):
    root = Path(root)
    excluded = {Path(value) for value in excluded}
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(value == path or value in path.parents for value in excluded):
            continue
        rows.append({
            "path": str(path.relative_to(root)),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        })
    return rows


archive_path = OUT / f"stage34_2_split_path_result_bundle_{RUN_SIGNATURE[:12]}.zip"
manifest_path = OUT / "result_zip_manifest.json"
archive_path.unlink(missing_ok=True)
rows = manifest_rows(OUT, excluded=[archive_path, manifest_path])
write_json(manifest_path, rows)
with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path != archive_path:
            archive.write(path, arcname=str(path.relative_to(OUT)))
print(json.dumps({
    "bundle": str(archive_path), "bytes": archive_path.stat().st_size,
    "files": len(rows) + 1, "status": DECISION["status"],
}, indent=2))
if DOWNLOAD_RESULTS:
    from google.colab import files
    files.download(str(archive_path))
'''


cells = [
    markdown(introduction, "stage342-00"),
    code(configuration, "stage342-01"),
    code(setup, "stage342-02"),
    code(binding, "stage342-03"),
    code(data_helpers, "stage342-04"),
    code(dino_diagnostic, "stage342-05"),
    code(jepa_sufficiency, "stage342-06"),
    code(runtime_install, "stage342-07"),
    code(model_runtime, "stage342-08"),
    code(causal_design, "stage342-09"),
    code(causal_inference, "stage342-10"),
    code(decision, "stage342-11"),
    code(package, "stage342-12"),
]

protocol_sources = [cell["source"].strip() for cell in cells]
protocol_digest = hashlib.sha256(
    json.dumps(protocol_sources, ensure_ascii=False, separators=(",", ":")).encode()
).hexdigest()
cells[1]["source"] = cells[1]["source"].replace("__PROTOCOL_DIGEST__", protocol_digest)

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "L4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
TARGET.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(f"wrote {TARGET} ({len(cells)} cells, protocol {protocol_digest})")
