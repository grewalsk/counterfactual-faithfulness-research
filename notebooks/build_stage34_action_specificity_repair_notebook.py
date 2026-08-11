"""Build the deterministic Stage 34.1 action-specificity repair notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "34_1_action_specificity_repair.ipynb"


def markdown(text: str, cell_id: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": text.strip() + "\n",
    }


def code(text: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": text.strip() + "\n",
    }


introduction = r'''
# Stage 34.1: leakage-free action-specificity repair

## Decision before computation

Stage 34 completed without an execution failure and passed its simulator-only
rank gate, but stopped at action specificity.  Both checkpoints strongly beat
within-length action shuffling.  They failed only against a comparator that
mapped exact physical state to one concatenated response atlas.  Because each
target column permanently identified an action word and prefix, that regression
was not action blind.

This notebook preserves Stage 34 unchanged and performs one bounded diagnostic
repair on its frozen artifacts.  It reshapes every response into rows of the
form

\[
  (\text{state},\ \text{length},\ \text{prefix step})
  \longrightarrow \Delta\phi \in \mathbb{R}^{11}.
\]

The repaired state-only comparator receives identical inputs for different
action words at the same state, length, and prefix step.  It therefore cannot
recover action identity from output-column position.  A capacity-matched
state-plus-action regression is included as a physical action-necessity
positive control.  Model predictions must independently beat both the
within-length derangement and the leakage-free state-only baseline.

## Claim boundary

This is a post-outcome diagnostic repair, not a new confirmation and not a
rerun of Stage 34.  A pass only authorizes continuation to the still-unobserved
predictive-sufficiency and causal gates.  It does not establish predictive
sufficiency, causal use, two-sided commutativity, planning value, or a general
shared abstraction.  All consumed files are checked against the frozen Stage
34 raw manifest before the locked evaluation is scored.
'''


configuration = r'''
# Frozen repair contract. Edit RUN_MODE only for plumbing; smoke is never evidence.
RUN_MODE = "pilot"
DOWNLOAD_RESULTS = True
MOUNT_DRIVE = True

PROTOCOL_ID = "stage34.1-leakage-free-action-specificity-repair-v1"
NOTEBOOK_PROTOCOL_SHA256 = "__PROTOCOL_DIGEST__"
EVIDENCE_STATUS = "POST_OUTCOME_DIAGNOSTIC_REPAIR_NOT_CONFIRMATION"

EXPERIMENT_REPOSITORY = "grewalsk/counterfactual-faithfulness-research"
EXPERIMENT_SOURCE_REF = "codex/stage34-predictive-fiber-abstraction"
EXPERIMENT_NOTEBOOK_PATH = "notebooks/34_1_action_specificity_repair.ipynb"
EXPERIMENT_BUILDER_PATH = "notebooks/build_stage34_action_specificity_repair_notebook.py"
EXPERIMENT_NUMERICAL_PATH = "src/cf_faithfulness/stage34_action_specificity_repair.py"

UPSTREAM_PROTOCOL_ID = "stage34-predictive-fiber-causal-abstraction-v1"
UPSTREAM_RUN_SIGNATURE = "d3f4f88426afff4d964bb4f1f1556c94ec3613b667edd9403ddfcd0fd78ded84"
UPSTREAM_SOURCE_COMMIT = "db130a3d25505b7fa69efbcd88009365cb266688"
UPSTREAM_NOTEBOOK_PROTOCOL_SHA256 = "e934fa6507084fa4c2b850b20013eb6bc61e7914020d87cbe256e3ee23fe1425"
UPSTREAM_RAW_MANIFEST_SHA256 = "2d2cf86fdeae5cb1034535104782dc526b8203b2e567e081ed530dbd288cb47e"

DRIVE_STAGE34_ROOT = "/content/drive/MyDrive/counterfactual_faithfulness_stage34_pfca"
DRIVE_OUTPUT_ROOT = "/content/drive/MyDrive/counterfactual_faithfulness_stage34_1_action_specificity"

SEED = 341101
RFF_SEED = 341131
BOOTSTRAP_SEED = 341159
CONTROL_SEED = 341173
BOOTSTRAP_DRAWS = 2000 if RUN_MODE == "pilot" else 100
RFF_WIDTH = 256 if RUN_MODE == "pilot" else 32
RIDGE_PENALTIES = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
HOLM_ALPHA = 0.05

FRAMESKIP = 5
MAX_WORD_LENGTH = 8
MODE_LABELS = ["free", "pre_contact", "contact", "post_contact"]
EVALUATION_WORD_NAMES = [
    "AABAB", "BABAA", "AABBAB", "BABAAB",
    "AAABBAB", "BABAAAB", "AABBABAB", "BABAABBA",
]
ZERO_WORD_NAMES = {length: f"zero{length}" for length in range(1, 9)}
MODEL_SHORT_NAMES = ["jepa", "dino"]

MIN_ACTION_SHUFFLE_ADVANTAGE = 0.10
MIN_ACTION_BLIND_ADVANTAGE = 0.00
MIN_PHYSICAL_ACTION_NECESSITY = 0.10
MAX_ACTION_BLIND_PREDICTION_SPREAD = 1e-12

assert RUN_MODE in {"smoke", "pilot"}
assert len(EVALUATION_WORD_NAMES) == 8
assert {len(word) for word in EVALUATION_WORD_NAMES} == {5, 6, 7, 8}
assert all(sum(len(word) == length for word in EVALUATION_WORD_NAMES) == 2 for length in range(5, 9))
'''


environment = r'''
# Mount Drive, bind this execution to repository source, and create an isolated output root.
import csv
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if MOUNT_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")

REPOSITORY = Path("/content/counterfactual-faithfulness-stage341")
REMOTE = f"https://github.com/{EXPERIMENT_REPOSITORY}.git"
if (REPOSITORY / ".git").is_dir():
    subprocess.run(["git", "-C", str(REPOSITORY), "fetch", "origin", EXPERIMENT_SOURCE_REF], check=True)
    subprocess.run(["git", "-C", str(REPOSITORY), "checkout", "--detach", "FETCH_HEAD"], check=True)
else:
    subprocess.run([
        "git", "clone", "--depth", "1", "--branch", EXPERIMENT_SOURCE_REF,
        REMOTE, str(REPOSITORY),
    ], check=True)

SOURCE_COMMIT = subprocess.check_output(
    ["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"], text=True
).strip()
sys.path.insert(0, str(REPOSITORY / "src"))

from cf_faithfulness.stage34_action_specificity_repair import (
    Stage341Gates,
    action_blind_context_features,
    action_prefix_features,
    action_response_path_rows,
    clustered_bootstrap_interval,
    deranged_word_rows,
    derive_stage341_decision,
    fit_grouped_rff_ridge,
    grouped_record_mse,
    predict_grouped_rff_ridge,
    relative_advantage,
)


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
        raise ValueError(f"refusing to write empty evidence table {path}")
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


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


def stable_seed(*parts):
    payload = ":".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32 - 1)


CONFIG = {
    name: globals()[name]
    for name in [
        "RUN_MODE", "PROTOCOL_ID", "NOTEBOOK_PROTOCOL_SHA256", "EVIDENCE_STATUS",
        "EXPERIMENT_REPOSITORY", "EXPERIMENT_SOURCE_REF", "EXPERIMENT_NOTEBOOK_PATH",
        "EXPERIMENT_BUILDER_PATH", "EXPERIMENT_NUMERICAL_PATH",
        "UPSTREAM_PROTOCOL_ID", "UPSTREAM_RUN_SIGNATURE", "UPSTREAM_SOURCE_COMMIT",
        "UPSTREAM_NOTEBOOK_PROTOCOL_SHA256", "UPSTREAM_RAW_MANIFEST_SHA256",
        "SEED", "RFF_SEED", "BOOTSTRAP_SEED", "CONTROL_SEED", "BOOTSTRAP_DRAWS",
        "RFF_WIDTH", "RIDGE_PENALTIES", "HOLM_ALPHA", "FRAMESKIP",
        "MAX_WORD_LENGTH", "MODE_LABELS", "EVALUATION_WORD_NAMES", "ZERO_WORD_NAMES",
        "MODEL_SHORT_NAMES", "MIN_ACTION_SHUFFLE_ADVANTAGE", "MIN_ACTION_BLIND_ADVANTAGE",
        "MIN_PHYSICAL_ACTION_NECESSITY", "MAX_ACTION_BLIND_PREDICTION_SPREAD",
    ]
}
RUN_SIGNATURE = hashlib.sha256(
    json.dumps(CONFIG, sort_keys=True, allow_nan=False).encode()
).hexdigest()
OUT = Path(DRIVE_OUTPUT_ROOT) / f"{RUN_MODE}_{RUN_SIGNATURE[:12]}"
EVIDENCE_DIR = OUT / "evaluation_evidence"
PLOT_DIR = OUT / "plots"
for directory in [OUT, EVIDENCE_DIR, PLOT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

FAILURES = []
RUN_STARTED_AT = time.time()
write_json(OUT / "config.json", {**CONFIG, "run_signature": RUN_SIGNATURE, "source_commit": SOURCE_COMMIT})
print(json.dumps({
    "protocol": PROTOCOL_ID,
    "run_signature": RUN_SIGNATURE,
    "source_commit": SOURCE_COMMIT,
    "output": str(OUT),
}, indent=2))
'''


source_binding = r'''
# Verify the complete Stage 34 input contract and every consumed raw shard before analysis.
UPSTREAM = Path(DRIVE_STAGE34_ROOT) / f"pilot_{UPSTREAM_RUN_SIGNATURE[:12]}"
if not UPSTREAM.is_dir():
    available = sorted(path.name for path in Path(DRIVE_STAGE34_ROOT).glob("pilot_*"))
    raise FileNotFoundError(
        f"Missing frozen Stage 34 directory {UPSTREAM}. Available pilot directories: {available}"
    )

raw_manifest_path = UPSTREAM / "raw_manifest.json"
if sha256_file(raw_manifest_path) != UPSTREAM_RAW_MANIFEST_SHA256:
    raise RuntimeError("Stage 34 raw manifest hash does not match the audited bundle")
raw_manifest = json.loads(raw_manifest_path.read_text())
manifest_by_path = {row["path"]: row for row in raw_manifest}
if len(manifest_by_path) != len(raw_manifest):
    raise RuntimeError("Stage 34 raw manifest contains duplicate paths")


def verify_upstream(relative):
    relative = str(relative)
    if relative not in manifest_by_path:
        raise RuntimeError(f"unmanifested Stage 34 input: {relative}")
    path = UPSTREAM / relative
    row = manifest_by_path[relative]
    if not path.is_file():
        raise FileNotFoundError(f"missing Stage 34 input: {path}")
    if path.stat().st_size != int(row["bytes"]):
        raise RuntimeError(f"byte-count mismatch for {relative}")
    if sha256_file(path) != row["sha256"]:
        raise RuntimeError(f"content-hash mismatch for {relative}")
    return path


for relative in [
    "config.json", "source_identity.json", "run_provenance_certificate.json",
    "stage34_decision.json", "FAILURE_TRACE.txt",
    "design/selected_calibration_trajectories.json",
    "design/selected_evaluation_trajectories.json",
]:
    verify_upstream(relative)

upstream_config = json.loads((UPSTREAM / "config.json").read_text())
upstream_source = json.loads((UPSTREAM / "source_identity.json").read_text())
upstream_provenance = json.loads((UPSTREAM / "run_provenance_certificate.json").read_text())
upstream_decision = json.loads((UPSTREAM / "stage34_decision.json").read_text())
if upstream_config.get("PROTOCOL_ID") != UPSTREAM_PROTOCOL_ID:
    raise RuntimeError("wrong upstream protocol")
if upstream_config.get("NOTEBOOK_PROTOCOL_SHA256") != UPSTREAM_NOTEBOOK_PROTOCOL_SHA256:
    raise RuntimeError("wrong upstream notebook protocol digest")
if upstream_source.get("resolved_commit") != UPSTREAM_SOURCE_COMMIT:
    raise RuntimeError("wrong upstream source commit")
if upstream_provenance.get("run_signature") != UPSTREAM_RUN_SIGNATURE:
    raise RuntimeError("wrong upstream run signature")
if upstream_decision.get("protocol_decision", {}).get("first_failed_gate") != "action_specificity":
    raise RuntimeError("Stage 34 did not stop at the gate this notebook repairs")
if not upstream_provenance.get("confirmation_eligible", False):
    raise RuntimeError("upstream Stage 34 run was not confirmation eligible")
if (UPSTREAM / "FAILURE_TRACE.txt").read_text().strip() != "NONE":
    raise RuntimeError("upstream Stage 34 contains an execution failure")


def load_split(split):
    payload = json.loads((UPSTREAM / f"design/selected_{split}_trajectories.json").read_text())
    if payload.get("protocol_id") != UPSTREAM_PROTOCOL_ID or payload.get("split") != split:
        raise RuntimeError(f"stale Stage 34 {split} selection")
    if payload.get("model_outputs_used") or payload.get("effect_magnitude_used"):
        raise RuntimeError(f"Stage 34 {split} selection was not model free")
    return payload["records"]


CALIBRATION_RECORDS = load_split("calibration")
EVALUATION_RECORDS = load_split("evaluation")
if len(CALIBRATION_RECORDS) != 64 or len(EVALUATION_RECORDS) != 128:
    raise RuntimeError("Stage 34 pilot record counts changed")
calibration_trajectories = {int(row["trajectory_id"]) for row in CALIBRATION_RECORDS}
evaluation_trajectories = {int(row["trajectory_id"]) for row in EVALUATION_RECORDS}
if calibration_trajectories & evaluation_trajectories:
    raise RuntimeError("calibration and evaluation trajectories overlap")

consumed = []
for record in CALIBRATION_RECORDS + EVALUATION_RECORDS:
    consumed.append(f"truth/truth_{int(record['record_id'])}.npz")
for short in MODEL_SHORT_NAMES:
    for record in EVALUATION_RECORDS:
        consumed.append(f"baseline_shards/{short}_{int(record['record_id'])}.npz")
for index, relative in enumerate(consumed, start=1):
    verify_upstream(relative)
    if index % 100 == 0:
        print(f"verified {index}/{len(consumed)} frozen Stage 34 shards")

UPSTREAM_BINDING_GATE = True
binding_certificate = {
    "upstream_root": str(UPSTREAM),
    "upstream_protocol_id": UPSTREAM_PROTOCOL_ID,
    "upstream_run_signature": UPSTREAM_RUN_SIGNATURE,
    "upstream_source_commit": UPSTREAM_SOURCE_COMMIT,
    "upstream_raw_manifest_sha256": UPSTREAM_RAW_MANIFEST_SHA256,
    "consumed_files": len(consumed) + 7,
    "consumed_shards": len(consumed),
    "calibration_trajectories": len(calibration_trajectories),
    "evaluation_trajectories": len(evaluation_trajectories),
    "trajectory_disjoint": True,
    "upstream_confirmation_eligible": True,
    "upstream_failure_trace": "NONE",
    "binding_passed": True,
}
write_json(OUT / "upstream_binding_certificate.json", binding_certificate)
print(json.dumps(binding_certificate, indent=2))
'''


row_materialization = r'''
# Reshape physical responses into action-prefix rows; action identity is never a target column.
def grounded_state(state):
    value = np.asarray(state, dtype=np.float64)
    if value.shape != (10,):
        raise ValueError("Stage 34 dynamic state must have width 10")
    return np.asarray([
        value[0] / 512.0, value[1] / 512.0,
        value[2] / 512.0, value[3] / 512.0,
        np.sin(value[4]), np.cos(value[4]),
        value[5] / 50.0, value[6] / 50.0,
        value[7] / 50.0, value[8] / 50.0, value[9] / 5.0,
    ], dtype=np.float64)


def load_truth_rows(record):
    path = UPSTREAM / f"truth/truth_{int(record['record_id'])}.npz"
    with np.load(path, allow_pickle=False) as payload:
        names = [str(value) for value in payload["word_names"]]
        lengths = payload["word_lengths"].astype(np.int64)
        target, metadata = action_response_path_rows(
            payload["path_observables"], names, lengths,
            EVALUATION_WORD_NAMES, ZERO_WORD_NAMES,
        )
        action = action_prefix_features(
            payload["actions"], payload["action_mask"], names,
            EVALUATION_WORD_NAMES, lengths, frameskip=FRAMESKIP,
        )
    count = len(target)
    state = np.repeat(grounded_state(record["state"])[None], count, axis=0)
    context = action_blind_context_features(
        state, metadata["length"], metadata["step"],
        [record["mode"]] * count, MODE_LABELS,
        maximum_length=MAX_WORD_LENGTH,
    )
    return {
        "target": target,
        "metadata": metadata,
        "context": context,
        "action": action,
        "trajectory": np.repeat(int(record["trajectory_id"]), count),
        "record": np.repeat(int(record["record_id"]), count),
        "mode": np.repeat(str(record["mode"]), count),
    }


def stack_truth_records(records):
    blocks = [load_truth_rows(record) for record in records]
    return {
        "target": np.concatenate([block["target"] for block in blocks]),
        "context": np.concatenate([block["context"] for block in blocks]),
        "action": np.concatenate([block["action"] for block in blocks]),
        "trajectory": np.concatenate([block["trajectory"] for block in blocks]),
        "record": np.concatenate([block["record"] for block in blocks]),
        "mode": np.concatenate([block["mode"] for block in blocks]),
        "blocks": blocks,
    }


CALIBRATION = stack_truth_records(CALIBRATION_RECORDS)
EVALUATION = stack_truth_records(EVALUATION_RECORDS)
expected_rows_per_record = sum(len(word) for word in EVALUATION_WORD_NAMES)
if expected_rows_per_record != 52:
    raise RuntimeError("evaluation response-row count changed")
if len(CALIBRATION["target"]) != 64 * expected_rows_per_record:
    raise RuntimeError("calibration row count changed")
if len(EVALUATION["target"]) != 128 * expected_rows_per_record:
    raise RuntimeError("evaluation row count changed")
print(json.dumps({
    "target_width": int(EVALUATION["target"].shape[1]),
    "rows_per_record": expected_rows_per_record,
    "calibration_rows": len(CALIBRATION["target"]),
    "evaluation_rows": len(EVALUATION["target"]),
    "action_identity_target_columns": 0,
}, indent=2))
'''


baseline_fit = r'''
# Fit capacity-matched action-blind and state-plus-action physical controls on calibration only.
STATE_ONLY_MODEL = fit_grouped_rff_ridge(
    CALIBRATION["context"], CALIBRATION["target"], CALIBRATION["trajectory"],
    width=RFF_WIDTH, penalties=RIDGE_PENALTIES, folds=4, seed=RFF_SEED,
)
STATE_ACTION_MODEL = fit_grouped_rff_ridge(
    np.column_stack([CALIBRATION["context"], CALIBRATION["action"]]),
    CALIBRATION["target"], CALIBRATION["trajectory"],
    width=RFF_WIDTH, penalties=RIDGE_PENALTIES, folds=4, seed=RFF_SEED + 1,
)
STATE_PREDICTION = predict_grouped_rff_ridge(STATE_ONLY_MODEL, EVALUATION["context"])
STATE_ACTION_PREDICTION = predict_grouped_rff_ridge(
    STATE_ACTION_MODEL,
    np.column_stack([EVALUATION["context"], EVALUATION["action"]]),
)

# Same state, length, and prefix step must yield exactly the same state-only prediction
# for both words. This executable invariant is the central repair.
spreads = []
offset = 0
for block in EVALUATION["blocks"]:
    count = len(block["target"])
    prediction = STATE_PREDICTION[offset:offset + count]
    metadata = block["metadata"]
    for length in range(5, 9):
        for step in range(1, length + 1):
            mask = (metadata["length"] == length) & (metadata["step"] == step)
            spreads.append(float(np.max(np.ptp(prediction[mask], axis=0))))
    offset += count
MAX_OBSERVED_ACTION_BLIND_SPREAD = max(spreads)
LEAKAGE_INVARIANT_GATE = bool(
    MAX_OBSERVED_ACTION_BLIND_SPREAD <= MAX_ACTION_BLIND_PREDICTION_SPREAD
)

state_error, record_order = grouped_record_mse(
    STATE_PREDICTION, EVALUATION["target"], EVALUATION["record"]
)
action_error, action_record_order = grouped_record_mse(
    STATE_ACTION_PREDICTION, EVALUATION["target"], EVALUATION["record"]
)
if not np.array_equal(record_order, action_record_order):
    raise RuntimeError("physical-control record order changed")
record_by_id = {int(row["record_id"]): row for row in EVALUATION_RECORDS}
record_groups = np.asarray([
    int(record_by_id[int(record_id)]["trajectory_id"]) for record_id in record_order
])
record_modes = np.asarray([
    str(record_by_id[int(record_id)]["mode"]) for record_id in record_order
])
physical_advantage = relative_advantage(action_error, state_error)
physical_interval = clustered_bootstrap_interval(
    physical_advantage, record_groups, draws=BOOTSTRAP_DRAWS,
    seed=BOOTSTRAP_SEED, alpha=HOLM_ALPHA,
)
physical_mode_means = {
    mode: float(np.mean(physical_advantage[record_modes == mode]))
    for mode in MODE_LABELS
}
PHYSICAL_ACTION_NECESSITY_GATE = bool(
    np.mean(physical_advantage) >= MIN_PHYSICAL_ACTION_NECESSITY
    and physical_interval[0] > 0
    and all(value > 0 for value in physical_mode_means.values())
)
PHYSICAL_CONTROL_SUMMARY = {
    "state_only_oof_mse": STATE_ONLY_MODEL["oof_mse"],
    "state_plus_action_oof_mse": STATE_ACTION_MODEL["oof_mse"],
    "state_only_evaluation_mse": float(np.mean(state_error)),
    "state_plus_action_evaluation_mse": float(np.mean(action_error)),
    "mean_action_necessity_advantage": float(np.mean(physical_advantage)),
    "action_necessity_ci95": physical_interval,
    "mode_mean_action_necessity": physical_mode_means,
    "max_action_blind_prediction_spread": MAX_OBSERVED_ACTION_BLIND_SPREAD,
    "leakage_invariant_passed": LEAKAGE_INVARIANT_GATE,
    "physical_action_necessity_passed": PHYSICAL_ACTION_NECESSITY_GATE,
}
write_json(EVIDENCE_DIR / "physical_action_necessity_summary.json", jsonable(PHYSICAL_CONTROL_SUMMARY))
print(json.dumps(PHYSICAL_CONTROL_SUMMARY, indent=2))
'''


locked_evaluation = r'''
# Score each frozen model against same-length derangement and the leakage-free baseline.
ACTION_SPECIFICITY_ROWS = []
ACTION_SPECIFICITY_SUMMARY = {}
MODEL_GATES = {}
state_error_by_record = {int(record_id): float(error) for record_id, error in zip(record_order, state_error)}


def load_model_rows(short, record):
    path = UPSTREAM / f"baseline_shards/{short}_{int(record['record_id'])}.npz"
    with np.load(path, allow_pickle=False) as payload:
        names = [str(value) for value in payload["word_names"]]
        return action_response_path_rows(
            payload["grounded_predictions"], names, payload["word_lengths"],
            EVALUATION_WORD_NAMES, ZERO_WORD_NAMES,
        )


for short in MODEL_SHORT_NAMES:
    primary_errors = []
    shuffled_errors = []
    static_errors = []
    groups = []
    modes = []
    for index, record in enumerate(EVALUATION_RECORDS):
        primary_rows, metadata = load_model_rows(short, record)
        truth_rows = EVALUATION["blocks"][index]["target"]
        shuffled_rows = deranged_word_rows(
            primary_rows, metadata,
            seed=stable_seed(CONTROL_SEED, short, record["record_id"]),
        )
        primary_mse = float(np.mean((primary_rows - truth_rows) ** 2))
        shuffled_mse = float(np.mean((shuffled_rows - truth_rows) ** 2))
        state_mse = state_error_by_record[int(record["record_id"])]
        primary_errors.append(primary_mse)
        shuffled_errors.append(shuffled_mse)
        static_errors.append(state_mse)
        groups.append(int(record["trajectory_id"]))
        modes.append(str(record["mode"]))
        ACTION_SPECIFICITY_ROWS.append({
            "model": short,
            "record_id": int(record["record_id"]),
            "trajectory_id": int(record["trajectory_id"]),
            "mode": str(record["mode"]),
            "primary_path_mse": primary_mse,
            "same_length_derangement_mse": shuffled_mse,
            "action_blind_state_mse": state_mse,
        })
    primary_errors = np.asarray(primary_errors)
    shuffled_errors = np.asarray(shuffled_errors)
    static_errors = np.asarray(static_errors)
    groups = np.asarray(groups)
    modes = np.asarray(modes)
    shuffle_advantage = relative_advantage(primary_errors, shuffled_errors)
    state_advantage = relative_advantage(primary_errors, static_errors)
    shuffle_interval = clustered_bootstrap_interval(
        shuffle_advantage, groups, draws=BOOTSTRAP_DRAWS,
        seed=stable_seed(BOOTSTRAP_SEED, short, "derangement"), alpha=HOLM_ALPHA,
    )
    state_interval = clustered_bootstrap_interval(
        state_advantage, groups, draws=BOOTSTRAP_DRAWS,
        seed=stable_seed(BOOTSTRAP_SEED, short, "action_blind"), alpha=HOLM_ALPHA,
    )
    mode_shuffle = {
        mode: float(np.mean(shuffle_advantage[modes == mode])) for mode in MODE_LABELS
    }
    mode_state = {
        mode: float(np.mean(state_advantage[modes == mode])) for mode in MODE_LABELS
    }
    passed = bool(
        np.mean(shuffle_advantage) >= MIN_ACTION_SHUFFLE_ADVANTAGE
        and shuffle_interval[0] > 0
        and all(value > 0 for value in mode_shuffle.values())
        and np.mean(state_advantage) > MIN_ACTION_BLIND_ADVANTAGE
        and state_interval[0] > 0
        and all(value > 0 for value in mode_state.values())
    )
    MODEL_GATES[short] = passed
    ACTION_SPECIFICITY_SUMMARY[short] = {
        "records": len(primary_errors),
        "response_rows_per_record": expected_rows_per_record,
        "mean_primary_path_mse": float(np.mean(primary_errors)),
        "mean_same_length_derangement_mse": float(np.mean(shuffled_errors)),
        "mean_action_blind_state_mse": float(np.mean(static_errors)),
        "mean_derangement_advantage": float(np.mean(shuffle_advantage)),
        "derangement_advantage_ci95": shuffle_interval,
        "mean_action_blind_advantage": float(np.mean(state_advantage)),
        "action_blind_advantage_ci95": state_interval,
        "mode_mean_derangement_advantage": mode_shuffle,
        "mode_mean_action_blind_advantage": mode_state,
        "passed": passed,
    }

write_csv(EVIDENCE_DIR / "locked_action_specificity_repair_rows.csv", ACTION_SPECIFICITY_ROWS)
write_json(EVIDENCE_DIR / "action_specificity_repair_summary.json", {
    "by_model": jsonable(ACTION_SPECIFICITY_SUMMARY),
    "model_gates": MODEL_GATES,
    "thresholds": {
        "minimum_derangement_advantage": MIN_ACTION_SHUFFLE_ADVANTAGE,
        "minimum_action_blind_advantage": MIN_ACTION_BLIND_ADVANTAGE,
        "confidence_lower_endpoints_must_exceed_zero": True,
        "every_contact_mode_must_be_positive": True,
    },
})
print(json.dumps(ACTION_SPECIFICITY_SUMMARY, indent=2))
'''


decision_and_plot = r'''
# Apply the bounded repair decision and produce an auditable summary plot.
DECISION = derive_stage341_decision(
    Stage341Gates(
        upstream_binding=UPSTREAM_BINDING_GATE,
        leakage_invariant=LEAKAGE_INVARIANT_GATE,
        physical_action_necessity=PHYSICAL_ACTION_NECESSITY_GATE,
        jepa_action_specificity=MODEL_GATES.get("jepa", False),
        dino_action_specificity=MODEL_GATES.get("dino", False),
    ),
    run_mode=RUN_MODE,
)
DECISION.update({
    "protocol_id": PROTOCOL_ID,
    "protocol_sha256": NOTEBOOK_PROTOCOL_SHA256,
    "run_signature": RUN_SIGNATURE,
    "source_commit": SOURCE_COMMIT,
    "upstream_run_signature": UPSTREAM_RUN_SIGNATURE,
    "post_outcome_diagnostic": True,
    "confirmation_eligible": False,
    "action_specificity_summary": ACTION_SPECIFICITY_SUMMARY,
    "physical_control_summary": PHYSICAL_CONTROL_SUMMARY,
})
write_json(OUT / "stage34_1_decision.json", jsonable(DECISION))

labels = ["physical\naction", "JEPA\nderanged", "JEPA\nstate-only", "DINO\nderanged", "DINO\nstate-only"]
values = [
    PHYSICAL_CONTROL_SUMMARY["mean_action_necessity_advantage"],
    ACTION_SPECIFICITY_SUMMARY["jepa"]["mean_derangement_advantage"],
    ACTION_SPECIFICITY_SUMMARY["jepa"]["mean_action_blind_advantage"],
    ACTION_SPECIFICITY_SUMMARY["dino"]["mean_derangement_advantage"],
    ACTION_SPECIFICITY_SUMMARY["dino"]["mean_action_blind_advantage"],
]
intervals = [
    PHYSICAL_CONTROL_SUMMARY["action_necessity_ci95"],
    ACTION_SPECIFICITY_SUMMARY["jepa"]["derangement_advantage_ci95"],
    ACTION_SPECIFICITY_SUMMARY["jepa"]["action_blind_advantage_ci95"],
    ACTION_SPECIFICITY_SUMMARY["dino"]["derangement_advantage_ci95"],
    ACTION_SPECIFICITY_SUMMARY["dino"]["action_blind_advantage_ci95"],
]
lower = [value - interval[0] for value, interval in zip(values, intervals)]
upper = [interval[1] - value for value, interval in zip(values, intervals)]
figure, axis = plt.subplots(figsize=(10, 5.5))
colors = ["#64748b", "#2563eb", "#60a5fa", "#ea580c", "#fb923c"]
axis.bar(range(len(values)), values, color=colors, yerr=[lower, upper], capsize=5)
axis.axhline(0.0, color="black", linewidth=1)
axis.axhline(MIN_ACTION_SHUFFLE_ADVANTAGE, color="#475569", linestyle="--", linewidth=1)
axis.set_xticks(range(len(labels)), labels)
axis.set_ylabel("relative MSE advantage")
axis.set_title(f"Stage 34.1 leakage-free action specificity: {DECISION['status']}")
axis.grid(axis="y", alpha=0.2)
figure.tight_layout()
figure.savefig(PLOT_DIR / "stage34_1_action_specificity_repair.png", dpi=180)
plt.show()

if DECISION["passed"]:
    interpretation = (
        "The leakage-free diagnostic establishes action-specific prediction for both "
        "frozen checkpoints on the reused Stage 34 panel. The original static-only label "
        "should not be interpreted literally. Continue to the still-unobserved sufficiency "
        "and causal gates, then require a fresh confirmation before a publication claim."
    )
else:
    interpretation = (
        "The repaired diagnostic did not establish action specificity for both checkpoints. "
        "Do not continue to causal-abstraction claims from this panel; inspect the first failed "
        "repair gate in stage34_1_decision.json."
    )
automatic = f"""# Automatic Stage 34.1 interpretation

Status: **{DECISION['status'].upper()}**

{interpretation}

This is a post-outcome diagnostic repair and is not confirmation evidence. It
does not evaluate predictive sufficiency, internal causal use, two-sided
commutativity, or planning value.
"""
(OUT / "AUTOMATIC_INTERPRETATION.md").write_text(automatic)
(OUT / "FAILURE_TRACE.txt").write_text("NONE\n")
print(json.dumps({"status": DECISION["status"], "passed": DECISION["passed"]}, indent=2))
'''


package = r'''
# Package compact evidence; the 477 MB frozen Stage 34 input directory remains in Drive.
source_identity = {
    "repository": EXPERIMENT_REPOSITORY,
    "source_ref": EXPERIMENT_SOURCE_REF,
    "commit": SOURCE_COMMIT,
    "protocol_sha256": NOTEBOOK_PROTOCOL_SHA256,
    "files": {},
}
for relative in [EXPERIMENT_NOTEBOOK_PATH, EXPERIMENT_BUILDER_PATH, EXPERIMENT_NUMERICAL_PATH]:
    path = REPOSITORY / relative
    if not path.is_file():
        raise FileNotFoundError(f"missing committed Stage 34.1 source file: {relative}")
    source_identity["files"][relative] = sha256_file(path)
write_json(OUT / "source_identity.json", source_identity)
write_json(OUT / "timings.json", {
    "elapsed_seconds": time.time() - RUN_STARTED_AT,
    "gpu_required": False,
    "model_forwards": 0,
    "upstream_shards_reused": len(consumed),
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


archive_base = OUT / f"stage34_1_action_specificity_result_bundle_{RUN_SIGNATURE[:12]}"
archive_path = Path(str(archive_base) + ".zip")
manifest_path = OUT / "result_zip_manifest.json"
if archive_path.exists():
    archive_path.unlink()
rows = manifest_rows(OUT, excluded=[archive_path, manifest_path])
write_json(manifest_path, rows)
with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path != archive_path:
            archive.write(path, arcname=str(path.relative_to(OUT)))
print(json.dumps({
    "bundle": str(archive_path),
    "bundle_bytes": archive_path.stat().st_size,
    "files": len(rows) + 1,
    "status": DECISION["status"],
    "full_stage34_input_retained": str(UPSTREAM),
}, indent=2))
if DOWNLOAD_RESULTS:
    from google.colab import files
    files.download(str(archive_path))
'''


cells = [
    markdown(introduction, "stage341-00"),
    code(configuration, "stage341-01"),
    code(environment, "stage341-02"),
    code(source_binding, "stage341-03"),
    code(row_materialization, "stage341-04"),
    code(baseline_fit, "stage341-05"),
    code(locked_evaluation, "stage341-06"),
    code(decision_and_plot, "stage341-07"),
    code(package, "stage341-08"),
]

protocol_sources = [cell["source"].strip() for cell in cells]
protocol_digest = hashlib.sha256(
    json.dumps(protocol_sources, ensure_ascii=False, separators=(",", ":")).encode()
).hexdigest()
cells[1]["source"] = cells[1]["source"].replace("__PROTOCOL_DIGEST__", protocol_digest)

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {
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
TARGET.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(f"wrote {TARGET} ({len(cells)} cells, protocol {protocol_digest})")
