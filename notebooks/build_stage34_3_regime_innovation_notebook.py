"""Build the deterministic Stage 34.3 regime/innovation diagnostic notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "34_3_regime_innovation_diagnostic.ipynb"


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
# Stage 34.3: regime-aware JEPA innovation diagnostic

## Decision before computation

Stage 34.2 completed without an execution failure and rejected both proposed
repairs. DINO's 22-parameter calibration recovered only 2.2% of error and
remained far worse than the action-blind physical state prior. JEPA's
rank-five response chart was action specific but not predictively sufficient:
carrier residuals helped strongly in free and pre-contact states, while
deleting the fifth response coordinate improved rather than harmed prediction.

Stage 34.3 therefore pauses DINO and asks one narrow CPU-only question: can a
small, pre-evaluation JEPA state repair explain the failure?

The frozen family crosses:

- four versus five registered response coordinates;
- one universal transition map versus four physical-mode maps; and
- zero through three supervised carrier-innovation coordinates.

Every innovation direction is learned only from transition residuals in the
training fold. Rank, innovation dimension, mode structure, random-feature
ridge, and the final candidate are chosen by trajectory-grouped out-of-fold
error on the model-selection split. The chosen representation is refit once on
calibration and evaluated once on the already-open length-5--8 panel.

## Falsification controls

The candidate must improve over the registered rank-five universal baseline,
leave no material predictive gain for the full 64-dimensional carrier sketch,
and lose performance when *any* retained state coordinate is deleted. If it
uses physical-mode experts, it must also beat equal-capacity experts receiving
within-trajectory permuted mode identities.

## Claim boundary

This is a post-outcome diagnostic on reused evaluation trajectories. A pass
would identify a bounded observational JEPA state candidate worth testing on
fresh trajectories. It would not establish recursive closure, native causal
use, planning value, confirmation, or a shared JEPA--DINO mechanism.
'''


configuration = r'''
# Frozen Stage 34.3 contract. Pilot is diagnostic evidence; smoke is plumbing only.
RUN_MODE = "pilot"
MOUNT_DRIVE = True
DOWNLOAD_RESULTS = True

PROTOCOL_ID = "stage34.3-regime-aware-jepa-innovation-diagnostic-v1"
NOTEBOOK_PROTOCOL_SHA256 = "__PROTOCOL_DIGEST__"
EVIDENCE_STATUS = "POST_OUTCOME_CPU_DIAGNOSTIC_NOT_CONFIRMATION"

EXPERIMENT_REPOSITORY = "grewalsk/counterfactual-faithfulness-research"
EXPERIMENT_SOURCE_REF = "codex/stage34-predictive-fiber-abstraction"
EXPERIMENT_NOTEBOOK_PATH = "notebooks/34_3_regime_innovation_diagnostic.ipynb"
EXPERIMENT_BUILDER_PATH = "notebooks/build_stage34_3_regime_innovation_notebook.py"
EXPERIMENT_NUMERICAL_PATH = "src/cf_faithfulness/stage34_3_regime_innovation.py"

UPSTREAM_STAGE34_PROTOCOL_ID = "stage34-predictive-fiber-causal-abstraction-v1"
UPSTREAM_STAGE34_RUN_SIGNATURE = "d3f4f88426afff4d964bb4f1f1556c94ec3613b667edd9403ddfcd0fd78ded84"
UPSTREAM_STAGE34_SOURCE_COMMIT = "db130a3d25505b7fa69efbcd88009365cb266688"
UPSTREAM_STAGE34_RAW_MANIFEST_SHA256 = "2d2cf86fdeae5cb1034535104782dc526b8203b2e567e081ed530dbd288cb47e"
UPSTREAM_STAGE342_PROTOCOL_ID = "stage34.2-split-path-predictive-causal-continuation-v1"
UPSTREAM_STAGE342_RUN_SIGNATURE = "9fcedcf036a83c2ed234a39c62b4f4a7c2535dca7b0ba56cb0dd4c848c89ddb6"
UPSTREAM_STAGE342_SOURCE_COMMIT = "e46a1bbe9b861ebc27d201d5d33142bb26009d59"
UPSTREAM_STAGE342_MANIFEST_SHA256 = "8c203828867eb88b19dd6eb8af578f84e8a463dceb69ee8b469203d9d11de3b7"
UPSTREAM_STAGE342_DECISION_SHA256 = "514231d1faba4ec113e0e7ff865ddb89fa721f95cda8a662819959d660d9e792"
UPSTREAM_STAGE342_SUFFICIENCY_SHA256 = "d65a8dbc3a6dd2adaca6ffa30454fddd9bc81264ef4ddf5208c960c1b2d2daeb"

DRIVE_STAGE34_ROOT = "/content/drive/MyDrive/counterfactual_faithfulness_stage34_pfca"
DRIVE_STAGE342_ROOT = "/content/drive/MyDrive/counterfactual_faithfulness_stage34_2_split_path"
DRIVE_OUTPUT_ROOT = "/content/drive/MyDrive/counterfactual_faithfulness_stage34_3_regime_innovation"

SEED = 343101
SELECTION_SEED = 343137
CALIBRATION_SEED = 343179
BOOTSTRAP_SEED = 343211
CONTROL_SEED = 343253
BOOTSTRAP_DRAWS = 5000 if RUN_MODE == "pilot" else 100
FOLDS = 4 if RUN_MODE == "pilot" else 2
RFF_WIDTH = 128 if RUN_MODE == "pilot" else 32

CANDIDATE_STATE_RANKS = [4, 5]
CANDIDATE_INNOVATION_RANKS = [0, 1, 2, 3] if RUN_MODE == "pilot" else [0, 1]
CANDIDATE_REGIMES = ["universal", "physical_mode"]
OPERATOR_RIDGES = [1e-4, 1e-3, 1e-2, 1e-1, 1.0] if RUN_MODE == "pilot" else [1e-3, 1e-1]
SELECTION_RELATIVE_TOLERANCE = 0.02

MIN_SELECTION_IMPROVEMENT = 0.05
MIN_EVALUATION_IMPROVEMENT = 0.05
MAX_EXTRA_RESIDUAL_IMPROVEMENT = 0.05
MAX_EXTRA_RESIDUAL_CI_UPPER = 0.10
MIN_COORDINATE_NECESSITY = 0.02
MIN_MODE_CONTROL_ADVANTAGE = 0.05
HOLM_ALPHA = 0.05

MODE_LABELS = ["free", "pre_contact", "contact", "post_contact"]
CANONICAL_RANK = 5
STATE_CARRIER_SKETCH_DIM = 64
MODEL_SELECTION_RECORDS = 64
CALIBRATION_RECORDS = 64
EVALUATION_RECORDS = 128
CORE_WORD_COUNT = 15
EVALUATION_WORD_COUNT = 8

assert RUN_MODE in {"pilot", "smoke"}
assert CANDIDATE_STATE_RANKS == [4, 5]
assert max(CANDIDATE_INNOVATION_RANKS) <= 3
assert CANONICAL_RANK == 5
'''


setup = r'''
# Mount Drive, resolve committed source, and initialize one deterministic output directory.
import csv
import hashlib
import importlib
import json
import os
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

SOURCE_REPOSITORY = Path("/content/counterfactual-faithfulness-stage343")
REMOTE = f"https://github.com/{EXPERIMENT_REPOSITORY}.git"
if (SOURCE_REPOSITORY / ".git").is_dir():
    subprocess.run(
        ["git", "-C", str(SOURCE_REPOSITORY), "fetch", "origin", EXPERIMENT_SOURCE_REF],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(SOURCE_REPOSITORY), "checkout", "--detach", "FETCH_HEAD"],
        check=True,
    )
else:
    subprocess.run([
        "git", "clone", "--depth", "1", "--branch", EXPERIMENT_SOURCE_REF,
        REMOTE, str(SOURCE_REPOSITORY),
    ], check=True)
SOURCE_COMMIT = subprocess.check_output(
    ["git", "-C", str(SOURCE_REPOSITORY), "rev-parse", "HEAD"], text=True
).strip()
sys.path.insert(0, str(SOURCE_REPOSITORY / "src"))

stage343_module = importlib.import_module("cf_faithfulness.stage34_3_regime_innovation")
Stage343Gates = stage343_module.Stage343Gates
aggregate_relative_gain = stage343_module.aggregate_relative_gain
candidate_state_features = stage343_module.candidate_state_features
clustered_relative_gain_interval = stage343_module.clustered_relative_gain_interval
derive_stage343_decision = stage343_module.derive_stage343_decision
fit_candidate_model = stage343_module.fit_candidate_model
fit_regime_dynamics = stage343_module.fit_regime_dynamics
grouped_candidate_oof = stage343_module.grouped_candidate_oof
predict_candidate_model = stage343_module.predict_candidate_model
predict_regime_dynamics = stage343_module.predict_regime_dynamics
select_simplest_candidate = stage343_module.select_simplest_candidate
stable_seed = stage343_module.stable_seed
within_group_permuted_labels = stage343_module.within_group_permuted_labels


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [jsonable(row) for row in rows]
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


CONFIG_NAMES = [
    "RUN_MODE", "PROTOCOL_ID", "NOTEBOOK_PROTOCOL_SHA256", "EVIDENCE_STATUS",
    "UPSTREAM_STAGE34_RUN_SIGNATURE", "UPSTREAM_STAGE34_RAW_MANIFEST_SHA256",
    "UPSTREAM_STAGE342_RUN_SIGNATURE", "UPSTREAM_STAGE342_MANIFEST_SHA256",
    "UPSTREAM_STAGE342_DECISION_SHA256", "CANDIDATE_STATE_RANKS",
    "CANDIDATE_INNOVATION_RANKS", "CANDIDATE_REGIMES", "OPERATOR_RIDGES",
    "SELECTION_RELATIVE_TOLERANCE", "MIN_SELECTION_IMPROVEMENT",
    "MIN_EVALUATION_IMPROVEMENT", "MAX_EXTRA_RESIDUAL_IMPROVEMENT",
    "MAX_EXTRA_RESIDUAL_CI_UPPER", "MIN_COORDINATE_NECESSITY",
    "MIN_MODE_CONTROL_ADVANTAGE", "RFF_WIDTH", "FOLDS", "BOOTSTRAP_DRAWS",
    "SEED", "SELECTION_SEED", "CALIBRATION_SEED", "BOOTSTRAP_SEED", "CONTROL_SEED",
]
CONFIG = {name: globals()[name] for name in CONFIG_NAMES}
RUN_SIGNATURE = hashlib.sha256(json.dumps(
    {"config": CONFIG, "source_commit": SOURCE_COMMIT}, sort_keys=True,
    separators=(",", ":"),
).encode()).hexdigest()
OUT = Path(DRIVE_OUTPUT_ROOT) / f"{RUN_MODE}_{RUN_SIGNATURE[:12]}"
EVIDENCE_DIR = OUT / "evaluation_evidence"
PLOT_DIR = OUT / "plots"
for directory in [OUT, EVIDENCE_DIR, PLOT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
RUN_STARTED_AT = time.time()
(OUT / "FAILURE_TRACE.txt").write_text("PENDING\n")
write_json(OUT / "config.json", {
    **CONFIG, "run_signature": RUN_SIGNATURE, "source_commit": SOURCE_COMMIT,
})
print(json.dumps({
    "protocol": PROTOCOL_ID, "run_signature": RUN_SIGNATURE,
    "source_commit": SOURCE_COMMIT, "output": str(OUT),
    "gpu_required": False,
}, indent=2))
'''


binding = r'''
# Hash-bind the exact Stage 34 raw run and the exact Stage 34.2 stopped decision.
STAGE34_ROOT = Path(DRIVE_STAGE34_ROOT) / f"pilot_{UPSTREAM_STAGE34_RUN_SIGNATURE[:12]}"
STAGE342_ROOT = Path(DRIVE_STAGE342_ROOT) / f"pilot_{UPSTREAM_STAGE342_RUN_SIGNATURE[:12]}"
if not STAGE34_ROOT.is_dir():
    raise FileNotFoundError(f"missing complete Stage 34 Drive directory: {STAGE34_ROOT}")
if not STAGE342_ROOT.is_dir():
    raise FileNotFoundError(f"missing complete Stage 34.2 Drive directory: {STAGE342_ROOT}")

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


stage342_manifest_path = STAGE342_ROOT / "result_zip_manifest.json"
if sha256_file(stage342_manifest_path) != UPSTREAM_STAGE342_MANIFEST_SHA256:
    raise RuntimeError("Stage 34.2 result manifest does not match the audited bundle")
stage342_manifest = json.loads(stage342_manifest_path.read_text())
stage342_by_path = {row["path"]: row for row in stage342_manifest}


def verify_stage342(relative):
    relative = str(relative)
    if relative not in stage342_by_path:
        raise RuntimeError(f"unmanifested Stage 34.2 input: {relative}")
    path = STAGE342_ROOT / relative
    row = stage342_by_path[relative]
    if not path.is_file() or path.stat().st_size != int(row["bytes"]):
        raise RuntimeError(f"missing or wrong-sized Stage 34.2 input: {relative}")
    if sha256_file(path) != row["sha256"]:
        raise RuntimeError(f"Stage 34.2 hash mismatch: {relative}")
    return path


for relative in stage342_by_path:
    verify_stage342(relative)
if sha256_file(STAGE342_ROOT / "stage34_2_decision.json") != UPSTREAM_STAGE342_DECISION_SHA256:
    raise RuntimeError("Stage 34.2 decision content changed")
if sha256_file(
    STAGE342_ROOT / "evaluation_evidence/jepa_predictive_sufficiency_summary.json"
) != UPSTREAM_STAGE342_SUFFICIENCY_SHA256:
    raise RuntimeError("Stage 34.2 JEPA sufficiency result changed")

stage342_decision = json.loads((STAGE342_ROOT / "stage34_2_decision.json").read_text())
stage342_source = json.loads((STAGE342_ROOT / "source_identity.json").read_text())
if stage342_decision.get("protocol_id") != UPSTREAM_STAGE342_PROTOCOL_ID:
    raise RuntimeError("wrong Stage 34.2 protocol")
if stage342_decision.get("run_signature") != UPSTREAM_STAGE342_RUN_SIGNATURE:
    raise RuntimeError("wrong Stage 34.2 run signature")
if stage342_source.get("commit") != UPSTREAM_STAGE342_SOURCE_COMMIT:
    raise RuntimeError("wrong Stage 34.2 source commit")
if stage342_decision.get("status") != "jepa_response_state_insufficient":
    raise RuntimeError("Stage 34.2 does not have the registered JEPA insufficiency outcome")
if stage342_decision.get("checks", {}).get("jepa_predictive_sufficiency", True):
    raise RuntimeError("Stage 34.2 JEPA sufficiency failure is not bound")
if stage342_decision.get("checks", {}).get("jepa_causal_evaluated", True):
    raise RuntimeError("Stage 34.2 unexpectedly evaluated native causality")
if (STAGE342_ROOT / "FAILURE_TRACE.txt").read_text().strip() != "NONE":
    raise RuntimeError("Stage 34.2 contains an execution failure")

for relative in [
    "config.json", "source_identity.json", "run_provenance_certificate.json",
    "FAILURE_TRACE.txt", "physical_response_chart/rank_lock.json",
    "design/selected_model_selection_trajectories.json",
    "design/selected_calibration_trajectories.json",
    "design/selected_evaluation_trajectories.json",
]:
    verify_stage34(relative)
stage34_config = json.loads((STAGE34_ROOT / "config.json").read_text())
stage34_source = json.loads((STAGE34_ROOT / "source_identity.json").read_text())
stage34_provenance = json.loads((STAGE34_ROOT / "run_provenance_certificate.json").read_text())
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
expected_counts = {
    "model_selection": MODEL_SELECTION_RECORDS,
    "calibration": CALIBRATION_RECORDS,
    "evaluation": EVALUATION_RECORDS,
}
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
for split, records in SELECTED_RECORDS.items():
    for record in records:
        consumed_stage34.add(
            f"baseline_shards/transitions_jepa_{split}_{int(record['record_id'])}.npz"
        )
for relative in sorted(consumed_stage34):
    verify_stage34(relative)
rank_lock = json.loads((STAGE34_ROOT / "physical_response_chart/rank_lock.json").read_text())
if int(rank_lock.get("diagnostic_rank", -1)) != CANONICAL_RANK:
    raise RuntimeError("Stage 34 canonical rank changed")

UPSTREAM_BINDING_GATE = True
STAGE342_BINDING_GATE = True
write_json(OUT / "upstream_binding_certificate.json", {
    "stage34_run_signature": UPSTREAM_STAGE34_RUN_SIGNATURE,
    "stage342_run_signature": UPSTREAM_STAGE342_RUN_SIGNATURE,
    "stage34_transition_shards": len(consumed_stage34),
    "stage342_verified_files": len(stage342_by_path),
    "trajectory_split_counts": {key: len(value) // 4 for key, value in SELECTED_RECORDS.items()},
    "trajectory_disjoint": True,
    "canonical_rank": CANONICAL_RANK,
    "stage342_status": stage342_decision["status"],
    "stage342_causal_gate_skipped": True,
    "binding_passed": True,
})
print(f"Verified {len(consumed_stage34)} transition shards and {len(stage342_by_path)} Stage 34.2 files")
'''


data_loading = r'''
# Materialize the frozen JEPA transition table without model loading or simulator reruns.
def transition_path(record, split):
    return STAGE34_ROOT / (
        f"baseline_shards/transitions_jepa_{split}_{int(record['record_id'])}.npz"
    )


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


DATA = {
    split: load_transition_rows(split)
    for split in ["model_selection", "calibration", "evaluation"]
}
expected_rows = {
    "model_selection": MODEL_SELECTION_RECORDS * CORE_WORD_COUNT,
    "calibration": CALIBRATION_RECORDS * CORE_WORD_COUNT,
    "evaluation": EVALUATION_RECORDS * EVALUATION_WORD_COUNT,
}
for split, rows in DATA.items():
    if len(rows["state"]) != expected_rows[split]:
        raise RuntimeError(f"{split} transition row count changed")
    if rows["state"].shape[1] != CANONICAL_RANK:
        raise RuntimeError(f"{split} response-state width changed")
    if rows["target"].shape[1] != CANONICAL_RANK:
        raise RuntimeError(f"{split} target-state width changed")
    if rows["residual"].shape[1] != STATE_CARRIER_SKETCH_DIM:
        raise RuntimeError(f"{split} carrier-sketch width changed")
    for key in ["state", "action", "target", "residual"]:
        if not np.all(np.isfinite(rows[key])):
            raise RuntimeError(f"{split} {key} contains nonfinite values")
print(json.dumps({
    split: {
        "rows": len(rows["state"]),
        "trajectories": len(np.unique(rows["group"])),
        "records": len(np.unique(rows["record_id"])),
        "words": len(np.unique(rows["word"])),
    }
    for split, rows in DATA.items()
}, indent=2))
'''


selection = r'''
# Select rank, innovation width, mode structure, and ridge before evaluation is touched.
SELECTION = DATA["model_selection"]
CANDIDATE_ROWS = []
for state_rank in CANDIDATE_STATE_RANKS:
    for innovation_rank in CANDIDATE_INNOVATION_RANKS:
        for regime_name in CANDIDATE_REGIMES:
            regime_specific = regime_name == "physical_mode"
            candidate_seed = stable_seed(
                SELECTION_SEED, state_rank, innovation_rank, regime_name
            )
            for penalty in OPERATOR_RIDGES:
                result = grouped_candidate_oof(
                    SELECTION["state"], SELECTION["action"], SELECTION["residual"],
                    SELECTION["target"], SELECTION["group"], SELECTION["mode"],
                    state_rank=state_rank, innovation_rank=innovation_rank,
                    regime_specific=regime_specific, mode_labels=MODE_LABELS,
                    width=RFF_WIDTH, penalty=penalty, folds=FOLDS,
                    seed=candidate_seed,
                )
                CANDIDATE_ROWS.append({
                    "state_rank": state_rank,
                    "innovation_rank": innovation_rank,
                    "regime": regime_name,
                    "regime_specific": regime_specific,
                    "penalty": penalty,
                    "oof_mse": result["oof_mse"],
                    "state_coordinate_count": state_rank + innovation_rank,
                    "fold_rows": json.dumps(result["fold_rows"]),
                })
                print(
                    f"candidate q{state_rank}+u{innovation_rank} {regime_name} "
                    f"ridge={penalty:g} oof_mse={result['oof_mse']:.6g}"
                )

SELECTED_CANDIDATE = select_simplest_candidate(
    CANDIDATE_ROWS, relative_tolerance=SELECTION_RELATIVE_TOLERANCE
)
BASELINE_CANDIDATE = min(
    (
        row for row in CANDIDATE_ROWS
        if row["state_rank"] == 5
        and row["innovation_rank"] == 0
        and not row["regime_specific"]
    ),
    key=lambda row: row["oof_mse"],
)
SELECTION_IMPROVEMENT = float(
    (BASELINE_CANDIDATE["oof_mse"] - SELECTED_CANDIDATE["oof_mse"])
    / max(BASELINE_CANDIDATE["oof_mse"], 1e-12)
)
SELECTION_IMPROVEMENT_GATE = bool(
    SELECTION_IMPROVEMENT >= MIN_SELECTION_IMPROVEMENT
)
SELECTION_MANIFEST = {
    "selected_candidate": SELECTED_CANDIDATE,
    "registered_baseline": BASELINE_CANDIDATE,
    "selection_improvement": SELECTION_IMPROVEMENT,
    "selection_improvement_passed": SELECTION_IMPROVEMENT_GATE,
    "selection_rows": len(SELECTION["state"]),
    "selection_trajectories": len(np.unique(SELECTION["group"])),
    "candidate_count": len(CANDIDATE_ROWS),
    "selection_uses_evaluation_rows": False,
}
write_csv(EVIDENCE_DIR / "candidate_selection_rows.csv", CANDIDATE_ROWS)
write_json(OUT / "frozen_candidate_selection.json", SELECTION_MANIFEST)
(OUT / "frozen_candidate_selection.json.sha256").write_text(
    sha256_file(OUT / "frozen_candidate_selection.json") + "\n"
)
print(json.dumps(SELECTION_MANIFEST, indent=2))
'''


evaluation = r'''
# Refit the frozen candidate on calibration and evaluate once on long, unseen words.
CALIBRATION = DATA["calibration"]
EVALUATION = DATA["evaluation"]
selected_regime_specific = bool(SELECTED_CANDIDATE["regime_specific"])
SELECTED_FIT_SEED = stable_seed(CALIBRATION_SEED, "selected")
SELECTED_MODEL = fit_candidate_model(
    CALIBRATION["state"], CALIBRATION["action"], CALIBRATION["residual"],
    CALIBRATION["target"], CALIBRATION["mode"],
    state_rank=int(SELECTED_CANDIDATE["state_rank"]),
    innovation_rank=int(SELECTED_CANDIDATE["innovation_rank"]),
    regime_specific=selected_regime_specific, mode_labels=MODE_LABELS,
    width=RFF_WIDTH, penalty=float(SELECTED_CANDIDATE["penalty"]),
    seed=SELECTED_FIT_SEED,
)
selected_prediction = predict_candidate_model(
    SELECTED_MODEL, EVALUATION["state"], EVALUATION["action"],
    EVALUATION["residual"], EVALUATION["mode"],
)
selected_error = np.mean((selected_prediction - EVALUATION["target"]) ** 2, axis=1)

BASELINE_MODEL = fit_candidate_model(
    CALIBRATION["state"], CALIBRATION["action"], CALIBRATION["residual"],
    CALIBRATION["target"], CALIBRATION["mode"],
    state_rank=5, innovation_rank=0, regime_specific=False,
    mode_labels=MODE_LABELS, width=RFF_WIDTH,
    penalty=float(BASELINE_CANDIDATE["penalty"]),
    seed=stable_seed(CALIBRATION_SEED, "registered_baseline"),
)
baseline_prediction = predict_candidate_model(
    BASELINE_MODEL, EVALUATION["state"], EVALUATION["action"],
    EVALUATION["residual"], EVALUATION["mode"],
)
baseline_error = np.mean((baseline_prediction - EVALUATION["target"]) ** 2, axis=1)
EVALUATION_IMPROVEMENT = aggregate_relative_gain(selected_error, baseline_error)
EVALUATION_IMPROVEMENT_CI = clustered_relative_gain_interval(
    selected_error, baseline_error, EVALUATION["group"],
    draws=BOOTSTRAP_DRAWS,
    seed=stable_seed(BOOTSTRAP_SEED, "evaluation_improvement"),
    alpha=HOLM_ALPHA,
)
MODE_EVALUATION_IMPROVEMENT = {
    mode: aggregate_relative_gain(
        selected_error[EVALUATION["mode"] == mode],
        baseline_error[EVALUATION["mode"] == mode],
    )
    for mode in MODE_LABELS
}
EVALUATION_IMPROVEMENT_GATE = bool(
    EVALUATION_IMPROVEMENT >= MIN_EVALUATION_IMPROVEMENT
    and EVALUATION_IMPROVEMENT_CI[0] > 0
    and all(value > 0 for value in MODE_EVALUATION_IMPROVEMENT.values())
)
EVALUATION_ROWS_TABLE = [{
    "record_id": int(EVALUATION["record_id"][index]),
    "trajectory_id": int(EVALUATION["group"][index]),
    "mode": str(EVALUATION["mode"][index]),
    "word": str(EVALUATION["word"][index]),
    "word_length": int(EVALUATION["length"][index]),
    "selected_mse": float(selected_error[index]),
    "registered_baseline_mse": float(baseline_error[index]),
} for index in range(len(selected_error))]
write_csv(EVIDENCE_DIR / "locked_candidate_evaluation_rows.csv", EVALUATION_ROWS_TABLE)
print(json.dumps({
    "selected_evaluation_mse": float(np.mean(selected_error)),
    "baseline_evaluation_mse": float(np.mean(baseline_error)),
    "evaluation_improvement": EVALUATION_IMPROVEMENT,
    "evaluation_improvement_ci95": EVALUATION_IMPROVEMENT_CI,
    "mode_improvements": MODE_EVALUATION_IMPROVEMENT,
    "passed": EVALUATION_IMPROVEMENT_GATE,
}, indent=2))
'''


controls = r'''
# Test residual sufficiency, every-coordinate necessity, and matched mode specificity.
calibration_state = candidate_state_features(
    SELECTED_MODEL, CALIBRATION["state"], CALIBRATION["residual"]
)
evaluation_state = candidate_state_features(
    SELECTED_MODEL, EVALUATION["state"], EVALUATION["residual"]
)
carrier_mean = np.mean(CALIBRATION["residual"], axis=0)
carrier_scale = np.maximum(np.std(CALIBRATION["residual"], axis=0, ddof=1), 1e-8)
calibration_full_carrier = (CALIBRATION["residual"] - carrier_mean) / carrier_scale
evaluation_full_carrier = (EVALUATION["residual"] - carrier_mean) / carrier_scale
FULL_RESIDUAL_MODEL = fit_regime_dynamics(
    np.column_stack([
        calibration_state, CALIBRATION["action"], calibration_full_carrier
    ]),
    CALIBRATION["target"], CALIBRATION["mode"],
    regime_specific=selected_regime_specific, mode_labels=MODE_LABELS,
    width=RFF_WIDTH, penalty=float(SELECTED_CANDIDATE["penalty"]),
    seed=stable_seed(CALIBRATION_SEED, "full_residual_control"),
)
full_residual_prediction = predict_regime_dynamics(
    FULL_RESIDUAL_MODEL,
    np.column_stack([
        evaluation_state, EVALUATION["action"], evaluation_full_carrier
    ]),
    EVALUATION["mode"],
)
full_residual_error = np.mean(
    (full_residual_prediction - EVALUATION["target"]) ** 2, axis=1
)
EXTRA_RESIDUAL_IMPROVEMENT = aggregate_relative_gain(
    full_residual_error, selected_error
)
EXTRA_RESIDUAL_CI = clustered_relative_gain_interval(
    full_residual_error, selected_error, EVALUATION["group"],
    draws=BOOTSTRAP_DRAWS,
    seed=stable_seed(BOOTSTRAP_SEED, "extra_residual"),
    alpha=HOLM_ALPHA,
)
MODE_EXTRA_RESIDUAL_IMPROVEMENT = {
    mode: aggregate_relative_gain(
        full_residual_error[EVALUATION["mode"] == mode],
        selected_error[EVALUATION["mode"] == mode],
    )
    for mode in MODE_LABELS
}
RESIDUAL_SUFFICIENCY_GATE = bool(
    EXTRA_RESIDUAL_IMPROVEMENT <= MAX_EXTRA_RESIDUAL_IMPROVEMENT
    and EXTRA_RESIDUAL_CI[1] <= MAX_EXTRA_RESIDUAL_CI_UPPER
    and all(
        value <= MAX_EXTRA_RESIDUAL_CI_UPPER
        for value in MODE_EXTRA_RESIDUAL_IMPROVEMENT.values()
    )
)

coordinate_names = [
    *(f"q{index + 1}" for index in range(int(SELECTED_CANDIDATE["state_rank"]))),
    *(f"u{index + 1}" for index in range(int(SELECTED_CANDIDATE["innovation_rank"]))),
]
COORDINATE_ROWS = []
DELETION_ERRORS = {}
for coordinate_index, coordinate_name in enumerate(coordinate_names):
    calibration_deleted_state = calibration_state.copy()
    evaluation_deleted_state = evaluation_state.copy()
    replacement = float(np.mean(calibration_state[:, coordinate_index]))
    calibration_deleted_state[:, coordinate_index] = replacement
    evaluation_deleted_state[:, coordinate_index] = replacement
    deleted_model = fit_regime_dynamics(
        np.column_stack([calibration_deleted_state, CALIBRATION["action"]]),
        CALIBRATION["target"], CALIBRATION["mode"],
        regime_specific=selected_regime_specific, mode_labels=MODE_LABELS,
        width=RFF_WIDTH, penalty=float(SELECTED_CANDIDATE["penalty"]),
        seed=stable_seed(SELECTED_FIT_SEED, "candidate_dynamics"),
    )
    deleted_prediction = predict_regime_dynamics(
        deleted_model,
        np.column_stack([evaluation_deleted_state, EVALUATION["action"]]),
        EVALUATION["mode"],
    )
    deleted_error = np.mean((deleted_prediction - EVALUATION["target"]) ** 2, axis=1)
    DELETION_ERRORS[coordinate_name] = deleted_error
    necessity = aggregate_relative_gain(selected_error, deleted_error)
    interval = clustered_relative_gain_interval(
        selected_error, deleted_error, EVALUATION["group"],
        draws=BOOTSTRAP_DRAWS,
        seed=stable_seed(BOOTSTRAP_SEED, "delete", coordinate_name),
        alpha=HOLM_ALPHA,
    )
    mode_necessity = {
        mode: aggregate_relative_gain(
            selected_error[EVALUATION["mode"] == mode],
            deleted_error[EVALUATION["mode"] == mode],
        )
        for mode in MODE_LABELS
    }
    COORDINATE_ROWS.append({
        "coordinate": coordinate_name,
        "deleted_evaluation_mse": float(np.mean(deleted_error)),
        "necessity_gain": necessity,
        "ci95_lower": interval[0],
        "ci95_upper": interval[1],
        "mode_necessity": json.dumps(mode_necessity, sort_keys=True),
        "passed": bool(
            necessity >= MIN_COORDINATE_NECESSITY
            and interval[0] > 0
            and all(value > 0 for value in mode_necessity.values())
        ),
    })
COORDINATE_NECESSITY_GATE = bool(
    COORDINATE_ROWS and all(row["passed"] for row in COORDINATE_ROWS)
)

MODE_CONTROL_APPLICABLE = selected_regime_specific
if MODE_CONTROL_APPLICABLE:
    calibration_permuted_modes = within_group_permuted_labels(
        CALIBRATION["mode"], CALIBRATION["group"], CALIBRATION["record_id"],
        seed=stable_seed(CONTROL_SEED, "calibration_mode_permutation"),
    )
    evaluation_permuted_modes = within_group_permuted_labels(
        EVALUATION["mode"], EVALUATION["group"], EVALUATION["record_id"],
        seed=stable_seed(CONTROL_SEED, "evaluation_mode_permutation"),
    )
    permuted_mode_model = fit_regime_dynamics(
        np.column_stack([calibration_state, CALIBRATION["action"]]),
        CALIBRATION["target"], calibration_permuted_modes,
        regime_specific=True, mode_labels=MODE_LABELS, width=RFF_WIDTH,
        penalty=float(SELECTED_CANDIDATE["penalty"]),
        seed=stable_seed(SELECTED_FIT_SEED, "candidate_dynamics"),
    )
    permuted_mode_prediction = predict_regime_dynamics(
        permuted_mode_model,
        np.column_stack([evaluation_state, EVALUATION["action"]]),
        evaluation_permuted_modes,
    )
    permuted_mode_error = np.mean(
        (permuted_mode_prediction - EVALUATION["target"]) ** 2, axis=1
    )
    MODE_CONTROL_ADVANTAGE = aggregate_relative_gain(selected_error, permuted_mode_error)
    MODE_CONTROL_CI = clustered_relative_gain_interval(
        selected_error, permuted_mode_error, EVALUATION["group"],
        draws=BOOTSTRAP_DRAWS,
        seed=stable_seed(BOOTSTRAP_SEED, "permuted_mode_control"),
        alpha=HOLM_ALPHA,
    )
    MODE_SPECIFICITY_GATE = bool(
        MODE_CONTROL_ADVANTAGE >= MIN_MODE_CONTROL_ADVANTAGE
        and MODE_CONTROL_CI[0] > 0
    )
else:
    permuted_mode_error = np.full(len(selected_error), np.nan)
    MODE_CONTROL_ADVANTAGE = 0.0
    MODE_CONTROL_CI = [0.0, 0.0]
    MODE_SPECIFICITY_GATE = True

CONTROL_SUMMARY = {
    "selected_state_coordinates": coordinate_names,
    "extra_residual_improvement": EXTRA_RESIDUAL_IMPROVEMENT,
    "extra_residual_ci95": EXTRA_RESIDUAL_CI,
    "mode_extra_residual_improvement": MODE_EXTRA_RESIDUAL_IMPROVEMENT,
    "residual_sufficiency_passed": RESIDUAL_SUFFICIENCY_GATE,
    "coordinate_necessity": COORDINATE_ROWS,
    "coordinate_necessity_passed": COORDINATE_NECESSITY_GATE,
    "mode_control_applicable": MODE_CONTROL_APPLICABLE,
    "mode_control_advantage": MODE_CONTROL_ADVANTAGE,
    "mode_control_ci95": MODE_CONTROL_CI,
    "mode_specificity_passed": MODE_SPECIFICITY_GATE,
}
write_csv(EVIDENCE_DIR / "coordinate_deletion_summary.csv", COORDINATE_ROWS)
write_json(EVIDENCE_DIR / "falsification_control_summary.json", CONTROL_SUMMARY)
print(json.dumps(CONTROL_SUMMARY, indent=2))
'''


decision = r'''
# Derive the bounded diagnostic decision; no branch can claim causality or confirmation.
DECISION = derive_stage343_decision(
    Stage343Gates(
        upstream_binding=UPSTREAM_BINDING_GATE,
        stage342_binding=STAGE342_BINDING_GATE,
        selection_improvement=SELECTION_IMPROVEMENT_GATE,
        evaluation_improvement=EVALUATION_IMPROVEMENT_GATE,
        residual_sufficiency=RESIDUAL_SUFFICIENCY_GATE,
        coordinate_necessity=COORDINATE_NECESSITY_GATE,
        mode_specificity=MODE_SPECIFICITY_GATE,
    ),
    run_mode=RUN_MODE,
)
DECISION.update({
    "protocol_id": PROTOCOL_ID,
    "protocol_sha256": NOTEBOOK_PROTOCOL_SHA256,
    "run_signature": RUN_SIGNATURE,
    "source_commit": SOURCE_COMMIT,
    "upstream_stage34_run_signature": UPSTREAM_STAGE34_RUN_SIGNATURE,
    "upstream_stage342_run_signature": UPSTREAM_STAGE342_RUN_SIGNATURE,
    "selected_candidate": SELECTED_CANDIDATE,
    "registered_baseline": BASELINE_CANDIDATE,
    "selection_improvement": SELECTION_IMPROVEMENT,
    "evaluation_improvement": EVALUATION_IMPROVEMENT,
    "evaluation_improvement_ci95": EVALUATION_IMPROVEMENT_CI,
    "mode_evaluation_improvement": MODE_EVALUATION_IMPROVEMENT,
    "selected_evaluation_mse": float(np.mean(selected_error)),
    "baseline_evaluation_mse": float(np.mean(baseline_error)),
    "control_summary": CONTROL_SUMMARY,
    "dino_branch_paused": True,
    "native_checkpoint_loaded": False,
    "shared_abstraction_claimed": False,
    "recursive_closure_claimed": False,
    "confirmation_eligible": False,
})
write_json(OUT / "stage34_3_decision.json", DECISION)

figure, axes = plt.subplots(2, 2, figsize=(13, 9))
selection_plot = sorted(CANDIDATE_ROWS, key=lambda row: row["oof_mse"])[:12]
axes[0, 0].barh(
    [f"q{row['state_rank']}+u{row['innovation_rank']} {row['regime']}" for row in selection_plot][::-1],
    [row["oof_mse"] for row in selection_plot][::-1],
    color="#2563eb",
)
axes[0, 0].set_title("Best pre-evaluation candidate fits")
axes[0, 0].set_xlabel("grouped OOF MSE")
axes[0, 1].bar(
    ["baseline", "selected", "full residual"],
    [np.mean(baseline_error), np.mean(selected_error), np.mean(full_residual_error)],
    color=["#64748b", "#2563eb", "#7c3aed"],
)
axes[0, 1].set_title("Locked evaluation MSE")
axes[1, 0].bar(
    MODE_LABELS,
    [MODE_EVALUATION_IMPROVEMENT[mode] for mode in MODE_LABELS],
    color="#0ea5e9",
)
axes[1, 0].axhline(0, color="black", linewidth=1)
axes[1, 0].set_title("Selected gain over registered baseline")
axes[1, 0].tick_params(axis="x", rotation=20)
axes[1, 1].bar(
    [row["coordinate"] for row in COORDINATE_ROWS],
    [row["necessity_gain"] for row in COORDINATE_ROWS],
    color="#f97316",
)
axes[1, 1].axhline(MIN_COORDINATE_NECESSITY, color="black", linestyle="--")
axes[1, 1].set_title("Leave-one-coordinate-out necessity")
figure.suptitle(f"Stage 34.3: {DECISION['status']}")
figure.tight_layout()
figure.savefig(PLOT_DIR / "stage34_3_regime_innovation_summary.png", dpi=180)
plt.show()

interpretations = {
    "no_selected_regime_innovation_repair": (
        "No low-dimensional rank/mode/innovation candidate improved enough before evaluation."
    ),
    "selected_repair_did_not_transfer": (
        "The pre-evaluation candidate did not transfer across unseen long action words."
    ),
    "selected_state_still_carrier_incomplete": (
        "The candidate transferred, but the remaining carrier sketch still predicted material dynamics."
    ),
    "selected_state_not_minimal": (
        "The candidate remained redundant because at least one retained coordinate was unnecessary."
    ),
    "physical_mode_structure_not_specific": (
        "Physical mode experts did not beat equal-capacity permuted-mode experts."
    ),
    "bounded_jepa_state_candidate_repaired": (
        "A small observational JEPA state survived all registered diagnostic controls."
    ),
}
interpretation = interpretations.get(
    DECISION["status"], "The diagnostic was inconclusive before the scientific gates."
)
(OUT / "AUTOMATIC_INTERPRETATION.md").write_text(
    f"# Automatic Stage 34.3 interpretation\n\n"
    f"Status: **{DECISION['status'].upper()}**\n\n{interpretation}\n\n"
    "This reused-panel CPU diagnostic is neither causal nor confirmatory. A positive result "
    "requires fresh trajectories before native intervention. DINO remains paused.\n"
)
(OUT / "FAILURE_TRACE.txt").write_text("NONE\n")
print(json.dumps({"status": DECISION["status"], "passed": DECISION["passed"]}, indent=2))
'''


package = r'''
# Package compact evidence and immutable source identities.
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
        raise FileNotFoundError(f"missing committed Stage 34.3 source: {relative}")
    source_identity["files"][relative] = sha256_file(path)
write_json(OUT / "source_identity.json", source_identity)
write_json(OUT / "timings.json", {
    "elapsed_seconds": time.time() - RUN_STARTED_AT,
    "gpu_required": False,
    "native_model_forwards": 0,
    "candidate_fits": len(CANDIDATE_ROWS),
    "bootstrap_draws_per_gate": BOOTSTRAP_DRAWS,
    "stage34_transition_shards_reused": len(consumed_stage34),
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


archive_path = OUT / f"stage34_3_regime_innovation_result_bundle_{RUN_SIGNATURE[:12]}.zip"
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
    markdown(introduction, "stage343-00"),
    code(configuration, "stage343-01"),
    code(setup, "stage343-02"),
    code(binding, "stage343-03"),
    code(data_loading, "stage343-04"),
    code(selection, "stage343-05"),
    code(evaluation, "stage343-06"),
    code(controls, "stage343-07"),
    code(decision, "stage343-08"),
    code(package, "stage343-09"),
]

protocol_sources = [cell["source"].strip() for cell in cells]
protocol_digest = hashlib.sha256(
    json.dumps(protocol_sources, ensure_ascii=False, separators=(",", ":")).encode()
).hexdigest()
cells[1]["source"] = cells[1]["source"].replace("__PROTOCOL_DIGEST__", protocol_digest)

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {
            "display_name": "Python 3", "language": "python", "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
TARGET.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(f"wrote {TARGET} ({len(cells)} cells, protocol {protocol_digest})")
