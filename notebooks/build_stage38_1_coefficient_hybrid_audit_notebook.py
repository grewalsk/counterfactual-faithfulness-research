"""Build the source-bound Stage 38.1 development audit notebook."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
TARGET = ROOT / "38_1_coefficient_matched_hybrid_audit.ipynb"
NUMERICAL = REPOSITORY / "src/cf_faithfulness/stage38_1_coefficient_hybrid_audit.py"

spec = importlib.util.spec_from_file_location(
    "stage38_builder", ROOT / "build_stage38_cross_model_pscd_notebook.py"
)
STAGE38 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(STAGE38)

code = STAGE38.code
markdown = STAGE38.markdown
replace_assignment = STAGE38.replace_assignment
assigned_uppercase_names = STAGE38.assigned_uppercase_names
function_sources = STAGE38.function_sources


def rename(value: str) -> str:
    for old, new in [
        ("Stage 38", "Stage 38.1"), ("STAGE38", "STAGE381"),
        ("stage38", "stage381"),
    ]:
        value = value.replace(old, new)
    return value


introduction = r'''# Stage 38.1: coefficient-matched and hybrid closure audit

## V2 rendered-dataclass and cache-migration repair

The source-bound v1 run verified all 512 Stage 38 development shards and
completed all sixteen registered two-seed Tier A construction fits.  It then
stopped while constructing the first calibration decision because the notebook
builder copied the `TierAGates` and `TierBGates` class bodies without their
`@dataclass(frozen=True)` decorators.  No Tier A decision was emitted, the
precommitted third seed did not run, and Tier B remained sealed.  V2 restores
both decorators and makes the validator instantiate the rendered classes.

V2 may migrate only the sixteen v1 construction-only screening artifacts from
exact run `0b09871c37cc`.  Migration requires their array/schema hashes,
sidecars, protocol identity, objective coefficients, seeds, epochs, dimensions,
and finite parameters to match.  It reads no v1 calibration statistic or Stage
38 evaluation artifact.  A missing cache is refit normally; an invalid cache is
rejected rather than silently trusted.

## Decision before computation

Stage 38 established a valid negative/diagnostic result on frozen JEPA-WM and
DINO-WM carriers, but its full S-PSCD versus latent-only overshooting contrast
was not coefficient matched.  The full objective assigned 0.45 of its
semigroup weight to latent consistency while the control assigned 1.00.  At a
shared outer weight, the control therefore received 2.222... times the latent
pressure.  This development-only audit repairs that comparison before any
Stage 39, planning, Koopman replacement, or larger hybrid architecture.

Tier A refits four parameter- and initialization-matched models on construction
only: ordinary PSCD, the historical mass-matched overshooting sensitivity
control, coefficient-matched overshooting, and full S-PSCD.  The coefficient-
matched outer weights are fixed algebraically at 0.90 for JEPA and 0.45 for
DINO.  Two screening seeds run first.  The precommitted third seed runs only
when both representation panels remain eligible for promotion.  All outcomes
are scored on untouched calibration carrier shards.  Stage 38 locked closure
rows, evaluation shards, evaluation decisions, and planning artifacts are
forbidden inputs.

Only if Tier A promotes does Tier B fit a 32-hidden-unit event gate and jump
residual.  It first tests an oracle-event ceiling.  Only useful oracle headroom
opens the label-free, parameter-matched smooth, and shuffled-supervision
controls.  No tail loss is used until the structural intervention passes.  A
family-level CVaR90 training extension is then optional and separately gated.

## Claim boundary

This notebook is a development decision instrument, not confirmation.  A Tier
A pass isolates the value of grounded semigroup components at equal latent
pressure.  A Tier B pass supports an event/reset repair hypothesis on the
existing PushT development panels.  Neither result establishes native
checkpoint closure, cross-environment generality, planning value, a causal
mechanism inside JEPA/DINO, or an ICLR-ready claim.  Fresh Stage 39 data remain
mandatory after any complete pass.
'''


configuration = STAGE38.configuration
configuration = configuration.replace(
    "# SINGLE CONFIGURATION BLOCK — no Stage 38 secrets required.",
    "# SINGLE CONFIGURATION BLOCK — no Stage 38.1 secrets required.",
    1,
)
for name, value in {
    "PROTOCOL_ID": '"stage38.1-coefficient-matched-hybrid-audit-v2"',
    "NOTEBOOK_PROTOCOL_SHA256": '"__PROTOCOL_DIGEST__"',
    "EVIDENCE_STATUS": '"V2_RENDERED_DATACLASS_REPAIR_WITH_HASH_BOUND_TIER_A_MIGRATION"',
    "EXPERIMENT_NOTEBOOK_PATH": '"notebooks/38_1_coefficient_matched_hybrid_audit.ipynb"',
    "EXPERIMENT_BUILDER_PATH": '"notebooks/build_stage38_1_coefficient_hybrid_audit_notebook.py"',
    "EXPERIMENT_NUMERICAL_PATH": '"src/cf_faithfulness/stage38_1_coefficient_hybrid_audit.py"',
    "OUTPUT_DIR": '"/content/counterfactual_faithfulness_stage38_1_cmha"',
    "DRIVE_OUTPUT_DIR": '"/content/drive/MyDrive/counterfactual_faithfulness_stage38_1_cmha"',
    "RUN_REQUEST_PATH": '"/content/drive/MyDrive/counterfactual_faithfulness_stage38_1_cmha/stage381_run_request.json"',
    "MAX_ESTIMATED_TOTAL_MINUTES": "600.0",
    "SEED": "381001", "DESIGN_SEED": "381041", "DECODER_SEED": "381083",
    "RANK_SEED": "381113", "CALIBRATION_SEED": "381153",
    "BOOTSTRAP_SEED": "381183", "CONTROL_SEED": "381251",
}.items():
    configuration = replace_assignment(configuration, name, value)
configuration = re.sub(
    r"PINNED = \[.*?\]\n\nassert INTERVENTION_BLOCK",
    '''PINNED = [
    "exact_stage38_ceb85af5b4b9_development_shards_only",
    "exact_v1_0b09871c37cc_construction_model_migration_only",
    "construction_only_refits_and_calibration_only_scoring",
    "mass_matched_and_coefficient_matched_overshooting_controls",
    "two_seed_screen_and_precommitted_conditional_third_seed",
    "full_component_losses_coefficients_and_gradient_norms",
    "tier_b_sealed_until_both_tier_a_panels_pass",
    "oracle_ceiling_before_label_free_hybrid_controls",
    "capacity_matched_smooth_and_shuffled_supervision_controls",
    "family_risk_training_sealed_until_structural_pass",
    "no_stage38_evaluation_rows_or_planning_artifacts",
    "development_only_not_stage39_not_confirmation",
]

assert INTERVENTION_BLOCK''',
    configuration,
    count=1,
    flags=re.S,
)

# Remove the Stage 38 terminal constants and replace them with the exact
# development-only sequential design.  Earlier word declarations are retained
# solely to verify the source shards; no evaluation path is ever opened.
configuration = configuration.split("\n\nMAX_CARRIER_PROJECTION_DIM = 256", 1)[0]
configuration += r'''

SOURCE_STAGE38_RUN_DIR = "/content/drive/MyDrive/counterfactual_faithfulness_stage38_xmpscd/pilot_ceb85af5b4b9"
SOURCE_STAGE38_PROTOCOL_ID = "stage38-cross-model-pscd-confirmation-v1"
SOURCE_STAGE38_RUN_SIGNATURE = "ceb85af5b4b90ad3a3cecb3da5a7fa1d2eea597b13040a01f55ecc713227740a"
SOURCE_STAGE38_COMMIT = "a7ed07e2e79bc4da77e022f7765239b260bff35c"
LEGACY_STAGE381_RUN_DIR = "/content/drive/MyDrive/counterfactual_faithfulness_stage38_1_cmha/pilot_0b09871c37cc"
LEGACY_STAGE381_PROTOCOL_ID = "stage38.1-coefficient-matched-hybrid-audit-v1"
LEGACY_STAGE381_RUN_SIGNATURE = "0b09871c37ccd3487f58df8ccda13b8c468877af27f32b176894de513c321915"
LEGACY_STAGE381_SOURCE_COMMIT = "d570ce091cc4c22e7a76cb91fce7a782484ac616"
DEVELOPMENT_SPLITS = ["construction", "model_selection", "calibration"]
EVALUATION_ACCESS_PERMITTED = False
PLANNING_ACCESS_PERMITTED = False

FIXED_CARRIER_DIM = 256
FIXED_HISTORY_LENGTH = 4
FIXED_LATENT_DIM = 256
FIXED_DYNAMICS = "mixture"
SEMIGROUP_HORIZONS = [2, 4, 8]
FULL_COMPONENT_WEIGHTS = [0.35, 0.20, 0.45]
LATENT_ONLY_COMPONENT_WEIGHTS = [0.0, 0.0, 1.0]
FULL_OUTER_WEIGHTS = {"jepa": 2.0, "dino": 1.0}
COEFFICIENT_MATCHED_OUTER_WEIGHTS = {"jepa": 0.90, "dino": 0.45}
TIER_A_VARIANTS = ["pscd", "mass_overshoot", "coefficient_overshoot", "full"]
SCREENING_SEEDS = [38101, 38102]
PRECOMMITTED_THIRD_SEED = 38103
ALL_PRECOMMITTED_SEEDS = SCREENING_SEEDS + [PRECOMMITTED_THIRD_SEED]
TIER_B_VARIANTS = ["hybrid", "smooth", "shuffled"]
FINAL_EPOCHS = 320
ACTIVE_FINAL_EPOCHS = FINAL_EPOCHS if RUN_MODE == "pilot" else 4
PSCD_LEARNING_RATE = 1e-3
EVENT_HIDDEN = 32
EVENT_LOSS_WEIGHT = 0.10
RISK_ALPHA = 0.90
RISK_LOSS_WEIGHT = 0.10
ACTIVE_BOOTSTRAP_DRAWS = 4000 if RUN_MODE == "pilot" else 100

MIN_COEFFICIENT_GAIN = 0.05
MAX_TAIL_RATIO = 1.05
MIN_HISTORY_GAIN = 0.05
MAX_MEAN_PHYSICAL_NMSE = 0.25
MAX_P95_PHYSICAL_NMSE = 0.35
MAX_CATASTROPHIC_RATE = 0.02
MAX_LENGTH_PHYSICAL_NMSE = 0.35
MAX_MODE_PHYSICAL_NMSE = 0.40
MIN_EVENT_CONDITIONAL_GAIN = 0.25
MIN_OVERALL_P95_GAIN = 0.10
MAX_OVERALL_MEAN_DEGRADATION = 0.02
MIN_EVENT_AUROC = 0.80
MIN_BRIER_SKILL = 0.10
MAX_EVENT_ECE = 0.05
MIN_SHUFFLED_GAIN = 0.10
MIN_RISK_CVAR_GAIN = 0.15

assert FULL_OUTER_WEIGHTS["jepa"] * FULL_COMPONENT_WEIGHTS[2] == 0.90
assert FULL_OUTER_WEIGHTS["dino"] * FULL_COMPONENT_WEIGHTS[2] == 0.45
assert not EVALUATION_ACCESS_PERMITTED and not PLANNING_ACCESS_PERMITTED
assert len(SCREENING_SEEDS) == 2 and PRECOMMITTED_THIRD_SEED not in SCREENING_SEEDS
'''
configuration += "\n\nPROTOCOL_CONFIG_KEYS = " + repr(
    assigned_uppercase_names(configuration)
) + "\n"


installation = STAGE38.installation
setup = rename(STAGE38.setup).replace(
    "counterfactual_faithfulness_stage381_xmpscd",
    "counterfactual_faithfulness_stage38_1_cmha",
)

new_helpers = [
    "coefficient_matched_outer_weight", "macro_contact_events", "shuffled_event_labels",
    "_model_from_artifact", "_gradient_norm", "semigroup_component_diagnostics",
    "EventFactorizedPredictiveStateClosureModel", "_validate_training_arrays",
    "fit_event_factorized_pscd", "rollout_event_factorized_pscd",
    "hierarchical_relative_gain_interval", "hierarchical_statistic_interval",
    "leave_one_family_out_relative_gain",
    "event_classification_metrics", "TierAGates", "TierBGates", "sequential_decision",
]
analysis_helpers = STAGE38.analysis_helpers + "\n\nimport torch.nn.functional as F\n\n" + function_sources(
    NUMERICAL.read_text(), new_helpers
)
analysis_helpers = analysis_helpers.replace(
    "# Tested predictive-state adapter, controls, metrics, and decision gates.",
    "# Tested coefficient audit, event/reset controls, metrics, and decisions.",
    1,
)
analysis_helpers = analysis_helpers.replace(
    "class TierAGates:\n", "@dataclass(frozen=True)\nclass TierAGates:\n", 1
).replace(
    "class TierBGates:\n", "@dataclass(frozen=True)\nclass TierBGates:\n", 1
)
if (
    analysis_helpers.count("@dataclass(frozen=True)\nclass TierAGates:") != 1
    or analysis_helpers.count("@dataclass(frozen=True)\nclass TierBGates:") != 1
):
    raise RuntimeError("Stage 38.1 rendered gate decorators were not restored exactly once")


source_binding = r'''# Bind only the three Stage 38 development shard families.
SOURCE_STAGE38 = Path(SOURCE_STAGE38_RUN_DIR)
SOURCE_RECORDS = {}
SOURCE_SHARD_HASHES = []
SOURCE_BINDING_VERIFIED = False


def source_stage38_path(short, split, record_id):
    if str(split) not in DEVELOPMENT_SPLITS:
        raise RuntimeError(f"forbidden Stage 38.1 source split: {split!r}")
    return SOURCE_STAGE38 / "prefix_carrier_paths" / f"{short}_{split}_{int(record_id)}.npz"


def validate_source_shard(short, split, record, expected_words):
    path = source_stage38_path(short, split, record["record_id"])
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing full Stage 38 Drive shard: {path}. The compact result zip is not enough; "
            "retain the original pilot_ceb85af5b4b9 Drive directory."
        )
    required = {
        "identity", "word_names", "word_lengths", "initial_carrier", "initial_physical",
        "actions", "carrier_paths", "native_grounded_paths", "simulator_grounded_paths",
        "path_mask", "source_modes", "target_modes",
    }
    with np.load(path, allow_pickle=False) as payload:
        if not required.issubset(payload.files):
            raise RuntimeError(f"incomplete Stage 38 source shard: {path.name}")
        identity = str(payload["identity"].item())
        prefix = (
            f"{SOURCE_STAGE38_PROTOCOL_ID}:{SOURCE_STAGE38_RUN_SIGNATURE}:{short}:"
            f"{int(record['record_id'])}:{split}:horizon-matched-carrier-path-v1:"
        )
        if not identity.startswith(prefix):
            raise RuntimeError(f"source shard identity mismatch: {path.name}")
        words = [str(value) for value in payload["word_names"]]
        if words != list(expected_words):
            raise RuntimeError(f"source shard word order mismatch: {path.name}")
        if payload["initial_carrier"].shape != (FIXED_CARRIER_DIM,):
            raise RuntimeError(f"source carrier width changed: {path.name}")
        if not np.all(np.isfinite(payload["carrier_paths"])):
            raise RuntimeError(f"source carrier path is nonfinite: {path.name}")
    SOURCE_SHARD_HASHES.append({
        "model": short, "split": split, "record_id": int(record["record_id"]),
        "sha256": sha256_file(path),
    })


if not PIPELINE_FAILED:
    try:
        verify_executed_notebook_through(
            "# Bind only the three Stage 38 development shard families."
        )
        if EVALUATION_ACCESS_PERMITTED or PLANNING_ACCESS_PERMITTED:
            raise RuntimeError("Stage 38.1 development isolation was disabled")
        required = [
            SOURCE_STAGE38 / "config.json", SOURCE_STAGE38 / "source_identity.json",
            SOURCE_STAGE38 / "FAILURE_TRACE.txt",
            SOURCE_STAGE38 / "checkpoints/stage38_non_evaluation_paths_complete.json",
        ]
        if not all(path.is_file() for path in required):
            raise FileNotFoundError(
                "The exact complete Stage 38 Drive run pilot_ceb85af5b4b9 is required."
            )
        source_config = json.loads(required[0].read_text())
        source_identity_payload = json.loads(required[1].read_text())
        source_checkpoint = json.loads(required[3].read_text())
        if source_config.get("PROTOCOL_ID") != SOURCE_STAGE38_PROTOCOL_ID:
            raise RuntimeError("Stage 38 source protocol mismatch")
        if source_config.get("run_signature") != SOURCE_STAGE38_RUN_SIGNATURE:
            raise RuntimeError("Stage 38 source signature mismatch")
        if source_identity_payload.get("resolved_commit") != SOURCE_STAGE38_COMMIT:
            raise RuntimeError("Stage 38 source commit mismatch")
        if source_identity_payload.get("status") != "SOURCE_BOUND_EXECUTION_VERIFIED":
            raise RuntimeError("Stage 38 source execution was not verified")
        if required[2].read_text().strip() != "NONE":
            raise RuntimeError("Stage 38 source reports a pipeline failure")
        if source_checkpoint.get("payload", {}).get("evaluation_paths_materialized") is not False:
            raise RuntimeError("non-evaluation checkpoint metadata changed")

        split_words = {
            "construction": CONSTRUCTION_WORD_NAMES,
            "model_selection": MODEL_SELECTION_WORD_NAMES,
            "calibration": CALIBRATION_WORD_NAMES,
        }
        expected_ranges = {
            "construction": (56000, 57999),
            "model_selection": (58000, 59999),
            "calibration": (60000, 61999),
        }
        for split in DEVELOPMENT_SPLITS:
            manifest = SOURCE_STAGE38 / "design" / f"selected_{split}_trajectories.json"
            validate_digest_sidecar(manifest)
            payload = json.loads(manifest.read_text())
            records = list(payload["records"])
            target = {"construction": 24, "model_selection": 16, "calibration": 24}[split]
            groups = {int(row["trajectory_id"]) for row in records}
            low, high = expected_ranges[split]
            if payload.get("protocol_id") != SOURCE_STAGE38_PROTOCOL_ID:
                raise RuntimeError(f"{split} source manifest protocol mismatch")
            if len(groups) != target or not all(low <= value <= high for value in groups):
                raise RuntimeError(f"{split} trajectory-family binding changed")
            if any(str(row["split"]) != split for row in records):
                raise RuntimeError(f"{split} record label changed")
            SOURCE_RECORDS[split] = records
            for short in ["jepa", "dino"]:
                for record in records:
                    validate_source_shard(short, split, record, split_words[split])
        family_sets = [
            {int(row["trajectory_id"]) for row in SOURCE_RECORDS[split]}
            for split in DEVELOPMENT_SPLITS
        ]
        if any(
            not family_sets[left].isdisjoint(family_sets[right])
            for left in range(3) for right in range(left + 1, 3)
        ):
            raise RuntimeError("Stage 38 development trajectory families overlap")
        SOURCE_BINDING_VERIFIED = True
        write_json(OUT / "stage38_source_development_binding.json", {
            "verified": True, "source_protocol_id": SOURCE_STAGE38_PROTOCOL_ID,
            "source_run_signature": SOURCE_STAGE38_RUN_SIGNATURE,
            "source_commit": SOURCE_STAGE38_COMMIT,
            "splits": {
                split: {
                    "records": len(SOURCE_RECORDS[split]),
                    "trajectory_families": len(family_sets[index]),
                }
                for index, split in enumerate(DEVELOPMENT_SPLITS)
            },
            "validated_shards": len(SOURCE_SHARD_HASHES),
            "evaluation_files_read": 0, "planning_files_read": 0,
            "shard_hashes": SOURCE_SHARD_HASHES,
        })
        atomic_checkpoint("stage381_source_binding", {
            "verified": True, "validated_shards": len(SOURCE_SHARD_HASHES),
            "evaluation_files_read": 0,
        })
        print("Development-only Stage 38 source binding verified.")
    except Exception:
        record_failure("stage381_source_binding")
'''


data_helpers = r'''# Load construction, model-selection, and calibration arrays only.
def load_stage381_sequences(short, split):
    if str(split) not in DEVELOPMENT_SPLITS:
        raise RuntimeError(f"Stage 38.1 cannot load split {split!r}")
    rows = {key: [] for key in [
        "initial_carrier", "initial_physical", "actions", "carrier", "native",
        "simulator", "mask", "source_mode", "target_mode", "word", "length",
        "group", "record_id", "initial_mode",
    ]}
    for record in SOURCE_RECORDS[str(split)]:
        path = source_stage38_path(short, split, record["record_id"])
        with np.load(path, allow_pickle=False) as payload:
            words = [str(value) for value in payload["word_names"]]
            count = len(words)
            rows["initial_carrier"].extend(np.repeat(payload["initial_carrier"][None], count, axis=0))
            rows["initial_physical"].extend(np.repeat(payload["initial_physical"][None], count, axis=0))
            mapping = {
                "actions": "actions", "carrier": "carrier_paths",
                "native": "native_grounded_paths", "simulator": "simulator_grounded_paths",
                "mask": "path_mask", "source_mode": "source_modes", "target_mode": "target_modes",
            }
            for key, payload_key in mapping.items():
                rows[key].extend(payload[payload_key])
            rows["word"].extend(words)
            rows["length"].extend(payload["word_lengths"].astype(int).tolist())
            rows["group"].extend([int(record["trajectory_id"])] * count)
            rows["record_id"].extend([int(record["record_id"])] * count)
            rows["initial_mode"].extend([str(record["mode"])] * count)
    for key in ["initial_carrier", "initial_physical", "actions", "carrier", "native", "simulator"]:
        rows[key] = np.asarray(rows[key], dtype=np.float64)
    rows["mask"] = np.asarray(rows["mask"], dtype=bool)
    for key in ["source_mode", "target_mode", "word", "initial_mode"]:
        rows[key] = np.asarray(rows[key]).astype(str)
    for key in ["length", "group", "record_id"]:
        rows[key] = np.asarray(rows[key], dtype=np.int64)
    rows["event"] = macro_contact_events(rows["source_mode"], rows["target_mode"])
    if not all(len(value) == len(rows["word"]) for value in rows.values()):
        raise RuntimeError(f"Stage 38.1 {short}/{split} arrays are misaligned")
    return rows


def terminal_path_labels(values, mask):
    labels = np.asarray(values).astype(str)
    valid = np.asarray(mask, dtype=bool)
    indices = np.max(np.where(valid, np.arange(valid.shape[1])[None, :], -1), axis=1)
    if np.any(indices < 0):
        raise ValueError("every sequence requires a terminal step")
    return labels[np.arange(len(labels)), indices]


def stage381_physical_scale(data):
    return np.maximum(np.std(data["simulator"][data["mask"]], axis=0, ddof=1), 1e-8)


DEVELOPMENT_DATA = {}
PHYSICAL_SCALES = {}
if not PIPELINE_FAILED and SOURCE_BINDING_VERIFIED:
    try:
        verify_executed_notebook_through(
            "# Load construction, model-selection, and calibration arrays only."
        )
        for short in ["jepa", "dino"]:
            DEVELOPMENT_DATA[short] = {
                split: load_stage381_sequences(short, split) for split in DEVELOPMENT_SPLITS
            }
            PHYSICAL_SCALES[short] = stage381_physical_scale(
                DEVELOPMENT_DATA[short]["construction"]
            )
            construction = DEVELOPMENT_DATA[short]["construction"]
            if not np.any(construction["event"][construction["mask"]]):
                raise RuntimeError(f"{short} construction panel has no contact event")
        write_json(OUT / "development_data_contract.json", {
            short: {
                split: {
                    "rows": len(DEVELOPMENT_DATA[short][split]["word"]),
                    "families": len(np.unique(DEVELOPMENT_DATA[short][split]["group"])),
                    "valid_transitions": int(np.sum(DEVELOPMENT_DATA[short][split]["mask"])),
                    "event_prevalence": float(np.mean(
                        DEVELOPMENT_DATA[short][split]["event"][DEVELOPMENT_DATA[short][split]["mask"]]
                    )),
                }
                for split in DEVELOPMENT_SPLITS
            }
            for short in ["jepa", "dino"]
        })
    except Exception:
        record_failure("stage381_development_data_load")
'''


tier_a_fit = r'''# Fit Tier A construction-only matched controls for two screening seeds.
ARTIFACT_DIR = OUT / "frozen_models"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
TIER_A_MODELS = {short: {} for short in ["jepa", "dino"]}
TIER_A_DIAGNOSTICS = []
LEGACY_MIGRATION_ROOT = Path(LEGACY_STAGE381_RUN_DIR)
LEGACY_MIGRATION_AVAILABLE = LEGACY_MIGRATION_ROOT.is_dir()
LEGACY_MIGRATION_VERIFIED = False
LEGACY_IMPORT_ROWS = []


def stage381_artifact_paths(short, tier, variant, seed):
    stem = ARTIFACT_DIR / f"stage381_{short}_{tier}_{variant}_seed{int(seed)}"
    return Path(str(stem) + ".npz"), Path(str(stem) + "_schema.json")


def stage381_migration_receipt_path(short, variant, seed):
    return ARTIFACT_DIR / f"stage381_{short}_tier_a_{variant}_seed{int(seed)}_migration.json"


def record_legacy_import(row):
    identity = (row["model"], row["variant"], int(row["seed"]))
    existing = {
        (item["model"], item["variant"], int(item["seed"]))
        for item in LEGACY_IMPORT_ROWS
    }
    if identity not in existing:
        LEGACY_IMPORT_ROWS.append(dict(row))


def encode_artifact(value, arrays):
    if isinstance(value, np.ndarray):
        key = f"array_{len(arrays):05d}"
        arrays[key] = value
        return {"kind": "array", "key": key}
    if isinstance(value, dict):
        return {"kind": "dict", "items": {
            str(key): encode_artifact(item, arrays) for key, item in sorted(value.items())
        }}
    if isinstance(value, (list, tuple)):
        return {"kind": "list", "items": [encode_artifact(item, arrays) for item in value]}
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return {"kind": "scalar", "value": value}
    raise TypeError(f"unsupported artifact value: {type(value)}")


def decode_artifact(node, arrays):
    if node["kind"] == "array":
        return np.asarray(arrays[node["key"]])
    if node["kind"] == "dict":
        return {key: decode_artifact(value, arrays) for key, value in node["items"].items()}
    if node["kind"] == "list":
        return [decode_artifact(value, arrays) for value in node["items"]]
    if node["kind"] == "scalar":
        return node["value"]
    raise ValueError(f"unknown artifact node {node['kind']!r}")


def save_stage381_artifact(short, tier, variant, seed, artifact):
    array_path, schema_path = stage381_artifact_paths(short, tier, variant, seed)
    arrays = {}
    schema = encode_artifact(artifact, arrays)
    atomic_npz(array_path, **arrays)
    write_digest_sidecar(array_path)
    write_json(schema_path, {
        "protocol_id": PROTOCOL_ID, "run_signature": RUN_SIGNATURE,
        "model": short, "tier": tier, "variant": variant, "seed": int(seed),
        "array_sha256": sha256_file(array_path), "schema": schema,
    })
    write_digest_sidecar(schema_path)


def load_stage381_artifact(short, tier, variant, seed):
    array_path, schema_path = stage381_artifact_paths(short, tier, variant, seed)
    validate_digest_sidecar(array_path)
    validate_digest_sidecar(schema_path)
    metadata = json.loads(schema_path.read_text())
    observed = (
        metadata["protocol_id"], metadata["run_signature"], metadata["model"],
        metadata["tier"], metadata["variant"], int(metadata["seed"]),
    )
    expected = (PROTOCOL_ID, RUN_SIGNATURE, short, tier, variant, int(seed))
    if observed != expected or metadata["array_sha256"] != sha256_file(array_path):
        raise RuntimeError(f"Stage 38.1 artifact binding failed: {short}/{tier}/{variant}/{seed}")
    with np.load(array_path, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    artifact = decode_artifact(metadata["schema"], arrays)
    if tier == "tier_a":
        receipt_path = stage381_migration_receipt_path(short, variant, seed)
        if receipt_path.is_file() and Path(str(receipt_path) + ".sha256").is_file():
            validate_digest_sidecar(receipt_path)
            receipt = json.loads(receipt_path.read_text())
            if (
                receipt.get("protocol_id") != PROTOCOL_ID
                or receipt.get("run_signature") != RUN_SIGNATURE
                or receipt.get("model") != short
                or receipt.get("variant") != variant
                or int(receipt.get("seed", -1)) != int(seed)
                or receipt.get("current_array_sha256") != sha256_file(array_path)
                or receipt.get("current_schema_sha256") != sha256_file(schema_path)
            ):
                raise RuntimeError(f"migration receipt binding failed: {short}/{variant}/{seed}")
            record_legacy_import(receipt)
    return artifact


def validate_legacy_migration_root():
    required = {
        "config": LEGACY_MIGRATION_ROOT / "config.json",
        "identity": LEGACY_MIGRATION_ROOT / "source_identity.json",
        "failure": LEGACY_MIGRATION_ROOT / "FAILURE_TRACE.txt",
        "decision": LEGACY_MIGRATION_ROOT / "stage381_decision.json",
        "checkpoint": LEGACY_MIGRATION_ROOT / "checkpoints/stage381_tier_a_screening_models.json",
    }
    if not all(path.is_file() for path in required.values()):
        raise RuntimeError("the v1 Stage 38.1 migration root is incomplete")
    old_config = json.loads(required["config"].read_text())
    old_identity = json.loads(required["identity"].read_text())
    old_decision = json.loads(required["decision"].read_text())
    old_checkpoint = json.loads(required["checkpoint"].read_text())
    failure = required["failure"].read_text()
    if (
        old_config.get("PROTOCOL_ID") != LEGACY_STAGE381_PROTOCOL_ID
        or old_config.get("run_signature") != LEGACY_STAGE381_RUN_SIGNATURE
        or old_identity.get("resolved_commit") != LEGACY_STAGE381_SOURCE_COMMIT
        or old_identity.get("status") != "SOURCE_BOUND_EXECUTION_VERIFIED"
        or old_decision.get("status") != "INCONCLUSIVE_PIPELINE_FAILURE"
        or old_decision.get("evaluation_rows_read") != 0
        or old_decision.get("planning_rows_read") != 0
        or old_checkpoint.get("protocol_id") != LEGACY_STAGE381_PROTOCOL_ID
        or old_checkpoint.get("run_signature") != LEGACY_STAGE381_RUN_SIGNATURE
    ):
        raise RuntimeError("the v1 Stage 38.1 migration binding changed")
    payload = old_checkpoint.get("payload", {})
    if (
        payload.get("seeds") != SCREENING_SEEDS
        or payload.get("models") != 2
        or payload.get("variants_per_model_seed") != len(TIER_A_VARIANTS)
        or payload.get("training_split") != "construction"
        or payload.get("calibration_rows_read_during_fit") != 0
    ):
        raise RuntimeError("the v1 Stage 38.1 training-only checkpoint changed")
    if (
        "STAGE: stage381_tier_a_decision" not in failure
        or "TypeError: TierAGates() takes no arguments" not in failure
    ):
        raise RuntimeError("the v1 Stage 38.1 failure boundary changed")
    forbidden_outputs = [
        LEGACY_MIGRATION_ROOT / "tier_a_decision.json",
        LEGACY_MIGRATION_ROOT / "evaluation_evidence/tier_a_calibration_rows.csv",
        LEGACY_MIGRATION_ROOT / "evaluation_evidence/tier_a_final_decisions.json",
    ]
    if any(path.exists() for path in forbidden_outputs):
        raise RuntimeError("v1 calibration outcomes exist; migration is no longer construction-only")
    return {
        "legacy_protocol_id": LEGACY_STAGE381_PROTOCOL_ID,
        "legacy_run_signature": LEGACY_STAGE381_RUN_SIGNATURE,
        "legacy_source_commit": LEGACY_STAGE381_SOURCE_COMMIT,
        "failure_boundary": "TierAGates constructor before any persisted calibration decision",
        "calibration_outcomes_imported": 0,
        "evaluation_rows_imported": 0,
    }


def tier_a_objective(short, variant):
    full = float(FULL_OUTER_WEIGHTS[short])
    table = {
        "pscd": dict(
            free_weight=1.0, semigroup_weight=0.0,
            semigroup_component_weights=FULL_COMPONENT_WEIGHTS,
        ),
        "mass_overshoot": dict(
            free_weight=1.0, semigroup_weight=full,
            semigroup_component_weights=LATENT_ONLY_COMPONENT_WEIGHTS,
        ),
        "coefficient_overshoot": dict(
            free_weight=1.0, semigroup_weight=COEFFICIENT_MATCHED_OUTER_WEIGHTS[short],
            semigroup_component_weights=LATENT_ONLY_COMPONENT_WEIGHTS,
        ),
        "full": dict(
            free_weight=1.0, semigroup_weight=full,
            semigroup_component_weights=FULL_COMPONENT_WEIGHTS,
        ),
    }
    return table[str(variant)]


def load_legacy_tier_a_artifact(short, variant, seed):
    stem = (
        LEGACY_MIGRATION_ROOT / "frozen_models"
        / f"stage381_{short}_tier_a_{variant}_seed{int(seed)}"
    )
    array_path = Path(str(stem) + ".npz")
    schema_path = Path(str(stem) + "_schema.json")
    validate_digest_sidecar(array_path)
    validate_digest_sidecar(schema_path)
    metadata = json.loads(schema_path.read_text())
    observed = (
        metadata.get("protocol_id"), metadata.get("run_signature"), metadata.get("model"),
        metadata.get("tier"), metadata.get("variant"), int(metadata.get("seed", -1)),
    )
    expected = (
        LEGACY_STAGE381_PROTOCOL_ID, LEGACY_STAGE381_RUN_SIGNATURE, short,
        "tier_a", variant, int(seed),
    )
    if observed != expected or metadata.get("array_sha256") != sha256_file(array_path):
        raise RuntimeError(f"legacy artifact binding failed: {short}/{variant}/{seed}")
    with np.load(array_path, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    artifact = decode_artifact(metadata["schema"], arrays)
    objective = tier_a_objective(short, variant)
    config = artifact.get("config", {})
    train = DEVELOPMENT_DATA[short]["construction"]
    if (
        int(config.get("seed", -1)) != int(seed)
        or int(config.get("epochs", -1)) != FINAL_EPOCHS
        or int(config.get("carrier_dim", -1)) != FIXED_CARRIER_DIM
        or int(config.get("history_length", -1)) != FIXED_HISTORY_LENGTH
        or int(config.get("latent_dim", -1)) != FIXED_LATENT_DIM
        or int(config.get("action_dim", -1)) != train["actions"].shape[2]
        or int(config.get("physical_dim", -1)) != train["native"].shape[2]
        or str(config.get("dynamics")) != FIXED_DYNAMICS
        or float(config.get("learning_rate", -1.0)) != PSCD_LEARNING_RATE
        or float(config.get("free_weight", -1.0)) != float(objective["free_weight"])
        or float(config.get("semigroup_weight", -1.0)) != float(objective["semigroup_weight"])
        or list(config.get("semigroup_component_weights", []))
            != list(objective["semigroup_component_weights"])
        or list(config.get("semigroup_horizons", [])) != SEMIGROUP_HORIZONS
    ):
        raise RuntimeError(f"legacy artifact objective changed: {short}/{variant}/{seed}")
    state_dict = artifact.get("state_dict", {})
    if not state_dict or any(
        not np.all(np.isfinite(np.asarray(value))) for value in state_dict.values()
    ):
        raise RuntimeError(f"legacy artifact parameters are invalid: {short}/{variant}/{seed}")
    scalar_checks = [artifact.get("loss_initial"), artifact.get("loss_final")]
    if not np.all(np.isfinite(np.asarray(scalar_checks, dtype=np.float64))):
        raise RuntimeError(f"legacy artifact losses are invalid: {short}/{variant}/{seed}")
    import_row = {
        "protocol_id": PROTOCOL_ID, "run_signature": RUN_SIGNATURE,
        "model": short, "variant": variant, "seed": int(seed),
        "source_array_sha256": sha256_file(array_path),
        "source_schema_sha256": sha256_file(schema_path),
        "source_run_signature": LEGACY_STAGE381_RUN_SIGNATURE,
        "training_split": "construction", "calibration_outcomes_imported": 0,
    }
    save_stage381_artifact(short, "tier_a", variant, seed, artifact)
    current_array, current_schema = stage381_artifact_paths(
        short, "tier_a", variant, seed
    )
    import_row.update({
        "current_array_sha256": sha256_file(current_array),
        "current_schema_sha256": sha256_file(current_schema),
    })
    receipt_path = stage381_migration_receipt_path(short, variant, seed)
    write_json(receipt_path, import_row)
    write_digest_sidecar(receipt_path)
    record_legacy_import(import_row)
    return artifact


def fit_or_load_tier_a(short, variant, seed):
    array_path, schema_path = stage381_artifact_paths(short, "tier_a", variant, seed)
    if all(path.is_file() for path in [array_path, schema_path, Path(str(array_path) + ".sha256"), Path(str(schema_path) + ".sha256")]):
        return load_stage381_artifact(short, "tier_a", variant, seed)
    if LEGACY_MIGRATION_VERIFIED and int(seed) in SCREENING_SEEDS:
        return load_legacy_tier_a_artifact(short, variant, seed)
    train = DEVELOPMENT_DATA[short]["construction"]
    objective = tier_a_objective(short, variant)
    artifact = fit_weighted_semigroup_predictive_state_closure(
        train["initial_carrier"], train["actions"], train["carrier"], train["native"],
        train["mask"], history_length=FIXED_HISTORY_LENGTH, latent_dim=FIXED_LATENT_DIM,
        dynamics=FIXED_DYNAMICS, epochs=ACTIVE_FINAL_EPOCHS,
        learning_rate=PSCD_LEARNING_RATE, seed=int(seed),
        semigroup_horizons=SEMIGROUP_HORIZONS, **objective,
    )
    save_stage381_artifact(short, "tier_a", variant, seed, artifact)
    return artifact


def freeze_tier_a_seed(seed):
    for short in ["jepa", "dino"]:
        train = DEVELOPMENT_DATA[short]["construction"]
        TIER_A_MODELS[short].setdefault(int(seed), {})
        for variant in TIER_A_VARIANTS:
            artifact = fit_or_load_tier_a(short, variant, seed)
            expected = tier_a_objective(short, variant)
            if (
                int(artifact["config"]["seed"]) != int(seed)
                or float(artifact["config"]["semigroup_weight"]) != float(expected["semigroup_weight"])
                or list(artifact["config"]["semigroup_component_weights"])
                    != list(expected["semigroup_component_weights"])
            ):
                raise RuntimeError(f"Tier A objective binding failed: {short}/{variant}/{seed}")
            TIER_A_MODELS[short][int(seed)][variant] = artifact
            diagnostic = semigroup_component_diagnostics(
                artifact, train["initial_carrier"], train["actions"], train["carrier"],
                train["native"], train["mask"],
            )
            TIER_A_DIAGNOSTICS.append({
                "model": short, "variant": variant, "seed": int(seed),
                "epochs": int(artifact["config"]["epochs"]),
                "optimizer_steps": int(artifact["config"]["epochs"]),
                "trainable_parameters": int(sum(np.asarray(value).size for value in artifact["state_dict"].values())),
                **diagnostic,
            })
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if not PIPELINE_FAILED and SOURCE_BINDING_VERIFIED:
    try:
        verify_executed_notebook_through(
            "# Fit Tier A construction-only matched controls for two screening seeds."
        )
        if LEGACY_MIGRATION_AVAILABLE:
            migration_binding = validate_legacy_migration_root()
            LEGACY_MIGRATION_VERIFIED = True
        else:
            migration_binding = {
                "legacy_run_present": False, "fallback": "refit screening models",
            }
        for seed in SCREENING_SEEDS:
            freeze_tier_a_seed(seed)
        write_json(EVIDENCE_DIR / "tier_a_legacy_migration.json", {
            **migration_binding, "migration_verified": LEGACY_MIGRATION_VERIFIED,
            "imported_artifacts": LEGACY_IMPORT_ROWS,
            "imported_count": len(LEGACY_IMPORT_ROWS),
        })
        write_json(EVIDENCE_DIR / "tier_a_component_diagnostics.json", TIER_A_DIAGNOSTICS)
        atomic_checkpoint("stage381_tier_a_screening_models", {
            "seeds": SCREENING_SEEDS, "models": 2,
            "variants_per_model_seed": len(TIER_A_VARIANTS),
            "training_split": "construction", "calibration_rows_read_during_fit": 0,
            "legacy_migration_verified": LEGACY_MIGRATION_VERIFIED,
            "legacy_imported_artifacts": len(LEGACY_IMPORT_ROWS),
        })
    except Exception:
        record_failure("stage381_tier_a_screening_fit")
'''


tier_a_decision = r'''# Score untouched calibration, then conditionally run the third seed.
TIER_A_ROWS = []
TIER_A_SCREEN = {}
TIER_A_FINAL = {}
TIER_A_PROMOTED = False
THIRD_SEED_EXECUTED = False


def score_base_artifact(artifact, data, scale, histories_override=None):
    rollout = rollout_predictive_state_closure(
        artifact, data["initial_carrier"], data["actions"], data["carrier"], data["mask"],
        histories_override=histories_override,
    )
    errors = scaled_path_mse(
        rollout["physical"], data["simulator"], rollout["evaluation_mask"], scale,
    )
    return errors, rollout


def safe_ratio(primary, comparator):
    first, second = float(primary), float(comparator)
    if second == 0.0:
        return 1.0 if first == 0.0 else float("inf")
    return first / second


def summarize_tier_a_model(short, seeds, require_three):
    data, scale = DEVELOPMENT_DATA[short]["calibration"], PHYSICAL_SCALES[short]
    groups = data["group"]
    histories = history_tensor(
        data["initial_carrier"], data["carrier"], data["mask"], FIXED_HISTORY_LENGTH
    )
    errors = {variant: [] for variant in TIER_A_VARIANTS}
    wrong = []
    source_terminal = terminal_path_labels(data["source_mode"], data["mask"])
    target_terminal = terminal_path_labels(data["target_mode"], data["mask"])
    seed_summaries = []
    for seed in seeds:
        for variant in TIER_A_VARIANTS:
            values, _ = score_base_artifact(
                TIER_A_MODELS[short][int(seed)][variant], data, scale
            )
            errors[variant].append(values)
        false_history = permute_past_history(
            histories, groups, data["mask"],
            seed=stable_seed(CONTROL_SEED, "stage381_wrong_history", short, int(seed)),
        )
        false_values, _ = score_base_artifact(
            TIER_A_MODELS[short][int(seed)]["full"], data, scale,
            histories_override=false_history,
        )
        wrong.append(false_values)
    errors = {key: np.stack(value) for key, value in errors.items()}
    wrong = np.stack(wrong)
    full, coefficient = errors["full"], errors["coefficient_overshoot"]
    coefficient_gain = (np.mean(coefficient) - np.mean(full)) / max(np.mean(coefficient), 1e-12)
    history_gain = (np.mean(wrong) - np.mean(full)) / max(np.mean(wrong), 1e-12)
    coefficient_interval = hierarchical_relative_gain_interval(
        full, coefficient, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
        seed=stable_seed(BOOTSTRAP_SEED, "tier_a_coefficient", short, len(seeds)),
        confidence=0.90,
    )
    history_interval = hierarchical_relative_gain_interval(
        full, wrong, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
        seed=stable_seed(BOOTSTRAP_SEED, "tier_a_history", short, len(seeds)),
        confidence=0.90,
    )
    pooled_full, pooled_coefficient = tail_risk_summary(full.reshape(-1)), tail_risk_summary(coefficient.reshape(-1))
    full_cvar95_interval = hierarchical_statistic_interval(
        full, groups, statistic="cvar95", draws=ACTIVE_BOOTSTRAP_DRAWS,
        seed=stable_seed(BOOTSTRAP_SEED, "tier_a_cvar95", short, len(seeds)),
    )
    tail_noninferior = all([
        safe_ratio(pooled_full["p95"], pooled_coefficient["p95"]) <= MAX_TAIL_RATIO,
        safe_ratio(pooled_full["cvar95"], pooled_coefficient["cvar95"]) <= MAX_TAIL_RATIO,
        safe_ratio(
            pooled_full["catastrophic_rate_gt_1"], pooled_coefficient["catastrophic_rate_gt_1"]
        ) <= MAX_TAIL_RATIO,
    ])
    for seed_index, seed in enumerate(seeds):
        values = full[seed_index]
        tail = tail_risk_summary(values)
        lengths = {
            str(length): float(np.mean(values[data["length"] == length]))
            for length in sorted(np.unique(data["length"]))
        }
        initial_modes = {
            str(mode): float(np.mean(values[data["initial_mode"] == mode]))
            for mode in sorted(np.unique(data["initial_mode"]))
        }
        terminal_modes = {
            str(mode): float(np.mean(values[target_terminal == mode]))
            for mode in sorted(np.unique(target_terminal))
        }
        mode_pairs = {}
        for source, target in sorted(set(zip(source_terminal, target_terminal))):
            selected = (source_terminal == source) & (target_terminal == target)
            mode_pairs[f"{source}->{target}"] = float(np.mean(values[selected]))
        seed_gain = (np.mean(coefficient[seed_index]) - np.mean(values)) / max(
            np.mean(coefficient[seed_index]), 1e-12
        )
        seed_summaries.append({
            "seed": int(seed), "coefficient_gain": float(seed_gain),
            "history_gain": float((np.mean(wrong[seed_index]) - np.mean(values)) / max(np.mean(wrong[seed_index]), 1e-12)),
            "mean": float(np.mean(values)), "tail": tail,
            "length_physical_nmse": lengths,
            "initial_mode_physical_nmse": initial_modes,
            "terminal_mode_physical_nmse": terminal_modes,
            "mode_pair_physical_nmse": mode_pairs,
        })
        for row in range(len(data["word"])):
            TIER_A_ROWS.append({
                "model": short, "seed": int(seed), "trajectory_id": int(groups[row]),
                "record_id": int(data["record_id"][row]), "word": str(data["word"][row]),
                "word_length": int(data["length"][row]),
                "source_terminal_mode": str(source_terminal[row]),
                "target_terminal_mode": str(target_terminal[row]),
                "pscd_nmse": float(errors["pscd"][seed_index, row]),
                "mass_overshoot_nmse": float(errors["mass_overshoot"][seed_index, row]),
                "coefficient_overshoot_nmse": float(coefficient[seed_index, row]),
                "full_nmse": float(values[row]), "wrong_history_nmse": float(wrong[seed_index, row]),
            })
    absolute = all(
        row["mean"] <= MAX_MEAN_PHYSICAL_NMSE
        and row["tail"]["p95"] <= MAX_P95_PHYSICAL_NMSE
        and row["tail"]["catastrophic_rate_gt_1"] <= MAX_CATASTROPHIC_RATE
        and all(value <= MAX_LENGTH_PHYSICAL_NMSE for value in row["length_physical_nmse"].values())
        and all(
            value <= MAX_MODE_PHYSICAL_NMSE
            for value in list(row["initial_mode_physical_nmse"].values())
                + list(row["terminal_mode_physical_nmse"].values())
        )
        for row in seed_summaries
    )
    all_signs = all(row["coefficient_gain"] > 0 and row["history_gain"] > 0 for row in seed_summaries)
    gates = TierAGates(
        coefficient_specificity=bool(
            coefficient_gain >= MIN_COEFFICIENT_GAIN and coefficient_interval[0] > 0
        ),
        tail_noninferiority=bool(tail_noninferior),
        correct_history_specificity=bool(history_gain >= MIN_HISTORY_GAIN and history_interval[0] > 0),
        absolute_viability=bool(absolute),
        three_seed_stability=bool((not require_three or len(seeds) == 3) and all_signs),
    )
    decision = sequential_decision(
        gates, passed_status="tier_a_promoted" if require_three else "tier_a_screen_passed"
    )
    family_effects = {}
    leave_one_family_out_tail = {}
    for group in np.unique(groups):
        selected, keep = groups == group, groups != group
        family_effects[str(group)] = {
            "full_mean": float(np.mean(full[:, selected])),
            "coefficient_mean": float(np.mean(coefficient[:, selected])),
            "relative_gain": float(
                (np.mean(coefficient[:, selected]) - np.mean(full[:, selected]))
                / max(np.mean(coefficient[:, selected]), 1e-12)
            ),
        }
        full_without = tail_risk_summary(full[:, keep].reshape(-1))
        coefficient_without = tail_risk_summary(coefficient[:, keep].reshape(-1))
        leave_one_family_out_tail[str(group)] = {
            "p95_ratio": float(safe_ratio(full_without["p95"], coefficient_without["p95"])),
            "cvar95_ratio": float(safe_ratio(full_without["cvar95"], coefficient_without["cvar95"])),
            "catastrophic_rate_difference": float(
                full_without["catastrophic_rate_gt_1"]
                - coefficient_without["catastrophic_rate_gt_1"]
            ),
        }
    return {
        "decision": decision, "seeds": list(map(int, seeds)),
        "coefficient_gain": float(coefficient_gain),
        "coefficient_gain_one_sided_95_interval": coefficient_interval,
        "correct_history_gain": float(history_gain),
        "correct_history_gain_one_sided_95_interval": history_interval,
        "full_tail": pooled_full, "full_cvar95_hierarchical_interval": full_cvar95_interval,
        "coefficient_tail": pooled_coefficient, "family_effects": family_effects,
        "leave_one_family_out_coefficient_gain": leave_one_family_out_relative_gain(
            full, coefficient, groups
        ),
        "leave_one_family_out_tail_ratios": leave_one_family_out_tail,
        "seed_summaries": seed_summaries,
    }


if not PIPELINE_FAILED and SOURCE_BINDING_VERIFIED:
    try:
        verify_executed_notebook_through(
            "# Score untouched calibration, then conditionally run the third seed."
        )
        TIER_A_SCREEN = {
            short: summarize_tier_a_model(short, SCREENING_SEEDS, require_three=False)
            for short in ["jepa", "dino"]
        }
        screen_passed = all(TIER_A_SCREEN[short]["decision"]["passed"] for short in ["jepa", "dino"])
        if screen_passed:
            THIRD_SEED_EXECUTED = True
            freeze_tier_a_seed(PRECOMMITTED_THIRD_SEED)
            TIER_A_ROWS.clear()
            TIER_A_FINAL = {
                short: summarize_tier_a_model(short, ALL_PRECOMMITTED_SEEDS, require_three=True)
                for short in ["jepa", "dino"]
            }
            TIER_A_PROMOTED = all(
                TIER_A_FINAL[short]["decision"]["passed"] for short in ["jepa", "dino"]
            )
        else:
            TIER_A_FINAL = TIER_A_SCREEN
        write_json(EVIDENCE_DIR / "tier_a_component_diagnostics.json", TIER_A_DIAGNOSTICS)
        write_csv(EVIDENCE_DIR / "tier_a_calibration_rows.csv", TIER_A_ROWS)
        write_json(EVIDENCE_DIR / "tier_a_screen_decisions.json", TIER_A_SCREEN)
        write_json(EVIDENCE_DIR / "tier_a_final_decisions.json", TIER_A_FINAL)
        write_json(OUT / "tier_a_decision.json", {
            "promoted": TIER_A_PROMOTED, "screen_passed": screen_passed,
            "third_seed_precommitted": PRECOMMITTED_THIRD_SEED,
            "third_seed_executed": THIRD_SEED_EXECUTED,
            "training_split": "construction", "scoring_split": "calibration",
            "evaluation_rows_read": 0, "planning_rows_read": 0,
            "models": TIER_A_FINAL,
        })
        atomic_checkpoint("stage381_tier_a_decision", {
            "promoted": TIER_A_PROMOTED, "third_seed_executed": THIRD_SEED_EXECUTED,
        })
        print(json.dumps({
            "tier_a_promoted": TIER_A_PROMOTED,
            "third_seed_executed": THIRD_SEED_EXECUTED,
            "screen": {short: TIER_A_SCREEN[short]["decision"] for short in ["jepa", "dino"]},
        }, indent=2))
    except Exception:
        record_failure("stage381_tier_a_decision")
'''


tier_b_fit = r'''# Conditionally test oracle event headroom before matched hybrid controls.
TIER_B_MODELS = {short: {} for short in ["jepa", "dino"]}
ORACLE_SUMMARIES = {}
ORACLE_HEADROOM_PASSED = False
TIER_B_CONTROLS_OPENED = False


def fit_or_load_tier_b(short, variant, seed, risk_weight=0.0):
    tier = "tier_b_risk" if risk_weight > 0 else "tier_b"
    array_path, schema_path = stage381_artifact_paths(short, tier, variant, seed)
    if all(path.is_file() for path in [array_path, schema_path, Path(str(array_path) + ".sha256"), Path(str(schema_path) + ".sha256")]):
        return load_stage381_artifact(short, tier, variant, seed)
    train = DEVELOPMENT_DATA[short]["construction"]
    events = train["event"]
    kind = "hybrid" if variant in {"hybrid", "shuffled", "risk"} else "smooth"
    if variant == "shuffled":
        events = shuffled_event_labels(
            events, train["mask"],
            seed=stable_seed(CONTROL_SEED, "stage381_shuffled_events", short, int(seed)),
        )
    artifact = fit_event_factorized_pscd(
        train["initial_carrier"], train["actions"], train["carrier"], train["native"],
        train["mask"], events, groups=train["group"],
        history_length=FIXED_HISTORY_LENGTH, latent_dim=FIXED_LATENT_DIM,
        dynamics=FIXED_DYNAMICS, epochs=ACTIVE_FINAL_EPOCHS,
        learning_rate=PSCD_LEARNING_RATE, seed=int(seed),
        semigroup_horizons=SEMIGROUP_HORIZONS,
        semigroup_weight=FULL_OUTER_WEIGHTS[short],
        semigroup_component_weights=FULL_COMPONENT_WEIGHTS,
        event_weight=EVENT_LOSS_WEIGHT, transition_kind=kind, event_hidden=EVENT_HIDDEN,
        risk_weight=float(risk_weight), risk_alpha=RISK_ALPHA,
    )
    save_stage381_artifact(short, tier, variant, seed, artifact)
    return artifact


def terminal_transition_mask(data, source="post_contact", target="contact"):
    return (
        terminal_path_labels(data["source_mode"], data["mask"]) == str(source)
    ) & (
        terminal_path_labels(data["target_mode"], data["mask"]) == str(target)
    )


def score_hybrid_artifact(artifact, data, scale, oracle_events=None):
    rollout = rollout_event_factorized_pscd(
        artifact, data["initial_carrier"], data["actions"], data["carrier"], data["mask"],
        oracle_events=oracle_events,
    )
    return scaled_path_mse(
        rollout["physical"], data["simulator"], rollout["evaluation_mask"], scale,
    ), rollout


def oracle_summary(short):
    data, scale = DEVELOPMENT_DATA[short]["calibration"], PHYSICAL_SCALES[short]
    transition = terminal_transition_mask(data)
    if not np.any(transition):
        raise RuntimeError(f"{short} calibration panel lacks post_contact->contact endpoints")
    full_errors, oracle_errors = [], []
    for seed in ALL_PRECOMMITTED_SEEDS:
        full, _ = score_base_artifact(TIER_A_MODELS[short][seed]["full"], data, scale)
        oracle, _ = score_hybrid_artifact(
            TIER_B_MODELS[short][seed]["hybrid"], data, scale, oracle_events=data["event"]
        )
        full_errors.append(full)
        oracle_errors.append(oracle)
    full, oracle = np.stack(full_errors), np.stack(oracle_errors)
    full_conditional = tail_risk_summary(full[:, transition].reshape(-1))
    oracle_conditional = tail_risk_summary(oracle[:, transition].reshape(-1))
    mean_gain = (np.mean(full[:, transition]) - np.mean(oracle[:, transition])) / max(
        np.mean(full[:, transition]), 1e-12
    )
    cvar_gain = (full_conditional["cvar95"] - oracle_conditional["cvar95"]) / max(
        full_conditional["cvar95"], 1e-12
    )
    p95_gain = (np.quantile(full, 0.95) - np.quantile(oracle, 0.95)) / max(
        np.quantile(full, 0.95), 1e-12
    )
    passed = bool(
        (mean_gain >= MIN_EVENT_CONDITIONAL_GAIN or cvar_gain >= MIN_EVENT_CONDITIONAL_GAIN)
        and p95_gain >= MIN_OVERALL_P95_GAIN
    )
    return {
        "passed": passed, "conditional_rows": int(np.sum(transition)),
        "conditional_mean_gain": float(mean_gain), "conditional_cvar95_gain": float(cvar_gain),
        "overall_p95_gain": float(p95_gain),
        "full_conditional": full_conditional, "oracle_conditional": oracle_conditional,
    }


if not PIPELINE_FAILED and TIER_A_PROMOTED:
    try:
        verify_executed_notebook_through(
            "# Conditionally test oracle event headroom before matched hybrid controls."
        )
        for short in ["jepa", "dino"]:
            for seed in ALL_PRECOMMITTED_SEEDS:
                TIER_B_MODELS[short].setdefault(seed, {})
                TIER_B_MODELS[short][seed]["hybrid"] = fit_or_load_tier_b(
                    short, "hybrid", seed
                )
        ORACLE_SUMMARIES = {short: oracle_summary(short) for short in ["jepa", "dino"]}
        ORACLE_HEADROOM_PASSED = all(ORACLE_SUMMARIES[short]["passed"] for short in ["jepa", "dino"])
        if ORACLE_HEADROOM_PASSED:
            TIER_B_CONTROLS_OPENED = True
            for short in ["jepa", "dino"]:
                for seed in ALL_PRECOMMITTED_SEEDS:
                    for variant in ["smooth", "shuffled"]:
                        TIER_B_MODELS[short][seed][variant] = fit_or_load_tier_b(
                            short, variant, seed
                        )
                    counts = {
                        variant: TIER_B_MODELS[short][seed][variant]["config"]["trainable_parameters"]
                        for variant in ["hybrid", "smooth", "shuffled"]
                    }
                    if len(set(counts.values())) != 1:
                        raise RuntimeError(f"Tier B parameter counts differ: {short}/{seed}/{counts}")
        write_json(EVIDENCE_DIR / "tier_b_oracle_headroom.json", ORACLE_SUMMARIES)
        atomic_checkpoint("stage381_tier_b_oracle", {
            "tier_a_promoted": True, "oracle_headroom_passed": ORACLE_HEADROOM_PASSED,
            "matched_controls_opened": TIER_B_CONTROLS_OPENED,
        })
        print(json.dumps({
            "oracle_headroom_passed": ORACLE_HEADROOM_PASSED,
            "matched_controls_opened": TIER_B_CONTROLS_OPENED,
            "models": ORACLE_SUMMARIES,
        }, indent=2))
    except Exception:
        record_failure("stage381_tier_b_oracle_and_controls")
'''


tier_b_decision = r'''# Evaluate the label-free hybrid and conditionally test family risk.
TIER_B_ROWS = []
TIER_B_DECISIONS = {}
TIER_B_PROMOTED = False
RISK_EXTENSION_OPENED = False
RISK_DECISIONS = {}


def family_cvar95(errors, groups):
    values = np.asarray(errors, dtype=np.float64)
    labels = np.asarray(groups)
    family = np.asarray([
        np.mean(values[:, labels == group]) for group in np.unique(labels)
    ])
    return tail_risk_summary(family)["cvar95"]


def summarize_tier_b_model(short):
    data, scale = DEVELOPMENT_DATA[short]["calibration"], PHYSICAL_SCALES[short]
    transition = terminal_transition_mask(data)
    source_terminal = terminal_path_labels(data["source_mode"], data["mask"])
    target_terminal = terminal_path_labels(data["target_mode"], data["mask"])
    matrices = {variant: [] for variant in ["full", "hybrid", "smooth", "shuffled"]}
    event_metrics = []
    parameter_counts = []
    for seed in ALL_PRECOMMITTED_SEEDS:
        matrices["full"].append(score_base_artifact(
            TIER_A_MODELS[short][seed]["full"], data, scale
        )[0])
        rollouts = {}
        for variant in ["hybrid", "smooth", "shuffled"]:
            errors, rollout = score_hybrid_artifact(
                TIER_B_MODELS[short][seed][variant], data, scale
            )
            matrices[variant].append(errors)
            rollouts[variant] = rollout
        event_metrics.append(event_classification_metrics(
            rollouts["hybrid"]["event_probability"], data["event"],
            rollouts["hybrid"]["evaluation_mask"],
        ))
        parameter_counts.append({
            variant: int(TIER_B_MODELS[short][seed][variant]["config"]["trainable_parameters"])
            for variant in ["hybrid", "smooth", "shuffled"]
        })
    matrices = {key: np.stack(value) for key, value in matrices.items()}
    hybrid = matrices["hybrid"]

    def relative(subset, comparator):
        return (np.mean(comparator[:, subset]) - np.mean(hybrid[:, subset])) / max(
            np.mean(comparator[:, subset]), 1e-12
        )

    conditional_gain_full = relative(transition, matrices["full"])
    conditional_gain_smooth = relative(transition, matrices["smooth"])
    conditional_gain_shuffled = relative(transition, matrices["shuffled"])
    hybrid_conditional = tail_risk_summary(hybrid[:, transition].reshape(-1))
    full_conditional = tail_risk_summary(matrices["full"][:, transition].reshape(-1))
    smooth_conditional = tail_risk_summary(matrices["smooth"][:, transition].reshape(-1))
    conditional_cvar_gains = [
        (full_conditional["cvar95"] - hybrid_conditional["cvar95"]) / max(full_conditional["cvar95"], 1e-12),
        (smooth_conditional["cvar95"] - hybrid_conditional["cvar95"]) / max(smooth_conditional["cvar95"], 1e-12),
    ]
    hybrid_tail = tail_risk_summary(hybrid.reshape(-1))
    full_tail = tail_risk_summary(matrices["full"].reshape(-1))
    smooth_tail = tail_risk_summary(matrices["smooth"].reshape(-1))
    p95_gains = [
        (full_tail["p95"] - hybrid_tail["p95"]) / max(full_tail["p95"], 1e-12),
        (smooth_tail["p95"] - hybrid_tail["p95"]) / max(smooth_tail["p95"], 1e-12),
    ]
    mean_degradation = (np.mean(hybrid) - np.mean(matrices["full"])) / max(
        np.mean(matrices["full"]), 1e-12
    )
    mode_catastrophes = {}
    for source, target in sorted(set(zip(source_terminal, target_terminal))):
        selected = (source_terminal == source) & (target_terminal == target)
        mode_catastrophes[f"{source}->{target}"] = float(np.mean(hybrid[:, selected] > 1.0))
    stable = all(
        (np.mean(matrices[comparator][index, transition]) - np.mean(hybrid[index, transition]))
        / max(np.mean(matrices[comparator][index, transition]), 1e-12)
        >= MIN_EVENT_CONDITIONAL_GAIN
        for index in range(3) for comparator in ["full", "smooth"]
    )
    gates = TierBGates(
        oracle_headroom=bool(ORACLE_SUMMARIES[short]["passed"]),
        event_tail_repair=bool(
            min(conditional_gain_full, conditional_gain_smooth) >= MIN_EVENT_CONDITIONAL_GAIN
            and min(conditional_cvar_gains) >= MIN_EVENT_CONDITIONAL_GAIN
        ),
        overall_tail_repair=bool(min(p95_gains) >= MIN_OVERALL_P95_GAIN and mean_degradation <= MAX_OVERALL_MEAN_DEGRADATION),
        catastrophic_control=bool(
            hybrid_tail["catastrophic_rate_gt_1"] <= MAX_CATASTROPHIC_RATE
            and all(value <= MAX_CATASTROPHIC_RATE for value in mode_catastrophes.values())
        ),
        event_identifiability=bool(all(
            row["auroc"] >= MIN_EVENT_AUROC
            and row["brier_skill"] >= MIN_BRIER_SKILL
            and row["ece"] <= MAX_EVENT_ECE
            for row in event_metrics
        )),
        shuffled_supervision_specificity=bool(conditional_gain_shuffled >= MIN_SHUFFLED_GAIN),
        label_free_inference=True,
        seed_and_model_stability=bool(stable),
    )
    decision = sequential_decision(gates, passed_status="tier_b_hybrid_promoted")
    for seed_index, seed in enumerate(ALL_PRECOMMITTED_SEEDS):
        for row in range(len(data["word"])):
            TIER_B_ROWS.append({
                "model": short, "seed": int(seed), "trajectory_id": int(data["group"][row]),
                "record_id": int(data["record_id"][row]), "word": str(data["word"][row]),
                "source_terminal_mode": str(source_terminal[row]),
                "target_terminal_mode": str(target_terminal[row]),
                "full_nmse": float(matrices["full"][seed_index, row]),
                "hybrid_nmse": float(hybrid[seed_index, row]),
                "smooth_nmse": float(matrices["smooth"][seed_index, row]),
                "shuffled_nmse": float(matrices["shuffled"][seed_index, row]),
            })
    return {
        "decision": decision, "conditional_rows": int(np.sum(transition)),
        "conditional_mean_gain_vs_full": float(conditional_gain_full),
        "conditional_mean_gain_vs_smooth": float(conditional_gain_smooth),
        "conditional_mean_gain_vs_shuffled": float(conditional_gain_shuffled),
        "conditional_cvar95_gains": list(map(float, conditional_cvar_gains)),
        "overall_p95_gains": list(map(float, p95_gains)),
        "overall_mean_degradation_vs_full": float(mean_degradation),
        "hybrid_tail": hybrid_tail, "event_metrics_by_seed": event_metrics,
        "mode_pair_catastrophic_rates": mode_catastrophes,
        "parameter_counts": parameter_counts,
    }


if not PIPELINE_FAILED and TIER_B_CONTROLS_OPENED:
    try:
        verify_executed_notebook_through(
            "# Evaluate the label-free hybrid and conditionally test family risk."
        )
        TIER_B_DECISIONS = {short: summarize_tier_b_model(short) for short in ["jepa", "dino"]}
        TIER_B_PROMOTED = all(TIER_B_DECISIONS[short]["decision"]["passed"] for short in ["jepa", "dino"])
        if TIER_B_PROMOTED:
            RISK_EXTENSION_OPENED = True
            for short in ["jepa", "dino"]:
                data, scale = DEVELOPMENT_DATA[short]["calibration"], PHYSICAL_SCALES[short]
                transition = terminal_transition_mask(data)
                no_tail, risk, full, smooth, risk_event_metrics = [], [], [], [], []
                for seed in ALL_PRECOMMITTED_SEEDS:
                    risk_artifact = fit_or_load_tier_b(
                        short, "risk", seed, risk_weight=RISK_LOSS_WEIGHT
                    )
                    no_tail.append(score_hybrid_artifact(
                        TIER_B_MODELS[short][seed]["hybrid"], data, scale
                    )[0])
                    risk_errors, risk_rollout = score_hybrid_artifact(risk_artifact, data, scale)
                    risk.append(risk_errors)
                    full.append(score_base_artifact(
                        TIER_A_MODELS[short][seed]["full"], data, scale
                    )[0])
                    smooth.append(score_hybrid_artifact(
                        TIER_B_MODELS[short][seed]["smooth"], data, scale
                    )[0])
                    risk_event_metrics.append(event_classification_metrics(
                        risk_rollout["event_probability"], data["event"],
                        risk_rollout["evaluation_mask"],
                    ))
                no_tail, risk = np.stack(no_tail), np.stack(risk)
                full, smooth = np.stack(full), np.stack(smooth)
                base_cvar = family_cvar95(no_tail, data["group"])
                risk_cvar = family_cvar95(risk, data["group"])
                gain = (base_cvar - risk_cvar) / max(base_cvar, 1e-12)
                mean_degradation = (np.mean(risk) - np.mean(no_tail)) / max(np.mean(no_tail), 1e-12)
                conditional_gains = [
                    (np.mean(comparator[:, transition]) - np.mean(risk[:, transition]))
                    / max(np.mean(comparator[:, transition]), 1e-12)
                    for comparator in [full, smooth]
                ]
                risk_tail = tail_risk_summary(risk.reshape(-1))
                p95_gains = [
                    (tail_risk_summary(comparator.reshape(-1))["p95"] - risk_tail["p95"])
                    / max(tail_risk_summary(comparator.reshape(-1))["p95"], 1e-12)
                    for comparator in [full, smooth]
                ]
                event_gate_retained = all(
                    row["auroc"] >= MIN_EVENT_AUROC
                    and row["brier_skill"] >= MIN_BRIER_SKILL
                    and row["ece"] <= MAX_EVENT_ECE
                    for row in risk_event_metrics
                )
                RISK_DECISIONS[short] = {
                    "passed": bool(
                        gain >= MIN_RISK_CVAR_GAIN
                        and mean_degradation <= MAX_OVERALL_MEAN_DEGRADATION
                        and min(conditional_gains) >= MIN_EVENT_CONDITIONAL_GAIN
                        and min(p95_gains) >= MIN_OVERALL_P95_GAIN
                        and risk_tail["catastrophic_rate_gt_1"] <= MAX_CATASTROPHIC_RATE
                        and event_gate_retained
                    ),
                    "family_cvar95_gain": float(gain),
                    "mean_degradation": float(mean_degradation),
                    "conditional_mean_gains": list(map(float, conditional_gains)),
                    "overall_p95_gains": list(map(float, p95_gains)),
                    "tail": risk_tail, "event_metrics_by_seed": risk_event_metrics,
                    "alpha": RISK_ALPHA, "outer_weight": RISK_LOSS_WEIGHT,
                }
        write_csv(EVIDENCE_DIR / "tier_b_calibration_rows.csv", TIER_B_ROWS)
        write_json(EVIDENCE_DIR / "tier_b_decisions.json", TIER_B_DECISIONS)
        write_json(EVIDENCE_DIR / "risk_extension_decisions.json", RISK_DECISIONS)
        atomic_checkpoint("stage381_tier_b_decision", {
            "tier_b_promoted": TIER_B_PROMOTED,
            "risk_extension_opened": RISK_EXTENSION_OPENED,
        })
        print(json.dumps({
            "tier_b_promoted": TIER_B_PROMOTED,
            "risk_extension_opened": RISK_EXTENSION_OPENED,
            "decisions": TIER_B_DECISIONS, "risk": RISK_DECISIONS,
        }, indent=2))
    except Exception:
        record_failure("stage381_tier_b_decision")
'''


packaging = r'''# Emit the development decision and compact result bundle.
if PIPELINE_FAILED:
    STATUS = "INCONCLUSIVE_PIPELINE_FAILURE"
elif not TIER_A_PROMOTED:
    STATUS = "tier_a_coefficient_component_claim_rejected"
elif not ORACLE_HEADROOM_PASSED:
    STATUS = "tier_b_oracle_event_hypothesis_rejected"
elif not TIER_B_PROMOTED:
    STATUS = "tier_b_label_free_hybrid_rejected"
elif RISK_EXTENSION_OPENED and all(row.get("passed", False) for row in RISK_DECISIONS.values()):
    STATUS = "stage381_complete_risk_extension_promoted_to_fresh_stage39"
else:
    STATUS = "stage381_structural_hybrid_promoted_to_fresh_stage39"

DECISION_PAYLOAD = {
    "status": STATUS,
    "passed": bool(TIER_A_PROMOTED and TIER_B_PROMOTED),
    "protocol_id": PROTOCOL_ID, "run_signature": RUN_SIGNATURE,
    "source_stage38_run_signature": SOURCE_STAGE38_RUN_SIGNATURE,
    "source_stage38_commit": SOURCE_STAGE38_COMMIT,
    "development_only": True, "stage39_opened": False, "planning_opened": False,
    "evaluation_rows_read": 0, "planning_rows_read": 0,
    "legacy_v1_migration_available": LEGACY_MIGRATION_AVAILABLE,
    "legacy_v1_migration_verified": LEGACY_MIGRATION_VERIFIED,
    "legacy_v1_imported_artifacts": len(LEGACY_IMPORT_ROWS),
    "legacy_v1_calibration_outcomes_imported": 0,
    "tier_a_promoted": TIER_A_PROMOTED,
    "third_seed_executed": THIRD_SEED_EXECUTED,
    "oracle_headroom_passed": ORACLE_HEADROOM_PASSED,
    "tier_b_controls_opened": TIER_B_CONTROLS_OPENED,
    "tier_b_promoted": TIER_B_PROMOTED,
    "risk_extension_opened": RISK_EXTENSION_OPENED,
    "tier_a": TIER_A_FINAL, "oracle": ORACLE_SUMMARIES,
    "tier_b": TIER_B_DECISIONS, "risk": RISK_DECISIONS,
    "claim_boundary": {
        "coefficient_identified_development_result": bool(TIER_A_PROMOTED),
        "event_reset_development_result": bool(TIER_B_PROMOTED),
        "fresh_confirmation_claimed": False, "native_closure_claimed": False,
        "planning_claimed": False, "cross_environment_claimed": False,
        "iclr_ready_claimed": False,
    },
}
write_json(OUT / "stage381_decision.json", DECISION_PAYLOAD)

interpretation = f"""# Automatic Stage 38.1 interpretation

Status: **{STATUS}**

Tier A promoted: **{TIER_A_PROMOTED}**. The precommitted third seed executed:
**{THIRD_SEED_EXECUTED}**. Oracle event headroom passed:
**{ORACLE_HEADROOM_PASSED}**. Label-free Tier B promoted:
**{TIER_B_PROMOTED}**.

V1 construction-only artifacts imported: **{len(LEGACY_IMPORT_ROWS)}**. V1
calibration outcomes imported: **0**.

This is development evidence only. No Stage 38 locked evaluation row or
planning artifact was read. Even a full pass authorizes only a separately
source-bound Stage 39 on fresh trajectories, words, carriers, readouts, model
weights, thresholds, and evaluation certificates.
"""
(OUT / "AUTOMATIC_INTERPRETATION.md").write_text(interpretation)

if not PIPELINE_FAILED:
    (OUT / "FAILURE_TRACE.txt").write_text("NONE\n")
compact = [
    OUT / "config.json", OUT / "source_identity.json", OUT / "FAILURE_TRACE.txt",
    OUT / "stage38_source_development_binding.json", OUT / "development_data_contract.json",
    OUT / "tier_a_decision.json", OUT / "stage381_decision.json",
    OUT / "AUTOMATIC_INTERPRETATION.md",
]
compact.extend(sorted(EVIDENCE_DIR.glob("*.json")))
compact.extend(sorted(EVIDENCE_DIR.glob("*.csv")))
manifest_rows = []
for path in compact:
    if path.is_file():
        manifest_rows.append({
            "path": str(path.relative_to(OUT)), "sha256": sha256_file(path),
            "bytes": int(path.stat().st_size),
        })
write_json(OUT / "result_zip_manifest.json", {"files": manifest_rows})
zip_base = Path("/content") / f"stage381_cmha_result_bundle_{RUN_SIGNATURE[:12]}"
staging = Path("/content") / f"stage381_cmha_bundle_staging_{RUN_SIGNATURE[:12]}"
if staging.is_dir():
    shutil.rmtree(staging)
staging.mkdir(parents=True, exist_ok=False)
for path in compact + [OUT / "result_zip_manifest.json"]:
    if path.is_file():
        target = staging / path.relative_to(OUT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
archive = shutil.make_archive(str(zip_base), "zip", root_dir=staging)
shutil.rmtree(staging)
print(json.dumps(DECISION_PAYLOAD, indent=2))
print(f"Result bundle: {archive}")
try:
    from google.colab import files
    files.download(archive)
except Exception as exc:
    print(f"Automatic download unavailable: {exc}")
'''


protocol_sources = [
    introduction, configuration, installation, setup, analysis_helpers,
    source_binding, data_helpers, tier_a_fit, tier_a_decision,
    tier_b_fit, tier_b_decision, packaging,
]
protocol_sources = [value.strip() for value in protocol_sources]
protocol_digest = hashlib.sha256(
    json.dumps(protocol_sources, ensure_ascii=False, separators=(",", ":")).encode()
).hexdigest()
configuration = configuration.replace("__PROTOCOL_DIGEST__", protocol_digest)
if "__PROTOCOL_DIGEST__" in configuration:
    raise RuntimeError("Stage 38.1 protocol digest placeholder was not replaced")

cells = [
    markdown(introduction), code(configuration), code(installation), code(setup),
    code(analysis_helpers), code(source_binding), code(data_helpers), code(tier_a_fit),
    code(tier_a_decision), code(tier_b_fit), code(tier_b_decision), code(packaging),
]
for index, cell in enumerate(cells):
    cell["id"] = f"stage381-{index:02d}"

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "L4", "provenance": []},
        "kernpec": {"display_name": "Python 3", "name": "python3"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
notebook["metadata"].pop("kernpec")
TARGET.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(f"Wrote {TARGET}")
