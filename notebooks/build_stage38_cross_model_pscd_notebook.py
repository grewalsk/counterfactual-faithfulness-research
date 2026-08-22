"""Build the source-bound Stage 38 cross-model PSCD Colab notebook."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
TARGET = ROOT / "38_cross_model_pscd_confirmation.ipynb"
NUMERICAL = REPOSITORY / "src/cf_faithfulness/stage38_cross_model_pscd.py"

spec = importlib.util.spec_from_file_location(
    "stage37_builder", ROOT / "build_stage37_semigroup_pscd_planning_notebook.py"
)
STAGE37 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(STAGE37)

code = STAGE37.code
markdown = STAGE37.markdown
replace_assignment = STAGE37.replace_assignment
replace_block = STAGE37.replace_block
assigned_uppercase_names = STAGE37.assigned_uppercase_names
function_sources = STAGE37.function_sources


def rename(value: str) -> str:
    for old, new in [
        ("Stage 37", "Stage 38"), ("STAGE37", "STAGE38"),
        ("stage37", "stage38"),
    ]:
        value = value.replace(old, new)
    return value


introduction = r'''# Stage 38: cross-model predictive-state closure confirmation

## Frozen decision before computation

Stage 37.1 calibrated the measuring instrument on exact PushT Markov state.
Its horizon-matched 256-state mixture achieved locked physical NMSE 0.072,
semigroup NMSE 0.079, and passed every registered length and physical-mode
gate.  Stage 38 now tests the actual method claim: whether a post-hoc state can
make representations from frozen joint-embedding world models recursively
usable.

The experiment is prospective and standalone.  It uses fresh trajectory IDs
and exact action words, all of length 9--12, in construction, selection,
calibration, and locked closure panels.  The official frozen JEPA-WM PushT
checkpoint is the primary model; the official frozen DINO-WM PushT checkpoint
is an independent representation replication.  Neither checkpoint is updated.

For each representation, the construction-only grounded readout and carrier
projection are frozen before model selection.  One semigroup strength is
selected on disjoint words.  Final models are then fit with three independent
optimization seeds and identical architecture, data, initialization within
each seed, and epoch budget:

1. one-step prediction only;
2. free-running PSCD with no semigroup term;
3. a latent-only multi-step overshooting control; and
4. full S-PSCD with latent, carrier, and grounded multi-anchor consistency.

Locked closure requires absolute physical accuracy, improvement over native
model rollout, direct-versus-composed consistency, correct-history specificity,
three-seed stability, and registered horizon, mode, and tail-risk bounds.
Open-loop planning remains sealed unless both JEPA and DINO closure decisions
pass.  If opened, all methods rank the same fixed candidate bank against goals
chosen before simulator endpoint costs are read.

## Claim boundary

A closure pass supports post-hoc cross-representation PSCD closure on PushT.
A planning pass additionally supports finite-bank open-loop value.  This is not
cross-environment evidence, native closure inside either checkpoint, a causal
mechanism claim, closed-loop planning, or a proof of minimal state.
'''


configuration = rename(STAGE37.configuration)
for name, value in {
    "EXPERIMENT_SOURCE_REF": '"codex/stage34-predictive-fiber-abstraction"',
    "PROTOCOL_ID": '"stage38-cross-model-pscd-confirmation-v1"',
    "NOTEBOOK_PROTOCOL_SHA256": '"__PROTOCOL_DIGEST__"',
    "EVIDENCE_STATUS": '"FRESH_PROSPECTIVE_CROSS_MODEL_CONFIRMATION"',
    "EXPERIMENT_NOTEBOOK_PATH": '"notebooks/38_cross_model_pscd_confirmation.ipynb"',
    "EXPERIMENT_BUILDER_PATH": '"notebooks/build_stage38_cross_model_pscd_notebook.py"',
    "EXPERIMENT_NUMERICAL_PATH": '"src/cf_faithfulness/stage38_cross_model_pscd.py"',
    "OUTPUT_DIR": '"/content/counterfactual_faithfulness_stage38_xmpscd"',
    "DRIVE_OUTPUT_DIR": '"/content/drive/MyDrive/counterfactual_faithfulness_stage38_xmpscd"',
    "RUN_REQUEST_PATH": '"/content/drive/MyDrive/counterfactual_faithfulness_stage38_xmpscd/stage38_run_request.json"',
    "MAX_ESTIMATED_TOTAL_MINUTES": "720.0",
    "SEED": "380101", "DESIGN_SEED": "380141", "DECODER_SEED": "380183",
    "RANK_SEED": "380213", "CALIBRATION_SEED": "380253",
    "BOOTSTRAP_SEED": "380283", "CONTROL_SEED": "380351",
    "MODEL_NAMES": '["jepa_wm_pusht", "dino_wm_pusht"]',
    "MODEL_SHORT_NAMES": '{"jepa_wm_pusht": "jepa", "dino_wm_pusht": "dino"}',
    "CONSTRUCTION_TRAJECTORY_POOL": "list(range(56000, 58000))",
    "MODEL_SELECTION_TRAJECTORY_POOL": "list(range(58000, 60000))",
    "CALIBRATION_TRAJECTORY_POOL": "list(range(60000, 62000))",
    "EVALUATION_TRAJECTORY_POOL": "list(range(62000, 66000))",
    "CONSTRUCTION_TRAJECTORIES": "24", "MODEL_SELECTION_TRAJECTORIES": "16",
    "CALIBRATION_TRAJECTORIES": "24", "EVALUATION_TRAJECTORIES": "32",
    "TASK_ID_OFFSET": "380000",
}.items():
    configuration = replace_assignment(configuration, name, value)

configuration = replace_block(
    configuration,
    "CONSTRUCTION_WORD_NAMES = [",
    "CALIBRATION_INTERCHANGE_PAIRS =",
    r'''CANONICAL_RESPONSE_WORD_NAMES = ["A", "B", "C", "D", "AB", "CD", "BA", "DC"]
CONSTRUCTION_WORD_NAMES = [
    "CDDCDABCB", "DCAABBCBC", "DDDBABCAAA", "AACAADBBBC",
    "BDDADDCBDBC", "DADDCBAACAB", "DBDCABBDDCBC", "BCBBABCBADAC",
]
MODEL_SELECTION_WORD_NAMES = [
    "CABADBBCC", "CDBABBDCB", "DACCAACBBB", "DDDCDADDAB",
    "CADCDCBDBCC", "CBBDCACBCAC", "BDBCCDACCDAA", "ACADCDBBAAAA",
]
CALIBRATION_WORD_NAMES = [
    "DCAAABACD", "CAACBCDCD", "ADDDBACBBC", "AADABCBADC",
    "CDDCABBDAAC", "DABDBACCDBB", "CDBCDACABBBA", "BBDABDADBDBC",
]
STAGE38_CORE_WORD_NAMES = sorted(
    set(
        CANONICAL_RESPONSE_WORD_NAMES + CONSTRUCTION_WORD_NAMES
        + MODEL_SELECTION_WORD_NAMES + CALIBRATION_WORD_NAMES
    ),
    key=lambda value: (len(value), value),
)
CORE_WORD_SPECS = [stage38_word_spec(name) for name in STAGE38_CORE_WORD_NAMES]
''',
)
configuration = replace_block(
    configuration,
    "CLOSURE_EVALUATION_WORD_NAMES = [",
    "EVALUATION_INTERCHANGE_PAIRS =",
    r'''CLOSURE_EVALUATION_WORD_NAMES = [
    "ADAABCDCB", "DDCCDAABD", "BDDBCDCACC", "BBCBDAACAD",
    "BDACBAACBDB", "BADBBABDABC", "ACACADCDACBD", "ADCBBDABDDAD",
]
PLANNING_WORD_NAMES = [
    "ADABAAACDC", "BAACBDBDCA", "BADAACBDAD", "BBDDBBABBC",
    "ADAAABDDDC", "BDADBDCDAA", "BDDBABDBCD", "CACADAACAB",
    "ABDDDBABCC", "DACBADAABB", "BCCABACADB", "DDBBAAACDA",
]
EVALUATION_WORD_NAMES_REGISTERED = sorted(
    set(CLOSURE_EVALUATION_WORD_NAMES + PLANNING_WORD_NAMES),
    key=lambda value: (len(value), value),
)
EVALUATION_WORD_SPECS = [
    stage38_word_spec(name) for name in EVALUATION_WORD_NAMES_REGISTERED
]
''',
)
configuration = configuration.split("\n\nV2_PREFLIGHT_HELPER_ORDER_AMENDMENT =", 1)[0]
configuration = configuration.replace(
    'assert {len(row["angles"]) for row in CORE_WORD_SPECS} == set(range(1, 9))',
    'assert {len(row["angles"]) for row in CORE_WORD_SPECS} == {1, 2, 9, 10, 11, 12}',
)
configuration = re.sub(
    r"PINNED = \[.*?\]\n\nassert INTERVENTION_BLOCK",
    '''PINNED = [
    "official_frozen_jepa_and_dino_pusht_checkpoints",
    "fresh_trajectory_ids_56000_to_65999",
    "four_disjoint_horizon_matched_word_banks",
    "three_independent_final_training_seeds",
    "fixed_256_state_mixture_and_four_carrier_history",
    "one_step_free_running_latent_overshoot_and_full_semigroup_controls",
    "locked_post_contact_and_tail_risk_gates",
    "planning_opened_only_after_both_closure_pass",
    "source_bound_hash_validated_resume", "no_synthetic_fallback",
    "not_cross_environment", "not_native_closure", "not_causal",
]

assert INTERVENTION_BLOCK''',
    configuration,
    count=1,
    flags=re.S,
)
configuration += r'''

MAX_CARRIER_PROJECTION_DIM = 256
FIXED_CARRIER_DIM = 256
FIXED_HISTORY_LENGTH = 4
FIXED_LATENT_DIM = 256
FIXED_DYNAMICS = "mixture"
SEMIGROUP_HORIZONS = [2, 4, 8]
SEMIGROUP_WEIGHTS = [0.5, 1.0, 2.0] if RUN_MODE == "pilot" else [0.5]
FINAL_TRAINING_SEEDS = [3801, 3802, 3803] if RUN_MODE == "pilot" else [3801, 3802]
SELECTION_EPOCHS = 120
FINAL_EPOCHS = 320
SIMULATOR_PREFLIGHT_EPOCHS = 180
SIMULATOR_FINAL_EPOCHS = 360
ACTIVE_SELECTION_EPOCHS = SELECTION_EPOCHS if RUN_MODE == "pilot" else 4
ACTIVE_FINAL_EPOCHS = FINAL_EPOCHS if RUN_MODE == "pilot" else 6
ACTIVE_SIMULATOR_PREFLIGHT_EPOCHS = SIMULATOR_PREFLIGHT_EPOCHS if RUN_MODE == "pilot" else 4
ACTIVE_SIMULATOR_FINAL_EPOCHS = SIMULATOR_FINAL_EPOCHS if RUN_MODE == "pilot" else 6
PSCD_LEARNING_RATE = 1e-3
FULL_SEMIGROUP_COMPONENT_WEIGHTS = [0.35, 0.20, 0.45]
OVERSHOOT_COMPONENT_WEIGHTS = [0.0, 0.0, 1.0]

MAX_SIMULATOR_PREFLIGHT_NMSE = 0.30
MAX_SIMULATOR_LOCKED_NMSE = 0.25
MIN_SIMULATOR_GAIN = 0.50
MIN_NATIVE_FIDELITY_GAIN = 0.10
MAX_RECURSIVE_PHYSICAL_NMSE = 0.25
MIN_REPAIR_ADVANTAGE = 0.05
MIN_SEMIGROUP_ADVANTAGE = 0.05
MIN_RECURSION_ADVANTAGE = 0.05
MIN_HISTORY_ADVANTAGE = 0.05
MAX_OVERSHOOT_PHYSICAL_RATIO = 1.05
MAX_OVERSHOOT_SEMIGROUP_RATIO = 1.10
MAX_LENGTH_PHYSICAL_NMSE = 0.35
MAX_MODE_PHYSICAL_NMSE = 0.40
MAX_P95_PHYSICAL_NMSE = 0.35
MAX_CATASTROPHIC_RATE = 0.02
MIN_PLANNING_REGRET_REDUCTION = 0.02
PLANNING_DIMENSIONS = [2, 3, 4, 5]

if RUN_MODE == "smoke":
    ACTIVE_CONSTRUCTION_TRAJECTORIES = 2
    ACTIVE_MODEL_SELECTION_TRAJECTORIES = 2
    ACTIVE_CALIBRATION_TRAJECTORIES = 2
    ACTIVE_EVALUATION_TRAJECTORIES = 2

TASK_WORD_BANKS = [
    CONSTRUCTION_WORD_NAMES, MODEL_SELECTION_WORD_NAMES,
    CALIBRATION_WORD_NAMES, CLOSURE_EVALUATION_WORD_NAMES,
]
assert all(len(bank) == 8 for bank in TASK_WORD_BANKS)
assert all({len(word) for word in bank} == {9, 10, 11, 12} for bank in TASK_WORD_BANKS)
assert all(
    set(TASK_WORD_BANKS[left]).isdisjoint(TASK_WORD_BANKS[right])
    for left in range(4) for right in range(left + 1, 4)
)
assert set(PLANNING_WORD_NAMES).isdisjoint(set().union(*map(set, TASK_WORD_BANKS)))
assert len(PLANNING_WORD_NAMES) == 12
assert {len(word) for word in PLANNING_WORD_NAMES} == {10}
assert set().union(*(set(word) for bank in TASK_WORD_BANKS for word in bank)) <= set(STAGE38_TOKEN_SPECS)
assert set().union(*map(set, PLANNING_WORD_NAMES)) <= set(STAGE38_TOKEN_SPECS)
'''
configuration += "\n\nPROTOCOL_CONFIG_KEYS = " + repr(
    assigned_uppercase_names(configuration)
) + "\n"


installation = STAGE37.installation
setup = rename(STAGE37.setup).replace("stage38_spscd", "stage38_xmpscd")

stage38_helpers = [
    "fit_weighted_semigroup_predictive_state_closure",
    "select_stage38_semigroup_candidate", "tail_risk_summary",
    "hierarchical_seed_trajectory_interval", "Stage38ModelGates", "Stage38Gates",
    "derive_stage38_model_decision", "derive_stage38_decision",
]
analysis_helpers = STAGE37.analysis_helpers + "\n\n" + function_sources(
    NUMERICAL.read_text(), stage38_helpers
)
analysis_helpers = analysis_helpers.replace(
    "class Stage38ModelGates:\n", "@dataclass(frozen=True)\nclass Stage38ModelGates:\n"
).replace(
    "class Stage38Gates:\n", "@dataclass(frozen=True)\nclass Stage38Gates:\n"
)

model_helpers = rename(STAGE37.model_helpers)
design_and_runtime_helpers = rename(STAGE37.design_and_runtime_helpers)
design_and_runtime_helpers = design_and_runtime_helpers.replace(
    "np.cross(diagnostic[i], diagnostic[j])",
    "diagnostic[i, 0] * diagnostic[j, 1] - diagnostic[i, 1] * diagnostic[j, 0]",
)
physical_truth = rename(STAGE37.physical_truth)


simulator_preflight = r'''# Reconfirm the horizon-matched true-state operator before loading either checkpoint.
SIMULATOR_PREFLIGHT_PASSED = False
SIMULATOR_PREFLIGHT_ARTIFACT = None


def load_stage38_physical_sequences(split):
    split_names = {
        "construction": CONSTRUCTION_WORD_NAMES,
        "model_selection": MODEL_SELECTION_WORD_NAMES,
        "calibration": CALIBRATION_WORD_NAMES,
        "evaluation": EVALUATION_WORD_NAMES,
    }
    names = list(split_names[str(split)])
    rows = {key: [] for key in [
        "initial", "actions", "targets", "mask", "word", "length",
        "group", "record_id", "initial_mode",
    ]}
    for record in SELECTED_RECORDS[str(split)]:
        with np.load(truth_path(record), allow_pickle=False) as payload:
            lookup = {str(value): index for index, value in enumerate(payload["word_names"])}
            for name in names:
                length = int(WORD_BY_NAME[name]["length"])
                actions, _ = word_actions(record, WORD_BY_NAME[name])
                chunks = actions.reshape(length, FRAMESKIP, 2).mean(axis=1)
                padded = np.zeros((MAX_WORD_LENGTH, 3), dtype=np.float64)
                padded[:length, :2] = chunks
                padded[:length, 2] = np.linalg.norm(chunks, axis=1)
                target = np.zeros((MAX_WORD_LENGTH, len(GROUNDED_OBSERVABLES)))
                target[:length] = payload["path_observables"][lookup[name], :length]
                valid = np.zeros(MAX_WORD_LENGTH, dtype=bool)
                valid[:length] = True
                rows["initial"].append(grounded_observables(record["state"]))
                rows["actions"].append(padded)
                rows["targets"].append(target)
                rows["mask"].append(valid)
                rows["word"].append(name)
                rows["length"].append(length)
                rows["group"].append(int(record["trajectory_id"]))
                rows["record_id"].append(int(record["record_id"]))
                rows["initial_mode"].append(str(record["mode"]))
    for key in ["initial", "actions", "targets"]:
        rows[key] = np.asarray(rows[key], dtype=np.float64)
    rows["mask"] = np.asarray(rows["mask"], dtype=bool)
    rows["word"] = np.asarray(rows["word"]).astype(str)
    rows["initial_mode"] = np.asarray(rows["initial_mode"]).astype(str)
    for key in ["length", "group", "record_id"]:
        rows[key] = np.asarray(rows[key], dtype=np.int64)
    return rows


def stage38_physical_score(artifact, data):
    result = rollout_predictive_state_closure(
        artifact, data["initial"], data["actions"], data["targets"], data["mask"]
    )
    valid = result["evaluation_mask"]
    error = scaled_path_mse(
        result["physical"], data["targets"], valid,
        artifact["normalization"]["physical_scale"], final_only=False,
    )
    persistence = np.repeat(data["initial"][:, None, :], MAX_WORD_LENGTH, axis=1)
    persistence_error = scaled_path_mse(
        persistence, data["targets"], valid,
        artifact["normalization"]["physical_scale"], final_only=False,
    )
    return {
        "physical_nmse": float(np.mean(error)),
        "persistence_nmse": float(np.mean(persistence_error)),
        "gain": float(np.mean(relative_gain(error, persistence_error))),
    }


if not PIPELINE_FAILED:
    try:
        verify_executed_notebook_through(
            "# Reconfirm the horizon-matched true-state operator before loading either checkpoint."
        )
        train = load_stage38_physical_sequences("construction")
        validation = load_stage38_physical_sequences("model_selection")
        SIMULATOR_PREFLIGHT_ARTIFACT = fit_weighted_semigroup_predictive_state_closure(
            train["initial"], train["actions"], train["targets"], train["targets"],
            train["mask"], history_length=1, latent_dim=FIXED_LATENT_DIM,
            dynamics=FIXED_DYNAMICS, epochs=ACTIVE_SIMULATOR_PREFLIGHT_EPOCHS,
            learning_rate=PSCD_LEARNING_RATE,
            seed=stable_seed(CONTROL_SEED, "simulator_preflight"),
            semigroup_horizons=SEMIGROUP_HORIZONS, semigroup_weight=1.0,
            semigroup_component_weights=FULL_SEMIGROUP_COMPONENT_WEIGHTS,
        )
        scores = stage38_physical_score(SIMULATOR_PREFLIGHT_ARTIFACT, validation)
        SIMULATOR_PREFLIGHT_PASSED = bool(
            scores["physical_nmse"] <= MAX_SIMULATOR_PREFLIGHT_NMSE
            and scores["gain"] >= MIN_SIMULATOR_GAIN
        )
        write_json(OUT / "simulator_preflight.json", {
            "passed": SIMULATOR_PREFLIGHT_PASSED, "scores": scores,
            "jepa_loaded": False, "dino_loaded": False,
        })
        atomic_checkpoint("stage38_simulator_preflight", {
            "passed": SIMULATOR_PREFLIGHT_PASSED, "scores": scores,
        })
        print(json.dumps({"simulator_preflight": scores}, indent=2))
    except Exception:
        record_failure("stage38_simulator_preflight")
'''


construction_and_paths = r'''# Freeze both grounded readouts and materialize non-evaluation carrier paths.
def stage38_names_for_split(split):
    table = {
        "construction": CONSTRUCTION_WORD_NAMES,
        "model_selection": MODEL_SELECTION_WORD_NAMES,
        "calibration": CALIBRATION_WORD_NAMES,
        "evaluation": EVALUATION_WORD_NAMES,
        "evaluation_closure": CLOSURE_EVALUATION_WORD_NAMES,
        "evaluation_planning": PLANNING_WORD_NAMES,
    }
    if str(split) not in table:
        raise ValueError(f"unknown Stage 38 split {split!r}")
    return list(table[str(split)])


def stage38_feature_tensor(outputs, names):
    tensor = np.zeros(
        (len(names), MAX_WORD_LENGTH, VISUAL_SKETCH_DIM + PROPRIO_PAD_DIM),
        dtype=np.float32,
    )
    widths = []
    for word_index, name in enumerate(names):
        visual, proprio = (np.asarray(value) for value in outputs[name])
        length = int(WORD_BY_NAME[name]["length"])
        if visual.shape != (length, EXPECTED_VISUAL_TOKENS, EXPECTED_VISUAL_WIDTH):
            raise RuntimeError(f"{name} visual output shape changed: {visual.shape}")
        if proprio.ndim != 3 or proprio.shape[:2] != (length, EXPECTED_PROPRIO_TOKENS):
            raise RuntimeError(f"{name} proprio output shape changed: {proprio.shape}")
        if not np.all(np.isfinite(visual)) or not np.all(np.isfinite(proprio)):
            raise RuntimeError(f"{name} output is nonfinite")
        for step in range(length):
            tensor[word_index, step, :VISUAL_SKETCH_DIM] = count_sketch(
                visual[step : step + 1], VISUAL_SKETCH_DIM,
                stable_seed(DECODER_SEED, "visual_sketch"),
            )[0]
            pooled = pool_spatial_proprio_features(
                proprio[step], expected_tokens=EXPECTED_PROPRIO_TOKENS,
                max_width=PROPRIO_PAD_DIM,
            )
            tensor[word_index, step, VISUAL_SKETCH_DIM : VISUAL_SKETCH_DIM + len(pooled)] = pooled
            widths.append(len(pooled))
    if len(set(widths)) != 1:
        raise RuntimeError("pooled proprio width changed within a model")
    return tensor, int(widths[0])


def stage38_output_preflight(bundle):
    name = CONSTRUCTION_WORD_NAMES[0]
    length = int(WORD_BY_NAME[name]["length"])
    record = SELECTED_RECORDS["construction"][0]
    outputs, traces = grouped_model_words(bundle, record, [name])
    visual, proprio = outputs[name]
    expected = {
        "visual": (length, EXPECTED_VISUAL_TOKENS, EXPECTED_VISUAL_WIDTH),
        "proprio": (
            length, EXPECTED_PROPRIO_TOKENS,
            EXPECTED_PROPRIO_FEATURE_WIDTHS[bundle["name"]],
        ),
        "carrier": (length, EXPECTED_VISUAL_TOKENS, bundle["carrier_width"]),
    }
    observed = {
        "visual": tuple(np.asarray(visual).shape),
        "proprio": tuple(np.asarray(proprio).shape),
        "carrier": tuple(np.asarray(traces[name]).shape),
    }
    if observed != expected:
        raise RuntimeError(f"{bundle['name']} output contract changed: {observed}")
    _, pooled_width = stage38_feature_tensor(outputs, [name])
    contract = {
        "model": bundle["name"], "model_short": bundle["short"],
        "observed": {key: list(value) for key, value in observed.items()},
        "pooled_proprio_width": pooled_width, "all_outputs_finite": True,
        "evaluation_rows_used": 0,
    }
    write_json(OUT / f"model_output_contract_{bundle['short']}.json", contract)
    PROVENANCE_COUNTS["model_output_contract_preflights"][bundle["short"]] += 1
    return contract


def stage38_response_rows(tensor, names):
    rows, metadata = [], []
    for word_index, name in enumerate(names):
        for step in range(int(WORD_BY_NAME[name]["length"])):
            rows.append(tensor[word_index, step])
            metadata.append((name, step))
    return np.asarray(rows, dtype=np.float64), metadata


def stage38_truth_rows(record, names):
    with np.load(truth_path(record), allow_pickle=False) as truth:
        lookup = {str(name): index for index, name in enumerate(truth["word_names"])}
        rows, metadata = [], []
        for name in names:
            index = lookup[name]
            for step in range(int(WORD_BY_NAME[name]["length"])):
                rows.append(truth["path_observables"][index, step])
                metadata.append((name, step))
    return np.asarray(rows, dtype=np.float64), metadata


def stage38_carrier_projection(value, bundle):
    carrier = np.asarray(value, dtype=np.float32)
    expected = (EXPECTED_VISUAL_TOKENS, EXPECTED_CARRIER_WIDTHS[bundle["name"]])
    if carrier.shape != expected:
        raise RuntimeError(f"{bundle['name']} carrier shape changed: {carrier.shape}")
    return count_sketch(
        carrier.reshape(1, -1), FIXED_CARRIER_DIM,
        stable_seed(CONTROL_SEED, "stage38_carrier_projection", bundle["short"]),
    )[0].astype(np.float32)


def stage38_mode_paths(record, contact_counts, length):
    contacts = np.asarray(contact_counts, dtype=np.int64)[: int(length) * FRAMESKIP]
    source = [str(record["mode"])]
    ever_contact = str(record["mode"]) in {"contact", "post_contact"}
    for step in range(1, int(length)):
        previous = contacts[(step - 1) * FRAMESKIP : step * FRAMESKIP]
        future = contacts[step * FRAMESKIP : (step + 1) * FRAMESKIP]
        previous_any, future_any = bool(np.any(previous > 0)), bool(np.any(future > 0))
        ever_contact = ever_contact or previous_any
        if previous_any:
            label = "contact"
        elif not ever_contact and future_any:
            label = "pre_contact"
        elif ever_contact:
            label = "post_contact"
        else:
            label = "free"
        source.append(label)
    final = contacts[(int(length) - 1) * FRAMESKIP : int(length) * FRAMESKIP]
    prior = bool(np.any(contacts[: (int(length) - 1) * FRAMESKIP] > 0))
    terminal = "contact" if np.any(final > 0) else (
        "post_contact" if prior or str(record["mode"]) in {"contact", "post_contact"} else "free"
    )
    return source, source[1:] + [terminal]


def stage38_decoder_paths(short):
    return (
        SUBSPACE_DIR / f"decoder_{short}.npz",
        SUBSPACE_DIR / f"decoder_manifest_{short}.json",
    )


def fit_stage38_decoder(bundle):
    names = stage38_names_for_split("construction")
    feature_rows, target_rows, groups = [], [], []
    for index, record in enumerate(SELECTED_RECORDS["construction"]):
        outputs, _ = grouped_model_words(bundle, record, names)
        tensor, _ = stage38_feature_tensor(outputs, names)
        features, feature_meta = stage38_response_rows(tensor, names)
        targets, target_meta = stage38_truth_rows(record, names)
        if feature_meta != target_meta:
            raise RuntimeError("Stage 38 decoder row order changed")
        feature_rows.append(features)
        target_rows.append(targets)
        groups.extend([int(record["trajectory_id"])] * len(features))
        write_json(OUT / f"construction_{bundle['short']}_progress.json", {
            "completed": index + 1, "total": len(SELECTED_RECORDS["construction"]),
            "last_record_id": int(record["record_id"]),
        })
    return fit_grouped_ridge(
        np.concatenate(feature_rows), np.concatenate(target_rows),
        np.asarray(groups, dtype=np.int64), penalties=DECODER_RIDGES,
        folds=min(4, len(set(groups))),
        seed=stable_seed(DECODER_SEED, "stage38_decoder", bundle["short"]),
    )


def save_stage38_decoder(short, decoder):
    path, manifest_path = stage38_decoder_paths(short)
    atomic_npz(
        path, weight=np.asarray(decoder["weight"]), intercept=np.asarray(decoder["intercept"]),
        penalty=np.asarray(decoder["penalty"]), oof_mse=np.asarray(decoder["oof_mse"]),
    )
    write_digest_sidecar(path)
    write_json(manifest_path, {
        "model": short, "decoder_sha256": sha256_file(path),
        "training_split": "construction", "evaluation_rows_used": 0,
    })
    write_digest_sidecar(manifest_path)


def load_stage38_decoder(short):
    path, manifest_path = stage38_decoder_paths(short)
    validate_digest_sidecar(path)
    validate_digest_sidecar(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest["decoder_sha256"] != sha256_file(path):
        raise RuntimeError("Stage 38 decoder manifest mismatch")
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def stage38_path(short, record, split):
    return PATH_DIR / f"{short}_{split}_{int(record['record_id'])}.npz"


def generate_stage38_path_record(bundle, record, split, decoder):
    short = bundle["short"]
    path = stage38_path(short, record, split)
    asset = f"{bundle['name']}.pth.tar"
    identity = (
        f"{PROTOCOL_ID}:{RUN_SIGNATURE}:{short}:{record['record_id']}:{split}:"
        f"horizon-matched-carrier-path-v1:{EXPECTED_PRETRAINED_ASSET_SHA256[asset]}"
    )
    required = {
        "identity", "word_names", "word_lengths", "initial_carrier", "initial_physical",
        "actions", "carrier_paths", "native_grounded_paths", "simulator_grounded_paths",
        "path_mask", "source_modes", "target_modes",
    }
    if validate_npz_shard(path, required, identity):
        PROVENANCE_COUNTS["validated_cache_hits"] += 1
        return path
    words = stage38_names_for_split(split)
    inference_names = sorted(set(words) | {ZERO_WORD_NAMES[1]}, key=lambda x: (len(x), x))
    outputs, traces = grouped_model_words(bundle, record, inference_names)
    tensor, _ = stage38_feature_tensor(outputs, inference_names)
    grounded = tensor.astype(np.float64) @ decoder["weight"] + decoder["intercept"]
    output_lookup = {name: index for index, name in enumerate(inference_names)}
    with np.load(truth_path(record), allow_pickle=False) as truth:
        truth_lookup = {str(name): index for index, name in enumerate(truth["word_names"])}
        actions = np.zeros((len(words), MAX_WORD_LENGTH, 3), dtype=np.float64)
        carriers = np.zeros((len(words), MAX_WORD_LENGTH, FIXED_CARRIER_DIM), dtype=np.float32)
        native = np.zeros((len(words), MAX_WORD_LENGTH, len(GROUNDED_OBSERVABLES)), dtype=np.float32)
        simulator = np.zeros_like(native)
        mask = np.zeros((len(words), MAX_WORD_LENGTH), dtype=bool)
        source_modes = np.full((len(words), MAX_WORD_LENGTH), "", dtype="<U16")
        target_modes = np.full((len(words), MAX_WORD_LENGTH), "", dtype="<U16")
        for word_index, name in enumerate(words):
            length = int(WORD_BY_NAME[name]["length"])
            model_index, truth_index = output_lookup[name], truth_lookup[name]
            macro_actions, _ = word_actions(record, WORD_BY_NAME[name])
            chunks = macro_actions.reshape(length, FRAMESKIP, 2).mean(axis=1)
            actions[word_index, :length, :2] = chunks
            actions[word_index, :length, 2] = np.linalg.norm(chunks, axis=1)
            carriers[word_index, :length] = np.stack([
                stage38_carrier_projection(value, bundle) for value in traces[name]
            ])
            native[word_index, :length] = grounded[model_index, :length]
            simulator[word_index, :length] = truth["path_observables"][truth_index, :length]
            mask[word_index, :length] = True
            source, target = stage38_mode_paths(record, truth["contact_counts"][truth_index], length)
            source_modes[word_index, :length] = source
            target_modes[word_index, :length] = target
    initial = stage38_carrier_projection(traces[ZERO_WORD_NAMES[1]][0], bundle)
    atomic_npz(
        path, identity=np.asarray(identity), word_names=np.asarray(words),
        word_lengths=np.asarray([len(name) for name in words], dtype=np.int64),
        initial_carrier=initial, initial_physical=grounded_observables(record["state"]),
        actions=actions.astype(np.float32), carrier_paths=carriers,
        native_grounded_paths=native, simulator_grounded_paths=simulator,
        path_mask=mask, source_modes=source_modes, target_modes=target_modes,
    )
    PROVENANCE_COUNTS["model_record_forwards"][short] += 1
    return path


DECODERS = {}
if not PIPELINE_FAILED and SIMULATOR_PREFLIGHT_PASSED:
    try:
        verify_executed_notebook_through(
            "# Freeze both grounded readouts and materialize non-evaluation carrier paths."
        )
        for model_name in MODEL_NAMES:
            bundle = load_world_model(model_name)
            short = bundle["short"]
            try:
                stage38_output_preflight(bundle)
                decoder_path, manifest_path = stage38_decoder_paths(short)
                if decoder_path.is_file() and manifest_path.is_file():
                    DECODERS[short] = load_stage38_decoder(short)
                    PROVENANCE_COUNTS["validated_cache_hits"] += 1
                else:
                    DECODERS[short] = fit_stage38_decoder(bundle)
                    save_stage38_decoder(short, DECODERS[short])
                for split in ["construction", "model_selection", "calibration"]:
                    for index, record in enumerate(SELECTED_RECORDS[split]):
                        generate_stage38_path_record(bundle, record, split, DECODERS[short])
                        write_json(OUT / f"model_{short}_{split}_progress.json", {
                            "completed": index + 1, "total": len(SELECTED_RECORDS[split]),
                            "last_record_id": int(record["record_id"]),
                        })
            finally:
                unload_world_model(bundle)
            memory_report(f"stage38_{short}_non_evaluation_paths_complete")
        atomic_checkpoint("stage38_non_evaluation_paths_complete", {
            "models": list(MODEL_NAMES), "evaluation_paths_materialized": False,
            "decoder_hashes": {
                short: sha256_file(stage38_decoder_paths(short)[0]) for short in DECODERS
            },
        })
    except Exception:
        record_failure("stage38_decoders_or_non_evaluation_paths")
'''


data_and_selection = r'''# Select semigroup strength independently for each representation.
def load_stage38_sequences(short, split, selected_names=None):
    rows = {key: [] for key in [
        "initial_carrier", "initial_physical", "actions", "carrier", "native",
        "simulator", "mask", "source_mode", "target_mode", "word", "length",
        "group", "record_id", "initial_mode",
    ]}
    wanted = None if selected_names is None else set(map(str, selected_names))
    record_split = "evaluation" if str(split).startswith("evaluation_") else str(split)
    for record in SELECTED_RECORDS[record_split]:
        with np.load(stage38_path(short, record, split), allow_pickle=False) as payload:
            words = [str(value) for value in payload["word_names"]]
            indices = [index for index, word in enumerate(words) if wanted is None or word in wanted]
            rows["initial_carrier"].extend(np.repeat(payload["initial_carrier"][None], len(indices), axis=0))
            rows["initial_physical"].extend(np.repeat(payload["initial_physical"][None], len(indices), axis=0))
            mapping = {
                "actions": "actions", "carrier": "carrier_paths",
                "native": "native_grounded_paths", "simulator": "simulator_grounded_paths",
                "mask": "path_mask", "source_mode": "source_modes", "target_mode": "target_modes",
            }
            for key, payload_key in mapping.items():
                rows[key].extend(payload[payload_key][indices])
            rows["word"].extend([words[index] for index in indices])
            rows["length"].extend(payload["word_lengths"][indices].astype(int).tolist())
            rows["group"].extend([int(record["trajectory_id"])] * len(indices))
            rows["record_id"].extend([int(record["record_id"])] * len(indices))
            rows["initial_mode"].extend([str(record["mode"])] * len(indices))
    for key in ["initial_carrier", "initial_physical", "actions", "carrier", "native", "simulator"]:
        rows[key] = np.asarray(rows[key], dtype=np.float64)
    rows["mask"] = np.asarray(rows["mask"], dtype=bool)
    for key in ["source_mode", "target_mode", "word", "initial_mode"]:
        rows[key] = np.asarray(rows[key]).astype(str)
    for key in ["length", "group", "record_id"]:
        rows[key] = np.asarray(rows[key], dtype=np.int64)
    if not all(len(value) == len(rows["word"]) for value in rows.values()):
        raise RuntimeError(f"Stage 38 {short}/{split} arrays are misaligned")
    return rows


def concatenate_stage38_sequences(*bundles):
    return {
        key: np.concatenate([bundle[key] for bundle in bundles], axis=0)
        for key in bundles[0]
    }


def stage38_validation_scores(artifact, data):
    result = rollout_predictive_state_closure(
        artifact, data["initial_carrier"], data["actions"], data["carrier"], data["mask"]
    )
    valid = result["evaluation_mask"]
    physical = scaled_path_mse(
        result["physical"], data["native"], valid,
        artifact["normalization"]["physical_scale"], final_only=False,
    )
    direct = result["direct_state"][valid]
    state_scale = np.maximum(np.std(direct, axis=0, ddof=1), 1e-6)
    semigroup = scaled_path_mse(
        result["state"], result["direct_state"], valid, state_scale, final_only=False,
    )
    return {
        "physical_nmse": float(np.mean(physical)),
        "semigroup_nmse": float(np.mean(semigroup)),
        "score": float(np.mean(physical) + 0.5 * np.mean(semigroup)),
    }


SELECTED_SEMIGROUP = {}
SELECTION_ROWS = []
if not PIPELINE_FAILED and SIMULATOR_PREFLIGHT_PASSED:
    try:
        verify_executed_notebook_through(
            "# Select semigroup strength independently for each representation."
        )
        for short in sorted(DECODERS):
            train = load_stage38_sequences(short, "construction")
            validation = load_stage38_sequences(short, "model_selection")
            rows = []
            for weight in SEMIGROUP_WEIGHTS:
                artifact = fit_weighted_semigroup_predictive_state_closure(
                    train["initial_carrier"], train["actions"], train["carrier"],
                    train["native"], train["mask"], history_length=FIXED_HISTORY_LENGTH,
                    latent_dim=FIXED_LATENT_DIM, dynamics=FIXED_DYNAMICS,
                    epochs=ACTIVE_SELECTION_EPOCHS, learning_rate=PSCD_LEARNING_RATE,
                    seed=stable_seed(CALIBRATION_SEED, "selection", short),
                    semigroup_horizons=SEMIGROUP_HORIZONS, semigroup_weight=weight,
                    semigroup_component_weights=FULL_SEMIGROUP_COMPONENT_WEIGHTS,
                )
                row = {
                    "model": short, "semigroup_weight": float(weight),
                    "loss_initial": artifact["loss_initial"], "loss_final": artifact["loss_final"],
                    **stage38_validation_scores(artifact, validation),
                }
                rows.append(row)
                SELECTION_ROWS.append(row)
                del artifact
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            SELECTED_SEMIGROUP[short] = select_stage38_semigroup_candidate(rows)
        write_csv(EVIDENCE_DIR / "stage38_semigroup_selection_rows.csv", SELECTION_ROWS)
        selection_path = CALIBRATION_MODEL_DIR / "frozen_stage38_selection.json"
        write_json(selection_path, {
            "protocol_id": PROTOCOL_ID, "selected": SELECTED_SEMIGROUP,
            "candidate_rows": SELECTION_ROWS, "evaluation_rows_used": 0,
        })
        write_digest_sidecar(selection_path)
        atomic_checkpoint("stage38_semigroup_selection_complete", {
            "selection_sha256": sha256_file(selection_path), "selected": SELECTED_SEMIGROUP,
        })
        print(json.dumps({"selected_semigroup": SELECTED_SEMIGROUP}, indent=2))
    except Exception:
        record_failure("stage38_semigroup_selection")
'''


calibration = r'''# Freeze all matched-seed models and scales before opening evaluation.
FROZEN_MODELS = {}
STATE_SCALES = {}
PHYSICAL_SCALES = {}
CARRIER_SCALES = {}
SIMULATOR_FINAL = None
EVALUATION_OPENED = False


def stage38_artifact_paths(short, variant, seed):
    stem = CALIBRATION_MODEL_DIR / f"stage38_{short}_{variant}_seed{int(seed)}"
    return Path(str(stem) + ".npz"), Path(str(stem) + "_schema.json")


def stage38_encode_artifact(value, arrays, prefix="root"):
    if isinstance(value, np.ndarray):
        key = f"array_{len(arrays):05d}"
        arrays[key] = value
        return {"kind": "array", "key": key}
    if isinstance(value, dict):
        return {
            "kind": "dict",
            "items": {
                str(key): stage38_encode_artifact(item, arrays, f"{prefix}.{key}")
                for key, item in sorted(value.items())
            },
        }
    if isinstance(value, (list, tuple)):
        return {
            "kind": "list",
            "items": [
                stage38_encode_artifact(item, arrays, f"{prefix}.{index}")
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return {"kind": "scalar", "value": value}
    raise TypeError(f"unsupported Stage 38 artifact value at {prefix}: {type(value)}")


def stage38_decode_artifact(node, arrays):
    kind = str(node["kind"])
    if kind == "array":
        return np.asarray(arrays[str(node["key"])])
    if kind == "dict":
        return {
            key: stage38_decode_artifact(value, arrays)
            for key, value in node["items"].items()
        }
    if kind == "list":
        return [stage38_decode_artifact(value, arrays) for value in node["items"]]
    if kind == "scalar":
        return node["value"]
    raise ValueError(f"unknown Stage 38 artifact node {kind!r}")


def save_stage38_artifact(short, variant, seed, artifact):
    array_path, schema_path = stage38_artifact_paths(short, variant, seed)
    arrays = {}
    schema = stage38_encode_artifact(artifact, arrays)
    atomic_npz(array_path, **arrays)
    write_json(schema_path, {
        "protocol_id": PROTOCOL_ID, "run_signature": RUN_SIGNATURE,
        "model": str(short), "variant": str(variant), "seed": int(seed),
        "array_sha256": sha256_file(array_path), "schema": schema,
    })
    write_digest_sidecar(schema_path)
    return array_path, schema_path


def load_stage38_artifact(short, variant, seed):
    array_path, schema_path = stage38_artifact_paths(short, variant, seed)
    validate_digest_sidecar(array_path)
    validate_digest_sidecar(schema_path)
    metadata = json.loads(schema_path.read_text())
    expected = (PROTOCOL_ID, RUN_SIGNATURE, str(short), str(variant), int(seed))
    observed = (
        metadata["protocol_id"], metadata["run_signature"], metadata["model"],
        metadata["variant"], int(metadata["seed"]),
    )
    if observed != expected or metadata["array_sha256"] != sha256_file(array_path):
        raise RuntimeError(f"Stage 38 frozen artifact binding failed for {short}/{variant}/{seed}")
    with np.load(array_path, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    return stage38_decode_artifact(metadata["schema"], arrays)


def stage38_variant_configuration(short, variant):
    selected = float(SELECTED_SEMIGROUP[short]["semigroup_weight"])
    table = {
        "one_step": {
            "free_weight": 0.0, "semigroup_weight": 0.0,
            "semigroup_component_weights": FULL_SEMIGROUP_COMPONENT_WEIGHTS,
        },
        "pscd": {
            "free_weight": 1.0, "semigroup_weight": 0.0,
            "semigroup_component_weights": FULL_SEMIGROUP_COMPONENT_WEIGHTS,
        },
        "overshoot": {
            "free_weight": 1.0, "semigroup_weight": selected,
            "semigroup_component_weights": OVERSHOOT_COMPONENT_WEIGHTS,
        },
        "spscd": {
            "free_weight": 1.0, "semigroup_weight": selected,
            "semigroup_component_weights": FULL_SEMIGROUP_COMPONENT_WEIGHTS,
        },
    }
    if str(variant) not in table:
        raise KeyError(f"unknown Stage 38 variant {variant!r}")
    return table[str(variant)]


def fit_or_load_stage38_model(short, variant, seed, data):
    array_path, schema_path = stage38_artifact_paths(short, variant, seed)
    sidecars = [Path(str(array_path) + ".sha256"), Path(str(schema_path) + ".sha256")]
    if array_path.is_file() and schema_path.is_file() and all(path.is_file() for path in sidecars):
        PROVENANCE_COUNTS["validated_cache_hits"] += 1
        return load_stage38_artifact(short, variant, seed)
    objective = stage38_variant_configuration(short, variant)
    artifact = fit_weighted_semigroup_predictive_state_closure(
        data["initial_carrier"], data["actions"], data["carrier"],
        data["native"], data["mask"], history_length=FIXED_HISTORY_LENGTH,
        latent_dim=FIXED_LATENT_DIM, dynamics=FIXED_DYNAMICS,
        epochs=ACTIVE_FINAL_EPOCHS, learning_rate=PSCD_LEARNING_RATE,
        seed=int(seed), semigroup_horizons=SEMIGROUP_HORIZONS, **objective,
    )
    save_stage38_artifact(short, variant, seed, artifact)
    return artifact


if not PIPELINE_FAILED and SIMULATOR_PREFLIGHT_PASSED:
    try:
        verify_executed_notebook_through(
            "# Freeze all matched-seed models and scales before opening evaluation."
        )
        if set(SELECTED_SEMIGROUP) != {"jepa", "dino"}:
            raise RuntimeError("both Stage 38 semigroup selections must be frozen")
        model_file_manifest = []
        for short in ["jepa", "dino"]:
            construction = load_stage38_sequences(short, "construction")
            calibration_only = load_stage38_sequences(short, "calibration")
            data = concatenate_stage38_sequences(construction, calibration_only)
            valid = data["mask"]
            PHYSICAL_SCALES[short] = np.maximum(
                np.std(data["simulator"][valid], axis=0, ddof=1), 1e-8
            )
            CARRIER_SCALES[short] = np.maximum(
                np.std(data["carrier"][valid], axis=0, ddof=1), 1e-8
            )
            FROZEN_MODELS[short] = {}
            STATE_SCALES[short] = {}
            for variant in ["one_step", "pscd", "overshoot", "spscd"]:
                FROZEN_MODELS[short][variant] = {}
                STATE_SCALES[short][variant] = {}
                for seed in FINAL_TRAINING_SEEDS:
                    artifact = fit_or_load_stage38_model(short, variant, seed, data)
                    expected = stage38_variant_configuration(short, variant)
                    if (
                        int(artifact["config"]["seed"]) != int(seed)
                        or float(artifact["config"]["free_weight"]) != float(expected["free_weight"])
                        or float(artifact["config"]["semigroup_weight"]) != float(expected["semigroup_weight"])
                        or list(artifact["config"]["semigroup_component_weights"])
                            != list(expected["semigroup_component_weights"])
                    ):
                        raise RuntimeError(f"matched-control binding failed for {short}/{variant}/{seed}")
                    FROZEN_MODELS[short][variant][int(seed)] = artifact
                    calibration_rollout = rollout_predictive_state_closure(
                        artifact, data["initial_carrier"], data["actions"],
                        data["carrier"], data["mask"],
                    )
                    evaluated = calibration_rollout["evaluation_mask"]
                    STATE_SCALES[short][variant][int(seed)] = np.maximum(
                        np.std(calibration_rollout["direct_state"][evaluated], axis=0, ddof=1),
                        1e-8,
                    )
                    array_path, schema_path = stage38_artifact_paths(short, variant, seed)
                    model_file_manifest.append({
                        "model": short, "variant": variant, "seed": int(seed),
                        "array_sha256": sha256_file(array_path),
                        "schema_sha256": sha256_file(schema_path),
                    })
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        physical = concatenate_stage38_sequences(
            load_stage38_physical_sequences("construction"),
            load_stage38_physical_sequences("calibration"),
        )
        simulator_seed = stable_seed(CONTROL_SEED, "stage38_simulator_final")
        simulator_array, simulator_schema = stage38_artifact_paths(
            "simulator", "exact_state", simulator_seed
        )
        simulator_sidecars = [
            Path(str(simulator_array) + ".sha256"), Path(str(simulator_schema) + ".sha256")
        ]
        if (
            simulator_array.is_file() and simulator_schema.is_file()
            and all(path.is_file() for path in simulator_sidecars)
        ):
            SIMULATOR_FINAL = load_stage38_artifact(
                "simulator", "exact_state", simulator_seed
            )
            PROVENANCE_COUNTS["validated_cache_hits"] += 1
        else:
            SIMULATOR_FINAL = fit_weighted_semigroup_predictive_state_closure(
                physical["initial"], physical["actions"], physical["targets"],
                physical["targets"], physical["mask"], history_length=1,
                latent_dim=FIXED_LATENT_DIM, dynamics=FIXED_DYNAMICS,
                epochs=ACTIVE_SIMULATOR_FINAL_EPOCHS,
                learning_rate=PSCD_LEARNING_RATE, seed=simulator_seed,
                semigroup_horizons=SEMIGROUP_HORIZONS, semigroup_weight=1.0,
                semigroup_component_weights=FULL_SEMIGROUP_COMPONENT_WEIGHTS,
            )
            save_stage38_artifact(
                "simulator", "exact_state", simulator_seed, SIMULATOR_FINAL
            )
        model_file_manifest.append({
            "model": "simulator", "variant": "exact_state", "seed": int(simulator_seed),
            "array_sha256": sha256_file(simulator_array),
            "schema_sha256": sha256_file(simulator_schema),
        })

        scale_path = CALIBRATION_MODEL_DIR / "stage38_frozen_scales.npz"
        scale_arrays = {}
        for short in ["jepa", "dino"]:
            scale_arrays[f"physical_{short}"] = PHYSICAL_SCALES[short]
            scale_arrays[f"carrier_{short}"] = CARRIER_SCALES[short]
            for variant in ["one_step", "pscd", "overshoot", "spscd"]:
                for seed in FINAL_TRAINING_SEEDS:
                    scale_arrays[f"state_{short}_{variant}_{int(seed)}"] = (
                        STATE_SCALES[short][variant][int(seed)]
                    )
        atomic_npz(scale_path, **scale_arrays)
        certificate_path = CALIBRATION_MODEL_DIR / "evaluation_open_certificate.json"
        write_json(certificate_path, {
            "protocol_id": PROTOCOL_ID, "run_signature": RUN_SIGNATURE,
            "source_commit": SOURCE_IDENTITY.get("resolved_commit"),
            "selection_sha256": sha256_file(
                CALIBRATION_MODEL_DIR / "frozen_stage38_selection.json"
            ),
            "scale_sha256": sha256_file(scale_path),
            "models": model_file_manifest,
            "final_training_seeds": list(map(int, FINAL_TRAINING_SEEDS)),
            "matched_seed_across_variants": True,
            "evaluation_statistics_read": False,
            "evaluation_metrics_computed": False,
            "planning_opened": False,
            "checkpoint_parameters_updated": False,
        })
        write_digest_sidecar(certificate_path)
        atomic_checkpoint("stage38_models_frozen", {
            "certificate_sha256": sha256_file(certificate_path),
            "frozen_model_count": len(model_file_manifest),
            "evaluation_opened": EVALUATION_OPENED,
        })
        print(json.dumps({
            "selected_semigroup": SELECTED_SEMIGROUP,
            "final_training_seeds": FINAL_TRAINING_SEEDS,
            "frozen_model_count": len(model_file_manifest),
            "evaluation_opened": EVALUATION_OPENED,
        }, indent=2))
    except Exception:
        record_failure("stage38_calibration_model_freeze")
'''


locked_evaluation = r'''# Open closure first; open planning only after both representation panels pass.
DECISION_PAYLOAD = {
    "status": (
        "INCONCLUSIVE_PIPELINE_FAILURE" if PIPELINE_FAILED
        else "operator_class_failed_positive_control_preflight"
        if not SIMULATOR_PREFLIGHT_PASSED
        else "INCONCLUSIVE_PIPELINE_FAILURE"
    ),
    "passed": False,
}
EVALUATION_ROWS = []
PLANNING_ROWS = []
SUMMARY = {}
MODEL_DECISIONS = {}
PLANNING_OPENED = False


def subset_stage38(data, selected):
    selected = np.asarray(selected, dtype=bool)
    return {key: np.asarray(value)[selected] for key, value in data.items()}


def stage38_terminal_labels(labels, mask):
    values = np.asarray(labels).astype(str)
    valid = np.asarray(mask, dtype=bool)
    indices = np.max(
        np.where(valid, np.arange(valid.shape[1])[None, :], -1), axis=1
    )
    if np.any(indices < 0):
        raise ValueError("every Stage 38 label path needs an endpoint")
    return values[np.arange(len(values)), indices]


def stage38_hierarchical_ci(matrix, groups, label):
    return hierarchical_seed_trajectory_interval(
        matrix, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
        seed=stable_seed(BOOTSTRAP_SEED, "stage38", label),
    )


def stage38_model_closure_panel(short, closure):
    mask = closure["mask"]
    groups = closure["group"]
    scale = PHYSICAL_SCALES[short]
    physical_persistence = np.repeat(
        closure["initial_physical"][:, None, :], MAX_WORD_LENGTH, axis=1
    )
    base_evaluated = rollout_evaluation_mask(mask, FIXED_HISTORY_LENGTH)
    native_error = scaled_path_mse(
        closure["native"], closure["simulator"], base_evaluated, scale
    )
    persistence_error = scaled_path_mse(
        physical_persistence, closure["simulator"], base_evaluated, scale
    )
    native_gain = relative_gain(native_error, persistence_error)
    native_ci = clustered_mean_interval(
        native_gain, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
        seed=stable_seed(BOOTSTRAP_SEED, "stage38_native", short),
    )
    physical_by_variant = {
        variant: [] for variant in ["one_step", "pscd", "overshoot", "spscd"]
    }
    semigroup_by_variant = {
        variant: [] for variant in ["one_step", "pscd", "overshoot", "spscd"]
    }
    false_history_errors = []
    seed_rows = []
    native_history = history_tensor(
        closure["initial_carrier"], closure["carrier"], mask, FIXED_HISTORY_LENGTH
    )
    terminal_modes = stage38_terminal_labels(closure["target_mode"], mask)
    for seed in FINAL_TRAINING_SEEDS:
        false_history = permute_past_history(
            native_history, groups, mask,
            seed=stable_seed(CONTROL_SEED, "stage38_false_history", short, int(seed)),
        )
        seed_rollouts = {}
        for variant in ["one_step", "pscd", "overshoot", "spscd"]:
            artifact = FROZEN_MODELS[short][variant][int(seed)]
            rollout = rollout_predictive_state_closure(
                artifact, closure["initial_carrier"], closure["actions"],
                closure["carrier"], mask,
            )
            seed_rollouts[variant] = rollout
            evaluated = rollout["evaluation_mask"]
            physical_by_variant[variant].append(scaled_path_mse(
                rollout["physical"], closure["simulator"], evaluated, scale
            ))
            semigroup_by_variant[variant].append(scaled_path_mse(
                rollout["state"], rollout["direct_state"], evaluated,
                STATE_SCALES[short][variant][int(seed)], final_only=False,
            ))
        spscd_artifact = FROZEN_MODELS[short]["spscd"][int(seed)]
        false_rollout = rollout_predictive_state_closure(
            spscd_artifact, closure["initial_carrier"], closure["actions"],
            closure["carrier"], mask, histories_override=false_history,
        )
        false_history_errors.append(scaled_path_mse(
            false_rollout["physical"], closure["simulator"],
            false_rollout["evaluation_mask"], scale,
        ))
    physical_by_variant = {
        key: np.stack(value, axis=0) for key, value in physical_by_variant.items()
    }
    semigroup_by_variant = {
        key: np.stack(value, axis=0) for key, value in semigroup_by_variant.items()
    }
    false_history_errors = np.stack(false_history_errors, axis=0)
    native_matrix = np.repeat(native_error[None, :], len(FINAL_TRAINING_SEEDS), axis=0)
    repair_gain = relative_gain(
        physical_by_variant["spscd"].reshape(-1), native_matrix.reshape(-1)
    ).reshape(native_matrix.shape)
    semigroup_gain = relative_gain(
        semigroup_by_variant["spscd"].reshape(-1),
        semigroup_by_variant["pscd"].reshape(-1),
    ).reshape(native_matrix.shape)
    recursion_gain = relative_gain(
        physical_by_variant["spscd"].reshape(-1),
        physical_by_variant["one_step"].reshape(-1),
    ).reshape(native_matrix.shape)
    history_gain = relative_gain(
        physical_by_variant["spscd"].reshape(-1), false_history_errors.reshape(-1)
    ).reshape(native_matrix.shape)
    intervals = {
        "repair": stage38_hierarchical_ci(repair_gain, groups, f"repair_{short}"),
        "semigroup": stage38_hierarchical_ci(
            semigroup_gain, groups, f"semigroup_{short}"
        ),
        "recursion": stage38_hierarchical_ci(
            recursion_gain, groups, f"recursion_{short}"
        ),
        "history": stage38_hierarchical_ci(history_gain, groups, f"history_{short}"),
    }
    seed_summaries = []
    for seed_index, seed in enumerate(FINAL_TRAINING_SEEDS):
        primary_errors = physical_by_variant["spscd"][seed_index]
        tail = tail_risk_summary(primary_errors)
        length_means = {
            str(length): float(np.mean(primary_errors[closure["length"] == length]))
            for length in sorted(set(closure["length"].tolist()))
        }
        initial_mode_means = {
            mode: float(np.mean(primary_errors[closure["initial_mode"] == mode]))
            for mode in sorted(set(closure["initial_mode"].tolist()))
        }
        terminal_mode_means = {
            mode: float(np.mean(primary_errors[terminal_modes == mode]))
            for mode in sorted(set(terminal_modes.tolist()))
        }
        contact_tail = np.isin(terminal_modes, ["contact", "post_contact"])
        if not np.any(contact_tail):
            raise RuntimeError("Stage 38 locked panel contains no contact/post-contact endpoints")
        overshoot_physical_ratio = float(
            np.mean(primary_errors)
            / max(np.mean(physical_by_variant["overshoot"][seed_index]), 1e-12)
        )
        overshoot_semigroup_ratio = float(
            np.mean(semigroup_by_variant["spscd"][seed_index])
            / max(np.mean(semigroup_by_variant["overshoot"][seed_index]), 1e-12)
        )
        seed_summaries.append({
            "seed": int(seed),
            "spscd_physical_nmse": float(np.mean(primary_errors)),
            "spscd_semigroup_nmse": float(
                np.mean(semigroup_by_variant["spscd"][seed_index])
            ),
            "repair_advantage": float(np.mean(repair_gain[seed_index])),
            "semigroup_advantage": float(np.mean(semigroup_gain[seed_index])),
            "recursion_advantage": float(np.mean(recursion_gain[seed_index])),
            "history_advantage": float(np.mean(history_gain[seed_index])),
            "overshoot_physical_ratio": overshoot_physical_ratio,
            "overshoot_semigroup_ratio": overshoot_semigroup_ratio,
            "length_physical_nmse": length_means,
            "initial_mode_physical_nmse": initial_mode_means,
            "terminal_mode_physical_nmse": terminal_mode_means,
            "contact_post_contact_physical_nmse": float(np.mean(primary_errors[contact_tail])),
            "tail": tail,
        })
        for row_index in range(len(closure["word"])):
            EVALUATION_ROWS.append({
                "model": short, "seed": int(seed),
                "record_id": int(closure["record_id"][row_index]),
                "trajectory_id": int(groups[row_index]),
                "initial_mode": str(closure["initial_mode"][row_index]),
                "terminal_mode": str(terminal_modes[row_index]),
                "word": str(closure["word"][row_index]),
                "word_length": int(closure["length"][row_index]),
                "native_physical_nmse": float(native_error[row_index]),
                "one_step_physical_nmse": float(
                    physical_by_variant["one_step"][seed_index, row_index]
                ),
                "pscd_physical_nmse": float(
                    physical_by_variant["pscd"][seed_index, row_index]
                ),
                "overshoot_physical_nmse": float(
                    physical_by_variant["overshoot"][seed_index, row_index]
                ),
                "spscd_physical_nmse": float(primary_errors[row_index]),
                "wrong_history_physical_nmse": float(false_history_errors[seed_index, row_index]),
                "pscd_semigroup_nmse": float(
                    semigroup_by_variant["pscd"][seed_index, row_index]
                ),
                "overshoot_semigroup_nmse": float(
                    semigroup_by_variant["overshoot"][seed_index, row_index]
                ),
                "spscd_semigroup_nmse": float(
                    semigroup_by_variant["spscd"][seed_index, row_index]
                ),
            })
    all_lengths_pass = all(
        value <= MAX_LENGTH_PHYSICAL_NMSE
        for row in seed_summaries for value in row["length_physical_nmse"].values()
    )
    all_modes_pass = all(
        value <= MAX_MODE_PHYSICAL_NMSE
        for row in seed_summaries
        for value in list(row["initial_mode_physical_nmse"].values())
            + list(row["terminal_mode_physical_nmse"].values())
    )
    tails_pass = all(
        row["tail"]["p95"] <= MAX_P95_PHYSICAL_NMSE
        and row["tail"]["catastrophic_rate_gt_1"] <= MAX_CATASTROPHIC_RATE
        and row["contact_post_contact_physical_nmse"] <= MAX_MODE_PHYSICAL_NMSE
        for row in seed_summaries
    )
    seed_consistency = all(
        row["spscd_physical_nmse"] <= MAX_RECURSIVE_PHYSICAL_NMSE
        and row["repair_advantage"] > 0
        and row["semigroup_advantage"] > 0
        and row["recursion_advantage"] > 0
        and row["history_advantage"] > 0
        for row in seed_summaries
    )
    model_gates = Stage38ModelGates(
        native_fidelity=bool(
            np.mean(native_gain) >= MIN_NATIVE_FIDELITY_GAIN and native_ci[0] > 0
        ),
        absolute_recursive_closure=bool(
            all(row["spscd_physical_nmse"] <= MAX_RECURSIVE_PHYSICAL_NMSE
                for row in seed_summaries)
        ),
        repair_advantage=bool(
            np.mean(repair_gain) >= MIN_REPAIR_ADVANTAGE and intervals["repair"][0] > 0
        ),
        semigroup_specificity=bool(
            np.mean(semigroup_gain) >= MIN_SEMIGROUP_ADVANTAGE
            and intervals["semigroup"][0] > 0
        ),
        overshooting_noninferiority=bool(all(
            row["overshoot_physical_ratio"] <= MAX_OVERSHOOT_PHYSICAL_RATIO
            and row["overshoot_semigroup_ratio"] <= MAX_OVERSHOOT_SEMIGROUP_RATIO
            for row in seed_summaries
        )),
        recursion_and_history_specificity=bool(
            np.mean(recursion_gain) >= MIN_RECURSION_ADVANTAGE
            and intervals["recursion"][0] > 0
            and np.mean(history_gain) >= MIN_HISTORY_ADVANTAGE
            and intervals["history"][0] > 0
        ),
        seed_consistency=bool(seed_consistency),
        horizon_mode_tail_consistency=bool(all_lengths_pass and all_modes_pass and tails_pass),
    )
    decision = derive_stage38_model_decision(model_gates)
    metrics = {
        "native_physical_nmse": float(np.mean(native_error)),
        "native_fidelity_gain": float(np.mean(native_gain)),
        "native_fidelity_gain_ci95": native_ci,
        "one_step_physical_nmse": float(np.mean(physical_by_variant["one_step"])),
        "pscd_physical_nmse": float(np.mean(physical_by_variant["pscd"])),
        "overshoot_physical_nmse": float(np.mean(physical_by_variant["overshoot"])),
        "spscd_physical_nmse": float(np.mean(physical_by_variant["spscd"])),
        "pscd_semigroup_nmse": float(np.mean(semigroup_by_variant["pscd"])),
        "overshoot_semigroup_nmse": float(np.mean(semigroup_by_variant["overshoot"])),
        "spscd_semigroup_nmse": float(np.mean(semigroup_by_variant["spscd"])),
        "repair_advantage": float(np.mean(repair_gain)),
        "semigroup_advantage": float(np.mean(semigroup_gain)),
        "recursion_advantage": float(np.mean(recursion_gain)),
        "history_advantage": float(np.mean(history_gain)),
        "intervals": intervals, "seeds": seed_summaries,
    }
    return {"decision": decision, "metrics": metrics}


def stage38_planning_panel(short, planning):
    scale = PHYSICAL_SCALES[short]
    truth_end = terminal_values(planning["simulator"], planning["mask"])
    native_end = terminal_values(planning["native"], planning["mask"])
    goals = np.zeros_like(truth_end)
    goal_words = {}
    for record_id in np.unique(planning["record_id"]):
        rows = np.flatnonzero(planning["record_id"] == record_id)
        target_word = PLANNING_WORD_NAMES[
            stable_seed(DESIGN_SEED, "planning_goal", int(record_id))
            % len(PLANNING_WORD_NAMES)
        ]
        target_rows = rows[planning["word"][rows] == target_word]
        if len(target_rows) != 1:
            raise RuntimeError("Stage 38 planning goal is not unique within a record")
        goals[rows] = truth_end[target_rows[0]]
        goal_words[int(record_id)] = target_word
    truth_cost = goal_cost(truth_end, goals, scale, PLANNING_DIMENSIONS)
    native_cost = goal_cost(native_end, goals, scale, PLANNING_DIMENSIONS)
    native_metrics = grouped_planner_metrics(
        native_cost, truth_cost, planning["record_id"]
    )
    variant_metrics = {
        variant: [] for variant in ["one_step", "pscd", "overshoot", "spscd"]
    }
    for seed in FINAL_TRAINING_SEEDS:
        for variant in variant_metrics:
            rollout = rollout_predictive_state_from_initial(
                FROZEN_MODELS[short][variant][int(seed)],
                planning["initial_carrier"], planning["actions"], planning["mask"],
            )
            endpoint = terminal_values(rollout["physical"], rollout["evaluation_mask"])
            cost = goal_cost(endpoint, goals, scale, PLANNING_DIMENSIONS)
            variant_metrics[variant].append(grouped_planner_metrics(
                cost, truth_cost, planning["record_id"]
            ))
    metric_arrays = {}
    for variant, rows in variant_metrics.items():
        metric_arrays[variant] = {
            metric: np.stack([row[metric] for row in rows], axis=0)
            for metric in ["regret", "success", "pairwise_accuracy"]
        }
    native_regret = np.repeat(
        native_metrics["regret"][None, :], len(FINAL_TRAINING_SEEDS), axis=0
    )
    native_reduction = native_regret - metric_arrays["spscd"]["regret"]
    pscd_reduction = (
        metric_arrays["pscd"]["regret"] - metric_arrays["spscd"]["regret"]
    )
    planning_clusters = np.asarray([
        planning["group"][np.flatnonzero(planning["record_id"] == record_id)[0]]
        for record_id in native_metrics["groups"]
    ])
    native_ci = stage38_hierarchical_ci(
        native_reduction, planning_clusters, f"planning_native_{short}"
    )
    pscd_ci = stage38_hierarchical_ci(
        pscd_reduction, planning_clusters, f"planning_pscd_{short}"
    )
    passed = bool(
        np.mean(native_reduction) >= MIN_PLANNING_REGRET_REDUCTION
        and np.mean(pscd_reduction) >= MIN_PLANNING_REGRET_REDUCTION
        and native_ci[0] > 0 and pscd_ci[0] > 0
        and np.mean(metric_arrays["spscd"]["pairwise_accuracy"])
            >= np.mean(native_metrics["pairwise_accuracy"])
        and np.mean(metric_arrays["spscd"]["pairwise_accuracy"])
            >= np.mean(metric_arrays["pscd"]["pairwise_accuracy"])
    )
    for seed_index, seed in enumerate(FINAL_TRAINING_SEEDS):
        for group_index, record_id in enumerate(native_metrics["groups"]):
            PLANNING_ROWS.append({
                "model": short, "seed": int(seed), "record_id": int(record_id),
                "goal_word": goal_words[int(record_id)],
                "native_regret": float(native_metrics["regret"][group_index]),
                "one_step_regret": float(
                    metric_arrays["one_step"]["regret"][seed_index, group_index]
                ),
                "pscd_regret": float(
                    metric_arrays["pscd"]["regret"][seed_index, group_index]
                ),
                "overshoot_regret": float(
                    metric_arrays["overshoot"]["regret"][seed_index, group_index]
                ),
                "spscd_regret": float(
                    metric_arrays["spscd"]["regret"][seed_index, group_index]
                ),
                "native_pairwise_accuracy": float(
                    native_metrics["pairwise_accuracy"][group_index]
                ),
                "spscd_pairwise_accuracy": float(
                    metric_arrays["spscd"]["pairwise_accuracy"][seed_index, group_index]
                ),
            })
    return {
        "passed": passed,
        "native_regret": float(np.mean(native_metrics["regret"])),
        "one_step_regret": float(np.mean(metric_arrays["one_step"]["regret"])),
        "pscd_regret": float(np.mean(metric_arrays["pscd"]["regret"])),
        "overshoot_regret": float(np.mean(metric_arrays["overshoot"]["regret"])),
        "spscd_regret": float(np.mean(metric_arrays["spscd"]["regret"])),
        "native_regret_reduction": float(np.mean(native_reduction)),
        "pscd_regret_reduction": float(np.mean(pscd_reduction)),
        "native_regret_reduction_ci95": native_ci,
        "pscd_regret_reduction_ci95": pscd_ci,
        "native_pairwise_accuracy": float(np.mean(native_metrics["pairwise_accuracy"])),
        "spscd_pairwise_accuracy": float(
            np.mean(metric_arrays["spscd"]["pairwise_accuracy"])
        ),
    }


if not PIPELINE_FAILED and not SIMULATOR_PREFLIGHT_PASSED:
    DECISION_PAYLOAD.update({
        "protocol_id": PROTOCOL_ID, "run_signature": RUN_SIGNATURE,
        "first_failed_gate": "simulator_positive_control",
        "evaluation_opened": False, "planning_opened": False,
        "checkpoint_models_loaded": False,
        "scientific_failure_not_pipeline_error": True,
    })
    write_json(OUT / "stage38_decision.json", DECISION_PAYLOAD)


if not PIPELINE_FAILED and SIMULATOR_PREFLIGHT_PASSED:
    try:
        verify_executed_notebook_through(
            "# Open closure first; open planning only after both representation panels pass."
        )
        certificate_path = CALIBRATION_MODEL_DIR / "evaluation_open_certificate.json"
        validate_digest_sidecar(certificate_path)
        certificate = json.loads(certificate_path.read_text())
        if (
            certificate["protocol_id"] != PROTOCOL_ID
            or certificate["run_signature"] != RUN_SIGNATURE
            or certificate["evaluation_statistics_read"]
            or certificate["planning_opened"]
        ):
            raise RuntimeError("Stage 38 evaluation-open certificate is invalid")

        # Only closure words are sent through the checkpoints at this point.
        for model_name in MODEL_NAMES:
            bundle = load_world_model(model_name)
            short = bundle["short"]
            try:
                for index, record in enumerate(SELECTED_RECORDS["evaluation"]):
                    generate_stage38_path_record(
                        bundle, record, "evaluation_closure", DECODERS[short]
                    )
                    write_json(OUT / f"model_{short}_closure_progress.json", {
                        "completed": index + 1,
                        "total": len(SELECTED_RECORDS["evaluation"]),
                        "last_record_id": int(record["record_id"]),
                    })
            finally:
                unload_world_model(bundle)
        EVALUATION_OPENED = True

        closure_data = {
            short: load_stage38_sequences(short, "evaluation_closure")
            for short in ["jepa", "dino"]
        }
        reference_closure = closure_data["jepa"]
        simulator_rollout = rollout_predictive_state_closure(
            SIMULATOR_FINAL, reference_closure["initial_physical"],
            reference_closure["actions"], reference_closure["simulator"],
            reference_closure["mask"],
        )
        simulator_error = scaled_path_mse(
            simulator_rollout["physical"], reference_closure["simulator"],
            simulator_rollout["evaluation_mask"], PHYSICAL_SCALES["jepa"],
        )
        simulator_persistence = np.repeat(
            reference_closure["initial_physical"][:, None, :], MAX_WORD_LENGTH, axis=1
        )
        simulator_persistence_error = scaled_path_mse(
            simulator_persistence, reference_closure["simulator"],
            simulator_rollout["evaluation_mask"], PHYSICAL_SCALES["jepa"],
        )
        simulator_gain = relative_gain(simulator_error, simulator_persistence_error)
        simulator_ci = clustered_mean_interval(
            simulator_gain, reference_closure["group"], draws=ACTIVE_BOOTSTRAP_DRAWS,
            seed=stable_seed(BOOTSTRAP_SEED, "stage38_simulator_locked"),
        )
        simulator_gate = bool(
            np.mean(simulator_error) <= MAX_SIMULATOR_LOCKED_NMSE
            and np.mean(simulator_gain) >= MIN_SIMULATOR_GAIN
            and simulator_ci[0] > 0
        )

        closure_results = {
            short: stage38_model_closure_panel(short, closure_data[short])
            for short in ["jepa", "dino"]
        }
        MODEL_DECISIONS = {
            short: closure_results[short]["decision"] for short in ["jepa", "dino"]
        }
        source_gate = bool(
            SOURCE_IDENTITY.get("confirmation_eligible", False)
            and EVALUATION_OPENED
            and len(set(reference_closure["group"].tolist())) >= MIN_EVALUATION_TRAJECTORIES
            and PROVENANCE_COUNTS["patched_forwards"] == 0
            and all(PROVENANCE_COUNTS["model_output_contract_preflights"][short] >= 1
                    for short in ["jepa", "dino"])
        )
        closure_panels_passed = bool(
            source_gate and simulator_gate
            and MODEL_DECISIONS["jepa"]["passed"]
            and MODEL_DECISIONS["dino"]["passed"]
        )

        planning_results = {}
        if closure_panels_passed:
            # The fixed planning candidate bank is opened only after closure passes.
            for model_name in MODEL_NAMES:
                bundle = load_world_model(model_name)
                short = bundle["short"]
                try:
                    for index, record in enumerate(SELECTED_RECORDS["evaluation"]):
                        generate_stage38_path_record(
                            bundle, record, "evaluation_planning", DECODERS[short]
                        )
                        write_json(OUT / f"model_{short}_planning_progress.json", {
                            "completed": index + 1,
                            "total": len(SELECTED_RECORDS["evaluation"]),
                            "last_record_id": int(record["record_id"]),
                        })
                finally:
                    unload_world_model(bundle)
            PLANNING_OPENED = True
            for short in ["jepa", "dino"]:
                planning = load_stage38_sequences(short, "evaluation_planning")
                planning_results[short] = stage38_planning_panel(short, planning)
        planning_gate = bool(
            PLANNING_OPENED
            and set(planning_results) == {"jepa", "dino"}
            and all(planning_results[short]["passed"] for short in ["jepa", "dino"])
        )
        decision = derive_stage38_decision(Stage38Gates(
            source_and_split_binding=source_gate,
            simulator_positive_control=simulator_gate,
            jepa_confirmation=MODEL_DECISIONS["jepa"]["passed"],
            dino_replication=MODEL_DECISIONS["dino"]["passed"],
            planning_value=planning_gate,
        ), run_mode=RUN_MODE)
        SUMMARY = {
            "selected_semigroup": SELECTED_SEMIGROUP,
            "final_training_seeds": list(map(int, FINAL_TRAINING_SEEDS)),
            "simulator_physical_nmse": float(np.mean(simulator_error)),
            "simulator_gain": float(np.mean(simulator_gain)),
            "simulator_gain_ci95": simulator_ci,
            "models": {
                short: closure_results[short]["metrics"] for short in ["jepa", "dino"]
            },
            "model_decisions": MODEL_DECISIONS,
            "planning_opened": PLANNING_OPENED,
            "planning": planning_results,
        }
        DECISION_PAYLOAD = {
            **decision, "protocol_id": PROTOCOL_ID, "run_signature": RUN_SIGNATURE,
            "source_commit": SOURCE_IDENTITY.get("resolved_commit"),
            "model_decisions": MODEL_DECISIONS, "summary": SUMMARY,
            "claim_boundary": {
                "fresh_stage38_panel": True, "environment": ENVIRONMENT,
                "frozen_checkpoints": list(MODEL_NAMES),
                "checkpoint_parameters_updated": False,
                "representations_confirmed": ["JEPA-WM", "DINO-WM"],
                "open_loop_fixed_candidate_planning": bool(PLANNING_OPENED),
                "closed_loop_planning_claimed": False,
                "native_checkpoint_closure_claimed": False,
                "cross_environment_claimed": False,
                "minimal_state_claimed": False, "causal_evidence_claimed": False,
            },
        }
        write_csv(EVIDENCE_DIR / "locked_closure_rows.csv", EVALUATION_ROWS)
        write_csv(EVIDENCE_DIR / "locked_planning_rows.csv", PLANNING_ROWS)
        write_json(EVIDENCE_DIR / "stage38_summary.json", SUMMARY)
        write_json(EVIDENCE_DIR / "stage38_model_decisions.json", MODEL_DECISIONS)
        write_json(OUT / "stage38_decision.json", DECISION_PAYLOAD)
        atomic_checkpoint("stage38_locked_evaluation_complete", {
            "decision_sha256": sha256_file(OUT / "stage38_decision.json"),
            "status": DECISION_PAYLOAD["status"],
            "closure_rows": len(EVALUATION_ROWS),
            "planning_rows": len(PLANNING_ROWS),
            "planning_opened": PLANNING_OPENED,
        })

        figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        labels = ["one-step", "PSCD", "overshoot", "S-PSCD"]
        colors = ["#94a3b8", "#f97316", "#0ea5e9", "#7c3aed"]
        for axis_index, short in enumerate(["jepa", "dino"]):
            metrics = SUMMARY["models"][short]
            axes[axis_index].bar(labels, [
                metrics["one_step_physical_nmse"], metrics["pscd_physical_nmse"],
                metrics["overshoot_physical_nmse"], metrics["spscd_physical_nmse"],
            ], color=colors)
            axes[axis_index].axhline(
                MAX_RECURSIVE_PHYSICAL_NMSE, color="black", linestyle="--"
            )
            axes[axis_index].set_title(f"{short.upper()} locked physical NMSE")
            axes[axis_index].tick_params(axis="x", rotation=25)
        if PLANNING_OPENED:
            axes[2].bar(["JEPA", "DINO"], [
                planning_results["jepa"]["spscd_regret"],
                planning_results["dino"]["spscd_regret"],
            ], color=["#7c3aed", "#0ea5e9"])
            axes[2].set_title("S-PSCD open-loop regret")
        else:
            axes[2].text(
                0.5, 0.5, "Planning sealed:\nclosure gates did not all pass",
                ha="center", va="center", transform=axes[2].transAxes,
            )
            axes[2].set_axis_off()
        figure.suptitle(f"Stage 38: {DECISION_PAYLOAD['status']}")
        figure.tight_layout()
        figure.savefig(PLOT_DIR / "stage38_cross_model_confirmation.png", dpi=180)
        plt.close(figure)
        interpretation = f"""# Automatic Stage 38 interpretation

Status: **{DECISION_PAYLOAD['status'].upper()}**

Closure confirmed: **{DECISION_PAYLOAD['closure_confirmed']}**. Planning opened:
**{PLANNING_OPENED}**. Planning confirmed: **{DECISION_PAYLOAD['planning_confirmed']}**.
The first failed global gate is `{DECISION_PAYLOAD['first_failed_gate']}`.

A closure pass supports post-hoc PSCD closure across the frozen JEPA-WM and
DINO-WM PushT representations. A planning pass additionally supports value in
this fixed finite open-loop candidate bank. It does not support native-model,
causal, closed-loop, minimal-state, or cross-environment claims.
"""
        retry_drive_io(
            "write automatic Stage 38 interpretation",
            lambda: (OUT / "AUTOMATIC_INTERPRETATION.md").write_text(interpretation),
        )
        print(json.dumps(DECISION_PAYLOAD, indent=2))
    except Exception:
        record_failure("stage38_locked_cross_model_evaluation")
'''


packaging = rename(STAGE37.packaging)
packaging = packaging.replace("stage38_spscd", "stage38_xmpscd")
packaging = packaging.replace(
    "semigroup_pscd_planning", "cross_model_pscd_confirmation"
)


protocol_sources = [
    introduction, configuration, installation, setup, analysis_helpers,
    model_helpers, design_and_runtime_helpers, physical_truth, simulator_preflight,
    construction_and_paths, data_and_selection, calibration, locked_evaluation,
    packaging,
]
protocol_sources = [value.strip() for value in protocol_sources]
protocol_digest = hashlib.sha256(
    json.dumps(protocol_sources, ensure_ascii=False, separators=(",", ":")).encode()
).hexdigest()
configuration = configuration.replace("__PROTOCOL_DIGEST__", protocol_digest)
if "__PROTOCOL_DIGEST__" in configuration:
    raise RuntimeError("Stage 38 protocol digest placeholder was not replaced")

cells = [
    markdown(introduction), code(configuration), code(installation), code(setup),
    code(analysis_helpers), code(model_helpers), code(design_and_runtime_helpers),
    code(physical_truth), code(simulator_preflight), code(construction_and_paths),
    code(data_and_selection), code(calibration), code(locked_evaluation), code(packaging),
]
for index, cell in enumerate(cells):
    cell["id"] = f"stage38-{index:02d}"

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "L4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
TARGET.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(f"Wrote {TARGET}")
