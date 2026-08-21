"""Build the source-bound Stage 37 Colab notebook deterministically."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
TARGET = ROOT / "37_semigroup_pscd_planning_value.ipynb"
NUMERICAL = REPOSITORY / "src/cf_faithfulness/stage37_semigroup_pscd.py"

spec = importlib.util.spec_from_file_location(
    "stage36_builder", ROOT / "build_stage36_predictive_state_closure_notebook.py"
)
STAGE36 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(STAGE36)

code = STAGE36.code
markdown = STAGE36.markdown
replace_assignment = STAGE36.replace_assignment
replace_block = STAGE36.replace_block
assigned_uppercase_names = STAGE36.assigned_uppercase_names
function_sources = STAGE36.function_sources


def rename(value: str) -> str:
    replacements = [
        ("Stage 36", "Stage 37"),
        ("STAGE36", "STAGE37"),
        ("stage36", "stage37"),
        ("predictive-state closure distillation", "semigroup-regularized predictive-state closure"),
        ("Predictive-state closure distillation", "Semigroup-regularized predictive-state closure"),
        ("counterfactual_faithfulness_stage37_pscd", "counterfactual_faithfulness_stage37_spscd"),
        ("stage37_pscd_result_bundle_", "stage37_spscd_result_bundle_"),
    ]
    for old, new in replacements:
        value = value.replace(old, new)
    return value


introduction = r'''# Stage 37: semigroup-regularized PSCD and open-loop planning value

## Frozen decision before computation

Stage 36 was the first complete finite-history predictive-state result.  Its
selected 256-coordinate, four-step-history, 128-state mixture reduced recursive
physical error to 0.792 times native JEPA error, and passed state recovery,
history specificity, length, and mode-family checks.  It nevertheless failed
the preregistered composition and semigroup gates (0.286 and 0.333 versus 0.25),
and its low-capacity simulator positive control failed.  It therefore supports
a useful recurrent compensator, not recursive closure.

Stage 37 tests the shortest scientifically discriminating repair.  The frozen
Stage 36 architecture is retained, while training adds direct-versus-composed
agreement from **every eligible native-history anchor** at horizons 2, 4, and
8.  A capacity-matched legacy PSCD receives the same data and optimization but
zero weight on this new objective.  Before JEPA is loaded, a capacity-escalated
neural transition must close directly observed simulator state on disjoint
model-selection trajectories.  Failure stops expensive JEPA inference and is a
valid operator-class result, not a pipeline error.

All trajectory identifiers, action words, and evaluation paths are fresh
relative to Stage 36.  The evaluation also contains a fixed twelve-candidate,
ten-step action bank.  Each physical initial state receives a deterministically
hidden goal equal to one candidate's simulator endpoint.  Native JEPA, legacy
PSCD, and semigroup PSCD rank the same candidates with the same goal and budget;
simulator endpoints are opened only for regret scoring.

## Registered primary questions

1. Can the capacity-escalated operator close true PushT Markov state?
2. Does the semigroup objective improve direct-versus-composed agreement over
   a capacity-matched legacy objective on untouched trajectories and words?
3. Does that improvement cross the absolute composition/closure thresholds?
4. Does it reduce open-loop action-selection regret relative to both frozen
   native JEPA and legacy PSCD?

## Claim boundary

A full pass supports post-hoc semigroup-regularized predictive-state repair of
one frozen JEPA-WM checkpoint and improved finite-bank open-loop planning on
PushT.  It does not establish a minimal state, a native JEPA mechanism, causal
use inside JEPA, closed-loop CEM improvement, or cross-environment generality.
Those remain later confirmatory experiments.
'''


configuration = rename(STAGE36.configuration)
for name, value in {
    "EXPERIMENT_SOURCE_REF": '"codex/stage34-predictive-fiber-abstraction"',
    "PROTOCOL_ID": '"stage37-semigroup-pscd-planning-v1"',
    "NOTEBOOK_PROTOCOL_SHA256": '"__PROTOCOL_DIGEST__"',
    "EVIDENCE_STATUS": '"FRESH_PROSPECTIVE_SEMIGROUP_REPAIR_AND_PLANNING_TEST"',
    "EXPERIMENT_NOTEBOOK_PATH": '"notebooks/37_semigroup_pscd_planning_value.ipynb"',
    "EXPERIMENT_BUILDER_PATH": '"notebooks/build_stage37_semigroup_pscd_planning_notebook.py"',
    "EXPERIMENT_NUMERICAL_PATH": '"src/cf_faithfulness/stage37_semigroup_pscd.py"',
    "OUTPUT_DIR": '"/content/counterfactual_faithfulness_stage37_spscd"',
    "DRIVE_OUTPUT_DIR": '"/content/drive/MyDrive/counterfactual_faithfulness_stage37_spscd"',
    "RUN_REQUEST_PATH": '"/content/drive/MyDrive/counterfactual_faithfulness_stage37_spscd/stage37_run_request.json"',
    "MAX_ESTIMATED_TOTAL_MINUTES": "480.0",
    "SEED": "37101",
    "DESIGN_SEED": "37141",
    "DECODER_SEED": "37183",
    "RANK_SEED": "37213",
    "CALIBRATION_SEED": "37253",
    "BOOTSTRAP_SEED": "37283",
    "CONTROL_SEED": "37351",
    "MAX_WORD_LENGTH": "12",
    "MAX_COMPOSED_LENGTH": "16",
    "CONSTRUCTION_TRAJECTORY_POOL": "list(range(40000, 41600))",
    "MODEL_SELECTION_TRAJECTORY_POOL": "list(range(41600, 43200))",
    "CALIBRATION_TRAJECTORY_POOL": "list(range(43200, 44800))",
    "EVALUATION_TRAJECTORY_POOL": "list(range(44800, 48000))",
    "CONSTRUCTION_TRAJECTORIES": "16",
    "MODEL_SELECTION_TRAJECTORIES": "16",
    "CALIBRATION_TRAJECTORIES": "16",
    "EVALUATION_TRAJECTORIES": "24",
    "TASK_ID_OFFSET": "37000",
}.items():
    configuration = replace_assignment(configuration, name, value)

configuration = replace_block(
    configuration,
    "def stage37_binary_word_spec(name):",
    "CALIBRATION_INTERCHANGE_PAIRS =",
    r'''STAGE37_TOKEN_SPECS = {
    "A": (-45.0, 0.16), "B": (45.0, 0.16),
    "C": (-15.0, 0.20), "D": (15.0, 0.20),
}


def stage37_word_spec(name):
    unknown = sorted(set(str(name)) - set(STAGE37_TOKEN_SPECS))
    if unknown:
        raise ValueError(f"unknown Stage 37 word symbols: {unknown}")
    return {
        "name": str(name),
        "angles": [STAGE37_TOKEN_SPECS[value][0] for value in str(name)],
        "magnitudes": [STAGE37_TOKEN_SPECS[value][1] for value in str(name)],
    }


CONSTRUCTION_WORD_NAMES = [
    "A", "B", "C", "D", "AB", "CD", "BA", "DC", "AC", "BD", "CA", "DB",
    "ABC", "CDA", "BAD", "DCB", "ABAC", "CDBD", "BACD", "DCAB",
    "ACBDA", "BDACB", "AACDBD", "BBDACA", "ACDBACD", "BDACBDA",
    "ACDBACBD", "BDACBDAC",
]
MODEL_SELECTION_WORD_NAMES = [
    "ACBDA", "BDACB", "AACDBD", "BBDACA",
    "ACDBACD", "BDACBDA", "ACDBACBD", "BDACBDAC",
]
CONSTRUCTION_WORD_NAMES = [
    value for value in CONSTRUCTION_WORD_NAMES if value not in MODEL_SELECTION_WORD_NAMES
]
CALIBRATION_WORD_NAMES = [
    "A", "B", "C", "D", "AC", "BD", "CA", "DB",
    "ABC", "CDA", "BAD", "DCB", "ABAC", "CDBD", "BACD", "DCAB",
]
CANONICAL_RESPONSE_WORD_NAMES = ["A", "B", "C", "D", "AB", "CD", "BA", "DC"]
STAGE37_CORE_WORD_NAMES = sorted(
    set(CONSTRUCTION_WORD_NAMES + MODEL_SELECTION_WORD_NAMES + CALIBRATION_WORD_NAMES),
    key=lambda value: (len(value), value),
)
CORE_WORD_SPECS = [stage37_word_spec(name) for name in STAGE37_CORE_WORD_NAMES]
''',
)

configuration = replace_block(
    configuration,
    "CALIBRATION_INTERCHANGE_PAIRS =",
    "EVALUATION_WORD_SPECS =",
    "CALIBRATION_INTERCHANGE_PAIRS = []\n",
)
configuration = replace_block(
    configuration,
    "EVALUATION_WORD_SPECS =",
    "EVALUATION_INTERCHANGE_PAIRS =",
    r'''CLOSURE_EVALUATION_WORD_NAMES = [
    "ACDBACBDA", "BDACBDACB",
    "AACDBACBDA", "BBDACBDACB",
    "ACDBAACBDAC", "BDACBBDACBD",
    "AACDBAACBDAC", "BBDACBBDACBD",
]
PLANNING_WORD_NAMES = [
    "ACBDACBDAC", "CADBCADBCA", "DBACDBACDB", "BDCABDCABD",
    "AACDBACBDA", "BBDACBDACB", "ACACBDBDAC", "BDBDACACBD",
    "CDABCDABCA", "DCBADCABDC", "ABDCABDCAB", "BACDBACDBA",
]
EVALUATION_WORD_NAMES_REGISTERED = sorted(
    set(CLOSURE_EVALUATION_WORD_NAMES + PLANNING_WORD_NAMES),
    key=lambda value: (len(value), value),
)
EVALUATION_WORD_SPECS = [stage37_word_spec(name) for name in EVALUATION_WORD_NAMES_REGISTERED]
''',
)
configuration = replace_block(
    configuration,
    "EVALUATION_INTERCHANGE_PAIRS =",
    "ZERO_WORD_NAMES =",
    "EVALUATION_INTERCHANGE_PAIRS = []\n",
)
configuration = replace_assignment(
    configuration, "ZERO_WORD_NAMES", '{length: f"zero{length}" for length in range(1, 13)}'
)
configuration = replace_assignment(configuration, "CORE_ORDER_PAIRS", '[("AB", "BA"), ("CD", "DC")]')
configuration = replace_block(
    configuration,
    "EVALUATION_ORDER_PAIRS =",
    "STATE_CARRIER_SKETCH_DIM =",
    "EVALUATION_ORDER_PAIRS = []\n",
)

# Remove the Stage 36 appended search block; Stage 37 freezes the selected
# architecture and searches only the registered semigroup strength.
configuration = configuration.split("\n\nPROTOCOL_CONFIG_KEYS =", 1)[0]
_legacy_start = configuration.rfind("\n\nCONSTRUCTION_WORD_NAMES = [")
_legacy_end_marker = (
    'assert max(HISTORY_LENGTHS) < min(len(row["name"]) '
    "for row in EVALUATION_WORD_SPECS)\n"
)
if _legacy_start < 0:
    raise RuntimeError("could not locate the Stage 36 appended search block")
_legacy_end = configuration.index(_legacy_end_marker, _legacy_start) + len(
    _legacy_end_marker
)
configuration = configuration[:_legacy_start] + configuration[_legacy_end:]
configuration += r'''

MAX_CARRIER_PROJECTION_DIM = 256
FIXED_CARRIER_DIM = 256
FIXED_HISTORY_LENGTH = 4
FIXED_LATENT_DIM = 128
FIXED_DYNAMICS = "mixture"
SEMIGROUP_HORIZONS = [2, 4, 8]
SEMIGROUP_WEIGHTS = [0.25, 1.0, 2.0] if RUN_MODE == "pilot" else [0.5]
SIMULATOR_LATENT_DIMS = [128, 256] if RUN_MODE == "pilot" else [32]
SIMULATOR_DYNAMICS = ["single", "mixture"] if RUN_MODE == "pilot" else ["single"]
CANDIDATE_EPOCHS = 100
FINAL_EPOCHS = 300
SIMULATOR_CANDIDATE_EPOCHS = 160
SIMULATOR_FINAL_EPOCHS = 360
ACTIVE_CANDIDATE_EPOCHS = CANDIDATE_EPOCHS if RUN_MODE == "pilot" else 4
ACTIVE_FINAL_EPOCHS = FINAL_EPOCHS if RUN_MODE == "pilot" else 8
ACTIVE_SIMULATOR_CANDIDATE_EPOCHS = SIMULATOR_CANDIDATE_EPOCHS if RUN_MODE == "pilot" else 4
ACTIVE_SIMULATOR_FINAL_EPOCHS = SIMULATOR_FINAL_EPOCHS if RUN_MODE == "pilot" else 8
PSCD_LEARNING_RATE = 1e-3

MAX_SIMULATOR_PREFLIGHT_NMSE = 0.30
MAX_SIMULATOR_CONTROL_NMSE = 0.25
MIN_SIMULATOR_CONTROL_GAIN = 0.50
MIN_NATIVE_FIDELITY_GAIN = 0.10
MIN_SEMIGROUP_ADVANTAGE = 0.05
MIN_HISTORY_ADVANTAGE = 0.05
MAX_RECURSIVE_TO_NATIVE_PHYSICAL_RATIO = 1.00
MAX_RECURSIVE_RATIO_CI_UPPER = 1.20
MAX_COMPOSITION_DISCREPANCY_NMSE = 0.25
MAX_SEMIGROUP_NMSE = 0.25
MAX_RECURSIVE_SUPPORT_ESCAPE_RATE = 0.10
MAX_LENGTH_FAMILY_RATIO = 1.35
MAX_MODE_FAMILY_RATIO = 1.75
MIN_PLANNING_REGRET_REDUCTION = 0.02
PLANNING_DIMENSIONS = [2, 3, 4, 5]

if RUN_MODE == "smoke":
    ACTIVE_MODEL_SELECTION_TRAJECTORIES = 2
    ACTIVE_CALIBRATION_TRAJECTORIES = 2

assert set(MODEL_SELECTION_WORD_NAMES).isdisjoint(set(CONSTRUCTION_WORD_NAMES))
assert set(CANONICAL_RESPONSE_WORD_NAMES).issubset(set(CONSTRUCTION_WORD_NAMES))
assert {len(value) for value in MODEL_SELECTION_WORD_NAMES} == {5, 6, 7, 8}
assert {len(value) for value in CLOSURE_EVALUATION_WORD_NAMES} == {9, 10, 11, 12}
assert {len(value) for value in PLANNING_WORD_NAMES} == {10}
assert set(EVALUATION_WORD_NAMES_REGISTERED).isdisjoint(set(STAGE37_CORE_WORD_NAMES))
assert set().union(*(set(value) for value in EVALUATION_WORD_NAMES_REGISTERED)) <= set(STAGE37_TOKEN_SPECS)
assert max(SEMIGROUP_HORIZONS) <= MAX_WORD_LENGTH
'''
configuration = re.sub(
    r"PINNED = \[.*?\]\n\nassert INTERVENTION_BLOCK",
    '''PINNED = [
    "official_frozen_jepa_wm_pusht_checkpoint", "exact_pusht_state_restoration",
    "fresh_disjoint_trajectory_families_40000_to_47999",
    "fresh_four_symbol_action_bank", "stage36_architecture_frozen",
    "multi_anchor_semigroup_horizons_2_4_8",
    "capacity_matched_zero_semigroup_control", "false_history_control",
    "capacity_escalated_neural_simulator_positive_control_before_jepa",
    "construction_training_model_selection_calibration_then_locked_evaluation",
    "fixed_twelve_candidate_open_loop_planning_bank",
    "simulator_truth_opened_only_for_planning_regret",
    "one_environment_one_checkpoint", "not_closed_loop_planning",
    "not_native_jepa_closure", "not_minimal_state", "observational_not_causal",
    "hash_validated_resume", "transient_drive_and_http_retries",
    "no_synthetic_fallback", "no_required_colab_secret",
]

assert INTERVENTION_BLOCK''',
    configuration,
    count=1,
    flags=re.S,
)
configuration += "\n\nPROTOCOL_CONFIG_KEYS = " + repr(
    assigned_uppercase_names(configuration)
) + "\n"


installation = STAGE36.installation

setup = rename(STAGE36.setup)
setup = setup.replace(
    'log = logging.getLogger("stage37_pscd")',
    'log = logging.getLogger("stage37_spscd")',
)

stage37_helpers = [
    "registered_semigroup_horizons", "fit_semigroup_predictive_state_closure",
    "rollout_predictive_state_from_initial", "select_semigroup_candidate",
    "terminal_values", "goal_cost",
    "grouped_planner_metrics", "Stage37Gates", "derive_stage37_decision",
]
analysis_helpers = STAGE36.analysis_helpers + "\n\n" + function_sources(
    NUMERICAL.read_text(), stage37_helpers
)
analysis_helpers = analysis_helpers.replace(
    "class Stage37Gates:\n", "@dataclass(frozen=True)\nclass Stage37Gates:\n"
)

model_helpers = rename(STAGE36.model_helpers)

design_and_runtime_helpers = rename(STAGE36.design_and_runtime_helpers)
design_and_runtime_helpers = "\n".join(
    line for line in design_and_runtime_helpers.splitlines()
    if not re.match(r'\s*"v[2-7]_', line)
)
design_and_runtime_helpers = replace_block(
    design_and_runtime_helpers,
    "def token_definition(symbol):",
    "def spec_from_name(name):",
    r'''def token_definition(symbol):
    if symbol == "0":
        return (0.0, 0.0)
    if symbol not in STAGE37_TOKEN_SPECS:
        raise KeyError(f"unknown Stage 37 action token {symbol!r}")
    return STAGE37_TOKEN_SPECS[symbol]


''',
)

physical_truth = rename(STAGE36.physical_truth)

simulator_control = r'''# Prove the operator class on true Markov state before loading JEPA-WM.
SIMULATOR_PREFLIGHT_PASSED = False
SELECTED_SIMULATOR_CONTROL = None
SIMULATOR_SELECTION_ROWS = []


def load_stage37_physical_sequences(split):
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
                padded_actions = np.zeros((MAX_WORD_LENGTH, 3), dtype=np.float64)
                padded_actions[:length, :2] = chunks
                padded_actions[:length, 2] = np.linalg.norm(chunks, axis=1)
                target = np.zeros((MAX_WORD_LENGTH, len(GROUNDED_OBSERVABLES)))
                target[:length] = payload["path_observables"][lookup[name], :length]
                valid = np.zeros(MAX_WORD_LENGTH, dtype=bool)
                valid[:length] = True
                rows["initial"].append(grounded_observables(record["state"]))
                rows["actions"].append(padded_actions)
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


def simulator_control_score(artifact, data):
    rollout = rollout_predictive_state_closure(
        artifact, data["initial"], data["actions"], data["targets"], data["mask"]
    )
    valid = rollout["evaluation_mask"]
    error = scaled_path_mse(
        rollout["physical"], data["targets"], valid,
        artifact["normalization"]["physical_scale"], final_only=False,
    )
    persistence = np.repeat(
        data["initial"][:, None, :], MAX_WORD_LENGTH, axis=1
    )
    persistence_error = scaled_path_mse(
        persistence, data["targets"], valid,
        artifact["normalization"]["physical_scale"], final_only=False,
    )
    direct = rollout["direct_state"][valid]
    state_scale = np.maximum(np.std(direct, axis=0, ddof=1), 1e-6)
    semigroup = scaled_path_mse(
        rollout["state"], rollout["direct_state"], valid, state_scale,
        final_only=False,
    )
    return {
        "physical_nmse": float(np.mean(error)),
        "persistence_nmse": float(np.mean(persistence_error)),
        "gain": float(np.mean(relative_gain(error, persistence_error))),
        "semigroup_nmse": float(np.mean(semigroup)),
        "validation_score": float(np.mean(error) + 0.25 * np.mean(semigroup)),
    }


if not PIPELINE_FAILED:
    try:
        verify_executed_notebook_through(
            "# Prove the operator class on true Markov state before loading JEPA-WM."
        )
        train = load_stage37_physical_sequences("construction")
        validation = load_stage37_physical_sequences("model_selection")
        for latent_dim in SIMULATOR_LATENT_DIMS:
            for dynamics in SIMULATOR_DYNAMICS:
                artifact = fit_semigroup_predictive_state_closure(
                    train["initial"], train["actions"], train["targets"],
                    train["targets"], train["mask"], history_length=1,
                    latent_dim=latent_dim, dynamics=dynamics,
                    epochs=ACTIVE_SIMULATOR_CANDIDATE_EPOCHS,
                    learning_rate=PSCD_LEARNING_RATE,
                    seed=stable_seed(CONTROL_SEED, "simulator", latent_dim, dynamics),
                    semigroup_horizons=SEMIGROUP_HORIZONS, semigroup_weight=1.0,
                )
                scores = simulator_control_score(artifact, validation)
                SIMULATOR_SELECTION_ROWS.append({
                    "latent_dim": int(latent_dim), "dynamics": str(dynamics), **scores,
                })
                del artifact
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        SIMULATOR_SELECTION_ROWS.sort(key=lambda row: (
            float(row["validation_score"]), int(row["latent_dim"]), str(row["dynamics"])
        ))
        SELECTED_SIMULATOR_CONTROL = dict(SIMULATOR_SELECTION_ROWS[0])
        SIMULATOR_PREFLIGHT_PASSED = bool(
            SELECTED_SIMULATOR_CONTROL["physical_nmse"] <= MAX_SIMULATOR_PREFLIGHT_NMSE
            and SELECTED_SIMULATOR_CONTROL["gain"] >= MIN_SIMULATOR_CONTROL_GAIN
        )
        write_csv(EVIDENCE_DIR / "simulator_control_selection_rows.csv", SIMULATOR_SELECTION_ROWS)
        write_json(OUT / "simulator_preflight.json", {
            "passed": SIMULATOR_PREFLIGHT_PASSED,
            "selected": SELECTED_SIMULATOR_CONTROL,
            "jepa_loaded_before_decision": False,
            "failure_is_scientific_not_pipeline": True,
        })
        atomic_checkpoint("simulator_operator_preflight", {
            "passed": SIMULATOR_PREFLIGHT_PASSED,
            "selected": SELECTED_SIMULATOR_CONTROL,
        })
        print(json.dumps({
            "simulator_preflight_passed": SIMULATOR_PREFLIGHT_PASSED,
            "selected": SELECTED_SIMULATOR_CONTROL,
        }, indent=2))
    except Exception:
        record_failure("stage37_simulator_operator_preflight")
'''


construction_and_paths = rename(STAGE36.construction_and_paths)
construction_and_paths = replace_block(
    construction_and_paths,
    "def stage37_carrier_projection(value):",
    "def stage37_mode_paths(record, contact_counts, length):",
    r'''def stage37_carrier_projection(value):
    carrier = np.asarray(value, dtype=np.float32)
    expected = (
        EXPECTED_VISUAL_TOKENS,
        EXPECTED_CARRIER_WIDTHS["jepa_wm_pusht"],
    )
    if carrier.shape != expected:
        raise RuntimeError(f"JEPA carrier shape changed: {carrier.shape}")
    return count_sketch(
        carrier.reshape(1, -1), FIXED_CARRIER_DIM,
        stable_seed(CONTROL_SEED, "stage37_fixed_carrier_projection"),
    )[0].astype(np.float32)


''',
)
construction_and_paths = construction_and_paths.replace(
    "if not PIPELINE_FAILED:\n    try:",
    "if not PIPELINE_FAILED and SIMULATOR_PREFLIGHT_PASSED:\n    try:",
    1,
)
construction_and_paths = construction_and_paths.replace(
    'if split == "model_selection":\n        names = ["A", "B"]\n    elif split == "calibration":\n        names = ["A", "B"]',
    'if split == "model_selection":\n        names = list(MODEL_SELECTION_WORD_NAMES)\n    elif split == "calibration":\n        names = list(CALIBRATION_WORD_NAMES)',
)
construction_and_paths = construction_and_paths.replace(
    'if not value or not set(value).issubset({"A", "B"})',
    'if not value or not set(value).issubset(set(STAGE37_TOKEN_SPECS))',
)

data_loader = rename(STAGE36.data_loader)

model_selection = data_loader + r'''

def stage37_slice(data):
    result = dict(data)
    result["initial_carrier"] = data["initial_carrier"][:, :FIXED_CARRIER_DIM]
    result["carrier"] = data["carrier"][:, :, :FIXED_CARRIER_DIM]
    return result


def stage37_pscd_scores(artifact, data):
    result = rollout_predictive_state_closure(
        artifact, data["initial_carrier"], data["actions"], data["carrier"], data["mask"]
    )
    valid = result["evaluation_mask"]
    carrier_error = scaled_path_mse(
        result["carrier"], data["carrier"], valid,
        artifact["normalization"]["carrier_scale"], final_only=False,
    )
    physical_error = scaled_path_mse(
        result["physical"], data["native"], valid,
        artifact["normalization"]["physical_scale"], final_only=False,
    )
    direct = result["direct_state"][valid]
    state_scale = np.maximum(np.std(direct, axis=0, ddof=1), 1e-6)
    semigroup_error = scaled_path_mse(
        result["state"], result["direct_state"], valid, state_scale,
        final_only=False,
    )
    return {
        "carrier_nmse": float(np.mean(carrier_error)),
        "recursive_physical_nmse": float(np.mean(physical_error)),
        "semigroup_nmse": float(np.mean(semigroup_error)),
        "validation_score": float(
            np.mean(carrier_error) + np.mean(physical_error) + np.mean(semigroup_error)
        ),
    }


SELECTED_SEMIGROUP = None
SELECTION_ROWS = []
if not PIPELINE_FAILED and SIMULATOR_PREFLIGHT_PASSED:
    try:
        verify_executed_notebook_through(
            "# Load split-bound teacher sequences without opening evaluation statistics."
        )
        train = stage37_slice(load_stage37_sequences("construction"))
        validation = stage37_slice(load_stage37_sequences("model_selection"))
        for semigroup_weight in SEMIGROUP_WEIGHTS:
            artifact = fit_semigroup_predictive_state_closure(
                train["initial_carrier"], train["actions"], train["carrier"],
                train["native"], train["mask"], history_length=FIXED_HISTORY_LENGTH,
                latent_dim=FIXED_LATENT_DIM, dynamics=FIXED_DYNAMICS,
                epochs=ACTIVE_CANDIDATE_EPOCHS, learning_rate=PSCD_LEARNING_RATE,
                seed=stable_seed(CALIBRATION_SEED, "semigroup_weight_screen"),
                semigroup_horizons=SEMIGROUP_HORIZONS,
                semigroup_weight=semigroup_weight,
            )
            scores = stage37_pscd_scores(artifact, validation)
            SELECTION_ROWS.append({
                "carrier_dim": FIXED_CARRIER_DIM,
                "history_length": FIXED_HISTORY_LENGTH,
                "latent_dim": FIXED_LATENT_DIM,
                "dynamics": FIXED_DYNAMICS,
                "semigroup_weight": float(semigroup_weight),
                "training_loss_initial": artifact["loss_initial"],
                "training_loss_final": artifact["loss_final"],
                **scores,
            })
            del artifact
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        SELECTED_SEMIGROUP = select_semigroup_candidate(SELECTION_ROWS)
        write_csv(EVIDENCE_DIR / "semigroup_model_selection_rows.csv", SELECTION_ROWS)
        selection_path = CALIBRATION_MODEL_DIR / "frozen_semigroup_selection.json"
        write_json(selection_path, {
            "protocol_id": PROTOCOL_ID, "selected": SELECTED_SEMIGROUP,
            "candidate_rows": SELECTION_ROWS, "evaluation_rows_used": 0,
            "stage36_architecture_frozen": True,
        })
        write_digest_sidecar(selection_path)
        atomic_checkpoint("semigroup_model_selection_complete", {
            "selection_sha256": sha256_file(selection_path),
            "selected": SELECTED_SEMIGROUP,
        })
        print(json.dumps({"selected_semigroup": SELECTED_SEMIGROUP}, indent=2))
    except Exception:
        record_failure("stage37_semigroup_model_selection")
'''


calibration = r'''# Freeze S-PSCD, capacity controls, and the physical control before evaluation.
PRIMARY_SPSCD = None
LEGACY_PSCD = None
ONE_STEP_CONTROL = None
FALSE_HISTORY_CONTROL = None
SIMULATOR_CONTROL = None
PHYSICAL_SCALE = None
CARRIER_SCALE = None
PRIMARY_STATE_SCALE = None
LEGACY_STATE_SCALE = None
SUPPORT_REFERENCE = None
CALIBRATION = None
EVALUATION_OPENED = False


def concatenate_stage37_sequences(*bundles):
    return {
        key: np.concatenate([bundle[key] for bundle in bundles], axis=0)
        for key in bundles[0]
    }


def flatten_model_artifact(models):
    arrays, metadata = {}, {}
    def visit(prefix, value):
        if isinstance(value, np.ndarray):
            arrays[prefix] = value
        elif isinstance(value, dict):
            for key, item in sorted(value.items()):
                visit(f"{prefix}.{key}" if prefix else str(key), item)
        elif isinstance(value, (list, tuple)):
            metadata[prefix] = list(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            metadata[prefix] = value
        else:
            raise TypeError(f"unsupported frozen-model value at {prefix}: {type(value)}")
    visit("", models)
    return arrays, metadata


if not PIPELINE_FAILED and SIMULATOR_PREFLIGHT_PASSED:
    try:
        verify_executed_notebook_through(
            "# Freeze S-PSCD, capacity controls, and the physical control before evaluation."
        )
        if SELECTED_SEMIGROUP is None or SELECTED_SIMULATOR_CONTROL is None:
            raise RuntimeError("Stage 37 selections were not frozen")
        construction = load_stage37_sequences("construction")
        calibration_only = load_stage37_sequences("calibration")
        CALIBRATION = stage37_slice(
            concatenate_stage37_sequences(construction, calibration_only)
        )
        common = {
            "history_length": FIXED_HISTORY_LENGTH,
            "latent_dim": FIXED_LATENT_DIM,
            "dynamics": FIXED_DYNAMICS,
            "epochs": ACTIVE_FINAL_EPOCHS,
            "learning_rate": PSCD_LEARNING_RATE,
            "semigroup_horizons": SEMIGROUP_HORIZONS,
        }
        matched_objective_seed = stable_seed(CALIBRATION_SEED, "matched_objective")
        PRIMARY_SPSCD = fit_semigroup_predictive_state_closure(
            CALIBRATION["initial_carrier"], CALIBRATION["actions"],
            CALIBRATION["carrier"], CALIBRATION["native"], CALIBRATION["mask"],
            seed=matched_objective_seed,
            semigroup_weight=float(SELECTED_SEMIGROUP["semigroup_weight"]), **common,
        )
        LEGACY_PSCD = fit_semigroup_predictive_state_closure(
            CALIBRATION["initial_carrier"], CALIBRATION["actions"],
            CALIBRATION["carrier"], CALIBRATION["native"], CALIBRATION["mask"],
            seed=matched_objective_seed, semigroup_weight=0.0, **common,
        )
        ONE_STEP_CONTROL = fit_semigroup_predictive_state_closure(
            CALIBRATION["initial_carrier"], CALIBRATION["actions"],
            CALIBRATION["carrier"], CALIBRATION["native"], CALIBRATION["mask"],
            seed=stable_seed(CONTROL_SEED, "one_step"), semigroup_weight=0.0,
            free_weight=0.0, **common,
        )
        native_history = history_tensor(
            CALIBRATION["initial_carrier"], CALIBRATION["carrier"],
            CALIBRATION["mask"], FIXED_HISTORY_LENGTH,
        )
        false_history = permute_past_history(
            native_history, CALIBRATION["group"], CALIBRATION["mask"],
            seed=stable_seed(CONTROL_SEED, "false_history_train"),
        )
        FALSE_HISTORY_CONTROL = fit_semigroup_predictive_state_closure(
            CALIBRATION["initial_carrier"], CALIBRATION["actions"],
            CALIBRATION["carrier"], CALIBRATION["native"], CALIBRATION["mask"],
            seed=stable_seed(CONTROL_SEED, "false_history_model"),
            semigroup_weight=float(SELECTED_SEMIGROUP["semigroup_weight"]),
            histories_override=false_history, **common,
        )
        physical_construction = load_stage37_physical_sequences("construction")
        physical_calibration = load_stage37_physical_sequences("calibration")
        physical = concatenate_stage37_sequences(
            physical_construction, physical_calibration
        )
        SIMULATOR_CONTROL = fit_semigroup_predictive_state_closure(
            physical["initial"], physical["actions"], physical["targets"],
            physical["targets"], physical["mask"], history_length=1,
            latent_dim=int(SELECTED_SIMULATOR_CONTROL["latent_dim"]),
            dynamics=str(SELECTED_SIMULATOR_CONTROL["dynamics"]),
            epochs=ACTIVE_SIMULATOR_FINAL_EPOCHS, learning_rate=PSCD_LEARNING_RATE,
            seed=stable_seed(CONTROL_SEED, "simulator_final"),
            semigroup_horizons=SEMIGROUP_HORIZONS, semigroup_weight=1.0,
        )
        valid = CALIBRATION["mask"]
        CARRIER_SCALE = np.maximum(np.std(CALIBRATION["carrier"][valid], axis=0, ddof=1), 1e-8)
        PHYSICAL_SCALE = np.maximum(np.std(CALIBRATION["simulator"][valid], axis=0, ddof=1), 1e-8)
        primary_cal = rollout_predictive_state_closure(
            PRIMARY_SPSCD, CALIBRATION["initial_carrier"], CALIBRATION["actions"],
            CALIBRATION["carrier"], CALIBRATION["mask"],
        )
        legacy_cal = rollout_predictive_state_closure(
            LEGACY_PSCD, CALIBRATION["initial_carrier"], CALIBRATION["actions"],
            CALIBRATION["carrier"], CALIBRATION["mask"],
        )
        cal_valid = primary_cal["evaluation_mask"]
        PRIMARY_STATE_SCALE = np.maximum(
            np.std(primary_cal["direct_state"][cal_valid], axis=0, ddof=1), 1e-8
        )
        LEGACY_STATE_SCALE = np.maximum(
            np.std(legacy_cal["direct_state"][cal_valid], axis=0, ddof=1), 1e-8
        )
        SUPPORT_REFERENCE = fit_support_reference(CALIBRATION["carrier"][cal_valid])
        arrays, metadata = flatten_model_artifact({
            "primary": PRIMARY_SPSCD, "legacy": LEGACY_PSCD,
            "one_step": ONE_STEP_CONTROL, "false_history": FALSE_HISTORY_CONTROL,
            "simulator": SIMULATOR_CONTROL, "carrier_scale": CARRIER_SCALE,
            "physical_scale": PHYSICAL_SCALE, "primary_state_scale": PRIMARY_STATE_SCALE,
            "legacy_state_scale": LEGACY_STATE_SCALE,
            "support_reference": SUPPORT_REFERENCE,
        })
        model_path = CALIBRATION_MODEL_DIR / "frozen_stage37_models.npz"
        atomic_npz(model_path, **arrays)
        write_json(CALIBRATION_MODEL_DIR / "frozen_stage37_models_schema.json", metadata)
        certificate_path = CALIBRATION_MODEL_DIR / "evaluation_open_certificate.json"
        write_json(certificate_path, {
            "protocol_id": PROTOCOL_ID, "run_signature": RUN_SIGNATURE,
            "selection_sha256": sha256_file(
                CALIBRATION_MODEL_DIR / "frozen_semigroup_selection.json"
            ),
            "models_sha256": sha256_file(model_path),
            "evaluation_statistics_read": False, "evaluation_metrics_computed": False,
            "planning_goal_indices_opened": False, "jepa_parameters_updated": False,
        })
        write_digest_sidecar(certificate_path)
        atomic_checkpoint("stage37_models_frozen", {
            "certificate_sha256": sha256_file(certificate_path),
            "models_sha256": sha256_file(model_path),
        })
        print(json.dumps({
            "selected_semigroup": SELECTED_SEMIGROUP,
            "selected_simulator_control": SELECTED_SIMULATOR_CONTROL,
            "evaluation_opened": EVALUATION_OPENED,
        }, indent=2))
    except Exception:
        record_failure("stage37_calibration_model_freeze")
'''


locked_evaluation = r'''# Open fresh closure and planning evaluation once and derive every Stage 37 gate.
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


def subset_stage37(data, selected):
    selected = np.asarray(selected, dtype=bool)
    return {key: np.asarray(value)[selected] for key, value in data.items()}


if not PIPELINE_FAILED and not SIMULATOR_PREFLIGHT_PASSED:
    DECISION_PAYLOAD.update({
        "protocol_id": PROTOCOL_ID, "run_signature": RUN_SIGNATURE,
        "first_failed_gate": "simulator_positive_control",
        "selected_simulator_control": SELECTED_SIMULATOR_CONTROL,
        "jepa_loaded": False, "evaluation_opened": False,
        "scientific_failure_not_pipeline_error": True,
    })
    write_json(OUT / "stage37_decision.json", DECISION_PAYLOAD)


if not PIPELINE_FAILED and SIMULATOR_PREFLIGHT_PASSED:
    try:
        verify_executed_notebook_through(
            "# Open fresh closure and planning evaluation once and derive every Stage 37 gate."
        )
        certificate_path = CALIBRATION_MODEL_DIR / "evaluation_open_certificate.json"
        validate_digest_sidecar(certificate_path)
        bundle = load_world_model("jepa_wm_pusht")
        try:
            for index, record in enumerate(SELECTED_RECORDS["evaluation"]):
                generate_stage37_path_record(bundle, record, "evaluation", JEPA_DECODER)
                write_json(OUT / "model_jepa_evaluation_progress.json", {
                    "completed": index + 1,
                    "total": len(SELECTED_RECORDS["evaluation"]),
                    "last_record_id": int(record["record_id"]),
                })
        finally:
            unload_world_model(bundle)
        EVALUATION_OPENED = True
        all_evaluation = stage37_slice(load_stage37_sequences("evaluation"))
        closure = subset_stage37(
            all_evaluation,
            np.isin(all_evaluation["word"], CLOSURE_EVALUATION_WORD_NAMES),
        )
        planning = subset_stage37(
            all_evaluation,
            np.isin(all_evaluation["word"], PLANNING_WORD_NAMES),
        )
        mask = closure["mask"]
        groups = closure["group"]
        primary = rollout_predictive_state_closure(
            PRIMARY_SPSCD, closure["initial_carrier"], closure["actions"],
            closure["carrier"], mask,
        )
        legacy = rollout_predictive_state_closure(
            LEGACY_PSCD, closure["initial_carrier"], closure["actions"],
            closure["carrier"], mask,
        )
        one_step = rollout_predictive_state_closure(
            ONE_STEP_CONTROL, closure["initial_carrier"], closure["actions"],
            closure["carrier"], mask,
        )
        native_history = history_tensor(
            closure["initial_carrier"], closure["carrier"], mask, FIXED_HISTORY_LENGTH,
        )
        false_history = permute_past_history(
            native_history, groups, mask,
            seed=stable_seed(CONTROL_SEED, "false_history_evaluation"),
        )
        false_control = rollout_predictive_state_closure(
            FALSE_HISTORY_CONTROL, closure["initial_carrier"], closure["actions"],
            closure["carrier"], mask, histories_override=false_history,
        )
        simulator = rollout_predictive_state_closure(
            SIMULATOR_CONTROL, closure["initial_physical"], closure["actions"],
            closure["simulator"], mask,
        )
        evaluated = primary["evaluation_mask"]
        physical_persistence = np.repeat(
            closure["initial_physical"][:, None, :], MAX_WORD_LENGTH, axis=1
        )
        native_error = scaled_path_mse(
            closure["native"], closure["simulator"], evaluated, PHYSICAL_SCALE
        )
        primary_error = scaled_path_mse(
            primary["physical"], closure["simulator"], evaluated, PHYSICAL_SCALE
        )
        legacy_error = scaled_path_mse(
            legacy["physical"], closure["simulator"], evaluated, PHYSICAL_SCALE
        )
        one_step_error = scaled_path_mse(
            one_step["physical"], closure["simulator"], evaluated, PHYSICAL_SCALE
        )
        false_error = scaled_path_mse(
            false_control["physical"], closure["simulator"], evaluated, PHYSICAL_SCALE
        )
        persistence_error = scaled_path_mse(
            physical_persistence, closure["simulator"], evaluated, PHYSICAL_SCALE
        )
        simulator_error = scaled_path_mse(
            simulator["physical"], closure["simulator"], evaluated, PHYSICAL_SCALE
        )
        composition_error = scaled_path_mse(
            primary["physical"], closure["native"], evaluated, PHYSICAL_SCALE
        )
        primary_semigroup = scaled_path_mse(
            primary["state"], primary["direct_state"], evaluated,
            PRIMARY_STATE_SCALE, final_only=False,
        )
        legacy_semigroup = scaled_path_mse(
            legacy["state"], legacy["direct_state"], evaluated,
            LEGACY_STATE_SCALE, final_only=False,
        )
        simulator_gain = relative_gain(simulator_error, persistence_error)
        native_gain = relative_gain(native_error, persistence_error)
        semigroup_gain = relative_gain(primary_semigroup, legacy_semigroup)
        history_gain = relative_gain(primary_error, false_error)
        one_step_gain = relative_gain(primary_error, one_step_error)
        simulator_ci = clustered_mean_interval(
            simulator_gain, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
            seed=stable_seed(BOOTSTRAP_SEED, "simulator"),
        )
        native_ci = clustered_mean_interval(
            native_gain, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
            seed=stable_seed(BOOTSTRAP_SEED, "native"),
        )
        semigroup_ci = clustered_mean_interval(
            semigroup_gain, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
            seed=stable_seed(BOOTSTRAP_SEED, "semigroup"),
        )
        history_ci = clustered_mean_interval(
            history_gain, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
            seed=stable_seed(BOOTSTRAP_SEED, "history"),
        )
        recursive_ratio = float(np.mean(primary_error) / max(np.mean(native_error), 1e-12))
        recursive_ratio_ci = clustered_ratio_interval(
            primary_error, native_error, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
            seed=stable_seed(BOOTSTRAP_SEED, "recursive_ratio"),
        )
        support_escape = support_exceedance_rate(
            SUPPORT_REFERENCE, primary["carrier"][evaluated]
        )

        # Equal-budget open-loop planning on a fixed 12-candidate, length-10 bank.
        primary_plan = rollout_predictive_state_from_initial(
            PRIMARY_SPSCD, planning["initial_carrier"], planning["actions"],
            planning["mask"],
        )
        legacy_plan = rollout_predictive_state_from_initial(
            LEGACY_PSCD, planning["initial_carrier"], planning["actions"],
            planning["mask"],
        )
        native_end = terminal_values(planning["native"], planning["mask"])
        primary_end = terminal_values(primary_plan["physical"], primary_plan["evaluation_mask"])
        legacy_end = terminal_values(legacy_plan["physical"], legacy_plan["evaluation_mask"])
        truth_end = terminal_values(planning["simulator"], planning["mask"])
        goals = np.zeros_like(truth_end)
        goal_words = {}
        for record_id in np.unique(planning["record_id"]):
            rows = np.flatnonzero(planning["record_id"] == record_id)
            target_word = PLANNING_WORD_NAMES[
                stable_seed(DESIGN_SEED, "planning_goal", int(record_id)) % len(PLANNING_WORD_NAMES)
            ]
            target_rows = rows[planning["word"][rows] == target_word]
            if len(target_rows) != 1:
                raise RuntimeError("planning goal word is not unique within a record")
            goals[rows] = truth_end[target_rows[0]]
            goal_words[int(record_id)] = target_word
        true_cost = goal_cost(truth_end, goals, PHYSICAL_SCALE, PLANNING_DIMENSIONS)
        native_cost = goal_cost(native_end, goals, PHYSICAL_SCALE, PLANNING_DIMENSIONS)
        primary_cost = goal_cost(primary_end, goals, PHYSICAL_SCALE, PLANNING_DIMENSIONS)
        legacy_cost = goal_cost(legacy_end, goals, PHYSICAL_SCALE, PLANNING_DIMENSIONS)
        plan_group = planning["record_id"]
        native_metrics = grouped_planner_metrics(native_cost, true_cost, plan_group)
        primary_metrics = grouped_planner_metrics(primary_cost, true_cost, plan_group)
        legacy_metrics = grouped_planner_metrics(legacy_cost, true_cost, plan_group)
        planning_clusters = np.asarray([
            planning["group"][np.flatnonzero(planning["record_id"] == record_id)[0]]
            for record_id in primary_metrics["groups"]
        ])
        native_regret_reduction = native_metrics["regret"] - primary_metrics["regret"]
        legacy_regret_reduction = legacy_metrics["regret"] - primary_metrics["regret"]
        native_planning_ci = clustered_mean_interval(
            native_regret_reduction, planning_clusters, draws=ACTIVE_BOOTSTRAP_DRAWS,
            seed=stable_seed(BOOTSTRAP_SEED, "planning_native"),
        )
        legacy_planning_ci = clustered_mean_interval(
            legacy_regret_reduction, planning_clusters, draws=ACTIVE_BOOTSTRAP_DRAWS,
            seed=stable_seed(BOOTSTRAP_SEED, "planning_legacy"),
        )
        length_ratios = {
            str(length): float(np.mean(primary_error[closure["length"] == length]) /
                max(np.mean(native_error[closure["length"] == length]), 1e-12))
            for length in sorted(set(closure["length"].tolist()))
        }
        mode_ratios = {
            mode: float(np.mean(primary_error[closure["initial_mode"] == mode]) /
                max(np.mean(native_error[closure["initial_mode"] == mode]), 1e-12))
            for mode in MODE_LABELS
        }
        source_gate = bool(
            SOURCE_IDENTITY.get("confirmation_eligible", False)
            and EVALUATION_OPENED
            and len(set(groups.tolist())) >= MIN_EVALUATION_TRAJECTORIES
        )
        simulator_gate = bool(
            np.mean(simulator_error) <= MAX_SIMULATOR_CONTROL_NMSE
            and np.mean(simulator_gain) >= MIN_SIMULATOR_CONTROL_GAIN
            and simulator_ci[0] > 0
        )
        native_gate = bool(np.mean(native_gain) >= MIN_NATIVE_FIDELITY_GAIN and native_ci[0] > 0)
        semigroup_gate = bool(
            np.mean(semigroup_gain) >= MIN_SEMIGROUP_ADVANTAGE
            and semigroup_ci[0] > 0
            and np.mean(primary_semigroup) <= MAX_SEMIGROUP_NMSE
        )
        recursive_gate = bool(
            recursive_ratio <= MAX_RECURSIVE_TO_NATIVE_PHYSICAL_RATIO
            and recursive_ratio_ci[1] <= MAX_RECURSIVE_RATIO_CI_UPPER
            and np.mean(composition_error) <= MAX_COMPOSITION_DISCREPANCY_NMSE
            and support_escape <= MAX_RECURSIVE_SUPPORT_ESCAPE_RATE
        )
        planning_gate = bool(
            np.mean(native_regret_reduction) >= MIN_PLANNING_REGRET_REDUCTION
            and np.mean(legacy_regret_reduction) >= MIN_PLANNING_REGRET_REDUCTION
            and native_planning_ci[0] > 0 and legacy_planning_ci[0] > 0
            and np.mean(primary_metrics["pairwise_accuracy"]) >= np.mean(native_metrics["pairwise_accuracy"])
            and np.mean(primary_metrics["pairwise_accuracy"]) >= np.mean(legacy_metrics["pairwise_accuracy"])
        )
        history_gate = bool(
            np.mean(history_gain) >= MIN_HISTORY_ADVANTAGE and history_ci[0] > 0
            and np.mean(one_step_gain) > 0
        )
        family_gate = bool(
            all(value <= MAX_LENGTH_FAMILY_RATIO for value in length_ratios.values())
            and all(value <= MAX_MODE_FAMILY_RATIO for value in mode_ratios.values())
        )
        decision = derive_stage37_decision(Stage37Gates(
            source_and_split_binding=source_gate,
            simulator_positive_control=simulator_gate,
            native_physical_fidelity=native_gate,
            semigroup_regularization_advantage=semigroup_gate,
            recursive_closure=recursive_gate,
            planning_value=planning_gate,
            history_specificity=history_gate,
            family_consistency=family_gate,
        ), run_mode=RUN_MODE)
        SUMMARY = {
            "selected_semigroup": SELECTED_SEMIGROUP,
            "selected_simulator_control": SELECTED_SIMULATOR_CONTROL,
            "simulator_control_nmse": float(np.mean(simulator_error)),
            "simulator_control_gain": float(np.mean(simulator_gain)),
            "simulator_control_gain_ci95": simulator_ci,
            "native_physical_nmse": float(np.mean(native_error)),
            "primary_recursive_physical_nmse": float(np.mean(primary_error)),
            "legacy_recursive_physical_nmse": float(np.mean(legacy_error)),
            "recursive_to_native_ratio": recursive_ratio,
            "recursive_to_native_ratio_ci95": recursive_ratio_ci,
            "composition_discrepancy_nmse": float(np.mean(composition_error)),
            "primary_semigroup_nmse": float(np.mean(primary_semigroup)),
            "legacy_semigroup_nmse": float(np.mean(legacy_semigroup)),
            "semigroup_advantage": float(np.mean(semigroup_gain)),
            "semigroup_advantage_ci95": semigroup_ci,
            "history_advantage": float(np.mean(history_gain)),
            "history_advantage_ci95": history_ci,
            "support_escape_rate": support_escape,
            "native_planning_regret": float(np.mean(native_metrics["regret"])),
            "legacy_planning_regret": float(np.mean(legacy_metrics["regret"])),
            "primary_planning_regret": float(np.mean(primary_metrics["regret"])),
            "native_regret_reduction": float(np.mean(native_regret_reduction)),
            "legacy_regret_reduction": float(np.mean(legacy_regret_reduction)),
            "native_regret_reduction_ci95": native_planning_ci,
            "legacy_regret_reduction_ci95": legacy_planning_ci,
            "native_pairwise_accuracy": float(np.mean(native_metrics["pairwise_accuracy"])),
            "legacy_pairwise_accuracy": float(np.mean(legacy_metrics["pairwise_accuracy"])),
            "primary_pairwise_accuracy": float(np.mean(primary_metrics["pairwise_accuracy"])),
            "native_success": float(np.mean(native_metrics["success"])),
            "legacy_success": float(np.mean(legacy_metrics["success"])),
            "primary_success": float(np.mean(primary_metrics["success"])),
            "length_ratios": length_ratios, "initial_mode_ratios": mode_ratios,
        }
        DECISION_PAYLOAD = {
            **decision, "protocol_id": PROTOCOL_ID, "run_signature": RUN_SIGNATURE,
            "source_commit": SOURCE_IDENTITY.get("resolved_commit"), "summary": SUMMARY,
            "claim_boundary": {
                "fresh_stage37_panel": True, "one_environment": ENVIRONMENT,
                "one_jepa_checkpoint": True, "jepa_parameters_updated": False,
                "open_loop_fixed_candidate_planning": True,
                "closed_loop_planning_claimed": False,
                "native_jepa_mechanism_claimed": False,
                "minimal_state_claimed": False, "causal_evidence_claimed": False,
            },
        }
        for index in range(len(closure["word"])):
            EVALUATION_ROWS.append({
                "record_id": int(closure["record_id"][index]),
                "trajectory_id": int(groups[index]),
                "initial_mode": str(closure["initial_mode"][index]),
                "word": str(closure["word"][index]),
                "word_length": int(closure["length"][index]),
                "native_physical_mse": float(native_error[index]),
                "primary_physical_mse": float(primary_error[index]),
                "legacy_physical_mse": float(legacy_error[index]),
                "primary_semigroup_mse": float(primary_semigroup[index]),
                "legacy_semigroup_mse": float(legacy_semigroup[index]),
            })
        for index, record_id in enumerate(primary_metrics["groups"]):
            PLANNING_ROWS.append({
                "record_id": int(record_id),
                "goal_word": goal_words[int(record_id)],
                "native_regret": float(native_metrics["regret"][index]),
                "legacy_regret": float(legacy_metrics["regret"][index]),
                "primary_regret": float(primary_metrics["regret"][index]),
                "native_pairwise_accuracy": float(native_metrics["pairwise_accuracy"][index]),
                "legacy_pairwise_accuracy": float(legacy_metrics["pairwise_accuracy"][index]),
                "primary_pairwise_accuracy": float(primary_metrics["pairwise_accuracy"][index]),
            })
        write_csv(EVIDENCE_DIR / "locked_closure_rows.csv", EVALUATION_ROWS)
        write_csv(EVIDENCE_DIR / "locked_planning_rows.csv", PLANNING_ROWS)
        write_json(EVIDENCE_DIR / "stage37_summary.json", SUMMARY)
        write_json(OUT / "stage37_decision.json", DECISION_PAYLOAD)
        atomic_checkpoint("locked_evaluation_complete", {
            "decision_sha256": sha256_file(OUT / "stage37_decision.json"),
            "status": DECISION_PAYLOAD["status"],
            "closure_rows": len(EVALUATION_ROWS), "planning_queries": len(PLANNING_ROWS),
        })

        figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        axes[0].bar(["native", "legacy", "S-PSCD"], [
            np.mean(native_error), np.mean(legacy_error), np.mean(primary_error)
        ], color=["#0ea5e9", "#f97316", "#7c3aed"])
        axes[0].set_title("Locked physical NMSE")
        axes[1].bar(["legacy", "S-PSCD"], [
            np.mean(legacy_semigroup), np.mean(primary_semigroup)
        ], color=["#f97316", "#7c3aed"])
        axes[1].axhline(MAX_SEMIGROUP_NMSE, color="black", linestyle="--")
        axes[1].set_title("Semigroup NMSE")
        axes[2].bar(["native", "legacy", "S-PSCD"], [
            np.mean(native_metrics["regret"]), np.mean(legacy_metrics["regret"]),
            np.mean(primary_metrics["regret"]),
        ], color=["#0ea5e9", "#f97316", "#7c3aed"])
        axes[2].set_title("Open-loop planning regret")
        figure.suptitle(f"Stage 37: {DECISION_PAYLOAD['status']}")
        figure.tight_layout()
        figure.savefig(PLOT_DIR / "stage37_closure_and_planning.png", dpi=180)
        plt.close(figure)
        interpretation = f"""# Automatic Stage 37 interpretation

Status: **{DECISION_PAYLOAD['status'].upper()}**

The first failed gate is `{DECISION_PAYLOAD['first_failed_gate']}`.  A full pass
supports semigroup-regularized post-hoc repair and finite-bank open-loop
planning value for this frozen PushT JEPA-WM.  It is not closed-loop planning,
cross-environment generality, or native causal-mechanism evidence.
"""
        retry_drive_io(
            "write automatic interpretation",
            lambda: (OUT / "AUTOMATIC_INTERPRETATION.md").write_text(interpretation),
        )
        print(json.dumps(DECISION_PAYLOAD, indent=2))
    except Exception:
        record_failure("stage37_locked_closure_and_planning_evaluation")
'''


packaging = rename(STAGE36.packaging)
packaging = packaging.replace("stage37_pscd", "stage37_spscd")
packaging = packaging.replace("predictive_state_closure", "semigroup_pscd_planning")


protocol_sources = [
    introduction, configuration, installation, setup, analysis_helpers,
    model_helpers, design_and_runtime_helpers, physical_truth, simulator_control,
    construction_and_paths, model_selection, calibration, locked_evaluation,
    packaging,
]
protocol_sources = [value.strip() for value in protocol_sources]
protocol_digest = hashlib.sha256(
    json.dumps(protocol_sources, ensure_ascii=False, separators=(",", ":")).encode()
).hexdigest()
configuration = configuration.replace("__PROTOCOL_DIGEST__", protocol_digest)
if "__PROTOCOL_DIGEST__" in configuration:
    raise RuntimeError("Stage 37 protocol digest placeholder was not replaced")

cells = [
    markdown(introduction), code(configuration), code(installation), code(setup),
    code(analysis_helpers), code(model_helpers), code(design_and_runtime_helpers),
    code(physical_truth), code(simulator_control), code(construction_and_paths),
    code(model_selection), code(calibration), code(locked_evaluation), code(packaging),
]
for index, cell in enumerate(cells):
    cell["id"] = f"stage37-{index:02d}"

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
