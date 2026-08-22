"""Build the source-bound Stage 37.1 horizon-control Colab notebook."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
TARGET = ROOT / "37_1_horizon_matched_operator_calibration.ipynb"
NUMERICAL = REPOSITORY / "src/cf_faithfulness/stage37_1_horizon_control.py"

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
        ("Stage 37", "Stage 37.1"),
        ("STAGE37", "STAGE37_1"),
        ("stage37", "stage37_1"),
    ]:
        value = value.replace(old, new)
    return value


introduction = r'''# Stage 37.1: horizon-matched true-state operator calibration

## Frozen decision before computation

Stage 37 produced a complete and informative negative result.  Its
semigroup-regularized adapter improved recursive physical NMSE from 1.465 to
0.833 relative to a matched legacy adapter and reduced latent semigroup NMSE
by 19.4%.  It nevertheless failed absolute closure, history specificity, and
native-JEPA planning value.  More fundamentally, the true-state simulator
control passed length-5--8 development (physical NMSE 0.114) but failed the
fresh length-9--12 panel (NMSE 1.426).  That first failed gate prevents the
absolute JEPA closure result from being interpreted confirmatorily.

Stage 37.1 calibrates the measuring instrument before another JEPA run.  It is
simulator-only: JEPA-WM and every pretrained checkpoint are prohibited.  The
model receives the exact 11-dimensional PushT Markov state and action.  Every
training, selection, calibration, and evaluation word has length 9--12, while
trajectory families and exact action words remain disjoint across all splits.

The selected neural operator is trained with free-running rollout and
multi-anchor direct-versus-composed agreement at horizons 2, 4, and 8.  It is
compared with the same architecture and initialization trained without the
semigroup objective, and with a one-step-only control.  Evaluation is opened
once after all three models and their normalization scales are frozen.

## Sequential gates

1. Exact source, word, and trajectory split binding.
2. Length-9--12 development preflight on disjoint words and trajectories.
3. Locked physical closure: NMSE at most 0.25 and at least 50% improvement over
   state persistence with a positive clustered confidence interval.
4. Locked semigroup closure: direct-versus-composed NMSE at most 0.25.
5. Objective specificity: the primary must beat both zero-semigroup and
   one-step-only matched controls.
6. Every length and physical mode must remain below its registered error cap.

A full pass calibrates this operator class and authorizes a later, completely
fresh JEPA S-PSCD confirmation.  It is not itself evidence about JEPA, planning,
representation closure, or causal mechanisms.
'''


configuration = rename(STAGE37.configuration)
for name, value in {
    "EXPERIMENT_SOURCE_REF": '"codex/stage34-predictive-fiber-abstraction"',
    "PROTOCOL_ID": '"stage37-1-horizon-matched-operator-calibration-v1"',
    "NOTEBOOK_PROTOCOL_SHA256": '"__PROTOCOL_DIGEST__"',
    "EVIDENCE_STATUS": '"FRESH_PROSPECTIVE_SIMULATOR_OPERATOR_CALIBRATION"',
    "EXPERIMENT_NOTEBOOK_PATH": '"notebooks/37_1_horizon_matched_operator_calibration.ipynb"',
    "EXPERIMENT_BUILDER_PATH": '"notebooks/build_stage37_1_horizon_control_notebook.py"',
    "EXPERIMENT_NUMERICAL_PATH": '"src/cf_faithfulness/stage37_1_horizon_control.py"',
    "OUTPUT_DIR": '"/content/counterfactual_faithfulness_stage37_1_hmoc"',
    "DRIVE_OUTPUT_DIR": '"/content/drive/MyDrive/counterfactual_faithfulness_stage37_1_hmoc"',
    "RUN_REQUEST_PATH": '"/content/drive/MyDrive/counterfactual_faithfulness_stage37_1_hmoc/stage37_1_run_request.json"',
    "MAX_ESTIMATED_TOTAL_MINUTES": "180.0",
    "SEED": "371101",
    "DESIGN_SEED": "371141",
    "DECODER_SEED": "371183",
    "RANK_SEED": "371213",
    "CALIBRATION_SEED": "371253",
    "BOOTSTRAP_SEED": "371283",
    "CONTROL_SEED": "371351",
    "CONSTRUCTION_TRAJECTORY_POOL": "list(range(48000, 49600))",
    "MODEL_SELECTION_TRAJECTORY_POOL": "list(range(49600, 51200))",
    "CALIBRATION_TRAJECTORY_POOL": "list(range(51200, 52800))",
    "EVALUATION_TRAJECTORY_POOL": "list(range(52800, 56000))",
    "CONSTRUCTION_TRAJECTORIES": "16",
    "MODEL_SELECTION_TRAJECTORIES": "16",
    "CALIBRATION_TRAJECTORIES": "16",
    "EVALUATION_TRAJECTORIES": "32",
    "TASK_ID_OFFSET": "371000",
}.items():
    configuration = replace_assignment(configuration, name, value)

configuration = replace_block(
    configuration,
    "CONSTRUCTION_WORD_NAMES = [",
    "CALIBRATION_INTERCHANGE_PAIRS =",
    r'''CANONICAL_RESPONSE_WORD_NAMES = ["A", "B", "C", "D", "AB", "CD", "BA", "DC"]
CONSTRUCTION_WORD_NAMES = [
    "ACCCCBBBD", "ACBBABDAB", "CCACBDBCDB", "ABACCDDAAB",
    "DBDCABABACB", "CADDADBCCCB", "ADBDCACDCBDA", "BCACADDBCDCC",
]
MODEL_SELECTION_WORD_NAMES = [
    "DDBBBDCAD", "DCCADBDAA", "DADCBAABAD", "ACBAADDBCC",
    "CBCCACCDCDA", "BCCADDCDCAD", "CCAABCBDDCBC", "CCBBDAAABBDB",
]
CALIBRATION_WORD_NAMES = [
    "DCDADBCDB", "ADADCBCBC", "BBAABBCCDC", "ABACBABACD",
    "ACDBBCDCAAD", "DCBACCABDAA", "ACADDBBBBDCD", "CDDBCACACBCC",
]
STAGE37_1_CORE_WORD_NAMES = sorted(
    set(
        CANONICAL_RESPONSE_WORD_NAMES + CONSTRUCTION_WORD_NAMES
        + MODEL_SELECTION_WORD_NAMES + CALIBRATION_WORD_NAMES
    ),
    key=lambda value: (len(value), value),
)
CORE_WORD_SPECS = [stage37_1_word_spec(name) for name in STAGE37_1_CORE_WORD_NAMES]
''',
)
configuration = replace_block(
    configuration,
    "CLOSURE_EVALUATION_WORD_NAMES = [",
    "EVALUATION_INTERCHANGE_PAIRS =",
    r'''EVALUATION_WORD_NAMES_REGISTERED = [
    "ACADAACDB", "ADABDCBCD", "DDBDDACDDC", "CBCACDBCAA",
    "CDCBDBADADD", "ACDAABAABBA", "DDBDDDDBABCB", "BCDCDADCBBAB",
]
EVALUATION_WORD_SPECS = [
    stage37_1_word_spec(name) for name in EVALUATION_WORD_NAMES_REGISTERED
]
''',
)
configuration = replace_assignment(configuration, "CORE_ORDER_PAIRS", '[("AB", "BA"), ("CD", "DC")]')
configuration = configuration.replace(
    'assert {len(row["angles"]) for row in CORE_WORD_SPECS} == set(range(1, 9))',
    'assert {len(row["angles"]) for row in CORE_WORD_SPECS} == {1, 2, 9, 10, 11, 12}',
)
configuration = configuration.split(
    "\n\nV2_PREFLIGHT_HELPER_ORDER_AMENDMENT =", 1
)[0]
configuration = re.sub(
    r"PINNED = \[.*?\]\n\nassert INTERVENTION_BLOCK",
    '''PINNED = [
    "simulator_only_no_jepa_or_pretrained_checkpoint",
    "exact_pusht_markov_state", "fresh_trajectory_ids_48000_to_55999",
    "four_disjoint_length_9_to_12_action_word_banks",
    "same_horizon_distribution_across_all_splits",
    "multi_anchor_semigroup_horizons_2_4_8",
    "matched_zero_semigroup_control", "matched_one_step_only_control",
    "construction_training_model_selection_calibration_then_locked_evaluation",
    "locked_evaluation_opened_once", "clustered_trajectory_bootstrap",
    "hash_validated_resume", "transient_drive_and_http_retries",
    "no_synthetic_fallback", "no_required_colab_secret",
]

assert INTERVENTION_BLOCK''',
    configuration,
    count=1,
    flags=re.S,
)
configuration += r'''

MAX_CARRIER_PROJECTION_DIM = 256
SEMIGROUP_HORIZONS = [2, 4, 8]
SIMULATOR_LATENT_DIMS = [128, 256] if RUN_MODE == "pilot" else [32]
SIMULATOR_DYNAMICS = ["single", "mixture"] if RUN_MODE == "pilot" else ["single"]
CANDIDATE_EPOCHS = 180
FINAL_EPOCHS = 420
ACTIVE_CANDIDATE_EPOCHS = CANDIDATE_EPOCHS if RUN_MODE == "pilot" else 4
ACTIVE_FINAL_EPOCHS = FINAL_EPOCHS if RUN_MODE == "pilot" else 8
LEARNING_RATE = 1e-3
SEMIGROUP_WEIGHT = 1.0

MAX_DEVELOPMENT_PHYSICAL_NMSE = 0.30
MIN_CONTROL_GAIN = 0.50
MAX_LOCKED_PHYSICAL_NMSE = 0.25
MAX_LOCKED_SEMIGROUP_NMSE = 0.25
MIN_OBJECTIVE_ADVANTAGE = 0.05
MAX_LENGTH_PHYSICAL_NMSE = 0.35
MAX_MODE_PHYSICAL_NMSE = 0.40

if RUN_MODE == "smoke":
    ACTIVE_MODEL_SELECTION_TRAJECTORIES = 2
    ACTIVE_CALIBRATION_TRAJECTORIES = 2

TASK_WORD_BANKS = [
    CONSTRUCTION_WORD_NAMES, MODEL_SELECTION_WORD_NAMES,
    CALIBRATION_WORD_NAMES, EVALUATION_WORD_NAMES_REGISTERED,
]
assert all({len(value) for value in bank} == {9, 10, 11, 12} for bank in TASK_WORD_BANKS)
assert all(
    not set(TASK_WORD_BANKS[left]) & set(TASK_WORD_BANKS[right])
    for left in range(4) for right in range(left + 1, 4)
)
assert all(len(bank) == 8 for bank in TASK_WORD_BANKS)
assert set().union(*(set(value) for bank in TASK_WORD_BANKS for value in bank)) <= set(STAGE37_1_TOKEN_SPECS)
'''
configuration += "\n\nPROTOCOL_CONFIG_KEYS = " + repr(
    assigned_uppercase_names(configuration)
) + "\n"


installation = STAGE37.installation
setup = rename(STAGE37.setup)
setup = setup.replace("stage37_1_spscd", "stage37_1_hmoc")

stage371_helpers = ["select_horizon_control_candidate", "Stage371Gates", "derive_stage371_decision"]
analysis_helpers = STAGE37.analysis_helpers + "\n\n" + function_sources(
    NUMERICAL.read_text(), stage371_helpers
)
analysis_helpers = analysis_helpers.replace(
    "class Stage371Gates:\n", "@dataclass(frozen=True)\nclass Stage371Gates:\n"
)

model_helpers = rename(STAGE37.model_helpers)
design_and_runtime_helpers = rename(STAGE37.design_and_runtime_helpers)
physical_truth = rename(STAGE37.physical_truth)

selection = rename(STAGE37.simulator_control)
selection = selection.replace(
    "# Prove the operator class on true Markov state before loading JEPA-WM.",
    "# Select a horizon-matched true-state operator without loading JEPA-WM.",
)
selection = selection.replace("SIMULATOR_PREFLIGHT_PASSED", "DEVELOPMENT_PREFLIGHT_PASSED")
selection = selection.replace("SELECTED_SIMULATOR_CONTROL", "SELECTED_HORIZON_CONTROL")
selection = selection.replace("SIMULATOR_SELECTION_ROWS", "CONTROL_SELECTION_ROWS")
selection = selection.replace("simulator_control_score", "horizon_control_score")
selection = selection.replace("SIMULATOR_LATENT_DIMS", "SIMULATOR_LATENT_DIMS")
selection = selection.replace("ACTIVE_SIMULATOR_CANDIDATE_EPOCHS", "ACTIVE_CANDIDATE_EPOCHS")
selection = selection.replace("PSCD_LEARNING_RATE", "LEARNING_RATE")
selection = selection.replace("semigroup_weight=1.0", "semigroup_weight=SEMIGROUP_WEIGHT")
selection = selection.replace("MAX_SIMULATOR_PREFLIGHT_NMSE", "MAX_DEVELOPMENT_PHYSICAL_NMSE")
selection = selection.replace("MIN_SIMULATOR_CONTROL_GAIN", "MIN_CONTROL_GAIN")
selection = selection.replace(
    '''CONTROL_SELECTION_ROWS.sort(key=lambda row: (
            float(row["validation_score"]), int(row["latent_dim"]), str(row["dynamics"])
        ))
        SELECTED_HORIZON_CONTROL = dict(CONTROL_SELECTION_ROWS[0])''',
    "SELECTED_HORIZON_CONTROL = select_horizon_control_candidate(CONTROL_SELECTION_ROWS)",
)
if "CONTROL_SELECTION_ROWS.sort" in selection:
    raise RuntimeError("Stage 37.1 candidate selection replacement failed")
selection = selection.replace("simulator_control_selection_rows.csv", "horizon_control_selection_rows.csv")
selection = selection.replace("simulator_preflight.json", "development_preflight.json")
selection = selection.replace("jepa_loaded_before_decision", "jepa_loaded")
selection = selection.replace("failure_is_scientific_not_pipeline", "development_gate_is_scientific")
selection = selection.replace("simulator_operator_preflight", "horizon_operator_development_preflight")
selection = selection.replace("stage37_1_simulator_operator_preflight", "stage37_1_horizon_operator_preflight")
selection = selection.replace("simulator_preflight_passed", "development_preflight_passed")


calibration = r'''# Freeze primary and matched horizon controls before locked evaluation.
PRIMARY_CONTROL = None
ZERO_SEMIGROUP_CONTROL = None
ONE_STEP_CONTROL = None
PHYSICAL_SCALE = None
PRIMARY_STATE_SCALE = None
ZERO_STATE_SCALE = None
CALIBRATION = None
EVALUATION_OPENED = False


def concatenate_stage371_sequences(*bundles):
    return {
        key: np.concatenate([bundle[key] for bundle in bundles], axis=0)
        for key in bundles[0]
    }


def flatten_stage371_models(models):
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
            raise TypeError(f"unsupported frozen model value at {prefix}: {type(value)}")
    visit("", models)
    return arrays, metadata


if not PIPELINE_FAILED and DEVELOPMENT_PREFLIGHT_PASSED:
    try:
        verify_executed_notebook_through(
            "# Freeze primary and matched horizon controls before locked evaluation."
        )
        construction = load_stage37_1_physical_sequences("construction")
        calibration_only = load_stage37_1_physical_sequences("calibration")
        CALIBRATION = concatenate_stage371_sequences(construction, calibration_only)
        common = {
            "history_length": 1,
            "latent_dim": int(SELECTED_HORIZON_CONTROL["latent_dim"]),
            "dynamics": str(SELECTED_HORIZON_CONTROL["dynamics"]),
            "epochs": ACTIVE_FINAL_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "semigroup_horizons": SEMIGROUP_HORIZONS,
        }
        matched_seed = stable_seed(CALIBRATION_SEED, "matched_horizon_controls")
        PRIMARY_CONTROL = fit_semigroup_predictive_state_closure(
            CALIBRATION["initial"], CALIBRATION["actions"], CALIBRATION["targets"],
            CALIBRATION["targets"], CALIBRATION["mask"], seed=matched_seed,
            semigroup_weight=SEMIGROUP_WEIGHT, free_weight=1.0, **common,
        )
        ZERO_SEMIGROUP_CONTROL = fit_semigroup_predictive_state_closure(
            CALIBRATION["initial"], CALIBRATION["actions"], CALIBRATION["targets"],
            CALIBRATION["targets"], CALIBRATION["mask"], seed=matched_seed,
            semigroup_weight=0.0, free_weight=1.0, **common,
        )
        ONE_STEP_CONTROL = fit_semigroup_predictive_state_closure(
            CALIBRATION["initial"], CALIBRATION["actions"], CALIBRATION["targets"],
            CALIBRATION["targets"], CALIBRATION["mask"], seed=matched_seed,
            semigroup_weight=0.0, free_weight=0.0, **common,
        )
        valid = CALIBRATION["mask"]
        PHYSICAL_SCALE = np.maximum(
            np.std(CALIBRATION["targets"][valid], axis=0, ddof=1), 1e-8
        )
        primary_cal = rollout_predictive_state_closure(
            PRIMARY_CONTROL, CALIBRATION["initial"], CALIBRATION["actions"],
            CALIBRATION["targets"], CALIBRATION["mask"],
        )
        zero_cal = rollout_predictive_state_closure(
            ZERO_SEMIGROUP_CONTROL, CALIBRATION["initial"], CALIBRATION["actions"],
            CALIBRATION["targets"], CALIBRATION["mask"],
        )
        PRIMARY_STATE_SCALE = np.maximum(
            np.std(primary_cal["direct_state"][valid], axis=0, ddof=1), 1e-8
        )
        ZERO_STATE_SCALE = np.maximum(
            np.std(zero_cal["direct_state"][valid], axis=0, ddof=1), 1e-8
        )
        arrays, metadata = flatten_stage371_models({
            "primary": PRIMARY_CONTROL, "zero_semigroup": ZERO_SEMIGROUP_CONTROL,
            "one_step": ONE_STEP_CONTROL, "physical_scale": PHYSICAL_SCALE,
            "primary_state_scale": PRIMARY_STATE_SCALE,
            "zero_state_scale": ZERO_STATE_SCALE,
        })
        model_path = CALIBRATION_MODEL_DIR / "frozen_stage37_1_controls.npz"
        atomic_npz(model_path, **arrays)
        write_json(CALIBRATION_MODEL_DIR / "frozen_stage37_1_controls_schema.json", metadata)
        certificate_path = CALIBRATION_MODEL_DIR / "evaluation_open_certificate.json"
        write_json(certificate_path, {
            "protocol_id": PROTOCOL_ID, "run_signature": RUN_SIGNATURE,
            "models_sha256": sha256_file(model_path),
            "selected_control": SELECTED_HORIZON_CONTROL,
            "evaluation_statistics_read": False, "jepa_loaded": False,
        })
        write_digest_sidecar(certificate_path)
        atomic_checkpoint("horizon_controls_frozen", {
            "certificate_sha256": sha256_file(certificate_path),
            "models_sha256": sha256_file(model_path),
        })
        print(json.dumps({
            "selected_control": SELECTED_HORIZON_CONTROL,
            "calibration_sequences": len(CALIBRATION["word"]),
            "evaluation_opened": EVALUATION_OPENED,
        }, indent=2))
    except Exception:
        record_failure("stage37_1_horizon_control_freeze")
'''


locked_evaluation = r'''# Open fresh horizon-matched evaluation once and derive every Stage 37.1 gate.
DECISION_PAYLOAD = {
    "status": (
        "INCONCLUSIVE_PIPELINE_FAILURE" if PIPELINE_FAILED
        else "operator_failed_development_preflight"
        if not DEVELOPMENT_PREFLIGHT_PASSED
        else "INCONCLUSIVE_PIPELINE_FAILURE"
    ),
    "passed": False,
}
EVALUATION_ROWS = []
SUMMARY = {}
if not PIPELINE_FAILED and not DEVELOPMENT_PREFLIGHT_PASSED:
    DECISION_PAYLOAD.update({
        "protocol_id": PROTOCOL_ID, "run_signature": RUN_SIGNATURE,
        "first_failed_gate": "development_preflight",
        "selected_control": SELECTED_HORIZON_CONTROL,
        "jepa_loaded": False, "evaluation_opened": False,
    })
    write_json(OUT / "stage37_1_decision.json", DECISION_PAYLOAD)

if not PIPELINE_FAILED and DEVELOPMENT_PREFLIGHT_PASSED:
    try:
        verify_executed_notebook_through(
            "# Open fresh horizon-matched evaluation once and derive every Stage 37.1 gate."
        )
        certificate_path = CALIBRATION_MODEL_DIR / "evaluation_open_certificate.json"
        validate_digest_sidecar(certificate_path)
        evaluation = load_stage37_1_physical_sequences("evaluation")
        EVALUATION_OPENED = True
        primary = rollout_predictive_state_closure(
            PRIMARY_CONTROL, evaluation["initial"], evaluation["actions"],
            evaluation["targets"], evaluation["mask"],
        )
        zero = rollout_predictive_state_closure(
            ZERO_SEMIGROUP_CONTROL, evaluation["initial"], evaluation["actions"],
            evaluation["targets"], evaluation["mask"],
        )
        one_step = rollout_predictive_state_closure(
            ONE_STEP_CONTROL, evaluation["initial"], evaluation["actions"],
            evaluation["targets"], evaluation["mask"],
        )
        valid = primary["evaluation_mask"]
        persistence = np.repeat(
            evaluation["initial"][:, None, :], MAX_WORD_LENGTH, axis=1
        )
        primary_error = scaled_path_mse(
            primary["physical"], evaluation["targets"], valid, PHYSICAL_SCALE
        )
        zero_error = scaled_path_mse(
            zero["physical"], evaluation["targets"], valid, PHYSICAL_SCALE
        )
        one_step_error = scaled_path_mse(
            one_step["physical"], evaluation["targets"], valid, PHYSICAL_SCALE
        )
        persistence_error = scaled_path_mse(
            persistence, evaluation["targets"], valid, PHYSICAL_SCALE
        )
        primary_semigroup = scaled_path_mse(
            primary["state"], primary["direct_state"], valid,
            PRIMARY_STATE_SCALE, final_only=False,
        )
        zero_semigroup = scaled_path_mse(
            zero["state"], zero["direct_state"], valid,
            ZERO_STATE_SCALE, final_only=False,
        )
        persistence_gain = relative_gain(primary_error, persistence_error)
        semigroup_gain = relative_gain(primary_semigroup, zero_semigroup)
        one_step_gain = relative_gain(primary_error, one_step_error)
        groups = evaluation["group"]
        persistence_ci = clustered_mean_interval(
            persistence_gain, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
            seed=stable_seed(BOOTSTRAP_SEED, "persistence"),
        )
        semigroup_ci = clustered_mean_interval(
            semigroup_gain, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
            seed=stable_seed(BOOTSTRAP_SEED, "semigroup"),
        )
        one_step_ci = clustered_mean_interval(
            one_step_gain, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
            seed=stable_seed(BOOTSTRAP_SEED, "one_step"),
        )
        length_nmse = {
            str(length): float(np.mean(primary_error[evaluation["length"] == length]))
            for length in sorted(set(evaluation["length"].tolist()))
        }
        mode_nmse = {
            mode: float(np.mean(primary_error[evaluation["initial_mode"] == mode]))
            for mode in MODE_LABELS
        }
        pretrained_forward_count = sum(
            int(PROVENANCE_COUNTS[key][model])
            for key in [
                "model_output_contract_preflights", "model_record_forwards",
                "native_forward_pred_calls", "native_predicted_word_sequences",
            ]
            for model in ["jepa", "dino"]
        )
        source_gate = bool(
            SOURCE_IDENTITY.get("confirmation_eligible", False)
            and EVALUATION_OPENED
            and len(set(groups.tolist())) >= MIN_EVALUATION_TRAJECTORIES
            and pretrained_forward_count == 0
            and int(PROVENANCE_COUNTS["patched_forwards"]) == 0
        )
        physical_gate = bool(
            np.mean(primary_error) <= MAX_LOCKED_PHYSICAL_NMSE
            and np.mean(persistence_gain) >= MIN_CONTROL_GAIN
            and persistence_ci[0] > 0
        )
        semigroup_gate = bool(np.mean(primary_semigroup) <= MAX_LOCKED_SEMIGROUP_NMSE)
        specificity_gate = bool(
            np.mean(semigroup_gain) >= MIN_OBJECTIVE_ADVANTAGE and semigroup_ci[0] > 0
            and np.mean(one_step_gain) >= MIN_OBJECTIVE_ADVANTAGE and one_step_ci[0] > 0
        )
        length_gate = bool(all(value <= MAX_LENGTH_PHYSICAL_NMSE for value in length_nmse.values()))
        mode_gate = bool(all(value <= MAX_MODE_PHYSICAL_NMSE for value in mode_nmse.values()))
        decision = derive_stage371_decision(Stage371Gates(
            source_and_split_binding=source_gate,
            development_preflight=DEVELOPMENT_PREFLIGHT_PASSED,
            locked_physical_closure=physical_gate,
            locked_semigroup_closure=semigroup_gate,
            objective_specificity=specificity_gate,
            horizon_family_consistency=length_gate,
            mode_family_consistency=mode_gate,
        ), run_mode=RUN_MODE)
        SUMMARY = {
            "selected_control": SELECTED_HORIZON_CONTROL,
            "primary_physical_nmse": float(np.mean(primary_error)),
            "zero_semigroup_physical_nmse": float(np.mean(zero_error)),
            "one_step_physical_nmse": float(np.mean(one_step_error)),
            "persistence_physical_nmse": float(np.mean(persistence_error)),
            "persistence_gain": float(np.mean(persistence_gain)),
            "persistence_gain_ci95": persistence_ci,
            "primary_semigroup_nmse": float(np.mean(primary_semigroup)),
            "zero_semigroup_nmse": float(np.mean(zero_semigroup)),
            "semigroup_advantage": float(np.mean(semigroup_gain)),
            "semigroup_advantage_ci95": semigroup_ci,
            "one_step_advantage": float(np.mean(one_step_gain)),
            "one_step_advantage_ci95": one_step_ci,
            "length_nmse": length_nmse, "mode_nmse": mode_nmse,
        }
        DECISION_PAYLOAD = {
            **decision, "protocol_id": PROTOCOL_ID, "run_signature": RUN_SIGNATURE,
            "source_commit": SOURCE_IDENTITY.get("resolved_commit"),
            "summary": SUMMARY,
            "claim_boundary": {
                "simulator_only": True, "jepa_loaded": False,
                "same_horizon_distribution": True,
                "trajectory_disjoint": True, "exact_word_disjoint": True,
                "jepa_result_claimed": False, "planning_result_claimed": False,
            },
        }
        for index in range(len(evaluation["word"])):
            EVALUATION_ROWS.append({
                "record_id": int(evaluation["record_id"][index]),
                "trajectory_id": int(groups[index]),
                "initial_mode": str(evaluation["initial_mode"][index]),
                "word": str(evaluation["word"][index]),
                "word_length": int(evaluation["length"][index]),
                "primary_physical_mse": float(primary_error[index]),
                "zero_semigroup_physical_mse": float(zero_error[index]),
                "one_step_physical_mse": float(one_step_error[index]),
                "persistence_physical_mse": float(persistence_error[index]),
                "primary_semigroup_mse": float(primary_semigroup[index]),
                "zero_semigroup_mse": float(zero_semigroup[index]),
            })
        write_csv(EVIDENCE_DIR / "locked_horizon_control_rows.csv", EVALUATION_ROWS)
        write_json(EVIDENCE_DIR / "horizon_control_summary.json", SUMMARY)
        write_json(OUT / "stage37_1_decision.json", DECISION_PAYLOAD)
        atomic_checkpoint("locked_horizon_control_evaluation_complete", {
            "decision_sha256": sha256_file(OUT / "stage37_1_decision.json"),
            "status": DECISION_PAYLOAD["status"], "rows": len(EVALUATION_ROWS),
        })
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].bar(["persistence", "one-step", "zero-SG", "primary"], [
            np.mean(persistence_error), np.mean(one_step_error),
            np.mean(zero_error), np.mean(primary_error),
        ], color=["#64748b", "#f59e0b", "#f97316", "#7c3aed"])
        axes[0].axhline(MAX_LOCKED_PHYSICAL_NMSE, color="black", linestyle="--")
        axes[0].set_title("Locked physical NMSE")
        axes[1].bar(["zero-SG", "primary"], [
            np.mean(zero_semigroup), np.mean(primary_semigroup),
        ], color=["#f97316", "#7c3aed"])
        axes[1].axhline(MAX_LOCKED_SEMIGROUP_NMSE, color="black", linestyle="--")
        axes[1].set_title("Locked semigroup NMSE")
        figure.suptitle(f"Stage 37.1: {DECISION_PAYLOAD['status']}")
        figure.tight_layout()
        figure.savefig(PLOT_DIR / "stage37_1_horizon_control.png", dpi=180)
        plt.close(figure)
        interpretation = f"""# Automatic Stage 37.1 interpretation

Status: **{DECISION_PAYLOAD['status'].upper()}**

The first failed gate is `{DECISION_PAYLOAD['first_failed_gate']}`. A full pass
calibrates the true-state operator class for a later fresh JEPA confirmation.
This simulator-only run contains no JEPA or planning result.
"""
        retry_drive_io(
            "write automatic interpretation",
            lambda: (OUT / "AUTOMATIC_INTERPRETATION.md").write_text(interpretation),
        )
        print(json.dumps(DECISION_PAYLOAD, indent=2))
    except Exception:
        record_failure("stage37_1_locked_horizon_control_evaluation")
'''


packaging = rename(STAGE37.packaging)
packaging = packaging.replace("stage37_1_spscd_result_bundle_", "stage37_1_hmoc_result_bundle_")
packaging = packaging.replace("raw_roots = [TRUTH_DIR, PATH_DIR]", "raw_roots = [TRUTH_DIR]")

protocol_sources = [
    introduction, configuration, installation, setup, analysis_helpers,
    model_helpers, design_and_runtime_helpers, physical_truth, selection,
    calibration, locked_evaluation, packaging,
]
protocol_sources = [value.strip() for value in protocol_sources]
protocol_digest = hashlib.sha256(
    json.dumps(protocol_sources, ensure_ascii=False, separators=(",", ":")).encode()
).hexdigest()
configuration = configuration.replace("__PROTOCOL_DIGEST__", protocol_digest)
if "__PROTOCOL_DIGEST__" in configuration:
    raise RuntimeError("Stage 37.1 protocol digest placeholder was not replaced")

cells = [
    markdown(introduction), code(configuration), code(installation), code(setup),
    code(analysis_helpers), code(model_helpers), code(design_and_runtime_helpers),
    code(physical_truth), code(selection), code(calibration),
    code(locked_evaluation), code(packaging),
]
for index, cell in enumerate(cells):
    cell["id"] = f"stage371-{index:02d}"

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
