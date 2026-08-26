"""Build the sealed fresh PushT coefficient-matched replication Colab."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
TARGET = ROOT / "39_fresh_coefficient_matched_replication.ipynb"
NUMERICAL = REPOSITORY / "src/cf_faithfulness/stage39_replication.py"


def load_builder(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BASE = load_builder(
    ROOT / "build_stage38_cross_model_pscd_notebook.py", "stage38_builder_for_stage39"
)

code = BASE.code
markdown = BASE.markdown
replace_assignment = BASE.replace_assignment
replace_block = BASE.replace_block
assigned_uppercase_names = BASE.assigned_uppercase_names
function_sources = BASE.function_sources


def rename(value: str) -> str:
    for old, new in [
        ("Stage 38", "Stage 39"),
        ("STAGE38", "STAGE39"),
        ("stage38", "stage39"),
    ]:
        value = value.replace(old, new)
    return value


introduction = r'''# Stage 39: fresh coefficient-matched replication

## Frozen decision before computation

Stage 38.1 found near-zero full-minus-coefficient-matched gains on development
data.  Stage 39 is the first locked replication of that result.  It does not
reuse Stage 38 trajectories, action words, carrier shards, fitted adapters,
calibration outcomes, or evaluation rows.  The official frozen JEPA-WM and
DINO-WM PushT checkpoints are loaded sequentially and are never updated.

The primary estimand is the mean paired row-wise relative gain

\[
G_{row}=E[(e_{matched}-e_{full})/\max(e_{matched},10^{-12})],
\]

where positive values favor the full carrier/physical/latent composition
objective.  A hierarchical bootstrap resamples optimization seeds and complete
trajectory families while retaining their repeated records and words.  The
90% interval and the symmetric, prespecified practical-equivalence band
`[-5%, +5%]` implement an equivalence test; failure to reject a difference is
not called replication.  The pooled ratio of means is reported only as a
sensitivity estimand.

Three matched optimization seeds compare exactly two models: full S-PSCD and
a latent-only control whose outer weight is `0.45 * lambda_full`, making its
effective latent semigroup coefficient exactly equal to the full objective.
All architectures, initializations, data, updates, and epochs are otherwise
matched.  Native rollout, wrong-history evaluation, and an exact-state
simulator operator are calibration anchors, not alternative primary outcomes.

The two checkpoint panels receive separate decisions.  No cross-model pooling,
planning, event/reset extension, or Stage 38 threshold revision is allowed.
'''


configuration = rename(BASE.configuration)
for name, value in {
    "PROTOCOL_ID": '"stage39-fresh-coefficient-matched-replication-v1"',
    "NOTEBOOK_PROTOCOL_SHA256": '"__PROTOCOL_DIGEST__"',
    "EVIDENCE_STATUS": '"FRESH_PROSPECTIVE_LOCKED_REPLICATION"',
    "EXPERIMENT_NOTEBOOK_PATH": '"notebooks/39_fresh_coefficient_matched_replication.ipynb"',
    "EXPERIMENT_BUILDER_PATH": '"notebooks/build_stage39_fresh_coefficient_replication_notebook.py"',
    "EXPERIMENT_NUMERICAL_PATH": '"src/cf_faithfulness/stage39_replication.py"',
    "OUTPUT_DIR": '"/content/counterfactual_faithfulness_stage39_fcmr"',
    "DRIVE_OUTPUT_DIR": '"/content/drive/MyDrive/counterfactual_faithfulness_stage39_fcmr"',
    "RUN_REQUEST_PATH": '"/content/drive/MyDrive/counterfactual_faithfulness_stage39_fcmr/stage39_run_request.json"',
    "MAX_ESTIMATED_TOTAL_MINUTES": "720.0",
    "SEED": "390101",
    "DESIGN_SEED": "390141",
    "DECODER_SEED": "390183",
    "RANK_SEED": "390213",
    "CALIBRATION_SEED": "390253",
    "BOOTSTRAP_SEED": "390283",
    "CONTROL_SEED": "390351",
    "CONSTRUCTION_TRAJECTORY_POOL": "list(range(66000, 68000))",
    "MODEL_SELECTION_TRAJECTORY_POOL": "list(range(68000, 70000))",
    "CALIBRATION_TRAJECTORY_POOL": "list(range(70000, 72000))",
    "EVALUATION_TRAJECTORY_POOL": "list(range(72000, 78000))",
    "CONSTRUCTION_TRAJECTORIES": "24",
    "MODEL_SELECTION_TRAJECTORIES": "16",
    "CALIBRATION_TRAJECTORIES": "24",
    "EVALUATION_TRAJECTORIES": "48",
    "TASK_ID_OFFSET": "390000",
}.items():
    configuration = replace_assignment(configuration, name, value)

configuration = replace_block(
    configuration,
    "CANONICAL_RESPONSE_WORD_NAMES = [",
    "CALIBRATION_INTERCHANGE_PAIRS =",
    r'''CANONICAL_RESPONSE_WORD_NAMES = ["A", "B", "C", "D", "AB", "CD", "BA", "DC"]
CONSTRUCTION_WORD_NAMES = [
    "BDBAAADDC", "DCBDDCDDA", "BAAAAADACA", "CDBBAACAAB",
    "DBABCCDACDB", "DBBADCCCDAC", "ADBABDDCBACD", "CBDBCBBDACAC",
]
MODEL_SELECTION_WORD_NAMES = [
    "DCBDACBBC", "BAABCAABB", "BBDBACADAA", "CCCBCDDAAD",
    "BAAACDABAAC", "DBBBADCDDCC", "ADAABDCDDCCA", "CDCBCBCBDBBA",
]
CALIBRATION_WORD_NAMES = [
    "BCDDCBBBA", "BCBBBCDCA", "BAADAABBDB", "ACBCBBCCAD",
    "CBDDDCABAAD", "BCDADCDCBDD", "BDBCBDBCCBBD", "BDCBDBAAADBB",
]
STAGE39_CORE_WORD_NAMES = sorted(
    set(
        CANONICAL_RESPONSE_WORD_NAMES + CONSTRUCTION_WORD_NAMES
        + MODEL_SELECTION_WORD_NAMES + CALIBRATION_WORD_NAMES
    ),
    key=lambda value: (len(value), value),
)
CORE_WORD_SPECS = [stage39_word_spec(name) for name in STAGE39_CORE_WORD_NAMES]
''',
)
configuration = replace_block(
    configuration,
    "CLOSURE_EVALUATION_WORD_NAMES = [",
    "EVALUATION_INTERCHANGE_PAIRS =",
    r'''CLOSURE_EVALUATION_WORD_NAMES = [
    "CADBAACAA", "ABDABCDAB", "DDABCADDDB", "CDBDADBADC",
    "DBDBCACDCAD", "AADDBDBADAA", "ADBCBBCCCADC", "DACACBBCADBB",
]
PLANNING_WORD_NAMES = []
EVALUATION_WORD_NAMES_REGISTERED = list(CLOSURE_EVALUATION_WORD_NAMES)
EVALUATION_WORD_SPECS = [
    stage39_word_spec(name) for name in EVALUATION_WORD_NAMES_REGISTERED
]
''',
)
configuration = configuration.replace(
    "assert len(PLANNING_WORD_NAMES) == 12\nassert {len(word) for word in PLANNING_WORD_NAMES} == {10}",
    "assert PLANNING_WORD_NAMES == []",
)
configuration = configuration.replace(
    '"planning_opened_only_after_both_closure_pass",',
    '"planning_permanently_sealed",\n    "fixed_plus_or_minus_five_percent_equivalence_band",',
)
configuration = configuration.replace(
    '"fresh_trajectory_ids_56000_to_65999",',
    '"fresh_trajectory_ids_66000_to_77999",',
)
configuration = configuration.replace(
    'FINAL_TRAINING_SEEDS = [3801, 3802, 3803] if RUN_MODE == "pilot" else [3801, 3802]',
    'FINAL_TRAINING_SEEDS = [3901, 3902, 3903] if RUN_MODE == "pilot" else [3901, 3902]',
)
configuration += r'''

EQUIVALENCE_MARGIN = 0.05
PRIMARY_CONFIDENCE = 0.90
COEFFICIENT_MATCH_FACTOR = 0.45
PRIMARY_VARIANTS = ["coefficient_matched", "full"]
assert EQUIVALENCE_MARGIN == 0.05
assert PRIMARY_CONFIDENCE == 0.90
assert COEFFICIENT_MATCH_FACTOR == FULL_SEMIGROUP_COMPONENT_WEIGHTS[2]
'''
configuration = re.sub(
    r"\n\nPROTOCOL_CONFIG_KEYS = .*?\n(?=import subprocess)", "\n", configuration, flags=re.S
)
configuration += "\n\nPROTOCOL_CONFIG_KEYS = " + repr(
    assigned_uppercase_names(configuration)
) + "\n"

installation = BASE.installation
setup = rename(BASE.setup).replace("stage39_xmpscd", "stage39_fcmr")

analysis_helpers = rename(BASE.analysis_helpers)
statistics = function_sources(
    NUMERICAL.read_text(),
    [
        "paired_rowwise_relative_gain",
        "pooled_ratio_of_means_gain",
        "hierarchical_seed_family_interval",
        "Stage39PanelDecision",
        "derive_stage39_panel_decision",
        "derive_stage39_decision",
    ],
)
statistics = statistics.replace(
    "class Stage39PanelDecision:\n", "@dataclass(frozen=True)\nclass Stage39PanelDecision:\n"
)
analysis_helpers += "\n\n" + statistics

model_helpers = rename(BASE.model_helpers)
design_and_runtime_helpers = rename(BASE.design_and_runtime_helpers)
physical_truth = rename(BASE.physical_truth)
simulator_preflight = rename(BASE.simulator_preflight)
construction_and_paths = rename(BASE.construction_and_paths)
data_and_selection = rename(BASE.data_and_selection)


calibration = r'''# Freeze matched full and coefficient-matched models before evaluation.
FROZEN_MODELS = {}
STATE_SCALES = {}
PHYSICAL_SCALES = {}
CARRIER_SCALES = {}
SIMULATOR_FINAL = None
EVALUATION_OPENED = False


def stage39_artifact_paths(short, variant, seed):
    stem = CALIBRATION_MODEL_DIR / f"stage39_{short}_{variant}_seed{int(seed)}"
    return Path(str(stem) + ".npz"), Path(str(stem) + "_schema.json")


def stage39_encode_artifact(value, arrays, prefix="root"):
    if isinstance(value, np.ndarray):
        key = f"array_{len(arrays):05d}"
        arrays[key] = value
        return {"kind": "array", "key": key}
    if isinstance(value, dict):
        return {"kind": "dict", "items": {
            str(key): stage39_encode_artifact(item, arrays, f"{prefix}.{key}")
            for key, item in sorted(value.items())
        }}
    if isinstance(value, (list, tuple)):
        return {"kind": "list", "items": [
            stage39_encode_artifact(item, arrays, f"{prefix}.{index}")
            for index, item in enumerate(value)
        ]}
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return {"kind": "scalar", "value": value}
    raise TypeError(f"unsupported Stage 39 artifact at {prefix}: {type(value)}")


def stage39_decode_artifact(node, arrays):
    if node["kind"] == "array":
        return np.asarray(arrays[str(node["key"])])
    if node["kind"] == "dict":
        return {key: stage39_decode_artifact(value, arrays) for key, value in node["items"].items()}
    if node["kind"] == "list":
        return [stage39_decode_artifact(value, arrays) for value in node["items"]]
    if node["kind"] == "scalar":
        return node["value"]
    raise ValueError(f"unknown Stage 39 artifact node {node['kind']!r}")


def save_stage39_artifact(short, variant, seed, artifact):
    array_path, schema_path = stage39_artifact_paths(short, variant, seed)
    arrays = {}
    schema = stage39_encode_artifact(artifact, arrays)
    atomic_npz(array_path, **arrays)
    write_json(schema_path, {
        "protocol_id": PROTOCOL_ID, "run_signature": RUN_SIGNATURE,
        "model": str(short), "variant": str(variant), "seed": int(seed),
        "array_sha256": sha256_file(array_path), "schema": schema,
    })
    write_digest_sidecar(schema_path)


def load_stage39_artifact(short, variant, seed):
    array_path, schema_path = stage39_artifact_paths(short, variant, seed)
    validate_digest_sidecar(array_path)
    validate_digest_sidecar(schema_path)
    metadata = json.loads(schema_path.read_text())
    expected = (PROTOCOL_ID, RUN_SIGNATURE, str(short), str(variant), int(seed))
    observed = (
        metadata["protocol_id"], metadata["run_signature"], metadata["model"],
        metadata["variant"], int(metadata["seed"]),
    )
    if observed != expected or metadata["array_sha256"] != sha256_file(array_path):
        raise RuntimeError(f"Stage 39 artifact binding failed for {short}/{variant}/{seed}")
    with np.load(array_path, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    return stage39_decode_artifact(metadata["schema"], arrays)


def stage39_variant_configuration(short, variant):
    selected = float(SELECTED_SEMIGROUP[short]["semigroup_weight"])
    table = {
        "coefficient_matched": {
            "free_weight": 1.0,
            "semigroup_weight": COEFFICIENT_MATCH_FACTOR * selected,
            "semigroup_component_weights": OVERSHOOT_COMPONENT_WEIGHTS,
        },
        "full": {
            "free_weight": 1.0,
            "semigroup_weight": selected,
            "semigroup_component_weights": FULL_SEMIGROUP_COMPONENT_WEIGHTS,
        },
    }
    if str(variant) not in table:
        raise KeyError(f"unknown Stage 39 variant {variant!r}")
    return table[str(variant)]


def effective_latent_coefficient(objective):
    weights = np.asarray(objective["semigroup_component_weights"], dtype=np.float64)
    return float(objective["semigroup_weight"] * weights[2] / np.sum(weights))


def fit_or_load_stage39_model(short, variant, seed, data):
    array_path, schema_path = stage39_artifact_paths(short, variant, seed)
    sidecars = [Path(str(array_path) + ".sha256"), Path(str(schema_path) + ".sha256")]
    if array_path.is_file() and schema_path.is_file() and all(path.is_file() for path in sidecars):
        PROVENANCE_COUNTS["validated_cache_hits"] += 1
        return load_stage39_artifact(short, variant, seed)
    objective = stage39_variant_configuration(short, variant)
    artifact = fit_weighted_semigroup_predictive_state_closure(
        data["initial_carrier"], data["actions"], data["carrier"],
        data["native"], data["mask"], history_length=FIXED_HISTORY_LENGTH,
        latent_dim=FIXED_LATENT_DIM, dynamics=FIXED_DYNAMICS,
        epochs=ACTIVE_FINAL_EPOCHS, learning_rate=PSCD_LEARNING_RATE,
        seed=int(seed), semigroup_horizons=SEMIGROUP_HORIZONS, **objective,
    )
    save_stage39_artifact(short, variant, seed, artifact)
    return artifact


if not PIPELINE_FAILED and SIMULATOR_PREFLIGHT_PASSED:
    try:
        verify_executed_notebook_through(
            "# Freeze matched full and coefficient-matched models before evaluation."
        )
        if set(SELECTED_SEMIGROUP) != {"jepa", "dino"}:
            raise RuntimeError("both Stage 39 semigroup selections must be frozen")
        model_manifest, coefficient_receipts = [], []
        for short in ["jepa", "dino"]:
            construction = load_stage39_sequences(short, "construction")
            calibration_only = load_stage39_sequences(short, "calibration")
            data = concatenate_stage39_sequences(construction, calibration_only)
            valid = data["mask"]
            PHYSICAL_SCALES[short] = np.maximum(
                np.std(data["simulator"][valid], axis=0, ddof=1), 1e-8
            )
            CARRIER_SCALES[short] = np.maximum(
                np.std(data["carrier"][valid], axis=0, ddof=1), 1e-8
            )
            FROZEN_MODELS[short], STATE_SCALES[short] = {}, {}
            objectives = {
                variant: stage39_variant_configuration(short, variant)
                for variant in PRIMARY_VARIANTS
            }
            full_coefficient = effective_latent_coefficient(objectives["full"])
            matched_coefficient = effective_latent_coefficient(objectives["coefficient_matched"])
            if not np.isclose(full_coefficient, matched_coefficient, atol=1e-15, rtol=0):
                raise RuntimeError("effective latent coefficients are not exactly matched")
            coefficient_receipts.append({
                "model": short, "selected_full_outer_weight": float(
                    SELECTED_SEMIGROUP[short]["semigroup_weight"]
                ),
                "full_effective_latent_coefficient": full_coefficient,
                "matched_effective_latent_coefficient": matched_coefficient,
                "coefficients_equal": True,
            })
            for variant in PRIMARY_VARIANTS:
                FROZEN_MODELS[short][variant], STATE_SCALES[short][variant] = {}, {}
                for seed in FINAL_TRAINING_SEEDS:
                    artifact = fit_or_load_stage39_model(short, variant, seed, data)
                    expected = objectives[variant]
                    if (
                        int(artifact["config"]["seed"]) != int(seed)
                        or float(artifact["config"]["semigroup_weight"])
                        != float(expected["semigroup_weight"])
                        or list(artifact["config"]["semigroup_component_weights"])
                        != list(expected["semigroup_component_weights"])
                    ):
                        raise RuntimeError(f"matched binding failed for {short}/{variant}/{seed}")
                    FROZEN_MODELS[short][variant][int(seed)] = artifact
                    rollout = rollout_predictive_state_closure(
                        artifact, data["initial_carrier"], data["actions"],
                        data["carrier"], data["mask"],
                    )
                    evaluated = rollout["evaluation_mask"]
                    STATE_SCALES[short][variant][int(seed)] = np.maximum(
                        np.std(rollout["direct_state"][evaluated], axis=0, ddof=1), 1e-8
                    )
                    array_path, schema_path = stage39_artifact_paths(short, variant, seed)
                    model_manifest.append({
                        "model": short, "variant": variant, "seed": int(seed),
                        "array_sha256": sha256_file(array_path),
                        "schema_sha256": sha256_file(schema_path),
                    })
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        physical = concatenate_stage39_sequences(
            load_stage39_physical_sequences("construction"),
            load_stage39_physical_sequences("calibration"),
        )
        simulator_seed = stable_seed(CONTROL_SEED, "stage39_simulator_final")
        simulator_array, simulator_schema = stage39_artifact_paths(
            "simulator", "exact_state", simulator_seed
        )
        if all(path.is_file() for path in [
            simulator_array, simulator_schema, Path(str(simulator_array) + ".sha256"),
            Path(str(simulator_schema) + ".sha256"),
        ]):
            SIMULATOR_FINAL = load_stage39_artifact("simulator", "exact_state", simulator_seed)
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
            save_stage39_artifact("simulator", "exact_state", simulator_seed, SIMULATOR_FINAL)

        scale_path = CALIBRATION_MODEL_DIR / "stage39_frozen_scales.npz"
        scale_arrays = {}
        for short in ["jepa", "dino"]:
            scale_arrays[f"physical_{short}"] = PHYSICAL_SCALES[short]
            scale_arrays[f"carrier_{short}"] = CARRIER_SCALES[short]
            for variant in PRIMARY_VARIANTS:
                for seed in FINAL_TRAINING_SEEDS:
                    scale_arrays[f"state_{short}_{variant}_{seed}"] = (
                        STATE_SCALES[short][variant][int(seed)]
                    )
        atomic_npz(scale_path, **scale_arrays)
        write_json(EVIDENCE_DIR / "coefficient_match_receipts.json", coefficient_receipts)
        certificate_path = CALIBRATION_MODEL_DIR / "evaluation_open_certificate.json"
        write_json(certificate_path, {
            "protocol_id": PROTOCOL_ID, "run_signature": RUN_SIGNATURE,
            "source_commit": SOURCE_IDENTITY.get("resolved_commit"),
            "models": model_manifest, "coefficient_receipts": coefficient_receipts,
            "scale_sha256": sha256_file(scale_path),
            "final_training_seeds": list(map(int, FINAL_TRAINING_SEEDS)),
            "matched_seed_architecture_data_epochs": True,
            "evaluation_statistics_read": False, "evaluation_metrics_computed": False,
            "planning_permanently_sealed": True,
            "checkpoint_parameters_updated": False,
        })
        write_digest_sidecar(certificate_path)
        atomic_checkpoint("stage39_models_frozen", {
            "certificate_sha256": sha256_file(certificate_path),
            "evaluation_opened": False, "planning_opened": False,
        })
        print(json.dumps({
            "selected_semigroup": SELECTED_SEMIGROUP,
            "final_training_seeds": FINAL_TRAINING_SEEDS,
            "coefficient_receipts": coefficient_receipts,
        }, indent=2))
    except Exception:
        record_failure("stage39_calibration_model_freeze")
'''


locked_evaluation = r'''# Open the locked replication panel; planning remains sealed.
DECISION_PAYLOAD = {
    "status": "INCONCLUSIVE_PIPELINE_FAILURE",
    "passed": False,
    "planning_opened": False,
}
EVALUATION_ROWS = []
SUMMARY = {}
PANEL_DECISIONS = {}


def stage39_terminal_labels(labels, mask):
    values = np.asarray(labels).astype(str)
    valid = np.asarray(mask, dtype=bool)
    index = np.max(np.where(valid, np.arange(valid.shape[1])[None, :], -1), axis=1)
    if np.any(index < 0):
        raise ValueError("each label path needs an endpoint")
    return values[np.arange(len(values)), index]


def stage39_replication_panel(short, closure, quality_control_passed):
    scale = PHYSICAL_SCALES[short]
    groups = closure["group"]
    if len(np.unique(groups)) < MIN_EVALUATION_TRAJECTORIES:
        raise RuntimeError("locked replication has too few trajectory families")
    masks, physical = {}, {variant: [] for variant in PRIMARY_VARIANTS}
    false_history_errors = []
    native_history = history_tensor(
        closure["initial_carrier"], closure["carrier"], closure["mask"],
        FIXED_HISTORY_LENGTH,
    )
    for seed in FINAL_TRAINING_SEEDS:
        for variant in PRIMARY_VARIANTS:
            artifact = FROZEN_MODELS[short][variant][int(seed)]
            rollout = rollout_predictive_state_closure(
                artifact, closure["initial_carrier"], closure["actions"],
                closure["carrier"], closure["mask"],
            )
            masks[variant] = rollout["evaluation_mask"]
            physical[variant].append(scaled_path_mse(
                rollout["physical"], closure["simulator"], masks[variant], scale,
            ))
        false_history = permute_past_history(
            native_history, groups, closure["mask"],
            seed=stable_seed(CONTROL_SEED, "stage39_wrong_history", short, int(seed)),
        )
        false_rollout = rollout_predictive_state_closure(
            FROZEN_MODELS[short]["full"][int(seed)],
            closure["initial_carrier"], closure["actions"], closure["carrier"],
            closure["mask"], histories_override=false_history,
        )
        false_history_errors.append(scaled_path_mse(
            false_rollout["physical"], closure["simulator"],
            false_rollout["evaluation_mask"], scale,
        ))
    physical = {key: np.stack(value, axis=0) for key, value in physical.items()}
    false_history_errors = np.stack(false_history_errors, axis=0)
    row_gain = paired_rowwise_relative_gain(
        physical["full"], physical["coefficient_matched"]
    )
    interval90 = hierarchical_seed_family_interval(
        row_gain, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
        seed=stable_seed(BOOTSTRAP_SEED, "stage39_primary", short),
        confidence=PRIMARY_CONFIDENCE,
    )
    mean_gain = float(np.mean(row_gain))
    decision = derive_stage39_panel_decision(
        mean_gain, interval90, equivalence_margin=EQUIVALENCE_MARGIN,
        quality_control_passed=quality_control_passed,
    )
    history_gain = paired_rowwise_relative_gain(
        physical["full"], false_history_errors
    )
    history_interval = hierarchical_seed_family_interval(
        history_gain, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
        seed=stable_seed(BOOTSTRAP_SEED, "stage39_history", short),
        confidence=PRIMARY_CONFIDENCE,
    )
    terminal_modes = stage39_terminal_labels(closure["target_mode"], closure["mask"])
    seed_summaries = []
    for seed_index, seed in enumerate(FINAL_TRAINING_SEEDS):
        full_error = physical["full"][seed_index]
        matched_error = physical["coefficient_matched"][seed_index]
        seed_summaries.append({
            "seed": int(seed),
            "full_mean_nmse": float(np.mean(full_error)),
            "matched_mean_nmse": float(np.mean(matched_error)),
            "mean_rowwise_gain": float(np.mean(row_gain[seed_index])),
            "pooled_ratio_of_means_gain": pooled_ratio_of_means_gain(
                full_error, matched_error
            ),
            "full_tail": tail_risk_summary(full_error),
            "matched_tail": tail_risk_summary(matched_error),
        })
        for row_index in range(len(groups)):
            EVALUATION_ROWS.append({
                "model": short, "seed": int(seed),
                "trajectory_id": int(groups[row_index]),
                "record_id": int(closure["record_id"][row_index]),
                "initial_mode": str(closure["initial_mode"][row_index]),
                "terminal_mode": str(terminal_modes[row_index]),
                "word": str(closure["word"][row_index]),
                "word_length": int(closure["length"][row_index]),
                "full_physical_nmse": float(full_error[row_index]),
                "coefficient_matched_physical_nmse": float(matched_error[row_index]),
                "full_minus_matched_rowwise_gain": float(row_gain[seed_index, row_index]),
                "wrong_history_physical_nmse": float(false_history_errors[seed_index, row_index]),
            })
    return decision, {
        "mean_rowwise_relative_gain": mean_gain,
        "hierarchical_interval90": list(interval90),
        "equivalence_margin": EQUIVALENCE_MARGIN,
        "pooled_ratio_of_means_gain": pooled_ratio_of_means_gain(
            physical["full"], physical["coefficient_matched"]
        ),
        "full_mean_nmse": float(np.mean(physical["full"])),
        "coefficient_matched_mean_nmse": float(np.mean(physical["coefficient_matched"])),
        "history_gain": float(np.mean(history_gain)),
        "history_interval90": list(history_interval),
        "seed_summaries": seed_summaries,
        "classification": decision.classification,
    }


if not PIPELINE_FAILED and SIMULATOR_PREFLIGHT_PASSED:
    try:
        verify_executed_notebook_through(
            "# Open the locked replication panel; planning remains sealed."
        )
        certificate_path = CALIBRATION_MODEL_DIR / "evaluation_open_certificate.json"
        validate_digest_sidecar(certificate_path)
        certificate = json.loads(certificate_path.read_text())
        if (
            certificate["protocol_id"] != PROTOCOL_ID
            or certificate["run_signature"] != RUN_SIGNATURE
            or certificate["evaluation_statistics_read"]
            or not certificate["planning_permanently_sealed"]
        ):
            raise RuntimeError("Stage 39 evaluation-open certificate is invalid")
        for model_name in MODEL_NAMES:
            bundle = load_world_model(model_name)
            short = bundle["short"]
            try:
                for index, record in enumerate(SELECTED_RECORDS["evaluation"]):
                    generate_stage39_path_record(
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
            short: load_stage39_sequences(short, "evaluation_closure")
            for short in ["jepa", "dino"]
        }
        reference = closure_data["jepa"]
        simulator_rollout = rollout_predictive_state_closure(
            SIMULATOR_FINAL, reference["initial_physical"], reference["actions"],
            reference["simulator"], reference["mask"],
        )
        simulator_error = scaled_path_mse(
            simulator_rollout["physical"], reference["simulator"],
            simulator_rollout["evaluation_mask"],
            SIMULATOR_FINAL["normalization"]["physical_scale"],
        )
        persistence = np.repeat(
            reference["initial_physical"][:, None, :], MAX_WORD_LENGTH, axis=1
        )
        persistence_error = scaled_path_mse(
            persistence, reference["simulator"], simulator_rollout["evaluation_mask"],
            SIMULATOR_FINAL["normalization"]["physical_scale"],
        )
        simulator_gain = float(np.mean(relative_gain(simulator_error, persistence_error)))
        simulator_passed = bool(
            np.mean(simulator_error) <= MAX_SIMULATOR_LOCKED_NMSE
            and simulator_gain >= MIN_SIMULATOR_GAIN
        )
        for short in ["jepa", "dino"]:
            panel, metrics = stage39_replication_panel(
                short, closure_data[short], simulator_passed
            )
            PANEL_DECISIONS[short] = panel
            SUMMARY[short] = metrics
        DECISION_PAYLOAD = derive_stage39_decision(PANEL_DECISIONS)
        DECISION_PAYLOAD.update({
            "protocol_id": PROTOCOL_ID, "run_signature": RUN_SIGNATURE,
            "evaluation_opened": True, "planning_opened": False,
            "simulator_control": {
                "passed": simulator_passed,
                "mean_nmse": float(np.mean(simulator_error)),
                "gain_over_persistence": simulator_gain,
            },
            "primary_estimand": "mean_paired_rowwise_relative_gain",
            "primary_confidence": PRIMARY_CONFIDENCE,
            "equivalence_band": [-EQUIVALENCE_MARGIN, EQUIVALENCE_MARGIN],
            "panels_pooled": False,
            "stage38_evidence_consumed": False,
        })
        write_csv(EVIDENCE_DIR / "locked_replication_rows.csv", EVALUATION_ROWS)
        write_json(EVIDENCE_DIR / "stage39_summary.json", SUMMARY)
        write_json(EVIDENCE_DIR / "stage39_panel_decisions.json", {
            short: {
                "mean_gain": panel.mean_gain,
                "interval90": list(panel.interval90),
                "equivalence_margin": panel.equivalence_margin,
                "quality_control_passed": panel.quality_control_passed,
                "classification": panel.classification,
            }
            for short, panel in PANEL_DECISIONS.items()
        })
        write_json(OUT / "stage39_decision.json", DECISION_PAYLOAD)
        atomic_checkpoint("stage39_locked_replication_complete", {
            "decision_sha256": sha256_file(OUT / "stage39_decision.json"),
            "status": DECISION_PAYLOAD["status"],
            "rows": len(EVALUATION_ROWS), "planning_opened": False,
        })

        figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        for axis, short in zip(axes, ["jepa", "dino"]):
            estimate = SUMMARY[short]["mean_rowwise_relative_gain"]
            low, high = SUMMARY[short]["hierarchical_interval90"]
            axis.errorbar([0], [estimate], yerr=[[estimate - low], [high - estimate]], fmt="o")
            axis.axhspan(-EQUIVALENCE_MARGIN, EQUIVALENCE_MARGIN, alpha=0.18, color="#0ea5e9")
            axis.axhline(0, color="black", linewidth=1)
            axis.set(xticks=[], ylabel="full minus matched relative gain", title=short.upper())
        figure.suptitle(f"Stage 39: {DECISION_PAYLOAD['status']}")
        figure.tight_layout()
        figure.savefig(PLOT_DIR / "stage39_equivalence_replication.png", dpi=180)
        plt.close(figure)
        interpretation = f"""# Automatic Stage 39 interpretation

Status: **{DECISION_PAYLOAD['status'].upper()}**

The decision uses separate JEPA and DINO 90% hierarchical intervals against
the fixed ±5% practical-equivalence band.  It never pools predictor panels.
Planning remained sealed.  A practical-equivalence pass replicates the bounded
claim that the extra carrier/physical composition terms did not add a
practically meaningful mean endpoint gain over equal latent pressure on this
fresh PushT bank.  It does not prove universal equivalence, native JEPA
closure, causal mechanism, or planning value.
"""
        (OUT / "AUTOMATIC_INTERPRETATION.md").write_text(interpretation)
        print(json.dumps(DECISION_PAYLOAD, indent=2))
    except Exception:
        record_failure("stage39_locked_replication")
'''

packaging = rename(BASE.packaging)
packaging = packaging.replace("stage39_xmpscd", "stage39_fcmr")
packaging = packaging.replace("cross_model_pscd_confirmation", "fresh_coefficient_matched_replication")

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
    raise RuntimeError("Stage 39 protocol digest placeholder was not replaced")

protocol_sources[1] = configuration
cells = [markdown(introduction)] + [code(value) for value in protocol_sources[1:]]
for index, cell in enumerate(cells):
    cell["id"] = f"stage39-{index:02d}"

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
