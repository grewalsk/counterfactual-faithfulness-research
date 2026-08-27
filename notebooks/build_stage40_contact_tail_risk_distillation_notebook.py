"""Build the prospective Stage 40 contact-tail risk distillation Colab."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
TARGET = ROOT / "40_contact_tail_risk_distillation.ipynb"
NUMERICAL = REPOSITORY / "src/cf_faithfulness/stage40_contact_tail.py"


def load_builder(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BASE = load_builder(
    ROOT / "build_stage39_2_contact_tail_qualified_replication_notebook.py",
    "stage392_builder_for_stage40",
)

code = BASE.code
markdown = BASE.markdown
replace_assignment = BASE.replace_assignment
replace_block = BASE.replace_block
assigned_uppercase_names = BASE.assigned_uppercase_names
function_sources = BASE.BASE.function_sources


introduction = r'''# Stage 40: contact-tail risk distillation

## Frozen decision before computation

Stage 39.2 was a valid negative result.  DINO retained coefficient-matched
mean equivalence, JEPA did not, and both representations failed the registered
absolute p95 and contact-mode gates.  The failure was broad: deleting the
largest shared outlier did not repair either p95 or the JEPA estimand.

Stage 40 tests one minimal repair rather than changing the metric.  It starts
from the coefficient-matched latent-only recursive adapter, grounds its
physical decoder on simulator truth available only in non-evaluation splits,
and changes the training measure to emphasize transitions whose target is
contact.  A post-contact-to-contact re-entry receives one additional copy of
the same excess weight.  Weights are normalized to mean one, all physical
scales and evaluation thresholds remain unchanged, and no contact label is
needed at inference.

The contact multiplier is selected independently for JEPA and DINO from
`[1, 2, 4, 8]` on disjoint model-selection trajectories.  Selection minimizes
p95 physical NMSE subject to at most 5% mean degradation from the uniform
grounded control.  Final uniform and selected-risk models use identical
architecture, data, initialization within seed, update count, learning rate,
and effective latent semigroup coefficient.

Locked success requires, for both predictor panels separately:

1. a 90% hierarchical interval for repair-over-uniform mean row-wise gain
   whose lower bound is at least `-5%`;
2. at least 10% p95 improvement for every final seed;
3. at least 10% terminal-contact mean improvement for every final seed; and
4. the repaired model passing the unchanged absolute length, mode, p95, and
   catastrophic-rate thresholds for every seed.

The overall decision is conjunctive and never pools JEPA with DINO.  Stage
39.2 informed the repair family but its artifacts and evaluation rows are not
read.  Stage 40 uses fresh trajectories, action words, optimization seeds,
fitted models, and output storage.  Planning remains sealed.
'''


configuration = BASE.configuration
for name, value in {
    "PROTOCOL_ID": '"stage40-contact-tail-risk-distillation-v1"',
    "NOTEBOOK_PROTOCOL_SHA256": '"__PROTOCOL_DIGEST__"',
    "EVIDENCE_STATUS": '"FRESH_PROSPECTIVE_CONTACT_TAIL_REPAIR"',
    "EXPERIMENT_NOTEBOOK_PATH": '"notebooks/40_contact_tail_risk_distillation.ipynb"',
    "EXPERIMENT_BUILDER_PATH": '"notebooks/build_stage40_contact_tail_risk_distillation_notebook.py"',
    "EXPERIMENT_NUMERICAL_PATH": '"src/cf_faithfulness/stage40_contact_tail.py"',
    "OUTPUT_DIR": '"/content/counterfactual_faithfulness_stage40_ctrd"',
    "DRIVE_OUTPUT_DIR": '"/content/drive/MyDrive/counterfactual_faithfulness_stage40_ctrd"',
    "RUN_REQUEST_PATH": '"/content/drive/MyDrive/counterfactual_faithfulness_stage40_ctrd/stage40_run_request.json"',
    "SEED": "400101",
    "DESIGN_SEED": "400141",
    "DECODER_SEED": "400183",
    "RANK_SEED": "400213",
    "CALIBRATION_SEED": "400253",
    "BOOTSTRAP_SEED": "400283",
    "CONTROL_SEED": "400351",
    "CONSTRUCTION_TRAJECTORY_POOL": "list(range(102000, 104000))",
    "MODEL_SELECTION_TRAJECTORY_POOL": "list(range(104000, 106000))",
    "CALIBRATION_TRAJECTORY_POOL": "list(range(106000, 108000))",
    "EVALUATION_TRAJECTORY_POOL": "list(range(108000, 114000))",
    "TASK_ID_OFFSET": "400000",
}.items():
    configuration = replace_assignment(configuration, name, value)
configuration = replace_assignment(
    configuration,
    "FINAL_TRAINING_SEEDS",
    '[4001, 4002, 4003] if RUN_MODE == "pilot" else [4001, 4002]',
)
configuration = replace_assignment(
    configuration,
    "PRIMARY_VARIANTS",
    '["uniform_grounded", "contact_risk_grounded"]',
)
configuration = replace_assignment(
    configuration,
    "TAIL_QUALIFICATION_VARIANTS",
    '["uniform_grounded", "contact_risk_grounded"]',
)
configuration = replace_block(
    configuration,
    "CANONICAL_RESPONSE_WORD_NAMES = [",
    "CALIBRATION_INTERCHANGE_PAIRS =",
    r'''CANONICAL_RESPONSE_WORD_NAMES = ["A", "B", "C", "D", "AB", "CD", "BA", "DC"]
CONSTRUCTION_WORD_NAMES = [
    "ADBDADACC", "DDDAABADD", "ACCBCDADBC", "DDAAAAABBD",
    "BDDDDABBBCC", "ADADCBADACC", "CBADCBABCADD", "AABABBAACBAB",
]
MODEL_SELECTION_WORD_NAMES = [
    "ACDBCBADB", "BDAABBACA", "DABABCDCCB", "DBBACCBABB",
    "ADADACABBDC", "AAABADACDAA", "ABBCCABACCDC", "BDBDACBADCCC",
]
CALIBRATION_WORD_NAMES = [
    "CBAADACCB", "BDCCABDAB", "DDCADBBBAA", "ADDDADCCBA",
    "ABCAADACBDA", "DDADCAAABBC", "AAAABBBADDAD", "DDCDBBBCCCAB",
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
    "CDDADADCC", "ADBCDCDDD", "BDDBADDCBD", "CBACDBCBBD",
    "DADBCCCABAC", "CCBAABACBBB", "BBBABAADDCBA", "DCBDDBCAACDC",
]
PLANNING_WORD_NAMES = []
EVALUATION_WORD_NAMES_REGISTERED = list(CLOSURE_EVALUATION_WORD_NAMES)
EVALUATION_WORD_SPECS = [
    stage39_word_spec(name) for name in EVALUATION_WORD_NAMES_REGISTERED
]
''',
)
configuration = configuration.replace(
    '"fresh_trajectory_ids_90000_to_101999",',
    '"fresh_trajectory_ids_102000_to_113999",',
)
configuration = re.sub(
    r"^PROTOCOL_CONFIG_KEYS = \[.*\]\n?", "", configuration, flags=re.M
)
configuration += r'''

PINNED_LATENT_OUTER_WEIGHTS = {"jepa": 2.0, "dino": 0.5}
CONTACT_MULTIPLIER_CANDIDATES = [1.0, 2.0, 4.0, 8.0]
RISK_SELECTION_SEEDS = [40011, 40012] if RUN_MODE == "pilot" else [40011]
MAX_SELECTION_MEAN_RATIO = 1.05
MIN_P95_RELATIVE_IMPROVEMENT = 0.10
MIN_CONTACT_RELATIVE_IMPROVEMENT = 0.10
MEAN_NONINFERIORITY_MARGIN = 0.05
assert CONTACT_MULTIPLIER_CANDIDATES[0] == 1.0
assert MIN_P95_RELATIVE_IMPROVEMENT == 0.10
assert MIN_CONTACT_RELATIVE_IMPROVEMENT == 0.10
assert MEAN_NONINFERIORITY_MARGIN == 0.05
assert PRIMARY_VARIANTS == TAIL_QUALIFICATION_VARIANTS
assert set(PINNED_LATENT_OUTER_WEIGHTS) == {"jepa", "dino"}
'''
configuration += "\n\nPROTOCOL_CONFIG_KEYS = " + repr(
    assigned_uppercase_names(configuration)
) + "\n"

installation = BASE.installation
setup = BASE.setup.replace("stage39_2_ctqr", "stage40_ctrd")
analysis_helpers = BASE.analysis_helpers + "\n\n" + function_sources(
    NUMERICAL.read_text(),
    [
        "contact_transition_weights",
        "_weighted_mse",
        "fit_contact_risk_predictive_state_closure",
        "select_contact_risk_candidate",
        "Stage40PanelDecision",
        "derive_stage40_panel_decision",
        "derive_stage40_decision",
    ],
)
analysis_helpers = analysis_helpers.replace(
    "class Stage40PanelDecision:\n", "@dataclass(frozen=True)\nclass Stage40PanelDecision:\n"
)
model_helpers = BASE.model_helpers
design_and_runtime_helpers = BASE.design_and_runtime_helpers
physical_truth = BASE.physical_truth
simulator_preflight = BASE.simulator_preflight
construction_and_paths = BASE.construction_and_paths
data_and_selection = BASE.data_and_selection
data_and_selection = data_and_selection.replace(
    "# Select semigroup strength independently for each representation.",
    "# Freeze prior-developed latent strength and define split-safe loaders.",
)
data_and_selection = data_and_selection[:data_and_selection.index(
    "SELECTED_SEMIGROUP = {}"
)] + r'''SELECTED_SEMIGROUP = {
    short: {
        "model": short,
        "semigroup_weight": float(weight),
        "selection_source": "pinned_before_stage40_from_prior_development",
    }
    for short, weight in PINNED_LATENT_OUTER_WEIGHTS.items()
}
if not PIPELINE_FAILED and SIMULATOR_PREFLIGHT_PASSED:
    try:
        verify_executed_notebook_through(
            "# Freeze prior-developed latent strength and define split-safe loaders."
        )
        selection_path = CALIBRATION_MODEL_DIR / "frozen_stage40_latent_strength.json"
        write_json(selection_path, {
            "protocol_id": PROTOCOL_ID,
            "run_signature": RUN_SIGNATURE,
            "selected": SELECTED_SEMIGROUP,
            "evaluation_rows_used": 0,
            "stage39_2_artifacts_read": False,
        })
        write_digest_sidecar(selection_path)
        atomic_checkpoint("stage40_latent_strength_frozen", {
            "selection_sha256": sha256_file(selection_path),
            "selected": SELECTED_SEMIGROUP,
        })
        print(json.dumps({"pinned_semigroup": SELECTED_SEMIGROUP}, indent=2))
    except Exception:
        record_failure("stage40_latent_strength_freeze")
'''


risk_selection = r'''# Select contact-risk strength on development trajectories only.
SELECTED_CONTACT_RISK = {}
CONTACT_RISK_SELECTION_ROWS = []
CONTACT_RISK_SELECTION_SEED_ROWS = []


def stage40_terminal_labels(labels, mask):
    values = np.asarray(labels).astype(str)
    valid = np.asarray(mask, dtype=bool)
    index = np.max(np.where(valid, np.arange(valid.shape[1])[None, :], -1), axis=1)
    if np.any(index < 0):
        raise ValueError("each label path needs an endpoint")
    return values[np.arange(len(values)), index]


def stage40_validation_errors(artifact, data, scale):
    rollout = rollout_predictive_state_closure(
        artifact, data["initial_carrier"], data["actions"],
        data["carrier"], data["mask"],
    )
    return scaled_path_mse(
        rollout["physical"], data["simulator"], rollout["evaluation_mask"], scale,
    )


if not PIPELINE_FAILED and SIMULATOR_PREFLIGHT_PASSED:
    try:
        verify_executed_notebook_through(
            "# Select contact-risk strength on development trajectories only."
        )
        for short in ["jepa", "dino"]:
            train = load_stage39_sequences(short, "construction")
            validation = load_stage39_sequences(short, "model_selection")
            scale = np.maximum(
                np.std(train["simulator"][train["mask"]], axis=0, ddof=1), 1e-8
            )
            terminal = stage40_terminal_labels(
                validation["target_mode"], validation["mask"]
            )
            contact_rows = terminal == "contact"
            if not np.any(contact_rows):
                raise RuntimeError("Stage 40 development split has no contact endpoints")
            model_rows = []
            for multiplier in CONTACT_MULTIPLIER_CANDIDATES:
                risk = contact_transition_weights(
                    train["initial_mode"], train["target_mode"], train["mask"],
                    contact_multiplier=multiplier,
                )
                seed_errors = []
                for selection_seed in RISK_SELECTION_SEEDS:
                    artifact = fit_contact_risk_predictive_state_closure(
                        train["initial_carrier"], train["actions"], train["carrier"],
                        train["simulator"], train["mask"], risk,
                        history_length=FIXED_HISTORY_LENGTH,
                        latent_dim=FIXED_LATENT_DIM, dynamics=FIXED_DYNAMICS,
                        epochs=ACTIVE_SELECTION_EPOCHS,
                        learning_rate=PSCD_LEARNING_RATE,
                        seed=stable_seed(
                            CALIBRATION_SEED, "stage40_risk_selection", short,
                            int(selection_seed),
                        ),
                        semigroup_horizons=SEMIGROUP_HORIZONS,
                        semigroup_weight=(
                            COEFFICIENT_MATCH_FACTOR
                            * float(SELECTED_SEMIGROUP[short]["semigroup_weight"])
                        ),
                        semigroup_component_weights=OVERSHOOT_COMPONENT_WEIGHTS,
                        free_weight=1.0,
                    )
                    errors = stage40_validation_errors(artifact, validation, scale)
                    seed_errors.append(errors)
                    CONTACT_RISK_SELECTION_SEED_ROWS.append({
                        "model": short,
                        "contact_multiplier": float(multiplier),
                        "selection_seed": int(selection_seed),
                        "mean_nmse": float(np.mean(errors)),
                        "p95_nmse": float(np.quantile(errors, 0.95)),
                        "terminal_contact_nmse": float(np.mean(errors[contact_rows])),
                        "catastrophic_rate": float(np.mean(errors > 1.0)),
                    })
                    del artifact
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                stacked = np.concatenate(seed_errors)
                repeated_contact = np.tile(contact_rows, len(seed_errors))
                row = {
                    "model": short,
                    "contact_multiplier": float(multiplier),
                    "mean_nmse": float(np.mean(stacked)),
                    "p95_nmse": float(np.quantile(stacked, 0.95)),
                    "terminal_contact_nmse": float(np.mean(stacked[repeated_contact])),
                    "catastrophic_rate": float(np.mean(stacked > 1.0)),
                    "selection_seeds": list(map(int, RISK_SELECTION_SEEDS)),
                }
                model_rows.append(row)
                CONTACT_RISK_SELECTION_ROWS.append(row)
            SELECTED_CONTACT_RISK[short] = select_contact_risk_candidate(
                model_rows, max_mean_ratio=MAX_SELECTION_MEAN_RATIO
            )
        write_csv(
            EVIDENCE_DIR / "stage40_contact_risk_selection_seed_rows.csv",
            CONTACT_RISK_SELECTION_SEED_ROWS,
        )
        write_csv(
            EVIDENCE_DIR / "stage40_contact_risk_selection_rows.csv",
            CONTACT_RISK_SELECTION_ROWS,
        )
        selection_path = CALIBRATION_MODEL_DIR / "frozen_stage40_contact_risk.json"
        write_json(selection_path, {
            "protocol_id": PROTOCOL_ID,
            "run_signature": RUN_SIGNATURE,
            "selected": SELECTED_CONTACT_RISK,
            "candidate_rows": CONTACT_RISK_SELECTION_ROWS,
            "evaluation_rows_used": 0,
            "stage39_2_artifacts_read": False,
        })
        write_digest_sidecar(selection_path)
        atomic_checkpoint("stage40_contact_risk_selection_complete", {
            "selection_sha256": sha256_file(selection_path),
            "selected": SELECTED_CONTACT_RISK,
            "evaluation_opened": False,
        })
        print(json.dumps({"selected_contact_risk": SELECTED_CONTACT_RISK}, indent=2))
    except Exception:
        record_failure("stage40_contact_risk_selection")
'''


calibration = BASE.calibration.replace(
    "# Freeze matched full and coefficient-matched models before evaluation.",
    "# Freeze matched uniform and contact-risk grounded models before evaluation.",
)
calibration = replace_block(
    calibration,
    "def stage39_variant_configuration(short, variant):",
    "def effective_latent_coefficient(objective):",
    r'''def stage39_variant_configuration(short, variant):
    selected = float(SELECTED_SEMIGROUP[short]["semigroup_weight"])
    common = {
        "free_weight": 1.0,
        "semigroup_weight": COEFFICIENT_MATCH_FACTOR * selected,
        "semigroup_component_weights": OVERSHOOT_COMPONENT_WEIGHTS,
    }
    table = {
        "uniform_grounded": {**common, "contact_multiplier": 1.0},
        "contact_risk_grounded": {
            **common,
            "contact_multiplier": float(
                SELECTED_CONTACT_RISK[short]["contact_multiplier"]
            ),
        },
    }
    if str(variant) not in table:
        raise KeyError(f"unknown Stage 40 variant {variant!r}")
    return table[str(variant)]


''',
)
calibration = replace_block(
    calibration,
    "def fit_or_load_stage39_model(short, variant, seed, data):",
    "if not PIPELINE_FAILED and SIMULATOR_PREFLIGHT_PASSED:",
    r'''def fit_or_load_stage39_model(short, variant, seed, data):
    array_path, schema_path = stage39_artifact_paths(short, variant, seed)
    sidecars = [Path(str(array_path) + ".sha256"), Path(str(schema_path) + ".sha256")]
    if array_path.is_file() and schema_path.is_file() and all(path.is_file() for path in sidecars):
        PROVENANCE_COUNTS["validated_cache_hits"] += 1
        return load_stage39_artifact(short, variant, seed)
    objective = stage39_variant_configuration(short, variant)
    risk = contact_transition_weights(
        data["initial_mode"], data["target_mode"], data["mask"],
        contact_multiplier=objective["contact_multiplier"],
    )
    artifact = fit_contact_risk_predictive_state_closure(
        data["initial_carrier"], data["actions"], data["carrier"],
        data["simulator"], data["mask"], risk,
        history_length=FIXED_HISTORY_LENGTH,
        latent_dim=FIXED_LATENT_DIM, dynamics=FIXED_DYNAMICS,
        epochs=ACTIVE_FINAL_EPOCHS, learning_rate=PSCD_LEARNING_RATE,
        seed=int(seed), semigroup_horizons=SEMIGROUP_HORIZONS,
        free_weight=objective["free_weight"],
        semigroup_weight=objective["semigroup_weight"],
        semigroup_component_weights=objective["semigroup_component_weights"],
    )
    artifact["config"]["contact_multiplier"] = float(
        objective["contact_multiplier"]
    )
    artifact["config"]["physical_target"] = "simulator_ground_truth"
    save_stage39_artifact(short, variant, seed, artifact)
    return artifact


''',
)
calibration = calibration.replace(
    'full_coefficient = effective_latent_coefficient(objectives["full"])\n'
    '            matched_coefficient = effective_latent_coefficient(objectives["coefficient_matched"])\n'
    '            if not np.isclose(full_coefficient, matched_coefficient, atol=1e-15, rtol=0):',
    'uniform_coefficient = effective_latent_coefficient(objectives["uniform_grounded"])\n'
    '            risk_coefficient = effective_latent_coefficient(objectives["contact_risk_grounded"])\n'
    '            if not np.isclose(uniform_coefficient, risk_coefficient, atol=1e-15, rtol=0):',
)
calibration = calibration.replace(
    '"full_effective_latent_coefficient": full_coefficient,\n'
    '                "matched_effective_latent_coefficient": matched_coefficient,',
    '"uniform_effective_latent_coefficient": uniform_coefficient,\n'
    '                "risk_effective_latent_coefficient": risk_coefficient,\n'
    '                "selected_contact_multiplier": float(\n'
    '                    SELECTED_CONTACT_RISK[short]["contact_multiplier"]\n'
    '                ),',
)
calibration = calibration.replace(
    'or list(artifact["config"]["semigroup_component_weights"])\n'
    '                        != list(expected["semigroup_component_weights"])',
    'or list(artifact["config"]["semigroup_component_weights"])\n'
    '                        != list(expected["semigroup_component_weights"])\n'
    '                        or float(artifact["config"]["contact_multiplier"])\n'
    '                        != float(expected["contact_multiplier"])\n'
    '                        or artifact["config"]["physical_target"]\n'
    '                        != "simulator_ground_truth"',
)
calibration = calibration.replace("Stage 39", "Stage 40")
calibration = calibration.replace("stage39_models_frozen", "stage40_models_frozen")
calibration = calibration.replace('f"stage39_{short}_{variant}_seed', 'f"stage40_{short}_{variant}_seed')
calibration = calibration.replace("stage39_simulator_final", "stage40_simulator_final")
calibration = calibration.replace("stage39_frozen_scales.npz", "stage40_frozen_scales.npz")
calibration = calibration.replace(
    '        certificate_path = CALIBRATION_MODEL_DIR / "evaluation_open_certificate.json"',
    '        risk_selection_path = CALIBRATION_MODEL_DIR / "frozen_stage40_contact_risk.json"\n'
    '        validate_digest_sidecar(risk_selection_path)\n'
    '        certificate_path = CALIBRATION_MODEL_DIR / "evaluation_open_certificate.json"',
)
calibration = calibration.replace(
    '            "scale_sha256": sha256_file(scale_path),',
    '            "scale_sha256": sha256_file(scale_path),\n'
    '            "contact_risk_selection_sha256": sha256_file(risk_selection_path),',
)
calibration = calibration.replace(
    'record_failure("stage39_calibration_model_freeze")',
    'record_failure("stage40_calibration_model_freeze")',
)


locked_evaluation = r'''# Open the locked Stage 40 repair panel; development remains sealed.
DECISION_PAYLOAD = {
    "status": "INCONCLUSIVE_PIPELINE_FAILURE",
    "passed": False,
    "planning_opened": False,
}
EVALUATION_ROWS = []
SUMMARY = {}
PANEL_DECISIONS = {}


def stage40_absolute_summary(errors, closure, terminal_modes):
    values = np.asarray(errors, dtype=np.float64)
    initial_modes = np.asarray(closure["initial_mode"]).astype(str)
    lengths = np.asarray(closure["length"], dtype=np.int64)
    length_means = {
        str(length): float(np.mean(values[lengths == length]))
        for length in sorted(set(lengths.tolist()))
    }
    initial_mode_means = {
        mode: float(np.mean(values[initial_modes == mode]))
        for mode in sorted(set(initial_modes.tolist()))
    }
    terminal_mode_means = {
        mode: float(np.mean(values[terminal_modes == mode]))
        for mode in sorted(set(terminal_modes.tolist()))
    }
    if "contact" not in terminal_mode_means:
        raise RuntimeError("locked Stage 40 panel has no contact endpoints")
    tail = tail_risk_summary(values)
    gates = {
        "word_length_means": bool(all(
            value <= MAX_LENGTH_PHYSICAL_NMSE for value in length_means.values()
        )),
        "initial_mode_means": bool(all(
            value <= MAX_MODE_PHYSICAL_NMSE for value in initial_mode_means.values()
        )),
        "terminal_mode_means": bool(all(
            value <= MAX_MODE_PHYSICAL_NMSE for value in terminal_mode_means.values()
        )),
        "p95": bool(tail["p95"] <= MAX_P95_PHYSICAL_NMSE),
        "catastrophic_rate": bool(
            tail["catastrophic_rate_gt_1"] <= MAX_CATASTROPHIC_RATE
        ),
    }
    return {
        "passed": bool(all(gates.values())),
        "gates": gates,
        "length_physical_nmse": length_means,
        "initial_mode_physical_nmse": initial_mode_means,
        "terminal_mode_physical_nmse": terminal_mode_means,
        "tail": tail,
    }


def stage40_panel(short, closure, simulator_quality_passed):
    scale = PHYSICAL_SCALES[short]
    groups = closure["group"]
    if len(np.unique(groups)) < MIN_EVALUATION_TRAJECTORIES:
        raise RuntimeError("locked Stage 40 panel has too few families")
    physical = {variant: [] for variant in PRIMARY_VARIANTS}
    false_history_errors = []
    native_history = history_tensor(
        closure["initial_carrier"], closure["carrier"], closure["mask"],
        FIXED_HISTORY_LENGTH,
    )
    for seed in FINAL_TRAINING_SEEDS:
        for variant in PRIMARY_VARIANTS:
            rollout = rollout_predictive_state_closure(
                FROZEN_MODELS[short][variant][int(seed)],
                closure["initial_carrier"], closure["actions"],
                closure["carrier"], closure["mask"],
            )
            physical[variant].append(scaled_path_mse(
                rollout["physical"], closure["simulator"],
                rollout["evaluation_mask"], scale,
            ))
        false_history = permute_past_history(
            native_history, groups, closure["mask"],
            seed=stable_seed(CONTROL_SEED, "stage40_wrong_history", short, int(seed)),
        )
        false_rollout = rollout_predictive_state_closure(
            FROZEN_MODELS[short]["contact_risk_grounded"][int(seed)],
            closure["initial_carrier"], closure["actions"],
            closure["carrier"], closure["mask"], histories_override=false_history,
        )
        false_history_errors.append(scaled_path_mse(
            false_rollout["physical"], closure["simulator"],
            false_rollout["evaluation_mask"], scale,
        ))
    physical = {key: np.stack(value, axis=0) for key, value in physical.items()}
    false_history_errors = np.stack(false_history_errors, axis=0)
    repair = physical["contact_risk_grounded"]
    uniform = physical["uniform_grounded"]
    row_gain = paired_rowwise_relative_gain(repair, uniform)
    interval90 = hierarchical_seed_family_interval(
        row_gain, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
        seed=stable_seed(BOOTSTRAP_SEED, "stage40_primary", short),
        confidence=PRIMARY_CONFIDENCE,
    )
    history_gain = paired_rowwise_relative_gain(repair, false_history_errors)
    history_interval = hierarchical_seed_family_interval(
        history_gain, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
        seed=stable_seed(BOOTSTRAP_SEED, "stage40_history", short),
        confidence=PRIMARY_CONFIDENCE,
    )
    terminal_modes = stage40_terminal_labels(closure["target_mode"], closure["mask"])
    seed_summaries = []
    p95_passes, contact_passes, absolute_passes = [], [], []
    for seed_index, seed in enumerate(FINAL_TRAINING_SEEDS):
        absolute = {
            variant: stage40_absolute_summary(
                physical[variant][seed_index], closure, terminal_modes
            )
            for variant in PRIMARY_VARIANTS
        }
        uniform_absolute = absolute["uniform_grounded"]
        repair_absolute = absolute["contact_risk_grounded"]
        p95_gain = float(
            (uniform_absolute["tail"]["p95"] - repair_absolute["tail"]["p95"])
            / max(uniform_absolute["tail"]["p95"], 1e-12)
        )
        uniform_contact = uniform_absolute["terminal_mode_physical_nmse"]["contact"]
        repair_contact = repair_absolute["terminal_mode_physical_nmse"]["contact"]
        contact_gain = float(
            (uniform_contact - repair_contact) / max(uniform_contact, 1e-12)
        )
        p95_passes.append(p95_gain >= MIN_P95_RELATIVE_IMPROVEMENT)
        contact_passes.append(contact_gain >= MIN_CONTACT_RELATIVE_IMPROVEMENT)
        absolute_passes.append(repair_absolute["passed"])
        seed_summaries.append({
            "seed": int(seed),
            "uniform_mean_nmse": float(np.mean(uniform[seed_index])),
            "repair_mean_nmse": float(np.mean(repair[seed_index])),
            "mean_rowwise_gain": float(np.mean(row_gain[seed_index])),
            "p95_relative_improvement": p95_gain,
            "terminal_contact_relative_improvement": contact_gain,
            "absolute_qualification": absolute,
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
                "uniform_grounded_physical_nmse": float(uniform[seed_index, row_index]),
                "contact_risk_grounded_physical_nmse": float(repair[seed_index, row_index]),
                "repair_over_uniform_rowwise_gain": float(row_gain[seed_index, row_index]),
                "wrong_history_physical_nmse": float(false_history_errors[seed_index, row_index]),
            })
    decision = derive_stage40_panel_decision(
        float(np.mean(row_gain)), interval90,
        p95_improvement=bool(all(p95_passes)),
        contact_improvement=bool(all(contact_passes)),
        absolute_tail_qualified=bool(all(absolute_passes)),
        quality_control_passed=simulator_quality_passed,
        noninferiority_margin=MEAN_NONINFERIORITY_MARGIN,
    )
    return decision, {
        "selected_contact_multiplier": float(
            SELECTED_CONTACT_RISK[short]["contact_multiplier"]
        ),
        "mean_rowwise_relative_gain": float(np.mean(row_gain)),
        "hierarchical_interval90": list(interval90),
        "mean_noninferiority_margin": MEAN_NONINFERIORITY_MARGIN,
        "uniform_mean_nmse": float(np.mean(uniform)),
        "repair_mean_nmse": float(np.mean(repair)),
        "pooled_ratio_of_means_gain": pooled_ratio_of_means_gain(repair, uniform),
        "history_gain": float(np.mean(history_gain)),
        "history_interval90": list(history_interval),
        "p95_improvement_all_seeds": bool(all(p95_passes)),
        "contact_improvement_all_seeds": bool(all(contact_passes)),
        "absolute_tail_qualified_all_seeds": bool(all(absolute_passes)),
        "seed_summaries": seed_summaries,
        "classification": decision.classification,
    }


if not PIPELINE_FAILED and SIMULATOR_PREFLIGHT_PASSED:
    try:
        verify_executed_notebook_through(
            "# Open the locked Stage 40 repair panel; development remains sealed."
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
            raise RuntimeError("Stage 40 evaluation-open certificate is invalid")
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
            panel, metrics = stage40_panel(short, closure_data[short], simulator_passed)
            PANEL_DECISIONS[short] = panel
            SUMMARY[short] = metrics
        DECISION_PAYLOAD = derive_stage40_decision(PANEL_DECISIONS)
        DECISION_PAYLOAD.update({
            "protocol_id": PROTOCOL_ID,
            "run_signature": RUN_SIGNATURE,
            "evaluation_opened": True,
            "planning_opened": False,
            "simulator_control": {
                "passed": simulator_passed,
                "mean_nmse": float(np.mean(simulator_error)),
                "gain_over_persistence": simulator_gain,
            },
            "primary_estimand": "mean_paired_rowwise_repair_over_uniform_gain",
            "primary_confidence": PRIMARY_CONFIDENCE,
            "mean_noninferiority_margin": MEAN_NONINFERIORITY_MARGIN,
            "minimum_p95_relative_improvement": MIN_P95_RELATIVE_IMPROVEMENT,
            "minimum_contact_relative_improvement": MIN_CONTACT_RELATIVE_IMPROVEMENT,
            "panels_pooled": False,
            "stage39_2_artifacts_read": False,
            "stage39_2_evaluation_rows_consumed": False,
        })
        write_csv(EVIDENCE_DIR / "locked_stage40_rows.csv", EVALUATION_ROWS)
        write_json(EVIDENCE_DIR / "stage40_summary.json", SUMMARY)
        write_json(EVIDENCE_DIR / "stage40_panel_decisions.json", {
            short: {
                "mean_gain": panel.mean_gain,
                "interval90": list(panel.interval90),
                "mean_noninferiority": panel.mean_noninferiority,
                "p95_improvement": panel.p95_improvement,
                "contact_improvement": panel.contact_improvement,
                "absolute_tail_qualified": panel.absolute_tail_qualified,
                "quality_control_passed": panel.quality_control_passed,
                "classification": panel.classification,
            }
            for short, panel in PANEL_DECISIONS.items()
        })
        write_json(OUT / "stage40_decision.json", DECISION_PAYLOAD)
        atomic_checkpoint("stage40_locked_repair_complete", {
            "decision_sha256": sha256_file(OUT / "stage40_decision.json"),
            "status": DECISION_PAYLOAD["status"],
            "rows": len(EVALUATION_ROWS),
            "planning_opened": False,
        })

        figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        for axis, short in zip(axes, ["jepa", "dino"]):
            estimate = SUMMARY[short]["mean_rowwise_relative_gain"]
            low, high = SUMMARY[short]["hierarchical_interval90"]
            axis.errorbar(
                [0], [estimate], yerr=[[estimate - low], [high - estimate]], fmt="o"
            )
            axis.axhline(-MEAN_NONINFERIORITY_MARGIN, color="#dc2626", linestyle="--")
            axis.axhline(0, color="black", linewidth=1)
            axis.set(
                xticks=[], ylabel="contact-risk repair over uniform gain",
                title=f"{short.upper()}: {PANEL_DECISIONS[short].classification}",
            )
        figure.suptitle(f"Stage 40: {DECISION_PAYLOAD['status']}")
        figure.tight_layout()
        figure.savefig(PLOT_DIR / "stage40_contact_tail_repair.png", dpi=180)
        plt.close(figure)
        interpretation = f"""# Automatic Stage 40 interpretation

Status: **{DECISION_PAYLOAD['status'].upper()}**

Stage 40 compares a simulator-grounded, contact-risk-weighted recursive
adapter with its exactly matched uniform-training control.  A confirmed repair
requires mean noninferiority, p95 and terminal-contact improvement for every
seed, and unchanged absolute tail qualification in both JEPA and DINO panels.
Planning remained sealed and no Stage 39.2 artifact or evaluation row was read.
"""
        (OUT / "AUTOMATIC_INTERPRETATION.md").write_text(interpretation)
        print(json.dumps(DECISION_PAYLOAD, indent=2))
    except Exception:
        record_failure("stage40_locked_repair")
'''


packaging = BASE.packaging
packaging = packaging.replace("stage39_2_ctqr", "stage40_ctrd")
packaging = packaging.replace(
    "contact_tail_qualified_replication", "contact_tail_risk_distillation"
)
packaging = packaging.replace(
    "if DOWNLOAD_RESULTS:\n",
    'if DOWNLOAD_RESULTS and not PIPELINE_FAILED and (OUT / "stage40_decision.json").is_file():\n',
)

protocol_sources = [
    introduction, configuration, installation, setup, analysis_helpers,
    model_helpers, design_and_runtime_helpers, physical_truth,
    simulator_preflight, construction_and_paths, data_and_selection,
    risk_selection, calibration, locked_evaluation, packaging,
]
protocol_sources = [value.strip() for value in protocol_sources]
protocol_digest = hashlib.sha256(
    json.dumps(protocol_sources, ensure_ascii=False, separators=(",", ":")).encode()
).hexdigest()
configuration = configuration.replace("__PROTOCOL_DIGEST__", protocol_digest)
if "__PROTOCOL_DIGEST__" in configuration:
    raise RuntimeError("Stage 40 protocol digest placeholder was not replaced")
protocol_sources[1] = configuration

cells = [markdown(introduction)] + [code(value) for value in protocol_sources[1:]]
for index, cell in enumerate(cells):
    cell["id"] = f"stage40-{index:02d}"
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
