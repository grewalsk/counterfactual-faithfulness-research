import hashlib
import importlib.util
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
TARGET = ROOT / "35_hybrid_predictive_composition_closure.ipynb"
NUMERICAL = REPOSITORY / "src/cf_faithfulness/stage35_hybrid_composition.py"

spec = importlib.util.spec_from_file_location(
    "stage34_builder", ROOT / "build_stage34_predictive_fiber_abstraction_notebook.py"
)
STAGE34 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(STAGE34)

code = STAGE34.code
markdown = STAGE34.markdown
assigned_uppercase_names = STAGE34.assigned_uppercase_names
function_sources = STAGE34.function_sources
replace_assignment = STAGE34.replace_assignment
remove_assignment = STAGE34.remove_assignment
replace_block = STAGE34.replace_block


introduction = r'''# Stage 35: JEPA hybrid predictive composition and closure

## Frozen decision before computation

Stage 34.3 found real physical-mode structure but its selected seven-coordinate
repair failed on unseen long action words.  Selection improved 28.9%, yet locked
evaluation worsened 2.5%; pre-contact worsened 47.2%.  The diagnostic also used
one source-mode label for an entire word even though the saved simulator path
could cross contact internally.  It did not test recursive state closure.

This fresh experiment asks a materially different question:

> Does a distributed JEPA carrier admit a local, guard-aware update rule whose
> recursive composition reproduces JEPA's own unseen long-word predictions and
> preserves their physical fidelity?

For carrier state `c`, action `a`, and transition class
`g=(source_mode,target_mode)`, the candidate update is

\[
c_{t+1}=F_g(c_t,a_t).
\]

The primary gate is label-free at evaluation: a classifier predicts `g` from
the current JEPA carrier and action, and its probabilities mix transition
experts.  Simulator labels are used only to train the registered guard and to
form an oracle upper bound.  They are never supplied to the primary recursive
rollout on locked evaluation.

## What is genuinely new relative to Stage 34.3

- Every native JEPA carrier is saved at every action prefix, rather than only
  at the starting state and word endpoint.
- Local maps are fitted only to atomic transitions and selected on disjoint
  short compositions; evaluation uses unseen words of lengths 5--8.
- A fixed-source-mode expert exactly represents the Stage 34.3 structural
  baseline.  It is compared with stepwise oracle modes, oracle guard/reset
  classes, and the label-free predicted guard.
- A simulator-state recursion is a mandatory positive control.
- Permuted and one-step time-shifted transition labels are capacity-matched
  guard controls.
- The native JEPA rollout is the direct-prediction reference.  A flexible
  predictor is not allowed to stand in for mechanistic evidence.

## Sequential gates

1. **Source/split binding:** exact committed source, official checkpoint hash,
   four disjoint fresh trajectory pools, and an unopened evaluation certificate.
2. **Simulator positive control:** the registered local operator class must
   recursively predict physical state before it is used to judge JEPA.
3. **Native physical fidelity:** direct JEPA long-word predictions must improve
   over physical-state persistence on the fresh panel.
4. **Guard transfer:** the predicted guard must beat the frozen source-mode
   expert by at least 10% on words that cross modes, with a positive clustered
   interval.
5. **Guard specificity:** it must separately beat permuted and time-shifted
   transition-label models.
6. **Recursive closure:** recursively decoded JEPA outcomes must stay within
   1.25 times native physical error, with bounded composition discrepancy and
   carrier-support escape.
7. **Family consistency:** no length or starting-mode family may hide a
   catastrophic reversal.

A full pass supports only a bounded, observational claim that this checkpoint
contains a distributed hybrid predictive state over this finite action bank.
It is not causal evidence, does not recover a minimal state, says nothing about
DINO, and does not establish shared circuitry.  A native intervention stage is
permitted only after all gates pass.

Methodological anchors are predictive-state recursion
([Sun et al., 2016](https://proceedings.mlr.press/v48/sun16.html)), nonlinear
observable predictive states ([Boots, Gordon, and Gretton,
2013](https://arxiv.org/abs/1309.6819)), hybrid guard/reset sensitivity
([Kong et al., 2024](https://arxiv.org/abs/2306.06862)), and distributed causal
alignment ([Geiger et al., 2024](https://proceedings.mlr.press/v236/geiger24a)).
'''


configuration = STAGE34.configuration.split("\n\nPROTOCOL_CONFIG_KEYS =", 1)[0]
for old, new in [
    ("Stage 34", "Stage 35"),
    ("STAGE34", "STAGE35"),
    ("stage34-predictive-fiber-abstraction", "stage35-hybrid-predictive-composition"),
    ("counterfactual_faithfulness_stage34_pfca", "counterfactual_faithfulness_stage35_hpcc"),
    ("stage34_run_request.json", "stage35_run_request.json"),
    ("notebooks/34_predictive_fiber_causal_abstraction.ipynb", "notebooks/35_hybrid_predictive_composition_closure.ipynb"),
    ("notebooks/build_stage34_predictive_fiber_abstraction_notebook.py", "notebooks/build_stage35_hybrid_composition_notebook.py"),
    ("src/cf_faithfulness/stage34_predictive_fiber_abstraction.py", "src/cf_faithfulness/stage35_hybrid_composition.py"),
]:
    configuration = configuration.replace(old, new)

for name, value in {
    "EXPERIMENT_SOURCE_REF": '"codex/stage34-predictive-fiber-abstraction"',
    "PROTOCOL_ID": '"stage35-jepa-hybrid-predictive-composition-closure-v1"',
    "NOTEBOOK_PROTOCOL_SHA256": '"__PROTOCOL_DIGEST__"',
    "EVIDENCE_STATUS": '"FRESH_PROSPECTIVE_JEPA_ONLY_OBSERVATIONAL_CLOSURE_TEST"',
    "MAX_ESTIMATED_TOTAL_MINUTES": "360.0",
    "SEED": "35101",
    "DESIGN_SEED": "35141",
    "DECODER_SEED": "35183",
    "RANK_SEED": "35213",
    "CALIBRATION_SEED": "35253",
    "BOOTSTRAP_SEED": "35283",
    "CONTROL_SEED": "35351",
    "MODEL_NAMES": '["jepa_wm_pusht"]',
    "MODEL_SHORT_NAMES": '{"jepa_wm_pusht": "jepa"}',
    "MAX_WORD_LENGTH": "8",
    "CONSTRUCTION_TRAJECTORY_POOL": "list(range(16000, 17600))",
    "MODEL_SELECTION_TRAJECTORY_POOL": "list(range(17600, 19200))",
    "CALIBRATION_TRAJECTORY_POOL": "list(range(19200, 20800))",
    "EVALUATION_TRAJECTORY_POOL": "list(range(20800, 24000))",
    "CONSTRUCTION_TRAJECTORIES": "16",
    "MODEL_SELECTION_TRAJECTORIES": "16",
    "CALIBRATION_TRAJECTORIES": "16",
    "EVALUATION_TRAJECTORIES": "32",
    "TASK_ID_OFFSET": "35000",
}.items():
    configuration = replace_assignment(configuration, name, value)

configuration = replace_block(
    configuration,
    "CORE_WORD_SPECS = [",
    "CALIBRATION_INTERCHANGE_PAIRS =",
    r'''CORE_WORD_SPECS = [
    {"name": "L", "angles": [-30.0], "magnitudes": [0.14]},
    {"name": "R", "angles": [30.0], "magnitudes": [0.14]},
    {"name": "S", "angles": [0.0], "magnitudes": [0.10]},
    {"name": "A", "angles": [-40.0], "magnitudes": [0.18]},
    {"name": "B", "angles": [40.0], "magnitudes": [0.18]},
    {"name": "AB", "angles": [-40.0, 40.0], "magnitudes": [0.18] * 2},
    {"name": "BA", "angles": [40.0, -40.0], "magnitudes": [0.18] * 2},
    {"name": "AA", "angles": [-40.0, -40.0], "magnitudes": [0.18] * 2},
    {"name": "BB", "angles": [40.0, 40.0], "magnitudes": [0.18] * 2},
    {"name": "AAB", "angles": [-40.0, -40.0, 40.0], "magnitudes": [0.18] * 3},
    {"name": "BBA", "angles": [40.0, 40.0, -40.0], "magnitudes": [0.18] * 3},
    {"name": "ABA", "angles": [-40.0, 40.0, -40.0], "magnitudes": [0.18] * 3},
    {"name": "BAB", "angles": [40.0, -40.0, 40.0], "magnitudes": [0.18] * 3},
    {"name": "ABBA", "angles": [-40.0, 40.0, 40.0, -40.0], "magnitudes": [0.18] * 4},
    {"name": "BAAB", "angles": [40.0, -40.0, -40.0, 40.0], "magnitudes": [0.18] * 4},
    {"name": "AAAB", "angles": [-40.0, -40.0, -40.0, 40.0], "magnitudes": [0.18] * 4},
    {"name": "BBBA", "angles": [40.0, 40.0, 40.0, -40.0], "magnitudes": [0.18] * 4},
    {"name": "AABB", "angles": [-40.0, -40.0, 40.0, 40.0], "magnitudes": [0.18] * 4},
    {"name": "BBAA", "angles": [40.0, 40.0, -40.0, -40.0], "magnitudes": [0.18] * 4},
]
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
    r'''EVALUATION_WORD_SPECS = [
    {"name": "AABAB", "angles": [-40.0, -40.0, 40.0, -40.0, 40.0], "magnitudes": [0.18] * 5},
    {"name": "BABAA", "angles": [40.0, -40.0, 40.0, -40.0, -40.0], "magnitudes": [0.18] * 5},
    {"name": "AABBAB", "angles": [-40.0, -40.0, 40.0, 40.0, -40.0, 40.0], "magnitudes": [0.18] * 6},
    {"name": "BABAAB", "angles": [40.0, -40.0, 40.0, -40.0, -40.0, 40.0], "magnitudes": [0.18] * 6},
    {"name": "AAABBAB", "angles": [-40.0, -40.0, -40.0, 40.0, 40.0, -40.0, 40.0], "magnitudes": [0.18] * 7},
    {"name": "BABAAAB", "angles": [40.0, -40.0, 40.0, -40.0, -40.0, -40.0, 40.0], "magnitudes": [0.18] * 7},
    {"name": "AABBABAB", "angles": [-40.0, -40.0, 40.0, 40.0, -40.0, 40.0, -40.0, 40.0], "magnitudes": [0.18] * 8},
    {"name": "BABAABBA", "angles": [40.0, -40.0, 40.0, -40.0, -40.0, 40.0, 40.0, -40.0], "magnitudes": [0.18] * 8},
]
''',
)
configuration = replace_block(
    configuration,
    "EVALUATION_INTERCHANGE_PAIRS =",
    "ZERO_WORD_NAMES =",
    r'''EVALUATION_INTERCHANGE_PAIRS = [
    ["AABAB", "BABAA", 0], ["AABBAB", "BABAAB", 1],
    ["AAABBAB", "BABAAAB", 2], ["AABBABAB", "BABAABBA", 3],
]
''',
)
configuration = replace_assignment(
    configuration, "ZERO_WORD_NAMES", '{length: f"zero{length}" for length in range(1, 9)}'
)

configuration += r'''

CONSTRUCTION_WORD_NAMES = [row["name"] for row in CORE_WORD_SPECS]
MODEL_SELECTION_WORD_NAMES = ["A", "B", "AB", "BA", "AAB", "BBA", "ABBA", "BAAB"]
CALIBRATION_WORD_NAMES = ["A", "B", "AA", "BB", "ABA", "BAB", "AAAB", "BBBA", "AABB", "BBAA"]
PATH_CARRIER_SKETCH_DIM = 256
GUARD_RFF_WIDTHS = [128, 256] if RUN_MODE == "pilot" else [32]
GUARD_RIDGES = [1e-3, 1e-2, 1e-1] if RUN_MODE == "pilot" else [1e-2]
GUARD_SELECTION_FOLDS = 4 if RUN_MODE == "pilot" else 2

MIN_SIMULATOR_CONTROL_GAIN = 0.50
MAX_SIMULATOR_CONTROL_NMSE = 0.25
MIN_NATIVE_FIDELITY_GAIN = 0.10
MIN_CROSSING_GUARD_GAIN = 0.10
MIN_GUARD_CONTROL_ADVANTAGE = 0.05
MAX_RECURSIVE_TO_NATIVE_PHYSICAL_RATIO = 1.25
MAX_RECURSIVE_RATIO_CI_UPPER = 1.50
MAX_COMPOSITION_DISCREPANCY_NMSE = 0.25
MAX_RECURSIVE_SUPPORT_ESCAPE_RATE = 0.10
MAX_LENGTH_FAMILY_RATIO = 1.50
MAX_MODE_FAMILY_RATIO = 2.00

if RUN_MODE == "smoke":
    # Two complete groups are the minimum for grouped model selection.
    ACTIVE_MODEL_SELECTION_TRAJECTORIES = 2
    ACTIVE_CALIBRATION_TRAJECTORIES = 2

assert set(MODEL_SELECTION_WORD_NAMES).issubset(set(CONSTRUCTION_WORD_NAMES))
assert set(CALIBRATION_WORD_NAMES).issubset(set(CONSTRUCTION_WORD_NAMES))
assert set(MODEL_SELECTION_WORD_NAMES) != set(CALIBRATION_WORD_NAMES)
assert {len(name) for name in MODEL_SELECTION_WORD_NAMES} == {1, 2, 3, 4}
assert {len(name) for name in CALIBRATION_WORD_NAMES} == {1, 2, 3, 4}
assert {len(row["name"]) for row in EVALUATION_WORD_SPECS} == {5, 6, 7, 8}
'''

configuration = re.sub(
    r"PINNED = \[.*?\]\n\nassert INTERVENTION_BLOCK",
    '''PINNED = [
    "official_jepa_wm_pusht_checkpoint", "exact_pusht_state_restoration",
    "fresh_disjoint_trajectory_families_16000_to_23999",
    "construction_decoder_only", "model_selection_recursive_guard_hyperparameters_only",
    "calibration_dynamics_and_bridge_only", "locked_evaluation_once",
    "prefix_level_native_carriers", "local_atomic_transition_learning",
    "unseen_length_5_to_8_action_compositions", "fixed_source_mode_baseline",
    "oracle_stepwise_mode_and_guard_upper_bounds", "label_free_predicted_guard_evaluation",
    "permuted_and_time_shifted_guard_controls", "simulator_recursive_positive_control",
    "native_direct_rollout_reference", "distributed_state_not_minimal_state",
    "observational_not_causal", "dino_branch_paused", "no_synthetic_fallback",
    "hash_validated_resume", "transient_drive_io_retries", "no_required_colab_secret",
]

assert INTERVENTION_BLOCK''',
    configuration,
    count=1,
    flags=re.S,
)
configuration = configuration.replace(
    "assert MAX_COMPOSED_LENGTH == MAX_WORD_LENGTH + max(\n    len(row[\"name\"]) for row in CORE_WORD_SPECS\n)\n", ""
)
configuration = configuration.replace(
    "assert MAX_WORD_LENGTH == 8 and STATES_PER_TRAJECTORY == len(MODE_LABELS)",
    "assert MAX_WORD_LENGTH == 8 and STATES_PER_TRAJECTORY == len(MODE_LABELS)",
)

configuration += "\n\nPROTOCOL_CONFIG_KEYS = " + repr(
    assigned_uppercase_names(configuration)
) + "\n"


installation = STAGE34.installation


setup = STAGE34.setup
for old, new in [
    ("Stage 34", "Stage 35"),
    ("STAGE34", "STAGE35"),
    ("stage34", "stage35"),
    ("pfca", "hpcc"),
]:
    setup = setup.replace(old, new)

retry_definition = r'''
def retry_drive_io(label, operation, attempts=8):
    delay = 1.0
    for attempt in range(1, int(attempts) + 1):
        try:
            return operation()
        except OSError as error:
            if attempt == int(attempts):
                raise RuntimeError(
                    f"Google Drive remained unavailable during {label} after {attempts} attempts"
                ) from error
            print(
                f"Transient Drive error during {label} "
                f"(attempt {attempt}/{attempts}): {error}; retrying in {delay:.0f}s"
            )
            time.sleep(delay)
            delay = min(2.0 * delay, 15.0)


'''
setup = setup.replace("\n\ndef ensure_colab_drive():", "\n\n" + retry_definition + "def ensure_colab_drive():", 1)
setup = setup.replace(
    "output_root.mkdir(parents=True, exist_ok=True)",
    'retry_drive_io("create output root", lambda: output_root.mkdir(parents=True, exist_ok=True))',
)
setup = setup.replace(
    "    directory.mkdir(parents=True, exist_ok=True)",
    '    retry_drive_io(f"create {directory}", lambda directory=directory: directory.mkdir(parents=True, exist_ok=True))',
)
setup = setup.replace(
    "CACHE_ROOT.mkdir(parents=True, exist_ok=True)",
    'retry_drive_io("create cache root", lambda: CACHE_ROOT.mkdir(parents=True, exist_ok=True))',
)
setup = replace_block(
    setup,
    "def write_json(path, payload):",
    "def write_csv(path, rows):",
    r'''def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(".tmp.json")
    serialized = json.dumps(payload, indent=2, allow_nan=False) + "\n"

    def commit():
        temporary.write_text(serialized)
        temporary.replace(path)

    retry_drive_io(f"write {path.name}", commit)


''',
)
setup = replace_block(
    setup,
    "def write_csv(path, rows):",
    "def atomic_npz(path, **arrays):",
    r'''def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    temporary = path.with_suffix(".tmp.csv")

    def commit():
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)

    retry_drive_io(f"write {path.name}", commit)


''',
)
setup = replace_block(
    setup,
    "def atomic_npz(path, **arrays):",
    "def sha256_file(path, chunk_bytes=8 * 1024 * 1024):",
    r'''def atomic_npz(path, **arrays):
    path = Path(path)

    def commit():
        temporary = Path(str(path) + ".tmp.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
        digest_path = Path(str(path) + ".sha256")
        digest_temporary = Path(str(digest_path) + ".tmp")
        digest_temporary.write_text(sha256_file(path) + "\n")
        digest_temporary.replace(digest_path)

    retry_drive_io(f"write {path.name}", commit)


    ''',
)
setup = replace_block(
    setup,
    "def write_digest_sidecar(path):",
    "def validate_digest_sidecar(path):",
    r'''def write_digest_sidecar(path):
    path = Path(path)

    def commit():
        digest_path = Path(str(path) + ".sha256")
        temporary = Path(str(digest_path) + ".tmp")
        temporary.write_text(sha256_file(path) + "\n")
        temporary.replace(digest_path)

    retry_drive_io(f"write digest for {path.name}", commit)
    return Path(str(path) + ".sha256")


''',
)
setup += r'''

PATH_DIR = OUT / "prefix_carrier_paths"
CALIBRATION_MODEL_DIR = OUT / "calibration_models"
CONTROL_DIR = OUT / "guard_controls"
for directory in [PATH_DIR, CALIBRATION_MODEL_DIR, CONTROL_DIR]:
    retry_drive_io(
        f"create {directory}",
        lambda directory=directory: directory.mkdir(parents=True, exist_ok=True),
    )
'''


numerical_names = [
    "_matrix", "_labels", "stable_seed", "transition_labels",
    "sequence_source_states", "flatten_sequence_transitions",
    "time_shifted_sequence_labels", "permuted_sequence_labels", "_ridge_fit",
    "fit_rff_ridge", "predict_rff_ridge", "fit_rff_classifier",
    "predict_rff_classifier", "fit_experts", "predict_experts",
    "predict_expert_mixture", "fit_hybrid_family", "recursive_rollout",
    "fit_family_from_sequences", "grouped_sequence_folds", "final_values",
    "scaled_sequence_mse", "relative_gain", "clustered_mean_interval",
    "clustered_ratio_interval", "select_guard_hyperparameters",
    "fit_support_reference", "support_exceedance_rate", "Stage35Gates",
    "derive_stage35_decision",
]
analysis_helpers = r'''# Tested NumPy implementation of hybrid local operators and recursive gates.
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

''' + function_sources(NUMERICAL.read_text(), numerical_names)
analysis_helpers = analysis_helpers.replace(
    "class Stage35Gates:\n", "@dataclass(frozen=True)\nclass Stage35Gates:\n"
)


model_helpers = STAGE34.model_helpers
for old, new in [("stage34", "stage35"), ("Stage 34", "Stage 35")]:
    model_helpers = model_helpers.replace(old, new)


design_and_runtime_helpers = STAGE34.design_and_runtime_helpers
for old, new in [("Stage 34", "Stage 35"), ("stage34", "stage35")]:
    design_and_runtime_helpers = design_and_runtime_helpers.replace(old, new)
design_and_runtime_helpers = design_and_runtime_helpers.replace("3300000", "3500000")


physical_truth = STAGE34.physical_truth
for old, new in [("Stage 34", "Stage 35"), ("stage34", "stage35")]:
    physical_truth = physical_truth.replace(old, new)
physical_truth = physical_truth.replace(
    '"path_states", "path_observables", "path_mask", "path_visuals",',
    '"path_states", "path_observables", "path_mask",',
)
physical_truth = physical_truth.replace("    path_visuals = []\n", "")
physical_truth = physical_truth.replace(
    "        result = rollout_word(record, word, retain_visual=True)",
    "        result = rollout_word(record, word, retain_visual=False)",
)
physical_truth = physical_truth.replace(
    "    words = ALL_WORD_SPECS",
    '''    split_names = {
        "construction": CONSTRUCTION_WORD_NAMES,
        "model_selection": MODEL_SELECTION_WORD_NAMES,
        "calibration": CALIBRATION_WORD_NAMES,
        "evaluation": EVALUATION_WORD_NAMES,
    }
    names = split_names[str(record["split"])]
    words = [WORD_BY_NAME[name] for name in names]''',
)
physical_truth = re.sub(
    r"        padded_visual = np\.zeros\(.*?        path_visuals\.append\(padded_visual\)\n",
    "",
    physical_truth,
    count=1,
    flags=re.S,
)
physical_truth = physical_truth.replace("        path_visuals=np.stack(path_visuals),\n", "")


construction_prefix = STAGE34.construction_and_models.split(
    "\ndef response_signature_from_grounded", 1
)[0]
construction_prefix = construction_prefix.replace(
    "# Fit model-specific grounded readouts and carrier interfaces on construction trajectories only.",
    "# Fit the construction-only grounded JEPA readout and save every native prefix carrier.",
)

construction_and_paths = construction_prefix + r'''


def stage35_names_for_split(split):
    if split == "construction":
        names = list(CONSTRUCTION_WORD_NAMES)
    elif split == "model_selection":
        names = list(MODEL_SELECTION_WORD_NAMES)
    elif split == "calibration":
        names = list(CALIBRATION_WORD_NAMES)
    elif split == "evaluation":
        names = list(EVALUATION_WORD_NAMES)
    else:
        raise ValueError(f"unknown Stage 35 split {split!r}")
    return sorted(set(names), key=lambda name: (WORD_BY_NAME[name]["length"], name))


def stage35_carrier_sketch(value):
    carrier = np.asarray(value, dtype=np.float32)
    if carrier.shape != (EXPECTED_VISUAL_TOKENS, EXPECTED_CARRIER_WIDTHS["jepa_wm_pusht"]):
        raise RuntimeError(f"JEPA carrier shape changed: {carrier.shape}")
    return count_sketch(
        carrier.reshape(1, -1), PATH_CARRIER_SKETCH_DIM,
        stable_seed(CONTROL_SEED, "stage35_path_carrier_sketch"),
    )[0]


def stage35_mode_paths(record, contact_counts, length):
    contacts = np.asarray(contact_counts, dtype=np.int64)[: int(length) * FRAMESKIP]
    source = [str(record["mode"])]
    ever_contact = str(record["mode"]) in {"contact", "post_contact"}
    for step in range(1, int(length)):
        previous = contacts[(step - 1) * FRAMESKIP : step * FRAMESKIP]
        future = contacts[step * FRAMESKIP : (step + 1) * FRAMESKIP]
        previous_any = bool(np.any(previous > 0))
        future_any = bool(np.any(future > 0))
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
    final_segment = contacts[(int(length) - 1) * FRAMESKIP : int(length) * FRAMESKIP]
    contacted_before_final = bool(np.any(contacts[: (int(length) - 1) * FRAMESKIP] > 0))
    if bool(np.any(final_segment > 0)):
        terminal = "contact"
    elif contacted_before_final or str(record["mode"]) in {"contact", "post_contact"}:
        terminal = "post_contact"
    else:
        terminal = "free"
    target = source[1:] + [terminal]
    return source, target


def fit_stage35_decoder(bundle):
    names = stage35_names_for_split("construction")
    feature_rows, target_rows, groups = [], [], []
    for index, record in enumerate(SELECTED_RECORDS["construction"]):
        outputs, _ = grouped_model_words(bundle, record, names)
        tensor, _ = feature_tensor_from_outputs(outputs, names)
        features, feature_meta = response_rows_from_feature_tensor(tensor, names)
        targets, target_meta = truth_rows(record, names)
        if feature_meta != target_meta:
            raise RuntimeError("Stage 35 construction decoder row order changed")
        feature_rows.append(features)
        target_rows.append(targets)
        groups.extend([int(record["trajectory_id"])] * len(features))
        write_json(OUT / "construction_jepa_progress.json", {
            "completed": index + 1,
            "total": len(SELECTED_RECORDS["construction"]),
            "last_record_id": int(record["record_id"]),
        })
    return fit_grouped_ridge(
        np.concatenate(feature_rows), np.concatenate(target_rows),
        np.asarray(groups, dtype=np.int64), penalties=DECODER_RIDGES,
        folds=min(4, len(set(groups))), seed=stable_seed(DECODER_SEED, "stage35_jepa"),
    )


def save_stage35_decoder(decoder):
    path = SUBSPACE_DIR / "decoder_jepa.npz"
    atomic_npz(
        path, weight=np.asarray(decoder["weight"]),
        intercept=np.asarray(decoder["intercept"]),
        penalty=np.asarray(decoder["penalty"]),
        oof_mse=np.asarray(decoder["oof_mse"]),
    )
    write_json(SUBSPACE_DIR / "decoder_manifest_jepa.json", {
        "model": "jepa", "decoder_sha256": sha256_file(path),
        "training_split": "construction", "evaluation_rows_used": 0,
        "grounded_observables": list(GROUNDED_OBSERVABLES),
    })


def load_stage35_decoder():
    path = SUBSPACE_DIR / "decoder_jepa.npz"
    validate_digest_sidecar(path)
    manifest = json.loads((SUBSPACE_DIR / "decoder_manifest_jepa.json").read_text())
    if manifest.get("decoder_sha256") != sha256_file(path):
        raise RuntimeError("Stage 35 decoder manifest mismatch")
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def stage35_path(record, split):
    return PATH_DIR / f"jepa_{split}_{int(record['record_id'])}.npz"


def generate_stage35_path_record(bundle, record, split, decoder):
    path = stage35_path(record, split)
    identity = (
        f"{PROTOCOL_ID}:{RUN_SIGNATURE}:jepa:{record['record_id']}:{split}:"
        f"prefix-carrier-path-v1:{EXPECTED_PRETRAINED_ASSET_SHA256['jepa_wm_pusht.pth.tar']}"
    )
    required = {
        "identity", "word_names", "word_lengths", "initial_carrier",
        "initial_physical", "actions", "carrier_paths", "native_grounded_paths",
        "simulator_grounded_paths", "path_mask", "source_modes", "target_modes",
    }
    if validate_npz_shard(path, required, identity):
        PROVENANCE_COUNTS["validated_cache_hits"] += 1
        return path
    words = stage35_names_for_split(split)
    inference_names = sorted(
        set(words) | {ZERO_WORD_NAMES[1]},
        key=lambda name: (WORD_BY_NAME[name]["length"], name),
    )
    outputs, traces = grouped_model_words(bundle, record, inference_names)
    tensor, _ = feature_tensor_from_outputs(outputs, inference_names)
    grounded = tensor.astype(np.float64) @ decoder["weight"] + decoder["intercept"]
    output_lookup = {name: index for index, name in enumerate(inference_names)}
    with np.load(truth_path(record), allow_pickle=False) as truth:
        truth_lookup = {str(name): index for index, name in enumerate(truth["word_names"])}
        actions = np.zeros((len(words), MAX_WORD_LENGTH, 3), dtype=np.float64)
        carriers = np.zeros(
            (len(words), MAX_WORD_LENGTH, PATH_CARRIER_SKETCH_DIM), dtype=np.float32
        )
        native = np.zeros(
            (len(words), MAX_WORD_LENGTH, len(GROUNDED_OBSERVABLES)), dtype=np.float32
        )
        simulator = np.zeros_like(native)
        mask = np.zeros((len(words), MAX_WORD_LENGTH), dtype=bool)
        source_modes = np.full((len(words), MAX_WORD_LENGTH), "", dtype="<U16")
        target_modes = np.full((len(words), MAX_WORD_LENGTH), "", dtype="<U16")
        for word_index, name in enumerate(words):
            length = int(WORD_BY_NAME[name]["length"])
            model_index = output_lookup[name]
            truth_index = truth_lookup[name]
            macro_actions, _ = word_actions(record, WORD_BY_NAME[name])
            chunks = macro_actions.reshape(length, FRAMESKIP, 2).mean(axis=1)
            actions[word_index, :length, :2] = chunks
            actions[word_index, :length, 2] = np.linalg.norm(chunks, axis=1)
            carriers[word_index, :length] = np.stack([
                stage35_carrier_sketch(value) for value in traces[name]
            ]).astype(np.float32)
            native[word_index, :length] = grounded[model_index, :length].astype(np.float32)
            simulator[word_index, :length] = truth["path_observables"][
                truth_index, :length
            ].astype(np.float32)
            mask[word_index, :length] = True
            source, target = stage35_mode_paths(
                record, truth["contact_counts"][truth_index], length
            )
            source_modes[word_index, :length] = source
            target_modes[word_index, :length] = target
    initial = stage35_carrier_sketch(traces[ZERO_WORD_NAMES[1]][0]).astype(np.float32)
    atomic_npz(
        path, identity=np.asarray(identity), word_names=np.asarray(words),
        word_lengths=np.asarray([WORD_BY_NAME[name]["length"] for name in words], dtype=np.int64),
        initial_carrier=initial,
        initial_physical=grounded_observables(record["state"]).astype(np.float32),
        actions=actions.astype(np.float32), carrier_paths=carriers,
        native_grounded_paths=native, simulator_grounded_paths=simulator,
        path_mask=mask, source_modes=source_modes, target_modes=target_modes,
    )
    PROVENANCE_COUNTS["model_record_forwards"]["jepa"] += 1
    return path


JEPA_DECODER = None
if not PIPELINE_FAILED:
    try:
        verify_executed_notebook_through(
            "# Fit the construction-only grounded JEPA readout and save every native prefix carrier."
        )
        verify_pretrained_assets()
        bundle = load_world_model("jepa_wm_pusht")
        try:
            output_contract = preflight_model_output_contract(bundle)
            if (
                (SUBSPACE_DIR / "decoder_jepa.npz").is_file()
                and (SUBSPACE_DIR / "decoder_manifest_jepa.json").is_file()
            ):
                JEPA_DECODER = load_stage35_decoder()
                PROVENANCE_COUNTS["validated_cache_hits"] += 1
            else:
                JEPA_DECODER = fit_stage35_decoder(bundle)
                save_stage35_decoder(JEPA_DECODER)
            # Locked evaluation forwards are deliberately deferred until after
            # the calibration freeze and evaluation-open certificate.
            for split in ["model_selection", "calibration"]:
                for index, record in enumerate(SELECTED_RECORDS[split]):
                    generate_stage35_path_record(bundle, record, split, JEPA_DECODER)
                    write_json(OUT / f"model_jepa_{split}_progress.json", {
                        "completed": index + 1,
                        "total": len(SELECTED_RECORDS[split]),
                        "last_record_id": int(record["record_id"]),
                    })
            memory_report("stage35_jepa_prefix_paths_complete")
        finally:
            unload_world_model(bundle)
        atomic_checkpoint("jepa_prefix_paths_complete", {
            "decoder_sha256": sha256_file(SUBSPACE_DIR / "decoder_jepa.npz"),
            "checkpoint_sha256": EXPECTED_PRETRAINED_ASSET_SHA256["jepa_wm_pusht.pth.tar"],
            "path_shards": sum(len(SELECTED_RECORDS[split]) for split in [
                "model_selection", "calibration"
            ]),
            "evaluation_paths_materialized": False,
            "dino_loaded": False,
        })
    except Exception:
        record_failure("stage35_construction_decoder_or_prefix_carrier_paths")
'''


data_loader = r'''# Select predicted-guard capacity using model-selection trajectories only.
def load_stage35_sequences(split):
    rows = {key: [] for key in [
        "initial_carrier", "initial_physical", "actions", "carrier",
        "native", "simulator", "mask", "source_mode", "target_mode",
        "word", "length", "group", "record_id", "initial_mode",
    ]}
    for record in SELECTED_RECORDS[split]:
        with np.load(stage35_path(record, split), allow_pickle=False) as payload:
            words = [str(value) for value in payload["word_names"]]
            count = len(words)
            rows["initial_carrier"].extend(np.repeat(
                payload["initial_carrier"][None], count, axis=0
            ))
            rows["initial_physical"].extend(np.repeat(
                payload["initial_physical"][None], count, axis=0
            ))
            rows["actions"].extend(payload["actions"])
            rows["carrier"].extend(payload["carrier_paths"])
            rows["native"].extend(payload["native_grounded_paths"])
            rows["simulator"].extend(payload["simulator_grounded_paths"])
            rows["mask"].extend(payload["path_mask"])
            rows["source_mode"].extend(payload["source_modes"])
            rows["target_mode"].extend(payload["target_modes"])
            rows["word"].extend(words)
            rows["length"].extend(payload["word_lengths"].astype(int).tolist())
            rows["group"].extend([int(record["trajectory_id"])] * count)
            rows["record_id"].extend([int(record["record_id"])] * count)
            rows["initial_mode"].extend([str(record["mode"])] * count)
    for key in [
        "initial_carrier", "initial_physical", "actions", "carrier", "native", "simulator"
    ]:
        rows[key] = np.asarray(rows[key], dtype=np.float64)
    rows["mask"] = np.asarray(rows["mask"], dtype=bool)
    for key in ["source_mode", "target_mode", "word", "initial_mode"]:
        rows[key] = np.asarray(rows[key]).astype(str)
    for key in ["length", "group", "record_id"]:
        rows[key] = np.asarray(rows[key], dtype=np.int64)
    if not all(len(value) == len(rows["word"]) for value in rows.values()):
        raise RuntimeError(f"Stage 35 {split} sequence arrays are misaligned")
    return rows
'''


model_selection = data_loader + r'''

# Select capacity by grouped recursive error on disjoint short compositions.
SELECTED_GUARD = None
SELECTION_ROWS = []
if not PIPELINE_FAILED:
    try:
        verify_executed_notebook_through(
            "# Select predicted-guard capacity using model-selection trajectories only."
        )
        selection = load_stage35_sequences("model_selection")
        SELECTED_GUARD, SELECTION_ROWS = select_guard_hyperparameters(
            selection["initial_carrier"], selection["actions"], selection["carrier"],
            selection["mask"], selection["source_mode"], selection["target_mode"],
            selection["group"], widths=GUARD_RFF_WIDTHS, penalties=GUARD_RIDGES,
            folds=GUARD_SELECTION_FOLDS, seed=CALIBRATION_SEED,
        )
        write_csv(EVIDENCE_DIR / "guard_model_selection_rows.csv", SELECTION_ROWS)
        selection_path = CALIBRATION_MODEL_DIR / "frozen_guard_selection.json"
        write_json(selection_path, {
            "protocol_id": PROTOCOL_ID,
            "selected": SELECTED_GUARD,
            "candidate_rows": SELECTION_ROWS,
            "split": "model_selection",
            "evaluation_rows_used": 0,
        })
        write_digest_sidecar(selection_path)
        atomic_checkpoint("guard_model_selection_complete", {
            "selection_sha256": sha256_file(selection_path),
            "selected": SELECTED_GUARD,
        })
        print(json.dumps({"selected_guard": SELECTED_GUARD}, indent=2))
    except Exception:
        record_failure("stage35_guard_model_selection")
'''


calibration = r'''# Freeze calibration dynamics, label controls, bridge, and support before evaluation is opened.
CALIBRATION = None
CARRIER_FAMILY = None
PERMUTED_FAMILY = None
SHIFTED_FAMILY = None
PHYSICAL_FAMILY = None
CARRIER_TO_GROUNDED = None
CARRIER_SCALE = None
PHYSICAL_SCALE = None
SUPPORT_REFERENCE = None
EVALUATION_OPENED = False


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


if not PIPELINE_FAILED:
    try:
        verify_executed_notebook_through(
            "# Freeze calibration dynamics, label controls, bridge, and support before evaluation is opened."
        )
        if SELECTED_GUARD is None:
            raise RuntimeError("guard hyperparameters were not frozen")
        CALIBRATION = load_stage35_sequences("calibration")
        width = int(SELECTED_GUARD["width"])
        penalty = float(SELECTED_GUARD["penalty"])
        CARRIER_FAMILY = fit_family_from_sequences(
            CALIBRATION["initial_carrier"], CALIBRATION["actions"],
            CALIBRATION["carrier"], CALIBRATION["mask"],
            CALIBRATION["source_mode"], CALIBRATION["target_mode"],
            width=width, penalty=penalty, seed=stable_seed(CALIBRATION_SEED, "carrier"),
        )
        true_transition = transition_labels(
            CALIBRATION["source_mode"], CALIBRATION["target_mode"]
        )
        permuted_transition = permuted_sequence_labels(
            true_transition, CALIBRATION["mask"], CALIBRATION["group"],
            seed=stable_seed(CONTROL_SEED, "permuted_guard"),
        )
        shifted_transition = time_shifted_sequence_labels(
            true_transition, CALIBRATION["mask"]
        )
        PERMUTED_FAMILY = fit_family_from_sequences(
            CALIBRATION["initial_carrier"], CALIBRATION["actions"],
            CALIBRATION["carrier"], CALIBRATION["mask"],
            CALIBRATION["source_mode"], CALIBRATION["target_mode"],
            width=width, penalty=penalty, seed=stable_seed(CALIBRATION_SEED, "permuted"),
            transition_override=permuted_transition,
        )
        SHIFTED_FAMILY = fit_family_from_sequences(
            CALIBRATION["initial_carrier"], CALIBRATION["actions"],
            CALIBRATION["carrier"], CALIBRATION["mask"],
            CALIBRATION["source_mode"], CALIBRATION["target_mode"],
            width=width, penalty=penalty, seed=stable_seed(CALIBRATION_SEED, "shifted"),
            transition_override=shifted_transition,
        )
        PHYSICAL_FAMILY = fit_family_from_sequences(
            CALIBRATION["initial_physical"], CALIBRATION["actions"],
            CALIBRATION["simulator"], CALIBRATION["mask"],
            CALIBRATION["source_mode"], CALIBRATION["target_mode"],
            width=width, penalty=penalty, seed=stable_seed(CALIBRATION_SEED, "physical"),
        )
        valid = CALIBRATION["mask"]
        CARRIER_TO_GROUNDED = fit_rff_ridge(
            CALIBRATION["carrier"][valid], CALIBRATION["native"][valid],
            width=width, penalty=penalty,
            seed=stable_seed(CALIBRATION_SEED, "carrier_to_grounded"),
        )
        CARRIER_SCALE = np.maximum(
            np.std(CALIBRATION["carrier"][valid], axis=0, ddof=1), 1e-8
        )
        PHYSICAL_SCALE = np.maximum(
            np.std(CALIBRATION["simulator"][valid], axis=0, ddof=1), 1e-8
        )
        SUPPORT_REFERENCE = fit_support_reference(CALIBRATION["carrier"][valid])
        arrays, metadata = flatten_model_artifact({
            "carrier_family": CARRIER_FAMILY,
            "permuted_family": PERMUTED_FAMILY,
            "shifted_family": SHIFTED_FAMILY,
            "physical_family": PHYSICAL_FAMILY,
            "carrier_to_grounded": CARRIER_TO_GROUNDED,
            "support_reference": SUPPORT_REFERENCE,
            "carrier_scale": CARRIER_SCALE,
            "physical_scale": PHYSICAL_SCALE,
        })
        model_path = CALIBRATION_MODEL_DIR / "frozen_stage35_models.npz"
        atomic_npz(model_path, **arrays)
        write_json(CALIBRATION_MODEL_DIR / "frozen_stage35_models_schema.json", metadata)
        certificate_path = CALIBRATION_MODEL_DIR / "evaluation_open_certificate.json"
        write_json(certificate_path, {
            "protocol_id": PROTOCOL_ID,
            "run_signature": RUN_SIGNATURE,
            "selection_sha256": sha256_file(
                CALIBRATION_MODEL_DIR / "frozen_guard_selection.json"
            ),
            "models_sha256": sha256_file(model_path),
            "calibration_trajectory_ids": sorted(set(CALIBRATION["group"].tolist())),
            "evaluation_statistics_read": False,
            "evaluation_metrics_computed": False,
            "primary_uses_simulator_labels_at_evaluation": False,
        })
        write_digest_sidecar(certificate_path)
        atomic_checkpoint("calibration_models_frozen", {
            "certificate_sha256": sha256_file(certificate_path),
            "models_sha256": sha256_file(model_path),
        })
        print(json.dumps({
            "calibration_sequences": len(CALIBRATION["word"]),
            "selected_width": width, "selected_penalty": penalty,
            "evaluation_opened": EVALUATION_OPENED,
        }, indent=2))
    except Exception:
        record_failure("stage35_calibration_model_freeze")
'''


locked_evaluation = r'''# Open fresh evaluation exactly once and derive every registered Stage 35 gate.
DECISION_PAYLOAD = {"status": "INCONCLUSIVE_PIPELINE_FAILURE"}
EVALUATION_ROWS = []
SUMMARY = {}
if not PIPELINE_FAILED:
    try:
        verify_executed_notebook_through(
            "# Open fresh evaluation exactly once and derive every registered Stage 35 gate."
        )
        certificate_path = CALIBRATION_MODEL_DIR / "evaluation_open_certificate.json"
        validate_digest_sidecar(certificate_path)
        # Only now may the official checkpoint produce locked evaluation paths.
        bundle = load_world_model("jepa_wm_pusht")
        try:
            for index, record in enumerate(SELECTED_RECORDS["evaluation"]):
                generate_stage35_path_record(bundle, record, "evaluation", JEPA_DECODER)
                write_json(OUT / "model_jepa_evaluation_progress.json", {
                    "completed": index + 1,
                    "total": len(SELECTED_RECORDS["evaluation"]),
                    "last_record_id": int(record["record_id"]),
                })
        finally:
            unload_world_model(bundle)
        atomic_checkpoint("locked_evaluation_paths_materialized", {
            "certificate_sha256": sha256_file(certificate_path),
            "evaluation_path_shards": len(SELECTED_RECORDS["evaluation"]),
            "scientific_statistics_read": False,
        })
        EVALUATION_OPENED = True
        evaluation = load_stage35_sequences("evaluation")
        mask = evaluation["mask"]
        groups = evaluation["group"]
        predicted_carrier = recursive_rollout(
            CARRIER_FAMILY, evaluation["initial_carrier"], evaluation["actions"], mask,
            strategy="predicted_guard",
        )
        fixed_carrier = recursive_rollout(
            CARRIER_FAMILY, evaluation["initial_carrier"], evaluation["actions"], mask,
            strategy="fixed_source", source_modes=evaluation["source_mode"],
        )
        oracle_source_carrier = recursive_rollout(
            CARRIER_FAMILY, evaluation["initial_carrier"], evaluation["actions"], mask,
            strategy="oracle_source", source_modes=evaluation["source_mode"],
        )
        oracle_transition_carrier = recursive_rollout(
            CARRIER_FAMILY, evaluation["initial_carrier"], evaluation["actions"], mask,
            strategy="oracle_transition", source_modes=evaluation["source_mode"],
            target_modes=evaluation["target_mode"],
        )
        global_carrier = recursive_rollout(
            CARRIER_FAMILY, evaluation["initial_carrier"], evaluation["actions"], mask,
            strategy="global",
        )
        permuted_carrier = recursive_rollout(
            PERMUTED_FAMILY, evaluation["initial_carrier"], evaluation["actions"], mask,
            strategy="predicted_guard",
        )
        shifted_carrier = recursive_rollout(
            SHIFTED_FAMILY, evaluation["initial_carrier"], evaluation["actions"], mask,
            strategy="predicted_guard",
        )
        physical_prediction = recursive_rollout(
            PHYSICAL_FAMILY, evaluation["initial_physical"], evaluation["actions"], mask,
            strategy="global",
        )

        def bridge_path(path):
            values = np.asarray(path, dtype=np.float64)
            prediction = predict_rff_ridge(
                CARRIER_TO_GROUNDED, values.reshape(-1, values.shape[-1])
            )
            return prediction.reshape(len(values), values.shape[1], -1)

        recursive_grounded = bridge_path(predicted_carrier)
        persistence_physical = np.repeat(
            evaluation["initial_physical"][:, None, :], MAX_WORD_LENGTH, axis=1
        )
        persistence_carrier = np.repeat(
            evaluation["initial_carrier"][:, None, :], MAX_WORD_LENGTH, axis=1
        )
        carrier_error = scaled_sequence_mse(
            predicted_carrier, evaluation["carrier"], mask, CARRIER_SCALE
        )
        fixed_error = scaled_sequence_mse(
            fixed_carrier, evaluation["carrier"], mask, CARRIER_SCALE
        )
        oracle_source_error = scaled_sequence_mse(
            oracle_source_carrier, evaluation["carrier"], mask, CARRIER_SCALE
        )
        oracle_transition_error = scaled_sequence_mse(
            oracle_transition_carrier, evaluation["carrier"], mask, CARRIER_SCALE
        )
        global_error = scaled_sequence_mse(
            global_carrier, evaluation["carrier"], mask, CARRIER_SCALE
        )
        permuted_error = scaled_sequence_mse(
            permuted_carrier, evaluation["carrier"], mask, CARRIER_SCALE
        )
        shifted_error = scaled_sequence_mse(
            shifted_carrier, evaluation["carrier"], mask, CARRIER_SCALE
        )
        carrier_persistence_error = scaled_sequence_mse(
            persistence_carrier, evaluation["carrier"], mask, CARRIER_SCALE
        )
        native_physical_error = scaled_sequence_mse(
            evaluation["native"], evaluation["simulator"], mask, PHYSICAL_SCALE
        )
        recursive_physical_error = scaled_sequence_mse(
            recursive_grounded, evaluation["simulator"], mask, PHYSICAL_SCALE
        )
        composition_error = scaled_sequence_mse(
            recursive_grounded, evaluation["native"], mask, PHYSICAL_SCALE
        )
        simulator_control_error = scaled_sequence_mse(
            physical_prediction, evaluation["simulator"], mask, PHYSICAL_SCALE
        )
        physical_persistence_error = scaled_sequence_mse(
            persistence_physical, evaluation["simulator"], mask, PHYSICAL_SCALE
        )
        crossing = np.any(
            (evaluation["source_mode"] != evaluation["target_mode"]) & mask, axis=1
        )
        minimum_crossing_clusters = 4 if RUN_MODE == "pilot" else 1
        if len(np.unique(groups[crossing])) < minimum_crossing_clusters:
            raise RuntimeError("too few crossing trajectory clusters for Stage 35")

        simulator_gain = relative_gain(simulator_control_error, physical_persistence_error)
        native_gain = relative_gain(native_physical_error, physical_persistence_error)
        guard_gain = relative_gain(carrier_error, fixed_error)
        permutation_advantage = relative_gain(carrier_error, permuted_error)
        shift_advantage = relative_gain(carrier_error, shifted_error)
        carrier_persistence_gain = relative_gain(carrier_error, carrier_persistence_error)
        simulator_ci = clustered_mean_interval(
            simulator_gain, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
            seed=stable_seed(BOOTSTRAP_SEED, "simulator_control"),
        )
        native_ci = clustered_mean_interval(
            native_gain, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
            seed=stable_seed(BOOTSTRAP_SEED, "native_fidelity"),
        )
        crossing_guard_ci = clustered_mean_interval(
            guard_gain[crossing], groups[crossing], draws=ACTIVE_BOOTSTRAP_DRAWS,
            seed=stable_seed(BOOTSTRAP_SEED, "crossing_guard"),
        )
        permutation_ci = clustered_mean_interval(
            permutation_advantage, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
            seed=stable_seed(BOOTSTRAP_SEED, "permutation_control"), alpha=0.025,
        )
        shift_ci = clustered_mean_interval(
            shift_advantage, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
            seed=stable_seed(BOOTSTRAP_SEED, "shift_control"), alpha=0.025,
        )
        carrier_persistence_ci = clustered_mean_interval(
            carrier_persistence_gain, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
            seed=stable_seed(BOOTSTRAP_SEED, "carrier_persistence"),
        )
        recursive_ratio = float(
            np.mean(recursive_physical_error) / max(np.mean(native_physical_error), 1e-12)
        )
        recursive_ratio_ci = clustered_ratio_interval(
            recursive_physical_error, native_physical_error, groups,
            draws=ACTIVE_BOOTSTRAP_DRAWS,
            seed=stable_seed(BOOTSTRAP_SEED, "recursive_native_ratio"),
        )
        predicted_valid = predicted_carrier[mask]
        support_escape = support_exceedance_rate(SUPPORT_REFERENCE, predicted_valid)
        length_ratios = {
            str(length): float(
                np.mean(recursive_physical_error[evaluation["length"] == length])
                / max(np.mean(native_physical_error[evaluation["length"] == length]), 1e-12)
            )
            for length in sorted(set(evaluation["length"].tolist()))
        }
        mode_ratios = {
            mode: float(
                np.mean(recursive_physical_error[evaluation["initial_mode"] == mode])
                / max(np.mean(native_physical_error[evaluation["initial_mode"] == mode]), 1e-12)
            )
            for mode in MODE_LABELS
        }
        prefix_invariance_max = 0.0
        for record_id in np.unique(evaluation["record_id"]):
            selected = evaluation["record_id"] == record_id
            words = evaluation["word"][selected]
            paths = evaluation["carrier"][selected]
            for left in range(len(words)):
                for right in range(left + 1, len(words)):
                    shared = 0
                    for a, b in zip(words[left], words[right]):
                        if a != b:
                            break
                        shared += 1
                    if shared:
                        prefix_invariance_max = max(
                            prefix_invariance_max,
                            float(np.max(np.abs(paths[left, :shared] - paths[right, :shared]))),
                        )
        simulator_gate = bool(
            np.mean(simulator_gain) >= MIN_SIMULATOR_CONTROL_GAIN
            and simulator_ci[0] > 0
            and np.mean(simulator_control_error) <= MAX_SIMULATOR_CONTROL_NMSE
        )
        native_gate = bool(
            np.mean(native_gain) >= MIN_NATIVE_FIDELITY_GAIN and native_ci[0] > 0
        )
        guard_gate = bool(
            np.mean(guard_gain[crossing]) >= MIN_CROSSING_GUARD_GAIN
            and crossing_guard_ci[0] > 0
            and np.mean(guard_gain) > 0
        )
        specificity_gate = bool(
            np.mean(permutation_advantage) >= MIN_GUARD_CONTROL_ADVANTAGE
            and permutation_ci[0] > 0
            and np.mean(shift_advantage) >= MIN_GUARD_CONTROL_ADVANTAGE
            and shift_ci[0] > 0
        )
        recursive_gate = bool(
            recursive_ratio <= MAX_RECURSIVE_TO_NATIVE_PHYSICAL_RATIO
            and recursive_ratio_ci[1] <= MAX_RECURSIVE_RATIO_CI_UPPER
            and np.mean(composition_error) <= MAX_COMPOSITION_DISCREPANCY_NMSE
            and support_escape <= MAX_RECURSIVE_SUPPORT_ESCAPE_RATE
            and np.mean(carrier_persistence_gain) > 0
            and carrier_persistence_ci[0] > 0
        )
        family_gate = bool(
            all(value <= MAX_LENGTH_FAMILY_RATIO for value in length_ratios.values())
            and all(value <= MAX_MODE_FAMILY_RATIO for value in mode_ratios.values())
        )
        source_gate = bool(
            SOURCE_IDENTITY.get("confirmation_eligible", False)
            and EVALUATION_OPENED
            and len(set(groups.tolist())) >= MIN_EVALUATION_TRAJECTORIES
            and prefix_invariance_max <= 1e-6
        )
        decision = derive_stage35_decision(
            Stage35Gates(
                source_and_split_binding=source_gate,
                simulator_positive_control=simulator_gate,
                native_physical_fidelity=native_gate,
                guard_transfer=guard_gate,
                guard_specificity=specificity_gate,
                recursive_closure=recursive_gate,
                family_consistency=family_gate,
            ),
            run_mode=RUN_MODE,
        )
        SUMMARY = {
            "simulator_control_nmse": float(np.mean(simulator_control_error)),
            "simulator_control_gain": float(np.mean(simulator_gain)),
            "simulator_control_gain_ci95": simulator_ci,
            "native_physical_nmse": float(np.mean(native_physical_error)),
            "native_fidelity_gain": float(np.mean(native_gain)),
            "native_fidelity_gain_ci95": native_ci,
            "predicted_guard_carrier_nmse": float(np.mean(carrier_error)),
            "fixed_source_carrier_nmse": float(np.mean(fixed_error)),
            "oracle_source_carrier_nmse": float(np.mean(oracle_source_error)),
            "oracle_transition_carrier_nmse": float(np.mean(oracle_transition_error)),
            "global_carrier_nmse": float(np.mean(global_error)),
            "crossing_guard_gain": float(np.mean(guard_gain[crossing])),
            "crossing_guard_gain_ci95": crossing_guard_ci,
            "permuted_guard_advantage": float(np.mean(permutation_advantage)),
            "permuted_guard_advantage_ci975": permutation_ci,
            "shifted_guard_advantage": float(np.mean(shift_advantage)),
            "shifted_guard_advantage_ci975": shift_ci,
            "recursive_physical_nmse": float(np.mean(recursive_physical_error)),
            "recursive_to_native_physical_ratio": recursive_ratio,
            "recursive_to_native_ratio_ci95": recursive_ratio_ci,
            "composition_discrepancy_nmse": float(np.mean(composition_error)),
            "recursive_support_escape_rate": support_escape,
            "carrier_persistence_gain": float(np.mean(carrier_persistence_gain)),
            "carrier_persistence_gain_ci95": carrier_persistence_ci,
            "length_ratios": length_ratios,
            "initial_mode_ratios": mode_ratios,
            "crossing_sequences": int(np.sum(crossing)),
            "prefix_invariance_max_abs": prefix_invariance_max,
        }
        DECISION_PAYLOAD = {
            **decision,
            "protocol_id": PROTOCOL_ID,
            "run_signature": RUN_SIGNATURE,
            "source_commit": SOURCE_IDENTITY.get("resolved_commit"),
            "selected_guard": SELECTED_GUARD,
            "summary": SUMMARY,
            "dino_branch_paused": True,
            "native_intervention_run": False,
            "claim_boundary": {
                "fresh_trajectory_panel": True,
                "finite_action_bank": True,
                "evaluation_word_lengths": [5, 6, 7, 8],
                "one_environment": ENVIRONMENT,
                "one_jepa_checkpoint": True,
                "distributed_carrier_sketch_width": PATH_CARRIER_SKETCH_DIM,
                "minimal_state_claimed": False,
                "causal_mechanism_claimed": False,
                "cross_model_claimed": False,
            },
        }
        for index in range(len(evaluation["word"])):
            EVALUATION_ROWS.append({
                "record_id": int(evaluation["record_id"][index]),
                "trajectory_id": int(groups[index]),
                "initial_mode": str(evaluation["initial_mode"][index]),
                "word": str(evaluation["word"][index]),
                "word_length": int(evaluation["length"][index]),
                "crosses_mode": bool(crossing[index]),
                "predicted_guard_carrier_mse": float(carrier_error[index]),
                "fixed_source_carrier_mse": float(fixed_error[index]),
                "oracle_transition_carrier_mse": float(oracle_transition_error[index]),
                "permuted_guard_carrier_mse": float(permuted_error[index]),
                "shifted_guard_carrier_mse": float(shifted_error[index]),
                "native_physical_mse": float(native_physical_error[index]),
                "recursive_physical_mse": float(recursive_physical_error[index]),
                "composition_discrepancy_mse": float(composition_error[index]),
            })
        write_csv(EVIDENCE_DIR / "locked_hybrid_composition_rows.csv", EVALUATION_ROWS)
        write_json(EVIDENCE_DIR / "hybrid_composition_summary.json", SUMMARY)
        write_json(OUT / "stage35_decision.json", DECISION_PAYLOAD)
        atomic_checkpoint("locked_evaluation_complete", {
            "decision_sha256": sha256_file(OUT / "stage35_decision.json"),
            "status": DECISION_PAYLOAD["status"],
            "rows": len(EVALUATION_ROWS),
        })

        figure, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes[0, 0].bar(
            ["persistence", "native", "recursive"],
            [np.mean(physical_persistence_error), np.mean(native_physical_error),
             np.mean(recursive_physical_error)],
            color=["#64748b", "#0ea5e9", "#7c3aed"],
        )
        axes[0, 0].set_title("Locked physical NMSE")
        axes[0, 1].bar(
            ["global", "fixed source", "predicted guard", "oracle guard"],
            [np.mean(global_error), np.mean(fixed_error), np.mean(carrier_error),
             np.mean(oracle_transition_error)],
            color=["#94a3b8", "#f97316", "#2563eb", "#16a34a"],
        )
        axes[0, 1].set_title("Final carrier NMSE")
        axes[0, 1].tick_params(axis="x", rotation=15)
        axes[1, 0].bar(length_ratios.keys(), length_ratios.values(), color="#0ea5e9")
        axes[1, 0].axhline(MAX_LENGTH_FAMILY_RATIO, color="black", linestyle="--")
        axes[1, 0].set_title("Recursive/native ratio by word length")
        control_values = [
            np.mean(guard_gain[crossing]), np.mean(permutation_advantage),
            np.mean(shift_advantage),
        ]
        axes[1, 1].bar(
            ["vs fixed", "vs permuted", "vs shifted"], control_values,
            color=["#2563eb", "#a855f7", "#ec4899"],
        )
        axes[1, 1].axhline(0, color="black", linewidth=1)
        axes[1, 1].set_title("Predicted-guard relative gains")
        figure.suptitle(f"Stage 35: {DECISION_PAYLOAD['status']}")
        figure.tight_layout()
        figure.savefig(PLOT_DIR / "stage35_hybrid_composition_summary.png", dpi=180)
        plt.close(figure)

        interpretation = f"""# Automatic Stage 35 interpretation

Status: **{DECISION_PAYLOAD['status'].upper()}**

The first failed gate is `{DECISION_PAYLOAD['first_failed_gate']}`.

This was a fresh, source-bound JEPA-only observational test.  A full pass means
that a fixed distributed carrier sketch admitted a label-free hybrid recursive
update on the registered unseen action compositions.  It does not establish
causal use, minimality, DINO agreement, or shared neural circuitry.  Native
interchange/path intervention is allowed only after a full pass.
"""
        retry_drive_io(
            "write automatic interpretation",
            lambda: (OUT / "AUTOMATIC_INTERPRETATION.md").write_text(interpretation),
        )
        print(json.dumps(DECISION_PAYLOAD, indent=2))
    except Exception:
        record_failure("stage35_locked_hybrid_composition_evaluation")
'''


packaging = STAGE34.packaging
for old, new in [
    ("Stage 34", "Stage 35"),
    ("stage34", "stage35"),
    ("pfca", "hpcc"),
]:
    packaging = packaging.replace(old, new)
packaging = packaging.replace(
    "raw_roots = [TRUTH_DIR, BASELINE_DIR, FIBER_DIR]",
    "raw_roots = [TRUTH_DIR, PATH_DIR]",
)
packaging = packaging.replace(
    '(OUT / "FAILURE_TRACE.txt").write_text("NONE\\n")',
    'retry_drive_io("finalize failure trace", lambda: (OUT / "FAILURE_TRACE.txt").write_text("NONE\\n"))',
)


protocol_sources = [
    introduction, configuration, installation, setup, analysis_helpers,
    model_helpers, design_and_runtime_helpers, physical_truth,
    construction_and_paths, model_selection, calibration, locked_evaluation,
    packaging,
]
protocol_sources = [value.strip() for value in protocol_sources]
protocol_digest = hashlib.sha256(
    json.dumps(protocol_sources, ensure_ascii=False, separators=(",", ":")).encode()
).hexdigest()
configuration = configuration.replace("__PROTOCOL_DIGEST__", protocol_digest)
if "__PROTOCOL_DIGEST__" in configuration:
    raise RuntimeError("Stage 35 protocol digest placeholder was not replaced")

cells = [
    markdown(introduction),
    code(configuration),
    code(installation),
    code(setup),
    code(analysis_helpers),
    code(model_helpers),
    code(design_and_runtime_helpers),
    code(physical_truth),
    code(construction_and_paths),
    code(model_selection),
    code(calibration),
    code(locked_evaluation),
    code(packaging),
]
for index, cell in enumerate(cells):
    cell["id"] = f"stage35-{index:02d}"

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "L4", "name": TARGET.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
TARGET.write_text(json.dumps(notebook, indent=1) + "\n")
print(f"Wrote {TARGET}")
