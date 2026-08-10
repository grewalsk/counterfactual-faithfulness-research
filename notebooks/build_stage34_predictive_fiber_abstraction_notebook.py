import hashlib
import importlib.util
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
TARGET = ROOT / "34_predictive_fiber_causal_abstraction.ipynb"
NUMERICAL = REPOSITORY / "src/cf_faithfulness/stage34_predictive_fiber_abstraction.py"
STAGE33_NUMERICAL = REPOSITORY / "src/cf_faithfulness/stage33_interventional_abstraction.py"

# Stage 34 reuses the audited model-interface, source-binding, simulator, and
# checkpoint implementation.  The rendered notebook remains self-contained.
spec = importlib.util.spec_from_file_location(
    "stage33_builder",
    ROOT / "build_stage33_bounded_interventional_abstraction_notebook.py",
)
STAGE33 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(STAGE33)

code = STAGE33.code
markdown = STAGE33.markdown
assigned_uppercase_names = STAGE33.assigned_uppercase_names
function_sources = STAGE33.function_sources


def replace_assignment(text, name, expression):
    pattern = rf"(?m)^{re.escape(name)} = .*?$"
    result, count = re.subn(pattern, f"{name} = {expression}", text, count=1)
    if count != 1:
        raise RuntimeError(f"could not replace assignment {name}")
    return result


def remove_assignment(text, name):
    pattern = rf"(?m)^{re.escape(name)} = [^\n]*\n?"
    result, count = re.subn(pattern, "", text, count=1)
    if count != 1:
        raise RuntimeError(f"could not remove assignment {name}")
    return result


def replace_block(text, start, end, replacement):
    left = text.index(start)
    right = text.index(end, left)
    return text[:left] + replacement.rstrip() + "\n" + text[right:]


introduction = r'''# Stage 34: predictive-fiber causal abstraction

## Decision before computation

Stage 33 completed cleanly and rejected the direct shared-realization claim.
Both checkpoints decoded physical state well and admitted a stable rank-three
absolute response chart, but action shuffling was essentially as predictive as
the global operator, physical mode conditioning reversed on locked evaluation,
held-out conjugacy failed, causal interchange was weak, and transported
planning regret increased sharply.  A direct JEPA-to-DINO map is therefore not
the next experiment.

This notebook tests a narrower alternative that separates four questions which
Stage 33 combined:

1. Does cross-model agreement survive after absolute state offsets are removed?
2. Is the resulting bounded action-response state sufficient for unseen future
   action compositions inside each model?
3. Is that state causally used under matched, on-manifold interventions?
4. Do JEPA and DINO each separately commute with one simulator-defined
   high-level transition?

The notebook never fits a JEPA-to-DINO state transformation.  Construction
simulator trajectories define one canonical physical response chart.  The two
models receive separate construction/calibration alignments to that chart and
are judged independently on one locked evaluation panel.

## Bounded action-response state

For model `m`, history `h`, action word `w`, and horizon `k`, define

\[
\Delta_m(h,w,k)=
\widehat\phi_m(h,\operatorname{do}(w_{1:k}))-
\widehat\phi_m(h,\operatorname{do}(0^k)).
\]

Fixed-multiset order contrasts are

\[
\Gamma_m(h;u,v)=\Delta_m(h,uv)-\Delta_m(h,vu).
\]

The simulator-only construction matrix of `Delta` and `Gamma` determines a
frozen chart `s_star`.  Its numerical rank is a finite-bank effective rank, not
the rank of an infinite controlled Hankel operator and not a universal minimal
state.

## Sequential gates

The pilot stops expensive inference after the first failed scientific gate.

1. **Action specificity:** model response coordinates must beat action-word
   shuffles and static-state controls on unseen words.
2. **Predictive sufficiency:** adding a capacity-matched residual carrier sketch
   may improve locked transition prediction by at most 5%, with a 10% upper
   confidence limit; deleting a real response coordinate must hurt.
3. **On-manifold causal use:** predictive-fiber edits must be approximately
   inert, while response-state edits must retain at least half of a same-model
   full-swap positive control and remain within the natural carrier support.
4. **Two-sided commutativity:** each model must match the same frozen physical
   transition within 1.25 times its physical/model reference-error budget and beat
   shuffled controls across every contact stratum and word-length family.

Planning is deliberately deferred.  It becomes meaningful only if all four
gates pass.  A full Stage 34 pass supports a bounded shared *high-level causal
abstraction* for these two checkpoints; it does not identify shared circuitry,
a universal causal state, or a general property of JEPA-style models.  Return
`stage34_pfca_result_bundle_<signature>.zip` and retain the complete resumable
Drive directory containing raw model and intervention shards.

## Methodological anchors

The construction follows three bodies of work without claiming to instantiate
their strongest theorems.  Predictive-state representations motivate defining
state through action-conditional future tests ([Littman, Sutton, and Singh,
2001](https://proceedings.neurips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html)).
Exact and approximate causal abstraction motivate a commuting high-level model
with explicit intervention semantics ([Rubenstein et al.,
2017](https://arxiv.org/abs/1707.00819); [Beckers, Eberhardt, and Halpern,
2020](https://proceedings.mlr.press/v115/beckers20a.html)).  Interchange
intervention analysis motivates testing causal use rather than accepting a
decodable correlate ([Geiger et al., 2021](https://arxiv.org/abs/2106.02997)).
Because this experiment has one simulator, a finite word bank, deterministic
checkpoints, and approximate empirical gates, its conclusion is intentionally
bounded to the registered panel.
'''


configuration = STAGE33.configuration
configuration = configuration.split("\n\nPROTOCOL_CONFIG_KEYS =", 1)[0]
for old, new in [
    ("Stage 33", "Stage 34"),
    ("STAGE33", "STAGE34"),
    ("stage33-bounded-interventional-abstraction", "stage34-predictive-fiber-abstraction"),
    ("counterfactual_faithfulness_stage33_bipca", "counterfactual_faithfulness_stage34_pfca"),
    ("stage33_run_request.json", "stage34_run_request.json"),
    ("stage33-bounded-interventional-predictive-causal-abstraction-v3", "stage34-predictive-fiber-causal-abstraction-v1"),
    ("CONFIRMATORY_V3_ONLY_IF_SOURCE_BOUND_SPLIT_LOCKED_AND_CAUSALLY_TRANSPORTED", "PILOT_V1_ONLY_IF_SOURCE_BOUND_SPLIT_LOCKED_ACTION_SPECIFIC_SUFFICIENT_AND_ON_MANIFOLD"),
    ("notebooks/33_bounded_interventional_predictive_causal_abstraction.ipynb", "notebooks/34_predictive_fiber_causal_abstraction.ipynb"),
    ("notebooks/build_stage33_bounded_interventional_abstraction_notebook.py", "notebooks/build_stage34_predictive_fiber_abstraction_notebook.py"),
    ("src/cf_faithfulness/stage33_interventional_abstraction.py", "src/cf_faithfulness/stage34_predictive_fiber_abstraction.py"),
]:
    configuration = configuration.replace(old, new)

for name, value in {
    "NOTEBOOK_PROTOCOL_SHA256": '"__PROTOCOL_DIGEST__"',
    "MAX_ESTIMATED_TOTAL_MINUTES": "900",
    "SEED": "34103",
    "DESIGN_SEED": "34141",
    "DECODER_SEED": "34183",
    "RANK_SEED": "34213",
    "CALIBRATION_SEED": "34253",
    "BOOTSTRAP_SEED": "34283",
    "MAP_SEED": "34313",
    "CONTROL_SEED": "34351",
    "MAX_WORD_LENGTH": "8",
    "CONSTRUCTION_TRAJECTORY_POOL": "list(range(10000, 11200))",
    "MODEL_SELECTION_TRAJECTORY_POOL": "list(range(11200, 12400))",
    "CALIBRATION_TRAJECTORY_POOL": "list(range(12400, 13600))",
    "EVALUATION_TRAJECTORY_POOL": "list(range(13600, 16000))",
    "CONSTRUCTION_TRAJECTORIES": "16",
    "MODEL_SELECTION_TRAJECTORIES": "16",
    "CALIBRATION_TRAJECTORIES": "16",
    "EVALUATION_TRAJECTORIES": "32",
    "TASK_ID_OFFSET": "34000",
    "MAX_EFFECTIVE_RANK": "16",
    "CARRIER_RANK": "16",
}.items():
    configuration = replace_assignment(configuration, name, value)

configuration = configuration.replace(
    "MAX_WORD_LENGTH = 8", "MAX_WORD_LENGTH = 8\nMAX_COMPOSED_LENGTH = 12"
)

core_words = r'''CORE_WORD_SPECS = [
    {"name": "L", "angles": [-30.0], "magnitudes": [0.14]},
    {"name": "R", "angles": [30.0], "magnitudes": [0.14]},
    {"name": "S", "angles": [0.0], "magnitudes": [0.10]},
    {"name": "LR", "angles": [-30.0, 30.0], "magnitudes": [0.14, 0.14]},
    {"name": "RL", "angles": [30.0, -30.0], "magnitudes": [0.14, 0.14]},
    {"name": "LL", "angles": [-30.0, -30.0], "magnitudes": [0.14, 0.14]},
    {"name": "RR", "angles": [30.0, 30.0], "magnitudes": [0.14, 0.14]},
    {"name": "LRL", "angles": [-30.0, 30.0, -30.0], "magnitudes": [0.14] * 3},
    {"name": "RLR", "angles": [30.0, -30.0, 30.0], "magnitudes": [0.14] * 3},
    {"name": "LLR", "angles": [-30.0, -30.0, 30.0], "magnitudes": [0.14] * 3},
    {"name": "RRL", "angles": [30.0, 30.0, -30.0], "magnitudes": [0.14] * 3},
    {"name": "LRLR", "angles": [-30.0, 30.0, -30.0, 30.0], "magnitudes": [0.14] * 4},
    {"name": "RLRL", "angles": [30.0, -30.0, 30.0, -30.0], "magnitudes": [0.14] * 4},
    {"name": "LLRR", "angles": [-30.0, -30.0, 30.0, 30.0], "magnitudes": [0.14] * 4},
    {"name": "RRLL", "angles": [30.0, 30.0, -30.0, -30.0], "magnitudes": [0.14] * 4},
]
'''
configuration = replace_block(
    configuration, "CORE_WORD_SPECS = [", "CALIBRATION_INTERCHANGE_PAIRS =", core_words
)
configuration = replace_block(
    configuration,
    "CALIBRATION_INTERCHANGE_PAIRS =",
    "EVALUATION_WORD_SPECS =",
    r'''CALIBRATION_INTERCHANGE_PAIRS = [
    ["LR", "RL", 0], ["LLR", "RLL", 0],
    ["RRL", "LRR", 0], ["LRLR", "RLRL", 1],
]
''',
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
    configuration,
    "ZERO_WORD_NAMES",
    '{length: f"zero{length}" for length in range(1, 9)}',
)

configuration += r'''

CORE_ORDER_PAIRS = [("LR", "RL"), ("LRLR", "RLRL"), ("LLRR", "RRLL")]
EVALUATION_ORDER_PAIRS = [
    ("AABAB", "BABAA"), ("AABBAB", "BABAAB"),
    ("AAABBAB", "BABAAAB"), ("AABBABAB", "BABAABBA"),
]
STATE_CARRIER_SKETCH_DIM = 64
TRANSITION_RANDOM_FEATURES = 256
CAUSAL_PAIRS_PER_MODE = 8
CAUSAL_WORDS = ["zero1"]
ACTIVE_CAUSAL_PAIRS_PER_MODE = 1 if RUN_MODE == "smoke" else CAUSAL_PAIRS_PER_MODE

MIN_ACTION_SHUFFLE_ADVANTAGE = 0.10
MAX_RESIDUAL_RELATIVE_IMPROVEMENT = 0.05
MAX_RESIDUAL_CI_UPPER = 0.10
MIN_DELETION_CONTROL_IMPROVEMENT = 0.10
MAX_FIBER_EFFECT_RATIO = 1.25
MIN_STATE_EFFECT_RETENTION = 0.50
MIN_STATE_INTERVENTION_COSINE = 0.20
MAX_INTERVENTION_OOD_RATE = 0.05
MAX_COMMUTATIVITY_REFERENCE_ERROR_RATIO = 1.25
MIN_STAGE34_CONTROL_ADVANTAGE = 0.10

assert MAX_COMPOSED_LENGTH == MAX_WORD_LENGTH + max(
    len(row["name"]) for row in CORE_WORD_SPECS
)
assert {len(row["angles"]) for row in CORE_WORD_SPECS} == {1, 2, 3, 4}
assert {len(row["angles"]) for row in EVALUATION_WORD_SPECS} == {5, 6, 7, 8}
assert set(CAUSAL_WORDS).issubset(set(ZERO_WORD_NAMES.values()))
'''

configuration = re.sub(
    r"PINNED = \[.*?\]\n\nassert INTERVENTION_BLOCK",
    '''PINNED = [
    "real_official_jepa_and_dino_predictions", "exact_pusht_restoration",
    "simulator_only_canonical_response_chart", "no_direct_cross_model_map",
    "no_op_corrected_action_responses", "fixed_multiset_order_contrasts",
    "construction_model_selection_calibration_evaluation_trajectory_splits",
    "length_1_to_4_construction_and_length_5_to_8_evaluation_words",
    "capacity_matched_residual_sufficiency", "predictive_fiber_interventions",
    "on_manifold_ood_gate", "same_model_full_swap_positive_control",
    "two_separate_model_to_physical_commutativity_diagrams",
    "free_pre_contact_contact_post_contact", "shared_target_control",
    "planning_deferred_until_all_stage34_gates_pass", "no_synthetic_fallback",
    "hash_validated_resume", "no_required_colab_secret",
]

assert INTERVENTION_BLOCK''',
    configuration,
    count=1,
    flags=re.S,
)
configuration = configuration.replace(
    "assert MAX_WORD_LENGTH == 4 and STATES_PER_TRAJECTORY == len(MODE_LABELS)",
    "assert MAX_WORD_LENGTH == 8 and STATES_PER_TRAJECTORY == len(MODE_LABELS)",
)
configuration = configuration.replace(
    'assert {len(row["angles"]) for row in CORE_WORD_SPECS} == {1, 2, 3}',
    'assert {len(row["angles"]) for row in CORE_WORD_SPECS} == {1, 2, 3, 4}',
)
configuration = configuration.replace(
    'assert {len(row["angles"]) for row in EVALUATION_WORD_SPECS} == {1, 2, 3, 4}',
    'assert {len(row["angles"]) for row in EVALUATION_WORD_SPECS} == {5, 6, 7, 8}',
)
for legacy_name in [
    "MAP_SEED", "MAX_RANK_DIFFERENCE", "CARRIER_MAP_MAX_CONDITION",
    "MIN_MAP_SINGULAR_VALUE", "NONLINEAR_RANDOM_FEATURES",
    "MIN_DECODER_MEDIAN_R2", "MIN_HYBRID_RELATIVE_GAIN",
    "MIN_LABEL_FREE_GAIN_RETENTION", "MAX_GLOBAL_TO_ACTION_SPECIFIC_ERROR_RATIO",
    "MAX_CONJUGACY_RELATIVE_ERROR", "MAX_SAME_MODEL_SPLIT_HALF_ERROR",
    "MIN_CONTROL_ADVANTAGE", "MIN_GROUNDED_INTERCHANGE_COSINE",
    "MIN_INTERCHANGE_RELATIVE_ERROR_GAIN", "MIN_GROUNDED_EFFECT_ENERGY",
    "MAX_ZERO_EDIT_ERROR", "MAX_PLANNING_REGRET_DEGRADATION",
    "PLANNING_GOALS_PER_RECORD", "MIN_EVALUATION_MODE_TRAJECTORIES",
]:
    configuration = remove_assignment(configuration, legacy_name)
configuration += "\n\nPROTOCOL_CONFIG_KEYS = " + repr(
    assigned_uppercase_names(configuration)
) + "\n"


installation = STAGE33.installation


setup = STAGE33.setup
for old, new in [
    ("Stage 33", "Stage 34"),
    ("STAGE33", "STAGE34"),
    ("stage33", "stage34"),
    ("bipca", "pfca"),
]:
    setup = setup.replace(old, new)
setup += r'''

PHYSICAL_CHART_DIR = OUT / "physical_response_chart"
FIBER_DIR = OUT / "predictive_fibers"
ABSTRACTION_DIR = OUT / "causal_abstraction"
for directory in [PHYSICAL_CHART_DIR, FIBER_DIR, ABSTRACTION_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
'''


analysis_helpers = r'''from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

''' + function_sources(
    STAGE33_NUMERICAL.read_text(),
    [
        "pool_spatial_proprio_features",
        "effective_rank",
        "select_stable_rank",
        "fit_grouped_ridge",
        "clustered_bootstrap_interval",
        "holm_adjust",
    ],
) + "\n\n" + function_sources(
    NUMERICAL.read_text(),
    [
        "_finite_array",
        "_word_lookup",
        "action_contrast_signature",
        "fit_response_chart",
        "response_coordinates",
        "_ordered_unique",
        "grouped_folds",
        "_ridge_fit",
        "grouped_ridge_oof",
        "nested_predictive_sufficiency",
        "fit_supervised_subspace",
        "split_carrier_delta",
        "matched_fiber_pairs",
        "intervention_ood_ratio",
        "cosine_rows",
        "commutativity_metrics",
        "Stage34Gates",
        "derive_stage34_decision",
    ],
)
analysis_helpers = analysis_helpers.replace(
    "class Stage34Gates:\n", "@dataclass(frozen=True)\nclass Stage34Gates:\n"
)


model_helpers = STAGE33.model_helpers.replace("stage33", "stage34").replace(
    "Stage 33", "Stage 34"
)


design_and_runtime_helpers = STAGE33.design_and_runtime_helpers.replace(
    "Stage 32 exposed", "Stage 32 exposed"
)


physical_truth = STAGE33.physical_truth
physical_truth = physical_truth.replace("Stage 33", "Stage 34").replace(
    "stage33", "stage34"
)
physical_truth += r'''


def response_signature_from_truth_path(path, response_words, order_pairs):
    with np.load(path, allow_pickle=False) as payload:
        return action_contrast_signature(
            payload["path_observables"],
            [str(value) for value in payload["word_names"]],
            payload["word_lengths"], response_words, ZERO_WORD_NAMES,
            order_pairs=order_pairs,
        )


def simulator_response_signature(record, response_words=None, order_pairs=None):
    response_words = CORE_WORD_NAMES if response_words is None else list(response_words)
    order_pairs = CORE_ORDER_PAIRS if order_pairs is None else list(order_pairs)
    return response_signature_from_truth_path(
        truth_path(record), response_words, order_pairs
    )


def stage34_jsonable(value):
    if isinstance(value, dict):
        return {str(key): stage34_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [stage34_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return stage34_jsonable(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return stage34_jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


CANONICAL_RESPONSE_CHART = None
CANONICAL_RANK_RESULT = None
CANONICAL_RANK = MIN_COMMON_RANK
CANONICAL_RANK_GATE = False
if not PIPELINE_FAILED:
    try:
        construction_signatures = np.asarray([
            simulator_response_signature(record)
            for record in SELECTED_RECORDS["construction"]
        ])
        selection_signatures = np.asarray([
            simulator_response_signature(record)
            for record in SELECTED_RECORDS["model_selection"]
        ])
        selection_scale = np.maximum(
            np.std(selection_signatures, axis=0, ddof=1), 1e-8
        )
        CANONICAL_RANK_RESULT = select_stable_rank(
            (selection_signatures - np.mean(selection_signatures, axis=0)) / selection_scale,
            np.asarray([
                record["trajectory_id"]
                for record in SELECTED_RECORDS["model_selection"]
            ], dtype=np.int64),
            max_rank=MAX_EFFECTIVE_RANK,
            n_bootstrap=ACTIVE_RANK_BOOTSTRAPS,
            n_permutations=ACTIVE_RANK_PERMUTATIONS,
            stability_floor=RANK_STABILITY_FLOOR,
            null_quantile=RANK_NULL_QUANTILE,
            seed=RANK_SEED,
        )
        selected_rank = int(CANONICAL_RANK_RESULT["rank"])
        CANONICAL_RANK_GATE = bool(selected_rank >= MIN_COMMON_RANK)
        CANONICAL_RANK = min(
            MAX_EFFECTIVE_RANK, max(MIN_COMMON_RANK, selected_rank)
        )
        CANONICAL_RESPONSE_CHART = fit_response_chart(
            construction_signatures, CANONICAL_RANK
        )
        chart_path = PHYSICAL_CHART_DIR / "canonical_response_chart.npz"
        atomic_npz(
            chart_path,
            mean=np.asarray(CANONICAL_RESPONSE_CHART["mean"]),
            scale=np.asarray(CANONICAL_RESPONSE_CHART["scale"]),
            basis=np.asarray(CANONICAL_RESPONSE_CHART["basis"]),
            singular_values=np.asarray(CANONICAL_RESPONSE_CHART["singular_values"]),
            rank=np.asarray(CANONICAL_RANK, dtype=np.int64),
        )
        write_json(PHYSICAL_CHART_DIR / "rank_lock.json", {
            "model_outputs_used": False,
            "construction_trajectory_ids": sorted({
                int(row["trajectory_id"])
                for row in SELECTED_RECORDS["construction"]
            }),
            "model_selection_trajectory_ids": sorted({
                int(row["trajectory_id"])
                for row in SELECTED_RECORDS["model_selection"]
            }),
            "selected_rank": selected_rank,
            "diagnostic_rank": int(CANONICAL_RANK),
            "rank_gate_passed": CANONICAL_RANK_GATE,
            "rank_result": stage34_jsonable(CANONICAL_RANK_RESULT),
            "evaluation_rows_used": 0,
        })
        write_digest_sidecar(PHYSICAL_CHART_DIR / "rank_lock.json")
        atomic_checkpoint("canonical_physical_chart_complete", {
            "chart_sha256": sha256_file(chart_path),
            "rank_lock_sha256": sha256_file(PHYSICAL_CHART_DIR / "rank_lock.json"),
            "rank": int(CANONICAL_RANK),
        })
        print(json.dumps({
            "canonical_physical_response_rank": int(CANONICAL_RANK),
            "rank_gate_passed": CANONICAL_RANK_GATE,
            "construction_rows": len(construction_signatures),
            "model_selection_rows": len(selection_signatures),
            "model_outputs_used": False,
        }, indent=2))
    except Exception:
        record_failure("stage34_simulator_only_canonical_response_chart")
'''


construction_prefix = STAGE33.construction_and_models.split(
    "\nMODEL_ARTIFACTS = {}", 1
)[0]
construction_prefix = construction_prefix.replace(
    "# Fit grounded readouts, predictive charts, and carrier bases on construction trajectories only.",
    "# Fit model-specific grounded readouts and carrier interfaces on construction trajectories only.",
)

construction_and_models = construction_prefix + r'''


def response_signature_from_grounded(grounded, names, split):
    response_words = CORE_WORD_NAMES if split != "evaluation" else EVALUATION_WORD_NAMES
    order_pairs = CORE_ORDER_PAIRS if split != "evaluation" else EVALUATION_ORDER_PAIRS
    lengths = np.asarray([WORD_BY_NAME[name]["length"] for name in names], dtype=np.int64)
    return action_contrast_signature(
        grounded, names, lengths, response_words, ZERO_WORD_NAMES,
        order_pairs=order_pairs,
    )


def canonical_coordinates_from_grounded(grounded, names, split):
    signature = response_signature_from_grounded(grounded, names, split)
    if split == "evaluation":
        # Evaluation words have a different raw signature width.  They are
        # compared directly with simulator evaluation contrasts in Gate 0 and
        # are not projected through the construction chart.
        return signature
    return response_coordinates(signature, CANONICAL_RESPONSE_CHART)


def construction_decoder_and_carrier(bundle):
    names = names_for_split("construction")
    feature_rows, target_rows, groups, carrier_delta_blocks = [], [], [], []
    for index, record in enumerate(SELECTED_RECORDS["construction"]):
        outputs, traces = grouped_model_words(bundle, record, names)
        tensor, _ = feature_tensor_from_outputs(outputs, names)
        x, x_meta = response_rows_from_feature_tensor(tensor, names)
        y, y_meta = truth_rows(record, names)
        if x_meta != y_meta:
            raise RuntimeError("Stage 34 construction decoder row mismatch")
        feature_rows.append(x)
        target_rows.append(y)
        groups.extend([int(record["trajectory_id"])] * len(x))
        deltas, _ = carrier_delta_rows(traces, CALIBRATION_INTERCHANGE_PAIRS)
        carrier_delta_blocks.append(deltas)
        write_json(OUT / f"construction_{bundle['short']}_progress.json", {
            "completed": index + 1,
            "total": len(SELECTED_RECORDS["construction"]),
            "last_record_id": int(record["record_id"]),
        })
    decoder = fit_grouped_ridge(
        np.concatenate(feature_rows), np.concatenate(target_rows),
        np.asarray(groups, dtype=np.int64), penalties=DECODER_RIDGES,
        folds=min(4, len(set(groups))),
        seed=stable_seed(DECODER_SEED, bundle["short"]),
    )
    carrier = fit_dual_basis(
        np.concatenate(carrier_delta_blocks), bundle["carrier_width"], CARRIER_RANK,
        stable_seed(CONTROL_SEED, bundle["short"], "carrier_control"),
    )
    return decoder, carrier


def save_stage34_model_artifacts(short, decoder, carrier, width):
    decoder_path = SUBSPACE_DIR / f"decoder_{short}.npz"
    carrier_path = SUBSPACE_DIR / f"carrier_basis_{short}.npz"
    atomic_npz(
        decoder_path, weight=np.asarray(decoder["weight"]),
        intercept=np.asarray(decoder["intercept"]),
        penalty=np.asarray(decoder["penalty"]),
    )
    atomic_npz(
        carrier_path, mean=np.asarray(carrier["mean"]), scale=np.asarray(carrier["scale"]),
        basis=np.asarray(carrier["basis"]), random_basis=np.asarray(carrier["random_basis"]),
        singular_values=np.asarray(carrier["singular_values"]),
        width=np.asarray(width, dtype=np.int64), rank=np.asarray(carrier["rank"], dtype=np.int64),
    )
    write_json(SUBSPACE_DIR / f"artifact_manifest_{short}.json", {
        "model": short,
        "decoder_sha256": sha256_file(decoder_path),
        "carrier_sha256": sha256_file(carrier_path),
        "decoder_training_split": "construction",
        "carrier_training_split": "construction",
        "canonical_chart_source": "simulator_only",
        "cross_model_parameters": 0,
        "evaluation_rows_used": 0,
        "proprio_feature_pooling": PROPRIO_FEATURE_POOLING,
    })


def load_stage34_model_artifacts(short):
    manifest = json.loads((SUBSPACE_DIR / f"artifact_manifest_{short}.json").read_text())
    result = {}
    for label in ["decoder", "carrier"]:
        path = SUBSPACE_DIR / (
            f"decoder_{short}.npz" if label == "decoder" else f"carrier_basis_{short}.npz"
        )
        validate_digest_sidecar(path)
        if manifest[f"{label}_sha256"] != sha256_file(path):
            raise RuntimeError(f"Stage 34 frozen {short} {label} manifest mismatch")
        with np.load(path, allow_pickle=False) as payload:
            result[label] = {key: payload[key] for key in payload.files}
    return result["decoder"], result["carrier"]


def carrier_state_sketch(carrier, short):
    value = np.asarray(carrier, dtype=np.float32)
    return count_sketch(
        value.reshape(1, -1), STATE_CARRIER_SKETCH_DIM,
        stable_seed(CONTROL_SEED, short, "state_carrier_sketch"),
    )[0]


def generate_stage34_model_record(bundle, record, split, decoder, carrier):
    path = model_path(bundle["short"], record)
    names = names_for_split(split)
    identity = (
        f"{PROTOCOL_ID}:{RUN_SIGNATURE}:{bundle['short']}:{record['record_id']}:"
        f"{REPO_COMMIT}:{EXPECTED_PRETRAINED_ASSET_SHA256[bundle['name'] + '.pth.tar']}"
    )
    required = {
        "identity", "word_names", "word_lengths", "feature_tensor",
        "grounded_predictions", "pair_coordinates", "pair_metadata",
        "response_coordinates", "response_metadata", "state_carrier",
        "state_carrier_sketch", "canonical_coordinates",
    }
    if validate_npz_shard(path, required, identity):
        PROVENANCE_COUNTS["validated_cache_hits"] += 1
        return path
    outputs, traces = grouped_model_words(bundle, record, names)
    tensor, proprio_width = feature_tensor_from_outputs(outputs, names)
    grounded = tensor.astype(np.float64) @ decoder["weight"] + decoder["intercept"]
    pairs = CALIBRATION_INTERCHANGE_PAIRS if split != "evaluation" else EVALUATION_INTERCHANGE_PAIRS
    pair_deltas, pair_meta = carrier_delta_rows(traces, pairs)
    pair_coordinates = project_carrier_deltas(
        pair_deltas, carrier, bundle["carrier_width"]
    )
    response_names = CORE_WORD_NAMES if split != "evaluation" else EVALUATION_WORD_NAMES
    response_carrier, response_meta = projected_action_response_rows(
        traces, response_names, carrier, bundle["carrier_width"]
    )
    zero_trace = traces[ZERO_WORD_NAMES[1]][0].astype(np.float32)
    canonical = canonical_coordinates_from_grounded(grounded, names, split)
    atomic_npz(
        path,
        identity=np.asarray(identity), word_names=np.asarray(names),
        word_lengths=np.asarray([WORD_BY_NAME[name]["length"] for name in names], dtype=np.int64),
        feature_tensor=tensor, proprio_width=np.asarray(proprio_width, dtype=np.int64),
        grounded_predictions=grounded.astype(np.float32),
        pair_coordinates=pair_coordinates.astype(np.float32),
        pair_metadata=np.asarray([json.dumps(row, sort_keys=True) for row in pair_meta]),
        response_coordinates=response_carrier.astype(np.float32),
        response_metadata=np.asarray([json.dumps(row, sort_keys=True) for row in response_meta]),
        state_carrier=zero_trace,
        state_carrier_sketch=carrier_state_sketch(zero_trace, bundle["short"]).astype(np.float32),
        canonical_coordinates=np.asarray(canonical, dtype=np.float32),
    )
    PROVENANCE_COUNTS["model_record_forwards"][bundle["short"]] += 1
    return path


def transition_words_for_split(split):
    return CORE_WORD_NAMES if split in {"model_selection", "calibration"} else EVALUATION_WORD_NAMES


def word_feature_vector(record, name):
    _, invariants = word_actions(record, WORD_BY_NAME[name])
    return np.asarray([
        invariants["impulse"][0], invariants["impulse"][1], invariants["energy"],
        invariants["signed_area"], float(WORD_BY_NAME[name]["length"]),
    ], dtype=np.float64)


def direct_simulator_response_signature(record):
    paths = np.zeros((len(CORE_WORD_NAMES) + 4, 4, len(GROUNDED_OBSERVABLES)))
    names = list(CORE_WORD_NAMES) + [ZERO_WORD_NAMES[length] for length in range(1, 5)]
    lengths = []
    for index, name in enumerate(names):
        rollout = rollout_word(record, WORD_BY_NAME[name], retain_visual=False)
        length = int(WORD_BY_NAME[name]["length"])
        paths[index, :length] = np.asarray([
            grounded_observables(value) for value in rollout["path_states"]
        ])
        lengths.append(length)
    return action_contrast_signature(
        paths, names, np.asarray(lengths), CORE_WORD_NAMES, ZERO_WORD_NAMES,
        order_pairs=CORE_ORDER_PAIRS,
    )


def generate_stage34_transition_record(bundle, record, split, decoder):
    path = transition_path(bundle["short"], record, split)
    identity = (
        f"{PROTOCOL_ID}:{RUN_SIGNATURE}:{bundle['short']}:{record['record_id']}:"
        f"{split}:predictive-fiber-transition-v1"
    )
    required = {
        "identity", "words", "word_lengths", "actions", "source_coordinates",
        "target_coordinates", "simulator_source_coordinates",
        "simulator_target_coordinates", "state_carrier_sketch", "source_mode",
        "target_mode",
    }
    if validate_npz_shard(path, required, identity):
        PROVENANCE_COUNTS["validated_cache_hits"] += 1
        return path
    with np.load(model_path(bundle["short"], record), allow_pickle=False) as base:
        names = [str(value) for value in base["word_names"]]
        grounded = base["grounded_predictions"].astype(np.float64)
        source_q = canonical_coordinates_from_grounded(grounded, names, "construction")
        state_sketch = base["state_carrier_sketch"].astype(np.float64)
    simulator_source_q = response_coordinates(
        simulator_response_signature(record), CANONICAL_RESPONSE_CHART
    )
    words = transition_words_for_split(split)
    target_q, simulator_target_q, actions, target_modes = [], [], [], []
    for name in words:
        successor, rollout = record_after_word(record, name)
        future_names = sorted(
            set(CORE_WORD_NAMES) | {ZERO_WORD_NAMES[length] for length in range(1, 5)},
            key=lambda value: (WORD_BY_NAME[value]["length"], value),
        )
        outputs, _ = grouped_model_words(bundle, successor, future_names)
        tensor, _ = feature_tensor_from_outputs(outputs, future_names)
        successor_grounded = tensor.astype(np.float64) @ decoder["weight"] + decoder["intercept"]
        target_q.append(canonical_coordinates_from_grounded(
            successor_grounded, future_names, "construction"
        ))
        simulator_target_q.append(response_coordinates(
            direct_simulator_response_signature(successor), CANONICAL_RESPONSE_CHART
        ))
        actions.append(word_feature_vector(record, name))
        sequence = source_mode_sequence(record, name, rollout)
        target_modes.append(str(sequence[-1]))
    atomic_npz(
        path, identity=np.asarray(identity), words=np.asarray(words),
        word_lengths=np.asarray([WORD_BY_NAME[name]["length"] for name in words], dtype=np.int64),
        actions=np.asarray(actions),
        source_coordinates=np.repeat(np.asarray(source_q)[None], len(words), axis=0),
        target_coordinates=np.asarray(target_q),
        simulator_source_coordinates=np.repeat(
            np.asarray(simulator_source_q)[None], len(words), axis=0
        ),
        simulator_target_coordinates=np.asarray(simulator_target_q),
        state_carrier_sketch=np.repeat(state_sketch[None], len(words), axis=0),
        source_mode=np.repeat(np.asarray(str(record["mode"])), len(words)),
        target_mode=np.asarray(target_modes),
    )
    return path


MODEL_ARTIFACTS = {}
if not PIPELINE_FAILED:
    try:
        verify_executed_notebook_through(
            "# Fit model-specific grounded readouts and carrier interfaces on construction trajectories only."
        )
        verify_pretrained_assets()
        for model_name in MODEL_NAMES:
            bundle = load_world_model(model_name)
            short = bundle["short"]
            try:
                output_contract = preflight_model_output_contract(bundle)
                artifact_paths = [
                    SUBSPACE_DIR / f"decoder_{short}.npz",
                    SUBSPACE_DIR / f"carrier_basis_{short}.npz",
                    SUBSPACE_DIR / f"artifact_manifest_{short}.json",
                ]
                if all(path.is_file() for path in artifact_paths):
                    decoder, carrier = load_stage34_model_artifacts(short)
                    PROVENANCE_COUNTS["validated_cache_hits"] += 2
                else:
                    decoder, carrier = construction_decoder_and_carrier(bundle)
                    save_stage34_model_artifacts(
                        short, decoder, carrier, bundle["carrier_width"]
                    )
                MODEL_ARTIFACTS[short] = {
                    "decoder": decoder, "carrier": carrier,
                    "carrier_width": int(bundle["carrier_width"]),
                    "output_contract": output_contract,
                }
                for split in ["construction", "model_selection", "calibration", "evaluation"]:
                    for index, record in enumerate(SELECTED_RECORDS[split]):
                        generate_stage34_model_record(
                            bundle, record, split, decoder, carrier
                        )
                        if split in {"model_selection", "calibration", "evaluation"}:
                            generate_stage34_transition_record(
                                bundle, record, split, decoder
                            )
                        write_json(OUT / f"model_{short}_{split}_progress.json", {
                            "completed": index + 1,
                            "total": len(SELECTED_RECORDS[split]),
                            "last_record_id": int(record["record_id"]),
                        })
                memory_report(f"stage34_{short}_model_shards_complete")
            finally:
                unload_world_model(bundle)
        atomic_checkpoint("construction_models_complete", {
            "models": list(MODEL_ARTIFACTS),
            "canonical_chart_sha256": sha256_file(
                PHYSICAL_CHART_DIR / "canonical_response_chart.npz"
            ),
            "artifact_manifests": {
                short: sha256_file(SUBSPACE_DIR / f"artifact_manifest_{short}.json")
                for short in MODEL_ARTIFACTS
            },
        })
    except Exception:
        record_failure("stage34_model_contract_decoder_carrier_or_transition_shards")
'''


action_specificity = r'''# Open the locked evaluation once and test action specificity against state/target controls.


def model_payload(short, record):
    with np.load(model_path(short, record), allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def simulator_evaluation_signature(record):
    return response_signature_from_truth_path(
        truth_path(record), EVALUATION_WORD_NAMES, EVALUATION_ORDER_PAIRS
    )


def shuffled_evaluation_signature(payload, seed):
    names = [str(value) for value in payload["word_names"]]
    grounded = payload["grounded_predictions"].astype(np.float64).copy()
    lookup = {name: index for index, name in enumerate(names)}
    rng = np.random.default_rng(int(seed))
    for length in sorted({WORD_BY_NAME[name]["length"] for name in EVALUATION_WORD_NAMES}):
        words = [name for name in EVALUATION_WORD_NAMES if WORD_BY_NAME[name]["length"] == length]
        order = rng.permutation(len(words))
        original = grounded.copy()
        for target_index, source_index in enumerate(order):
            grounded[lookup[words[target_index]], :length] = original[
                lookup[words[source_index]], :length
            ]
    return response_signature_from_grounded(grounded, names, "evaluation")


def current_grounded_state(record):
    return grounded_observables(np.asarray(record["state"], dtype=np.float64))


ACTION_SPECIFICITY_SUMMARY = {}
ACTION_SPECIFICITY_ROWS = []
ACTION_SPECIFICITY_GATE = False
EVALUATION_OPENED = False
if not PIPELINE_FAILED:
    try:
        verify_executed_notebook_through(
            "# Open the locked evaluation once and test action specificity against state/target controls."
        )
        write_json(ABSTRACTION_DIR / "evaluation_open_certificate.json", {
            "protocol_id": PROTOCOL_ID,
            "run_signature": RUN_SIGNATURE,
            "evaluation_trajectory_ids": sorted({
                int(record["trajectory_id"])
                for record in SELECTED_RECORDS["evaluation"]
            }),
            "canonical_chart_sha256": sha256_file(
                PHYSICAL_CHART_DIR / "canonical_response_chart.npz"
            ),
            "cross_model_map_count": 0,
        })
        write_digest_sidecar(ABSTRACTION_DIR / "evaluation_open_certificate.json")
        EVALUATION_OPENED = True

        calibration_state = np.asarray([
            current_grounded_state(record)
            for record in SELECTED_RECORDS["calibration"]
        ])
        calibration_truth = np.asarray([
            simulator_evaluation_signature(record)
            for record in SELECTED_RECORDS["calibration"]
        ])
        state_baseline = fit_grouped_ridge(
            calibration_state, calibration_truth,
            np.asarray([record["trajectory_id"] for record in SELECTED_RECORDS["calibration"]]),
            penalties=DECODER_RIDGES, folds=4,
            seed=stable_seed(CALIBRATION_SEED, "static_state_baseline"),
        )
        evaluation_state = np.asarray([
            current_grounded_state(record)
            for record in SELECTED_RECORDS["evaluation"]
        ])
        state_prediction = (
            evaluation_state @ state_baseline["weight"] + state_baseline["intercept"]
        )

        for short in ["jepa", "dino"]:
            primary_errors, shuffled_errors, state_errors, groups, modes = [], [], [], [], []
            for index, record in enumerate(SELECTED_RECORDS["evaluation"]):
                payload = model_payload(short, record)
                primary = payload["canonical_coordinates"].astype(np.float64)
                truth = simulator_evaluation_signature(record)
                shuffled = shuffled_evaluation_signature(
                    payload, stable_seed(CONTROL_SEED, short, record["record_id"], "shuffle")
                )
                primary_error = float(np.mean((primary - truth) ** 2))
                shuffled_error = float(np.mean((shuffled - truth) ** 2))
                state_error = float(np.mean((state_prediction[index] - truth) ** 2))
                primary_errors.append(primary_error)
                shuffled_errors.append(shuffled_error)
                state_errors.append(state_error)
                groups.append(int(record["trajectory_id"]))
                modes.append(str(record["mode"]))
                ACTION_SPECIFICITY_ROWS.append({
                    "model": short, "record_id": int(record["record_id"]),
                    "trajectory_id": int(record["trajectory_id"]),
                    "mode": str(record["mode"]),
                    "primary_mse": primary_error, "action_shuffle_mse": shuffled_error,
                    "static_state_mse": state_error,
                })
            primary_errors = np.asarray(primary_errors)
            shuffled_errors = np.asarray(shuffled_errors)
            state_errors = np.asarray(state_errors)
            groups = np.asarray(groups, dtype=np.int64)
            advantage = (shuffled_errors - primary_errors) / np.maximum(shuffled_errors, 1e-12)
            state_advantage = (state_errors - primary_errors) / np.maximum(state_errors, 1e-12)
            mode_means = {
                mode: float(np.mean(advantage[np.asarray(modes) == mode]))
                for mode in MODE_LABELS
            }
            interval = clustered_bootstrap_interval(
                advantage, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
                seed=stable_seed(BOOTSTRAP_SEED, short, "action_specificity"),
                alpha=HOLM_ALPHA,
            )
            ACTION_SPECIFICITY_SUMMARY[short] = {
                "rows": len(advantage),
                "mean_action_shuffle_advantage": float(np.mean(advantage)),
                "action_shuffle_advantage_ci95": interval,
                "mean_static_state_advantage": float(np.mean(state_advantage)),
                "mode_mean_action_advantage": mode_means,
                "passed": bool(
                    np.mean(advantage) >= MIN_ACTION_SHUFFLE_ADVANTAGE
                    and interval[0] > 0
                    and np.mean(state_advantage) > 0
                    and all(value > 0 for value in mode_means.values())
                ),
            }
        ACTION_SPECIFICITY_GATE = bool(
            CANONICAL_RANK_GATE
            and all(value["passed"] for value in ACTION_SPECIFICITY_SUMMARY.values())
        )
        write_csv(
            EVIDENCE_DIR / "locked_action_specificity_rows.csv",
            ACTION_SPECIFICITY_ROWS,
        )
        write_json(EVIDENCE_DIR / "action_specificity_summary.json", {
            "gate_passed": ACTION_SPECIFICITY_GATE,
            "by_model": ACTION_SPECIFICITY_SUMMARY,
            "expensive_gates_will_run": ACTION_SPECIFICITY_GATE,
        })
        atomic_checkpoint("action_specificity_complete", {
            "gate_passed": ACTION_SPECIFICITY_GATE,
            "rows": len(ACTION_SPECIFICITY_ROWS),
        })
        print(json.dumps({
            "action_specificity_gate": ACTION_SPECIFICITY_GATE,
            "summaries": ACTION_SPECIFICITY_SUMMARY,
        }, indent=2))
    except Exception:
        record_failure("stage34_locked_action_specificity")
'''


predictive_sufficiency = r'''# Test whether residual carrier information improves unseen transition prediction.


def load_transition_rows(short, split):
    rows = {key: [] for key in [
        "state", "action", "target", "sim_state", "sim_target", "residual",
        "group", "mode", "word", "length", "record_id",
    ]}
    for record in SELECTED_RECORDS[split]:
        with np.load(transition_path(short, record, split), allow_pickle=False) as payload:
            count = len(payload["words"])
            rows["state"].extend(payload["source_coordinates"])
            rows["action"].extend(payload["actions"])
            rows["target"].extend(payload["target_coordinates"])
            rows["sim_state"].extend(payload["simulator_source_coordinates"])
            rows["sim_target"].extend(payload["simulator_target_coordinates"])
            rows["residual"].extend(payload["state_carrier_sketch"])
            rows["group"].extend([int(record["trajectory_id"])] * count)
            rows["mode"].extend([str(value) for value in payload["source_mode"]])
            rows["word"].extend([str(value) for value in payload["words"]])
            rows["length"].extend(payload["word_lengths"].astype(int).tolist())
            rows["record_id"].extend([int(record["record_id"])] * count)
    for key in ["state", "action", "target", "sim_state", "sim_target", "residual"]:
        rows[key] = np.asarray(rows[key], dtype=np.float64)
    rows["group"] = np.asarray(rows["group"], dtype=np.int64)
    rows["length"] = np.asarray(rows["length"], dtype=np.int64)
    rows["record_id"] = np.asarray(rows["record_id"], dtype=np.int64)
    rows["mode"] = np.asarray(rows["mode"])
    rows["word"] = np.asarray(rows["word"])
    return rows


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
    selection_features = rff_apply(selection_x, parameters)
    if len(set(np.asarray(selection_groups).tolist())) >= 2:
        selected = grouped_ridge_oof(
            selection_features, selection_y, selection_groups,
            penalties=OPERATOR_RIDGES, folds=4, seed=seed,
        )
    else:
        # Smoke has one model-selection trajectory. It exercises the complete
        # path with a frozen ridge but is never scientific evidence.
        selected = {"penalty": float(OPERATOR_RIDGES[0]), "oof_mse": float("nan")}
    calibration_features = rff_apply(calibration_x, parameters)
    weight, intercept = _ridge_fit(
        calibration_features, np.asarray(calibration_y, dtype=np.float64),
        selected["penalty"],
    )
    return {
        "parameters": parameters, "weight": weight, "intercept": intercept,
        "penalty": float(selected["penalty"]),
        "selection_oof_mse": float(selected["oof_mse"]),
    }


def apply_locked_rff(model, inputs):
    return rff_apply(inputs, model["parameters"]) @ model["weight"] + model["intercept"]


SUFFICIENCY_SUMMARY = {}
SUFFICIENCY_ROWS = []
PREDICTIVE_SUFFICIENCY_GATE = False
TRANSITION_DATA = {}
if ACTION_SPECIFICITY_GATE and not PIPELINE_FAILED:
    try:
        verify_executed_notebook_through(
            "# Test whether residual carrier information improves unseen transition prediction."
        )
        for short in ["jepa", "dino"]:
            selection = load_transition_rows(short, "model_selection")
            calibration = load_transition_rows(short, "calibration")
            evaluation = load_transition_rows(short, "evaluation")
            TRANSITION_DATA[short] = evaluation
            base_selection = np.column_stack([selection["state"], selection["action"]])
            base_calibration = np.column_stack([calibration["state"], calibration["action"]])
            base_evaluation = np.column_stack([evaluation["state"], evaluation["action"]])
            enriched_selection = np.column_stack([
                selection["state"], selection["action"], selection["residual"]
            ])
            enriched_calibration = np.column_stack([
                calibration["state"], calibration["action"], calibration["residual"]
            ])
            enriched_evaluation = np.column_stack([
                evaluation["state"], evaluation["action"], evaluation["residual"]
            ])
            deleted_selection = np.column_stack([
                selection["state"][:, :-1], selection["action"]
            ])
            deleted_calibration = np.column_stack([
                calibration["state"][:, :-1], calibration["action"]
            ])
            deleted_evaluation = np.column_stack([
                evaluation["state"][:, :-1], evaluation["action"]
            ])
            base_model = fit_locked_rff(
                base_selection, selection["target"], selection["group"],
                base_calibration, calibration["target"],
                stable_seed(CALIBRATION_SEED, short, "base_transition"),
            )
            enriched_model = fit_locked_rff(
                enriched_selection, selection["target"], selection["group"],
                enriched_calibration, calibration["target"],
                stable_seed(CALIBRATION_SEED, short, "enriched_transition"),
            )
            deleted_model = fit_locked_rff(
                deleted_selection, selection["target"], selection["group"],
                deleted_calibration, calibration["target"],
                stable_seed(CALIBRATION_SEED, short, "deleted_transition"),
            )
            base_prediction = apply_locked_rff(base_model, base_evaluation)
            enriched_prediction = apply_locked_rff(enriched_model, enriched_evaluation)
            deleted_prediction = apply_locked_rff(deleted_model, deleted_evaluation)
            base_error = np.mean((base_prediction - evaluation["target"]) ** 2, axis=1)
            enriched_error = np.mean((enriched_prediction - evaluation["target"]) ** 2, axis=1)
            deleted_error = np.mean((deleted_prediction - evaluation["target"]) ** 2, axis=1)
            residual_gain = (base_error - enriched_error) / np.maximum(base_error, 1e-12)
            deletion_gain = (deleted_error - base_error) / np.maximum(deleted_error, 1e-12)
            residual_ci = clustered_bootstrap_interval(
                residual_gain, evaluation["group"], draws=ACTIVE_BOOTSTRAP_DRAWS,
                seed=stable_seed(BOOTSTRAP_SEED, short, "residual_sufficiency"),
                alpha=HOLM_ALPHA,
            )
            deletion_ci = clustered_bootstrap_interval(
                deletion_gain, evaluation["group"], draws=ACTIVE_BOOTSTRAP_DRAWS,
                seed=stable_seed(BOOTSTRAP_SEED, short, "deletion_control"),
                alpha=HOLM_ALPHA,
            )
            mode_residual = {
                mode: float(np.mean(residual_gain[evaluation["mode"] == mode]))
                for mode in MODE_LABELS
            }
            SUFFICIENCY_SUMMARY[short] = {
                "rows": len(base_error),
                "mean_residual_relative_improvement": float(np.mean(residual_gain)),
                "residual_improvement_ci95": residual_ci,
                "mean_deletion_control_improvement": float(np.mean(deletion_gain)),
                "deletion_control_ci95": deletion_ci,
                "mode_residual_improvements": mode_residual,
                "base_mse": float(np.mean(base_error)),
                "enriched_mse": float(np.mean(enriched_error)),
                "passed": bool(
                    np.mean(residual_gain) <= MAX_RESIDUAL_RELATIVE_IMPROVEMENT
                    and residual_ci[1] <= MAX_RESIDUAL_CI_UPPER
                    and np.mean(deletion_gain) >= MIN_DELETION_CONTROL_IMPROVEMENT
                    and deletion_ci[0] > 0
                    and all(value <= MAX_RESIDUAL_CI_UPPER for value in mode_residual.values())
                ),
            }
            for index in range(len(base_error)):
                SUFFICIENCY_ROWS.append({
                    "model": short, "record_id": int(evaluation["record_id"][index]),
                    "trajectory_id": int(evaluation["group"][index]),
                    "mode": str(evaluation["mode"][index]),
                    "word": str(evaluation["word"][index]),
                    "word_length": int(evaluation["length"][index]),
                    "base_mse": float(base_error[index]),
                    "enriched_mse": float(enriched_error[index]),
                    "deleted_coordinate_mse": float(deleted_error[index]),
                    "residual_relative_improvement": float(residual_gain[index]),
                    "deletion_control_improvement": float(deletion_gain[index]),
                })
        PREDICTIVE_SUFFICIENCY_GATE = bool(
            all(value["passed"] for value in SUFFICIENCY_SUMMARY.values())
        )
        write_csv(EVIDENCE_DIR / "locked_predictive_sufficiency_rows.csv", SUFFICIENCY_ROWS)
        write_json(EVIDENCE_DIR / "predictive_sufficiency_summary.json", {
            "gate_passed": PREDICTIVE_SUFFICIENCY_GATE,
            "by_model": SUFFICIENCY_SUMMARY,
        })
        atomic_checkpoint("predictive_sufficiency_complete", {
            "gate_passed": PREDICTIVE_SUFFICIENCY_GATE,
            "rows": len(SUFFICIENCY_ROWS),
        })
        print(json.dumps({
            "predictive_sufficiency_gate": PREDICTIVE_SUFFICIENCY_GATE,
            "summaries": SUFFICIENCY_SUMMARY,
        }, indent=2))
    except Exception:
        record_failure("stage34_predictive_sufficiency")
elif not ACTION_SPECIFICITY_GATE:
    print("Skipping predictive sufficiency: action-specificity gate failed.")
'''


causal_fibers = r'''# Test matched predictive fibers and response-state edits with on-manifold diagnostics.


def fit_matched_control_basis(carriers, primary_subspace, rank):
    values = np.asarray(carriers, dtype=np.float64)
    white = (values - primary_subspace["mean"]) / primary_subspace["scale"]
    primary = np.asarray(primary_subspace["basis"], dtype=np.float64)
    residual = white - (white @ primary) @ primary.T
    _, singular, right = np.linalg.svd(residual, full_matrices=False)
    keep = min(int(rank), int(np.sum(singular > max(singular[0] * 1e-8, 1e-10))))
    if keep < int(rank):
        raise RuntimeError("matched carrier control is rank deficient")
    return right[: int(rank)].T


def project_delta_to_basis(delta, subspace, basis):
    value = np.asarray(delta, dtype=np.float64).reshape(-1)
    white = value / np.asarray(subspace["scale"], dtype=np.float64)
    projected = np.asarray(basis) @ (np.asarray(basis).T @ white)
    return projected * np.asarray(subspace["scale"], dtype=np.float64)


def final_grounded_prediction(outputs, name, decoder):
    tensor, _ = feature_tensor_from_outputs(outputs, [name])
    grounded = tensor.astype(np.float64) @ decoder["weight"] + decoder["intercept"]
    return grounded[0, WORD_BY_NAME[name]["length"] - 1]


def evaluation_record_arrays(short):
    records = SELECTED_RECORDS["evaluation"]
    q, residual, carriers = [], [], []
    for record in records:
        payload = model_payload(short, record)
        # Matching uses the core canonical state from the transition shard,
        # not the different-width long-word Gate-0 signature.
        with np.load(transition_path(short, record, "evaluation"), allow_pickle=False) as transition:
            q.append(transition["source_coordinates"][0])
        residual.append(payload["state_carrier_sketch"])
        carriers.append(payload["state_carrier"].reshape(-1))
    return np.asarray(q), np.asarray(residual), np.asarray(carriers)


def select_pairs_by_mode(pairs, records):
    selected = []
    used_base_trajectories = set()
    for mode in MODE_LABELS:
        candidates = [
            pair for pair in pairs
            if records[int(pair[0])]["mode"] == mode
        ]
        for pair in candidates:
            trajectory = int(records[int(pair[0])]["trajectory_id"])
            if RUN_MODE != "smoke" and trajectory in used_base_trajectories:
                continue
            selected.append(pair)
            used_base_trajectories.add(trajectory)
            if sum(records[int(value[0])]["mode"] == mode for value in selected) >= ACTIVE_CAUSAL_PAIRS_PER_MODE:
                break
    expected = len(MODE_LABELS) * ACTIVE_CAUSAL_PAIRS_PER_MODE
    if len(selected) != expected:
        raise RuntimeError(
            f"matched-pair panel has {len(selected)} rows; expected {expected}"
        )
    return np.asarray(selected, dtype=np.int64)


CAUSAL_SUMMARY = {}
CAUSAL_ROWS = []
ON_MANIFOLD_CAUSAL_GATE = False
CARRIER_SUBSPACES = {}
if PREDICTIVE_SUFFICIENCY_GATE and not PIPELINE_FAILED:
    try:
        verify_executed_notebook_through(
            "# Test matched predictive fibers and response-state edits with on-manifold diagnostics."
        )
        for short in ["jepa", "dino"]:
            calibration_carriers, calibration_q = [], []
            for record in SELECTED_RECORDS["calibration"]:
                payload = model_payload(short, record)
                calibration_carriers.append(payload["state_carrier"].reshape(-1))
                with np.load(transition_path(short, record, "calibration"), allow_pickle=False) as transition:
                    calibration_q.append(transition["source_coordinates"][0])
            calibration_carriers = np.asarray(calibration_carriers, dtype=np.float64)
            calibration_q = np.asarray(calibration_q, dtype=np.float64)
            causal_rank = (
                1 if RUN_MODE == "smoke"
                else min(CANONICAL_RANK, calibration_q.shape[1])
            )
            primary_subspace = fit_supervised_subspace(
                calibration_carriers, calibration_q, rank=causal_rank, ridge=1e-3
            )
            control_basis = fit_matched_control_basis(
                calibration_carriers, primary_subspace, causal_rank
            )
            CARRIER_SUBSPACES[short] = {
                "primary": primary_subspace, "control_basis": control_basis,
            }
            atomic_npz(
                FIBER_DIR / f"carrier_alignment_{short}.npz",
                mean=np.asarray(primary_subspace["mean"]),
                scale=np.asarray(primary_subspace["scale"]),
                basis=np.asarray(primary_subspace["basis"]),
                control_basis=np.asarray(control_basis),
                singular_values=np.asarray(primary_subspace["singular_values"]),
            )

            q, residual_sketch, _ = evaluation_record_arrays(short)
            records = SELECTED_RECORDS["evaluation"]
            modes = np.asarray([record["mode"] for record in records])
            trajectory_ids = np.asarray([record["trajectory_id"] for record in records])
            pair_sets = {
                kind: select_pairs_by_mode(
                    matched_fiber_pairs(q, residual_sketch, modes, trajectory_ids, kind=kind),
                    records,
                ) for kind in ["fiber", "state"]
            }
            natural_sketches = residual_sketch
            bundle = load_world_model(
                next(name for name in MODEL_NAMES if MODEL_SHORT_NAMES[name] == short)
            )
            try:
                decoder = MODEL_ARTIFACTS[short]["decoder"]
                width = int(bundle["carrier_width"])
                for kind, pairs in pair_sets.items():
                    for pair_index, (base_index, donor_index) in enumerate(pairs):
                        base = records[int(base_index)]
                        donor = records[int(donor_index)]
                        word = CAUSAL_WORDS[pair_index % len(CAUSAL_WORDS)]
                        base_outputs, base_traces = grouped_model_words(bundle, base, [word])
                        donor_outputs, donor_traces = grouped_model_words(bundle, donor, [word])
                        base_prediction = final_grounded_prediction(base_outputs, word, decoder)
                        donor_prediction = final_grounded_prediction(donor_outputs, word, decoder)
                        delta = (
                            donor_traces[word][0] - base_traces[word][0]
                        ).reshape(-1).astype(np.float64)
                        aligned, fiber = split_carrier_delta(delta, primary_subspace)
                        primary_delta = fiber if kind == "fiber" else aligned
                        random_delta = project_delta_to_basis(
                            delta, primary_subspace, control_basis
                        )
                        primary_norm = float(np.linalg.norm(primary_delta))
                        random_norm = float(np.linalg.norm(random_delta))
                        if primary_norm > 1e-12 and random_norm > 1e-12:
                            random_delta = random_delta * (primary_norm / random_norm)
                        conditions = {
                            "primary": primary_delta,
                            "random_matched_subspace": random_delta,
                            "full_swap_positive": delta,
                        }
                        condition_predictions = {}
                        condition_sketches = {}
                        for condition, edit in conditions.items():
                            edit_field = edit.reshape(256, width).astype(np.float32)
                            outputs, traces = grouped_model_words(
                                bundle, base, [word],
                                intervention_lookup={(word, 0): edit_field},
                            )
                            condition_predictions[condition] = final_grounded_prediction(
                                outputs, word, decoder
                            )
                            patched_carrier = base_traces[word][0] + edit_field
                            condition_sketches[condition] = carrier_state_sketch(
                                patched_carrier, short
                            )
                        ood = {
                            condition: float(intervention_ood_ratio(
                                np.asarray([sketch]), natural_sketches
                            )[0])
                            for condition, sketch in condition_sketches.items()
                        }
                        intended = donor_prediction - base_prediction
                        intended_norm = float(np.linalg.norm(intended))
                        for condition, prediction in condition_predictions.items():
                            observed = prediction - base_prediction
                            observed_norm = float(np.linalg.norm(observed))
                            cosine = float(cosine_rows(
                                np.asarray([observed]), np.asarray([intended])
                            )[0])
                            baseline_error = float(np.mean((base_prediction - donor_prediction) ** 2))
                            patched_error = float(np.mean((prediction - donor_prediction) ** 2))
                            error_gain = (baseline_error - patched_error) / max(baseline_error, 1e-12)
                            fiber_ratio = observed_norm / max(intended_norm, 1e-6)
                            CAUSAL_ROWS.append({
                                "model": short, "kind": kind, "condition": condition,
                                "base_record_id": int(base["record_id"]),
                                "donor_record_id": int(donor["record_id"]),
                                "trajectory_id": int(base["trajectory_id"]),
                                "mode": str(base["mode"]), "word": word,
                                "effect_cosine": cosine,
                                "error_gain": float(error_gain),
                                "effect_norm": observed_norm,
                                "intended_norm": intended_norm,
                                "fiber_effect_ratio": float(fiber_ratio),
                                "ood_ratio": ood[condition],
                            })
            finally:
                unload_world_model(bundle)

            model_rows = [row for row in CAUSAL_ROWS if row["model"] == short]
            state_primary = [
                row for row in model_rows if row["kind"] == "state" and row["condition"] == "primary"
            ]
            state_positive = [
                row for row in model_rows if row["kind"] == "state" and row["condition"] == "full_swap_positive"
            ]
            state_random = [
                row for row in model_rows if row["kind"] == "state" and row["condition"] == "random_matched_subspace"
            ]
            fiber_primary = [
                row for row in model_rows if row["kind"] == "fiber" and row["condition"] == "primary"
            ]
            primary_gain = np.asarray([row["error_gain"] for row in state_primary])
            positive_gain = np.asarray([row["error_gain"] for row in state_positive])
            random_gain = np.asarray([row["error_gain"] for row in state_random])
            mean_positive_gain = float(np.mean(positive_gain))
            retention = float(
                np.mean(primary_gain) / max(mean_positive_gain, 1e-12)
            )
            control_advantage = float(np.mean(primary_gain - random_gain))
            fiber_ratio = float(np.mean([
                row["fiber_effect_ratio"] for row in fiber_primary
            ]))
            ood_rate = float(np.mean([
                row["ood_ratio"] > 1.0 for row in state_primary + fiber_primary
            ]))
            mode_retention = {}
            mode_positive_gain = {}
            for mode in MODE_LABELS:
                primary_mode = [row["error_gain"] for row in state_primary if row["mode"] == mode]
                positive_mode = [row["error_gain"] for row in state_positive if row["mode"] == mode]
                mode_positive_gain[mode] = float(np.mean(positive_mode))
                mode_retention[mode] = float(
                    np.mean(primary_mode) / max(mode_positive_gain[mode], 1e-12)
                )
            mean_cosine = float(np.mean([row["effect_cosine"] for row in state_primary]))
            CAUSAL_SUMMARY[short] = {
                "state_rows": len(state_primary), "fiber_rows": len(fiber_primary),
                "mean_state_effect_retention": retention,
                "mean_full_swap_positive_gain": mean_positive_gain,
                "mean_state_effect_cosine": mean_cosine,
                "mean_control_advantage": control_advantage,
                "mean_fiber_effect_ratio": fiber_ratio,
                "intervention_ood_rate": ood_rate,
                "mode_state_retention": mode_retention,
                "mode_full_swap_positive_gain": mode_positive_gain,
                "passed": bool(
                    mean_positive_gain > 0
                    and retention >= MIN_STATE_EFFECT_RETENTION
                    and mean_cosine >= MIN_STATE_INTERVENTION_COSINE
                    and control_advantage >= MIN_STAGE34_CONTROL_ADVANTAGE
                    and fiber_ratio <= MAX_FIBER_EFFECT_RATIO
                    and ood_rate <= MAX_INTERVENTION_OOD_RATE
                    and all(value > 0 for value in mode_positive_gain.values())
                    and all(value > 0 for value in mode_retention.values())
                ),
            }
        ON_MANIFOLD_CAUSAL_GATE = bool(
            all(value["passed"] for value in CAUSAL_SUMMARY.values())
        )
        write_csv(EVIDENCE_DIR / "locked_predictive_fiber_intervention_rows.csv", CAUSAL_ROWS)
        write_json(EVIDENCE_DIR / "predictive_fiber_causal_summary.json", {
            "gate_passed": ON_MANIFOLD_CAUSAL_GATE,
            "by_model": CAUSAL_SUMMARY,
        })
        atomic_checkpoint("on_manifold_causal_use_complete", {
            "gate_passed": ON_MANIFOLD_CAUSAL_GATE,
            "rows": len(CAUSAL_ROWS),
        })
        print(json.dumps({
            "on_manifold_causal_gate": ON_MANIFOLD_CAUSAL_GATE,
            "summaries": CAUSAL_SUMMARY,
        }, indent=2))
    except Exception:
        record_failure("stage34_predictive_fiber_causal_interventions")
elif not PREDICTIVE_SUFFICIENCY_GATE:
    print("Skipping causal fibers: predictive-sufficiency gate failed.")
'''


commutativity_and_decision = r'''# Test two model-to-physical diagrams, apply sequential gates, and interpret.


COMMUTATIVITY_SUMMARY = {}
COMMUTATIVITY_ROWS = []
TWO_SIDED_COMMUTATIVITY_GATE = False
CONTROLS_REJECTED_GATE = False
FAMILY_CONSISTENCY_GATE = False
if ON_MANIFOLD_CAUSAL_GATE and not PIPELINE_FAILED:
    try:
        verify_executed_notebook_through(
            "# Test two model-to-physical diagrams, apply sequential gates, and interpret."
        )
        selection = load_transition_rows("jepa", "model_selection")
        calibration = load_transition_rows("jepa", "calibration")
        evaluation_reference = load_transition_rows("jepa", "evaluation")
        physical_selection_x = np.column_stack([selection["sim_state"], selection["action"]])
        physical_calibration_x = np.column_stack([calibration["sim_state"], calibration["action"]])
        physical_model = fit_locked_rff(
            physical_selection_x, selection["sim_target"], selection["group"],
            physical_calibration_x, calibration["sim_target"],
            stable_seed(CALIBRATION_SEED, "physical_transition"),
        )
        physical_oracle_prediction = apply_locked_rff(
            physical_model,
            np.column_stack([
                evaluation_reference["sim_state"], evaluation_reference["action"]
            ]),
        )
        oracle_relative = np.linalg.norm(
            physical_oracle_prediction - evaluation_reference["sim_target"], axis=1
        ) / np.maximum(
            np.linalg.norm(evaluation_reference["sim_target"], axis=1), 1e-12
        )

        for short in ["jepa", "dino"]:
            evaluation = TRANSITION_DATA[short]
            primary_prediction = apply_locked_rff(
                physical_model,
                np.column_stack([evaluation["state"], evaluation["action"]]),
            )
            rng = np.random.default_rng(stable_seed(CONTROL_SEED, short, "commute_shuffle"))
            shuffled_actions = evaluation["action"][rng.permutation(len(evaluation["action"]))]
            control_prediction = apply_locked_rff(
                physical_model,
                np.column_stack([evaluation["state"], shuffled_actions]),
            )
            target = evaluation["target"]
            measurement_relative = np.linalg.norm(
                target - evaluation["sim_target"], axis=1
            ) / np.maximum(np.linalg.norm(evaluation["sim_target"], axis=1), 1e-12)
            reference_error_budget = max(
                0.05, float(np.mean(oracle_relative + measurement_relative))
            )
            primary_relative = np.linalg.norm(primary_prediction - target, axis=1) / np.maximum(
                np.linalg.norm(target, axis=1), 1e-12
            )
            control_relative = np.linalg.norm(control_prediction - target, axis=1) / np.maximum(
                np.linalg.norm(target, axis=1), 1e-12
            )
            normalized = primary_relative / reference_error_budget
            advantage = (control_relative - primary_relative) / np.maximum(control_relative, 1e-12)
            mode_errors = {
                mode: float(np.mean(normalized[evaluation["mode"] == mode]))
                for mode in MODE_LABELS
            }
            length_errors = {
                str(length): float(np.mean(normalized[evaluation["length"] == length]))
                for length in sorted(set(evaluation["length"].tolist()))
            }
            interval = clustered_bootstrap_interval(
                advantage, evaluation["group"], draws=ACTIVE_BOOTSTRAP_DRAWS,
                seed=stable_seed(BOOTSTRAP_SEED, short, "commutativity_control"),
                alpha=HOLM_ALPHA,
            )
            COMMUTATIVITY_SUMMARY[short] = {
                "rows": len(primary_relative),
                "reference_error_budget": reference_error_budget,
                "mean_relative_error": float(np.mean(primary_relative)),
                "mean_reference_normalized_error": float(np.mean(normalized)),
                "mean_control_advantage": float(np.mean(advantage)),
                "control_advantage_ci95": interval,
                "mode_normalized_errors": mode_errors,
                "word_length_normalized_errors": length_errors,
                "passed": bool(
                    np.mean(normalized) <= MAX_COMMUTATIVITY_REFERENCE_ERROR_RATIO
                    and np.mean(advantage) >= MIN_STAGE34_CONTROL_ADVANTAGE
                    and interval[0] > 0
                    and all(value <= 2 * MAX_COMMUTATIVITY_REFERENCE_ERROR_RATIO for value in mode_errors.values())
                    and all(value <= 2 * MAX_COMMUTATIVITY_REFERENCE_ERROR_RATIO for value in length_errors.values())
                ),
            }
            for index in range(len(primary_relative)):
                COMMUTATIVITY_ROWS.append({
                    "model": short, "record_id": int(evaluation["record_id"][index]),
                    "trajectory_id": int(evaluation["group"][index]),
                    "mode": str(evaluation["mode"][index]),
                    "word": str(evaluation["word"][index]),
                    "word_length": int(evaluation["length"][index]),
                    "primary_relative_error": float(primary_relative[index]),
                    "control_relative_error": float(control_relative[index]),
                    "reference_normalized_error": float(normalized[index]),
                    "control_advantage": float(advantage[index]),
                })
        TWO_SIDED_COMMUTATIVITY_GATE = bool(
            all(value["passed"] for value in COMMUTATIVITY_SUMMARY.values())
        )
        CONTROLS_REJECTED_GATE = bool(
            all(
                ACTION_SPECIFICITY_SUMMARY[short]["mean_action_shuffle_advantage"]
                    >= MIN_ACTION_SHUFFLE_ADVANTAGE
                and CAUSAL_SUMMARY[short]["mean_control_advantage"]
                    >= MIN_STAGE34_CONTROL_ADVANTAGE
                and COMMUTATIVITY_SUMMARY[short]["mean_control_advantage"]
                    >= MIN_STAGE34_CONTROL_ADVANTAGE
                for short in ["jepa", "dino"]
            )
        )
        FAMILY_CONSISTENCY_GATE = bool(
            all(
                all(value > 0 for value in ACTION_SPECIFICITY_SUMMARY[short]["mode_mean_action_advantage"].values())
                and all(value <= MAX_RESIDUAL_CI_UPPER for value in SUFFICIENCY_SUMMARY[short]["mode_residual_improvements"].values())
                and all(value > 0 for value in CAUSAL_SUMMARY[short]["mode_state_retention"].values())
                and all(value <= 2 * MAX_COMMUTATIVITY_REFERENCE_ERROR_RATIO for value in COMMUTATIVITY_SUMMARY[short]["mode_normalized_errors"].values())
                for short in ["jepa", "dino"]
            )
        )
        write_csv(EVIDENCE_DIR / "locked_two_sided_commutativity_rows.csv", COMMUTATIVITY_ROWS)
        write_json(EVIDENCE_DIR / "two_sided_commutativity_summary.json", {
            "gate_passed": TWO_SIDED_COMMUTATIVITY_GATE,
            "controls_rejected": CONTROLS_REJECTED_GATE,
            "family_consistency": FAMILY_CONSISTENCY_GATE,
            "by_model": COMMUTATIVITY_SUMMARY,
        })
    except Exception:
        record_failure("stage34_two_sided_commutativity")
elif not ON_MANIFOLD_CAUSAL_GATE:
    print("Skipping commutativity: on-manifold causal-use gate failed.")


def make_stage34_plot():
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    models = ["jepa", "dino"]
    axes[0, 0].bar(models, [
        ACTION_SPECIFICITY_SUMMARY.get(name, {}).get("mean_action_shuffle_advantage", 0.0)
        for name in models
    ])
    axes[0, 0].axhline(MIN_ACTION_SHUFFLE_ADVANTAGE, color="black", linestyle="--")
    axes[0, 0].set_title("Action-shuffle advantage")
    axes[0, 1].bar(models, [
        SUFFICIENCY_SUMMARY.get(name, {}).get("mean_residual_relative_improvement", 0.0)
        for name in models
    ])
    axes[0, 1].axhline(MAX_RESIDUAL_RELATIVE_IMPROVEMENT, color="black", linestyle="--")
    axes[0, 1].set_title("Residual predictive improvement")
    axes[1, 0].bar(models, [
        CAUSAL_SUMMARY.get(name, {}).get("mean_state_effect_retention", 0.0)
        for name in models
    ])
    axes[1, 0].axhline(MIN_STATE_EFFECT_RETENTION, color="black", linestyle="--")
    axes[1, 0].set_title("On-manifold state-effect retention")
    axes[1, 1].bar(models, [
        COMMUTATIVITY_SUMMARY.get(name, {}).get("mean_reference_normalized_error", 0.0)
        for name in models
    ])
    axes[1, 1].axhline(MAX_COMMUTATIVITY_REFERENCE_ERROR_RATIO, color="black", linestyle="--")
    axes[1, 1].set_title("Commutativity / reference budget")
    figure.tight_layout()
    figure.savefig(PLOT_DIR / "stage34_pfca_summary.png", dpi=180)
    plt.close(figure)


confirmation_eligible = bool(
    SOURCE_IDENTITY.get("confirmation_eligible", False)
    and EVALUATION_OPENED
    and not PIPELINE_FAILED
    and len({
        record["trajectory_id"]
        for record in SELECTED_RECORDS.get("evaluation", [])
    })
        >= MIN_EVALUATION_TRAJECTORIES
)
raw_decision = derive_stage34_decision(
    Stage34Gates(
        action_specificity=ACTION_SPECIFICITY_GATE,
        predictive_sufficiency=PREDICTIVE_SUFFICIENCY_GATE,
        on_manifold_causal_use=ON_MANIFOLD_CAUSAL_GATE,
        two_sided_commutativity=TWO_SIDED_COMMUTATIVITY_GATE,
        controls_rejected=CONTROLS_REJECTED_GATE,
        family_consistency=FAMILY_CONSISTENCY_GATE,
    ),
    run_mode=RUN_MODE,
    confirmation_eligible=confirmation_eligible,
)
status_map = {
    "smoke_only": "SMOKE_ONLY",
    "inconclusive_source_or_split_failure": "INCONCLUSIVE_SOURCE_OR_SPLIT_FAILURE",
    "bounded_two_sided_causal_abstraction_supported": "BOUNDED_TWO_SIDED_CAUSAL_ABSTRACTION_SUPPORTED",
    "shared_static_state_geometry_only": "SHARED_STATIC_STATE_GEOMETRY_ONLY",
    "candidate_predictive_state_insufficient": "CANDIDATE_PREDICTIVE_STATE_INSUFFICIENT",
    "predictive_summary_not_causally_used": "PREDICTIVE_SUMMARY_NOT_CAUSALLY_USED",
    "models_do_not_share_the_high_level_transition": "MODELS_DO_NOT_SHARE_HIGH_LEVEL_TRANSITION",
    "bounded_abstraction_not_supported": "BOUNDED_CAUSAL_ABSTRACTION_NOT_SUPPORTED",
}
status = status_map[raw_decision["status"]]
DECISION_PAYLOAD = {
    "status": status,
    "protocol_decision": raw_decision,
    "confirmation_eligible": confirmation_eligible,
    "canonical_response_rank": int(CANONICAL_RANK),
    "canonical_rank_gate": bool(CANONICAL_RANK_GATE),
    "action_specificity_summary": ACTION_SPECIFICITY_SUMMARY,
    "predictive_sufficiency_summary": SUFFICIENCY_SUMMARY,
    "causal_fiber_summary": CAUSAL_SUMMARY,
    "commutativity_summary": COMMUTATIVITY_SUMMARY,
    "cross_model_map_count": 0,
    "planning_run": False,
    "claim_boundary": {
        "finite_action_bank": True,
        "construction_word_lengths": [1, 2, 3, 4],
        "evaluation_word_lengths": [5, 6, 7, 8],
        "one_environment": ENVIRONMENT,
        "one_checkpoint_per_family": True,
        "shared_dinov2_target_family_confound": True,
        "universal_minimal_state_claimed": False,
        "shared_circuitry_claimed": False,
        "planning_deferred_until_stage34_pass": True,
    },
    "provenance_counts": PROVENANCE_COUNTS,
}
write_json(OUT / "stage34_decision.json", DECISION_PAYLOAD)
write_json(OUT / "run_provenance_certificate.json", {
    "protocol_id": PROTOCOL_ID, "run_signature": RUN_SIGNATURE,
    "run_nonce": RUN_NONCE, "resumed_run": bool(RESUMED_RUN),
    "source_bound": bool(SOURCE_IDENTITY.get("confirmation_eligible", False)),
    "cross_model_map_count": 0,
    "split_trajectory_counts": {
        split: len({record["trajectory_id"] for record in records})
        for split, records in SELECTED_RECORDS.items()
    },
    "provenance_counts": PROVENANCE_COUNTS,
    "confirmation_eligible": confirmation_eligible,
})
make_stage34_plot()
interpretation = f"""# Automatic Stage 34 interpretation

Status: **{status}**

Stage 34 is sequential.  The first failed gate is
`{raw_decision['first_failed_gate']}`.  Later expensive gates are not treated as
negative evidence when they were skipped.

The canonical simulator-only response rank is {CANONICAL_RANK}.  No
JEPA-to-DINO state map was fitted.  The tested object is a finite action-response
chart with construction words of lengths 1--4 and locked evaluation words of
lengths 5--8.

A full pass means that both frozen checkpoints separately realize the same
bounded high-level physical causal abstraction under on-manifold interventions.
It does not establish shared neural circuitry, a universal minimal state, or a
general theorem about JEPA-style models.  Planning remains untested by design.
"""
(OUT / "AUTOMATIC_INTERPRETATION.md").write_text(interpretation)
print(json.dumps(DECISION_PAYLOAD, indent=2))
'''


packaging = STAGE33.packaging
for old, new in [
    ("Stage 33", "Stage 34"),
    ("stage33", "stage34"),
    ("bipca", "pfca"),
]:
    packaging = packaging.replace(old, new)
packaging = packaging.replace(
    "raw_roots = [TRUTH_DIR, BASELINE_DIR, CAUSAL_DIR]",
    "raw_roots = [TRUTH_DIR, BASELINE_DIR, FIBER_DIR]",
)


protocol_sources = [
    introduction, configuration, installation, setup, analysis_helpers,
    model_helpers, design_and_runtime_helpers, physical_truth,
    construction_and_models, action_specificity, predictive_sufficiency,
    causal_fibers, commutativity_and_decision, packaging,
]
protocol_sources = [value.strip() for value in protocol_sources]
protocol_digest = hashlib.sha256(
    json.dumps(protocol_sources, ensure_ascii=False, separators=(",", ":")).encode()
).hexdigest()
configuration = configuration.replace("__PROTOCOL_DIGEST__", protocol_digest)
if "__PROTOCOL_DIGEST__" in configuration:
    raise RuntimeError("protocol digest placeholder was not replaced")

cells = [
    markdown(introduction),
    code(configuration),
    code(installation),
    code(setup),
    code(analysis_helpers),
    code(model_helpers),
    code(design_and_runtime_helpers),
    code(physical_truth),
    code(construction_and_models),
    code(action_specificity),
    code(predictive_sufficiency),
    code(causal_fibers),
    code(commutativity_and_decision),
    code(packaging),
]
for index, cell in enumerate(cells):
    cell["id"] = f"stage34-{index:02d}"

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "L4", "name": TARGET.name, "provenance": []},
        "kernpec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
# Remove the deliberately duplicated typo guard before rendering.
notebook["metadata"].pop("kernpec")
TARGET.write_text(json.dumps(notebook, indent=1) + "\n")
print(f"Wrote {TARGET}")
