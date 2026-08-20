import hashlib
import importlib.util
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
TARGET = ROOT / "36_predictive_state_closure_distillation.ipynb"
NUMERICAL = REPOSITORY / "src/cf_faithfulness/stage36_predictive_state_closure.py"
STAGE35_NUMERICAL = REPOSITORY / "src/cf_faithfulness/stage35_hybrid_composition.py"
STAGE34_NUMERICAL = REPOSITORY / "src/cf_faithfulness/stage34_predictive_fiber_abstraction.py"
STAGE33_NUMERICAL = REPOSITORY / "src/cf_faithfulness/stage33_interventional_abstraction.py"

spec = importlib.util.spec_from_file_location(
    "stage35_builder", ROOT / "build_stage35_hybrid_composition_notebook.py"
)
STAGE35 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(STAGE35)

code = STAGE35.code
markdown = STAGE35.markdown
assigned_uppercase_names = STAGE35.assigned_uppercase_names
function_sources = STAGE35.function_sources
replace_assignment = STAGE35.replace_assignment
replace_block = STAGE35.replace_block


introduction = r'''# Stage 36: predictive-state closure distillation

## V5 registered action-vocabulary repair

The source-bound v4 pilot completed physical truth, passed the simulator-only
rank gate at five, verified the official checkpoint, and loaded JEPA-WM.  It
then stopped before the first model forward because an inherited Stage 35
output preflight requested legacy word `L`, while Stage 36 registers only the
binary `A/B` alphabet.  V5 binds the preflight to the first registered
construction response word, removes legacy executable `L/R/S/a/b` tokens, and
executes the preflight and transition-prefix vocabulary contracts locally.
The observed simulator rank does not change any candidate, threshold, control,
split, gate, or claim.

## V4 retryable exact-source binding

The v3 notebook stopped during setup, before source identity was written or any
simulator/model work began, when GitHub returned HTTP 504 while resolving or
fetching the committed source.  V4 adds bounded exponential-backoff retries for
retryable GitHub/API failures while retaining exact commit resolution, byte
hashing, and executed-cell verification.  It never falls back to unbound or
unverified code.  No scientific configuration or decision changes.

## V3 model-free truth-coverage repair

The source-bound v2 pilot completed all 320 physical-truth records and then
stopped, still before loading JEPA-WM, while building the simulator-only
canonical response chart.  Construction shards contained the registered
canonical response words, but model-selection shards contained only the unseen
length-5--8 task words even though the rank-selection consumer also required
the canonical length-1--4 words and their zero controls.  V3 makes that
readout-only truth dependency explicit, validates the exact per-split word
contract on cache hits, and exercises the real construction and model-selection
schemas before publication.  The held-out model-selection task bank remains
disjoint from construction; no trajectory split, seed, candidate, control,
threshold, gate, or scientific claim changes.

## V2 model-free helper-dependency repair

The source-bound v1 pilot generated and checkpointed every physical-truth
record, then stopped in the simulator-only canonical-response-chart step before
loading JEPA-WM.  The rendered helper cell omitted the inherited Stage 33/34
pooling, grouped-ridge, stable-rank, action-contrast, and response-chart
functions used by later cells.  V2 restores that complete dependency chain and
adds an executable validator for each inherited helper and the truth-path
response signature.  No checkpoint output, carrier, adapter, model-selection
score, evaluation metric, gate, or scientific outcome was observed in v1.  V2
therefore leaves all trajectory pools, seeds, words, candidate models, losses,
thresholds, controls, and claim boundaries unchanged.

## Frozen decision before computation

Stage 35 established an important negative result.  Direct JEPA predictions
were physically strong, but a recursively updated 256-coordinate carrier was
2.66 times worse than native prediction.  Its semantic contact guard was also
falsified: a capacity-matched permuted-label system was better and a shifted
label system was indistinguishable.  The failure occurred with zero carrier-
support escape.  A more elaborate contact classifier is therefore not the
registered next move.

This experiment changes the state definition.  It keeps JEPA-WM completely
frozen and distills its native prefix predictions into a compact recurrent
state that is trained explicitly for recursive composition:

\[
z_t=E(c_{t-h+1:t}),\qquad z_{t+1}=T(z_t,a_t),\qquad
(\hat c_{t+1},\hat y_{t+1})=D(z_{t+1}).
\]

The adapter is called **predictive-state closure distillation (PSCD)**.  Its
loss combines native-carrier recovery, native physical grounding, one-step
teacher-state consistency, free-running multi-step rollout, and direct-versus-
composed latent consistency.  The teacher is the frozen official JEPA-WM
checkpoint; neither its encoder nor its predictor is updated.

## What is new relative to Stage 35

- A nested 1,024-coordinate carrier projection tests whether the previous
  256-coordinate sketch discarded state information.
- Registered native history lengths 1, 2, and 4 test whether a single carrier
  is non-Markov because velocity or contact memory is omitted.
- Every candidate is trained on its own recursive states, eliminating the
  teacher-forcing mismatch measured in Stage 35.
- A single recurrent transition competes with a label-free three-expert
  mixture.  Physical contact labels never select the primary transition.
- Model selection uses unseen words of lengths 5--8.  Locked evaluation uses
  fresh trajectories and unseen words of lengths 9--12.
- Capacity-matched one-step-only, false-history, and Stage-35-style Markov
  controls separate useful state construction from generic nonlinear fitting.

## Sequential gates

1. **Source/split binding:** exact committed source, official checkpoint,
   disjoint fresh panels, and an unopened evaluation certificate.
2. **Simulator positive control:** the registered nonlinear operator class
   must close directly observed physical state.
3. **Native physical fidelity:** frozen direct JEPA predictions must remain
   better than physical persistence.
4. **Distilled-state recovery:** recursive PSCD carriers must beat carrier
   persistence.
5. **Closure improvement:** PSCD must beat both the Markov carrier recursion
   and the one-step-only capacity control; if history is selected, it must also
   beat the false-history control.
6. **Recursive closure:** decoded recursive error must be at most 1.25 times
   native JEPA error, with bounded composition discrepancy and support escape.
7. **Semigroup consistency:** composed latent states must agree with states
   encoded directly from the corresponding native prefix history.
8. **Family consistency:** no evaluation length or starting physical mode may
   hide a catastrophic reversal.

A pass means only that a finite-history adapter can distill bounded recursive
closure from this frozen checkpoint on this finite action bank.  It does not
show that the original JEPA carrier was already closed, that the learned state
is minimal, or that the adapter is causally used by JEPA.

Methodological anchors are predictive-state representations
([Littman, Sutton, and Singh, 2001](https://proceedings.neurips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html)),
closed-loop predictive-state inference
([Sun et al., 2016](https://proceedings.mlr.press/v48/sun16.html)), recurrent
predictive-state networks
([Hefny et al., 2018](https://proceedings.mlr.press/v80/hefny18a.html)), and
direct-versus-composed semigroup consistency
([Tallarico et al., 2026](https://arxiv.org/abs/2605.26324)).
'''


configuration = STAGE35.configuration.split("\n\nPROTOCOL_CONFIG_KEYS =", 1)[0]
for old, new in [
    ("Stage 35", "Stage 36"),
    ("STAGE35", "STAGE36"),
    ("stage35-hybrid-predictive-composition", "stage36-predictive-state-closure"),
    ("counterfactual_faithfulness_stage35_hpcc", "counterfactual_faithfulness_stage36_pscd"),
    ("stage35_run_request.json", "stage36_run_request.json"),
    ("notebooks/35_hybrid_predictive_composition_closure.ipynb", "notebooks/36_predictive_state_closure_distillation.ipynb"),
    ("notebooks/build_stage35_hybrid_composition_notebook.py", "notebooks/build_stage36_predictive_state_closure_notebook.py"),
    ("src/cf_faithfulness/stage35_hybrid_composition.py", "src/cf_faithfulness/stage36_predictive_state_closure.py"),
]:
    configuration = configuration.replace(old, new)

for name, value in {
    "EXPERIMENT_SOURCE_REF": '"codex/stage34-predictive-fiber-abstraction"',
    "PROTOCOL_ID": '"stage36-predictive-state-closure-distillation-v5"',
    "NOTEBOOK_PROTOCOL_SHA256": '"__PROTOCOL_DIGEST__"',
    "EVIDENCE_STATUS": '"FRESH_PROSPECTIVE_JEPA_ONLY_ADAPTER_CLOSURE_TEST"',
    "MAX_ESTIMATED_TOTAL_MINUTES": "360.0",
    "SEED": "36101",
    "DESIGN_SEED": "36141",
    "DECODER_SEED": "36183",
    "RANK_SEED": "36213",
    "CALIBRATION_SEED": "36253",
    "BOOTSTRAP_SEED": "36283",
    "CONTROL_SEED": "36351",
    "MAX_WORD_LENGTH": "12",
    "MAX_COMPOSED_LENGTH": "16",
    "CONSTRUCTION_TRAJECTORY_POOL": "list(range(24000, 25600))",
    "MODEL_SELECTION_TRAJECTORY_POOL": "list(range(25600, 27200))",
    "CALIBRATION_TRAJECTORY_POOL": "list(range(27200, 28800))",
    "EVALUATION_TRAJECTORY_POOL": "list(range(28800, 32000))",
    "CONSTRUCTION_TRAJECTORIES": "16",
    "MODEL_SELECTION_TRAJECTORIES": "16",
    "CALIBRATION_TRAJECTORIES": "16",
    "EVALUATION_TRAJECTORIES": "32",
    "TASK_ID_OFFSET": "36000",
}.items():
    configuration = replace_assignment(configuration, name, value)

configuration = replace_block(
    configuration,
    "CORE_WORD_SPECS = [",
    "CALIBRATION_INTERCHANGE_PAIRS =",
    r'''def stage36_binary_word_spec(name):
    return {
        "name": str(name),
        "angles": [-40.0 if letter == "A" else 40.0 for letter in str(name)],
        "magnitudes": [0.18] * len(str(name)),
    }


STAGE36_CORE_WORD_NAMES = [
    "A", "B", "AA", "BB", "AB", "BA", "AAB", "BBA", "ABA", "BAB",
    "ABBA", "BAAB",
    "ABABA", "BABAB", "AABBAB", "BBAABA", "AAABBAB", "BBABAAB",
    "AABBABAB", "BBAABAAB",
    "ABAAB", "BABBA", "AABABB", "BBABAA", "AABBABA", "BBAABAB",
    "ABBAABBA", "BAABBAAB",
]
CORE_WORD_SPECS = [stage36_binary_word_spec(name) for name in STAGE36_CORE_WORD_NAMES]
''',
)
configuration = replace_block(
    configuration, "CALIBRATION_INTERCHANGE_PAIRS =", "EVALUATION_WORD_SPECS =",
    "CALIBRATION_INTERCHANGE_PAIRS = []\n",
)
configuration = replace_block(
    configuration, "EVALUATION_WORD_SPECS =", "EVALUATION_INTERCHANGE_PAIRS =",
    r'''EVALUATION_WORD_SPECS = [stage36_binary_word_spec(name) for name in [
    "AABBABABA", "BBAABABAB",
    "AAABBABAAB", "BBBAABABBA",
    "AABBABAABAB", "BBAABABBABA",
    "AAABBABAABAB", "BBBAABABBABA",
]]
''',
)
configuration = replace_block(
    configuration, "EVALUATION_INTERCHANGE_PAIRS =", "ZERO_WORD_NAMES =",
    r'''EVALUATION_INTERCHANGE_PAIRS = [
    ["AABBABABA", "BBAABABAB", 0],
    ["AAABBABAAB", "BBBAABABBA", 1],
    ["AABBABAABAB", "BBAABABBABA", 2],
    ["AAABBABAABAB", "BBBAABABBABA", 3],
]
''',
)
configuration = replace_assignment(
    configuration, "ZERO_WORD_NAMES", '{length: f"zero{length}" for length in range(1, 13)}'
)
configuration = replace_assignment(
    configuration, "CORE_ORDER_PAIRS", '[("AB", "BA"), ("ABBA", "BAAB")]'
)
configuration = replace_block(
    configuration, "EVALUATION_ORDER_PAIRS =", "STATE_CARRIER_SKETCH_DIM =",
    r'''EVALUATION_ORDER_PAIRS = [
    ("AABBABABA", "BBAABABAB"),
    ("AAABBABAAB", "BBBAABABBA"),
    ("AABBABAABAB", "BBAABABBABA"),
    ("AAABBABAABAB", "BBBAABABBABA"),
]
''',
)
configuration = configuration.replace(
    "assert MAX_WORD_LENGTH == 8 and STATES_PER_TRAJECTORY == len(MODE_LABELS)",
    "assert MAX_WORD_LENGTH == 12 and STATES_PER_TRAJECTORY == len(MODE_LABELS)",
)
configuration = configuration.replace(
    'assert {len(row["angles"]) for row in CORE_WORD_SPECS} == {1, 2, 3, 4}',
    'assert {len(row["angles"]) for row in CORE_WORD_SPECS} == set(range(1, 9))',
)
configuration = configuration.replace(
    'assert {len(row["angles"]) for row in EVALUATION_WORD_SPECS} == {5, 6, 7, 8}',
    'assert {len(row["angles"]) for row in EVALUATION_WORD_SPECS} == {9, 10, 11, 12}',
)
configuration = re.sub(
    r"\nCONSTRUCTION_WORD_NAMES = .*?\nif RUN_MODE == \"smoke\":\n    # Two complete groups are the minimum for grouped model selection\.\n    ACTIVE_MODEL_SELECTION_TRAJECTORIES = 2\n    ACTIVE_CALIBRATION_TRAJECTORIES = 2\n\nassert .*?\nassert \{len\(row\[\"name\"\]\) for row in EVALUATION_WORD_SPECS\} == \{5, 6, 7, 8\}\n",
    "\n",
    configuration,
    count=1,
    flags=re.S,
)
configuration += r'''

CONSTRUCTION_WORD_NAMES = [
    "A", "B", "AA", "BB", "AB", "BA", "AAB", "BBA", "ABA", "BAB",
    "ABBA", "BAAB",
    "ABAAB", "BABBA", "AABABB", "BBABAA", "AABBABA", "BBAABAB",
    "ABBAABBA", "BAABBAAB",
]
MODEL_SELECTION_WORD_NAMES = [
    "ABABA", "BABAB", "AABBAB", "BBAABA", "AAABBAB", "BBABAAB",
    "AABBABAB", "BBAABAAB",
]
CALIBRATION_WORD_NAMES = [
    "A", "B", "AB", "BA", "AAB", "BBA", "ABBA", "BAAB",
    "ABAAB", "BABBA", "AABABB", "BBABAA", "AABBABA", "BBAABAB",
    "ABBAABBA", "BAABBAAB",
]
CANONICAL_RESPONSE_WORD_NAMES = ["A", "B", "AB", "BA", "AAB", "BBA", "ABBA", "BAAB"]
MAX_CARRIER_PROJECTION_DIM = 1024
CARRIER_PROJECTION_DIMS = [256, 1024] if RUN_MODE == "pilot" else [256]
HISTORY_LENGTHS = [1, 2, 4] if RUN_MODE == "pilot" else [1, 2]
LATENT_DIMS = [64, 128] if RUN_MODE == "pilot" else [32]
DYNAMICS_FAMILIES = ["single", "mixture"] if RUN_MODE == "pilot" else ["single"]
CANDIDATE_EPOCHS = 80
FINAL_EPOCHS = 240
ACTIVE_CANDIDATE_EPOCHS = CANDIDATE_EPOCHS if RUN_MODE == "pilot" else 4
ACTIVE_FINAL_EPOCHS = FINAL_EPOCHS if RUN_MODE == "pilot" else 8
PSCD_LEARNING_RATE = 1e-3
MARKOV_RFF_WIDTH = 128 if RUN_MODE == "pilot" else 32
MARKOV_RIDGE = 1e-2

MIN_SIMULATOR_CONTROL_GAIN = 0.50
MAX_SIMULATOR_CONTROL_NMSE = 0.25
MIN_NATIVE_FIDELITY_GAIN = 0.10
MIN_STATE_RECOVERY_GAIN = 0.10
MIN_CLOSURE_CONTROL_GAIN = 0.05
MAX_RECURSIVE_TO_NATIVE_PHYSICAL_RATIO = 1.25
MAX_RECURSIVE_RATIO_CI_UPPER = 1.50
MAX_COMPOSITION_DISCREPANCY_NMSE = 0.25
MAX_SEMIGROUP_NMSE = 0.25
MAX_RECURSIVE_SUPPORT_ESCAPE_RATE = 0.10
MAX_LENGTH_FAMILY_RATIO = 1.50
MAX_MODE_FAMILY_RATIO = 2.00

if RUN_MODE == "smoke":
    ACTIVE_MODEL_SELECTION_TRAJECTORIES = 2
    ACTIVE_CALIBRATION_TRAJECTORIES = 2

assert set(CONSTRUCTION_WORD_NAMES).issubset(set(STAGE36_CORE_WORD_NAMES))
assert set(MODEL_SELECTION_WORD_NAMES).issubset(set(STAGE36_CORE_WORD_NAMES))
assert set(CALIBRATION_WORD_NAMES).issubset(set(STAGE36_CORE_WORD_NAMES))
assert set(MODEL_SELECTION_WORD_NAMES).isdisjoint(set(CONSTRUCTION_WORD_NAMES))
assert set(CANONICAL_RESPONSE_WORD_NAMES).issubset(set(CONSTRUCTION_WORD_NAMES))
assert {
    name for pair in CORE_ORDER_PAIRS for name in pair
}.issubset(set(CANONICAL_RESPONSE_WORD_NAMES))
assert {len(name) for name in MODEL_SELECTION_WORD_NAMES} == {5, 6, 7, 8}
assert {len(row["name"]) for row in EVALUATION_WORD_SPECS} == {9, 10, 11, 12}
assert max(HISTORY_LENGTHS) < min(len(row["name"]) for row in EVALUATION_WORD_SPECS)
'''

configuration = re.sub(
    r"PINNED = \[.*?\]\n\nassert INTERVENTION_BLOCK",
    '''PINNED = [
    "official_jepa_wm_pusht_checkpoint", "exact_pusht_state_restoration",
    "fresh_disjoint_trajectory_families_24000_to_31999",
    "construction_decoder_and_candidate_training_only",
    "model_selection_unseen_lengths_5_to_8_only",
    "calibration_final_adapter_and_controls_only", "locked_evaluation_once",
    "nested_256_and_1024_carrier_projections", "finite_native_history_1_2_4",
    "frozen_jepa_teacher", "free_running_recursive_training",
    "one_step_only_capacity_control", "false_history_capacity_control",
    "stage35_style_markov_control", "label_free_single_or_mixture_transition",
    "unseen_length_9_to_12_action_compositions",
    "simulator_recursive_positive_control", "native_direct_rollout_reference",
    "adapter_closure_not_original_carrier_closure", "not_minimal_state",
    "observational_not_causal", "dino_branch_paused", "no_synthetic_fallback",
    "hash_validated_resume", "transient_drive_io_retries", "no_required_colab_secret",
    "v2_complete_inherited_helper_dependency_chain_no_scientific_change",
    "v3_complete_truth_consumer_coverage_no_scientific_change",
    "v4_retryable_exact_source_binding_no_scientific_change",
    "v5_registered_action_vocabulary_no_model_outcome_change",
]

assert INTERVENTION_BLOCK''',
    configuration,
    count=1,
    flags=re.S,
)
configuration += "\n\nPROTOCOL_CONFIG_KEYS = " + repr(
    assigned_uppercase_names(configuration)
) + "\n"


installation = STAGE35.installation


setup = STAGE35.setup
for old, new in [
    ("Stage 35", "Stage 36"), ("STAGE35", "STAGE36"),
    ("stage35", "stage36"), ("hpcc", "pscd"),
]:
    setup = setup.replace(old, new)
setup = setup.replace(
    "import urllib.parse\nimport urllib.request",
    "import urllib.error\nimport urllib.parse\nimport urllib.request",
)
setup = setup.replace(
    "def download_asset(name):",
    r'''RETRYABLE_HTTP_STATUS = {408, 425, 429, 500, 502, 503, 504}


def fetch_url_bytes(target, label, attempts=6, timeout_seconds=45.0):
    """Fetch exact bytes with bounded retries for transient network failures."""
    attempts = int(attempts)
    delay = 1.0
    for attempt in range(1, attempts + 1):
        request = target if isinstance(target, urllib.request.Request) else (
            urllib.request.Request(
                str(target), headers={"User-Agent": "stage36-source-binder"}
            )
        )
        try:
            with urllib.request.urlopen(request, timeout=float(timeout_seconds)) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            retryable = int(error.code) in RETRYABLE_HTTP_STATUS
            if not retryable or attempt == attempts:
                raise RuntimeError(
                    f"{label} failed with HTTP {error.code} after {attempt} attempt(s)"
                ) from error
            detail = f"HTTP {error.code}"
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
            if attempt == attempts:
                raise RuntimeError(
                    f"{label} remained unavailable after {attempts} attempts"
                ) from error
            detail = f"{type(error).__name__}: {error}"
        print(
            f"Transient network error during {label} "
            f"(attempt {attempt}/{attempts}): {detail}; retrying in {delay:.0f}s"
        )
        time.sleep(delay)
        delay = min(2.0 * delay, 16.0)
    raise AssertionError("unreachable HTTP retry state")


def download_asset(name):''',
)
setup = replace_block(
    setup,
    "def source_identity():",
    "def verify_executed_notebook_through",
    r'''def source_identity():
    global REMOTE_NOTEBOOK_CODE_CELLS
    payload = {
        "protocol_id": PROTOCOL_ID,
        "notebook_protocol_sha256": NOTEBOOK_PROTOCOL_SHA256,
        "repository": EXPERIMENT_REPOSITORY,
        "source_ref": EXPERIMENT_SOURCE_REF,
        "execution_verified": False,
    }
    if not EXPERIMENT_SOURCE_REF:
        raise RuntimeError("Stage 36 requires its committed GitHub branch")
    source_ref = str(EXPERIMENT_SOURCE_REF).strip()
    if len(source_ref) == 40 and all(
        value in "0123456789abcdef" for value in source_ref.lower()
    ):
        resolved = source_ref.lower()
    else:
        encoded_ref = urllib.parse.quote(source_ref, safe="")
        request = urllib.request.Request(
            f"https://api.github.com/repos/{EXPERIMENT_REPOSITORY}/commits/{encoded_ref}",
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "stage36-source-binder",
            },
        )
        content = fetch_url_bytes(request, f"resolve source ref {source_ref!r}")
        resolved = str(json.loads(content.decode())["sha"]).lower()
    if len(resolved) != 40 or any(
        value not in "0123456789abcdef" for value in resolved
    ):
        raise RuntimeError(
            f"GitHub returned an invalid commit for {source_ref!r}: {resolved!r}"
        )
    payload["resolved_commit"] = resolved
    base = f"https://raw.githubusercontent.com/{EXPERIMENT_REPOSITORY}/{resolved}/"
    payload["files"] = {}
    for label, relative in [
        ("notebook", EXPERIMENT_NOTEBOOK_PATH),
        ("builder", EXPERIMENT_BUILDER_PATH),
        ("numerical", EXPERIMENT_NUMERICAL_PATH),
    ]:
        content = fetch_url_bytes(base + relative, f"fetch committed {label}")
        payload["files"][label] = {
            "path": relative,
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
        }
        if label == "notebook":
            remote_notebook = json.loads(content.decode())
            REMOTE_NOTEBOOK_CODE_CELLS = [
                canonical_cell_source("".join(cell.get("source", [])))
                for cell in remote_notebook["cells"]
                if cell.get("cell_type") == "code"
            ]
            payload["remote_code_cells"] = len(REMOTE_NOTEBOOK_CODE_CELLS)
    payload["status"] = "SOURCE_BOUND_EXECUTION_UNVERIFIED"
    payload["confirmation_eligible"] = False
    return payload


''',
)


stage35_helpers = [
    "_matrix", "_labels", "transition_labels", "flatten_sequence_transitions",
    "_ridge_fit", "fit_rff_ridge", "predict_rff_ridge", "fit_rff_classifier",
    "predict_rff_classifier", "fit_experts", "predict_experts",
    "predict_expert_mixture", "fit_hybrid_family", "recursive_rollout",
    "fit_family_from_sequences", "clustered_mean_interval",
    "clustered_ratio_interval", "fit_support_reference", "support_exceedance_rate",
]
stage36_helpers = [
    "sequence_source_states", "history_tensor", "next_history_tensor",
    "rollout_evaluation_mask", "permute_past_history", "_mean_scale",
    "PredictiveStateClosureModel", "_artifact_model", "fit_predictive_state_closure",
    "rollout_predictive_state_closure", "scaled_path_mse", "relative_gain",
    "select_pscd_candidate", "Stage36Gates", "derive_stage36_decision",
]
analysis_helpers = r'''# Tested predictive-state adapter, controls, metrics, and decision gates.
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from numpy.typing import ArrayLike, NDArray
import torch
from torch import nn

FloatArray = NDArray[np.float64]

''' + function_sources(
    STAGE33_NUMERICAL.read_text(),
    [
        "pool_spatial_proprio_features",
        "effective_rank",
        "select_stable_rank",
        "fit_grouped_ridge",
    ],
) + "\n\n" + function_sources(
    STAGE34_NUMERICAL.read_text(),
    ["_finite_array", "_word_lookup", "action_contrast_signature", "fit_response_chart"],
) + "\n\n" + function_sources(
    STAGE35_NUMERICAL.read_text(), stage35_helpers
) + "\n\n" + function_sources(
    NUMERICAL.read_text(), stage36_helpers
)
analysis_helpers = analysis_helpers.replace(
    "class Stage36Gates:\n", "@dataclass(frozen=True)\nclass Stage36Gates:\n"
)


model_helpers = STAGE35.model_helpers
for old, new in [("stage35", "stage36"), ("Stage 35", "Stage 36")]:
    model_helpers = model_helpers.replace(old, new)


design_and_runtime_helpers = STAGE35.design_and_runtime_helpers
for old, new in [("stage35", "stage36"), ("Stage 35", "Stage 36")]:
    design_and_runtime_helpers = design_and_runtime_helpers.replace(old, new)
design_and_runtime_helpers = design_and_runtime_helpers.replace("3500000", "3600000")
design_and_runtime_helpers = replace_block(
    design_and_runtime_helpers,
    "def token_definition(symbol):",
    "def spec_from_name(name):",
    r'''def token_definition(symbol):
    # Stage 36 has one registered binary alphabet.  Keeping legacy Stage 35
    # tokens executable would allow silent word-manifest drift.
    table = {
        "A": (-40.0, 0.18),
        "B": (40.0, 0.18),
        "0": (0.0, 0.0),
    }
    if symbol not in table:
        raise KeyError(f"unknown Stage 36 action token {symbol!r}")
    return table[symbol]


''',
)
design_and_runtime_helpers = design_and_runtime_helpers.replace(
    '    "v2_scientific_outcomes_observed_before_amendment": False,\n',
    '    "v2_scientific_outcomes_observed_before_amendment": False,\n'
    '    "v3_truth_consumer_coverage_amendment": True,\n'
    '    "v3_scientific_outcomes_observed_before_amendment": False,\n',
)
design_and_runtime_helpers = design_and_runtime_helpers.replace(
    '    "v3_scientific_outcomes_observed_before_amendment": False,\n',
    '    "v3_scientific_outcomes_observed_before_amendment": False,\n'
    '    "v4_source_binding_retry_amendment": True,\n'
    '    "v4_scientific_outcomes_observed_before_amendment": False,\n',
)
design_and_runtime_helpers = design_and_runtime_helpers.replace(
    '    "v4_scientific_outcomes_observed_before_amendment": False,\n',
    '    "v4_scientific_outcomes_observed_before_amendment": False,\n'
    '    "v5_action_vocabulary_amendment": True,\n'
    '    "v5_model_outputs_observed_before_amendment": False,\n'
    '    "v5_locked_evaluation_observed_before_amendment": False,\n',
)


physical_truth = STAGE35.physical_truth
for old, new in [("stage35", "stage36"), ("Stage 35", "Stage 36")]:
    physical_truth = physical_truth.replace(old, new)
physical_truth = replace_block(
    physical_truth,
    "def stage36_truth_word_names(split):",
    "def generate_truth_record(record):",
    r'''def stage36_truth_word_names(split):
    split_names = {
        "construction": CONSTRUCTION_WORD_NAMES,
        "model_selection": MODEL_SELECTION_WORD_NAMES,
        "calibration": CALIBRATION_WORD_NAMES,
        "evaluation": EVALUATION_WORD_NAMES,
    }
    split = str(split)
    if split not in split_names:
        raise ValueError(f"unknown Stage 36 truth split {split!r}")
    names = list(split_names[split])
    if split in {"construction", "model_selection"}:
        # The canonical response chart is fit on construction trajectories and
        # rank-selected on independent model-selection trajectories.  These
        # readout-only simulator words do not alter the held-out task bank.
        names.extend(CANONICAL_RESPONSE_WORD_NAMES)
        names.extend(name for pair in CORE_ORDER_PAIRS for name in pair)
    controls = {
        ZERO_WORD_NAMES[int(WORD_BY_NAME[name]["length"])] for name in names
    }
    result = sorted(
        set(names) | controls,
        key=lambda name: (int(WORD_BY_NAME[name]["length"]), name),
    )
    if split in {"construction", "model_selection"}:
        required = set(CANONICAL_RESPONSE_WORD_NAMES)
        required.update(name for pair in CORE_ORDER_PAIRS for name in pair)
        required.update({
            ZERO_WORD_NAMES[int(WORD_BY_NAME[name]["length"])]
            for name in tuple(required)
        })
        missing = sorted(required - set(result))
        if missing:
            raise RuntimeError(
                f"Stage 36 {split} truth misses canonical response words: {missing}"
            )
    return result


''',
)
physical_truth = physical_truth.replace(
    '''    if validate_npz_shard(path, required, identity):
        PROVENANCE_COUNTS["validated_cache_hits"] += 1
        return path
    names = stage36_truth_word_names(record["split"])
''',
    '''    names = stage36_truth_word_names(record["split"])
    if validate_npz_shard(path, required, identity):
        with np.load(path, allow_pickle=False) as cached:
            cached_names = [str(value) for value in cached["word_names"]]
        if cached_names == names:
            PROVENANCE_COUNTS["validated_cache_hits"] += 1
            return path
        print(
            f"Regenerating truth shard {path.name}: cached word contract changed"
        )
''',
)


construction_and_paths = STAGE35.construction_and_paths
for old, new in [("stage35", "stage36"), ("Stage 35", "Stage 36")]:
    construction_and_paths = construction_and_paths.replace(old, new)
construction_and_paths = construction_and_paths.replace(
    "PATH_CARRIER_SKETCH_DIM", "MAX_CARRIER_PROJECTION_DIM"
)
construction_and_paths = construction_and_paths.replace(
    '    name = "L"\n',
    '''    name = str(CANONICAL_RESPONSE_WORD_NAMES[0])
    if name not in WORD_BY_NAME or name not in CONSTRUCTION_WORD_NAMES:
        raise RuntimeError(
            f"Stage 36 preflight word is outside the registered construction bank: {name!r}"
        )
''',
)
construction_and_paths = replace_block(
    construction_and_paths,
    "def transition_prefixes(split):",
    "def source_mode_sequence(record, word, rollout):",
    r'''def transition_prefixes(split):
    if split == "model_selection":
        names = ["A", "B"]
    elif split == "calibration":
        names = ["A", "B"]
        for base, donor, step in CALIBRATION_INTERCHANGE_PAIRS:
            names.extend([base, donor, donor[: step + 1] + base[step + 1 :]])
    elif split == "evaluation":
        names = list(EVALUATION_WORD_NAMES)
        for base, donor, step in EVALUATION_INTERCHANGE_PAIRS:
            names.extend([base, donor, donor[: step + 1] + base[step + 1 :]])
    else:
        raise ValueError(f"unknown Stage 36 transition split {split!r}")
    names = sorted(set(names), key=lambda value: (len(value), value))
    prefixes = sorted(
        {name[:step] for name in names for step in range(1, len(name) + 1)},
        key=lambda value: (len(value), value),
    )
    invalid = sorted(
        value for value in set(names) | set(prefixes)
        if not value or not set(value).issubset({"A", "B"})
    )
    if invalid:
        raise RuntimeError(f"unregistered Stage 36 transition words: {invalid}")
    return names, prefixes


''',
)
construction_and_paths = replace_block(
    construction_and_paths,
    "def stage36_carrier_sketch(value):",
    "def stage36_mode_paths(record, contact_counts, length):",
    r'''def stage36_carrier_projection(value):
    carrier = np.asarray(value, dtype=np.float32)
    expected = (
        EXPECTED_VISUAL_TOKENS,
        EXPECTED_CARRIER_WIDTHS["jepa_wm_pusht"],
    )
    if carrier.shape != expected:
        raise RuntimeError(f"JEPA carrier shape changed: {carrier.shape}")
    flattened = carrier.reshape(1, -1)
    # The first 256 coordinates are themselves a complete count sketch of the
    # carrier.  The remaining coordinates form an independent 768-coordinate
    # sketch, so both registered candidate widths cover every native feature.
    base = count_sketch(
        flattened, 256,
        stable_seed(CONTROL_SEED, "stage36_nested_carrier_projection", "base"),
    )[0]
    extra = count_sketch(
        flattened, MAX_CARRIER_PROJECTION_DIM - 256,
        stable_seed(CONTROL_SEED, "stage36_nested_carrier_projection", "extra"),
    )[0]
    return np.concatenate([base, extra]).astype(np.float32)


''',
)
construction_and_paths = construction_and_paths.replace(
    "stage36_carrier_sketch(", "stage36_carrier_projection("
)
construction_and_paths = construction_and_paths.replace(
    "prefix-carrier-path-v1", "predictive-state-teacher-path-v1"
)
construction_and_paths = construction_and_paths.replace(
    'for split in ["model_selection", "calibration"]:',
    'for split in ["construction", "model_selection", "calibration"]:',
)
construction_and_paths = construction_and_paths.replace(
    'sum(len(SELECTED_RECORDS[split]) for split in [\n                "model_selection", "calibration"\n            ])',
    'sum(len(SELECTED_RECORDS[split]) for split in [\n                "construction", "model_selection", "calibration"\n            ])',
)


data_loader = STAGE35.data_loader
for old, new in [("stage35", "stage36"), ("Stage 35", "Stage 36")]:
    data_loader = data_loader.replace(old, new)
data_loader = data_loader.replace(
    "# Select predicted-guard capacity using model-selection trajectories only.",
    "# Load split-bound teacher sequences without opening evaluation statistics.",
)


model_selection = data_loader + r'''

def stage36_slice(data, carrier_dim):
    result = dict(data)
    result["initial_carrier"] = data["initial_carrier"][:, : int(carrier_dim)]
    result["carrier"] = data["carrier"][:, :, : int(carrier_dim)]
    return result


def stage36_pscd_scores(artifact, data):
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
            np.mean(carrier_error) + np.mean(physical_error) + 0.25 * np.mean(semigroup_error)
        ),
    }


# Select the state definition on construction-trained models and disjoint unseen words.
SELECTED_PSCD = None
SELECTION_ROWS = []
if not PIPELINE_FAILED:
    try:
        verify_executed_notebook_through(
            "# Select the state definition on construction-trained models and disjoint unseen words."
        )
        construction = load_stage36_sequences("construction")
        selection = load_stage36_sequences("model_selection")
        for carrier_dim in CARRIER_PROJECTION_DIMS:
            train = stage36_slice(construction, carrier_dim)
            validation = stage36_slice(selection, carrier_dim)
            for history_length in HISTORY_LENGTHS:
                for latent_dim in LATENT_DIMS:
                    for dynamics in DYNAMICS_FAMILIES:
                        candidate_seed = stable_seed(
                            CALIBRATION_SEED, "candidate", carrier_dim,
                            history_length, latent_dim, dynamics,
                        )
                        artifact = fit_predictive_state_closure(
                            train["initial_carrier"], train["actions"], train["carrier"],
                            train["native"], train["mask"],
                            history_length=history_length, latent_dim=latent_dim,
                            dynamics=dynamics, epochs=ACTIVE_CANDIDATE_EPOCHS,
                            learning_rate=PSCD_LEARNING_RATE, seed=candidate_seed,
                        )
                        scores = stage36_pscd_scores(artifact, validation)
                        SELECTION_ROWS.append({
                            "carrier_dim": int(carrier_dim),
                            "history_length": int(history_length),
                            "latent_dim": int(latent_dim),
                            "dynamics": str(dynamics),
                            "training_loss_initial": artifact["loss_initial"],
                            "training_loss_final": artifact["loss_final"],
                            **scores,
                        })
                        del artifact
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
        SELECTED_PSCD = select_pscd_candidate(SELECTION_ROWS)
        write_csv(EVIDENCE_DIR / "pscd_model_selection_rows.csv", SELECTION_ROWS)
        selection_path = CALIBRATION_MODEL_DIR / "frozen_pscd_selection.json"
        write_json(selection_path, {
            "protocol_id": PROTOCOL_ID, "selected": SELECTED_PSCD,
            "candidate_rows": SELECTION_ROWS, "training_split": "construction",
            "selection_split": "model_selection", "evaluation_rows_used": 0,
        })
        write_digest_sidecar(selection_path)
        atomic_checkpoint("pscd_model_selection_complete", {
            "selection_sha256": sha256_file(selection_path), "selected": SELECTED_PSCD,
        })
        print(json.dumps({"selected_pscd": SELECTED_PSCD}, indent=2))
    except Exception:
        record_failure("stage36_pscd_model_selection")
'''


calibration = r'''# Freeze the final PSCD adapter and capacity-matched controls before evaluation.
PRIMARY_PSCD = None
ONE_STEP_CONTROL = None
FALSE_HISTORY_CONTROL = None
MARKOV_FAMILY = None
PHYSICAL_FAMILY = None
CARRIER_TO_GROUNDED = None
CARRIER_SCALE = None
PHYSICAL_SCALE = None
STATE_SCALE = None
SUPPORT_REFERENCE = None
CALIBRATION = None
EVALUATION_OPENED = False


def concatenate_stage36_sequences(*bundles):
    result = {}
    for key in bundles[0]:
        result[key] = np.concatenate([bundle[key] for bundle in bundles], axis=0)
    return result


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
            "# Freeze the final PSCD adapter and capacity-matched controls before evaluation."
        )
        if SELECTED_PSCD is None:
            raise RuntimeError("PSCD state definition was not frozen")
        construction = load_stage36_sequences("construction")
        calibration_only = load_stage36_sequences("calibration")
        CALIBRATION = concatenate_stage36_sequences(construction, calibration_only)
        carrier_dim = int(SELECTED_PSCD["carrier_dim"])
        CALIBRATION = stage36_slice(CALIBRATION, carrier_dim)
        fit_kwargs = {
            "history_length": int(SELECTED_PSCD["history_length"]),
            "latent_dim": int(SELECTED_PSCD["latent_dim"]),
            "dynamics": str(SELECTED_PSCD["dynamics"]),
            "epochs": ACTIVE_FINAL_EPOCHS,
            "learning_rate": PSCD_LEARNING_RATE,
        }
        PRIMARY_PSCD = fit_predictive_state_closure(
            CALIBRATION["initial_carrier"], CALIBRATION["actions"],
            CALIBRATION["carrier"], CALIBRATION["native"], CALIBRATION["mask"],
            seed=stable_seed(CALIBRATION_SEED, "primary"), **fit_kwargs,
        )
        ONE_STEP_CONTROL = fit_predictive_state_closure(
            CALIBRATION["initial_carrier"], CALIBRATION["actions"],
            CALIBRATION["carrier"], CALIBRATION["native"], CALIBRATION["mask"],
            seed=stable_seed(CONTROL_SEED, "one_step"), free_weight=0.0,
            consistency_weight=0.25, **fit_kwargs,
        )
        FALSE_HISTORY_CONTROL = None
        if int(SELECTED_PSCD["history_length"]) > 1:
            native_history = history_tensor(
                CALIBRATION["initial_carrier"], CALIBRATION["carrier"],
                CALIBRATION["mask"], int(SELECTED_PSCD["history_length"]),
            )
            false_history = permute_past_history(
                native_history, CALIBRATION["group"], CALIBRATION["mask"],
                seed=stable_seed(CONTROL_SEED, "false_history_train"),
            )
            FALSE_HISTORY_CONTROL = fit_predictive_state_closure(
                CALIBRATION["initial_carrier"], CALIBRATION["actions"],
                CALIBRATION["carrier"], CALIBRATION["native"], CALIBRATION["mask"],
                seed=stable_seed(CONTROL_SEED, "false_history_model"),
                histories_override=false_history, **fit_kwargs,
            )
        MARKOV_FAMILY = fit_family_from_sequences(
            CALIBRATION["initial_carrier"], CALIBRATION["actions"],
            CALIBRATION["carrier"], CALIBRATION["mask"],
            CALIBRATION["source_mode"], CALIBRATION["target_mode"],
            width=MARKOV_RFF_WIDTH, penalty=MARKOV_RIDGE,
            seed=stable_seed(CALIBRATION_SEED, "markov"),
        )
        PHYSICAL_FAMILY = fit_family_from_sequences(
            CALIBRATION["initial_physical"], CALIBRATION["actions"],
            CALIBRATION["simulator"], CALIBRATION["mask"],
            CALIBRATION["source_mode"], CALIBRATION["target_mode"],
            width=MARKOV_RFF_WIDTH, penalty=MARKOV_RIDGE,
            seed=stable_seed(CALIBRATION_SEED, "physical"),
        )
        valid = CALIBRATION["mask"]
        CARRIER_TO_GROUNDED = fit_rff_ridge(
            CALIBRATION["carrier"][valid], CALIBRATION["native"][valid],
            width=MARKOV_RFF_WIDTH, penalty=MARKOV_RIDGE,
            seed=stable_seed(CALIBRATION_SEED, "carrier_to_grounded"),
        )
        CARRIER_SCALE = np.maximum(
            np.std(CALIBRATION["carrier"][valid], axis=0, ddof=1), 1e-8
        )
        PHYSICAL_SCALE = np.maximum(
            np.std(CALIBRATION["simulator"][valid], axis=0, ddof=1), 1e-8
        )
        calibration_rollout = rollout_predictive_state_closure(
            PRIMARY_PSCD, CALIBRATION["initial_carrier"], CALIBRATION["actions"],
            CALIBRATION["carrier"], CALIBRATION["mask"],
        )
        calibration_valid = calibration_rollout["evaluation_mask"]
        STATE_SCALE = np.maximum(
            np.std(calibration_rollout["direct_state"][calibration_valid], axis=0, ddof=1),
            1e-8,
        )
        SUPPORT_REFERENCE = fit_support_reference(
            CALIBRATION["carrier"][calibration_valid]
        )
        arrays, metadata = flatten_model_artifact({
            "primary": PRIMARY_PSCD, "one_step_control": ONE_STEP_CONTROL,
            "false_history_control": FALSE_HISTORY_CONTROL,
            "markov_family": MARKOV_FAMILY, "physical_family": PHYSICAL_FAMILY,
            "carrier_to_grounded": CARRIER_TO_GROUNDED,
            "carrier_scale": CARRIER_SCALE, "physical_scale": PHYSICAL_SCALE,
            "state_scale": STATE_SCALE, "support_reference": SUPPORT_REFERENCE,
        })
        model_path = CALIBRATION_MODEL_DIR / "frozen_stage36_models.npz"
        atomic_npz(model_path, **arrays)
        write_json(CALIBRATION_MODEL_DIR / "frozen_stage36_models_schema.json", metadata)
        certificate_path = CALIBRATION_MODEL_DIR / "evaluation_open_certificate.json"
        write_json(certificate_path, {
            "protocol_id": PROTOCOL_ID, "run_signature": RUN_SIGNATURE,
            "selection_sha256": sha256_file(
                CALIBRATION_MODEL_DIR / "frozen_pscd_selection.json"
            ),
            "models_sha256": sha256_file(model_path),
            "calibration_trajectory_ids": sorted(set(CALIBRATION["group"].tolist())),
            "evaluation_statistics_read": False,
            "evaluation_metrics_computed": False,
            "jepa_parameters_updated": False,
            "primary_uses_simulator_modes": False,
        })
        write_digest_sidecar(certificate_path)
        atomic_checkpoint("calibration_models_frozen", {
            "certificate_sha256": sha256_file(certificate_path),
            "models_sha256": sha256_file(model_path),
        })
        print(json.dumps({
            "selected_pscd": SELECTED_PSCD,
            "final_training_sequences": len(CALIBRATION["word"]),
            "evaluation_opened": EVALUATION_OPENED,
        }, indent=2))
    except Exception:
        record_failure("stage36_calibration_model_freeze")
'''


locked_evaluation = r'''# Open fresh evaluation once and derive every registered Stage 36 gate.
DECISION_PAYLOAD = {"status": "INCONCLUSIVE_PIPELINE_FAILURE"}
EVALUATION_ROWS = []
SUMMARY = {}
if not PIPELINE_FAILED:
    try:
        verify_executed_notebook_through(
            "# Open fresh evaluation once and derive every registered Stage 36 gate."
        )
        certificate_path = CALIBRATION_MODEL_DIR / "evaluation_open_certificate.json"
        validate_digest_sidecar(certificate_path)
        bundle = load_world_model("jepa_wm_pusht")
        try:
            for index, record in enumerate(SELECTED_RECORDS["evaluation"]):
                generate_stage36_path_record(bundle, record, "evaluation", JEPA_DECODER)
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
        evaluation = stage36_slice(
            load_stage36_sequences("evaluation"), int(SELECTED_PSCD["carrier_dim"])
        )
        mask = evaluation["mask"]
        groups = evaluation["group"]
        primary = rollout_predictive_state_closure(
            PRIMARY_PSCD, evaluation["initial_carrier"], evaluation["actions"],
            evaluation["carrier"], mask,
        )
        evaluated = primary["evaluation_mask"]
        one_step = rollout_predictive_state_closure(
            ONE_STEP_CONTROL, evaluation["initial_carrier"], evaluation["actions"],
            evaluation["carrier"], mask,
        )
        false_history = None
        if FALSE_HISTORY_CONTROL is not None:
            native_history = history_tensor(
                evaluation["initial_carrier"], evaluation["carrier"], mask,
                int(SELECTED_PSCD["history_length"]),
            )
            permuted_history = permute_past_history(
                native_history, groups, mask,
                seed=stable_seed(CONTROL_SEED, "false_history_evaluation"),
            )
            false_history = rollout_predictive_state_closure(
                FALSE_HISTORY_CONTROL, evaluation["initial_carrier"],
                evaluation["actions"], evaluation["carrier"], mask,
                histories_override=permuted_history,
            )
        markov_carrier = recursive_rollout(
            MARKOV_FAMILY, evaluation["initial_carrier"], evaluation["actions"], mask,
            strategy="global",
        )
        physical_prediction = recursive_rollout(
            PHYSICAL_FAMILY, evaluation["initial_physical"], evaluation["actions"], mask,
            strategy="global",
        )
        carrier_persistence = np.repeat(
            evaluation["initial_carrier"][:, None, :], MAX_WORD_LENGTH, axis=1
        )
        physical_persistence = np.repeat(
            evaluation["initial_physical"][:, None, :], MAX_WORD_LENGTH, axis=1
        )
        primary_carrier_error = scaled_path_mse(
            primary["carrier"], evaluation["carrier"], evaluated, CARRIER_SCALE
        )
        markov_carrier_error = scaled_path_mse(
            markov_carrier, evaluation["carrier"], evaluated, CARRIER_SCALE
        )
        persistence_carrier_error = scaled_path_mse(
            carrier_persistence, evaluation["carrier"], evaluated, CARRIER_SCALE
        )
        native_physical_error = scaled_path_mse(
            evaluation["native"], evaluation["simulator"], evaluated, PHYSICAL_SCALE
        )
        recursive_physical_error = scaled_path_mse(
            primary["physical"], evaluation["simulator"], evaluated, PHYSICAL_SCALE
        )
        one_step_physical_error = scaled_path_mse(
            one_step["physical"], evaluation["simulator"], evaluated, PHYSICAL_SCALE
        )
        false_history_physical_error = None
        if false_history is not None:
            false_history_physical_error = scaled_path_mse(
                false_history["physical"], evaluation["simulator"], evaluated,
                PHYSICAL_SCALE,
            )
        physical_persistence_error = scaled_path_mse(
            physical_persistence, evaluation["simulator"], evaluated, PHYSICAL_SCALE
        )
        simulator_control_error = scaled_path_mse(
            physical_prediction, evaluation["simulator"], evaluated, PHYSICAL_SCALE
        )
        composition_error = scaled_path_mse(
            primary["physical"], evaluation["native"], evaluated, PHYSICAL_SCALE
        )
        semigroup_error = scaled_path_mse(
            primary["state"], primary["direct_state"], evaluated, STATE_SCALE,
            final_only=False,
        )
        simulator_gain = relative_gain(simulator_control_error, physical_persistence_error)
        native_gain = relative_gain(native_physical_error, physical_persistence_error)
        state_recovery_gain = relative_gain(primary_carrier_error, persistence_carrier_error)
        markov_gain = relative_gain(primary_carrier_error, markov_carrier_error)
        one_step_gain = relative_gain(recursive_physical_error, one_step_physical_error)
        history_gain = None if false_history_physical_error is None else relative_gain(
            recursive_physical_error, false_history_physical_error
        )
        simulator_ci = clustered_mean_interval(
            simulator_gain, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
            seed=stable_seed(BOOTSTRAP_SEED, "simulator"),
        )
        native_ci = clustered_mean_interval(
            native_gain, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
            seed=stable_seed(BOOTSTRAP_SEED, "native"),
        )
        recovery_ci = clustered_mean_interval(
            state_recovery_gain, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
            seed=stable_seed(BOOTSTRAP_SEED, "recovery"),
        )
        markov_ci = clustered_mean_interval(
            markov_gain, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
            seed=stable_seed(BOOTSTRAP_SEED, "markov"),
        )
        one_step_ci = clustered_mean_interval(
            one_step_gain, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
            seed=stable_seed(BOOTSTRAP_SEED, "one_step"),
        )
        history_ci = None if history_gain is None else clustered_mean_interval(
            history_gain, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
            seed=stable_seed(BOOTSTRAP_SEED, "history"),
        )
        recursive_ratio = float(
            np.mean(recursive_physical_error) / max(np.mean(native_physical_error), 1e-12)
        )
        recursive_ratio_ci = clustered_ratio_interval(
            recursive_physical_error, native_physical_error, groups,
            draws=ACTIVE_BOOTSTRAP_DRAWS,
            seed=stable_seed(BOOTSTRAP_SEED, "recursive_ratio"),
        )
        support_escape = support_exceedance_rate(
            SUPPORT_REFERENCE, primary["carrier"][evaluated]
        )
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
        source_gate = bool(
            SOURCE_IDENTITY.get("confirmation_eligible", False)
            and EVALUATION_OPENED
            and len(set(groups.tolist())) >= MIN_EVALUATION_TRAJECTORIES
        )
        simulator_gate = bool(
            np.mean(simulator_gain) >= MIN_SIMULATOR_CONTROL_GAIN
            and simulator_ci[0] > 0
            and np.mean(simulator_control_error) <= MAX_SIMULATOR_CONTROL_NMSE
        )
        native_gate = bool(
            np.mean(native_gain) >= MIN_NATIVE_FIDELITY_GAIN and native_ci[0] > 0
        )
        recovery_gate = bool(
            np.mean(state_recovery_gain) >= MIN_STATE_RECOVERY_GAIN and recovery_ci[0] > 0
        )
        improvement_gate = bool(
            np.mean(markov_gain) >= MIN_CLOSURE_CONTROL_GAIN and markov_ci[0] > 0
            and np.mean(one_step_gain) >= MIN_CLOSURE_CONTROL_GAIN and one_step_ci[0] > 0
            and (
                history_gain is None
                or (np.mean(history_gain) >= MIN_CLOSURE_CONTROL_GAIN and history_ci[0] > 0)
            )
        )
        recursive_gate = bool(
            recursive_ratio <= MAX_RECURSIVE_TO_NATIVE_PHYSICAL_RATIO
            and recursive_ratio_ci[1] <= MAX_RECURSIVE_RATIO_CI_UPPER
            and np.mean(composition_error) <= MAX_COMPOSITION_DISCREPANCY_NMSE
            and support_escape <= MAX_RECURSIVE_SUPPORT_ESCAPE_RATE
        )
        semigroup_gate = bool(np.mean(semigroup_error) <= MAX_SEMIGROUP_NMSE)
        family_gate = bool(
            all(value <= MAX_LENGTH_FAMILY_RATIO for value in length_ratios.values())
            and all(value <= MAX_MODE_FAMILY_RATIO for value in mode_ratios.values())
        )
        decision = derive_stage36_decision(
            Stage36Gates(
                source_and_split_binding=source_gate,
                simulator_positive_control=simulator_gate,
                native_physical_fidelity=native_gate,
                distilled_state_recovery=recovery_gate,
                closure_improvement=improvement_gate,
                recursive_closure=recursive_gate,
                semigroup_consistency=semigroup_gate,
                family_consistency=family_gate,
            ),
            run_mode=RUN_MODE,
        )
        SUMMARY = {
            "selected_pscd": SELECTED_PSCD,
            "simulator_control_nmse": float(np.mean(simulator_control_error)),
            "simulator_control_gain": float(np.mean(simulator_gain)),
            "simulator_control_gain_ci95": simulator_ci,
            "native_physical_nmse": float(np.mean(native_physical_error)),
            "native_fidelity_gain": float(np.mean(native_gain)),
            "native_fidelity_gain_ci95": native_ci,
            "recursive_physical_nmse": float(np.mean(recursive_physical_error)),
            "recursive_to_native_physical_ratio": recursive_ratio,
            "recursive_to_native_ratio_ci95": recursive_ratio_ci,
            "composition_discrepancy_nmse": float(np.mean(composition_error)),
            "semigroup_nmse": float(np.mean(semigroup_error)),
            "recursive_support_escape_rate": support_escape,
            "state_recovery_gain": float(np.mean(state_recovery_gain)),
            "state_recovery_gain_ci95": recovery_ci,
            "markov_control_gain": float(np.mean(markov_gain)),
            "markov_control_gain_ci95": markov_ci,
            "one_step_control_gain": float(np.mean(one_step_gain)),
            "one_step_control_gain_ci95": one_step_ci,
            "false_history_control_gain": None if history_gain is None else float(np.mean(history_gain)),
            "false_history_control_gain_ci95": history_ci,
            "length_ratios": length_ratios,
            "initial_mode_ratios": mode_ratios,
        }
        DECISION_PAYLOAD = {
            **decision, "protocol_id": PROTOCOL_ID, "run_signature": RUN_SIGNATURE,
            "source_commit": SOURCE_IDENTITY.get("resolved_commit"),
            "selected_pscd": SELECTED_PSCD,
            "summary": SUMMARY, "dino_branch_paused": True,
            "native_intervention_run": False,
            "claim_boundary": {
                "fresh_trajectory_panel": True, "finite_action_bank": True,
                "evaluation_word_lengths": [9, 10, 11, 12],
                "one_environment": ENVIRONMENT, "one_jepa_checkpoint": True,
                "jepa_parameters_updated": False,
                "original_jepa_carrier_claimed_closed": False,
                "minimal_state_claimed": False, "causal_mechanism_claimed": False,
            },
        }
        for index in range(len(evaluation["word"])):
            EVALUATION_ROWS.append({
                "record_id": int(evaluation["record_id"][index]),
                "trajectory_id": int(groups[index]),
                "initial_mode": str(evaluation["initial_mode"][index]),
                "word": str(evaluation["word"][index]),
                "word_length": int(evaluation["length"][index]),
                "native_physical_mse": float(native_physical_error[index]),
                "recursive_physical_mse": float(recursive_physical_error[index]),
                "primary_carrier_mse": float(primary_carrier_error[index]),
                "markov_carrier_mse": float(markov_carrier_error[index]),
                "one_step_physical_mse": float(one_step_physical_error[index]),
                "semigroup_mse": float(semigroup_error[index]),
            })
        write_csv(EVIDENCE_DIR / "locked_predictive_state_closure_rows.csv", EVALUATION_ROWS)
        write_json(EVIDENCE_DIR / "predictive_state_closure_summary.json", SUMMARY)
        write_json(OUT / "stage36_decision.json", DECISION_PAYLOAD)
        atomic_checkpoint("locked_evaluation_complete", {
            "decision_sha256": sha256_file(OUT / "stage36_decision.json"),
            "status": DECISION_PAYLOAD["status"], "rows": len(EVALUATION_ROWS),
        })

        figure, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes[0, 0].bar(
            ["persistence", "native", "PSCD"],
            [np.mean(physical_persistence_error), np.mean(native_physical_error),
             np.mean(recursive_physical_error)],
            color=["#64748b", "#0ea5e9", "#7c3aed"],
        )
        axes[0, 0].set_title("Locked physical NMSE")
        axes[0, 1].bar(
            ["persistence", "Markov", "PSCD"],
            [np.mean(persistence_carrier_error), np.mean(markov_carrier_error),
             np.mean(primary_carrier_error)],
            color=["#64748b", "#f97316", "#2563eb"],
        )
        axes[0, 1].set_title("Locked carrier NMSE")
        axes[1, 0].bar(length_ratios.keys(), length_ratios.values(), color="#0ea5e9")
        axes[1, 0].axhline(MAX_LENGTH_FAMILY_RATIO, color="black", linestyle="--")
        axes[1, 0].set_title("Recursive/native ratio by word length")
        controls = [np.mean(markov_gain), np.mean(one_step_gain)]
        labels = ["vs Markov", "vs one-step"]
        if history_gain is not None:
            controls.append(np.mean(history_gain))
            labels.append("vs false history")
        axes[1, 1].bar(labels, controls, color=["#2563eb", "#a855f7", "#ec4899"])
        axes[1, 1].axhline(0, color="black", linewidth=1)
        axes[1, 1].set_title("PSCD relative gains")
        figure.suptitle(f"Stage 36: {DECISION_PAYLOAD['status']}")
        figure.tight_layout()
        figure.savefig(PLOT_DIR / "stage36_predictive_state_closure_summary.png", dpi=180)
        plt.close(figure)

        interpretation = f"""# Automatic Stage 36 interpretation

Status: **{DECISION_PAYLOAD['status'].upper()}**

The first failed gate is `{DECISION_PAYLOAD['first_failed_gate']}`.

This was a fresh, source-bound JEPA-only adapter test.  A full pass means that
a frozen finite-history predictive-state adapter closed on the registered
unseen action compositions.  It does not mean the original JEPA carrier was
already recursively closed, does not identify a minimal state, and supplies no
native causal-intervention evidence.
"""
        retry_drive_io(
            "write automatic interpretation",
            lambda: (OUT / "AUTOMATIC_INTERPRETATION.md").write_text(interpretation),
        )
        print(json.dumps(DECISION_PAYLOAD, indent=2))
    except Exception:
        record_failure("stage36_locked_predictive_state_evaluation")
'''


packaging = STAGE35.packaging
for old, new in [
    ("Stage 35", "Stage 36"), ("stage35", "stage36"), ("hpcc", "pscd"),
]:
    packaging = packaging.replace(old, new)


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
    raise RuntimeError("Stage 36 protocol digest placeholder was not replaced")

cells = [
    markdown(introduction), code(configuration), code(installation), code(setup),
    code(analysis_helpers), code(model_helpers), code(design_and_runtime_helpers),
    code(physical_truth), code(construction_and_paths), code(model_selection),
    code(calibration), code(locked_evaluation), code(packaging),
]
for index, cell in enumerate(cells):
    cell["id"] = f"stage36-{index:02d}"

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
