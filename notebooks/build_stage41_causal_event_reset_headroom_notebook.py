"""Build the prospective Stage 41 causal event/reset headroom Colab."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
TARGET = ROOT / "41_causal_event_reset_headroom.ipynb"
NUMERICAL = REPOSITORY / "src/cf_faithfulness/stage41_causal_event_reset.py"


def load_builder(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BASE = load_builder(
    ROOT / "build_stage40_contact_tail_risk_distillation_notebook.py",
    "stage40_builder_for_stage41",
)

code = BASE.code
markdown = BASE.markdown
replace_assignment = BASE.replace_assignment
replace_block = BASE.replace_block
assigned_uppercase_names = BASE.assigned_uppercase_names
function_sources = BASE.function_sources


introduction = r'''# Stage 41: causal event/reset headroom audit

## Frozen development-only decision before computation

Stage 40 rejected contact-risk reweighting: it did not repair the recursive
contact tail in either JEPA or DINO.  Stage 41 asks the narrower causal
question that must be answered before building another learned architecture:
**would exact hybrid-event information repair the frozen recursive predictor
at all?**

For every registered initial state and action word, the simulator executes an
ordinary branch and an exact matched intervention in which agent--block
collision is disabled.  Ordinary-minus-ghost grounded state is the finite
physical contact effect.  A post-solve callback records event occurrence,
within-macro first-event time, contact normal, and normal/tangential impulse
coordinates.  The frozen recursive PSCD model never sees the ghost outcome.

Equal-width ridge correction heads form a prespecified oracle ladder:

1. smooth outcome-independent sham features;
2. shuffled event metadata;
3. exact event occurrence;
4. event occurrence plus first-event time;
5. time plus contact normal; and
6. the full event/time/normal/impulse reset ceiling.

Every rung has the same design width.  Missing oracle coordinates are replaced
by a frozen random smooth projection of pre-outcome predictor features.  Ridge
penalties are selected on model-selection trajectories, then heads are refit
on construction plus calibration trajectories.  Evaluation trajectories and
their paired intervention outcomes remain unopened until all models, heads,
normalizers, penalties, and hashes are frozen.

The full oracle ceiling passes only if every seed in both JEPA and DINO has at
least 25% improvement in post-contact-to-contact upper-tail error, at least 10%
overall p95 improvement, no more than 2% mean degradation, dominance over both
matched controls, leave-one-family-out robustness, and causal-effect alignment
better than the zero-effect baseline.  Planning remains sealed.

This is a headroom and identifiability audit, not a learned causal model.  Even
a positive result authorizes only a subsequent label-free event-state test;
it does not authorize causal, planning, or deployment claims.
'''


configuration = BASE.configuration
for name, value in {
    "PROTOCOL_ID": '"stage41-causal-event-reset-headroom-v1"',
    "NOTEBOOK_PROTOCOL_SHA256": '"__PROTOCOL_DIGEST__"',
    "EVIDENCE_STATUS": '"FRESH_DEVELOPMENT_ONLY_CAUSAL_HEADROOM_AUDIT"',
    "EXPERIMENT_NOTEBOOK_PATH": '"notebooks/41_causal_event_reset_headroom.ipynb"',
    "EXPERIMENT_BUILDER_PATH": '"notebooks/build_stage41_causal_event_reset_headroom_notebook.py"',
    "EXPERIMENT_NUMERICAL_PATH": '"src/cf_faithfulness/stage41_causal_event_reset.py"',
    "OUTPUT_DIR": '"/content/counterfactual_faithfulness_stage41_cerh"',
    "DRIVE_OUTPUT_DIR": '"/content/drive/MyDrive/counterfactual_faithfulness_stage41_cerh"',
    "RUN_REQUEST_PATH": '"/content/drive/MyDrive/counterfactual_faithfulness_stage41_cerh/stage41_run_request.json"',
    "SEED": "410101",
    "DESIGN_SEED": "410141",
    "DECODER_SEED": "410183",
    "RANK_SEED": "410213",
    "CALIBRATION_SEED": "410253",
    "BOOTSTRAP_SEED": "410283",
    "CONTROL_SEED": "410351",
    "CONSTRUCTION_TRAJECTORY_POOL": "list(range(114000, 116000))",
    "MODEL_SELECTION_TRAJECTORY_POOL": "list(range(116000, 118000))",
    "CALIBRATION_TRAJECTORY_POOL": "list(range(118000, 120000))",
    "EVALUATION_TRAJECTORY_POOL": "list(range(120000, 126000))",
    "TASK_ID_OFFSET": "410000",
}.items():
    configuration = replace_assignment(configuration, name, value)
configuration = replace_assignment(
    configuration, "FINAL_TRAINING_SEEDS",
    '[4101, 4102, 4103] if RUN_MODE == "pilot" else [4101]',
)
configuration = replace_assignment(
    configuration, "PRIMARY_VARIANTS", '["uniform_grounded"]',
)
configuration = replace_assignment(
    configuration, "TAIL_QUALIFICATION_VARIANTS", '["uniform_grounded"]',
)
configuration = replace_block(
    configuration,
    "CANONICAL_RESPONSE_WORD_NAMES = [",
    "CALIBRATION_INTERCHANGE_PAIRS =",
    r'''CANONICAL_RESPONSE_WORD_NAMES = ["A", "B", "C", "D", "AB", "CD", "BA", "DC"]
CONSTRUCTION_WORD_NAMES = [
    "CAAABBBAC", "DABCBADAD", "DCACDABACC", "DCBDDCCDBA",
    "ABDCBDABACC", "BBBBBBCCAAC", "BCDACCDCCADB", "ACCDADAAADBB",
]
MODEL_SELECTION_WORD_NAMES = [
    "BABCDACCB", "DCDCDDDDC", "DDADCCDCCB", "ABDCBCABCA",
    "ACDCDCDDDBD", "ADABCDBABBD", "BCCABBDDDDDD", "BCCCBBDBCDAB",
]
CALIBRATION_WORD_NAMES = [
    "AACABBBCB", "BBDDACDAA", "DCCBBABAAC", "AADCCDCDCD",
    "ADCBDADABAD", "DBCCAABBCAB", "CBCDBBCCDBAD", "DCBACDACACAC",
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
    "BDCDBBBDD", "DCBBBCDAD", "ACCCCDACAB", "DCDAADADBA",
    "DDAACBCDDDB", "BDBCDCBBDAC", "CADCAADCDACB", "CADABCDCCACC",
]
PLANNING_WORD_NAMES = []
EVALUATION_WORD_NAMES_REGISTERED = list(CLOSURE_EVALUATION_WORD_NAMES)
EVALUATION_WORD_SPECS = [
    stage39_word_spec(name) for name in EVALUATION_WORD_NAMES_REGISTERED
]
''',
)
configuration = configuration.replace(
    '"fresh_trajectory_ids_102000_to_113999",',
    '"fresh_trajectory_ids_114000_to_125999",',
)
configuration = re.sub(r"^PROTOCOL_CONFIG_KEYS = \[.*\]\n?", "", configuration, flags=re.M)
configuration += r'''

PINNED_LATENT_OUTER_WEIGHTS = {"jepa": 2.0, "dino": 0.5}
CAUSAL_VARIANTS = [
    "smooth_matched", "shuffled_event", "oracle_event", "oracle_time",
    "oracle_geometry", "oracle_reset_ceiling",
]
RIDGE_PENALTIES = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
MIN_CONTACT_TAIL_RELATIVE_IMPROVEMENT = 0.25
MIN_P95_RELATIVE_IMPROVEMENT = 0.10
MAX_MEAN_RELATIVE_DEGRADATION = 0.02
MIN_LOO_CONTACT_TAIL_RELATIVE_IMPROVEMENT = 0.10
MIN_CAUSAL_EFFECT_GAIN = 0.10
MIN_CAUSAL_EFFECT_COSINE = 0.25
CONTACT_TAIL_MASS = 0.25
MIN_REENTRY_ROWS = 8
assert CAUSAL_VARIANTS[-1] == "oracle_reset_ceiling"
assert MIN_CONTACT_TAIL_RELATIVE_IMPROVEMENT == 0.25
assert MIN_P95_RELATIVE_IMPROVEMENT == 0.10
assert MAX_MEAN_RELATIVE_DEGRADATION == 0.02
assert PLANNING_WORD_NAMES == []
'''
configuration += "\n\nPROTOCOL_CONFIG_KEYS = " + repr(
    assigned_uppercase_names(configuration)
) + "\n"


installation = BASE.installation
setup = BASE.setup.replace("stage40_ctrd", "stage41_cerh")
analysis_helpers = BASE.analysis_helpers + "\n\n" + function_sources(
    NUMERICAL.read_text(),
    [
        "_as_matrix", "mean_scale", "fixed_sham_projection",
        "deterministic_permutation", "causal_design_matrix", "fit_ridge",
        "ridge_predict", "select_ridge_penalty", "upper_tail_mean",
        "causal_effect_metrics", "Stage41PanelDecision",
        "derive_stage41_panel_decision", "derive_stage41_decision",
    ],
)
analysis_helpers = analysis_helpers.replace(
    "class Stage41PanelDecision:\n", "@dataclass(frozen=True)\nclass Stage41PanelDecision:\n"
)
model_helpers = BASE.model_helpers
design_and_runtime_helpers = BASE.design_and_runtime_helpers
physical_truth = BASE.physical_truth
simulator_preflight = BASE.simulator_preflight
construction_and_paths = BASE.construction_and_paths
data_and_selection = BASE.data_and_selection
data_and_selection = data_and_selection.replace(
    "# Freeze prior-developed latent strength and define split-safe loaders.",
    "# Freeze the prior-developed latent strength and split-safe loaders.",
)
data_and_selection = data_and_selection.replace("stage40", "stage41")
data_and_selection = data_and_selection.replace("Stage 40", "Stage 41")


causal_interventions = r'''# Materialize paired interventions for development splits only.


def stage41_pair_path(record):
    return CAUSAL_DIR / f"paired_{int(record['record_id'])}.npz"


def install_stage41_contact_recorder(environment, action_clock):
    events = []
    original = environment._handle_collision

    def post_solve(arbiter, space, data):
        original(arbiter, space, data)
        shapes = list(arbiter.shapes)
        bodies = [shape.body for shape in shapes]
        if environment.agent not in bodies or environment.block not in bodies:
            return
        block_index = bodies.index(environment.block)
        impulse = np.asarray(arbiter.total_impulse, dtype=np.float64)
        if block_index != 0:
            impulse = -impulse
        points = arbiter.contact_point_set
        normal = np.asarray(points.normal, dtype=np.float64)
        if block_index == 0:
            normal = -normal
        distances = [float(point.distance) for point in points.points]
        events.append({
            "action_step": int(action_clock["step"]),
            "impulse": impulse,
            "normal": normal,
            "distances": distances,
        })

    environment.collision_handeler.post_solve = post_solve
    return events


def disable_stage41_agent_block_collision(environment):
    import pymunk

    agent_category, block_category = 1 << 0, 1 << 1
    all_masks = int(pymunk.ShapeFilter.ALL_MASKS())
    for shape in environment.agent.shapes:
        shape.filter = pymunk.ShapeFilter(
            categories=agent_category, mask=all_masks ^ block_category,
        )
    for shape in environment.block.shapes:
        shape.filter = pymunk.ShapeFilter(
            categories=block_category, mask=all_masks ^ agent_category,
        )


def aggregate_stage41_macro_events(events, macro_start):
    if not events:
        return np.zeros(6, dtype=np.float64)
    impulse = np.sum([row["impulse"] for row in events], axis=0)
    weights = np.asarray([
        max(float(np.linalg.norm(row["impulse"])), 1e-12) for row in events
    ])
    normal = np.average(
        np.stack([row["normal"] for row in events]), axis=0, weights=weights
    )
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-12:
        normal, normal_norm = impulse.copy(), float(np.linalg.norm(impulse))
    if normal_norm > 1e-12:
        normal = normal / normal_norm
    if float(np.dot(impulse, normal)) < 0:
        normal = -normal
    tangent = np.asarray([-normal[1], normal[0]], dtype=np.float64)
    normal_impulse = max(float(np.dot(impulse, normal)), 0.0)
    tangent_impulse = float(np.dot(impulse, tangent))
    first_step = min(int(row["action_step"]) for row in events)
    tau = np.clip((first_step - int(macro_start) + 0.5) / FRAMESKIP, 0.0, 1.0)
    return np.asarray([
        1.0, tau, normal[0], normal[1],
        np.log1p(normal_impulse), np.arcsinh(tangent_impulse),
    ], dtype=np.float64)


def rollout_stage41_branch(record, specification, ghost=False):
    environment = reset_dynamic_environment(
        record["state"], record["goal"], record["evaluation_seed"]
    )
    action_clock = {"step": -1}
    events = install_stage41_contact_recorder(environment, action_clock)
    if ghost:
        disable_stage41_agent_block_collision(environment)
    actions, _ = word_actions(record, specification)
    path_states, metadata = [], []
    cursor = 0
    try:
        for action_step, action in enumerate(actions):
            action_clock["step"] = int(action_step)
            environment.step(action)
            if (action_step + 1) % FRAMESKIP == 0:
                path_states.append(dynamic_state_from_environment(environment))
                chunk = events[cursor:]
                metadata.append(aggregate_stage41_macro_events(
                    chunk, action_step + 1 - FRAMESKIP
                ))
                cursor = len(events)
    finally:
        environment.close()
    return {
        "states": np.asarray(path_states, dtype=np.float64),
        "metadata": np.asarray(metadata, dtype=np.float64),
        "event_count": int(len(events)),
    }


def generate_stage41_paired_record(record, protocol_split):
    path = stage41_pair_path(record)
    identity = (
        f"{PROTOCOL_ID}:{RUN_SIGNATURE}:{record['record_id']}:{protocol_split}:"
        "ordinary-vs-agent-block-collision-disabled-v1"
    )
    required = {
        "identity", "word_names", "word_lengths", "path_mask", "normal_observables",
        "ghost_observables", "physical_effects", "causal_metadata", "event_counts",
    }
    names = stage39_names_for_split(protocol_split)
    if validate_npz_shard(path, required, identity):
        with np.load(path, allow_pickle=False) as cached:
            if [str(value) for value in cached["word_names"]] == names:
                PROVENANCE_COUNTS["validated_cache_hits"] += 1
                return path
    normal = np.zeros(
        (len(names), MAX_WORD_LENGTH, len(GROUNDED_OBSERVABLES)), dtype=np.float64
    )
    ghost = np.zeros_like(normal)
    metadata = np.zeros((len(names), MAX_WORD_LENGTH, 6), dtype=np.float64)
    mask = np.zeros((len(names), MAX_WORD_LENGTH), dtype=bool)
    event_counts = np.zeros(len(names), dtype=np.int64)
    with np.load(truth_path(record), allow_pickle=False) as truth:
        truth_lookup = {str(name): index for index, name in enumerate(truth["word_names"])}
        for word_index, name in enumerate(names):
            specification = WORD_BY_NAME[name]
            length = int(specification["length"])
            ordinary = rollout_stage41_branch(record, specification, ghost=False)
            intervention = rollout_stage41_branch(record, specification, ghost=True)
            if intervention["event_count"] != 0:
                raise RuntimeError("ghost collision callback fired")
            ordinary_observables = np.stack([
                grounded_observables(value) for value in ordinary["states"]
            ])
            ghost_observables = np.stack([
                grounded_observables(value) for value in intervention["states"]
            ])
            expected = truth["path_observables"][truth_lookup[name], :length]
            if not np.allclose(ordinary_observables, expected, atol=1e-9, rtol=0):
                raise RuntimeError("instrumented ordinary rollout drifted from frozen truth")
            normal[word_index, :length] = ordinary_observables
            ghost[word_index, :length] = ghost_observables
            metadata[word_index, :length] = ordinary["metadata"]
            mask[word_index, :length] = True
            event_counts[word_index] = ordinary["event_count"]
    atomic_npz(
        path, identity=np.asarray(identity), word_names=np.asarray(names),
        word_lengths=np.asarray([len(name) for name in names], dtype=np.int64),
        path_mask=mask, normal_observables=normal, ghost_observables=ghost,
        physical_effects=normal - ghost, causal_metadata=metadata,
        event_counts=event_counts,
    )
    return path


def load_stage41_pairs(data, protocol_split):
    metadata = np.zeros((len(data["word"]), MAX_WORD_LENGTH, 6), dtype=np.float64)
    normal = np.zeros_like(data["simulator"], dtype=np.float64)
    ghost = np.zeros_like(data["simulator"], dtype=np.float64)
    cache = {}
    for row_index, (record_id, word) in enumerate(zip(data["record_id"], data["word"])):
        record_id = int(record_id)
        if record_id not in cache:
            record = next(row for row in ALL_RECORDS if int(row["record_id"]) == record_id)
            path = stage41_pair_path(record)
            with np.load(path, allow_pickle=False) as payload:
                cache[record_id] = {key: payload[key].copy() for key in payload.files}
        payload = cache[record_id]
        lookup = {str(name): index for index, name in enumerate(payload["word_names"])}
        pair_index = lookup[str(word)]
        metadata[row_index] = payload["causal_metadata"][pair_index]
        normal[row_index] = payload["normal_observables"][pair_index]
        ghost[row_index] = payload["ghost_observables"][pair_index]
    if not np.allclose(normal[data["mask"]], data["simulator"][data["mask"]], atol=1e-9, rtol=0):
        raise RuntimeError(f"paired ordinary truth mismatch on {protocol_split}")
    return {
        "metadata": metadata, "normal": normal, "ghost": ghost,
        "physical_effect": normal - ghost,
    }


if not PIPELINE_FAILED and SIMULATOR_PREFLIGHT_PASSED:
    try:
        verify_executed_notebook_through(
            "# Materialize paired interventions for development splits only."
        )
        for split in ["construction", "model_selection", "calibration"]:
            for index, record in enumerate(SELECTED_RECORDS[split]):
                generate_stage41_paired_record(record, split)
                write_json(OUT / f"paired_{split}_progress.json", {
                    "completed": index + 1, "total": len(SELECTED_RECORDS[split]),
                    "last_record_id": int(record["record_id"]),
                    "evaluation_pairs_opened": False,
                })
        atomic_checkpoint("stage41_development_pairs_complete", {
            "evaluation_pairs_opened": False,
            "intervention": "agent_block_collision_disabled",
        })
    except Exception:
        record_failure("stage41_development_paired_interventions")
'''


calibration = r'''# Freeze the recursive baselines and equal-width causal ladder before heldout access.
FROZEN_MODELS = {}
PHYSICAL_SCALES = {}
HEADS = {}
HEAD_SELECTION_ROWS = []
EVALUATION_OPENED = False


def stage41_artifact_paths(short, variant, seed):
    stem = CALIBRATION_MODEL_DIR / f"stage41_{short}_{variant}_seed{int(seed)}"
    return Path(str(stem) + ".npz"), Path(str(stem) + "_schema.json")


def stage41_encode_artifact(value, arrays, prefix="root"):
    if isinstance(value, np.ndarray):
        key = f"array_{len(arrays):05d}"
        arrays[key] = value
        return {"kind": "array", "key": key}
    if isinstance(value, dict):
        return {"kind": "dict", "items": {
            str(key): stage41_encode_artifact(item, arrays, f"{prefix}.{key}")
            for key, item in sorted(value.items())
        }}
    if isinstance(value, (list, tuple)):
        return {"kind": "list", "items": [
            stage41_encode_artifact(item, arrays, f"{prefix}.{index}")
            for index, item in enumerate(value)
        ]}
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return {"kind": "scalar", "value": value}
    raise TypeError(f"unsupported Stage 41 artifact at {prefix}: {type(value)}")


def stage41_decode_artifact(node, arrays):
    if node["kind"] == "array":
        return np.asarray(arrays[str(node["key"])])
    if node["kind"] == "dict":
        return {key: stage41_decode_artifact(value, arrays) for key, value in node["items"].items()}
    if node["kind"] == "list":
        return [stage41_decode_artifact(value, arrays) for value in node["items"]]
    if node["kind"] == "scalar":
        return node["value"]
    raise ValueError(f"unknown Stage 41 artifact node {node['kind']!r}")


def save_stage41_artifact(short, variant, seed, artifact):
    array_path, schema_path = stage41_artifact_paths(short, variant, seed)
    arrays = {}
    schema = stage41_encode_artifact(artifact, arrays)
    atomic_npz(array_path, **arrays)
    write_json(schema_path, {
        "protocol_id": PROTOCOL_ID, "run_signature": RUN_SIGNATURE,
        "model": str(short), "variant": str(variant), "seed": int(seed),
        "array_sha256": sha256_file(array_path), "schema": schema,
    })
    write_digest_sidecar(schema_path)


def load_stage41_artifact(short, variant, seed):
    array_path, schema_path = stage41_artifact_paths(short, variant, seed)
    validate_digest_sidecar(array_path)
    validate_digest_sidecar(schema_path)
    metadata = json.loads(schema_path.read_text())
    expected = (PROTOCOL_ID, RUN_SIGNATURE, str(short), str(variant), int(seed))
    observed = (
        metadata["protocol_id"], metadata["run_signature"], metadata["model"],
        metadata["variant"], int(metadata["seed"]),
    )
    if observed != expected or metadata["array_sha256"] != sha256_file(array_path):
        raise RuntimeError(f"Stage 41 artifact binding failed for {short}/{variant}/{seed}")
    with np.load(array_path, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    return stage41_decode_artifact(metadata["schema"], arrays)


def stage41_base_objective(short):
    return {
        "free_weight": 1.0,
        "semigroup_weight": COEFFICIENT_MATCH_FACTOR * float(
            PINNED_LATENT_OUTER_WEIGHTS[short]
        ),
        "semigroup_component_weights": OVERSHOOT_COMPONENT_WEIGHTS,
    }


def fit_or_load_stage41_base(short, seed, data):
    variant = "uniform_grounded"
    array_path, schema_path = stage41_artifact_paths(short, variant, seed)
    sidecars = [Path(str(array_path) + ".sha256"), Path(str(schema_path) + ".sha256")]
    if array_path.is_file() and schema_path.is_file() and all(path.is_file() for path in sidecars):
        PROVENANCE_COUNTS["validated_cache_hits"] += 1
        return load_stage41_artifact(short, variant, seed)
    objective = stage41_base_objective(short)
    risk = contact_transition_weights(
        data["initial_mode"], data["target_mode"], data["mask"], contact_multiplier=1.0
    )
    artifact = fit_contact_risk_predictive_state_closure(
        data["initial_carrier"], data["actions"], data["carrier"],
        data["simulator"], data["mask"], risk,
        history_length=FIXED_HISTORY_LENGTH, latent_dim=FIXED_LATENT_DIM,
        dynamics=FIXED_DYNAMICS, epochs=ACTIVE_FINAL_EPOCHS,
        learning_rate=PSCD_LEARNING_RATE, seed=int(seed),
        semigroup_horizons=SEMIGROUP_HORIZONS,
        free_weight=objective["free_weight"],
        semigroup_weight=objective["semigroup_weight"],
        semigroup_component_weights=objective["semigroup_component_weights"],
    )
    artifact["config"]["physical_target"] = "simulator_ground_truth"
    artifact["config"]["stage41_role"] = "frozen_uniform_recursive_baseline"
    save_stage41_artifact(short, variant, seed, artifact)
    return artifact


def stage41_flat_bundle(short, seed, split):
    data = load_stage39_sequences(short, split)
    rollout = rollout_predictive_state_closure(
        FROZEN_MODELS[short][int(seed)], data["initial_carrier"], data["actions"],
        data["carrier"], data["mask"],
    )
    pairs = load_stage41_pairs(data, split)
    valid = np.asarray(rollout["evaluation_mask"], dtype=bool)
    initial = np.repeat(
        data["initial_physical"][:, None, :], data["actions"].shape[1], axis=1
    )
    base_tensor = np.concatenate([
        np.asarray(rollout["physical"], dtype=np.float64),
        np.asarray(rollout["state"], dtype=np.float64),
        np.asarray(data["actions"], dtype=np.float64), initial,
    ], axis=-1)
    groups = np.repeat(data["group"][:, None], data["actions"].shape[1], axis=1)
    return {
        "base": base_tensor[valid],
        "target": (
            (data["simulator"] - rollout["physical"]) / PHYSICAL_SCALES[short]
        )[valid],
        "metadata": pairs["metadata"][valid],
        "physical_effect": pairs["physical_effect"][valid],
        "groups": groups[valid].astype(np.int64),
    }


def concatenate_stage41_flat(*bundles):
    return {
        key: np.concatenate([bundle[key] for bundle in bundles], axis=0)
        for key in bundles[0]
    }


def stage41_design(bundle, artifact, variant, seed):
    permutation = None
    if variant == "shuffled_event":
        permutation = deterministic_permutation(
            len(bundle["base"]), stable_seed(seed, "stage41_shuffle", len(bundle["base"]))
        )
    return causal_design_matrix(
        bundle["base"], bundle["metadata"], variant=variant,
        base_mean=artifact["base_mean"], base_scale=artifact["base_scale"],
        metadata_mean=artifact["metadata_mean"], metadata_scale=artifact["metadata_scale"],
        sham_projection={
            "weight": artifact["sham_weight"], "bias": artifact["sham_bias"]
        },
        permutation=permutation,
    )


def fit_stage41_head(short, seed, variant, construction, validation, final_fit):
    # Every rung receives the identical frozen smooth basis within a
    # model/seed panel; only the registered oracle-coordinate prefix changes.
    selection_seed = stable_seed(
        CALIBRATION_SEED, short, int(seed), "shared_head_design"
    )
    base_mean, base_scale = mean_scale(construction["base"])
    metadata_mean, metadata_scale = mean_scale(construction["metadata"])
    projection = fixed_sham_projection(construction["base"].shape[1], 6, selection_seed)
    selection_artifact = {
        "base_mean": base_mean, "base_scale": base_scale,
        "metadata_mean": metadata_mean, "metadata_scale": metadata_scale,
        "sham_weight": projection["weight"], "sham_bias": projection["bias"],
    }
    train_design = stage41_design(construction, selection_artifact, variant, selection_seed)
    validation_design = stage41_design(validation, selection_artifact, variant, selection_seed + 1)
    selection = select_ridge_penalty(
        train_design, construction["target"], validation_design,
        validation["target"], RIDGE_PENALTIES,
    )
    final_base_mean, final_base_scale = mean_scale(final_fit["base"])
    final_metadata_mean, final_metadata_scale = mean_scale(final_fit["metadata"])
    final_projection = fixed_sham_projection(
        final_fit["base"].shape[1], 6, selection_seed
    )
    artifact = {
        "variant": variant, "seed": int(seed), "model": short,
        "base_mean": final_base_mean, "base_scale": final_base_scale,
        "metadata_mean": final_metadata_mean, "metadata_scale": final_metadata_scale,
        "sham_weight": final_projection["weight"], "sham_bias": final_projection["bias"],
        "physical_scale": PHYSICAL_SCALES[short],
        "selected_penalty": float(selection["selected_penalty"]),
        "selection_rows": selection["candidate_rows"],
        "nominal_design_width": int(final_fit["base"].shape[1] + 6),
        "evaluation_rows_used": 0,
    }
    final_design = stage41_design(final_fit, artifact, variant, selection_seed)
    ridge = fit_ridge(final_design, final_fit["target"], artifact["selected_penalty"])
    artifact["weight"] = ridge["weight"]
    artifact["intercept"] = ridge["intercept"]
    return artifact


if not PIPELINE_FAILED and SIMULATOR_PREFLIGHT_PASSED:
    try:
        verify_executed_notebook_through(
            "# Freeze the recursive baselines and equal-width causal ladder before heldout access."
        )
        model_manifest = []
        for short in ["jepa", "dino"]:
            construction = load_stage39_sequences(short, "construction")
            calibration_only = load_stage39_sequences(short, "calibration")
            base_fit = concatenate_stage39_sequences(construction, calibration_only)
            valid = base_fit["mask"]
            PHYSICAL_SCALES[short] = np.maximum(
                np.std(base_fit["simulator"][valid], axis=0, ddof=1), 1e-8
            )
            FROZEN_MODELS[short], HEADS[short] = {}, {}
            for seed in FINAL_TRAINING_SEEDS:
                base = fit_or_load_stage41_base(short, int(seed), base_fit)
                FROZEN_MODELS[short][int(seed)] = base
                construction_flat = stage41_flat_bundle(short, seed, "construction")
                validation_flat = stage41_flat_bundle(short, seed, "model_selection")
                calibration_flat = stage41_flat_bundle(short, seed, "calibration")
                final_flat = concatenate_stage41_flat(construction_flat, calibration_flat)
                HEADS[short][int(seed)] = {}
                for variant in CAUSAL_VARIANTS:
                    array_path, schema_path = stage41_artifact_paths(
                        short, f"head_{variant}", seed
                    )
                    if all(path.is_file() for path in [
                        array_path, schema_path, Path(str(array_path) + ".sha256"),
                        Path(str(schema_path) + ".sha256"),
                    ]):
                        head = load_stage41_artifact(short, f"head_{variant}", seed)
                    else:
                        head = fit_stage41_head(
                            short, int(seed), variant, construction_flat,
                            validation_flat, final_flat,
                        )
                        save_stage41_artifact(short, f"head_{variant}", seed, head)
                    if head["nominal_design_width"] != final_flat["base"].shape[1] + 6:
                        raise RuntimeError("Stage 41 causal head width changed")
                    HEADS[short][int(seed)][variant] = head
                    HEAD_SELECTION_ROWS.extend({
                        "model": short, "seed": int(seed), "variant": variant,
                        **row, "selected": bool(
                            float(row["penalty"]) == float(head["selected_penalty"])
                        ),
                    } for row in head["selection_rows"])
                    model_manifest.append({
                        "model": short, "seed": int(seed), "variant": variant,
                        "array_sha256": sha256_file(array_path),
                        "schema_sha256": sha256_file(schema_path),
                    })
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        write_csv(EVIDENCE_DIR / "stage41_head_selection_rows.csv", HEAD_SELECTION_ROWS)
        scale_path = CALIBRATION_MODEL_DIR / "stage41_frozen_physical_scales.npz"
        atomic_npz(scale_path, **{
            f"physical_{short}": value for short, value in PHYSICAL_SCALES.items()
        })
        certificate_path = CALIBRATION_MODEL_DIR / "evaluation_open_certificate.json"
        write_json(certificate_path, {
            "protocol_id": PROTOCOL_ID, "run_signature": RUN_SIGNATURE,
            "source_commit": SOURCE_IDENTITY.get("resolved_commit"),
            "models_and_heads": model_manifest,
            "scale_sha256": sha256_file(scale_path),
            "final_training_seeds": list(map(int, FINAL_TRAINING_SEEDS)),
            "causal_variants": CAUSAL_VARIANTS,
            "equal_nominal_head_width": True,
            "evaluation_statistics_read": False,
            "evaluation_pairs_generated": False,
            "planning_permanently_sealed": True,
            "checkpoint_parameters_updated": False,
        })
        write_digest_sidecar(certificate_path)
        atomic_checkpoint("stage41_models_and_heads_frozen", {
            "certificate_sha256": sha256_file(certificate_path),
            "evaluation_opened": False, "planning_opened": False,
        })
    except Exception:
        record_failure("stage41_model_and_head_freeze")
'''


heldout_evaluation = r'''# Open the development-heldout causal panel after every head is frozen.
DECISION_PAYLOAD = {
    "status": "INCONCLUSIVE_PIPELINE_FAILURE", "passed": False,
    "planning_opened": False, "causal_claim_authorized": False,
}
EVALUATION_ROWS = []
SUMMARY = {}
PANEL_DECISIONS = {}


def stage41_terminal(values, mask):
    array = np.asarray(values)
    valid = np.asarray(mask, dtype=bool)
    index = np.max(np.where(valid, np.arange(valid.shape[1])[None, :], -1), axis=1)
    if np.any(index < 0):
        raise ValueError("every Stage 41 path needs an endpoint")
    return array[np.arange(len(array)), index]


def stage41_prediction(short, seed, data, rollout, pairs, variant):
    valid = np.asarray(rollout["evaluation_mask"], dtype=bool)
    initial = np.repeat(
        data["initial_physical"][:, None, :], data["actions"].shape[1], axis=1
    )
    base_tensor = np.concatenate([
        rollout["physical"], rollout["state"], data["actions"], initial,
    ], axis=-1)
    head = HEADS[short][int(seed)][variant]
    flat = {
        "base": base_tensor[valid], "metadata": pairs["metadata"][valid],
    }
    design = stage41_design(
        flat, head, variant,
        stable_seed(CONTROL_SEED, "heldout_design", short, int(seed), variant),
    )
    correction = ridge_predict(head, design) * PHYSICAL_SCALES[short]
    prediction = np.asarray(rollout["physical"], dtype=np.float64).copy()
    prediction[valid] += correction
    return prediction, base_tensor, valid


def stage41_leave_one_family_tail_gain(repair, baseline, reentry, groups):
    values = []
    for family in sorted(set(np.asarray(groups, dtype=np.int64).tolist())):
        selected = np.asarray(reentry, dtype=bool) & (np.asarray(groups) != family)
        if np.sum(selected) < MIN_REENTRY_ROWS:
            continue
        reference = upper_tail_mean(np.asarray(baseline)[selected], CONTACT_TAIL_MASS)
        candidate = upper_tail_mean(np.asarray(repair)[selected], CONTACT_TAIL_MASS)
        values.append((reference - candidate) / max(reference, 1e-12))
    return float(min(values)) if values else float("-inf")


def stage41_panel(short, data, pairs):
    groups = data["group"]
    source_terminal = stage41_terminal(data["source_mode"], data["mask"]).astype(str)
    target_terminal = stage41_terminal(data["target_mode"], data["mask"]).astype(str)
    reentry = (source_terminal == "post_contact") & (target_terminal == "contact")
    if int(np.sum(reentry)) < MIN_REENTRY_ROWS:
        raise RuntimeError("heldout Stage 41 panel has too few post-contact re-entry rows")
    all_errors = {variant: [] for variant in ["baseline"] + CAUSAL_VARIANTS}
    seed_summaries = []
    gate_rows = []
    causal_alignment_rows = []
    for seed in FINAL_TRAINING_SEEDS:
        rollout = rollout_predictive_state_closure(
            FROZEN_MODELS[short][int(seed)], data["initial_carrier"], data["actions"],
            data["carrier"], data["mask"],
        )
        baseline_error = scaled_path_mse(
            rollout["physical"], data["simulator"], rollout["evaluation_mask"],
            PHYSICAL_SCALES[short],
        )
        all_errors["baseline"].append(baseline_error)
        predictions = {}
        base_tensor = valid = None
        for variant in CAUSAL_VARIANTS:
            prediction, base_tensor, valid = stage41_prediction(
                short, seed, data, rollout, pairs, variant
            )
            predictions[variant] = prediction
            all_errors[variant].append(scaled_path_mse(
                prediction, data["simulator"], rollout["evaluation_mask"],
                PHYSICAL_SCALES[short],
            ))
        reset_error = all_errors["oracle_reset_ceiling"][-1]
        smooth_error = all_errors["smooth_matched"][-1]
        shuffled_error = all_errors["shuffled_event"][-1]
        baseline_tail = upper_tail_mean(baseline_error[reentry], CONTACT_TAIL_MASS)
        reset_tail = upper_tail_mean(reset_error[reentry], CONTACT_TAIL_MASS)
        smooth_tail = upper_tail_mean(smooth_error[reentry], CONTACT_TAIL_MASS)
        shuffled_tail = upper_tail_mean(shuffled_error[reentry], CONTACT_TAIL_MASS)
        tail_gain = (baseline_tail - reset_tail) / max(baseline_tail, 1e-12)
        baseline_p95 = float(np.quantile(baseline_error, 0.95))
        reset_p95 = float(np.quantile(reset_error, 0.95))
        p95_gain = (baseline_p95 - reset_p95) / max(baseline_p95, 1e-12)
        mean_ratio = float(np.mean(reset_error) / max(np.mean(baseline_error), 1e-12))
        loo_gain = stage41_leave_one_family_tail_gain(
            reset_error, baseline_error, reentry, groups
        )
        control_dominance = bool(
            reset_tail < min(smooth_tail, shuffled_tail)
            and reset_p95 < min(
                float(np.quantile(smooth_error, 0.95)),
                float(np.quantile(shuffled_error, 0.95)),
            )
        )

        head = HEADS[short][int(seed)]["oracle_reset_ceiling"]
        flat_base = base_tensor[valid]
        actual_meta = pairs["metadata"][valid]
        actual_bundle = {"base": flat_base, "metadata": actual_meta}
        zero_bundle = {"base": flat_base, "metadata": np.zeros_like(actual_meta)}
        actual_design = stage41_design(
            actual_bundle, head, "oracle_reset_ceiling",
            stable_seed(CONTROL_SEED, "effect_actual", short, int(seed)),
        )
        zero_design = stage41_design(
            zero_bundle, head, "oracle_reset_ceiling",
            stable_seed(CONTROL_SEED, "effect_zero", short, int(seed)),
        )
        predicted_effect = (
            ridge_predict(head, actual_design) - ridge_predict(head, zero_design)
        ) * PHYSICAL_SCALES[short]
        alignment = causal_effect_metrics(
            predicted_effect, pairs["physical_effect"][valid], PHYSICAL_SCALES[short],
            actual_meta[:, 0] > 0.5,
        )
        causal_alignment_rows.append(alignment)
        gates = {
            "tail_improvement": bool(
                tail_gain >= MIN_CONTACT_TAIL_RELATIVE_IMPROVEMENT
                and loo_gain >= MIN_LOO_CONTACT_TAIL_RELATIVE_IMPROVEMENT
            ),
            "p95_improvement": bool(p95_gain >= MIN_P95_RELATIVE_IMPROVEMENT),
            "mean_noninferiority": bool(mean_ratio <= 1 + MAX_MEAN_RELATIVE_DEGRADATION),
            "control_dominance": control_dominance,
            "causal_alignment": bool(
                alignment["relative_gain_over_zero"] >= MIN_CAUSAL_EFFECT_GAIN
                and alignment["mean_cosine"] >= MIN_CAUSAL_EFFECT_COSINE
            ),
        }
        gate_rows.append(gates)
        variant_metrics = {}
        for variant in CAUSAL_VARIANTS:
            errors = all_errors[variant][-1]
            variant_metrics[variant] = {
                "mean_nmse": float(np.mean(errors)),
                "p95_nmse": float(np.quantile(errors, 0.95)),
                "reentry_tail_nmse": upper_tail_mean(errors[reentry], CONTACT_TAIL_MASS),
            }
        seed_summaries.append({
            "seed": int(seed), "reentry_rows": int(np.sum(reentry)),
            "baseline_mean_nmse": float(np.mean(baseline_error)),
            "baseline_p95_nmse": baseline_p95,
            "baseline_reentry_tail_nmse": baseline_tail,
            "oracle_tail_relative_improvement": float(tail_gain),
            "oracle_p95_relative_improvement": float(p95_gain),
            "oracle_mean_ratio": mean_ratio,
            "minimum_leave_one_family_tail_gain": loo_gain,
            "control_dominance": control_dominance,
            "causal_alignment": alignment, "gates": gates,
            "oracle_ladder": variant_metrics,
        })
        for row_index in range(len(groups)):
            row = {
                "model": short, "seed": int(seed),
                "trajectory_id": int(groups[row_index]),
                "record_id": int(data["record_id"][row_index]),
                "initial_mode": str(data["initial_mode"][row_index]),
                "terminal_source_mode": str(source_terminal[row_index]),
                "terminal_target_mode": str(target_terminal[row_index]),
                "word": str(data["word"][row_index]),
                "baseline_nmse": float(baseline_error[row_index]),
            }
            for variant in CAUSAL_VARIANTS:
                row[f"{variant}_nmse"] = float(all_errors[variant][-1][row_index])
            EVALUATION_ROWS.append(row)
    stacked = {key: np.stack(value, axis=0) for key, value in all_errors.items()}
    row_gain = paired_rowwise_relative_gain(
        stacked["oracle_reset_ceiling"], stacked["baseline"]
    )
    interval90 = hierarchical_seed_family_interval(
        row_gain, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
        seed=stable_seed(BOOTSTRAP_SEED, "stage41_oracle", short),
        confidence=PRIMARY_CONFIDENCE,
    )
    panel = derive_stage41_panel_decision(
        short,
        all_seed_tail_improvement=all(row["tail_improvement"] for row in gate_rows),
        all_seed_p95_improvement=all(row["p95_improvement"] for row in gate_rows),
        all_seed_mean_noninferiority=all(row["mean_noninferiority"] for row in gate_rows),
        all_seed_control_dominance=all(row["control_dominance"] for row in gate_rows),
        all_seed_causal_alignment=all(row["causal_alignment"] for row in gate_rows),
    )
    return panel, {
        "classification": panel.classification,
        "mean_rowwise_oracle_gain": float(np.mean(row_gain)),
        "hierarchical_interval90": list(interval90),
        "seed_summaries": seed_summaries,
        "panels_pooled": False,
    }


if not PIPELINE_FAILED and SIMULATOR_PREFLIGHT_PASSED:
    try:
        verify_executed_notebook_through(
            "# Open the development-heldout causal panel after every head is frozen."
        )
        certificate_path = CALIBRATION_MODEL_DIR / "evaluation_open_certificate.json"
        validate_digest_sidecar(certificate_path)
        certificate = json.loads(certificate_path.read_text())
        if (
            certificate["protocol_id"] != PROTOCOL_ID
            or certificate["run_signature"] != RUN_SIGNATURE
            or certificate["evaluation_statistics_read"]
            or certificate["evaluation_pairs_generated"]
            or not certificate["planning_permanently_sealed"]
        ):
            raise RuntimeError("Stage 41 evaluation-open certificate is invalid")
        for index, record in enumerate(SELECTED_RECORDS["evaluation"]):
            generate_stage41_paired_record(record, "evaluation_closure")
            write_json(OUT / "paired_evaluation_progress.json", {
                "completed": index + 1, "total": len(SELECTED_RECORDS["evaluation"]),
                "last_record_id": int(record["record_id"]),
            })
        for model_name in MODEL_NAMES:
            bundle = load_world_model(model_name)
            short = bundle["short"]
            try:
                for index, record in enumerate(SELECTED_RECORDS["evaluation"]):
                    generate_stage39_path_record(
                        bundle, record, "evaluation_closure", DECODERS[short]
                    )
                    write_json(OUT / f"model_{short}_evaluation_progress.json", {
                        "completed": index + 1,
                        "total": len(SELECTED_RECORDS["evaluation"]),
                        "last_record_id": int(record["record_id"]),
                    })
            finally:
                unload_world_model(bundle)
        EVALUATION_OPENED = True
        for short in ["jepa", "dino"]:
            data = load_stage39_sequences(short, "evaluation_closure")
            pairs = load_stage41_pairs(data, "evaluation_closure")
            panel, metrics = stage41_panel(short, data, pairs)
            PANEL_DECISIONS[short] = panel
            SUMMARY[short] = metrics
        DECISION_PAYLOAD = derive_stage41_decision(PANEL_DECISIONS)
        DECISION_PAYLOAD.update({
            "protocol_id": PROTOCOL_ID, "run_signature": RUN_SIGNATURE,
            "evaluation_opened": True, "planning_opened": False,
            "evidence_tier": "development_only_oracle_headroom",
            "physical_intervention": "ordinary_minus_agent_block_collision_disabled",
            "primary_variant": "oracle_reset_ceiling",
            "causal_variants": CAUSAL_VARIANTS,
            "contact_tail_mass": CONTACT_TAIL_MASS,
            "minimum_contact_tail_relative_improvement": MIN_CONTACT_TAIL_RELATIVE_IMPROVEMENT,
            "minimum_p95_relative_improvement": MIN_P95_RELATIVE_IMPROVEMENT,
            "maximum_mean_relative_degradation": MAX_MEAN_RELATIVE_DEGRADATION,
            "causal_claim_authorized": False,
            "learned_deployment_claim_authorized": False,
            "stage40_artifacts_read": False,
            "stage40_evaluation_rows_consumed": False,
        })
        write_csv(EVIDENCE_DIR / "heldout_stage41_rows.csv", EVALUATION_ROWS)
        write_json(EVIDENCE_DIR / "stage41_summary.json", SUMMARY)
        write_json(OUT / "stage41_decision.json", DECISION_PAYLOAD)
        atomic_checkpoint("stage41_causal_headroom_complete", {
            "decision_sha256": sha256_file(OUT / "stage41_decision.json"),
            "status": DECISION_PAYLOAD["status"], "planning_opened": False,
        })

        figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        for axis, short in zip(axes, ["jepa", "dino"]):
            estimate = SUMMARY[short]["mean_rowwise_oracle_gain"]
            low, high = SUMMARY[short]["hierarchical_interval90"]
            axis.errorbar([0], [estimate], yerr=[[estimate - low], [high - estimate]], fmt="o")
            axis.axhline(0, color="black", linewidth=1)
            axis.set(
                xticks=[], ylabel="oracle reset over frozen baseline gain",
                title=f"{short.upper()}: {PANEL_DECISIONS[short].classification}",
            )
        figure.suptitle(f"Stage 41: {DECISION_PAYLOAD['status']}")
        figure.tight_layout()
        figure.savefig(PLOT_DIR / "stage41_causal_headroom.png", dpi=180)
        plt.close(figure)
        interpretation = f"""# Automatic Stage 41 interpretation

Status: **{DECISION_PAYLOAD['status'].upper()}**

This development-only audit compares exact simulator event/reset metadata with
equal-width smooth and shuffled controls on top of a frozen recursive model.
A positive result is headroom evidence only and authorizes a separate
label-free identifiability experiment.  It is not a learned causal, planning,
or deployment result.  Planning remained sealed.
"""
        (OUT / "AUTOMATIC_INTERPRETATION.md").write_text(interpretation)
        print(json.dumps(DECISION_PAYLOAD, indent=2))
    except Exception:
        record_failure("stage41_heldout_causal_panel")
'''


packaging = BASE.packaging
packaging = packaging.replace("stage40_ctrd", "stage41_cerh")
packaging = packaging.replace(
    "contact_tail_risk_distillation", "causal_event_reset_headroom"
)
packaging = packaging.replace("stage40_decision.json", "stage41_decision.json")
packaging = packaging.replace("raw_roots = [TRUTH_DIR, PATH_DIR]", "raw_roots = [TRUTH_DIR, PATH_DIR, CAUSAL_DIR]")


protocol_sources = [
    introduction, configuration, installation, setup, analysis_helpers,
    model_helpers, design_and_runtime_helpers, physical_truth,
    simulator_preflight, construction_and_paths, data_and_selection,
    causal_interventions, calibration, heldout_evaluation, packaging,
]
protocol_sources = [value.strip() for value in protocol_sources]
protocol_digest = hashlib.sha256(
    json.dumps(protocol_sources, ensure_ascii=False, separators=(",", ":")).encode()
).hexdigest()
configuration = configuration.replace("__PROTOCOL_DIGEST__", protocol_digest)
if "__PROTOCOL_DIGEST__" in configuration:
    raise RuntimeError("Stage 41 protocol digest placeholder was not replaced")
protocol_sources[1] = configuration

cells = [markdown(introduction)] + [code(value) for value in protocol_sources[1:]]
for index, cell in enumerate(cells):
    cell["id"] = f"stage41-{index:02d}"
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
