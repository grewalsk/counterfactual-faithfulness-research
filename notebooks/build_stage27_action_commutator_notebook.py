import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "27_causal_action_commutator.ipynb"
BASE = json.loads((ROOT / "19_unseen_action_family_transfer.ipynb").read_text())
NUMERICAL = ROOT.parent / "src/cf_faithfulness/stage27_action_commutator.py"


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(keepends=True),
    }


def markdown(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
    }


def base_source(index):
    return "".join(BASE["cells"][index]["source"])


def checked_replace(source, old, new):
    if old not in source:
        raise RuntimeError(f"Stage 19 template changed; missing {old[:100]!r}")
    return source.replace(old, new)


def assigned_uppercase_names(source):
    tree = ast.parse(source)
    names = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id.isupper():
                names.append(target.id)
    return list(dict.fromkeys(names))


def function_sources(source, names):
    tree = ast.parse(source)
    by_name = {
        node.name: ast.get_source_segment(source, node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    missing = [name for name in names if name not in by_name]
    if missing:
        raise RuntimeError(f"missing source definitions: {missing}")
    return "\n\n\n".join(by_name[name] for name in names)


def without_definitions(source, names):
    tree = ast.parse(source)
    nodes = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    missing = sorted(set(names).difference(nodes))
    if missing:
        raise RuntimeError(f"missing removable definitions: {missing}")
    removed = set()
    for name in names:
        node = nodes[name]
        removed.update(range(node.lineno - 1, node.end_lineno))
    lines = source.splitlines(keepends=True)
    return "".join(line for index, line in enumerate(lines) if index not in removed)


introduction = r'''# Stage 27: causal finite-action commutator

Stages 18 and 19 established a distributed predictor-block-4 carrier that is
causally sufficient and necessary for action-conditioned predicted
consequences.  The same frozen carrier transferred to rotated directions, new
magnitudes, and delayed and pulsed equal-impulse action families.  Stages
22--26 then found that contact, normal impulse, operator mode, completion, and
a contact-frame response field were readable but did not act as compact causal
variables.

Stage 27 tests the resulting mathematical hypothesis: contact-sensitive
dynamics may be represented as an **ordered interaction between action
effects**, rather than as a stored contact scalar.  For two finite control
pulses (u) and (v), compare

\[
u\!\rightarrow\!v \quad\text{with}\quad v\!\rightarrow\!u.
\]

The paired sequences have exactly the same pulses, integrated impulse,
duration, and energy.  Only temporal order changes.  Their exact simulator
difference is a finite action commutator.  In a smooth small-pulse limit it is
second order and approaches a Lie bracket, but this notebook computes only
finite simulator rollouts and finite model interventions—no Jacobians.

Six direction pairs are frozen: four contact-seeking paths and two paths
pointing away from the object.  Model-blind simulator screening retains fresh
states with both contact and no-contact pairs, without looking at commutator
magnitude. The physical gate then tests for a nondegenerate order effect, and
the full model must predict its direction. The
exact frozen Stage 18 rank-128 subspace is then tested for causal sufficiency
and pair-specific necessity, with rank-64, shuffled, empirical-span random,
wrong-state, common-mode, no-contact, reverse-dose, and full-activation-swap
controls.

This notebook fits no new representation, reader, layer, or intervention
direction.  It uses an automatically generated fresh run nonce and resolves
its committed source from the GitHub branch, so no Stage 27 Colab secrets are
required.  An existing account-level `HF_TOKEN` is used automatically when
available, but the pinned public assets can normally be fetched without it.

Return `stage27_action_commutator_result_bundle_<signature>.zip`.
'''


configuration = r'''# SINGLE CONFIGURATION BLOCK — no Stage 27 secrets required.
# Run all on a GPU. The nonce is fresh automatically and the branch is resolved
# to an exact commit before model activations are opened.
import secrets as _secrets
import time as _time

RUN_MODE = "pilot"
EXPERIMENT_SOURCE_REF = "codex/stage27-causal-action-commutator"
RUN_NONCE = f"auto_{_time.strftime('%Y%m%d_%H%M%S')}_{_secrets.token_hex(4)}"

# Optional account-level Hugging Face token. It is never written to config,
# logs, Drive artifacts, or the result bundle.
try:
    import os as _os
    from google.colab import userdata as _colab_userdata

    _hf_token = str(_colab_userdata.get("HF_TOKEN") or "").strip()
    if _hf_token:
        _os.environ["HF_TOKEN"] = _hf_token
        _os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_token
except Exception:
    _hf_token = ""

if not all(value.isalnum() or value in "-_" for value in RUN_NONCE):
    raise ValueError("automatic RUN_NONCE contains an invalid character")

MOUNT_DRIVE = True
DOWNLOAD_RESULTS = True
CONTINUE_AFTER_BENCHMARK = True
MAX_ESTIMATED_TOTAL_MINUTES = 45.0
FRESH_RUN_REQUIRED = True

OUTPUT_DIR = "/content/counterfactual_faithfulness_stage27_commutator"
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/counterfactual_faithfulness_stage27_commutator"
STAGE18_SEARCH_ROOT = "/content/drive/MyDrive"
STAGE19_SEARCH_ROOT = "/content/drive/MyDrive"

PROTOCOL_ID = "stage27-causal-finite-action-commutator-v1"
NOTEBOOK_PROTOCOL_SHA256 = "__PROTOCOL_DIGEST__"
EVIDENCE_STATUS = "CONFIRMATORY_ONLY_IF_SOURCE_BOUND_FRESH_STAGE18_FROZEN_AND_STAGE19_BOUND"
EXPERIMENT_REPOSITORY = "grewalsk/counterfactual-faithfulness-research"
EXPERIMENT_NOTEBOOK_PATH = "notebooks/27_causal_action_commutator.ipynb"
EXPERIMENT_BUILDER_PATH = "notebooks/build_stage27_action_commutator_notebook.py"
EXPERIMENT_NUMERICAL_PATH = "src/cf_faithfulness/stage27_action_commutator.py"

EXPECTED_STAGE18_SUBSPACE_SHA256 = "2f9c496d54623a9062e465a18c70039acc18cb8a1cc2833a5f4ade162ca3f90b"
EXPECTED_STAGE18_SOURCE_COMMIT = "16edd247cddcb1aa121340eb5fa42bd9e07004c3"
EXPECTED_STAGE18_STATUS = "CONFIRMED_BIDIRECTIONAL_RANK64_MEDIATOR"
EXPECTED_STAGE18_AMBIENT_DIMENSION = 102400
EXPECTED_STAGE18_MAX_RANK = 128
EXPECTED_STAGE19_STATUS = "CONFIRMED_TRANSFER_ALL_UNSEEN_ACTION_FAMILIES"
EXPECTED_STAGE19_RUN_SUFFIX = "b7f2b6cef37f"
EXPECTED_STAGE19_SOURCE_COMMIT = "bf8c3950fc1112b38baa2453e39793592537ec47"

SEED = 27101
DESIGN_SEED = 27137
MODEL_NAME = "jepa_wm_pusht"
ENVIRONMENT = "PushT"
FRAMESKIP = 5
PRIMARY_HORIZON = 3
TARGET_STEPS = [PRIMARY_HORIZON]
FIXED_BLOCK = 4
ACTIVE_BLOCKS = [FIXED_BLOCK]
EXPECTED_CARRIER_CHANNELS = 400

EVALUATION_POOL_TRAJECTORIES = list(range(3000, 3080))
EVALUATION_TRAJECTORY_TARGET = 40
TASK_ID_OFFSET = 13000
STATES_PER_TRAJECTORY = 1
ACTION_STEPS = PRIMARY_HORIZON * FRAMESKIP
PULSE_STEPS = 5
PULSE_MAGNITUDE = 0.18
ACTION_PAIR_COUNT = 6
ACTIONS_PER_STATE = 2 * ACTION_PAIR_COUNT
APPROACH_DISTANCE = 80.0
PAIR_ANGLES_DEGREES = [
    [-30.0, 30.0], [-45.0, 15.0], [-15.0, 45.0], [-60.0, 60.0],
    [150.0, 210.0], [135.0, 225.0],
]
PAIR_DESIGN_LABELS = [
    "contact_symmetric_30", "contact_left_biased",
    "contact_right_biased", "contact_symmetric_60",
    "free_symmetric_30", "free_symmetric_45",
]
MIN_ELIGIBLE_CONTACT_PAIRS = 2
MIN_ELIGIBLE_FREE_PAIRS = 1
MIN_ELIGIBLE_CONTACT_COMMUTATOR_NORM = 1e-4
MIN_VALID_CONTACT_PAIRS = 60

OUTPUT_SKETCH_DIM = 256
TRAIN_OUTPUT_SKETCH_SEED = 18161
EVAL_OUTPUT_SKETCH_SEED = 18183
PRIMARY_RANK = 128
SENSITIVITY_RANKS = [64, 128]
MAX_SUBSPACE_RANK = 128
BOOTSTRAP_SEED = 27269
CAUSAL_RANDOM_DRAWS = 4
CAUSAL_DOSES = [-0.5, 0.25, 0.5, 1.0]
BOOTSTRAP_DRAWS = 10000
INTERVENTION_FORWARDS_PER_RECORD = 30
MAX_ZERO_EDIT_ERROR = 1e-6

MIN_CONTACT_TO_FREE_TRUE_NORM_RATIO = 1.5
MIN_MODEL_CONTACT_COMMUTATOR_COSINE = 0.10
MIN_MODEL_ALIGNMENT_GAIN_OVER_SHUFFLED = 0.05
MIN_FULL_SWAP_COEFFICIENT = 0.75
MIN_PRIMARY_CONTACT_COEFFICIENT = 0.15
MIN_PRIMARY_CONTACT_COSINE = 0.15
MIN_PRIMARY_GAIN_OVER_RANDOM = 0.05
MIN_PRIMARY_GAIN_OVER_SHUFFLED = 0.05
REQUIRED_POSITIVE_EVALUATION_TRAJECTORIES = 28
MIN_NECESSITY_REDUCTION = 0.05
MIN_NECESSITY_GAIN_OVER_RANDOM = 0.025
MIN_NECESSITY_GAIN_OVER_SHUFFLED = 0.025
REQUIRED_POSITIVE_NECESSITY_TRAJECTORIES = 28

if RUN_MODE == "smoke":
    ACTIVE_EVALUATION_POOL_TRAJECTORIES = EVALUATION_POOL_TRAJECTORIES[:8]
    ACTIVE_EVALUATION_TARGET = 2
    ACTIVE_SENSITIVITY_RANKS = [PRIMARY_RANK]
    ACTIVE_CAUSAL_RANDOM_DRAWS = 1
    ACTIVE_CAUSAL_DOSES = [1.0]
    ACTIVE_BOOTSTRAP_DRAWS = 64
    ACTIVE_MIN_VALID_CONTACT_PAIRS = 1
elif RUN_MODE == "pilot":
    ACTIVE_EVALUATION_POOL_TRAJECTORIES = EVALUATION_POOL_TRAJECTORIES
    ACTIVE_EVALUATION_TARGET = EVALUATION_TRAJECTORY_TARGET
    ACTIVE_SENSITIVITY_RANKS = SENSITIVITY_RANKS
    ACTIVE_CAUSAL_RANDOM_DRAWS = CAUSAL_RANDOM_DRAWS
    ACTIVE_CAUSAL_DOSES = CAUSAL_DOSES
    ACTIVE_BOOTSTRAP_DRAWS = BOOTSTRAP_DRAWS
    ACTIVE_MIN_VALID_CONTACT_PAIRS = MIN_VALID_CONTACT_PAIRS
else:
    raise ValueError(f"RUN_MODE must be 'smoke' or 'pilot'; received {RUN_MODE!r}")

REPO_URL = "https://github.com/facebookresearch/jepa-wms.git"
REPO_COMMIT = "13cf1d9c7e476f53c17714d2e0f1dc239a883ce0"
EXPECTED_HF_REVISION = "9b9c41ef249466630dbf1a20e78391865d07b3b9"
EXPECTED_PRETRAINED_ASSET_SHA256 = {
    "jepa_wm_pusht.pth.tar": "9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb",
    "dinov2_vits14_pretrain.pth": "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9",
}
ASSET_REPOSITORY = "grewalsk/counterfactual-faithfulness-research"
ASSET_COMMIT = "2326e74556f6f81db2560e4396f4cc52c16a28f4"
ASSET_SPECS = {
    "physical_decoders.pt": {
        "path": "results/bundles/stage12_result_bundle/frozen_training_decoders/jepa_wm_pusht_f975a0a746e7_training_decoders.pt",
        "sha256": "51b2dbb0a81df432a2db5b941de83717e9979e761d57365f47d93d2dd0c0c694",
    },
}

assert ACTION_STEPS == 15 and PULSE_STEPS == 5
assert ACTION_PAIR_COUNT == len(PAIR_DESIGN_LABELS) == 6
assert ACTIONS_PER_STATE == 12
assert FIXED_BLOCK == 4
assert PRIMARY_RANK == 128 and SENSITIVITY_RANKS == [64, 128]
assert MAX_SUBSPACE_RANK == EXPECTED_STAGE18_MAX_RANK
'''
configuration += "\n\nPROTOCOL_CONFIG_KEYS = " + repr(
    assigned_uppercase_names(configuration)
) + "\n"


installation = base_source(2)


setup = base_source(3)
setup = setup.replace("Stage 19", "Stage 27").replace("STAGE19", "STAGE27")
setup = setup.replace("stage19_transfer", "stage27_commutator")
setup = setup.replace(
    "fresh pilot output already exists; choose a new STAGE27_RUN_NONCE",
    "fresh pilot output already exists; rerun the configuration cell for a fresh automatic nonce",
)
setup = setup.replace(
    "stage19_unseen_action_transfer_result_bundle_",
    "stage27_action_commutator_result_bundle_",
)
setup = setup.replace("import urllib.request", "import urllib.parse\nimport urllib.request")
source_start = setup.index("def source_identity():")
source_stop = setup.index("\n\ndef verify_executed_notebook_through", source_start)
setup = setup[:source_start] + r'''def source_identity():
    global REMOTE_NOTEBOOK_CODE_CELLS
    payload = {
        "protocol_id": PROTOCOL_ID,
        "notebook_protocol_sha256": NOTEBOOK_PROTOCOL_SHA256,
        "repository": EXPERIMENT_REPOSITORY,
        "source_ref": EXPERIMENT_SOURCE_REF,
        "execution_verified": False,
    }
    if not EXPERIMENT_SOURCE_REF:
        raise RuntimeError("Stage 27 requires its committed GitHub branch")
    source_ref = str(EXPERIMENT_SOURCE_REF).strip()
    if len(source_ref) == 40 and all(value in "0123456789abcdef" for value in source_ref.lower()):
        resolved = source_ref.lower()
    else:
        encoded_ref = urllib.parse.quote(source_ref, safe="")
        request = urllib.request.Request(
            f"https://api.github.com/repos/{EXPERIMENT_REPOSITORY}/commits/{encoded_ref}",
            headers={"Accept": "application/vnd.github+json", "User-Agent": "stage27-source-binder"},
        )
        with urllib.request.urlopen(request) as response:
            resolved = str(json.loads(response.read().decode())["sha"]).lower()
    if len(resolved) != 40 or any(value not in "0123456789abcdef" for value in resolved):
        raise RuntimeError(f"GitHub returned an invalid commit for {source_ref!r}: {resolved!r}")
    payload["resolved_commit"] = resolved
    base = f"https://raw.githubusercontent.com/{EXPERIMENT_REPOSITORY}/{resolved}/"
    payload["files"] = {}
    for label, relative in [
        ("notebook", EXPERIMENT_NOTEBOOK_PATH),
        ("builder", EXPERIMENT_BUILDER_PATH),
        ("numerical", EXPERIMENT_NUMERICAL_PATH),
    ]:
        with urllib.request.urlopen(base + relative) as response:
            content = response.read()
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
''' + setup[source_stop:]


analysis_helpers = without_definitions(
    base_source(4),
    ["rotate_vector", "unseen_action_bank"],
)
analysis_helpers += "\n\n\n" + function_sources(
    NUMERICAL.read_text(),
    [
        "rotate_vector",
        "pair_swap_permutation",
        "ordered_pulse_bank",
        "paired_antisymmetric_component",
        "paired_swap_delta",
        "paired_ablation_delta",
        "_pair_rows",
        "paired_transfer_metrics",
        "paired_energy_metrics",
        "pair_contact_masks",
        "commutator_contrasts",
        "commutator_norms",
        "commutator_alignment_metrics",
    ],
)


model_helpers = base_source(5).replace("stage19-jepa-wms", "stage27-jepa-wms")
model_helpers = model_helpers.replace("Stage 19 supports PushT only", "Stage 27 supports PushT only")


design = r'''# Freeze the paired action-order design before simulator or model data.


def trajectory_specs():
    specs = []
    center = np.asarray([256.0, 256.0])
    total = len(EVALUATION_POOL_TRAJECTORIES)
    for design_index, trajectory_id in enumerate(EVALUATION_POOL_TRAJECTORIES):
        phase = 0.37 + 2.0 * np.pi * design_index / total
        block = center + 44.0 * np.asarray([np.cos(phase), np.sin(phase)])
        block_angle = ((1.7 * phase + np.pi) % (2.0 * np.pi)) - np.pi
        approach = phase + [
            np.pi / 6, 5 * np.pi / 6, 7 * np.pi / 6, 11 * np.pi / 6
        ][design_index % 4]
        approach += 0.11 * np.cos(5 * design_index)
        agent = block + APPROACH_DISTANCE * np.asarray([
            np.cos(approach), np.sin(approach)
        ])
        goal_index = (17 * design_index + 5) % total
        goal_phase = 0.79 + 2.0 * np.pi * goal_index / total
        goal_xy = center + 72.0 * np.asarray([
            np.cos(goal_phase), np.sin(goal_phase)
        ])
        specs.append(
            {
                "design_index": int(design_index),
                "record_id": int(300000 + trajectory_id),
                "trajectory_id": int(trajectory_id),
                "task_id": int(TASK_ID_OFFSET + design_index),
                "time_index": 0,
                "physical_step": 0,
                "split": "evaluation",
                "evaluation_seed": int(DESIGN_SEED + 1009 * design_index),
                "goal": np.asarray(
                    [
                        goal_xy[0], goal_xy[1],
                        ((1.3 * goal_phase + np.pi) % (2.0 * np.pi)) - np.pi,
                    ],
                    dtype=np.float64,
                ),
                "state": np.asarray(
                    [
                        agent[0], agent[1], block[0], block[1], block_angle,
                        0.0, 0.0, 0.0, 0.0, 0.0,
                    ],
                    dtype=np.float64,
                ),
            }
        )
    return specs


ALL_POOL_SPECS = trajectory_specs()
POOL_SPECS = [
    row for row in ALL_POOL_SPECS
    if row["trajectory_id"] in ACTIVE_EVALUATION_POOL_TRAJECTORIES
]


def candidate_action_bank(record):
    state = np.asarray(record["state"], dtype=np.float64)
    if state.shape != (10,):
        raise ValueError("candidate state must be a ten-dimensional dynamic PushT state")
    return ordered_pulse_bank(
        state[2:4] - state[:2],
        steps=ACTION_STEPS,
        pulse_steps=PULSE_STEPS,
        magnitude=PULSE_MAGNITUDE,
    )


np.savez_compressed(
    DESIGN_DIR / "stage27_action_commutator_pool_design.npz",
    record_ids=np.asarray([row["record_id"] for row in ALL_POOL_SPECS]),
    trajectory_ids=np.asarray([row["trajectory_id"] for row in ALL_POOL_SPECS]),
    initial_states=np.stack([row["state"] for row in ALL_POOL_SPECS]),
    goals=np.stack([row["goal"] for row in ALL_POOL_SPECS]),
    pair_design_labels=np.asarray(PAIR_DESIGN_LABELS),
    pair_swap_permutation=pair_swap_permutation(ACTION_PAIR_COUNT),
)
POOL_MANIFEST = {
    "specs": [
        {
            **{key: value for key, value in row.items() if key not in {"state", "goal"}},
            "state": row["state"].tolist(),
            "goal": row["goal"].tolist(),
        }
        for row in ALL_POOL_SPECS
    ],
    "active_pool_trajectory_ids": ACTIVE_EVALUATION_POOL_TRAJECTORIES,
    "evaluation_target": ACTIVE_EVALUATION_TARGET,
    "pair_design_labels": PAIR_DESIGN_LABELS,
    "pair_swap_permutation": pair_swap_permutation(ACTION_PAIR_COUNT).tolist(),
    "paired_sequences_have_equal_impulse_energy_and_duration": True,
    "selection_uses_model_outputs": False,
    "eligibility": {
        "min_contact_pairs": MIN_ELIGIBLE_CONTACT_PAIRS,
        "min_free_pairs": MIN_ELIGIBLE_FREE_PAIRS,
        "physical_commutator_magnitude_used_for_selection": False,
    },
}
write_json(DESIGN_DIR / "candidate_pool_manifest.json", POOL_MANIFEST)
DESIGN_FREEZE = {
    "created_before_simulator_or_model_data": True,
    "protocol_id": PROTOCOL_ID,
    "run_signature": RUN_SIGNATURE,
    "source_identity": SOURCE_IDENTITY,
    "candidate_pool_sha256": sha256_file(
        DESIGN_DIR / "stage27_action_commutator_pool_design.npz"
    ),
    "pool_manifest_sha256": sha256_file(
        DESIGN_DIR / "candidate_pool_manifest.json"
    ),
    "expected_stage18_subspace_sha256": EXPECTED_STAGE18_SUBSPACE_SHA256,
    "fixed_block": FIXED_BLOCK,
    "fixed_primary_rank": PRIMARY_RANK,
    "fixed_sensitivity_ranks": SENSITIVITY_RANKS,
    "subspace_refit_allowed": False,
    "coordinate_reader_used": False,
    "jacobian_used": False,
    "model_loaded": bool("MODEL" in globals()),
}
if DESIGN_FREEZE["model_loaded"]:
    raise RuntimeError("model was loaded before Stage 27 design freeze")
write_json(DESIGN_DIR / "design_freeze.json", DESIGN_FREEZE)
'''


truth_generation = r'''# Generate exact paired simulator truth before loading the model.


def record_task(record):
    return {"goal": np.asarray(record["goal"], dtype=np.float64).tolist()}


def dynamic_state_from_environment(environment):
    return np.asarray(
        [
            *environment.agent.position,
            *environment.block.position,
            float(environment.block.angle),
            *environment.agent.velocity,
            *environment.block.velocity,
            float(environment.block.angular_velocity),
        ],
        dtype=np.float64,
    )


def reset_dynamic_environment(dynamic_state, task, seed):
    state = np.asarray(dynamic_state, dtype=np.float64)
    if state.shape != (10,):
        raise ValueError(f"expected ten-dimensional dynamic state, found {state.shape}")
    environment = make_environment(REPO, ENVIRONMENT)
    environment.seed(int(seed))
    environment.reset_to_state = np.asarray([*state[:5], 0.0, 0.0], dtype=np.float64)
    environment.reset()
    environment.agent.position = tuple(state[:2])
    environment.block.angle = float(state[4])
    environment.block.position = tuple(state[2:4])
    environment.agent.velocity = tuple(state[5:7])
    environment.block.velocity = tuple(state[7:9])
    environment.block.angular_velocity = float(state[9])
    environment.set_task_goal(np.asarray(task["goal"], dtype=np.float64))
    restored = dynamic_state_from_environment(environment)
    if not np.allclose(restored, state, atol=1e-12, rtol=0):
        raise RuntimeError(
            f"full dynamic restoration drifted: {np.max(np.abs(restored - state))}"
        )
    observation = {
        "visual": np.asarray(environment.render("rgb_array")).copy(),
        "proprio": np.asarray(
            [*environment.agent.position, *environment.agent.velocity],
            dtype=np.float32,
        ),
    }
    return environment, observation


def rollout_dynamic_branch(record, actions):
    environment, initial = reset_dynamic_environment(
        record["state"], record_task(record), record["evaluation_seed"]
    )
    cumulative = 0
    endpoint_observation = None
    endpoint_state = None
    try:
        for step, action in enumerate(actions, start=1):
            observation, _, _, info = environment.step(action)
            cumulative += int(info.get("n_contacts", 0))
            if step == ACTION_STEPS:
                endpoint_observation = {
                    "visual": np.asarray(observation["visual"]).copy(),
                    "proprio": np.asarray(observation["proprio"]).copy(),
                }
                endpoint_state = dynamic_state_from_environment(environment)
    finally:
        environment.close()
    if endpoint_observation is None or endpoint_state is None:
        raise RuntimeError("dynamic rollout missed the primary horizon")
    return initial, endpoint_observation, endpoint_state, cumulative


def exact_dynamic_restore_test(record):
    first, first_observation = reset_dynamic_environment(
        record["state"], record_task(record), record["evaluation_seed"]
    )
    second, second_observation = reset_dynamic_environment(
        record["state"], record_task(record), record["evaluation_seed"]
    )
    first_state = dynamic_state_from_environment(first)
    second_state = dynamic_state_from_environment(second)
    test_action = candidate_action_bank(record)[0, 0]
    first.step(test_action)
    second.step(test_action)
    first_next = dynamic_state_from_environment(first)
    second_next = dynamic_state_from_environment(second)
    first.close()
    second.close()
    result = {
        "state_exact": bool(np.allclose(first_state, second_state, atol=1e-12, rtol=0)),
        "visual_exact": bool(np.array_equal(
            first_observation["visual"], second_observation["visual"]
        )),
        "proprio_exact": bool(np.array_equal(
            first_observation["proprio"], second_observation["proprio"]
        )),
        "one_step_continuation_exact": bool(np.allclose(
            first_next, second_next, atol=1e-12, rtol=0
        )),
    }
    result["passed"] = bool(all(result.values()))
    if not result["passed"]:
        raise RuntimeError(f"full dynamic restore test failed: {result}")
    return result


def branch_path(record_id):
    return TRUTH_DIR / f"state_{int(record_id):06d}.npz"


def generate_truth(records):
    started = time.perf_counter()
    for index, record in enumerate(records):
        destination = branch_path(record["record_id"])
        if destination.exists():
            PROVENANCE_COUNTS["cache_hits"] += 1
            raise RuntimeError(f"fresh-run truth shard already exists: {destination}")
        action_bank = candidate_action_bank(record)
        initials, initial_proprios = [], []
        endpoint_visuals, endpoint_states, interaction_counts = [], [], []
        for action in action_bank:
            initial, endpoint, state, contacts = rollout_dynamic_branch(record, action)
            initials.append(initial["visual"])
            initial_proprios.append(initial["proprio"])
            endpoint_visuals.append(endpoint["visual"])
            endpoint_states.append(state)
            interaction_counts.append(contacts)
        if not all(np.array_equal(initials[0], value) for value in initials[1:]):
            raise AssertionError("initial visual drift across action-order branches")
        if not all(np.array_equal(initial_proprios[0], value) for value in initial_proprios[1:]):
            raise AssertionError("initial proprio drift across action-order branches")
        atomic_npz(
            destination,
            record_id=np.asarray(record["record_id"], dtype=np.int64),
            trajectory_id=np.asarray(record["trajectory_id"], dtype=np.int64),
            task_id=np.asarray(record["task_id"], dtype=np.int64),
            split=np.asarray(record["split"]),
            state=np.asarray(record["state"], dtype=np.float64),
            goal=np.asarray(record["goal"], dtype=np.float64),
            initial_visual=np.asarray(initials[0], dtype=np.uint8),
            initial_proprio=np.asarray(initial_proprios[0], dtype=np.float32),
            selected_actions=action_bank.astype(np.float32),
            endpoint_visuals=np.asarray(endpoint_visuals, dtype=np.uint8),
            endpoint_states=np.asarray(endpoint_states, dtype=np.float64),
            interaction_counts=np.asarray(interaction_counts, dtype=np.int32),
        )
        PROVENANCE_COUNTS["truth_generated"] += 1
        write_json(
            OUT / "truth_commutator_pool_progress.json",
            {
                "completed": index + 1,
                "total": len(records),
                "last_record_id": int(record["record_id"]),
            },
        )
    TIMINGS["truth_commutator_pool_seconds"] = time.perf_counter() - started


def truth_pair_rows(record):
    with np.load(branch_path(record["record_id"])) as payload:
        endpoints = payload["endpoint_states"].astype(np.float64)
        contacts = payload["interaction_counts"].astype(np.int64)
        actions = payload["selected_actions"].astype(np.float64)
    poses = pose_target(endpoints)
    masks = pair_contact_masks(contacts)
    norms = commutator_norms(poses)
    rows = []
    for pair_index, label in enumerate(PAIR_DESIGN_LABELS):
        regime = (
            "both_contact" if masks["both_contact"][pair_index]
            else "one_contact" if masks["one_contact"][pair_index]
            else "free"
        )
        rows.append(
            {
                "record_id": int(record["record_id"]),
                "trajectory_id": int(record["trajectory_id"]),
                "pair_index": int(pair_index),
                "pair_design": label,
                "contact_regime": regime,
                "first_order_contacts": int(contacts[2 * pair_index]),
                "second_order_contacts": int(contacts[2 * pair_index + 1]),
                "physical_commutator_norm": float(norms[pair_index]),
                "equal_integrated_impulse": bool(np.allclose(
                    np.sum(actions[2 * pair_index], axis=0),
                    np.sum(actions[2 * pair_index + 1], axis=0),
                    atol=1e-7,
                )),
                "equal_control_energy": bool(np.isclose(
                    np.sum(actions[2 * pair_index] ** 2),
                    np.sum(actions[2 * pair_index + 1] ** 2),
                    atol=1e-7,
                )),
            }
        )
    return rows


def select_records(records, target):
    pair_rows = [row for record in records for row in truth_pair_rows(record)]
    by_record = defaultdict(list)
    for row in pair_rows:
        by_record[int(row["record_id"])].append(row)
    eligibility = []
    for record in records:
        rows = by_record[int(record["record_id"])]
        contact = [row for row in rows if row["contact_regime"] != "free"]
        free = [row for row in rows if row["contact_regime"] == "free"]
        nondegenerate = [
            row for row in contact
            if row["physical_commutator_norm"] >= MIN_ELIGIBLE_CONTACT_COMMUTATOR_NORM
        ]
        eligible = bool(
            len(contact) >= MIN_ELIGIBLE_CONTACT_PAIRS
            and len(free) >= MIN_ELIGIBLE_FREE_PAIRS
            and all(row["equal_integrated_impulse"] for row in rows)
            and all(row["equal_control_energy"] for row in rows)
        )
        eligibility.append(
            {
                "record_id": int(record["record_id"]),
                "trajectory_id": int(record["trajectory_id"]),
                "contact_pairs": int(len(contact)),
                "free_pairs": int(len(free)),
                "nondegenerate_contact_pairs": int(len(nondegenerate)),
                "median_contact_commutator_norm": float(np.median([
                    row["physical_commutator_norm"] for row in contact
                ])) if contact else 0.0,
                "median_free_commutator_norm": float(np.median([
                    row["physical_commutator_norm"] for row in free
                ])) if free else 0.0,
                "eligible": eligible,
            }
        )
    selected_ids = [row["record_id"] for row in eligibility if row["eligible"]][: int(target)]
    if len(selected_ids) != int(target):
        raise RuntimeError(
            f"physical eligibility produced {len(selected_ids)} records but requires {target}"
        )
    selected = [record for record in records if record["record_id"] in selected_ids]
    return selected, eligibility, pair_rows


def make_truth_montage(records):
    sample = records[:4] + records[-4:]
    figure, axes = plt.subplots(len(sample), 3, figsize=(9, 2.7 * len(sample)))
    for row_index, record in enumerate(sample):
        with np.load(branch_path(record["record_id"])) as payload:
            initial = payload["initial_visual"]
            endpoint_visuals = payload["endpoint_visuals"]
            endpoints = payload["endpoint_states"].astype(np.float64)
        norms = commutator_norms(pose_target(endpoints))
        pair_index = int(np.argmax(norms))
        images = [
            (initial, "initial"),
            (endpoint_visuals[2 * pair_index], f"u→v pair {pair_index}"),
            (endpoint_visuals[2 * pair_index + 1], f"v→u |C|={norms[pair_index]:.3g}"),
        ]
        for column, (image, title) in enumerate(images):
            axes[row_index, column].imshow(image)
            axes[row_index, column].set_title(title)
            axes[row_index, column].axis("off")
        axes[row_index, 0].set_ylabel(str(record["trajectory_id"]))
    figure.tight_layout()
    figure.savefig(PLOT_DIR / "stage27_commutator_truth_montage.png", dpi=150)
    plt.close(figure)


if not PIPELINE_FAILED:
    try:
        REPO = configure_repo()
        RESTORE_TEST = exact_dynamic_restore_test(POOL_SPECS[0])
        write_json(OUT / "restore_test.json", RESTORE_TEST)
        generate_truth(POOL_SPECS)
        if "MODEL" in globals():
            raise RuntimeError("model was loaded before physical eligibility selection")
        ALL_EVALUATION_RECORDS, ELIGIBILITY_ROWS, PHYSICAL_PAIR_ROWS = select_records(
            POOL_SPECS, ACTIVE_EVALUATION_TARGET
        )
        write_csv(EVIDENCE_DIR / "physical_eligibility_rows.csv", ELIGIBILITY_ROWS)
        write_csv(EVIDENCE_DIR / "physical_commutator_pair_rows.csv", PHYSICAL_PAIR_ROWS)
        selected_ids = [int(record["record_id"]) for record in ALL_EVALUATION_RECORDS]
        wrong_state_map = {
            str(record_id): selected_ids[(index + 1) % len(selected_ids)]
            for index, record_id in enumerate(selected_ids)
        }
        SELECTION_CERTIFICATE = {
            "selection_completed_before_model_load": True,
            "selection_used_only_simulator_truth": True,
            "physical_commutator_magnitude_used_for_selection": False,
            "selected_trajectory_ids": [
                int(record["trajectory_id"]) for record in ALL_EVALUATION_RECORDS
            ],
            "eligible_pool_count": int(sum(row["eligible"] for row in ELIGIBILITY_ROWS)),
            "pair_swap_permutation": pair_swap_permutation(ACTION_PAIR_COUNT).tolist(),
            "wrong_state_map": wrong_state_map,
            "eligibility_rows_sha256": sha256_file(
                EVIDENCE_DIR / "physical_eligibility_rows.csv"
            ),
            "physical_pair_rows_sha256": sha256_file(
                EVIDENCE_DIR / "physical_commutator_pair_rows.csv"
            ),
        }
        write_json(DESIGN_DIR / "physical_selection_freeze.json", SELECTION_CERTIFICATE)
        make_truth_montage(ALL_EVALUATION_RECORDS)
        memory_report("physical_truth_and_selection_complete")
    except Exception:
        record_failure("physical_truth_selection")
'''


artifact_import = r'''# Auto-locate and validate the frozen Stage 18 carrier and Stage 19 certificate.
STAGE18_ARTIFACT_VALIDATED = False
STAGE19_UPSTREAM_BOUND = False


def unique_matching_path(candidates, expected_hash=None):
    existing = sorted({Path(value) for value in candidates if Path(value).is_file()})
    if expected_hash is not None:
        existing = [value for value in existing if sha256_file(value) == expected_hash]
    if not existing:
        raise FileNotFoundError("no matching frozen upstream artifact was found in MyDrive")
    if len(existing) > 1:
        # Multiple byte-identical copies are harmless; choose the shortest stable path.
        existing.sort(key=lambda value: (len(str(value)), str(value)))
    return existing[0]


if not PIPELINE_FAILED:
    try:
        verify_executed_notebook_through(
            "# Auto-locate and validate the frozen Stage 18 carrier and Stage 19 certificate."
        )
        stage18_root = Path(STAGE18_SEARCH_ROOT)
        preferred_stage18 = (
            stage18_root / "counterfactual_faithfulness_stage18_rank64" /
            "pilot_f1b34beffcac" / "subspaces" /
            "frozen_rank64_confirmation_subspaces.npz"
        )
        stage18_candidates = [preferred_stage18]
        stage18_candidates.extend(stage18_root.glob(
            "counterfactual_faithfulness_stage18_rank64/pilot_*/subspaces/"
            "frozen_rank64_confirmation_subspaces.npz"
        ))
        FROZEN_SUBSPACE_PATH = unique_matching_path(
            stage18_candidates, EXPECTED_STAGE18_SUBSPACE_SHA256
        )
        stage18_run_dir = FROZEN_SUBSPACE_PATH.parent.parent
        stage18_decision_path = stage18_run_dir / "stage18_decision.json"
        stage18_manifest_path = stage18_run_dir / "subspaces/subspace_manifest.json"
        stage18_source_path = stage18_run_dir / "source_identity.json"
        for required in [stage18_decision_path, stage18_manifest_path, stage18_source_path]:
            if not required.is_file():
                raise FileNotFoundError(f"Stage 18 provenance file is missing: {required}")
        stage18_decision = json.loads(stage18_decision_path.read_text())
        stage18_manifest = json.loads(stage18_manifest_path.read_text())
        stage18_source = json.loads(stage18_source_path.read_text())
        if stage18_decision.get("status") != EXPECTED_STAGE18_STATUS:
            raise RuntimeError("Stage 18 decision is not the frozen confirmatory result")
        if not bool(stage18_decision.get("confirmation_eligible", False)):
            raise RuntimeError("Stage 18 result was not confirmation eligible")
        if stage18_manifest.get("subspace_sha256") != EXPECTED_STAGE18_SUBSPACE_SHA256:
            raise RuntimeError("Stage 18 manifest does not bind the required subspace")
        if stage18_source.get("resolved_commit") != EXPECTED_STAGE18_SOURCE_COMMIT:
            raise RuntimeError("Stage 18 source commit mismatch")
        if not bool(stage18_source.get("confirmation_eligible", False)):
            raise RuntimeError("Stage 18 source execution was not verified")
        with np.load(FROZEN_SUBSPACE_PATH) as payload:
            FROZEN_SUBSPACES = {name: payload[name].copy() for name in payload.files}
        artifact_contract = validate_stage18_subspace_arrays(
            FROZEN_SUBSPACES,
            ambient=EXPECTED_STAGE18_AMBIENT_DIMENSION,
            max_rank=EXPECTED_STAGE18_MAX_RANK,
        )
        STAGE18_ARTIFACT_CERTIFICATE = {
            "validated_before_stage27_model_activations": True,
            "auto_located_without_stage27_secret": True,
            "path": str(FROZEN_SUBSPACE_PATH),
            "bytes": int(FROZEN_SUBSPACE_PATH.stat().st_size),
            "sha256": sha256_file(FROZEN_SUBSPACE_PATH),
            "expected_stage18_source_commit": EXPECTED_STAGE18_SOURCE_COMMIT,
            "stage18_decision_status": stage18_decision["status"],
            "stage18_confirmation_eligible": stage18_decision["confirmation_eligible"],
            "stage18_decision_sha256": sha256_file(stage18_decision_path),
            "stage18_manifest_sha256": sha256_file(stage18_manifest_path),
            "stage18_source_identity_sha256": sha256_file(stage18_source_path),
            "artifact_contract": artifact_contract,
            "stage27_subspace_refit": False,
            "stage27_basis_rotation_or_tuning": False,
        }
        write_json(OUT / "stage18_artifact_certificate.json", STAGE18_ARTIFACT_CERTIFICATE)
        STAGE18_ARTIFACT_VALIDATED = True

        stage19_root = Path(STAGE19_SEARCH_ROOT)
        preferred_stage19 = (
            stage19_root / "counterfactual_faithfulness_stage19_transfer" /
            f"pilot_{EXPECTED_STAGE19_RUN_SUFFIX}" / "stage19_decision.json"
        )
        stage19_candidates = [preferred_stage19]
        stage19_candidates.extend(stage19_root.glob(
            f"counterfactual_faithfulness_stage19_transfer/"
            f"pilot_{EXPECTED_STAGE19_RUN_SUFFIX}/stage19_decision.json"
        ))
        stage19_decision_path = unique_matching_path(stage19_candidates)
        stage19_run_dir = stage19_decision_path.parent
        stage19_source_path = stage19_run_dir / "source_identity.json"
        if not stage19_source_path.is_file():
            raise FileNotFoundError(f"Stage 19 source identity is missing: {stage19_source_path}")
        stage19_decision = json.loads(stage19_decision_path.read_text())
        stage19_source = json.loads(stage19_source_path.read_text())
        if stage19_decision.get("status") != EXPECTED_STAGE19_STATUS:
            raise RuntimeError("Stage 19 decision is not the required broad-transfer result")
        if not bool(stage19_decision.get("confirmation_eligible", False)):
            raise RuntimeError("Stage 19 decision was not confirmation eligible")
        if stage19_source.get("resolved_commit") != EXPECTED_STAGE19_SOURCE_COMMIT:
            raise RuntimeError("Stage 19 source commit mismatch")
        if not bool(stage19_source.get("confirmation_eligible", False)):
            raise RuntimeError("Stage 19 source execution was not verified")
        STAGE19_CERTIFICATE = {
            "validated_before_stage27_model_activations": True,
            "auto_located_without_stage27_secret": True,
            "path": str(stage19_decision_path),
            "decision_status": stage19_decision["status"],
            "confirmation_eligible": stage19_decision["confirmation_eligible"],
            "resolved_commit": stage19_source["resolved_commit"],
            "decision_sha256": sha256_file(stage19_decision_path),
            "source_identity_sha256": sha256_file(stage19_source_path),
        }
        write_json(OUT / "stage19_upstream_certificate.json", STAGE19_CERTIFICATE)
        STAGE19_UPSTREAM_BOUND = True
        memory_report("upstream_artifacts_validated")
    except Exception:
        record_failure("upstream_artifact_import")
'''


model_and_baselines = base_source(9)
model_and_baselines = model_and_baselines.replace(
    "# Load frozen JEPA-WM and generate fresh unseen-family baselines at fixed block 4.",
    "# Load frozen JEPA-WM and generate Stage 27 paired-order baselines at block 4.",
)
model_and_baselines = checked_replace(
    model_and_baselines,
    '    interventions_per_record = INTERVENTION_FORWARDS_PER_RECORD if RUN_MODE == "pilot" else 9',
    '    interventions_per_record = INTERVENTION_FORWARDS_PER_RECORD if RUN_MODE == "pilot" else 5',
)
model_final = model_and_baselines.index("EVALUATION_OPENED = False")
model_and_baselines = model_and_baselines[:model_final] + r'''def evaluate_model_physical_commutators(records):
    pair_rows = []
    record_rows = []
    for record in records:
        payload = load_baseline(record["record_id"])
        predicted_pose = payload["decoded_pose"].astype(np.float64)
        with np.load(branch_path(record["record_id"])) as truth:
            truth_pose = pose_target(truth["endpoint_states"].astype(np.float64))
            contacts = truth["interaction_counts"].astype(np.int64)
        masks = pair_contact_masks(contacts)
        predicted_contrasts = commutator_contrasts(predicted_pose)
        truth_contrasts = commutator_contrasts(truth_pose)
        for scope, mask in [("all", None), ("contact", masks["contact"]), ("free", masks["free"])]:
            metrics = commutator_alignment_metrics(predicted_pose, truth_pose, mask)
            record_rows.append({
                "record_id": int(record["record_id"]),
                "trajectory_id": int(record["trajectory_id"]),
                "scope": scope,
                **metrics,
            })
        for pair_index, label in enumerate(PAIR_DESIGN_LABELS):
            regime = (
                "both_contact" if masks["both_contact"][pair_index]
                else "one_contact" if masks["one_contact"][pair_index]
                else "free"
            )
            row = {
                "record_id": int(record["record_id"]),
                "trajectory_id": int(record["trajectory_id"]),
                "pair_index": int(pair_index),
                "pair_design": label,
                "contact_regime": regime,
                "predicted_norm": float(np.linalg.norm(predicted_contrasts[pair_index])),
                "truth_norm": float(np.linalg.norm(truth_contrasts[pair_index])),
            }
            for component in range(truth_contrasts.shape[1]):
                row[f"predicted_{component}"] = float(predicted_contrasts[pair_index, component])
                row[f"truth_{component}"] = float(truth_contrasts[pair_index, component])
            pair_rows.append(row)
    write_csv(EVIDENCE_DIR / "model_physical_commutator_pair_rows.csv", pair_rows)
    write_csv(EVIDENCE_DIR / "model_physical_commutator_record_rows.csv", record_rows)
    return pair_rows, record_rows


EVALUATION_OPENED = False
if not PIPELINE_FAILED:
    try:
        if not STAGE18_ARTIFACT_VALIDATED or not STAGE19_UPSTREAM_BOUND:
            raise RuntimeError("frozen Stage 18 and Stage 19 evidence must be bound first")
        MODEL, PREPROCESSOR, PREDICTOR, PREDICTOR_BLOCK_MODULES = load_frozen_model()
        if len(PREDICTOR_BLOCK_MODULES) != 6:
            raise RuntimeError("predictor block count changed")
        for module in PREDICTOR_BLOCK_MODULES:
            if not isinstance(module, torch.nn.Module) or getattr(module, "register_forward_hook", None) is None:
                raise RuntimeError("predictor block does not support forward hooks")
        TRAIN_OUTPUT_PROJECTOR = CountSketchProjector(
            256 * 384, OUTPUT_SKETCH_DIM, TRAIN_OUTPUT_SKETCH_SEED
        )
        EVAL_OUTPUT_PROJECTOR = CountSketchProjector(
            256 * 384, OUTPUT_SKETCH_DIM, EVAL_OUTPUT_SKETCH_SEED
        )
        DECODE_PHYSICAL_POSE = physical_pose_decoder()
        first_record_id = ALL_EVALUATION_RECORDS[0]["record_id"]
        HOOK_IDENTITY = hook_identity_test(first_record_id)
        FORWARD_BENCHMARK = forward_benchmark(first_record_id)
        extract_baselines(ALL_EVALUATION_RECORDS, [FIXED_BLOCK])
        MODEL_PHYSICAL_PAIR_ROWS, MODEL_PHYSICAL_RECORD_ROWS = (
            evaluate_model_physical_commutators(ALL_EVALUATION_RECORDS)
        )
        EVALUATION_OPENED = True
        write_json(
            OUT / "evaluation_open_certificate.json",
            {
                "opened": True,
                "source_identity": SOURCE_IDENTITY,
                "stage18_artifact_certificate_sha256": sha256_file(
                    OUT / "stage18_artifact_certificate.json"
                ),
                "stage19_upstream_certificate_sha256": sha256_file(
                    OUT / "stage19_upstream_certificate.json"
                ),
                "physical_selection_freeze_sha256": sha256_file(
                    DESIGN_DIR / "physical_selection_freeze.json"
                ),
                "evaluation_records": len(ALL_EVALUATION_RECORDS),
                "fit_or_selection_model_activations": [],
                "stage27_subspace_refit": False,
            },
        )
        memory_report("stage27_model_baselines_complete")
    except Exception:
        record_failure("stage27_model_baselines")
'''


causal_interchange = r'''# Intervene on only the within-pair order component of the frozen carrier.


def load_frozen_subspaces():
    if not STAGE18_ARTIFACT_VALIDATED:
        raise RuntimeError("Stage 18 artifact is not validated")
    return FROZEN_SUBSPACES


def whiten_carrier(values, subspaces):
    return transform_primal_channels(
        np.asarray(values, dtype=np.float64), subspaces["channel_inverse_square_root"]
    )


def native_edit(values, subspaces):
    return inverse_transform_primal_channels(
        np.asarray(values, dtype=np.float64), subspaces["channel_square_root"]
    )


def intervention_path(record_id):
    return INTERVENTION_DIR / f"state_{int(record_id):06d}.json"


def finite_json_rows(rows):
    return [
        {
            key: None if isinstance(value, (float, np.floating)) and not np.isfinite(value) else value
            for key, value in row.items()
        }
        for row in rows
    ]


def wrong_state_order_delta(wrong_white, basis):
    wrong_order = paired_antisymmetric_component(wrong_white).reshape(ACTIONS_PER_STATE, -1)
    projected = (wrong_order @ basis) @ basis.T
    return (-2.0 * projected).reshape(wrong_white.shape)


def intervention_specs(record, carrier, subspaces):
    record_id = int(record["record_id"])
    white = whiten_carrier(carrier, subspaces)
    primary_basis = subspaces["primary_basis"][:, :PRIMARY_RANK]
    primary_swap = paired_swap_delta(white, basis=primary_basis, dose=1.0)
    primary_ablation = paired_ablation_delta(white, primary_basis, dose=1.0)
    full_swap = paired_swap_delta(white, basis=None, dose=1.0)
    if min(np.linalg.norm(primary_swap), np.linalg.norm(primary_ablation)) <= 1e-12:
        raise RuntimeError("primary order intervention is degenerate")
    specs = []

    def add(condition, family, mode, rank, dose, delta):
        specs.append({
            "condition": condition,
            "family": family,
            "mode": mode,
            "rank": int(rank),
            "dose": float(dose),
            "delta_white": np.asarray(delta, dtype=np.float64),
        })

    for dose in ACTIVE_CAUSAL_DOSES:
        add(
            f"primary_r{PRIMARY_RANK:03d}", "primary", "sufficiency",
            PRIMARY_RANK, dose, float(dose) * primary_swap,
        )
    for rank in ACTIVE_SENSITIVITY_RANKS:
        learned = paired_swap_delta(
            white, basis=subspaces["primary_basis"][:, :rank], dose=1.0
        )
        if rank != PRIMARY_RANK:
            add(f"learned_r{rank:03d}", "rank_sensitivity", "sufficiency", rank, 1.0, learned)
        shuffled = paired_swap_delta(
            white, basis=subspaces["shuffled_basis"][:, :rank], dose=1.0
        )
        add(
            f"shuffled_r{rank:03d}", "matched_shuffled_control", "sufficiency",
            rank, 1.0, norm_match(shuffled, learned),
        )
        for draw in range(ACTIVE_CAUSAL_RANDOM_DRAWS):
            random_delta = paired_swap_delta(
                white, basis=subspaces[f"random_basis_{draw:02d}"][:, :rank], dose=1.0
            )
            add(
                f"random_r{rank:03d}_{draw:02d}", "empirical_span_random_control",
                "sufficiency", rank, 1.0, norm_match(random_delta, learned),
            )

    wrong_id = int(SELECTION_CERTIFICATE["wrong_state_map"][str(record_id)])
    wrong_carrier = carrier_for_block(load_baseline(wrong_id), FIXED_BLOCK)
    wrong_delta = wrong_state_order_delta(
        whiten_carrier(wrong_carrier, subspaces), primary_basis
    )
    add(
        f"wrong_state_r{PRIMARY_RANK:03d}", "state_specificity_control", "sufficiency",
        PRIMARY_RANK, 1.0, norm_match(wrong_delta, primary_swap),
    )
    add(
        f"common_mode_r{PRIMARY_RANK:03d}", "matched_common_mode_control", "sufficiency",
        PRIMARY_RANK, 1.0, matched_common_mode(primary_swap, primary_basis[:, 0]),
    )
    add("full_activation_swap", "positive_control_only", "sufficiency", -1, 1.0, full_swap)

    for rank in ACTIVE_SENSITIVITY_RANKS:
        learned_basis = subspaces["primary_basis"][:, :rank]
        learned = paired_ablation_delta(white, learned_basis, dose=1.0)
        learned_name = (
            f"ablate_primary_r{rank:03d}" if rank == PRIMARY_RANK
            else f"ablate_learned_r{rank:03d}"
        )
        add(
            learned_name, "primary" if rank == PRIMARY_RANK else "rank_sensitivity",
            "necessity", rank, 1.0, learned,
        )
        shuffled = paired_ablation_delta(
            white, subspaces["shuffled_basis"][:, :rank], dose=1.0
        )
        add(
            f"ablate_shuffled_r{rank:03d}", "matched_shuffled_control", "necessity",
            rank, 1.0, norm_match(shuffled, learned),
        )
        for draw in range(ACTIVE_CAUSAL_RANDOM_DRAWS):
            random_delta = paired_ablation_delta(
                white, subspaces[f"random_basis_{draw:02d}"][:, :rank], dose=1.0
            )
            add(
                f"ablate_random_r{rank:03d}_{draw:02d}",
                "empirical_span_random_control", "necessity", rank, 1.0,
                norm_match(random_delta, learned),
            )
    if RUN_MODE == "pilot" and len(specs) != INTERVENTION_FORWARDS_PER_RECORD:
        raise RuntimeError(
            f"expected {INTERVENTION_FORWARDS_PER_RECORD} interventions, found {len(specs)}"
        )
    for specification in specs:
        specification["edit_norm"] = float(np.linalg.norm(specification["delta_white"]))
        specification["primary_swap_norm"] = float(np.linalg.norm(primary_swap))
        specification["primary_ablation_norm"] = float(np.linalg.norm(primary_ablation))
        specification["full_swap_norm"] = float(np.linalg.norm(full_swap))
    return specs


def scoped_metrics(baseline_output, patched_output, baseline_pose, patched_pose, masks):
    row = {}
    for scope, mask in [("all", None), ("contact", masks["contact"]), ("free", masks["free"])]:
        output = paired_transfer_metrics(baseline_output, patched_output, mask)
        pose = paired_transfer_metrics(baseline_pose, patched_pose, mask)
        energy = paired_energy_metrics(baseline_output, patched_output, mask)
        for name, value in output.items():
            row[f"{scope}_output_{name}"] = value
        for name, value in pose.items():
            row[f"{scope}_pose_{name}"] = value
        for name, value in energy.items():
            row[f"{scope}_output_order_{name}"] = value
    return row


def make_result_row(record, condition, family, mode, rank, dose, masks,
                    baseline_output, patched_output, baseline_pose, patched_pose,
                    edit_norm, primary_swap_norm, primary_ablation_norm, full_swap_norm):
    return {
        "record_id": int(record["record_id"]),
        "trajectory_id": int(record["trajectory_id"]),
        "task_id": int(record["task_id"]),
        "selected_block": FIXED_BLOCK,
        "condition": condition,
        "family": family,
        "mode": mode,
        "rank": int(rank),
        "dose": float(dose),
        "contact_pairs": int(np.sum(masks["contact"])),
        "free_pairs": int(np.sum(masks["free"])),
        "output_rms_change": float(np.sqrt(np.mean((patched_output - baseline_output) ** 2))),
        "carrier_edit_whitened_norm": float(edit_norm),
        "primary_swap_norm": float(primary_swap_norm),
        "primary_ablation_norm": float(primary_ablation_norm),
        "full_swap_norm": float(full_swap_norm),
        "edit_to_full_swap_ratio": float(edit_norm) / max(float(full_swap_norm), 1e-12),
        **scoped_metrics(
            baseline_output, patched_output, baseline_pose, patched_pose, masks
        ),
    }


def run_record_interventions(record, subspaces):
    destination = intervention_path(record["record_id"])
    if destination.exists():
        PROVENANCE_COUNTS["cache_hits"] += 1
        raise RuntimeError(f"fresh-run intervention shard already exists: {destination}")
    payload = load_baseline(record["record_id"])
    carrier = carrier_for_block(payload, FIXED_BLOCK)
    baseline_output = payload["output_eval_sketch"].astype(np.float64)
    baseline_pose = payload["decoded_pose"].astype(np.float64)
    with np.load(branch_path(record["record_id"])) as truth:
        masks = pair_contact_masks(truth["interaction_counts"].astype(np.int64))
    specifications = intervention_specs(record, carrier, subspaces)
    rows = [make_result_row(
        record, "no_edit", "baseline", "baseline", 0, 0.0, masks,
        baseline_output, baseline_output, baseline_pose, baseline_pose,
        0.0, 0.0, 0.0, 0.0,
    )]
    initial, actions = state_model_inputs(record["record_id"])
    for specification in specifications:
        delta_native = native_edit(specification["delta_white"], subspaces)
        delta_tensor = torch.as_tensor(delta_native, device="cuda", dtype=torch.float32)
        with torch.inference_mode():
            patched, _, _ = forward_with_carriers(
                initial,
                actions,
                PRIMARY_HORIZON,
                capture_blocks=[FIXED_BLOCK],
                intervention={"block": FIXED_BLOCK, "delta": delta_tensor},
            )
            patched_output = EVAL_OUTPUT_PROJECTOR(patched).cpu().numpy()
            patched_pose = DECODE_PHYSICAL_POSE(patched).cpu().numpy()
        rows.append(make_result_row(
            record,
            specification["condition"],
            specification["family"],
            specification["mode"],
            specification["rank"],
            specification["dose"],
            masks,
            baseline_output,
            patched_output,
            baseline_pose,
            patched_pose,
            specification["edit_norm"],
            specification["primary_swap_norm"],
            specification["primary_ablation_norm"],
            specification["full_swap_norm"],
        ))
        del patched, patched_output, patched_pose, delta_tensor
    write_json(destination, finite_json_rows(rows))
    PROVENANCE_COUNTS["intervention_generated"] += 1
    del initial, actions
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def run_all_interventions(records):
    started = time.perf_counter()
    subspaces = load_frozen_subspaces()
    rows = []
    for index, record in enumerate(records):
        rows.extend(run_record_interventions(record, subspaces))
        write_json(
            OUT / "intervention_progress.json",
            {"completed": index + 1, "total": len(records), "last_record_id": int(record["record_id"])},
        )
    TIMINGS["causal_intervention_seconds"] = time.perf_counter() - started
    write_csv(EVIDENCE_DIR / "commutator_intervention_rows.csv", rows)
    return rows


if not PIPELINE_FAILED and EVALUATION_OPENED:
    try:
        INTERVENTION_ROWS = run_all_interventions(ALL_EVALUATION_RECORDS)
        memory_report("stage27_causal_interventions_complete")
    except Exception:
        record_failure("stage27_causal_interchange")
'''


decision_and_plots = r'''# Apply the frozen physical, predictive, and causal commutator gates.


def lookup(rows, trajectory_id, condition, dose=1.0, key="contact_output_coefficient"):
    values = [
        row[key]
        for row in rows
        if int(row["trajectory_id"]) == int(trajectory_id)
        and row["condition"] == condition
        and np.isclose(float(row["dose"]), float(dose))
    ]
    return float(values[0]) if len(values) == 1 else math.nan


def random_median(rows, trajectory_id, rank, key, ablate=False):
    prefix = "ablate_random" if ablate else "random"
    values = [
        lookup(rows, trajectory_id, f"{prefix}_r{rank:03d}_{draw:02d}", key=key)
        for draw in range(ACTIVE_CAUSAL_RANDOM_DRAWS)
    ]
    return float(np.nanmedian(values))


def bootstrap_interval(values, trajectories, label):
    draws = clustered_bootstrap_mean(
        np.asarray(values, dtype=np.float64),
        np.asarray(trajectories),
        ACTIVE_BOOTSTRAP_DRAWS,
        stable_seed(BOOTSTRAP_SEED, label) % (2**31 - 1),
    )
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def physical_commutator_gate():
    selected_ids = {int(record["record_id"]) for record in ALL_EVALUATION_RECORDS}
    selected = [row for row in PHYSICAL_PAIR_ROWS if int(row["record_id"]) in selected_ids]
    contact = [row for row in selected if row["contact_regime"] != "free"]
    free = [row for row in selected if row["contact_regime"] == "free"]
    contact_norms = np.asarray([row["physical_commutator_norm"] for row in contact])
    free_norms = np.asarray([row["physical_commutator_norm"] for row in free])
    contact_median = float(np.median(contact_norms)) if len(contact_norms) else 0.0
    free_median = float(np.median(free_norms)) if len(free_norms) else 0.0
    ratio = contact_median / max(free_median, 1e-12)
    passed = bool(
        len(contact) >= ACTIVE_MIN_VALID_CONTACT_PAIRS
        and len(free) >= len(ALL_EVALUATION_RECORDS)
        and contact_median >= MIN_ELIGIBLE_CONTACT_COMMUTATOR_NORM
        and ratio >= MIN_CONTACT_TO_FREE_TRUE_NORM_RATIO
        and all(row["equal_integrated_impulse"] for row in selected)
        and all(row["equal_control_energy"] for row in selected)
    )
    return {
        "selected_records": len(selected_ids),
        "contact_pairs": len(contact),
        "free_pairs": len(free),
        "median_contact_commutator_norm": contact_median,
        "median_free_commutator_norm": free_median,
        "contact_to_free_norm_ratio": ratio,
        "same_pulses_impulse_energy_duration": bool(
            all(row["equal_integrated_impulse"] for row in selected)
            and all(row["equal_control_energy"] for row in selected)
        ),
        "passed": passed,
    }


def model_commutator_gate():
    contact = [
        row for row in MODEL_PHYSICAL_PAIR_ROWS if row["contact_regime"] != "free"
    ]
    if not contact:
        raise RuntimeError("model commutator evidence contains no contact pairs")
    dimensions = sorted(
        int(key.split("_")[-1])
        for key in contact[0]
        if key.startswith("truth_")
    )
    predicted = np.asarray([
        [row[f"predicted_{value}"] for value in dimensions] for row in contact
    ])
    truth = np.asarray([
        [row[f"truth_{value}"] for value in dimensions] for row in contact
    ])
    dot = float(np.sum(predicted * truth))
    true_energy = float(np.sum(truth**2))
    predicted_energy = float(np.sum(predicted**2))
    aggregate_coefficient = dot / max(true_energy, 1e-12)
    aggregate_cosine = dot / max(math.sqrt(true_energy * predicted_energy), 1e-12)

    truth_lookup = {
        (int(row["record_id"]), int(row["pair_index"])):
        np.asarray([row[f"truth_{value}"] for value in dimensions])
        for row in MODEL_PHYSICAL_PAIR_ROWS
    }
    wrong_truth = np.stack([
        truth_lookup[
            (
                int(SELECTION_CERTIFICATE["wrong_state_map"][str(int(row["record_id"]))]),
                int(row["pair_index"]),
            )
        ]
        for row in contact
    ])
    wrong_dot = float(np.sum(predicted * wrong_truth))
    wrong_cosine = wrong_dot / max(
        math.sqrt(float(np.sum(predicted**2)) * float(np.sum(wrong_truth**2))), 1e-12
    )
    record_rows = [
        row for row in MODEL_PHYSICAL_RECORD_ROWS if row["scope"] == "contact"
    ]
    record_cosines = np.asarray([row["cosine"] for row in record_rows])
    trajectories = np.asarray([row["trajectory_id"] for row in record_rows])
    cosine_ci = bootstrap_interval(record_cosines, trajectories, "model_contact_cosine")
    passed = bool(
        np.all(np.isfinite(record_cosines))
        and aggregate_cosine >= MIN_MODEL_CONTACT_COMMUTATOR_COSINE
        and aggregate_cosine - wrong_cosine >= MIN_MODEL_ALIGNMENT_GAIN_OVER_SHUFFLED
        and (cosine_ci[0] > 0 if RUN_MODE == "pilot" else True)
    )
    return {
        "contact_pairs": len(contact),
        "aggregate_coefficient": float(aggregate_coefficient),
        "aggregate_cosine": float(aggregate_cosine),
        "wrong_state_truth_cosine": float(wrong_cosine),
        "alignment_gain_over_wrong_state": float(aggregate_cosine - wrong_cosine),
        "mean_record_cosine": float(np.mean(record_cosines)),
        "record_cosine_ci95": cosine_ci,
        "passed": passed,
    }


def causal_commutator_gate(rows):
    trajectories = [int(record["trajectory_id"]) for record in ALL_EVALUATION_RECORDS]
    primary_name = f"primary_r{PRIMARY_RANK:03d}"
    primary = np.asarray([
        lookup(rows, value, primary_name, key="contact_output_coefficient")
        for value in trajectories
    ])
    primary_cosine = np.asarray([
        lookup(rows, value, primary_name, key="contact_output_cosine")
        for value in trajectories
    ])
    full = np.asarray([
        lookup(rows, value, "full_activation_swap", key="contact_output_coefficient")
        for value in trajectories
    ])
    shuffled = np.asarray([
        lookup(rows, value, f"shuffled_r{PRIMARY_RANK:03d}", key="contact_output_coefficient")
        for value in trajectories
    ])
    random_values = np.asarray([
        random_median(rows, value, PRIMARY_RANK, "contact_output_coefficient")
        for value in trajectories
    ])
    wrong = np.asarray([
        lookup(rows, value, f"wrong_state_r{PRIMARY_RANK:03d}", key="contact_output_coefficient")
        for value in trajectories
    ])
    free_primary = np.asarray([
        lookup(rows, value, primary_name, key="free_output_coefficient")
        for value in trajectories
    ])
    gain_random = primary - random_values
    gain_shuffled = primary - shuffled
    gain_ci = bootstrap_interval(gain_random, trajectories, "causal_sufficiency")
    gain_sign = exact_positive_sign_test(gain_random)

    positive_doses = sorted(value for value in ACTIVE_CAUSAL_DOSES if value > 0)
    dose_slopes = []
    for trajectory_id in trajectories:
        if len(positive_doses) < 2:
            dose_slopes.append(math.nan)
            continue
        values = np.asarray([
            lookup(rows, trajectory_id, primary_name, dose=value,
                   key="contact_output_coefficient")
            for value in positive_doses
        ])
        dose_slopes.append(float(np.polyfit(positive_doses, values, 1)[0]))
    negative = (
        np.asarray([
            lookup(rows, value, primary_name, dose=-0.5,
                   key="contact_output_coefficient")
            for value in trajectories
        ])
        if -0.5 in ACTIVE_CAUSAL_DOSES else np.full(len(trajectories), np.nan)
    )

    necessity_name = f"ablate_primary_r{PRIMARY_RANK:03d}"
    necessity_key = "contact_output_order_energy_reduction"
    necessity = np.asarray([
        lookup(rows, value, necessity_name, key=necessity_key) for value in trajectories
    ])
    necessity_shuffled = np.asarray([
        lookup(rows, value, f"ablate_shuffled_r{PRIMARY_RANK:03d}", key=necessity_key)
        for value in trajectories
    ])
    necessity_random = np.asarray([
        random_median(rows, value, PRIMARY_RANK, necessity_key, ablate=True)
        for value in trajectories
    ])
    necessity_gain_random = necessity - necessity_random
    necessity_gain_shuffled = necessity - necessity_shuffled
    necessity_ci = bootstrap_interval(
        necessity_gain_random, trajectories, "causal_necessity"
    )
    necessity_sign = exact_positive_sign_test(necessity_gain_random)
    finite = bool(all(np.all(np.isfinite(value)) for value in [
        primary, primary_cosine, full, shuffled, random_values, wrong, free_primary,
        necessity, necessity_shuffled, necessity_random,
    ]))
    sufficiency_pass = bool(
        finite
        and np.mean(full) >= MIN_FULL_SWAP_COEFFICIENT
        and np.mean(primary) >= MIN_PRIMARY_CONTACT_COEFFICIENT
        and np.mean(primary_cosine) >= MIN_PRIMARY_CONTACT_COSINE
        and np.mean(gain_random) >= MIN_PRIMARY_GAIN_OVER_RANDOM
        and np.mean(gain_shuffled) >= MIN_PRIMARY_GAIN_OVER_SHUFFLED
        and gain_sign["positive"] >= min(REQUIRED_POSITIVE_EVALUATION_TRAJECTORIES, len(trajectories))
        and (gain_sign["p_value"] <= 0.05 if RUN_MODE == "pilot" else True)
        and (gain_ci[0] > 0 if RUN_MODE == "pilot" else True)
        and (
            RUN_MODE == "smoke" or (
                np.sum(np.asarray(dose_slopes) > 0) >= REQUIRED_POSITIVE_EVALUATION_TRAJECTORIES
                and np.mean(negative) < 0
            )
        )
    )
    necessity_pass = bool(
        finite
        and np.mean(necessity) >= MIN_NECESSITY_REDUCTION
        and np.mean(necessity_gain_random) >= MIN_NECESSITY_GAIN_OVER_RANDOM
        and np.mean(necessity_gain_shuffled) >= MIN_NECESSITY_GAIN_OVER_SHUFFLED
        and necessity_sign["positive"] >= min(REQUIRED_POSITIVE_NECESSITY_TRAJECTORIES, len(trajectories))
        and (necessity_sign["p_value"] <= 0.05 if RUN_MODE == "pilot" else True)
        and (necessity_ci[0] > 0 if RUN_MODE == "pilot" else True)
    )
    return {
        "trajectories": len(trajectories),
        "all_required_metrics_finite": finite,
        "mean_primary_contact_coefficient": float(np.mean(primary)),
        "mean_primary_contact_cosine": float(np.mean(primary_cosine)),
        "mean_primary_free_coefficient": float(np.mean(free_primary)),
        "mean_full_swap_contact_coefficient": float(np.mean(full)),
        "mean_random_contact_coefficient": float(np.mean(random_values)),
        "mean_shuffled_contact_coefficient": float(np.mean(shuffled)),
        "mean_wrong_state_contact_coefficient": float(np.mean(wrong)),
        "mean_gain_over_random": float(np.mean(gain_random)),
        "mean_gain_over_shuffled": float(np.mean(gain_shuffled)),
        "gain_over_random_ci95": gain_ci,
        "gain_over_random_sign_test": gain_sign,
        "positive_dose_slope_trajectories": int(np.sum(np.asarray(dose_slopes) > 0)),
        "negative_dose_mean": float(np.nanmean(negative)) if np.any(np.isfinite(negative)) else None,
        "mean_necessity_reduction": float(np.mean(necessity)),
        "mean_necessity_random_reduction": float(np.mean(necessity_random)),
        "mean_necessity_shuffled_reduction": float(np.mean(necessity_shuffled)),
        "mean_necessity_gain_over_random": float(np.mean(necessity_gain_random)),
        "mean_necessity_gain_over_shuffled": float(np.mean(necessity_gain_shuffled)),
        "necessity_gain_over_random_ci95": necessity_ci,
        "necessity_gain_over_random_sign_test": necessity_sign,
        "sufficiency_gate_pass": sufficiency_pass,
        "necessity_gate_pass": necessity_pass,
        "passed": bool(sufficiency_pass and necessity_pass),
    }


def rank_sensitivity(rows):
    trajectories = [int(record["trajectory_id"]) for record in ALL_EVALUATION_RECORDS]
    results = []
    for rank in ACTIVE_SENSITIVITY_RANKS:
        name = f"primary_r{rank:03d}" if rank == PRIMARY_RANK else f"learned_r{rank:03d}"
        ablate = (
            f"ablate_primary_r{rank:03d}" if rank == PRIMARY_RANK
            else f"ablate_learned_r{rank:03d}"
        )
        results.append({
            "rank": int(rank),
            "mean_contact_coefficient": float(np.mean([
                lookup(rows, value, name, key="contact_output_coefficient")
                for value in trajectories
            ])),
            "mean_contact_necessity_reduction": float(np.mean([
                lookup(rows, value, ablate, key="contact_output_order_energy_reduction")
                for value in trajectories
            ])),
        })
    write_csv(ANALYSIS_DIR / "rank_sensitivity.csv", results)
    return results


def fresh_run_certificate():
    expected = {
        "truth_generated": len(POOL_SPECS),
        "baseline_generated": len(ALL_EVALUATION_RECORDS),
        "intervention_generated": len(ALL_EVALUATION_RECORDS),
        "cache_hits": 0,
    }
    passed = bool(not OUT_PREEXISTED and PROVENANCE_COUNTS == expected)
    payload = {
        "out_preexisted": bool(OUT_PREEXISTED),
        "fresh_run_required": bool(FRESH_RUN_REQUIRED),
        "observed_counts": dict(PROVENANCE_COUNTS),
        "expected_counts": expected,
        "passed": passed,
    }
    write_json(OUT / "fresh_run_certificate.json", payload)
    return payload


def make_plots(physical, model, causal, ranks):
    figure, axes = plt.subplots(1, 4, figsize=(20, 4.6))
    axes[0].bar(
        ["contact", "free"],
        [physical["median_contact_commutator_norm"], physical["median_free_commutator_norm"]],
    )
    axes[0].set(title="Exact physical order effect", ylabel="median finite-commutator norm")
    axes[1].bar(
        ["matched", "wrong state"],
        [model["aggregate_cosine"], model["wrong_state_truth_cosine"]],
    )
    axes[1].axhline(MIN_MODEL_CONTACT_COMMUTATOR_COSINE, color="black", linestyle="--")
    axes[1].set(title="Model–physics alignment", ylabel="commutator cosine")
    axes[2].bar(
        ["learned", "random", "shuffled"],
        [causal["mean_primary_contact_coefficient"], causal["mean_random_contact_coefficient"],
         causal["mean_shuffled_contact_coefficient"]],
    )
    axes[2].set(title="Causal sufficiency", ylabel="opposite-order coefficient")
    axes[3].bar(
        [f"r{row['rank']}" for row in ranks],
        [row["mean_contact_necessity_reduction"] for row in ranks],
    )
    axes[3].set(title="Pair-specific necessity", ylabel="order-energy reduction")
    figure.tight_layout()
    figure.savefig(PLOT_DIR / "stage27_causal_commutator_summary.png", dpi=180)
    plt.close(figure)


if PIPELINE_FAILED:
    DECISION_PAYLOAD = {"status": "INCONCLUSIVE", "failure": FAILURE_MESSAGE}
elif not EVALUATION_OPENED:
    DECISION_PAYLOAD = {"status": "INCONCLUSIVE", "reason": "model evidence was not opened"}
else:
    try:
        PHYSICAL_GATE = physical_commutator_gate()
        MODEL_GATE = model_commutator_gate()
        CAUSAL_GATE = causal_commutator_gate(INTERVENTION_ROWS)
        RANK_SENSITIVITY = rank_sensitivity(INTERVENTION_ROWS)
        FRESH_CERTIFICATE = fresh_run_certificate()
        if RUN_MODE == "smoke":
            candidate_status = "SMOKE_ONLY"
        elif not PHYSICAL_GATE["passed"]:
            candidate_status = "NO_CONTACT_AMPLIFIED_PHYSICAL_ORDER_EFFECT"
        elif not MODEL_GATE["passed"]:
            candidate_status = "MODEL_DOES_NOT_CAPTURE_PHYSICAL_ACTION_COMMUTATOR"
        elif not CAUSAL_GATE["passed"]:
            candidate_status = "ORDER_EFFECT_NOT_MEDIATED_BY_FROZEN_ACTION_CARRIER"
        else:
            candidate_status = "CAUSAL_NONCOMMUTATIVE_ACTION_DYNAMICS_SUPPORTED"
        confirmation_eligible = bool(
            SOURCE_IDENTITY.get("confirmation_eligible", False)
            and STAGE18_ARTIFACT_VALIDATED
            and STAGE19_UPSTREAM_BOUND
            and FRESH_CERTIFICATE["passed"]
        )
        status = (
            candidate_status
            if RUN_MODE == "smoke" or confirmation_eligible
            else "UNBOUND_NONFRESH_OR_WRONG_UPSTREAM_EXPLORATORY_RESULT"
        )
        DECISION_PAYLOAD = {
            "status": status,
            "candidate_status": candidate_status,
            "confirmation_eligible": confirmation_eligible,
            "source_bound_claim_eligible": bool(SOURCE_IDENTITY.get("confirmation_eligible", False)),
            "stage18_artifact_claim_eligible": bool(STAGE18_ARTIFACT_VALIDATED),
            "stage19_upstream_claim_eligible": bool(STAGE19_UPSTREAM_BOUND),
            "fresh_run_claim_eligible": FRESH_CERTIFICATE["passed"],
            "physical_commutator_gate": PHYSICAL_GATE,
            "model_physical_commutator_gate": MODEL_GATE,
            "causal_carrier_commutator_gate": CAUSAL_GATE,
            "rank_sensitivity": RANK_SENSITIVITY,
            "claim_boundary": {
                "finite_action_commutator_only": True,
                "infinitesimal_lie_bracket_established": False,
                "same_pulses_impulse_energy_and_duration_within_pair": True,
                "stage18_subspace_refit_or_tuning": False,
                "coordinate_reader_used": False,
                "jacobian_jvp_vjp_or_gradient_used": False,
                "one_model_checkpoint": True,
                "one_environment": True,
                "generalization_to_other_models_or_environments": False,
                "causal_claim": (
                    "the frozen block-4 action carrier mediates a contact-amplified finite "
                    "order effect only if all three frozen gates pass"
                ),
            },
            "prespecified_next_step_if_positive": (
                "replicate the finite commutator and carrier mediation across checkpoints "
                "and a second contact-rich environment"
            ),
        }
        write_json(OUT / "stage27_decision.json", DECISION_PAYLOAD)
        make_plots(PHYSICAL_GATE, MODEL_GATE, CAUSAL_GATE, RANK_SENSITIVITY)
        print(json.dumps(DECISION_PAYLOAD, indent=2))
    except Exception:
        record_failure("stage27_decision_and_plots")
        DECISION_PAYLOAD = {"status": "INCONCLUSIVE", "failure": FAILURE_MESSAGE}

if not (OUT / "stage27_decision.json").exists():
    write_json(OUT / "stage27_decision.json", DECISION_PAYLOAD)
'''


packaging = base_source(12)
packaging = packaging.replace(
    "stage19_unseen_action_transfer_result_bundle_",
    "stage27_action_commutator_result_bundle_",
)


protocol_sources = [
    introduction,
    configuration,
    installation,
    setup,
    analysis_helpers,
    model_helpers,
    design,
    truth_generation,
    artifact_import,
    model_and_baselines,
    causal_interchange,
    decision_and_plots,
    packaging,
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
    code(design),
    code(truth_generation),
    code(artifact_import),
    code(model_and_baselines),
    code(causal_interchange),
    code(decision_and_plots),
    code(packaging),
]
for index, cell in enumerate(cells):
    cell["id"] = f"stage27-{index:02d}"

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
