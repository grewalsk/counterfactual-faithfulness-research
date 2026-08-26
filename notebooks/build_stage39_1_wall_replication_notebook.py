"""Build the independent Wall-environment coefficient-matched Colab."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "39_1_wall_cross_environment_replication.ipynb"


def load_builder(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BASE = load_builder(
    ROOT / "build_stage39_fresh_coefficient_replication_notebook.py",
    "stage39_builder_for_wall",
)

code = BASE.code
markdown = BASE.markdown
replace_assignment = BASE.replace_assignment
replace_block = BASE.replace_block
assigned_uppercase_names = BASE.assigned_uppercase_names


def rename(value: str) -> str:
    for old, new in [
        ("Stage 39", "Stage 39.1"),
        ("STAGE39", "STAGE391"),
        ("stage39", "stage391"),
    ]:
        value = value.replace(old, new)
    return value


def replace_function(source: str, name: str, replacement: str) -> str:
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name
    )
    lines = source.splitlines(keepends=True)
    start = min([node.lineno] + [item.lineno for item in node.decorator_list]) - 1
    end = node.end_lineno
    return "".join(lines[:start]) + replacement.strip() + "\n\n\n" + "".join(lines[end:])


def replace_top_level_assignment(source: str, name: str, value: str) -> str:
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in (
                item.targets if isinstance(item, ast.Assign) else [item.target]
            )
        )
    )
    lines = source.splitlines(keepends=True)
    return (
        "".join(lines[: node.lineno - 1])
        + f"{name} = {value}\n"
        + "".join(lines[node.end_lineno :])
    )


introduction = r'''# Stage 39.1: independent Wall-environment replication

## Frozen decision before computation

This notebook transports the Stage 39 coefficient-matched comparison from
PushT to the official Wall world-model checkpoints and a new, simulator-built
trajectory bank.  Wall is a distinct controlled physical environment: a point
moves through or collides with a wall containing a door.  No PushT trajectory,
carrier, fitted adapter, calibration outcome, threshold, or evaluation row is
read.  The JEPA-WM and DINO-WM Wall checkpoints remain frozen.

The primary estimand, ±5% equivalence band, three matched seeds, family-level
hierarchical bootstrap, source binding, and full-versus-equal-latent-pressure
comparison are identical to Stage 39.  Wall receives its own decision.  PushT
and Wall are never pooled, and a result in either environment cannot rescue a
failure in the other.

The four prespecified Wall strata are geometric rather than retrospective:
free-far, pre-wall, wall-blocked, and doorway states.  Layout, side, state, and
action words are generated before model access.  Simulator movement clipping
provides a collision diagnostic, while the locked physical target is the
normalized two-dimensional point path.  Planning remains permanently sealed.
'''


configuration = rename(BASE.configuration)
configuration = replace_assignment(
    configuration, "NOTEBOOK_PROTOCOL_SHA256", '"__PROTOCOL_DIGEST__"'
)
for name, value in {
    "PROTOCOL_ID": '"stage39.1-wall-cross-environment-replication-v1"',
    "EVIDENCE_STATUS": '"FRESH_PROSPECTIVE_CROSS_ENVIRONMENT_REPLICATION"',
    "EXPERIMENT_NOTEBOOK_PATH": '"notebooks/39_1_wall_cross_environment_replication.ipynb"',
    "EXPERIMENT_BUILDER_PATH": '"notebooks/build_stage39_1_wall_replication_notebook.py"',
    "EXPERIMENT_NUMERICAL_PATH": '"src/cf_faithfulness/stage39_replication.py"',
    "OUTPUT_DIR": '"/content/counterfactual_faithfulness_stage39_1_wall"',
    "DRIVE_OUTPUT_DIR": '"/content/drive/MyDrive/counterfactual_faithfulness_stage39_1_wall"',
    "RUN_REQUEST_PATH": '"/content/drive/MyDrive/counterfactual_faithfulness_stage39_1_wall/stage391_run_request.json"',
    "SEED": "391101",
    "DESIGN_SEED": "391141",
    "DECODER_SEED": "391183",
    "RANK_SEED": "391213",
    "CALIBRATION_SEED": "391253",
    "BOOTSTRAP_SEED": "391283",
    "CONTROL_SEED": "391351",
    "ENVIRONMENT": '"Wall"',
    "MODEL_NAMES": '["jepa_wm_wall", "dino_wm_wall"]',
    "MODEL_SHORT_NAMES": '{"jepa_wm_wall": "jepa", "dino_wm_wall": "dino"}',
    "EXPECTED_MODEL_TYPES": '{"jepa_wm_wall": "AdaLN", "dino_wm_wall": "dino_wm"}',
    "EXPECTED_CARRIER_WIDTHS": '{"jepa_wm_wall": 400, "dino_wm_wall": 404}',
    "EXPECTED_PROPRIO_FEATURE_WIDTHS": '{"jepa_wm_wall": 16, "dino_wm_wall": 10}',
    "MODE_LABELS": '["free_far", "pre_wall", "wall_blocked", "doorway"]',
    "TRAJECTORY_GEOMETRY_VERSION": '"wall_layout_geometric_strata_v1"',
    "CONSTRUCTION_TRAJECTORY_POOL": "list(range(78000, 80000))",
    "MODEL_SELECTION_TRAJECTORY_POOL": "list(range(80000, 82000))",
    "CALIBRATION_TRAJECTORY_POOL": "list(range(82000, 84000))",
    "EVALUATION_TRAJECTORY_POOL": "list(range(84000, 90000))",
    "TASK_ID_OFFSET": "391000",
    "FINAL_TRAINING_SEEDS": '[3911, 3912, 3913] if RUN_MODE == "pilot" else [3911, 3912]',
}.items():
    configuration = replace_assignment(configuration, name, value)

configuration = replace_top_level_assignment(
    configuration, "GROUNDED_OBSERVABLES", '["dot_x", "dot_y"]'
)
configuration = replace_top_level_assignment(
    configuration,
    "STAGE391_TOKEN_SPECS",
    '{"A": (-35.0, 0.42), "B": (35.0, 0.42), "C": (-10.0, 0.34), "D": (10.0, 0.34)}',
)

configuration = replace_block(
    configuration,
    "CANONICAL_RESPONSE_WORD_NAMES = [",
    "CALIBRATION_INTERCHANGE_PAIRS =",
    r'''CANONICAL_RESPONSE_WORD_NAMES = ["A", "B", "C", "D", "AB", "CD", "BA", "DC"]
CONSTRUCTION_WORD_NAMES = [
    "BDBBADDAC", "BADBADADB", "CACBBBDAAC", "CDAACACDDD",
    "AAACDDACCDB", "DADDDDCCADC", "DDCBDDBDDCCA", "DCACBACAACBB",
]
MODEL_SELECTION_WORD_NAMES = [
    "DACBBABDA", "ADCCBDAAD", "ACCDCABBBD", "CCDADCCADA",
    "BABDBDDCDBA", "DAABACCABBC", "DBADBACABABC", "AAAAAACDDCCA",
]
CALIBRATION_WORD_NAMES = [
    "ABABDAADD", "CCACABBAD", "CACCDCBABB", "CDBDABCBBC",
    "DCDBCAACCBC", "ABBDDBBDBDB", "DADBDBDADADC", "BDBADBBAABBC",
]
STAGE391_CORE_WORD_NAMES = sorted(
    set(
        CANONICAL_RESPONSE_WORD_NAMES + CONSTRUCTION_WORD_NAMES
        + MODEL_SELECTION_WORD_NAMES + CALIBRATION_WORD_NAMES
    ),
    key=lambda value: (len(value), value),
)
CORE_WORD_SPECS = [stage391_word_spec(name) for name in STAGE391_CORE_WORD_NAMES]
''',
)
configuration = replace_block(
    configuration,
    "CLOSURE_EVALUATION_WORD_NAMES = [",
    "EVALUATION_INTERCHANGE_PAIRS =",
    r'''CLOSURE_EVALUATION_WORD_NAMES = [
    "CBBCABCAB", "DBCBACACD", "ADAABCCABD", "AAADADAADD",
    "CBCDACDCADA", "BDDACDBBDDA", "CCCBADDCCBAA", "CDACADCBBCDA",
]
PLANNING_WORD_NAMES = []
EVALUATION_WORD_NAMES_REGISTERED = list(CLOSURE_EVALUATION_WORD_NAMES)
EVALUATION_WORD_SPECS = [
    stage391_word_spec(name) for name in EVALUATION_WORD_NAMES_REGISTERED
]
''',
)
configuration = replace_top_level_assignment(
    configuration,
    "EXPECTED_PRETRAINED_ASSET_SHA256",
    '''{
    "jepa_wm_wall.pth.tar": "8efb0623cfba1cb3ca210de26f7579c83dd24936635f11989c515afcb23bea1e",
    "dino_wm_wall.pth.tar": "ff170be5aec9249768be4a220d600b8f00a8589b2a78982ecf9273809f2767df",
    "dinov2_vits14_pretrain.pth": "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9",
}''',
)
configuration = configuration.replace(
    '"official_frozen_jepa_and_dino_pusht_checkpoints",',
    '"official_frozen_jepa_and_dino_wall_checkpoints",',
).replace(
    '"fresh_trajectory_ids_66000_to_77999",',
    '"fresh_wall_trajectory_ids_78000_to_89999",',
).replace(
    '"not_cross_environment",',
    '"independent_environment_decision_no_cross_environment_pooling",',
)
configuration += r'''

WALL_IMAGE_SIZE = 65.0
WALL_DOOR_HALF_WIDTH = 4.0
WALL_MODE_NEAR_DISTANCE = 7.0
assert len(MODE_LABELS) == STATES_PER_TRAJECTORY == 4
'''
configuration = re.sub(
    r"\n\nPROTOCOL_CONFIG_KEYS = \[[^\n]*\]\n", "\n", configuration
)
configuration += "\n\nPROTOCOL_CONFIG_KEYS = " + repr(
    assigned_uppercase_names(configuration)
) + "\n"

installation = BASE.installation
setup = rename(BASE.setup).replace("stage391_fcmr", "stage391_wall")
setup = setup.replace(
    "import torch\nimport torchvision", "import torch\nimport torch.nn.functional as torch_functional\nimport torchvision"
)
analysis_helpers = rename(BASE.analysis_helpers).replace(
    "public PushT checkpoints", "public Wall checkpoints"
).replace(
    "decoded four-coordinate physical state", "decoded two-coordinate physical state"
).replace(
    "used by the Stage 33 grounded readout", "used by the Stage 39.1 grounded readout"
)
model_helpers = rename(BASE.model_helpers).replace(
    'raise RuntimeError("Stage 14 supports PushT only")',
    'raise RuntimeError("Wall execution requires a bound layout record")',
)

design_and_runtime_helpers = rename(BASE.design_and_runtime_helpers)
design_and_runtime_helpers = replace_function(
    design_and_runtime_helpers,
    "screening_policy",
    r'''
def screening_policy(record):
    direction = np.asarray([np.sign(float(record["wall_x"]) - record["state"][0]), 0.0])
    macros = [0.40 * direction] * 8 + [-0.35 * direction] * 4
    return np.concatenate([
        np.repeat(np.asarray(action, dtype=np.float32)[None], FRAMESKIP, axis=0)
        for action in macros
    ], axis=0)
''',
)
design_and_runtime_helpers = replace_function(
    design_and_runtime_helpers,
    "word_actions",
    r'''
def word_actions(record, specification):
    state = np.asarray(record["state"], dtype=np.float64)
    toward = normalized(np.asarray([float(record["wall_x"]) - state[0], 0.0]))
    macros = []
    for angle, magnitude in zip(specification["angles"], specification["magnitudes"]):
        pulse = rotate_vector(toward, angle) * float(magnitude)
        macros.append(np.repeat(pulse[None], FRAMESKIP, axis=0))
    actions = np.concatenate(macros, axis=0).astype(np.float32)
    if actions.shape != (int(specification["length"]) * FRAMESKIP, 2):
        raise RuntimeError("Wall action word shape changed")
    if np.max(np.abs(actions)) > 1.0:
        raise RuntimeError("Wall action exceeded the official action space")
    diagnostic = actions.astype(np.float64)
    impulse = np.sum(diagnostic, axis=0)
    energy = float(np.sum(diagnostic**2))
    signed_area = float(sum(
        diagnostic[i, 0] * diagnostic[j, 1] - diagnostic[i, 1] * diagnostic[j, 0]
        for i in range(len(diagnostic)) for j in range(i + 1, len(diagnostic))
    ))
    return actions, {"impulse": impulse, "energy": energy, "signed_area": signed_area}
''',
)
design_and_runtime_helpers = replace_function(
    design_and_runtime_helpers,
    "reset_dynamic_environment",
    r'''
def reset_dynamic_environment(dynamic_state, goal, seed, task=None):
    if task is None:
        raise RuntimeError("Wall restoration requires the frozen layout record")
    state = np.asarray(dynamic_state, dtype=np.float32)
    environment = make_environment(REPO, ENVIRONMENT, task)
    environment.seed(int(seed))
    environment.reset_to_state = torch.as_tensor(state, dtype=torch.float32)
    observation, restored = environment.reset()
    restored = np.asarray(restored.detach().cpu(), dtype=np.float32)
    if not np.allclose(restored, state, atol=1e-7, rtol=0):
        environment.close()
        raise RuntimeError("Wall state restoration drifted")
    environment._stage391_initial_observation = {
        "visual": wall_visual(environment),
        "proprio": np.asarray(observation["proprio"].detach().cpu(), dtype=np.float32),
    }
    return environment
''',
)
design_and_runtime_helpers = replace_function(
    design_and_runtime_helpers,
    "dynamic_state_from_environment",
    r'''
def dynamic_state_from_environment(environment):
    return np.asarray(environment.dot_position.detach().cpu(), dtype=np.float64)
''',
)
design_and_runtime_helpers = replace_function(
    design_and_runtime_helpers,
    "grounded_observables",
    r'''
def grounded_observables(state):
    value = np.asarray(state, dtype=np.float64)
    if value.shape != (2,):
        raise ValueError("Wall grounded state must have two coordinates")
    result = value / WALL_IMAGE_SIZE
    if len(result) != len(GROUNDED_OBSERVABLES):
        raise RuntimeError("Wall grounded observable schema changed")
    return result
''',
)
design_and_runtime_helpers = replace_function(
    design_and_runtime_helpers,
    "rollout_word",
    r'''
def wall_visual(environment):
    value = environment.render().float()[None]
    resized = torch_functional.interpolate(
        value, size=(224, 224), mode="bilinear", align_corners=False,
        antialias=True,
    )[0]
    return (
        torch.clamp(torch.round(resized), 0, 255).to(torch.uint8)
        .permute(1, 2, 0).cpu().numpy()
    )


def rollout_word(record, specification, retain_visual=True):
    environment = reset_dynamic_environment(
        record["state"], record["goal"], record["evaluation_seed"], task=record
    )
    actions, invariants = word_actions(record, specification)
    path_states, path_visuals, collisions = [], [], []
    initial_visual = environment._stage391_initial_observation["visual"]
    initial_proprio = environment._stage391_initial_observation["proprio"]
    previous = dynamic_state_from_environment(environment)
    try:
        for step, action in enumerate(actions, start=1):
            observation, _, _, info = environment.step(
                torch.as_tensor(action, dtype=torch.float32)
            )
            current = np.asarray(info["state"].detach().cpu(), dtype=np.float64)
            proposed = previous + 2.0 * np.asarray(action, dtype=np.float64)
            collisions.append(int(np.linalg.norm(current - proposed) > 1e-5))
            previous = current
            if step % FRAMESKIP == 0:
                path_states.append(current)
                if retain_visual:
                    path_visuals.append(wall_visual(environment))
    finally:
        environment.close()
    first_collision = next((i for i, value in enumerate(collisions) if value > 0), -1)
    return {
        "initial_visual": np.asarray(initial_visual, dtype=np.uint8),
        "initial_proprio": np.asarray(initial_proprio, dtype=np.float32),
        "path_states": np.asarray(path_states, dtype=np.float64),
        "path_visuals": np.asarray(path_visuals, dtype=np.uint8),
        "contacts": np.asarray(collisions, dtype=np.int64),
        "first_contact_step": int(first_collision),
        **invariants,
    }
''',
)
design_and_runtime_helpers = replace_function(
    design_and_runtime_helpers,
    "encoded_initial_from_record",
    r'''
def encoded_initial_from_record(bundle, record):
    environment = reset_dynamic_environment(
        record["state"], record["goal"], record["evaluation_seed"], task=record
    )
    try:
        payload = environment._stage391_initial_observation
        observation = to_model_observation(payload["visual"], payload["proprio"])
        with torch.inference_mode():
            encoded = bundle["model"].encode(observation)
        return {key: value.detach() for key, value in encoded.items()}
    finally:
        environment.close()
''',
)

physical_truth = rename(BASE.physical_truth)
design_and_runtime_helpers = replace_function(
    design_and_runtime_helpers,
    "initial_trajectory_record",
    r'''
def initial_trajectory_record(trajectory_id, split, pool):
    trajectory_id = int(trajectory_id)
    if trajectory_id not in pool:
        raise ValueError("trajectory_id lies outside its declared split pool")
    slot = (37 * trajectory_id + DESIGN_SEED) % 49
    wall_x = 29.0 + float(slot % 7)
    door_y = 17.0 + float((11 * slot) % 31)
    side = -1.0 if trajectory_id % 2 == 0 else 1.0
    y_offset = 12.0 if door_y <= 34.0 else -12.0
    return {
        "trajectory_id": trajectory_id, "split": str(split),
        "state_family_id": trajectory_id,
        "trajectory_geometry_version": TRAJECTORY_GEOMETRY_VERSION,
        "evaluation_seed": int(DESIGN_SEED + 1009 * trajectory_id),
        "task_id": int(TASK_ID_OFFSET + trajectory_id),
        "wall_x": wall_x, "door_y": door_y, "side": side,
        "blocked_y": door_y + y_offset,
        "state": np.asarray([wall_x + side * 20.0, door_y + y_offset], dtype=np.float64),
        "goal": np.asarray([wall_x - side * 20.0, door_y], dtype=np.float64),
    }
''',
)
design_and_runtime_helpers = replace_function(
    design_and_runtime_helpers,
    "trajectory_mode_snapshots",
    r'''
def trajectory_mode_snapshots(record):
    wall_x, door_y = float(record["wall_x"]), float(record["door_y"])
    side, blocked_y = float(record["side"]), float(record["blocked_y"])
    states = {
        "free_far": np.asarray([wall_x + side * 20.0, blocked_y]),
        "pre_wall": np.asarray([wall_x + side * 7.0, blocked_y]),
        "wall_blocked": np.asarray([wall_x + side * 3.5, blocked_y]),
        "doorway": np.asarray([wall_x + side * 6.0, door_y]),
    }
    rows = []
    for mode_index, label in enumerate(MODE_LABELS):
        state = states[label].astype(np.float64)
        if np.any(state < 6.0) or np.any(state > 59.0):
            raise RuntimeError("generated Wall state left the safe interior")
        rows.append({
            **{key: value for key, value in record.items() if key != "state"},
            "record_id": int(3910000 + 10 * record["trajectory_id"] + mode_index),
            "mode": label, "mode_index": int(mode_index),
            "trajectory_step": int(mode_index), "state": state,
        })
    return rows
''',
)
physical_truth = physical_truth.replace(
    "(len(words), MAX_WORD_LENGTH, 10)", "(len(words), MAX_WORD_LENGTH, 2)"
)
physical_truth = physical_truth.replace(
    '"selection_uses_contact_timing_only": True',
    '"selection_uses_geometry_only": True',
).replace(
    '"selection_uses_contact_timing_only": True,',
    '"selection_uses_geometry_only": True,',
)

simulator_preflight = rename(BASE.simulator_preflight)
construction_and_paths = rename(BASE.construction_and_paths)
construction_and_paths = replace_function(
    construction_and_paths,
    "stage391_mode_paths",
    r'''
def stage391_mode_paths(record, contact_counts, path_states, length):
    contacts = np.asarray(contact_counts, dtype=np.int64)[: int(length) * FRAMESKIP]
    states = np.asarray(path_states, dtype=np.float64)[: int(length)]
    wall_x, door_y = float(record["wall_x"]), float(record["door_y"])

    def label(state, collided):
        if bool(collided):
            return "wall_blocked"
        if abs(float(state[0]) - wall_x) <= WALL_MODE_NEAR_DISTANCE:
            if abs(float(state[1]) - door_y) <= WALL_DOOR_HALF_WIDTH + 1.0:
                return "doorway"
            return "pre_wall"
        return "free_far"

    target = []
    for step in range(int(length)):
        window = contacts[step * FRAMESKIP : (step + 1) * FRAMESKIP]
        target.append(label(states[step], np.any(window > 0)))
    source = [str(record["mode"])] + target[:-1]
    return source, target
''',
)
construction_and_paths = construction_and_paths.replace(
    'stage391_mode_paths(record, truth["contact_counts"][truth_index], length)',
    'stage391_mode_paths(\n                record, truth["contact_counts"][truth_index],\n                truth["path_states"][truth_index], length\n            )',
)
data_and_selection = rename(BASE.data_and_selection)
calibration = rename(BASE.calibration)
locked_evaluation = rename(BASE.locked_evaluation)
locked_evaluation = locked_evaluation.replace(
    "fresh PushT bank", "fresh Wall bank"
).replace(
    "universal equivalence, native JEPA", "universal equivalence, cross-environment universality, native JEPA"
)
packaging = rename(BASE.packaging).replace("stage391_fcmr", "stage391_wall")
packaging = packaging.replace(
    "fresh_coefficient_matched_replication", "wall_cross_environment_replication"
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
    raise RuntimeError("Stage 39.1 protocol digest placeholder was not replaced")
protocol_sources[1] = configuration

cells = [markdown(introduction)] + [code(value) for value in protocol_sources[1:]]
for index, cell in enumerate(cells):
    cell["id"] = f"stage391-{index:02d}"

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
