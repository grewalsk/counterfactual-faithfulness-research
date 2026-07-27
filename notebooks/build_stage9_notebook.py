import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "08_counterfactual_decision_energy.ipynb"
TARGET = ROOT / "09_counterfactual_value_equivalent_adaln.ipynb"


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


base = json.loads(SOURCE.read_text())

config = '''# SINGLE CONFIGURATION BLOCK — edit only this cell before a run.
RUN_MODE = "full"  # smoke or full
OUTPUT_DIR = "/content/counterfactual_faithfulness_stage9"
SEED = 913
MODEL_NAME = ["jepa_wm_pusht", "jepa_wm_wall"]
ENVIRONMENT = ["PushT", "Wall"]
HORIZONS = [1, 3, 6]
NUM_STATES = 24  # per environment in smoke mode
ACTIONS_PER_STATE = 10

MOUNT_DRIVE = False
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/counterfactual_faithfulness_stage9"
REUSE_STAGE7_CACHE = True
STAGE7_OUTPUT_DIR = "/content/counterfactual_faithfulness_stage7"
STAGE7_DRIVE_OUTPUT_DIR = (
    "/content/drive/MyDrive/counterfactual_faithfulness_stage7"
)
REPO_URL = "https://github.com/facebookresearch/jepa-wms.git"
REPO_COMMIT = "13cf1d9c7e476f53c17714d2e0f1dc239a883ce0"
FRAMESKIP = 5
TARGET_STEPS = list(range(1, max(HORIZONS) + 1))
TASKS_PER_ENVIRONMENT = 12
TASK_SPLIT_COUNTS = [6, 3, 0, 3]
EVALUATION_SEEDS = [913, 1297, 1709]
RIDGE_LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
BOOTSTRAP_REPS = 200
RANKING_TIE = 1e-9

# Kept compatible with the Stage 7 transition cache.
AUDIT_PROJECTION_DIM = 64
AUDIT_PROJECTION_SEEDS = [7101]
GOAL_PROJECTION_DIM = 32
GOAL_PROJECTION_SEEDS = [8101, 10101]
ENERGY_HEAD_SEEDS = [8301]
ENERGY_METHODS = []
ENERGY_HIDDEN_DIM = 96
ENERGY_DROPOUT = 0.10
ENERGY_IMPLEMENTATION_ID = "unused_stage9_compatibility"
TRAINING_BATCH_STATES = 1
TRAINING_LR = 3e-4
TRAINING_WEIGHT_DECAY = 1e-3
PAIRWISE_WEIGHT = 1.0
LISTWISE_WEIGHT = 1.0
COST_SHAPE_WEIGHT = 0.25
PAIRWISE_TEMPERATURE = 0.25
LISTWISE_TEMPERATURE = 0.20

# Stage 9: CAVE-JEPA action-path adaptation.
ADAPTATION_METHODS = [
    "latent_only_action_path",
    "shuffled_cave_action_path",
    "matched_cave_action_path",
]
ADAPTATION_SEED = 9401
ADAPTATION_EPOCHS = 2
SELECTION_EPOCHS = [1, 2]
TRAINING_EPOCHS = ADAPTATION_EPOCHS
ADAPTATION_LR = 1e-5
ADAPTATION_WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
READOUT_PROJECTION_DIM = 128
EVALUATION_PROJECTION_SEED = 11903
ACTION_GROUP_SIZE = 6
LOSS_WEIGHTS = {
    "anchor": 0.25,
    "state": 1.0,
    "effect": 2.0,
    "action": 0.25,
    "proximity": 1e-4,
}
USE_ACTIVATION_CHECKPOINTING = True
NATIVE_NONINFERIORITY_TOLERANCE = 0.02

DOWNLOAD_RESULTS = True
EVIDENCE_STATUS = "EXPLORATORY_METHOD_DEVELOPMENT"
TASK_FAMILY_ID = "stage5_tasks_reused_for_stage9_development"
DEVELOPMENT_SPLIT = "development_holdout"

if RUN_MODE == "full":
    NUM_STATES = 96
    AUDIT_PROJECTION_DIM = 128
    AUDIT_PROJECTION_SEEDS = [7101, 9101]
    READOUT_PROJECTION_DIM = 256
    ADAPTATION_EPOCHS = 12
    TRAINING_EPOCHS = ADAPTATION_EPOCHS
    SELECTION_EPOCHS = [4, 8, 12]
    BOOTSTRAP_REPS = 2000
elif RUN_MODE != "smoke":
    raise ValueError("RUN_MODE must be 'smoke' or 'full'")

assert MODEL_NAME == ["jepa_wm_pusht", "jepa_wm_wall"]
assert ENVIRONMENT == ["PushT", "Wall"]
assert HORIZONS == [1, 3, 6]
assert TARGET_STEPS == [1, 2, 3, 4, 5, 6]
assert ACTIONS_PER_STATE == 10
assert NUM_STATES % TASKS_PER_ENVIRONMENT == 0
assert sum(TASK_SPLIT_COUNTS) == TASKS_PER_ENVIRONMENT
assert SELECTION_EPOCHS[-1] == ADAPTATION_EPOCHS
assert 2 <= ACTION_GROUP_SIZE <= ACTIONS_PER_STATE
'''

intro = r'''# Stage 9: counterfactual value-equivalent AdaLN adaptation

This notebook tests a different class of fix from Stages 5--8.

Those stages kept the JEPA transition function frozen, or corrected its output
after the transition was already formed. Stage 9 updates only the real
action-conditioning pathway in the pinned JEPA-WM predictor: the action
encoder and the six AdaLN modulation maps. The visual encoder and all
state/content weights remain frozen.

For each exactly restored simulator state, the notebook rolls out a null branch
and nine alternative action sequences. It trains predicted latent differences
to represent the corresponding **goal-independent physical outcome
differences**. The native latent target remains as an anchor.

The core loss is

\[
\mathcal L =
\lambda_z L_{\rm latent}
+\lambda_x L_{\rm state}
+\lambda_\Delta L_{\rm same\text{-}state\ effect}
+\lambda_a L_{\rm displacement\ action}.
\]

The temporary semantic and action heads are discarded. Every predictor is then
evaluated through a new, identical linear physical-state readout fitted only on
the probe-training tasks. This makes the test about transition geometry rather
than head memorization.

Controls:

- ordinary latent-only adaptation of the identical parameters;
- a deterministic within-state shuffle of non-null action/outcome assignment;
- the frozen predictor;
- native goal-latent distance and simulator oracle.

This is exploratory method development on an inspected task family. A positive
result must be followed by a new-task, multi-seed confirmation.
'''

setup = "".join(base["cells"][3]["source"])
setup = setup.replace("stage8", "stage9").replace("Stage 8", "Stage 9")
setup = setup.replace(
    '    "pinned_dependencies": PINNED,\n',
    '''    "ADAPTATION_METHODS": ADAPTATION_METHODS,
    "ADAPTATION_SEED": ADAPTATION_SEED,
    "ADAPTATION_EPOCHS": ADAPTATION_EPOCHS,
    "ADAPTATION_LR": ADAPTATION_LR,
    "ADAPTATION_WEIGHT_DECAY": ADAPTATION_WEIGHT_DECAY,
    "GRADIENT_CLIP": GRADIENT_CLIP,
    "READOUT_PROJECTION_DIM": READOUT_PROJECTION_DIM,
    "EVALUATION_PROJECTION_SEED": EVALUATION_PROJECTION_SEED,
    "ACTION_GROUP_SIZE": ACTION_GROUP_SIZE,
    "LOSS_WEIGHTS": LOSS_WEIGHTS,
    "USE_ACTIVATION_CHECKPOINTING": USE_ACTIVATION_CHECKPOINTING,
    "NATIVE_NONINFERIORITY_TOLERANCE": NATIVE_NONINFERIORITY_TOLERANCE,
    "pinned_dependencies": PINNED,
''',
)

phase_c = r'''# Phase C — adapt the causal action-conditioning path.

ADAPTED_DIR = OUT / "adapted_action_paths"
ADAPTATION_FEATURE_DIR = INTERMEDIATE / "stage9_rollout_features"
for path in [ADAPTED_DIR, ADAPTATION_FEATURE_DIR]:
    path.mkdir(parents=True, exist_ok=True)

CONFIG.update(
    {
        "ADAPTATION_METHODS": ADAPTATION_METHODS,
        "ADAPTATION_SEED": ADAPTATION_SEED,
        "ADAPTATION_EPOCHS": ADAPTATION_EPOCHS,
        "ADAPTATION_LR": ADAPTATION_LR,
        "ADAPTATION_WEIGHT_DECAY": ADAPTATION_WEIGHT_DECAY,
        "GRADIENT_CLIP": GRADIENT_CLIP,
        "READOUT_PROJECTION_DIM": READOUT_PROJECTION_DIM,
        "EVALUATION_PROJECTION_SEED": EVALUATION_PROJECTION_SEED,
        "ACTION_GROUP_SIZE": ACTION_GROUP_SIZE,
        "LOSS_WEIGHTS": LOSS_WEIGHTS,
        "USE_ACTIVATION_CHECKPOINTING": USE_ACTIVATION_CHECKPOINTING,
        "EVIDENCE_STATUS": EVIDENCE_STATUS,
        "TASK_FAMILY_ID": TASK_FAMILY_ID,
    }
)
write_json(OUT / "config.json", CONFIG)


class TemporarySemanticHead(torch.nn.Module):
    def __init__(self, input_dim, pose_dim, action_dim):
        super().__init__()
        self.pose = torch.nn.Linear(input_dim, pose_dim)
        self.action = torch.nn.Linear(input_dim, action_dim)


def action_path_named_parameters(predictor):
    named = []
    for name, parameter in predictor.action_encoder.named_parameters():
        named.append((f"action_encoder.{name}", parameter))
    for index, block in enumerate(predictor.predictor_blocks):
        for name, parameter in block.adaLN_modulation[1].named_parameters():
            named.append(
                (f"predictor_blocks.{index}.adaLN_modulation.1.{name}", parameter)
            )
    return named


def extract_action_path_state(predictor):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in action_path_named_parameters(predictor)
    }


def load_action_path_state(predictor, state):
    named = dict(action_path_named_parameters(predictor))
    if set(named) != set(state):
        raise RuntimeError(
            f"action-path keys differ: model={sorted(named)} state={sorted(state)}"
        )
    with torch.no_grad():
        for name, parameter in named.items():
            parameter.copy_(state[name].to(parameter.device, parameter.dtype))


def set_action_path_trainability(model, trainable):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    predictor, _ = validate_jepa_predictor(model, "loaded_model")
    for _, parameter in action_path_named_parameters(predictor):
        parameter.requires_grad_(trainable)
    return predictor


def state_ids_for_split(environment, split_name):
    return [
        record["state_id"]
        for record in build_state_records(environment)
        if record["split"] == split_name
    ]


def deterministic_outcome_permutation(state_id):
    rng = np.random.default_rng(
        ADAPTATION_SEED + 1009 * int(state_id)
    )
    non_null = np.arange(1, ACTIONS_PER_STATE, dtype=np.int64)
    rng.shuffle(non_null)
    return np.concatenate(
        [np.asarray([0], dtype=np.int64), non_null]
    )


def action_groups():
    non_null = list(range(1, ACTIONS_PER_STATE))
    width = ACTION_GROUP_SIZE - 1
    return [
        np.asarray([0, *non_null[start : start + width]], dtype=np.int64)
        for start in range(0, len(non_null), width)
    ]


def pose_statistics(environment, state_ids):
    values = []
    truth_dir = TRUTH_ROOT / environment.lower()
    for state_id in state_ids:
        with np.load(truth_dir / f"state_{state_id:04d}.npz") as shard:
            values.append(
                pose_target(environment, shard["all_endpoint_states"])
            )
    values = np.concatenate(
        [value.reshape(-1, value.shape[-1]) for value in values],
        axis=0,
    )
    mean = values.mean(axis=0).astype(np.float32)
    scale = values.std(axis=0).astype(np.float32)
    scale[scale < 1e-4] = 1.0
    return mean, scale


def cache_initial_encodings(model, environment, state_ids):
    cache = {}
    truth_dir = TRUTH_ROOT / environment.lower()
    with torch.inference_mode():
        for state_id in state_ids:
            with np.load(
                truth_dir / f"state_{state_id:04d}.npz"
            ) as shard:
                encoded = model.encode(
                    to_model_observation(
                        shard["initial_visual"],
                        shard["initial_proprio"],
                    )
                )
            cache[state_id] = {
                key: value.detach().to("cpu", dtype=torch.float16)
                for key, value in encoded.items()
            }
    return cache


def initial_to_cuda(initial):
    return {
        key: value.to("cuda", dtype=torch.float32, non_blocking=True)
        for key, value in initial.items()
    }


def differentiable_unroll(model, initial_encoded, action_batch):
    batch = action_batch.shape[0]
    action_features = model.model.encode_act(action_batch)
    visual_history = initial_encoded["visual"].expand(
        batch, *initial_encoded["visual"].shape[1:]
    )
    proprio_history = initial_encoded["proprio"].expand(
        batch, *initial_encoded["proprio"].shape[1:]
    )
    predictions = []
    displacements = []
    proprios = []
    for step_index in range(max(HORIZONS)):
        predicted_visual, _, predicted_proprio = model.model.forward_pred(
            visual_history[:, -model.ctxt_window :],
            action_features[:, : step_index + 1][
                :, -model.ctxt_window :
            ],
            proprio_history[:, -model.ctxt_window :],
        )
        next_visual = predicted_visual[:, -1:]
        next_proprio = predicted_proprio[:, -1:]
        current_visual = visual_history[:, -1:]
        next_tokens = next_visual[:, 0, 0].flatten(1, 2)
        current_tokens = current_visual[:, 0, 0].flatten(1, 2)
        predictions.append(next_tokens)
        displacements.append(next_tokens - current_tokens)
        proprios.append(next_proprio[:, 0])
        visual_history = torch.cat([visual_history, next_visual], dim=1)
        proprio_history = torch.cat(
            [proprio_history, next_proprio], dim=1
        )
    return (
        torch.stack(predictions, dim=1),
        torch.stack(displacements, dim=1),
        torch.stack(proprios, dim=1),
    )


def load_adaptation_state(environment, model_name, state_id):
    truth_path = (
        TRUTH_ROOT
        / environment.lower()
        / f"state_{state_id:04d}.npz"
    )
    transition_path = (
        TRANSITION_ROOT
        / model_name
        / f"state_{state_id:04d}.npz"
    )
    with np.load(truth_path) as truth:
        pose = pose_target(
            environment, truth["all_endpoint_states"]
        ).astype(np.float32)
    with np.load(transition_path) as transition:
        actions = transition["normalized_action"].astype(np.float32)
        true_tokens = transition["true_tokens"].astype(np.float32)
    return pose, actions, true_tokens


def normalized_pose_tensor(pose, mean, scale):
    return torch.as_tensor(
        (pose - mean[None, None]) / scale[None, None],
        device="cuda",
        dtype=torch.float32,
    )


def proximity_loss(predictor, reference):
    pieces = []
    for name, parameter in action_path_named_parameters(predictor):
        target = reference[name].to(parameter.device, parameter.dtype)
        denominator = target.square().mean().clamp_min(1e-6)
        pieces.append((parameter - target).square().mean() / denominator)
    return torch.stack(pieces).mean()


def adaptation_losses(
    method,
    predicted_tokens,
    displacements,
    true_tokens,
    target_pose,
    action_target,
    head,
    projector,
    predictor,
    reference,
):
    anchor = torch_functional.smooth_l1_loss(
        predicted_tokens, true_tokens
    )
    zero = anchor.new_zeros(())
    state = zero
    effect = zero
    action = zero
    if method != "latent_only_action_path":
        batch, steps, tokens, channels = predicted_tokens.shape
        projected = projector(
            predicted_tokens.reshape(batch * steps, tokens, channels)
        ).reshape(batch, steps, -1)
        projected_delta = projector(
            displacements.reshape(batch * steps, tokens, channels)
        ).reshape(batch, steps, -1)
        predicted_pose = head.pose(projected)
        predicted_action = head.action(projected_delta)
        state = torch_functional.smooth_l1_loss(
            predicted_pose, target_pose
        )
        effect = torch_functional.smooth_l1_loss(
            predicted_pose - predicted_pose[:1],
            target_pose - target_pose[:1],
        )
        action = torch_functional.mse_loss(
            predicted_action, action_target
        )
    proximity = proximity_loss(predictor, reference)
    total = (
        LOSS_WEIGHTS["anchor"] * anchor
        + LOSS_WEIGHTS["state"] * state
        + LOSS_WEIGHTS["effect"] * effect
        + LOSS_WEIGHTS["action"] * action
        + LOSS_WEIGHTS["proximity"] * proximity
    )
    return total, {
        "anchor": float(anchor.detach()),
        "state": float(state.detach()),
        "effect": float(effect.detach()),
        "action": float(action.detach()),
        "proximity": float(proximity.detach()),
        "total": float(total.detach()),
    }


def calibration_objective(
    model,
    method,
    environment,
    model_name,
    state_ids,
    initial_cache,
    projector,
    head,
    pose_mean,
    pose_scale,
):
    losses = []
    with torch.inference_mode():
        for state_id in state_ids:
            pose, actions, true_tokens = load_adaptation_state(
                environment, model_name, state_id
            )
            if method == "shuffled_cave_action_path":
                pose = pose[deterministic_outcome_permutation(state_id)]
            initial = initial_to_cuda(initial_cache[state_id])
            action_tensor = torch.as_tensor(
                actions, device="cuda", dtype=torch.float32
            )
            predicted, displacement, _ = differentiable_unroll(
                model, initial, action_tensor
            )
            true_tensor = torch.as_tensor(
                true_tokens, device="cuda", dtype=torch.float32
            )
            anchor = torch_functional.smooth_l1_loss(
                predicted, true_tensor
            )
            if method == "latent_only_action_path":
                losses.append(float(anchor))
                continue
            target = normalized_pose_tensor(
                pose, pose_mean, pose_scale
            )
            batch, steps, tokens, channels = predicted.shape
            projected = projector(
                predicted.reshape(batch * steps, tokens, channels)
            ).reshape(batch, steps, -1)
            predicted_pose = head.pose(projected)
            state = torch_functional.smooth_l1_loss(
                predicted_pose, target
            )
            effect = torch_functional.smooth_l1_loss(
                predicted_pose - predicted_pose[:1],
                target - target[:1],
            )
            losses.append(
                float(
                    LOSS_WEIGHTS["anchor"] * anchor
                    + LOSS_WEIGHTS["state"] * state
                    + LOSS_WEIGHTS["effect"] * effect
                )
            )
    return float(np.mean(losses))


def train_one_action_path(
    model,
    environment,
    model_name,
    method,
    base_state,
    train_ids,
    calibration_ids,
    initial_cache,
    pose_mean,
    pose_scale,
):
    output_path = ADAPTED_DIR / f"{model_name}_{method}.pt"
    if output_path.exists():
        log.info("adaptation resume: keeping %s", output_path.name)
        return torch.load(output_path, map_location="cpu", weights_only=False)

    predictor = set_action_path_trainability(model, True)
    load_action_path_state(predictor, base_state)
    predictor.use_activation_checkpointing = USE_ACTIVATION_CHECKPOINTING
    pose_dim = 4 if environment == "PushT" else 2
    action_dim = int(
        np.load(
            TRANSITION_ROOT / model_name / "state_0000.npz"
        )["normalized_action"].shape[-1]
    )
    torch.manual_seed(
        ADAPTATION_SEED + 1000 * ENVIRONMENT.index(environment)
        + 100 * ADAPTATION_METHODS.index(method)
    )
    torch.cuda.manual_seed_all(
        ADAPTATION_SEED + 1000 * ENVIRONMENT.index(environment)
        + 100 * ADAPTATION_METHODS.index(method)
    )
    head = TemporarySemanticHead(
        READOUT_PROJECTION_DIM, pose_dim, action_dim
    ).cuda()
    head.apply(
        lambda module: (
            torch.nn.init.xavier_uniform_(module.weight)
            if isinstance(module, torch.nn.Linear)
            else None
        )
    )
    projector = CountSketchProjector(
        16 * 16 * int(predictor.predictor_embed_dim),
        READOUT_PROJECTION_DIM,
        ADAPTATION_SEED + 17 * ENVIRONMENT.index(environment),
    )
    parameters = [
        parameter
        for _, parameter in action_path_named_parameters(predictor)
    ]
    if method != "latent_only_action_path":
        parameters += list(head.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=ADAPTATION_LR,
        weight_decay=ADAPTATION_WEIGHT_DECAY,
    )
    rng = np.random.default_rng(
        ADAPTATION_SEED + 31 * ADAPTATION_METHODS.index(method)
    )
    history = []
    best = None

    for epoch in range(1, ADAPTATION_EPOCHS + 1):
        order = np.asarray(train_ids, dtype=np.int64).copy()
        rng.shuffle(order)
        epoch_terms = []
        model.eval()
        for state_id in order:
            pose, actions, true_tokens = load_adaptation_state(
                environment, model_name, int(state_id)
            )
            if method == "shuffled_cave_action_path":
                pose = pose[
                    deterministic_outcome_permutation(int(state_id))
                ]
            initial = initial_to_cuda(initial_cache[int(state_id)])
            for group in action_groups():
                action_tensor = torch.as_tensor(
                    actions[group], device="cuda", dtype=torch.float32
                )
                target_tokens = torch.as_tensor(
                    true_tokens[group],
                    device="cuda",
                    dtype=torch.float32,
                )
                target_pose = normalized_pose_tensor(
                    pose[group], pose_mean, pose_scale
                )
                optimizer.zero_grad(set_to_none=True)
                predicted, displacement, _ = differentiable_unroll(
                    model, initial, action_tensor
                )
                total, terms = adaptation_losses(
                    method,
                    predicted,
                    displacement,
                    target_tokens,
                    target_pose,
                    action_tensor,
                    head,
                    projector,
                    predictor,
                    base_state,
                )
                total.backward()
                torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP)
                optimizer.step()
                epoch_terms.append(terms)

        row = {
            "environment": environment,
            "model": model_name,
            "method": method,
            "epoch": epoch,
        }
        for key in ["anchor", "state", "effect", "action", "proximity", "total"]:
            row[key] = float(np.mean([item[key] for item in epoch_terms]))

        if epoch in SELECTION_EPOCHS:
            predictor.use_activation_checkpointing = False
            score = calibration_objective(
                model,
                method,
                environment,
                model_name,
                calibration_ids,
                initial_cache,
                projector,
                head,
                pose_mean,
                pose_scale,
            )
            predictor.use_activation_checkpointing = USE_ACTIVATION_CHECKPOINTING
            row["calibration_objective"] = score
            candidate = {
                "method": method,
                "environment": environment,
                "model": model_name,
                "selected_epoch": epoch,
                "calibration_objective": score,
                "action_path": extract_action_path_state(predictor),
                "pose_mean": pose_mean,
                "pose_scale": pose_scale,
                "config": {
                    "loss_weights": LOSS_WEIGHTS,
                    "lr": ADAPTATION_LR,
                    "weight_decay": ADAPTATION_WEIGHT_DECAY,
                    "seed": ADAPTATION_SEED,
                },
            }
            if best is None or score < best["calibration_objective"]:
                best = candidate
        else:
            row["calibration_objective"] = float("nan")
        history.append(row)
        log.info(
            "%s %s epoch=%d total=%.6f calibration=%s",
            model_name,
            method,
            epoch,
            row["total"],
            row["calibration_objective"],
        )

    if best is None:
        raise RuntimeError("no adaptation checkpoint was eligible")
    torch.save(best, output_path)
    existing = []
    history_path = OUT / "action_path_training_history.csv"
    if history_path.exists():
        with history_path.open() as handle:
            existing = list(csv.DictReader(handle))
    write_csv(history_path, [*existing, *history])
    load_action_path_state(predictor, best["action_path"])
    predictor.use_activation_checkpointing = False
    return best


def adapt_action_paths():
    manifest = []
    repo = configure_repo()
    for environment in ENVIRONMENT:
        model_name = MODEL_BY_ENVIRONMENT[environment][0]
        model, _ = torch.hub.load(
            str(repo),
            model_name,
            source="local",
            pretrained=True,
            device="cuda:0",
            trust_repo=True,
        )
        model.eval()
        predictor = set_action_path_trainability(model, False)
        base_state = extract_action_path_state(predictor)
        train_ids = state_ids_for_split(environment, "probe_train")
        calibration_ids = state_ids_for_split(
            environment, "probe_calibration"
        )
        initial_cache = cache_initial_encodings(
            model, environment, [*train_ids, *calibration_ids]
        )
        pose_mean, pose_scale = pose_statistics(environment, train_ids)
        for method in ADAPTATION_METHODS:
            result = train_one_action_path(
                model,
                environment,
                model_name,
                method,
                base_state,
                train_ids,
                calibration_ids,
                initial_cache,
                pose_mean,
                pose_scale,
            )
            manifest.append(
                {
                    "environment": environment,
                    "model": model_name,
                    "method": method,
                    "selected_epoch": result["selected_epoch"],
                    "calibration_objective": result[
                        "calibration_objective"
                    ],
                    "trainable_parameter_count": int(
                        sum(
                            parameter.numel()
                            for _, parameter in action_path_named_parameters(
                                predictor
                            )
                        )
                    ),
                }
            )
        del model, predictor, initial_cache
        gc.collect()
        torch.cuda.empty_cache()
    write_json(
        OUT / "action_path_adaptation_manifest.json",
        {
            "run_signature": RUN_SIGNATURE,
            "updated_modules": [
                "predictor.action_encoder",
                "predictor.predictor_blocks.*.adaLN_modulation[1]",
            ],
            "frozen_modules": [
                "visual encoder",
                "predictor attention and MLP content weights",
                "predictor output projection",
                "proprio path",
            ],
            "records": manifest,
        },
    )


if not PIPELINE_FAILED:
    try:
        adapt_action_paths()
    except Exception:
        record_failure("counterfactual_value_equivalent_action_path_adaptation")
'''

phase_d = r'''# Phase D — discard training heads; refit identical linear readouts and evaluate.

EVALUATION_VARIANTS = ["frozen", *ADAPTATION_METHODS]
HORIZON_INDICES = [horizon - 1 for horizon in HORIZONS]


def rollout_feature_path(model_name, variant, state_id):
    return (
        ADAPTATION_FEATURE_DIR
        / model_name
        / variant
        / f"state_{state_id:04d}.npz"
    )


def cache_variant_rollouts():
    repo = configure_repo()
    for environment in ENVIRONMENT:
        model_name = MODEL_BY_ENVIRONMENT[environment][0]
        model, _ = torch.hub.load(
            str(repo),
            model_name,
            source="local",
            pretrained=True,
            device="cuda:0",
            trust_repo=True,
        )
        model.eval()
        predictor = set_action_path_trainability(model, False)
        predictor.use_activation_checkpointing = False
        base_state = extract_action_path_state(predictor)
        initial_cache = cache_initial_encodings(
            model, environment, list(range(NUM_STATES))
        )
        # Deliberately differs from the projection that carried adaptation
        # gradients. This blocks projection-specific codebook shortcuts.
        projector = CountSketchProjector(
            16 * 16 * int(predictor.predictor_embed_dim),
            READOUT_PROJECTION_DIM,
            EVALUATION_PROJECTION_SEED
            + 17 * ENVIRONMENT.index(environment),
        )
        with np.load(GOAL_ROOT / f"{model_name}.npz") as goals:
            goal_visual = goals["visual"].astype(np.float32)

        for variant in EVALUATION_VARIANTS:
            if variant == "frozen":
                load_action_path_state(predictor, base_state)
            else:
                checkpoint = torch.load(
                    ADAPTED_DIR / f"{model_name}_{variant}.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                load_action_path_state(
                    predictor, checkpoint["action_path"]
                )
            for state_id in range(NUM_STATES):
                path = rollout_feature_path(
                    model_name, variant, state_id
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    continue
                with np.load(
                    TRANSITION_ROOT
                    / model_name
                    / f"state_{state_id:04d}.npz"
                ) as transition:
                    actions = transition["normalized_action"].astype(
                        np.float32
                    )
                    task_id = int(transition["task_id"])
                initial = initial_to_cuda(initial_cache[state_id])
                action_tensor = torch.as_tensor(
                    actions, device="cuda", dtype=torch.float32
                )
                with torch.inference_mode():
                    predicted, displacement, predicted_proprio = (
                        differentiable_unroll(
                            model, initial, action_tensor
                        )
                    )
                    batch, steps, tokens, channels = predicted.shape
                    projected = projector(
                        predicted.reshape(
                            batch * steps, tokens, channels
                        )
                    ).reshape(batch, steps, -1)
                    goal = torch.as_tensor(
                        goal_visual[task_id],
                        device="cuda",
                        dtype=torch.float32,
                    )
                    latent_cost = torch.sqrt(
                        torch.mean(
                            (
                                predicted
                                - goal[None, None, :, :]
                            )
                            ** 2,
                            dim=(2, 3),
                        ).clamp_min(1e-12)
                    )
                atomic_npz_uncompressed(
                    path,
                    projected=projected.detach().cpu().numpy().astype(
                        np.float16
                    ),
                    latent_cost=latent_cost.detach().cpu().numpy().astype(
                        np.float32
                    ),
                    predicted_proprio=predicted_proprio.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float16),
                )
            log.info("%s cached variant %s", model_name, variant)
        del model, predictor, initial_cache
        gc.collect()
        torch.cuda.empty_cache()


def load_truth_evaluation(environment):
    payload = {
        "pose": [],
        "physical_cost": [],
        "interactions": [],
        "task_id": [],
        "split": [],
        "evaluation_seed": [],
    }
    for state_id in range(NUM_STATES):
        with np.load(
            TRUTH_ROOT
            / environment.lower()
            / f"state_{state_id:04d}.npz"
        ) as shard:
            payload["pose"].append(
                pose_target(environment, shard["all_endpoint_states"])[
                    :, HORIZON_INDICES
                ]
            )
            payload["physical_cost"].append(shard["physical_cost"])
            payload["interactions"].append(shard["interactions"])
            payload["task_id"].append(int(shard["task_id"]))
            payload["split"].append(str(shard["task_split"]))
            payload["evaluation_seed"].append(
                int(shard["evaluation_seed"])
            )
    return {
        key: np.asarray(value)
        for key, value in payload.items()
    }


def load_variant_features(model_name, variant):
    projected = []
    latent_cost = []
    for state_id in range(NUM_STATES):
        with np.load(
            rollout_feature_path(model_name, variant, state_id)
        ) as shard:
            projected.append(
                shard["projected"][:, HORIZON_INDICES].astype(np.float32)
            )
            latent_cost.append(
                shard["latent_cost"][:, HORIZON_INDICES].astype(
                    np.float64
                )
            )
    return np.asarray(projected), np.asarray(latent_cost)


def split_indices(truth, name):
    return np.flatnonzero(truth["split"] == name)


def flatten_selected(values, indices):
    selected = np.asarray(values)[indices]
    return selected.reshape(-1, selected.shape[-1])


def evaluate_stage9_methods():
    unit_rows = []
    probe_rows = []
    tasks_lookup = {
        environment: {
            task["task_id"]: task for task in TASKS[environment]
        }
        for environment in ENVIRONMENT
    }

    for environment in ENVIRONMENT:
        model_name = MODEL_BY_ENVIRONMENT[environment][0]
        truth = load_truth_evaluation(environment)
        train_ids = split_indices(truth, "probe_train")
        calibration_ids = split_indices(
            truth, "probe_calibration"
        )
        development_ids = split_indices(
            truth, DEVELOPMENT_SPLIT
        )
        method_costs = {}
        method_poses = {}

        for variant in EVALUATION_VARIANTS:
            features, latent_cost = load_variant_features(
                model_name, variant
            )
            probe = fit_linear_readout(
                flatten_selected(features, train_ids),
                flatten_selected(truth["pose"], train_ids),
                flatten_selected(features, calibration_ids),
                flatten_selected(truth["pose"], calibration_ids),
            )
            predicted_pose = predict_linear_readout(
                probe, features.reshape(-1, features.shape[-1])
            ).reshape(*features.shape[:-1], truth["pose"].shape[-1])
            method_poses[f"{variant}_linear_pose"] = predicted_pose
            method_costs[f"{variant}_latent_distance"] = latent_cost
            probe_rows.append(
                {
                    "environment": environment,
                    "model": model_name,
                    "variant": variant,
                    "ridge": probe["ridge"],
                    "calibration_pose_mse": probe[
                        "calibration_pose_mse"
                    ],
                    "readout_projection_dim": READOUT_PROJECTION_DIM,
                    "evaluation_projection_seed": (
                        EVALUATION_PROJECTION_SEED
                        + 17 * ENVIRONMENT.index(environment)
                    ),
                }
            )

        for state_id in development_ids:
            task_id = int(truth["task_id"][state_id])
            task = tasks_lookup[environment][task_id]
            true_cost = truth["physical_cost"][state_id]
            true_pose = truth["pose"][state_id]
            costs = {
                key: value[state_id]
                for key, value in method_costs.items()
            }
            for key, value in method_poses.items():
                costs[key] = decoded_task_cost(
                    environment, value[state_id], task
                )
            costs["oracle_pose"] = true_cost.copy()

            for horizon_index, horizon in enumerate(HORIZONS):
                for method, predicted in costs.items():
                    ranking = ranking_metrics(
                        true_cost[:, horizon_index],
                        predicted[:, horizon_index],
                    )
                    if method.endswith("_linear_pose"):
                        pose_prediction = method_poses[method][
                            state_id, :, horizon_index
                        ]
                        pose_error = float(
                            np.mean(
                                physical_pose_error(
                                    environment,
                                    pose_prediction,
                                    true_pose[:, horizon_index],
                                )
                            )
                        )
                    else:
                        pose_error = float("nan")
                    unit_rows.append(
                        {
                            "environment": environment,
                            "model": model_name,
                            "state_id": int(state_id),
                            "task_id": task_id,
                            "evaluation_seed": int(
                                truth["evaluation_seed"][state_id]
                            ),
                            "split": DEVELOPMENT_SPLIT,
                            "method": method,
                            "horizon": horizon,
                            "normalized_regret": ranking[
                                "normalized_regret"
                            ],
                            "weighted_pairwise_accuracy": ranking[
                                "weighted_pairwise_accuracy"
                            ],
                            "top1_correct": ranking["top1_correct"],
                            "normalized_margin_rmse": ranking[
                                "normalized_margin_rmse"
                            ],
                            "pose_error": pose_error,
                            "selected_action": ranking[
                                "selected_action"
                            ],
                            "oracle_action": ranking["oracle_action"],
                            "selected_is_null": int(
                                ranking["selected_action"] == 0
                            ),
                            "interaction_count_selected": int(
                                truth["interactions"][
                                    state_id,
                                    ranking["selected_action"],
                                    horizon_index,
                                ]
                            ),
                        }
                    )

    write_csv(OUT / "stage9_unit_metrics.csv", unit_rows)
    write_csv(OUT / "stage9_probe_selection.csv", probe_rows)
    return unit_rows


def summarize_and_contrast(unit_rows):
    summary = []
    fields = [
        "normalized_regret",
        "weighted_pairwise_accuracy",
        "top1_correct",
        "normalized_margin_rmse",
        "pose_error",
    ]
    keys = sorted(
        {
            (row["environment"], row["method"], int(row["horizon"]))
            for row in unit_rows
        }
    )
    for environment, method, horizon in keys:
        selected = [
            row
            for row in unit_rows
            if row["environment"] == environment
            and row["method"] == method
            and int(row["horizon"]) == horizon
        ]
        for metric in fields:
            result = bootstrap_mean(
                [float(row[metric]) for row in selected],
                [int(row["state_id"]) for row in selected],
                BOOTSTRAP_REPS,
                SEED
                + 101 * ENVIRONMENT.index(environment)
                + 13 * HORIZONS.index(horizon),
            )
            summary.append(
                {
                    "environment": environment,
                    "method": method,
                    "horizon": horizon,
                    "metric": metric,
                    **result,
                }
            )

    contrasts = []
    comparisons = [
        (
            "matched_cave_action_path_linear_pose",
            "frozen_linear_pose",
        ),
        (
            "matched_cave_action_path_linear_pose",
            "latent_only_action_path_linear_pose",
        ),
        (
            "matched_cave_action_path_linear_pose",
            "shuffled_cave_action_path_linear_pose",
        ),
        (
            "matched_cave_action_path_latent_distance",
            "frozen_latent_distance",
        ),
    ]
    for environment in ENVIRONMENT:
        for treatment, baseline in comparisons:
            for horizon in HORIZONS:
                for metric in [
                    "normalized_regret",
                    "weighted_pairwise_accuracy",
                    "top1_correct",
                ]:
                    treatment_rows = {
                        int(row["state_id"]): float(row[metric])
                        for row in unit_rows
                        if row["environment"] == environment
                        and row["method"] == treatment
                        and int(row["horizon"]) == horizon
                    }
                    baseline_rows = {
                        int(row["state_id"]): float(row[metric])
                        for row in unit_rows
                        if row["environment"] == environment
                        and row["method"] == baseline
                        and int(row["horizon"]) == horizon
                    }
                    common = sorted(
                        set(treatment_rows) & set(baseline_rows)
                    )
                    sign = -1.0 if metric == "normalized_regret" else 1.0
                    values = [
                        sign
                        * (
                            treatment_rows[state_id]
                            - baseline_rows[state_id]
                        )
                        for state_id in common
                    ]
                    result = bootstrap_mean(
                        values,
                        common,
                        BOOTSTRAP_REPS,
                        SEED + 7001 + 17 * HORIZONS.index(horizon),
                    )
                    contrasts.append(
                        {
                            "environment": environment,
                            "treatment": treatment,
                            "baseline": baseline,
                            "horizon": horizon,
                            "metric": metric,
                            "positive_means_treatment_better": True,
                            **result,
                        }
                    )
    write_csv(OUT / "stage9_metrics_summary.csv", summary)
    write_csv(OUT / "stage9_method_contrasts.csv", contrasts)
    return summary, contrasts


if not PIPELINE_FAILED:
    try:
        cache_variant_rollouts()
        STAGE9_UNIT_ROWS = evaluate_stage9_methods()
        STAGE9_SUMMARY, STAGE9_CONTRASTS = summarize_and_contrast(
            STAGE9_UNIT_ROWS
        )
    except Exception:
        record_failure("fresh_readout_and_development_evaluation")
'''

phase_e = r'''# Phase E — decision gate and compact diagnostics.


def contrast_lookup(contrasts, environment, treatment, baseline, metric):
    selected = [
        row
        for row in contrasts
        if row["environment"] == environment
        and row["treatment"] == treatment
        and row["baseline"] == baseline
        and row["metric"] == metric
    ]
    return selected


def stage9_decision():
    treatment = "matched_cave_action_path_linear_pose"
    frozen = "frozen_linear_pose"
    latent_only = "latent_only_action_path_linear_pose"
    shuffled = "shuffled_cave_action_path_linear_pose"
    gates = {}
    for environment in ENVIRONMENT:
        comparisons = {}
        for label, baseline in [
            ("versus_frozen", frozen),
            ("versus_latent_only", latent_only),
            ("versus_shuffled", shuffled),
        ]:
            regret = contrast_lookup(
                STAGE9_CONTRASTS,
                environment,
                treatment,
                baseline,
                "normalized_regret",
            )
            ranking = contrast_lookup(
                STAGE9_CONTRASTS,
                environment,
                treatment,
                baseline,
                "weighted_pairwise_accuracy",
            )
            comparisons[label] = {
                "regret_positive_horizons": int(
                    sum(float(row["estimate"]) > 0 for row in regret)
                ),
                "ranking_positive_horizons": int(
                    sum(float(row["estimate"]) > 0 for row in ranking)
                ),
                "regret_all": regret,
                "ranking_all": ranking,
            }
        gates[environment] = comparisons

    continue_gate = all(
        gates[environment]["versus_frozen"][
            "regret_positive_horizons"
        ]
        >= 2
        and gates[environment]["versus_frozen"][
            "ranking_positive_horizons"
        ]
        >= 2
        and gates[environment]["versus_latent_only"][
            "regret_positive_horizons"
        ]
        >= 2
        and gates[environment]["versus_shuffled"][
            "ranking_positive_horizons"
        ]
        >= 2
        for environment in ENVIRONMENT
    )
    payload = {
        "evidence_status": EVIDENCE_STATUS,
        "decision": (
            "ADVANCE_TO_NEW_TASK_CONFIRMATION"
            if continue_gate
            else "DO_NOT_CONFIRM_YET"
        ),
        "criterion": (
            "matched CAVE must improve regret and ranking at >=2 horizons "
            "in both environments, beat latent-only, and beat shuffled"
        ),
        "gates": gates,
        "interpretation_guardrail": (
            "This inspected task family is development evidence only. "
            "No positive result is confirmatory."
        ),
    }
    write_json(OUT / "stage9_development_decision.json", payload)
    return payload


def plot_stage9():
    methods = [
        "frozen_linear_pose",
        "latent_only_action_path_linear_pose",
        "shuffled_cave_action_path_linear_pose",
        "matched_cave_action_path_linear_pose",
    ]
    labels = ["Frozen", "Latent only", "Shuffled CAVE", "Matched CAVE"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for row_index, environment in enumerate(ENVIRONMENT):
        for column_index, metric in enumerate(
            ["normalized_regret", "weighted_pairwise_accuracy"]
        ):
            axis = axes[row_index, column_index]
            for method, label in zip(methods, labels):
                values = []
                for horizon in HORIZONS:
                    selected = [
                        float(row["estimate"])
                        for row in STAGE9_SUMMARY
                        if row["environment"] == environment
                        and row["method"] == method
                        and int(row["horizon"]) == horizon
                        and row["metric"] == metric
                    ]
                    values.append(selected[0])
                axis.plot(HORIZONS, values, marker="o", label=label)
            axis.set_title(f"{environment}: {metric}")
            axis.set_xlabel("horizon")
            axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "stage9_planning_comparison.png", dpi=180)
    plt.close(fig)


if not PIPELINE_FAILED:
    try:
        STAGE9_DECISION = stage9_decision()
        plot_stage9()
        print(json.dumps(STAGE9_DECISION, indent=2))
    except Exception:
        record_failure("stage9_decision_and_plots")
'''

phase_f = r'''# Phase F — package all non-cache outputs and download one result bundle.


def package_stage9_results():
    result_zip = Path("/content/stage9_result_bundle.zip")
    excluded_roots = {
        str(INTERMEDIATE.resolve()),
        str(CACHE_ROOT.resolve()),
    }
    files = []
    for path in OUT.rglob("*"):
        if not path.is_file():
            continue
        resolved = str(path.resolve())
        if any(
            resolved == root or resolved.startswith(root + os.sep)
            for root in excluded_roots
        ):
            continue
        files.append(path)
    manifest = [
        {
            "path": str(path.relative_to(OUT)),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for path in sorted(files)
    ]
    write_json(
        OUT / "result_zip_manifest.json",
        {
            "run_signature": RUN_SIGNATURE,
            "pipeline_failed": bool(PIPELINE_FAILED),
            "files": manifest,
        },
    )
    files = sorted(
        {
            *files,
            OUT / "result_zip_manifest.json",
            OUT / "FAILURE_TRACE.txt",
        }
    )
    with zipfile.ZipFile(
        result_zip,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for path in files:
            if path.exists():
                archive.write(path, path.relative_to(OUT))
    print(f"RESULT_ZIP: {result_zip}")
    print("RUN_STATUS:", "FAILED" if PIPELINE_FAILED else "SUCCESS")
    if DOWNLOAD_RESULTS:
        from google.colab import files as colab_files

        colab_files.download(str(result_zip))
    return result_zip


try:
    RESULT_ZIP = package_stage9_results()
except Exception:
    record_failure("stage9_packaging")
    raise
'''

cells = [
    code(config),
    markdown(intro),
    base["cells"][2],
    code(setup),
    base["cells"][4],
    base["cells"][5],
    base["cells"][6],
    code(phase_c),
    code(phase_d),
    code(phase_e),
    code(phase_f),
]

notebook = {
    "cells": cells,
    "metadata": base["metadata"],
    "nbformat": 4,
    "nbformat_minor": 5,
}
TARGET.write_text(json.dumps(notebook, indent=1) + "\n")
print(TARGET)
