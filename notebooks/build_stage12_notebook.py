import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "11_action_response_geometry_pilot.ipynb"
TARGET = ROOT / "12_shared_target_metric_bridge.ipynb"


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

config = "".join(base["cells"][0]["source"])
config = config.replace(
    '# SINGLE CONFIGURATION BLOCK — the defaults are the compute-gated pilot.',
    "# SINGLE CONFIGURATION BLOCK — Stage 12 Shared Target-Metric Bridge pilot.",
)
config = config.replace(
    'RUN_MODE = "pilot"  # pilot or full; do not use full until the pilot promotes',
    'RUN_MODE = "full"  # required: rebuild the complete three-seed Stage 11 matrix',
)
config = config.replace(
    'OUTPUT_DIR = "/content/counterfactual_faithfulness_stage11"',
    'OUTPUT_DIR = "/content/counterfactual_faithfulness_stage12"',
)
config = config.replace(
    'DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/counterfactual_faithfulness_stage11"',
    'DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/counterfactual_faithfulness_stage12"',
)
config = config.replace(
    "DOWNLOAD_PHASE_C_RESCUE = True",
    "DOWNLOAD_PHASE_C_RESCUE = False",
)
config = config.replace(
    "DOWNLOAD_RESULTS = True",
    "DOWNLOAD_RESULTS = False",
)

assert_marker = "assert MODEL_NAME == "
stage12_constants = r'''
# Stage 12: one target-only planner metric shared by every transition arm.
STAGE12_FORCE_ALL_TRANSITION_SEEDS = True
BRIDGE_IMPLEMENTATION_ID = "stage12_shared_target_metric_bridge_v1"
BRIDGE_PROJECTION_DIM = 128
BRIDGE_PROJECTION_SEED = 15001
EVALUATION_PROJECTION_DIM = BRIDGE_PROJECTION_DIM
EVALUATION_PROJECTION_SEEDS = [BRIDGE_PROJECTION_SEED]
DIRECT_PROJECTION_CONSENSUS_REQUIRED = 1
READOUT_PROJECTION_CONSENSUS_REQUIRED = 1
BRIDGE_RANKS = [2, 8]
BRIDGE_REGULARIZERS = [1e-3, 1e-2]
BRIDGE_OPTIMIZATION_SEEDS = [15101, 15119]
BRIDGE_WHITENING_RIDGE_FRACTION = 0.01
BRIDGE_HUBER_DELTA = 0.1
BRIDGE_PAIR_WEIGHT_FLOOR = 0.05
BRIDGE_CONDITION_LIMIT = 20.0
BRIDGE_CONDITION_HEADROOM = 0.95
BRIDGE_LEARNING_RATE = 3e-2
BRIDGE_MAX_EPOCHS = 600
BRIDGE_MIN_EPOCHS = 120
BRIDGE_PLATEAU_PATIENCE = 60
BRIDGE_PLATEAU_RELATIVE_TOLERANCE = 1e-5
BRIDGE_COST_RANGE_EPS = 1e-6
BRIDGE_BOOTSTRAP_REPS = 2000
BRIDGE_BOOTSTRAP_SEED = 15201
INCLUDE_ALL_TRANSITION_CHECKPOINTS_IN_ZIP = True
DOWNLOAD_STAGE12_RESULTS = True

assert RUN_MODE == "full"
assert NUM_STATES == 96
assert ADAPTATION_SEEDS == [11401, 11419, 11437]
assert EVALUATION_PROJECTION_SEEDS == [15001]
'''
config = config.replace(
    assert_marker,
    stage12_constants + "\n" + assert_marker,
    1,
)

intro = r'''# Stage 12: Shared Target-Metric Bridge

Stage 11 repaired relative action-response geometry, but its planning test
refitted a different physical decoder for every transition arm. That design
could not tell whether matched ARGA made the transition representation more
decision-useful or merely changed which arm-specific decoder was easiest to
fit.

This notebook runs the smallest clean causal bridge test:

1. reconstruct the frozen, latent-only, shuffled-ARGA, and matched-ARGA
   transition arms with all three Stage 11 seeds;
2. freeze every transition model;
3. use target future tokens, target goals, and simulator cost margins only to
   fit one low-rank positive-semidefinite goal metric per environment;
4. freeze that metric and apply the exact same planner to every arm; and
5. compare matched ARGA with frozen, latent-only, and shuffled controls on the
   untouched development split.

For a fixed 128-dimensional CountSketch \(P\) and a train-only whitening
transform \(Q\), let \(y(z)=QP\,\mathrm{vec}(z)\). The planner metric is

\[
M={d\over\operatorname{tr}(I+L^\top L)}(I+L^\top L),
\qquad
d_M(z,g)=(y(z)-y(g))^\top M(y(z)-y(g)),
\]

where \(L\) has rank 2 or 8 and the spectrum is projected to keep
\(\kappa(M)<20\). Three positive horizon scales are learned because they do
not alter the argmin. All 45 candidate pairs enter a task-equal weighted Huber
objective. No predicted rollout, treatment identity, candidate identity,
development outcome, or fitted physical decoder is visible during metric
training or selection.

A deterministic task-level goal derangement is trained as a shortcut control.
Target-future latents provide a metric-capacity ceiling; simulator costs
provide the oracle. The final JSON gate distinguishes:

- `PROMOTE_TO_UNTOUCHED_TASK_CONFIRMATION`;
- `STOP_METRIC_CLASS_NOT_VIABLE`;
- `STOP_NO_CAUSAL_BRIDGE_SIGNAL`; and
- `AMBIGUOUS_DO_NOT_TUNE_ON_DEVELOPMENT_TASKS`.

This remains a development pilot. A pass authorizes a new-task confirmation;
it is not itself a paper-level claim.
'''


def transformed_cell(index):
    cell = json.loads(json.dumps(base["cells"][index]))
    source = "".join(cell.get("source", []))
    source = source.replace("stage11", "stage12")
    source = source.replace("Stage 11", "Stage 12 prerequisite")
    source = source.replace("STAGE11", "STAGE12")
    if index == 7:
        source = source.replace(
            'and not screening["passed"]',
            'and not (screening["passed"] or STAGE12_FORCE_ALL_TRANSITION_SEEDS)',
        )
        source = source.replace(
            '"screening gate failed; skipping confirmation seed %d"',
            '"screening gate failed, but Stage 12 requires confirmation seed %d"',
        )
    cell["source"] = source.splitlines(keepends=True)
    cell["execution_count"] = None
    cell["outputs"] = []
    return cell


setup = transformed_cell(3)
setup_source = "".join(setup["source"])
signature_marker = "RUN_SIGNATURE ="
bridge_config = r'''CONFIG.update({
    "bridge_implementation_id": BRIDGE_IMPLEMENTATION_ID,
    "stage12_force_all_transition_seeds": (
        STAGE12_FORCE_ALL_TRANSITION_SEEDS
    ),
    "bridge_projection_dim": BRIDGE_PROJECTION_DIM,
    "bridge_projection_seed": BRIDGE_PROJECTION_SEED,
    "bridge_ranks": BRIDGE_RANKS,
    "bridge_regularizers": BRIDGE_REGULARIZERS,
    "bridge_optimization_seeds": BRIDGE_OPTIMIZATION_SEEDS,
    "bridge_whitening_ridge_fraction": (
        BRIDGE_WHITENING_RIDGE_FRACTION
    ),
    "bridge_huber_delta": BRIDGE_HUBER_DELTA,
    "bridge_pair_weight_floor": BRIDGE_PAIR_WEIGHT_FLOOR,
    "bridge_condition_limit": BRIDGE_CONDITION_LIMIT,
    "bridge_condition_headroom": BRIDGE_CONDITION_HEADROOM,
    "bridge_learning_rate": BRIDGE_LEARNING_RATE,
    "bridge_max_epochs": BRIDGE_MAX_EPOCHS,
    "bridge_min_epochs": BRIDGE_MIN_EPOCHS,
    "bridge_plateau_patience": BRIDGE_PLATEAU_PATIENCE,
    "bridge_plateau_relative_tolerance": (
        BRIDGE_PLATEAU_RELATIVE_TOLERANCE
    ),
    "bridge_bootstrap_reps": BRIDGE_BOOTSTRAP_REPS,
    "bridge_bootstrap_seed": BRIDGE_BOOTSTRAP_SEED,
})

'''
setup_source = setup_source.replace(
    signature_marker,
    bridge_config + signature_marker,
    1,
)
setup["source"] = setup_source.splitlines(keepends=True)

bridge_intro = r'''## Shared target-only metric and causal bridge gate

The Stage 12 cells below begin only after the complete prerequisite matrix has
been reconstructed and checked. Metric fitting reads target transitions,
encoded task goals, and simulator costs from `probe_train`. Candidate-arm
rollouts are opened only after target-latent calibration has selected and
frozen the metric.
'''

bridge = r'''# Phase G — fit one target-only shared metric and evaluate every frozen arm.

BRIDGE_DIR = OUT / "shared_target_metric"
BRIDGE_DIR.mkdir(parents=True, exist_ok=True)
BRIDGE_METRIC_DIR = BRIDGE_DIR / "metric_checkpoints"
BRIDGE_METRIC_DIR.mkdir(parents=True, exist_ok=True)
BRIDGE_PAIR_LEFT, BRIDGE_PAIR_RIGHT = pair_indices(ACTIONS_PER_STATE)
STAGE12_BOOTSTRAP_DRAW_ROWS = []


def bridge_sha256_array(value):
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def bridge_project_target_and_goals(environment, model_name):
    sample = load_training_state(environment, model_name, 0)
    input_dim = int(
        sample["true_tokens"].shape[-2]
        * sample["true_tokens"].shape[-1]
    )
    projector = CountSketchProjector(
        input_dim,
        BRIDGE_PROJECTION_DIM,
        BRIDGE_PROJECTION_SEED,
    )
    targets = []
    native_costs = []
    with np.load(GOAL_ROOT / f"{model_name}.npz") as goal_payload:
        raw_goals = goal_payload["visual"].astype(np.float32)
    with torch.inference_mode():
        goal_tensor = torch.as_tensor(
            raw_goals, device="cuda", dtype=torch.float32
        )
        projected_goals = projector(goal_tensor).cpu().numpy()
        for state_id in range(NUM_STATES):
            shard = load_training_state(
                environment, model_name, state_id
            )
            task_id = int(shard["task_id"])
            target = torch.as_tensor(
                shard["true_tokens"][:, HORIZON_INDICES],
                device="cuda",
                dtype=torch.float32,
            )
            actions, horizons, tokens, channels = target.shape
            projected = projector(
                target.reshape(actions * horizons, tokens, channels)
            ).reshape(
                actions, horizons, BRIDGE_PROJECTION_DIM
            )
            native = torch.sqrt(
                torch.mean(
                    (
                        target
                        - goal_tensor[task_id][None, None]
                    ).square(),
                    dim=(2, 3),
                ).clamp_min(1e-12)
            )
            targets.append(projected.cpu().numpy())
            native_costs.append(native.cpu().numpy())
    return {
        "target_projected": np.asarray(targets, dtype=np.float32),
        "goal_projected": np.asarray(
            projected_goals, dtype=np.float32
        ),
        "target_native_cost": np.asarray(
            native_costs, dtype=np.float64
        ),
        "projection_checksum": countsketch_checksum(projector),
    }


def fit_bridge_whitener(target_projected, goal_projected, truth):
    train_ids = split_indices(truth, "probe_train")
    train_tasks = sorted(
        set(int(value) for value in truth["task_id"][train_ids])
    )
    values = np.concatenate(
        [
            target_projected[train_ids].reshape(
                -1, BRIDGE_PROJECTION_DIM
            ),
            goal_projected[train_tasks],
        ],
        axis=0,
    ).astype(np.float64)
    mean = values.mean(axis=0)
    centered = values - mean[None]
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    mean_variance = float(
        np.trace(covariance) / BRIDGE_PROJECTION_DIM
    )
    ridge = max(
        1e-8,
        BRIDGE_WHITENING_RIDGE_FRACTION
        * max(mean_variance, 1e-8),
    )
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    regularized = np.maximum(eigenvalue + ridge, 1e-8)
    whitener = (
        eigenvector * regularized[None] ** -0.5
    ) @ eigenvector.T
    payload = {
        "mean": mean.astype(np.float64),
        "whitener": whitener.astype(np.float64),
        "ridge": float(ridge),
        "minimum_eigenvalue": float(eigenvalue.min()),
        "maximum_eigenvalue": float(eigenvalue.max()),
        "condition_after_ridge": float(
            regularized.max() / regularized.min()
        ),
        "fit_split": "probe_train",
        "uses_predicted_rollouts": False,
        "uses_development_outcomes": False,
    }
    return payload


def apply_bridge_whitener(values, whitening):
    values = np.asarray(values, dtype=np.float64)
    return (
        values - whitening["mean"]
    ) @ whitening["whitener"]


def normalized_true_costs(physical_cost):
    physical_cost = np.asarray(physical_cost, dtype=np.float64)
    minimum = physical_cost.min(axis=1, keepdims=True)
    spread = (
        physical_cost.max(axis=1, keepdims=True) - minimum
    )
    return (
        (physical_cost - minimum)
        / np.maximum(spread, BRIDGE_COST_RANGE_EPS)
    )


def deterministic_goal_derangement(task_ids, seed):
    task_ids = np.asarray(sorted(set(int(x) for x in task_ids)))
    rng = np.random.default_rng(seed)
    permuted = task_ids.copy()
    for _ in range(10000):
        rng.shuffle(permuted)
        if np.all(permuted != task_ids):
            return {
                int(source): int(target)
                for source, target in zip(task_ids, permuted)
            }
    raise RuntimeError("could not construct task-level goal derangement")


def bridge_metric_matrix_torch(low_rank):
    dimension = low_rank.shape[1]
    identity = torch.eye(
        dimension, device=low_rank.device, dtype=low_rank.dtype
    )
    unscaled = identity + low_rank.T @ low_rank
    return dimension * unscaled / torch.trace(unscaled)


def bridge_metric_matrix_numpy(low_rank):
    low_rank = np.asarray(low_rank, dtype=np.float64)
    dimension = low_rank.shape[1]
    unscaled = np.eye(dimension) + low_rank.T @ low_rank
    return dimension * unscaled / np.trace(unscaled)


def bridge_metric_costs_numpy(
    future_features, goal_features, metric, beta
):
    future_features = np.asarray(future_features, dtype=np.float64)
    goal_features = np.asarray(goal_features, dtype=np.float64)
    metric = np.asarray(metric, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    difference = future_features - goal_features[:, None, None, :]
    raw = np.einsum(
        "sahd,de,sahe->sah", difference, metric, difference
    )
    return raw * beta[None, None, :]


def project_bridge_low_rank(low_rank):
    with torch.no_grad():
        singular = torch.linalg.svdvals(low_rank)
        maximum = singular.max()
        limit = math.sqrt(
            (BRIDGE_CONDITION_LIMIT - 1.0)
            * BRIDGE_CONDITION_HEADROOM
        )
        if float(maximum) > limit:
            low_rank.mul_(limit / maximum)


def bridge_training_loss(
    low_rank,
    log_beta,
    future_features,
    goal_features,
    normalized_cost,
    state_task_ids,
    regularizer,
):
    metric = bridge_metric_matrix_torch(low_rank)
    beta = torch.exp(log_beta)
    difference = (
        future_features - goal_features[:, None, None, :]
    )
    raw_cost = torch.einsum(
        "sahd,de,sahe->sah", difference, metric, difference
    )
    predicted_margin = beta[None, None, :] * (
        raw_cost[:, BRIDGE_PAIR_LEFT]
        - raw_cost[:, BRIDGE_PAIR_RIGHT]
    )
    target_margin = (
        normalized_cost[:, BRIDGE_PAIR_LEFT]
        - normalized_cost[:, BRIDGE_PAIR_RIGHT]
    )
    weight = target_margin.abs() + BRIDGE_PAIR_WEIGHT_FLOOR
    pair_loss = torch_functional.smooth_l1_loss(
        predicted_margin,
        target_margin,
        beta=BRIDGE_HUBER_DELTA,
        reduction="none",
    )
    state_horizon_loss = (pair_loss * weight).sum(dim=1) / (
        weight.sum(dim=1).clamp_min(1e-12)
    )
    task_losses = []
    for task_id in torch.unique(state_task_ids):
        task_losses.append(
            state_horizon_loss[state_task_ids == task_id].mean()
        )
    data_loss = torch.stack(task_losses).mean()
    penalty = float(regularizer) * low_rank.square().sum()
    return data_loss + penalty, {
        "data_loss": data_loss,
        "penalty": penalty,
        "metric": metric,
        "beta": beta,
    }


def fit_one_shared_metric(
    environment,
    target_features,
    goal_features,
    normalized_cost,
    state_task_ids,
    rank,
    regularizer,
    optimization_seed,
    goal_permutation=None,
):
    control_name = (
        "goal_permuted" if goal_permutation is not None else "matched_goal"
    )
    checkpoint_path = BRIDGE_METRIC_DIR / (
        f"{environment.lower()}_{control_name}_r{rank}_"
        f"reg{regularizer:.0e}_seed{optimization_seed}_"
        f"{RUN_SIGNATURE[:12]}.pt"
    )
    source_digest = hashlib.sha256(
        (
            bridge_sha256_array(target_features)
            + bridge_sha256_array(goal_features)
            + bridge_sha256_array(normalized_cost)
            + bridge_sha256_array(state_task_ids)
            + json.dumps(goal_permutation, sort_keys=True)
        ).encode()
    ).hexdigest()
    expected = {
        "run_signature": RUN_SIGNATURE,
        "bridge_implementation_id": BRIDGE_IMPLEMENTATION_ID,
        "environment": environment,
        "rank": int(rank),
        "regularizer": float(regularizer),
        "optimization_seed": int(optimization_seed),
        "control_name": control_name,
        "source_digest": source_digest,
    }
    if checkpoint_path.exists():
        payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        for key, value in expected.items():
            if payload.get(key) != value:
                raise RuntimeError(
                    f"metric checkpoint mismatch {key}: "
                    f"{payload.get(key)} != {value}"
                )
        return payload

    torch.manual_seed(int(optimization_seed))
    np.random.seed(int(optimization_seed))
    device = torch.device("cuda")
    future = torch.as_tensor(
        target_features, device=device, dtype=torch.float32
    )
    goals_numpy = np.asarray(goal_features, dtype=np.float64)
    if goal_permutation is not None:
        goals_numpy = np.asarray(
            [
                goals_numpy[goal_permutation[int(task_id)]]
                for task_id in state_task_ids
            ]
        )
    else:
        goals_numpy = np.asarray(
            [goals_numpy[int(task_id)] for task_id in state_task_ids]
        )
    goals = torch.as_tensor(
        goals_numpy, device=device, dtype=torch.float32
    )
    costs = torch.as_tensor(
        normalized_cost, device=device, dtype=torch.float32
    )
    task_ids = torch.as_tensor(
        state_task_ids, device=device, dtype=torch.long
    )
    low_rank = torch.nn.Parameter(
        0.02
        * torch.randn(
            int(rank),
            BRIDGE_PROJECTION_DIM,
            device=device,
            dtype=torch.float32,
        )
    )
    log_beta = torch.nn.Parameter(
        torch.zeros(len(HORIZONS), device=device)
    )
    optimizer = torch.optim.Adam(
        [low_rank, log_beta], lr=BRIDGE_LEARNING_RATE
    )
    curve = []
    best_loss = float("inf")
    best_state = None
    stale = 0
    converged = False
    for epoch in range(1, BRIDGE_MAX_EPOCHS + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, pieces = bridge_training_loss(
            low_rank,
            log_beta,
            future,
            goals,
            costs,
            task_ids,
            regularizer,
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("shared metric loss became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([low_rank, log_beta], 5.0)
        optimizer.step()
        project_bridge_low_rank(low_rank)
        with torch.no_grad():
            log_beta.clamp_(-9.0, 9.0)
        value = float(loss.detach())
        relative = (
            (best_loss - value) / max(abs(best_loss), 1e-12)
            if np.isfinite(best_loss)
            else float("inf")
        )
        if value < best_loss:
            best_loss = value
            best_state = {
                "low_rank": low_rank.detach().cpu().clone(),
                "log_beta": log_beta.detach().cpu().clone(),
                "epoch": int(epoch),
            }
        if (
            epoch >= BRIDGE_MIN_EPOCHS
            and relative < BRIDGE_PLATEAU_RELATIVE_TOLERANCE
        ):
            stale += 1
        else:
            stale = 0
        if epoch == 1 or epoch % 10 == 0:
            curve.append(
                {
                    "epoch": int(epoch),
                    "loss": value,
                    "data_loss": float(pieces["data_loss"].detach()),
                    "penalty": float(pieces["penalty"].detach()),
                }
            )
        if (
            epoch >= BRIDGE_MIN_EPOCHS
            and stale >= BRIDGE_PLATEAU_PATIENCE
        ):
            converged = True
            break
    if best_state is None:
        raise RuntimeError("shared metric optimizer produced no state")
    matrix = bridge_metric_matrix_numpy(best_state["low_rank"].numpy())
    eigenvalue = np.linalg.eigvalsh(matrix)
    beta = np.exp(best_state["log_beta"].numpy()).astype(np.float64)
    payload = {
        **expected,
        "selected_epoch": int(best_state["epoch"]),
        "completed_epochs": int(epoch),
        "converged_before_max_epochs": bool(converged),
        "low_rank": best_state["low_rank"],
        "log_beta": best_state["log_beta"],
        "metric": torch.as_tensor(matrix, dtype=torch.float64),
        "beta": torch.as_tensor(beta, dtype=torch.float64),
        "minimum_eigenvalue": float(eigenvalue.min()),
        "maximum_eigenvalue": float(eigenvalue.max()),
        "condition_number": float(
            eigenvalue.max() / eigenvalue.min()
        ),
        "trace": float(np.trace(matrix)),
        "parameter_count": int(rank * BRIDGE_PROJECTION_DIM + len(HORIZONS)),
        "curve": curve,
        "uses_target_future_latents": True,
        "uses_encoded_goals": True,
        "uses_normalized_physical_cost_margins": True,
        "uses_predicted_rollouts": False,
        "uses_treatment_identity": False,
        "uses_candidate_identity_as_feature": False,
        "uses_development_outcomes": False,
        "uses_physical_decoder": False,
        "goal_permutation": goal_permutation,
    }
    atomic_torch_save(payload, checkpoint_path)
    return payload


def fit_positive_horizon_scales(raw_cost, normalized_cost, state_ids):
    raw_cost = np.asarray(raw_cost, dtype=np.float64)
    normalized_cost = np.asarray(normalized_cost, dtype=np.float64)
    scales = []
    for horizon_index in range(len(HORIZONS)):
        x = (
            raw_cost[state_ids][:, BRIDGE_PAIR_LEFT, horizon_index]
            - raw_cost[state_ids][:, BRIDGE_PAIR_RIGHT, horizon_index]
        ).reshape(-1)
        y = (
            normalized_cost[state_ids][
                :, BRIDGE_PAIR_LEFT, horizon_index
            ]
            - normalized_cost[state_ids][
                :, BRIDGE_PAIR_RIGHT, horizon_index
            ]
        ).reshape(-1)
        weight = np.abs(y) + BRIDGE_PAIR_WEIGHT_FLOOR
        numerator = float(np.sum(weight * x * y))
        denominator = float(np.sum(weight * x * x))
        scales.append(max(numerator / max(denominator, 1e-12), 1e-8))
    return np.asarray(scales, dtype=np.float64)


def bridge_state_metrics(true_cost, predicted_cost):
    true_cost = np.asarray(true_cost, dtype=np.float64)
    predicted_cost = np.asarray(predicted_cost, dtype=np.float64)
    ranking = ranking_metrics(true_cost, predicted_cost)
    normalized = (
        true_cost - true_cost.min()
    ) / max(
        true_cost.max() - true_cost.min(),
        BRIDGE_COST_RANGE_EPS,
    )
    true_margin = (
        normalized[BRIDGE_PAIR_LEFT]
        - normalized[BRIDGE_PAIR_RIGHT]
    )
    predicted_margin = (
        predicted_cost[BRIDGE_PAIR_LEFT]
        - predicted_cost[BRIDGE_PAIR_RIGHT]
    )
    denominator = np.sqrt(np.mean(true_margin**2))
    margin_rmse = (
        float(
            np.sqrt(np.mean((predicted_margin - true_margin) ** 2))
            / denominator
        )
        if denominator > 1e-12
        else 0.0
    )
    return {
        "normalized_regret": float(ranking["normalized_regret"]),
        "weighted_pairwise_accuracy": float(
            ranking["weighted_pairwise_accuracy"]
        ),
        "top1_correct": float(ranking["top1_correct"]),
        "normalized_margin_rmse": margin_rmse,
        "selected_action": int(ranking["selected_action"]),
        "oracle_action": int(ranking["oracle_action"]),
    }


def task_equal_horizon_summary(rows, split, planner, method):
    selected = [
        row for row in rows
        if row["split"] == split
        and row["planner"] == planner
        and row["method"] == method
    ]
    output = {}
    for horizon in HORIZONS:
        horizon_rows = [
            row for row in selected if int(row["horizon"]) == horizon
        ]
        metrics = {}
        for metric in [
            "normalized_regret",
            "weighted_pairwise_accuracy",
            "normalized_margin_rmse",
            "top1_correct",
        ]:
            by_task = {}
            for row in horizon_rows:
                by_task.setdefault(int(row["task_id"]), []).append(
                    float(row[metric])
                )
            metrics[metric] = float(
                np.mean(
                    [
                        np.mean(values)
                        for values in by_task.values()
                    ]
                )
            )
        output[int(horizon)] = metrics
    return output


def fit_and_select_bridge_metrics(environment, target_payload, truth):
    train_ids = split_indices(truth, "probe_train")
    calibration_ids = split_indices(truth, "probe_calibration")
    whitening = fit_bridge_whitener(
        target_payload["target_projected"],
        target_payload["goal_projected"],
        truth,
    )
    target_y = apply_bridge_whitener(
        target_payload["target_projected"], whitening
    )
    goal_y = apply_bridge_whitener(
        target_payload["goal_projected"], whitening
    )
    normalized_cost = normalized_true_costs(truth["physical_cost"])
    candidates = []
    for rank in BRIDGE_RANKS:
        for regularizer in BRIDGE_REGULARIZERS:
            for optimization_seed in BRIDGE_OPTIMIZATION_SEEDS:
                payload = fit_one_shared_metric(
                    environment,
                    target_y[train_ids],
                    goal_y,
                    normalized_cost[train_ids],
                    truth["task_id"][train_ids],
                    rank,
                    regularizer,
                    optimization_seed,
                )
                metric = payload["metric"].numpy()
                beta = payload["beta"].numpy()
                goals_by_state = goal_y[
                    truth["task_id"][calibration_ids].astype(int)
                ]
                predicted_cost = bridge_metric_costs_numpy(
                    target_y[calibration_ids],
                    goals_by_state,
                    metric,
                    beta,
                )
                values = []
                for local_index, state_id in enumerate(calibration_ids):
                    for horizon_index, horizon in enumerate(HORIZONS):
                        metrics = bridge_state_metrics(
                            truth["physical_cost"][
                                state_id, :, horizon_index
                            ],
                            predicted_cost[
                                local_index, :, horizon_index
                            ],
                        )
                        values.append(
                            {
                                "task_id": int(
                                    truth["task_id"][state_id]
                                ),
                                "horizon": int(horizon),
                                **metrics,
                            }
                        )
                score = task_equal_horizon_summary(
                    [
                        {
                            **row,
                            "split": "probe_calibration",
                            "planner": "candidate",
                            "method": "target_oracle",
                        }
                        for row in values
                    ],
                    "probe_calibration",
                    "candidate",
                    "target_oracle",
                )
                task_equal_margin_rmse = float(
                    np.mean(
                        [
                            score[h]["normalized_margin_rmse"]
                            for h in HORIZONS
                        ]
                    )
                )
                candidates.append(
                    {
                        "rank": int(rank),
                        "regularizer": float(regularizer),
                        "optimization_seed": int(optimization_seed),
                        "calibration_margin_rmse": (
                            task_equal_margin_rmse
                        ),
                        "checkpoint": payload,
                    }
                )
    candidates.sort(
        key=lambda row: (
            row["calibration_margin_rmse"],
            row["rank"],
            -row["regularizer"],
            row["optimization_seed"],
        )
    )
    selected = candidates[0]
    train_tasks = sorted(
        set(int(x) for x in truth["task_id"][train_ids])
    )
    goal_permutation = deterministic_goal_derangement(
        train_tasks, BRIDGE_PROJECTION_SEED
    )
    permuted_payload = fit_one_shared_metric(
        environment,
        target_y[train_ids],
        goal_y,
        normalized_cost[train_ids],
        truth["task_id"][train_ids],
        selected["rank"],
        selected["regularizer"],
        selected["optimization_seed"],
        goal_permutation=goal_permutation,
    )
    native_scale = fit_positive_horizon_scales(
        target_payload["target_native_cost"],
        normalized_cost,
        train_ids,
    )
    selection_payload = {
        "environment": environment,
        "selection_split": "probe_calibration",
        "selection_metric": (
            "task_equal_horizon_mean_normalized_margin_rmse"
        ),
        "tie_break": [
            "lower_rank",
            "larger_regularizer",
            "lower_optimization_seed",
        ],
        "selected": {
            key: selected[key]
            for key in [
                "rank",
                "regularizer",
                "optimization_seed",
                "calibration_margin_rmse",
            ]
        },
        "all_candidates": [
            {
                key: row[key]
                for key in [
                    "rank",
                    "regularizer",
                    "optimization_seed",
                    "calibration_margin_rmse",
                ]
            }
            for row in candidates
        ],
        "goal_permutation": goal_permutation,
        "whitening": {
            key: value
            for key, value in whitening.items()
            if key not in {"mean", "whitener"}
        },
        "whitening_mean_checksum": bridge_sha256_array(
            whitening["mean"]
        ),
        "whitener_checksum": bridge_sha256_array(
            whitening["whitener"]
        ),
        "projection_checksum": target_payload[
            "projection_checksum"
        ],
        "native_horizon_scale": native_scale.tolist(),
    }
    write_json(
        BRIDGE_DIR / f"{environment.lower()}_metric_selection.json",
        selection_payload,
    )
    return {
        "target_y": target_y,
        "goal_y": goal_y,
        "whitening": whitening,
        "selected": selected["checkpoint"],
        "goal_permuted": permuted_payload,
        "native_scale": native_scale,
        "selection": selection_payload,
    }


def add_bridge_rows(
    rows,
    environment,
    truth,
    split,
    planner,
    method,
    adaptation_seed,
    predicted_cost,
    diagnostics=None,
):
    state_ids = split_indices(truth, split)
    for state_id in state_ids:
        for horizon_index, horizon in enumerate(HORIZONS):
            metrics = bridge_state_metrics(
                truth["physical_cost"][state_id, :, horizon_index],
                predicted_cost[state_id, :, horizon_index],
            )
            true_cost = np.asarray(
                truth["physical_cost"][state_id, :, horizon_index],
                dtype=np.float64,
            )
            planner_cost = np.asarray(
                predicted_cost[state_id, :, horizon_index],
                dtype=np.float64,
            )
            true_margin = (
                true_cost[BRIDGE_PAIR_LEFT]
                - true_cost[BRIDGE_PAIR_RIGHT]
            )
            predicted_margin = (
                planner_cost[BRIDGE_PAIR_LEFT]
                - planner_cost[BRIDGE_PAIR_RIGHT]
            )
            sorted_cost = np.sort(true_cost)
            diagnostic = (
                diagnostics[state_id, horizon_index]
                if diagnostics is not None
                else {}
            )
            rows.append(
                {
                    "environment": environment,
                    "state_id": int(state_id),
                    "task_id": int(truth["task_id"][state_id]),
                    "split": split,
                    "planner": planner,
                    "method": method,
                    "adaptation_seed": int(adaptation_seed),
                    "horizon": int(horizon),
                    "true_cost_json": json.dumps(true_cost.tolist()),
                    "predicted_cost_json": json.dumps(
                        planner_cost.tolist()
                    ),
                    "true_margin_json": json.dumps(
                        true_margin.tolist()
                    ),
                    "predicted_margin_json": json.dumps(
                        predicted_margin.tolist()
                    ),
                    "physical_cost_range": float(
                        true_cost.max() - true_cost.min()
                    ),
                    "best_second_gap": float(
                        sorted_cost[1] - sorted_cost[0]
                    ),
                    "absolute_latent_rmse": float(
                        diagnostic.get("absolute_latent_rmse", np.nan)
                    ),
                    "centered_action_geometry_rmse": float(
                        diagnostic.get(
                            "centered_action_geometry_rmse", np.nan
                        )
                    ),
                    "common_mode_rmse": float(
                        diagnostic.get("common_mode_rmse", np.nan)
                    ),
                    "native_fidelity_mse": float(
                        diagnostic.get("native_fidelity_mse", np.nan)
                    ),
                    **metrics,
                }
            )


def evaluate_shared_bridge(
    environment, target_payload, metric_payload, truth
):
    rows = []
    target_y = metric_payload["target_y"]
    goal_y = metric_payload["goal_y"]
    goals_by_state = goal_y[truth["task_id"].astype(int)]
    selected_metric = metric_payload["selected"]["metric"].numpy()
    selected_beta = metric_payload["selected"]["beta"].numpy()
    permuted_metric = metric_payload[
        "goal_permuted"
    ]["metric"].numpy()
    permuted_beta = metric_payload[
        "goal_permuted"
    ]["beta"].numpy()
    target_shared = bridge_metric_costs_numpy(
        target_y, goals_by_state, selected_metric, selected_beta
    )
    target_permuted = bridge_metric_costs_numpy(
        target_y, goals_by_state, permuted_metric, permuted_beta
    )
    target_native = (
        target_payload["target_native_cost"]
        * metric_payload["native_scale"][None, None, :]
    )
    simulator_oracle = normalized_true_costs(truth["physical_cost"])
    for split in ["probe_calibration", DEVELOPMENT_SPLIT]:
        add_bridge_rows(
            rows,
            environment,
            truth,
            split,
            "shared_metric",
            "target_oracle",
            -1,
            target_shared,
        )
        add_bridge_rows(
            rows,
            environment,
            truth,
            split,
            "goal_permuted_metric",
            "target_oracle",
            -1,
            target_permuted,
        )
        add_bridge_rows(
            rows,
            environment,
            truth,
            split,
            "native_metric",
            "target_oracle",
            -1,
            target_native,
        )
        add_bridge_rows(
            rows,
            environment,
            truth,
            split,
            "simulator_oracle",
            "target_oracle",
            -1,
            simulator_oracle,
        )

    model_name = MODEL_BY_ENVIRONMENT[environment][0]
    for record in EVALUATION_VARIANT_RECORDS:
        loaded = load_variant_features(
            model_name, record["variant"], 0
        )
        projected_y = apply_bridge_whitener(
            loaded["projected"], metric_payload["whitening"]
        )
        shared_cost = bridge_metric_costs_numpy(
            projected_y,
            goals_by_state,
            selected_metric,
            selected_beta,
        )
        permuted_cost = bridge_metric_costs_numpy(
            projected_y,
            goals_by_state,
            permuted_metric,
            permuted_beta,
        )
        native_cost = (
            loaded["latent_cost"]
            * metric_payload["native_scale"][None, None, :]
        )
        error = projected_y - target_y
        common_mode = error.mean(axis=1)
        centered_error = error - common_mode[:, None]
        absolute_rmse = np.sqrt(np.mean(error**2, axis=(1, 3)))
        common_mode_rmse = np.sqrt(
            np.mean(common_mode**2, axis=2)
        )
        centered_rmse = np.sqrt(
            np.mean(centered_error**2, axis=(1, 3))
        )
        diagnostics = np.empty(
            (NUM_STATES, len(HORIZONS)), dtype=object
        )
        for state_id in range(NUM_STATES):
            for horizon_index in range(len(HORIZONS)):
                diagnostics[state_id, horizon_index] = {
                    "absolute_latent_rmse": float(
                        absolute_rmse[state_id, horizon_index]
                    ),
                    "centered_action_geometry_rmse": float(
                        centered_rmse[state_id, horizon_index]
                    ),
                    "common_mode_rmse": float(
                        common_mode_rmse[state_id, horizon_index]
                    ),
                    "native_fidelity_mse": float(
                        loaded["native_horizon"][
                            state_id, horizon_index
                        ]
                    ),
                }
        for split in [DEVELOPMENT_SPLIT]:
            add_bridge_rows(
                rows,
                environment,
                truth,
                split,
                "shared_metric",
                record["method"],
                record["adaptation_seed"],
                shared_cost,
                diagnostics,
            )
            add_bridge_rows(
                rows,
                environment,
                truth,
                split,
                "goal_permuted_metric",
                record["method"],
                record["adaptation_seed"],
                permuted_cost,
                diagnostics,
            )
            add_bridge_rows(
                rows,
                environment,
                truth,
                split,
                "native_metric",
                record["method"],
                record["adaptation_seed"],
                native_cost,
                diagnostics,
            )
    return rows


def collapse_seed_rows(rows):
    grouped = {}
    identity = [
        "environment",
        "state_id",
        "task_id",
        "split",
        "planner",
        "method",
        "horizon",
    ]
    metric_keys = [
        "normalized_regret",
        "weighted_pairwise_accuracy",
        "top1_correct",
        "normalized_margin_rmse",
    ]
    for row in rows:
        key = tuple(row[name] for name in identity)
        grouped.setdefault(key, []).append(row)
    collapsed = []
    for key, values in grouped.items():
        item = dict(zip(identity, key))
        for metric in metric_keys:
            item[metric] = float(
                np.mean([float(row[metric]) for row in values])
            )
        item["transition_seeds_averaged"] = int(len(values))
        collapsed.append(item)
    return collapsed


def bridge_contrast(
    collapsed,
    environment,
    planner,
    baseline,
    horizon,
):
    matched = {
        int(row["state_id"]): row
        for row in collapsed
        if row["environment"] == environment
        and row["planner"] == planner
        and row["method"]
        == "fidelity_constrained_matched_geometry"
        and int(row["horizon"]) == int(horizon)
        and row["split"] == DEVELOPMENT_SPLIT
    }
    control = {
        int(row["state_id"]): row
        for row in collapsed
        if row["environment"] == environment
        and row["planner"] == planner
        and row["method"] == baseline
        and int(row["horizon"]) == int(horizon)
        and row["split"] == DEVELOPMENT_SPLIT
    }
    if set(matched) != set(control):
        raise RuntimeError(
            f"unpaired Stage 12 contrast {environment} {planner} "
            f"{baseline} h{horizon}"
        )
    result = {}
    for output_name, matched_key, sign in [
        ("delta_regret", "normalized_regret", -1.0),
        (
            "delta_weighted_accuracy",
            "weighted_pairwise_accuracy",
            1.0,
        ),
    ]:
        values = []
        for state_id in matched:
            difference = sign * (
                float(matched[state_id][matched_key])
                - float(control[state_id][matched_key])
            )
            values.append(
                {
                    "task_id": int(matched[state_id]["task_id"]),
                    output_name: difference,
                }
            )
        bootstrap_seed = (
            BRIDGE_BOOTSTRAP_SEED
            + 101 * ENVIRONMENT.index(environment)
            + 17 * HORIZONS.index(horizon)
            + 5
            * [
                "frozen",
                "fidelity_constrained_shuffled_geometry",
                "fidelity_constrained_latent_only",
            ].index(baseline)
        )
        result[output_name] = bootstrap_equal_task_mean(
            values,
            output_name,
            BRIDGE_BOOTSTRAP_REPS,
            bootstrap_seed,
        )
        task_means = json.loads(
            result[output_name]["task_means_json"]
        )
        task_ids = sorted(task_means, key=int)
        task_values = np.asarray(
            [task_means[task_id] for task_id in task_ids],
            dtype=np.float64,
        )
        rng = np.random.default_rng(bootstrap_seed)
        for draw_index in range(BRIDGE_BOOTSTRAP_REPS):
            sampled = rng.integers(
                0, len(task_values), size=len(task_values)
            )
            STAGE12_BOOTSTRAP_DRAW_ROWS.append(
                {
                    "environment": environment,
                    "planner": planner,
                    "baseline": baseline,
                    "horizon": int(horizon),
                    "metric": output_name,
                    "bootstrap_seed": int(bootstrap_seed),
                    "draw_index": int(draw_index),
                    "estimate": float(
                        np.mean(task_values[sampled])
                    ),
                }
            )
    return result


def evaluate_phase_a(rows, environment):
    summary = {}
    for planner in [
        "shared_metric",
        "native_metric",
        "goal_permuted_metric",
    ]:
        summary[planner] = task_equal_horizon_summary(
            rows,
            "probe_calibration",
            planner,
            "target_oracle",
        )
    margin_native = np.mean(
        [
            summary["native_metric"][h]["normalized_margin_rmse"]
            for h in HORIZONS
        ]
    )
    margin_shared = np.mean(
        [
            summary["shared_metric"][h]["normalized_margin_rmse"]
            for h in HORIZONS
        ]
    )
    margin_improvement = (
        1.0 - margin_shared / max(margin_native, 1e-12)
    )
    horizon_pass = {}
    for horizon in HORIZONS:
        shared = summary["shared_metric"][horizon]
        native = summary["native_metric"][horizon]
        horizon_pass[horizon] = bool(
            shared["normalized_regret"]
            <= min(
                0.10,
                0.75 * native["normalized_regret"],
            )
            + 1e-12
            and shared["weighted_pairwise_accuracy"]
            >= max(
                0.80,
                native["weighted_pairwise_accuracy"] + 0.03,
            )
            - 1e-12
        )
    shared_regret = np.mean(
        [
            summary["shared_metric"][h]["normalized_regret"]
            for h in HORIZONS
        ]
    )
    permuted_regret = np.mean(
        [
            summary["goal_permuted_metric"][h][
                "normalized_regret"
            ]
            for h in HORIZONS
        ]
    )
    shared_accuracy = np.mean(
        [
            summary["shared_metric"][h][
                "weighted_pairwise_accuracy"
            ]
            for h in HORIZONS
        ]
    )
    permuted_accuracy = np.mean(
        [
            summary["goal_permuted_metric"][h][
                "weighted_pairwise_accuracy"
            ]
            for h in HORIZONS
        ]
    )
    goal_specificity = bool(
        permuted_regret - shared_regret >= 0.02
        and shared_accuracy - permuted_accuracy >= 0.02
    )
    passed = bool(
        margin_improvement >= 0.20
        and sum(horizon_pass.values()) >= 2
        and goal_specificity
    )
    return {
        "environment": environment,
        "passed": passed,
        "margin_rmse_relative_improvement": float(
            margin_improvement
        ),
        "horizon_pass": horizon_pass,
        "goal_specificity_pass": goal_specificity,
        "goal_specificity_regret_gain": float(
            permuted_regret - shared_regret
        ),
        "goal_specificity_accuracy_gain": float(
            shared_accuracy - permuted_accuracy
        ),
        "summary": summary,
    }


def task_majority_gate(collapsed, environment, baseline):
    selected = [
        row
        for row in collapsed
        if row["environment"] == environment
        and row["planner"] == "shared_metric"
        and row["split"] == DEVELOPMENT_SPLIT
        and row["method"]
        in {
            "fidelity_constrained_matched_geometry",
            baseline,
        }
    ]
    by_key = {}
    for row in selected:
        key = (
            int(row["task_id"]),
            int(row["state_id"]),
            int(row["horizon"]),
            row["method"],
        )
        by_key[key] = row
    tasks = sorted(set(int(row["task_id"]) for row in selected))
    task_pass = {}
    for task_id in tasks:
        regrets = []
        accuracies = []
        for state_id in sorted(
            set(
                int(row["state_id"])
                for row in selected
                if int(row["task_id"]) == task_id
            )
        ):
            for horizon in HORIZONS:
                matched = by_key[
                    (
                        task_id,
                        state_id,
                        horizon,
                        "fidelity_constrained_matched_geometry",
                    )
                ]
                control = by_key[
                    (task_id, state_id, horizon, baseline)
                ]
                regrets.append(
                    float(control["normalized_regret"])
                    - float(matched["normalized_regret"])
                )
                accuracies.append(
                    float(matched["weighted_pairwise_accuracy"])
                    - float(control["weighted_pairwise_accuracy"])
                )
        task_pass[task_id] = bool(
            np.mean(regrets) > 0 and np.mean(accuracies) > 0
        )
    return {
        "task_pass": task_pass,
        "passing_tasks": int(sum(task_pass.values())),
        "passed": bool(sum(task_pass.values()) >= 2),
    }


def planner_nonharm_gate(collapsed, environment):
    result = {}
    for horizon in HORIZONS:
        matched = [
            row
            for row in collapsed
            if row["environment"] == environment
            and row["planner"] == "shared_metric"
            and row["method"]
            == "fidelity_constrained_matched_geometry"
            and int(row["horizon"]) == horizon
            and row["split"] == DEVELOPMENT_SPLIT
        ]
        frozen_native = [
            row
            for row in collapsed
            if row["environment"] == environment
            and row["planner"] == "native_metric"
            and row["method"] == "frozen"
            and int(row["horizon"]) == horizon
            and row["split"] == DEVELOPMENT_SPLIT
        ]
        def equal_task_mean(source, key):
            grouped = {}
            for row in source:
                grouped.setdefault(int(row["task_id"]), []).append(
                    float(row[key])
                )
            return float(
                np.mean(
                    [
                        np.mean(values)
                        for values in grouped.values()
                    ]
                )
            )

        matched_regret = equal_task_mean(
            matched, "normalized_regret"
        )
        frozen_regret = equal_task_mean(
            frozen_native, "normalized_regret"
        )
        matched_accuracy = equal_task_mean(
            matched, "weighted_pairwise_accuracy"
        )
        frozen_accuracy = equal_task_mean(
            frozen_native, "weighted_pairwise_accuracy"
        )
        result[horizon] = {
            "regret_harm": float(matched_regret - frozen_regret),
            "accuracy_harm": float(
                frozen_accuracy - matched_accuracy
            ),
            "passed": bool(
                matched_regret - frozen_regret <= 0.02 + 1e-12
                and frozen_accuracy - matched_accuracy
                <= 0.02 + 1e-12
            ),
        }
    return {
        "horizons": result,
        "passed": all(item["passed"] for item in result.values()),
    }


def evaluate_phase_b(
    collapsed, environment, selected_metric_payload
):
    contrasts = {}
    for planner in ["shared_metric", "goal_permuted_metric"]:
        contrasts[planner] = {}
        for baseline in [
            "frozen",
            "fidelity_constrained_shuffled_geometry",
            "fidelity_constrained_latent_only",
        ]:
            contrasts[planner][baseline] = {
                horizon: bridge_contrast(
                    collapsed,
                    environment,
                    planner,
                    baseline,
                    horizon,
                )
                for horizon in HORIZONS
            }
    common_horizons = []
    for horizon in HORIZONS:
        strong = all(
            contrasts["shared_metric"][baseline][horizon][
                "delta_regret"
            ]["estimate"]
            >= 0.015
            and contrasts["shared_metric"][baseline][horizon][
                "delta_weighted_accuracy"
            ]["estimate"]
            >= 0.01
            for baseline in [
                "frozen",
                "fidelity_constrained_shuffled_geometry",
            ]
        )
        latent_directional = bool(
            contrasts["shared_metric"][
                "fidelity_constrained_latent_only"
            ][horizon]["delta_regret"]["estimate"]
            > 0
            and contrasts["shared_metric"][
                "fidelity_constrained_latent_only"
            ][horizon]["delta_weighted_accuracy"]["estimate"]
            > 0
        )
        if strong and latent_directional:
            common_horizons.append(horizon)
    task_majority = {
        baseline: task_majority_gate(
            collapsed, environment, baseline
        )
        for baseline in [
            "frozen",
            "fidelity_constrained_shuffled_geometry",
        ]
    }
    true_gain = {
        metric: float(
            np.mean(
                [
                    contrasts["shared_metric"]["frozen"][h][metric][
                        "estimate"
                    ]
                    for h in HORIZONS
                ]
            )
        )
        for metric in [
            "delta_regret",
            "delta_weighted_accuracy",
        ]
    }
    permuted_gain = {
        metric: float(
            np.mean(
                [
                    contrasts["goal_permuted_metric"]["frozen"][h][
                        metric
                    ]["estimate"]
                    for h in HORIZONS
                ]
            )
        )
        for metric in [
            "delta_regret",
            "delta_weighted_accuracy",
        ]
    }
    specificity = {
        metric: true_gain[metric] - permuted_gain[metric]
        for metric in true_gain
    }
    specificity_pass = all(
        value >= 0.01 for value in specificity.values()
    )
    nonharm = planner_nonharm_gate(collapsed, environment)
    metric_condition = float(
        selected_metric_payload["condition_number"]
    )
    metric_interior = bool(
        metric_condition
        < BRIDGE_CONDITION_LIMIT * 0.99
        and selected_metric_payload["converged_before_max_epochs"]
    )
    matched_manifest = [
        row
        for row in STAGE12_ADAPTATION_MANIFEST
        if row["environment"] == environment
        and row["method"]
        == "fidelity_constrained_matched_geometry"
    ]
    fidelity_pass = bool(
        len(matched_manifest) == len(ADAPTATION_SEEDS)
        and all(
            row["fidelity_feasible"]
            and all(
                float(value)
                <= 1.0 + NATIVE_NONINFERIORITY_TOLERANCE + 1e-7
                for value in row["calibration_native_ratios"]
            )
            for row in matched_manifest
        )
    )
    checkpoint_count = sum(
        1
        for row in STAGE12_ADAPTATION_MANIFEST
        if row["environment"] == environment
    )
    checkpoint_pass = checkpoint_count == (
        len(ADAPTATION_METHODS) * len(ADAPTATION_SEEDS)
    )
    passed = bool(
        len(common_horizons) >= 2
        and all(item["passed"] for item in task_majority.values())
        and specificity_pass
        and nonharm["passed"]
        and metric_interior
        and fidelity_pass
        and checkpoint_pass
    )
    return {
        "environment": environment,
        "passed": passed,
        "common_strong_and_latent_directional_horizons": (
            common_horizons
        ),
        "contrasts": contrasts,
        "task_majority": task_majority,
        "goal_specificity_gain": specificity,
        "goal_specificity_pass": specificity_pass,
        "complete_planner_nonharm": nonharm,
        "metric_condition_number": metric_condition,
        "metric_optimizer_and_condition_interior": metric_interior,
        "native_fidelity_pass": fidelity_pass,
        "transition_checkpoint_count": checkpoint_count,
        "transition_checkpoint_integrity_pass": checkpoint_pass,
    }


def write_bridge_margin_rows(rows):
    raw_rows = []
    for row in rows:
        raw_rows.append(
            {
                key: row[key]
                for key in [
                    "environment",
                    "state_id",
                    "task_id",
                    "split",
                    "planner",
                    "method",
                    "adaptation_seed",
                    "horizon",
                    "normalized_regret",
                    "weighted_pairwise_accuracy",
                    "top1_correct",
                    "normalized_margin_rmse",
                    "selected_action",
                    "oracle_action",
                    "true_cost_json",
                    "predicted_cost_json",
                    "true_margin_json",
                    "predicted_margin_json",
                    "physical_cost_range",
                    "best_second_gap",
                    "absolute_latent_rmse",
                    "centered_action_geometry_rmse",
                    "common_mode_rmse",
                    "native_fidelity_mse",
                ]
            }
        )
    write_csv(BRIDGE_DIR / "stage12_unit_metrics.csv", raw_rows)


def plot_bridge_summary(collapsed):
    methods = [
        "frozen",
        "fidelity_constrained_latent_only",
        "fidelity_constrained_shuffled_geometry",
        "fidelity_constrained_matched_geometry",
    ]
    labels = ["Frozen", "Latent only", "Shuffled", "Matched"]
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
                        row[metric]
                        for row in collapsed
                        if row["environment"] == environment
                        and row["planner"] == "shared_metric"
                        and row["method"] == method
                        and int(row["horizon"]) == horizon
                        and row["split"] == DEVELOPMENT_SPLIT
                    ]
                    values.append(float(np.mean(selected)))
                axis.plot(HORIZONS, values, marker="o", label=label)
            axis.set_title(f"{environment}: {metric}")
            axis.grid(alpha=0.25)
            axis.set_xlabel("horizon")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(
        PLOT_DIR / "stage12_shared_metric_bridge.png", dpi=180
    )
    plt.close(fig)


if not PIPELINE_FAILED:
    try:
        STAGE12_BRIDGE_ROWS = []
        STAGE12_METRIC_PAYLOADS = {}
        STAGE12_PHASE_A = {}
        for environment in ENVIRONMENT:
            model_name = MODEL_BY_ENVIRONMENT[environment][0]
            truth = load_truth_evaluation(environment)
            target_payload = bridge_project_target_and_goals(
                environment, model_name
            )
            metric_payload = fit_and_select_bridge_metrics(
                environment, target_payload, truth
            )
            STAGE12_METRIC_PAYLOADS[environment] = metric_payload
            environment_rows = evaluate_shared_bridge(
                environment, target_payload, metric_payload, truth
            )
            STAGE12_BRIDGE_ROWS.extend(environment_rows)
            STAGE12_PHASE_A[environment] = evaluate_phase_a(
                environment_rows, environment
            )
            clear_training_state_memory_cache(environment)
            gc.collect()
            torch.cuda.empty_cache()

        write_bridge_margin_rows(STAGE12_BRIDGE_ROWS)
        STAGE12_COLLAPSED_ROWS = collapse_seed_rows(
            STAGE12_BRIDGE_ROWS
        )
        write_csv(
            BRIDGE_DIR / "stage12_seed_collapsed_metrics.csv",
            STAGE12_COLLAPSED_ROWS,
        )
        STAGE12_PHASE_B = {
            environment: evaluate_phase_b(
                STAGE12_COLLAPSED_ROWS,
                environment,
                STAGE12_METRIC_PAYLOADS[environment]["selected"],
            )
            for environment in ENVIRONMENT
        }
        write_csv(
            BRIDGE_DIR / "stage12_bootstrap_draws.csv",
            STAGE12_BOOTSTRAP_DRAW_ROWS,
        )
        phase_a_pass = all(
            item["passed"] for item in STAGE12_PHASE_A.values()
        )
        phase_b_pass = all(
            item["passed"] for item in STAGE12_PHASE_B.values()
        )
        one_environment_pass = any(
            item["passed"] for item in STAGE12_PHASE_B.values()
        )
        if not phase_a_pass:
            decision = "STOP_METRIC_CLASS_NOT_VIABLE"
        elif phase_b_pass:
            decision = "PROMOTE_TO_UNTOUCHED_TASK_CONFIRMATION"
        elif one_environment_pass:
            decision = (
                "AMBIGUOUS_DO_NOT_TUNE_ON_DEVELOPMENT_TASKS"
            )
        else:
            decision = "STOP_NO_CAUSAL_BRIDGE_SIGNAL"
        STAGE12_DECISION = {
            "run_signature": RUN_SIGNATURE,
            "evidence_status": (
                "DEVELOPMENT_PILOT_NOT_CONFIRMATORY_EVIDENCE"
            ),
            "decision": decision,
            "phase_a_metric_class_viability": STAGE12_PHASE_A,
            "phase_b_causal_bridge": STAGE12_PHASE_B,
            "promotion_authorizes_only": (
                "frozen-recipe untouched-task confirmation"
            ),
            "prohibited_response": (
                "Do not tune this metric or ARGA again on the three "
                "development tasks after an ambiguous or failed result."
            ),
        }
        write_json(
            BRIDGE_DIR / "stage12_pilot_gate.json",
            STAGE12_DECISION,
        )
        plot_bridge_summary(STAGE12_COLLAPSED_ROWS)
        print(json.dumps(STAGE12_DECISION, indent=2))
    except Exception:
        record_failure("stage12_shared_target_metric_bridge")
'''

package = r'''# Phase H — package evidence, every transition checkpoint, and download.


def package_stage12_results():
    result_zip = Path("/content/stage12_result_bundle.zip")
    if not PIPELINE_FAILED:
        (OUT / "FAILURE_TRACE.txt").write_text(
            "SUCCESS: no captured pipeline failure\n"
        )
    excluded_roots = {
        str(INTERMEDIATE.resolve()),
        str(CACHE_ROOT.resolve()),
    }
    manifest_path = OUT / "stage12_result_zip_manifest.json"
    files = []
    for path in OUT.rglob("*"):
        if not path.is_file() or path == manifest_path:
            continue
        resolved = str(path.resolve())
        if any(
            resolved == root or resolved.startswith(root + os.sep)
            for root in excluded_roots
        ):
            continue
        if (
            not INCLUDE_ALL_TRANSITION_CHECKPOINTS_IN_ZIP
            and str(path.resolve()).startswith(
                str(ADAPTED_DIR.resolve()) + os.sep
            )
        ):
            continue
        files.append(path)
    if PIPELINE_FAILED:
        files.extend(
            path
            for path in ADAPTED_DIR.glob("*_latest.pt")
            if RUN_SIGNATURE[:12] in path.name
        )
    files = sorted(set(files))
    manifest = [
        {
            "path": str(path.relative_to(OUT)),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    write_json(
        manifest_path,
        {
            "run_signature": RUN_SIGNATURE,
            "pipeline_failed": bool(PIPELINE_FAILED),
            "all_transition_checkpoints_included": bool(
                INCLUDE_ALL_TRANSITION_CHECKPOINTS_IN_ZIP
            ),
            "files": manifest,
        },
    )
    files = sorted(
        {
            *files,
            manifest_path,
            OUT / "FAILURE_TRACE.txt",
        }
    )
    if result_zip.exists():
        result_zip.unlink()
    with zipfile.ZipFile(
        result_zip, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in files:
            archive.write(path, arcname=str(path.relative_to(OUT)))
    print("Stage 12 result bundle:", result_zip)
    print("Stage 12 result bundle SHA256:", sha256_file(result_zip))
    print("RUN_STATUS:", "FAILED" if PIPELINE_FAILED else "SUCCESS")
    if DOWNLOAD_STAGE12_RESULTS:
        try:
            from google.colab import files as colab_files

            colab_files.download(str(result_zip))
        except Exception as download_error:
            print(
                "Automatic download unavailable; use the Colab Files "
                "pane.",
                download_error,
            )
    return result_zip


STAGE12_RESULT_ZIP = package_stage12_results()
'''

cells = [
    code(config),
    markdown(intro),
    transformed_cell(2),
    setup,
    transformed_cell(4),
    transformed_cell(5),
    transformed_cell(6),
    transformed_cell(7),
    transformed_cell(8),
    transformed_cell(9),
    markdown(bridge_intro),
    code(bridge),
    code(package),
]
for index, cell in enumerate(cells):
    cell["id"] = f"stage12-{index:02d}"
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

metadata = json.loads(json.dumps(base["metadata"]))
metadata.setdefault("colab", {})["name"] = TARGET.name

notebook = {
    "cells": cells,
    "metadata": metadata,
    "nbformat": 4,
    "nbformat_minor": 5,
}
TARGET.write_text(json.dumps(notebook, indent=1) + "\n")
print(TARGET)
