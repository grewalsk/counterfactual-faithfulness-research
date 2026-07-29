import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "10_fidelity_constrained_pairwise_margin_adaptation.ipynb"
TARGET = ROOT / "11_action_response_geometry_pilot.ipynb"


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

config = '''# SINGLE CONFIGURATION BLOCK — the defaults are the compute-gated pilot.
RUN_MODE = "pilot"  # pilot or full; do not use full until the pilot promotes
OUTPUT_DIR = "/content/counterfactual_faithfulness_stage11"
SEED = 913
MODEL_NAME = ["jepa_wm_pusht", "jepa_wm_wall"]
ENVIRONMENT = ["PushT", "Wall"]
HORIZONS = [1, 3, 6]
NUM_STATES = 36  # 3 exact states for each of 12 tasks, per environment
ACTIONS_PER_STATE = 10

# The pilot stays on local Colab disk and downloads two compact ZIPs.
MOUNT_DRIVE = False
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/counterfactual_faithfulness_stage11"
REUSE_STAGE7_CACHE = True
STAGE7_OUTPUT_DIR = "/content/counterfactual_faithfulness_stage7"
STAGE7_DRIVE_OUTPUT_DIR = (
    "/content/drive/MyDrive/counterfactual_faithfulness_stage7"
)
DOWNLOAD_PHASE_C_RESCUE = True
DOWNLOAD_RESULTS = True

REPO_URL = "https://github.com/facebookresearch/jepa-wms.git"
REPO_COMMIT = "13cf1d9c7e476f53c17714d2e0f1dc239a883ce0"
EXPECTED_HF_REVISION = "9b9c41ef249466630dbf1a20e78391865d07b3b9"
EXPECTED_PRETRAINED_ASSET_SHA256 = {
    "jepa_wm_pusht.pth.tar": (
        "9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb"
    ),
    "jepa_wm_wall.pth.tar": (
        "8efb0623cfba1cb3ca210de26f7579c83dd24936635f11989c515afcb23bea1e"
    ),
    "dinov2_vits14_pretrain.pth": (
        "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
    ),
}
FRAMESKIP = 5
TARGET_STEPS = list(range(1, max(HORIZONS) + 1))
TASKS_PER_ENVIRONMENT = 12
TASK_SPLIT_COUNTS = [6, 3, 0, 3]
EVALUATION_SEEDS = [913, 1297, 1709]
BOOTSTRAP_REPS = 500
RANKING_TIE = 1e-9
RIDGE_LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]

# Cache compatibility with Stage 7.
AUDIT_PROJECTION_DIM = 64
AUDIT_PROJECTION_SEEDS = [7101]
GOAL_PROJECTION_DIM = 32
GOAL_PROJECTION_SEEDS = [8101, 10101]
ENERGY_HEAD_SEEDS = [8301]
ENERGY_METHODS = []
ENERGY_HIDDEN_DIM = 96
ENERGY_DROPOUT = 0.10
ENERGY_IMPLEMENTATION_ID = "unused_stage11_compatibility"
TRAINING_BATCH_STATES = 1
TRAINING_LR = 3e-4
TRAINING_WEIGHT_DECAY = 1e-3
PAIRWISE_WEIGHT = 1.0
LISTWISE_WEIGHT = 1.0
COST_SHAPE_WEIGHT = 0.25
PAIRWISE_TEMPERATURE = 0.25
LISTWISE_TEMPERATURE = 0.20

# Stage 11: readout-free Action-Response Geometry Adaptation (ARGA).
METHOD_IMPLEMENTATION_ID = "stage11_arga_whitened_centered_v1"
ADAPTATION_METHODS = [
    "fidelity_constrained_latent_only",
    "fidelity_constrained_shuffled_geometry",
    "fidelity_constrained_matched_geometry",
]
SCREENING_ADAPTATION_SEEDS = [11401]
CONFIRMATION_ADAPTATION_SEEDS = [11419]
ADAPTATION_SEEDS = [
    *SCREENING_ADAPTATION_SEEDS,
    *CONFIRMATION_ADAPTATION_SEEDS,
]
INITIAL_EPOCH_LIMIT = 6
EXTENSION_EPOCH_LIMITS = [8, 10]
CHECKPOINT_EVERY = 2
EARLY_STOPPING_CHECKPOINT_PATIENCE = 2
ADAPTATION_LR = 1e-5
ADAPTATION_WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
USE_ACTIVATION_CHECKPOINTING = True

TRAINING_GEOMETRY_PROJECTION_DIM = 96
TRAINING_GEOMETRY_PROJECTION_SEEDS = [13001, 13019]
GEOMETRY_WHITENING_RIDGE_FRACTION = 0.01
GEOMETRY_WHITENING_EIGEN_FLOOR = 1e-5
SCREENING_DIRECT_MIN_RELATIVE_GAIN = 0.03
SCREENING_LABEL_SPECIFICITY_GAIN = 0.01
SCREENING_MAX_OTHER_ENVIRONMENT_HARM = 0.05

# These physical decoders are evaluation-only and never enter the loss.
TRAINING_DECODER_COUNT = 2
TRAINING_DECODER_PROJECTION_DIM = 96
TRAINING_DECODER_SEEDS = [13501, 13519]
EVALUATION_PROJECTION_DIM = 72
EVALUATION_PROJECTION_SEEDS = [14011, 14029, 14047]
RIDGE_INITIAL_EXPONENTS = [-8, -6, -4, -2, 0, 2, 4]
RIDGE_EXPANSION_STEPS = 4

# Compatibility constants used only by evaluation certificates.
PAIRWISE_P = 8
PAIRWISE_SMOOTH_EPS = 1e-6
COST_SCALE_EPS = 1e-6
MIN_NORMALIZED_GAP = 0.02
TOP1_CERTIFICATE_WEIGHT = 0.25
NATIVE_NONINFERIORITY_TOLERANCE = 0.02
NATIVE_PLANNER_HARM_TOLERANCE = 0.02
AUGLAG_BETA = 10.0
PARAMETER_TRUST_RADIUS = 0.05
TRUST_REFERENCE_RMS_FLOOR = 1e-3
NATIVE_DENOMINATOR_EPS = 1e-8
BOUNDARY_IMPROVEMENT_TOLERANCE = 1e-4
DIRECT_PROJECTION_CONSENSUS_REQUIRED = 2
READOUT_PROJECTION_CONSENSUS_REQUIRED = 2
COLLAPSE_MAX_CANDIDATE_SHARE = 0.85
COLLAPSE_MAX_NULL_SHARE = 0.80
COLLAPSE_MIN_UNIQUE_ACTIONS = 2

EVIDENCE_STATUS = "COMPUTE_GATED_EXPLORATORY_PILOT"
TASK_FAMILY_ID = "stage5_tasks_reused_for_stage11_development"
DEVELOPMENT_SPLIT = "development_holdout"
STAGE7_TASK_FAMILY_ID = "stage5_tasks_reused_for_stage7_development"
STAGE7_DEVELOPMENT_SPLIT = "development_holdout"

if RUN_MODE == "full":
    NUM_STATES = 96
    AUDIT_PROJECTION_DIM = 128
    AUDIT_PROJECTION_SEEDS = [7101, 9101]
    CONFIRMATION_ADAPTATION_SEEDS = [11419, 11437]
    ADAPTATION_SEEDS = [
        *SCREENING_ADAPTATION_SEEDS,
        *CONFIRMATION_ADAPTATION_SEEDS,
    ]
    INITIAL_EPOCH_LIMIT = 16
    EXTENSION_EPOCH_LIMITS = [24, 32]
    EARLY_STOPPING_CHECKPOINT_PATIENCE = 3
    TRAINING_GEOMETRY_PROJECTION_DIM = 128
    TRAINING_DECODER_COUNT = 3
    TRAINING_DECODER_PROJECTION_DIM = 192
    TRAINING_DECODER_SEEDS = [13501, 13519, 13537]
    EVALUATION_PROJECTION_DIM = 128
    EVALUATION_PROJECTION_SEEDS = [
        14011, 14029, 14047, 14065, 14083
    ]
    DIRECT_PROJECTION_CONSENSUS_REQUIRED = 4
    READOUT_PROJECTION_CONSENSUS_REQUIRED = 4
    BOOTSTRAP_REPS = 2000
elif RUN_MODE != "pilot":
    raise ValueError("RUN_MODE must be 'pilot' or 'full'")

assert MODEL_NAME == ["jepa_wm_pusht", "jepa_wm_wall"]
assert ENVIRONMENT == ["PushT", "Wall"]
assert HORIZONS == [1, 3, 6]
assert TARGET_STEPS == [1, 2, 3, 4, 5, 6]
assert ACTIONS_PER_STATE == 10
assert NUM_STATES % TASKS_PER_ENVIRONMENT == 0
assert sum(TASK_SPLIT_COUNTS) == TASKS_PER_ENVIRONMENT
assert len(TRAINING_DECODER_SEEDS) == TRAINING_DECODER_COUNT
assert set(TRAINING_GEOMETRY_PROJECTION_SEEDS).isdisjoint(
    EVALUATION_PROJECTION_SEEDS
)
assert set(TRAINING_DECODER_SEEDS).isdisjoint(
    EVALUATION_PROJECTION_SEEDS
)
assert INITIAL_EPOCH_LIMIT % CHECKPOINT_EVERY == 0
assert all(
    limit % CHECKPOINT_EVERY == 0 for limit in EXTENSION_EPOCH_LIMITS
)
assert RUN_MODE != "pilot" or (
    NUM_STATES == 36
    and len(ADAPTATION_SEEDS) == 2
    and len(EVALUATION_PROJECTION_SEEDS) == 3
    and max(EXTENSION_EPOCH_LIMITS) == 10
)
'''

intro = r'''# Stage 11: compute-gated action-response geometry pilot

Stage 10 showed that directly optimizing a few physical readouts is not enough:
the matched treatment did not transfer to fresh readouts, and a shuffled
control could look equally good. Stage 11 tests a sharper hypothesis before
spending full-run compute:

> JEPA-WM recommends poor actions because its *relative latent response to
> alternative actions* is wrong, not merely because its absolute rollout is
> noisy or because a particular decision head is weak.

For state \(s\), horizon \(h\), candidate action \(a\), and fixed random
projection \(P_r\), define the centered action response

\[
g^\theta_{sahr}
=P_r\hat z^\theta_{sah}
-\frac{1}{A}\sum_b P_r\hat z^\theta_{sbh},
\qquad
g^*_{sahr}
=P_r z^*_{sah}
-\frac{1}{A}\sum_b P_r z^*_{sbh}.
\]

The matched objective minimizes

\[
\mathcal L_{\rm geom}
=\mathbb E_{s,a,h,r}
\left\|W_{hr}
\left(g^\theta_{sahr}-g^*_{sahr}\right)\right\|_2^2,
\]

where \(W_{hr}=(\widehat\Sigma_{hr}+\lambda I)^{-1/2}\) is fitted only
from target action responses on the probe-training split. Centering removes
common-mode rollout error. Whitening prevents high-variance latent directions
from dominating. No task goal, physical pose, cost, action label, or learned
readout enters this loss.

For any linear downstream score \(f_w(z)=w^\top z\), Cauchy--Schwarz gives

\[
\left|
\big(f_w(\hat z_a)-f_w(\hat z_b)\big)
-\big(f_w(z^*_a)-f_w(z^*_b)\big)
\right|
\le
\|\widehat\Sigma^{1/2}w\|_2
\left\|W\big[(g^\theta_a-g^\theta_b)
-(g^*_a-g^*_b)\big]\right\|_2.
\]

Thus the geometry error simultaneously controls pairwise margin error for
every bounded linear readout in the projected space; it is not tied to one
fitted decoder. This is a conditional guarantee, not a promise for nonlinear
planners or directions discarded by projection, so Stage 11 retains a small
absolute-latent fidelity constraint and evaluates three completely unseen
projections with newly fitted physical readouts.

The default run is deliberately a pilot: 36 states per environment, six
training tasks, three calibration tasks, three development tasks, ten
candidates, and at most ten epochs. One seed runs first. A second seed runs
only if matched geometry improves the calibration geometry over both frozen
JEPA and a within-state shuffled-correspondence control without damaging the
other environment. Failure of this necessary condition ends the experiment
early. Passing the pilot only authorizes the larger `RUN_MODE="full"` matrix;
it is not paper-level confirmation.
'''

setup = "".join(base["cells"][3]["source"])
setup = setup.replace("stage10", "stage11").replace("Stage 10", "Stage 11")
config_start = setup.index("CONFIG = {")
signature_start = setup.index("RUN_SIGNATURE =", config_start)
stage11_config = '''CONFIG = {
    "run_mode": RUN_MODE,
    "output_dir": str(OUT),
    "seed": SEED,
    "models": MODEL_NAME,
    "environments": ENVIRONMENT,
    "horizons": HORIZONS,
    "num_states": NUM_STATES,
    "actions_per_state": ACTIONS_PER_STATE,
    "frameskip": FRAMESKIP,
    "target_steps": TARGET_STEPS,
    "tasks_per_environment": TASKS_PER_ENVIRONMENT,
    "task_split_counts": TASK_SPLIT_COUNTS,
    "evaluation_seeds": EVALUATION_SEEDS,
    "bootstrap_reps": BOOTSTRAP_REPS,
    "ranking_tie": RANKING_TIE,
    "repo_url": REPO_URL,
    "repo_commit": REPO_COMMIT,
    "expected_hf_revision": EXPECTED_HF_REVISION,
    "expected_pretrained_asset_sha256": EXPECTED_PRETRAINED_ASSET_SHA256,
    "audit_projection_dim": AUDIT_PROJECTION_DIM,
    "audit_projection_seeds": AUDIT_PROJECTION_SEEDS,
    "goal_projection_dim": GOAL_PROJECTION_DIM,
    "goal_projection_seeds": GOAL_PROJECTION_SEEDS,
    "method_implementation_id": METHOD_IMPLEMENTATION_ID,
    "adaptation_methods": ADAPTATION_METHODS,
    "screening_adaptation_seeds": SCREENING_ADAPTATION_SEEDS,
    "confirmation_adaptation_seeds": CONFIRMATION_ADAPTATION_SEEDS,
    "adaptation_seeds": ADAPTATION_SEEDS,
    "initial_epoch_limit": INITIAL_EPOCH_LIMIT,
    "extension_epoch_limits": EXTENSION_EPOCH_LIMITS,
    "checkpoint_every": CHECKPOINT_EVERY,
    "early_stopping_checkpoint_patience": EARLY_STOPPING_CHECKPOINT_PATIENCE,
    "adaptation_lr": ADAPTATION_LR,
    "adaptation_weight_decay": ADAPTATION_WEIGHT_DECAY,
    "gradient_clip": GRADIENT_CLIP,
    "use_activation_checkpointing": USE_ACTIVATION_CHECKPOINTING,
    "training_geometry_projection_dim": TRAINING_GEOMETRY_PROJECTION_DIM,
    "training_geometry_projection_seeds": TRAINING_GEOMETRY_PROJECTION_SEEDS,
    "geometry_whitening_ridge_fraction": GEOMETRY_WHITENING_RIDGE_FRACTION,
    "geometry_whitening_eigen_floor": GEOMETRY_WHITENING_EIGEN_FLOOR,
    "screening_direct_min_relative_gain": (
        SCREENING_DIRECT_MIN_RELATIVE_GAIN
    ),
    "screening_label_specificity_gain": SCREENING_LABEL_SPECIFICITY_GAIN,
    "screening_max_other_environment_harm": (
        SCREENING_MAX_OTHER_ENVIRONMENT_HARM
    ),
    "training_decoder_count": TRAINING_DECODER_COUNT,
    "training_decoder_projection_dim": TRAINING_DECODER_PROJECTION_DIM,
    "training_decoder_seeds": TRAINING_DECODER_SEEDS,
    "evaluation_projection_dim": EVALUATION_PROJECTION_DIM,
    "evaluation_projection_seeds": EVALUATION_PROJECTION_SEEDS,
    "ridge_initial_exponents": RIDGE_INITIAL_EXPONENTS,
    "ridge_expansion_steps": RIDGE_EXPANSION_STEPS,
    "native_noninferiority_tolerance": NATIVE_NONINFERIORITY_TOLERANCE,
    "native_planner_harm_tolerance": NATIVE_PLANNER_HARM_TOLERANCE,
    "auglag_beta": AUGLAG_BETA,
    "parameter_trust_radius": PARAMETER_TRUST_RADIUS,
    "trust_reference_rms_floor": TRUST_REFERENCE_RMS_FLOOR,
    "native_denominator_eps": NATIVE_DENOMINATOR_EPS,
    "boundary_improvement_tolerance": BOUNDARY_IMPROVEMENT_TOLERANCE,
    "direct_projection_consensus_required": (
        DIRECT_PROJECTION_CONSENSUS_REQUIRED
    ),
    "readout_projection_consensus_required": (
        READOUT_PROJECTION_CONSENSUS_REQUIRED
    ),
    "evidence_status": EVIDENCE_STATUS,
    "task_family_id": TASK_FAMILY_ID,
    "development_split": DEVELOPMENT_SPLIT,
    "stage7_task_family_id": STAGE7_TASK_FAMILY_ID,
    "stage7_development_split": STAGE7_DEVELOPMENT_SPLIT,
    "pinned_dependencies": PINNED,
    "runtime_abi": {
        "python_major_minor": ".".join(
            platform.python_version_tuple()[:2]
        ),
        "torch": VERSIONS["torch"],
        "torchvision": VERSIONS["torchvision"],
        "numpy": VERSIONS["numpy"],
        "cuda_runtime": VERSIONS["cuda_runtime"],
        "cudnn": VERSIONS["cudnn"],
    },
}
'''
setup = setup[:config_start] + stage11_config + setup[signature_start:]

shared_definitions = json.loads(json.dumps(base["cells"][4]))
shared_source = "".join(shared_definitions["source"])
shared_source = shared_source.replace(
    "FPMA requires all ten candidates at the same parameter state",
    "ARGA requires all ten candidates at the same parameter state",
)
shared_definitions["source"] = shared_source.splitlines(keepends=True)

phase_a = json.loads(json.dumps(base["cells"][5]))
phase_a["source"] = (
    "".join(phase_a["source"])
    .replace("stage10", "stage11")
    .replace("Stage 10", "Stage 11")
    .splitlines(keepends=True)
)
phase_b = json.loads(json.dumps(base["cells"][6]))
phase_b["source"] = (
    "".join(phase_b["source"])
    .replace("stage10", "stage11")
    .replace("Stage 10", "Stage 11")
    .splitlines(keepends=True)
)

stage10_phase_c = "".join(base["cells"][7]["source"])
phase_c_prefix = stage10_phase_c[
    : stage10_phase_c.index("def pairwise_margin_certificate")
]
phase_c_prefix = (
    phase_c_prefix.replace("stage10", "stage11")
    .replace("Stage 10", "Stage 11")
    .replace(
        "# Phase C — fit frozen physical decoders, then adapt the causal action path.",
        "# Phase C — adapt centered action-response geometry with compute gates.",
    )
    .replace(
        '''CONSTRAINED_METHODS = {
    "fidelity_constrained_latent_only",
    "fidelity_constrained_shuffled_fpma",
    "fidelity_constrained_matched_fpma",
}
FPMA_METHODS = {
    "fidelity_constrained_shuffled_fpma",
    "fidelity_constrained_matched_fpma",
    "unconstrained_matched_fpma",
}
''',
        '''CONSTRAINED_METHODS = set(ADAPTATION_METHODS)
GEOMETRY_METHODS = {
    "fidelity_constrained_shuffled_geometry",
    "fidelity_constrained_matched_geometry",
}
''',
    )
)

phase_c = phase_c_prefix + r'''
def optimizer_state_to_cpu(state):
    if isinstance(state, torch.Tensor):
        return state.detach().cpu().clone()
    if isinstance(state, dict):
        return {
            key: optimizer_state_to_cpu(value)
            for key, value in state.items()
        }
    if isinstance(state, list):
        return [optimizer_state_to_cpu(value) for value in state]
    if isinstance(state, tuple):
        return tuple(optimizer_state_to_cpu(value) for value in state)
    return deepcopy(state)


def atomic_torch_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def build_optimizer(parameters, learning_rate):
    return torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=ADAPTATION_WEIGHT_DECAY,
    )


def prepare_action_path_trust_region(predictor, reference):
    groups = action_path_parameter_groups(predictor)
    named = dict(action_path_named_parameters(predictor))
    prepared = {}
    for group_name, parameters in groups.items():
        names = [
            name
            for name, parameter in action_path_named_parameters(predictor)
            if any(parameter is candidate for candidate in parameters)
        ]
        targets = [
            reference[name].to(named[name].device, named[name].dtype)
            for name in names
        ]
        reference_norm = torch.sqrt(
            torch.stack([target.square().sum() for target in targets]).sum()
        )
        count = sum(named[name].numel() for name in names)
        floor = reference_norm.new_tensor(
            TRUST_REFERENCE_RMS_FLOOR * math.sqrt(max(count, 1))
        )
        prepared[group_name] = {
            "parameters": [named[name] for name in names],
            "targets": targets,
            "limit": PARAMETER_TRUST_RADIUS
            * torch.maximum(reference_norm, floor),
        }
    return prepared


def project_action_path_trust_region(prepared):
    events = []
    with torch.no_grad():
        for group in prepared.values():
            delta_norm = torch.sqrt(
                torch.stack(
                    [
                        (parameter - target).square().sum()
                        for parameter, target in zip(
                            group["parameters"], group["targets"]
                        )
                    ]
                ).sum()
            )
            scale = torch.clamp(
                group["limit"] / delta_norm.clamp_min(1e-12),
                max=1.0,
            )
            for parameter, target in zip(
                group["parameters"], group["targets"]
            ):
                parameter.copy_(target + scale * (parameter - target))
            events.append((scale < 1.0).to(torch.float32))
    return torch.stack(events).sum()


def projected_centered(tokens, projectors):
    actions, horizons, token_count, channels = tokens.shape
    flattened = tokens.reshape(
        actions * horizons, token_count, channels
    )
    projected = torch.stack(
        [
            projector(flattened).reshape(
                actions, horizons, projector.output_dim
            )
            for projector in projectors
        ],
        dim=0,
    )
    return projected - projected.mean(dim=1, keepdim=True)


def fit_geometry_reference(
    environment, model_name, predictor, train_ids
):
    output_path = (
        OUT
        / "geometry_references"
        / f"{model_name}_{RUN_SIGNATURE[:12]}.pt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_dim = 16 * 16 * int(predictor.predictor_embed_dim)
    projectors = [
        CountSketchProjector(
            input_dim,
            TRAINING_GEOMETRY_PROJECTION_DIM,
            projection_seed,
        )
        for projection_seed in TRAINING_GEOMETRY_PROJECTION_SEEDS
    ]
    projector_checksums = [
        countsketch_checksum(projector) for projector in projectors
    ]
    if output_path.exists():
        payload = torch.load(
            output_path, map_location="cpu", weights_only=False
        )
        expected = {
            "run_signature": RUN_SIGNATURE,
            "environment": environment,
            "model": model_name,
            "projection_seeds": TRAINING_GEOMETRY_PROJECTION_SEEDS,
            "projection_dim": TRAINING_GEOMETRY_PROJECTION_DIM,
            "projector_checksums": projector_checksums,
            "cache_content_digest": CACHE_CONTENT_DIGEST,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise RuntimeError(
                    f"geometry reference mismatch {key}: "
                    f"{payload.get(key)} != {value}"
                )
        whiteners = payload["whiteners"].cuda()
        return projectors, whiteners, payload

    chunks = [
        [[] for _ in HORIZONS]
        for _ in TRAINING_GEOMETRY_PROJECTION_SEEDS
    ]
    with torch.inference_mode():
        for state_id in train_ids:
            shard = load_training_state(
                environment, model_name, int(state_id)
            )
            target = torch.as_tensor(
                shard["true_tokens"][:, HORIZON_INDICES],
                device="cuda",
                dtype=torch.float32,
            )
            centered = projected_centered(target, projectors)
            for projection_index in range(len(projectors)):
                for horizon_index in range(len(HORIZONS)):
                    chunks[projection_index][horizon_index].append(
                        centered[
                            projection_index, :, horizon_index
                        ].detach().cpu()
                    )

    whiteners = []
    spectra = []
    for projection_index in range(len(projectors)):
        projection_whiteners = []
        projection_spectra = []
        for horizon_index in range(len(HORIZONS)):
            values = torch.cat(
                chunks[projection_index][horizon_index], dim=0
            ).double()
            covariance = values.T @ values / max(values.shape[0] - 1, 1)
            mean_variance = float(
                torch.trace(covariance) / covariance.shape[0]
            )
            ridge = max(
                GEOMETRY_WHITENING_EIGEN_FLOOR,
                GEOMETRY_WHITENING_RIDGE_FRACTION
                * max(mean_variance, GEOMETRY_WHITENING_EIGEN_FLOOR),
            )
            eigenvalue, eigenvector = torch.linalg.eigh(covariance)
            regularized = torch.clamp(
                eigenvalue + ridge,
                min=GEOMETRY_WHITENING_EIGEN_FLOOR,
            )
            whitener = (
                eigenvector
                @ torch.diag(regularized.rsqrt())
                @ eigenvector.T
            ).float()
            projection_whiteners.append(whitener)
            projection_spectra.append(
                {
                    "horizon": HORIZONS[horizon_index],
                    "ridge": ridge,
                    "minimum_eigenvalue": float(eigenvalue.min()),
                    "maximum_eigenvalue": float(eigenvalue.max()),
                    "condition_after_ridge": float(
                        regularized.max() / regularized.min()
                    ),
                }
            )
        whiteners.append(torch.stack(projection_whiteners))
        spectra.append(projection_spectra)
    whiteners = torch.stack(whiteners)
    payload = {
        "run_signature": RUN_SIGNATURE,
        "environment": environment,
        "model": model_name,
        "projection_seeds": list(TRAINING_GEOMETRY_PROJECTION_SEEDS),
        "projection_dim": TRAINING_GEOMETRY_PROJECTION_DIM,
        "projector_checksums": projector_checksums,
        "cache_content_digest": CACHE_CONTENT_DIGEST,
        "fit_split": "probe_train",
        "uses_physical_pose_goal_cost_or_readout": False,
        "spectra": spectra,
        "whiteners": whiteners.cpu(),
    }
    atomic_torch_save(payload, output_path)
    return projectors, whiteners.cuda(), payload


def geometry_loss_by_horizon(
    predicted_tokens,
    true_tokens,
    projectors,
    whiteners,
    permutation=None,
):
    predicted = projected_centered(
        predicted_tokens[:, HORIZON_INDICES], projectors
    )
    target = projected_centered(
        true_tokens[:, HORIZON_INDICES], projectors
    )
    if permutation is not None:
        target = target[
            :,
            torch.as_tensor(
                permutation, device=target.device, dtype=torch.long
            ),
        ]
    difference = predicted - target
    whitened = torch.einsum(
        "pahd,phde->pahe", difference, whiteners
    )
    result = whitened.square().mean(dim=(0, 1, 3))
    if result.shape != (len(HORIZONS),):
        raise RuntimeError(
            f"geometry loss has wrong shape {tuple(result.shape)}"
        )
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("geometry loss is non-finite")
    return result


def geometry_calibration_metrics(
    model,
    method,
    adaptation_seed,
    environment,
    model_name,
    state_ids,
    initial_cache,
    projectors,
    whiteners,
    force_matched=False,
):
    geometry_rows = []
    native_rows = []
    with torch.inference_mode():
        for state_id in state_ids:
            shard = load_training_state(
                environment, model_name, int(state_id)
            )
            predicted, _ = differentiable_unroll(
                model,
                initial_to_cuda(initial_cache[int(state_id)]),
                torch.as_tensor(
                    shard["actions"],
                    device="cuda",
                    dtype=torch.float32,
                ),
            )
            target = torch.as_tensor(
                shard["true_tokens"],
                device="cuda",
                dtype=torch.float32,
            )
            permutation = None
            if (
                method == "fidelity_constrained_shuffled_geometry"
                and not force_matched
            ):
                permutation = deterministic_non_null_derangement(
                    int(state_id), adaptation_seed
                )
            geometry_rows.append(
                geometry_loss_by_horizon(
                    predicted,
                    target,
                    projectors,
                    whiteners,
                    permutation,
                )
                .detach()
                .cpu()
                .numpy()
            )
            native_rows.append(
                native_horizon_loss(predicted, target)
                .detach()
                .cpu()
                .numpy()
            )
    geometry = np.mean(np.asarray(geometry_rows), axis=0)
    native = np.mean(np.asarray(native_rows), axis=0)
    if not np.all(np.isfinite(geometry)) or not np.all(
        np.isfinite(native)
    ):
        raise RuntimeError("calibration metrics are non-finite")
    return {
        "geometry_mse": geometry,
        "geometry_rmse": np.sqrt(np.maximum(geometry, 0.0)),
        "native_horizon": native,
    }


def native_profile_metrics(
    model,
    environment,
    model_name,
    state_ids,
    initial_cache,
):
    rows = []
    by_state = {}
    with torch.inference_mode():
        for state_id in state_ids:
            shard = load_training_state(
                environment, model_name, int(state_id)
            )
            predicted, _ = differentiable_unroll(
                model,
                initial_to_cuda(initial_cache[int(state_id)]),
                torch.as_tensor(
                    shard["actions"],
                    device="cuda",
                    dtype=torch.float32,
                ),
            )
            value = (
                native_horizon_loss(
                    predicted,
                    torch.as_tensor(
                        shard["true_tokens"],
                        device="cuda",
                        dtype=torch.float32,
                    ),
                )
                .detach()
                .cpu()
                .numpy()
            )
            rows.append(value)
            by_state[int(state_id)] = value
    mean = np.mean(np.asarray(rows), axis=0)
    return {"native_horizon": mean, "native_by_state": by_state}


def geometry_checkpoint_payload(
    predictor,
    optimizer,
    method,
    adaptation_seed,
    environment,
    model_name,
    epoch,
    score,
    geometry_rmse,
    native_ratios,
    fidelity_feasible,
    base_checksum,
    evaluation_decoder_checksum,
):
    return {
        "run_signature": RUN_SIGNATURE,
        "method_implementation_id": METHOD_IMPLEMENTATION_ID,
        "method": method,
        "adaptation_seed": int(adaptation_seed),
        "environment": environment,
        "model": model_name,
        "cache_content_digest": CACHE_CONTENT_DIGEST,
        "pretrained_asset_digest": pretrained_asset_digest(model_name),
        "base_action_path_checksum": base_checksum,
        "training_decoder_checksum": evaluation_decoder_checksum,
        "training_decoders_used_by_objective": False,
        "selected_epoch": int(epoch),
        "calibration_selection_score": float(score),
        "calibration_geometry_rmse": [
            float(value) for value in geometry_rmse
        ],
        "calibration_native_ratios": [
            float(value) for value in native_ratios
        ],
        "fidelity_feasible": bool(fidelity_feasible),
        "action_path": extract_action_path_state(predictor),
        "optimizer_state": optimizer_state_to_cpu(
            optimizer.state_dict()
        ),
    }


def train_one_geometry_path(
    model,
    environment,
    model_name,
    method,
    adaptation_seed,
    base_state,
    base_checksum,
    train_ids,
    calibration_ids,
    initial_cache,
    projectors,
    whiteners,
    baseline_calibration_native,
    baseline_train_native,
    baseline_train_native_by_state,
    evaluation_decoder_checksum,
):
    output_path = (
        ADAPTED_DIR
        / (
            f"{model_name}_{method}_seed{adaptation_seed}_"
            f"{RUN_SIGNATURE[:12]}.pt"
        )
    )
    latest_path = output_path.with_name(
        output_path.stem + "_latest.pt"
    )
    if output_path.exists():
        payload = torch.load(
            output_path, map_location="cpu", weights_only=False
        )
        expected = {
            "run_signature": RUN_SIGNATURE,
            "method": method,
            "adaptation_seed": int(adaptation_seed),
            "environment": environment,
            "model": model_name,
            "base_action_path_checksum": base_checksum,
            "cache_content_digest": CACHE_CONTENT_DIGEST,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise RuntimeError(
                    f"adaptation resume mismatch {key}: "
                    f"{payload.get(key)} != {value}"
                )
        validate_action_path_state(
            payload["action_path"],
            payload["selected_action_path_checksum"],
        )
        return payload

    predictor = set_action_path_trainability(model, True)
    load_action_path_state(predictor, base_state)
    predictor.use_activation_checkpointing = USE_ACTIVATION_CHECKPOINTING
    parameters = [
        parameter
        for _, parameter in action_path_named_parameters(predictor)
    ]
    torch.manual_seed(int(adaptation_seed))
    torch.cuda.manual_seed_all(int(adaptation_seed))
    optimizer = build_optimizer(parameters, ADAPTATION_LR)
    trust_region = prepare_action_path_trust_region(
        predictor, base_state
    )
    rng = np.random.default_rng(int(adaptation_seed))
    dual = np.zeros(len(HORIZONS), dtype=np.float64)
    trust_projection_count = torch.zeros(
        (), device="cuda", dtype=torch.float32
    )
    epoch_limits = [INITIAL_EPOCH_LIMIT, *EXTENSION_EPOCH_LIMITS]
    limit_index = 0
    active_limit = epoch_limits[limit_index]

    if latest_path.exists():
        latest = torch.load(
            latest_path, map_location="cpu", weights_only=False
        )
        expected = {
            "run_signature": RUN_SIGNATURE,
            "method": method,
            "adaptation_seed": int(adaptation_seed),
            "environment": environment,
            "model": model_name,
            "base_action_path_checksum": base_checksum,
            "cache_content_digest": CACHE_CONTENT_DIGEST,
        }
        for key, value in expected.items():
            if latest.get(key) != value:
                raise RuntimeError(
                    f"latest checkpoint mismatch {key}: "
                    f"{latest.get(key)} != {value}"
                )
        validate_action_path_state(
            latest["current_action_path"],
            latest["current_action_path_checksum"],
        )
        load_action_path_state(
            predictor, latest["current_action_path"]
        )
        optimizer.load_state_dict(latest["optimizer_state"])
        dual = np.asarray(latest["dual"], dtype=np.float64)
        rng.bit_generator.state = latest["numpy_rng_state"]
        torch.set_rng_state(latest["torch_rng_state"])
        torch.cuda.set_rng_state_all(latest["cuda_rng_state_all"])
        history = latest["history"]
        best = latest["best"]
        last_feasible = latest["last_feasible"]
        eligible_scores = [
            (int(item[0]), float(item[1]))
            for item in latest["eligible_scores"]
        ]
        no_improvement_checkpoints = int(
            latest["no_improvement_checkpoints"]
        )
        epoch = int(latest["epoch"])
        limit_index = int(latest["limit_index"])
        active_limit = int(latest["active_limit"])
        unresolved_boundary = bool(
            latest["unresolved_boundary"]
        )
        trust_projection_count.fill_(
            float(latest["trust_projection_count"])
        )
        if (
            limit_index >= len(epoch_limits)
            or active_limit != epoch_limits[limit_index]
            or epoch % CHECKPOINT_EVERY != 0
        ):
            raise RuntimeError(
                "invalid latest checkpoint state machine"
            )
        if bool(latest.get("stop_after_checkpoint", False)):
            active_limit = epoch
        log.info(
            "%s %s seed=%d resuming at epoch %d",
            model_name,
            method,
            adaptation_seed,
            epoch,
        )
    else:
        predictor.use_activation_checkpointing = False
        epoch_zero = geometry_calibration_metrics(
            model,
            method,
            adaptation_seed,
            environment,
            model_name,
            calibration_ids,
            initial_cache,
            projectors,
            whiteners,
        )
        predictor.use_activation_checkpointing = (
            USE_ACTIVATION_CHECKPOINTING
        )
        epoch_zero_score = (
            float(np.mean(epoch_zero["native_horizon"]))
            if method == "fidelity_constrained_latent_only"
            else float(np.mean(epoch_zero["geometry_mse"]))
        )
        best = geometry_checkpoint_payload(
            predictor,
            optimizer,
            method,
            adaptation_seed,
            environment,
            model_name,
            0,
            epoch_zero_score,
            epoch_zero["geometry_rmse"],
            np.ones(len(HORIZONS)),
            True,
            base_checksum,
            evaluation_decoder_checksum,
        )
        last_feasible = deepcopy(best)
        best.pop("optimizer_state", None)
        history = []
        eligible_scores = [(0, epoch_zero_score)]
        no_improvement_checkpoints = 0
        epoch = 0
        unresolved_boundary = False

    while epoch < active_limit:
        epoch += 1
        order = np.asarray(train_ids, dtype=np.int64).copy()
        rng.shuffle(order)
        train_objectives = []
        for state_id_value in order:
            state_id = int(state_id_value)
            shard = load_training_state(
                environment, model_name, state_id
            )
            optimizer.zero_grad(set_to_none=True)
            predicted, _ = differentiable_unroll(
                model,
                initial_to_cuda(initial_cache[state_id]),
                torch.as_tensor(
                    shard["actions"],
                    device="cuda",
                    dtype=torch.float32,
                ),
            )
            target = torch.as_tensor(
                shard["true_tokens"],
                device="cuda",
                dtype=torch.float32,
            )
            native_by_horizon = native_horizon_loss(
                predicted, target
            )
            baseline_train_mean = torch.as_tensor(
                baseline_train_native,
                device="cuda",
                dtype=torch.float32,
            )
            baseline_train_denom = baseline_train_mean.clamp_min(
                NATIVE_DENOMINATOR_EPS
            )
            baseline_state = torch.as_tensor(
                baseline_train_native_by_state[state_id],
                device="cuda",
                dtype=torch.float32,
            )
            native_g = (
                (native_by_horizon - baseline_state)
                / baseline_train_denom
                + baseline_train_mean / baseline_train_denom
                - (1.0 + NATIVE_NONINFERIORITY_TOLERANCE)
            )
            if method == "fidelity_constrained_latent_only":
                objective_by_horizon = native_by_horizon
            else:
                permutation = None
                if method == "fidelity_constrained_shuffled_geometry":
                    permutation = deterministic_non_null_derangement(
                        state_id, adaptation_seed
                    )
                objective_by_horizon = geometry_loss_by_horizon(
                    predicted,
                    target,
                    projectors,
                    whiteners,
                    permutation,
                )
            decision_objective = objective_by_horizon.mean()
            dual_tensor = torch.as_tensor(
                dual, device="cuda", dtype=torch.float32
            )
            shifted = torch.clamp(
                dual_tensor + AUGLAG_BETA * native_g,
                min=0.0,
            )
            augmented_penalty = torch.sum(
                (shifted.square() - dual_tensor.square())
                / (2.0 * AUGLAG_BETA)
            )
            total = decision_objective + augmented_penalty
            if not bool(torch.isfinite(total)):
                raise RuntimeError(
                    f"non-finite training objective at state {state_id}"
                )
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters,
                GRADIENT_CLIP,
                error_if_nonfinite=True,
            )
            optimizer.step()
            trust_projection_count += project_action_path_trust_region(
                trust_region
            )
            train_objectives.append(
                objective_by_horizon.detach().cpu().numpy()
            )

        if epoch % CHECKPOINT_EVERY != 0:
            continue
        predictor.use_activation_checkpointing = False
        calibration = geometry_calibration_metrics(
            model,
            method,
            adaptation_seed,
            environment,
            model_name,
            calibration_ids,
            initial_cache,
            projectors,
            whiteners,
        )
        train_native = native_profile_metrics(
            model,
            environment,
            model_name,
            train_ids,
            initial_cache,
        )
        predictor.use_activation_checkpointing = USE_ACTIVATION_CHECKPOINTING
        ratios = calibration["native_horizon"] / np.maximum(
            baseline_calibration_native,
            NATIVE_DENOMINATOR_EPS,
        )
        fidelity_feasible = bool(
            np.all(
                ratios
                <= 1.0
                + NATIVE_NONINFERIORITY_TOLERANCE
                + 1e-7
            )
        )
        score = (
            float(np.mean(calibration["native_horizon"]))
            if method == "fidelity_constrained_latent_only"
            else float(np.mean(calibration["geometry_mse"]))
        )
        improved = False
        rollback = not fidelity_feasible
        if fidelity_feasible:
            candidate = geometry_checkpoint_payload(
                predictor,
                optimizer,
                method,
                adaptation_seed,
                environment,
                model_name,
                epoch,
                score,
                calibration["geometry_rmse"],
                ratios,
                True,
                base_checksum,
                evaluation_decoder_checksum,
            )
            last_feasible = deepcopy(candidate)
            eligible_scores.append((epoch, score))
            if score < best["calibration_selection_score"] - 1e-12:
                best = deepcopy(candidate)
                best.pop("optimizer_state", None)
                improved = True
                no_improvement_checkpoints = 0
            else:
                no_improvement_checkpoints += 1
        else:
            current_learning_rates = [
                float(group["lr"]) for group in optimizer.param_groups
            ]
            load_action_path_state(
                predictor, last_feasible["action_path"]
            )
            optimizer.load_state_dict(
                deepcopy(last_feasible["optimizer_state"])
            )
            for group, learning_rate in zip(
                optimizer.param_groups, current_learning_rates
            ):
                group["lr"] = learning_rate * 0.5
            no_improvement_checkpoints += 1

        train_native_ratio = (
            train_native["native_horizon"]
            / np.maximum(
                baseline_train_native,
                NATIVE_DENOMINATOR_EPS,
            )
        )
        dual = np.maximum(
            0.0,
            dual
            + AUGLAG_BETA
            * (
                train_native_ratio
                - (1.0 + NATIVE_NONINFERIORITY_TOLERANCE)
            ),
        )
        history.append(
            {
                "environment": environment,
                "model": model_name,
                "method": method,
                "adaptation_seed": adaptation_seed,
                "epoch": epoch,
                "selection_score": score,
                "geometry_rmse_h1": float(
                    calibration["geometry_rmse"][0]
                ),
                "geometry_rmse_h3": float(
                    calibration["geometry_rmse"][1]
                ),
                "geometry_rmse_h6": float(
                    calibration["geometry_rmse"][2]
                ),
                "native_ratio_h1": float(ratios[0]),
                "native_ratio_h3": float(ratios[1]),
                "native_ratio_h6": float(ratios[2]),
                "fidelity_feasible": fidelity_feasible,
                "rollback": rollback,
                "improved": improved,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "trust_projection_count": int(
                    trust_projection_count.detach().cpu()
                ),
                "train_objective_h1": float(
                    np.mean(np.asarray(train_objectives)[:, 0])
                ),
                "train_objective_h3": float(
                    np.mean(np.asarray(train_objectives)[:, 1])
                ),
                "train_objective_h6": float(
                    np.mean(np.asarray(train_objectives)[:, 2])
                ),
            }
        )
        log.info(
            "%s %s seed=%d epoch=%d score=%.6f ratios=%s",
            model_name,
            method,
            adaptation_seed,
            epoch,
            score,
            np.round(ratios, 4).tolist(),
        )

        should_stop_early = bool(
            epoch >= INITIAL_EPOCH_LIMIT
            and no_improvement_checkpoints
            >= EARLY_STOPPING_CHECKPOINT_PATIENCE
        )
        if epoch == active_limit and not should_stop_early:
            improving_boundary = (
                best["selected_epoch"] == epoch
                and len(eligible_scores) >= 2
                and eligible_scores[-2][1] - eligible_scores[-1][1]
                > BOUNDARY_IMPROVEMENT_TOLERANCE
            )
            if improving_boundary and limit_index + 1 < len(epoch_limits):
                limit_index += 1
                active_limit = epoch_limits[limit_index]
            elif improving_boundary:
                unresolved_boundary = True

        current_action_path = extract_action_path_state(predictor)
        atomic_torch_save(
            {
                "run_signature": RUN_SIGNATURE,
                "method_implementation_id": METHOD_IMPLEMENTATION_ID,
                "method": method,
                "adaptation_seed": int(adaptation_seed),
                "environment": environment,
                "model": model_name,
                "cache_content_digest": CACHE_CONTENT_DIGEST,
                "pretrained_asset_digest": pretrained_asset_digest(
                    model_name
                ),
                "base_action_path_checksum": base_checksum,
                "training_decoder_checksum": (
                    evaluation_decoder_checksum
                ),
                "epoch": int(epoch),
                "current_action_path": current_action_path,
                "current_action_path_checksum": action_path_checksum(
                    current_action_path
                ),
                "optimizer_state": optimizer_state_to_cpu(
                    optimizer.state_dict()
                ),
                "dual": dual.copy(),
                "numpy_rng_state": deepcopy(rng.bit_generator.state),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
                "history": deepcopy(history),
                "best": deepcopy(best),
                "last_feasible": deepcopy(last_feasible),
                "eligible_scores": deepcopy(eligible_scores),
                "no_improvement_checkpoints": int(
                    no_improvement_checkpoints
                ),
                "limit_index": int(limit_index),
                "active_limit": int(active_limit),
                "unresolved_boundary": bool(unresolved_boundary),
                "trust_projection_count": float(
                    trust_projection_count.detach().cpu()
                ),
                "stop_after_checkpoint": should_stop_early,
            },
            latest_path,
        )

        if should_stop_early:
            log.info(
                "%s %s seed=%d early-stopped at epoch %d",
                model_name,
                method,
                adaptation_seed,
                epoch,
            )
            break

    best["undertrained_inconclusive"] = bool(unresolved_boundary)
    best["base_action_path_checksum"] = base_checksum
    best["selected_action_path_checksum"] = action_path_checksum(
        best["action_path"]
    )
    best["trust_projection_count"] = int(
        trust_projection_count.detach().cpu()
    )
    best["completed_epoch_limit"] = int(epoch)
    best["all_checkpoint_history"] = history
    best.pop("optimizer_state", None)
    atomic_torch_save(best, output_path)
    if latest_path.exists():
        latest_path.unlink()
    load_action_path_state(predictor, best["action_path"])
    predictor.use_activation_checkpointing = False
    return best


def evaluate_seed_direct_geometry(
    adaptation_seed, environment_payloads
):
    rows = []
    variants = [
        "frozen",
        "fidelity_constrained_latent_only",
        "fidelity_constrained_shuffled_geometry",
        "fidelity_constrained_matched_geometry",
    ]
    repo = configure_repo()
    for environment in ENVIRONMENT:
        payload = environment_payloads[environment]
        model_name = payload["model_name"]
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
        for variant in variants:
            if variant == "frozen":
                load_action_path_state(
                    predictor, payload["base_state"]
                )
            else:
                checkpoint = torch.load(
                    ADAPTED_DIR
                    / (
                        f"{model_name}_{variant}_seed{adaptation_seed}_"
                        f"{RUN_SIGNATURE[:12]}.pt"
                    ),
                    map_location="cpu",
                    weights_only=False,
                )
                load_action_path_state(
                    predictor, checkpoint["action_path"]
                )
            metrics = geometry_calibration_metrics(
                model,
                variant,
                adaptation_seed,
                environment,
                model_name,
                payload["calibration_ids"],
                payload["initial_cache"],
                payload["projectors"],
                payload["whiteners"],
                force_matched=True,
            )
            for horizon_index, horizon in enumerate(HORIZONS):
                rows.append(
                    {
                        "environment": environment,
                        "model": model_name,
                        "adaptation_seed": adaptation_seed,
                        "method": variant,
                        "horizon": horizon,
                        "matched_geometry_rmse": float(
                            metrics["geometry_rmse"][horizon_index]
                        ),
                        "native_anchor": float(
                            metrics["native_horizon"][horizon_index]
                        ),
                    }
                )
        del model, predictor
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def screening_gate(rows):
    by_environment = {}
    for environment in ENVIRONMENT:
        ratios = {}
        for horizon in HORIZONS:
            values = {
                row["method"]: float(row["matched_geometry_rmse"])
                for row in rows
                if row["environment"] == environment
                and int(row["horizon"]) == horizon
            }
            matched = values[
                "fidelity_constrained_matched_geometry"
            ]
            ratios[horizon] = {
                "matched_over_frozen": matched
                / max(values["frozen"], 1e-12),
                "matched_over_shuffled": matched
                / max(
                    values[
                        "fidelity_constrained_shuffled_geometry"
                    ],
                    1e-12,
                ),
            }
        strong_horizons = [
            horizon
            for horizon, value in ratios.items()
            if value["matched_over_frozen"]
            <= 1.0 - SCREENING_DIRECT_MIN_RELATIVE_GAIN
            and value["matched_over_shuffled"]
            <= 1.0 - SCREENING_LABEL_SPECIFICITY_GAIN
        ]
        catastrophic = any(
            value["matched_over_frozen"]
            > 1.0 + SCREENING_MAX_OTHER_ENVIRONMENT_HARM
            for value in ratios.values()
        )
        by_environment[environment] = {
            "ratios": ratios,
            "strong_horizons": strong_horizons,
            "strong_signal": len(strong_horizons) >= 2,
            "catastrophic_harm": catastrophic,
        }
    passed = bool(
        any(
            value["strong_signal"]
            for value in by_environment.values()
        )
        and not any(
            value["catastrophic_harm"]
            for value in by_environment.values()
        )
    )
    payload = {
        "passed": passed,
        "criterion": (
            "run confirmation seed only if at least one environment has "
            "matched/frozen <= 0.97 and matched/shuffled <= 0.99 at >=2/3 "
            "horizons, with no environment >5% worse than frozen"
        ),
        "environments": by_environment,
    }
    write_json(OUT / "stage11_screening_gate.json", payload)
    return payload


def package_phase_c_rescue(executed_seeds):
    path = Path("/content/stage11_phase_c_checkpoint_rescue.zip")
    candidates = [
        OUT / "config.json",
        OUT / "stage11_cache_binding.json",
        OUT / "stage11_screening_gate.json",
        OUT / "stage11_direct_calibration_geometry.csv",
        OUT / "stage11_geometry_adaptation_manifest.json",
    ]
    candidates.extend(
        file
        for file in ADAPTED_DIR.glob("*.pt")
        if not file.name.endswith("_latest.pt")
    )
    candidates.extend((OUT / "geometry_references").glob("*.pt"))
    files = sorted({file for file in candidates if file.exists()})
    with zipfile.ZipFile(
        path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for file in files:
            archive.write(file, file.relative_to(OUT))
    print(f"PHASE_C_RESCUE_ZIP: {path}")
    print(f"EXECUTED_ADAPTATION_SEEDS: {executed_seeds}")
    if DOWNLOAD_PHASE_C_RESCUE:
        try:
            from google.colab import files as colab_files

            colab_files.download(str(path))
        except Exception as error:
            log.warning(
                "phase-C rescue download failed; local file remains: %s",
                error,
            )
    return path


def adapt_geometry_paths():
    manifest = []
    direct_rows = []
    environment_payloads = {}
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
        loaded_asset_digest = verify_loaded_pretrained_assets(model_name)
        model.eval()
        predictor = set_action_path_trainability(model, False)
        base_state = extract_action_path_state(predictor)
        base_checksum = action_path_checksum(base_state)
        train_ids = state_ids_for_split(environment, "probe_train")
        calibration_ids = state_ids_for_split(
            environment, "probe_calibration"
        )
        initial_cache = cache_initial_encodings(
            model, environment, [*train_ids, *calibration_ids]
        )
        projectors, whiteners, geometry_reference = (
            fit_geometry_reference(
                environment, model_name, predictor, train_ids
            )
        )
        # Fitted now for later evaluation only. No decoder object is passed
        # into train_one_geometry_path or geometry_loss_by_horizon.
        decoder_payload = fit_frozen_training_decoders(
            environment, model_name, predictor
        )
        evaluation_decoders = cuda_training_decoders(
            decoder_payload, predictor
        )
        evaluation_decoder_checksum = training_decoder_checksum(
            evaluation_decoders
        )
        baseline_calibration = geometry_calibration_metrics(
            model,
            "frozen",
            SCREENING_ADAPTATION_SEEDS[0],
            environment,
            model_name,
            calibration_ids,
            initial_cache,
            projectors,
            whiteners,
            force_matched=True,
        )
        baseline_train = native_profile_metrics(
            model,
            environment,
            model_name,
            train_ids,
            initial_cache,
        )
        environment_payloads[environment] = {
            "model_name": model_name,
            "base_state": base_state,
            "base_checksum": base_checksum,
            "train_ids": train_ids,
            "calibration_ids": calibration_ids,
            "initial_cache": initial_cache,
            "projectors": projectors,
            "whiteners": whiteners,
            "geometry_reference": geometry_reference,
            "baseline_calibration_native": baseline_calibration[
                "native_horizon"
            ],
            "baseline_train_native": baseline_train["native_horizon"],
            "baseline_train_native_by_state": baseline_train[
                "native_by_state"
            ],
            "evaluation_decoder_checksum": (
                evaluation_decoder_checksum
            ),
            "loaded_asset_digest": loaded_asset_digest,
        }
        del model, predictor, evaluation_decoders
        gc.collect()
        torch.cuda.empty_cache()

    executed_seeds = []
    screening = None
    for adaptation_seed in ADAPTATION_SEEDS:
        if (
            adaptation_seed in CONFIRMATION_ADAPTATION_SEEDS
            and screening is not None
            and not screening["passed"]
        ):
            log.info(
                "screening gate failed; skipping confirmation seed %d",
                adaptation_seed,
            )
            continue
        for environment in ENVIRONMENT:
            payload = environment_payloads[environment]
            model_name = payload["model_name"]
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
            for method in ADAPTATION_METHODS:
                load_action_path_state(
                    predictor, payload["base_state"]
                )
                result = train_one_geometry_path(
                    model,
                    environment,
                    model_name,
                    method,
                    adaptation_seed,
                    payload["base_state"],
                    payload["base_checksum"],
                    payload["train_ids"],
                    payload["calibration_ids"],
                    payload["initial_cache"],
                    payload["projectors"],
                    payload["whiteners"],
                    payload["baseline_calibration_native"],
                    payload["baseline_train_native"],
                    payload["baseline_train_native_by_state"],
                    payload["evaluation_decoder_checksum"],
                )
                manifest.append(
                    {
                        "environment": environment,
                        "model": model_name,
                        "method": method,
                        "adaptation_seed": adaptation_seed,
                        "selected_epoch": int(
                            result["selected_epoch"]
                        ),
                        "completed_epoch_limit": int(
                            result["completed_epoch_limit"]
                        ),
                        "calibration_selection_score": float(
                            result["calibration_selection_score"]
                        ),
                        "calibration_geometry_rmse": result[
                            "calibration_geometry_rmse"
                        ],
                        "calibration_native_ratios": result[
                            "calibration_native_ratios"
                        ],
                        "fidelity_feasible": bool(
                            result["fidelity_feasible"]
                        ),
                        "undertrained_inconclusive": bool(
                            result["undertrained_inconclusive"]
                        ),
                        "base_action_path_checksum": result[
                            "base_action_path_checksum"
                        ],
                        "selected_action_path_checksum": result[
                            "selected_action_path_checksum"
                        ],
                        "cache_content_digest": CACHE_CONTENT_DIGEST,
                        "pretrained_asset_digest": payload[
                            "loaded_asset_digest"
                        ],
                        "training_decoders_used_by_objective": False,
                    }
                )
            del model, predictor
            clear_training_state_memory_cache(environment)
            gc.collect()
            torch.cuda.empty_cache()
        executed_seeds.append(int(adaptation_seed))
        seed_rows = evaluate_seed_direct_geometry(
            adaptation_seed, environment_payloads
        )
        direct_rows.extend(seed_rows)
        write_csv(
            OUT / "stage11_direct_calibration_geometry.csv",
            direct_rows,
        )
        if adaptation_seed in SCREENING_ADAPTATION_SEEDS:
            screening = screening_gate(seed_rows)

    write_json(
        OUT / "stage11_geometry_adaptation_manifest.json",
        {
            "run_signature": RUN_SIGNATURE,
            "method_implementation_id": METHOD_IMPLEMENTATION_ID,
            "updated_modules": [
                "predictor.action_encoder",
                "predictor.predictor_blocks.*.adaLN_modulation[1]",
            ],
            "training_objective_inputs": [
                "predicted JEPA tokens",
                "target JEPA tokens",
                "candidate identity only through within-state correspondence",
            ],
            "training_objective_excludes": [
                "physical pose",
                "task goal",
                "physical cost",
                "learned physical readout",
                "planner action label",
            ],
            "evaluation_decoders_used_by_objective": False,
            "executed_adaptation_seeds": executed_seeds,
            "screening_gate": screening,
            "records": manifest,
        },
    )
    package_phase_c_rescue(executed_seeds)
    return manifest, executed_seeds, direct_rows, screening


if not PIPELINE_FAILED:
    try:
        (
            STAGE11_ADAPTATION_MANIFEST,
            EXECUTED_ADAPTATION_SEEDS,
            STAGE11_DIRECT_CALIBRATION_ROWS,
            STAGE11_SCREENING_GATE,
        ) = adapt_geometry_paths()
    except Exception:
        record_failure("readout_free_geometry_adaptation")
'''

phase_d = "".join(base["cells"][8]["source"])
phase_d = (
    phase_d.replace("stage10", "stage11")
    .replace("Stage 10", "Stage 11")
    .replace("STAGE10", "STAGE11")
    .replace("ADAPTATION_SEEDS", "EXECUTED_ADAPTATION_SEEDS")
    .replace(
        "fidelity_constrained_matched_fpma",
        "fidelity_constrained_matched_geometry",
    )
    .replace(
        "fidelity_constrained_shuffled_fpma",
        "fidelity_constrained_shuffled_geometry",
    )
    .replace('        "unconstrained_matched_fpma",\n', "")
)

phase_e = r'''# Phase E — unseen geometry, fresh readouts, and the pilot gate.


def evaluation_target_geometry(environment, model_name, projection_seed):
    sample = load_training_state(environment, model_name, 0)
    input_dim = int(
        sample["true_tokens"].shape[-2]
        * sample["true_tokens"].shape[-1]
    )
    projector = CountSketchProjector(
        input_dim, EVALUATION_PROJECTION_DIM, projection_seed
    )
    targets = []
    with torch.inference_mode():
        for state_id in range(NUM_STATES):
            shard = load_training_state(
                environment, model_name, state_id
            )
            value = torch.as_tensor(
                shard["true_tokens"][:, HORIZON_INDICES],
                device="cuda",
                dtype=torch.float32,
            )
            targets.append(
                projected_centered(value, [projector])[0]
                .detach()
                .cpu()
                .numpy()
            )
    targets = np.asarray(targets, dtype=np.float64)
    truth = load_truth_evaluation(environment)
    train_ids = split_indices(truth, "probe_train")
    whiteners = []
    for horizon_index in range(len(HORIZONS)):
        values = targets[
            train_ids, :, horizon_index
        ].reshape(-1, EVALUATION_PROJECTION_DIM)
        covariance = values.T @ values / max(values.shape[0] - 1, 1)
        ridge = max(
            GEOMETRY_WHITENING_EIGEN_FLOOR,
            GEOMETRY_WHITENING_RIDGE_FRACTION
            * max(
                float(np.trace(covariance) / covariance.shape[0]),
                GEOMETRY_WHITENING_EIGEN_FLOOR,
            ),
        )
        eigenvalue, eigenvector = np.linalg.eigh(covariance)
        regularized = np.maximum(
            eigenvalue + ridge,
            GEOMETRY_WHITENING_EIGEN_FLOOR,
        )
        whiteners.append(
            (eigenvector * regularized[None] ** -0.5)
            @ eigenvector.T
        )
    return targets, np.asarray(whiteners), truth


def evaluate_unseen_geometry():
    rows = []
    for environment in ENVIRONMENT:
        model_name = MODEL_BY_ENVIRONMENT[environment][0]
        for projection_index, projection_seed in enumerate(
            EVALUATION_PROJECTION_SEEDS
        ):
            target, whiteners, truth = evaluation_target_geometry(
                environment, model_name, projection_seed
            )
            development_ids = split_indices(
                truth, DEVELOPMENT_SPLIT
            )
            for record in EVALUATION_VARIANT_RECORDS:
                loaded = load_variant_features(
                    model_name,
                    record["variant"],
                    projection_index,
                )
                predicted = loaded["projected"].astype(np.float64)
                predicted = (
                    predicted
                    - predicted.mean(axis=1, keepdims=True)
                )
                difference = predicted - target
                whitened = np.einsum(
                    "sahd,hde->sahe", difference, whiteners
                )
                state_rmse = np.sqrt(
                    np.mean(whitened**2, axis=(1, 3))
                )
                for state_id in development_ids:
                    for horizon_index, horizon in enumerate(HORIZONS):
                        rows.append(
                            {
                                "environment": environment,
                                "model": model_name,
                                "state_id": int(state_id),
                                "task_id": int(
                                    truth["task_id"][state_id]
                                ),
                                "method": record["method"],
                                "variant": record["variant"],
                                "adaptation_seed": int(
                                    record["adaptation_seed"]
                                ),
                                "projection_seed": int(
                                    projection_seed
                                ),
                                "horizon": horizon,
                                "whitened_geometry_rmse": float(
                                    state_rmse[
                                        state_id, horizon_index
                                    ]
                                ),
                            }
                        )
    write_csv(OUT / "stage11_geometry_unit_metrics.csv", rows)
    return rows


def geometry_task_contrasts(rows):
    treatment = "fidelity_constrained_matched_geometry"
    baselines = [
        "frozen",
        "fidelity_constrained_latent_only",
        "fidelity_constrained_shuffled_geometry",
    ]
    contrasts = []
    for environment in ENVIRONMENT:
        for projection_seed in EVALUATION_PROJECTION_SEEDS:
            for horizon in HORIZONS:
                for baseline in baselines:
                    differences = []
                    for adaptation_seed in EXECUTED_ADAPTATION_SEEDS:
                        treatment_rows = {
                            int(row["state_id"]): row
                            for row in rows
                            if row["environment"] == environment
                            and row["method"] == treatment
                            and int(row["adaptation_seed"])
                            == adaptation_seed
                            and int(row["projection_seed"])
                            == projection_seed
                            and int(row["horizon"]) == horizon
                        }
                        baseline_rows = {
                            int(row["state_id"]): row
                            for row in rows
                            if row["environment"] == environment
                            and row["method"] == baseline
                            and (
                                baseline == "frozen"
                                or int(row["adaptation_seed"])
                                == adaptation_seed
                            )
                            and int(row["projection_seed"])
                            == projection_seed
                            and int(row["horizon"]) == horizon
                        }
                        if set(treatment_rows) != set(baseline_rows):
                            raise RuntimeError(
                                "unseen geometry contrast is not paired"
                            )
                        for state_id in treatment_rows:
                            treatment_row = treatment_rows[state_id]
                            baseline_row = baseline_rows[state_id]
                            differences.append(
                                {
                                    "state_id": state_id,
                                    "task_id": int(
                                        treatment_row["task_id"]
                                    ),
                                    "difference": float(
                                        baseline_row[
                                            "whitened_geometry_rmse"
                                        ]
                                        - treatment_row[
                                            "whitened_geometry_rmse"
                                        ]
                                    ),
                                }
                            )
                    collapsed = {}
                    for row in differences:
                        key = (row["task_id"], row["state_id"])
                        collapsed.setdefault(key, []).append(
                            row["difference"]
                        )
                    task_rows = [
                        {
                            "task_id": task_id,
                            "state_id": state_id,
                            "difference": float(np.mean(values)),
                        }
                        for (task_id, state_id), values in collapsed.items()
                    ]
                    result = bootstrap_equal_task_mean(
                        task_rows,
                        "difference",
                        BOOTSTRAP_REPS,
                        SEED
                        + 15001
                        + 103
                        * EVALUATION_PROJECTION_SEEDS.index(
                            projection_seed
                        )
                        + 17 * HORIZONS.index(horizon),
                    )
                    contrasts.append(
                        {
                            "environment": environment,
                            "treatment": treatment,
                            "baseline": baseline,
                            "projection_seed": projection_seed,
                            "horizon": horizon,
                            "metric": "whitened_geometry_rmse",
                            "positive_means_treatment_better": True,
                            "adaptation_seeds_averaged_within_task": len(
                                EXECUTED_ADAPTATION_SEEDS
                            ),
                            **result,
                        }
                    )
    write_csv(
        OUT / "stage11_geometry_task_contrasts.csv", contrasts
    )
    return contrasts


def contrast_lookup(rows, environment, projection_seed, baseline, metric):
    selected = [
        row
        for row in rows
        if row["environment"] == environment
        and int(row["projection_seed"]) == int(projection_seed)
        and row["baseline"] == baseline
        and row["metric"] == metric
    ]
    return {int(row["horizon"]): row for row in selected}


def pilot_decision():
    direct_projection_gates = {}
    readout_projection_gates = {}
    direct_consensus = {}
    readout_consensus = {}
    for environment in ENVIRONMENT:
        direct_projection_gates[environment] = {}
        readout_projection_gates[environment] = {}
        for projection_seed in EVALUATION_PROJECTION_SEEDS:
            direct_sets = []
            for baseline in [
                "frozen",
                "fidelity_constrained_shuffled_geometry",
            ]:
                lookup = contrast_lookup(
                    STAGE11_GEOMETRY_CONTRASTS,
                    environment,
                    projection_seed,
                    baseline,
                    "whitened_geometry_rmse",
                )
                direct_sets.append(
                    {
                        horizon
                        for horizon in HORIZONS
                        if float(lookup[horizon]["estimate"]) > 0
                    }
                )
            direct_common = set.intersection(*direct_sets)
            direct_projection_gates[environment][
                str(projection_seed)
            ] = {
                "pass": len(direct_common) >= 2,
                "common_positive_horizons": sorted(direct_common),
            }

            readout_common = set(HORIZONS)
            for baseline in [
                "frozen",
                "fidelity_constrained_shuffled_geometry",
            ]:
                regret = contrast_lookup(
                    STAGE11_CONTRASTS,
                    environment,
                    projection_seed,
                    baseline,
                    "normalized_regret",
                )
                ranking = contrast_lookup(
                    STAGE11_CONTRASTS,
                    environment,
                    projection_seed,
                    baseline,
                    "weighted_pairwise_accuracy",
                )
                readout_common &= {
                    horizon
                    for horizon in HORIZONS
                    if float(regret[horizon]["estimate"]) > 0
                    and float(ranking[horizon]["estimate"]) > 0
                }
            readout_projection_gates[environment][
                str(projection_seed)
            ] = {
                "pass": len(readout_common) >= 2,
                "common_positive_horizons": sorted(
                    readout_common
                ),
            }
        direct_consensus[environment] = sum(
            row["pass"]
            for row in direct_projection_gates[
                environment
            ].values()
        )
        readout_consensus[environment] = sum(
            row["pass"]
            for row in readout_projection_gates[
                environment
            ].values()
        )

    direct_pass = all(
        direct_consensus[environment]
        >= DIRECT_PROJECTION_CONSENSUS_REQUIRED
        for environment in ENVIRONMENT
    )
    readout_pass = all(
        readout_consensus[environment]
        >= READOUT_PROJECTION_CONSENSUS_REQUIRED
        for environment in ENVIRONMENT
    )
    matched_records = [
        row
        for row in STAGE11_ADAPTATION_MANIFEST
        if row["method"]
        == "fidelity_constrained_matched_geometry"
    ]
    fidelity_pass = bool(
        matched_records
        and all(
            row["fidelity_feasible"]
            and all(
                float(ratio)
                <= 1.0
                + NATIVE_NONINFERIORITY_TOLERANCE
                + 1e-7
                for ratio in row["calibration_native_ratios"]
            )
            for row in matched_records
        )
    )
    unresolved_boundary = any(
        row["undertrained_inconclusive"]
        for row in matched_records
    )
    native_nondestruction = {}
    for environment in ENVIRONMENT:
        selected = [
            row
            for row in STAGE11_NATIVE_PLANNER_CONTRASTS
            if row["environment"] == environment
            and row["baseline"] == "frozen"
            and row["metric"]
            in ["normalized_regret", "weighted_pairwise_accuracy"]
        ]
        native_nondestruction[environment] = bool(
            selected
            and all(
                float(row["estimate"])
                >= -NATIVE_PLANNER_HARM_TOLERANCE
                for row in selected
            )
        )
    native_pass = all(native_nondestruction.values())
    two_seed_stability = len(EXECUTED_ADAPTATION_SEEDS) >= 2

    if not STAGE11_SCREENING_GATE["passed"]:
        decision = "STOP_NO_DIRECT_GEOMETRY_SIGNAL"
    elif not fidelity_pass or not native_pass:
        decision = "STOP_NATIVE_FIDELITY_FAILURE"
    elif direct_pass and readout_pass and two_seed_stability:
        decision = (
            "PROMOTE_TO_FULL_RUN_WITH_EPOCH_EXTENSION"
            if unresolved_boundary
            else "PROMOTE_TO_FULL_RUN"
        )
    elif direct_pass:
        decision = "GEOMETRY_ONLY_DIAGNOSIS"
    else:
        decision = "STOP_NO_ROBUST_UNSEEN_GEOMETRY_GAIN"

    payload = {
        "evidence_status": EVIDENCE_STATUS,
        "decision": decision,
        "pilot_promotes_to_full": decision.startswith(
            "PROMOTE_TO_FULL_RUN"
        ),
        "screening_gate": STAGE11_SCREENING_GATE,
        "executed_adaptation_seeds": EXECUTED_ADAPTATION_SEEDS,
        "two_seed_stability_available": two_seed_stability,
        "direct_geometry_projection_consensus": direct_consensus,
        "direct_geometry_projection_gates": direct_projection_gates,
        "direct_geometry_gate_pass": direct_pass,
        "fresh_readout_projection_consensus": readout_consensus,
        "fresh_readout_projection_gates": readout_projection_gates,
        "fresh_readout_gate_pass": readout_pass,
        "matched_native_fidelity_pass": fidelity_pass,
        "native_planner_nondestruction": native_nondestruction,
        "native_planner_nondestruction_pass": native_pass,
        "unresolved_final_epoch_boundary": unresolved_boundary,
        "interpretation": {
            "promote": (
                "The action path learned label-specific relative dynamics "
                "that transfer to unseen projections and fresh physical "
                "readouts. Run the larger matrix before making a claim."
            ),
            "geometry_only": (
                "The relative latent dynamics improved, but action decisions "
                "did not. The remaining bottleneck is representation/readout "
                "alignment rather than action conditioning."
            ),
            "stop": (
                "The necessary direct geometry signal failed or did not "
                "generalize. Do not spend the full-run compute on ARGA."
            ),
        },
        "guardrail": (
            "This is a compute-gated development pilot on an inspected task "
            "family. Point-estimate projection consensus is a promotion rule, "
            "not confirmatory evidence. A successful pilot must be rerun with "
            "RUN_MODE='full' and then on a numerically new task family."
        ),
    }
    write_json(OUT / "stage11_pilot_decision.json", payload)
    return payload


def plot_stage11():
    methods = [
        "frozen",
        "fidelity_constrained_latent_only",
        "fidelity_constrained_shuffled_geometry",
        "fidelity_constrained_matched_geometry",
    ]
    labels = ["Frozen", "Latent only", "Shuffled geometry", "Matched geometry"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for row_index, environment in enumerate(ENVIRONMENT):
        for column_index, metric in enumerate(
            ["whitened_geometry_rmse", "normalized_regret"]
        ):
            axis = axes[row_index, column_index]
            source = (
                STAGE11_GEOMETRY_ROWS
                if metric == "whitened_geometry_rmse"
                else STAGE11_UNIT_ROWS
            )
            for method, label in zip(methods, labels):
                values = [
                    float(
                        np.mean(
                            [
                                float(row[metric])
                                for row in source
                                if row["environment"] == environment
                                and row["method"] == method
                                and int(row["horizon"]) == horizon
                            ]
                        )
                    )
                    for horizon in HORIZONS
                ]
                axis.plot(HORIZONS, values, marker="o", label=label)
            axis.set_title(f"{environment}: {metric}")
            axis.set_xlabel("horizon")
            axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(
        PLOT_DIR / "stage11_geometry_and_planning.png", dpi=180
    )
    plt.close(fig)


if not PIPELINE_FAILED:
    try:
        STAGE11_GEOMETRY_ROWS = evaluate_unseen_geometry()
        STAGE11_GEOMETRY_CONTRASTS = geometry_task_contrasts(
            STAGE11_GEOMETRY_ROWS
        )
        STAGE11_DECISION = pilot_decision()
        plot_stage11()
        print(json.dumps(STAGE11_DECISION, indent=2))
    except Exception:
        record_failure("stage11_unseen_geometry_and_pilot_gate")
'''

phase_f = "".join(base["cells"][10]["source"])
phase_f = (
    phase_f.replace("stage10", "stage11")
    .replace("Stage 10", "Stage 11")
    .replace(
        "*_fidelity_constrained_matched_fpma_seed*_*.pt",
        "*_fidelity_constrained_matched_geometry_seed*_*.pt",
    )
)

cells = [
    code(config),
    markdown(intro),
    base["cells"][2],
    code(setup),
    shared_definitions,
    phase_a,
    phase_b,
    code(phase_c),
    code(phase_d),
    code(phase_e),
    code(phase_f),
]
for index, cell in enumerate(cells):
    cell["id"] = f"stage11-{index:02d}"

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
