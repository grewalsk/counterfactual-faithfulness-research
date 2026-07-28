import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "09_counterfactual_value_equivalent_adaln.ipynb"
TARGET = ROOT / "10_fidelity_constrained_pairwise_margin_adaptation.ipynb"


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
OUTPUT_DIR = "/content/counterfactual_faithfulness_stage10"
SEED = 913  # held fixed so the complete Stage 7 cache remains compatible
MODEL_NAME = ["jepa_wm_pusht", "jepa_wm_wall"]
ENVIRONMENT = ["PushT", "Wall"]
HORIZONS = [1, 3, 6]
NUM_STATES = 24  # per environment in smoke mode
ACTIONS_PER_STATE = 10

# Full FPMA can outlive a single Colab lease, so checkpoints persist by default.
MOUNT_DRIVE = True
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/counterfactual_faithfulness_stage10"
REUSE_STAGE7_CACHE = True
STAGE7_OUTPUT_DIR = "/content/counterfactual_faithfulness_stage7"
STAGE7_DRIVE_OUTPUT_DIR = (
    "/content/drive/MyDrive/counterfactual_faithfulness_stage7"
)
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
BOOTSTRAP_REPS = 200
RANKING_TIE = 1e-9
RIDGE_LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]  # shared-cell compatibility

# Kept exactly compatible with the Stage 7 truth/transition cache.
AUDIT_PROJECTION_DIM = 64
AUDIT_PROJECTION_SEEDS = [7101]
GOAL_PROJECTION_DIM = 32
GOAL_PROJECTION_SEEDS = [8101, 10101]
ENERGY_HEAD_SEEDS = [8301]
ENERGY_METHODS = []
ENERGY_HIDDEN_DIM = 96
ENERGY_DROPOUT = 0.10
ENERGY_IMPLEMENTATION_ID = "unused_stage10_compatibility"
TRAINING_BATCH_STATES = 1
TRAINING_LR = 3e-4
TRAINING_WEIGHT_DECAY = 1e-3
PAIRWISE_WEIGHT = 1.0
LISTWISE_WEIGHT = 1.0
COST_SHAPE_WEIGHT = 0.25
PAIRWISE_TEMPERATURE = 0.25
LISTWISE_TEMPERATURE = 0.20

# Stage 10: Fidelity-Constrained Pairwise Margin Adaptation (FPMA).
METHOD_IMPLEMENTATION_ID = "stage10_fpma_sum_pnorm_v5"
ADAPTATION_METHODS = [
    "fidelity_constrained_latent_only",
    "fidelity_constrained_shuffled_fpma",
    "fidelity_constrained_matched_fpma",
    "unconstrained_matched_fpma",
]
ADAPTATION_SEEDS = [10401]
INITIAL_EPOCH_LIMIT = 4
EXTENSION_EPOCH_LIMITS = [6, 8]
CHECKPOINT_EVERY = 2
ADAPTATION_LR = 8e-6
ADAPTATION_WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
USE_ACTIVATION_CHECKPOINTING = True

TRAINING_DECODER_COUNT = 3
TRAINING_DECODER_PROJECTION_DIM = 128
TRAINING_DECODER_SEEDS = [10501, 10519, 10537]
EVALUATION_PROJECTION_DIM = 96
EVALUATION_PROJECTION_SEEDS = [12011, 12029, 12047, 12065, 12083]
RIDGE_INITIAL_EXPONENTS = [-8, -6, -4, -2, 0, 2, 4]
RIDGE_EXPANSION_STEPS = 4

PAIRWISE_P = 8
PAIRWISE_SMOOTH_EPS = 1e-6
COST_SCALE_EPS = 1e-6
MIN_NORMALIZED_GAP = 0.02
TOP1_CERTIFICATE_WEIGHT = 0.25
NATIVE_NONINFERIORITY_TOLERANCE = 0.02
NATIVE_PLANNER_HARM_TOLERANCE = 0.02
AUGLAG_BETA = 10.0
GROUP_DRO_ETA = 0.05
GROUP_DRO_LOGIT_CLIP = 20.0
PARAMETER_TRUST_RADIUS = 0.05
TRUST_REFERENCE_RMS_FLOOR = 1e-3
NATIVE_DENOMINATOR_EPS = 1e-8
BOUNDARY_IMPROVEMENT_TOLERANCE = 1e-4
PROJECTION_CONSENSUS_REQUIRED = 4
COLLAPSE_MAX_CANDIDATE_SHARE = 0.80
COLLAPSE_MAX_NULL_SHARE = 0.75
COLLAPSE_MIN_UNIQUE_ACTIONS = 3

DOWNLOAD_RESULTS = True
EVIDENCE_STATUS = "EXPLORATORY_METHOD_DEVELOPMENT"
TASK_FAMILY_ID = "stage5_tasks_reused_for_stage10_development"
DEVELOPMENT_SPLIT = "development_holdout"
STAGE7_TASK_FAMILY_ID = "stage5_tasks_reused_for_stage7_development"
STAGE7_DEVELOPMENT_SPLIT = "development_holdout"

if RUN_MODE == "full":
    NUM_STATES = 96
    AUDIT_PROJECTION_DIM = 128
    AUDIT_PROJECTION_SEEDS = [7101, 9101]
    ADAPTATION_SEEDS = [10401, 10419, 10437]
    INITIAL_EPOCH_LIMIT = 24
    EXTENSION_EPOCH_LIMITS = [36, 48]
    TRAINING_DECODER_PROJECTION_DIM = 256
    EVALUATION_PROJECTION_DIM = 128
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
assert len(TRAINING_DECODER_SEEDS) == TRAINING_DECODER_COUNT == 3
assert len(EVALUATION_PROJECTION_SEEDS) == 5
assert set(TRAINING_DECODER_SEEDS).isdisjoint(EVALUATION_PROJECTION_SEEDS)
assert PAIRWISE_P >= 2
assert INITIAL_EPOCH_LIMIT % CHECKPOINT_EVERY == 0
assert all(
    limit % CHECKPOINT_EVERY == 0 for limit in EXTENSION_EPOCH_LIMITS
)
'''

intro = r'''# Stage 10: fidelity-constrained pairwise margin adaptation

Stage 9 established one real but narrow signal: changing JEPA-WM's actual
action-conditioning pathway reduced long-horizon PushT regret, including under
the native latent planner, but the gain did not transfer across horizons or to
Wall. Its mean endpoint objective could improve a few severe decisions while
still worsening top-1 choice, and its nominal native-fidelity tolerance was
not enforced.

Stage 10 replaces that surrogate with **Fidelity-Constrained Pairwise Margin
Adaptation (FPMA)**. Three goal-independent physical decoders are first fitted
on the frozen JEPA rollouts and then frozen. Only the action encoder and six
AdaLN modulation maps may change. For every state, all ten candidates and all
45 unordered action pairs are evaluated at the same parameter value.

For decoder \(k\), state \(s\), horizon \(h\), and pair \(a<b\), define

\[
e^{(k)}_{ab}
=
\frac{(\hat c^{(k)}_a-\hat c^{(k)}_b)-(c_a-c_b)}
{\max(\max_a c_a-\min_a c_a,\epsilon_q)}
\]

and the robust certificate

\[
B_{s,h}=\max_k
\left[
\sum_{a<b}\left((e^{(k)}_{ab})^2+\epsilon_0^2\right)^{p/2}
\right]^{1/p}.
\]

This is a sum p-norm, never a mean. It upper-bounds every normalized pairwise
margin error represented by each frozen training decoder. Consequently, the
normalized regret of the action selected through any training decoder is at
most \(B_{s,h}\); if the normalized oracle gap is strictly larger than
\(B_{s,h}\), that decoder must select a candidate in the same
\(10^{-9}\)-tolerant optimal set used by the reported top-1 metric.

The objective directly reduces this certificate and its gap-normalized
top-1 bound. A per-horizon augmented Lagrangian encourages native latent
fidelity, while checkpoint eligibility independently enforces
\(A_h(\theta)/A_h(\theta_0)\le 1.02\) for every horizon. Epoch zero is always
eligible. A violating checkpoint is rolled back, including optimizer state,
and its learning rate is halved.

Controls use the identical parameter scope, decoder ensemble, optimizer
budget, and checkpoint opportunities: constrained latent-only adaptation,
constrained FPMA with a non-null derangement of outcome labels, correctly
matched constrained FPMA, and matched FPMA without the native constraint.

Five unseen projections and newly fitted physical readouts test whether the
adapted transition geometry transfers beyond the three training decoders.
They are empirical transfer tests, not covered by the deterministic training
certificate. The native goal-latent planner is also evaluated and must avoid
material harm in both environments. All inferential summaries treat tasks—not
states, projections, or adaptation seeds—as the resampling unit.
'''

setup = "".join(base["cells"][3]["source"])
setup = setup.replace("stage9", "stage10").replace("Stage 9", "Stage 10")
setup = setup.replace(
    '        "TASK_SPLIT_COUNTS": TASK_SPLIT_COUNTS,\n'
    '        "AUDIT_PROJECTION_DIM": AUDIT_PROJECTION_DIM,\n',
    '        "TASK_SPLIT_COUNTS": TASK_SPLIT_COUNTS,\n'
    '        "EVALUATION_SEEDS": EVALUATION_SEEDS,\n'
    '        "TASK_FAMILY_ID": STAGE7_TASK_FAMILY_ID,\n'
    '        "DEVELOPMENT_SPLIT": STAGE7_DEVELOPMENT_SPLIT,\n'
    '        "AUDIT_PROJECTION_DIM": AUDIT_PROJECTION_DIM,\n',
)
setup = setup.replace(
    '    expected = [\n',
    '    checkpoint_manifest_path = STAGE7_ROOT / "checkpoints_manifest.json"\n'
    '    if not checkpoint_manifest_path.exists():\n'
    '        return False, "Stage 7 pretrained-asset manifest is missing"\n'
    '    checkpoint_manifest = json.loads(\n'
    '        checkpoint_manifest_path.read_text()\n'
    '    )\n'
    '    previous_assets = {\n'
    '        Path(row["path"]).name: row["sha256"]\n'
    '        for row in checkpoint_manifest.get("cached_files", [])\n'
    '    }\n'
    '    previous_asset_paths = {\n'
    '        Path(row["path"]).name: Path(row["path"])\n'
    '        for row in checkpoint_manifest.get("cached_files", [])\n'
    '    }\n'
    '    asset_mismatch = [\n'
    '        name\n'
    '        for name, digest in EXPECTED_PRETRAINED_ASSET_SHA256.items()\n'
    '        if previous_assets.get(name) != digest\n'
    '    ]\n'
    '    if asset_mismatch:\n'
    '        return False, (\n'
    '            "Stage 7 pretrained-asset hashes differ: "\n'
    '            f"{asset_mismatch}"\n'
    '        )\n'
    '    revision_mismatch = [\n'
    '        name for name in ["jepa_wm_pusht.pth.tar", '
    '"jepa_wm_wall.pth.tar"]\n'
    '        if EXPECTED_HF_REVISION\n'
    '        not in previous_asset_paths.get(name, Path()).parts\n'
    '    ]\n'
    '    if revision_mismatch:\n'
    '        return False, (\n'
    '            "Stage 7 Hugging Face revision differs: "\n'
    '            f"{revision_mismatch}"\n'
    '        )\n'
    '    expected = [\n',
    1,
)
setup = setup.replace(
    'if (\n'
    '    LOCAL_INTERMEDIATE.exists()\n'
    '    and any(LOCAL_INTERMEDIATE.rglob("state_*.npz"))\n'
    '):\n'
    '    INTERMEDIATE = LOCAL_INTERMEDIATE\n'
    '    CACHE_REUSED = False\n'
    '    CACHE_REUSE_REASON = "resuming Stage 10 cache"\n'
    'else:\n'
    '    CACHE_REUSED, CACHE_REUSE_REASON = stage7_cache_compatible()\n'
    '    INTERMEDIATE = (\n'
    '        STAGE7_ROOT / "intermediate"\n'
    '        if CACHE_REUSED\n'
    '        else LOCAL_INTERMEDIATE\n'
    '    )\n',
    'if (\n'
    '    LOCAL_INTERMEDIATE.exists()\n'
    '    and any(LOCAL_INTERMEDIATE.rglob("state_*.npz"))\n'
    '):\n'
    '    previous_config_path = OUT / "config.json"\n'
    '    if not previous_config_path.exists():\n'
    '        raise RuntimeError(\n'
    '            "local Stage 10 cache has no provenance config; use a new "\n'
    '            "OUTPUT_DIR"\n'
    '        )\n'
    '    previous_local = json.loads(previous_config_path.read_text())\n'
    '    required_local = {\n'
    '        "seed": SEED,\n'
    '        "models": MODEL_NAME,\n'
    '        "environments": ENVIRONMENT,\n'
    '        "horizons": HORIZONS,\n'
    '        "num_states": NUM_STATES,\n'
    '        "actions_per_state": ACTIONS_PER_STATE,\n'
    '        "frameskip": FRAMESKIP,\n'
    '        "target_steps": TARGET_STEPS,\n'
    '        "tasks_per_environment": TASKS_PER_ENVIRONMENT,\n'
    '        "task_split_counts": TASK_SPLIT_COUNTS,\n'
    '        "evaluation_seeds": EVALUATION_SEEDS,\n'
    '        "repo_commit": REPO_COMMIT,\n'
    '        "expected_hf_revision": EXPECTED_HF_REVISION,\n'
    '        "expected_pretrained_asset_sha256": (\n'
    '            EXPECTED_PRETRAINED_ASSET_SHA256\n'
    '        ),\n'
    '        "task_family_id": TASK_FAMILY_ID,\n'
    '        "development_split": DEVELOPMENT_SPLIT,\n'
    '    }\n'
    '    local_mismatch = [\n'
    '        key for key, value in required_local.items()\n'
    '        if previous_local.get(key) != value\n'
    '    ]\n'
    '    if local_mismatch:\n'
    '        raise RuntimeError(\n'
    '            "local Stage 10 cache provenance differs for "\n'
    '            f"{local_mismatch}; use a new OUTPUT_DIR"\n'
    '        )\n'
    '    INTERMEDIATE = LOCAL_INTERMEDIATE\n'
    '    CACHE_REUSED = False\n'
    '    CACHE_REUSE_REASON = "resuming provenance-compatible Stage 10 cache"\n'
    'else:\n'
    '    CACHE_REUSED, CACHE_REUSE_REASON = stage7_cache_compatible()\n'
    '    INTERMEDIATE = (\n'
    '        STAGE7_ROOT / "intermediate"\n'
    '        if CACHE_REUSED\n'
    '        else LOCAL_INTERMEDIATE\n'
    '    )\n',
)
config_start = setup.index("CONFIG = {")
signature_start = setup.index("RUN_SIGNATURE =", config_start)
stage10_config = '''CONFIG = {
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
    "adaptation_seeds": ADAPTATION_SEEDS,
    "initial_epoch_limit": INITIAL_EPOCH_LIMIT,
    "extension_epoch_limits": EXTENSION_EPOCH_LIMITS,
    "checkpoint_every": CHECKPOINT_EVERY,
    "adaptation_lr": ADAPTATION_LR,
    "adaptation_weight_decay": ADAPTATION_WEIGHT_DECAY,
    "gradient_clip": GRADIENT_CLIP,
    "use_activation_checkpointing": USE_ACTIVATION_CHECKPOINTING,
    "training_decoder_count": TRAINING_DECODER_COUNT,
    "training_decoder_projection_dim": TRAINING_DECODER_PROJECTION_DIM,
    "training_decoder_seeds": TRAINING_DECODER_SEEDS,
    "evaluation_projection_dim": EVALUATION_PROJECTION_DIM,
    "evaluation_projection_seeds": EVALUATION_PROJECTION_SEEDS,
    "ridge_initial_exponents": RIDGE_INITIAL_EXPONENTS,
    "ridge_expansion_steps": RIDGE_EXPANSION_STEPS,
    "pairwise_p": PAIRWISE_P,
    "pairwise_smooth_eps": PAIRWISE_SMOOTH_EPS,
    "cost_scale_eps": COST_SCALE_EPS,
    "min_normalized_gap": MIN_NORMALIZED_GAP,
    "top1_certificate_weight": TOP1_CERTIFICATE_WEIGHT,
    "native_noninferiority_tolerance": NATIVE_NONINFERIORITY_TOLERANCE,
    "native_planner_harm_tolerance": NATIVE_PLANNER_HARM_TOLERANCE,
    "auglag_beta": AUGLAG_BETA,
    "group_dro_eta": GROUP_DRO_ETA,
    "group_dro_logit_clip": GROUP_DRO_LOGIT_CLIP,
    "parameter_trust_radius": PARAMETER_TRUST_RADIUS,
    "trust_reference_rms_floor": TRUST_REFERENCE_RMS_FLOOR,
    "native_denominator_eps": NATIVE_DENOMINATOR_EPS,
    "boundary_improvement_tolerance": BOUNDARY_IMPROVEMENT_TOLERANCE,
    "projection_consensus_required": PROJECTION_CONSENSUS_REQUIRED,
    "collapse_max_candidate_share": COLLAPSE_MAX_CANDIDATE_SHARE,
    "collapse_max_null_share": COLLAPSE_MAX_NULL_SHARE,
    "collapse_min_unique_actions": COLLAPSE_MIN_UNIQUE_ACTIONS,
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
setup = setup[:config_start] + stage10_config + setup[signature_start:]

# Phase strings are inserted below.
phase_c = r'''# Phase C — fit frozen physical decoders, then adapt the causal action path.

ADAPTED_DIR = OUT / "adapted_action_paths"
DECODER_DIR = OUT / "frozen_training_decoders"
ADAPTATION_FEATURE_DIR = INTERMEDIATE / "stage10_rollout_features"
for path in [ADAPTED_DIR, DECODER_DIR, ADAPTATION_FEATURE_DIR]:
    path.mkdir(parents=True, exist_ok=True)

HORIZON_INDICES = [horizon - 1 for horizon in HORIZONS]
CONSTRAINED_METHODS = {
    "fidelity_constrained_latent_only",
    "fidelity_constrained_shuffled_fpma",
    "fidelity_constrained_matched_fpma",
}
FPMA_METHODS = {
    "fidelity_constrained_shuffled_fpma",
    "fidelity_constrained_matched_fpma",
    "unconstrained_matched_fpma",
}
PAIR_LEFT, PAIR_RIGHT = pair_indices(ACTIONS_PER_STATE)
assert len(PAIR_LEFT) == ACTIONS_PER_STATE * (ACTIONS_PER_STATE - 1) // 2
assert len({(int(a), int(b)) for a, b in zip(PAIR_LEFT, PAIR_RIGHT)}) == 45


def compute_stage10_cache_binding():
    required = []
    for environment in ENVIRONMENT:
        required.extend(
            (
                f"truth/{environment.lower()}/state_{state_id:04d}.npz",
                TRUTH_ROOT
                / environment.lower()
                / f"state_{state_id:04d}.npz",
            )
            for state_id in range(NUM_STATES)
        )
    for model_name in MODEL_NAME:
        required.extend(
            (
                f"transitions/{model_name}/state_{state_id:04d}.npz",
                TRANSITION_ROOT
                / model_name
                / f"state_{state_id:04d}.npz",
            )
            for state_id in range(NUM_STATES)
        )
        required.append(
            (
                f"goals/{model_name}.npz",
                GOAL_ROOT / f"{model_name}.npz",
            )
        )
    for name in [
        "tasks.json",
        "split_manifest.json",
        "candidate_design_summary.json",
    ]:
        required.append((name, OUT / name))
    missing = [
        logical_path
        for logical_path, path in required
        if not path.exists()
    ]
    if missing:
        raise RuntimeError(
            f"cannot bind incomplete Stage 10 cache: {missing[:8]}"
        )
    records = [
        {
            "logical_path": logical_path,
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for logical_path, path in sorted(required)
    ]
    digest = hashlib.sha256(
        json.dumps(records, sort_keys=True).encode()
    ).hexdigest()
    payload = {
        "run_signature": RUN_SIGNATURE,
        "cache_content_digest": digest,
        "file_count": len(records),
        "records": records,
    }
    write_json(OUT / "stage10_cache_binding.json", payload)
    return digest


CACHE_CONTENT_DIGEST = "unavailable_due_to_earlier_failure"
if not PIPELINE_FAILED:
    try:
        CACHE_CONTENT_DIGEST = compute_stage10_cache_binding()
    except Exception:
        record_failure("stage10_cache_content_binding")


ASSET_HASH_CACHE = {}
PRETRAINED_ASSET_VERIFICATION = {}


def pretrained_asset_digest(model_name):
    required_names = [
        f"{model_name}.pth.tar",
        "dinov2_vits14_pretrain.pth",
    ]
    expected = {
        name: EXPECTED_PRETRAINED_ASSET_SHA256[name]
        for name in required_names
    }
    return hashlib.sha256(
        json.dumps(expected, sort_keys=True).encode()
    ).hexdigest()


def verify_loaded_pretrained_assets(model_name):
    required_names = [
        f"{model_name}.pth.tar",
        "dinov2_vits14_pretrain.pth",
    ]
    roots = [
        Path(os.environ["HF_HOME"]) / "hub",
        Path(os.environ["TORCH_HOME"]),
    ]
    records = []
    for name in required_names:
        candidates = sorted(
            {
                path.absolute()
                for root in roots
                if root.exists()
                for path in root.rglob(name)
                if path.is_file()
            }
        )
        if name.startswith("jepa_wm_"):
            candidates = [
                path
                for path in candidates
                if EXPECTED_HF_REVISION in path.parts
            ]
        expected = EXPECTED_PRETRAINED_ASSET_SHA256[name]
        observed = []
        for path in candidates:
            stat = path.stat()
            key = (str(path), int(stat.st_size), int(stat.st_mtime_ns))
            if key not in ASSET_HASH_CACHE:
                ASSET_HASH_CACHE[key] = sha256_file(path)
            digest = ASSET_HASH_CACHE[key]
            observed.append((path, digest))
        if not observed or any(
            digest != expected for _, digest in observed
        ):
            raise RuntimeError(
                f"pretrained asset {name} does not match the pinned SHA256; "
                f"observed={[digest for _, digest in observed]}"
            )
        matched = observed[0][0]
        records.append(
            {
                "name": name,
                "path": str(matched),
                "size_bytes": int(matched.stat().st_size),
                "sha256": expected,
            }
        )
    digest = pretrained_asset_digest(model_name)
    PRETRAINED_ASSET_VERIFICATION[model_name] = {
        "model": model_name,
        "asset_digest": digest,
        "records": records,
    }
    write_json(
        OUT / "pretrained_asset_verification.json",
        {
            "run_signature": RUN_SIGNATURE,
            "models": PRETRAINED_ASSET_VERIFICATION,
        },
    )
    return digest


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


def action_path_parameter_groups(predictor):
    groups = {
        "action_encoder": [
            parameter
            for _, parameter in predictor.action_encoder.named_parameters()
        ]
    }
    for index, block in enumerate(predictor.predictor_blocks):
        groups[f"adaln_block_{index + 1}"] = [
            parameter
            for _, parameter in block.adaLN_modulation[1].named_parameters()
        ]
    if set(groups) != {
        "action_encoder",
        *{f"adaln_block_{index + 1}" for index in range(6)},
    }:
        raise RuntimeError(f"unexpected action-path groups: {sorted(groups)}")
    if any(not parameters for parameters in groups.values()):
        raise RuntimeError("every action-path trust-region group must be nonempty")
    return groups


def extract_action_path_state(predictor):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in action_path_named_parameters(predictor)
    }


def action_path_checksum(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def validate_action_path_state(state, expected_checksum=None):
    if not state:
        raise RuntimeError("action-path state is empty")
    for name, value in state.items():
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(
                f"non-finite action-path tensor in {name}"
            )
    checksum = action_path_checksum(state)
    if (
        expected_checksum is not None
        and checksum != expected_checksum
    ):
        raise RuntimeError(
            f"action-path checksum mismatch: {checksum} "
            f"!= {expected_checksum}"
        )
    return checksum


def load_action_path_state(predictor, state):
    validate_action_path_state(state)
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
    trainable_names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    allowed_suffixes = {
        name for name, _ in action_path_named_parameters(predictor)
    }
    if trainable:
        if len(trainable_names) != len(allowed_suffixes):
            raise RuntimeError(
                f"trainable parameter count mismatch: {sorted(trainable_names)}"
            )
        if not all(
            any(full_name.endswith(suffix) for full_name in trainable_names)
            for suffix in allowed_suffixes
        ):
            raise RuntimeError(
                "a parameter outside the action path became trainable"
            )
    elif trainable_names:
        raise RuntimeError(
            f"frozen model retained trainable parameters: {sorted(trainable_names)}"
        )
    return predictor


def state_ids_for_split(environment, split_name):
    return [
        record["state_id"]
        for record in build_state_records(environment)
        if record["split"] == split_name
    ]


def deterministic_non_null_derangement(state_id, adaptation_seed):
    rng = np.random.default_rng(
        int(adaptation_seed) + 1009 * int(state_id)
    )
    original = np.arange(1, ACTIONS_PER_STATE, dtype=np.int64)
    permuted = original.copy()
    for _ in range(1000):
        rng.shuffle(permuted)
        if np.all(permuted != original):
            result = np.concatenate(
                [np.asarray([0], dtype=np.int64), permuted]
            )
            assert result[0] == 0
            assert np.all(result[1:] != np.arange(1, ACTIONS_PER_STATE))
            assert sorted(result.tolist()) == list(range(ACTIONS_PER_STATE))
            return result
    raise RuntimeError("failed to construct deterministic non-null derangement")


def validate_training_shard(environment, model_name, state_id, truth, transition):
    expected_horizons = max(HORIZONS)
    pose = pose_target(environment, truth["all_endpoint_states"])
    if pose.shape[:2] != (ACTIONS_PER_STATE, expected_horizons):
        raise RuntimeError(f"bad pose shape for state {state_id}: {pose.shape}")
    if truth["physical_cost"].shape != (ACTIONS_PER_STATE, len(HORIZONS)):
        raise RuntimeError(
            f"bad physical_cost shape for state {state_id}: "
            f"{truth['physical_cost'].shape}"
        )
    for key in ["true_tokens", "base_prediction"]:
        value = transition[key]
        if value.ndim != 4 or value.shape[:2] != (
            ACTIONS_PER_STATE,
            expected_horizons,
        ):
            raise RuntimeError(
                f"bad {key} shape for {model_name} state {state_id}: {value.shape}"
            )
        if value.shape[2] != 16 * 16:
            raise RuntimeError(f"{key} does not contain 256 patch tokens")
    actions = transition["normalized_action"]
    if actions.ndim != 3 or actions.shape[:2] != (
        ACTIONS_PER_STATE,
        expected_horizons,
    ):
        raise RuntimeError(
            f"bad normalized_action shape for state {state_id}: {actions.shape}"
        )


TRAINING_STATE_MEMORY_CACHE = {}


def load_training_state(environment, model_name, state_id):
    cache_key = (environment, model_name, int(state_id))
    if cache_key in TRAINING_STATE_MEMORY_CACHE:
        return TRAINING_STATE_MEMORY_CACHE[cache_key]
    truth_path = (
        TRUTH_ROOT / environment.lower() / f"state_{state_id:04d}.npz"
    )
    transition_path = (
        TRANSITION_ROOT / model_name / f"state_{state_id:04d}.npz"
    )
    with np.load(truth_path) as truth, np.load(transition_path) as transition:
        validate_training_shard(
            environment, model_name, state_id, truth, transition
        )
        payload = {
            "pose": pose_target(
                environment, truth["all_endpoint_states"]
            ).astype(np.float32),
            "physical_cost": truth["physical_cost"].astype(np.float32),
            "actions": transition["normalized_action"].astype(np.float32),
            # Keep large token arrays in their on-disk float16 dtype in host
            # RAM. Each GPU transfer casts only the state currently in use.
            "true_tokens": transition["true_tokens"].copy(),
            "base_prediction": transition["base_prediction"].copy(),
            "task_id": int(truth["task_id"]),
            "split": str(truth["task_split"]),
        }
    TRAINING_STATE_MEMORY_CACHE[cache_key] = payload
    return payload


def clear_training_state_memory_cache(environment):
    stale = [
        key
        for key in TRAINING_STATE_MEMORY_CACHE
        if key[0] == environment
    ]
    for key in stale:
        del TRAINING_STATE_MEMORY_CACHE[key]


def drop_cached_base_predictions(environment):
    for key, payload in TRAINING_STATE_MEMORY_CACHE.items():
        if key[0] == environment:
            payload.pop("base_prediction", None)


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
    if action_batch.shape[0] != ACTIONS_PER_STATE:
        raise RuntimeError(
            "FPMA requires all ten candidates at the same parameter state"
        )
    batch = action_batch.shape[0]
    action_features = model.model.encode_act(action_batch)
    visual_history = initial_encoded["visual"].expand(
        batch, *initial_encoded["visual"].shape[1:]
    )
    proprio_history = initial_encoded["proprio"].expand(
        batch, *initial_encoded["proprio"].shape[1:]
    )
    predictions = []
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
        predictions.append(next_visual[:, 0, 0].flatten(1, 2))
        proprios.append(next_proprio[:, 0])
        visual_history = torch.cat([visual_history, next_visual], dim=1)
        proprio_history = torch.cat(
            [proprio_history, next_proprio], dim=1
        )
    return torch.stack(predictions, dim=1), torch.stack(proprios, dim=1)


def fit_ridge_with_expansion(x_train, y_train, x_calibration, y_calibration):
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    x_calibration = np.asarray(x_calibration, dtype=np.float64)
    y_calibration = np.asarray(y_calibration, dtype=np.float64)
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    train = (x_train - mean) / scale
    calibration = (x_calibration - mean) / scale
    intercept = y_train.mean(axis=0)
    centered_target = y_train - intercept
    gram = train.T @ train
    cross = train.T @ centered_target
    eigenvalue, eigenvector = np.linalg.eigh(gram)
    eigenvalue = np.maximum(eigenvalue, 0.0)
    rotated_cross = eigenvector.T @ cross
    exponents = list(RIDGE_INITIAL_EXPONENTS)
    audit = []

    for expansion in range(RIDGE_EXPANSION_STEPS + 1):
        candidates = []
        for exponent in sorted(set(exponents)):
            ridge = float(10.0 ** exponent)
            coefficient = eigenvector @ (
                rotated_cross / (eigenvalue[:, None] + ridge)
            )
            prediction = calibration @ coefficient + intercept
            loss = float(np.mean((prediction - y_calibration) ** 2))
            candidates.append(
                {
                    "ridge": ridge,
                    "exponent": int(exponent),
                    "calibration_pose_mse": loss,
                    "coefficient": coefficient,
                }
            )
        best_index = int(
            np.argmin(
                [candidate["calibration_pose_mse"] for candidate in candidates]
            )
        )
        audit.append(
            {
                "expansion": expansion,
                "exponents": [item["exponent"] for item in candidates],
                "best_index": best_index,
                "best_ridge": candidates[best_index]["ridge"],
                "best_loss": candidates[best_index]["calibration_pose_mse"],
            }
        )
        if 0 < best_index < len(candidates) - 1:
            best = candidates[best_index]
            return {
                "ridge": best["ridge"],
                "calibration_pose_mse": best["calibration_pose_mse"],
                "mean": mean,
                "scale": scale,
                "intercept": intercept,
                "coefficient": best["coefficient"],
                "ridge_search_audit": audit,
                "ridge_optimum_interior": True,
            }
        if expansion == RIDGE_EXPANSION_STEPS:
            break
        if best_index == 0:
            exponents.insert(0, min(exponents) - 2)
        else:
            exponents.append(max(exponents) + 2)
    raise RuntimeError(
        "ridge optimum remained on a search boundary after all expansions"
    )


def countsketch_checksum(projector):
    digest = hashlib.sha256()
    for value in [projector.bucket, projector.sign, projector.scale]:
        digest.update(
            value.detach().cpu().contiguous().numpy().tobytes()
        )
    return digest.hexdigest()


def fit_frozen_training_decoders(environment, model_name, predictor):
    output_path = (
        DECODER_DIR
        / f"{model_name}_{RUN_SIGNATURE[:12]}_training_decoders.pt"
    )
    if output_path.exists():
        payload = torch.load(
            output_path, map_location="cpu", weights_only=False
        )
        required = {
            "run_signature": RUN_SIGNATURE,
            "environment": environment,
            "model": model_name,
            "projection_seeds": TRAINING_DECODER_SEEDS,
            "cache_content_digest": CACHE_CONTENT_DIGEST,
            "pretrained_asset_digest": pretrained_asset_digest(
                model_name
            ),
        }
        for key, expected in required.items():
            if payload.get(key) != expected:
                raise RuntimeError(
                    f"decoder resume mismatch for {key}: "
                    f"{payload.get(key)} != {expected}"
                )
        return payload

    train_ids = state_ids_for_split(environment, "probe_train")
    calibration_ids = state_ids_for_split(
        environment, "probe_calibration"
    )
    input_dim = 16 * 16 * int(predictor.predictor_embed_dim)
    projectors = [
        CountSketchProjector(
            input_dim,
            TRAINING_DECODER_PROJECTION_DIM,
            projection_seed,
        )
        for projection_seed in TRAINING_DECODER_SEEDS
    ]
    feature_chunks = {
        "probe_train": [[] for _ in projectors],
        "probe_calibration": [[] for _ in projectors],
    }
    pose_chunks = {"probe_train": [], "probe_calibration": []}

    with torch.inference_mode():
        for split_name, state_ids in [
            ("probe_train", train_ids),
            ("probe_calibration", calibration_ids),
        ]:
            for state_id in state_ids:
                shard = load_training_state(
                    environment, model_name, int(state_id)
                )
                prediction = torch.as_tensor(
                    shard["base_prediction"],
                    device="cuda",
                    dtype=torch.float32,
                )
                flattened = prediction.reshape(
                    -1, prediction.shape[-2], prediction.shape[-1]
                )
                for decoder_index, projector in enumerate(projectors):
                    feature_chunks[split_name][decoder_index].append(
                        projector(flattened).detach().cpu().numpy()
                    )
                pose_chunks[split_name].append(
                    shard["pose"].reshape(-1, shard["pose"].shape[-1])
                )

    train_pose = np.concatenate(pose_chunks["probe_train"], axis=0)
    calibration_pose = np.concatenate(
        pose_chunks["probe_calibration"], axis=0
    )
    decoders = []
    manifest_rows = []
    for decoder_index, projection_seed in enumerate(
        TRAINING_DECODER_SEEDS
    ):
        probe = fit_ridge_with_expansion(
            np.concatenate(
                feature_chunks["probe_train"][decoder_index], axis=0
            ),
            train_pose,
            np.concatenate(
                feature_chunks["probe_calibration"][decoder_index], axis=0
            ),
            calibration_pose,
        )
        decoder = {
            "projection_seed": int(projection_seed),
            "projection_dim": int(TRAINING_DECODER_PROJECTION_DIM),
            "projection_checksum": countsketch_checksum(
                projectors[decoder_index]
            ),
            "ridge": float(probe["ridge"]),
            "calibration_pose_mse": float(
                probe["calibration_pose_mse"]
            ),
            "mean": torch.as_tensor(probe["mean"], dtype=torch.float32),
            "scale": torch.as_tensor(probe["scale"], dtype=torch.float32),
            "intercept": torch.as_tensor(
                probe["intercept"], dtype=torch.float32
            ),
            "coefficient": torch.as_tensor(
                probe["coefficient"], dtype=torch.float32
            ),
            "ridge_search_audit": probe["ridge_search_audit"],
            "ridge_optimum_interior": bool(
                probe["ridge_optimum_interior"]
            ),
        }
        decoders.append(decoder)
        coefficient_bytes = (
            decoder["coefficient"].contiguous().numpy().tobytes()
        )
        manifest_rows.append(
            {
                "decoder_index": decoder_index,
                "projection_seed": int(projection_seed),
                "projection_checksum": decoder[
                    "projection_checksum"
                ],
                "ridge": decoder["ridge"],
                "calibration_pose_mse": decoder[
                    "calibration_pose_mse"
                ],
                "ridge_optimum_interior": True,
                "coefficient_sha256": hashlib.sha256(
                    coefficient_bytes
                ).hexdigest(),
                "goal_task_action_inputs": "none",
            }
        )
    payload = {
        "run_signature": RUN_SIGNATURE,
        "environment": environment,
        "model": model_name,
        "cache_content_digest": CACHE_CONTENT_DIGEST,
        "pretrained_asset_digest": pretrained_asset_digest(model_name),
        "projection_seeds": list(TRAINING_DECODER_SEEDS),
        "projection_dim": TRAINING_DECODER_PROJECTION_DIM,
        "decoders": decoders,
        "manifest": manifest_rows,
        "fit_split": "probe_train",
        "ridge_selection_split": "probe_calibration",
        "targets": "goal-independent physical pose only",
        "frozen_before_adaptation": True,
    }
    atomic_torch_save(payload, output_path)
    write_json(
        DECODER_DIR / f"{model_name}_decoder_manifest.json",
        {
            key: value
            for key, value in payload.items()
            if key != "decoders"
        },
    )
    return payload


def cuda_training_decoders(payload, predictor):
    input_dim = 16 * 16 * int(predictor.predictor_embed_dim)
    result = []
    for decoder in payload["decoders"]:
        projector = CountSketchProjector(
            input_dim,
            decoder["projection_dim"],
            decoder["projection_seed"],
        )
        if countsketch_checksum(projector) != decoder[
            "projection_checksum"
        ]:
            raise RuntimeError(
                "reconstructed CountSketch differs from frozen decoder"
            )
        result.append(
            {
                "projector": projector,
                "mean": decoder["mean"].cuda(),
                "scale": decoder["scale"].cuda(),
                "intercept": decoder["intercept"].cuda(),
                "coefficient": decoder["coefficient"].cuda(),
                "projection_seed": decoder["projection_seed"],
            }
        )
    if any(
        value.requires_grad
        for decoder in result
        for key, value in decoder.items()
        if isinstance(value, torch.Tensor)
    ):
        raise RuntimeError("training decoder tensors must remain frozen")
    return result


def training_decoder_checksum(decoders):
    digest = hashlib.sha256()
    for decoder in decoders:
        digest.update(str(decoder["projection_seed"]).encode())
        for key in ["mean", "scale", "intercept", "coefficient"]:
            digest.update(
                decoder[key].detach().cpu().contiguous().numpy().tobytes()
            )
    return digest.hexdigest()


def torch_decoded_task_cost(environment, predicted_pose, task):
    if environment == "PushT":
        goal = torch.as_tensor(
            task["goal"],
            device=predicted_pose.device,
            dtype=predicted_pose.dtype,
        )
        raw_sine = predicted_pose[..., 2]
        raw_cosine = predicted_pose[..., 3]
        norm = torch.sqrt(
            raw_sine.square() + raw_cosine.square() + 1e-12
        )
        small = norm < 1e-4
        sine = torch.where(
            small,
            torch.zeros_like(raw_sine),
            raw_sine / norm,
        )
        cosine = torch.where(
            small,
            torch.ones_like(raw_cosine),
            raw_cosine / norm,
        )
        angle = torch.atan2(sine, cosine)
        angle_error = torch.atan2(
            torch.sin(angle - goal[2]),
            torch.cos(angle - goal[2]),
        )
        pieces = torch.cat(
            [
                predicted_pose[..., :2] - goal[:2] / 512.0,
                (angle_error / math.pi).unsqueeze(-1),
            ],
            dim=-1,
        )
    else:
        goal = torch.as_tensor(
            task["goal"],
            device=predicted_pose.device,
            dtype=predicted_pose.dtype,
        )
        pieces = predicted_pose[..., :2] - goal / 65.0
    return torch.sqrt(
        torch.sum(pieces.square(), dim=-1).clamp_min(1e-12)
    )


def decoded_costs_from_tokens(
    predicted_tokens, environment, task, decoders
):
    actions, steps, tokens, channels = predicted_tokens.shape
    flattened = predicted_tokens.reshape(actions * steps, tokens, channels)
    costs = []
    poses = []
    for decoder in decoders:
        projected = decoder["projector"](flattened)
        standardized = (
            projected - decoder["mean"][None]
        ) / decoder["scale"][None]
        predicted_pose = (
            standardized @ decoder["coefficient"]
            + decoder["intercept"][None]
        ).reshape(actions, steps, -1)
        if not bool(torch.isfinite(predicted_pose).all()):
            raise RuntimeError("physical decoder produced a non-finite pose")
        costs.append(
            torch_decoded_task_cost(
                environment, predicted_pose, task
            )
        )
        poses.append(predicted_pose)
    stacked_costs = torch.stack(costs, dim=0)
    if not bool(torch.isfinite(stacked_costs).all()):
        raise RuntimeError("analytic physical cost is non-finite")
    return stacked_costs, torch.stack(poses, dim=0)


def native_horizon_loss(predicted_tokens, true_tokens):
    predicted = predicted_tokens[:, HORIZON_INDICES]
    target = true_tokens[:, HORIZON_INDICES]
    elementwise = torch_functional.smooth_l1_loss(
        predicted, target, reduction="none"
    )
    return elementwise.mean(dim=(0, 2, 3))


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
    if not np.all(np.isfinite(mean)):
        raise RuntimeError("native train profile is non-finite")
    return {"native_horizon": mean, "native_by_state": by_state}


def pairwise_margin_certificate(predicted_costs, true_cost):
    if predicted_costs.ndim != 3:
        raise RuntimeError(
            f"predicted costs must be [decoder, action, horizon], "
            f"got {tuple(predicted_costs.shape)}"
        )
    if true_cost.shape != (ACTIONS_PER_STATE, len(HORIZONS)):
        raise RuntimeError(
            f"true costs must be [action, horizon], got {tuple(true_cost.shape)}"
        )
    left = torch.as_tensor(
        PAIR_LEFT, device=predicted_costs.device, dtype=torch.long
    )
    right = torch.as_tensor(
        PAIR_RIGHT, device=predicted_costs.device, dtype=torch.long
    )
    scale = (
        true_cost.max(dim=0).values - true_cost.min(dim=0).values
    ).clamp_min(COST_SCALE_EPS)
    predicted_margin = (
        predicted_costs[:, left, :] - predicted_costs[:, right, :]
    )
    true_margin = true_cost[left, :] - true_cost[right, :]
    error = (predicted_margin - true_margin[None]) / scale[None, None]
    # The SUM is theorem-critical: a mean p-norm need not upper-bound max |e|.
    magnitude = torch.sqrt(
        error.square() + PAIRWISE_SMOOTH_EPS ** 2
    )
    maximum = magnitude.amax(dim=1)
    decoder_bound = maximum * torch.sum(
        (magnitude / maximum[:, None, :]) ** PAIRWISE_P,
        dim=1,
    ) ** (1.0 / PAIRWISE_P)
    if not bool(torch.isfinite(decoder_bound).all()):
        raise RuntimeError("pairwise certificate produced non-finite values")
    robust_bound = decoder_bound.max(dim=0).values

    gaps = []
    gap_eligible = []
    gap_exists = []
    for horizon_index in range(len(HORIZONS)):
        column = true_cost[:, horizon_index]
        best = column.min()
        nonoptimal = column > best + RANKING_TIE
        if bool(nonoptimal.any()):
            gap = (column[nonoptimal].min() - best) / scale[horizon_index]
            gaps.append(gap)
            gap_eligible.append(gap >= MIN_NORMALIZED_GAP)
            gap_exists.append(True)
        else:
            gaps.append(column.new_tensor(float("inf")))
            gap_eligible.append(False)
            gap_exists.append(False)
    normalized_gap = torch.stack(gaps)
    eligible = torch.as_tensor(
        gap_eligible, device=true_cost.device, dtype=torch.bool
    )
    finite_gap = torch.as_tensor(
        gap_exists, device=true_cost.device, dtype=torch.bool
    )
    gap_term = torch.zeros_like(robust_bound)
    gap_term[eligible] = (
        robust_bound[eligible] / normalized_gap[eligible]
    )
    decision_loss = (
        robust_bound + TOP1_CERTIFICATE_WEIGHT * gap_term
    )
    reported_wrong_action_bound = torch.zeros_like(robust_bound)
    reported_wrong_action_bound[finite_gap] = torch.clamp(
        robust_bound[finite_gap] / normalized_gap[finite_gap],
        max=1.0,
    )
    return {
        "decision_loss": decision_loss,
        "robust_bound": robust_bound,
        "decoder_bound": decoder_bound,
        "normalized_gap": normalized_gap,
        "gap_eligible": eligible,
        "reported_wrong_action_bound": reported_wrong_action_bound,
    }


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
            reference[name].to(
                named[name].device, named[name].dtype
            )
            for name in names
        ]
        reference_norm = torch.sqrt(
            torch.stack(
                [target.square().sum() for target in targets]
            ).sum()
        )
        count = sum(named[name].numel() for name in names)
        floor = reference_norm.new_tensor(
            TRUST_REFERENCE_RMS_FLOOR
            * math.sqrt(max(count, 1))
        )
        prepared[group_name] = {
            "parameters": [named[name] for name in names],
            "targets": targets,
            "limit": (
                PARAMETER_TRUST_RADIUS
                * torch.maximum(reference_norm, floor)
            ),
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
                parameter.copy_(
                    target + scale * (parameter - target)
                )
            events.append((scale < 1.0).to(torch.float32))
    return torch.stack(events).sum()


def calibration_metrics(
    model,
    method,
    adaptation_seed,
    environment,
    model_name,
    state_ids,
    initial_cache,
    decoders,
):
    tasks_lookup = {
        task["task_id"]: task for task in TASKS[environment]
    }
    native_rows = []
    native_by_state = {}
    decision_rows = []
    with torch.inference_mode():
        for state_id in state_ids:
            shard = load_training_state(
                environment, model_name, int(state_id)
            )
            initial = initial_to_cuda(initial_cache[int(state_id)])
            action_tensor = torch.as_tensor(
                shard["actions"], device="cuda", dtype=torch.float32
            )
            predicted, _ = differentiable_unroll(
                model, initial, action_tensor
            )
            true_tokens = torch.as_tensor(
                shard["true_tokens"],
                device="cuda",
                dtype=torch.float32,
            )
            native_value = (
                native_horizon_loss(predicted, true_tokens)
                .detach()
                .cpu()
                .numpy()
            )
            native_rows.append(native_value)
            native_by_state[int(state_id)] = native_value
            if method == "fidelity_constrained_latent_only":
                continue
            task = tasks_lookup[shard["task_id"]]
            decoder_cost, _ = decoded_costs_from_tokens(
                predicted[:, HORIZON_INDICES],
                environment,
                task,
                decoders,
            )
            true_cost = torch.as_tensor(
                shard["physical_cost"],
                device="cuda",
                dtype=torch.float32,
            )
            if method == "fidelity_constrained_shuffled_fpma":
                permutation = deterministic_non_null_derangement(
                    int(state_id), adaptation_seed
                )
                true_cost = true_cost[
                    torch.as_tensor(
                        permutation, device="cuda", dtype=torch.long
                    )
                ]
            certificate = pairwise_margin_certificate(
                decoder_cost, true_cost
            )
            for horizon_index, value in enumerate(
                certificate["decision_loss"].detach().cpu().numpy()
            ):
                decision_rows.append(
                    {
                        "task_id": shard["task_id"],
                        "horizon_index": horizon_index,
                        "value": float(value),
                    }
                )
    native = np.mean(np.asarray(native_rows), axis=0)
    if not np.all(np.isfinite(native)):
        raise RuntimeError("calibration native profile is non-finite")
    group_values = {}
    for row in decision_rows:
        key = (row["task_id"], row["horizon_index"])
        group_values.setdefault(key, []).append(row["value"])
    group_means = {
        key: float(np.mean(values))
        for key, values in group_values.items()
    }
    if not all(np.isfinite(value) for value in group_means.values()):
        raise RuntimeError("calibration decision score is non-finite")
    worst_decision = (
        float(max(group_means.values()))
        if group_means
        else float("nan")
    )
    mean_decision = (
        float(np.mean(list(group_means.values())))
        if group_means
        else float("nan")
    )
    return {
        "native_horizon": native,
        "native_by_state": native_by_state,
        "worst_task_horizon_decision": worst_decision,
        "mean_task_horizon_decision": mean_decision,
        "group_means": {
            f"task_{key[0]}_h{HORIZONS[key[1]]}": value
            for key, value in group_means.items()
        },
    }


def build_optimizer(parameters, learning_rate):
    return torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=ADAPTATION_WEIGHT_DECAY,
    )


def atomic_torch_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def checkpoint_payload(
    predictor,
    optimizer,
    method,
    adaptation_seed,
    environment,
    model_name,
    epoch,
    score,
    ratios,
    feasible,
    decoder_checksum,
    undertrained_inconclusive=False,
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
        "selected_epoch": int(epoch),
        "calibration_selection_score": float(score),
        "calibration_native_ratios": [
            float(value) for value in ratios
        ],
        "fidelity_feasible": bool(feasible),
        "undertrained_inconclusive": bool(
            undertrained_inconclusive
        ),
        "action_path": extract_action_path_state(predictor),
        "optimizer_state": optimizer_state_to_cpu(
            optimizer.state_dict()
        ),
        "training_decoder_seeds": list(TRAINING_DECODER_SEEDS),
        "training_decoder_checksum": decoder_checksum,
        "config": {
            "lr": ADAPTATION_LR,
            "weight_decay": ADAPTATION_WEIGHT_DECAY,
            "pairwise_p": PAIRWISE_P,
            "native_tolerance": NATIVE_NONINFERIORITY_TOLERANCE,
            "trust_radius": PARAMETER_TRUST_RADIUS,
        },
    }


def train_one_fpma_path(
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
    decoders,
    decoder_checksum,
    baseline_calibration_native,
    baseline_train_native,
    baseline_train_native_by_state,
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
            "training_decoder_checksum": decoder_checksum,
            "cache_content_digest": CACHE_CONTENT_DIGEST,
            "pretrained_asset_digest": pretrained_asset_digest(
                model_name
            ),
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
    dual = np.zeros(len(HORIZONS), dtype=np.float64)
    train_tasks = sorted(
        {
            load_training_state(environment, model_name, int(state_id))[
                "task_id"
            ]
            for state_id in train_ids
        }
    )
    task_index = {
        task_id: index for index, task_id in enumerate(train_tasks)
    }
    group_weight = np.full(
        (len(train_tasks), len(HORIZONS)),
        1.0 / (len(train_tasks) * len(HORIZONS)),
        dtype=np.float64,
    )
    tasks_lookup = {
        task["task_id"]: task for task in TASKS[environment]
    }
    rng = np.random.default_rng(int(adaptation_seed))
    trust_projection_count = torch.zeros(
        (), device="cuda", dtype=torch.float32
    )
    epoch_limits = [INITIAL_EPOCH_LIMIT, *EXTENSION_EPOCH_LIMITS]
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
            "training_decoder_checksum": decoder_checksum,
            "cache_content_digest": CACHE_CONTENT_DIGEST,
            "pretrained_asset_digest": pretrained_asset_digest(
                model_name
            ),
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
        group_weight = np.asarray(
            latest["group_weight"], dtype=np.float64
        )
        if (
            group_weight.shape
            != (len(train_tasks), len(HORIZONS))
            or not np.all(np.isfinite(group_weight))
        ):
            raise RuntimeError("invalid resumed GroupDRO weights")
        rng.bit_generator.state = latest["numpy_rng_state"]
        torch.set_rng_state(latest["torch_rng_state"])
        torch.cuda.set_rng_state_all(latest["cuda_rng_state_all"])
        history = latest["history"]
        eligible_scores = [
            (int(item[0]), float(item[1]))
            for item in latest["eligible_scores"]
        ]
        best = latest["best"]
        last_feasible = latest["last_feasible"]
        validate_action_path_state(best["action_path"])
        validate_action_path_state(last_feasible["action_path"])
        if (
            np.asarray(last_feasible["group_weight"]).shape
            != group_weight.shape
        ):
            raise RuntimeError(
                "invalid rollback GroupDRO state in latest checkpoint"
            )
        trust_projection_count.fill_(
            float(latest["trust_projection_count"])
        )
        limit_index = int(latest["limit_index"])
        active_limit = int(latest["active_limit"])
        epoch = int(latest["epoch"])
        unresolved_boundary = bool(
            latest["unresolved_boundary"]
        )
        if (
            limit_index >= len(epoch_limits)
            or active_limit != epoch_limits[limit_index]
            or epoch % CHECKPOINT_EVERY != 0
        ):
            raise RuntimeError("invalid latest checkpoint state machine")
        log.info(
            "%s %s seed=%d resuming at epoch %d",
            model_name,
            method,
            adaptation_seed,
            epoch,
        )
    else:
        history = []
        eligible_scores = []
        predictor.use_activation_checkpointing = False
        epoch_zero_metrics = calibration_metrics(
            model,
            method,
            adaptation_seed,
            environment,
            model_name,
            calibration_ids,
            initial_cache,
            decoders,
        )
        predictor.use_activation_checkpointing = (
            USE_ACTIVATION_CHECKPOINTING
        )
        epoch_zero_ratios = np.ones(
            len(HORIZONS), dtype=np.float64
        )
        epoch_zero_score = (
            float(np.mean(epoch_zero_metrics["native_horizon"]))
            if method == "fidelity_constrained_latent_only"
            else epoch_zero_metrics["worst_task_horizon_decision"]
        )
        best = checkpoint_payload(
            predictor,
            optimizer,
            method,
            adaptation_seed,
            environment,
            model_name,
            0,
            epoch_zero_score,
            epoch_zero_ratios,
            True,
            decoder_checksum,
        )
        best["group_weight"] = group_weight.copy()
        last_feasible = deepcopy(best)
        best.pop("optimizer_state", None)
        eligible_scores.append((0, epoch_zero_score))
        history.append(
            {
                "environment": environment,
                "model": model_name,
                "method": method,
                "adaptation_seed": adaptation_seed,
                "epoch": 0,
                "selection_score": epoch_zero_score,
                "native_ratio_h1": 1.0,
                "native_ratio_h3": 1.0,
                "native_ratio_h6": 1.0,
                "train_native_ratio_h1": 1.0,
                "train_native_ratio_h3": 1.0,
                "train_native_ratio_h6": 1.0,
                "fidelity_feasible": True,
                "checkpoint_eligible": True,
                "rollback": False,
                "learning_rate": ADAPTATION_LR,
                "trust_projection_count": 0,
                "group_weight_entropy": float(
                    -np.sum(group_weight * np.log(group_weight))
                ),
                "train_decision_h1": float("nan"),
                "train_decision_h3": float("nan"),
                "train_decision_h6": float("nan"),
                "train_native_h1": float("nan"),
                "train_native_h3": float("nan"),
                "train_native_h6": float("nan"),
            }
        )
        limit_index = 0
        active_limit = epoch_limits[limit_index]
        epoch = 0
        unresolved_boundary = False
    while epoch < active_limit:
        epoch += 1
        order = np.asarray(train_ids, dtype=np.int64).copy()
        rng.shuffle(order)
        model.eval()
        epoch_decision = []
        for state_id_value in order:
            state_id = int(state_id_value)
            shard = load_training_state(
                environment, model_name, state_id
            )
            initial = initial_to_cuda(initial_cache[state_id])
            actions = torch.as_tensor(
                shard["actions"], device="cuda", dtype=torch.float32
            )
            true_tokens = torch.as_tensor(
                shard["true_tokens"],
                device="cuda",
                dtype=torch.float32,
            )
            optimizer.zero_grad(set_to_none=True)
            predicted, _ = differentiable_unroll(
                model, initial, actions
            )
            native_by_horizon = native_horizon_loss(
                predicted, true_tokens
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
                decision_objective = native_by_horizon.mean()
                decision_by_horizon = native_by_horizon.detach()
            else:
                task = tasks_lookup[shard["task_id"]]
                decoder_cost, _ = decoded_costs_from_tokens(
                    predicted[:, HORIZON_INDICES],
                    environment,
                    task,
                    decoders,
                )
                true_cost = torch.as_tensor(
                    shard["physical_cost"],
                    device="cuda",
                    dtype=torch.float32,
                )
                if method == "fidelity_constrained_shuffled_fpma":
                    permutation = deterministic_non_null_derangement(
                        state_id, adaptation_seed
                    )
                    true_cost = true_cost[
                        torch.as_tensor(
                            permutation,
                            device="cuda",
                            dtype=torch.long,
                        )
                    ]
                certificate = pairwise_margin_certificate(
                    decoder_cost, true_cost
                )
                decision_by_horizon = certificate["decision_loss"]
                weights = torch.as_tensor(
                    group_weight[task_index[shard["task_id"]]],
                    device="cuda",
                    dtype=torch.float32,
                )
                decision_objective = (
                    len(train_tasks)
                    * torch.sum(weights * decision_by_horizon)
                )

            if method in CONSTRAINED_METHODS:
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
            else:
                augmented_penalty = predicted.new_zeros(())
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
            epoch_decision.append(
                decision_by_horizon.detach().cpu().numpy()
            )
            if method in FPMA_METHODS:
                task_slot = task_index[shard["task_id"]]
                update = np.exp(
                    GROUP_DRO_ETA
                    * np.clip(
                        decision_by_horizon.detach()
                        .cpu()
                        .numpy(),
                        0.0,
                        GROUP_DRO_LOGIT_CLIP,
                    )
                )
                group_weight[task_slot] *= update
                group_weight /= group_weight.sum()
                if not np.all(np.isfinite(group_weight)):
                    raise RuntimeError("GroupDRO weights became non-finite")

        if epoch % CHECKPOINT_EVERY != 0:
            continue

        predictor.use_activation_checkpointing = False
        calibration = calibration_metrics(
            model,
            method,
            adaptation_seed,
            environment,
            model_name,
            calibration_ids,
            initial_cache,
            decoders,
        )
        train_checkpoint_native = native_profile_metrics(
            model,
            environment,
            model_name,
            train_ids,
            initial_cache,
        )
        predictor.use_activation_checkpointing = USE_ACTIVATION_CHECKPOINTING
        ratios = (
            calibration["native_horizon"]
            / np.maximum(
                baseline_calibration_native,
                NATIVE_DENOMINATOR_EPS,
            )
        )
        fidelity_feasible = bool(
            np.all(
                ratios
                <= 1.0
                + NATIVE_NONINFERIORITY_TOLERANCE
                + 1e-7
            )
        )
        checkpoint_eligible = (
            fidelity_feasible
            if method in CONSTRAINED_METHODS
            else True
        )
        score = (
            float(np.mean(calibration["native_horizon"]))
            if method == "fidelity_constrained_latent_only"
            else calibration["worst_task_horizon_decision"]
        )
        if not np.all(np.isfinite(ratios)) or not np.isfinite(score):
            raise RuntimeError(
                f"non-finite checkpoint metrics at epoch {epoch}"
            )
        rollback = False
        if checkpoint_eligible:
            candidate = checkpoint_payload(
                predictor,
                optimizer,
                method,
                adaptation_seed,
                environment,
                model_name,
                epoch,
                score,
                ratios,
                fidelity_feasible,
                decoder_checksum,
            )
            candidate["group_weight"] = group_weight.copy()
            last_feasible = deepcopy(candidate)
            eligible_scores.append((epoch, score))
            if score < best["calibration_selection_score"] - 1e-12:
                best = deepcopy(candidate)
                best.pop("optimizer_state", None)
        else:
            rollback = True
            current_learning_rates = [
                float(group["lr"]) for group in optimizer.param_groups
            ]
            load_action_path_state(
                predictor, last_feasible["action_path"]
            )
            optimizer.load_state_dict(
                deepcopy(last_feasible["optimizer_state"])
            )
            group_weight = np.asarray(
                last_feasible["group_weight"],
                dtype=np.float64,
            ).copy()
            for parameter_group, current_lr in zip(
                optimizer.param_groups, current_learning_rates
            ):
                parameter_group["lr"] = current_lr * 0.5

        train_native_ratio = (
            train_checkpoint_native["native_horizon"]
            / np.maximum(
                baseline_train_native,
                NATIVE_DENOMINATOR_EPS,
            )
        )
        constraint_g = (
            train_native_ratio
            - (1.0 + NATIVE_NONINFERIORITY_TOLERANCE)
        )
        dual = np.maximum(
            0.0, dual + AUGLAG_BETA * constraint_g
        )
        if not np.all(np.isfinite(dual)):
            raise RuntimeError("augmented-Lagrangian dual became non-finite")
        row = {
            "environment": environment,
            "model": model_name,
            "method": method,
            "adaptation_seed": adaptation_seed,
            "epoch": epoch,
            "selection_score": score,
            "native_ratio_h1": float(ratios[0]),
            "native_ratio_h3": float(ratios[1]),
            "native_ratio_h6": float(ratios[2]),
            "train_native_ratio_h1": float(train_native_ratio[0]),
            "train_native_ratio_h3": float(train_native_ratio[1]),
            "train_native_ratio_h6": float(train_native_ratio[2]),
            "fidelity_feasible": fidelity_feasible,
            "checkpoint_eligible": checkpoint_eligible,
            "rollback": rollback,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "trust_projection_count": int(
                trust_projection_count.detach().cpu()
            ),
            "group_weight_entropy": float(
                -np.sum(
                    group_weight
                    * np.log(np.maximum(group_weight, 1e-30))
                )
            ),
            "train_decision_h1": float(
                np.mean(np.asarray(epoch_decision)[:, 0])
            ),
            "train_decision_h3": float(
                np.mean(np.asarray(epoch_decision)[:, 1])
            ),
            "train_decision_h6": float(
                np.mean(np.asarray(epoch_decision)[:, 2])
            ),
            "train_native_h1": float(
                train_checkpoint_native["native_horizon"][0]
            ),
            "train_native_h3": float(
                train_checkpoint_native["native_horizon"][1]
            ),
            "train_native_h6": float(
                train_checkpoint_native["native_horizon"][2]
            ),
        }
        history.append(row)
        log.info(
            "%s %s seed=%d epoch=%d score=%.6f ratios=%s eligible=%s rollback=%s",
            model_name,
            method,
            adaptation_seed,
            epoch,
            score,
            np.round(ratios, 4).tolist(),
            checkpoint_eligible,
            rollback,
        )

        if epoch == active_limit:
            improving_boundary = False
            if (
                best["selected_epoch"] == epoch
                and len(eligible_scores) >= 2
            ):
                improvement = (
                    eligible_scores[-2][1]
                    - eligible_scores[-1][1]
                )
                improving_boundary = (
                    improvement > BOUNDARY_IMPROVEMENT_TOLERANCE
                )
            if improving_boundary and limit_index + 1 < len(epoch_limits):
                limit_index += 1
                active_limit = epoch_limits[limit_index]
                log.info(
                    "%s %s seed=%d extending boundary to epoch %d",
                    model_name,
                    method,
                    adaptation_seed,
                    active_limit,
                )
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
                "training_decoder_checksum": decoder_checksum,
                "epoch": int(epoch),
                "limit_index": int(limit_index),
                "active_limit": int(active_limit),
                "unresolved_boundary": bool(
                    unresolved_boundary
                ),
                "current_action_path": current_action_path,
                "current_action_path_checksum": (
                    action_path_checksum(current_action_path)
                ),
                "optimizer_state": optimizer_state_to_cpu(
                    optimizer.state_dict()
                ),
                "dual": dual.copy(),
                "group_weight": group_weight.copy(),
                "numpy_rng_state": deepcopy(
                    rng.bit_generator.state
                ),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": (
                    torch.cuda.get_rng_state_all()
                ),
                "history": deepcopy(history),
                "eligible_scores": deepcopy(eligible_scores),
                "best": deepcopy(best),
                "last_feasible": deepcopy(last_feasible),
                "trust_projection_count": float(
                    trust_projection_count.detach().cpu()
                ),
            },
            latest_path,
        )

    best["undertrained_inconclusive"] = bool(unresolved_boundary)
    best["base_action_path_checksum"] = base_checksum
    best["selected_action_path_checksum"] = action_path_checksum(
        best["action_path"]
    )
    best["trust_projection_count"] = int(
        trust_projection_count.detach().cpu()
    )
    best["completed_epoch_limit"] = int(active_limit)
    best["all_checkpoint_history"] = history
    best.pop("optimizer_state", None)
    atomic_torch_save(best, output_path)
    if latest_path.exists():
        latest_path.unlink()
    load_action_path_state(predictor, best["action_path"])
    predictor.use_activation_checkpointing = False
    return best


def adapt_fpma_paths():
    manifest = []
    decoder_integrity_records = []
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
        all_adaptation_ids = [*train_ids, *calibration_ids]
        initial_cache = cache_initial_encodings(
            model, environment, all_adaptation_ids
        )
        decoder_payload = fit_frozen_training_decoders(
            environment, model_name, predictor
        )
        drop_cached_base_predictions(environment)
        decoders = cuda_training_decoders(
            decoder_payload, predictor
        )
        decoder_checksum_before = training_decoder_checksum(decoders)
        predictor.use_activation_checkpointing = False
        baseline = calibration_metrics(
            model,
            "fidelity_constrained_matched_fpma",
            ADAPTATION_SEEDS[0],
            environment,
            model_name,
            calibration_ids,
            initial_cache,
            decoders,
        )
        baseline_calibration_native = baseline["native_horizon"]
        baseline_train = native_profile_metrics(
            model,
            environment,
            model_name,
            train_ids,
            initial_cache,
        )
        baseline_train_native = baseline_train["native_horizon"]
        baseline_train_native_by_state = baseline_train[
            "native_by_state"
        ]
        drop_cached_base_predictions(environment)

        for adaptation_seed in ADAPTATION_SEEDS:
            for method in ADAPTATION_METHODS:
                decoder_checksum_method_before = (
                    training_decoder_checksum(decoders)
                )
                if decoder_checksum_method_before != decoder_checksum_before:
                    raise RuntimeError(
                        "frozen decoder changed before a treatment"
                    )
                load_action_path_state(predictor, base_state)
                result = train_one_fpma_path(
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
                    decoders,
                    decoder_checksum_before,
                    baseline_calibration_native,
                    baseline_train_native,
                    baseline_train_native_by_state,
                )
                decoder_checksum_method_after = (
                    training_decoder_checksum(decoders)
                )
                if decoder_checksum_method_after != decoder_checksum_before:
                    raise RuntimeError(
                        "frozen decoder changed during a treatment"
                    )
                manifest.append(
                    {
                        "environment": environment,
                        "model": model_name,
                        "method": method,
                        "adaptation_seed": int(adaptation_seed),
                        "selected_epoch": int(
                            result["selected_epoch"]
                        ),
                        "calibration_selection_score": float(
                            result["calibration_selection_score"]
                        ),
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
                        "training_decoder_checksum": result[
                            "training_decoder_checksum"
                        ],
                        "decoder_checksum_verified_before_and_after": (
                            True
                        ),
                        "cache_content_digest": CACHE_CONTENT_DIGEST,
                        "pretrained_asset_digest": loaded_asset_digest,
                        "trainable_parameter_count": int(
                            sum(
                                parameter.numel()
                                for _, parameter in action_path_named_parameters(
                                    predictor
                                )
                            )
                        ),
                        "trust_projection_count": int(
                            result["trust_projection_count"]
                        ),
                        "completed_epoch_limit": int(
                            result["completed_epoch_limit"]
                        ),
                    }
                )
        decoder_checksum_after = training_decoder_checksum(decoders)
        if decoder_checksum_before != decoder_checksum_after:
            raise RuntimeError("a frozen physical decoder changed during adaptation")
        decoder_integrity_records.append(
            {
                "environment": environment,
                "model": model_name,
                "checksum_before": decoder_checksum_before,
                "checksum_after": decoder_checksum_after,
                "unchanged": True,
            }
        )
        del model, predictor, initial_cache, decoders
        clear_training_state_memory_cache(environment)
        gc.collect()
        torch.cuda.empty_cache()

    history_rows = [
        row
        for environment in ENVIRONMENT
        for model_name in [MODEL_BY_ENVIRONMENT[environment][0]]
        for method in ADAPTATION_METHODS
        for adaptation_seed in ADAPTATION_SEEDS
        for row in torch.load(
            ADAPTED_DIR
            / (
                f"{model_name}_{method}_seed{adaptation_seed}_"
                f"{RUN_SIGNATURE[:12]}.pt"
            ),
            map_location="cpu",
            weights_only=False,
        )["all_checkpoint_history"]
    ]
    write_csv(OUT / "fpma_training_history.csv", history_rows)
    write_json(
        OUT / "fpma_adaptation_manifest.json",
        {
            "run_signature": RUN_SIGNATURE,
            "method_implementation_id": METHOD_IMPLEMENTATION_ID,
            "updated_modules": [
                "predictor.action_encoder",
                "predictor.predictor_blocks.*.adaLN_modulation[1]",
            ],
            "frozen_modules": [
                "visual encoder",
                "predictor attention and MLP content weights",
                "predictor output projection",
                "proprio path",
                "three physical training decoders",
            ],
            "all_actions_weighted_once_per_state": True,
            "unordered_pairs_per_state_horizon": 45,
            "epoch_zero_eligible": True,
            "per_horizon_fidelity_constraint_enforced": True,
            "frozen_decoder_integrity": decoder_integrity_records,
            "records": manifest,
        },
    )
    return manifest


if not PIPELINE_FAILED:
    try:
        STAGE10_ADAPTATION_MANIFEST = adapt_fpma_paths()
    except Exception:
        record_failure("frozen_decoder_fit_and_fpma_adaptation")
'''
phase_d = r'''# Phase D — unseen-projection readouts, certificates, and task inference.


def evaluation_variant_records():
    records = [
        {
            "variant": "frozen",
            "method": "frozen",
            "adaptation_seed": -1,
        }
    ]
    for adaptation_seed in ADAPTATION_SEEDS:
        for method in ADAPTATION_METHODS:
            records.append(
                {
                    "variant": f"{method}__seed{adaptation_seed}",
                    "method": method,
                    "adaptation_seed": int(adaptation_seed),
                }
            )
    return records


EVALUATION_VARIANT_RECORDS = evaluation_variant_records()


def rollout_feature_path(model_name, variant, state_id):
    return (
        ADAPTATION_FEATURE_DIR
        / RUN_SIGNATURE[:12]
        / model_name
        / variant
        / f"state_{state_id:04d}.npz"
    )


def validate_rollout_feature_shard(
    path, record, model_name, expected_action_path_checksum
):
    with np.load(path) as shard:
        if str(shard["run_signature"]) != RUN_SIGNATURE:
            raise RuntimeError(f"stale rollout shard {path}")
        expected_metadata = {
            "variant": record["variant"],
            "method": record["method"],
            "adaptation_seed": int(record["adaptation_seed"]),
            "action_path_checksum": expected_action_path_checksum,
            "cache_content_digest": CACHE_CONTENT_DIGEST,
            "pretrained_asset_digest": pretrained_asset_digest(
                model_name
            ),
        }
        observed_metadata = {
            "variant": str(shard["variant"]),
            "method": str(shard["method"]),
            "adaptation_seed": int(shard["adaptation_seed"]),
            "action_path_checksum": str(
                shard["action_path_checksum"]
            ),
            "cache_content_digest": str(
                shard["cache_content_digest"]
            ),
            "pretrained_asset_digest": str(
                shard["pretrained_asset_digest"]
            ),
        }
        if observed_metadata != expected_metadata:
            raise RuntimeError(
                f"rollout metadata mismatch in {path}: "
                f"{observed_metadata} != {expected_metadata}"
            )
        expected_projected = (
            len(EVALUATION_PROJECTION_SEEDS),
            ACTIONS_PER_STATE,
            max(HORIZONS),
            EVALUATION_PROJECTION_DIM,
        )
        if shard["projected"].shape != expected_projected:
            raise RuntimeError(
                f"bad projected shape {shard['projected'].shape} "
                f"in {path}; expected {expected_projected}"
            )
        expected_decoder_cost = (
            TRAINING_DECODER_COUNT,
            ACTIONS_PER_STATE,
            len(HORIZONS),
        )
        if shard["training_decoder_cost"].shape != expected_decoder_cost:
            raise RuntimeError(
                f"bad training_decoder_cost shape in {path}: "
                f"{shard['training_decoder_cost'].shape}"
            )
        if shard["native_horizon"].shape != (len(HORIZONS),):
            raise RuntimeError(f"bad native_horizon shape in {path}")
        for key in [
            "projected",
            "training_decoder_cost",
            "native_horizon",
            "latent_cost",
        ]:
            if not np.all(np.isfinite(shard[key])):
                raise RuntimeError(
                    f"non-finite rollout field {key} in {path}"
                )


def cache_variant_rollouts():
    repo = configure_repo()
    variant_checksums = {}
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
        predictor.use_activation_checkpointing = False
        base_state = extract_action_path_state(predictor)
        base_checksum = action_path_checksum(base_state)
        initial_cache = cache_initial_encodings(
            model, environment, list(range(NUM_STATES))
        )
        input_dim = 16 * 16 * int(predictor.predictor_embed_dim)
        evaluation_projectors = [
            CountSketchProjector(
                input_dim,
                EVALUATION_PROJECTION_DIM,
                projection_seed,
            )
            for projection_seed in EVALUATION_PROJECTION_SEEDS
        ]
        decoder_payload = torch.load(
            DECODER_DIR
            / f"{model_name}_{RUN_SIGNATURE[:12]}_training_decoders.pt",
            map_location="cpu",
            weights_only=False,
        )
        decoder_expected = {
            "run_signature": RUN_SIGNATURE,
            "environment": environment,
            "model": model_name,
            "cache_content_digest": CACHE_CONTENT_DIGEST,
            "pretrained_asset_digest": loaded_asset_digest,
        }
        for key, value in decoder_expected.items():
            if decoder_payload.get(key) != value:
                raise RuntimeError(
                    f"training decoder mismatch {key}: "
                    f"{decoder_payload.get(key)} != {value}"
                )
        training_decoders = cuda_training_decoders(
            decoder_payload, predictor
        )
        tasks_lookup = {
            task["task_id"]: task for task in TASKS[environment]
        }
        with np.load(GOAL_ROOT / f"{model_name}.npz") as goals:
            goal_visual = goals["visual"].astype(np.float32)

        for record in EVALUATION_VARIANT_RECORDS:
            variant = record["variant"]
            if record["method"] == "frozen":
                load_action_path_state(predictor, base_state)
                selected_action_path_checksum = base_checksum
            else:
                checkpoint_path = (
                    ADAPTED_DIR
                    / (
                        f"{model_name}_{record['method']}_"
                        f"seed{record['adaptation_seed']}_"
                        f"{RUN_SIGNATURE[:12]}.pt"
                    )
                )
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=False,
                )
                expected = {
                    "run_signature": RUN_SIGNATURE,
                    "method": record["method"],
                    "adaptation_seed": record["adaptation_seed"],
                    "environment": environment,
                    "model": model_name,
                    "base_action_path_checksum": base_checksum,
                    "cache_content_digest": CACHE_CONTENT_DIGEST,
                    "pretrained_asset_digest": loaded_asset_digest,
                }
                for key, value in expected.items():
                    if checkpoint.get(key) != value:
                        raise RuntimeError(
                            f"checkpoint mismatch {key} in "
                            f"{checkpoint_path.name}"
                        )
                validate_action_path_state(
                    checkpoint["action_path"],
                    checkpoint["selected_action_path_checksum"],
                )
                selected_action_path_checksum = checkpoint[
                    "selected_action_path_checksum"
                ]
                load_action_path_state(
                    predictor, checkpoint["action_path"]
                )
            variant_checksums[
                f"{model_name}:{variant}"
            ] = selected_action_path_checksum

            for state_id in range(NUM_STATES):
                path = rollout_feature_path(
                    model_name, variant, state_id
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    validate_rollout_feature_shard(
                        path,
                        record,
                        model_name,
                        selected_action_path_checksum,
                    )
                    continue
                shard = load_training_state(
                    environment, model_name, state_id
                )
                initial = initial_to_cuda(initial_cache[state_id])
                actions = torch.as_tensor(
                    shard["actions"],
                    device="cuda",
                    dtype=torch.float32,
                )
                true_tokens = torch.as_tensor(
                    shard["true_tokens"],
                    device="cuda",
                    dtype=torch.float32,
                )
                with torch.inference_mode():
                    predicted, predicted_proprio = differentiable_unroll(
                        model, initial, actions
                    )
                    actions_count, steps, tokens, channels = predicted.shape
                    flattened = predicted.reshape(
                        actions_count * steps, tokens, channels
                    )
                    projected = torch.stack(
                        [
                            projector(flattened).reshape(
                                actions_count,
                                steps,
                                EVALUATION_PROJECTION_DIM,
                            )
                            for projector in evaluation_projectors
                        ],
                        dim=0,
                    )
                    task = tasks_lookup[shard["task_id"]]
                    training_decoder_cost, _ = (
                        decoded_costs_from_tokens(
                            predicted[:, HORIZON_INDICES],
                            environment,
                            task,
                            training_decoders,
                        )
                    )
                    native = native_horizon_loss(
                        predicted, true_tokens
                    )
                    goal = torch.as_tensor(
                        goal_visual[shard["task_id"]],
                        device="cuda",
                        dtype=torch.float32,
                    )
                    latent_cost = torch.sqrt(
                        torch.mean(
                            (predicted - goal[None, None]) ** 2,
                            dim=(2, 3),
                        ).clamp_min(1e-12)
                    )
                atomic_npz_uncompressed(
                    path,
                    run_signature=np.asarray(RUN_SIGNATURE),
                    variant=np.asarray(variant),
                    method=np.asarray(record["method"]),
                    adaptation_seed=np.asarray(
                        record["adaptation_seed"], dtype=np.int64
                    ),
                    action_path_checksum=np.asarray(
                        selected_action_path_checksum
                    ),
                    cache_content_digest=np.asarray(
                        CACHE_CONTENT_DIGEST
                    ),
                    pretrained_asset_digest=np.asarray(
                        loaded_asset_digest
                    ),
                    projected=projected.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float16),
                    training_decoder_cost=training_decoder_cost.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32),
                    native_horizon=native.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32),
                    latent_cost=latent_cost.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32),
                    predicted_proprio=predicted_proprio.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float16),
                )
                validate_rollout_feature_shard(
                    path,
                    record,
                    model_name,
                    selected_action_path_checksum,
                )
            log.info("%s cached Stage 10 variant %s", model_name, variant)
        del (
            model,
            predictor,
            initial_cache,
            training_decoders,
            evaluation_projectors,
        )
        clear_training_state_memory_cache(environment)
        gc.collect()
        torch.cuda.empty_cache()
    return variant_checksums


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
            pose = pose_target(
                environment, shard["all_endpoint_states"]
            )
            if pose.shape[:2] != (
                ACTIONS_PER_STATE,
                max(HORIZONS),
            ):
                raise RuntimeError(
                    f"truth axis mismatch for {environment} state {state_id}"
                )
            payload["pose"].append(pose[:, HORIZON_INDICES])
            payload["physical_cost"].append(shard["physical_cost"])
            payload["interactions"].append(shard["interactions"])
            payload["task_id"].append(int(shard["task_id"]))
            payload["split"].append(str(shard["task_split"]))
            payload["evaluation_seed"].append(
                int(shard["evaluation_seed"])
            )
    result = {
        key: np.asarray(value) for key, value in payload.items()
    }
    if result["physical_cost"].shape != (
        NUM_STATES,
        ACTIONS_PER_STATE,
        len(HORIZONS),
    ):
        raise RuntimeError(
            f"canonical cost axis must be [state, action, horizon], got "
            f"{result['physical_cost'].shape}"
        )
    return result


def load_variant_features(model_name, variant, projection_index):
    projected = []
    latent_cost = []
    native_horizon = []
    training_decoder_cost = []
    record = next(
        row
        for row in EVALUATION_VARIANT_RECORDS
        if row["variant"] == variant
    )
    expected_checksum = STAGE10_VARIANT_CHECKSUMS[
        f"{model_name}:{variant}"
    ]
    for state_id in range(NUM_STATES):
        path = rollout_feature_path(model_name, variant, state_id)
        if projection_index == 0:
            validate_rollout_feature_shard(
                path, record, model_name, expected_checksum
            )
        with np.load(path) as shard:
            projected.append(
                shard["projected"][projection_index][
                    :, HORIZON_INDICES, :
                ].astype(np.float32)
            )
            latent_cost.append(
                shard["latent_cost"][:, HORIZON_INDICES].astype(
                    np.float64
                )
            )
            native_horizon.append(
                shard["native_horizon"].astype(np.float64)
            )
            training_decoder_cost.append(
                shard["training_decoder_cost"].astype(np.float64)
            )
    return {
        "projected": np.asarray(projected),
        "latent_cost": np.asarray(latent_cost),
        "native_horizon": np.asarray(native_horizon),
        "training_decoder_cost": np.asarray(training_decoder_cost),
    }


def split_indices(truth, name):
    return np.flatnonzero(truth["split"] == name)


def flatten_selected(values, indices):
    selected = np.asarray(values)[indices]
    return selected.reshape(-1, selected.shape[-1])


def numpy_pairwise_certificate(predicted_cost, true_cost):
    prediction = np.asarray(predicted_cost, dtype=np.float64)
    truth = np.asarray(true_cost, dtype=np.float64)
    scale = max(float(np.max(truth) - np.min(truth)), COST_SCALE_EPS)
    error = (
        (prediction[PAIR_LEFT] - prediction[PAIR_RIGHT])
        - (truth[PAIR_LEFT] - truth[PAIR_RIGHT])
    ) / scale
    bound = float(
        np.sum(
            (error ** 2 + PAIRWISE_SMOOTH_EPS ** 2)
            ** (PAIRWISE_P / 2.0)
        )
        ** (1.0 / PAIRWISE_P)
    )
    best = float(np.min(truth))
    nonoptimal = truth > best + RANKING_TIE
    gap = (
        float(np.min(truth[nonoptimal]) - best) / scale
        if np.any(nonoptimal)
        else float("inf")
    )
    ranking = ranking_metrics(truth, prediction)
    certificate_normalized_regret = (
        float(truth[ranking["selected_action"]] - best) / scale
    )
    holds = certificate_normalized_regret <= bound + 1e-7
    certified = bool(np.isfinite(gap) and bound < gap)
    if not holds:
        raise RuntimeError(
            f"certificate violation: regret={certificate_normalized_regret} "
            f"bound={bound}"
        )
    if certified and not bool(ranking["top1_correct"]):
        raise RuntimeError("strict gap certificate predicted a wrong action")
    return {
        "certificate_bound": bound,
        "normalized_gap": gap,
        "certificate_normalized_regret_q": (
            certificate_normalized_regret
        ),
        "conventional_normalized_regret": ranking[
            "normalized_regret"
        ],
        "regret_bound_holds": bool(holds),
        "top1_certified": certified,
        "top1_correct": bool(ranking["top1_correct"]),
        "selected_action": ranking["selected_action"],
        "oracle_action": ranking["oracle_action"],
    }


def bootstrap_equal_task_mean(rows, value_key, repetitions, seed):
    task_values = {}
    for row in rows:
        value = float(row[value_key])
        if np.isfinite(value):
            task_values.setdefault(int(row["task_id"]), []).append(
                value
            )
    task_means = {
        task_id: float(np.mean(values))
        for task_id, values in task_values.items()
    }
    task_ids = sorted(task_means)
    values = np.asarray(
        [task_means[task_id] for task_id in task_ids],
        dtype=np.float64,
    )
    if len(values) == 0:
        return {
            "estimate": float("nan"),
            "low": float("nan"),
            "high": float("nan"),
            "n_tasks": 0,
            "n_bootstrap": int(repetitions),
            "task_means_json": "{}",
        }
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sampled = rng.integers(0, len(values), size=len(values))
        draws[index] = float(np.mean(values[sampled]))
    return {
        "estimate": float(np.mean(values)),
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
        "n_tasks": int(len(values)),
        "n_bootstrap": int(repetitions),
        "task_means_json": json.dumps(
            {str(key): value for key, value in task_means.items()},
            sort_keys=True,
        ),
    }


def evaluate_stage10_methods():
    unit_rows = []
    probe_rows = []
    certificate_rows = []
    native_rows = []
    native_planner_rows = []
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

        for projection_index, projection_seed in enumerate(
            EVALUATION_PROJECTION_SEEDS
        ):
            for record in EVALUATION_VARIANT_RECORDS:
                loaded = load_variant_features(
                    model_name,
                    record["variant"],
                    projection_index,
                )
                features = loaded["projected"]
                probe = fit_ridge_with_expansion(
                    flatten_selected(features, train_ids),
                    flatten_selected(truth["pose"], train_ids),
                    flatten_selected(features, calibration_ids),
                    flatten_selected(truth["pose"], calibration_ids),
                )
                predicted_pose = (
                    (
                        features.reshape(-1, features.shape[-1])
                        - probe["mean"][None]
                    )
                    / probe["scale"][None]
                    @ probe["coefficient"]
                    + probe["intercept"][None]
                ).reshape(
                    *features.shape[:-1], truth["pose"].shape[-1]
                )
                probe_rows.append(
                    {
                        "environment": environment,
                        "model": model_name,
                        "variant": record["variant"],
                        "method": record["method"],
                        "adaptation_seed": record["adaptation_seed"],
                        "evaluation_projection_seed": projection_seed,
                        "evaluation_projection_dim": (
                            EVALUATION_PROJECTION_DIM
                        ),
                        "ridge": probe["ridge"],
                        "calibration_pose_mse": probe[
                            "calibration_pose_mse"
                        ],
                        "ridge_optimum_interior": probe[
                            "ridge_optimum_interior"
                        ],
                        "ridge_search_audit_json": json.dumps(
                            probe["ridge_search_audit"],
                            sort_keys=True,
                        ),
                    }
                )

                for state_id in development_ids:
                    state_id = int(state_id)
                    task_id = int(truth["task_id"][state_id])
                    task = tasks_lookup[environment][task_id]
                    predicted_cost = decoded_task_cost(
                        environment,
                        predicted_pose[state_id],
                        task,
                    )
                    true_cost = truth["physical_cost"][state_id]
                    for horizon_index, horizon in enumerate(HORIZONS):
                        ranking = ranking_metrics(
                            true_cost[:, horizon_index],
                            predicted_cost[:, horizon_index],
                        )
                        pose_error = float(
                            np.mean(
                                physical_pose_error(
                                    environment,
                                    predicted_pose[
                                        state_id, :, horizon_index
                                    ],
                                    truth["pose"][
                                        state_id, :, horizon_index
                                    ],
                                )
                            )
                        )
                        unit_rows.append(
                            {
                                "environment": environment,
                                "model": model_name,
                                "state_id": state_id,
                                "task_id": task_id,
                                "evaluation_seed": int(
                                    truth["evaluation_seed"][state_id]
                                ),
                                "split": DEVELOPMENT_SPLIT,
                                "variant": record["variant"],
                                "method": record["method"],
                                "adaptation_seed": record[
                                    "adaptation_seed"
                                ],
                                "projection_seed": int(
                                    projection_seed
                                ),
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
                                "oracle_action": ranking[
                                    "oracle_action"
                                ],
                                "selected_is_null": int(
                                    ranking["selected_action"] == 0
                                ),
                            }
                        )

                if projection_index == 0:
                    for state_id in development_ids:
                        state_id = int(state_id)
                        task_id = int(truth["task_id"][state_id])
                        for horizon_index, horizon in enumerate(HORIZONS):
                            native_ranking = ranking_metrics(
                                truth["physical_cost"][
                                    state_id, :, horizon_index
                                ],
                                loaded["latent_cost"][
                                    state_id, :, horizon_index
                                ],
                            )
                            native_planner_rows.append(
                                {
                                    "environment": environment,
                                    "model": model_name,
                                    "state_id": state_id,
                                    "task_id": task_id,
                                    "variant": record["variant"],
                                    "method": record["method"],
                                    "adaptation_seed": record[
                                        "adaptation_seed"
                                    ],
                                    "horizon": horizon,
                                    "normalized_regret": native_ranking[
                                        "normalized_regret"
                                    ],
                                    "weighted_pairwise_accuracy": (
                                        native_ranking[
                                            "weighted_pairwise_accuracy"
                                        ]
                                    ),
                                    "top1_correct": native_ranking[
                                        "top1_correct"
                                    ],
                                    "selected_action": native_ranking[
                                        "selected_action"
                                    ],
                                    "oracle_action": native_ranking[
                                        "oracle_action"
                                    ],
                                }
                            )
                            native_rows.append(
                                {
                                    "environment": environment,
                                    "model": model_name,
                                    "state_id": state_id,
                                    "task_id": task_id,
                                    "variant": record["variant"],
                                    "method": record["method"],
                                    "adaptation_seed": record[
                                        "adaptation_seed"
                                    ],
                                    "horizon": horizon,
                                    "native_anchor": float(
                                        loaded["native_horizon"][
                                            state_id, horizon_index
                                        ]
                                    ),
                                }
                            )
                            for decoder_index in range(
                                TRAINING_DECODER_COUNT
                            ):
                                certificate = (
                                    numpy_pairwise_certificate(
                                        loaded[
                                            "training_decoder_cost"
                                        ][
                                            state_id,
                                            decoder_index,
                                            :,
                                            horizon_index,
                                        ],
                                        truth["physical_cost"][
                                            state_id, :, horizon_index
                                        ],
                                    )
                                )
                                certificate_rows.append(
                                    {
                                        "environment": environment,
                                        "model": model_name,
                                        "state_id": state_id,
                                        "task_id": task_id,
                                        "variant": record["variant"],
                                        "method": record["method"],
                                        "adaptation_seed": record[
                                            "adaptation_seed"
                                        ],
                                        "decoder_index": decoder_index,
                                        "decoder_projection_seed": (
                                            TRAINING_DECODER_SEEDS[
                                                decoder_index
                                            ]
                                        ),
                                        "horizon": horizon,
                                        **certificate,
                                    }
                                )

    write_csv(OUT / "stage10_unit_metrics.csv", unit_rows)
    write_csv(OUT / "stage10_probe_selection.csv", probe_rows)
    write_csv(OUT / "stage10_certificate_metrics.csv", certificate_rows)
    write_csv(OUT / "stage10_native_fidelity.csv", native_rows)
    write_csv(
        OUT / "stage10_native_planner_metrics.csv",
        native_planner_rows,
    )
    return (
        unit_rows,
        probe_rows,
        certificate_rows,
        native_rows,
        native_planner_rows,
    )


def summarize_stage10_units(unit_rows):
    summary = []
    metrics = [
        "normalized_regret",
        "weighted_pairwise_accuracy",
        "top1_correct",
        "normalized_margin_rmse",
        "pose_error",
        "selected_is_null",
    ]
    keys = sorted(
        {
            (
                row["environment"],
                row["method"],
                int(row["adaptation_seed"]),
                int(row["projection_seed"]),
                int(row["horizon"]),
            )
            for row in unit_rows
        }
    )
    for (
        environment,
        method,
        adaptation_seed,
        projection_seed,
        horizon,
    ) in keys:
        selected = [
            row
            for row in unit_rows
            if row["environment"] == environment
            and row["method"] == method
            and int(row["adaptation_seed"]) == adaptation_seed
            and int(row["projection_seed"]) == projection_seed
            and int(row["horizon"]) == horizon
        ]
        for metric in metrics:
            result = bootstrap_equal_task_mean(
                selected,
                metric,
                BOOTSTRAP_REPS,
                SEED
                + 101 * ENVIRONMENT.index(environment)
                + 17 * HORIZONS.index(horizon)
                + int(projection_seed),
            )
            summary.append(
                {
                    "environment": environment,
                    "method": method,
                    "adaptation_seed": adaptation_seed,
                    "projection_seed": projection_seed,
                    "horizon": horizon,
                    "metric": metric,
                    **result,
                }
            )
    write_csv(OUT / "stage10_task_clustered_summary.csv", summary)
    return summary


def paired_task_contrasts(unit_rows):
    treatment = "fidelity_constrained_matched_fpma"
    baselines = [
        "frozen",
        "fidelity_constrained_latent_only",
        "fidelity_constrained_shuffled_fpma",
        "unconstrained_matched_fpma",
    ]
    metrics = [
        "normalized_regret",
        "weighted_pairwise_accuracy",
        "top1_correct",
    ]
    contrasts = []
    for environment in ENVIRONMENT:
        for projection_seed in EVALUATION_PROJECTION_SEEDS:
            for horizon in HORIZONS:
                for baseline in baselines:
                    for metric in metrics:
                        state_seed_differences = []
                        for adaptation_seed in ADAPTATION_SEEDS:
                            treatment_rows = {
                                int(row["state_id"]): row
                                for row in unit_rows
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
                                for row in unit_rows
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
                                    "paired contrast has missing or extra state rows"
                                )
                            for state_id in sorted(treatment_rows):
                                treatment_row = treatment_rows[state_id]
                                baseline_row = baseline_rows[state_id]
                                if (
                                    int(treatment_row["task_id"])
                                    != int(baseline_row["task_id"])
                                ):
                                    raise RuntimeError(
                                        "paired contrast task mismatch"
                                    )
                                sign = (
                                    -1.0
                                    if metric == "normalized_regret"
                                    else 1.0
                                )
                                treatment_value = float(
                                    treatment_row[metric]
                                )
                                baseline_value = float(
                                    baseline_row[metric]
                                )
                                if not (
                                    np.isfinite(treatment_value)
                                    and np.isfinite(baseline_value)
                                ):
                                    continue
                                state_seed_differences.append(
                                    {
                                        "state_id": state_id,
                                        "task_id": int(
                                            treatment_row["task_id"]
                                        ),
                                        "adaptation_seed": adaptation_seed,
                                        "difference": sign
                                        * (
                                            treatment_value
                                            - baseline_value
                                        ),
                                    }
                                )
                        state_seed_average = {}
                        for row in state_seed_differences:
                            key = (row["task_id"], row["state_id"])
                            state_seed_average.setdefault(key, []).append(
                                row["difference"]
                            )
                        collapsed = [
                            {
                                "task_id": task_id,
                                "state_id": state_id,
                                "difference": float(np.mean(values)),
                            }
                            for (task_id, state_id), values in (
                                state_seed_average.items()
                            )
                        ]
                        result = bootstrap_equal_task_mean(
                            collapsed,
                            "difference",
                            BOOTSTRAP_REPS,
                            SEED
                            + 7001
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
                                "metric": metric,
                                "positive_means_treatment_better": True,
                                "adaptation_seeds_averaged_within_task": (
                                    len(ADAPTATION_SEEDS)
                                ),
                                "paired_state_seed_rows": len(
                                    state_seed_differences
                                ),
                                **result,
                            }
                        )
    write_csv(OUT / "stage10_task_clustered_contrasts.csv", contrasts)
    return contrasts


def paired_native_planner_contrasts(native_planner_rows):
    treatment = "fidelity_constrained_matched_fpma"
    baselines = [
        "frozen",
        "fidelity_constrained_latent_only",
        "fidelity_constrained_shuffled_fpma",
        "unconstrained_matched_fpma",
    ]
    metrics = [
        "normalized_regret",
        "weighted_pairwise_accuracy",
        "top1_correct",
    ]
    contrasts = []
    for environment in ENVIRONMENT:
        for horizon in HORIZONS:
            for baseline in baselines:
                for metric in metrics:
                    differences = []
                    for adaptation_seed in ADAPTATION_SEEDS:
                        treatment_rows = {
                            int(row["state_id"]): row
                            for row in native_planner_rows
                            if row["environment"] == environment
                            and row["method"] == treatment
                            and int(row["adaptation_seed"])
                            == adaptation_seed
                            and int(row["horizon"]) == horizon
                        }
                        baseline_rows = {
                            int(row["state_id"]): row
                            for row in native_planner_rows
                            if row["environment"] == environment
                            and row["method"] == baseline
                            and (
                                baseline == "frozen"
                                or int(row["adaptation_seed"])
                                == adaptation_seed
                            )
                            and int(row["horizon"]) == horizon
                        }
                        if set(treatment_rows) != set(baseline_rows):
                            raise RuntimeError(
                                "native planner contrast is not exactly paired"
                            )
                        for state_id in treatment_rows:
                            treatment_row = treatment_rows[state_id]
                            baseline_row = baseline_rows[state_id]
                            sign = (
                                -1.0
                                if metric == "normalized_regret"
                                else 1.0
                            )
                            treatment_value = float(
                                treatment_row[metric]
                            )
                            baseline_value = float(
                                baseline_row[metric]
                            )
                            if not (
                                np.isfinite(treatment_value)
                                and np.isfinite(baseline_value)
                            ):
                                continue
                            differences.append(
                                {
                                    "state_id": state_id,
                                    "task_id": int(
                                        treatment_row["task_id"]
                                    ),
                                    "adaptation_seed": adaptation_seed,
                                    "difference": sign
                                    * (
                                        treatment_value
                                        - baseline_value
                                    ),
                                }
                            )
                    collapsed_lookup = {}
                    for row in differences:
                        key = (row["task_id"], row["state_id"])
                        collapsed_lookup.setdefault(key, []).append(
                            row["difference"]
                        )
                    collapsed = [
                        {
                            "task_id": task_id,
                            "state_id": state_id,
                            "difference": float(np.mean(values)),
                        }
                        for (task_id, state_id), values in (
                            collapsed_lookup.items()
                        )
                    ]
                    result = bootstrap_equal_task_mean(
                        collapsed,
                        "difference",
                        BOOTSTRAP_REPS,
                        SEED
                        + 11003
                        + 31 * HORIZONS.index(horizon),
                    )
                    contrasts.append(
                        {
                            "environment": environment,
                            "treatment": treatment,
                            "baseline": baseline,
                            "horizon": horizon,
                            "metric": metric,
                            "positive_means_treatment_better": True,
                            "adaptation_seeds_averaged_within_task": (
                                len(ADAPTATION_SEEDS)
                            ),
                            **result,
                        }
                    )
    write_csv(
        OUT / "stage10_native_planner_contrasts.csv",
        contrasts,
    )
    return contrasts


if not PIPELINE_FAILED:
    try:
        STAGE10_VARIANT_CHECKSUMS = cache_variant_rollouts()
        (
            STAGE10_UNIT_ROWS,
            STAGE10_PROBE_ROWS,
            STAGE10_CERTIFICATE_ROWS,
            STAGE10_NATIVE_ROWS,
            STAGE10_NATIVE_PLANNER_ROWS,
        ) = evaluate_stage10_methods()
        STAGE10_SUMMARY = summarize_stage10_units(
            STAGE10_UNIT_ROWS
        )
        STAGE10_CONTRASTS = paired_task_contrasts(
            STAGE10_UNIT_ROWS
        )
        STAGE10_NATIVE_PLANNER_CONTRASTS = (
            paired_native_planner_contrasts(
                STAGE10_NATIVE_PLANNER_ROWS
            )
        )
    except Exception:
        record_failure("fresh_projection_transfer_and_task_inference")
'''
phase_e = r'''# Phase E — prospective gate and compact diagnostics.


def contrast_rows(environment, projection_seed, baseline, metric):
    selected = [
        row
        for row in STAGE10_CONTRASTS
        if row["environment"] == environment
        and int(row["projection_seed"]) == int(projection_seed)
        and row["baseline"] == baseline
        and row["metric"] == metric
    ]
    if len(selected) != len(HORIZONS):
        raise RuntimeError(
            f"expected {len(HORIZONS)} contrast rows, found {len(selected)}"
        )
    return {
        int(row["horizon"]): row for row in selected
    }


def method_collapse_diagnostics():
    result = {}
    method = "fidelity_constrained_matched_fpma"
    for environment in ENVIRONMENT:
        strata = {}
        for adaptation_seed in ADAPTATION_SEEDS:
            for projection_seed in EVALUATION_PROJECTION_SEEDS:
                for horizon in HORIZONS:
                    selected = [
                        row
                        for row in STAGE10_UNIT_ROWS
                        if row["environment"] == environment
                        and row["method"] == method
                        and int(row["adaptation_seed"])
                        == adaptation_seed
                        and int(row["projection_seed"])
                        == projection_seed
                        and int(row["horizon"]) == horizon
                    ]
                    action_counts = {
                        action: int(
                            sum(
                                int(row["selected_action"]) == action
                                for row in selected
                            )
                        )
                        for action in range(ACTIONS_PER_STATE)
                    }
                    total = max(len(selected), 1)
                    maximum_share = max(action_counts.values()) / total
                    null_share = action_counts[0] / total
                    unique_actions = int(
                        sum(
                            count > 0
                            for count in action_counts.values()
                        )
                    )
                    stratum_pass = bool(
                        maximum_share
                        <= COLLAPSE_MAX_CANDIDATE_SHARE
                        and null_share <= COLLAPSE_MAX_NULL_SHARE
                        and unique_actions
                        >= COLLAPSE_MIN_UNIQUE_ACTIONS
                    )
                    strata[
                        (
                            f"seed_{adaptation_seed}_"
                            f"projection_{projection_seed}_h{horizon}"
                        )
                    ] = {
                        "rows": len(selected),
                        "action_counts": action_counts,
                        "maximum_candidate_share": maximum_share,
                        "null_share": null_share,
                        "unique_selected_actions": unique_actions,
                        "pass": stratum_pass,
                    }
        result[environment] = {
            "strata": strata,
            "pass": bool(
                strata and all(row["pass"] for row in strata.values())
            ),
        }
    return result


def stage10_decision():
    full_scientific_matrix = bool(
        RUN_MODE == "full"
        and NUM_STATES == 96
        and HORIZONS == [1, 3, 6]
        and ACTIONS_PER_STATE == 10
        and TASKS_PER_ENVIRONMENT == 12
        and TASK_SPLIT_COUNTS == [6, 3, 0, 3]
        and ADAPTATION_SEEDS == [10401, 10419, 10437]
        and ADAPTATION_METHODS
        == [
            "fidelity_constrained_latent_only",
            "fidelity_constrained_shuffled_fpma",
            "fidelity_constrained_matched_fpma",
            "unconstrained_matched_fpma",
        ]
        and INITIAL_EPOCH_LIMIT == 24
        and EXTENSION_EPOCH_LIMITS == [36, 48]
        and TRAINING_DECODER_COUNT == 3
        and TRAINING_DECODER_SEEDS == [10501, 10519, 10537]
        and EVALUATION_PROJECTION_SEEDS
        == [12011, 12029, 12047, 12065, 12083]
        and BOOTSTRAP_REPS == 2000
    )
    expected_development_tasks = TASK_SPLIT_COUNTS[3]
    complete_task_inference = bool(
        STAGE10_CONTRASTS
        and STAGE10_NATIVE_PLANNER_CONTRASTS
        and all(
            int(row["n_tasks"]) == expected_development_tasks
            and np.isfinite(float(row["estimate"]))
            for row in [
                *STAGE10_CONTRASTS,
                *STAGE10_NATIVE_PLANNER_CONTRASTS,
            ]
        )
    )
    projection_gates = {}
    baselines = [
        "frozen",
        "fidelity_constrained_latent_only",
        "fidelity_constrained_shuffled_fpma",
    ]
    for environment in ENVIRONMENT:
        projection_gates[environment] = {}
        for projection_seed in EVALUATION_PROJECTION_SEEDS:
            baseline_results = {}
            common_positive = set(HORIZONS)
            common_interval_supported = set(HORIZONS)
            for baseline in baselines:
                regret = contrast_rows(
                    environment,
                    projection_seed,
                    baseline,
                    "normalized_regret",
                )
                ranking = contrast_rows(
                    environment,
                    projection_seed,
                    baseline,
                    "weighted_pairwise_accuracy",
                )
                jointly_positive = [
                    horizon
                    for horizon in HORIZONS
                    if float(regret[horizon]["estimate"]) > 0
                    and float(ranking[horizon]["estimate"]) > 0
                ]
                interval_supported = [
                    horizon
                    for horizon in HORIZONS
                    if float(regret[horizon]["low"]) > 0
                    and float(ranking[horizon]["low"]) > 0
                ]
                baseline_results[baseline] = {
                    "jointly_positive_horizons": jointly_positive,
                    "interval_supported_horizons": interval_supported,
                    "regret": regret,
                    "ranking": ranking,
                }
                common_positive &= set(jointly_positive)
                common_interval_supported &= set(interval_supported)
            projection_pass = (
                len(common_interval_supported) >= 2
            )
            projection_gates[environment][str(projection_seed)] = {
                "pass": projection_pass,
                "common_jointly_positive_horizons": sorted(
                    common_positive
                ),
                "common_interval_supported_horizons": sorted(
                    common_interval_supported
                ),
                "baselines": baseline_results,
            }

    projection_consensus = {
        environment: int(
            sum(
                projection_gates[environment][str(seed)]["pass"]
                for seed in EVALUATION_PROJECTION_SEEDS
            )
        )
        for environment in ENVIRONMENT
    }
    native_planner_gates = {}
    for environment in ENVIRONMENT:
        frozen_rows = [
            row
            for row in STAGE10_NATIVE_PLANNER_CONTRASTS
            if row["environment"] == environment
            and row["baseline"] == "frozen"
        ]
        regret = {
            int(row["horizon"]): row
            for row in frozen_rows
            if row["metric"] == "normalized_regret"
        }
        ranking = {
            int(row["horizon"]): row
            for row in frozen_rows
            if row["metric"] == "weighted_pairwise_accuracy"
        }
        if set(regret) != set(HORIZONS) or set(ranking) != set(
            HORIZONS
        ):
            raise RuntimeError(
                "native planner gate is missing a horizon"
            )
        jointly_positive = [
            horizon
            for horizon in HORIZONS
            if float(regret[horizon]["estimate"]) > 0
            and float(ranking[horizon]["estimate"]) > 0
        ]
        no_material_harm = all(
            float(regret[horizon]["estimate"])
            >= -NATIVE_PLANNER_HARM_TOLERANCE
            and float(ranking[horizon]["estimate"])
            >= -NATIVE_PLANNER_HARM_TOLERANCE
            for horizon in HORIZONS
        )
        native_planner_gates[environment] = {
            "jointly_positive_horizons": jointly_positive,
            "no_material_harm_all_horizons": no_material_harm,
            "harm_tolerance": NATIVE_PLANNER_HARM_TOLERANCE,
            "pass": bool(
                len(jointly_positive) >= 1 and no_material_harm
            ),
            "regret": regret,
            "ranking": ranking,
        }
    native_planner_pass = all(
        native_planner_gates[environment]["pass"]
        for environment in ENVIRONMENT
    )
    collapse = method_collapse_diagnostics()
    matched_records = [
        row
        for row in STAGE10_ADAPTATION_MANIFEST
        if row["method"] == "fidelity_constrained_matched_fpma"
    ]
    constrained_records = [
        row
        for row in STAGE10_ADAPTATION_MANIFEST
        if row["method"] in CONSTRAINED_METHODS
    ]
    matched_fidelity = bool(
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
    constrained_epoch_zero_fallback_valid = bool(
        constrained_records
        and all(row["fidelity_feasible"] for row in constrained_records)
    )
    no_unresolved_boundary = not any(
        row["undertrained_inconclusive"]
        for row in STAGE10_ADAPTATION_MANIFEST
    )
    ridge_integrity = bool(
        STAGE10_PROBE_ROWS
        and all(
            bool(row["ridge_optimum_interior"])
            for row in STAGE10_PROBE_ROWS
        )
    )
    certificate_integrity = bool(
        STAGE10_CERTIFICATE_ROWS
        and all(
            bool(row["regret_bound_holds"])
            and (
                not bool(row["top1_certified"])
                or bool(row["top1_correct"])
            )
            for row in STAGE10_CERTIFICATE_ROWS
        )
    )
    projection_seed_integrity = bool(
        len(set(EVALUATION_PROJECTION_SEEDS)) == 5
        and set(TRAINING_DECODER_SEEDS).isdisjoint(
            EVALUATION_PROJECTION_SEEDS
        )
    )
    cross_environment_gain = all(
        projection_consensus[environment]
        >= PROJECTION_CONSENSUS_REQUIRED
        for environment in ENVIRONMENT
    )
    no_collapse = all(
        collapse[environment]["pass"]
        for environment in ENVIRONMENT
    )
    advancement = all(
        [
            full_scientific_matrix,
            matched_fidelity,
            constrained_epoch_zero_fallback_valid,
            no_unresolved_boundary,
            ridge_integrity,
            certificate_integrity,
            projection_seed_integrity,
            complete_task_inference,
            cross_environment_gain,
            native_planner_pass,
            no_collapse,
        ]
    )
    if not full_scientific_matrix:
        decision = "NONPROTOCOL_RUN_NO_SCIENTIFIC_DECISION"
    elif not complete_task_inference:
        decision = "INCOMPLETE_TASK_INFERENCE"
    elif not no_unresolved_boundary:
        decision = "UNDERTRAINED_INCONCLUSIVE"
    elif not matched_fidelity:
        decision = "NATIVE_FIDELITY_FAILURE"
    elif advancement:
        decision = "ADVANCE_TO_NEW_TASK_CONFIRMATION"
    else:
        decision = "NO_ROBUST_FPMA_GAIN"
    payload = {
        "evidence_status": EVIDENCE_STATUS,
        "decision": decision,
        "advancement_gate_passed": advancement,
        "full_scientific_matrix_complete": full_scientific_matrix,
        "criterion": (
            "matched constrained FPMA must be calibration-feasible at every "
            "horizon, jointly improve regret and weighted ranking over frozen, "
            "latent-only, and shuffled controls at >=2/3 horizons in >=4/5 "
            "unseen projections in both environments, avoid candidate "
            "collapse, preserve and improve the native latent planner, and "
            "have no unresolved final checkpoint boundary"
        ),
        "projection_consensus_required": (
            PROJECTION_CONSENSUS_REQUIRED
        ),
        "projection_consensus_observed": projection_consensus,
        "projection_gates": projection_gates,
        "native_planner_gates": native_planner_gates,
        "native_planner_gate_pass": native_planner_pass,
        "matched_native_fidelity_pass": matched_fidelity,
        "all_constrained_selected_checkpoints_feasible": (
            constrained_epoch_zero_fallback_valid
        ),
        "no_unresolved_checkpoint_boundary": (
            no_unresolved_boundary
        ),
        "fresh_ridge_optima_interior": ridge_integrity,
        "complete_three_task_inference": complete_task_inference,
        "deterministic_certificate_integrity": (
            certificate_integrity
        ),
        "training_evaluation_seed_disjointness": (
            projection_seed_integrity
        ),
        "candidate_collapse_diagnostics": collapse,
        "interpretation_guardrail": (
            "This is development evidence on an inspected task family with "
            "three development tasks per environment. Projection seeds and "
            "adaptation seeds are stability checks, not independent samples. "
            "The deterministic certificate is conditional on each frozen "
            "training decoder and does not prove fresh-readout transfer."
        ),
    }
    write_json(
        OUT / "stage10_development_decision.json", payload
    )
    return payload


def plot_stage10():
    methods = [
        "frozen",
        "fidelity_constrained_latent_only",
        "fidelity_constrained_shuffled_fpma",
        "fidelity_constrained_matched_fpma",
        "unconstrained_matched_fpma",
    ]
    labels = [
        "Frozen",
        "Latent only",
        "Shuffled FPMA",
        "Matched FPMA",
        "Unconstrained FPMA",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for row_index, environment in enumerate(ENVIRONMENT):
        for column_index, metric in enumerate(
            ["normalized_regret", "weighted_pairwise_accuracy"]
        ):
            axis = axes[row_index, column_index]
            for method, label in zip(methods, labels):
                values = []
                for horizon in HORIZONS:
                    selected = [
                        float(row[metric])
                        for row in STAGE10_UNIT_ROWS
                        if row["environment"] == environment
                        and row["method"] == method
                        and int(row["horizon"]) == horizon
                    ]
                    values.append(float(np.mean(selected)))
                axis.plot(
                    HORIZONS,
                    values,
                    marker="o",
                    label=label,
                )
            axis.set_title(f"{environment}: {metric}")
            axis.set_xlabel("horizon")
            axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(
        PLOT_DIR / "stage10_fpma_planning_comparison.png",
        dpi=180,
    )
    plt.close(fig)

    matched = [
        row
        for row in STAGE10_CERTIFICATE_ROWS
        if row["method"] == "fidelity_constrained_matched_fpma"
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, environment in zip(axes, ENVIRONMENT):
        selected = [
            row for row in matched
            if row["environment"] == environment
        ]
        axis.scatter(
            [float(row["certificate_bound"]) for row in selected],
            [
                float(row["certificate_normalized_regret_q"])
                for row in selected
            ],
            alpha=0.25,
            s=14,
        )
        limit = max(
            [
                1.0,
                *[
                    float(row["certificate_bound"])
                    for row in selected
                ],
            ]
        )
        axis.plot([0, limit], [0, limit], linestyle="--", color="black")
        axis.set_title(f"{environment}: certified decoder regret")
        axis.set_xlabel("pairwise certificate B")
        axis.set_ylabel("normalized regret")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        PLOT_DIR / "stage10_certificate_check.png",
        dpi=180,
    )
    plt.close(fig)


if not PIPELINE_FAILED:
    try:
        STAGE10_DECISION = stage10_decision()
        plot_stage10()
        print(json.dumps(STAGE10_DECISION, indent=2))
    except Exception:
        record_failure("stage10_decision_gate_and_plots")
'''

phase_f = r'''# Phase F — package non-cache evidence and download one result bundle.


def package_stage10_results():
    result_zip = Path("/content/stage10_result_bundle.zip")
    if not PIPELINE_FAILED:
        (OUT / "FAILURE_TRACE.txt").write_text(
            "SUCCESS: no captured pipeline failure\n"
        )
    excluded_roots = {
        str(INTERMEDIATE.resolve()),
        str(CACHE_ROOT.resolve()),
        str(ADAPTED_DIR.resolve()),
    }
    result_manifest_path = OUT / "result_zip_manifest.json"
    files = []
    for path in OUT.rglob("*"):
        if not path.is_file():
            continue
        if path == result_manifest_path:
            continue
        resolved = str(path.resolve())
        if any(
            resolved == root or resolved.startswith(root + os.sep)
            for root in excluded_roots
        ):
            continue
        files.append(path)

    # Keep the scientifically relevant matched checkpoints in the bundle.
    # Control checkpoints remain resumable in OUTPUT_DIR but would make the
    # automatic download unnecessarily large.
    matched_checkpoints = sorted(
        path
        for path in ADAPTED_DIR.glob(
            "*_fidelity_constrained_matched_fpma_seed*_*.pt"
        )
        if not path.name.endswith("_latest.pt")
        and RUN_SIGNATURE[:12] in path.name
    )
    files.extend(matched_checkpoints)
    failed_run_latest_checkpoints = []
    if PIPELINE_FAILED:
        # At most one treatment should be unfinished in the sequential state
        # machine, but include every atomic latest file if a failure occurred.
        failed_run_latest_checkpoints = sorted(
            path
            for path in ADAPTED_DIR.glob("*_latest.pt")
            if RUN_SIGNATURE[:12] in path.name
        )
        files.extend(failed_run_latest_checkpoints)
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
        result_manifest_path,
        {
            "run_signature": RUN_SIGNATURE,
            "pipeline_failed": bool(PIPELINE_FAILED),
            "matched_checkpoints_included": len(
                matched_checkpoints
            ),
            "failed_run_latest_checkpoints_included": len(
                failed_run_latest_checkpoints
            ),
            "completed_control_checkpoints_excluded_from_download": True,
            "files": manifest,
        },
    )
    files = sorted(
        {
            *files,
            result_manifest_path,
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
    if MOUNT_DRIVE:
        drive_copy = OUT.parent / result_zip.name
        try:
            shutil.copy2(result_zip, drive_copy)
            print(f"DRIVE_RESULT_ZIP: {drive_copy}")
        except Exception as drive_copy_error:
            log.warning(
                "Drive result copy failed; local zip remains at %s: %s",
                result_zip,
                drive_copy_error,
            )
            print(
                "DRIVE_COPY_WARNING: result remains at "
                f"{result_zip}"
            )
    if DOWNLOAD_RESULTS:
        try:
            from google.colab import files as colab_files

            colab_files.download(str(result_zip))
        except Exception as download_error:
            log.warning(
                "automatic browser download failed; result remains at %s: %s",
                result_zip,
                download_error,
            )
            print(
                "DOWNLOAD_WARNING: automatic download failed; "
                f"retrieve {result_zip} manually"
            )
    return result_zip


try:
    RESULT_ZIP = package_stage10_results()
except Exception:
    record_failure("stage10_packaging")
    raise
'''

shared_definitions = json.loads(json.dumps(base["cells"][4]))
shared_source = "".join(shared_definitions["source"]).replace(
    "        remaining = primitive_steps - step\n",
    "",
)
repo_pin_patch = r'''    # Pin the exact Hugging Face snapshot used by the Stage 7 cache.
    hubconf_text = hubconf.read_text()
    filename_marker = 'filename=f"{model_name}.pth.tar",'
    revision_marker = f'revision="{EXPECTED_HF_REVISION}"'
    if revision_marker not in hubconf_text:
        if hubconf_text.count(filename_marker) != 1:
            raise RuntimeError(
                "cannot pin the checkpoint revision in hubconf.py"
            )
        hubconf_text = hubconf_text.replace(
            filename_marker,
            filename_marker
            + f'\n            revision="{EXPECTED_HF_REVISION}",',
        )
        hubconf.write_text(hubconf_text)
    hubconf_text = hubconf.read_text()
    fallback_marker = (
        "        except Exception:\n"
        "            # Fall back to fbaipublicfiles URL\n"
        "            pass\n"
    )
    fail_closed_marker = "pinned Hugging Face checkpoint retrieval failed"
    if fail_closed_marker not in hubconf_text:
        if hubconf_text.count(fallback_marker) != 1:
            raise RuntimeError(
                "cannot disable mutable checkpoint fallback in hubconf.py"
            )
        fallback_replacement = (
            "        except Exception as error:\n"
            "            raise RuntimeError(\n"
            '                "pinned Hugging Face checkpoint retrieval failed"\n'
            "            ) from error\n"
        )
        hubconf_text = hubconf_text.replace(
            fallback_marker,
            fallback_replacement,
        )
        hubconf.write_text(hubconf_text)

'''
shared_source = shared_source.replace(
    "    # PushT and Wall do not use the DROID pose helper.\n",
    repo_pin_patch
    + "    # PushT and Wall do not use the DROID pose helper.\n",
)
shared_source = shared_source.replace(
    '        angle = np.arctan2(prediction[..., 2], prediction[..., 3])\n',
    '        raw_sine = prediction[..., 2]\n'
    '        raw_cosine = prediction[..., 3]\n'
    '        norm = np.sqrt(raw_sine ** 2 + raw_cosine ** 2 + 1e-12)\n'
    '        small = norm < 1e-4\n'
    '        sine = np.where(small, 0.0, raw_sine / norm)\n'
    '        cosine = np.where(small, 1.0, raw_cosine / norm)\n'
    '        angle = np.arctan2(sine, cosine)\n',
)
shared_definitions["source"] = shared_source.splitlines(
    keepends=True
)

cells = [
    code(config),
    markdown(intro),
    base["cells"][2],
    code(setup),
    shared_definitions,
    base["cells"][5],
    base["cells"][6],
    code(phase_c),
    code(phase_d),
    code(phase_e),
    code(phase_f),
]
for index, cell in enumerate(cells):
    cell["id"] = f"stage10-{index:02d}"

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
