import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "31_cross_model_grounded_closure_certificate.ipynb"
NUMERICAL = (
    ROOT.parent / "src/cf_faithfulness/stage31_cross_model_certificate.py"
)
STAGE14_NUMERICAL = ROOT.parent / "src/cf_faithfulness/stage14_pcj.py"
STAGE17_NUMERICAL = ROOT.parent / "src/cf_faithfulness/stage17_action_contrast.py"
STAGE18_NUMERICAL = ROOT.parent / "src/cf_faithfulness/stage18_rank_confirmation.py"

spec = importlib.util.spec_from_file_location(
    "stage30_builder", ROOT / "build_stage30_grounded_planning_value_notebook.py"
)
STAGE30 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(STAGE30)

code = STAGE30.code
markdown = STAGE30.markdown
assigned_uppercase_names = STAGE30.assigned_uppercase_names
function_sources = STAGE30.function_sources


introduction = r'''# Stage 31: cross-model grounded causal closure certificate

Stage 30 found that physically grounded causal closure predicted held-out
planning regret inside one JEPA-WM, while the same carrier's self-consistent
causal effect and ordinary prediction error did not suffice.  It did **not**
show that the carrier was uniquely necessary for planning.  The highest-value
next question is therefore not another carrier optimization on the inspected
model.  It is whether grounded closure is a model-comparative reliability
certificate rather than a checkpoint-specific correlation.

This notebook compares the official public `jepa_wm_pusht` and
`dino_wm_pusht` checkpoints on exactly paired, fresh PushT counterfactuals.
The two models share the physical tasks and DINOv2 target family but differ in
world-model architecture, action conditioning, and training objective.  For
each model, a rank-128 action-consequence carrier is learned only from a
separate construction split using the frozen Stage 18 recipe: construction
action contrasts, a construction-only layer screen, channel whitening, and an
output-aligned ridge/SVD subspace.  No simulator outcome magnitude, evaluation
activation, planning label, or grounded-closure label enters that fit.

Evaluation uses the Stage 30 histogram-, impulse-, energy-, and duration-
matched signed-area schedules on new states.  Closure is measured on the four
interior schedules; terminal goals are the two excluded extreme schedules.
The primary planner score is now the exact public objective,

\[
c_m(a;g)=\operatorname{MSE}(\hat z^v_{m,a},z^{v,*}_{m,g})
+0.1\,\operatorname{MSE}(\hat z^p_{m,a},z^{p,*}_{m,g}),
\]

not the Stage 30 visual-only sensitivity.  Joint closure is measured in the
Euclidean chart whose squared norm is exactly this objective.

The primary estimand is paired across models.  For each physical state and
magnitude,

\[
Y=R_{\mathrm{DINO}}-R_{\mathrm{JEPA}}.
\]

A state-grouped out-of-fold model first predicts `Y` from magnitude, contact
regime, ordinary target error, and self-consistent causal closure differences.
The preregistered test asks whether adding the *difference in physically
grounded closure* lowers held-out MSE.  Separate within-model replications,
matched ablation controls, visual-only sensitivity, and exact free-motion
nulls are secondary.

This is a method-generalization experiment, not a claim that the two learned
bases share coordinates.  It uses no decoder, reader, Jacobian, JVP, VJP,
gradient, or evaluation-set subspace tuning.  Return
`stage31_cross_model_certificate_result_bundle_<signature>.zip`.
'''


configuration = r'''# SINGLE CONFIGURATION BLOCK — no Stage 31 secrets required.
# Public checkpoints download without a token; a cached optional HF_TOKEN is
# used automatically when present.  The nonce is fresh on every Run all.
import secrets as _secrets
import time as _time

RUN_MODE = "pilot"
EXPERIMENT_SOURCE_REF = "codex/stage31-cross-model-grounded-certificate"
RUN_NONCE = f"auto_{_time.strftime('%Y%m%d_%H%M%S')}_{_secrets.token_hex(4)}"

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
MAX_ESTIMATED_TOTAL_MINUTES = 180.0
FRESH_RUN_REQUIRED = True

OUTPUT_DIR = "/content/counterfactual_faithfulness_stage31_cross_model"
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/counterfactual_faithfulness_stage31_cross_model"

PROTOCOL_ID = "stage31-cross-model-grounded-closure-certificate-v1"
NOTEBOOK_PROTOCOL_SHA256 = "__PROTOCOL_DIGEST__"
EVIDENCE_STATUS = "CONFIRMATORY_ONLY_IF_SOURCE_BOUND_FRESH_CONSTRUCTION_FROZEN_AND_PAIRED"
EXPERIMENT_REPOSITORY = "grewalsk/counterfactual-faithfulness-research"
EXPERIMENT_NOTEBOOK_PATH = "notebooks/31_cross_model_grounded_closure_certificate.ipynb"
EXPERIMENT_BUILDER_PATH = "notebooks/build_stage31_cross_model_grounded_certificate_notebook.py"
EXPERIMENT_NUMERICAL_PATH = "src/cf_faithfulness/stage31_cross_model_certificate.py"

SEED = 31101
DESIGN_SEED = 31137
BOOTSTRAP_SEED = 31269
CROSSFIT_SEED = 31319
PERMUTATION_SEED = 31357
NULL_ROOT_SEED = 31411
ENVIRONMENT = "PushT"
MODEL_NAMES = ["jepa_wm_pusht", "dino_wm_pusht"]
MODEL_SHORT_NAMES = {"jepa_wm_pusht": "jepa", "dino_wm_pusht": "dino"}
EXPECTED_MODEL_TYPES = {"jepa_wm_pusht": "AdaLN", "dino_wm_pusht": "dino_wm"}
EXPECTED_CARRIER_WIDTHS = {"jepa_wm_pusht": 400, "dino_wm_pusht": 414}
FRAMESKIP = 5
PRIMARY_HORIZON = 3
TARGET_STEPS = [PRIMARY_HORIZON]
ACTION_STEPS = PRIMARY_HORIZON * FRAMESKIP
PREDICTOR_BLOCKS = list(range(6))

CONSTRUCTION_POOL_TRAJECTORIES = list(range(4000, 4160))
EVALUATION_POOL_TRAJECTORIES = list(range(4200, 4600))
CONSTRUCTION_TARGET_PER_CONTACT_STRATUM = 16
CONSTRUCTION_TARGET = 32
EVALUATION_TARGET_PER_STRATUM = 40
EVALUATION_TARGET = 120
TASK_ID_OFFSET = 31000
DISTANCE_GRID = [55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0]
STRATUM_LABELS = ["persistent_contact", "boundary_switching", "free"]
CONTACT_STRATA = ["persistent_contact", "boundary_switching"]

CONSTRUCTION_ACTION_MAGNITUDE = 0.12
CONSTRUCTION_ACTIONS_PER_STATE = 13
MIN_CONSTRUCTION_COST_SPREAD = 0.02
MIN_CONSTRUCTION_NON_TIED_PAIR_FRACTION = 0.20
MIN_CONSTRUCTION_CONTACT_BRANCHES = 2
PHYSICAL_COST_TIE = 1e-6

SELECTED_MAGNITUDES = [0.10, 0.14, 0.18, 0.22]
MAGNITUDE_COUNT = 4
SCHEDULE_STRINGS = [
    "uuuuuvvvvv", "uuvuuvvuvv", "uvuvuvuvuv",
    "vuvuvuvuvu", "vvuvvuuvuu", "vvvvvuuuuu",
]
SCHEDULE_COUNT = 6
SCHEDULE_INVERSION_COUNTS = [0, 5, 10, 15, 20, 25]
SIGNED_AREA_LEVELS = [25, 15, 5, -5, -15, -25]
ANGLE_PAIR_DEGREES = [-30.0, 30.0]
EVALUATION_ACTIONS_PER_STATE = MAGNITUDE_COUNT * SCHEDULE_COUNT
DIAGNOSTIC_SCHEDULES = [1, 2, 3, 4]
PLANNING_GOAL_SCHEDULES = [0, 5]
OFFICIAL_PROPRIO_ALPHA = 0.1

OUTPUT_SKETCH_DIM = 256
TRAIN_OUTPUT_SKETCH_SEED = 31513
PRIMARY_RANK = 128
CAUSAL_RANDOM_DRAWS = 2
INTERVENTION_FORWARDS_PER_RECORD = 9
RIDGE_MULTIPLIERS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
CHANNEL_SHRINKAGE = 0.05
CHANNEL_EIGEN_FLOOR = 1e-4
CONSTRUCTION_SHUFFLE_DRAWS = 8
MIN_CONSTRUCTION_CKA = 0.15
MIN_CONSTRUCTION_CKA_ADVANTAGE = 0.03
REQUIRED_POSITIVE_CONSTRUCTION_STATES = 24

BOOTSTRAP_DRAWS = 10000
CROSSFIT_FOLDS = 5
MAX_ZERO_EDIT_ERROR = 1e-6
MIN_PLANNING_TRUE_COST_SPREAD = 1e-5
MAX_FREE_TRUE_COST_SPREAD = 1e-6
MIN_SELF_GROUND_COEFFICIENT_GAP = 0.10
MIN_SELF_CLOSURE_COEFFICIENT = 0.15
MIN_WITHIN_MODEL_RELATIVE_MSE_IMPROVEMENT = 0.01
MIN_PAIRED_RELATIVE_MSE_IMPROVEMENT = 0.05
MIN_ELIGIBLE_CONTACT_STATES = 50

if RUN_MODE == "smoke":
    ACTIVE_CONSTRUCTION_POOL_TRAJECTORIES = CONSTRUCTION_POOL_TRAJECTORIES[:30]
    ACTIVE_EVALUATION_POOL_TRAJECTORIES = EVALUATION_POOL_TRAJECTORIES[:30]
    ACTIVE_CONSTRUCTION_TARGET_PER_STRATUM = 1
    ACTIVE_EVALUATION_TARGET_PER_STRATUM = 1
    ACTIVE_CAUSAL_RANDOM_DRAWS = 1
    ACTIVE_BOOTSTRAP_DRAWS = 64
    ACTIVE_CROSSFIT_FOLDS = 2
elif RUN_MODE == "pilot":
    ACTIVE_CONSTRUCTION_POOL_TRAJECTORIES = CONSTRUCTION_POOL_TRAJECTORIES
    ACTIVE_EVALUATION_POOL_TRAJECTORIES = EVALUATION_POOL_TRAJECTORIES
    ACTIVE_CONSTRUCTION_TARGET_PER_STRATUM = CONSTRUCTION_TARGET_PER_CONTACT_STRATUM
    ACTIVE_EVALUATION_TARGET_PER_STRATUM = EVALUATION_TARGET_PER_STRATUM
    ACTIVE_CAUSAL_RANDOM_DRAWS = CAUSAL_RANDOM_DRAWS
    ACTIVE_BOOTSTRAP_DRAWS = BOOTSTRAP_DRAWS
    ACTIVE_CROSSFIT_FOLDS = CROSSFIT_FOLDS
else:
    raise ValueError("RUN_MODE must be 'smoke' or 'pilot'")

ACTIVE_CONSTRUCTION_TARGET = 2 * ACTIVE_CONSTRUCTION_TARGET_PER_STRATUM
ACTIVE_EVALUATION_TARGET = 3 * ACTIVE_EVALUATION_TARGET_PER_STRATUM
ACTIVE_INTERVENTION_FORWARDS_PER_RECORD = 5 + 2 * ACTIVE_CAUSAL_RANDOM_DRAWS

REPO_URL = "https://github.com/facebookresearch/jepa-wms.git"
REPO_COMMIT = "13cf1d9c7e476f53c17714d2e0f1dc239a883ce0"
EXPECTED_HF_REVISION = "9b9c41ef249466630dbf1a20e78391865d07b3b9"
EXPECTED_PRETRAINED_ASSET_SHA256 = {
    "jepa_wm_pusht.pth.tar": "9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb",
    "dino_wm_pusht.pth.tar": "8ec9cb05f22812d7f12e3c216b0637f41641055c0653e503e2746edb981b550f",
    "dinov2_vits14_pretrain.pth": "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9",
}
ASSET_REPOSITORY = "grewalsk/counterfactual-faithfulness-research"
ASSET_COMMIT = "2326e74556f6f81db2560e4396f4cc52c16a28f4"
ASSET_SPECS = {}
PINNED = [
    "public_jepa_wm_and_dino_wm_weights", "fresh_paired_physical_states",
    "separate_construction_and_evaluation_splits", "construction_only_layer_screen",
    "model_specific_output_aligned_rank128_subspaces", "frozen_signed_area_bank",
    "interior_schedule_closure", "heldout_extreme_schedule_goals",
    "official_visual_plus_proprio_l2_objective", "exact_simulator_regret",
    "state_grouped_cross_fitting", "paired_model_difference_estimand",
    "matched_ablation_controls", "no_required_colab_secrets",
]

assert ACTION_STEPS == 15
assert CONSTRUCTION_ACTIONS_PER_STATE == 13
assert EVALUATION_ACTIONS_PER_STATE == 24
assert len(SCHEDULE_STRINGS) == SCHEDULE_COUNT == 6
assert DIAGNOSTIC_SCHEDULES == [1, 2, 3, 4]
assert PLANNING_GOAL_SCHEDULES == [0, 5]
assert not set(DIAGNOSTIC_SCHEDULES) & set(PLANNING_GOAL_SCHEDULES)
assert OUTPUT_SKETCH_DIM >= PRIMARY_RANK
assert INTERVENTION_FORWARDS_PER_RECORD == 5 + 2 * CAUSAL_RANDOM_DRAWS
'''
configuration += "\n\nPROTOCOL_CONFIG_KEYS = " + repr(
    assigned_uppercase_names(configuration)
) + "\n"


installation = STAGE30.installation


setup = STAGE30.setup
setup = setup.replace("Stage 30", "Stage 31").replace("STAGE30", "STAGE31")
setup = setup.replace("stage30_grounded_planning", "stage31_cross_model")
setup = setup.replace("stage30-source-binder", "stage31-source-binder")
setup = setup.replace(
    "stage30_grounded_planning_value_result_bundle_",
    "stage31_cross_model_certificate_result_bundle_",
)


analysis_helpers = STAGE30.analysis_helpers + "\n\n\n" + function_sources(
    NUMERICAL.read_text(),
    [
        "planner_metric_features",
        "official_native_terminal_costs",
        "official_terminal_planning_rows",
        "paired_model_difference_rows",
    ],
)
analysis_helpers += "\n\n\n" + function_sources(
    STAGE14_NUMERICAL.read_text(), ["channel_metric_from_moments"]
)
analysis_helpers += "\n\n\n" + function_sources(
    STAGE17_NUMERICAL.read_text(), ["linear_cka", "grouped_kernel_ridge_cv"]
)
analysis_helpers += "\n\n\n" + function_sources(
    STAGE18_NUMERICAL.read_text(),
    ["nested_orthonormalize_basis", "lower_triangle_principal_overlap"],
)


model_helpers = function_sources(
    STAGE30.model_helpers,
    [
        "to_model_observation",
        "configure_repo",
        "make_environment",
        "verify_pretrained_assets",
    ],
)
model_helpers = model_helpers.replace("stage30-jepa-wms", "stage31-jepa-wms")
model_helpers = model_helpers.replace("Stage 30 supports PushT only", "Stage 31 supports PushT only")
model_helpers += r'''


def cached_verified_asset(name):
    expected = EXPECTED_PRETRAINED_ASSET_SHA256[name]
    matching = [
        path for path in CACHE_ROOT.rglob(name)
        if path.is_file() and sha256_file(path) == expected
    ]
    if not matching:
        raise RuntimeError(f"verified pretrained asset not found after load: {name}")
    return {"name": name, "path": str(matching[0]), "sha256": expected}


def validate_world_model(model, model_name):
    core = model.model
    predictor = core.predictor
    observed_type = str(getattr(core, "pred_type", ""))
    expected_type = EXPECTED_MODEL_TYPES[model_name]
    if observed_type != expected_type:
        raise RuntimeError(
            f"{model_name} predictor type changed: {observed_type!r}"
        )
    if int(getattr(model, "ctxt_window", -1)) != 2:
        raise RuntimeError(f"{model_name} context window changed")
    if expected_type == "AdaLN":
        blocks = list(getattr(predictor, "predictor_blocks", []))
        hook_kinds = ["direct"] * len(blocks)
        width = int(getattr(predictor, "predictor_total_embed_dim", -1))
    elif expected_type == "dino_wm":
        layers = list(getattr(getattr(predictor, "transformer", None), "layers", []))
        if not all(len(layer) == 2 for layer in layers):
            raise RuntimeError("DINO-WM transformer layer structure changed")
        # Hook the feed-forward residual branch.  Adding delta to its output
        # adds exactly delta to the post-block residual stream.
        blocks = [layer[1] for layer in layers]
        hook_kinds = ["dino_ff_residual"] * len(blocks)
        width = int(predictor.pos_embedding.shape[-1])
    else:
        raise RuntimeError(f"unsupported predictor type {expected_type}")
    if len(blocks) != 6:
        raise RuntimeError(f"{model_name} expected six blocks, found {len(blocks)}")
    if width != EXPECTED_CARRIER_WIDTHS[model_name]:
        raise RuntimeError(
            f"{model_name} carrier width changed: {width}"
        )
    if not all(
        isinstance(module, torch.nn.Module)
        and callable(getattr(module, "register_forward_hook", None))
        for module in blocks
    ):
        raise RuntimeError(f"{model_name} hook modules changed")
    return {
        "name": model_name,
        "short": MODEL_SHORT_NAMES[model_name],
        "model": model,
        "preprocessor": None,
        "predictor": predictor,
        "blocks": blocks,
        "hook_kinds": hook_kinds,
        "carrier_width": width,
        "pred_type": observed_type,
    }


def load_world_model(model_name):
    model, preprocessor = torch.hub.load(
        str(REPO),
        model_name,
        source="local",
        pretrained=True,
        device="cuda:0",
        trust_repo=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    bundle = validate_world_model(model, model_name)
    bundle["preprocessor"] = preprocessor
    assets = [cached_verified_asset(f"{model_name}.pth.tar")]
    assets.append(cached_verified_asset("dinov2_vits14_pretrain.pth"))
    write_json(OUT / f"pretrained_asset_verification_{bundle['short']}.json", assets)
    return bundle


def unload_world_model(bundle):
    if bundle is not None:
        bundle["model"].cpu()
    gc.collect()
    torch.cuda.empty_cache()


def model_action_tensor(preprocessor, selected_actions, horizon):
    selected_actions = np.asarray(selected_actions, dtype=np.float32)
    action_count = len(selected_actions)
    chunks = torch.from_numpy(
        selected_actions[:, : horizon * FRAMESKIP].reshape(
            action_count, horizon, FRAMESKIP, 2
        )
    ).float()
    normalized = preprocessor.normalize_actions(chunks)
    return (
        normalized.reshape(action_count, horizon, -1)
        .permute(1, 0, 2)
        .contiguous()
        .cuda()
    )


def layer_tokens_full(capture, action_count, carrier_width):
    if capture.ndim != 3 or capture.shape[1] % 256:
        raise ValueError(f"unexpected block output {tuple(capture.shape)}")
    if capture.shape[0] != int(action_count) or capture.shape[-1] != int(carrier_width):
        raise ValueError(
            f"unexpected carrier shape {tuple(capture.shape)} for "
            f"actions={action_count}, width={carrier_width}"
        )
    return capture.view(
        capture.shape[0], capture.shape[1] // 256, 256, capture.shape[-1]
    )[:, -1]


def forward_with_carriers(
    bundle,
    initial,
    actions,
    horizon,
    capture_blocks=(),
    intervention=None,
):
    captures = {int(block): [] for block in capture_blocks}
    context = {"step": -1}
    handles = []
    for block_index in capture_blocks:
        module = bundle["blocks"][int(block_index)]
        hook_kind = bundle["hook_kinds"][int(block_index)]

        def hook(_module, inputs, output, block_index=int(block_index), hook_kind=hook_kind):
            if hook_kind == "direct":
                post_block = output
            elif hook_kind == "dino_ff_residual":
                post_block = inputs[0] + output
            else:
                raise RuntimeError(f"unknown hook kind {hook_kind}")
            captures[block_index].append(post_block)
            if (
                intervention is None
                or block_index != int(intervention["block"])
                or context["step"] != horizon - 1
            ):
                return output
            delta = intervention["delta"].to(output.device, output.dtype)
            view = post_block.view(
                post_block.shape[0], post_block.shape[1] // 256,
                256, post_block.shape[-1],
            )
            if tuple(delta.shape) != tuple(view[:, -1].shape):
                raise RuntimeError(
                    f"intervention shape {tuple(delta.shape)} does not match "
                    f"carrier {tuple(view[:, -1].shape)}"
                )
            if hook_kind == "direct":
                changed = post_block.clone()
                changed_view = changed.view_as(view)
                changed_view[:, -1] = changed_view[:, -1] + delta
                return changed_view.reshape_as(output)
            # The enclosing DINO transformer adds the feed-forward output to
            # its input, so modifying the branch output modifies the block
            # residual by the same delta.
            changed = output.clone()
            changed_view = changed.view(
                changed.shape[0], changed.shape[1] // 256,
                256, changed.shape[-1],
            )
            changed_view[:, -1] = changed_view[:, -1] + delta
            return changed_view.reshape_as(output)

        handles.append(module.register_forward_hook(hook))

    model = bundle["model"]
    try:
        batch = actions.shape[1]
        action_batch = actions[:horizon].permute(1, 0, 2).contiguous()
        with torch.inference_mode():
            action_features = model.model.encode_act(action_batch)
            visual_history = initial["visual"].expand(
                batch, *initial["visual"].shape[1:]
            ).detach().clone()
            proprio_history = initial["proprio"].expand(
                batch, *initial["proprio"].shape[1:]
            ).detach().clone()
            predicted_tokens = None
            predicted_proprio = None
            for step_index in range(horizon):
                context["step"] = step_index
                predicted_visual, _, predicted_proprio = model.model.forward_pred(
                    visual_history[:, -model.ctxt_window :],
                    action_features[:, : step_index + 1][:, -model.ctxt_window :],
                    proprio_history[:, -model.ctxt_window :],
                )
                next_visual = predicted_visual[:, -1:]
                next_proprio = predicted_proprio[:, -1:]
                predicted_tokens = next_visual[:, 0, 0].flatten(1, 2)
                if predicted_tokens.shape[1:] != (256, 384):
                    raise RuntimeError(
                        f"unexpected predicted visual grid {predicted_tokens.shape[1:]}"
                    )
                visual_history = torch.cat([visual_history, next_visual], dim=1)
                proprio_history = torch.cat([proprio_history, next_proprio], dim=1)
        final_captures = {
            block: captures[block][-1] for block in capture_blocks
        }
        return predicted_tokens, predicted_proprio[:, -1], final_captures
    finally:
        for handle in handles:
            handle.remove()
'''


design = r'''# Freeze disjoint construction/evaluation tasks and both action banks before simulator or model use.


def make_specs(trajectory_ids, split):
    all_ids = CONSTRUCTION_POOL_TRAJECTORIES + EVALUATION_POOL_TRAJECTORIES
    center = np.asarray([256.0, 256.0], dtype=np.float64)
    specs = []
    for trajectory_id in trajectory_ids:
        global_index = all_ids.index(int(trajectory_id))
        phase = 0.371 + 2.0 * np.pi * global_index / len(all_ids)
        block = center + 43.0 * np.asarray([np.cos(phase), np.sin(phase)])
        block_angle = ((1.73 * phase + np.pi) % (2.0 * np.pi)) - np.pi
        offsets = [np.pi / 6, 5 * np.pi / 6, 7 * np.pi / 6, 11 * np.pi / 6]
        approach = phase + offsets[global_index % 4] + 0.11 * np.sin(5 * global_index)
        distance = float(
            DISTANCE_GRID[
                (5 * global_index + global_index // len(DISTANCE_GRID))
                % len(DISTANCE_GRID)
            ]
        )
        agent = block + distance * np.asarray([np.cos(approach), np.sin(approach)])
        goal_index = (23 * global_index + 11) % len(all_ids)
        goal_phase = 0.917 + 2.0 * np.pi * goal_index / len(all_ids)
        goal_xy = center + 72.0 * np.asarray([np.cos(goal_phase), np.sin(goal_phase)])
        prefix = 510000 if split == "construction" else 520000
        specs.append({
            "design_index": int(global_index),
            "record_id": int(prefix + trajectory_id),
            "trajectory_id": int(trajectory_id),
            "task_id": int(TASK_ID_OFFSET + global_index),
            "split": split,
            "evaluation_seed": int(DESIGN_SEED + 1013 * global_index),
            "approach_distance": distance,
            "goal": np.asarray([
                goal_xy[0], goal_xy[1],
                ((1.31 * goal_phase + np.pi) % (2.0 * np.pi)) - np.pi,
            ], dtype=np.float64),
            "state": np.asarray([
                agent[0], agent[1], block[0], block[1], block_angle,
                0.0, 0.0, 0.0, 0.0, 0.0,
            ], dtype=np.float64),
        })
    return specs


CONSTRUCTION_POOL_SPECS = make_specs(
    ACTIVE_CONSTRUCTION_POOL_TRAJECTORIES, "construction"
)
EVALUATION_POOL_SPECS = make_specs(
    ACTIVE_EVALUATION_POOL_TRAJECTORIES, "evaluation"
)


def construction_action_bank(record):
    vector = np.asarray(record["state"][2:4] - record["state"][:2], dtype=np.float64)
    vector /= max(float(np.linalg.norm(vector)), 1e-12)
    directions = [
        rotate_vector(vector, angle)
        for angle in np.linspace(-np.pi, np.pi, CONSTRUCTION_ACTIONS_PER_STATE, endpoint=False)
    ]
    return np.asarray([
        np.repeat(
            (CONSTRUCTION_ACTION_MAGNITUDE * direction)[None], ACTION_STEPS, axis=0
        )
        for direction in directions
    ], dtype=np.float32)


def evaluation_action_bank(record):
    return area_action_bank(
        np.asarray(record["state"][2:4] - record["state"][:2], dtype=np.float64),
        SELECTED_MAGNITUDES,
        steps=ACTION_STEPS,
        angle_pair_degrees=ANGLE_PAIR_DEGREES,
        schedules=SCHEDULE_STRINGS,
    ).astype(np.float32)


def screening_action_bank(record):
    # Contact strata use the same frozen 24-branch schedule geometry in both
    # splits. Construction activations themselves use the separate 13-ray bank.
    return evaluation_action_bank(record)


all_specs = CONSTRUCTION_POOL_SPECS + EVALUATION_POOL_SPECS
np.savez_compressed(
    DESIGN_DIR / "stage31_cross_model_design.npz",
    record_ids=np.asarray([row["record_id"] for row in all_specs], dtype=np.int64),
    split=np.asarray([row["split"] for row in all_specs]),
    initial_states=np.stack([row["state"] for row in all_specs]),
    goals=np.stack([row["goal"] for row in all_specs]),
    evaluation_magnitudes=np.asarray(SELECTED_MAGNITUDES, dtype=np.float64),
    schedules=np.asarray(SCHEDULE_STRINGS),
    diagnostic_schedules=np.asarray(DIAGNOSTIC_SCHEDULES, dtype=np.int64),
    planning_goal_schedules=np.asarray(PLANNING_GOAL_SCHEDULES, dtype=np.int64),
    construction_angles=np.linspace(
        -np.pi, np.pi, CONSTRUCTION_ACTIONS_PER_STATE, endpoint=False
    ),
)
write_json(DESIGN_DIR / "candidate_pool_manifest.json", {
    "construction_pool": [
        {**{key: value for key, value in row.items() if key not in {"state", "goal"}},
         "state": row["state"].tolist(), "goal": row["goal"].tolist()}
        for row in CONSTRUCTION_POOL_SPECS
    ],
    "evaluation_pool": [
        {**{key: value for key, value in row.items() if key not in {"state", "goal"}},
         "state": row["state"].tolist(), "goal": row["goal"].tolist()}
        for row in EVALUATION_POOL_SPECS
    ],
    "selection_rule": (
        "contact stratum only for evaluation; contact stratum plus frozen physical "
        "diversity floors for construction; never model output"
    ),
    "construction_target_per_contact_stratum": ACTIVE_CONSTRUCTION_TARGET_PER_STRATUM,
    "evaluation_target_per_stratum": ACTIVE_EVALUATION_TARGET_PER_STRATUM,
    "construction_evaluation_record_ids_disjoint": not bool(
        set(row["record_id"] for row in CONSTRUCTION_POOL_SPECS)
        & set(row["record_id"] for row in EVALUATION_POOL_SPECS)
    ),
    "closure_and_goal_schedule_sets_disjoint": True,
})
DESIGN_FREEZE = {
    "created_before_simulator_or_model_data": True,
    "protocol_id": PROTOCOL_ID,
    "run_signature": RUN_SIGNATURE,
    "source_identity": SOURCE_IDENTITY,
    "design_sha256": sha256_file(DESIGN_DIR / "stage31_cross_model_design.npz"),
    "pool_manifest_sha256": sha256_file(DESIGN_DIR / "candidate_pool_manifest.json"),
    "model_loaded": bool("MODEL_BUNDLE" in globals()),
    "decoder_reader_gradient_allowed": False,
    "evaluation_data_allowed_in_subspace_fit": False,
}
if DESIGN_FREEZE["model_loaded"]:
    raise RuntimeError("a model was loaded before the Stage 31 design freeze")
write_json(DESIGN_DIR / "design_freeze.json", DESIGN_FREEZE)
'''


physical_truth = r'''# Screen fresh physics, select disjoint strata, and materialize immutable truth before model loading.
PROVENANCE_COUNTS = {
    "screened_construction_states": 0,
    "screened_evaluation_states": 0,
    "construction_truth_generated": 0,
    "evaluation_truth_generated": 0,
    "jepa_baseline_generated": 0,
    "jepa_intervention_generated": 0,
    "dino_baseline_generated": 0,
    "dino_intervention_generated": 0,
    "cache_hits": 0,
}


def record_task(record):
    return {"goal": np.asarray(record["goal"], dtype=np.float64).tolist()}


def dynamic_state_from_environment(environment):
    return np.asarray([
        *environment.agent.position, *environment.block.position,
        float(environment.block.angle), *environment.agent.velocity,
        *environment.block.velocity, float(environment.block.angular_velocity),
    ], dtype=np.float64)


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
            [*environment.agent.position, *environment.agent.velocity], dtype=np.float32
        ),
    }
    return environment, observation


def rollout_dynamic_branch(record, actions, retain_visual):
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
                    "visual": np.asarray(observation["visual"]).copy() if retain_visual else None,
                    "proprio": np.asarray(observation["proprio"]).copy(),
                }
                endpoint_state = dynamic_state_from_environment(environment)
    finally:
        environment.close()
    if endpoint_observation is None or endpoint_state is None:
        raise RuntimeError("dynamic rollout missed the primary horizon")
    return initial, endpoint_observation, endpoint_state, cumulative


def physical_diversity(endpoint_states, goal):
    costs = decoded_task_cost(pose_target(endpoint_states), goal)
    left, right = np.triu_indices(len(costs), k=1)
    non_tied = np.abs(costs[left] - costs[right]) > PHYSICAL_COST_TIE
    return {
        "cost_spread": float(np.max(costs) - np.min(costs)),
        "non_tied_pair_fraction": float(np.mean(non_tied)),
    }


def screen_pool(records, split):
    rows = []
    started = time.perf_counter()
    for index, record in enumerate(records):
        screen_contacts = []
        for branch in screening_action_bank(record):
            _, _, _, count = rollout_dynamic_branch(record, branch, retain_visual=False)
            screen_contacts.append(count)
        screen_contacts = np.asarray(screen_contacts, dtype=np.int64).reshape(
            MAGNITUDE_COUNT, SCHEDULE_COUNT
        )
        regime = contact_regime(screen_contacts)
        diversity = {"cost_spread": 0.0, "non_tied_pair_fraction": 0.0}
        construction_contacts = []
        if split == "construction" and regime in CONTACT_STRATA:
            endpoints = []
            for branch in construction_action_bank(record):
                _, _, endpoint_state, count = rollout_dynamic_branch(
                    record, branch, retain_visual=False
                )
                endpoints.append(endpoint_state)
                construction_contacts.append(count)
            diversity = physical_diversity(np.asarray(endpoints), record["goal"])
        rows.append({
            "record_id": int(record["record_id"]),
            "trajectory_id": int(record["trajectory_id"]),
            "split": split,
            "approach_distance": float(record["approach_distance"]),
            "regime": regime,
            "contact_fraction": float(np.mean(screen_contacts > 0)),
            "total_contacts": int(np.sum(screen_contacts)),
            "construction_contact_branches": int(
                np.sum(np.asarray(construction_contacts) > 0)
            ),
            **diversity,
        })
        PROVENANCE_COUNTS[f"screened_{split}_states"] += 1
        write_json(OUT / f"physical_screen_{split}_progress.json", {
            "completed": index + 1, "total": len(records),
            "last_record_id": int(record["record_id"]),
        })
    TIMINGS[f"physical_screen_{split}_seconds"] = time.perf_counter() - started
    write_csv(EVIDENCE_DIR / f"physical_screen_{split}_rows.csv", rows)
    return rows


def select_records(records, screen_rows, split):
    lookup = {int(row["record_id"]): row for row in screen_rows}
    labels = CONTACT_STRATA if split == "construction" else STRATUM_LABELS
    target = (
        ACTIVE_CONSTRUCTION_TARGET_PER_STRATUM
        if split == "construction" else ACTIVE_EVALUATION_TARGET_PER_STRATUM
    )
    selected = []
    for label in labels:
        candidates = [row for row in records if lookup[int(row["record_id"])]["regime"] == label]
        if split == "construction":
            candidates = [
                row for row in candidates
                if lookup[int(row["record_id"])]["cost_spread"] >= MIN_CONSTRUCTION_COST_SPREAD
                and lookup[int(row["record_id"])]["non_tied_pair_fraction"]
                >= MIN_CONSTRUCTION_NON_TIED_PAIR_FRACTION
                and lookup[int(row["record_id"])]["construction_contact_branches"]
                >= MIN_CONSTRUCTION_CONTACT_BRANCHES
            ]
        if len(candidates) < target:
            raise RuntimeError(
                f"fresh {split} pool has {len(candidates)} eligible {label} states; requires {target}"
            )
        selected.extend(candidates[:target])
    for record in selected:
        record["regime"] = lookup[int(record["record_id"])]["regime"]
    return selected


def truth_path(record_or_id):
    record_id = int(
        record_or_id["record_id"] if isinstance(record_or_id, dict) else record_or_id
    )
    return TRUTH_DIR / f"state_{record_id:06d}.npz"


def generate_selected_truth(records, split):
    started = time.perf_counter()
    for index, record in enumerate(records):
        destination = truth_path(record)
        if destination.exists():
            PROVENANCE_COUNTS["cache_hits"] += 1
            raise RuntimeError(f"fresh-run truth shard already exists: {destination}")
        action_bank = (
            construction_action_bank(record)
            if split == "construction" else evaluation_action_bank(record)
        )
        initials, initial_proprios = [], []
        endpoint_visuals, endpoint_proprios, endpoint_states, contacts = [], [], [], []
        for branch in action_bank:
            initial, endpoint, state, count = rollout_dynamic_branch(
                record, branch, retain_visual=True
            )
            initials.append(initial["visual"])
            initial_proprios.append(initial["proprio"])
            endpoint_visuals.append(endpoint["visual"])
            endpoint_proprios.append(endpoint["proprio"])
            endpoint_states.append(state)
            contacts.append(count)
        if not all(np.array_equal(initials[0], value) for value in initials[1:]):
            raise RuntimeError("initial visual drift across exact branches")
        if not all(np.array_equal(initial_proprios[0], value) for value in initial_proprios[1:]):
            raise RuntimeError("initial proprio drift across exact branches")
        atomic_npz(
            destination,
            record_id=np.asarray(record["record_id"], dtype=np.int64),
            trajectory_id=np.asarray(record["trajectory_id"], dtype=np.int64),
            split=np.asarray(split), regime=np.asarray(record["regime"]),
            state=np.asarray(record["state"], dtype=np.float64),
            goal=np.asarray(record["goal"], dtype=np.float64),
            initial_visual=np.asarray(initials[0], dtype=np.uint8),
            initial_proprio=np.asarray(initial_proprios[0], dtype=np.float32),
            selected_actions=action_bank.astype(np.float32),
            endpoint_visuals=np.asarray(endpoint_visuals, dtype=np.uint8),
            endpoint_proprios=np.asarray(endpoint_proprios, dtype=np.float32),
            endpoint_states=np.asarray(endpoint_states, dtype=np.float64),
            interaction_counts=np.asarray(contacts, dtype=np.int32),
        )
        PROVENANCE_COUNTS[f"{split}_truth_generated"] += 1
        write_json(OUT / f"selected_truth_{split}_progress.json", {
            "completed": index + 1, "total": len(records),
            "last_record_id": int(record["record_id"]),
        })
    TIMINGS[f"selected_truth_{split}_seconds"] = time.perf_counter() - started


if not PIPELINE_FAILED:
    try:
        verify_executed_notebook_through(
            "# Screen fresh physics, select disjoint strata, and materialize immutable truth before model loading."
        )
        REPO = configure_repo()
        CONSTRUCTION_SCREEN_ROWS = screen_pool(CONSTRUCTION_POOL_SPECS, "construction")
        EVALUATION_SCREEN_ROWS = screen_pool(EVALUATION_POOL_SPECS, "evaluation")
        CONSTRUCTION_RECORDS = select_records(
            CONSTRUCTION_POOL_SPECS, CONSTRUCTION_SCREEN_ROWS, "construction"
        )
        EVALUATION_RECORDS = select_records(
            EVALUATION_POOL_SPECS, EVALUATION_SCREEN_ROWS, "evaluation"
        )
        if len(CONSTRUCTION_RECORDS) != ACTIVE_CONSTRUCTION_TARGET:
            raise RuntimeError("construction selection returned the wrong count")
        if len(EVALUATION_RECORDS) != ACTIVE_EVALUATION_TARGET:
            raise RuntimeError("evaluation selection returned the wrong count")
        if set(row["record_id"] for row in CONSTRUCTION_RECORDS) & set(
            row["record_id"] for row in EVALUATION_RECORDS
        ):
            raise RuntimeError("construction and evaluation records overlap")
        generate_selected_truth(CONSTRUCTION_RECORDS, "construction")
        generate_selected_truth(EVALUATION_RECORDS, "evaluation")
        SELECTION_CERTIFICATE = {
            "created_before_model_loading": True,
            "construction_record_ids": [int(row["record_id"]) for row in CONSTRUCTION_RECORDS],
            "evaluation_record_ids": [int(row["record_id"]) for row in EVALUATION_RECORDS],
            "construction_evaluation_disjoint": True,
            "construction_counts": {
                label: sum(row["regime"] == label for row in CONSTRUCTION_RECORDS)
                for label in CONTACT_STRATA
            },
            "evaluation_counts": {
                label: sum(row["regime"] == label for row in EVALUATION_RECORDS)
                for label in STRATUM_LABELS
            },
            "model_outputs_used_for_selection": False,
            "evaluation_effect_magnitude_used_for_selection": False,
            "construction_physical_diversity_floor_frozen": True,
        }
        write_json(DESIGN_DIR / "physical_selection_freeze.json", SELECTION_CERTIFICATE)
        memory_report("stage31_fresh_physical_truth_selected")
    except Exception:
        record_failure("stage31_fresh_physical_screen_or_truth")
'''


construction_and_subspaces = r'''# Learn each model's carrier only on construction states, then freeze both before evaluation.


def state_model_inputs(bundle, record, horizon=PRIMARY_HORIZON):
    with np.load(truth_path(record)) as payload:
        initial_visual = payload["initial_visual"]
        initial_proprio = payload["initial_proprio"]
        selected_actions = payload["selected_actions"]
    with torch.inference_mode():
        initial = bundle["model"].encode(
            to_model_observation(initial_visual, initial_proprio)
        )
    initial = {name: value.detach() for name, value in initial.items()}
    actions = model_action_tensor(bundle["preprocessor"], selected_actions, horizon)
    return initial, actions


def encode_true_state(bundle, record):
    with np.load(truth_path(record)) as payload:
        visual = payload["endpoint_visuals"][:, None]
        states = payload["endpoint_states"].astype(np.float32)
    proprio = np.concatenate([states[:, :2], states[:, 5:7]], axis=1)[:, None]
    with torch.inference_mode():
        encoded = bundle["model"].encode(to_model_observation(visual, proprio))
    visual_tokens = encoded["visual"][:, :, 0]
    visual_tokens = visual_tokens.reshape(len(visual), 256, visual_tokens.shape[-1])
    proprio_tokens = encoded["proprio"][:, 0]
    if visual_tokens.shape != (len(visual), 256, 384):
        raise RuntimeError(f"unexpected true visual-token shape {visual_tokens.shape}")
    if proprio_tokens.shape[0] != len(visual):
        raise RuntimeError("true proprio-token action axis changed")
    return visual_tokens.detach(), proprio_tokens.detach()


def count_sketch(values, dimension, seed):
    array = np.asarray(values, dtype=np.float32).reshape(len(values), -1)
    rng = np.random.default_rng(int(seed))
    buckets = rng.integers(0, int(dimension), size=array.shape[1], dtype=np.int64)
    signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=array.shape[1])
    result = np.stack([
        np.bincount(buckets, weights=row * signs, minlength=int(dimension))
        for row in array
    ]).astype(np.float64)
    return result / math.sqrt(max(array.shape[1] / int(dimension), 1.0))


def layer_screen(bundle):
    per_block = {block: [] for block in PREDICTOR_BLOCKS}
    per_block_shuffle = {block: [] for block in PREDICTOR_BLOCKS}
    rows = []
    for record in CONSTRUCTION_RECORDS:
        initial, actions = state_model_inputs(bundle, record)
        with torch.inference_mode():
            predicted, _, captures = forward_with_carriers(
                bundle, initial, actions, PRIMARY_HORIZON,
                capture_blocks=PREDICTOR_BLOCKS,
            )
        output = count_sketch(
            predicted.detach().float().cpu().numpy(), OUTPUT_SKETCH_DIM,
            stable_seed(TRAIN_OUTPUT_SKETCH_SEED, bundle["short"]),
        )
        for block in PREDICTOR_BLOCKS:
            carrier = layer_tokens_full(
                captures[block], CONSTRUCTION_ACTIONS_PER_STATE,
                bundle["carrier_width"],
            ).detach().float().cpu().numpy()
            observed = float(linear_cka(carrier, output))
            shuffled = []
            for draw in range(CONSTRUCTION_SHUFFLE_DRAWS):
                permutation = fixed_derangement(
                    CONSTRUCTION_ACTIONS_PER_STATE,
                    stable_seed(PERMUTATION_SEED, bundle["short"], record["record_id"], block, draw),
                )
                shuffled.append(float(linear_cka(carrier, output[permutation])))
            shuffle_mean = float(np.mean(shuffled))
            per_block[block].append(observed)
            per_block_shuffle[block].append(shuffle_mean)
            rows.append({
                "model": bundle["short"], "record_id": int(record["record_id"]),
                "regime": record["regime"], "block": int(block),
                "output_cka": observed, "shuffle_cka": shuffle_mean,
                "cka_advantage": observed - shuffle_mean,
            })
        del initial, actions, predicted, captures
        gc.collect()
        torch.cuda.empty_cache()
    summaries = []
    for block in PREDICTOR_BLOCKS:
        values = np.asarray(per_block[block], dtype=np.float64)
        controls = np.asarray(per_block_shuffle[block], dtype=np.float64)
        summaries.append({
            "block": int(block),
            "mean_output_cka": float(np.mean(values)),
            "standard_error": float(np.std(values, ddof=1) / math.sqrt(len(values)))
            if len(values) > 1 else 0.0,
            "mean_shuffle_cka": float(np.mean(controls)),
            "mean_cka_advantage": float(np.mean(values - controls)),
            "positive_states": int(np.sum(values > controls)),
        })
    best = max(summaries, key=lambda row: row["mean_output_cka"])
    threshold = best["mean_output_cka"] - best["standard_error"]
    selected = min(
        row["block"] for row in summaries
        if row["mean_output_cka"] >= threshold
    )
    selected_row = next(row for row in summaries if row["block"] == selected)
    required_positive = min(REQUIRED_POSITIVE_CONSTRUCTION_STATES, len(CONSTRUCTION_RECORDS))
    gate = {
        "model": bundle["short"], "selected_block": int(selected),
        "selection_rule": "earliest block within one standard error of best mean CKA",
        "best_block": int(best["block"]),
        "best_mean_output_cka": best["mean_output_cka"],
        "selected_summary": selected_row,
        "minimum_cka": MIN_CONSTRUCTION_CKA,
        "minimum_cka_advantage": MIN_CONSTRUCTION_CKA_ADVANTAGE,
        "required_positive_states": required_positive,
        "passed": bool(
            selected_row["mean_output_cka"] >= MIN_CONSTRUCTION_CKA
            and selected_row["mean_cka_advantage"] >= MIN_CONSTRUCTION_CKA_ADVANTAGE
            and selected_row["positive_states"] >= required_positive
        ),
    }
    write_csv(EVIDENCE_DIR / f"construction_layer_screen_{bundle['short']}_rows.csv", rows)
    write_json(SUBSPACE_DIR / f"construction_layer_screen_{bundle['short']}.json", {
        "gate": gate, "block_summaries": summaries,
    })
    return gate


def fitted_basis_cuda(features, targets, penalty):
    x = torch.as_tensor(features, device="cuda", dtype=torch.float32)
    y = torch.as_tensor(targets, device="cuda", dtype=torch.float32)
    identity = torch.eye(len(x), device="cuda", dtype=torch.float32)
    gram = x @ x.T
    alpha = torch.linalg.solve(gram + float(penalty) * identity, y)
    weight = x.T @ alpha
    left, singular, _ = torch.linalg.svd(weight, full_matrices=False)
    basis = left[:, :PRIMARY_RANK].detach().cpu().numpy().astype(np.float64)
    singular = singular.detach().cpu().numpy().astype(np.float64)
    del x, y, identity, gram, alpha, weight, left
    torch.cuda.empty_cache()
    return nested_orthonormalize_basis(basis), singular


def empirical_random_basis_cuda(features, excluded, seed):
    x = torch.as_tensor(features, device="cuda", dtype=torch.float32)
    excluded_tensor = torch.as_tensor(excluded, device="cuda", dtype=torch.float32)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(int(seed))
    coefficients = torch.randn(
        len(x), PRIMARY_RANK + 16, generator=generator, device="cuda"
    )
    candidate = x.T @ coefficients
    candidate = candidate - excluded_tensor @ (excluded_tensor.T @ candidate)
    q, _ = torch.linalg.qr(candidate, mode="reduced")
    basis = q[:, :PRIMARY_RANK].detach().cpu().numpy().astype(np.float64)
    del x, excluded_tensor, coefficients, candidate, q
    torch.cuda.empty_cache()
    return nested_orthonormalize_basis(basis)


def fit_model_subspace(bundle, selected_block):
    carriers, outputs, state_ids = [], [], []
    count = 0
    total = np.zeros(bundle["carrier_width"], dtype=np.float64)
    cross = np.zeros(
        (bundle["carrier_width"], bundle["carrier_width"]), dtype=np.float64
    )
    for record in CONSTRUCTION_RECORDS:
        initial, actions = state_model_inputs(bundle, record)
        with torch.inference_mode():
            predicted, _, captures = forward_with_carriers(
                bundle, initial, actions, PRIMARY_HORIZON,
                capture_blocks=[selected_block],
            )
        carrier = layer_tokens_full(
            captures[selected_block], CONSTRUCTION_ACTIONS_PER_STATE,
            bundle["carrier_width"],
        ).detach().float().cpu().numpy()
        carrier = carrier - carrier.mean(axis=0, keepdims=True)
        flattened_channels = carrier.reshape(-1, bundle["carrier_width"]).astype(np.float64)
        count += len(flattened_channels)
        total += flattened_channels.sum(axis=0)
        cross += flattened_channels.T @ flattened_channels
        carriers.append(carrier.astype(np.float32))
        outputs.append(count_sketch(
            predicted.detach().float().cpu().numpy(), OUTPUT_SKETCH_DIM,
            stable_seed(TRAIN_OUTPUT_SKETCH_SEED, bundle["short"]),
        ))
        state_ids.extend([int(record["record_id"])] * CONSTRUCTION_ACTIONS_PER_STATE)
        del initial, actions, predicted, captures
    metric = channel_metric_from_moments(
        count, total, cross, shrinkage=CHANNEL_SHRINKAGE,
        relative_floor=CHANNEL_EIGEN_FLOOR,
    )
    feature_rows, target_rows = [], []
    for carrier, output in zip(carriers, outputs):
        white = transform_primal_channels(carrier, metric["inverse_square_root"])
        feature_rows.append(candidate_center(white).reshape(CONSTRUCTION_ACTIONS_PER_STATE, -1))
        target_rows.append(candidate_center(output))
    features = np.concatenate(feature_rows).astype(np.float32)
    features /= math.sqrt(features.shape[1])
    targets = np.concatenate(target_rows).astype(np.float64)
    target_scale = np.std(targets, axis=0, ddof=1)
    target_scale = np.maximum(target_scale, 1e-8)
    targets = targets / target_scale
    state_ids = np.asarray(state_ids, dtype=np.int64)
    unique_states = sorted(set(state_ids.tolist()))
    state_fold = {
        record_id: index % min(5, len(unique_states))
        for index, record_id in enumerate(unique_states)
    }
    cv_groups = np.asarray([state_fold[int(value)] for value in state_ids], dtype=np.int64)
    kernel = np.asarray(features, dtype=np.float64) @ np.asarray(features, dtype=np.float64).T
    cv = grouped_kernel_ridge_cv(kernel, targets, cv_groups, RIDGE_MULTIPLIERS)
    primary, singular = fitted_basis_cuda(features, targets, cv["penalty"])
    shuffled_targets = targets.copy()
    for record_id in unique_states:
        indices = np.flatnonzero(state_ids == record_id)
        permutation = fixed_derangement(
            len(indices), stable_seed(PERMUTATION_SEED, bundle["short"], record_id, "fit")
        )
        shuffled_targets[indices] = targets[indices[permutation]]
    shuffled, shuffled_singular = fitted_basis_cuda(
        features, shuffled_targets, cv["penalty"]
    )
    random_bases = [
        empirical_random_basis_cuda(
            features, primary,
            stable_seed(NULL_ROOT_SEED, bundle["short"], draw),
        )
        for draw in range(ACTIVE_CAUSAL_RANDOM_DRAWS)
    ]
    payload = {
        "primary_basis": primary.astype(np.float32),
        "shuffled_basis": shuffled.astype(np.float32),
        "channel_square_root": metric["square_root"].astype(np.float64),
        "channel_inverse_square_root": metric["inverse_square_root"].astype(np.float64),
        "singular_values": singular.astype(np.float64),
        "shuffled_singular_values": shuffled_singular.astype(np.float64),
        "selected_block": np.asarray(selected_block, dtype=np.int64),
    }
    for draw, basis in enumerate(random_bases):
        payload[f"random_basis_{draw:02d}"] = basis.astype(np.float32)
    path = SUBSPACE_DIR / f"frozen_{bundle['short']}_rank128_subspaces.npz"
    atomic_npz(path, **payload)
    orthogonality = {
        name: float(np.max(np.abs(
            value.astype(np.float64).T @ value.astype(np.float64)
            - np.eye(value.shape[1])
        )))
        for name, value in payload.items()
        if name.endswith("basis") or name.startswith("random_basis_")
    }
    if max(orthogonality.values()) > 2e-5:
        raise RuntimeError(f"stored basis lost orthogonality: {orthogonality}")
    manifest = {
        "model": bundle["short"], "model_name": bundle["name"],
        "pred_type": bundle["pred_type"], "selected_block": int(selected_block),
        "carrier_width": int(bundle["carrier_width"]),
        "ambient_dimension": int(features.shape[1]), "rank": PRIMARY_RANK,
        "construction_states": len(CONSTRUCTION_RECORDS),
        "construction_rows": int(len(features)),
        "evaluation_rows_used": 0, "planning_labels_used": 0,
        "simulator_effect_magnitudes_used_for_fit": 0,
        "ridge_cv_grouped_by_state": True,
        "ridge_cv": cv,
        "channel_metric_condition_number": float(metric["condition_number"]),
        "orthogonality_max_abs_errors": orthogonality,
        "subspace_sha256": sha256_file(path),
    }
    write_json(SUBSPACE_DIR / f"subspace_manifest_{bundle['short']}.json", manifest)
    return manifest


def hook_identity_test(bundle, record, selected_block):
    initial, actions = state_model_inputs(bundle, record)
    action_count = actions.shape[1]
    with torch.inference_mode():
        baseline, _, _ = forward_with_carriers(
            bundle, initial, actions, PRIMARY_HORIZON,
            capture_blocks=[selected_block],
        )
        patched, _, _ = forward_with_carriers(
            bundle, initial, actions, PRIMARY_HORIZON,
            capture_blocks=[selected_block],
            intervention={
                "block": selected_block,
                "delta": torch.zeros(
                    action_count, 256, bundle["carrier_width"],
                    device="cuda", dtype=torch.float32,
                ),
            },
        )
    error = float(torch.max(torch.abs(patched - baseline)).cpu())
    result = {
        "model": bundle["short"], "record_id": int(record["record_id"]),
        "max_abs_error": error, "passed": error <= MAX_ZERO_EDIT_ERROR,
    }
    write_json(OUT / f"hook_identity_test_{bundle['short']}.json", result)
    if not result["passed"]:
        raise RuntimeError(f"zero hook identity failed: {result}")
    return result


CONSTRUCTION_GATES = {}
SUBSPACE_MANIFESTS = {}
FORWARD_BENCHMARKS = {}
ALL_CONSTRUCTION_GATES_PASS = False
if not PIPELINE_FAILED:
    try:
        verify_executed_notebook_through(
            "# Learn each model's carrier only on construction states, then freeze both before evaluation."
        )
        for model_name in MODEL_NAMES:
            MODEL_BUNDLE = load_world_model(model_name)
            short = MODEL_BUNDLE["short"]
            gate = layer_screen(MODEL_BUNDLE)
            CONSTRUCTION_GATES[short] = gate
            if gate["passed"]:
                SUBSPACE_MANIFESTS[short] = fit_model_subspace(
                    MODEL_BUNDLE, gate["selected_block"]
                )
                hook_identity_test(
                    MODEL_BUNDLE, CONSTRUCTION_RECORDS[0], gate["selected_block"]
                )
                initial, actions = state_model_inputs(
                    MODEL_BUNDLE, EVALUATION_RECORDS[0]
                )
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.inference_mode():
                    forward_with_carriers(
                        MODEL_BUNDLE, initial, actions, PRIMARY_HORIZON,
                        capture_blocks=[gate["selected_block"]],
                    )
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                FORWARD_BENCHMARKS[short] = {
                    "seconds_per_24_branch_predictor_batch": seconds,
                    "predictor_batches_per_record": ACTIVE_INTERVENTION_FORWARDS_PER_RECORD + 1,
                    "target_encoder_batches_per_record": 1,
                    "evaluation_records": len(EVALUATION_RECORDS),
                    "estimated_predictor_minutes": seconds * len(EVALUATION_RECORDS)
                    * (ACTIVE_INTERVENTION_FORWARDS_PER_RECORD + 1) / 60.0,
                }
                write_json(OUT / f"forward_benchmark_{short}.json", FORWARD_BENCHMARKS[short])
            unload_world_model(MODEL_BUNDLE)
            MODEL_BUNDLE = None
        ALL_CONSTRUCTION_GATES_PASS = bool(
            len(CONSTRUCTION_GATES) == 2
            and all(gate["passed"] for gate in CONSTRUCTION_GATES.values())
            and len(SUBSPACE_MANIFESTS) == 2
        )
        write_json(OUT / "construction_freeze_certificate.json", {
            "all_gates_pass": ALL_CONSTRUCTION_GATES_PASS,
            "construction_gates": CONSTRUCTION_GATES,
            "subspace_manifests": SUBSPACE_MANIFESTS,
            "both_subspaces_frozen_before_evaluation": True,
            "model_coordinate_transfer_claimed": False,
        })
        memory_report("stage31_model_specific_subspaces_frozen")
    except Exception:
        if "MODEL_BUNDLE" in globals() and MODEL_BUNDLE is not None:
            unload_world_model(MODEL_BUNDLE)
        record_failure("stage31_construction_or_subspace_fit")
'''


evaluation = r'''# Evaluate both frozen carriers on the same new states under the exact official objective.


def load_subspace_cuda(short):
    path = SUBSPACE_DIR / f"frozen_{short}_rank128_subspaces.npz"
    with np.load(path) as payload:
        arrays = {name: payload[name].copy() for name in payload.files}
    cuda = {
        name: torch.as_tensor(value, device="cuda", dtype=torch.float32)
        for name, value in arrays.items()
        if name.endswith("basis") or name.startswith("random_basis_")
        or name in {"channel_square_root", "channel_inverse_square_root"}
    }
    return arrays, cuda


def matched_tensor_norm(value, reference):
    norm = torch.linalg.vector_norm(value)
    target = torch.linalg.vector_norm(reference)
    if float(norm) <= 1e-12 or float(target) <= 1e-12:
        raise RuntimeError("cannot norm-match a degenerate intervention")
    return value * (target / norm)


def intervention_specs_cuda(carrier, subspace_cuda):
    width = carrier.shape[-1]
    white = carrier.float() @ subspace_cuda["channel_inverse_square_root"].T
    flat = white.reshape(EVALUATION_ACTIONS_PER_STATE, -1)
    grouped = flat.reshape(MAGNITUDE_COUNT, SCHEDULE_COUNT, -1)
    reversed_flat = torch.flip(grouped, dims=[1]).reshape_as(flat)
    swap_target = reversed_flat - flat
    antisymmetric = 0.5 * (flat - reversed_flat)

    def projected(values, basis):
        return (values @ basis) @ basis.T

    primary_basis = subspace_cuda["primary_basis"][:, :PRIMARY_RANK]
    shuffled_basis = subspace_cuda["shuffled_basis"][:, :PRIMARY_RANK]
    primary_swap = projected(swap_target, primary_basis)
    primary_ablation = -projected(antisymmetric, primary_basis)
    specs = [
        {"condition": "primary_r128_swap", "family": "primary", "mode": "swap", "delta_white": primary_swap},
        {"condition": "shuffled_r128_swap", "family": "matched_shuffled_control", "mode": "swap", "delta_white": matched_tensor_norm(projected(swap_target, shuffled_basis), primary_swap)},
    ]
    for draw in range(ACTIVE_CAUSAL_RANDOM_DRAWS):
        basis = subspace_cuda[f"random_basis_{draw:02d}"][:, :PRIMARY_RANK]
        specs.append({
            "condition": f"random_r128_{draw:02d}_swap",
            "family": "empirical_span_random_control", "mode": "swap",
            "delta_white": matched_tensor_norm(projected(swap_target, basis), primary_swap),
        })
    specs.append({
        "condition": "full_activation_swap", "family": "positive_control_only",
        "mode": "swap", "delta_white": swap_target,
    })
    specs.extend([
        {"condition": "primary_r128_ablation", "family": "primary", "mode": "ablation", "delta_white": primary_ablation},
        {"condition": "shuffled_r128_ablation", "family": "matched_shuffled_control", "mode": "ablation", "delta_white": matched_tensor_norm(-projected(antisymmetric, shuffled_basis), primary_ablation)},
    ])
    for draw in range(ACTIVE_CAUSAL_RANDOM_DRAWS):
        basis = subspace_cuda[f"random_basis_{draw:02d}"][:, :PRIMARY_RANK]
        specs.append({
            "condition": f"random_r128_{draw:02d}_ablation",
            "family": "empirical_span_random_control", "mode": "ablation",
            "delta_white": matched_tensor_norm(-projected(antisymmetric, basis), primary_ablation),
        })
    if len(specs) != ACTIVE_INTERVENTION_FORWARDS_PER_RECORD:
        raise RuntimeError(f"unexpected intervention count {len(specs)}")
    root = subspace_cuda["channel_square_root"]
    for spec in specs:
        spec["edit_norm"] = float(torch.linalg.vector_norm(spec["delta_white"]).cpu())
        spec["primary_swap_norm"] = float(torch.linalg.vector_norm(primary_swap).cpu())
        spec["primary_ablation_norm"] = float(torch.linalg.vector_norm(primary_ablation).cpu())
        spec["delta_native"] = (
            spec.pop("delta_white").reshape(EVALUATION_ACTIONS_PER_STATE, 256, width)
            @ root.T
        )
    return specs


def baseline_alignment_rows(record, short, predicted_visual, target_visual, predicted_proprio, target_proprio):
    prediction_joint = planner_metric_features(
        predicted_visual, predicted_proprio, OFFICIAL_PROPRIO_ALPHA
    ).reshape(MAGNITUDE_COUNT, SCHEDULE_COUNT, -1)
    target_joint = planner_metric_features(
        target_visual, target_proprio, OFFICIAL_PROPRIO_ALPHA
    ).reshape(MAGNITUDE_COUNT, SCHEDULE_COUNT, -1)
    prediction_visual = np.asarray(predicted_visual).reshape(
        MAGNITUDE_COUNT, SCHEDULE_COUNT, -1
    )
    truth_visual = np.asarray(target_visual).reshape(
        MAGNITUDE_COUNT, SCHEDULE_COUNT, -1
    )
    rows = []
    for magnitude_index in range(MAGNITUDE_COUNT):
        joint = vector_alignment(
            prediction_joint[magnitude_index], target_joint[magnitude_index]
        )
        visual = vector_alignment(
            prediction_visual[magnitude_index], truth_visual[magnitude_index]
        )
        rows.append({
            "model": short, "record_id": int(record["record_id"]),
            "trajectory_id": int(record["trajectory_id"]), "regime": record["regime"],
            "magnitude_index": int(magnitude_index),
            "magnitude": float(SELECTED_MAGNITUDES[magnitude_index]),
            **{f"native_joint_{key}": value for key, value in joint.items()},
            **{f"native_visual_{key}": value for key, value in visual.items()},
        })
    return rows


def attach_identity(rows, record, short, condition, family, objective):
    return [{
        "model": short, "record_id": int(record["record_id"]),
        "trajectory_id": int(record["trajectory_id"]), "regime": record["regime"],
        "condition": condition, "family": family, "objective": objective,
        "magnitude": float(SELECTED_MAGNITUDES[row["magnitude_index"]]),
        **row,
    } for row in rows]


def run_evaluation_record(bundle, record, subspace_cuda):
    short = bundle["short"]
    selected_block = int(CONSTRUCTION_GATES[short]["selected_block"])
    initial, actions = state_model_inputs(bundle, record)
    target_visual_tensor, target_proprio_tensor = encode_true_state(bundle, record)
    with np.load(truth_path(record)) as payload:
        endpoint_states = payload["endpoint_states"].astype(np.float64)
    with torch.inference_mode():
        predicted_visual_tensor, predicted_proprio_tensor, captures = forward_with_carriers(
            bundle, initial, actions, PRIMARY_HORIZON,
            capture_blocks=[selected_block],
        )
    predicted_visual = predicted_visual_tensor.detach().float().cpu().numpy()
    predicted_proprio = predicted_proprio_tensor.detach().float().cpu().numpy()
    target_visual = target_visual_tensor.detach().float().cpu().numpy()
    target_proprio = target_proprio_tensor.detach().float().cpu().numpy()
    carrier = layer_tokens_full(
        captures[selected_block], EVALUATION_ACTIONS_PER_STATE,
        bundle["carrier_width"],
    ).detach()
    alignment = baseline_alignment_rows(
        record, short, predicted_visual, target_visual,
        predicted_proprio, target_proprio,
    )
    planning = attach_identity(
        official_terminal_planning_rows(
            predicted_visual, target_visual, predicted_proprio, target_proprio,
            endpoint_states, MAGNITUDE_COUNT, SCHEDULE_COUNT,
            goal_schedules=PLANNING_GOAL_SCHEDULES,
            alpha=OFFICIAL_PROPRIO_ALPHA,
        ), record, short, "baseline", "native_unedited", "official_visual_plus_proprio"
    )
    visual_planning = attach_identity(
        terminal_planning_rows(
            predicted_visual, target_visual, endpoint_states,
            MAGNITUDE_COUNT, SCHEDULE_COUNT,
            goal_schedules=PLANNING_GOAL_SCHEDULES,
        ), record, short, "baseline", "native_unedited", "visual_only_sensitivity"
    )
    closure, visual_closure = [], []
    for spec in intervention_specs_cuda(carrier, subspace_cuda):
        with torch.inference_mode():
            patched_visual_tensor, patched_proprio_tensor, _ = forward_with_carriers(
                bundle, initial, actions, PRIMARY_HORIZON,
                capture_blocks=[selected_block],
                intervention={"block": selected_block, "delta": spec["delta_native"]},
            )
        patched_visual = patched_visual_tensor.detach().float().cpu().numpy()
        patched_proprio = patched_proprio_tensor.detach().float().cpu().numpy()
        baseline_joint = planner_metric_features(
            predicted_visual, predicted_proprio, OFFICIAL_PROPRIO_ALPHA
        )
        patched_joint = planner_metric_features(
            patched_visual, patched_proprio, OFFICIAL_PROPRIO_ALPHA
        )
        target_joint = planner_metric_features(
            target_visual, target_proprio, OFFICIAL_PROPRIO_ALPHA
        )
        common = {
            "model": short, "record_id": int(record["record_id"]),
            "trajectory_id": int(record["trajectory_id"]), "regime": record["regime"],
            "condition": spec["condition"], "family": spec["family"],
            "rank": -1 if spec["condition"] == "full_activation_swap" else PRIMARY_RANK,
            "carrier_edit_whitened_norm": spec["edit_norm"],
            "primary_swap_whitened_norm": spec["primary_swap_norm"],
            "primary_ablation_whitened_norm": spec["primary_ablation_norm"],
        }
        for row in diagnostic_closure_rows(
            baseline_joint, patched_joint, target_joint,
            MAGNITUDE_COUNT, SCHEDULE_COUNT,
            diagnostic_schedules=DIAGNOSTIC_SCHEDULES, mode=spec["mode"],
        ):
            closure.append({
                **common, "magnitude": float(SELECTED_MAGNITUDES[row["magnitude_index"]]),
                **row,
            })
        for row in diagnostic_closure_rows(
            predicted_visual, patched_visual, target_visual,
            MAGNITUDE_COUNT, SCHEDULE_COUNT,
            diagnostic_schedules=DIAGNOSTIC_SCHEDULES, mode=spec["mode"],
        ):
            visual_closure.append({
                **common, "magnitude": float(SELECTED_MAGNITUDES[row["magnitude_index"]]),
                **row,
            })
        planning.extend(attach_identity(
            official_terminal_planning_rows(
                patched_visual, target_visual, patched_proprio, target_proprio,
                endpoint_states, MAGNITUDE_COUNT, SCHEDULE_COUNT,
                goal_schedules=PLANNING_GOAL_SCHEDULES,
                alpha=OFFICIAL_PROPRIO_ALPHA,
            ), record, short, spec["condition"], spec["family"],
            "official_visual_plus_proprio",
        ))
        visual_planning.extend(attach_identity(
            terminal_planning_rows(
                patched_visual, target_visual, endpoint_states,
                MAGNITUDE_COUNT, SCHEDULE_COUNT,
                goal_schedules=PLANNING_GOAL_SCHEDULES,
            ), record, short, spec["condition"], spec["family"],
            "visual_only_sensitivity",
        ))
        del patched_visual_tensor, patched_proprio_tensor, patched_visual, patched_proprio
    PROVENANCE_COUNTS[f"{short}_baseline_generated"] += 1
    PROVENANCE_COUNTS[f"{short}_intervention_generated"] += 1
    del initial, actions, target_visual_tensor, target_proprio_tensor
    del predicted_visual_tensor, predicted_proprio_tensor, captures, carrier
    gc.collect()
    torch.cuda.empty_cache()
    return alignment, closure, visual_closure, planning, visual_planning


BASELINE_ALIGNMENT_ROWS = []
CLOSURE_ROWS = []
VISUAL_CLOSURE_ROWS = []
PLANNING_ROWS = []
VISUAL_PLANNING_ROWS = []
if not PIPELINE_FAILED and ALL_CONSTRUCTION_GATES_PASS:
    try:
        verify_executed_notebook_through(
            "# Evaluate both frozen carriers on the same new states under the exact official objective."
        )
        started = time.perf_counter()
        for model_name in MODEL_NAMES:
            MODEL_BUNDLE = load_world_model(model_name)
            short = MODEL_BUNDLE["short"]
            _, SUBSPACE_CUDA = load_subspace_cuda(short)
            for index, record in enumerate(EVALUATION_RECORDS):
                alignment, closure, visual_closure, planning, visual_planning = (
                    run_evaluation_record(MODEL_BUNDLE, record, SUBSPACE_CUDA)
                )
                BASELINE_ALIGNMENT_ROWS.extend(alignment)
                CLOSURE_ROWS.extend(closure)
                VISUAL_CLOSURE_ROWS.extend(visual_closure)
                PLANNING_ROWS.extend(planning)
                VISUAL_PLANNING_ROWS.extend(visual_planning)
                write_json(OUT / f"evaluation_{short}_progress.json", {
                    "completed": index + 1, "total": len(EVALUATION_RECORDS),
                    "last_record_id": int(record["record_id"]),
                })
            del SUBSPACE_CUDA
            unload_world_model(MODEL_BUNDLE)
            MODEL_BUNDLE = None
        TIMINGS["paired_cross_model_evaluation_seconds"] = time.perf_counter() - started
        write_csv(EVIDENCE_DIR / "baseline_native_alignment_rows.csv", BASELINE_ALIGNMENT_ROWS)
        write_csv(EVIDENCE_DIR / "joint_diagnostic_closure_rows.csv", CLOSURE_ROWS)
        write_csv(EVIDENCE_DIR / "visual_diagnostic_closure_rows.csv", VISUAL_CLOSURE_ROWS)
        write_csv(EVIDENCE_DIR / "official_terminal_planning_rows.csv", PLANNING_ROWS)
        write_csv(EVIDENCE_DIR / "visual_terminal_planning_rows.csv", VISUAL_PLANNING_ROWS)
        write_json(OUT / "evaluation_open_certificate.json", {
            "opened": True, "source_identity": SOURCE_IDENTITY,
            "construction_freeze_sha256": sha256_file(
                OUT / "construction_freeze_certificate.json"
            ),
            "paired_physical_records": len(EVALUATION_RECORDS),
            "official_objective": "visual_mse_plus_0.1_proprio_mse",
            "both_subspaces_frozen_before_evaluation": True,
            "decoder_reader_gradient_used": False,
        })
        memory_report("stage31_paired_cross_model_evaluation_complete")
    except Exception:
        if "MODEL_BUNDLE" in globals() and MODEL_BUNDLE is not None:
            unload_world_model(MODEL_BUNDLE)
        record_failure("stage31_paired_cross_model_evaluation")
'''


decision = r'''# Apply the preregistered within-model and paired cross-model closure gates.


def bootstrap_state_mean(values, record_ids, label):
    draws = clustered_bootstrap_mean(
        np.asarray(values, dtype=np.float64),
        np.asarray(record_ids, dtype=np.int64),
        ACTIVE_BOOTSTRAP_DRAWS,
        stable_seed(BOOTSTRAP_SEED, label) % (2**31 - 1),
    )
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def row_map(rows, keys):
    result = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        if key in result:
            raise RuntimeError(f"duplicate row key: {key}")
        result[key] = row
    return result


def joined_model_dataset(short):
    alignment = row_map(
        [row for row in BASELINE_ALIGNMENT_ROWS if row["model"] == short],
        ["record_id", "magnitude_index"],
    )
    closure = row_map(
        [row for row in CLOSURE_ROWS
         if row["model"] == short and row["condition"] == "primary_r128_swap"],
        ["record_id", "magnitude_index"],
    )
    planning = row_map(
        [row for row in PLANNING_ROWS if row["model"] == short],
        ["record_id", "magnitude_index", "goal_schedule", "condition"],
    )
    tasks = []
    for key, base in planning.items():
        record_id, magnitude_index, goal_schedule, condition = key
        if condition != "baseline" or base["regime"] not in CONTACT_STRATA:
            continue
        if float(base["true_cost_spread"]) < MIN_PLANNING_TRUE_COST_SPREAD:
            continue
        align = alignment[(record_id, magnitude_index)]
        close = closure[(record_id, magnitude_index)]
        row = {
            "record_id": int(record_id), "magnitude_index": int(magnitude_index),
            "regime": base["regime"], "goal_schedule": int(goal_schedule),
            "outcome": float(base["normalized_regret"]),
            "native_joint_normalized_rmse": float(
                align["native_joint_normalized_rmse"]
            ),
            "self_coefficient": float(close["self_coefficient"]),
            "self_cosine": float(close["self_cosine"]),
            "grounded_coefficient": float(close["grounded_coefficient"]),
            "grounded_cosine": float(close["grounded_cosine"]),
        }
        if all(
            np.isfinite(value) for name, value in row.items()
            if name not in {"regime"}
        ):
            tasks.append(row)
    grouped = defaultdict(list)
    for row in tasks:
        grouped[(row["record_id"], row["magnitude_index"])].append(row)
    rows = []
    for key, values in sorted(grouped.items()):
        if len(values) != len(PLANNING_GOAL_SCHEDULES):
            raise RuntimeError(f"incomplete planning-goal panel for {short} {key}")
        first = values[0]
        rows.append({
            **{name: value for name, value in first.items()
               if name not in {"goal_schedule", "outcome"}},
            "goal_tasks": len(values),
            "outcome": float(np.mean([value["outcome"] for value in values])),
        })
    return rows


def predictive_result(rows, added_features, label, minimum_relative):
    groups = np.asarray([row["record_id"] for row in rows], dtype=np.int64)
    base_features = np.asarray([[
        row["magnitude_index"] / max(MAGNITUDE_COUNT - 1, 1),
        float(row["regime"] == "boundary_switching"),
        row["native_joint_normalized_rmse"],
        row["self_coefficient"], row["self_cosine"],
    ] for row in rows], dtype=np.float64)
    added = np.asarray([[
        row[name] for name in added_features
    ] for row in rows], dtype=np.float64)
    outcome = np.asarray([row["outcome"] for row in rows], dtype=np.float64)
    result = cross_fitted_incremental_value(
        outcome, groups, base_features, added,
        folds=ACTIVE_CROSSFIT_FOLDS,
        seed=stable_seed(CROSSFIT_SEED, label), ridge=1e-6,
    )
    group_values = np.asarray([
        row["mse_improvement"] for row in result["group_rows"]
    ], dtype=np.float64)
    group_ids = np.asarray([
        int(row["group"]) for row in result["group_rows"]
    ], dtype=np.int64)
    ci = bootstrap_state_mean(group_values, group_ids, f"{label}_mse")
    evidence = []
    for index, row in enumerate(rows):
        evidence.append({
            **row, "fold": int(result["fold_id"][index]),
            "base_prediction": float(result["base_prediction"][index]),
            "grounded_prediction": float(result["grounded_prediction"][index]),
            "base_squared_error": float(
                (row["outcome"] - result["base_prediction"][index]) ** 2
            ),
            "grounded_squared_error": float(
                (row["outcome"] - result["grounded_prediction"][index]) ** 2
            ),
        })
    write_csv(EVIDENCE_DIR / f"{label}_crossfit_rows.csv", evidence)
    required_states = min(
        MIN_ELIGIBLE_CONTACT_STATES,
        2 * ACTIVE_EVALUATION_TARGET_PER_STRATUM,
    )
    return {
        "label": label, "eligible_states": int(len(np.unique(groups))),
        "state_magnitude_rows": len(rows), "crossfit_folds": ACTIVE_CROSSFIT_FOLDS,
        "base_features": [
            "magnitude", "boundary_indicator", "native_joint_nrmse",
            "self_coefficient", "self_cosine",
        ],
        "added_features": list(added_features),
        "base_oof_mse": float(result["base_mse"]),
        "grounded_oof_mse": float(result["grounded_mse"]),
        "relative_oof_mse_improvement": float(result["relative_mse_improvement"]),
        "base_oof_r_squared": float(result["base_oof_r_squared"]),
        "grounded_oof_r_squared": float(result["grounded_oof_r_squared"]),
        "state_mean_mse_improvement_ci95": ci,
        "minimum_relative_improvement": float(minimum_relative),
        "required_states": int(required_states),
        "passed": bool(
            len(np.unique(groups)) >= required_states
            and result["relative_mse_improvement"] >= minimum_relative
            and (ci[0] > 0 if RUN_MODE == "pilot" else True)
        ),
    }


def paired_predictive_gate(model_rows):
    feature_names = [
        "native_joint_normalized_rmse", "self_coefficient", "self_cosine",
        "grounded_coefficient", "grounded_cosine",
    ]
    left_index = {
        (row["record_id"], row["magnitude_index"]): row
        for row in model_rows["jepa"]
    }
    right_index = {
        (row["record_id"], row["magnitude_index"]): row
        for row in model_rows["dino"]
    }
    common_keys = sorted(set(left_index) & set(right_index))
    rows = paired_model_difference_rows(
        [left_index[key] for key in common_keys],
        [right_index[key] for key in common_keys],
        feature_names,
    )
    groups = np.asarray([row["record_id"] for row in rows], dtype=np.int64)
    base = np.asarray([[
        row["magnitude_index"] / max(MAGNITUDE_COUNT - 1, 1),
        float(row["regime"] == "boundary_switching"),
        row["difference_native_joint_normalized_rmse"],
        row["difference_self_coefficient"],
        row["difference_self_cosine"],
    ] for row in rows], dtype=np.float64)
    added = np.asarray([[
        row["difference_grounded_coefficient"],
        row["difference_grounded_cosine"],
    ] for row in rows], dtype=np.float64)
    outcome = np.asarray([row["outcome"] for row in rows], dtype=np.float64)
    result = cross_fitted_incremental_value(
        outcome, groups, base, added,
        folds=ACTIVE_CROSSFIT_FOLDS,
        seed=stable_seed(CROSSFIT_SEED, "paired_dino_minus_jepa"), ridge=1e-6,
    )
    group_values = np.asarray([
        row["mse_improvement"] for row in result["group_rows"]
    ], dtype=np.float64)
    group_ids = np.asarray([
        int(row["group"]) for row in result["group_rows"]
    ], dtype=np.int64)
    ci = bootstrap_state_mean(
        group_values, group_ids, "paired_dino_minus_jepa_mse"
    )
    evidence = []
    for index, row in enumerate(rows):
        evidence.append({
            **row, "fold": int(result["fold_id"][index]),
            "base_prediction": float(result["base_prediction"][index]),
            "grounded_prediction": float(result["grounded_prediction"][index]),
        })
    write_csv(EVIDENCE_DIR / "paired_model_difference_crossfit_rows.csv", evidence)
    required_states = min(
        MIN_ELIGIBLE_CONTACT_STATES,
        2 * ACTIVE_EVALUATION_TARGET_PER_STRATUM,
    )
    return {
        "estimand": "DINO-WM normalized regret minus JEPA-WM normalized regret",
        "jepa_unpaired_rows_dropped": int(len(left_index) - len(common_keys)),
        "dino_unpaired_rows_dropped": int(len(right_index) - len(common_keys)),
        "eligible_states": int(len(np.unique(groups))),
        "paired_state_magnitude_rows": len(rows),
        "base_features": [
            "magnitude", "boundary_indicator", "delta_native_joint_nrmse",
            "delta_self_coefficient", "delta_self_cosine",
        ],
        "added_features": [
            "delta_grounded_coefficient", "delta_grounded_cosine",
        ],
        "base_oof_mse": float(result["base_mse"]),
        "grounded_oof_mse": float(result["grounded_mse"]),
        "relative_oof_mse_improvement": float(result["relative_mse_improvement"]),
        "base_oof_r_squared": float(result["base_oof_r_squared"]),
        "grounded_oof_r_squared": float(result["grounded_oof_r_squared"]),
        "state_mean_mse_improvement_ci95": ci,
        "minimum_relative_improvement": MIN_PAIRED_RELATIVE_MSE_IMPROVEMENT,
        "required_states": int(required_states),
        "passed": bool(
            len(np.unique(groups)) >= required_states
            and result["relative_mse_improvement"]
            >= MIN_PAIRED_RELATIVE_MSE_IMPROVEMENT
            and (ci[0] > 0 if RUN_MODE == "pilot" else True)
        ),
    }, rows


def replication_gap(short):
    rows = [
        row for row in CLOSURE_ROWS
        if row["model"] == short and row["condition"] == "primary_r128_swap"
        and row["regime"] == "persistent_contact"
        and np.isfinite(row["grounded_coefficient"])
    ]
    gaps = np.asarray([
        row["self_coefficient"] - row["grounded_coefficient"] for row in rows
    ], dtype=np.float64)
    ids = np.asarray([row["record_id"] for row in rows], dtype=np.int64)
    ci = bootstrap_state_mean(gaps, ids, f"{short}_self_ground_gap")
    self_mean = float(np.mean([row["self_coefficient"] for row in rows]))
    grounded_mean = float(np.mean([row["grounded_coefficient"] for row in rows]))
    return {
        "model": short, "persistent_states": int(len(np.unique(ids))),
        "mean_self_coefficient": self_mean,
        "mean_grounded_coefficient": grounded_mean,
        "mean_self_minus_grounded": float(np.mean(gaps)),
        "self_minus_grounded_ci95": ci,
        "passed": bool(
            self_mean >= MIN_SELF_CLOSURE_COEFFICIENT
            and np.mean(gaps) >= MIN_SELF_GROUND_COEFFICIENT_GAP
            and (ci[0] > 0 if RUN_MODE == "pilot" else True)
        ),
    }


def ablation_control_summary(short):
    planning = row_map(
        [row for row in PLANNING_ROWS if row["model"] == short],
        ["record_id", "magnitude_index", "goal_schedule", "condition"],
    )
    by_state = defaultdict(list)
    for key, baseline in planning.items():
        record_id, magnitude_index, goal_schedule, condition = key
        if condition != "baseline" or baseline["regime"] not in CONTACT_STRATA:
            continue
        if baseline["true_cost_spread"] < MIN_PLANNING_TRUE_COST_SPREAD:
            continue
        primary = planning[(record_id, magnitude_index, goal_schedule, "primary_r128_ablation")]
        controls = [
            candidate[3] for candidate in planning
            if candidate[:3] == (record_id, magnitude_index, goal_schedule)
            and (candidate[3] == "shuffled_r128_ablation"
                 or candidate[3].startswith("random_r128_")
                 and candidate[3].endswith("_ablation"))
        ]
        primary_change = primary["normalized_regret"] - baseline["normalized_regret"]
        control_changes = [
            planning[(record_id, magnitude_index, goal_schedule, name)]["normalized_regret"]
            - baseline["normalized_regret"] for name in controls
        ]
        by_state[int(record_id)].append(
            float(primary_change - np.median(control_changes))
        )
    state_rows = [{
        "model": short, "record_id": record_id,
        "primary_minus_median_control_regret_change": float(np.mean(values)),
    } for record_id, values in sorted(by_state.items())]
    values = np.asarray([
        row["primary_minus_median_control_regret_change"] for row in state_rows
    ], dtype=np.float64)
    ids = np.asarray([row["record_id"] for row in state_rows], dtype=np.int64)
    ci = bootstrap_state_mean(values, ids, f"{short}_ablation_control")
    write_csv(EVIDENCE_DIR / f"ablation_control_{short}_state_rows.csv", state_rows)
    return {
        "model": short, "eligible_states": len(state_rows),
        "mean_primary_minus_median_control_regret_change": float(np.mean(values)),
        "ci95": ci, "passed": bool(
            np.mean(values) > 0 and (ci[0] > 0 if RUN_MODE == "pilot" else True)
        ),
    }


def free_null_summary(short):
    baseline = [
        row for row in PLANNING_ROWS
        if row["model"] == short and row["condition"] == "baseline"
        and row["regime"] == "free"
    ]
    closure = [
        row for row in CLOSURE_ROWS
        if row["model"] == short and row["condition"] == "primary_r128_swap"
        and row["regime"] == "free"
    ]
    exact = [row for row in baseline if row["true_cost_spread"] <= MAX_FREE_TRUE_COST_SPREAD]
    return {
        "model": short, "free_states": len({row["record_id"] for row in baseline}),
        "free_planning_tasks": len(baseline), "exact_physical_null_tasks": len(exact),
        "median_true_cost_spread": float(np.median([row["true_cost_spread"] for row in baseline])),
        "median_model_cost_spread_on_exact_null": (
            float(np.median([row["model_cost_spread"] for row in exact])) if exact else None
        ),
        "median_primary_intervention_energy": float(np.median([
            row["effect_energy"] for row in closure
        ])),
    }


def fresh_run_certificate():
    expected = {
        "screened_construction_states": len(CONSTRUCTION_POOL_SPECS),
        "screened_evaluation_states": len(EVALUATION_POOL_SPECS),
        "construction_truth_generated": len(CONSTRUCTION_RECORDS),
        "evaluation_truth_generated": len(EVALUATION_RECORDS),
        "jepa_baseline_generated": len(EVALUATION_RECORDS),
        "jepa_intervention_generated": len(EVALUATION_RECORDS),
        "dino_baseline_generated": len(EVALUATION_RECORDS),
        "dino_intervention_generated": len(EVALUATION_RECORDS),
        "cache_hits": 0,
    }
    passed = bool(
        not OUT_PREEXISTED and PROVENANCE_COUNTS == expected
        and SOURCE_IDENTITY.get("confirmation_eligible", False)
        and ALL_CONSTRUCTION_GATES_PASS
    )
    payload = {
        "out_preexisted": bool(OUT_PREEXISTED),
        "observed_counts": dict(PROVENANCE_COUNTS), "expected_counts": expected,
        "source_execution_verified": bool(SOURCE_IDENTITY.get("confirmation_eligible", False)),
        "all_construction_gates_pass": bool(ALL_CONSTRUCTION_GATES_PASS),
        "passed": passed,
    }
    write_json(OUT / "fresh_run_certificate.json", payload)
    return payload


def make_plots(within, paired, gaps):
    figure, axes = plt.subplots(1, 4, figsize=(20, 4.7))
    for short, color in [("jepa", "#4c78a8"), ("dino", "#f58518")]:
        payload = json.loads(
            (SUBSPACE_DIR / f"construction_layer_screen_{short}.json").read_text()
        )
        rows = payload["block_summaries"]
        axes[0].plot(
            [row["block"] for row in rows], [row["mean_output_cka"] for row in rows],
            marker="o", label=short.upper(), color=color,
        )
    axes[0].set(xlabel="predictor block", ylabel="construction CKA", title="Construction-only layer screen")
    axes[0].legend()

    paired_rows = paired[1]
    axes[1].scatter(
        [row["difference_grounded_coefficient"] for row in paired_rows],
        [row["outcome"] for row in paired_rows], alpha=0.3, color="#54a24b",
    )
    axes[1].set(
        xlabel="DINO − JEPA grounded coefficient",
        ylabel="DINO − JEPA regret", title="Paired model difference",
    )

    labels = ["JEPA", "DINO", "paired Δ"]
    gates = [within["jepa"], within["dino"], paired[0]]
    positions = np.arange(3)
    axes[2].bar(positions - 0.18, [gate["base_oof_mse"] for gate in gates], 0.36, label="base")
    axes[2].bar(positions + 0.18, [gate["grounded_oof_mse"] for gate in gates], 0.36, label="+ grounded")
    axes[2].set_xticks(positions, labels)
    axes[2].set(ylabel="out-of-fold MSE", title="Incremental closure value")
    axes[2].legend()

    axes[3].bar(
        ["JEPA", "DINO"],
        [gaps[short]["mean_self_minus_grounded"] for short in ["jepa", "dino"]],
        color=["#4c78a8", "#f58518"],
    )
    axes[3].axhline(0, color="black", linewidth=0.8)
    axes[3].set(ylabel="self − grounded coefficient", title="Grounding gap replication")
    figure.tight_layout()
    figure.savefig(PLOT_DIR / "stage31_cross_model_certificate_summary.png", dpi=180)
    plt.close(figure)


DECISION_PAYLOAD = {"status": "INCONCLUSIVE"}
if not PIPELINE_FAILED:
    try:
        verify_executed_notebook_through(
            "# Apply the preregistered within-model and paired cross-model closure gates."
        )
        if not ALL_CONSTRUCTION_GATES_PASS:
            DECISION_PAYLOAD = {
                "status": "CONSTRUCTION_GATE_FAILED_NO_EVALUATION",
                "candidate_status": "CONSTRUCTION_GATE_FAILED_NO_EVALUATION",
                "confirmation_eligible": False,
                "construction_gates": CONSTRUCTION_GATES,
            }
        else:
            MODEL_DATASETS = {
                short: joined_model_dataset(short) for short in ["jepa", "dino"]
            }
            WITHIN_MODEL_GATES = {
                short: predictive_result(
                    MODEL_DATASETS[short],
                    ["grounded_coefficient", "grounded_cosine"],
                    f"within_{short}", MIN_WITHIN_MODEL_RELATIVE_MSE_IMPROVEMENT,
                ) for short in ["jepa", "dino"]
            }
            PAIRED_GATE, PAIRED_ROWS = paired_predictive_gate(MODEL_DATASETS)
            REPLICATION_GAPS = {
                short: replication_gap(short) for short in ["jepa", "dino"]
            }
            ABLATION_CONTROLS = {
                short: ablation_control_summary(short) for short in ["jepa", "dino"]
            }
            FREE_NULLS = {short: free_null_summary(short) for short in ["jepa", "dino"]}
            FRESH_CERTIFICATE = fresh_run_certificate()
            both_within = all(gate["passed"] for gate in WITHIN_MODEL_GATES.values())
            if RUN_MODE == "smoke":
                candidate_status = "SMOKE_ONLY"
            elif both_within and PAIRED_GATE["passed"]:
                candidate_status = "CROSS_MODEL_GROUNDED_CLOSURE_CERTIFICATE_SUPPORTED"
            elif PAIRED_GATE["passed"]:
                candidate_status = "PAIRED_CROSS_MODEL_CERTIFICATE_ONLY"
            elif both_within:
                candidate_status = "WITHIN_MODEL_REPLICATION_WITHOUT_PAIRED_CERTIFICATE"
            elif WITHIN_MODEL_GATES["jepa"]["passed"]:
                candidate_status = "JEPA_ONLY_GROUNDED_CLOSURE_REPLICATION"
            else:
                candidate_status = "NO_CROSS_MODEL_GROUNDED_CLOSURE_GENERALIZATION"
            confirmation_eligible = bool(
                SOURCE_IDENTITY.get("confirmation_eligible", False)
                and FRESH_CERTIFICATE["passed"]
            )
            status = (
                candidate_status if RUN_MODE == "smoke" or confirmation_eligible
                else "UNBOUND_NONFRESH_EXPLORATORY_RESULT"
            )
            DECISION_PAYLOAD = {
                "status": status, "candidate_status": candidate_status,
                "confirmation_eligible": confirmation_eligible,
                "construction_gates": CONSTRUCTION_GATES,
                "within_model_grounded_reliability_gates": WITHIN_MODEL_GATES,
                "paired_cross_model_grounded_reliability_gate": PAIRED_GATE,
                "causal_grounding_gap_replications": REPLICATION_GAPS,
                "matched_ablation_control_summaries": ABLATION_CONTROLS,
                "free_motion_null_summaries": FREE_NULLS,
                "fresh_run_certificate": FRESH_CERTIFICATE,
                "estimand_contract": {
                    "primary_outcome": "DINO-WM minus JEPA-WM normalized physical regret",
                    "closure_schedules": DIAGNOSTIC_SCHEDULES,
                    "planning_goal_schedules": PLANNING_GOAL_SCHEDULES,
                    "closure_and_goal_contrasts_disjoint": True,
                    "planner_score": "visual MSE plus 0.1 proprio MSE",
                    "physical_outcome": "exact simulator normalized block-pose cost",
                    "inference_and_bootstrap_unit": "initial physical state",
                    "subspaces": "separate model-specific construction-only bases",
                },
                "claim_boundary": {
                    "models": MODEL_NAMES, "one_environment": True,
                    "same_physical_tasks_paired_across_models": True,
                    "common_coordinate_system_between_carriers_claimed": False,
                    "learned_decoder_or_reader_used": False,
                    "evaluation_set_subspace_tuning": False,
                    "jacobian_jvp_vjp_or_gradient_used": False,
                    "terminal_exhaustive_planner_not_full_closed_loop_mpc": True,
                    "architecture_or_training_objective_cause_identified": False,
                },
                "prespecified_next_step_if_positive": (
                    "full closed-loop CEM intervention study plus a third architecture/environment"
                ),
            }
            make_plots(WITHIN_MODEL_GATES, (PAIRED_GATE, PAIRED_ROWS), REPLICATION_GAPS)
        write_json(OUT / "stage31_decision.json", DECISION_PAYLOAD)
        print(json.dumps(DECISION_PAYLOAD, indent=2))
    except Exception:
        record_failure("stage31_decision_and_plots")
        DECISION_PAYLOAD = {"status": "INCONCLUSIVE", "failure": FAILURE_MESSAGE}

if not (OUT / "stage31_decision.json").exists():
    write_json(OUT / "stage31_decision.json", DECISION_PAYLOAD)
'''


packaging = STAGE30.packaging.replace(
    "stage30_grounded_planning_value_result_bundle_",
    "stage31_cross_model_certificate_result_bundle_",
)


protocol_sources = [
    introduction,
    configuration,
    installation,
    setup,
    analysis_helpers,
    model_helpers,
    design,
    physical_truth,
    construction_and_subspaces,
    evaluation,
    decision,
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
    code(physical_truth),
    code(construction_and_subspaces),
    code(evaluation),
    code(decision),
    code(packaging),
]
for index, cell in enumerate(cells):
    cell["id"] = f"stage31-{index:02d}"

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "L4", "name": TARGET.name, "provenance": []},
        "kernelspec": {
            "display_name": "Python 3", "language": "python", "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
TARGET.write_text(json.dumps(notebook, indent=1) + "\n")
print(f"Wrote {TARGET}")
