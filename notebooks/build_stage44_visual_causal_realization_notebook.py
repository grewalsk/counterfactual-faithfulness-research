"""Build the prospective Stage 44 visual--causal realization audit Colab."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
TARGET = ROOT / "44_visual_causal_realization_audit.ipynb"
NUMERICAL = REPOSITORY / "src/cf_faithfulness/stage44_visual_causal_realization.py"


def load_builder(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BASE = load_builder(
    ROOT / "build_stage42_event_conditioned_oracle_hybrid_notebook.py",
    "stage42_builder_for_stage44",
)

code = BASE.code
markdown = BASE.markdown
replace_assignment = BASE.replace_assignment
replace_block = BASE.replace_block
assigned_uppercase_names = BASE.assigned_uppercase_names
function_sources = BASE.function_sources


def rename(value: str) -> str:
    for old, new in [
        ("Stage 42", "Stage 44"),
        ("STAGE42", "STAGE44"),
        ("stage42", "stage44"),
    ]:
        value = value.replace(old, new)
    return value


def must_replace(value: str, old: str, new: str, *, count: int = 1) -> str:
    observed = value.count(old)
    if observed != count:
        raise RuntimeError(f"expected {count} copies of {old!r}, found {observed}")
    return value.replace(old, new, count)


introduction = r'''# Stage 44: visual--causal realization audit

## Prospective representation / predictor / recursion decision experiment

Stage 43 established a sharp negative result on a fresh event-rich PushT bank:
post-hoc and recursive oracle reset families did not repair the registered
contact-tail failure.  It did **not** establish that recursive states leave the
visual support of the target encoder, nor did the compressed Stage 33--43
carrier charts show what the predicted patch fields represent.  Stage 44 asks
the missing question before another architecture is trained.

For both fixed public checkpoints, this notebook preserves the native
`16 x 16 x 384` visual field and separates three maps:

1. the frozen target encoder applied to the true successor RGB frame;
2. a one-step, teacher-forced prediction from true encoded history; and
3. the ordinary recursive prediction from predicted history.

A channel basis, physical readout, and patch-localization probe are fit on
construction data only, with ridge penalties chosen only on model-selection
families and final refits using construction plus calibration.  Evaluation
families remain unopened until all artifacts and thresholds are hashed.  Local
nearest-support distance and local tangent/normal residuals are reported
separately; reconstruction error is never relabeled as an off-manifold result.

The official upstream VM2M decoder heads are loaded exactly as assigned by the
pinned JEPA-WMs PushT configs: the INet decoder for JEPA-WM and the 05norm
decoder for DINO-WM.  Their downloaded bytes are hashed before use.  True
target features are decoded alongside predicted features to expose each
decoder's own reconstruction floor.  RGB panels are qualitative diagnostics,
not decision gates.

Matched action-word pairs measure whether the predicted visual change aligns
with the true counterfactual visual change.  A prespecified oracle block-patch
swap and an equal-area background swap test whether the frozen physical probe
responds specifically to object-region content.  This is an intervention on a
saved representation, not a claim that the patch mask is available to a
deployed agent.

The outcome is a fail-closed architecture decision.  If true target features
do not expose object and physical state, an object-centric encoder experiment
is authorized.  If the target encoder is sufficient but one-step prediction
fails, counterfactual predictive-state training is authorized.  If one-step
prediction works but recursion expands normal-to-support error, recursive
closure training is authorized.  Only if visual state, recursion, and causal
effects all pass is a sealed planning-objective audit next.  No planner is run,
no checkpoint is fine-tuned, and no causal-identification, minimal-state,
Koopman, deployment, or general-world-model claim is authorized here.
'''


configuration = rename(BASE.configuration)
for name, value in {
    "PROTOCOL_ID": '"stage44-visual-causal-realization-audit-v2"',
    "NOTEBOOK_PROTOCOL_SHA256": '"__PROTOCOL_DIGEST__"',
    "EVIDENCE_STATUS": '"FRESH_VISUAL_CAUSAL_REALIZATION_AUDIT"',
    "EXPERIMENT_NOTEBOOK_PATH": '"notebooks/44_visual_causal_realization_audit.ipynb"',
    "EXPERIMENT_BUILDER_PATH": '"notebooks/build_stage44_visual_causal_realization_notebook.py"',
    "EXPERIMENT_NUMERICAL_PATH": '"src/cf_faithfulness/stage44_visual_causal_realization.py"',
    "OUTPUT_DIR": '"/content/counterfactual_faithfulness_stage44_vcra"',
    "DRIVE_OUTPUT_DIR": '"/content/drive/MyDrive/counterfactual_faithfulness_stage44_vcra"',
    "RUN_REQUEST_PATH": '"/content/drive/MyDrive/counterfactual_faithfulness_stage44_vcra/stage44_run_request.json"',
    "SEED": "440101",
    "DESIGN_SEED": "440141",
    "DECODER_SEED": "440183",
    "RANK_SEED": "440213",
    "CALIBRATION_SEED": "440253",
    "BOOTSTRAP_SEED": "440283",
    "CONTROL_SEED": "440351",
    "CONSTRUCTION_TRAJECTORY_POOL": "list(range(192000, 194000))",
    "MODEL_SELECTION_TRAJECTORY_POOL": "list(range(194000, 196000))",
    "CALIBRATION_TRAJECTORY_POOL": "list(range(196000, 198000))",
    "EVALUATION_TRAJECTORY_POOL": "list(range(198000, 214000))",
    "TASK_ID_OFFSET": "440000",
}.items():
    configuration = replace_assignment(configuration, name, value)
configuration = replace_assignment(
    configuration, "FINAL_TRAINING_SEEDS", '[4401] if RUN_MODE == "pilot" else [4401]'
)
configuration = replace_block(
    configuration,
    "CANONICAL_RESPONSE_WORD_NAMES = [",
    "CALIBRATION_INTERCHANGE_PAIRS =",
    r'''CANONICAL_RESPONSE_WORD_NAMES = ["A", "B", "C", "D", "AB", "CD", "BA", "DC"]
CONSTRUCTION_WORD_NAMES = [
    "APQSSSQPP", "BPQSSSQPP",
    "APQSSS0QPP", "BPQSSS0QPP",
    "APQSSS00QPP", "BPQSSS00QPP",
    "APQSSS000QPP", "BPQSSS000QPP",
]
MODEL_SELECTION_WORD_NAMES = [
    "CPQSSSQPP", "DPQSSSQPP",
    "CPQSSS0QPP", "DPQSSS0QPP",
    "CPQSSS00QPP", "DPQSSS00QPP",
    "CPQSSS000QPP", "DPQSSS000QPP",
]
CALIBRATION_WORD_NAMES = [
    "QAPSSSPPQ", "QBPSSSPPQ",
    "QAPSSS0PPQ", "QBPSSS0PPQ",
    "QAPSSS00PPQ", "QBPSSS00PPQ",
    "QAPSSS000PPQ", "QBPSSS000PPQ",
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
    "LQPSSSQPP", "TQPSSSQPP",
    "LQPSSS0QPP", "TQPSSS0QPP",
    "LQPSSS00QPP", "TQPSSS00QPP",
    "LQPSSS000QPP", "TQPSSS000QPP",
]
PLANNING_WORD_NAMES = []
EVALUATION_WORD_NAMES_REGISTERED = list(CLOSURE_EVALUATION_WORD_NAMES)
EVALUATION_WORD_SPECS = [
    stage39_word_spec(name) for name in EVALUATION_WORD_NAMES_REGISTERED
]
''',
)
configuration = re.sub(r"^PROTOCOL_CONFIG_KEYS = \[.*\]\n?", "", configuration, flags=re.M)
configuration += r'''

COUNTERFACTUAL_PAIRS_BY_SPLIT = {
    "construction": list(zip(CONSTRUCTION_WORD_NAMES[0::2], CONSTRUCTION_WORD_NAMES[1::2])),
    "model_selection": list(zip(MODEL_SELECTION_WORD_NAMES[0::2], MODEL_SELECTION_WORD_NAMES[1::2])),
    "calibration": list(zip(CALIBRATION_WORD_NAMES[0::2], CALIBRATION_WORD_NAMES[1::2])),
    "evaluation_closure": list(zip(CLOSURE_EVALUATION_WORD_NAMES[0::2], CLOSURE_EVALUATION_WORD_NAMES[1::2])),
}
PRIMARY_MODEL = "jepa"
VISUAL_PCA_RANK = 8
VISUAL_PCA_MAX_PATCH_ROWS = 16384
SPATIAL_GRID_SIZE = 16
SPATIAL_PYRAMID_BINS = 4
PATCH_CLASS_COUNT = 3
PATCH_AGENT_RADIUS = 0.80
PATCH_BLOCK_RADIUS = 1.75
PATCH_RIDGE_MAX_ROWS = 120000
VISUAL_RIDGE_PENALTIES = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
SUPPORT_REFERENCE_MAX_ROWS = 1024
SUPPORT_QUERY_MAX_ROWS = 1024
SUPPORT_NEIGHBORS = 16
SUPPORT_TANGENT_RANK = 8
VISUALIZATION_FAMILIES = 4
MIN_TARGET_PHYSICAL_R2 = 0.50
MIN_TARGET_PATCH_MACRO_AUROC = 0.72
MAX_TEACHER_VISUAL_NMSE = 1.00
MAX_RECURSIVE_TO_TEACHER_NMSE_RATIO = 1.25
MAX_RECURSIVE_TO_TEACHER_NORMAL_RATIO = 1.25
MIN_COUNTERFACTUAL_VISUAL_COSINE = 0.40
MIN_COUNTERFACTUAL_PHYSICAL_COSINE = 0.30
MIN_OBJECT_TO_BACKGROUND_MEDIATION_RATIO = 1.25
OFFICIAL_IMAGE_DECODER_FILENAMES = {
    "jepa_wm_pusht": "vm2m_lpips_dv2vits_vitldec_224_INet.pth.tar",
    "dino_wm_pusht": "vm2m_lpips_dv2vits_vitldec_224_05norm.pth.tar",
}
OFFICIAL_IMAGE_DECODER_URLS = {
    "jepa_wm_pusht": "https://dl.fbaipublicfiles.com/jepa-wms/vm2m_lpips_dv2vits_vitldec_224_INet.pth.tar",
    "dino_wm_pusht": "https://dl.fbaipublicfiles.com/jepa-wms/vm2m_lpips_dv2vits_vitldec_224_05norm.pth.tar",
}
STAGE43_NEGATIVE_AUDIT = {
    "protocol_id": "stage43-recursive-reset-sufficiency-v1",
    "run_signature_prefix": "2f1e70f121fb",
    "status": "reset_hypothesis_not_supported",
    "role": "prior_negative_evidence_only",
}
assert PRIMARY_MODEL == "jepa"
assert VISUAL_PCA_RANK == 8
assert SUPPORT_TANGENT_RANK < SUPPORT_NEIGHBORS
assert PLANNING_WORD_NAMES == []
'''
configuration += "\n\nPROTOCOL_CONFIG_KEYS = " + repr(
    assigned_uppercase_names(configuration)
) + "\n"


installation = BASE.installation
setup = rename(BASE.setup).replace("stage44_ecoh_v2", "stage44_vcra")
setup += r'''

DENSE_VISUAL_DIR = OUT / "dense_visual"
VISUAL_PROBE_DIR = OUT / "visual_probes"
VISUAL_PANEL_DIR = PLOT_DIR / "visual_realizations"
for directory in [DENSE_VISUAL_DIR, VISUAL_PROBE_DIR, VISUAL_PANEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
'''


analysis_helpers = rename(BASE.analysis_helpers)
visual_helpers = function_sources(
    NUMERICAL.read_text(),
    [
        "_finite_matrix", "canonical_visual_tokens", "fit_channel_pca",
        "project_visual_tokens", "spatial_pyramid_summary", "fit_ridge",
        "ridge_predict", "select_ridge_penalty", "variance_weighted_r2",
        "binary_auroc", "macro_one_vs_rest_auroc", "local_support_geometry",
        "row_cosine", "counterfactual_realization_metrics",
        "masked_effect_energy", "Stage44Decision", "derive_stage44_decision",
    ],
)
visual_helpers = visual_helpers.replace(
    "class Stage44Decision:\n", "@dataclass(frozen=True)\nclass Stage44Decision:\n"
)
analysis_helpers += "\n\n" + visual_helpers


model_helpers = BASE.model_helpers
disabled_heads = '''    for config_path in config_paths:
        config = yaml.safe_load(config_path.read_text())
        heads = config["model_kwargs"]["pretrain_kwargs"]["heads_cfg"]
        heads["architectures"] = {}
        heads["pretrain_dec_path"] = None
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return repo'''
enabled_heads = '''    for config_path in config_paths:
        config = yaml.safe_load(config_path.read_text())
        heads = config["model_kwargs"]["pretrain_kwargs"]["heads_cfg"]
        image_architecture = heads.get("architectures", {}).get("image_head", {})
        if image_architecture.get("kind") != "vit":
            raise RuntimeError(f"official image decoder architecture changed: {config_path}")
        expected_name = (
            "dino_wm_pusht" if "/dino-wm/" in str(config_path)
            else "jepa_wm_pusht"
        )
        observed_url = heads.get("pretrain_dec_path", {}).get("image_head")
        if observed_url != OFFICIAL_IMAGE_DECODER_URLS[expected_name]:
            raise RuntimeError(f"official image decoder URL changed: {config_path}")
        # The upstream PushT planning config also names a locally trained state
        # head under JEPAWM_LOGS.  Stage 44 needs only the official public image
        # head, so remove that unavailable local head without changing the image
        # architecture, source URL, or checkpoint.
        heads["architectures"] = {"image_head": image_architecture}
        heads["pretrain_dec_path"] = {"image_head": observed_url}
        heads["new_path_heads"] = {"image_head": True}
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return repo'''
model_helpers = must_replace(model_helpers, disabled_heads, enabled_heads)
model_helpers = must_replace(
    model_helpers,
    '    bundle = validate_world_model(model, model_name)\n    bundle["preprocessor"] = preprocessor',
    '''    bundle = validate_world_model(model, model_name)
    if "image_head" not in getattr(model, "heads", {}):
        raise RuntimeError(f"{model_name} official image decoder is unavailable")
    bundle["preprocessor"] = preprocessor
    decoder_name = OFFICIAL_IMAGE_DECODER_FILENAMES[model_name]
    decoder_matches = sorted(path for path in CACHE_ROOT.rglob(decoder_name) if path.is_file())
    if not decoder_matches:
        raise RuntimeError(f"cached official decoder is missing: {decoder_name}")
    decoder_digests = {sha256_file(path) for path in decoder_matches}
    if len(decoder_digests) != 1:
        raise RuntimeError(f"conflicting cached official decoders found: {decoder_name}")
    decoder_asset = {
        "name": decoder_name,
        "path": str(decoder_matches[0]),
        "sha256": next(iter(decoder_digests)),
        "source_url": OFFICIAL_IMAGE_DECODER_URLS[model_name],
        "upstream_commit": REPO_COMMIT,
    }
    bundle["decoder_asset"] = decoder_asset''',
)
model_helpers = must_replace(
    model_helpers,
    '    assets.append(cached_verified_asset("dinov2_vits14_pretrain.pth"))',
    '    assets.append(cached_verified_asset("dinov2_vits14_pretrain.pth"))\n    assets.append(decoder_asset)',
)


design_and_runtime_helpers = BASE.design_and_runtime_helpers
design_and_runtime_helpers = design_and_runtime_helpers.replace(
    "np.cross(diagnostic[i], diagnostic[j])",
    "diagnostic[i, 0] * diagnostic[j, 1] - diagnostic[i, 1] * diagnostic[j, 0]",
)
physical_truth = rename(BASE.physical_truth)
split_word_helpers = r'''
def stage44_names_for_split(split):
    table = {
        "construction": CONSTRUCTION_WORD_NAMES,
        "model_selection": MODEL_SELECTION_WORD_NAMES,
        "calibration": CALIBRATION_WORD_NAMES,
        "evaluation": CLOSURE_EVALUATION_WORD_NAMES,
        "evaluation_closure": CLOSURE_EVALUATION_WORD_NAMES,
    }
    if str(split) not in table:
        raise KeyError(f"unknown Stage 44 split {split!r}")
    return list(table[str(split)])


def stage39_truth_word_names(split):
    split_name = str(split)
    names = list(stage44_names_for_split(split_name))
    if split_name in {"construction", "model_selection"}:
        # The inherited simulator-only chart is a pre-model integrity gate. It
        # requires the canonical response and order-control words, but never
        # reads evaluation trajectories or prediction outcomes.
        names.extend(CANONICAL_RESPONSE_WORD_NAMES)
        names.extend(name for pair in CORE_ORDER_PAIRS for name in pair)
    controls = {
        ZERO_WORD_NAMES[int(WORD_BY_NAME[name]["length"])] for name in names
    }
    return sorted(
        set(names) | controls,
        key=lambda name: (int(WORD_BY_NAME[name]["length"]), name),
    )
'''
truth_names_start = physical_truth.index("def stage39_truth_word_names(split):")
truth_generator_start = physical_truth.index("def generate_truth_record(record):")
physical_truth = (
    physical_truth[:truth_names_start] + split_word_helpers + "\n\n"
    + physical_truth[truth_generator_start:]
)
physical_truth = must_replace(
    physical_truth,
    '        "path_states", "path_observables", "path_mask",',
    '        "path_states", "path_observables", "path_mask", "path_visuals",',
)
physical_truth = must_replace(
    physical_truth,
    "    path_mask = np.zeros((len(words), MAX_WORD_LENGTH), dtype=bool)\n    contacts =",
    "    path_mask = np.zeros((len(words), MAX_WORD_LENGTH), dtype=bool)\n"
    "    path_visuals = np.zeros((len(words), MAX_WORD_LENGTH, 224, 224, 3), dtype=np.uint8)\n"
    "    contacts =",
)
physical_truth = must_replace(
    physical_truth,
    "        result = rollout_word(record, word, retain_visual=False)",
    "        result = rollout_word(record, word, retain_visual=True)",
)
physical_truth = must_replace(
    physical_truth,
    "        path_mask[index, :length] = True\n        contacts[index, :action_steps] = result[\"contacts\"]",
    "        path_mask[index, :length] = True\n"
    "        path_visuals[index, :length] = result[\"path_visuals\"]\n"
    "        contacts[index, :action_steps] = result[\"contacts\"]",
)
physical_truth = must_replace(
    physical_truth,
    "        path_mask=path_mask,\n        contact_counts=contacts,",
    "        path_mask=path_mask,\n        path_visuals=path_visuals,\n        contact_counts=contacts,",
)


simulator_preflight = r'''# Verify the sealed physical design and official decoder contract before development fitting.
SIMULATOR_PREFLIGHT_PASSED = False
DECODER_CONTRACTS = {}


def stage44_names_for_split(split):
    table = {
        "construction": CONSTRUCTION_WORD_NAMES,
        "model_selection": MODEL_SELECTION_WORD_NAMES,
        "calibration": CALIBRATION_WORD_NAMES,
        "evaluation": CLOSURE_EVALUATION_WORD_NAMES,
        "evaluation_closure": CLOSURE_EVALUATION_WORD_NAMES,
    }
    if str(split) not in table:
        raise KeyError(f"unknown Stage 44 split {split!r}")
    return list(table[str(split)])


def stage44_truth_contract(record, split):
    path = truth_path(record)
    validate_digest_sidecar(path)
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "identity", "word_names", "word_lengths", "path_states",
            "path_observables", "path_mask", "path_visuals", "initial_visual",
            "initial_proprio", "contact_counts",
        }
        if not required.issubset(payload.files):
            raise RuntimeError(f"truth shard is incomplete: {path}")
        names = set(map(str, payload["word_names"]))
        expected = set(stage44_names_for_split(split))
        if not expected.issubset(names):
            raise RuntimeError(f"truth word bank changed: {path}")
    return True


if not PIPELINE_FAILED:
    try:
        verify_executed_notebook_through(
            "# Verify the sealed physical design and official decoder contract before development fitting."
        )
        for split in ["construction", "model_selection", "calibration", "evaluation"]:
            for record in SELECTED_RECORDS[split]:
                stage44_truth_contract(record, split)
        SIMULATOR_PREFLIGHT_PASSED = True
        write_json(OUT / "stage44_physical_design_preflight.json", {
            "passed": True, "model_outputs_read": False,
            "evaluation_visual_outcomes_read": False,
            "planning_opened": False,
        })
        atomic_checkpoint("stage44_physical_design_verified", {
            "passed": True, "planning_opened": False,
        })
        stage44_phase("physical_design_verified")
    except Exception:
        record_failure("stage44_physical_design_preflight")
'''


dense_helpers = r'''# Materialize dense target, teacher-forced, and recursive visual fields on development splits.

def stage44_names_for_split(split):
    table = {
        "construction": CONSTRUCTION_WORD_NAMES,
        "model_selection": MODEL_SELECTION_WORD_NAMES,
        "calibration": CALIBRATION_WORD_NAMES,
        "evaluation": CLOSURE_EVALUATION_WORD_NAMES,
        "evaluation_closure": CLOSURE_EVALUATION_WORD_NAMES,
    }
    if str(split) not in table:
        raise KeyError(f"unknown Stage 44 split {split!r}")
    return list(table[str(split)])


def stage44_dense_path(short, split, record):
    return DENSE_VISUAL_DIR / f"{short}_{split}_{int(record['record_id'])}.npz"


def stage44_truth_word(bundle, record, payload, name):
    lookup = {str(value): index for index, value in enumerate(payload["word_names"])}
    index = lookup[str(name)]
    length = int(WORD_BY_NAME[name]["length"])
    visuals = np.asarray(payload["path_visuals"][index, :length], dtype=np.uint8)
    states = np.asarray(payload["path_states"][index, :length], dtype=np.float32)
    proprios = np.concatenate([states[:, :2], states[:, 5:7]], axis=1)
    observation = to_model_observation(visuals[None], proprios[None])
    with torch.inference_mode():
        encoded = bundle["model"].encode(observation)
    target = canonical_visual_tokens(
        encoded["visual"].detach().float().cpu().numpy(),
        expected_tokens=EXPECTED_VISUAL_TOKENS,
        expected_width=EXPECTED_VISUAL_WIDTH,
    )
    if target.shape != (length, EXPECTED_VISUAL_TOKENS, EXPECTED_VISUAL_WIDTH):
        raise RuntimeError(f"target visual contract changed for {name}: {target.shape}")
    return encoded, target, visuals, states


def stage44_teacher_forced_word(bundle, record, name, target_encoded):
    length = int(WORD_BY_NAME[name]["length"])
    initial = encoded_initial_from_record(bundle, record)
    actions, _ = word_actions(record, WORD_BY_NAME[name])
    action_tensor = model_action_tensor(bundle["preprocessor"], actions[None], length)
    model = bundle["model"]
    with torch.inference_mode():
        action_features = model.model.encode_act(action_tensor.permute(1, 0, 2).contiguous())
        visual_history = initial["visual"]
        proprio_history = initial["proprio"]
        predictions = []
        for step in range(length):
            visual, _, _ = model.model.forward_pred(
                visual_history[:, -model.ctxt_window:],
                action_features[:, : step + 1][:, -model.ctxt_window:],
                proprio_history[:, -model.ctxt_window:],
            )
            predictions.append(visual[:, -1:])
            visual_history = torch.cat([
                visual_history, target_encoded["visual"][:, step: step + 1]
            ], dim=1)
            proprio_history = torch.cat([
                proprio_history, target_encoded["proprio"][:, step: step + 1]
            ], dim=1)
        predicted = torch.cat(predictions, dim=1)
    result = canonical_visual_tokens(
        predicted.detach().float().cpu().numpy(),
        expected_tokens=EXPECTED_VISUAL_TOKENS,
        expected_width=EXPECTED_VISUAL_WIDTH,
    )
    del action_tensor
    return result


def stage44_patch_labels(states):
    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 10:
        raise ValueError("physical state path must have width ten")
    coordinates = (np.arange(SPATIAL_GRID_SIZE, dtype=np.float64) + 0.5)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    patch_xy = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    labels = np.zeros((len(values), SPATIAL_GRID_SIZE ** 2), dtype=np.int64)
    for step, state in enumerate(values):
        agent = state[:2] / 512.0 * SPATIAL_GRID_SIZE
        block = state[2:4] / 512.0 * SPATIAL_GRID_SIZE
        agent_mask = np.linalg.norm(patch_xy - agent, axis=1) <= PATCH_AGENT_RADIUS
        block_mask = np.linalg.norm(patch_xy - block, axis=1) <= PATCH_BLOCK_RADIUS
        labels[step, agent_mask] = 1
        labels[step, block_mask] = 2
    return labels


def stage44_decode_tokens(bundle, tokens):
    values = np.asarray(tokens, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (
        EXPECTED_VISUAL_TOKENS, EXPECTED_VISUAL_WIDTH
    ):
        raise ValueError("decoder tokens have the wrong shape")
    tensor = torch.as_tensor(values, device="cuda").reshape(
        len(values), 1, 1, SPATIAL_GRID_SIZE, SPATIAL_GRID_SIZE,
        EXPECTED_VISUAL_WIDTH,
    )
    with torch.inference_mode():
        decoded = bundle["model"].decode_unroll(tensor, batch=False)
    result = np.asarray(decoded)
    if result.shape != (len(values), 224, 224, 3) or not np.all(np.isfinite(result)):
        raise RuntimeError(f"official decoder output changed: {result.shape}")
    return np.clip(result, 0, 255).astype(np.uint8)


def stage44_pca_sample(bundle):
    samples = []
    for record in SELECTED_RECORDS["construction"]:
        with np.load(truth_path(record), allow_pickle=False) as payload:
            for name in CONSTRUCTION_WORD_NAMES:
                encoded, target, _, _ = stage44_truth_word(bundle, record, payload, name)
                stride = max(1, len(target.reshape(-1, target.shape[-1])) // 32)
                samples.append(target.reshape(-1, target.shape[-1])[::stride][:32])
                del encoded
        if sum(len(value) for value in samples) >= VISUAL_PCA_MAX_PATCH_ROWS:
            break
    matrix = np.concatenate(samples)[:VISUAL_PCA_MAX_PATCH_ROWS]
    if len(matrix) < VISUAL_PCA_RANK:
        raise RuntimeError("too few construction patches for PCA")
    return fit_channel_pca(matrix, VISUAL_PCA_RANK)


def stage44_save_pca(short, artifact):
    path = VISUAL_PROBE_DIR / f"channel_pca_{short}.npz"
    atomic_npz(path, **{key: np.asarray(value) for key, value in artifact.items()})
    write_json(VISUAL_PROBE_DIR / f"channel_pca_{short}.json", {
        "model": short, "training_split": "construction",
        "evaluation_rows_used": 0, "pca_rank": VISUAL_PCA_RANK,
        "array_sha256": sha256_file(path),
    })
    write_digest_sidecar(VISUAL_PROBE_DIR / f"channel_pca_{short}.json")


def stage44_load_pca(short):
    path = VISUAL_PROBE_DIR / f"channel_pca_{short}.npz"
    validate_digest_sidecar(path)
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def stage44_write_panel(bundle, record, name, true_rgb, target, teacher, recursive):
    decoded_target = stage44_decode_tokens(bundle, target)
    decoded_teacher = stage44_decode_tokens(bundle, teacher)
    decoded_recursive = stage44_decode_tokens(bundle, recursive)
    final = len(true_rgb) - 1
    images = [
        true_rgb[final], decoded_target[final],
        decoded_teacher[final], decoded_recursive[final],
    ]
    titles = ["true RGB", "target-encoding decode", "teacher-forced", "recursive"]
    figure, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    for axis, image, title in zip(axes, images, titles):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(f"{bundle['short'].upper()} record {record['record_id']} word {name}")
    figure.tight_layout()
    path = VISUAL_PANEL_DIR / f"{bundle['short']}_{record['record_id']}_{name}.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return {
        "model": bundle["short"], "record_id": int(record["record_id"]),
        "word": str(name), "path": str(path.relative_to(OUT)),
        "target_decoder_mse": float(np.mean((decoded_target.astype(float) - true_rgb) ** 2) / 255.0**2),
        "teacher_decoder_mse": float(np.mean((decoded_teacher.astype(float) - true_rgb) ** 2) / 255.0**2),
        "recursive_decoder_mse": float(np.mean((decoded_recursive.astype(float) - true_rgb) ** 2) / 255.0**2),
        "decision_gate": False,
    }


def stage44_materialize_record(bundle, record, split, pca, *, make_panel=False):
    short = bundle["short"]
    path = stage44_dense_path(short, split, record)
    identity = f"{PROTOCOL_ID}:{RUN_SIGNATURE}:{short}:{split}:{record['record_id']}:dense-v1"
    required = {
        "identity", "word_names", "word_lengths", "mask", "target_summary",
        "teacher_summary", "recursive_summary", "target_patches",
        "teacher_patches", "recursive_patches", "patch_labels",
        "physical_targets", "trajectory_id", "record_id", "initial_mode",
    }
    if validate_npz_shard(path, required, identity):
        PROVENANCE_COUNTS["validated_cache_hits"] += 1
        return path, []
    names = stage44_names_for_split(split)
    recursive_outputs, _ = grouped_model_words(bundle, record, names)
    dimensions = (len(names), MAX_WORD_LENGTH)
    summaries = {
        kind: np.zeros((*dimensions, SPATIAL_PYRAMID_BINS ** 2 * VISUAL_PCA_RANK + VISUAL_PCA_RANK), dtype=np.float32)
        for kind in ["target", "teacher", "recursive"]
    }
    patches = {
        kind: np.zeros((*dimensions, EXPECTED_VISUAL_TOKENS, VISUAL_PCA_RANK), dtype=np.float16)
        for kind in ["target", "teacher", "recursive"]
    }
    labels = np.zeros((*dimensions, EXPECTED_VISUAL_TOKENS), dtype=np.int8)
    physical = np.zeros((*dimensions, len(GROUNDED_OBSERVABLES)), dtype=np.float32)
    mask = np.zeros(dimensions, dtype=bool)
    panel_rows = []
    with np.load(truth_path(record), allow_pickle=False) as payload:
        lookup = {str(value): index for index, value in enumerate(payload["word_names"])}
        for word_index, name in enumerate(names):
            length = int(WORD_BY_NAME[name]["length"])
            target_encoded, target, true_rgb, states = stage44_truth_word(
                bundle, record, payload, name
            )
            teacher = stage44_teacher_forced_word(bundle, record, name, target_encoded)
            recursive = np.asarray(recursive_outputs[name][0], dtype=np.float64)
            if recursive.shape != target.shape or teacher.shape != target.shape:
                raise RuntimeError(f"dense prediction alignment changed for {name}")
            for kind, value in [
                ("target", target), ("teacher", teacher), ("recursive", recursive)
            ]:
                projected = project_visual_tokens(value, pca)
                patches[kind][word_index, :length] = projected.astype(np.float16)
                summaries[kind][word_index, :length] = spatial_pyramid_summary(
                    projected, grid_size=SPATIAL_GRID_SIZE,
                    bins=SPATIAL_PYRAMID_BINS,
                ).astype(np.float32)
            labels[word_index, :length] = stage44_patch_labels(states)
            physical[word_index, :length] = payload["path_observables"][lookup[name], :length]
            mask[word_index, :length] = True
            if make_panel and word_index == 0:
                panel_rows.append(stage44_write_panel(
                    bundle, record, name, true_rgb, target, teacher, recursive
                ))
            del target_encoded
    atomic_npz(
        path, identity=np.asarray(identity), word_names=np.asarray(names),
        word_lengths=np.asarray([WORD_BY_NAME[name]["length"] for name in names], dtype=np.int64),
        mask=mask, target_summary=summaries["target"],
        teacher_summary=summaries["teacher"], recursive_summary=summaries["recursive"],
        target_patches=patches["target"], teacher_patches=patches["teacher"],
        recursive_patches=patches["recursive"], patch_labels=labels,
        physical_targets=physical,
        trajectory_id=np.asarray(int(record["trajectory_id"])),
        record_id=np.asarray(int(record["record_id"])),
        initial_mode=np.asarray(str(record["mode"])),
    )
    PROVENANCE_COUNTS["model_record_forwards"][short] += 1
    return path, panel_rows


PCA_ARTIFACTS = {}
DEVELOPMENT_PANEL_ROWS = []
if not PIPELINE_FAILED and SIMULATOR_PREFLIGHT_PASSED:
    try:
        verify_executed_notebook_through(
            "# Materialize dense target, teacher-forced, and recursive visual fields on development splits."
        )
        REPO = configure_repo()
        verify_pretrained_assets()
        for model_name in MODEL_NAMES:
            bundle = load_world_model(model_name)
            short = bundle["short"]
            try:
                DECODER_CONTRACTS[short] = {
                    **bundle["decoder_asset"], "model": model_name,
                    "decode_method": "EncPredWM.decode_unroll",
                    "evaluation_rows_used": 0,
                }
                pca_path = VISUAL_PROBE_DIR / f"channel_pca_{short}.npz"
                if pca_path.is_file():
                    pca = stage44_load_pca(short)
                else:
                    pca = stage44_pca_sample(bundle)
                    stage44_save_pca(short, pca)
                PCA_ARTIFACTS[short] = pca
                for split in ["construction", "model_selection", "calibration"]:
                    for index, record in enumerate(SELECTED_RECORDS[split]):
                        _, panel_rows = stage44_materialize_record(
                            bundle, record, split, pca,
                            make_panel=bool(split == "construction" and index == 0),
                        )
                        DEVELOPMENT_PANEL_ROWS.extend(panel_rows)
                        write_json(OUT / f"dense_{short}_{split}_progress.json", {
                            "completed": index + 1, "total": len(SELECTED_RECORDS[split]),
                            "last_record_id": int(record["record_id"]),
                        })
            finally:
                unload_world_model(bundle)
        write_json(OUT / "official_decoder_contracts.json", DECODER_CONTRACTS)
        write_digest_sidecar(OUT / "official_decoder_contracts.json")
        write_csv(EVIDENCE_DIR / "development_decoder_diagnostics.csv", DEVELOPMENT_PANEL_ROWS)
        atomic_checkpoint("stage44_development_dense_complete", {
            "decoder_contract_sha256": sha256_file(OUT / "official_decoder_contracts.json"),
            "evaluation_rows_used": 0, "planning_opened": False,
        })
        stage44_phase("development_dense_complete")
    except Exception:
        record_failure("stage44_development_dense_materialization")
'''


probe_freeze = r'''# Select and freeze visual probes and local-support references without held-out access.
VISUAL_ARTIFACTS = {}
SELECTION_ROWS = []


def stage44_load_split(short, split):
    result = {key: [] for key in [
        "target_summary", "teacher_summary", "recursive_summary",
        "target_patches", "teacher_patches", "recursive_patches",
        "patch_labels", "physical_targets", "groups", "record_ids",
    ]}
    record_split = "evaluation" if str(split) == "evaluation_closure" else str(split)
    for record in SELECTED_RECORDS[record_split]:
        path = stage44_dense_path(short, split, record)
        validate_digest_sidecar(path)
        with np.load(path, allow_pickle=False) as payload:
            valid = np.asarray(payload["mask"], dtype=bool)
            count = int(np.sum(valid))
            for key in ["target_summary", "teacher_summary", "recursive_summary", "physical_targets"]:
                result[key].append(np.asarray(payload[key], dtype=np.float64)[valid])
            for key in ["target_patches", "teacher_patches", "recursive_patches", "patch_labels"]:
                result[key].append(np.asarray(payload[key])[valid])
            result["groups"].append(np.repeat(int(payload["trajectory_id"]), count))
            result["record_ids"].append(np.repeat(int(payload["record_id"]), count))
    return {key: np.concatenate(value) for key, value in result.items()}


def stage44_patch_design(projected):
    value = np.asarray(projected, dtype=np.float64)
    if value.ndim != 3 or value.shape[1:] != (EXPECTED_VISUAL_TOKENS, VISUAL_PCA_RANK):
        raise ValueError("patch design input has the wrong shape")
    coordinates = (np.arange(SPATIAL_GRID_SIZE, dtype=np.float64) + 0.5) / SPATIAL_GRID_SIZE
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    position = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    position = np.repeat(position[None], len(value), axis=0)
    interactions = np.concatenate([
        value * position[:, :, 0:1], value * position[:, :, 1:2]
    ], axis=-1)
    return np.concatenate([value, position, interactions], axis=-1).reshape(-1, 3 * VISUAL_PCA_RANK + 2)


def stage44_balanced_patch_rows(projected, labels, maximum_rows, seed):
    design = stage44_patch_design(projected)
    target = np.asarray(labels, dtype=np.int64).reshape(-1)
    rng = np.random.default_rng(int(seed))
    per_class = max(1, int(maximum_rows) // PATCH_CLASS_COUNT)
    selected = []
    for label in range(PATCH_CLASS_COUNT):
        indices = np.flatnonzero(target == label)
        if len(indices) == 0:
            raise RuntimeError(f"patch class {label} has no rows")
        selected.extend(rng.choice(indices, min(per_class, len(indices)), replace=False))
    selected = np.asarray(sorted(selected), dtype=np.int64)
    one_hot = np.eye(PATCH_CLASS_COUNT, dtype=np.float64)[target[selected]]
    return design[selected], one_hot


def stage44_save_visual_artifact(short, artifact):
    path = VISUAL_PROBE_DIR / f"visual_artifact_{short}.npz"
    atomic_npz(path, **{
        key: np.asarray(value) for key, value in artifact.items()
        if key not in {"metadata"}
    })
    manifest = {
        **artifact["metadata"], "model": short,
        "array_sha256": sha256_file(path), "evaluation_rows_used": 0,
    }
    manifest_path = VISUAL_PROBE_DIR / f"visual_artifact_{short}.json"
    write_json(manifest_path, manifest)
    write_digest_sidecar(manifest_path)


def stage44_load_visual_artifact(short):
    path = VISUAL_PROBE_DIR / f"visual_artifact_{short}.npz"
    manifest_path = VISUAL_PROBE_DIR / f"visual_artifact_{short}.json"
    validate_digest_sidecar(path)
    validate_digest_sidecar(manifest_path)
    with np.load(path, allow_pickle=False) as payload:
        result = {key: np.asarray(payload[key]) for key in payload.files}
    result["metadata"] = json.loads(manifest_path.read_text())
    return result


if not PIPELINE_FAILED and SIMULATOR_PREFLIGHT_PASSED:
    try:
        verify_executed_notebook_through(
            "# Select and freeze visual probes and local-support references without held-out access."
        )
        for short in ["jepa", "dino"]:
            train = stage44_load_split(short, "construction")
            validation = stage44_load_split(short, "model_selection")
            calibration_only = stage44_load_split(short, "calibration")
            final = {
                key: np.concatenate([train[key], calibration_only[key]])
                for key in train
            }
            physical_selection = select_ridge_penalty(
                train["target_summary"], train["physical_targets"],
                validation["target_summary"], validation["physical_targets"],
                VISUAL_RIDGE_PENALTIES,
            )
            physical_probe = fit_ridge(
                final["target_summary"], final["physical_targets"],
                physical_selection["selected_penalty"],
            )
            train_patch_x, train_patch_y = stage44_balanced_patch_rows(
                train["target_patches"], train["patch_labels"],
                PATCH_RIDGE_MAX_ROWS, stable_seed(CALIBRATION_SEED, short, "patch_train"),
            )
            validation_patch_x, validation_patch_y = stage44_balanced_patch_rows(
                validation["target_patches"], validation["patch_labels"],
                PATCH_RIDGE_MAX_ROWS // 2,
                stable_seed(CALIBRATION_SEED, short, "patch_validation"),
            )
            patch_selection = select_ridge_penalty(
                train_patch_x, train_patch_y, validation_patch_x,
                validation_patch_y, VISUAL_RIDGE_PENALTIES,
            )
            final_patch_x, final_patch_y = stage44_balanced_patch_rows(
                final["target_patches"], final["patch_labels"],
                PATCH_RIDGE_MAX_ROWS, stable_seed(CALIBRATION_SEED, short, "patch_final"),
            )
            patch_probe = fit_ridge(
                final_patch_x, final_patch_y, patch_selection["selected_penalty"]
            )
            reference = final["target_summary"]
            if len(reference) > SUPPORT_REFERENCE_MAX_ROWS:
                rng = np.random.default_rng(stable_seed(CALIBRATION_SEED, short, "support"))
                reference = reference[np.sort(rng.choice(
                    len(reference), SUPPORT_REFERENCE_MAX_ROWS, replace=False
                ))]
            summary_scale = np.maximum(np.std(final["target_summary"], axis=0, ddof=1), 1e-8)
            artifact = {
                "physical_weight": physical_probe["weight"],
                "physical_intercept": physical_probe["intercept"],
                "patch_weight": patch_probe["weight"],
                "patch_intercept": patch_probe["intercept"],
                "support_reference": reference.astype(np.float32),
                "summary_scale": summary_scale,
                "metadata": {
                    "physical_penalty": physical_selection["selected_penalty"],
                    "patch_penalty": patch_selection["selected_penalty"],
                    "physical_selection_rows": physical_selection["candidate_rows"],
                    "patch_selection_rows": patch_selection["candidate_rows"],
                    "pca_training_split": "construction",
                    "penalty_selection_split": "model_selection",
                    "final_fit_splits": ["construction", "calibration"],
                    "support_reference_splits": ["construction", "calibration"],
                    "evaluation_rows_used": 0,
                },
            }
            stage44_save_visual_artifact(short, artifact)
            VISUAL_ARTIFACTS[short] = artifact
            for probe, selection in [
                ("physical", physical_selection), ("patch", patch_selection)
            ]:
                for row in selection["candidate_rows"]:
                    SELECTION_ROWS.append({
                        "model": short, "probe": probe,
                        "penalty": row["penalty"],
                        "validation_mse": row["validation_mse"],
                        "selected": bool(row["penalty"] == selection["selected_penalty"]),
                    })
        write_csv(EVIDENCE_DIR / "stage44_probe_selection_rows.csv", SELECTION_ROWS)
        open_certificate = {
            "protocol_id": PROTOCOL_ID, "run_signature": RUN_SIGNATURE,
            "channel_pca_frozen": True, "physical_probes_frozen": True,
            "patch_probes_frozen": True, "support_references_frozen": True,
            "official_decoder_contracts_hashed": True,
            "evaluation_statistics_read": False,
            "evaluation_visual_fields_encoded": False,
            "planning_permanently_sealed": True,
        }
        certificate_path = VISUAL_PROBE_DIR / "evaluation_open_certificate.json"
        write_json(certificate_path, open_certificate)
        write_digest_sidecar(certificate_path)
        atomic_checkpoint("stage44_visual_artifacts_frozen", {
            "certificate_sha256": sha256_file(certificate_path),
            "evaluation_rows_used": 0, "planning_opened": False,
        })
        stage44_phase("visual_artifacts_frozen")
    except Exception:
        record_failure("stage44_visual_probe_freeze")
'''


heldout_evaluation = r'''# Open the held-out dense visual panel only after every probe and threshold is frozen.
DECISION_PAYLOAD = {
    "status": "INCONCLUSIVE_PIPELINE_FAILURE", "passed": False,
    "planning_opened": False, "causal_claim_authorized": False,
}
SUMMARY = {}
EVALUATION_ROWS = []
HELDOUT_PANEL_ROWS = []


def stage44_probe_artifacts(artifact):
    return (
        {"weight": artifact["physical_weight"], "intercept": artifact["physical_intercept"]},
        {"weight": artifact["patch_weight"], "intercept": artifact["patch_intercept"]},
    )


def stage44_swap_diagnostics(short, split, artifact):
    physical_probe, _ = stage44_probe_artifacts(artifact)
    visual_rows, physical_rows, object_effects, background_effects = [], [], [], []
    for record in SELECTED_RECORDS["evaluation"]:
        path = stage44_dense_path(short, split, record)
        with np.load(path, allow_pickle=False) as payload:
            lookup = {str(value): index for index, value in enumerate(payload["word_names"])}
            for left_name, right_name in COUNTERFACTUAL_PAIRS_BY_SPLIT[split]:
                left_index, right_index = lookup[left_name], lookup[right_name]
                left_step = int(payload["word_lengths"][left_index]) - 1
                right_step = int(payload["word_lengths"][right_index]) - 1
                target_left = np.asarray(payload["target_summary"][left_index, left_step], dtype=np.float64)
                target_right = np.asarray(payload["target_summary"][right_index, right_step], dtype=np.float64)
                recursive_left = np.asarray(payload["recursive_summary"][left_index, left_step], dtype=np.float64)
                recursive_right = np.asarray(payload["recursive_summary"][right_index, right_step], dtype=np.float64)
                visual_rows.append((target_left, target_right, recursive_left, recursive_right))
                physical_rows.append((
                    np.asarray(payload["physical_targets"][left_index, left_step], dtype=np.float64),
                    np.asarray(payload["physical_targets"][right_index, right_step], dtype=np.float64),
                    ridge_predict(physical_probe, recursive_left[None])[0],
                    ridge_predict(physical_probe, recursive_right[None])[0],
                ))
                left_patch = np.asarray(payload["recursive_patches"][left_index, left_step], dtype=np.float64)
                right_patch = np.asarray(payload["recursive_patches"][right_index, right_step], dtype=np.float64)
                block_mask = np.asarray(payload["patch_labels"][right_index, right_step]) == 2
                background = np.flatnonzero(np.asarray(payload["patch_labels"][right_index, right_step]) == 0)
                block_indices = np.flatnonzero(block_mask)
                if len(block_indices) == 0 or len(background) < len(block_indices):
                    continue
                background_indices = background[:len(block_indices)]
                object_swap = right_patch.copy()
                object_swap[block_indices] = left_patch[block_indices]
                background_swap = right_patch.copy()
                background_swap[background_indices] = left_patch[background_indices]
                right_value = ridge_predict(
                    physical_probe, spatial_pyramid_summary(right_patch[None], grid_size=16, bins=4)
                )[0]
                object_value = ridge_predict(
                    physical_probe, spatial_pyramid_summary(object_swap[None], grid_size=16, bins=4)
                )[0]
                background_value = ridge_predict(
                    physical_probe, spatial_pyramid_summary(background_swap[None], grid_size=16, bins=4)
                )[0]
                object_effects.append(object_value - right_value)
                background_effects.append(background_value - right_value)
    def packed(rows, indices=None):
        values = [np.stack([row[index] for row in rows]) for index in range(4)]
        if indices is not None:
            values = [value[:, indices] for value in values]
        return counterfactual_realization_metrics(*values)
    visual = packed(visual_rows)
    physical = packed(physical_rows, np.asarray([2, 3, 4]))
    object_norm = np.linalg.norm(np.asarray(object_effects), axis=1)
    background_norm = np.linalg.norm(np.asarray(background_effects), axis=1)
    if len(object_norm) == 0:
        raise RuntimeError("Stage 44 object-patch intervention has no eligible rows")
    mediation_ratio = float(np.median(object_norm / np.maximum(background_norm, 1e-12)))
    return visual, physical, mediation_ratio


def stage44_evaluate_model(short, data, artifact):
    physical_probe, patch_probe = stage44_probe_artifacts(artifact)
    target_physical = ridge_predict(physical_probe, data["target_summary"])
    teacher_physical = ridge_predict(physical_probe, data["teacher_summary"])
    recursive_physical = ridge_predict(physical_probe, data["recursive_summary"])
    target_r2 = variance_weighted_r2(data["physical_targets"], target_physical)
    teacher_r2 = variance_weighted_r2(data["physical_targets"], teacher_physical)
    recursive_r2 = variance_weighted_r2(data["physical_targets"], recursive_physical)
    labels = np.asarray(data["patch_labels"], dtype=np.int64).reshape(-1)
    target_patch_scores = ridge_predict(patch_probe, stage44_patch_design(data["target_patches"]))
    recursive_patch_scores = ridge_predict(patch_probe, stage44_patch_design(data["recursive_patches"]))
    target_patch_auroc = macro_one_vs_rest_auroc(labels, target_patch_scores)
    recursive_patch_auroc = macro_one_vs_rest_auroc(labels, recursive_patch_scores)
    scale = np.asarray(artifact["summary_scale"], dtype=np.float64)
    teacher_nmse_rows = np.mean(((data["teacher_summary"] - data["target_summary"]) / scale) ** 2, axis=1)
    recursive_nmse_rows = np.mean(((data["recursive_summary"] - data["target_summary"]) / scale) ** 2, axis=1)
    support_reference = np.asarray(artifact["support_reference"], dtype=np.float64)
    support_indices = np.arange(len(data["teacher_summary"]), dtype=np.int64)
    if len(support_indices) > SUPPORT_QUERY_MAX_ROWS:
        support_indices = np.linspace(
            0, len(support_indices) - 1, SUPPORT_QUERY_MAX_ROWS, dtype=np.int64
        )
    teacher_geometry = local_support_geometry(
        support_reference, data["teacher_summary"][support_indices],
        neighbors=SUPPORT_NEIGHBORS, tangent_rank=SUPPORT_TANGENT_RANK,
    )
    recursive_geometry = local_support_geometry(
        support_reference, data["recursive_summary"][support_indices],
        neighbors=SUPPORT_NEIGHBORS, tangent_rank=SUPPORT_TANGENT_RANK,
    )
    visual_cf, physical_cf, mediation_ratio = stage44_swap_diagnostics(
        short, "evaluation_closure", artifact
    )
    metrics = {
        "target_physical_r2": target_r2,
        "teacher_physical_r2": teacher_r2,
        "recursive_physical_r2": recursive_r2,
        "target_patch_macro_auroc": target_patch_auroc,
        "recursive_patch_macro_auroc": recursive_patch_auroc,
        "teacher_visual_nmse": float(np.mean(teacher_nmse_rows)),
        "recursive_visual_nmse": float(np.mean(recursive_nmse_rows)),
        "recursive_to_teacher_nmse_ratio": float(
            np.mean(recursive_nmse_rows) / max(np.mean(teacher_nmse_rows), 1e-12)
        ),
        "teacher_median_nearest_support": float(np.median(teacher_geometry["nearest_distance"])),
        "recursive_median_nearest_support": float(np.median(recursive_geometry["nearest_distance"])),
        "teacher_median_normal_support": float(np.median(teacher_geometry["normal_distance"])),
        "recursive_median_normal_support": float(np.median(recursive_geometry["normal_distance"])),
        "recursive_to_teacher_normal_ratio": float(
            np.median(recursive_geometry["normal_distance"])
            / max(np.median(teacher_geometry["normal_distance"]), 1e-12)
        ),
        "counterfactual_visual": visual_cf,
        "counterfactual_physical": physical_cf,
        "object_to_background_mediation_ratio": mediation_ratio,
    }
    gates = {
        "encoder_observable": bool(
            target_r2 >= MIN_TARGET_PHYSICAL_R2
            and target_patch_auroc >= MIN_TARGET_PATCH_MACRO_AUROC
        ),
        "one_step_adequate": bool(np.mean(teacher_nmse_rows) <= MAX_TEACHER_VISUAL_NMSE),
        "recursive_stable": bool(
            metrics["recursive_to_teacher_nmse_ratio"] <= MAX_RECURSIVE_TO_TEACHER_NMSE_RATIO
            and metrics["recursive_to_teacher_normal_ratio"] <= MAX_RECURSIVE_TO_TEACHER_NORMAL_RATIO
        ),
        "causal_realization": bool(
            visual_cf["median_cosine"] >= MIN_COUNTERFACTUAL_VISUAL_COSINE
            and physical_cf["median_cosine"] >= MIN_COUNTERFACTUAL_PHYSICAL_COSINE
            and mediation_ratio >= MIN_OBJECT_TO_BACKGROUND_MEDIATION_RATIO
        ),
    }
    for local_index, index in enumerate(support_indices):
        EVALUATION_ROWS.append({
            "model": short, "trajectory_id": int(data["groups"][index]),
            "record_id": int(data["record_ids"][index]),
            "teacher_visual_nmse": float(teacher_nmse_rows[index]),
            "recursive_visual_nmse": float(recursive_nmse_rows[index]),
            "teacher_normal_support": float(teacher_geometry["normal_distance"][local_index]),
            "recursive_normal_support": float(recursive_geometry["normal_distance"][local_index]),
        })
    return {"metrics": metrics, "gates": gates}


if not PIPELINE_FAILED and SIMULATOR_PREFLIGHT_PASSED:
    try:
        verify_executed_notebook_through(
            "# Open the held-out dense visual panel only after every probe and threshold is frozen."
        )
        certificate_path = VISUAL_PROBE_DIR / "evaluation_open_certificate.json"
        validate_digest_sidecar(certificate_path)
        certificate = json.loads(certificate_path.read_text())
        required_true = [
            "channel_pca_frozen", "physical_probes_frozen", "patch_probes_frozen",
            "support_references_frozen", "official_decoder_contracts_hashed",
            "planning_permanently_sealed",
        ]
        if (
            certificate.get("protocol_id") != PROTOCOL_ID
            or certificate.get("run_signature") != RUN_SIGNATURE
            or any(not certificate.get(key) for key in required_true)
            or certificate.get("evaluation_statistics_read")
            or certificate.get("evaluation_visual_fields_encoded")
        ):
            raise RuntimeError("Stage 44 evaluation-open certificate is invalid")
        support_path = DESIGN_DIR / "stage44_event_support_certificate.json"
        validate_digest_sidecar(support_path)
        support_certificate = json.loads(support_path.read_text())
        support_decision = derive_stage44_support_decision(
            support_certificate, expected_families=ACTIVE_EVALUATION_TRAJECTORIES,
            minimum_reentry_rows=MIN_REENTRY_ROWS,
        )
        if not support_decision.passed:
            raise RuntimeError(support_decision.classification)
        stage44_phase("heldout_dense_start")
        panel_family_ids = sorted({
            int(record["trajectory_id"]) for record in SELECTED_RECORDS["evaluation"]
        })[:VISUALIZATION_FAMILIES]
        REPO = configure_repo()
        for model_name in MODEL_NAMES:
            bundle = load_world_model(model_name)
            short = bundle["short"]
            try:
                pca = stage44_load_pca(short)
                for index, record in enumerate(SELECTED_RECORDS["evaluation"]):
                    _, panel_rows = stage44_materialize_record(
                        bundle, record, "evaluation_closure", pca,
                        make_panel=bool(
                            int(record["trajectory_id"]) in panel_family_ids
                            and str(record["mode"]) == "post_contact"
                        ),
                    )
                    HELDOUT_PANEL_ROWS.extend(panel_rows)
                    write_json(OUT / f"dense_{short}_evaluation_progress.json", {
                        "completed": index + 1,
                        "total": len(SELECTED_RECORDS["evaluation"]),
                        "last_record_id": int(record["record_id"]),
                    })
            finally:
                unload_world_model(bundle)
        stage44_phase("heldout_dense_complete")
        for short in ["jepa", "dino"]:
            artifact = stage44_load_visual_artifact(short)
            data = stage44_load_split(short, "evaluation_closure")
            SUMMARY[short] = stage44_evaluate_model(short, data, artifact)
        primary = SUMMARY[PRIMARY_MODEL]["gates"]
        decision = derive_stage44_decision(
            support_certified=True, decoder_contract_valid=True,
            encoder_observable=primary["encoder_observable"],
            one_step_adequate=primary["one_step_adequate"],
            recursive_stable=primary["recursive_stable"],
            causal_realization=primary["causal_realization"],
        )
        DECISION_PAYLOAD = {
            "status": decision.classification,
            "next_step": decision.classification,
            "passed": bool(decision.passed),
            "encoder_observable": bool(decision.encoder_observable),
            "one_step_adequate": bool(decision.one_step_adequate),
            "recursive_stable": bool(decision.recursive_stable),
            "causal_realization": bool(decision.causal_realization),
            "counterfactual_training_authorized": bool(decision.counterfactual_training_authorized),
            "object_centric_encoder_authorized": bool(decision.object_centric_encoder_authorized),
            "planning_audit_authorized": bool(decision.planning_audit_authorized),
            "primary_model": PRIMARY_MODEL,
            "protocol_id": PROTOCOL_ID, "run_signature": RUN_SIGNATURE,
            "evidence_tier": "fresh_visual_causal_realization_audit",
            "evaluation_opened": True, "planning_opened": False,
            "causal_claim_authorized": False,
            "deployment_claim_authorized": False,
            "rgb_reconstruction_is_a_decision_gate": False,
            "oracle_patch_mask_available_at_deployment": False,
            "shared_dinov2_encoder_confound_retained": True,
            "stage43_negative_result_retained": True,
        }
        write_csv(EVIDENCE_DIR / "heldout_stage44_rows.csv", EVALUATION_ROWS)
        write_csv(EVIDENCE_DIR / "heldout_decoder_diagnostics.csv", HELDOUT_PANEL_ROWS)
        write_json(EVIDENCE_DIR / "stage44_summary.json", SUMMARY)
        write_json(OUT / "stage44_decision.json", DECISION_PAYLOAD)
        atomic_checkpoint("stage44_visual_causal_complete", {
            "decision_sha256": sha256_file(OUT / "stage44_decision.json"),
            "status": DECISION_PAYLOAD["status"], "planning_opened": False,
        })
        figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        for axis, short in zip(axes, ["jepa", "dino"]):
            metrics = SUMMARY[short]["metrics"]
            axis.bar(["teacher", "recursive"], [
                metrics["teacher_visual_nmse"], metrics["recursive_visual_nmse"]
            ])
            axis.set_title(short.upper())
            axis.set_ylabel("target-normalized visual-summary MSE")
        figure.suptitle(f"Stage 44: {DECISION_PAYLOAD['status']}")
        figure.tight_layout()
        figure.savefig(PLOT_DIR / "stage44_teacher_recursive_error.png", dpi=180)
        plt.close(figure)
        interpretation = f"""# Automatic interpretation

Status: `{DECISION_PAYLOAD['status']}`.

The primary decision applies to the fixed JEPA-WM checkpoint.  DINO-WM is a
matched comparator and shares the DINOv2 family of visual encoders, so it is
not an independent encoder replication.  Official-decoder panels are
diagnostic only.  The registered gates use target-feature observability,
teacher-forced and recursive errors, local support geometry, matched
counterfactual effects, and oracle object-versus-background patch swaps.
Planning remained sealed.
"""
        (OUT / "AUTOMATIC_INTERPRETATION.md").write_text(interpretation)
        stage44_phase("visual_causal_complete", status=DECISION_PAYLOAD["status"])
    except Exception:
        record_failure("stage44_heldout_visual_causal")
'''


packaging = rename(BASE.packaging).replace("stage44_ecoh_v2", "stage44_vcra")
packaging = packaging.replace("event_conditioned_oracle_hybrid", "visual_causal_realization_audit")


protocol_sources = [
    introduction, configuration, installation, setup, analysis_helpers,
    model_helpers, design_and_runtime_helpers, physical_truth,
    simulator_preflight, dense_helpers, probe_freeze, heldout_evaluation,
    packaging,
]
protocol_sources = [value.strip() for value in protocol_sources]
protocol_digest = hashlib.sha256(
    json.dumps(protocol_sources, ensure_ascii=False, separators=(",", ":")).encode()
).hexdigest()
configuration = configuration.replace("__PROTOCOL_DIGEST__", protocol_digest)
if "__PROTOCOL_DIGEST__" in configuration:
    raise RuntimeError("Stage 44 protocol digest placeholder was not replaced")
protocol_sources[1] = configuration

cells = [markdown(introduction)] + [code(value) for value in protocol_sources[1:]]
for index, cell in enumerate(cells):
    cell["id"] = f"stage44-{index:02d}"
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
