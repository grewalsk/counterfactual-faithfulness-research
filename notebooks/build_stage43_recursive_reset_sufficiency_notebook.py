"""Build the prospective Stage 43 recursive reset sufficiency Colab."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
TARGET = ROOT / "43_recursive_reset_sufficiency.ipynb"
NUMERICAL = REPOSITORY / "src/cf_faithfulness/stage43_recursive_reset.py"


def load_builder(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BASE = load_builder(
    ROOT / "build_stage42_event_conditioned_oracle_hybrid_notebook.py",
    "stage42_builder_for_stage43",
)

code = BASE.code
markdown = BASE.markdown
replace_assignment = BASE.replace_assignment
replace_block = BASE.replace_block
assigned_uppercase_names = BASE.assigned_uppercase_names
function_sources = BASE.function_sources


def rename(value: str) -> str:
    for old, new in [
        ("Stage 42", "Stage 43"),
        ("STAGE42", "STAGE43"),
        ("stage42", "stage43"),
    ]:
        value = value.replace(old, new)
    return value


def must_replace(value: str, old: str, new: str, *, count: int = 1) -> str:
    observed = value.count(old)
    if observed != count:
        raise RuntimeError(f"expected {count} copies of {old!r}, found {observed}")
    return value.replace(old, new, count)


introduction = r'''# Stage 43: recursive reset sufficiency lattice

## Prospective state-by-operator falsification

Stage 42 v2 completed without a pipeline failure.  It certified 48 fresh
event-rich families and found that exact event/time/normal/impulse metadata was
decodable, yet its affine correction failed the registered tail gates across
JEPA and DINO.  That result is permanently retained as negative evidence.  It
does not establish that a carrier reset is impossible: the Stage 42 head added
a correction to decoded physical outputs but never changed the latent carrier
from which later predictions were generated.

Stage 43 tests the narrower missing mechanism.  At a recorded contact event it
fits a reset of the recursive predictive state itself,

`z_plus = z_proposed + R(z_proposed, action, event_metadata)`,

and all later predictions start from `z_plus`.  The primary operator is a
tensor-product ridge map containing state, metadata, and state-by-metadata
interactions.  A matched smooth-sham tensor map keeps the exact event gate but
replaces its six coordinate values with a frozen smooth projection; it tests
whether the oracle coordinates add information beyond event timing.  A
nonrecursive tensor output correction separates operator nonlinearity from
carrier feedback; and a parameter-matched MLP tests whether the tensor
parameterization is too restrictive.

The information ladder is nested:

1. the current proposed predictive carrier and action;
2. the current carrier plus two previous recursive carriers; and
3. the current carrier plus exact pre-macro physical state, used only as a
   privileged oracle ceiling.

This gives a falsifiable decision.  Success with the current carrier identifies
an operator/recursion defect.  Success only with history identifies a missing
memory coordinate.  Success only with exact physical state identifies an
insufficient frozen carrier.  Failure of the physical oracle rejects the reset
hypothesis for this finite bank and sends the project back to within-mode flow
and state abstraction.

The experiment uses new action words and trajectory identifiers.  Before any
checkpoint is loaded, exact simulator contact incidence selects the earliest
48 complete evaluation families with positive re-entry support.  Model errors,
reset outcomes, and physical-effect magnitudes are forbidden during support
selection.  Construction, model-selection, calibration, and evaluation remain
disjoint; penalties and MLP weight decay are selected on model-selection data;
all artifacts are refit and hashed before evaluation opens.

The reset is saltation-motivated but this notebook does **not** claim to recover
the mathematical saltation matrix.  It tests finite state-dependent jump
operators.  A dedicated local-perturbation saltation experiment is authorized
only if recursive headroom is first established.  No causal, planning,
minimal-state, Koopman, or deployment claim is authorized here.  Planning is
sealed.
'''


configuration = rename(BASE.configuration)
configuration = configuration.replace("STAGE43_V1_SUPPORT_AUDIT", "STAGE42_V1_SUPPORT_AUDIT")
configuration = configuration.replace(
    "stage43_v1_zero_of_16000_development_support",
    "stage42_v1_zero_of_16000_development_support",
)
for name, value in {
    "PROTOCOL_ID": '"stage43-recursive-reset-sufficiency-v1"',
    "NOTEBOOK_PROTOCOL_SHA256": '"__PROTOCOL_DIGEST__"',
    "EVIDENCE_STATUS": '"FRESH_RECURSIVE_RESET_SUFFICIENCY"',
    "EXPERIMENT_NOTEBOOK_PATH": '"notebooks/43_recursive_reset_sufficiency.ipynb"',
    "EXPERIMENT_BUILDER_PATH": '"notebooks/build_stage43_recursive_reset_sufficiency_notebook.py"',
    "EXPERIMENT_NUMERICAL_PATH": '"src/cf_faithfulness/stage43_recursive_reset.py"',
    "OUTPUT_DIR": '"/content/counterfactual_faithfulness_stage43_rrsl"',
    "DRIVE_OUTPUT_DIR": '"/content/drive/MyDrive/counterfactual_faithfulness_stage43_rrsl"',
    "RUN_REQUEST_PATH": '"/content/drive/MyDrive/counterfactual_faithfulness_stage43_rrsl/stage43_run_request.json"',
    "SEED": "430101",
    "DESIGN_SEED": "430141",
    "DECODER_SEED": "430183",
    "RANK_SEED": "430213",
    "CALIBRATION_SEED": "430253",
    "BOOTSTRAP_SEED": "430283",
    "CONTROL_SEED": "430351",
    "CONSTRUCTION_TRAJECTORY_POOL": "list(range(170000, 172000))",
    "MODEL_SELECTION_TRAJECTORY_POOL": "list(range(172000, 174000))",
    "CALIBRATION_TRAJECTORY_POOL": "list(range(174000, 176000))",
    "EVALUATION_TRAJECTORY_POOL": "list(range(176000, 192000))",
    "TASK_ID_OFFSET": "430000",
}.items():
    configuration = replace_assignment(configuration, name, value)
configuration = replace_assignment(
    configuration, "FINAL_TRAINING_SEEDS",
    '[4301, 4302, 4303] if RUN_MODE == "pilot" else [4301]',
)
configuration = replace_block(
    configuration,
    "CANONICAL_RESPONSE_WORD_NAMES = [",
    "CALIBRATION_INTERCHANGE_PAIRS =",
    r'''CANONICAL_RESPONSE_WORD_NAMES = ["A", "B", "C", "D", "AB", "CD", "BA", "DC"]
CONSTRUCTION_WORD_NAMES = [
    "QPPSSSPPQ", "PQPSSSQPP",
    "QPPSSS0PPQ", "PQPSSS0QPP",
    "QPPSSS00PPQ", "PQPSSS00QPP",
    "QPPSSS000PPQ", "PQPSSS000QPP",
]
MODEL_SELECTION_WORD_NAMES = [
    "LQPSSSPPQ", "TQPSSSPPQ",
    "LQPSSS0PPQ", "TQPSSS0PPQ",
    "LQPSSS00PPQ", "TQPSSS00PPQ",
    "LQPSSS000PPQ", "TQPSSS000PPQ",
]
CALIBRATION_WORD_NAMES = [
    "QLPSSSPPQ", "QTPSSSPPQ",
    "QLPSSS0PPQ", "QTPSSS0PPQ",
    "QLPSSS00PPQ", "QTPSSS00PPQ",
    "QLPSSS000PPQ", "QTPSSS000PPQ",
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
    "LPQSSSQPP", "TPQSSSQPP",
    "LPQSSS0QPP", "TPQSSS0QPP",
    "LPQSSS00QPP", "TPQSSS00QPP",
    "LPQSSS000QPP", "TPQSSS000QPP",
]
PLANNING_WORD_NAMES = []
EVALUATION_WORD_NAMES_REGISTERED = list(CLOSURE_EVALUATION_WORD_NAMES)
EVALUATION_WORD_SPECS = [
    stage39_word_spec(name) for name in EVALUATION_WORD_NAMES_REGISTERED
]
''',
)
configuration = configuration.replace(
    '"fresh_trajectory_ids_148000_to_169999",',
    '"fresh_trajectory_ids_170000_to_191999",\n'
    '    "fresh_stage43_action_words",\n'
    '    "stage42_negative_run_34cbe4d0760c_retained",',
)
configuration = re.sub(r"^PROTOCOL_CONFIG_KEYS = \[.*\]\n?", "", configuration, flags=re.M)
configuration += r'''

RESET_VARIANTS = [
    "affine_output_control",
    "current_nonrecursive_tensor",
    "sham_recursive_tensor",
    "current_recursive_tensor",
    "current_recursive_mlp",
    "history_recursive_tensor",
    "physical_recursive_tensor",
]
RESET_VARIANT_SPECS = {
    "current_nonrecursive_tensor": {
        "representation": "current", "operator": "tensor",
        "metadata_mode": "oracle", "target": "physical", "recursive": False,
    },
    "sham_recursive_tensor": {
        "representation": "current", "operator": "tensor",
        "metadata_mode": "sham", "target": "latent", "recursive": True,
    },
    "current_recursive_tensor": {
        "representation": "current", "operator": "tensor",
        "metadata_mode": "oracle", "target": "latent", "recursive": True,
    },
    "current_recursive_mlp": {
        "representation": "current", "operator": "mlp",
        "metadata_mode": "oracle", "target": "latent", "recursive": True,
    },
    "history_recursive_tensor": {
        "representation": "history", "operator": "tensor",
        "metadata_mode": "oracle", "target": "latent", "recursive": True,
    },
    "physical_recursive_tensor": {
        "representation": "physical_oracle", "operator": "tensor",
        "metadata_mode": "oracle", "target": "latent", "recursive": True,
    },
}
RESET_HISTORY_LAGS = 3
RESET_TENSOR_STATE_RANK = 32
RESET_RIDGE_PENALTIES = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
RESET_MLP_HIDDEN = 112
RESET_MLP_WEIGHT_DECAYS = [1e-5, 1e-4, 1e-3]
RESET_MLP_LEARNING_RATE = 1e-3
RESET_MLP_SELECTION_EPOCHS = 120
RESET_MLP_FINAL_EPOCHS = 200
RESET_DELTA_NORM_QUANTILE = 0.99
ACTIVE_RESET_MLP_SELECTION_EPOCHS = (
    8 if RUN_MODE == "smoke" else RESET_MLP_SELECTION_EPOCHS
)
ACTIVE_RESET_MLP_FINAL_EPOCHS = (
    12 if RUN_MODE == "smoke" else RESET_MLP_FINAL_EPOCHS
)
MAX_CURRENT_OPERATOR_PARAMETER_RATIO = 1.50
PRIMARY_RESET_VARIANT = "current_recursive_tensor"
MIN_CONTACT_TAIL_RELATIVE_IMPROVEMENT = 0.25
MIN_P95_RELATIVE_IMPROVEMENT = 0.10
MAX_MEAN_RELATIVE_DEGRADATION = 0.02
MIN_LOO_CONTACT_TAIL_RELATIVE_IMPROVEMENT = 0.10
STAGE42_NEGATIVE_AUDIT = {
    "protocol_id": "stage42-action-conditioned-hybrid-defect-v2",
    "run_signature": "34cbe4d0760cdc650ff883001f44ab09ab403a9003075c33881dfc578dbc8e3e",
    "status": "no_oracle_reset_headroom",
    "evaluation_opened": True,
    "planning_opened": False,
    "role": "prior_negative_evidence_only",
}
assert PRIMARY_RESET_VARIANT == "current_recursive_tensor"
assert RESET_HISTORY_LAGS == 3
assert RESET_TENSOR_STATE_RANK == 32
assert RESET_DELTA_NORM_QUANTILE == 0.99
assert MIN_REENTRY_ROWS == 32
assert PLANNING_WORD_NAMES == []
'''
configuration += "\n\nPROTOCOL_CONFIG_KEYS = " + repr(
    assigned_uppercase_names(configuration)
) + "\n"


installation = BASE.installation
setup = rename(BASE.setup).replace("stage43_ecoh_v2", "stage43_rrsl")
analysis_helpers = rename(BASE.analysis_helpers)
extra_helpers = function_sources(
    NUMERICAL.read_text(),
    [
        "_finite_matrix", "lagged_state_features", "preceding_physical_state",
        "reset_base_tensor", "fixed_state_projection", "tensor_reset_design",
        "clip_row_norms",
        "reset_risk_metrics",
        "fit_ridge", "ridge_predict", "select_ridge_penalty",
        "passes_registered_reset_gates", "Stage43Decision",
        "derive_stage43_decision",
    ],
)
extra_helpers = extra_helpers.replace(
    "class Stage43Decision:\n", "@dataclass(frozen=True)\nclass Stage43Decision:\n"
)
analysis_helpers += "\n\n" + extra_helpers

model_helpers = BASE.model_helpers
design_and_runtime_helpers = BASE.design_and_runtime_helpers
physical_truth = rename(BASE.physical_truth)
simulator_preflight = BASE.simulator_preflight
construction_and_paths = BASE.construction_and_paths
data_and_selection = rename(BASE.data_and_selection)
causal_interventions = rename(BASE.causal_interventions)


reset_helpers = r'''

def stage43_reset_bundle(short, seed, split, representation, target_kind):
    data = load_stage39_sequences(short, split)
    rollout = rollout_predictive_state_closure(
        FROZEN_MODELS[short][int(seed)], data["initial_carrier"], data["actions"],
        data["carrier"], data["mask"],
    )
    pairs = load_stage43_pairs(data, split)
    valid = np.asarray(rollout["evaluation_mask"], dtype=bool)
    events = valid & (np.asarray(pairs["metadata"][:, :, 0]) > 0.5)
    if int(np.sum(events)) < MIN_REENTRY_ROWS:
        raise RuntimeError(f"too few reset-training events in {split}")
    base_tensor = reset_base_tensor(
        rollout["state"], data["actions"], valid,
        representation=str(representation), history_lags=RESET_HISTORY_LAGS,
        initial_physical=data["initial_physical"], physical_path=data["simulator"],
    )
    if str(target_kind) == "latent":
        target = (
            (rollout["direct_state"] - rollout["state"])
            / STATE_SCALES[short][int(seed)]
        )
    elif str(target_kind) == "physical":
        target = (
            (data["simulator"] - rollout["physical"])
            / PHYSICAL_SCALES[short]
        )
    else:
        raise KeyError(f"unknown reset target {target_kind!r}")
    groups = np.repeat(data["group"][:, None], data["actions"].shape[1], axis=1)
    return {
        "base": np.asarray(base_tensor[events], dtype=np.float64),
        "metadata": np.asarray(pairs["metadata"][events], dtype=np.float64),
        "target": np.asarray(target[events], dtype=np.float64),
        "groups": np.asarray(groups[events], dtype=np.int64),
        "event_rows": int(np.sum(events)),
    }


def concatenate_stage43_reset_bundles(*bundles):
    return {
        key: np.concatenate([bundle[key] for bundle in bundles], axis=0)
        for key in ["base", "metadata", "target", "groups"]
    }


def stage43_tensor_design(bundle, artifact):
    return tensor_reset_design(
        bundle["base"], bundle["metadata"],
        base_mean=artifact["base_mean"], base_scale=artifact["base_scale"],
        metadata_mean=artifact["metadata_mean"],
        metadata_scale=artifact["metadata_scale"],
        metadata_mode=artifact["metadata_mode"],
        sham_projection={
            "weight": artifact["sham_weight"], "bias": artifact["sham_bias"],
        },
        state_projection={
            "weight": artifact["state_projection_weight"],
            "bias": artifact["state_projection_bias"],
        },
    )


def fit_stage43_tensor_reset(short, seed, variant, construction, validation, final_fit):
    spec = RESET_VARIANT_SPECS[variant]
    selection_seed = stable_seed(CALIBRATION_SEED, short, int(seed), variant)
    base_mean, base_scale = mean_scale(construction["base"])
    metadata_mean, metadata_scale = mean_scale(construction["metadata"])
    sham = fixed_sham_projection(construction["base"].shape[1], 6, selection_seed)
    state_projection = fixed_state_projection(
        construction["base"].shape[1], RESET_TENSOR_STATE_RANK, selection_seed + 1
    )
    selection_artifact = {
        "base_mean": base_mean, "base_scale": base_scale,
        "metadata_mean": metadata_mean, "metadata_scale": metadata_scale,
        "metadata_mode": spec["metadata_mode"],
        "sham_weight": sham["weight"], "sham_bias": sham["bias"],
        "state_projection_weight": state_projection["weight"],
        "state_projection_bias": state_projection["bias"],
    }
    train_design = stage43_tensor_design(construction, selection_artifact)
    validation_design = stage43_tensor_design(validation, selection_artifact)
    selection = select_ridge_penalty(
        train_design, construction["target"], validation_design,
        validation["target"], RESET_RIDGE_PENALTIES,
    )
    final_base_mean, final_base_scale = mean_scale(final_fit["base"])
    final_metadata_mean, final_metadata_scale = mean_scale(final_fit["metadata"])
    final_sham = fixed_sham_projection(final_fit["base"].shape[1], 6, selection_seed)
    final_state_projection = fixed_state_projection(
        final_fit["base"].shape[1], RESET_TENSOR_STATE_RANK, selection_seed + 1
    )
    artifact = {
        "variant": variant, "operator": "tensor", "model": short,
        "seed": int(seed), "representation": spec["representation"],
        "target_kind": spec["target"], "recursive": bool(spec["recursive"]),
        "metadata_mode": spec["metadata_mode"],
        "base_mean": final_base_mean, "base_scale": final_base_scale,
        "metadata_mean": final_metadata_mean,
        "metadata_scale": final_metadata_scale,
        "sham_weight": final_sham["weight"], "sham_bias": final_sham["bias"],
        "state_projection_weight": final_state_projection["weight"],
        "state_projection_bias": final_state_projection["bias"],
        "state_projection_rank": int(RESET_TENSOR_STATE_RANK),
        "selected_penalty": float(selection["selected_penalty"]),
        "selection_rows": selection["candidate_rows"],
        "delta_norm_cap": max(float(np.quantile(
            np.linalg.norm(final_fit["target"], axis=1),
            RESET_DELTA_NORM_QUANTILE,
        )), 1e-8),
        "delta_norm_quantile": float(RESET_DELTA_NORM_QUANTILE),
        "evaluation_rows_used": 0,
    }
    final_design = stage43_tensor_design(final_fit, artifact)
    fitted = fit_ridge(final_design, final_fit["target"], artifact["selected_penalty"])
    artifact["weight"] = fitted["weight"]
    artifact["intercept"] = fitted["intercept"]
    artifact["design_width"] = int(final_design.shape[1])
    artifact["parameter_count"] = int(
        fitted["weight"].size + fitted["intercept"].size
    )
    return artifact


class Stage43ResetMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(int(input_dim), int(hidden_dim)),
            torch.nn.SiLU(),
            torch.nn.Linear(int(hidden_dim), int(output_dim)),
        )

    def forward(self, value):
        return self.network(value)


def stage43_mlp_input(bundle, artifact):
    base = (np.asarray(bundle["base"]) - artifact["base_mean"]) / artifact["base_scale"]
    metadata = (
        (np.asarray(bundle["metadata"]) - artifact["metadata_mean"])
        / artifact["metadata_scale"]
    )
    return np.concatenate([base, metadata], axis=1).astype(np.float32)


def train_stage43_mlp(x, y, *, weight_decay, epochs, seed):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Stage43ResetMLP(x.shape[1], RESET_MLP_HIDDEN, y.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=RESET_MLP_LEARNING_RATE,
        weight_decay=float(weight_decay),
    )
    inputs = torch.as_tensor(x, dtype=torch.float32, device=device)
    targets = torch.as_tensor(y, dtype=torch.float32, device=device)
    model.train()
    losses = []
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(inputs) - targets) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return model, losses


def fit_stage43_mlp_reset(short, seed, variant, construction, validation, final_fit):
    spec = RESET_VARIANT_SPECS[variant]
    base_mean, base_scale = mean_scale(construction["base"])
    metadata_mean, metadata_scale = mean_scale(construction["metadata"])
    selection_artifact = {
        "base_mean": base_mean, "base_scale": base_scale,
        "metadata_mean": metadata_mean, "metadata_scale": metadata_scale,
    }
    train_x = stage43_mlp_input(construction, selection_artifact)
    validation_x = stage43_mlp_input(validation, selection_artifact)
    rows = []
    for index, weight_decay in enumerate(RESET_MLP_WEIGHT_DECAYS):
        model, losses = train_stage43_mlp(
            train_x, construction["target"].astype(np.float32),
            weight_decay=weight_decay, epochs=ACTIVE_RESET_MLP_SELECTION_EPOCHS,
            seed=stable_seed(CALIBRATION_SEED, short, int(seed), variant, index),
        )
        device = next(model.parameters()).device
        model.eval()
        with torch.inference_mode():
            prediction = model(torch.as_tensor(validation_x, device=device)).cpu().numpy()
        rows.append({
            "weight_decay": float(weight_decay),
            "validation_mse": float(np.mean((prediction - validation["target"]) ** 2)),
            "training_loss_initial": float(losses[0]),
            "training_loss_final": float(losses[-1]),
        })
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    selected = min(rows, key=lambda row: (row["validation_mse"], row["weight_decay"]))
    final_base_mean, final_base_scale = mean_scale(final_fit["base"])
    final_metadata_mean, final_metadata_scale = mean_scale(final_fit["metadata"])
    artifact = {
        "variant": variant, "operator": "mlp", "model": short,
        "seed": int(seed), "representation": spec["representation"],
        "target_kind": spec["target"], "recursive": True,
        "metadata_mode": "oracle", "base_mean": final_base_mean,
        "base_scale": final_base_scale, "metadata_mean": final_metadata_mean,
        "metadata_scale": final_metadata_scale,
        "selected_weight_decay": float(selected["weight_decay"]),
        "selection_rows": rows, "hidden_dim": int(RESET_MLP_HIDDEN),
        "delta_norm_cap": max(float(np.quantile(
            np.linalg.norm(final_fit["target"], axis=1),
            RESET_DELTA_NORM_QUANTILE,
        )), 1e-8),
        "delta_norm_quantile": float(RESET_DELTA_NORM_QUANTILE),
        "evaluation_rows_used": 0,
    }
    final_x = stage43_mlp_input(final_fit, artifact)
    model, losses = train_stage43_mlp(
        final_x, final_fit["target"].astype(np.float32),
        weight_decay=artifact["selected_weight_decay"],
        epochs=ACTIVE_RESET_MLP_FINAL_EPOCHS,
        seed=stable_seed(CALIBRATION_SEED, short, int(seed), variant, "final"),
    )
    artifact["input_dim"] = int(final_x.shape[1])
    artifact["output_dim"] = int(final_fit["target"].shape[1])
    artifact["state_dict"] = {
        key: value.detach().cpu().numpy().astype(np.float32)
        for key, value in model.state_dict().items()
    }
    artifact["parameter_count"] = int(sum(value.numel() for value in model.parameters()))
    artifact["training_loss_initial"] = float(losses[0])
    artifact["training_loss_final"] = float(losses[-1])
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return artifact


def stage43_loaded_mlp(artifact, device):
    model = Stage43ResetMLP(
        artifact["input_dim"], artifact["hidden_dim"], artifact["output_dim"]
    ).to(device)
    model.load_state_dict({
        key: torch.as_tensor(value, dtype=torch.float32, device=device)
        for key, value in artifact["state_dict"].items()
    })
    model.eval()
    return model
'''


calibration = rename(BASE.calibration)
calibration = must_replace(
    calibration,
    "GUARD_PROBES = {}\nHEAD_SELECTION_ROWS = []\nGUARD_SELECTION_ROWS = []",
    "GUARD_PROBES = {}\nRESET_ARTIFACTS = {}\nHEAD_SELECTION_ROWS = []\n"
    "GUARD_SELECTION_ROWS = []\nRESET_SELECTION_ROWS = []",
)
calibration = must_replace(
    calibration,
    "\nif not PIPELINE_FAILED and SIMULATOR_PREFLIGHT_PASSED:",
    reset_helpers + "\n\nif not PIPELINE_FAILED and SIMULATOR_PREFLIGHT_PASSED:",
)
reset_fit_block = r'''
        # Fit the Stage 43 state-by-operator lattice after every baseline,
        # affine control, scale, and guard artifact is frozen on development data.
        for short in ["jepa", "dino"]:
            RESET_ARTIFACTS[short] = {}
            for seed in FINAL_TRAINING_SEEDS:
                RESET_ARTIFACTS[short][int(seed)] = {}
                bundle_cache = {}
                for variant, spec in RESET_VARIANT_SPECS.items():
                    cache_key = (spec["representation"], spec["target"])
                    if cache_key not in bundle_cache:
                        construction_reset = stage43_reset_bundle(
                            short, seed, "construction", *cache_key
                        )
                        validation_reset = stage43_reset_bundle(
                            short, seed, "model_selection", *cache_key
                        )
                        calibration_reset = stage43_reset_bundle(
                            short, seed, "calibration", *cache_key
                        )
                        final_reset = concatenate_stage43_reset_bundles(
                            construction_reset, calibration_reset
                        )
                        bundle_cache[cache_key] = (
                            construction_reset, validation_reset, final_reset
                        )
                    construction_reset, validation_reset, final_reset = bundle_cache[cache_key]
                    array_path, schema_path = stage43_artifact_paths(
                        short, f"reset_{variant}", seed
                    )
                    required = [
                        array_path, schema_path, Path(str(array_path) + ".sha256"),
                        Path(str(schema_path) + ".sha256"),
                    ]
                    if all(path.is_file() for path in required):
                        artifact = load_stage43_artifact(short, f"reset_{variant}", seed)
                    elif spec["operator"] == "tensor":
                        artifact = fit_stage43_tensor_reset(
                            short, int(seed), variant, construction_reset,
                            validation_reset, final_reset,
                        )
                        save_stage43_artifact(short, f"reset_{variant}", seed, artifact)
                    else:
                        artifact = fit_stage43_mlp_reset(
                            short, int(seed), variant, construction_reset,
                            validation_reset, final_reset,
                        )
                        save_stage43_artifact(short, f"reset_{variant}", seed, artifact)
                    if int(artifact["evaluation_rows_used"]) != 0:
                        raise RuntimeError("Stage 43 reset artifact consumed evaluation rows")
                    RESET_ARTIFACTS[short][int(seed)][variant] = artifact
                    selected_value = (
                        artifact.get("selected_penalty")
                        if artifact["operator"] == "tensor"
                        else artifact.get("selected_weight_decay")
                    )
                    for row in artifact["selection_rows"]:
                        candidate_value = row.get("penalty", row.get("weight_decay"))
                        RESET_SELECTION_ROWS.append({
                            "model": short, "seed": int(seed), "variant": variant,
                            "candidate_regularization": float(candidate_value),
                            "validation_mse": float(row["validation_mse"]),
                            "training_loss_initial": row.get("training_loss_initial"),
                            "training_loss_final": row.get("training_loss_final"),
                            "selected": bool(float(candidate_value) == float(selected_value)),
                        })
                    model_manifest.append({
                        "model": short, "seed": int(seed),
                        "variant": f"reset_{variant}",
                        "array_sha256": sha256_file(array_path),
                        "schema_sha256": sha256_file(schema_path),
                        "parameter_count": int(artifact["parameter_count"]),
                    })
                tensor_count = RESET_ARTIFACTS[short][int(seed)][
                    "current_recursive_tensor"
                ]["parameter_count"]
                mlp_count = RESET_ARTIFACTS[short][int(seed)][
                    "current_recursive_mlp"
                ]["parameter_count"]
                ratio = max(tensor_count, mlp_count) / max(min(tensor_count, mlp_count), 1)
                if ratio > MAX_CURRENT_OPERATOR_PARAMETER_RATIO:
                    raise RuntimeError(
                        f"current tensor/MLP parameter ratio {ratio:.3f} exceeds cap"
                    )
        write_csv(EVIDENCE_DIR / "stage43_reset_selection_rows.csv", RESET_SELECTION_ROWS)
'''
calibration = must_replace(
    calibration,
    '        write_csv(EVIDENCE_DIR / "stage43_head_selection_rows.csv", HEAD_SELECTION_ROWS)',
    reset_fit_block
    + '        write_csv(EVIDENCE_DIR / "stage43_head_selection_rows.csv", HEAD_SELECTION_ROWS)',
)
calibration = must_replace(
    calibration,
    '"equal_nominal_head_width": True,',
    '"equal_nominal_head_width": True,\n'
    '            "recursive_reset_lattice_frozen": True,\n'
    '            "reset_variants": RESET_VARIANTS,\n'
    '            "reset_selection_uses_model_selection_only": True,',
)


heldout_evaluation = r'''# Open the held-out recursive reset sufficiency panel after every artifact is frozen.
DECISION_PAYLOAD = {
    "status": "INCONCLUSIVE_PIPELINE_FAILURE", "passed": False,
    "planning_opened": False, "causal_claim_authorized": False,
}
EVALUATION_ROWS = []
SUMMARY = {}
STAGE43_SUPPORT_CERTIFICATE = None


def stage43_terminal(values, mask):
    labels = np.asarray(values).astype(str)
    valid = np.asarray(mask, dtype=bool)
    result = []
    for row in range(len(labels)):
        indices = np.flatnonzero(valid[row])
        result.append(labels[row, indices[-1]] if len(indices) else "invalid")
    return np.asarray(result)


def stage43_affine_output_prediction(short, seed, data, rollout, pairs):
    valid = np.asarray(rollout["evaluation_mask"], dtype=bool)
    initial = np.repeat(
        data["initial_physical"][:, None, :], data["actions"].shape[1], axis=1
    )
    base = np.concatenate([
        rollout["physical"], rollout["state"], data["actions"], initial,
    ], axis=-1)
    artifact = HEADS[short][int(seed)]["oracle_reset_ceiling"]
    flat = {"base": base[valid], "metadata": pairs["metadata"][valid]}
    design = stage43_design(
        flat, artifact, "oracle_reset_ceiling",
        stable_seed(CONTROL_SEED, "stage43_affine", short, int(seed)),
    )
    prediction = np.asarray(rollout["physical"], dtype=np.float64).copy()
    prediction[valid] += ridge_predict(artifact, design) * PHYSICAL_SCALES[short]
    return prediction


def stage43_reset_delta(artifact, base, metadata, device, mlp=None):
    bundle = {
        "base": np.asarray(base, dtype=np.float64),
        "metadata": np.asarray(metadata, dtype=np.float64),
    }
    if artifact["operator"] == "tensor":
        prediction = ridge_predict(artifact, stage43_tensor_design(bundle, artifact))
    elif artifact["operator"] == "mlp":
        model = mlp if mlp is not None else stage43_loaded_mlp(artifact, device)
        inputs = torch.as_tensor(stage43_mlp_input(bundle, artifact), device=device)
        with torch.inference_mode():
            prediction = model(inputs).cpu().numpy()
    else:
        raise KeyError(f"unknown reset operator {artifact['operator']!r}")
    return clip_row_norms(prediction, artifact["delta_norm_cap"])


def stage43_step_base(proposed, action, history, exact_prephysical, representation):
    if representation == "current":
        return np.concatenate([proposed, action], axis=1)
    if representation == "history":
        return np.concatenate([
            proposed, history[:, : RESET_HISTORY_LAGS - 1].reshape(len(proposed), -1),
            action,
        ], axis=1)
    if representation == "physical_oracle":
        return np.concatenate([proposed, action, exact_prephysical], axis=1)
    raise KeyError(f"unknown recursive representation {representation!r}")


def stage43_recursive_prediction(short, seed, data, pairs, variant):
    base_artifact = FROZEN_MODELS[short][int(seed)]
    reset = RESET_ARTIFACTS[short][int(seed)][variant]
    config = base_artifact["config"]
    normalization = base_artifact["normalization"]
    first = np.asarray(data["initial_carrier"], dtype=np.float32)
    action = np.asarray(data["actions"], dtype=np.float32)
    carrier = np.asarray(data["carrier"], dtype=np.float32)
    valid = np.asarray(data["mask"], dtype=bool)
    histories = history_tensor(first, carrier, valid, config["history_length"]).astype(np.float32)
    carrier_mean = np.asarray(normalization["carrier_mean"], dtype=np.float32)
    carrier_scale = np.asarray(normalization["carrier_scale"], dtype=np.float32)
    action_mean = np.asarray(normalization["action_mean"], dtype=np.float32)
    action_scale = np.asarray(normalization["action_scale"], dtype=np.float32)
    physical_mean = np.asarray(normalization["physical_mean"], dtype=np.float32)
    physical_scale = np.asarray(normalization["physical_scale"], dtype=np.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _artifact_model(base_artifact, device)
    history_n = torch.as_tensor((histories - carrier_mean) / carrier_scale, device=device)
    action_n = torch.as_tensor((action - action_mean) / action_scale, device=device)
    output = np.zeros(
        (*carrier.shape[:2], int(config["physical_dim"])), dtype=np.float32
    )
    start = int(config["history_length"]) - 1
    exact_prephysical = preceding_physical_state(
        data["initial_physical"], data["simulator"], valid
    )
    mlp = stage43_loaded_mlp(reset, device) if reset["operator"] == "mlp" else None
    with torch.inference_mode():
        teacher = model.encode(history_n)
        state = teacher[:, start]
        previous = np.repeat(
            state.detach().cpu().numpy()[:, None, :], RESET_HISTORY_LAGS - 1, axis=1
        )
        for step in range(start, action.shape[1]):
            active = valid[:, step]
            if not np.any(active):
                continue
            active_tensor = torch.as_tensor(active, device=device)
            proposed = model.transition(state[active_tensor], action_n[active_tensor, step])
            updated = proposed.clone()
            active_indices = np.flatnonzero(active)
            event_local = np.asarray(pairs["metadata"])[active, step, 0] > 0.5
            if np.any(event_local):
                proposed_np = proposed.detach().cpu().numpy()[event_local]
                base = stage43_step_base(
                    proposed_np, action[active, step][event_local],
                    previous[active_indices][event_local],
                    exact_prephysical[active, step][event_local],
                    reset["representation"],
                )
                metadata = np.asarray(pairs["metadata"])[active, step][event_local]
                normalized_delta = stage43_reset_delta(
                    reset, base, metadata, device, mlp=mlp
                )
                delta = normalized_delta * STATE_SCALES[short][int(seed)]
                event_tensor = torch.as_tensor(event_local, device=device)
                updated[event_tensor] += torch.as_tensor(
                    delta, dtype=updated.dtype, device=device
                )
            state = state.clone()
            state[active_tensor] = updated
            _, decoded = model.decode(updated)
            output[active, step] = (
                decoded.cpu().numpy() * physical_scale + physical_mean
            )
            updated_np = updated.detach().cpu().numpy()
            if previous.shape[1] > 1:
                previous[active_indices, 1:] = previous[active_indices, :-1]
            previous[active_indices, 0] = updated_np
    return output.astype(np.float64)


def stage43_nonrecursive_tensor_prediction(short, seed, data, rollout, pairs):
    artifact = RESET_ARTIFACTS[short][int(seed)]["current_nonrecursive_tensor"]
    valid = np.asarray(rollout["evaluation_mask"], dtype=bool)
    events = valid & (np.asarray(pairs["metadata"][:, :, 0]) > 0.5)
    base = reset_base_tensor(
        rollout["state"], data["actions"], valid, representation="current"
    )
    prediction = np.asarray(rollout["physical"], dtype=np.float64).copy()
    prediction[events] += stage43_reset_delta(
        artifact, base[events], pairs["metadata"][events],
        "cuda" if torch.cuda.is_available() else "cpu",
    ) * PHYSICAL_SCALES[short]
    return prediction


def stage43_dominates(candidate, *controls):
    return bool(all(
        candidate["tail_relative_gain"] > control["tail_relative_gain"]
        and candidate["p95_relative_gain"] > control["p95_relative_gain"]
        for control in controls
    ))


def stage43_panel(short, data, pairs):
    groups = np.asarray(data["group"], dtype=np.int64)
    source_terminal = stage43_terminal(data["source_mode"], data["mask"])
    target_terminal = stage43_terminal(data["target_mode"], data["mask"])
    reentry = (source_terminal == "post_contact") & (target_terminal == "contact")
    if int(np.sum(reentry)) != int(STAGE43_SUPPORT_CERTIFICATE["total_reentry_rows"]):
        raise RuntimeError("Stage 43 support count changed after certificate freeze")
    seed_summaries = []
    gate_rows = []
    for seed in FINAL_TRAINING_SEEDS:
        rollout = rollout_predictive_state_closure(
            FROZEN_MODELS[short][int(seed)], data["initial_carrier"], data["actions"],
            data["carrier"], data["mask"],
        )
        evaluated = np.asarray(rollout["evaluation_mask"], dtype=bool)
        predictions = {
            "baseline": np.asarray(rollout["physical"], dtype=np.float64),
            "affine_output_control": stage43_affine_output_prediction(
                short, seed, data, rollout, pairs
            ),
            "current_nonrecursive_tensor": stage43_nonrecursive_tensor_prediction(
                short, seed, data, rollout, pairs
            ),
        }
        for variant in [
            "sham_recursive_tensor", "current_recursive_tensor",
            "current_recursive_mlp", "history_recursive_tensor",
            "physical_recursive_tensor",
        ]:
            predictions[variant] = stage43_recursive_prediction(
                short, seed, data, pairs, variant
            )
        errors = {
            name: scaled_path_mse(
                prediction, data["simulator"], evaluated, PHYSICAL_SCALES[short]
            )
            for name, prediction in predictions.items()
        }
        metrics = {
            name: reset_risk_metrics(
                value, errors["baseline"], reentry, groups,
                tail_mass=CONTACT_TAIL_MASS, minimum_rows=MIN_REENTRY_ROWS,
            )
            for name, value in errors.items() if name != "baseline"
        }
        risk_pass = {
            name: passes_registered_reset_gates(
                value,
                minimum_tail_gain=MIN_CONTACT_TAIL_RELATIVE_IMPROVEMENT,
                minimum_p95_gain=MIN_P95_RELATIVE_IMPROVEMENT,
                maximum_mean_ratio=1.0 + MAX_MEAN_RELATIVE_DEGRADATION,
                minimum_leave_one_gain=MIN_LOO_CONTACT_TAIL_RELATIVE_IMPROVEMENT,
            )
            for name, value in metrics.items()
        }
        controls = [
            metrics["sham_recursive_tensor"],
            metrics["current_nonrecursive_tensor"],
        ]
        gates = {
            "current_tensor": bool(
                risk_pass["current_recursive_tensor"]
                and stage43_dominates(metrics["current_recursive_tensor"], *controls)
            ),
            "current_nonlinear": bool(
                risk_pass["current_recursive_mlp"]
                and stage43_dominates(metrics["current_recursive_mlp"], *controls)
            ),
            "history": bool(risk_pass["history_recursive_tensor"]),
            "physical_oracle": bool(risk_pass["physical_recursive_tensor"]),
        }
        gate_rows.append(gates)
        seed_summaries.append({
            "seed": int(seed), "reentry_rows": int(np.sum(reentry)),
            "metrics": metrics, "risk_pass": risk_pass, "gates": gates,
            "parameter_counts": {
                name: int(artifact["parameter_count"])
                for name, artifact in RESET_ARTIFACTS[short][int(seed)].items()
            },
        })
        for index in range(len(groups)):
            row = {
                "model": short, "seed": int(seed),
                "trajectory_id": int(groups[index]),
                "record_id": int(data["record_id"][index]),
                "initial_mode": str(data["initial_mode"][index]),
                "terminal_source_mode": str(source_terminal[index]),
                "terminal_target_mode": str(target_terminal[index]),
                "word": str(data["word"][index]),
            }
            for name, value in errors.items():
                row[f"{name}_nmse"] = float(value[index])
            EVALUATION_ROWS.append(row)
    return {
        "seed_summaries": seed_summaries,
        "all_seed_current_tensor": bool(all(row["current_tensor"] for row in gate_rows)),
        "all_seed_current_nonlinear": bool(all(row["current_nonlinear"] for row in gate_rows)),
        "all_seed_history": bool(all(row["history"] for row in gate_rows)),
        "all_seed_physical_oracle": bool(all(row["physical_oracle"] for row in gate_rows)),
        "panels_pooled": False,
    }


if not PIPELINE_FAILED and SIMULATOR_PREFLIGHT_PASSED:
    try:
        verify_executed_notebook_through(
            "# Open the held-out recursive reset sufficiency panel after every artifact is frozen."
        )
        certificate_path = CALIBRATION_MODEL_DIR / "evaluation_open_certificate.json"
        support_path = DESIGN_DIR / "stage43_event_support_certificate.json"
        validate_digest_sidecar(certificate_path)
        validate_digest_sidecar(support_path)
        certificate = json.loads(certificate_path.read_text())
        STAGE43_SUPPORT_CERTIFICATE = json.loads(support_path.read_text())
        support_decision = derive_stage43_support_decision(
            STAGE43_SUPPORT_CERTIFICATE,
            expected_families=ACTIVE_EVALUATION_TRAJECTORIES,
            minimum_reentry_rows=MIN_REENTRY_ROWS,
        )
        if not support_decision.passed:
            raise RuntimeError(support_decision.classification)
        if (
            certificate["protocol_id"] != PROTOCOL_ID
            or certificate["run_signature"] != RUN_SIGNATURE
            or certificate["evaluation_statistics_read"]
            or certificate["evaluation_pairs_generated"]
            or not certificate["planning_permanently_sealed"]
            or not certificate["recursive_reset_lattice_frozen"]
        ):
            raise RuntimeError("Stage 43 evaluation-open certificate is invalid")
        stage43_phase("heldout_pairs_start")
        for index, record in enumerate(SELECTED_RECORDS["evaluation"]):
            generate_stage43_paired_record(record, "evaluation_closure")
            write_json(OUT / "paired_evaluation_progress.json", {
                "completed": index + 1, "total": len(SELECTED_RECORDS["evaluation"]),
                "last_record_id": int(record["record_id"]),
            })
        stage43_phase("heldout_pairs_complete")
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
            pairs = load_stage43_pairs(data, "evaluation_closure")
            SUMMARY[short] = stage43_panel(short, data, pairs)
        decision = derive_stage43_decision(
            support_certified=True,
            current_tensor_headroom=bool(all(
                SUMMARY[short]["all_seed_current_tensor"] for short in ["jepa", "dino"]
            )),
            current_nonlinear_headroom=bool(all(
                SUMMARY[short]["all_seed_current_nonlinear"] for short in ["jepa", "dino"]
            )),
            history_headroom=bool(all(
                SUMMARY[short]["all_seed_history"] for short in ["jepa", "dino"]
            )),
            physical_oracle_headroom=bool(all(
                SUMMARY[short]["all_seed_physical_oracle"] for short in ["jepa", "dino"]
            )),
        )
        DECISION_PAYLOAD = {
            "status": decision.classification,
            "next_step": decision.classification,
            "passed": bool(decision.passed),
            "support_certified": bool(decision.support_certified),
            "current_tensor_headroom": bool(decision.current_tensor_headroom),
            "current_nonlinear_headroom": bool(decision.current_nonlinear_headroom),
            "history_headroom": bool(decision.history_headroom),
            "physical_oracle_headroom": bool(decision.physical_oracle_headroom),
            "learned_recursive_reset_authorized": bool(
                decision.learned_recursive_reset_authorized
            ),
            "protocol_id": PROTOCOL_ID, "run_signature": RUN_SIGNATURE,
            "evidence_tier": "fresh_recursive_reset_sufficiency",
            "evaluation_opened": True, "planning_opened": False,
            "causal_claim_authorized": False,
            "deployment_claim_authorized": False,
            "saltation_matrix_identified": False,
            "stage42_negative_result_retained": True,
            "support_certificate_sha256": sha256_file(support_path),
        }
        write_csv(EVIDENCE_DIR / "heldout_stage43_rows.csv", EVALUATION_ROWS)
        write_json(EVIDENCE_DIR / "stage43_summary.json", SUMMARY)
        write_json(OUT / "stage43_decision.json", DECISION_PAYLOAD)
        atomic_checkpoint("stage43_recursive_reset_complete", {
            "decision_sha256": sha256_file(OUT / "stage43_decision.json"),
            "status": DECISION_PAYLOAD["status"], "planning_opened": False,
        })
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        plotted = [
            "affine_output_control", "current_nonrecursive_tensor",
            "sham_recursive_tensor", "current_recursive_tensor",
            "current_recursive_mlp", "history_recursive_tensor",
            "physical_recursive_tensor",
        ]
        for axis, short in zip(axes, ["jepa", "dino"]):
            values = [
                np.mean([
                    row["metrics"][name]["tail_relative_gain"]
                    for row in SUMMARY[short]["seed_summaries"]
                ])
                for name in plotted
            ]
            axis.bar(np.arange(len(plotted)), values)
            axis.axhline(MIN_CONTACT_TAIL_RELATIVE_IMPROVEMENT, color="black", ls="--")
            axis.set_xticks(np.arange(len(plotted)), plotted, rotation=70, ha="right")
            axis.set_ylabel("mean seed re-entry tail gain")
            axis.set_title(short.upper())
        figure.suptitle(f"Stage 43: {DECISION_PAYLOAD['status']}")
        figure.tight_layout()
        figure.savefig(PLOT_DIR / "stage43_recursive_reset_lattice.png", dpi=180)
        plt.close(figure)
        interpretation = f"""# Automatic interpretation

Status: `{DECISION_PAYLOAD['status']}`.

This prospective state-by-operator lattice used exact event metadata only as
oracle headroom information.  A current-carrier success authorizes a separate
label-free reset experiment.  History-only or physical-only success diagnoses
state insufficiency but does not authorize deployment.  Failure of the physical
oracle rejects reset repair for this finite bank.  Planning remained sealed.
"""
        (OUT / "AUTOMATIC_INTERPRETATION.md").write_text(interpretation)
        stage43_phase("recursive_reset_complete", status=DECISION_PAYLOAD["status"])
    except Exception:
        record_failure("stage43_heldout_recursive_reset")
'''


packaging = rename(BASE.packaging).replace("stage43_ecoh_v2", "stage43_rrsl")
packaging = packaging.replace("event_conditioned_oracle_hybrid", "recursive_reset_sufficiency")


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
    raise RuntimeError("Stage 43 protocol digest placeholder was not replaced")
protocol_sources[1] = configuration

cells = [markdown(introduction)] + [code(value) for value in protocol_sources[1:]]
for index, cell in enumerate(cells):
    cell["id"] = f"stage43-{index:02d}"
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
