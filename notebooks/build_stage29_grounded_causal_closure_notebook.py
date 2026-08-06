import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "29_grounded_causal_closure.ipynb"
NUMERICAL = ROOT.parent / "src/cf_faithfulness/stage29_grounded_closure.py"

spec = importlib.util.spec_from_file_location(
    "stage28_builder", ROOT / "build_stage28_hybrid_control_area_notebook.py"
)
STAGE28 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(STAGE28)

code = STAGE28.code
markdown = STAGE28.markdown
assigned_uppercase_names = STAGE28.assigned_uppercase_names
function_sources = STAGE28.function_sources


introduction = r'''# Stage 29: grounded causal closure

Stage 28 produced a sharp dissociation.  The PushT simulator obeyed a
contact-amplified signed-control-area law, while the JEPA-WM physical readout
did not reproduce that law.  Nevertheless, swapping the frozen Stage 18
rank-128 action carrier reliably reversed the model's own latent area contrast,
including in free-motion states where the simulator contrast was exactly zero.

That result has two possible explanations which Stage 28 could not separate:

1. the predictor implements a causally coherent but incorrect transition law;
2. the predictor is correct in JEPA latent space, but the frozen physical
   decoder cannot resolve the small schedule-dependent effect.

Stage 29 resolves the ambiguity without fitting a new reader.  It reuses the
exact source-bound Stage 28 states, actions, contact strata, and simulator
endpoint images.  For each action schedule it compares the predicted latent

\[
\hat z_\pi = F(E(o_t), a_\pi)
\]

with the frozen encoder representation of the exact simulator future

\[
z^*_\pi = E(o^{\mathrm{sim}}_{t+H,\pi}).
\]

These tensors share the model's native target-latent coordinates.  The primary
test compares their magnitude-centered, area-antisymmetric contrasts exactly
in the full 256-by-384 token space.  The frozen physical decoder is also
applied separately to predicted and encoded-true futures, localizing any
failure to encoder sensitivity, predictor dynamics, or readout.

Finally, the frozen rank-128 block-4 intervention is scored against two targets:
the opposite *model prediction* (self-consistent closure) and the opposite
*encoded simulator future* (grounded closure).  Shuffled, empirical-span
random, wrong-state, full-swap, and ablation controls remain frozen.  A patch
that follows the former but not the latter is a causally effective internal
mechanism that steers the model along its own counterfactual dynamics rather
than the environment's future.

No representation, decoder, subspace, state, action, threshold, or layer is
fit to Stage 29 outputs.  No Jacobian, JVP, VJP, gradient, or visual judgment is
used.  Return `stage29_grounded_closure_result_bundle_<signature>.zip`.
'''


configuration = r'''# SINGLE CONFIGURATION BLOCK — no Stage 29 secrets required.
# Run all on a GPU. The notebook creates a fresh nonce automatically and binds
# itself, Stage 18, and the exact successful Stage 28 Drive run before opening
# model activations. HF_TOKEN is read only if the pinned checkpoint cache needs it.
import secrets as _secrets
import time as _time

RUN_MODE = "pilot"
EXPERIMENT_SOURCE_REF = "codex/stage29-grounded-causal-closure"
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

OUTPUT_DIR = "/content/counterfactual_faithfulness_stage29_grounded_closure"
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/counterfactual_faithfulness_stage29_grounded_closure"
STAGE18_SEARCH_ROOT = "/content/drive/MyDrive"
STAGE28_SEARCH_ROOT = "/content/drive/MyDrive"

PROTOCOL_ID = "stage29-grounded-causal-closure-v1"
NOTEBOOK_PROTOCOL_SHA256 = "__PROTOCOL_DIGEST__"
EVIDENCE_STATUS = "CONFIRMATORY_DIAGNOSTIC_ONLY_IF_SOURCE_BOUND_FRESH_STAGE18_FROZEN_AND_EXACT_STAGE28_BOUND"
EXPERIMENT_REPOSITORY = "grewalsk/counterfactual-faithfulness-research"
EXPERIMENT_NOTEBOOK_PATH = "notebooks/29_grounded_causal_closure.ipynb"
EXPERIMENT_BUILDER_PATH = "notebooks/build_stage29_grounded_causal_closure_notebook.py"
EXPERIMENT_NUMERICAL_PATH = "src/cf_faithfulness/stage29_grounded_closure.py"

EXPECTED_STAGE18_SUBSPACE_SHA256 = "2f9c496d54623a9062e465a18c70039acc18cb8a1cc2833a5f4ade162ca3f90b"
EXPECTED_STAGE18_SOURCE_COMMIT = "16edd247cddcb1aa121340eb5fa42bd9e07004c3"
EXPECTED_STAGE18_STATUS = "CONFIRMED_BIDIRECTIONAL_RANK64_MEDIATOR"
EXPECTED_STAGE18_AMBIENT_DIMENSION = 102400
EXPECTED_STAGE18_MAX_RANK = 128
EXPECTED_STAGE28_STATUS = "MODEL_DOES_NOT_CAPTURE_PHYSICAL_CONTROL_AREA_LAW"
EXPECTED_STAGE28_SOURCE_COMMIT = "917228edb9e7143c58bdd9640afe08ead75fa34c"
EXPECTED_STAGE28_PROTOCOL_ID = "stage28-hybrid-control-area-law-v1"
EXPECTED_STAGE28_MAGNITUDES = [0.10, 0.14, 0.18, 0.22]
EXPECTED_STAGE28_RECORDS = 36
EXPECTED_STAGE28_COUNTS = {
    "persistent_contact": 12,
    "boundary_switching": 12,
    "free": 12,
}

SEED = 29101
BOOTSTRAP_SEED = 29269
MODEL_NAME = "jepa_wm_pusht"
ENVIRONMENT = "PushT"
FRAMESKIP = 5
PRIMARY_HORIZON = 3
TARGET_STEPS = [PRIMARY_HORIZON]
ACTION_STEPS = PRIMARY_HORIZON * FRAMESKIP
FIXED_BLOCK = 4
ACTIVE_BLOCKS = [FIXED_BLOCK]
EXPECTED_CARRIER_CHANNELS = 400
MAGNITUDE_COUNT = 4
SCHEDULE_COUNT = 6
ACTIONS_PER_STATE = MAGNITUDE_COUNT * SCHEDULE_COUNT
PRIMARY_RANK = 128
CAUSAL_RANDOM_DRAWS = 4
BOOTSTRAP_DRAWS = 10000
PILOT_INTERVENTION_FORWARDS_PER_RECORD = 9
SMOKE_INTERVENTION_FORWARDS_PER_RECORD = 6
MAX_ZERO_EDIT_ERROR = 1e-6

MIN_TARGET_AREA_MEAN_SQUARE = 1e-10
MIN_TARGET_PERSISTENT_TO_FREE_RATIO = 2.0
MIN_PREDICTOR_TARGET_AREA_COSINE = 0.10
MIN_PREDICTOR_GAIN_OVER_WRONG_STATE = 0.05
MIN_TRUE_TARGET_DECODER_AREA_COSINE = 0.10
MIN_PRIMARY_CLOSURE_COEFFICIENT = 0.15
MIN_PRIMARY_CLOSURE_COSINE = 0.15
MIN_PRIMARY_GAIN_OVER_RANDOM = 0.05
MIN_FULL_SWAP_SELF_COEFFICIENT = 0.75
REQUIRED_POSITIVE_PERSISTENT_RECORDS = 9

if RUN_MODE == "smoke":
    ACTIVE_RECORDS_PER_STRATUM = 1
    ACTIVE_CAUSAL_RANDOM_DRAWS = 1
    ACTIVE_BOOTSTRAP_DRAWS = 64
    ACTIVE_INTERVENTION_FORWARDS_PER_RECORD = SMOKE_INTERVENTION_FORWARDS_PER_RECORD
elif RUN_MODE == "pilot":
    ACTIVE_RECORDS_PER_STRATUM = 12
    ACTIVE_CAUSAL_RANDOM_DRAWS = CAUSAL_RANDOM_DRAWS
    ACTIVE_BOOTSTRAP_DRAWS = BOOTSTRAP_DRAWS
    ACTIVE_INTERVENTION_FORWARDS_PER_RECORD = PILOT_INTERVENTION_FORWARDS_PER_RECORD
else:
    raise ValueError("RUN_MODE must be 'smoke' or 'pilot'")

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
PINNED = [
    "model_weights", "decoder_weights", "stage18_frozen_subspace",
    "stage28_exact_source_bound_run", "stage28_selected_truth_shards",
    "stage28_contact_strata", "stage28_signed_area_schedules",
    "native_target_latent_metric", "frozen_rank128_grounded_intervention",
]

assert ACTION_STEPS == 15
assert ACTIONS_PER_STATE == 24
assert FIXED_BLOCK == 4 and PRIMARY_RANK == 128
assert EXPECTED_STAGE28_RECORDS == sum(EXPECTED_STAGE28_COUNTS.values())
assert PILOT_INTERVENTION_FORWARDS_PER_RECORD == 9
'''
configuration += "\n\nPROTOCOL_CONFIG_KEYS = " + repr(
    assigned_uppercase_names(configuration)
) + "\n"


installation = STAGE28.installation


setup = STAGE28.setup
setup = setup.replace("Stage 28", "Stage 29").replace("STAGE28", "STAGE29")
setup = setup.replace("stage28_control_area", "stage29_grounded_closure")
setup = setup.replace("stage28-source-binder", "stage29-source-binder")
setup = setup.replace(
    "stage28_hybrid_control_area_result_bundle_",
    "stage29_grounded_closure_result_bundle_",
)


analysis_helpers = STAGE28.analysis_helpers + "\n\n\n" + function_sources(
    NUMERICAL.read_text(),
    [
        "vector_alignment",
        "latent_closure_metrics",
        "ideal_contrast_effect",
        "ideal_absolute_target",
        "grounded_intervention_metrics",
    ],
)


model_helpers = STAGE28.model_helpers.replace(
    "stage28-jepa-wms", "stage29-jepa-wms"
).replace("Stage 28 supports PushT only", "Stage 29 supports PushT only")


artifact_import = r'''# Bind exact Stage 28 truth and the frozen Stage 18 carrier before model use.
STAGE18_ARTIFACT_VALIDATED = False
STAGE28_UPSTREAM_BOUND = False


def unique_matching_path(candidates, expected_hash=None):
    existing = sorted({Path(value) for value in candidates if Path(value).is_file()})
    if expected_hash is not None:
        existing = [value for value in existing if sha256_file(value) == expected_hash]
    if not existing:
        raise FileNotFoundError("no matching frozen upstream artifact was found in MyDrive")
    existing.sort(key=lambda value: (len(str(value)), str(value)))
    return existing[0]


def read_csv_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def branch_path(record_id):
    if not STAGE28_UPSTREAM_BOUND:
        raise RuntimeError("Stage 28 truth requested before upstream validation")
    return STAGE28_TRUTH_DIR / f"state_{int(record_id):06d}.npz"


if not PIPELINE_FAILED:
    try:
        verify_executed_notebook_through(
            "# Bind exact Stage 28 truth and the frozen Stage 18 carrier before model use."
        )

        stage18_root = Path(STAGE18_SEARCH_ROOT)
        stage18_candidates = list(stage18_root.glob(
            "counterfactual_faithfulness_stage18_rank64/pilot_*/subspaces/frozen_rank64_confirmation_subspaces.npz"
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
        if (
            stage18_decision.get("status") != EXPECTED_STAGE18_STATUS
            or not stage18_decision.get("confirmation_eligible", False)
        ):
            raise RuntimeError("Stage 18 decision is not the frozen confirmation")
        if stage18_manifest.get("subspace_sha256") != EXPECTED_STAGE18_SUBSPACE_SHA256:
            raise RuntimeError("Stage 18 manifest does not bind the required subspace")
        if (
            stage18_source.get("resolved_commit") != EXPECTED_STAGE18_SOURCE_COMMIT
            or not stage18_source.get("confirmation_eligible", False)
        ):
            raise RuntimeError("Stage 18 source binding mismatch")
        with np.load(FROZEN_SUBSPACE_PATH) as payload:
            FROZEN_SUBSPACES = {name: payload[name].copy() for name in payload.files}
        artifact_contract = validate_stage18_subspace_arrays(
            FROZEN_SUBSPACES,
            ambient=EXPECTED_STAGE18_AMBIENT_DIMENSION,
            max_rank=EXPECTED_STAGE18_MAX_RANK,
        )
        STAGE18_ARTIFACT_CERTIFICATE = {
            "validated_before_stage29_model_activations": True,
            "path": str(FROZEN_SUBSPACE_PATH),
            "bytes": int(FROZEN_SUBSPACE_PATH.stat().st_size),
            "sha256": sha256_file(FROZEN_SUBSPACE_PATH),
            "artifact_contract": artifact_contract,
            "stage29_subspace_refit": False,
            "stage29_basis_rotation_or_tuning": False,
        }
        write_json(OUT / "stage18_artifact_certificate.json", STAGE18_ARTIFACT_CERTIFICATE)
        STAGE18_ARTIFACT_VALIDATED = True

        stage28_root = Path(STAGE28_SEARCH_ROOT)
        decision_candidates = list(stage28_root.glob(
            "counterfactual_faithfulness_stage28_control_area/pilot_*/stage28_decision.json"
        ))
        valid_stage28 = []
        for decision_path in decision_candidates:
            run_dir = decision_path.parent
            source_path = run_dir / "source_identity.json"
            selection_path = run_dir / "design/physical_selection_freeze.json"
            magnitude_path = run_dir / "design/magnitude_selection_freeze.json"
            candidate_path = run_dir / "design/candidate_pool_manifest.json"
            physical_path = run_dir / "evaluation_evidence/physical_control_area_record_rows.csv"
            raw_manifest_path = run_dir / "raw_shard_manifest.json"
            required = [
                source_path, selection_path, magnitude_path, candidate_path,
                physical_path, raw_manifest_path,
            ]
            if not all(path.is_file() for path in required):
                continue
            decision = json.loads(decision_path.read_text())
            source = json.loads(source_path.read_text())
            if (
                decision.get("status") == EXPECTED_STAGE28_STATUS
                and decision.get("confirmation_eligible", False)
                and decision.get("physical_control_area_gate", {}).get("passed", False)
                and not decision.get("model_physical_control_area_gate", {}).get("passed", True)
                and decision.get("causal_carrier_control_area_gate", {}).get("passed", False)
                and source.get("protocol_id") == EXPECTED_STAGE28_PROTOCOL_ID
                and source.get("resolved_commit") == EXPECTED_STAGE28_SOURCE_COMMIT
                and source.get("confirmation_eligible", False)
            ):
                valid_stage28.append((run_dir, decision, source))
        if not valid_stage28:
            raise FileNotFoundError(
                "No complete source-bound Stage 28 failure run was found in MyDrive. "
                "Keep the complete Stage 28 Drive directory, not only the downloaded zip."
            )
        valid_stage28.sort(key=lambda row: str(row[0]))
        STAGE28_RUN_DIR, STAGE28_DECISION, STAGE28_SOURCE = valid_stage28[-1]
        STAGE28_TRUTH_DIR = STAGE28_RUN_DIR / "truth"
        selection_path = STAGE28_RUN_DIR / "design/physical_selection_freeze.json"
        magnitude_path = STAGE28_RUN_DIR / "design/magnitude_selection_freeze.json"
        candidate_path = STAGE28_RUN_DIR / "design/candidate_pool_manifest.json"
        physical_path = STAGE28_RUN_DIR / "evaluation_evidence/physical_control_area_record_rows.csv"
        raw_manifest_path = STAGE28_RUN_DIR / "raw_shard_manifest.json"
        STAGE28_SELECTION = json.loads(selection_path.read_text())
        STAGE28_MAGNITUDE = json.loads(magnitude_path.read_text())
        STAGE28_CANDIDATES = json.loads(candidate_path.read_text())
        STAGE28_PHYSICAL_ROWS = read_csv_rows(physical_path)
        SELECTED_MAGNITUDES = [float(value) for value in STAGE28_MAGNITUDE["selected_panel"]]
        if not np.allclose(SELECTED_MAGNITUDES, EXPECTED_STAGE28_MAGNITUDES, atol=0, rtol=0):
            raise RuntimeError("Stage 28 magnitude panel changed")
        selected_ids = [int(value) for value in STAGE28_SELECTION["selected_record_ids"]]
        if len(selected_ids) != EXPECTED_STAGE28_RECORDS:
            raise RuntimeError("Stage 28 selected record count changed")
        if STAGE28_SELECTION.get("selected_counts") != EXPECTED_STAGE28_COUNTS:
            raise RuntimeError("Stage 28 contact-stratum counts changed")

        raw_manifest = {
            str(row["path"]): row for row in json.loads(raw_manifest_path.read_text())
        }
        for record_id in selected_ids:
            relative = f"truth/state_{record_id:06d}.npz"
            path = STAGE28_RUN_DIR / relative
            expected = raw_manifest.get(relative)
            if expected is None or not path.is_file():
                raise FileNotFoundError(f"required Stage 28 truth shard is missing: {relative}")
            if int(path.stat().st_size) != int(expected["bytes"]):
                raise RuntimeError(f"Stage 28 truth shard size changed: {relative}")
            if sha256_file(path) != expected["sha256"]:
                raise RuntimeError(f"Stage 28 truth shard hash changed: {relative}")

        record_specs = {
            int(row["record_id"]): dict(row)
            for row in STAGE28_CANDIDATES["confirmation_specs"]
        }
        physical_lookup = {
            int(row["record_id"]): row for row in STAGE28_PHYSICAL_ROWS
        }
        SELECTED_RECORDS = []
        for record_id in selected_ids:
            if record_id not in record_specs or record_id not in physical_lookup:
                raise RuntimeError(f"Stage 28 record metadata missing for {record_id}")
            record = record_specs[record_id]
            record["regime"] = physical_lookup[record_id]["regime"]
            SELECTED_RECORDS.append(record)
        ALL_EVALUATION_RECORDS = []
        for label in EXPECTED_STAGE28_COUNTS:
            ALL_EVALUATION_RECORDS.extend(
                [row for row in SELECTED_RECORDS if row["regime"] == label][
                    :ACTIVE_RECORDS_PER_STRATUM
                ]
            )
        expected_active = 3 * ACTIVE_RECORDS_PER_STRATUM
        if len(ALL_EVALUATION_RECORDS) != expected_active:
            raise RuntimeError("active Stage 29 stratified record count changed")
        RECORD_BY_ID = {int(row["record_id"]): row for row in record_specs.values()}
        for record in ALL_EVALUATION_RECORDS:
            with np.load(STAGE28_TRUTH_DIR / f"state_{int(record['record_id']):06d}.npz") as payload:
                if payload["selected_actions"].shape != (ACTIONS_PER_STATE, ACTION_STEPS, 2):
                    raise RuntimeError("Stage 28 action-bank shape changed")
                if payload["endpoint_visuals"].shape[0] != ACTIONS_PER_STATE:
                    raise RuntimeError("Stage 28 endpoint-visual count changed")

        STAGE28_CERTIFICATE = {
            "validated_before_stage29_model_activations": True,
            "run_dir": str(STAGE28_RUN_DIR),
            "decision_status": STAGE28_DECISION["status"],
            "resolved_commit": STAGE28_SOURCE["resolved_commit"],
            "decision_sha256": sha256_file(STAGE28_RUN_DIR / "stage28_decision.json"),
            "source_identity_sha256": sha256_file(STAGE28_RUN_DIR / "source_identity.json"),
            "selection_sha256": sha256_file(selection_path),
            "magnitude_sha256": sha256_file(magnitude_path),
            "raw_manifest_sha256": sha256_file(raw_manifest_path),
            "selected_truth_shards_verified": len(selected_ids),
            "active_records": [int(row["record_id"]) for row in ALL_EVALUATION_RECORDS],
            "truth_reused_without_resimulation": True,
        }
        write_json(OUT / "stage28_upstream_certificate.json", STAGE28_CERTIFICATE)
        write_json(DESIGN_DIR / "stage29_grounded_closure_design.json", {
            "created_before_stage29_model_activations": True,
            "selected_magnitudes": SELECTED_MAGNITUDES,
            "selected_record_ids": [int(row["record_id"]) for row in ALL_EVALUATION_RECORDS],
            "selected_counts": {
                label: sum(row["regime"] == label for row in ALL_EVALUATION_RECORDS)
                for label in EXPECTED_STAGE28_COUNTS
            },
            "wrong_state_map": STAGE28_SELECTION["wrong_state_map"],
            "stage28_selection_reused_without_refit": True,
            "stage29_reader_fit": False,
            "stage29_subspace_refit": False,
            "jacobian_or_gradient_used": False,
        })
        STAGE28_UPSTREAM_BOUND = True
        memory_report("upstream_stage18_and_stage28_validated")
    except Exception:
        record_failure("upstream_stage18_or_stage28_import")
'''


model_initialization = r'''# Load frozen JEPA-WM and verify native target-token and hook contracts.


def state_model_inputs(record_id, horizon=PRIMARY_HORIZON):
    with np.load(branch_path(record_id)) as truth:
        initial_visual = truth["initial_visual"]
        initial_proprio = truth["initial_proprio"]
        selected_actions = truth["selected_actions"]
    with torch.inference_mode():
        initial = MODEL.encode(to_model_observation(initial_visual, initial_proprio))
    initial = {name: value.detach() for name, value in initial.items()}
    actions = model_action_tensor(PREPROCESSOR, selected_actions, horizon)
    return initial, actions


def encode_true_tokens(record_id):
    with np.load(branch_path(record_id)) as payload:
        visual = payload["endpoint_visuals"][:, None]
        states = payload["endpoint_states"].astype(np.float32)
    proprio = np.concatenate([states[:, :2], states[:, 5:7]], axis=1)[:, None]
    with torch.inference_mode():
        encoded = MODEL.encode(to_model_observation(visual, proprio))
    tokens = encoded["visual"][:, :, 0]
    tokens = tokens.reshape(ACTIONS_PER_STATE, 256, tokens.shape[-1])
    if tokens.shape != (ACTIONS_PER_STATE, 256, 384):
        raise RuntimeError(f"unexpected true target-token shape {tuple(tokens.shape)}")
    return tokens.detach()


def tensor_alignment(source, target):
    left = source.float().reshape(-1)
    right = target.float().reshape(-1)
    source_energy = torch.dot(left, left)
    target_energy = torch.dot(right, right)
    dot = torch.dot(left, right)
    source_value = float(source_energy.cpu())
    target_value = float(target_energy.cpu())
    dot_value = float(dot.cpu())
    if target_value <= 1e-20:
        return {
            "source_energy": source_value, "target_energy": target_value,
            "dot": dot_value, "coefficient": math.nan, "cosine": math.nan,
            "normalized_rmse": math.nan,
        }
    denominator = math.sqrt(max(source_value * target_value, 0.0))
    difference = left - right
    return {
        "source_energy": source_value,
        "target_energy": target_value,
        "dot": dot_value,
        "coefficient": dot_value / target_value,
        "cosine": dot_value / denominator if denominator > 1e-20 else 0.0,
        "normalized_rmse": float(torch.sqrt(torch.dot(difference, difference) / target_energy).cpu()),
    }


def tensor_magnitude_center(values):
    grouped = values.reshape(MAGNITUDE_COUNT, SCHEDULE_COUNT, *values.shape[1:])
    return (grouped - grouped.mean(dim=1, keepdim=True)).reshape_as(values)


def tensor_area_component(values):
    permutation = torch.as_tensor(
        area_reversal_permutation(MAGNITUDE_COUNT), device=values.device
    )
    return 0.5 * (values - values[permutation])


def tensor_ideal_effect(values, mode):
    if mode == "swap":
        permutation = torch.as_tensor(
            area_reversal_permutation(MAGNITUDE_COUNT), device=values.device
        )
        return values[permutation] - values
    if mode == "ablation":
        return -tensor_area_component(values)
    raise ValueError("mode must be swap or ablation")


def tensor_absolute_target(values, mode):
    if mode == "swap":
        permutation = torch.as_tensor(
            area_reversal_permutation(MAGNITUDE_COUNT), device=values.device
        )
        return values[permutation]
    if mode == "ablation":
        return values - tensor_area_component(values)
    raise ValueError("mode must be swap or ablation")


def tensor_grounded_metrics(baseline, patched, target, mode):
    effect = patched - baseline
    self_metrics = tensor_alignment(effect, tensor_ideal_effect(baseline, mode))
    grounded_metrics = tensor_alignment(effect, tensor_ideal_effect(target, mode))
    absolute_target = tensor_absolute_target(target, mode)
    before = float(torch.sum((baseline.float() - absolute_target.float()) ** 2).cpu())
    after = float(torch.sum((patched.float() - absolute_target.float()) ** 2).cpu())
    self_cosine = self_metrics["cosine"]
    grounded_cosine = grounded_metrics["cosine"]
    return {
        "effect_energy": float(torch.sum(effect.float() ** 2).cpu()),
        **{f"self_{key}": value for key, value in self_metrics.items()},
        **{f"grounded_{key}": value for key, value in grounded_metrics.items()},
        "self_minus_grounded_cosine": (
            self_cosine - grounded_cosine
            if np.isfinite(self_cosine) and np.isfinite(grounded_cosine)
            else math.nan
        ),
        "absolute_target_error_before": before,
        "absolute_target_error_after": after,
        "absolute_target_error_reduction": 1.0 - after / before if before > 1e-20 else math.nan,
    }


def hook_identity_test(record_id):
    initial, actions = state_model_inputs(record_id)
    with torch.inference_mode():
        baseline, _, _ = forward_with_carriers(
            initial, actions, PRIMARY_HORIZON, capture_blocks=[FIXED_BLOCK]
        )
        patched, _, _ = forward_with_carriers(
            initial, actions, PRIMARY_HORIZON, capture_blocks=[FIXED_BLOCK],
            intervention={
                "block": FIXED_BLOCK,
                "delta": torch.zeros(
                    ACTIONS_PER_STATE, 256, EXPECTED_CARRIER_CHANNELS,
                    device="cuda", dtype=torch.float32,
                ),
            },
        )
    error = float(torch.max(torch.abs(patched - baseline)).cpu())
    result = {"record_id": int(record_id), "max_abs_error": error, "passed": error <= MAX_ZERO_EDIT_ERROR}
    write_json(OUT / "hook_identity_test.json", result)
    if not result["passed"]:
        raise RuntimeError(f"zero hook identity failed: {result}")
    return result


MODEL_READY = False
if not PIPELINE_FAILED:
    try:
        if not STAGE18_ARTIFACT_VALIDATED or not STAGE28_UPSTREAM_BOUND:
            raise RuntimeError("Stage 18 and exact Stage 28 evidence must be bound first")
        REPO = configure_repo()
        MODEL, PREPROCESSOR, PREDICTOR, PREDICTOR_BLOCK_MODULES = load_frozen_model()
        if len(PREDICTOR_BLOCK_MODULES) != 6:
            raise RuntimeError("predictor block count changed")
        if not all(
            isinstance(module, torch.nn.Module)
            and callable(getattr(module, "register_forward_hook", None))
            for module in PREDICTOR_BLOCK_MODULES
        ):
            raise RuntimeError("predictor block hook contract changed")
        DECODE_PHYSICAL_POSE = physical_pose_decoder()
        probe_record = ALL_EVALUATION_RECORDS[0]
        probe_target = encode_true_tokens(probe_record["record_id"])
        if probe_target.requires_grad:
            raise RuntimeError("true target tokens unexpectedly require gradients")
        del probe_target
        HOOK_IDENTITY = hook_identity_test(probe_record["record_id"])
        initial, actions = state_model_inputs(probe_record["record_id"])
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            _, _, _ = forward_with_carriers(
                initial, actions, PRIMARY_HORIZON, capture_blocks=[FIXED_BLOCK]
            )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        estimated_batches = len(ALL_EVALUATION_RECORDS) * (
            ACTIVE_INTERVENTION_FORWARDS_PER_RECORD + 2
        )
        FORWARD_BENCHMARK = {
            "seconds_per_24_branch_predictor_batch": seconds,
            "predictor_batches_per_record": ACTIVE_INTERVENTION_FORWARDS_PER_RECORD + 2,
            "target_encoder_batches_per_record": 2,
            "evaluation_records": len(ALL_EVALUATION_RECORDS),
            "estimated_predictor_minutes": seconds * estimated_batches / 60.0,
            "warning_threshold_minutes": MAX_ESTIMATED_TOTAL_MINUTES,
        }
        write_json(OUT / "forward_benchmark.json", FORWARD_BENCHMARK)
        if (
            FORWARD_BENCHMARK["estimated_predictor_minutes"] > MAX_ESTIMATED_TOTAL_MINUTES
            and not CONTINUE_AFTER_BENCHMARK
        ):
            raise RuntimeError("measured forward estimate exceeds the configured credit guard")
        del initial, actions
        MODEL_READY = True
        write_json(OUT / "evaluation_open_certificate.json", {
            "opened": True,
            "source_identity": SOURCE_IDENTITY,
            "stage18_artifact_certificate_sha256": sha256_file(OUT / "stage18_artifact_certificate.json"),
            "stage28_upstream_certificate_sha256": sha256_file(OUT / "stage28_upstream_certificate.json"),
            "design_sha256": sha256_file(DESIGN_DIR / "stage29_grounded_closure_design.json"),
            "fit_or_selection_model_activations": [],
            "stage29_subspace_refit": False,
            "stage29_reader_fit": False,
        })
        memory_report("stage29_model_and_target_contracts_verified")
    except Exception:
        record_failure("stage29_model_initialization")
'''


grounded_evaluation = r'''# Measure native predictor-target closure and grounded rank-128 interventions.


def whiten_carrier(values, subspaces):
    return transform_primal_channels(
        np.asarray(values, dtype=np.float64),
        subspaces["channel_inverse_square_root"],
    )


def native_edit(values, subspaces):
    return inverse_transform_primal_channels(
        np.asarray(values, dtype=np.float64), subspaces["channel_square_root"]
    )


def matched_norm(value, reference):
    array = np.asarray(value, dtype=np.float64)
    target = float(np.linalg.norm(reference))
    norm = float(np.linalg.norm(array))
    if norm <= 1e-12 or target <= 1e-12:
        raise RuntimeError("cannot norm-match a degenerate intervention")
    return array * (target / norm)


def wrong_state_swap_delta(wrong_white, basis):
    component = area_antisymmetric_component(
        wrong_white, MAGNITUDE_COUNT
    ).reshape(ACTIONS_PER_STATE, -1)
    projected = (component @ basis) @ basis.T
    return (-2.0 * projected).reshape(wrong_white.shape)


def intervention_specs(carrier, wrong_carrier, subspaces):
    white = whiten_carrier(carrier, subspaces)
    wrong_white = whiten_carrier(wrong_carrier, subspaces)
    primary_basis = subspaces["primary_basis"][:, :PRIMARY_RANK]
    primary = area_swap_delta(
        white, MAGNITUDE_COUNT, basis=primary_basis, dose=1.0
    )
    shuffled = matched_norm(
        area_swap_delta(
            white, MAGNITUDE_COUNT,
            basis=subspaces["shuffled_basis"][:, :PRIMARY_RANK], dose=1.0,
        ),
        primary,
    )
    specifications = [
        {"condition": "primary_r128_swap", "family": "primary", "mode": "swap", "delta_white": primary},
        {"condition": "shuffled_r128_swap", "family": "matched_shuffled_control", "mode": "swap", "delta_white": shuffled},
    ]
    for draw in range(ACTIVE_CAUSAL_RANDOM_DRAWS):
        random_delta = area_swap_delta(
            white, MAGNITUDE_COUNT,
            basis=subspaces[f"random_basis_{draw:02d}"][:, :PRIMARY_RANK], dose=1.0,
        )
        specifications.append({
            "condition": f"random_r128_{draw:02d}_swap",
            "family": "empirical_span_random_control",
            "mode": "swap",
            "delta_white": matched_norm(random_delta, primary),
        })
    wrong = wrong_state_swap_delta(wrong_white, primary_basis)
    specifications.append({
        "condition": "wrong_state_r128_swap",
        "family": "state_specificity_control",
        "mode": "swap",
        "delta_white": matched_norm(wrong, primary),
    })
    full = area_swap_delta(white, MAGNITUDE_COUNT, basis=None, dose=1.0)
    specifications.append({
        "condition": "full_activation_swap",
        "family": "positive_control_only",
        "mode": "swap",
        "delta_white": full,
    })
    ablation = area_ablation_delta(
        white, MAGNITUDE_COUNT, primary_basis, dose=1.0
    )
    specifications.append({
        "condition": "primary_r128_ablation",
        "family": "primary_necessity",
        "mode": "ablation",
        "delta_white": ablation,
    })
    if len(specifications) != ACTIVE_INTERVENTION_FORWARDS_PER_RECORD:
        raise RuntimeError(
            f"expected {ACTIVE_INTERVENTION_FORWARDS_PER_RECORD} interventions, "
            f"found {len(specifications)}"
        )
    for specification in specifications:
        specification["edit_norm"] = float(np.linalg.norm(specification["delta_white"]))
        specification["primary_norm"] = float(np.linalg.norm(primary))
        specification["full_norm"] = float(np.linalg.norm(full))
    return specifications


def decoder_alignment_row(decoded, truth_pose):
    metrics = model_physics_area_metrics(
        decoded, truth_pose, MAGNITUDE_COUNT
    )
    return {
        "source_energy": metrics["predicted_energy"],
        "target_energy": metrics["target_energy"],
        "dot": metrics["coefficient"] * metrics["target_energy"],
        "coefficient": metrics["coefficient"],
        "cosine": metrics["cosine"],
        "normalized_rmse": metrics["normalized_rmse"],
    }


def baseline_record_row(record, predicted, target, wrong_target):
    total = tensor_alignment(predicted, target)
    centered = tensor_alignment(
        tensor_magnitude_center(predicted), tensor_magnitude_center(target)
    )
    predicted_area = tensor_area_component(predicted)
    target_area = tensor_area_component(target)
    wrong_area = tensor_area_component(wrong_target)
    area = tensor_alignment(predicted_area, target_area)
    wrong = tensor_alignment(predicted_area, wrong_area)
    with np.load(branch_path(record["record_id"])) as payload:
        truth_pose = pose_target(payload["endpoint_states"].astype(np.float64))
    with torch.inference_mode():
        predicted_pose = DECODE_PHYSICAL_POSE(predicted).detach().cpu().numpy()
        target_pose = DECODE_PHYSICAL_POSE(target).detach().cpu().numpy()
    predicted_decoder = decoder_alignment_row(predicted_pose, truth_pose)
    target_decoder = decoder_alignment_row(target_pose, truth_pose)
    token_count = int(target_area.numel())
    return {
        "record_id": int(record["record_id"]),
        "trajectory_id": int(record["trajectory_id"]),
        "regime": record["regime"],
        "target_area_mean_square": area["target_energy"] / token_count,
        **{f"native_total_{key}": value for key, value in total.items()},
        **{f"native_centered_{key}": value for key, value in centered.items()},
        **{f"native_area_{key}": value for key, value in area.items()},
        **{f"wrong_area_{key}": value for key, value in wrong.items()},
        **{f"pred_decoder_{key}": value for key, value in predicted_decoder.items()},
        **{f"target_decoder_{key}": value for key, value in target_decoder.items()},
    }


def finite_json_rows(rows):
    output = []
    for row in rows:
        output.append({
            key: (
                None
                if isinstance(value, (float, np.floating)) and not np.isfinite(value)
                else value
            )
            for key, value in row.items()
        })
    return output


def run_grounded_record(record, subspaces):
    record_id = int(record["record_id"])
    destination = INTERVENTION_DIR / f"state_{record_id:06d}.json"
    if destination.exists():
        PROVENANCE_COUNTS["cache_hits"] += 1
        raise RuntimeError(f"fresh-run grounded shard already exists: {destination}")
    wrong_id = int(STAGE28_SELECTION["wrong_state_map"][str(record_id)])
    initial, actions = state_model_inputs(record_id)
    wrong_initial, wrong_actions = state_model_inputs(wrong_id)
    target = encode_true_tokens(record_id)
    wrong_target = encode_true_tokens(wrong_id)
    with torch.inference_mode():
        predicted, _, captures = forward_with_carriers(
            initial, actions, PRIMARY_HORIZON, capture_blocks=[FIXED_BLOCK]
        )
        _, _, wrong_captures = forward_with_carriers(
            wrong_initial, wrong_actions, PRIMARY_HORIZON,
            capture_blocks=[FIXED_BLOCK],
        )
    carrier = layer_tokens_full(captures[FIXED_BLOCK]).detach().float().cpu().numpy()
    wrong_carrier = layer_tokens_full(
        wrong_captures[FIXED_BLOCK]
    ).detach().float().cpu().numpy()
    baseline_row = baseline_record_row(record, predicted, target, wrong_target)
    specifications = intervention_specs(carrier, wrong_carrier, subspaces)
    rows = []
    for specification in specifications:
        delta_native = native_edit(specification["delta_white"], subspaces)
        delta_tensor = torch.as_tensor(
            delta_native, device="cuda", dtype=torch.float32
        )
        with torch.inference_mode():
            patched, _, _ = forward_with_carriers(
                initial, actions, PRIMARY_HORIZON,
                capture_blocks=[FIXED_BLOCK],
                intervention={"block": FIXED_BLOCK, "delta": delta_tensor},
            )
        metrics = tensor_grounded_metrics(
            predicted, patched, target, specification["mode"]
        )
        rows.append({
            "record_id": record_id,
            "trajectory_id": int(record["trajectory_id"]),
            "regime": record["regime"],
            "wrong_state_record_id": wrong_id,
            "condition": specification["condition"],
            "family": specification["family"],
            "mode": specification["mode"],
            "rank": PRIMARY_RANK if "full_activation" not in specification["condition"] else -1,
            "carrier_edit_whitened_norm": specification["edit_norm"],
            "primary_swap_whitened_norm": specification["primary_norm"],
            "full_swap_whitened_norm": specification["full_norm"],
            **metrics,
        })
        del patched, delta_tensor
    write_json(destination, {
        "baseline": finite_json_rows([baseline_row])[0],
        "interventions": finite_json_rows(rows),
    })
    PROVENANCE_COUNTS["baseline_generated"] += 1
    PROVENANCE_COUNTS["intervention_generated"] += 1
    del (
        initial, actions, wrong_initial, wrong_actions, target, wrong_target,
        predicted, captures, wrong_captures, carrier, wrong_carrier,
    )
    gc.collect()
    torch.cuda.empty_cache()
    return baseline_row, rows


BASELINE_ROWS = []
INTERVENTION_ROWS = []
if not PIPELINE_FAILED and MODEL_READY:
    try:
        started = time.perf_counter()
        for index, record in enumerate(ALL_EVALUATION_RECORDS):
            baseline_row, intervention_rows = run_grounded_record(
                record, FROZEN_SUBSPACES
            )
            BASELINE_ROWS.append(baseline_row)
            INTERVENTION_ROWS.extend(intervention_rows)
            write_json(OUT / "grounded_evaluation_progress.json", {
                "completed": index + 1,
                "total": len(ALL_EVALUATION_RECORDS),
                "last_record_id": int(record["record_id"]),
            })
        TIMINGS["grounded_closure_evaluation_seconds"] = time.perf_counter() - started
        write_csv(EVIDENCE_DIR / "native_predictor_target_rows.csv", BASELINE_ROWS)
        write_csv(EVIDENCE_DIR / "grounded_intervention_rows.csv", INTERVENTION_ROWS)
        memory_report("stage29_grounded_evaluation_complete")
    except Exception:
        record_failure("stage29_grounded_evaluation")
'''


decision_and_plots = r'''# Localize encoder, predictor, readout, and causal-grounding outcomes.


def aggregate_alignment(rows, prefix):
    source_energy = float(sum(float(row[f"{prefix}source_energy"]) for row in rows))
    target_energy = float(sum(float(row[f"{prefix}target_energy"]) for row in rows))
    dot = float(sum(float(row[f"{prefix}dot"]) for row in rows))
    denominator = math.sqrt(max(source_energy * target_energy, 0.0))
    return {
        "records": len(rows),
        "source_energy": source_energy,
        "target_energy": target_energy,
        "dot": dot,
        "coefficient": dot / max(target_energy, 1e-20),
        "cosine": dot / denominator if denominator > 1e-20 else math.nan,
    }


def bootstrap_interval(values, trajectories, label):
    draws = clustered_bootstrap_mean(
        np.asarray(values, dtype=np.float64),
        np.asarray(trajectories, dtype=np.int64),
        ACTIVE_BOOTSTRAP_DRAWS,
        stable_seed(BOOTSTRAP_SEED, label) % (2**31 - 1),
    )
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def target_encoder_gate():
    persistent = [row for row in BASELINE_ROWS if row["regime"] == "persistent_contact"]
    free = [row for row in BASELINE_ROWS if row["regime"] == "free"]
    persistent_energy = float(np.median([row["target_area_mean_square"] for row in persistent]))
    free_energy = float(np.median([row["target_area_mean_square"] for row in free]))
    ratio = persistent_energy / max(free_energy, 1e-20)
    return {
        "persistent_records": len(persistent),
        "free_records": len(free),
        "median_persistent_target_area_mean_square": persistent_energy,
        "median_free_target_area_mean_square": free_energy,
        "persistent_to_free_ratio": ratio,
        "minimum_persistent_energy": MIN_TARGET_AREA_MEAN_SQUARE,
        "minimum_ratio": MIN_TARGET_PERSISTENT_TO_FREE_RATIO,
        "passed": bool(
            persistent_energy >= MIN_TARGET_AREA_MEAN_SQUARE
            and ratio >= MIN_TARGET_PERSISTENT_TO_FREE_RATIO
        ),
    }


def predictor_target_gate():
    persistent = [row for row in BASELINE_ROWS if row["regime"] == "persistent_contact"]
    aligned = aggregate_alignment(persistent, "native_area_")
    wrong = aggregate_alignment(persistent, "wrong_area_")
    gain = aligned["cosine"] - wrong["cosine"]
    record_cosines = np.asarray([row["native_area_cosine"] for row in persistent], dtype=np.float64)
    ci = bootstrap_interval(
        record_cosines,
        [row["trajectory_id"] for row in persistent],
        "predictor_target_area_cosine",
    )
    return {
        "persistent_records": len(persistent),
        "aggregate": aligned,
        "wrong_state_aggregate": wrong,
        "alignment_gain_over_wrong_state": gain,
        "mean_record_cosine": float(np.nanmean(record_cosines)),
        "record_cosine_ci95": ci,
        "passed": bool(
            aligned["cosine"] >= MIN_PREDICTOR_TARGET_AREA_COSINE
            and gain >= MIN_PREDICTOR_GAIN_OVER_WRONG_STATE
        ),
    }


def decoder_localization_gate():
    persistent = [row for row in BASELINE_ROWS if row["regime"] == "persistent_contact"]
    predicted = aggregate_alignment(persistent, "pred_decoder_")
    encoded_true = aggregate_alignment(persistent, "target_decoder_")
    return {
        "persistent_records": len(persistent),
        "predicted_latent_decoder": predicted,
        "encoded_true_latent_decoder": encoded_true,
        "encoded_true_minus_predicted_cosine": encoded_true["cosine"] - predicted["cosine"],
        "passed": bool(
            encoded_true["cosine"] >= MIN_TRUE_TARGET_DECODER_AREA_COSINE
        ),
    }


def intervention_gate(prefix):
    persistent = [
        row for row in INTERVENTION_ROWS
        if row["regime"] == "persistent_contact"
    ]
    primary = [row for row in persistent if row["condition"] == "primary_r128_swap"]
    full = [row for row in persistent if row["condition"] == "full_activation_swap"]
    gains = []
    trajectories = []
    for row in primary:
        record_id = int(row["record_id"])
        random_values = [
            candidate[f"{prefix}_coefficient"]
            for candidate in persistent
            if int(candidate["record_id"]) == record_id
            and candidate["family"] == "empirical_span_random_control"
        ]
        gains.append(float(row[f"{prefix}_coefficient"] - np.median(random_values)))
        trajectories.append(int(row["trajectory_id"]))
    coefficients = np.asarray([row[f"{prefix}_coefficient"] for row in primary], dtype=np.float64)
    cosines = np.asarray([row[f"{prefix}_cosine"] for row in primary], dtype=np.float64)
    finite = np.isfinite(coefficients) & np.isfinite(cosines)
    finite_gains = np.asarray(gains, dtype=np.float64)
    gain_ci = bootstrap_interval(finite_gains, trajectories, f"{prefix}_gain_random")
    sign = exact_positive_sign_test(finite_gains)
    result = {
        "persistent_records": len(primary),
        "finite_primary_records": int(np.sum(finite)),
        "mean_primary_coefficient": float(np.nanmean(coefficients)),
        "mean_primary_cosine": float(np.nanmean(cosines)),
        "mean_gain_over_random": float(np.nanmean(finite_gains)),
        "gain_over_random_ci95": gain_ci,
        "gain_over_random_sign_test": sign,
        "positive_gain_records": int(np.sum(finite_gains > 0)),
    }
    if prefix == "self":
        result["mean_full_swap_coefficient"] = float(
            np.nanmean([row["self_coefficient"] for row in full])
        )
        result["passed"] = bool(
            result["mean_full_swap_coefficient"] >= MIN_FULL_SWAP_SELF_COEFFICIENT
            and result["mean_primary_coefficient"] >= MIN_PRIMARY_CLOSURE_COEFFICIENT
            and result["mean_primary_cosine"] >= MIN_PRIMARY_CLOSURE_COSINE
            and result["mean_gain_over_random"] >= MIN_PRIMARY_GAIN_OVER_RANDOM
            and result["positive_gain_records"] >= REQUIRED_POSITIVE_PERSISTENT_RECORDS
        )
    else:
        result["passed"] = bool(
            result["mean_primary_coefficient"] >= MIN_PRIMARY_CLOSURE_COEFFICIENT
            and result["mean_primary_cosine"] >= MIN_PRIMARY_CLOSURE_COSINE
            and result["mean_gain_over_random"] >= MIN_PRIMARY_GAIN_OVER_RANDOM
            and result["positive_gain_records"] >= REQUIRED_POSITIVE_PERSISTENT_RECORDS
        )
    return result


def free_motion_leakage_summary():
    rows = [
        row for row in INTERVENTION_ROWS
        if row["regime"] == "free" and row["condition"] == "primary_r128_swap"
    ]
    baselines = [row for row in BASELINE_ROWS if row["regime"] == "free"]
    return {
        "free_records": len(rows),
        "median_true_target_area_mean_square": float(np.median([
            row["target_area_mean_square"] for row in baselines
        ])),
        "median_primary_effect_energy": float(np.median([
            row["effect_energy"] for row in rows
        ])),
        "mean_self_coefficient": float(np.nanmean([
            row["self_coefficient"] for row in rows
        ])),
        "grounded_metrics_defined_records": int(sum(
            np.isfinite(row["grounded_coefficient"]) for row in rows
        )),
    }


def make_plots(encoder_gate, predictor_gate, decoder_gate, self_gate, grounded_gate):
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    regimes = list(EXPECTED_STAGE28_COUNTS)
    energy = [
        np.median([
            row["target_area_mean_square"] for row in BASELINE_ROWS
            if row["regime"] == label
        ])
        for label in regimes
    ]
    axes[0].bar(regimes, np.log10(np.asarray(energy) + 1e-20), color=["#d95f02", "#7570b3", "#1b9e77"])
    axes[0].set_ylabel("log10 encoded-true area MSE")
    axes[0].set_title("Target encoder sensitivity")
    axes[0].tick_params(axis="x", rotation=22)

    for label, color in zip(regimes, ["#d95f02", "#7570b3", "#1b9e77"]):
        rows = [row for row in BASELINE_ROWS if row["regime"] == label]
        axes[1].scatter(
            [label] * len(rows), [row["native_area_cosine"] for row in rows],
            color=color, alpha=0.8,
        )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].set_ylabel("predicted vs encoded-true cosine")
    axes[1].set_title("Native area-contrast closure")
    axes[1].tick_params(axis="x", rotation=22)

    primary = [
        row for row in INTERVENTION_ROWS
        if row["regime"] == "persistent_contact"
        and row["condition"] == "primary_r128_swap"
    ]
    axes[2].scatter(
        [row["self_cosine"] for row in primary],
        [row["grounded_cosine"] for row in primary],
        color="#4c78a8",
    )
    axes[2].axline((0, 0), slope=1, color="black", linestyle="--", linewidth=0.8)
    axes[2].axhline(0, color="gray", linewidth=0.7)
    axes[2].axvline(0, color="gray", linewidth=0.7)
    axes[2].set_xlim(-1.05, 1.05); axes[2].set_ylim(-1.05, 1.05)
    axes[2].set_xlabel("self-consistent closure cosine")
    axes[2].set_ylabel("grounded closure cosine")
    axes[2].set_title("Frozen rank-128 intervention")
    figure.tight_layout()
    figure.savefig(PLOT_DIR / "stage29_grounded_causal_closure_summary.png", dpi=180)
    plt.close(figure)


DECISION_PAYLOAD = {"status": "INCONCLUSIVE"}
if not PIPELINE_FAILED:
    try:
        ENCODER_GATE = target_encoder_gate()
        PREDICTOR_GATE = predictor_target_gate()
        DECODER_GATE = decoder_localization_gate()
        SELF_GATE = intervention_gate("self")
        GROUNDED_GATE = intervention_gate("grounded")
        FREE_LEAKAGE = free_motion_leakage_summary()
        FRESH_CERTIFICATE = {
            "out_preexisted": bool(OUT_PREEXISTED),
            "cache_hits": int(PROVENANCE_COUNTS["cache_hits"]),
            "baseline_records_generated": int(PROVENANCE_COUNTS["baseline_generated"]),
            "intervention_records_generated": int(PROVENANCE_COUNTS["intervention_generated"]),
            "expected_records": len(ALL_EVALUATION_RECORDS),
            "source_execution_verified": bool(SOURCE_IDENTITY.get("confirmation_eligible", False)),
            "stage18_artifact_validated": bool(STAGE18_ARTIFACT_VALIDATED),
            "stage28_exact_run_bound": bool(STAGE28_UPSTREAM_BOUND),
        }
        FRESH_CERTIFICATE["passed"] = bool(
            not FRESH_CERTIFICATE["out_preexisted"]
            and FRESH_CERTIFICATE["cache_hits"] == 0
            and FRESH_CERTIFICATE["baseline_records_generated"] == FRESH_CERTIFICATE["expected_records"]
            and FRESH_CERTIFICATE["intervention_records_generated"] == FRESH_CERTIFICATE["expected_records"]
            and FRESH_CERTIFICATE["source_execution_verified"]
            and FRESH_CERTIFICATE["stage18_artifact_validated"]
            and FRESH_CERTIFICATE["stage28_exact_run_bound"]
        )
        write_json(OUT / "fresh_run_certificate.json", FRESH_CERTIFICATE)

        if not ENCODER_GATE["passed"]:
            candidate_status = "ENCODED_TRUE_FUTURES_DO_NOT_RESOLVE_CONTROL_AREA"
        elif PREDICTOR_GATE["passed"] and DECODER_GATE["passed"] and GROUNDED_GATE["passed"]:
            candidate_status = "GROUNDED_CAUSAL_CLOSURE_SUPPORTED"
        elif PREDICTOR_GATE["passed"] and not DECODER_GATE["passed"]:
            candidate_status = "PHYSICAL_READOUT_LIMITATION_SUPPORTED"
        elif not PREDICTOR_GATE["passed"] and SELF_GATE["passed"] and not GROUNDED_GATE["passed"]:
            candidate_status = "CAUSAL_SELF_CONSISTENCY_WITHOUT_GROUNDED_CLOSURE"
        elif not PREDICTOR_GATE["passed"]:
            candidate_status = "PREDICTOR_TARGET_CLOSURE_FAILED"
        else:
            candidate_status = "MIXED_GROUNDED_CLOSURE_RESULT"
        confirmation_eligible = bool(
            SOURCE_IDENTITY.get("confirmation_eligible", False)
            and STAGE18_ARTIFACT_VALIDATED
            and STAGE28_UPSTREAM_BOUND
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
            "exact_stage28_upstream_claim_eligible": bool(STAGE28_UPSTREAM_BOUND),
            "fresh_run_claim_eligible": FRESH_CERTIFICATE["passed"],
            "target_encoder_sensitivity_gate": ENCODER_GATE,
            "native_predictor_target_closure_gate": PREDICTOR_GATE,
            "frozen_decoder_localization_gate": DECODER_GATE,
            "self_consistent_causal_closure_gate": SELF_GATE,
            "grounded_causal_closure_gate": GROUNDED_GATE,
            "free_motion_internal_effect_summary": FREE_LEAKAGE,
            "decision_logic": {
                "encoder_failure": "encoded exact futures do not distinguish the persistent-contact area contrast",
                "predictor_failure": "encoded exact futures distinguish it, but predicted native contrasts do not align",
                "readout_failure": "native predictor-target closure passes while the frozen decoder on encoded truth fails",
                "causal_self_without_grounding": "rank-128 edits follow the model donor but not the encoded simulator donor",
            },
            "claim_boundary": {
                "same_checkpoint_encoder_and_predictor": True,
                "exact_stage28_simulator_futures_reused": True,
                "full_native_256_by_384_token_metrics": True,
                "new_reader_fit": False,
                "stage18_subspace_refit_or_tuning": False,
                "jacobian_jvp_vjp_or_gradient_used": False,
                "one_model_checkpoint": True,
                "one_environment": True,
                "visual_quality_claim": False,
                "generalization_to_other_models_or_environments": False,
            },
            "prespecified_next_step": (
                "replicate the localized encoder, predictor, readout, or causal-grounding failure "
                "across a second checkpoint and contact-rich environment"
            ),
        }
        write_json(OUT / "stage29_decision.json", DECISION_PAYLOAD)
        make_plots(ENCODER_GATE, PREDICTOR_GATE, DECODER_GATE, SELF_GATE, GROUNDED_GATE)
        print(json.dumps(DECISION_PAYLOAD, indent=2))
    except Exception:
        record_failure("stage29_decision_and_plots")
        DECISION_PAYLOAD = {"status": "INCONCLUSIVE", "failure": FAILURE_MESSAGE}

if not (OUT / "stage29_decision.json").exists():
    write_json(OUT / "stage29_decision.json", DECISION_PAYLOAD)
'''


packaging = STAGE28.packaging.replace(
    "stage28_hybrid_control_area_result_bundle_",
    "stage29_grounded_closure_result_bundle_",
)


protocol_sources = [
    introduction,
    configuration,
    installation,
    setup,
    analysis_helpers,
    model_helpers,
    artifact_import,
    model_initialization,
    grounded_evaluation,
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
    code(artifact_import),
    code(model_initialization),
    code(grounded_evaluation),
    code(decision_and_plots),
    code(packaging),
]
for index, cell in enumerate(cells):
    cell["id"] = f"stage29-{index:02d}"

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "L4", "name": TARGET.name, "provenance": []},
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
TARGET.write_text(json.dumps(notebook, indent=1) + "\n")
print(f"Wrote {TARGET}")
