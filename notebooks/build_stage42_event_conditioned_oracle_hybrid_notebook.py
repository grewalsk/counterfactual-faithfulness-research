"""Build the prospective Stage 42 event-conditioned oracle hybrid Colab."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
TARGET = ROOT / "42_event_conditioned_oracle_hybrid.ipynb"
NUMERICAL = REPOSITORY / "src/cf_faithfulness/stage42_event_conditioned_hybrid.py"


def load_builder(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BASE = load_builder(
    ROOT / "build_stage41_causal_event_reset_headroom_notebook.py",
    "stage41_builder_for_stage42",
)

code = BASE.code
markdown = BASE.markdown
replace_assignment = BASE.replace_assignment
replace_block = BASE.replace_block
assigned_uppercase_names = BASE.assigned_uppercase_names
function_sources = BASE.function_sources


def rename(value: str) -> str:
    for old, new in [
        ("Stage 41", "Stage 42"),
        ("STAGE41", "STAGE42"),
        ("stage41", "stage42"),
    ]:
        value = value.replace(old, new)
    return value


introduction = r'''# Stage 42: action-conditioned hybrid defect decomposition

## Prospective event-excitation support-qualified headroom test

Stage 41 reached its first held-out panel but correctly stopped because the
ordinary evaluation bank contained fewer than eight post-contact-to-contact
re-entry rows.  Stage 42 v1 then screened all 16,000 registered evaluation
families and found zero re-entry-positive families: its inward-biased A--D action
vocabulary could not excite the leave--gap--return transition.  That negative
support result is development evidence, not an oracle-head outcome.  Stage 42
v2 does not lower the threshold, enlarge the failed pool, or reuse any of its
evaluation identifiers.  It prospectively registers a fresh trajectory pool
and an explicit event-excitation vocabulary with inward, transverse, outward,
and zero-hold macros.  The estimand remains an event-rich finite PushT bank.

Before either frozen checkpoint is loaded, candidate evaluation families are
screened using only exact simulator contact incidence under the newly
registered action words.  The deterministic rule retains the earliest 48 complete
four-mode families having at least one terminal post-contact-to-contact
re-entry.  It requires at least 32 qualifying rows in total.  Model outputs,
prediction errors, intervention magnitudes, oracle-head outcomes, and planning
statistics are forbidden during selection.  The selected identifiers, family
counts, screening rows, and source digest are sealed in a support certificate.

For a complete controlled state, contact does **not** invalidate composition:
for action words `u` followed by `v`, `T_{uv} = T_v o T_u`.  Hybrid dynamics
instead factor each word into within-mode flows, guard crossings, and reset
maps.  Stage 42 therefore measures three distinct defects:

1. **flow-composition defect**: recursive versus direct frozen-state error on
   paths with no recorded contact event;
2. **guard defect**: development-trained linear-probe AUROC, balanced accuracy,
   Brier score, and event-time MAE from frozen action-conditioned rollout
   features; and
3. **reset defect/headroom**: physical error at post-contact-to-contact
   re-entry and the residual after an oracle event/time/geometry/impulse head.

The reset test repeats the equal-width Stage 41 oracle ladder on the newly
frozen bank: smooth sham, shuffled event metadata, event, event time, event
geometry, and the complete event/time/normal/impulse reset ceiling.  Every
ridge head has identical nominal width and is frozen before the held-out model
panel is opened.  The guard probe is trained only on construction data,
selected on model-selection data, refit on construction plus calibration, and
then frozen before held-out access.  It consumes predicted rollout features,
actions, and the registered initial state, never held-out event labels.

The complete oracle ceiling passes only if every optimization seed in both
JEPA and DINO meets the registered contact-tail, p95, mean-noninferiority,
matched-control, leave-one-family, and physical-effect-alignment gates.
The learned event-memory/jump experiment is authorized only if event support,
oracle reset headroom, and every frozen guard-information gate pass for both
models and every optimization seed.  Passing does not establish a learned
repair, causal identification, planning value, a Koopman representation, a
minimal predictive-state dimension, or deployment reliability.  Planning
remains sealed.  The hybrid error propagation expression
`sum_k epsilon_k prod_{ell>k} L_ell` is recorded as a conditional bound for a
fixed correct mode path; guard/mode-sequence mistakes are reported separately.
'''


configuration = rename(BASE.configuration)
for name, value in {
    "PROTOCOL_ID": '"stage42-action-conditioned-hybrid-defect-v2"',
    "NOTEBOOK_PROTOCOL_SHA256": '"__PROTOCOL_DIGEST__"',
    "EVIDENCE_STATUS": '"FRESH_EVENT_EXCITATION_ORACLE_HEADROOM"',
    "EXPERIMENT_NOTEBOOK_PATH": '"notebooks/42_event_conditioned_oracle_hybrid.ipynb"',
    "EXPERIMENT_BUILDER_PATH": '"notebooks/build_stage42_event_conditioned_oracle_hybrid_notebook.py"',
    "EXPERIMENT_NUMERICAL_PATH": '"src/cf_faithfulness/stage42_event_conditioned_hybrid.py"',
    "OUTPUT_DIR": '"/content/counterfactual_faithfulness_stage42_ecoh_v2"',
    "DRIVE_OUTPUT_DIR": '"/content/drive/MyDrive/counterfactual_faithfulness_stage42_ecoh_v2"',
    "RUN_REQUEST_PATH": '"/content/drive/MyDrive/counterfactual_faithfulness_stage42_ecoh_v2/stage42_run_request.json"',
    "SEED": "420101",
    "DESIGN_SEED": "420141",
    "DECODER_SEED": "420183",
    "RANK_SEED": "420213",
    "CALIBRATION_SEED": "420253",
    "BOOTSTRAP_SEED": "420283",
    "CONTROL_SEED": "420351",
    "CONSTRUCTION_TRAJECTORY_POOL": "list(range(148000, 150000))",
    "MODEL_SELECTION_TRAJECTORY_POOL": "list(range(150000, 152000))",
    "CALIBRATION_TRAJECTORY_POOL": "list(range(152000, 154000))",
    "EVALUATION_TRAJECTORY_POOL": "list(range(154000, 170000))",
    "TASK_ID_OFFSET": "420000",
}.items():
    configuration = replace_assignment(configuration, name, value)
configuration = replace_assignment(
    configuration, "FINAL_TRAINING_SEEDS",
    '[4201, 4202, 4203] if RUN_MODE == "pilot" else [4201]',
)
configuration = replace_block(
    configuration,
    "STAGE39_TOKEN_SPECS = {",
    "\n\n\ndef stage39_word_spec",
    r'''STAGE39_TOKEN_SPECS = {
    # Original Stage 39 coefficient-matched development vocabulary.
    "A": (-45.0, 0.16), "B": (45.0, 0.16),
    "C": (-15.0, 0.20), "D": (15.0, 0.20),
    # Stage 42 v2 prospective event-excitation additions. P approaches the
    # block, L/T provide mirrored transverse approaches, S retreats, and 0
    # holds the action at zero while the contact gap remains open.
    "P": (0.0, 0.20), "Q": (0.0, 0.14),
    "L": (-35.0, 0.18), "T": (35.0, 0.18),
    "S": (180.0, 0.14), "0": (0.0, 0.0),
}
EVENT_EXCITATION_TOKEN_NAMES = ["P", "Q", "L", "T", "S", "0"]
STAGE42_V1_SUPPORT_AUDIT = {
    "role": "development_support_falsification_only",
    "result_bundle_sha256": "72c1b2c24495cfb189f9014a4a061897b582fe27ef12a2b95465145f30521ebd",
    "evaluation_pool_start": 132000,
    "evaluation_pool_stop": 148000,
    "families_screened": 16000,
    "event_rich_families": 0,
    "model_outputs_read": False,
}
SUPPORT_CHECKPOINT_INTERVAL = 64
''',
)
configuration = replace_block(
    configuration,
    "CANONICAL_RESPONSE_WORD_NAMES = [",
    "CALIBRATION_INTERCHANGE_PAIRS =",
    r'''CANONICAL_RESPONSE_WORD_NAMES = ["A", "B", "C", "D", "AB", "CD", "BA", "DC"]
CONSTRUCTION_WORD_NAMES = [
    "PPPSSSPPP", "PPQSSSPPP",
    "PPPSSS0PPP", "PPQSSS0PPP",
    "PPPSSS00PPP", "PPQSSS00PPP",
    "PPPSSS000PPP", "PPQSSS000PPP",
]
MODEL_SELECTION_WORD_NAMES = [
    "PPLSSSPPP", "PPTSSSPPP",
    "PPLSSS0PPP", "PPTSSS0PPP",
    "PPLSSS00PPP", "PPTSSS00PPP",
    "PPLSSS000PPP", "PPTSSS000PPP",
]
CALIBRATION_WORD_NAMES = [
    "PLPSSSPPP", "PTPSSSPPP",
    "PLPSSS0PPP", "PTPSSS0PPP",
    "PLPSSS00PPP", "PTPSSS00PPP",
    "PLPSSS000PPP", "PTPSSS000PPP",
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
    "LPPSSSPPP", "TPPSSSPPP",
    "LPPSSS0PPP", "TPPSSS0PPP",
    "LPPSSS00PPP", "TPPSSS00PPP",
    "LPPSSS000PPP", "TPPSSS000PPP",
]
PLANNING_WORD_NAMES = []
EVALUATION_WORD_NAMES_REGISTERED = list(CLOSURE_EVALUATION_WORD_NAMES)
EVALUATION_WORD_SPECS = [
    stage39_word_spec(name) for name in EVALUATION_WORD_NAMES_REGISTERED
]
''',
)
configuration = replace_assignment(configuration, "MIN_REENTRY_ROWS", "32")
configuration = configuration.replace("assert MIN_REENTRY_ROWS == 32\n", "")
configuration = configuration.replace(
    '"fresh_trajectory_ids_114000_to_125999",',
    '"fresh_trajectory_ids_148000_to_169999",\n'
    '    "stage42_v1_zero_of_16000_development_support",\n'
    '    "prospective_event_excitation_vocabulary",\n'
    '    "evaluation_conditioned_on_reentry_incidence_only",',
)
configuration = re.sub(r"^PROTOCOL_CONFIG_KEYS = \[.*\]\n?", "", configuration, flags=re.M)
configuration += r'''

GUARD_THRESHOLDS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
MIN_GUARD_AUROC = 0.70
MIN_GUARD_BALANCED_ACCURACY = 0.65
MAX_GUARD_BRIER = 0.20
MAX_EVENT_TIME_MAE = 0.25
assert MIN_REENTRY_ROWS == 32
assert MIN_GUARD_AUROC == 0.70
assert MIN_GUARD_BALANCED_ACCURACY == 0.65
assert MAX_GUARD_BRIER == 0.20
assert MAX_EVENT_TIME_MAE == 0.25
'''
configuration += "\n\nPROTOCOL_CONFIG_KEYS = " + repr(
    assigned_uppercase_names(configuration)
) + "\n"


installation = BASE.installation
setup = rename(BASE.setup).replace("stage42_cerh_v3", "stage42_ecoh_v2")
analysis_helpers = rename(BASE.analysis_helpers)
support_helpers = function_sources(
    NUMERICAL.read_text(),
    [
        "macro_contact_flags", "terminal_mode_pair_from_contacts",
        "family_reentry_count", "select_event_rich_families",
        "Stage42SupportDecision", "derive_stage42_support_decision",
        "binary_auroc", "balanced_accuracy", "select_guard_threshold",
        "guard_probe_metrics", "partition_hybrid_defects",
        "propagated_hybrid_error_bound", "Stage42DefectDecision",
        "derive_stage42_defect_decision",
    ],
)
support_helpers = support_helpers.replace(
    "class Stage42SupportDecision:\n",
    "@dataclass(frozen=True)\nclass Stage42SupportDecision:\n",
)
support_helpers = support_helpers.replace(
    "class Stage42DefectDecision:\n",
    "@dataclass(frozen=True)\nclass Stage42DefectDecision:\n",
)
analysis_helpers += "\n\n" + support_helpers

model_helpers = BASE.model_helpers
design_and_runtime_helpers = BASE.design_and_runtime_helpers
physical_truth = rename(BASE.physical_truth)

support_selection = r'''

def stage42_candidate_reentry_count(records):
    total = 0
    for record in records:
        # Use the canonical specifications constructed by canonical_word_specs().
        # The configuration-level rows intentionally contain only the registered
        # angles and magnitudes; WORD_BY_NAME is where derived invariants such as
        # ``length`` are validated and attached for rollout helpers.
        for name in EVALUATION_WORD_NAMES:
            specification = WORD_BY_NAME[name]
            rollout = rollout_word(record, specification, retain_visual=False)
            pair = terminal_mode_pair_from_contacts(
                rollout["contacts"], length=int(specification["length"]),
                frameskip=FRAMESKIP, initial_mode=str(record["mode"]),
            )
            total += int(pair == ("post_contact", "contact"))
    return int(total)


def select_stage42_event_rich_evaluation(pool, target):
    path = DESIGN_DIR / "selected_evaluation_trajectories.json"
    certificate_path = DESIGN_DIR / "stage42_event_support_certificate.json"
    partial_path = DESIGN_DIR / "stage42_event_support_screen_partial.json"
    pool_sha256 = hashlib.sha256(
        json.dumps(list(pool), separators=(",", ":")).encode()
    ).hexdigest()
    if path.is_file() or certificate_path.is_file():
        if not path.is_file() or not certificate_path.is_file():
            raise RuntimeError("partial Stage 42 support cache")
        validate_digest_sidecar(path)
        validate_digest_sidecar(certificate_path)
        payload = json.loads(path.read_text())
        certificate = json.loads(certificate_path.read_text())
        decision = derive_stage42_support_decision(
            certificate, expected_families=int(target),
            minimum_reentry_rows=MIN_REENTRY_ROWS,
        )
        if (
            payload.get("protocol_id") != PROTOCOL_ID
            or payload.get("pool") != list(pool)
            or payload.get("target") != int(target)
            or not decision.passed
        ):
            raise RuntimeError("stale Stage 42 event-support cache")
        records = [restore_json_record(row) for row in payload["records"]]
        if {int(row["trajectory_id"]) for row in records} != set(
            map(int, certificate["trajectory_ids"])
        ):
            raise RuntimeError("Stage 42 support certificate/record mismatch")
        PROVENANCE_COUNTS["validated_cache_hits"] += 2
        return records

    candidate_ids, candidate_counts = [], []
    if partial_path.is_file():
        partial = json.loads(partial_path.read_text())
        if (
            partial.get("protocol_id") != PROTOCOL_ID
            or partial.get("run_signature") != RUN_SIGNATURE
            or partial.get("candidate_pool_sha256") != pool_sha256
            or int(partial.get("candidate_pool_size", -1)) != len(pool)
            or int(partial.get("target_families", -1)) != int(target)
        ):
            raise RuntimeError("stale Stage 42 partial support screen")
        candidate_ids = [int(value) for value in partial["candidate_ids"]]
        candidate_counts = [int(value) for value in partial["candidate_counts"]]
        if candidate_ids != list(map(int, pool[: len(candidate_ids)])):
            raise RuntimeError("Stage 42 partial support screen is not a pool prefix")
        PROVENANCE_COUNTS["validated_cache_hits"] += 1
    records_by_id = {}
    for trajectory_id in pool[len(candidate_ids):]:
        base = initial_trajectory_record(trajectory_id, "evaluation", list(pool))
        snapshots = trajectory_mode_snapshots(base)
        PROVENANCE_COUNTS["trajectory_candidates_screened"] += 1
        if snapshots is None:
            count = 0
            complete = False
        else:
            complete = True
            count = stage42_candidate_reentry_count(snapshots)
            records_by_id[int(trajectory_id)] = snapshots
        candidate_ids.append(int(trajectory_id))
        candidate_counts.append(int(count))
        event_rich_families = int(sum(value > 0 for value in candidate_counts))
        checkpoint_due = bool(
            len(candidate_ids) % SUPPORT_CHECKPOINT_INTERVAL == 0
            or event_rich_families >= int(target)
            or len(candidate_ids) == len(pool)
        )
        if checkpoint_due:
            write_json(OUT / "physical_screen_evaluation_progress.json", {
                "screened": len(candidate_ids), "pool": len(pool),
                "event_rich_families": event_rich_families,
                "target": int(target), "last_trajectory_id": int(trajectory_id),
            })
            write_json(partial_path, {
                "protocol_id": PROTOCOL_ID, "run_signature": RUN_SIGNATURE,
                "candidate_pool_sha256": pool_sha256,
                "candidate_pool_size": len(pool), "target_families": int(target),
                "candidate_ids": candidate_ids,
                "candidate_counts": candidate_counts,
                "complete": False,
                "screening_observable": "simulator_contact_incidence_only",
            })
        if event_rich_families >= int(target):
            break
    screen_rows = [
        {
            "split": "evaluation", "trajectory_id": trajectory_id,
            "complete_four_mode_family": bool(trajectory_id in records_by_id),
            "terminal_reentry_rows": count,
            "selected": False,
            "model_outputs_used": False,
            "prediction_errors_used": False,
            "effect_magnitudes_used": False,
        }
        for trajectory_id, count in zip(candidate_ids, candidate_counts)
    ]
    certificate = select_event_rich_families(
        candidate_ids, candidate_counts, target_families=int(target),
        minimum_total_rows=MIN_REENTRY_ROWS,
    )
    certificate.update({
        "protocol_id": PROTOCOL_ID, "run_signature": RUN_SIGNATURE,
        "split": "evaluation", "candidate_pool": list(pool),
        "screening_observable": "simulator_contact_incidence_only",
        "created_before_model_loading": True,
        "trajectory_geometry_version": TRAJECTORY_GEOMETRY_VERSION,
        "conditional_estimand": "event_rich_finite_bank",
    })
    selected_ids = set(map(int, certificate["trajectory_ids"]))
    for row in screen_rows:
        row["selected"] = int(row["trajectory_id"]) in selected_ids
    write_csv(EVIDENCE_DIR / "physical_screen_evaluation_rows.csv", screen_rows)
    write_json(certificate_path, certificate)
    write_digest_sidecar(certificate_path)
    decision = derive_stage42_support_decision(
        certificate, expected_families=int(target),
        minimum_reentry_rows=MIN_REENTRY_ROWS,
    )
    if not decision.passed:
        raise RuntimeError(decision.classification)
    # On a resumed screen, reconstruct only the selected families and verify
    # their deterministic support counts before opening any model checkpoint.
    selected = []
    registered_counts = certificate["family_reentry_counts"]
    for trajectory_id in certificate["trajectory_ids"]:
        snapshots = records_by_id.get(int(trajectory_id))
        if snapshots is None:
            base = initial_trajectory_record(trajectory_id, "evaluation", list(pool))
            snapshots = trajectory_mode_snapshots(base)
        if snapshots is None:
            raise RuntimeError("selected Stage 42 family no longer has four modes")
        observed = stage42_candidate_reentry_count(snapshots)
        if observed != int(registered_counts[str(int(trajectory_id))]):
            raise RuntimeError("selected Stage 42 family support changed on resume")
        selected.extend(snapshots)
    payload = {
        "protocol_id": PROTOCOL_ID, "split": "evaluation", "pool": list(pool),
        "target": int(target),
        "selection_rule": certificate["selection_rule"],
        "selection_uses_simulator_contact_incidence_only": True,
        "model_outputs_used": False, "prediction_errors_used": False,
        "effect_magnitudes_used": False,
        "support_certificate_sha256": sha256_file(certificate_path),
        "trajectory_geometry_version": TRAJECTORY_GEOMETRY_VERSION,
        "records": [json_record(row) for row in selected],
    }
    write_json(path, payload)
    write_digest_sidecar(path)
    write_json(partial_path, {
        "protocol_id": PROTOCOL_ID, "run_signature": RUN_SIGNATURE,
        "candidate_pool_sha256": pool_sha256,
        "candidate_pool_size": len(pool), "target_families": int(target),
        "candidate_ids": candidate_ids, "candidate_counts": candidate_counts,
        "complete": True,
        "certificate_sha256": sha256_file(certificate_path),
        "screening_observable": "simulator_contact_incidence_only",
    })
    PROVENANCE_COUNTS["trajectory_families_selected"] += int(target)
    return selected
'''
physical_truth = physical_truth.replace(
    "\ndef select_complete_trajectories(split, pool, target):",
    support_selection + "\n\ndef select_complete_trajectories(split, pool, target):",
    1,
)
physical_truth = physical_truth.replace(
    'def select_complete_trajectories(split, pool, target):\n    path =',
    'def select_complete_trajectories(split, pool, target):\n'
    '    if str(split) == "evaluation":\n'
    '        return select_stage42_event_rich_evaluation(pool, target)\n'
    '    path =',
    1,
)
physical_truth = physical_truth.replace(
    '"selection_uses_contact_timing_only": True,\n            "model_outputs_used": False,',
    '"selection_uses_simulator_contact_structure_only": True,\n'
    '            "evaluation_reentry_incidence_used": True,\n'
    '            "model_outputs_used": False,',
    1,
)

simulator_preflight = BASE.simulator_preflight
construction_and_paths = BASE.construction_and_paths
data_and_selection = rename(BASE.data_and_selection)
causal_interventions = rename(BASE.causal_interventions)
calibration = rename(BASE.calibration)
calibration = calibration.replace(
    "PHYSICAL_SCALES = {}\nHEADS = {}\nHEAD_SELECTION_ROWS = []",
    "PHYSICAL_SCALES = {}\nSTATE_SCALES = {}\nHEADS = {}\nGUARD_PROBES = {}\n"
    "HEAD_SELECTION_ROWS = []\nGUARD_SELECTION_ROWS = []",
    1,
)
guard_probe_helpers = r'''

def stage42_guard_design(base, artifact):
    values = np.asarray(base, dtype=np.float64)
    return (values - artifact["base_mean"]) / artifact["base_scale"]


def fit_stage42_guard_probe(short, seed, construction, validation, final_fit):
    # The two regression outputs are event occurrence and normalized first-event
    # time.  No held-out labels or errors participate in fitting or selection.
    selection_mean, selection_scale = mean_scale(construction["base"])
    selection_artifact = {
        "base_mean": selection_mean, "base_scale": selection_scale,
    }
    train_design = stage42_guard_design(construction["base"], selection_artifact)
    validation_design = stage42_guard_design(validation["base"], selection_artifact)
    selection = select_ridge_penalty(
        train_design, construction["metadata"][:, :2], validation_design,
        validation["metadata"][:, :2], RIDGE_PENALTIES,
    )
    final_mean, final_scale = mean_scale(final_fit["base"])
    artifact = {
        "model": short, "seed": int(seed), "base_mean": final_mean,
        "base_scale": final_scale,
        "selected_penalty": float(selection["selected_penalty"]),
        "selection_rows": selection["candidate_rows"],
        "nominal_design_width": int(final_fit["base"].shape[1]),
        "evaluation_rows_used": 0,
        "features": "frozen_predicted_physical_state_action_and_initial_physical",
        "targets": ["event_occurrence", "normalized_first_event_time"],
    }
    final_design = stage42_guard_design(final_fit["base"], artifact)
    fitted = fit_ridge(
        final_design, final_fit["metadata"][:, :2], artifact["selected_penalty"]
    )
    artifact["weight"] = fitted["weight"]
    artifact["intercept"] = fitted["intercept"]
    frozen_validation_score = ridge_predict(
        artifact, stage42_guard_design(validation["base"], artifact)
    )[:, 0]
    threshold = select_guard_threshold(
        validation["metadata"][:, 0], frozen_validation_score, GUARD_THRESHOLDS
    )
    artifact["selected_threshold"] = float(threshold["selected_threshold"])
    artifact["threshold_selection_rows"] = threshold["candidate_rows"]
    return artifact
'''
calibration = calibration.replace(
    "\nif not PIPELINE_FAILED and SIMULATOR_PREFLIGHT_PASSED:",
    guard_probe_helpers + "\n\nif not PIPELINE_FAILED and SIMULATOR_PREFLIGHT_PASSED:",
    1,
)
calibration = calibration.replace(
    "FROZEN_MODELS[short], HEADS[short] = {}, {}",
    "FROZEN_MODELS[short], HEADS[short] = {}, {}\n"
    "            STATE_SCALES[short], GUARD_PROBES[short] = {}, {}",
    1,
)
calibration = calibration.replace(
    "FROZEN_MODELS[short][int(seed)] = base\n"
    "                stage42_phase(\"base_fit_complete\", model=short, seed=int(seed))",
    "FROZEN_MODELS[short][int(seed)] = base\n"
    "                scale_rollout = rollout_predictive_state_closure(\n"
    "                    base, base_fit[\"initial_carrier\"], base_fit[\"actions\"],\n"
    "                    base_fit[\"carrier\"], base_fit[\"mask\"],\n"
    "                )\n"
    "                scale_valid = np.asarray(scale_rollout[\"evaluation_mask\"], dtype=bool)\n"
    "                STATE_SCALES[short][int(seed)] = np.maximum(\n"
    "                    np.std(scale_rollout[\"direct_state\"][scale_valid], axis=0, ddof=1),\n"
    "                    1e-8,\n"
    "                )\n"
    "                stage42_phase(\"base_fit_complete\", model=short, seed=int(seed))",
    1,
)
calibration = calibration.replace(
    "HEADS[short][int(seed)] = {}\n                for variant in CAUSAL_VARIANTS:",
    "HEADS[short][int(seed)] = {}\n"
    "                guard_array, guard_schema = stage42_artifact_paths(\n"
    "                    short, \"guard_probe\", seed\n"
    "                )\n"
    "                if all(path.is_file() for path in [\n"
    "                    guard_array, guard_schema, Path(str(guard_array) + \".sha256\"),\n"
    "                    Path(str(guard_schema) + \".sha256\"),\n"
    "                ]):\n"
    "                    guard = load_stage42_artifact(short, \"guard_probe\", seed)\n"
    "                else:\n"
    "                    guard = fit_stage42_guard_probe(\n"
    "                        short, int(seed), construction_flat, validation_flat, final_flat\n"
    "                    )\n"
    "                    save_stage42_artifact(short, \"guard_probe\", seed, guard)\n"
    "                GUARD_PROBES[short][int(seed)] = guard\n"
    "                GUARD_SELECTION_ROWS.extend({\n"
    "                    \"model\": short, \"seed\": int(seed), **row,\n"
    "                    \"selected\": bool(\n"
    "                        float(row[\"penalty\"]) == float(guard[\"selected_penalty\"])\n"
    "                    ),\n"
    "                } for row in guard[\"selection_rows\"])\n"
    "                model_manifest.append({\n"
    "                    \"model\": short, \"seed\": int(seed), \"variant\": \"guard_probe\",\n"
    "                    \"array_sha256\": sha256_file(guard_array),\n"
    "                    \"schema_sha256\": sha256_file(guard_schema),\n"
    "                })\n"
    "                for variant in CAUSAL_VARIANTS:",
    1,
)
calibration = calibration.replace(
    'write_csv(EVIDENCE_DIR / "stage42_head_selection_rows.csv", HEAD_SELECTION_ROWS)',
    'write_csv(EVIDENCE_DIR / "stage42_head_selection_rows.csv", HEAD_SELECTION_ROWS)\n'
    '        write_csv(EVIDENCE_DIR / "stage42_guard_selection_rows.csv", GUARD_SELECTION_ROWS)',
    1,
)
calibration = calibration.replace(
    'f"physical_{short}": value for short, value in PHYSICAL_SCALES.items()\n        })',
    'f"physical_{short}": value for short, value in PHYSICAL_SCALES.items()\n'
    '        }, **{\n'
    '            f"state_{short}_seed{int(seed)}": value\n'
    '            for short, panel in STATE_SCALES.items() for seed, value in panel.items()\n'
    '        })',
    1,
)
calibration = calibration.replace(
    '"equal_nominal_head_width": True,',
    '"equal_nominal_head_width": True,\n'
    '            "guard_probes_frozen": True,\n'
    '            "guard_thresholds_selected_on_model_selection_only": True,\n'
    '            "state_scales_frozen_before_evaluation": True,',
    1,
)
calibration = calibration.replace(
    '"evaluation_statistics_read": False,\n            "evaluation_pairs_generated": False,',
    '"evaluation_contact_incidence_screened": True,\n'
    '            "evaluation_prediction_errors_read": False,\n'
    '            "evaluation_effect_magnitudes_read": False,\n'
    '            "evaluation_statistics_read": False,\n'
    '            "evaluation_pairs_generated": False,',
    1,
)

heldout_evaluation = rename(BASE.heldout_evaluation)
heldout_evaluation = heldout_evaluation.replace(
    'EVALUATION_ROWS = []\nSUMMARY = {}\nPANEL_DECISIONS = {}',
    'EVALUATION_ROWS = []\nSUMMARY = {}\nPANEL_DECISIONS = {}\n'
    'STAGE42_SUPPORT_CERTIFICATE = None',
    1,
)
guard_evaluation_helper = r'''

def stage42_guard_prediction(short, seed, base_tensor, valid):
    artifact = GUARD_PROBES[short][int(seed)]
    design = stage42_guard_design(np.asarray(base_tensor)[valid], artifact)
    return ridge_predict(artifact, design)
'''
heldout_evaluation = heldout_evaluation.replace(
    "\ndef stage42_leave_one_family_tail_gain(",
    guard_evaluation_helper + "\n\ndef stage42_leave_one_family_tail_gain(",
    1,
)
heldout_evaluation = heldout_evaluation.replace(
    'reentry = (source_terminal == "post_contact") & (target_terminal == "contact")',
    'reentry = (source_terminal == "post_contact") & (target_terminal == "contact")\n'
    '    path_event = np.any(\n'
    '        (pairs["metadata"][:, :, 0] > 0.5) & np.asarray(data["mask"], dtype=bool),\n'
    '        axis=1,\n'
    '    )',
    1,
)
heldout_evaluation = heldout_evaluation.replace(
    'all_errors = {variant: [] for variant in ["baseline"] + CAUSAL_VARIANTS}',
    'all_errors = {variant: [] for variant in ["baseline"] + CAUSAL_VARIANTS}\n'
    '    all_composition_errors = []',
    1,
)
heldout_evaluation = heldout_evaluation.replace(
    'all_errors["baseline"].append(baseline_error)',
    'all_errors["baseline"].append(baseline_error)\n'
    '        composition_error = scaled_path_mse(\n'
    '            rollout["state"], rollout["direct_state"], rollout["evaluation_mask"],\n'
    '            STATE_SCALES[short][int(seed)],\n'
    '        )\n'
    '        all_composition_errors.append(composition_error)',
    1,
)
heldout_evaluation = heldout_evaluation.replace(
    'reset_error = all_errors["oracle_reset_ceiling"][-1]',
    'reset_error = all_errors["oracle_reset_ceiling"][-1]\n'
    '        guard_score = stage42_guard_prediction(short, seed, base_tensor, valid)\n'
    '        guard = GUARD_PROBES[short][int(seed)]\n'
    '        guard_metrics = guard_probe_metrics(\n'
    '            pairs["metadata"][valid][:, 0], guard_score[:, 0],\n'
    '            pairs["metadata"][valid][:, 1], guard_score[:, 1],\n'
    '            threshold=float(guard["selected_threshold"]),\n'
    '        )\n'
    '        guard_identifiable = bool(\n'
    '            guard_metrics["event_auroc"] >= MIN_GUARD_AUROC\n'
    '            and guard_metrics["event_balanced_accuracy"] >= MIN_GUARD_BALANCED_ACCURACY\n'
    '            and guard_metrics["event_brier"] <= MAX_GUARD_BRIER\n'
    '            and guard_metrics["event_time_mae"] <= MAX_EVENT_TIME_MAE\n'
    '        )\n'
    '        defects = partition_hybrid_defects(\n'
    '            composition_error, baseline_error, reset_error, path_event, reentry\n'
    '        )',
    1,
)
heldout_evaluation = heldout_evaluation.replace(
    '"causal_alignment": alignment, "gates": gates,',
    '"causal_alignment": alignment, "guard_probe": guard_metrics,\n'
    '            "guard_identifiable": guard_identifiable,\n'
    '            "hybrid_defects": defects, "gates": gates,',
    1,
)
heldout_evaluation = heldout_evaluation.replace(
    "This development-only audit compares exact simulator event/reset metadata with\n"
    "equal-width smooth and shuffled controls on top of a frozen recursive model.\n"
    "A positive result is headroom evidence only and authorizes a separate\n"
    "label-free identifiability experiment.  It is not a learned causal, planning,\n"
    "or deployment result.  Planning remained sealed.",
    "This prospective event-rich audit reports flow composition, frozen-feature\n"
    "guard identifiability, and oracle reset headroom separately.  A learned\n"
    "event-reset experiment is authorized only when all three registered gates\n"
    "support it across JEPA, DINO, and every seed.  It is not a learned causal,\n"
    "planning, or deployment result.  Planning remained sealed.",
    1,
)
heldout_evaluation = heldout_evaluation.replace(
    '"panels_pooled": False,',
    '"all_seed_guard_identifiable": bool(\n'
    '            all(row["guard_identifiable"] for row in seed_summaries)\n'
    '        ),\n'
    '        "flow_guard_reset_reported_separately": True,\n'
    '        "panels_pooled": False,',
    1,
)
heldout_evaluation = heldout_evaluation.replace(
    'DECISION_PAYLOAD = derive_stage42_decision(PANEL_DECISIONS)\n        DECISION_PAYLOAD.update({',
    'DECISION_PAYLOAD = derive_stage42_decision(PANEL_DECISIONS)\n'
    '        oracle_headroom_passed = bool(DECISION_PAYLOAD["passed"])\n'
    '        frozen_guard_identifiable = bool(all(\n'
    '            SUMMARY[short]["all_seed_guard_identifiable"] for short in ["jepa", "dino"]\n'
    '        ))\n'
    '        defect_decision = derive_stage42_defect_decision(\n'
    '            support_certified=bool(support_decision.passed),\n'
    '            oracle_reset_headroom=oracle_headroom_passed,\n'
    '            frozen_guard_identifiable=frozen_guard_identifiable,\n'
    '        )\n'
    '        oracle_status = DECISION_PAYLOAD["status"]\n'
    '        DECISION_PAYLOAD.update({\n'
    '            "oracle_headroom_status": oracle_status,\n'
    '            "oracle_headroom_passed": oracle_headroom_passed,\n'
    '            "status": defect_decision.classification,\n'
    '            "passed": defect_decision.passed,\n'
    '            "next_step": defect_decision.classification,\n'
    '            "frozen_guard_identifiable": frozen_guard_identifiable,\n'
    '            "learned_event_reset_experiment_authorized": (\n'
    '                defect_decision.learned_event_reset_experiment_authorized\n'
    '            ),\n'
    '            "flow_guard_reset_reported_separately": True,\n'
    '            "fixed_mode_path_bound_only": True,\n'
    '        })\n'
    '        DECISION_PAYLOAD.update({',
    1,
)
heldout_evaluation = heldout_evaluation.replace(
    'if int(np.sum(reentry)) < MIN_REENTRY_ROWS:\n'
    '        raise RuntimeError("heldout Stage 42 panel has too few post-contact re-entry rows")',
    'observed_reentry_rows = int(np.sum(reentry))\n'
    '    if observed_reentry_rows < MIN_REENTRY_ROWS:\n'
    '        raise RuntimeError("heldout Stage 42 panel has too few post-contact re-entry rows")\n'
    '    if observed_reentry_rows != int(STAGE42_SUPPORT_CERTIFICATE["total_reentry_rows"]):\n'
    '        raise RuntimeError("Stage 42 support count changed after certificate freeze")',
    1,
)
heldout_evaluation = heldout_evaluation.replace(
    'certificate_path = CALIBRATION_MODEL_DIR / "evaluation_open_certificate.json"\n'
    '        validate_digest_sidecar(certificate_path)',
    'certificate_path = CALIBRATION_MODEL_DIR / "evaluation_open_certificate.json"\n'
    '        validate_digest_sidecar(certificate_path)\n'
    '        support_path = DESIGN_DIR / "stage42_event_support_certificate.json"\n'
    '        validate_digest_sidecar(support_path)\n'
    '        STAGE42_SUPPORT_CERTIFICATE = json.loads(support_path.read_text())\n'
    '        support_decision = derive_stage42_support_decision(\n'
    '            STAGE42_SUPPORT_CERTIFICATE,\n'
    '            expected_families=ACTIVE_EVALUATION_TRAJECTORIES,\n'
    '            minimum_reentry_rows=MIN_REENTRY_ROWS,\n'
    '        )\n'
    '        if not support_decision.passed:\n'
    '            raise RuntimeError(support_decision.classification)',
    1,
)
heldout_evaluation = heldout_evaluation.replace(
    '"evidence_tier": "development_only_oracle_headroom",',
    '"evidence_tier": "fresh_event_conditioned_oracle_headroom",\n'
    '            "conditional_estimand": "event_rich_finite_bank",\n'
    '            "support_certificate_sha256": sha256_file(\n'
    '                DESIGN_DIR / "stage42_event_support_certificate.json"\n'
    '            ),',
    1,
)

packaging = rename(BASE.packaging).replace("stage42_cerh_v3", "stage42_ecoh_v2")
packaging = packaging.replace(
    "causal_event_reset_headroom", "event_conditioned_oracle_hybrid"
)


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
    raise RuntimeError("Stage 42 protocol digest placeholder was not replaced")
protocol_sources[1] = configuration

cells = [markdown(introduction)] + [code(value) for value in protocol_sources[1:]]
for index, cell in enumerate(cells):
    cell["id"] = f"stage42-{index:02d}"
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
