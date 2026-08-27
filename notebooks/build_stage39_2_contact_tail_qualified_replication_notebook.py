"""Build the fresh, tail-qualified Stage 39.2 PushT replication Colab."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "39_2_contact_tail_qualified_replication.ipynb"


def load_builder(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BASE = load_builder(
    ROOT / "build_stage39_fresh_coefficient_replication_notebook.py",
    "stage39_builder_for_stage392",
)

code = BASE.code
markdown = BASE.markdown
replace_assignment = BASE.replace_assignment
replace_block = BASE.replace_block
assigned_uppercase_names = BASE.assigned_uppercase_names


introduction = r'''# Stage 39.2: contact-tail-qualified coefficient replication

## Frozen decision before computation

Stage 39 replicated practical mean equivalence between full S-PSCD and an
exactly coefficient-matched latent-only objective on a fresh PushT bank.  Its
registered absolute tail thresholds were reported but were not connected to a
separate reliability decision.  A shared free-to-contact row also produced a
large absolute error for every model and objective.  Stage 39 remains an
immutable result; this notebook does not recompute, overwrite, or reclassify
its exposed evaluation bank.

Stage 39.2 is a new prospective replication using fresh trajectory IDs, fresh
action words, fresh training seeds, fresh fitted adapters, and a new output
root.  It registers two conclusions that must never be conflated:

1. **Comparative mean equivalence.**  For JEPA and DINO separately, the 90%
   hierarchical interval for the mean paired row-wise full-minus-matched gain
   must lie inside the fixed `[-5%, +5%]` band.
2. **Absolute tail qualification.**  For both objectives and every training
   seed, registered p95, catastrophic-rate, word-length, initial-mode,
   terminal-mode, and contact/post-contact bounds must all pass.

The comparative conclusion remains interpretable if the tail gate fails, but
the run is not called absolutely reliable.  The overall pass is conjunctive.
The hierarchical bootstrap resamples optimization seeds and complete
trajectory families.  JEPA and DINO are never pooled.  Planning remains
sealed, official checkpoints remain frozen, and no Stage 39 evaluation row or
outcome is consumed.
'''


configuration = BASE.configuration
for name, value in {
    "PROTOCOL_ID": '"stage39.2-contact-tail-qualified-replication-v1"',
    "NOTEBOOK_PROTOCOL_SHA256": '"__PROTOCOL_DIGEST__"',
    "EVIDENCE_STATUS": '"FRESH_PROSPECTIVE_TAIL_QUALIFIED_REPLICATION"',
    "EXPERIMENT_NOTEBOOK_PATH": '"notebooks/39_2_contact_tail_qualified_replication.ipynb"',
    "EXPERIMENT_BUILDER_PATH": '"notebooks/build_stage39_2_contact_tail_qualified_replication_notebook.py"',
    "OUTPUT_DIR": '"/content/counterfactual_faithfulness_stage39_2_ctqr"',
    "DRIVE_OUTPUT_DIR": '"/content/drive/MyDrive/counterfactual_faithfulness_stage39_2_ctqr"',
    "RUN_REQUEST_PATH": '"/content/drive/MyDrive/counterfactual_faithfulness_stage39_2_ctqr/stage39_2_run_request.json"',
    "SEED": "392101",
    "DESIGN_SEED": "392141",
    "DECODER_SEED": "392183",
    "RANK_SEED": "392213",
    "CALIBRATION_SEED": "392253",
    "BOOTSTRAP_SEED": "392283",
    "CONTROL_SEED": "392351",
    "CONSTRUCTION_TRAJECTORY_POOL": "list(range(90000, 92000))",
    "MODEL_SELECTION_TRAJECTORY_POOL": "list(range(92000, 94000))",
    "CALIBRATION_TRAJECTORY_POOL": "list(range(94000, 96000))",
    "EVALUATION_TRAJECTORY_POOL": "list(range(96000, 102000))",
    "TASK_ID_OFFSET": "392000",
}.items():
    configuration = replace_assignment(configuration, name, value)

configuration = replace_assignment(
    configuration,
    "FINAL_TRAINING_SEEDS",
    '[3921, 3922, 3923] if RUN_MODE == "pilot" else [3921, 3922]',
)
configuration = replace_block(
    configuration,
    "CANONICAL_RESPONSE_WORD_NAMES = [",
    "CALIBRATION_INTERCHANGE_PAIRS =",
    r'''CANONICAL_RESPONSE_WORD_NAMES = ["A", "B", "C", "D", "AB", "CD", "BA", "DC"]
CONSTRUCTION_WORD_NAMES = [
    "ABBBBCDAA", "CAACDABBB", "ADDCACBCAC", "DDDCBBDCAA",
    "DBBCBACABDB", "CACBACBBADC", "BDBBBBCABDAD", "ADBACBDCDDCD",
]
MODEL_SELECTION_WORD_NAMES = [
    "BBADDCCAD", "DDCBDCBAB", "CBDDBBAADA", "DCBACBDBBA",
    "DDDCABDDDBD", "BDBBDABDCBD", "CBDACACDDCCD", "CCBCDBBBBDAD",
]
CALIBRATION_WORD_NAMES = [
    "CACCBCDDA", "CDDDCADAD", "ADBAACBBAB", "BDBBDBDCBA",
    "CDCCCCDBBAC", "CCBAABABBCD", "CDABDDBAACAA", "DBBACCDCCBDD",
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
    "DCBDADABD", "DCDABCADD", "ACDADABCDA", "AAABCDDBDD",
    "DBDCCBBACBB", "ADBADBDCAAD", "DDCCACBCCADC", "ABADDCACBADC",
]
PLANNING_WORD_NAMES = []
EVALUATION_WORD_NAMES_REGISTERED = list(CLOSURE_EVALUATION_WORD_NAMES)
EVALUATION_WORD_SPECS = [
    stage39_word_spec(name) for name in EVALUATION_WORD_NAMES_REGISTERED
]
''',
)
configuration = configuration.replace(
    '"fresh_trajectory_ids_66000_to_77999",',
    '"fresh_trajectory_ids_90000_to_101999",',
)
configuration = configuration.replace(
    '"fixed_plus_or_minus_five_percent_equivalence_band",',
    '"fixed_plus_or_minus_five_percent_equivalence_band",\n'
    '    "separate_locked_absolute_tail_qualification",',
)
configuration = re.sub(
    r"^PROTOCOL_CONFIG_KEYS = \[.*\]\n?", "", configuration, flags=re.M
)
configuration += r'''

TAIL_QUALIFICATION_VARIANTS = ["coefficient_matched", "full"]
TAIL_CONTACT_MODES = ["contact", "post_contact"]
assert TAIL_QUALIFICATION_VARIANTS == PRIMARY_VARIANTS
assert set(TAIL_CONTACT_MODES) <= set(MODE_LABELS)
'''
configuration += "\n\nPROTOCOL_CONFIG_KEYS = " + repr(
    assigned_uppercase_names(configuration)
) + "\n"

installation = BASE.installation
setup = BASE.setup.replace("stage39_fcmr", "stage39_2_ctqr")
analysis_helpers = BASE.analysis_helpers
model_helpers = BASE.model_helpers
design_and_runtime_helpers = BASE.design_and_runtime_helpers
physical_truth = BASE.physical_truth
simulator_preflight = BASE.simulator_preflight
construction_and_paths = BASE.construction_and_paths
data_and_selection = BASE.data_and_selection
calibration = BASE.calibration


locked_evaluation = r'''# Open the fresh locked panel and issue separate comparative and tail decisions.
DECISION_PAYLOAD = {
    "status": "INCONCLUSIVE_PIPELINE_FAILURE",
    "passed": False,
    "planning_opened": False,
}
EVALUATION_ROWS = []
SUMMARY = {}
PANEL_DECISIONS = {}


def stage39_terminal_labels(labels, mask):
    values = np.asarray(labels).astype(str)
    valid = np.asarray(mask, dtype=bool)
    index = np.max(np.where(valid, np.arange(valid.shape[1])[None, :], -1), axis=1)
    if np.any(index < 0):
        raise ValueError("each label path needs an endpoint")
    return values[np.arange(len(values)), index]


def stage392_absolute_summary(errors, closure, terminal_modes):
    values = np.asarray(errors, dtype=np.float64)
    initial_modes = np.asarray(closure["initial_mode"]).astype(str)
    lengths = np.asarray(closure["length"], dtype=np.int64)
    contact_rows = np.isin(terminal_modes, TAIL_CONTACT_MODES)
    if not np.any(contact_rows):
        raise RuntimeError("locked panel contains no contact/post-contact endpoint")
    length_means = {
        str(length): float(np.mean(values[lengths == length]))
        for length in sorted(set(lengths.tolist()))
    }
    initial_mode_means = {
        mode: float(np.mean(values[initial_modes == mode]))
        for mode in sorted(set(initial_modes.tolist()))
    }
    terminal_mode_means = {
        mode: float(np.mean(values[terminal_modes == mode]))
        for mode in sorted(set(terminal_modes.tolist()))
    }
    tail = tail_risk_summary(values)
    gates = {
        "word_length_means": bool(
            all(value <= MAX_LENGTH_PHYSICAL_NMSE for value in length_means.values())
        ),
        "initial_mode_means": bool(
            all(value <= MAX_MODE_PHYSICAL_NMSE for value in initial_mode_means.values())
        ),
        "terminal_mode_means": bool(
            all(value <= MAX_MODE_PHYSICAL_NMSE for value in terminal_mode_means.values())
        ),
        "contact_post_contact_mean": bool(
            np.mean(values[contact_rows]) <= MAX_MODE_PHYSICAL_NMSE
        ),
        "p95": bool(tail["p95"] <= MAX_P95_PHYSICAL_NMSE),
        "catastrophic_rate": bool(
            tail["catastrophic_rate_gt_1"] <= MAX_CATASTROPHIC_RATE
        ),
    }
    return {
        "passed": bool(all(gates.values())),
        "gates": gates,
        "length_physical_nmse": length_means,
        "initial_mode_physical_nmse": initial_mode_means,
        "terminal_mode_physical_nmse": terminal_mode_means,
        "contact_post_contact_physical_nmse": float(np.mean(values[contact_rows])),
        "tail": tail,
    }


def stage39_replication_panel(short, closure, simulator_quality_passed):
    scale = PHYSICAL_SCALES[short]
    groups = closure["group"]
    if len(np.unique(groups)) < MIN_EVALUATION_TRAJECTORIES:
        raise RuntimeError("locked replication has too few trajectory families")
    masks, physical = {}, {variant: [] for variant in PRIMARY_VARIANTS}
    false_history_errors = []
    native_history = history_tensor(
        closure["initial_carrier"], closure["carrier"], closure["mask"],
        FIXED_HISTORY_LENGTH,
    )
    for seed in FINAL_TRAINING_SEEDS:
        for variant in PRIMARY_VARIANTS:
            artifact = FROZEN_MODELS[short][variant][int(seed)]
            rollout = rollout_predictive_state_closure(
                artifact, closure["initial_carrier"], closure["actions"],
                closure["carrier"], closure["mask"],
            )
            masks[variant] = rollout["evaluation_mask"]
            physical[variant].append(scaled_path_mse(
                rollout["physical"], closure["simulator"], masks[variant], scale,
            ))
        false_history = permute_past_history(
            native_history, groups, closure["mask"],
            seed=stable_seed(CONTROL_SEED, "stage392_wrong_history", short, int(seed)),
        )
        false_rollout = rollout_predictive_state_closure(
            FROZEN_MODELS[short]["full"][int(seed)],
            closure["initial_carrier"], closure["actions"], closure["carrier"],
            closure["mask"], histories_override=false_history,
        )
        false_history_errors.append(scaled_path_mse(
            false_rollout["physical"], closure["simulator"],
            false_rollout["evaluation_mask"], scale,
        ))
    physical = {key: np.stack(value, axis=0) for key, value in physical.items()}
    false_history_errors = np.stack(false_history_errors, axis=0)
    row_gain = paired_rowwise_relative_gain(
        physical["full"], physical["coefficient_matched"]
    )
    interval90 = hierarchical_seed_family_interval(
        row_gain, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
        seed=stable_seed(BOOTSTRAP_SEED, "stage392_primary", short),
        confidence=PRIMARY_CONFIDENCE,
    )
    mean_gain = float(np.mean(row_gain))
    comparative_decision = derive_stage39_panel_decision(
        mean_gain, interval90, equivalence_margin=EQUIVALENCE_MARGIN,
        quality_control_passed=simulator_quality_passed,
    )
    history_gain = paired_rowwise_relative_gain(
        physical["full"], false_history_errors
    )
    history_interval = hierarchical_seed_family_interval(
        history_gain, groups, draws=ACTIVE_BOOTSTRAP_DRAWS,
        seed=stable_seed(BOOTSTRAP_SEED, "stage392_history", short),
        confidence=PRIMARY_CONFIDENCE,
    )
    terminal_modes = stage39_terminal_labels(closure["target_mode"], closure["mask"])
    seed_summaries = []
    for seed_index, seed in enumerate(FINAL_TRAINING_SEEDS):
        full_error = physical["full"][seed_index]
        matched_error = physical["coefficient_matched"][seed_index]
        absolute = {
            variant: stage392_absolute_summary(
                physical[variant][seed_index], closure, terminal_modes
            )
            for variant in TAIL_QUALIFICATION_VARIANTS
        }
        seed_summaries.append({
            "seed": int(seed),
            "full_mean_nmse": float(np.mean(full_error)),
            "matched_mean_nmse": float(np.mean(matched_error)),
            "mean_rowwise_gain": float(np.mean(row_gain[seed_index])),
            "pooled_ratio_of_means_gain": pooled_ratio_of_means_gain(
                full_error, matched_error
            ),
            "absolute_qualification": absolute,
        })
        for row_index in range(len(groups)):
            EVALUATION_ROWS.append({
                "model": short, "seed": int(seed),
                "trajectory_id": int(groups[row_index]),
                "record_id": int(closure["record_id"][row_index]),
                "initial_mode": str(closure["initial_mode"][row_index]),
                "terminal_mode": str(terminal_modes[row_index]),
                "word": str(closure["word"][row_index]),
                "word_length": int(closure["length"][row_index]),
                "full_physical_nmse": float(full_error[row_index]),
                "coefficient_matched_physical_nmse": float(matched_error[row_index]),
                "full_minus_matched_rowwise_gain": float(row_gain[seed_index, row_index]),
                "wrong_history_physical_nmse": float(false_history_errors[seed_index, row_index]),
                "full_catastrophic_gt_1": bool(full_error[row_index] > 1.0),
                "matched_catastrophic_gt_1": bool(matched_error[row_index] > 1.0),
            })
    tail_passed = bool(all(
        result["passed"]
        for seed in seed_summaries
        for result in seed["absolute_qualification"].values()
    ))
    return comparative_decision, {
        "mean_rowwise_relative_gain": mean_gain,
        "hierarchical_interval90": list(interval90),
        "equivalence_margin": EQUIVALENCE_MARGIN,
        "pooled_ratio_of_means_gain": pooled_ratio_of_means_gain(
            physical["full"], physical["coefficient_matched"]
        ),
        "full_mean_nmse": float(np.mean(physical["full"])),
        "coefficient_matched_mean_nmse": float(np.mean(physical["coefficient_matched"])),
        "history_gain": float(np.mean(history_gain)),
        "history_interval90": list(history_interval),
        "seed_summaries": seed_summaries,
        "comparative_classification": comparative_decision.classification,
        "absolute_tail_qualification_passed": tail_passed,
        "absolute_tail_thresholds": {
            "max_length_physical_nmse": MAX_LENGTH_PHYSICAL_NMSE,
            "max_mode_physical_nmse": MAX_MODE_PHYSICAL_NMSE,
            "max_p95_physical_nmse": MAX_P95_PHYSICAL_NMSE,
            "max_catastrophic_rate": MAX_CATASTROPHIC_RATE,
        },
    }


if not PIPELINE_FAILED and SIMULATOR_PREFLIGHT_PASSED:
    try:
        verify_executed_notebook_through(
            "# Open the fresh locked panel and issue separate comparative and tail decisions."
        )
        certificate_path = CALIBRATION_MODEL_DIR / "evaluation_open_certificate.json"
        validate_digest_sidecar(certificate_path)
        certificate = json.loads(certificate_path.read_text())
        if (
            certificate["protocol_id"] != PROTOCOL_ID
            or certificate["run_signature"] != RUN_SIGNATURE
            or certificate["evaluation_statistics_read"]
            or not certificate["planning_permanently_sealed"]
        ):
            raise RuntimeError("Stage 39.2 evaluation-open certificate is invalid")
        for model_name in MODEL_NAMES:
            bundle = load_world_model(model_name)
            short = bundle["short"]
            try:
                for index, record in enumerate(SELECTED_RECORDS["evaluation"]):
                    generate_stage39_path_record(
                        bundle, record, "evaluation_closure", DECODERS[short]
                    )
                    write_json(OUT / f"model_{short}_closure_progress.json", {
                        "completed": index + 1,
                        "total": len(SELECTED_RECORDS["evaluation"]),
                        "last_record_id": int(record["record_id"]),
                    })
            finally:
                unload_world_model(bundle)
        EVALUATION_OPENED = True
        closure_data = {
            short: load_stage39_sequences(short, "evaluation_closure")
            for short in ["jepa", "dino"]
        }
        reference = closure_data["jepa"]
        simulator_rollout = rollout_predictive_state_closure(
            SIMULATOR_FINAL, reference["initial_physical"], reference["actions"],
            reference["simulator"], reference["mask"],
        )
        simulator_error = scaled_path_mse(
            simulator_rollout["physical"], reference["simulator"],
            simulator_rollout["evaluation_mask"],
            SIMULATOR_FINAL["normalization"]["physical_scale"],
        )
        persistence = np.repeat(
            reference["initial_physical"][:, None, :], MAX_WORD_LENGTH, axis=1
        )
        persistence_error = scaled_path_mse(
            persistence, reference["simulator"], simulator_rollout["evaluation_mask"],
            SIMULATOR_FINAL["normalization"]["physical_scale"],
        )
        simulator_gain = float(np.mean(relative_gain(simulator_error, persistence_error)))
        simulator_passed = bool(
            np.mean(simulator_error) <= MAX_SIMULATOR_LOCKED_NMSE
            and simulator_gain >= MIN_SIMULATOR_GAIN
        )
        for short in ["jepa", "dino"]:
            panel, metrics = stage39_replication_panel(
                short, closure_data[short], simulator_passed
            )
            PANEL_DECISIONS[short] = panel
            SUMMARY[short] = metrics
        comparative = derive_stage39_decision(PANEL_DECISIONS)
        tail_passed = bool(all(
            SUMMARY[short]["absolute_tail_qualification_passed"]
            for short in ["jepa", "dino"]
        ))
        comparative_status = str(comparative["status"])
        if comparative["passed"] and tail_passed:
            overall_status = f"{comparative_status}_and_absolute_tail_qualified"
        elif comparative["passed"]:
            overall_status = f"{comparative_status}_but_absolute_tail_unqualified"
        else:
            overall_status = comparative_status
        DECISION_PAYLOAD = {
            **comparative,
            "status": overall_status,
            "passed": bool(comparative["passed"] and tail_passed),
            "comparative_status": comparative_status,
            "comparative_passed": bool(comparative["passed"]),
            "absolute_tail_qualification_passed": tail_passed,
            "protocol_id": PROTOCOL_ID,
            "run_signature": RUN_SIGNATURE,
            "evaluation_opened": True,
            "planning_opened": False,
            "simulator_control": {
                "passed": simulator_passed,
                "mean_nmse": float(np.mean(simulator_error)),
                "gain_over_persistence": simulator_gain,
            },
            "primary_estimand": "mean_paired_rowwise_relative_gain",
            "primary_confidence": PRIMARY_CONFIDENCE,
            "equivalence_band": [-EQUIVALENCE_MARGIN, EQUIVALENCE_MARGIN],
            "panels_pooled": False,
            "stage38_evidence_consumed": False,
            "stage39_evaluation_outcomes_consumed": False,
        }
        write_csv(EVIDENCE_DIR / "locked_replication_rows.csv", EVALUATION_ROWS)
        write_json(EVIDENCE_DIR / "stage39_2_summary.json", SUMMARY)
        write_json(EVIDENCE_DIR / "stage39_2_panel_decisions.json", {
            short: {
                "mean_gain": panel.mean_gain,
                "interval90": list(panel.interval90),
                "equivalence_margin": panel.equivalence_margin,
                "simulator_quality_control_passed": panel.quality_control_passed,
                "comparative_classification": panel.classification,
                "absolute_tail_qualification_passed": SUMMARY[short][
                    "absolute_tail_qualification_passed"
                ],
            }
            for short, panel in PANEL_DECISIONS.items()
        })
        write_json(OUT / "stage39_2_decision.json", DECISION_PAYLOAD)
        atomic_checkpoint("stage39_2_locked_replication_complete", {
            "decision_sha256": sha256_file(OUT / "stage39_2_decision.json"),
            "status": DECISION_PAYLOAD["status"],
            "rows": len(EVALUATION_ROWS), "planning_opened": False,
        })

        figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        for axis, short in zip(axes, ["jepa", "dino"]):
            estimate = SUMMARY[short]["mean_rowwise_relative_gain"]
            low, high = SUMMARY[short]["hierarchical_interval90"]
            axis.errorbar(
                [0], [estimate], yerr=[[estimate - low], [high - estimate]], fmt="o"
            )
            axis.axhspan(
                -EQUIVALENCE_MARGIN, EQUIVALENCE_MARGIN,
                alpha=0.18, color="#0ea5e9",
            )
            axis.axhline(0, color="black", linewidth=1)
            tail_label = (
                "tail pass" if SUMMARY[short]["absolute_tail_qualification_passed"]
                else "tail fail"
            )
            axis.set(
                xticks=[], ylabel="full minus matched relative gain",
                title=f"{short.upper()} ({tail_label})",
            )
        figure.suptitle(f"Stage 39.2: {DECISION_PAYLOAD['status']}")
        figure.tight_layout()
        figure.savefig(PLOT_DIR / "stage39_2_equivalence_and_tail.png", dpi=180)
        plt.close(figure)
        interpretation = f"""# Automatic Stage 39.2 interpretation

Overall status: **{DECISION_PAYLOAD['status'].upper()}**

Comparative status: **{DECISION_PAYLOAD['comparative_status'].upper()}**

Absolute tail qualification: **{'PASS' if tail_passed else 'FAIL'}**

The comparative decision uses separate JEPA and DINO 90% hierarchical
intervals against the fixed ±5% band.  The absolute decision separately checks
every seed and both objectives against the registered word-length, mode,
contact/post-contact, p95, and catastrophic-rate bounds.  Mean equivalence does
not imply pointwise or contact-tail reliability.  Planning remained sealed,
and no Stage 39 evaluation outcome was consumed.
"""
        (OUT / "AUTOMATIC_INTERPRETATION.md").write_text(interpretation)
        print(json.dumps(DECISION_PAYLOAD, indent=2))
    except Exception:
        record_failure("stage39_2_locked_replication")
'''


packaging = BASE.packaging
packaging = packaging.replace("stage39_fcmr", "stage39_2_ctqr")
packaging = packaging.replace(
    "fresh_coefficient_matched_replication",
    "contact_tail_qualified_replication",
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
    raise RuntimeError("Stage 39.2 protocol digest placeholder was not replaced")

protocol_sources[1] = configuration
cells = [markdown(introduction)] + [code(value) for value in protocol_sources[1:]]
for index, cell in enumerate(cells):
    cell["id"] = f"stage392-{index:02d}"

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
