"""Static, split, support, source-binding, and digest validation for Stage 42."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
NOTEBOOK = ROOT / "42_event_conditioned_oracle_hybrid.ipynb"
PRIOR_NOTEBOOKS = [
    ROOT / "38_cross_model_pscd_confirmation.ipynb",
    ROOT / "39_fresh_coefficient_matched_replication.ipynb",
    ROOT / "39_1_wall_cross_environment_replication.ipynb",
    ROOT / "39_2_contact_tail_qualified_replication.ipynb",
    ROOT / "40_contact_tail_risk_distillation.ipynb",
    ROOT / "41_causal_event_reset_headroom.ipynb",
]


def source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def configuration(notebook: dict) -> dict:
    namespace: dict = {}
    exec(compile(source(notebook["cells"][1]), "stage42-config", "exec"), namespace)
    return namespace


def protocol_digest(notebook: dict, config: dict) -> None:
    sources = [source(cell).strip() for cell in notebook["cells"]]
    restored = re.sub(
        r'NOTEBOOK_PROTOCOL_SHA256\s*=\s*"[0-9a-f]{64}"',
        'NOTEBOOK_PROTOCOL_SHA256 = "__PROTOCOL_DIGEST__"',
        sources[1], count=1,
    )
    assert restored != sources[1]
    sources[1] = restored
    expected = hashlib.sha256(
        json.dumps(sources, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    assert config["NOTEBOOK_PROTOCOL_SHA256"] == expected


def banks(config: dict) -> set[str]:
    return set().union(*[
        set(config[name]) for name in [
            "CONSTRUCTION_WORD_NAMES", "MODEL_SELECTION_WORD_NAMES",
            "CALIBRATION_WORD_NAMES", "CLOSURE_EVALUATION_WORD_NAMES",
        ]
    ])


def trajectories(config: dict) -> set[int]:
    return set().union(*[
        set(config[name]) for name in [
            "CONSTRUCTION_TRAJECTORY_POOL", "MODEL_SELECTION_TRAJECTORY_POOL",
            "CALIBRATION_TRAJECTORY_POOL", "EVALUATION_TRAJECTORY_POOL",
        ]
    ])


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] == 5
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert len(notebook["cells"]) == 15
    identifiers = [cell["id"] for cell in notebook["cells"]]
    assert len(identifiers) == len(set(identifiers))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile(source(cell), f"{NOTEBOOK.name}:cell{index}", "exec")
            assert cell.get("execution_count") is None
            assert cell.get("outputs", []) == []

    config = configuration(notebook)
    protocol_digest(notebook, config)
    assert config["PROTOCOL_ID"] == "stage42-action-conditioned-hybrid-defect-v2"
    assert config["EVIDENCE_STATUS"] == "FRESH_EVENT_EXCITATION_ORACLE_HEADROOM"
    assert config["DRIVE_OUTPUT_DIR"].endswith("counterfactual_faithfulness_stage42_ecoh_v2")
    assert config["FINAL_TRAINING_SEEDS"] == [4201, 4202, 4203]
    assert config["PRIMARY_VARIANTS"] == ["uniform_grounded"]
    assert config["CAUSAL_VARIANTS"] == [
        "smooth_matched", "shuffled_event", "oracle_event", "oracle_time",
        "oracle_geometry", "oracle_reset_ceiling",
    ]
    assert config["EVALUATION_TRAJECTORIES"] == 48
    assert config["MIN_REENTRY_ROWS"] == 32
    assert config["MIN_CONTACT_TAIL_RELATIVE_IMPROVEMENT"] == 0.25
    assert config["MIN_P95_RELATIVE_IMPROVEMENT"] == 0.10
    assert config["MAX_MEAN_RELATIVE_DEGRADATION"] == 0.02
    assert config["MAX_INSTRUMENTATION_DRIFT"] == 1e-6
    assert config["MIN_GUARD_AUROC"] == 0.70
    assert config["MIN_GUARD_BALANCED_ACCURACY"] == 0.65
    assert config["MAX_GUARD_BRIER"] == 0.20
    assert config["MAX_EVENT_TIME_MAE"] == 0.25
    assert config["PLANNING_WORD_NAMES"] == []
    assert config["DOWNLOAD_RESULTS"] is True
    assert len(banks(config)) == 32
    assert config["CLOSURE_EVALUATION_WORD_NAMES"] == [
        "LPPSSSPPP", "TPPSSSPPP",
        "LPPSSS0PPP", "TPPSSS0PPP",
        "LPPSSS00PPP", "TPPSSS00PPP",
        "LPPSSS000PPP", "TPPSSS000PPP",
    ]
    assert config["EVENT_EXCITATION_TOKEN_NAMES"] == ["P", "Q", "L", "T", "S", "0"]
    assert config["STAGE39_TOKEN_SPECS"]["P"] == (0.0, 0.20)
    assert config["STAGE39_TOKEN_SPECS"]["Q"] == (0.0, 0.14)
    assert config["STAGE39_TOKEN_SPECS"]["S"] == (180.0, 0.14)
    assert config["STAGE39_TOKEN_SPECS"]["0"] == (0.0, 0.0)
    assert config["SUPPORT_CHECKPOINT_INTERVAL"] == 64
    assert config["STAGE42_V1_SUPPORT_AUDIT"] == {
        "role": "development_support_falsification_only",
        "result_bundle_sha256": "72c1b2c24495cfb189f9014a4a061897b582fe27ef12a2b95465145f30521ebd",
        "evaluation_pool_start": 132000,
        "evaluation_pool_stop": 148000,
        "families_screened": 16000,
        "event_rich_families": 0,
        "model_outputs_read": False,
    }
    assert config["CONSTRUCTION_TRAJECTORY_POOL"] == list(range(148000, 150000))
    assert config["MODEL_SELECTION_TRAJECTORY_POOL"] == list(range(150000, 152000))
    assert config["CALIBRATION_TRAJECTORY_POOL"] == list(range(152000, 154000))
    assert config["EVALUATION_TRAJECTORY_POOL"] == list(range(154000, 170000))

    for key in [
        "EXPERIMENT_NOTEBOOK_PATH", "EXPERIMENT_BUILDER_PATH",
        "EXPERIMENT_NUMERICAL_PATH",
    ]:
        assert (REPOSITORY / config[key]).is_file(), (key, config[key])
    prior_configs = [configuration(json.loads(path.read_text())) for path in PRIOR_NOTEBOOKS]
    assert all(banks(config).isdisjoint(banks(prior)) for prior in prior_configs)
    assert all(
        trajectories(config).isdisjoint(trajectories(prior))
        for prior in prior_configs
    )

    text = "\n".join(source(cell) for cell in notebook["cells"])
    for required in [
        "select_stage42_event_rich_evaluation",
        "Prospective event-excitation support-qualified headroom test",
        "zero re-entry-positive families",
        "prospective_event_excitation_vocabulary",
        "earliest_complete_families_with_positive_reentry_incidence",
        "simulator_contact_incidence_only",
        "stage42_event_support_certificate.json",
        "stage42_event_support_screen_partial.json",
        "derive_stage42_support_decision",
        "partition_hybrid_defects",
        "guard_probe_metrics",
        "derive_stage42_defect_decision",
        "flow_composition_nmse",
        "event_auroc",
        "event_time_mae",
        "guard_probes_frozen",
        "state_scales_frozen_before_evaluation",
        "flow_guard_reset_reported_separately",
        "fixed_mode_path_bound_only",
        "learned_event_reset_experiment_authorized",
        "effect_magnitudes_used",
        "prediction_errors_used",
        "created_before_model_loading",
        "conditional_estimand",
        "support count changed after certificate freeze",
        "oracle_reset_ceiling",
        "equal_nominal_head_width",
        '"causal_claim_authorized": False',
        '"learned_deployment_claim_authorized": False',
        "planning_permanently_sealed",
        'if DOWNLOAD_RESULTS:',
        "STAGE42_FAILURE_TRACE_BEGIN",
        "failure_report.json",
    ]:
        assert required in text, required

    selection = source(notebook["cells"][7])
    assert "load_world_model(" not in selection
    assert "prediction_errors_used\": False" in selection
    assert "effect_magnitudes_used\": False" in selection
    assert selection.index("stage42_event_support_certificate.json") < selection.index(
        "generate_truth_record(record)"
    )
    assert "pool[len(candidate_ids):]" in selection
    assert "selected Stage 42 family support changed on resume" in selection
    assert "for name in EVALUATION_WORD_NAMES:" in selection
    assert "specification = WORD_BY_NAME[name]" in selection
    assert "for specification in EVALUATION_WORD_SPECS:" not in selection
    assert "SUPPORT_CHECKPOINT_INTERVAL" in selection
    assert "if checkpoint_due:" in selection
    adaptive = "\n".join(source(notebook["cells"][index]) for index in [10, 11, 12])
    assert 'load_stage39_sequences(short, "evaluation_closure")' not in adaptive
    assert 'generate_stage42_paired_record(record, "evaluation_closure")' not in adaptive
    assert 'fit_stage42_guard_probe' in adaptive
    assert 'evaluation_rows_used": 0' in adaptive
    heldout = source(notebook["cells"][13])
    assert heldout.index("derive_stage42_support_decision") < heldout.index(
        'generate_stage42_paired_record(record, "evaluation_closure")'
    )
    assert "__PROTOCOL_DIGEST__" not in text
    print("STAGE42_NOTEBOOK_VALIDATION_PASS")


if __name__ == "__main__":
    main()
