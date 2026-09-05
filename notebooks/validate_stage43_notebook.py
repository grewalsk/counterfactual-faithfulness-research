"""Static, split, source-binding, and digest validation for Stage 43."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
NOTEBOOK = ROOT / "43_recursive_reset_sufficiency.ipynb"
PRIOR_NOTEBOOKS = [
    ROOT / "38_cross_model_pscd_confirmation.ipynb",
    ROOT / "39_fresh_coefficient_matched_replication.ipynb",
    ROOT / "39_1_wall_cross_environment_replication.ipynb",
    ROOT / "39_2_contact_tail_qualified_replication.ipynb",
    ROOT / "40_contact_tail_risk_distillation.ipynb",
    ROOT / "41_causal_event_reset_headroom.ipynb",
    ROOT / "42_event_conditioned_oracle_hybrid.ipynb",
]


def source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def configuration(notebook: dict) -> dict:
    namespace: dict = {}
    exec(compile(source(notebook["cells"][1]), "stage43-config", "exec"), namespace)
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
    assert config["PROTOCOL_ID"] == "stage43-recursive-reset-sufficiency-v1"
    assert config["EVIDENCE_STATUS"] == "FRESH_RECURSIVE_RESET_SUFFICIENCY"
    assert config["DRIVE_OUTPUT_DIR"].endswith("counterfactual_faithfulness_stage43_rrsl")
    assert config["FINAL_TRAINING_SEEDS"] == [4301, 4302, 4303]
    assert config["EVALUATION_TRAJECTORIES"] == 48
    assert config["MIN_REENTRY_ROWS"] == 32
    assert config["RESET_HISTORY_LAGS"] == 3
    assert config["RESET_TENSOR_STATE_RANK"] == 32
    assert config["RESET_MLP_HIDDEN"] == 112
    assert config["RESET_DELTA_NORM_QUANTILE"] == 0.99
    assert config["ACTIVE_RESET_MLP_SELECTION_EPOCHS"] == 120
    assert config["ACTIVE_RESET_MLP_FINAL_EPOCHS"] == 200
    assert config["MAX_CURRENT_OPERATOR_PARAMETER_RATIO"] == 1.50
    assert config["PRIMARY_RESET_VARIANT"] == "current_recursive_tensor"
    assert config["MIN_CONTACT_TAIL_RELATIVE_IMPROVEMENT"] == 0.25
    assert config["MIN_P95_RELATIVE_IMPROVEMENT"] == 0.10
    assert config["MAX_MEAN_RELATIVE_DEGRADATION"] == 0.02
    assert config["MIN_LOO_CONTACT_TAIL_RELATIVE_IMPROVEMENT"] == 0.10
    assert config["PLANNING_WORD_NAMES"] == []
    assert config["DOWNLOAD_RESULTS"] is True
    assert config["RESET_VARIANTS"] == [
        "affine_output_control", "current_nonrecursive_tensor",
        "sham_recursive_tensor", "current_recursive_tensor",
        "current_recursive_mlp", "history_recursive_tensor",
        "physical_recursive_tensor",
    ]
    assert set(config["RESET_VARIANT_SPECS"]) == set(config["RESET_VARIANTS"]) - {
        "affine_output_control"
    }
    assert config["RESET_VARIANT_SPECS"]["current_nonrecursive_tensor"]["recursive"] is False
    assert config["RESET_VARIANT_SPECS"]["physical_recursive_tensor"]["representation"] == "physical_oracle"
    assert config["STAGE42_NEGATIVE_AUDIT"] == {
        "protocol_id": "stage42-action-conditioned-hybrid-defect-v2",
        "run_signature": "34cbe4d0760cdc650ff883001f44ab09ab403a9003075c33881dfc578dbc8e3e",
        "status": "no_oracle_reset_headroom",
        "evaluation_opened": True,
        "planning_opened": False,
        "role": "prior_negative_evidence_only",
    }
    assert config["STAGE42_V1_SUPPORT_AUDIT"]["event_rich_families"] == 0
    assert "STAGE43_V1_SUPPORT_AUDIT" not in config
    assert config["CONSTRUCTION_TRAJECTORY_POOL"] == list(range(170000, 172000))
    assert config["MODEL_SELECTION_TRAJECTORY_POOL"] == list(range(172000, 174000))
    assert config["CALIBRATION_TRAJECTORY_POOL"] == list(range(174000, 176000))
    assert config["EVALUATION_TRAJECTORY_POOL"] == list(range(176000, 192000))
    assert len(banks(config)) == 32
    assert all(9 <= len(word) <= 12 for word in banks(config))
    assert config["CLOSURE_EVALUATION_WORD_NAMES"] == [
        "LPQSSSQPP", "TPQSSSQPP",
        "LPQSSS0QPP", "TPQSSS0QPP",
        "LPQSSS00QPP", "TPQSSS00QPP",
        "LPQSSS000QPP", "TPQSSS000QPP",
    ]

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
        "recursive reset sufficiency lattice",
        "Stage 42 v2 completed without a pipeline failure",
        "does **not** claim to recover",
        "select_stage43_event_rich_evaluation",
        "stage43_event_support_certificate.json",
        "simulator_contact_incidence_only",
        "stage43_reset_bundle",
        "tensor_reset_design",
        "clip_row_norms",
        "state_projection_rank",
        "Stage43ResetMLP",
        "stage43_recursive_prediction",
        "current_nonrecursive_tensor",
        "sham_recursive_tensor",
        "physical_recursive_tensor",
        "recursive_reset_lattice_frozen",
        "reset_selection_uses_model_selection_only",
        '"delta_norm_quantile"',
        "support count changed after certificate freeze",
        "derive_stage43_decision",
        '"saltation_matrix_identified": False',
        '"causal_claim_authorized": False',
        '"deployment_claim_authorized": False',
        '"planning_opened": False',
        "planning_permanently_sealed",
        'if DOWNLOAD_RESULTS:',
        "STAGE43_FAILURE_TRACE_BEGIN",
        "failure_report.json",
    ]:
        assert required in text, required

    selection = source(notebook["cells"][7])
    assert "load_world_model(" not in selection
    assert 'prediction_errors_used": False' in selection
    assert 'effect_magnitudes_used": False' in selection
    assert selection.index("stage43_event_support_certificate.json") < selection.index(
        "generate_truth_record(record)"
    )
    assert "pool[len(candidate_ids):]" in selection
    assert "SUPPORT_CHECKPOINT_INTERVAL" in selection
    assert "for name in EVALUATION_WORD_NAMES:" in selection

    development = "\n".join(source(notebook["cells"][index]) for index in [10, 11, 12])
    assert '"evaluation_closure"' not in development
    assert 'evaluation_rows_used": 0' in development
    assert "RESET_SELECTION_ROWS" in development
    assert "selected_value" in development
    assert "MAX_CURRENT_OPERATOR_PARAMETER_RATIO" in development
    assert '"delta_norm_cap"' in development

    tensor_parameters = (32 + 6 + 32 * 6) * 256 + 256
    mlp_parameters = (256 + 2 + 6) * 112 + 112 + 112 * 256 + 256
    ratio = max(tensor_parameters, mlp_parameters) / min(
        tensor_parameters, mlp_parameters
    )
    assert ratio <= config["MAX_CURRENT_OPERATOR_PARAMETER_RATIO"]

    heldout = source(notebook["cells"][13])
    assert heldout.index("derive_stage43_support_decision") < heldout.index(
        'generate_stage43_paired_record(record, "evaluation_closure")'
    )
    assert heldout.index("recursive_reset_lattice_frozen") < heldout.index(
        'generate_stage43_paired_record(record, "evaluation_closure")'
    )
    assert "updated[event_tensor] +=" in heldout
    assert "state[active_tensor] = updated" in heldout
    assert "current_tensor_headroom" in heldout
    assert "physical_oracle_headroom" in heldout
    assert "__PROTOCOL_DIGEST__" not in text
    print("STAGE43_NOTEBOOK_VALIDATION_PASS")


if __name__ == "__main__":
    main()
