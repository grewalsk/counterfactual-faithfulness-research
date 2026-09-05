"""Static, split, source-binding, and digest validation for Stage 44."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
NOTEBOOK = ROOT / "44_visual_causal_realization_audit.ipynb"
PRIOR_NOTEBOOKS = [
    ROOT / "38_cross_model_pscd_confirmation.ipynb",
    ROOT / "39_fresh_coefficient_matched_replication.ipynb",
    ROOT / "39_1_wall_cross_environment_replication.ipynb",
    ROOT / "39_2_contact_tail_qualified_replication.ipynb",
    ROOT / "40_contact_tail_risk_distillation.ipynb",
    ROOT / "41_causal_event_reset_headroom.ipynb",
    ROOT / "42_event_conditioned_oracle_hybrid.ipynb",
    ROOT / "43_recursive_reset_sufficiency.ipynb",
]


def source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def configuration(notebook: dict) -> dict:
    namespace: dict = {}
    exec(compile(source(notebook["cells"][1]), "stage44-config", "exec"), namespace)
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
    assert len(notebook["cells"]) == 13
    identifiers = [cell["id"] for cell in notebook["cells"]]
    assert len(identifiers) == len(set(identifiers))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile(source(cell), f"{NOTEBOOK.name}:cell{index}", "exec")
            assert cell.get("execution_count") is None
            assert cell.get("outputs", []) == []

    config = configuration(notebook)
    protocol_digest(notebook, config)
    assert config["PROTOCOL_ID"] == "stage44-visual-causal-realization-audit-v1"
    assert config["EVIDENCE_STATUS"] == "FRESH_VISUAL_CAUSAL_REALIZATION_AUDIT"
    assert config["DRIVE_OUTPUT_DIR"].endswith("counterfactual_faithfulness_stage44_vcra")
    assert config["PRIMARY_MODEL"] == "jepa"
    assert config["FINAL_TRAINING_SEEDS"] == [4401]
    assert config["EVALUATION_TRAJECTORIES"] == 48
    assert config["VISUAL_PCA_RANK"] == 8
    assert config["SPATIAL_GRID_SIZE"] == 16
    assert config["SPATIAL_PYRAMID_BINS"] == 4
    assert config["SUPPORT_REFERENCE_MAX_ROWS"] == 1024
    assert config["SUPPORT_QUERY_MAX_ROWS"] == 1024
    assert config["SUPPORT_NEIGHBORS"] == 16
    assert config["SUPPORT_TANGENT_RANK"] == 8
    assert config["MIN_TARGET_PHYSICAL_R2"] == 0.50
    assert config["MIN_TARGET_PATCH_MACRO_AUROC"] == 0.72
    assert config["MAX_TEACHER_VISUAL_NMSE"] == 1.00
    assert config["MAX_RECURSIVE_TO_TEACHER_NMSE_RATIO"] == 1.25
    assert config["MAX_RECURSIVE_TO_TEACHER_NORMAL_RATIO"] == 1.25
    assert config["MIN_COUNTERFACTUAL_VISUAL_COSINE"] == 0.40
    assert config["MIN_COUNTERFACTUAL_PHYSICAL_COSINE"] == 0.30
    assert config["MIN_OBJECT_TO_BACKGROUND_MEDIATION_RATIO"] == 1.25
    assert config["PLANNING_WORD_NAMES"] == []
    assert config["DOWNLOAD_RESULTS"] is True
    assert config["OFFICIAL_IMAGE_DECODER_FILENAMES"] == {
        "jepa_wm_pusht": "vm2m_lpips_dv2vits_vitldec_224_INet.pth.tar",
        "dino_wm_pusht": "vm2m_lpips_dv2vits_vitldec_224_05norm.pth.tar",
    }
    assert config["CONSTRUCTION_TRAJECTORY_POOL"] == list(range(192000, 194000))
    assert config["MODEL_SELECTION_TRAJECTORY_POOL"] == list(range(194000, 196000))
    assert config["CALIBRATION_TRAJECTORY_POOL"] == list(range(196000, 198000))
    assert config["EVALUATION_TRAJECTORY_POOL"] == list(range(198000, 214000))
    assert len(banks(config)) == 32
    assert all(9 <= len(word) <= 12 for word in banks(config))
    assert all(len(value) == 4 for value in config["COUNTERFACTUAL_PAIRS_BY_SPLIT"].values())

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
        "visual--causal realization audit",
        "reconstruction error is never relabeled as an off-manifold result",
        "decode_unroll",
        "vm2m_lpips_dv2vits_vitldec_224_INet.pth.tar",
        "vm2m_lpips_dv2vits_vitldec_224_05norm.pth.tar",
        "target_summary", "teacher_summary", "recursive_summary",
        "stage44_teacher_forced_word", "stage44_patch_labels",
        "spatial_pyramid_summary", "local_support_geometry",
        "counterfactual_realization_metrics", "stage44_swap_diagnostics",
        "physical_probes_frozen", "support_references_frozen",
        "derive_stage44_decision",
        '"rgb_reconstruction_is_a_decision_gate": False',
        '"oracle_patch_mask_available_at_deployment": False',
        '"shared_dinov2_encoder_confound_retained": True',
        '"causal_claim_authorized": False',
        '"planning_opened": False',
        "planning_permanently_sealed",
        "STAGE44_FAILURE_TRACE_BEGIN",
        "failure_report.json",
        "if DOWNLOAD_RESULTS:",
    ]:
        assert required in text, required
    assert "np.cross(diagnostic[i], diagnostic[j])" not in text

    physical = source(notebook["cells"][7])
    assert "load_world_model(" not in physical
    assert 'prediction_errors_used": False' in physical
    assert 'effect_magnitudes_used": False' in physical
    assert physical.index("stage44_event_support_certificate.json") < physical.index(
        "generate_truth_record(record)"
    )

    development = "\n".join(source(notebook["cells"][index]) for index in [8, 9, 10])
    assert 'stage44_materialize_record(\n                        bundle, record, "evaluation_closure"' not in development
    assert '"evaluation_rows_used": 0' in development
    assert '"evaluation_statistics_read": False' in development
    assert '"evaluation_visual_fields_encoded": False' in development
    assert "selected_penalty" in development
    assert "official_decoder_contracts.json" in development

    heldout = source(notebook["cells"][11])
    assert heldout.index("evaluation_open_certificate.json") < heldout.index(
        'stage44_materialize_record(\n                        bundle, record, "evaluation_closure"'
    )
    assert heldout.index("derive_stage44_support_decision") < heldout.index(
        'stage44_materialize_record(\n                        bundle, record, "evaluation_closure"'
    )
    assert "support_indices" in heldout
    assert "object_to_background_mediation_ratio" in heldout
    assert "planning_audit_authorized" in heldout
    assert "__PROTOCOL_DIGEST__" not in text
    print("STAGE44_NOTEBOOK_VALIDATION_PASS")


if __name__ == "__main__":
    main()
