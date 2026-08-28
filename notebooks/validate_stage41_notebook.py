"""Static, split, source-binding, and digest validation for Stage 41."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
NOTEBOOK = ROOT / "41_causal_event_reset_headroom.ipynb"
PRIOR_NOTEBOOKS = [
    ROOT / "38_cross_model_pscd_confirmation.ipynb",
    ROOT / "39_fresh_coefficient_matched_replication.ipynb",
    ROOT / "39_1_wall_cross_environment_replication.ipynb",
    ROOT / "39_2_contact_tail_qualified_replication.ipynb",
    ROOT / "40_contact_tail_risk_distillation.ipynb",
]


def source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def configuration(notebook: dict) -> dict:
    namespace: dict = {}
    exec(compile(source(notebook["cells"][1]), "stage41-config", "exec"), namespace)
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
    ids = [cell["id"] for cell in notebook["cells"]]
    assert len(ids) == len(set(ids))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile(source(cell), f"{NOTEBOOK.name}:cell{index}", "exec")
            assert cell.get("execution_count") is None
            assert cell.get("outputs", []) == []
    config = configuration(notebook)
    protocol_digest(notebook, config)
    assert config["PROTOCOL_ID"] == "stage41-causal-event-reset-headroom-v3"
    assert config["FINAL_TRAINING_SEEDS"] == [4101, 4102, 4103]
    assert config["PRIMARY_VARIANTS"] == ["uniform_grounded"]
    assert config["CAUSAL_VARIANTS"] == [
        "smooth_matched", "shuffled_event", "oracle_event", "oracle_time",
        "oracle_geometry", "oracle_reset_ceiling",
    ]
    assert config["MIN_CONTACT_TAIL_RELATIVE_IMPROVEMENT"] == 0.25
    assert config["MIN_P95_RELATIVE_IMPROVEMENT"] == 0.10
    assert config["MAX_MEAN_RELATIVE_DEGRADATION"] == 0.02
    assert config["MAX_INSTRUMENTATION_DRIFT"] == 1e-6
    assert config["DOWNLOAD_RESULTS"] is True
    assert config["EVALUATION_TRAJECTORIES"] == 48
    assert config["PLANNING_WORD_NAMES"] == []
    assert len(banks(config)) == 32
    for key in [
        "EXPERIMENT_NOTEBOOK_PATH", "EXPERIMENT_BUILDER_PATH",
        "EXPERIMENT_NUMERICAL_PATH",
    ]:
        assert (REPOSITORY / config[key]).is_file(), (key, config[key])
    prior_configs = [configuration(json.loads(path.read_text())) for path in PRIOR_NOTEBOOKS]
    assert all(banks(config).isdisjoint(banks(prior)) for prior in prior_configs)
    assert all(
        trajectories(config).isdisjoint(trajectories(prior)) for prior in prior_configs
    )
    text = "\n".join(source(cell) for cell in notebook["cells"])
    for required in [
        "disable_stage41_agent_block_collision",
        "ordinary-vs-agent-block-collision-disabled-v1",
        "causal_design_matrix", "oracle_reset_ceiling",
        "storage_equivalent",
        "storage_dtype=np.float32",
        "equal_nominal_head_width", "MIN_CAUSAL_EFFECT_GAIN = 0.10",
        "MIN_CAUSAL_EFFECT_COSINE = 0.25",
        '"causal_claim_authorized": False',
        '"learned_deployment_claim_authorized": False',
        '"stage40_artifacts_read": False',
        '"stage40_evaluation_rows_consumed": False',
        "planning_permanently_sealed",
        'if DOWNLOAD_RESULTS:',
        '"instrumentation_drifts"',
        'STAGE41_FAILURE_TRACE_BEGIN',
        'failure_report.json',
        'stage41_phase.json',
    ]:
        assert required in text, required
    adaptive = "\n".join(source(notebook["cells"][index]) for index in [10, 11, 12])
    assert 'load_stage39_sequences(short, "evaluation_closure")' not in adaptive
    assert 'generate_stage41_paired_record(record, "evaluation_closure")' not in adaptive
    evaluation = source(notebook["cells"][13])
    assert evaluation.index("validate_digest_sidecar(certificate_path)") < evaluation.index(
        'generate_stage41_paired_record(record, "evaluation_closure")'
    )
    assert "__PROTOCOL_DIGEST__" not in text
    print("STAGE41_NOTEBOOK_VALIDATION_PASS")


if __name__ == "__main__":
    main()
