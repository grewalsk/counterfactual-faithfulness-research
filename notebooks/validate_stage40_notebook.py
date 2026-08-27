"""Static, split, source-binding, and digest validation for Stage 40."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
NOTEBOOK = ROOT / "40_contact_tail_risk_distillation.ipynb"
PRIOR_NOTEBOOKS = [
    ROOT / "38_cross_model_pscd_confirmation.ipynb",
    ROOT / "39_fresh_coefficient_matched_replication.ipynb",
    ROOT / "39_1_wall_cross_environment_replication.ipynb",
    ROOT / "39_2_contact_tail_qualified_replication.ipynb",
]


def source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def configuration(notebook: dict) -> dict:
    namespace: dict = {}
    exec(compile(source(notebook["cells"][1]), "stage40-config", "exec"), namespace)
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
        set(config[name])
        for name in [
            "CONSTRUCTION_WORD_NAMES", "MODEL_SELECTION_WORD_NAMES",
            "CALIBRATION_WORD_NAMES", "CLOSURE_EVALUATION_WORD_NAMES",
        ]
    ])


def trajectories(config: dict) -> set[int]:
    return set().union(*[
        set(config[name])
        for name in [
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
    assert config["PROTOCOL_ID"] == "stage40-contact-tail-risk-distillation-v1"
    assert config["FINAL_TRAINING_SEEDS"] == [4001, 4002, 4003]
    assert config["PRIMARY_VARIANTS"] == [
        "uniform_grounded", "contact_risk_grounded",
    ]
    assert config["CONTACT_MULTIPLIER_CANDIDATES"] == [1.0, 2.0, 4.0, 8.0]
    assert config["PINNED_LATENT_OUTER_WEIGHTS"] == {"jepa": 2.0, "dino": 0.5}
    assert config["DOWNLOAD_RESULTS"] is True
    assert config["EVALUATION_TRAJECTORIES"] == 48
    assert config["PLANNING_WORD_NAMES"] == []
    assert len(banks(config)) == 32
    for key in [
        "EXPERIMENT_NOTEBOOK_PATH", "EXPERIMENT_BUILDER_PATH",
        "EXPERIMENT_NUMERICAL_PATH",
    ]:
        assert (REPOSITORY / config[key]).is_file(), (key, config[key])

    prior_configs = [
        configuration(json.loads(path.read_text())) for path in PRIOR_NOTEBOOKS
    ]
    assert all(banks(config).isdisjoint(banks(prior)) for prior in prior_configs)
    assert all(
        trajectories(config).isdisjoint(trajectories(prior))
        for prior in prior_configs
    )
    text = "\n".join(source(cell) for cell in notebook["cells"])
    for required in [
        "fit_contact_risk_predictive_state_closure",
        "simulator_ground_truth",
        "MIN_P95_RELATIVE_IMPROVEMENT = 0.10",
        "MIN_CONTACT_RELATIVE_IMPROVEMENT = 0.10",
        "MEAN_NONINFERIORITY_MARGIN = 0.05",
        '"stage39_2_artifacts_read": False',
        '"stage39_2_evaluation_rows_consumed": False',
        "contact_risk_selection_sha256",
        'if DOWNLOAD_RESULTS and not PIPELINE_FAILED and (OUT / "stage40_decision.json").is_file():',
    ]:
        assert required in text, required
    adaptive_cells = "\n".join(
        source(notebook["cells"][index]) for index in [10, 11, 12]
    )
    assert 'load_stage39_sequences(short, "evaluation_closure")' not in adaptive_cells
    assert 'generate_stage39_path_record(\n                        bundle, record, "evaluation_closure"' not in adaptive_cells
    assert (
        'CALIBRATION_SEED, "stage40_risk_selection", short,\n'
        '                            int(selection_seed),' in adaptive_cells
    )
    assert (
        'CALIBRATION_SEED, "stage40_risk_selection", short,\n'
        '                            float(multiplier)' not in adaptive_cells
    )
    assert "planning_permanently_sealed" in text
    assert "__PROTOCOL_DIGEST__" not in text
    print("STAGE40_NOTEBOOK_VALIDATION_PASS")


if __name__ == "__main__":
    main()
