"""Static and protocol-digest validation for the Stage 39 Colabs."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
NOTEBOOKS = [
    ROOT / "39_fresh_coefficient_matched_replication.ipynb",
    ROOT / "39_1_wall_cross_environment_replication.ipynb",
    ROOT / "39_2_contact_tail_qualified_replication.ipynb",
]


def source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def load_configuration(notebook: dict) -> dict:
    namespace: dict = {}
    exec(compile(source(notebook["cells"][1]), "configuration", "exec"), namespace)
    return namespace


def validate_protocol_digest(notebook: dict, configuration: dict) -> None:
    protocol_sources = [source(cell).strip() for cell in notebook["cells"]]
    observed = str(configuration["NOTEBOOK_PROTOCOL_SHA256"])
    placeholder = re.sub(
        r'NOTEBOOK_PROTOCOL_SHA256\s*=\s*"[0-9a-f]{64}"',
        'NOTEBOOK_PROTOCOL_SHA256 = "__PROTOCOL_DIGEST__"',
        protocol_sources[1],
        count=1,
    )
    if placeholder == protocol_sources[1]:
        raise AssertionError("could not restore protocol digest placeholder")
    protocol_sources[1] = placeholder
    expected = hashlib.sha256(
        json.dumps(
            protocol_sources, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()
    if observed != expected:
        raise AssertionError(f"protocol digest mismatch: {observed} != {expected}")


def validate_notebook(path: Path) -> dict:
    notebook = json.loads(path.read_text())
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert len(notebook["cells"]) == 14
    ids = [cell["id"] for cell in notebook["cells"]]
    assert len(ids) == len(set(ids))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile(source(cell), f"{path.name}:cell{index}", "exec")
            assert cell.get("execution_count") is None
            assert cell.get("outputs", []) == []
    text = "\n".join(source(cell) for cell in notebook["cells"])
    assert "__PROTOCOL_DIGEST__" not in text
    assert "planning_permanently_sealed" in text
    assert "EQUIVALENCE_MARGIN = 0.05" in text
    assert "PRIMARY_CONFIDENCE = 0.90" in text
    assert "COEFFICIENT_MATCH_FACTOR = 0.45" in text
    assert "stage38_evidence_consumed\": False" in text

    configuration = load_configuration(notebook)
    validate_protocol_digest(notebook, configuration)
    assert configuration["PLANNING_WORD_NAMES"] == []
    assert configuration["PRIMARY_VARIANTS"] == ["coefficient_matched", "full"]
    expected_final_seed = {
        "stage39-fresh-coefficient-matched-replication-v1": 3903,
        "stage39.1-wall-cross-environment-replication-v2": 3913,
        "stage39.2-contact-tail-qualified-replication-v1": 3923,
    }[configuration["PROTOCOL_ID"]]
    assert configuration["FINAL_TRAINING_SEEDS"][-1] == expected_final_seed
    assert configuration["EVALUATION_TRAJECTORIES"] == 48
    assert set(configuration["CONSTRUCTION_TRAJECTORY_POOL"]).isdisjoint(
        configuration["MODEL_SELECTION_TRAJECTORY_POOL"]
    )
    assert set(configuration["CONSTRUCTION_TRAJECTORY_POOL"]).isdisjoint(
        configuration["CALIBRATION_TRAJECTORY_POOL"]
    )
    assert set(configuration["CONSTRUCTION_TRAJECTORY_POOL"]).isdisjoint(
        configuration["EVALUATION_TRAJECTORY_POOL"]
    )
    banks = [
        set(configuration[name])
        for name in [
            "CONSTRUCTION_WORD_NAMES",
            "MODEL_SELECTION_WORD_NAMES",
            "CALIBRATION_WORD_NAMES",
            "CLOSURE_EVALUATION_WORD_NAMES",
        ]
    ]
    assert all(len(bank) == 8 for bank in banks)
    assert all(
        banks[left].isdisjoint(banks[right])
        for left in range(len(banks))
        for right in range(left + 1, len(banks))
    )
    for key in [
        "EXPERIMENT_NOTEBOOK_PATH",
        "EXPERIMENT_BUILDER_PATH",
        "EXPERIMENT_NUMERICAL_PATH",
    ]:
        assert (REPOSITORY / configuration[key]).is_file(), (key, configuration[key])
    return configuration


def main() -> None:
    configurations = [validate_notebook(path) for path in NOTEBOOKS]
    stage38 = json.loads((ROOT / "38_cross_model_pscd_confirmation.ipynb").read_text())
    stage38_configuration = load_configuration(stage38)
    assert configurations[0]["ENVIRONMENT"] == "PushT"
    assert configurations[1]["ENVIRONMENT"] == "Wall"
    assert configurations[2]["ENVIRONMENT"] == "PushT"
    assert configurations[0]["MODEL_NAMES"] == ["jepa_wm_pusht", "dino_wm_pusht"]
    assert configurations[1]["MODEL_NAMES"] == ["jepa_wm_wall", "dino_wm_wall"]
    assert configurations[2]["MODEL_NAMES"] == ["jepa_wm_pusht", "dino_wm_pusht"]
    assert configurations[1]["PROTOCOL_ID"] == (
        "stage39.1-wall-cross-environment-replication-v2"
    )
    assert configurations[1]["SIMULATOR_LATENT_DIM"] == 64
    assert configurations[1]["SIMULATOR_CARRIER_SCHEMA"] == [
        "dot_x", "dot_y", "wall_x", "door_y",
    ]
    assert configurations[1]["SIMULATOR_PHYSICAL_SCHEMA"] == ["dot_x", "dot_y"]
    assert configurations[1]["DRIVE_OUTPUT_DIR"].endswith("stage39_1_wall_v2")
    assert configurations[2]["PROTOCOL_ID"] == (
        "stage39.2-contact-tail-qualified-replication-v1"
    )
    assert configurations[2]["DRIVE_OUTPUT_DIR"].endswith("stage39_2_ctqr")
    assert configurations[2]["TAIL_QUALIFICATION_VARIANTS"] == [
        "coefficient_matched", "full",
    ]
    wall_text = "\n".join(
        source(cell) for cell in json.loads(NOTEBOOKS[1].read_text())["cells"]
    )
    assert 'train["initial_carrier"]' in wall_text
    assert 'train["carrier_targets"]' in wall_text
    assert 'train["physical_targets"]' in wall_text
    assert 'reference["simulator_initial_carrier"]' in wall_text
    assert 'reference["simulator_carrier"]' in wall_text
    assert 'SIMULATOR_FINAL, reference["initial_physical"]' not in wall_text
    assert "FAIL_CLOSED_GATE: stage391_simulator_preflight" in wall_text
    assert '"world_model_checkpoints_loaded": False' in wall_text
    assert '"evaluation_rows_used": 0' in wall_text
    assert configurations[1]["EXPECTED_PRETRAINED_ASSET_SHA256"][
        "dino_wm_wall.pth.tar"
    ] == "ff170be5aec9249768be4a220d600b8f00a8589b2a78982ecf9273809f2767df"
    first_words = set().union(*[
        set(configurations[0][key])
        for key in [
            "CONSTRUCTION_WORD_NAMES", "MODEL_SELECTION_WORD_NAMES",
            "CALIBRATION_WORD_NAMES", "CLOSURE_EVALUATION_WORD_NAMES",
        ]
    ])
    second_words = set().union(*[
        set(configurations[1][key])
        for key in [
            "CONSTRUCTION_WORD_NAMES", "MODEL_SELECTION_WORD_NAMES",
            "CALIBRATION_WORD_NAMES", "CLOSURE_EVALUATION_WORD_NAMES",
        ]
    ])
    assert first_words.isdisjoint(second_words)
    third_words = set().union(*[
        set(configurations[2][key])
        for key in [
            "CONSTRUCTION_WORD_NAMES", "MODEL_SELECTION_WORD_NAMES",
            "CALIBRATION_WORD_NAMES", "CLOSURE_EVALUATION_WORD_NAMES",
        ]
    ])
    assert first_words.isdisjoint(third_words)
    assert second_words.isdisjoint(third_words)
    stage38_words = set().union(*[
        set(stage38_configuration[key])
        for key in [
            "CONSTRUCTION_WORD_NAMES", "MODEL_SELECTION_WORD_NAMES",
            "CALIBRATION_WORD_NAMES", "CLOSURE_EVALUATION_WORD_NAMES",
        ]
    ])
    assert first_words.isdisjoint(stage38_words)
    assert second_words.isdisjoint(stage38_words)
    assert third_words.isdisjoint(stage38_words)
    stage38_trajectories = set().union(*[
        set(stage38_configuration[key])
        for key in [
            "CONSTRUCTION_TRAJECTORY_POOL", "MODEL_SELECTION_TRAJECTORY_POOL",
            "CALIBRATION_TRAJECTORY_POOL", "EVALUATION_TRAJECTORY_POOL",
        ]
    ])
    for configuration in configurations:
        new_trajectories = set().union(*[
            set(configuration[key])
            for key in [
                "CONSTRUCTION_TRAJECTORY_POOL", "MODEL_SELECTION_TRAJECTORY_POOL",
                "CALIBRATION_TRAJECTORY_POOL", "EVALUATION_TRAJECTORY_POOL",
            ]
        ])
        assert new_trajectories.isdisjoint(stage38_trajectories)
    trajectory_sets = [
        set().union(*[
            set(configuration[key])
            for key in [
                "CONSTRUCTION_TRAJECTORY_POOL", "MODEL_SELECTION_TRAJECTORY_POOL",
                "CALIBRATION_TRAJECTORY_POOL", "EVALUATION_TRAJECTORY_POOL",
            ]
        ])
        for configuration in configurations
    ]
    assert all(
        trajectory_sets[left].isdisjoint(trajectory_sets[right])
        for left in range(len(trajectory_sets))
        for right in range(left + 1, len(trajectory_sets))
    )
    stage392_text = "\n".join(
        source(cell) for cell in json.loads(NOTEBOOKS[2].read_text())["cells"]
    )
    for required in [
        "stage39_evaluation_outcomes_consumed\": False",
        "absolute_tail_qualification_passed",
        "MAX_P95_PHYSICAL_NMSE",
        "MAX_CATASTROPHIC_RATE",
        "contact_post_contact_physical_nmse",
    ]:
        assert required in stage392_text
    print("STAGE39_NOTEBOOK_VALIDATION_PASS")


if __name__ == "__main__":
    main()
