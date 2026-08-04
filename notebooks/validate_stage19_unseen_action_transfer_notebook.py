import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "19_unseen_action_family_transfer.ipynb"
BUILDER = ROOT / "build_stage19_unseen_action_transfer_notebook.py"


def source(cell):
    return "".join(cell.get("source", []))


def assigned_value(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"missing assignment {name}")


def validate():
    before = NOTEBOOK.read_bytes()
    subprocess.run([sys.executable, str(BUILDER)], check=True, capture_output=True)
    after = NOTEBOOK.read_bytes()
    assert before == after, "Stage 19 builder is not deterministic"

    notebook = json.loads(after)
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["metadata"]["colab"]["gpuType"] == "L4"
    assert len(notebook["cells"]) == 13
    assert notebook["cells"][0]["cell_type"] == "markdown"
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 12
    assert all(cell.get("outputs") == [] for cell in code_cells)
    assert all(cell.get("execution_count") is None for cell in code_cells)
    for cell in code_cells:
        ast.parse(source(cell))

    config_source = source(code_cells[0])
    config_tree = ast.parse(config_source)
    assert config_source.startswith("# SINGLE CONFIGURATION BLOCK")
    assert assigned_value(config_tree, "PROTOCOL_ID") == "stage19-frozen-subspace-unseen-action-transfer-v1"
    assert assigned_value(config_tree, "FIXED_BLOCK") == 4
    assert assigned_value(config_tree, "PRIMARY_RANK") == 64
    assert assigned_value(config_tree, "SENSITIVITY_RANKS") == [64, 128]
    assert assigned_value(config_tree, "INTERVENTION_FORWARDS_PER_RECORD") == 30
    assert assigned_value(config_tree, "EVALUATION_TRAJECTORY_TARGET_PER_FAMILY") == 24
    assert assigned_value(config_tree, "EXPECTED_STAGE18_SUBSPACE_SHA256") == (
        "2f9c496d54623a9062e465a18c70039acc18cb8a1cc2833a5f4ade162ca3f90b"
    )
    families = assigned_value(config_tree, "TRANSFER_FAMILIES")
    assert families == [
        "rotated_direction",
        "magnitude_0p08",
        "magnitude_0p16",
        "delayed_equal_impulse",
        "pulsed_equal_impulse",
    ]
    assert "STAGE19_RUN_MODE" in config_source
    assert "STAGE19_SOURCE_COMMIT" in config_source
    assert "STAGE19_RUN_NONCE" in config_source
    assert "strip().lower()" in config_source

    combined = "\n".join(source(cell) for cell in code_cells)
    assert "fit_and_freeze_subspaces" not in combined
    assert "fit_dual_ridge_basis" not in combined
    assert "grouped_kernel_ridge_cv" not in combined
    assert "torch.linalg.svd" not in combined
    assert "requires_grad_(True)" not in combined
    assert "torch.autograd" not in combined
    assert "jvp(" not in combined.lower()
    assert "jacrev" not in combined.lower()
    assert "gradient" not in source(code_cells[9]).lower()
    assert "EXPECTED_STAGE18_SUBSPACE_SHA256" in source(code_cells[7])
    assert "observed_subspace_sha256" in source(code_cells[7])
    assert "validate_stage18_subspace_arrays" in source(code_cells[7])
    assert "validated_before_stage19_model_activations" in source(code_cells[7])
    assert "stage19_subspace_refit\": False" in source(code_cells[7])
    assert source(code_cells[8]).index("STAGE18_ARTIFACT_VALIDATED") < source(code_cells[8]).index("load_frozen_model")
    assert "ALL_EVALUATION_RECORDS" in source(code_cells[8])
    assert "run_all_interventions(ALL_EVALUATION_RECORDS)" in source(code_cells[9])
    assert "bidirectional_transfer_gate_pass" in source(code_cells[10])
    assert "all_families_must_pass_for_broad_transfer_claim" in source(code_cells[10])
    assert "secondary_only_does_not_enter_representation_transfer_gate" in source(code_cells[10])
    assert "CONFIRMED_TRANSFER_ALL_UNSEEN_ACTION_FAMILIES" in source(code_cells[10])
    assert "stage19_unseen_action_transfer_result_bundle_" in source(code_cells[11])

    # The digest binds every protocol source with the placeholder still present.
    observed_digest = assigned_value(config_tree, "NOTEBOOK_PROTOCOL_SHA256")
    protocol_sources = [source(notebook["cells"][0])]
    placeholder_config = config_source.replace(observed_digest, "__PROTOCOL_DIGEST__", 1)
    protocol_sources.append(placeholder_config)
    protocol_sources.extend(source(cell) for cell in code_cells[1:])
    expected_digest = hashlib.sha256(
        json.dumps(protocol_sources, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    assert observed_digest == expected_digest, "protocol digest mismatch"

    # The pilot intervention count is mechanically implied by the frozen design.
    doses, ranks, nulls = 4, 2, 4
    sufficiency = doses + ranks * (1 + nulls) + (ranks - 1) + 3
    necessity = ranks * (2 + nulls)
    assert sufficiency + necessity == 30

    print("Stage 19 notebook validation passed")


if __name__ == "__main__":
    validate()
