"""Static and reproducibility checks for the Stage 33 Colab notebook.

The validator deliberately does not execute the GPU experiment.  It verifies
that the checked-in notebook is a deterministic rendering of its builder and
that the rendered source preserves the preregistered provenance, split, model,
intervention, evaluation, and resumability contracts.
"""

import ast
import builtins
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "33_bounded_interventional_predictive_causal_abstraction.ipynb"
BUILDER = ROOT / "build_stage33_bounded_interventional_abstraction_notebook.py"


def source(cell):
    return "".join(cell.get("source", []))


def assigned_value(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return static_value(node.value)
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name:
                return static_value(node.value)
    raise AssertionError(f"missing literal assignment {name}")


def static_value(node):
    """Evaluate the small, side-effect-free expression subset used in config."""

    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        pass
    if isinstance(node, ast.List):
        return [static_value(value) for value in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(static_value(value) for value in node.elts)
    if isinstance(node, ast.Dict):
        return {
            static_value(key): static_value(value)
            for key, value in zip(node.keys, node.values)
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = static_value(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mult)):
        left, right = static_value(node.left), static_value(node.right)
        return left + right if isinstance(node.op, ast.Add) else left * right
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "range"
        and not node.keywords
    ):
        return range(*(static_value(argument) for argument in node.args))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "list"
        and len(node.args) == 1
        and not node.keywords
    ):
        return list(static_value(node.args[0]))
    raise AssertionError(f"unsupported nonliteral configuration expression: {ast.dump(node)}")


def function_source(cells, name):
    for text in cells:
        tree = ast.parse(text)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name == name:
                    return ast.get_source_segment(text, node)
    raise AssertionError(f"missing function {name}")


def require_all(text, fragments, section):
    for fragment in fragments:
        assert fragment in text, f"{section}: missing {fragment!r}"


def defined_names(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(
            node.ctx, (ast.Store, ast.Param)
        ):
            names.add(node.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name)
    return names


def loaded_names(tree):
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


def validate_obvious_names(code_cells):
    """Catch unresolved protocol constants while tolerating notebook globals.

    Full Python name resolution is intentionally out of scope for a static
    notebook check.  Upper-case names are, however, the protocol/configuration
    interface and should always be defined in one of the rendered cells.
    """

    trees = [ast.parse(text) for text in code_cells]
    defined = set(dir(builtins))
    for tree in trees:
        defined.update(defined_names(tree))
    loaded = set().union(*(loaded_names(tree) for tree in trees))
    missing = sorted(
        name for name in loaded if name.isupper() and name not in defined
    )
    assert not missing, f"obvious undefined protocol names: {missing}"


def validate_protocol_digest(notebook, config, observed_digest):
    protocol_sources = [source(cell).strip() for cell in notebook["cells"]]
    replaced = False
    for index, text in enumerate(protocol_sources):
        if observed_digest in text and "NOTEBOOK_PROTOCOL_SHA256" in text:
            protocol_sources[index] = text.replace(
                observed_digest, "__PROTOCOL_DIGEST__", 1
            )
            replaced = True
            break
    assert replaced, "could not reconstruct the protocol digest source"
    expected = hashlib.sha256(
        json.dumps(
            protocol_sources, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()
    assert observed_digest == expected, "notebook protocol digest is stale"
    assert "__PROTOCOL_DIGEST__" not in config


def validate():
    assert BUILDER.is_file(), f"missing Stage 33 builder: {BUILDER}"
    assert NOTEBOOK.is_file(), f"missing rendered Stage 33 notebook: {NOTEBOOK}"

    before = NOTEBOOK.read_bytes()
    subprocess.run(
        [sys.executable, str(BUILDER)],
        check=True,
        capture_output=True,
        env=dict(os.environ),
    )
    after = NOTEBOOK.read_bytes()
    assert before == after, "Stage 33 builder is not deterministic"

    notebook = json.loads(after)
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] == 5
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["metadata"]["colab"]["gpuType"] == "L4"
    assert notebook["metadata"]["kernelspec"] == {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    assert len(notebook["cells"]) == 14
    assert notebook["cells"][0]["cell_type"] == "markdown"
    assert source(notebook["cells"][0]).startswith(
        "# Stage 33: bounded interventional predictive causal abstraction\n"
    )
    code_cells = [
        source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    assert len(code_cells) == 13
    assert code_cells[0].startswith(
        "# SINGLE CONFIGURATION BLOCK — no Stage 33 secrets required.\n"
    )
    assert [cell.splitlines()[0] for cell in code_cells[5:]] == [
        "# Freeze trajectory families and action compositions before simulator or model access.",
        "# Select complete physical trajectories and materialize exact multi-step truth without model access.",
        "# Fit grounded readouts, predictive charts, and carrier bases on construction trajectories only.",
        "# Lock rank, operator class, and regularization on model-selection trajectories; fit maps on calibration only.",
        "# Open the locked evaluation once and score multi-step realization and conjugacy controls.",
        "# Transport reachable internal responses through the single frozen predictive map and test planning.",
        "# Apply preregistered cumulative gates, multiplicity correction, and automatic interpretation.",
        "# Package compact audit evidence while retaining the complete resumable Drive directory.",
    ]
    for index, cell in enumerate(notebook["cells"]):
        assert cell["id"] == f"stage33-{index:02d}"
        assert not cell.get("outputs")
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            ast.parse(source(cell))
    validate_obvious_names(code_cells)

    config = code_cells[0]
    tree = ast.parse(config)
    assert assigned_value(tree, "PROTOCOL_ID") == (
        "stage33-bounded-interventional-predictive-causal-abstraction-v2"
    )
    assert assigned_value(tree, "RUN_MODE") == "pilot"
    assert assigned_value(tree, "EXPERIMENT_SOURCE_REF") == (
        "codex/stage33-bounded-interventional-abstraction"
    )
    assert assigned_value(tree, "MODEL_NAMES") == [
        "jepa_wm_pusht", "dino_wm_pusht"
    ]
    assert assigned_value(tree, "EXPECTED_MODEL_TYPES") == {
        "jepa_wm_pusht": "AdaLN", "dino_wm_pusht": "dino_wm"
    }
    assert assigned_value(tree, "EXPECTED_CARRIER_WIDTHS") == {
        "jepa_wm_pusht": 400, "dino_wm_pusht": 414
    }
    assert assigned_value(tree, "INTERVENTION_BLOCK") == 4
    assert assigned_value(tree, "REPO_COMMIT") == (
        "13cf1d9c7e476f53c17714d2e0f1dc239a883ce0"
    )
    assert assigned_value(tree, "EXPECTED_HF_REVISION") == (
        "9b9c41ef249466630dbf1a20e78391865d07b3b9"
    )
    assert assigned_value(tree, "EXPECTED_PRETRAINED_ASSET_SHA256") == {
        "jepa_wm_pusht.pth.tar": (
            "9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb"
        ),
        "dino_wm_pusht.pth.tar": (
            "8ec9cb05f22812d7f12e3c216b0637f41641055c0653e503e2746edb981b550f"
        ),
        "dinov2_vits14_pretrain.pth": (
            "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
        ),
    }
    assert assigned_value(tree, "ASSET_SPECS") == {}
    assert assigned_value(tree, "MODE_LABELS") == [
        "free", "pre_contact", "contact", "post_contact"
    ]
    assert assigned_value(tree, "TRAJECTORY_GEOMETRY_VERSION") == (
        "absolute_golden_angle_v2"
    )
    assert assigned_value(tree, "TRAJECTORY_PHASE_INCREMENT") == (
        0.6180339887498949
    )
    assert assigned_value(tree, "CONSTRUCTION_TRAJECTORIES") == 8
    assert assigned_value(tree, "MODEL_SELECTION_TRAJECTORIES") == 8
    assert assigned_value(tree, "CALIBRATION_TRAJECTORIES") == 8
    assert assigned_value(tree, "EVALUATION_TRAJECTORIES") == 16
    assert assigned_value(tree, "STATES_PER_TRAJECTORY") == 4
    assert assigned_value(tree, "MAX_WORD_LENGTH") == 4
    assert assigned_value(tree, "NONLINEAR_RANDOM_FEATURES") == 208
    assert assigned_value(tree, "MIN_EVALUATION_TRAJECTORIES") == 12
    assert assigned_value(tree, "ASSET_SPECS") == {}

    construction_pool = assigned_value(tree, "CONSTRUCTION_TRAJECTORY_POOL")
    model_selection_pool = assigned_value(tree, "MODEL_SELECTION_TRAJECTORY_POOL")
    calibration_pool = assigned_value(tree, "CALIBRATION_TRAJECTORY_POOL")
    evaluation_pool = assigned_value(tree, "EVALUATION_TRAJECTORY_POOL")
    assert construction_pool == list(range(6000, 6800))
    assert model_selection_pool == list(range(6800, 7600))
    assert calibration_pool == list(range(7600, 8400))
    assert evaluation_pool == list(range(8400, 10000))
    pools = [construction_pool, model_selection_pool, calibration_pool, evaluation_pool]
    assert all(
        not set(pools[left]) & set(pools[right])
        for left in range(len(pools)) for right in range(left + 1, len(pools))
    )
    evaluation_words = assigned_value(tree, "EVALUATION_WORD_SPECS")
    assert {len(row["angles"]) for row in evaluation_words} == {1, 2, 3, 4}
    assert {-40.0, -20.0, 20.0, 40.0}.issubset(
        {angle for row in evaluation_words for angle in row["angles"]}
    )
    require_all(
        config,
        [
            "token_hex(4)", '_colab_userdata.get("HF_TOKEN")',
            "RESUME_INCOMPLETE = True", "hash_validated_resume",
            "no_synthetic_fallback", "one_map_all_actions_modes_steps",
            "construction_rank_and_decoder_only",
            "model_selection",
            "calibration_operators_and_map_only", "locked_evaluation",
            "model_native_internal_interchange", "planning_transport",
            "shared_dinov2_target_is_a_declared_confound",
            "model_free_v1_coverage_amendment",
            "stable_trajectory_id_geometry",
        ],
        "configuration contract",
    )

    joined = "\n".join(code_cells)
    require_all(
        joined,
        [
            "def select_stable_rank(", "def fit_grouped_ridge(",
            "def fit_affine_bilinear_operator(",
            "def compose_affine_bilinear(", "def fit_whitened_similarity(",
            "def operator_intertwining_metrics(",
            "def reachability_observability_diagnostics(",
            "def interchange_metrics(", "def clustered_bootstrap_interval(",
            "def holm_adjust(", "def derive_decision(",
            "def forward_with_trace(", "register_forward_hook",
            "intervention_by_step", "reconstruct_carrier_delta(",
            "WITHIN_MODEL_BRIDGES", '"cross_model_map_count": 1',
            "carrier_J_to_delta_q_J_to_S_delta_q_J_to_delta_q_D_to_carrier_D",
            "empirical_simulator_noise_floor", "zero_edit_max_abs",
            "MAX_PLANNING_REGRET_DEGRADATION", "planning",
            "stage33_bipca_result_bundle_",
        ],
        "bounded causal abstraction implementation",
    )
    for prohibited in [
        "EXPECTED_STAGE31", "EXPECTED_STAGE32", "STAGE31_SUBSPACE",
        "STAGE32_SUBSPACE", "stage31_cross_model_certificate_result_bundle_",
        "stage32_bounded_confirmation_result_bundle_", "import_stage31",
        "import_stage32", "DummyModel", "FakeModel", "mock_predictions",
        "synthetic_predictions", "torch.autograd", ".backward(",
        "torch.func.jvp", "torch.func.vjp", "jacrev", "jacfwd",
        "CARRIER_MAPS", "frozen_carrier_map",
    ]:
        assert prohibited not in joined, f"prohibited inherited/fallback machinery: {prohibited}"

    hook_source = function_source(code_cells, "forward_with_trace")
    require_all(
        hook_source,
        [
            'hook_kind == "direct"', "inputs[0] + output",
            "intervention_by_step.get", "handle.remove()",
            "len(captures) != horizon",
        ],
        "real recurrent hook transport",
    )
    assert joined.count("forward_with_trace(") >= 2, (
        "forward_with_trace is defined but never used for a real intervention"
    )

    require_all(
        joined,
        [
            "created_before_model_loading", "entire_trajectory_disjoint",
            "selection_uses_contact_timing_only", '"model_outputs_used": False',
            '"decoder_training_split": "construction"',
            '"rank_training_split": "construction"',
            '"carrier_training_split": "construction"',
            '"evaluation_rows_used": 0', "calibration", "locked",
            "EVALUATION_INTERCHANGE_PAIRS", "EVALUATION_WORD_SPECS",
        ],
        "construction/model-selection/calibration/evaluation separation",
    )
    geometry_source = function_source(code_cells, "initial_trajectory_record")
    require_all(
        geometry_source,
        [
            "TRAJECTORY_PHASE_INCREMENT",
            "trajectory_geometry_version",
            "37 * trajectory_id + DESIGN_SEED",
        ],
        "stable trajectory-id geometry",
    )
    assert "pool.index" not in geometry_source
    selector_source = function_source(code_cells, "select_complete_trajectories")
    assert selector_source.index("selected.extend(snapshots)") < selector_source.index(
        'write_json(OUT / f"physical_screen_{split}_progress.json"'
    )
    assert selector_source.index(
        'write_csv(EVIDENCE_DIR / f"physical_screen_{split}_rows.csv"'
    ) < selector_source.index(
        'f"{split} produced fewer than {target} complete four-mode trajectories"'
    )
    require_all(
        joined,
        [
            "INCOMPLETE_POINTER", "RESUME_KEY", "RESUMED_RUN",
            "def atomic_checkpoint(", "def validated_checkpoint(",
            "def validate_npz_shard(", "protocol_sha256",
            "run_signature", "identity mismatch", "sha256_file",
            "FAILURE_TRACE.txt",
        ],
        "resumability and hash validation",
    )
    require_all(
        joined,
        [
            "MIN_HYBRID_RELATIVE_GAIN", "MAX_CONJUGACY_RELATIVE_ERROR",
            "MIN_CONTROL_ADVANTAGE", "MIN_GROUNDED_INTERCHANGE_COSINE",
            "MIN_INTERCHANGE_RELATIVE_ERROR_GAIN",
            "MAX_PLANNING_REGRET_DEGRADATION", "HOLM_ALPHA",
        ],
        "frozen causal and planning gates",
    )

    observed_digest = assigned_value(tree, "NOTEBOOK_PROTOCOL_SHA256")
    validate_protocol_digest(notebook, config, observed_digest)
    print("Stage 33 notebook validation passed")


if __name__ == "__main__":
    validate()
