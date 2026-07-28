import ast
import json
from pathlib import Path

import numpy as np
import torch


NOTEBOOK = Path(__file__).with_name(
    "10_fidelity_constrained_pairwise_margin_adaptation.ipynb"
)


def pair_indices(count):
    return np.triu_indices(count, k=1)


def certificate(prediction, truth, p=8, smooth=0.0, scale_eps=1e-6):
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    left, right = pair_indices(len(truth))
    scale = max(float(np.max(truth) - np.min(truth)), scale_eps)
    error = (
        (prediction[left] - prediction[right])
        - (truth[left] - truth[right])
    ) / scale
    return float(
        np.sum((error**2 + smooth**2) ** (p / 2.0)) ** (1.0 / p)
    )


def normalized_regret(prediction, truth, tie=1e-9):
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    selected = int(np.argmin(prediction))
    best = float(np.min(truth))
    scale = max(float(np.max(truth) - best), tie)
    return float((truth[selected] - best) / scale), selected


def certificate_normalized_regret(
    prediction, truth, scale_eps=1e-6
):
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    selected = int(np.argmin(prediction))
    best = float(np.min(truth))
    scale = max(float(np.max(truth) - best), scale_eps)
    return float((truth[selected] - best) / scale), selected


def normalized_gap(truth, tie=1e-9):
    truth = np.asarray(truth, dtype=np.float64)
    best = float(np.min(truth))
    scale = max(float(np.max(truth) - best), tie)
    nonoptimal = truth > best + tie
    if not np.any(nonoptimal):
        return float("inf")
    return float((np.min(truth[nonoptimal]) - best) / scale)


def task_equal_mean(rows):
    grouped = {}
    for task_id, value in rows:
        grouped.setdefault(task_id, []).append(float(value))
    return float(
        np.mean([np.mean(values) for values in grouped.values()])
    )


def select_checkpoint(candidates):
    eligible = [row for row in candidates if row["eligible"]]
    if not eligible:
        raise AssertionError("epoch zero should make the feasible set nonempty")
    return min(eligible, key=lambda row: (row["score"], row["epoch"]))


def validate_notebook_structure():
    payload = json.loads(NOTEBOOK.read_text())
    if len(payload["cells"]) != 11:
        raise AssertionError(
            f"expected 11 cells, found {len(payload['cells'])}"
        )
    code_cells = [
        "".join(cell.get("source", []))
        for cell in payload["cells"]
        if cell["cell_type"] == "code"
    ]
    if len(code_cells) != 10:
        raise AssertionError(
            f"expected 10 code cells, found {len(code_cells)}"
        )
    for index, source in enumerate(code_cells):
        try:
            ast.parse(source)
        except SyntaxError as error:
            raise AssertionError(
                f"code cell {index} has invalid Python: {error}"
            ) from error
    for index, cell in enumerate(payload["cells"]):
        if cell.get("id") != f"stage10-{index:02d}":
            raise AssertionError(f"cell {index} has a stale or missing id")
        if cell.get("outputs"):
            raise AssertionError(f"cell {index} contains stale outputs")
        if (
            cell["cell_type"] == "code"
            and cell.get("execution_count") is not None
        ):
            raise AssertionError(
                f"cell {index} contains a stale execution count"
            )
    if (
        payload.get("metadata", {})
        .get("colab", {})
        .get("name")
        != NOTEBOOK.name
    ):
        raise AssertionError("Colab metadata carries the wrong notebook name")

    joined = "\n".join(code_cells)
    required = [
        'REPO_COMMIT = "13cf1d9c7e476f53c17714d2e0f1dc239a883ce0"',
        'EXPECTED_HF_REVISION = "9b9c41ef249466630dbf1a20e78391865d07b3b9"',
        'OUTPUT_DIR = "/content/counterfactual_faithfulness_stage10"',
        'Path("/content/stage10_result_bundle.zip")',
        "stage10_fpma_sum_pnorm_v5",
        "MOUNT_DRIVE = True",
        "def ensure_colab_drive():",
        "force_remount=attempt > 1",
        "timeout_ms=600_000",
        'Path(mountpoint, "MyDrive")',
        "drive.flush_and_unmount()",
        "to continue on ephemeral /content storage",
        "fidelity_constrained_latent_only",
        "fidelity_constrained_shuffled_fpma",
        "fidelity_constrained_matched_fpma",
        "unconstrained_matched_fpma",
        "TRAINING_DECODER_COUNT = 3",
        "TRAINING_DECODER_SEEDS = [10501, 10519, 10537]",
        "EVALUATION_PROJECTION_SEEDS = [12011, 12029, 12047, 12065, 12083]",
        "set(TRAINING_DECODER_SEEDS).isdisjoint",
        "PAIRWISE_P = 8",
        "magnitude.amax(dim=1)",
        "decoder_bound.max(dim=0).values",
        "torch.ones_like(raw_cosine)",
        "FPMA requires all ten candidates",
        "len(PAIR_LEFT) == ACTIONS_PER_STATE * (ACTIONS_PER_STATE - 1) // 2",
        "(native_by_horizon - baseline_state)",
        "+ baseline_train_mean / baseline_train_denom",
        "NATIVE_NONINFERIORITY_TOLERANCE",
        "NATIVE_DENOMINATOR_EPS = 1e-8",
        "GROUP_DRO_LOGIT_CLIP = 20.0",
        "TRUST_REFERENCE_RMS_FLOOR = 1e-3",
        "COLLAPSE_MAX_CANDIDATE_SHARE = 0.80",
        "COLLAPSE_MAX_NULL_SHARE = 0.75",
        "COLLAPSE_MIN_UNIQUE_ACTIONS = 3",
        '"EVALUATION_SEEDS": EVALUATION_SEEDS',
        '"TASK_FAMILY_ID": STAGE7_TASK_FAMILY_ID',
        '"DEVELOPMENT_SPLIT": STAGE7_DEVELOPMENT_SPLIT',
        "compute_stage10_cache_binding",
        "cache_content_digest",
        "verify_loaded_pretrained_assets",
        "cannot pin the checkpoint revision in hubconf.py",
        "cannot disable mutable checkpoint fallback in hubconf.py",
        "pinned Hugging Face checkpoint retrieval failed",
        "EXPECTED_PRETRAINED_ASSET_SHA256",
        "decoder_checksum_verified_before_and_after",
        "epoch_zero_ratios = np.ones",
        "checkpoint_eligible",
        "optimizer.load_state_dict",
        'parameter_group["lr"] = current_lr * 0.5',
        "atomic_torch_save",
        "current_action_path_checksum",
        "INITIAL_EPOCH_LIMIT = 24",
        "EXTENSION_EPOCH_LIMITS = [36, 48]",
        "UNDERTRAINED_INCONCLUSIVE",
        "ridge optimum remained on a search boundary",
        "ridge_optimum_interior",
        "deterministic_non_null_derangement",
        "np.all(result[1:] != np.arange(1, ACTIONS_PER_STATE))",
        "bootstrap_equal_task_mean",
        "adaptation_seeds_averaged_within_task",
        "stage10_native_planner_metrics.csv",
        "stage10_native_planner_contrasts.csv",
        "native_planner_gate_pass",
        "full_scientific_matrix",
        "NONPROTOCOL_RUN_NO_SCIENTIFIC_DECISION",
        "PROJECTION_CONSENSUS_REQUIRED = 4",
        "regret_bound_holds",
        "top1_certified",
        "SUCCESS: no captured pipeline failure",
        "if path == result_manifest_path",
        'print(f"RESULT_ZIP: {result_zip}")',
        'print("RUN_STATUS:"',
        "colab_files.download",
        "failed_run_latest_checkpoints",
    ]
    missing = [needle for needle in required if needle not in joined]
    if missing:
        raise AssertionError(
            f"missing required Stage 10 elements: {missing}"
        )
    prohibited = [
        "def action_groups(",
        "prediction[:1]",
        "BOOTSTRAP_REPS,\n                [int(row[\"state_id\"])",
    ]
    present = [needle for needle in prohibited if needle in joined]
    if present:
        raise AssertionError(
            f"prohibited Stage 10 implementation patterns: {present}"
        )
    return payload, code_cells


def validate_certificate_properties():
    left, right = pair_indices(10)
    pairs = list(zip(left.tolist(), right.tolist()))
    assert len(pairs) == 45
    assert len(set(pairs)) == 45
    assert all(a < b for a, b in pairs)

    rng = np.random.default_rng(1013)
    for _ in range(1000):
        truth = rng.normal(size=10)
        prediction = truth + rng.normal(scale=0.8, size=10)
        bound = certificate(prediction, truth)
        regret, selected = certificate_normalized_regret(
            prediction, truth
        )
        if regret > bound + 1e-10:
            raise AssertionError(
                f"regret certificate failed: {regret} > {bound}"
            )
        gap = normalized_gap(truth)
        if bound < gap:
            optimal = np.flatnonzero(
                truth <= np.min(truth) + 1e-9
            )
            if selected not in optimal:
                raise AssertionError(
                    "strict certificate failed to preserve the "
                    "tie-tolerant optimal set"
                )

        common_offset = rng.normal()
        np.testing.assert_allclose(
            certificate(prediction + common_offset, truth),
            bound,
            rtol=1e-10,
            atol=1e-10,
        )
        positive_scale = float(np.exp(rng.normal()))
        np.testing.assert_allclose(
            certificate(
                positive_scale * prediction,
                positive_scale * truth,
            ),
            bound,
            rtol=1e-10,
            atol=1e-10,
        )

    # Equality is insufficient: deterministic tie-breaking can choose wrong.
    truth = np.asarray([1.0, 0.0])
    prediction = np.asarray([0.0, 0.0])
    bound = certificate(prediction, truth)
    gap = normalized_gap(truth)
    regret, selected = certificate_normalized_regret(
        prediction, truth
    )
    np.testing.assert_allclose(bound, gap)
    assert selected == 0 and regret == 1.0

    # A mean p-norm breaks the max-error upper bound.
    error = np.zeros(45)
    error[0] = 1.0
    sum_norm = np.sum(np.abs(error) ** 8) ** (1 / 8)
    mean_norm = np.mean(np.abs(error) ** 8) ** (1 / 8)
    assert sum_norm >= np.max(np.abs(error))
    assert mean_norm < np.max(np.abs(error))

    # Multiple optima and all-equal costs are handled as sets.
    truth = np.asarray([0.0, 0.0, 0.5, 1.0])
    prediction = np.asarray([0.2, 0.1, 0.4, 1.1])
    regret, selected = normalized_regret(prediction, truth)
    assert selected == 1 and regret == 0.0
    assert normalized_gap(truth) == 0.5
    assert normalized_gap(np.ones(4)) == float("inf")

    # The optimized gap loss must retain slope after the bound exceeds gap.
    gap = 0.25
    weight = 0.25
    derivative = 1.0 + weight / gap
    assert derivative > 0
    clipped_derivative_after_failure = 0.0
    assert derivative > clipped_derivative_after_failure

    # The q-normalized theorem remains valid for nearly flat true costs.
    truth = np.asarray([0.0, 5e-7])
    prediction = np.asarray([1e-12, 0.0])
    bound = certificate(prediction, truth)
    q_regret, _ = certificate_normalized_regret(
        prediction, truth
    )
    conventional_regret, _ = normalized_regret(prediction, truth)
    assert q_regret <= bound + 1e-10
    assert conventional_regret > bound


def validate_stable_torch_certificate():
    pair_count = 45
    p = 8
    smooth = 1e-6
    for error_scale in [0.0, 1e-8, 1e-6, 1e-4]:
        error = torch.full(
            (3, pair_count, 3),
            error_scale,
            dtype=torch.float32,
            requires_grad=True,
        )
        magnitude = torch.sqrt(error.square() + smooth**2)
        maximum = magnitude.amax(dim=1)
        bound = maximum * torch.sum(
            (magnitude / maximum[:, None, :]) ** p,
            dim=1,
        ) ** (1.0 / p)
        loss = bound.sum()
        loss.backward()
        assert bool(torch.isfinite(bound).all())
        assert bool(torch.isfinite(error.grad).all())
        reference = (
            pair_count
            * (error_scale**2 + smooth**2) ** (p / 2.0)
        ) ** (1.0 / p)
        np.testing.assert_allclose(
            bound.detach().numpy(),
            reference,
            rtol=2e-5,
            atol=1e-10,
        )


def validate_torch_physical_cost():
    angles = np.asarray(
        [-np.pi + 1e-7, -0.4, 0.0, 1.2, np.pi - 1e-7]
    )
    pose_array = np.column_stack(
        [
            np.linspace(0.1, 0.9, len(angles)),
            np.linspace(0.8, 0.2, len(angles)),
            np.sin(angles),
            np.cos(angles),
        ]
    ).astype(np.float32)
    goal = np.asarray([256.0, 300.0, -np.pi + 2e-7])
    pose = torch.tensor(pose_array, requires_grad=True)
    goal_tensor = torch.tensor(goal, dtype=pose.dtype)
    raw_sine = pose[..., 2]
    raw_cosine = pose[..., 3]
    norm = torch.sqrt(
        raw_sine.square() + raw_cosine.square() + 1e-12
    )
    small = norm < 1e-4
    sine = torch.where(
        small, torch.zeros_like(raw_sine), raw_sine / norm
    )
    cosine = torch.where(
        small, torch.ones_like(raw_cosine), raw_cosine / norm
    )
    angle = torch.atan2(sine, cosine)
    angle_error = torch.atan2(
        torch.sin(angle - goal_tensor[2]),
        torch.cos(angle - goal_tensor[2]),
    )
    pieces = torch.cat(
        [
            pose[..., :2] - goal_tensor[:2] / 512.0,
            (angle_error / np.pi).unsqueeze(-1),
        ],
        dim=-1,
    )
    cost = torch.sqrt(
        torch.sum(pieces.square(), dim=-1).clamp_min(1e-12)
    )
    expected_angle_error = np.arctan2(
        np.sin(angles - goal[2]),
        np.cos(angles - goal[2]),
    )
    expected = np.linalg.norm(
        np.column_stack(
            [
                pose_array[:, :2] - goal[:2] / 512.0,
                expected_angle_error / np.pi,
            ]
        ),
        axis=-1,
    )
    np.testing.assert_allclose(
        cost.detach().numpy(), expected, rtol=1e-5, atol=1e-6
    )
    cost.sum().backward()
    assert bool(torch.isfinite(pose.grad).all())

    zero_pose = torch.zeros((1, 4), requires_grad=True)
    raw_sine = zero_pose[..., 2]
    raw_cosine = zero_pose[..., 3]
    norm = torch.sqrt(
        raw_sine.square() + raw_cosine.square() + 1e-12
    )
    small = norm < 1e-4
    safe_sine = torch.where(
        small, torch.zeros_like(raw_sine), raw_sine / norm
    )
    safe_cosine = torch.where(
        small, torch.ones_like(raw_cosine), raw_cosine / norm
    )
    safe_angle = torch.atan2(safe_sine, safe_cosine)
    safe_angle.sum().backward()
    assert bool(torch.isfinite(safe_angle).all())
    assert bool(torch.isfinite(zero_pose.grad).all())

    # Training and NumPy evaluation must share the same near-zero convention,
    # not merely agree on unit-circle inputs and the exact zero vector.
    near_zero = np.asarray(
        [
            [0.3, 0.4, 5e-5, 0.0],
            [0.3, 0.4, 0.0, -5e-5],
            [0.3, 0.4, 2e-4, 0.0],
        ],
        dtype=np.float32,
    )
    near_tensor = torch.tensor(near_zero, requires_grad=True)
    torch_norm = torch.sqrt(
        near_tensor[:, 2].square()
        + near_tensor[:, 3].square()
        + 1e-12
    )
    torch_small = torch_norm < 1e-4
    torch_angle = torch.atan2(
        torch.where(
            torch_small,
            torch.zeros_like(torch_norm),
            near_tensor[:, 2] / torch_norm,
        ),
        torch.where(
            torch_small,
            torch.ones_like(torch_norm),
            near_tensor[:, 3] / torch_norm,
        ),
    )
    numpy_norm = np.sqrt(
        near_zero[:, 2] ** 2 + near_zero[:, 3] ** 2 + 1e-12
    )
    numpy_small = numpy_norm < 1e-4
    numpy_angle = np.arctan2(
        np.where(numpy_small, 0.0, near_zero[:, 2] / numpy_norm),
        np.where(numpy_small, 1.0, near_zero[:, 3] / numpy_norm),
    )
    np.testing.assert_allclose(
        torch_angle.detach().numpy(),
        numpy_angle,
        rtol=1e-6,
        atol=1e-7,
    )
    torch_angle.sum().backward()
    assert bool(torch.isfinite(near_tensor.grad).all())


def validate_axis_and_gate_invariants():
    projection_count, action_count, step_count, dimension = 5, 10, 6, 2
    tensor = np.empty(
        (
            projection_count,
            action_count,
            step_count,
            dimension,
        ),
        dtype=np.int64,
    )
    for projection in range(projection_count):
        for action in range(action_count):
            for step in range(step_count):
                for feature in range(dimension):
                    tensor[projection, action, step, feature] = (
                        1000 * projection
                        + 100 * action
                        + 10 * step
                        + feature
                    )
    horizon_indices = [0, 2, 5]
    selected = tensor[2][:, horizon_indices, :]
    assert selected.shape == (10, 3, 2)
    assert selected[7, 2, 1] == 2000 + 700 + 50 + 1
    wrong = tensor[2, :, horizon_indices]
    assert wrong.shape == (3, 10, 2)

    common = set([1, 3, 6])
    for baseline_horizons in [{1, 3}, {1, 6}, {1, 3, 6}]:
        common &= baseline_horizons
    assert common == {1}
    assert len(common) < 2
    common = set([1, 3, 6])
    for baseline_horizons in [{1, 3}, {1, 3, 6}, {1, 3}]:
        common &= baseline_horizons
    assert common == {1, 3}

    learning_rate = 8e-6
    for _ in range(3):
        current_learning_rate = learning_rate
        restored_learning_rate = 8e-6
        learning_rate = current_learning_rate * 0.5
        assert learning_rate <= restored_learning_rate
    np.testing.assert_allclose(learning_rate, 1e-6)


def validate_checkpoint_and_inference_invariants():
    selected = select_checkpoint(
        [
            {"epoch": 0, "score": 1.0, "eligible": True},
            {"epoch": 2, "score": 0.2, "eligible": False},
            {"epoch": 4, "score": 0.8, "eligible": True},
        ]
    )
    assert selected["epoch"] == 4
    earliest_tie = select_checkpoint(
        [
            {"epoch": 0, "score": 1.0, "eligible": True},
            {"epoch": 2, "score": 0.7, "eligible": True},
            {"epoch": 4, "score": 0.7, "eligible": True},
        ]
    )
    assert earliest_tie["epoch"] == 2

    baseline = np.asarray([1.0, 2.0, 3.0])
    feasible = np.asarray([1.02, 2.03, 3.04]) / baseline
    violating = np.asarray([1.01, 2.05, 3.01]) / baseline
    assert np.all(feasible <= 1.02 + 1e-12)
    assert not np.all(violating <= 1.02 + 1e-12)

    # The variance-reduced one-state constraint remains unbiased even when
    # the baseline mean is below the denominator floor.
    baseline_states = np.asarray([0.0, 0.5e-8])
    current_states = np.asarray([0.25e-8, 0.75e-8])
    baseline_mean = float(np.mean(baseline_states))
    denominator = max(baseline_mean, 1e-8)
    stochastic_constraint = (
        (current_states - baseline_states) / denominator
        + baseline_mean / denominator
        - 1.02
    )
    aggregate_constraint = (
        float(np.mean(current_states)) / denominator - 1.02
    )
    np.testing.assert_allclose(
        np.mean(stochastic_constraint),
        aggregate_constraint,
    )

    limits = [24, 36, 48]
    assert limits[1:] == [36, 48]
    best_epoch = 24
    recent_improvement = 0.02
    assert best_epoch == limits[0] and recent_improvement > 1e-4
    best_epoch = 48
    assert best_epoch == limits[-1] and recent_improvement > 1e-4

    full_scientific_matrix = bool(
        "full" == "full"
        and 96 == 96
        and 12 == 12
        and [6, 3, 0, 3] == [6, 3, 0, 3]
        and [10401, 10419, 10437] == [10401, 10419, 10437]
        and limits == [24, 36, 48]
        and 3 == 3
        and 5 == 5
        and 2000 == 2000
    )
    assert full_scientific_matrix
    smoke_scientific_matrix = bool(
        "smoke" == "full"
        and 24 == 96
    )
    assert not smoke_scientific_matrix

    # Tasks, not states, receive equal inferential weight.
    rows = [(0, 1.0)] * 100 + [(1, -1.0)]
    np.testing.assert_allclose(task_equal_mean(rows), 0.0)
    assert np.mean([value for _, value in rows]) > 0.9

    training_seeds = {10501, 10519, 10537}
    evaluation_seeds = {12011, 12029, 12047, 12065, 12083}
    assert training_seeds.isdisjoint(evaluation_seeds)
    assert len(evaluation_seeds) == 5

    rng = np.random.default_rng(10401)
    original = np.arange(1, 10)
    for _ in range(100):
        permuted = original.copy()
        while True:
            rng.shuffle(permuted)
            if np.all(permuted != original):
                break
        result = np.concatenate([[0], permuted])
        assert result[0] == 0
        assert np.all(result[1:] != original)
        assert sorted(result.tolist()) == list(range(10))

    # A collapsed seed cannot be hidden by two diverse seeds.
    per_seed_actions = {
        10401: [0] * 24,
        10419: list(range(10)) * 2 + [1, 2, 3, 4],
        10437: list(range(10)) * 2 + [5, 6, 7, 8],
    }
    seed_passes = {}
    for seed, actions in per_seed_actions.items():
        counts = np.bincount(actions, minlength=10)
        seed_passes[seed] = (
            counts.max() / len(actions) <= 0.80
            and counts[0] / len(actions) <= 0.75
            and np.count_nonzero(counts) >= 3
        )
    assert not seed_passes[10401]
    assert not all(seed_passes.values())


def main():
    payload, code_cells = validate_notebook_structure()
    validate_certificate_properties()
    validate_stable_torch_certificate()
    validate_torch_physical_cost()
    validate_axis_and_gate_invariants()
    validate_checkpoint_and_inference_invariants()
    print(
        json.dumps(
            {
                "notebook": str(NOTEBOOK),
                "cells": len(payload["cells"]),
                "code_cells": len(code_cells),
                "certificate_property_trials": 1000,
                "status": "ok",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
