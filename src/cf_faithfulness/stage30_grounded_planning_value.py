"""Numerical primitives for Stage 30 grounded causal planning value.

Stage 30 asks whether a causally effective internal action carrier is useful
for *physical* counterfactual choice.  GPU/model execution stays in the Colab
notebook; this module keeps the closure, terminal-planning, and grouped
cross-fitting estimands independently testable with NumPy.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from .stage17_action_contrast import decoded_task_cost, pose_target, ranking_metrics
from .stage29_grounded_closure import vector_alignment


def _validated_layout(values, magnitude_count, schedule_count, name):
    array = np.asarray(values, dtype=np.float64)
    expected = int(magnitude_count) * int(schedule_count)
    if array.ndim < 2 or len(array) != expected:
        raise ValueError(
            f"{name} must have {expected} action rows; found {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains nonfinite values")
    return array.reshape(int(magnitude_count), int(schedule_count), *array.shape[1:])


def schedule_reversal(schedule_count):
    """Return the frozen signed-area reversal over schedule positions."""

    count = int(schedule_count)
    if count < 2 or count % 2:
        raise ValueError("schedule_count must be a positive even integer")
    return np.arange(count - 1, -1, -1, dtype=np.int64)


def diagnostic_closure_rows(
    baseline,
    patched,
    target,
    magnitude_count,
    schedule_count,
    diagnostic_schedules=(1, 2, 3, 4),
    mode="swap",
):
    """Measure self and grounded closure on schedules excluded as goal actions.

    The planning goals use the two extreme schedules (0 and S-1).  Closure is
    estimated only on the interior schedules by default, preventing the
    primary planning test from reusing its goal-action contrast as the closure
    target.
    """

    base = _validated_layout(
        baseline, magnitude_count, schedule_count, "baseline"
    )
    edit = _validated_layout(patched, magnitude_count, schedule_count, "patched")
    truth = _validated_layout(target, magnitude_count, schedule_count, "target")
    if base.shape != edit.shape or base.shape != truth.shape:
        raise ValueError("baseline, patched, and target must have equal shapes")
    schedules = np.asarray(tuple(int(value) for value in diagnostic_schedules))
    if (
        schedules.ndim != 1
        or len(schedules) == 0
        or len(np.unique(schedules)) != len(schedules)
        or np.any(schedules < 0)
        or np.any(schedules >= int(schedule_count))
    ):
        raise ValueError("diagnostic schedules are malformed")
    reversal = schedule_reversal(schedule_count)
    if set(schedules.tolist()) != set(reversal[schedules].tolist()):
        raise ValueError("diagnostic schedules must be closed under reversal")
    if mode not in {"swap", "ablation"}:
        raise ValueError("mode must be 'swap' or 'ablation'")

    rows = []
    for magnitude_index in range(int(magnitude_count)):
        observed = edit[magnitude_index, schedules] - base[magnitude_index, schedules]
        if mode == "swap":
            self_target = (
                base[magnitude_index, reversal[schedules]]
                - base[magnitude_index, schedules]
            )
            ground_target = (
                truth[magnitude_index, reversal[schedules]]
                - truth[magnitude_index, schedules]
            )
        else:
            self_target = -0.5 * (
                base[magnitude_index, schedules]
                - base[magnitude_index, reversal[schedules]]
            )
            ground_target = -0.5 * (
                truth[magnitude_index, schedules]
                - truth[magnitude_index, reversal[schedules]]
            )
        self_metrics = vector_alignment(observed, self_target)
        grounded_metrics = vector_alignment(observed, ground_target)
        native_total = vector_alignment(
            base[magnitude_index], truth[magnitude_index]
        )
        base_centered = base[magnitude_index] - np.mean(
            base[magnitude_index], axis=0, keepdims=True
        )
        truth_centered = truth[magnitude_index] - np.mean(
            truth[magnitude_index], axis=0, keepdims=True
        )
        native_centered = vector_alignment(base_centered, truth_centered)
        rows.append(
            {
                "magnitude_index": int(magnitude_index),
                "mode": mode,
                "diagnostic_schedules": " ".join(str(v) for v in schedules),
                "effect_energy": float(np.sum(observed**2)),
                **{f"self_{key}": value for key, value in self_metrics.items()},
                **{
                    f"grounded_{key}": value
                    for key, value in grounded_metrics.items()
                },
                **{
                    f"native_total_{key}": value
                    for key, value in native_total.items()
                },
                **{
                    f"native_centered_{key}": value
                    for key, value in native_centered.items()
                },
            }
        )
    return rows


def native_terminal_costs(
    predicted,
    encoded_targets,
    magnitude_count,
    schedule_count,
    goal_schedule,
):
    """Terminal native JEPA cost to one exact encoded simulator goal."""

    prediction = _validated_layout(
        predicted, magnitude_count, schedule_count, "predicted"
    )
    target = _validated_layout(
        encoded_targets, magnitude_count, schedule_count, "encoded_targets"
    )
    goal_schedule = int(goal_schedule)
    if not 0 <= goal_schedule < int(schedule_count):
        raise ValueError("goal_schedule is out of range")
    difference = prediction - target[:, goal_schedule : goal_schedule + 1]
    axes = tuple(range(2, difference.ndim))
    return np.sqrt(np.mean(difference**2, axis=axes))


def physical_terminal_costs(
    endpoint_states,
    magnitude_count,
    schedule_count,
    goal_schedule,
):
    """Exact normalized PushT block-pose cost to a candidate endpoint goal."""

    states = _validated_layout(
        endpoint_states, magnitude_count, schedule_count, "endpoint_states"
    )
    if states.shape[-1] != 10:
        raise ValueError("PushT endpoint states must have ten coordinates")
    goal_schedule = int(goal_schedule)
    if not 0 <= goal_schedule < int(schedule_count):
        raise ValueError("goal_schedule is out of range")
    poses = pose_target(states)
    costs = []
    for magnitude_index in range(int(magnitude_count)):
        goal_state = states[magnitude_index, goal_schedule]
        goal = np.asarray(
            [goal_state[2], goal_state[3], goal_state[4]], dtype=np.float64
        )
        costs.append(decoded_task_cost(poses[magnitude_index], goal))
    return np.asarray(costs, dtype=np.float64)


def terminal_planning_rows(
    predicted,
    encoded_targets,
    endpoint_states,
    magnitude_count,
    schedule_count,
    goal_schedules=(0, 5),
    tie=1e-9,
):
    """Score an exhaustive terminal latent planner against simulator truth."""

    goals = tuple(int(value) for value in goal_schedules)
    if len(goals) == 0 or len(set(goals)) != len(goals):
        raise ValueError("goal schedules must be unique and nonempty")
    extremes = np.asarray([0, int(schedule_count) - 1], dtype=np.int64)
    rows = []
    for goal_schedule in goals:
        model_costs = native_terminal_costs(
            predicted,
            encoded_targets,
            magnitude_count,
            schedule_count,
            goal_schedule,
        )
        true_costs = physical_terminal_costs(
            endpoint_states,
            magnitude_count,
            schedule_count,
            goal_schedule,
        )
        for magnitude_index in range(int(magnitude_count)):
            model = model_costs[magnitude_index]
            truth = true_costs[magnitude_index]
            full = ranking_metrics(truth, model, tie=tie)
            strict = ranking_metrics(
                truth[extremes], model[extremes], tie=tie
            )
            selected = int(full["selected_action"])
            rows.append(
                {
                    "magnitude_index": int(magnitude_index),
                    "goal_schedule": int(goal_schedule),
                    "selected_schedule": selected,
                    "oracle_schedule": int(full["oracle_action"]),
                    "top1_correct": float(full["top1_correct"]),
                    "normalized_regret": float(full["normalized_regret"]),
                    "weighted_pairwise_accuracy": float(
                        full["weighted_pairwise_accuracy"]
                    ),
                    "selected_true_cost": float(truth[selected]),
                    "true_cost_spread": float(np.max(truth) - np.min(truth)),
                    "model_cost_spread": float(np.max(model) - np.min(model)),
                    "goal_prediction_error": float(model[goal_schedule]),
                    "strict_extreme_top1_correct": float(strict["top1_correct"]),
                    "strict_extreme_normalized_regret": float(
                        strict["normalized_regret"]
                    ),
                    "true_extreme_cost_spread": float(
                        np.max(truth[extremes]) - np.min(truth[extremes])
                    ),
                    "model_extreme_preference_margin": float(
                        abs(model[extremes[0]] - model[extremes[1]])
                    ),
                }
            )
    return rows


def deterministic_group_folds(groups, folds, seed):
    """Assign every group to one deterministic, approximately balanced fold."""

    labels = np.asarray(groups)
    unique = np.unique(labels)
    folds = int(folds)
    if folds < 2 or len(unique) < folds:
        raise ValueError("cross-fitting requires at least one group per fold")
    keyed = []
    for value in unique:
        digest = hashlib.sha256(f"{seed}|{value}".encode()).digest()
        keyed.append((int.from_bytes(digest[:8], "little"), value))
    ordered = [value for _, value in sorted(keyed, key=lambda item: item[0])]
    mapping = {value: index % folds for index, value in enumerate(ordered)}
    return np.asarray([mapping[value] for value in labels], dtype=np.int64)


def _ridge_prediction(train_x, train_y, test_x, ridge):
    mean = np.mean(train_x, axis=0)
    scale = np.std(train_x, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    train = (train_x - mean) / scale
    test = (test_x - mean) / scale
    train = np.concatenate([np.ones((len(train), 1)), train], axis=1)
    test = np.concatenate([np.ones((len(test), 1)), test], axis=1)
    penalty = float(ridge) * np.eye(train.shape[1])
    penalty[0, 0] = 0.0
    coefficient = np.linalg.solve(train.T @ train + penalty, train.T @ train_y)
    return test @ coefficient


def cross_fitted_incremental_value(
    outcome,
    groups,
    base_features,
    grounded_features,
    folds=5,
    seed=30191,
    ridge=1e-6,
):
    """Compare grouped out-of-fold loss with and without grounded features."""

    y = np.asarray(outcome, dtype=np.float64).reshape(-1)
    labels = np.asarray(groups)
    base = np.asarray(base_features, dtype=np.float64)
    grounded = np.asarray(grounded_features, dtype=np.float64)
    if base.ndim == 1:
        base = base[:, None]
    if grounded.ndim == 1:
        grounded = grounded[:, None]
    if (
        len(y) != len(labels)
        or len(y) != len(base)
        or len(y) != len(grounded)
        or len(y) == 0
    ):
        raise ValueError("cross-fitting arrays have inconsistent lengths")
    if not all(np.all(np.isfinite(value)) for value in [y, base, grounded]):
        raise ValueError("cross-fitting inputs contain nonfinite values")
    fold_id = deterministic_group_folds(labels, folds, seed)
    base_prediction = np.full(len(y), np.nan, dtype=np.float64)
    grounded_prediction = np.full(len(y), np.nan, dtype=np.float64)
    combined = np.concatenate([base, grounded], axis=1)
    for fold in range(int(folds)):
        test = fold_id == fold
        train = ~test
        if not np.any(test) or len(np.unique(labels[train])) < 2:
            raise ValueError("a cross-fitting fold has insufficient groups")
        base_prediction[test] = _ridge_prediction(
            base[train], y[train], base[test], ridge
        )
        grounded_prediction[test] = _ridge_prediction(
            combined[train], y[train], combined[test], ridge
        )
    if not np.all(np.isfinite(base_prediction)) or not np.all(
        np.isfinite(grounded_prediction)
    ):
        raise RuntimeError("cross-fitting did not produce complete predictions")
    base_loss = (y - base_prediction) ** 2
    grounded_loss = (y - grounded_prediction) ** 2
    group_rows = []
    for group in np.unique(labels):
        selected = labels == group
        group_rows.append(
            {
                "group": group.item() if hasattr(group, "item") else group,
                "observations": int(np.sum(selected)),
                "base_mse": float(np.mean(base_loss[selected])),
                "grounded_mse": float(np.mean(grounded_loss[selected])),
                "mse_improvement": float(
                    np.mean(base_loss[selected] - grounded_loss[selected])
                ),
            }
        )
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "fold_id": fold_id,
        "base_prediction": base_prediction,
        "grounded_prediction": grounded_prediction,
        "base_mse": float(np.mean(base_loss)),
        "grounded_mse": float(np.mean(grounded_loss)),
        "relative_mse_improvement": float(
            (np.mean(base_loss) - np.mean(grounded_loss))
            / max(np.mean(base_loss), 1e-20)
        ),
        "base_oof_r_squared": (
            1.0 - float(np.sum(base_loss)) / denominator
            if denominator > 1e-20
            else math.nan
        ),
        "grounded_oof_r_squared": (
            1.0 - float(np.sum(grounded_loss)) / denominator
            if denominator > 1e-20
            else math.nan
        ),
        "group_rows": group_rows,
    }
