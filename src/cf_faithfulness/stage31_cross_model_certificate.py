"""Numerical primitives for Stage 31 cross-model grounded closure.

Stage 31 compares JEPA-WM and DINO-WM on the same physical counterfactual
tasks.  The official simulator planning objective combines mean-squared
visual and proprioceptive latent distances.  This module makes that metric,
the corresponding joint feature chart, and paired model differences
independently testable without loading either model.
"""

from __future__ import annotations

import math

import numpy as np

from .stage17_action_contrast import ranking_metrics
from .stage30_grounded_planning_value import (
    _validated_layout,
    physical_terminal_costs,
)


def planner_metric_features(visual, proprio, alpha=0.1):
    """Return vectors whose squared Euclidean distance is the native L2 cost.

    The public JEPA-WM planner scores

    ``mean((visual-goal_visual)^2) + alpha * mean((prop-goal_prop)^2)``.

    Scaling flattened visual and proprioceptive blocks by their dimensional
    square roots therefore embeds the exact planner metric in one Euclidean
    chart.  These vectors are used only for alignment measurement; no reader
    or decoder is fitted.
    """

    visual_array = np.asarray(visual, dtype=np.float64)
    proprio_array = np.asarray(proprio, dtype=np.float64)
    if visual_array.ndim < 2 or proprio_array.ndim < 2:
        raise ValueError("visual and proprio arrays require an action axis")
    if len(visual_array) != len(proprio_array):
        raise ValueError("visual and proprio arrays have different action counts")
    if not np.all(np.isfinite(visual_array)) or not np.all(
        np.isfinite(proprio_array)
    ):
        raise ValueError("planner metric inputs contain nonfinite values")
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    visual_flat = visual_array.reshape(len(visual_array), -1)
    proprio_flat = proprio_array.reshape(len(proprio_array), -1)
    visual_scaled = visual_flat / math.sqrt(visual_flat.shape[1])
    proprio_scaled = (
        math.sqrt(alpha) * proprio_flat / math.sqrt(proprio_flat.shape[1])
    )
    return np.concatenate([visual_scaled, proprio_scaled], axis=1)


def official_native_terminal_costs(
    predicted_visual,
    encoded_visual,
    predicted_proprio,
    encoded_proprio,
    magnitude_count,
    schedule_count,
    goal_schedule,
    alpha=0.1,
):
    """Compute the public planner's terminal visual-plus-proprio L2 cost."""

    pred_visual = _validated_layout(
        predicted_visual, magnitude_count, schedule_count, "predicted_visual"
    )
    true_visual = _validated_layout(
        encoded_visual, magnitude_count, schedule_count, "encoded_visual"
    )
    pred_proprio = _validated_layout(
        predicted_proprio, magnitude_count, schedule_count, "predicted_proprio"
    )
    true_proprio = _validated_layout(
        encoded_proprio, magnitude_count, schedule_count, "encoded_proprio"
    )
    if pred_visual.shape != true_visual.shape:
        raise ValueError("predicted and target visual tensors differ in shape")
    if pred_proprio.shape != true_proprio.shape:
        raise ValueError("predicted and target proprio tensors differ in shape")
    goal_schedule = int(goal_schedule)
    if not 0 <= goal_schedule < int(schedule_count):
        raise ValueError("goal_schedule is out of range")
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    visual_difference = (
        pred_visual - true_visual[:, goal_schedule : goal_schedule + 1]
    )
    proprio_difference = (
        pred_proprio - true_proprio[:, goal_schedule : goal_schedule + 1]
    )
    visual_axes = tuple(range(2, visual_difference.ndim))
    proprio_axes = tuple(range(2, proprio_difference.ndim))
    return np.mean(visual_difference**2, axis=visual_axes) + alpha * np.mean(
        proprio_difference**2, axis=proprio_axes
    )


def official_terminal_planning_rows(
    predicted_visual,
    encoded_visual,
    predicted_proprio,
    encoded_proprio,
    endpoint_states,
    magnitude_count,
    schedule_count,
    goal_schedules=(0, 5),
    alpha=0.1,
    tie=1e-9,
):
    """Evaluate exhaustive terminal choice under the official latent metric."""

    goals = tuple(int(value) for value in goal_schedules)
    if not goals or len(set(goals)) != len(goals):
        raise ValueError("goal schedules must be unique and nonempty")
    extremes = np.asarray([0, int(schedule_count) - 1], dtype=np.int64)
    rows = []
    for goal_schedule in goals:
        model_costs = official_native_terminal_costs(
            predicted_visual,
            encoded_visual,
            predicted_proprio,
            encoded_proprio,
            magnitude_count,
            schedule_count,
            goal_schedule,
            alpha=alpha,
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
            strict = ranking_metrics(truth[extremes], model[extremes], tie=tie)
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
                    "strict_extreme_top1_correct": float(
                        strict["top1_correct"]
                    ),
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


def paired_model_difference_rows(left_rows, right_rows, feature_names):
    """Join state-magnitude records and compute right-minus-left differences.

    ``outcome`` and every requested feature must be present and finite.  The
    function deliberately requires a one-to-one panel so that paired model
    comparisons cannot silently mix different physical tasks.
    """

    features = tuple(str(value) for value in feature_names)
    keys = ("record_id", "magnitude_index")

    def index(rows, side):
        result = {}
        for row in rows:
            key = tuple(int(row[name]) for name in keys)
            if key in result:
                raise ValueError(f"duplicate {side} paired key {key}")
            result[key] = row
        return result

    left = index(left_rows, "left")
    right = index(right_rows, "right")
    if set(left) != set(right):
        raise ValueError("paired model panels have different state-magnitude keys")
    rows = []
    for key in sorted(left):
        left_row, right_row = left[key], right[key]
        if left_row.get("regime") != right_row.get("regime"):
            raise ValueError(f"paired regime mismatch for {key}")
        values = {
            f"difference_{name}": float(right_row[name]) - float(left_row[name])
            for name in features
        }
        outcome = float(right_row["outcome"]) - float(left_row["outcome"])
        if not np.isfinite(outcome) or not all(np.isfinite(v) for v in values.values()):
            raise ValueError(f"nonfinite paired value for {key}")
        rows.append(
            {
                "record_id": key[0],
                "magnitude_index": key[1],
                "regime": left_row["regime"],
                "left_outcome": float(left_row["outcome"]),
                "right_outcome": float(right_row["outcome"]),
                "outcome": outcome,
                **values,
            }
        )
    return rows
