"""Numerical core for Stage 35 hybrid predictive composition.

The module is intentionally NumPy-only so that its model selection, recursive
rollouts, clustered uncertainty, and decision semantics can be tested locally.
It does not load a simulator or a neural checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _matrix(value: ArrayLike, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite matrix")
    return result


def _labels(value: ArrayLike, rows: int, name: str) -> NDArray[np.str_]:
    result = np.asarray(value).astype(str)
    if result.ndim != 1 or len(result) != int(rows):
        raise ValueError(f"{name} must be a row-aligned label vector")
    return result


def stable_seed(root: int, *parts: object) -> int:
    payload = ":".join([str(int(root)), *map(str, parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def transition_labels(source_modes: ArrayLike, target_modes: ArrayLike) -> NDArray[np.str_]:
    source = np.asarray(source_modes).astype(str)
    target = np.asarray(target_modes).astype(str)
    if source.shape != target.shape:
        raise ValueError("source and target modes must have identical shapes")
    return np.char.add(np.char.add(source, "->"), target)


def sequence_source_states(initial: ArrayLike, targets: ArrayLike) -> FloatArray:
    first = _matrix(initial, "initial")
    path = np.asarray(targets, dtype=np.float64)
    if path.ndim != 3 or path.shape[0] != len(first) or path.shape[2] != first.shape[1]:
        raise ValueError("targets must be a sequence tensor aligned with initial")
    if not np.all(np.isfinite(path)):
        raise ValueError("targets contain nonfinite values")
    source = np.empty_like(path)
    source[:, 0] = first
    source[:, 1:] = path[:, :-1]
    return source


def flatten_sequence_transitions(
    initial: ArrayLike,
    actions: ArrayLike,
    targets: ArrayLike,
    mask: ArrayLike,
    source_modes: ArrayLike,
    target_modes: ArrayLike,
) -> dict[str, np.ndarray]:
    source = sequence_source_states(initial, targets)
    action = np.asarray(actions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    source_label = np.asarray(source_modes).astype(str)
    target_label = np.asarray(target_modes).astype(str)
    expected = target.shape[:2]
    if action.ndim != 3 or action.shape[:2] != expected:
        raise ValueError("actions must align with target sequences")
    if valid.shape != expected or source_label.shape != expected or target_label.shape != expected:
        raise ValueError("mask and mode tensors must align with target sequences")
    return {
        "state": source[valid],
        "action": action[valid],
        "target": target[valid],
        "source_mode": source_label[valid],
        "target_mode": target_label[valid],
        "transition": transition_labels(source_label[valid], target_label[valid]),
    }


def time_shifted_sequence_labels(labels: ArrayLike, mask: ArrayLike) -> NDArray[np.str_]:
    values = np.asarray(labels).astype(str)
    valid = np.asarray(mask, dtype=bool)
    if values.shape != valid.shape or values.ndim != 2:
        raise ValueError("labels and mask must be aligned sequence matrices")
    shifted = values.copy()
    for index in range(len(values)):
        positions = np.flatnonzero(valid[index])
        if len(positions) > 1:
            shifted[index, positions] = np.roll(values[index, positions], 1)
    return shifted


def permuted_sequence_labels(
    labels: ArrayLike,
    mask: ArrayLike,
    groups: ArrayLike,
    *,
    seed: int,
) -> NDArray[np.str_]:
    values = np.asarray(labels).astype(str)
    valid = np.asarray(mask, dtype=bool)
    group = np.asarray(groups)
    if values.shape != valid.shape or values.ndim != 2 or len(group) != len(values):
        raise ValueError("permutation inputs are not sequence aligned")
    result = values.copy()
    rng = np.random.default_rng(int(seed))
    for current in np.unique(group):
        rows = np.flatnonzero(group == current)
        positions = [(int(row), int(step)) for row in rows for step in np.flatnonzero(valid[row])]
        if len(positions) < 2:
            continue
        observed = np.asarray([values[row, step] for row, step in positions])
        permuted = observed[rng.permutation(len(observed))]
        for (row, step), label in zip(positions, permuted, strict=True):
            result[row, step] = label
    return result


def _ridge_fit(features: FloatArray, targets: FloatArray, penalty: float) -> tuple[FloatArray, FloatArray]:
    x = _matrix(features, "features")
    y = _matrix(targets, "targets")
    if len(x) != len(y) or float(penalty) < 0:
        raise ValueError("ridge inputs are not aligned")
    x_mean = np.mean(x, axis=0)
    y_mean = np.mean(y, axis=0)
    centered = x - x_mean
    gram = centered.T @ centered + float(penalty) * np.eye(x.shape[1])
    weight = np.linalg.solve(gram, centered.T @ (y - y_mean))
    intercept = y_mean - x_mean @ weight
    return weight, intercept


def fit_rff_ridge(
    inputs: ArrayLike,
    targets: ArrayLike,
    *,
    width: int,
    penalty: float,
    seed: int,
) -> dict[str, Any]:
    x = _matrix(inputs, "inputs")
    y = _matrix(targets, "targets")
    if len(x) != len(y) or int(width) < 1:
        raise ValueError("RFF ridge inputs are invalid")
    mean = np.mean(x, axis=0)
    scale = np.maximum(np.std(x, axis=0, ddof=1), 1e-8)
    rng = np.random.default_rng(int(seed))
    random_weight = rng.normal(size=(x.shape[1], int(width))) / np.sqrt(x.shape[1])
    random_bias = rng.uniform(-np.pi, np.pi, size=int(width))
    standardized = (x - mean) / scale
    random = np.sqrt(2.0 / int(width)) * np.cos(
        standardized @ random_weight + random_bias
    )
    features = np.column_stack([standardized, random])
    weight, intercept = _ridge_fit(features, y, float(penalty))
    return {
        "mean": mean,
        "scale": scale,
        "random_weight": random_weight,
        "random_bias": random_bias,
        "weight": weight,
        "intercept": intercept,
        "width": int(width),
        "penalty": float(penalty),
    }


def predict_rff_ridge(model: Mapping[str, Any], inputs: ArrayLike) -> FloatArray:
    x = _matrix(inputs, "inputs")
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    random_weight = np.asarray(model["random_weight"], dtype=np.float64)
    random_bias = np.asarray(model["random_bias"], dtype=np.float64)
    if x.shape[1] != len(mean):
        raise ValueError("RFF input width changed")
    standardized = (x - mean) / scale
    random = np.sqrt(2.0 / random_weight.shape[1]) * np.cos(
        standardized @ random_weight + random_bias
    )
    features = np.column_stack([standardized, random])
    return features @ np.asarray(model["weight"]) + np.asarray(model["intercept"])


def fit_rff_classifier(
    inputs: ArrayLike,
    labels: ArrayLike,
    *,
    classes: Sequence[str],
    width: int,
    penalty: float,
    seed: int,
) -> dict[str, Any]:
    x = _matrix(inputs, "inputs")
    observed = _labels(labels, len(x), "labels")
    ordered = [str(value) for value in classes]
    if len(set(ordered)) != len(ordered) or not set(observed).issubset(set(ordered)):
        raise ValueError("classifier classes do not cover observed labels")
    one_hot = np.column_stack([(observed == value).astype(np.float64) for value in ordered])
    model = fit_rff_ridge(
        x, one_hot, width=width, penalty=penalty, seed=seed
    )
    model["classes"] = ordered
    return model


def predict_rff_classifier(model: Mapping[str, Any], inputs: ArrayLike) -> FloatArray:
    scores = predict_rff_ridge(model, inputs)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exponential = np.exp(np.clip(scores, -60.0, 0.0))
    return exponential / np.maximum(np.sum(exponential, axis=1, keepdims=True), 1e-12)


def fit_experts(
    states: ArrayLike,
    actions: ArrayLike,
    targets: ArrayLike,
    labels: ArrayLike,
    *,
    classes: Sequence[str],
    width: int,
    penalty: float,
    seed: int,
) -> dict[str, Any]:
    state = _matrix(states, "states")
    action = _matrix(actions, "actions")
    target = _matrix(targets, "targets")
    observed = _labels(labels, len(state), "labels")
    if not (len(state) == len(action) == len(target)):
        raise ValueError("expert inputs are not aligned")
    inputs = np.column_stack([state, action])
    universal = fit_rff_ridge(
        inputs, target, width=width, penalty=penalty,
        seed=stable_seed(seed, "universal"),
    )
    models: dict[str, Mapping[str, Any]] = {}
    counts: dict[str, int] = {}
    # A 256-dimensional response does not require 258 examples before ridge is
    # mathematically defined: regularization makes the multi-output problem
    # well posed.  The square-root rule still prevents tiny transition classes
    # from producing effectively single-example experts.
    minimum_rows = max(12, int(2 * np.sqrt(target.shape[1])))
    for label in [str(value) for value in classes]:
        selected = observed == label
        counts[label] = int(np.sum(selected))
        if counts[label] < minimum_rows:
            models[label] = universal
        else:
            models[label] = fit_rff_ridge(
                inputs[selected], target[selected], width=width, penalty=penalty,
                seed=stable_seed(seed, "expert", label),
            )
    return {
        "classes": [str(value) for value in classes],
        "models": models,
        "universal": universal,
        "counts": counts,
    }


def predict_experts(
    model: Mapping[str, Any],
    states: ArrayLike,
    actions: ArrayLike,
    labels: ArrayLike,
) -> FloatArray:
    state = _matrix(states, "states")
    action = _matrix(actions, "actions")
    observed = _labels(labels, len(state), "labels")
    inputs = np.column_stack([state, action])
    output_width = np.asarray(model["universal"]["intercept"]).size
    prediction = np.empty((len(state), output_width), dtype=np.float64)
    assigned = np.zeros(len(state), dtype=bool)
    for label in model["classes"]:
        selected = observed == str(label)
        if np.any(selected):
            prediction[selected] = predict_rff_ridge(model["models"][str(label)], inputs[selected])
            assigned[selected] = True
    if np.any(~assigned):
        prediction[~assigned] = predict_rff_ridge(model["universal"], inputs[~assigned])
    return prediction


def predict_expert_mixture(
    model: Mapping[str, Any],
    states: ArrayLike,
    actions: ArrayLike,
    probabilities: ArrayLike,
) -> FloatArray:
    state = _matrix(states, "states")
    action = _matrix(actions, "actions")
    probability = _matrix(probabilities, "probabilities")
    if probability.shape != (len(state), len(model["classes"])):
        raise ValueError("mixture probabilities have the wrong shape")
    inputs = np.column_stack([state, action])
    components = np.stack([
        predict_rff_ridge(model["models"][str(label)], inputs)
        for label in model["classes"]
    ], axis=1)
    return np.sum(components * probability[:, :, None], axis=1)


def fit_hybrid_family(
    states: ArrayLike,
    actions: ArrayLike,
    targets: ArrayLike,
    source_modes: ArrayLike,
    target_modes: ArrayLike,
    *,
    width: int,
    penalty: float,
    seed: int,
    transition_override: ArrayLike | None = None,
) -> dict[str, Any]:
    state = _matrix(states, "states")
    action = _matrix(actions, "actions")
    target = _matrix(targets, "targets")
    source = _labels(source_modes, len(state), "source_modes")
    target_mode = _labels(target_modes, len(state), "target_modes")
    transitions = (
        transition_labels(source, target_mode)
        if transition_override is None
        else _labels(transition_override, len(state), "transition_override")
    )
    source_classes = sorted(set(source.tolist()))
    transition_classes = sorted(set(transitions.tolist()))
    universal_labels = np.repeat("__global__", len(state))
    inputs = np.column_stack([state, action])
    return {
        "global": fit_experts(
            state, action, target, universal_labels, classes=["__global__"],
            width=width, penalty=penalty, seed=stable_seed(seed, "global"),
        ),
        "source": fit_experts(
            state, action, target, source, classes=source_classes,
            width=width, penalty=penalty, seed=stable_seed(seed, "source"),
        ),
        "transition": fit_experts(
            state, action, target, transitions, classes=transition_classes,
            width=width, penalty=penalty, seed=stable_seed(seed, "transition"),
        ),
        "gate": fit_rff_classifier(
            inputs, transitions, classes=transition_classes,
            width=width, penalty=penalty, seed=stable_seed(seed, "gate"),
        ),
        "width": int(width),
        "penalty": float(penalty),
        "source_classes": source_classes,
        "transition_classes": transition_classes,
    }


def recursive_rollout(
    family: Mapping[str, Any],
    initial: ArrayLike,
    actions: ArrayLike,
    mask: ArrayLike,
    *,
    strategy: str,
    source_modes: ArrayLike | None = None,
    target_modes: ArrayLike | None = None,
) -> FloatArray:
    current = _matrix(initial, "initial")
    action = np.asarray(actions, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if action.ndim != 3 or action.shape[:2] != valid.shape or len(action) != len(current):
        raise ValueError("rollout arrays are not aligned")
    source = None if source_modes is None else np.asarray(source_modes).astype(str)
    target = None if target_modes is None else np.asarray(target_modes).astype(str)
    if source is not None and source.shape != valid.shape:
        raise ValueError("source modes do not align with rollout")
    if target is not None and target.shape != valid.shape:
        raise ValueError("target modes do not align with rollout")
    output = np.zeros((len(current), action.shape[1], current.shape[1]), dtype=np.float64)
    frozen_source = None if source is None else source[:, 0]
    for step in range(action.shape[1]):
        active = valid[:, step]
        if not np.any(active):
            continue
        state = current[active]
        control = action[active, step]
        if strategy == "global":
            labels = np.repeat("__global__", len(state))
            updated = predict_experts(family["global"], state, control, labels)
        elif strategy == "fixed_source":
            if frozen_source is None:
                raise ValueError("fixed_source requires source modes")
            updated = predict_experts(family["source"], state, control, frozen_source[active])
        elif strategy == "oracle_source":
            if source is None:
                raise ValueError("oracle_source requires source modes")
            updated = predict_experts(family["source"], state, control, source[active, step])
        elif strategy == "oracle_transition":
            if source is None or target is None:
                raise ValueError("oracle_transition requires source and target modes")
            labels = transition_labels(source[active, step], target[active, step])
            updated = predict_experts(family["transition"], state, control, labels)
        elif strategy == "predicted_guard":
            inputs = np.column_stack([state, control])
            probability = predict_rff_classifier(family["gate"], inputs)
            updated = predict_expert_mixture(
                family["transition"], state, control, probability
            )
        else:
            raise ValueError(f"unknown rollout strategy {strategy!r}")
        current = current.copy()
        current[active] = updated
        output[active, step] = updated
    return output


def fit_family_from_sequences(
    initial: ArrayLike,
    actions: ArrayLike,
    targets: ArrayLike,
    mask: ArrayLike,
    source_modes: ArrayLike,
    target_modes: ArrayLike,
    *,
    width: int,
    penalty: float,
    seed: int,
    transition_override: ArrayLike | None = None,
) -> dict[str, Any]:
    rows = flatten_sequence_transitions(
        initial, actions, targets, mask, source_modes, target_modes
    )
    override_rows = None
    if transition_override is not None:
        override = np.asarray(transition_override).astype(str)
        valid = np.asarray(mask, dtype=bool)
        if override.shape != valid.shape:
            raise ValueError("transition override does not align with sequences")
        override_rows = override[valid]
    return fit_hybrid_family(
        rows["state"], rows["action"], rows["target"],
        rows["source_mode"], rows["target_mode"], width=width,
        penalty=penalty, seed=seed, transition_override=override_rows,
    )


def grouped_sequence_folds(groups: ArrayLike, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    group = np.asarray(groups)
    unique = np.unique(group)
    if not 2 <= int(folds) <= len(unique):
        raise ValueError("fold count must lie between two and the group count")
    rng = np.random.default_rng(int(seed))
    shuffled = unique[rng.permutation(len(unique))]
    partitions = np.array_split(shuffled, int(folds))
    result = []
    for held in partitions:
        test = np.isin(group, held)
        result.append((np.flatnonzero(~test), np.flatnonzero(test)))
    return result


def final_values(path: ArrayLike, mask: ArrayLike) -> FloatArray:
    values = np.asarray(path, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if values.ndim != 3 or values.shape[:2] != valid.shape:
        raise ValueError("path and mask are not aligned")
    lengths = np.sum(valid, axis=1)
    if np.any(lengths < 1):
        raise ValueError("every sequence must contain a valid transition")
    return values[np.arange(len(values)), lengths - 1]


def scaled_sequence_mse(
    prediction: ArrayLike,
    target: ArrayLike,
    mask: ArrayLike,
    scale: ArrayLike,
    *,
    final_only: bool = True,
) -> FloatArray:
    predicted = np.asarray(prediction, dtype=np.float64)
    observed = np.asarray(target, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    width_scale = np.maximum(np.asarray(scale, dtype=np.float64), 1e-8)
    if predicted.shape != observed.shape or predicted.ndim != 3:
        raise ValueError("prediction and target paths are not aligned")
    if predicted.shape[:2] != valid.shape or predicted.shape[2] != len(width_scale):
        raise ValueError("mask or scale does not align with path")
    if final_only:
        delta = (final_values(predicted, valid) - final_values(observed, valid)) / width_scale
        return np.mean(delta**2, axis=1)
    squared = np.mean(((predicted - observed) / width_scale) ** 2, axis=2)
    return np.sum(squared * valid, axis=1) / np.sum(valid, axis=1)


def relative_gain(primary_error: ArrayLike, comparator_error: ArrayLike) -> FloatArray:
    primary = np.asarray(primary_error, dtype=np.float64)
    comparator = np.asarray(comparator_error, dtype=np.float64)
    if primary.shape != comparator.shape or primary.ndim != 1:
        raise ValueError("relative gain inputs must be aligned vectors")
    return (comparator - primary) / np.maximum(comparator, 1e-12)


def clustered_mean_interval(
    values: ArrayLike,
    groups: ArrayLike,
    *,
    draws: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    sample = np.asarray(values, dtype=np.float64)
    group = np.asarray(groups)
    if sample.ndim != 1 or len(sample) != len(group) or not np.all(np.isfinite(sample)):
        raise ValueError("clustered interval inputs are invalid")
    unique = np.unique(group)
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(int(draws), dtype=np.float64)
    blocks = {value: sample[group == value] for value in unique}
    for draw in range(int(draws)):
        selected = unique[rng.integers(0, len(unique), size=len(unique))]
        estimates[draw] = np.mean(np.concatenate([blocks[value] for value in selected]))
    return tuple(np.quantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0]).tolist())


def clustered_ratio_interval(
    numerator: ArrayLike,
    denominator: ArrayLike,
    groups: ArrayLike,
    *,
    draws: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    top = np.asarray(numerator, dtype=np.float64)
    bottom = np.asarray(denominator, dtype=np.float64)
    group = np.asarray(groups)
    if top.shape != bottom.shape or top.ndim != 1 or len(top) != len(group):
        raise ValueError("clustered ratio inputs are invalid")
    unique = np.unique(group)
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(int(draws), dtype=np.float64)
    blocks = {value: np.flatnonzero(group == value) for value in unique}
    for draw in range(int(draws)):
        selected = unique[rng.integers(0, len(unique), size=len(unique))]
        indices = np.concatenate([blocks[value] for value in selected])
        estimates[draw] = np.mean(top[indices]) / max(np.mean(bottom[indices]), 1e-12)
    return tuple(np.quantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0]).tolist())


def select_guard_hyperparameters(
    initial: ArrayLike,
    actions: ArrayLike,
    targets: ArrayLike,
    mask: ArrayLike,
    source_modes: ArrayLike,
    target_modes: ArrayLike,
    groups: ArrayLike,
    *,
    widths: Sequence[int],
    penalties: Sequence[float],
    folds: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    initial_array = _matrix(initial, "initial")
    action_array = np.asarray(actions, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    source = np.asarray(source_modes).astype(str)
    target_mode = np.asarray(target_modes).astype(str)
    group = np.asarray(groups)
    scale = np.maximum(np.std(target_array[valid], axis=0, ddof=1), 1e-8)
    splits = grouped_sequence_folds(group, folds, seed)
    rows: list[dict[str, Any]] = []
    for width in widths:
        for penalty in penalties:
            fold_errors = []
            for fold_index, (train, test) in enumerate(splits):
                family = fit_family_from_sequences(
                    initial_array[train], action_array[train], target_array[train],
                    valid[train], source[train], target_mode[train], width=int(width),
                    penalty=float(penalty), seed=stable_seed(seed, width, penalty, fold_index),
                )
                prediction = recursive_rollout(
                    family, initial_array[test], action_array[test], valid[test],
                    strategy="predicted_guard",
                )
                fold_errors.extend(scaled_sequence_mse(
                    prediction, target_array[test], valid[test], scale,
                ).tolist())
            rows.append({
                "width": int(width),
                "penalty": float(penalty),
                "oof_recursive_mse": float(np.mean(fold_errors)),
                "folds": int(folds),
                "rows": len(fold_errors),
            })
    best = min(rows, key=lambda row: (row["oof_recursive_mse"], row["width"], -row["penalty"]))
    ceiling = float(best["oof_recursive_mse"]) * 1.02
    eligible = [row for row in rows if row["oof_recursive_mse"] <= ceiling]
    selected = min(eligible, key=lambda row: (row["width"], -row["penalty"], row["oof_recursive_mse"]))
    return {
        **selected,
        "best_oof_recursive_mse": float(best["oof_recursive_mse"]),
        "selection_ceiling": ceiling,
    }, rows


def fit_support_reference(states: ArrayLike, quantile: float = 0.99) -> dict[str, Any]:
    values = _matrix(states, "states")
    mean = np.mean(values, axis=0)
    scale = np.maximum(np.std(values, axis=0, ddof=1), 1e-8)
    radius = np.sqrt(np.mean(((values - mean) / scale) ** 2, axis=1))
    return {"mean": mean, "scale": scale, "radius": float(np.quantile(radius, quantile))}


def support_exceedance_rate(reference: Mapping[str, Any], states: ArrayLike) -> float:
    values = _matrix(states, "states")
    radius = np.sqrt(np.mean(
        ((values - np.asarray(reference["mean"])) / np.asarray(reference["scale"])) ** 2,
        axis=1,
    ))
    return float(np.mean(radius > float(reference["radius"])))


@dataclass(frozen=True)
class Stage35Gates:
    source_and_split_binding: bool
    simulator_positive_control: bool
    native_physical_fidelity: bool
    guard_transfer: bool
    guard_specificity: bool
    recursive_closure: bool
    family_consistency: bool


def derive_stage35_decision(gates: Stage35Gates, *, run_mode: str) -> dict[str, Any]:
    checks = {
        "source_and_split_binding": bool(gates.source_and_split_binding),
        "simulator_positive_control": bool(gates.simulator_positive_control),
        "native_physical_fidelity": bool(gates.native_physical_fidelity),
        "guard_transfer": bool(gates.guard_transfer),
        "guard_specificity": bool(gates.guard_specificity),
        "recursive_closure": bool(gates.recursive_closure),
        "family_consistency": bool(gates.family_consistency),
    }
    if run_mode == "smoke":
        status = "smoke_only"
    elif not checks["source_and_split_binding"]:
        status = "inconclusive_source_or_split_failure"
    elif not checks["simulator_positive_control"]:
        status = "simulator_operator_class_invalid"
    elif not checks["native_physical_fidelity"]:
        status = "native_jepa_not_physically_predictive"
    elif not checks["guard_transfer"]:
        status = "guard_reset_structure_did_not_transfer"
    elif not checks["guard_specificity"]:
        status = "guard_signal_not_specific"
    elif not checks["recursive_closure"]:
        status = "distributed_carrier_not_recursively_closed"
    elif not checks["family_consistency"]:
        status = "distributed_closure_not_family_consistent"
    else:
        status = "bounded_distributed_hybrid_closure_supported"
    passed = bool(run_mode == "pilot" and all(checks.values()))
    return {
        "status": status,
        "passed": passed,
        "checks": checks,
        "first_failed_gate": next((name for name, value in checks.items() if not value), None),
        "causal_evidence": False,
        "shared_cross_model_mechanism_claimed": False,
        "low_dimensional_state_claimed": False,
        "confirmation_eligible": bool(run_mode == "pilot" and checks["source_and_split_binding"]),
    }
