"""Numerical core for the Stage 34.1 action-specificity repair.

Stage 34 compared a decoded, action-conditioned response signature with a
state-to-entire-response-atlas regression.  Output-column position identified
the action word in that comparator, so it was not action blind.  This module
implements the repaired estimand: one row is one action prefix and the
state-only comparator receives exactly the same features for different words
at the same state, word length, and prefix step.

The functions are NumPy-only so the design can be unit-tested without PushT,
model checkpoints, a GPU, or the Stage 34 artifact directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _finite_matrix(values: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not len(array):
        raise ValueError(f"{name} must be a nonempty matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains nonfinite values")
    return array


def _word_lookup(word_names: Sequence[str]) -> dict[str, int]:
    names = [str(value) for value in word_names]
    if len(names) != len(set(names)):
        raise ValueError("word_names must be unique")
    return {name: index for index, name in enumerate(names)}


def action_response_path_rows(
    grounded_paths: ArrayLike,
    word_names: Sequence[str],
    word_lengths: ArrayLike,
    response_words: Sequence[str],
    zero_word_by_length: Mapping[int, str],
) -> tuple[FloatArray, dict[str, NDArray[Any]]]:
    """Return no-op-corrected response rows and their action-prefix metadata.

    Rows are ordered by ``response_words`` and then prefix step.  Unlike a
    concatenated response atlas, word identity never becomes a target-column
    index.  The returned target width is only the grounded-observable width.
    """

    paths = np.asarray(grounded_paths, dtype=np.float64)
    if paths.ndim != 3 or not paths.shape[0] or not np.all(np.isfinite(paths)):
        raise ValueError("grounded_paths must be a finite nonempty 3D array")
    lengths = np.asarray(word_lengths, dtype=np.int64).reshape(-1)
    if len(lengths) != len(paths) or len(word_names) != len(paths):
        raise ValueError("word metadata must match grounded_paths")
    lookup = _word_lookup(word_names)
    values: list[FloatArray] = []
    words: list[str] = []
    row_lengths: list[int] = []
    steps: list[int] = []
    for word in map(str, response_words):
        if word not in lookup:
            raise KeyError(f"missing response word {word!r}")
        word_index = lookup[word]
        length = int(lengths[word_index])
        if not 1 <= length <= paths.shape[1]:
            raise ValueError(f"invalid length for response word {word!r}")
        zero = str(zero_word_by_length.get(length, ""))
        if zero not in lookup:
            raise KeyError(f"missing length-{length} zero word")
        zero_index = lookup[zero]
        if int(lengths[zero_index]) != length:
            raise ValueError("zero word length does not match response word")
        delta = paths[word_index, :length] - paths[zero_index, :length]
        values.extend(delta)
        words.extend([word] * length)
        row_lengths.extend([length] * length)
        steps.extend(range(1, length + 1))
    if not values:
        raise ValueError("response_words must be nonempty")
    return np.asarray(values, dtype=np.float64), {
        "word": np.asarray(words),
        "length": np.asarray(row_lengths, dtype=np.int64),
        "step": np.asarray(steps, dtype=np.int64),
    }


def action_blind_context_features(
    states: ArrayLike,
    lengths: ArrayLike,
    steps: ArrayLike,
    modes: Sequence[str],
    mode_levels: Sequence[str],
    *,
    maximum_length: int,
) -> FloatArray:
    """Build context features that provably contain no action-word identity."""

    state = _finite_matrix(states, "states")
    length = np.asarray(lengths, dtype=np.int64).reshape(-1)
    step = np.asarray(steps, dtype=np.int64).reshape(-1)
    labels = np.asarray([str(value) for value in modes])
    if not (len(state) == len(length) == len(step) == len(labels)):
        raise ValueError("state, length, step, and mode rows must align")
    maximum = int(maximum_length)
    if maximum < 1 or np.any(length < 1) or np.any(length > maximum):
        raise ValueError("word length lies outside the registered maximum")
    if np.any(step < 1) or np.any(step > length):
        raise ValueError("prefix step lies outside its word")
    levels = [str(value) for value in mode_levels]
    unknown = sorted(set(labels.tolist()) - set(levels))
    if unknown:
        raise ValueError(f"unknown mode labels: {unknown}")
    one_hot = np.column_stack([labels == level for level in levels]).astype(np.float64)
    normalized_length = length.astype(np.float64) / maximum
    normalized_step = step.astype(np.float64) / maximum
    within_word_fraction = step.astype(np.float64) / length
    return np.column_stack([
        state,
        normalized_length,
        normalized_step,
        within_word_fraction,
        one_hot,
    ])


def action_prefix_features(
    actions: ArrayLike,
    action_mask: ArrayLike,
    word_names: Sequence[str],
    response_words: Sequence[str],
    word_lengths: ArrayLike,
    *,
    frameskip: int,
) -> FloatArray:
    """Return registered action features aligned with response-prefix rows.

    Features contain cumulative impulse, cumulative energy, signed cumulative
    area, the current macro action, and the prefix displacement path length.
    They are used only for the physical action-necessity positive control.
    """

    action_array = np.asarray(actions, dtype=np.float64)
    mask = np.asarray(action_mask, dtype=bool)
    lengths = np.asarray(word_lengths, dtype=np.int64).reshape(-1)
    if action_array.ndim != 3 or action_array.shape[2] != 2:
        raise ValueError("actions must have shape (word, frame, 2)")
    if mask.shape != action_array.shape[:2] or len(lengths) != len(action_array):
        raise ValueError("action metadata must align")
    lookup = _word_lookup(word_names)
    skip = int(frameskip)
    if skip < 1:
        raise ValueError("frameskip must be positive")
    rows: list[FloatArray] = []
    for word in map(str, response_words):
        if word not in lookup:
            raise KeyError(f"missing action word {word!r}")
        index = lookup[word]
        length = int(lengths[index])
        for step in range(1, length + 1):
            stop = step * skip
            valid = action_array[index, :stop][mask[index, :stop]]
            if len(valid) != stop:
                raise ValueError("action mask does not cover a complete prefix")
            impulse = np.sum(valid, axis=0)
            energy = float(np.sum(valid**2))
            area = float(sum(
                valid[left, 0] * valid[right, 1]
                - valid[left, 1] * valid[right, 0]
                for left in range(len(valid))
                for right in range(left + 1, len(valid))
            ))
            current = np.mean(valid[-skip:], axis=0)
            path_length = float(np.sum(np.linalg.norm(valid, axis=1)))
            rows.append(np.asarray([
                impulse[0], impulse[1], energy, area,
                current[0], current[1], path_length,
            ], dtype=np.float64))
    return np.asarray(rows, dtype=np.float64)


def deranged_word_rows(
    rows: ArrayLike,
    metadata: Mapping[str, ArrayLike],
    *,
    seed: int,
) -> FloatArray:
    """Replace each word path with a different same-length word path."""

    values = _finite_matrix(rows, "rows")
    words = np.asarray(metadata["word"])
    lengths = np.asarray(metadata["length"], dtype=np.int64)
    steps = np.asarray(metadata["step"], dtype=np.int64)
    if not (len(values) == len(words) == len(lengths) == len(steps)):
        raise ValueError("row metadata must align")
    result = np.empty_like(values)
    rng = np.random.default_rng(int(seed))
    for length in sorted(set(lengths.tolist())):
        candidates = sorted(set(words[lengths == length].tolist()))
        if len(candidates) < 2:
            raise ValueError("each length needs at least two response words")
        offset = int(rng.integers(1, len(candidates)))
        donors = candidates[offset:] + candidates[:offset]
        for target, donor in zip(candidates, donors):
            target_indices = np.flatnonzero(words == target)
            donor_indices = np.flatnonzero(words == donor)
            if not np.array_equal(steps[target_indices], steps[donor_indices]):
                raise ValueError("same-length paths have inconsistent steps")
            result[target_indices] = values[donor_indices]
    return result


def _ordered_unique(values: NDArray[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[Any] = set()
    for value in values.tolist():
        key = value.item() if isinstance(value, np.generic) else value
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result


def grouped_folds(groups: ArrayLike, folds: int, seed: int) -> tuple[NDArray[np.bool_], ...]:
    labels = np.asarray(groups).reshape(-1)
    unique = np.asarray(_ordered_unique(labels), dtype=object)
    count = min(int(folds), len(unique))
    if count < 2:
        raise ValueError("at least two independent groups are required")
    rng = np.random.default_rng(int(seed))
    parts = np.array_split(unique[rng.permutation(len(unique))], count)
    return tuple(np.isin(labels, part) for part in parts)


def _ridge_fit(x: FloatArray, y: FloatArray, penalty: float) -> tuple[FloatArray, FloatArray]:
    mean_x = np.mean(x, axis=0)
    mean_y = np.mean(y, axis=0)
    centered_x = x - mean_x
    centered_y = y - mean_y
    gram = centered_x.T @ centered_x
    weight = np.linalg.solve(
        gram + float(penalty) * np.eye(gram.shape[0], dtype=np.float64),
        centered_x.T @ centered_y,
    )
    return weight, mean_y - mean_x @ weight


def _rff_parameters(features: FloatArray, width: int, seed: int) -> dict[str, FloatArray]:
    selected_width = int(width)
    if selected_width < 1:
        raise ValueError("random-feature width must be positive")
    mean = np.mean(features, axis=0)
    scale = np.maximum(np.std(features, axis=0, ddof=1), 1e-8)
    rng = np.random.default_rng(int(seed))
    weight = rng.normal(size=(features.shape[1], selected_width)) / np.sqrt(features.shape[1])
    bias = rng.uniform(-np.pi, np.pi, size=selected_width)
    return {"mean": mean, "scale": scale, "weight": weight, "bias": bias}


def _rff_apply(features: ArrayLike, parameters: Mapping[str, ArrayLike]) -> FloatArray:
    x = _finite_matrix(features, "features")
    mean = np.asarray(parameters["mean"], dtype=np.float64)
    scale = np.asarray(parameters["scale"], dtype=np.float64)
    weight = np.asarray(parameters["weight"], dtype=np.float64)
    bias = np.asarray(parameters["bias"], dtype=np.float64)
    standardized = (x - mean) / scale
    return np.sqrt(2.0 / weight.shape[1]) * np.cos(standardized @ weight + bias)


def fit_grouped_rff_ridge(
    features: ArrayLike,
    targets: ArrayLike,
    groups: ArrayLike,
    *,
    width: int = 256,
    penalties: Iterable[float] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
    folds: int = 4,
    seed: int = 0,
) -> dict[str, Any]:
    """Select ridge strength by trajectory-grouped OOF loss and refit."""

    x = _finite_matrix(features, "features")
    y = _finite_matrix(targets, "targets")
    labels = np.asarray(groups).reshape(-1)
    if len(x) != len(y) or len(labels) != len(x):
        raise ValueError("features, targets, and groups must align")
    parameters = _rff_parameters(x, width, seed)
    mapped = _rff_apply(x, parameters)
    candidates = [float(value) for value in penalties]
    if not candidates or any(value < 0 or not np.isfinite(value) for value in candidates):
        raise ValueError("penalties must be finite and nonnegative")
    masks = grouped_folds(labels, folds, seed + 1)
    losses: list[float] = []
    for penalty in candidates:
        prediction = np.empty_like(y)
        for held_out in masks:
            weight, intercept = _ridge_fit(mapped[~held_out], y[~held_out], penalty)
            prediction[held_out] = mapped[held_out] @ weight + intercept
        losses.append(float(np.mean((prediction - y) ** 2)))
    selected = int(np.argmin(losses))
    weight, intercept = _ridge_fit(mapped, y, candidates[selected])
    return {
        "parameters": parameters,
        "weight": weight,
        "intercept": intercept,
        "penalty": candidates[selected],
        "all_oof_mse": losses,
        "oof_mse": losses[selected],
    }


def predict_grouped_rff_ridge(model: Mapping[str, Any], features: ArrayLike) -> FloatArray:
    mapped = _rff_apply(features, model["parameters"])
    return mapped @ np.asarray(model["weight"]) + np.asarray(model["intercept"])


def grouped_record_mse(
    predictions: ArrayLike,
    targets: ArrayLike,
    record_ids: ArrayLike,
) -> tuple[FloatArray, IntArray]:
    prediction = _finite_matrix(predictions, "predictions")
    target = _finite_matrix(targets, "targets")
    records = np.asarray(record_ids, dtype=np.int64).reshape(-1)
    if prediction.shape != target.shape or len(records) != len(prediction):
        raise ValueError("prediction, target, and record rows must align")
    row_error = np.mean((prediction - target) ** 2, axis=1)
    unique = np.asarray(_ordered_unique(records), dtype=np.int64)
    return np.asarray([np.mean(row_error[records == value]) for value in unique]), unique


def relative_advantage(primary_error: ArrayLike, control_error: ArrayLike) -> FloatArray:
    primary = np.asarray(primary_error, dtype=np.float64).reshape(-1)
    control = np.asarray(control_error, dtype=np.float64).reshape(-1)
    if primary.shape != control.shape or not len(primary):
        raise ValueError("primary and control errors must align")
    if not np.all(np.isfinite(primary)) or not np.all(np.isfinite(control)):
        raise ValueError("errors must be finite")
    return (control - primary) / np.maximum(control, 1e-12)


def clustered_bootstrap_interval(
    values: ArrayLike,
    groups: ArrayLike,
    *,
    draws: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    observations = np.asarray(values, dtype=np.float64).reshape(-1)
    labels = np.asarray(groups).reshape(-1)
    if len(observations) != len(labels) or not len(observations):
        raise ValueError("values and groups must be nonempty and aligned")
    unique = np.asarray(_ordered_unique(labels), dtype=object)
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(int(draws), dtype=np.float64)
    by_group = [observations[labels == group] for group in unique]
    for index in range(int(draws)):
        sampled = rng.integers(0, len(unique), size=len(unique))
        estimates[index] = float(np.mean(np.concatenate([by_group[item] for item in sampled])))
    return (
        float(np.quantile(estimates, alpha / 2.0)),
        float(np.quantile(estimates, 1.0 - alpha / 2.0)),
    )


@dataclass(frozen=True)
class Stage341Gates:
    upstream_binding: bool
    leakage_invariant: bool
    physical_action_necessity: bool
    jepa_action_specificity: bool
    dino_action_specificity: bool


def derive_stage341_decision(
    gates: Stage341Gates | Mapping[str, bool],
    *,
    run_mode: str = "pilot",
) -> dict[str, Any]:
    """Return the bounded diagnostic decision; never claim full abstraction."""

    values = gates.__dict__ if isinstance(gates, Stage341Gates) else dict(gates)
    order = [
        "upstream_binding",
        "leakage_invariant",
        "physical_action_necessity",
        "jepa_action_specificity",
        "dino_action_specificity",
    ]
    missing = [name for name in order if name not in values]
    if missing:
        raise ValueError(f"missing Stage 34.1 gates: {missing}")
    checks = {name: bool(values[name]) for name in order}
    first_failure = next((name for name in order if not checks[name]), None)
    passed = bool(run_mode == "pilot" and first_failure is None)
    if run_mode == "smoke":
        status = "smoke_only"
    elif not checks["upstream_binding"]:
        status = "inconclusive_upstream_binding_failure"
    elif passed:
        status = "action_specificity_repaired_continue_stage34"
    else:
        status = "action_specificity_not_established"
    return {
        "status": status,
        "passed": passed,
        "first_failed_gate": first_failure,
        "checks": checks,
        "failed_checks": [name for name in order if not checks[name]],
        "run_mode": str(run_mode),
        "claim_scope": "diagnostic_repair_only_not_full_causal_abstraction",
    }
