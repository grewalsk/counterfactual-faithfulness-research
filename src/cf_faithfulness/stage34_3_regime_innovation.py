"""Numerical core for the Stage 34.3 JEPA regime/innovation diagnostic.

Stage 34.2 falsified sufficiency of the registered rank-five response chart.
This module tests a deliberately bounded repair family without touching the
native checkpoint: rank four or five, universal or physical-mode dynamics,
and at most three supervised carrier-innovation coordinates.

Candidate choice is based on trajectory-grouped out-of-fold error.  The final
candidate is refit on calibration data and evaluated separately by the Colab
notebook.  No routine here can support a causal or confirmatory conclusion.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .stage34_predictive_fiber_abstraction import grouped_folds


FloatArray = NDArray[np.float64]


def _matrix(values: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not len(array):
        raise ValueError(f"{name} must be a nonempty matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains nonfinite values")
    return array


def _labels(values: ArrayLike, size: int, name: str) -> NDArray[Any]:
    array = np.asarray(values).reshape(-1)
    if len(array) != int(size):
        raise ValueError(f"{name} must align with the row count")
    return array


def stable_seed(seed: int, *parts: object) -> int:
    payload = ":".join([str(int(seed)), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") % (2**32)


def _ridge_fit(inputs: FloatArray, targets: FloatArray, penalty: float) -> tuple[FloatArray, FloatArray]:
    x = _matrix(inputs, "inputs")
    y = _matrix(targets, "targets")
    if len(x) != len(y):
        raise ValueError("inputs and targets must align")
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


def rff_parameters(inputs: ArrayLike, *, width: int, seed: int) -> dict[str, FloatArray]:
    x = _matrix(inputs, "inputs")
    selected_width = int(width)
    if selected_width < 1:
        raise ValueError("width must be positive")
    mean = np.mean(x, axis=0)
    scale = np.maximum(np.std(x, axis=0, ddof=1), 1e-8)
    generator = np.random.default_rng(int(seed))
    weight = generator.normal(size=(x.shape[1], selected_width)) / np.sqrt(x.shape[1])
    bias = generator.uniform(-np.pi, np.pi, size=selected_width)
    return {"mean": mean, "scale": scale, "weight": weight, "bias": bias}


def rff_apply(inputs: ArrayLike, parameters: Mapping[str, ArrayLike]) -> FloatArray:
    x = _matrix(inputs, "inputs")
    mean = np.asarray(parameters["mean"], dtype=np.float64)
    scale = np.asarray(parameters["scale"], dtype=np.float64)
    weight = np.asarray(parameters["weight"], dtype=np.float64)
    bias = np.asarray(parameters["bias"], dtype=np.float64)
    if x.shape[1] != len(mean) or len(mean) != len(scale) or weight.shape[0] != len(mean):
        raise ValueError("random-feature dimensions do not agree")
    return np.sqrt(2.0 / weight.shape[1]) * np.cos(
        ((x - mean) / scale) @ weight + bias
    )


def fit_rff_model(
    inputs: ArrayLike,
    targets: ArrayLike,
    *,
    width: int,
    penalty: float,
    seed: int,
) -> dict[str, Any]:
    x = _matrix(inputs, "inputs")
    y = _matrix(targets, "targets")
    parameters = rff_parameters(x, width=width, seed=seed)
    weight, intercept = _ridge_fit(rff_apply(x, parameters), y, float(penalty))
    return {
        "parameters": parameters,
        "weight": weight,
        "intercept": intercept,
        "penalty": float(penalty),
        "width": int(width),
    }


def predict_rff_model(model: Mapping[str, Any], inputs: ArrayLike) -> FloatArray:
    features = rff_apply(inputs, model["parameters"])
    return features @ np.asarray(model["weight"]) + np.asarray(model["intercept"])


def fit_regime_dynamics(
    inputs: ArrayLike,
    targets: ArrayLike,
    modes: ArrayLike,
    *,
    regime_specific: bool,
    mode_labels: Sequence[str],
    width: int,
    penalty: float,
    seed: int,
) -> dict[str, Any]:
    """Fit either one RFF transition map or one capacity-matched map per mode."""

    x = _matrix(inputs, "inputs")
    y = _matrix(targets, "targets")
    labels = _labels(modes, len(x), "modes").astype(str)
    if len(y) != len(x):
        raise ValueError("inputs and targets must align")
    if not regime_specific:
        return {
            "regime_specific": False,
            "models": {
                "__universal__": fit_rff_model(
                    x, y, width=width, penalty=penalty,
                    seed=stable_seed(seed, "universal"),
                )
            },
            "mode_labels": tuple(map(str, mode_labels)),
        }
    models = {}
    for mode in map(str, mode_labels):
        mask = labels == mode
        if not np.any(mask):
            raise ValueError(f"training data omit mode {mode!r}")
        models[mode] = fit_rff_model(
            x[mask], y[mask], width=width, penalty=penalty,
            seed=stable_seed(seed, "mode", mode),
        )
    return {
        "regime_specific": True,
        "models": models,
        "mode_labels": tuple(map(str, mode_labels)),
    }


def predict_regime_dynamics(
    model: Mapping[str, Any],
    inputs: ArrayLike,
    modes: ArrayLike,
) -> FloatArray:
    x = _matrix(inputs, "inputs")
    labels = _labels(modes, len(x), "modes").astype(str)
    models = model["models"]
    if not bool(model["regime_specific"]):
        return predict_rff_model(models["__universal__"], x)
    output_width = np.asarray(next(iter(models.values()))["intercept"]).size
    prediction = np.empty((len(x), output_width), dtype=np.float64)
    assigned = np.zeros(len(x), dtype=bool)
    for mode in model["mode_labels"]:
        mask = labels == str(mode)
        if np.any(mask):
            prediction[mask] = predict_rff_model(models[str(mode)], x[mask])
            assigned[mask] = True
    if not np.all(assigned):
        raise ValueError(f"unknown evaluation modes: {sorted(set(labels[~assigned]))}")
    return prediction


def fit_innovation_basis(
    carrier_features: ArrayLike,
    target_residual: ArrayLike,
    *,
    rank: int,
) -> dict[str, Any]:
    """Fit a low-rank carrier direction aligned only with transition residuals."""

    carrier = _matrix(carrier_features, "carrier_features")
    residual = _matrix(target_residual, "target_residual")
    if len(carrier) != len(residual):
        raise ValueError("carrier features and target residual must align")
    selected_rank = int(rank)
    if not 1 <= selected_rank <= min(carrier.shape[1], residual.shape[1]):
        raise ValueError("innovation rank lies outside matrix dimensions")
    mean = np.mean(carrier, axis=0)
    scale = np.maximum(np.std(carrier, axis=0, ddof=1), 1e-8)
    white = (carrier - mean) / scale
    target_scale = np.maximum(np.std(residual, axis=0, ddof=1), 1e-8)
    cross = white.T @ (residual / target_scale) / len(white)
    left, singular, _ = np.linalg.svd(cross, full_matrices=False)
    if singular[selected_rank - 1] <= max(float(singular[0]) * 1e-10, 1e-12):
        raise ValueError("supervised innovation map is rank deficient")
    basis = left[:, :selected_rank]
    # Stabilize otherwise arbitrary SVD signs for exact reproducibility.
    for column in range(basis.shape[1]):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0:
            basis[:, column] *= -1
    return {
        "mean": mean,
        "scale": scale,
        "basis": basis,
        "singular_values": singular,
        "rank": selected_rank,
    }


def transform_innovation(
    model: Mapping[str, Any],
    carrier_features: ArrayLike,
) -> FloatArray:
    carrier = _matrix(carrier_features, "carrier_features")
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    basis = np.asarray(model["basis"], dtype=np.float64)
    if carrier.shape[1] != len(mean) or len(mean) != len(scale) or basis.shape[0] != len(mean):
        raise ValueError("innovation dimensions do not agree")
    return ((carrier - mean) / scale) @ basis


def fit_candidate_model(
    state: ArrayLike,
    actions: ArrayLike,
    carrier_features: ArrayLike,
    targets: ArrayLike,
    modes: ArrayLike,
    *,
    state_rank: int,
    innovation_rank: int,
    regime_specific: bool,
    mode_labels: Sequence[str],
    width: int,
    penalty: float,
    seed: int,
) -> dict[str, Any]:
    """Fit a frozen candidate state and its one-step/word transition map."""

    q_all = _matrix(state, "state")
    action = _matrix(actions, "actions")
    carrier = _matrix(carrier_features, "carrier_features")
    target = _matrix(targets, "targets")
    labels = _labels(modes, len(q_all), "modes")
    if not (len(action) == len(carrier) == len(target) == len(q_all)):
        raise ValueError("candidate arrays must align")
    selected_state_rank = int(state_rank)
    selected_innovation_rank = int(innovation_rank)
    if not 1 <= selected_state_rank <= q_all.shape[1]:
        raise ValueError("state_rank lies outside state width")
    q = q_all[:, :selected_state_rank]
    innovation_model = None
    innovation = np.empty((len(q), 0), dtype=np.float64)
    if selected_innovation_rank:
        base_inputs = np.column_stack([q, action])
        base = fit_regime_dynamics(
            base_inputs, target, labels,
            regime_specific=regime_specific, mode_labels=mode_labels,
            width=width, penalty=penalty,
            seed=stable_seed(seed, "innovation_base"),
        )
        base_prediction = predict_regime_dynamics(base, base_inputs, labels)
        innovation_model = fit_innovation_basis(
            carrier, target - base_prediction, rank=selected_innovation_rank
        )
        innovation = transform_innovation(innovation_model, carrier)
    state_features = np.column_stack([q, innovation])
    dynamics = fit_regime_dynamics(
        np.column_stack([state_features, action]), target, labels,
        regime_specific=regime_specific, mode_labels=mode_labels,
        width=width, penalty=penalty,
        seed=stable_seed(seed, "candidate_dynamics"),
    )
    return {
        "state_rank": selected_state_rank,
        "innovation_rank": selected_innovation_rank,
        "regime_specific": bool(regime_specific),
        "innovation_model": innovation_model,
        "dynamics": dynamics,
        "state_coordinate_count": selected_state_rank + selected_innovation_rank,
    }


def candidate_state_features(
    model: Mapping[str, Any],
    state: ArrayLike,
    carrier_features: ArrayLike,
) -> FloatArray:
    q = _matrix(state, "state")[:, : int(model["state_rank"])]
    innovation_model = model.get("innovation_model")
    if innovation_model is None:
        return q
    return np.column_stack([
        q,
        transform_innovation(innovation_model, carrier_features),
    ])


def predict_candidate_model(
    model: Mapping[str, Any],
    state: ArrayLike,
    actions: ArrayLike,
    carrier_features: ArrayLike,
    modes: ArrayLike,
) -> FloatArray:
    state_features = candidate_state_features(model, state, carrier_features)
    action = _matrix(actions, "actions")
    return predict_regime_dynamics(
        model["dynamics"], np.column_stack([state_features, action]), modes
    )


def grouped_candidate_oof(
    state: ArrayLike,
    actions: ArrayLike,
    carrier_features: ArrayLike,
    targets: ArrayLike,
    groups: ArrayLike,
    modes: ArrayLike,
    *,
    state_rank: int,
    innovation_rank: int,
    regime_specific: bool,
    mode_labels: Sequence[str],
    width: int,
    penalty: float,
    folds: int,
    seed: int,
) -> dict[str, Any]:
    q = _matrix(state, "state")
    action = _matrix(actions, "actions")
    carrier = _matrix(carrier_features, "carrier_features")
    target = _matrix(targets, "targets")
    group = _labels(groups, len(q), "groups")
    labels = _labels(modes, len(q), "modes")
    prediction = np.empty_like(target)
    fold_rows = []
    for fold_index, held_out in enumerate(grouped_folds(group, int(folds), int(seed))):
        fitted = fit_candidate_model(
            q[~held_out], action[~held_out], carrier[~held_out], target[~held_out],
            labels[~held_out], state_rank=state_rank,
            innovation_rank=innovation_rank, regime_specific=regime_specific,
            mode_labels=mode_labels, width=width, penalty=penalty,
            seed=stable_seed(seed, "fold", fold_index),
        )
        prediction[held_out] = predict_candidate_model(
            fitted, q[held_out], action[held_out], carrier[held_out], labels[held_out]
        )
        fold_rows.append(int(np.sum(held_out)))
    error = np.mean((prediction - target) ** 2, axis=1)
    return {
        "prediction": prediction,
        "row_mse": error,
        "oof_mse": float(np.mean(error)),
        "fold_rows": fold_rows,
    }


def select_simplest_candidate(
    rows: Iterable[Mapping[str, Any]],
    *,
    relative_tolerance: float,
) -> dict[str, Any]:
    candidates = [dict(row) for row in rows]
    if not candidates:
        raise ValueError("candidate table is empty")
    losses = np.asarray([float(row["oof_mse"]) for row in candidates])
    if not np.all(np.isfinite(losses)) or np.any(losses < 0):
        raise ValueError("candidate losses must be finite and nonnegative")
    best = float(np.min(losses))
    ceiling = best * (1.0 + float(relative_tolerance))
    eligible = [row for row in candidates if float(row["oof_mse"]) <= ceiling]
    selected = min(
        eligible,
        key=lambda row: (
            int(row["state_rank"]) + int(row["innovation_rank"]),
            int(row["innovation_rank"]),
            bool(row["regime_specific"]),
            int(row["state_rank"]),
            float(row["oof_mse"]),
            float(row["penalty"]),
        ),
    )
    return {**selected, "best_oof_mse": best, "selection_ceiling": ceiling}


def aggregate_relative_gain(primary_error: ArrayLike, comparator_error: ArrayLike) -> float:
    primary = np.asarray(primary_error, dtype=np.float64).reshape(-1)
    comparator = np.asarray(comparator_error, dtype=np.float64).reshape(-1)
    if len(primary) != len(comparator) or not len(primary):
        raise ValueError("paired error arrays must be nonempty and aligned")
    if not np.all(np.isfinite(primary)) or not np.all(np.isfinite(comparator)):
        raise ValueError("paired error arrays contain nonfinite values")
    return float((np.mean(comparator) - np.mean(primary)) / max(np.mean(comparator), 1e-12))


def clustered_relative_gain_interval(
    primary_error: ArrayLike,
    comparator_error: ArrayLike,
    groups: ArrayLike,
    *,
    draws: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    primary = np.asarray(primary_error, dtype=np.float64).reshape(-1)
    comparator = np.asarray(comparator_error, dtype=np.float64).reshape(-1)
    labels = _labels(groups, len(primary), "groups")
    if len(primary) != len(comparator):
        raise ValueError("paired errors must align")
    unique = np.unique(labels)
    if len(unique) < 2 or int(draws) < 1:
        raise ValueError("cluster bootstrap requires at least two groups and one draw")
    generator = np.random.default_rng(int(seed))
    values = np.empty(int(draws), dtype=np.float64)
    row_indices = {group: np.flatnonzero(labels == group) for group in unique}
    for index in range(int(draws)):
        sampled = generator.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([row_indices[group] for group in sampled])
        values[index] = aggregate_relative_gain(primary[rows], comparator[rows])
    return (
        float(np.quantile(values, float(alpha) / 2.0)),
        float(np.quantile(values, 1.0 - float(alpha) / 2.0)),
    )


def within_group_permuted_labels(
    modes: ArrayLike,
    groups: ArrayLike,
    units: ArrayLike,
    *,
    seed: int,
) -> NDArray[np.str_]:
    """Permute record-level mode identities independently within trajectories."""

    labels = np.asarray(modes).reshape(-1).astype(str)
    group = _labels(groups, len(labels), "groups")
    unit = _labels(units, len(labels), "units")
    result = np.empty(len(labels), dtype=object)
    for group_value in np.unique(group):
        mask = group == group_value
        group_units = np.unique(unit[mask])
        source_labels = []
        for unit_value in group_units:
            observed = np.unique(labels[mask & (unit == unit_value)])
            if len(observed) != 1:
                raise ValueError("each unit must have exactly one mode label")
            source_labels.append(str(observed[0]))
        generator = np.random.default_rng(stable_seed(seed, group_value))
        permuted = generator.permutation(source_labels)
        for unit_value, mode in zip(group_units, permuted):
            result[mask & (unit == unit_value)] = str(mode)
    return result.astype(str)


@dataclass(frozen=True)
class Stage343Gates:
    upstream_binding: bool
    stage342_binding: bool
    selection_improvement: bool
    evaluation_improvement: bool
    residual_sufficiency: bool
    coordinate_necessity: bool
    mode_specificity: bool


def derive_stage343_decision(
    gates: Stage343Gates | Mapping[str, bool],
    *,
    run_mode: str = "pilot",
) -> dict[str, Any]:
    values = gates.__dict__ if isinstance(gates, Stage343Gates) else dict(gates)
    required = [
        "upstream_binding", "stage342_binding", "selection_improvement",
        "evaluation_improvement", "residual_sufficiency",
        "coordinate_necessity", "mode_specificity",
    ]
    missing = [name for name in required if name not in values]
    if missing:
        raise ValueError(f"missing Stage 34.3 gates: {missing}")
    checks = {name: bool(values[name]) for name in required}
    if run_mode == "smoke":
        status = "smoke_only"
    elif not checks["upstream_binding"] or not checks["stage342_binding"]:
        status = "inconclusive_upstream_binding_failure"
    elif not checks["selection_improvement"]:
        status = "no_selected_regime_innovation_repair"
    elif not checks["evaluation_improvement"]:
        status = "selected_repair_did_not_transfer"
    elif not checks["residual_sufficiency"]:
        status = "selected_state_still_carrier_incomplete"
    elif not checks["coordinate_necessity"]:
        status = "selected_state_not_minimal"
    elif not checks["mode_specificity"]:
        status = "physical_mode_structure_not_specific"
    else:
        status = "bounded_jepa_state_candidate_repaired"
    passed = bool(run_mode == "pilot" and all(checks.values()))
    return {
        "status": status,
        "passed": passed,
        "checks": checks,
        "failed_checks": [name for name in required if not checks[name]],
        "run_mode": str(run_mode),
        "confirmation_eligible": False,
        "causal_evidence": False,
        "claim_scope": "post_outcome_cpu_diagnostic_of_bounded_jepa_state",
    }
