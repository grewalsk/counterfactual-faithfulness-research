"""Numerical core for Stage 34 predictive-fiber causal abstraction.

Stage 34 deliberately avoids fitting a JEPA-to-DINO state map.  It defines a
single model-free response chart from simulator action contrasts and asks
whether each frozen world model separately implements that high-level state.

The routines here are NumPy-only so the important estimands can be tested
locally without model checkpoints or a GPU.  Arrays store observations in
rows.  Grouped procedures split and resample whole trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _finite_array(values: ArrayLike, name: str, *, ndim: int | None = None) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must be nonempty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains nonfinite values")
    return array


def _word_lookup(word_names: Sequence[str]) -> dict[str, int]:
    names = [str(value) for value in word_names]
    if len(names) != len(set(names)):
        raise ValueError("word_names must be unique")
    return {name: index for index, name in enumerate(names)}


def action_contrast_signature(
    grounded_paths: ArrayLike,
    word_names: Sequence[str],
    word_lengths: ArrayLike,
    response_words: Sequence[str],
    zero_word_by_length: Mapping[int, str],
    *,
    order_pairs: Sequence[tuple[str, str]] = (),
) -> FloatArray:
    """Return no-op-corrected path and fixed-multiset order contrasts.

    ``grounded_paths`` has shape ``(word, step, observable)``.  For every
    response word, the full valid path is differenced against a zero-action
    word of the same length.  Each order pair contributes an additional final
    step contrast.  Static offsets shared by every action word cancel exactly.
    """

    paths = _finite_array(grounded_paths, "grounded_paths", ndim=3)
    lengths = np.asarray(word_lengths, dtype=np.int64).reshape(-1)
    if len(lengths) != len(paths) or len(word_names) != len(paths):
        raise ValueError("word metadata must match grounded_paths")
    if np.any(lengths < 1) or np.any(lengths > paths.shape[1]):
        raise ValueError("word_lengths lie outside grounded_paths")
    lookup = _word_lookup(word_names)
    pieces: list[FloatArray] = []
    deltas: dict[str, FloatArray] = {}
    for word in map(str, response_words):
        if word not in lookup:
            raise KeyError(f"missing response word {word!r}")
        index = lookup[word]
        length = int(lengths[index])
        zero = str(zero_word_by_length.get(length, ""))
        if zero not in lookup:
            raise KeyError(f"missing length-{length} zero word for {word!r}")
        zero_index = lookup[zero]
        if int(lengths[zero_index]) != length:
            raise ValueError("zero word length does not match response word")
        delta = paths[index, :length] - paths[zero_index, :length]
        deltas[word] = delta
        pieces.append(delta.reshape(-1))
    for left, right in order_pairs:
        first, second = str(left), str(right)
        if first not in deltas or second not in deltas:
            raise KeyError("order pairs must reference response_words")
        if deltas[first].shape != deltas[second].shape:
            raise ValueError("order-pair paths must have the same shape")
        pieces.append((deltas[first][-1] - deltas[second][-1]).reshape(-1))
    if not pieces:
        raise ValueError("response_words must be nonempty")
    result = np.concatenate(pieces).astype(np.float64, copy=False)
    if not np.all(np.isfinite(result)):
        raise ValueError("action contrast signature is nonfinite")
    return result


def fit_response_chart(signatures: ArrayLike, rank: int) -> dict[str, FloatArray | int]:
    """Fit a standardized physical-response PCA chart at a fixed rank."""

    values = _finite_array(signatures, "signatures", ndim=2)
    selected_rank = int(rank)
    if not 1 <= selected_rank <= min(values.shape):
        raise ValueError("rank lies outside the signature matrix")
    mean = np.mean(values, axis=0)
    scale = np.maximum(np.std(values, axis=0, ddof=1), 1e-8)
    standardized = (values - mean) / scale
    _, singular_values, right = np.linalg.svd(standardized, full_matrices=False)
    return {
        "mean": mean,
        "scale": scale,
        "basis": right[:selected_rank].T,
        "singular_values": singular_values,
        "rank": selected_rank,
    }


def response_coordinates(
    signatures: ArrayLike,
    chart: Mapping[str, ArrayLike | int],
    *,
    rank: int | None = None,
) -> FloatArray:
    """Project one or more response signatures into a frozen chart."""

    values = np.asarray(signatures, dtype=np.float64)
    was_vector = values.ndim == 1
    values = np.atleast_2d(values)
    mean = _finite_array(chart["mean"], "chart mean").reshape(-1)
    scale = _finite_array(chart["scale"], "chart scale").reshape(-1)
    basis = _finite_array(chart["basis"], "chart basis", ndim=2)
    if values.shape[1] != len(mean) or len(scale) != len(mean) or basis.shape[0] != len(mean):
        raise ValueError("signature and chart dimensions do not agree")
    selected = basis.shape[1] if rank is None else int(rank)
    if not 1 <= selected <= basis.shape[1]:
        raise ValueError("requested rank lies outside the chart")
    coordinates = ((values - mean) / scale) @ basis[:, :selected]
    return coordinates[0] if was_vector else coordinates


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
    """Return deterministic held-out masks that keep trajectories intact."""

    labels = np.asarray(groups).reshape(-1)
    unique = np.asarray(_ordered_unique(labels), dtype=object)
    count = min(int(folds), len(unique))
    if count < 2:
        raise ValueError("at least two independent groups are required")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(unique))
    parts = np.array_split(unique[order], count)
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
    intercept = mean_y - mean_x @ weight
    return weight, intercept


def grouped_ridge_oof(
    features: ArrayLike,
    targets: ArrayLike,
    groups: ArrayLike,
    *,
    penalties: Iterable[float] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0),
    folds: int = 4,
    seed: int = 0,
) -> dict[str, Any]:
    """Choose ridge penalty by trajectory-grouped out-of-fold MSE."""

    x = _finite_array(features, "features", ndim=2)
    y = np.asarray(targets, dtype=np.float64)
    target_was_vector = y.ndim == 1
    y = np.atleast_2d(y).T if target_was_vector else _finite_array(y, "targets", ndim=2)
    if len(x) != len(y):
        raise ValueError("features and targets must have the same rows")
    labels = np.asarray(groups).reshape(-1)
    if len(labels) != len(x):
        raise ValueError("groups must have one value per row")
    masks = grouped_folds(labels, folds, seed)
    candidates = [float(value) for value in penalties]
    if not candidates or any(not np.isfinite(value) or value < 0 for value in candidates):
        raise ValueError("penalties must be finite and nonnegative")
    losses, predictions = [], []
    for penalty in candidates:
        prediction = np.empty_like(y)
        for held_out in masks:
            weight, intercept = _ridge_fit(x[~held_out], y[~held_out], penalty)
            prediction[held_out] = x[held_out] @ weight + intercept
        losses.append(float(np.mean((prediction - y) ** 2)))
        predictions.append(prediction)
    selected = int(np.argmin(losses))
    weight, intercept = _ridge_fit(x, y, candidates[selected])
    oof = predictions[selected]
    return {
        "weight": weight[:, 0] if target_was_vector else weight,
        "intercept": float(intercept[0]) if target_was_vector else intercept,
        "penalty": candidates[selected],
        "oof_prediction": oof[:, 0] if target_was_vector else oof,
        "oof_mse": losses[selected],
        "all_oof_mse": losses,
    }


def nested_predictive_sufficiency(
    state: ArrayLike,
    actions: ArrayLike,
    residual_features: ArrayLike,
    targets: ArrayLike,
    groups: ArrayLike,
    *,
    penalties: Iterable[float] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0),
    folds: int = 4,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare a predictive-state update with one augmented by residual latent data."""

    q = _finite_array(state, "state", ndim=2)
    a = _finite_array(actions, "actions", ndim=2)
    residual = _finite_array(residual_features, "residual_features", ndim=2)
    if not (len(q) == len(a) == len(residual)):
        raise ValueError("state, actions, and residual_features must align")
    base_features = np.column_stack([q, a])
    enriched_features = np.column_stack([q, a, residual])
    base = grouped_ridge_oof(
        base_features, targets, groups, penalties=penalties, folds=folds, seed=seed
    )
    enriched = grouped_ridge_oof(
        enriched_features,
        targets,
        groups,
        penalties=penalties,
        folds=folds,
        seed=seed + 1,
    )
    improvement = (base["oof_mse"] - enriched["oof_mse"]) / max(base["oof_mse"], 1e-12)
    return {
        "base": base,
        "enriched": enriched,
        "residual_relative_improvement": float(improvement),
    }


def fit_supervised_subspace(
    carriers: ArrayLike,
    coordinates: ArrayLike,
    *,
    rank: int,
    ridge: float = 1e-3,
) -> dict[str, FloatArray | int | float]:
    """Fit an output-aligned carrier subspace without forming an ambient covariance.

    The returned orthonormal basis spans the carrier directions used by a ridge
    map to the canonical response coordinates.  ``carriers`` may be much wider
    than the number of calibration examples.
    """

    x = _finite_array(carriers, "carriers", ndim=2)
    y = _finite_array(coordinates, "coordinates", ndim=2)
    if len(x) != len(y):
        raise ValueError("carriers and coordinates must have the same rows")
    selected_rank = int(rank)
    if not 1 <= selected_rank <= min(x.shape[0] - 1, x.shape[1], y.shape[1]):
        raise ValueError("rank lies outside supervised subspace dimensions")
    mean = np.mean(x, axis=0)
    scale = np.maximum(np.std(x, axis=0, ddof=1), 1e-8)
    white = (x - mean) / scale
    centered_y = y - np.mean(y, axis=0)
    dual = np.linalg.solve(
        white @ white.T + float(ridge) * np.eye(len(white), dtype=np.float64),
        centered_y,
    )
    weights = white.T @ dual
    left, singular, _ = np.linalg.svd(weights, full_matrices=False)
    if len(singular) < selected_rank or singular[selected_rank - 1] <= 1e-12:
        raise ValueError("supervised carrier map is rank deficient")
    basis = left[:, :selected_rank]
    return {
        "mean": mean,
        "scale": scale,
        "basis": basis,
        "singular_values": singular,
        "rank": selected_rank,
        "ridge": float(ridge),
    }


def split_carrier_delta(
    delta: ArrayLike,
    subspace: Mapping[str, ArrayLike | int | float],
) -> tuple[FloatArray, FloatArray]:
    """Split a carrier delta into aligned and predictive-fiber components."""

    value = _finite_array(delta, "delta").reshape(-1)
    scale = _finite_array(subspace["scale"], "subspace scale").reshape(-1)
    basis = _finite_array(subspace["basis"], "subspace basis", ndim=2)
    if len(value) != len(scale) or basis.shape[0] != len(value):
        raise ValueError("delta and subspace dimensions do not agree")
    white = value / scale
    aligned_white = basis @ (basis.T @ white)
    residual_white = white - aligned_white
    return aligned_white * scale, residual_white * scale


def matched_fiber_pairs(
    coordinates: ArrayLike,
    residual_features: ArrayLike,
    modes: Sequence[Any],
    trajectory_ids: ArrayLike,
    *,
    kind: str = "fiber",
) -> IntArray:
    """Match trajectory-disjoint pairs within mode for fiber or state interventions.

    ``fiber`` pairs prioritize small canonical-state distance and large residual
    distance.  ``state`` pairs prioritize large canonical-state distance and
    small residual distance.  Every row receives one deterministic donor.
    """

    q = _finite_array(coordinates, "coordinates", ndim=2)
    r = _finite_array(residual_features, "residual_features", ndim=2)
    labels = np.asarray(modes).reshape(-1)
    trajectories = np.asarray(trajectory_ids).reshape(-1)
    if not (len(q) == len(r) == len(labels) == len(trajectories)):
        raise ValueError("matching arrays must have equal row counts")
    if kind not in {"fiber", "state"}:
        raise ValueError("kind must be 'fiber' or 'state'")
    q_scale = np.maximum(np.std(q, axis=0, ddof=1), 1e-8)
    r_scale = np.maximum(np.std(r, axis=0, ddof=1), 1e-8)
    donors = np.full(len(q), -1, dtype=np.int64)
    for index in range(len(q)):
        candidates = np.flatnonzero(
            (labels == labels[index]) & (trajectories != trajectories[index])
        )
        if not len(candidates):
            raise ValueError("each row needs a same-mode trajectory-disjoint donor")
        q_distance = np.linalg.norm((q[candidates] - q[index]) / q_scale, axis=1)
        r_distance = np.linalg.norm((r[candidates] - r[index]) / r_scale, axis=1)
        if kind == "fiber":
            score = q_distance / np.maximum(r_distance, 1e-8)
        else:
            score = r_distance / np.maximum(q_distance, 1e-8)
        donors[index] = int(candidates[int(np.argmin(score))])
    return np.column_stack([np.arange(len(q), dtype=np.int64), donors])


def intervention_ood_ratio(
    intervened_sketches: ArrayLike,
    natural_sketches: ArrayLike,
) -> FloatArray:
    """Nearest-natural distance divided by the natural leave-one-out 95% radius."""

    intervention = _finite_array(intervened_sketches, "intervened_sketches", ndim=2)
    natural = _finite_array(natural_sketches, "natural_sketches", ndim=2)
    if intervention.shape[1] != natural.shape[1] or len(natural) < 3:
        raise ValueError("sketch dimensions must agree and need at least three natural rows")
    natural_distances = np.linalg.norm(natural[:, None] - natural[None], axis=2)
    np.fill_diagonal(natural_distances, np.inf)
    natural_nearest = np.min(natural_distances, axis=1)
    radius = max(float(np.quantile(natural_nearest, 0.95)), 1e-12)
    nearest = np.min(
        np.linalg.norm(intervention[:, None] - natural[None], axis=2), axis=1
    )
    return nearest / radius


def cosine_rows(left: ArrayLike, right: ArrayLike) -> FloatArray:
    first = _finite_array(left, "left", ndim=2)
    second = _finite_array(right, "right", ndim=2)
    if first.shape != second.shape:
        raise ValueError("cosine arrays must have the same shape")
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    return np.divide(
        np.sum(first * second, axis=1),
        denominator,
        out=np.zeros(len(first), dtype=np.float64),
        where=denominator > 1e-12,
    )


def commutativity_metrics(
    predicted_next: ArrayLike,
    reference_next: ArrayLike,
    *,
    reference_error: float,
) -> dict[str, float]:
    """Measure model-to-high-level transition commutativity."""

    predicted = _finite_array(predicted_next, "predicted_next", ndim=2)
    reference = _finite_array(reference_next, "reference_next", ndim=2)
    if predicted.shape != reference.shape:
        raise ValueError("predicted and reference transitions must have equal shape")
    error = np.linalg.norm(predicted - reference, axis=1)
    scale = np.maximum(np.linalg.norm(reference, axis=1), 1e-12)
    relative = error / scale
    ceiling = float(reference_error)
    if not np.isfinite(ceiling) or ceiling <= 0:
        raise ValueError("reference_error must be positive and finite")
    return {
        "mean_relative_error": float(np.mean(relative)),
        "median_relative_error": float(np.median(relative)),
        "mean_cosine": float(np.mean(cosine_rows(predicted, reference))),
        "reference_normalized_error": float(np.mean(relative) / ceiling),
    }


@dataclass(frozen=True)
class Stage34Gates:
    action_specificity: bool
    predictive_sufficiency: bool
    on_manifold_causal_use: bool
    two_sided_commutativity: bool
    controls_rejected: bool
    family_consistency: bool


def derive_stage34_decision(
    gates: Stage34Gates | Mapping[str, bool],
    *,
    run_mode: str = "pilot",
    confirmation_eligible: bool = True,
) -> dict[str, Any]:
    """Return the preregistered sequential Stage 34 decision."""

    values = gates.__dict__ if isinstance(gates, Stage34Gates) else dict(gates)
    order = [
        "action_specificity",
        "predictive_sufficiency",
        "on_manifold_causal_use",
        "two_sided_commutativity",
        "controls_rejected",
        "family_consistency",
    ]
    missing = [name for name in order if name not in values]
    if missing:
        raise ValueError(f"missing Stage 34 gates: {missing}")
    checks = {name: bool(values[name]) for name in order}
    first_failure = next((name for name in order if not checks[name]), None)
    level = len(order) if first_failure is None else order.index(first_failure)
    passed = bool(
        run_mode == "pilot"
        and confirmation_eligible
        and first_failure is None
    )
    if run_mode == "smoke":
        status = "smoke_only"
    elif not confirmation_eligible:
        status = "inconclusive_source_or_split_failure"
    elif passed:
        status = "bounded_two_sided_causal_abstraction_supported"
    elif first_failure == "action_specificity":
        status = "shared_static_state_geometry_only"
    elif first_failure == "predictive_sufficiency":
        status = "candidate_predictive_state_insufficient"
    elif first_failure == "on_manifold_causal_use":
        status = "predictive_summary_not_causally_used"
    elif first_failure == "two_sided_commutativity":
        status = "models_do_not_share_the_high_level_transition"
    else:
        status = "bounded_abstraction_not_supported"
    return {
        "status": status,
        "passed": passed,
        "level": int(level),
        "first_failed_gate": first_failure,
        "checks": checks,
        "failed_checks": [name for name in order if not checks[name]],
        "run_mode": str(run_mode),
        "confirmation_eligible": bool(confirmation_eligible),
    }
