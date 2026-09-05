"""Numerical core for the Stage 44 visual--causal realization audit.

The routines in this module deliberately avoid simulator and model imports so
that the prospective geometry, probes, counterfactual metrics, and decision
tree can be tested locally.  RGB reconstructions are diagnostic in Stage 44;
none of the decision functions below treats a visually plausible decoder
output as evidence that a predictive state is physically correct.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike


def _finite_matrix(value: ArrayLike, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite matrix")
    return result


def canonical_visual_tokens(
    value: ArrayLike,
    *,
    expected_tokens: int = 256,
    expected_width: int = 384,
) -> np.ndarray:
    """Canonicalize a visual field to ``[frames, patches, channels]``.

    Accepted inputs end in either ``[patches, channels]`` or
    ``[height, width, channels]``.  Singleton view dimensions are harmless;
    every remaining leading dimension is treated as part of the frame index.
    """

    array = np.asarray(value)
    if array.ndim < 2 or not np.all(np.isfinite(array)):
        raise ValueError("visual field must be finite and at least two-dimensional")
    if array.shape[-2:] == (int(expected_tokens), int(expected_width)):
        result = array.reshape(-1, int(expected_tokens), int(expected_width))
    elif (
        array.ndim >= 3
        and array.shape[-1] == int(expected_width)
        and int(array.shape[-3]) * int(array.shape[-2]) == int(expected_tokens)
    ):
        result = array.reshape(-1, int(expected_tokens), int(expected_width))
    else:
        raise ValueError(f"unexpected visual field shape {array.shape}")
    return np.asarray(result, dtype=np.float64)


def fit_channel_pca(value: ArrayLike, rank: int) -> dict[str, np.ndarray]:
    """Fit a construction-only PCA basis to patch-channel observations."""

    matrix = _finite_matrix(value, "patch channels")
    selected_rank = int(rank)
    if selected_rank < 1 or selected_rank > min(matrix.shape):
        raise ValueError("PCA rank is outside the matrix dimensions")
    mean = np.mean(matrix, axis=0)
    centered = matrix - mean
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    components = right[:selected_rank]
    explained = singular[:selected_rank] ** 2
    total = float(np.sum(singular**2))
    ratio = explained / max(total, 1e-12)
    return {
        "mean": mean,
        "components": components,
        "explained_variance_ratio": ratio,
    }


def project_visual_tokens(value: ArrayLike, artifact: Mapping[str, ArrayLike]) -> np.ndarray:
    """Project canonical patch fields through a frozen channel PCA basis."""

    tokens = np.asarray(value, dtype=np.float64)
    if tokens.ndim != 3 or not np.all(np.isfinite(tokens)):
        raise ValueError("tokens must have shape [frames, patches, channels]")
    mean = np.asarray(artifact["mean"], dtype=np.float64)
    components = _finite_matrix(artifact["components"], "PCA components")
    if tokens.shape[-1] != len(mean) or components.shape[1] != len(mean):
        raise ValueError("tokens and PCA artifact do not align")
    return np.einsum("npc,rc->npr", tokens - mean, components)


def spatial_pyramid_summary(value: ArrayLike, *, grid_size: int = 16, bins: int = 4) -> np.ndarray:
    """Pool projected patch fields into a compact position-sensitive summary."""

    tokens = np.asarray(value, dtype=np.float64)
    if tokens.ndim != 3 or not np.all(np.isfinite(tokens)):
        raise ValueError("projected tokens must have shape [frames, patches, channels]")
    grid, bin_count = int(grid_size), int(bins)
    if grid < 1 or bin_count < 1 or grid % bin_count or tokens.shape[1] != grid * grid:
        raise ValueError("grid and bin geometry are incompatible")
    fields = tokens.reshape(len(tokens), grid, grid, tokens.shape[-1])
    side = grid // bin_count
    pooled = fields.reshape(
        len(tokens), bin_count, side, bin_count, side, tokens.shape[-1]
    ).mean(axis=(2, 4))
    global_std = np.std(fields, axis=(1, 2))
    result = np.concatenate([pooled.reshape(len(tokens), -1), global_std], axis=1)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("spatial summary contains nonfinite values")
    return result


def fit_ridge(design: ArrayLike, target: ArrayLike, penalty: float) -> dict[str, np.ndarray]:
    """Fit multi-output ridge with an unpenalized intercept."""

    x = _finite_matrix(design, "design")
    y = _finite_matrix(target, "target")
    lam = float(penalty)
    if len(x) != len(y) or not np.isfinite(lam) or lam < 0:
        raise ValueError("ridge inputs are invalid")
    x_mean, y_mean = np.mean(x, axis=0), np.mean(y, axis=0)
    xc, yc = x - x_mean, y - y_mean
    if x.shape[1] <= x.shape[0]:
        gram = xc.T @ xc + lam * np.eye(x.shape[1])
        weight = np.linalg.solve(gram, xc.T @ yc)
    else:
        gram = xc @ xc.T + lam * np.eye(x.shape[0])
        weight = xc.T @ np.linalg.solve(gram, yc)
    return {"weight": weight, "intercept": y_mean - x_mean @ weight}


def ridge_predict(artifact: Mapping[str, ArrayLike], design: ArrayLike) -> np.ndarray:
    x = _finite_matrix(design, "design")
    weight = _finite_matrix(artifact["weight"], "weight")
    intercept = np.asarray(artifact["intercept"], dtype=np.float64)
    if x.shape[1] != weight.shape[0] or intercept.shape != (weight.shape[1],):
        raise ValueError("ridge artifact and design do not align")
    return x @ weight + intercept


def select_ridge_penalty(
    train_design: ArrayLike,
    train_target: ArrayLike,
    validation_design: ArrayLike,
    validation_target: ArrayLike,
    penalties: list[float],
) -> dict[str, Any]:
    """Select regularization strictly on a model-selection split."""

    x_train = _finite_matrix(train_design, "train design")
    y_train = _finite_matrix(train_target, "train target")
    x_val = _finite_matrix(validation_design, "validation design")
    y_val = _finite_matrix(validation_target, "validation target")
    candidates = sorted(set(float(value) for value in penalties))
    if len(x_train) != len(y_train) or len(x_val) != len(y_val) or not candidates:
        raise ValueError("ridge selection inputs are invalid")
    rows = []
    for penalty in candidates:
        model = fit_ridge(x_train, y_train, penalty)
        prediction = ridge_predict(model, x_val)
        rows.append({"penalty": penalty, "validation_mse": float(np.mean((prediction - y_val) ** 2))})
    selected = min(rows, key=lambda row: (row["validation_mse"], row["penalty"]))
    return {"selected_penalty": float(selected["penalty"]), "candidate_rows": rows}


def variance_weighted_r2(target: ArrayLike, prediction: ArrayLike) -> float:
    truth = _finite_matrix(target, "target")
    pred = _finite_matrix(prediction, "prediction")
    if truth.shape != pred.shape:
        raise ValueError("target and prediction do not align")
    residual = float(np.sum((truth - pred) ** 2))
    centered = float(np.sum((truth - np.mean(truth, axis=0)) ** 2))
    return float(1.0 - residual / max(centered, 1e-12))


def binary_auroc(labels: ArrayLike, scores: ArrayLike) -> float:
    """Tie-correct binary AUROC using the Mann--Whitney identity."""

    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(y) != len(s) or set(np.unique(y)) - {0, 1} or not np.all(np.isfinite(s)):
        raise ValueError("binary AUROC inputs are invalid")
    positive, negative = int(np.sum(y == 1)), int(np.sum(y == 0))
    if positive == 0 or negative == 0:
        raise ValueError("binary AUROC requires both classes")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    start = 0
    while start < len(s):
        stop = start + 1
        while stop < len(s) and s[order[stop]] == s[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return float((np.sum(ranks[y == 1]) - positive * (positive + 1) / 2) / (positive * negative))


def macro_one_vs_rest_auroc(labels: ArrayLike, scores: ArrayLike) -> float:
    """Macro one-vs-rest AUROC for frozen patch-localization probes."""

    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    matrix = _finite_matrix(scores, "scores")
    if len(y) != len(matrix) or np.any(y < 0) or np.any(y >= matrix.shape[1]):
        raise ValueError("multiclass AUROC inputs are invalid")
    values = [binary_auroc(y == index, matrix[:, index]) for index in range(matrix.shape[1])]
    return float(np.mean(values))


def local_support_geometry(
    reference: ArrayLike,
    query: ArrayLike,
    *,
    neighbors: int = 16,
    tangent_rank: int = 8,
) -> dict[str, np.ndarray]:
    """Measure nearest-support and local normal residuals.

    A query can be close to a sample while leaving the locally estimated
    tangent plane.  Reporting both distances avoids calling a representation
    "off manifold" merely because reconstruction quality is poor.
    """

    ref = _finite_matrix(reference, "reference")
    qry = _finite_matrix(query, "query")
    if ref.shape[1] != qry.shape[1]:
        raise ValueError("reference and query widths differ")
    k = int(neighbors)
    rank = int(tangent_rank)
    if k < 3 or k > len(ref) or rank < 1 or rank >= k or rank > ref.shape[1]:
        raise ValueError("local support geometry parameters are invalid")
    scale = np.maximum(np.std(ref, axis=0, ddof=1), 1e-8)
    refn = (ref - np.mean(ref, axis=0)) / scale
    qryn = (qry - np.mean(ref, axis=0)) / scale
    nearest, normal, tangent = [], [], []
    for row in qryn:
        distance = np.sum((refn - row) ** 2, axis=1)
        indices = np.argpartition(distance, k - 1)[:k]
        neighborhood = refn[indices]
        center = np.mean(neighborhood, axis=0)
        _, _, right = np.linalg.svd(neighborhood - center, full_matrices=False)
        delta = row - center
        coordinates = right[:rank] @ delta
        tangent_part = coordinates @ right[:rank]
        normal_part = delta - tangent_part
        nearest.append(float(np.sqrt(np.min(distance))))
        tangent.append(float(np.linalg.norm(tangent_part)))
        normal.append(float(np.linalg.norm(normal_part)))
    return {
        "nearest_distance": np.asarray(nearest),
        "tangent_distance": np.asarray(tangent),
        "normal_distance": np.asarray(normal),
    }


def row_cosine(left: ArrayLike, right: ArrayLike) -> np.ndarray:
    first = _finite_matrix(left, "left")
    second = _finite_matrix(right, "right")
    if first.shape != second.shape:
        raise ValueError("cosine rows do not align")
    numerator = np.sum(first * second, axis=1)
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    return numerator / np.maximum(denominator, 1e-12)


def counterfactual_realization_metrics(
    target_left: ArrayLike,
    target_right: ArrayLike,
    predicted_left: ArrayLike,
    predicted_right: ArrayLike,
) -> dict[str, float]:
    """Compare matched predicted and target counterfactual effects."""

    truth = _finite_matrix(target_left, "target left") - _finite_matrix(target_right, "target right")
    prediction = _finite_matrix(predicted_left, "predicted left") - _finite_matrix(predicted_right, "predicted right")
    if truth.shape != prediction.shape:
        raise ValueError("counterfactual effects do not align")
    truth_norm = np.linalg.norm(truth, axis=1)
    prediction_norm = np.linalg.norm(prediction, axis=1)
    active = truth_norm > 1e-10
    if not np.any(active):
        raise ValueError("counterfactual target effect is identically zero")
    cosine = row_cosine(truth[active], prediction[active])
    relative_error = np.linalg.norm(prediction[active] - truth[active], axis=1) / np.maximum(truth_norm[active], 1e-12)
    gain = prediction_norm[active] / np.maximum(truth_norm[active], 1e-12)
    return {
        "effect_rows": int(np.sum(active)),
        "median_cosine": float(np.median(cosine)),
        "median_relative_error": float(np.median(relative_error)),
        "median_magnitude_ratio": float(np.median(gain)),
    }


def masked_effect_energy(value: ArrayLike, mask: ArrayLike) -> np.ndarray:
    """Return the fraction of counterfactual feature energy inside a patch mask."""

    effect = np.asarray(value, dtype=np.float64)
    selected = np.asarray(mask, dtype=bool)
    if effect.ndim != 3 or selected.shape != effect.shape[:2] or not np.all(np.isfinite(effect)):
        raise ValueError("effect and mask are not patch-aligned")
    energy = np.sum(effect**2, axis=-1)
    return np.sum(energy * selected, axis=1) / np.maximum(np.sum(energy, axis=1), 1e-12)


@dataclass(frozen=True)
class Stage44Decision:
    classification: str
    passed: bool
    encoder_observable: bool
    one_step_adequate: bool
    recursive_stable: bool
    causal_realization: bool
    counterfactual_training_authorized: bool
    object_centric_encoder_authorized: bool
    planning_audit_authorized: bool


def derive_stage44_decision(
    *,
    support_certified: bool,
    decoder_contract_valid: bool,
    encoder_observable: bool,
    one_step_adequate: bool,
    recursive_stable: bool,
    causal_realization: bool,
) -> Stage44Decision:
    """Fail-closed decision tree for the next architecture experiment."""

    if not support_certified:
        classification = "event_support_not_certified"
    elif not decoder_contract_valid:
        classification = "official_decoder_contract_failure"
    elif not encoder_observable:
        classification = "encoder_observability_insufficient"
    elif not one_step_adequate:
        classification = "one_step_predictor_failure"
    elif not recursive_stable:
        classification = "recursive_predictor_failure"
    elif not causal_realization:
        classification = "causal_visual_content_missing"
    else:
        classification = "visual_causal_state_adequate"
    counterfactual = bool(
        support_certified and decoder_contract_valid and encoder_observable
        and (not one_step_adequate or not recursive_stable or not causal_realization)
    )
    object_centric = bool(support_certified and decoder_contract_valid and not encoder_observable)
    planning = bool(classification == "visual_causal_state_adequate")
    return Stage44Decision(
        classification=classification,
        passed=planning,
        encoder_observable=bool(encoder_observable),
        one_step_adequate=bool(one_step_adequate),
        recursive_stable=bool(recursive_stable),
        causal_realization=bool(causal_realization),
        counterfactual_training_authorized=counterfactual,
        object_centric_encoder_authorized=object_centric,
        planning_audit_authorized=planning,
    )
