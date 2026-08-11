import numpy as np

from cf_faithfulness.stage34_2_split_path_continuation import (
    Stage342Gates,
    derive_stage342_decision,
    fit_grouped_diagonal_affine,
    fit_matched_control_basis,
    predict_diagonal_affine,
    project_delta_to_basis,
    row_cosine,
    row_norm_ratio,
    summarize_causal_rows,
)


def test_grouped_diagonal_affine_recovers_scale_and_bias():
    rng = np.random.default_rng(34201)
    groups = np.repeat(np.arange(16), 8)
    predictors = rng.normal(size=(len(groups), 3))
    targets = predictors * np.array([2.0, -0.5, 1.2]) + np.array([0.3, -1.0, 2.0])
    model = fit_grouped_diagonal_affine(
        predictors, targets, groups, penalties=(0.0, 1e-3), seed=11
    )
    prediction = predict_diagonal_affine(model, predictors)
    assert model["parameter_count"] == 6
    assert np.mean((prediction - targets) ** 2) < 1e-20
    assert np.allclose(model["scale"], [2.0, -0.5, 1.2])


def test_cosine_and_norm_ratio_are_rowwise_and_zero_safe():
    first = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 2.0]])
    second = np.array([[2.0, 0.0], [1.0, 0.0], [0.0, -4.0]])
    assert np.allclose(row_cosine(first, second), [1.0, 0.0, -1.0])
    assert np.allclose(row_norm_ratio(first[[0, 2]], second[[0, 2]]), [0.5, 0.5])


def test_matched_control_basis_is_orthogonal_to_primary():
    rng = np.random.default_rng(34202)
    carriers = rng.normal(size=(80, 12))
    primary = np.eye(12)[:, :2]
    subspace = {
        "mean": np.mean(carriers, axis=0),
        "scale": np.maximum(np.std(carriers, axis=0, ddof=1), 1e-8),
        "basis": primary,
    }
    control = fit_matched_control_basis(carriers, subspace, rank=3)
    assert control.shape == (12, 3)
    assert np.linalg.norm(primary.T @ control) < 1e-10
    delta = rng.normal(size=12)
    projected = project_delta_to_basis(delta, subspace, control)
    white = projected / subspace["scale"]
    assert np.linalg.norm(primary.T @ white) < 1e-10


def _causal_rows():
    rows = []
    for mode in ["free", "contact"]:
        rows.extend([
            {
                "kind": "state", "condition": "primary", "mode": mode,
                "error_gain": 0.6, "effect_cosine": 0.8,
                "fiber_effect_ratio": 0.4, "ood_ratio": 0.4,
            },
            {
                "kind": "state", "condition": "full_swap_positive", "mode": mode,
                "error_gain": 0.8, "effect_cosine": 1.0,
                "fiber_effect_ratio": 1.0, "ood_ratio": 0.4,
            },
            {
                "kind": "state", "condition": "random_matched_subspace", "mode": mode,
                "error_gain": 0.1, "effect_cosine": 0.0,
                "fiber_effect_ratio": 0.2, "ood_ratio": 0.4,
            },
            {
                "kind": "fiber", "condition": "primary", "mode": mode,
                "error_gain": 0.0, "effect_cosine": 0.0,
                "fiber_effect_ratio": 0.5, "ood_ratio": 0.4,
            },
        ])
    return rows


def test_causal_summary_applies_all_frozen_thresholds():
    summary = summarize_causal_rows(
        _causal_rows(), ["free", "contact"],
        minimum_retention=0.5,
        minimum_cosine=0.2,
        minimum_control_advantage=0.1,
        maximum_fiber_ratio=1.25,
        maximum_ood_rate=0.05,
    )
    assert summary["passed"]
    assert np.isclose(summary["mean_state_effect_retention"], 0.75)
    assert np.isclose(summary["mean_control_advantage"], 0.5)

    broken = _causal_rows()
    for row in broken:
        if row["kind"] in {"state", "fiber"} and row["condition"] == "primary":
            row["ood_ratio"] = 2.0
    failed = summarize_causal_rows(
        broken, ["free", "contact"],
        minimum_retention=0.5,
        minimum_cosine=0.2,
        minimum_control_advantage=0.1,
        maximum_fiber_ratio=1.25,
        maximum_ood_rate=0.05,
    )
    assert not failed["passed"]


def test_split_path_decision_never_revives_shared_claim():
    passed = derive_stage342_decision(
        Stage342Gates(True, True, True, True, True, True, False)
    )
    assert passed["passed"]
    assert passed["status"] == "jepa_only_causal_state_dino_not_calibration_recoverable"
    assert not passed["confirmation_eligible"]

    calibrated = derive_stage342_decision(
        Stage342Gates(True, True, True, True, True, True, True)
    )
    assert calibrated["status"] == "jepa_causal_state_dino_calibration_limited"

    insufficient = derive_stage342_decision(
        Stage342Gates(True, True, True, False, False, False, True)
    )
    assert insufficient["status"] == "jepa_response_state_insufficient"
    assert not insufficient["passed"]
