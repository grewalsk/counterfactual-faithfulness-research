import numpy as np
import pytest

from cf_faithfulness.stage33_interventional_abstraction import (
    AffineBilinearOperator,
    WhitenedProcrustesMap,
    clustered_bootstrap_interval,
    compose_affine_bilinear,
    conjugate_affine_bilinear_operator,
    derive_decision,
    effective_rank,
    epsilon_predictively_related,
    fit_affine_bilinear_operator,
    fit_grouped_ridge,
    fit_whitened_similarity,
    holm_adjust,
    interchange_metrics,
    operator_intertwining_metrics,
    planning_decision_metrics,
    predict_affine_bilinear,
    reachability_observability_diagnostics,
    select_stable_rank,
    signature_pseudometric,
)


def _operator(mode=0):
    if mode == 0:
        return {
            "c": np.array([0.08, -0.03, 0.02]),
            "A": np.array(
                [[0.86, 0.08, 0.00], [0.00, 0.78, 0.12], [0.04, 0.00, 0.72]]
            ),
            "B": np.array([[0.30, 0.00], [0.00, 0.24], [0.10, -0.08]]),
            "N": np.array(
                [
                    [[0.03, 0.00, 0.00], [0.00, -0.02, 0.00], [0.01, 0.00, 0.00]],
                    [[0.00, 0.04, 0.00], [0.00, 0.00, 0.00], [0.00, 0.00, -0.03]],
                ]
            ),
        }
    return {
        "c": np.array([-0.02, 0.05, -0.01]),
        "A": np.array(
            [[0.68, -0.12, 0.06], [0.10, 0.70, 0.00], [0.00, 0.14, 0.80]]
        ),
        "B": np.array([[0.12, 0.04], [-0.04, 0.31], [0.18, 0.02]]),
        "N": np.array(
            [
                [[-0.02, 0.00, 0.03], [0.01, 0.02, 0.00], [0.00, 0.00, 0.01]],
                [[0.00, 0.00, 0.00], [0.03, -0.01, 0.00], [0.00, 0.02, 0.02]],
            ]
        ),
    }


def _typed(operator):
    return AffineBilinearOperator(
        c=operator["c"], A=operator["A"], B=operator["B"], N=operator["N"]
    )


def _as_dict(operator):
    return {"c": operator.c, "A": operator.A, "B": operator.B, "N": operator.N}


def _transport(matrix, offset):
    dimension = len(offset)
    return WhitenedProcrustesMap(
        matrix=matrix,
        offset=offset,
        source_mean=np.zeros(dimension),
        target_mean=offset,
        condition_number=float(np.linalg.cond(matrix)),
        source_covariance_condition=1.0,
        target_covariance_condition=1.0,
        calibration_relative_rmse=0.0,
    )


def test_epsilon_signature_relation_is_not_transitive():
    first = np.array([0.0])
    middle = np.array([0.6])
    last = np.array([1.2])
    assert epsilon_predictively_related(first, middle, 0.7, bound=10.0)
    assert epsilon_predictively_related(middle, last, 0.7, bound=10.0)
    assert not epsilon_predictively_related(first, last, 0.7, bound=10.0)
    assert signature_pseudometric(first, first) == 0.0
    assert signature_pseudometric(first, last, bound=1.0) == 1.0


def test_deterministic_structured_rank_recovery():
    rng = np.random.default_rng(33)
    group_count, block_length = 12, 8
    scores = rng.normal(size=(group_count * block_length, 2))
    loading = np.array(
        [[1.0, 0.0, 0.7, -0.4, 0.2, 0.0], [0.0, 1.0, 0.2, 0.8, -0.5, 0.3]]
    )
    signatures = scores @ loading + 1e-3 * rng.normal(size=(len(scores), 6))
    groups = np.repeat(np.arange(group_count), block_length)
    kwargs = dict(
        max_rank=5,
        n_bootstrap=100,
        n_permutations=100,
        stability_floor=0.75,
        null_quantile=0.95,
        seed=9,
    )
    first = select_stable_rank(signatures, groups, **kwargs)
    second = select_stable_rank(signatures, groups, **kwargs)
    assert first["rank"] == 2
    assert first["rank_ci95"] == (2, 2)
    assert np.array_equal(first["rank_draws"], second["rank_draws"])
    assert np.allclose(first["null_singular_values"], second["null_singular_values"])
    assert 1.0 < effective_rank(signatures, center=True) <= 2.1


def test_grouped_ridge_and_mode_conditioned_operator_recover_exact_systems():
    rng = np.random.default_rng(34)
    rows_per_mode = 180
    states = rng.normal(size=(2 * rows_per_mode, 3))
    actions = rng.uniform(-0.8, 0.8, size=(len(states), 2))
    modes = np.repeat([0, 1], rows_per_mode)
    next_states = np.empty_like(states)
    for mode in (0, 1):
        selected = modes == mode
        next_states[selected] = predict_affine_bilinear(
            _operator(mode), states[selected], actions[selected]
        )
    fitted = fit_affine_bilinear_operator(
        states, actions, next_states, ridge=0.0, modes=modes
    )
    assert fitted["global_operator"]["n_samples"] == len(states)
    for mode in (0, 1):
        selected = modes == mode
        prediction = predict_affine_bilinear(
            fitted["operators"][mode], states[selected], actions[selected]
        )
        assert np.max(np.abs(prediction - next_states[selected])) < 1e-11

    outcome = 0.4 + states @ np.array([0.7, -0.3, 0.2])
    groups = np.repeat(np.arange(12), len(states) // 12)
    ridge = fit_grouped_ridge(
        states, outcome, groups, penalties=(0.0, 1e-4), folds=4, seed=5
    )
    assert ridge["penalty"] == 0.0
    assert np.max(np.abs(ridge["oof_prediction"] - outcome)) < 1e-11


def test_exact_similar_hybrid_systems_pass_one_fixed_map_and_composition():
    rng = np.random.default_rng(35)
    matrix = np.array([[1.25, 0.12, 0.00], [0.08, 0.82, 0.10], [0.00, 0.16, 1.08]])
    offset = np.array([0.25, -0.18, 0.07])
    source_calibration = rng.normal(size=(160, 3))
    target_calibration = source_calibration @ matrix.T + offset
    learned_map = fit_whitened_similarity(source_calibration, target_calibration)
    assert np.allclose(learned_map["matrix"], matrix, atol=1e-8)
    assert learned_map["calibration_rmse"] < 1e-9

    transport = _transport(matrix, offset)
    source = {
        "operators": {0: _operator(0), 1: _operator(1)},
        "global_operator": _operator(0),
    }
    target = {
        "operators": {
            mode: _as_dict(
                conjugate_affine_bilinear_operator(_typed(_operator(mode)), transport)
            )
            for mode in (0, 1)
        },
    }
    target["global_operator"] = target["operators"][0]
    for mode in (0, 1):
        metrics = operator_intertwining_metrics(
            source["operators"][mode], target["operators"][mode], learned_map
        )
        assert metrics["aggregate_relative_error"] < 1e-8

    initial_source = rng.normal(size=(9, 3))
    initial_target = initial_source @ matrix.T + offset
    action_words = rng.uniform(-0.5, 0.5, size=(9, 5, 2))
    mode_words = np.tile(np.array([0, 0, 1, 1, 0]), (9, 1))
    source_final = compose_affine_bilinear(
        source, initial_source, action_words, mode_words
    )
    target_final = compose_affine_bilinear(
        target, initial_target, action_words, mode_words
    )
    assert np.allclose(target_final, source_final @ matrix.T + offset, atol=1e-10)

    exact_effect = target_final - initial_target
    interchange = interchange_metrics(
        exact_effect,
        exact_effect,
        baseline_errors=np.full(len(exact_effect), 1.0),
        patched_errors=np.zeros(len(exact_effect)),
    )
    assert np.isclose(interchange["effect_cosine"], 1.0)
    assert interchange["relative_effect_error"] < 1e-12
    assert np.isclose(interchange["error_gain"], 1.0)


def test_shared_decodability_and_conjugacy_do_not_imply_internal_transport():
    """Concrete false-positive: output charts align, but the patch is causally inert."""

    rng = np.random.default_rng(36)
    source_states = rng.normal(size=(120, 3))
    matrix = np.array([[1.1, 0.2, 0.0], [0.0, 0.9, 0.1], [0.1, 0.0, 1.0]])
    offset = np.array([0.2, -0.1, 0.05])
    target_states = source_states @ matrix.T + offset
    physical_output = source_states @ np.array([0.8, -0.4, 0.3])
    groups = np.repeat(np.arange(12), 10)
    source_decoder = fit_grouped_ridge(
        source_states, physical_output, groups, penalties=(0.0,), folds=4, seed=1
    )
    target_decoder = fit_grouped_ridge(
        target_states, physical_output, groups, penalties=(0.0,), folds=4, seed=1
    )
    assert source_decoder["oof_mse"] < 1e-20
    assert target_decoder["oof_mse"] < 1e-20

    learned_map = fit_whitened_similarity(source_states, target_states)
    transport = _transport(matrix, offset)
    target_operator = _as_dict(
        conjugate_affine_bilinear_operator(_typed(_operator(0)), transport)
    )
    conjugacy = operator_intertwining_metrics(
        _operator(0), target_operator, learned_map
    )
    assert conjugacy["aggregate_relative_error"] < 1e-8

    intended_effect = np.tile(np.array([0.5, -0.2, 0.1]), (20, 1))
    causally_inert_patch = np.zeros_like(intended_effect)
    interchange = interchange_metrics(causally_inert_patch, intended_effect)
    assert interchange["effect_cosine"] == 0.0
    assert np.isclose(interchange["relative_effect_error"], 1.0)
    gate = derive_decision(
        {
            "rank": True,
            "hybrid": True,
            "fixed_map": True,
            "conjugacy": True,
            "interchange": False,
            "planning": True,
            "controls": True,
            "families": True,
        }
    )
    assert gate["status"] == "partial_pass"
    assert not gate["passed"]


def test_similarity_condition_check_rejects_rank_deficient_calibration():
    coordinate = np.linspace(-1.0, 1.0, 30)
    rank_one = np.column_stack([coordinate, 2.0 * coordinate, -coordinate])
    with pytest.raises(ValueError, match="rank deficient"):
        fit_whitened_similarity(rank_one, rank_one)


def test_reachability_observability_stats_multiplicity_planning_and_gates():
    A = np.array([[1.0, 1.0], [0.0, 1.0]])
    B = np.array([[0.0], [1.0]])
    C = np.array([[1.0, 0.0]])
    diagnostics = reachability_observability_diagnostics(A, B, C)
    assert diagnostics["reachability_rank"] == 2
    assert diagnostics["observability_rank"] == 2

    values = np.array([0.0, 0.2, 0.8, 1.0, 1.1, 0.9])
    groups = np.repeat(np.arange(3), 2)
    first = clustered_bootstrap_interval(values, groups, draws=200, seed=7)
    second = clustered_bootstrap_interval(values, groups, draws=200, seed=7)
    assert first == second
    assert first[0] <= np.mean(values) <= first[1]

    holm = holm_adjust([0.01, 0.04, 0.03], alpha=0.05)
    assert np.allclose(holm["adjusted_pvalues"], [0.03, 0.06, 0.06])
    assert np.array_equal(holm["reject"], [True, False, False])

    oracle = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 0.5]])
    planning = planning_decision_metrics(oracle, oracle)
    assert planning["top1_accuracy"] == 1.0
    assert planning["mean_normalized_regret"] == 0.0

    all_pass = derive_decision(
        {
            "rank": True,
            "hybrid": True,
            "fixed_map": True,
            "conjugacy": True,
            "interchange": True,
            "planning": True,
            "controls": True,
            "families": True,
        },
        run_mode="pilot",
    )
    assert all_pass["status"] == "pass" and all_pass["level"] == 6
    smoke = derive_decision(all_pass["checks"], run_mode="smoke")
    assert smoke["status"] == "fail"
