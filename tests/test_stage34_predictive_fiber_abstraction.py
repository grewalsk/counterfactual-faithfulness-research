import numpy as np

from cf_faithfulness.stage34_predictive_fiber_abstraction import (
    Stage34Gates,
    action_contrast_signature,
    commutativity_metrics,
    derive_stage34_decision,
    fit_response_chart,
    fit_supervised_subspace,
    intervention_ood_ratio,
    matched_fiber_pairs,
    nested_predictive_sufficiency,
    response_coordinates,
    split_carrier_delta,
)


def _paths(offset=0.0):
    names = ["L", "R", "LR", "RL", "zero1", "zero2"]
    lengths = np.array([1, 1, 2, 2, 1, 2])
    values = np.zeros((len(names), 2, 2), dtype=np.float64) + float(offset)
    values[0, 0] += [1.0, 0.0]
    values[1, 0] += [0.0, 2.0]
    values[2, :2] += [[1.0, 0.0], [1.0, 3.0]]
    values[3, :2] += [[0.0, 2.0], [4.0, 2.0]]
    return values, names, lengths


def test_action_contrast_removes_static_offset_and_keeps_order_effect():
    first, names, lengths = _paths(0.0)
    shifted, _, _ = _paths(17.0)
    kwargs = dict(
        word_names=names,
        word_lengths=lengths,
        response_words=["L", "R", "LR", "RL"],
        zero_word_by_length={1: "zero1", 2: "zero2"},
        order_pairs=[("LR", "RL")],
    )
    signature = action_contrast_signature(first, **kwargs)
    shifted_signature = action_contrast_signature(shifted, **kwargs)
    assert np.array_equal(signature, shifted_signature)
    assert np.array_equal(signature[-2:], np.array([-3.0, 1.0]))


def test_response_chart_is_frozen_and_projects_consistently():
    rng = np.random.default_rng(3401)
    latent = rng.normal(size=(40, 2))
    signatures = latent @ rng.normal(size=(2, 12))
    chart = fit_response_chart(signatures, rank=2)
    coordinates = response_coordinates(signatures, chart)
    assert coordinates.shape == (40, 2)
    assert np.allclose(response_coordinates(signatures[3], chart), coordinates[3])


def test_nested_sufficiency_detects_predictive_residual_information():
    rng = np.random.default_rng(3402)
    groups = np.repeat(np.arange(20), 8)
    state = rng.normal(size=(len(groups), 2))
    actions = rng.normal(size=(len(groups), 2))
    residual = rng.normal(size=(len(groups), 3))
    target = np.column_stack([
        state[:, 0] + actions[:, 0] + 2.0 * residual[:, 0],
        state[:, 1] - actions[:, 1] - residual[:, 1],
    ])
    result = nested_predictive_sufficiency(
        state, actions, residual, target, groups, penalties=(0.0, 1e-4), seed=7
    )
    assert result["residual_relative_improvement"] > 0.9

    sufficient_target = np.column_stack([
        state[:, 0] + actions[:, 0], state[:, 1] - actions[:, 1]
    ])
    sufficient = nested_predictive_sufficiency(
        state, actions, residual, sufficient_target, groups,
        penalties=(0.0, 1e-4), seed=7,
    )
    assert sufficient["residual_relative_improvement"] < 1e-8


def test_supervised_subspace_splits_aligned_and_fiber_deltas():
    rng = np.random.default_rng(3403)
    carriers = rng.normal(size=(120, 10))
    coordinates = carriers[:, :2] @ np.array([[1.0, 0.2], [-0.1, 0.8]])
    subspace = fit_supervised_subspace(carriers, coordinates, rank=2, ridge=1e-8)
    aligned, residual = split_carrier_delta(np.arange(10, dtype=float), subspace)
    assert np.allclose(aligned + residual, np.arange(10, dtype=float))
    white_residual = residual / subspace["scale"]
    assert np.linalg.norm(subspace["basis"].T @ white_residual) < 1e-10


def test_fiber_matching_is_mode_and_trajectory_disjoint():
    coordinates = np.array([[0.0], [0.1], [4.0], [4.1], [0.2], [4.2]])
    residual = np.array([[0.0], [5.0], [0.0], [5.0], [9.0], [9.0]])
    modes = np.array(["free", "free", "contact", "contact", "free", "contact"])
    trajectories = np.arange(len(coordinates))
    pairs = matched_fiber_pairs(coordinates, residual, modes, trajectories, kind="fiber")
    assert np.all(modes[pairs[:, 0]] == modes[pairs[:, 1]])
    assert np.all(trajectories[pairs[:, 0]] != trajectories[pairs[:, 1]])


def test_ood_commutativity_and_sequential_decision():
    natural = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]])
    ratios = intervention_ood_ratio(np.array([[0.05, 0.05], [10.0, 10.0]]), natural)
    assert ratios[0] < 1.0 < ratios[1]

    reference = np.array([[1.0, 0.0], [0.0, 1.0]])
    metrics = commutativity_metrics(reference, reference, reference_error=0.1)
    assert metrics["mean_relative_error"] == 0.0
    assert metrics["mean_cosine"] == 1.0

    first_fail = derive_stage34_decision(
        Stage34Gates(False, True, True, True, True, True)
    )
    assert first_fail["status"] == "shared_static_state_geometry_only"
    passed = derive_stage34_decision(Stage34Gates(True, True, True, True, True, True))
    assert passed["passed"]
    assert passed["level"] == 6
