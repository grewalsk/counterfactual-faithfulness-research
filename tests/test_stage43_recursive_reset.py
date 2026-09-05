import numpy as np
import pytest

from cf_faithfulness.stage43_recursive_reset import (
    clip_row_norms,
    derive_stage43_decision,
    fixed_state_projection,
    fixed_sham_projection,
    lagged_state_features,
    passes_registered_reset_gates,
    preceding_physical_state,
    recursive_reset_rollout_numpy,
    reset_base_tensor,
    reset_risk_metrics,
    tensor_reset_design,
)


def test_clip_row_norms_enforces_trust_region_without_changing_small_rows():
    values = np.asarray([[3.0, 4.0], [0.3, 0.4], [0.0, 0.0]])
    clipped = clip_row_norms(values, 1.0)
    np.testing.assert_allclose(clipped[0], [0.6, 0.8])
    np.testing.assert_allclose(clipped[1:], values[1:])
    assert np.max(np.linalg.norm(clipped, axis=1)) <= 1.0 + 1e-12


def test_clip_row_norms_rejects_invalid_limit():
    with pytest.raises(ValueError):
        clip_row_norms(np.ones((2, 2)), 0.0)


def test_lagged_state_features_never_cross_sequence_rows():
    states = np.asarray([[[1.0], [2.0], [3.0]], [[10.0], [20.0], [30.0]]])
    mask = np.ones((2, 3), dtype=bool)
    result = lagged_state_features(states, mask, lags=3)
    np.testing.assert_allclose(result[0, 2], [3.0, 2.0, 1.0])
    np.testing.assert_allclose(result[1, 1], [20.0, 10.0, 10.0])


def test_preceding_physical_state_uses_initial_then_previous_truth():
    initial = np.asarray([[1.0, 2.0]])
    path = np.asarray([[[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
    result = preceding_physical_state(initial, path, np.ones((1, 3), dtype=bool))
    np.testing.assert_allclose(result[0], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


def test_nested_reset_base_widths_are_strictly_ordered():
    state = np.ones((2, 4, 3))
    action = np.ones((2, 4, 2))
    mask = np.ones((2, 4), dtype=bool)
    physical = np.ones((2, 4, 5))
    initial = np.ones((2, 5))
    current = reset_base_tensor(state, action, mask, representation="current")
    history = reset_base_tensor(
        state, action, mask, representation="history", history_lags=3
    )
    physical_oracle = reset_base_tensor(
        state, action, mask, representation="physical_oracle",
        initial_physical=initial, physical_path=physical,
    )
    assert current.shape[-1] == 5
    assert history.shape[-1] == 11
    assert physical_oracle.shape[-1] == 10


def test_tensor_design_contains_main_effects_and_products():
    base = np.asarray([[1.0, 2.0], [2.0, 4.0], [4.0, 8.0]])
    metadata = np.asarray([[0.0, 1.0], [1.0, 2.0], [1.0, 4.0]])
    bmean, bscale = np.mean(base, axis=0), np.std(base, axis=0, ddof=1)
    emean, escale = np.mean(metadata, axis=0), np.std(metadata, axis=0, ddof=1)
    oracle = tensor_reset_design(
        base, metadata, base_mean=bmean, base_scale=bscale,
        metadata_mean=emean, metadata_scale=escale,
    )
    assert oracle.shape == (3, 2 + 2 + 2 * 2)
    sham = fixed_sham_projection(2, 2, seed=43)
    controlled = tensor_reset_design(
        base, metadata, base_mean=bmean, base_scale=bscale,
        metadata_mean=emean, metadata_scale=escale,
        metadata_mode="sham", sham_projection=sham,
    )
    assert controlled.shape == oracle.shape
    assert not np.allclose(controlled, oracle)


def test_low_rank_tensor_projection_controls_design_width():
    rng = np.random.default_rng(4)
    base = rng.normal(size=(12, 10))
    metadata = rng.normal(size=(12, 6))
    projection = fixed_state_projection(10, 4, seed=9)
    design = tensor_reset_design(
        base, metadata,
        base_mean=np.mean(base, axis=0), base_scale=np.std(base, axis=0, ddof=1),
        metadata_mean=np.mean(metadata, axis=0),
        metadata_scale=np.std(metadata, axis=0, ddof=1),
        state_projection=projection,
    )
    assert design.shape == (12, 4 + 6 + 4 * 6)


def test_recursive_reset_changes_all_later_states_not_only_event_output():
    initial = np.asarray([[0.0]])
    actions = np.ones((1, 3, 1))
    metadata = np.zeros((1, 3, 1))
    metadata[0, 1, 0] = 1.0
    mask = np.ones((1, 3), dtype=bool)
    output = recursive_reset_rollout_numpy(
        initial, actions, metadata, mask,
        transition=lambda state, action: state + action,
        correction=lambda proposed, action, meta, history: np.full_like(proposed, 10.0),
    )
    np.testing.assert_allclose(output[0, :, 0], [1.0, 12.0, 13.0])


def test_registered_risk_gates_are_conjunctive():
    baseline = np.asarray([1.0, 2.0, 3.0, 4.0] * 16)
    candidate = 0.6 * baseline
    reentry = np.ones_like(baseline, dtype=bool)
    groups = np.repeat(np.arange(16), 4)
    metrics = reset_risk_metrics(candidate, baseline, reentry, groups)
    assert passes_registered_reset_gates(metrics)
    failed = dict(metrics, p95_relative_gain=0.09)
    assert not passes_registered_reset_gates(failed)


@pytest.mark.parametrize(
    "tensor,nonlinear,history,physical,expected",
    [
        (True, True, True, True, "recursive_tensor_reset_headroom_confirmed"),
        (False, True, True, True, "nonlinear_recursive_reset_required"),
        (False, False, True, True, "short_history_state_completion_required"),
        (False, False, False, True, "frozen_carrier_state_insufficient"),
        (False, False, False, False, "reset_hypothesis_not_supported"),
    ],
)
def test_stage43_decision_localizes_the_missing_ingredient(
    tensor, nonlinear, history, physical, expected
):
    decision = derive_stage43_decision(
        support_certified=True,
        current_tensor_headroom=tensor,
        current_nonlinear_headroom=nonlinear,
        history_headroom=history,
        physical_oracle_headroom=physical,
    )
    assert decision.classification == expected
    assert decision.learned_recursive_reset_authorized is (tensor or nonlinear)


def test_stage43_decision_fails_closed_without_event_support():
    decision = derive_stage43_decision(
        support_certified=False,
        current_tensor_headroom=True,
        current_nonlinear_headroom=True,
        history_headroom=True,
        physical_oracle_headroom=True,
    )
    assert decision.classification == "event_support_not_certified"
    assert not decision.passed
