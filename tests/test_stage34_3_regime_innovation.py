import numpy as np

from cf_faithfulness.stage34_3_regime_innovation import (
    Stage343Gates,
    aggregate_relative_gain,
    clustered_relative_gain_interval,
    derive_stage343_decision,
    fit_candidate_model,
    fit_innovation_basis,
    predict_candidate_model,
    select_simplest_candidate,
    transform_innovation,
    within_group_permuted_labels,
)


def test_supervised_innovation_finds_predictive_carrier_direction():
    rng = np.random.default_rng(34301)
    carrier = rng.normal(size=(400, 8))
    target_residual = np.column_stack([
        3.0 * carrier[:, 3] + 0.03 * rng.normal(size=400),
        -2.0 * carrier[:, 3] + 0.03 * rng.normal(size=400),
    ])
    model = fit_innovation_basis(carrier, target_residual, rank=1)
    innovation = transform_innovation(model, carrier)[:, 0]
    assert abs(np.corrcoef(innovation, carrier[:, 3])[0, 1]) > 0.97
    assert model["rank"] == 1


def test_mode_specific_candidate_captures_switched_dynamics():
    rng = np.random.default_rng(34302)
    rows = 320
    state = rng.normal(size=(rows, 5))
    action = rng.normal(size=(rows, 2))
    carrier = rng.normal(size=(rows, 6))
    modes = np.where(np.arange(rows) % 2, "contact", "free")
    sign = np.where(modes == "free", 1.0, -1.0)
    target = np.column_stack([
        sign * state[:, 0] + 0.2 * action[:, 0],
        sign * state[:, 1] - 0.1 * action[:, 1],
    ])
    train = np.arange(rows) % 4 != 0
    test = ~train
    model = fit_candidate_model(
        state[train], action[train], carrier[train], target[train], modes[train],
        state_rank=2, innovation_rank=0, regime_specific=True,
        mode_labels=["free", "contact"], width=128, penalty=1e-4, seed=9,
    )
    prediction = predict_candidate_model(
        model, state[test], action[test], carrier[test], modes[test]
    )
    assert np.mean((prediction - target[test]) ** 2) < 0.03


def test_candidate_selection_uses_complexity_tolerance():
    rows = [
        {
            "state_rank": 5, "innovation_rank": 3, "regime_specific": True,
            "penalty": 0.01, "oof_mse": 0.100,
        },
        {
            "state_rank": 4, "innovation_rank": 1, "regime_specific": False,
            "penalty": 0.1, "oof_mse": 0.101,
        },
        {
            "state_rank": 4, "innovation_rank": 0, "regime_specific": False,
            "penalty": 0.1, "oof_mse": 0.104,
        },
    ]
    selected = select_simplest_candidate(rows, relative_tolerance=0.02)
    assert selected["state_rank"] == 4
    assert selected["innovation_rank"] == 1
    assert selected["best_oof_mse"] == 0.1


def test_aggregate_gain_and_cluster_interval_are_stable():
    primary = np.full(80, 0.8)
    comparator = np.full(80, 1.0)
    groups = np.repeat(np.arange(10), 8)
    assert np.isclose(aggregate_relative_gain(primary, comparator), 0.2)
    interval = clustered_relative_gain_interval(
        primary, comparator, groups, draws=200, seed=3
    )
    assert np.allclose(interval, [0.2, 0.2])


def test_mode_permutation_preserves_each_group_multiset():
    modes = np.tile(np.repeat(["free", "pre", "contact", "post"], 2), 3)
    groups = np.repeat(np.arange(3), 8)
    units = np.tile(np.repeat(np.arange(4), 2), 3) + np.repeat(np.arange(3) * 10, 8)
    permuted = within_group_permuted_labels(modes, groups, units, seed=343)
    for group in np.unique(groups):
        observed = sorted(set(permuted[groups == group]))
        assert observed == ["contact", "free", "post", "pre"]
    assert np.any(permuted != modes)


def test_stage343_decision_is_never_confirmatory_or_causal():
    passed = derive_stage343_decision(Stage343Gates(*(True,) * 7))
    assert passed["status"] == "bounded_jepa_state_candidate_repaired"
    assert passed["passed"]
    assert not passed["confirmation_eligible"]
    assert not passed["causal_evidence"]

    failed = derive_stage343_decision(
        Stage343Gates(True, True, True, True, False, False, True)
    )
    assert failed["status"] == "selected_state_still_carrier_incomplete"
    assert not failed["passed"]
