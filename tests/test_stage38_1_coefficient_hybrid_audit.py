import numpy as np

from cf_faithfulness.stage38_cross_model_pscd import (
    fit_weighted_semigroup_predictive_state_closure,
)
from cf_faithfulness.stage38_1_coefficient_hybrid_audit import (
    TierAGates,
    coefficient_matched_outer_weight,
    event_classification_metrics,
    fit_event_factorized_pscd,
    hierarchical_relative_gain_interval,
    hierarchical_statistic_interval,
    leave_one_family_out_relative_gain,
    macro_contact_events,
    rollout_event_factorized_pscd,
    semigroup_component_diagnostics,
    sequential_decision,
    shuffled_event_labels,
)


def synthetic_hybrid(seed=381, sequences=12, steps=6):
    rng = np.random.default_rng(seed)
    actions = rng.normal(size=(sequences, steps, 3)).astype(np.float32)
    initial = rng.normal(scale=0.2, size=(sequences, 4)).astype(np.float32)
    targets = np.zeros((sequences, steps, 4), dtype=np.float32)
    events = np.zeros((sequences, steps), dtype=bool)
    state = initial.copy()
    for step in range(steps):
        events[:, step] = ((np.arange(sequences) + step) % 5) == 0
        state = np.column_stack([
            0.8 * state[:, 0] + 0.2 * actions[:, step, 0],
            0.7 * state[:, 1] - 0.2 * actions[:, step, 1],
            0.6 * state[:, 2] + 0.1 * actions[:, step, 2],
            0.75 * state[:, 3] + events[:, step].astype(float) * 0.4,
        ]).astype(np.float32)
        targets[:, step] = state
    mask = np.ones((sequences, steps), dtype=bool)
    groups = np.repeat(np.arange(sequences // 2), 2)
    return initial, actions, targets, mask, events, groups


def test_coefficient_matching_and_event_label_controls():
    assert coefficient_matched_outer_weight(2.0) == 0.9
    assert coefficient_matched_outer_weight(1.0) == 0.45
    source = np.asarray([["free", "post_contact", "contact"]])
    target = np.asarray([["pre_contact", "contact", "post_contact"]])
    events = macro_contact_events(source, target)
    assert events.tolist() == [[False, True, False]]
    tiled = np.tile(events, (4, 1))
    mask = np.ones_like(tiled)
    shuffled = shuffled_event_labels(tiled, mask, seed=38)
    assert np.sum(shuffled) == np.sum(tiled)


def test_component_diagnostics_expose_exact_effective_coefficients():
    initial, actions, targets, mask, _, _ = synthetic_hybrid()
    artifact = fit_weighted_semigroup_predictive_state_closure(
        initial, actions, targets, targets, mask,
        history_length=2, latent_dim=12, dynamics="single", epochs=4,
        learning_rate=3e-3, seed=38101, semigroup_horizons=[2, 4],
        semigroup_weight=2.0, semigroup_component_weights=[0.35, 0.20, 0.45],
        device="cpu",
    )
    diagnostics = semigroup_component_diagnostics(
        artifact, initial, actions, targets, targets, mask, device="cpu"
    )
    assert diagnostics["effective_coefficients"] == [0.7, 0.4, 0.9]
    assert diagnostics["registered_anchor_pairs"] > 0
    assert all(np.isfinite(list(diagnostics["raw_losses"].values())))
    assert all(value >= 0 for value in diagnostics["raw_gradient_norms"].values())


def test_hybrid_and_smooth_controls_are_capacity_matched_and_roll_out():
    initial, actions, targets, mask, events, groups = synthetic_hybrid()
    artifacts = {}
    for kind in ["hybrid", "smooth"]:
        artifacts[kind] = fit_event_factorized_pscd(
            initial, actions, targets, targets, mask, events, groups=groups,
            history_length=2, latent_dim=12, dynamics="single", epochs=5,
            learning_rate=3e-3, seed=38102, semigroup_horizons=[2, 4],
            semigroup_weight=0.5, transition_kind=kind, event_hidden=8,
            device="cpu",
        )
        result = rollout_event_factorized_pscd(
            artifacts[kind], initial, actions, targets, mask, device="cpu"
        )
        assert np.all(np.isfinite(result["physical"]))
        assert np.all(np.isfinite(result["event_probability"][result["evaluation_mask"]]))
    assert (
        artifacts["hybrid"]["config"]["trainable_parameters"]
        == artifacts["smooth"]["config"]["trainable_parameters"]
    )
    oracle = rollout_event_factorized_pscd(
        artifacts["hybrid"], initial, actions, targets, mask,
        oracle_events=events, device="cpu",
    )
    assert oracle["physical"].shape == targets.shape


def test_family_risk_training_path_is_finite():
    initial, actions, targets, mask, events, groups = synthetic_hybrid()
    artifact = fit_event_factorized_pscd(
        initial, actions, targets, targets, mask, events, groups=groups,
        history_length=2, latent_dim=12, dynamics="single", epochs=3,
        learning_rate=3e-3, seed=38103, semigroup_horizons=[2, 4],
        semigroup_weight=0.5, transition_kind="hybrid", event_hidden=8,
        risk_weight=0.10, risk_alpha=0.90, device="cpu",
    )
    assert np.isfinite(artifact["risk_tau_final"])
    assert np.isfinite(artifact["loss_components_final"]["family_risk"])


def test_cluster_inference_event_metrics_and_sequential_decision():
    primary = np.asarray([[0.5, 0.7, 0.4, 0.6], [0.4, 0.6, 0.3, 0.5]])
    comparator = primary + 0.2
    groups = np.asarray([0, 0, 1, 1])
    interval = hierarchical_relative_gain_interval(
        primary, comparator, groups, draws=100, seed=381
    )
    assert interval[0] > 0
    tail_interval = hierarchical_statistic_interval(
        primary, groups, statistic="cvar95", draws=100, seed=382
    )
    assert tail_interval[0] <= np.quantile(primary, 1.0) <= tail_interval[1]
    sensitivity = leave_one_family_out_relative_gain(primary, comparator, groups)
    assert set(sensitivity) == {"0", "1"}
    probability = np.asarray([[0.05, 0.2, 0.8, 0.95]])
    labels = np.asarray([[False, False, True, True]])
    metrics = event_classification_metrics(probability, labels, np.ones_like(labels))
    assert metrics["auroc"] == 1.0 and metrics["brier_skill"] > 0
    decision = sequential_decision(TierAGates(*(True,) * 5), passed_status="tier_a_promoted")
    assert decision["passed"] and decision["status"] == "tier_a_promoted"
