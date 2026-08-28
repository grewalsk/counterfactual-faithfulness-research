import numpy as np

from cf_faithfulness.stage41_causal_event_reset import (
    CAUSAL_VARIANTS,
    causal_design_matrix,
    causal_effect_metrics,
    derive_stage41_decision,
    derive_stage41_panel_decision,
    deterministic_permutation,
    fit_ridge,
    fixed_sham_projection,
    mean_scale,
    ridge_predict,
    select_ridge_penalty,
    storage_equivalent,
    upper_tail_mean,
)


def test_oracle_ladder_keeps_equal_width_and_reveals_registered_prefix():
    rng = np.random.default_rng(41)
    base = rng.normal(size=(32, 7))
    metadata = rng.normal(size=(32, 6))
    base_mean, base_scale = mean_scale(base)
    metadata_mean, metadata_scale = mean_scale(metadata)
    projection = fixed_sham_projection(7, 6, 4101)
    designs = {}
    for variant in CAUSAL_VARIANTS:
        permutation = deterministic_permutation(len(base), 99) if variant == "shuffled_event" else None
        designs[variant] = causal_design_matrix(
            base, metadata, variant=variant,
            base_mean=base_mean, base_scale=base_scale,
            metadata_mean=metadata_mean, metadata_scale=metadata_scale,
            sham_projection=projection, permutation=permutation,
        )
    assert {value.shape for value in designs.values()} == {(32, 13)}
    normalized = (metadata - metadata_mean) / metadata_scale
    np.testing.assert_allclose(designs["oracle_event"][:, 7], normalized[:, 0])
    np.testing.assert_allclose(designs["oracle_time"][:, 7:9], normalized[:, :2])
    np.testing.assert_allclose(designs["oracle_geometry"][:, 7:11], normalized[:, :4])
    np.testing.assert_allclose(designs["oracle_reset_ceiling"][:, 7:], normalized)


def test_ridge_selection_and_prediction_recover_linear_target():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(200, 5))
    weight = rng.normal(size=(5, 3))
    y = x @ weight + np.array([0.5, -0.2, 1.1])
    selection = select_ridge_penalty(
        x[:120], y[:120], x[120:], y[120:], [0.0, 0.1, 10.0]
    )
    assert selection["selected_penalty"] == 0.0
    artifact = fit_ridge(x, y, selection["selected_penalty"])
    np.testing.assert_allclose(ridge_predict(artifact, x), y, atol=1e-10)


def test_storage_equivalence_respects_the_declared_float32_boundary():
    reference = np.array([1.0 / 3.0, np.pi, -1e-4], dtype=np.float64)
    stored = reference.astype(np.float32).astype(np.float64)
    assert not np.array_equal(reference, stored)
    assert storage_equivalent(reference, stored, storage_dtype=np.float32)
    changed = stored.copy()
    changed[1] = np.nextafter(changed[1].astype(np.float32), np.float32(np.inf))
    assert not storage_equivalent(reference, changed, storage_dtype=np.float32)


def test_causal_effect_score_rewards_correct_effect_and_tail_is_registered():
    truth = np.array([[1.0, 0.0], [0.0, 2.0], [0.5, -0.5]])
    mask = np.array([True, True, False])
    score = causal_effect_metrics(truth, truth, np.ones(2), mask)
    assert score["rows"] == 2
    assert np.isclose(score["relative_gain_over_zero"], 1.0)
    assert np.isclose(score["mean_cosine"], 1.0)
    assert upper_tail_mean([1, 2, 3, 4], 0.25) == 4.0


def test_global_decision_is_cross_model_conjunctive_and_noncausal():
    passed = derive_stage41_panel_decision(
        "jepa", all_seed_tail_improvement=True, all_seed_p95_improvement=True,
        all_seed_mean_noninferiority=True, all_seed_control_dominance=True,
        all_seed_causal_alignment=True,
    )
    failed = derive_stage41_panel_decision(
        "dino", all_seed_tail_improvement=True, all_seed_p95_improvement=True,
        all_seed_mean_noninferiority=True, all_seed_control_dominance=False,
        all_seed_causal_alignment=True,
    )
    result = derive_stage41_decision({"jepa": passed, "dino": failed})
    assert result["passed"] is False
    assert result["causal_claim_authorized"] is False
    assert result["learned_deployment_claim_authorized"] is False
