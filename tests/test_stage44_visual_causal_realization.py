import numpy as np
import pytest

from cf_faithfulness.stage44_visual_causal_realization import (
    canonical_visual_tokens,
    counterfactual_realization_metrics,
    derive_stage44_decision,
    fit_channel_pca,
    local_support_geometry,
    macro_one_vs_rest_auroc,
    masked_effect_energy,
    project_visual_tokens,
    spatial_pyramid_summary,
)


def test_canonical_visual_tokens_accepts_flat_and_grid_layouts():
    flat = np.zeros((2, 256, 384), dtype=np.float32)
    grid = flat.reshape(2, 1, 16, 16, 384)
    assert canonical_visual_tokens(flat).shape == (2, 256, 384)
    assert canonical_visual_tokens(grid).shape == (2, 256, 384)
    with pytest.raises(ValueError):
        canonical_visual_tokens(np.zeros((2, 255, 384)))


def test_channel_pca_and_spatial_summary_are_frozen_geometry():
    rng = np.random.default_rng(44)
    training = rng.normal(size=(400, 12))
    artifact = fit_channel_pca(training, rank=4)
    fields = rng.normal(size=(3, 16, 12))
    projected = project_visual_tokens(fields, artifact)
    summary = spatial_pyramid_summary(projected, grid_size=4, bins=2)
    assert projected.shape == (3, 16, 4)
    assert summary.shape == (3, 2 * 2 * 4 + 4)
    np.testing.assert_allclose(
        artifact["components"] @ artifact["components"].T,
        np.eye(4), atol=1e-10,
    )


def test_local_support_geometry_detects_normal_not_tangent_escape():
    x = np.linspace(-2.0, 2.0, 80)
    reference = np.stack([x, 0.02 * np.sin(x), np.zeros_like(x)], axis=1)
    tangent_query = np.asarray([[0.25, 0.0, 0.0]])
    normal_query = np.asarray([[0.25, 0.0, 2.0]])
    tangent = local_support_geometry(reference, tangent_query, neighbors=12, tangent_rank=1)
    normal = local_support_geometry(reference, normal_query, neighbors=12, tangent_rank=1)
    assert normal["normal_distance"][0] > 20.0 * tangent["normal_distance"][0]


def test_counterfactual_metrics_reward_aligned_effects():
    target_left = np.asarray([[2.0, 0.0], [0.0, 3.0]])
    target_right = np.zeros_like(target_left)
    prediction = 0.8 * target_left
    metrics = counterfactual_realization_metrics(
        target_left, target_right, prediction, np.zeros_like(prediction)
    )
    assert metrics["median_cosine"] == pytest.approx(1.0)
    assert metrics["median_magnitude_ratio"] == pytest.approx(0.8)


def test_masked_effect_energy_uses_patch_axis():
    effect = np.zeros((2, 4, 3))
    effect[:, 1] = 2.0
    mask = np.zeros((2, 4), dtype=bool)
    mask[:, 1] = True
    np.testing.assert_allclose(masked_effect_energy(effect, mask), 1.0)


def test_macro_auroc_is_one_for_separable_patch_classes():
    labels = np.asarray([0, 0, 1, 1, 2, 2])
    scores = np.eye(3)[labels]
    assert macro_one_vs_rest_auroc(labels, scores) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "support,decoder,encoder,one_step,recursive,causal,expected",
    [
        (False, True, True, True, True, True, "event_support_not_certified"),
        (True, False, True, True, True, True, "official_decoder_contract_failure"),
        (True, True, False, True, True, True, "encoder_observability_insufficient"),
        (True, True, True, False, True, True, "one_step_predictor_failure"),
        (True, True, True, True, False, True, "recursive_predictor_failure"),
        (True, True, True, True, True, False, "causal_visual_content_missing"),
        (True, True, True, True, True, True, "visual_causal_state_adequate"),
    ],
)
def test_stage44_decision_tree_is_ordered_and_fail_closed(
    support, decoder, encoder, one_step, recursive, causal, expected
):
    decision = derive_stage44_decision(
        support_certified=support,
        decoder_contract_valid=decoder,
        encoder_observable=encoder,
        one_step_adequate=one_step,
        recursive_stable=recursive,
        causal_realization=causal,
    )
    assert decision.classification == expected
    assert decision.planning_audit_authorized is (expected == "visual_causal_state_adequate")
    assert decision.object_centric_encoder_authorized is (
        support and decoder and not encoder
    )
