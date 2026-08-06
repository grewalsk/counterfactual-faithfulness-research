import numpy as np

from cf_faithfulness.stage28_hybrid_control_area import (
    SIGNED_AREA_LEVELS,
    SCHEDULE_INVERSION_COUNTS,
    SCHEDULE_STRINGS,
    area_ablation_delta,
    area_action_bank,
    area_antisymmetric_component,
    area_energy_metrics,
    area_law_metrics,
    area_reversal_permutation,
    area_swap_delta,
    area_transfer_metrics,
    contact_regime,
    magnitude_center,
    model_physics_area_metrics,
    schedule_inversion_count,
    signed_control_area,
)


def test_frozen_schedules_are_symmetric_in_area():
    assert [schedule_inversion_count(value) for value in SCHEDULE_STRINGS] == list(
        SCHEDULE_INVERSION_COUNTS
    )
    assert list(SIGNED_AREA_LEVELS) == [25, 15, 5, -5, -15, -25]
    assert tuple(value[::-1] for value in SCHEDULE_STRINGS) == SCHEDULE_STRINGS[::-1]


def test_area_action_bank_controls_impulse_energy_and_duration():
    magnitudes = [0.08, 0.12, 0.16, 0.20]
    actions = area_action_bank([3.0, 4.0], magnitudes)
    assert actions.shape == (24, 15, 2)
    areas = np.asarray([signed_control_area(value) for value in actions]).reshape(4, 6)
    for magnitude_index in range(4):
        group = actions[magnitude_index * 6 : (magnitude_index + 1) * 6]
        assert np.allclose(np.sum(group, axis=1), np.sum(group[0], axis=0))
        assert np.allclose(np.sum(group**2, axis=(1, 2)), np.sum(group[0] ** 2))
        assert np.array_equal(np.sum(np.linalg.norm(group, axis=2) > 0, axis=1), [10] * 6)
        assert np.allclose(group[:, 10:], 0.0)
        assert np.allclose(areas[magnitude_index], -areas[magnitude_index, ::-1])
    assert np.allclose(areas[1] / areas[0], (0.12 / 0.08) ** 2)


def test_area_swap_and_ablation_algebra():
    rng = np.random.default_rng(28)
    values = rng.normal(size=(24, 3, 4))
    permutation = area_reversal_permutation(4)
    assert np.allclose(values + area_swap_delta(values, 4), values[permutation])
    antisymmetric = area_antisymmetric_component(values, 4)
    assert np.allclose(antisymmetric, -antisymmetric[permutation])
    basis, _ = np.linalg.qr(rng.normal(size=(12, 5)))
    ablated = values + area_ablation_delta(values, 4, basis)
    residual = area_antisymmetric_component(ablated, 4).reshape(24, -1)
    assert np.linalg.norm(residual @ basis) < 1e-10


def test_transfer_energy_and_contact_regimes():
    rng = np.random.default_rng(41)
    baseline = rng.normal(size=(24, 7))
    swapped = baseline[area_reversal_permutation(4)]
    transfer = area_transfer_metrics(baseline, swapped, 4)
    assert np.isclose(transfer["coefficient"], 1.0)
    erased = baseline - area_antisymmetric_component(baseline, 4)
    energy = area_energy_metrics(baseline, erased, 4)
    assert np.isclose(energy["energy_reduction"], 1.0)
    assert contact_regime(np.ones((4, 6), dtype=int)) == "persistent_contact"
    assert contact_regime(np.zeros((4, 6), dtype=int)) == "free"
    mixed = np.zeros((4, 6), dtype=int)
    mixed[-1, :3] = 1
    assert contact_regime(mixed) == "boundary_switching"


def test_area_law_recovers_second_order_scaling_and_model_alignment():
    magnitudes = np.asarray([0.08, 0.12, 0.16, 0.20])
    actions = area_action_bank([1.0, 0.0], magnitudes)
    areas = np.asarray([signed_control_area(value) for value in actions])
    direction = np.asarray([2.0, -1.0, 0.5])
    intercepts = np.repeat(np.arange(4, dtype=float), 6)[:, None]
    truth = intercepts + areas[:, None] * direction[None]
    metrics = area_law_metrics(truth, actions, magnitudes)
    assert np.isclose(metrics["area_r_squared"], 1.0)
    assert np.isclose(metrics["mean_slope_direction_cosine"], 1.0)
    assert np.isclose(metrics["magnitude_exponent"], 2.0)
    assert metrics["epsilon_squared_collapse_error"] < 1e-6
    predicted = 0.4 * magnitude_center(truth, 4) + intercepts
    alignment = model_physics_area_metrics(predicted, truth, 4)
    assert np.isclose(alignment["coefficient"], 0.4)
    assert np.isclose(alignment["cosine"], 1.0)

