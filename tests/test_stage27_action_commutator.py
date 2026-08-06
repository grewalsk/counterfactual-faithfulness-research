import numpy as np

from cf_faithfulness.stage27_action_commutator import (
    commutator_alignment_metrics,
    commutator_contrasts,
    commutator_norms,
    ordered_pulse_bank,
    pair_contact_masks,
    pair_swap_permutation,
    paired_ablation_delta,
    paired_antisymmetric_component,
    paired_energy_metrics,
    paired_swap_delta,
    paired_transfer_metrics,
)


def test_ordered_pulses_preserve_pairwise_controls():
    actions = ordered_pulse_bank([3.0, 4.0])
    assert actions.shape == (12, 15, 2)
    for pair in range(6):
        first, second = actions[2 * pair : 2 * pair + 2]
        assert np.allclose(np.sum(first, axis=0), np.sum(second, axis=0), atol=1e-7)
        assert np.isclose(np.sum(first**2), np.sum(second**2), atol=1e-7)
        assert np.array_equal(first[:5], second[5:10])
        assert np.array_equal(first[5:10], second[:5])
        assert np.array_equal(first[10:], np.zeros((5, 2), dtype=np.float32))


def test_pair_swap_and_antisymmetric_algebra():
    rng = np.random.default_rng(27)
    values = rng.normal(size=(12, 8, 5))
    permutation = pair_swap_permutation(6)
    swapped = values + paired_swap_delta(values)
    assert np.allclose(swapped, values[permutation], atol=1e-12)
    antisymmetric = paired_antisymmetric_component(values)
    assert np.allclose(antisymmetric[0::2], -antisymmetric[1::2], atol=1e-12)


def test_pair_specific_ablation_preserves_pair_means():
    rng = np.random.default_rng(31)
    values = rng.normal(size=(12, 3, 4))
    basis, _ = np.linalg.qr(rng.normal(size=(12, 5)))
    ablated = values + paired_ablation_delta(values, basis)
    assert np.allclose(
        values.reshape(6, 2, -1).mean(axis=1),
        ablated.reshape(6, 2, -1).mean(axis=1),
        atol=1e-12,
    )
    residual = paired_antisymmetric_component(ablated).reshape(12, -1)
    assert np.linalg.norm(residual @ basis) < 1e-10


def test_scoped_transfer_and_energy_metrics():
    rng = np.random.default_rng(37)
    baseline = rng.normal(size=(12, 9))
    swapped = baseline[pair_swap_permutation(6)]
    mask = np.asarray([True, False, True, False, True, False])
    transfer = paired_transfer_metrics(baseline, swapped, mask)
    assert np.isclose(transfer["coefficient"], 1.0)
    assert np.isclose(transfer["cosine"], 1.0)
    erased = baseline - paired_antisymmetric_component(baseline)
    energy = paired_energy_metrics(baseline, erased, mask)
    assert np.isclose(energy["energy_reduction"], 1.0)


def test_contact_masks_and_physical_alignment():
    masks = pair_contact_masks([2, 3, 0, 4, 0, 0, 1, 0, 0, 0, 5, 7])
    assert np.array_equal(masks["both_contact"], [True, False, False, False, False, True])
    assert np.array_equal(masks["one_contact"], [False, True, False, True, False, False])
    assert np.array_equal(masks["free"], [False, False, True, False, True, False])

    truth = np.arange(48, dtype=np.float64).reshape(12, 4)
    predicted = 0.4 * truth
    contrasts = commutator_contrasts(truth)
    assert contrasts.shape == (6, 4)
    assert np.allclose(commutator_norms(truth), np.linalg.norm(contrasts, axis=1))
    metrics = commutator_alignment_metrics(predicted, truth, masks["contact"])
    assert np.isclose(metrics["coefficient"], 0.4)
    assert np.isclose(metrics["cosine"], 1.0)
