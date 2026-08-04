import numpy as np
import pytest

from cf_faithfulness.stage19_unseen_action_transfer import (
    TRANSFER_FAMILIES,
    unseen_action_bank,
    validate_stage18_subspace_arrays,
)


@pytest.mark.parametrize("family", TRANSFER_FAMILIES)
def test_unseen_action_banks_are_noop_plus_antithetic_pairs(family):
    actions = unseen_action_bank([1.0, 0.0], family)
    assert actions.shape == (13, 15, 2)
    assert actions.dtype == np.float32
    assert np.array_equal(actions[0], np.zeros((15, 2), dtype=np.float32))
    for index in range(1, 7):
        assert np.allclose(actions[index], -actions[index + 6], atol=1e-7)


def test_rotated_directions_are_stage18_angular_midpoints():
    actions = unseen_action_bank([1.0, 0.0], "rotated_direction")
    observed = np.arctan2(actions[1, 0, 1], actions[1, 0, 0])
    assert np.isclose(observed, np.pi / 12.0)
    assert np.allclose(np.linalg.norm(actions[1:, 0], axis=1), 0.12)


def test_new_magnitudes_are_exact_and_constant():
    low = unseen_action_bank([1.0, 0.0], "magnitude_0p08")
    high = unseen_action_bank([1.0, 0.0], "magnitude_0p16")
    assert np.allclose(np.linalg.norm(low[1:], axis=-1), 0.08)
    assert np.allclose(np.linalg.norm(high[1:], axis=-1), 0.16)


def test_temporal_profiles_equal_stage18_integrated_impulse():
    delayed = unseen_action_bank([1.0, 0.0], "delayed_equal_impulse")
    pulsed = unseen_action_bank([1.0, 0.0], "pulsed_equal_impulse")
    assert np.allclose(delayed[1].sum(axis=0), [1.8, 0.0])
    assert np.allclose(pulsed[1].sum(axis=0), [1.8, 0.0])
    assert np.array_equal(delayed[1, :5], np.zeros((5, 2), dtype=np.float32))
    assert np.array_equal(pulsed[1, 5:10], np.zeros((5, 2), dtype=np.float32))


def test_stage18_artifact_contract_accepts_correct_synthetic_payload():
    # A small ambient override keeps this CPU-only contract test cheap.
    identity = np.eye(8)
    arrays = {
        "primary_basis": identity,
        "shuffled_basis": identity,
        "channel_square_root": np.eye(400),
        "channel_inverse_square_root": np.eye(400),
        **{f"random_basis_{draw:02d}": identity for draw in range(4)},
    }
    result = validate_stage18_subspace_arrays(arrays, ambient=8, max_rank=8)
    assert result["validated"]


def test_stage18_artifact_contract_rejects_missing_control():
    with pytest.raises(ValueError, match="missing arrays"):
        validate_stage18_subspace_arrays({}, ambient=8, max_rank=8)
