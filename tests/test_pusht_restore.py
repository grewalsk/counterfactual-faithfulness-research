import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("gym_pusht")

from cf_faithfulness.pusht_restore import (  # noqa: E402
    assert_exact_reset_restoration,
    rollout_from_reset_state,
)


@pytest.fixture()
def env():
    instance = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="state",
        render_mode="rgb_array",
        max_episode_steps=100,
    )
    yield instance
    instance.close()


def test_reset_branch_is_bitwise_exact(env):
    state = np.array([256.0, 420.0, 256.0, 300.0, 0.0])
    actions = np.tile(np.array([256.0, 200.0]), (6, 1))
    result = assert_exact_reset_restoration(
        env, state, actions, repeats=4, seed=123, atol=0.0
    )
    assert result["bitwise_exact"] is True


def test_alternative_actions_cause_different_outcomes(env):
    state = np.array([256.0, 420.0, 256.0, 300.0, 0.0])
    upward = np.tile(np.array([256.0, 200.0]), (6, 1))
    rightward = np.tile(np.array([400.0, 400.0]), (6, 1))
    first = rollout_from_reset_state(env, state, upward, seed=123)
    second = rollout_from_reset_state(env, state, rightward, seed=123)
    assert not np.array_equal(first.endpoint, second.endpoint)
    assert np.sum(first.contacts) > 0

