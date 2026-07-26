"""Exact reset-state branching checks for ``gym-pusht``.

The tested protocol reconstructs a fresh Pymunk space at every branch point and
sets the same canonical state before applying each alternative action sequence.
It intentionally does not claim that the package's five-number public reset
state is sufficient to restore an arbitrary in-contact, nonzero-velocity state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PushTRestoreResult:
    initial_observation: FloatArray
    endpoint: FloatArray
    rewards: FloatArray
    contacts: NDArray[np.int64]
    terminated: bool
    truncated: bool


def rollout_from_reset_state(
    env: Any,
    reset_state: ArrayLike,
    actions: Iterable[ArrayLike],
    *,
    seed: int = 0,
) -> PushTRestoreResult:
    """Rebuild the simulator, set a canonical state, and execute actions."""

    state = np.asarray(reset_state, dtype=np.float64)
    if state.shape != (5,):
        raise ValueError(f"reset_state must have shape (5,), got {state.shape}")
    initial, _ = env.reset(seed=seed, options={"reset_to_state": state.copy()})
    rewards: list[float] = []
    contacts: list[int] = []
    endpoint = np.asarray(initial, dtype=np.float64)
    terminated = False
    truncated = False
    for action in actions:
        endpoint, reward, terminated, truncated, info = env.step(
            np.asarray(action, dtype=np.float32)
        )
        rewards.append(float(reward))
        contacts.append(int(info.get("n_contacts", 0)))
        if terminated or truncated:
            break
    return PushTRestoreResult(
        initial_observation=np.asarray(initial, dtype=np.float64),
        endpoint=np.asarray(endpoint, dtype=np.float64),
        rewards=np.asarray(rewards, dtype=np.float64),
        contacts=np.asarray(contacts, dtype=np.int64),
        terminated=bool(terminated),
        truncated=bool(truncated),
    )


def assert_exact_reset_restoration(
    env: Any,
    reset_state: ArrayLike,
    actions: Iterable[ArrayLike],
    *,
    repeats: int = 3,
    seed: int = 0,
    atol: float = 0.0,
) -> dict[str, float | int | bool]:
    """Repeat one branch and fail unless observations and diagnostics match."""

    if repeats < 2:
        raise ValueError("repeats must be at least 2")
    action_list = [np.asarray(action, dtype=np.float32).copy() for action in actions]
    runs = [
        rollout_from_reset_state(env, reset_state, action_list, seed=seed)
        for _ in range(repeats)
    ]
    reference = runs[0]
    endpoint_diffs = [
        float(np.max(np.abs(run.endpoint - reference.endpoint))) for run in runs[1:]
    ]
    initial_diffs = [
        float(np.max(np.abs(run.initial_observation - reference.initial_observation)))
        for run in runs[1:]
    ]
    diagnostics_equal = all(
        np.array_equal(run.rewards, reference.rewards)
        and np.array_equal(run.contacts, reference.contacts)
        and run.terminated == reference.terminated
        and run.truncated == reference.truncated
        for run in runs[1:]
    )
    max_endpoint_diff = max(endpoint_diffs, default=0.0)
    max_initial_diff = max(initial_diffs, default=0.0)
    if max_endpoint_diff > atol or max_initial_diff > atol or not diagnostics_equal:
        raise AssertionError(
            "Push-T restoration was not exact: "
            f"initial={max_initial_diff}, endpoint={max_endpoint_diff}, "
            f"diagnostics_equal={diagnostics_equal}"
        )
    return {
        "repeats": repeats,
        "atol": atol,
        "max_initial_abs_diff": max_initial_diff,
        "max_endpoint_abs_diff": max_endpoint_diff,
        "diagnostics_equal": diagnostics_equal,
        "bitwise_exact": max_endpoint_diff == 0.0
        and max_initial_diff == 0.0
        and diagnostics_equal,
    }

