#!/usr/bin/env python3
"""Run the deterministic Push-T and synthetic-metric Stage 0 smoke checks."""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

import gymnasium as gym
import gym_pusht  # noqa: F401
import numpy as np
import pymunk

from cf_faithfulness.metrics import paired_counterfactual_metrics, ranking_metrics
from cf_faithfulness.pusht_restore import (
    assert_exact_reset_restoration,
    rollout_from_reset_state,
)


def main() -> int:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    out_dir = Path(__file__).resolve().parents[1] / "cpu_smoke_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="state",
        render_mode="rgb_array",
        max_episode_steps=100,
    )
    state = np.array([256.0, 420.0, 256.0, 300.0, 0.0])
    action_sequences = [
        np.tile(np.array([256.0, 200.0]), (6, 1)),
        np.tile(np.array([400.0, 400.0]), (6, 1)),
        np.tile(np.array([100.0, 400.0]), (6, 1)),
        np.tile(np.array([256.0, 420.0]), (6, 1)),
    ]
    restoration = assert_exact_reset_restoration(
        env, state, action_sequences[0], repeats=4, seed=123, atol=0.0
    )
    branches = [
        rollout_from_reset_state(env, state, actions, seed=123)
        for actions in action_sequences
    ]
    env.close()

    truth = np.stack([branch.endpoint for branch in branches])[None, :, None, :]
    common_bias = np.array([2.0, -1.0, 0.5, -0.25, 0.1])[None, None, None, :]
    action_error = np.linspace(-0.2, 0.2, truth.shape[1])[None, :, None, None]
    prediction = truth + common_bias + action_error
    paired = paired_counterfactual_metrics(truth, prediction)

    goal = np.array([256.0, 256.0, np.pi / 4])
    true_cost = np.empty((1, truth.shape[1], 1), dtype=np.float64)
    for action_idx, endpoint in enumerate(truth[0, :, 0]):
        angle_error = np.arctan2(
            np.sin(endpoint[4] - goal[2]), np.cos(endpoint[4] - goal[2])
        )
        true_cost[0, action_idx, 0] = np.linalg.norm(
            np.r_[(endpoint[2:4] - goal[:2]) / 512.0, angle_error / np.pi]
        )
    predicted_cost = true_cost + np.array([0.2, -0.1, 0.0, 0.1])[None, :, None]
    ranking = ranking_metrics(true_cost, predicted_cost)

    payload = {
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "gymnasium": gym.__version__,
            "gym_pusht": "0.1.6",
            "pymunk": pymunk.version,
        },
        "restoration": restoration,
        "branch_endpoints": [branch.endpoint.tolist() for branch in branches],
        "branch_contact_sums": [int(np.sum(branch.contacts)) for branch in branches],
        "paired_summary": paired.summary(),
        "ranking_summary": ranking.summary(),
        "status": "PASS",
    }
    output_path = out_dir / "cpu_smoke_results.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

