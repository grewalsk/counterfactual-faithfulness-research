#!/usr/bin/env python3
"""Execute the frozen Stage 15 trajectory/action design in local PushT physics."""

from __future__ import annotations

import ast
import json
import math
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/15_longitudinal_predictive_control_bundle.ipynb"
DEFAULT_JEPA_REPO = ROOT.parent / "jepa-wms"


def function_source(cells, name):
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return ast.get_source_segment(source, node)
    raise RuntimeError(f"notebook function missing: {name}")


def main():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    jepa_repo = Path(os.environ.get("STAGE15_LOCAL_JEPA_REPO", DEFAULT_JEPA_REPO))
    if not jepa_repo.is_dir():
        raise RuntimeError(f"local JEPA-WM checkout not found: {jepa_repo}")
    sys.path.insert(0, str(jepa_repo))
    from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv

    def dynamic_state(environment):
        return np.asarray(
            [
                *environment.agent.position,
                *environment.block.position,
                float(environment.block.angle),
                *environment.agent.velocity,
                *environment.block.velocity,
                float(environment.block.angular_velocity),
            ],
            dtype=np.float64,
        )

    def reset_dynamic(state, seed):
        value = np.asarray(state, dtype=np.float64)
        environment = PushTEnv(with_velocity=True, with_target=True)
        environment.seed(int(seed))
        environment.reset_to_state = np.asarray([*value[:5], 0.0, 0.0])
        environment.reset()
        environment.agent.position = tuple(value[:2])
        environment.block.angle = float(value[4])
        environment.block.position = tuple(value[2:4])
        environment.agent.velocity = tuple(value[5:7])
        environment.block.velocity = tuple(value[7:9])
        environment.block.angular_velocity = float(value[9])
        restored = dynamic_state(environment)
        if not np.allclose(restored, value, atol=1e-12, rtol=0):
            raise AssertionError(
                "full local dynamic reset was not exact: "
                f"wanted={value.tolist()} restored={restored.tolist()}"
            )
        return environment

    notebook = json.loads(NOTEBOOK.read_text())
    namespace = {
        "np": np,
        "math": math,
        "TOTAL_TRAJECTORIES": 8,
        "CONSTRUCTION_TRAJECTORIES": [0, 2, 4, 6],
        "TASK_ID_OFFSET": 300,
        "DESIGN_SEED": 15137,
        "LONGITUDINAL_SAVE_STEPS": [0, 5, 10, 15, 20],
        "HORIZONS": [1, 3],
        "FRAMESKIP": 5,
        "ACTION_PROFILES": 3,
        "ACTION_BASIS_DIM": 6,
        "ACTION_TANGENT_NORM": 0.35,
        "ACTIONS_PER_STATE": 13,
    }
    for name in ["temporal_action_basis", "trajectory_specs", "candidate_action_bank"]:
        exec(function_source(notebook["cells"], name), namespace)
    namespace["RAW_ACTION_BASIS"] = namespace["temporal_action_basis"](15, 3)
    specs = namespace["trajectory_specs"]()
    candidate_action_bank = namespace["candidate_action_bank"]

    rows = []
    realized_states = []
    total_branch_contacts = 0
    contact_branches = 0
    free_branches = 0
    maximum_continuation_error = 0.0
    for spec in specs:
        environment = reset_dynamic(spec["initial_state"], spec["seed"])
        environment.set_task_goal(np.array([256.0, 256.0, 0.0]))
        states = [dynamic_state(environment)]
        path_contacts = 0
        clone = reset_dynamic(states[0], spec["seed"])
        clone.step(spec["controls"][0])
        expected_next = dynamic_state(clone)
        clone.close()
        for step, action in enumerate(spec["controls"], start=1):
            _, _, _, info = environment.step(action)
            if expected_next is not None:
                error = float(
                    np.max(np.abs(dynamic_state(environment) - expected_next))
                )
                maximum_continuation_error = max(
                    maximum_continuation_error, error
                )
                if error > 1e-8:
                    raise AssertionError(
                        "full-state reset does not reproduce continuation: "
                        f"trajectory={spec['trajectory_id']} step={step} error={error}"
                    )
                expected_next = None
            path_contacts += int(info.get("n_contacts", 0))
            if step in [5, 10, 15, 20]:
                states.append(dynamic_state(environment))
                if step < 20:
                    clone = reset_dynamic(states[-1], spec["seed"])
                    clone.step(spec["controls"][step])
                    expected_next = dynamic_state(clone)
                    clone.close()
        environment.close()
        if len(states) != 5 or not np.all(np.isfinite(states)):
            raise AssertionError("bad realized longitudinal states")
        realized_states.extend(states)

        displacement = float(np.linalg.norm(states[-1][:5] - states[0][:5]))
        if displacement <= 1.0:
            raise AssertionError("frozen controller produced a degenerate trajectory")
        trajectory_branch_contacts = 0
        trajectory_contact_branches = 0
        trajectory_free_branches = 0
        for time_index, state in enumerate(states):
            try:
                actions = candidate_action_bank(state)
            except Exception as error:
                raise RuntimeError(
                    f"candidate bank failed at trajectory {spec['trajectory_id']} "
                    f"time {time_index}, agent={state[:2].tolist()}"
                ) from error
            for branch in actions:
                branch_env = reset_dynamic(state, spec["seed"])
                branch_restored = dynamic_state(branch_env)
                if not np.allclose(branch_restored, state, atol=1e-12, rtol=0):
                    raise AssertionError(
                        "branch reset drifted from saved state: "
                        f"trajectory={spec['trajectory_id']} time={time_index} "
                        f"saved={state.tolist()} restored={branch_restored.tolist()}"
                    )
                contacts = 0
                for action in branch:
                    _, _, _, info = branch_env.step(action)
                    contacts += int(info.get("n_contacts", 0))
                branch_env.close()
                total_branch_contacts += contacts
                trajectory_branch_contacts += contacts
                if contacts:
                    contact_branches += 1
                    trajectory_contact_branches += 1
                else:
                    free_branches += 1
                    trajectory_free_branches += 1
        rows.append(
            {
                "trajectory_id": int(spec["trajectory_id"]),
                "split": spec["split"],
                "state_displacement": displacement,
                "path_contacts": int(path_contacts),
                "branch_contacts": int(trajectory_branch_contacts),
                "contact_branches": int(trajectory_contact_branches),
                "free_branches": int(trajectory_free_branches),
            }
        )

    realized = np.asarray(realized_states)
    payload = {
        "status": "PASS",
        "trajectories": rows,
        "states": int(len(realized)),
        "action_branches": int(len(realized) * 13),
        "simulator_steps": int(8 * 20 + len(realized) * 13 * 15),
        "initial_agent_block_pearson_x": float(
            np.corrcoef(realized[::5, 0], realized[::5, 2])[0, 1]
        ),
        "initial_agent_block_pearson_y": float(
            np.corrcoef(realized[::5, 1], realized[::5, 3])[0, 1]
        ),
        "contact_branches": int(contact_branches),
        "free_branches": int(free_branches),
        "total_branch_contacts": int(total_branch_contacts),
        "minimum_trajectory_displacement": float(
            min(row["state_displacement"] for row in rows)
        ),
        "maximum_trajectory_displacement": float(
            max(row["state_displacement"] for row in rows)
        ),
        "maximum_reset_continuation_error": maximum_continuation_error,
    }
    if not contact_branches or not free_branches:
        raise AssertionError("frozen design does not contain both contact and free branches")
    output = ROOT / "audits/stage15/local_trajectory_smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
