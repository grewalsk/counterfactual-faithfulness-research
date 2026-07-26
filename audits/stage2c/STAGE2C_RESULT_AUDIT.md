# Stage 2C result audit

## Verdict

`TASK_ALIGNED_SIGNAL` is supported by the returned bundle and reproduces from
the raw final-test tables.

The negative Stage 2B result did not show that the frozen world models lacked
action-conditioned physical information. It showed that Euclidean distance in
the learned latent space was not itself a usable task cost. A state-disjoint
linear readout recovers block pose from predicted future features and converts
the same frozen model predictions into substantially better action rankings and
lower simulator regret.

This is a simulator planning result. It does not establish real-robot
reliability.

## Primary result

The pre-specified primary comparison pooled two frozen checkpoints, horizons 3
and 6, and 90 held-out simulator states. Improvements are paired within
state/model/horizon and use a state-clustered 2,000-replicate bootstrap.

| Final-test comparison | Estimate | 95% CI |
|---|---:|---:|
| Linear readout reduction in normalized regret | 0.391 | [0.329, 0.452] |
| Linear readout increase in margin-weighted pair accuracy | 0.414 | [0.371, 0.454] |
| Linear readout increase in top-1 action accuracy | 0.411 | [0.336, 0.481] |
| MLP readout reduction in normalized regret | 0.414 | [0.349, 0.478] |
| MLP readout increase in margin-weighted pair accuracy | 0.452 | [0.406, 0.497] |

Both lower confidence bounds required by the primary linear gate are well above
zero.

Pooled descriptive performance:

| Readout | Top-1 action accuracy | Normalized regret (lower is better) | Weighted pair accuracy |
|---|---:|---:|---:|
| Raw latent distance | 0.139 | 0.502 | 0.463 |
| Linear pose | 0.550 | 0.111 | 0.877 |
| MLP pose | 0.581 | 0.088 | 0.915 |
| Linear pose, action-shuffled | 0.103 | 0.488 | 0.528 |
| Action-blind | 0.022 | 0.546 | 0.500 |
| Oracle | 1.000 | 0.000 | 1.000 |

The primary result is not dependent on the nonlinear probe: the linear probe
already passes. The shuffled and action-blind controls collapse toward poor
planning, so the gain is not explained by target-frequency or constant-output
effects.

## Checkpoint and horizon consistency

The signal appears in every checkpoint/horizon cell.

| Checkpoint | Horizon | Readout | Top-1 | Normalized regret | Weighted pair accuracy |
|---|---:|---|---:|---:|---:|
| DINO-WM PushT | 3 | Latent | 0.156 | 0.488 | 0.491 |
| DINO-WM PushT | 3 | Linear | 0.556 | 0.068 | 0.865 |
| DINO-WM PushT | 6 | Latent | 0.222 | 0.458 | 0.488 |
| DINO-WM PushT | 6 | Linear | 0.567 | 0.111 | 0.890 |
| JEPA-WM PushT | 3 | Latent | 0.122 | 0.552 | 0.439 |
| JEPA-WM PushT | 3 | Linear | 0.500 | 0.168 | 0.851 |
| JEPA-WM PushT | 6 | Latent | 0.056 | 0.510 | 0.435 |
| JEPA-WM PushT | 6 | Linear | 0.578 | 0.098 | 0.901 |

Linear decoded-pose error is approximately 0.053 at horizon 3 and 0.062–0.063
at horizon 6 in the normalized physical pose metric. The decoded task-cost RMSE
is approximately 0.043–0.053.

## Integrity and leakage audit

All 18 audit checks passed:

- 300/300 simulator states and 300/300 states for each frozen checkpoint
  completed.
- Exact restoration passed for endpoints, initial renders, and diagnostics;
  maximum endpoint difference was exactly zero.
- The split is exactly 150 probe-train, 60 calibration, and 90 final-test
  states; the sets are disjoint and cover state IDs 0–299.
- The probe manifest records frozen world models and no final-test fitting.
- All expected rows are present: 2,520 unit rows, 25,200 action rows, 113,400
  pair rows, and 28 summary rows.
- Key columns contain no duplicate rows.
- Every unit has exactly 10 action rows and 45 unordered action-pair rows.
- Simulator truth is identical across copied model/readout rows.
- Action-blind, shifted-action, and oracle controls reconstruct exactly.
- Unit metrics recompute from action rows; pair metrics recompute from pair
  rows; summary means recompute from unit rows.
- The reported bootstrap estimates and interval endpoints reproduce to
  floating-point precision using the notebook's stated seeds and clustered
  algorithm.
- Decoded costs reproduce from saved predicted poses. True costs reproduce from
  saved simulator poses to within `3e-8`, the expected float32 storage error.

The result ZIP intentionally excludes the 900 intermediate feature/truth shards.
Therefore this audit can exactly reproduce the reported final-table analysis but
cannot independently refit the probes without rerunning the notebook.

## Interaction interpretation

For informative pairs, the linear readout reaches weighted pair accuracy 0.917
when both actions involve contact and 0.868 when exactly one does, versus 0.462
and 0.450 for latent distance. In the held-out table, pairs where neither action
contacts have zero physical-cost margin and therefore no defined ranking label.

The evidence is consequently strongest around contact-sensitive intervention
regimes, not passive non-contact motion. Stage 3 should preserve this taxonomy
and add a second simulator with a different interaction boundary.

## Remaining limitations

- Only PushT is tested.
- There are two public environment-specific checkpoints but no public
  training-seed replicas.
- Probe targets are simulator state variables. This establishes that useful
  physical state is accessible in frozen predictions; it does not show that
  raw latent geometry is intrinsically task-aligned.
- Both MLP probes selected the final allowed epoch (199), so nonlinear-probe
  convergence is not demonstrated. This does not affect the primary linear
  conclusion.
- Candidate-set regret is an executable discrete planning test, not continuous
  closed-loop control.

## Decision

Proceed to Stage 3. The next benchmark should test whether the linear
task-aligned result survives held-out tasks and states in both PushT and Wall,
whether paired margin error predicts planning regret beyond ordinary endpoint
error on a separately held-out regression set, and whether standard versus
counterfactual metrics change checkpoint rankings.
