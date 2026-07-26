# Stage 3 independent result audit

## Bottom line

Stage 3 contains a valid cross-environment planning result, but it does **not**
contain a valid held-out regression result for the task-aligned pair metric.

The defensible scientific conclusion is:

> A linear, task-aligned physical-state readout from the frozen world-model
> features selects counterfactual actions substantially better than raw
> latent-goal distance in both PushT and Wall. This generalizes across held-out
> tasks and states inside simulation. Whether task-aligned pair error adds
> predictive information beyond ordinary rollout errors remains unresolved
> because the pre-specified regression was numerically invalid.

The bundle calls this `CROSS_ENV_PLANNING_SIGNAL_ONLY`. That label is broadly
right for the planning result, but “regression missing/inconclusive” is more
accurate than “regression negative.”

## Integrity audit

All core exported tables reconcile:

- 72,000 action rows, 324,000 pair rows, 60 summary rows, 12 model-ranking
  rows, and 1,440 held-out regression rows;
- no duplicate primary keys;
- exactly 10 actions and 45 unordered pairs per evaluation unit;
- exact task/state disjointness for probe-train, calibration,
  regression-train, and final-test splits;
- all four model runs and both simulator runs completed 240/240 states;
- PushT and Wall restoration were bitwise exact across endpoint, initial
  render, and diagnostics;
- all 12 saved probes were finite, frozen, and fit without test states;
- action-blind, oracle, and cyclic action-shift controls were exact;
- every pair margin, pair weight, ranking credit, and action label reconciled
  to the action table;
- every reported point estimate and all nine primary clustered bootstrap
  intervals were independently reproduced from the raw rows;
- the candidate-design summaries reproduced exactly and passed their
  pre-specified validity thresholds.

## Primary planning result

The reported improvement is linear physical-state readout minus raw latent
distance, with regret oriented so that positive is better.

| Environment | Normalized-regret reduction (95% CI) | Weighted-ranking gain (95% CI) | Top-1 gain (95% CI) |
|---|---:|---:|---:|
| PushT | 0.218 [0.147, 0.294] | 0.228 [0.160, 0.302] | 0.194 [0.104, 0.288] |
| Wall | 0.230 [0.186, 0.273] | 0.272 [0.239, 0.307] | 0.132 [0.065, 0.197] |
| Pooled | 0.224 [0.183, 0.266] | 0.251 [0.213, 0.289] | 0.163 [0.108, 0.223] |

Absolute averages across models and horizons:

| Environment/readout | Top-1 | Normalized regret | Weighted pair accuracy |
|---|---:|---:|---:|
| PushT latent distance | 0.296 | 0.392 | 0.548 |
| PushT linear pose | 0.490 | 0.174 | 0.771 |
| Wall latent distance | 0.125 | 0.357 | 0.673 |
| Wall linear pose | 0.257 | 0.127 | 0.944 |

This is not merely a constant or ordering artifact. Against the action-shifted
linear control, the intact readout gained 0.208 weighted accuracy in PushT
(95% CI [0.160, 0.256]) and 0.167 in Wall ([0.150, 0.185]). Regret also
improved clearly in PushT. In Wall the regret improvement over the shifted
control was 0.042, but its interval narrowly crossed zero
([-0.002, 0.087]); the much more data-efficient weighted-ranking measure still
showed an unambiguous action-ordering effect.

## Interaction-conditioned behavior

The advantage survives physically difficult branches.

- PushT, using the exported contact strata: for pairs where both actions made
  contact, weighted ranking was 0.829 for the linear readout versus 0.495 for
  latent distance; for one-contact pairs it was 0.763 versus 0.566.
- Wall, after reclassifying door crossings as interactions as the written
  protocol specifies: linear versus latent weighted ranking was 0.937 versus
  0.818 for both-interacting pairs, 0.974 versus 0.581 for one-interacting
  pairs, and 0.940 versus 0.577 for neither-interacting pairs.

Thus the cross-environment result is not driven only by free-space motion.

## Held-out regression defect

The ordinary-only regression had MAE 0.1881 and negative held-out
R² (-0.0686). Adding raw frozen-feature pair error made it slightly worse:
MAE 0.1906, with MAE improvement -0.00257
(95% CI [-0.00353, -0.00168]). This is valid negative evidence for the raw
pair metric.

The task-aligned pair regression is different: it never produced predictions.
All 1,440 `ordinary_plus_task_pair_prediction` values are NaN, so its MAE,
R², confidence interval, and cluster count are undefined.

The immediate cause is a legitimate edge case that was not handled by the
analysis code. Twelve of the 40 PushT final-test states have exactly zero
physical-cost spread at horizon 1. With no non-tied action margins, normalized
margin RMSE is undefined. These 12 states create 72 undefined linear-readout
rows after the two models and three probe seeds are replicated. The regression
standardization propagated the NaNs through every prediction. The same issue
also makes the right panel of `held_out_regression.png` blank.

Consequently, the pre-specified incremental-prediction hypothesis was **not
tested**. It was not falsified.

On the final-test rows only, an exploratory association remains compatible
with the intended idea: task-margin error had Spearman correlation 0.381 with
regret in PushT and 0.352 in Wall. These are descriptive, not a substitute for
the held-out regression.

## Model-ranking artifact

The displayed “1/6 rank reversals” result is invalid. Both PushT horizon-1
counterfactual metrics are NaN, and `argsort` assigned arbitrary ranks to the
two NaNs. The only plotted reversal is therefore a missing-data artifact.

The ordinary and finite counterfactual rankings otherwise both favor JEPA-WM
in the available comparisons. Planning regret does disagree with ordinary
error in PushT at horizons 1 and 3, where DINO-WM has lower regret despite
worse ordinary cost RMSE. With only two model families and one public training
seed per configuration, model-ranking claims should remain descriptive.

## Wall taxonomy mismatch

The protocol says Wall collisions and successful door crossings are
interactions. The exported neither/one/both strata use collision counts only,
although `door_cross` remains in the action-level labels. Under the written
definition, 26,820 replicated final-test pair rows change stratum. Across the
full Wall design, interaction fractions become 0.039, 0.595, and 0.862 at
horizons 1, 3, and 6 rather than the collision-only 0.039, 0.470, and 0.730.

All three strata remain present, and re-stratification preserves the strong
linear-readout advantage. This mismatch does not affect the primary planning
metrics, but it should be corrected before publication.

## Recommended next step

Run a small Stage 3B **analysis repair**, not a new conceptual experiment:

1. Export the unit-level rows for both `regression_train` and `final_test`.
2. Pre-specify normalized margin error only on units with nonzero true
   cost spread. Compare all regressions on the same finite rows and report the
   number of excluded no-decision units by environment, split, and horizon.
3. Repeat the fixed-ridge, task/state-disjoint held-out regression and
   state-clustered bootstrap without inspecting final-test outcomes.
4. Count Wall `door_cross` as an interaction when constructing strata.
5. Treat nonfinite model metrics as unranked and remove the artificial
   PushT-H1 reversal.
6. Export the repaired unit table so the regression can be independently
   reproduced without model reruns.

If the Colab intermediates still exist, this should require only rerunning the
analysis phase. The downloaded bundle alone is insufficient to repair the
held-out regression because it does not contain the regression-train
unit-level metrics.

