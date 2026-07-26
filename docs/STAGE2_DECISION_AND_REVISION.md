# Stage 2 decision and confirmatory revision

## Stage 2 result

The full Stage 2 GPU run completed all 250 Push-T states, two checkpoints,
three horizons, and three model variants. Exact simulator restoration was
bitwise exact. The exported data contain all 4,500 unit rows, 202,500
action-pair rows, and 240 repeated-CV rows with no duplicate keys.

The final reporting cell failed because it attempted to cluster-bootstrap a
contact-stratum ranking field for which every value was undefined. This was a
reporting-only failure; the pre-specified decision can be recovered exactly
from the exported cross-validation and bootstrap files.

The pre-specified primary result is `INCONCLUSIVE`:

- median repeated held-out delta R²: 0.003359;
- fraction of repeated splits with positive delta R²: 1.00;
- state-clustered MSE-improvement estimate: 0.000783;
- state-clustered 95% interval: [-0.000700, 0.002376].

The point estimate favors the joint counterfactual block, but its interval
crosses zero. It is therefore neither a `NONREDUNDANT_SIGNAL` nor a
well-powered `NEGATIVE_SIGNAL`.

## Diagnosed intervention-design limitation

The fixed Stage 2 action library produced a floor-heavy physical endpoint:

- the block started near the executable goal;
- action 0 was a no-op;
- the physical oracle chose action 0 in 71.7% of real-model rows;
- normalized physical regret was zero in 74.2% of real-model rows;
- the action-blind control also had zero physical regret in 71.7% of rows;
- only 48.8% of real unit rows had finite physical pairwise accuracy.

Pair-level normalization was additionally unstable for the 3.9% of real pair
rows whose simulator-derived action-effect scale was below 1e-6. This does not
change the aggregate primary decision, but it makes unthresholded normalized
contact-stratum means unsuitable.

Stage 3 is not authorized by this result.

## Frozen Stage 2B revision

Stage 2B changes the intervention design while retaining the same models,
metrics, primary endpoint, grouped cross-validation, clustered bootstrap, and
decision rule.

- Sample 250 initial states with the block 90–130 pixels from the goal.
- Place the agent behind the block relative to the goal, with alternating
  60–68 and 72–80 pixel starting distances.
- Generate a deterministic 22-sequence state-relative action library.
- Freeze a fixed subset of 10 candidates—no-op, several push durations, and
  symmetric angular deviations—before model evaluation. Candidate selection
  uses the current state and goal geometry but not future simulator outcomes.
- Retain horizons 1, 3, and 6 and both `dino_wm_pusht` and
  `jepa_wm_pusht`.
- Preserve each initial state as the independent cluster.
- Treat pair-normalized metrics as undefined when pair effect scale is below
  1e-6; report raw pair effect RMSE for contact strata.
- Restrict secondary pairwise-outcome regression to finite rows.

The simulator-only design gate, checked before model evaluation, requires:

- final-horizon no-op oracle fraction below 20%;
- final-horizon positive no-op regret in more than 80% of states;
- median final-horizon physical-cost spread above 0.08;
- observations in all `neither`, `one`, and `both` contact strata.

A 36-state exact Push-T CPU smoke test of the fixed candidate set passed these
gates:

- final-horizon no-op oracle fraction: 5.6%;
- final-horizon positive no-op regret fraction: 94.4%;
- median final-horizon physical-cost spread: 0.281;
- contact-pair counts: 707 neither, 863 one, and 3,290 both.

## Unchanged confirmatory decision rule

The primary endpoint remains normalized executable physical regret.

The base model contains ordinary rollout RMSE, horizon, checkpoint, aggregate
ground-truth effect scale, contact fraction, and design stratum. The full model
adds normalized paired error and one minus paired-effect cosine.

The result is `NONREDUNDANT_SIGNAL` only when:

1. median repeated held-out delta R² is positive; and
2. the state-clustered 95% interval for out-of-fold MSE improvement excludes
   zero on the positive side.

An interval excluding zero on the negative side with a negative median is
`NEGATIVE_SIGNAL`; all other outcomes are `INCONCLUSIVE`.

Do not proceed to Stage 3 unless Stage 2B satisfies a predeclared Stage 3 gate.
All conclusions remain limited to controlled simulator interventions and
executable Push-T planning.
