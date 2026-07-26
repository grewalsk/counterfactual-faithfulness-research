# Stage 3B analysis-repair protocol

## Why this repair exists

The completed Stage 3 run established a cross-environment planning advantage
for the linear physical-state readout over raw latent distance. Its separate
held-out task-margin regression did not run to completion: normalized margin
RMSE is undefined on units where all candidate actions have tied true physical
cost, and those NaNs propagated through the regression standardization.

Stage 3B repairs that analysis edge case without changing checkpoints, tasks,
state splits, candidates, horizons, readout fitting, or the primary planning
gate.

## Frozen repair rules

1. The regression analysis uses only `linear_pose` rows from the original
   `regression_train` and `final_test` splits.
2. A decision unit is finite only if the outcome and every predictor needed by
   any of the three regression specifications are finite.
3. All three regression specifications use the same finite training rows and
   the same finite final-test rows.
4. Units with tied true candidate costs and therefore undefined normalized
   margin error are excluded from every specification. They are not imputed.
5. Exclusions are reported by environment, split, horizon, row count, and
   state-cluster count.
6. Ridge strength remains fixed at the Stage 3 value. No final-test outcome is
   used for fitting or hyperparameter selection.
7. The state-clustered bootstrap and the original regression decision gate are
   unchanged.

## Additional audited corrections

- In Wall, both collisions and successful `door_cross` actions count as
  interactions when constructing action fractions and neither/one/both pair
  strata.
- Nonfinite model metrics are unranked. Finite counterfactual model summaries
  report their contributing row counts.
- `unit_metrics.csv`, the repaired decision, and a machine-readable revision
  manifest are included in `stage3b_result_bundle.zip`.

## Execution behavior

The notebook uses the same Stage 3 output directory and run signature. If the
full simulator and model shards remain in Colab or Google Drive, they are
reused and only the analysis is regenerated. If those intermediates are
missing, the notebook performs the original Stage 3 computation before running
the repaired analysis.

## Interpretation boundary

Stage 3B remains simulator-only. A passing regression gate would show that
task-aligned counterfactual margin error improves held-out prediction of
planning regret beyond the ordinary-error baseline on meaningful decision
units. It would not establish real-robot reliability or a causal relationship.

