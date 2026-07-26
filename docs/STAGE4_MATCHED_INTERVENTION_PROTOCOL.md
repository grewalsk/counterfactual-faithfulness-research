# Stage 4 matched-error intervention protocol

## Motivation

Stage 3B established that a frozen linear physical-state readout supports
better action ranking and lower planning regret than raw latent distance in
PushT and Wall. It did not establish that the proposed task-margin error
predicts naturally occurring planning failures beyond ordinary prediction
error.

Stage 4 asks a different, mechanistic question:

> At the same perturbation magnitude, is planning more sensitive to corruption
> of action-specific predicted consequences than to a shared prediction error
> applied to every candidate action?

This is a readout-level causal intervention study. It does not alter or retrain
the world models, and it does not claim that the intervention identifies how
the representation was learned.

## Frozen evidence and analysis population

The input is the completed Stage 3B result bundle. Stage 4 uses only
`linear_pose` rows from the untouched Stage 3B `final_test` split:

- 2 environments: PushT and Wall;
- 2 public world-model families per environment;
- 40 final-test state clusters per environment;
- 3 frozen probe seeds;
- 3 planning horizons;
- 10 fixed candidate actions per state.

The Stage 3B tasks, candidate actions, true simulator costs, predicted physical
poses, state clusters, checkpoints, probes, and horizons remain fixed. No
Stage 4 outcome is used to select a task, row, perturbation direction, severity,
or decision threshold.

## Pose normalization and task costs

Interventions act in the exact normalized physical-state coordinates decoded
by the Stage 3 linear probes:

- PushT: block `x/512`, block `y/512`, `sin(theta)`, `cos(theta)`;
- Wall: point `x/65`, point `y/65`.

After intervention, the notebook recomputes task costs using the frozen Stage
3 decoded-cost definitions. PushT angular coordinates are renormalized to the
unit circle before evaluating the wrapped angular goal error.

## Matched interventions

For a state/model/probe/horizon unit, let the ten predicted poses be
`z_i = mean(z) + r_i`, where the residuals `r_i` sum to zero across candidate
actions.

Five severities are fixed in advance:

`rho in {0.00, 0.25, 0.50, 0.75, 1.00}`.

Five deterministic intervention seeds are fixed in advance:

`{1103, 2203, 3301, 4409, 5501}`.

Each nonzero-severity seed defines:

1. a derangement `pi` of the ten candidate identities; and
2. a unit direction `u` in normalized pose space.

The two interventions are:

### Action-structure corruption

`z_i^action(rho) = mean(z) + (1-rho) r_i + rho r_pi(i)`

This preserves the state-level centroid and, at full severity, preserves the
empirical set of action residuals while destroying which predicted consequence
belongs to which candidate action.

### Common-mode corruption

Let

`d = sqrt(mean_i ||r_pi(i) - r_i||^2)`.

Then

`z_i^common(rho) = z_i + rho d u`.

Every candidate receives the same displacement. For every unit, severity, and
intervention seed, the root-mean-square pose displacement from the intact
prediction is exactly matched between action-structure and common-mode
corruption, up to numerical tolerance.

At `rho = 0`, both interventions must reconstruct the intact predictions and
metrics exactly. The full derangement must have no fixed candidate identities.

## Outcomes

The co-primary planning outcomes are:

1. physical normalized planning regret, lower is better; and
2. physical margin-weighted pairwise action-ranking accuracy, higher is better.

Supplemental outcomes are top-1 action accuracy, ordinary decoded physical-cost
RMSE, pose perturbation RMSE, and normalized task-margin RMSE.

All uncertainty is computed with 2,000 state-cluster bootstrap replicates.
Probe seeds, intervention seeds, models, and horizons are repeated
measurements within a state cluster, not independent samples.

The co-primary tests use one common finite full-severity sample. Units with
tied true candidate costs have no physical ranking decision and therefore an
undefined weighted-ranking outcome; they are excluded from both co-primary
estimands rather than retained for regret alone or imputed. Exclusions and the
remaining state-cluster counts are reported by environment.

## Co-primary estimands

Within each environment, at full severity:

1. **specific regret damage**

   `mean(regret_action - regret_common)`;

2. **specific ranking damage**

   `mean(weighted_accuracy_common - weighted_accuracy_action)`.

The dose-response supplements fit a fixed linear slope over severity within
each intervention and compare the action-structure slope with the common-mode
slope. These slopes are descriptive unless both full-severity co-primary gates
pass.

## Decision gate

Integrity must first pass:

- Stage 3B reports success and no failure trace;
- all required Stage 3B files and finite linear-pose final-test rows exist;
- each unit contains exactly ten actions;
- all full-severity permutations are derangements;
- intact reconstruction is exact;
- maximum absolute matched-RMSE discrepancy is at most `1e-10`.
- each environment retains at least 20 finite full-severity state clusters
  after the common no-decision filter.

Each environment passes only if the state-clustered 95% lower confidence bound
is above zero for both full-severity co-primary estimands.

Decision labels:

- `CROSS_ENV_ACTION_STRUCTURE_CAUSAL_SIGNAL`: integrity and both co-primary
  estimands pass in both environments;
- `MIXED_ACTION_STRUCTURE_SIGNAL`: integrity passes and at least one, but not
  both, environments passes;
- `NO_ACTION_STRUCTURE_SPECIFICITY`: integrity passes and neither environment
  passes;
- `INCONCLUSIVE`: any integrity condition fails.

The cross-environment label licenses the claim that action-specific structure
in the decoded predicted consequences is causally necessary for planning under
these fixed simulator candidate sets, beyond an equally large shared decoded
state error. It does not establish real-robot reliability, identify the
training mechanism, or show that the Stage 3B per-instance error score predicts
natural failures.

## Outputs

The notebook must export:

- `stage4_config.json`;
- `source_bundle_manifest.json`;
- `intervention_unit_metrics.csv`;
- `intervention_summary.csv`;
- `dose_response_slopes.csv`;
- `subgroup_specificity.csv`;
- `matched_error_integrity.json`;
- `stage4_decision.json`;
- `plots/matched_regret_dose_response.png`;
- `plots/matched_ranking_dose_response.png`;
- `plots/full_severity_specificity.png`;
- `FAILURE_TRACE.txt`;
- `stage4_result_bundle.zip`.

The result ZIP is downloaded automatically at completion.
