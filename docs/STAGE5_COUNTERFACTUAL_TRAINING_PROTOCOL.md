# Stage 5 counterfactual decision-readout training protocol

## Motivation and scope

Stage 4 established a readout-level causal result: at exactly matched decoded
pose-error magnitude, destroying the assignment between actions and predicted
consequences damages planning more than a shared common-mode error.

Stage 5 asks the actionable learning question:

> With the world-model checkpoint and training data held fixed, does explicitly
> training a decision readout to preserve same-state action differences improve
> planning on new tasks without materially worsening ordinary endpoint error?

This is a simulator-only, prospective training experiment. It trains compact
physical-state decision readouts over frozen public world-model predictions. It
does not fine-tune the visual encoder or action-conditioned latent predictor and
must not be described as full world-model retraining.

## Prospective evidence

Stage 5 generates a new task family that is numerically distinct from all Stage
3 tasks. No Stage 5 final-task simulator outcome, latent prediction, decoded
pose, or planning metric existed when this protocol and its decision rule were
frozen.

The experiment uses:

- environments: PushT and Wall;
- frozen checkpoints: DINO-WM and JEPA-WM for each environment;
- horizons: 1, 3, and 6 model steps;
- 12 new tasks per environment;
- 20 states nested within each task in the full run;
- 10 fixed state/task-relative candidate actions per state;
- three evaluation seeds;
- three readout projection/training seeds.

The seeded task partition is:

- 6 tasks / 120 states for readout training;
- 3 tasks / 60 states for descriptive calibration;
- 0 regression-training tasks;
- 3 tasks / 60 states for one-shot final testing.

Task identity and state identity are disjoint across partitions. Calibration
outcomes do not select a loss weight, architecture, epoch, or training seed.
All training hyperparameters are fixed before the final run.

## Frozen representation and readout

Each public world model is frozen. Its predicted future visual feature map is
spatially pooled exactly as in Stage 3 and projected to 256 dimensions by a
saved Gaussian projection.

For every environment, checkpoint, and readout seed, the same compact MLP maps
each projected future feature vector to normalized physical pose:

- PushT: block `x/512`, block `y/512`, `sin(theta)`, `cos(theta)`;
- Wall: point `x/65`, point `y/65`.

The readout architecture, initialization, minibatch state sets, optimizer,
learning-rate schedule, update count, absolute endpoint term, and training
examples are identical across objectives. Only the relation target differs.
All ten candidate actions from a state and horizon are presented together
during training. Initial-parameter hashes and completed update counts are
recorded for every fitted head.

## Equal-data, equal-update objectives

Let `y_si` be the true normalized endpoint for action `i` from state `s`, and
let `p_si` be the readout prediction. All losses operate after standardizing
pose components using training-partition statistics only.

Every condition includes the same pointwise endpoint loss

`L_abs = mean_si ||p_si - y_si||^2`.

The fixed relation weight is `lambda = 1`. There is no Stage 5 hyperparameter
search.

### Ordinary endpoint

`L = L_abs`.

This is the primary same-architecture baseline.

### Independent-pair control

`L = L_abs + lambda L_independent`.

`L_independent` preserves differences between examples drawn from different
initial-state clusters. It controls for additional difference supervision and
gradient computation without supplying the same-state counterfactual relation.

### Counterfactual difference

`L = L_abs + lambda L_counterfactual`.

`L_counterfactual` is the mean squared error between all 45 predicted and true
within-state action differences:

`(p_si - p_sj)` versus `(y_si - y_sj)`, for all `i < j`.

This is the proposed remedy.

### Shuffled-pair control

`L = L_abs + lambda L_shuffled`.

The same 45 within-state predicted action differences are compared with true
differences after a fixed derangement of target action identities. The control
retains pair count, target scale, state grouping, and compute while removing the
correct action-to-consequence assignment.

## Evaluation

All objectives are evaluated on calibration tasks for descriptive optimization
checks and exactly once on the untouched final tasks. The primary comparison is
counterfactual-difference training versus ordinary-endpoint training.

Co-primary final-test planning outcomes are:

1. reduction in physical normalized planning regret;
2. increase in physical margin-weighted pairwise action-ranking accuracy.

Ordinary physical pose error, decoded physical-cost RMSE, normalized cost-margin
RMSE, and top-1 action accuracy are reported. Uncertainty uses 2,000 bootstrap
replicates over state clusters. Checkpoints, readout seeds, and horizons are
repeated observations within a state, not independent samples.

Units whose true candidate costs are all tied have no physical ranking
decision. Their regret and ordinary-error outcomes remain defined, but their
weighted-ranking outcome is undefined. Ranking contrasts use the common finite
paired sample without imputation. Each environment must retain at least 20
final state clusters with a finite ranking decision.

## Ordinary-error noninferiority

The planning claim is licensed only if the proposed readout is noninferior in
ordinary physical pose error. Within each environment, bootstrap replicates
estimate the ratio

`mean(pose_error_counterfactual) / mean(pose_error_ordinary)`.

The pre-specified noninferiority margin is 1.05. The upper endpoint of the 95%
cluster-bootstrap interval must be at most 1.05. This supports the wording
“planning improvement at matched ordinary endpoint error”; it does not imply
bitwise equality of the two errors.

## Integrity gates

The analysis is `INCONCLUSIVE` unless all of the following hold:

- both exact simulator-restoration checks pass;
- the candidate design passes the frozen Stage 3 diversity thresholds;
- all four public checkpoint configurations finish;
- final tasks and state IDs are absent from training;
- every unit has exactly ten candidate actions;
- every objective uses identical training units and completed update counts;
- initial-parameter hashes match across objectives within a fitted-head group;
- all saved predictions, ordinary errors, and regret outcomes are finite;
- undefined weighted-ranking values occur only as no-decision exclusions and
  each environment retains at least 20 finite-ranking final state clusters;
- each environment retains all 60 final state clusters;
- no final outcome is used for model, epoch, seed, loss-weight, or architecture
  selection.

## Decision rule

Within an environment, the primary training gate passes only if:

- the 95% lower confidence bound for
  `regret_ordinary - regret_counterfactual` is above zero;
- the 95% lower confidence bound for
  `ranking_counterfactual - ranking_ordinary` is above zero; and
- the upper confidence bound for the ordinary-pose-error ratio is at most 1.05.

Objective specificity is stronger evidence. It additionally requires the
counterfactual objective to outperform both the independent-pair and
shuffled-pair controls on both co-primary planning outcomes, with lower
confidence bounds above zero, in both environments.

Decision labels:

- `CROSS_ENV_OBJECTIVE_SPECIFIC_FIX`: integrity passes; the primary training
  gate passes in both environments; and both control-specificity gates pass in
  both environments;
- `CROSS_ENV_COUNTERFACTUAL_TRAINING_FIX`: integrity and the primary training
  gate pass in both environments, but full control specificity does not;
- `PLANNING_GAIN_WITH_ERROR_TRADEOFF`: both planning contrasts pass in both
  environments, but ordinary-error noninferiority fails in at least one;
- `MIXED_TRAINING_SIGNAL`: integrity passes and the complete primary gate passes
  in exactly one environment;
- `NO_TRAINING_FIX`: integrity passes and neither environment passes the
  complete primary gate;
- `INCONCLUSIVE`: any integrity condition fails.

The strongest label licenses the claim that, for these frozen simulator world
models and new task families, explicitly preserving correct same-state
action-consequence differences improves planning beyond ordinary endpoint
training and matched relation controls at noninferior ordinary endpoint error.
It does not establish real-robot reliability or full-backbone world-model
repair.

## Required outputs

The Colab notebook exports:

- `config.json`;
- `versions.json`;
- `tasks.json`;
- `split_manifest.json`;
- `restore_test.json`;
- `candidate_design_summary.json`;
- `checkpoints_manifest.json`;
- `training_manifest.json`;
- `training_history.csv`;
- `action_predictions.csv`;
- `unit_metrics.csv`;
- `objective_contrasts.csv`;
- `ordinary_error_noninferiority.csv`;
- `stage5_decision.json`;
- `plots/planning_by_objective.png`;
- `plots/ordinary_error_by_objective.png`;
- `plots/objective_contrasts.png`;
- trained readout checkpoints;
- logs and `FAILURE_TRACE.txt`;
- `stage5_result_bundle.zip`, downloaded automatically.
