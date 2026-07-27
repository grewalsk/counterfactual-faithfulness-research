# Stage 6 structured action-effect development protocol

## Status and evidential boundary

Stage 6 is an exploratory method-development experiment. It deliberately
reuses the Stage 5 task family and its original train/calibration/final task
partition because Stage 5 outcomes have already been inspected. The former
Stage 5 final partition is therefore renamed `development_holdout` throughout
Stage 6. It is not an untouched test set and cannot support a new confirmatory
paper claim.

Stage 6 may select an architecture, loss, epoch, and training recipe. A
successful Stage 6 result nominates one frozen candidate for a later Stage 6B
experiment on a numerically new task family. Stage 6B, not Stage 6, must provide
the prospective efficacy result.

## Motivation

Stage 4 showed that action-specific consequence structure is causally necessary
for planning under matched decoded-pose error. Stage 5 then tested whether an
ordinary per-action readout could be repaired by adding an all-pairs endpoint
difference loss.

Stage 5 completed with valid integrity checks but returned `NO_TRAINING_FIX`.
The same-state loss modestly improved some planning metrics over endpoint-only
training, while an independent-state difference control was equally good or
better. Algebraically, the Stage 5 relation residual for actions `i` and `j`
was only

`(p_i - p_j) - (y_i - y_j) = e_i - e_j`,

where `e_i = p_i - y_i`. It reweighted variation in ordinary endpoint errors;
it did not directly shape action-conditioned dynamics, action identity,
decision margins, or regret.

Stage 6 asks a narrower development question:

> Can an explicitly set-aware, action-conditioned adapter learn a
> task-general decomposition between shared world evolution and
> action-specific effects, and can decision-aligned supervision turn that
> decomposition into better candidate ranking without unacceptable endpoint
> error?

## Fixed simulator and world-model inputs

Stage 6 reuses the exact Stage 5 simulator design:

- PushT and Wall;
- DINO-WM and JEPA-WM checkpoints for each environment;
- 240 states per environment in the full run;
- 10 candidate action sequences per state;
- model horizons 1, 3, and 6;
- the Stage 5 6/3/0/3 task partition;
- exact state restoration and candidate-diversity checks.

All public visual encoders and world-model dynamics predictors remain frozen.
Stage 6 trains only compact adapters over their predicted future features.

The 6,144-dimensional pooled future feature is no longer passed through a
fixed Gaussian projection. Each adapter learns its own low-dimensional
projection jointly with the decision objective.

## Structured adapter

The adapter receives all ten candidate futures for one state and horizon as a
set. It also receives a compact descriptor of each executable action sequence
and a horizon embedding.

For projected candidate representation `z_i`, the adapter computes:

- shared context `c = mean_i z_i`;
- centered candidate feature `d_i = z_i - c`;
- a baseline physical endpoint from `c`;
- an action-specific physical displacement from `d_i`, the action descriptor,
  and the horizon embedding.

Action index zero is the fixed no-op candidate. Predicted action effects are
anchored by subtracting the no-op displacement, so

`predicted_pose_i = predicted_noop_pose + predicted_effect_i`

and `predicted_effect_0 = 0` by construction.

An auxiliary decoder predicts the executed action descriptor from the centered
world-model feature before the explicit action descriptor is concatenated. It
therefore measures and trains the learned projection to expose action
information already present in the frozen future prediction.

## Training losses

All conditions use the same architecture, initialization within comparison
groups, training examples, minibatch order, optimizer, maximum epoch count,
and calibration checkpoints.

Every condition includes standardized endpoint Huber loss:

`L_endpoint = Huber(predicted_pose, true_pose)`.

Additional components are:

### Same-state no-op-relative effect loss

`L_effect = Huber(predicted_pose_i - predicted_pose_0,
                  true_pose_i - true_pose_0)`.

Unlike the Stage 5 all-pairs loss, this is coupled to an explicit baseline plus
action-effect architecture.

### Independent-state effect control

The same computation is performed against a no-op endpoint rolled from another
state/horizon unit in the minibatch. It controls for added difference
supervision without providing the correct same-state intervention.

### Action decoding

`L_action` reconstructs the standardized executable-action descriptor from the
centered predicted-future representation.

### Decision-aligned ranking

Predicted physical poses are converted differentiably to the same analytic
PushT or Wall task cost used at evaluation. `L_rank` is a cost-gap-weighted
pairwise logistic loss. It directly penalizes reversed candidate ordering and
gives larger weight to action pairs with larger true physical-cost gaps.

## Development conditions

Stage 6 compares six same-architecture conditions:

1. `endpoint_only`;
2. `action_decode_only`;
3. `ranking_only`;
4. `counterfactual_effect_only`;
5. `independent_action_effect`;
6. `counterfactual_action_effect` — the proposed combined adapter.

The combined adapter uses endpoint, same-state effect, action-decoding, and
ranking losses. The independent control replaces only the same-state effect
target.

## Calibration and evaluation

Every method trains to the same maximum epoch count. Common checkpoint epochs
are evaluated on the original Stage 5 calibration tasks using one fixed
selection score:

`normalized_regret + (1 - weighted_pairwise_accuracy) + 0.25 * pose_error`.

The lowest calibration score selects the checkpoint independently for each
environment, checkpoint, adapter seed, and condition. The former Stage 5 final
partition is evaluated afterward and labeled `development_holdout`.

Stage 6 reports state-clustered bootstrap contrasts for the proposed combined
adapter against every ablation. Positive contrast values always mean that the
combined adapter is better. It also reports the combined/baseline physical-pose
error ratio with an exploratory 1.10 noninferiority margin.

## Development decision

The result is `INCONCLUSIVE` if simulator, model, training, split, finite-value,
or checkpoint-selection integrity fails.

Otherwise:

- `DEVELOPMENT_CANDIDATE_READY`: in both environments, the combined adapter
  improves normalized regret and weighted pairwise accuracy over
  `endpoint_only`, passes pose-error noninferiority, and improves both planning
  metrics over `independent_action_effect`;
- `PROMISING_BUT_NOT_SPECIFIC`: the combined adapter passes the endpoint and
  pose gates in both environments but does not beat the independent control in
  both;
- `MIXED_DEVELOPMENT_SIGNAL`: the complete endpoint/pose gate passes in exactly
  one environment;
- `NO_DEVELOPMENT_GAIN`: the complete endpoint/pose gate passes in neither
  environment.

These labels guide development only. None licenses a held-out efficacy claim.

## Required outputs

The notebook exports:

- configuration, dependency, task, split, restoration, candidate-design, and
  checkpoint manifests;
- adapter-training and checkpoint-selection manifests;
- training histories;
- unit-level and action-level predictions;
- method summaries and state-clustered contrasts;
- pose-error noninferiority intervals;
- the Stage 6 development decision;
- diagnostic plots;
- trained adapter checkpoints;
- logs and a failure trace;
- `stage6_result_bundle.zip`, downloaded automatically.
