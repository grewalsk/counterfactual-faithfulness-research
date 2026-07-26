# Stage 3 full counterfactual benchmark protocol

## Decision basis

Stage 2C passed its pre-specified linear task-aligned gate. Stage 3 therefore
tests generalization rather than introducing a new probe after seeing Stage 3
outcomes.

## Scope

- Environments: PushT and Wall.
- Frozen public checkpoints:
  - `dino_wm_pusht`
  - `jepa_wm_pusht`
  - `dino_wm_wall`
  - `jepa_wm_wall`
- Horizons: 1, 3, and 6 model steps.
- States: 240 per environment in the full run.
- Candidate action sequences: 10 per state, selected by fixed state/task-relative
  rules before any future simulator outcomes are computed.
- Evaluation seeds: 71, 131, and 191.
- Linear-probe projection seeds: 2071, 4071, and 6071.

The four checkpoint/environment configurations satisfy the requested breadth of
substantively different public checkpoints. The pinned public registry exposes
one training seed per configuration. This is reported as a limitation;
evaluation/readout seeds are not described as training replicas.

## Held-out design

Each environment has 12 fixed tasks. A seeded permutation allocates:

- 6 tasks to probe training;
- 2 tasks to probe calibration;
- 2 tasks to held-out regression training;
- 2 tasks to the untouched final test.

States are nested within one task, so task and state IDs are disjoint between all
four partitions. With 240 states per environment, the partition sizes are
120/40/40/40.

PushT tasks vary the external block-pose goal. Wall tasks vary wall location,
door location, and goal position. Goal images are encoded only for the raw
latent-distance baseline; the task-aligned planner computes cost from decoded
physical state and the external task definition.

## Readout and controls

The primary readout is a frozen linear ridge map from a saved, seeded
256-dimensional Gaussian projection of predicted future visual features to:

- PushT: block x/512, block y/512, sin(theta), cos(theta);
- Wall: dot x/65, dot y/65.

Ridge strength is selected only by pose MSE on probe-calibration tasks/states.
Final-test rows are never used for fitting or hyperparameter selection.

Controls:

- raw latent-goal distance;
- action-blind constant cost;
- action-shifted linear costs;
- simulator oracle cost.

## Intervention taxonomy

PushT records cumulative physical contact. Wall records collision with a wall or
border and successful door crossing. Action pairs are stratified as neither,
one, or both actions interacting, with the action-level interaction types also
retained.

Candidate design is invalid if, at the final horizon:

- no-op is oracle in 25% or more states;
- no-op has positive regret in 75% or fewer states;
- median physical-cost spread is at most 0.08 in PushT or 0.05 in Wall; or
- any neither/one/both pair-interaction stratum is absent.

An invalid design or failed exact restoration forces an `INCONCLUSIVE` decision.

## Primary planning gate

Within each environment, compare linear readout with raw latent distance on the
final-test tasks/states. Pair rows by environment, state, checkpoint, horizon,
and projection seed. Resample state clusters with 2,000 bootstrap replicates.

Each environment passes only if the 95% lower confidence bound is above zero
for both:

1. reduction in physical normalized planning regret; and
2. increase in margin-weighted pairwise action-ranking accuracy.

Top-1 action accuracy is supplemental.

## Held-out nonredundancy regression

Fit fixed-ridge regressions only on the separate regression-train tasks/states
and evaluate once on final-test tasks/states.

Outcome:

- normalized planning regret from the linear readout.

Ordinary-only predictors:

- decoded physical pose error;
- decoded physical-cost RMSE;
- ordinary frozen-feature rollout RMSE;
- common-mode frozen-feature RMSE;
- interaction fraction;
- environment, model-family, and horizon indicators.

Ablations add either:

- raw frozen-feature normalized paired error; or
- task-aligned normalized cost-margin error.

The regression gate passes only if the state-clustered 95% lower confidence
bound for final-test MAE improvement from task-aligned margin error is above
zero. This is predictive evidence, not a causal estimate.

## Model rankings

For each environment/horizon, rank the two public model families under:

- ordinary decoded cost RMSE;
- counterfactual normalized margin RMSE;
- final planning regret.

The result bundle records any reversal between ordinary and counterfactual
rankings.

## Decision labels

- `CROSS_ENV_TASK_ALIGNED_SIGNAL`: integrity passes, both environments pass the
  planning gate, and the held-out regression gate passes.
- `CROSS_ENV_PLANNING_SIGNAL_ONLY`: integrity and both environment planning
  gates pass, but the regression gate does not.
- `MIXED_GENERALIZATION`: integrity passes but at least one environment planning
  gate fails.
- `INCONCLUSIVE`: candidate-design or exact-restoration integrity fails.

## Reproducibility and outputs

The notebook runs from a fresh Colab runtime, installs pinned non-PyTorch
dependencies, pins JEPA-WMs to commit
`13cf1d9c7e476f53c17714d2e0f1dc239a883ce0`, caches checkpoints, writes
resumable truth/model shards, reports GPU memory, saves raw CSV/JSON tables and
plots, packages a single `stage3_result_bundle.zip`, and requests an automatic
browser download when Google Drive is not mounted.

Intermediate feature shards are excluded from the result ZIP to keep the
download tractable. They remain resumable in `OUTPUT_DIR`.
