# Stage 2C task-aligned readout protocol

## Motivation

Stage 2B produced a clean `NEGATIVE_SIGNAL`: aggregate paired latent error did
not predict executable physical regret beyond ordinary rollout error and
difficulty covariates. At the same time, the real models strongly outperformed
action-blind and action-shuffled controls on action-conditioned latent
prediction and improved physical action selection at horizons 3 and 6.

The present planner ranks predicted futures by Euclidean distance to one
encoded goal image. Stage 2C tests the narrow explanation that the world model
contains task-relevant physical state but raw latent geometry is not itself a
valid Push-T cost.

## Frozen design

- Simulator: Push-T with exact snapshot-and-restore branching.
- World models: `dino_wm_pusht` and `jepa_wm_pusht`, frozen.
- States: 300 in full mode.
- Candidates: the same 10 fixed, state-relative candidates from Stage 2B.
- Candidate selection never consults future simulator outcomes.
- Horizons: 3 and 6. Horizon 1 is excluded because 39.2% of Stage 2B states
  still made the no-op candidate oracle-optimal.
- Split: 50% probe training, 20% calibration, and 30% untouched final test,
  assigned by initial simulator state.
- All actions and horizons belonging to one state remain in the same split.

## Readout target

For each predicted future latent, decode the simulator-sufficient block pose:

`(block_x / 512, block_y / 512, sin(theta), cos(theta))`.

The physical Push-T cost is then computed deterministically from decoded pose.
The goal does not enter probe fitting, preventing a direct fixed-goal value
lookup.

Before probe fitting, predicted latent features are compressed through a saved
256-dimensional Gaussian random projection. A separate deterministic
projection is used for each checkpoint.

## Readouts and controls

- `latent_distance`: current raw predicted-latent to goal-latent distance.
- `linear_pose`: primary ridge pose decoder.
- `mlp_pose`: secondary one-hidden-layer pose decoder.
- `action_blind`: constant predicted cost, selecting candidate zero.
- `linear_pose_shuffled`: decoded costs shifted across action identities.
- `mlp_pose_shuffled`: decoded costs shifted across action identities.
- `oracle_pose`: exact simulator physical cost upper bound.

Ridge strength and MLP early stopping are selected using calibration pose MSE
only. Final test states are never used for fitting or hyperparameter selection.

## Primary outcomes

1. Physical normalized regret after executing the selected candidate.
2. Margin-weighted pairwise ranking accuracy.

Pair weights are absolute simulator cost margins, so consequential action
differences receive more weight and near-ties receive less.

For each task-aligned readout, compute paired improvements over
`latent_distance` for every `(state, model, horizon)` row. Resample whole initial
states 2,000 times in full mode.

## Decision rule

The linear probe earns `TASK_ALIGNED_SIGNAL` only if the state-clustered 95%
bootstrap lower bounds are above zero for both:

- normalized-regret improvement; and
- margin-weighted pair-ranking improvement.

If the linear probe does not pass but the MLP passes both conditions, label
`NONLINEAR_TASK_ALIGNED_SIGNAL`.

If both readouts have non-positive upper confidence bounds for regret
improvement, label `NO_TASK_ALIGNED_SIGNAL`. All remaining outcomes are
`INCONCLUSIVE`.

## Interpretation

- `TASK_ALIGNED_SIGNAL`: predicted latents contain linearly accessible physical
  dynamics, but raw latent-goal distance was the wrong planning readout.
- `NONLINEAR_TASK_ALIGNED_SIGNAL`: task state is recoverable but not linearly
  accessible under this compression and target.
- `NO_TASK_ALIGNED_SIGNAL`: the task-relevant physical state is not reliably
  recoverable from predicted latents under these probes.
- `INCONCLUSIVE`: uncertainty remains too large for the stated gate.

This is a simulator-only representational and planning diagnostic. It makes no
claim about real-robot reliability.
