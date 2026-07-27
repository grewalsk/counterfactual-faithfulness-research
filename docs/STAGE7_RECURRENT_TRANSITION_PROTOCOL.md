# Stage 7 recurrent counterfactual transition protocol

## Status

Stage 7 is an exploratory method-development experiment. It reuses the
previously inspected Stage 5 task family and therefore renames the former final
partition `development_holdout`. No Stage 7 result is confirmatory. A positive
result may only nominate a frozen recipe for evaluation on numerically new
tasks.

## Motivation

Stage 6 trained terminal readouts over pooled features after a complete
JEPA-WM rollout. Its short-horizon component gains disappeared or reversed at
horizon six. The public JEPA-WM inference path, however, applies an
action-conditioned AdaLN predictor recurrently: each predicted visual latent
is reused as the context for the next action. A terminal readout cannot repair
action distinctions that vanished during earlier recurrent steps.

Stage 7 tests two linked hypotheses:

1. physical action-effect information is present in unpooled intermediate
   AdaLN tokens but weakens across predictor layers or recurrent steps;
2. a small transition residual applied before recurrent feedback can preserve
   the same-state counterfactual structure and improve native latent-space
   action ranking.

## Fixed scope

- simulation only: PushT and Wall;
- exact state restoration and the Stage 5 fixed candidate library;
- public `jepa_wm_pusht` and `jepa_wm_wall` checkpoints;
- horizons 1, 3, and 6, with targets retained at every intermediate step;
- ten action sequences per initial state, including a fixed no-op;
- frozen public DINO visual encoder and JEPA-WM predictor;
- no real-robot data or execution.

The full development run uses 96 states per environment. The 12 tasks retain
the Stage 5 split counts: six training, three calibration, zero regression,
and three development-holdout tasks.

## Layerwise audit

For every state, action, and rollout step, the notebook records:

- the unpooled 16 by 16 predicted visual-token grid;
- outputs of all six AdaLN predictor blocks;
- the final predictor output;
- the corresponding encoded true future.

For memory control, no-op-relative flattened token differences are compressed
with deterministic CountSketch projections. Task-disjoint ridge probes are fit
on training tasks, select regularization on calibration tasks, and report
physical-effect prediction on the development holdout. Multiple projection
seeds quantify projection sensitivity.

The audit is diagnostic and does not select the adapter architecture or touch
the development holdout during model selection.

## Recurrent residual

Let the frozen predictor produce

`z_base(t+1, a) = P(z_t, a_t)`.

A shared token residual receives the base transition difference, normalized
action, spatial position, and rollout-step embedding:

`z_corrected(t+1, a) = z_base(t+1, a) + R_phi(...)`.

The corrected latent is inserted after every `forward_pred` call and becomes
the context for the next recurrent step. The adapter operates on all 256 patch
tokens; there is no spatial pooling before correction.

The final residual layer is zero initialized so every method begins as the
exact frozen world model.

## Training conditions

All adapter conditions use identical examples, initialization, minibatch
order, optimizer, maximum epochs, and checkpoint epochs.

1. `absolute_residual`: weighted latent prediction loss only;
2. `independent_delta_control`: absolute loss plus a delta target anchored to
   the no-op from another state;
3. `counterfactual_recurrent`: absolute loss plus the correct same-state,
   no-op-relative latent-difference loss.

True counterfactual token magnitude defines a bounded token weight so localized
physical changes are not overwhelmed by static background. No ranking loss
backpropagates into the transition adapter.

Checkpoints are selected on calibration tasks using latent prediction,
counterfactual latent error, native latent-space planning regret, and weighted
pairwise accuracy. The development holdout is evaluated only after selection.
The expensive adapter-in-the-loop recurrent evaluation is run only on that
development holdout; calibration is not re-reported as evaluation evidence.

## Recurrent evaluation

Selected adapters are inserted into the actual autoregressive inference loop.
Candidate actions are scored using the public model's native visual plus
proprioceptive latent goal distance. Simulator physical costs are used only to
measure:

- normalized planning regret;
- margin-weighted pairwise accuracy;
- top-1 action accuracy;
- normalized decision-margin error.

The notebook also reports ordinary and paired latent errors.

The primary comparison is `counterfactual_recurrent` versus the frozen base
world model. Specificity comparisons use `absolute_residual` and
`independent_delta_control`. Uncertainty is clustered by restored initial
state.

## Development labels

- `RECURRENT_COUNTERFACTUAL_CANDIDATE_READY`: the proposed adapter improves
  both planning metrics over the frozen model in both environments, passes the
  ordinary-latent-error noninferiority margin, and shows specificity over both
  trained controls;
- `RECURRENT_GAIN_NOT_SPECIFIC`: both environment/base gates pass, but
  specificity does not;
- `MIXED_RECURRENT_SIGNAL`: the complete base gate passes in one environment;
- `NO_RECURRENT_DEVELOPMENT_GAIN`: the complete base gate passes in neither;
- `INCONCLUSIVE`: an integrity or execution failure prevents interpretation.

These labels guide development only.

## Required output

The notebook exports:

- configuration, version, checkpoint, task, split, restoration, and candidate
  manifests;
- layerwise audit rows and audit gate;
- adapter training histories, selected checkpoints, and training manifest;
- unit-level and action-level recurrent evaluation tables;
- clustered method contrasts and latent-error noninferiority intervals;
- plots, logs, decision JSON, and failure trace;
- one automatically downloaded `stage7_result_bundle.zip`.

Large simulator, token, and model caches are resumable but excluded from the
downloaded bundle.
