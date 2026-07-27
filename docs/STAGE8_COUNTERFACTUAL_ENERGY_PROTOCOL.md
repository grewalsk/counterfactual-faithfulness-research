# Stage 8 counterfactual decision-energy protocol

## Status

Stage 8 is an exploratory method-development experiment. It reuses the
previously inspected Stage 5/7 task family and the same
`development_holdout`; it is not confirmatory evidence. A successful recipe
may only be frozen and nominated for later evaluation on numerically new
tasks.

## Motivation

Stage 7 returned `NO_RECURRENT_DEVELOPMENT_GAIN`. Its recurrent
counterfactual adapter reduced ordinary latent error on every development
state in both environments, but did not reliably improve action selection.
In Wall, ordinary latent error fell by about 12 percent while regret and
weighted pairwise accuracy became worse. At the same time, the layer audit
decoded Wall physical effects with development R-squared as high as 0.978.

Those results distinguish two objects:

1. the predicted latent state, which can become more accurate under a
   reconstruction-style objective; and
2. the planning energy used to order candidate actions, whose geometry can
   remain misaligned with physical task cost.

Stage 8 therefore leaves the public JEPA-WM dynamics exactly frozen and learns
only a small decision-energy residual.

## Fixed scope

- simulator-only PushT and Wall;
- exact state restoration and the fixed Stage 5 candidate libraries;
- public `jepa_wm_pusht` and `jepa_wm_wall` checkpoints;
- horizons 1, 3, and 6;
- ten candidate action sequences per restored state, including no-op;
- 12 tasks per environment split into six training, three calibration, and
  three development-holdout tasks;
- 96 states per environment in the full run;
- no real-robot data or execution.

The notebook may reuse a compatible Stage 7 simulator/transition-token cache.
Compatibility is checked from the Stage 7 configuration and required shard
counts. A missing or incompatible cache triggers a complete reconstruction
under the Stage 8 output directory.

## Energy features

Every frozen rollout produces one feature set for each state, horizon, and
candidate action. The feature vector contains:

- native visual and proprioceptive goal distances;
- multiscale spatial maps of patch-token goal error;
- fixed random projections of signed and squared channelwise goal error;
- no-op-relative CountSketch outputs from all six AdaLN predictor blocks and
  the final predictor output;
- the normalized action chunk for the evaluated model step.

All candidate features are centered within the same state and horizon before
training. Standardization statistics are fitted only on training tasks.

## Energy model and objective

The energy head is a small shared MLP with a zero-initialized output layer. For
the final-token and proposed methods it begins as the exact native JEPA goal
energy and learns a residual:

`E_phi(a) = standardized_native_energy(a) + residual_phi(features(a))`.

Training uses only within-state comparisons. Physical simulator costs are
normalized by the candidate-set spread and supervise:

- a margin-weighted pairwise logistic ordering loss;
- a listwise soft-target cross-entropy loss;
- a low-weight centered cost-shape loss.

The world-model encoder and predictor receive no gradient and their cached
predictions are identical for every energy method.

## Controlled methods

All trained methods use identical head architecture, optimizer, minibatch
order, training examples, head seeds, and checkpoint epochs. Feature masks or
alignment differ:

1. `final_token_energy`: native goal distance plus final-token goal features,
   without intermediate action-effect features;
2. `action_prior_control`: normalized action chunks only, with no world-model
   or goal features;
3. `wrong_state_control`: the proposed world-model features, but rotated among
   different states of the same task while labels stay fixed;
4. `counterfactual_energy`: native/final goal features plus correctly aligned
   no-op-relative features from every audited predictor layer.

`native_world_model` is the untrained public JEPA latent goal distance.

The action-prior control measures structure introduced by the fixed
task-relative candidate library. The wrong-state control preserves task and
candidate identity while destroying state-specific world-model alignment.

## Selection and evaluation

Heads train on `probe_train` tasks. Checkpoints are selected at fixed epochs
using a calibration-only score combining normalized regret and weighted
pairwise accuracy. The development holdout is not used for standardization,
optimization, early stopping, architecture choice, or checkpoint selection.

Selected heads are evaluated once on the development holdout. Primary metrics
are:

- normalized planning regret;
- margin-weighted pairwise accuracy;
- top-1 action accuracy;
- normalized decision-margin error.

Uncertainty is bootstrapped by restored initial state. The notebook also
exports task-level descriptive contrasts because only three development tasks
exist per environment; no confirmatory population inference is claimed.

## Development decision

For an environment, the base gate requires the proposed energy to improve both
normalized regret and weighted pairwise accuracy over
`native_world_model`, with both 95 percent cluster-bootstrap lower bounds above
zero. Both head seeds must have nonnegative mean direction on both primary
metrics.

Specificity requires at least one primary planning metric with a positive
lower bound against each of `final_token_energy`, `action_prior_control`, and
`wrong_state_control`.

- `DECISION_ENERGY_CANDIDATE_READY`: both environment/base gates and all
  specificity gates pass;
- `DECISION_ENERGY_GAIN_NOT_SPECIFIC`: both environment/base gates pass but
  specificity fails;
- `MIXED_DECISION_ENERGY_SIGNAL`: the complete base gate passes in one
  environment;
- `NO_DECISION_ENERGY_GAIN`: neither complete base gate passes;
- `INCONCLUSIVE`: execution or integrity failure prevents interpretation.

These labels guide exploratory development only.

## Required output

The notebook exports:

- configuration, versions, cache reuse, task, split, restoration, candidate,
  and checkpoint manifests;
- energy-feature and training manifests;
- epoch histories and calibration checkpoint tables;
- state-level metrics and action-level predicted energies;
- state-clustered method contrasts and task-level descriptive contrasts;
- diagnostic plots, decision JSON, logs, and failure trace;
- trained energy-head checkpoints;
- one automatically downloaded `stage8_result_bundle.zip`.
