# Stage 34 predictive-fiber causal abstraction protocol

## Question and claim boundary

Stage 33 rejected the registered direct shared-realization hypothesis. Stage 34
therefore asks a narrower question: do the frozen JEPA-WM and DINO-WM PushT
checkpoints each separately realize the same simulator-defined, finite
action-response causal abstraction?

The experiment fits no JEPA-to-DINO state map. A positive result is bounded to
one deterministic PushT environment, one checkpoint per architecture, one
shared DINOv2 target family, the registered action-word panel, horizons at most
eight, and the selected predictor carrier at block 4. It does not establish
common neural circuitry, an infinite-horizon minimal state, architecture-family
generality, or planning value.

## Methodological basis

Predictive-state representations motivate defining state through controlled
future tests rather than an arbitrary latent coordinate system ([Littman,
Sutton, and Singh, 2001](https://proceedings.neurips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html)).
Exact and approximate causal abstraction motivate testing whether low- and
high-level interventions commute ([Rubenstein et al.,
2017](https://arxiv.org/abs/1707.00819); [Beckers, Eberhardt, and Halpern,
2020](https://proceedings.mlr.press/v115/beckers20a.html)). Interchange
interventions motivate requiring causal behavior from a decoded internal
variable rather than treating decodability as sufficient ([Geiger et al.,
2021](https://arxiv.org/abs/2106.02997)).

These papers motivate the estimands. Their general identification theorems do
not automatically apply to this finite deterministic experiment.

## Splits and response chart

Complete trajectories, not individual state records, are split before model
access:

| Split | Trajectories | Records | Role |
|---|---:|---:|---|
| construction | 16 | 64 | grounded decoders and simulator response chart |
| model selection | 16 | 64 | chart rank and transition regularization |
| calibration | 16 | 64 | frozen transition fits and carrier alignment |
| locked evaluation | 32 | 128 | all reported scientific gates |

Every complete trajectory contributes free, pre-contact, contact, and
post-contact records. Candidate pools are disjoint and selection uses physical
contact timing only.

For action word `w` of length `k`, model or simulator response is

\[
\Delta(h,w,j)=\phi(h,\operatorname{do}(w_{1:j}))
               -\phi(h,\operatorname{do}(0^j)),\qquad j=1,\ldots,k.
\]

The signature concatenates every valid `Delta` path and registered terminal
order contrasts `Delta(h,u)-Delta(h,v)` for fixed-multiset pairs. Construction
uses words of lengths 1--4. Evaluation uses new compositions and magnitudes of
lengths 5--8. A standardized PCA chart is fitted to construction simulator
signatures only; its effective rank is selected on model-selection simulator
signatures only. Model outputs never select the canonical chart or its rank.

## Sequential gates

Later expensive gates run only when every preceding gate passes.

### Gate 1: unseen-action specificity

On locked length-5--8 words, each model's grounded response signature must beat
both a within-length action-word shuffle and a calibration-fitted static
physical-state predictor. Mean action-shuffle relative advantage must be at
least 0.10, its trajectory-clustered interval lower endpoint must exceed zero,
the static-state advantage must be positive, and action advantage must be
positive in every contact mode.

Failure label: `SHARED_STATIC_STATE_GEOMETRY_ONLY`.

### Gate 2: predictive sufficiency

A fixed-width nonlinear transition predictor receives canonical response state
and a registered five-coordinate action summary. A capacity-matched enriched
predictor additionally receives a fixed 64-coordinate carrier sketch. Model
selection chooses regularization; calibration fits coefficients; locked
length-5--8 rows are scored by whole-trajectory bootstrap.

Residual relative improvement must be at most 0.05, its interval upper endpoint
at most 0.10, and every mode at most 0.10. As a non-vacuity control, deleting a
real response-state coordinate must worsen error by at least 0.10 with a
positive interval lower endpoint.

Failure label: `CANDIDATE_PREDICTIVE_STATE_INSUFFICIENT`.

### Gate 3: on-manifold causal fibers

Calibration-only supervised carrier directions align block-4 zero-action
carriers with canonical response coordinates. Locked records are matched
within contact mode and across trajectories in two ways: similar response state
with different residual carrier (`fiber`), and different response state with
similar residual carrier (`state`). Exactly eight base records per mode enter
each panel.

For every pair, the base-to-donor carrier difference is split into aligned and
orthogonal components in the standardized carrier metric. The registered edit,
an equal-energy matched-rank control-subspace edit, and the full same-model swap
are injected through the real recurrent predictor hook. Patched carrier sketches
must remain inside the natural leave-one-out 95% neighborhood, allowing at most
5% OOD rows.

The aligned state edit must retain at least half the full-swap error gain, have
mean intended-effect cosine at least 0.20, and beat the matched control by at
least 0.10. The full-swap gain and per-mode retentions must be positive. The
fiber edit's effect norm may be at most 1.25 times its paired donor effect.

Failure label: `PREDICTIVE_SUMMARY_NOT_CAUSALLY_USED`.

### Gate 4: two-sided commutativity

One simulator-only high-level transition is selected on simulator
model-selection rows and fitted on simulator calibration rows. It is applied
separately to JEPA response coordinates and DINO response coordinates. No
parameter maps one checkpoint into the other. Each model's transition defect
must remain within 1.25 times the registered physical/model reference error
budget, beat an action-shuffled control by at least 0.10 with a positive
clustered interval, and remain bounded in every contact mode and word-length
family.

Failure label: `MODELS_DO_NOT_SHARE_HIGH_LEVEL_TRANSITION`.

## Decision and interpretation

`BOUNDED_TWO_SIDED_CAUSAL_ABSTRACTION_SUPPORTED` is available only to a
source-bound pilot passing all four gates, all controls, and every family
consistency check. Smoke runs always return `SMOKE_ONLY`. Source, split, asset,
or execution failures return `INCONCLUSIVE_SOURCE_OR_SPLIT_FAILURE` and are not
scientific negatives.

Planning is not run in Stage 34. If the full result passes, a separately
preregistered Stage 35 can test whether the abstraction preserves action
ranking or planning value without reopening this evaluation panel.
