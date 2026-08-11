# Stage 34.1 leakage-free action-specificity repair protocol

## Purpose and evidence status

Stage 34 stopped at its first scientific gate even though both frozen world
models beat a within-length action shuffle. The failure was caused by the
second comparator: exact physical state was regressed onto one concatenated
response atlas. Each output coordinate permanently denoted a particular action
word and prefix step, so target-column position exposed action identity.

Stage 34.1 is a post-outcome diagnostic repair. It preserves the original
Stage 34 decision and consumes only its hash-bound calibration and evaluation
artifacts. It is not an independent confirmation. A pass authorizes continued
testing of the still-unobserved Stage 34 sufficiency and causal questions; it
does not itself support the full causal-abstraction claim.

## Frozen upstream run

The only eligible upstream input is Stage 34 pilot run
`d3f4f88426afff4d964bb4f1f1556c94ec3613b667edd9403ddfcd0fd78ded84`
at source commit `db130a3d25505b7fa69efbcd88009365cb266688`.

Before analysis, the notebook verifies:

- the audited raw-manifest digest;
- the upstream protocol and notebook digest;
- the source commit and complete run signature;
- confirmation eligibility and `FAILURE_TRACE = NONE`;
- the model-free calibration/evaluation split manifests;
- trajectory disjointness; and
- byte count and SHA-256 for every consumed truth and model shard.

The full Stage 34 Drive directory is required. The compact returned bundle is
not sufficient because it intentionally excludes the raw shards.

## Leakage-free row estimand

For each record, held-out action word `w` of length `k`, and prefix step `j`,
the target is the 11-dimensional no-op-corrected physical response

\[
Y(h,w,j)=\phi(h,\operatorname{do}(w_{1:j}))
         -\phi(h,\operatorname{do}(0^j)).
\]

One action prefix is one statistical row. The target width is always 11; it
does not grow with the number of action words. The state-only comparator sees

\[
X_0=(\phi(h), k, j, j/k, \text{contact-mode one-hot}),
\]

and never receives `w`, an action vector, an action-indexed output block, or an
action-derived ordering. Consequently, two words at the same state, length,
and prefix step have exactly identical `X_0`. The notebook executes this as an
invariant and requires the resulting prediction spread to be at most `1e-12`.

## Capacity-matched physical control

Two width-256 random-Fourier ridge regressions are fitted on the frozen 16
calibration trajectories with trajectory-grouped four-fold penalty selection:

1. state-only, using `X_0`; and
2. state-plus-action, adding cumulative impulse, energy, signed area, current
   macro action, and cumulative action-path length.

Both predict the same 11-dimensional row target and use the same candidate
ridge grid. On the 32 locked evaluation trajectories, the state-plus-action
model must improve record-level MSE over the state-only model by at least 10%,
with a trajectory-clustered 95% interval above zero and positive improvement
in every contact mode. This is a design positive control showing that the
frozen physical panel actually requires action information.

## Model action-specificity gate

JEPA and DINO are scored separately. For each model, the primary decoded path
must beat both:

- a deterministic derangement that replaces every predicted word path with a
  different word path of the same length; and
- the repaired state-only physical predictor.

The derangement advantage must average at least 10%. Both advantages must have
trajectory-clustered 95% lower endpoints above zero and must be positive in
free, pre-contact, contact, and post-contact strata. All metrics are first
computed per complete record; resampling clusters whole trajectories.

## Decision

`ACTION_SPECIFICITY_REPAIRED_CONTINUE_STAGE34` requires upstream binding, the
executable no-leakage invariant, physical action necessity, and both model
gates. It means only that action-specific prediction survived a fair
action-blind comparator on this reused panel.

`ACTION_SPECIFICITY_NOT_ESTABLISHED` means at least one diagnostic gate failed.
`INCONCLUSIVE_UPSTREAM_BINDING_FAILURE` is an integrity failure, not a
scientific negative. Smoke mode always returns `SMOKE_ONLY`.

Regardless of outcome, Stage 34.1 records `confirmation_eligible = false`.
A fresh, separately frozen confirmation is required before turning the repair
into a publication-level positive claim.
