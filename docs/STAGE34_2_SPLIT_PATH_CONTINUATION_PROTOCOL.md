# Stage 34.2 split-path predictive and causal continuation protocol

## Evidence status and rationale

Stage 34.1 established a sharp model asymmetry after removing the original
action-indexed-column leak. JEPA beat the leakage-free state-only predictor and
the within-length derangement. DINO beat derangement but was worse than the
state-only physical prior in every trajectory average.

Stage 34.2 is a post-outcome diagnostic continuation. It does not reopen a
two-model gate. DINO receives a calibration-only error decomposition, while
JEPA alone advances through the previously unopened Stage 34 predictive-
sufficiency and native causal-use questions. All conclusions remain
confirmation-ineligible.

## Bound inputs

The experiment consumes only:

- Stage 34 pilot
  `d3f4f88426afff4d964bb4f1f1556c94ec3613b667edd9403ddfcd0fd78ded84`;
- Stage 34.1 diagnostic
  `208b72f570749a48719ddf22d693cde5ee0fd2c8b525021506d356cd6242a8ac`;
- the official JEPA-WM PushT checkpoint and shared DINOv2 target encoder under
  the exact Stage 34 repository, revision, and asset hashes.

The notebook verifies both upstream manifests, source commits, decisions,
split identities, and every consumed raw shard before analysis. The 16
model-selection, 16 calibration, and 32 evaluation trajectories remain
disjoint.

## DINO calibration diagnosis

Calibration uses no evaluation rows. The input consists of DINO's
no-op-corrected grounded response rows for the Stage 34 length-1--4 core words
on the 16 calibration trajectories. The target is the matching 11-dimensional
simulator response.

The primary diagnostic map has only 22 parameters:

\[
\widehat y_j = a_j x_j + b_j, \qquad j=1,\ldots,11.
\]

Its ridge penalty is selected by four-fold trajectory-grouped out-of-fold MSE.
The fitted map is then frozen and applied to length-5--8 responses on the 32
evaluation trajectories. A full 11-by-11 affine map is reported descriptively
but can never determine the diagnosis.

`DINO_DIAGONAL_RECOVERABILITY` requires the diagonal map to:

- improve over raw DINO by at least 10% with clustered lower confidence bound
  above zero;
- beat the frozen Stage 34.1 leakage-free state error with lower confidence
  bound above zero and positive advantage in every contact mode; and
- beat a calibrated same-length word derangement by at least 10%, again with a
  positive clustered lower bound and positive advantage in every mode.

A pass localizes the observed failure to low-capacity per-observable
calibration. It is not evidence of causal use.

## JEPA predictive sufficiency

This gate is exactly the Stage 34 gate that remained unopened. A width-256
random-Fourier transition predictor receives the rank-five response state and
the registered five-coordinate action summary. The enriched predictor adds
the frozen 64-coordinate carrier sketch; the deletion control removes the
last real response-state coordinate. Model selection chooses ridge penalty,
calibration fits coefficients, and evaluation contains length-5--8 words.

The response state is sufficient only when:

- mean residual-carrier improvement is at most 5%;
- its trajectory-clustered interval upper endpoint is at most 10%;
- every contact-mode improvement is at most 10%; and
- real-coordinate deletion worsens prediction by at least 10%, with lower
  confidence endpoint above zero.

Five thousand clustered bootstrap draws preserve the original Stage 34 pilot
contract. If this gate fails, all checkpoint loading and causal inference are
skipped.

## JEPA native causal use

Conditional on sufficiency, calibration carriers fit a rank-five supervised
subspace aligned with the canonical response state. Evaluation matching
freezes eight trajectory-distinct base records per contact mode for each of
two panels:

- similar response state with different residual carrier (`fiber`); and
- different response state with similar residual carrier (`state`).

For a zero-action word, the live block-4 carrier difference is separated into
aligned and fiber components. Each pair receives the registered primary edit,
an equal-energy residual-subspace control, and a same-model full swap. Live
unpatched predictions and carriers must reproduce the Stage 34 cached shards
within `5e-4` maximum absolute error before the intervention is accepted.

The frozen Stage 34 causal thresholds are retained: positive full-swap gain in
every mode, at least 50% state-effect retention, mean intended-effect cosine at
least 0.20, control advantage at least 0.10, fiber-effect ratio at most 1.25,
positive retention in every mode, and at most 5% primary edits outside the
natural 95% carrier neighborhood.

Each pair writes an independently hashed JSON shard. A disconnected Colab
runtime can resume by selecting **Run all** without recomputing completed
pairs.

## Decisions

- `JEPA_RESPONSE_STATE_INSUFFICIENT`: stop before native interventions.
- `JEPA_RESPONSE_STATE_NOT_CAUSALLY_USED`: observational sufficiency passed,
  but the matched native intervention gate failed.
- `JEPA_CAUSAL_STATE_DINO_CALIBRATION_LIMITED`: JEPA passed the three-step
  diagnostic chain and DINO was recovered by the diagonal calibration.
- `JEPA_ONLY_CAUSAL_STATE_DINO_NOT_CALIBRATION_RECOVERABLE`: JEPA passed while
  DINO's defect was not explained by low-capacity scale/bias.

None of these decisions establishes a shared abstraction or an independent
confirmation. A positive JEPA outcome motivates a fresh-trajectory,
single-model confirmation. A positive DINO calibration result motivates a
separate fresh calibration-transfer test before any DINO causal experiment.
