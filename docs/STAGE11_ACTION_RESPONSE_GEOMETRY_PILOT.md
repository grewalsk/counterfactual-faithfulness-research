# Stage 11: compute-gated action-response geometry pilot

## Question

Stage 11 asks whether the remaining JEPA-WM planning failures are caused by
incorrect *relative latent responses to alternative actions*. It is a
low-compute falsification pilot, not a confirmatory experiment.

Stages 7 and 9 showed that ordinary latent error can improve without reliable
action selection. Stage 10 showed that optimizing the margins of a small set
of physical readouts does not reliably transfer to new readouts and can be
matched by a shuffled-outcome control. The next intervention therefore trains
neither an endpoint decoder nor a decision head.

## Objective

For state \(s\), action \(a\), horizon \(h\), and fixed CountSketch projection
\(P_r\), let

\[
g^\theta_{sahr}
= P_r\hat z^\theta_{sah}
- \frac{1}{A}\sum_{b=1}^{A}P_r\hat z^\theta_{sbh},
\]

and define \(g^*_{sahr}\) analogously from target JEPA tokens. A whitening
operator is fitted on probe-training targets only:

\[
W_{hr}
= \left(\widehat\Sigma_{hr}+\lambda I\right)^{-1/2}.
\]

The matched Action-Response Geometry Adaptation loss is

\[
\mathcal L_{\mathrm{ARGA}}
= \mathbb E_{s,a,h,r}
\left\|W_{hr}
\left(g^\theta_{sahr}-g^*_{sahr}\right)\right\|_2^2.
\]

Centering removes any error shared by all candidate actions. Whitening gives
equalized weight to target action-response directions instead of letting the
largest-variance coordinates dominate.

The shuffled control uses the same states, candidates, optimizer, parameter
scope, projections, epochs, and checkpoint opportunities, but applies a
deterministic non-null within-state derangement to target action identity. The
latent-only control optimizes ordinary target-token error. All treatments
update only the action encoder and six AdaLN modulation maps.

No physical pose, task goal, physical cost, fitted readout, or selected-action
label enters the ARGA objective or checkpoint score. Physical decoders are
fitted only for post-training evaluation.

## Conditional mathematical guarantee

For a linear score \(f_w(z)=w^\top z\), a candidate pair \(a,b\), and positive
definite \(\Sigma\),

\[
\begin{aligned}
&\left|
\left[f_w(\hat z_a)-f_w(\hat z_b)\right]
-\left[f_w(z^*_a)-f_w(z^*_b)\right]
\right|\\
&\quad=
\left|
(\Sigma^{1/2}w)^\top
\Sigma^{-1/2}
\left[
(\hat z_a-\hat z_b)-(z^*_a-z^*_b)
\right]
\right|\\
&\quad\le
\|\Sigma^{1/2}w\|_2
\left\|
\Sigma^{-1/2}
\left[
(\hat z_a-\hat z_b)-(z^*_a-z^*_b)
\right]
\right\|_2.
\end{aligned}
\]

The inequality is Cauchy--Schwarz. Because centering preserves every pairwise
difference, reducing whitened centered-response error controls margin error
for every bounded linear readout in the projected space at once.

This guarantee is deliberately limited. It does not cover nonlinear planners,
unbounded readouts, directions discarded by projection, or distribution
shift. Stage 11 therefore evaluates independent projections and retains a
two-percent per-horizon native latent-fidelity constraint.

## Compute gate

The default `RUN_MODE="pilot"` uses:

- 36 exact states per environment;
- 12 tasks split 6/3/0/3 into probe training, calibration, unused, and
  development holdout;
- 10 candidate action sequences and horizons 1, 3, and 6;
- two fixed training projections;
- three unseen evaluation projections;
- one screening seed and at most one confirmation seed;
- checkpoints every two epochs, with early stopping and a maximum of ten
  epochs per treatment.

The confirmation seed runs only if at least one environment improves matched
calibration geometry over frozen JEPA by at least three percent and over the
shuffled control by at least one percent at two of three horizons, with no
environment more than five percent worse than frozen.

This is a necessary-condition screen. If it fails, the notebook still
evaluates and packages the screening seed but does not spend the confirmation
seed.

## Promotion rule

The pilot promotes to `RUN_MODE="full"` only if:

1. the screening gate runs the second seed;
2. matched ARGA preserves the two-percent native-fidelity bound;
3. in both environments, matched ARGA improves unseen-projection geometry over
   frozen and shuffled controls at two of three horizons in at least two of
   three projections;
4. in both environments, fresh physical readouts improve both normalized
   regret and weighted pairwise accuracy over frozen and shuffled controls at
   two of three horizons in at least two of three projections; and
5. the native goal-latent planner is not materially harmed.

Pilot promotion uses directional task-equal point estimates because only three
development tasks are available. It is an allocation decision, not a
statistical claim. The full matrix raises the state, seed, epoch, projection,
and bootstrap budgets. A later numerically new task family remains necessary
for confirmation.

## Interpretation

- `PROMOTE_TO_FULL_RUN`: the relative action geometry changed in a
  label-specific way and transferred to fresh decisions.
- `GEOMETRY_ONLY_DIAGNOSIS`: the action-conditioned transition geometry
  improved, but a fresh readout still could not use it; the remaining
  bottleneck is representation/readout alignment.
- `STOP_NO_DIRECT_GEOMETRY_SIGNAL` or
  `STOP_NO_ROBUST_UNSEEN_GEOMETRY_GAIN`: do not spend full-run compute on this
  method.
- `STOP_NATIVE_FIDELITY_FAILURE`: the adaptation is unsafe relative to the
  frozen transition model.

The notebook writes an atomic latest checkpoint every two epochs, downloads a
compact Phase-C rescue ZIP after adaptation, and downloads the final result
bundle after evaluation. Large simulator and token caches are excluded.
