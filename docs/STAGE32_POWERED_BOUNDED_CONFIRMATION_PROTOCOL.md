# Stage 32 powered bounded cross-model confirmation

## Decision rationale

Stage 31 replicated the incremental planning-reliability value of physically
grounded closure inside both JEPA-WM and DINO-WM. Its paired model-difference
test achieved 6.17% relative held-out MSE improvement, above the frozen 5%
effect threshold, but its state-bootstrap interval narrowly crossed zero.
Stage 31 also exposed six JEPA rows whose grounded target energy was effectively
zero, making coefficient ratios numerically meaningless. Removing those rows
strengthened the JEPA within-model result but did not change the clean paired
panel.

The official [JEPA-WM asset inventory](https://huggingface.co/facebook/jepa-wms/tree/main)
contains one public PushT checkpoint per model family. Stage 32 therefore does
not label artificial weight perturbations or truncated networks as independent
checkpoints. It instead provides the strongest honest confirmation available:
two distinct architectures, three action families, and a fourfold increase in
the persistent-contact state count.

## Frozen data and representation contract

- Import the exact Stage 31 JEPA and DINO rank-128 bases and channel metrics by
  hash. No refit, rotation, rank selection, or evaluation tuning is allowed.
- Generate 800 entirely new PushT candidate states from trajectory IDs
  5000–5799.
- Select the first 160 states with physical contact on every branch in all
  three action families: ±20°, ±30°, and ±40°.
- Each family contains four magnitudes and six impulse-, energy-, duration-,
  and histogram-matched signed-area schedules.
- Selection uses contact only. Model outputs and physical effect magnitudes are
  unavailable during selection.

For each model, state, family, and magnitude, Stage 32 records the baseline
prediction and four norm-matched swaps:

1. the Stage 31 primary rank-128 basis;
2. the shuffled-output rank-128 basis;
3. empirical-span random basis 0;
4. empirical-span random basis 1.

## Bounded closure

For an observed intervention effect \(e\) and encoded-physical target contrast
\(t\), the only grounded feature is

\[
\gamma(e,t)=\frac{\langle e,t\rangle}{\|e\|\,\|t\|}\in[-1,1].
\]

It is defined only when \(\|t\|^2\ge 10^{-6}\). Ineligible rows fail closed and
cannot enter any regression. No grounded coefficient ratio is computed or
used. Closure uses schedules 1–4; the two extreme schedules 0 and 5 remain
reserved as planning goals.

Planning uses the exact public terminal objective

\[
\operatorname{MSE}(\hat z^v,z^{v,*})
+0.1\operatorname{MSE}(\hat z^p,z^{p,*}),
\]

while normalized regret is computed from the exact simulator block pose.

## Primary gate

The primary outcome is DINO-WM minus JEPA-WM normalized regret. Five-fold
cross-fitting groups every family and magnitude from the same initial state.
The base features are magnitude, action geometry, ordinary joint target error
difference, and self-consistent causal cosine difference. The primary model
adds only the difference in bounded grounded cosine.

A full certificate requires:

- at least 140 eligible new persistent-contact states;
- at least 5% relative out-of-fold MSE improvement;
- a state-bootstrap 95% interval for absolute MSE improvement above zero;
- positive mean improvement in all three action families;
- a state-bootstrap interval above zero for primary improvement minus the
  median improvement from the three placebo subspaces;
- independent bounded within-model replication in JEPA-WM and DINO-WM.

The inference, cross-fit grouping, and bootstrap unit is always the initial
physical state. A positive result supports a bounded cross-model reliability
certificate. It does not yet establish causal necessity, full closed-loop MPC
value, environment generality, or a shared coordinate system between models.
