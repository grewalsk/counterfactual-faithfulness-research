# Stage 31: cross-model grounded causal closure certificate

## Why this is the highest-EV next experiment

Stage 30 established a useful within-checkpoint fact: physically grounded
closure added held-out information about planning regret beyond ordinary
prediction error and self-consistent causal closure. It did not establish that
the result was method-general, or that the identified carrier was more than a
checkpoint-specific correlate.

For the paper, the most valuable uncertainty reduction is therefore a direct
model comparison on identical physical counterfactuals. A positive result
would turn Grounded Causal Closure from a JEPA-WM observation into a candidate
mechanistic reliability certificate. A null is also decisive: it would prevent
the paper from overclaiming universality and localize the effect to JEPA-WM.

This choice follows three lessons from the modern literature:

- The official [JEPA-WM study](https://arxiv.org/abs/2512.24497) reports that
  rollout fidelity and planning success need not coincide, and releases matched
  JEPA-WM and DINO-WM PushT checkpoints.
- Causal representation work such as
  [RAVEL](https://proceedings.iclr.cc/paper_files/paper/2025/file/180d2acb13633fe78688d0d2347c731f-Paper-Conference.pdf)
  distinguishes causal intervention success from observational decoding.
- Recent work on
  [interpretability without actionability](https://arxiv.org/abs/2603.18353)
  and [off-manifold intervention failures](https://arxiv.org/abs/2511.04638)
  motivates matched controls, construction/evaluation separation, and a
  physical rather than self-consistent target.

## Frozen experiment

The notebook uses the official public `jepa_wm_pusht` and `dino_wm_pusht`
checkpoints from the same pinned JEPA-WM repository commit. No secret is
required. An optional existing `HF_TOKEN` is used only to make public checkpoint
downloads more robust.

The physical design is shared, but the internal bases are not:

1. Generate 32 construction states and 120 evaluation states from disjoint,
   fresh trajectory-ID ranges.
2. On construction states only, screen all six predictor blocks separately for
   each model using action-contrast/output CKA.
3. Select the earliest block within one standard error of the best construction
   CKA, subject to frozen CKA and shuffle-control floors.
4. Fit a model-specific rank-128 output-aligned subspace with channel whitening
   and grouped ridge/SVD. Fit shuffled-output and empirical-span random controls.
5. Freeze both models' subspaces before evaluating either model.
6. Evaluate both models on the same 120 physical states and the same 24
   histogram-, impulse-, energy-, and duration-matched signed-area actions.

The four interior schedules are used for closure. The two excluded extreme
schedules are used as planning goals, so the primary planning contrast is not
reused to define closure.

## Exact objective and primary estimand

For model \(m\), candidate \(a\), and terminal goal \(g\), the score is the
official public objective

\[
c_m(a;g)=\operatorname{MSE}(\hat z^v_{m,a},z^{v,*}_{m,g})
+0.1\operatorname{MSE}(\hat z^p_{m,a},z^{p,*}_{m,g}).
\]

Grounded closure is measured in the joint Euclidean chart whose squared
distance is exactly this score. Physical normalized regret is always computed
from the simulator's block pose, not from a learned decoder.

The primary paired outcome for each state and magnitude is

\[
Y=R_{\mathrm{DINO}}-R_{\mathrm{JEPA}}.
\]

A five-fold, state-grouped out-of-fold ridge model first predicts \(Y\) from
magnitude, contact regime, the difference in ordinary joint prediction error,
and the differences in self-consistent causal coefficient/cosine. The test adds
only the difference in physically grounded coefficient/cosine.

The primary gate requires:

- at least 50 eligible contact states;
- at least 5% relative held-out MSE improvement;
- a state-bootstrap 95% interval for absolute MSE improvement whose lower bound
  is greater than zero.

Separate within-model tests require at least 1% relative held-out MSE
improvement with the same positive bootstrap condition. Construction gates,
matched ablation controls, visual-only sensitivity, causal-grounding-gap
replication, and exact free-motion nulls are reported separately.

## Interpretation boundary

A full positive result supports a cross-model *certificate*: differences in
physical grounded closure explain differences in planning reliability beyond
ordinary prediction error and self-consistent causal effects. It does not claim
that JEPA-WM and DINO-WM use a shared coordinate system, that architecture alone
causes the difference, or that the result already extends beyond PushT.

The notebook uses no decoder, probe, reader, gradient, Jacobian, JVP, or VJP.
It is an exhaustive terminal-action test, not yet a closed-loop CEM study. If
positive, the next paper-critical experiment is closed-loop intervention under
the official planner plus a third architecture or environment.
