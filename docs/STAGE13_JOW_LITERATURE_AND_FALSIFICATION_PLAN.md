# Stage 13 Jacobian Outcome Workspace: literature basis and falsification plan

**Status: literature and test-design memo. This is not a frozen protocol.**

## Bottom line

There is no published demonstration of a J-space-like causal workspace inside
an action-conditioned JEPA-WM. The hypothesis is therefore novel and could be
false.

There is, however, convergent evidence for each prerequisite:

1. transformers can contain small output-causal intermediate subspaces;
2. visual transformers can repurpose tokens for internal computation and admit
   sparse patch-level features;
3. action-conditioned JEPAs can separate transformation-sensitive structure
   from invariant structure;
4. long-term policy dynamics can admit low-rank predictive factorizations;
5. control-relevant JEPA representations can be much smaller than the original
   visual embedding;
6. interchange interventions provide a rigorous way to distinguish a causal
   internal variable from a decodable correlate.

That evidence makes a small, training-free causal pilot scientifically
justified. It does not justify assuming the workspace exists.

## Evidence map

### Direct conceptual precedent: output-causal sparse subspaces

[Verbalizable Representations Form a Global Workspace in Language
Models](https://transformer-circuits.pub/2026/workspace/index.html) derives
layer-specific directions from averaged Jacobians between intermediate
activations and future output logits. Sparse combinations of those directions
explain little total activation variance but causally mediate some flexible
reasoning and can be reused by several downstream computations.

What transfers:

- define a direction through its causal effect on a model-native output;
- look for sparse occupancy rather than high explained variance;
- require coordinate swaps and ablations;
- require flexible reuse and layer localization.

What does not transfer:

- JEPA-WM has no token vocabulary or unembedding matrix;
- physical outcomes are continuous, spatial, relational, and signed;
- the six-block predictor is much shallower than the studied language models;
- JEPA-WM may simply distribute outcome computation across its token grid.

This paper supplies the method-level analogy, not evidence about JEPA itself.

### Global workspaces are computationally viable

[Coordination Among Neural Modules Through a Shared Global
Workspace](https://openreview.net/forum?id=XzTtHjgPDsT) shows that an explicitly
bandwidth-limited shared workspace can coordinate structured neural modules and
encourage specialization and compositionality.

This establishes that workspace-style bottlenecks are useful computational
objects. Because that workspace is designed into the architecture, it does not
show that JEPA-WM learned one spontaneously.

### Vision transformers create internal-computation tokens

[Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) finds
that supervised and self-supervised vision transformers produce high-norm
tokens in uninformative image regions and repurpose them for internal
computation. Adding explicit register tokens lets the model move that
computation into designated storage.

This supports the general possibility that a vision transformer uses a
restricted representational substrate for nonlocal computation. Registers are
not by themselves a causal workspace, and the DINOv2-S encoder or JEPA
predictor used here may behave differently.

### Sparse visual features can be recovered patch-wise

[PatchSAE](https://arxiv.org/abs/2412.05276) recovers sparse, interpretable
features and spatial patch attributions from CLIP vision-transformer
activations. It also finds that many adaptation gains remap existing features
rather than create wholly new ones.

This supports:

- sparse feature structure in visual-transformer activations;
- preservation of spatial tokens;
- comparing frozen and adapted models in a shared dictionary.

Sparse-autoencoder reconstruction or interpretability is still correlational.
An SAE is therefore a baseline for JOW, not evidence of JOW.

### Action-conditioned JEPA can separate invariant and equivariant structure

[seq-JEPA](https://openreview.net/forum?id=GKt3VRaCU1) reports an
architecturally separated invariant aggregate representation and an
action-equivariant encoder representation. Its ablations show that action
conditioning is important for equivariance and that relatively small action
embeddings can capture transformation structure.

This supports the possibility that action-conditioned predictive training
organizes transformation information into a compact component. seq-JEPA has a
different architecture and objective; its separation cannot be assumed to
exist in JEPA-WM's AdaLN blocks.

### Long-term JEPA dynamics can be low-rank and reusable

[TD-JEPA](https://arxiv.org/abs/2510.00739) gives the strongest theoretical
support. In an idealized linear setting, its learned encoders recover a
low-rank factorization of long-term policy dynamics, while the predictor
recovers successor features usable for zero-shot optimization across reward
functions.

This is close to the desired "broadcast" property: one predictive
representation can support multiple downstream tasks. The result concerns a
purpose-trained TD-JEPA, not the internal layers of the frozen JEPA-WM
checkpoint, so it supports plausibility rather than existence.

### JEPA control-sufficient spaces can be substantially smaller

[Learning Invariant Visual Representations for Planning with
JEPA-WMs](https://arxiv.org/abs/2602.18639) adds a bisimulation encoder that
preserves transition-relevant state equivalence while suppressing slow visual
features. It reports planning-robust embeddings with a per-patch dimension ten
times smaller than the original DINO-WM features.

This directly supports the proposition that total visual-embedding variance is
not the same as control relevance and that compact transition-sufficient
representations are possible. The compact space is learned explicitly and is
not shown to be an emergent intermediate workspace.

### Interchange interventions are the right causal test

[Causal Abstractions of Neural
Networks](https://arxiv.org/abs/2106.02997) distinguishes structural decoding
from causal implementation. It aligns a hypothesized high-level variable with
a neural representation, swaps that representation between inputs, and tests
whether the network behaves as if the high-level variable had been
intervened on.

This maps exactly onto JOW:

- high-level variable: the counterfactual physical outcome of an action;
- source and recipient: two actions executed from the identical state;
- internal intervention: swap only the hypothesized outcome coordinates;
- required output: continuation behaves like the donor action's outcome.

Recent work on [divergent representations from causal
interventions](https://openreview.net/pdf/f608b52d1638ab06c9c0f72be594048e574c9484.pdf)
warns that activation edits can leave the model's natural activation
distribution and trigger misleading behavior. The JOW test must therefore use
interpolated swaps, activation-distribution checks, and norm-matched controls.

## How strong is the prior?

The literature supports three nested hypotheses:

### H0: distributed computation

Action-outcome information is spread across tokens, channels, and layers.
Sparse directions can decode it, but no small component is causally privileged.

### H1: sparse outcome representation

A small dictionary reconstructs or decodes outcome effects well, but its
directions are not disproportionately causal and are not flexibly reused.

### H2: workspace-like outcome representation

A small intermediate component:

- is written by executable actions;
- causally determines predicted outcome identity;
- can be swapped between same-state actions;
- is read coherently by multiple downstream goal functions;
- appears in a reproducible layer band.

Only H2 supports the Jacobian Outcome Workspace claim. Existing literature
raises the prior probability of H1 and H2 above zero, but does not distinguish
them for JEPA-WM.

## Concrete falsification experiment

## 1. Data isolation

Use only the Stage 12 probe-training split to construct the outcome dictionary
and averaged Jacobian lens. Use probe calibration to select:

- prototype count;
- sparse-code size;
- intervention magnitude;
- any layer band.

Do not use the inspected development tasks for tuning. Any final claim must use
newly generated states and task/goal definitions.

The Phase 0 feasibility pilot may use:

- PushT only;
- 24 probe states;
- horizons 1 and 3;
- all ten actions per state;
- frozen and matched-ARGA predictors;
- all six AdaLN blocks.

## 2. Construct three frozen outcome dictionaries

For each state and horizon, form true target-encoder effects

\[
\Delta Z_{sah}=Z_{sah}-Z_{s0h},
\]

where action \(0\) is no-op. Center over all ten actions:

\[
\widetilde{\Delta}Z_{sah}
  = \Delta Z_{sah}
    - \frac{1}{10}\sum_{a'=0}^{9}\Delta Z_{sa'h}.
\]

Whiten using probe-training effects only. Retain the unpooled patch grid.

Build:

1. **spherical k-means prototypes**, representing recurring future effects;
2. **principal-effect directions**, an orthogonal non-sparse dictionary;
3. **covariance-matched random directions**, a null dictionary.

The primary Phase 0 size is \(K=16\). Pair every learned atom with its negative,
because physical changes are signed. Require any apparent workspace result to
hold for both the clustered and principal-effect dictionaries and to exceed
the random dictionary.

No dictionary may receive:

- task or goal identity;
- simulator cost;
- candidate label;
- development outcome;
- predicted-treatment identity.

## 3. Compute the Jacobian outcome lens

For prototype \(p_k\), score a predicted centered action effect as

\[
q_k(\hat Z_{sah})
  =
  \left\langle
    p_k,\widetilde{\Delta}\hat Z_{sah}
  \right\rangle.
\]

For each predictor block \(\ell\), estimate

\[
v_{\ell k}
  =
  \mathbb{E}_{s,a,h}
  \left[
    \nabla_{H_\ell}q_k(\hat Z_{sah})
  \right].
\]

Use vector-Jacobian products. Never materialize the full Jacobian of the
approximately \(256\times384\) token activation.

The expectation is computed on probe training. Preserve the full
token-by-channel gradient template. Sparse nonnegative coding over signed atoms
defines the local JOW component; the reconstruction residual is the non-JOW
component.

## 4. Run same-state interchange interventions

Choose recipient action \(a_i\) and donor action \(a_j\) from the identical
simulator state. Let \(P_\ell\) project or sparsely reconstruct the candidate
JOW component at block \(\ell\). Interpolate the donor difference:

\[
H'_{\ell,i}(\lambda)
  =
  H_{\ell,i}
  +
  \lambda
  P_\ell
  \left(H_{\ell,j}-H_{\ell,i}\right),
\qquad
\lambda\in\{0.25,0.5,0.75,1.0\}.
\]

Resume the frozen predictor after block \(\ell\).

The clean causal target is the model's own donor-action continuation. The
physical validation target is the true simulator outcome of the donor action.

Define normalized donor-transfer in an output representation \(y\) as

\[
T_y(\lambda)
  =
  \frac{
    \left\langle
      y(H'_{\ell,i}(\lambda))-y(H_{\ell,i}),
      y(H_{\ell,j})-y(H_{\ell,i})
    \right\rangle
  }{
    \|y(H_{\ell,j})-y(H_{\ell,i})\|_2^2+\epsilon
  }.
\]

An ideal linear donor swap gives \(T_y(1)\approx1\); an irrelevant edit gives
approximately zero.

Measure \(T_y\) in:

- final predicted JEPA tokens;
- action-relative geometry coordinates;
- independent decoded physical state;
- each downstream goal-cost margin.

## 5. Mandatory intervention controls

Every JOW swap must be compared with:

1. equal-norm JOW-orthogonal donor difference;
2. equal-norm random subspace;
3. principal activation components;
4. PatchSAE-style sparse features fit to the same activations;
5. state-shuffled donor;
6. action-shuffled outcome dictionary;
7. the Stage 12 final-token metric directions.

All edits use the same layer, token support, interpolation magnitude, and
activation norm.

## 6. Activation-distribution safety checks

For each intervention and \(\lambda\), record:

- activation norm and LayerNorm statistics;
- distance to nearest clean activation;
- Mahalanobis distance under the clean layer distribution;
- sparse reconstruction residual;
- final-prediction norm and ordinary latent error.

Require a monotone dose response over a range that does not exceed the
calibration distribution's predeclared tail threshold. Large effects appearing
only after an out-of-distribution jump do not count as causal-workspace
evidence.

## 7. Ablation test

Remove the recipient's active JOW component rather than replacing it:

\[
H'_{\ell,i}
  =
  H_{\ell,i}
  -
  \lambda P_\ell
  \left(H_{\ell,i}-\bar H_\ell\right).
\]

Compare action-effect damage and planning damage with every equal-norm control.
The primary question is not whether latent prediction error rises. It is
whether action identity and goal-dependent action ranking are disproportionately
damaged.

## 8. Broadcast test

Use the identical saved activation intervention—without refitting it—under:

- the native latent planner;
- at least three independently fitted physical decoders;
- multiple held-out goal functions;
- horizons 1, 3, and 6.

The intervention should cause coherent donor-directed changes across consumers.
An effect isolated to the readout used to construct the dictionary is not a
workspace.

## 9. Layer-band test

Repeat swaps and ablations after all six blocks. A workspace-like pattern
should show:

- weak or unstructured effects early;
- a reproducible intermediate peak;
- possible conversion to endpoint-specific representation late.

Predeclare how a band is detected using calibration data. A flat effect across
all layers supports distributed propagation, not a privileged workspace.

## 10. Treatment test

Repeat the frozen lens on:

1. frozen JEPA-WM;
2. latent-only adaptation;
3. shuffled-geometry adaptation;
4. matched-geometry adaptation.

Do not refit a treatment-specific outcome dictionary. The important result is
whether matched ARGA selectively improves JOW swap fidelity, causal
concentration, or broadcast relative to both frozen and shuffled controls.

## Candidate decision quantities

These quantities should be finalized only after CPU/small-GPU smoke testing:

- donor-transfer advantage over the best equal-norm control;
- interchange-intervention accuracy;
- JOW-ablation damage divided by best-control damage;
- fraction of readouts with donor-directed change;
- activation variance explained by the JOW component;
- active atom count;
- layer-band concentration;
- matched-minus-shuffled treatment contrast.

Candidate qualitative gate:

1. donor transfer is materially above every control over an in-distribution
   interpolation range;
2. JOW ablation causes at least twice the action-ranking damage of the strongest
   control;
3. the direction broadcasts across independent readouts and unseen goals;
4. the effect localizes to a stable layer band;
5. the result is robust to two independently constructed outcome dictionaries;
6. matched geometry beats frozen and shuffled geometry;
7. the frozen result replicates on numerically new tasks in both environments.

Compactness alone, high probe \(R^2\), sparse reconstruction, or a single
successful action swap is insufficient.

## Interpretation of possible outcomes

### No sparse reconstruction

Outcome information is too distributed for the proposed dictionary. Stop JOW.

### Sparse reconstruction but no causal advantage

The atoms are useful descriptions or probes, not a workspace. This is H1, not
H2.

### Causal swaps but no broadcast

The method found a specialized dynamics subspace, not a flexible shared
workspace. It may still be scientifically useful but should be named
accordingly.

### Broadcast but no layer band

There may be a distributed causal outcome representation. Avoid the workspace
claim.

### Full causal, broadcast, and layer-specific signal

Proceed to a separate, frozen experiment mapping current-to-goal differences
into the same outcome coordinates. Do not train a new goal metric until the
workspace claim is independently established.

## Recommended next action

Implement only Phase 0:

- frozen model and matched ARGA;
- PushT;
- 24 probe states;
- two horizons;
- \(K=16\);
- six layers;
- clustered, principal-effect, and random dictionaries;
- interpolated donor swaps plus equal-norm controls.

The deliverable should be a feasibility matrix and activation-distribution
audit, not a promotion claim. If donor transfer cannot beat controls cleanly,
the project stops before another full Colab run.
