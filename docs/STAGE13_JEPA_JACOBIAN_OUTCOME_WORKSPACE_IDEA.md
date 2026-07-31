# Stage 13 ideation: a Jacobian Outcome Workspace for JEPA-WMs

**Status: research ideation, not a frozen protocol or authorization to run.**

The companion
[`STAGE13_JOW_LITERATURE_AND_FALSIFICATION_PLAN.md`](STAGE13_JOW_LITERATURE_AND_FALSIFICATION_PLAN.md)
separates direct evidence from analogy and specifies the proposed
interchange-intervention test in greater detail.
The
[`STAGE13_JOW_COMPUTE_MINIMAL_COLAB.md`](STAGE13_JOW_COMPUTE_MINIMAL_COLAB.md)
turns Phase 0 into a sequentially gated Colab design that can fail cheaply
before any expanded run.

## Core idea

Anthropic's J-space is not an architecturally designated compartment. The
[Jacobian lens](https://transformer-circuits.pub/2026/workspace/index.html)
defines intermediate directions by their average first-order causal effect on
future output logits. A sparse, nonnegative set of these output-linked
directions appears to mediate some flexible reasoning even though it explains
only a small fraction of activation variance.

An action-conditioned JEPA has no token vocabulary or unembedding matrix, so a
literal port is impossible. It does, however, have a natural class of potential
outputs: encoded future outcomes under executable actions. We can replace the
LLM vocabulary with a fixed dictionary of target-encoder **future-effect
prototypes** and ask:

> Does JEPA-WM route action-specific future consequences through a small,
> sparse, causally privileged intermediate workspace that multiple downstream
> goals and planners can read?

Call the candidate object the **Jacobian Outcome Workspace (JOW)**.

This is a diagnostic hypothesis first, not a proposal to train another
high-capacity planner.

## Why this follows from Stages 4–12

The evidence now points to a representation-interface failure:

1. Stage 4 showed that action-specific predicted consequence structure is
   causally necessary for planning.
2. Stage 7 found strong physical action-effect decodability in intermediate
   AdaLN predictor blocks, even where final planning remained poor.
3. Stage 11 showed that relative action-response geometry can be repaired
   without a task or goal objective.
4. Stage 12 showed that a low-rank quadratic metric on final target latents
   does not reliably convert that repair into decisions.

Stage 12 assumed that the useful decision axes are globally recoverable from
the final target-token geometry. The J-space result suggests a different
possibility: the causally important content may occupy a small, layer-local
component and explain little total variance. A global endpoint metric can miss
such a component even when probes can decode it.

The official JEPA-WM formulation trains an action-conditioned predictor solely
against future encoder embeddings and plans by comparing predicted embeddings
to goal embeddings; it has no reward, value, or reconstruction head
([paper](https://arxiv.org/html/2512.24497v3),
[code](https://github.com/facebookresearch/jepa-wms)). That makes the choice
of which latent directions a planner reads especially consequential.

## Proposed construction

Let

- \(H_\ell(s,a,h)\) be the unpooled 256-token activation after predictor block
  \(\ell\);
- \(\hat Z_h^a\) be the final predicted token grid at horizon \(h\) under
  candidate action \(a\);
- \(Z_h^a\) be the target-encoder token grid for the true simulator outcome;
- \(\widetilde{\Delta}Z_h^a\) be the whitened, action-relative true effect,
  centered within the candidate set for the same state.

### 1. Construct an outcome vocabulary

Using only probe-training transitions, cluster or sparsely factor the true
action-relative effects \(\widetilde{\Delta}Z_h^a\) into a small dictionary
\(\{p_k\}_{k=1}^{K}\). Candidate values are \(K=16,32,64\).

The construction uses:

- true target-encoder latents;
- state identity only for within-state centering;
- no goal, task label, simulator cost, candidate identity, development outcome,
  or predicted rollout.

Motion and interaction effects are signed, unlike ordinary token concepts.
Each prototype should therefore have paired atoms \(+p_k\) and \(-p_k\), while
the sparse coefficients remain nonnegative. This preserves the cone
interpretation without pretending that "left" and "right" are the same
concept.

Raw future-state clusters are a useful control but not the primary dictionary:
they can be dominated by static appearance. The primary atoms should describe
counterfactual changes relative to the same state.

### 2. Define an outcome lens

For a predicted action-relative effect
\(\widetilde{\Delta}\hat Z_h^a\), define each prototype score

\[
q_k(\hat Z_h^a)
  = \langle p_k,\widetilde{\Delta}\hat Z_h^a\rangle .
\]

At predictor block \(\ell\), define the context-averaged vector-Jacobian
product

\[
v_{\ell k}
  = \mathbb{E}_{s,a,h}
    \left[
      \frac{\partial q_k(\hat Z_h^a)}
           {\partial H_\ell(s,a,h)}
    \right].
\]

Each \(v_{\ell k}\) is a full spatial-token gradient template rather than a
pooled channel vector. This matters because Stage 7 found useful information
in the unpooled token grid.

The candidate workspace at layer \(\ell\) is the set of sparse nonnegative
combinations of the signed lens atoms:

\[
\mathcal{J}^{\mathrm{out}}_\ell
  =
  \left\{
    \sum_k \alpha_k v_{\ell k}:
    \alpha_k\geq0,\;
    \|\alpha\|_0\leq r
  \right\}.
\]

Sparse coding gives an activation's local outcome-workspace coordinates. The
residual after reconstruction is its non-workspace component.

### 3. Add a write/read privilege test

J-space is compelling partly because many circuits can write to and read from
it. The JEPA analogue should test both sides:

- **write Jacobian:** how strongly executable action changes can move a
  direction, using \(\partial H_\ell/\partial a\);
- **read Jacobian:** how strongly the same direction changes future-effect
  scores, using \(\partial q/\partial H_\ell\).

A control-theoretic robustness analysis can form empirical controllability and
observability Gramians and compute a balanced subspace. Directions that are
both easy for actions to write and easy for future-outcome functions to read
are better workspace candidates than directions selected only by activation
variance.

This balanced subspace is complementary to, not a replacement for, the sparse
prototype lens. Agreement between the two would be strong evidence.

## The decisive tests are causal

Decodability is insufficient; Stage 7 already established it. JOW should only
be claimed if interventions outperform matched controls.

### Coordinate swap

For two candidate actions from the same simulator state:

1. decompose each intermediate activation into JOW and residual components;
2. swap only their sparse JOW coordinates;
3. preserve the residual components;
4. continue the frozen predictor from the intervention layer.

The predicted physical consequence and action-margin ordering should move
toward the swapped-in action. A norm-matched swap in the non-JOW residual is
the primary control.

### Ablation

Ablate the active JOW component while preserving the residual. Compare against:

- equal-norm random directions;
- equal-norm principal components;
- sparse-autoencoder directions;
- JOW-orthogonal activation directions.

A privileged workspace should produce disproportionate damage to
action-specific outcomes and planning, not merely raise ordinary latent error.

### Broadcast

Apply the identical coordinate edit and evaluate it with:

- the native latent planner;
- multiple independently fitted physical decoders;
- several held-out goal-cost functions;
- horizon-specific action-ranking margins.

If the edit only affects the readout used to define it, it is a specialized
probe direction, not a workspace. A workspace edit should coherently redirect
several downstream consumers.

### Layer localization

Repeat the measurements after all six AdaLN blocks. A J-space-like result
should show a reproducible layer band where:

- sparse outcome content becomes stronger;
- action identity is converted into outcome identity;
- causal swaps are effective;
- the final layer may become more endpoint-specific.

If every layer behaves similarly, "workspace" is probably the wrong
description.

## Treatments and controls

The frozen Stage 12 matrix is unusually useful for this experiment:

1. frozen JEPA-WM;
2. fidelity-constrained latent-only adaptation;
3. shuffled-geometry adaptation;
4. matched-geometry adaptation.

The key comparison is whether matched ARGA selectively improves the emergence,
stability, or causal efficacy of JOW coordinates relative to shuffled geometry,
not merely whether its activations become easier to classify.

The outcome dictionary must be frozen before examining the matched-versus-
control planning contrasts.

## Suggested pilot

### Phase 0: feasibility

- One environment.
- 24 probe-training states and probe-calibration only.
- Horizons 1 and 3.
- \(K=16\) signed prototype pairs.
- All six predictor blocks.
- Frozen and matched ARGA conditions.
- Randomized vector-Jacobian products; never materialize a full Jacobian.

Stop if the prototype scores do not reconstruct true action-relative effects
or if coordinate interventions are numerically unstable.

### Phase 1: workspace diagnosis

- Both PushT and Wall.
- Probe-training construction and probe-calibration selection only.
- Frozen, latent-only, shuffled, and matched conditions.
- \(K\in\{16,32,64\}\), chosen without development planning outcomes.
- Coordinate-swap, ablation, layer-band, and broadcast controls.

This phase should not fit a goal metric.

### Phase 2: untouched causal test

Generate new simulator states and new task/goal definitions. Freeze the
dictionary, layer band, sparsity, intervention magnitude, and all thresholds
before evaluation.

Only after a successful Phase 2 should we consider using JOW coordinates for
planning.

## Candidate go/no-go criteria

These are ideation-level targets, not yet a preregistration:

1. **Compactness:** the selected component explains no more than 20% of
   activation variance.
2. **Causal concentration:** JOW ablation causes at least twice the
   action-ranking damage of every equal-norm control.
3. **Swap fidelity:** JOW coordinate swaps recover at least half of the full
   source-to-target physical action-effect displacement.
4. **Broadcast:** the intervention direction agrees across the native planner,
   at least two independent physical decoders, and at least two unseen goal
   functions.
5. **Layer specificity:** the causal effect peaks in a reproducible
   intermediate layer band.
6. **Treatment specificity:** matched geometry improves JOW causal efficacy
   over both frozen and shuffled geometry in both environments.
7. **Task generalization:** the frozen result repeats on numerically new tasks.

Failure of criteria 1–5 would argue that JEPA-WM's actionable content is
distributed rather than workspace-like. Failure of criteria 6–7 would show
that a workspace may exist but ARGA does not repair it in a generalizable way.

## If the workspace exists

The first planner to test should not be another free Mahalanobis metric.
Instead:

1. express the current-to-goal target difference in the same frozen outcome
   prototype coordinates;
2. express each predicted candidate effect in those coordinates;
3. rank candidates by agreement between desired and predicted outcome codes;
4. keep all prototype and sparsity choices frozen;
5. compare against native, Stage 12 shared metric, and goal-permuted controls.

This directly aligns the goal query with the internal format that causally
mediates action outcomes.

## What would make this paper-worthy

The strongest result would not be "a new lens improves planning." It would be:

> Action-conditioned JEPAs contain a small, causally privileged outcome
> workspace. Executable actions write to it, multiple task functions read from
> it, and repairing its geometry—not total latent error—predicts
> counterfactual planning performance.

The equally valuable falsification is:

> Unlike language-model J-space, actionable JEPA-WM representations are
> distributed across token space and layers; sparse output-causal coordinates
> do not mediate planning.

Either outcome is cleaner than another unconstrained adapter sweep.

## Main risks

- The prototype vocabulary may manufacture apparent sparsity. Compare multiple
  frozen dictionaries and random/prototype controls.
- Averaged Jacobians can erase state-dependent mechanisms. Report
  context-conditioned variance and test local as well as global lenses.
- A first-order lens may fail under large swaps. Establish a dose-response
  range and require local linearity.
- Outcome concepts are relational and spatial; a bag of prototypes may miss
  binding between object, direction, contact, and time.
- Probe-training target latents are valid for discovery, but any use of
  simulator cost or inspected development outcomes would turn the diagnostic
  into another task-specific planner fit.
- Calling the result a "workspace" requires compactness, causal privilege,
  flexible downstream use, and layer localization—not merely interpretability.
