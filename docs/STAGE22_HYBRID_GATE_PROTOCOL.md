# Stage 22: latent hybrid-gate causal interaction protocol

## Scientific question

Stages 18 and 19 established a low-rank intervention-defined mediator of the
model's own action-conditioned future prediction.  Stage 21 showed that this
consequence representation reaches the last predictor block coherently but
does not support specific held-out physical action utility.  Stage 22 asks a
deeper mechanistic question: does the frozen JEPA-WM compute action
consequences through a state-dependent event gate that conditionally enables
a distinct action-effect carrier?

This is a falsification-first pilot of one necessary component of a larger
mechanistic-dynamics compiler.  It does not claim to extract a complete hybrid
automaton.

## Mathematical hypothesis

The candidate computation is

\[
\hat z^+ = b(z,a)+\sum_{k=1}^{K} g_k(z,a)u_k(z,a),
\]

where a sparse mode variable `g` selects a continuous update `u`.  The primary
two-mode pilot tests an off/on factorization.  For a base candidate and an
on-mode donor, let `G` be a one-dimensional construction-fitted gate direction
and `U` an orthogonal construction-fitted effect subspace at one frozen
predictor block.  Four finite interventions produce

\[
Y_{ij}=F_\theta(\operatorname{do}(G=G_i),
                 \operatorname{do}(U=U_j)).
\]

The non-additive interaction is

\[
I=Y_{11}-Y_{10}-Y_{01}+Y_{00}.
\]

For an additive representation `b + G + U`, `I=0`.  For a gated computation
`b + G U`, `I=(G_1-G_0)(U_1-U_0)`.  All edits are finite donor-minus-base
interchanges; no Jacobian, JVP, VJP, or gradient probe is computed.

## Label-free discovery

Construction activations are candidate-centered within state and projected by
a frozen CountSketch.  A standardized PCA followed by deterministic two-means
discovers two activation modes.  Clustering uses only model activations.  The
cluster with larger mean model-predicted consequence energy is named `mode_on`.
Simulator contact counts do not fit the partition, choose its orientation, or
select the predictor block.

The block is selected on construction data by a frozen score combining
cluster separation, balance, within-state two-mode coverage, and output-energy
contrast.  After selection, full construction carriers are re-extracted.  A
channel metric is frozen, the gate direction is the whitened difference of
the two discovered mode means, and an output-aligned ridge subspace is fitted
and orthogonalized against the gate.

Evaluation mode assignments and base/donor pairs are frozen from model data
before evaluation contact labels are opened.  Physical labels are used only
to test whether the discovered mode corresponds to contact and to define the
physically aligned subset for the causal interaction endpoint.

## Controls

- a within-state label-shuffled gate direction;
- an empirical-span random gate direction orthogonal to the learned gate and
  effect subspace;
- exactly norm-matched gate edits for both controls;
- the same learned effect edit under every gate condition;
- a complete donor activation swap as an on-manifold positive control;
- zero-edit hook identity;
- source binding, fresh shards, sealed evaluation assignments, and
  trajectory-clustered bootstrap inference.

## Pilot decisions

- `EVENT_GATED_CAUSAL_INTERACTION_CONFIRMED`: label-free mode discovery aligns
  with held-out physical contact and the learned gate produces a positive
  gate-by-effect interaction exceeding shuffled and random controls.
- `PHYSICAL_MODE_WITHOUT_CAUSAL_INTERACTION`: the internal mode aligns with
  contact but the factorial causal prediction fails.
- `INTERNAL_MODE_NOT_PHYSICAL_CONTACT`: a stable internal partition exists but
  does not align with contact.
- `NO_STABLE_INTERNAL_MODE_PARTITION`: construction discovery fails.
- `SMOKE_ONLY` or `INCONCLUSIVE`: no scientific claim.

Even a pass establishes only a two-mode, one-environment, one-checkpoint
event-gating mechanism.  It does not yet establish a complete causal program,
symbolic rule, general hybrid automaton, physical truth, or planning utility.
