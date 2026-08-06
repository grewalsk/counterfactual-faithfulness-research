# Stage 25: causal KKT tomography of latent contact dynamics

## Scientific question

Stage 22 found a contact-aligned internal state variable, but Stage 23 found no
evidence that changing it selected a different downstream transition operator.
Stage 24 then rejected a shared rank-64 additive completion of the contact
effect. Those negatives make another global contact bit or fixed contact
subspace a poor next hypothesis.

Stage 25 instead tests whether predictor block 1 contains a state- and
action-dependent **contact-impulse coordinate** that is both readable and
causally used. The experiment is non-visual in its scientific judgment: images
are supplied to the frozen encoder, while every gate is a numerical latent,
simulator, or intervention metric.

## Mathematical hypothesis

Rigid contact is governed locally by an impulse correction to free motion,

\[
v^+ = v_{\mathrm{free}} + M^{-1}J(x)^\top\lambda^*,
\qquad 0\leq\lambda_n\perp g(x^+)\geq 0.
\]

The binary contact mode identifies only the active constraint. The solution
\(\lambda^*\) varies with state, action, geometry, and approach velocity. This
explains why a contact-aligned bit could be real while a fixed additive
completion and a same-layer multiplicative gate both fail.

The tested latent coordinates are

\[
q_1=\log(1+|\lambda_n|),\qquad
q_2=\operatorname{asinh}(\lambda_t),
\]

where normal and tangential components use the impulse-weighted contact frame.
Construction data fit one frozen linear readout of \((q_1,q_2)\) from a
256-dimensional CountSketch of block-1 activations. The exact CountSketch
adjoint maps the fitted readout back to two full activation covectors.

For a held-out activation \(h\), the impulse-erasure edit is the minimum-norm
solution

\[
\delta^*=\arg\min_\delta\|\delta\|_2^2
\quad\text{s.t.}\quad
W_\lambda^\top\delta=-\hat q(h),\;
W_m^\top\delta=0,
\]

where \(W_m\) contains the eight frozen Stage 23 mode covectors. Thus the edit
removes the inferred impulse coordinates while exactly preserving the known
contact-aligned mode coordinates.

## Exact physical counterfactual

Every selected state/action branch is rolled out twice from the same full
dynamic state and seed:

1. the ordinary PushT simulator;
2. a ghost branch with only agent–block collision disabled by reciprocal
   Pymunk collision filters.

Walls and every other simulator component remain active. A Pymunk `post_solve`
callback logs agent–block total impulse, contact normal, contact points, and
penetration distance. Ordinary-minus-ghost endpoint motion is therefore a
finite contact counterfactual, not an estimated derivative.

The frozen model's preference between encoded ordinary and ghost endpoints is
the native contact signal. A correct impulse-erasure intervention should move
the prediction toward the ghost endpoint. A reverse edit should move less or
oppositely.

## Split and leakage boundary

- Construction trajectory pool: IDs 2100–2199; first 48 physically eligible
  states in pilot mode.
- Evaluation trajectory pool: IDs 2300–2399; first 48 physically eligible
  states in pilot mode.
- Each state has a frozen 13-action bank and 15 simulator control steps.
- Physical eligibility is computed before model loading.
- The reader, ridge penalty, CountSketch adjoint, and activation covectors are
  frozen using construction activations only.
- Evaluation activations are opened only after the reader freeze certificate
  is written.
- Bootstrap clusters are evaluation states, not individual actions.

Stage 25 intentionally uses construction physical labels to fit the proposed
impulse coordinates. The novel claim is causal internal use, not unsupervised
discovery of impulse semantics.

## Controls and integrity checks

- collision instrumentation must agree with block momentum on each physics
  step (PushT's zero-damping setting makes an endpoint-minus-cumulative-impulse
  check invalid);
- construction grouped cross-validation and held-out evaluation both test
  impulse readout and contact discrimination;
- zero-edit hook identity is exact;
- the treatment exactly solves its coordinate constraints and protects all
  Stage 23 mode coordinates;
- a deterministic norm-matched random edit lies in the joint impulse/mode
  nullspace;
- a reverse-sign intervention tests directionality;
- the notebook binds the exact negative Stage 24 run and inherited Stage 23
  hashes;
- model, asset, notebook, builder, numerical source, and executed-cell hashes
  are verified;
- no Jacobian, JVP, VJP, gradient probe, or parameter update is used.

## Preregistered decision

`LATENT_CONTACT_IMPULSE_MECHANISM_SUPPORTED` requires every gate:

- median instrumentation residual at most 0.35;
- construction grouped-CV mean impulse \(R^2\) at least 0.10 and contact AUC
  at least 0.75;
- held-out mean impulse \(R^2\) at least 0.10 and contact AUC at least 0.75;
- at least 40 valid held-out contact branches;
- native contact coefficient bootstrap lower bound at least 0.10;
- impulse-erasure transfer-to-ghost lower bound at least 0.10;
- erasure-minus-random mean at least 0.05 with a positive bootstrap lower
  bound;
- reverse edit worse than erasure;
- mode-coordinate drift and impulse-coordinate residual at most \(10^{-6}\).

The ordered negative outcomes distinguish invalid instrumentation, absent
construction representation, failed held-out transfer, absent native contact
signal, and readable-but-not-causally-used impulse information. Non-pilot runs
always return `SMOKE_ONLY`, even if their reduced execution checks happen to
cross the numerical thresholds.

## Claim boundary

A positive result would support a causally used, two-coordinate latent
contact-impulse mechanism at block 1 of this frozen JEPA-WM on PushT. It would
not establish that the network implements a complete complementarity solver,
recover contact Jacobians or friction cones, generalize across environments or
models, or improve planning. Those are separate experiments.
