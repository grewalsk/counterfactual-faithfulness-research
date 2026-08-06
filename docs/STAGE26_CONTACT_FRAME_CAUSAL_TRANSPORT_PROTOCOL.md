# Stage 26: contact-frame causal transport

## Question

Stage 25 found held-out normal-impulse information at predictor block 1, but
erasing the easiest linear readout caused essentially no output transfer. The
tangential target was numerically zero. Stage 26 asks whether the causal
contact carrier is instead a spatial field whose basis changes with contact
point and normal.

For contact geometry \(\xi=(c,n)\), the tested family is

\[
\delta h_\ell=U_\ell(\xi)q,
\qquad U_\ell(g\xi)\simeq\rho_\ell(g)U_\ell(\xi).
\]

The experiment tests finite causal transport. It does not claim exact group
equivariance from representational similarity alone.

## Contact chart

For token center \(p\), contact normal \(n\), tangent \(t\), and radius
\(R=96\), define

\[
u={n^\top(p-c)\over R},\qquad
v={t^\top(p-c)\over R},\qquad
w=\exp[-(u^2+v^2)/2].
\]

The three raw spatial functions \((w,wu,wv)\) are orthonormalized with a
deterministic QR sign convention. For a token field
\(H\in\mathbb R^{256\times400}\), the canonical feature is

\[
x_\xi=\operatorname{vec}(Q_\xi^\top H)\in\mathbb R^{1200}.
\]

Because \(Q_\xi^\top Q_\xi=I\), a canonical displacement is injected exactly
as \(\delta H=Q_\xi\operatorname{unvec}(\delta x)\). The notebook verifies the
round-trip residual is at most \(10^{-6}\).

## Construction-only discovery

- Pools 2500--2599 and 2700--2799 are disjoint from all earlier stages.
- Forty physically eligible construction and forty evaluation states are
  selected before model loading.
- Each of 13 action branches is simulated normally and with only agent--block
  collision disabled.
- The simulator records impulse-weighted contact point and normal.
- Construction activations from all six predictor blocks are represented in
  both the contact chart and a world-x-axis chart.
- Grouped out-of-fold ridge prediction targets normal impulse plus the exact
  normal, tangential, and angular ordinary-minus-ghost corrections.
- Numerically degenerate response components are excluded by frozen scale
  floors. Normal impulse is always retained.
- One block and ridge penalty are selected on construction data only.
- The left singular vectors of the standardized response map freeze a
  rank-at-most-four response fiber.

Construction passes only if contact-frame mean out-of-fold \(R^2\ge0.10\) and
its advantage over the world-axis chart is at least 0.05.

## Sealed finite interventions

For each held-out contact branch, a construction donor is chosen using a fixed
contact-frame nuisance descriptor and must have at most half the recipient's
normal impulse when such a donor exists. Let \(P_U=UU^\top\) be the frozen
fiber projector. The primary edit is

\[
\delta x_i=P_U\left[{x_j-x_i\over s_x}\right]\odot s_x,
\qquad
\delta H_i=Q_{\xi_i}\operatorname{unvec}(\delta x_i).
\]

The coordinate value therefore comes from a real construction activation; it
is not optimized against the held-out output.

The identical coordinate displacement is tested under:

- recipient contact-frame alignment;
- a world-x-axis basis at the recipient point;
- the donor's original world contact frame;
- a fiber-orthogonal, local, norm-matched random direction;
- reversed sign;
- a full local-chart donor swap as an intervention-assay positive control.

The causal target is the appropriately dosed movement from the native
prediction toward the exact collision-disabled endpoint. Independent output
sketch and physical-pose readouts are both saved.

## Decision

`CONTACT_FRAME_CAUSAL_TRANSPORT_SUPPORTED` requires:

- the Stage 25 negative and source provenance bind exactly;
- the construction contact-frame gate passes;
- at least 40 held-out contact branches remain valid;
- robust median native contact coefficient is at least 0.10;
- the full local swap moves at least 5% of the desired-output norm;
- aligned transfer has a trajectory-bootstrap lower bound of at least 0.05;
- aligned-minus-each-of-world-axis, donor-location, and random-local controls
  has mean at least 0.02 and a positive lower bound;
- reversed alignment has the opposite sign;
- contact-coordinate reconstruction error is at most \(10^{-6}\).

If construction alignment fails, the status is
`NO_CONTACT_FRAME_RESPONSE_FIELD`. If the full-swap assay is inactive, the
causal result is uninterpretable. If the assay works but transport fails, the
status is `CONTACT_FIELD_READABLE_BUT_NOT_CAUSALLY_TRANSPORTABLE`, and the
low-dimensional contact-mediator program closes in favor of path-level
distributed computation.

## Claim boundary

A positive result establishes a state-conditioned, contact-frame-transportable
causal response fiber at one frozen JEPA-WM/PushT checkpoint. It does not yet
establish exact SE(2) equivariance, cross-model generality, a complete contact
solver, or planning benefit. Those require new checkpoints, environments, and
held-out transformations.
