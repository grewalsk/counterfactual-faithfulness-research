# Stage 17 finite action-contrast causal interchange protocol

## Question

Stage 4 established that planning is more sensitive to destruction of the
action-to-predicted-consequence correspondence than to an exactly
magnitude-matched shared decoded-state error. Stage 17 asks whether that
action-relative information is causally mediated by a proper subspace at an
internal JEPA-WM predictor block.

This experiment does not use coordinate moments, Jacobians, VJPs, JVPs, or an
unconstrained nonlinear probe.

## Mathematical object

For one state and (K=13) fixed candidate actions, let

\[
H_l(s)=[h_l(s,a_1)^\top;\ldots;h_l(s,a_K)^\top]
\]

be the flattened token activations at predictor block (l). Define

\[
C=I_K-K^{-1}\mathbf1\mathbf1^\top,
\qquad R_l(s)=CH_l(s).
\]

(R_l) represents candidate-action information modulo an arbitrary shared
activation displacement. Construction data fit an orthonormal basis (U_l)
from finite hidden-action residuals to a frozen CountSketch of the model's
future-token residuals. The primary rank is 32 and
(P_l=U_lU_l^\top).

For a frozen derangement \(\Pi\), the primary edit is

\[
H_l'(\rho)=H_l+\rho(\Pi-I)R_lP_l.
\]

At (P_l=I) and ρ=1 this reduces exactly to \(H_l'=\Pi H_l\). Consequently,
the complete swap is an on-manifold positive control, not the primary
mechanistic result.

## Data and sampling units

- Environment/model: frozen public `jepa_wm_pusht`.
- Horizon: 3 prediction steps, 5 simulator frames per prediction step.
- Six predictor blocks are screened on construction data.
- Sixteen entirely new trajectory/task groups use IDs 200--215, disjoint from
  inspected Stage 15 IDs 0--7 and Stage 16 IDs 100--123.
- Eight even-ID trajectories are construction; eight odd-ID trajectories are
  evaluation.
- Three frozen timepoints per trajectory are repeated measurements.
- Thirteen common finite action branches consist of a no-op and six
  antithetic temporal action-basis pairs.
- The independent trajectory, not the action, token, patch, or timepoint, is
  the statistical unit.

## Construction-only selection

At each block, linear CKA compares the candidate-centered hidden Gram matrix
with a 128-dimensional training output sketch. Action-shuffled and wrong-state
output geometries are controls. The earliest block within one trajectory-level
standard error of the best mean CKA is selected.

The full pilot stops before evaluation unless the selected block has:

- mean CKA at least 0.15;
- mean advantage over action-shuffled output geometry at least 0.03; and
- positive shuffle advantage in at least six of eight construction
  trajectories.

A construction-only channel covariance defines the activation metric. A
leave-one-trajectory-out kernel-ridge screen chooses one of the frozen relative
penalties. The primary and shuffled-correspondence subspaces are then frozen.
Four equal-rank controls are sampled from the empirical construction action
span and orthogonalized against the primary rank-32 space.

## Evaluation interventions

Evaluation uses a second independently seeded output CountSketch that never
participates in layer, ridge, or subspace selection.

The intervention families are:

1. primary rank-32 projected donor edit at doses -0.5, 0.25, 0.5, and 1.0;
2. rank 4/8/16/64 compression sensitivities at dose 1;
3. a rank-32 shuffled-correspondence fitted subspace;
4. four equal-rank empirical-span random subspaces;
5. a wrong-state donor residual through the primary subspace;
6. a common-mode direction with exactly matched whitened activation energy;
7. the complete activation swap as a positive control.

Shuffled, random, wrong-state, and common-mode controls are exactly norm-matched
to the primary rank-32 dose-one edit in the construction-fitted whitened
activation metric.

## Primary endpoint

Let (Z) and (Z') be candidate rows in the independent output sketch. The
donor target and observed change are

\[
T=(\Pi-I)CZ,\qquad E=C(Z'-Z).
\]

The primary coefficient is

\[
\eta=\frac{\langle E,T\rangle_F}{\lVert T\rVert_F^2}.
\]

No edit has η=0; exact donor transfer has η=1. Donor cosine,
reconstruction, candidate-mean drift, decoded physical-pose transfer, action
ranking, and normalized planning regret are also saved. The frozen physical
decoder is secondary and cannot select the internal mechanism.

## Frozen causal gate

The full pilot passes only if all of the following hold:

- complete-swap mean donor coefficient is at least 0.35;
- primary mean donor coefficient is at least 0.15;
- primary mean donor cosine is at least 0.20;
- primary mean candidate-mean-shift ratio is at most 0.25;
- primary minus median-random coefficient is at least 0.08;
- primary minus shuffled-fit coefficient is at least 0.05;
- primary beats median random in at least seven of eight trajectories and the
  one-sided exact sign-test p-value is at most 0.05;
- positive-dose slopes are positive in at least seven trajectories; and
- the mean negative-dose coefficient is negative.

Decision labels are:

- `FINITE_ACTION_CONTRAST_CAUSAL_MEDIATION`;
- `FULL_SWAP_ONLY_NO_COMPRESSED_MEDIATION`;
- `NO_INTERNAL_ACTION_CONTRAST_SIGNAL`;
- `STOP_NO_CONSTRUCTION_ACTION_GEOMETRY`;
- `INCONCLUSIVE`; or
- `SMOKE_ONLY`.

An otherwise passing unbound run is labeled `UNBOUND_EXPLORATORY_RESULT` and
retains its candidate status. A source-bound claim requires exact committed
notebook-prefix verification before evaluation is opened.

## Claim boundary

A pass would support causal mediation of predicted action-consequence identity
through a finite, reduced-rank internal action-contrast subspace under these
PushT candidates. It would not establish a physical coordinate chart, a
Jacobian field, temporal transport, equivariance, a Koopman system, an
object-centric representation, or real-world control.
