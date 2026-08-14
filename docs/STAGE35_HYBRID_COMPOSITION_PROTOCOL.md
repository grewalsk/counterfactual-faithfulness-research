# Stage 35 JEPA hybrid predictive composition and closure protocol

## Evidential motivation

Stage 34.3 was execution-valid but its selected low-dimensional repair did not
transfer. A rank-four response state plus three supervised carrier innovations
improved model-selection error by 28.9%, then worsened locked evaluation by
2.5%. The pre-contact stratum worsened by 47.2%. Only one of the seven retained
coordinates was necessary in every mode.

The result rejected that particular bounded state, not every distributed JEPA
state. It also left a structural ambiguity: Stage 34.3 selected one expert from
the mode at the beginning of a length-5--8 action word, although such a word can
cross into or out of contact. No intermediate carrier state was saved or
recursively updated.

Stage 35 is a prospective experiment on entirely fresh trajectories. It tests
whether a fixed distributed JEPA carrier supports local hybrid updates that
compose across unseen action words. DINO remains paused.

## Registered object

At predictor block four, let (c_j\in\mathbb R^{256}) be a fixed count sketch
of the complete (256\times400) JEPA carrier after prefix (a_{1:j}). The
sketch seed and width are fixed before any trajectory is selected. The initial
state (c_0) is the block-four carrier under the registered one-step no-op.

For simulator modes (z_j) and transition class
(g_j=(z_j,z_{j+1})), the oracle hybrid update is

\[
 c_{j+1}=F_{g_j}(c_j,a_j).
\]

The primary rollout does not observe (g_j). It estimates

\[
 \pi_g(c_j,a_j)=\Pr(g_j=g\mid c_j,a_j)
\]

and recursively applies the mixture

\[
 \widehat c_{j+1}=\sum_g
 \pi_g(\widehat c_j,a_j)F_g(\widehat c_j,a_j).
\]

Each (F_g) is a capacity-matched random-feature ridge map with its linear
input coordinates retained. This is an empirical finite-bank operator, not an
identified saltation matrix or a theorem-backed minimal realization.

## Fresh split

No Stage 34 trajectory is reused. Complete four-mode trajectory families are
selected from disjoint identifier pools:

- construction: 16 trajectories from 16000--17599;
- model selection: 16 trajectories from 17600--19199;
- calibration: 16 trajectories from 19200--20799; and
- locked evaluation: 32 trajectories from 20800--23999.

Each trajectory contributes free, pre-contact, contact, and post-contact
states. Selection uses contact timing only, before either checkpoint loading or
effect measurement.

Construction fits the grounded readout. Model selection chooses random-feature
width and ridge using grouped recursive error on short words. Calibration fits
all transition experts, the predicted gate, matched label controls, the
carrier-to-grounded bridge, and the support reference. Evaluation is opened
only after these artifacts and their hashes are frozen.

Atomic actions `A` and `B` are present before evaluation, but the evaluated
compositions are unseen paired words of lengths five through eight. Thus the
test is compositional extrapolation, not extrapolation to unseen atomic action
magnitudes.

## Comparators and controls

The registered carrier models are:

1. one global local map;
2. source-mode experts with the initial mode frozen for the whole word, matching
   the Stage 34.3 structural assumption;
3. source-mode experts supplied the true mode at every step;
4. transition experts supplied true source-to-target mode classes;
5. the primary label-free predicted transition gate;
6. an otherwise identical model trained on within-trajectory-permuted
   transition labels; and
7. an identical model trained on one-step time-shifted labels.

A simulator-state recursive model is a positive control. The official JEPA
rollout is the direct prediction reference. A calibration-only bridge maps
native carrier states to the grounded native predictions; it is not fitted to
evaluation or used to define the carrier target.

## Locked gates

Intervals resample complete trajectories 5,000 times. The gates are cumulative:

1. **Source and split binding.** Exact committed notebook prefix, official
   checkpoint hash, disjoint pools, at least 12 evaluation trajectory families,
   and prefix invariance no worse than (10^{-6}).
2. **Simulator positive control.** Recursive normalized MSE at most 0.25 and at
   least 50% improvement over state persistence, with positive interval lower
   endpoint.
3. **Native physical fidelity.** Direct JEPA improves at least 10% over
   persistence, with positive interval lower endpoint.
4. **Guard transfer.** On words containing a mode transition, the predicted
   guard improves at least 10% over the frozen-source baseline, with positive
   interval lower endpoint; pooled improvement must also be positive.
5. **Guard specificity.** The predicted gate separately improves at least 5%
   over permuted and time-shifted controls. Each multiplicity-adjusted interval
   must have a positive lower endpoint.
6. **Recursive closure.** Recursive physical error is at most 1.25 times native
   physical error; the clustered ratio upper endpoint is at most 1.50; direct
   versus recursive discrepancy is at most 0.25 normalized MSE; at most 10% of
   predicted states leave the calibration support; and the recursive carrier
   beats carrier persistence.
7. **Family consistency.** The recursive/native error ratio is at most 1.50 for
   every word length and at most 2.00 for every initial physical mode.

Smoke mode exercises plumbing only and can never support a scientific result.

## Decision meanings

- `SIMULATOR_OPERATOR_CLASS_INVALID`: the selected local function class cannot
  recursively represent the task even with simulator state.
- `NATIVE_JEPA_NOT_PHYSICALLY_PREDICTIVE`: JEPA does not beat persistence on
  the fresh long words, so physical closure is not meaningful.
- `GUARD_RESET_STRUCTURE_DID_NOT_TRANSFER`: explicit transition structure does
  not repair the Stage 34.3 failure.
- `GUARD_SIGNAL_NOT_SPECIFIC`: the improvement can be reproduced by incorrect
  transition labels.
- `DISTRIBUTED_CARRIER_NOT_RECURSIVELY_CLOSED`: the carrier is predictive when
  read directly but does not form a closed recursively updateable state.
- `DISTRIBUTED_CLOSURE_NOT_FAMILY_CONSISTENT`: a pooled result hides a length or
  mode reversal.
- `BOUNDED_DISTRIBUTED_HYBRID_CLOSURE_SUPPORTED`: the fixed carrier sketch
  survives every observational gate on the fresh finite bank.

Even the final status is not causal evidence. It nominates the distributed
carrier and guard for a later native interchange/path intervention. Failure of
that intervention would still reject the mechanistic claim.

## Literature boundary

Predictive-state recursion is established in PSR and predictive-state inference
work; continuous nonlinear predictive states can be represented with kernel
embeddings; hybrid systems require explicit handling of discontinuous modes and
resets; and causal abstraction requires interventions rather than probes. Stage
35 combines those established ideas as a falsification test of a pretrained
visual world model. Novelty would require a later result showing that the
recursively closed distributed state is causally used.

