# Stage 18 rank-64 action-contrast confirmation protocol

## Question

Stage 17 found a positive but gate-failing rank-32 transfer effect, a larger
rank-64 sensitivity effect, and a near-exact complete activation-swap positive
control. Stage 18 tests the narrow next hypothesis: at frozen predictor block
4, does a construction-fitted rank-64 action-contrast subspace both transfer
donor-action predicted consequences and selectively remove naturally occurring
action-dependent consequence structure?

This is a finite-intervention experiment. It computes no Jacobian, VJP, JVP,
coordinate moment, or nonlinear probe.

## Mathematical object

For one state and 13 candidate actions, let the flattened block-4 carriers be

\[
H(s)=[h(s,a_1)^\top;\ldots;h(s,a_{13})^\top],
\qquad C=I-13^{-1}\mathbf1\mathbf1^\top.
\]

Construction data fit an output-aligned ridge map from whitened hidden
contrasts \(CH\) to a frozen 256-dimensional sketch of predicted-future
contrasts. Its ordered left singular vectors define nested bases \(U_r\) and
projections \(P_r=U_rU_r^\top\), for
\(r\in\{16,32,64,96,128\}\). The primary rank is fixed at 64.

For a frozen candidate derangement \(\Pi\), sufficiency uses

\[
\Delta^+_r(\rho)=\rho(\Pi-I)CHP_r.
\]

At full ambient rank and dose one this becomes the exact candidate-row
permutation. The complete activation swap is therefore retained only as a
positive control.

Necessity uses

\[
\Delta^-_r=-CHP_r.
\]

At dose one, adding this edit preserves the shared candidate mean and deletes
only the naturally occurring action-centered component in the selected space.
The endpoint is reduction in candidate-centered predicted-future energy,
relative to equal-rank shuffled-fit and empirical-span random ablations.

## Frozen sampling design

- Model/environment: public frozen `jepa_wm_pusht` in PushT.
- Horizon: 3 prediction steps and 5 simulator frames per step.
- Internal site: predictor block 4, fixed before this experiment.
- Candidate bank: one no-op plus 12 equal-norm radial branches, arranged in
  six antithetic pairs around the agent-to-block direction.
- Independent unit: one state from one independent trajectory.
- Construction: first 24 eligible states from a frozen pool of 48 trajectory
  IDs 300--347.
- Evaluation: first 32 eligible states from a disjoint frozen pool of 64
  trajectory IDs 400--463.

Before any model or encoder is loaded, every pool state is rolled out through
all candidates in the simulator. Eligibility requires true-cost spread at
least 0.02, at least 20% non-tied candidate pairs, and contact in at least two
branches. The selection rule, pool order, thresholds, actions, and seeds are
frozen. Evaluation simulator truth may determine eligibility, but evaluation
model activations remain sealed until the subspace and executed source prefix
are frozen.

This screen conditions the claim on physically action-informative PushT
states. It does not estimate performance over the unfiltered state
distribution.

## Construction-only fit and controls

The fixed block must first show mean linear CKA at least 0.15, mean advantage
over action-shuffled output geometry at least 0.03, and positive shuffle
advantage in at least 18 of 24 construction trajectories. Failure stops the
pilot before evaluation model activations are opened.

A construction-only channel covariance defines the whitening metric.
Leave-one-trajectory-out kernel-ridge validation chooses a frozen relative
penalty. The primary nested basis, an action-shuffled fitted basis, and four
empirical-action-span random bases are then frozen. Each random rank-128 basis
is orthogonal to the primary rank-128 basis. The independent evaluation output
sketch never participates in fitting or selection.

At every tested rank, learned, shuffled-fit, and four random subspaces receive
exactly norm-matched dose-one sufficiency edits. The rank-64 primary also gets
doses -0.5, 0.25, 0.5, and 1.0, plus wrong-state and common-mode controls.
Rank-64 necessity compares the primary ablation with exactly norm-matched
shuffled-fit and four random ablations. There are 42 intervention forwards per
evaluation trajectory.

## Frozen primary gates

The sufficiency gate requires all of:

- mean complete-swap donor coefficient at least 0.80;
- mean rank-64 coefficient at least 0.15 and cosine at least 0.20;
- mean candidate-mean-shift ratio at most 0.25;
- mean advantage over median random at least 0.05;
- mean advantage over shuffled fit at least 0.05;
- primary beats median random in at least 24 of 32 trajectories, with one-sided
  exact sign-test p-value at most 0.05 and clustered-bootstrap 95% lower bound
  above zero;
- positive dose slope in at least 24 trajectories and negative mean transfer
  for the -0.5 dose.

The necessity gate requires all of:

- mean output contrast-energy reduction at least 0.03;
- mean advantage over median random reduction at least 0.02;
- mean advantage over shuffled-fit reduction at least 0.02;
- positive primary-minus-random reduction in at least 24 of 32 trajectories,
  with one-sided exact sign-test p-value at most 0.05 and clustered-bootstrap
  95% lower bound above zero.

Decision labels are
`CONFIRMED_BIDIRECTIONAL_RANK64_MEDIATOR`,
`SUFFICIENCY_ONLY_RANK64_TRANSFER`,
`FULL_SWAP_ONLY_NO_CONFIRMED_RANK64_MEDIATOR`,
`NO_ACTION_CONTRAST_CAUSAL_SIGNAL`,
`STOP_NO_FIXED_BLOCK_ACTION_GEOMETRY`, and `INCONCLUSIVE`.

## Freshness, source binding, and claim boundary

A confirmatory pilot requires a full 40-hex source commit and a new run nonce.
The notebook refuses a pre-existing output directory and refuses to reuse any
truth, baseline, or intervention shard. The bundle records generated/cache
counts, raw-shard hashes, all physical eligibility rows, exact restore checks,
source hashes, and an initial/best/worst simulator montage. Only a source-bound
fresh run may emit a confirmatory label.

A bidirectional pass would support a rank-64 intervention-defined mediator of
candidate-action predicted consequences at block 4 under this candidate and
state distribution. It would not show that 64 is an intrinsic dimension, that
the space is unique or identifiable, that it is a physical coordinate chart,
or that it implements a Jacobian, Koopman operator, equivariant representation,
or general planning algorithm. Planning metrics are recorded as secondary
outcomes and require separate support for a planning-mediation claim.
