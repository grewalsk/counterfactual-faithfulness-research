# Stage 20 frozen-subspace causal planner-steering protocol

## Question

Stages 18 and 19 established that a fixed block-4 subspace is sufficient and
necessary for part of JEPA-WM's action-specific predicted-consequence signal,
including transfer to unseen action families. Stage 20 asks whether that
mechanism actually reaches the model's downstream numerical action choice:

> Can an intervention in the exact frozen subspace make a prespecified
> near-frontier action inherit the predicted consequence of the model's
> current best action, rise in rank, and become selected more often than
> rank- and norm-matched controls?

This is a one-step causal-planning assay. It is non-visual and does not train,
refit, rotate, or select a new representation.

## Prior binding and fresh evaluation

The treatment is the exact Stage 18 block-4 artifact with its rank-128 basis
used as the primary edit and rank 64 retained as a sensitivity. Its required
SHA-256 is
`2f9c496d54623a9062e465a18c70039acc18cb8a1cc2833a5f4ade162ca3f90b`.
Stage 20 also requires the claim-eligible Stage 19 decision at SHA-256
`493fdf5c707189caea11043db7d208dbc38677dcf5881008e13bede87f40be9c`.
Both are verified with adjacent provenance before model activations.

Fresh trajectory IDs 600–679 are fixed before simulator or model data. The
two separately gated families are `rotated_direction` and
`pulsed_equal_impulse`. Simulator-only eligibility selects the first 32 of 80
states per family satisfying frozen diversity/contact thresholds. The model
cannot affect this physical state selection.

## Exact steering target

For a selected state, let the untouched model assign numerical predicted task
costs

\[
q(a), \qquad a\in\{0,\ldots,12\},
\]

where smaller is better. The baseline-best action is the donor \(b\). Before
any intervention, actions at stable baseline ranks 2, 3, and 4 are frozen as
three targets \(t\). This uses the untouched model's scores only—never
simulator endpoint costs or intervention outputs.

For every target, a deterministic derangement \(\pi_t\) is fixed such that
\(\pi_t(t)=b\). The complete counterfactual score vector is

\[
q_t^{\mathrm{cf}}(a)=q(\pi_t(a)).
\]

Because the target inherits the unique baseline minimum, its intended
counterfactual rank and choice are exact:

\[
\arg\min_a q_t^{\mathrm{cf}}(a)=t.
\]

The notebook verifies this equality before opening evaluation.

## Intervention and controls

At predictor block 4, let \(z_a\) be the whitened activation for candidate
\(a\), let \(\bar z\) be the candidate mean, and let \(P_r\) be the frozen
rank-\(r\) projector. The learned targeted replacement at dose \(d\) is

\[
z'_a=z_a+dP_r\left[(z_{\pi_t(a)}-\bar z)-(z_a-\bar z)\right].
\]

The primary test uses rank 128 at doses 0.5 and 1.0. Rank 64 is a sensitivity.
Each target also receives the frozen shuffled-fit basis, four empirical-span
random bases, a wrong-state edit, a matched common-mode edit, and a complete
activation swap positive control. All stochastic controls are frozen by
seed, and random/shuffled edit norms are matched to the learned edit. One
learned, one shuffled, and four random necessity ablations are computed once
per record.

## Readouts

The treatment chain is evaluated at four linked levels:

1. transfer of the intended predicted output contrast;
2. change in the target action's numerical rank;
3. whether the selected action matches the exact counterfactual target;
4. simulator cost of the action the patched planner selects.

The simulator cost is opened only after targets are frozen. It describes the
physical consequence of executing the induced choice; it does not enter the
target definition or primary gate and cannot rescue a failed steering result.
Stage 20 does not claim that arbitrary internal steering improves task reward.

## Frozen family gates

Each family has a representation gate and a planner-steering gate.

The representation gate requires finite metrics, complete-swap coefficient at
least 0.80, mean learned rank-128 output coefficient at least 0.25, learned
gain over random at least 0.10 and positive gain over shuffled, and necessity
reduction at least 0.03 with gains of at least 0.02 over both random and
shuffled controls. In pilot mode, clustered intervals for output and necessity
gains must exclude zero, the dose-1 effect must exceed dose 0.5, and the
trajectory-level output-gain sign test must have \(p\le0.05\).

The planner-steering gate requires finite metrics, complete-swap target-choice
rate at least 0.90, learned target-rank gain over random at least 0.25, learned
counterfactual-choice gain over random at least 0.05, positive gains over
shuffled controls, clustered rank and choice intervals above zero, and a
trajectory-level rank-gain sign test with \(p\le0.05\).

A family passes the causal chain only if both gates pass. The 32 trajectories,
not the 96 target attempts, are the sign-test units; bootstrap resampling is
clustered by trajectory.

## Decisions

- `CONFIRMED_CAUSAL_PLANNER_STEERING_BOTH_FAMILIES`: both families pass the
  complete representation-to-choice chain.
- `PARTIAL_CAUSAL_PLANNER_STEERING`: exactly one family passes the chain.
- `PREDICTION_MEDIATOR_TRANSFER_WITHOUT_CONFIRMED_PLANNER_STEERING`: both
  representation gates pass but neither planner chain does.
- `NO_CONFIRMED_STAGE20_CAUSAL_CHAIN`: no broader condition applies.

Confirmatory labels require exact source binding, both prior artifacts, a new
nonce/output directory, expected generation counts, and zero cache hits.

## Claim boundary

A broad pass would show that, in this checkpoint and two local PushT action
families, a reusable internal action-consequence mechanism is causally upstream
of numerical action ranking and selection. It would be stronger than
decodability or output mediation alone. It would not establish beneficial
control, human-interpretable concepts, rank 128 as an intrinsic dimension,
multi-step closed-loop planning, visual plausibility, or generalization to a
new environment, checkpoint, or architecture.
