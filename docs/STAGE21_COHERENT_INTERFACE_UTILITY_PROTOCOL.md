# Stage 21 coherent-interface and held-out utility protocol

## Motivation

Stage 20 reconfirmed strong representation mediation but failed reliable
planner steering. The rank-128 edit transferred approximately half of the
intended latent output contrast, yet only about 40% of the decoded score
contrast. It also intervened after predictor block 4 while block 5 remained
conditioned on the recipient action. Stage 21 separates that handoff question
from the distinct question of useful physical action selection.

## Phase A: coherent handoff

For each fresh evaluation state, the untouched decoded planner freezes its
rank-2, rank-3, and rank-4 actions as targets. For target \(t\), the baseline
best action \(b\), and deterministic derangement \(\pi_t\) satisfying
\(\pi_t(t)=b\), four counterfactual conditions are compared:

1. direct final-output permutation oracle;
2. complete candidate activation swap after block 5;
3. complete candidate activation swap after block 4;
4. the frozen Stage 18 rank-128 projected swap after block 4.

No simulator endpoint is used in Phase A. The final-output oracle must select
the target exactly. A coherent last-block handoff requires output and score
transfer coefficients of at least 0.995, target-choice rate at least 0.99, and
normalized counterfactual score RMSE at most 0.02. The learned block-4 edit
must retain mean output coefficient at least 0.25. Improvement of the block-5
choice rate over block 4 by at least 0.05 is reported separately as evidence
that downstream action conditioning caused the Stage 20 hybrid.

## Phase B: goal-independent correction

Let \(h_a\) be the whitened block-4 carrier for candidate action \(a\), and
let \(U\) be one frozen rank-128 basis. The feature is

\[
x_a=(h_a-\bar h)^\top U.
\]

The frozen physical decoder provides pose \(\hat p_a\), while simulator truth
provides \(p_a\). The supervised target is candidate-centered pose error

\[
y_a=(p_a-\hat p_a)-\mathbb{E}_{a'}[p_{a'}-\hat p_{a'}].
\]

A standardized multi-output ridge map \(Bx_a\) predicts this correction. It
uses no goal. Each learned, shuffled, and random-basis condition receives the
same construction examples and ridge grid. Ridge strength is selected by pose
MSE on a separate calibration split, after which the model is refit on the
combined construction and calibration data.

Evaluation predictions, score vectors, and selected actions are saved and
hashed before evaluation endpoint truth is authorized. Only then are actual
normalized regret, weighted pairwise accuracy, top-1 accuracy, and selected
true cost computed. A wrong-state learned-coordinate condition tests state
specificity.

## Frozen splits

Both `rotated_direction` and `pulsed_equal_impulse` use disjoint trajectory
pools. Model-blind physical eligibility selects per family:

- 32 construction records from 48 candidates;
- 16 calibration records from 32 candidates;
- 32 evaluation records from 48 candidates.

The same physical trajectory can contribute both action families within one
split, but no trajectory can cross construction, calibration, and evaluation.
Evaluation families are gated separately.

## Utility gate

For each action family, the learned-basis correction must satisfy all of:

- mean normalized-regret improvement over baseline at least 0.03 with a
  trajectory-bootstrap interval above zero;
- regret gain over median random at least 0.02 with interval above zero and a
  one-sided trajectory sign-test \(p\le0.05\);
- positive regret gain over shuffled;
- regret gain over wrong-state learned coordinates at least 0.02;
- weighted-pairwise-accuracy gain over baseline at least 0.02 with interval
  above zero;
- pairwise gain over median random at least 0.01;
- top-1 accuracy harm no worse than 0.02.

All required values must be finite. Simulator truth cannot rescue or alter a
previously frozen evaluation choice.

## Decisions

- `CONFIRMED_COHERENT_HANDOFF_AND_CAUSAL_SUBSPACE_UTILITY_BOTH_FAMILIES`:
  coherent handoff and utility both pass in both families.
- `COHERENT_HANDOFF_WITH_PARTIAL_UTILITY`: handoff passes in both families and
  utility passes in exactly one.
- `COHERENT_HANDOFF_WITHOUT_CAUSAL_SUBSPACE_UTILITY`: handoff passes in both,
  utility passes in neither.
- `HANDOFF_NOT_COHERENT_DIAGNOSTIC_FAILED`: either coherent-handoff gate fails.

Claim eligibility additionally requires exact source execution through the
decision cell, exact prior hashes, disjoint split certificates, complete fresh
work counts, and zero cache reuse.

## Claim boundary

A broad pass would show that frozen coordinates from a previously established
causal action-consequence subspace can support a calibrated correction that
improves one-step action selection on fresh PushT states. It would not show
that the original projected activation edit itself is a controller, that JEPA
has a native policy, that the correction works without physical supervision,
or that it generalizes to closed-loop horizons, another environment, another
checkpoint, or another architecture.
