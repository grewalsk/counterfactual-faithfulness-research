# Stage 13 JOW Colab: optimality audit

**Status: pre-run design audit. No Stage 13 result has been observed.**

## Judgment

The optimal first allocation is not a small version of the entire workspace
study. It is a training-free necessary-condition screen with one frozen output
dictionary, one frozen-model Jacobian lens, strong same-state causal controls,
and automatic termination. The executable notebook now follows that design.

The screen is optimized for information gained per GPU-minute, not for making
a workspace claim. A pass authorizes a larger layer/horizon experiment; a
failure should terminate this formulation without searches over seeds, goals,
or prototype counts.

## Material design choices

### Build the expensive lens once

The primary lens is estimated in the frozen model and reused unchanged in the
frozen, matched-ARGA, and shuffled-ARGA arms. This is preferable in the first
screen because it:

- removes a condition-specific degree of freedom;
- asks whether the same candidate workspace becomes more causally effective;
- avoids repeating construction-state VJPs for adapted checkpoints; and
- makes matched-versus-shuffled differences attributable to the action-path
  intervention rather than to separately optimized lenses.

Condition-specific lenses remain a secondary expansion analysis. They should
not replace the shared-lens comparison.

### Separate existence from treatment

Two gates are reported:

1. **Workspace gate:** does the frozen model contain a compact, decodable,
   output-linked subspace whose same-state swaps beat controls?
2. **Treatment gate:** does matched ARGA improve the causal efficacy of that
   fixed subspace over frozen and shuffled geometry?

ARGA can fail without falsifying the workspace hypothesis. The earlier design
incorrectly made treatment improvement necessary for workspace promotion.

### Use a natural residual as the primary control

The strongest cheap control is the donor-minus-recipient activation difference
after removing its JOW projection. It comes from the same state, actions,
layer, and natural activation pair, then is scaled to exactly the JOW edit
norm. A random orthogonal edit is retained as a second control.

This is stricter than comparing only with Gaussian directions, which can be
easy to beat because they need not follow the model's natural activation
geometry.

### Test compactness and distribution shift directly

Before the causal gate, the notebook now requires:

- the best lens to have full requested rank;
- held-out JOW coordinates to beat a random orthogonal activation basis;
- the selected subspace to explain no more than 20% of activation variance;
- the two-coordinate sparse code to retain at least 40% of JOW-coordinate
  energy; and
- construction-state VJP sketches to have positive mean alignment.

Each intervention records both its norm relative to natural same-state action
differences and a local diagonal activation-distance ratio. Excessive shift on
either diagnostic counts as out of distribution.

### Spend pairs on diversity

Action pairs are chosen by true, goal-free target-effect separation, but the
screen requires disjoint actions before reusing an action. This keeps the
signal-rich feasibility design while preventing one extreme action from
dominating every intervention row.

### Select the primary layer before causal outcomes

The primary horizon-layer is chosen by state-grouped construction-split
coordinate prediction, then frozen. Calibration coordinate validity is checked
at that fixed layer before interventions. The causal gate cannot choose the
layer with the largest intervention effect, and matched and shuffled arms use
the exact frozen-model layer. The default screen therefore spends swaps on one
preselected layer; the expansion may measure all layers descriptively while
retaining the same fixed primary gate.

The VJP/coordinate screen covers horizons 1 and 3 and all six blocks. Once
swaps are restricted to the preselected layer, this broader necessary-condition
coverage is a better use of compute than testing many interventions at only
three sampled blocks. It materially reduces the chance of declaring failure
because the workspace was localized to an unsampled layer or appeared only
after a longer counterfactual rollout.

## Sequential compute policy

1. Verify assets, simulator restoration, hook identity, finite VJPs, and
   measured runtime.
2. Test compact target-effect structure.
3. Build the frozen lens once.
4. Test held-out compactness and coordinate validity.
5. Run frozen same-state causal swaps.
6. Download and test matched ARGA only after a frozen causal pass.
7. Download shuffled ARGA only if matched improves over frozen.
8. Stop after any failed necessary condition and package the failure evidence.

The adapted arms reuse the frozen lens, so a successful three-condition screen
costs one VJP construction plus three intervention evaluations rather than
three VJP constructions.

## What remains deliberately unresolved

The default screen has only four calibration states and is not confirmatory.
It does not establish:

- precise uncertainty intervals;
- both-environment generalization;
- all-layer or horizon localization;
- robustness to several independently constructed dictionaries;
- broadcast to unseen goal functions and planners;
- a condition-specific workspace reorganization; or
- untouched-task replication.

Those are expensive discriminators that become worthwhile only after the
frozen causal gate passes. Adding them to the initial Colab would lower
information gained per unit compute because a negative basic signal would make
their results moot.

## Remaining failure interpretations

- Dictionary failure supports a distributed or higher-rank outcome space.
- Coordinate failure means the averaged output lens does not recover a compact
  held-out representation, even if outcome information is decodable elsewhere.
- Causal failure means the candidate subspace is correlational or too locally
  linear to mediate swaps.
- Workspace pass plus treatment failure supports JOW existence but not ARGA as
  its repair mechanism.
- Both passes justify the expanded layer/horizon study, not a paper-level
  workspace claim.
