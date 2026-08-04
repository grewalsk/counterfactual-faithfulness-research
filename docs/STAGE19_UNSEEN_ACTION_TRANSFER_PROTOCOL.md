# Stage 19 frozen-subspace unseen-action transfer protocol

## Question

Stage 18 established that a construction-fitted block-4 action-contrast
subspace was both sufficient and necessary for part of JEPA-WM's
candidate-specific predicted-future contrast. Stage 19 asks whether that
causal effect transfers beyond the exact action bank used to discover and
confirm it.

The broad hypothesis is:

> The exact frozen Stage 18 projector mediates predicted consequences for
> prespecified directions, magnitudes, and temporal profiles that were absent
> from Stage 18 construction and evaluation.

This is a representation-transfer test, not an intrinsic-dimension,
coordinate-chart, planning, or multi-model claim.

## Immutable prior artifact

The only admissible treatment projector is the successful Stage 18 artifact:

- block: 4;
- primary rank: 64;
- sensitivity rank: 128;
- SHA-256:
  `2f9c496d54623a9062e465a18c70039acc18cb8a1cc2833a5f4ade162ca3f90b`;
- Stage 18 source commit:
  `16edd247cddcb1aa121340eb5fa42bd9e07004c3`;
- Stage 18 decision: `CONFIRMED_BIDIRECTIONAL_RANK64_MEDIATOR`.

The whitening transform, shuffled-fit basis, and four empirical-span random
bases are imported from the same file. Stage 19 may not refit, rotate, tune,
select, or reconstruct them. Artifact and adjacent provenance mismatches stop
the run before model activations.

## Frozen action families

Each bank contains no-op plus twelve antithetic radial branches over fifteen
environment steps:

| Family | Direction grid | Stepwise profile |
|---|---|---|
| `rotated_direction` | Stage 18 midpoints (+15°) | 0.12 for all 15 steps |
| `magnitude_0p08` | Stage 18 grid | 0.08 for all 15 steps |
| `magnitude_0p16` | Stage 18 grid | 0.16 for all 15 steps |
| `delayed_equal_impulse` | Stage 18 grid | 0 for 5, then 0.18 for 10 |
| `pulsed_equal_impulse` | Stage 18 grid | 0.18 for 5, 0 for 5, 0.18 for 5 |

The temporal profiles sum to the same vector impulse as fifteen constant
steps at magnitude 0.12. They still differ dynamically; equal impulse is a
design control, not a claim that PushT physics is path-independent.

## Fresh states and selection

Sixty-four new state/goal specifications (trajectory IDs 500–563) are fixed
before simulator or model data. All five action families are simulated from
each state. Within each family, the first 24 records satisfying frozen
model-blind thresholds are selected:

- true-cost spread at least 0.02;
- non-tied pair fraction at least 0.20;
- at least two branches with contact.

The same physical state may appear in several families. Family-specific gates
use the 24 state IDs as independent units; no naive 120-record pooled
p-value is used. Wrong-state donors are drawn within family.

## Interventions

At rank 64, sufficiency applies donor-permuted action-contrast transfer at
doses −0.5, 0.25, 0.5, and 1.0. At ranks 64 and 128 it also evaluates:

- learned sufficiency;
- equal-norm shuffled-fit sufficiency;
- four equal-norm empirical-span random sufficiency controls;
- learned, shuffled, and four random necessity ablations.

Rank-64 wrong-state, matched common-mode, and full-swap controls are retained.
A separate zero-edit hook identity test must have maximum error at most
`1e-6`. There are 30 patched forwards per selected record in pilot mode.

## Family-level gate

For each action family, sufficiency requires all of:

- full-swap donor coefficient at least 0.75;
- rank-64 coefficient at least 0.12 and cosine at least 0.15;
- action-mean shift ratio at most 0.25;
- mean gain over random and shuffled at least 0.04;
- at least 18/24 positive learned-minus-random trajectories;
- exact sign-test p-value at most 0.05 and clustered-bootstrap lower bound
  above zero;
- at least 18 positive dose slopes and a negative mean coefficient at dose
  −0.5.

Necessity requires:

- mean action-contrast energy reduction at least 0.025;
- mean gain over random and shuffled at least 0.015;
- at least 18/24 positive learned-minus-random trajectories;
- exact sign-test p-value at most 0.05 and clustered-bootstrap lower bound
  above zero.

All required metrics must be finite. Both halves must pass for a family to
count as bidirectional transfer.

## Decisions

- `CONFIRMED_TRANSFER_ALL_UNSEEN_ACTION_FAMILIES`: all five families pass.
- `PARTIAL_TRANSFER_DIRECTIONS_AND_SOME_FAMILIES`: rotated directions and at
  least two other families pass.
- `LIMITED_ACTION_FAMILY_TRANSFER`: one or more families pass without the
  partial-transfer rule.
- `NO_CONFIRMED_UNSEEN_ACTION_TRANSFER`: no family passes.

The emitted scientific status is confirmatory only when the Stage 19 source
prefix is verified, the prior artifact is exact, the output directory is
fresh, and provenance counts show zero cache reuse.

Planning regret and pairwise-accuracy changes are preregistered secondary
readouts. They do not enter or rescue the representation-transfer decision.

## Claim boundary

A broad pass supports a reusable causal action-consequence subspace over these
five local PushT action transformations in this frozen JEPA-WM. It does not
establish rank 64 as intrinsic dimension, identify human-interpretable
coordinates, validate Jacobian methods, or generalize to other environments,
checkpoints, architectures, or long-horizon closed-loop planning.
