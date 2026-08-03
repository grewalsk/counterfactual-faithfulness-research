# Stage 15: fixed-reader longitudinal predictive-control bundle pilot

## Question

Does a frozen PushT JEPA-WM carry a state-conditioned predictive-control
bundle whose local spaces transport smoothly along real trajectories and whose
transported modes causally control fixed physical predictions?

Stage 14 established a discrete spatial field of local read/write geometry,
but its target-energy readers changed with state and its operators were saved
only at rollout initial states. Stage 15 changes both facts. It does not search
again for a single global frame.

## Frozen design

- Eight deterministic PushT trajectories, interleaved into four construction
  and four untouched evaluation trajectories.
- Five evenly spaced physical states per trajectory.
- Exact intermediate-state restoration includes agent velocity, block velocity,
  and block angular velocity; a one-step continuation identity is checked
  before target or operator extraction.
- Horizons 1 and 3 and all six predictor blocks.
- One fixed six-coordinate reader: agent x/y, block x/y, and block sin/cos.
- Three fixed CountSketch projections. Ridge strength is chosen using
  leave-one-construction-trajectory-out CV; all reader parameters are frozen
  before evaluation targets are opened.
- One six-dimensional DCT-like action-tangent basis in normalized executable
  action coordinates, identical at every state.
- Exactly six directional JVPs are evaluated in that frozen basis; the
  unused 30-coordinate horizon-3 action Jacobian is never materialized.
- One construction-only channel metric per predictor block.
- The primary causal assay uses predictor block 4 (zero-indexed block 3) at
  horizon 3. Other blocks and horizon 1 are robustness analyses.

Smoke mode uses fewer trajectories/states and can never authorize a scientific
claim. The pilot is confirmation-eligible only when the exact committed
notebook prefix is verified before evaluation is opened.

## Primary geometric tests

For every within-trajectory state pair, measure physical distance, temporal
separation, fixed-coordinate K distance, read/write principal-angle distance,
and read/write spatial-energy distance. Independently permute time labels
inside each trajectory to obtain the null.

Primary support requires, on untouched evaluation trajectories:

1. nearby states have smaller read- and write-subspace distances;
2. fixed-coordinate K changes with physical/temporal separation rather than
   behaving as an exchangeable sequence; and
3. adjacent modes retain more observable gain than nonadjacent time-shuffled
   modes.

Because K uses the same physical-reader rows and action-basis columns at every
state, raw K is the primary estimand. Procrustes alignment is applied only to
carrier-mode bases as a coordinate diagnostic; it is not allowed to rotate the
semantic rows or action columns of K.

## Primary causal test

At each adjacent pair on an evaluation trajectory:

1. extract the dominant controllable write mode at the source state;
2. reuse that source mode at the destination state after norm matching to the
   destination's local positive-control mode;
3. apply positive and negative carrier patches;
4. compare the observed fixed-reader displacement with the destination VJP
   prediction; and
5. compare transported observable gain with equal-energy covariance-shaped,
   exact token-support-matched, time-shuffled, local-mode, and no-edit controls.

Support requires directional linearity, substantial recovery relative to the
local mode, and positive gain over matched nulls. A local-mode effect without
neighbor transfer is classified as local causal control, not bundle transport.

## Claim ladder

- `LONGITUDINAL_CAUSAL_BUNDLE_SUPPORTED`: reader validity, evaluation
  smoothness, transported-mode causal recovery, matched-null advantage, and
  source binding all pass.
- `LOCAL_CAUSAL_ONLY`: fixed local modes are causally effective but do not
  transfer to adjacent states.
- `SMOOTH_NONCAUSAL_FIELD`: longitudinal geometry passes but causal transport
  does not.
- `NO_LONGITUDINAL_BUNDLE_EVIDENCE`: the fixed-coordinate longitudinal tests
  fail.
- `SMOKE_ONLY`: plumbing only; no scientific conclusion.

Even a positive pilot establishes the result only for the frozen PushT
checkpoint. A paper-level generality claim requires preregistered replication
across seeds, environments, and JEPA variants.
