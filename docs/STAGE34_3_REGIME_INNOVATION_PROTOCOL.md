# Stage 34.3 regime-aware JEPA innovation diagnostic protocol

## Why this stage exists

Stage 34.2 was execution-valid and rejected the registered rank-five JEPA
response state. Adding the 64-dimensional carrier sketch materially improved
held-out transition prediction, while deleting the fifth chart coordinate did
not hurt. The failure was strongly heterogeneous: carrier information helped
in free and pre-contact states but not in contact and post-contact states.

Stage 34.3 is a post-outcome diagnostic of two specific explanations:

1. the physical dynamics require explicit regime conditioning; or
2. the response chart omits a very small carrier innovation.

DINO is paused. No native checkpoint, simulator, or GPU is used.

## Bound evidence

The notebook consumes the exact Stage 34 transition shards from run
`d3f4f88426afff4d964bb4f1f1556c94ec3613b667edd9403ddfcd0fd78ded84`
and binds the exact Stage 34.2 stopped decision
`9fcedcf036a83c2ed234a39c62b4f4a7c2535dca7b0ba56cb0dd4c848c89ddb6`.
Both manifests, source commits, split identities, and every consumed file are
verified before analysis.

The registered split remains:

- model selection: 16 trajectories, 64 physical-mode records, 960 short-word
  transition rows;
- calibration: 16 disjoint trajectories, 64 records, 960 short-word rows; and
- evaluation: 32 further disjoint trajectories, 128 records, 1,024 unseen
  length-5--8 word rows.

Evaluation was observed in Stage 34.2, so Stage 34.3 can never be confirmation
evidence even though all new choices are frozen before it is read.

## Candidate family

For the registered response coordinates $q\in\mathbb R^5$, action summary
$a\in\mathbb R^5$, and 64-dimensional carrier sketch $s$, candidates use:

- $q_{1:4}$ or $q_{1:5}$;
- zero, one, two, or three carrier-innovation coordinates; and
- either one universal transition map or four source-mode transition maps.

An innovation basis is fitted within each training fold. First, a
capacity-matched random-Fourier predictor estimates the transition from the
candidate response coordinates and action. If $e$ is its five-coordinate
target residual and \(\widetilde s\) is the standardized carrier sketch, the
innovation directions are the first $k\leq3$ left singular vectors of

\[
\frac{1}{n}\widetilde s^\top
\operatorname{diag}(\widehat\sigma_e)^{-1}e.
\]

The state coordinate is $u=U_k^\top\widetilde s$. The basis is supervised,
but it is low rank, fold-local during selection, and never fitted using
evaluation. This is a bounded predictive repair, not an independently
identified physical observable.

All transition maps use the same fixed width-128 random-feature capacity.
Candidate rank, innovation rank, universal versus mode-specific dynamics, and
ridge penalty are scored by four-fold trajectory-grouped out-of-fold MSE on
model selection. Among candidates within 2% of the best error, the smallest
state is selected, then lower innovation rank and universal dynamics are
preferred. This simplicity rule is fixed before evaluation.

## Locked tests

The selected state is refit on calibration and tested once on evaluation.
Stable aggregate-MSE ratios are used instead of averages of rowwise ratios,
which Stage 34.2 showed can be dominated by rows with nearly zero denominator.
Intervals resample whole trajectories 5,000 times.

The candidate must pass every gate:

1. **Selection improvement:** at least 5% grouped-OOF improvement over the
   rank-five, zero-innovation, universal baseline.
2. **Evaluation transfer:** at least 5% aggregate evaluation improvement over
   that baseline, positive clustered lower endpoint, and positive improvement
   in all four modes.
3. **Residual sufficiency:** adding the complete standardized 64-dimensional
   carrier sketch improves by at most 5% on average, with interval upper
   endpoint and every mode at most 10%.
4. **Coordinate necessity:** deleting and refitting without each retained
   $q_j$ or $u_j$ must worsen aggregate error by at least 2%, have a
   positive clustered lower endpoint, and worsen every physical mode.
5. **Mode specificity:** if physical-mode experts are selected, they must beat
   equal-capacity experts trained and tested with independently
   within-trajectory-permuted record-mode identities by at least 5%, with a
   positive clustered lower endpoint. The gate is inapplicable when the
   universal candidate is selected.

Coordinate deletion replaces the selected coordinate by its calibration mean
while preserving input width. Both deletion models and the permuted-mode
control reuse the selected dynamics model's exact random-feature draw. Their
comparisons therefore change the registered information, not random-feature
luck.

## Interpretations

- `NO_SELECTED_REGIME_INNOVATION_REPAIR`: the bounded family did not improve
  enough even before evaluation.
- `SELECTED_REPAIR_DID_NOT_TRANSFER`: model-selection gains did not generalize
  to long words and new trajectories.
- `SELECTED_STATE_STILL_CARRIER_INCOMPLETE`: the small candidate transferred,
  but substantial predictive information remained in the carrier.
- `SELECTED_STATE_NOT_MINIMAL`: at least one retained coordinate was redundant.
- `PHYSICAL_MODE_STRUCTURE_NOT_SPECIFIC`: extra expert capacity, rather than
  physical regime identity, explained the gain.
- `BOUNDED_JEPA_STATE_CANDIDATE_REPAIRED`: one small candidate passed every
  observational diagnostic.

Even the final status supports only a candidate bounded predictive state on a
reused panel. Fresh trajectory families must independently confirm its
sufficiency and minimality before any native carrier intervention. Recursive
closure, causal use, planning value, and cross-model equivalence remain open.
