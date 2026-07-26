# Stage 2 result audit

## Outcome

The full GPU experiment completed for all 250 states, both public Push-T
checkpoints, all three horizons, and all three model variants. Exact simulator
restoration was bitwise exact. The notebook then failed in its reporting cell
while bootstrapping an all-missing contact-stratum ranking field; the exported
unit, pair, summary, cross-validation, and bootstrap data are complete.

The pre-specified decision is **`INCONCLUSIVE`**.

- Median repeated held-out delta R²: **0.003359**
- Repeats with positive delta R²: **100%**
- State-clustered MSE-improvement estimate: **0.000783**
- State-clustered 95% interval: **[-0.000700, 0.002376]**

The point estimate is favorable, but the interval crosses zero. Therefore the
pre-specified criterion for `NONREDUNDANT_SIGNAL` is not met, and this is not a
well-powered negative result either.

## Integrity checks

- Expected/observed unit rows: **4500/4500**
- Expected/observed pair rows: **202500/202500**
- Expected/observed CV rows: **240/240**
- Duplicate unit/pair/CV keys: **0/0/0**
- Exact endpoint restoration: **True**
- Exact initial render restoration: **True**
- Maximum paired-identity residual: **1.110e-15**
- Observed contact strata: **both, neither, one**

## Real-checkpoint metrics

| model | horizon | ordinary RMSE | normalized paired RMSE | effect cosine | physical top-1 | physical normalized regret | latent top-1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dino_wm_pusht | 1 | 0.134 [0.131, 0.138] | 0.474 [0.463, 0.486] | 0.860 [0.854, 0.866] | 0.956 [0.928, 0.980] | 0.035 [0.016, 0.057] | 0.564 [0.504, 0.624] |
| dino_wm_pusht | 3 | 0.198 [0.194, 0.203] | 0.433 [0.421, 0.447] | 0.883 [0.876, 0.889] | 0.740 [0.688, 0.796] | 0.207 [0.158, 0.255] | 0.592 [0.532, 0.652] |
| dino_wm_pusht | 6 | 0.258 [0.253, 0.263] | 0.509 [0.496, 0.521] | 0.773 [0.764, 0.783] | 0.532 [0.472, 0.588] | 0.421 [0.365, 0.481] | 0.648 [0.584, 0.708] |
| jepa_wm_pusht | 1 | 0.159 [0.155, 0.162] | 0.544 [0.534, 0.555] | 0.812 [0.805, 0.818] | 0.960 [0.932, 0.980] | 0.029 [0.012, 0.051] | 0.520 [0.460, 0.584] |
| jepa_wm_pusht | 3 | 0.208 [0.204, 0.213] | 0.432 [0.421, 0.444] | 0.883 [0.877, 0.888] | 0.732 [0.676, 0.788] | 0.209 [0.162, 0.258] | 0.624 [0.560, 0.684] |
| jepa_wm_pusht | 6 | 0.245 [0.240, 0.249] | 0.460 [0.450, 0.470] | 0.795 [0.786, 0.802] | 0.532 [0.472, 0.592] | 0.404 [0.348, 0.462] | 0.704 [0.644, 0.760] |

## Incremental-validity sensitivity analyses

| outcome | counterfactual_block | median_delta_r2 | fraction_positive | median_rmse_improvement |
| --- | --- | --- | --- | --- |
| coverage_normalized_regret | direction | -0.0007 | 0.3000 | -0.0002 |
| coverage_normalized_regret | joint | 0.0057 | 0.9500 | 0.0012 |
| coverage_normalized_regret | magnitude | 0.0066 | 1.0000 | 0.0014 |
| latent_normalized_regret | direction | -0.0011 | 0.2000 | -0.0001 |
| latent_normalized_regret | joint | 0.0127 | 1.0000 | 0.0011 |
| latent_normalized_regret | magnitude | 0.0142 | 1.0000 | 0.0012 |
| physical_normalized_regret | direction | -0.0020 | 0.0000 | -0.0004 |
| physical_normalized_regret | joint | 0.0034 | 1.0000 | 0.0007 |
| physical_normalized_regret | magnitude | 0.0048 | 1.0000 | 0.0011 |
| physical_pairwise_error | direction | NA | 0.0000 | NA |
| physical_pairwise_error | joint | NA | 0.0000 | NA |
| physical_pairwise_error | magnitude | NA | 0.0000 | NA |

## Why the primary result is inconclusive

The executable physical endpoint was weakly informative for this candidate
library. The block was initialized near the task goal and action 0 was a no-op,
so the physical oracle selected action 0 in
**71.7%**
of real-model rows. Real-model normalized physical regret was exactly zero in
**74.2%**
of rows; even the action-blind control had zero regret in
**71.7%**.
Only **48.8%**
of real unit rows had a finite physical pairwise-accuracy value. This creates a
floor/tie-heavy primary outcome and explains the reporting failure for strata
where every physical ranking credit was undefined.

Pair-level normalization is also unstable for branches whose simulator-derived
effect scale is essentially zero:
**3.9%**
of real pair rows have an effect scale below 1e-6. Contact-stratum summaries
should therefore foreground raw pair effect RMSE and treat normalized
pair-level values as undefined below a threshold fixed before the next run.
This issue does not change the already-exported pre-specified primary decision,
which used candidate-set aggregate metrics.

The real models clearly outperform action-blind and action-shuffled controls on
representation-matched ranking, so the model is action-sensitive. What remains
unresolved is whether the paired metric predicts *executable task regret*
beyond ordinary rollout error under a physically discriminative intervention
set.

## Staged-workflow decision

Do **not** proceed to the broad Stage 3 benchmark from this result. The fixed
gate requires positive incremental validity, a stable ranking reversal tied to
a physical regime, or a well-powered interpretable negative result. None is
established here.

The defensible next step is a narrow confirmatory Stage 2 revision: place the
block away from the goal and use a fixed, state-relative action generator that
produces multiple task-improving and task-worsening contact branches. Freeze
the candidate subset before confirmatory evaluation rather than selecting test
actions from their future simulator costs, and keep the held-out
regression/cluster-bootstrap decision rule unchanged.

This is simulator evidence about controlled Push-T interventions. It does not
establish real-world robotic reliability.
