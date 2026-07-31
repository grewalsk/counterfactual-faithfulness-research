# Stage 12 independent result audit

## Verdict

**Independent decision: `NO_GO_TO_UNTOUCHED_TASK_CONFIRMATION`.**

The returned bundle is internally complete and numerically reproducible. The
declared `STOP_METRIC_CLASS_NOT_VIABLE` decision is robust to the reporting bug
described below: neither PushT nor Wall passes the Phase A shared-target-metric
gate after undefined rows are handled correctly, and neither environment shows
the preregistered Phase B causal bridge signal.

The scientifically defensible interpretation is narrower than the notebook's
label. This run rules out promotion of the *fitted Stage 12 recipe*. It is not a
clean proof that every low-rank positive-semidefinite target metric is
intrinsically incapable, because every metric optimization reached the
600-epoch limit without satisfying its convergence criterion.

## Scope and provenance

- Run signature:
  `f975a0a746e793b3ceb5d270189e9e9e9571e68176bc6e7ca9a153a78adc6a91`
- Environments: PushT and Wall.
- Design: 96 states and 12 tasks per environment, split into six probe-training,
  three probe-calibration, and three development-holdout tasks.
- Candidate set: ten actions evaluated at horizons 1, 3, and 6.
- Transition seeds: 11401, 11419, and 11437.
- Returned accelerator: NVIDIA RTX PRO 6000 Blackwell.
- Bundle failure trace: `success`.

## Integrity audit

The independent checker reproduced or verified all of the following:

| Check | Result |
|---|---:|
| Manifest entries with matching size and SHA-256 | 82 / 82 |
| Unexpected files | only the manifest itself, as designed |
| Exact simulator restoration | pass in both environments; maximum difference 0 |
| Transition checkpoints | 18 / 18, complete environment × method × seed matrix |
| Metric checkpoints | 18 / 18 |
| Transition tensor checksums and finiteness | pass |
| Metric symmetry, positive definiteness, trace, construction, condition bound | pass |
| Metric leakage flags | pass |
| Raw state rows recomputed from saved cost arrays | 5,472 / 5,472 |
| Seed-collapsed rows independently reproduced | 2,880 / 2,880 |
| Bootstrap draws and percentile intervals reproduced | 144,000 / 144,000 |

All pretrained asset hashes also match the bundle's declared verification
records. There is no evidence of a partial run, cache mismatch, corrupt
checkpoint, failed restore, missing seed, or arithmetic error in the stored
state-level metrics.

## Phase A: shared target metric viability

The selected metric in each environment was rank 2 with regularization 0.01:

| Environment | Optimization seed | Calibration margin RMSE | Margin-RMSE change vs native | Horizons passing joint gate | Goal-specificity gate |
|---|---:|---:|---:|---:|---:|
| PushT | 15101 | 1.4162 | **56.3% worse** | 0 / 3 | fail |
| Wall | 15119 | 2.3693 | **158.5% worse** | 0 / 3 | fail |

Finite-row recalculation of the main per-horizon metrics gives:

| Environment | Horizon | Shared regret | Native regret | Shared weighted accuracy | Native weighted accuracy | Joint gate |
|---|---:|---:|---:|---:|---:|---:|
| PushT | 1 | 0.2502 | 0.2910 | 0.6871 | 0.5905 | fail |
| PushT | 3 | 0.3840 | 0.4520 | 0.6301 | 0.5554 | fail |
| PushT | 6 | 0.3568 | 0.4059 | 0.5529 | 0.5632 | fail |
| Wall | 1 | 0.5065 | 0.3463 | 0.4044 | 0.5288 | fail |
| Wall | 3 | 0.4541 | 0.2864 | 0.3540 | 0.5036 | fail |
| Wall | 6 | 0.1072 | 0.1400 | 0.7053 | 0.7802 | fail |

PushT's shared metric improves regret and ranking accuracy over native at some
horizons, but it does not reach the absolute and relative joint thresholds.
Wall is mostly worse. More fundamentally, the learned metric's normalized
margin error is substantially higher than the train-scaled native target
metric in both environments.

The Phase A goal-permutation test also fails. PushT has sufficient regret
specificity but an accuracy specificity gain of -0.0068. Wall has gains of
0.0152 regret and 0.0196 accuracy, both below the required 0.02.

## Phase B: causal bridge

Positive values below favor matched action-response geometry. The primary
matched-versus-frozen contrasts are:

| Environment | Horizon | Δ regret | Δ weighted accuracy |
|---|---:|---:|---:|
| PushT | 1 | -0.0049 | +0.0215 |
| PushT | 3 | -0.0137 | -0.0114 |
| PushT | 6 | +0.0497 | +0.0054 |
| Wall | 1 | -0.0742 | +0.0214 |
| Wall | 3 | +0.0641 | +0.0493 |
| Wall | 6 | -0.0080 | +0.0254 |

The matched-versus-shuffled-geometry contrasts are:

| Environment | Horizon | Δ regret | Δ weighted accuracy |
|---|---:|---:|---:|
| PushT | 1 | -0.0049 | -0.0125 |
| PushT | 3 | -0.1057 | -0.0346 |
| PushT | 6 | -0.0022 | -0.0111 |
| Wall | 1 | -0.0371 | +0.0234 |
| Wall | 3 | +0.0327 | -0.0032 |
| Wall | 6 | -0.0165 | -0.0251 |

No horizon in either environment simultaneously:

1. beats frozen and shuffled geometry by the preregistered regret and accuracy
   thresholds, and
2. improves both metrics directionally against the latent-only control.

The required majority of development tasks also fails against both frozen and
shuffled controls in both environments. Averaged across horizons, the true
goal metric does not outperform the goal-permuted metric with the required
specificity: the true-minus-permuted gain is (-0.0380 regret, -0.0203 accuracy)
in PushT and (+0.0035, +0.0000) in Wall.

Therefore the saved Phase B data independently support **no causal bridge
signal** for the combined ARGA plus shared-target-metric method.

## What remains positive

The negative planning result does not erase the action-relative geometry
finding. Matched geometry beats frozen geometry at every horizon in both
environments, with all six task-clustered intervals above zero. It also beats
shuffled geometry at all PushT horizons and clearly at Wall horizon 3.

This sharpens the diagnosis:

> action-relative geometry is repairable, but this target-only low-rank
> quadratic planner does not reliably convert that repair into robust
> goal-conditioned action selection.

That is useful negative evidence and a coherent paper direction. It is not a
reason to spend untouched-task confirmation compute on this recipe.

## Implementation findings

### 1. Metric fits did not converge

All 18 metric checkpoints report `completed_epochs = 600` and
`converged_before_max_epochs = false`. The selected matrices are numerically
valid and comfortably satisfy the condition-number limit, but their
optimization curves ended at the configured boundary. This violates the
experiment's own clean-interpretation guardrail.

Effect on the decision: **no change to the no-go**. A nonconverged recipe that
fails Phase A should not be promoted. Effect on the scientific claim: replace
"the entire metric class is not viable" with "the tested fitting recipe did
not establish metric-class viability."

### 2. Undefined weighted-accuracy rows propagate NaNs

Some PushT states have no non-tied true action pairs, for which weighted
pairwise accuracy is correctly undefined. The notebook's task-equal summary
uses ordinary means, so these values propagate into PushT horizon-1 Phase A
reporting and the complete-planner non-harm gate.

Finite-row aggregation repairs the affected summaries. PushT horizon-1 shared
accuracy becomes 0.6871 and its horizon-1 complete-planner non-harm check
passes, but PushT horizons 3 and 6 fail. Both Phase A environment decisions,
both Phase B environment decisions, and the final no-go remain unchanged.

## Recommended next action

Do not run untouched-task confirmation and do not tune this bridge again on
the three inspected development tasks.

For the paper, retain Stage 12 as diagnostic negative evidence alongside the
positive geometry intervention. If method development continues, use only the
probe training/calibration split to repair and verify optimizer convergence,
freeze a numerically distinct planner recipe, and evaluate it on newly
generated tasks rather than reopening these development outcomes.

## Reproduction

From the repository root:

```bash
python scripts/audit_stage12_bundle.py \
  --output results/stage12_full_development_audit.json
```

The machine-readable audit is
`results/stage12_full_development_audit.json`; the original extracted artifacts
are preserved under `results/bundles/stage12_result_bundle/`.
