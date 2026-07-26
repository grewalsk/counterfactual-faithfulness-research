# Stage 2B Result Audit

## Bottom line

The Stage 2B run is complete and internally consistent. Its pre-specified
decision is **`NEGATIVE_SIGNAL`**.

The models clearly contain action-conditioned predictive information: the real
models outperform action-blind and action-shuffled controls on latent
counterfactual prediction and, at horizons 3 and 6, on executable Push-T action
selection. However, the two proposed aggregate counterfactual metrics do **not**
explain additional held-out variation in physical planning regret after
controlling for ordinary rollout error, horizon, model, effect magnitude,
contact fraction, and design stratum.

This is not evidence that the models ignore actions. It is evidence that these
particular latent-space counterfactual scores are not useful stand-alone
proxies for physical planning quality in this experiment.

## Run and data integrity

- Full run on an NVIDIA A100-SXM4-40GB.
- 250/250 simulator states completed.
- 250/250 states completed for each of `dino_wm_pusht` and
  `jepa_wm_pusht`.
- `FAILURE_TRACE.txt` contains `NONE`.
- Final bundle status is `SUCCESS`.
- Exact-state restoration passed all three repetitions:
  - endpoint bitwise exact;
  - initial render bitwise exact;
  - diagnostic values exact;
  - maximum endpoint absolute difference: `0.0`.
- Expected and observed table sizes match exactly:

| Table | Expected rows | Observed rows | Duplicate keys |
|---|---:|---:|---:|
| Unit metrics | 4,500 | 4,500 | 0 |
| Pair metrics | 202,500 | 202,500 | 0 |
| Metric summaries | 18 | 18 | 0 |
| Contact summaries | 18 | 18 | 0 |
| Incremental-validity results | 240 | 240 | 0 |

The published summary means reproduce from the unit table to floating-point
precision. Candidate-design diagnostics and contact-stratum counts also
reproduce exactly.

Missing pairwise ranking values are expected ties, not failed computations.
Physical pairwise ranking is finite for 82.2% of pairs and unit-level physical
pairwise accuracy for 89.7% of units. All primary regression variables are
finite.

## Candidate design

Candidate selection was frozen before evaluation and did not consult future
simulator outcomes.

| Horizon | No-op is oracle | No-op has positive regret | Median physical cost spread |
|---:|---:|---:|---:|
| 1 | 39.2% | 60.8% | 0.023 |
| 3 | 2.8% | 97.2% | 0.128 |
| 6 | 1.6% | 98.4% | 0.283 |

The redesign successfully removes the earlier no-op degeneracy at horizons 3
and 6. Horizon 1 remains comparatively weak and should not be interpreted as
an equally decisive planning test.

Only 2.96% of real-model action pairs have ground-truth effect scale below the
`1e-6` analysis threshold: 6.67% at horizon 1, 2.22% at horizon 3, and 0% at
horizon 6.

## The models are action-conditioned

The following values average the two real checkpoints.

| Horizon | Real normalized paired RMSE | Action-blind | Action-shuffled | Real effect cosine |
|---:|---:|---:|---:|---:|
| 1 | 0.693 | 1.000 | 1.352 | 0.632 |
| 3 | 0.572 | 1.000 | 1.360 | 0.772 |
| 6 | 0.564 | 1.000 | 1.408 | 0.810 |

Lower normalized RMSE and higher cosine are better. Action-blind prediction has
normalized error 1 and zero direction by construction; shuffled actions are
worse than action-blind and have negative cosine. The real models therefore
predict action-specific latent changes rather than merely extrapolating a
common future.

## Physical planning improves, but imperfectly

Values below average the two checkpoints. Lower normalized regret is better.

| Horizon | Real top-1 | Action-blind | Action-shuffled | Real regret | Blind regret | Shuffled regret |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 43.2% | 39.2% | 36.2% | 0.421 | 0.418 | 0.421 |
| 3 | 20.4% | 2.8% | 8.8% | 0.481 | 0.590 | 0.538 |
| 6 | 31.4% | 1.6% | 7.4% | 0.380 | 0.546 | 0.472 |

At horizon 1 the real model provides essentially no regret improvement over the
action-blind control. At horizons 3 and 6 it provides substantial improvement:
normalized regret drops by 0.109 and 0.166, respectively, relative to the
action-blind control.

DINO-WM is consistently better than JEPA-WM on physical regret:

| Horizon | DINO-WM | JEPA-WM |
|---:|---:|---:|
| 1 | 0.407 | 0.435 |
| 3 | 0.460 | 0.502 |
| 6 | 0.353 | 0.407 |

This ordering is accompanied by lower ordinary rollout error for DINO-WM, so it
does not establish an independent counterfactual advantage.

## Primary incremental-validity result

The primary held-out regression added:

- normalized paired-effect RMSE; and
- one minus paired-effect cosine

to a base model containing:

- ordinary rollout RMSE;
- horizon;
- model identity;
- ground-truth effect magnitude;
- contact fraction; and
- near/far design stratum.

All rows from the same simulator state stayed in the same cross-validation
fold.

| Primary quantity | Result |
|---|---:|
| Median base held-out R² | 0.1095 |
| Median full held-out R² | 0.1049 |
| Median ΔR² | **−0.00427** |
| Positive ΔR² repetitions | **0/20** |
| Bootstrap MSE improvement | **−0.000949** |
| 95% CI, MSE improvement | **[−0.001576, −0.000380]** |
| Bootstrap ΔR² | **−0.00567** |
| 95% CI, ΔR² | **[−0.00951, −0.00226]** |

“MSE improvement” is base MSE minus full MSE, so a negative value means the
added counterfactual block predicts slightly worse out of sample.

The effect is small in practical magnitude, but its direction is consistent.
The correct conclusion is **no incremental physical-planning validity**, not
that counterfactual metrics cause worse planning.

Ordinary rollout RMSE and normalized paired-effect RMSE have Pearson
correlation `0.596`, supporting a redundancy explanation. Raw correlations
between physical regret and normalized counterfactual magnitude or direction
are only `0.049` and `0.064` when all real-model rows are pooled.

## Secondary results

- Counterfactual metrics predict **latent normalized regret** modestly in
  repeated cross-validation, but the cluster-bootstrap confidence intervals
  cross zero. This is an internal latent-space result, not confirmation of
  physical planning relevance.
- Coverage-regret point estimates are weakly positive but bootstrap intervals
  also cross zero.
- Physical pairwise-error analyses are null or negative.
- Direction error alone has significantly negative incremental estimates for
  both physical normalized regret and physical pairwise error.

An exploratory sensitivity analysis allowing horizon-specific slopes does not
rescue the primary claim:

| Exploratory model | Median ΔR² | Positive repetitions |
|---|---:|---:|
| Horizon 1 only | −0.00972 | 0/20 |
| Horizon 3 only | −0.00688 | 1/20 |
| Horizon 6 only | −0.00375 | 7/20 |
| Pooled horizon interactions | −0.00767 | 0/20 |

The one narrow hint is JEPA-WM at horizon 6: exploratory median ΔR² `+0.00294`
with 14/20 positive repetitions. This is post hoc, small, and not sufficient
to overturn the confirmatory decision.

## Contact interpretation

Contact makes prediction harder: at horizon 6, unnormalized paired-effect RMSE
increases from approximately 0.15–0.17 when neither action contacts the object
to approximately 0.36–0.37 when both do.

Yet high latent directional agreement does not reliably become correct
physical ranking. At horizon 6:

| Model / stratum | Effect cosine | Latent pair ranking | Physical pair ranking |
|---|---:|---:|---:|
| DINO / one contacts | 0.864 | 76.0% | 56.5% |
| DINO / both contact | 0.787 | 77.8% | 53.7% |
| JEPA / one contacts | 0.869 | 82.9% | 45.0% |
| JEPA / both contact | 0.780 | 78.5% | 51.9% |

This is the clearest scientific story: the models can predict the direction of
latent state changes while still failing to preserve the task-relevant
ordering of executable physical outcomes.

## Scientific meaning

The experiment separates three claims:

1. **Does the model respond to actions?** Yes.
2. **Can that response improve action selection?** Yes, especially at horizons
   3 and 6, but performance remains imperfect.
3. **Do these two aggregate counterfactual error scores tell us which cases
   will plan well beyond ordinary rollout error and basic difficulty
   covariates?** No.

A likely explanation is objective mismatch. The counterfactual metrics average
error over the latent change vector, whereas physical regret depends on a small
task-relevant projection of the future state. A model may predict most of the
latent change accurately while getting the action ordering near the goal wrong.
Ordinary rollout error already absorbs much of the same information, leaving
the aggregate paired metrics redundant.

## Scope and limitations

- One simulator: Push-T.
- Two checkpoints from the same JEPA-WMs ecosystem, not two broadly independent
  model families.
- A fixed, hand-designed, goal-conditioned action library rather than a learned
  proposal policy or continuous optimizer.
- Aggregate latent metrics may obscure task-relevant dimensions and nonlinear
  or model-specific relationships.
- The results establish simulator behavior under exact controlled
  interventions. They do not establish real-robot reliability.

## Recommendation

Stage 2B satisfies the original Stage 3 gate through the
“surprising, scientifically interpretable negative result” clause, not through
metric non-redundancy.

Proceed only with a scoped Stage 3 replication designed around the negative
claim:

1. Add a second exact-restoration environment and a substantively different
   model family.
2. Pre-register horizon/model interactions rather than adding them post hoc.
3. Retain the current aggregate metrics unchanged as confirmatory baselines.
4. Add an explicitly task-aligned counterfactual metric that weights latent
   errors by their influence on executable cost.
5. Compare fixed-library ranking with policy- or CEM-proposed actions.
6. Do not begin Tier-3 V-JEPA 2-AC scaling or a training experiment yet.

The strongest current paper direction is therefore not “paired metrics solve
planning evaluation.” It is:

> Action-conditioned latent prediction can be demonstrably counterfactual
> without aggregate latent counterfactual error providing incremental evidence
> about executable planning quality.
