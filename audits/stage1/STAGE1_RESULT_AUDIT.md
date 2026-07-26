# Stage 1 result audit

## Decision

**PASS — proceed to a decisive Stage 2 pilot.**

The returned bundle is a clean, complete pipeline validation. It is not a
hypothesis test and does not yet establish that counterfactual error is
non-redundant or planning-relevant.

## Run integrity

- Full mode: 20 independent initial states, 4 action alternatives, horizons 1
  and 2.
- Hardware: NVIDIA A100-SXM4-40GB; PyTorch 2.11.0+cu128; CUDA 12.8.
- Model: `dino_wm_pusht`; the predictor, action encoder, and proprio encoder
  checkpoint keys all matched.
- Exact restoration passed three repetitions:
  - initial render bitwise exact;
  - endpoint state bitwise exact;
  - contact diagnostics exact;
  - maximum endpoint difference: 0.
- All 20 state checkpoints completed.
- All arrays are finite.
- Recomputed ordinary error, paired error, and regret match the exported CSV
  values exactly.
- `FAILURE_TRACE.txt` contains `NONE`.

The main result tensor has shape `[20 states, 4 actions, 2 horizons, 6144
features]`.

## Descriptive results

| Horizon | Ordinary RMSE | Paired-effect RMSE | Normalized paired error | Effect cosine | Top-1 action accuracy | Normalized regret | Pairwise accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.341 | 0.449 | 0.854 | 0.571 | 0.55 | 0.379 | 0.55 |
| 2 | 0.358 | 0.478 | 0.712 | 0.697 | 0.45 | 0.485 | 0.55 |

Cluster-bootstrap 95% intervals are wide, as expected with only 20 states. For
example, top-1 accuracy is 0.35–0.75 at horizon 1 and 0.25–0.65 at horizon 2.

## What the pilot suggests

The model does not rank the four executable alternatives reliably in this
small run. Its top-1 accuracy falls from 55% to 45% across the two horizons,
and pairwise accuracy is 55% at both horizons.

Ordinary and paired-effect RMSE are almost collinear in this design (Pearson
`r = 0.982`). This is partly structural: with a complete action set and squared
error, paired-effect RMSE is exactly

`sqrt(2A/(A-1)) × action-centered RMSE`.

For four actions, the multiplier is 1.632993, which the output reproduces to
floating-point precision. The useful comparison is therefore not “paired
error versus all action-centered errors”; it is whether separating
action-dependent error from common-mode rollout error predicts planning
failure better than aggregate ordinary error.

An exploratory repeated grouped cross-validation gives a reason to run Stage
2. After adjusting for horizon and simulator effect scale, normalized paired
error improved prediction of normalized regret in all 200 split seeds, with a
median held-out delta R² of approximately 0.49. This is a suppression pattern:
the unadjusted correlation is near zero, and both the base and full held-out R²
values are poor in this tiny sample. It is a design signal, not confirmatory
evidence.

## Stage 1 limitations that Stage 2 must fix

1. There are only 20 independent states.
2. Every state-horizon row is marked contact-positive, so the current
   state-level summary cannot compare contact and non-contact regimes.
3. The complete-action paired RMSE is algebraically redundant with
   action-centered RMSE; Stage 2 must retain pair-level errors and explicit
   common-mode decomposition.
4. The end-to-end ranking selects actions by predicted latent distance to the
   goal but evaluates regret with physical simulator cost. That is a valid
   planning-system test, but it combines representation and dynamics errors.
   Stage 2 must additionally report representation-matched latent-cost ranking
   to isolate dynamics.
5. Only one checkpoint was evaluated.

## Stage 2 design response

The decisive pilot should use:

- 250 exact simulator states;
- 10 alternative action sequences;
- horizons 1, 3, and 6;
- both `dino_wm_pusht` and `jepa_wm_pusht`;
- action-pair contact strata (`neither`, `one`, or `both` branches contact);
- aggregate ordinary error plus common-mode and action-dependent components;
- pair-level intervention-effect error;
- both latent-consistent and executable physical planning regret;
- state-grouped held-out regression and clustered bootstrap intervals;
- action-shuffled and action-blind negative controls.

The confirmatory decision remains: after controlling for ordinary rollout
error, horizon, simulator effect scale, and contact regime, does
counterfactual error improve held-out prediction of ranking error or regret?

