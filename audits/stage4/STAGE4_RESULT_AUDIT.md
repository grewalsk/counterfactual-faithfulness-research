# Stage 4 matched-intervention result audit

## Bottom line

Stage 4 returned:

`CROSS_ENV_ACTION_STRUCTURE_CAUSAL_SIGNAL`

The frozen co-primary gate passed independently in PushT and Wall. At exactly
matched decoded-pose perturbation magnitude, corrupting which predicted
consequence belongs to which candidate action caused substantially more
planning damage than applying a shared common-mode decoded-state error.

The defensible conclusion is:

> Action-specific structure in the decoded predicted consequences is causally
> necessary for planning under the fixed PushT and Wall candidate sets. The
> result cannot be explained by perturbation magnitude alone.

This result does not reverse the Stage 3B natural-failure result. Task-margin
error still did not improve held-out per-instance regret prediction under
natural model variation. Stage 4 instead establishes causal construct validity
under controlled, magnitude-matched interventions.

## Integrity

The independent audit passed every check:

- source Stage 3B run reported success and no failure trace;
- 72,000 intervention-unit rows had unique primary keys;
- both environments retained all 40 final-test state clusters;
- five severities and five deterministic intervention seeds were present;
- every full permutation was a derangement;
- intact decoded costs reconstructed to floating-point precision;
- maximum action-versus-common pose-RMS mismatch was
  `2.78e-16`, far below the frozen `1e-10` threshold;
- tied-cost no-decision units used one common finite sample for both
  co-primary endpoints;
- all reported primary and dose-response bootstrap intervals reproduced
  independently from the exported rows;
- every result-manifest entry was present.

## Confirmatory result

Positive values favor the hypothesis that action-structure corruption is more
damaging than an equally large common-mode perturbation.

| Environment | Specific normalized-regret damage (95% CI) | Specific weighted-ranking damage (95% CI) |
|---|---:|---:|
| PushT | 0.179 [0.129, 0.229] | 0.241 [0.203, 0.277] |
| Wall | 0.210 [0.189, 0.230] | 0.423 [0.408, 0.439] |

All four lower confidence bounds are well above zero.

At full severity:

- PushT regret was 0.418 under action-structure corruption versus 0.257 under
  common-mode corruption; weighted ranking was 0.468 versus 0.709.
- Wall regret was 0.401 versus 0.191; weighted ranking was 0.452 versus 0.876.

## Dose response

The pre-specified descriptive slope contrasts also passed clearly:

| Environment | Specific regret slope (95% CI) | Specific ranking-loss slope (95% CI) |
|---|---:|---:|
| PushT | 0.192 [0.131, 0.247] | 0.260 [0.221, 0.299] |
| Wall | 0.193 [0.172, 0.213] | 0.456 [0.441, 0.471] |

The action-structure and common-mode curves begin at the same intact endpoint.
As perturbation severity increases, action-structure corruption separates
sharply in both physical regret and weighted action ranking.

## Breadth

The primary direction was positive in all 12 descriptive
environment-by-model-by-horizon cells:

- PushT and Wall;
- DINO-WM and JEPA-WM;
- horizons 1, 3, and 6.

Top-1 specificity was also positive in all 12 cells. Task-margin specificity
was positive in 11 of 12 cells.

## Paper implication

Stages 2B through 4 now support a coherent distinction:

1. raw latent distance is not a reliable counterfactual planning metric;
2. a simple task-aligned physical-state readout enables substantially better
   planning;
3. natural per-instance task-margin error does not independently predict
   regret beyond ordinary errors;
4. nevertheless, controlled destruction of action-specific decoded
   consequences causes much greater planning damage than an equally large
   common-mode decoded-state error.

The main paper claim should therefore concern **error structure**, not a
universal scalar failure score:

> For planning, preserving the correspondence between actions and their
> predicted physical consequences matters more than minimizing an equally
> large shared prediction error.

## Remaining limitation and next experiment

Stage 4 intervenes on the frozen decoded predictions. It establishes
readout-level causal necessity but does not show that changing a training
objective creates better action-specific representations.

The next ICLR-strengthening experiment should be an equal-data, equal-compute
training comparison between:

- ordinary prediction loss;
- independent paired examples;
- explicit counterfactual-difference loss;
- shuffled-pair control.

The decisive endpoint is improved held-out action ranking and planning at
matched ordinary rollout error, not merely a lower training loss.
