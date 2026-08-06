# Stage 24 Colab run guide

Notebook: `notebooks/24_causal_completion_rank.ipynb`

Stage 24 imports the exact frozen Stage 23 partition and geometry from Google
Drive.  Keep the complete Stage 23 run directory at:

`MyDrive/counterfactual_faithfulness_stage23_mode_operator/pilot_d47dee8b6789`

The compact Stage 23 ZIP is insufficient because it excludes the large
`frozen_mode_partition.npz` and `frozen_mode_operator_geometry.npz` files.
Stage 24 verifies the Stage 23 source commit, decision, Stage 22 binding,
transport certificate, partition hash, and geometry hash before proceeding.

## Colab secrets

Create these three secrets and enable notebook access for each:

- `STAGE24_RUN_MODE`: `pilot`
- `STAGE24_SOURCE_COMMIT`: the full 40-character commit from the handoff
- `STAGE24_RUN_NONCE`: a new value, for example `causal_completion_20260805_a`

Select the fastest available GPU and choose **Runtime → Run all**.  Do not edit
or selectively rerun source-bound cells.  The pilot performs exactly 18,496
patched candidate-batch forwards: 64 held-out pairs × 289 forwards.  On the
RTX PRO 6000 Blackwell used for Stage 23, expect roughly 30–45 minutes plus
setup.  The notebook records an exact-machine benchmark before interventions.

## Scientific decision

The notebook freezes completion ranks `[0, 4, 8, 16, 32, 64]`.  It reports
`CCR_0.8`, the smallest rank whose trajectory-bootstrap 95% lower confidence
bound recovers at least 80% of the native contact-operator change.  A positive
result must also beat both same-mode and random mode-null bases by at least
0.20 with positive lower confidence bounds and the preregistered sign criterion.

## Expected artifact

The final cell downloads
`stage24_causal_completion_result_bundle_<signature>.zip`.  The decisive files
are:

- `stage24_decision.json`
- `evaluation_evidence/causal_completion_rank_curve.csv`
- `evaluation_evidence/causal_completion_operator_rows.csv`
- `evaluation_evidence/completion_context_diagnostics.csv`
- `evaluation_evidence/completion_pair_diagnostics.csv`
- `evaluation_evidence/layerwise_completion_curve.csv`
- `plots/stage24_causal_completion_summary.png`
- `subspaces/causal_completion_basis_freeze.json`
- `subspaces/stage23_upstream_binding.json`
- `source_identity.json`
- `FAILURE_TRACE.txt`

Smoke mode checks execution only and cannot authorize a scientific claim.
