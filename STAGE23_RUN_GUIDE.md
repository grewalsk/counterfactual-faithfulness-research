# Stage 23 Colab run guide

Notebook: `notebooks/23_causal_mode_manifold_operator_switch.ipynb`

Stage 23 imports the exact frozen Stage 22 mode partition from Google Drive.
Keep the complete Stage 22 run directory at:

`MyDrive/counterfactual_faithfulness_stage22_hybrid_gate/pilot_7b0be321cc7d`

The downloaded compact Stage 22 ZIP alone is insufficient because it excludes
the large `subspaces/frozen_mode_partition.npz` file.  The notebook verifies
the upstream source commit, decision, selected block, and partition SHA-256
before using it.

## Colab secrets

Create these three secrets and enable notebook access for each:

- `STAGE23_RUN_MODE`: `pilot`
- `STAGE23_SOURCE_COMMIT`: the full 40-character commit from the handoff
- `STAGE23_RUN_NONCE`: a new value, for example `mode_operator_20260805_a`

Select the fastest available GPU and choose **Runtime → Run all**.  Do not edit
or selectively rerun source-bound cells.  Stage 23 benchmarks the exact
patched-forward workload before beginning it.  The frozen pilot performs
5,184 patched candidate-batch forwards (64 pairs × 81 forwards), so it is
intentionally much larger than Stage 22.

## Expected artifact

The final cell downloads
`stage23_mode_operator_result_bundle_<signature>.zip`.  The decisive files are:

- `stage23_decision.json`
- `evaluation_evidence/mode_transport_diagnostics.csv`
- `evaluation_evidence/operator_transfer_rows.csv`
- `evaluation_evidence/layerwise_operator_transfer.csv`
- `evaluation_evidence/heldout_contact_alignment.json`
- `plots/stage23_mode_operator_summary.png`
- `subspaces/stage22_upstream_binding.json`
- `source_identity.json`
- `FAILURE_TRACE.txt`

Smoke mode checks execution only and cannot authorize a scientific claim.
