# Stage 22 Colab run guide

Notebook: `notebooks/22_latent_hybrid_gate_interaction.ipynb`

Stage 22 is standalone: it downloads the frozen public JEPA-WM and the pinned
physical-decoder asset.  It does not require a Stage 18–21 result bundle.

## Colab secrets

Create these three secrets and enable notebook access for each:

- `STAGE22_RUN_MODE`: `pilot`
- `STAGE22_SOURCE_COMMIT`: the full 40-character commit from the handoff
- `STAGE22_RUN_NONCE`: a new value, for example `hybrid_gate_20260804_a`

The secret names and values are separate fields.  Do not put `pilot`, the
commit, or the nonce in a secret whose name is truncated to `STAGE22_`.

Select an L4 GPU when available and choose **Runtime → Run all**.  Do not edit
or selectively rerun source-bound cells.  The notebook measures a forward-pass
benchmark before the factorial workload and records the estimate.

## Expected artifact

The final cell downloads
`stage22_hybrid_gate_result_bundle_<signature>.zip`.  Preserve the zip without
renaming its internal files.  The decisive files are:

- `stage22_decision.json`
- `analysis/construction_mode_selection.json`
- `evaluation_evidence/heldout_contact_alignment.json`
- `evaluation_evidence/factorial_interaction_rows.csv`
- `plots/stage22_hybrid_gate_summary.png`
- `source_identity.json`
- `FAILURE_TRACE.txt`

Smoke mode checks execution only and cannot authorize a scientific claim.
