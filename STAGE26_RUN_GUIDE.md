# Stage 26 Colab run guide

Notebook: `notebooks/26_contact_frame_causal_transport.ipynb`

Stage 26 binds the exact Stage 25 negative stored in this complete Drive run:

`MyDrive/counterfactual_faithfulness_stage25_causal_kkt/pilot_0c557d94ceae`

The compact downloaded ZIP is insufficient because the notebook deliberately
requires the complete source-bound upstream directory.

## Colab secrets

Create and enable notebook access for:

- `STAGE26_RUN_MODE`: `pilot`
- `STAGE26_SOURCE_COMMIT`: the full 40-character commit from the handoff
- `STAGE26_RUN_NONCE`: a new value, such as
  `contact_transport_20260806_a`

The nonce identifies a fresh execution; it is not a password or access token.
Select an L4, G4-class, or faster GPU and use **Runtime → Run all**. Do not edit
or selectively rerun source-bound cells.

The run creates new paired simulator truth for two 100-state pools, opens 40
construction and 40 evaluation states, captures all six blocks only on
construction, then runs six patched candidate batches per evaluation state.
Plan for approximately 15--35 minutes after cached dependency/checkpoint
downloads. Simulator rendering is mostly CPU-bound, and the notebook prints a
machine-specific estimate for its evaluation-forward phase.

Keep the complete new Drive directory. The final cell also downloads
`stage26_contact_transport_result_bundle_<signature>.zip` containing the
decision, layer-selection table, intervention rows, edit diagnostics, source
and upstream bindings, timings, plot, and failure trace.

Smoke mode is only an execution check and always returns `SMOKE_ONLY`.
