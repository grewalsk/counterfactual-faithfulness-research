# Stage 25 Colab run guide

Notebook: `notebooks/25_causal_kkt_tomography.ipynb`

Stage 25 imports the exact frozen Stage 23 partition and geometry through the
successful provenance chain stored in the complete Stage 24 Drive run. Keep
this directory at exactly:

`MyDrive/counterfactual_faithfulness_stage24_causal_completion/pilot_b18ea7810677`

The downloaded compact Stage 24 ZIP is insufficient because the notebook
requires the large inherited NPZ geometry files. Before loading them, Stage 25
checks the Stage 24 source commit and negative decision, the nested Stage 23
binding, and both frozen artifact hashes.

## Colab secrets

Create these three secrets and enable notebook access for each:

- `STAGE25_RUN_MODE`: `pilot`
- `STAGE25_SOURCE_COMMIT`: the full 40-character commit from the handoff
- `STAGE25_RUN_NONCE`: a fresh value, for example
  `latent_kkt_20260806_a`

Select the fastest available GPU and choose **Runtime → Run all**. Do not edit
or selectively rerun source-bound cells. The nonce identifies this particular
fresh execution; it is not an access token.

The pilot generates two simulator branches for every state/action pair and
encodes both endpoints, so its first substantial phase is CPU/rendering heavy.
The intervention phase is small: only three patched candidate-batch forwards
per selected evaluation state. A reasonable planning envelope on an RTX PRO
6000 Blackwell/G4-class Colab runtime is about 20–45 minutes plus dependency
and checkpoint download time. The notebook records its own machine-specific
intervention-forward estimate against the frozen 120-minute warning threshold;
the paired simulator phase is reported separately in its timing artifact.

## Required sanity evidence

The run is not scientifically interpretable unless all of these are present:

- both construction and evaluation pools select the requested number of
  physically eligible states;
- ordinary branches contain nonzero agent–block contact events and ghost
  branches contain none;
- paired initial observations and restored dynamic states pass exact checks;
- the per-physics-step `median_contact_momentum_residual` passes the
  instrumentation gate;
- held-out reader and causal intervention rows are nonempty;
- source and Stage 24 upstream bindings are fully true.

## Expected artifact

The final cell downloads
`stage25_causal_kkt_result_bundle_<signature>.zip`. The decisive files are:

- `stage25_decision.json`
- `evaluation_evidence/physical_eligibility_rows.csv`
- `evaluation_evidence/construction_reader_cv.csv`
- `evaluation_evidence/heldout_impulse_reader.json`
- `evaluation_evidence/heldout_impulse_reader_rows.csv`
- `evaluation_evidence/causal_impulse_intervention_rows.csv`
- `evaluation_evidence/causal_impulse_edit_diagnostics.csv`
- `subspaces/impulse_reader_freeze.json`
- `subspaces/stage24_upstream_binding.json`
- `plots/stage25_causal_kkt_summary.png`
- `source_identity.json`
- `FAILURE_TRACE.txt`

Smoke mode checks execution only and cannot support a scientific claim.
