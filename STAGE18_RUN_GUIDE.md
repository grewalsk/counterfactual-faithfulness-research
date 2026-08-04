# Stage 18 Colab run guide

Notebook: `notebooks/18_rank64_action_contrast_confirmation.ipynb`

## Before opening Colab

Create three Colab secrets:

- `STAGE18_RUN_MODE` = `pilot`
- `STAGE18_SOURCE_COMMIT` = the full 40-character commit printed in the handoff
- `STAGE18_RUN_NONCE` = a new label used for no prior Stage 18 run, for example
  `confirm_20260804_a`

Do not edit notebook cells for the source-bound run. The notebook verifies the
exact committed notebook, builder, numerical module, configuration, and
executed cell prefix before it opens evaluation model activations. Reusing a
nonce deliberately aborts rather than resuming cached evidence.

## Run

1. Open the notebook through the supplied **Open in Colab** link.
2. Select a GPU runtime. L4 is the preferred Colab GPU; a 16 GB T4 is the
   supported minimum.
3. Leave Google Drive mounting enabled so public checkpoint downloads and the
   fresh run directory survive a transient disconnect. Do not rerun the same
   nonce after a disconnect; start a new nonce for valid confirmatory evidence.
4. Choose **Runtime > Run all**. No runtime restart is expected.
5. Return the automatically downloaded
   `stage18_rank64_result_bundle_<signature>.zip`.

If `STAGE18_RUN_MODE` is absent, the notebook runs a two-construction/two-eval
smoke test at rank 8. Smoke validates execution only and cannot support the
Stage 18 hypothesis.

## Expected resources

- First-run setup and verified checkpoint downloads: approximately 15--30
  minutes.
- Full L4-class pilot: approximately 45--110 minutes, depending on the assigned
  accelerator and Drive throughput.
- T4: approximately 90--180 minutes.
- Peak VRAM: expected below 12 GB; 16 GB is the supported floor.
- Persistent storage: approximately 3--5 GB including public model assets and
  raw fresh-run shards. The returned audit ZIP excludes raw arrays and should
  be much smaller.

The notebook times a real 13-candidate, horizon-3 forward and multiplies by the
frozen 42 interventions per evaluation trajectory. It writes
`forward_benchmark.json` and flags estimates above the configured three-hour
warning threshold. That device-specific estimate is more reliable than these
envelopes.

## Expected checkpoints

Before evaluation model activations open, confirm that output includes:

- exact full-dynamic-state and one-step continuation restoration;
- verified hashes for the JEPA-WM and DINOv2 checkpoints;
- a physical-selection freeze reporting 24 construction and 32 evaluation
  states chosen before model load;
- zero-edit hook error at or below `1e-6`;
- a fixed block-4 construction geometry gate;
- `subspace_freeze.json` reporting a rank-128 ceiling and no evaluation model
  activations seen;
- an evaluation-open certificate whose hash binds the subspace freeze.

Final output prints the candidate decision, claim-eligibility flags, result ZIP
path, and ZIP SHA-256. A construction-gate stop is a valid scientific stop.
`fresh_run_certificate.json` must show zero cache hits, and
`FAILURE_TRACE.txt` must contain `NONE` unless execution actually failed.
