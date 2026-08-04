# Stage 17 Colab run guide

Notebook: `notebooks/17_finite_action_contrast_interchange.ipynb`

## Before opening Colab

Create two Colab secrets:

- `STAGE17_RUN_MODE` = `pilot`
- `STAGE17_SOURCE_COMMIT` = the full 40-character commit printed in the handoff

Do not edit notebook cells for the source-bound run. The notebook checks the
executed cell prefix against the exact committed artifact before opening any
evaluation trajectory.

## Run

1. Open the notebook through the supplied **Open in Colab** link.
2. Select a GPU runtime. An L4/G4-class GPU is recommended; a 16 GB T4 is the
   supported minimum.
3. Leave Google Drive mounting enabled so simulator, baseline, subspace, and
   intervention shards survive a disconnect.
4. Choose **Runtime > Run all**. No runtime restart is expected.
5. Return the automatically downloaded
   `stage17_action_contrast_result_bundle_<signature>.zip`.

The notebook begins in `smoke` only when the `STAGE17_RUN_MODE` secret is
missing. Smoke mode validates execution but cannot pass scientifically.

## Expected resources

- First-run setup/checkpoint downloads: approximately 15--30 minutes.
- Full G4/L4-class run: approximately 70--150 minutes, depending on the
  assigned accelerator and Drive throughput.
- T4: approximately 120--210 minutes.
- Peak VRAM: expected below 12 GB; 16 GB is the supported floor.
- Persistent Drive usage: approximately 2--3 GB including cached public model
  assets; the compact returned ZIP should be much smaller.

The notebook times a real 13-candidate, horizon-3 forward batch before the
main intervention matrix and writes `forward_benchmark.json`. Its measured
estimate is more reliable than these envelopes.

## Expected checkpoints

Before evaluation opens, confirm that output includes:

- exact full-dynamic-state and one-step continuation restoration;
- verified hashes for `jepa_wm_pusht.pth.tar` and the DINOv2 encoder;
- zero-edit hook error at or below `1e-6`;
- a construction layer-selection table;
- `subspace_freeze.json` reporting no evaluation trajectories seen;
- an evaluation-open certificate that follows the subspace freeze.

Final output should print one decision label, the result ZIP path, and its
SHA-256 hash. A construction-gate stop is a valid scientific stop, not a
pipeline failure. `FAILURE_TRACE.txt` must contain `NONE` unless execution
actually failed.
