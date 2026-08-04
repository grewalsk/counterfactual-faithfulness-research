# Stage 19 Colab run guide

Notebook: `notebooks/19_unseen_action_family_transfer.ipynb`

## Required Stage 18 artifact

Stage 19 imports the exact raw subspace from the successful Stage 18 run. The
default location is:

`/content/drive/MyDrive/counterfactual_faithfulness_stage18_rank64/pilot_f1b34beffcac/subspaces/frozen_rank64_confirmation_subspaces.npz`

Its required SHA-256 is
`2f9c496d54623a9062e465a18c70039acc18cb8a1cc2833a5f4ade162ca3f90b`.
Do not rename or modify the successful Stage 18 run directory. If the artifact
is elsewhere in Drive, create the optional `STAGE19_STAGE18_SUBSPACE_PATH`
secret with its full `.npz` path. The compact Stage 18 result ZIP is not enough;
the raw subspace was deliberately excluded because it is large.

## Colab secrets

Create:

- `STAGE19_RUN_MODE` = `pilot`
- `STAGE19_SOURCE_COMMIT` = the full 40-character commit in the handoff link
- `STAGE19_RUN_NONCE` = a new label, for example `transfer_20260804_a`

Do not edit notebook cells. Stage 19 verifies the committed notebook prefix,
the Stage 19 source files, the Stage 18 decision/provenance files, and the raw
subspace hash before loading the model. A reused nonce aborts.

## Run

1. Open the supplied **Open in Colab** link.
2. Select a GPU runtime; L4 or better is recommended.
3. Keep Google Drive mounting enabled.
4. Choose **Runtime > Run all**.
5. Return `stage19_unseen_action_transfer_result_bundle_<signature>.zip`.

The pilot screens 64 fresh states for each of five prespecified action families,
selects 24 per family using simulator truth only, generates 120 model baselines,
and runs 30 interventions per selected record. The families are interleaved
directions, magnitudes 0.08/0.16, and delayed/pulsed equal-impulse profiles.

Approximate end-to-end times:

- RTX PRO 6000 Blackwell-class GPU: 12–25 minutes;
- L4: 20–45 minutes;
- T4: 45–90 minutes.

The notebook writes a measured 120-record intervention estimate before the
main causal loop. Simulator truth generation is CPU-bound and is not fully
captured by that estimate. Expect roughly 1 GB of additional Stage 19 raw
shards, plus the existing Stage 18 artifact and model cache in Drive.

## Required audit outputs

- `stage18_artifact_certificate.json` must report the exact expected hash and
  validation before Stage 19 model activations.
- `physical_selection_freeze.json` must contain 24 selected trajectories per
  family in pilot mode and state that selection used simulator truth only.
- `hook_identity_test.json` must pass at error at most `1e-6`.
- `fresh_run_certificate.json` must report 320 generated truth records, 120
  generated baselines, 120 generated intervention shards, and zero cache hits.
- `stage19_decision.json` reports each family separately. The broad-transfer
  status is permitted only if all five bidirectional gates pass.
- `FAILURE_TRACE.txt` must contain `NONE` for a completed run.

If the raw Stage 18 artifact is missing or has a different hash, stop. Do not
reconstruct or refit it inside Stage 19; that would invalidate the transfer
test.
