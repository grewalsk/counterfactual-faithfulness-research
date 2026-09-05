# Stage 44 visual--causal realization audit

Notebook: `notebooks/44_visual_causal_realization_audit.ipynb`

Stage 44 is the decision experiment after the negative Stage 43 recursive-reset
lattice. It retains the native `16 x 16 x 384` visual fields and compares the
true target encoding, a one-step teacher-forced prediction, and an ordinary
recursive prediction. Frozen physical and patch-localization probes test what
the fields contain. Local tangent/normal diagnostics test support escape, and
matched action pairs plus object/background patch swaps test whether visual
counterfactual content is specific to the manipulated object.

The notebook also loads the official upstream VM2M decoder heads and writes
RGB panels. Those panels are diagnostic only: the registered decision never
uses visual plausibility or reconstruction MSE as a scientific gate.

## Run

1. Open the notebook in a fresh Colab runtime.
2. Leave `RUN_MODE="pilot"` and all scientific settings unchanged.
3. Select the G4 / RTX PRO 6000 Blackwell runtime when available. An A100 is a
   reasonable alternative; L4 is supported but slower.
4. Keep `MOUNT_DRIVE=True`. Dense float16 visual summaries, truth images,
   source hashes, and phase checkpoints are resumable under
   `/content/drive/MyDrive/counterfactual_faithfulness_stage44_vcra/`.
5. Run all thirteen cells in order.
6. Return the automatically downloaded
   `stage44_vcra_result_bundle_<signature>.zip`.
   If the run stops, return the failure bundle without deleting the resumable
   Drive directory.

Estimated first-run wall time, including checkpoint and two public decoder
downloads:

- RTX PRO 6000 Blackwell / G4: about 1–2 hours;
- A100 80 GB: about 1.5–2.5 hours;
- L4: about 3–5 hours.

These are engineering estimates, not Colab billing rates. Reuse of verified
model/decoder caches and completed truth shards can shorten a resumed run.
Expect several gigabytes of persistent storage because the public image
decoder heads, RGB truth paths, and dense projected patch fields are retained.

## Expected result

The terminal output reports `RUN_STATUS`, the complete resumable Drive
directory, the result-bundle path and SHA256, counts, GPU, and elapsed time.
The bundle includes:

- `stage44_decision.json` and `AUTOMATIC_INTERPRETATION.md`;
- frozen probe and source manifests with decoder hashes;
- held-out summary and per-row CSV evidence;
- teacher-versus-recursive geometry plots;
- true RGB, target-decoder, teacher-forced, and recursive visual panels;
- a complete failure trace if any contract or scientific precondition fails.

The decision can authorize an object-centric encoder experiment,
counterfactual one-step training, recursive closure training, or a sealed
planning-objective audit. It cannot itself authorize a causal, deployment,
minimal-state, or general-world-model claim.
