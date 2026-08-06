# Stage 29 grounded causal closure: Colab run guide

Notebook: `notebooks/29_grounded_causal_closure.ipynb`

Stage 29 requires the **complete Stage 28 Drive run directory** produced by the
source-bound run at commit `917228edb9e7143c58bdd9640afe08ead75fa34c`.
The small downloaded result zip is not sufficient because Stage 29 reuses the
exact simulator endpoint images in the excluded `truth/` shards.

No Stage 29 Colab secrets are required. The notebook automatically:

- uses `RUN_MODE = "pilot"`;
- generates a fresh nonce;
- resolves its committed GitHub branch to an exact commit;
- locates and validates the frozen Stage 18 carrier;
- locates the correct Stage 28 run and hashes all 36 selected truth shards;
- reads `HF_TOKEN` only if the pinned checkpoint is not already cached.

## Run

1. Open the committed notebook in Colab.
2. Select a GPU runtime. An L4 is the notebook default; another CUDA GPU is OK.
3. Choose **Runtime → Run all**. Do not execute isolated cells or edit the
   configuration cell, because the source-binding check verifies the exact
   committed prefix.
4. Authorize Google Drive mounting when prompted.
5. Keep the complete output directory:
   `MyDrive/counterfactual_faithfulness_stage29_grounded_closure/pilot_<signature>/`.
6. Upload the downloaded
   `stage29_grounded_closure_result_bundle_<signature>.zip` for interpretation.

The benchmark cell reports the measured predictor time before the full loop.
The experiment uses one 24-branch baseline, one wrong-state baseline, two
target-encoder batches, and nine intervention batches per record. Its compact
result zip excludes the raw per-record shards but includes their cryptographic
manifest.

## Primary outcome labels

- `CAUSAL_SELF_CONSISTENCY_WITHOUT_GROUNDED_CLOSURE`: the frozen carrier moves
  predictions toward the model's own donor future but not the encoded exact
  simulator future.
- `PREDICTOR_TARGET_CLOSURE_FAILED`: the native predicted area contrast does
  not match the encoder representation of the exact future.
- `PHYSICAL_READOUT_LIMITATION_SUPPORTED`: native predictor-target closure
  passes, while the frozen physical decoder fails on encoded true futures.
- `ENCODED_TRUE_FUTURES_DO_NOT_RESOLVE_CONTROL_AREA`: the target encoder itself
  is insensitive to the physically verified contrast.
- `GROUNDED_CAUSAL_CLOSURE_SUPPORTED`: native prediction and the frozen causal
  intervention both close against encoded simulator futures.
