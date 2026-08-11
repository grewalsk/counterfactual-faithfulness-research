# Stage 34.1 action-specificity repair Colab run guide

Notebook: `notebooks/34_1_action_specificity_repair.ipynb`

Stage 34.1 reanalyzes the exact frozen Stage 34 calibration/evaluation shards
with a genuinely action-blind comparator. It does not load either checkpoint,
run PushT, or perform GPU inference.

## Before running

Keep this complete directory in Google Drive:

`MyDrive/counterfactual_faithfulness_stage34_pfca/pilot_d3f4f88426af/`

The 27 MB result bundle alone is insufficient. The full directory is roughly
477 MB and contains the truth and model shards recorded in `raw_manifest.json`.

## Run

1. Open `notebooks/34_1_action_specificity_repair.ipynb` from the committed
   branch in Google Colab.
2. A CPU runtime is sufficient. An attached GPU is harmless but unused.
3. Leave `RUN_MODE = "pilot"` for the diagnostic result.
4. Select **Runtime -> Run all** and authorize Google Drive.
5. Return the downloaded
   `stage34_1_action_specificity_result_bundle_<signature>.zip`.

No Hugging Face token or other secret is required. The notebook verifies every
consumed shard against the audited Stage 34 manifest before evaluating either
model.

## Expected runtime

On Colab, manifest verification is expected to take approximately 2–8 minutes
depending on Drive throughput. The two CPU regressions and bootstrap normally
take another 2–6 minutes. A reasonable planning range is **5–15 minutes**.

## Reading the result

Read `stage34_1_decision.json` first.

- `ACTION_SPECIFICITY_REPAIRED_CONTINUE_STAGE34`: the leakage-free diagnostic
  passed for both models; proceed to a separately frozen continuation of the
  unobserved sufficiency and causal gates.
- `ACTION_SPECIFICITY_NOT_ESTABLISHED`: stop the abstraction chain and inspect
  the named failed gate and model summaries.
- `INCONCLUSIVE_UPSTREAM_BINDING_FAILURE`: restore the exact full Stage 34
  Drive directory; this is not a scientific result.

The notebook is deliberately labeled post-outcome diagnostic evidence. A pass
does not overwrite the original Stage 34 preregistered decision and is not a
full causal-abstraction result.
