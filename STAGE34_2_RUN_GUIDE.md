# Stage 34.2 split-path continuation Colab run guide

Notebook: `notebooks/34_2_split_path_continuation.ipynb`

## Required Drive directories

Keep both complete directories at their original paths:

```text
MyDrive/counterfactual_faithfulness_stage34_pfca/pilot_d3f4f88426af/
MyDrive/counterfactual_faithfulness_stage34_1_action_specificity/pilot_208b72f57074/
```

The compact downloaded Stage 34 bundle is insufficient because Stage 34.2
uses the raw model and transition shards.

## Run

1. Open `notebooks/34_2_split_path_continuation.ipynb` from the committed
   branch in a fresh Colab runtime.
2. Prefer a full G4 Blackwell runtime. L4 is supported.
3. Leave `RUN_MODE = "pilot"` and select **Runtime -> Run all**.
4. Authorize Drive. No secret is required; `HF_TOKEN` remains optional for the
   public checkpoint download.
5. Return `stage34_2_split_path_result_bundle_<signature>.zip`.

The notebook first runs the CPU-only DINO diagnostic and JEPA sufficiency
gate. If JEPA sufficiency fails, it packages the result without installing or
loading the model. If sufficiency passes, it installs the pinned runtime and
runs 64 native matched pairs, checkpointing each pair independently.

## Runtime planning

| Path | G4 Blackwell | L4 |
|---|---:|---:|
| DINO diagnosis + JEPA sufficiency | 5–15 min | 5–15 min |
| Conditional JEPA causal gate | 30–90 min | 1–3 hr |

Drive verification and public checkpoint download can dominate. If Colab
disconnects during causal inference, reconnect and use **Run all**; completed
pair shards are hash-validated and reused.

## Read the result

Read `stage34_2_decision.json` before the plot. The result is explicitly
post-outcome diagnostic evidence and never revives the rejected shared
JEPA–DINO claim. `FAILURE_TRACE.txt = NONE` is required for interpretation.
