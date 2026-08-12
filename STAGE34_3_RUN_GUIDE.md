# Stage 34.3 regime-aware JEPA innovation Colab run guide

Notebook: `notebooks/34_3_regime_innovation_diagnostic.ipynb`

## Required Drive directories

Keep both complete output directories at their original paths:

```text
MyDrive/counterfactual_faithfulness_stage34_pfca/pilot_d3f4f88426af/
MyDrive/counterfactual_faithfulness_stage34_2_split_path/pilot_9fcedcf036a8/
```

The downloaded Stage 34 bundle is insufficient because Stage 34.3 reads the
256 raw JEPA transition shards. The complete Stage 34.2 directory is normally
already present from the preceding run.

## Run

1. Open `notebooks/34_3_regime_innovation_diagnostic.ipynb` from the committed
   branch in a fresh Colab runtime.
2. A standard CPU runtime is sufficient. A GPU or G4 runtime provides no
   scientific benefit because no checkpoint is loaded.
3. Leave `RUN_MODE = "pilot"` and select **Runtime -> Run all**.
4. Authorize Drive.
5. Return
   `stage34_3_regime_innovation_result_bundle_<signature>.zip`.

The notebook evaluates 80 frozen hyperparameter candidates with
trajectory-grouped folds, then performs one locked evaluation and 5,000-draw
cluster bootstraps. Expected wall time is approximately 10–30 minutes on a
standard Colab CPU, depending on the assigned BLAS runtime and Drive speed.

## Read the result

Read `stage34_3_decision.json` first, then
`evaluation_evidence/falsification_control_summary.json`. Interpretation
requires `FAILURE_TRACE.txt = NONE` and a valid result manifest.

A pass does not authorize a causal or shared-mechanism claim. It nominates one
small JEPA state for a fresh-trajectory confirmation. Any failure identifies
the first falsification gate and should stop further native interventions on
this candidate.
