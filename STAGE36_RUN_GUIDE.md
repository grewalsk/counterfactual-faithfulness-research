# Stage 36 predictive-state closure distillation Colab run guide

Version 2 repairs the model-free v1 helper-dependency failure. The v1 run
stopped before loading JEPA-WM or observing any scientific statistic. No
scientific design, seed, candidate, control, threshold, or claim changed.

Notebook: `notebooks/36_predictive_state_closure_distillation.ipynb`

## Requirements

- Open the notebook from the committed repository branch. Source binding rejects
  copied or edited protocol cells.
- Select a GPU runtime. Google G4 (`RTX PRO 6000 Blackwell`) is preferred;
  A100 and L4 are supported. T4 is expected to be substantially slower.
- No Stage 35 output is consumed. Stage 36 generates a fresh panel and uses
  Stage 35 only as the preregistered scientific motivation.
- `HF_TOKEN` is optional unless the public checkpoint host requires it in the
  assigned runtime.

## Run

1. Leave `RUN_MODE = "pilot"`.
2. Select **Runtime -> Run all**.
3. Authorize Google Drive.
4. Do not execute protocol cells out of order. Exact notebook-prefix
   verification is part of the evidence.
5. Return `stage36_pscd_result_bundle_<signature>.zip`.

The notebook writes hash-validated truth, carrier paths, frozen selection,
adapter, controls, evaluation certificate, and final evidence under:

```text
MyDrive/counterfactual_faithfulness_stage36_pscd/
```

Rerunning **Run all** resumes compatible truth and carrier shards. A hash
mismatch stops the run rather than silently mixing protocols. Transient Google
Drive operations use bounded retries.

## Expected resources

The pilot uses 16/16/16/32 complete trajectory families and one frozen JEPA-WM
checkpoint. It screens 24 adapter configurations, then trains one primary and
two capacity controls. Carrier paths are generated once and reused across all
candidate state definitions.

- G4 Blackwell: approximately 1.5--2.5 hours; reserve 3 GPU-hours.
- A100: approximately 1.8--3.2 hours.
- L4: approximately 3--5 hours.
- T4: approximately 6--10 hours.

Runtime depends on Colab allocation and Drive latency. The notebook records the
actual GPU model, total elapsed time, memory peak, forward count, and cache hits.

## Read the result

Require `FAILURE_TRACE.txt = NONE` and verify `result_zip_manifest.json`. Read:

1. `stage36_decision.json`;
2. `evaluation_evidence/predictive_state_closure_summary.json`; and
3. `evaluation_evidence/locked_predictive_state_closure_rows.csv`.

A full pass supports a bounded adapter-distillation result. It does not show
that the original JEPA carrier was already recursively closed, identify a
minimal predictive state, or provide native causal-intervention evidence.
