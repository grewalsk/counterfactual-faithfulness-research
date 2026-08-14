# Stage 35 JEPA hybrid composition Colab run guide

Notebook: `notebooks/35_hybrid_predictive_composition_closure.ipynb`

## Requirements

- Open the notebook from the committed repository branch. Source binding rejects
  copied or edited cells.
- Select a GPU runtime. A G4-class runtime is preferred; L4 is supported.
- No Stage 34 output is consumed. Stage 35 generates a completely fresh panel.
- `HF_TOKEN` is optional unless the public checkpoint host requires it in the
  assigned runtime.

## Run

1. Leave `RUN_MODE = "pilot"`.
2. Select **Runtime -> Run all**.
3. Authorize Google Drive.
4. Do not execute individual cells out of order. Exact notebook-prefix
   verification is part of the evidence.
5. Return `stage35_hpcc_result_bundle_<signature>.zip`.

The notebook writes hash-validated truth, prefix-carrier, selection,
calibration, and locked-evaluation checkpoints under:

```text
MyDrive/counterfactual_faithfulness_stage35_hpcc/
```

Rerunning **Run all** resumes the incomplete run whose protocol and nonce match
the Drive pointer. Transient Drive operations retry eight times with bounded
backoff. A content-hash mismatch stops rather than silently regenerating a
scientifically different shard.

## Expected resources

Pilot sizes are 16/16/16/32 complete trajectory families across construction,
model selection, calibration, and evaluation. Only JEPA is loaded. Truth is
generated from split-specific words and does not store path images.

Expected G4/L4 wall time is approximately 2--4 hours, dominated by official
JEPA forward passes and Drive I/O. CPU model selection and 5,000-draw clustered
intervals ordinarily take another 10--30 minutes. The notebook prints actual
forward counts, device, peak memory, elapsed time, and cache hits.

## Read the result

Require `FAILURE_TRACE.txt = NONE` and verify `result_zip_manifest.json`. Read:

1. `stage35_decision.json`;
2. `evaluation_evidence/hybrid_composition_summary.json`; and
3. `evaluation_evidence/locked_hybrid_composition_rows.csv`.

A full pass is observational evidence for bounded distributed recursive
closure. It is not authorization to claim a minimal state, causal mechanism,
cross-model agreement, or shared circuitry. The next stage after a pass is a
fresh native intervention; after any scientific failure, interpret the first
failed gate and stop that branch.

