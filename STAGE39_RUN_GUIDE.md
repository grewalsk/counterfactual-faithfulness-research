# Stage 39 fresh coefficient-matched replication Colab guide

Notebook: `notebooks/39_fresh_coefficient_matched_replication.ipynb`

Colab: <https://colab.research.google.com/github/grewalsk/counterfactual-faithfulness-research/blob/codex/stage34-predictive-fiber-abstraction/notebooks/39_fresh_coefficient_matched_replication.ipynb>

Stage 39 is a locked, fresh-data replication of the near-zero Stage 38.1
full-minus-coefficient-matched result. It does not read any Stage 38/38.1
trajectory, carrier shard, fitted model, calibration outcome, or evaluation
row. Planning is permanently sealed.

## Primary decision

For each frozen PushT predictor panel separately, the primary estimand is the
mean paired row-wise relative gain of full S-PSCD over the coefficient-matched
latent-only control. Three matched optimization seeds and complete trajectory
families are resampled hierarchically. A 90% interval entirely inside the
prespecified `[-5%, +5%]` band establishes practical equivalence. A null
p-value or interval crossing zero is not sufficient.

The latent-only outer semigroup weight is exactly `0.45 * lambda_full`, so its
effective latent coefficient equals the full objective. Architecture,
initialization within seed, data, update count, learning rate, and epoch budget
are matched.

## Run

1. Open the committed notebook in Colab.
2. Select a GPU runtime. G4 (`RTX PRO 6000 Blackwell`) is preferred; A100 and
   L4 are supported.
3. Leave `RUN_MODE = "pilot"`.
4. Select **Runtime → Run all** and authorize Google Drive.
5. Do not edit cells, run them out of order, or point the notebook at an older
   result directory.
6. Return `stage39_fcmr_result_bundle_<signature>.zip`.

Resumable output:

```text
MyDrive/counterfactual_faithfulness_stage39_fcmr/
```

No earlier Drive directory is required.

## Expected resources

Approximate first-run reservations:

- G4 Blackwell: 4–7 GPU-hours;
- A100: 5–8 GPU-hours;
- L4: 8–14 GPU-hours;
- T4: 16–26 GPU-hours.

Simulator construction and Drive latency add wall-clock variability. Completed
truth shards, carrier paths, and fitted models are hash-validated and resumed.

## Read the result

Require `FAILURE_TRACE.txt = NONE`, then read:

1. `stage39_decision.json`;
2. `evaluation_evidence/stage39_panel_decisions.json`;
3. `evaluation_evidence/stage39_summary.json`;
4. `evaluation_evidence/locked_replication_rows.csv`; and
5. `evaluation_evidence/coefficient_match_receipts.json`.

`coefficient_matched_equivalence_replicated` means both predictor panels have
90% intervals inside ±5%. `full_objective_specificity_confirmed` requires both
lower bounds to clear +5%. Mixed or wider panels are reported as
`heterogeneous_or_inconclusive`; thresholds are never changed after opening
evaluation.
