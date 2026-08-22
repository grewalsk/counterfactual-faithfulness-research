# Stage 37.1 horizon-matched operator-calibration Colab guide

Notebook: `notebooks/37_1_horizon_matched_operator_calibration.ipynb`

Stage 37.1 resolves the measuring-instrument failure exposed by Stage 37. The
earlier true-state control trained on length-5--8 words did not extrapolate to
the locked length-9--12 panel. This notebook therefore uses length-9--12 words
in construction, model selection, calibration, and evaluation, with exact
words and trajectory families disjoint across all four splits.

This is a simulator-only experiment. It never loads JEPA-WM, DINO-WM, or any
pretrained checkpoint, and it cannot produce a JEPA, planning, representation,
or causal claim. A full pass only validates the operator class and authorizes a
later fresh JEPA confirmation.

## Run

1. Open the notebook from its committed repository branch.
2. Select a GPU runtime. G4 Blackwell or A100 is preferred; L4 and T4 are
   supported.
3. Leave `RUN_MODE = "pilot"`.
4. Select **Runtime -> Run all** and authorize Google Drive.
5. Do not execute cells out of order. The notebook verifies its committed cell
   prefix before each irreversible stage.
6. Return `stage37_1_hmoc_result_bundle_<signature>.zip`.

The resumable directory is:

```text
MyDrive/counterfactual_faithfulness_stage37_1_hmoc/
```

No prior result directory or Hugging Face token is needed. Physical truth and
model artifacts are content-hashed, transient GitHub and Drive operations use
bounded retries, and a mismatched resume stops rather than combining runs.

## Frozen experiment

- 16 construction, 16 model-selection, 16 calibration, and 32 locked
  evaluation trajectory families.
- Eight registered words per split, balanced across lengths 9, 10, 11, and 12.
- Four development candidates: latent widths 128/256 crossed with single and
  mixture transitions.
- One primary free-running operator with semigroup anchors 2/4/8.
- Same-architecture, same-initialization zero-semigroup and one-step-only
  controls.
- Clustered trajectory bootstrap and separate caps for every word length and
  physical mode.

## Expected resources

No checkpoint inference is performed, so the run is materially cheaper than
Stage 37. Budget approximately:

- G4 Blackwell: 1--2.5 hours.
- A100: 1.5--3 hours.
- L4: 2.5--5 hours.
- T4: 5--9 hours.

The first run also generates exact PushT trajectories; Drive latency and the
assigned CPU can dominate this part. The notebook records measured wall time,
device, cache hits, and trajectory counts.

## Read the result

Require `FAILURE_TRACE.txt = NONE` and a valid `result_zip_manifest.json`, then
read:

1. `stage37_1_decision.json`;
2. `development_preflight.json`;
3. `evaluation_evidence/horizon_control_summary.json`; and
4. `evaluation_evidence/locked_horizon_control_rows.csv`.

`horizon_matched_operator_class_calibrated` is the only full-pass status. It
means the true-state recursion passed absolute physical and semigroup closure,
beat persistence plus both matched objective controls, and remained consistent
across every registered horizon and physical mode. The next experiment would
then rerun S-PSCD on a completely fresh JEPA panel without changing this
operator design.
