# Stage 38 cross-model PSCD confirmation Colab guide

Notebook: `notebooks/38_cross_model_pscd_confirmation.ipynb`

Stage 38 is the fresh confirmation experiment after the Stage 37.1 measuring
instrument passed. It tests whether a post-hoc predictive state can make two
different frozen PushT representations recursively usable: the official
JEPA-WM checkpoint is primary and the official DINO-WM checkpoint is the
replication.

The checkpoints remain frozen. Simulator state is never a training target for
the learned representation repair; it is used only by the positive control,
locked scoring, and planning regret. Construction, selection, calibration, and
evaluation use disjoint trajectory pools and disjoint words of lengths 9--12.

## Run

1. Open the notebook from the committed repository branch.
2. Select a GPU runtime. G4 (`RTX PRO 6000 Blackwell`) is preferred. A100 and
   L4 are supported; T4 is possible but slow.
3. Leave `RUN_MODE = "pilot"`.
4. Select **Runtime -> Run all** and authorize Google Drive.
5. Do not edit or run cells out of order. The notebook verifies its committed
   source prefix before every irreversible phase.
6. Return `stage38_xmpscd_result_bundle_<signature>.zip`.

The resumable directory is:

```text
MyDrive/counterfactual_faithfulness_stage38_xmpscd/
```

Transient Drive and GitHub failures use bounded retries. Truth, path shards,
decoders, final model artifacts, scales, and the evaluation-open certificate
are hash validated. A binding mismatch stops rather than combining runs.

## Frozen experiment

- 24 construction, 16 selection, 24 calibration, and 32 locked evaluation
  trajectory families.
- Eight closure words per split, balanced over lengths 9, 10, 11, and 12.
- Frozen JEPA-WM and DINO-WM carriers projected to the same 256-dimensional
  capacity; history length four, latent width 256, and mixture dynamics.
- Three independent final optimization seeds.
- Four matched variants per seed: one-step only, free-running PSCD,
  latent-only overshooting, and full carrier/physical/latent S-PSCD.
- Absolute physical closure, native-repair, semigroup-specificity,
  recursion, correct-history, seed, length, mode, post-contact, and tail-risk
  gates.
- The twelve-candidate planning bank remains sealed unless both representation
  panels and the simulator/source gates pass.

## Expected resources

Stage 38 loads two checkpoints sequentially and fits 24 final neural controls
plus selection and simulator controls. Approximate first-run budgets are:

- G4 Blackwell: 6--12 hours; reserve 12 GPU-hours.
- A100: 7--13 hours.
- L4: 12--22 hours.
- T4: 24--40 hours.

Drive latency and CPU trajectory generation add variability. Compatible reruns
resume completed shards and frozen models, so a disconnected Colab session does
not need to restart the whole experiment.

## Read the result

Require `FAILURE_TRACE.txt = NONE` and a valid `result_zip_manifest.json`, then
read:

1. `stage38_decision.json`;
2. `evaluation_evidence/stage38_model_decisions.json`;
3. `evaluation_evidence/stage38_summary.json`;
4. `evaluation_evidence/locked_closure_rows.csv`; and
5. `evaluation_evidence/locked_planning_rows.csv` if planning opened.

`cross_model_pscd_closure_confirmed_without_planning_value` is a valid closure
pass: both frozen representations passed but planning did not add measurable
value. `cross_model_pscd_closure_and_planning_confirmed` is the stronger pass.
Neither status establishes native checkpoint closure, causality, closed-loop
control, minimal state, or cross-environment generality.
