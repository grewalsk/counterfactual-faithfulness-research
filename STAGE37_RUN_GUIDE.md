# Stage 37 semigroup-regularized PSCD and planning-value Colab guide

Notebook: `notebooks/37_semigroup_pscd_planning_value.ipynb`

Version 2 repairs the v1 simulator-preflight helper-order failure. V1 completed
all simulator truth and physical trajectory selection, then stopped before the
first control fit because `stable_seed` was defined one cell after its first
use. JEPA was never loaded and no scientific statistic was observed. V2 moves
that unchanged helper before the preflight and adds ordered-namespace testing;
the scientific protocol is unchanged.

Stage 37 is a fresh follow-up to the complete Stage 36 result. It freezes the
selected Stage 36 JEPA carrier width, history length, latent width, and mixture
dynamics, then screens only the strength of a new multi-anchor semigroup loss.
The primary model is compared with a capacity- and initialization-matched model
whose semigroup weight is exactly zero.

Before JEPA-WM is loaded, a neural transition with direct access to the true
11-dimensional PushT Markov state must pass a disjoint simulator preflight. If
it fails, the notebook skips all expensive JEPA inference and returns a valid
operator-class failure result. This prevents another ambiguous positive-control
failure from consuming the locked JEPA panel.

## Requirements

- Open the notebook from its committed repository branch. Source binding
  rejects copied or edited protocol cells.
- Select a GPU runtime. G4 (`RTX PRO 6000 Blackwell`) is preferred; A100 and L4
  are supported. T4 is possible but slow.
- No Stage 36 directory is consumed. All trajectory IDs, action words, truth,
  carrier paths, models, and evaluation rows are newly generated.
- `HF_TOKEN` is optional unless the public checkpoint host requests it.

## Run

1. Leave `RUN_MODE = "pilot"`.
2. Select **Runtime → Run all**.
3. Authorize Google Drive.
4. Do not execute cells out of order; committed-prefix verification is part of
   the evidence.
5. Return `stage37_spscd_result_bundle_<signature>.zip`.

The resumable run is written under:

```text
MyDrive/counterfactual_faithfulness_stage37_spscd/
```

Truth and carrier shards are content-hashed. Compatible reruns resume the same
incomplete run. Transient GitHub and Drive errors use bounded retries; a hash
mismatch stops instead of silently mixing runs.

## What the notebook tests

- Neural true-state simulator closure before any JEPA inference.
- Direct-versus-composed agreement from every eligible history anchor at
  horizons 2, 4, and 8.
- A semigroup-weight screen on disjoint length-5–8 words.
- Locked closure on fresh length-9–12 words.
- Equal-budget open-loop ranking of twelve fixed length-10 candidate words.
- Matched legacy-objective, one-step-only, and false-history controls.

The open-loop planning rollout starts from the current carrier repeated across
the four history slots. It never uses a candidate's future JEPA carriers for
warmup. Each goal is selected deterministically only after all models are
frozen, and simulator endpoints are used solely to score regret.

## Expected resources

The pilot uses 16/16/16/24 trajectory families and one frozen JEPA-WM
checkpoint. It screens four simulator controls and three semigroup weights,
then fits the primary and matched controls once.

- G4 Blackwell: approximately 2–4 hours; reserve 5 GPU-hours.
- A100: approximately 2.5–4.5 hours.
- L4: approximately 4–7 hours.
- T4: approximately 8–14 hours.

Actual time depends on Colab allocation and Drive latency. The notebook records
wall time, device, memory, cache hits, and forward counts.

## Read the result

Require `FAILURE_TRACE.txt = NONE` and verify `result_zip_manifest.json`. Then
read:

1. `stage37_decision.json`;
2. `evaluation_evidence/stage37_summary.json`;
3. `evaluation_evidence/locked_closure_rows.csv`; and
4. `evaluation_evidence/locked_planning_rows.csv`.

A full pass supports post-hoc semigroup-regularized repair and finite-bank
open-loop planning value for this PushT checkpoint. It is not closed-loop CEM
evidence, a native JEPA causal mechanism, or cross-environment generality.
