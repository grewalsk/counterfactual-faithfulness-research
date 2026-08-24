# Stage 38.1 coefficient-matched and hybrid audit Colab guide

Notebook: `notebooks/38_1_coefficient_matched_hybrid_audit.ipynb`

Stage 38.1 is a development-only audit. It repairs the Stage 38 latent-pressure
confound before any new confirmation or planning experiment. It reuses only
the exact construction, model-selection, and calibration carrier shards from
source-bound Stage 38 run `ceb85af5b4b9`. It never reads Stage 38 evaluation
rows, evaluation carrier shards, decisions, or planning artifacts.

## Required Drive source

Keep this complete resumable directory from the successful Stage 38 run:

```text
MyDrive/counterfactual_faithfulness_stage38_xmpscd/pilot_ceb85af5b4b9/
```

The downloaded compact Stage 38 result bundle is not sufficient because it
does not contain `prefix_carrier_paths/`. The notebook validates the exact
Stage 38 protocol, run signature, source commit, development trajectory
manifests, every reused shard identity, and every reused shard hash before
training.

## Run

1. Open the notebook from the committed repository branch.
2. Select a GPU runtime. G4 (`RTX PRO 6000 Blackwell`) is preferred; L4 and
   A100 are supported.
3. Leave `RUN_MODE = "pilot"`.
4. Select **Runtime → Run all** and authorize Google Drive.
5. Do not edit cells, run cells out of order, or point the notebook at another
   Stage 38 directory.
6. Return `stage381_cmha_result_bundle_<signature>.zip`.

The resumable output directory is:

```text
MyDrive/counterfactual_faithfulness_stage38_1_cmha/
```

## Sequential experiment

Tier A trains four construction-only variants for JEPA and DINO:

- ordinary PSCD with no semigroup term;
- historical mass-matched latent overshooting at 2.0/1.0;
- coefficient-matched latent overshooting at 0.90/0.45;
- full S-PSCD at 2.0/1.0 with component weights `(0.35, 0.20, 0.45)`.

Seeds `38101` and `38102` screen first. Precommitted seed `38103` runs only if
both representation panels remain promotion eligible. Architecture,
initialization seed, full-batch order, optimizer, learning rate, 320 epochs,
history length, and horizons are matched. Raw component losses, effective
coefficients, gradient norms, anchor counts, transition counts, and parameter
counts are logged.

Tier B remains sealed unless Tier A passes every gate for both representations.
It first trains the small event model and tests an oracle-event rollout ceiling.
Only adequate oracle headroom opens the label-free rollout, parameter-matched
smooth residual, and shuffled-label controls. Family-level CVaR90 training is
opened only if the no-tail structural hybrid passes.

## Expected time

- Tier A screening on G4 Blackwell: approximately 1–2 GPU-hours.
- Conditional third seed: included in the upper end of that estimate.
- Tier B oracle and controls, if opened: another 2–4 GPU-hours.
- Full conservative reservation: 4–6 G4 hours or roughly 6–10 L4 hours.

Stage 38 already suggests the absolute p95 gate may stop Tier A after two
seeds. That is an informative scientific stop, not an error.

## Read the result

Require `FAILURE_TRACE.txt = NONE`, then read:

1. `stage381_decision.json`;
2. `tier_a_decision.json`;
3. `evaluation_evidence/tier_a_final_decisions.json`;
4. `evaluation_evidence/tier_b_oracle_headroom.json`, if Tier B opened;
5. `evaluation_evidence/tier_b_decisions.json`, if controls opened; and
6. `evaluation_evidence/risk_extension_decisions.json`, if risk training opened.

The result statuses are deliberately kill-oriented. A Tier A rejection kills
the claim that the extra carrier/physical semigroup components add value at
equal latent pressure. Oracle failure kills the event/reset repair hypothesis
at this sampling rate. Oracle success with label-free failure indicates that
the frozen carrier/history does not identify the needed contact state. Only a
complete Tier B pass authorizes a separately preregistered Stage 39 on fresh
data; it is not itself confirmation or an ICLR-ready result.
