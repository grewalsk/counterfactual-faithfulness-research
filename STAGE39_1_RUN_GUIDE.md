# Stage 39.1 Wall cross-environment replication Colab guide

Notebook: `notebooks/39_1_wall_cross_environment_replication.ipynb`

Colab: <https://colab.research.google.com/github/grewalsk/counterfactual-faithfulness-research/blob/codex/stage34-predictive-fiber-abstraction/notebooks/39_1_wall_cross_environment_replication.ipynb>

Stage 39.1 repeats the coefficient-matched test on the official frozen
JEPA-WM and DINO-WM **Wall** checkpoints. Its layouts, states, trajectory IDs,
action words, adapters, and evaluation rows are independent of PushT. Wall and
PushT are decided separately and are never pooled.

This is protocol v2. The first v1 bundle stopped at the fail-closed simulator
preflight before either world-model checkpoint was loaded and before any
evaluation row was used. V1 omitted varying wall and doorway coordinates from
the simulator state. V2 fixes that structural error by carrying normalized
`(dot_x, dot_y, wall_x, door_y)` through the exact-state anchor while still
scoring only the two-dimensional dot path. Thresholds and primary models are
unchanged.

The four frozen Wall strata are `free_far`, `pre_wall`, `wall_blocked`, and
`doorway`. Layout and state selection are geometric and precede model access.
The simulator's clipped motion supplies a collision diagnostic; normalized
two-dimensional point-path error is the locked physical score.

## Run

1. Open the committed notebook in Colab.
2. Select G4 Blackwell, A100, or L4.
3. Leave `RUN_MODE = "pilot"`.
4. Select **Runtime → Run all** and authorize Google Drive.
5. Do not edit cells or reuse the Stage 39 PushT directory.
6. Return `stage391_wall_result_bundle_<signature>.zip`.

Resumable output:

```text
MyDrive/counterfactual_faithfulness_stage39_1_wall_v2/
```

No prior result directory is required.

## Expected resources

Approximate first-run reservations:

- G4 Blackwell: 4–7 GPU-hours;
- A100: 5–8 GPU-hours;
- L4: 8–14 GPU-hours;
- T4: 16–26 GPU-hours.

These are planning envelopes, not measured telemetry. The run resumes verified
shards and models after a Colab disconnect.

## Read the result

Require `FAILURE_TRACE.txt = NONE`, then read the Wall equivalents of:

1. `stage391_decision.json`;
2. `evaluation_evidence/stage391_panel_decisions.json`;
3. `evaluation_evidence/stage391_summary.json`;
4. `evaluation_evidence/locked_replication_rows.csv`; and
5. `evaluation_evidence/coefficient_match_receipts.json`.

A Wall equivalence pass expands the bounded result to a second environment. It
does not establish universal world-model closure, an independent visual
encoder replication, planning value, or real-robot validity.
