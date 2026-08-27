# Stage 40 contact-tail risk distillation Colab guide

Notebook: `notebooks/40_contact_tail_risk_distillation.ipynb`

Colab: <https://colab.research.google.com/github/grewalsk/counterfactual-faithfulness-research/blob/codex/stage34-predictive-fiber-abstraction/notebooks/40_contact_tail_risk_distillation.ipynb>

Stage 40 tests a prospective repair for the broad contact-tail weakness found
by Stage 39.2. It does not change the evaluation scale or thresholds and does
not read the Stage 39.2 result bundle.

## Method

Both compared adapters use the coefficient-matched latent-only recursive
objective and train their physical decoder against simulator truth on
non-evaluation data. The repair changes only the training measure:

- the uniform control assigns every valid transition equal weight;
- the repair upweights contact targets and gives additional weight to
  post-contact-to-contact re-entry;
- weights are normalized to mean one and are unnecessary at inference.

The latent outer weights are pinned before Stage 40. A contact multiplier from
`[1, 2, 4, 8]` is selected separately for JEPA and DINO on model-selection
trajectories, minimizing p95 error subject to no more than 5% mean degradation.

## Locked decision

Each predictor panel must independently pass all of the following:

1. repair-over-uniform mean noninferiority with a 90% hierarchical interval
   lower bound of at least `-5%`;
2. at least 10% p95 improvement for every final seed;
3. at least 10% terminal-contact improvement for every final seed; and
4. the unchanged absolute length, mode, p95, and catastrophic-rate gates.

The overall decision never pools JEPA and DINO. Planning remains sealed.

## Run

1. Open the committed notebook in Colab.
2. Select G4 (`RTX PRO 6000 Blackwell`) if available; A100 or L4 also works.
3. Leave `RUN_MODE = "pilot"` and `DOWNLOAD_RESULTS = True`.
4. Select **Runtime -> Run all** and authorize Google Drive.
5. Do not edit cells, run them out of order, or reuse an earlier stage folder.

Resumable output:

```text
MyDrive/counterfactual_faithfulness_stage40_ctrd/
```

The completed bundle automatically downloads only after the decision file is
written and the pipeline has no software failure. A negative scientific result
still downloads. Expect approximately 45-75 minutes on G4, with Drive latency
adding variability.

Return `stage40_ctrd_result_bundle_<signature>.zip`.

## Read the result

Require `FAILURE_TRACE.txt = NONE`, then inspect:

1. `stage40_decision.json`;
2. `evaluation_evidence/stage40_panel_decisions.json`;
3. `evaluation_evidence/stage40_summary.json`;
4. `evaluation_evidence/locked_stage40_rows.csv`;
5. `evaluation_evidence/stage40_contact_risk_selection_rows.csv`; and
6. `evaluation_evidence/coefficient_match_receipts.json`.
