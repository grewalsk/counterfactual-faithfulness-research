# Stage 39.2 contact-tail-qualified replication Colab guide

Notebook: `notebooks/39_2_contact_tail_qualified_replication.ipynb`

Colab: <https://colab.research.google.com/github/grewalsk/counterfactual-faithfulness-research/blob/codex/stage34-predictive-fiber-abstraction/notebooks/39_2_contact_tail_qualified_replication.ipynb>

Stage 39.2 corrects the evaluation gap exposed by Stage 39 without changing
the completed Stage 39 result. It uses new trajectory IDs, action words,
optimization seeds, fitted artifacts, and a separate Drive output root.

## Two locked decisions

The notebook reports two distinct outcomes for JEPA and DINO:

1. comparative mean equivalence of full S-PSCD and the exactly
   coefficient-matched latent-only objective, using the unchanged 90%
   hierarchical interval and `[-5%, +5%]` band; and
2. absolute reliability of both objectives across every seed, using the
   registered word-length, initial-mode, terminal-mode, contact/post-contact,
   p95, and catastrophic-rate thresholds.

The overall pass is conjunctive. A mean-equivalence pass with a tail failure
is explicitly reported as mean-equivalent but tail-unqualified. It is not
silently converted into a successful reliability result.

## Run

1. Open the committed notebook in Colab.
2. Select a GPU runtime. G4 (`RTX PRO 6000 Blackwell`) is preferred; A100 and
   L4 are supported.
3. Leave `RUN_MODE = "pilot"`.
4. Select **Runtime -> Run all** and authorize Google Drive.
5. Do not edit cells, run them out of order, or reuse a Stage 39 directory.
6. Return `stage39_2_ctqr_result_bundle_<signature>.zip`.

Resumable output:

```text
MyDrive/counterfactual_faithfulness_stage39_2_ctqr/
```

No earlier result directory is required or read. The completed Stage 39 G4
run took about 37 minutes; Stage 39.2 has the same main panel size, so reserve
roughly 35-60 minutes on G4 plus Drive variability.

## Read the result

Require `FAILURE_TRACE.txt = NONE`, then read:

1. `stage39_2_decision.json`;
2. `evaluation_evidence/stage39_2_panel_decisions.json`;
3. `evaluation_evidence/stage39_2_summary.json`;
4. `evaluation_evidence/locked_replication_rows.csv`; and
5. `evaluation_evidence/coefficient_match_receipts.json`.

The decision JSON contains `comparative_status` and
`absolute_tail_qualification_passed` separately. The notebook never pools the
JEPA and DINO panels and leaves planning permanently sealed.
