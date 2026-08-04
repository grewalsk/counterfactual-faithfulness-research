# Stage 20 Colab run guide

Notebook: `notebooks/20_causal_planner_steering.ipynb`

## Required frozen inputs

Stage 20 will not reconstruct or refit either prior result. It requires:

1. The exact successful Stage 18 raw subspace:

   `/content/drive/MyDrive/counterfactual_faithfulness_stage18_rank64/pilot_f1b34beffcac/subspaces/frozen_rank64_confirmation_subspaces.npz`

   Required SHA-256:
   `2f9c496d54623a9062e465a18c70039acc18cb8a1cc2833a5f4ade162ca3f90b`.

2. The exact successful Stage 19 decision:

   `/content/drive/MyDrive/counterfactual_faithfulness_stage19_transfer/pilot_b7f2b6cef37f/stage19_decision.json`

   Required SHA-256:
   `493fdf5c707189caea11043db7d208dbc38677dcf5881008e13bede87f40be9c`.

Keep the adjacent provenance files in both run directories. If either input
is elsewhere, set `STAGE20_STAGE18_SUBSPACE_PATH` or
`STAGE20_STAGE19_DECISION_PATH` to its full Drive path.

## Colab secrets

Create:

- `STAGE20_RUN_MODE` = `pilot`
- `STAGE20_SOURCE_COMMIT` = the full 40-character commit in the handoff link
- `STAGE20_RUN_NONCE` = a new label, for example `steering_20260804_a`

Do not put the nonce in `STAGE20_RUN_MODE`. The mode secret must contain only
`pilot`. Do not edit notebook cells after opening the commit-pinned Colab
link. Stage 20 checks the executed cells against that exact commit, and a
reused nonce stops rather than reusing cached evidence.

## Run

1. Open the supplied **Open in Colab** link.
2. Select a GPU runtime; L4 or faster is recommended.
3. Keep Google Drive mounting enabled.
4. Choose **Runtime > Run all**.
5. Return
   `stage20_causal_planner_steering_result_bundle_<signature>.zip`.

The pilot generates simulator truth for 80 fresh states in each of two action
families, physically selects 32 per family, and makes baseline predictions
before freezing any steering targets. For each selected state it fixes the
baseline rank-2, rank-3, and rank-4 actions as targets and then performs 39
patched model forwards. Total pilot accounting is 64 baselines, 2,496 patched
forwards, and 3,456 result rows.

Approximate end-to-end times:

- RTX PRO 6000 Blackwell / G4: 8–20 minutes;
- L4: 15–40 minutes;
- T4: 30–75 minutes.

These are planning envelopes, not evidence that a short run skipped work. The
notebook writes exact forward counters and a measured intervention estimate.
The fresh-run certificate, not wall-clock time, determines whether all work
was generated.

## Required audit outputs

- `prior_artifact_certificate.json` must bind both prior inputs before Stage
  20 model activations.
- `physical_selection_freeze.json` must contain 32 selected records per
  family in pilot mode and state that selection was model-blind.
- `steering_target_freeze.json` must state that target selection used baseline
  scores but neither simulator endpoint costs nor intervention outputs.
- `hook_identity_test.json` must pass at maximum error at most `1e-6`.
- `fresh_run_certificate.json` must report 160 truth records, 64 baselines, 64
  intervention shards, 2,496 patched forwards, and zero cache hits.
- `stage20_decision.json` reports representation and planner-steering gates
  separately for both families.
- `FAILURE_TRACE.txt` must contain `NONE` for a completed run.

The executed action's simulator cost is a descriptive downstream readout. It
is deliberately not an optimization or improvement claim: Stage 20 asks
whether a frozen internal edit can causally move prediction, rank, and choice
toward a prespecified counterfactual action.
