# Stage 21 Colab run guide

Notebook: `notebooks/21_coherent_interface_and_heldout_utility.ipynb`

## Required frozen inputs

Stage 21 requires the complete successful Drive directories from the previous
three stages. The default files are:

- Stage 18 subspace:
  `/content/drive/MyDrive/counterfactual_faithfulness_stage18_rank64/pilot_f1b34beffcac/subspaces/frozen_rank64_confirmation_subspaces.npz`
- Stage 19 decision:
  `/content/drive/MyDrive/counterfactual_faithfulness_stage19_transfer/pilot_b7f2b6cef37f/stage19_decision.json`
- Stage 20 decision:
  `/content/drive/MyDrive/counterfactual_faithfulness_stage20_steering/pilot_234bd6f0f2ae/stage20_decision.json`

Their required SHA-256 values are, respectively:

- `2f9c496d54623a9062e465a18c70039acc18cb8a1cc2833a5f4ade162ca3f90b`
- `493fdf5c707189caea11043db7d208dbc38677dcf5881008e13bede87f40be9c`
- `57e6ec6ab60415d782bd37e773842b96a8fef596ead111df33b7b816c83d601e`

Keep each adjacent `source_identity.json` and all Stage 18 subspace provenance.
If a file moved, use `STAGE21_STAGE18_SUBSPACE_PATH`,
`STAGE21_STAGE19_DECISION_PATH`, or `STAGE21_STAGE20_DECISION_PATH`.

## Colab secrets

Create:

- `STAGE21_RUN_MODE` = `pilot`
- `STAGE21_SOURCE_COMMIT` = the full 40-character commit in the handoff link
- `STAGE21_RUN_NONCE` = a new label such as `coherent_utility_20260804_a`

The mode secret must contain only `pilot`. Open the commit-pinned notebook and
do not edit its cells. Stage 21 verifies the executed prefix before model
loading and verifies the complete experimental execution again in the final
decision cell.

## Run

1. Open the supplied **Open in Colab** link.
2. Select a GPU runtime; L4 or faster is recommended.
3. Keep Google Drive mounting enabled.
4. Choose **Runtime > Run all**.
5. Return `stage21_coherent_utility_result_bundle_<signature>.zip`.

The pilot generates 256 fresh simulator records. Model-blind physical
eligibility selects, per action family, 32 construction states, 16 calibration
states, and 32 evaluation states. It extracts 160 model baselines and performs
576 patched forwards for the coherent-interface assay. The compact bundle
contains 960 interface rows and 448 held-out utility rows.

Approximate end-to-end times:

- RTX PRO 6000 Blackwell / G4: 6–18 minutes;
- L4: 12–35 minutes;
- T4: 25–65 minutes.

Expect roughly 1–2 GB of new raw Drive data, mostly truth and carrier shards.
Exact generated-work counts, rather than elapsed time, determine freshness.

## Audit order

1. `prior_artifact_certificate.json` must bind Stages 18–20 before Stage 21
   model activations.
2. `physical_split_selection_freeze.json` must show disjoint construction,
   calibration, and evaluation trajectories with the expected counts.
3. `correction_fit_freeze.json` must show zero evaluation records in fitting,
   a goal-independent pose-error target, and identical ridge grids for learned,
   shuffled, and random bases.
4. `evaluation_choice_freeze.json` must exist before
   `evaluation_truth_open_certificate.json`; all score hashes must reproduce.
5. `fresh_run_certificate.json` must report 256 truth records, 160 baselines,
   64 interface shards, 576 patched forwards, and zero cache hits.
6. `stage21_decision.json` reports coherent-handoff and held-out-utility gates
   separately for both action families.
7. `FAILURE_TRACE.txt` must contain `NONE`.

The final utility result concerns a supervised, goal-independent correction of
an external frozen physical decoder. It is not a native JEPA policy or
multi-step closed-loop control result.
