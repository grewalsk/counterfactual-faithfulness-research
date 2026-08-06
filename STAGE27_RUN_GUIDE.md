# Stage 27 Colab run guide

Notebook: `notebooks/27_causal_action_commutator.ipynb`

Stage 27 asks whether contact-sensitive dynamics appear as a finite action-order
commutator and whether the already frozen Stage 18 block-4 action carrier
causally mediates it. Each comparison contains exactly the same two control
pulses, active duration, integrated impulse, and energy; only pulse order is
reversed. This is a finite forward-pass experiment, not a Jacobian or claimed
infinitesimal Lie-bracket measurement.

## Secrets

No `STAGE27_*` Colab secrets are required. The notebook:

- starts in `pilot` mode;
- generates a fresh nonce each time the configuration cell runs;
- resolves `codex/stage27-causal-action-commutator` to an exact GitHub commit;
- automatically locates and hash-validates the successful Stage 18 carrier and
  Stage 19 decision in MyDrive.

Your existing account-level `HF_TOKEN` secret is used automatically if enabled
for the notebook, but it is optional for the pinned public model assets and is
never persisted in the result bundle.

Keep these complete upstream Drive directories:

- `MyDrive/counterfactual_faithfulness_stage18_rank64/pilot_f1b34beffcac`
- `MyDrive/counterfactual_faithfulness_stage19_transfer/pilot_b7f2b6cef37f`

The compact downloaded ZIPs are insufficient because Stage 27 deliberately
binds raw upstream provenance and the Stage 18 subspace file.

Select an L4, G4-class, or faster GPU and use **Runtime → Run all**. Do not edit
or selectively rerun source-bound cells. The notebook first runs 80 × 12 exact
simulator branches, then evaluates 40 selected states with 12 model candidates
and 30 intervention batches per state. Plan for roughly 15–35 minutes after
cached downloads; simulator work is substantially CPU-bound, and the notebook
prints a measured intervention estimate before the expensive causal phase.

The final cell downloads
`stage27_action_commutator_result_bundle_<signature>.zip`. Keep the complete
new Drive directory as well as the compact ZIP.
