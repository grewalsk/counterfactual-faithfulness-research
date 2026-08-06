# Stage 28 run guide: hybrid control-area law

Stage 28 tests whether the Stage 27 order effect obeys a signed control-area
law across multiple schedules and magnitudes, and whether the exact frozen
Stage 18 block-4 action carrier causally mediates that law.

## Required upstream artifacts

Keep the complete successful Stage 18 Drive directory. Stage 28 auto-locates
the 606 MB frozen subspace by SHA-256.

Stage 28 also requires one clean run of the repaired Stage 27 notebook at
commit `78d241fce761babdc4c11c51bfc5758867ecea07`. The earlier Stage 27 bundle
whose final aggregation crashed is scientifically recoverable but is not an
accepted upstream certificate.

No Stage 28-specific Colab secrets are required. An existing `HF_TOKEN` secret
is used automatically if present; the pinned public assets normally work
without it.

## Run

1. Select a GPU runtime.
2. Use **Runtime → Run all** without editing cells.
3. Allow Google Drive mounting.
4. Keep the complete Drive directory
   `counterfactual_faithfulness_stage28_control_area/pilot_<signature>/`.
5. Download and return
   `stage28_hybrid_control_area_result_bundle_<signature>.zip`.

The notebook first screens three frozen magnitude panels on 30 development
states using simulator contact coverage only. It then generates 120 disjoint
confirmation states, selects 12 persistent-contact, 12 boundary-switching, and
12 free states without using area-effect magnitude or model outputs, and runs
30 causal interventions per selected state.

The measured forward benchmark is saved before interventions. On an L4/G4-class
GPU, budget roughly 45–120 minutes; a faster Blackwell GPU may finish much
sooner. The notebook fails closed if the exact source, repaired Stage 27 run,
frozen Stage 18 artifact, fresh-run certificate, or execution history does not
validate.

