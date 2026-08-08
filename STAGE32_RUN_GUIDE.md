# Stage 32 Colab run guide

Open `notebooks/32_powered_bounded_cross_model_confirmation.ipynb` from its
committed GitHub branch, select an L4-class or better GPU, and use **Runtime →
Run all** once. Do not edit or selectively rerun protocol cells; the notebook
verifies its executed source prefix against GitHub.

No Stage 32 secret is required. An existing optional `HF_TOKEN` is used only
for reliable access to the public model cache.

Keep the complete successful Stage 31 Drive directory under
`MyDrive/counterfactual_faithfulness_stage31_cross_model`. Stage 32 binds the
exact Stage 31 decision, source identity, manifests, and the two raw frozen
subspace arrays. The compact Stage 31 ZIP alone is insufficient because it
contains hashes rather than the large arrays.

Stage 32 writes its durable output under:

`MyDrive/counterfactual_faithfulness_stage32_bounded_cross_model/pilot_<signature>`

The physical screen evaluates 800 candidate states across 72 branches and then
materializes 160 all-family persistent-contact states. Expected runtime is
approximately 45–120 minutes on an L4, depending mostly on Drive and simulator
speed; a Blackwell-class GPU will be faster. The compact result ZIP excludes
raw endpoint images while binding them in `raw_shard_manifest.json`.
