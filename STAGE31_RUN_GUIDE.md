# Stage 31 Colab run guide

Open `notebooks/31_cross_model_grounded_closure_certificate.ipynb` from the
committed GitHub branch and select an L4-class GPU runtime. Use **Runtime → Run
all** once and do not edit or selectively rerun protocol cells; the notebook
source-binds its executed prefix.

No Stage 31 secret is required. If `HF_TOKEN` already exists in Colab secrets,
the notebook uses it for more reliable access to the public checkpoint cache.
The run mode, source branch, and fresh nonce are automatic.

The notebook creates a durable directory under:

`MyDrive/counterfactual_faithfulness_stage31_cross_model/pilot_<signature>`

Keep that complete directory. The downloaded compact result bundle includes
the decision, evidence tables, manifests, hashes, timings, logs, and plots. It
excludes large raw simulator images and model-specific basis arrays, which are
instead bound by `raw_shard_manifest.json`.

Expected L4-class runtime is roughly 90–180 minutes after the first checkpoint
download, with substantial variation from Colab I/O and simulator speed. The
construction gates fail closed: if either model lacks construction-only
action/output alignment, the notebook reports that result and does not spend
credits on the confirmatory evaluation.
