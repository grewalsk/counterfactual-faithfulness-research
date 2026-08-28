# Stage 41 causal event/reset headroom

Open `notebooks/41_causal_event_reset_headroom.ipynb` in Google Colab and use
**Runtime → Run all**. The registered pilot requests an L4 GPU. A T4 should
also run it more slowly; A100/H100 primarily shorten checkpoint inference and
recursive-adapter fitting because exact Pymunk paired rollouts remain CPU
bound.

The notebook mounts Drive, resumes only hash-bound incomplete runs, and writes
under:

`MyDrive/counterfactual_faithfulness_stage41_cerh_v3/`

On successful pipeline completion, including an honest negative scientific
decision, it automatically downloads:

`stage41_cerh_v3_result_bundle_<run-signature>.zip`

The compact bundle excludes large truth, carrier-path, and paired-intervention
shards; those remain in the resumable Drive directory with manifests and hash
bindings. V3 downloads failed bundles too, clearly labeled by
`INCONCLUSIVE_PIPELINE_FAILURE`, so `FAILURE_TRACE.txt`, `failure_report.json`,
and the last stage receipt are never hidden behind the generic footer.

Stage 41 is development-only. Its oracle event/reset inputs are unavailable
to a deployed predictor. A positive decision authorizes only the next
label-free event-state identifiability experiment and does not authorize a
causal or planning claim.

V3 corrects the V2 false pipeline failure: paired truth is retained as
float64, while model-path archives intentionally store it as float32. Their
binding is now required to be bit-exact after the declared float32 cast rather
than compared at an impossible float64 `1e-9` tolerance.
