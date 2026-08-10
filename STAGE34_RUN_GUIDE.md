# Stage 34 predictive-fiber causal abstraction Colab run guide

Notebook:
`notebooks/34_predictive_fiber_causal_abstraction.ipynb`

Stage 34 follows the clean negative Stage 33 result. It does not retry a direct
JEPA-to-DINO map. Instead, the simulator defines one canonical finite
action-response chart, and JEPA and DINO are tested separately against it.
The expensive stages are sequential: action specificity, predictive
sufficiency, on-manifold causal use, and two-sided model-to-simulator
commutativity. Planning is intentionally deferred unless all four gates pass.
The complete mathematical and decision contract is in
`docs/STAGE34_PREDICTIVE_FIBER_CAUSAL_ABSTRACTION_PROTOCOL.md`.

## What to run

1. Open the committed notebook from the Stage 34 branch in a fresh Google
   Colab runtime.
2. Prefer a full Google G4 runtime (NVIDIA RTX PRO 6000 Blackwell Server
   Edition). An L4 is supported but slower.
3. Leave `RUN_MODE = "pilot"` for evidence. Use `"smoke"` only to check
   plumbing; smoke can never produce a positive scientific decision.
4. Select **Runtime -> Run all**. Do not edit a protocol cell or start from a
   later cell: the run binds the executed prefix to the committed notebook,
   builder, and numerical source.
5. Authorize Google Drive when prompted. If the runtime disconnects, start a
   fresh runtime and use **Run all** again; hash-validated shards are resumed.
6. Return `stage34_pfca_result_bundle_<signature>.zip` and keep the complete
   Drive run directory until the bundle has been audited.

No new secret is required. `HF_TOKEN` is optional for more reliable access to
the public Hugging Face snapshot and must never be pasted into a cell.

## Frozen pilot design

The pilot requires:

- 16 construction, 16 model-selection, 16 calibration, and 32 locked
  evaluation trajectories;
- four records per trajectory: free, pre-contact, contact, and post-contact;
- disjoint candidate pools `[10000,11200)`, `[11200,12400)`,
  `[12400,13600)`, and `[13600,16000)`;
- construction response words of lengths 1--4 and genuinely held-out
  evaluation compositions of lengths 5--8;
- no-op-corrected response paths plus fixed-multiset order contrasts;
- one simulator-only canonical chart and exactly zero JEPA-to-DINO maps;
- independent JEPA-to-physical and DINO-to-physical diagrams; and
- native block-4 recurrent interventions, a same-model full-swap positive
  control, a matched random-subspace control, and a natural-support OOD gate.

The two official checkpoints, source commit, Hugging Face revision, file
hashes, DINOv2 shared-target confound, and model output shapes remain the exact
source-bound Stage 33 contracts.

## Runtime estimate

Stage 34 is larger than Stage 33 and materializes many successor-state response
tests. These are planning estimates, not guarantees:

| Mode | Full G4 Blackwell | L4 | Peak GPU memory | Drive storage |
|---|---:|---:|---:|---:|
| smoke | 20--45 minutes | 35--75 minutes | below 8 GiB | below 3 GiB |
| pilot | 4--8 hours | 8--14 hours | below 8 GiB | 15--30 GiB |

Simulator screening, Drive latency, public-model download speed, and Colab
preemption can dominate. The pilot is designed to stop after the first failed
scientific gate, so a negative result can finish substantially earlier. The
resumable Drive directory is more important than an uninterrupted session.
`MAX_ESTIMATED_TOTAL_MINUTES = 900` is recorded as a planning envelope; it is
not a timeout and never shrinks the registered design.

## Sequential decisions

Read `stage34_decision.json`, not the plots alone.

- `BOUNDED_TWO_SIDED_CAUSAL_ABSTRACTION_SUPPORTED` requires every gate to pass
  in an eligible pilot. It supports only a finite-bank, one-environment,
  two-checkpoint high-level abstraction.
- `SHARED_STATIC_STATE_GEOMETRY_ONLY` means unseen action words did not beat
  action-shuffled and static-state controls.
- `CANDIDATE_PREDICTIVE_STATE_INSUFFICIENT` means residual carrier information
  still materially improves transition prediction or deletion of a true
  response coordinate does not hurt.
- `PREDICTIVE_SUMMARY_NOT_CAUSALLY_USED` means the matched, on-manifold internal
  interventions fail even though observational sufficiency passed.
- `MODELS_DO_NOT_SHARE_HIGH_LEVEL_TRANSITION` means at least one separate
  model-to-physical diagram fails the locked commutativity/control tests.
- `SMOKE_ONLY` and `INCONCLUSIVE_SOURCE_OR_SPLIT_FAILURE` are not scientific
  negatives.

A pass does not establish common circuitry, an infinite-horizon minimal state,
checkpoint-family generality, or planning value.

## Drive and returned artifact

The durable root is:

`MyDrive/counterfactual_faithfulness_stage34_pfca/`

The compact ZIP excludes large raw truth, baseline, and carrier shards but
records their paths and hashes in the raw manifest. Keep the complete run
directory. The ZIP contains the protocol/source certificate, selection and
coverage audits, simulator chart and rank lock, model artifact manifests,
locked row-level evidence, gate summaries, plots, timing/memory reports,
automatic interpretation, failure trace, and content manifest.

Methodologically, the bounded response-state construction is motivated by
[predictive-state representations](https://proceedings.neurips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html),
the commuting tests by [approximate causal
abstraction](https://proceedings.mlr.press/v115/beckers20a.html), and the native
patching tests by [interchange intervention
analysis](https://arxiv.org/abs/2106.02997).
