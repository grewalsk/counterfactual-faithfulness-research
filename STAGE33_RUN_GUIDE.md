# Stage 33 BIPCA Colab run guide

Notebook:
`notebooks/33_bounded_interventional_predictive_causal_abstraction.ipynb`

Stage 33 tests a bounded interventional predictive causal abstraction (BIPCA)
between the two public PushT world models. It does not test or claim a globally
minimal physical realization. The mathematical and decision contract is in
`docs/STAGE33_BOUNDED_INTERVENTIONAL_PREDICTIVE_CAUSAL_ABSTRACTION_PROTOCOL.md`.

## What to run

1. Open the notebook from its committed GitHub branch in a fresh Google Colab
   runtime.
2. Select a G4- or L4-class GPU. Another CUDA GPU with at least 16 GiB device
   memory is acceptable, but the estimates below are for G4/L4.
3. Leave `RUN_MODE = "pilot"` for the scientific run. Use `"smoke"` only to
   validate plumbing.
4. Choose **Runtime -> Run all**. Do not edit protocol cells or selectively run
   later cells. The notebook binds the executed prefix to the committed source.
5. Authorize Google Drive when prompted and keep the complete run directory.
6. When the notebook finishes, return the downloaded Stage 33 result ZIP. Keep
   the Drive directory until the bundle audit is complete.

No runtime restart is expected. If Colab asks for a restart after dependency
installation, restart once and use **Run all** again; do not jump directly to a
later cell.

## Pilot is not smoke

The pilot configuration must print and assert the complete frozen design:

- 8 construction, 8 model-selection, 8 calibration, and 16 locked-evaluation
  complete trajectories;
- exactly four mode records per trajectory: 32/32/32/64 records, 160 total;
- disjoint candidate pools `[6000,6200)`, `[6200,6400)`, `[6400,6600)`, and
  `[6600,7000)` in split order;
- the 11 core words of lengths 1--3 for construction/model
  selection/calibration and the 12 composition-held-out evaluation words of
  lengths 1--4, plus hashed actual prefixes and zero words of lengths 1--4;
- a maximum horizon of four model transitions (20 simulator controls); and
- free, pre-contact, sustained-contact, and post-contact coverage in every
  complete trajectory.

Smoke uses 1/1/1/2 complete trajectories, or 4/4/4/8 records. It also reduces
rank/permutation/final-bootstrap draws to 32/16/64 for plumbing only.

Pilot execution stops if any scientific count silently inherits a smoke value.
Smoke deliberately uses a much smaller design, writes `SMOKE_ONLY`, and can
never pass a scientific gate regardless of its numerical metrics.

## Assets and secrets

There are exactly two public world-model checkpoints:

- `jepa_wm_pusht.pth.tar`, SHA-256
  `9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb`;
- `dino_wm_pusht.pth.tar`, SHA-256
  `8ec9cb05f22812d7f12e3c216b0637f41641055c0653e503e2746edb981b550f`.

They are loaded from Hugging Face revision
`9b9c41ef249466630dbf1a20e78391865d07b3b9` and source commit
`13cf1d9c7e476f53c17714d2e0f1dc239a883ce0`.

Both models use the same public DINOv2 target/encoder file,
`dinov2_vits14_pretrain.pth`, SHA-256
`b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.
That file is a shared-target confound, not a third checkpoint. The notebook
records it explicitly and will not describe the two model outputs as
independent target representations.

No new Colab secret is required for run mode, source commit, nonce, Drive, or
checkpoint identity. `HF_TOKEN` is optional and is read only to make retrieval
of the public Hugging Face snapshot more reliable. A missing token must not
block a normal public download. Never paste a token into a notebook cell.

The default nonce is generated freshly from UTC time plus cryptographic random
bytes. Leave `MANUAL_RUN_NONCE` blank in the committed notebook. The remote
execution cell is bound to the exact committed source, so editing that field in
an opened committed notebook causes a source mismatch. Use the external Drive
request file described below when an exact nonce must be supplied. A nonblank
manual field is supported only in a newly rebuilt and committed derivative
whose hashes bind that edit; otherwise the run is exploratory or fails closed.
A nonce is provenance, never a secret.

## Compute estimate

These are planning estimates, not guarantees:

| Mode | G4/L4 wall time | Peak GPU memory | CPU RAM | Drive storage |
|---|---:|---:|---:|---:|
| smoke | 20--40 minutes | below 6 GiB | 8--16 GiB available | below the pilot allocation |
| pilot | 3--6 hours | below 6 GiB | 8--16 GiB | 8--15 GiB |

Simulator screening, Google Drive latency, cache state, and Colab preemption
can dominate wall time. A free Colab GPU may have enough device memory but is
not recommended for the uninterrupted pilot because of its session limit and
variable availability. `MAX_ESTIMATED_TOTAL_MINUTES = 420` is a configuration
and planning estimate only. The implementation records elapsed time but does
not enforce a 420-minute timeout or shrink the scientific design.

## Expected run phases and sanity output

Before locked evaluation, the notebook must print:

1. source commit and hashes for the notebook, builder, and numerical module;
2. exact checkpoint and DINOv2 hashes;
3. resolved nonce, fresh/resumed status, run signature, device,
   Python/CUDA/PyTorch versions, and cache status;
4. candidate ranges, selected complete-trajectory IDs, 32/32/32/64 record
   counts, mode counts, exact core/evaluation word values, and prefix audit;
5. the empirical simulator-determinism floor: two exact restored rollouts for
   each of `a`, `ab`, `AAB`, and `ABAB` on every evaluation record, with branch
   count, maximum absolute difference, and RMSE saved;
6. construction and model-selection rank spectra and bootstrap stability;
7. the frozen common rank (cap 12), carrier-basis rank (cap 16),
   interface/operator/mode-cluster hashes, and intervention locality/fidelity
   checks;
8. the two within-model bridge hashes, the sole calibration-only JEPA-to-DINO
   map hash, strict-fit and conditioning diagnostics, and the evaluation-open
   certificate; and
9. expected versus observed simulator branches and model forward passes.

The locked evaluation then reports global, physical-mode, and label-free
hybrid errors; the 208-feature cap-matched (or rank-overmatched) nonlinear
capacity control; same-model split-half reliability and the DINO self-positive
edit; the sole forward map and action-conditioned local-map upper bound;
JEPA-to-DINO additive grounded interchange and decoder-mediated additive
planning transport; zero-edit, state-permutation, random-orthogonal-map, and
random-matched-subspace controls; 5,000-draw clustered intervals with the five
preregistered one-sided hypotheses Holm-adjusted; memory; and elapsed time.

Stop and preserve `FAILURE_TRACE.txt` if the notebook reports a stale or mock
asset, hash mismatch, failed restore, non-executable action, non-prefix-closed
bank, missing regime coverage, pilot/smoke count mismatch, evaluation leakage,
or incompatible cache. Do not bypass the error by editing a later cell.

## Drive directory and safe resume

The durable root is:

`MyDrive/counterfactual_faithfulness_stage33_bipca/`

A run writes to a directory named with the first 12 hex characters of the run
signature:

`<mode>_<run_signature[:12]>/`

The directory name and startup log record the resolved nonce. With
`RESUME_INCOMPLETE = True`, an intact compatible incomplete-run pointer is
resumed automatically when neither a manual nonce nor a Drive request is
present. After a Colab runtime loss, use this explicit method when automatic
resume does not select the intended incomplete run:

1. create
   `MyDrive/counterfactual_faithfulness_stage33_bipca/stage33_run_request.json`
   containing
   `{"protocol_id":"stage33-bounded-interventional-predictive-causal-abstraction-v1","run_nonce":"<exact nonce>"}`;
2. keep `MANUAL_RUN_NONCE` blank in the committed notebook; and
3. keep run mode, committed source, checkpoints, configuration, Drive root, and design
   unchanged, then use **Runtime -> Run all** from a fresh runtime.

Remove the Drive request after the intended run completes so a future run does
not deliberately reuse its nonce. Leaving the request absent and the manual
field blank starts a fresh automatic nonce unless a compatible incomplete
pointer exists. Directly editing `MANUAL_RUN_NONCE` in the opened committed
notebook violates exact source binding; use that field only in a rebuilt and
committed derivative with newly bound hashes.

Resume is content-addressed. A completed atomic shard is reused only if its
configuration, source, checkpoint, split, action-bank, row-count, and content
hashes all match. Temporary or incomplete shards are discarded and recomputed.
A changed source/configuration fails compatibility and requires a fresh nonce.

Once `evaluation_open_certificate.json` exists, the construction,
model-selection, calibration, rank, operators, mode clusters, within-model
bridges, intervention rules, and sole cross-model map are immutable for that
nonce. Resume may finish missing locked-evaluation shards, analysis, plots, or
packaging; it may not refit an earlier object. A completed result directory is
idempotent: rerunning packages the same verified evidence rather than appending
a second analysis.

## Returned bundle

The notebook downloads a ZIP named like:

`stage33_bipca_result_bundle_<run_signature>.zip`

The compact ZIP must include:

- configuration, versions, source and checkpoint identities, cache state,
  nonce/resume provenance, and integrity certificates;
- physical design, trajectory/state-family/action-composition splits, action
  values, prefix-closure and coverage manifests;
- restore and executable-action audits, the exact two-restore simulator floor
  for all four diagnostic words, and the evaluation lock;
- observed spectra, all 256 structured-null and 512 trajectory-bootstrap rank
  draws per reported model/split analysis (or lossless summaries), per-model
  selected ranks, the common-rank lock, and model-selection history;
- decoder, predictive-chart, rank-16 carrier-basis, operator, physical/label-free
  mode, within-model bridge, and sole cross-model map manifests and hashes;
- raw or losslessly summarized unit rows for global/hybrid prediction, forward
  conjugacy, JEPA-to-DINO additive grounded interchange, and decoded additive
  planning transport;
- zero-edit, state-permutation, random-orthogonal-map,
  random-matched-subspace, DINO self-positive, action-reversal,
  mode-permutation, decoder-only mean, same-model split-half,
  action-conditioned-map, no-label, nonlinear, and simulator-oracle controls;
- 5,000-draw clustered-bootstrap seeds/indices or sufficient summaries, 95%
  intervals, Holm multiplicity records, family tables, plots, timing and
  memory reports;
- `stage33_decision.json`, the automatic interpretation report, logs,
  `FAILURE_TRACE.txt`, and a ZIP manifest; and
- hashes and relative Drive paths for large raw activation/image shards omitted
  from the compact download.

The ZIP is the normal artifact to return. If its audit says a required large
shard was omitted rather than merely hash-bound, also return the complete Drive
run directory; the notebook must state this explicitly.

## Reading the result

Read the top-level `status` first:

- `BOUNDED_INTERVENTIONAL_PREDICTIVE_CAUSAL_ABSTRACTION_SUPPORTED` is the only
  label supporting the full bounded forward-direction claim. It requires an
  eligible pilot to pass grounding/rank, hybrid, strict one-map, same-model
  positive-control, held-out conjugacy, additive interchange, additive decoded
  planning, family-consistency, and specificity gates.
- `OPERATOR_CONJUGACY_WITHOUT_FULL_CAUSAL_PLANNING_CERTIFICATE` supports only
  the bounded forward operator-conjugacy result: the prerequisite gates pass,
  but the full interchange/planning conjunction does not.
- `BOUNDED_SHARED_ABSTRACT_MECHANISM_NOT_SUPPORTED` is the mechanically derived
  negative label for a completed eligible pilot whose same-model positive
  control passes but whose evidence is weaker than the first two labels.
- `INCONCLUSIVE_SAME_MODEL_POSITIVE_CONTROL_FAILURE` means split-half
  reliability or the DINO self-positive edit failed. Do not interpret it as
  evidence against a cross-model mechanism.
- `SMOKE_ONLY` identifies every smoke run, regardless of favorable metrics.
- `INCONCLUSIVE_PIPELINE_FAILURE` may be written after an execution exception;
  it is an operational trace, not a scientific result.

Inside the numerical report, the nested `protocol_decision.status` remains the
raw aggregate `pass`, `partial_pass`, or `fail`; it is not the top-level result
label. `partial_pass` corresponds to the operator-conjugacy-only case when the
required prerequisite and same-model gates pass.

Do not infer success from notebook completion, attractive figures, similar
ranks, or a low pooled error. The label is derived mechanically from the
preregistered family-level gates.
