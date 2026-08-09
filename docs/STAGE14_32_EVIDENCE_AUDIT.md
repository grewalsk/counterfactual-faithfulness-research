# Stage 14–32 evidence audit

This ledger records the evidential boundary used to design Stage 33. It was
reconstructed from the checked-in notebooks and the locally saved result
bundles, not from the project summary alone. A positive model-internal result
is not relabeled as grounded physics, and a failed preregistered gate is not
rescued by a favorable descriptive statistic.

## Stage 32 integrity and result

Audited bundle:
`stage32_bounded_confirmation_result_bundle_904cced9e478`

- All 38 files in the compact bundle match the SHA-256 values in
  `result_zip_manifest.json`; `FAILURE_TRACE.txt` is `NONE`.
- The source-bound run resolved commit
  `7c9650e23d8737ee535269eee78b92342d865369` and the frozen JEPA, DINO-WM,
  and DINOv2 hashes match the notebook contract.
- The run screened 800 states, selected 160 fresh persistent-contact states,
  and evaluated 480 state-family rows per model with zero cache hits. Hook
  identity error was zero.
- The exact status is `PAIRED_SIGNAL_WITHOUT_SUBSPACE_SPECIFICITY`.
- JEPA within-model out-of-fold MSE fell from `0.041524` to `0.034562`
  (16.77%); the state-bootstrap absolute-improvement interval was
  `[0.004860, 0.008998]`.
- DINO-WM MSE fell from `0.035635` to `0.031758` (10.88%); its interval was
  `[0.002277, 0.005656]`.
- In the paired DINO-minus-JEPA panel, MSE fell from `0.055215` to `0.051631`
  (6.49%); the interval was `[0.001915, 0.005380]`, with positive mean
  improvement in all three action families.
- The shuffled control improved by only 0.82%, but the two construction-frozen
  empirical-action-span random bases improved by 6.94% and 6.82%. Primary
  minus median placebo was `-0.000275`, interval
  `[-0.001143, 0.000614]`. The preregistered specificity gate therefore
  failed.
- Direct grounded cosine was descriptively higher for the primary basis than
  for either random basis in both models (JEPA `0.0786` versus approximately
  `0.053`; DINO `0.0794` versus approximately `0.058`). This descriptive
  alignment advantage did not yield unique downstream planning-reliability
  information.
- The two random controls are not arbitrary ambient subspaces. They are
  rank-128, norm-matched, primary-orthogonal bases inside the construction
  action-contrast span. Only two draws were tested, so the result establishes
  non-uniqueness in that empirical span, not invariance to every random edit.
- The compact ZIP omits the raw truth shards and upstream Stage 31 basis
  arrays. Their hashes and row summaries are present, but the local compact
  bundle alone cannot replay the GPU forward passes.

## Frozen decisions by stage

| Stage | Frozen outcome | Strongest warranted interpretation |
|---|---|---|
| 14 | `NO_EVIDENCE_FOR_THIS_FROZEN_GLOBAL_FRAME` | The proposed global frame failed reconstruction, causal mediation, specificity, and temporal gates. |
| 15 | `PIPELINE_FAILURE` | The fixed flat reader failed before operator/intervention inference. A later evaluation-informed coordinate reader is exploratory only. |
| 16 | `FAIL` | Physical quantities are accessible to a coordinate-moment reader, but reader specificity and trajectory-robust advantage failed. |
| 17 | `FULL_SWAP_ONLY_NO_COMPRESSED_MEDIATION` | A full activation swap worked; the rank-32 compressed mediator did not meet absolute or specificity gates. |
| 18 | `CONFIRMED_BIDIRECTIONAL_RANK64_MEDIATOR` | A rank-64 model-output mediator transferred the model's own action-conditioned effect. Rank 128 performed better, so rank 64 is not an identified intrinsic dimension. |
| 19 | `CONFIRMED_TRANSFER_ALL_UNSEEN_ACTION_FAMILIES` | Model-own sufficiency/necessity transferred across five action families. Planning evidence was secondary and inconsistent. |
| 20 | `PREDICTION_MEDIATOR_TRANSFER_WITHOUT_CONFIRMED_PLANNER_STEERING` | Output transfer replicated, but neither planner-chain family passed steering gates. |
| 21 | `COHERENT_HANDOFF_WITHOUT_CAUSAL_SUBSPACE_UTILITY` | Complete swaps reproduced oracle choices, while held-out subspace utility failed in both families. |
| 22 | `PHYSICAL_MODE_WITHOUT_CAUSAL_INTERACTION` | A label-free two-mode readout aligned strongly with contact, but the causal interaction was too small and nonspecific. |
| 23 | `MODE_FLIPS_WITHOUT_OPERATOR_SWITCH` | The edit flipped the mode readout, but output operator transfer was effectively zero. This rejects an established internal free/contact operator switch. |
| 24 | `NO_RANK64_CAUSAL_COMPLETION` | Structured residual directions exist, but no tested compact completion met the frozen criterion. |
| 25 | `IMPULSE_READABLE_BUT_NOT_CAUSALLY_USED` | Impulse/contact were accurately decoded; erasing the representation produced essentially no causal effect. |
| 26 | `CONTACT_FIELD_READABLE_BUT_NOT_CAUSALLY_TRANSPORTABLE` | A contact-frame reader beat a world-axis reader, but aligned low-rank causal transport failed. |
| 27 | `CAUSAL_NONCOMMUTATIVE_ACTION_DYNAMICS_SUPPORTED` | A finite, contact-amplified action-order effect and a JEPA carrier effect passed. This is not an infinitesimal Lie-bracket theorem. |
| 28 | `MODEL_DOES_NOT_CAPTURE_PHYSICAL_CONTROL_AREA_LAW` | The simulator obeyed the tested signed-area law; the model was misaligned with it even though its carrier mediated a model-self area effect. |
| 29 | `PHYSICAL_READOUT_LIMITATION_SUPPORTED` | Native predicted/encoded-future closure and self-intervention closure passed. Frozen grounded closure remained weak and failed its coefficient/gain thresholds. |
| 30 | `GROUNDED_CLOSURE_PREDICTS_PLANNING_RELIABILITY_ONLY` | Grounded alignment predicted native planning regret, but did not establish causal ablation value or a control advantage. |
| 31 | `WITHIN_MODEL_REPLICATION_WITHOUT_PAIRED_CERTIFICATE` | Both within-model reliability effects replicated; the paired interval crossed zero and both matched-ablation-control gates failed. |
| 32 | `PAIRED_SIGNAL_WITHOUT_SUBSPACE_SPECIFICITY` | The paired reliability association passed, but matched empirical-span random bases reproduced it. |

## Audit of the project-summary statements

1. **“Both representations predict grounded physical quantities.”** Supported
   as common decodability, including the state-disjoint Stage 2C probes and
   Stage 31/32 reliability panels. It does not imply a shared internal state.
2. **“Stage 32 found a positive paired JEPA-versus-DINO signal.”** Supported
   only as incremental prediction of DINO-minus-JEPA normalized regret. It is
   not a preregistered mean JEPA-superiority result.
3. **“The primary subspace was not specific.”** Supported within the tested
   construction action span; the two random controls were slightly better on
   the downstream regression.
4. **“The primary subspace had stronger causal-closure alignment.”** Supported
   descriptively by bounded grounded cosine and self-consistency, but not by
   the downstream specificity gate.
5. **“Contact-aligned state.”** Supported observationally. Causal sufficiency,
   Markov-state status, and a selected internal operator are not supported.
6. **“Free/contact operator differences.”** Supported in the physical
   simulator; not established as a causally selected neural operator. Stage 23
   is a direct negative result.
7. **“Action-order noncommutativity.”** Supported narrowly as a finite
   contact-amplified order effect in one JEPA PushT checkpoint.
8. **“Pulse area.”** The simulator law is supported; the model's capture of it
   is explicitly rejected by Stage 28.
9. **“Contact transport.”** Not supported as a neural causal-transport claim;
   Stage 26 failed.
10. **“Grounded closure and planning value.”** Grounded alignment predicts
    planning reliability. Direct grounded closure, planner steering, causal
    subspace utility, and ablation value remain unsupported.

The defensible synthesis is therefore:

> Physically grounded alignment is associated with planning reliability in
> both public PushT predictors, and that association is non-unique across the
> tested empirical action-span subspaces.

“Broad redundant physics” is a hypothesis, not yet a result. The random bases
could carry distributed physical information, exploit shared output geometry,
or track state/output nuisances. Stage 33 distinguishes these explanations by
requiring one held-out operator map to survive model-native internal transport
and physical planning controls.

## Cross-stage caveats

- Both public models share a DINOv2 target/output family, so common output
  geometry is a live alternative explanation.
- Only one checkpoint per model family and one environment are available.
- The models are consistently more self-consistent than physically grounded;
  successful mediation of a model's own prediction is not physical validity.
- No stage identified a stable minimal predictive rank. Performance often
  improved from rank 64 to 128, and Stage 24 found no compact completion.
- Stage 31/32 used separate model-specific bases and did not test a common
  state map, operator conjugacy, cross-model internal interchange, or planning
  preservation after transport.
- The research sequence is adaptive even when each local notebook freezes its
  own gates. Stage 33 is prospective only when its new source-bound splits and
  evaluation lock are respected.
