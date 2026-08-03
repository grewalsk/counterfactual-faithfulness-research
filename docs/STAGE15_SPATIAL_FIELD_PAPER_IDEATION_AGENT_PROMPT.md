# Prompt for an independent JEPA spatial-field research agent

Copy the prompt below into a research-capable agent with access to this
repository and the internet. The requested work is analysis and experiment
design, not execution.

---

You are an independent research lead deciding whether a new mechanistic
interpretability program for JEPA world models can support an ICLR paper,
spotlight-level contribution, or a genuinely best-paper-caliber result. Be
ambitious, but adversarial. Do not assume that the existing “spatial field” or
“predictive-control bundle” interpretation is correct merely because it is
mathematically appealing.

Your objective is to design the shortest falsifiable research program capable
of climbing this claim ladder:

1. **Fresh reader confirmation:** establish a credible ICLR paper foundation.
2. **Generalization across layers, environments, checkpoints, and JEPA
   models:** reach strong ICLR or spotlight territory.
3. **Causal coordinate operators:** move predicted objects while preserving
   unrelated represented state, creating a potentially major interpretability
   result.
4. **Unified mathematical framework:** predict where the representation lives,
   how it transports through time, and when JEPA succeeds or fails, creating a
   best-paper ceiling.

You must decide what experiments could actually earn each claim. A list of
interesting ideas is not sufficient.

## Operating constraints

- Begin read-only. Do not start Colab, allocate a GPU, download a checkpoint,
  or launch a large experiment while producing the design memo.
- Use only new data for confirmation. Stage 15 evaluation trajectories 1, 3,
  5, and 7 have been inspected and may never be used for new feature-family,
  threshold, layer, horizon, or hyperparameter selection.
- Separate exploratory, validation, and confirmatory evidence explicitly.
- Treat trajectories or independently generated tasks—not token patches,
  timepoints, actions, or individual target predictions—as the sampling unit.
- Propose cheap reader-only falsification before Jacobians or causal
  interventions, and a narrow causal gate before a full layer/model matrix.
- Do not recommend an unconstrained nonlinear probe as primary evidence.
  Probe capacity, spatial inductive bias, data volume, and coordinate access
  must be controlled separately.
- Never infer causal representation from decoding alone.
- Preserve negative results. Do not rewrite the failed preregistered flat
  CountSketch as though the coordinate reader had been specified in advance.
- Prefer public checkpoints, public environments, reproducible source binding,
  resumable shards, and CPU analysis after extraction.
- Estimate GPU hours, peak memory, storage, and approximate Colab cost for each
  proposed phase. Design every notebook to stop before expensive stages when a
  scientific gate fails.

## Evidence you must inspect

Read the implementation and evidence rather than relying on this prompt:

1. `README.md`
2. `docs/MATHEMATICAL_SPEC.md`
3. `docs/STAGE13_JEPA_JACOBIAN_OUTCOME_WORKSPACE_IDEA.md`
4. `docs/STAGE13_JOW_LITERATURE_AND_FALSIFICATION_PLAN.md`
5. `notebooks/14_predictive_control_j_bundle_pilot.ipynb`
6. `notebooks/build_stage14_predictive_control_notebook.py`
7. `src/cf_faithfulness/stage14_pcj.py`
8. `docs/STAGE15_LONGITUDINAL_BUNDLE_PROTOCOL.md`
9. `notebooks/15_longitudinal_predictive_control_bundle.ipynb`
10. `src/cf_faithfulness/stage15_bundle.py`
11. `tests/test_stage15_bundle.py`
12. `audits/stage15/reader_failure_audit/AUDIT.md`
13. `audits/stage15/reader_failure_audit/reader_failure_audit.json`
14. `audits/stage15/reader_failure_audit/model_metrics.csv`
15. `audits/stage15/reader_failure_audit/construction_cv.csv`
16. `audits/stage15/reader_failure_audit/renderer_geometry.csv`
17. `audits/stage15/reader_failure_audit/raw_full_manifest.json`
18. `scripts/audit_stage15_reader_failure.py`

If the external raw bundle is available, verify it using
`audits/stage15/reader_failure_audit/RAW_EVIDENCE.md` and recompute the main
numbers locally. The raw tensors are not required merely to write the design
memo. Do not use the observed evaluation split to select another reader.

## Current evidence to independently verify

The stopped Stage 15 run contains 520 construction examples from trajectories
0, 2, 4, and 6, plus 520 now-inspected evaluation examples from trajectories
1, 3, 5, and 7. It saved target tokens at two horizons but stopped before
operator or causal shards were generated.

The preregistered flattened 192-dimensional CountSketch reader failed with
median evaluation R² 0.144 and minimum spatial R² -0.543. Increasing the
random feature count did not rescue it:

| Reader | Evaluation median R² | Minimum spatial R² |
|---|---:|---:|
| Flat CountSketch, 1,152 features × 3 | 0.153 | -0.141 |
| Flat CountSketch, 2,304 features × 3 | 0.189 | -0.178 |
| Coordinate moments, degree 1 | 0.329 | 0.207 |
| Coordinate moments, degree 2 | 0.407 | 0.190 |

The coordinate readers are fixed global linear maps of the token tensor. For
patch coordinate `u_p=(x_p,y_p)` and token vector `z_p`, they use moments such
as

\[
\frac1P\sum_p z_p,\quad
\frac1P\sum_p x_pz_p,\quad
\frac1P\sum_p y_pz_p,
\]

with degree 2 adding `x²`, `xy`, and `y²`. Ridge strengths were selected only
by leave-one-construction-trajectory-out CV. Both readers pass the old pooled
gate at both horizons, but each passes only one of four
leave-one-evaluation-trajectory-out aggregate subsets. Therefore these are
post-hoc and fragile observations, not confirmation.

The narrow evidence-supported statement is:

> JEPA target tokens contain transferable physical information whose linear
> accessibility depends strongly on respecting token coordinates.

The candidate paper thesis—still unconfirmed—is:

> A JEPA world model represents predictive state as a spatially indexed,
> temporally transported field rather than primarily as a fixed global
> subspace, so physically meaningful observables and interventions require
> coordinate-respecting charts.

## Required theoretical audit

Introduce precise notation, for example a layer- and horizon-specific token
field

\[
Z_{\ell,h}(s,a)\in\mathbb{R}^{P\times d},
\]

but do not inherit the existing vocabulary uncritically. Determine whether the
right object is actually:

- a spatially indexed feature field;
- an equivariant representation;
- a vector bundle over physical or latent state;
- a family of local controllability/observability spaces;
- a Koopman-like observable system;
- or something simpler that does not justify bundle language.

For every proposed mathematical object, define:

1. the base space, fiber, chart, and observable, if those terms are used;
2. which transformations are coordinate changes or gauge freedoms;
3. the action differential or write operator;
4. the output differential or read operator;
5. a transport map between states or times;
6. at least one falsifiable empirical prediction not already guaranteed by
   spatial patch embeddings or renderer geometry;
7. a counterexample showing when the formulation should fail.

Distinguish these possibilities:

- spatial information is merely linearly decodable;
- the token lattice supplies a useful fixed basis;
- local operators vary smoothly with physical state;
- modes transport across time;
- transported modes causally control a stable physical observable;
- the proposed geometry predicts generalization or failure before evaluation.

The last two distinctions are essential for a major paper.

## Design the four-stage evidence ladder

### A. Fresh reader confirmation

Write a preregisterable reader-only experiment using entirely new trajectories.
Start from the current proposal of 16 trajectories, split 8 construction and 8
evaluation, five timepoints, and horizons 1 and 3, but change these numbers if
power or trajectory diversity requires it.

Freeze degree-1 coordinate moments as the primary reader and degree 2 as a
prespecified sensitivity analysis. Include at least:

- the original 192-dimensional CountSketch;
- dimension-matched random sketches;
- channel means;
- spatial pooling with matched effective degrees of freedom;
- coordinate permutations or scrambled patch coordinates;
- image/render-only oracle ceilings reported separately from JEPA readouts;
- native predicted proprioception, if the checkpoint exposes it;
- trajectory-clustered uncertainty and leave-one-trajectory-out stability;
- horizons reported separately;
- a precise rule for handling low within-trajectory target variance.

Audit whether the proposed thresholds—median six-coordinate R² at least 0.30,
every spatial R² at least 0.20, and trajectory-deletion stability—are justified
or need replacement. Specify one immutable confirmatory decision rule and all
possible outcomes.

### B. Generalization

Design a staged matrix across layers, horizons, physical environments,
checkpoint seeds/training stages, and genuinely different JEPA-family models.
Do not propose a Cartesian product by default. Rank axes by information gained
per GPU hour and identify the minimal result that distinguishes:

- generic ViT positional information;
- visual-object localization;
- learned predictive state;
- action-conditioned predictive structure;
- JEPA-specific organization.

Require matched non-JEPA or architecture controls where necessary. Verify that
every named checkpoint and environment is publicly obtainable and suitable
before recommending it. Explain how readers are frozen or transferred across
models with different patch grids and channel dimensions. Provide multiplicity
control and define what “generalizes” means before observing results.

### C. Causal coordinate operators

Design interventions that attempt to move the model's predicted agent or
block while preserving unrelated state. Decoding alone is insufficient.
Specify:

- where in the predictor the edit is injected;
- how a desired physical displacement is converted into a token-space edit;
- whether the operator is local, state-conditioned, transported, or shared;
- how edit energy and token support are matched;
- how positive/negative dose response and linearity are tested;
- how the outcome is read independently of the reader used to construct the
  edit;
- preservation metrics for object identity, orientation, agent position,
  block position, background, and ordinary future prediction;
- exact-support random, covariance-shaped, coordinate-scrambled,
  time-shuffled, local-mode, no-edit, and renderer-artifact controls;
- tests distinguishing a genuine transported mode from independently
  rediscovering a local edit at every state;
- failure criteria for off-manifold edits and general activation corruption.

The primary causal endpoint must be a selective physical displacement with a
prespecified sign and dose response, not merely a changed latent norm or
reader score.

### D. Unified predictive theory

Propose the smallest theory that could explain all validated results while
making new predictions. Derive at least one theorem, bound, or testable
proposition connecting some subset of:

- coordinate moments and physical observability;
- token-field smoothness and sample-efficient decoding;
- action Jacobians and controllable write spaces;
- output gradients and observable read spaces;
- temporal transport and mode persistence;
- equivariance, patch resolution, and failure under coordinate scrambling;
- layer depth and the transition from local visual support to predictive
  control structure.

State the assumptions and identify which are empirically testable. Explain
what result would prove the theory is only retrospective vocabulary. A
best-paper-caliber theory should correctly predict at least one layer,
horizon, environment, checkpoint, or model where the effect appears or fails
before that condition is evaluated.

## Literature and novelty audit

Search current primary papers and official repositories. Cover at minimum:

- JEPA, V-JEPA, V-JEPA 2, and JEPA world-model objectives;
- mechanistic interpretability of vision transformers and world models;
- distributed and spatial probing, equivariant representations, and object
  slots;
- causal representation editing and activation interventions;
- controllability/observability, Koopman representations, and successor
  features;
- vector bundles, gauge alignment, and representation transport where these
  have actually been operationalized in machine learning.

Identify the five closest papers. For each, give the precise overlap, the
remaining novelty, and an experiment that would distinguish this project from
that work. Use primary sources and direct citations. Do not claim novelty from
terminology alone.

## Required deliverable

Produce one decision memo with the following sections.

### 1. Executive verdict

State the current evidence tier and choose exactly one immediate action:

- run the fresh reader confirmation;
- run a cheaper diagnostic first;
- redesign the primary reader before confirmation;
- or stop this direction.

Explain why it dominates the alternatives.

### 2. Claim ladder

For each of the four target tiers, provide:

- exact claim text;
- minimum experiment set;
- primary estimand and sampling unit;
- controls;
- pass, fail, and ambiguous outcomes;
- compute and storage estimate;
- likely paper value if successful;
- what must not be claimed.

### 3. Formal model

Define the proposed mathematical object, derive its observable and controllable
quantities, state assumptions, and give at least three predictions that can be
tested prospectively.

### 4. Ranked experimental program

Give a dependency-ordered sequence, not a wish list. For every experiment,
specify what is frozen, what is fitted, which split can select parameters, the
stopping gate, and what later work is canceled after failure.

### 5. Exact fresh-confirmation protocol

Provide enough detail that another agent could build the notebook without
making scientific choices: seeds, trajectories, states, horizons, layers,
feature maps, normalization, regression, uncertainty, thresholds, saved
artifacts, source binding, resumability, and runtime guards.

### 6. Exact causal pilot

Give equations or pseudocode for the edit construction, independent outcome
measurement, preservation tests, matched controls, intervention doses, and
decision rule. Explain how the design rules out “the probe moved because we
optimized the probe.”

### 7. Generalization matrix

Give a prioritized table of environments, checkpoints, layers, and models,
including availability, expected cost, scientific role, and the order in which
they should be attempted.

### 8. Novelty comparison

Compare against the closest primary work and state the narrowest defensible
novel contribution.

### 9. Risk register

Cover post-hoc selection, trajectory leakage, patch-coordinate confounding,
renderer leakage, probe capacity, low-variance R² instability, off-manifold
edits, model-specific hooks, multiplicity, checkpoint availability, and Colab
cost overruns.

### 10. Final recommendation

End with:

- the single next experiment;
- the frozen decision gate;
- expected runtime and cost;
- the strongest claim after a pass;
- the interpretation after a failure;
- and the one later observation that would most increase the work's ceiling
  from ICLR-level to best-paper-level.

Do not end with “more research is needed.” Make the decision.

---
