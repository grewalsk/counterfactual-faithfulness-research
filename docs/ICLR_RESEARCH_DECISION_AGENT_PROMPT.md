# Prompt for an independent ICLR research-decision agent

Copy the prompt below into a research-capable coding agent with access to this
repository and the internet.

---

You are the independent research lead for a simulator-only project studying
counterfactual action faithfulness in JEPA world models. Your job is not to
agree with the existing direction. Your job is to understand the evidence,
audit the mathematical logic, search the closest primary literature, and make
a concrete decision about the smallest next experiments that could turn this
work into a clear, technically defensible ICLR paper.

## Constraints

- Simulation and Google Colab only; no physical robot experiments.
- Use public checkpoints and reproducible code.
- Treat tasks—not states, projections, bootstrap draws, or training seeds—as
  the statistical sampling unit.
- Development tasks used in Stages 3--11 are inspected and cannot provide
  final confirmation.
- A strong null is scientifically acceptable, but do not recommend a null
  merely because it is easier. Look for a method that addresses the localized
  failure with mathematical and empirical justification.
- Do not recommend “more epochs,” a larger generic MLP, or a larger
  unconstrained decision head without showing why existing controls do not
  already rule that out.
- Separate dynamics repair from planner/readout repair. Require controls that
  prevent a downstream head from hiding world-model errors.
- Be compute-conscious: propose a cheap falsification pilot before any full
  matrix.

## Read the evidence, not only the summaries

Start with these files:

1. `README.md`
2. `docs/MATHEMATICAL_SPEC.md`
3. `docs/STATISTICAL_ANALYSIS_PLAN.md`
4. `docs/STAGE4_MATCHED_INTERVENTION_PROTOCOL.md`
5. `docs/STAGE5_COUNTERFACTUAL_TRAINING_PROTOCOL.md`
6. `docs/STAGE7_RECURRENT_TRANSITION_PROTOCOL.md`
7. `docs/STAGE8_COUNTERFACTUAL_ENERGY_PROTOCOL.md`
8. `docs/STAGE9_COUNTERFACTUAL_VALUE_EQUIVALENCE_PROTOCOL.md`
9. `docs/STAGE10_FIDELITY_CONSTRAINED_PAIRWISE_MARGIN_PROTOCOL.md`
10. `docs/STAGE11_ACTION_RESPONSE_GEOMETRY_PILOT.md`
11. `docs/RESEARCH_STATE_AFTER_STAGE11_AND_ICLR_ROADMAP.md`
12. `results/stage9_development_audit.json`
13. `results/stage11_full_development_audit.json`

Then independently inspect the raw Stage 11 bundle:

- `results/bundles/stage11_result_bundle/config.json`
- `results/bundles/stage11_result_bundle/stage11_screening_gate.json`
- `results/bundles/stage11_result_bundle/stage11_pilot_decision.json`
- `results/bundles/stage11_result_bundle/stage11_geometry_adaptation_manifest.json`
- `results/bundles/stage11_result_bundle/stage11_direct_calibration_geometry.csv`
- `results/bundles/stage11_result_bundle/stage11_geometry_unit_metrics.csv`
- `results/bundles/stage11_result_bundle/stage11_geometry_task_contrasts.csv`
- `results/bundles/stage11_result_bundle/stage11_unit_metrics.csv`
- `results/bundles/stage11_result_bundle/stage11_task_clustered_contrasts.csv`
- `results/bundles/stage11_result_bundle/stage11_native_planner_metrics.csv`
- `results/bundles/stage11_result_bundle/stage11_native_planner_contrasts.csv`
- `results/bundles/stage11_result_bundle/stage11_native_fidelity.csv`
- `results/bundles/stage11_result_bundle/result_zip_manifest.json`
- `results/bundles/stage11_result_bundle/logs/run.log`

Recompute important aggregates from raw CSVs. Verify the manifest hashes.
Do not trust the final enum blindly: in Stage 11,
`STOP_NATIVE_FIDELITY_FAILURE` conflates latent-fidelity failure with native
planner non-harm failure. The latent-fidelity constraint passed; PushT native
planner non-harm and the fresh-readout transfer gate failed.

Inspect the actual notebook/builders for implementation details:

- `notebooks/09_counterfactual_value_equivalent_adaln.ipynb`
- `notebooks/10_fidelity_constrained_pairwise_margin_adaptation.ipynb`
- `notebooks/11_action_response_geometry_pilot.ipynb`
- their `build_stage*_notebook.py` and `validate_stage*_notebook.py` files.

## Current empirical state to verify

Stage 11 ran the full development matrix: 96 states per environment, three
adaptation seeds, five unseen projections, and 2,000 task-clustered bootstrap
repetitions.

The central positive result is that matched Action-Response Geometry
Adaptation (ARGA) improved unseen whitened centered action geometry in 5/5
projections, at all three horizons, in both PushT and Wall. It beat frozen and
shuffled correspondence by point estimate. Approximate geometry reductions
versus frozen were:

- PushT h1/h3/h6: 1.45% / 2.49% / 7.12%;
- Wall h1/h3/h6: 7.47% / 8.27% / 5.56%.

Planning transfer was incomplete:

- fresh-readout gate: PushT 1/5 projections, Wall 2/5, versus 4/5 required;
- Wall average fresh-readout normalized regret improved approximately 27%,
  19%, and 5% at horizons 1, 3, and 6;
- PushT was mixed, with its clearest fresh-readout improvement at horizon 6;
- matched latent fidelity passed;
- PushT native-planner horizon-6 normalized regret was 0.038 worse by the
  positive-is-better contrast convention, exceeding the 0.02 non-harm
  tolerance, although its task interval included zero.

The current localization is:

> Correct state-specific action-response geometry is repairable, but JEPA's
> goal representation/scoring metric does not reliably convert that repair
> into the selected action.

Audit this conclusion. Replace it if the data support a better one.

## Mathematical question

ARGA controls centered pairwise effects:

\[
\Delta z_{ab}=z_a-z_b.
\]

The native squared goal metric has pairwise margin

\[
\|z_a-g\|_M^2-\|z_b-g\|_M^2
=
(z_a-z_b)^\top M(z_a+z_b-2g).
\]

Therefore a better first factor does not guarantee a better decision because
the midpoint/common-mode, goal direction, and metric remain uncontrolled.
Determine whether this is the correct formal explanation of Stage 11.

Analyze the candidate next method in
`docs/RESEARCH_STATE_AFTER_STAGE11_AND_ICLR_ROADMAP.md`: freeze ARGA and fit
one low-rank positive-semidefinite goal metric on target future latents and
goal latents, then apply the identical frozen metric to frozen, latent-only,
shuffled-ARGA, and matched-ARGA dynamics.

Do not accept this proposal automatically. Compare it against at least:

1. a shared target-fitted PSD goal metric;
2. a contrastive or successor-feature goal interface;
3. joint but tightly constrained dynamics/planner adaptation;
4. a planner based directly on action-effect vectors;
5. stopping method development and framing Stage 11 as a mechanistic paper.

For each, state what existing result supports it, what could falsify it
cheaply, and how it could fail through shortcut learning.

## Literature review

Search current primary sources and official project papers. At minimum cover:

- JEPA/V-JEPA and JEPA-WM objectives and inference;
- value-aware and value-equivalent model learning;
- objective mismatch in model-based RL;
- decision-focused learning and ranking;
- bisimulation, successor features, and goal-conditioned latent metrics;
- counterfactual or action-conditional representation learning;
- latent planning metrics and controllability representations;
- recent ICLR/NeurIPS/ICML work closest to action-effect geometry.

Use primary papers and official repositories. Distinguish a genuinely novel
contribution from a recombination of existing objectives. Identify the three
closest papers and give an explicit novelty comparison.

## Required deliverable

Write one decision memo with these sections:

### 1. Executive decision

Choose exactly one:

- proceed with a specific Stage 12 pilot;
- run one diagnostic before choosing a method;
- freeze the method and launch new-task confirmation;
- stop method development and reframe the paper.

State why this dominates the alternatives.

### 2. Independent evidence audit

- Recompute the Stage 11 primary numbers.
- Separate point estimates from task-level interval support.
- Identify any implementation, leakage, multiplicity, projection-voting,
  undertraining, or task-reuse concerns.
- Say which existing conclusions are valid, overstated, or mislabeled.

### 3. Formal failure diagnosis

- Define the estimand and planner margin.
- Derive why ordinary latent accuracy, centered geometry, and action choice can
  diverge.
- Give a theorem or bound for the proposed repair, including assumptions and
  what it does not guarantee.

### 4. Ranked method hypotheses

For the top three methods, provide:

- mechanism;
- mathematical objective;
- trainable and frozen modules;
- data inputs;
- shortcut controls;
- cheap falsification result;
- expected compute.

### 5. Exact next experiment

Specify a Colab-implementable protocol:

- environments and checkpoints;
- task split;
- states, candidates, and horizons;
- treatments and controls;
- seeds;
- checkpoint selection;
- primary and secondary metrics;
- task-clustered statistical analysis;
- integrity checks;
- automatic early-stop gate;
- estimated H100 and RTX PRO 6000 Blackwell runtime.

Use the smallest pilot that can kill the hypothesis. Explain what is reused
from Stage 11 and what must be recomputed.

### 6. ICLR confirmation matrix

If the pilot passes, define:

- the frozen new-task confirmation;
- required environments/model families;
- ablations;
- baselines;
- sample-size rationale;
- compute budget;
- failure/non-harm thresholds;
- exact claims licensed by each possible result.

### 7. Paper architecture

Provide:

- one-sentence thesis;
- tentative title;
- contributions;
- main theorem;
- main figure and table;
- claim/evidence map;
- closest-work novelty table;
- limitations.

### 8. Stop/go rules

Write prospective, numerical rules for:

- stop immediately;
- iterate once;
- launch full confirmation;
- submit as a mechanistic paper without a complete repair.

## Standards

- Be adversarial and specific.
- Do not count seeds or projections as independent samples.
- Do not call inspected development tasks held-out confirmation.
- Do not infer that geometry gains cause decision gains without a controlled
  shared-planner comparison.
- Prefer falsifiable inequalities, paired contrasts, and exact controls over
  vague intuition.
- Separate “representation contains information” from “planner uses it.”
- Include realistic compute and engineering risks.
- End with one unambiguous recommended action.

---
