# Statistical analysis plan for the decisive pilot

This plan is fixed before Stage 2 model outputs are examined.

## Primary question

After controlling for ordinary rollout error, does paired counterfactual error
explain additional held-out variation in executable action-selection regret?

## Fixed sampling and rows

- 250 independently sampled Push-T branch states.
- 10 fixed alternative action sequences per state.
- Horizons 1, 3, and 6, with five simulator control steps per model step.
- Two public Push-T checkpoints: `dino_wm_pusht` and `jepa_wm_pusht`.
- Preserve the state as the independent cluster. The \(\binom{A}{2}\) action
  pairs are repeated measurements, not independent observations.
- Predeclare pair-level contact strata using simulator collision events only:
  `neither` branch contacts, `one` branch contacts, or `both` branches contact.

The primary analysis row is `(initial_state, model, horizon)`. Each row contains
ordinary error, normalized paired error, effect cosine, candidate-set regret,
pairwise ranking accuracy, horizon, contact fraction, initial-state design
stratum, and ground-truth action-effect scale.

## Primary endpoint and model comparison

Primary endpoint: normalized simulator regret of the candidate selected by the
world model.

1. Split whole initial states into five folds. Every model and horizon row for
   one state remains in the same fold.
2. Base model:
   `regret ~ ordinary_error + horizon + model + effect_scale + contact_fraction + design_stratum`.
3. Full model:
   add `normalized_paired_error + (1 - paired_effect_cosine)`.
4. Repeat grouped five-fold cross-validation for 20 deterministic split seeds.
   Report held-out \(\Delta R^2\), held-out RMSE change, and the fraction of
   repeats favoring the full model.
5. For split seed zero, resample whole initial-state clusters 2,000 times and
   form a percentile interval for the out-of-fold mean-squared-error
   improvement and \(\Delta R^2\).

The notebook labels a `NONREDUNDANT_SIGNAL` only when the median repeated
held-out \(\Delta R^2\) is positive and the state-clustered 95% interval for
out-of-fold mean-squared-error improvement excludes zero on the positive side.
An interval excluding zero on the negative side with a negative median is a
`NEGATIVE_SIGNAL`; all other outcomes are `INCONCLUSIVE`.

## Secondary endpoints

- Representation-matched latent-cost normalized regret.
- Coverage-cost normalized regret.
- Pairwise action-ranking error.
- Top-1 candidate accuracy.
- Raw and normalized regret.
- Pair-level contact-stratified counterfactual error.
- Model ranking changes under ordinary versus paired metrics.
- Magnitude-only and direction-only counterfactual predictor blocks.
- Action-blind and action-shuffled negative controls.

All secondary analyses are labeled secondary. No pair-level p-values.

## Uncertainty

- 2,000 clustered bootstrap replicates, resampling initial states.
- When reporting contact strata, resample initial states while retaining all
  action-pair measurements from a sampled state.
- Report point estimates, 95% percentile intervals, sample counts, and the
  number of independent states.
- For model-rank comparisons, recompute the full ranking inside every bootstrap
  replicate.

## Multiplicity and exclusions

- One primary endpoint and one joint counterfactual predictor block; no
  multiplicity correction is applied to that preregistered comparison.
- Secondary analyses are descriptive and interval-based in this pilot.
- Exclude a state only for simulator failure, non-finite output, corrupt model
  output, or a predeclared minimum ground-truth action-effect threshold.
- Never exclude a valid high-error rollout.
- Report every exclusion and failure trace.

## Sensitivity checks

- Raw versus normalized paired error.
- Representation-matched latent regret versus executable physical regret.
- Removing low-effect candidate sets.
- Pair-level `neither`, `one`, and `both` contact strata.
- Leave-one-model-out and leave-one-horizon-out evaluation.
- Common-mode error entered alongside ordinary error.
- Robust loss or rank transform for heavy-tailed regret.
- The algebraic identity between complete-pair squared error and
  action-centered squared error.

## Decision gates

Proceed to a full benchmark only if at least one holds:

1. Paired error provides positive held-out incremental validity.
2. The paired metric changes model ranking with a stable, interpretable physical
   failure regime.
3. A well-powered negative result shows redundancy under explicitly stated
   conditions.

Otherwise revise the intervention design or stop; do not increase benchmark
breadth merely to search for significance.

## Confirmatory Stage 2B amendment

Stage 2 returned an `INCONCLUSIVE` primary result and exposed a floor-heavy
physical candidate set. Before inspecting any Stage 2B model output, the
confirmatory intervention revision is frozen in
`STAGE2_DECISION_AND_REVISION.md`.

Stage 2B retains the primary endpoint, base and full predictor blocks, grouped
five-fold protocol, 20 split seeds, 2,000 state-cluster bootstrap replicates,
and decision thresholds above. It changes only the initial-state and candidate
action design to ensure meaningful executable physical-cost variation. The
fixed state-relative action subset is frozen before model evaluation and does
not use future simulator outcomes to select candidates for each test state.
Pair-normalized contact summaries additionally apply a predeclared 1e-6
ground-truth effect-scale threshold.

## Stage 2C task-aligned readout amendment

Stage 2B returned `NEGATIVE_SIGNAL` under the confirmatory design while showing
that real action-conditioned predictions outperform blind and shuffled
controls. Before inspecting any Stage 2C model output, the task-aligned readout
protocol is frozen in `STAGE2C_TASK_ALIGNED_PROTOCOL.md`.

Stage 2C uses 300 states, horizons 3 and 6, the same fixed candidate set, and
frozen world models. Initial states are split 50/20/30 into probe training,
calibration, and untouched test sets. The primary probe is a ridge decoder of
block pose from predicted future latents; a one-hidden-layer MLP is secondary.
Hyperparameters are selected on calibration pose error only.

The primary comparison is paired by `(state, model, horizon)` against the raw
latent-goal-distance planner. Whole test states are resampled for 2,000
bootstrap replicates. A task-aligned signal requires 95% bootstrap lower bounds
above zero for both physical normalized-regret improvement and margin-weighted
pair-ranking improvement. The linear probe is primary; an MLP-only pass is
labeled separately. All other analyses are secondary.
