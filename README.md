# Counterfactual action faithfulness — Stage 0

This package implements and tests a simulator-only protocol for asking:

> From an identical simulator state, does an action-conditioned world model
> reproduce the *difference* caused by alternative executable actions, and does
> that error predict action-ranking or planning regret beyond ordinary rollout
> error?

No physical robot is required or implied. Results can support claims about
controlled interventions in executable simulators, not real-world robotic
reliability.

This research repository contains the staged notebook history, frozen
protocols, tested CPU utilities, independent result audits, and selected
downloaded Colab result bundles.

## Contents

- `docs/STAGE0_REPORT.md` — literature/novelty, simulator and checkpoint audit,
  conclusions, risks, and go/no-go gates.
- `docs/CHECKPOINT_AUDIT.md` — public weight sizes, tiers, downloads, and
  licensing note.
- `docs/MATHEMATICAL_SPEC.md` — estimands and metric definitions.
- `docs/STATISTICAL_ANALYSIS_PLAN.md` — pre-registered Stage 2 analysis.
- `docs/STAGE2_DECISION_AND_REVISION.md` — audited Stage 2 decision and the
  frozen confirmatory intervention revision.
- `docs/STAGE2C_TASK_ALIGNED_PROTOCOL.md` — pre-run task-aligned readout
  hypothesis, split, metrics, and decision gate.
- `docs/STAGE3_BENCHMARK_PROTOCOL.md` — cross-environment held-out benchmark,
  regression, integrity checks, and decision rule.
- `docs/STAGE3B_ANALYSIS_REPAIR.md` — finite-row regression repair, Wall
  interaction correction, and NaN-safe ranking rules.
- `docs/STAGE4_MATCHED_INTERVENTION_PROTOCOL.md` — pre-specified matched-error
  causal test of action-specific decoded prediction structure.
- `docs/STAGE5_COUNTERFACTUAL_TRAINING_PROTOCOL.md` — prospective equal-data,
  equal-update training test of a counterfactual decision readout on new tasks.
- `docs/STAGE6_ACTION_EFFECT_DEVELOPMENT_PROTOCOL.md` — exploratory,
  set-aware action-effect adapter development after the Stage 5 result.
- `docs/STAGE7_RECURRENT_TRANSITION_PROTOCOL.md` — inference-path audit and
  recurrent, unpooled-token transition-adapter development protocol.
- `docs/STAGE8_COUNTERFACTUAL_ENERGY_PROTOCOL.md` — frozen-dynamics,
  task-disjoint calibration of decision energy with action-prior and
  wrong-state controls.
- `docs/STAGE9_COUNTERFACTUAL_VALUE_EQUIVALENCE_PROTOCOL.md` —
  counterfactual, goal-independent adaptation of JEPA-WM's actual AdaLN
  action-conditioning pathway.
- `src/cf_faithfulness/` — tested NumPy metric and grouped-analysis code.
- `tests/` — synthetic and exact-state restoration tests.
- `notebooks/01_model_and_environment_smoke_test.ipynb` — Stage 1 Colab.
- `notebooks/02_counterfactual_faithfulness_pilot.ipynb` — decisive Stage 2
  Colab pilot.
- `notebooks/02b_counterfactual_faithfulness_confirmatory.ipynb` — CPU-tested
  confirmatory Stage 2 intervention revision.
- `notebooks/02c_task_aligned_readout.ipynb` — frozen-world-model task-aligned
  readout diagnosis.
- `notebooks/03_full_counterfactual_benchmark.ipynb` — full PushT/Wall
  counterfactual benchmark.
- `notebooks/03b_stage3_analysis_repair.ipynb` — audited Stage 3 analysis
  repair with automatic result download.
- `notebooks/04_matched_action_structure_intervention.ipynb` — CPU-only
  matched decoded-pose intervention with automatic source and result download.
- `notebooks/05_counterfactual_decision_readout_training.ipynb` — prospective
  PushT/Wall counterfactual readout-training experiment with new final tasks.
- `notebooks/06_structured_action_effect_development.ipynb` — exploratory
  learned-projection, action-effect, and decision-ranking adapter development.
- `notebooks/07_recurrent_counterfactual_transition_adapter.ipynb` —
  exploratory AdaLN-layer audit and recurrent transition-adapter experiment.
- `notebooks/08_counterfactual_decision_energy.ipynb` — exploratory
  counterfactual decision-energy calibration with frozen JEPA-WM rollouts.
- `notebooks/09_counterfactual_value_equivalent_adaln.ipynb` — exploratory
  value-equivalent fine-tuning of the JEPA-WM action encoder and AdaLN
  modulation maps, with latent-only and shuffled-outcome controls.
- `cpu_smoke_outputs/cpu_smoke_results.json` — generated local evidence.
- `audits/` — independent post-run audits and supporting summaries.
- `results/bundles/` — original Stage 1 through Stage 3 Colab ZIP bundles.

## Reproduce the local CPU checks

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-cpu.txt
PYTHONPATH=src SDL_VIDEODRIVER=dummy .venv/bin/python -m pytest
PYTHONPATH=src SDL_VIDEODRIVER=dummy .venv/bin/python scripts/run_cpu_smoke.py
```

The `pymunk==6.8.0` pin is essential. `gym-pusht==0.1.6` declares a permissive
Pymunk dependency, but its `add_collision_handler` call is incompatible with
Pymunk 7.

## Current stage boundary

Stages 0 through 3 are complete. Stage 2B returned `NEGATIVE_SIGNAL` for raw
latent counterfactual metrics. Stage 2C returned a fully audited
`TASK_ALIGNED_SIGNAL`. Stage 3 then confirmed a cross-environment planning
advantage in PushT and Wall, while its task-margin regression was invalidated
by unhandled no-decision rows with undefined normalized margins.

Stage 3B completed successfully and returned
`CROSS_ENV_PLANNING_SIGNAL_ONLY`: the linear physical-state readout improved
planning in both environments, while task-margin error did not improve
held-out prediction of natural planning regret.

Stage 4 then returned `CROSS_ENV_ACTION_STRUCTURE_CAUSAL_SIGNAL`. Under exactly
matched decoded-pose perturbation magnitude, corrupting action-specific
consequence structure caused greater regret and ranking damage than
common-mode corruption in both environments. The independent audit reproduced
all primary intervals and found the primary direction in all 12 descriptive
environment-by-model-by-horizon cells.

Stage 5 completed with valid integrity checks and returned `NO_TRAINING_FIX`.
The counterfactual readout preserved noninferior ordinary pose prediction and
modestly improved selected planning metrics over endpoint-only training, but it
did not pass both co-primary planning gates in either environment. More
importantly, the independent-pair control was equally good or better, so the
gain was not specific to correct same-state counterfactual supervision.

Stage 6 completed as explicitly exploratory development and returned
`NO_DEVELOPMENT_GAIN`. Its terminal set-aware readout did not beat the
endpoint-only baseline or preserve pose-error noninferiority in either
environment. This rules out that particular post-rollout repair, not
counterfactual dynamics learning in general.

Stage 7 completed and returned `NO_RECURRENT_DEVELOPMENT_GAIN`. Its recurrent
counterfactual residual improved ordinary latent prediction on every
development state in both environments, but did not pass the planning gate in
either environment. Wall was especially diagnostic: ordinary latent error
improved by about 12 percent even though regret and weighted pairwise accuracy
worsened, while physical action effects were strongly decodable from internal
predictor layers.

Stage 8 returned `NO_DECISION_ENERGY_GAIN`. Wall weighted pairwise accuracy
improved, but regret remained inconclusive and a wrong-state control nearly
matched the learned energies. Margin calibration also degraded sharply. The
result says that decision information is present but a direct high-capacity
energy head can exploit task/candidate regularities without repairing
state-specific counterfactual dynamics.

Stage 9 therefore changes the transition model for the first time. It freezes
the visual encoder and all state/content weights, then fine-tunes only the
action encoder and six AdaLN modulation maps. Its same-state loss aligns
no-op-relative predicted latent effects with goal-independent physical outcome
effects while retaining the native latent target as an anchor. Temporary
training heads are discarded; identical fresh linear readouts evaluate frozen,
latent-only, shuffled-correspondence, and correctly matched predictors. Stage 9
is exploratory on the inspected task family and has not yet been run.
