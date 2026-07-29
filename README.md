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
- `docs/STAGE10_FIDELITY_CONSTRAINED_PAIRWISE_MARGIN_PROTOCOL.md` — direct
  pairwise regret-certificate optimization with hard per-horizon native
  fidelity constraints.
- `docs/STAGE11_ACTION_RESPONSE_GEOMETRY_PILOT.md` — compute-gated,
  readout-free adaptation of whitened centered action-response geometry.
- `docs/RESEARCH_STATE_AFTER_STAGE11_AND_ICLR_ROADMAP.md` — audited synthesis
  of the full evidence chain, the remaining mathematical failure, and the
  proposed shared goal-metric bridge.
- `docs/ICLR_RESEARCH_DECISION_AGENT_PROMPT.md` — adversarial, copy-ready
  prompt for an independent agent to audit the work and choose the next
  ICLR-critical experiment.
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
- `notebooks/10_fidelity_constrained_pairwise_margin_adaptation.ipynb` —
  FPMA-JEPA training with three frozen physical decoders, all 45 candidate
  pairs, checkpoint rollback, and five unseen evaluation projections.
- `notebooks/11_action_response_geometry_pilot.ipynb` — low-compute ARGA
  pilot with a sequential seed gate, unseen geometry projections, fresh
  readouts, atomic checkpoints, and automatic result downloads.
- `cpu_smoke_outputs/cpu_smoke_results.json` — generated local evidence.
- `audits/` — independent post-run audits and supporting summaries.
- `results/bundles/` — original Stage 1 through Stage 3 Colab ZIP bundles.
- `results/bundles/stage11_result_bundle/` — complete extracted Stage 11 full
  result bundle, including raw CSVs, manifests, logs, plots, evaluation-only
  decoders, geometry references, and all six matched ARGA checkpoints.
- `results/stage11_full_development_audit.json` — compact independent Stage 11
  numerical audit with source hashes and corrected decision semantics.

## Reproduce the local CPU checks

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-cpu.txt
PYTHONPATH=src SDL_VIDEODRIVER=dummy .venv/bin/python -m pytest
PYTHONPATH=src SDL_VIDEODRIVER=dummy .venv/bin/python scripts/run_cpu_smoke.py
python scripts/validate_stage11_bundle.py
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

Stage 9 completed successfully and produced a narrow but genuine intervention
signal. Correctly matched action-path adaptation reduced PushT horizon-6 regret
under both a fresh physical readout and the native latent planner; the native
contrast was positive under its state-clustered interval. The result did not
transfer to early PushT horizons or Wall, and top-1 choice did not improve.
Every selected checkpoint also landed on the final epoch boundary. Stage 9
therefore shows that JEPA's action path can correct some severe long-horizon
mistakes, but its mean endpoint objective is not a reliable cross-environment
repair. The compact numerical provenance is recorded in
`results/stage9_development_audit.json`.

Stage 10 directly optimizes a deterministic pairwise-margin upper bound on
normalized planning regret. Three goal-independent physical decoders are fit
on the frozen model and then frozen; the intervention still updates only the
action encoder and six AdaLN modulation maps. It evaluates all ten candidates
and all 45 unordered pairs together, enforces a two-percent native latent
constraint separately at every horizon, makes epoch zero an eligible fallback,
and rolls back violating checkpoints. Five unseen projections with newly fit
physical readouts test whether any gain transfers beyond the training
decoders, while the original goal-latent planner is evaluated as a separate
non-harm gate. Stage 10 remains development evidence on the inspected task
family.

The Stage 10 full run returned `UNDERTRAINED_INCONCLUSIVE`, but additional
epochs alone do not explain its central failure: matched decoder-margin
training did not transfer to fresh readouts in either environment, and some
shuffled controls were competitive. Stage 11 therefore tests a
readout-independent alternative before allocating another full run. It aligns
whitened, within-state centered action-response geometry in JEPA token space,
uses no physical goal, cost, pose, or readout in the training objective, and
runs a second adaptation seed only after a necessary direct-geometry screen.
The default Stage 11 run is a compute-allocation pilot and cannot by itself
support a scientific claim.

The returned Stage 11 run used the full development matrix: 96 states per
environment, three adaptation seeds, five unseen evaluation projections, and
2,000 task-clustered bootstrap repetitions. All bundle hashes and exact
simulator restoration checks passed. Matched action-response geometry improved
over frozen and shuffled controls by point estimate at every horizon in all
five unseen projections in both PushT and Wall. The direct geometry gate
therefore passed 5/5 in each environment.

Planning transfer remained incomplete. The fresh-readout joint gate passed
only 1/5 PushT projections and 2/5 Wall projections, below the required 4/5.
Wall nevertheless showed substantial average normalized-regret reductions of
approximately 27%, 19%, and 5% at horizons 1, 3, and 6. The matched latent
fidelity constraint passed, but the PushT native planner exceeded its
horizon-6 non-harm tolerance. The notebook label
`STOP_NATIVE_FIDELITY_FAILURE` is therefore semantically misleading: latent
fidelity passed; native-planner non-harm and fresh-readout transfer failed.

Stage 11 localizes the next problem. Correct relative action geometry is
repairable, but the goal representation and planner metric do not reliably
consume it. The recommended next falsification pilot freezes ARGA and learns
one low-capacity positive-semidefinite goal metric from target latents, then
applies the identical frozen metric to frozen, latent-only, shuffled, and
matched dynamics. The full reasoning and ICLR go/no-go plan are in
`docs/RESEARCH_STATE_AFTER_STAGE11_AND_ICLR_ROADMAP.md`.
