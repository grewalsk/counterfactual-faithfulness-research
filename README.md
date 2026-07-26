# Counterfactual action faithfulness — Stage 0

This package implements and tests a simulator-only protocol for asking:

> From an identical simulator state, does an action-conditioned world model
> reproduce the *difference* caused by alternative executable actions, and does
> that error predict action-ranking or planning regret beyond ordinary rollout
> error?

No physical robot is required or implied. Results can support claims about
controlled interventions in executable simulators, not real-world robotic
reliability.

This private research repository contains the complete staged notebook
history, frozen protocols, tested CPU utilities, independent result audits, and
the original downloaded Colab result bundles. The public Colab repository is a
smaller execution-facing mirror and is not the evidence archive.

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

The current boundary is Stage 3B. Run
`03b_stage3_analysis_repair.ipynb` in Google Colab and return
`stage3b_result_bundle.zip`. It preserves the Stage 3 protocol, applies a
common finite-row sample to every held-out regression specification, counts
Wall door crossings as interactions, makes model rankings NaN-safe, and
exports the previously omitted unit-level table.
