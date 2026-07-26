# Execution constraints: simulation and Google Colab only

This project must not require access to a physical robot.

The intended contribution is:

> A simulator-grounded, paired-intervention evaluation of counterfactual action faithfulness in action-conditioned world models.

Use exact simulator state restoration as the source of interventional ground truth. Prioritize:

- Push-T;
- MetaWorld;
- RoboCasa or RoboSuite;
- PointMaze or Wall;
- other open simulators with exact snapshot-and-restore support.

Do not require or recommend real-robot experiments as a condition for publication. Acknowledge simulation-to-reality limitations explicitly, but do not allow them to expand the project.

Do not claim that simulator results establish real-world robotic reliability. The scientifically defensible claim is about whether learned action-conditioned models reproduce controlled physical interventions and whether this predicts planning performance in executable environments.

# Compute arrangement

Assume that development, literature review, code inspection, metric design, and CPU smoke tests can be performed locally.

All substantial GPU experiments will be executed manually by the user in Google Colab. Do not attempt lengthy training or large-model evaluation locally.

When GPU execution becomes necessary:

1. Finish all code and CPU-level validation first.
2. Prepare a self-contained Colab notebook.
3. Stop and issue a clearly marked handoff:

    COLAB RUN REQUIRED

4. State exactly:
   - which notebook to run;
   - recommended GPU and minimum VRAM;
   - expected runtime;
   - expected Google Drive storage;
   - checkpoints and datasets that will be downloaded;
   - whether a free Colab GPU is sufficient;
   - which cells may require restarting the runtime;
   - expected sanity-check outputs;
   - exact output files that the user should return.

5. Do not continue as if the experiment succeeded. Wait for the user to supply the generated outputs.
6. Once outputs are returned, inspect them, diagnose failures, revise the notebook if needed, and prepare the next run.

Never fabricate experimental values or infer that an unfinished Colab run succeeded.

# Colab design requirements

Every GPU notebook must:

- run from a fresh Colab runtime;
- install pinned dependencies;
- print Python, CUDA, PyTorch, and GPU versions;
- mount Google Drive optionally rather than requiring it;
- use resumable checkpointing;
- cache downloaded models and datasets;
- set and report all random seeds;
- include a fast smoke-test mode;
- include a full-run mode controlled by one configuration cell;
- monitor GPU memory;
- save intermediate results frequently;
- export metrics as CSV or JSON;
- save logs, plots, configurations, and failure traces;
- package results into one downloadable ZIP file;
- avoid relying on state from a previous notebook.

Each notebook should begin with a single configuration block containing:

    RUN_MODE = "smoke"  # smoke or full
    OUTPUT_DIR = ...
    SEED = ...
    MODEL_NAME = ...
    ENVIRONMENT = ...
    HORIZONS = ...
    NUM_STATES = ...
    ACTIONS_PER_STATE = ...

Make the notebooks robust to Colab disconnections.

# Model tiers

Design experiments in tiers so the project does not depend on obtaining an expensive GPU.

## Tier 1: 16 GB or less

Use smaller publicly available JEPA-WM, DINO-WM, DINOv2, Push-T, MetaWorld, PointMaze, or Wall configurations.

This tier must be sufficient for the first decisive pilot.

## Tier 2: approximately 24 GB

Evaluate larger encoders or more states, actions, and horizons.

## Tier 3: 40 GB or more

Evaluate V-JEPA 2-AC or similarly large models only if the first two tiers establish that the metric is non-redundant and planning-relevant.

Do not begin with V-JEPA 2-AC simply because it is prominent. Start with models that permit rapid iteration and exact controlled evaluation.

# Required staged workflow

## Stage 0: No GPU

Complete:

- literature and novelty audit;
- simulator selection;
- public checkpoint audit;
- mathematical formulation;
- metric implementation;
- statistical plan;
- exact-state restoration tests;
- synthetic unit tests;
- CPU smoke tests;
- Colab notebook preparation.

Do not request GPU execution before these are complete.

## Stage 1: First Colab run

Prepare:

    01_model_and_environment_smoke_test.ipynb

Purpose:

- load one small world model;
- load one simulator;
- restore an identical state multiple times;
- execute two alternative action sequences;
- verify deterministic or controlled outcomes;
- compute ordinary and paired counterfactual metrics;
- save a small result bundle.

Target:

- 10–20 initial states;
- 2–4 actions per state;
- one or two horizons;
- less than approximately 30 minutes.

This run validates the pipeline, not the hypothesis.

## Stage 2: Decisive pilot

Prepare:

    02_counterfactual_faithfulness_pilot.ipynb

Target:

- 200–300 exact simulator states;
- 8–12 alternative actions or action sequences per state;
- at least three horizons;
- two or three model checkpoints;
- contact and non-contact strata;
- ordinary rollout metrics;
- paired counterfactual metrics;
- action-ranking accuracy;
- planning regret.

The decisive question is:

> After controlling for ordinary rollout error, does counterfactual error explain additional variation in action-ranking or planning performance?

Do not propose a full benchmark until this question has an empirical answer.

## Stage 3: Full evaluation

Proceed only if Stage 2 demonstrates non-redundancy or a surprising, scientifically interpretable negative result.

Prepare:

    03_full_counterfactual_benchmark.ipynb

Include:

- at least two environments;
- at least three model families or substantively different checkpoints;
- several training seeds when available;
- intervention taxonomy;
- held-out states and tasks;
- clustered bootstrap confidence intervals;
- mixed-effects or held-out regression analysis;
- model rankings under standard versus counterfactual metrics.

## Stage 4: Optional training experiment

Proceed only if the evaluation reveals a genuine diagnostic gap.

Prepare:

    04_counterfactual_training.ipynb

Compare:

- ordinary prediction training;
- paired examples treated as independent samples;
- explicit counterfactual-difference loss;
- shuffled-pair control;
- equal-data and equal-compute controls.

The important result is not lower counterfactual error alone. Test whether the intervention improves action ranking or planning success at similar ordinary rollout error.

# Publication standard without a real robot

A strong result must satisfy at least three of the following:

1. Counterfactual metrics distinguish models matched on ordinary prediction accuracy.
2. Counterfactual metrics predict simulator planning failure beyond rollout error.
3. Failures concentrate around meaningful physical regimes such as contact and grasp transitions.
4. Results reproduce across multiple simulators and model families.
5. Counterfactual training improves planning without simply improving every prediction metric.
6. The benchmark changes the ranking of world models relative to standard evaluation.
7. A theoretically motivated negative result demonstrates that counterfactual evaluation is redundant under identifiable conditions.

Do not suggest adding a robot merely to make the paper appear stronger. Strengthen causal control, cross-model breadth, statistics, and reproducibility instead.

# First GPU handoff rule

Do not ask the user to open Colab yet.

First complete Stage 0 and produce the tested notebook for Stage 1. Only then say:

    COLAB RUN REQUIRED: STAGE 1

Include one direct notebook link or file, the exact cells to run, estimated runtime, required GPU tier, and the output ZIP filename to return.