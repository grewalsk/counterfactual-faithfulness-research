# Colab instructions

## Stage 1

Notebook: `01_model_and_environment_smoke_test.ipynb`

Stage 1 completed successfully on an A100. Its returned-bundle audit is in
`../../stage1-analysis/STAGE1_RESULT_AUDIT.md`.

## Stage 2 decisive pilot

Notebook: `02_counterfactual_faithfulness_pilot.ipynb`

1. Open the notebook in a fresh Google Colab runtime.
2. Set `RUN_MODE="full"` in the first cell.
3. Select an A100 if available. An L4 is a good alternative; a 16 GB T4 is the
   minimum supported GPU.
4. Optionally set `MOUNT_DRIVE=True` to make state shards survive a disconnect.
5. Run all nine cells in order. No runtime restart is expected.
6. Return `stage2_result_bundle.zip`.

The full run uses 250 states, 10 action sequences, horizons 1/3/6, and the
public `dino_wm_pusht` and `jepa_wm_pusht` checkpoints. It creates contact-pair
strata, real/action-blind/action-shuffled variants, representation-matched and
physical-regret analyses, clustered bootstrap intervals, and repeated
state-grouped held-out regressions.

Expected runtime:

- A100: about 30–50 minutes;
- L4: about 45–75 minutes;
- T4: about 75–120 minutes.

Allow approximately 3 GB of temporary/cache storage. The notebook downloads
two Push-T checkpoints plus the shared DINOv2 ViT-S/14 encoder; it downloads no
dataset or image decoder. Intermediate shards are excluded from the compact
result ZIP.

Expected sanity outputs:

- printed Python, CUDA, PyTorch, torchvision, and GPU versions;
- bitwise exact simulator restoration;
- 250 simulator truth shards and 250 state shards per model in full mode;
- all three pair-contact strata where physically observed;
- model, negative-control, ranking, and regret tables;
- clustered bootstrap and grouped cross-validation outputs;
- `RUN_STATUS: SUCCESS`;
- automatic download of `stage2_result_bundle.zip`.

If a simulator, model, or analysis phase fails, later cells package the failure
trace and available logs automatically.

## Stage 2B confirmatory revision

Notebook: `02b_counterfactual_faithfulness_confirmatory.ipynb`

Stage 2 completed its GPU evaluation, but its pre-specified primary result was
inconclusive because the executable candidate set was dominated by a no-op
physical oracle. Stage 2B was the confirmatory repair; its later full run
returned `NEGATIVE_SIGNAL`. These instructions are retained for reproducibility.

1. Open the Stage 2B notebook in a fresh Google Colab runtime.
2. Set `RUN_MODE="full"` in the first cell.
3. Select an A100 if available; an L4 is a good alternative and a 16 GB T4 is
   sufficient.
4. Optionally set `MOUNT_DRIVE=True` for resumable intermediate shards.
5. Run all nine cells in order. No runtime restart is expected.
6. Return `stage2b_result_bundle.zip`.

Stage 2B retains the same checkpoints and statistical decision rule while
using a fixed state-relative candidate set with non-degenerate physical costs.
Future simulator outcomes are not used to select candidates for each test
state. The notebook automatically downloads the result ZIP after either success
or a captured failure.

## Stage 2C task-aligned readout

Notebook: `02c_task_aligned_readout.ipynb`

Stage 2B completed successfully and returned `NEGATIVE_SIGNAL`: latent paired
metrics did not add held-out physical-regret validity even though the models
were action-conditioned. Stage 2C freezes both world models and tests whether a
state-disjoint physical-pose readout can convert predicted latents into useful
Push-T action costs.

1. Open the Stage 2C notebook in a fresh Google Colab runtime.
2. Set `RUN_MODE="full"` in the first cell.
3. Select an A100 if available. An L4 is a good alternative; a 16 GB T4 is
   sufficient.
4. Optionally set `MOUNT_DRIVE=True` for resumable simulator and model shards.
5. Run all nine cells in order. No runtime restart is expected.
6. Return `stage2c_result_bundle.zip`.

The full run uses 300 exact states, the same 10 fixed candidates, horizons 3
and 6, and a 50/20/30 state split for probe training, calibration, and final
testing. It evaluates raw latent distance, a primary linear pose decoder, a
secondary MLP decoder, blind/shuffled controls, and the simulator oracle.

Expected runtime:

- A100: about 20–35 minutes;
- L4: about 30–50 minutes;
- T4: about 50–80 minutes.

Allow approximately 4 GB of temporary/cache storage. The notebook downloads
the same two Push-T checkpoints and DINOv2 encoder as Stage 2B. Intermediate
feature shards are resumable and excluded from the result ZIP. Probe
checkpoints, split manifests, action predictions, metrics, plots, logs, and the
pre-specified decision are included. The ZIP downloads automatically after
success or a captured failure.

## Stage 3 full counterfactual benchmark

Notebook: `03_full_counterfactual_benchmark.ipynb`

Stage 2C returned an audited `TASK_ALIGNED_SIGNAL`, so the pre-specified gate
for the multi-environment benchmark is open.

1. Open the Stage 3 notebook in a fresh Google Colab runtime.
2. Set `RUN_MODE="full"` in the first cell.
3. Select an A100 if available. An L4 is supported; a 16 GB T4 is the minimum
   but will be substantially slower.
4. Set `MOUNT_DRIVE=True` if you want the truth/model shards to survive a
   disconnect. Drive is optional.
5. Run all ten cells in order. No runtime restart is expected.
6. Return `stage3_result_bundle.zip`.

The full run evaluates 240 exact states per environment in PushT and Wall, 10
fixed state/task-relative action sequences, horizons 1/3/6, and four public
environment-specific checkpoints: DINO-WM and JEPA-WM in each environment.
Twelve tasks per environment are split 6/2/2/2 into probe training,
calibration, held-out regression training, and untouched final testing. Three
evaluation seeds and three projection/readout seeds are included.

Expected runtime:

- A100: about 35–60 minutes;
- L4: about 55–90 minutes;
- T4: about 90–150 minutes.

Allow approximately 6 GB of temporary/cache storage, or approximately 8 GB in
Drive for a conservative resumable run. The notebook downloads four public
world-model checkpoints plus the shared DINOv2 ViT-S/14 encoder; the total
download is approximately 1.1 GB. It downloads no training dataset and no image
decoder.

Expected sanity outputs:

- printed Python, CUDA, PyTorch, torchvision, and GPU versions;
- bitwise-exact restoration reports for both PushT and Wall;
- 240 truth shards per environment and 240 state shards per checkpoint;
- valid no-op, cost-spread, and neither/one/both interaction diagnostics;
- final-test raw action, pair, and unit tables;
- clustered cross-environment planning intervals;
- a held-out ordinary-versus-counterfactual regression comparison;
- standard versus counterfactual model-ranking tables;
- one of `CROSS_ENV_TASK_ALIGNED_SIGNAL`,
  `CROSS_ENV_PLANNING_SIGNAL_ONLY`, `MIXED_GENERALIZATION`, or
  `INCONCLUSIVE`;
- automatic download of `stage3_result_bundle.zip`.

Intermediate feature shards are resumable and excluded from the result ZIP.
The ZIP contains probes, split/task manifests, exact-restore and candidate
design audits, raw tables, plots, checkpoint hashes, logs, and failure traces.

## Stage 3B audited analysis repair

Notebook: `03b_stage3_analysis_repair.ipynb`

Stage 3 confirmed the cross-environment planning signal, but its task-margin
regression contained only NaN predictions because tied-cost decision units
were not handled before standardization.

1. Open the Stage 3B notebook in Google Colab.
2. Leave `RUN_MODE="full"` in the first cell.
3. Use the same Stage 3 output directory. If Drive was used for Stage 3, set
   `MOUNT_DRIVE=True` again so the notebook reuses the existing shards.
4. Select an A100 or L4. Cached analysis is quick; a GPU is required only
   because the notebook can reconstruct missing Stage 3 intermediates.
5. Run all ten cells in order.
6. Return `stage3b_result_bundle.zip`.

Stage 3B runs all three regression specifications on one common finite sample,
excludes tied-cost no-decision units without imputation, reports exclusions,
counts both Wall collisions and door crossings as interactions, makes model
ranking NaN-safe, and includes `unit_metrics.csv` in the downloaded bundle.

If the Stage 3 truth/model intermediates are still present, the expensive
simulation and model phases are reused. If only the downloaded Stage 3 bundle
remains, a full reconstruction is required because that bundle omitted the
regression-train unit table.

## Stage 4 matched action-structure intervention

Notebook: `04_matched_action_structure_intervention.ipynb`

Stage 3B returned `CROSS_ENV_PLANNING_SIGNAL_ONLY`. Stage 4 tests the missing
mechanism by comparing two exactly magnitude-matched interventions on the
untouched Stage 3B final-test decoded poses: corruption of action-specific
residual structure and a shared common-mode pose displacement.

1. Open the Stage 4 notebook in a fresh Google Colab runtime.
2. Leave the frozen configuration unchanged.
3. A CPU runtime is sufficient; no GPU or Drive mount is required.
4. Run all eleven cells in order.
5. Return `stage4_result_bundle.zip`, which downloads automatically.

The notebook downloads the compact frozen Stage 3B result bundle from this
public repository. It does not download checkpoints or rerun a simulator. It
evaluates five pre-specified severities and five deterministic intervention
seeds, verifies exact perturbation-magnitude matching, clusters uncertainty by
the 40 final-test states per environment, and emits the frozen Stage 4
decision. Expected runtime is several minutes on a standard Colab CPU.

The completed deterministic run returned
`CROSS_ENV_ACTION_STRUCTURE_CAUSAL_SIGNAL`. Re-running the notebook is a
reproducibility check; its archived bundle and independent audit are in
`../results/bundles/` and `../audits/stage4/`.

## Stage 5 counterfactual decision-readout training

Notebook: `05_counterfactual_decision_readout_training.ipynb`

Stage 4 established that action-specific consequence structure is causally
necessary for planning under matched decoded-pose error. Stage 5 prospectively
tests a remedy: train compact physical-state decision readouts over frozen
DINO-WM and JEPA-WM predictions using an explicit same-state counterfactual
difference loss.

1. Open the Stage 5 notebook in a fresh Google Colab runtime.
2. Leave `RUN_MODE="full"` and the frozen configuration unchanged.
3. Select an A100 if available. An L4 is a good alternative; a 16 GB T4 is
   supported but substantially slower.
4. Set `MOUNT_DRIVE=True` if you want simulator/model shards to survive a
   disconnect. Drive is optional.
5. Run all eleven cells in order. No runtime restart is expected.
6. Return `stage5_result_bundle.zip`, which downloads automatically.

The full run creates twelve numerically new tasks per environment and 240
states per environment. Tasks are split 6/3/0/3 into readout training,
descriptive calibration, unused regression, and untouched final testing. It
evaluates horizons 1/3/6, ten candidates, two public model families per
environment, and three projection/training seeds.

Four same-architecture heads receive identical examples, initialization,
minibatch schedules, and optimizer updates:

- ordinary endpoint prediction;
- independent-state pair differences;
- correct same-state counterfactual differences;
- shuffled same-state pair differences.

Expected runtime:

- A100: about 50–80 minutes;
- L4: about 75–120 minutes;
- T4: about 120–200 minutes.

Allow approximately 7 GB of temporary/cache storage, or 9 GB in Drive for a
conservative resumable run. The notebook downloads four public world-model
checkpoints plus the shared encoder, approximately 1.1 GB total. Intermediate
feature shards are excluded from the result ZIP; trained readout checkpoints,
raw prediction/metric tables, clustered intervals, plots, logs, and the frozen
decision are included.

The primary gate requires lower normalized planning regret, better
margin-weighted action ranking, and an upper 95% confidence bound no greater
than 1.05 for the counterfactual/ordinary physical-pose-error ratio in both
environments. The strongest result additionally requires superiority to both
paired controls. This is a readout-level remedy over frozen world-model
predictions, not full-backbone world-model fine-tuning.

## Stage 1 historical instructions

1. Open `01_model_and_environment_smoke_test.ipynb` in a fresh Colab runtime.
2. Select a GPU runtime. A T4 with 16 GB is sufficient.
3. Optionally set `MOUNT_DRIVE=True` in the first cell. Drive is not required.
4. Run all cells in order. No runtime restart is expected.
5. Return `stage1_result_bundle.zip`.

Smoke defaults: 10 exact branch states, 3 action alternatives per state, two
world-model horizons, and five primitive Push-T actions per model step. Full
mode raises the Stage 1 pipeline check to 20 states and 4 alternatives; it is
still not the decisive Stage 2 pilot.

Expected runtime on a T4: about 15–30 minutes, including downloads. Expected
Drive storage: approximately 1 GB with caches, under 100 MB for result files.
The notebook downloads the official 275 MB DINO-WM Push-T checkpoint and the
DINOv2 ViT-S/14 encoder; no dataset and no 3.64 GB image decoder are downloaded.

Expected sanity outputs:

- printed Python, CUDA, PyTorch, torchvision, and GPU versions;
- exact restore report with bitwise-identical endpoints and initial renders;
- finite ordinary and paired metrics at both horizons;
- ranking and regret rows;
- GPU-memory reports;
- `RUN_STATUS: SUCCESS`;
- a downloadable `stage1_result_bundle.zip`.

If the run fails, the notebook still packages `FAILURE_TRACE.txt` when execution
has reached the main run cell. Return that ZIP as-is for diagnosis.
