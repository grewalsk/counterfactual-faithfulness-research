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

The completed run passed every integrity condition and returned
`NO_TRAINING_FIX`. The counterfactual objective was noninferior in pose error
and produced modest planning improvements relative to endpoint-only training,
but the two co-primary planning intervals did not both exclude zero in either
environment. The independent-pair control was equally good or better, so the
effect was not counterfactual-specific.

## Stage 6 structured action-effect development

Notebook: `06_structured_action_effect_development.ipynb`

Stage 6 develops the stronger repair suggested by the Stage 5 audit. It uses a
learned projection over raw pooled future features, processes all ten candidate
actions jointly, decomposes a shared predicted endpoint from no-op-relative
action effects, decodes action descriptors from centered future features, and
adds a physical-cost-weighted ranking loss.

1. Open the Stage 6 notebook in a fresh Google Colab runtime.
2. Leave `RUN_MODE="full"` and the development configuration unchanged.
3. Select an A100. An L4 is supported but the learned 6,144-to-128 projections
   make it substantially slower; a T4 is not recommended for the full run.
4. Set `MOUNT_DRIVE=True` if you want simulator, model, and completed-adapter
   checkpoints to survive a disconnect.
5. Run all eleven cells in order.
6. Return `stage6_result_bundle.zip`, which downloads automatically.

Stage 6 deliberately reuses the now-inspected Stage 5 task family. The original
training and calibration tasks retain their roles, while the former final split
is renamed `development_holdout`. It must not be represented as an untouched
test set. Common checkpoints at epochs 80, 120, and 160 are scored on
calibration tasks only; the development holdout is evaluated afterward.

Six same-architecture training conditions isolate endpoint, action-decoding,
ranking, same-state action-effect, and independent-state action-effect
supervision. The proposed `counterfactual_action_effect` condition combines
endpoint, no-op-relative effect, action-decoding, and ranking losses.

Expected runtime:

- A100: approximately 90–150 minutes;
- L4: approximately 150–240 minutes.

Allow approximately 9 GB of temporary/cache storage. Trained adapter
checkpoints can make the downloaded result bundle substantially larger than
Stage 5. The notebook resumes completed adapter heads when Drive is enabled and
the run signature is unchanged.

A positive Stage 6 result is a development result only. It nominates one frozen
method for a later Stage 6B run on numerically new tasks.

## Stage 7 recurrent counterfactual transition adapter

Notebook: `07_recurrent_counterfactual_transition_adapter.ipynb`

Stage 6 showed that terminal action-effect readouts can improve short-horizon
development metrics without surviving reliably to horizon six. Stage 7 tests
whether the loss occurs inside JEPA-WM inference. It audits unpooled tokens
after all six AdaLN blocks and trains a small transition residual that is
inserted after every predictor call before the prediction is recycled as the
next context.

1. Open the Stage 7 notebook in a fresh Google Colab runtime.
2. Leave `RUN_MODE="full"` and all scientific settings unchanged.
3. Select an A100 with at least 40 GB. An L4 is suitable for smoke mode but is
   not recommended for the full unpooled-token run.
4. Set `MOUNT_DRIVE=True` for the full run. It is technically optional, but
   strongly recommended because the token caches are several gigabytes and
   make the run resumable after a disconnect.
5. Run all eleven cells in order. No runtime restart is expected.
6. Return `stage7_result_bundle.zip`, which downloads automatically.

The full run uses 96 exact states per environment, ten candidates, all six
intermediate rollout steps, and the public PushT and Wall JEPA-WM checkpoints.
It deliberately focuses on the AdaLN inference path before testing whether a
successful recipe transfers to DINO-WM.

Three identical recurrent residuals isolate ordinary latent correction,
independent-state delta supervision, and correct same-state counterfactual
delta supervision. The residual is zero-initialized, retains the 16×16 token
grid, and receives no planning-ranking gradient.

Expected runtime:

- A100 40 GB or 80 GB: approximately 90–180 minutes;
- L4 smoke mode: approximately 30–60 minutes.

Allow approximately 10–14 GB of temporary or Drive storage. The notebook
downloads two public world-model checkpoints plus the shared DINOv2 encoder,
approximately 650 MB total. Intermediate simulator and unpooled-token shards
are excluded from the downloaded result bundle.

Expected sanity outputs:

- Python, CUDA, PyTorch, and GPU versions;
- bitwise-exact restoration for PushT and Wall;
- confirmation of six-block `VisionTransformerAdaLN` predictors;
- 96 transition-token shards per environment in full mode;
- finite layerwise physical-effect audit rows;
- equal initialization hashes across all adapter conditions;
- selected calibration checkpoints for two adapter seeds;
- recurrent metrics and clustered contrasts on the development holdout only;
- one of `RECURRENT_COUNTERFACTUAL_CANDIDATE_READY`,
  `RECURRENT_GAIN_NOT_SPECIFIC`, `MIXED_RECURRENT_SIGNAL`,
  `NO_RECURRENT_DEVELOPMENT_GAIN`, or `INCONCLUSIVE`;
- `RUN_STATUS: SUCCESS`;
- automatic download of `stage7_result_bundle.zip`.

Stage 7 reuses inspected tasks and is exploratory. Even a positive result must
be frozen and tested later on numerically new tasks.

## Stage 8 counterfactual decision energy

Notebook: `08_counterfactual_decision_energy.ipynb`

Stage 7 improved ordinary latent prediction without reliably improving
planning. Stage 8 keeps every JEPA-WM prediction frozen and instead trains a
small set-centered energy residual to order the ten candidate actions. The
proposed head combines native/final goal features with correctly aligned
no-op-relative features from all audited predictor layers.

1. Open the Stage 8 notebook in the same active Colab runtime when possible.
2. Leave `RUN_MODE="full"` and all scientific settings unchanged.
3. Select an A100 with at least 40 GB.
4. Keep `REUSE_STAGE7_CACHE=True`. If the compatible Stage 7 cache remains at
   `/content/counterfactual_faithfulness_stage7`, the notebook reuses its
   simulator and transition-token shards automatically.
5. Run all eleven cells in order. A missing or incompatible Stage 7 cache
   triggers a complete self-contained reconstruction.
6. Return `stage8_result_bundle.zip`, which downloads automatically.

The full run uses 96 states per environment, ten candidate actions, horizons
1, 3, and 6, two head seeds, and four identically optimized energy heads:
final-token, action-prior, wrong-state, and correctly aligned counterfactual
energy. The public encoder and predictor are frozen and receive no gradient.

Expected runtime when the complete Stage 7 cache is reused:

- A100 40 GB or 80 GB: approximately 20–45 minutes.

Without the cache, allow approximately the Stage 7 reconstruction time plus
20–45 minutes for Stage 8 training and evaluation. The large simulator and
token shards are excluded from the downloaded bundle.

Expected sanity outputs:

- printed cache compatibility/reuse decision;
- bitwise-exact restoration for PushT and Wall;
- confirmation of six-block `VisionTransformerAdaLN` predictors;
- aligned feature tensors with three horizons, ten actions, and a fixed feature
  manifest;
- equal zero-initialization hashes across all four trained conditions;
- calibration-only checkpoint selection for two head seeds;
- development metrics, state-clustered intervals, and task-level descriptive
  contrasts;
- one of `DECISION_ENERGY_CANDIDATE_READY`,
  `DECISION_ENERGY_GAIN_NOT_SPECIFIC`, `MIXED_DECISION_ENERGY_SIGNAL`,
  `NO_DECISION_ENERGY_GAIN`, or `INCONCLUSIVE`;
- `RUN_STATUS: SUCCESS`;
- automatic download of `stage8_result_bundle.zip`.

Stage 8 reuses inspected tasks and is exploratory. Any successful method must
be frozen before a later run on numerically new tasks.

## Stage 9 counterfactual value-equivalent AdaLN adaptation

Notebook: `09_counterfactual_value_equivalent_adaln.ipynb`

Stage 9 tests the first repair that changes JEPA-WM's transition function.
The visual encoder, predictor attention/MLP content weights, proprio pathway,
and output projection remain frozen. Only the action encoder and the six AdaLN
modulation linear maps receive gradients.

The matched method combines the original future-latent anchor with a compact
goal-independent physical endpoint loss, an explicit same-state no-op-relative
physical-effect loss, and a latent-displacement action decoder. Two controls
update the identical parameters using latent prediction alone or shuffled
within-state action/outcome correspondence. Temporary training heads are
discarded before evaluation, the training projection is replaced by an
independently seeded projection, and every predictor receives the same newly
fitted linear physical-state readout.

1. Open the Stage 9 notebook in the same active Colab runtime when possible.
2. Leave `RUN_MODE="full"` and the scientific settings unchanged.
3. Select an A100 with at least 40 GB. Backpropagation through recurrent,
   unpooled-token rollouts makes an A100 substantially faster and safer than an
   L4; a T4 is intended only for smoke mode.
4. Keep `REUSE_STAGE7_CACHE=True`. A compatible Stage 7 truth/transition cache
   is reused automatically; otherwise the notebook reconstructs it.
5. Run all eleven cells in order.
6. Return `stage9_result_bundle.zip`, which downloads automatically on success
   or captured failure.

The full run uses 96 exact states per environment, ten candidates, all six
rollout steps, and calibration-only checkpoint selection at epochs 4, 8, and
12. Candidate actions are split into null-anchored groups during training to
bound activation memory. Evaluation uses the development holdout at horizons
1, 3, and 6.

Expected runtime with a complete Stage 7 cache:

- A100 80 GB: approximately 60–120 minutes;
- A100 40 GB: approximately 90–150 minutes;
- L4 smoke mode: approximately 25–50 minutes.

Without the Stage 7 cache, add the simulator/transition reconstruction time.
The downloaded ZIP excludes the large intermediate token shards but includes
adapted action-path checkpoints, fresh readout selections, raw development
metrics, clustered contrasts, plots, logs, and the decision gate.

Stage 9 is exploratory method development. Its success condition requires the
correctly matched method to beat the frozen readout, latent-only adaptation,
and shuffled-correspondence control in both environments at multiple horizons.
Only then should the recipe be frozen for a numerically new, multi-seed task
family.

## Stage 10 fidelity-constrained pairwise margin adaptation

Notebook: `10_fidelity_constrained_pairwise_margin_adaptation.ipynb`

Stage 10 follows the narrow Stage 9 PushT horizon-6 signal with a
decision-aligned objective. Three goal-independent physical-state decoders are
fit on the frozen JEPA rollout and frozen before adaptation. The notebook then
updates the same action encoder and six AdaLN modulation maps using all ten
candidates and all 45 unordered action pairs at once.

The FPMA loss minimizes a sum p-norm upper bound on normalized planning regret
and a non-saturating gap-normalized top-1 term. A per-horizon augmented
Lagrangian encourages native latent fidelity, while calibration checkpoint
eligibility independently enforces a two-percent limit at horizons 1, 3, and
6. Epoch zero is eligible, so an unsafe update cannot displace the pretrained
model. Violations restore both parameters and optimizer momentum and halve the
learning rate.

1. Open the Stage 10 notebook in a fresh or cache-compatible Colab runtime.
2. Leave `RUN_MODE="full"` and all scientific settings unchanged.
3. Select an A100 80 GB if available; an A100 40 GB is the minimum intended
   full-run GPU. L4 and T4 runtimes are for smoke mode only.
4. Leave `MOUNT_DRIVE=True` for the full run. This is the default because a
   boundary extension can outlive a single Colab lease.
5. Keep `REUSE_STAGE7_CACHE=True`. A compatible Stage 7 cache avoids repeating
   simulator and frozen-transition reconstruction.
6. Run all eleven cells in order.
7. Return `stage10_result_bundle.zip`, which downloads automatically after
   either a completed run or a captured failure.

The full run uses 96 states per environment, three adaptation seeds, four
trained conditions, three frozen training decoders, five unseen evaluation
projections, and checkpoints `0,2,...,24`. A prospectively defined improving
final-boundary rule can extend an individual treatment to epoch 36 and then
48. Atomic latest checkpoints resume an interrupted treatment from its most
recent calibration boundary; completed treatments are reused only when their
base model, frozen decoders, cache-content digest, pinned pretrained assets,
and full configuration signature match. A reduced smoke matrix always returns
`NONPROTOCOL_RUN_NO_SCIENTIFIC_DECISION`; it cannot trigger the advancement
gate.

Expected runtime with a complete Stage 7 cache:

- A100 80 GB: approximately 6–10 hours before rare boundary extensions;
- A100 40 GB: approximately 8–12 hours before rare boundary extensions;
- A100 smoke mode: approximately 30–75 minutes.

The bundle excludes large transition/rollout caches and completed control
checkpoints. If a treatment fails, its atomic latest checkpoint is included so
the failure bundle remains resumable.
It includes the selected matched FPMA checkpoints, frozen-decoder manifests,
all training and checkpoint histories, fresh-projection readout selections,
task-clustered metrics and contrasts, deterministic certificate checks, plots,
native goal-latent planner metrics, the prospective decision gate, logs, and
checksums. All truth/transition/goal shards and task/split definitions are
SHA256-bound before training, and the exact JEPA-WM and DINOv2 asset hashes are
verified after model load.

Stage 10 is still development evidence because the task family has informed
the method. Advancement requires the matched constrained method to improve
regret and ranking over frozen, latent-only, and shuffled controls at at least
two common, task-interval-supported horizons in at least four of five unseen
projections in both environments, while satisfying every native-fidelity
constraint, preserving the native latent planner, and avoiding candidate
collapse in every seed/projection/horizon stratum. Passing freezes the recipe
for a numerically new task-family run.

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
