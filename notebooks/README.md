# Colab instructions

## Stage 37.1 horizon-matched true-state operator calibration

Notebook: `37_1_horizon_matched_operator_calibration.ipynb`

Stage 37.1 is a simulator-only calibration prompted by Stage 37's failed
length-9--12 true-state control. All four splits now use disjoint words of
lengths 9--12. The selected semigroup/free-running operator is evaluated
against persistence, a same-initialization zero-semigroup control, and a
one-step-only control before any new JEPA panel is authorized. It cannot emit a
JEPA, planning, representation, or causal claim.

Follow `../STAGE37_1_RUN_GUIDE.md`. No prior result directory or checkpoint
token is required.

## Stage 37 semigroup-regularized PSCD and planning value

Notebook: `37_semigroup_pscd_planning_value.ipynb`

Stage 37 freezes the selected Stage 36 adapter architecture, adds explicit
direct-versus-composed training from every eligible history anchor, and tests
it against an initialization- and capacity-matched zero-semigroup control. A
capacity-escalated neural simulator control must pass before JEPA is loaded.
Locked evaluation uses fresh closure words and a leakage-free twelve-candidate
open-loop planning bank. A pass remains a post-hoc one-environment repair
result, not native JEPA closure or closed-loop planning evidence.

Protocol v2 moves the unchanged deterministic seed helper before the simulator
preflight after v1 stopped before its first control fit. No v1 model outcome or
locked statistic was observed.

Follow `../STAGE37_RUN_GUIDE.md`. No prior result directory is required.

## Stage 36 predictive-state closure distillation

Notebook: `36_predictive_state_closure_distillation.ipynb`

Stage 36 keeps JEPA-WM frozen and replaces the failed Stage 35 semantic-guard
recursion with a learned finite-history predictive state. It selects carrier
width, history, latent width, and a label-free transition family on unseen
words of lengths five through eight, freezes one final adapter and matched
controls, then evaluates fresh words of lengths nine through twelve. A pass is
adapter-distillation evidence only; it is not evidence that the original JEPA
carrier was already closed.

Follow `../STAGE36_RUN_GUIDE.md`. No prior result directory is required.

## Stage 35 JEPA hybrid predictive composition and closure

Notebook: `35_hybrid_predictive_composition_closure.ipynb`

Stage 35 uses completely fresh trajectory families and saves the official
JEPA block-four carrier at every action prefix. It selects local guard-aware
operators on short compositions, freezes calibration models and matched
permuted/time-shifted controls, then recursively evaluates unseen words of
lengths five through eight. A simulator recursion is a mandatory positive
control, and the primary evaluation predicts its guard without simulator mode
labels. A pass is observational distributed-closure evidence only.

Follow `../STAGE35_RUN_GUIDE.md`. No prior result directory is required.

## Stage 34.3 regime-aware JEPA innovation diagnostic

Notebook: `34_3_regime_innovation_diagnostic.ipynb`

Stage 34.3 is a CPU-only response to the valid Stage 34.2 JEPA insufficiency
result. It freezes rank four/five, universal/physical-mode dynamics, zero to
three carrier innovations, and ridge selection before reading evaluation. The
locked candidate must survive full-carrier, coordinate-deletion, and
permuted-mode controls. It cannot emit a causal or confirmatory claim.

Follow `../STAGE34_3_RUN_GUIDE.md`. The complete Stage 34 and Stage 34.2 Drive
directories are required.

## Stage 34.2 split-path predictive and causal continuation

Notebook: `34_2_split_path_continuation.ipynb`

Stage 34.2 keeps the Stage 34.1 model asymmetry explicit. DINO receives a
calibration-only diagonal scale/bias diagnosis, while JEPA alone advances to
the previously unopened predictive-sufficiency gate and, conditionally, the
native block-4 causal-use gate. GPU inference is skipped unless sufficiency
passes, and every causal pair is independently resumable.

Follow `../STAGE34_2_RUN_GUIDE.md`. The complete Stage 34 and Stage 34.1 Drive
directories are required.

## Stage 34.1 leakage-free action-specificity repair

Notebook: `34_1_action_specificity_repair.ipynb`

Stage 34.1 preserves the original Stage 34 result and re-scores its exact
frozen calibration/evaluation shards after reshaping the action response into
one 11-dimensional row per word prefix. The state-only comparator is
executable-proofed to make identical predictions for different words at the
same state, length, and prefix step, eliminating the action-indexed target
column leak. No model inference or GPU is required.

Follow `../STAGE34_1_RUN_GUIDE.md`. The full Stage 34 Drive run is required;
the compact downloaded bundle does not contain the raw shards.

## Stage 34 predictive-fiber causal abstraction

Notebook: `34_predictive_fiber_causal_abstraction.ipynb`

Stage 34 does not retry the direct JEPA-to-DINO map rejected by Stage 33. The
simulator alone defines a finite no-op-corrected action-response chart; the two
official checkpoints are then tested separately for unseen-action specificity,
predictive sufficiency, on-manifold causal use, and commutativity with one
frozen high-level physical transition. The gates are sequential and planning
is deferred.

Follow `../STAGE34_RUN_GUIDE.md`. Prefer a G4 Blackwell runtime; L4 is supported.
The run is source-bound and hash-resumable, requires no Stage 34 secret, and
returns `stage34_pfca_result_bundle_<signature>.zip`.

## Stage 33 bounded interventional predictive causal abstraction

Notebook: `33_bounded_interventional_predictive_causal_abstraction.ipynb`

Stage 33 starts from fresh, trajectory-disjoint PushT construction,
model-selection, calibration, and locked evaluation panels. It tests a bounded claim: whether
the official JEPA-WM and DINO-WM checkpoints admit low-effective-rank,
mode-conditioned predictive realizations connected by one calibration-only
map, and whether that map transports real recurrent internal interventions on
unseen action compositions without degrading physical planning value. The
design does not import the Stage 31 or Stage 32 bases; construction fits only
decoders, charts, and carrier coordinates, model selection locks rank and
operator choices, and calibration fits operators and the cross-model map.

Follow `../STAGE33_RUN_GUIDE.md`. The pilot is resumable through source-bound,
hash-validated checkpoints and shards. No Stage 33 secret is required; a
public-checkpoint token is optional.

Protocol v3 retains v2's model-free repair of the v1 15/16 coverage failure.
The v2 run then stopped before any scientific statistic because the official
feature-conditioned proprio output is a 256-patch latent field rather than a
short physical-state vector. V3 freezes spatial mean pooling into the native
16-channel JEPA or 20-channel DINO feature, pads that vector to 64 coordinates,
and runs a real shape/finiteness preflight for both checkpoints before fitting.
No v1/v2 scientific outcome was observed or reused.

## Stage 32 powered bounded cross-model confirmation

Notebook: `32_powered_bounded_cross_model_confirmation.ipynb`

Stage 32 resolves the Stage 31 near-miss without reusing its evaluation states
or tuning its model-specific rank-128 bases. It selects 160 new states that
maintain contact across all 72 branches from ±20°, ±30°, and ±40° action
families. Grounded closure is represented only by a cosine bounded to [-1, 1]
and is excluded below a frozen target-energy floor. The primary DINO-minus-JEPA
gate must improve grouped held-out prediction, remain positive in every action
family, and beat shuffled plus empirical-span random-subspace placebos.

Follow `../STAGE32_RUN_GUIDE.md`. No Stage 32 secrets are required. Keep the
complete source-bound Stage 31 Drive directory because the compact download
does not include its large basis arrays.

## Stage 31 cross-model grounded causal closure certificate

Notebook: `31_cross_model_grounded_closure_certificate.ipynb`

Stage 31 compares the official JEPA-WM and DINO-WM PushT checkpoints on the
same 120 fresh physical counterfactual states. Each model receives its own
construction-only layer screen and rank-128 output-aligned subspace; both are
frozen before evaluation. The primary paired gate asks whether the difference
in physically grounded closure improves held-out prediction of DINO-minus-JEPA
physical planning regret beyond differences in ordinary error and
self-consistent causal closure. The planner score is the exact official visual
MSE plus 0.1 proprio MSE objective.

Follow `../STAGE31_RUN_GUIDE.md`. No Stage 31 secrets or prior Drive artifacts
are required; a public-checkpoint token is optional.

## Stage 30 grounded causal planning value

Notebook: `30_grounded_causal_planning_value.ipynb`

Stage 30 uses 120 fresh contact-stratified PushT states to test whether
grounded causal closure predicts native terminal planning regret and the
physical planning-value loss caused by ablating the frozen Stage 18 carrier.
Closure is measured on interior schedules while planning goals use disjoint
extreme schedules. No learned decoder, reader, or subspace refit is used.

Follow `../STAGE30_RUN_GUIDE.md`. No Stage 30 secrets are required. Keep the
complete successful Stage 18 and exact Stage 29 Drive directories.

## Stage 29 grounded causal closure

Notebook: `29_grounded_causal_closure.ipynb`

Stage 29 reuses the exact 36 source-bound Stage 28 states and simulator
endpoint images. It compares each predicted future directly with the frozen
encoder representation of that exact future in the model's full native token
space. The frozen rank-128 intervention is then scored against both the
model's own opposite-schedule prediction and the encoded opposite-schedule
simulator future, separating self-consistent causal steering from grounded
causal closure. No new reader or subspace is fit.

Follow `../STAGE29_RUN_GUIDE.md`. No Stage 29 secrets are required; a fresh
nonce and exact source commit are resolved automatically. Keep the complete
successful Stage 18 directory and the complete source-bound Stage 28 Drive
directory. The compact downloaded Stage 28 zip does not contain the endpoint
image shards required by this experiment.

## Stage 26 contact-frame causal transport

Notebook: `26_contact_frame_causal_transport.ipynb`

Stage 26 tests whether contact computation occupies a spatial causal response
fiber that moves with the contact point and normal. Construction-only data
select one predictor block and freeze a rank-at-most-four canonical fiber.
Sealed evaluation transports natural low-impulse donor coordinates into the
recipient contact frame and compares them with world-axis, donor-location,
random-local, reverse-sign, and full-local-swap controls. It uses finite
forwards only and exact ordinary-versus-collision-disabled simulator targets.

Follow `../STAGE26_RUN_GUIDE.md`. A pilot requires
`STAGE26_RUN_MODE=pilot`, `STAGE26_SOURCE_COMMIT=<full 40-hex commit>`, and a
new `STAGE26_RUN_NONCE`. Keep the complete successful Stage 25 Drive directory.

## Stage 25 causal KKT tomography

Notebook: `25_causal_kkt_tomography.ipynb`

Stage 25 tests whether the contact-aligned variable found earlier is the active
set of a richer, causally used latent contact-impulse computation. It creates
ordinary and agent–block-collision-disabled PushT counterfactuals from the same
dynamic state, records exact Pymunk contact impulses, freezes a
construction-only two-coordinate impulse readout, and erases those coordinates
on held-out states while protecting the eight Stage 23 mode coordinates.

Follow `../STAGE25_RUN_GUIDE.md`. A pilot requires
`STAGE25_RUN_MODE=pilot`, `STAGE25_SOURCE_COMMIT=<full 40-hex commit>`, and a
new `STAGE25_RUN_NONCE`. Keep the complete source-bound Stage 24 Drive run at
the path specified in the guide.

## Stage 22 latent hybrid gate interaction

Notebook: `22_latent_hybrid_gate_interaction.ipynb`

Stage 22 asks whether JEPA-WM contains a genuine event-gated hybrid mechanism,
not merely a predictive action subspace. It discovers a two-mode split using
construction activations and predicted consequences only, freezes a gate
direction and an orthogonal rank-32 effect subspace, and then runs an exact
held-out 2-by-2 gate/effect intervention. Simulator contact is withheld until
the discovery, block choice, and evaluation pairs have been frozen.

Follow `../STAGE22_RUN_GUIDE.md`. A pilot requires
`STAGE22_RUN_MODE=pilot`, `STAGE22_SOURCE_COMMIT=<full 40-hex commit>`, and a
new `STAGE22_RUN_NONCE`. Stage 22 is standalone and does not import prior-stage
artifacts.

## Stage 21 coherent interface and held-out utility

Notebook: `21_coherent_interface_and_heldout_utility.ipynb`

Stage 21 first checks whether complete candidate swaps become exact after the
last action-conditioned predictor block. It then fits equal-budget,
goal-independent pose-error corrections from the frozen learned, shuffled, and
random rank-128 coordinates. Evaluation score vectors and selected actions are
frozen before evaluation endpoint truth is opened.

Follow `../STAGE21_RUN_GUIDE.md`. A pilot requires
`STAGE21_RUN_MODE=pilot`, `STAGE21_SOURCE_COMMIT=<full 40-hex commit>`, and a
new `STAGE21_RUN_NONCE`. Keep the complete successful Stage 18, Stage 19, and
Stage 20 Drive directories.

## Stage 20 frozen-subspace causal planner steering

Notebook: `20_causal_planner_steering.ipynb`

Stage 20 binds the exact successful Stage 18 subspace and Stage 19 decision,
then tests whether targeted internal edits causally move predicted scores,
near-frontier action ranks, and the model's numerical action choice. It uses
two fresh transferred action families and no visual evaluation, human scoring,
training, or subspace refitting.

Follow `../STAGE20_RUN_GUIDE.md`. A pilot requires
`STAGE20_RUN_MODE=pilot`, `STAGE20_SOURCE_COMMIT=<full 40-hex commit>`, and a
new `STAGE20_RUN_NONCE`. Keep both successful prior run directories and their
provenance files in Drive.

## Stage 19 frozen-subspace unseen-action transfer

Notebook: `19_unseen_action_family_transfer.ipynb`

Stage 19 imports the exact successful Stage 18 block-4 artifact by SHA-256 and
tests it without refitting on interleaved directions, two unseen magnitudes,
and two equal-impulse temporal profiles. Each family receives model-blind
physical selection and separate sufficiency/necessity gates with frozen
random, shuffled, wrong-state, common-mode, dose, and rank controls.

Follow `../STAGE19_RUN_GUIDE.md`. A pilot requires
`STAGE19_RUN_MODE=pilot`, `STAGE19_SOURCE_COMMIT=<full 40-hex commit>`, and a
new `STAGE19_RUN_NONCE`. The successful Stage 18 raw subspace must remain in
Google Drive at the exact expected hash.

## Stage 18 rank-64 action-contrast confirmation

Notebook: `18_rank64_action_contrast_confirmation.ipynb`

Stage 18 freezes the Stage 17 block-4/rank-64 hypothesis and tests both
sufficiency (donor-action contrast transfer) and necessity (selective removal
of native action-dependent output energy). It uses 24 construction and 32
held-out evaluation trajectories, model-blind simulator eligibility, nested
rank sensitivities, shuffled-fit and empirical-span controls, and fresh-run
provenance. It uses no Jacobians.

Follow `../STAGE18_RUN_GUIDE.md`. A confirmatory run requires Colab secrets
`STAGE18_RUN_MODE=pilot`, `STAGE18_SOURCE_COMMIT=<full 40-hex commit>`, and a
new `STAGE18_RUN_NONCE`.

## Stage 17 finite action-contrast causal interchange

Notebook: `17_finite_action_contrast_interchange.ipynb`

Stage 17 tests whether a proper, construction-fitted part of the internal
same-state action-contrast geometry causally transfers donor-specific predicted
consequences. It uses finite activation interchanges only—no Jacobians—and
scores held-out donor transfer with an independent output sketch. The complete
activation swap is an on-manifold positive control, while shuffled-fit,
empirical-span random, wrong-state, common-mode, negative-dose, and compression
controls determine whether a smaller subspace is specifically causal.

Follow `../STAGE17_RUN_GUIDE.md`. The full run requires Colab secrets
`STAGE17_RUN_MODE=pilot` and `STAGE17_SOURCE_COMMIT=<full 40-hex commit>`.

## Stage 15 fixed-reader longitudinal bundle pilot

Notebook: `15_longitudinal_predictive_control_bundle.ipynb`

Stage 15 tests whether the state-conditioned spatial field found offline after
Stage 14 transports along exact PushT trajectories and supports causal physical
prediction control. It freezes construction-only agent/block pose readers and
one common action basis before opening evaluation trajectories, restores full
agent/block dynamic state, extracts all six predictor blocks at five times per
trajectory, and compares neighboring transported modes with support-matched,
covariance-shaped, and time-shuffled controls.

Follow `../STAGE15_RUN_GUIDE.md`. The local validation and 7,960-step physics
smoke are already complete; do not spend a Colab run on smoke mode.

## Stage 13 compute-minimal JOW screen

Notebook: `13_jacobian_outcome_workspace_screen.ipynb`

1. Open the notebook in a fresh Colab GPU runtime.
2. Leave `RUN_MODE="screen"` and `MOUNT_DRIVE=True`.
3. Run all cells in order.
4. Review the measured ETA printed by the integrity benchmark.
5. Return `stage13_jow_result_bundle.zip`.

The default screen performs no training. It reconstructs 8 construction and 4
calibration PushT states, uses horizons 1/3 and all six predictor blocks,
computes eight streamed vector-Jacobian products per state-horizon, preselects
one causal horizon-layer before intervention outcomes, and applies same-state JOW,
residual-swap, and random-orthogonal interventions. One lens is frozen in the
base model and reused across treatment arms. The matched 22 MB checkpoint is
downloaded only after the frozen causal gate passes; shuffled geometry is
downloaded only if matched beats frozen.

Plan for roughly 15–30 minutes of first-run setup and verified checkpoint
downloads. A failed hypothesis should normally stop within another 30–60
minutes on a G4. Because matched and shuffled reuse one frozen lens and one
preselected causal layer, a successful three-condition screen should usually
take roughly 60–120 minutes total.
The notebook replaces these envelopes with a measured estimate from the actual
assigned GPU before the main screen.

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

## Stage 11 compute-gated action-response geometry pilot

Notebook: `11_action_response_geometry_pilot.ipynb`

Stage 11 is the inexpensive falsification step after Stage 10. It updates the
same JEPA action encoder and six AdaLN modulation maps, but its matched loss
contains no physical decoder, goal, pose, cost, or action-selection label. It
matches whitened, candidate-centered target-token responses across all ten
same-state alternatives. A latent-only treatment and a deterministic
within-state shuffled-correspondence treatment use the same optimizer and
checkpoint budget.

1. Open the Stage 11 notebook in a fresh Colab GPU runtime.
2. Leave `RUN_MODE="pilot"` and all other defaults unchanged.
3. Use the G4 / RTX PRO 6000 Blackwell runtime if it is still available. An
   A100 or L4 is also sufficient.
4. Run all eleven cells in order. Google Drive is off by default, so Drive
   quota does not affect the run.
5. Save both automatic downloads:
   `stage11_phase_c_checkpoint_rescue.zip` and
   `stage11_result_bundle.zip`.

The pilot uses 36 exact states per environment, one mandatory adaptation seed,
at most one confirmation seed, three treatments, two training projections,
three unseen evaluation projections, and at most ten epochs. The second seed
runs only when matched geometry beats both frozen and shuffled geometry on
calibration without cross-environment harm. Each treatment writes an atomic
local checkpoint every two epochs and resumes from that checkpoint if the cell
is rerun in the same runtime.

Expected runtime without an existing compatible cache:

- RTX PRO 6000 Blackwell / G4: approximately 35–70 minutes;
- A100 80 GB: approximately 40–80 minutes;
- L4: approximately 60–110 minutes.

Failure of the screening gate can save roughly one third of the adaptation
compute. Cache generation and fresh-readout evaluation still run so that a
stopped pilot returns an interpretable bundle.

The pilot may return `PROMOTE_TO_FULL_RUN`,
`PROMOTE_TO_FULL_RUN_WITH_EPOCH_EXTENSION`,
`GEOMETRY_ONLY_DIAGNOSIS`, `STOP_NO_DIRECT_GEOMETRY_SIGNAL`,
`STOP_NO_ROBUST_UNSEEN_GEOMETRY_GAIN`, or
`STOP_NATIVE_FIDELITY_FAILURE`. Promotion only authorizes changing the first
cell to `RUN_MODE="full"`; it is not confirmatory evidence.

The completed returned run used `RUN_MODE="full"` and passed the direct
geometry gate in all five unseen projections in both environments. It did not
pass the fresh-readout consensus gate, and PushT failed the native-planner
horizon-6 non-harm tolerance. The raw label
`STOP_NATIVE_FIDELITY_FAILURE` conflates native latent fidelity with native
planner safety: the matched latent-fidelity constraint passed. See
`../results/stage11_full_development_audit.json` and
`../docs/RESEARCH_STATE_AFTER_STAGE11_AND_ICLR_ROADMAP.md` before designing a
follow-up.

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
