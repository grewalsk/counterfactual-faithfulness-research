# Stage 0 report

Audit date: 2026-07-25.

## Outcome

The project is feasible without a physical robot, but the novelty claim must be
narrow. Recent work already covers paired counterfactual prompts, action
following, functional world-model evaluation, decision-based regret, and the
general argument that world models should be evaluated under intervention.
The defensible contribution is:

> Exact same-state simulator branching, a quantitative metric for error in the
> *difference* caused by executable alternative actions, and a preregistered
> test of whether that action-dependent residual predicts ranking or regret
> beyond ordinary rollout error.

Do not claim the first counterfactual world-model benchmark. Publication
novelty remains contingent on Stage 2 demonstrating non-redundancy or a strong,
interpretable negative result.

## Literature and novelty audit

| Work | What it establishes | Remaining distinction |
|---|---|---|
| [DINO-WM (2024)](https://arxiv.org/abs/2411.04983) | Frozen DINOv2 features plus action-conditioned prediction support zero-shot planning in Push-T, PointMaze, Wall, and other environments. | It does not center errors across exact same-state action branches or test incremental prediction of regret. |
| [JEPA-WMs (2025/2026)](https://openreview.net/forum?id=TuYC5Fpp7M) and [official code](https://github.com/facebookresearch/jepa-wms) | Studies architecture/training/planning choices and releases Push-T checkpoints for JEPA-WM and DINO-WM. | Supplies the most practical model substrate, but not this paired residual estimand. |
| [WorldArena (2026)](https://arxiv.org/abs/2602.08971) | Unifies perceptual and functional evaluation, including action planning and policy evaluation. | Broad utility benchmark, not exact-state quantitative intervention-effect recovery. |
| [MiraBench (2026)](https://arxiv.org/abs/2605.29360) | Closest collision: action-following fidelity, failure-inducing perturbations, optimism bias, human/VLM scoring. | Uses annotated video judgments; it does not supply exact simulator branch ground truth or the proposed residual/incremental-validity test. |
| [What-If World (2026)](https://arxiv.org/abs/2605.27589) | Paired prompts vary one physical detail and score outcome divergence. | Prompt intervention over real frames, not executable action sequences restored from one simulator state. |
| [WFM-Eval (2026)](https://openreview.net/forum?id=gvtwynIibB) | Object-level hallucination diagnostics can predict downstream policy success. | No paired action intervention. It raises the bar: the proposed metric must add value beyond ordinary diagnostics. |
| [WorldModelGym (2026)](https://reka.ai/labs/research/worldmodelgym) | Scores decision-based fidelity using normalized regret over open-loop choices. | It overlaps on regret but not on decomposition of same-state intervention-effect error. |
| [Decision-making-centric position (2026)](https://arxiv.org/abs/2606.15032) | Explicitly advocates counterfactual action fidelity, policy ranking, regret, planning, and intervention. | This sharply limits conceptual novelty; empirical protocol and evidence must carry the contribution. |

The core algebra is also important: paired effect MSE equals a fixed multiple of
the action-centered prediction-error variance. Therefore the metric should be
presented as a transparent decomposition of error, not as proof that a neural
network has learned causality.

## Simulator audit

### Selected: Push-T

Reasons:

- Fast CPU physics and rendering.
- Public DINO-WM and JEPA-WM checkpoints use the same task family.
- Low-dimensional privileged state and contact events make validation possible.
- The public [gym-pusht API](https://pypi.org/project/gym-pusht/) documents
  reset to a supplied state.
- Local Stage 0 tests reconstruct a fresh Pymunk space for every branch and
  obtain bitwise-identical endpoints and diagnostics over four repetitions.

Important scope: the public five-number reset state contains agent position,
block position, and block angle. It is exact for the tested canonical stationary
branch states. It is not sufficient to claim arbitrary mid-contact restoration
because it omits velocities and contact-solver cache. Stage 1 branches only
from freshly reconstructed stationary states. A later mid-trajectory protocol
must serialize all dynamical state or use MuJoCo integration-state copying.

### Recommended second environment: PointMaze or Wall

These are preferable to MetaWorld for the first replication because they are
cheap, already supported by the same checkpoints, and reduce integration risk.
PointMaze's MuJoCo substrate has explicit integration-state APIs.

### Later environments: MetaWorld, RoboSuite/RoboCasa

MuJoCo documents that copying `mjSTATE_INTEGRATION` and stepping source and
destination produces identical results, including warm-start state needed for
perfect numerical reproducibility
([MuJoCo simulation-state documentation](https://mujoco.readthedocs.io/en/latest/programming/simulation.html)).
RoboSuite exposes `get_state`, `set_state`, and `set_state_from_flattened`, with
`forward()` required after restoration
([RoboSuite API](https://robosuite.ai/docs/source/robosuite.utils.html)).
These are strong Stage 3 candidates, but their renderer/controller state and
task randomization must also be captured. RoboCasa is built on RoboSuite and
MuJoCo, but is too integration-heavy for the first pilot.

## Checkpoint audit and model tiers

The official [JEPA-WMs repository](https://github.com/facebookresearch/jepa-wms)
offers DINO-WM and JEPA-WM Push-T, PointMaze, Wall, and MetaWorld checkpoints
with DINOv2 ViT-S/14 encoders. The Hugging Face listing reports:

- DINO-WM Push-T: 275 MB.
- JEPA-WM Push-T: 212 MB.
- Optional image decoders: 3.64 GB each.

Stage 1 selects DINO-WM Push-T and deliberately disables decoder heads. It
downloads the 275 MB predictor checkpoint and the much smaller DINOv2 ViT-S/14
encoder weights; it downloads no dataset. Metrics operate in frozen latent
space. This avoids the 3.64 GB decoder and fits Tier 1.

Stage 2 should compare at least DINO-WM Push-T and JEPA-WM Push-T, plus a
substantively different checkpoint if one can be integrated without training.
The [Nano World Model repository](https://github.com/simchowitzlabpublic/nano-world-model)
advertises public DINO-WM-domain checkpoints and is a candidate third family,
but its interface was not needed for Stage 1.

Do not start with V-JEPA 2-AC. The official JEPA-WMs listing places its DROID
checkpoint in multi-gigabyte territory and it does not serve the fast Push-T
pilot.

## Implemented Stage 0 artifacts

- Paired effect RMSE, normalized paired RMSE, effect cosine, common-mode and
  action-dependent error decomposition.
- Top-1 action accuracy, pairwise ranking accuracy, raw regret, and normalized
  regret.
- State-clustered bootstrap interval.
- Grouped cross-validated incremental validity with whole states held out.
- Exact stationary reset-state Push-T test.
- Synthetic tests covering perfect prediction, common bias cancellation, the
  pairwise identity, ranking/regret, grouped bootstrap, and incremental signal.
- A self-contained Stage 1 Colab with pinned non-PyTorch dependencies, one
  configuration cell, optional Drive, checkpoints/cache, resume state, GPU
  monitoring, logs, plots, JSON/CSV export, failure trace, and ZIP packaging.

## Risks and controls

- **Novelty risk: high.** MiraBench, WorldModelGym, and the June 2026 position
  paper occupy adjacent claims. Control: make the exact paired residual and
  incremental-regret result central.
- **Metric circularity.** Paired error and ranking can use related features.
  Control: Stage 2 primary regret comes from executing the chosen candidate in
  the simulator; include privileged-state costs and sensitivity readouts.
- **Low-effect pairs.** If alternative actions cause indistinguishable outcomes,
  normalized scores become unstable. Control: report effect scale and
  preregister a minimum-effect sensitivity analysis.
- **Pseudo-replication.** There are many pairs per state. Control: state-clustered
  resampling and held-out state folds.
- **Checkpoint/environment mismatch.** Rendering or action convention drift can
  invalidate results. Control: use the environment shipped at a pinned
  JEPA-WMs commit and assert action chunk dimensions, image shape, and
  deterministic branches before model evaluation.
- **Simulator-to-reality limitation.** A simulator may omit phenomena present in
  hardware. The study makes no claim about real-world robotic reliability.

## Stage 1 success criteria

The first Colab run validates the pipeline only. It passes if:

1. The pinned repository and DINO-WM checkpoint load on a Tier 1 GPU.
2. Repeating one restored branch is bitwise exact.
3. Alternative actions produce at least two distinct simulator outcomes.
4. Both ordinary and paired metrics are finite.
5. Ranking/regret records are exported.
6. A result ZIP contains configuration, versions, logs, checkpoint manifest,
   intermediate records, plots, CSV/JSON metrics, and `FAILURE_TRACE.txt`.

No hypothesis is accepted or rejected from this smoke test.

