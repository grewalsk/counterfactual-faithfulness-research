# Stage 10: fidelity-constrained pairwise margin adaptation

## Evidence status

`EXPLORATORY_METHOD_DEVELOPMENT`

Stage 10 uses the task family and failure modes already inspected in Stages
3--9. It can select a method for later confirmation, but it cannot itself
provide confirmatory evidence.

## Decision

Stage 10 tests **Fidelity-Constrained Pairwise Margin Adaptation (FPMA-JEPA)**.
The intervention changes only:

1. `predictor.action_encoder`;
2. each of the six `predictor_blocks[i].adaLN_modulation[1]` maps.

The visual encoder, attention and MLP content weights, proprio pathway,
predictor output projection, target encodings, and three physical-state
decoders remain frozen.

The experiment does not rerun Stage 9 with more epochs. It replaces Stage 9's
mean physical endpoint objective with an explicit upper bound on planning
regret and makes native JEPA fidelity a hard checkpoint constraint.

## Why this follows from Stage 9

The completed Stage 9 run was valid, but did not establish a general repair.
Its compact result audit and source-bundle hashes are checked into
`results/stage9_development_audit.json`.

- Correctly matched action-path adaptation improved long-horizon PushT
  normalized regret under both a fresh physical readout and the native latent
  planner.
- The strongest native-planner PushT horizon-6 contrast was positive, which
  shows that changing JEPA's real action pathway can correct some severe
  counterfactual decisions.
- The gain was sparse and did not transfer reliably to PushT horizons 1 and 3
  or to Wall.
- PushT top-1 accuracy did not improve with the horizon-6 regret reduction.
  The method sometimes made bad choices less severe without choosing the best
  candidate more often.
- The matched physical endpoint calibration objective worsened in PushT, all
  selected checkpoints were on the final epoch boundary, and the declared
  two-percent native non-inferiority tolerance was not used to reject
  checkpoints.

This pattern is the objective-mismatch failure expected when a model is trained
for average prediction while it is used for an argmin decision. Small errors
in unimportant directions can dominate the loss, while one wrong action
margin can dominate planning regret.

The design is motivated by value-aware model learning, value equivalence, and
decision-focused ranking:

- Farahmand et al., [Value-Aware Loss Function for Model-based
  Reinforcement Learning](https://proceedings.mlr.press/v54/farahmand17a.html)
- Grimm et al., [The Value Equivalence
  Principle](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3bb585ea00014b0e3ebe4c6dd165a358-Abstract.html)
- Mandi et al., [Decision-Focused Learning through
  Ranking](https://proceedings.mlr.press/v162/mandi22a.html)
- Lambert et al., [Objective Mismatch in Model-based Reinforcement
  Learning](https://proceedings.mlr.press/v120/lambert20a.html)

## Frozen physical decoders

Before any adaptation, fit \(K=3\) linear, goal-independent physical decoders
on the frozen JEPA rollout:

\[
  D_k(P_k\hat z^h_{s,a})=\hat x^{h,k}_{s,a}.
\]

The decoder targets are:

- PushT: block \(x/512,y/512,\sin\phi,\cos\phi\);
- Wall: agent \(x/65,y/65\).

No decoder receives a goal, task identifier, action identifier, candidate
index, or scalar cost. Each CountSketch projection has a distinct fixed seed.
The decoder is fitted on `probe_train`; its ridge coefficient is selected on
`probe_calibration`. If the best ridge value lies on either edge of the grid,
the grid expands geometrically. A decoder fit fails rather than accepting a
truncated boundary optimum.

All three decoders are fitted and frozen before any treatment is trained.
Their checksums must remain unchanged after every treatment.

The predicted cost is analytic:

\[
  \hat c^{h,k}_{s,a}
  =
  C_{g_s}\!\left(D_k(P_k\hat z^h_{s,a})\right),
\]

where \(C_g\) is the simulator's physical goal cost. The decoder is goal
independent even though the analytic decision cost uses the current goal.

## Pairwise regret certificate

For one state and horizon, write \(c_a\) for the true simulator cost and
\(\hat c^{k}_a\) for the cost obtained through decoder \(k\). Define

\[
  q_s
  =
  \max\left(\max_a c_a-\min_a c_a,\epsilon_q\right)
\]

and, for every ordered pair \(a\neq b\),

\[
  e^k_{ab}
  =
  \frac{
    (\hat c^k_a-\hat c^k_b)-(c_a-c_b)
  }{q_s}.
\]

Then \(e^k_{ba}=-e^k_{ab}\), so the certificate below needs only one
orientation, \(a<b\), from each unordered pair.

With \(p=8\), define

\[
  B^k_p
  =
  \left[
    \sum_{a<b}
    \left((e^k_{ab})^2+\epsilon_0^2\right)^{p/2}
  \right]^{1/p},
  \qquad
  B_p=\max_k B^k_p.
\]

The sum is essential. Replacing it with a mean can make the p-norm smaller
than the largest pair error and invalidates the guarantee.

All ten candidates are processed at one parameter value, so the notebook
computes all 45 unique unordered pairs. It does not use a rotating partial-pair
surrogate and does not duplicate the no-op branch.

### Theorem 1: normalized regret bound

Let

\[
  a^*\in\arg\min_a c_a,\qquad
  \hat a_k\in\arg\min_a\hat c^k_a.
\]

Then, for every frozen training decoder \(k\),

\[
  \frac{c_{\hat a_k}-c_{a^*}}{q_s}
  \leq B^k_p
  \leq B_p.
\]

### Proof

Because \(\hat a_k\) minimizes predicted cost,

\[
  \hat c^k_{\hat a_k}-\hat c^k_{a^*}\leq0.
\]

Rearranging the definition of \(e^k_{\hat a_k,a^*}\),

\[
\begin{aligned}
  c_{\hat a_k}-c_{a^*}
  &=
  \hat c^k_{\hat a_k}-\hat c^k_{a^*}
  -q_s e^k_{\hat a_k,a^*}\\
  &\leq
  q_s\left|e^k_{\hat a_k,a^*}\right|.
\end{aligned}
\]

Antisymmetry makes
\(\lvert e^k_{\hat a_k,a^*}\rvert\) equal to the magnitude of the
corresponding stored \(a<b\) coordinate. Every such coordinate magnitude is
bounded by its sum p-norm:

\[
  |e^k_{\hat a_k,a^*}|\leq B^k_p\leq B_p.
\]

Dividing by \(q_s>0\) proves the claim. Multiple true optima do not change the
argument. \(\square\)

### Corollary 1: tie-tolerant optimal-action certificate

Let the numerical ranking tolerance be \(\tau=10^{-9}\), define the
tie-tolerant optimal set

\[
  \mathcal A^*_\tau
  =
  \{a:c_a\leq\min_b c_b+\tau\},
\]

and let

\[
  \Delta_{s,\tau}
  =
  \min_{a\notin\mathcal A^*_\tau}
  \frac{c_a-\min_b c_b}{q_s}.
\]

When \(\mathcal A^*_\tau\) contains every candidate, define
\(\Delta_{s,\tau}=+\infty\).

If \(B^k_p<\Delta_{s,\tau}\), decoder \(k\)'s selected candidate is in
\(\mathcal A^*_\tau\). This is the same numerical optimal-set convention used
by the reported top-1 metric.

The inequality is strict. Equality permits a predicted tie whose deterministic
tie break can select a candidate outside \(\mathcal A^*_\tau\).

The corresponding deterministic indicator bound is

\[
  \mathbf 1[\hat a_k\notin\mathcal A^*_\tau]
  \leq
  \min\left(1,\frac{B^k_p}{\Delta_{s,\tau}}\right).
\]

The clipped expression is reported, not optimized. Its gradient would be zero
after a serious violation. Training instead uses

\[
  L_{s,h}^{\mathrm{decision}}
  =
  B_p
  +
  \lambda_{\mathrm{top1}}
  \frac{B_p}{\Delta_{s,\tau}}
\]

when \(\Delta_{s,\tau}\geq\Delta_{\min}\), and \(B_p\) alone for near ties. If
every candidate lies in \(\mathcal A^*_\tau\), the gap term is omitted.

The theorem is deterministic conditional on a particular frozen decoder. It
does not prove optimization success, population generalization, or transfer to
a newly fitted decoder.

## Worst-group optimization

The train split is balanced over tasks. A GroupDRO weight is maintained for
each train task by horizon group and updated from the detached certificate
loss. The differentiable per-state objective uses those weights, approximating

\[
  \min_\theta\max_{t,h}
  \mathbb E_{s\mid t}\left[
    L_{s,h}^{\mathrm{decision}}(\theta)
  \right].
\]

Checkpoint selection computes the complete calibration objective separately
inside each environment and chooses the lowest worst-task-by-horizon value. A
long-horizon gain therefore cannot hide an early-horizon or task failure
within that environment. The later cross-environment gate prevents PushT from
hiding a Wall failure.

## Native-fidelity constraint

For horizon \(h\), let

\[
  A_h(\theta)
  =
  \mathbb E\left[
    \operatorname{SmoothL1}
    \left(
      F_\theta^h(E(o_s),a),
      z^h_{s,a}
    \right)
  \right].
\]

Every action receives one weight. Targets remain correctly aligned even in the
shuffled FPMA control.

The constraint is

\[
  g_h(\theta)
  =
  \frac{A_h(\theta)}
  {\max(A_h(\theta_0),\epsilon_A)}
  -1.02
  \leq0
  \quad
  \text{for every }h\in\{1,3,6\}.
\]

Training uses the projected augmented Lagrangian

\[
  L_{\mathrm{AL}}
  =
  \sum_h
  \frac{
    \max(0,\lambda_h+\beta g_h)^2-\lambda_h^2
  }{2\beta},
\]

with

\[
  \lambda_h
  \leftarrow
  \max(0,\lambda_h+\beta g_h).
\]

The one-state stochastic constraint used for gradients is

\[
  \tilde g_{s,h}(\theta)
  =
  \frac{
    A_{s,h}(\theta)-A_{s,h}(\theta_0)
  }{
    \max(A_h(\theta_0),\epsilon_A)
  }
  +
  \frac{
    A_h(\theta_0)
  }{
    \max(A_h(\theta_0),\epsilon_A)
  }
  -1.02.
\]

The one-state quantity is an unbiased estimator of \(g_h\), but the nonlinear
augmented penalty applied to it is deliberately **not** described as an
unbiased estimator of the aggregate augmented Lagrangian. For fixed
\(\lambda_h\), its expectation is a conservative Jensen upper surrogate for
the aggregate penalty. The baseline state losses and aggregate denominator
are measured through an epoch-zero forward pass using the exact same cached
initial encodings as adaptation. At every calibration boundary, the current
model is rerun over the complete training split and that exact aggregate
updates the dual; calibration data never update the multiplier.

The augmented term encourages feasibility but does not establish it. At each
checkpoint, the complete calibration set is rerun and every horizon is checked
independently. An infeasible checkpoint is ineligible regardless of decision
loss. Parameters and optimizer momentum roll back to the last feasible
checkpoint, then the current learning rate is halved cumulatively.

Epoch zero has native ratios exactly equal to one and is always eligible.
Therefore the constrained feasible set cannot be empty. If no update is both
useful and feasible, the notebook selects the pretrained action path.

Each optimizer step is also projected into a relative trust region separately
for:

1. the action encoder;
2. AdaLN block 1;
3. AdaLN block 2;
4. AdaLN block 3;
5. AdaLN block 4;
6. AdaLN block 5;
7. AdaLN block 6.

## Treatments

For each of three adaptation seeds:

1. `frozen`;
2. `fidelity_constrained_latent_only`;
3. `fidelity_constrained_shuffled_fpma`;
4. `fidelity_constrained_matched_fpma`;
5. `unconstrained_matched_fpma`.

The four trained treatments share parameter initialization, task and state
order, all ten candidates, decoder weights, parameter scope, optimizer,
checkpoint opportunities, and trust-region machinery.

The shuffled treatment keeps the no-op label fixed and uses a deterministic
derangement of all nine non-null outcomes. It preserves the outcome multiset
and has no non-null fixed point. Only physical decision labels are shuffled;
the native latent targets are never shuffled.

The unconstrained treatment isolates the consequence of the two-percent
native-fidelity constraint.

## Checkpoint rule

Calibration checkpoints are evaluated at epochs

\[
  0,2,4,\ldots,24.
\]

Only fidelity-feasible checkpoints can be selected for constrained methods.
Ties select the earliest epoch.

If epoch 24 is the best eligible checkpoint and its improvement over the
previous eligible checkpoint exceeds \(10^{-4}\), training extends to epoch
36. The same rule can extend epoch 36 to 48. If epoch 48 remains an improving
best boundary, the run emits `UNDERTRAINED_INCONCLUSIVE` and cannot advance.

Checkpoint filenames encode the method, environment/model, adaptation seed,
and configuration signature. Resume loading verifies each field.
An atomic `latest` checkpoint is also written after every calibration boundary
and contains the current parameters, optimizer, dual variables, GroupDRO
weights, RNG state, best checkpoint, and rollback anchor. A restarted cell
resumes from that boundary; completed treatments are never retrained.

## Fresh-readout evaluation

The training decoders are not used for the primary empirical transfer result.
For each adapted predictor:

1. generate rollouts once;
2. project them through five unseen CountSketch seeds;
3. fit a new linear physical-state readout for each projection on
   `probe_train`;
4. choose its ridge coefficient on `probe_calibration`;
5. evaluate planning on `development_holdout`.

Training and evaluation projection seeds are disjoint. Each projection is
analyzed separately and is not counted as an independent sample.

The primary metrics are:

- normalized planning regret;
- weighted pairwise accuracy;
- tie-tolerant optimal-set top-1 accuracy (\(\tau=10^{-9}\)).

Pose error, margin RMSE, null selection, candidate concentration, native
latent error, and the training-decoder certificate are diagnostics.

## Inference

The generalization unit is the task.

For a paired method contrast:

1. compute matched state-level differences for each adaptation seed;
2. average adaptation seeds within state;
3. average states within task;
4. give each task equal weight;
5. bootstrap tasks.

Projection seeds are reported separately. With three development tasks per
environment, intervals are descriptive and cannot justify a confirmatory
claim.

Undefined ranking values from all-tied candidate sets are never converted to
zeros. They are omitted only for that metric; if any primary contrast then
lacks all three development tasks, the decision is
`INCOMPLETE_TASK_INFERENCE` and cannot advance.

## Prospective advancement gate

Advance to a numerically new task-family confirmation only if all conditions
hold:

1. every selected matched checkpoint satisfies the two-percent native
   constraint at horizons 1, 3, and 6;
2. matched FPMA jointly improves regret and weighted pairwise accuracy over
   frozen, latent-only, and shuffled controls at at least two of three
   *common* horizons, with both task-bootstrap lower bounds above zero;
3. condition 2 holds for at least four of five unseen evaluation projections
   separately in both PushT and Wall;
4. every ridge optimum is interior after expansion;
5. every computed regret satisfies its deterministic training-decoder bound,
   and every strict top-1 certificate is correct;
6. no required treatment has an unresolved final checkpoint boundary;
7. no matched adaptation seed, projection, or horizon collapses to the no-op
   or one candidate index: maximum candidate share must be at most 0.80,
   no-op share at most 0.75, and at least three distinct candidates must be
   selected in every seed/projection/horizon stratum;
8. the native goal-latent planner has no regret or ranking degradation larger
   than 0.02 at any horizon and jointly improves both metrics at at least one
   horizon in each environment.

Passing this gate means only that the recipe is worth freezing. The next run
must use numerically new goals/layouts and no further method changes.

The three-task bootstrap intervals remain development diagnostics rather than
confirmatory uncertainty statements. Requiring their lower bounds here is an
intentionally conservative method-selection rule, not a claim that projection
or adaptation seeds increase the task sample size.

If no feasible update improves the worst-group certificate, the result
falsifies this action-path-only FPMA intervention. It would imply that the
repair needs a wider transition-model update, a different physical bottleneck,
or new pretraining data—not another post-hoc decision head.

## Execution

The default full run uses:

- 96 exactly restored states per environment;
- ten executable candidates per state;
- horizons 1, 3, and 6;
- three adaptation seeds;
- three frozen training decoders;
- five unseen evaluation projections;
- four trained controls plus the frozen model;
- checkpoint opportunities through epoch 24, with prospective extensions to
  36 and 48.

An A100 with at least 40 GB is required for the full run. The notebook reuses a
compatible Stage 7 simulator and transition cache when present, supports
checkpoint-level treatment resume, packages captured failures, evaluates the
native latent planner as well as fresh readouts, and automatically downloads
`stage10_result_bundle.zip`. With the Stage 7 cache, the initial 24-epoch
three-seed matrix is expected to take roughly 6--10 hours on an A100 80 GB or
8--12 hours on an A100 40 GB. A rare extension to epoch 48 can approach
15--20 hours, so Drive persistence is enabled by default for a full run.

Every truth, transition, goal, task, and split artifact is SHA256-bound before
adaptation. The exact JEPA-WM and DINOv2 asset hashes are pinned and verified
after each model load. The JEPA-WM hub loader is locally patched at the pinned
source commit to request Hugging Face snapshot
`9b9c41ef249466630dbf1a20e78391865d07b3b9` explicitly; the resolved
snapshot-path asset is then hash-checked. Loading fails closed if that exact
snapshot cannot be retrieved; the upstream mutable-URL fallback is disabled.
Those identities are also embedded in decoder, checkpoint, and rollout
metadata. A smoke or otherwise reduced matrix can exercise the pipeline but is
hard-gated to
`NONPROTOCOL_RUN_NO_SCIENTIFIC_DECISION`.
