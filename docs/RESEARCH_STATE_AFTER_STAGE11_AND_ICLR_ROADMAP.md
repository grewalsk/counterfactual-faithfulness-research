# Research state after Stage 11 and the path to a clear ICLR paper

## Status

This document was written after the completed Stage 11 full development run.
It is a synthesis of the repository's staged evidence, not a preregistration
and not a claim of confirmation.

The raw Stage 11 evidence is preserved in
`results/bundles/stage11_result_bundle/`. The independently reconstructed
numbers and source hashes are in
`results/stage11_full_development_audit.json`.

## Executive conclusion

The project has crossed an important boundary:

> We can now improve JEPA-WM's correct, state-specific latent response to
> alternative actions in a way that transfers to unseen projections and beats
> a shuffled action/outcome control.

That is a real mechanistic positive result. It is not yet the complete result
needed for an ICLR paper because improved action-response geometry does not
reliably change the action selected by either a fresh physical readout or
JEPA's native goal-latent planner.

The remaining problem is no longer simply "the world model does not represent
action effects." Stage 11 shows that those effects can be repaired. The
remaining problem is:

> How should a goal-conditioned planner consume repaired counterfactual
> transition geometry so that better geometry produces better decisions
> without a high-capacity head hiding transition errors?

The next experiment should address this interface directly. Repeating ARGA
with more epochs is not justified.

## The scientific problem

For a state \(s\), candidate action sequence \(a\), horizon \(h\), and task
goal \(g\), let the simulator produce a physical outcome \(x^*_{sah}\) and
cost \(c_g(x^*_{sah})\). A world model produces a latent prediction
\(\hat z^\theta_{sah}\).

Ordinary predictive training minimizes an average error such as

\[
\mathbb E_{s,a,h}
\left\|\hat z^\theta_{sah}-z^*_{sah}\right\|^2.
\]

Planning instead selects an argmin:

\[
\hat a
=
\arg\min_a q(\hat z^\theta_{sah},g).
\]

The argmin depends on pairwise action margins, not average prediction error.
A model can reduce ordinary loss while changing the wrong candidate margin,
and therefore recommend a worse action.

The controlled counterfactual object is the same-state action effect

\[
\Delta z_{sabh}
=
z_{sah}-z_{sbh}.
\]

Counterfactual action faithfulness asks whether predicted pairwise effects
match true pairwise effects closely enough for action ordering and regret.

## What the stages have established

| Stage | Main result | What it ruled in or out |
|---|---|---|
| 2B | Raw latent counterfactual metrics were negative | Latent distance alone is not an actionable counterfactual metric |
| 2C--3B | Physical state is linearly decodable and improves planning across PushT and Wall | Useful decision information exists in the latent, but ordinary regret prediction remains limited |
| 4 | Destroying action-specific structure hurts more than matched common-mode corruption | Correct same-state action structure is causally important |
| 5 | Counterfactual readout training did not beat an independent-pair control | A downstream gain is not enough unless it is specific to correct correspondence |
| 6--7 | Structured and recurrent adapters improved ordinary latent fit but not planning | Better average prediction is not a sufficient repair |
| 8 | A high-capacity decision energy was matched by a wrong-state control | A decision head can exploit regularities without repairing dynamics |
| 9 | Updating JEPA's real action path produced a narrow PushT horizon-6 native-planner gain | The causal action pathway can change real decisions |
| 10 | Decoder-specific pairwise-margin training did not transfer to fresh readouts | Optimizing a few decision heads can overfit the readout interface |
| 11 | Matched ARGA improved unseen action geometry in 5/5 projections in both environments, but planning transfer was incomplete | Correct relative action geometry is repairable; the goal/planner interface is now the main bottleneck |

This sequence is unusually informative. Several tempting explanations have
already been tested:

1. The problem is not merely that physical state is absent from the latent.
2. The problem is not solved by reducing ordinary latent error.
3. The problem is not solved by adding a flexible downstream energy.
4. The problem is not solved by optimizing a small set of physical readouts.
5. Correct action/outcome correspondence matters: Stage 11 matched geometry
   beats shuffled geometry on unseen projections.

## Stage 11: audited result

The returned run was the full development matrix, not the default cheap pilot:

- 96 exact states per environment;
- PushT and Wall;
- three adaptation seeds;
- five unseen evaluation projections;
- 2,000 task-clustered bootstrap repetitions;
- up to 32 epochs;
- exact simulator restoration;
- all 45 packaged manifest files hash-verified.

### Direct action-response geometry

Matched ARGA passed all ten environment-by-projection gates. Every projection
had positive point-estimate improvement over both frozen JEPA and shuffled
geometry at all three horizons.

Task-equal mean unseen whitened-geometry RMSE changed as follows:

| Environment | Horizon | Frozen | Matched ARGA | Relative reduction |
|---|---:|---:|---:|---:|
| PushT | 1 | 1.0289 | 1.0140 | 1.45% |
| PushT | 3 | 0.8382 | 0.8174 | 2.49% |
| PushT | 6 | 0.8314 | 0.7722 | 7.12% |
| Wall | 1 | 1.4649 | 1.3554 | 7.47% |
| Wall | 3 | 2.1128 | 1.9381 | 8.27% |
| Wall | 6 | 2.2334 | 2.1093 | 5.56% |

This is the strongest positive result in the project so far. It improves the
property explicitly shown by Stage 4 to be causally relevant and survives
unseen projections.

The specificity evidence is not uniform. PushT matched-versus-shuffled
intervals are strong. In Wall, horizon 3 is clearly specific, while horizons
1 and 6 have positive point estimates but weak three-task intervals. A new
task family remains essential.

### Fresh-readout planning

The full gate required joint regret and weighted-ranking improvement over
frozen and shuffled controls at two horizons in at least four of five
projections.

- PushT passed in 1/5 projections.
- Wall passed in 2/5 projections.

The gate therefore failed.

Wall nevertheless contains a substantial actionable signal:

| Horizon | Frozen regret | Matched regret | Relative reduction |
|---:|---:|---:|---:|
| 1 | 0.1212 | 0.0884 | 27.07% |
| 3 | 0.1680 | 0.1358 | 19.16% |
| 6 | 0.2359 | 0.2243 | 4.90% |

Weighted pairwise accuracy also improved at all three Wall horizons. The
problem is stability across projections and task-level intervals, not the
absence of any downstream signal.

PushT remained sparse: regret improved slightly at horizon 1, worsened
slightly at horizon 3, and improved by about 7.7% at horizon 6. Ranking and
top-1 changes were mixed.

### Native latent planner

The formal latent-fidelity constraint passed. The largest selected matched
calibration ratio was about 1.0198, within the 1.02 limit. Development
ordinary latent error was preserved at horizon 1 and improved at horizons 3
and 6.

The native planner non-harm gate failed only in PushT. At horizon 6, the
positive-is-better normalized-regret contrast was \(-0.0380\), beyond the
allowed \(-0.02\) harm tolerance. Its task interval
\([-0.1250,0.0109]\) includes zero, so the result is not conclusive evidence
of harm, but it fails the prospective safety rule.

The notebook's enum `STOP_NATIVE_FIDELITY_FAILURE` conflates two conditions.
It is semantically inaccurate for this run:

- `matched_native_fidelity_pass = true`;
- `native_planner_nondestruction_pass = false`.

This is a reporting-label bug, not a corruption of the raw metrics.

## Why better ARGA geometry need not improve the native planner

This outcome is mathematically coherent.

ARGA centers candidates and controls relative action effects:

\[
\tilde z_a=z_a-\frac1A\sum_b z_b,
\qquad
\tilde z_a-\tilde z_b=z_a-z_b.
\]

Its guarantee applies directly to bounded linear readouts of pairwise
differences. JEPA's native planner instead uses a squared goal distance. For a
positive-semidefinite metric \(M\),

\[
d_M(z,g)=(z-g)^\top M(z-g).
\]

The action margin is

\[
\begin{aligned}
d_M(z_a,g)-d_M(z_b,g)
&=
(z_a-z_b)^\top M(z_a+z_b-2g).
\end{aligned}
\]

The first factor is the pairwise action effect that ARGA improves. The second
factor depends on:

- the candidate midpoint \(z_a+z_b\);
- common-mode state prediction;
- the goal representation \(g\);
- the planner metric \(M\).

ARGA deliberately removes common-mode components from its main loss. A small
ordinary-latent constraint prevents gross damage but does not force the
pairwise action effect to align with the native goal direction. Therefore,
improving centered action geometry can improve every unseen geometry
projection while leaving the native squared-distance argmin unchanged or
worse.

This also explains why the Stage 11 result does not contradict its linear
readout theorem. Squared goal distance is a state- and goal-dependent
quadratic score, not one fixed bounded linear functional.

## The clearest next method: a shared goal-metric bridge

### Principle

Do not allow a planner head to compensate separately for each dynamics model.
Instead:

1. Freeze the Stage 11 matched ARGA action path.
2. Fit one low-capacity positive-semidefinite goal metric using target
   future-token and goal-token pairs on training tasks only.
3. Freeze that metric.
4. Apply the identical metric to frozen-JEPA, latent-only, shuffled-ARGA, and
   matched-ARGA rollouts.
5. Ask whether matched ARGA becomes better than frozen JEPA under the same
   planner.

Parameterize

\[
M=L^\top L+\epsilon I
\]

with a low rank \(L\), strong regularization, and explicit condition-number
control. Fit \(M\) so that distances between *target* future latents and target
goal latents reproduce simulator cost margins on probe-training tasks.

Training the metric on target latents rather than predicted latents is
important. It calibrates the representation-to-goal interface without giving
the metric an opportunity to memorize or hide the error pattern of a
particular dynamics treatment.

### Error decomposition

Let

\[
m^*_{ab}=c_a-c_b
\]

be the simulator cost margin,

\[
m^M_{ab}=d_M(z^*_a,g)-d_M(z^*_b,g)
\]

the target-latent metric margin, and

\[
\hat m^{M,\theta}_{ab}
=
d_M(\hat z^\theta_a,g)-d_M(\hat z^\theta_b,g)
\]

the predicted margin. Then

\[
\left|\hat m^{M,\theta}_{ab}-m^*_{ab}\right|
\le
\underbrace{
\left|\hat m^{M,\theta}_{ab}-m^M_{ab}\right|
}_{\text{dynamics/interface error}}
+
\underbrace{
\left|m^M_{ab}-m^*_{ab}\right|
}_{\text{target metric calibration error}}.
\]

The shared target-fitted \(M\) makes the second term identical across dynamics
treatments. Stage 11 ARGA is specifically intended to reduce the first term.
This decomposition yields a falsifiable comparison:

> If matched ARGA plus the shared metric does not beat frozen JEPA plus the
> same metric, then the repaired random-projection geometry is not sufficient
> for goal-directed planning.

### Required controls

The minimum development matrix is:

1. frozen JEPA + native Euclidean metric;
2. matched ARGA + native Euclidean metric;
3. frozen JEPA + shared learned metric;
4. latent-only adaptation + shared learned metric;
5. shuffled-geometry adaptation + shared learned metric;
6. matched ARGA + shared learned metric;
7. wrong-goal or task-shuffled metric control;
8. simulator oracle.

The decisive comparison is item 6 versus both items 3 and 5. Item 6 versus
item 3 isolates the value of repaired dynamics under an identical planner.
Item 6 versus item 5 tests correspondence specificity.

### Minimal cheap pilot

Before another expensive full run:

- reuse the six matched Stage 11 checkpoints;
- do not retrain ARGA;
- use two low ranks and two metric seeds;
- fit on probe-training tasks and select rank/regularization on calibration
  tasks;
- evaluate only the existing development tasks;
- stop if matched ARGA plus the shared metric does not directionally beat
  frozen plus the same metric and shuffled ARGA plus the same metric in both
  environments at at least two horizons.

This pilot is mostly readout computation plus cached/frozen rollouts. It should
be much cheaper than Stage 11 adaptation.

## What is required for an ICLR-quality paper

### A single clear thesis

A defensible thesis would be:

> Predictive JEPA world models can be ordinarily accurate yet
> counterfactually unfaithful for action selection. Same-state centered
> action-response geometry identifies and repairs the dynamics failure, while
> a shared goal-metric bridge is required to convert that repair into robust
> planning.

The paper cannot currently claim the last clause. Stage 12 must establish it.

### Evidence layers

An ICLR submission needs all four layers:

1. **Failure demonstration:** ordinary prediction error is insufficient for
   action ranking and regret.
2. **Causal mechanism:** matched action structure matters more than
   common-mode error, and matched ARGA beats shuffled correspondence.
3. **Actionable repair:** the complete method improves regret, weighted
   ranking, and preferably top-1 action choice.
4. **Generalization:** gains hold on frozen, numerically new task families and
   ideally an additional model or environment.

Stages 2--11 provide strong evidence for layers 1 and 2. Wall provides partial
evidence for layer 3. Layers 3 and 4 remain open.

### Confirmation protocol

After the planner bridge is selected, freeze:

- method architecture;
- parameter scope;
- loss weights;
- ranks and regularization;
- checkpoint selection;
- task generator;
- primary horizons and metrics;
- failure and non-harm tolerances.

Then run on a numerically new task family that was not used in Stages 3--12.
Task—not state, projection, or seed—must remain the inferential unit.

At minimum, require:

- both PushT and Wall;
- multiple training seeds;
- simulator-exact restoration;
- matched versus shuffled correspondence;
- shared-planner comparison between frozen and ARGA dynamics;
- ordinary latent non-inferiority;
- regret and weighted pairwise co-primary outcomes;
- task-clustered uncertainty;
- no projection voting presented as independent statistical replication.

For broad ICLR claims, add another public checkpoint or environment if
technically possible. Otherwise state the scope as JEPA-WM action-conditioned
manipulation models rather than all world models.

### Development go/no-go gate

Do not launch new-task confirmation unless the development bridge:

1. beats frozen dynamics under the identical learned metric;
2. beats shuffled ARGA under the identical learned metric;
3. improves both regret and weighted ranking at at least two horizons in both
   environments;
4. avoids native/ordinary prediction harm;
5. does not rely on one projection, task, or seed;
6. has no final-boundary checkpoint ambiguity.

If it works only in Wall, the honest result is a narrower Wall/mechanism paper
or additional method development—not a cross-environment repair claim.

## What not to do next

- Do not spend compute extending Stage 11 epochs; matched checkpoints were not
  boundary-inconclusive.
- Do not use a high-capacity end-to-end decision head; Stage 8 already shows
  how that can pass by exploiting task or candidate regularities.
- Do not optimize only the same physical readouts used for evaluation; Stage
  10 shows the transfer failure.
- Do not use new-task results to choose the method and then call the same
  tasks confirmatory.
- Do not describe `STOP_NATIVE_FIDELITY_FAILURE` literally; the latent
  fidelity constraint passed.
- Do not claim that random-projection geometry alone solves planning.

## Paper-readiness decision

The project is closer to a strong paper than after Stage 10 because it now has
a specific positive mechanism rather than a collection of null repairs.

It is not ready for ICLR submission today. The missing result is compact and
well localized:

> Show that a planner calibrated independently on target latents can convert
> matched ARGA's repaired action geometry into robust, correspondence-specific
> regret and ranking improvements on both environments, then freeze and
> confirm on new tasks.

If that succeeds, the paper has a coherent problem, theorem, causal controls,
repair, and generalization story. If it fails, Stage 11 remains a valuable
mechanistic result, but the central "framework to fix planning" claim must be
narrowed.
