# Stage 9: counterfactual value-equivalent action-path adaptation

## Decision

Stage 9 changes the JEPA-WM transition model itself. It does not add another
post-hoc energy head or residual after a frozen predictor.

The intervention updates only the parameters through which actions enter the
six-block `VisionTransformerAdaLN` predictor:

1. `predictor.action_encoder`;
2. each block's `adaLN_modulation[1]` linear map.

The visual encoder, the predictor's content/attention weights, the proprio
path, the output projection, and all target encodings remain frozen.

## Why this is the next test

The preceding stages jointly localize the failure.

- Stage 3B showed that a compact physical-state readout from predicted JEPA
  features improves planning in both environments.
- Stage 4 showed that action-specific prediction errors, rather than matched
  common-mode errors, causally damage action selection.
- Stage 5 showed that reweighting the loss of a frozen readout does not repair
  the transition representation.
- Stage 6 showed that a terminal action-effect head on a frozen rollout does
  not repair it.
- Stage 7 improved ordinary recurrent latent error without improving planning.
  Its layer audit nevertheless found action-effect information inside the
  AdaLN predictor, especially in Wall.
- Stage 8 showed that a high-capacity direct decision-energy head can improve
  one metric while nearly matching a wrong-state control. That is evidence of
  task/candidate shortcuts rather than state-specific counterfactual dynamics.

The shared failure of Stages 5--8 is therefore not insufficient head capacity.
Those methods left the causal action-conditioning pathway frozen or attempted
to correct its output after the transition had already been formed.

## Objective

For a fixed initial state \(s\), let \(a\in\mathcal A_s\) denote one of the
matched candidate action sequences, \(a_0\) the null-action branch, and

\[
  \hat z^h_{s,a}=F_\theta^h(E(o_s),a)
\]

the recurrent JEPA prediction at horizon \(h\). A fixed differentiable
CountSketch \(P\) and a temporary linear semantic map \(D_\psi\) produce

\[
  \hat x^h_{s,a}=D_\psi(P\hat z^h_{s,a}),
\]

where \(x\) is a goal-independent physical endpoint:

- PushT: normalized block \(x,y,\sin\phi,\cos\phi\);
- Wall: normalized agent \(x,y\).

The matched intervention minimizes

\[
\begin{aligned}
\mathcal L_{\mathrm{CAVE}}
=&
\lambda_z\mathcal L_{\mathrm{anchor}}
+\lambda_x\mathcal L_{\mathrm{state}}
+\lambda_\Delta\mathcal L_{\mathrm{effect}}
+\lambda_a\mathcal L_{\mathrm{action}},\\
\mathcal L_{\mathrm{anchor}}
=&
\operatorname{SmoothL1}(\hat z^h_{s,a},z^h_{s,a}),\\
\mathcal L_{\mathrm{state}}
=&
\operatorname{SmoothL1}(\hat x^h_{s,a},x^h_{s,a}),\\
\mathcal L_{\mathrm{effect}}
=&
\operatorname{SmoothL1}\left(
  \hat x^h_{s,a}-\hat x^h_{s,a_0},
  x^h_{s,a}-x^h_{s,a_0}
\right),\\
\mathcal L_{\mathrm{action}}
=&
\left\|A_\omega\!\left(P(\hat z^h_{s,a}-\hat z^{h-1}_{s,a})\right)
  -a_h\right\|_2^2.
\end{aligned}
\]

The anchor protects the pretrained representation. The state term makes the
latent usable through a small goal-independent decoder. The same-state
effect term removes common-mode state error from the training signal and
directly aligns the differences that determine counterfactual action choice.
The displacement action loss prevents action-insensitive local geometry, but
is auxiliary: Stage 7 showed that action decodability alone is not sufficient.

The temporary heads are discarded after adaptation. Every adapted predictor
is evaluated through a newly fitted, identical frozen linear readout using a
separately seeded projection that never carried an adaptation gradient. This
prevents either a powerful training head or a projection-specific codebook
from receiving credit for memorizing task or candidate identity.

## Decision guarantee

Let the true goal cost \(c_g(x)\) be \(L_g\)-Lipschitz and suppose the semantic
rollout obeys

\[
  \max_a\|\hat x_{s,a}-x_{s,a}\|\leq\epsilon.
\]

Then

\[
  \max_a|c_g(\hat x_{s,a})-c_g(x_{s,a})|\leq L_g\epsilon.
\]

If the true gap between the best and second-best candidate is
\(\Delta_g(s)>2L_g\epsilon\), the predicted argmin is the true argmin.

The tighter quantity for action selection is the pairwise margin error

\[
  \epsilon_{\mathrm{pair}}
  =
  \max_{a,b}
  \left|
  [\hat c_g(a)-\hat c_g(b)]-[c_g(a)-c_g(b)]
  \right|.
\]

The best action is preserved whenever
\(\epsilon_{\mathrm{pair}}<\Delta_g(s)\). Common-mode endpoint error cancels
from this expression. This is why the no-op-relative effect loss is the
principal intervention rather than another ordinary prediction loss.

Because the learned endpoint is goal independent, the same bound applies to
unseen goals from the same Lipschitz task family. This is materially stronger
than fitting a scalar energy to the inspected development goals.

## Methods and controls

All methods use the same states, candidate branches, checkpoints, target
encodings, optimizer budget, fixed projection, and post-adaptation readout.

1. `frozen_linear_pose`: no predictor adaptation; the Stage 3B-style baseline.
2. `latent_only_action_path`: adapt the same action pathway using only the
   ordinary latent anchor.
3. `shuffled_cave_action_path`: use the full objective but deterministically
   permute non-null physical outcomes within each same-state candidate set.
4. `matched_cave_action_path`: use the full objective with the correct
   same-state action/outcome assignment.
5. `native_latent_distance`: the original checkpoint and original goal-latent
   scoring rule.
6. `oracle_pose`: simulator endpoint lower bound.

The shuffled control keeps outcome statistics and optimizer exposure while
destroying only the causal action-to-outcome correspondence. The null branch
is kept fixed.

## Development decision

This is an exploratory development experiment on the already inspected task
family. It is not confirmatory evidence.

Continue to a new task-family, multi-seed confirmation only if:

1. `matched_cave_action_path` improves normalized regret and weighted pairwise
   accuracy over `frozen_linear_pose` in both environments;
2. it also beats `latent_only_action_path`;
3. matched minus shuffled has the correct sign in both environments;
4. native latent prediction is non-inferior under a prospectively fixed
   tolerance;
5. the gain is present at more than one horizon and is not produced solely by
   null-action choices.

If matched does not beat frozen, the conclusion is that limited action-path
adaptation is insufficient. If matched and shuffled are similar, the apparent
gain is not evidence of correct counterfactual correspondence.

## Evidence status

`EXPLORATORY_METHOD_DEVELOPMENT`

No Stage 9 result may be described as held-out confirmation because the task
family and failure modes informed this intervention.
