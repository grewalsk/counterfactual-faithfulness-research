# Mathematical specification

## Experimental unit and intervention

Let \(s_i\) be a fully controlled simulator branch state and let
\(a_{ij}^{1:H}\), \(j=1,\ldots,A\), be executable alternative action sequences.
The simulator produces

\[
y_{ijh}=F_h(s_i,a_{ij}^{1:h}),
\]

and the learned world model produces

\[
\hat y_{ijh}=\hat F_h(o(s_i),a_{ij}^{1:h}).
\]

Evaluation occurs in a declared feature map \(\phi\). In Stage 1, \(\phi\) is the
same frozen DINOv2 patch representation used by DINO-WM. Later stages must also
report task-state readouts so a favorable encoder cannot hide physical error.
The independent sampling unit is \(s_i\), never an individual action pair.

## Ordinary rollout error

For feature dimension \(D\),

\[
E^\text{ord}_{ih}
=
\sqrt{
\frac{1}{AD}
\sum_{j=1}^{A}
\|\hat z_{ijh}-z_{ijh}\|_2^2
},
\quad
z=\phi(y).
\]

## Paired counterfactual effect error

For every unordered action pair \(j<k\),

\[
\Delta z_{ijkh}=z_{ijh}-z_{ikh}, \qquad
\widehat{\Delta z}_{ijkh}=\hat z_{ijh}-\hat z_{ikh}.
\]

The primary paired metric is

\[
E^\text{cf}_{ih}
=
\sqrt{
\frac{1}{\binom{A}{2}D}
\sum_{j<k}
\|\widehat{\Delta z}_{ijkh}-\Delta z_{ijkh}\|_2^2
}.
\]

Report its normalized form

\[
\widetilde E^\text{cf}_{ih}
=
\frac{E^\text{cf}_{ih}}
{\sqrt{\frac{1}{\binom{A}{2}D}\sum_{j<k}\|\Delta z_{ijkh}\|_2^2}+\epsilon}
\]

and mean cosine alignment of the predicted and true effect vectors. Low
ground-truth-effect pairs must be flagged rather than allowed to dominate the
normalized score.

## Common-mode versus action-dependent error

Let \(e_j=\hat z_j-z_j\) and \(\bar e=A^{-1}\sum_j e_j\). Then

\[
\operatorname{MSE}_\text{pair}
=
\frac{2A}{A-1}
\frac{1}{AD}\sum_j\|e_j-\bar e\|_2^2.
\]

Thus the paired score is not mystical “causal understanding.” It is the
action-dependent part of prediction error after removing a same-state,
same-horizon common-mode component. This algebraic identity is implemented and
unit-tested. A model may have poor ordinary error but faithful action effects,
or good ordinary error with an action-insensitive common future.

## Action ranking and regret

Let \(c_{ijh}\) be the simulator task cost and \(\hat c_{ijh}\) the model-implied
cost. The model selects \(\hat j=\arg\min_j\hat c_{ijh}\).

- Top-1 accuracy: the selected action is simulator-optimal, allowing declared
  numerical ties.
- Pairwise accuracy: fraction of non-tied action pairs ordered correctly.
- Regret:
  \[
  R_{ih}=c_{i\hat jh}-\min_j c_{ijh}.
  \]
- Normalized regret divides by the within-set cost range.

Stage 1 uses a one-shot candidate-set decision, not a full planner. Stage 2 must
add planner-generated candidate sets and execute the selected candidate in the
simulator.

## Interpretation boundary

The intervention is on the simulator action input while holding the simulator
state fixed. This establishes controlled simulator ground truth. It does not
identify real-world counterfactuals and does not establish sim-to-real
reliability.

