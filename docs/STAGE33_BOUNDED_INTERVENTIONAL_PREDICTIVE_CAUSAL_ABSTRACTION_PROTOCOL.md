# Stage 33: bounded interventional predictive causal abstraction

## Status and decision

**Protocol ID:** `stage33-bounded-interventional-predictive-causal-abstraction-v3`

**Evidence status:** prospective, source-bound pilot. Smoke output is never
scientific evidence. The pilot is confirmation-eligible only if every split,
asset, source, cache, and evaluation-lock check below passes before the locked
evaluation is opened.

### V3 model-interface amendment

The source-bound v2 pilot
(`stage33_bipca_result_bundle_0e63f7cb92cd`) completed the repaired physical
selection and all 160 physical-truth records. It loaded the exact pinned JEPA
checkpoint and executed the first real construction prediction, then stopped
before fitting any grounded decoder because the notebook flattened the native
proprio output into a block frozen at 64 entries. All compact-manifest and
selection-sidecar hashes matched. No decoder, rank, operator, mode choice,
cross-model map, interchange effect, planning result, p-value, or scientific
gate was computed. Its status is `INCONCLUSIVE_PIPELINE_FAILURE`.

Inspection of the pinned public implementation resolves the contract without
using outcome values. Both PushT checkpoints use feature-conditioned
proprioception. The target encoder repeats one global latent proprio feature
over the 16-by-16 visual patch grid, and the predictor returns that field as
`[time, 256 patches, channels]`: 16 channels for JEPA-WM and 20 for DINO-WM.
Flattening therefore produces 4,096 or 5,120 entries even though the semantic
proprio feature is only 16- or 20-dimensional.

V3 freezes spatial mean pooling over exactly 256 patches, preserving the
native channel vector before padding it to the unchanged 64-coordinate
readout block. This is the architecture-aligned invariant summary of a global
feature-conditioned signal. Merely increasing the pad to 5,120 would create a
model-asymmetric high-dimensional decoder and is rejected. V3 also requires a
one-word real construction preflight for each checkpoint that verifies the
complete visual, proprio-field, pooled-readout, and carrier shapes and
finiteness before any fitted artifact is reused or created. The preflight
records no scientific metric and uses no model-selection, calibration, or
evaluation row.

V3 otherwise preserves v2's selected physical design, action banks, split
roles, estimands, rank/operator rules, controls, thresholds, and automatic
decision labels. The amendment is explicitly interface-informed but remains
scientific-outcome-blind.

### V2 model-free coverage amendment

The source-bound v1 pilot (`stage33_bipca_result_bundle_5f0ab5e6e24a`) stopped
after 168 seconds because its evaluation candidate pool yielded 15 complete
four-mode physical trajectories rather than the required 16. The bundle bound
commit `f1a59715675c54886e1fade66dbdd8e95e547dcd`; all compact-manifest hashes
matched, `models_loaded` was false, GPU allocation remained zero, and no model
activation, prediction, decoded effect, planning score, or scientific gate was
observed. The v1 status is therefore `INCONCLUSIVE_PIPELINE_FAILURE`.

V2 preserves the 8/8/8/16 complete-trajectory targets, all action banks,
estimands, rank/operator rules, controls, thresholds, and automatic result
labels. It makes candidate geometry a deterministic function of trajectory ID
rather than pool length, expands the four mutually disjoint model-free
candidate pools, logs the selected count after accepting a candidate, and
writes every physical-screen row before raising a coverage exception. This is
a prospective design repair following model-free feasibility information, not
an outcome-informed scientific refit.

A simulator-only preflight of the implemented v2 selector against the exact
pinned `jepa-wms` commit reached construction 8/8 after 209 candidates,
model-selection 8/8 after 197, calibration 8/8 after 277, and evaluation 16/16
after 307 of 1,600 available candidates. The corresponding evaluation IDs were
`8431, 8436, 8448, 8486, 8491, 8541, 8546, 8571, 8596, 8601, 8626, 8648,
8651, 8681, 8703, 8706`. The preflight imported the simulator only and did not
load either world model or inspect any scientific outcome. The source-bound
notebook recomputes and hashes selection independently; these diagnostic IDs
are a coverage audit, not an input manifest or a scientific result.

Stage 33 rejects the proposed **minimal hybrid predictive realization** as the
name of the empirical object. The original proposal combines three claims that
the available experiment cannot identify:

1. equality of a finite collection of terminal conditional means is weaker
   than equality of predictive distributions and does not generally yield a
   Markov quotient;
2. rank of a finite, noisy controlled response table is not the dimension of a
   nonlinear physical mechanism; and
3. a similarity fitted through a common decoded output can be high even when
   two networks use unrelated internal computations.

The replacement is a **bounded interventional predictive causal abstraction
(BIPCA)**. “Bounded” is substantive: the state families, interventions, action
words, horizons, observables, internal sites, and error metric are all finite
and preregistered. “Predictive” refers to complete multi-step traces rather than
only terminal means. “Causal” is earned only by model-native JEPA-to-DINO
additive interchange and transported planning relative to matched controls.
“Abstraction” avoids the unjustified claims of global minimality, unique
physical state recovery, or discovery of the networks' entire mechanism.

The strongest possible positive conclusion is:

> For the two frozen public PushT world models, the preregistered state and
> action families admit a stable low-effective-rank, mode-sensitive predictive
> interface such that one calibration-only coordinate map transports held-out
> operators, internal counterfactual interventions, and planning value from
> JEPA into DINO better than matched controls.

It is not a claim of universal predictive equivalence, globally minimal
physics, unique contact-mode identification, independent convergence across
training seeds, or equality of the networks' full internal algorithms.

## Evidential motivation

The saved Stage 32 bundle reports
`PAIRED_SIGNAL_WITHOUT_SUBSPACE_SPECIFICITY`. The primary paired feature gave
6.49% relative out-of-fold MSE improvement with a state-bootstrap interval for
absolute improvement of `[0.001915, 0.005380]`, and both within-model gates
passed. However, the two matched random rank-128 subspaces gave 6.94% and 6.82%
relative improvement, versus 6.49% for the proposed subspace. Primary minus
median-placebo improvement was negative (`-0.000275`) with interval
`[-0.001143, 0.000613]`. The shuffled-output subspace did not reproduce the
effect.

Thus the physical and planning signal is real enough to investigate, but the
evidence does not isolate a privileged linear carrier. Stages 22--30 also make
contact mode, action order, signed area, contact-frame transport, grounded
closure, and planning value scientifically relevant. Stage 33 uses those
findings to choose regimes and falsification tests; it does not reuse their
evaluation states or treat their selected subspaces as truth.

## 1. Mathematical object

### 1.1 Controlled physical process and genuine interventions

Let

\[
S_{t+1}\sim P(\,\cdot\mid S_t,A_t),\qquad
O_t\sim G(\,\cdot\mid S_t,\eta_t)
\]

be the restored PushT simulator state, executable action, rendered observation,
and controlled rendering nuisance. A state snapshot contains every simulator
quantity needed for exact continuation, including agent and block pose and
velocities. A history is

\[
H_t=(O_{t-L+1:t},A_{t-L+1:t-1}).
\]

For an action word $w=(a_0,\ldots,a_{k-1})$,
`do(A_{t:t+k-1}=w)` means restoring the identical snapshot and replacing the
future action inputs by $w$. It is not shorthand for conditioning on actions
selected by a behavior policy. Failed restoration identity, uncontrolled
randomness, or non-executable clipping makes the affected family ineligible.

One word symbol is one public-world-model transition: five executable PushT
simulator controls under the frozen frame-skip contract. The 11-word core bank
has lengths 1--3. Its primitives are `L=(-30 degrees, 0.14)`,
`R=(+30 degrees, 0.14)`, and `S=(0 degrees, 0.10)`; it contains `L`, `R`, `S`,
`LR`, `RL`, `LL`, `RR`, `LRL`, `RLR`, `LLR`, and `RRL`.

The locked evaluation bank contains 12 composition-disjoint words with
lengths 1--4. Its primitives are `a=(-20 degrees, 0.10)`,
`b=(+20 degrees, 0.22)`, `A=(-40 degrees, 0.18)`, and
`B=(+40 degrees, 0.18)`; the words are `a`, `b`, `A`, `B`, `ab`, `ba`, `AAB`,
`BAA`, `ABB`, `BBA`, `ABAB`, and `BABA`. Zero words of every length and hybrid
interchange words are auxiliary controls. Operator transition shards contain
all unique prefixes: model selection and calibration use the `L`, `R`, and `S`
prefixes, while evaluation uses the genuinely unseen evaluation-word prefixes
through length 4. The maximum horizon is therefore four model transitions, or
20 simulator controls. No Stage 33 result supports a length-5 claim.

### 1.2 Grounded, trace-valued observables

The grounded coordinate at a simulator step contains the exact 11-coordinate
schema

\[
\phi(S_t)=
\bigl(p^a_x/512,p^a_y/512,p^b_x/512,p^b_y/512,
\sin\theta^b,\cos\theta^b,
v^a_x/50,v^a_y/50,v^b_x/50,v^b_y/50,\omega^b/5\bigr),
\]

where $a$ and $b$ denote agent and block. Raw dynamic states are also saved so
normalization cannot conceal an effect. Simulator contact counts and first
contact time define the four physical strata and transition modes, but are not
targets in the grounded decoder. This separation prevents a supplied contact
label from trivially satisfying grounded prediction.

For a history $h$ and word $w\in\mathcal W_{\mathcal B}$, define the
interventional trace law

\[
\mathcal R(h,w)
=\mathcal L\!\left(
(\phi_{t+1},\ldots,\phi_{t+|w|})
\mid H_t=h,\operatorname{do}(w)
\right).
\]

The exact restored simulator makes the implemented target a deterministic
Dirac law. Locked evaluation nevertheless audits simulator determinism by
executing two exact restores for each of `a`, `ab`, `AAB`, and `ABAB` on every
evaluation record and saving the branch count, maximum absolute grounded-trace
difference, and RMSE. It does not run rendering-nuisance replicates or repeated
model forwards. If a future stochastic version is used, the object must remain
the law, not only its mean, and broader repeated sampling plus a
characteristic-kernel MMD or Wasserstein metric would be required.

### 1.3 Exact bounded equivalence

For the fixed domain

\[
\mathcal B=(\mathcal H_{\mathcal B},
\mathcal W_{\mathcal B},\mathcal K,\phi),
\]

define

\[
h\equiv_{\mathcal B}h'
\quad\Longleftrightarrow\quad
\mathcal R(h,w)=\mathcal R(h',w)
\quad\text{for every }w\in\mathcal W_{\mathcal B}.
\]

Provided each interventional kernel is well-defined, this is an equivalence
relation because equality of the indexed response maps is reflexive,
symmetric, and transitive. It says nothing about words outside the bank.

It produces a time-homogeneous Markov quotient only under the additional
**controlled congruence/lumpability condition**: for every allowed primitive
action, equivalent histories must have the same immediate output law and the
same push-forward distribution over successor equivalence classes. With a
finite horizon $K$, the natural objects are time-indexed relations

\[
\equiv_K\;\longrightarrow\;\equiv_{K-1};
\]

one may not silently treat $\equiv_K$ as a stationary sufficient state.
Prefix closure lets Stage 33 test the required recursive consistency, but a
finite test cannot prove it universally.

The proposed terminal-mean relation fails this requirement. For example, a
state producing $Y\in\{-1,+1\}$ equiprobably and a state producing $Y=0$
have the same mean but different laws. If the next transition depends on
$Y^2$, merging them also changes future predictions. Terminal means can
therefore agree while distributional and recursive sufficiency fail.

### 1.4 Approximation is a pseudometric, not a quotient

Let $d_{\mathcal Y}$ be the whitened trace distance (or a distributional
integral probability metric), and let fixed positive weights
$\lambda_{w,j}$ sum to one. Define

\[
d_{\mathcal B}(h,h')^2
=\sum_{w,j}\lambda_{w,j}
d_{\mathcal Y}\!\left(
\mathcal R_j(h,w),\mathcal R_j(h',w)
\right)^2.
\]

Stage 33 estimates this finite pseudometric and an $r$-dimensional embedding
of it. It never defines $h\sim_\epsilon h'$ and calls the result a quotient:
the relation $d(h,h')\leq\epsilon$ is not transitive (points at
$0,0.75\epsilon,1.5\epsilon$ are the elementary counterexample). Approximate
results are reported as errors, distortions, and confidence intervals.

### 1.5 Model-level predictive causal interface

There are exactly two frozen models,

\[
m\in\{J,D\}=\{\texttt{jepa_wm_pusht},
\texttt{dino_wm_pusht}\}.
\]

Let $X_m(h)\in\mathbb R^{256\times d_m}$ be the activation at frozen predictor
block index 4 (the fifth of six blocks), with expected widths 400 for JEPA and
414 for DINO. This exact site is used in both models and no layer search is
performed.

For every recurrent prediction, the native output feature is a deterministic
256-dimensional CountSketch of the visual prediction concatenated with the
spatial mean of the native feature-conditioned proprio field, padded to 64
coordinates. The patch axis is fixed at 256; the pooled channel width is fixed
at 16 for JEPA-WM and 20 for DINO-WM. It is a latent proprio feature, not a
decoded four-coordinate physical state. A model-specific grouped-ridge decoder
is fit on construction trajectories only to predict the 11 grounded
coordinates.
The penalties are selected by trajectory-grouped inner folds from

\[
\{10^{-4},10^{-3},10^{-2},10^{-1},1,10\}.
\]

Concatenating every decoded intermediate step for the 11 core words gives the
finite grounded signature $Q_m(h)$. Construction-only centering, scaling, and
SVD define a predictive chart $z_m(h)$; the common effective rank is capped at
12. Because the chart is fitted through shared physical labels, it establishes
only levels 1--3 until the native intervention tests pass.

The writable carrier is fitted separately from real block-4 action-response
deltas. Each channel is divided by its construction RMS (with a frozen
positive floor), flattened, centered, and factorized in the sample Gram chart.
The leading empirical basis $U_m$ has fixed maximum rank 16; rank below 3 is an
integrity failure. An equal-rank empirical-span basis orthogonal to $U_m$ is
sampled before calibration as the primary random-subspace control. If
$c_m(\Delta X)=U_m^\top\operatorname{white}_m(\Delta X)$, writing an abstract
delta means

\[
\Delta X_m=
\operatorname{unwhite}_m\!\left(U_m c_m\right)
\]

at the same recurrent block and step. Calibration fits a separate within-model
bridge in each network between native carrier-response coordinates and the
common-rank predictive-state effect. Those two bridges are interfaces inside
their respective models, not cross-model transformations. Cross transport is
therefore

\[
c_J\longrightarrow\Delta z_J\longrightarrow
S^*\Delta z_J\longrightarrow\Delta z_D\longrightarrow c_D,
\]

The implemented causal chain stops at DINO: JEPA carrier-response coordinates
are mapped into JEPA predictive effects, transported by $S^*$, converted by the
DINO within-model inverse bridge, and additively written into DINO block 4.
There is no DINO-to-JEPA causal run. Interchange uses frozen base/donor word
pairs and compares the patched DINO base word with the real hybrid word.
Controls are matched on recurrent step and intervention norm; the random
carrier control uses an equal-rank empirical-span basis.

Off-manifold risk is not assumed away. Baseline error, edit norms, a DINO
self-positive edit, and empirical-span, state-permuted, and random-map controls
are reported. No separate ablation or rescue experiment is implemented. A
large but nonspecific disruption is a failure, not evidence of causality.

An $\epsilon$-BIPCA is an empirical statement that, on $\mathcal B$, this
interface has small held-out predictive distortion, approximately closes under
prefix transitions, and makes the interchange diagram approximately commute
in both the model-native and grounded trace metrics. It is not asserted that
$z_m$ is the unique or globally minimal state.

## 2. What realization theory does and does not license

### 2.1 Controlled Hankel block

For an admissible prefix $p=(h,u)$, suffix $v$, and linear trace functional
$f_\ell$, define the finite controlled Hankel block

\[
\widehat{\mathsf H}_m[p,(v,\ell)]
=\widehat{\mathbb E}\left[
f_\ell(Y_m)\mid h,\operatorname{do}(uv)
\right].
\]

Histories, prefixes, suffixes, and all word permutations from one composition
group remain in the same split. The simulator-oracle block replaces $Y_m$
by the grounded trace. Native output sketches and grounded traces are analyzed
separately.

For an exact finite-dimensional linear input-output series with sufficiently
rich prefix and suffix sets, finite Hankel rank is the minimal linear
realization dimension. Reachable and observable minimal LTI realizations are
unique up to an invertible similarity. Analogous statements exist for defined
classes of linear switched or bilinear systems using generalized Hankel
matrices and admissible mode words.

None of those theorems implies that:

- a truncated noisy block identifies the infinite Hankel rank;
- its effective rank is the dimension of nonlinear PushT physics;
- arbitrary neural predictors have finite Koopman-invariant coordinates;
- contact guards and resets are identifiable when mode labels are hidden; or
- two approximate, nonminimal neural interfaces must be related by a linear
  similarity.

Stage 33 therefore reports **effective finite-block rank** with uncertainty and
uses similarity only as a falsifiable empirical hypothesis.

### 2.2 Rank estimation and uncertainty

Rank and chart construction are separated. Construction-only centering and
scaling define each model's chart and preserve the complete singular spectra.
Model selection then locks the common rank; calibration and evaluation never
change it.

1. Standardize every grounded-signature column with construction-only moments.
2. On each model-selection matrix, generate 256 structured null spectra by
   independently exchanging whole equal-length trajectory blocks by column.
   The retained null edge is the componentwise 95th percentile.
3. Generate 512 trajectory-block bootstrap spectra. A direction is retained
   only when its observed singular value exceeds its null edge and at least
   80% of bootstrap singular values exceed that same edge. Directions must be
   consecutive from the leading singular vector.
4. Search only through the frozen predictive-rank cap (12 in the source-bound
   pilot). Both selected ranks must be at least 3 and may differ by no more
   than 2. The common rank is the smaller of the two accepted ranks and is
   locked on model selection. The separate native carrier remains capped at
   rank 16.
5. Save the observed, null, and bootstrap spectra, rank draws, 95% rank
   intervals, and per-direction stability. The Gavish--Donoho white-noise
   threshold may be reported as a sensitivity analysis but is not primary
   because controlled-response rows and columns are correlated.

Rank stability is a necessary lower-level gate, never sufficient evidence of
a shared mechanism.

### 2.3 Controllability, observability, and mode identifiability

Classical minimality requires both reachability/controllability and
observability. In Stage 33, the action bank can expose only directions reached
by its bounded words, and the trace functionals can separate only directions
that affect the chosen native or grounded outputs. A direction that is not
excited or observed is absent from the estimated Hankel rank even if it exists
in the model. Construction reports the smallest singular values of the
empirical reachability and observability factors by regime; a value at or below
the simultaneous null edge prevents minimality language.

Contact labels are not generically identifiable from input-output data.
Identification requires, at minimum, distinct regime responses, sufficient
action excitation, enough entries/releases and dwell, and a mode-partition
class capable of separating them, with labels still identifiable only up to
permutation. True simulator labels therefore define an oracle upper bound. The
claim-eligible label-free variant uses four deterministic model-state clusters
whose centers are locked on model selection; an oracle-only pass establishes
annotated hybrid predictability, not a model-native hybrid state.

### 2.4 Hybrid response operators

Each simulator transition receives exactly one physical regime label from the
contact indicator before and after the transition:

| Regime | Contact pair | Meaning |
|---|---:|---|
| `free` | $0\to0$ | no contact during the transition |
| `pre_contact` | $0\to1$ | contact entry / impact |
| `contact` | $1\to1$ | sustained contact |
| `post_contact` | $1\to0$ | release / reset |

The primary model-specific operator family is affine-bilinear,

\[
z_{t+1}=b_{m,g}+A_{m,g}z_t+B_{m,g}a_t
+\sum_{j=1}^{d_a}a_{t,j}N_{m,g,j}z_t+\xi_t,
\qquad g\in\mathcal G.
\]

The entry and release operators are allowed distinct affine terms and hence can
represent an empirical reset. This is not a theorem that rigid contact is
bilinear. It is the smallest preregistered family that can express the earlier
order/area interactions without hiding all structure inside a black box.

Three versions are compared:

1. one global affine-bilinear operator;
2. an oracle hybrid supplied true simulator contact transitions;
3. a label-free hybrid that assigns each predictive state to the nearest of
   four deterministic centers selected on model-selection trajectories. The
   centers are frozen before calibration and receive no physical contact label
   at calibration or evaluation.

The label-free hybrid is claim-eligible. The physical-label hybrid measures
whether useful contact structure exists at all. The model-selection-frozen
capacity match is a nonlinear control with 208 frozen random Fourier features
of the same state/action input followed by the same grouped-ridge selection.
The width equals
$4\{1+12+3+(3\times12)\}=208$ features at the rank-12 cap; if the selected
rank is smaller it is a conservative over-capacity control. It tests whether a
gain is merely generic nonlinear flexibility.

### 2.5 One cross-model map

After the construction interfaces and the model-selection rank, operator
ridge, mode variant, and capacity are frozen, calibration alone fits one
invertible affine whitened-Procrustes map

\[
T_{J\to D}(z)=S^*z+b^*.
\]

The fit centers and whitens the paired calibration charts, solves one
orthogonal Procrustes problem in whitened coordinates, and recolors into the
DINO chart. Hence $S^*$ need not be orthogonal in the original coordinates.
It is eligible only if its condition number is at most 100 and its smallest
singular value is at least $10^{-3}$. Calibration pools all four strata and
does not use a mode label in the map fit.

The strict fit must meet those nonsingularity diagnostics; a least-squares
fallback is saved for diagnosis but automatically fails the fixed-map gate.
Stage 33 evaluates only $J\to D$ and does not verify or score reverse transport.
No map is refit by action, horizon, state family, metric, intervention type, or
evaluation subset. Frozen calibration-only within-model bridges lift this same
map to native activation deltas; because offsets cancel for a delta, transport
uses the linear part $S^*$. Three action-conditioned maps are reported only as
an ineligible upper-bound control, and no native carrier-to-carrier cross-map
is fitted.

For every regime the numerical conjugacy audit applies the exact affine change
of coordinates to drift, state, action, and bilinear blocks, including the
offset-induced action and drift corrections. The primary quantity is
nevertheless held-out composed-transition error, not a potentially
ill-conditioned parameter norm. For example,

\[
E_{J\to D}^{(k)}=
\frac{\sum_i\|T_{J\to D}(\widehat z_{J,i}^{(k)})-
z_{D,i}^{(k)}\|_2^2}
{\sum_i\|z_{D,i}^{(k)}-\bar z_D^{(k)}\|_2^2}.
\]

The metric definition is frozen in advance; its target center is the locked
evaluation DINO target mean. No $D\to J$ score is computed.

## 3. Exact and approximate hypotheses

The following are nested findings, not synonyms.

1. **Common decodability:** construction-only decoders predict the same
   grounded variables from both models on held-out families.
2. **Similar effective rank:** finite controlled Hankel blocks have compatible,
   stable above-noise rank.
3. **Similar predictive subspaces:** paired abstract coordinates align on
   calibration/evaluation states.
4. **Held-out operator conjugacy:** one affine $T^*$ approximately carries JEPA
   global/hybrid operators and unseen compositions into DINO coordinates.
5. **Counterfactual interchange:** the same $T^*$, lifted through the frozen
   within-model read/write interfaces, additively transports JEPA response
   effects into DINO and produces the intended grounded multi-step effect.
6. **Planning transport:** the same JEPA-to-DINO additive transport preserves
   physical planning value on unseen candidate sets.

The exact null and alternative at level 4 are

\[
H_0^{\mathrm{op}}: \text{no single affine }T\text{ beats the preregistered matched controls
on locked families},
\]

versus the bounded exact ideal

\[
H_{1,0}^{\mathrm{op}}:\exists T(z)=Sz+b\text{ with invertible }S\text{ for which every}
\quad\text{preregistered operator and composed response commutes exactly.}
\]

Finite precision makes exact equality untestable. The operative alternative
$H_{1,\epsilon}^{\mathrm{BIPCA}}$ is the conjunction of the numerical rank,
hybrid, one-map, JEPA-to-DINO causal-interchange, planning, and specificity
gates in Section 7. Levels 1--3 alone explicitly do not support a
shared-mechanism claim.

## 4. Frozen assets and the shared-target confound

Stage 33 uses only the two public PushT world-model checkpoints released in
the official JEPA-WMs repository. They are not training-seed replications.

| Asset | Frozen identity |
|---|---|
| JEPA-WMs source | git commit `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0` |
| Hugging Face snapshot | revision `9b9c41ef249466630dbf1a20e78391865d07b3b9` |
| `jepa_wm_pusht.pth.tar` | SHA-256 `9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb` |
| `dino_wm_pusht.pth.tar` | SHA-256 `8ec9cb05f22812d7f12e3c216b0637f41641055c0653e503e2746edb981b550f` |
| shared `dinov2_vits14_pretrain.pth` | SHA-256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9` |

The DINOv2 file is a shared visual encoder/target asset, **not a third world
model**. Both public world models are trained and evaluated against DINOv2
representation targets. Consequently, matching output coordinates, decoded
physics, effective ranks, or even an identity-like output map can be induced by
the shared target. Stage 33 cannot remove that experimental confound with the
available public inventory. It mitigates it by requiring effects at the two
models' native pre-output sites, a JEPA-sourced additive intervention at the
DINO site, unseen action compositions, and physical planning. It does not
include an independent-target baseline, ablation/rescue, or reverse causal
direction. Even a full pass remains a two-checkpoint, common-target result;
“independently converged representation” would be too strong.

Loading any different checkpoint, manufacturing pseudo-checkpoints by weight
perturbation or truncation, changing the DINOv2 target, or accepting a hash
mismatch makes the run ineligible rather than expanding the comparison.

## 5. Prospective design and leakage barriers

### 5.1 Four disjoint splits

The pilot selects 40 complete simulator trajectories. A complete trajectory is
the state family and contributes exactly four restored records: `free`,
`pre_contact`, `contact`, and `post_contact`. Its four records, all model
forwards, action branches, word prefixes, intervention pairs, nuisances, and
planning goals stay in the same split. The resulting literal design is 160
records, not 160 independent trajectories.

| Split | Candidate trajectory IDs | Complete trajectories | Four-mode records | Permitted use |
|---|---:|---:|---:|---|
| construction | `[6000,6800)` | 8 | 32 | fit normalization, grouped-ridge decoders, predictive charts, native carrier bases, and structured null machinery |
| model selection | `[6800,7600)` | 8 | 32 | lock the common rank, operator ridges, global/physical/label-free variant, and nonlinear-control capacity |
| calibration | `[7600,8400)` | 8 | 32 | refit final frozen operators and within-model bridges, then fit exactly one cross-model affine map with every choice fixed |
| locked evaluation | `[8400,10000)` | 16 | 64 | one sealed opening for all primary metrics, controls, intervals, and the mechanical decision |

The smoke plumbing check uses 1/1/1/2 complete trajectories, or 4/4/4/8
records, and is never scientific evidence. Exact selected IDs, record IDs,
mode counts, and hashes are committed before model outcomes are opened.

A candidate trajectory is accepted only if physical screening finds all four
mode snapshots. Screening may use simulator geometry, exact contact timing,
and restoration validity, but never activations, predictions, decoded effect
magnitudes, planning success, or any Stage 33 outcome. Previous Stage 14--32
evaluation IDs are forbidden. Locked evaluation targets 16 distinct complete
trajectories and therefore 16 records in every stratum; at least 12 distinct
evaluation trajectories and at least 10 trajectories per stratum are required
for an evidence-eligible result. Inadequate coverage is inconclusive, not
negative evidence.

Candidate phase is generated from an absolute golden-angle sequence keyed by
trajectory ID, and distance is generated from a separate deterministic
trajectory-ID slot. Neither depends on candidate-pool length. Expanding a pool
therefore adds candidates without changing the geometry of existing IDs.

### 5.2 Action words and composition holdout

Construction, model selection, and calibration use the fixed 11-word core bank
listed in Section 1.1. Locked evaluation uses the 12-word bank with new
primitive angles, magnitudes, words, and compositions. Core words may still be
run on evaluation records to anchor frozen decoders and charts, but they are
not evidence for unseen composition. The primary generalization test is the
evaluation bank and its actual prefix rollouts. Maximum word length is four.

The calibration interchange pairs are `LR/RL` at step 0, `LLR/RLL` at step 0,
`RRL/LRR` at step 0, and `LRL/RLR` at step 1. The evaluation pairs are
`ab/ba` at step 0, `AAB/BAA` at step 0, `ABB/BBA` at step 0, and
`ABAB/BABA` at step 1. In every pair, replacing the selected donor prefix of
the base defines a real hybrid word whose unpatched rollout is the intended
counterfactual. Zero words of lengths 1--4 provide identity/no-action controls.

For the executable simulator-control sequence
$u_0,\ldots,u_{5|w|-1}$, the frozen action summaries are

\[
I(w)=\sum_i u_i,\qquad
E(w)=\sum_i\|u_i\|_2^2,\qquad
\mathcal A(w)=\sum_{0\leq i<j<5|w|}\det(u_i,u_j).
\]

Fixed-multiset order pairs preserve duration, component multiset, total
impulse, and energy while changing order and signed area. Every word value,
prefix, pair, zero control, invariant, and split assignment is written to a
hashed manifest before model evaluation and is never regenerated in response
to outcomes.

### 5.3 Freeze sequence

The notebook enforces the following state machine:

1. bind source, checkpoints, simulator, trajectory lists, and action banks;
2. select and hash complete physical trajectories without model outcomes;
3. open construction; fit scales, grouped-ridge decoders, predictive charts,
   native carrier bases, and structured rank-null reference distributions;
4. open model selection once; freeze the common rank, operator ridge penalties,
   global/physical/label-free operator variants, nonlinear-control capacity,
   and every metric definition;
5. open calibration once; refit final operators, fit the two within-model
   carrier-response/predictive-effect bridges, and fit exactly one affine
   cross-model map $T^*$ from JEPA to DINO;
6. write `evaluation_open_certificate.json` containing hashes of all frozen
   objects;
7. open locked evaluation exactly once; compute all methods and controls; and
8. derive the result label mechanically from Section 7.

An evaluation shard cannot be used to return to an earlier state. A changed
configuration, source file, checkpoint, split, action bank, frozen object, or
metric implementation requires a new nonce and is a new experiment.

### 5.4 Empirical references and normalization

The locked realization rows include an oracle error field of zero, the measured
simulator-determinism RMSE from the two-restore probe, and a decoder-only
mean-target baseline. The determinism probe covers `a`, `ab`, `AAB`, and
`ABAB`; it is not a repeated-model-forward or rendering-nuisance estimate.
Relative squared prediction error is

\[
E(\hat z,z;\bar z)=
\frac{\operatorname{mean}_j(\hat z_j-z_j)^2}
{\max\{\operatorname{mean}_j(z_j-\bar z_j)^2,\tau_z\}},
\]

where $\bar z$ is the locked-evaluation target mean and
$\tau_z=\max(10^{-6}\operatorname{median}D,10^{-12})$ for the evaluated
denominators $D$. This center is used only to score the already frozen methods;
it is not used to fit or select them. Planning separately reports exact
simulator normalized regret with oracle regret zero.

## 6. Model-native additive transport and planning

### 6.1 Positive and reliability controls

Construction trajectories are split into two deterministic trajectory halves,
independent same-model predictive charts are fitted, and a calibration-only
affine map aligns them. Locked-evaluation relative RMSE is reported for JEPA
and DINO as an interface-reliability diagnostic; this is not a causal
interchange, ablation, or rescue assay.

The causal evaluation also includes a `dino_self_positive` condition. It
reconstructs DINO's own carrier response for the same base/donor/hybrid word
triple and additively writes that response into DINO. This is the attainable
same-model positive control for the local patch. No separate ablation or
rescue experiment is implemented.

### 6.2 JEPA-to-DINO additive interchange

Let $G_J$ be the frozen within-JEPA bridge from carrier response to predictive
effect, and let $G_D^{-}$ be the separately fitted within-DINO reverse bridge.
For a JEPA donor/base carrier contrast $\Delta c_J$, the DINO write is

\[
\Delta c_D^{\mathrm{edit}}=G_D^{-}
\!\left(S^*G_J(\Delta c_J)\right),\qquad
X_D^{\mathrm{edit}}=X_D^{\mathrm{base}}+
\operatorname{unwhite}_D(U_D\Delta c_D^{\mathrm{edit}}).
\]

The affine cross-map offset is omitted because the transported object is an
effect contrast. The patched DINO base word is compared with the exact
simulator trace for the real hybrid word. A secondary self-consistency score
compares it with DINO's own unpatched hybrid-minus-base decoded effect. There
is no DINO-to-JEPA causal evaluation and no second carrier-to-carrier
cross-model map.

For an intended grounded effect $\Delta^*$ and observed decoded effect
$\Delta$, Stage 33 records

\[
\operatorname{cos}(\Delta,\Delta^*)
=\frac{\langle\Delta,\Delta^*\rangle}
{\|\Delta\|\,\|\Delta^*\|},\qquad
\operatorname{nerr}=\frac{\|\Delta-\Delta^*\|}{\|\Delta^*\|}.
\]

A row is eligible only if $\|\Delta^*\|^2\ge10^{-6}$. The primary error gain
is the relative decrease in decoded DINO path MSE to the physical hybrid target
versus the unpatched DINO base path. The implemented causal conditions are the
primary map, state-permuted map, random orthogonal map, equal-rank
empirical-span random carrier, and DINO self-positive edit. The three matched
cross-model controls are norm-matched to the primary write. The intervention
is native because it is additively applied at DINO predictor block 4; the
primary scored response is the grounded decoded path, not a separate raw native
output-effect metric.

### 6.3 Additive planning transport

For every one of the 12 evaluation candidates, the notebook takes JEPA's
response relative to the zero word of the same length, transports each
step-specific response through $G_J\to S^*\to G_D^{-}$, and additively writes
the resulting deltas into a DINO zero-word rollout. This synthesizes one
transported DINO grounded endpoint per candidate. State-permuted,
random-orthogonal-map, and random-carrier conditions use the same procedure and
norm matching.

Each evaluation record uses four deterministically sampled goals drawn from
the exact physical candidate endpoints. Candidate selection uses decoded
grounded endpoints. Its cost is

\[
c(a;g)=\|p^b_a-p^b_g\|_2^2
+0.1\|[\sin\theta^b_a,\cos\theta^b_a]
-[\sin\theta^b_g,\cos\theta^b_g]\|_2^2.
\]

The same cost applied to exact simulator endpoints determines normalized
physical regret

\[
R=\frac{c^{\mathrm{phys}}_{\hat a}-\min_a c^{\mathrm{phys}}_a}
{\max_a c^{\mathrm{phys}}_a-\min_a c^{\mathrm{phys}}_a+\epsilon_R}.
\]

The assay compares unpatched DINO-native decoded planning, primary additive
JEPA-to-DINO transport, the three transported controls, and oracle regret zero.
The grounded decoder therefore mediates candidate selection; this is not the
official native visual/proprio planner and does not establish decoder-free
planning transport.

## 7. Preregistered inference and gates

### 7.1 Inference unit and multiplicity

The resampling and cross-fitting unit is the entire trajectory/state family.
No action row, prefix, horizon, or source/base pair is treated as independent.
All primary intervals use 5,000 trajectory-cluster bootstrap draws with the
fixed bootstrap seed and retains all rows from each sampled trajectory. Five
preregistered one-sided sign-flip hypotheses--JEPA hybrid gain, DINO hybrid
gain, map-control advantage, interchange-control advantage, and
planning-control advantage--receive Holm step-down correction at familywise
$\alpha=0.05$. The percentile intervals themselves are ordinary 95%
trajectory-cluster intervals. Both pooled and family means are saved.

### 7.2 Implemented numerical gates

“CI” below means the 95% trajectory-cluster percentile interval. Individual
gates are Boolean in code; `partial_pass` is defined only for their aggregate
in Section 7.3.

| Gate | Implemented pass condition | Failure interpretation |
|---|---|---|
| **I. Integrity/coverage** | every source/checkpoint/design/shard hash and freeze transition passes; pilot targets are 8/8/8/16 complete trajectories and 32/32/32/64 records; evaluation contains at least 12 trajectories; no evaluation refit | a pipeline failure is operationally inconclusive, not negative scientific evidence |
| **D+R. Decoder and rank** | both construction-fitted decoders have locked-evaluation median coordinate-wise grounded $R^2\ge0.20$; both selected ranks are at least 3 and differ by at most 2; selection uses the 95% structured-null edge and 0.80 bootstrap stability rule with cap 12 | the implemented bounded predictive chart is not sufficiently grounded and rank-compatible |
| **H. Hybrid structure** | for each model and evaluation words of length at least 2: physical-mode hybrid mean gain over global $\ge0.05$, CI lower bound $>0$, label-free gain at least 50% of physical gain, every discovered cluster nonempty, the decoder-only mean and 208-feature nonlinear controls are worse, and every word-family and physical-mode mean gain is positive; both Holm-adjusted hybrid hypotheses reject | the mode split lacks held-out value, is label-dependent, or is reproduced by a trivial decoder mean or the cap-matched/overmatched nonlinear comparator |
| **M. One-map validity** | the sole JEPA-to-DINO calibration map used a strict whitened fit, has condition number $\le100$ and minimum singular value $\ge10^{-3}$, and the calibration-lock hash is unchanged | a fallback/ill-conditioned fit or changed lock defeats the fixed-map claim; no reverse-map test is part of this gate |
| **S. Same-model positive control** | both models' construction split-half relative RMSE values are $\le0.35$, and the DINO self-positive edit has mean eligible self-consistency cosine $\ge0.20$ against DINO's native hybrid-minus-base effect | the implemented interface is not reliable enough, or DINO cannot reproduce its own local carrier response; the cross-model assay is operationally inconclusive rather than negative evidence |
| **C. Held-out conjugacy** | on evaluation words of length at least 2: mean JEPA-to-DINO operator-conjugacy relative error $\le0.35$; primary map error is at most $1.25\times$ the action-conditioned-map error; mean advantage over the better state-permuted/random-map control is $\ge0.10$ with CI lower bound $>0$; every word-family primary error is $\le0.70$; the Holm-adjusted map hypothesis rejects | one fixed forward map is worse than the stated tolerances or no better than matched/local controls |
| **X. Additive interchange** | mean eligible grounded cosine $\ge0.20$; mean grounded hybrid-target error gain $\ge0.10$; primary gain minus the best state-permuted/random-map/random-carrier gain has CI lower bound $>0$; every physical-mode and interchange-pair mean gain is positive; the Holm-adjusted interchange hypothesis rejects | the JEPA-sourced additive DINO patch is inaccurate, nonspecific, or family-inconsistent |
| **P. Additive planning transport** | the CI upper bound for primary-minus-DINO-native normalized regret is $\le0.02$; primary regret advantage over the best transported control has CI lower bound $>0$; every mode's mean degradation is $\le0.04$; the Holm-adjusted planning hypothesis rejects | transported decoded planning is inferior beyond the margin or indistinguishable from a control |
| **F. Family consistency** | both models' hybrid gains are positive in every implemented word family, and JEPA-to-DINO interchange gain is positive for every frozen evaluation pair | a pooled result is driven by a family reversal |
| **N. Control specificity** | map mean control advantage $\ge0.10$; interchange and planning mean control advantages are positive; physical hybrid beats action reversal and mode permutation in both models; the zero-edit check has maximum absolute error $\le10^{-6}$ | the main chain is reproduced by a preregistered matched control, or the nominal zero intervention is not numerically neutral |

### 7.3 Automatic result labels

The numerical core stores a nested `protocol_decision.status` of `pass`,
`partial_pass`, or `fail`. The top-level emitted status is exactly:

- `BOUNDED_INTERVENTIONAL_PREDICTIVE_CAUSAL_ABSTRACTION_SUPPORTED` when an
  eligible pilot passes D+R, H, M, S, C, X, P, F, and N. Only this status
  supports the bounded, forward-direction claim at the top.
- `OPERATOR_CONJUGACY_WITHOUT_FULL_CAUSAL_PLANNING_CERTIFICATE` when D+R, H, M,
  S, C, F, and N pass but the full X/P conjunction does not. It supports a
  bounded operator-conjugacy result, not a full causal-planning certificate.
- `INCONCLUSIVE_SAME_MODEL_POSITIVE_CONTROL_FAILURE` when either split-half
  reliability threshold or the DINO self-positive threshold fails. This takes
  precedence over a negative scientific label because the implemented local
  interface has not shown the required attainable response.
- `BOUNDED_SHARED_ABSTRACT_MECHANISM_NOT_SUPPORTED` for any weaker completed
  pilot whose same-model positive-control gate passes. Lower-level decoder or
  rank findings may still be reported.
- `SMOKE_ONLY` for every smoke run, regardless of favorable metrics.

An execution exception may write `INCONCLUSIVE_PIPELINE_FAILURE`; that is an
operational failure trace rather than a scientific result. No manual narrative
can override the mechanically derived status.

## 8. Required controls and what each falsifies

| Control | Frozen implementation | What a match would show |
|---|---|---|
| Action reversal | reverse each evaluation action sequence for the global operator | held-out hybrid advantage does not depend on action order |
| State permutation | fit a calibration control map after rotating trajectory identity while preserving the four-mode pairing | paired cross-model state correspondence is unnecessary |
| Mode-label permutation | cyclically relabel the four physical modes at evaluation | hybrid gain is not contact-structure-specific |
| Random matched-dimensional subspace | construction-frozen equal-rank empirical-span basis orthogonal to $U_m$, with causal deltas norm-matched to primary | another reachable carrier works equally well, repeating the Stage 32 specificity failure |
| Random orthogonal map | seeded orthogonal map frozen before evaluation, with the calibration means used only for its offset | a fitted coordinate map is no better than gauge-random alignment |
| Same-model split half | independent predictive charts on deterministic construction-trajectory halves plus a calibration-only held-out map | reports finite-data interface reliability; it does not validate causal patching |
| Decoder-only mean | evaluation target-center predictor reported in realization rows | a trivial center explains the predictive-chart error |
| One global operator | one affine-bilinear operator across all transitions | explicit switch/reset structure is unnecessary |
| Physical-label hybrid | true simulator contact-transition labels supplied | distinguishes absent hybrid structure from failure of label-free mode discovery |
| Capacity-matched nonlinear control | 208 model-selection-frozen random Fourier features, equal to $4\{1+12+3+3(12)\}$ at the rank cap, followed by grouped ridge | predictive gain reflects generic nonlinear capacity rather than mode-conditioned operator structure; it is over-capacity if selected rank is lower |
| Action-conditioned maps | three calibration-only maps fitted on `L`, `S`, and `R` first-step targets, reported solely as an upper bound | local action-conditioned alignment is easier than one map; these maps are never used for causal transport |
| DINO self-positive edit | DINO's own carrier response for the same interchange triple is additively patched into DINO | establishes the attainable same-model local patch response |
| No explicit contact labels | four model-state clusters selected on model selection; frozen nearest-center assignments at calibration/evaluation | the result survives when physical mode annotations are unavailable to the model |
| Simulator oracle / determinism floor | exact physical endpoints plus two restored rollouts for `a`, `ab`, `AAB`, and `ABAB` on every evaluation record | anchors planning regret and measures exact-restore numerical variability |

Matched comparisons within each assay use the same eligible rows. There is no
independent-target or dedicated shared-DINOv2-output baseline, so the shared
target remains an explicit unresolved confound rather than a rejected control.

## 9. Adversarial counterexamples and design responses

### Counterexample 1: same means, different stochastic process

Let model 1 predict $Y=\pm1$ with equal probability and model 2 predict
$Y=0$. Both terminal means are zero. A terminal-mean signature declares them
equivalent, but a variance-sensitive future or cost distinguishes them.

**Response:** define full trace laws; in deterministic Stage 33 retain every
intermediate coordinate and contact event. Means are an explicitly secondary
sensitivity analysis.

### Counterexample 2: finite-delay disagreement

Two controlled systems can agree for every word of length at most four and
differ only after a five-step delay state becomes observable. Every Stage 33
test then passes although universal equivalence is false.

**Response:** make the word and horizon bound part of the claim. Multi-length
prefix closure tests recursion inside the bound, but no text may replace
“bounded” by “for all actions or horizons.”

### Counterexample 3: common decoder with unrelated nuisance dynamics

Let

\[
X_J=(p,n_J),\qquad X_D=(Tp,n_D),
\]

where $p$ is a decoded physical label and the nuisance processes $n_J,n_D$
are arbitrary and causally dominate the networks. A map $T$ between the
decoded coordinates gives perfect rank and output conjugacy without a shared
internal mechanism.

**Response:** count this only as levels 1--3; require the JEPA-derived additive
edit at DINO's native pre-output site to improve the real grounded hybrid
target and decoded planning relative to state-permuted, random-map, and random
carrier controls. Because this remains one-directional and decoder-mediated,
even a pass is narrower than full mechanism identity.

### Counterexample 4: one coordinate system per action

Suppose

\[
A_{D,a}=S_aA_{J,a}S_a^{-1}
\]

for every action, but no single $S$ works for two actions. Independently
fitted action maps make every local comparison look conjugate.

**Response:** fit one affine $T^*$ without action or mode labels on calibration,
reuse it for every tested JEPA-to-DINO horizon, and report the three
action-conditioned maps only as an ineligible upper bound.

### Counterexample 5: the common DINOv2 target

Two unrelated predictor circuits can both learn lookup tables into the same
DINOv2 target chart. Output identity, grounded decoding, and planning can then
look similar without independently learned latent physics.

**Response:** use distinct native internal sites and require causal edits before
the output projection. The confound is mitigated, not eliminated; independent
target encoders or training seeds remain necessary for a convergence claim.

### Counterexample 6: global averaging hides contact

If free and contact transitions have operators $A_f\neq A_c$, a dataset with
fixed occupancy can give both models the same least-squares average
$\pi A_f+(1-\pi)A_c$. Global conjugacy can pass while impact behavior differs.

**Response:** use entry, sustained-contact, release, and free operators; test
unseen mode sequences and fixed-multiset contact timing; require label-free
hybrid advantage.

### Counterexample 7: padded nonminimal realizations

Both models may contain a common observable subsystem plus different
uncontrollable or unobservable padding. The minimal observable quotients are
isomorphic even though the full networks are not.

**Response:** this is not a failure of BIPCA. The permitted claim concerns a
bounded causal interface, not the whole network. It is precisely why “same
internal algorithm” and “minimal physical mechanism” are forbidden.

### Counterexample 8: off-manifold patch artifacts

An activation swap can move a network into an unsupported region and cause a
large effect that happens to align with a target direction.

**Response:** use empirical-span writes, norm-matched controls, a zero-edit
identity check with maximum error $10^{-6}$, and the DINO self-positive patch.
These diagnostics reduce but cannot prove away every off-manifold concern; no
ablation/rescue assay is present.

### Counterexample 9: memorizing a finite response table

A sufficiently flexible black box can memorize every inspected state/action
pair and yield a low evaluation error if near-duplicates cross the split.

**Response:** split entire trajectories, state families, and action-composition
orbits; hold out primitive angles and magnitudes; compare the frozen
random-feature nonlinear control; open evaluation only after every fit is
frozen.

## 10. Interpretation of major failure outcomes

- **Grounded decoder fails:** the frozen outputs do not support the declared
  physical signature at the required fidelity. Rank, operator, and map scores
  in that chart cannot support a grounded BIPCA claim.
- **Unstable or high effective rank:** the sampled response block does not
  support a robust low-rank approximation at this power/noise level. This does
  not prove that the true system is infinite-dimensional; it defeats the
  Stage 33 low-rank claim.
- **JEPA and DINO rank intervals differ:** the two interfaces do not have
  compatible finite-block complexity. Common physical decodability remains
  possible.
- **Global equals or beats hybrid:** the preregistered contact split adds no
  predictive value, the operator family is misspecified, or the sampled bound
  is too narrow. Do not claim hybrid contact realization.
- **Physical-label hybrid passes but label-free clustering fails:** contact
  structure matters physically but is not identifiable from the chosen
  model-native state and action without external labels.
- **Only explicit-label evaluation passes:** the abstraction is an annotated
  system-identification model, not a model-native causal mechanism.
- **Per-action maps pass but one map fails:** the models have local,
  action-dependent alignments rather than one shared coordinate change.
- **The affine map is ill-conditioned:** apparent correspondence relies on a
  nearly singular calibration chart and is not claim-eligible.
- **A within-model bridge fails:** the predictive chart may align across
  models, but the experiment has not connected that chart reliably to the
  writable native carrier in the failing model. Cross-model causal transport
  is therefore untestable, not negative evidence about operator similarity.
- **One-step passes but longer words fail:** apparent similarity is local and
  does not compose; recurrent errors or mode switches break closure.
- **Free motion passes but contact entry/release fails:** restrict any result to
  smooth dynamics and abandon shared contact-physics language; the implemented
  top-level status remains fail/partial rather than emitting a special smooth
  label.
- **Operator conjugacy passes but additive interchange fails:** a bounded
  input-output coordinate map exists, but the JEPA-sourced DINO patch does not
  establish the implemented causal level.
- **DINO self-positive or split-half reliability fails:** the patch/interface
  machinery lacks its required positive control. The notebook emits
  `INCONCLUSIVE_SAME_MODEL_POSITIVE_CONTROL_FAILURE`, not negative cross-model
  evidence.
- **DINO self-consistency is positive but the grounded hybrid target fails:**
  the patch moves DINO in one of its own response directions without realizing
  the intended physics.
- **Grounded decoding passes but additive interchange fails:** shared labels or
  the common DINOv2 target may explain the lower-level result.
- **Interchange passes but planning transport fails:** the shared causal
  response does not preserve candidate ordering under the implemented decoded
  grounded planner.
- **Planning is noninferior but a transported control matches:** transport is
  harmless, not specifically valuable.
- **Random subspaces match the complete chain:** the Stage 32 broad-redundancy
  explanation survives; abandon privileged-interface claims.
- **Decoder-only mean performs well:** the sampled predictive chart has weak
  state-specific structure relative to its center baseline.
- **Zero edit changes DINO by more than $10^{-6}$:** the intervention plumbing
  is not identity-preserving, so the specificity gate fails.
- **Unseen compositions alone fail:** the learned map interpolates inspected
  action structure and cannot support the bounded generalization claim.

## 11. Literature adjacency and novelty audit

Only primary papers are used in this table. “Established” describes the cited
result's actual scope, not a theorem transferred to Stage 33.

| Area and primary source | What is established | What it does not establish here | Stage 33 relation |
|---|---|---|---|
| Ho & Kalman, [Effective construction of linear state-variable models from input/output functions](https://doi.org/10.1524/auto.1966.14.112.545) | construction of finite-dimensional LTI realizations from Markov parameters; minimal reachable/observable systems are related by similarity | noisy finite blocks, nonlinear neural networks, hidden contact guards | motivates the controlled-Hankel and similarity ideal only |
| Jaeger, [Observable Operator Models for Discrete Stochastic Time Series](https://doi.org/10.1162/089976600300015411) | observable-operator representation and learning for linearly dependent stochastic processes | controlled contact dynamics or internal neural causality | motivates distribution-valued observable state |
| Littman, Sutton & Singh, [Predictive Representations of State](https://proceedings.neurips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html) | action-conditional multi-step predictions can form linear PSRs; finite POMDPs admit bounded PSR dimension | that a finite terminal-mean bank is sufficient or minimal | motivates action-conditioned trace tests |
| Boots, Siddiqi & Gordon, [Closing the learning-planning loop with predictive state representations](https://doi.org/10.1177/0278364911404092) | spectral PSR learning and downstream planning under stated sampling/rank assumptions | cross-network mechanistic identity | planning is required rather than inferred from rank |
| Sussmann, [Existence and Uniqueness of Minimal Realizations of Nonlinear Systems](https://doi.org/10.1007/BF01683278) | nonlinear realization uniqueness under analytic/accessibility/observability hypotheses, with nonlinear state isomorphisms | a global linear similarity for arbitrary learned/contact systems | blocks transfer of the LTI similarity theorem to the original proposal |
| Petreczky, [Realization theory for linear switched systems](https://doi.org/10.1016/j.sysconle.2007.04.006) and Petreczky, Bako & van Schuppen, [discrete-time linear switched systems](https://arxiv.org/abs/1103.1343) | generalized Hankel rank, minimality, and isomorphism for specified linear switched-system classes and admissible mode words | generic hybrid guards, endogenous unknown modes, nonlinear impact/reset physics | justifies a falsifiable switched-operator comparison, not a theorem about PushT |
| Mu et al., [Persistence of Excitation for Identifying Switched Linear Systems](https://doi.org/10.1016/j.automatica.2021.110142) | sufficient excitation conditions for uniqueness of switched-linear parameter sets | identifiability from a narrow or confounded action bank | motivates composition holdout and explicit excitation coverage |
| Proctor, Brunton & Kutz, [Dynamic Mode Decomposition with Control](https://doi.org/10.1137/15M1013857), and Brunton et al., [Koopman invariant subspaces](https://doi.org/10.1371/journal.pone.0150171) | data-driven controlled linear operators; exact finite Koopman representations require suitable invariant observables | that such a finite invariant subspace exists for contact or learned latents | global Koopman language is not used as an assumption |
| O'Neill, Terrones & Asada, [Koopman global linearization of contact dynamics](https://doi.org/10.1038/s41467-026-72485-7) | a recent constructive global Koopman approach for particular viscoelastic contact models and robot tasks | universality for rigid PushT contact or neural world-model internals | makes the global operator a serious comparator, not the default truth |
| Givan, Dean & Greig, [Equivalence notions and model minimization in MDPs](https://doi.org/10.1016/S0004-3702(02)00376-4) | stochastic bisimulation/lumpability-style partitions preserve specified transitions and rewards | sufficiency of terminal means | supplies the recursive congruence requirement |
| Ferns, Panangaden & Precup, [Metrics for Finite Markov Decision Processes](https://www.cs.mcgill.ca/~prakash/Pubs/Ferns_MetricsForMDPs.pdf) | behavioral metrics relax exact bisimulation in finite MDPs | a transitive epsilon quotient or neural mechanism identification | motivates reporting pseudometric distortion rather than approximate equivalence classes |
| Rubenstein et al., [Causal Consistency of Structural Equation Models](http://auai.org/uai2017/proceedings/papers/11.pdf), and Beckers & Halpern, [Abstracting Causal Models](https://doi.org/10.1609/aaai.v33i01.33012678) | exact intervention-commuting transformations between causal models under allowed-intervention maps | that observational prediction gives a causal abstraction | motivates a diagram tested by internal interventions |
| Beckers, Eberhardt & Halpern, [Approximate Causal Abstractions](https://proceedings.mlr.press/v115/beckers20a.html) | metric treatment of approximate causal-model abstraction, including probabilistic cases | automatic identifiability of a useful abstraction from finite neural data | motivates explicit approximation errors and intervention bounds |
| Geiger et al., [Causal Abstractions of Neural Networks](https://proceedings.neurips.cc/paper/2021/file/4f5c422f4d49a5a807eda27434231040-Paper.pdf) and [Interchange Intervention Training](https://proceedings.mlr.press/v162/geiger22a.html) | interchange interventions test/induce neural causal abstractions; zero IIT loss has an exact result under the specified alignment/intervention domain | that a post-hoc, approximate edit is on-manifold, bidirectional, or exhaustive | motivates the additive internal interchange; the absent ablation/reverse assays remain limitations |
| Ahuja et al., [Interventional Causal Representation Learning](https://proceedings.mlr.press/v202/ahuja23a.html) | latent-factor identifiability under particular perfect/imperfect intervention and support assumptions | those assumptions for pretrained JEPA/DINO activations | prevents generic “interventions imply identifiability” language |
| Hsu, Kakade & Zhang, [A spectral algorithm for learning Hidden Markov Models](https://doi.org/10.1016/j.jcss.2011.12.025) | provable spectral learning under rank/separation conditions | hidden contact-mode recovery without separation/excitation | motivates singular-gap and mode-identifiability diagnostics |
| Gavish & Donoho, [The Optimal Hard Threshold for Singular Values](https://doi.org/10.1109/TIT.2014.2323359) | optimal asymptotic hard threshold under a specified white-noise matrix model | validity for correlated controlled-Hankel rows/columns | retained only as sensitivity; structured shuffle and cluster bootstrap are primary |
| Oquab et al., [DINOv2](https://arxiv.org/abs/2304.07193), and Terver et al., [What Drives Success in Physical Planning with JEPA World Models?](https://arxiv.org/abs/2512.24497) | public visual representation and the two PushT world-model families/checkpoints; latent prediction and planning behavior | independent target representations, mechanistic equivalence, or replicated training seeds | defines the model inventory and exposes the common DINOv2-target confound |
| RAVEL, [Benchmarking interpretability methods through causal interventions](https://proceedings.iclr.cc/paper_files/paper/2025/file/180d2acb13633fe78688d0d2347c731f-Paper-Conference.pdf) | causal intervention benchmarks distinguish localization/decoding from intervention success | controlled hybrid realization in visual world models | adjacent methodological standard for causal, not merely probe-based, evidence |

The ingredients are established: PSRs, finite Hankel realization, switched
system identification, approximate causal abstraction, interchange
interventions, and planning evaluation. The potentially differentiated
contribution is their adversarial combination in frozen visual world models:
one cross-model map must survive composition-held-out hybrid operators,
JEPA-to-DINO additive internal intervention, and simulator-valued decoded
planning. That is a meaningful application and test, not a new general
realization theorem. Its one-directional, decoder-mediated causal assay and
shared target materially limit novelty. With only two common-target checkpoints
and one environment, even a positive pilot is not yet an ICLR-level generality
claim.

## 12. Reproducibility, artifacts, and claim boundary

The implementation notebook is
`notebooks/33_bounded_interventional_predictive_causal_abstraction.ipynb`,
built from
`notebooks/build_stage33_bounded_interventional_abstraction_notebook.py`.
It must print the selected and candidate state-family counts, counts by regime,
word length and composition, paired order controls, model forward-pass counts,
simulator branch counts, cache hits/misses, source/checkpoint hashes, device,
peak memory, and elapsed time before the evaluation lock is opened.

The returned bundle must contain at least:

- configuration, versions, source identity, checkpoint identity, and
  fresh/resumed-run certificate;
- per-checkpoint real output-contract preflights binding visual, spatial
  proprio-field, pooled-readout, and carrier shapes before fitted artifacts;
- trajectory/state-family/action-composition split manifests and design freeze;
- restoration, executable-action, prefix-closure, coverage, and two-restore
  simulator-determinism audits;
- construction/model-selection/calibration/evaluation-open certificates;
- raw singular spectra, all 512 rank-bootstrap and 256 structured-permutation
  draws per reported model/split rank analysis (or lossless sufficient
  summaries), selected ranks, common-rank lock, and null thresholds;
- fitted interface/operator/mode-cluster/within-model-bridge/single-map
  manifests and hashes, but no unbound model weights and no second cross-map;
- unit-level global, hybrid, conjugacy, additive-interchange, and additive
  planning-transport rows;
- state-permuted, random-map, random-carrier, DINO self-positive, zero-edit,
  action-reversal, mode-permutation, same-model split-half, decoder-mean,
  action-conditioned-map, and 208-feature random-feature controls;
- 5,000-draw trajectory-bootstrap indices or reproducible seeds, 95%
  intervals, family summaries, and Holm multiplicity records;
- figures, tables, memory/timing logs, cache audit, `FAILURE_TRACE.txt`, and a
  mechanically generated interpretation report; and
- a raw-shard manifest binding any large activation/image arrays retained only
  in Drive.

No synthetic fallback is permitted. Missing, stale, mock, unhashed, or
incompatible assets stop the run. Resume may reuse only completed atomic shards
whose configuration, source, checkpoint, split, action-bank, row-count, and
content hashes match; partial shards are recomputed. Evaluation opening is
irreversible for a nonce.

### What a positive pilot still would not establish

- equivalence beyond PushT, these state families, word lengths, and actions;
- a minimal or unique physical realization;
- independent convergence across target encoders, training runs, or seeds;
- correct uncertainty under arbitrary environment stochasticity;
- a global Koopman representation of contact;
- equality of complete network computations;
- DINO-to-JEPA causal transport, causal necessity from ablation, or rescue of a
  removed interface;
- decoder-free native planning transport;
- closed-loop real-robot reliability; or
- causal necessity of the interface outside the tested local interventions.

An ICLR-level submission would still need a prospectively frozen replication
on at least one additional environment with different contact structure,
independent training seeds/checkpoints and preferably a non-DINOv2 target,
greater horizon/action coverage, an external or independently implemented
replication of the intervention assay, and a clear result that levels 4--6 add
scientific value beyond decoder-only and generic black-box system
identification.

## 13. Plain-language interpretation contract

We are looking for a compact *control panel* inside each model: a small set of
internal quantities that predicts how several future pushes unfold, including
when contact begins and ends. Stage 32 showed that useful physical information
is spread broadly enough that random internal slices predict planning almost as
well as the proposed slice. That rules out declaring one convenient subspace
the mechanism.

The intuition behind a predictive state is simple: two internal histories are
treated as the same only when every future push in the declared test bank
unfolds the same way in the grounded trace. A compact predictive chart tries to
retain just the differences that matter for those futures. Calling it
*globally minimal* would require all actions and horizons plus much stronger
realization assumptions, so Stage 33 deliberately makes only the bounded
version of that statement.

The stronger Stage 33 test asks whether one fixed dictionary translates the
two models' control panels across new states and new multi-step pushes. A
different dictionary for every push would be easy to fit and scientifically
weak. One dictionary that also uses a JEPA response to additively move DINO's
own internal prediction in the intended grounded direction, and preserves
decoded choices that work in the simulator, is much harder to fake. This pilot
does not test the reverse direction or remove-and-rescue the proposed control
panel.

Contact matters because motion before impact, impact itself, sustained pushing,
and release follow different local rules. A single averaged rule can look good
while being wrong exactly where planning is difficult. The experiment therefore
compares one global rule with explicit and internally inferred switch rules.

Convincing evidence requires the whole chain: stable compactness, unseen-word
operator transport, a specific JEPA-to-DINO additive internal edit, and
preserved decoded physical planning, while state permutations, random maps,
random carriers, action reversal, mode permutation, and the nonlinear capacity
control fail to reproduce it. Failure of the one fixed map on contact
transitions, or success of matched controls on the additive
interchange-to-planning chain, is the result that should make us abandon the
shared bounded mechanism hypothesis in this form. Even success remains
decoder-mediated and confounded by the common DINOv2 target.
