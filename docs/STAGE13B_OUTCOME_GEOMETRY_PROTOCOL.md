# Stage 13b: outcome-geometry diagnostic

**Status: prospective representation-only protocol. Stage 13 remains a failed
screen and is not reclassified.**

## Objective

Stage 13b determines whether the failed eight-axis Stage 13 dictionary was a
projection/sample artifact, evidence for horizon-specific or state-local
geometry, or evidence that the outcome space is genuinely higher-rank. It
does not construct Jacobians, edit activations, load ARGA checkpoints, or test
a workspace causally.

One of five outcomes is allowed:

1. promote a compact global outcome vocabulary to a later J-lens study;
2. promote separate horizon-specific vocabularies;
3. promote a query-conditioned local outcome tangent bundle;
4. retain only a distributed/higher-rank outcome-space hypothesis; or
5. abandon this JOW formulation at the tested scale.

## Frozen execution order

### E0: observed-cache audit

The 12 Stage 13 target-token shards are observational. If present on Drive,
the notebook separates horizons, collapses identical action prefixes, replays
all preregistered sketch seeds, and emits per-state diagnostics. E0 cannot
select a confirmatory method or satisfy a gate. Its absence does not block the
study.

### E1: method selection

E1 uses 48 previously unused states: four from each of the 12 existing PushT
tasks. States used by Stage 13 are excluded, and the remaining state IDs are
ordered by a namespace-, ID-, and state-content-bound SHA-256 value. The first
four per task are frozen before any Stage 13b encoding.

Leave-one-task-out reconstruction chooses the smallest rank within one
standard error of the best task-equal mean over

\[
K \in \{1,2,4,6,8,12,16,24,32\}.
\]

E1 also selects the RBF bandwidth/rank pair with the same one-standard-error
principle. Among eligible pairs it prefers the smallest rank and then the
smoother bandwidth.

### Freeze boundary

After E1, the notebook writes and hashes:

- the configuration and exact source identity;
- action, task, state, split, and random-schedule hashes;
- selected ranks and the absolute local bandwidth;
- frozen dual coefficients and diagonal scales;
- sketch selections and every threshold; and
- a certificate asserting that zero confirmation target shards existed.

On a resumed run, recomputed E1 selections must exactly equal the saved
freeze. Confirmation encoding fails closed on any discrepancy.

### C: untouched confirmation

C contains eight new PushT task definitions and four new states per task.
Tasks, states, task halves, actions, and random schedules are generated and
hashed before E1 encoding. C is not encoded until the freeze certificate
exists. Task labels are used only for grouped evaluation, never as local-model
inputs.

## Goal-independent action design

Every E1 and C state receives the same 13 frozen sequences for 15 primitive
steps: one no-op plus six exactly antithetic pairs—constant x, constant y, two
diagonals, a two-phase turn, and a three-phase path.

All non-noop prefixes are unique as arrays after primitive steps 5 and 15.
The array dtype, shape, contents, labels, and SHA-256 are saved. No task goal,
cost, planner outcome, or future simulator result affects the action menu.
The frozen action-array hash is
`802129bd281fdd2d42a395429e5a0e00df2dc10032b339ecb8bdc8b2521d9fd2`.

## Representations

The primary representation is the full centered `256 x 384` frozen target
token effect. Each state and horizon is centered across its 13 actions. PCA is
fit through row Gram matrices; the notebook never constructs a
`98,304 x 98,304` covariance.

The confirmatory primary method is raw native geometry with separate h1 and h3
bases. Frozen diagnostics are:

- E1-fitted diagonal standardization;
- the original normalized 128-D CountSketch at seed 13119;
- 32 additional preregistered CountSketch seeds, summarized without choosing
  the best;
- pooled-horizon PCA at the sum of the two selected atom budgets;
- an RBF model whose only query is the frozen mean initial visual encoding;
  and
- a within-state action-split oracle, reported only as an upper bound.

## Metrics and nulls

For state \(s\), horizon \(h\), and orthonormal basis \(U\),

\[
R_{s,h}(U)=
\frac{\lVert X_{s,h} U U^\top\rVert_F^2}
     {\lVert X_{s,h}\rVert_F^2}.
\]

States are averaged within tasks and tasks equally. The notebook saves the
complete rank curve, spectral effective rank, state/task learning curves,
cross-horizon overlap, fixed task-half overlap and principal angles, and
per-state/task gains.

Each primary fold receives 1,024 covariance-shaped random orthogonal
subspaces. The same preregistered seed schedule is used at confirmation. The
analytic Haar expectation \(K/98{,}304\) is saved separately. The local model
also receives 1,024 frozen task-block permutations of the query-to-basis
mapping.

Uncertainty is task-level:

- a one-sided 95% `t_7` lower bound over eight confirmation task means;
- a one-sided exact task-sign test; and
- 10,000 descriptive hierarchical bootstrap draws resampling tasks and then
  the four states within sampled tasks.

Actions are never treated as independent inferential units.

## Prospective representation gate

For each horizon, a global native representation passes only if:

- its E1-selected rank is at most eight;
- its C equal-task mean gain over the replicated covariance null is at least
  0.05;
- its PCA/null ratio is at least 1.25;
- the one-sided task-level lower bound on gain exceeds zero;
- at least seven of eight tasks have positive gain;
- its selected rank retains at least 80% of the reconstruction of the frozen
  rank-32 basis; and
- its fixed C task-half overlap is at least 0.5 and exceeds the 97.5th null
  percentile.

A local replacement must satisfy the applicable rank, null, uncertainty, task
consistency, and rank-32 criteria. It must additionally beat the global model
by at least 0.05, have a positive task-level lower bound for that improvement,
and improve on at least seven of eight tasks. Task-block permutations remain
a reported mapping-specificity diagnostic.

No failed criterion can be repaired by opening more ranks, seeds,
preprocessing variants, or methods on C. Stage 13b contains no VJP code, so
every outcome stops before Jacobians by construction.

## Decision mapping

| Untouched result | Decision |
|---|---|
| Both global bases pass; pooled basis passes; cross-horizon overlap at least 0.5 | Promote global outcome vocabulary |
| Separate h1/h3 bases pass but pooling/alignment fails | Promote separate horizon lenses only |
| Global fails and the conditional local model passes both horizons | Promote a state-conditioned tangent bundle |
| Stable selected rank exceeds eight while all other global checks pass | Retain a distributed/higher-rank hypothesis; no compact workspace claim |
| Neither global nor predictable local geometry passes | Abandon this JOW formulation at this scale |

Sketch variability, native-versus-standardized disagreement, the local oracle,
and E1-versus-C degradation qualify the interpretation but cannot override the
decision table.

## Reproducibility and returned artifacts

The default `full` run saves:

- repository commit, notebook and builder hashes, declared patch hash,
  upstream code/checkpoint hashes, and dependency versions;
- frozen actions, tasks, states, splits, schedules, and certificates;
- complete E1 curves, fold rows, null arrays, learning curves, sketch results,
  dual coefficients, and local parameters;
- C state/task tables, null draws, bootstrap indices/draws, overlaps,
  principal angles, permutation draws, gates, plots, timings, and memory;
- a compact `stage13b_result_bundle.zip`; and
- `stage13b_full_evidence_bundle.zip`, which additionally contains every
  target-token shard required to reconstruct the centered effects.

`smoke` mode uses reduced tasks, ranks, seeds, and draws. It can return only
`SMOKE_COMPLETE_NO_SCIENTIFIC_DECISION`.

## Compute envelope

Use one T4 or L4 with at least 16 GiB system RAM and approximately 8 GiB free
Drive storage. Target tokens should occupy roughly 390 MiB; truth frames and
temporary effect matrices require additional durable space. Expected full
runtime is approximately 45–90 minutes, depending mostly on simulator and
streamed Gram performance. No runtime restart is expected.
