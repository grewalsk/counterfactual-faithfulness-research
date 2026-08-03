# Stage 15 fixed-reader failure audit

## Bottom line

The preregistered Stage 15 reader failed because its flat CountSketch destroyed
useful spatial structure, not because the target tokens lacked transferable
physical information. Raising the random-sketch dimension from 192 to 1,152 or
2,304 does not rescue the gate, while dimension-matched coordinate moments do.
Two post-hoc, fixed, linear, and coordinate-aware readers clear the original
aggregate evaluation gate without using evaluation data for fitting or
hyperparameter selection.

This rescues the *measurement strategy*, not the Stage 15 scientific claim.
The evaluation trajectories have now been observed, the result is fragile to
trajectory deletion, and no predictor operators or causal interventions were
computed. A fresh source-bound reader confirmation is required before the
causal transport experiment.

## Evidence integrity and scope

- Source run: `f07a078c34b534bd8a0c5f760af8eba7ccf1bb5c`
- Saved run: `pilot_5703d35fb8b5`
- Raw files verified: 113/113
- Raw bytes verified: 382,416,576
- Missing files or SHA-256 mismatches: 0
- Construction examples: 520 from trajectories 0, 2, 4, 6
- Evaluation examples: 520 from trajectories 1, 3, 5, 7
- Frozen-reader reproduction error: maximum absolute R² error
  `1.47e-7`

All alternative feature maps and ridge strengths were fitted and selected by
leave-one-construction-trajectory-out CV. Evaluation results below are still
post-hoc because the feature families themselves were introduced after the
original evaluation gate was observed.

## What failed

The frozen reader compressed each flattened `256 × 384 = 98,304`-dimensional
target-token tensor to 192 random CountSketch coordinates. Its evaluation
median R² was 0.144 and its minimum spatial R² was -0.543, below the frozen
thresholds of 0.25 and 0.10.

Extending the ridge grid did not rescue it: the three-sketch ensemble reached
only 0.117 median R² with the same -0.545 minimum spatial R². The failure is
therefore not explained by the original ridge grid ending at 10.

Nor is feature count alone sufficient. Three-sketch ensembles with 1,152 and
2,304 random features reach only 0.153 and 0.189 median evaluation R², and both
retain a negative minimum spatial R². Coordinate structure, rather than merely
more random features, is the relevant change.

## Coordinate-aware alternative

For patch `p`, token vector `z_p`, and normalized patch coordinates `(x_p,y_p)`,
the degree-1 reader constructs

\[
m_0=\frac{1}{P}\sum_p z_p,\qquad
m_x=\frac{1}{P}\sum_p x_pz_p,\qquad
m_y=\frac{1}{P}\sum_p y_pz_p.
\]

A construction-only ridge maps `[m_0,m_x,m_y]` to the six physical outputs.
The degree-2 reader additionally uses `x²`, `xy`, and `y²` moments. These
readers are fixed global linear functions of the token tensor and remain fully
differentiable. They do not use the current state, task, goal, action, or an
evaluation-specific frame.

| Reader | Features | Construction CV MSE | Eval median R² | Eval min spatial R² | Old aggregate gate |
|---|---:|---:|---:|---:|---|
| Frozen flat CountSketch | 192 × 3 | — | 0.144 | -0.543 | Fail |
| Flat CountSketch, extended ridge | 192 × 3 | 1.091 | 0.117 | -0.545 | Fail |
| Flat CountSketch, extended ridge | 1,152 × 3 | 1.001 | 0.153 | -0.141 | Fail |
| Flat CountSketch, extended ridge | 2,304 × 3 | 1.021 | 0.189 | -0.178 | Fail |
| Channel mean only | 384 | 1.117 | 0.071 | -0.321 | Fail |
| Coordinate moments, degree 1 | 1,152 | **0.721** | 0.329 | 0.207 | **Pass** |
| Coordinate moments, degree 2 | 2,304 | 0.810 | **0.407** | 0.190 | **Pass** |
| 4×4 spatial pooling | 6,144 | 0.883 | 0.230 | 0.179 | Fail |

Degree 1 is the construction-CV winner among the audited token readers. Its
evaluation coordinate R² values are:

| Coordinate | R² |
|---|---:|
| Agent x | 0.839 |
| Agent y | 0.451 |
| Block x | 0.207 |
| Block y | 0.580 |
| Block sin | 0.091 |
| Block cos | 0.206 |

The improvement is horizon-stable:

| Reader | Horizon | Median R² | Minimum spatial R² | Old aggregate gate |
|---|---:|---:|---:|---|
| Degree 1 | 1 | 0.340 | 0.191 | Pass |
| Degree 1 | 3 | 0.340 | 0.223 | Pass |
| Degree 2 | 1 | 0.413 | 0.179 | Pass |
| Degree 2 | 3 | 0.401 | 0.201 | Pass |

## Renderer-to-token geometry

The audit recovered the exact 16×16 patch geometry from the saved 224×224
frames. The agent renderer centroid differs from the simulator coordinate by a
median 0.00090 in normalized image units (0.46 simulator pixels). Quantizing to
14×14 DINO patches raises this to only 0.00548 (2.81 simulator pixels).

The T-block color centroid is offset from its physics-body origin by about
0.080 normalized units (41 simulator pixels). This is a stable property of the
asymmetric T renderer, not a token-coordinate mismatch. An oracle reader using
only the patch-mask centroid and low-order shape moments attains 0.455 median
R² and 0.454 minimum spatial R² on evaluation, demonstrating that the token
grid has adequate spatial resolution. It is a ceiling diagnostic, not an
available JEPA readout.

## Interpretation

The data support the following narrow statement:

> JEPA target tokens contain transferable physical information whose linear
> accessibility depends strongly on respecting token coordinates.

They reject the stronger measurement assumption that 192 random coordinates
of a flattened token tensor provide a stable semantic chart. This distinction
fits the predictive-control-bundle hypothesis: the carrier is a spatial field,
so a useful chart must retain low-order spatial structure.

The result is not yet robust enough for a paper claim. Only four evaluation
trajectories were used, and each coordinate-moment reader passes the aggregate
gate for only one of four leave-one-evaluation-trajectory-out subsets. The
construction leave-one-trajectory-out gate also fails on block x, even though
its mean CV loss selects the degree-1 model. The interleaved evaluation split
is an interpolation test, not broad trajectory extrapolation.

Native predicted proprioception could provide a stronger agent-position
readout, but predictor proprio outputs were not saved in this stopped run and
cannot be tested offline. No operator or causal shard exists because the
reader gate stopped the notebook as designed.

## Frozen next experiment

1. Preserve this Stage 15 gate failure and the flat-reader ablation.
2. Freeze degree-1 coordinate moments as the primary reader and degree 2 as a
   sensitivity analysis. Do not select further readers using trajectories
   1, 3, 5, or 7.
3. Run a reader-only confirmation on 16 new interleaved trajectories: eight
   construction and eight evaluation trajectories, five timepoints, both
   horizons. Save native predicted proprioception as a secondary agent readout.
4. Require median six-coordinate R² ≥ 0.30, every spatial R² ≥ 0.20, and report
   both horizons separately. Require stability across leave-one-evaluation-
   trajectory-out subsets rather than only the pooled aggregate.
5. Only after that gate passes, run the primary causal test at horizon 3,
   predictor block 4. Expand to all layers and both horizons only if the primary
   causal test is positive.

This sequence tests the transported-field idea without repeating the failed
global random-frame experiment and without spending GPU time before the
semantic measurement is independently validated.

## Files

- `reader_failure_audit.json`: complete machine-readable results
- `model_metrics.csv`: model/output R² table
- `construction_cv.csv`: construction-only ridge selection losses
- `renderer_geometry.csv`: per-example renderer/token-coordinate diagnostics
- `reader_model_comparison.png`: compact comparison plot
- `scripts/audit_stage15_reader_failure.py`: deterministic audit program
