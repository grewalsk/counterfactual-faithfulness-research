# Stage 15 run guide

Notebook: `notebooks/15_longitudinal_predictive_control_bundle.ipynb`

## Safety rule

Do not start the pilot merely to discover whether the notebook launches.
The generated artifact has already been compiled, structurally validated,
numerically unit-tested, and run through 7,960 local PushT simulator steps.
Use Colab only for the JEPA-WM GPU execution that cannot run on the local CPU.

## First GPU run

1. Push the exact generated notebook, builder, numerical module, validator, and
   protocol to GitHub in one commit.
2. In Colab secrets set:
   - `STAGE15_RUN_MODE` to `pilot`
   - `STAGE15_SOURCE_COMMIT` to that full 40-character commit SHA
3. Open the notebook from the commit-pinned GitHub URL in a fresh GPU runtime.
4. Use a G4, L4, A100, or equivalent CUDA runtime. Do not edit notebook cells.
5. Select **Runtime → Run all** exactly once.
6. Read `benchmark.json` when the measured ETA is printed. The notebook is
   resumable in its signature-specific Drive directory.
7. Return `stage15_longitudinal_bundle_result_bundle.zip`. Keep the durable raw
   directory until its manifest has been audited locally.

The pilot performs one scientific run, not a smoke-then-pilot pair. Smoke mode
cannot authorize any claim and is unnecessary after the completed local tests.

## Expected scale

- 8 trajectories × 5 states = 40 longitudinal states.
- 13 exact action branches per state.
- Horizons 1 and 3.
- All 6 predictor blocks captured in shared JVP passes.
- 6 fixed physical readers and a common 6-dimensional action basis.
- 16 adjacent evaluation transitions with matched causal controls.

Allow roughly 60–120 minutes on a G4, subject to the notebook's measured
one-state benchmark. The implementation computes only the six preregistered
action-tangent JVPs instead of an unused 30-dimensional Jacobian and batches
each transition's causal controls. If the measured upper-bound ETA exceeds
150 minutes, an automatic credit guard stops the run before the full operator
sweep. Raw resumable evidence may occupy roughly
5–8 GB in Drive; the
downloaded compact result bundle excludes those recomputable shards.

## Automatic stopping and outcomes

The run stops before expensive operator extraction if the construction-only
fixed reader fails on held-out evaluation target tokens. Final outcomes are:

- `LONGITUDINAL_CAUSAL_BUNDLE_SUPPORTED`
- `LOCAL_CAUSAL_ONLY`
- `SMOOTH_NONCAUSAL_FIELD`
- `NO_LONGITUDINAL_BUNDLE_EVIDENCE`
- `PIPELINE_FAILURE`

The notebook never authorizes a global J-space claim or cross-model generality.
