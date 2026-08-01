# Stage 14 predictive-control J-bundle pilot

This is a switch away from the failed global Jacobian Outcome Workspace (JOW),
not a rescue of it. The notebook tests a narrower JEPA-native object:

\[
G_s=D_Hq(F_h),\qquad B_s=D_uH,\qquad K_s=G_sB_s.
\]

`G` measures which internal carrier directions a future-embedding query can
read. `B` measures which directions executable action trajectories can write.
The candidate object is a state-indexed predictive-control bundle/atlas of
directions that are both writable and readable.

## What a pass would and would not mean

A pass supports only `JEPA_NATIVE_PREDICTIVE_CONTROL_BUNDLE_CANDIDATE` for this
frozen PushT model, carrier, two-dimensional tangent design, and state-specific
oracle target readers. It does **not** establish a universal JEPA “J-space,” a
global workspace, a deployable reader, or a native policy interface.

The notebook separately tests:

- train/train transfer on new tasks;
- held-out actions with construction query families;
- held-out query families with construction actions;
- the joint held-out action/query cell;
- a context-transported local atlas alternative;
- hidden-path sufficiency and full-action necessity using the actual natural
  activation delta;
- dose response, complement, full-linear, natural-activation, dense, and
  native-norm-matched Haar controls;
- horizon-3 frame reuse at horizon 1; and
- task-equal free/contact diagnostics, with both strata required to contain at
  least four construction-threshold-stable tasks before causal promotion.

## First run: smoke mode

1. Upload `14_predictive_control_j_bundle_pilot.ipynb` to Google Colab.
2. Select a GPU runtime.
3. Leave the Colab secrets absent and run all cells.
4. Smoke mode validates the model/API, PushT goal binding, hooks, exact JVP/VJP
   plumbing, causal interventions, and packaging. It can never make a
   scientific pass claim.
5. If it fails, preserve the generated result archive; `FAILURE_TRACE.txt` and
   progress files identify the failing stage.

The exact-Jacobian benchmark is written before the full extraction. The causal
section uses 32 coherent null draws and is expected to dominate runtime; use
the notebook's measured benchmark rather than a guessed wall-clock estimate.

## Untouched source-bound pilot

Do not edit the notebook in Colab.

1. Commit and push the exact notebook, builder, validator, numerical module,
   and test file to `grewalsk/counterfactual-faithfulness-research`.
2. Copy the resulting full 40-character commit SHA.
3. In Colab, add secrets:
   - `STAGE14_RUN_MODE` = `pilot`
   - `STAGE14_SOURCE_COMMIT` = the full commit SHA
4. Open the exact notebook from that commit, restart the runtime, and use
   **Run all** once, in order.

The notebook fetches the committed artifact, compares every executed code cell
through evaluation opening, decision, and packaging, and makes all positive
labels exploratory if that binding is absent. Construction-derived thresholds,
the frame, both horizon-specific nondegeneracy floors, null schedules, and all
gates are frozen before evaluation outcomes are generated.

## Decision labels

- `NO_EVIDENCE_FOR_THIS_FROZEN_GLOBAL_FRAME`: this frame failed; it is not a
  verdict on every possible JEPA interface.
- `STATE_CONDITIONED_PREDICTIVE_ATLAS_CANDIDATE`: the global frame failed but
  construction-only context transport worked.
- `NO_CAUSAL_CONFIRMATION_FOR_THIS_FRAME`: representational transfer worked,
  but the hidden path did not pass sufficiency/necessity controls.
- `H3_PREDICTIVE_CONTROL_BUNDLE_WITHOUT_H1_REUSE`: the horizon-3 result did not
  transfer to horizon 1.
- `JEPA_NATIVE_PREDICTIVE_CONTROL_BUNDLE_CANDIDATE`: all frozen gates passed.

Every positive label is prefixed with `EXPLORATORY_` unless the executed source
is bound to the immutable commit.

## Audit outputs

The downloaded archive includes the preregistration certificates, selected
carrier and frame, state/task rows, coherent null rows, causal rows, plots,
source identity, an exact self-excluding compact-archive manifest, a full
raw-file hash manifest, and reduced evaluation evidence containing `G`,
train/test writes, all four local transfer cells, separations, and normalized
Hankel energy. Large truth, target, scan, and full Jacobian files remain in the
durable Drive run directory and are committed by the full manifest.

## Local validation

- Builder SHA-256: `18ca558f631d01c2f82e206e65b1c19b414d329529f0171049dbb1fdcec627d4`
- Notebook SHA-256: `a2f0d7d504550a35eff98303e8412bd3415c55fcb443dc04c4fbbc946f4e30e0`
- Static/reproducibility validator: passed
- NumPy numerical tests: 6 passed
- End-to-end frozen-model execution: requires the Colab GPU smoke run above
