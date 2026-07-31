# Stage 13 JOW: compute-minimal Colab design

**Status: executable-design memo, not a frozen protocol or authorization for a
confirmatory run.**

## Decision

The first Colab should be a **training-free, sequentially gated diagnostic**.
It should reuse the Stage 12 PushT design and checkpoints, begin with a tiny
Jacobian-and-intervention smoke test, and stop automatically unless each
necessary condition for a Jacobian Outcome Workspace (JOW) is present.

This is deliberately smaller than the Phase 0 experiment in
[`STAGE13_JEPA_JACOBIAN_OUTCOME_WORKSPACE_IDEA.md`](STAGE13_JEPA_JACOBIAN_OUTCOME_WORKSPACE_IDEA.md).
Its job is to decide whether the full Phase 0 is worth running.

## What the screening run includes

| Choice | Screening value | Expansion value, only after a pass |
|---|---:|---:|
| Environment | PushT | PushT |
| Conditions | frozen | frozen, matched, then shuffled |
| States | 8 construction + 4 calibration | 16 construction + 8 calibration |
| Actions | all 10 for the dictionary | all 10 |
| Horizons | 1 | 1 and 3 |
| Predictor blocks | 2, 4, and 6 | all six |
| Signed prototype pairs | 8 | 16 |
| Dictionaries | PCA and covariance-matched random | add spherical k-means |
| Swap doses | 0.5 and 1.0 | 0.25, 0.5, 0.75, and 1.0 |
| Action pairs per state | two most target-separated pairs | four |
| Training | none | none |

The action pairs are selected using true, goal-free target-effect distance
within the same state. No task cost, planner outcome, or inspected development
contrast may enter pair selection.

## Required Colab cells

### 1. Fixed configuration

Expose one small configuration object at the top:

```python
RUN_MODE = "screen"             # "screen" or "expand"
ENVIRONMENT = "pusht"
CONSTRUCTION_STATES = 8
CALIBRATION_STATES = 4
HORIZONS = (1,)
BLOCKS = (1, 3, 5)              # zero-indexed blocks 2, 4, and 6
SIGNED_PROTOTYPE_PAIRS = 8
SWAP_DOSES = (0.5, 1.0)
ACTION_PAIRS_PER_STATE = 2
ADAPTATION_SEED = 11401
GLOBAL_SEED = 13101
STOP_ON_FAILED_GATE = True
```

`RUN_MODE="expand"` changes only the preregistered values in the table above.
It must not select settings from the same evaluation rows used to report the
effect.

### 2. Minimal installation and asset retrieval

- Do not reinstall PyTorch on Colab.
- Pin only the simulator and model dependencies already used by Stage 12.
- Use a depth-one, blob-filtered checkout.
- Fetch only the PushT design, the chosen PushT adapted checkpoints, the frozen
  physical decoder, and the public pretrained model assets.
- Verify every asset against the saved Stage 12 manifests before execution.

The repository already contains the small `pusht_design.npz`, the 22 MB matched
and shuffled seed-11401 action-path checkpoints, and the saved frozen physical
decoder. It does not contain the large intermediate transition cache, so this
pilot should regenerate only its selected simulator outcomes and activations.

### 3. Drive-backed resume directory

Mount Drive if available and create a run directory keyed by:

- git commit;
- configuration hash;
- pretrained checkpoint hash;
- adapted-checkpoint hash;
- design hash.

Each gate writes an atomic result file and a completion marker. A restarted
runtime skips only a gate whose hashes and completion marker match. Large
autograd graphs are never serialized.

### 4. Integrity smoke test

Use two states, two actions, horizon 1, and block 4 to require:

1. exact checkpoint restoration;
2. clean resumed-forward output equal to the uninterrupted forward output;
3. a zero-dose intervention equal to the clean output;
4. finite vector-Jacobian products with the expected token-grid shape;
5. a nonzero random-direction edit with a recorded activation-distribution
   distance.

Any failure stops the run before the scientific screen.

### 5. Small target-effect cache

For the selected states and all ten actions:

1. restore the saved PushT state;
2. execute only the required horizon;
3. encode the true future frame;
4. compute the action-relative, within-state-centered target effect;
5. save target latents in `float16` on CPU/Drive.

The cache should contain tensors and hashes, not simulator objects or GPU
graphs.

### 6. Frozen outcome dictionaries

Fit on the construction states only:

- primary screen: whitened PCA effect directions;
- null: covariance-matched random orthogonal directions.

Sign each direction to obtain paired atoms. Freeze whitening, means, scales,
and atoms before opening calibration results. Spherical k-means is deferred
until expansion so the screen does not spend compute on dictionary sweeps.

### 7. Streaming vector-Jacobian products

For one state-action item at a time:

1. run a single predictor forward while capturing the selected block outputs;
2. form the eight signed prototype scores at the final predicted effect;
3. use vector-Jacobian products, retaining the forward graph only for those
   scores;
4. update a running `float32` mean lens;
5. immediately release the graph and move summaries to CPU.

Never materialize the full Jacobian. Even the expanded
`16 × 6 × 256 × 384` mean lens is only about 38 MB in `float32`; the live
autograd graph, not the saved lens, is the memory risk.

### 8. Calibration-only signal gate

On the four calibration states, require all of the following before performing
adapted-model comparisons:

- prototype scores reconstruct nontrivial held-out true-effect structure;
- the PCA lens beats the covariance-matched random lens;
- results are not carried by one state or one action;
- gradients and inferred sparse coordinates are numerically stable.

The notebook should report confidence intervals and diagnostics, but this gate
is a feasibility filter rather than a confirmatory statistical claim.

### 9. Same-state causal swap engine

At each selected block:

1. compute the recipient activation's sparse JOW component;
2. replace it toward the donor action's component at the requested dose;
3. preserve the recipient residual;
4. continue the predictor from that block;
5. measure normalized donor transfer in final predicted target tokens and with
   the already-frozen physical decoder.

Run equal-norm orthogonal and random swaps with the same recipient, donor,
layer, and dose. Record norm, cosine, Mahalanobis distance from natural
activations, and resumed-forward error for every intervention.

### 10. Causal gate and treatment sequence

Stop unless the JOW swap:

- moves the prediction in the donor direction;
- exceeds both matched-dose controls;
- has a dose response;
- remains within the calibrated activation-distribution envelope;
- repeats across states.

If it passes, run the same frozen dictionary and lens procedure on the
seed-11401 matched checkpoint. Load the shuffled checkpoint only if matched
appears better than frozen. This ordering avoids paying for negative controls
when the proposed treatment has no signal, while still requiring shuffled
geometry before any treatment-specific conclusion.

### 11. Optional Phase 0 expansion

Only a screening pass unlocks:

- the remaining states;
- horizon 3;
- all six layers;
- 16 signed prototype pairs;
- spherical k-means robustness;
- four intervention doses;
- frozen, matched, and shuffled conditions.

The expansion still fits no planner and no goal metric. A full broadcast test,
the second environment, latent-only treatment, multiple seeds, and new
confirmatory tasks belong to later phases.

### 12. Result package

Always save a compact bundle containing:

- exact config and hashes;
- timing and peak-memory table;
- integrity results;
- frozen dictionaries and mean lenses;
- state-level intervention rows;
- gate decisions with reasons;
- plots;
- failure trace, if any.

Do not package target-token caches or captured activation batches by default.
They can be regenerated from the hashed design and checkpoint.

## Compute and time control

The notebook should benchmark the two-state integrity cell and calculate its
own projected time before the user approves the screen. A fixed ETA would be
less reliable across Colab G4 runtimes.

A reasonable planning envelope is:

- setup and verified asset retrieval: roughly 10–25 minutes;
- integrity smoke: roughly 5–10 minutes;
- frozen screen: roughly 20–45 minutes;
- matched and conditional shuffled checks: roughly 20–50 minutes;
- full Phase 0 expansion after a pass: roughly 1–2.5 additional hours.

Thus a negative hypothesis should usually stop in under an hour after setup.
A successful screen plus expansion is more likely to occupy roughly 2–4 hours
on a G4, with the notebook replacing these estimates using measured
per-item timing.

## What this run can and cannot establish

A successful screen would show that a full JOW causal experiment is worth its
compute. It would not establish a workspace, treatment generalization, or
planning improvement.

The cheapest result that could support the workspace label still requires the
expanded layer-localization and matched controls, followed later by broadcast
tests and untouched newly generated tasks. If the screening causal gate fails,
the notebook should save the falsification evidence and stop rather than
searching additional layers, seeds, prototype counts, or goals.
