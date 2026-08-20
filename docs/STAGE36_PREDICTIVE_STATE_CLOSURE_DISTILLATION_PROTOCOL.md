# Stage 36 protocol: predictive-state closure distillation

## V5 implementation amendment

The source-bound v4 run completed physical truth, selected simulator response
rank five, verified the official checkpoint, and loaded JEPA-WM. It stopped
before the first model forward because an inherited output preflight requested
legacy action word `L`, outside Stage 36's registered binary alphabet. V5 binds
the preflight to the first registered construction response word, removes
legacy executable `L/R/S/a/b` tokens, and validates the preflight and transition
word generators against the real manifest. No model output or locked evaluation
was observed, and no candidate, threshold, control, split, gate, or claim
changes.

## V4 implementation amendment

The v3 notebook stopped during setup when GitHub returned HTTP 504, before
source identity was written and before any simulator or model computation. V4
adds bounded exponential-backoff retries for retryable GitHub API and committed
raw-file failures. Exact commit resolution, file hashing, and executed-cell
verification remain mandatory; the notebook never falls back to unbound or
unverified source. No scientific configuration or decision changes.

## V3 implementation amendment

The source-bound v2 run completed 320 physical-truth records and stopped before
JEPA-WM loading while rank-selecting the simulator-only canonical response
chart. Model-selection truth shards contained only the registered unseen
length-5--8 task words, while that consumer also required canonical
length-1--4 response words and their zero controls. V3 adds the canonical bank
to model-selection truth coverage, validates the exact shard word contract on
cache hits, and tests the real per-split schemas. The model-selection task bank
remains disjoint from construction, and no seed, trajectory split, candidate,
control, threshold, gate, or claim changes.

## V2 implementation amendment

The source-bound v1 run completed physical-truth generation and stopped in the
simulator-only response-chart step because inherited Stage 33/34 numerical
helpers were not embedded in the rendered notebook. No world model was loaded
and no carrier, adapter, model-selection statistic, evaluation metric, or gate
was observed. V2 restores the complete helper dependency chain and adds an
executable truth-path response-signature validation. The scientific protocol
below is unchanged.

## Motivation

Stage 35 passed source binding, simulator control, native JEPA physical
fidelity, and apparent guard transfer, but failed guard specificity and
recursive closure. The predicted recursive physical error was 2.66 times the
native error. Permuted contact labels outperformed true contact labels, shifted
labels were indistinguishable, and support escape was zero. The evidence favors
an incomplete state or transition representation rather than an out-of-support
rollout or an incorrectly tuned semantic contact guard.

Stage 36 therefore changes the state definition while freezing JEPA-WM.

## Estimand

For native projected carrier `c`, finite native history `h`, action `a`, and a
learned state `z`, PSCD fits

```text
z_t = E(c_{t-h+1:t})
z_{t+1} = T(z_t, a_t)
(c_hat_{t+1}, y_hat_{t+1}) = D(z_{t+1})
```

The primary estimand is the ratio between recursively decoded physical error
and frozen native JEPA physical error on trajectory-disjoint, unseen action
words of lengths 9--12. Secondary estimands are direct-versus-composed latent
discrepancy, carrier recovery, support escape, and gains over capacity-matched
controls.

## Frozen model-selection space

- Carrier projection: 256 or 1,024 nested deterministic coordinates.
- Native history: 1, 2, or 4 prefix carriers.
- Latent state: 64 or 128 coordinates.
- Transition: one residual operator or a label-free three-expert mixture.
- Candidate training: 80 epochs on construction trajectories.
- Final training: 240 epochs on construction plus calibration trajectories.

Construction includes words of lengths 1--8. Candidate selection uses disjoint
trajectories and distinct unseen words of lengths 5--8. Evaluation remains unopened
until the selected state definition, final adapter, controls, scales, support
reference, and certificate are frozen.

## Objective

The adapter loss combines:

1. next native-carrier reconstruction;
2. next native grounded-output reconstruction;
3. one-step transition agreement with the next directly encoded history;
4. free-running multi-step carrier and grounded-output prediction; and
5. free-running agreement with directly encoded future history states.

The official JEPA encoder, predictor, action encoder, and checkpoint parameters
remain frozen. Simulator physical truth is not a training target for the
primary adapter; it is used for the positive control and locked grounding
evaluation.

## Controls

- Direct physical-state recursion verifies that the operator class can close a
  known Markov state.
- The Stage-35-style nonlinear Markov carrier recursion tests whether the new
  state definition adds value beyond capacity.
- A capacity-matched one-step-only adapter removes free-running training.
- When history exceeds one carrier, a capacity-matched false-history adapter
  permutes past slots within trajectory groups while preserving the current
  carrier.
- Native JEPA direct prediction and physical/carrier persistence remain fixed
  references.

No physical-mode or contact label gates the primary adapter. Simulator modes
are used only for the positive control and prespecified family diagnostics.

## Gates

The gates are evaluated sequentially:

1. exact source, checkpoint, split, and evaluation-opening integrity;
2. simulator-control gain at least 50%, positive clustered interval, and NMSE
   at most 0.25;
3. native JEPA physical-fidelity gain at least 10% with positive interval;
4. recursive carrier-recovery gain at least 10% over persistence;
5. at least 5% improvement over Markov and one-step-only controls, plus the
   false-history control when applicable, each with positive interval;
6. recursive/native physical ratio at most 1.25, interval upper endpoint at
   most 1.50, composition discrepancy at most 0.25, and support escape at most
   10%;
7. mean direct-versus-composed latent NMSE at most 0.25; and
8. recursive/native ratio at most 1.50 in each length family and at most 2.00
   in each starting-mode family.

## Claim boundary

A pass supports only this statement: a low-capacity finite-history adapter can
distill bounded recursive closure from the frozen JEPA checkpoint over the
registered PushT action bank. It does not imply that the original carrier was a
closed Markov state, that the learned state is minimal, that the result
generalizes beyond this checkpoint/environment, or that JEPA causally uses the
adapter state.
