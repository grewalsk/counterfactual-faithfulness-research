# Stage 30 grounded causal planning value: Colab run guide

Notebook: `notebooks/30_grounded_causal_planning_value.ipynb`

Stage 30 is a fresh, prospective decision-value experiment. It requires the
complete source-bound Stage 29 Drive directory and the frozen Stage 18
subspace directory. It does **not** reuse Stage 29 states or require the failed
physical decoder.

No Stage 30 secrets are required. The notebook automatically:

- uses `RUN_MODE = "pilot"`;
- generates a fresh nonce;
- resolves its committed GitHub branch to an exact commit;
- locates and validates the frozen Stage 18 carrier;
- binds the exact successful Stage 29 decision by SHA-256;
- reads `HF_TOKEN` only if the public checkpoint is not already cached.

## Run

1. Open the committed notebook in Colab.
2. Select a GPU runtime. L4 is the notebook default; another CUDA GPU is fine.
3. Choose **Runtime → Run all**. Do not edit the configuration cell or execute
   isolated later cells because the source binder verifies the committed
   notebook prefix.
4. Authorize Google Drive mounting when prompted.
5. Keep the complete output directory:
   `MyDrive/counterfactual_faithfulness_stage30_grounded_planning/pilot_<signature>/`.
6. Upload the downloaded
   `stage30_grounded_planning_value_result_bundle_<signature>.zip` for
   interpretation.

## What the run does

- Screens 400 new simulator states using contact counts only.
- Selects 40 persistent-contact, 40 boundary-switching, and 40 free-motion
  states.
- Generates 24 exact futures per selected state.
- Evaluates a six-schedule terminal native latent planner at four magnitudes
  and two held-out goal schedules.
- Measures closure on four disjoint interior schedules.
- Runs rank-128 swap and ablation interventions with shuffled,
  empirical-span-random, and full-swap controls.
- Uses five-fold cross-fitting grouped by initial state.

The physical screening pass is CPU/simulator-heavy. The benchmark cell reports
the measured GPU predictor estimate, but total wall time also includes the
fresh simulator screen. Keep the tab connected until the result ZIP downloads.

## Primary outcome labels

- `GROUNDED_CLOSURE_PREDICTS_CAUSAL_PLANNING_VALUE`: the fresh causal-grounding
  gap replicates, grounded closure improves held-out prediction of planning
  regret and ablation value, and primary ablation harms physical planning more
  than matched controls.
- `GROUNDED_CLOSURE_PREDICTS_PLANNING_RELIABILITY_ONLY`: grounded closure adds
  held-out information about native planning regret, but causal ablation value
  does not clear every gate.
- `GROUNDED_CLOSURE_PREDICTS_ABLATION_VALUE_ONLY`: the causal-value test passes,
  but the native-regret prediction gate does not.
- `GROUNDING_GAP_REPLICATED_WITHOUT_PLANNING_VALUE`: the mechanistic gap
  replicates but does not predict the preregistered planning outcomes.
- `CAUSAL_GROUNDING_GAP_NOT_REPLICATED`: the fresh-state self-versus-grounded
  coefficient gap does not clear its frozen gate.
