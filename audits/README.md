# Independent result audits

These reports were produced after inspecting the downloaded Colab result
bundles rather than trusting notebook headline labels.

- `stage1/` — pipeline and exact-restoration audit.
- `stage2/` — decisive-pilot and candidate-degeneracy audit.
- `stage2b/` — confirmatory raw-latent negative result.
- `stage2c/` — task-aligned readout signal.
- `stage3/` — cross-environment planning result, regression NaN defect, Wall
  interaction-taxonomy mismatch, and model-ranking artifact.
- `stage4/` — independently reproduced matched-intervention causal result,
  dose response, and model-by-horizon direction consistency.
- `scripts/audit_stage3.py` — row-level Stage 3 reconciliation script.
- `scripts/audit_stage4.py` — independent Stage 4 integrity and decision
  reconciliation script.

The Stage 3 audit motivated `notebooks/03b_stage3_analysis_repair.ipynb`.
Stage 4 then converted the remaining correlational ambiguity into a controlled
matched-error intervention test.
