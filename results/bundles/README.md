# Original Colab result bundles

These ZIP files are the unmodified bundles downloaded after each manual Colab
run:

- `stage1_result_bundle.zip`
- `stage2_result_bundle.zip`
- `stage2b_result_bundle.zip`
- `stage2c_result_bundle.zip`
- `stage3_result_bundle.zip`
- `stage3b_result_bundle.zip`
- `stage4_result_bundle.zip`
- `stage11_result_bundle/` (extracted because the complete bundle is larger
  than GitHub's single-file limit)

They are retained for provenance and independent re-analysis. The Stage 3B
bundle is also the immutable input to the CPU-only Stage 4 matched-intervention
notebook. The Stage 4 bundle is the independently audited output of that
deterministic analysis.

The Stage 11 directory is the complete returned full-development result, not a
curated subset. It contains:

- all raw unit, geometry, planning, native-planner, fidelity, probe-selection,
  certificate, and contrast tables;
- the exact run configuration and task/split manifests;
- the result ZIP manifest and all 45 files covered by it;
- six selected matched-ARGA checkpoints;
- evaluation-only physical decoders and geometry-whitening references;
- exact-restoration, asset, cache, and transition provenance;
- the run log and summary plot.

An independent copy audit verified all 45 manifest file sizes and SHA256
digests with zero missing or mismatched files. The compact reconstructed result
is `../stage11_full_development_audit.json`.
