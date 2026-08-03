# Stage 15 raw-evidence provenance

The full Stage 15 reader-failure evidence is intentionally not committed to
Git. It contains 382,416,576 bytes, primarily forty compressed target-token
arrays. The repository contains the complete hash manifest, deterministic
audit program, numerical outputs, and report.

## Bundle identity

- Source notebook commit:
  `f07a078c34b534bd8a0c5f760af8eba7ccf1bb5c`
- Saved run: `pilot_5703d35fb8b5`
- Google Drive folder:
  `My Drive/counterfactual_faithfulness_stage15_bundle/pilot_5703d35fb8b5`
- Manifest entries: 113
- Manifest bytes: 382,416,576
- Verified missing files: 0
- Verified SHA-256 mismatches: 0
- Construction trajectories: 0, 2, 4, 6
- Evaluation trajectories: 1, 3, 5, 7

`raw_full_manifest.json` is the authoritative file-level inventory. Do not
replace or edit the saved raw bundle after retrieving it.

## Local reproduction

Download the complete Drive folder without starting a Colab runtime, then run:

```bash
python scripts/audit_stage15_reader_failure.py \
  --root /absolute/path/to/pilot_5703d35fb8b5 \
  --output audits/stage15/reader_failure_audit
```

The program verifies every file against `full_manifest.json` before reading
the truth or target-token tensors. It is CPU-only and does not load JEPA, the
PushT simulator, CUDA, or a Colab runtime.

The current audit outputs are post-hoc. Reproducing them does not convert the
observed evaluation trajectories into fresh confirmation data.
