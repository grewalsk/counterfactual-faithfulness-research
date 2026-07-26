# Public checkpoint audit

Audit date: 2026-07-25.

Primary source: [official JEPA-WMs repository](https://github.com/facebookresearch/jepa-wms)
and its [Hugging Face model listing](https://huggingface.co/facebook/jepa-wms/tree/main).

| Model | Environment | Encoder | Listed checkpoint size | Pilot role |
|---|---|---:|---:|---|
| DINO-WM | Push-T | DINOv2 ViT-S/14 | 275 MB | **Stage 1 selection** |
| JEPA-WM | Push-T | DINOv2 ViT-S/14 | 212 MB | Stage 2 comparison |
| DINO-WM | PointMaze | DINOv2 ViT-S/14 | 275 MB | Low-cost second environment |
| JEPA-WM | PointMaze | DINOv2 ViT-S/14 | 212 MB | Low-cost second environment |
| DINO-WM | Wall | DINOv2 ViT-S/14 | 269 MB | Alternative low-cost replication |
| JEPA-WM | Wall | DINOv2 ViT-S/14 | 212 MB | Alternative low-cost replication |
| DINO-WM | MetaWorld | DINOv2 ViT-S/14 | 281 MB | Later manipulation breadth |
| JEPA-WM | MetaWorld | DINOv2 ViT-S/14 | 212 MB | Later manipulation breadth |
| V-JEPA-2-AC | DROID | V-JEPA-2 ViT-G/16 | 3.66 GB | Tier 3 only; not Push-T |

The optional DINOv2 image decoder is listed at 3.64 GB. It is not needed for
latent prediction error, paired effect error, or latent goal distance. The
Stage 1 notebook patches only the copied evaluation configuration to disable
decoder heads; it leaves the public predictor checkpoint and encoder intact.

## Stage 1 downloads

- `facebook/jepa-wms:dino_wm_pusht.pth.tar` (listed as 275 MB).
- `facebookresearch/dinov2:dinov2_vits14` encoder weights and Torch Hub source.
- The JEPA-WMs Git repository at commit
  `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0`.
- No trajectory dataset.
- No image decoder.

The notebook hashes the files it actually finds in the Hugging Face and Torch
Hub caches and exports `checkpoints_manifest.json`. This runtime manifest, not
the listing above, is the record of the exact downloaded bytes.

## Licensing note

The GitHub source is distributed under the repository's license and the Hugging
Face model page identifies the model repository as CC-BY-NC-4.0. Verify that the
intended publication and distribution comply with the weight and encoder
licenses before redistributing checkpoints. The result ZIP contains hashes and
metrics, not model weights.

