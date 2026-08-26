# Third-party notices and license boundary

Audit date: 2026-08-25. This inventory is informational and is not legal
advice.

The root `LICENSE` applies to original code, notebooks, tests, analysis
scripts, documentation, and figures authored for this repository. It does not
relicense third-party source, model weights, datasets, styles, or dependencies.
Those materials remain governed by their respective licenses.

The experiment notebooks do not bundle the model weights or upstream
repositories listed below. They download or clone exact revisions at runtime,
verify configured hashes, and package only our manifests, metrics, derived
arrays, and figures.

## JEPA-WMs source and checkpoints

- Project: **JEPA-WMs**, Meta Platforms, Inc. and contributors
- Source: <https://github.com/facebookresearch/jepa-wms>
- Source revision used here: `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0`
- Model repository: <https://huggingface.co/facebook/jepa-wms>
- Model revision used here: `9b9c41ef249466630dbf1a20e78391865d07b3b9`
- Upstream license at those distributions: **Creative Commons
  Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**
- License: <https://creativecommons.org/licenses/by-nc/4.0/>

The research uses the official frozen JEPA-WM and DINO-WM PushT and Wall
checkpoints. The files are downloaded into the runtime cache and are excluded
from result bundles and repository releases. Use of these materials must
remain within the upstream noncommercial terms and provide the required
attribution. Nothing in this repository implies endorsement by Meta or the
JEPA-WMs authors.

## DINOv2

- Project: **DINOv2**, Meta Platforms, Inc. and contributors
- Source: <https://github.com/facebookresearch/dinov2>
- License: **Apache License 2.0**
- License: <https://www.apache.org/licenses/LICENSE-2.0>

The notebooks use the public DINOv2 ViT-S/14 encoder through the upstream
JEPA-WMs loader. The verified encoder checkpoint hash is
`b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.
The encoder weight is not redistributed here.

## PushT, Wall, and DINO-WM lineage

- DINO-WM source lineage: <https://github.com/gaoyuezhou/dino_wm>, MIT
  License, copyright (c) 2025 gaoyuezhou.
- PushT implementation lineage: the Diffusion Policy repository at
  <https://github.com/real-stanford/diffusion_policy>, MIT License, copyright
  (c) 2023 Columbia Artificial Intelligence and Robotics Lab.
- The exact PushT and Wall code executed by these notebooks is obtained from
  the pinned JEPA-WMs revision above and therefore remains subject to that
  distribution's notices and license boundary.

The notebooks construct new simulator trajectories at runtime. No third-party
trajectory dataset is included in this repository or its compact result
bundles.

## Python and system dependencies

The notebooks install or import PyTorch, torchvision, NumPy, SciPy,
scikit-learn, Matplotlib, Gym, Pygame, Pymunk, OpenCV, Shapely, timm, Hydra,
OmegaConf, Hugging Face Hub, and related packages. These packages are not
vendored and retain their own licenses. Their names are used only to identify
runtime dependencies.

## NeurIPS style files

Files obtained from the official NeurIPS 2026 author kit are used only to
compile the manuscript and are not covered by this repository's MIT license.
They retain the notices and terms supplied by the official distribution.

## Anonymous-review note

`Copyright (c) 2026 The Authors` preserves double blindness. The copyright
notice can be replaced with the final author or institutional attribution
after review without changing the MIT grant. Anonymous reviewer artifacts
must not link to an identity-bearing repository.
