# Third-Party Notices

scDINO redistributes source code from the projects listed below. That code is
**not** covered by this repository's MIT `LICENSE`; each entry is governed by
its own terms, reproduced verbatim under `licenses/`.

If you redistribute scDINO, you must carry these notices and license texts with
it.

---

## 1. DINOv3 — Meta Platforms, Inc.

| | |
|---|---|
| Upstream | <https://github.com/facebookresearch/dinov3> |
| License | **DINOv3 License Agreement** (proprietary; *not* an OSI-approved license) |
| Full text | [`licenses/DINOv3-License-Agreement.md`](licenses/DINOv3-License-Agreement.md) |
| Vendored in | `src/scdino/models/backbones/dinov3.py`, everything **above** the `# scDINO integration:` banner |

That region (`DinoVisionTransformer`, the RoPE position embedding, the
attention/FFN blocks, and the `vit_*` constructors) is Meta's code. Everything
below the banner is original scDINO code under this repository's MIT license.

**Obligations this imposes on you and on downstream users:**

- **Redistribution requires shipping the Agreement.** Section 1 requires that a
  copy of the Agreement accompany any distribution of DINO Materials. That is
  why `licenses/DINOv3-License-Agreement.md` exists; do not delete it.
- **Acceptable-use restrictions apply.** The Agreement forbids use for military
  or warfare purposes, nuclear applications, espionage, and the development or
  use of guns or illegal weapons, and requires compliance with applicable trade
  controls and sanctions.
- **Publication acknowledgement.** If you publish research performed using DINO
  Materials, the Agreement requires that you acknowledge their use.
- **Governing law** is California, and Meta may amend the Agreement over time.

Read the full text before relying on this summary — it is a summary, not legal
advice, and the Agreement itself controls.

> If you need scDINO to be uniformly MIT-licensed, delete
> `src/scdino/models/backbones/dinov3.py` together with the `dinov3` model,
> config, and Lightning entry points. Everything else in the repository is MIT.

---

## 2. Lightly — Lightly AG

| | |
|---|---|
| Upstream | <https://github.com/lightly-ai/lightly> |
| License | **MIT** |
| Full text | [`licenses/lightly-MIT.txt`](licenses/lightly-MIT.txt) |
| Source headers | `Copyright (c) 2020. Lightly AG and its affiliates.` |

Vendored locations:

| File | What was taken |
|---|---|
| `src/scdino/models/utils.py` | Entire file — SSL model utilities (token masking, positional-embedding init, weight/momentum helpers) |
| `src/scdino/models/lightning/utils.py` | `DINOLoss`, `Center`, `center_mean`, `center_momentum`, `IBOTPatchLoss`, `KoLeoLoss`, `random_block_mask`, `random_block_mask_image`, `update_momentum`, `cosine_schedule` |
| `src/scdino/models/backbones/dinov2.py` | `ProjectionHead`, `DINOv2ProjectionHead`, `MaskedVisionTransformer`, `MaskedVisionTransformerTIMM`, `update_drop_path_rate` |
| `src/scdino/models/backbones/dino.py` | `ProjectionHead`, `DINOProjectionHead` |

Some of these are in turn Lightly's own reimplementations of methods published
by Meta (DINO, iBOT, DINOv2) and by Sablayrolles et al. (KoLeo); the relevant
papers and reference implementations are cited in the docstrings of each class.
The code as vendored here came from Lightly and is used under Lightly's MIT
license.

---

## Runtime dependencies

Libraries that scDINO imports but does **not** redistribute — `torch`,
`timm` (Apache-2.0), `transformers` (Apache-2.0), `lightning`, `cellpose`,
`scikit-learn`, `umap-learn`, and others — are declared in `pyproject.toml` and
resolved to exact versions in `uv.lock`. Their licenses ship with the installed
packages and are not reproduced here.

## Pretrained weights

Model weights fetched at runtime (for example `facebook/dinov2-*` and
`facebook/dinov3-*` via `transformers`) carry their own licenses, which are
separate from the code licenses above. The `facebook/dinov3-*` checkpoints are
released under the same DINOv3 License Agreement reproduced in `licenses/`.
