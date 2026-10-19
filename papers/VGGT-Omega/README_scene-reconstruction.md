# scene-reconstruction Notes for VGGT-Omega

This directory is a copy of the official VGGT-Omega codebase.

## Upstream

```text
Repository: https://github.com/facebookresearch/vggt-omega
Imported commit: 48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6 (2026-09-22)
Import date: 2026-10-19
Weights: https://huggingface.co/facebook/VGGT-Omega (see README.md for the checkpoint table)
License: FAIR Noncommercial Research License (see LICENSE)
```

## Local Changes

- None. Copied without `.git`; upstream has no submodules or LFS files.
- The root `.gitignore` ignores every `data/` folder, so it has an exception
  for `training/data/`, which holds dataloader source code.

## Downloaded at Runtime (not tracked)

- Checkpoints from Hugging Face `facebook/VGGT-Omega`.
- `skyseg.onnx`, fetched by `visual_util.py` into the current working directory.
- Eval datasets, fetched by `eval/prepare.py`.

## Setup

Not set up yet. Upstream needs Python 3.10 or newer and torch 2.6 or newer
(see `README.md` and `pyproject.toml`).
