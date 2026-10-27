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

## Checkpoints

- Checkpoints from Hugging Face `facebook/VGGT-Omega`.

## Setup

This setup uses the same torch and CUDA versions as
GenRecon (torch 2.9.1 with CUDA 13.0), so it also runs on Blackwell GPUs.

```bash
cd papers/VGGT-Omega
uv venv --python 3.12 --prompt vggt-omega .venv
source .venv/bin/activate
uv pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130
uv pip install -e .
```

