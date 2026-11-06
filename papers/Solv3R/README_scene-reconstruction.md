# scene-reconstruction Notes for Solv3R

This directory is a copy of the official Solv3R codebase.

## Upstream

```text
Repository: https://github.com/ubc-vision/solv3r
Imported commit: ef8da34f3736d6c241b3ba3cd074073c06e90b8f (2026-11-06)
Import date: 2026-11-06
Weights: https://huggingface.co/jgkwak/solv3r (revision c95998a2458e49232b2fbef7a247a2de9edb03fe)
License: Apache-2.0 for the code (see LICENSE), NVIDIA non-commercial license for the weights (see LICENSES/)
```

## Local Changes

- Copied without `.git`, `.github/`, `assets/` (README images and animations, about 14 MB) and `examples/data/`
  (sample inputs, about 15 MB). The README still links the removed images.
- The root `.gitignore` ignores every `data/` folder, so it has an exception
  for `eval/config/data/`, which holds the benchmark dataset configs.

## Checkpoints

- Download `solv3r.safetensors` (470 MB) from Hugging Face `jgkwak/solv3r`.
- For the experiments to work, save it as `papers/Solv3R/checkpoints/solv3r.safetensors`.

## Setup

This setup uses the same torch and CUDA versions as VGGT-Omega and GenRecon (torch 2.9.1 with CUDA 13.0), so it also
runs on Blackwell GPUs.

```bash
cd papers/Solv3R
uv venv --python 3.12 --prompt solv3r .venv
uv pip install --python .venv/bin/python torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/bin/python -e .
```
