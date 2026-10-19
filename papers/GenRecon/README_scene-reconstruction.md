# scene-reconstruction Notes for GenRecon

This directory is a copy of the official GenRecon codebase.

## Upstream

```text
Repository: https://github.com/kasothaphie/GenRecon
Imported commit: eaf1468118d20469d17079a4a19737297d2ef87b
Imported branch: main
Import date: 2026-10-19
Paper: https://arxiv.org/abs/2605.23888
Project page: https://kasothaphie.github.io/GenRecon/
Weights: https://kaldir.vc.cit.tum.de/genrecon/ (sparse_structure.pt, shape_slat.pt, texture_slat.pt)
License: MIT (see LICENSE)
```

## Local Changes

- `.gitmodules` was removed. Its only submodule, `o-voxel/third_party/eigen`
  (`https://gitlab.com/libeigen/eigen.git`, commit
  `21e4582d1739107337a03460c81412981130373e`, MPL-2.0), is used instead.
  Only the `Eigen/` headers and the license files are kept, since o-voxel only
  includes `<Eigen/Dense>`.

## Setup

Not set up yet. Upstream uses conda with Python 3.10, torch 2.6.0 (CUDA 12.x)
and `setup.sh` to build flash-attn, nvdiffrast, nvdiffrec, CuMesh, FlexGEMM
and o-voxel (see `README.md`).
