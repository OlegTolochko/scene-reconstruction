from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SelectedImages:
    """Scene-wide image set (used for the 3D cond by projecting onto a global
    voxel grid) plus one visibility-vetted view per kept chunk (used for the
    2D cross-attention cond). All extrinsics are OpenCV world-to-camera
    expressed in the **chunk-0 local frame** — the same frame `joint_decode`
    uses for the merged scene output.
    """

    scene_images_1024: torch.Tensor  # [N, 3, 1024, 1024]
    scene_images_512: torch.Tensor  # [N, 3, 512, 512]
    scene_intrinsics: torch.Tensor  # [N, 3, 3]
    scene_extrinsics_c0: torch.Tensor  # [N, 4, 4]

    cond2d_images_1024: list[torch.Tensor]  # K × [3, 1024, 1024]
    cond2d_images_512: list[torch.Tensor]  # K × [3, 512, 512]
    cond2d_intrinsics: list[torch.Tensor]  # K × [3, 3]
    cond2d_extrinsics_c0: list[torch.Tensor]  # K × [4, 4]

    chunk_indices: list[int]
