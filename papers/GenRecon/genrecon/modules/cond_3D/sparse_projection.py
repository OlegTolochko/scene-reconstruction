from typing import Tuple

import torch
from torch import nn

from .. import sparse as sp
from .projection import project_points_to_patches


class Projection(nn.Module):
    def __init__(
        self,
        grid_res: int,
        img_patch_res: int,
        refinement_factor: int = 1,
        global_img_tokens: int = 5,
    ):
        super().__init__()

        self.grid_res = grid_res
        self.img_patch_res = img_patch_res
        self.global_img_tokens = global_img_tokens

        if not isinstance(refinement_factor, int) or refinement_factor < 1:
            raise ValueError(f"refinement_factor must be an int >= 1, got {refinement_factor}")
        self.refinement_factor = refinement_factor

        offsets = torch.meshgrid(
            *[torch.arange(refinement_factor) for _ in range(3)],
            indexing="ij",
        )
        offsets = torch.stack(offsets, dim=-1).reshape(-1, 3)
        self.register_buffer("subvoxel_offsets", offsets, persistent=False)

    def forward(
        self, x: sp.SparseTensor, cond: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor, eps=1e-6
    ) -> Tuple[sp.SparseTensor, torch.Tensor]:
        B, N, T, D = cond.shape

        assert (
            T == self.img_patch_res**2 + self.global_img_tokens
        ), f"expected {self.img_patch_res **2 + self.global_img_tokens} tokens but got {T}"
        assert (
            B,
            N,
            4,
            4,
        ) == extrinsics.shape, f"expected {(B, N, 4, 4)} as extrinsics shape but got {extrinsics.shape}"
        assert (
            B,
            N,
            3,
            3,
        ) == intrinsics.shape, f"expected {(B, N, 3, 3)} as intrinsics shape but got {intrinsics.shape}"

        coarse_coords = x.coords
        coarse_b_idx = coarse_coords[:, 0].long()
        coarse_xyz = coarse_coords[:, 1:].long()
        num_coarse = coarse_xyz.shape[0]

        if self.refinement_factor > 1:
            rf = self.refinement_factor
            offsets = self.subvoxel_offsets.to(device=coarse_coords.device, dtype=coarse_xyz.dtype)  # [K, 3]
            K = offsets.shape[0]

            fine_xyz = coarse_xyz[:, None, :] * rf + offsets[None, :, :]  # [S, K, 3]
            fine_xyz = fine_xyz.reshape(-1, 3)  # [S*K, 3]
            b_idx = coarse_b_idx.repeat_interleave(K)  # [S*K]
            parent_idx = torch.arange(num_coarse, device=coarse_coords.device).repeat_interleave(K)
            coords = torch.cat([b_idx[:, None], fine_xyz], dim=1).int()
        else:
            fine_xyz = coarse_xyz
            b_idx = coarse_b_idx
            parent_idx = torch.arange(num_coarse, device=coarse_coords.device)
            coords = coarse_coords

        xyz = fine_xyz.float()
        S = xyz.shape[0]

        coords_world = (xyz + 0.5) / float(self.grid_res * self.refinement_factor) - 0.5
        ones = torch.ones((S, 1), dtype=coords_world.dtype, device=coords_world.device)
        coords_hom = torch.cat((coords_world, ones), dim=-1)  # [S, 4]

        extr_S = extrinsics[b_idx].to(coords_hom.dtype)  # [S, N, 4, 4]
        intr_S = intrinsics[b_idx].to(coords_hom.dtype)  # [S, N, 3, 3]

        # coords_hom carries an explicit singleton view dim so the shared helper
        # broadcasts each voxel against its N cameras as [S, N] (and never
        # collapses to [S, S] when N == 1).
        _, valid, patch_ids = project_points_to_patches(
            coords_hom[:, None, :],  # [S, 1, 4]
            extr_S,  # [S, N, 4, 4]
            intr_S,  # [S, N, 3, 3]
            self.img_patch_res,
            self.global_img_tokens,
            eps=eps,
        )  # valid [S, N], patch_ids [S, N]

        # Flatten [B, N, T, D] -> [B*N*T, D] and index rows directly.
        cond_flat = cond.reshape(B * N * T, D)
        view_offset = torch.arange(N, device=cond.device, dtype=torch.long).view(1, N) * T  # [1, N]
        base = b_idx.view(S, 1) * (N * T)  # [S, 1]
        flat_index = (base + view_offset + patch_ids.long()).reshape(-1)  # [S*N]
        proj_feats = cond_flat.index_select(0, flat_index).reshape(S, N, D)  # [S, N, D]
        proj_feats *= valid.unsqueeze(-1)

        projected = x.replace(proj_feats, coords=coords)
        projected.clear_spatial_cache()
        projected.register_spatial_cache("projection_parent_idx", parent_idx)
        projected.register_spatial_cache("projection_coarse_coords", coarse_coords.clone())

        return projected, valid
