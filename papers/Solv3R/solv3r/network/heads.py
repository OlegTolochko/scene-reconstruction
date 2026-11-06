# Adapted from DVLT (https://github.com/nv-tlabs/dvlt), Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
# ResidualConvBlock, the conv output stage of DecoderHead and uv_grid are adapted from MoGe
# (https://github.com/microsoft/MoGe), Copyright (c) Microsoft Corporation, MIT License (below).
# Modified for Solv3R: inference only.
#
# MIT License
#
# Copyright (c) Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Output heads: the dense ray and depth heads and the camera head."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .layers import Block


def uv_grid(width, height, aspect_ratio, dtype=None, device=None):
    """Normalized view-plane UV grid (width, height, 2), MoGe's normalized_view_plane_uv convention."""
    span_x = aspect_ratio / (1 + aspect_ratio**2) ** 0.5
    span_y = 1 / (1 + aspect_ratio**2) ** 0.5
    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype, device=device)
    v = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing="xy")
    return torch.stack([u, v], dim=-1)


def _concat_uv(x: Tensor, aspect_ratio: float) -> Tensor:
    h, w = x.shape[-2:]
    uv = uv_grid(w, h, aspect_ratio=aspect_ratio, dtype=x.dtype, device=x.device)
    uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
    return torch.cat([x, uv], dim=1)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels, hidden_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.GroupNorm(1, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, hidden_channels, 3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(max(1, hidden_channels // 32), hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 3, padding=1, padding_mode="replicate"),
        )

    def forward(self, x):
        return x + self.layers(x)


class DecoderHead(nn.Module):
    """Per-view transformer (proj_in -> blocks -> LayerNorm) followed by a dense output stage:
    "linear" (per-patch linear + pixel shuffle) or "conv" (progressive upsampling with UV inputs, after MoGe)."""

    def __init__(self, in_dim, out_dim, embed_dim=384, depth=2, num_heads=6, patch_size=14, mlp_ratio=4.0, rope=None,
                 head_type="linear", dim_upsample=(256, 128, 64), num_res_blocks=2, hidden_dim_multiplier=2,
                 last_conv_channels=32):
        super().__init__()
        self.patch_size = patch_size
        self.head_type = head_type
        self.proj_in = nn.Linear(in_dim, embed_dim)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qk_norm=True, rope=rope) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-5)
        if head_type == "conv":
            in_channels = [embed_dim] + list(dim_upsample[:-1])
            self.upsample_blocks = nn.ModuleList()
            for in_ch, out_ch in zip(in_channels, dim_upsample):
                block = nn.ModuleList([nn.Sequential(nn.ConvTranspose2d(in_ch + 2, out_ch, kernel_size=2, stride=2),
                                                     nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode="replicate"))])
                block.extend(ResidualConvBlock(out_ch, hidden_dim_multiplier * out_ch) for _ in range(num_res_blocks))
                self.upsample_blocks.append(block)
            self.output_block = nn.Sequential(
                nn.Conv2d(dim_upsample[-1] + 2, last_conv_channels, 3, padding=1, padding_mode="replicate"),
                nn.ReLU(inplace=True),
                nn.Conv2d(last_conv_channels, out_dim, 1),
            )
        else:
            self.head = nn.Linear(embed_dim, out_dim * patch_size * patch_size)

    def tokens(self, x, pos=None):
        """Transformer features (B, N, embed_dim) before the final norm; token 0 feeds the camera head."""
        x = self.proj_in(x)
        for blk in self.blocks:
            x = blk(x, pos=pos)
        return x

    def forward(self, x, H, W, patch_start_idx=0, pos=None):
        """x (B, N, in_dim) -> (B, out_dim, H, W) and the transformer features."""
        B = x.shape[0]
        ph, pw = H // self.patch_size, W // self.patch_size
        features = self.tokens(x, pos)
        x = self.norm(features)[:, patch_start_idx:]
        if self.head_type == "conv":
            aspect_ratio = W / H
            x = x.permute(0, 2, 1).reshape(B, -1, ph, pw)
            for block in self.upsample_blocks:
                x = _concat_uv(x, aspect_ratio)
                for layer in block:
                    x = layer(x)
            x = F.interpolate(x, (H, W), mode="bilinear", align_corners=False)
            x = _concat_uv(x, aspect_ratio)
            with torch.amp.autocast("cuda", enabled=False):
                x = self.output_block(x.float())
        else:
            with torch.amp.autocast("cuda", enabled=False):
                x = self.head(x.float())
                x = x.transpose(-1, -2).view(B, -1, ph, pw)
                x = F.pixel_shuffle(x, self.patch_size)
        return x, features


class CameraHead(nn.Module):
    """Camera token (N, C) -> pose encoding (N, 9): translation, quaternion (xyzw), vertical / horizontal field of view."""

    def __init__(self, in_dim, hidden_dim=256, pose_dim=9):
        super().__init__()
        self.mlp = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.fc_pose = nn.Linear(hidden_dim, pose_dim)

    def forward(self, cls_token):
        x = self.mlp(cls_token)
        with torch.amp.autocast("cuda", enabled=False):
            pose = self.fc_pose(x.float())
            pose[:, 7:] = F.relu(pose[:, 7:])
        return pose
