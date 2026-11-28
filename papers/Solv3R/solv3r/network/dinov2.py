# Adapted from DVLT (https://github.com/nv-tlabs/dvlt), Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
# Originally from DINOv2 (https://github.com/facebookresearch/dinov2), Copyright (c) Meta Platforms, Inc. and affiliates,
# Apache-2.0; see also https://github.com/facebookresearch/dino and https://github.com/rwightman/pytorch-image-models.
# Modified for Solv3R: the ViT-B/14 with 4 registers only, patch-token features only.
"""DINOv2 ViT-B/14 (4 registers) image encoder."""
from __future__ import annotations

import math

import torch
from torch import nn

from .layers import Block, PatchEmbed


class DinoV2Encoder(nn.Module):
    """Image (B, 3, H, W), normalized -> patch tokens (B, H/14 * W/14, C) after a final affine-free LayerNorm."""

    def __init__(self, img_size=518, patch_size=14, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 num_register_tokens=4, interpolate_antialias=True):
        super().__init__()
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.patch_embed = PatchEmbed(patch_size=patch_size, embed_dim=embed_dim)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, init_values=1.0, norm_eps=1e-6)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)    # DVLT's feature normalization

    def interpolate_pos_encoding(self, x, h, w):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and h == w:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        h0, w0 = h // self.patch_size, w // self.patch_size
        M = int(math.sqrt(N))
        assert N == M * M
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
                                                    mode="bicubic", antialias=self.interpolate_antialias, size=(h0, w0))
        assert (h0, w0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def forward(self, x):
        _, _, h, w = x.shape
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, h, w)
        x = torch.cat((x[:, :1], self.register_tokens.expand(x.shape[0], -1, -1), x[:, 1:]), dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)[:, self.num_register_tokens + 1:]
