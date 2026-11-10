# Adapted from DVLT (https://github.com/nv-tlabs/dvlt), Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
# Originally from DINOv2 (https://github.com/facebookresearch/dinov2), Copyright (c) Meta Platforms, Inc. and affiliates,
# Apache-2.0. Modified for Solv3R: inference-only (no dropout, drop path, nested tensors or alternative attention backends).
"""Transformer primitives: attention, LayerScale, MLP, pre-norm block, patch embedding."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Attention(nn.Module):
    """Multi-head self-attention with optional QK normalization and 2D RoPE."""

    def __init__(self, dim, num_heads=8, qkv_bias=True, proj_bias=True, qk_norm=False, rope=None):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.rope = rope

    def forward(self, x: Tensor, pos=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, bias=True):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """Pre-norm attention + MLP block; LayerScale only when init_values is set."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, init_values=None, norm_eps=1e-5, qk_norm=False, rope=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = Attention(dim, num_heads=num_heads, qk_norm=qk_norm, rope=rope)
        self.ls1 = LayerScale(dim, init_values) if init_values else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim, init_values) if init_values else nn.Identity()

    def forward(self, x: Tensor, pos=None) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x), pos=pos))
        return x + self.ls2(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    """(B, 3, H, W) image -> (B, H/p * W/p, C) patch tokens."""

    def __init__(self, patch_size=14, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        assert H % self.patch_size[0] == 0 and W % self.patch_size[1] == 0, f"image size {H}x{W} is not a multiple of the patch size"
        return self.proj(x).flatten(2).transpose(1, 2)
