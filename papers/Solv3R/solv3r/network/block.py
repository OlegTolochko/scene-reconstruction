# Adapted from DVLT (https://github.com/nv-tlabs/dvlt), Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
# Modified for Solv3R: the step-conditioned depth scaling of DVLT's looped block is replaced by constant gates, an input
# injection and one state scale per call; adds the key / value cache of held views.
"""The Solv3R block: one call of the solver on the states of all views."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .layers import Attention, LayerScale, Mlp


class GatedAttentionBlock(nn.Module):
    """Pre-norm attention + MLP with constant gates (3, C): attention residual, MLP residual, output carry."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, init_values=0.01, rope=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qk_norm=True, rope=rope)
        self.ls1 = LayerScale(dim, init_values)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim, init_values)

    def forward(self, x, gates, pos=None, attn_fn=None):
        h = self.ls1(attn_fn(self.norm1(x)) if attn_fn is not None else self.attn(self.norm1(x), pos=pos))
        x = x + gates[0] * h
        x = x + gates[1] * self.ls2(self.mlp(self.norm2(x)))
        return gates[2] * x


def _attn_kv(attn, x):
    """Keys and values (1, heads, N, D) of the global attention for tokens x (1, N, C)."""
    B, N, C = x.shape
    qkv = attn.qkv(x).reshape(B, N, 3, attn.num_heads, C // attn.num_heads).permute(2, 0, 3, 1, 4)
    _, k, v = qkv.unbind(0)
    return attn.k_norm(k).to(v.dtype), v


def _attn_with_cache(attn, x, k_cache, v_cache):
    """Global attention of tokens x (1, n, C) over cached keys / values plus their own."""
    B, n, C = x.shape
    qkv = attn.qkv(x).reshape(B, n, 3, attn.num_heads, C // attn.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = attn.q_norm(q), attn.k_norm(k)
    keys = torch.cat([k_cache, k.to(v.dtype)], 2)
    values = torch.cat([v_cache, v], 2)
    o = F.scaled_dot_product_attention(q.to(v.dtype), keys, values)
    return attn.proj(o.transpose(1, 2).reshape(B, n, C))


class SolverBlock(nn.Module):
    """z (S, P, C) states, x0 (S, P, C) encodings, pos (S, P, 2) RoPE positions -> the states after one call.

    The encodings are injected at every call, frame attention runs within each view (with RoPE) and global attention
    across all views, and the result is rescaled to a fixed RMS. There is no step index.
    """

    def __init__(self, dim=768, num_heads=12, mlp_ratio=4.0, rope=None):
        super().__init__()
        self.frame_attn = GatedAttentionBlock(dim, num_heads, mlp_ratio, rope=rope)
        self.global_attn = GatedAttentionBlock(dim, num_heads, mlp_ratio)
        self.g_frame = nn.Parameter(torch.zeros(3, dim))
        self.g_global = nn.Parameter(torch.zeros(3, dim))
        self.inj_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.inj = nn.Linear(dim, dim)
        self.register_buffer("target_rms", torch.ones(4))

    def _frame(self, z, x0, pos):
        return self.frame_attn(z + self.inj(self.inj_norm(x0)), self.g_frame, pos)

    def _rescale(self, y, rms):
        return y * (self.target_rms[3] / rms).to(y.dtype)

    def forward(self, z, x0, pos):
        S, P, C = z.shape
        y = self._frame(z, x0, pos)
        y = self.global_attn(y.reshape(1, S * P, C), self.g_global)
        rms = y.float().pow(2).mean((1, 2), keepdim=True).clamp_min(1e-12).sqrt()
        return self._rescale(y, rms).reshape(S, P, C)

    def cache(self, z, x0, pos):
        """Global-attention keys / values of held views and their contribution to the state scale."""
        S, P, C = z.shape
        y = self._frame(z, x0, pos).reshape(1, S * P, C)
        k, v = _attn_kv(self.global_attn.attn, self.global_attn.norm1(y))
        sum_sq = self.global_attn(y, self.g_global).float().pow(2).sum()
        return dict(k=k, v=v, sum_sq=sum_sq, numel=z.numel())

    def forward_cached(self, z, x0, pos, cache):
        """One call for the views z, attending to themselves and to the cached held views."""
        S, P, C = z.shape
        y = self._frame(z, x0, pos).reshape(1, S * P, C)
        attn = self.global_attn.attn
        y = self.global_attn(y, self.g_global,
                             attn_fn=lambda x: _attn_with_cache(attn, x, cache["k"], cache["v"])).reshape(S, P, C)
        sum_sq = cache["sum_sq"] + y.float().pow(2).sum()
        return self._rescale(y, (sum_sq / (cache["numel"] + y.numel())).clamp_min(1e-12).sqrt())
