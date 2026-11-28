# Adapted from DVLT (https://github.com/nv-tlabs/dvlt), Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
# 2D rotary position embeddings originally distributed by Meta Platforms, Inc. and affiliates under Apache-2.0
# (inspired by https://github.com/meta-llama/codellama and https://github.com/naver-ai/rope-vit). Modified for Solv3R.
"""2D rotary position embeddings (RoPE) over the patch grid."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PositionGetter:
    """(y, x) grid positions of the patches, cached per grid size."""

    def __init__(self):
        self.position_cache = {}

    def __call__(self, batch_size, height, width, device):
        if (height, width) not in self.position_cache:
            y = torch.arange(height, device=device)
            x = torch.arange(width, device=device)
            self.position_cache[height, width] = torch.cartesian_prod(y, x)
        return self.position_cache[height, width].view(1, height * width, 2).expand(batch_size, -1, -1).clone()


class RotaryPositionEmbedding2D(nn.Module):
    """Rotates the first half of the head features by the y position and the second half by the x position."""

    def __init__(self, frequency=100.0, scaling_factor=1.0):
        super().__init__()
        self.base_frequency = frequency
        self.scaling_factor = scaling_factor
        self.frequency_cache = {}

    def _compute_frequency_components(self, dim, seq_len, device, dtype):
        key = (dim, seq_len, device, dtype)
        if key not in self.frequency_cache:
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency**exponents)
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq).to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            self.frequency_cache[key] = (angles.cos().to(dtype), angles.sin().to(dtype))
        return self.frequency_cache[key]

    @staticmethod
    def _rotate_features(x):
        d = x.shape[-1]
        x1, x2 = x[..., : d // 2], x[..., d // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(self, tokens, positions, cos_comp, sin_comp):
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]
        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def forward(self, tokens, positions):
        """tokens (B, heads, N, D), positions (B, N, 2) integer (y, x)."""
        assert tokens.size(-1) % 2 == 0, "Feature dimension must be even"
        assert positions.ndim == 3 and positions.shape[-1] == 2, "Positions must have shape (batch_size, n_tokens, 2)"
        feature_dim = tokens.size(-1) // 2
        max_position = int(positions.max()) + 1
        cos_comp, sin_comp = self._compute_frequency_components(feature_dim, max_position, tokens.device, tokens.dtype)
        vertical, horizontal = tokens.chunk(2, dim=-1)
        vertical = self._apply_1d_rope(vertical, positions[..., 0], cos_comp, sin_comp)
        horizontal = self._apply_1d_rope(horizontal, positions[..., 1], cos_comp, sin_comp)
        return torch.cat((vertical, horizontal), dim=-1)
