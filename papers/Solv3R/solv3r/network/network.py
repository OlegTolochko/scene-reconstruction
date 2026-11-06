# Adapted from DVLT (https://github.com/nv-tlabs/dvlt), Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
# Modified for Solv3R: the encoder, view tokens and output heads of DVLT around the Solv3R block; no step schedule,
# training or configuration code.
"""The Solv3R network: image encoder, solver block, output heads."""
from __future__ import annotations

import torch
from torch import nn

from .block import SolverBlock
from .dinov2 import DinoV2Encoder
from .heads import CameraHead, DecoderHead
from .rope import PositionGetter, RotaryPositionEmbedding2D

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Solv3RNetwork(nn.Module):
    """Every view is a token sequence [camera token, 4 register tokens, patch tokens] of width C.

    encoder      DINOv2 ViT-B/14 image features (frozen, from DVLT)
    block        the Solv3R block, applied once per solver call to the states of all views
    ray_head     dense rays (direction, origin) per pixel     \
    depth_head   dense depth and its confidence per pixel      } output heads (frozen, from DVLT)
    camera_head  pose encoding from the camera token          /
    """

    def __init__(self, patch_size=14, embed_dim=768, num_heads=12, num_register_tokens=4, head_dim=384, head_depth=2,
                 head_heads=6, decode_chunk_size=8):
        super().__init__()
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.patch_start_idx = 1 + num_register_tokens
        self.decode_chunk_size = decode_chunk_size

        self.encoder = DinoV2Encoder(patch_size=patch_size, embed_dim=embed_dim, num_register_tokens=num_register_tokens)
        self.rope = RotaryPositionEmbedding2D(frequency=100)
        self.position_getter = PositionGetter()
        self.camera_token = nn.Parameter(torch.zeros(1, 2, 1, embed_dim))     # [reference view, other views]
        self.register_token = nn.Parameter(torch.zeros(1, 1, num_register_tokens, embed_dim))
        self.block = SolverBlock(embed_dim, num_heads, rope=self.rope)
        head_kw = dict(in_dim=embed_dim, embed_dim=head_dim, depth=head_depth, num_heads=head_heads,
                       patch_size=patch_size, rope=self.rope)
        self.ray_head = DecoderHead(out_dim=6, head_type="linear", **head_kw)
        self.depth_head = DecoderHead(out_dim=2, head_type="conv", **head_kw)
        self.camera_head = CameraHead(head_dim, head_dim, pose_dim=9)
        self.register_buffer("_mean", torch.FloatTensor(_RESNET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.FloatTensor(_RESNET_STD).view(1, 3, 1, 1), persistent=False)

    # ------------------------------------------------------------------ inputs
    def encode_images(self, images):
        """images (S, 3, H, W) in [0, 1] -> patch features (S, H/14 * W/14, C)."""
        return self.encoder((images - self._mean) / self._std)

    def view_tokens(self, S, reference_first=True):
        """Camera and register tokens (S, 5, C). With reference_first, view 0 takes the reference camera token."""
        reg = self.register_token.expand(1, S, -1, -1).reshape(S, self.num_register_tokens, -1)
        if reference_first:
            cam = torch.cat([self.camera_token[:, 0:1].expand(1, 1, -1, -1),
                             self.camera_token[:, 1:2].expand(1, S - 1, -1, -1)], dim=1).reshape(S, 1, -1)
        else:
            cam = self.camera_token[0, 1:2].expand(S, -1, -1)
        return torch.cat([cam, reg], dim=1)

    def rope_positions(self, S, H, W, device):
        """(S, P, 2) patch-grid positions (from 1); the camera and register tokens sit at 0."""
        pos = self.position_getter(S, H // self.patch_size, W // self.patch_size, device) + 1
        special = torch.zeros(S, self.patch_start_idx, 2, device=device, dtype=pos.dtype)
        return torch.cat([special, pos], dim=1)

    # ------------------------------------------------------------------ outputs
    def decode(self, z, H, W, pos):
        """States (S, P, C) -> depth (S, H, W, 1), depth_conf (S, H, W), rays (S, H, W, 6), world_points (S, H, W, 3),
        world_points_conf, pose_enc (S, 9). The heads run per view, in chunks of decode_chunk_size views."""
        S = z.shape[0]
        rays, depth, cam_tokens = [], [], []
        for s in range(0, S, self.decode_chunk_size):
            f, p = z[s:s + self.decode_chunk_size], pos[s:s + self.decode_chunk_size]
            ray_out, ray_features = self.ray_head(f, H=H, W=W, patch_start_idx=self.patch_start_idx, pos=p)
            depth_out, _ = self.depth_head(f, H=H, W=W, patch_start_idx=self.patch_start_idx, pos=p)
            rays.append(ray_out)
            depth.append(depth_out)
            cam_tokens.append(ray_features[:, 0])
        cat = lambda xs: xs[0] if len(xs) == 1 else torch.cat(xs, dim=0)  # noqa: E731
        rays = cat(rays).movedim(1, -1).contiguous().view(S, H, W, 6)
        depth, conf = torch.split(cat(depth).movedim(1, -1).contiguous(), [1, 1], dim=-1)
        depth = torch.exp(depth.clamp(max=15.0)).view(S, H, W, 1)          # bf16-safe ceiling on the exponent
        conf = (torch.exp(conf.squeeze(-1)) + 1.0).view(S, H, W)
        return {
            "world_points": rays[..., 3:6] + rays[..., :3] * depth,         # ray origin + direction x depth
            "world_points_conf": conf,
            "depth": depth,
            "depth_conf": conf,
            "rays": rays,
            "pose_enc": self.camera_head(cat(cam_tokens)),
        }

    def decode_cameras(self, z, pos):
        """Pose encodings (S, 9) only: the ray-head transformer and the camera head, without the dense outputs."""
        tokens = [self.ray_head.tokens(z[s:s + self.decode_chunk_size], pos[s:s + self.decode_chunk_size])[:, 0]
                  for s in range(0, z.shape[0], self.decode_chunk_size)]
        return self.camera_head(torch.cat(tokens))
