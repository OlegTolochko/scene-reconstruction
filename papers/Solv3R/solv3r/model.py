"""Solv3R: a 3D looped Transformer used as a multi-view solver.

Solv3R keeps one state per view and updates all states with the same Transformer block at every call. The update has
no step index, so a solve can be continued, extended with new views, or partially held fixed at any call.

The image encoder and the output heads are those of DVLT (https://github.com/nv-tlabs/dvlt), kept frozen; the network
is defined in solv3r/network.
"""
from __future__ import annotations

import contextlib
import os
import re
from dataclasses import dataclass

import torch
import torch.nn as nn

from .network import Solv3RNetwork
from .network.cameras import pose_enc_to_extri_intri, rays_to_pose, to4x4

HF_REPO = "jgkwak/solv3r"                       # Hugging Face model repo of the released weights
WEIGHTS_FILE = "solv3r.safetensors"
LOCAL_WEIGHTS = os.path.join("checkpoints", WEIGHTS_FILE)

# Parameter names in the released checkpoint (DVLT-style) -> names in Solv3R. The rules are mutually exclusive.
_CHECKPOINT_KEYS = [
    (r"base\.patch_embed_encoder\.mask_token", None),       # DINOv2's masked-training token, unused at inference
    (r"base\.patch_embed_encoder\.(?!mask_token$)", "net.encoder."),
    (r"base\.camera_token", "net.camera_token"),
    (r"base\.register_token", "net.register_token"),
    (r"base\.ray_decoder\.", "net.ray_head."),
    (r"base\.depth_decoder\.", "net.depth_head."),
    (r"base\.camera_head\.", "net.camera_head."),
    (r"block\.", "net.block."),
    (r"(g_frame|g_global|inj\.weight|inj\.bias|target_rms)$", r"net.block.\1"),
]


def remap_checkpoint(state_dict):
    """Released checkpoint -> Solv3R state dict. Every key must match exactly one rule (a rule mapping to None drops
    the key), and every rule must be used."""
    out, used = {}, set()
    for key, value in state_dict.items():
        hits = [i for i, (pat, _) in enumerate(_CHECKPOINT_KEYS) if re.match(pat, key)]
        if len(hits) != 1:
            raise KeyError(f"checkpoint key {key} matches {len(hits)} remapping rules")
        pat, repl = _CHECKPOINT_KEYS[hits[0]]
        used.add(hits[0])
        if repl is not None:
            out[re.sub("^" + pat, repl, key)] = value
    if len(used) != len(_CHECKPOINT_KEYS):
        raise KeyError(f"unused remapping rules: {[_CHECKPOINT_KEYS[i][0] for i in sorted(set(range(len(_CHECKPOINT_KEYS))) - used)]}")
    return out


@dataclass
class Context:
    """Encoded views: x0 (S, P, C) start tokens and pos (S, P, 2) RoPE positions, for images of size (H, W)."""

    x0: torch.Tensor
    pos: torch.Tensor
    H: int
    W: int

    @property
    def S(self) -> int:
        return self.x0.shape[0]

    def subset(self, idx) -> "Context":
        return Context(self.x0[idx], self.pos[idx], self.H, self.W)

    @staticmethod
    def cat(ctxs) -> "Context":
        return Context(torch.cat([c.x0 for c in ctxs]), torch.cat([c.pos for c in ctxs]), ctxs[0].H, ctxs[0].W)


def autocast():
    """Solv3R runs in bfloat16 autocast, as in training and in all reported results."""
    if torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)
    return contextlib.nullcontext()


class Solv3R(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = Solv3RNetwork()

    @classmethod
    def from_pretrained(cls, checkpoint=None, device="cuda"):
        """checkpoint: a local .safetensors file or a Hugging Face model repo id.  By default checkpoints/solv3r.safetensors
        under the working directory if it exists, else the released weights (downloaded once into the Hugging Face cache)."""
        from safetensors.torch import load_file

        if checkpoint is None:
            checkpoint = LOCAL_WEIGHTS if os.path.isfile(LOCAL_WEIGHTS) else HF_REPO
        if not os.path.isfile(checkpoint):
            if checkpoint.endswith(".safetensors") or os.path.isabs(checkpoint) or os.path.isdir(checkpoint):
                raise FileNotFoundError(f"no checkpoint file at {checkpoint}")
            from huggingface_hub import hf_hub_download

            checkpoint = hf_hub_download(checkpoint, WEIGHTS_FILE)
        model = cls()
        model.load_state_dict(remap_checkpoint(load_file(checkpoint)), strict=True)
        model.requires_grad_(False)
        return model.eval().to(device)

    @property
    def device(self):
        return self.net.block.target_rms.device

    # ------------------------------------------------------------------ encode / start
    @torch.no_grad()
    def encode(self, images, reference_first=True, chunk=32) -> Context:
        """images (S, 3, H, W) in [0, 1], H and W multiples of 14 (see solv3r.images.load_images).

        The first view of a solve holds the reference camera token and defines the world frame. Views encoded on their
        own and added to an existing solve later take reference_first=False.
        """
        S, _, H, W = images.shape
        with autocast():
            z = torch.cat([self.net.encode_images(images[i:i + chunk].to(self.device)) for i in range(0, S, chunk)])
            x0 = torch.cat([self.net.view_tokens(S, reference_first), z], dim=1).float()
            pos = self.net.rope_positions(S, H, W, self.device)
        return Context(x0, pos, H, W)

    @staticmethod
    def start(ctx: Context):
        """The start state of every view is its encoding."""
        return ctx.x0.clone()

    @torch.no_grad()
    def add_views(self, z, ctx: Context, images):
        """Appends new views to an ongoing solve: their encodings join the state, the solved views keep theirs.  The new
        images must have the size of the solve: load them with load_images(paths, size=(ctx.H, ctx.W))."""
        if tuple(images.shape[-2:]) != (ctx.H, ctx.W):
            raise ValueError(f"new views are {tuple(images.shape[-2:])}, the solve is {(ctx.H, ctx.W)}: "
                             "load them with load_images(paths, size=(ctx.H, ctx.W))")
        new = self.encode(images, reference_first=False)
        return torch.cat([z, self.start(new).to(z.dtype)]), Context.cat([ctx, new])

    # ------------------------------------------------------------------ one solver call
    @torch.no_grad()
    def step(self, z, ctx: Context):
        """One call: frame attention within each view, global attention across all views, one state scale.
        step, cache and step_cached run in the caller's precision: call them inside autocast(), as solve() does."""
        return self.net.block(z, ctx.x0, ctx.pos)

    @torch.no_grad()
    def solve(self, ctx: Context, calls, z=None):
        """`calls` calls from the state z (default: the start state, the encodings). Returns the new state."""
        z = self.start(ctx) if z is None else z
        with autocast():
            for _ in range(calls):
                z = self.step(z, ctx)
        return z

    @torch.no_grad()
    def residual(self, z, ctx: Context, rows=None):
        """Relative change of the state (or of the given rows) under one more call: the stopping criterion."""
        with autocast():
            zn = self.step(z, ctx)
        if rows is not None:
            z, zn = z[rows], zn[rows]
        return float((zn.float() - z.float()).norm() / z.float().norm().clamp_min(1e-9))

    # ------------------------------------------------------------------ held views (key / value cache)
    @torch.no_grad()
    def cache(self, z_held, ctx_held: Context):
        """Cache of views held fixed: their global-attention keys / values and their contribution to the state scale.

        A held view's keys / values depend only on its own state (frame attention is per view), so caching them is
        exact. Their contribution to the state scale is computed with attention among the held views only, which is
        close to the full call when the reference view (view 0) is held, as in streaming. Calls with step_cached then
        update only the other views, at a cost proportional to their number.
        """
        return self.net.block.cache(z_held, ctx_held.x0, ctx_held.pos)

    @torch.no_grad()
    def step_cached(self, z, ctx: Context, cache):
        """One call for the views z against a cache of held views."""
        return self.net.block.forward_cached(z, ctx.x0, ctx.pos, cache)

    # ------------------------------------------------------------------ decoding
    @torch.no_grad()
    def decode(self, z, ctx: Context):
        """Outputs of every view: depth (S, H, W, 1), depth_conf (S, H, W), rays (S, H, W, 6), world_points
        (S, H, W, 3, ray origin + direction x depth), world_points_conf (= depth_conf), pose_enc (S, 9, camera head)."""
        with autocast():
            return self.net.decode(z, ctx.H, ctx.W, ctx.pos)

    @torch.no_grad()
    def decode_cameras(self, z, ctx: Context):
        """Camera-head output (pose_enc, (S, 9)) only, without the dense decoders."""
        with autocast():
            return self.net.decode_cameras(z, ctx.pos)


def cameras_from_pose_enc(pose_enc, hw):
    """pose_enc (S, 9) -> camera-to-world (S, 4, 4) in the frame of view 0, intrinsics (S, 3, 3) in pixels at hw."""
    extr, K = pose_enc_to_extri_intri(pose_enc[None].detach().float().cpu(), hw)
    return to4x4(extr)[0], K[0]


def cameras_from_rays(rays, depth_conf=None, patch_size=14):
    """Camera-to-world (S, 4, 4) and intrinsics (S, 3, 3) fitted to the predicted rays (the benchmark protocol)."""
    S, H, W, _ = rays.shape
    w = torch.ones_like(rays[..., 0]) if depth_conf is None else depth_conf
    c2w, K = rays_to_pose(rays[None].float(), w[None].float(), H, W, patch_size)
    return c2w[0], K[0]
