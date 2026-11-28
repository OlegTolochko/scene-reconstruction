"""Streaming reconstruction with Solv3R: every frame is answered when it arrives, in the frame of the first frame.

The solver window holds
  - the anchors: the first frames, solved jointly once and then held fixed,
  - the memory: every memory_stride-th frame that left the recent window, the latest `memory` of them, held fixed,
  - the recent frames, which keep being refined,
  - the new frame, which starts from its encoding plus the solved offset of the previous frame.
Held frames enter every call through a key / value cache, so a call costs in proportion to the frames that move.
Every refresh_check frames one extra call measures how far the held frames are from the current solution; above
refresh_tau, that frame's calls update the whole window and the cache is rebuilt.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .model import Context, autocast, cameras_from_pose_enc


@dataclass
class StreamConfig:
    anchors: int = 8
    recent: int = 6
    memory: int = 64
    memory_stride: int = 16
    calls: int = 8
    anchor_calls: int = 16
    refresh_tau: float = 0.02
    refresh_check: int = 4


PRESETS = {
    "L8": StreamConfig(),
    "L4": StreamConfig(calls=4),
    "S4": StreamConfig(anchors=4, recent=4, memory=16, calls=4),
}


@dataclass
class StreamResult:
    c2w: np.ndarray                     # (T, 4, 4) camera-to-world when each frame arrived (frame 0 = identity)
    c2w_exit: np.ndarray                # (T, 4, 4) when each frame left the recent window (its last update)
    intrinsics: np.ndarray              # (T, 3, 3) on arrival, pixels at the model resolution
    depth: np.ndarray | None = None     # (T, h, w) on arrival, in the scale of c2w (with depth=True)
    depth_conf: np.ndarray | None = None
    refreshed: list = field(default_factory=list)
    ms_per_frame: float = 0.0


def _relative(pose_enc, hw):
    """Rotations and centres of all window rows relative to row 0 (camera-to-world)."""
    c2w, K = cameras_from_pose_enc(pose_enc, hw)
    c2w = c2w.double()
    R0, C0 = c2w[0, :3, :3], c2w[0, :3, 3]
    return R0.T[None] @ c2w[:, :3, :3], (c2w[:, :3, 3] - C0[None]) @ R0, K


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class StreamingSolver:
    def __init__(self, model, config: StreamConfig | str = "L8", depth=False):
        self.model = model
        self.cfg = PRESETS[config] if isinstance(config, str) else config
        self.depth = depth
        self.x0, self.states = {}, {}
        self.pos = None
        self.memory, self.recent = [], []
        self.R, self.C, self.R_last, self.C_last, self.R_exit, self.C_exit = {}, {}, {}, {}, {}, {}
        self.K, self.depths, self.confs = {}, {}, {}
        self.cache, self.cache_key, self.n_refresh = None, None, 0
        self.refreshed, self.ms = [], []
        self.t = -1

    @torch.no_grad()
    def push(self, image):
        """Adds the next frame (3, h, w) in [0, 1] at the model size. Returns the frames answered by this call."""
        self.t += 1
        t, a = self.t, self.cfg.anchors
        ctx1 = self.model.encode(image[None], reference_first=(t == 0))
        self.x0[t], self.pos, self.hw = ctx1.x0[0], ctx1.pos[:1], (ctx1.H, ctx1.W)
        if t < a - 1:
            return []
        _sync(self.model.device)
        t0 = time.time()
        if t == a - 1:
            wids, held = list(range(a)), []
        else:
            held = list(range(a)) + self.memory
            wids = held + self.recent + [t]
        ctx = Context(torch.stack([self.x0[f] for f in wids]), self.pos.expand(len(wids), -1, -1), *self.hw)
        z = self.model.start(ctx)
        z_start = z.clone()
        for r, f in enumerate(wids):
            if f in self.states:
                z[r] = self.states[f].to(z.dtype)
        if t > a - 1 and (t - 1) in self.states:
            r = wids.index(t - 1)
            z[-1] = z_start[-1] + (self.states[t - 1].to(z.dtype) - z_start[r])
        hidx = list(range(len(held)))
        refresh = False
        if held and t > a and (t - a) % self.cfg.refresh_check == 0:
            refresh = self.model.residual(z, ctx, rows=hidx) > self.cfg.refresh_tau
        calls = self.cfg.anchor_calls if t == a - 1 else self.cfg.calls
        with autocast():
            if held and not refresh:
                key = (tuple(held), self.n_refresh)
                if self.cache_key != key:
                    self.cache, self.cache_key = self.model.cache(z[hidx], ctx.subset(hidx)), key
                act = list(range(len(held), len(wids)))
                ctx_act, zq = ctx.subset(act), z[act]
                for _ in range(calls):
                    zq = self.model.step_cached(zq, ctx_act, self.cache)
                z[act] = zq
            else:
                for _ in range(calls):
                    z = self.model.step(z, ctx)
        if refresh:
            self.n_refresh += 1
        pose_enc = self.model.decode_cameras(z, ctx)
        _sync(self.model.device)
        self.ms.append(1000 * (time.time() - t0))
        self.refreshed.append(bool(refresh))
        return self._place(z, ctx, wids, held, refresh, pose_enc)

    def _place(self, z, ctx, wids, held, refresh, pose_enc):
        t, a = self.t, self.cfg.anchors
        Rr, cr, K = _relative(pose_enc, self.hw)
        new = wids[-1]
        overlap = [r for r, f in enumerate(wids) if r > 0 and f != new and f in self.C]
        s = 1.0
        if overlap:
            Q, Y = cr[overlap], torch.stack([self.C[wids[r]] for r in overlap])
            den = float((Q * Q).sum())
            s = float((Q * Y).sum()) / den if den > 1e-12 else 1.0
        answered = list(range(a)) if t == a - 1 else [new]
        for f in answered:
            r = wids.index(f)
            self.R[f], self.C[f], self.K[f] = Rr[r], s * cr[r], K[r]
            if self.depth:
                with autocast():
                    p = self.model.decode(z[r:r + 1], ctx.subset([r]))
                self.depths[f] = (s * p["depth"][0, ..., 0].float()).cpu().numpy()
                self.confs[f] = p["depth_conf"][0].float().cpu().numpy()
        for r, f in enumerate(wids):
            if f not in held or refresh:
                self.R_last[f], self.C_last[f] = Rr[r], (self.C[new] if f == new else s * cr[r])
                self.states[f] = z[r].detach().clone()
        if t >= a:
            self.recent.append(t)
            if len(self.recent) > self.cfg.recent:
                f = self.recent.pop(0)
                self.R_exit[f], self.C_exit[f] = self.R_last[f], self.C_last[f]
                if f % self.cfg.memory_stride == 0:
                    self.memory.append(f)
                    if len(self.memory) > self.cfg.memory:
                        f = self.memory.pop(0)
                        self.states.pop(f, None)
                        self.x0.pop(f, None)                # encodings are kept only for frames that can be in a window
                else:
                    self.states.pop(f, None)
                    self.x0.pop(f, None)
        return answered

    def result(self) -> StreamResult:
        T = self.t + 1
        if T < self.cfg.anchors:
            raise ValueError(f"streaming needs at least {self.cfg.anchors} frames (the anchors), got {T}")
        R_exit, C_exit = dict(self.R_exit), dict(self.C_exit)
        for f in self.recent:
            R_exit[f], C_exit[f] = self.R_last[f], self.C_last[f]
        for f in range(min(T, self.cfg.anchors)):
            R_exit[f], C_exit[f] = self.R[f], self.C[f]

        def stack(R, C):
            E = np.tile(np.eye(4), (T, 1, 1))
            for f in range(T):
                E[f, :3, :3], E[f, :3, 3] = R[f].cpu().numpy(), C[f].cpu().numpy()
            return E

        return StreamResult(
            c2w=stack(self.R, self.C), c2w_exit=stack(R_exit, C_exit),
            intrinsics=np.stack([self.K[f].numpy() for f in range(T)]),
            depth=np.stack([self.depths[f] for f in range(T)]) if self.depth else None,
            depth_conf=np.stack([self.confs[f] for f in range(T)]) if self.depth else None,
            refreshed=self.refreshed, ms_per_frame=float(np.mean(self.ms[1:])) if len(self.ms) > 1 else 0.0)


def run_stream(model, frames, config="L8", depth=False):
    """frames: (T, 3, h, w) in [0, 1] at the model size, or an iterable of (3, h, w)."""
    solver = StreamingSolver(model, config, depth=depth)
    for image in frames:
        solver.push(image)
    return solver.result()
