"""The aggregator variants of the MLP bench. To test an idea, add a class here, try it (seconds, no GPU) and
submit it by its name:

    uv run --with torch experiments/genrecon_mlp_bench/variants.py my_idea
    sbatch experiments/genrecon_mlp_bench/run.sbatch my_idea

A variant subclasses Aggregator (aggregator.py, which documents the steps and the depth cues) and overrides what
it changes. Rules of thumb:
  - create new layers in __init__ with zero_linear, so the variant starts exactly at GenRecon (the job prints
    whether it does);
  - index the cues by name: cues[..., BEHIND], cues[..., KNOWN], ...;
  - once a variant has been submitted, give a changed version a new name: its runs keep the old code's results.
"""

import math

import torch
from torch import nn

from aggregator import BEHIND, BEHIND_REL, CONFIDENCE, KNOWN, N_CUES, Aggregator, register, zero_linear


@register("depth_cue_first_layer")
class DepthInputs(Aggregator):
    """The depth cues as extra inputs of the first layer of both MLPs (variant A of genrecon_depth_cue_finetuning)."""

    def __init__(self, base):
        super().__init__(base)
        self.cue_feature = zero_linear(N_CUES, self.channels)
        self.cue_weight = zero_linear(N_CUES, self.channels)

    def view_features(self, feats, mu, var, mask, cues):
        h = self.first_layer(self.feature_mlp, feats, mu, var) + self.cue_feature(cues)
        return self.rest(self.feature_mlp, h)

    def view_logits(self, feats, mu, var, mask, cues):
        h = self.first_layer(self.weight_mlp, feats, mu, var) + self.cue_weight(cues)
        return self.rest(self.weight_mlp, h)


@register("depth_cue_first_layer_x10")
class FastDepthInputs(DepthInputs):
    """depth_cue_first_layer with the cues times 10. The cue layers are linear, so their outputs are 10 times as
    large, which under Adam is a 10 times larger learning rate for these layers alone."""

    gain = 10.0

    def view_features(self, feats, mu, var, mask, cues):
        return super().view_features(feats, mu, var, mask, self.gain * cues)

    def view_logits(self, feats, mu, var, mask, cues):
        return super().view_logits(feats, mu, var, mask, self.gain * cues)


@register("depth_cue_only_x10")
class FastDepthInputsOnly(FastDepthInputs):
    """depth_cue_first_layer_x10 with only the cue layers training."""

    train_only = ("cue_",)


@register("depth_routes_all")
class DepthRoutes(Aggregator):
    """Depth through small cue networks, on three routes into GenRecon's fusion. Only the cue networks train:
    GenRecon's two MLPs stay frozen, so nothing can memorise the rooms from their DINOv3 features, and what a run
    gains over its own step 0 comes from the cues. (After the cue modulator of genrecon_mlp_finetuning,
    feat/depth-cue-finetuning; here each route can run alone.)

        weights   per view: an offset on the view's logit, and a weight of the view in the mean and the variance.
                  Chooses among the views, so the output stays within what their features give.
        gate      per voxel: scales the fused feature. Towards 0 it becomes the feature of a voxel no view sees.
        residual  per voxel: adds a learned feature, a direction the views' features do not have.

    The networks see only depth: per view the cues, bumps around "the voxel is on the surface this view sees"
    (BEHIND = 0, at four widths) and steps for "in front of it" and "behind it"; per voxel the mean and the
    maximum of those over the views with depth, and the share of the voxel's views that have depth. The gate and
    the residual act only on voxels with depth in at least one view: elsewhere the output is GenRecon's.
    """

    routes = ("weights", "gate", "residual")
    use_depth = True  # False: the control, which knows where depth exists but not its values
    train_only = ("cue_",)
    widths = (0.5, 1.0, 2.0, 4.0)  # of the surface bumps, in voxels
    hidden = 128

    def __init__(self, base):
        super().__init__(base)
        n_view = 6 + len(self.widths)
        n_voxel = 2 * n_view + 1

        def trunk(n_in):
            return nn.Sequential(nn.Linear(n_in, self.hidden), nn.SiLU(), nn.Linear(self.hidden, self.hidden), nn.SiLU())

        if "weights" in self.routes:
            self.cue_view = trunk(n_view + n_voxel)
            self.cue_logit = zero_linear(self.hidden, 1)
            self.cue_stat = zero_linear(self.hidden, 1)
        if "gate" in self.routes or "residual" in self.routes:
            self.cue_voxel = trunk(n_voxel)
        if "gate" in self.routes:
            self.cue_gate = zero_linear(self.hidden, 1)
        if "residual" in self.routes:
            self.cue_residual = zero_linear(self.hidden, self.channels)
        self._logit = self._stat = None  # the per-view outputs of the current forward

    @torch.no_grad()
    def cue_inputs(self, cues, mask):
        """Per view [B, N, V, 10], per voxel [B, V, 21] and whether any view has depth at the voxel [B, V, 1], in
        float32. A view counts only where the aggregator uses it (mask) and its depth is known."""
        valid = mask.float()
        known = cues[..., KNOWN].float() * valid
        if self.use_depth:
            behind, rel, conf = (cues[..., i].float() * known for i in (BEHIND, BEHIND_REL, CONFIDENCE))
        else:
            behind = rel = conf = torch.zeros_like(known)
        on_surface = [known * torch.exp(-(behind**2) / (2 * w * w)) for w in self.widths]
        in_front = known * torch.sigmoid(-2 * behind)
        hidden = known * torch.sigmoid(2 * behind)
        view = torch.stack([behind / 8, rel * 2, conf / 3, known, *on_surface, in_front, hidden], dim=-1)  # 0 if unknown

        count = known.sum(1).unsqueeze(-1)  # [B, V, 1]
        any_known = count > 0
        mean = view.sum(1) / count.clamp_min(1)
        top = view.masked_fill(known.unsqueeze(-1) == 0, float("-inf")).amax(1)
        top = torch.where(any_known, top, torch.zeros_like(top))
        coverage = count / valid.sum(1).unsqueeze(-1).clamp_min(1)
        return view, torch.cat([mean, top, coverage], dim=-1), any_known.float()

    def forward(self, feats, mask, camera_emb=None):
        B, N, V, _ = feats.shape
        cues = self.cues if self.cues is not None else feats.new_zeros(B, N, V, N_CUES)
        assert cues.shape == (B, N, V, N_CUES), f"cues {tuple(cues.shape)} do not match the features {(B, N, V)}"
        view, voxel, any_known = self.cue_inputs(cues, mask)
        dtype = self.feature_mlp[0].weight.dtype
        if "weights" in self.routes:
            h = self.cue_view(torch.cat([view, voxel.unsqueeze(1).expand(-1, N, -1, -1)], dim=-1).to(dtype))
            self._logit, self._stat = self.cue_logit(h), self.cue_stat(h)
        try:
            out = super().forward(feats, mask, camera_emb)  # [B, V, D]
        finally:
            self._logit = self._stat = None
        if "gate" in self.routes or "residual" in self.routes:
            h = self.cue_voxel(voxel.to(dtype))
            any_known = any_known.to(out.dtype)
            if "gate" in self.routes:
                out = out * (1 + any_known * self.cue_gate(h).to(out.dtype))
            if "residual" in self.routes:
                # Scaled by 1 / sqrt(channels): under Adam every weight moves equally fast, and the residual's
                # many channels would otherwise outgrow the gate and the logits.
                out = out + any_known * self.cue_residual(h).to(out.dtype) / math.sqrt(self.channels)
        return out

    def statistics(self, feats, mask, cues):
        if self._stat is None:
            return super().statistics(feats, mask, cues)
        # GenRecon's mean and variance, with each view weighted by exp(its score): 1 at the start.
        a = (mask.unsqueeze(-1) * torch.exp(self._stat.float().clamp(-8, 8))).to(feats.dtype)
        n = a.sum(dim=1, keepdim=True)
        n = torch.where(n > 0, n, torch.ones_like(n))
        masked = feats * a
        mu = masked.sum(dim=1, keepdim=True) / n
        var = ((masked * feats).sum(dim=1, keepdim=True) / n - mu * mu).clamp_min(0)
        return mu, var

    def view_logits(self, feats, mu, var, mask, cues):
        logits = super().view_logits(feats, mu, var, mask, cues)
        return logits if self._logit is None else logits + self._logit.to(logits.dtype)


@register("depth_route_weights")
class DepthRouteWeights(DepthRoutes):
    """Only the route `weights` of depth_routes_all: depth chooses among the views."""

    routes = ("weights",)


@register("depth_route_gate")
class DepthRouteGate(DepthRoutes):
    """Only the route `gate` of depth_routes_all: depth scales the fused feature of a voxel."""

    routes = ("gate",)


@register("depth_route_residual")
class DepthRouteResidual(DepthRoutes):
    """Only the route `residual` of depth_routes_all: depth adds a learned feature to a voxel."""

    routes = ("residual",)


@register("depth_routes_control")
class DepthRoutesControl(DepthRoutes):
    """depth_routes_all without the depth values: the networks see only which views have depth at a voxel. What
    this run gains is not due to depth, so depth_routes_all has to beat it."""

    use_depth = False


@register("global_attention")
class GlobalAttention(Aggregator):
    """GenRecon's own global refinement of the aggregator (aggregation_net.py, use_self_attention), which the
    released model has switched off. After the fusion, one self-attention layer runs over all voxels of a chunk:

        f = f + output_layer(self_attention(f))

    4 heads, 3D rotary positions of the voxels' grid indices, RMS-normalised queries and keys, full attention over
    the resolution^3 = 4096 voxels. output_layer starts at zero, so the variant starts exactly at GenRecon. The
    layers carry GenRecon's names (self_attention.to_qkv, ..., output_layer), so its weights load into GenRecon's
    aggregator built with use_self_attention. The whole aggregator trains, as in the baseline.

    The attention needs a whole chunk's grid at once, in GenRecon's voxel order (x, then y, then z). In training
    every forward is one chunk. GenRecon's full-scene pipeline instead fuses the whole scene's grid in batches of
    voxels and then cuts out the chunks, so mesh.py sets refine_in_forward = False and calls refine() on every
    chunk's cut.
    """

    num_heads = 4
    rope_freq = (1.0, 10000.0)
    refine_in_forward = True

    class RMSNorm(nn.Module):
        """GenRecon's MultiHeadRMSNorm."""

        def __init__(self, dim, heads):
            super().__init__()
            self.scale = dim**0.5
            self.gamma = nn.Parameter(torch.ones(heads, dim))

        def forward(self, x):
            return (torch.nn.functional.normalize(x.float(), dim=-1) * self.gamma * self.scale).to(x.dtype)

    class SelfAttention(nn.Module):
        """GenRecon's MultiHeadAttention(type="self", attn_mode="full", use_rope=True, qk_rms_norm=True)."""

        def __init__(self, channels, num_heads):
            super().__init__()
            self.num_heads = num_heads
            self.to_qkv = nn.Linear(channels, 3 * channels)
            self.q_rms_norm = GlobalAttention.RMSNorm(channels // num_heads, num_heads)
            self.k_rms_norm = GlobalAttention.RMSNorm(channels // num_heads, num_heads)
            self.to_out = nn.Linear(channels, channels)

        @staticmethod
        def rotate(x, phases):  # GenRecon's RotaryPositionEmbedder.apply_rotary_embedding
            x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
            return torch.view_as_real(x_complex * phases.unsqueeze(-2)).reshape(*x.shape).to(x.dtype)

        def forward(self, x, phases):
            B, L, C = x.shape
            q, k, v = self.to_qkv(x).reshape(B, L, 3, self.num_heads, -1).unbind(dim=2)  # [B, L, H, C / H]
            q, k = self.rotate(self.q_rms_norm(q), phases), self.rotate(self.k_rms_norm(k), phases)
            h = torch.nn.functional.scaled_dot_product_attention(*(t.transpose(1, 2) for t in (q, k, v)))
            return self.to_out(h.transpose(1, 2).reshape(B, L, C))

    def __init__(self, base):
        super().__init__(base)
        self.resolution = base.resolution
        self.self_attention = self.SelfAttention(self.channels, self.num_heads)
        self.output_layer = zero_linear(self.channels, self.channels, bias=True)
        self._phases = None  # complex, so not a buffer: casting the module to bfloat16 would drop the imaginary part

    def rope_phases(self, device):
        """[R^3, head_dim / 2] complex phases of the voxels' grid indices (GenRecon's RotaryPositionEmbedder with 3
        axes: head_dim // 6 frequencies per axis, the rest padded with phase 0)."""
        if self._phases is None or self._phases.device != torch.device(device):
            R, head_dim = self.resolution, self.channels // self.num_heads
            coords = torch.stack(torch.meshgrid(*[torch.arange(R)] * 3, indexing="ij"), dim=-1).reshape(-1, 3)
            n = head_dim // 2 // 3
            freqs = self.rope_freq[0] / self.rope_freq[1] ** (torch.arange(n, dtype=torch.float32) / max(n, 1))
            angles = torch.outer(coords.reshape(-1).float(), freqs).reshape(R**3, 3 * n)
            angles = torch.cat([angles, angles.new_zeros(R**3, head_dim // 2 - 3 * n)], dim=-1)
            self._phases = torch.polar(torch.ones_like(angles), angles).to(device)
        return self._phases

    def refine(self, f):
        """f [B, R^3, D], the fused features of whole chunks -> f + output_layer(self_attention(f))."""
        assert f.shape[1] == self.resolution**3, \
            f"global_attention needs a whole chunk ({self.resolution}^3 voxels), got {f.shape[1]} voxels"
        return f + self.output_layer(self.self_attention(f, self.rope_phases(f.device)))

    def forward(self, feats, mask, camera_emb=None):
        out = super().forward(feats, mask, camera_emb)
        return self.refine(out) if self.refine_in_forward else out


@register("global_attention_only")
class GlobalAttentionOnly(GlobalAttention):
    """global_attention with only the attention training: GenRecon's fusion MLPs stay frozen, so what the run gains
    over its step 0 comes from the global refinement alone."""

    train_only = ("self_attention.", "output_layer.")


if __name__ == "__main__":  # python variants.py [NAME ...]: tries the named variants, or all of them
    import sys

    from aggregator import VARIANTS, preflight

    for name in sys.argv[1:] or list(VARIANTS):
        print(preflight(name))
