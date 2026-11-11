"""Per-view depth cues for GenRecon's dense (sparse-structure) aggregator, and the layers that feed them in.

Cues per (view, voxel), all 0 where the view has no valid depth at the voxel:
  0  clip(Delta / voxel, -8, 8)    Delta = voxel depth - observed depth; < 0 free space, > 0 occluded
  1  clip(Delta / D, -0.5, 0.5)    the same relative to the observed depth D (depth error grows with D)
  2  confidence, standardised with (conf_mean, conf_std) and clipped to [-3, 3]
  3  known: 1 where the voxel projects onto a valid depth pixel

add_cue_layers gives the aggregator two Linear(N_CUES, D) without bias, zero-initialised, whose outputs are
added to the first-layer pre-activations of feature_mlp and weight_mlp. That is the same as extra input
columns of those layers, but leaves GenRecon's parameters and checkpoint keys untouched. With zero weights,
or with agg.cues = None, the aggregator is exactly GenRecon's.

Extra cues (add_cue_layers(agg, extra=...), opt-in; they follow the 4 depth cues in this order):
  footprint   log2 of the voxel's size in DINOv3 patches in this view, f * voxel / (z * patch), clipped to
              [-4, 4]; f is the focal length of the crop the features come from. Needs no depth. Set as
              agg.footprint [B, N, V] by whoever projects the features (footprint_cue).
  similarity  cosine between this view's feature and the mean of the other valid views' features at the voxel
              (0 with fewer than 2 valid views). Computed in the aggregator from its own input.
Both are 0 where the view does not see the voxel, and all cues are 0 with agg.cues_enabled = False (control).
"""

import torch
import torch.nn.functional as F
from torch import nn

N_CUES = 4  # the depth cues of compute_cues
EXTRA_CUES = ("footprint", "similarity")


def compute_cues(
    points: torch.Tensor,  # [V, 3] voxel centres, in the frame of w2c
    w2c: torch.Tensor,  # [B, N, 4, 4] OpenCV world-to-camera of the depth maps
    K: torch.Tensor,  # [B, N, 3, 3] intrinsics normalised to the depth map (u, v in [0, 1))
    depth: torch.Tensor,  # [B, N, H, W] z-depth in the units of w2c; <= 0 or non-finite = unknown
    conf: torch.Tensor,  # [B, N, H, W]
    voxel_size: float,
    conf_mean: float = 0.0,
    conf_std: float = 1.0,
) -> torch.Tensor:
    """Returns the cues [B, N, V, N_CUES] in float32."""
    B, N, H, W = depth.shape
    w2c, K, points = w2c.float(), K.float(), points.float()

    cam = torch.einsum("bnij,vj->bnvi", w2c[..., :3, :3], points) + w2c[..., None, :3, 3]  # [B, N, V, 3]
    z = cam[..., 2]
    front = z > 1e-6
    uv = torch.einsum("bnij,bnvj->bnvi", K, cam / torch.where(front, z, torch.ones_like(z))[..., None])
    u, v = uv[..., 0], uv[..., 1]
    inside = front & (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
    # Nearest pixel: interpolating would mix depths across occlusion edges.
    px = torch.floor(u * W).long().clamp(0, W - 1)
    py = torch.floor(v * H).long().clamp(0, H - 1)
    idx = (py * W + px).reshape(B, N, -1)
    D = depth.reshape(B, N, -1).float().gather(2, idx).reshape(z.shape)
    c = conf.reshape(B, N, -1).float().gather(2, idx).reshape(z.shape)

    known = inside & torch.isfinite(D) & (D > 0) & torch.isfinite(c)
    D = torch.where(known, D, torch.ones_like(D))
    delta = z - D
    cues = torch.stack(
        [
            (delta / voxel_size).clamp(-8.0, 8.0),
            (delta / D).clamp(-0.5, 0.5),
            ((c - conf_mean) / conf_std).clamp(-3.0, 3.0),
            torch.ones_like(z),
        ],
        dim=-1,
    )
    return torch.where(known[..., None], cues, torch.zeros_like(cues))


def footprint_cue(
    points: torch.Tensor,  # [V, 3] voxel centres, in the frame of w2c
    w2c: torch.Tensor,  # [B, N, 4, 4] OpenCV world-to-camera of the feature crops
    K: torch.Tensor,  # [B, N, 3, 3] intrinsics normalised to the feature crop
    voxel_size: float,  # in the units of w2c
    patch_res: int,  # DINOv3 patches per crop side
) -> torch.Tensor:
    """log2 of the voxel's projected size in patches, [B, N, V] float32; 0 where it does not project into the crop."""
    w2c, K, points = w2c.float(), K.float(), points.float()
    cam = torch.einsum("bnij,vj->bnvi", w2c[..., :3, :3], points) + w2c[..., None, :3, 3]
    z = cam[..., 2]
    front = z > 1e-6
    zs = torch.where(front, z, torch.ones_like(z))
    uv = torch.einsum("bnij,bnvj->bnvi", K, cam / zs[..., None])
    inside = front & (uv[..., 0] >= 0) & (uv[..., 0] < 1) & (uv[..., 1] >= 0) & (uv[..., 1] < 1)
    f = (K[..., 0, 0] * K[..., 1, 1]).clamp_min(1e-12).sqrt()[..., None]  # [B, N, 1]
    r = torch.log2((f * voxel_size * patch_res / zs).clamp_min(1e-6)).clamp(-4.0, 4.0)
    return torch.where(inside, r, torch.zeros_like(r))


@torch.no_grad()
def similarity_cue(feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cosine between each view's feature and the mean of the other valid views', [B, N, V] float32.
    feats [B, N, V, D], mask [B, N, V] bool."""
    m = mask[..., None].to(feats.dtype)
    x = feats * m
    n = mask.sum(1, keepdim=True)  # [B, 1, V]
    others = (x.sum(1, keepdim=True) - x) / (n - 1).clamp_min(1)[..., None].to(feats.dtype)
    cos = F.cosine_similarity(feats.float(), others.float(), dim=-1, eps=1e-6)
    return torch.where(mask & (n >= 2), cos, torch.zeros_like(cos))


def add_cue_layers(agg: nn.Module, extra=()) -> nn.Module:
    """Adds cue_feature and cue_weight to a dense MultiViewFeatAggregator. Set agg.cues = [B, N, V, N_CUES]
    before each forward (None = no cues), and with "footprint" in extra also agg.footprint = [B, N, V]."""
    extra = tuple(extra)
    assert set(extra) <= set(EXTRA_CUES) and len(set(extra)) == len(extra), f"unknown or repeated extra cues {extra}"
    extra = tuple(c for c in EXTRA_CUES if c in extra)  # a fixed channel order
    n_in = N_CUES + len(extra)
    w = agg.feature_mlp[0].weight
    agg.cue_feature = nn.Linear(n_in, agg.feature_mlp[0].out_features, bias=False, device=w.device, dtype=w.dtype)
    agg.cue_weight = nn.Linear(n_in, agg.weight_mlp[0].out_features, bias=False, device=w.device, dtype=w.dtype)
    nn.init.zeros_(agg.cue_feature.weight)
    nn.init.zeros_(agg.cue_weight.weight)
    agg.cues = None
    agg.extra_cues = extra
    agg.cues_enabled = True
    agg.footprint = None
    agg._cue_input = None

    def agg_hook(module, args):
        # The full cue input of this forward: the depth cues, then the extra cues from the aggregator's input.
        if agg.cues is None:
            agg._cue_input = None
            return None
        x = agg.cues
        if extra:
            feats, mask = args[0], args[1]
            parts = [x.float()]
            for name in extra:
                if name == "footprint":
                    assert agg.footprint is not None, "footprint cue enabled but agg.footprint not set"
                    parts.append(agg.footprint.float()[..., None])
                else:
                    parts.append(similarity_cue(feats, mask)[..., None])
            x = torch.cat(parts, dim=-1)
        if not agg.cues_enabled:
            x = torch.zeros_like(x)
        agg._cue_input = x
        return None

    agg.register_forward_pre_hook(agg_hook)

    def pre_hook(layer):
        # GenRecon's dense forward applies the first Linear itself and then calls mlp[1:] (ReLU, ...) one by one,
        # so the input of mlp[1] is the first layer's pre-activation [B, N, V, D].
        def hook(module, args):
            c = agg._cue_input
            if c is None:
                return None
            h = args[0]
            assert h.shape[:-1] == c.shape[:-1], f"cues {tuple(c.shape)} do not match {tuple(h.shape)}"
            return (h + layer(c.to(layer.weight.dtype)).to(h.dtype),)

        return hook

    agg.feature_mlp[1].register_forward_pre_hook(pre_hook(agg.cue_feature))
    agg.weight_mlp[1].register_forward_pre_hook(pre_hook(agg.cue_weight))
    return agg
