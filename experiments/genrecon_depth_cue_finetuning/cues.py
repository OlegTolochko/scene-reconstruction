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
"""

import torch
from torch import nn

N_CUES = 4


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


def add_cue_layers(agg: nn.Module) -> nn.Module:
    """Adds cue_feature and cue_weight to a dense MultiViewFeatAggregator. Set agg.cues = [B, N, V, N_CUES]
    before each forward (None = no cues)."""
    w = agg.feature_mlp[0].weight
    agg.cue_feature = nn.Linear(N_CUES, agg.feature_mlp[0].out_features, bias=False, device=w.device, dtype=w.dtype)
    agg.cue_weight = nn.Linear(N_CUES, agg.weight_mlp[0].out_features, bias=False, device=w.device, dtype=w.dtype)
    nn.init.zeros_(agg.cue_feature.weight)
    nn.init.zeros_(agg.cue_weight.weight)
    agg.cues = None

    def pre_hook(layer):
        # GenRecon's dense forward applies the first Linear itself and then calls mlp[1:] (ReLU, ...) one by one,
        # so the input of mlp[1] is the first layer's pre-activation [B, N, V, D].
        def hook(module, args):
            if agg.cues is None:
                return None
            h = args[0]
            assert h.shape[:-1] == agg.cues.shape[:-1], f"cues {tuple(agg.cues.shape)} do not match {tuple(h.shape)}"
            return (h + layer(agg.cues.to(layer.weight.dtype)).to(h.dtype),)

        return hook

    agg.feature_mlp[1].register_forward_pre_hook(pre_hook(agg.cue_feature))
    agg.weight_mlp[1].register_forward_pre_hook(pre_hook(agg.cue_weight))
    return agg
