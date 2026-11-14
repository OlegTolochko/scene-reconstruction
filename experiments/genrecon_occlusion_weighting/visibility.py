"""Per-(view, voxel) visibility weights and the v-weighted GenRecon aggregators.

Pure torch, no GenRecon import, so the tests run without the GenRecon env. inject.py wires these into
GenRecon; the aggregator functions take the frozen aggregator module as `agg` and use its weights.

    v_i = valid_i * (c_i * g(Delta_i) + 1 - c_i),   Delta_i = z_i(x) - D_i(pi_i(x))
    g(Delta) = sigmoid((tau - Delta) / (s * tau)),   tau = max(tau_vox * voxel_size, rho * D_i)

then, per voxel, alpha = min(1, sum_i v_i / kappa) and v <- alpha * v + (1 - alpha) * valid, so a voxel
hidden in (almost) every view gets GenRecon's own weights back. The aggregator uses v for the mean and
variance and adds lam * log((v + eps) / (1 + eps)) to the view logits. That term is 0 at v = 1, so
v = valid reproduces GenRecon exactly (same arithmetic, same dtypes).
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Params:
    tau_vox: float = 1.5  # voxels; > sqrt(3)/2, the farthest a surface in the voxel is from its centre
    rho: float = 0.02  # relative depth error that grows with distance
    softness: float = 0.25  # s: g(0) = sigmoid(1/s) = 0.98, g(2 tau) = 0.02
    lam: float = 1.0  # lambda, weight of the log-visibility term in the view logits
    eps: float = 1e-3
    kappa: float = 1.0  # sum of v below which we blend back to GenRecon's weights


def visibility(
    points: torch.Tensor,  # [S, 3] voxel centres, in the frame of w2c
    w2c: torch.Tensor,  # [N, 4, 4] OpenCV world-to-camera (as GenRecon's scene extrinsics)
    K: torch.Tensor,  # [N, 3, 3] intrinsics of the full depth map, normalised (u, v in [0, 1))
    depth: torch.Tensor,  # [N, H, W] z-depth, same units as the camera frame of w2c
    conf: torch.Tensor,  # [N, H, W] confidence c in [0, 1]; 0 = unknown, keep GenRecon's weight
    valid: torch.Tensor,  # [N, S] bool, GenRecon's mask (voxel in front of the camera and inside its crop)
    voxel_size: float,  # voxel edge, in the units of the camera frame
    p: Params,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns v [N, S] (before the kappa fallback), Delta / tau [N, S] and c [N, S] at the voxels."""
    points, w2c, K = points.float(), w2c.float(), K.float()
    N, H, W = depth.shape

    cam = torch.einsum("nij,sj->nsi", w2c[:, :3, :3], points) + w2c[:, None, :3, 3]  # [N, S, 3]
    z = cam[..., 2]
    in_front = z > 1e-6
    uv = torch.einsum("nij,nsj->nsi", K, cam / torch.where(in_front, z, torch.ones_like(z))[..., None])
    # Nearest pixel: interpolating would mix depths across occlusion edges.
    px = torch.floor(uv[..., 0] * W).long().clamp(0, W - 1)
    py = torch.floor(uv[..., 1] * H).long().clamp(0, H - 1)
    idx = py * W + px  # [N, S]
    D = depth.reshape(N, -1).float().gather(1, idx)
    c = conf.reshape(N, -1).float().gather(1, idx)

    known = torch.isfinite(D) & (D > 0)
    c = torch.where(known, c, torch.zeros_like(c))
    D = torch.where(known, D, z)
    tau = torch.clamp(p.rho * D, min=p.tau_vox * voxel_size)
    delta_rel = (z - D) / tau
    g = torch.sigmoid((1.0 - delta_rel) / p.softness)  # = sigmoid((tau - Delta) / (s tau))
    v = valid.float() * (c * g + 1.0 - c)
    return v, delta_rel, c


def fallback(v: torch.Tensor, valid: torch.Tensor, kappa: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Blend each voxel back to GenRecon's weights when sum_i v_i < kappa. v, valid [N, S] -> v, alpha [S]."""
    alpha = (v.sum(0) / kappa).clamp(max=1.0)
    return alpha * v + (1.0 - alpha) * valid.float(), alpha


def _logit_bias(v: torch.Tensor, p: Params, dtype: torch.dtype) -> torch.Tensor:
    """lam * log((v + eps) / (1 + eps)): exactly 0 at v = 1, about lam * log(eps) at v = 0."""
    return (p.lam * torch.log((v.float() + p.eps) / (1.0 + p.eps))).to(dtype)


def dense_forward(agg, feats, mask, v, p: Params, camera_emb=None):
    """aggregation_net.MultiViewFeatAggregator.forward with weights v [B, N, V] instead of the mask."""
    if agg.use_conv_net or agg.use_self_attention or agg.refinement_factor > 1:
        raise NotImplementedError("the GenRecon configs use none of conv_net, self_attention, refinement")
    B, N, V, D = feats.shape
    assert mask.shape == (B, N, V) and v.shape == (B, N, V)

    mask_f = mask.unsqueeze(-1)
    w_f = (v * mask).unsqueeze(-1).to(feats.dtype)  # [B, N, V, 1]
    all_invalid = (~mask).all(dim=1, keepdim=True).unsqueeze(-1)
    # GenRecon divides by max(count, 1); counts are integers, so a tiny floor gives the same values at
    # v = valid and keeps fractional sums of v unchanged.
    sum_w = w_f.sum(dim=1, keepdim=True).clamp_min(1e-6)

    feats_w = feats * w_f
    mu = feats_w.sum(dim=1, keepdim=True) / sum_w
    var = ((feats_w * feats).sum(dim=1, keepdim=True) / sum_w - mu * mu).clamp_min(0)

    feat_W, feat_b = agg.feature_mlp[0].weight, agg.feature_mlp[0].bias
    weight_W, weight_b = agg.weight_mlp[0].weight, agg.weight_mlp[0].bias
    h = F.linear(feats, feat_W[:, :D]) + F.linear(mu, feat_W[:, D:2 * D]) + F.linear(var, feat_W[:, 2 * D:3 * D]) + feat_b
    wl = (F.linear(feats, weight_W[:, :D]) + F.linear(mu, weight_W[:, D:2 * D])
          + F.linear(var, weight_W[:, 2 * D:3 * D]) + weight_b)
    if agg.use_camera:
        h = h + F.linear(camera_emb, feat_W[:, 3 * D:])
        wl = wl + F.linear(camera_emb, weight_W[:, 3 * D:])
    for layer in agg.feature_mlp[1:]:
        h = layer(h)
    for layer in agg.weight_mlp[1:]:
        wl = layer(wl)

    wl = wl + _logit_bias(v, p, wl.dtype).unsqueeze(-1)
    wl = wl.masked_fill(~mask_f, -1e9)
    w = torch.softmax(wl, dim=1)
    w = torch.where(all_invalid, torch.zeros_like(w), w)
    return (mu + (h * w).sum(dim=1, keepdim=True)).squeeze(1)  # [B, V, D]


def sparse_forward(agg, x, mask, v, p: Params):
    """sparse_aggregation_net.MultiViewFeatAggregator.forward with weights v [S, N] instead of the mask."""
    feats = x.feats
    S, N, D = feats.shape
    assert mask.shape == (S, N) and v.shape == (S, N)

    mask_f = mask.unsqueeze(-1)
    w_f = (v * mask).unsqueeze(-1).to(feats.dtype)
    all_invalid = (~mask).all(dim=1, keepdim=True).unsqueeze(-1)
    sum_w = w_f.sum(dim=1, keepdim=True).clamp_min(1e-6)

    mu = (feats * w_f).sum(dim=1, keepdim=True) / sum_w
    var = (((feats - mu) ** 2) * w_f).sum(dim=1, keepdim=True) / sum_w

    h_flat = torch.cat([feats, mu.expand(-1, N, -1), var.expand(-1, N, -1)], dim=-1).reshape(S * N, 3 * D)
    feats_prime = agg.feature_mlp(h_flat).reshape(S, N, D)
    wl = agg.weight_mlp(h_flat).reshape(S, N, 1)

    wl = wl + _logit_bias(v, p, wl.dtype).unsqueeze(-1)
    wl = wl.masked_fill(~mask_f, -1e9)
    w = torch.softmax(wl, dim=1)
    w = torch.where(all_invalid, torch.zeros_like(w), w)

    aggregated = x.replace((mu + (feats_prime * w).sum(dim=1, keepdim=True)).squeeze(1))
    if agg.refinement_factor > 1:
        aggregated = agg._collapse_refined_to_coarse(aggregated)
    return aggregated
