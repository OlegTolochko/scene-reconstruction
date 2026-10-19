import torch
from torch import nn

from .. import sparse as sp


class MultiViewFeatAggregator(nn.Module):
    def __init__(self, channels: int, refinement_factor: int = 1):

        super().__init__()
        self.channels = channels

        if not isinstance(refinement_factor, int) or refinement_factor < 1:
            raise ValueError(f"refinement_factor must be an int >= 1, got {refinement_factor}")
        self.refinement_factor = refinement_factor

        self.feature_mlp = nn.Sequential(
            nn.Linear(3 * channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
        )
        nn.init.zeros_(self.feature_mlp[-1].weight)
        nn.init.zeros_(self.feature_mlp[-1].bias)

        self.weight_mlp = nn.Sequential(
            nn.Linear(3 * channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 1),
        )

    def _collapse_refined_to_coarse(self, x: sp.SparseTensor) -> sp.SparseTensor:
        parent_idx = x.get_spatial_cache("projection_parent_idx")
        coarse_coords = x.get_spatial_cache("projection_coarse_coords")
        assert parent_idx is not None, "missing parent_idx cache from sparse projection"
        assert coarse_coords is not None, "missing coarse_coords cache from sparse projection"

        parent_idx = parent_idx.long()
        fill_value = torch.finfo(x.feats.dtype).min

        coarse_feats = torch.full(
            (coarse_coords.shape[0], x.feats.shape[-1]),
            fill_value,
            device=x.feats.device,
            dtype=x.feats.dtype,
        )

        coarse_feats = torch.scatter_reduce(
            coarse_feats,
            dim=0,
            index=parent_idx.unsqueeze(1).expand(-1, x.feats.shape[-1]),
            src=x.feats,
            reduce="amax",
            include_self=True,
        )

        coarse_feats = torch.where(
            coarse_feats == fill_value,
            torch.zeros_like(coarse_feats),
            coarse_feats,
        )

        return x.replace(coarse_feats, coords=coarse_coords)

    def forward(self, x: sp.SparseTensor, mask: torch.Tensor) -> sp.SparseTensor:
        feats = x.feats
        S, N, D = feats.shape

        assert D == self.channels, f"expected dimension {self.channels} but got {D}"
        assert mask.shape == (S, N), f"expected mask to be of shape {(S, N)} but got {mask.shape}"

        mask_f = mask.unsqueeze(-1)  # [S, N 1]
        sum_valid = mask_f.sum(dim=1, keepdim=True).clamp_min(1)  # [S, 1, 1]
        all_invalid = (~mask).all(dim=1, keepdim=True).unsqueeze(-1)  # [S, 1, 1]

        mu = (feats * mask_f).sum(dim=1, keepdim=True) / sum_valid  # [S, 1, D]
        var = (((feats - mu) ** 2) * mask_f).sum(dim=1, keepdim=True) / sum_valid

        mu_exp = mu.expand(-1, N, -1)
        var_exp = var.expand(-1, N, -1)

        h = torch.cat([feats, mu_exp, var_exp], dim=-1)  # [S, N, 3*D]
        h_flat = h.reshape(S * N, 3 * D)

        feats_prime = self.feature_mlp(h_flat).reshape(S, N, D)

        w_logits = self.weight_mlp(h_flat).reshape(S, N, 1)
        w_logits = w_logits.masked_fill(~mask_f, -1e9)
        w = torch.softmax(w_logits, dim=1)
        w = torch.where(all_invalid, torch.zeros_like(w), w)

        feats_delta = (feats_prime * w).sum(dim=1, keepdim=True)
        feats_aggr = (mu + feats_delta).squeeze(1)

        aggregated = x.replace(feats_aggr)

        if self.refinement_factor > 1:
            aggregated = self._collapse_refined_to_coarse(aggregated)

        return aggregated
