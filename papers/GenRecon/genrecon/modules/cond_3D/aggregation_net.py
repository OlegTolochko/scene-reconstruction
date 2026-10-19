import torch
import torch.nn.functional as F
from torch import nn

from ..attention import MultiHeadAttention, RotaryPositionEmbedder


class MultiViewFeatAggregator(nn.Module):
    def __init__(
        self,
        channels: int,
        resolution: int,
        use_camera: bool = False,
        refinement_factor: int = 1,
        use_conv_net: bool = False,
        use_self_attention: bool = False,
    ):

        super().__init__()

        if not isinstance(refinement_factor, int) or refinement_factor < 1:
            raise ValueError(f"refinement_factor must be an int >= 1, got {refinement_factor}")

        self.channels = channels
        self.resolution = resolution
        self.use_camera = use_camera
        self.refinement_factor = refinement_factor
        self.use_conv_net = use_conv_net
        self.use_self_attention = use_self_attention

        # --- IBRNet ---
        input_dim = 3 * channels
        if self.use_camera:
            input_dim = 3 * channels + 5
        self.feature_mlp = nn.Sequential(
            nn.Linear(input_dim, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
        )
        nn.init.zeros_(self.feature_mlp[-1].weight)
        nn.init.zeros_(self.feature_mlp[-1].bias)

        self.weight_mlp = nn.Sequential(
            nn.Linear(input_dim, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 1),
        )

        # --- Local Refinement ---
        if use_conv_net:
            self.ConvNet = nn.Sequential(
                nn.Conv3d(channels, channels, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv3d(channels, channels, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            )
            nn.init.zeros_(self.ConvNet[-1].weight)
            nn.init.zeros_(self.ConvNet[-1].bias)

        # --- Global Refinement ---
        if use_self_attention:
            self.num_heads = 4

            pos_embedder = RotaryPositionEmbedder(channels // self.num_heads, 3)
            coords = torch.meshgrid(*[torch.arange(resolution) for _ in range(3)], indexing="ij")
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            rope_phases = pos_embedder(coords)
            self.register_buffer("rope_phases", rope_phases)

            self.self_attention = MultiHeadAttention(
                channels,
                num_heads=self.num_heads,
                type="self",
                attn_mode="full",
                use_rope=True,
                qk_rms_norm=True,
            )
            self.output_layer = nn.Linear(channels, channels)
            nn.init.zeros_(self.output_layer.weight)
            nn.init.zeros_(self.output_layer.bias)

    def forward(self, feats, mask, camera_emb=None):
        """
        args:
            projections [B, num_views, num_voxels, D]
            mask [B, num_views, num_voxels], bool
        returns:
            aggregated projection: [B, num_voxels, D]
        """
        assert feats.ndim == 4, f"expected feats to have shape [B, N, V, D], got {feats.shape}"
        assert mask.ndim == 3, f"expected mask to have shape [B, N, V], got {mask.shape}"

        B, N, V, D = feats.shape
        R = self.resolution * self.refinement_factor

        assert D == self.channels, f"expected dimension {self.channels} but got {D}"
        assert mask.shape == (B, N, V), f"expected mask to be of shape {(B, N, V)} but got {mask.shape}"
        assert mask.dtype == torch.bool, f"expected boolean mask, got {mask.dtype}"

        if self.use_camera:
            assert camera_emb is not None, f"camera emb as input required"
            assert camera_emb.shape == (
                B,
                N,
                V,
                5,
            ), f"incorrect camera shape: expected shape {(B, N, V, 5)} but got {camera_emb.shape}"

        mask_f = mask.unsqueeze(-1)  # [B, N, V, 1]
        sum_valid = mask_f.sum(dim=1, keepdim=True).clamp_min(1)  # [B, 1, V, 1]
        all_invalid = (~mask).all(dim=1, keepdim=True).unsqueeze(-1)

        # One-pass mean/variance: var = E[x^2] - (E[x])^2. Avoids the second
        # full read of feats that the two-pass form (feats - mu)**2 requires.
        feats_masked = feats * mask_f
        sum1 = feats_masked.sum(dim=1, keepdim=True)  # [B, 1, V, D]
        sum2 = (feats_masked * feats).sum(dim=1, keepdim=True)  # [B, 1, V, D]
        mu = sum1 / sum_valid  # [B, 1, V, D]
        var = (sum2 / sum_valid - mu * mu).clamp_min(0)  # [B, 1, V, D]

        # Split-linear application of the first MLP layer. Equivalent to
        # Linear(3D[+5], D) on cat([feats, mu_exp, var_exp (+camera_emb)]) but
        # avoids materializing the [B, N, V, 3D] cat: mu/var are projected at
        # their native [B, 1, V, D] shape and broadcast-added across N.
        feat_W = self.feature_mlp[0].weight
        feat_b = self.feature_mlp[0].bias
        weight_W = self.weight_mlp[0].weight
        weight_b = self.weight_mlp[0].bias
        h = (
            F.linear(feats, feat_W[:, :D])
            + F.linear(mu, feat_W[:, D : 2 * D])
            + F.linear(var, feat_W[:, 2 * D : 3 * D])
            + feat_b
        )
        wl = (
            F.linear(feats, weight_W[:, :D])
            + F.linear(mu, weight_W[:, D : 2 * D])
            + F.linear(var, weight_W[:, 2 * D : 3 * D])
            + weight_b
        )
        if self.use_camera:
            h = h + F.linear(camera_emb, feat_W[:, 3 * D :])
            wl = wl + F.linear(camera_emb, weight_W[:, 3 * D :])

        # Run the remaining MLP layers (ReLU + Linears) on the combined output.
        for i in range(1, len(self.feature_mlp)):
            h = self.feature_mlp[i](h)
        feats_prime = h  # [B, N, V, D]

        for i in range(1, len(self.weight_mlp)):
            wl = self.weight_mlp[i](wl)
        w_logits = wl  # [B, N, V, 1]

        w_logits = w_logits.masked_fill(~mask_f, -1e9)
        w = torch.softmax(w_logits, dim=1)
        w = torch.where(all_invalid, torch.zeros_like(w), w)

        feats_delta = (feats_prime * w).sum(dim=1, keepdim=True)  # [B, 1, V, D]
        feats_aggr = mu + feats_delta
        feats_aggr = feats_aggr.squeeze(1)  # [B, V, D]

        if self.use_conv_net:
            assert (
                V == R**3
            ), f"voxel count mismatch #1: expected {R ** 3} for resolution {self.resolution}, refinement_factor {self.refinement_factor} but got {V}"
            feats_grid = feats_aggr.reshape(B, R, R, R, D)
            feats_grid = feats_grid.permute(0, 4, 1, 2, 3)  # [B, D, R, R, R]
            feats_refined = self.ConvNet(feats_grid)
            feats_refined = feats_refined.permute(0, 2, 3, 4, 1)  # [B, R, R, R, D]
            feats_refined = feats_refined.reshape(B, -1, D)  # [B, V, D]
            feats_aggr += feats_refined

        if self.refinement_factor > 1:
            assert (
                V == R**3
            ), f"voxel count mismatch #2: expected {R ** 3} for resolution {self.resolution}, refinement_factor {self.refinement_factor} but got {V}"
            feats_fine = feats_aggr.reshape(B, R, R, R, D)
            feats_fine = feats_fine.permute(0, 4, 1, 2, 3)  # [B, D, R, R, R]
            feats_coarse = F.interpolate(
                feats_fine,
                size=(self.resolution, self.resolution, self.resolution),
                mode="trilinear",
                align_corners=False,
            )
            feats_coarse = feats_coarse.permute(0, 2, 3, 4, 1)  # [B, R, R, R, D]
            feats_aggr = feats_coarse.reshape(B, -1, D)  # [B, V, D]
            assert (
                feats_aggr.shape[1] == self.resolution**3
            ), f"voxel count mismatch, expected {self.resolution **3} but got {feats_aggr.shape[1]}"

        if self.use_self_attention:
            assert (
                self.rope_phases.shape[0] == feats_aggr.shape[1]
            ), f"RoPE token count mismatch: expected {feats_aggr.shape[1]} but got {self.rope_phases.shape[0]}"
            assert (
                2 * self.rope_phases.shape[-1] == feats_aggr.shape[-1] // self.num_heads
            ), f"RoPe dim mismatch: expected {feats_aggr.shape[-1]// self.num_heads} but got {2*self.rope_phases.shape[-1]}"
            h = self.self_attention(feats_aggr, phases=self.rope_phases)
            h = self.output_layer(h)
            feats_aggr += h

        return feats_aggr
