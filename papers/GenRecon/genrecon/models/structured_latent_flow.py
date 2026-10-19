from functools import partial
from typing import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules import sparse as sp
from ..modules.cond_3D.sparse_aggregation_net import MultiViewFeatAggregator
from ..modules.cond_3D.sparse_projection import Projection
from ..modules.sparse.transformer import ModulatedSparseTransformerCrossBlock
from ..modules.transformer import AbsolutePositionEmbedder
from ..modules.utils import convert_module_to, manual_cast, str_to_dtype
from .sparse_elastic_mixin import SparseTransformerElasticMixin
from .sparse_structure_flow import TimestepEmbedder


class SLatFlowModel(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        pe_mode: Literal["ape", "rope"] = "ape",
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        dtype: str = "float32",
        use_checkpoint: bool = False,
        share_mod: bool = False,
        initialization: str = "vanilla",
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.pe_mode = pe_mode
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.initialization = initialization
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = str_to_dtype(dtype)

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_channels, 6 * model_channels, bias=True))

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)

        self.input_layer = sp.SparseLinear(in_channels, model_channels)

        self.blocks = nn.ModuleList(
            [
                ModulatedSparseTransformerCrossBlock(
                    model_channels,
                    cond_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    attn_mode="full",
                    use_checkpoint=self.use_checkpoint,
                    use_rope=(pe_mode == "rope"),
                    rope_freq=rope_freq,
                    share_mod=self.share_mod,
                    qk_rms_norm=self.qk_rms_norm,
                    qk_rms_norm_cross=self.qk_rms_norm_cross,
                )
                for _ in range(num_blocks)
            ]
        )

        self.out_layer = sp.SparseLinear(model_channels, out_channels)

        self.initialize_weights()
        self.convert_to(self.dtype)

        self.proj_linears = None

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to(self, dtype: torch.dtype) -> None:
        """
        Convert the torso of the model to the specified dtype.
        """
        self.dtype = dtype
        self.blocks.apply(partial(convert_module_to, dtype=dtype))

    def initialize_weights(self) -> None:
        if self.initialization == "vanilla":
            # Initialize transformer layers:
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

            self.apply(_basic_init)

            # Initialize timestep embedding MLP:
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

            # Zero-out adaLN modulation layers in DiT blocks:
            if self.share_mod:
                nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
            else:
                for block in self.blocks:
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            # Zero-out output layers:
            nn.init.constant_(self.out_layer.weight, 0)
            nn.init.constant_(self.out_layer.bias, 0)

        elif self.initialization == "scaled":
            # Initialize transformer layers:
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.normal_(module.weight, std=np.sqrt(2.0 / (5.0 * self.model_channels)))
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

            self.apply(_basic_init)

            # Scaled init for to_out and ffn2
            def _scaled_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.normal_(module.weight, std=1.0 / np.sqrt(5 * self.num_blocks * self.model_channels))
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

            for block in self.blocks:
                block.self_attn.to_out.apply(_scaled_init)
                block.cross_attn.to_out.apply(_scaled_init)
                block.mlp.mlp[2].apply(_scaled_init)

            # Initialize input layer to make the initial representation have variance 1
            nn.init.normal_(self.input_layer.weight, std=1.0 / np.sqrt(self.in_channels))
            nn.init.zeros_(self.input_layer.bias)

            # Initialize timestep embedding MLP:
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

            # Zero-out adaLN modulation layers in DiT blocks:
            if self.share_mod:
                nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
            else:
                for block in self.blocks:
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            # Zero-out output layers:
            nn.init.constant_(self.out_layer.weight, 0)
            nn.init.constant_(self.out_layer.bias, 0)

    def add_projections(
        self, img_size, use_camera=False, refinement_factor=1, use_conv_net=False, use_self_attention=False
    ):
        assert not use_camera and not use_conv_net and not use_self_attention, "not implemented"

        self.projection = Projection(
            grid_res=self.resolution,
            img_patch_res=img_size // 16,
            refinement_factor=refinement_factor,
            global_img_tokens=5,
        )
        self.aggregator = MultiViewFeatAggregator(channels=self.cond_channels, refinement_factor=refinement_factor)
        self.proj_linears = nn.ModuleList(
            [nn.Linear(self.cond_channels, self.model_channels) for _ in range(self.num_blocks)]
        )
        for proj_linear in self.proj_linears:
            nn.init.zeros_(proj_linear.weight)
            nn.init.zeros_(proj_linear.bias)

    def get_3D_cond(self, img_feats_all, extrinsics, intrinsics, x_0):
        if not hasattr(self, "projection") or not hasattr(self, "aggregator"):
            return None
        projection, mask = self.projection(
            x_0, img_feats_all, extrinsics, intrinsics
        )  # [B, n_views, num_voxels, D_cond]
        projection_aggr = self.aggregator(projection, mask)
        return projection_aggr

    def forward(
        self,
        x: sp.SparseTensor,
        t: torch.Tensor,
        cond: dict,  # {"image_feat": [B, T, D], "aggregated_cond": SparseTensor or None}
        concat_cond: Optional[sp.SparseTensor] = None,
        **kwargs,
    ) -> sp.SparseTensor:

        if concat_cond is not None:
            x = sp.sparse_cat([x, concat_cond], dim=-1)

        h = self.input_layer(x)
        h = manual_cast(h, self.dtype)
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = manual_cast(t_emb, self.dtype)
        cond_2D = manual_cast(cond["cond_2D"], self.dtype)  # [B, T, D]
        cond_3D = cond["cond_3D"]
        if cond_3D is not None:
            cond_3D = manual_cast(cond_3D, self.dtype)

        assert cond_2D.ndim == 3, f"expected both conditions to have 3 dimensions but got {cond_2D.ndim}"
        assert (
            cond_2D.shape[0] == x.shape[0]
        ), f"expected condition 2D batch size {x.shape[0]} but got {cond_2D.shape[0]}"
        assert (
            cond_3D is None or cond_3D.shape[0] == x.shape[0]
        ), f"expected {x.shape[0]} but got {cond_3D.shape[0]} sparse voxels"

        if self.pe_mode == "ape":
            pe = self.pos_embedder(h.coords[:, 1:])
            h = h + manual_cast(pe, self.dtype)

        for i, block in enumerate(self.blocks):
            if self.proj_linears is not None:
                proj_update = cond_3D.replace(self.proj_linears[i](cond_3D.feats))
                h = h + proj_update
            h = block(h, t_emb, cond_2D)

        h = manual_cast(h, x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)
        return h


class ElasticSLatFlowModel(SparseTransformerElasticMixin, SLatFlowModel):
    """
    SLat Flow Model with elastic memory management.
    Used for training with low VRAM.
    """

    pass
