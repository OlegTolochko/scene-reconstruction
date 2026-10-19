from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Optional

import torch

from ..modules.cond_3D.projection import project_features_on_points
from ..modules.sparse import SparseTensor
from ..representations import Mesh, MeshWithVoxel
from .images_to_3d import ImagesTo3DPipeline
from .joint_decode import JointDecodeContext, joint_decode_shape, joint_decode_tex
from .samplers.multi_diff_orchestrator import MultiDiffusionOrchestrator
from .types import SelectedImages


@contextmanager
def _vram_peak(label: str):
    """Reset the CUDA peak memory counter, run the block, print peak on exit.

    No-op on CPU-only runs.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"[VRAM] {label}: peak {peak_gb:.2f} GB")


class FullSceneImagesTo3DPipeline(ImagesTo3DPipeline):

    # Peak memory in the aggregator scales as O(N * V * D). For large scenes +
    # many views the global projection tensor [1, N, V, D] would OOM, so we
    # stream the voxel dim through `project_features_on_points` + aggregator in chunks
    # of this size. Aggregation is pure per-voxel so this changes nothing
    # numerically. Shrink if you still OOM at higher `num_imgs_per_scene`.
    proj_batch_voxels = 8192

    # ─────────────────────────────────────────────────────────────────────
    # Global-grid geometry helpers
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _global_grid_layout(
        rel_t: list[torch.Tensor], R: int
    ) -> tuple[torch.Tensor, tuple[int, int, int], torch.Tensor]:
        """Return (offsets_R [K, 3], (GX, GY, GZ), mins [3]).

        Each chunk i's unit cube sits at center ``rel_t[i]`` in chunk-0 frame;
        its R-resolution sub-cube occupies ``[offsets_R[i] : offsets_R[i] + R]``
        per axis in the global grid.
        """
        rel = torch.stack([t.float() for t in rel_t])  # [K, 3]
        mins = rel.min(dim=0).values
        maxs = rel.max(dim=0).values
        offsets = ((rel - mins) * R).round().long()
        maxs_vox = ((maxs - mins) * R).round().long() + R
        GX, GY, GZ = maxs_vox.tolist()
        return offsets, (GX, GY, GZ), mins

    @staticmethod
    def _global_grid_points(mins: torch.Tensor, GX: int, GY: int, GZ: int, R: int) -> torch.Tensor:
        """Chunk-0-frame world positions for every cell of the global grid.

        Returns ``[GX*GY*GZ, 3]``. Ordering matches a C-contiguous reshape into
        ``[GX, GY, GZ]`` so that ``pts.view(GX, GY, GZ, 3)[ox:ox+R, oy:oy+R, oz:oz+R]``
        yields chunk i's voxel positions (equivalent to the per-chunk Projection
        cube with extrinsics rebased to chunk-i).
        """
        ix, iy, iz = torch.meshgrid(
            torch.arange(GX),
            torch.arange(GY),
            torch.arange(GZ),
            indexing="ij",
        )
        grid = torch.stack([ix, iy, iz], dim=-1).reshape(-1, 3).float()
        return (grid + 0.5) / R + (mins - 0.5)

    # ─────────────────────────────────────────────────────────────────────
    # Cond assembly (shared by dense and sparse paths)
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _assemble_cond(cond_2D: torch.Tensor, cond_3D) -> dict:
        if cond_3D is None:
            neg_cond_3D = None
        elif isinstance(cond_3D, torch.Tensor):
            neg_cond_3D = torch.zeros_like(cond_3D)
        else:
            neg_cond_3D = cond_3D.replace(torch.zeros_like(cond_3D.feats))
        return {
            "cond": {"cond_2D": cond_2D, "cond_3D": cond_3D},
            "neg_cond": {"cond_2D": torch.zeros_like(cond_2D), "cond_3D": neg_cond_3D},
        }

    # ─────────────────────────────────────────────────────────────────────
    # Dense SS: global grid → aggregate once → crop per chunk
    # ─────────────────────────────────────────────────────────────────────

    def _build_global_dense_cond(
        self,
        flow_model,
        scene_feats: torch.Tensor,  # [1, N, T, D]
        cond2d_feats: torch.Tensor,  # [1, K, T, D]
        scene_ext_c0: torch.Tensor,  # [1, N, 4, 4]
        scene_intr: torch.Tensor,  # [1, N, 3, 3]
        rel_t: list[torch.Tensor],
        R: int,
        zero_3d_cond: bool = False,
    ) -> list[dict]:
        if not hasattr(flow_model, "projection") or not hasattr(flow_model, "aggregator"):
            return [self._assemble_cond(cond_2D=cond2d_feats[:, i], cond_3D=None) for i in range(cond2d_feats.shape[1])]

        offsets, (GX, GY, GZ), mins = self._global_grid_layout(rel_t, R)
        pts = self._global_grid_points(mins, GX, GY, GZ, R).to(
            device=self.device, dtype=scene_feats.dtype
        )  # [GX*GY*GZ, 3]

        if self.low_vram:
            flow_model.to(self.device)
        V_total = pts.shape[0]
        batch = self.proj_batch_voxels
        agg_chunks: list[torch.Tensor] = []
        for start in range(0, V_total, batch):
            end = min(start + batch, V_total)
            proj, valid, cam_emb = project_features_on_points(
                flow_model.projection,
                pts[start:end],
                scene_feats,
                scene_ext_c0,
                scene_intr,
            )
            sub = flow_model.aggregator(proj, valid, cam_emb)  # [1, end-start, D]
            agg_chunks.append(sub.squeeze(0))
            del proj, valid, cam_emb, sub
        cond_3D_global = torch.cat(agg_chunks, dim=0).unsqueeze(0)  # [1, V, D]
        del agg_chunks
        if self.low_vram:
            flow_model.cpu()

        D = cond_3D_global.shape[-1]
        cond_3D_vol = cond_3D_global.view(1, GX, GY, GZ, D)

        conds: list[dict] = []
        for i, (ox, oy, oz) in enumerate(offsets.tolist()):
            chunk_cond_3D = cond_3D_vol[:, ox : ox + R, oy : oy + R, oz : oz + R, :].reshape(1, R**3, D)
            if zero_3d_cond:
                chunk_cond_3D = torch.zeros_like(chunk_cond_3D)
            conds.append(self._assemble_cond(cond_2D=cond2d_feats[:, i], cond_3D=chunk_cond_3D))
        return conds

    # ─────────────────────────────────────────────────────────────────────
    # Sparse SLat: global sparse grid → aggregate once → split per chunk
    # ─────────────────────────────────────────────────────────────────────

    def _build_global_sparse_cond(
        self,
        flow_model,
        scene_feats: torch.Tensor,  # [1, N, T, D]
        cond2d_feats: torch.Tensor,  # [1, K, T, D]
        scene_ext_c0: torch.Tensor,  # [1, N, 4, 4]
        scene_intr: torch.Tensor,  # [1, N, 3, 3]
        coords_list: list[torch.Tensor],  # K × [K_i, 4] (batch, x, y, z), chunk-local
        rel_t: list[torch.Tensor],
        R: int,
        zero_3d_cond: bool = False,
    ) -> list[dict]:
        if not hasattr(flow_model, "projection") or not hasattr(flow_model, "aggregator"):
            return [self._assemble_cond(cond_2D=cond2d_feats[:, i], cond_3D=None) for i in range(cond2d_feats.shape[1])]

        offsets, (GX, GY, GZ), mins = self._global_grid_layout(rel_t, R)
        offsets_dev = offsets.to(coords_list[0].device)

        # Shift each chunk's local coords to global chunk-0 grid coords.
        global_xyz_parts = [c[:, 1:].long() + offsets_dev[i] for i, c in enumerate(coords_list)]
        all_global = torch.cat(global_xyz_parts, dim=0)  # [sum_K, 3]
        key = all_global[:, 0] * GY * GZ + all_global[:, 1] * GZ + all_global[:, 2]
        unique_key, inv = key.unique(return_inverse=True)  # [S_global], [sum_K]

        ux = unique_key // (GY * GZ)
        uy = (unique_key // GZ) % GY
        uz = unique_key % GZ
        global_xyz_unique = torch.stack([ux, uy, uz], dim=1).int()  # [S_global, 3]

        pts = (global_xyz_unique.float() + 0.5) / R + (mins.to(global_xyz_unique.device) - 0.5)
        pts = pts.to(device=self.device, dtype=scene_feats.dtype)

        if self.low_vram:
            flow_model.to(self.device)
        S_global = pts.shape[0]
        global_xyz_dev = global_xyz_unique.to(self.device)
        batch = self.proj_batch_voxels
        agg_chunks: list[torch.Tensor] = []
        for start in range(0, S_global, batch):
            end = min(start + batch, S_global)
            proj, valid, _ = project_features_on_points(
                flow_model.projection,
                pts[start:end],
                scene_feats,
                scene_ext_c0,
                scene_intr,
            )  # [1, N, end-start, D], [1, N, end-start]
            feats_SND = proj.squeeze(0).permute(1, 0, 2).contiguous()  # [end-start, N, D]
            mask_SN = valid.squeeze(0).permute(1, 0).contiguous()  # [end-start, N]
            sub_batch_col = torch.zeros(end - start, 1, dtype=torch.int32, device=self.device)
            sub_coords = torch.cat([sub_batch_col, global_xyz_dev[start:end]], dim=1)
            sub_sparse = SparseTensor(feats=feats_SND, coords=sub_coords)
            sub_agg = flow_model.aggregator(sub_sparse, mask_SN)  # SparseTensor [end-start, D]
            agg_chunks.append(sub_agg.feats)
            del proj, valid, feats_SND, mask_SN, sub_sparse, sub_agg
        if self.low_vram:
            flow_model.cpu()

        # Scatter back: each chunk's original coords have deterministic row-order
        # in `inv` — slice it by per-chunk sizes.
        agg_feats = torch.cat(agg_chunks, dim=0)  # [S_global, D]
        del agg_chunks
        per_chunk_concat = agg_feats[inv]  # [sum_K, D]
        K_sizes = [c.shape[0] for c in coords_list]
        per_chunk_feats = torch.split(per_chunk_concat, K_sizes, dim=0)

        conds: list[dict] = []
        for i, feats_i in enumerate(per_chunk_feats):
            chunk_cond_3D = SparseTensor(feats=feats_i, coords=coords_list[i])
            if zero_3d_cond:
                chunk_cond_3D = chunk_cond_3D.replace(torch.zeros_like(chunk_cond_3D.feats))
            conds.append(self._assemble_cond(cond_2D=cond2d_feats[:, i], cond_3D=chunk_cond_3D))
        return conds

    # ─────────────────────────────────────────────────────────────────────
    # Joint-frame decoders (unchanged from prior revision)
    # ─────────────────────────────────────────────────────────────────────

    def extract_coords_from_occ_logits(
        self,
        aggr_occ_logit_list: list[torch.Tensor],  # each [1, 1, ss_res, ss_res, ss_res]
        ss_res: int,
        target_resolution: int,
        threshold: float = 0.0,
    ) -> list[torch.Tensor]:
        """
        Threshold aggregated occupancy logits and return per-chunk SparseTensor
        coordinate tensors of shape [K, 4] (batch=0, x, y, z).

        ``threshold`` is applied to the raw logits (default 0.0, i.e. sigmoid > 0.5).
        Higher values keep only more confident voxels; lower values are more permissive.

        If ss_res differs from target_resolution the xyz coords are rescaled and
        deduplicated so they lie in [0, target_resolution - 1].
        """
        coords_list = []
        for i, logit in enumerate(aggr_occ_logit_list):
            occupied = logit[0, 0] > threshold  # [ss_res, ss_res, ss_res]
            xyz = torch.argwhere(occupied).int()  # [K, 3]
            if ss_res != target_resolution:
                scale = target_resolution / ss_res
                xyz = (xyz.float() * scale).round().long().unique(dim=0).int()
            batch_col = torch.zeros(xyz.shape[0], 1, dtype=torch.int32, device=xyz.device)
            coords_list.append(torch.cat([batch_col, xyz], dim=1))  # [K, 4]
        return coords_list

    @torch.no_grad()
    def joint_decode_sparse_structure(
        self,
        z_s_list: list[torch.Tensor],
        relative_transl: list[torch.Tensor],
        ss_res: int,
    ) -> list[torch.Tensor]:
        """Joint sparse-structure decode: place every chunk's latent on a
        merged dense canvas, run ``sparse_structure_decoder`` once, max-pool to
        ``ss_res`` grid, split per chunk.

        Returns a list of ``[1, 1, ss_res, ss_res, ss_res]`` occupancy-logit
        tensors, one per chunk, ready for ``extract_coords_from_occ_logits``.

        Why this is valid: the SS decoder is a dense 3D CNN with only
        channel-wise normalization (GroupNorm / ChannelLayerNorm), so it is
        translation- and size-equivariant. Running it on a larger merged input
        gives the same result as running it per-chunk with neighbor-aware
        boundaries — no training-distribution shift.
        """
        dec = self.models["sparse_structure_decoder"]
        device = self.device
        dtype = z_s_list[0].dtype

        R_lat = z_s_list[0].shape[-1]
        offsets_lat = [(t.to(device) * R_lat).round().long() for t in relative_transl]
        mins_lat = torch.stack(offsets_lat).min(dim=0).values
        maxs_lat = torch.stack(offsets_lat).max(dim=0).values + R_lat
        B, C = z_s_list[0].shape[:2]
        GX, GY, GZ = (maxs_lat - mins_lat).tolist()

        accum = torch.zeros(B, C, GX, GY, GZ, device=device, dtype=dtype)
        count = torch.zeros(1, 1, GX, GY, GZ, device=device, dtype=dtype)
        for z, off in zip(z_s_list, offsets_lat):
            x, y, zc = (off - mins_lat).tolist()
            accum[:, :, x : x + R_lat, y : y + R_lat, zc : zc + R_lat] += z
            count[:, :, x : x + R_lat, y : y + R_lat, zc : zc + R_lat] += 1
        joint_z = accum / count.clamp(min=1)

        if self.low_vram:
            dec.to(device)
        decoder_dtype = next(dec.parameters()).dtype
        decoded_joint = dec(joint_z.to(dtype=decoder_dtype))
        if self.low_vram:
            dec.cpu()

        # Per-chunk: decoder upsamples R_lat → R_dec_per_chunk. Same factor at
        # joint scale. Max-pool joint output to ss_res lattice if needed.
        upscale = decoded_joint.shape[-1] // joint_z.shape[-1]
        per_chunk_dec_res = R_lat * upscale
        ratio = per_chunk_dec_res // ss_res
        if ratio > 1:
            decoded_joint = torch.nn.functional.max_pool3d(decoded_joint.float(), ratio, ratio, 0)

        # Split at ss_res scale. Offsets scale linearly with resolution.
        offsets_ss = [(t.to(device) * ss_res).round().long() for t in relative_transl]
        mins_ss = torch.stack(offsets_ss).min(dim=0).values
        per_chunk_occ: list[torch.Tensor] = []
        for off in offsets_ss:
            x, y, zc = (off - mins_ss).tolist()
            per_chunk_occ.append(decoded_joint[:, :, x : x + ss_res, y : y + ss_res, zc : zc + ss_res].clone())
        return per_chunk_occ

    @torch.no_grad()
    def joint_decode_shape_slats(
        self,
        slats: list[SparseTensor],
        resolution: int,
        relative_transl: list[torch.Tensor],
        max_chunks_per_group: Optional[int] = None,
        max_inflated_voxels: Optional[int] = None,
        overlap_r_in: int = 16,
    ) -> tuple[Mesh, torch.Tensor, torch.Tensor, JointDecodeContext, dict]:
        """Merge chunk SLats, run the shape decoder (single-pass or chunked
        with overlap), extract a single scene mesh via one
        ``flexible_dual_grid_to_mesh`` call.

        Chunked decode activates when either cap (``max_chunks_per_group`` or
        ``max_inflated_voxels``) is set and the merged input exceeds it.

        Returns ``(scene_mesh, aabb, grid_size_fine, ctx, meta)``.
        """
        dec = self.models["shape_slat_decoder"]
        dec.set_resolution(resolution)
        if self.low_vram:
            dec.to(self.device)
            dec.low_vram = True
        scene_mesh, aabb, grid_size_fine, ctx, meta = joint_decode_shape(
            dec,
            slats,
            relative_transl,
            max_chunks_per_group=max_chunks_per_group,
            max_inflated_voxels=max_inflated_voxels,
            overlap_r_in=overlap_r_in,
        )
        if self.low_vram:
            dec.cpu()
            dec.low_vram = False
        return scene_mesh, aabb, grid_size_fine, ctx, meta

    @torch.no_grad()
    def joint_decode_tex_slats(
        self,
        slats: list[SparseTensor],
        ctx: JointDecodeContext,
        relative_transl: list[torch.Tensor],
    ) -> SparseTensor:
        """Joint tex decode using the shape decoder's joint subdivs. Returns
        one scene-wide ``SparseTensor`` of PBR attributes in ``[0, 1]``.
        """
        dec = self.models["tex_slat_decoder"]
        if self.low_vram:
            dec.to(self.device)
        out = joint_decode_tex(dec, slats, ctx, relative_transl)
        if self.low_vram:
            dec.cpu()
        return out

    # ─────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────

    def _encode_scene(self, images: torch.Tensor, resolution: int) -> torch.Tensor:
        """DINO-encode a stack of images [M, C, H, W] → [1, M, T, D]."""
        return self._encode_conditioning_tokens(images.unsqueeze(0), resolution=resolution)

    def _prep_extrinsics_intrinsics(
        self, ext: torch.Tensor, intr: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            ext.unsqueeze(0).to(self.device, dtype=dtype),
            intr.unsqueeze(0).to(self.device, dtype=dtype),
        )

    @torch.no_grad()
    def run(
        self,
        sel: SelectedImages,
        relative_translations: list[torch.Tensor],  # per kept chunk, [3]
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        pipeline_type: Optional[str] = None,
        occ_threshold: float = 0.0,
        zero_3d_cond: bool = False,
    ) -> tuple[MeshWithVoxel, list[torch.Tensor]]:
        pipeline_type = pipeline_type or self.default_pipeline_type
        if pipeline_type not in ("512", "1024"):
            raise ValueError(f"FullSceneImagesTo3DPipeline supports '512' and '1024' only; got {pipeline_type!r}")
        self._require_run_components(pipeline_type)

        num_chunks = len(sel.chunk_indices)
        torch.manual_seed(seed)

        res = int(pipeline_type)
        ss_res = 32 if pipeline_type == "512" else 64
        shape_model_key = f"shape_slat_flow_model_{pipeline_type}"
        tex_model_key = f"tex_slat_flow_model_{pipeline_type}"
        scene_imgs_for_stage = sel.scene_images_512 if pipeline_type == "512" else sel.scene_images_1024
        cond2d_imgs_for_stage = sel.cond2d_images_512 if pipeline_type == "512" else sel.cond2d_images_1024

        def _amp_ctx(model):
            if self.device.type == "cuda" and getattr(model, "dtype", None) in {torch.float16, torch.bfloat16}:
                return torch.autocast(device_type="cuda", dtype=model.dtype)
            return nullcontext()

        # ── One-shot DINO: scene set (512 always; 1024 if needed) + per-chunk 2D cond ─
        scene_feats_512 = self._encode_scene(sel.scene_images_512, 512)  # [1, N, T, D]
        cond2d_feats_512 = self._encode_scene(torch.stack(sel.cond2d_images_512), 512)  # [1, K, T, D]
        if pipeline_type == "1024":
            scene_feats_stage = self._encode_scene(scene_imgs_for_stage, 1024)
            cond2d_feats_stage = self._encode_scene(torch.stack(cond2d_imgs_for_stage), 1024)
        else:
            scene_feats_stage = scene_feats_512
            cond2d_feats_stage = cond2d_feats_512

        # ── Stage 1: Sparse structure ─────────────────────────────────────────
        ss_model = self.models["sparse_structure_flow_model"]
        scene_ext_ss, scene_intr_ss = self._prep_extrinsics_intrinsics(
            sel.scene_extrinsics_c0,
            sel.scene_intrinsics,
            ss_model.dtype,
        )
        ss_conds = self._build_global_dense_cond(
            ss_model,
            scene_feats_512.to(ss_model.dtype),
            cond2d_feats_512.to(ss_model.dtype),
            scene_ext_ss,
            scene_intr_ss,
            relative_translations,
            ss_model.resolution,  # flow model samples at latent res; ss_res is the decoded occ res.
            zero_3d_cond=zero_3d_cond,
        )
        noise_ss = [
            torch.randn(
                1,
                ss_model.in_channels,
                ss_res,
                ss_res,
                ss_res,
                dtype=getattr(ss_model, "dtype", torch.float32),
                device=self.device,
            )
            for _ in range(num_chunks)
        ]
        params_ss = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}

        if self.low_vram:
            ss_model.to(self.device)
        with _amp_ctx(ss_model):
            z_s_list = MultiDiffusionOrchestrator(self.sparse_structure_sampler).sample(
                ss_model,
                noise_ss,
                [c["cond"] for c in ss_conds],
                [c["neg_cond"] for c in ss_conds],
                relative_translations,
                **params_ss,
                tqdm_desc="Sampling sparse structure",
            )
        if self.low_vram:
            ss_model.cpu()
        del ss_conds, noise_ss

        with _vram_peak(f"joint_decode_sparse_structure ({num_chunks} chunks, ss_res={ss_res})"):
            aggr_occ_logit_list = self.joint_decode_sparse_structure(z_s_list, relative_translations, ss_res)
        del z_s_list

        # ── Stage 2: Shape SLat ───────────────────────────────────────────────
        shape_model = self.models[shape_model_key]
        coords_list = self.extract_coords_from_occ_logits(
            aggr_occ_logit_list,
            ss_res,
            shape_model.resolution,
            threshold=occ_threshold,
        )
        del aggr_occ_logit_list
        std = torch.tensor(self.shape_slat_normalization["std"])[None].to(self.device)
        mean = torch.tensor(self.shape_slat_normalization["mean"])[None].to(self.device)

        scene_ext_shape, scene_intr_shape = self._prep_extrinsics_intrinsics(
            sel.scene_extrinsics_c0,
            sel.scene_intrinsics,
            shape_model.dtype,
        )
        shape_conds = self._build_global_sparse_cond(
            shape_model,
            scene_feats_stage.to(shape_model.dtype),
            cond2d_feats_stage.to(shape_model.dtype),
            scene_ext_shape,
            scene_intr_shape,
            coords_list,
            relative_translations,
            shape_model.resolution,
            zero_3d_cond=zero_3d_cond,
        )
        noise_shape = [
            SparseTensor(
                feats=torch.randn(coords_list[i].shape[0], shape_model.in_channels, device=self.device),
                coords=coords_list[i],
            )
            for i in range(num_chunks)
        ]
        params_shape = {**self.shape_slat_sampler_params, **shape_slat_sampler_params}

        if self.low_vram:
            shape_model.to(self.device)
        with _amp_ctx(shape_model):
            shape_slat_raw = MultiDiffusionOrchestrator(self.shape_slat_sampler).sample(
                shape_model,
                noise_shape,
                [c["cond"] for c in shape_conds],
                [c["neg_cond"] for c in shape_conds],
                relative_translations,
                **params_shape,
                tqdm_desc="Sampling shape SLat",
            )
        if self.low_vram:
            shape_model.cpu()
        del shape_conds, noise_shape

        shape_slat_list = [s * std + mean for s in shape_slat_raw]
        del shape_slat_raw

        # ── Stage 3: Tex SLat ─────────────────────────────────────────────────
        tex_model = self.models[tex_model_key]
        tex_std = torch.tensor(self.tex_slat_normalization["std"])[None].to(self.device)
        tex_mean = torch.tensor(self.tex_slat_normalization["mean"])[None].to(self.device)

        shape_slat_norm = [(s - mean) / std for s in shape_slat_list]

        scene_ext_tex, scene_intr_tex = self._prep_extrinsics_intrinsics(
            sel.scene_extrinsics_c0,
            sel.scene_intrinsics,
            tex_model.dtype,
        )
        tex_conds = self._build_global_sparse_cond(
            tex_model,
            scene_feats_stage.to(tex_model.dtype),
            cond2d_feats_stage.to(tex_model.dtype),
            scene_ext_tex,
            scene_intr_tex,
            [sn.coords for sn in shape_slat_norm],
            relative_translations,
            tex_model.resolution,
            zero_3d_cond=zero_3d_cond,
        )
        noise_tex = [
            sn.replace(
                feats=torch.randn(
                    sn.coords.shape[0],
                    tex_model.in_channels - sn.feats.shape[1],
                    device=self.device,
                )
            )
            for sn in shape_slat_norm
        ]
        chunk_kwargs_tex = [{"concat_cond": sn} for sn in shape_slat_norm]
        params_tex = {**self.tex_slat_sampler_params, **tex_slat_sampler_params}

        if self.low_vram:
            tex_model.to(self.device)
        with _amp_ctx(tex_model):
            tex_slat_raw = MultiDiffusionOrchestrator(self.tex_slat_sampler).sample(
                tex_model,
                noise_tex,
                [c["cond"] for c in tex_conds],
                [c["neg_cond"] for c in tex_conds],
                relative_translations,
                **params_tex,
                tqdm_desc="Sampling texture SLat",
                chunk_kwargs=chunk_kwargs_tex,
            )
        if self.low_vram:
            tex_model.cpu()
        del tex_conds, noise_tex, chunk_kwargs_tex, shape_slat_norm

        tex_slat_list = [s * tex_std + tex_mean for s in tex_slat_raw]
        del tex_slat_raw

        # ── Decode ────────────────────────────────────────────────────────────
        # Chunked joint decode is enabled for the 1024 pipeline only; 512 keeps
        # the original single-pass behavior verbatim.
        #
        # max_inflated_voxels caps the *actual* number of voxels the per-group
        # forward will see (joint_slat coords inside the inflated AABB) — the
        # direct memory governor. Empirical reference: in=99K → 43 GB peak on
        # an 80 GB card; in=164K → 63 GB peak with no headroom for the next
        # group. 100K leaves ~35 GB headroom for cumulative state + safety.
        # max_chunks_per_group is a soft secondary bound for very sparse scenes.
        max_chunks_per_group = 10 if pipeline_type == "1024" else None
        max_inflated_voxels = 100_000 if pipeline_type == "1024" else None
        torch.cuda.empty_cache()
        with _vram_peak(f"joint_decode_shape ({num_chunks} chunks, res={res})"):
            scene_mesh, aabb, grid_size_fine, decode_ctx, joint_meta = self.joint_decode_shape_slats(
                shape_slat_list,
                res,
                relative_translations,
                max_chunks_per_group=max_chunks_per_group,
                max_inflated_voxels=max_inflated_voxels,
            )
        del shape_slat_list
        torch.cuda.empty_cache()

        with _vram_peak(f"joint_decode_tex ({num_chunks} chunks, res={res})"):
            joint_tex = self.joint_decode_tex_slats(tex_slat_list, decode_ctx, relative_translations)
        del tex_slat_list, decode_ctx, joint_meta
        torch.cuda.empty_cache()

        with _vram_peak("fill_holes (scene)"):
            scene_mesh.fill_holes()

        # Build one scene-wide MeshWithVoxel in the joint frame (chunk-0 local).
        voxel_shape = torch.Size([1, joint_tex.feats.shape[1], *grid_size_fine.tolist()])
        scene = MeshWithVoxel(
            scene_mesh.vertices,
            scene_mesh.faces,
            origin=aabb[0].cpu().tolist(),
            voxel_size=1.0 / res,
            coords=joint_tex.coords[:, 1:],
            attrs=joint_tex.feats,
            voxel_shape=voxel_shape,
            layout=self.pbr_attr_layout,
        )
        return scene, coords_list
