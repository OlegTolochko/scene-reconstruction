from typing import *

import numpy as np
import torch
import tqdm

from ...modules.sparse.basic import SparseTensor


class MultiDiffusionOrchestrator:
    def __init__(self, sampler):
        self.sampler = sampler

    # ------------------------------------------------------------------
    # Prediction Aggregation
    # ------------------------------------------------------------------

    def _aggregate_overlaps(
        self, samples: list[Any], relative_transl: list[torch.Tensor], resolution: int
    ) -> list[Any]:
        """
        Average predictions in overlapping voxel regions across chunks (MultiDiffusion step).

        For each pair (i, j), maps chunk i's lattice coords into chunk j's frame via the
        integer offset  delta = round((rel_t_i - rel_t_j) * R),  finds the overlapping
        region, and accumulates a weighted sum.  Divides by the overlap count at the end
        so every overlapping voxel holds the mean of all chunks that cover it.

        Assumptions:
            - Voxels align exactly: (rel_t_i - rel_t_j) * resolution is integer-valued.
            - Dense tensors:  [B, C, R, R, R]
            - Sparse tensors: SparseTensor with coords [K, 4]  (batch, x, y, z)
        """
        if isinstance(samples[0], torch.Tensor):
            return self._aggregate_overlaps_dense(samples, relative_transl, resolution)
        else:
            return self._aggregate_overlaps_sparse(samples, relative_transl, resolution)

    def _aggregate_overlaps_boundary_sensitive(
        self,
        samples: list[Any],
        relative_transl: list[torch.Tensor],
        resolution: int,
        boundary_width: int = 2,
    ) -> list[Any]:
        if isinstance(samples[0], torch.Tensor):
            raise NotImplementedError(
                "boundary_sensitive aggregation is only implemented for sparse chunks "
                "(shape/tex SLat stages); the dense sparse-structure stage uses mean aggregation."
            )
        return self._aggregate_overlaps_boundary_sensitive_sparse(
            samples,
            relative_transl,
            resolution,
            boundary_width=boundary_width,
        )

    def _aggregate_overlaps_dense(
        self,
        samples: list[torch.Tensor],
        relative_transl: list[torch.Tensor],
        R: int,
    ) -> list[torch.Tensor]:
        B, C = samples[0].shape[:2]
        device = samples[0].device
        assert B == 1, f"expected batch size 1, got {B}"

        offsets = [((t * R).round().long()) for t in relative_transl]

        # Compute global bounding box
        mins = torch.stack(offsets).min(dim=0).values  # (3,)
        maxs = torch.stack(offsets).max(dim=0).values + R  # (3,)
        GX, GY, GZ = (maxs - mins).tolist()

        # Global accumulator
        accum = torch.zeros(B, C, GX, GY, GZ, device=device)
        count = torch.zeros(1, 1, GX, GY, GZ, device=device)

        for sample, offset in zip(samples, offsets):
            x, y, z = (offset - mins).tolist()
            accum[:, :, x : x + R, y : y + R, z : z + R] += sample
            count[:, :, x : x + R, y : y + R, z : z + R] += 1

        count = count.clamp(min=1)
        accum = accum / count

        # Read back each chunk's region from the averaged global buffer
        result = []
        for offset in offsets:
            x, y, z = (offset - mins).tolist()
            result.append(accum[:, :, x : x + R, y : y + R, z : z + R].clone())

        return result

    def _aggregate_overlaps_sparse(
        self,
        samples: list[SparseTensor],
        relative_transl: list[torch.Tensor],
        R: int,
    ) -> list[SparseTensor]:
        """Average sparse-tensor features at voxels shared by multiple chunks.

        Each chunk's coords are shifted into a common global frame by
        ``offset_i = round(relative_transl_i * R)`` (R = per-axis voxel resolution
        at the current scale), overlapping voxels are identified, features are
        averaged across all chunks that cover them, and the result is scattered
        back per chunk. Feature dimension is arbitrary.
        """
        device = samples[0].feats.device
        dtype = samples[0].feats.dtype

        offsets = [((t.to(device) * R).round().long()) for t in relative_transl]

        mins = torch.stack(offsets).min(dim=0).values
        maxs = torch.stack(offsets).max(dim=0).values + R
        GX, GY, GZ = (maxs - mins).tolist()

        all_global_coords = []
        all_feats = []
        for sample, offset in zip(samples, offsets):
            assert sample.coords.unique(dim=0).shape == sample.coords.shape, "Duplicate coords within a chunk"
            assert sample.coords[:, 0].max() == 0, f"expected batch index 0, got {sample.coords[:, 0].max()}"
            xyz = sample.coords[:, 1:]
            all_global_coords.append(xyz + (offset - mins).unsqueeze(0))
            all_feats.append(sample.feats)

        all_global_coords = torch.cat(all_global_coords, dim=0)
        all_feats = torch.cat(all_feats, dim=0)
        D = all_feats.shape[1]

        def _encode(c: torch.Tensor) -> torch.Tensor:
            return c[:, 0].long() * (GY * GZ) + c[:, 1].long() * GZ + c[:, 2].long()

        all_keys = _encode(all_global_coords)
        unique_keys, inv = torch.unique(all_keys, return_inverse=True)
        K_u = unique_keys.shape[0]

        acc = torch.zeros(K_u, D, device=device, dtype=dtype)
        cnt = torch.zeros(K_u, 1, device=device, dtype=dtype)
        acc.scatter_add_(0, inv.unsqueeze(1).expand(-1, D), all_feats)
        cnt.scatter_add_(0, inv.unsqueeze(1), torch.ones(len(all_feats), 1, device=device, dtype=dtype))

        avg = acc / cnt

        result = []
        for sample, offset in zip(samples, offsets):
            xyz = sample.coords[:, 1:] + (offset - mins).unsqueeze(0)
            chunk_keys = _encode(xyz)
            idx = torch.searchsorted(unique_keys, chunk_keys)
            result.append(SparseTensor(feats=avg[idx], coords=sample.coords))

        return result

    def _aggregate_overlaps_boundary_sensitive_sparse(
        self,
        samples: list[SparseTensor],
        relative_transl: list[torch.Tensor],
        R: int,
        boundary_width: int = 2,
    ) -> list[SparseTensor]:
        """
        Like _aggregate_overlaps_sparse, but boundary voxels (x or y face in local coords)
        do NOT contribute to the global mean — however every voxel (boundary included)
        RECEIVES the averaged value where at least one interior chunk contributed.

          - Boundary voxels do NOT add to acc / cnt.
          - All voxels read back the averaged value.
          - Fallback: voxels with zero interior contributors keep their original
            per-chunk prediction.

        ``boundary_width`` controls how many outermost rows per x/y face are
        flagged as boundary.
        """
        device = samples[0].feats.device
        dtype = samples[0].feats.dtype
        assert 1 <= boundary_width <= R // 2, f"boundary_width={boundary_width} out of range for R={R}"
        bw = boundary_width

        offsets = [((t.to(device) * R).round().long()) for t in relative_transl]
        mins = torch.stack(offsets).min(dim=0).values
        maxs = torch.stack(offsets).max(dim=0).values + R
        GX, GY, GZ = (maxs - mins).tolist()

        # ── Step 1: shift to global frame, pool coords + feats + interior masks ──
        all_global_coords = []
        all_feats = []
        all_interior = []  # per-voxel bool [K_i], accumulated to [K_total]
        interior_masks = []  # kept per-chunk for the read-back step

        for sample, offset in zip(samples, offsets):
            assert sample.coords.unique(dim=0).shape == sample.coords.shape, "Duplicate coords within a chunk"
            assert sample.coords[:, 0].max() == 0, f"expected batch index 0, got {sample.coords[:, 0].max()}"

            xyz = sample.coords[:, 1:]  # [K, 3] local
            x_loc, y_loc = xyz[:, 0], xyz[:, 1]
            # Boundary = outermost `bw` rows per face.
            interior = ~((x_loc < bw) | (x_loc >= R - bw) | (y_loc < bw) | (y_loc >= R - bw))  # [K] bool

            all_global_coords.append(xyz + (offset - mins).unsqueeze(0))
            all_feats.append(sample.feats)
            all_interior.append(interior)
            interior_masks.append(interior)

        all_global_coords = torch.cat(all_global_coords, dim=0)  # [K_total, 3]
        all_feats = torch.cat(all_feats, dim=0)  # [K_total, D]
        all_interior = torch.cat(all_interior, dim=0)  # [K_total] bool
        D = all_feats.shape[1]

        def _encode(c: torch.Tensor) -> torch.Tensor:
            return c[:, 0].long() * (GY * GZ) + c[:, 1].long() * GZ + c[:, 2].long()

        all_keys = _encode(all_global_coords)

        # ── Step 2: scatter-add, interior voxels only ─────────────────────────
        unique_keys, inv = torch.unique(all_keys, return_inverse=True)
        K_u = unique_keys.shape[0]

        acc = torch.zeros(K_u, D, device=device, dtype=dtype)
        cnt = torch.zeros(K_u, 1, device=device, dtype=dtype)

        # Zero out boundary feats before accumulating; only count interior voxels
        interior_feats = all_feats.masked_fill(~all_interior.unsqueeze(1), 0.0)
        interior_count = all_interior.to(dtype=dtype).unsqueeze(1)  # [K_total, 1]

        acc.scatter_add_(0, inv.unsqueeze(1).expand(-1, D), interior_feats)
        cnt.scatter_add_(0, inv.unsqueeze(1), interior_count)

        has_interior = cnt > 0  # [K_u, 1] bool
        cnt = cnt.clamp(min=1)
        avg = acc / cnt  # [K_u, D]

        # ── Step 3: read back ──────────────────────────────────────────────────
        # Every voxel takes the averaged global value; voxels with zero interior
        # contributors (scene-edge boundary, no interior neighbour) fall back to
        # the chunk's own prediction.
        result = []
        for sample, offset in zip(samples, offsets):
            global_xyz = sample.coords[:, 1:] + (offset - mins).unsqueeze(0)
            idx = torch.searchsorted(unique_keys, _encode(global_xyz))
            averaged = avg[idx]  # [K, D]
            has_int_local = has_interior[idx]  # [K, 1]
            updated = torch.where(has_int_local, averaged, sample.feats)
            result.append(SparseTensor(feats=updated, coords=sample.coords))

        return result

    # ------------------------------------------------------------------
    # Noise initialisation — one draw per unique global voxel
    # ------------------------------------------------------------------

    def _initialize_noise(
        self,
        noise: list[Any],
        relative_transl: list[torch.Tensor],
        resolution: int,
    ) -> list[Any]:
        """
        Ensure overlapping voxels start from identical noise across chunks.

        Averaging independent draws would collapse variance (N(0,σ²/n) instead of N(0,σ²)),
        pushing the starting point out of the training distribution.  Instead, we draw noise
        once per *unique global voxel* and slice back to per-chunk — so every chunk that
        covers a given world-space voxel gets exactly the same sample for it.

        The input `noise` is used only for shape / coords / device / dtype; its values are
        discarded and replaced by the coordinated draw.
        """
        if isinstance(noise[0], torch.Tensor):
            return self._initialize_noise_dense(noise, relative_transl, resolution)
        else:
            return self._initialize_noise_sparse(noise, relative_transl, resolution)

    def _initialize_noise_dense(
        self,
        noise: list[torch.Tensor],
        relative_transl: list[torch.Tensor],
        R: int,
    ) -> list[torch.Tensor]:
        B, C = noise[0].shape[:2]
        device = noise[0].device
        dtype = noise[0].dtype

        offsets = [((t * R).round().long()) for t in relative_transl]
        mins = torch.stack(offsets).min(dim=0).values
        maxs = torch.stack(offsets).max(dim=0).values + R
        GX, GY, GZ = (maxs - mins).tolist()

        # Single draw over the full scene bounding box
        global_noise = torch.randn(B, C, GX, GY, GZ, device=device, dtype=dtype)

        # Slice each chunk's region — identical to _aggregate_overlaps_dense read-back
        result = []
        for offset in offsets:
            x, y, z = (offset - mins).tolist()
            result.append(global_noise[:, :, x : x + R, y : y + R, z : z + R].clone())
        return result

    def _initialize_noise_sparse(
        self,
        noise: list[SparseTensor],
        relative_transl: list[torch.Tensor],
        R: int,
    ) -> list[SparseTensor]:
        device = noise[0].feats.device
        dtype = noise[0].feats.dtype
        D = noise[0].feats.shape[1]

        offsets = [((t.to(device) * R).round().long()) for t in relative_transl]
        mins = torch.stack(offsets).min(dim=0).values
        maxs = torch.stack(offsets).max(dim=0).values + R
        GX, GY, GZ = (maxs - mins).tolist()

        # Pool all global coords (same as _aggregate_overlaps_sparse step 1)
        all_global_xyz = []
        for sample, offset in zip(noise, offsets):
            xyz = sample.coords[:, 1:]
            all_global_xyz.append(xyz + (offset - mins).unsqueeze(0))
        all_global_xyz = torch.cat(all_global_xyz, dim=0)  # [K_total, 3]

        def _encode(c: torch.Tensor) -> torch.Tensor:
            return c[:, 0].long() * (GY * GZ) + c[:, 1].long() * GZ + c[:, 2].long()

        all_keys = _encode(all_global_xyz)
        unique_keys = torch.unique(all_keys)

        # One draw per unique global voxel — N(0,1), correct distribution
        global_noise = torch.randn(len(unique_keys), D, device=device, dtype=dtype)

        # Look up each chunk's voxels — identical to _aggregate_overlaps_sparse read-back
        result = []
        for sample, offset in zip(noise, offsets):
            xyz = sample.coords[:, 1:] + (offset - mins).unsqueeze(0)
            chunk_keys = _encode(xyz)
            idx = torch.searchsorted(unique_keys, chunk_keys)
            result.append(SparseTensor(feats=global_noise[idx], coords=sample.coords))
        return result

    # ------------------------------------------------------------------
    # Main sampling loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        model,
        noise: list[Any],
        cond: list[dict],
        neg_cond: list[dict],
        relative_transl: list[torch.Tensor],
        steps: int = 50,
        rescale_t: float = 1.0,
        guidance_strength: float = 3.0,
        guidance_interval: Tuple[float, float] = (0.0, 1.0),
        guidance_rescale: float = 0.0,
        verbose: bool = True,
        tqdm_desc: str = "Sampling",
        chunk_kwargs: Optional[list[dict]] = None,
        boundary_sensitive: bool = False,
        boundary_width: int = 2,
    ):
        num_chunks = len(noise)
        chunks = self._initialize_noise(noise, relative_transl, model.resolution)

        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_seq = t_seq.tolist()
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))

        if boundary_sensitive:

            def aggregate(chunks, relative_transl, resolution):
                return self._aggregate_overlaps_boundary_sensitive(
                    chunks,
                    relative_transl,
                    resolution,
                    boundary_width=boundary_width,
                )

        else:
            aggregate = self._aggregate_overlaps

        for t, t_prev in tqdm.tqdm(t_pairs, desc=tqdm_desc, disable=not verbose):
            for i in range(num_chunks):
                extra = chunk_kwargs[i] if chunk_kwargs is not None else {}
                v = self.sampler._inference_model(
                    model,
                    chunks[i],
                    t,
                    cond=cond[i],
                    neg_cond=neg_cond[i],
                    guidance_strength=guidance_strength,
                    guidance_interval=guidance_interval,
                    guidance_rescale=guidance_rescale,
                    **extra,
                )
                chunks[i] = chunks[i] - (t - t_prev) * v
            chunks = aggregate(chunks, relative_transl, model.resolution)

        return chunks
