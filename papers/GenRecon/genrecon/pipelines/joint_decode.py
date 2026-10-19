"""Scene-wide joint decoding of shape/texture SLats.

All chunk SLats are merged onto a single global-frame ``SparseTensor``, the
decoder runs once (or in spatial groups for very large scenes), and a single
scene mesh is extracted via one joint call to ``flexible_dual_grid_to_mesh``.
The texture decoder output stays on the joint lattice so the whole scene
becomes one ``MeshWithVoxel`` downstream.

Joint frame convention:
  * Origin at ``chunk_centers[0]`` (the chunker's reference chunk center).
  * Scale: ``1 / chunk_size`` (same as chunk-local frames), so ``m_c2o[0]``
    from the chunker is the joint→world transform.
  * Scene aabb in this frame: per axis ``[min_rel_t - 0.5, max_rel_t + 0.5]``,
    where ``rel_t`` is the chunker's ``relative_translation`` (unit =
    ``chunk_size``).

Chunked decode (1024 pipeline only):
  * For ``len(slats) > max_chunks_per_group``, chunks are spatially grouped
    via recursive longest-axis bisection until each leaf has
    ``≤ max_chunks_per_group`` chunks.
  * Each group's joint slat is inflated by ``overlap_r_in`` voxels (R_in
    scale) so the decoder's receptive field is intact at owned-region
    boundaries; outputs are filtered back to the owned region and concatenated.
  * The texture decoder reuses the same group structure and consumes each
    group's own subdivision tensors (no cross-group merging of subs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from o_voxel.convert import flexible_dual_grid_to_mesh

from ..models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder
from ..modules.sparse import SparseTensor
from ..representations import Mesh


@dataclass
class JointDecodeContext:
    """Per-group artifacts needed by the texture decoder.

    For the single-pass path, ``inflated_aabbs is None`` and ``subs_inflated``
    holds the single decode's subs as ``[subs_list]`` (one element). The tex
    decoder special-cases this to run a single pass on the full joint slat.
    """

    inflated_aabbs: Optional[List[torch.Tensor]]  # per group, [2, 3] at R_in scale
    owned_aabbs: Optional[List[torch.Tensor]]  # per group, [2, 3] at R_in scale
    subs_inflated: List[List[SparseTensor]]  # per group, per stage (raw, no owned filter)
    R_in: int
    upscale: int


def _merge_chunk_slats(
    slats: List[SparseTensor],
    relative_transl: List[torch.Tensor],
    R_in: int,
) -> Tuple[SparseTensor, Dict]:
    """Merge per-chunk SLats into one global-frame batch-1 ``SparseTensor``.

    Each chunk's coords are shifted by ``(offset_k - mins)`` so all chunks
    share a single (0, 0, 0)-origin lattice. Overlap voxels are deduplicated
    and features averaged (MultiDiffusion already made them near-equal; the
    mean absorbs any remaining FP drift).

    Returns ``(joint_slat, meta)`` with ``meta = {"mins", "maxs", "R_in"}``
    at the coarse input scale.
    """
    device = slats[0].feats.device
    dtype = slats[0].feats.dtype

    offsets = [((t.to(device) * R_in).round().long()) for t in relative_transl]
    mins = torch.stack(offsets).min(dim=0).values
    maxs = torch.stack(offsets).max(dim=0).values + R_in
    GY, GZ = (maxs - mins)[1].item(), (maxs - mins)[2].item()

    global_coords = []
    global_feats = []
    for slat, offset in zip(slats, offsets):
        assert slat.coords[:, 0].max() == 0, "expected batch index 0 in each chunk"
        xyz = slat.coords[:, 1:] + (offset - mins).unsqueeze(0)
        global_coords.append(xyz)
        global_feats.append(slat.feats)
    global_coords = torch.cat(global_coords, dim=0)
    global_feats = torch.cat(global_feats, dim=0)
    D = global_feats.shape[1]

    keys = global_coords[:, 0].long() * (GY * GZ) + global_coords[:, 1].long() * GZ + global_coords[:, 2].long()
    unique_keys, inv = torch.unique(keys, return_inverse=True)
    K_u = unique_keys.shape[0]

    acc = torch.zeros(K_u, D, device=device, dtype=dtype)
    cnt = torch.zeros(K_u, 1, device=device, dtype=dtype)
    acc.scatter_add_(0, inv.unsqueeze(1).expand(-1, D), global_feats)
    cnt.scatter_add_(0, inv.unsqueeze(1), torch.ones(len(global_feats), 1, device=device, dtype=dtype))
    avg = acc / cnt

    ux = (unique_keys // (GY * GZ)).int()
    uy = ((unique_keys // GZ) % GY).int()
    uz = (unique_keys % GZ).int()
    batch_col = torch.zeros(K_u, 1, dtype=torch.int32, device=device)
    unique_coords = torch.cat([batch_col, torch.stack([ux, uy, uz], dim=1)], dim=1)

    joint_slat = SparseTensor(feats=avg, coords=unique_coords)
    meta = {"mins": mins, "maxs": maxs, "R_in": R_in}
    return joint_slat, meta


def _make_groups(
    relative_transl: List[torch.Tensor],
    R_in: int,
    max_chunks_per_group: int,
    joint_slat: Optional[SparseTensor] = None,
    overlap_r_in: int = 16,
    max_inflated_voxels: Optional[int] = None,
) -> Tuple[List[List[int]], List[int]]:
    """Recursive longest-axis median bisection over chunk centers.

    A leaf is accepted only if BOTH:
      - ``len(leaf) <= max_chunks_per_group``, AND
      - the inflated AABB of the leaf contains ``<= max_inflated_voxels``
        voxels in ``joint_slat`` (when both are provided).

    The inflated count is the *actual* number of voxels the per-group decoder
    forward will consume (after merge dedup, including overlap from
    neighboring chunks) — it is the direct memory governor at res=1024, not
    the pre-merge owned voxel sum. Single-chunk leaves are returned even if
    they exceed the cap (cannot bisect further).

    Returns ``(groups, group_inflated_counts)``. The counts are zero when no
    measurement was performed (joint_slat or max_inflated_voxels missing).
    """
    centers = torch.stack([t.float() * R_in + R_in / 2 for t in relative_transl])

    do_measure = joint_slat is not None and max_inflated_voxels is not None
    if do_measure:
        device = joint_slat.feats.device
        offsets = torch.stack([(t.to(device) * R_in).round().long() for t in relative_transl])
        mins_global = offsets.min(dim=0).values
        offsets_minus_mins = offsets - mins_global  # [K, 3]
        coords_xyz = joint_slat.coords[:, 1:]

        def inflated_count(indices: List[int]) -> int:
            chunk_origins = offsets_minus_mins[torch.tensor(indices, device=device)]
            owned_min = chunk_origins.min(dim=0).values
            owned_max = chunk_origins.max(dim=0).values + R_in
            inf_min = (owned_min - overlap_r_in).to(coords_xyz.dtype)
            inf_max = (owned_max + overlap_r_in).to(coords_xyz.dtype)
            mask = ((coords_xyz >= inf_min) & (coords_xyz < inf_max)).all(dim=1)
            return int(mask.sum().item())

    else:

        def inflated_count(indices: List[int]) -> int:  # pragma: no cover
            return 0

    def is_leaf(indices: List[int]) -> bool:
        if len(indices) <= 1:
            return True  # cannot bisect further
        if len(indices) > max_chunks_per_group:
            return False
        if do_measure and inflated_count(indices) > max_inflated_voxels:
            return False
        return True

    def bisect(indices: List[int]) -> List[List[int]]:
        if is_leaf(indices):
            return [indices]
        sub = centers[torch.tensor(indices)]
        ranges = sub.max(dim=0).values - sub.min(dim=0).values
        axis = int(ranges.argmax().item())
        order = torch.argsort(sub[:, axis]).tolist()
        sorted_indices = [indices[i] for i in order]
        mid = len(sorted_indices) // 2
        return bisect(sorted_indices[:mid]) + bisect(sorted_indices[mid:])

    groups = bisect(list(range(len(relative_transl))))
    counts = [inflated_count(g) for g in groups] if do_measure else [0] * len(groups)
    return groups, counts


def _filter_sparse_to_aabb(
    st: SparseTensor,
    aabb_min: torch.Tensor,
    aabb_max: torch.Tensor,
) -> SparseTensor:
    """Return a new SparseTensor with only voxels whose xyz coord lies in
    ``[aabb_min, aabb_max)``. Coord frame is preserved (no shift)."""
    coords = st.coords
    aabb_min = aabb_min.to(device=coords.device, dtype=coords.dtype)
    aabb_max = aabb_max.to(device=coords.device, dtype=coords.dtype)
    xyz = coords[:, 1:]
    mask = ((xyz >= aabb_min) & (xyz < aabb_max)).all(dim=1)
    return SparseTensor(
        feats=st.feats[mask].contiguous(),
        coords=coords[mask].contiguous(),
    )


def _group_owned_aabb(
    group_chunk_idx: List[int],
    offsets_minus_mins: List[torch.Tensor],
    R_in: int,
) -> torch.Tensor:
    """Owned AABB ``[2, 3]`` at R_in scale in joint-frame integer coords."""
    origins = torch.stack([offsets_minus_mins[i] for i in group_chunk_idx])
    return torch.stack([origins.min(dim=0).values, origins.max(dim=0).values + R_in])


def _concat_sparse(parts: List[SparseTensor]) -> SparseTensor:
    """Concat along the voxel dimension. Assumes coord frames already match."""
    return SparseTensor(
        feats=torch.cat([p.feats for p in parts], dim=0),
        coords=torch.cat([p.coords for p in parts], dim=0),
    )


def _sparse_cpu_clean(st: SparseTensor) -> SparseTensor:
    """CPU-stash a SparseTensor without inheriting backend caches.

    ``SparseTensor.cpu()`` uses ``.replace()`` which shares ``_caches`` (torchsparse)
    and ``_spatial_cache`` (both backends) with the source — and those caches
    typically hold GPU tensors (subdivision indices, conv kernel maps). After
    ``.cpu()`` the new tensor's feats/coords are on CPU but the caches still
    pin GPU memory, so ``del original`` doesn't actually release it. Going
    through the fresh constructor yields empty caches.
    """
    return SparseTensor(feats=st.feats.cpu(), coords=st.coords.cpu())


def _sparse_to_device_clean(st: SparseTensor, device: torch.device) -> SparseTensor:
    """Restore a CPU-stashed SparseTensor to ``device`` via fresh constructor."""
    return SparseTensor(
        feats=st.feats.to(device, non_blocking=True),
        coords=st.coords.to(device, non_blocking=True),
    )


@torch.no_grad()
def joint_decode_shape(
    decoder,
    slats: List[SparseTensor],
    relative_transl: List[torch.Tensor],
    *,
    max_chunks_per_group: Optional[int] = None,
    max_inflated_voxels: Optional[int] = None,
    overlap_r_in: int = 16,
) -> Tuple[Mesh, torch.Tensor, torch.Tensor, JointDecodeContext, Dict]:
    """Joint shape decode + joint mesh extraction.

    With ``max_chunks_per_group is None`` (or ``len(slats) <=`` it AND every
    chunk is under the voxel cap) the decoder runs once on the full merged
    joint slat (original behavior). Otherwise the chunks are spatially grouped
    and decoded group-by-group with overlap.

    Both caps act as governors on per-group memory. The voxel cap (sum of
    pre-merge ``slats[i].coords.shape[0]`` over a group) is the real memory
    constraint at res=1024 — chunk-count alone misses density variation.

    Returns ``(scene_mesh, aabb, grid_size_fine, ctx, meta)``. ``ctx`` is
    consumed by :func:`joint_decode_tex`.
    """
    num_up = len(decoder.num_blocks) - 1
    R_in = decoder.resolution // (2**num_up)
    upscale = decoder.resolution // R_in

    joint_slat, meta = _merge_chunk_slats(slats, relative_transl, R_in)
    device = joint_slat.feats.device

    chunk_cap_active = max_chunks_per_group is not None and len(slats) > max_chunks_per_group
    voxel_cap_active = max_inflated_voxels is not None and int(joint_slat.coords.shape[0]) > max_inflated_voxels
    use_chunked = chunk_cap_active or voxel_cap_active

    # Scene aabb in chunk_size units (joint frame). Derivation: chunk 0's
    # voxel 0 must map to chunk-0-local position -0.5 + 0.5/R_fine, which
    # forces aabb[0] = mins_coarse/R_in - 0.5 per axis.
    aabb = torch.stack(
        [
            meta["mins"].float() / R_in - 0.5,
            meta["maxs"].float() / R_in - 0.5,
        ],
        dim=0,
    ).to(device)
    grid_size_fine = ((meta["maxs"] - meta["mins"]) * upscale).int()
    vm = decoder.voxel_margin

    if not use_chunked:
        # Single-pass: original behavior. Decode → channel-extract → mesh-extract once.
        h_joint, joint_subs = SparseUnetVaeDecoder.forward(decoder, joint_slat, return_subs=True)
        vertices = h_joint.replace((1 + 2 * vm) * F.sigmoid(h_joint.feats[..., 0:3]) - vm)
        intersected = h_joint.replace(h_joint.feats[..., 3:6] > 0)
        quad_lerp = h_joint.replace(F.softplus(h_joint.feats[..., 6:7]))
        meshes: List[Mesh] = [
            Mesh(
                *flexible_dual_grid_to_mesh(
                    v.coords[:, 1:],
                    v.feats,
                    i.feats,
                    q.feats,
                    aabb=aabb,
                    grid_size=grid_size_fine,
                    train=False,
                )
            )
            for v, i, q in zip(vertices, intersected, quad_lerp)
        ]
        assert len(meshes) == 1, f"expected batch size 1, got {len(meshes)}"
        scene_mesh = meshes[0]
        ctx = JointDecodeContext(
            inflated_aabbs=None,
            owned_aabbs=None,
            subs_inflated=[joint_subs],
            R_in=R_in,
            upscale=upscale,
        )
        return scene_mesh, aabb, grid_size_fine, ctx, meta

    # ── Chunked: spatially group chunks, decode + mesh-extract per group ──
    # We never form a global h_joint. ``flexible_dual_grid_to_mesh`` allocates
    # an O(N * 36) intermediate (edge_neighbor_voxel) that exceeds 80 GB at
    # full-scene N. Per-group extraction caps that allocation at per-group
    # voxel count. Each group mesh-extracts on its INFLATED voxel set so
    # boundary adjacency is intact, then keeps only faces whose centroid is
    # in the OWNED AABB (faces are claimed by exactly one group; owned AABBs
    # are spatially disjoint by construction). Boundary mesh vertices appear
    # in multiple groups' inflated extracts; we dedupe by voxel coord at the
    # end so the final mesh shares vertices across group boundaries.
    offsets = [(t.to(device) * R_in).round().long() for t in relative_transl]
    mins = torch.stack(offsets).min(dim=0).values
    offsets_minus_mins = [(o - mins) for o in offsets]

    chunk_cap = max_chunks_per_group if max_chunks_per_group is not None else len(slats)
    groups, group_inflated_counts = _make_groups(
        relative_transl,
        R_in,
        chunk_cap,
        joint_slat=joint_slat,
        overlap_r_in=overlap_r_in,
        max_inflated_voxels=max_inflated_voxels,
    )
    print(
        f"[joint_decode_shape] chunked: {len(slats)} chunks → {len(groups)} groups "
        f"(sizes: {[len(g) for g in groups]}, inflated-voxel counts: {group_inflated_counts})"
    )

    mesh_v_parts: List[torch.Tensor] = []  # CPU; per-group owned vertices [V_g, 3] float
    mesh_f_parts: List[torch.Tensor] = []  # CPU; per-group owned faces [F_g, 3] int (group-local indices)
    voxel_coords_parts: List[torch.Tensor] = []  # CPU; per-group voxel coords [V_g, 3] int — dedup keys
    inflated_aabbs: List[torch.Tensor] = []
    owned_aabbs: List[torch.Tensor] = []
    subs_inflated_per_group: List[List[SparseTensor]] = []

    for g_idx, group in enumerate(groups):
        owned_aabb = _group_owned_aabb(group, offsets_minus_mins, R_in)
        inflated_aabb = torch.stack([owned_aabb[0] - overlap_r_in, owned_aabb[1] + overlap_r_in])

        slat_g = _filter_sparse_to_aabb(joint_slat, inflated_aabb[0], inflated_aabb[1])
        n_in = slat_g.feats.shape[0]
        h_g, subs_g = SparseUnetVaeDecoder.forward(decoder, slat_g, return_subs=True)

        # Channel extraction on the inflated h_g (small enough per-group).
        v_g = h_g.replace((1 + 2 * vm) * F.sigmoid(h_g.feats[..., 0:3]) - vm)
        i_g = h_g.replace(h_g.feats[..., 3:6] > 0)
        q_g = h_g.replace(F.softplus(h_g.feats[..., 6:7]))

        # Mesh-extract on this group's inflated voxel set. coords are joint-frame
        # fine-scale integers; aabb/grid_size are scene-wide so output positions
        # are in the joint world frame and consistent across groups.
        coords_g = v_g.coords[:, 1:]  # [V_g, 3] int
        mesh_v_g, mesh_f_g = flexible_dual_grid_to_mesh(
            coords_g,
            v_g.feats,
            i_g.feats,
            q_g.feats,
            aabb=aabb,
            grid_size=grid_size_fine,
            train=False,
        )

        # Filter faces to those whose centroid lies in owned region (in world).
        # owned_world = owned_aabb_R_in / R_in + aabb[0].
        owned_world_min = (owned_aabb[0].float() / R_in + aabb[0]).to(mesh_v_g.dtype)
        owned_world_max = (owned_aabb[1].float() / R_in + aabb[0]).to(mesh_v_g.dtype)
        face_centroids = mesh_v_g[mesh_f_g.long()].mean(dim=1)  # [F_g, 3]
        in_owned = ((face_centroids >= owned_world_min) & (face_centroids < owned_world_max)).all(dim=1)
        mesh_f_owned = mesh_f_g[in_owned]

        # Compact: keep only vertices referenced by owned faces.
        used_v = torch.unique(mesh_f_owned.flatten().long())
        remap = torch.full((mesh_v_g.shape[0],), -1, dtype=torch.long, device=mesh_v_g.device)
        remap[used_v] = torch.arange(used_v.shape[0], device=mesh_v_g.device)
        mesh_v_compact = mesh_v_g[used_v]
        mesh_f_compact = remap[mesh_f_owned.long()].to(mesh_f_owned.dtype)
        # For dedup later: voxel coords of the compacted vertices.
        voxel_coords_compact = coords_g[used_v].to(torch.int64)

        mesh_v_parts.append(mesh_v_compact.cpu())
        mesh_f_parts.append(mesh_f_compact.cpu())
        voxel_coords_parts.append(voxel_coords_compact.cpu())

        inflated_aabbs.append(inflated_aabb)
        owned_aabbs.append(owned_aabb)
        subs_inflated_per_group.append([_sparse_cpu_clean(s) for s in subs_g])

        print(
            f"[joint_decode_shape] group {g_idx}: chunks={group} in={n_in:,} → "
            f"verts={mesh_v_compact.shape[0]:,}, faces={mesh_f_compact.shape[0]:,}"
        )

        del slat_g, h_g, subs_g, v_g, i_g, q_g, coords_g
        del mesh_v_g, mesh_f_g, mesh_f_owned, used_v, remap
        del mesh_v_compact, mesh_f_compact, voxel_coords_compact, face_centroids, in_owned
        torch.cuda.empty_cache()

    # ── Concat per-group meshes + dedupe boundary vertices ────────────────
    # Vertices coming from different groups but corresponding to the same
    # voxel coord must collapse to one entry, otherwise neighboring faces
    # across group boundaries don't share an edge in the index sense.
    cum_offsets: List[int] = []
    cum = 0
    for v in mesh_v_parts:
        cum_offsets.append(cum)
        cum += int(v.shape[0])

    final_verts = torch.cat([v.to(device) for v in mesh_v_parts], dim=0)
    final_faces = torch.cat(
        [f.to(device) + off for f, off in zip(mesh_f_parts, cum_offsets)],
        dim=0,
    )
    voxel_coords_all = torch.cat([c.to(device) for c in voxel_coords_parts], dim=0)
    del mesh_v_parts, mesh_f_parts, voxel_coords_parts

    # Dedupe by voxel coord. The grid is bounded; pack (x, y, z) into one int64.
    GY = int((meta["maxs"] - meta["mins"])[1].item()) * upscale + 1
    GZ = int((meta["maxs"] - meta["mins"])[2].item()) * upscale + 1
    keys = voxel_coords_all[:, 0] * (GY * GZ) + voxel_coords_all[:, 1] * GZ + voxel_coords_all[:, 2]
    unique_keys, inverse = torch.unique(keys, return_inverse=True)
    # Pick one source vertex per unique key. Sort by inverse so identical
    # keys are adjacent; the first row in each run is one valid representative.
    order = torch.argsort(inverse, stable=True)
    inverse_sorted = inverse[order]
    keep_mask = torch.ones(inverse.shape[0], dtype=torch.bool, device=device)
    keep_mask[1:] = inverse_sorted[1:] != inverse_sorted[:-1]
    representative = order[keep_mask]  # one original index per unique key, in unique-key order
    final_verts_dedup = final_verts[representative]
    final_faces_dedup = inverse[final_faces.long()].to(final_faces.dtype)
    del final_verts, voxel_coords_all, keys, inverse, order, inverse_sorted, keep_mask, representative
    torch.cuda.empty_cache()

    print(
        f"[joint_decode_shape] merged: {final_verts_dedup.shape[0]:,} verts (dedup from "
        f"{cum:,}), {final_faces_dedup.shape[0]:,} faces"
    )

    scene_mesh = Mesh(final_verts_dedup, final_faces_dedup)
    ctx = JointDecodeContext(
        inflated_aabbs=inflated_aabbs,
        owned_aabbs=owned_aabbs,
        subs_inflated=subs_inflated_per_group,
        R_in=R_in,
        upscale=upscale,
    )
    return scene_mesh, aabb, grid_size_fine, ctx, meta


@torch.no_grad()
def joint_decode_tex(
    decoder,
    slats: List[SparseTensor],
    ctx: JointDecodeContext,
    relative_transl: List[torch.Tensor],
) -> SparseTensor:
    """Joint tex decode. Returns a single joint-frame ``SparseTensor`` in
    ``[0, 1]`` covering the whole scene.

    When ``ctx.inflated_aabbs is None`` runs a single decode pass on the full
    merged slat (matches original behavior). Otherwise re-uses the shape
    decoder's group structure: each group's tex decode consumes its own
    subdivision tensors and is filtered to its owned region before concat.
    """
    joint_slat, _ = _merge_chunk_slats(slats, relative_transl, ctx.R_in)

    if ctx.inflated_aabbs is None:
        # Single-pass: original behavior.
        h_joint = decoder(joint_slat, guide_subs=ctx.subs_inflated[0])
        return h_joint * 0.5 + 0.5

    # Chunked path: per-group decode with own subs, merge owned outputs.
    # subs were CPU-stashed in joint_decode_shape; lift back to GPU per group.
    device = joint_slat.feats.device
    h_owned_parts: List[SparseTensor] = []
    for g_idx, (inflated, owned, subs_g_cpu) in enumerate(zip(ctx.inflated_aabbs, ctx.owned_aabbs, ctx.subs_inflated)):
        tex_slat_g = _filter_sparse_to_aabb(joint_slat, inflated[0], inflated[1])
        subs_g = [_sparse_to_device_clean(s, device) for s in subs_g_cpu]
        n_in = tex_slat_g.feats.shape[0]
        h_g = decoder(tex_slat_g, guide_subs=subs_g)
        h_g_owned = _filter_sparse_to_aabb(h_g, owned[0] * ctx.upscale, owned[1] * ctx.upscale)
        # CPU-offload like the shape pass to keep GPU clear between groups.
        h_owned_parts.append(_sparse_cpu_clean(h_g_owned))
        print(f"[joint_decode_tex] group {g_idx}: in={n_in:,} → out_owned={h_g_owned.feats.shape[0]:,}")
        del tex_slat_g, h_g, subs_g, h_g_owned
        torch.cuda.empty_cache()

    h_joint = _concat_sparse([_sparse_to_device_clean(p, device) for p in h_owned_parts])
    return h_joint * 0.5 + 0.5
