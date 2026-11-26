"""GenRecon's reconstruct_scene.py with stage 2 (shape) latent-inpainted where the input views' depth knows the
surface. For scenes made by experiments/unposed_genrecon (convert.py), whose cameras and depth come from Solv3R or
VGGT-Omega.

Stage 2 is inpainted RePaint-style, as experiments/genrecon_inpaint_occupancy (branch feat/stage1-inpainting) does
for stage 1: what the views know is encoded with the stage's VAE encoder, and after every Euler step (and the
MultiDiffusion averaging) the pinned latent voxels are reset to that encoding, noised to the next time with the
chunk's own starting noise. The DiT generates the rest. Stages 1 and 3 are untouched.

    1. Stage 2 is sampled as usual (pass A) and decoded to the shape decoder's fine voxels (FINE_PER_LATENT per
       latent voxel and axis, 512 per chunk side): a dual vertex and three edge-crossing flags each.
    2. The depth maps of the input images are fused into a TSDF (Open3D, TSDF_PER_CHUNK voxels per chunk side) and
       its surface is turned into fine voxels with mesh_to_flexible_dual_grid, as get_slats.py does for the
       training meshes.
    3. A latent voxel is known when MIN_VIEWS images see all of it: every point of a SUBSAMPLE^3 grid in it lies
       inside the image and at most TRUNCATION TSDF voxels behind the depth (so free space counts). The known latent
       voxels are shrunk by one latent voxel, as stage 1's pinning is.
    4. The target is the depth's fine voxels in the known latent voxels and pass A's elsewhere, so the encoder never
       reads holes. It is encoded with TRELLIS.2's shape encoder.
    5. Stage 2 is sampled again from pass A's starting noise (pass B), with the known latent voxels pinned in which
       the target has fine voxels. A known latent voxel the depth sees as empty has no encoding and is left free.

Counts are per image, not per crop: GenRecon cuts every wide image into two overlapping square crops.

Writes inpainting.json (shares of known and pinned latent voxels) and depth_surface.ply (the TSDF surface, in the
frame of mesh.ply) next to GenRecon's outputs. Patches MultiDiffusionOrchestrator.sample and
FullSceneImagesTo3DPipeline.run; GenRecon itself is not modified. Runs in GenRecon's environment, from papers/GenRecon:

    .venv/bin/python ../../experiments/genrecon_latent_inpainting/inject.py --npz NPZ <reconstruct_scene.py arguments>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import o_voxel
import open3d as o3d
import torch
import torch.nn.functional as F
import tqdm
import trimesh
from scipy.ndimage import binary_erosion, maximum_filter, minimum_filter

sys.path.insert(0, str(Path.cwd()))  # papers/GenRecon
import reconstruct_scene  # noqa: E402
from genrecon import models  # noqa: E402
from genrecon.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder  # noqa: E402
from genrecon.modules.sparse import SparseTensor  # noqa: E402
from genrecon.pipelines.full_scene_images_to_3d import FullSceneImagesTo3DPipeline  # noqa: E402
from genrecon.pipelines.joint_decode import _merge_chunk_slats  # noqa: E402
from genrecon.pipelines.samplers.multi_diff_orchestrator import MultiDiffusionOrchestrator  # noqa: E402

SHAPE_ENCODER = "microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16"
FINE_PER_LATENT = 16  # fine voxels per shape latent voxel and axis

MIN_VIEWS = 1  # images that must see all of a latent voxel for it to be known
EDGE_RATIO = 0.1  # depth pixels whose 3x3 depth range exceeds this share of their depth are dropped
TSDF_PER_CHUNK = 128  # TSDF voxels per chunk side
TRUNCATION = 3  # TSDF voxels: the TSDF's truncation, and how far behind the depth a point still counts as seen
SUBSAMPLE = 4  # points per latent voxel and axis that are tested for being seen
NEAR = 0.05  # depth units: points closer to a camera than this are not seen by it

STATE = {}  # the pipeline and its SelectedImages, stored by the patched run


def views(out: Path) -> dict[str, list[int]]:
    """The scene images' crops, grouped by image: name -> indices into SelectedImages' scene tensors."""
    groups = {}
    for i, view in enumerate(json.loads((out / "cameras.json").read_text())["scene"]):
        groups.setdefault(Path(view["img_path"]).name, []).append(i)
    return groups


def sample_loop(orchestrator, model, chunks, cond, neg_cond, relative_transl, t_pairs, guidance, aggregate, verbose,
                desc, pin=None):
    """MultiDiffusionOrchestrator.sample's loop for sparse latents. pin = (encodings, masks, starting noise) resets
    the pinned voxels after every step to the encoding, noised to the next time."""
    sigma_min = orchestrator.sampler.sigma_min
    for t, t_prev in tqdm.tqdm(t_pairs, desc=desc, disable=not verbose):
        for i in range(len(chunks)):
            v = orchestrator.sampler._inference_model(model, chunks[i], t, cond=cond[i], neg_cond=neg_cond[i],
                                                      **guidance)
            chunks[i] = chunks[i] - (t - t_prev) * v
        chunks = aggregate(chunks, relative_transl, model.resolution)
        if pin is not None:
            s = sigma_min + (1 - sigma_min) * t_prev
            chunks = [c.replace(m * ((1 - t_prev) * x0 + s * e.feats) + (1 - m) * c.feats)
                      for c, x0, m, e in zip(chunks, *pin)]
    return chunks


# ── The depth of the input images ────────────────────────────────────────────


def drop_edges(depth: np.ndarray) -> np.ndarray:
    """NaN at depth discontinuities, where predicted depth smears between the two surfaces."""
    return np.where(maximum_filter(depth, 3) - minimum_filter(depth, 3) < EDGE_RATIO * depth, depth, np.nan)


def input_views(npz: Path, out: Path) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per input image: depth (NaN where unknown), intrinsics in pixels at the depth's resolution, and the
    extrinsics from the chunk-0 frame (unit: chunk size) to camera coordinates in the depth's unit (OpenCV)."""
    sel = STATE["sel"]
    prediction = np.load(npz)
    result = []
    for name, crops in views(out).items():
        frame = int(Path(name).stem)  # convert.py names frame i of the .npz "{i:04d}.png"
        depth = drop_edges(prediction["depth"][frame, ..., 0].astype(np.float64))
        K = prediction["intrinsics"][frame].astype(np.float64)
        ext = sel.scene_extrinsics_c0[crops[0]].double().cpu().numpy()  # a crop changes only the intrinsics
        result.append((depth, K, ext))
    return result


def depth_surface(inputs) -> tuple[np.ndarray, np.ndarray]:
    """The surface of the TSDF fused from the input depth, as vertices (chunk-0 frame) and faces.

    The TSDF lives in the chunk-0 frame scaled to the depth's unit, where the cameras are rigid.
    """
    scales = np.array([np.linalg.norm(ext[0, :3]) for _, _, ext in inputs])  # depth units per chunk size
    s = scales.mean()
    STATE["stats"]["depth_units_per_chunk"] = [float(scales.min()), float(scales.max())]
    voxel = s / TSDF_PER_CHUNK
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=TRUNCATION * voxel,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
    for depth, K, ext in inputs:
        height, width = depth.shape
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.zeros((height, width, 3), dtype=np.uint8)),
            o3d.geometry.Image(np.nan_to_num(depth, nan=0.0).astype(np.float32)),  # 0: no depth
            depth_scale=1.0, depth_trunc=float(np.nanmax(depth)) + 1.0, convert_rgb_to_intensity=False)
        # Open3D puts pixel centres at integer coordinates, the intrinsics here at + 0.5
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, K[0, 0], K[1, 1], K[0, 2] - 0.5, K[1, 2] - 0.5)
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = ext[:3, :3] / s
        extrinsic[:3, 3] = ext[:3, 3]
        volume.integrate(rgbd, intrinsic, extrinsic)
    mesh = volume.extract_triangle_mesh()
    return np.asarray(mesh.vertices) / s, np.asarray(mesh.triangles)


@torch.no_grad()
def known_latents(inputs, coords: torch.Tensor, corner: torch.Tensor, R_lat: int) -> torch.Tensor:
    """Whether MIN_VIEWS images see all of each latent voxel (joint coarse coords [N, 3], chunk-0 frame cell
    corner + coords / R_lat): inside the image and at most TRUNCATION TSDF voxels behind its depth."""
    device = coords.device
    j = (torch.arange(SUBSAMPLE, device=device) + 0.5) / SUBSAMPLE
    sub = torch.stack(torch.meshgrid(j, j, j, indexing="ij"), dim=-1).reshape(-1, 3)
    points = corner + (coords[:, None, :].float() + sub[None]) / R_lat  # [N, SUBSAMPLE^3, 3]
    seen = torch.zeros(points.shape[:2], dtype=torch.int32, device=device)
    for depth, K, ext in inputs:
        d = torch.from_numpy(np.nan_to_num(depth, nan=-1.0)).float().to(device)  # -1: no depth
        K, ext = torch.from_numpy(K).float().to(device), torch.from_numpy(ext).float().to(device)
        cam = points @ ext[:3, :3].T + ext[:3, 3]
        z = cam[..., 2]
        safe = torch.where(z > NEAR, z, torch.ones_like(z))
        px = torch.floor(K[0, 0] * cam[..., 0] / safe + K[0, 2])
        py = torch.floor(K[1, 1] * cam[..., 1] / safe + K[1, 2])
        inside = (z > NEAR) & (px >= 0) & (px < d.shape[1]) & (py >= 0) & (py < d.shape[0])
        observed = d[py.long().clamp(0, d.shape[0] - 1), px.long().clamp(0, d.shape[1] - 1)]
        margin = TRUNCATION * ext[0, :3].norm() / TSDF_PER_CHUNK  # TSDF voxels to the depth's unit
        seen += (inside & (observed > 0) & (z <= observed + margin)).int()
    return (seen >= MIN_VIEWS).all(dim=1)


# ── Stage 2 ──────────────────────────────────────────────────────────────────


def quantize(vertices: torch.Tensor) -> torch.Tensor:
    """Dual vertices in the voxel, as get_slats.py stores them for the encoder: clamped to [0, 1], 8 bits."""
    return (vertices.clamp(0, 1) * 255).round() / 255


@torch.no_grad()
def decode_fine(joint: SparseTensor, res: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The shape decoder's fine voxels of a joint shape latent: coords [N, 3], dual vertices [N, 3] in the voxel,
    and edge-crossing flags [N, 3], as joint_decode_shape computes them before extracting the mesh."""
    pipeline = STATE["pipeline"]
    dec = pipeline.models["shape_slat_decoder"]
    dec.set_resolution(res)
    if 2 ** (len(dec.num_blocks) - 1) != FINE_PER_LATENT:
        raise RuntimeError(f"the shape decoder upsamples by {2 ** (len(dec.num_blocks) - 1)}, not {FINE_PER_LATENT}")
    if pipeline.low_vram:
        dec.to(pipeline.device)
    h = SparseUnetVaeDecoder.forward(dec, joint)
    vm = dec.voxel_margin
    vertices = quantize((1 + 2 * vm) * F.sigmoid(h.feats[:, 0:3].float()) - vm)
    intersected = h.feats[:, 3:6] > 0
    coords = h.coords[:, 1:].long()
    if pipeline.low_vram:
        dec.cpu()
    del h
    torch.cuda.empty_cache()
    return coords, vertices, intersected


@torch.no_grad()
def target_fine(vertices: np.ndarray, faces: np.ndarray, aabb: torch.Tensor, grid_size: torch.Tensor, device):
    """Fine voxels of a mesh (chunk-0 frame) on the joint fine grid, as get_slats.py voxelizes training meshes."""
    v = torch.from_numpy(vertices).float()
    f = torch.from_numpy(faces).long()
    lo, hi = aabb[0].cpu(), aabb[1].cpu()
    f = f[((v[f] >= lo) & (v[f] < hi)).all(dim=2).all(dim=1)]  # faces inside the scene's grid
    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        v, f, grid_size=grid_size.cpu().int(), aabb=torch.stack([lo, hi]).float(),
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2)
    voxel_size = (hi - lo) / grid_size.cpu().float()
    local = dual_vertices / voxel_size - voxel_indices  # dual vertices come relative to the aabb's corner
    return (voxel_indices.long().to(device), quantize(local.float()).to(device), intersected.bool().to(device))


def save_surface(vertices: np.ndarray, faces: np.ndarray, out: Path):
    """The TSDF surface in the frame of mesh.ply."""
    chunk0 = np.array(json.loads((out / "chunk_transforms.json").read_text())["chunks"][0]["M_chunk_to_original"])
    trimesh.Trimesh(vertices @ chunk0[:3, :3].T + chunk0[:3, 3], faces, process=False).export(out / "depth_surface.ply")


@torch.no_grad()
def shape_pins(pass_a: list[SparseTensor], relative_transl, R_lat: int, npz: Path, out: Path, trust_depth: bool):
    """Per chunk, the normalized encoding of the target and the mask of pinned latent voxels. With trust_depth, every
    latent voxel the depth's surface passes through is known, without the view count or the shrinking."""
    pipeline = STATE["pipeline"]
    device = pass_a[0].feats.device
    std = torch.tensor(pipeline.shape_slat_normalization["std"])[None].to(device)
    mean = torch.tensor(pipeline.shape_slat_normalization["mean"])[None].to(device)

    # The joint coarse grid, as joint_decode_shape lays it out
    joint, meta = _merge_chunk_slats([s * std + mean for s in pass_a], relative_transl, R_lat)
    dims = (meta["maxs"] - meta["mins"]).long()
    corner = meta["mins"].float() / R_lat - 0.5  # chunk-0 frame
    aabb = torch.stack([corner, meta["maxs"].float() / R_lat - 0.5])
    grid_size = dims * FINE_PER_LATENT
    coarse = joint.coords[:, 1:].long()

    def key(c):
        return (c[:, 0] * dims[1] + c[:, 1]) * dims[2] + c[:, 2]

    inputs = input_views(npz, out)
    surface_vertices, surface_faces = depth_surface(inputs)
    save_surface(surface_vertices, surface_faces, out)
    d_coords, d_vertices, d_intersected = target_fine(surface_vertices, surface_faces, aabb, grid_size, device)
    if trust_depth:
        # Known: every latent voxel of the sparse structure that the depth's surface passes through
        known = torch.isin(key(coarse), key(d_coords // FINE_PER_LATENT))
        known_keys = key(coarse[known]).sort().values
    else:
        # Known latent voxels, shrunk by one; cells outside the sparse structure are empty in both and count as known
        known = known_latents(inputs, coarse, corner, R_lat)
        grid = np.ones(tuple(dims.tolist()), dtype=bool)
        c = coarse.cpu().numpy()
        grid[tuple(c.T)] = known.cpu().numpy()
        grid = binary_erosion(grid, np.ones((3, 3, 3), dtype=bool), border_value=1)
        known_keys = key(coarse[torch.from_numpy(grid[tuple(c.T)]).to(device)]).sort().values

    def is_known(coarse_coords):
        k = key(coarse_coords)
        if not len(known_keys):
            return torch.zeros_like(k, dtype=torch.bool)
        return known_keys[torch.searchsorted(known_keys, k).clamp(max=len(known_keys) - 1)] == k

    # Target: the depth's fine voxels where known, pass A's elsewhere
    a_coords, a_vertices, a_intersected = decode_fine(joint, R_lat * FINE_PER_LATENT)
    del joint
    keep_a, keep_d = ~is_known(a_coords // FINE_PER_LATENT), is_known(d_coords // FINE_PER_LATENT)
    fine = torch.cat([a_coords[keep_a], d_coords[keep_d]])
    vertices = torch.cat([a_vertices[keep_a], d_vertices[keep_d]])
    intersected = torch.cat([a_intersected[keep_a], d_intersected[keep_d]])
    print(f"[inpaint] stage 2: {len(a_coords)} fine voxels from pass A, {len(d_coords)} from the depth; "
          f"target {int(keep_a.sum())} + {int(keep_d.sum())}", flush=True)
    del a_coords, a_vertices, a_intersected, d_coords, d_vertices, d_intersected

    order = torch.argsort(o_voxel.serialize.encode_seq(fine.int().cpu())).to(device)  # as get_slats.py
    coords = torch.cat([torch.zeros(len(fine), 1, dtype=torch.int32, device=device), fine.int()], dim=1)[order]
    encoder = models.from_pretrained(SHAPE_ENCODER).to(device).eval()
    z = encoder(SparseTensor(vertices[order], coords), SparseTensor(intersected[order], coords))
    del encoder, vertices, intersected, fine, coords
    torch.cuda.empty_cache()
    z_norm = (z.feats.float() - mean) / std
    z_keys, z_order = key(z.coords[:, 1:].long()).sort()

    # Per chunk: pinned where the latent voxel is known and the target has an encoding there
    offsets = [(t.to(device) * R_lat).round().long() for t in relative_transl]
    mins = torch.stack(offsets).min(dim=0).values
    x0, masks, total, known_total = [], [], 0, 0
    for chunk, offset in zip(pass_a, offsets):
        k = key(chunk.coords[:, 1:].long() + (offset - mins))
        j = torch.searchsorted(z_keys, k).clamp(max=len(z_keys) - 1)
        found = z_keys[j] == k
        known_chunk = is_known(chunk.coords[:, 1:].long() + (offset - mins))
        mask = (found & known_chunk).to(chunk.feats.dtype)[:, None]
        x0.append(torch.where(found[:, None], z_norm[z_order[j]], torch.zeros_like(z_norm[z_order[j]]))
                  .to(chunk.feats.dtype))
        masks.append(mask)
        total += mask.sum().item()
        known_total += known_chunk.sum().item()
    voxels = sum(len(c.coords) for c in pass_a)
    STATE["stats"].update(shape_known_latents=known.float().mean().item(),
                          shape_known_shrunk_latents=known_total / voxels, shape_pinned_latents=total / voxels)
    print(f"[inpaint] stage 2: {known.float().mean():.1%} of the latent voxels known, {known_total / voxels:.1%} "
          f"after shrinking, {total / voxels:.1%} pinned", flush=True)
    return x0, masks


# ── Patches ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--npz", type=Path, required=True, help="Depth and cameras of the input images.")
    parser.add_argument("--trust-depth", action="store_true",
                        help="Pin every latent voxel the depth's surface passes through (no view count, no shrinking).")
    args, rest = parser.parse_known_args()
    out = Path(reconstruct_scene.build_parser().parse_args(rest).output_path)
    STATE["stats"] = {"trust_depth": args.trust_depth}

    original_run = FullSceneImagesTo3DPipeline.run
    original_sample = MultiDiffusionOrchestrator.sample

    def run(self, sel, *a, **k):
        STATE.update(pipeline=self, sel=sel)
        return original_run(self, sel, *a, **k)

    @torch.no_grad()
    def sample(self, model, noise, cond, neg_cond, relative_transl, steps=50, rescale_t=1.0, guidance_strength=3.0,
               guidance_interval=(0.0, 1.0), guidance_rescale=0.0, verbose=True, tqdm_desc="Sampling",
               chunk_kwargs=None, boundary_sensitive=False, boundary_width=2):
        dense = isinstance(noise[0], torch.Tensor)
        tex = chunk_kwargs is not None and "concat_cond" in chunk_kwargs[0]
        if dense or tex:  # stages 1 and 3
            return original_sample(self, model, noise, cond, neg_cond, relative_transl, steps=steps,
                                   rescale_t=rescale_t, guidance_strength=guidance_strength,
                                   guidance_interval=guidance_interval, guidance_rescale=guidance_rescale,
                                   verbose=verbose, tqdm_desc=tqdm_desc, chunk_kwargs=chunk_kwargs,
                                   boundary_sensitive=boundary_sensitive, boundary_width=boundary_width)
        guidance = {"guidance_strength": guidance_strength, "guidance_interval": guidance_interval,
                    "guidance_rescale": guidance_rescale}
        if boundary_sensitive:
            def aggregate(c, t, r):
                return self._aggregate_overlaps_boundary_sensitive(c, t, r, boundary_width=boundary_width)
        else:
            aggregate = self._aggregate_overlaps
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = (rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)).tolist()
        t_pairs = list(zip(t_seq[:-1], t_seq[1:]))
        R_lat = model.resolution
        start = self._initialize_noise(noise, relative_transl, R_lat)

        def loop(chunks, desc, pin=None):
            return sample_loop(self, model, chunks, cond, neg_cond, relative_transl, t_pairs, guidance, aggregate,
                               verbose, desc, pin)

        pass_a = loop([c.replace(c.feats.clone()) for c in start], tqdm_desc + " (pass A)")
        with torch.autocast("cuda", enabled=False):
            x0, masks = shape_pins(pass_a, relative_transl, R_lat, args.npz, out, args.trust_depth)
        del pass_a
        torch.cuda.empty_cache()
        result = loop([c.replace(c.feats.clone()) for c in start], tqdm_desc + " (pass B, inpainted)",
                      (x0, masks, start))
        (out / "inpainting.json").write_text(json.dumps(STATE["stats"], indent=2))
        return result

    FullSceneImagesTo3DPipeline.run = run
    MultiDiffusionOrchestrator.sample = sample
    sys.argv = [sys.argv[0], *rest]
    reconstruct_scene.main()


if __name__ == "__main__":
    main()
