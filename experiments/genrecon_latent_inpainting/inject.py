"""GenRecon's reconstruct_scene.py with stage 1 (sparse structure) and stage 3 (texture) latent-inpainted around what
the input views observed. For scenes made by experiments/unposed_genrecon (convert.py), whose cameras and depth come
from Solv3R or VGGT-Omega.

Both stages are inpainted RePaint-style, as experiments/genrecon_inpaint_occupancy (branch feat/stage1-inpainting)
does for stage 1: what the views know is encoded with the stage's VAE encoder, and after every Euler step (and the
MultiDiffusion averaging) the pinned latent voxels are reset to that encoding, noised to the next time with the
chunk's own starting noise. The DiT generates the rest. Stage 2 (shape) is untouched.

Stage 1, --stages ss
    Every cell of the sparse-structure grid (32 per chunk side) is occupied (depth points from MIN_VIEWS images),
    free (MIN_VIEWS images see CARVE_MARGIN cells past it) or unknown, from the depth in the .npz of the input
    images. The occupied cells are encoded with TRELLIS's sparse-structure encoder. A latent voxel (2 cells per
    axis) is pinned when the depth knows all its cells, shrunk by one latent voxel, because the encoder reads unknown
    cells as empty.

Stage 3, --stages tex
    1. Stage 3 is sampled as usual (pass A), and the shape and texture are decoded.
    2. Each fine voxel (512 per chunk side) gets the mean colour of the images that see it: the voxel lies inside
       the image and at most VISIBILITY_TOLERANCE fine voxels behind the depth of the generated mesh, rendered from
       that camera. It is observed when MIN_VIEWS images see it and their colours agree (MAX_COLOUR_STD).
    3. Pass A's texture with the observed base colours pasted in is encoded with TRELLIS.2's texture encoder.
       Unobserved voxels keep pass A's colours, so the encoder never reads holes.
    4. Stage 3 is sampled again from pass A's starting noise, with the latent voxels pinned whose fine voxels are
       at least PIN_SHARE observed.

Counts are per image, not per crop: GenRecon cuts every wide image into two overlapping square crops.

Writes inpainting.json (shares of pinned cells and voxels) and observed_colours.ply (the observed fine voxels with
their colours, in the frame of mesh.ply) next to GenRecon's outputs. Patches MultiDiffusionOrchestrator.sample and
FullSceneImagesTo3DPipeline.run; GenRecon itself is not modified. Runs in GenRecon's environment, from papers/GenRecon:

    .venv/bin/python ../../experiments/genrecon_latent_inpainting/inject.py --npz NPZ --stages ss,tex \\
        <reconstruct_scene.py arguments>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import nvdiffrast.torch as dr
import o_voxel
import torch
import torch.nn.functional as F
import tqdm
import trimesh
from scipy.ndimage import binary_erosion, maximum_filter, minimum_filter

sys.path.insert(0, str(Path.cwd()))  # papers/GenRecon
import reconstruct_scene  # noqa: E402
from genrecon import models  # noqa: E402
from genrecon.modules.sparse import SparseTensor  # noqa: E402
from genrecon.pipelines.full_scene_images_to_3d import FullSceneImagesTo3DPipeline  # noqa: E402
from genrecon.pipelines.samplers.multi_diff_orchestrator import MultiDiffusionOrchestrator  # noqa: E402

SS_ENCODER = "microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16"
TEX_ENCODER = "microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16"
R = 32  # sparse-structure cells per chunk side
FINE_PER_LATENT = 16  # fine voxels per texture latent voxel and axis

MIN_VIEWS = 2  # images that must observe a cell or voxel
EDGE_RATIO = 0.1  # depth pixels whose 3x3 depth range exceeds this share of their depth are dropped
CARVE_MARGIN = 1.5  # cells: a free cell's centre lies this far in front of the depth
VISIBILITY_TOLERANCE = 3  # fine voxels: how far behind the rendered depth a voxel may lie and still be seen
MAX_COLOUR_STD = 0.1  # colours in [0, 1]: an observed voxel's images agree at least this well
PIN_SHARE = 0.5  # share of a texture latent voxel's fine voxels that must be observed to pin it
NEAR, FAR = 0.05, 100.0  # metres, rasterizer clip planes

STATE = {}  # the pipeline and its SelectedImages, stored by the patched run


# ── Shared ───────────────────────────────────────────────────────────────────


def views(out: Path) -> dict[str, list[int]]:
    """The scene images' crops, grouped by image: name -> indices into SelectedImages' scene tensors."""
    groups = {}
    for i, view in enumerate(json.loads((out / "cameras.json").read_text())["scene"]):
        groups.setdefault(Path(view["img_path"]).name, []).append(i)
    return groups


def grid_layout(relative_transl: list[torch.Tensor], res: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chunk offsets in a scene grid of res voxels per chunk side, the grid's shape, and the smallest chunk centre,
    as FullSceneImagesTo3DPipeline._global_grid_layout."""
    rel = torch.stack([t.float().cpu() for t in relative_transl]).numpy()
    mins = rel.min(axis=0)
    offsets = np.round((rel - mins) * res).astype(int)
    return offsets, offsets.max(axis=0) + res, mins


def sample_loop(orchestrator, model, chunks, cond, neg_cond, relative_transl, t_pairs, guidance, chunk_kwargs,
                aggregate, verbose, desc, pin=None):
    """MultiDiffusionOrchestrator.sample's loop. pin = (encodings, masks, starting noise) resets the pinned voxels
    after every step to the encoding, noised to the next time."""
    sigma_min = orchestrator.sampler.sigma_min
    for t, t_prev in tqdm.tqdm(t_pairs, desc=desc, disable=not verbose):
        for i in range(len(chunks)):
            extra = chunk_kwargs[i] if chunk_kwargs is not None else {}
            v = orchestrator.sampler._inference_model(model, chunks[i], t, cond=cond[i], neg_cond=neg_cond[i],
                                                      **guidance, **extra)
            chunks[i] = chunks[i] - (t - t_prev) * v
        chunks = aggregate(chunks, relative_transl, model.resolution)
        if pin is not None:
            s = sigma_min + (1 - sigma_min) * t_prev
            pinned = []
            for c, x0, m, e in zip(chunks, *pin):
                if isinstance(c, torch.Tensor):
                    pinned.append(m * ((1 - t_prev) * x0 + s * e) + (1 - m) * c)
                else:
                    pinned.append(c.replace(m * ((1 - t_prev) * x0 + s * e.feats) + (1 - m) * c.feats))
            chunks = pinned
    return chunks


# ── Stage 1: cells from the depth ────────────────────────────────────────────


def drop_edges(depth: np.ndarray) -> np.ndarray:
    """NaN at depth discontinuities, where predicted depth smears between the two surfaces."""
    return np.where(maximum_filter(depth, 3) - minimum_filter(depth, 3) < EDGE_RATIO * depth, depth, np.nan)


def depth_cells(npz: Path, out: Path, relative_transl) -> tuple[np.ndarray, np.ndarray]:
    """Occupied and free cells of the scene's cell grid, from the depth of the input images.

    The cameras are SelectedImages' extrinsics: OpenCV, from the chunk-0 frame (unit: chunk size) to camera
    coordinates in the depth's unit. A cell's centre in the chunk-0 frame is corner + (index + 0.5) / R.
    """
    sel = STATE["sel"]
    groups = views(out)
    prediction = np.load(npz)
    frames = [int(Path(name).stem) for name in groups]  # convert.py names frame i of the .npz "{i:04d}.png"
    depth_all = prediction["depth"]  # [S, H, W, 1]
    depths = [drop_edges(depth_all[f, ..., 0].astype(np.float64)) for f in frames]
    Ks = prediction["intrinsics"][frames].astype(np.float64)  # pixels, at the depth's resolution
    exts = [sel.scene_extrinsics_c0[g[0]].double().cpu().numpy() for g in groups.values()]
    height, width = depths[0].shape

    offsets, shape, mins = grid_layout(relative_transl, R)
    corner = mins - 0.5
    in_chunk = np.zeros(shape, dtype=bool)
    for x, y, z in offsets:
        in_chunk[x:x + R, y:y + R, z:z + R] = True

    occupied_views = np.zeros(shape, dtype=int)
    u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    for depth, K, ext in zip(depths, Ks, exts):
        valid = np.isfinite(depth) & (depth > NEAR)
        cam = np.stack([(u[valid] - K[0, 2]) / K[0, 0], (v[valid] - K[1, 2]) / K[1, 1], np.ones(valid.sum())], -1)
        cam = cam * depth[valid, None]
        points = (cam - ext[:3, 3]) @ np.linalg.inv(ext[:3, :3]).T  # chunk-0 frame
        cells = np.floor((points - corner) * R).astype(int)
        cells = cells[((cells >= 0) & (cells < shape)).all(axis=1)]
        cells = np.unique(cells[in_chunk[tuple(cells.T)]], axis=0)
        occupied_views[tuple(cells.T)] += 1
    occupied = occupied_views >= MIN_VIEWS

    cells = np.argwhere(in_chunk & ~occupied)
    centres = corner + (cells + 0.5) / R
    past = np.zeros(len(cells), dtype=int)
    for depth, K, ext in zip(depths, Ks, exts):
        cam = centres @ ext[:3, :3].T + ext[:3, 3]
        z = cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            px, py = np.floor(K[0, 0] * cam[:, 0] / z + K[0, 2]), np.floor(K[1, 1] * cam[:, 1] / z + K[1, 2])
        inside = (z > NEAR) & (px >= 0) & (px < width) & (py >= 0) & (py < height)
        observed = np.full(len(cells), np.nan)
        observed[inside] = depth[py[inside].astype(int), px[inside].astype(int)]
        cell_size = np.linalg.norm(ext[0, :3]) / R  # a chunk-0 unit is one chunk size
        with np.errstate(invalid="ignore"):
            past += z < observed - CARVE_MARGIN * cell_size  # NaN (outside, no depth, edge) compares False
    free = np.zeros(shape, dtype=bool)
    free[tuple(cells[past >= MIN_VIEWS].T)] = True
    covered = in_chunk.sum()
    STATE["stats"]["ss_occupied_cells"] = float(occupied.sum() / covered)
    STATE["stats"]["ss_free_cells"] = float(free.sum() / covered)
    print(f"[inpaint] cells: {occupied.sum() / covered:.1%} occupied, {free.sum() / covered:.1%} free", flush=True)
    return occupied, free


@torch.no_grad()
def ss_pins(occupied: np.ndarray, free: np.ndarray, relative_transl, R_lat: int, chunks: list[torch.Tensor]):
    """Per chunk, the encoding of the occupied cells and the mask of pinned latent voxels."""
    f = R // R_lat  # cells per latent voxel and axis
    lx, ly, lz = (s // f for s in occupied.shape)
    known = (occupied | free).reshape(lx, f, ly, f, lz, f).all(axis=(1, 3, 5))
    known = binary_erosion(known, np.ones((3, 3, 3), dtype=bool), border_value=1)
    STATE["stats"]["ss_pinned_latents"] = float(known.mean())
    print(f"[inpaint] stage 1: pinned {known.mean():.1%} of the scene's latent voxels", flush=True)

    device, dtype = chunks[0].device, chunks[0].dtype
    encoder = models.from_pretrained(SS_ENCODER).to(device).eval()
    # the encoder reads occupancy at the decoder's resolution, 2 voxels per cell and axis
    x = torch.from_numpy(occupied.repeat(2, 0).repeat(2, 1).repeat(2, 2)).float()[None, None].to(device)
    z = encoder(x).to(dtype)  # [1, 8, lx, ly, lz]
    del encoder, x
    torch.cuda.empty_cache()
    known = torch.from_numpy(known)[None, None].to(device, dtype)
    offsets, _, _ = grid_layout(relative_transl, R_lat)
    x0 = [z[:, :, x:x + R_lat, y:y + R_lat, w:w + R_lat] for x, y, w in offsets]
    masks = [known[:, :, x:x + R_lat, y:y + R_lat, w:w + R_lat] for x, y, w in offsets]
    return x0, masks


# ── Stage 3: colours of the fine voxels ──────────────────────────────────────


def render_depth(glctx, vertices: torch.Tensor, faces: torch.Tensor, ext: torch.Tensor, K: torch.Tensor,
                 size: int) -> torch.Tensor:
    """Camera depth of the mesh for a square view with normalized intrinsics, inf where it shows no mesh.

    Row r and column c of the result are at v = (r + 0.5) / size and u = (c + 0.5) / size: nvdiffrast's first row is
    clip y = -1, which is v = 0 here.
    """
    cam = vertices @ ext[:3, :3].T + ext[:3, 3]
    x, y, z = cam.unbind(-1)
    clip = torch.stack([2 * K[0, 0] * x + (2 * K[0, 2] - 1) * z,
                        2 * K[1, 1] * y + (2 * K[1, 2] - 1) * z,
                        (z * (FAR + NEAR) - 2 * FAR * NEAR) / (FAR - NEAR),
                        z], dim=-1)[None].contiguous()
    rast, _ = dr.rasterize(glctx, clip, faces, (size, size))
    depth, _ = dr.interpolate(z[None, :, None].contiguous(), rast, faces)
    return torch.where(rast[0, ..., 3] > 0, depth[0, ..., 0], torch.full_like(depth[0, ..., 0], float("inf")))


@torch.no_grad()
def observed_colours(points: torch.Tensor, mesh, voxel_size: float, out: Path):
    """Mean colour of the images that see each point (chunk-0 frame), and whether it counts as observed."""
    sel = STATE["sel"]
    device = points.device
    glctx = dr.RasterizeCudaContext(device=device)
    vertices, faces = mesh.vertices.float(), mesh.faces.int().contiguous()
    size = sel.scene_images_512.shape[-1]
    total, total_sq = torch.zeros(len(points), 3, device=device), torch.zeros(len(points), 3, device=device)
    count = torch.zeros(len(points), device=device)
    for name, crops in views(out).items():
        seen = torch.zeros(len(points), dtype=torch.bool, device=device)
        colour = torch.zeros(len(points), 3, device=device)
        for i in crops:
            ext = sel.scene_extrinsics_c0[i].float().to(device)
            K = sel.scene_intrinsics[i].float().to(device)
            depth = render_depth(glctx, vertices, faces, ext, K, size)
            cam = points @ ext[:3, :3].T + ext[:3, 3]
            z = cam[:, 2]
            safe = torch.where(z > NEAR, z, torch.ones_like(z))
            u, v = K[0, 0] * cam[:, 0] / safe + K[0, 2], K[1, 1] * cam[:, 1] / safe + K[1, 2]
            inside = (z > NEAR) & (u >= 0) & (u < 1) & (v >= 0) & (v < 1) & ~seen
            px = (u * size).long().clamp(0, size - 1)
            py = (v * size).long().clamp(0, size - 1)
            tolerance = VISIBILITY_TOLERANCE * voxel_size * ext[0, :3].norm()  # chunk-0 units to the depth's unit
            visible = inside & (z <= depth[py, px] + tolerance)
            idx = visible.nonzero().squeeze(1)
            if len(idx):
                image = sel.scene_images_1024[i].float().to(device)[None]  # [1, 3, H, W] in [0, 1]
                grid = torch.stack([2 * u[idx] - 1, 2 * v[idx] - 1], dim=-1)[None, None]
                colour[idx] = F.grid_sample(image, grid, align_corners=False)[0, :, 0].T
            seen |= visible
        count += seen.float()
        total += colour
        total_sq += colour**2
    mean = total / count.clamp(min=1)[:, None]
    std = (total_sq / count.clamp(min=1)[:, None] - mean**2).clamp(min=0).sqrt().mean(dim=1)
    observed = (count >= MIN_VIEWS) & (std < MAX_COLOUR_STD)
    return mean, observed, count


def save_colours(points: torch.Tensor, colours: torch.Tensor, out: Path, limit: int = 2_000_000):
    """The observed fine voxels as a coloured point cloud, in the frame of mesh.ply."""
    chunk0 = np.array(json.loads((out / "chunk_transforms.json").read_text())["chunks"][0]["M_chunk_to_original"])
    keep = torch.randperm(len(points), device=points.device)[:limit]
    p = points[keep].double().cpu().numpy() @ chunk0[:3, :3].T + chunk0[:3, 3]
    c = (colours[keep].clamp(0, 1) * 255).byte().cpu().numpy()
    trimesh.PointCloud(p, colors=np.concatenate([c, np.full((len(c), 1), 255, np.uint8)], axis=1)).export(
        out / "observed_colours.ply")


@torch.no_grad()
def tex_pins(texture: SparseTensor, mesh, aabb: torch.Tensor, res: int, relative_transl, R_lat: int,
             chunks: list[SparseTensor], out: Path):
    """Per chunk, the normalized encoding of pass A's texture with the observed colours pasted in, and the mask of
    pinned latent voxels."""
    pipeline = STATE["pipeline"]
    device = texture.feats.device
    fine = texture.coords[:, 1:].long()  # joint fine grid
    points = aabb[0].to(device) + (fine.float() + 0.5) / res  # chunk-0 frame
    mean, observed, count = observed_colours(points, mesh, 1.0 / res, out)
    STATE["stats"]["tex_observed_voxels"] = float(observed.float().mean())
    STATE["stats"]["tex_seen_voxels"] = float((count > 0).float().mean())
    print(f"[inpaint] stage 3: {(count > 0).float().mean():.1%} of the fine voxels seen, "
          f"{observed.float().mean():.1%} observed", flush=True)
    save_colours(points[observed], mean[observed], out)

    feats = texture.feats.float().clone()
    feats[observed, 0:3] = mean[observed]
    order = torch.argsort(o_voxel.serialize.encode_seq(fine.int().cpu())).to(device)  # as get_slats.py
    encoder = models.from_pretrained(TEX_ENCODER).to(device).eval()
    z = encoder(SparseTensor(feats[order] * 2 - 1, texture.coords[order]))  # coarse joint grid, R_lat per chunk side
    del encoder, feats
    torch.cuda.empty_cache()
    tex_mean = torch.tensor(pipeline.tex_slat_normalization["mean"])[None].to(device)
    tex_std = torch.tensor(pipeline.tex_slat_normalization["std"])[None].to(device)
    z_norm = (z.feats.float() - tex_mean) / tex_std

    # Share of observed fine voxels per latent voxel, keyed in the coarse joint grid
    coarse = z.coords[:, 1:].long()
    dims = torch.maximum(coarse.max(dim=0).values, (fine // FINE_PER_LATENT).max(dim=0).values) + 1

    def key(c):
        return (c[:, 0] * dims[1] + c[:, 1]) * dims[2] + c[:, 2]

    z_keys, z_order = key(coarse).sort()
    fine_keys = key(fine // FINE_PER_LATENT)
    share = torch.zeros(len(z_keys), device=device)
    n = torch.zeros(len(z_keys), device=device)
    idx = torch.searchsorted(z_keys, fine_keys).clamp(max=len(z_keys) - 1)
    found = z_keys[idx] == fine_keys
    share.index_add_(0, idx[found], observed[found].float())
    n.index_add_(0, idx[found], torch.ones(int(found.sum()), device=device))
    pinned = (share / n.clamp(min=1)) >= PIN_SHARE

    offsets, _, _ = grid_layout(relative_transl, R_lat)
    x0, masks, total = [], [], 0
    for c, offset in zip(chunks, offsets):
        k = key(c.coords[:, 1:].long() + torch.from_numpy(offset).to(device))
        j = torch.searchsorted(z_keys, k).clamp(max=len(z_keys) - 1)
        found = z_keys[j] == k
        mask = (found & pinned[j]).float()[:, None]
        x0.append(torch.where(found[:, None], z_norm[z_order[j]], torch.zeros_like(z_norm[z_order[j]])))
        masks.append(mask)
        total += mask.sum().item()
    share_pinned = total / sum(len(c.coords) for c in chunks)
    STATE["stats"]["tex_pinned_latents"] = share_pinned
    print(f"[inpaint] stage 3: pinned {share_pinned:.1%} of the chunks' latent voxels", flush=True)
    return x0, masks


# ── Patches ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--npz", type=Path, help="Depth and cameras of the input images (needed for --stages ss).")
    parser.add_argument("--stages", default="ss,tex", help="Comma-separated: ss, tex.")
    args, rest = parser.parse_known_args()
    stages = set(args.stages.split(","))
    if not stages <= {"ss", "tex"}:
        raise SystemExit(f"unknown stages: {stages - {'ss', 'tex'}}")
    if "ss" in stages and args.npz is None:
        raise SystemExit("--stages ss needs --npz")
    out = Path(reconstruct_scene.build_parser().parse_args(rest).output_path)
    STATE["stats"] = {"stages": sorted(stages)}

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
        tex = not dense and chunk_kwargs is not None and "concat_cond" in chunk_kwargs[0]
        if not ((dense and "ss" in stages) or (tex and "tex" in stages)):
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
            return sample_loop(self, model, chunks, cond, neg_cond, relative_transl, t_pairs, guidance,
                               chunk_kwargs, aggregate, verbose, desc, pin)

        if dense:  # stage 1
            with torch.autocast("cuda", enabled=False):
                occupied, free = depth_cells(args.npz, out, relative_transl)
                x0, masks = ss_pins(occupied, free, relative_transl, R_lat, start)
            result = loop([c.clone() for c in start], tqdm_desc + " (inpainted)", (x0, masks, start))
        else:  # stage 3
            pipeline = STATE["pipeline"]
            pass_a = loop([c.replace(c.feats.clone()) for c in start], tqdm_desc + " (pass A)")
            res = R_lat * FINE_PER_LATENT
            with torch.autocast("cuda", enabled=False):
                std = torch.tensor(pipeline.shape_slat_normalization["std"])[None].to(pipeline.device)
                mean = torch.tensor(pipeline.shape_slat_normalization["mean"])[None].to(pipeline.device)
                shapes = [k["concat_cond"] * std + mean for k in chunk_kwargs]
                mesh, aabb, _, ctx, _ = pipeline.joint_decode_shape_slats(shapes, res, relative_transl)
                del shapes
                tex_std = torch.tensor(pipeline.tex_slat_normalization["std"])[None].to(pipeline.device)
                tex_mean = torch.tensor(pipeline.tex_slat_normalization["mean"])[None].to(pipeline.device)
                texture = pipeline.joint_decode_tex_slats([s * tex_std + tex_mean for s in pass_a], ctx,
                                                          relative_transl)
                del ctx, pass_a
                torch.cuda.empty_cache()
                x0, masks = tex_pins(texture, mesh, aabb, res, relative_transl, R_lat, start, out)
                del texture, mesh
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
