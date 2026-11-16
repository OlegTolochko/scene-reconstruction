"""GenRecon's reconstruct_scene.py with stage 1 inpainted around the cells the input views' depth knows.

Stage 1 samples a latent of 16 voxels per chunk side, which the decoder turns into 32 cells per side, so each
latent voxel covers 2 cells per axis. Before sampling, the occupied cells of cells.py are encoded with TRELLIS's
sparse-structure encoder (the partner of the decoder GenRecon uses), once on the whole scene's canvas, as
joint_decode_sparse_structure decodes it. A latent voxel is pinned when the depth knows all its cells (occupied or
free), shrunk by one latent voxel, because the encoder reads unknown cells as empty and that leaks into its
neighbours. After every Euler step (and the MultiDiffusion averaging) the pinned voxels are reset to the encoding,
noised to the next time as GenRecon's FlowEulerSampler._apply_inpainting_constraint does, and the DiT generates
the rest. Shape and texture stages are untouched.

With --fix, the cells stage 2 gets are corrected afterwards as well: the occupied cells are added and the free
cells removed.

Patches MultiDiffusionOrchestrator.sample (and with --fix FullSceneImagesTo3DPipeline.extract_coords_from_occ_logits);
GenRecon itself is not modified. Runs in GenRecon's environment, from papers/GenRecon (run.py does this):
    .venv/bin/python ../../experiments/genrecon_inpaint_occupancy/inject.py CELLS.npz [--fix] <reconstruct_scene.py arguments>
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import tqdm
from scipy.ndimage import binary_erosion

sys.path.insert(0, str(Path.cwd()))  # papers/GenRecon
import reconstruct_scene  # noqa: E402
from genrecon import models  # noqa: E402
from genrecon.pipelines.full_scene_images_to_3d import FullSceneImagesTo3DPipeline  # noqa: E402
from genrecon.pipelines.samplers.multi_diff_orchestrator import MultiDiffusionOrchestrator  # noqa: E402

R = 32  # cells per chunk side
ENCODER = "microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16"


def grids(cells) -> tuple[np.ndarray, np.ndarray]:
    """The occupied and free cells as boolean grids over the scene's canvas."""
    occupied, free = np.zeros(cells["shape"], dtype=bool), np.zeros(cells["shape"], dtype=bool)
    occupied[tuple(cells["occupied"].astype(int).T)] = True
    free[tuple(cells["free"].astype(int).T)] = True
    return occupied, free


@torch.no_grad()
def pinned_latents(cells, device) -> tuple[torch.Tensor, torch.Tensor]:
    """The encoding of the occupied cells and the mask of pinned latent voxels, over the scene's canvas."""
    occupied, free = grids(cells)
    f = 2  # cells per latent voxel and axis
    lx, ly, lz = (s // f for s in occupied.shape)
    known = (occupied | free).reshape(lx, f, ly, f, lz, f).all(axis=(1, 3, 5))
    known = binary_erosion(known, np.ones((3, 3, 3), dtype=bool), border_value=1)

    encoder = models.from_pretrained(ENCODER).to(device).eval()
    # the encoder reads occupancy at the decoder's resolution, 2 voxels per cell and axis
    x = torch.from_numpy(occupied.repeat(2, 0).repeat(2, 1).repeat(2, 2)).float()[None, None].to(device)
    z = encoder(x)  # [1, 8, lx, ly, lz]
    del encoder, x
    torch.cuda.empty_cache()
    print(f"[inpaint] pinned {known.mean():.1%} of the scene's latent voxels", flush=True)
    return z, torch.from_numpy(known)[None, None].to(device)


def check_layout(cells, out: Path):
    chunks = json.loads((out / "chunk_transforms.json").read_text())["chunks"]
    if not np.allclose([c["crop_center"] for c in chunks], cells["centres"]):
        raise RuntimeError("the chunk layout differs from the baseline run's")


def main():
    cells = np.load(sys.argv[1])
    fix = sys.argv[2] == "--fix"
    sys.argv = [sys.argv[0], *sys.argv[3 if fix else 2:]]  # the reconstruct_scene.py arguments
    out = Path(reconstruct_scene.build_parser().parse_args(sys.argv[1:]).output_path)
    original_sample = MultiDiffusionOrchestrator.sample
    original_extract = FullSceneImagesTo3DPipeline.extract_coords_from_occ_logits

    @torch.no_grad()
    def inpainted_sample(self, model, noise, cond, neg_cond, relative_transl, steps=50, rescale_t=1.0,
                         verbose=True, tqdm_desc="Sampling", chunk_kwargs=None, **guidance):
        if not isinstance(noise[0], torch.Tensor):  # the sparse shape and texture stages
            return original_sample(self, model, noise, cond, neg_cond, relative_transl, steps=steps, rescale_t=rescale_t,
                                   verbose=verbose, tqdm_desc=tqdm_desc, chunk_kwargs=chunk_kwargs, **guidance)
        guidance = {k: v for k, v in guidance.items() if k.startswith("guidance")}
        check_layout(cells, out)
        R_lat = model.resolution
        offsets = torch.stack([(t.cpu() * R_lat).round().long() for t in relative_transl])
        offsets = (offsets - offsets.min(dim=0).values).numpy()
        if not (offsets * R // R_lat == cells["offsets"]).all():
            raise RuntimeError("the chunks' latent offsets differ from the cells' chunk offsets")

        chunks = self._initialize_noise(noise, relative_transl, R_lat)
        start = [c.clone() for c in chunks]
        z, known = pinned_latents(cells, chunks[0].device)
        x0 = [z[:, :, x:x + R_lat, y:y + R_lat, w:w + R_lat] for x, y, w in offsets]
        mask = [known[:, :, x:x + R_lat, y:y + R_lat, w:w + R_lat].to(z.dtype) for x, y, w in offsets]

        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = (rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)).tolist()
        sigma_min = self.sampler.sigma_min
        for t, t_prev in tqdm.tqdm(list(zip(t_seq[:-1], t_seq[1:])), desc=tqdm_desc, disable=not verbose):
            for i in range(len(chunks)):
                extra = chunk_kwargs[i] if chunk_kwargs is not None else {}
                v = self.sampler._inference_model(model, chunks[i], t, cond=cond[i], neg_cond=neg_cond[i], **guidance, **extra)
                chunks[i] = chunks[i] - (t - t_prev) * v
            chunks = self._aggregate_overlaps(chunks, relative_transl, R_lat)
            # the pinned voxels as the encoding noised to t_prev, with the chunk's own starting noise
            s = sigma_min + (1 - sigma_min) * t_prev
            chunks = [m * ((1 - t_prev) * x + s * e.to(c.dtype)) + (1 - m) * c
                      for c, x, m, e in zip(chunks, x0, mask, start)]
        return chunks

    def fixed_extract(self, aggr_occ_logit_list, *args, **kwargs):
        """Stage 1's cells with the occupied cells added and the free ones removed."""
        coords_list = original_extract(self, aggr_occ_logit_list, *args, **kwargs)
        occupied, free = grids(cells)
        fixed = []
        for coords, (x, y, w) in zip(coords_list, cells["offsets"]):
            grid = np.zeros((R, R, R), dtype=bool)
            grid[tuple(coords[:, 1:].long().cpu().numpy().T)] = True
            grid = (grid | occupied[x:x + R, y:y + R, w:w + R]) & ~free[x:x + R, y:y + R, w:w + R]
            if not grid.any():  # stage 2 fails on a chunk without cells
                fixed.append(coords)
                continue
            xyz = torch.from_numpy(np.argwhere(grid)).int().to(coords.device)
            fixed.append(torch.cat([torch.zeros(len(xyz), 1, dtype=torch.int32, device=coords.device), xyz], dim=1))
        print(f"[inpaint] fix: {sum(map(len, coords_list))} -> {sum(map(len, fixed))} cells", flush=True)
        return fixed

    MultiDiffusionOrchestrator.sample = inpainted_sample
    if fix:
        FullSceneImagesTo3DPipeline.extract_coords_from_occ_logits = fixed_extract
    reconstruct_scene.main()


if __name__ == "__main__":
    main()
