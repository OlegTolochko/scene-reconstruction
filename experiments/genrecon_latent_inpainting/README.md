# GenRecon with latent inpainting in stage 2 (shape)

Does the shape improve if GenRecon only generates the geometry the input views' depth doesn't show?

What the depth knows is written into stage 2's latents directly, RePaint-style, as
`experiments/genrecon_inpaint_occupancy` (branch `feat/stage1-inpainting`) does for stage 1: encode it with the
stage's VAE encoder, and after every sampling step reset the pinned latent voxels to that encoding, noised to the next
time. The DiT generates the rest. No training. Stages 1 and 3 are left alone.

Stage 2 samples twice. Pass A is the usual shape. The input depth maps are fused into a TSDF, and its surface replaces
pass A's fine voxels wherever at least 2 images see the whole latent voxel. Pass A fills everything else, so the
encoder never reads holes. Pass B is sampled from the same starting noise with those latent voxels pinned to the
encoding. Details and thresholds are in `inject.py`.

Requires a scene made by `experiments/unposed_genrecon` (its `scene/` folder and `.npz` with depth), the GenRecon env
(with Open3D), and the TRELLIS.2 shape encoder `microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16`
(downloaded from Hugging Face on first use).

## Classroom

The classroom video's 16 handpicked views, with Solv3R's cameras and depth and the settings of
`experiments/unposed_genrecon/classroom_handpicked.sbatch` (job 4254), whose output is the baseline:

```bash
mkdir -p logs
sbatch experiments/genrecon_latent_inpainting/classroom.sbatch
```

Outputs go to `outputs/genrecon_latent_inpainting/2026-11-26_classroom_handpicked/shape/`, as GenRecon's plus:

- `inpainting.json`: shares of known, known after shrinking, and pinned latent voxels
- `depth_surface.ply`: the TSDF surface of the input depth, in the frame of `mesh.ply`. Check that it lines up with
  the mesh before reading anything into the result.
