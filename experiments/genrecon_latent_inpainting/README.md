# GenRecon with latent inpainting in stages 1 and 3

Do the sparse structure and the texture improve if GenRecon only generates what the input views don't show?

GenRecon conditions each stage on DINOv3 features averaged over every view a voxel projects into, without an
occlusion test, so well-observed surfaces still get smeared textures. Here, what the views observed is written into
the latents directly, RePaint-style, as `experiments/genrecon_inpaint_occupancy` (branch `feat/stage1-inpainting`)
does for stage 1: encode it with the stage's VAE encoder, and after every sampling step reset the pinned latent
voxels to that encoding, noised to the next time. The DiT generates the rest. No training.

| Stage | What is pinned | Encoder |
| --- | --- | --- |
| 1, `ss` | Cells the depth knows: occupied (depth points from 2 images) or free (2 images see past it) | TRELLIS sparse structure |
| 3, `tex` | Texture of fine voxels that 2 images see unoccluded, with matching colours | TRELLIS.2 texture |

Stage 2 (shape) is left alone. Stage 3 samples twice. Pass A is the usual texture, whose decoded colours fill the
voxels no image observes, so the encoder never reads holes. Pass B is the inpainted texture, from the same starting
noise. Visibility comes from the depth of the generated mesh, rendered from each camera. Details and thresholds are
in `inject.py`.

Requires a scene made by `experiments/unposed_genrecon` (its `scene/` folder and `.npz` with depth), the GenRecon env,
and the TRELLIS encoders `microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16` and
`microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16` (downloaded from Hugging Face on first use).

## Classroom

The classroom video's 16 handpicked views, with Solv3R's cameras and depth and the settings of
`experiments/unposed_genrecon/classroom_handpicked.sbatch`, whose output is the baseline:

```bash
mkdir -p logs
sbatch experiments/genrecon_latent_inpainting/classroom.sbatch tex      # stage 3 only
sbatch experiments/genrecon_latent_inpainting/classroom.sbatch ss       # stage 1 only
sbatch experiments/genrecon_latent_inpainting/classroom.sbatch ss,tex   # both
```

Outputs go to `outputs/genrecon_latent_inpainting/2026-11-26_classroom_handpicked/<stages>/`, as GenRecon's plus:

- `inpainting.json`: shares of occupied, free and pinned cells (stage 1), and of seen, observed and pinned voxels
  (stage 3)
- `observed_colours.ply`: the observed fine voxels with their colours, in the frame of `mesh.ply`. Check that it
  lines up with the mesh before reading anything into the result.
