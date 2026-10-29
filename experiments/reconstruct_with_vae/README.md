# VAE reconstruction

`reconstruct_with_vae.py` reconstructs a reference mesh through GenRecon's
pretrained VAEs. Install the [GenRecon environment](../../papers/GenRecon/README_scene-reconstruction.md#environment),
then run from the repository root:

```bash
papers/GenRecon/.venv/bin/python experiments/reconstruct_with_vae/reconstruct_with_vae.py \
    --input path/to/reference_scene.glb \
    --output outputs/vae_reconstruction/example \
    --chunk-size 3.0
```

- Input: a mesh or scene file. `--chunk-size` is the cube edge in its source
  units; choose it for the dataset's scale.
- Output: `reconstructed.glb`, per-chunk latents and meshes, and settings and
  transforms in `run.json`. The output directory must be empty.
- Resolution: `--resolution 512` (default) or `1024`.
- Coordinates: GLB/glTF inputs default to Y-up, other formats to Z-up.
  Override with `--up-axis y|z`. Output GLBs are Y-up.

Add `--textures` for appearance reconstruction. `reconstructed.glb` includes
decoded vertex RGB; `textured_preview.glb` additionally bakes PBR textures.
It accepts PBR materials or vertex RGB (including ScanNet++ PLYs). Vertex RGB
is sampled onto the shape voxel grid and passed through the appearance VAE
with input defaults metallic=0, roughness=1, alpha=1; these are assumptions,
not measured materials. `--texture-size` defaults to 2048 per chunk. The preview
exporter cleans and simplifies geometry; use `reconstructed.glb` for raw geometry.

Chunks decode independently, so seams may remain. The experiment uses raw
encoded latents and their coordinates, without image conditioning or diffusion.
