# VAE reconstruction

`reconstruct_with_vae.py` reconstructs a reference mesh through GenRecon's
pretrained VAEs. Install the [GenRecon environment](../../papers/GenRecon/README_scene-reconstruction.md#environment),
then run from the repository root:

```bash
papers/GenRecon/.venv/bin/python experiments/reconstruct_with_vae/reconstruct_with_vae.py \
    --input path/to/reference_scene.glb \
    --genrecon-run outputs/genrecon/example \
    --output outputs/vae_reconstruction/example
```

- Input: an aligned reference mesh and the specific GenRecon run for its scene
  and camera set. Keep `chunk_transforms.json`, `chunk_inputs.pt` and `args.json`.
  Validation-crop runs are unsupported.
- Grid: clip only the cubes listed in `chunk_inputs.pt["chunk_indices"]`, using
  their saved transforms and order. Resolution comes from `args.json["pipeline"]`;
  voxel size is the saved cube edge / resolution. There is no independent grid.
- Output: `mesh.ply` in GenRecon's world frame for evaluation, `reconstructed.glb`,
  per-chunk latents, joint `decoded.ply`, and settings/transforms in `run.json`.
  The output directory must be empty. `decoded.ply` uses original chunk 0's frame,
  even if that chunk was dropped; `run.json["joint_to_world"]` maps it to world.
- Coordinates: GLB/glTF inputs default to Y-up, other formats to Z-up.
  Override with `--up-axis y|z`. Output GLBs are Y-up.

To evaluate, place `mesh.ply` at `<pred_dir>/<scene>/mesh.ply` and run
`scripts/evaluate_scannetpp.py <pred_dir>` (add `--data data/3dfront` for 3D-FRONT), see the
[README](../../README.md#scannet-evaluation). The GenRecon run is the baseline's output for that scene:
`outputs/genrecon_baseline[/3dfront]/<N>views/<scene>/`.

Add `--textures` for appearance reconstruction. `reconstructed.glb` includes
decoded vertex RGB; `textured_preview.glb` additionally bakes PBR textures.
It accepts PBR materials or vertex RGB (including ScanNet++ PLYs). Vertex RGB
is sampled onto the shape voxel grid and passed through the appearance VAE
with input defaults metallic=0, roughness=1, alpha=1; these are assumptions,
not measured materials. `--texture-size` defaults to 2048 for the scene. The preview
exporter cleans and simplifies geometry; use `reconstructed.glb` for raw geometry.

Upstream `joint_decode_shape` / `joint_decode_tex` average overlapping latents
and decode the scene jointly, followed by GenRecon's `fill_holes()`. Texture
attributes and previews share that joint lattice. The experiment uses raw encoded
latents, without image conditioning or diffusion; it is not a guaranteed metric upper bound.
