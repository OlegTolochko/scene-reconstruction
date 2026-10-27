# Unposed GenRecon

Run GenRecon on unposed images by estimating cameras with VGGT-Omega.

## Requirements

- The VGGT-Omega uv env and checkpoint, set up as in
  `papers/VGGT-Omega/README_scene-reconstruction.md`.
- The GenRecon uv env and checkpoints, set up as in
  `papers/GenRecon/README_scene-reconstruction.md`.


## Run


```bash
python experiments/unposed_genrecon/run.py path/to/images_or_video --name my_scene
```

Everything goes to `outputs/unposed_genrecon/<name>/`:

```text
<name>.npz        VGGT-Omega predictions
scene/            images + COLMAP files in the layout GenRecon's Iphone mode reads
genrecon/         GenRecon outputs, ending in scene.glb
```

A step is skipped if its output already exists, so rerunning continues where a
failed run stopped. Delete a step's output to redo it.

## Steps

| Step | Script | Env |
| ---- | ------ | --- |
| Estimate cameras and depth | `infer.py` | VGGT-Omega |
| Align, scale, write COLMAP files | `convert.py` | GenRecon |
| Generate the scene | `papers/GenRecon/reconstruct_scene.py` | GenRecon |
| Bake `scene.glb` | `papers/GenRecon/chunked_to_glb.py` | GenRecon |


## Check the VGGT-Omega result (optional)

Turn the predictions into a coloured point cloud, with camera positions in red:

```bash
papers/VGGT-Omega/.venv/bin/python experiments/unposed_genrecon/to_ply.py outputs/unposed_genrecon/my_scene/vggt.npz
```

Open the `.ply` in MeshLab, Blender, CloudCompare or https://3dviewer.net.
