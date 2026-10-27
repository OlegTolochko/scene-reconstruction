# Unposed GenRecon

Run GenRecon on unposed images by estimating cameras with VGGT-Omega first.

## Requirements

- The VGGT-Omega uv env and checkpoint, set up as in
  `papers/VGGT-Omega/README_scene-reconstruction.md`.

## 1. Run VGGT-Omega

Run from the repo root, on a video or on a folder of images:

```bash
# Video (16 evenly spaced frames by default, change with --num-frames)
papers/VGGT-Omega/.venv/bin/python experiments/unposed_genrecon/infer.py papers/VGGT-Omega/examples/forest_road.mp4

# Folder of images (uses every image in the folder)
papers/VGGT-Omega/.venv/bin/python experiments/unposed_genrecon/infer.py path/to/image_folder
```

The predictions (images, depth, depth confidence, extrinsics, intrinsics) are
saved to `outputs/unposed_genrecon/<input name>.npz`.

## 2. Check the result (optional)

Turn the predictions into a coloured point cloud, with camera positions in red:

```bash
papers/VGGT-Omega/.venv/bin/python experiments/unposed_genrecon/to_ply.py outputs/unposed_genrecon/forest_road.npz
```

Open the `.ply` in MeshLab, Blender, CloudCompare or https://3dviewer.net.
