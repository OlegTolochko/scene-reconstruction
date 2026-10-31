# Unposed GenRecon

Run GenRecon on unposed images by estimating cameras with VGGT-Omega.

## Requirements

- The VGGT-Omega uv env and checkpoint, set up as in
  `papers/VGGT-Omega/README_scene-reconstruction.md`.
- The GenRecon uv env and checkpoints, set up as in
  `papers/GenRecon/README_scene-reconstruction.md`.


## Run


```bash
python experiments/unposed_genrecon/run.py path/to/images_or_video 
```

## Outputs

Everything goes to `outputs/unposed_genrecon/`:

```text
outputs/unposed_genrecon/
├── <name>.npz                    VGGT-Omega predictions (infer.py)
├── scene/                        GenRecon input (convert.py)
│   ├── rgb/                      
│   └── colmap/
│       ├── cameras.txt           intrinsics, one camera per image
│       ├── images.txt            poses, z up, scaled to metres
│       └── points3D.txt          point cloud, used only to place the chunks
└── <name>_genrecon/              GenRecon outputs
    ├── scene.glb                 final textured scene (chunked_to_glb.py)
    ├── mesh.ply                  untextured scene mesh
    ├── chunk_layout.png          top-down view of the chunks
    ├── clean_points.ply          point cloud 
    ├── coords_000.ply ...        occupied voxels (per chunk)
    ├── cameras.json              images and poses GenRecon used
    ├── chunk_transforms.json     position and size of each chunk
    ├── args.json                 arguments of the run
    ├── to_glb_inputs.pt          input for chunked_to_glb.py
    ├── chunk_inputs.pt           input for chunked_to_glb.py
    └── chunks/                   per-chunk GLB pieces, kept so a rerun can resume
```

VGGT-Omega and the GenRecon steps are skipped if their output already exists,
so rerunning continues where a failed run stopped. Delete an output to redo
that step. `scene/` is shared by all inputs, so `convert.py` always reruns.

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
papers/VGGT-Omega/.venv/bin/python experiments/unposed_genrecon/to_ply.py outputs/unposed_genrecon/<name>.npz
```

Open the `.ply` in MeshLab, Blender, CloudCompare or https://3dviewer.net.
