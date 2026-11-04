# Camera accuracy

`eval_cameras.py` takes the camera poses VGGT-Omega predicted for each ScanNet++ eval scene
(`outputs/unposed_genrecon/<N>views/<scene>/scene/colmap/images.txt`, written by
`experiments/unposed_genrecon/run_eval.py`, which must have run first) and compares them with the
ScanNet++ poses of the same photos (`data/scannetpp/eval_inputs/<N>views/<scene>/dslr/nerfstudio/transforms_undistorted.json`).

```bash
uv run experiments/camera_poses/eval_cameras.py              # 32 views
```

Writes `cameras.json` into each scene folder of `outputs/unposed_genrecon/<N>views/` and prints one
line per scene (worst first), the mean over scenes, and the Spearman correlation of each metric with
the mesh F-score. 
