# scene-reconstruction

## Layout

```text
papers/       vendored paper code, each with its own environment
experiments/  one subfolder per experiment
scripts/      dataset download, preprocessing and evaluation
data/         datasets (not tracked)
```

## Setup

```bash
uv sync
```

### ScanNet++ data

Create a `.env` file in the repo root with your ScanNet++ token:

```bash
echo 'SCANNETPP_TOKEN=<your token>' > .env
```

Then download the data:

```bash
uv run python scripts/download_scannetpp.py scripts/download_scannetpp.yml
```

### ScanNet++ evaluation

Every experiment is evaluated the same way:

- Scenes: the 50 `nvs_sem_val` scenes.
- Input: N undistorted DSLR images per scene, picked by farthest-point sampling of the camera centres.
- Metrics: each predicted mesh is compared with the laser scan, then averaged over the scenes. Depth MAE,
  depth AbsRel and completeness come from depth rendered at the DSLR test views. Chamfer distance,
  F-score at 10 cm and normal consistency come from points sampled on both meshes.

```bash
uv run python scripts/download_scannetpp.py scripts/download_scannetpp_eval.yml  # the 50 scenes, ~57 GB
uv run scripts/select_views_scannetpp.py 32  # data/scannetpp/eval_inputs/32views/<scene>/
# run the method on each input scene and write <pred_dir>/<scene>/mesh.ply in the scan's coordinates
uv run scripts/evaluate_scannetpp.py <pred_dir>
```

## Papers

| Name     | Folder          | Upstream |
| -------- | --------------- | -------- |
| GenRecon | papers/GenRecon | https://github.com/kasothaphie/GenRecon |
| VGGT-Omega | papers/VGGT-Omega | https://github.com/facebookresearch/vggt-omega |

See `papers/README.md` for imported commits and licenses.
