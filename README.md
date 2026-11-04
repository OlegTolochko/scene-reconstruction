# scene-reconstruction

## Layout

```text
papers/       vendored paper code, each with its own environment
experiments/  one subfolder per experiment
scripts/      dataset download, preprocessing and evaluation
data/         datasets (not tracked)
outputs/      generated experiment results (not tracked)
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

### 3D-FRONT evaluation

The same scripts and metrics, on 3D-FRONT rooms rendered from cameras inside the room. The reference
mesh is furniture plus the room shell (walls, floor, ceiling) in metres. A room is written in the
ScanNet++ layout under `data/3dfront/`, with 32 held-out test views. GenRecon places its chunks with GT
depth points from the non-test renders, where ScanNet++ uses COLMAP points from all DSLR frames.

Needs Blender 4.5+ and the room's original `<house>.json` from
[3D-FRONT](https://github.com/3D-FRONT-FUTURE/3D-FRONT-ToolBox), which gives the metric scale of the
normalized MIDI GLBs (the renderer fails if the room architecture does not match within 5 mm).

```bash
uv run python scripts/download_rooms.py  # a MIDI room to data/3dfront/pilot
blender -b --python-exit-code 1 -P scripts/render_3dfront.py -- --data data/3dfront/pilot \
    --output data/3dfront/renders/pilot --front-json data/3dfront/original_json/<house>.json
papers/GenRecon/.venv/bin/python scripts/prepare_3dfront.py --renders data/3dfront/renders/pilot --scene pilot
uv run scripts/select_views_scannetpp.py 32 --data data/3dfront
uv run scripts/evaluate_scannetpp.py <pred_dir> --data data/3dfront
```

## Experiments

- [VAE reconstruction](experiments/reconstruct_with_vae/README.md)
- [GenRecon baseline](experiments/genrecon_baseline/README.md)
- [GenRecon setup and inference](papers/GenRecon/README_scene-reconstruction.md)

## Papers

| Name     | Folder          | Upstream |
| -------- | --------------- | -------- |
| GenRecon | papers/GenRecon | https://github.com/kasothaphie/GenRecon |
| VGGT-Omega | papers/VGGT-Omega | https://github.com/facebookresearch/vggt-omega |

See [imported papers and licenses](papers/README.md) for imported commits and licenses.
