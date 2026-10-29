# Dataset download and preprocessing

Run commands from the repository root. Downloads go under ignored `data/`.

## ScanNet++

Set `SCANNETPP_TOKEN` in your environment or the repository root `.env`.
Choose `data_root`, scene IDs, and assets in
[download_scannetpp.yml](download_scannetpp.yml), then run from the repository root:

```bash
uv run python scripts/download_scannetpp.py scripts/download_scannetpp.yml
```

The configuration selects DSLR assets for GenRecon. Add `scan_mesh_path`
to download reference geometry for the VAE experiment.
See [GenRecon instructions](../papers/GenRecon/README_scene-reconstruction.md#scannet-data)
for inference and iPhone frame extraction.

## 3D-FRONT

Download a rendered room from the processed MIDI release and prepare GenRecon
inputs. Run from the repository root:

```bash
uv run python scripts/download_rooms.py
uv run --extra 3dfront python scripts/prepare_3dfront.py \
    --data data/3dfront/pilot \
    --output data/3dfront/genrecon_pilot \
    --views 8
```

Use downloader flags `--room <house/room> --output <directory>` for another
test room. Completed downloads are reused; preparation needs an empty output
directory. Prepared images and COLMAP files go under `iphone/` in that directory.

Use the existing [GenRecon inference commands](../papers/GenRecon/README_scene-reconstruction.md#running-on-scannet)
with `--mode Scannet_iphone --num_imgs_per_scene 8` and the prepared input path.
Also pass `--min_points_per_chunk 30 --skip_point_cleaning --proj_batch_voxels 2048`
for this sparse support cloud.

Inputs use supplied cameras and ground-truth depth for chunk placement.
Coordinates retain dataset scale; physical metres are unverified.
