# scene-reconstruction

## Layout

```text
papers/       vendored paper code, each with its own environment
experiments/  one subfolder per experiment
scripts/      dataset download and preprocessing
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
uv run python data_download/download_scannetpp.py data_download/download_scannetpp.yml
```

## Run GenRecon

```bash
conda activate genrecon
export ATTN_BACKEND=flash_attn
export SPARSE_ATTN_BACKEND=flash_attn
cd ~/scene-reconstruction/papers/GenRecon

python reconstruct_scene.py \
    --mode Scannet_colmap \
    --path ~/scene-reconstruction/data/scannetpp/data/56a0ec536c \
    --output_path ~/scene-reconstruction/outputs/genrecon/56a0ec536c \
    --ss_ckpt checkpoints/ss/ckpts/sparse_structure.pt \
    --shape_ckpt checkpoints/shape/ckpts/shape_slat.pt \
    --tex_ckpt checkpoints/tex/ckpts/texture_slat.pt \
    --num_imgs_per_scene 32
```


## Papers

| Name     | Folder          | Upstream |
| -------- | --------------- | -------- |
| GenRecon | papers/GenRecon | https://github.com/kasothaphie/GenRecon |
| VGGT-Omega | papers/VGGT-Omega | https://github.com/facebookresearch/vggt-omega |

See `papers/README.md` for imported commits and licenses.
