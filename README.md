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
python data_download/download_scannetpp.py data_download/download_scannetpp.yml
```

## Papers

| Name     | Folder          | Upstream |
| -------- | --------------- | -------- |
| GenRecon | papers/GenRecon | https://github.com/kasothaphie/GenRecon |
| VGGT-Omega | papers/VGGT-Omega | https://github.com/facebookresearch/vggt-omega |

See `papers/README.md` for imported commits and licenses.
