# scene-reconstruction

## Layout

```text
papers/       vendored paper code, each with its own environment
experiments/  one subfolder per experiment
scripts/      dataset download and preprocessing
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

See [dataset download and preprocessing](scripts/README.md) for ScanNet++
configuration and 3D-FRONT preparation.

## Experiments

- [VAE reconstruction](experiments/reconstruct_with_vae/README.md)
- [GenRecon setup and inference](papers/GenRecon/README_scene-reconstruction.md)

## Papers

| Name     | Folder          | Upstream |
| -------- | --------------- | -------- |
| GenRecon | papers/GenRecon | https://github.com/kasothaphie/GenRecon |
| VGGT-Omega | papers/VGGT-Omega | https://github.com/facebookresearch/vggt-omega |

See [imported papers and licenses](papers/README.md) for imported commits and licenses.
