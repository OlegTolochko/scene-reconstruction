# ScanNet++ downloads

These files target ScanNet++, including the downloader's optional ScanNet++GS
mode. `scene_release.py` defines dataset asset paths; it is used by the downloader.

From the repository root:

```bash
uv sync
uv run python data_download/scannetpp/download_scannetpp.py data_download/scannetpp/download_scannetpp.yml
```

Set `SCANNETPP_TOKEN` in your environment or the repository root `.env` file.
Choose `data_root`, scene IDs and assets in `download_scannetpp.yml`. The existing
configuration selects the DSLR assets used by GenRecon; downloading reference
meshes for a latent baseline requires selecting those assets as well.

For iPhone frame extraction, use the existing script in
[`scripts/`](../../scripts/extract_scannetpp_iphone_frames.py).
