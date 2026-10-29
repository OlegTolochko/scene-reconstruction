# ScanNet++

Set `SCANNETPP_TOKEN` in your environment or the repository root `.env`.
Choose `data_root`, scene IDs, and assets in
[download_scannetpp.yml](download_scannetpp.yml), then run from the repository root:

```bash
uv run python data_download/scannetpp/download_scannetpp.py data_download/scannetpp/download_scannetpp.yml
```

The configuration selects DSLR assets for GenRecon. Add `scan_mesh_path`
to download reference geometry for the VAE experiment.
See [GenRecon instructions](../../papers/GenRecon/README_scene-reconstruction.md#scannet-data)
for inference and iPhone frame extraction.
