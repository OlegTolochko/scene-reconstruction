# GenRecon baseline

GenRecon on the ScanNet++ evaluation. T3lescope
([arXiv 2610.03308](https://arxiv.org/abs/2610.03308), Tables 3 and 4) reports GenRecon on the same
50 scenes with the same view selection.

Requires the GenRecon env and checkpoints (`papers/GenRecon/README_scene-reconstruction.md`) and `zstd`.

```bash
uv run scripts/select_views_scannetpp.py 32
uv run experiments/genrecon_baseline/run.py 32 # outputs/genrecon_baseline/<N>views/<scene>/mesh.ply.zst
uv run scripts/evaluate_scannetpp.py outputs/genrecon_baseline/32views
```