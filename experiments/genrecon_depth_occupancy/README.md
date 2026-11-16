# GenRecon with stage 1's cells edited with depth

Everything matches the baseline up to the cells handed to stage 2: the input views, cameras, chunk layout, seed
and stage 1. `cells.py` then replaces stage 1's cells with

- the surface cells of the depth: cells that hold depth points from enough views
- stage 1's cells outside every input image, so generation stays in place there.

All other stage 1 cells are deleted, including cells hidden behind a surface inside an image. The sources:

- `scan`: depth rendered from the ground truth scan at the true cameras, with one view per surface cell.
- `colmap`: VGGT-Omega's depth (`experiments/vggt_omega_depth`) with one scale per view, the median of
  COLMAP depth / VGGT-Omega depth over the COLMAP points the view observes, without the pixels at depth edges, with two views per surface cell.
  The COLMAP model comes from all images of the scene, as in the baseline, so a pipeline with only the N input
  views would not have it.
- `depth_only`: the same depth as `colmap` with one view per surface cell and no stage 1 cells, so the grid comes
  from depth alone. Stage 1 still runs, so the random state stays the same.
- `colmap_solv3r` and `depth_only_solv3r`: the same as `colmap` and `depth_only` with Solv3R's depth
  (`experiments/solv3r_depth`) instead of VGGT-Omega's. Solv3R's depth is 504×336 instead of 624×416, so each
  source projects at its own depth's resolution.
- `carve_scan`, `carve` and `carve_solv3r`: stage 1's cells minus the cells the depth shows to be empty (a cell goes
  when its centre lies more than 1.5 cells in front of the depth in at least one view for `carve_scan`, two views
  otherwise). Nothing is added, and cells behind surfaces or outside every image stay. The depth is the scan's,
  VGGT-Omega's and Solv3R's, scaled as for `colmap`.

Requires the baseline runs, the scan cells from the oracle experiment's `occupancy.py`, the GenRecon env, and for
`colmap` the output of `experiments/vggt_omega_depth/infer.py` (for the `_solv3r` sources,
`experiments/solv3r_depth/infer.py`).

```bash
uv run experiments/genrecon_depth_occupancy/cells.py scan 32  # outputs/genrecon_depth_occupancy/scan/cells/<N>views/<scene>.npz
uv run experiments/genrecon_depth_occupancy/run.py scan 32  # outputs/genrecon_depth_occupancy/scan/<N>views/<scene>/mesh.ply.zst
uv run scripts/evaluate_scannetpp.py outputs/genrecon_depth_occupancy/scan/32views
uv run experiments/genrecon_oracle_occupancy/fscores.py outputs/genrecon_depth_occupancy/scan/32views  # F@2/5/10cm
```

The same commands with `colmap`, `depth_only`, `colmap_solv3r`, `depth_only_solv3r`, `carve_scan`, `carve` or `carve_solv3r` run the other sources.
