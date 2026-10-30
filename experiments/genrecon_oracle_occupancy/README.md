# GenRecon with ground-truth occupancy

How good is GenRecon's stage-1 occupancy, and how much would a perfect one help?

GenRecon cuts a scene into overlapping chunks of about 3 m. For each chunk, stage 1 predicts which cells
of a 32³ grid (about 10 cm per cell) contain surface, and stages 2 and 3 generate geometry and texture only
inside those cells. This experiment voxelizes the laser scan on the same grid, compares it with the cells
stage 1 predicted in the baseline runs, and reruns GenRecon with stage 1's cells replaced by the scan's
cells that the input views see (the oracle). Everything else matches the baseline: the input views, chunk
layout, seed and settings.

Requires the baseline runs (`experiments/genrecon_baseline`) and the GenRecon env.

```bash
uv run experiments/genrecon_oracle_occupancy/occupancy.py 32  # scan cells, compared with stage 1's
uv run experiments/genrecon_oracle_occupancy/run.py 32  # outputs/genrecon_oracle_occupancy/<N>views/<scene>/mesh.ply.zst
uv run scripts/evaluate_scannetpp.py outputs/genrecon_oracle_occupancy/32views
uv run experiments/genrecon_oracle_occupancy/fscores.py outputs/genrecon_oracle_occupancy/32views  # F@2/5/10cm
uv run experiments/genrecon_oracle_occupancy/fscores.py outputs/genrecon_baseline/32views
```

`run.py` runs GenRecon through `inject.py`, which patches `extract_coords_from_occ_logits` to return the
scan's cells. GenRecon itself is not modified. A chunk without scan cells gets one placeholder cell, in its
corner farthest from the scan, because stage 2 fails on an empty chunk.

Viewer: the scan, the baseline and the oracle side by side, textured or coloured by distance to the scan,
with the cells stage 2 got. `bake.py` reruns GenRecon on a few scenes to bake textures, and `viewer.py`
prepares the data.

```bash
uv run experiments/genrecon_oracle_occupancy/bake.py 5f99900f09 cc5237fd77 27dd4da69e
uv run experiments/genrecon_oracle_occupancy/viewer.py 5f99900f09 cc5237fd77 27dd4da69e
python -m http.server -b 127.0.0.1 8765  # from the repository root
# http://127.0.0.1:8765/experiments/genrecon_oracle_occupancy/viewer/index.html#5f99900f09
```
