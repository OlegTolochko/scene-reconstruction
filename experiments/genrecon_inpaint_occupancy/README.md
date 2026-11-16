# GenRecon with stage 1 inpainted around the depth

Does stage 1 improve if its DiT only generates what the input views' depth does not know?

`genrecon_depth_occupancy` edits stage 1's cells after the fact. Here the depth constrains stage 1 itself, without
training. `cells.py` labels every cell of the sparse-structure grid (32 cells per chunk side, about 10 cm each):

- occupied: holds depth points from enough views
- free: enough views see past it, as in the carving sources
- unknown: everything else, i.e. behind surfaces, outside every image, or at depth edges.

`inject.py` encodes the occupied cells with TRELLIS's sparse-structure encoder, the partner of the decoder
GenRecon uses. It downloads `microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16` from Hugging Face on first
use. It then samples stage 1 as usual, RePaint-style, except that after every step the pinned latent voxels are reset
to the encoding, noised to the next time. GenRecon's `FlowEulerSampler._apply_inpainting_constraint` does the same
for a single object, but GenRecon's multi-chunk sampler never calls it.

- A latent voxel covers 2 cells per axis. It is pinned when the depth knows all of its cells. The pinned region is
  then shrunk by one latent voxel, because the encoder reads unknown cells as empty and that leaks into the
  neighbouring voxels.
- The encoding is computed once on the scene's whole canvas, as `joint_decode_sparse_structure` decodes it, so chunks
  agree where they overlap.
- Everything else matches the baseline: the input views, cameras, chunk layout, seed, and stages 2 and 3.

| Variant | Depth | Cells stage 2 gets |
| --- | --- | --- |
| `scan` | the scan's, one view per cell: the most inpainting can get | decoded from the inpainted latent |
| `solv3r` | Solv3R's, scaled per view with COLMAP as in `genrecon_depth_occupancy`, two views per cell | decoded from the inpainted latent |
| `scan_fix`, `solv3r_fix` | the same | plus the occupied cells, minus the free cells |

The `_fix` variants show how far the decoded cells drift from what the depth saw. As in `genrecon_depth_occupancy`,
the COLMAP model behind Solv3R's scale comes from all images of the scene, which a pipeline with only the N input
views would not have.

Requires the baseline runs, the scan cells from the oracle experiment's `occupancy.py`, the GenRecon env, and for
`solv3r` the output of `experiments/solv3r_depth/infer.py`.

```bash
uv run experiments/genrecon_inpaint_occupancy/cells.py solv3r 32  # outputs/genrecon_inpaint_occupancy/solv3r/cells/<N>views/<scene>.npz
uv run experiments/genrecon_inpaint_occupancy/run.py solv3r 32  # outputs/genrecon_inpaint_occupancy/solv3r/<N>views/<scene>/mesh.ply.zst
uv run scripts/evaluate_scannetpp.py outputs/genrecon_inpaint_occupancy/solv3r/32views
uv run experiments/genrecon_oracle_occupancy/fscores.py outputs/genrecon_inpaint_occupancy/solv3r/32views  # F@2/5/10cm
```

On the cluster, `run.sbatch` does everything for one variant: the baseline run and Solv3R's depth where they are
missing, then the four steps above, and the evaluation of the baseline next to the variant. Submitting it again
continues a job that hit the time limit:

```bash
mkdir -p logs && SCENES=5f99900f09 sbatch experiments/genrecon_inpaint_occupancy/run.sbatch solv3r  # one scene first
sbatch experiments/genrecon_inpaint_occupancy/run.sbatch solv3r  # every evaluation scene; or scan, scan_fix, solv3r_fix
```

The same commands run `scan`. The `_fix` variants use their source's cells (`run.py solv3r_fix 32` after
`cells.py solv3r 32`). Compare with the baseline, the oracle, `carve_solv3r` and `colmap_solv3r`.
