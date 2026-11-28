# VGGT-Omega's depth against the scan

How accurate is VGGT-Omega's depth at the input views, and how does that change with the number of views?

Each depth map is compared with the depth rendered from the laser scan at the view's true camera.

VGGT-Omega's depth has no absolute scale, so the error depends on the alignment. `evaluate.py` reports three:

- `scene`: one scale per scene, from fitting VGGT-Omega's camera centres to the true ones.
- `view`: one scale per view, fitted to the scan. This leaves only the error of the depth's shape.
- `points`: the depth in 3D with VGGT-Omega's own cameras, as distance to the scan. This combines the error
  of the depth and of the cameras.

Requires the baseline runs (`experiments/genrecon_baseline`) and the VGGT-Omega env.

```bash
papers/VGGT-Omega/.venv/bin/python experiments/vggt_omega_depth/infer.py 8 16 32 64 256  # outputs/vggt_omega_depth/<N>views/<scene>.npz
uv run experiments/vggt_omega_depth/evaluate.py 8 16 32 64 256
```
