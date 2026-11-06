# Solv3R's depth against the scan, next to VGGT-Omega's

How does Solv3R's depth at the input views compare with VGGT-Omega's?

Solv3R runs on the same inputs as `experiments/vggt_omega_depth`. 
Its cameras are fitted to its predicted rays, as in the paper's benchmark. 
The scoring is `evaluate()` of `experiments/vggt_omega_depth/evaluate.py`, so the metrics and the three alignments (scene,
view, points) are the same as there. VGGT-Omega's scores are read from `outputs/vggt_omega_depth`.

Requires the VGGT-Omega depth results (`experiments/vggt_omega_depth`) and the Solv3R env
(`papers/Solv3R/README_scene-reconstruction.md`).

```bash
papers/Solv3R/.venv/bin/python experiments/solv3r_depth/infer.py 8 16 32 64 256  # outputs/solv3r_depth/<N>views/<scene>.npz
uv run experiments/solv3r_depth/evaluate.py 8 16 32 64 256  # outputs/solv3r_depth/summary.csv
```