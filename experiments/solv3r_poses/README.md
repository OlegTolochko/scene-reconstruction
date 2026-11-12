# Solv3R's camera poses, next to VGGT-Omega's

How accurate are Solv3R's camera poses on the ScanNet++ eval scenes, compared with VGGT-Omega's?

Solv3R gets the same views as VGGT-Omega in `experiments/unposed_genrecon/run_eval.py`. Scenes that run out of
GPU memory are skipped and listed in `outputs/solv3r_poses/<N>views/oom.txt`. The metrics are those of
`experiments/camera_poses/eval_cameras.py` (`pose_metrics`). Solv3R is scored twice: with cameras fitted to its
predicted rays (the paper's protocol, also what `unposed_genrecon` uses) and with its camera head.

Requires the VGGT-Omega poses (`outputs/unposed_genrecon/<N>views/<scene>/vggt.npz`, from
`experiments/unposed_genrecon/run_eval.py`) and the Solv3R env (`papers/Solv3R/README_scene-reconstruction.md`).

Cluster:

```bash
mkdir -p logs && sbatch experiments/solv3r_poses/poses.sbatch      # VIEWS=8 sbatch ... for other view counts
```

Local:

```bash
papers/Solv3R/.venv/bin/python experiments/solv3r_poses/infer.py 32   # outputs/solv3r_poses/32views/<scene>.npz
uv run experiments/solv3r_poses/compare.py 32                       # outputs/solv3r_poses/32views/comparison.csv
```
