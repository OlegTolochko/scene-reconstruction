#!/bin/bash
# Solv3R's camera poses on the ScanNet++ eval scenes, then the comparison with VGGT-Omega's.
#SBATCH --job-name=solv3r_poses
#SBATCH --partition=rtx_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/rhome/%u/scene-reconstruction/logs/solv3r_poses_%j.out

set -e
cd ~/scene-reconstruction
VIEWS=${VIEWS:-32}
papers/Solv3R/.venv/bin/python experiments/solv3r_poses/infer.py "$VIEWS"
~/.local/bin/uv run experiments/solv3r_poses/compare.py "$VIEWS"
