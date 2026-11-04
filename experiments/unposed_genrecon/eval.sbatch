#!/bin/bash
# VGGT-Omega + GenRecon on the 50 ScanNet++ eval scenes, then the metrics.
#SBATCH --job-name=unposed_eval
#SBATCH --partition=rtx_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/rhome/%u/scene-reconstruction/logs/unposed_eval_%j.out

set -e
cd ~/scene-reconstruction
VIEWS=${VIEWS:-32}
# MATCHED=1 sbatch ... runs GenRecon with the baseline's settings into outputs/unposed_genrecon_matched/
if [ -n "$MATCHED" ]; then FLAG=--matched; OUT=outputs/unposed_genrecon_matched; else FLAG=; OUT=outputs/unposed_genrecon; fi
uv run scripts/select_views_scannetpp.py "$VIEWS"
uv run experiments/unposed_genrecon/run_eval.py "$VIEWS" $FLAG
uv run scripts/evaluate_scannetpp.py "$OUT/${VIEWS}views"
