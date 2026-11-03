#!/bin/bash
# VGGT-Omega + GenRecon on the 50 ScanNet++ eval scenes, then the metrics.
# Run from the repo root:  mkdir -p logs && sbatch experiments/unposed_genrecon/eval.sbatch
# Jobs stop after 6 h. Finished scenes are skipped, so resubmit until the log ends with "mean over 50 scenes".
#SBATCH --job-name=unposed_eval
#SBATCH --partition=rtx_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/unposed_eval_%j.out

set -e
cd "$SLURM_SUBMIT_DIR"
VIEWS=${VIEWS:-32}
uv run scripts/select_views_scannetpp.py "$VIEWS"
uv run experiments/unposed_genrecon/run_eval.py "$VIEWS"
uv run scripts/evaluate_scannetpp.py "outputs/unposed_genrecon/${VIEWS}views"
