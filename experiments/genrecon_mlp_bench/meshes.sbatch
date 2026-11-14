#!/bin/bash
#SBATCH --job-name=mlp-bench-meshes
#SBATCH --partition=rtx_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=mlp-bench-meshes-%j.log
#
# ScanNet++ meshes of a finished run of the bench, their metrics, and the metrics' means in the run's W&B
# summary (scannetpp_<N>views/...). From the repo root:
#
#   sbatch experiments/genrecon_mlp_bench/meshes.sbatch my_variant [VIEWS] [SEED]      # VIEWS defaults to 32
#   sbatch experiments/genrecon_mlp_bench/meshes.sbatch genrecon                       # the finetuned baseline
#
# Needs the unposed Solv3R run of those views (experiments/unposed_genrecon/run_eval.py VIEWS --model solv3r),
# whose meshes in outputs/unposed_genrecon_solv3r/ are released GenRecon's, to compare with.
# Scenes that already have a mesh are skipped, so resubmitting continues after the 6 hours.
set -euo pipefail

VARIANT=${1:?the variant whose run to reconstruct with}; VIEWS=${2:-32}; SEED=${3:-0}
REPO=$(pwd)
HERE=$REPO/experiments/genrecon_mlp_bench
[ -f "$HERE/settings.sh" ] || { echo "submit from the repo root: $HERE/settings.sh not found"; exit 1; }
# shellcheck source=settings.sh
source "$HERE/settings.sh"

RUN=$VARIANT
[ "$SEED" = 0 ] || RUN=${VARIANT}_s$SEED
RUN_DIR=$(realpath -m "$OUT_ROOT/$BENCH/$RUN")
[ -f "$RUN_DIR/DONE" ] || { echo "$RUN_DIR is not a finished run: train it first (run.sbatch $VARIANT)"; exit 1; }

status=0
uv run "$HERE/run_scannetpp.py" "$VIEWS" --run "$RUN_DIR" || status=$?   # 1: some scenes failed
uv run scripts/evaluate_scannetpp.py "$RUN_DIR/scannetpp/${VIEWS}views"
"$REPO/papers/GenRecon/.venv/bin/python" "$HERE/log_meshes.py" "$RUN_DIR" "$VIEWS"
exit $status
