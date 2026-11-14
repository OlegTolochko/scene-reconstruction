#!/bin/bash
#SBATCH --job-name=mlp-bench
#SBATCH --partition=rtx_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=mlp-bench-%j.log
#
# Trains one aggregator variant (variants.py) and logs it to Weights & Biases. From the repo root:
#
#   sbatch experiments/genrecon_mlp_bench/run.sbatch my_variant        # the variant, after the baseline if needed
#   sbatch experiments/genrecon_mlp_bench/run.sbatch                   # the baseline alone
#   sbatch experiments/genrecon_mlp_bench/run.sbatch genrecon 1        # ... with another training seed
#
# The baseline is the variant `genrecon`: GenRecon's unchanged aggregator, finetuned on the same data. A bench
# has one. A job that finds it unfinished, and not being trained by another job, trains it first. Every run
# logs its validation loss relative to it (vs_genrecon/val_mse_percent).
# Resubmitting a command resumes its run from the last checkpoint; a finished run is skipped.
# The data paths, the bench name and the W&B project are in settings.sh.
set -euo pipefail

VARIANT=${1:-genrecon}; SEED=${2:-0}
REPO=$(pwd)
HERE=$REPO/experiments/genrecon_mlp_bench
PY=$REPO/papers/GenRecon/.venv/bin/python
[ -f "$HERE/settings.sh" ] || { echo "submit from the repo root: $HERE/settings.sh not found"; exit 1; }
# shellcheck source=settings.sh
source "$HERE/settings.sh"
"$PY" -c "import wandb" 2>/dev/null || {
    echo "wandb is missing in GenRecon's env. Once, on the login node:"
    echo "  uv pip install --python $PY wandb && $REPO/papers/GenRecon/.venv/bin/wandb login"; exit 1; }
"$PY" "$HERE/variants.py" "$VARIANT"   # a wrong name or a broken variant stops the job here, before any training

OUT=$(realpath -m "$OUT_ROOT/$BENCH")
mkdir -p "$OUT"
(
    flock 9
    if [ ! -f "$OUT/config.json" ]; then
        case "$TRAIN $VAL" in *"/PATH/TO/"*) echo "set TRAIN and VAL in $HERE/settings.sh"; exit 1 ;; esac
        # shellcheck disable=SC2086  # TRAIN and VAL may hold several roots
        uv run "$HERE/setup_run.py" --out "$OUT" --train $TRAIN --val $VAL --cues "$CUES"
    fi
) 9>"$OUT/.setup.lock"
echo "bench $OUT on $(hostname), job ${SLURM_JOB_ID:-none}"

export ATTN_BACKEND=flash_attn SPARSE_ATTN_BACKEND=flash_attn PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# train VARIANT SEED: 0 once the run is finished, 3 if another job holds it, otherwise the trainer's status.
train() {
    local run=$1
    [ "$2" = 0 ] || run=${1}_s$2
    if [ -f "$OUT/$run/DONE" ]; then echo "$run: already finished"; return 0; fi
    mkdir -p "$OUT/$run"
    exec 8>"$OUT/$run/.lock"
    if ! flock -n 8; then echo "$run: another job is training it"; return 3; fi
    echo "=== $(date +%T) training $run"
    local status=0
    (cd "$REPO/papers/GenRecon" && "$PY" "$HERE/train.py" --variant "$1" --seed "$2" \
        --config "$OUT/config.json" --output_dir "$OUT/$run" \
        --data_dir "$(cat "$OUT/data_dir.json")" --val_data_dir "$(cat "$OUT/val_data_dir.json")" \
        --auto_retry 0) || status=$?
    flock -u 8
    return $status
}

if [ "$VARIANT" != genrecon ] || [ "$SEED" != 0 ]; then
    status=0
    train genrecon 0 || status=$?   # the baseline first, unless it is finished or another job has it
    [ $status = 0 ] || [ $status = 3 ] || { echo "the baseline failed (status $status)"; exit $status; }
fi
train "$VARIANT" "$SEED"
