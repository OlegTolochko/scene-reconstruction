#!/bin/bash
#SBATCH --job-name=mlp-bench
#SBATCH --partition=rtx_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=mlp-bench-%j.log
#
# Finetunes one aggregator variant (variants.py) and logs it to Weights & Biases. Each job trains only the
# variant it is given. From the repo root:
#
#   sbatch experiments/genrecon_mlp_bench/run.sbatch                   # the baseline: GenRecon's unchanged aggregator
#   sbatch experiments/genrecon_mlp_bench/run.sbatch depth_inputs      # that variant alone
#   sbatch experiments/genrecon_mlp_bench/run.sbatch genrecon 1        # ... with another training seed
#
# The baseline is the variant `genrecon`. A variant logs its validation loss relative to the baseline
# (vs_genrecon/val_mse_percent) once the baseline of the same bench has been trained.
# Every step skips what is done: resubmitting the same command after the 6 hours continues where it stopped.
set -euo pipefail

# ---- settings (any of them can be overridden for one job: sbatch --export=ALL,BENCH=... run.sbatch ...) ----

# One bench = one set of rooms and settings = one baseline. Its runs go to $OUT_ROOT/$BENCH/<run>/ and form one
# group on W&B. A new name starts a new bench, with config.json as it is at that moment and its own baseline.
BENCH=${BENCH:-2026-11-14_overfit10}

# Overfitting: train on the 10 rooms of rooms.csv and validate on the same rooms. The validation pass scores them
# with fixed timesteps and noise, so its curve is the clean overfitting curve (the training loss is noisy).
# Rooms per data root: ROOT:@split (a split of rooms.csv), ROOT:all or ROOT:room,room,... Several roots are
# separated by spaces. Only read when a bench is first set up.
# The 10 rooms are spread over three prepared roots (3 in SAGE10k_cues, 4 in _b, 3 in _c); each root
# contributes the rooms of the split that it has.
DATA=${DATA:-/rhome/pr26rb06/scene-reconstruction/data}
TRAIN=${TRAIN:-$DATA/SAGE10k_cues:@overfit $DATA/SAGE10k_cues_b:@overfit $DATA/SAGE10k_cues_c:@overfit}
VAL=${VAL:-$TRAIN}
CUES=${CUES:-depth_cues_solv3r}   # the cue folder inside each root; depth_cues_true is the true depth

OUT_ROOT=${OUT_ROOT:-outputs/genrecon_mlp_bench}

# W&B: the project, under the account that is logged in on the cluster (set WANDB_ENTITY for a team).
export WANDB_PROJECT=${WANDB_PROJECT:-genrecon-mlp-bench}

# ---- the job ----

VARIANT=${1:-genrecon}; SEED=${2:-0}
REPO=$(pwd)
HERE=$REPO/experiments/genrecon_mlp_bench
PY=$REPO/papers/GenRecon/.venv/bin/python
[ -f "$HERE/train.py" ] || { echo "submit from the repo root: $HERE/train.py not found"; exit 1; }
"$PY" -c "import wandb" 2>/dev/null || {
    echo "wandb is missing in GenRecon's env. Once, on the login node:"
    echo "  uv pip install --python $PY wandb && $REPO/papers/GenRecon/.venv/bin/wandb login"; exit 1; }
"$PY" "$HERE/variants.py" "$VARIANT"   # a wrong name or a broken variant stops the job here, before any training

OUT=$(realpath -m "$OUT_ROOT/$BENCH")
mkdir -p "$OUT"
(
    flock 9
    if [ ! -f "$OUT/config.json" ]; then
        # shellcheck disable=SC2086  # TRAIN and VAL may hold several roots
        uv run "$HERE/setup_run.py" --out "$OUT" --train $TRAIN --val $VAL --cues "$CUES"
    fi
) 9>"$OUT/.setup.lock"
echo "bench $OUT on $(hostname), job ${SLURM_JOB_ID:-none}"

export ATTN_BACKEND=flash_attn SPARSE_ATTN_BACKEND=flash_attn PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_name() { if [ "$2" = 0 ]; then echo "$1"; else echo "${1}_s$2"; fi; }

# train VARIANT SEED: 0 once the run is finished, 3 if another job holds it, otherwise the trainer's status.
train() {
    local run
    run=$(run_name "$1" "$2")
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

train "$VARIANT" "$SEED"
