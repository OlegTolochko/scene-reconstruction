#!/bin/bash
#SBATCH --job-name=mlp-bench
#SBATCH --partition=rtx_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=experiments/genrecon_mlp_bench/mlp-bench-%j.log
#
# Finetunes one aggregator variant (variants.py) and logs it to Weights & Biases: its validation loss and the
# occupancy IoU of its sparse structures. With --mesh it then also reconstructs the bench's validation rooms with
# it (mesh.py) and uploads the meshes to the same W&B run. A job does only what its command names. From the repo
# root:
#
#   sbatch experiments/genrecon_mlp_bench/run.sbatch                          # the baseline of a bench
#   sbatch experiments/genrecon_mlp_bench/run.sbatch depth_cue_only_x10       # that variant alone
#   sbatch experiments/genrecon_mlp_bench/run.sbatch --mesh depth_cue_only_x10  # ... and its meshes
#
# Without a variant, the job trains the baseline (the variant `genrecon`: GenRecon's unchanged aggregator
# finetuned like every variant). With --mesh it first makes the other references of the bench: the true rooms as
# meshes (ground_truth.py; the W&B run ground_truth) and the meshes of released GenRecon, without any finetuning
# (the W&B run genrecon_released). Meshes are off by default: they take long and the curves do not need them.
set -euo pipefail

# One bench = one set of rooms and settings = one baseline. Its runs go to $OUT_ROOT/$BENCH/<run>/ and form one
# group on W&B. A new name starts a new bench, with config.json as it is at that moment and its own baseline.
BENCH=${BENCH:-2026-11-14_overfit10}

# Rooms per data root: ROOT:@split (a split of rooms.csv). The 10 rooms are spread over three prepared 
# roots (3 in SAGE10k_cues, 4 in _b, 3 in _c); each root contributes the rooms of the split that it has.
DATA=${DATA:-/rhome/pr26rb06/scene-reconstruction/data}
TRAIN=${TRAIN:-$DATA/SAGE10k_cues:@overfit $DATA/SAGE10k_cues_b:@overfit $DATA/SAGE10k_cues_c:@overfit}
VAL=${VAL:-$TRAIN}
CUES=${CUES:-depth_cues_solv3r}   # the cue folder inside each root; depth_cues_true is the true depth

OUT_ROOT=${OUT_ROOT:-outputs/genrecon_mlp_bench}

# W&B: the project, under the account that is logged in on the cluster.
export WANDB_PROJECT=${WANDB_PROJECT:-genrecon-mlp-bench}

# Blender, which writes the true rooms as meshes (ground_truth.py): the installation the rooms were prepared with.
BLENDER=${BLENDER:-/cluster/vc17/pr26rb06/blender/blender-5.0.1-linux-x64/blender}

# ---- the job ----

MESH=0; ARGS=()
for arg in "$@"; do
    case $arg in
        --mesh) MESH=1 ;;
        -*) echo "unknown option $arg (known: --mesh)"; exit 1 ;;
        *) ARGS+=("$arg") ;;
    esac
done
set -- "${ARGS[@]+"${ARGS[@]}"}"
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

# meshes [RUN]: the meshes of the bench's validation rooms, on W&B. With RUN: reconstructed with that run's
# aggregator. Without: released GenRecon. Rooms that have a mesh are skipped.
meshes() {
    local name=${1:-genrecon_released}
    if [ -f "$OUT/$name/meshes/UPLOADED" ]; then echo "$name: meshes already uploaded"; return 0; fi
    mkdir -p "$OUT/$name"
    exec 7>"$OUT/$name/.meshes.lock"
    if ! flock -n 7; then echo "$name: another job is making its meshes"; return 0; fi
    echo "=== $(date +%T) meshes of $name"
    local status=0
    (cd "$REPO/papers/GenRecon" && "$PY" "$HERE/mesh.py" --bench "$OUT" ${1:+--run "$1"}) || status=$?
    flock -u 7
    return $status
}

# ground_truth: the true rooms of the bench as meshes, on W&B (the run ground_truth). Needs Blender, no GPU.
ground_truth() {
    if [ -f "$OUT/ground_truth/meshes/UPLOADED" ]; then echo "ground_truth: meshes already uploaded"; return 0; fi
    mkdir -p "$OUT/ground_truth"
    exec 6>"$OUT/ground_truth/.meshes.lock"
    if ! flock -n 6; then echo "ground_truth: another job is making its meshes"; return 0; fi
    echo "=== $(date +%T) meshes of ground_truth"
    local status=0
    "$PY" "$HERE/ground_truth.py" --bench "$OUT" --blender "$BLENDER" || status=$?
    flock -u 6
    return $status
}

failed=0
if [ $# -eq 0 ] && [ "$MESH" = 1 ]; then   # the mesh references of a bench: only without a variant, with --mesh
    ground_truth || { failed=1; echo "ground_truth failed: going on"; }       # the true rooms
    meshes || { failed=1; echo "genrecon_released failed: going on"; }        # released GenRecon
fi
train "$VARIANT" "$SEED"                      # (without a variant: the baseline) finetuning, with its curves
if [ "$MESH" = 1 ]; then
    meshes "$(run_name "$VARIANT" "$SEED")"   # and, with --mesh, its meshes
fi
exit $failed
