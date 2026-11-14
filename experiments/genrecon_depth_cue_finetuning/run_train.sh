#!/bin/bash
# One arm (cues or control) of a finetuning run, as a SLURM job. Resubmitting the same command resumes from
# the last checkpoint (train.py's --load_dir defaults to --output_dir), so a chain of these covers runs longer
# than the 6-hour limit. The first job of either arm writes the run's config with make_run.py.
#
#   sbatch ... experiments/genrecon_depth_cue_finetuning/run_train.sh OUT cues|control -- MAKE_RUN_ARGS
# from the repo root. OUT holds config.json, data_dir.json, ... and the arms' output folders cues/ and control/.
set -euo pipefail

OUT=$(realpath -m "$1"); ARM=$2; shift 2
[ "${1:-}" = "--" ] && shift
REPO=$(pwd)
HERE=$REPO/experiments/genrecon_depth_cue_finetuning
mkdir -p "$OUT"
(
    flock 9
    [ -f "$OUT/config.json" ] || uv run "$HERE/make_run.py" --out "$OUT" "$@"
) 9>"$OUT/.setup.lock"
cat "$OUT/rooms.json"; echo

case $ARM in
    cues) extra=() ;;
    control) extra=(--control) ;;
    *) echo "arm must be cues or control"; exit 1 ;;
esac
cd "$REPO/papers/GenRecon"
export ATTN_BACKEND=flash_attn SPARSE_ATTN_BACKEND=flash_attn PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python "$HERE/train.py" --config "$OUT/config.json" --output_dir "$OUT/$ARM" \
    --data_dir "$(cat "$OUT/data_dir.json")" --val_data_dir "$(cat "$OUT/val_data_dir.json")" \
    --auto_retry 0 "${extra[@]}"
