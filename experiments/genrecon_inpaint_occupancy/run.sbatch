#!/bin/bash
# GenRecon with stage 1 inpainted around the depth's cells: the cells, the runs, then the evaluation. From the repo root:
#
#   mkdir -p logs && sbatch experiments/genrecon_inpaint_occupancy/run.sbatch solv3r
#   sbatch experiments/genrecon_inpaint_occupancy/run.sbatch solv3r_fix
#   SCENES="5f99900f09 cc5237fd77" sbatch experiments/genrecon_inpaint_occupancy/run.sbatch scan
#
# The variant is one of scan, solv3r, scan_fix, solv3r_fix (default solv3r). Scenes that already have cells or a
# mesh are skipped, so a job that hits the time limit continues where it stopped when submitted again. The
# evaluation then covers every scene that has a mesh.
#SBATCH --job-name=inpaint_occupancy
#SBATCH --partition=rtx_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/rhome/%u/scene-reconstruction/logs/inpaint_occupancy_%j.out

set -euo pipefail
cd ~/scene-reconstruction
VARIANT=${1:-solv3r}
VIEWS=${VIEWS:-32}
SCENES=${SCENES:-}
UV=~/.local/bin/uv
SCENE_ARGS=()
[ -n "$SCENES" ] && SCENE_ARGS=(--scenes $SCENES)

$UV run experiments/genrecon_inpaint_occupancy/cells.py "${VARIANT%_fix}" "$VIEWS" ${SCENE_ARGS[@]+"${SCENE_ARGS[@]}"}
$UV run experiments/genrecon_inpaint_occupancy/run.py "$VARIANT" "$VIEWS" ${SCENE_ARGS[@]+"${SCENE_ARGS[@]}"}
$UV run scripts/evaluate_scannetpp.py "outputs/genrecon_inpaint_occupancy/$VARIANT/${VIEWS}views"
$UV run experiments/genrecon_oracle_occupancy/fscores.py "outputs/genrecon_inpaint_occupancy/$VARIANT/${VIEWS}views"
