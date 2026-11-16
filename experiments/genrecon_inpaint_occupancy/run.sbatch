#!/bin/bash
# GenRecon with stage 1 inpainted around the depth's cells, with everything it needs. From the repo root:
#
#   mkdir -p logs && SCENES=5f99900f09 sbatch experiments/genrecon_inpaint_occupancy/run.sbatch solv3r
#   sbatch experiments/genrecon_inpaint_occupancy/run.sbatch solv3r_fix        # every evaluation scene
#
# The variant is one of scan, solv3r, scan_fix, solv3r_fix (default solv3r). The job runs, for the scenes in SCENES
# (default: every scene of the evaluation inputs):
#   1. the GenRecon baseline, which gives the chunk layout and is the reference
#   2. Solv3R's depth (solv3r variants only)
#   3. the cells, the inpainted runs, and the evaluation of the baseline and the variant
# Every step skips scenes it has already done, so a job that hits the time limit continues where it stopped when
# submitted again.
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
SCENES=${SCENES:-$(ls "data/scannetpp/eval_inputs/${VIEWS}views")}
UV=~/.local/bin/uv
OUT=outputs/genrecon_inpaint_occupancy/$VARIANT/${VIEWS}views

$UV run experiments/genrecon_baseline/run.py "$VIEWS" --scenes $SCENES
if [[ $VARIANT == solv3r* ]]; then
    papers/Solv3R/.venv/bin/python experiments/solv3r_depth/infer.py "$VIEWS" --scenes $SCENES
fi
$UV run experiments/genrecon_inpaint_occupancy/cells.py "${VARIANT%_fix}" "$VIEWS" --scenes $SCENES
$UV run experiments/genrecon_inpaint_occupancy/run.py "$VARIANT" "$VIEWS" --scenes $SCENES

for dir in "outputs/genrecon_baseline/${VIEWS}views" "$OUT"; do
    echo "== $dir"
    $UV run scripts/evaluate_scannetpp.py "$dir"
    $UV run experiments/genrecon_oracle_occupancy/fscores.py "$dir"
done
