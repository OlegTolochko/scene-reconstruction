#!/bin/bash
# Prepares every room whose zip is in ROOT/zips for depth-cue finetuning, on one GPU node:
#   GenRecon's toolkit (unzip, build, chunks, renders, sparse-structure and shape latents, metadata),
#   the true depth of every render (export_room_mesh.py, true_depth.py), Solv3R on the renders
#   (unposed_genrecon/infer_solv3r.py), Solv3R's depth cues (prepare_cues.py) and check_depth.py.
# Every step skips rooms it has already done, so a rerun continues where a job stopped.
#
#   sbatch --partition=rtx_5090 --gres=gpu:1 --cpus-per-task=8 --mem=48G --time=06:00:00 \
#       --output=LOG experiments/genrecon_depth_cue_finetuning/rooms/prepare_rooms.sh ROOT
# from the repo root.
set -euo pipefail

ROOT=$(realpath "$1")
REPO=$(pwd)
HERE=$REPO/experiments/genrecon_depth_cue_finetuning/rooms
GR=$REPO/papers/GenRecon
PY=$GR/.venv/bin/python
SOLV3R_PY=$REPO/papers/Solv3R/.venv/bin/python
BLENDER_DIR=${BLENDER_DIR:-/cluster/vc17/pr26rb06/blender}
BLENDER=$BLENDER_DIR/blender-5.0.1-linux-x64/blender
export BLENDER_DIR
step() { echo; echo "=== $(date +%T) $*"; }

step "node $(hostname), $(ls "$ROOT"/zips/*.zip | wc -l) zips in $ROOT/zips"
missing=$(ldd "$BLENDER" | grep "not found" || true)
if [ -n "$missing" ]; then echo "Blender is missing libraries on this node:"; echo "$missing"; exit 1; fi
"$BLENDER" --version | head -1

cd "$GR"
step "unzip";            $PY data_toolkit_scenes/unzip.py --root "$ROOT"
step "build rooms";      $PY "$HERE/toolkit.py" build_rooms --root "$ROOT" --jobs 4
step "chunks";           $PY "$HERE/toolkit.py" create_chunks --root "$ROOT" --crop_size 2.7 3.0 --num_crops 3 --seed 0 --jobs 4
step "chunk metadata";   $PY data_toolkit_scenes/create_metadata_chunks.py --root "$ROOT"
step "render";           $PY "$HERE/toolkit.py" render_rooms --root "$ROOT" --num_views_per_m2 1.5
                         $PY "$HERE/toolkit.py" render_rooms --root "$ROOT" --num_views_per_m2 1.5 --finalize_only
step "latents (sparse structure at 64, shape at 1024)"
                         $PY data_toolkit_scenes/get_slats.py --root "$ROOT" --resolution_ovoxel 1024 --not_encode_tex
step "metadata";         $PY data_toolkit_scenes/build_metadata.py SAGE --root "$ROOT"

step "room meshes"
for blend in "$ROOT"/rooms_processed/*.blend; do
    room=$(basename "$blend" .blend)
    out=$ROOT/rooms_processed/${room}_mesh.npz
    [ -f "$out" ] || [ ! -f "$ROOT/renders_room/$room/transforms.json" ] && continue
    "$BLENDER" -b "$blend" --python-exit-code 1 -P "$HERE/export_room_mesh.py" -- "$out" 2>&1 \
        | grep "^wrote\|Error" || { echo "$room: mesh export failed"; exit 1; }
    [ -f "$out" ] || { echo "$room: mesh export failed"; exit 1; }
done

cd "$REPO"
step "true depth"
todo=()
for t in "$ROOT"/renders_room/*/transforms.json; do
    room=$(basename "$(dirname "$t")")
    [ -f "$ROOT/depth_cues_true/$room/cameras.npz" ] || todo+=("$room")
done
[ ${#todo[@]} -eq 0 ] || uv run experiments/genrecon_depth_cue_finetuning/rooms/true_depth.py \
    --base "$ROOT" --out "$ROOT/depth_cues_true" "${todo[@]}"

step "Solv3R"
mkdir -p "$ROOT/solv3r"
for t in "$ROOT"/renders_room/*/transforms.json; do
    dir=$(dirname "$t"); room=$(basename "$dir")
    out=$ROOT/solv3r/${room}_solv3r.npz
    [ -f "$out" ] && continue
    n=$(ls "$dir"/*.png | wc -l)
    echo "$room: $n views"
    $SOLV3R_PY experiments/unposed_genrecon/infer_solv3r.py "$dir" --num-images "$n" --output "$out" 2>&1 | tail -2
done

step "Solv3R cues"
todo=()
for f in "$ROOT"/solv3r/*_solv3r.npz; do
    room=$(basename "$f" _solv3r.npz)
    [ -f "$ROOT/depth_cues_solv3r/$room/cameras.npz" ] || todo+=("$f")
done
[ ${#todo[@]} -eq 0 ] || uv run experiments/genrecon_depth_cue_finetuning/prepare_cues.py \
    --base "$ROOT" --out "$ROOT/depth_cues_solv3r" "${todo[@]}"

step "depth check"
uv run experiments/genrecon_depth_cue_finetuning/rooms/check_depth.py --base "$ROOT" \
    --true "$ROOT/depth_cues_true" --solv3r "$ROOT/depth_cues_solv3r" --out "$ROOT/depth_check"
step "done"
