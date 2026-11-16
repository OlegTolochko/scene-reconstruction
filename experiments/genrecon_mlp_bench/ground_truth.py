"""The true rooms of a bench on Weights & Biases, to compare GenRecon's meshes with.

For every validation room of the bench, Blender writes the room that GenRecon's views were rendered from
(<root>/rooms_processed/<room>.blend) as a textured GLB (export_room_glb.py): in the frame of mesh.py's GLBs and,
like them, without what is above CEILING_CUT. Textures are shrunk to TEXTURE pixels.

Writes <bench>/ground_truth/meshes/<room>/mesh.glb and logs the meshes to the W&B run `ground_truth` of the
bench's group as mesh/<room>, at step 0. Rooms that already have a mesh.glb are skipped; meshes/UPLOADED marks
the upload. Needs no GPU.

In GenRecon's env (run.sbatch does this in the job without arguments):
    python experiments/genrecon_mlp_bench/ground_truth.py --bench OUT --blender BLENDER
"""

import argparse
import json
import subprocess
from pathlib import Path

from bench import CEILING_CUT, STEP, start_wandb

HERE = Path(__file__).resolve().parent
RUN = "ground_truth"  # the run folder (and W&B run) of the true rooms
TEXTURE = 1024  # pixels: the longer side of a texture in the uploaded mesh


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", type=Path, required=True, help="the bench folder (run.sbatch's OUT)")
    ap.add_argument("--blender", required=True, help="the Blender executable (run.sbatch's BLENDER)")
    args = ap.parse_args()

    bench = args.bench.resolve()
    out = bench / RUN
    if (out / "meshes" / "UPLOADED").exists():
        print(f"{RUN}: meshes already uploaded")
        return
    info = json.loads((bench / "rooms.json").read_text())
    rooms = [(Path(base), room) for base, names in info["val"].items() for room in names]
    glb = {room: out / "meshes" / room / "mesh.glb" for _, room in rooms}
    for base, room in rooms:
        if glb[room].exists():
            continue
        blend = base / "rooms_processed" / f"{room}.blend"
        if not blend.exists():
            raise SystemExit(f"{blend}: not found")
        glb[room].parent.mkdir(parents=True, exist_ok=True)
        tmp = glb[room].with_suffix(".tmp.glb")
        done = subprocess.run([args.blender, "-b", str(blend), "--python-exit-code", "1", "-P",
                               str(HERE / "export_room_glb.py"), "--", str(tmp), str(CEILING_CUT), str(TEXTURE)],
                              capture_output=True, text=True)
        if done.returncode or not tmp.exists():
            raise SystemExit(f"{room}: Blender failed\n{(done.stdout + done.stderr)[-3000:]}")
        tmp.rename(glb[room])  # a mesh.glb is always a complete one
        print(f"{room}: {glb[room]} ({glb[room].stat().st_size / 1e6:.0f} MB)", flush=True)

    import wandb

    run = start_wandb(out, {"variant": "ground truth"})
    run.log({**{f"mesh/{room}": wandb.Object3D(str(path)) for room, path in glb.items()}, STEP: 0})
    run.finish()
    (out / "meshes" / "UPLOADED").write_text("")
    print(f"{RUN}: {len(glb)} meshes logged to W&B (mesh/<room>, step 0)")


if __name__ == "__main__":
    main()
