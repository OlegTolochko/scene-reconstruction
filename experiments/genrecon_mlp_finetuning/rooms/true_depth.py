"""True depth for every render of SAGE rooms, in the depth-cue format train.py reads (the oracle cue).

Ray-casts the room mesh of export_room_mesh.py (rooms_processed/<room>_mesh.npz) from each camera of
renders_room/<room>/transforms.json with Open3D, as scripts/evaluate_scannetpp.py renders the ScanNet++ scan.

Writes <out>/<room>/cameras.npz (K [F, 3, 3] normalised, c2w [F, 4, 4] camera-to-room in Blender's convention:
the true cameras), <out>/<room>/<i:04d>.npz (depth [H, W] z-depth in room units, 0 where a ray hits nothing;
conf = 1; float16) and <out>/metadata.csv (the rooms' chunks, depth_cues = True), as prepare_cues.py does for
Solv3R.

From the repo root:
    uv run experiments/genrecon_mlp_finetuning/rooms/true_depth.py --base ROOT --out ROOT/depth_cues_true
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from evaluate_scannetpp import raycasting_scene, render_depth  # noqa: E402

FLIP = np.diag([1.0, -1.0, -1.0, 1.0])  # Blender <-> OpenCV camera axes


def room_depth(base: Path, room: str, out: Path, size: int) -> dict:
    data = np.load(base / "rooms_processed" / f"{room}_mesh.npz")
    scene = raycasting_scene(type("Mesh", (), {"vertices": data["vertices"], "faces": data["faces"]}))
    frames = json.loads((base / "renders_room" / room / "transforms.json").read_text())["frames"]
    W0, H0 = Image.open(base / "renders_room" / room / frames[0]["file_path"]).size
    W, H = size, round(size * H0 / W0)
    c2w = np.array([f["transform_matrix"] for f in frames], dtype=np.float64)
    f_norm = 0.5 / np.tan(np.array([f["camera_angle_x"] for f in frames]) / 2)  # focal / image width
    K = np.zeros((len(frames), 3, 3), dtype=np.float32)
    K[:, 0, 0], K[:, 1, 1] = f_norm, f_norm * W0 / H0
    K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = 0.5, 0.5, 1.0

    (out / room).mkdir(parents=True, exist_ok=True)
    np.savez(out / room / "cameras.npz", K=K, c2w=c2w.astype(np.float32))
    hit = []
    for i in range(len(frames)):
        Kpx = K[i].astype(np.float64) * np.array([[W], [H], [1.0]])
        depth = render_depth(scene, Kpx, c2w[i] @ FLIP, W, H)
        depth = np.where(np.isfinite(depth), depth, 0.0)
        hit.append(float((depth > 0).mean()))
        np.savez_compressed(out / room / f"{i:04d}.npz", depth=depth.astype(np.float16),
                            conf=np.ones_like(depth, dtype=np.float16))
    chunks = json.loads((base / "crops" / f"{room}.json").read_text())["chunks"]
    return {"room": room, "frames": len(frames), "hw": [H, W], "hit_min": min(hit), "hit_mean": float(np.mean(hit)),
            "chunks": [f"{room}_{int(c['index']):04d}" for c in chunks]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True, help="SAGE data root")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=512, help="depth map width in pixels")
    ap.add_argument("rooms", nargs="*", help="default: every room with renders and a mesh")
    args = ap.parse_args()

    rooms = args.rooms or sorted(p.parent.name for p in (args.base / "renders_room").glob("*/transforms.json")
                                 if (args.base / "rooms_processed" / f"{p.parent.name}_mesh.npz").is_file())
    args.out.mkdir(parents=True, exist_ok=True)
    meta_path = args.out / "metadata.csv"
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = {r["sha256"]: r["depth_cues"] for r in csv.DictReader(f)}
    for room in rooms:
        info = room_depth(args.base, room, args.out, args.size)
        meta.update({c: "True" for c in info.pop("chunks")})
        print(json.dumps(info), flush=True)
    with open(meta_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sha256", "depth_cues"])
        w.writerows(sorted(meta.items()))


if __name__ == "__main__":
    main()
