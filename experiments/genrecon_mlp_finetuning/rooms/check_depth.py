"""Checks the true depth of true_depth.py against the render and Solv3R's cue depth of prepare_cues.py.

Per room: the share of pixels with a true depth, and Solv3R's error against the true depth (both in room
units, Solv3R aligned by prepare_cues.py's similarity transform): median relative error and the share within
5% and 10%. A wrong camera convention or frame in true_depth.py shows up as a large error everywhere. Writes
<out>/<room>.png per room: render | true depth | Solv3R depth | relative error, for three frames.

From the repo root:
    uv run experiments/genrecon_mlp_finetuning/rooms/check_depth.py --base ROOT \
        --true ROOT/depth_cues_true --solv3r ROOT/depth_cues_solv3r --out ROOT/depth_check
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def gray(x: np.ndarray, lo: float, hi: float) -> Image.Image:
    return Image.fromarray((255 * np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)).astype(np.uint8)).convert("RGB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--true", type=Path, required=True)
    ap.add_argument("--solv3r", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for room_dir in sorted(p for p in args.true.iterdir() if (p / "cameras.npz").is_file()):
        room = room_dir.name
        if not (args.solv3r / room / "cameras.npz").is_file():
            print(json.dumps({"room": room, "solv3r": "missing"}))
            continue
        frames = json.loads((args.base / "renders_room" / room / "transforms.json").read_text())["frames"]
        rel_all, hit, tiles = [], [], []
        show = {0, len(frames) // 2, len(frames) - 1}
        for i, fr in enumerate(frames):
            t = np.load(room_dir / f"{i:04d}.npz")["depth"].astype(np.float32)
            s = np.load(args.solv3r / room / f"{i:04d}.npz")["depth"].astype(np.float32)
            Hs, Ws = s.shape
            ys = (np.arange(Hs) + 0.5) * t.shape[0] / Hs
            xs = (np.arange(Ws) + 0.5) * t.shape[1] / Ws
            t_s = t[ys.astype(int)[:, None], xs.astype(int)[None, :]]  # true depth on Solv3R's pixel grid
            valid = (t_s > 0) & (s > 0) & np.isfinite(s)
            rel = np.abs(s - t_s) / np.where(t_s > 0, t_s, 1)
            rel_all.append(rel[valid])
            hit.append(float((t > 0).mean()))
            if i in show:
                rgb = Image.open(args.base / "renders_room" / room / fr["file_path"]).convert("RGB").resize((Ws, Hs))
                hi = float(np.percentile(t_s[t_s > 0], 99)) if (t_s > 0).any() else 1.0
                err = np.where(valid, rel, 0)
                tiles.append([rgb, gray(t_s, 0, hi), gray(s, 0, hi), gray(err, 0, 0.2)])
        rel = np.concatenate(rel_all)
        print(json.dumps({"room": room, "frames": len(frames), "true_hit_min": round(min(hit), 4),
                          "rel_err_median": round(float(np.median(rel)), 4),
                          "within_5pct": round(float((rel < 0.05).mean()), 4),
                          "within_10pct": round(float((rel < 0.10).mean()), 4)}), flush=True)
        w, h = tiles[0][0].size
        sheet = Image.new("RGB", (4 * w, len(tiles) * h))
        for r, row in enumerate(tiles):
            for c, im in enumerate(row):
                sheet.paste(im, (c * w, r * h))
        sheet.save(args.out / f"{room}.png")


if __name__ == "__main__":
    main()
