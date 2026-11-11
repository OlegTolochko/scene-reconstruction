"""Sets up a train.py run: its config, and the data folders restricted to the rooms of the run.

For each data root (a prepare_rooms.sh ROOT), the run uses only the rooms given for it: they get a cue folder
<root>/runs/<name>/<train|val>/depth_cues/ with links to their depth cues and a metadata.csv of their chunks, which
CueRoomSparseStructureLatent filters by. The sparse-structure latents are the thinned ones of thin_walls.py
(--latents thinwalls, default) or get_slats.py's (--latents original).

The confidence standardisation (conf_mean, conf_std) is pooled over the depth maps of the training rooms, from
prepare_cues.py's align.json. Writes <out>/config.json (config_ss.json with the overrides), data_dir.json and
val_data_dir.json for train.py's --data_dir / --val_data_dir, and rooms.json.

    uv run experiments/genrecon_depth_cue_finetuning/make_run.py --name NAME --out OUT \
        --train ROOT:room,room,... [ROOT:all ...] --val ROOT:all [--max-steps 2000 ...]
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SS = "ss_enc_conv3d_16l8_fp16_64"


def parse(spec: str):
    root, _, rooms = spec.rpartition(":")
    return Path(root).resolve(), rooms


def rooms_of(root: Path, rooms: str, cues: str) -> list[str]:
    ready = sorted(p.parent.name for p in (root / cues).glob("*/cameras.npz"))
    if rooms == "all":
        return ready
    wanted = rooms.split(",")
    missing = [r for r in wanted if r not in ready]
    if missing:
        raise SystemExit(f"{root}: no depth cues for {missing}")
    return wanted


def data_dirs(specs, name: str, split: str, cues: str, latents: str):
    dirs, used = {}, {}
    for i, spec in enumerate(specs):
        root, rooms = parse(spec)
        rooms = rooms_of(root, rooms, cues)
        sub = root / "runs" / name / split / "depth_cues"
        sub.mkdir(parents=True, exist_ok=True)
        chunks = []
        for room in rooms:
            link = sub / room
            if not link.exists():
                link.symlink_to(root / cues / room)
            crops = json.loads((root / "crops" / f"{room}.json").read_text())["chunks"]
            chunks += [f"{room}_{int(c['index']):04d}" for c in crops]
        with open(sub / "metadata.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sha256", "depth_cues"])
            w.writerows([[c, True] for c in chunks])
        ss = root / "ss_latents" / (SS + ("_thinwalls" if latents == "thinwalls" else ""))
        have = {p.stem for p in ss.glob("*.npz")}
        missing = [c for c in chunks if c not in have]
        if missing:
            raise SystemExit(f"{root}: no {ss.name} latents for {missing[:5]} ({len(missing)} chunks)")
        dirs[f"sage{i}"] = {"base": str(root), "ss_latent": str(ss), "depth_cues": str(sub)}
        used[str(root)] = rooms
    return dirs, used


def conf_stats(dirs: dict) -> tuple[float, float]:
    """Pooled mean and std of Solv3R's confidence over all depth maps of these rooms (equal-size maps)."""
    n = s1 = s2 = 0.0
    for d in dirs.values():
        for room in Path(d["depth_cues"]).iterdir():
            if not room.is_dir():
                continue
            a = json.loads((room / "align.json").read_text())
            f = a["frames"]
            n += f
            s1 += f * a["conf_mean"]
            s2 += f * (a["conf_std"] ** 2 + a["conf_mean"] ** 2)
    mean = s1 / n
    return mean, float(np.sqrt(s2 / n - mean**2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", type=Path, required=True, help="the runs' parent folder (cues/ and control/ go inside)")
    ap.add_argument("--train", nargs="+", required=True, help="ROOT:room,room or ROOT:all")
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--cues", default="depth_cues_solv3r", help="cue folder in each root")
    ap.add_argument("--latents", choices=("thinwalls", "original"), default="thinwalls")
    ap.add_argument("--config", type=Path, default=HERE / "config_ss.json")
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--i-val", type=int, default=250)
    ap.add_argument("--i-save", type=int, default=500)
    args = ap.parse_args()

    train, train_rooms = data_dirs(args.train, args.name, "train", args.cues, args.latents)
    val, val_rooms = data_dirs(args.val, args.name, "val", args.cues, args.latents)
    overlap = {r for rs in train_rooms.values() for r in rs} & {r for rs in val_rooms.values() for r in rs}
    if overlap:
        raise SystemExit(f"rooms in both train and val: {sorted(overlap)}")
    mean, std = conf_stats(train)

    cfg = json.loads(args.config.read_text())
    t = cfg["trainer"]["args"]
    t.update(max_steps=args.max_steps, i_val=args.i_val, i_save=args.i_save, i_sample=10**9,
             ema_rate=[args.ema], conf_mean=mean, conf_std=std)
    t["optimizer"]["args"]["lr"]["new"] = args.lr
    t["lr_scheduler"]["args"]["warmup_steps"] = args.warmup
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(cfg, indent=4))
    (args.out / "data_dir.json").write_text(json.dumps(train))
    (args.out / "val_data_dir.json").write_text(json.dumps(val))
    info = {"train": train_rooms, "val": val_rooms, "latents": args.latents, "conf_mean": mean, "conf_std": std,
            "chunks": {k: sum(1 for _ in open(Path(v["depth_cues"]) / "metadata.csv")) - 1
                       for k, v in {**train, **{f"val_{k}": v for k, v in val.items()}}.items()}}
    (args.out / "rooms.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
