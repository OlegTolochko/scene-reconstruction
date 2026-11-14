"""Freezes the settings of a bench in OUT, once. Every run of the bench then trains and validates on the same
rooms with the same settings, which is what makes its curves comparable with the baseline's. To change a
setting, start a new bench (another OUT): it gets its own baseline.

Writes OUT/config.json (this folder's config.json plus the confidence statistics of the training rooms),
data_dir.json and val_data_dir.json for train.py's --data_dir / --val_data_dir, rooms.json, and OUT/data/: per
data root a depth-cue folder with links to the chosen rooms' cues and a metadata.csv of their chunks, which the
dataset filters by. Nothing is written into the data roots.

A data root is a ROOT prepared by genrecon_depth_cue_finetuning/rooms/prepare_rooms.sh (feat/depth-cue-finetuning):
renders_room/, crops/, ss_latents/, depth_cues_solv3r/, ... Rooms are chosen per root:
    ROOT:all            every room of the root that has depth cues
    ROOT:@overfit       the rooms of a split in rooms.csv (overfit: 10 rooms)
    ROOT:id,id,...      these rooms

    uv run experiments/genrecon_mlp_bench/setup_run.py --out OUT --train ROOT:@overfit --val ROOT:@overfit

(Adapted from genrecon_depth_cue_finetuning/make_run.py.)
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SS = "ss_enc_conv3d_16l8_fp16_64"  # get_slats.py's sparse-structure latents


def split_rooms(split: str) -> list[str]:
    with open(HERE / "rooms.csv") as f:
        rooms = [r["scene_id"] for r in csv.DictReader(f) if r["split"] == split]
    if not rooms:
        raise SystemExit(f"rooms.csv has no split {split!r}")
    return rooms


def rooms_of(root: Path, rooms: str, cues: str) -> tuple[list[str], list[str]]:
    """The chosen rooms that have depth cues in the root, and the chosen ones that do not."""
    ready = sorted(p.parent.name for p in (root / cues).glob("*/cameras.npz"))
    if not ready:
        raise SystemExit(f"{root / cues}: no room with depth cues (*/cameras.npz)")
    if rooms == "all":
        return ready, []
    wanted = split_rooms(rooms[1:]) if rooms.startswith("@") else rooms.split(",")
    missing = [r for r in wanted if r not in ready]
    if missing and not rooms.startswith("@"):
        raise SystemExit(f"{root}: no depth cues for {missing}")
    return [r for r in wanted if r in ready], missing  # a split's rooms without cues were dropped in preparation


def data_dirs(specs: list[str], out: Path, split: str, cues: str):
    dirs, used, dropped = {}, {}, {}
    for i, spec in enumerate(specs):
        root, _, rooms = spec.rpartition(":")
        if not root:
            raise SystemExit(f"{spec!r}: expected ROOT:all, ROOT:@split or ROOT:room,room,...")
        root = Path(root).resolve()
        rooms, missing = rooms_of(root, rooms, cues)
        if not rooms:
            raise SystemExit(f"{spec}: none of these rooms has depth cues in {root / cues}")
        sub = out / "data" / f"{split}{i}" / "depth_cues"
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
        ss = root / "ss_latents" / SS
        have = {p.stem for p in ss.glob("*.npz")}
        without = [c for c in chunks if c not in have]
        if without:
            raise SystemExit(f"{root}: no {ss.name} latents for {without[:5]} ({len(without)} chunks)")
        dirs[f"{split}{i}"] = {"base": str(root), "ss_latent": str(ss), "depth_cues": str(sub)}
        used[str(root)], dropped[str(root)] = rooms, missing
    return dirs, used, dropped


def chunk_count(dirs: dict) -> int:
    return sum(len((Path(d["depth_cues"]) / "metadata.csv").read_text().splitlines()) - 1 for d in dirs.values())


def conf_stats(dirs: dict) -> tuple[float, float]:
    """Pooled mean and std of the depth model's confidence over all depth maps of these rooms (equal-size maps),
    from the align.json that prepare_cues.py writes per room. (0, 1) for cues without it, such as the true depth."""
    n = s1 = s2 = 0.0
    for d in dirs.values():
        for room in Path(d["depth_cues"]).iterdir():
            if not room.is_dir():
                continue
            if not (room / "align.json").is_file():
                return 0.0, 1.0
            a = json.loads((room / "align.json").read_text())
            f = a["frames"]
            n += f
            s1 += f * a["conf_mean"]
            s2 += f * (a["conf_std"] ** 2 + a["conf_mean"] ** 2)
    mean = s1 / n
    return mean, float(np.sqrt(max(s2 / n - mean**2, 0.0))) or 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="the bench folder; its runs go inside")
    ap.add_argument("--train", nargs="+", required=True, help="ROOT:all, ROOT:@split or ROOT:room,room")
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--cues", default="depth_cues_solv3r", help="cue folder in each root (depth_cues_true: the oracle)")
    ap.add_argument("--config", type=Path, default=HERE / "config.json")
    args = ap.parse_args()

    out = args.out.resolve()
    train, train_rooms, train_dropped = data_dirs(args.train, out, "train", args.cues)
    val, val_rooms, val_dropped = data_dirs(args.val, out, "val", args.cues)
    overlap = {r for rs in train_rooms.values() for r in rs} & {r for rs in val_rooms.values() for r in rs}
    if overlap:  # allowed: an overfitting bench validates on its training rooms
        print(f"note: {len(overlap)} rooms are in both train and val")
    mean, std = conf_stats(train)

    cfg = json.loads(args.config.read_text())
    t = cfg["trainer"]["args"]
    # GenRecon's DataLoader drops the last incomplete batch and cycles forever: with fewer training chunks
    # than one batch it never yields one, and the run hangs silently instead of failing.
    if chunk_count(train) < t["batch_size_per_gpu"]:
        raise SystemExit(f"{chunk_count(train)} training chunks, fewer than one batch of {t['batch_size_per_gpu']}")
    t.update(conf_mean=mean, conf_std=std)
    info = {"cues": args.cues, "train": train_rooms, "val": val_rooms,
            "without_cues": {"train": train_dropped, "val": val_dropped},
            "chunks": {"train": chunk_count(train), "val": chunk_count(val)}, "conf_mean": mean, "conf_std": std}
    (out / "data_dir.json").write_text(json.dumps(train))
    (out / "val_data_dir.json").write_text(json.dumps(val))
    (out / "rooms.json").write_text(json.dumps(info, indent=2))
    (out / "config.json").write_text(json.dumps(cfg, indent=4))  # last: its presence marks the bench as set up
    rooms = {k: sum(len(r) for r in v.values()) for k, v in (("train", train_rooms), ("val", val_rooms))}
    print(f"bench {out.name}: {rooms['train']} training rooms ({info['chunks']['train']} chunks), "
          f"{rooms['val']} validation rooms ({info['chunks']['val']} chunks), cues {args.cues}, "
          f"confidence {mean:.2f} +- {std:.2f}")


if __name__ == "__main__":
    main()
