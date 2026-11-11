"""Picks the SAGE-10k rooms of the depth-cue finetuning, writes them to sage_rooms.csv, and downloads them.

The rooms are the last --num of SAGE-10k's sorted zip list. GenRecon's toolkit downloads from the start of the
list (download_data.py --num), so GenRecon most likely trained on the first rooms; from the end, an overlap is
unlikely, though not certain. The rooms are split into train and val per room with a fixed seed. The overfit
room of the first test (0063e67d, from the start of the list) is listed too, with split "overfit".

sage_rooms.csv: scene_id, split (overfit / train / val), list_index (position in the sorted list of 10,000), zip,
size_mb. It is committed, so every run uses the same rooms.

From papers/GenRecon, in GenRecon's env:
    python ../../experiments/genrecon_depth_cue_finetuning/rooms/select_rooms.py              # list only
    python ../../experiments/genrecon_depth_cue_finetuning/rooms/select_rooms.py --download ROOT
ROOT then holds zips/ for data_toolkit_scenes/unzip.py --root ROOT and the steps after it.
"""

import argparse
import csv
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "papers/GenRecon/data_toolkit_scenes"))
from download_data import atomic_copy, list_scene_zip_entries  # noqa: E402

REPO = "nvidia/SAGE-10k"
OVERFIT = ["0063e67d"]
CSV = HERE / "sage_rooms.csv"


def scene_id(zip_path: str) -> str:
    return Path(zip_path).stem.rsplit("_layout_", 1)[1]  # scenes/<date>_<time>_layout_<id>.zip


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num", type=int, default=60, help="rooms from the end of the list")
    ap.add_argument("--num-val", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--download", type=Path, metavar="ROOT", help="download the train/val zips to ROOT/zips")
    args = ap.parse_args()

    entries = list_scene_zip_entries(REPO)
    index = {scene_id(e.path): i for i, e in enumerate(entries)}
    tail = list(range(len(entries) - args.num, len(entries)))
    val = set(random.Random(args.seed).sample(tail, args.num_val))
    rows = [(s, "overfit", index[s]) for s in OVERFIT]
    rows += [(scene_id(entries[i].path), "val" if i in val else "train", i) for i in tail]

    with open(CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene_id", "split", "list_index", "zip", "size_mb"])
        for s, split, i in rows:
            e = entries[i]
            w.writerow([s, split, i, Path(e.path).name, f"{(e.size or 0) / 2**20:.1f}"])
    counts = {k: sum(r[1] == k for r in rows) for k in ("overfit", "train", "val")}
    print(f"{len(entries)} rooms in {REPO}; wrote {CSV.name}: {counts}")

    if args.download:
        from huggingface_hub import hf_hub_download

        out = args.download / "zips"
        out.mkdir(parents=True, exist_ok=True)
        for s, split, i in rows:
            if split == "overfit":
                continue
            e = entries[i]
            dst = out / Path(e.path).name
            if dst.exists() and (e.size is None or dst.stat().st_size == e.size):
                continue
            cached = hf_hub_download(repo_id=REPO, repo_type="dataset", filename=e.path,
                                     cache_dir=args.download / ".hf_hub_cache")
            atomic_copy(Path(cached), dst)
            print(f"downloaded {dst.name}", flush=True)


if __name__ == "__main__":
    main()
