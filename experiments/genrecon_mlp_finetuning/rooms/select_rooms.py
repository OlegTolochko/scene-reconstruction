"""Lists the SAGE-10k rooms of the depth-cue finetuning in sage_rooms.csv, and sets up a data root for a split.

The rooms are the 257 we downloaded: every second zip of the first 514 in SAGE-10k's sorted list (GenRecon's
data_toolkit_scenes/download_data.py --num 1000 --rank 0 --world_size 2, stopped after 257). Splits:
  overfit   0063e67d, the room of the first overfit test
  train_1   50 rooms for the first training run (generalisation pilot)
  val_1     10 rooms held out from it
  reserve   the other 196, for later runs
train_1 and val_1 are drawn from the 256 rooms other than the overfit room with a fixed seed.

sage_rooms.csv: scene_id, split, list_index (position in the sorted list of 10,000), zip, size_mb. It is
committed, so every run uses the same rooms.

From papers/GenRecon, in GenRecon's env:
    python ../../experiments/genrecon_mlp_finetuning/rooms/select_rooms.py                # list only
    python ../../experiments/genrecon_mlp_finetuning/rooms/select_rooms.py \
        --setup ROOT --splits train_1 val_1 --zips EXISTING/zips
--setup gives ROOT/zips the zips of those splits: a symlink when the zip is already in --zips, else a download.
ROOT then serves data_toolkit_scenes/unzip.py --root ROOT and the steps after it.
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
DOWNLOADED = slice(0, 514, 2)  # the 257 zips of download_data.py --rank 0 --world_size 2
OVERFIT = "0063e67d"
SPLITS = {"train_1": 50, "val_1": 10}
SEED = 0
CSV = HERE / "sage_rooms.csv"


def scene_id(zip_path: str) -> str:
    return Path(zip_path).stem.rsplit("_layout_", 1)[1]  # scenes/<date>_<time>_layout_<id>.zip


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setup", type=Path, metavar="ROOT", help="give ROOT/zips the zips of --splits")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), help="splits for --setup")
    ap.add_argument("--zips", type=Path, help="folder with zips already downloaded (symlinked instead)")
    args = ap.parse_args()

    entries = list_scene_zip_entries(REPO)
    rooms = list(range(len(entries)))[DOWNLOADED]
    pool = [i for i in rooms if scene_id(entries[i].path) != OVERFIT]
    assert len(pool) == len(rooms) - 1, f"{OVERFIT} is not among the downloaded rooms"
    drawn = random.Random(SEED).sample(pool, sum(SPLITS.values()))
    split = {i: "reserve" for i in pool}
    start = 0
    for name, n in SPLITS.items():
        split.update({i: name for i in drawn[start:start + n]})
        start += n
    split.update({i: "overfit" for i in rooms if i not in split})

    rows = [(scene_id(entries[i].path), split[i], i, entries[i]) for i in rooms]
    with open(CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene_id", "split", "list_index", "zip", "size_mb"])
        for s, sp, i, e in rows:
            w.writerow([s, sp, i, Path(e.path).name, f"{(e.size or 0) / 2**20:.1f}"])
    counts = {k: sum(r[1] == k for r in rows) for k in ["overfit", *SPLITS, "reserve"]}
    print(f"{len(entries)} rooms in {REPO}; wrote {CSV.name}: {counts}")

    if args.setup:
        from huggingface_hub import hf_hub_download

        out = args.setup / "zips"
        out.mkdir(parents=True, exist_ok=True)
        for s, sp, i, e in rows:
            if sp not in args.splits:
                continue
            name = Path(e.path).name
            dst = out / name
            if dst.exists():
                continue
            if args.zips and (args.zips / name).is_file():
                dst.symlink_to((args.zips / name).resolve())
                continue
            cached = hf_hub_download(repo_id=REPO, repo_type="dataset", filename=e.path,
                                     cache_dir=args.setup / ".hf_hub_cache")
            atomic_copy(Path(cached), dst)
            print(f"downloaded {name}", flush=True)
        print(f"{out}: {sum(1 for _ in out.glob('*.zip'))} zips")


if __name__ == "__main__":
    main()
