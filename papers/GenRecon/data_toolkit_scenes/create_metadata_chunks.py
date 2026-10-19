#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder containing crops/")
    ap.add_argument("--ext", default="glb", choices=["glb", "ply"], help="Chunk file extension to index (default: glb)")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    crops_dir = root / "crops"
    out_csv = root / "metadata.csv"

    if not crops_dir.is_dir():
        raise SystemExit(f"Missing folder: {crops_dir}")

    crop_files = sorted(crops_dir.glob(f"*.{args.ext}"))

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sha256", "aesthetic_score", "local_path"])
        writer.writeheader()
        for p in crop_files:
            writer.writerow(
                {
                    "sha256": p.stem,
                    "local_path": f"crops/{p.stem}.{args.ext}",
                    "aesthetic_score": 5.0,
                }
            )

    print(f"Wrote {len(crop_files)} rows to {out_csv}")


if __name__ == "__main__":
    main()
