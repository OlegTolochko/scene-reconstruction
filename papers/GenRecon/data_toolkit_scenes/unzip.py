#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path
from typing import List


def shard_round_robin(items: List[Path], rank: int, world_size: int) -> List[Path]:
    # rank r gets indices r, r+world_size, ...
    return [items[i] for i in range(rank, len(items), world_size)]


def is_extraction_complete(out_dir: Path) -> bool:
    """
    Resumability sentinel: if this file exists, we consider the extraction complete.
    """
    return (out_dir / ".done").exists()


def safe_extractall(zf: zipfile.ZipFile, dest: Path) -> None:
    """
    Safe-ish extraction: prevents path traversal ('../') writes outside dest.
    """
    dest = dest.resolve()
    for member in zf.infolist():
        member_path = dest / member.filename
        if not member_path.resolve().is_relative_to(dest):
            raise RuntimeError(f"Unsafe path in zip: {member.filename}")
    zf.extractall(dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root directory containing zips/; will write rooms_raw/")
    ap.add_argument("--rank", type=int, default=0, help="This worker rank (0..world_size-1)")
    ap.add_argument("--world_size", type=int, default=1, help="Total number of workers")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract even if output is marked complete (removes existing folder).",
    )
    args = ap.parse_args()

    if args.world_size < 1:
        raise SystemExit("--world_size must be >= 1")
    if not (0 <= args.rank < args.world_size):
        raise SystemExit("--rank must be in [0, world_size)")

    root = Path(args.root).expanduser()
    zip_folder = root / "zips"
    out_folder = root / "rooms_raw"

    if not zip_folder.is_dir():
        raise SystemExit(f"Missing zip folder: {zip_folder}")

    out_folder.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(zip_folder.glob("*.zip"))
    if not zip_files:
        raise SystemExit(f"No .zip files found in: {zip_folder}")

    my_zips = shard_round_robin(zip_files, args.rank, args.world_size)

    print(
        f"Found {len(zip_files)} zip files. "
        f"Rank {args.rank}/{args.world_size} will extract {len(my_zips)} into {out_folder}"
    )
    if my_zips:
        print("First few for this rank:", [p.name for p in my_zips[:10]])

    for zip_path in my_zips:
        filename = zip_path.name
        base_name = zip_path.stem
        unique_id = base_name.split("_")[-1]

        output_dir = out_folder / unique_id

        if args.overwrite and output_dir.exists():
            shutil.rmtree(output_dir)

        # resumable skip
        if output_dir.exists() and is_extraction_complete(output_dir):
            continue

        # extract to temp then atomically rename (prevents partial outputs)
        tmp_dir = out_folder / f".tmp_extract_{unique_id}.{os.getpid()}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                safe_extractall(zf, tmp_dir)

            # mark complete
            (tmp_dir / ".done").write_text("ok\n", encoding="utf-8")

            # atomic replace of the whole directory
            if output_dir.exists():
                shutil.rmtree(output_dir)
            os.replace(tmp_dir, output_dir)

            print(f"Extracted {filename} -> {unique_id}")

        except Exception as e:
            # cleanup temp dir on failure so we can resume cleanly
            try:
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir)
            except Exception:
                pass
            raise RuntimeError(f"Failed extracting {zip_path}: {e}") from e

    print(f"Rank {args.rank}: Done.")


if __name__ == "__main__":
    main()
