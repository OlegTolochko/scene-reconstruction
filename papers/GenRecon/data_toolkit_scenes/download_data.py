#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm


@dataclass(frozen=True)
class ZipEntry:
    path: str  # e.g. scenes/xxxx.zip
    size: Optional[int]  # remote bytes (None if unknown)


def list_scene_zip_entries(repo_id: str, revision: str = "main") -> List[ZipEntry]:
    """
    Return ZIP files directly under scenes/ (non-recursive), with remote size if available.
    """
    api = HfApi()

    # list_repo_tree gives richer metadata (incl. size; and for LFS often oid)
    items = api.list_repo_tree(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        path_in_repo="scenes",
        recursive=False,
    )

    out: List[ZipEntry] = []
    for it in items:
        p = getattr(it, "path", None)
        if not isinstance(p, str):
            continue
        if p.startswith("scenes/") and p.lower().endswith(".zip") and p.count("/") == 1:
            out.append(ZipEntry(path=p, size=getattr(it, "size", None)))

    return sorted(out, key=lambda e: e.path)


def iter_shard(items: List[ZipEntry], rank: int, world_size: int) -> Iterable[ZipEntry]:
    """
    Round-robin sharding: rank r gets indices r, r+world_size, ...
    Good load balance if file sizes vary.
    """
    return (items[i] for i in range(rank, len(items), world_size))


def atomic_copy(src: Path, dst: Path) -> None:
    """
    Copy bytes via a temp file then atomically replace destination.
    Prevents partial/corrupt final files if interrupted.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + f".tmp.{os.getpid()}")
    try:
        tmp.write_bytes(src.read_bytes())
        os.replace(tmp, dst)  # atomic rename
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="nvidia/SAGE-10k", help="HF dataset repo id")
    ap.add_argument("--revision", default="main", help="Branch/tag/commit")
    ap.add_argument("--num", type=int, default=100, help="How many ZIP scenes to download (from start of list)")
    ap.add_argument("--root", required=True, help="Root Directory")
    ap.add_argument(
        "--cache-dir",
        default=None,
        help=("HF download cache directory. Defaults to $HF_HUB_CACHE if set, " "otherwise <root>/.hf_hub_cache."),
    )

    # distributed params
    ap.add_argument("--rank", type=int, default=0, help="This worker rank (0..world_size-1)")
    ap.add_argument("--world_size", type=int, default=1, help="Total number of workers")

    # resume behavior
    ap.add_argument(
        "--skip-size-check",
        action="store_true",
        help="Skip remote-size verification and only use dst.exists() to resume.",
    )
    args = ap.parse_args()

    if args.world_size < 1:
        raise SystemExit("--world_size must be >= 1")
    if not (0 <= args.rank < args.world_size):
        raise SystemExit("--rank must be in [0, world_size)")

    root_dir = Path(args.root).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else None
    if cache_dir is None and not os.environ.get("HF_HUB_CACHE"):
        cache_dir = root_dir / ".hf_hub_cache"
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    out_dir = root_dir / "zips"
    out_dir.mkdir(parents=True, exist_ok=True)

    # quick write test
    test = out_dir / ".write_test"
    try:
        test.write_text("ok")
        test.unlink()
    except Exception as e:
        raise SystemExit(f"ERROR: cannot write to {out_dir}: {e}")

    entries = list_scene_zip_entries(args.repo, args.revision)
    if not entries:
        raise SystemExit("No ZIP files found directly under scenes/. Repo layout may have changed.")

    selected = entries[: args.num]
    shard = list(iter_shard(selected, args.rank, args.world_size))

    print(
        f"Found {len(entries)} ZIP scenes. Considering first {len(selected)}.\n"
        f"Rank {args.rank}/{args.world_size} will process {len(shard)} files to: {out_dir}\n"
        f"HF cache dir: {cache_dir if cache_dir is not None else 'default'}"
    )
    if shard:
        print("First few for this rank:", [Path(e.path).name for e in shard[:10]])

    for e in tqdm(shard, desc=f"Rank {args.rank} downloading"):
        filename = Path(e.path).name
        dst = out_dir / filename

        # --- Resume logic ---
        if dst.exists():
            if args.skip_size_check or e.size is None:
                # existence-only resume
                continue
            # size-based resume (Tier B)
            if dst.stat().st_size == e.size:
                continue
            # size mismatch: treat as partial/corrupt and re-copy
            # (optional) remove the bad file to be explicit
            try:
                dst.unlink()
            except Exception:
                pass

        # Download to HF cache (hf_hub_download verifies via ETag/cache)
        cached_path = hf_hub_download(
            repo_id=args.repo,
            repo_type="dataset",
            revision=args.revision,
            filename=e.path,
            cache_dir=cache_dir,
        )

        # Copy from cache to target folder atomically
        atomic_copy(Path(cached_path), dst)

        # Optional: re-check size after copy (cheap sanity check)
        if (not args.skip_size_check) and (e.size is not None):
            if dst.stat().st_size != e.size:
                raise RuntimeError(
                    f"Size mismatch after copy for {dst} (local {dst.stat().st_size} != remote {e.size})"
                )

    print(f"Rank {args.rank}: Done.")


if __name__ == "__main__":
    main()
