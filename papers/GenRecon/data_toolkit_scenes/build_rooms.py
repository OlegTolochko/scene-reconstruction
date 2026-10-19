#!/usr/bin/env python3
"""
Batch-process raw rooms by calling Blender in background mode.

Inputs:
- root folder
  - root/rooms_raw/<scene_id>/  (each contains layout_*.json, materials/, objects/)
Outputs:
- root/rooms_processed/<scene_id>.blend

Parallelization options:
1) Distributed sharding across processes/nodes:
   --rank R --world_size W
2) Local parallelism per rank:
   --jobs N   (runs N Blender subprocesses in parallel on this rank)

Resumable:
- Automatically skips scenes where output .blend already exists.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

BLENDER_SCRIPT = "data_toolkit_scenes/blender_scripts/build_rooms.py"

BLENDER_LINK = "https://ftp.halifax.rwth-aachen.de/blender/release/Blender5.0/blender-5.0.1-linux-x64.tar.xz"
BLENDER_INSTALLATION_PATH = "/tmp"
BLENDER_PATH = f"{BLENDER_INSTALLATION_PATH}/blender-5.0.1-linux-x64/blender"


def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        os.system("sudo apt-get update")
        os.system("sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6 libxfixes3 libgl1")
        os.system(f"wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}")
        os.system(f"tar -xvf {BLENDER_INSTALLATION_PATH}/blender-5.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}")


def is_scene_folder(p: Path) -> bool:
    if not p.is_dir():
        return False
    if not (p / "materials").is_dir():
        return False
    if not (p / "objects").is_dir():
        return False
    if not list(p.glob("layout_*.json")):
        return False
    return True


def iter_shard_round_robin(items: List[Path], rank: int, world_size: int) -> List[Path]:
    # rank r gets indices r, r+world_size, ...
    return [items[i] for i in range(rank, len(items), world_size)]


def run_blender_one(
    blender_exe: str,
    blender_script: Path,
    input_folder: Path,
    output_blend_path: Path,
    timeout_sec: int | None = None,
    verbose: bool = False,
) -> tuple[str, bool, str]:
    """
    Returns (scene_id, ok, message).
    """
    scene_id = input_folder.name
    output_blend_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        blender_exe,
        "-b",
        "-P",
        str(blender_script),
        "--",
        str(input_folder),
        str(output_blend_path),
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except FileNotFoundError as e:
        return scene_id, False, f"Blender executable not found: {blender_exe} ({e})"
    except subprocess.TimeoutExpired:
        return scene_id, False, f"Timeout after {timeout_sec}s"
    except Exception as e:
        return scene_id, False, f"Unexpected error: {e}"

    ok = proc.returncode == 0
    out = proc.stdout or ""

    # Basic success check: output file exists
    if ok and not output_blend_path.is_file():
        ok = False
        out += f"\n[BatchRunner] Blender returned 0 but output not found: {output_blend_path}"

    # Write logs: on failure always; on success only if verbose
    if verbose or (not ok):
        log_path = output_blend_path.with_name(f"{scene_id}_blender_log.txt")
        try:
            log_path.write_text(out, encoding="utf-8")
        except Exception:
            pass

    msg = "OK" if ok else f"FAILED (rc={proc.returncode})"
    return scene_id, ok, msg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root dataset folder (contains rooms_raw/).")
    ap.add_argument("--jobs", type=int, default=1, help="Parallel Blender processes per rank (default: 1).")

    # distributed sharding
    ap.add_argument("--rank", type=int, default=0, help="This worker rank (0..world_size-1).")
    ap.add_argument("--world_size", type=int, default=1, help="Total number of workers.")

    ap.add_argument("--timeout", type=int, default=0, help="Per-room timeout in seconds (0 = no timeout).")
    ap.add_argument("--verbose", action="store_true", help="Save <scene_id>_blender_log.txt for all scenes.")
    ap.add_argument("--only", nargs="*", default=None, help="Only process these scene_ids.")
    args = ap.parse_args()

    # install blender
    print("Checking blender...", flush=True)
    _install_blender()

    if args.world_size < 1:
        raise SystemExit("--world_size must be >= 1")
    if not (0 <= args.rank < args.world_size):
        raise SystemExit("--rank must be in [0, world_size)")

    root = Path(args.root).expanduser().resolve()
    raw_root = root / "rooms_raw"
    out_root = root / "rooms_processed"

    blender_script = Path(BLENDER_SCRIPT).expanduser().resolve()
    if not blender_script.is_file():
        raise SystemExit(f"Blender script not found: {blender_script}")

    if not raw_root.is_dir():
        raise SystemExit(f"rooms_raw folder not found: {raw_root}")

    out_root.mkdir(parents=True, exist_ok=True)

    # Discover scenes (global list)
    candidates = sorted([p for p in raw_root.iterdir() if is_scene_folder(p)])
    if args.only:
        wanted = set(args.only)
        candidates = [p for p in candidates if p.name in wanted]

    if not candidates:
        raise SystemExit(f"No valid scene folders found in: {raw_root}")

    # Shard by rank/world_size
    candidates = iter_shard_round_robin(candidates, args.rank, args.world_size)

    timeout = None if args.timeout <= 0 else args.timeout

    # Prepare tasks (auto-resume: skip existing outputs)
    tasks: List[Tuple[Path, Path]] = []
    for scene_in in candidates:
        scene_out = out_root / f"{scene_in.name}.blend"
        if scene_out.is_file():
            continue  # resume-friendly default
        tasks.append((scene_in, scene_out))

    if not tasks:
        print(f"Rank {args.rank}: Nothing to do (all assigned scenes already processed).")
        return

    failures: list[tuple[str, str]] = []

    jobs = max(1, int(args.jobs))
    desc = f"Rank {args.rank}/{args.world_size} (jobs={jobs})"
    with tqdm(total=len(tasks), desc=desc, unit="room") as pbar:
        if jobs == 1:
            for scene_in, scene_out in tasks:
                scene_id, ok, msg = run_blender_one(
                    blender_exe=BLENDER_PATH,
                    blender_script=blender_script,
                    input_folder=scene_in,
                    output_blend_path=scene_out,
                    timeout_sec=timeout,
                    verbose=args.verbose,
                )
                if not ok:
                    failures.append((scene_id, msg))
                pbar.set_postfix_str(f"{scene_id}: {msg}")
                pbar.update(1)
        else:
            # ThreadPool is fine: we are IO-bound on subprocess management; Blender does the CPU work.
            with ThreadPoolExecutor(max_workers=jobs) as ex:
                future_map = {
                    ex.submit(
                        run_blender_one,
                        BLENDER_PATH,
                        blender_script,
                        scene_in,
                        scene_out,
                        timeout,
                        args.verbose,
                    ): scene_in.name
                    for scene_in, scene_out in tasks
                }
                for fut in as_completed(future_map):
                    scene_id, ok, msg = fut.result()
                    if not ok:
                        failures.append((scene_id, msg))
                    pbar.set_postfix_str(f"{scene_id}: {msg}")
                    pbar.update(1)

    # Summary (per rank)
    print(f"\n=== Rank {args.rank} Summary ===")
    print(f"Assigned candidates: {len(candidates)}")
    print(f"Processed:          {len(tasks)}")
    print(f"Succeeded:          {len(tasks) - len(failures)}")
    print(f"Failed:             {len(failures)}")
    if failures:
        print("\nFailures:")
        for scene_id, msg in failures:
            print(f"  - {scene_id}: {msg}")
        print("\nTip: Check rooms_processed/<scene_id>_blender_log.txt for details.")

    # Non-zero exit if failures (keeps old behavior, but only for this rank)
    if failures:
        sys.exit(2)


if __name__ == "__main__":
    main()
