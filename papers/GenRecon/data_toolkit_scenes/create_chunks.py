from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

from tqdm import tqdm

BLENDER_SCRIPT = Path(__file__).parent / "blender_scripts" / "create_chunks.py"

BLENDER_LINK = "https://ftp.halifax.rwth-aachen.de/blender/release/Blender5.0/blender-5.0.1-linux-x64.tar.xz"
BLENDER_INSTALLATION_PATH = "/tmp"
BLENDER_PATH = f"{BLENDER_INSTALLATION_PATH}/blender-5.0.1-linux-x64/blender"


def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        os.system("sudo apt-get update")
        os.system("sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6 libxfixes3 libgl1")
        os.system(f"wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}")
        os.system(f"tar -xvf {BLENDER_INSTALLATION_PATH}/blender-5.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}")


def load_scene_dims(layout_json_path: Path) -> dict:
    with layout_json_path.open("r", encoding="utf-8") as f:
        layout = json.load(f)

    rooms = layout.get("rooms") or []
    if not rooms:
        raise ValueError("layout has no 'rooms' array")

    dims = rooms[0].get("dimensions") or {}
    for k in ("width", "length", "height"):
        if k not in dims:
            raise ValueError(f"missing dimensions.{k}")

    return {"width": float(dims["width"]), "length": float(dims["length"]), "height": float(dims["height"])}


def iter_shard_round_robin(items: List[Path], rank: int, world_size: int) -> List[Path]:
    return [items[i] for i in range(rank, len(items), world_size)]


def count_existing_crops(out_dir: Path, scene_id: str, num_crops: int) -> int:
    """Count how many crop GLBs already exist for this scene (0..num_crops-1)."""
    return sum(1 for i in range(num_crops) if (out_dir / f"{scene_id}_{i:04d}.glb").is_file())


def run_blender_for_scene(
    blender_exe: str,
    blender_script: Path,
    blend_path: Path,
    out_dir: Path,
    num_crops: int,
    crop_size_range: list,
    dims: dict,
    start_idx: int = 0,
    seed: int | None = None,
    eps: float | None = 1.0,
    timeout_sec: int | None = None,
    verbose: bool = False,
) -> tuple[str, bool, str]:
    scene_id = blend_path.stem
    dims_json = json.dumps(dims, separators=(",", ":"))
    crop_size_json = json.dumps(crop_size_range)

    cmd = [
        blender_exe,
        "-b",
        "-P",
        str(blender_script),
        "--",
        str(blend_path),
        str(out_dir),
        str(num_crops),
        crop_size_json,
        dims_json,
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if eps is not None:
        cmd += ["--eps", str(eps)]
    if start_idx > 0:
        cmd += ["--start_idx", str(start_idx)]

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

    # Guard against silent Blender script failures that still return rc=0.
    expected_files = [out_dir / f"{scene_id}_{i:04d}.glb" for i in range(num_crops)]
    missing_files = [p for p in expected_files if not p.is_file()]
    if ok and missing_files:
        ok = False
        out += (
            f"\n[Chunker] Blender returned 0 but missing {len(missing_files)}/{num_crops} expected GLBs. "
            f"First missing: {missing_files[0]}"
        )

    # Write logs on failure or if verbose.
    if verbose or (not ok):
        logs_dir = out_dir / "_logs"
        try:
            logs_dir.mkdir(parents=True, exist_ok=True)
            (logs_dir / f"{scene_id}.log.txt").write_text(out, encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"[Chunker] Warning: could not write log for {scene_id}: {e}")

    msg = "OK" if ok else f"FAILED (rc={proc.returncode})"
    return scene_id, ok, msg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder containing rooms_processed/ and rooms_raw/")
    ap.add_argument(
        "--crop_size",
        type=float,
        nargs=2,
        default=[2.7, 3.0],
        metavar=("LOW", "HIGH"),
        help="Crop cube edge length range [low, high] sampled uniformly per crop (default: 2.7 3.0)",
    )
    ap.add_argument("--num_crops", type=int, required=True, help="Number of crops per scene")
    ap.add_argument("--jobs", type=int, default=1, help="Parallel Blender processes per rank (default: 1)")
    ap.add_argument("--seed", type=int, default=None, help="Base seed (optional). If set, each scene gets seed+idx.")
    ap.add_argument("--eps", type=float, default=1.0, help="Sampling bound expansion epsilon (default: 1.0)")
    ap.add_argument("--timeout", type=int, default=0, help="Per-scene timeout seconds (0 = no timeout)")
    ap.add_argument("--verbose", action="store_true", help="Always write blender logs to crops/_logs/")

    # distributed sharding
    ap.add_argument("--rank", type=int, default=0, help="This worker rank (0..world_size-1)")
    ap.add_argument("--world_size", type=int, default=1, help="Total number of workers")

    args = ap.parse_args()

    # install blender
    print("Checking blender...", flush=True)
    _install_blender()

    if args.world_size < 1:
        raise SystemExit("--world_size must be >= 1")
    if not (0 <= args.rank < args.world_size):
        raise SystemExit("--rank must be in [0, world_size)")

    crop_size_range = args.crop_size
    if crop_size_range[0] > crop_size_range[1]:
        raise SystemExit(f"--crop_size LOW must be <= HIGH, got {crop_size_range}")

    root = Path(args.root).expanduser().resolve()
    pre_dir = root / "rooms_processed"
    raw_dir = root / "rooms_raw"
    out_dir = root / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)

    blender_script = BLENDER_SCRIPT.resolve()
    if not blender_script.is_file():
        raise SystemExit(f"Blender script not found: {blender_script}")
    if not pre_dir.is_dir():
        raise SystemExit(f"Missing folder: {pre_dir}")
    if not raw_dir.is_dir():
        raise SystemExit(f"Missing folder: {raw_dir}")

    blend_files_all = sorted(pre_dir.glob("*.blend"))
    if not blend_files_all:
        raise SystemExit(f"No .blend files found in: {pre_dir}")

    # Shard the blend files by rank/world_size
    blend_files = iter_shard_round_robin(blend_files_all, args.rank, args.world_size)

    # Prepare tasks with per-scene dims and start_idx (for resumability)
    tasks: list[tuple[Path, dict, int | None, int]] = []
    failures_meta: list[tuple[str, str]] = []

    for local_idx, blend_path in enumerate(blend_files):
        scene_id = blend_path.stem

        # Resume: count existing crops and skip only if all done
        start_idx = count_existing_crops(out_dir, scene_id, args.num_crops)
        if start_idx >= args.num_crops:
            continue

        layout_path = raw_dir / scene_id / f"layout_{scene_id}.json"
        if not layout_path.is_file():
            failures_meta.append((scene_id, f"Missing layout JSON: {layout_path}"))
            continue

        try:
            dims = load_scene_dims(layout_path)
        except Exception as e:
            failures_meta.append((scene_id, f"Bad layout JSON ({layout_path.name}): {e}"))
            continue

        # Seed: make deterministic per-scene within rank.
        scene_seed = (args.seed + local_idx) if args.seed is not None else None
        tasks.append((blend_path, dims, scene_seed, start_idx))

    timeout_sec = None if args.timeout <= 0 else args.timeout
    jobs = max(1, int(args.jobs))
    failures_run: list[tuple[str, str]] = []

    if not tasks and failures_meta:
        print(f"Rank {args.rank}: No scenes queued (all assigned scenes done or layouts invalid).")
        print("\nLayout issues (first 50):")
        for scene_id, msg in failures_meta[:50]:
            print(f"  - {scene_id}: {msg}")
        raise SystemExit(2)

    if not tasks:
        print(f"Rank {args.rank}: Nothing to do (all assigned scenes already fully cropped).")
        return

    total = len(tasks)
    desc = f"Rank {args.rank}/{args.world_size} crops (jobs={jobs})"
    with tqdm(total=total, desc=desc, unit="scene") as pbar:
        if jobs == 1:
            for blend_path, dims, scene_seed, start_idx in tasks:
                scene_id, ok, msg = run_blender_for_scene(
                    blender_exe=BLENDER_PATH,
                    blender_script=blender_script,
                    blend_path=blend_path,
                    out_dir=out_dir,
                    num_crops=args.num_crops,
                    crop_size_range=crop_size_range,
                    dims=dims,
                    start_idx=start_idx,
                    seed=scene_seed,
                    eps=args.eps,
                    timeout_sec=timeout_sec,
                    verbose=args.verbose,
                )
                if not ok:
                    failures_run.append((scene_id, msg))
                pbar.set_postfix_str(f"{scene_id}: {msg}")
                pbar.update(1)
        else:
            with ThreadPoolExecutor(max_workers=jobs) as ex:
                futs = [
                    ex.submit(
                        run_blender_for_scene,
                        BLENDER_PATH,
                        blender_script,
                        blend_path,
                        out_dir,
                        args.num_crops,
                        crop_size_range,
                        dims,
                        start_idx,
                        scene_seed,
                        args.eps,
                        timeout_sec,
                        args.verbose,
                    )
                    for (blend_path, dims, scene_seed, start_idx) in tasks
                ]
                for fut in as_completed(futs):
                    scene_id, ok, msg = fut.result()
                    if not ok:
                        failures_run.append((scene_id, msg))
                    pbar.set_postfix_str(f"{scene_id}: {msg}")
                    pbar.update(1)

    print(f"\n=== Rank {args.rank} Summary ===")
    print(f"Total .blend (global):     {len(blend_files_all)}")
    print(f"Assigned to this rank:     {len(blend_files)}")
    print(f"Queued (not fully done):   {total}")
    print(f"Layout issues:             {len(failures_meta)}")
    print(f"Succeeded:                 {total - len(failures_run)}")
    print(f"Failed (Blender):          {len(failures_run)}")

    if failures_meta:
        print("\nLayout issues (first 50):")
        for scene_id, msg in failures_meta[:50]:
            print(f"  - {scene_id}: {msg}")

    if failures_run:
        print("\nBlender failures (first 50):")
        for scene_id, msg in failures_run[:50]:
            print(f"  - {scene_id}: {msg}")
        print("\nCheck logs in:", out_dir / "_logs")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
