#!/usr/bin/env python3
"""
Batch room rendering runner.

Inputs:
- --root (expects <root>/rooms_processed/*.blend and <root>/rooms_raw/<scene_id>/layout_*.json)
- --num_views_per_m2 (float)

Outputs:
- <root>/renders_room/<scene_id>/000.png, 001.png, ...
- <root>/renders_room/<scene_id>/transforms.json
- <root>/renders_room/new_records/part_<job_id>_r<rank>.csv
- <root>/renders_room/metadata.csv (when finalized)
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from tqdm import tqdm

BLENDER_SCRIPT = "data_toolkit_scenes/blender_scripts/render_rooms.py"

BLENDER_LINK = "https://ftp.halifax.rwth-aachen.de/blender/release/Blender5.0/blender-5.0.1-linux-x64.tar.xz"
BLENDER_INSTALLATION_PATH = "/tmp"
BLENDER_PATH = f"{BLENDER_INSTALLATION_PATH}/blender-5.0.1-linux-x64/blender"

METADATA_FIELDS = [
    "sha256",
    "local_path",
    "cond_rendered",
    "status",
    "error",
    "rank",
    "world_size",
    "job_id",
    "updated_at",
]


def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        os.system("sudo apt-get update")
        os.system("sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6 libxfixes3 libgl1")
        os.system(f"wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}")
        os.system(f"tar -xvf {BLENDER_INSTALLATION_PATH}/blender-5.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}")


def _extract_room_dimensions(room_json_path: Path) -> Tuple[float, float, float]:
    with room_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    dims: Optional[Dict[str, float]] = None
    if isinstance(data, dict):
        if isinstance(data.get("rooms"), list) and data["rooms"]:
            room0 = data["rooms"][0]
            if isinstance(room0, dict) and isinstance(room0.get("dimensions"), dict):
                dims = room0["dimensions"]
        if dims is None and isinstance(data.get("dimensions"), dict):
            dims = data["dimensions"]
        if dims is None and all(k in data for k in ("width", "length", "height")):
            dims = data

    if dims is None:
        raise ValueError(f"Could not find room dimensions in JSON: {room_json_path}")

    width = float(dims["width"])
    length = float(dims["length"])
    height = float(dims["height"])
    if width <= 0.0 or length <= 0.0 or height <= 0.0:
        raise ValueError(
            f"Invalid room dimensions in {room_json_path}: width={width}, length={length}, height={height}"
        )
    return width, length, height


def _find_layout_json(raw_room_dir: Path, scene_id: str) -> Path:
    preferred = raw_room_dir / f"layout_{scene_id}.json"
    if preferred.is_file():
        return preferred

    candidates = sorted(raw_room_dir.glob("layout_*.json"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(f"No layout_*.json found in: {raw_room_dir}")


def run_blender_render(
    blender_exe: Path,
    blender_script: Path,
    room_blend: Path,
    room_json: Path,
    out_folder: Path,
    num_views: int,
    resolution: int,
    device: str,
    samples: int,
    engine: str,
    no_denoising: bool,
) -> None:
    cmd = [
        str(blender_exe),
        "-b",
        str(room_blend),
        "--python-exit-code",
        "1",
        "-P",
        str(blender_script),
        "--",
        "--room_json",
        str(room_json),
        "--out_folder",
        str(out_folder),
        "--num_views",
        str(num_views),
        "--render_resolution",
        str(resolution),
        "--engine",
        engine,
        "--device",
        device,
        "--samples",
        str(samples),
    ]
    if no_denoising:
        cmd.append("--no_denoising")

    subprocess.run(cmd, check=True)

    missing = [out_folder / f"{i:03d}.png" for i in range(num_views) if not (out_folder / f"{i:03d}.png").is_file()]
    if missing:
        first = ", ".join(str(p) for p in missing[:3])
        raise RuntimeError(f"Missing rendered images ({len(missing)} missing). Example: {first}")

    transforms = out_folder / "transforms.json"
    if not transforms.is_file():
        raise RuntimeError(f"Missing transforms.json: {transforms}")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _default_job_id() -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_pid{os.getpid()}"


def _truthy(value: Optional[Union[str, bool]]) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _normalize_record(row: Dict[str, str]) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    for key in METADATA_FIELDS:
        normalized[key] = str(row.get(key, ""))
    if normalized["cond_rendered"] == "":
        normalized["cond_rendered"] = "False"
    return normalized


def _read_csv_records(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [_normalize_record(dict(row)) for row in reader]


def _atomic_write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(_normalize_record(row))
    os.replace(tmp_path, path)


def merge_metadata_shards(out_root: Path) -> int:
    metadata_csv = out_root / "metadata.csv"
    new_records_dir = out_root / "new_records"
    lock_path = out_root / "metadata.lock"
    new_records_dir.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        existing_rows = _read_csv_records(metadata_csv)
        shard_rows: List[Dict[str, str]] = []
        for shard_csv in sorted(new_records_dir.glob("*.csv")):
            shard_rows.extend(_read_csv_records(shard_csv))

        all_rows = existing_rows + shard_rows
        merged_by_sha: Dict[str, Dict[str, str]] = {}
        for row in sorted(all_rows, key=lambda r: (r.get("updated_at", ""), r.get("job_id", ""), r.get("rank", ""))):
            sha256 = row.get("sha256", "").strip()
            if not sha256:
                continue
            merged_by_sha[sha256] = _normalize_record(row)

        merged_rows = [merged_by_sha[k] for k in sorted(merged_by_sha)]
        _atomic_write_csv(metadata_csv, merged_rows)

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    return len(merged_rows)


def _load_rendered_sha256_from_metadata(metadata_csv: Path) -> Set[str]:
    rendered: Set[str] = set()
    for row in _read_csv_records(metadata_csv):
        sha256 = row.get("sha256", "").strip()
        if sha256 and _truthy(row.get("cond_rendered")):
            rendered.add(sha256)
    return rendered


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch render room BLEND files under <root>/rooms_processed")
    ap.add_argument("--root", type=str, required=True, help="Dataset root directory.")
    ap.add_argument("--num_views_per_m2", type=float, required=True, help="Number of cameras per square meter.")
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--engine", type=str, default="CYCLES")
    ap.add_argument("--device", type=str, default="CUDA", choices=["CPU", "CUDA", "OPTIX", "HIP"])
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--min_views", type=int, default=16, help="Minimum number of views per room.")
    ap.add_argument("--max_views", type=int, default=160, help="Maximum number of views per room.")
    ap.add_argument("--no_denoising", action="store_true", default=False)
    ap.add_argument("--rank", type=int, default=0, help="Current worker rank.")
    ap.add_argument("--world_size", type=int, default=1, help="Total number of workers.")
    ap.add_argument(
        "--job_id",
        type=str,
        default=None,
        help="Unique ID for this run; used in new_records shard filenames.",
    )
    ap.add_argument(
        "--finalize_metadata",
        action="store_true",
        help="Merge renders_room/new_records/*.csv into renders_room/metadata.csv (file-lock protected).",
    )
    ap.add_argument(
        "--finalize_only",
        action="store_true",
        help="Only run metadata merge and exit without rendering.",
    )
    args = ap.parse_args()

    if args.num_views_per_m2 <= 0:
        raise ValueError(f"--num_views_per_m2 must be > 0, got: {args.num_views_per_m2}")
    if args.world_size <= 0:
        raise ValueError(f"--world_size must be >= 1, got: {args.world_size}")
    if args.rank < 0 or args.rank >= args.world_size:
        raise ValueError(f"--rank must be in [0, {args.world_size - 1}], got: {args.rank}")

    root = Path(args.root)
    rooms_processed = root / "rooms_processed"
    rooms_raw = root / "rooms_raw"
    out_root = root / "renders_room"
    new_records_dir = out_root / "new_records"
    metadata_csv = out_root / "metadata.csv"
    out_root.mkdir(parents=True, exist_ok=True)
    new_records_dir.mkdir(parents=True, exist_ok=True)

    if args.finalize_only:
        merged_count = merge_metadata_shards(out_root)
        print(f"Merged metadata rows: {merged_count} -> {metadata_csv}")
        return

    # install blender
    print("Checking blender...", flush=True)
    _install_blender()

    blender_exe = Path(BLENDER_PATH)
    blender_script = Path(BLENDER_SCRIPT)

    if not rooms_processed.is_dir():
        raise FileNotFoundError(f"Missing directory: {rooms_processed}")
    if not rooms_raw.is_dir():
        raise FileNotFoundError(f"Missing directory: {rooms_raw}")
    if not blender_exe.is_file():
        raise FileNotFoundError(f"Blender executable not found: {blender_exe}")
    if not blender_script.is_file():
        raise FileNotFoundError(f"Blender script not found: {blender_script}")

    blends = sorted(rooms_processed.glob("*.blend"))
    if not blends:
        print(f"No .blend rooms found in {rooms_processed}")
        return

    start = len(blends) * args.rank // args.world_size
    end = len(blends) * (args.rank + 1) // args.world_size
    assigned = blends[start:end]

    print(f"Rank {args.rank}/{args.world_size}: assigned {len(assigned)} rooms")

    job_id = args.job_id or _default_job_id()
    rendered_from_metadata = _load_rendered_sha256_from_metadata(metadata_csv)
    records: List[Dict[str, str]] = []
    pending: List[Path] = []

    for blend_path in assigned:
        scene_id = blend_path.stem
        transforms_json = out_root / scene_id / "transforms.json"
        if scene_id in rendered_from_metadata or transforms_json.is_file():
            records.append(
                {
                    "sha256": scene_id,
                    "local_path": f"rooms_processed/{blend_path.name}",
                    "cond_rendered": "True",
                    "status": "skipped_existing",
                    "error": "",
                    "rank": str(args.rank),
                    "world_size": str(args.world_size),
                    "job_id": job_id,
                    "updated_at": _utc_now_iso(),
                }
            )
        else:
            pending.append(blend_path)

    for blend_path in tqdm(pending, desc="Rendering rooms", unit="room"):
        scene_id = blend_path.stem
        raw_room_dir = rooms_raw / scene_id
        out_folder = out_root / scene_id
        out_folder.mkdir(parents=True, exist_ok=True)
        record = {
            "sha256": scene_id,
            "local_path": f"rooms_processed/{blend_path.name}",
            "cond_rendered": "False",
            "status": "failed",
            "error": "",
            "rank": str(args.rank),
            "world_size": str(args.world_size),
            "job_id": job_id,
            "updated_at": _utc_now_iso(),
        }
        try:
            room_json = _find_layout_json(raw_room_dir, scene_id)
            width, length, _height = _extract_room_dimensions(room_json)
            area = width * length
            density = args.num_views_per_m2 if area < 30 else args.num_views_per_m2 / 2
            num_views = min(args.max_views, max(args.min_views, int(math.ceil(area * density))))

            run_blender_render(
                blender_exe=blender_exe,
                blender_script=blender_script,
                room_blend=blend_path,
                room_json=room_json,
                out_folder=out_folder,
                num_views=num_views,
                resolution=args.resolution,
                device=args.device,
                samples=args.samples,
                engine=args.engine,
                no_denoising=args.no_denoising,
            )
            record["cond_rendered"] = "True"
            record["status"] = "rendered"
        except subprocess.CalledProcessError as e:
            record["error"] = f"blender_failed_rc_{e.returncode}"
        except Exception as e:  # noqa: BLE001
            record["error"] = str(e)
        record["updated_at"] = _utc_now_iso()
        records.append(record)

    shard_path = new_records_dir / f"part_{job_id}_r{args.rank}.csv"
    _atomic_write_csv(shard_path, records)

    success_count = sum(1 for r in records if _truthy(r.get("cond_rendered")))
    failed_count = sum(1 for r in records if not _truthy(r.get("cond_rendered")))
    print(f"Wrote shard with {len(records)} rows: {shard_path}")
    print(f"Rank {args.rank} summary: rendered_or_skipped={success_count}, failed={failed_count}")

    if args.finalize_metadata or args.world_size == 1:
        merged_count = merge_metadata_shards(out_root)
        print(f"Merged metadata rows: {merged_count} -> {metadata_csv}")

    if failed_count:
        sys.exit(2)


if __name__ == "__main__":
    main()
