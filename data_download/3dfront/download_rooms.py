"""Download one rendered room from the pinned 3D-FRONT MIDI test release."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import urllib.request


REVISION = "156fc5312ea613d4e92cbf5da1fddfa61bbad3b2"
BASE = f"https://huggingface.co/datasets/huanngzh/3D-Front/resolve/{REVISION}/"
ROOM = "fe6717d3-31e0-41a2-83e6-98c29c3461df/MasterBedroom-2820"


def download_room(room, output):
    house, room_name = room.split("/")
    if any(part in ("", ".", "..") for part in (house, room_name)):
        raise ValueError("Expected a room ID in house/room format")
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["room"] != room or manifest["revision"] != REVISION:
            raise ValueError("Output contains another room or release; choose a new --output")
        print(f"Using existing room: {output}")
        return
    with urllib.request.urlopen(BASE + "midi_test_room_ids.json", timeout=60) as response:
        if room not in json.load(response):
            raise ValueError(f"Not a room in the MIDI test release: {room}")

    output.mkdir(parents=True, exist_ok=True)
    files = []
    for kind in ("RENDER", "SCENE"):
        archive = f"3D-FRONT-TEST-{kind}.tar.gz"
        print(f"Reading {archive} for {room}", flush=True)
        found = False
        with urllib.request.urlopen(BASE + archive, timeout=120) as response:
            with tarfile.open(fileobj=response, mode="r|gz") as stream:
                for member in stream:
                    path = PurePosixPath(member.name)
                    if path.is_absolute() or ".." in path.parts:
                        raise ValueError(f"Unsafe archive path: {path}")
                    if len(path.parts) < 3:
                        continue
                    # This pinned release groups entries by house. Once the
                    # selected house ends, its room files have all been read.
                    if found and path.parts[1] != house:
                        break
                    if path.parts[1] != house or path.parts[2] not in (room_name, room_name + ".glb"):
                        continue
                    if not member.isfile():
                        continue
                    found = True
                    destination = output.joinpath(*path.parts)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_suffix(destination.suffix + ".part")
                    with stream.extractfile(member) as source, temporary.open("wb") as target:
                        shutil.copyfileobj(source, target)
                    temporary.replace(destination)
                    with destination.open("rb") as handle:
                        checksum = hashlib.file_digest(handle, "sha256").hexdigest()
                    files.append({"path": str(path), "bytes": member.size, "sha256": checksum})
        if not found:
            raise RuntimeError(f"Room missing from {archive}: {room}")

    render = output / f"3D-FRONT-TEST-RENDER/{room}/meta.json"
    scene = output / f"3D-FRONT-TEST-SCENE/{room}.glb"
    if not render.is_file() or not scene.is_file():
        raise RuntimeError("Required camera metadata or room mesh is missing")
    manifest_path.write_text(json.dumps({
        "dataset": "huanngzh/3D-Front", "revision": REVISION,
        "room": room, "files": files,
    }, indent=2) + "\n")
    print(f"Downloaded {room} to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room", default=ROOM, help="MIDI test room ID: house/room")
    parser.add_argument("--output", type=Path, default=Path("data/3dfront/pilot"))
    args = parser.parse_args()
    download_room(args.room, args.output)
