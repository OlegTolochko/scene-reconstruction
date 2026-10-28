"""Download selected rooms from the pinned processed MIDI test archives.

No archive or Hugging Face cache is written. The pilot selects a room in the
first house in both archives, so reading stops at the next house. It relies on
the house-grouped ordering of this pinned release, not general tar ordering.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import time
import urllib.request
import tarfile


REVISION = "156fc5312ea613d4e92cbf5da1fddfa61bbad3b2"
BASE = f"https://huggingface.co/datasets/huanngzh/3D-Front/resolve/{REVISION}/"
ROOM = "fe6717d3-31e0-41a2-83e6-98c29c3461df/MasterBedroom-2820"


class LimitedReader:
    def __init__(self, response, maximum):
        self.response, self.maximum, self.count = response, maximum, 0

    def read(self, size=-1):
        if self.count >= self.maximum:
            raise RuntimeError("Network byte limit reached")
        data = self.response.read(min(size if size >= 0 else 1024**2, self.maximum - self.count))
        self.count += len(data)
        return data


def download_pilot(root, max_bytes):
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        raise SystemExit("Manifest already exists; use the existing room or a new --output.")
    ids = json.load(urllib.request.urlopen(BASE + "midi_test_room_ids.json", timeout=60))
    if ROOM not in ids:
        raise RuntimeError("Pilot room is not in the published MIDI test list")
    house, room = ROOM.split("/")
    total = 0
    manifest = {"dataset": "huanngzh/3D-Front", "revision": REVISION,
                "room": ROOM, "midi_test_member": True, "archives": [], "files": []}
    for kind in ("RENDER", "SCENE"):
        archive = f"3D-FRONT-TEST-{kind}.tar.gz"
        began = time.monotonic()
        seen_house = False
        with urllib.request.urlopen(BASE + archive, timeout=60) as response:
            reader = LimitedReader(response, 512 * 1024**2)
            with tarfile.open(fileobj=reader, mode="r|gz") as stream:
                for member in stream:
                    path = PurePosixPath(member.name)
                    parts = path.parts
                    if path.is_absolute() or ".." in parts:
                        raise RuntimeError(f"Unsafe archive path: {path}")
                    if len(parts) < 2:
                        continue
                    if parts[1] == house:
                        seen_house = True
                    elif seen_house:
                        break
                    chosen = len(parts) >= 3 and parts[1] == house and (
                        parts[2] == room or (kind == "SCENE" and parts[2] == room + ".glb"))
                    if not chosen or not member.isfile():
                        continue
                    if total + member.size > max_bytes:
                        raise RuntimeError("Extracted-byte budget exceeded")
                    if shutil.disk_usage(root).free < member.size + 5 * 1024**3:
                        raise RuntimeError("Less than 5 GiB free-space headroom")
                    destination = root.joinpath(*parts)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.exists():
                        raise RuntimeError(f"Refusing to overwrite {destination}")
                    source = stream.extractfile(member)
                    with destination.open("wb") as output:
                        shutil.copyfileobj(source, output, length=1024**2)
                    total += member.size
                    with destination.open("rb") as handle:
                        checksum = hashlib.file_digest(handle, "sha256").hexdigest()
                    manifest["files"].append({"path": str(destination.relative_to(root)),
                                              "bytes": member.size, "sha256": checksum})
            manifest["archives"].append({"name": archive, "compressed_bytes_read": reader.count,
                                         "elapsed_seconds": round(time.monotonic() - began, 2)})
        print(f"{kind}: read {reader.count / 1024**2:.1f} MiB; retained total {total / 1024**2:.1f} MiB", flush=True)
    render = root / f"3D-FRONT-TEST-RENDER/{ROOM}"
    scene = root / f"3D-FRONT-TEST-SCENE/{ROOM}.glb"
    if not (render / "meta.json").is_file() or not scene.is_file():
        raise RuntimeError("Required metadata or combined room mesh was not found")
    manifest["retained_bytes"] = total
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Manifest: {root / 'manifest.json'}", flush=True)



def download_rooms(records):
    """Download room records with `room` and `data` fields using bounded streams."""
    pending = {r['room']: r for r in records if not (Path(r['data'])/'manifest.json').exists()}
    manifests = {room: {'dataset': 'huanngzh/3D-Front', 'revision': REVISION, 'room': room,
        'midi_test_member': True, 'files': [], 'archives': [], 'retained_bytes': 0} for room in pending}
    total = 0
    for kind in ('RENDER', 'SCENE'):
        if not pending:
            break
        archive = f'3D-FRONT-TEST-{kind}.tar.gz'
        seen = set()
        with urllib.request.urlopen(BASE+archive, timeout=120) as response:
            reader = LimitedReader(response, 2*1024**3)
            with tarfile.open(fileobj=reader, mode='r|gz') as stream:
                for member in stream:
                    p = PurePosixPath(member.name)
                    if p.is_absolute() or '..' in p.parts:
                        raise RuntimeError('Unsafe archive path')
                    if len(p.parts) < 3:
                        continue
                    room = '/'.join(p.parts[1:3])
                    if room.endswith('.glb'):
                        room = room[:-4]
                    if room not in pending:
                        if seen == set(pending):
                            break
                        continue
                    seen.add(room)
                    if not member.isfile():
                        continue
                    if total+member.size > 1024**3:
                        raise RuntimeError('1 GiB retained-data budget exceeded')
                    dest = Path(pending[room]['data']).joinpath(*p.parts)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if shutil.disk_usage(dest.parent).free < member.size+5*1024**3:
                        raise RuntimeError('Insufficient disk headroom')
                    payload = stream.extractfile(member)
                    if dest.exists():
                        # A partial earlier stream can be resumed by rereading; verify bytes.
                        digest = hashlib.sha256(payload.read()).hexdigest()
                        if hashlib.sha256(dest.read_bytes()).hexdigest() != digest:
                            raise RuntimeError(f'Existing file mismatch: {dest}')
                    else:
                        with dest.open('wb') as out:
                            shutil.copyfileobj(payload, out)
                    item = {'path': str(dest.relative_to(pending[room]['data'])), 'bytes': member.size,
                            'sha256': hashlib.sha256(dest.read_bytes()).hexdigest()}
                    manifests[room]['files'].append(item)
                    manifests[room]['retained_bytes'] += member.size
                    total += member.size
            if seen != set(pending):
                raise RuntimeError(f'Missing rooms: {set(pending)-seen}')
            for manifest in manifests.values():
                manifest['archives'].append({'name': archive, 'compressed_bytes_read': reader.count})
            print(f'{kind}: streamed {reader.count/1024**2:.1f} MiB; retained {total/1024**2:.1f} MiB', flush=True)
    for room, record in pending.items():
        root = Path(record['data'])
        meta = json.loads((root/f'3D-FRONT-TEST-RENDER/{room}/meta.json').read_text())
        if len(meta['locations']) < 9 or not (root/f'3D-FRONT-TEST-SCENE/{room}.glb').exists():
            raise RuntimeError(f'Insufficient data: {room}')
        (root/'manifest.json').write_text(json.dumps(manifests[room], indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--pilot', action='store_true', help='Download the original single-room pilot')
    selection.add_argument('--manifest', type=Path, help='JSON object containing a scenes list with room and data fields; relative data paths resolve from the working directory')
    parser.add_argument('--output', type=Path, default=Path('data/3dfront/pilot'), help='Pilot output directory')
    parser.add_argument('--max-extracted-mib', type=int, default=512, help='Pilot retained-data budget')
    args = parser.parse_args()
    if args.pilot:
        download_pilot(args.output, args.max_extracted_mib * 1024**2)
    else:
        download_rooms(json.loads(args.manifest.read_text())['scenes'])


if __name__ == '__main__':
    main()
