"""Fetch five preselected rooms in two bounded streams, without saving archives."""
import json
from pathlib import Path
import urllib.request
import importlib

_download = importlib.import_module("data_download.3dfront.download_rooms")
BASE, REVISION, ROOM = _download.BASE, _download.REVISION, _download.ROOM
prepare_pilot, download_rooms = _download.download_pilot, _download.download_rooms

ROOT = Path(__file__).resolve().parents[3]
ROOMS = [ROOM,
    'e9b1fe99-242b-46e5-85c5-59028015310d/LivingRoom-11414',
    'e6db34b3-c6a1-4e04-b95a-30c415767e82/Library-27940',
    'fd987ee8-eeac-4d95-a947-306db2feada5/Bedroom-18956',
    'e1d87fdc-c05f-4cca-8b11-c7d002fbc36b/DiningRoom-27978']

def main():
    base = ROOT / 'data/3dfront/benchmark'
    base.mkdir(parents=True, exist_ok=True)
    records = []
    ids = json.load(urllib.request.urlopen(BASE + 'midi_test_room_ids.json', timeout=60))
    for i, room in enumerate(ROOMS):
        if room not in ids:
            raise RuntimeError(f'Not a MIDI test room: {room}')
        records.append({'id': f'scene_{i+1:02d}', 'room': room,
            'data': str(ROOT / 'data/3dfront/pilot' if i == 0 else base / f'scene_{i+1:02d}'),
            'input': str(ROOT / 'data/3dfront/genrecon_pilot' if i == 0 else base / f'scene_{i+1:02d}_input'),
            'output': str(ROOT / 'outputs/genrecon/3dfront_pilot' if i == 0 else ROOT / f'outputs/genrecon/benchmark/scene_{i+1:02d}')})
    if not (Path(records[0]['data'])/'manifest.json').exists():
        prepare_pilot(Path(records[0]['data']),512*1024**2)
    protocol = {'revision': REVISION, 'selection': 'Frozen before new predictions: existing pilot plus four early-archive rooms spanning living room, library, bedroom and dining room, each from a different house. Convenience sample, not random or representative.',
        'seed': 42, 'input_views': 8, 'surface_samples_per_mesh': 100000,
        'distance_thresholds_fraction_gt_diagonal': [.005, .01, .02],
        'ray_grid': 192, 'scenes': records}
    path = base / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise RuntimeError('Frozen protocol differs')
    path.write_text(json.dumps(protocol, indent=2)+'\n')
    download_rooms(records[1:])
    print(path, flush=True)

if __name__ == '__main__':
    main()
