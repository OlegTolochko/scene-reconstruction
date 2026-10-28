# 3D-FRONT room downloads

`download_rooms.py` downloads selected rooms from the pinned processed MIDI
release `huanngzh/3D-Front`, revision
`156fc5312ea613d4e92cbf5da1fddfa61bbad3b2`. It streams archives, retains selected
files and records hashes without saving complete archives. It uses only the
Python standard library.

Run from the repository root:

```bash
# Original single-room pilot
python data_download/3dfront/download_rooms.py --pilot

# Rooms listed in an existing manifest
python data_download/3dfront/download_rooms.py --manifest data/3dfront/benchmark/protocol.json
```

A manifest contains a `scenes` list; each entry requires `room` (house/room ID)
and `data` (destination directory). Relative destinations resolve from the
working directory. Existing completed room manifests are reused.

The single-room pilot retains the original early-archive stopping rule and
512 MiB limits. Batch downloads retain the original 2 GiB read limit per archive
and 1 GiB total retained-data budget. This is a bounded downloader for the
selected rooms, not a general random-access dataset downloader.

To reproduce the frozen five-room selection and create its experiment protocol:

```bash
python -m experiments.3dfront_validation.preparation.prepare_benchmark
```

Scene selection, inference settings and evaluation settings belong to that
experiment. The downloader handles archive acquisition and extraction.
