import argparse
import os

import pandas as pd


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--ext",
        type=str,
        default="glb",
        choices=["glb", "ply"],
        help="Chunk file extension under crops/ to index (default: glb)",
    )


def get_metadata(root, ext="glb", **kwargs):
    """Build the initial metadata table for a SAGE scene-chunk dataset.

    Mirrors create_metadata_chunks.py: every file in ``<root>/crops/*.<ext>``
    becomes one row keyed by its filename stem (used as the ``sha256`` id).
    Only called by build_metadata.py when ``<root>/metadata.csv`` does not
    already exist; afterwards the existing metadata is loaded instead.
    """
    crops_dir = os.path.join(root, "crops")
    if not os.path.isdir(crops_dir):
        raise FileNotFoundError(f"Missing folder: {crops_dir}")

    records = []
    for fname in sorted(os.listdir(crops_dir)):
        if not fname.endswith(f".{ext}"):
            continue
        stem = os.path.splitext(fname)[0]
        records.append(
            {
                "sha256": stem,
                "local_path": f"crops/{stem}.{ext}",
                "aesthetic_score": 5.0,
            }
        )

    return pd.DataFrame.from_records(records, columns=["sha256", "aesthetic_score", "local_path"])
