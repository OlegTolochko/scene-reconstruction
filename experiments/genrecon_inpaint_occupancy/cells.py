"""What the input views' depth says about each cell of GenRecon's sparse-structure grid: occupied, free or unknown.

    occupied  the cell holds depth points from at least MIN_VIEWS views (surface_cells of genrecon_depth_occupancy)
    free      at least MIN_VIEWS views see past the cell (as carving does), and it is not occupied
    unknown   everything else: behind surfaces, outside every image, or at depth edges

inject.py pins stage 1's latent to the encoding of the occupied cells where the depth knows every cell, and the
DiT generates the rest. The sources, with depth as in genrecon_depth_occupancy:

    scan     depth rendered from the scan at the true cameras, one view per cell: the most inpainting can get
    solv3r   Solv3R's depth, scaled per view with COLMAP, two views per cell

Usage:
    uv run experiments/genrecon_inpaint_occupancy/cells.py {scan,solv3r} 32

writes outputs/genrecon_inpaint_occupancy/<source>/cells/<N>views/<scene>.npz (a scene that already has one is not
computed again) with the occupied and free cells in the scene's cell grid, the chunk centres and offsets, and prints
the mean share of each state over the cells the chunks cover.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/genrecon_oracle_occupancy"))
from occupancy import BASELINE, GT, R, layout  # noqa: E402

spec = importlib.util.spec_from_file_location("depth_cells", ROOT / "experiments/genrecon_depth_occupancy/cells.py")
depth_cells = importlib.util.module_from_spec(spec)
spec.loader.exec_module(depth_cells)

OUT = ROOT / "outputs/genrecon_inpaint_occupancy"
MODEL = {"scan": "scan", "solv3r": "solv3r_depth"}
MIN_VIEWS = {"scan": 1, "solv3r": 2}


def chunk_cells(lay: dict) -> np.ndarray:
    """All cells inside at least one chunk."""
    covered = np.zeros(lay["shape"], dtype=bool)
    for x, y, z in lay["offsets"]:
        covered[x:x + R, y:y + R, z:z + R] = True
    return np.argwhere(covered)


def label(scene: str, n: int, source: str) -> dict:
    lay = layout(BASELINE / f"{n}views" / scene)
    depths, rotations, translations, K = depth_cells.load_depths(scene, n, MODEL[source])
    occupied = depth_cells.surface_cells(depths, rotations, translations, K, lay, MIN_VIEWS[source])
    cells = chunk_cells(lay)
    free = cells[depth_cells.views_past(cells, depths, rotations, translations, K, lay) >= MIN_VIEWS[source]]
    is_occupied = np.zeros(lay["shape"], dtype=bool)
    is_occupied[tuple(occupied.T)] = True
    free = free[~is_occupied[tuple(free.T)]]
    return {"occupied": occupied.astype(np.int16), "free": free.astype(np.int16), "centres": lay["centres"],
            "offsets": lay["offsets"], "shape": np.array(lay["shape"]), "chunk_cells": np.array(len(cells))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", choices=MODEL)
    parser.add_argument("views", type=int, nargs="+", help="numbers of input views")
    parser.add_argument("--scenes", nargs="*", help="default: every scene with scan cells from occupancy.py")
    args = parser.parse_args()

    for n in args.views:
        (OUT / args.source / f"cells/{n}views").mkdir(parents=True, exist_ok=True)
        shares = []
        for scene in args.scenes or sorted(p.stem for p in (GT / f"{n}views").glob("*.npz")):
            path = OUT / args.source / f"cells/{n}views/{scene}.npz"
            if not path.exists():
                np.savez_compressed(path, **label(scene, n, args.source))
                print(f"computed {path}", flush=True)
            c = np.load(path)
            total = int(c["chunk_cells"])
            shares.append({"occupied": len(c["occupied"]) / total, "free": len(c["free"]) / total,
                           "unknown": 1 - (len(c["occupied"]) + len(c["free"])) / total})
        mean = {name: float(np.mean([s[name] for s in shares])) for name in shares[0]}
        print(f"{n} views, mean share of the chunks' cells over {len(shares)} scenes:")
        print(json.dumps(mean, indent=2))


if __name__ == "__main__":
    main()
