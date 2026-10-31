"""Select the input views of the ScanNet++ evaluation scenes.

Every method gets the same N undistorted DSLR images of a scene: training frames not flagged as
bad, picked by farthest-point sampling of the camera centres (as in T3lescope, arXiv 2610.03308),
starting at the first frame by file name. Scenes with fewer frames get all of them.

For each N and each scene in data/scannetpp/splits/nvs_sem_val.txt this writes
data/scannetpp/eval_inputs/<N>views/<scene>/dslr/: symlinks to the scene's dslr folder, except
nerfstudio/transforms_undistorted.json, which lists only the selected frames.

Usage:
    uv run scripts/select_views_scannetpp.py 32
"""

import argparse
import json
from pathlib import Path

import numpy as np

SCANNETPP = Path(__file__).resolve().parents[1] / "data/scannetpp"


def select_views(frames: list[dict], n: int) -> set[str]:
    """File paths of n frames, each new one the camera farthest from those already picked."""
    if len(frames) <= n:
        return {f["file_path"] for f in frames}

    centres = np.array([np.array(f["transform_matrix"])[:3, 3] for f in frames])
    picked = [0]
    distance = np.linalg.norm(centres - centres[0], axis=1)  # from each camera to its nearest picked one
    while len(picked) < n:
        picked.append(int(distance.argmax()))
        distance = np.minimum(distance, np.linalg.norm(centres - centres[picked[-1]], axis=1))
    return {frames[i]["file_path"] for i in picked}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("views", type=int, nargs="+", help="numbers of input views")
    args = parser.parse_args()

    for scene in (SCANNETPP / "splits/nvs_sem_val.txt").read_text().split():
        src = SCANNETPP / "data" / scene / "dslr"
        transforms = json.loads((src / "nerfstudio/transforms_undistorted.json").read_text())
        good = sorted((f for f in transforms["frames"] if not f["is_bad"]), key=lambda f: f["file_path"])

        for n in args.views:
            dslr = SCANNETPP / "eval_inputs" / f"{n}views" / scene / "dslr"
            (dslr / "nerfstudio").mkdir(parents=True, exist_ok=True)
            for entry in src.iterdir():
                if entry.name != "nerfstudio" and not (dslr / entry.name).exists():
                    (dslr / entry.name).symlink_to(entry)

            views = select_views(good, n)
            selected = {**transforms, "frames": [f for f in transforms["frames"] if f["file_path"] in views]}
            (dslr / "nerfstudio/transforms_undistorted.json").write_text(json.dumps(selected))


if __name__ == "__main__":
    main()
