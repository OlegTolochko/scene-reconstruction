"""GenRecon's reconstruct_scene.py, with stage 1's cells replaced by the scan's visible cells.

Patches FullSceneImagesTo3DPipeline.extract_coords_from_occ_logits, GenRecon itself is not modified.
Stage 1 still runs, so the random state matches the baseline run, and only the cells handed to stage 2
change.

Runs in GenRecon's environment, from papers/GenRecon (run.py does this):
    .venv/bin/python ../../experiments/genrecon_oracle_occupancy/inject.py GT.npz <reconstruct_scene.py arguments>

GT.npz is written by occupancy.py from the baseline run of the same scene and number of views, which
has the same chunk layout.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.cwd()))  # papers/GenRecon
import reconstruct_scene  # noqa: E402
from genrecon.pipelines.full_scene_images_to_3d import FullSceneImagesTo3DPipeline  # noqa: E402

R = 32


def oracle_coords(gt, out: Path) -> list[np.ndarray]:
    """The scan's visible cells as each chunk's local [0, 32)^3 coordinates."""
    # chunk offsets as in occupancy.layout()
    chunks = json.loads((out / "chunk_transforms.json").read_text())["chunks"]
    centres = np.array([c["crop_center"] for c in chunks])
    if not np.allclose(centres, gt["centres"]):
        raise RuntimeError("the chunk layout differs from the baseline run's")
    rel = (centres - centres[0]) / chunks[0]["M_chunk_to_original"][0][0]
    offsets = np.round((rel - rel.min(axis=0)) * R).astype(int)

    scan, cells = gt["all"].astype(int), gt["visible"].astype(int)
    coords = []
    for i, offset in enumerate(offsets):
        local = cells - offset
        local = local[((local >= 0) & (local < R)).all(axis=1)]
        if len(local) == 0:
            # Stage 2 fails on a chunk without cells. Give it one: the corner of the chunk farthest from
            # the scan, preferring corners that no other chunk covers.
            corners = offset + np.array([[x, y, z] for x in (0, R - 1) for y in (0, R - 1) for z in (0, R - 1)])
            shared = [any(((c >= o) & (c < o + R)).all() for j, o in enumerate(offsets) if j != i) for c in corners]
            distance = np.array([np.abs(scan - c).max(axis=1).min() for c in corners])
            best = np.lexsort((-distance, shared))[0]
            local = corners[best:best + 1] - offset
        coords.append(local)
    return coords


def main():
    gt = np.load(sys.argv[1])
    sys.argv = [sys.argv[0], *sys.argv[2:]]  # the reconstruct_scene.py arguments
    out = Path(reconstruct_scene.build_parser().parse_args(sys.argv[1:]).output_path)

    def extract_oracle(self, aggr_occ_logit_list, *args, **kwargs):
        coords = oracle_coords(gt, out)
        print(f"[oracle] stage 2 gets {sum(map(len, coords))} scan cells", flush=True)
        # in GenRecon's format: [cells, 4] (batch 0, x, y, z) per chunk
        device = aggr_occ_logit_list[0].device
        return [torch.from_numpy(np.concatenate([np.zeros((len(c), 1), int), c], axis=1)).int().to(device)
                for c in coords]

    FullSceneImagesTo3DPipeline.extract_coords_from_occ_logits = extract_oracle
    reconstruct_scene.main()


if __name__ == "__main__":
    main()
