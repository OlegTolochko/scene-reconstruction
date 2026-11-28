"""GenRecon's reconstruct_scene.py, with the cells stage 2 gets taken from cells.py instead of stage 1.

Runs in GenRecon's env from papers/GenRecon; experiment.py starts it:
    .venv/bin/python ../../experiments/genrecon/inject.py SPEC.json <reconstruct_scene.py arguments>

SPEC.json says where the cells come from (cells.py). GenRecon itself is not modified: this replaces
FullSceneImagesTo3DPipeline.extract_coords_from_occ_logits, which hands stage 1's cells to stage 2. By then
GenRecon has placed its chunks and written chunk_transforms.json, which gives the grid. When stage 1's cells
are not used at all, stage 1 (its sampling, the only one on a dense grid, and its decoding) is skipped.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(Path.cwd()), str(HERE.parents[1]), str(HERE)]  # GenRecon, the repo (evaluation/), cells.py
import cells as sources  # noqa: E402
import reconstruct_scene  # noqa: E402
from genrecon.pipelines.full_scene_images_to_3d import FullSceneImagesTo3DPipeline  # noqa: E402
from genrecon.pipelines.samplers.multi_diff_orchestrator import MultiDiffusionOrchestrator  # noqa: E402


def main():
    spec = json.loads(Path(sys.argv[1]).read_text())
    sys.argv = [sys.argv[0], *sys.argv[2:]]  # the reconstruct_scene.py arguments
    out = Path(reconstruct_scene.build_parser().parse_args(sys.argv[1:]).output_path)

    if not spec["keep_unseen"]:  # stage 1's cells are not used: skip its sampling and decoding
        sample = MultiDiffusionOrchestrator.sample

        def skip_dense(self, model, noise, *args, **kwargs):
            if isinstance(noise[0], torch.Tensor) and noise[0].dim() == 5:
                print("[cells] stage 1 skipped", flush=True)
                return noise
            return sample(self, model, noise, *args, **kwargs)

        MultiDiffusionOrchestrator.sample = skip_dense
        FullSceneImagesTo3DPipeline.joint_decode_sparse_structure = (
            lambda self, z_s_list, *args, **kwargs: [torch.zeros(1, device=z.device) for z in z_s_list])

    extract = FullSceneImagesTo3DPipeline.extract_coords_from_occ_logits

    def extract_from_source(self, aggr_occ_logit_list, *args, **kwargs):
        grid = sources.layout(out)  # every chunk reaches stage 2, so chunk i here is chunk i of the grid
        if spec["cells"] == "scan":
            cells = sources.scan_cells(spec, grid, out)
        else:
            stage1 = None
            if spec["keep_unseen"]:
                coords = extract(self, aggr_occ_logit_list, *args, **kwargs)
                stage1 = sources.from_chunk_coords([c[:, 1:].cpu().numpy() for c in coords], grid)
            cells = sources.depth_cells(spec, grid, stage1)
        print(f"[cells] stage 2 gets {len(cells)} cells from {spec['cells']}", flush=True)
        device = aggr_occ_logit_list[0].device
        # GenRecon's format: [cells, 4] (batch 0, x, y, z) per chunk
        return [torch.from_numpy(np.concatenate([np.zeros((len(c), 1), int), c], axis=1)).int().to(device)
                for c in sources.chunk_coords(cells, grid)]

    FullSceneImagesTo3DPipeline.extract_coords_from_occ_logits = extract_from_source
    reconstruct_scene.main()


if __name__ == "__main__":
    main()
