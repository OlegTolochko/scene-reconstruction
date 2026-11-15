"""Meshes of a bench's validation rooms on Weights & Biases, for one run or for released GenRecon.

For every validation room of the bench (in an overfitting bench: the rooms it trains on), this runs GenRecon's
full-scene pipeline as reconstruct_scene.py does in mode Sage_gt, with its default settings and two changes:
  - the cameras are the cue's (Solv3R's, aligned to the room by prepare_cues.py) and not the true ones of
    transforms.json, as in training (cue_cameras);
  - with --run, the sparse-structure stage uses that run's aggregator (its last checkpoint: the weights val/mse
    is measured on) and gets its depth cues. Without --run nothing is replaced: released GenRecon, the reference.
The shape and texture stages are always released GenRecon's. Views, chunks, seed and sampler settings are the
same for every run, so the meshes two runs give for a room differ only by the aggregator.

Writes <bench>/<run>/meshes/<room>/mesh.glb (GenRecon's textured GLB, as chunked_to_glb.py makes it: Y-up,
simplified, without the ceiling, so that the viewer looks into the room) and logs the meshes to the run's W&B run
as mesh/<room>. The reference is the run `genrecon_released` of the same W&B group; the true rooms are the run
`ground_truth` (ground_truth.py). Rooms that already have a
mesh.glb are skipped; meshes/UPLOADED marks the upload.

From papers/GenRecon, in GenRecon's env (run.sbatch does this after training):
    python ../../experiments/genrecon_mlp_bench/mesh.py --bench OUT [--run NAME]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
GENRECON = HERE.parents[1] / "papers/GenRecon"
sys.path[:0] = [str(GENRECON), str(HERE)]

import aggregator as agg_lib  # noqa: E402
import reconstruct_scene  # noqa: E402
import variants  # noqa: E402, F401  (registers the variants)
from bench import CEILING_CUT, STEP, seed_all, start_wandb  # noqa: E402
from genrecon.pipelines import full_scene_images_to_3d as fs  # noqa: E402
from inference.chunked_glb import chunked_to_glb  # noqa: E402
from inference.get_chunks import SageGtChunker  # noqa: E402
from inference.get_images import SageImageSelecter  # noqa: E402

REFERENCE = "genrecon_released"  # the run folder (and W&B run) of released GenRecon
FACES = 300_000  # per uploaded mesh; GenRecon's meshes have millions, too many for the W&B viewer
TEXTURE = 1024  # pixels, the side of each chunk's texture: about one pixel per colour voxel (5.6 mm)
# reconstruct_scene.py's default settings (views per scene, seed, chunk overlap, occupancy threshold, ...)
DEFAULTS = reconstruct_scene.build_parser().parse_args(["--mode", "Sage_gt", "--path", "", "--output_path", ""])


class CueCameras(SageImageSelecter):
    """GenRecon's image selecter for a SAGE room, with the cue's cameras in place of the true ones. `picked`
    holds the cameras of the scene views it selected, in their order; each has its frame index."""

    def __init__(self, cue_dir: Path):
        self.cue_dir, self.picked = cue_dir, []

    def _get_cameras(self, camera_params_path):
        cameras = super()._get_cameras(camera_params_path)  # transforms.json's frames, in its order
        cue = np.load(self.cue_dir / "cameras.npz")
        assert len(cue["c2w"]) == len(cameras), f"{self.cue_dir}: {len(cue['c2w'])} cameras for {len(cameras)} frames"
        for i, cam in enumerate(cameras):
            cam["frame"] = i
            cam["c2w_blender"] = torch.from_numpy(cue["c2w"][i]).float()
            cam["intrinsics"] = torch.from_numpy(cue["K"][i]).float()
        return cameras

    def _sample_scene_cameras(self, *args, **kwargs):
        self.picked = super()._sample_scene_cameras(*args, **kwargs)
        return self.picked


class DepthCues:
    """Gives the sparse-structure aggregator its depth cues for every batch of voxels the pipeline aggregates,
    as train.py's trainer does for a training batch. It wraps the pipeline's project_features_on_points, which
    runs right before the aggregator on the same voxels."""

    def __init__(self, model, conf_mean: float, conf_std: float):
        self.model, self.conf_mean, self.conf_std = model, conf_mean, conf_std
        self.maps = None  # of the current room (set_room)
        project = fs.project_features_on_points

        def project_with_cues(proj, points, *args, **kwargs):
            if proj is model.projection:  # the other stages have their own projection and aggregator
                model.aggregator.cues = agg_lib.compute_cues(points, *self.maps, self.conf_mean, self.conf_std)
            return project(proj, points, *args, **kwargs)

        fs.project_features_on_points = project_with_cues

    def set_room(self, cue_dir: Path, frames: list[int], w2c: torch.Tensor, chunk_size: float, device) -> None:
        """frames: the scene views, as frame indices; w2c [N, 4, 4]: their cameras, from the frame of chunk 0."""
        maps = [np.load(cue_dir / f"{i:04d}.npz") for i in frames]
        stack = lambda key: torch.from_numpy(np.stack([m[key] for m in maps]).astype(np.float32))  # noqa: E731
        K = torch.from_numpy(np.load(cue_dir / "cameras.npz")["K"][frames]).float()  # of the full render, as the maps
        # GenRecon's cameras give camera coordinates in room units, the units of the cue depth: the voxel size
        # is needed in room units too.
        voxel = chunk_size / self.model.resolution
        self.maps = (*(x[None].to(device) for x in (w2c.float(), K, stack("depth"), stack("conf"))), voxel)


def last_checkpoint(run: Path) -> tuple[Path, int]:
    """The run's last checkpoint of the weights it trains (not their EMA), and its step."""
    found = sorted((run / "ckpts").glob("denoiser_step*.pt"))
    if not found:
        raise SystemExit(f"{run / 'ckpts'}: no checkpoint (denoiser_step*.pt)")
    return found[-1], int(found[-1].stem.rsplit("step", 1)[1])


def load_aggregator(model, run: Path) -> dict:
    """Puts the run's variant, with the weights of its last checkpoint, in place of the model's aggregator.
    Returns the run's run.json."""
    info = json.loads((run / "run.json").read_text())
    variant = info["variant"]
    if agg_lib.source_hash(variant) != info["code_hash"]:
        raise SystemExit(f"The code of variant {variant!r} changed since {run} was trained: its weights belong "
                         "to the old code.")
    ckpt, _ = last_checkpoint(run)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.aggregator = agg_lib.build(variant, model.aggregator)
    model.aggregator.load_state_dict({k.removeprefix("aggregator."): v for k, v in state.items()}, strict=True)
    print(f"aggregator: variant {variant}, {ckpt}", flush=True)
    return info


def reconstruct(pipeline, base: Path, room: str, cue_dir: Path, out: Path, cues: DepthCues | None):
    """GenRecon's mesh of a room (in the frame of chunk 0), the transform from that frame to the room, and the
    centres of the room's chunks [K, 3], in the room."""
    d = DEFAULTS
    scene = base / "renders_room" / room
    seed_all(d.seed, cuda=True)
    _, m_o2c, m_c2o, rel_t = SageGtChunker(min_overlap_factor=d.min_overlap_factor).get_chunks(scene, out)
    selecter = CueCameras(cue_dir)
    sel = selecter.get_images(m_o2c, scene / "transforms.json", d.num_imgs_per_scene, out, seed=d.seed)
    if cues is not None:
        cues.set_room(cue_dir, [cam["frame"] for cam in selecter.picked], sel.scene_extrinsics_c0,
                      float(m_c2o[0][0, 0]), pipeline.device)
    slat = {"boundary_sensitive": True, "boundary_width": d.boundary_width_slat} if d.boundary_sensitive_slat else {}
    mesh, _ = pipeline.run(sel, [rel_t[i] for i in sel.chunk_indices], seed=d.seed, pipeline_type=d.pipeline,
                           shape_slat_sampler_params=slat, tex_slat_sampler_params=slat,
                           occ_threshold=d.occ_threshold)
    return mesh, m_c2o[0], torch.stack([m_c2o[i][:3, 3] for i in sel.chunk_indices])


def write_glb(mesh, chunk_to_room: torch.Tensor, centres: torch.Tensor, path: Path) -> int:
    """GenRecon's textured GLB of the mesh, in room coordinates, Y-up and without the ceiling: chunked_to_glb.py
    with its default settings, but FACES and TEXTURE. It fills the holes, remeshes, and per chunk simplifies,
    unwraps and bakes the colour voxels into a texture, on a double-sided material. (Colours per vertex of a
    simplified mesh are blurred, and the faces of GenRecon's raw mesh do not consistently face outwards, so a
    viewer that hides back faces hides half of them: every floor.) Returns its number of faces."""
    size, shift = float(chunk_to_room[0, 0]), chunk_to_room[:3, 3].to(mesh.vertices)  # a scale and a shift
    vertices, faces = mesh.vertices * size + shift, mesh.faces.long()
    used, faces = torch.unique(faces[(vertices[:, 2] < CEILING_CUT)[faces].any(dim=1)], return_inverse=True)
    vertices, faces = vertices[used], faces.int()  # without the ceiling's vertices
    voxel, origin = mesh.voxel_size * size, mesh.origin * size + shift
    # The remeshing halves the voxel grid down to 32: reconstruct_scene.py pads the grid in the same way.
    grid = torch.tensor([reconstruct_scene._next_halving_friendly(int(n)) for n in mesh.voxel_shape[2:]]).to(origin)
    scene = chunked_to_glb(
        vertices_world=vertices, faces=faces, attr_volume=mesh.attrs, coords=mesh.coords, attr_layout=mesh.layout,
        aabb_world=torch.stack([origin, origin + grid * voxel]), voxel_size_world=voxel,
        chunk_centers_world=centres, chunk_size_world=size, chunk_indices=range(len(centres)),
        simplify_threshold=FACES // len(centres), texture_size=TEXTURE, verbose=False)
    scene.export(path)
    return sum(len(g.faces) for g in scene.geometry.values())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", type=Path, required=True, help="the bench folder (run.sbatch's OUT)")
    ap.add_argument("--run", help=f"a run folder of the bench; without it: released GenRecon ({REFERENCE})")
    args = ap.parse_args()

    bench = args.bench.resolve()
    out = bench / (args.run or REFERENCE)
    if (out / "meshes" / "UPLOADED").exists():
        print(f"{out.name}: meshes already uploaded")
        return
    info = json.loads((bench / "rooms.json").read_text())
    rooms = [(Path(base), room) for base, names in info["val"].items() for room in names]
    glb = {room: out / "meshes" / room / "mesh.glb" for _, room in rooms}
    todo = [(base, room) for base, room in rooms if not glb[room].exists()]

    # A run's meshes are logged at the step of its checkpoint, the reference's at step 0.
    step, config = (last_checkpoint(out)[1], {}) if args.run else (0, {"variant": "released GenRecon"})
    if todo:
        ss_ckpt = json.loads((bench / "config.json").read_text())["models"]["denoiser"]["bench"]["init_ckpt"]
        pipeline = fs.FullSceneImagesTo3DPipeline.from_finetuned(
            stage_models={"sparse_structure_flow_model": ss_ckpt,
                          f"shape_slat_flow_model_{DEFAULTS.pipeline}": "checkpoints/shape_slat/ckpts/shape_slat.pt",
                          f"tex_slat_flow_model_{DEFAULTS.pipeline}": "checkpoints/texture_slat/ckpts/texture_slat.pt"},
            pipeline_config_file="configs/pipelines/original.json",
        )
        pipeline.to(torch.device("cuda"))
        cues = None
        if args.run:
            model = pipeline.models["sparse_structure_flow_model"].to(pipeline.device)
            run_info = load_aggregator(model, out)
            cues = DepthCues(model, run_info["conf_mean"], run_info["conf_std"])
        for n, (base, room) in enumerate(todo, 1):
            print(f"=== {out.name}: room {room} ({n}/{len(todo)})", flush=True)
            glb[room].parent.mkdir(parents=True, exist_ok=True)
            mesh, chunk_to_room, centres = reconstruct(pipeline, base, room, base / info["cues"] / room,
                                                       glb[room].parent, cues)
            faces = write_glb(mesh, chunk_to_room, centres, glb[room].with_suffix(".tmp.glb"))
            glb[room].with_suffix(".tmp.glb").rename(glb[room])  # a mesh.glb is always a complete one
            print(f"{room}: {mesh.faces.shape[0]:,} faces, {faces:,} in {glb[room]}", flush=True)
            del mesh
            torch.cuda.empty_cache()

    import wandb

    run = start_wandb(out, config)
    run.log({**{f"mesh/{room}": wandb.Object3D(str(path)) for room, path in glb.items()}, STEP: step})
    run.finish()
    (out / "meshes" / "UPLOADED").write_text("")
    print(f"{out.name}: {len(glb)} meshes logged to W&B (mesh/<room>, step {step})")


if __name__ == "__main__":
    main()
