"""Run COLMAP (structure from motion) on the ScanNet++ evaluation inputs written by scripts/select_views_scannetpp.py.

COLMAP gets the same N undistorted DSLR images as the other methods and none of their poses or
intrinsics: one shared PINHOLE camera (the images are undistorted and from one DSLR), estimated by
COLMAP. SIFT features, exhaustive matching, incremental mapping. If COLMAP splits the images into
several models, the one with the most registered images is kept.

For each scene this writes outputs/colmap/<N>views/<scene>/:
    images/          symlinks to the selected views
    database.db      features and matches
    sparse/<k>/      every model COLMAP built (binary)
    colmap/          the largest model as text: cameras.txt, images.txt (original DSLR file names),
                     points3D.txt. Arbitrary world frame and scale, as COLMAP returns it.
    summary.json     registered images, points, reprojection error, run time

Scenes that already have summary.json are skipped.

Usage:
    uv run experiments/colmap/run.py 8 32 64 96
    uv run experiments/colmap/run.py 96 --threads 8 --shard 0/2   # scenes 0, 2, 4, ... (one job of two)
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import pycolmap

ROOT = Path(__file__).resolve().parents[2]


def reconstruct(scene_in: Path, out: Path, threads: int = -1) -> dict:
    frames = json.loads((scene_in / "dslr/nerfstudio/transforms_undistorted.json").read_text())["frames"]
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    for f in frames:
        link = images / Path(f["file_path"]).name
        if not link.exists():
            link.symlink_to((scene_in / "dslr/resized_undistorted_images" / f["file_path"]).resolve())

    # Start from scratch: a partial database from a failed run would be reused otherwise.
    database, sparse = out / "database.db", out / "sparse"
    database.unlink(missing_ok=True)
    shutil.rmtree(sparse, ignore_errors=True)
    sparse.mkdir()

    start = time.time()
    pycolmap.extract_features(database, images, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=pycolmap.ImageReaderOptions(camera_model="PINHOLE"),
                              extraction_options=pycolmap.FeatureExtractionOptions(num_threads=threads))
    pycolmap.match_exhaustive(database, matching_options=pycolmap.FeatureMatchingOptions(num_threads=threads))
    models = pycolmap.incremental_mapping(database, images, sparse,
                                          options=pycolmap.IncrementalPipelineOptions(num_threads=threads))
    seconds = time.time() - start

    summary = {"views": len(frames), "models": len(models), "registered": 0, "points": 0, "seconds": seconds}
    if not models:
        return summary
    best = max(models.values(), key=lambda m: m.num_reg_images())
    (out / "colmap").mkdir(exist_ok=True)
    best.write_text(out / "colmap")
    return {
        **summary,
        "registered": best.num_reg_images(),
        "points": best.num_points3D(),
        "reprojection_error_px": best.compute_mean_reprojection_error(),
        "unregistered_images": sorted({Path(f["file_path"]).name for f in frames}
                                      - {img.name for img in best.images.values() if img.has_pose}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("views", type=int, nargs="+", help="numbers of input views")
    parser.add_argument("--threads", type=int, default=-1,
                        help="CPU threads (-1 = all cores of the machine); each SIFT thread needs ~1 GB of RAM")
    parser.add_argument("--shard", default="0/1",
                        help="I/K: only every K-th scene, starting at the I-th, to split the scenes over jobs")
    args = parser.parse_args()
    shard, shards = map(int, args.shard.split("/"))

    for n in args.views:
        for scene_in in sorted((ROOT / f"data/scannetpp/eval_inputs/{n}views").iterdir())[shard::shards]:
            out = ROOT / f"outputs/colmap/{n}views" / scene_in.name
            if (out / "summary.json").exists():
                continue
            print(f"{scene_in.name}, {n} views", flush=True)
            out.mkdir(parents=True, exist_ok=True)
            try:
                summary = reconstruct(scene_in, out, args.threads)
            except Exception as e:  # one bad scene should not stop the rest
                print(f"  failed ({e})", flush=True)
                continue
            (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(f"  {summary['registered']}/{summary['views']} registered, {summary['models']} model(s), "
                  f"{summary['points']} points, {summary['seconds']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
