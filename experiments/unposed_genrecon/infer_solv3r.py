"""Solv3R's cameras and depth for the same images as infer.py, saved with the fields convert.py reads.

Runs in Solv3R's environment. Differences from infer.py's VGGT-Omega output:
    - Solv3R returns camera-to-world poses; they are inverted to world-to-camera here
    - the cameras are fitted to Solv3R's predicted rays, as in Solv3R's paper
    - depth is at Solv3R's resolution, 504 pixels on the long side (convert.py rescales K to the image size)
    - there are no camera_tokens / registers (VGGT-Omega only; convert.py does not use them)
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch

from solv3r import Solv3R, cameras_from_rays, load_images

from inputs import scene_name, select_images

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = REPO_ROOT / "papers/Solv3R/checkpoints/solv3r.safetensors"
OUTPUT_DIR = REPO_ROOT / "outputs/unposed_genrecon"  # predictions saved as <input name>_solv3r.npz
CALLS = 16  # solver calls, as in Solv3R's paper (Table 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Image folder or video file.")
    parser.add_argument("--num-images", type=int, default=32, help="Evenly spaced images (or video frames) to use.")
    parser.add_argument("--output", type=Path, help="Where to save the .npz (default: outputs/unposed_genrecon/<name>_solv3r.npz).")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp_dir:
        image_names = select_images(args.input, args.num_images, tmp_dir)
        print(f"Images: {len(image_names)}")
        images = load_images(image_names)  # (S, 3, H, W) in [0, 1]

    model = Solv3R.from_pretrained(str(CHECKPOINT))
    ctx = model.encode(images)  # the first image defines the world frame, as for VGGT-Omega
    predictions = model.decode(model.solve(ctx, CALLS), ctx)
    cam_to_world, intrinsics = cameras_from_rays(predictions["rays"], predictions["depth_conf"])
    extrinsics = torch.linalg.inv(cam_to_world.double())[:, :3]  # world-to-camera [R | t], OpenCV

    def to_np(t: torch.Tensor) -> np.ndarray:
        return t.detach().float().cpu().numpy()

    print(f"depth {tuple(predictions['depth'].shape)}, extrinsics {tuple(extrinsics.shape)}, intrinsics {tuple(intrinsics.shape)}")
    output = args.output or OUTPUT_DIR / f"{scene_name(args.input)}_solv3r.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        image_paths=np.array(image_names),
        images=to_np(images),
        depth=to_np(predictions["depth"]),
        depth_conf=to_np(predictions["depth_conf"]),
        extrinsics=to_np(extrinsics),
        intrinsics=to_np(intrinsics),
    )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
