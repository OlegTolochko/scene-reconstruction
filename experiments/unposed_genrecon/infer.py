import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

from inputs import scene_name, select_images

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = REPO_ROOT / "papers/VGGT-Omega/checkpoints/VGGT-Omega-1B-512/model.pt"
OUTPUT_DIR = REPO_ROOT / "outputs/unposed_genrecon"  # predictions saved as <input name>.npz


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Image folder or video file.")
    parser.add_argument("--num-images", type=int, default=32, help="Evenly spaced images (or video frames) to use.")
    parser.add_argument("--output", type=Path, help="Where to save the .npz (default: outputs/unposed_genrecon/<name>.npz).")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        image_names = select_images(args.input, args.num_images, tmp_dir)
        print(f"Images: {len(image_names)}")

        model = VGGTOmega().eval()
        model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
        model = model.to(device)

        images = load_and_preprocess_images(image_names, image_resolution=512).to(device)

        with torch.inference_mode():
            predictions = model(images)

    extrinsics, intrinsics = encoding_to_camera(
        predictions["pose_enc"],
        predictions["images"].shape[-2:],
    )

    depth = predictions["depth"]
    depth_conf = predictions["depth_conf"]
    camera_and_register_tokens = predictions["camera_and_register_tokens"]
    camera_tokens = camera_and_register_tokens[:, :, :1]
    registers = camera_and_register_tokens[:, :, 1:]

    print(f"depth {tuple(depth.shape)}, extrinsics {tuple(extrinsics.shape)}, intrinsics {tuple(intrinsics.shape)}")

    def to_np(t: torch.Tensor) -> np.ndarray:
        return t[0].detach().float().cpu().numpy()  # drop the batch dimension

    output = args.output or OUTPUT_DIR / f"{scene_name(args.input)}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        output,
        image_paths=np.array(image_names),
        images=to_np(predictions["images"]),
        depth=to_np(depth),
        depth_conf=to_np(depth_conf),
        extrinsics=to_np(extrinsics),
        intrinsics=to_np(intrinsics),
        camera_tokens=to_np(camera_tokens),
        registers=to_np(registers),
    )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
