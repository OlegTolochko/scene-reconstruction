import argparse
import glob
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = REPO_ROOT / "papers/VGGT-Omega/checkpoints/VGGT-Omega-1B-512/model.pt"
OUTPUT_DIR = REPO_ROOT / "outputs/unposed_genrecon"  # predictions saved as <input name>.npz
IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def extract_frames(video_path: str, num_frames: int, out_dir: str) -> list[str]:
    """Save num_frames evenly spaced frames from the video and return their paths."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    paths = []
    for i, index in enumerate(np.linspace(0, total - 1, num_frames).astype(int)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            continue
        path = os.path.join(out_dir, f"{i:04d}.png")
        cv2.imwrite(path, frame)
        paths.append(path)
    cap.release()
    return paths


def pick_evenly(paths: list[str], num_images: int) -> list[str]:
    """Keep num_images evenly spaced paths (all of them if there are fewer), like GenRecon does."""
    if len(paths) <= num_images:
        return paths
    indices = np.linspace(0, len(paths) - 1, num_images).round().astype(int)
    return [paths[i] for i in indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Image folder or video file.")
    parser.add_argument("--num-images", type=int, default=32, help="Evenly spaced images (or video frames) to use.")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        if os.path.isdir(args.input):
            all_names = sorted(p for p in glob.glob(os.path.join(args.input, "*")) if p.lower().endswith(IMAGE_EXTS))
            image_names = pick_evenly(all_names, args.num_images)
        else:
            image_names = extract_frames(args.input, args.num_images, tmp_dir)
        if not image_names:
            raise SystemExit(f"No images found in {args.input}")
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

    output = OUTPUT_DIR / f"{Path(args.input).stem}.npz"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
