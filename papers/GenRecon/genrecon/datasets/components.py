import json
import os
from abc import abstractmethod
from typing import *

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class StandardDatasetBase(Dataset):
    """
    Base class for standard datasets.

    Args:
        roots (str): paths to the dataset
    """

    def __init__(
        self,
        roots: str,
    ):
        super().__init__()
        try:
            self.roots = json.loads(roots)
            root_type = "obj"
        except:
            self.roots = roots.split(",")
            root_type = "list"
        self.instances = []
        self.metadata = pd.DataFrame()

        self._stats = {}
        if root_type == "obj":
            for key, root in self.roots.items():
                self._stats[key] = {}
                metadata = pd.DataFrame(columns=["sha256"]).set_index("sha256")
                for _, r in root.items():
                    metadata = metadata.combine_first(pd.read_csv(os.path.join(r, "metadata.csv")).set_index("sha256"))
                self._stats[key]["Total"] = len(metadata)
                self._current_root = root
                metadata, stats = self.filter_metadata(metadata)
                self._stats[key].update(stats)
                self.instances.extend([(root, sha256) for sha256 in metadata.index.values])
                self.metadata = pd.concat([self.metadata, metadata])
        else:
            for root in self.roots:
                key = os.path.basename(root)
                self._stats[key] = {}
                metadata = pd.read_csv(os.path.join(root, "metadata.csv"))
                self._stats[key]["Total"] = len(metadata)
                self._current_root = root
                metadata, stats = self.filter_metadata(metadata)
                self._stats[key].update(stats)
                self.instances.extend([(root, sha256) for sha256 in metadata["sha256"].values])
                metadata.set_index("sha256", inplace=True)
                self.metadata = pd.concat([self.metadata, metadata])

    @abstractmethod
    def filter_metadata(self, metadata: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        pass

    @abstractmethod
    def get_instance(self, root, instance: str) -> Dict[str, Any]:
        pass

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index) -> Dict[str, Any]:
        try:
            root, instance = self.instances[index]
            return self.get_instance(root, instance)
        except Exception as e:
            print(f"Error loading {instance}: {e}")
            return self.__getitem__(np.random.randint(0, len(self)))

    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f"  - Total instances: {len(self)}")
        lines.append(f"  - Sources:")
        for key, stats in self._stats.items():
            lines.append(f"    - {key}:")
            for k, v in stats.items():
                lines.append(f"      - {k}: {v}")
        return "\n".join(lines)


class RoomCameraConditionedMixin:
    def __init__(self, roots, *, image_size=512, max_num_cond_views=1, max_view_sampling_attempts=50, **kwargs):
        self.image_size = image_size
        self.max_num_cond_views = max_num_cond_views
        self.max_view_sampling_attempts = max_view_sampling_attempts
        super().__init__(roots, **kwargs)

    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)

        # check for each sha <scene_id>_<chunk_id> whether room_renders/<scene_id>.json exisits. filter metadata to only contain these chunks
        metadata = metadata.copy()
        metadata["room_rendered"] = False

        if "sha256" in metadata.columns:
            instance_ids = metadata["sha256"].astype(str)
        else:
            instance_ids = metadata.index.to_series().astype(str)
        scene_ids = instance_ids.str.extract(r"^(.*)_\d+$")[0]

        src_root = self._current_root
        render_room_root = os.path.join(src_root["base"], "renders_room")
        crops_root = os.path.join(src_root["base"], "crops")

        def has_room_assets(scene_id: str) -> bool:
            room_tf = os.path.join(render_room_root, scene_id, "transforms.json")
            scene_meta = os.path.join(crops_root, f"{scene_id}.json")
            return os.path.isfile(room_tf) and os.path.isfile(scene_meta)

        room_rendered = scene_ids.map(has_room_assets).fillna(False)
        metadata = metadata[room_rendered.to_numpy()]
        stats["Room rendered"] = len(metadata)

        return metadata, stats

    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)

        scene_id, chunk_str = instance.rsplit("_", 1)
        chunk_idx = int(chunk_str)

        image_root = os.path.join(root["base"], "renders_room", scene_id)
        with open(os.path.join(image_root, "transforms.json")) as f:
            camera_data = json.load(f)
        total_views = len(camera_data["frames"])

        scene_meta_path = os.path.join(root["base"], "crops", f"{scene_id}.json")
        with open(scene_meta_path, "r", encoding="utf-8") as f:
            scene_meta = json.load(f)
        chunk_entry = next(c for c in scene_meta["chunks"] if int(c["index"]) == chunk_idx)
        M_original_to_chunk = torch.tensor(chunk_entry["M_original_to_chunk"], dtype=torch.float32)

        n_views = np.random.randint(1, self.max_num_cond_views + 1)

        images = []
        extrinsics = []
        intrinsics = []
        attempts = 0
        view_pool = list(np.random.permutation(total_views))

        while len(images) < n_views:
            if not view_pool:
                # All views exhausted; refill to allow repeats as fallback
                view_pool = list(np.random.permutation(total_views))
            view = view_pool.pop()
            camera = camera_data["frames"][view]

            fov_x = torch.tensor(camera["camera_angle_x"])
            f_xy = 0.5 / torch.tan(fov_x / 2)
            K_norm = torch.tensor([[f_xy, 0, 0.5], [0, f_xy, 0.5], [0, 0, 1.0]])
            T_c2w_blender_room = torch.tensor(camera["transform_matrix"])

            # apply chunk transform
            T_c2w_blender_chunk = M_original_to_chunk @ T_c2w_blender_room

            # convert to OpenCV
            T_w2c_blender = torch.inverse(T_c2w_blender_chunk)
            flip = torch.diag(torch.tensor([1, -1, -1, 1], dtype=torch.float32))
            T_w2c = flip @ T_w2c_blender

            # Cheap pre-check: reject views where the chunk projects to zero area
            # in the full (uncropped) image before any PIL decode.
            if not chunk_visible_in_camera(T_w2c, K_norm, min_area=0.0) and attempts < self.max_view_sampling_attempts:
                attempts += 1
                continue

            image_path = os.path.join(image_root, camera["file_path"])
            image = Image.open(image_path)
            W, H = image.size

            # Decode RGBA once for bbox; PIL keeps the raster for subsequent ops.
            img_np = np.asarray(image)  # [H, W, 4], uint8
            alpha_np = img_np[..., 3]
            rows_any = np.any(alpha_np, axis=1)
            cols_any = np.any(alpha_np, axis=0)
            ys = np.flatnonzero(rows_any)
            xs = np.flatnonzero(cols_any)
            if ys.size == 0 or xs.size == 0:
                attempts += 1
                continue
            x0, y0, x1, y1 = int(xs[0]), int(ys[0]), int(xs[-1]), int(ys[-1])
            center = [(x0 + x1) / 2, (y0 + y1) / 2]
            hsize = max(x1 - x0, y1 - y0) / 2
            aug_bbox = [
                int(center[0] - hsize),
                int(center[1] - hsize),
                int(center[0] + hsize),
                int(center[1] + hsize),
            ]

            # adapt camera parameters based on crop
            Wc = aug_bbox[2] - aug_bbox[0]
            Hc = aug_bbox[3] - aug_bbox[1]
            K = K_norm.clone().float()
            fxn, fyn = K[0, 0], K[1, 1]
            cxn, cyn = K[0, 2], K[1, 2]
            K[0, 0] = fxn * (W / Wc)
            K[1, 1] = fyn * (H / Hc)
            K[0, 2] = (cxn * W - aug_bbox[0]) / Wc
            K[1, 2] = (cyn * H - aug_bbox[1]) / Hc
            K_norm = K

            if not chunk_visible_in_camera(T_w2c, K) and attempts < self.max_view_sampling_attempts:
                attempts += 1
                continue

            image = image.crop(aug_bbox)
            image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            arr = np.asarray(image)  # [image_size, image_size, 4]
            rgb = torch.from_numpy(np.ascontiguousarray(arr[..., :3])).permute(2, 0, 1).float() / 255.0
            alpha = torch.from_numpy(np.ascontiguousarray(arr[..., 3])).float() / 255.0
            image = rgb * alpha.unsqueeze(0)

            images.append(image)
            extrinsics.append(T_w2c)
            intrinsics.append(K_norm)
            attempts = 0

        pack["cond"] = torch.stack(images, dim=0)  # [N, 3, H, W]
        pack["intrinsics"] = torch.stack(intrinsics, dim=0)  # [N, 3, 3]
        pack["extrinsics"] = torch.stack(extrinsics, dim=0)  # [N, 4, 4]

        return pack


def chunk_visible_in_camera(T_w2c, K, min_area: float = 0.4):
    # 8 corners + center of normalized chunk cube.
    points = torch.tensor(
        [
            [-0.5, -0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, 0.5, 0.5],
            [0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, -0.5],
            [0.5, 0.5, 0.5],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    ones = torch.ones((points.shape[0], 1), dtype=torch.float32)
    pts_h = torch.cat([points, ones], dim=1).T  # [4, 9]
    cam = (T_w2c @ pts_h).T[:, :3]  # [9, 3]

    z = cam[:, 2]
    front = z > 0
    if not torch.any(front):
        return False

    cam = cam[front]
    uvw = (K @ cam.T).T  # normalized K
    z_proj = uvw[:, 2]
    finite = torch.isfinite(uvw).all(dim=-1) & (torch.abs(z_proj) > 1e-8)
    if not torch.any(finite):
        return False

    uvw = uvw[finite]
    u = uvw[:, 0] / uvw[:, 2]
    v = uvw[:, 1] / uvw[:, 2]

    # Clip to normalized image boundaries [0, 1].
    u = torch.clamp(u, 0.0, 1.0)
    v = torch.clamp(v, 0.0, 1.0)

    umin, umax = torch.min(u), torch.max(u)
    vmin, vmax = torch.min(v), torch.max(v)
    area = torch.clamp(umax - umin, min=0.0) * torch.clamp(vmax - vmin, min=0.0)
    return bool(area >= float(min_area))
