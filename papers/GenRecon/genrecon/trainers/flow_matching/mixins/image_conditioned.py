from typing import *

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import DINOv3ViTModel

from ....utils import dist_utils


class DinoV3FeatureExtractor:
    """
    Feature extractor for DINOv3 models.
    """

    def __init__(self, model_name: str, image_size=512):
        self.model_name = model_name
        self.model = DINOv3ViTModel.from_pretrained(model_name)
        self.model.eval()
        self.image_size = image_size
        self.transform = transforms.Compose(
            [
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def to(self, device):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(self.model.embeddings.patch_embeddings.weight.dtype)
        hidden_states = self.model.embeddings(image, bool_masked_pos=None)
        position_embeddings = self.model.rope_embeddings(image)

        for i, layer_module in enumerate(self.model.layer):
            hidden_states = layer_module(
                hidden_states,
                position_embeddings=position_embeddings,
            )

        return F.layer_norm(hidden_states, hidden_states.shape[-1:])

    @torch.no_grad()
    def __call__(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Extract features from the image.

        Args:
            image: A batch of images as a tensor of shape (V, C, H, W).

        Returns:
            A tensor of shape (V, T, D) where T is the number of patches and D is the feature dimension.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be (V, C, H, W)"
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")

        image = self.transform(image).cuda()
        features = self.extract_features(image)  # [V, T, D]
        return features


class CameraConditionedMixin:
    """
    Mixin for image-conditioned models.

    Args:
        image_cond_model: The image conditioning model.
    """

    def __init__(self, *args, image_cond_model: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_cond_model_config = image_cond_model
        self.image_cond_model = None  # the model is init lazily

    def _init_image_cond_model(self):
        """
        Initialize the image conditioning model.
        """
        with dist_utils.local_master_first():
            self.image_cond_model = globals()[self.image_cond_model_config["name"]](
                **self.image_cond_model_config.get("args", {})
            )
            self.image_cond_model.cuda()

    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.
        """
        if self.image_cond_model is None:
            self._init_image_cond_model()
        features = self.image_cond_model(image)  # [V, T, D]
        return features

    def _build_padded_feats(self, cond, extrinsics, intrinsics, cond_batch_idx, B):
        """
        Reconstruct padded [B, N_max, ...] tensors from flat [V, ...] tensors so that
        Projection / Aggregator receive the same shapes as before.
        Zero-padded views are masked out naturally (depth=0 → valid_depth=False in Projection).
        """
        img_feats_flat = self.encode_image(cond)  # [V, T, D]
        views_per_batch = torch.bincount(cond_batch_idx, minlength=B)
        N_max = int(views_per_batch.max())
        T, D = img_feats_flat.shape[1], img_feats_flat.shape[2]

        img_feats_padded = img_feats_flat.new_zeros(B, N_max, T, D)
        extrinsics_padded = extrinsics.new_zeros(B, N_max, 4, 4)
        intrinsics_padded = intrinsics.new_zeros(B, N_max, 3, 3)
        offset = 0
        for b in range(B):
            n_b = int(views_per_batch[b])
            img_feats_padded[b, :n_b] = img_feats_flat[offset : offset + n_b]
            extrinsics_padded[b, :n_b] = extrinsics[offset : offset + n_b]
            intrinsics_padded[b, :n_b] = intrinsics[offset : offset + n_b]
            offset += n_b
        return img_feats_padded, extrinsics_padded, intrinsics_padded

    def get_cond(self, denoiser, cond, extrinsics, intrinsics, cond_batch_idx=None, x_0=None, **kwargs):
        """
        Get the conditioning data.
        cond: [V, C, H, W] flat (with cond_batch_idx) or [B, N, C, H, W] legacy
        """
        B = x_0.shape[0]
        if cond_batch_idx is not None:
            img_feats_all, extrinsics, intrinsics = self._build_padded_feats(
                cond, extrinsics, intrinsics, cond_batch_idx, B
            )
        else:
            # legacy [B, N, C, H, W] — flatten before encoding, reshape after
            Bb, N, C, H, W = cond.shape
            img_feats_all = self.encode_image(cond.reshape(Bb * N, C, H, W))
            img_feats_all = img_feats_all.reshape(Bb, N, img_feats_all.shape[1], img_feats_all.shape[2])

        cond_2D = img_feats_all[:, 0]  # [B, T, D]
        cond_3D = denoiser.get_3D_cond(img_feats_all, extrinsics, intrinsics, x_0)
        cond_dict = {"cond_2D": cond_2D, "cond_3D": cond_3D}

        # build negative cond
        if cond_3D is None:
            neg_cond_3D = None
        elif isinstance(cond_3D, torch.Tensor):
            neg_cond_3D = torch.zeros_like(cond_3D)
        else:  # SparseTensor
            neg_cond_3D = cond_3D.replace(torch.zeros_like(cond_3D.feats))
        neg_cond_dict = {"cond_2D": torch.zeros_like(cond_2D), "cond_3D": neg_cond_3D}

        kwargs["neg_cond"] = neg_cond_dict
        cond = super().get_cond(cond_dict, **kwargs)
        return cond

    def get_inference_cond(self, denoiser, cond, extrinsics, intrinsics, cond_batch_idx=None, x_0=None, **kwargs):
        """
        Get the conditioning data for inference.
        cond: [V, C, H, W] flat (with cond_batch_idx) or [B, N, C, H, W] legacy
        """
        B = x_0.shape[0]
        if cond_batch_idx is not None:
            img_feats_all, extrinsics, intrinsics = self._build_padded_feats(
                cond, extrinsics, intrinsics, cond_batch_idx, B
            )
        else:
            Bb, N, C, H, W = cond.shape
            img_feats_all = self.encode_image(cond.reshape(Bb * N, C, H, W))
            img_feats_all = img_feats_all.reshape(Bb, N, img_feats_all.shape[1], img_feats_all.shape[2])

        cond_2D = img_feats_all[:, 0]  # [B, T, D]
        cond_3D = denoiser.get_3D_cond(img_feats_all, extrinsics, intrinsics, x_0)
        cond_dict = {"cond_2D": cond_2D, "cond_3D": cond_3D}

        # build negative cond
        if cond_3D is None:
            neg_cond_3D = None
        elif isinstance(cond_3D, torch.Tensor):
            neg_cond_3D = torch.zeros_like(cond_3D)
        else:  # SparseTensor
            neg_cond_3D = cond_3D.replace(torch.zeros_like(cond_3D.feats))
        neg_cond_dict = {"cond_2D": torch.zeros_like(cond_2D), "cond_3D": neg_cond_3D}

        kwargs["neg_cond"] = neg_cond_dict
        cond = super().get_inference_cond(cond_dict, **kwargs)
        return cond

    def vis_cond(self, cond, cond_batch_idx=None, **kwargs):
        """
        Visualize the conditioning data.
        """
        if cond_batch_idx is not None:
            # flat format: cond [V, C, H, W]
            C, H, W = cond.shape[1:]
            B = int(cond_batch_idx.max().item()) + 1
            canvas = torch.zeros(B, C, H * 2, W * 2, device=cond.device, dtype=cond.dtype)
            for b in range(B):
                views = cond[cond_batch_idx == b]
                for i in range(min(len(views), 4)):
                    r, c = divmod(i, 2)
                    canvas[b, :, r * H : (r + 1) * H, c * W : (c + 1) * W] = views[i]
        else:
            # legacy format: cond [B, N, C, H, W]
            B, N, C, H, W = cond.shape
            canvas = torch.zeros(B, C, H * 2, W * 2, device=cond.device, dtype=cond.dtype)
            for i in range(min(N, 4)):
                r, c = divmod(i, 2)
                canvas[:, :, r * H : (r + 1) * H, c * W : (c + 1) * W] = cond[:, i]

        return {"image": {"value": canvas, "type": "image"}}
