import json
import os
from typing import *

import numpy as np
import torch

from .. import models
from ..modules.sparse import SparseTensor
from ..utils.render_utils import get_renderer, yaw_pitch_r_fov_to_extrinsics_intrinsics
from .components import RoomCameraConditionedMixin
from .structured_latent import SLat, SLatVisMixin


class SLatShapeVisMixin(SLatVisMixin):
    def _loading_slat_dec(self):
        if self.slat_dec is not None:
            return
        if self.slat_dec_path is not None:
            cfg = json.load(open(os.path.join(self.slat_dec_path, "config.json"), "r"))
            decoder = getattr(models, cfg["models"]["decoder"]["name"])(**cfg["models"]["decoder"]["args"])
            ckpt_path = os.path.join(self.slat_dec_path, "ckpts", f"decoder_{self.slat_dec_ckpt}.pt")
            decoder.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
        else:
            decoder = models.from_pretrained(self.pretrained_slat_dec)
        decoder.set_resolution(self.resolution)
        self.slat_dec = decoder.cuda().eval()

    @torch.no_grad()
    def visualize_sample(self, x_0: Union[SparseTensor, dict]):
        x_0 = x_0 if isinstance(x_0, SparseTensor) else x_0["x_0"]
        render_resolution = 512

        # build camera
        yaw = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaw_offset = -16 / 180 * np.pi
        yaw = [y + yaw_offset for y in yaw]
        pitch = [20 / 180 * np.pi for _ in range(4)]
        exts, ints = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, 2, 30)

        images = []

        for i in range(x_0.shape[0]):
            representation = self.decode_latent(x_0[i : i + 1].cuda(), batch_size=1)[0]
            renderer = get_renderer(representation, resolution=render_resolution, chunk_size=5000000)
            image = torch.zeros(3, render_resolution * 2, render_resolution * 2).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(representation, ext, intr)
                image[
                    :,
                    render_resolution * (j // tile[1]) : render_resolution * (j // tile[1] + 1),
                    render_resolution * (j % tile[1]) : render_resolution * (j % tile[1] + 1),
                ] = res["normal"]
            images.append(image.cpu())
            del renderer, representation, image
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        images = torch.stack(images)
        return images


class SLatShape(SLatShapeVisMixin, SLat):
    """
    structured latent for shape generation

    Args:
        roots (str): path to the dataset
        resolution (int): resolution of the shape
        min_aesthetic_score (float): minimum aesthetic score
        min_tokens (int): minimum number of tokens
        latent_key (str): key of the latent to be used
        normalization (dict): normalization stats
        pretrained_slat_dec (str): name of the pretrained slat decoder
        slat_dec_path (str): path to the slat decoder, if given, will override the pretrained_slat_dec
        slat_dec_ckpt (str): name of the slat decoder checkpoint
    """

    def __init__(
        self,
        roots: str,
        *,
        resolution: int,
        min_aesthetic_score: float = 5.0,
        min_tokens: int = 0,
        normalization: Optional[dict] = None,
        pretrained_slat_dec: str = "microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16",
        slat_dec_path: Optional[str] = None,
        slat_dec_ckpt: Optional[str] = None,
    ):
        super().__init__(
            roots,
            min_aesthetic_score=min_aesthetic_score,
            min_tokens=min_tokens,
            latent_key="shape_latent",
            normalization=normalization,
            pretrained_slat_dec=pretrained_slat_dec,
            slat_dec_path=slat_dec_path,
            slat_dec_ckpt=slat_dec_ckpt,
        )
        self.resolution = resolution


class RoomCameraConditionedSLatShape(RoomCameraConditionedMixin, SLatShape):
    """
    Image conditioned structured latent for shape generation
    """

    pass
