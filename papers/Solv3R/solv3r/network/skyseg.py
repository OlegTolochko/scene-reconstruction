# Adapted from DVLT (https://github.com/nv-tlabs/dvlt), Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
# Modified for Solv3R: returns the sky mask directly. The ONNX sky segmentation model is from
# https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing (downloaded from huggingface.co/JianyuanWang/skyseg).
"""Sky segmentation (onnxruntime), used to leave the sky out of outdoor point clouds."""
from __future__ import annotations

import copy
import os
import urllib.request

import numpy as np

SKYSEG_ONNX_PATH = "skyseg.onnx"
SKYSEG_URL = "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx"


def _load_session(intra_op_num_threads=4, inter_op_num_threads=1):
    """The skyseg ONNX model, downloaded to ./skyseg.onnx on first use."""
    import onnxruntime

    if not os.path.exists(SKYSEG_ONNX_PATH):
        print(f"Downloading {SKYSEG_ONNX_PATH}...")
        tmp = SKYSEG_ONNX_PATH + ".part"
        urllib.request.urlretrieve(SKYSEG_URL, tmp)
        os.replace(tmp, SKYSEG_ONNX_PATH)
    opts = onnxruntime.SessionOptions()
    opts.intra_op_num_threads = intra_op_num_threads
    opts.inter_op_num_threads = inter_op_num_threads
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return onnxruntime.InferenceSession(SKYSEG_ONNX_PATH, sess_options=opts, providers=providers)


def segment_sky(images):
    """images (S, 3, H, W) torch tensor in [0, 1] -> (S, H, W) bool, True on sky."""
    import cv2

    S, _, H, W = images.shape
    session = _load_session()
    masks = []
    for image in (images.permute(0, 2, 3, 1)[..., [2, 1, 0]] * 255).byte():   # BGR uint8
        mask = _non_sky_mask(image.cpu().numpy(), session)
        if mask.shape[0] != H or mask.shape[1] != W:
            mask = cv2.resize(mask, (W, H))
        masks.append(mask)
    return ~(np.array(masks) > 0.1)


def _non_sky_mask(image, session):
    """BGR uint8 image -> uint8 mask, 255 on non-sky pixels."""
    import cv2

    result = _run_skyseg(session, [320, 320], image)
    result = cv2.resize(result, (image.shape[1], image.shape[0]))
    mask = np.zeros_like(result)
    mask[result < 32] = 255          # the model outputs low values for non-sky
    return mask


def _run_skyseg(session, input_size, image):
    import cv2

    x = cv2.resize(copy.deepcopy(image), dsize=(input_size[0], input_size[1]))
    x = np.array(cv2.cvtColor(x, cv2.COLOR_BGR2RGB), dtype=np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x / 255 - mean) / std
    x = x.transpose(2, 0, 1).reshape(-1, 3, input_size[0], input_size[1]).astype("float32")
    out = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: x})
    out = np.array(out).squeeze()
    out = (out - np.min(out)) / (np.max(out) - np.min(out))
    out *= 255
    return out.astype("uint8")
