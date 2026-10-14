"""Square classifier inference with OpenCV's DNN module (no PyTorch needed).

The packaged app ships the classifier as ONNX (``export_onnx.py``) because PyTorch with CUDA is
~4 GB. The network is small (~0.6M parameters), so CPU inference on 64 crops takes milliseconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from chesstrainer.vision.geometry import CROP_SIZE, ImageU8
from chesstrainer.vision.labels import CLASS_NAMES

F32 = NDArray[np.float32]


def sidecar_path(onnx_path: Path) -> Path:
    return onnx_path.with_suffix(".json")


class OnnxClassifier:
    def __init__(self, path: Path) -> None:
        meta_file = sidecar_path(path)
        meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
        names = tuple(meta.get("class_names", CLASS_NAMES))
        if names != CLASS_NAMES:
            raise ValueError(f"{path}: class order {names} != {CLASS_NAMES}")
        self.crop_size = int(meta.get("crop_size", CROP_SIZE))
        self.path = path
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def probabilities(self, crops: ImageU8) -> F32:
        """(N, H, W, 3) uint8 RGB crops -> (N, 13) softmax probabilities."""
        x = np.ascontiguousarray(crops.transpose(0, 3, 1, 2), dtype=np.float32) / 255.0
        self.net.setInput(x)
        logits = np.asarray(self.net.forward(), dtype=np.float32)
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        probs: F32 = e / e.sum(axis=1, keepdims=True)
        return probs
