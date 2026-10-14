"""Run the square classifier on a detected board.

Two backends: PyTorch (source checkout, GPU) and ONNX through OpenCV's DNN module (the packaged
app, which does not include PyTorch). ``backend="auto"`` uses PyTorch when it is installed and the
checkpoint exists, else the ONNX model (downloaded from the GitHub Release if missing, see
``weights.py``). Both run on CPU-only machines; only PyTorch can use a CUDA GPU.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from chesstrainer.paths import ONNX_MODEL, model_dir
from chesstrainer.vision.detect import BoardBox
from chesstrainer.vision.geometry import CROP_SIZE, ImageU8, split_board

F32 = NDArray[np.float32]
Backend = Literal["auto", "torch", "onnx"]


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def resolve_device(device: str) -> str:
    if device == "auto":
        if not torch_available():
            return "cpu"
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def default_onnx() -> Path:
    """The installed ONNX model, or where it would be (see ``weights.find_model``)."""
    from chesstrainer.vision.weights import find_model

    return find_model() or model_dir() / ONNX_MODEL


def _torch_classifier(checkpoint: Path, device: str) -> tuple[Callable[[ImageU8], F32], int]:
    import torch

    from chesstrainer.vision.model import load_checkpoint

    model, meta = load_checkpoint(checkpoint, device)

    @torch.no_grad()
    def run(crops: ImageU8) -> F32:
        x = torch.from_numpy(np.ascontiguousarray(crops)).permute(0, 3, 1, 2).float().div_(255.0)
        probs: F32 = torch.softmax(model(x.to(device)), dim=1).cpu().numpy()
        return probs

    return run, meta.crop_size


class PieceRecognizer:
    """Classifies the 64 squares of a board. Skips inference when the board image is unchanged,
    except with ``keep_warm`` on CUDA: a laptop GPU left idle between moves clocks down and the
    first inference after a move then takes ~110 ms instead of ~20 ms, so it runs every frame."""

    def __init__(
        self,
        checkpoint: Path | None = None,
        device: str = "auto",
        keep_warm: bool = False,
        backend: Backend = "auto",
    ) -> None:
        if checkpoint is not None and checkpoint.suffix == ".onnx":
            backend = "onnx"
        if backend == "auto":
            pt = checkpoint or _default_checkpoint()
            backend = "torch" if torch_available() and pt is not None and pt.exists() else "onnx"
        self.backend: Backend = backend
        if backend == "torch":
            pt = checkpoint or _default_checkpoint()
            if pt is None:
                raise FileNotFoundError("PyTorch is not installed")
            self.device = resolve_device(device)
            self._run, self.crop_size = _torch_classifier(pt, self.device)
            self.model_path = pt
        else:
            from chesstrainer.vision.onnx_backend import OnnxClassifier
            from chesstrainer.vision.weights import ensure_model

            path = checkpoint or ensure_model()
            onnx = OnnxClassifier(path)
            self.device = "cpu"
            self._run, self.crop_size = onnx.probabilities, onnx.crop_size
            self.model_path = path
        self.keep_warm = keep_warm and self.device == "cuda"
        self._last_thumb: NDArray[np.int16] | None = None
        self._last_probs: F32 | None = None

    def classify_crops(self, crops: ImageU8) -> F32:
        """(N, H, W, 3) uint8 crops -> (N, 13) softmax probabilities."""
        return self._run(crops)

    def classify(self, frame: ImageU8, box: BoardBox, force: bool = False) -> tuple[F32, bool]:
        """Probabilities per screen cell, shape (64, 13), and whether inference actually ran."""
        x, y, w, h = box.as_int()
        board = frame[max(0, y) : y + h, max(0, x) : x + w]
        thumb = cv2.resize(board, (64, 64), interpolation=cv2.INTER_AREA).astype(np.int16)
        if (
            not (force or self.keep_warm)
            and self._last_probs is not None
            and self._last_thumb is not None
            and np.abs(thumb - self._last_thumb).mean() < 0.5
        ):
            return self._last_probs, False
        crops = split_board(frame, self.crop_size, bbox=box.as_tuple())
        probs = self.classify_crops(crops)
        self._last_thumb, self._last_probs = thumb, probs
        return probs, True


def _default_checkpoint() -> Path | None:
    if not torch_available():
        return None
    from chesstrainer.vision.model import DEFAULT_CHECKPOINT

    return DEFAULT_CHECKPOINT


__all__ = ["CROP_SIZE", "F32", "PieceRecognizer", "default_onnx", "resolve_device"]
