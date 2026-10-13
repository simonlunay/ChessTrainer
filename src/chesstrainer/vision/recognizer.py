"""Run the square classifier on a detected board."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from chesstrainer.vision.dataset import to_tensor
from chesstrainer.vision.detect import BoardBox
from chesstrainer.vision.geometry import ImageU8, split_board
from chesstrainer.vision.model import DEFAULT_CHECKPOINT, load_checkpoint

F32 = NDArray[np.float32]


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class PieceRecognizer:
    """Classifies the 64 squares of a board. Skips inference when the board image is unchanged,
    except with ``keep_warm`` on CUDA: a laptop GPU left idle between moves clocks down and the
    first inference after a move then takes ~110 ms instead of ~20 ms, so it runs every frame."""

    def __init__(
        self, checkpoint: Path = DEFAULT_CHECKPOINT, device: str = "auto", keep_warm: bool = False
    ) -> None:
        self.device = resolve_device(device)
        self.keep_warm = keep_warm and self.device == "cuda"
        self.model, self.meta = load_checkpoint(checkpoint, self.device)
        self._last_thumb: NDArray[np.int16] | None = None
        self._last_probs: F32 | None = None

    @torch.no_grad()
    def classify_crops(self, crops: ImageU8) -> F32:
        """(N, H, W, 3) uint8 crops -> (N, 13) softmax probabilities."""
        x = to_tensor(crops).to(self.device)
        probs: F32 = torch.softmax(self.model(x), dim=1).cpu().numpy()
        return probs

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
        crops = split_board(frame, self.meta.crop_size, bbox=box.as_tuple())
        probs = self.classify_crops(crops)
        self._last_thumb, self._last_probs = thumb, probs
        return probs, True
