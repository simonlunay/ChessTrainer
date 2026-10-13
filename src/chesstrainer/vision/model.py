"""Small CNN that classifies a square crop into one of 13 classes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from chesstrainer.vision.geometry import CROP_SIZE
from chesstrainer.vision.labels import CLASS_NAMES, NUM_CLASSES

MODEL_DIR = Path(__file__).resolve().parents[3] / "data" / "models"
DEFAULT_CHECKPOINT = MODEL_DIR / "square_classifier.pt"


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class SquareClassifier(nn.Module):
    """VGG-style CNN, ~0.6M parameters. Input: float RGB in [0, 1], shape (N, 3, 64, 64)."""

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(
        self, num_classes: int = NUM_CLASSES, width: int = 32, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.25, 0.25, 0.25]).view(1, 3, 1, 1))
        w = width
        self.features = nn.Sequential(
            _block(3, w),
            nn.MaxPool2d(2),  # 32x32
            _block(w, 2 * w),
            nn.MaxPool2d(2),  # 16x16
            _block(2 * w, 4 * w),
            nn.MaxPool2d(2),  # 8x8
            _block(4 * w, 4 * w),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(4 * w, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.head(self.features(x))  # type: ignore[no-any-return]


@dataclass
class CheckpointMeta:
    class_names: tuple[str, ...] = CLASS_NAMES
    crop_size: int = CROP_SIZE
    width: int = 32
    metrics: dict[str, float] | None = None


def save_checkpoint(model: SquareClassifier, path: Path, meta: CheckpointMeta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": asdict(meta)}, path)


def load_checkpoint(
    path: Path = DEFAULT_CHECKPOINT, device: str = "cpu"
) -> tuple[SquareClassifier, CheckpointMeta]:
    blob: dict[str, Any] = torch.load(path, map_location=device, weights_only=True)
    meta_dict = dict(blob["meta"])
    meta_dict["class_names"] = tuple(meta_dict["class_names"])
    meta = CheckpointMeta(**meta_dict)
    if meta.class_names != CLASS_NAMES:
        raise ValueError(f"checkpoint class order {meta.class_names} != {CLASS_NAMES}")
    model = SquareClassifier(width=meta.width)
    model.load_state_dict(blob["state_dict"])
    model.to(device).eval()
    return model, meta
