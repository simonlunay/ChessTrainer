"""Torch datasets over the synthetic generator, plus fixed evaluation sets.

Training streams freshly rendered crops forever (no epoch boundary, no disk), so the model never
sees the same augmented sample twice. Validation and test sets are rendered once from fixed seeds
and saved as ``.npz`` so metrics are comparable across runs.

Splits:
    * ``train``     - training piece sets / textures / palettes.
    * ``val``       - same styles as train, different seed (in-distribution).
    * ``heldout``   - piece sets, textures and palettes never used in training (generalization).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import IterableDataset, get_worker_info

from chesstrainer.vision.assets import (
    HELD_OUT_PIECE_SETS,
    HELD_OUT_TEXTURES,
    available_piece_sets,
    available_textures,
)
from chesstrainer.vision.geometry import ImageU8
from chesstrainer.vision.synth import (
    HELD_OUT_PALETTES,
    PALETTES,
    AssetLibrary,
    StyleSplit,
    sample_crops,
)

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
EVAL_DIR = DATA_DIR / "eval"


def train_split() -> StyleSplit:
    return StyleSplit(
        piece_sets=[s for s in available_piece_sets() if s not in HELD_OUT_PIECE_SETS],
        textures=[t for t in available_textures() if t not in HELD_OUT_TEXTURES],
        palettes=[p for p in PALETTES if p not in HELD_OUT_PALETTES],
        random_colors=True,
    )


def heldout_split() -> StyleSplit:
    return StyleSplit(
        piece_sets=[s for s in available_piece_sets() if s in HELD_OUT_PIECE_SETS],
        textures=[t for t in available_textures() if t in HELD_OUT_TEXTURES],
        palettes=sorted(HELD_OUT_PALETTES),
        random_colors=False,
    )


def to_tensor(crops: ImageU8) -> torch.Tensor:
    """uint8 NHWC RGB crops -> float NCHW tensor in [0, 1] (normalization lives in the model)."""
    return torch.from_numpy(np.ascontiguousarray(crops)).permute(0, 3, 1, 2).float().div_(255.0)


class SyntheticSquares(IterableDataset[tuple[torch.Tensor, torch.Tensor]]):
    """Infinite stream of ``(crop[3,H,W], label)`` pairs from freshly rendered boards."""

    def __init__(self, split: StyleSplit, seed: int = 0, empty_keep: float = 0.3) -> None:
        self.split = split
        self.seed = seed
        self.empty_keep = empty_keep

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        info = get_worker_info()
        worker = info.id if info is not None else 0
        rng = np.random.default_rng([self.seed, worker])
        lib = AssetLibrary()
        while True:
            crops, labels = sample_crops(rng, lib, self.split, self.empty_keep)
            x = to_tensor(crops)
            y = torch.from_numpy(labels)
            for i in rng.permutation(len(labels)):
                yield x[i], y[i]


def build_eval_set(
    split: StyleSplit, n_boards: int, seed: int
) -> tuple[ImageU8, NDArray[np.int64]]:
    """Render a fixed set of crops. All 64 squares are kept so per-board accuracy is defined."""
    rng = np.random.default_rng(seed)
    lib = AssetLibrary()
    xs, ys = [], []
    for _ in range(n_boards):
        crops, labels = sample_crops(rng, lib, split, empty_keep=1.0)
        # sample_crops mirrors half the crops; harmless for evaluation.
        xs.append(crops)
        ys.append(labels)
    return np.stack(xs), np.stack(ys)  # (B, 64, H, W, 3), (B, 64)


def load_or_build_eval(name: str, n_boards: int = 300) -> tuple[ImageU8, NDArray[np.int64]]:
    """Load ``data/eval/<name>.npz`` or render and cache it. ``name`` is ``val`` or ``heldout``."""
    path = EVAL_DIR / f"{name}.npz"
    if path.exists():
        data = np.load(path)
        return data["x"], data["y"]
    split, seed = {"val": (train_split(), 10_001), "heldout": (heldout_split(), 20_002)}[name]
    x, y = build_eval_set(split, n_boards, seed)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=x, y=y)
    return x, y
