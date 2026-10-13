from __future__ import annotations

import chess
import numpy as np
import pytest
import torch

from chesstrainer.vision.assets import available_piece_sets
from chesstrainer.vision.dataset import heldout_split, to_tensor, train_split
from chesstrainer.vision.geometry import CROP_SIZE, cell_to_square
from chesstrainer.vision.labels import NUM_CLASSES, piece_to_label
from chesstrainer.vision.model import SquareClassifier
from chesstrainer.vision.synth import (
    AssetLibrary,
    degrade,
    detection_offsets,
    random_position,
    render_scene,
    sample_crops,
)

needs_assets = pytest.mark.skipif(
    not available_piece_sets(), reason="run scripts/download_assets.py first"
)


def test_random_position_is_legal() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        board = random_position(rng)
        assert board.is_valid()
        assert len(board.pieces(chess.KING, chess.WHITE)) == 1


def test_detection_offsets_shape_and_range() -> None:
    off = detection_offsets(np.random.default_rng(0))
    assert off.shape == (64, 4)
    assert np.abs(off).max() <= 0.12 + 0.04 + 1e-9


def test_splits_are_disjoint() -> None:
    train, held = train_split(), heldout_split()
    assert not set(train.piece_sets) & set(held.piece_sets)
    assert not set(train.textures) & set(held.textures)
    assert not set(train.palettes) & set(held.palettes)


@needs_assets
@pytest.mark.parametrize("seed", range(5))
def test_scene_labels_match_position(seed: int) -> None:
    rng = np.random.default_rng(seed)
    scene = degrade(render_scene(rng, AssetLibrary(), train_split()), rng)
    expected = [
        piece_to_label(scene.board.piece_at(cell_to_square(c, scene.white_bottom)))
        for c in range(64)
    ]
    assert scene.labels.tolist() == expected
    x, y, w, h = scene.bbox
    assert x >= 0 and y >= 0
    assert x + w <= scene.image.shape[1] + 1 and y + h <= scene.image.shape[0] + 1


@needs_assets
def test_sample_crops_shapes() -> None:
    crops, labels = sample_crops(np.random.default_rng(1), AssetLibrary(), train_split())
    assert crops.dtype == np.uint8
    assert crops.shape[1:] == (CROP_SIZE, CROP_SIZE, 3)
    assert len(crops) == len(labels)
    assert labels.min() >= 0 and labels.max() < NUM_CLASSES


def test_model_forward() -> None:
    model = SquareClassifier().eval()
    x = to_tensor(np.zeros((4, CROP_SIZE, CROP_SIZE, 3), np.uint8))
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, NUM_CLASSES)
