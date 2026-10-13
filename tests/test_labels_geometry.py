from __future__ import annotations

import chess
import numpy as np
import pytest

from chesstrainer.vision.geometry import cell_to_square, split_board, square_to_cell
from chesstrainer.vision.labels import (
    CLASS_NAMES,
    EMPTY,
    NUM_CLASSES,
    label_to_piece,
    piece_asset_name,
    piece_to_label,
)


def test_label_round_trip() -> None:
    assert NUM_CLASSES == 13
    assert label_to_piece(EMPTY) is None
    for label in range(1, NUM_CLASSES):
        piece = label_to_piece(label)
        assert piece is not None
        assert piece_to_label(piece) == label
        assert piece.symbol() == CLASS_NAMES[label]


def test_asset_names() -> None:
    assert piece_asset_name(piece_to_label(chess.Piece(chess.KNIGHT, chess.WHITE))) == "wN"
    assert piece_asset_name(piece_to_label(chess.Piece(chess.KING, chess.BLACK))) == "bK"
    with pytest.raises(ValueError):
        piece_asset_name(EMPTY)


@pytest.mark.parametrize("white_bottom", [True, False])
def test_cell_square_bijection(white_bottom: bool) -> None:
    squares = [cell_to_square(c, white_bottom) for c in range(64)]
    assert sorted(squares) == list(chess.SQUARES)
    for c in range(64):
        assert square_to_cell(cell_to_square(c, white_bottom), white_bottom) == c


def test_orientation_corners() -> None:
    assert cell_to_square(0, True) == chess.A8
    assert cell_to_square(63, True) == chess.H1
    assert cell_to_square(0, False) == chess.H1
    assert cell_to_square(63, False) == chess.A8


def test_split_board_exact_squares() -> None:
    # Each 10x10 square filled with its cell index -> every crop is uniform with that value.
    img = np.zeros((80, 80, 3), np.uint8)
    for cell in range(64):
        r, c = divmod(cell, 8)
        img[r * 10 : (r + 1) * 10, c * 10 : (c + 1) * 10] = cell
    crops = split_board(img, crop_size=16)
    assert crops.shape == (64, 16, 16, 3)
    for cell in range(64):
        assert np.all(crops[cell] == cell)


def test_split_board_bbox_and_offsets_stay_in_bounds() -> None:
    img = np.random.default_rng(0).integers(0, 255, (120, 100, 3), dtype=np.uint8)
    offsets = np.full((64, 4), 0.2)
    offsets[:, :2] = -0.2
    crops = split_board(img, crop_size=32, offsets=offsets, bbox=(10.0, 20.0, 80.0, 80.0))
    assert crops.shape == (64, 32, 32, 3)
