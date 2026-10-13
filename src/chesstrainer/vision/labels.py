"""Square class labels shared by the data generator, the classifier, and FEN conversion.

There are 13 classes: empty, six white pieces, six black pieces. The index order is fixed
because trained checkpoints depend on it.
"""

from __future__ import annotations

import chess

#: Class names in index order. Uppercase = white, lowercase = black (FEN convention).
CLASS_NAMES: tuple[str, ...] = (
    "empty",
    "P",
    "N",
    "B",
    "R",
    "Q",
    "K",
    "p",
    "n",
    "b",
    "r",
    "q",
    "k",
)
NUM_CLASSES = len(CLASS_NAMES)
EMPTY = 0

_SYMBOL_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES) if i != EMPTY}


def piece_to_label(piece: chess.Piece | None) -> int:
    """Map a python-chess piece (or None for an empty square) to its class index."""
    if piece is None:
        return EMPTY
    return _SYMBOL_TO_INDEX[piece.symbol()]


def label_to_piece(label: int) -> chess.Piece | None:
    """Inverse of :func:`piece_to_label`."""
    if not 0 <= label < NUM_CLASSES:
        raise ValueError(f"label out of range: {label}")
    if label == EMPTY:
        return None
    return chess.Piece.from_symbol(CLASS_NAMES[label])


def piece_asset_name(label: int) -> str:
    """Lichess asset file stem for a label, e.g. ``wN`` or ``bK``."""
    piece = label_to_piece(label)
    if piece is None:
        raise ValueError("empty squares have no asset")
    color = "w" if piece.color == chess.WHITE else "b"
    return color + piece.symbol().upper()
