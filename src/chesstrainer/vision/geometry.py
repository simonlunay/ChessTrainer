"""Board geometry: mapping between screen cells and chess squares, and square cropping.

Screen cells are indexed row-major from the top-left of the board image: ``cell = row * 8 + col``.
Which chess square a cell shows depends on the board orientation.
"""

from __future__ import annotations

import chess
import cv2
import numpy as np
from numpy.typing import NDArray

#: Side length (pixels) of square crops fed to the classifier.
CROP_SIZE = 64

ImageU8 = NDArray[np.uint8]


def cell_to_square(cell: int, white_bottom: bool = True) -> chess.Square:
    """Chess square displayed in screen cell ``cell`` (0 = top-left)."""
    row, col = divmod(cell, 8)
    if white_bottom:
        return chess.square(col, 7 - row)
    return chess.square(7 - col, row)


def square_to_cell(square: chess.Square, white_bottom: bool = True) -> int:
    """Inverse of :func:`cell_to_square`."""
    file, rank = chess.square_file(square), chess.square_rank(square)
    if white_bottom:
        return (7 - rank) * 8 + file
    return rank * 8 + (7 - file)


def split_board(
    board_img: ImageU8,
    crop_size: int = CROP_SIZE,
    offsets: NDArray[np.float64] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
) -> ImageU8:
    """Cut a board image into 64 square crops in screen (cell) order.

    Args:
        board_img: HxWx3 uint8 image. Without ``bbox`` its borders are the board's outer edges.
        crop_size: output side length of each crop.
        offsets: optional (64, 4) array of per-cell fractional edge offsets
            ``(dx0, dy0, dx1, dy1)`` in units of one square, used by the data generator to
            imitate imperfect board detection. ``None`` means exact squares.
        bbox: optional board region ``(x, y, width, height)`` within ``board_img``.

    Returns:
        uint8 array of shape (64, crop_size, crop_size, 3).
    """
    if bbox is None:
        bx, by = 0.0, 0.0
        bw, bh = float(board_img.shape[1]), float(board_img.shape[0])
    else:
        bx, by, bw, bh = bbox
    sq_w, sq_h = bw / 8.0, bh / 8.0
    # Pad once so jittered crops that spill past the board edge stay in bounds.
    pad = int(np.ceil(max(sq_w, sq_h))) + 1
    padded = cv2.copyMakeBorder(board_img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    out = np.empty((64, crop_size, crop_size, 3), dtype=np.uint8)
    for cell in range(64):
        row, col = divmod(cell, 8)
        x0, y0 = bx + col * sq_w, by + row * sq_h
        x1, y1 = x0 + sq_w, y0 + sq_h
        if offsets is not None:
            dx0, dy0, dx1, dy1 = offsets[cell]
            x0, x1 = x0 + dx0 * sq_w, x1 + dx1 * sq_w
            y0, y1 = y0 + dy0 * sq_h, y1 + dy1 * sq_h
        ix0, iy0 = round(x0) + pad, round(y0) + pad
        ix1, iy1 = max(round(x1) + pad, ix0 + 1), max(round(y1) + pad, iy0 + 1)
        patch = padded[iy0:iy1, ix0:ix1]
        shrinking = patch.shape[0] > crop_size or patch.shape[1] > crop_size
        interp = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
        out[cell] = cv2.resize(patch, (crop_size, crop_size), interpolation=interp)
    return out
