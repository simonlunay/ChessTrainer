"""Map capture coordinates (physical pixels of an mss monitor) to Qt logical coordinates.

Qt 6 on Windows scales each screen about its own top-left corner: a screen whose native
top-left is ``(ox, oy)`` with device pixel ratio ``dpr`` shows native pixel ``(px, py)`` at
logical ``(ox + (px - ox) / dpr, oy + (py - oy) / dpr)``. At 150% scaling ``dpr = 1.5``.

Board geometry is kept in "board units": the board spans 0..8 in x and y, cell (row, col)
covers ``[col, col + 1] x [row, row + 1]`` with row 0 at the top of the screen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import chess

from chesstrainer.capture.screen import MonitorInfo
from chesstrainer.vision.detect import BoardBox
from chesstrainer.vision.geometry import square_to_cell


@dataclass(frozen=True)
class ScreenMap:
    """One monitor: its native rectangle (virtual-desktop physical px) and Qt scale factor."""

    monitor: MonitorInfo
    dpr: float
    #: Logical top-left of the matching QScreen (equals the native top-left in Qt 6).
    origin: tuple[float, float] | None = None

    def to_logical(self, box: BoardBox) -> tuple[float, float, float, float]:
        """Board box in monitor-relative physical px -> logical ``(x, y, w, h)`` (floats)."""
        m = self.monitor
        ox, oy = self.origin if self.origin is not None else (float(m.left), float(m.top))
        return (
            ox + box.x / self.dpr,
            oy + box.y / self.dpr,
            box.w / self.dpr,
            box.h / self.dpr,
        )

    def to_physical(self, x: float, y: float) -> tuple[float, float]:
        """Logical point -> monitor-relative physical px (inverse of :meth:`to_logical`)."""
        m = self.monitor
        ox, oy = self.origin if self.origin is not None else (float(m.left), float(m.top))
        return (x - ox) * self.dpr, (y - oy) * self.dpr


def square_center(square: chess.Square, white_bottom: bool) -> tuple[float, float]:
    """Centre of ``square`` in board units (0..8, y down)."""
    row, col = divmod(square_to_cell(square, white_bottom), 8)
    return col + 0.5, row + 0.5


def window_rect(
    logical: tuple[float, float, float, float],
    dpr: float = 1.0,
    origin: tuple[float, float] = (0.0, 0.0),
    pad: int = 2,
) -> tuple[tuple[int, int, int, int], tuple[float, float]]:
    """Integer window geometry covering a float rect, plus the rect's offset inside it.

    Windows can only sit on whole logical pixels, so the overlay window is slightly larger than
    the board and the painter is translated by the returned offset: sub-pixel exact drawing.
    The window's top-left is also chosen to land on a whole *physical* pixel (at 150% only every
    other logical pixel does), otherwise Qt would round it by up to half a physical pixel.
    """
    x, y, w, h = logical

    def snap(v: float, o: float) -> int:
        base = math.floor(v) - pad
        for k in range(4):
            native = (base - k - o) * dpr
            if abs(native - round(native)) < 1e-6:
                return base - k
        return base

    wx, wy = snap(x, origin[0]), snap(y, origin[1])
    ww = math.ceil(x + w) + pad - wx
    wh = math.ceil(y + h) + pad - wy
    return (wx, wy, ww, wh), (x - wx, y - wy)
