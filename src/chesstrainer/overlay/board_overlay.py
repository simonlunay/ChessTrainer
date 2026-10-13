"""Transparent, click-through, always-on-top window laid exactly over the detected board.

The window covers just the board (plus a 2 px margin) and is moved/resized whenever the board
moves. Everything is drawn in board units (0..8) through a painter transform, so arrows follow
any board size. Rank 1 is the engine's best move (bold green), ranks 2+ are lighter blue.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import chess
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPaintEvent, QPen, QPolygonF
from PyQt6.QtWidgets import QWidget

from chesstrainer.overlay.screen_map import square_center, window_rect


@dataclass(frozen=True)
class Arrow:
    from_sq: chess.Square
    to_sq: chess.Square
    rank: int = 1  # 1 = best
    promotion: chess.PieceType | None = None

    @classmethod
    def from_move(cls, move: chess.Move, rank: int) -> Arrow:
        return cls(move.from_square, move.to_square, rank, move.promotion)


@dataclass(frozen=True)
class ArrowStyle:
    rgb: tuple[int, int, int]
    alpha: int
    shaft: float  # shaft width, squares
    head_len: float
    head_width: float


STYLES = {
    1: ArrowStyle((21, 120, 27), 215, 0.20, 0.42, 0.50),
    2: ArrowStyle((0, 72, 186), 150, 0.13, 0.34, 0.38),
    3: ArrowStyle((0, 72, 186), 105, 0.11, 0.30, 0.33),
}
CALIBRATION_RGB = (255, 0, 255)


def style_for(rank: int) -> ArrowStyle:
    return STYLES.get(rank, STYLES[3])


def arrow_polygon(
    start: tuple[float, float], tip: tuple[float, float], style: ArrowStyle
) -> list[tuple[float, float]]:
    """Outline (board units) of a straight arrow from ``start`` to ``tip``: one polygon, so a
    translucent arrow has no darker overlap between shaft and head."""
    (x0, y0), (x1, y1) = start, tip
    length = math.hypot(x1 - x0, y1 - y0)
    if length < 1e-6:
        return []
    dx, dy = (x1 - x0) / length, (y1 - y0) / length
    nx, ny = -dy, dx
    head = min(style.head_len, 0.6 * length)
    bx, by = x1 - dx * head, y1 - dy * head  # head base on the axis
    s, h = style.shaft / 2, style.head_width / 2
    return [
        (x0 + nx * s, y0 + ny * s),
        (bx + nx * s, by + ny * s),
        (bx + nx * h, by + ny * h),
        (x1, y1),
        (bx - nx * h, by - ny * h),
        (bx - nx * s, by - ny * s),
        (x0 - nx * s, y0 - ny * s),
    ]


def arrow_endpoints(
    arrow: Arrow, white_bottom: bool
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Start a little way out of the origin square's centre (so the piece stays visible) and
    end with the tip just short of the destination square's centre."""
    (x0, y0), (x1, y1) = (
        square_center(arrow.from_sq, white_bottom),
        square_center(arrow.to_sq, white_bottom),
    )
    length = math.hypot(x1 - x0, y1 - y0)
    dx, dy = (x1 - x0) / length, (y1 - y0) / length
    return (x0 + dx * 0.22, y0 + dy * 0.22), (x1 - dx * 0.12, y1 - dy * 0.12)


class BoardOverlay(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ChessTrainer overlay")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.WindowDoesNotAcceptFocus
            | Qt.WindowType.NoDropShadowWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.arrows: list[Arrow] = []
        self.white_bottom = True
        self.show_grid = False
        self.calibration = False
        self._offset = (0.0, 0.0)
        self._board_size = (0.0, 0.0)  # logical w, h
        self._geometry: tuple[int, int, int, int] | None = None
        #: perf_counter of the last paint that drew arrows (for latency measurements).
        self.last_arrow_paint = 0.0

    def place(
        self, logical: tuple[float, float, float, float], dpr: float, origin: tuple[float, float]
    ) -> None:
        """Lay the window over the board rect given in logical coordinates (floats)."""
        geo, offset = window_rect(logical, dpr, origin)
        size = (logical[2], logical[3])
        if geo != self._geometry:
            self._geometry = geo
            self.setGeometry(*geo)
        if offset != self._offset or size != self._board_size:
            self._offset, self._board_size = offset, size
            self.update()

    def set_arrows(self, arrows: list[Arrow], white_bottom: bool) -> None:
        if arrows != self.arrows or white_bottom != self.white_bottom:
            self.arrows, self.white_bottom = arrows, white_bottom
            self.update()

    def clear(self) -> None:
        self.set_arrows([], self.white_bottom)

    def paintEvent(self, event: QPaintEvent | None) -> None:  # noqa: N802 - Qt API
        w, h = self._board_size
        if w <= 0 or h <= 0:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        p.fillRect(self.rect(), Qt.GlobalColor.transparent)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        p.translate(*self._offset)
        p.scale(w / 8.0, h / 8.0)
        if self.show_grid:
            self._paint_grid(p)
        if self.calibration:
            self._paint_calibration(p)
        # Lighter alternatives first so the best move is drawn on top.
        for arrow in sorted(self.arrows, key=lambda a: -a.rank):
            self._paint_arrow(p, arrow)
        p.end()
        if self.arrows:
            self.last_arrow_paint = time.perf_counter()

    def _paint_arrow(self, p: QPainter, arrow: Arrow) -> None:
        style = style_for(arrow.rank)
        start, tip = arrow_endpoints(arrow, self.white_bottom)
        poly = QPolygonF([QPointF(x, y) for x, y in arrow_polygon(start, tip, style)])
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(*style.rgb, style.alpha)))
        p.drawPolygon(poly)
        if arrow.promotion and arrow.promotion != chess.QUEEN:
            # Underpromotion: label the piece at the destination.
            # Fonts need whole pixel sizes, so draw in a 40x-magnified local frame.
            x, y = square_center(arrow.to_sq, self.white_bottom)
            font = QFont("Segoe UI")
            font.setPixelSize(18)
            font.setBold(True)
            p.save()
            p.translate(x, y)
            p.scale(1 / 40, 1 / 40)
            p.setFont(font)
            p.setPen(QColor(255, 255, 255))
            symbol = chess.piece_symbol(arrow.promotion).upper()
            p.drawText(QRectF(-20, -20, 40, 40), Qt.AlignmentFlag.AlignCenter, symbol)
            p.restore()

    def _paint_grid(self, p: QPainter) -> None:
        pen = QPen(QColor(255, 0, 255, 200))
        pen.setCosmetic(True)
        pen.setWidthF(1.0)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        for i in range(9):
            p.drawLine(QPointF(i, 0), QPointF(i, 8))
            p.drawLine(QPointF(0, i), QPointF(8, i))

    def _paint_calibration(self, p: QPainter) -> None:
        """Opaque dots at the 64 square centres, for measuring alignment from a screenshot."""
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(*CALIBRATION_RGB)))
        for row in range(8):
            for col in range(8):
                p.drawEllipse(QPointF(col + 0.5, row + 0.5), 0.07, 0.07)
