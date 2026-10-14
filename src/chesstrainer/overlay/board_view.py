"""A simple chess board widget for the review window: Unicode piece glyphs (no image assets, so
nothing licence-bound ships), coordinates, last-move tint and up to two arrows."""

from __future__ import annotations

import math

import chess
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
    QPolygonF,
)
from PyQt6.QtWidgets import QSizePolicy, QWidget

LIGHT, DARK = QColor(240, 217, 181), QColor(181, 136, 99)
TINT = QColor(205, 210, 106, 150)
GLYPHS = {
    chess.KING: "♚",
    chess.QUEEN: "♛",
    chess.ROOK: "♜",
    chess.BISHOP: "♝",
    chess.KNIGHT: "♞",
    chess.PAWN: "♟",
}
PLAYED_COLOR = QColor(220, 60, 50, 190)
BEST_COLOR = QColor(21, 120, 27, 200)


class BoardView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.board = chess.Board()
        self.white_bottom = True
        self.last_move: chess.Move | None = None
        self.arrows: list[tuple[chess.Move, QColor]] = []
        self.setMinimumSize(240, 240)
        sp = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sp.setHeightForWidth(True)
        self.setSizePolicy(sp)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt API
        return True

    def heightForWidth(self, w: int) -> int:  # noqa: N802 - Qt API
        return w

    def set_position(
        self,
        board: chess.Board,
        last_move: chess.Move | None = None,
        arrows: list[tuple[chess.Move, QColor]] | None = None,
    ) -> None:
        self.board = board.copy(stack=False)
        self.last_move = last_move
        self.arrows = arrows or []
        self.update()

    def set_orientation(self, white_bottom: bool) -> None:
        self.white_bottom = white_bottom
        self.update()

    # -- geometry ----------------------------------------------------------------------------

    def _square_rect(self, sq: chess.Square, origin: QPointF, size: float) -> QRectF:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        col, row = (f, 7 - r) if self.white_bottom else (7 - f, r)
        return QRectF(origin.x() + col * size, origin.y() + row * size, size, size)

    def paintEvent(self, event: QPaintEvent | None) -> None:  # noqa: N802 - Qt API
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        side = min(self.width(), self.height())
        size = side / 8
        origin = QPointF((self.width() - side) / 2, (self.height() - side) / 2)
        coord_font = QFont(self.font())
        coord_font.setPixelSize(max(8, int(size * 0.18)))
        for sq in chess.SQUARES:
            rect = self._square_rect(sq, origin, size)
            light = (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 1
            p.fillRect(rect, LIGHT if light else DARK)
            if self.last_move is not None and sq in (
                self.last_move.from_square,
                self.last_move.to_square,
            ):
                p.fillRect(rect, TINT)
        # Coordinates along the bottom and left edges.
        p.setFont(coord_font)
        for i in range(8):
            file_sq = chess.square(i, 0 if self.white_bottom else 7)
            rect = self._square_rect(file_sq, origin, size)
            p.setPen(DARK if (i % 2 == 1) == self.white_bottom else LIGHT)
            p.drawText(
                rect.adjusted(2, 0, -3, -1),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
                chess.FILE_NAMES[i],
            )
            rank_sq = chess.square(0 if self.white_bottom else 7, i)
            rect = self._square_rect(rank_sq, origin, size)
            p.setPen(LIGHT if (i % 2 == 1) == self.white_bottom else DARK)
            p.drawText(
                rect.adjusted(3, 1, 0, 0),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                chess.RANK_NAMES[i],
            )
        piece_font = QFont("Segoe UI Symbol")
        piece_font.setPixelSize(int(size * 0.8))
        for sq, piece in self.board.piece_map().items():
            self._draw_piece(p, piece, self._square_rect(sq, origin, size), piece_font)
        for move, color in self.arrows:
            self._draw_arrow(p, move, color, origin, size)
        p.end()

    def _draw_piece(self, p: QPainter, piece: chess.Piece, rect: QRectF, font: QFont) -> None:
        """Solid glyph as the body, outline glyph (U+2654..) on top for the lines."""
        solid, outline = QPainterPath(), QPainterPath()
        solid.addText(0, 0, font, GLYPHS[piece.piece_type])
        outline.addText(0, 0, font, chr(ord(GLYPHS[piece.piece_type]) - 6))
        shift = rect.center() - solid.boundingRect().center()
        solid.translate(shift)
        outline.translate(shift)
        p.setPen(Qt.PenStyle.NoPen)
        # Backing: solid + outline glyph together cover the whole silhouette.
        p.setBrush(QColor(250, 250, 250) if piece.color == chess.WHITE else QColor(225, 225, 225))
        p.drawPath(solid)
        p.drawPath(outline)
        p.setBrush(QColor(20, 20, 20))
        p.drawPath(outline if piece.color == chess.WHITE else solid)

    def _draw_arrow(
        self, p: QPainter, move: chess.Move, color: QColor, origin: QPointF, size: float
    ) -> None:
        a = self._square_rect(move.from_square, origin, size).center()
        b = self._square_rect(move.to_square, origin, size).center()
        dx, dy = b.x() - a.x(), b.y() - a.y()
        length = math.hypot(dx, dy)
        if length < 1:
            return
        ux, uy = dx / length, dy / length
        head, width = size * 0.42, size * 0.16
        tip = QPointF(b.x() - ux * size * 0.15, b.y() - uy * size * 0.15)
        base = QPointF(tip.x() - ux * head, tip.y() - uy * head)
        nx, ny = -uy, ux
        p.setPen(QPen(color, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
        p.drawLine(a, base)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        p.drawPolygon(
            QPolygonF(
                [
                    tip,
                    QPointF(base.x() + nx * head * 0.55, base.y() + ny * head * 0.55),
                    QPointF(base.x() - nx * head * 0.55, base.y() - ny * head * 0.55),
                ]
            )
        )
