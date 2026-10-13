"""Side panel: eval bar, candidate moves with their lines, explanation, level, and controls."""

from __future__ import annotations

import html

import chess
from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QColor, QFont, QPainter, QPaintEvent
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from chesstrainer.engine.analysis import Analysis, Line
from chesstrainer.levels import all_levels
from chesstrainer.overlay.board_overlay import style_for

PV_PLIES = 8


class EvalBar(QWidget):
    """Vertical bar: white share from the bottom (or top when Black is at the bottom)."""

    def __init__(self) -> None:
        super().__init__()
        self.fraction = 0.5
        self.text = ""
        self.white_bottom = True
        self.setFixedWidth(26)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

    def set_value(self, fraction: float, text: str, white_bottom: bool) -> None:
        self.fraction, self.text, self.white_bottom = fraction, text, white_bottom
        self.update()

    def paintEvent(self, event: QPaintEvent | None) -> None:  # noqa: N802 - Qt API
        p = QPainter(self)
        w, h = self.width(), self.height()
        white_h = round(h * self.fraction)
        p.fillRect(0, 0, w, h, QColor(64, 61, 57))
        top = h - white_h if self.white_bottom else 0
        p.fillRect(0, top, w, white_h, QColor(240, 240, 240))
        if self.text:
            font = QFont(self.font())
            font.setPointSizeF(7.5)
            font.setBold(True)
            p.setFont(font)
            white_ahead = self.fraction >= 0.5
            at_bottom = white_ahead == self.white_bottom
            p.setPen(QColor(40, 40, 40) if white_ahead else QColor(230, 230, 230))
            r = QRectF(0, h - 20, w, 18) if at_bottom else QRectF(0, 2, w, 18)
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, self.text.lstrip("+"))
        p.end()


class SidePanel(QWidget):
    flip_clicked = pyqtSignal()
    select_clicked = pyqtSignal()
    switch_turn_clicked = pyqtSignal()
    pause_toggled = pyqtSignal(bool)
    record_toggled = pyqtSignal(bool)
    level_changed = pyqtSignal(str)  # level key
    closed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ChessTrainer")
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.resize(380, 520)

        self.eval_bar = EvalBar()
        self.status = QLabel("Starting…")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self.status.setOpenExternalLinks(True)

        self.lines: list[QLabel] = []
        cand = QGridLayout()
        cand.setColumnStretch(1, 1)
        for i in range(3):
            label = QLabel()
            label.setWordWrap(True)
            label.setTextFormat(Qt.TextFormat.RichText)
            cand.addWidget(label, i, 0, 1, 2)
            self.lines.append(label)
        self.depth_label = QLabel()
        self.depth_label.setStyleSheet("color: gray")

        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color: #8a6d00")
        self.note.hide()
        self.explanation = QLabel()
        self.explanation.setWordWrap(True)
        self.explanation.setTextFormat(Qt.TextFormat.PlainText)
        self.explanation.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.explanation.setMinimumHeight(60)
        self.explanation.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.level = QComboBox()
        for lv in all_levels():
            self.level.addItem(lv.label, lv.key)
        self.level.setToolTip("Changes how many moves are shown and how they are explained.")
        self.level.currentIndexChanged.connect(
            lambda i: self.level_changed.emit(str(self.level.itemData(i)))
        )

        self.turn_button = QPushButton("Other side to move")
        self.turn_button.setToolTip("The side to move was guessed; analyse for the other side.")
        self.turn_button.hide()
        self.turn_button.clicked.connect(self.switch_turn_clicked)
        flip = QPushButton("Flip board")
        flip.setToolTip("The board is shown from the other side (fixes a wrong orientation).")
        flip.clicked.connect(self.flip_clicked)
        select = QPushButton("Select board…")
        select.setToolTip("Drag a box around the board if it is not found automatically.")
        select.clicked.connect(self.select_clicked)
        self.pause = QPushButton("Pause")
        self.pause.setCheckable(True)
        self.pause.toggled.connect(self._on_pause)
        self.record = QPushButton("● Record GIF")
        self.record.setCheckable(True)
        self.record.setToolTip(
            "Records the board as a GIF with the overlay visible (capture hiding is turned off "
            "while recording)."
        )
        self.record.toggled.connect(self.record_toggled)

        right = QVBoxLayout()
        right.addWidget(self.status)
        right.addWidget(self.turn_button)
        right.addWidget(_rule())
        right.addLayout(cand)
        right.addWidget(self.note)
        right.addWidget(self.depth_label)
        right.addWidget(_rule())
        right.addWidget(self.explanation, 1)
        level_row = QHBoxLayout()
        level_row.addWidget(QLabel("Level:"))
        level_row.addWidget(self.level, 1)
        right.addLayout(level_row)
        buttons = QGridLayout()
        buttons.addWidget(flip, 0, 0)
        buttons.addWidget(select, 0, 1)
        buttons.addWidget(self.pause, 1, 0)
        buttons.addWidget(self.record, 1, 1)
        right.addLayout(buttons)

        root = QHBoxLayout(self)
        root.addWidget(self.eval_bar)
        root.addLayout(right, 1)
        self.clear_analysis()

    # -- updates -----------------------------------------------------------------------------

    def set_status(self, text: str, rich: bool = False) -> None:
        self.status.setTextFormat(Qt.TextFormat.RichText if rich else Qt.TextFormat.PlainText)
        self.status.setText(text)

    def set_turn_assumed(self, assumed: bool) -> None:
        self.turn_button.setVisible(assumed)

    def clear_analysis(self, note: str = "") -> None:
        for label in self.lines:
            label.clear()
        self.note.hide()
        self.depth_label.setText(note)
        self.eval_bar.set_value(0.5, "", self.eval_bar.white_bottom)

    def show_analysis(
        self,
        analysis: Analysis,
        white_bottom: bool,
        lines: tuple[Line, ...] | None = None,
        note: str = "",
    ) -> None:
        """``lines``: the candidates to list (default: all); ``note``: e.g. practical move."""
        board = chess.Board(analysis.fen)
        shown = analysis.lines if lines is None else lines
        for i, label in enumerate(self.lines):
            if i < len(shown):
                label.setText(_line_html(board, shown[i]))
            else:
                label.clear()
        self.note.setText(note)
        self.note.setVisible(bool(note))
        best = analysis.best
        self.eval_bar.set_value(best.white_win_fraction(), best.score_text(), white_bottom)
        state = "done" if analysis.final else "thinking"
        self.depth_label.setText(f"depth {analysis.depth} · {analysis.elapsed:.1f}s · {state}")

    def set_explanation(self, text: str, *, status: bool = False, tooltip: str = "") -> None:
        """Explanation text, or a grey status line ("Explaining…", errors) when ``status``."""
        self.explanation.setStyleSheet("color: gray; font-style: italic" if status else "")
        self.explanation.setText(text)
        self.explanation.setToolTip(tooltip)

    def set_level(self, key: str) -> None:
        i = self.level.findData(key)
        if i >= 0 and i != self.level.currentIndex():
            self.level.blockSignals(True)
            self.level.setCurrentIndex(i)
            self.level.blockSignals(False)

    def set_recording(self, on: bool, text: str | None = None) -> None:
        self.record.blockSignals(True)
        self.record.setChecked(on)
        self.record.blockSignals(False)
        self.record.setText(text or ("■ Stop recording" if on else "● Record GIF"))

    # -- internals ---------------------------------------------------------------------------

    def _on_pause(self, paused: bool) -> None:
        self.pause.setText("Resume" if paused else "Pause")
        self.pause_toggled.emit(paused)

    def closeEvent(self, event: QCloseEvent | None) -> None:  # noqa: N802 - Qt API
        self.closed.emit()
        super().closeEvent(event)


def _rule() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line


def _line_html(board: chess.Board, line: Line) -> str:
    style = style_for(line.rank)
    r, g, b = style.rgb
    swatch = f"<span style='color: rgb({r},{g},{b}); font-size: 14pt'>■</span>"
    san = board.san(line.move)
    pv = board.variation_san(list(line.pv[:PV_PLIES]))
    weight = "bold" if line.rank == 1 else "normal"
    return (
        f"{swatch} <span style='font-weight: {weight}; font-size: 11pt'>{html.escape(san)}</span>"
        f"&nbsp;&nbsp;<b>{line.score_text()}</b><br>"
        f"<span style='color: gray'>{html.escape(pv)}</span>"
    )
