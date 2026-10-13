"""Drag-to-select fallback for when automatic board detection fails.

Shows a full-screen snapshot of the monitor, dimmed, and lets the user drag a rectangle over the
board. Coordinates are taken from the snapshot image itself (image pixels = mss physical pixels),
so the result is correct at any display scaling. Enter/release confirms, Esc cancels.
"""

from __future__ import annotations

from PyQt6.QtCore import QPoint, QRect, Qt, QThread
from PyQt6.QtGui import (
    QColor,
    QGuiApplication,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
)
from PyQt6.QtWidgets import QApplication, QWidget

from chesstrainer.capture.screen import ScreenCapture
from chesstrainer.vision.geometry import ImageU8


class _Selector(QWidget):
    def __init__(self, image: ImageU8) -> None:
        super().__init__()
        h, w = image.shape[:2]
        self._bytes = image.tobytes()  # QImage does not copy: keep the buffer alive
        self.image = QImage(self._bytes, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.start: QPoint | None = None
        self.end: QPoint | None = None
        self.result: tuple[int, int, int, int] | None = None
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def _to_image(self, p: QPoint) -> tuple[int, int]:
        sx = self.image.width() / max(1, self.width())
        sy = self.image.height() / max(1, self.height())
        return round(p.x() * sx), round(p.y() * sy)

    def paintEvent(self, event: QPaintEvent | None) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.drawImage(self.rect(), self.image)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 110))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(20, 30, "Drag a rectangle around the chess board. Esc cancels.")
        if self.start and self.end:
            r = QRect(self.start, self.end).normalized()
            painter.drawImage(r, self.image, QRect(*self._image_rect(r)))
            painter.setPen(QPen(QColor(0, 200, 255), 2))
            painter.drawRect(r)
        painter.end()

    def _image_rect(self, r: QRect) -> tuple[int, int, int, int]:
        x0, y0 = self._to_image(r.topLeft())
        x1, y1 = self._to_image(r.bottomRight())
        return x0, y0, x1 - x0, y1 - y0

    def mousePressEvent(self, event: QMouseEvent | None) -> None:  # noqa: N802
        if event is not None:
            self.start = self.end = event.position().toPoint()
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:  # noqa: N802
        if event is not None and self.start is not None:
            self.end = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:  # noqa: N802
        if self.start and self.end:
            x, y, w, h = self._image_rect(QRect(self.start, self.end).normalized())
            if w >= 32 and h >= 32:
                side = max(w, h)  # boards are square
                self.result = (x, y, side, side)
                self.close()

    def keyPressEvent(self, event: QKeyEvent | None) -> None:  # noqa: N802
        if event is not None and event.key() == Qt.Key.Key_Escape:
            self.close()


def select_region(monitor: int = 1) -> tuple[int, int, int, int] | None:
    """Let the user drag a box over the board. Returns ``(x, y, w, h)`` in monitor pixels."""
    capture = ScreenCapture(monitor)
    image = capture.grab()
    app = QApplication.instance() or QApplication([])  # noqa: F841 - must outlive the selector
    selector = _Selector(image)
    # Show on the Qt screen that matches the captured monitor (same order by position).
    mons = sorted(ScreenCapture.monitors()[1:], key=lambda m: (m.left, m.top))
    screens = sorted(QGuiApplication.screens(), key=lambda s: (s.geometry().x(), s.geometry().y()))
    target = next((i for i, m in enumerate(mons) if m.index == capture.monitor.index), 0)
    if monitor != 0 and target < len(screens):
        selector.setGeometry(screens[target].geometry())
    selector.showFullScreen()
    selector.activateWindow()
    while selector.isVisible():
        QApplication.processEvents()
        QThread.msleep(10)
    return selector.result
