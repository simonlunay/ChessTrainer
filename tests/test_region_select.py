from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtTest")
from PyQt6.QtCore import QPoint, Qt  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from chesstrainer.overlay.region_select import _Selector  # noqa: E402


def test_drag_maps_widget_coords_to_image_pixels() -> None:
    app = QApplication.instance() or QApplication([])  # noqa: F841
    image = np.zeros((1600, 2560, 3), np.uint8)  # physical-pixel capture (150% scaling)
    sel = _Selector(image)
    sel.resize(1707, 1067)  # logical size of the same screen
    sel.show()
    QTest.mousePress(sel, Qt.MouseButton.LeftButton, pos=QPoint(100, 200))
    QTest.mouseMove(sel, QPoint(500, 590))
    QTest.mouseRelease(sel, Qt.MouseButton.LeftButton, pos=QPoint(500, 590))
    assert sel.result is not None
    x, y, w, h = sel.result
    sx, sy = 2560 / 1707, 1600 / 1067
    assert abs(x - 100 * sx) <= 2 and abs(y - 200 * sy) <= 2
    assert w == h  # boards are square
    assert abs(w - 400 * sx) <= 3 or abs(w - 390 * sy) <= 3


def test_tiny_drag_is_ignored() -> None:
    app = QApplication.instance() or QApplication([])  # noqa: F841
    sel = _Selector(np.zeros((100, 100, 3), np.uint8))
    sel.resize(100, 100)
    sel.show()
    QTest.mousePress(sel, Qt.MouseButton.LeftButton, pos=QPoint(10, 10))
    QTest.mouseRelease(sel, Qt.MouseButton.LeftButton, pos=QPoint(15, 15))
    assert sel.result is None
