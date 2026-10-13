from __future__ import annotations

import os
from pathlib import Path

import chess
import numpy as np
import pytest
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from chesstrainer.capture.screen import MonitorInfo  # noqa: E402
from chesstrainer.overlay.board_overlay import (  # noqa: E402
    Arrow,
    arrow_endpoints,
    arrow_polygon,
    style_for,
)
from chesstrainer.overlay.gif_recorder import GifRecorder  # noqa: E402
from chesstrainer.overlay.screen_map import ScreenMap, square_center, window_rect  # noqa: E402
from chesstrainer.vision.detect import BoardBox  # noqa: E402

PRIMARY = MonitorInfo(1, 0, 0, 2560, 1600)


def test_physical_to_logical_at_150_percent() -> None:
    sm = ScreenMap(PRIMARY, 1.5, (0.0, 0.0))
    assert sm.to_logical(BoardBox(300, 150, 840, 840)) == (200.0, 100.0, 560.0, 560.0)
    assert sm.to_physical(200.0, 100.0) == (300.0, 150.0)


def test_secondary_monitor_scales_about_its_own_origin() -> None:
    # A 100% monitor left of a 150% primary: Qt keeps each screen's native top-left.
    sm = ScreenMap(MonitorInfo(2, -1920, 0, 1920, 1080), 1.0, (-1920.0, 0.0))
    assert sm.to_logical(BoardBox(100, 50, 400, 400))[:2] == (-1820.0, 50.0)


@pytest.mark.parametrize("x", [100.0, 101.0, 333.33, 77.7, -12.4])
def test_window_rect_lands_on_whole_physical_pixels(x: float) -> None:
    (wx, wy, ww, wh), (ox, oy) = window_rect((x, x + 3.1, 373.5, 373.5), dpr=1.5)
    assert (wx * 1.5) % 1 == 0 and (wy * 1.5) % 1 == 0
    assert wx + ox == pytest.approx(x) and wy + oy == pytest.approx(x + 3.1)
    assert wx + ww >= x + 373.5 and wy + wh >= x + 3.1 + 373.5
    assert 0 < ox < 6 and 0 < oy < 6


def test_square_centres_follow_orientation() -> None:
    assert square_center(chess.A1, True) == (0.5, 7.5)
    assert square_center(chess.H8, True) == (7.5, 0.5)
    assert square_center(chess.A1, False) == (7.5, 0.5)
    assert square_center(chess.E2, False) == (3.5, 1.5)


def test_arrow_points_from_origin_to_destination() -> None:
    arrow = Arrow.from_move(chess.Move.from_uci("e2e4"), 1)
    start, tip = arrow_endpoints(arrow, white_bottom=True)
    assert start[0] == tip[0] == 4.5
    assert 6.0 < start[1] < 6.5 and 4.5 < tip[1] < 4.7  # e2 centre 6.5, e4 centre 4.5
    poly = arrow_polygon(start, tip, style_for(1))
    assert len(poly) == 7 and poly[3] == tip
    xs = [p[0] for p in poly]
    assert max(xs) - 4.5 == pytest.approx(4.5 - min(xs))  # symmetric about the axis
    assert arrow_polygon((1.0, 1.0), (1.0, 1.0), style_for(1)) == []


def test_best_arrow_is_boldest() -> None:
    assert style_for(1).shaft > style_for(2).shaft >= style_for(3).shaft
    assert style_for(1).alpha > style_for(2).alpha > style_for(3).alpha
    assert style_for(7) == style_for(3)


def test_gif_recorder_writes_animation(tmp_path: Path) -> None:
    out = tmp_path / "demo.gif"
    rec = GifRecorder(1, (0, 0, 100, 100), out_path=out, fps=5)
    for shade in (0, 120, 240):
        rec.frames.append(Image.fromarray(np.full((60, 80, 3), shade, np.uint8)))
    assert rec._save() == out
    with Image.open(out) as gif:
        assert gif.n_frames == 3 and gif.size == (80, 60)


def test_overlay_widget_paints_offscreen() -> None:
    from PyQt6.QtGui import QImage
    from PyQt6.QtWidgets import QApplication

    from chesstrainer.overlay.board_overlay import BoardOverlay

    app = QApplication.instance() or QApplication([])  # noqa: F841
    overlay = BoardOverlay()
    overlay.place((10.0, 20.0, 400.0, 400.0), 1.0, (0.0, 0.0))
    overlay.set_arrows([Arrow.from_move(chess.Move.from_uci("e2e4"), 1)], True)
    img = QImage(overlay.size(), QImage.Format.Format_ARGB32)
    img.fill(0)
    overlay.render(img)
    # The e-file arrow (centre x = 4.5 squares = 225 px + 2 px margin) is painted, corners not.
    assert img.pixelColor(227, 2 + 275).alpha() > 0
    assert img.pixelColor(30, 30).alpha() == 0
