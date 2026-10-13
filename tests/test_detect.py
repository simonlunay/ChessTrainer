from __future__ import annotations

import numpy as np
import pytest

from chesstrainer.vision.assets import available_piece_sets
from chesstrainer.vision.dataset import train_split
from chesstrainer.vision.detect import BoardBox, BoardTracker, detect_board, detection_at, refine
from chesstrainer.vision.screens import render_screen
from chesstrainer.vision.synth import AssetLibrary

needs_assets = pytest.mark.skipif(
    not available_piece_sets(), reason="run scripts/download_assets.py first"
)

LIGHT, DARK = (238, 216, 192), (171, 122, 101)


def checkerboard(
    sq: int = 50, at: tuple[int, int] = (60, 40), size: tuple[int, int] = (640, 520)
) -> np.ndarray:
    img = np.full((size[1], size[0], 3), (40, 44, 52), np.uint8)
    x0, y0 = at
    for r in range(8):
        for c in range(8):
            img[y0 + r * sq : y0 + (r + 1) * sq, x0 + c * sq : x0 + (c + 1) * sq] = (
                LIGHT if (r + c) % 2 == 0 else DARK
            )
    return img


def test_box_metrics() -> None:
    a = BoardBox(0, 0, 80, 80)
    assert a.iou(a) == pytest.approx(1.0)
    assert a.iou(BoardBox(80, 0, 80, 80)) == 0.0
    assert a.corner_error(BoardBox(10, 0, 80, 80)) == pytest.approx(1.0)  # one square


def test_detects_plain_checkerboard_exactly() -> None:
    det = detect_board(checkerboard())
    assert det is not None
    assert det.box.corner_error(BoardBox(60, 40, 400, 400)) < 0.05
    assert det.score > 0.9
    assert np.allclose(det.light, LIGHT, atol=2) and np.allclose(det.dark, DARK, atol=2)


def test_board_filling_the_whole_image() -> None:
    img = checkerboard(sq=40, at=(0, 0), size=(320, 320))
    det = detect_board(img)
    assert det is not None and det.box.corner_error(BoardBox(0, 0, 320, 320)) < 0.05


def test_refine_snaps_rough_box_and_rejects_one_square_shift() -> None:
    img = checkerboard()
    det = refine(img, BoardBox(75, 30, 380, 420))
    assert det is not None and det.box.corner_error(BoardBox(60, 40, 400, 400)) < 0.05
    # A box shifted by one square has a dark top-left square: not a valid board orientation.
    shifted = detection_at(img, BoardBox(110, 40, 400, 400))
    assert shifted is None or shifted.score < 0.55


def test_highlighted_cells() -> None:
    img = checkerboard()
    img[40 + 6 * 50 : 40 + 7 * 50, 60 + 4 * 50 : 60 + 5 * 50] = (205, 210, 106)  # tint cell 52
    det = detect_board(img)
    assert det is not None
    assert det.highlighted_cells() == [52]


def test_no_board_in_noise() -> None:
    rng = np.random.default_rng(0)
    assert detect_board(rng.integers(0, 256, (400, 600, 3), dtype=np.uint8)) is None


def test_tracker_caches_and_redetects_after_move() -> None:
    tracker = BoardTracker(redetect_interval=0.0)
    img = checkerboard()
    for t in range(5):
        assert tracker.update(img, now=float(t)) is not None
    assert tracker.full_detections == 1
    moved = checkerboard(at=(150, 80))
    det = tracker.update(moved, now=10.0)
    assert det is not None and det.box.corner_error(BoardBox(150, 80, 400, 400)) < 0.05
    assert tracker.full_detections == 2


@needs_assets
@pytest.mark.parametrize("seed", range(6))
def test_detects_main_board_on_synthetic_screen(seed: int) -> None:
    screen = render_screen(np.random.default_rng(seed), AssetLibrary(), train_split())
    det = detect_board(screen.image)
    assert det is not None
    assert screen.box.corner_error(det.box) < 0.15
