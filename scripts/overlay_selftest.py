"""Phase 3 self-test: the real app (vision thread + Stockfish + overlay) against a known board.

Shows a real game rendered by lichess.org in a window at a known place, runs ``TrainerApp`` on
the live screen, and measures:

* latency: from the moment a new position is put on screen to the moment its arrows are painted,
  plus how fast the previous position's (stale) arrows disappear;
* alignment: the overlay draws calibration dots on the 64 square centres; the screen is grabbed
  with the overlay visible to capture and again hidden, and the difference gives where the dots
  really landed on the monitor (physical px), compared with the true square centres;
* following: the board window is moved and resized, and the time until the overlay sits on the
  new place (and arrows are back) is measured, then alignment is measured again.

Writes data/eval/phase3_report.txt and data/eval/phase3_arrows.png (a screenshot with arrows).
Uses a throwaway settings folder, so your saved settings are not touched. Needs Stockfish.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import chess
import cv2
import numpy as np
from PIL import Image
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel

sys.path.insert(0, str(Path(__file__).parent))
from live_selftest import fetch_frames  # noqa: E402

from chesstrainer.capture.screen import ScreenCapture  # noqa: E402
from chesstrainer.overlay.app import TrainerApp  # noqa: E402
from chesstrainer.overlay.board_overlay import CALIBRATION_RGB  # noqa: E402
from chesstrainer.overlay.capture_exclusion import apply_capture_setting  # noqa: E402
from chesstrainer.settings.config import Settings  # noqa: E402
from chesstrainer.vision.detect import BoardBox  # noqa: E402

OUT = Path("data/eval")


def pump(app: QApplication, seconds: float) -> None:
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        app.processEvents()
        time.sleep(0.002)


def wait_for(app: QApplication, cond: Callable[[], bool], timeout: float) -> float | None:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        app.processEvents()
        if cond():
            return time.perf_counter() - t0
        time.sleep(0.002)
    return None


class BoardWindow:
    """Frameless window showing a board image at an exact logical position and size."""

    def __init__(self, dpr: float) -> None:
        self.dpr = dpr
        self.label = QLabel()
        self.label.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
        )
        self.size = 0
        self.img: np.ndarray | None = None

    def show(self, img: np.ndarray) -> float:
        self.img = img
        side = round(self.size * self.dpr)
        h, w = img.shape[:2]
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg).scaled(
            side, side, transformMode=Qt.TransformationMode.SmoothTransformation
        )
        pix.setDevicePixelRatio(self.dpr)
        self.label.setPixmap(pix)
        self.label.repaint()
        return time.perf_counter()

    def place(self, x: int, y: int, size: int) -> None:
        self.size = size
        self.label.setFixedSize(size, size)
        self.label.move(x, y)
        if self.img is not None:
            self.show(self.img)
        self.label.show()

    def truth(self) -> BoardBox:
        p = self.label.mapToGlobal(QPoint(0, 0))
        return BoardBox(
            p.x() * self.dpr, p.y() * self.dpr, self.size * self.dpr, self.size * self.dpr
        )


@dataclass
class Alignment:
    name: str
    dots: int
    mean_px: float  # dot centre vs true square centre
    max_px: float
    bias: tuple[float, float]  # mean signed (dx, dy)
    draw_max_px: float  # dot centre vs the centre implied by the *detected* box (drawing only)
    square_px: float
    box_err_px: float  # detected box vs truth, largest edge difference


def _centres(box: BoardBox) -> np.ndarray:
    sq = box.w / 8
    return np.array(
        [
            (box.x + (c + 0.5) * sq, box.y + (r + 0.5) * box.h / 8)
            for r in range(8)
            for c in range(8)
        ]
    )


def measure_alignment(
    app: QApplication, trainer: TrainerApp, board: BoardWindow, name: str
) -> Alignment:
    capture = ScreenCapture(trainer.settings.monitor)
    trainer.vision.paused = True
    pump(app, 0.25)
    overlay = trainer.overlay
    overlay.calibration = True
    saved_paint = overlay._paint_arrow
    overlay._paint_arrow = lambda p, a: None  # type: ignore[method-assign]
    overlay.update()
    apply_capture_setting(overlay, False)
    pump(app, 0.3)
    with_dots = capture.grab().astype(np.int16)
    apply_capture_setting(overlay, True)
    pump(app, 0.3)
    clean = capture.grab().astype(np.int16)
    overlay.calibration = False
    overlay._paint_arrow = saved_paint  # type: ignore[method-assign]
    overlay.update()
    trainer.vision.paused = False

    target = np.array(CALIBRATION_RGB, dtype=np.int16)
    mask = (np.abs(with_dots - target).max(axis=2) < 60) & (
        np.abs(with_dots - clean).max(axis=2) > 40
    )
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    cents = cents + 0.5  # pixel (i, j) covers [i, i+1): its centre is at i + 0.5
    truth = board.truth()
    det = trainer.box or truth
    true_c, det_c = _centres(truth), _centres(det)
    sq = truth.w / 8
    errs, signed, draw = [], [], []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 4:
            continue
        d = np.linalg.norm(true_c - cents[i], axis=1)
        k = int(d.argmin())
        if d[k] < sq / 2:
            errs.append(float(d[k]))
            signed.append(cents[i] - true_c[k])
            draw.append(float(np.linalg.norm(cents[i] - det_c[k])))
    bias = tuple(float(v) for v in np.mean(signed, axis=0)) if signed else (np.nan, np.nan)
    return Alignment(
        name,
        len(errs),
        float(np.mean(errs)) if errs else float("nan"),
        float(np.max(errs)) if errs else float("nan"),
        (bias[0], bias[1]),
        float(np.max(draw)) if draw else float("nan"),
        sq,
        truth.corner_error(det) * truth.square,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hold", type=float, default=1.5, help="seconds per position")
    parser.add_argument("--size", type=int, default=560, help="board size, logical px")
    args = parser.parse_args()
    os.environ["CHESSTRAINER_HOME"] = tempfile.mkdtemp(prefix="ct_selftest_")

    frames = fetch_frames()
    app = QApplication.instance() or QApplication([])
    assert isinstance(app, QApplication)
    screen = app.primaryScreen()
    assert screen is not None
    dpr = screen.devicePixelRatio()
    board = BoardWindow(dpr)
    board.place(120, 140, args.size)
    board.show(frames[0][1])

    trainer = TrainerApp(Settings())
    if trainer.engine is None:
        print("Stockfish not found:\n" + trainer.engine_status)
        return
    events: list[tuple[float, int]] = []
    orig_set = trainer.overlay.set_arrows

    def logged_set(arrows: list, white_bottom: bool) -> None:  # type: ignore[type-arg]
        if bool(arrows) != bool(trainer.overlay.arrows) or not events:
            events.append((time.perf_counter(), len(arrows)))
        orig_set(arrows, white_bottom)

    trainer.overlay.set_arrows = logged_set  # type: ignore[method-assign]
    trainer.start()
    trainer.panel.move(120 + args.size + 20, 140)
    assert trainer.engine is not None
    trainer.engine.wait_ready()
    pump(app, 3.0)  # model load + first detection

    # ---- A. latency over a real game ---------------------------------------------------------
    shown: list[tuple[float, chess.Board]] = []
    fen_ok = 0
    for i, (truth_board, img) in enumerate(frames):
        if i > 0:
            shown.append((board.show(img), truth_board))
        else:
            shown.append((time.perf_counter(), truth_board))
        pump(app, args.hold)
        got = trainer.board.fen() if trainer.board is not None else None
        fen_ok += got is not None and got.split()[:4] == truth_board.fen().split()[:4]

    hide_ms, arrows_ms = [], []
    for t_show, _ in shown[1:]:
        after = [(t, n) for t, n in events if t >= t_show]
        hid = next((t for t, n in after if n == 0), None)
        drew = next((t for t, n in after if n > 0), None)
        if hid is not None:
            hide_ms.append((hid - t_show) * 1000)
        if drew is not None:
            arrows_ms.append((drew - t_show) * 1000)
    lat = trainer.latencies[1:]  # first entry is the initial position

    trainer.overlay.calibration = False
    snap_path = OUT / "phase3_arrows.png"
    OUT.mkdir(parents=True, exist_ok=True)
    apply_capture_setting(trainer.overlay, False)
    pump(app, 0.3)
    t = board.truth()
    shot = ScreenCapture(1).grab()
    pad = 20
    Image.fromarray(
        shot[
            max(0, int(t.y) - pad) : int(t.y + t.h) + pad,
            max(0, int(t.x) - pad) : int(t.x + t.w) + pad,
        ]
    ).save(snap_path)
    apply_capture_setting(trainer.overlay, True)

    # ---- B. alignment, then follow moves and resizes ----------------------------------------
    aligns = [measure_alignment(app, trainer, board, f"start ({args.size} px at 120,140)")]
    follow = []
    for name, (x, y, s) in [
        ("moved", (700, 300, args.size)),
        ("moved, odd logical position", (333, 211, args.size)),
        ("resized smaller", (333, 211, args.size - 151)),
        ("resized larger", (200, 120, args.size + 177)),
    ]:
        t0 = time.perf_counter()
        board.place(x, y, s)
        truth = board.truth()

        def on_new_place(truth: BoardBox = truth) -> bool:
            b = trainer.box
            return b is not None and truth.corner_error(b) < 0.05 and trainer.overlay.isVisible()

        placed = wait_for(app, on_new_place, 5.0)
        back = wait_for(app, lambda: bool(trainer.overlay.arrows), 5.0)
        total_back = None if back is None else time.perf_counter() - t0
        follow.append((name, placed, total_back))
        pump(app, 0.5)
        aligns.append(measure_alignment(app, trainer, board, name))

    trainer.quit()

    # ---- report ------------------------------------------------------------------------------
    def stats(v: list[float]) -> str:
        if not v:
            return "n/a"
        a = np.array(v)
        return (
            f"median {np.median(a):.0f} ms, p90 {np.percentile(a, 90):.0f} ms, "
            f"max {a.max():.0f} ms (n={len(a)})"
        )

    lines = [
        "== Phase 3 overlay self-test ==",
        f"display scale (devicePixelRatio): {dpr:g}  |  engine: {trainer.engine.name}, "
        f"{trainer.settings.engine_threads} threads, MultiPV {trainer.settings.multipv}  |  "
        f"capture {trainer.settings.capture_fps:g} fps",
        "",
        f"A. Latency over a real game ({len(frames) - 1} moves, lichess renders, no animation)",
        f"  position read correctly (placement/turn/castling/ep): {fen_ok}/{len(frames)}",
        f"  new position on screen -> stale arrows hidden : {stats(hide_ms)}",
        f"  new position on screen -> new arrows painted  : {stats(arrows_ms)}",
        f"  moves with arrows within 500 ms              : "
        f"{sum(v <= 500 for v in arrows_ms)}/{len(arrows_ms)}",
        "  app-side breakdown (first changed frame -> arrows):",
        f"    total            {stats([x.total_ms for x in lat])}",
        f"    vision (commit)  {stats([(x.committed - x.seen) * 1000 for x in lat])}",
        f"    engine (1st set) {stats([x.engine_s * 1000 for x in lat])}",
        f"    depth at first arrows: median {np.median([x.depth for x in lat]):.0f}" if lat else "",
        "",
        "B. Alignment: calibration dots on the 64 square centres vs true centres (physical px)",
    ]
    for a in aligns:
        lines.append(
            f"  {a.name:30s} dots {a.dots:2d}/64  error mean {a.mean_px:.2f} px, "
            f"max {a.max_px:.2f} px = {a.max_px / a.square_px * 100:.1f}% of a "
            f"{a.square_px:.0f} px square; bias ({a.bias[0]:+.2f}, {a.bias[1]:+.2f})"
        )
        lines.append(
            f"  {'':30s} of which detection: box edges off by up to {a.box_err_px:.2f} px; "
            f"drawing vs detected grid: max {a.draw_max_px:.2f} px"
        )
    lines += ["", "C. Following the board window (time after the window moved/resized)"]
    for name, placed, back in follow:
        p = "never" if placed is None else f"{placed * 1000:.0f} ms"
        b = "never" if back is None else f"{back * 1000:.0f} ms"
        lines.append(f"  {name:32s} overlay on new place: {p:>7s}   arrows back: {b:>7s}")
    lines.append(f"\nscreenshot with arrows (overlay shown to capture): {snap_path}")
    report = "\n".join(lines)
    print(report)
    (OUT / "phase3_report.txt").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
