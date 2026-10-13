"""End-to-end live test: real-site board images on screen, read back through screen capture.

Shows the positions of a real game, rendered by lichess.org (with its last-move highlights), one
after another in an always-on-top window. A 5 fps capture loop runs the full BoardReader on the
whole monitor, exactly like the app, and the script checks each committed FEN (placement, side
to move, castling, en passant) against the game. Finally it enables capture exclusion on the
window and checks that the board disappears from the capture.

Nothing is saved: captured frames stay in memory. Requires network access for the board images
(cached in data/real/live_game/ after the first run).
"""

from __future__ import annotations

import argparse
import io
import time
import urllib.parse
import urllib.request
from pathlib import Path

import chess
import numpy as np
from PIL import Image
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel

from chesstrainer.capture.screen import ScreenCapture
from chesstrainer.overlay.capture_exclusion import apply_capture_setting
from chesstrainer.settings.config import Settings
from chesstrainer.vision.detect import BoardBox, detect_boards
from chesstrainer.vision.reader import BoardReader

# Italian Game with an en passant capture (6.exd6), check, and castling on both sides.
GAME = (
    "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 d2d4 e5d4 e4e5 d7d5 e5d6 f8d6 "
    "e1g1 e8g8 f1e1 c8g4 h2h3 g4h5 c1g5 d8d7 b1d2 a8e8"
)
CACHE = Path("data/real/live_game")


def fetch_frames() -> list[tuple[chess.Board, np.ndarray]]:
    CACHE.mkdir(parents=True, exist_ok=True)
    board = chess.Board()
    frames = []
    for i, uci in enumerate([""] + GAME.split()):
        if uci:
            board.push_uci(uci)
        path = CACHE / f"ply_{i:02d}.gif"
        if not path.exists():
            params = {"fen": board.board_fen(), "theme": "brown"}
            if uci:
                params["lastMove"] = uci
            url = "https://lichess1.org/export/fen.gif?" + urllib.parse.urlencode(params)
            req = urllib.request.Request(url, headers={"User-Agent": "chesstrainer selftest"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                path.write_bytes(resp.read())
            time.sleep(0.3)
        img = np.asarray(Image.open(io.BytesIO(path.read_bytes())).convert("RGB"), dtype=np.uint8)
        frames.append((board.copy(), img))
    return frames


def to_pixmap(img: np.ndarray, size: int) -> QPixmap:
    h, w = img.shape[:2]
    qimg = QImage(img.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg).scaled(size, size)


def check_exclusion(app: QApplication, label: QLabel, capture: ScreenCapture) -> list[str]:
    """Toggle capture exclusion on the window and check whether the board shows up in a grab."""
    dpr = label.screen().devicePixelRatio()  # type: ignore[union-attr]
    geo = label.geometry()
    mon = capture.monitor
    expected = BoardBox(
        geo.x() * dpr - mon.left, geo.y() * dpr - mon.top, geo.width() * dpr, geo.height() * dpr
    )
    lines = []
    for hide in (False, True, False):
        ok = apply_capture_setting(label, hide=hide)
        t = time.monotonic()
        while time.monotonic() - t < 0.6:
            app.processEvents()
            time.sleep(0.02)
        seen = any(d.box.iou(expected) > 0.5 for d in detect_boards(capture.grab()))
        lines.append(
            f"hide_overlay_from_capture={hide}: API ok={ok}, board visible to capture: {seen}"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=640, help="board size on screen (logical px)")
    parser.add_argument("--hold", type=float, default=1.2, help="seconds per position")
    args = parser.parse_args()

    frames = fetch_frames()
    app = QApplication.instance() or QApplication([])
    label = QLabel()
    label.setWindowTitle("ChessTrainer live self-test")
    label.setWindowFlags(label.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
    label.setFixedSize(args.size, args.size)
    label.move(80, 80)
    label.setPixmap(to_pixmap(frames[0][1], args.size))
    label.show()
    apply_capture_setting(label, hide=False)

    capture = ScreenCapture(1)
    reader = BoardReader(Settings())
    results: list[tuple[int, str, str, bool]] = []
    shown = {"i": 0, "t": time.monotonic()}
    committed: dict[int, str] = {}
    timings: list[float] = []
    exclusion: list[str] = []

    def tick() -> None:
        frame = capture.grab()
        t0 = time.perf_counter()
        state = reader.process(frame)
        timings.append((time.perf_counter() - t0) * 1000)
        if state.changed and state.reading is not None:
            committed[shown["i"]] = state.reading.fen
        if time.monotonic() - shown["t"] >= args.hold:
            shown["i"] += 1
            shown["t"] = time.monotonic()
            if shown["i"] >= len(frames):
                timer.stop()
                exclusion.extend(check_exclusion(app, label, capture))
                app.quit()  # type: ignore[union-attr]
                return
            label.setPixmap(to_pixmap(frames[shown["i"]][1], args.size))

    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(200)
    app.exec()  # type: ignore[union-attr]

    print(f"\n== live game: {len(frames)} positions shown, read via full-monitor capture ==")
    last = None
    for i, (board, _) in enumerate(frames):
        fen = committed.get(i, last)
        last = fen
        truth = board.fen()
        ok = fen is not None and fen.split()[:4] == truth.split()[:4]
        results.append((i, truth, fen or "-", ok))
        print(
            f"{'OK ' if ok else 'BAD'} ply {i:2d}  read {fen}"
            + ("" if ok else f"\n               truth {truth}")
        )
    n_ok = sum(r[3] for r in results)
    print(f"exact FEN (placement, side to move, castling, en passant): {n_ok}/{len(results)}")
    print(
        f"per-frame processing: median {np.median(timings):.0f} ms, "
        f"p95 {np.percentile(timings, 95):.0f} ms"
    )
    for line in exclusion:
        print(line)

    label.close()


if __name__ == "__main__":
    main()
