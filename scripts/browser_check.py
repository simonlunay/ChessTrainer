"""Phase 3 check on a real browser: lichess.org in Chrome, read by the running app.

Opens a lichess analysis board (a real game loaded as PGN) in a separate Chrome window with a
throwaway profile, runs ``TrainerApp`` on the live screen, then:

* steps through the game with the Right arrow key (lichess animates each move) and measures
  key press -> arrows painted, plus the app-side first-changed-frame -> arrows time;
* measures alignment of calibration dots against the board found *independently* by segmenting
  lichess's two square colours in a clean screenshot (not the app's detector);
* moves and resizes the Chrome window with the Win32 API and measures how long the overlay takes
  to sit on the board again, then measures alignment again.

Windows only. Writes data/eval/phase3_browser.txt and data/eval/phase3_browser.png.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import tempfile
import time
from ctypes import wintypes
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PyQt6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).parent))
from overlay_selftest import pump, wait_for  # noqa: E402

from chesstrainer.capture.screen import ScreenCapture  # noqa: E402
from chesstrainer.overlay.app import TrainerApp  # noqa: E402
from chesstrainer.overlay.board_overlay import CALIBRATION_RGB  # noqa: E402
from chesstrainer.overlay.capture_exclusion import apply_capture_setting  # noqa: E402
from chesstrainer.settings.config import Settings  # noqa: E402
from chesstrainer.vision.detect import BoardBox  # noqa: E402

CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
PGN = (
    "1.e4 e5 2.Nf3 Nc6 3.Bc4 Nf6 4.d4 exd4 5.e5 d5 6.exd6 Bxd6 7.O-O O-O 8.Re1 Bg4 "
    "9.h3 Bh5 10.Bg5 Qd7 11.Nbd2 Rae8"
)
# lichess "brown" board colours (RGB)
LIGHT, DARK = (240, 217, 181), (181, 136, 99)
OUT = Path("data/eval")
user32 = ctypes.WinDLL("user32", use_last_error=True)
VK_RIGHT, VK_HOME = 0x27, 0x24


def find_window(pid: int, title_part: str) -> int | None:
    """A visible top-level window of process ``pid`` whose title contains ``title_part``."""
    found: list[int] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd: int, _: int) -> bool:
        n = user32.GetWindowTextLengthW(hwnd)
        if n and user32.IsWindowVisible(hwnd):
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            owner = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
            if title_part in buf.value and owner.value == pid:
                found.append(hwnd)
        return True

    user32.EnumWindows(cb, 0)
    return found[0] if found else None


def render_widget(hwnd: int) -> int:
    """Chrome's page child window, which takes keyboard messages without focus."""
    found: list[int] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(child: int, _: int) -> bool:
        buf = ctypes.create_unicode_buffer(64)
        user32.GetClassNameW(child, buf, 64)
        if buf.value == "Chrome_RenderWidgetHostHWND":
            found.append(child)
            return False
        return True

    user32.EnumChildWindows(hwnd, cb, 0)
    return found[0] if found else hwnd


def press(target: int, vk: int) -> float:
    """Post a key press to one window only: nothing reaches other apps, focus is not touched."""
    scan = user32.MapVirtualKeyW(vk, 0)
    down = 1 | (scan << 16) | (1 << 24)  # arrows/Home are extended keys
    user32.PostMessageW(target, 0x100, vk, down)  # WM_KEYDOWN
    user32.PostMessageW(target, 0x101, vk, down | (1 << 30) | (1 << 31))  # WM_KEYUP
    return time.perf_counter()


def move_window(hwnd: int, x: int, y: int, w: int, h: int, topmost: bool = False) -> None:
    """Physical px (the process is DPI aware). ``topmost`` once at the start keeps other
    windows from covering the test board; later moves keep the z-order (overlay stays above)."""
    flags = 0x0010  # SWP_NOACTIVATE
    after = -1 if topmost else 0
    if not topmost:
        flags |= 0x0004  # SWP_NOZORDER
    user32.SetWindowPos(wintypes.HWND(hwnd), wintypes.HWND(after), x, y, w, h, flags)


def board_by_colour(img: np.ndarray) -> BoardBox | None:
    """Board edges from the two square colours (independent of the app's detector)."""
    d = np.minimum(
        np.abs(img.astype(np.int16) - LIGHT).max(axis=2),
        np.abs(img.astype(np.int16) - DARK).max(axis=2),
    )
    mask = (d <= 3).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    if n < 2:
        return None
    # The board: the bounding box of all square-colour pixels around the largest component.
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = (int(v) for v in stats[i, :4])
    pad = w // 6
    sub = mask[max(0, y - pad) : y + h + pad, max(0, x - pad) : x + w + pad]
    cols = np.flatnonzero(sub.sum(axis=0) > sub.shape[0] * 0.05)
    rows = np.flatnonzero(sub.sum(axis=1) > sub.shape[1] * 0.05)
    x0, y0 = max(0, x - pad) + cols[0], max(0, y - pad) + rows[0]
    return BoardBox(
        float(x0), float(y0), float(cols[-1] - cols[0] + 1), float(rows[-1] - rows[0] + 1)
    )


def measure(app: QApplication, trainer: TrainerApp, name: str) -> str:
    capture = ScreenCapture(trainer.settings.monitor)
    trainer.vision.paused = True
    pump(app, 0.3)
    overlay = trainer.overlay
    overlay.calibration = True
    saved = overlay._paint_arrow
    overlay._paint_arrow = lambda p, a: None  # type: ignore[method-assign]
    overlay.update()
    apply_capture_setting(overlay, False)
    pump(app, 0.3)
    dots_img = capture.grab()
    apply_capture_setting(overlay, True)
    pump(app, 0.3)
    clean = capture.grab()
    overlay.calibration = False
    overlay._paint_arrow = saved  # type: ignore[method-assign]
    overlay.update()
    trainer.vision.paused = False

    truth = board_by_colour(clean)
    det = trainer.box
    if truth is None or det is None:
        return f"  {name:28s} board not found (truth {truth}, detected {det})"
    a, b = dots_img.astype(np.int16), clean.astype(np.int16)
    mask = (np.abs(a - np.array(CALIBRATION_RGB)).max(axis=2) < 60) & (
        np.abs(a - b).max(axis=2) > 40
    )
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    cents = cents + 0.5
    sq = truth.w / 8
    centres = np.array(
        [
            (truth.x + (c + 0.5) * truth.w / 8, truth.y + (r + 0.5) * truth.h / 8)
            for r in range(8)
            for c in range(8)
        ]
    )
    errs = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 4:
            d = np.linalg.norm(centres - cents[i], axis=1)
            if d.min() < sq / 2:
                errs.append(float(d.min()))
    if not errs:
        return f"  {name:28s} no calibration dots found"
    edge = truth.corner_error(det) * truth.square
    return (
        f"  {name:28s} board {truth.w:.0f} px; dots {len(errs)}/64, error vs true centres "
        f"mean {np.mean(errs):.2f} px, max {max(errs):.2f} px ({max(errs) / sq * 100:.1f}% "
        f"of a square); detected box edges off by up to {edge:.2f} px"
    )


def main() -> None:
    os.environ["CHESSTRAINER_HOME"] = tempfile.mkdtemp(prefix="ct_browser_")
    profile = tempfile.mkdtemp(prefix="ct_chrome_")
    url = "https://lichess.org/analysis/pgn/" + PGN.replace(" ", "_") + "#0"  # start position
    chrome = subprocess.Popen(
        [
            str(CHROME),
            f"--user-data-dir={profile}",
            "--no-first-run",
            "--no-default-browser-check",
            "--window-position=60,40",
            "--window-size=1000,900",
            f"--app={url}",
        ]
    )
    app = QApplication.instance() or QApplication([])
    assert isinstance(app, QApplication)
    lines = ["== Phase 3 real-browser check: lichess.org analysis board in Chrome =="]
    try:
        hwnd = None
        t0 = time.time()
        while hwnd is None and time.time() - t0 < 20:
            pump(app, 0.5)
            hwnd = find_window(chrome.pid, "lichess")
        if hwnd is None:
            print("Chrome window with lichess not found")
            return
        pump(app, 3.0)  # page load
        move_window(hwnd, 60, 40, 1500, 1350, topmost=True)
        page = render_widget(hwnd)
        press(page, VK_HOME)
        pump(app, 1.0)

        trainer = TrainerApp(Settings())
        assert trainer.engine is not None, "Stockfish not found"
        trainer.start()
        trainer.panel.move(1100, 60)
        trainer.engine.wait_ready()
        ok = wait_for(app, lambda: bool(trainer.overlay.arrows), 20.0)
        lines.append(f"first board + arrows after start: {'no' if ok is None else 'yes'}")
        if ok is None:
            print("\n".join(lines) + "\n(no board found: is the Chrome window covered?)")
            trainer.quit()
            return

        # ---- latency: step through the game with the arrow key ----------------------------
        key_to_arrows = []
        n_moves = len(PGN.replace(".", ". ").split()) - PGN.count(".")
        for _ in range(n_moves):
            before = len(trainer.latencies)
            t_key = press(page, VK_RIGHT)
            got = wait_for(app, lambda b=before: len(trainer.latencies) > b, 3.0)  # type: ignore[misc]
            if got is not None:
                key_to_arrows.append((trainer.latencies[-1].arrows - t_key) * 1000)
            if trainer.box is None:
                lines.append("board lost while stepping (window covered by another one?)")
                break
            pump(app, 1.0)
        lat = trainer.latencies[1:]
        final = trainer.board.fen() if trainer.board else "-"
        lines += [
            f"moves stepped: {n_moves}; moves that got new arrows: {len(key_to_arrows)}",
            f"final position read: {final}",
            "key press -> arrows painted (includes lichess's move animation): "
            f"median {np.median(key_to_arrows or [np.nan]):.0f} ms, "
            f"max {np.max(key_to_arrows or [np.nan]):.0f} ms; "
            f"within 500 ms: {sum(v <= 500 for v in key_to_arrows)}/{len(key_to_arrows)}",
            "first changed frame -> arrows (app side): "
            f"median {np.median([x.total_ms for x in lat]):.0f} ms, "
            f"max {np.max([x.total_ms for x in lat]):.0f} ms",
        ]
        pump(app, 1.0)
        apply_capture_setting(trainer.overlay, False)
        pump(app, 0.3)
        shot = ScreenCapture(1).grab()
        if trainer.box is not None:
            x, y, w, h = trainer.box.as_int()
            Image.fromarray(shot[max(0, y - 20) : y + h + 20, max(0, x - 20) : x + w + 20]).save(
                OUT / "phase3_browser.png"
            )
        apply_capture_setting(trainer.overlay, True)

        # ---- alignment, window moves and resizes --------------------------------------------
        lines.append("alignment (calibration dots vs board found by square colours):")
        lines.append(measure(app, trainer, "initial"))
        for name, rect in [
            ("window moved", (700, 200, 1500, 1350)),
            ("window moved again", (233, 117, 1500, 1350)),
            ("window resized smaller", (233, 117, 1150, 1000)),
            ("window resized larger", (100, 50, 1800, 1500)),
        ]:
            t_move = time.perf_counter()
            move_window(hwnd, *rect)
            prev = trainer.box

            def relocated(prev: BoardBox | None = prev) -> bool:
                b = trainer.box
                return b is not None and (prev is None or b.corner_error(prev) > 0.1)

            placed = wait_for(app, relocated, 5.0)
            back = wait_for(app, lambda: bool(trainer.overlay.arrows) and not trainer.moving, 5.0)
            p = "never" if placed is None else f"{placed * 1000:.0f} ms"
            b = "never" if back is None else f"{(time.perf_counter() - t_move) * 1000:.0f} ms"
            lines.append(f"  {name}: overlay on new place after {p}, arrows back after {b}")
            pump(app, 1.0)
            lines.append(measure(app, trainer, name))
        trainer.quit()
    finally:
        # Chrome keeps child processes (and the window) alive unless the whole tree goes.
        subprocess.run(["taskkill", "/PID", str(chrome.pid), "/T", "/F"], capture_output=True)
    report = "\n".join(lines)
    print(report)
    (OUT / "phase3_browser.txt").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
