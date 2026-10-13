"""Read the board from the live screen and print the FEN whenever it changes.

uv run python -m chesstrainer.vision.live              # watch the primary monitor
uv run python -m chesstrainer.vision.live --select     # drag a box around the board first
uv run python -m chesstrainer.vision.live --image shot.png
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from chesstrainer.capture.screen import ScreenCapture
from chesstrainer.settings.config import load_settings, save_settings
from chesstrainer.vision.geometry import ImageU8, cell_to_square
from chesstrainer.vision.labels import CLASS_NAMES
from chesstrainer.vision.reader import BoardReader, ReaderState, read_image

log = logging.getLogger(__name__)


def annotate(frame: ImageU8, state: ReaderState) -> ImageU8:
    """Board crop with the detected grid and each square's label, for debugging."""
    det = state.detection
    if det is None or state.reading is None:
        return frame
    x, y, w, h = det.box.as_int()
    board = np.ascontiguousarray(frame[y : y + h, x : x + w].copy())
    sw, sh = w / 8, h / 8
    for i in range(9):
        cv2.line(board, (round(i * sw), 0), (round(i * sw), h), (255, 0, 255), 1)
        cv2.line(board, (0, round(i * sh)), (w, round(i * sh)), (255, 0, 255), 1)
    for cell in range(64):
        r, c = divmod(cell, 8)
        sq = cell_to_square(cell, state.white_bottom)
        name = CLASS_NAMES[state.reading.labels[sq]]
        if name != "empty":
            org = (round(c * sw + 3), round(r * sh + sh - 4))
            cv2.putText(
                board, name, org, cv2.FONT_HERSHEY_SIMPLEX, max(0.4, sh / 70), (255, 0, 255), 2
            )
    return board


def describe(state: ReaderState) -> str:
    r = state.reading
    assert r is not None
    moves = " ".join(m.uci() for m in r.moves)
    extra = []
    if r.source == "history" and moves:
        extra.append(f"after {moves}")
    if r.turn_assumed:
        extra.append("side to move assumed")
    if not r.valid:
        extra.append("INVALID")
    if r.confidence < 0.6:
        extra.append(f"low confidence {r.confidence:.2f}")
    return f"{r.fen}" + (f"   ({'; '.join(extra)})" if extra else "")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--image", type=Path, help="read a screenshot file instead of the screen")
    parser.add_argument(
        "--select", action="store_true", help="drag-select the board region and save it"
    )
    parser.add_argument(
        "--clear-region", action="store_true", help="forget a saved region; auto-detect"
    )
    parser.add_argument(
        "--orientation", choices=["auto", "white", "black"], help="side at the bottom (saved)"
    )
    parser.add_argument("--monitor", type=int, help="mss monitor index (1 = primary; saved)")
    parser.add_argument("--seconds", type=float, help="stop after this many seconds")
    parser.add_argument(
        "--debug-dir", type=Path, help="save an annotated board image on each change"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    settings = load_settings()
    if args.orientation:
        settings.orientation = args.orientation
    if args.monitor is not None:
        settings.monitor = args.monitor
    if args.clear_region:
        settings.locked_region = None
    if args.select:
        from chesstrainer.overlay.region_select import select_region

        region = select_region(settings.monitor)
        if region is None:
            print("selection cancelled")
            return
        settings.locked_region = list(region)
    if args.orientation or args.monitor is not None or args.clear_region or args.select:
        save_settings(settings)

    if args.image:
        image = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.uint8)
        state = read_image(image, BoardReader(settings))
        if state.reading is None:
            print("no board found")
            return
        print(describe(state))
        for note in state.reading.notes:
            print("  note:", note)
        if args.debug_dir:
            args.debug_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(annotate(image, state)).save(
                args.debug_dir / f"{args.image.stem}_read.png"
            )
        return

    reader = BoardReader(settings)
    capture = ScreenCapture(settings.monitor)
    print(f"watching monitor {settings.monitor} at {settings.capture_fps:g} fps (Ctrl+C to stop)")
    t_end = time.monotonic() + args.seconds if args.seconds else None
    had_board = None
    n = 0
    try:
        for frame in capture.frames(settings.capture_fps):
            state = reader.process(frame)
            n += 1
            if (state.detection is not None) != had_board:
                had_board = state.detection is not None
                if had_board:
                    assert state.detection is not None
                    b = state.detection.box
                    print(f"board found at x={b.x:.0f} y={b.y:.0f} size={b.w:.0f}px")
                else:
                    print("no board on screen")
            if state.changed and state.reading is not None:
                ms = ", ".join(f"{k} {v:.0f}ms" for k, v in state.timings_ms.items())
                print(f"{describe(state)}   [{ms}]")
                if args.debug_dir:
                    args.debug_dir.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(annotate(frame, state)).save(
                        args.debug_dir / f"frame_{n:05d}.png"
                    )
            if t_end is not None and time.monotonic() > t_end:
                break
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
