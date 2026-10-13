"""Capture + board reading on a background thread, reporting a :class:`Snapshot` per frame."""

from __future__ import annotations

import logging
import math
import queue
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from PyQt6.QtCore import QThread, pyqtSignal

from chesstrainer.capture.screen import ScreenCapture
from chesstrainer.settings.config import Settings
from chesstrainer.vision.detect import BoardBox
from chesstrainer.vision.geometry import ImageU8
from chesstrainer.vision.position import PositionReading
from chesstrainer.vision.reader import BoardReader

log = logging.getLogger(__name__)

FAST_PERIOD = 0.035  # seconds between frames while the position is changing


@dataclass(frozen=True)
class Snapshot:
    t: float  # perf_counter just before the screen was grabbed
    box: BoardBox | None  # monitor-relative physical px
    white_bottom: bool
    reading: PositionReading | None  # committed position
    changed: bool  # ``reading`` was committed by this frame
    pending: bool  # the frame differs from the committed position (move in progress)
    timings_ms: dict[str, float] = field(default_factory=dict)


class VisionThread(QThread):
    snapshot = pyqtSignal(object)  # Snapshot
    failed = pyqtSignal(str)

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings
        self.paused = False
        self._stop = False
        self._commands: queue.Queue[Callable[[BoardReader], None]] = queue.Queue()

    def stop(self) -> None:
        self._stop = True
        self.wait(3000)

    def call(self, fn: Callable[[BoardReader], None]) -> None:
        """Run ``fn(reader)`` on the vision thread before the next frame."""
        self._commands.put(fn)

    def run(self) -> None:
        try:
            capture = ScreenCapture(self.settings.monitor)
            reader = BoardReader(self.settings)
        except Exception as err:  # noqa: BLE001 - reported in the panel
            log.exception("vision start-up failed")
            self.failed.emit(f"Could not start board reading: {err}")
            return
        period = 1.0 / max(0.5, self.settings.capture_fps)
        next_t = time.perf_counter()
        frame: ImageU8 | None = None
        while not self._stop:
            while not self._commands.empty():
                self._commands.get_nowait()(reader)
            if self.paused:
                self.msleep(100)
                next_t = time.perf_counter()
                continue
            t = time.perf_counter()
            try:
                frame, partial = self._grab(capture, reader, frame)
                state = reader.process(frame, full_search=not partial)
                if partial and state.detection is None:
                    # Lost the board in the region grab (window moved?): look at everything.
                    frame, _ = self._grab(capture, reader, None)
                    state = reader.process(frame)
            except Exception:  # noqa: BLE001 - keep running; one bad frame is not fatal
                log.exception("frame failed")
                frame = None
                self.msleep(200)
                continue
            pending = state.detection is not None and reader.positions.pending
            self.snapshot.emit(
                Snapshot(
                    t=t,
                    box=state.detection.box if state.detection else None,
                    white_bottom=state.white_bottom,
                    reading=state.reading,
                    changed=state.changed,
                    pending=pending,
                    timings_ms=state.timings_ms,
                )
            )
            # While a move is in progress (piece animating, change not yet stable) look again
            # soon: the commit needs two equal frames, and waiting a full period for the second
            # one is most of the move -> arrows latency.
            next_t = min(next_t + period, t + FAST_PERIOD) if pending else next_t + period
            delay = next_t - time.perf_counter()
            if delay > 0:
                self.msleep(int(delay * 1000))
            else:
                next_t = time.perf_counter()

    @staticmethod
    def _grab(
        capture: ScreenCapture, reader: BoardReader, frame: ImageU8 | None
    ) -> tuple[ImageU8, bool]:
        """Grab the screen. While a board is tracked only its neighbourhood is grabbed (a full
        2560x1600 grab costs ~45 ms, a board-sized one ~12 ms) and pasted into the previous
        full frame, so coordinates stay monitor-relative. Returns (frame, partial)."""
        det = reader.boards.current
        if det is None or frame is None:
            return capture.grab(), False
        b, m = det.box, capture.monitor
        margin = 0.6 * b.square
        x0, y0 = max(0, math.floor(b.x - margin)), max(0, math.floor(b.y - margin))
        x1 = min(m.width, math.ceil(b.x + b.w + margin))
        y1 = min(m.height, math.ceil(b.y + b.h + margin))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return capture.grab(), False
        roi = capture.grab((x0, y0, x1 - x0, y1 - y0))
        frame[y0 : y0 + roi.shape[0], x0 : x0 + roi.shape[1]] = roi
        return frame, True
