"""Screen capture with mss.

Frames are RGB uint8 arrays in physical pixels of the chosen monitor. mss makes the process
DPI-aware on Windows, so coordinates are physical even on scaled (e.g. 150%) displays; the
overlay converts to Qt's logical coordinates itself.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass

import mss
import numpy as np

from chesstrainer.vision.geometry import ImageU8


@dataclass(frozen=True)
class MonitorInfo:
    index: int
    left: int
    top: int
    width: int
    height: int


class ScreenCapture:
    """Grabs one monitor (or a region of it). mss handles are per thread, so each thread
    that calls :meth:`grab` lazily gets its own."""

    def __init__(self, monitor: int = 1) -> None:
        self._local = threading.local()
        monitors = self.monitors()
        if not 0 <= monitor < len(monitors):
            raise ValueError(f"monitor {monitor} not found; available: 0..{len(monitors) - 1}")
        self.monitor = monitors[monitor]

    def _sct(self) -> mss.base.MSSBase:
        sct = getattr(self._local, "sct", None)
        if sct is None:
            sct = mss.mss()
            self._local.sct = sct
        return sct

    @staticmethod
    def monitors() -> list[MonitorInfo]:
        with mss.mss() as sct:
            return [
                MonitorInfo(i, m["left"], m["top"], m["width"], m["height"])
                for i, m in enumerate(sct.monitors)
            ]

    def grab(self, region: tuple[int, int, int, int] | None = None) -> ImageU8:
        """Capture the monitor, or ``region = (x, y, w, h)`` relative to the monitor."""
        m = self.monitor
        if region is None:
            box = {"left": m.left, "top": m.top, "width": m.width, "height": m.height}
        else:
            x, y, w, h = region
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(m.width, x + w), min(m.height, y + h)
            if x1 <= x0 or y1 <= y0:
                raise ValueError(f"region {region} is outside the monitor")
            box = {"left": m.left + x0, "top": m.top + y0, "width": x1 - x0, "height": y1 - y0}
        shot = self._sct().grab(box)
        bgra = np.frombuffer(shot.bgra, dtype=np.uint8).reshape(shot.height, shot.width, 4)
        return np.ascontiguousarray(bgra[:, :, 2::-1])  # BGRA -> RGB

    def frames(self, fps: float) -> Iterator[ImageU8]:
        """Yield full-monitor frames at up to ``fps`` frames per second."""
        period = 1.0 / fps
        next_t = time.perf_counter()
        while True:
            yield self.grab()
            next_t += period
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.perf_counter()  # fell behind; don't try to catch up
