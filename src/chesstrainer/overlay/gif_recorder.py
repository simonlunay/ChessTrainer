"""Record a screen region to an animated GIF (for demo clips of the overlay).

The overlay is normally hidden from screen capture, so the app turns that off while a recording
runs; the recorder itself just grabs the region with mss on its own thread.
"""

from __future__ import annotations

import datetime as dt
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from chesstrainer.capture.screen import ScreenCapture

log = logging.getLogger(__name__)


def default_output_dir() -> Path:
    videos = Path.home() / "Videos"
    return (videos if videos.is_dir() else Path.home()) / "ChessTrainer"


class GifRecorder:
    def __init__(
        self,
        monitor: int,
        region: tuple[int, int, int, int],
        out_path: Path | None = None,
        fps: float = 10.0,
        max_seconds: float = 60.0,
        max_width: int = 720,
        on_done: Callable[[Path | None, str], None] | None = None,
    ) -> None:
        self.monitor = monitor
        self.region = region  # monitor-relative physical px (x, y, w, h)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_path = out_path or default_output_dir() / f"chesstrainer_{stamp}.gif"
        self.fps = fps
        self.max_seconds = max_seconds
        self.max_width = max_width
        self.on_done = on_done
        self.frames: list[Image.Image] = []
        self.started = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="gif-recorder", daemon=True)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started if self.started else 0.0

    def start(self) -> None:
        self.started = time.monotonic()
        self._thread.start()

    def stop(self) -> None:
        """Stop recording; the GIF is written on the recorder thread, then ``on_done`` fires."""
        self._stop.set()

    def _run(self) -> None:
        capture = ScreenCapture(self.monitor)
        period = 1.0 / self.fps
        next_t = time.perf_counter()
        x, y, w, h = self.region
        scale = min(1.0, self.max_width / max(1, w))
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        while not self._stop.is_set() and self.elapsed < self.max_seconds:
            img = Image.fromarray(capture.grab(self.region))
            if img.size != size:
                img = img.resize(size, Image.Resampling.LANCZOS)
            self.frames.append(img)
            next_t += period
            self._stop.wait(max(0.0, next_t - time.perf_counter()))
        path: Path | None = None
        msg = "nothing recorded"
        if self.frames:
            try:
                path = self._save()
                msg = f"saved {len(self.frames)} frames ({len(self.frames) / self.fps:.1f}s)"
            except OSError as err:
                msg = f"could not save GIF: {err}"
                log.exception("GIF save failed")
        if self.on_done:
            self.on_done(path, msg)

    def _save(self) -> Path:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        # One shared palette from a few sampled frames keeps colours steady (no flicker).
        step = max(1, len(self.frames) // 8)
        sample = self.frames[::step][:8]
        strip = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
        for i, f in enumerate(sample):
            strip.paste(f, (0, i * f.height))
        palette = strip.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
        frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in self.frames]
        frames[0].save(
            self.out_path,
            save_all=True,
            append_images=frames[1:],
            duration=round(1000 / self.fps),
            loop=0,
            optimize=False,
        )
        return self.out_path
