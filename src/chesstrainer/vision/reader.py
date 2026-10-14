"""Frame -> position: board detection, square classification, orientation, position tracking."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import chess
import numpy as np

from chesstrainer.settings.config import Settings
from chesstrainer.vision.detect import BoardBox, BoardTracker, Detection
from chesstrainer.vision.geometry import ImageU8, cell_to_square
from chesstrainer.vision.orientation import OrientationTracker, coordinate_score
from chesstrainer.vision.position import (
    PositionReading,
    PositionTracker,
    cells_to_squares,
    repair,
)
from chesstrainer.vision.recognizer import F32, PieceRecognizer


@dataclass
class ReaderState:
    detection: Detection | None
    white_bottom: bool
    reading: PositionReading | None  # the committed position (may be from an earlier frame)
    changed: bool  # True when ``reading`` was committed by this frame
    cell_probs: F32 | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def fen(self) -> str | None:
        return self.reading.fen if self.reading else None


class BoardReader:
    def __init__(
        self,
        settings: Settings | None = None,
        recognizer: PieceRecognizer | None = None,
        stable_frames: int = 2,
    ) -> None:
        settings = settings or Settings()
        locked = BoardBox(*settings.locked_region) if settings.locked_region else None
        self.boards = BoardTracker(locked=locked)
        self.recognizer = recognizer or PieceRecognizer(
            device=settings.device,
            keep_warm=settings.gpu_keep_warm,
            backend=settings.classifier_backend,  # type: ignore[arg-type]
        )
        self.orientation = OrientationTracker(settings.orientation)
        self.positions = PositionTracker(stable_frames=stable_frames)
        self._last_box: BoardBox | None = None

    def flip(self) -> None:
        self.orientation.flip()
        self.positions.reset()

    def _same_position(self, cell_probs: F32) -> bool:
        cur = self.positions.current
        if cur is None:
            return False
        labels, _ = repair(cells_to_squares(cell_probs, self.orientation.white_bottom))
        return bool(np.array_equal(labels, cur.labels))

    def process(
        self, frame: ImageU8, now: float | None = None, full_search: bool = True
    ) -> ReaderState:
        t0 = time.perf_counter()
        det = self.boards.update(frame, now, full_search)
        t1 = time.perf_counter()
        timings = {"detect": (t1 - t0) * 1000}
        if det is None:
            return ReaderState(
                None, self.orientation.white_bottom, self.positions.current, False, None, timings
            )
        moved = self._last_box is not None and det.box.iou(self._last_box) < 0.5
        self._last_box = det.box

        probs, _ = self.recognizer.classify(frame, det.box)
        t2 = time.perf_counter()
        timings["classify"] = (t2 - t1) * 1000
        if moved and self.orientation.mode == "auto" and not self._same_position(probs):
            # A different board: orientation must be re-learned. (The same position in a
            # window that was dragged or resized keeps its orientation and history.)
            self.orientation.reset("auto")

        coords = coordinate_score(frame, det) if self.orientation.mode == "auto" else None
        white_bottom = self.orientation.update(probs.argmax(axis=1), coords)
        if self.positions.current is not None and white_bottom != getattr(
            self, "_wb", white_bottom
        ):
            self.positions.reset()  # orientation flipped: square mapping changed
        self._wb = white_bottom

        highlights = [cell_to_square(c, white_bottom) for c in det.highlighted_cells()]
        reading = self.positions.update(
            cells_to_squares(probs, white_bottom), white_bottom, highlights
        )
        timings["position"] = (time.perf_counter() - t2) * 1000
        return ReaderState(
            det,
            white_bottom,
            self.positions.current,
            reading is not None,
            probs,
            timings,
        )


def read_image(image: ImageU8, reader: BoardReader | None = None) -> ReaderState:
    """Read a single static screenshot (feeds it enough times to pass the stability check)."""
    reader = reader or BoardReader()
    state = reader.process(image, now=0.0)
    for i in range(reader.positions.stable_frames):
        if state.reading is not None and state.changed:
            break
        state = reader.process(image, now=float(i + 1))
    return state


__all__ = ["BoardReader", "ReaderState", "chess", "read_image"]
