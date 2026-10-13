"""Decide which side is at the bottom of the board from where the pieces are.

Two cues, both scored so that positive means "white at the bottom":

* **Coordinate labels** (when the site draws them inside the edge squares). The rank label in the
  bottom row is "1" with white at the bottom and "8" with black at the bottom; an "8" has far more
  ink than a "1" in any font, so comparing the ink of the top and bottom rank labels is decisive.
* **Piece placement**: white pawns sit on lower ranks than black pawns (pawns never move
  backwards), kings start on their own back rank, and each side's pieces cluster toward its own
  edge. Weak in endgames, which is why the tracker keeps a decision once it is confident.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from chesstrainer.vision.detect import Detection
from chesstrainer.vision.geometry import ImageU8
from chesstrainer.vision.labels import CLASS_NAMES

_WHITE = np.array([n.isupper() for n in CLASS_NAMES])
_BLACK = np.array([n.islower() and n != "empty" for n in CLASS_NAMES])


def orientation_score(cell_labels: NDArray[np.int64]) -> float:
    """> 0 favours white at the bottom, < 0 black at the bottom; magnitude = confidence.

    ``cell_labels`` are class indices in screen order (row 0 = top).
    """
    rows = np.arange(64) // 8
    height = 7 - rows  # 0 = bottom row, 7 = top row
    names = np.array(CLASS_NAMES)[cell_labels]
    score = 0.0

    wp, bp = height[names == "P"], height[names == "p"]
    if len(wp) and len(bp):
        score += 1.5 * (bp.mean() - wp.mean())
    elif len(wp) or len(bp):  # one side's pawns only: compare with the board middle
        score += 0.75 * ((3.5 - wp.mean()) if len(wp) else (bp.mean() - 3.5))

    wk, bk = height[names == "K"], height[names == "k"]
    if len(wk) == 1 and len(bk) == 1:
        score += 0.5 * (bk[0] - wk[0]) / 7

    white = _WHITE[cell_labels] & (names != "P") & (names != "K")
    black = _BLACK[cell_labels] & (names != "p") & (names != "k")
    if white.any() and black.any():
        score += 0.5 * (height[black].mean() - height[white].mean()) / 3.5
    return float(score)


# Corner boxes (fractions of a square) where sites put rank labels.
_CORNERS = (
    (0.02, 0.02, 0.34, 0.38),  # top-left
    (0.66, 0.02, 0.98, 0.38),  # top-right
    (0.02, 0.62, 0.34, 0.98),  # bottom-left
    (0.66, 0.62, 0.98, 0.98),  # bottom-right
)


def _glyph_mask(
    frame: ImageU8, det: Detection, cell: int, corner: tuple[float, ...]
) -> NDArray[np.uint8]:
    """Binary mask of pixels in a corner of ``cell`` that differ from the square's background."""
    row, col = divmod(cell, 8)
    b = det.box
    sw, sh = b.w / 8, b.h / 8
    x0, x1 = round(b.x + (col + corner[0]) * sw), round(b.x + (col + corner[2]) * sw)
    y0, y1 = round(b.y + (row + corner[1]) * sh), round(b.y + (row + corner[3]) * sh)
    patch = frame[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)].astype(np.float64)
    if patch.size == 0:
        return np.zeros((1, 1), np.uint8)
    dev = np.linalg.norm(patch - det.cell_colors[cell], axis=2)
    return np.asarray(dev > 0.4 * det.contrast, dtype=np.uint8)


def _holes(mask: NDArray[np.uint8]) -> int:
    """Enclosed holes in the glyph: 2 for "8", 1 for "6"/"9", 0 for "1"/"3"."""
    if mask.sum() < 4:
        return -1
    padded = np.pad(mask, 1)
    contours, hierarchy = cv2.findContours(padded, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return 0
    min_area = max(3.0, 0.002 * mask.size)
    return sum(
        1
        for c, h in zip(contours, hierarchy[0], strict=True)
        if h[3] >= 0 and cv2.contourArea(c) >= min_area  # has a parent = it is a hole
    )


def coordinate_score(frame: ImageU8, det: Detection) -> float | None:
    """Orientation evidence from rank labels, or None if the board shows no usable labels.

    With white at the bottom the edge column reads 8,7,6,5,4,3,2,1 from the top: the top label
    ("8") has two holes and the bottom one ("1") none, and row 2 ("6") has a hole while row 5
    ("3") has none. With black at the bottom it is the reverse. Returns the hole-count difference
    (top minus bottom, both pairs), so +3 means clearly white at the bottom.
    """
    min_ink = max(4, int(0.004 * det.box.square**2))
    best: tuple[int, int] | None = None  # (labelled rows, score)
    for col in (0, 7):
        for corner in _CORNERS:
            masks = [_glyph_mask(frame, det, row * 8 + col, corner) for row in range(8)]
            present = sum(int(m.sum()) >= min_ink for m in masks)
            if present < 7:  # labels appear on every rank of one edge column
                continue
            holes = [_holes(m) for m in masks]
            if min(holes[0], holes[7], holes[2], holes[5]) < 0:
                continue
            score = (holes[0] - holes[7]) + (holes[2] - holes[5])
            if best is None or abs(score) > abs(best[1]):
                best = (present, score)
    return None if best is None else float(best[1])


class OrientationTracker:
    """Auto orientation with hysteresis, or a fixed side when the user set one."""

    def __init__(self, mode: str = "auto", switch_after: int = 3, confident: float = 1.5) -> None:
        self.mode = mode
        self.white_bottom = mode != "black"
        self.decided = mode != "auto"
        self.switch_after = switch_after
        self.confident = confident
        self._contrary = 0

    def flip(self) -> None:
        """Manual flip: overrides auto mode until :meth:`reset`."""
        self.white_bottom = not self.white_bottom
        self.mode = "white" if self.white_bottom else "black"
        self.decided = True

    def reset(self, mode: str | None = None) -> None:
        self.__init__(mode or self.mode, self.switch_after, self.confident)  # type: ignore[misc]

    def update(self, cell_labels: NDArray[np.int64], coords: float | None = None) -> bool:
        """``coords`` is :func:`coordinate_score` for this frame (None if no labels found)."""
        if self.mode != "auto":
            return self.white_bottom
        # Both cues add up: clear labels decide ambiguous endgames, while a strongly structured
        # position (e.g. the opening) outweighs a misread label on a tiny board.
        s = orientation_score(cell_labels) + (coords or 0.0)
        says_white = s > 0
        if not self.decided:
            if abs(s) >= 0.5:  # first usable evidence decides
                self.white_bottom = says_white
                self.decided = abs(s) >= self.confident
            return self.white_bottom
        if says_white != self.white_bottom and abs(s) >= self.confident:
            self._contrary += 1
            if self._contrary >= self.switch_after:
                self.white_bottom = says_white
                self._contrary = 0
        else:
            self._contrary = 0
        return self.white_bottom
