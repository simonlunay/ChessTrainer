"""Find the chess board in a screen capture.

Pipeline:

1. **Candidates.** Canny edges (two blur/threshold passes), then contours. Large square
   contours are board outlines; small square *hole* contours are individual squares, grouped
   into lattices of equal-size neighbours.
2. **Grid fit.** For each candidate, edge-strength projections along x and y are fitted with a
   regular lattice (spacing and phase), scoring lattice lines against the midpoints between them
   so half/double spacings lose.
3. **Checkerboard verification.** Each lattice cell's background colour is the median of eight
   small patches near its border (pieces sit in the middle; coordinates occupy at most two
   corners). Every 8x8 window of the lattice is scored by how cleanly its cells split into the two
   alternating colours, with the top-left square required to be the light one (a8/h1 are light
   squares on every standard board). The best window is the board.

:class:`BoardTracker` caches the result and re-runs detection only when the cached box stops
looking like the same board (moved, resized, covered, theme changed).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from chesstrainer.vision.geometry import ImageU8

F64 = NDArray[np.float64]

MIN_CONTRAST = 10.0  # RGB distance between light and dark square colours
MIN_SCORE = 0.55  # mean checkerboard margin in [-1, 1] for a valid board
VERIFY_SCORE = 0.45  # looser threshold for re-confirming a cached board


@dataclass(frozen=True)
class BoardBox:
    """Board rectangle in frame pixels; edges are the outer board edges."""

    x: float
    y: float
    w: float
    h: float

    @property
    def square(self) -> float:
        return (self.w + self.h) / 16.0

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.x, self.y, self.w, self.h

    def as_int(self) -> tuple[int, int, int, int]:
        return round(self.x), round(self.y), round(self.w), round(self.h)

    def iou(self, other: BoardBox) -> float:
        ix = max(0.0, min(self.x + self.w, other.x + other.w) - max(self.x, other.x))
        iy = max(0.0, min(self.y + self.h, other.y + other.h) - max(self.y, other.y))
        inter = ix * iy
        union = self.w * self.h + other.w * other.h - inter
        return inter / union if union > 0 else 0.0

    def corner_error(self, other: BoardBox) -> float:
        """Largest edge displacement between two boxes, in squares of ``self``."""
        diffs = (
            abs(self.x - other.x),
            abs(self.y - other.y),
            abs(self.x + self.w - other.x - other.w),
            abs(self.y + self.h - other.y - other.h),
        )
        return max(diffs) / self.square

    def translated(self, dx: float, dy: float) -> BoardBox:
        return BoardBox(self.x + dx, self.y + dy, self.w, self.h)

    def scaled(self, f: float) -> BoardBox:
        return BoardBox(self.x * f, self.y * f, self.w * f, self.h * f)


@dataclass
class Detection:
    box: BoardBox
    score: float  # mean checkerboard margin, 1.0 = perfect
    light: F64  # (3,) RGB of light squares
    dark: F64  # (3,) RGB of dark squares
    cell_colors: F64  # (64, 3) per-cell background colour, screen (cell) order

    @property
    def contrast(self) -> float:
        return float(np.linalg.norm(self.light - self.dark))

    def highlighted_cells(self, rel: float = 0.3, floor: float = 15.0) -> list[int]:
        """Cells whose background differs from their square colour (last-move/selection tints)."""
        parity = _PARITY.ravel()
        expected = np.where(parity[:, None] == 0, self.light, self.dark)
        dev = np.linalg.norm(self.cell_colors - expected, axis=1)
        return [int(c) for c in np.flatnonzero(dev > max(rel * self.contrast, floor))]


_PARITY = np.add.outer(np.arange(8), np.arange(8)) % 2  # 0 = light (top-left is light)


# --------------------------------------------------------------------------- colour sampling


def _cell_colors(img: ImageU8, xs: F64, ys: F64) -> F64:
    """Background colour of every lattice cell: (len(ys)-1, len(xs)-1, 3).

    ``xs``/``ys`` are lattice line coordinates within ``img``. Uses an integral image so the cost
    is independent of square size.
    """
    integral = cv2.integral(np.ascontiguousarray(img), sdepth=cv2.CV_64F)  # (H+1, W+1, 3)
    h, w = img.shape[:2]
    x0, x1 = xs[:-1, None], xs[1:, None]
    y0, y1 = ys[:-1, None], ys[1:, None]
    sw, sh = x1 - x0, y1 - y0
    # Eight patches per cell: 4 corners + 4 edge midpoints, inset 7%, 10% of the square wide.
    fx = np.array([0.07, 0.83, 0.07, 0.83, 0.45, 0.45, 0.07, 0.83])
    fy = np.array([0.07, 0.07, 0.83, 0.83, 0.07, 0.83, 0.45, 0.45])
    px0 = (x0 + fx[None, :] * sw)[None, :, :]  # (1, nx, 8)
    py0 = (y0 + fy[None, :] * sh)[:, None, :]  # (ny, 1, 8)
    pw = np.maximum(1.0, 0.1 * sw)[None, :, :]
    ph = np.maximum(1.0, 0.1 * sh)[:, None, :]
    ax0 = np.clip(np.round(px0), 0, w - 1).astype(np.int64)
    ay0 = np.clip(np.round(py0), 0, h - 1).astype(np.int64)
    ax1 = np.clip(np.round(px0 + pw), 1, w).astype(np.int64)
    ay1 = np.clip(np.round(py0 + ph), 1, h).astype(np.int64)
    ax0, ax1 = np.broadcast_arrays(ax0, ax1)
    ay0, ay1 = np.broadcast_arrays(ay0, ay1)
    ny, nx = len(ys) - 1, len(xs) - 1
    ax0, ax1 = np.broadcast_to(ax0, (ny, nx, 8)), np.broadcast_to(ax1, (ny, nx, 8))
    ay0, ay1 = np.broadcast_to(ay0, (ny, nx, 8)), np.broadcast_to(ay1, (ny, nx, 8))
    ax1 = np.maximum(ax1, ax0 + 1)
    ay1 = np.maximum(ay1, ay0 + 1)
    sums = integral[ay1, ax1] - integral[ay0, ax1] - integral[ay1, ax0] + integral[ay0, ax0]
    area = ((ax1 - ax0) * (ay1 - ay0))[..., None]
    means = sums / area  # (ny, nx, 8, 3)
    return np.asarray(np.median(means, axis=2), dtype=np.float64)


def _checker_score(colors: F64) -> tuple[float, F64, F64]:
    """Score an (8, 8, 3) colour grid. Returns (score, light, dark); score -1 if not a board."""
    a = colors[_PARITY == 0]
    b = colors[_PARITY == 1]
    ma, mb = np.median(a, axis=0), np.median(b, axis=0)
    d = float(np.linalg.norm(ma - mb))
    if d < MIN_CONTRAST or ma.sum() <= mb.sum():  # top-left must be the light colour
        return -1.0, ma, mb
    own = np.concatenate([np.linalg.norm(a - ma, axis=1), np.linalg.norm(b - mb, axis=1)])
    other = np.concatenate([np.linalg.norm(a - mb, axis=1), np.linalg.norm(b - ma, axis=1)])
    margin = np.clip((other - own) / d, -1.0, 1.0)
    return float(margin.mean()), ma, mb


def detection_at(frame: ImageU8, box: BoardBox) -> Detection | None:
    """Score a known box (no search). Returns None if it is not a plausible board."""
    xs = box.x + np.arange(9) * box.w / 8
    ys = box.y + np.arange(9) * box.h / 8
    h, w = frame.shape[:2]
    if xs[0] < -1 or ys[0] < -1 or xs[-1] > w + 1 or ys[-1] > h + 1 or box.w < 16:
        return None
    # Work on a crop so the integral image stays small.
    cx0, cy0 = max(0, int(xs[0])), max(0, int(ys[0]))
    cx1, cy1 = min(w, int(np.ceil(xs[-1])) + 1), min(h, int(np.ceil(ys[-1])) + 1)
    crop = frame[cy0:cy1, cx0:cx1]
    colors = _cell_colors(crop, xs - cx0, ys - cy0)
    score, light, dark = _checker_score(colors)
    if score < -0.5:
        return None
    return Detection(box, score, light, dark, colors.reshape(64, 3))


# --------------------------------------------------------------------------- grid fitting


def _edge_profiles(
    region: ImageU8, rows: tuple[int, int], cols: tuple[int, int]
) -> tuple[F64, F64]:
    """Per-column vertical-edge and per-row horizontal-edge strength (max over channels)."""

    def chan_max(d: NDArray[Any]) -> ImageU8:
        return np.asarray(cv2.max(cv2.max(d[..., 0], d[..., 1]), d[..., 2]), dtype=np.uint8)

    # uint8 absdiff/max in OpenCV: same values as int16 numpy, ~10x faster on big regions.
    gx = chan_max(cv2.absdiff(region[:, 1:], region[:, :-1]))  # (H, W-1): col i | col i+1
    gy = chan_max(cv2.absdiff(region[1:], region[:-1]))  # (H-1, W)
    px = gx[rows[0] : rows[1]].sum(axis=0, dtype=np.int64).astype(np.float64)
    py = gy[:, cols[0] : cols[1]].sum(axis=1, dtype=np.int64).astype(np.float64)
    return px, py


def _fit_lattice(
    profile: F64, s0: float, spacing: tuple[float, float] = (0.85, 1.18)
) -> tuple[float, float]:
    """Fit lines ``o + k*s`` to an edge profile. Returns (offset, spacing) in boundary coords.

    Profile index ``i`` is the boundary at coordinate ``i + 1``.
    """
    n = len(profile)
    prof = profile.copy()
    prof[1:] = np.maximum(prof[1:], profile[:-1])  # tolerate +-1px rounding of browser layouts
    prof[:-1] = np.maximum(prof[:-1], profile[1:])
    prof /= prof.max() + 1e-9
    best = (-np.inf, 0.0, s0)
    for s in np.arange(spacing[0] * s0, spacing[1] * s0, 0.25):
        offsets = np.arange(0.0, s, 0.5)
        k = np.arange(int(n / s) + 2)
        pos = offsets[:, None] + k[None, :] * s
        mid = pos + s / 2
        on_ok = pos < n
        mid_ok = mid < n
        on = np.where(on_ok, prof[np.clip(np.round(pos).astype(int), 0, n - 1)], 0.0)
        off = np.where(mid_ok, prof[np.clip(np.round(mid).astype(int), 0, n - 1)], 0.0)
        score = on.sum(1) / np.maximum(on_ok.sum(1), 1) - off.sum(1) / np.maximum(mid_ok.sum(1), 1)
        i = int(score.argmax())
        if score[i] > best[0]:
            best = (float(score[i]), float(offsets[i]), float(s))
    _, o, s = best
    return o + 1.0, s  # +1: profile index -> boundary coordinate


def _lattice_lines(offset: float, s: float, n: int) -> F64:
    k = np.arange(-1, int(n / s) + 2)
    lines = offset + k * s
    return np.asarray(lines[(lines >= -0.3 * s) & (lines <= n + 0.3 * s)], dtype=np.float64)


def refine(
    frame: ImageU8, approx: BoardBox, s0: float | None = None, known_size: bool = False
) -> Detection | None:
    """Snap an approximate board box to the exact square grid and verify it.
    ``known_size``: ``s0`` is the exact square size (only the position is searched)."""
    h, w = frame.shape[:2]
    s0 = s0 or approx.square
    if s0 < 6:
        return None
    # Search far enough that a full 8x8 window fits even when the candidate covers only part
    # of the board (e.g. back ranks full of pieces hide their square outlines).
    mx = 1.6 * s0 + max(0.0, 8 * s0 - approx.w)
    my = 1.6 * s0 + max(0.0, 8 * s0 - approx.h)
    rx0, ry0 = max(0, int(approx.x - mx)), max(0, int(approx.y - my))
    rx1, ry1 = min(w, int(approx.x + approx.w + mx)), min(h, int(approx.y + approx.h + my))
    if rx1 - rx0 < 4 * s0 or ry1 - ry0 < 4 * s0:
        return None
    region = frame[ry0:ry1, rx0:rx1]
    rows = (max(0, int(approx.y - ry0)), min(ry1 - ry0, int(approx.y + approx.h - ry0)))
    cols = (max(0, int(approx.x - rx0)), min(rx1 - rx0, int(approx.x + approx.w - rx0)))
    px, py = _edge_profiles(region, rows, cols)
    spacing = (0.99, 1.01 + 0.25 / s0) if known_size else (0.85, 1.18)
    ox, sx = _fit_lattice(px, s0, spacing)
    oy, sy = _fit_lattice(py, s0, spacing)
    xs = _lattice_lines(ox, sx, region.shape[1])
    ys = _lattice_lines(oy, sy, region.shape[0])
    if len(xs) < 9 or len(ys) < 9:
        return None
    colors = _cell_colors(region, np.clip(xs, 0, region.shape[1]), np.clip(ys, 0, region.shape[0]))
    best: tuple[float, int, int, F64, F64] | None = None
    for j in range(len(ys) - 8):
        for i in range(len(xs) - 8):
            score, light, dark = _checker_score(colors[j : j + 8, i : i + 8])
            if best is None or score > best[0]:
                best = (score, i, j, light, dark)
    if best is None or best[0] < MIN_SCORE:
        return None
    score, i, j, light, dark = best
    box = BoardBox(
        rx0 + max(0.0, float(xs[i])),
        ry0 + max(0.0, float(ys[j])),
        float(min(xs[i + 8], region.shape[1]) - max(xs[i], 0)),
        float(min(ys[j + 8], region.shape[0]) - max(ys[j], 0)),
    )
    if not 0.9 < box.w / box.h < 1.11:
        return None
    return Detection(box, score, light, dark, colors[j : j + 8, i : i + 8].reshape(64, 3))


# --------------------------------------------------------------------------- candidates


# (blur kernel, Canny low, Canny high). The second pass sees through wood/marble grain, whose
# edges are weaker than the light/dark square boundaries.
_EDGE_PASSES = ((5, 20, 50), (9, 40, 100))


def _candidates(frame: ImageU8, min_board: int) -> list[tuple[BoardBox, float]]:
    """Approximate board boxes (box, square size) from contours of the edge map.

    Per edge pass: lattices of square cells first (most reliable), then large square outlines.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    min_cell = max(8, min_board // 8 - 4)
    out: list[tuple[BoardBox, float]] = []
    for n_pass, (kernel, lo, hi) in enumerate(_EDGE_PASSES):
        if n_pass == 0:
            # Per colour channel: some themes have light and dark squares of nearly equal
            # brightness (pastels, white on pale cyan) that grayscale cannot tell apart.
            blurred = cv2.GaussianBlur(frame, (kernel, kernel), 0)
            edges = np.maximum.reduce(
                [cv2.Canny(np.ascontiguousarray(blurred[:, :, c]), lo, hi) for c in range(3)]
            )
        else:
            edges = cv2.Canny(cv2.GaussianBlur(gray, (kernel, kernel), 0), lo, hi)
        # Thicken 1px edges so each square's boundary is a closed band whose hole is the square.
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        outlines: list[tuple[BoardBox, float]] = []
        cells: list[tuple[float, float, float]] = []  # (cx, cy, size)
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw < min_cell or ch < min_cell:
                continue
            if abs(cw - ch) > 0.12 * max(cw, ch) + 2:
                continue
            if cv2.contourArea(cnt) < 0.8 * cw * ch:
                continue
            cells.append((x + cw / 2, y + ch / 2, (cw + ch) / 2 + 1))
            if cw >= min_board:
                outlines.append((BoardBox(x, y, cw, ch), (cw + ch) / 16))
        outlines.sort(key=lambda c: c[0].w, reverse=True)
        out += _cell_groups(cells) + outlines
    return out


def _near_step(v: float) -> bool:
    """True if a centre offset (in squares) is ~0 or ~1: neighbouring lattice positions."""
    return abs(v) < 0.2 or abs(v - 1) < 0.2


def _cell_groups(
    cells: list[tuple[float, float, float]], min_cells: int = 10
) -> list[tuple[BoardBox, float]]:
    """Group square contours into lattices of equal-size orthogonal/diagonal neighbours."""
    if len(cells) < min_cells:
        return []
    arr = np.array(cells)
    parent = list(range(len(arr)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    buckets: dict[tuple[int, int], list[int]] = {}
    grid = float(np.median(arr[:, 2])) if len(arr) else 1.0
    grid = max(grid, 8.0)
    for i, (cx, cy, _) in enumerate(arr):
        buckets.setdefault((int(cx // grid), int(cy // grid)), []).append(i)
    for i, (cx, cy, s) in enumerate(arr):
        gx, gy = int(cx // grid), int(cy // grid)
        reach = int(np.ceil(1.3 * s / grid))
        for bx in range(gx - reach, gx + reach + 1):
            for by in range(gy - reach, gy + reach + 1):
                for j in buckets.get((bx, by), ()):
                    if j <= i:
                        continue
                    sj = arr[j, 2]
                    if abs(sj - s) > 0.12 * max(s, sj):
                        continue
                    dx, dy = abs(arr[j, 0] - cx) / s, abs(arr[j, 1] - cy) / s
                    if _near_step(dx) and _near_step(dy) and (dx > 0.5 or dy > 0.5):
                        parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(arr)):
        groups.setdefault(find(i), []).append(i)
    out = []
    for members in groups.values():
        if len(members) < min_cells:
            continue
        g = arr[members]
        s = float(np.median(g[:, 2]))
        x0, y0 = g[:, 0].min() - s / 2, g[:, 1].min() - s / 2
        x1, y1 = g[:, 0].max() + s / 2, g[:, 1].max() + s / 2
        out.append((BoardBox(x0, y0, x1 - x0, y1 - y0), s))
    return out


def detect_boards(frame: ImageU8, min_board: int = 96) -> list[Detection]:
    """All boards found in ``frame``, largest first, overlapping duplicates removed."""
    scale = 1.0
    work: ImageU8 = frame
    if max(frame.shape[:2]) > 2200:  # 4K: find candidates at half resolution
        scale = 0.5
        work = np.asarray(
            cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), dtype=np.uint8
        )
    found: list[Detection] = []
    for approx, s in _candidates(work, int(min_board * scale)):
        approx = approx.scaled(1 / scale)
        if any(_inside(approx, f.box) for f in found):
            continue  # a square of a board we already have
        det = refine(frame, approx, s / scale)
        if det is None or det.box.w < min_board:
            continue
        if any(det.box.iou(f.box) > 0.5 for f in found):
            for k, f in enumerate(found):
                if det.box.iou(f.box) > 0.5 and det.score > f.score:
                    found[k] = det
            continue
        found.append(det)
    found.sort(key=lambda d: d.box.w * d.box.h, reverse=True)
    return found


def _inside(a: BoardBox, b: BoardBox, tol: float = 0.05) -> bool:
    """True if ``a`` lies within ``b`` (with a small tolerance)."""
    t = tol * b.w
    return (
        a.x >= b.x - t
        and a.y >= b.y - t
        and a.x + a.w <= b.x + b.w + t
        and a.y + a.h <= b.y + b.h + t
    )


def detect_board(frame: ImageU8, min_board: int = 96) -> Detection | None:
    """The main (largest) board in ``frame``, or None."""
    boards = detect_boards(frame, min_board)
    return boards[0] if boards else None


# --------------------------------------------------------------------------- tracking


class BoardTracker:
    """Caches the board location and only re-detects when the cached box stops matching.

    When the cached board disappears (window dragged), the same board at the same size is first
    looked for with template matching on a downscaled frame (~10 ms) before falling back to a
    full detection (~100-250 ms on a 2560x1600 screen).
    """

    THUMB = 48  # side of the board thumbnail used to relocate a moved board

    def __init__(
        self,
        min_board: int = 96,
        redetect_interval: float = 0.5,
        locked: BoardBox | None = None,
    ) -> None:
        self.min_board = min_board
        self.redetect_interval = redetect_interval
        self.locked = locked
        self.current: Detection | None = None
        self._lost: Detection | None = None  # board that just failed verification
        self._thumb: ImageU8 | None = None
        self._last_full = -np.inf
        self.full_detections = 0  # for diagnostics/tests
        self.relocations = 0

    def reset(self) -> None:
        self.current = None
        self._lost = None
        self._thumb = None
        self._last_full = -np.inf

    def update(
        self, frame: ImageU8, now: float | None = None, full_search: bool = True
    ) -> Detection | None:
        """Verify the cached board, else search for one. ``full_search=False`` only verifies
        (for frames where only the board's neighbourhood is fresh)."""
        now = time.monotonic() if now is None else now
        if self.current is not None:
            det = self._verify(frame, self.current)
            if det is not None:
                self.current = det
                self._thumb = _board_thumb(frame, det.box, self.THUMB)
                return det
            self._lost, self.current = self.current, None
        if not full_search:
            return None
        if self.locked is not None:
            self.current = refine(frame, self.locked) or detection_at(frame, self.locked)
            return self.current
        if self._lost is not None and self._thumb is not None:
            moved = self._relocate(frame, self._lost, self._thumb)
            if moved is not None:
                self.relocations += 1
                self.current = moved
                self._thumb = _board_thumb(frame, moved.box, self.THUMB)
                return moved
        if now - self._last_full < self.redetect_interval:
            return None
        self._last_full = now
        self.full_detections += 1
        self.current = detect_board(frame, self.min_board)
        self._lost = None
        self._thumb = _board_thumb(frame, self.current.box, self.THUMB) if self.current else None
        return self.current

    @staticmethod
    def _verify(frame: ImageU8, prev: Detection) -> Detection | None:
        det = detection_at(frame, prev.box)
        if det is None or det.score < VERIFY_SCORE:
            return None
        drift = max(np.linalg.norm(det.light - prev.light), np.linalg.norm(det.dark - prev.dark))
        if drift > 0.5 * prev.contrast:  # different theme or a different board at the same spot
            return None
        return det

    @staticmethod
    def _relocate(frame: ImageU8, prev: Detection, thumb: ImageU8) -> Detection | None:
        """The same board, same size, somewhere else on the screen (a dragged window)."""
        f = thumb.shape[1] / prev.box.w
        small = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), None, fx=f, fy=f, interpolation=cv2.INTER_AREA
        )
        if small.shape[0] < thumb.shape[0] or small.shape[1] < thumb.shape[1]:
            return None
        res = cv2.matchTemplate(small, thumb, cv2.TM_CCOEFF_NORMED)
        _, best, _, loc = cv2.minMaxLoc(res)
        if best < 0.7:
            return None
        approx = BoardBox(loc[0] / f, loc[1] / f, prev.box.w, prev.box.h)
        det = refine(frame, approx, prev.box.square, known_size=True)
        if det is None or abs(det.box.w - prev.box.w) > 0.02 * prev.box.w:
            return None  # resized: leave it to full detection
        drift = max(np.linalg.norm(det.light - prev.light), np.linalg.norm(det.dark - prev.dark))
        return None if drift > 0.5 * prev.contrast else det


def _board_thumb(frame: ImageU8, box: BoardBox, side: int) -> ImageU8:
    x, y, w, h = box.as_int()
    crop = frame[max(0, y) : y + h, max(0, x) : x + w]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    return np.asarray(cv2.resize(gray, (side, side), interpolation=cv2.INTER_AREA), dtype=np.uint8)
