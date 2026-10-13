"""Synthetic chess board renderer used to train the square classifier.

Each sample is a random legal position drawn with a random piece set, board theme, square size
and orientation, decorated with the clutter real chess UIs add (last-move tints, check glow,
move dots, hover outlines, coordinates, arrows), placed on a random page background, and then
degraded (rescaling, blur, JPEG, colour shifts, noise). Square crops are cut with the same
:func:`~chesstrainer.vision.geometry.split_board` used at runtime, with simulated detection error.
"""

from __future__ import annotations

import colorsys
import io
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import chess
import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from chesstrainer.vision.assets import BOARD_DIR, PIECE_DIR, PIECE_FILES
from chesstrainer.vision.geometry import CROP_SIZE, ImageU8, cell_to_square, split_board
from chesstrainer.vision.labels import EMPTY, piece_to_label

RGB = tuple[int, int, int]

#: Light/dark square colours of common site themes (lichess and chess.com defaults and others).
PALETTES: dict[str, tuple[RGB, RGB]] = {
    "lichess-brown": ((240, 217, 181), (181, 136, 99)),
    "lichess-blue": ((222, 227, 230), (140, 162, 173)),
    "lichess-green": ((255, 255, 221), (134, 166, 102)),
    "lichess-ic": ((236, 236, 236), (193, 193, 189)),
    "chesscom-green": ((235, 236, 208), (115, 149, 82)),
    "chesscom-brown": ((237, 214, 176), (184, 135, 98)),
    "chesscom-tan": ((238, 216, 192), (171, 122, 101)),
    "chesscom-dark": ((180, 180, 180), (110, 110, 110)),
    "chesscom-blue": ((234, 233, 210), (75, 115, 153)),
    "chesscom-purple": ((239, 239, 239), (136, 119, 183)),
    "chess24": ((234, 233, 210), (147, 161, 172)),
    "coral": ((177, 228, 185), (112, 162, 163)),
}
HELD_OUT_PALETTES: frozenset[str] = frozenset({"chesscom-blue", "chesscom-purple"})


@dataclass
class StyleSplit:
    """Which assets a generator may draw from. Train and held-out test use disjoint styles."""

    piece_sets: list[str]
    textures: list[str]
    palettes: list[str]
    random_colors: bool = True


class AssetLibrary:
    """Loads piece PNGs and board textures and caches resized copies."""

    def __init__(self, piece_dir: Path = PIECE_DIR, board_dir: Path = BOARD_DIR) -> None:
        self.piece_dir = piece_dir
        self.board_dir = board_dir

    # Caches are per process (one library per DataLoader worker), so keep them small.
    @lru_cache(maxsize=1024)  # noqa: B019 - library lives for the whole process
    def piece(self, piece_set: str, stem: str, size: int) -> Image.Image:
        img = self._piece_full(piece_set, stem)
        return img.resize((size, size), Image.Resampling.LANCZOS)

    @lru_cache(maxsize=512)  # noqa: B019
    def _piece_full(self, piece_set: str, stem: str) -> Image.Image:
        img = Image.open(self.piece_dir / piece_set / f"{stem}.png").convert("RGBA")
        img.load()
        return img

    def texture(self, name: str, size: int) -> Image.Image:
        full = self._texture_full(name)
        return Image.fromarray(cv2.resize(full, (size, size), interpolation=cv2.INTER_AREA))

    @lru_cache(maxsize=32)  # noqa: B019
    def _texture_full(self, name: str) -> ImageU8:
        with Image.open(self.board_dir / name) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)


@dataclass
class Scene:
    """A rendered board on a page background."""

    image: ImageU8  # HxWx3 RGB
    bbox: tuple[float, float, float, float]  # board region (x, y, w, h) in ``image``
    labels: NDArray[np.int64]  # (64,) class per screen cell
    white_bottom: bool
    board: chess.Board = field(repr=False)


# --------------------------------------------------------------------------- positions


def random_position(rng: np.random.Generator, max_plies: int = 160) -> chess.Board:
    """A legal position reached by a random game with a mild preference for captures.

    The capture bias thins the board out so material-light endgames are well represented.
    """
    board = chess.Board()
    plies = int(rng.integers(0, max_plies + 1))
    capture_bias = float(rng.uniform(0.0, 0.8))
    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves:
            break
        captures = [m for m in moves if board.is_capture(m)]
        pool = captures if captures and rng.random() < capture_bias else moves
        board.push(pool[int(rng.integers(len(pool)))])
    return board


# --------------------------------------------------------------------------- colours


def _hsv(h: float, s: float, v: float) -> RGB:
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, min(max(s, 0.0), 1.0), min(max(v, 0.0), 1.0))
    return int(r * 255), int(g * 255), int(b * 255)


def _jitter(color: RGB, rng: np.random.Generator, amount: int = 10) -> RGB:
    c = np.clip(np.array(color) + rng.integers(-amount, amount + 1, size=3), 0, 255)
    return int(c[0]), int(c[1]), int(c[2])


def _random_square_colors(rng: np.random.Generator) -> tuple[RGB, RGB]:
    hue = float(rng.random())
    light = _hsv(hue + rng.normal(0, 0.03), rng.uniform(0.0, 0.3), rng.uniform(0.78, 1.0))
    dark = _hsv(hue + rng.normal(0, 0.05), rng.uniform(0.15, 0.65), rng.uniform(0.3, 0.72))
    return light, dark


# --------------------------------------------------------------------------- rendering


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _tint(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], rgba: tuple[int, ...]) -> None:
    draw.rectangle(box, fill=rgba)


def render_scene(
    rng: np.random.Generator,
    lib: AssetLibrary,
    split: StyleSplit,
    board: chess.Board | None = None,
    square_px: int | None = None,
    decorate: bool = True,
) -> Scene:
    """Render one random board scene (before photometric degradation)."""
    if board is None:
        board = random_position(rng)
    sq = square_px if square_px is not None else int(rng.integers(22, 112))
    white_bottom = bool(rng.random() < 0.6)
    size = sq * 8
    piece_set = split.piece_sets[int(rng.integers(len(split.piece_sets)))]

    # Board base: texture image or two-colour checkerboard.
    use_texture = split.textures and rng.random() < 0.25
    light: RGB
    dark: RGB
    if use_texture:
        tex_name = split.textures[int(rng.integers(len(split.textures)))]
        base = lib.texture(tex_name, size).copy()
        light, dark = (240, 220, 190), (150, 110, 80)  # only used for coordinate text
    else:
        if split.random_colors and (not split.palettes or rng.random() < 0.35):
            light, dark = _random_square_colors(rng)
        else:
            name = split.palettes[int(rng.integers(len(split.palettes)))]
            light, dark = (_jitter(c, rng) for c in PALETTES[name])
        base = Image.new("RGB", (size, size), light)
        bd = ImageDraw.Draw(base)
        for cell in range(64):
            r, c = divmod(cell, 8)
            if (r + c) % 2 == 1:
                bd.rectangle((c * sq, r * sq, (c + 1) * sq - 1, (r + 1) * sq - 1), fill=dark)
        if rng.random() < 0.2:  # subtle grain, like wood/marble themes
            noise = rng.standard_normal((size, size, 1), dtype=np.float32) * rng.uniform(3, 10)
            arr = np.clip(np.asarray(base, dtype=np.float32) + noise, 0, 255).astype(np.uint8)
            base = Image.fromarray(arr)

    canvas = base.convert("RGBA")
    under = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ud = ImageDraw.Draw(under)

    def cell_box(cell: int) -> tuple[int, int, int, int]:
        r, c = divmod(cell, 8)
        return c * sq, r * sq, (c + 1) * sq - 1, (r + 1) * sq - 1

    cells = [cell_to_square(i, white_bottom) for i in range(64)]
    occupied = [i for i in range(64) if board.piece_at(cells[i]) is not None]
    empty = [i for i in range(64) if board.piece_at(cells[i]) is None]

    if decorate:
        # Last-move highlight (two squares) and/or a selected/premove square.
        if rng.random() < 0.6:
            color = _jitter((205, 210, 106), rng, 40) + (int(rng.integers(70, 170)),)
            picks = rng.choice(64, size=2, replace=False)
            for i in picks:
                _tint(ud, cell_box(int(i)), color)
        if rng.random() < 0.15:
            color = _jitter((20, 85, 30), rng, 60) + (int(rng.integers(60, 140)),)
            _tint(ud, cell_box(int(rng.integers(64))), color)
        # Check glow under a king.
        if rng.random() < 0.1:
            king_cells = [i for i in occupied if board.piece_at(cells[i]).piece_type == chess.KING]  # type: ignore[union-attr]
            if king_cells:
                x0, y0, x1, y1 = cell_box(int(rng.choice(king_cells)))
                for k in range(sq // 2, 0, -1):
                    a = int(220 * (1 - k / (sq / 2)))
                    ud.ellipse(
                        (x0 + sq / 2 - k, y0 + sq / 2 - k, x0 + sq / 2 + k, y0 + sq / 2 + k),
                        fill=(255, 0, 0, a),
                    )
        # Hover / drag outline.
        if rng.random() < 0.1:
            x0, y0, x1, y1 = cell_box(int(rng.integers(64)))
            w = max(1, sq // 16)
            ud.rectangle((x0, y0, x1, y1), outline=(255, 255, 255, 200), width=w)
    canvas = Image.alpha_composite(canvas, under)

    # Pieces.
    labels = np.zeros(64, dtype=np.int64)
    scale = float(rng.uniform(0.78, 1.0))
    for i in occupied:
        piece = board.piece_at(cells[i])
        labels[i] = piece_to_label(piece)
        stem = PIECE_FILES[labels[i] - 1]
        psz = max(4, int(round(sq * scale * rng.uniform(0.97, 1.03))))
        img = lib.piece(piece_set, stem, psz)
        x0, y0, _, _ = cell_box(i)
        jitter = sq * 0.03
        px = int(round(x0 + (sq - psz) / 2 + rng.uniform(-jitter, jitter)))
        py = int(round(y0 + (sq - psz) / 2 + rng.uniform(-jitter, jitter)))
        canvas.alpha_composite(img, (max(px, 0), max(py, 0)))
    assert all(labels[i] == EMPTY for i in empty)

    if decorate:
        over = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(over)
        # Legal-move markers: dots on empty squares, rings/corners on occupied squares.
        if rng.random() < 0.15:
            dot = (20, 85, 30, int(rng.integers(60, 140)))
            for i in rng.choice(64, size=int(rng.integers(1, 10)), replace=False):
                x0, y0, x1, y1 = cell_box(int(i))
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                if labels[int(i)] == EMPTY:
                    rad = sq * rng.uniform(0.1, 0.18)
                    od.ellipse((cx - rad, cy - rad, cx + rad, cy + rad), fill=dot)
                else:
                    od.ellipse((x0, y0, x1, y1), outline=dot, width=max(2, sq // 12))
        # Coordinates inside edge squares, in the opposite square colour.
        if rng.random() < 0.6:
            font = _font(max(7, int(sq * rng.uniform(0.16, 0.26))))
            files = "abcdefgh" if white_bottom else "hgfedcba"
            ranks = "87654321" if white_bottom else "12345678"
            for c in range(8):
                fill = light if (7 + c) % 2 == 1 else dark
                od.text(
                    (c * sq + sq - 2, 8 * sq - 2),
                    files[c],
                    fill=fill + (255,),
                    font=font,
                    anchor="rd",
                )
            for r in range(8):
                fill = light if r % 2 == 1 else dark
                od.text((2, r * sq + 1), ranks[r], fill=fill + (255,), font=font, anchor="lt")
        # User/engine arrows drawn on top of pieces.
        for _ in range(int(rng.integers(0, 3)) if rng.random() < 0.15 else 0):
            a, b = rng.choice(64, size=2, replace=False)
            (ay, ax), (by, bx) = divmod(int(a), 8), divmod(int(b), 8)
            p0 = ((ax + 0.5) * sq, (ay + 0.5) * sq)
            p1 = ((bx + 0.5) * sq, (by + 0.5) * sq)
            col = [(21, 120, 27), (230, 120, 0), (200, 30, 30), (0, 48, 136)][int(rng.integers(4))]
            od.line((p0, p1), fill=col + (int(rng.integers(120, 200)),), width=max(2, sq // 6))
        canvas = Image.alpha_composite(canvas, over)

    # Place the board on a page background with a random margin (detection crops can spill).
    margin = int(sq * rng.uniform(0.3, 1.0))
    bg = _jitter(_hsv(rng.random(), rng.uniform(0, 0.3), rng.uniform(0.05, 0.95)), rng, 5)
    page = Image.new("RGB", (size + 2 * margin, size + 2 * margin), bg)
    if rng.random() < 0.3:  # a frame around the board
        fd = ImageDraw.Draw(page)
        fw = int(rng.integers(1, max(2, margin)))
        fd.rectangle(
            (margin - fw, margin - fw, margin + size + fw - 1, margin + size + fw - 1),
            fill=_jitter(dark, rng, 30),
        )
    page.paste(canvas.convert("RGB"), (margin, margin))
    return Scene(
        image=np.asarray(page, dtype=np.uint8).copy(),
        bbox=(float(margin), float(margin), float(size), float(size)),
        labels=labels,
        white_bottom=white_bottom,
        board=board,
    )


# --------------------------------------------------------------------------- degradation


def degrade(scene: Scene, rng: np.random.Generator) -> Scene:
    """Photometric/resampling degradation imitating screenshots of real sites."""
    img: NDArray[Any] = scene.image
    x, y, w, h = scene.bbox
    # Non-integer rescale (browser zoom, HiDPI scaling).
    if rng.random() < 0.7:
        f = float(rng.uniform(0.6, 1.4))
        new_w = max(8, int(round(img.shape[1] * f)))
        new_h = max(8, int(round(img.shape[0] * f)))
        fx, fy = new_w / img.shape[1], new_h / img.shape[0]
        interp = (
            cv2.INTER_AREA
            if f < 1
            else int(rng.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST]))
        )
        img = cv2.resize(img, (new_w, new_h), interpolation=interp)
        x, y, w, h = x * fx, y * fy, w * fx, h * fy
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), float(rng.uniform(0.3, 1.2)))
    if rng.random() < 0.5:  # brightness / contrast / gamma / white balance (night-light filters)
        ramp = np.arange(256, dtype=np.float32) / 255.0
        ramp = ramp ** float(rng.uniform(0.75, 1.3)) * float(rng.uniform(0.8, 1.2))
        ramp = ramp + float(rng.uniform(-0.08, 0.08))
        luts = [
            np.clip(ramp * float(rng.uniform(0.92, 1.08)) * 255.0, 0, 255).astype(np.uint8)
            for _ in range(3)
        ]
        img = cv2.merge([cv2.LUT(ch, lut) for ch, lut in zip(cv2.split(img), luts, strict=True)])
    if rng.random() < 0.3:  # saturation
        gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        alpha = float(rng.uniform(0.6, 1.3))
        img = cv2.addWeighted(img, alpha, gray, 1.0 - alpha, 0.0)
    if rng.random() < 0.2:
        noise = rng.standard_normal(img.shape, dtype=np.float32) * float(rng.uniform(2, 8))
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.4:
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=int(rng.integers(30, 95)))
        img = np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)
    return Scene(img, (x, y, w, h), scene.labels, scene.white_bottom, scene.board)


def detection_offsets(
    rng: np.random.Generator, max_edge: float = 0.12, max_cell: float = 0.04
) -> NDArray[np.float64]:
    """Per-cell crop offsets for an imperfect board bounding box.

    A board-level error moves each bbox edge by up to ``max_edge`` squares; it is interpolated
    across the grid so cells near a misplaced edge are shifted most. Each cell also gets an
    independent error of up to ``max_cell`` squares.
    """
    left, top, right, bottom = rng.uniform(-max_edge, max_edge, size=4)
    offsets = np.empty((64, 4), dtype=np.float64)
    for cell in range(64):
        r, c = divmod(cell, 8)
        offsets[cell, 0] = left + (right - left) * c / 8
        offsets[cell, 2] = left + (right - left) * (c + 1) / 8
        offsets[cell, 1] = top + (bottom - top) * r / 8
        offsets[cell, 3] = top + (bottom - top) * (r + 1) / 8
    offsets += rng.uniform(-max_cell, max_cell, size=(64, 4))
    return offsets


def sample_crops(
    rng: np.random.Generator,
    lib: AssetLibrary,
    split: StyleSplit,
    empty_keep: float = 0.3,
    crop_size: int = CROP_SIZE,
) -> tuple[ImageU8, NDArray[np.int64]]:
    """Render one degraded board and return its (subsampled) square crops and labels."""
    scene = degrade(render_scene(rng, lib, split), rng)
    crops = split_board(scene.image, crop_size, detection_offsets(rng), scene.bbox)
    keep = (scene.labels != EMPTY) | (rng.random(64) < empty_keep)
    crops, labels = crops[keep], scene.labels[keep]
    flip = rng.random(len(crops)) < 0.5  # mirrored knights etc. are still the same piece
    crops[flip] = crops[flip, :, ::-1]
    return crops, labels
