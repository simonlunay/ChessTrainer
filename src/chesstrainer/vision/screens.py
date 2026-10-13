"""Synthetic full-screen captures for evaluating board detection.

A page of random UI clutter (panels, text, buttons, noise images, smaller "thumbnail" boards like
a site's TV or puzzle sidebar) with one main board pasted at a random position and size. The
ground-truth board box is known exactly.
"""

from __future__ import annotations

import string
from dataclasses import dataclass

import chess
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from chesstrainer.vision.detect import BoardBox
from chesstrainer.vision.geometry import ImageU8
from chesstrainer.vision.synth import AssetLibrary, Scene, StyleSplit, _font, degrade, render_scene


@dataclass
class Screen:
    image: ImageU8
    box: BoardBox  # main board
    labels: NDArray[np.int64]  # (64,) per screen cell of the main board
    white_bottom: bool
    board: chess.Board


def _clutter(rng: np.random.Generator, page: Image.Image) -> None:
    d = ImageDraw.Draw(page)
    w, h = page.size
    for _ in range(int(rng.integers(4, 12))):  # panels
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        x1, y1 = x0 + int(rng.integers(80, w // 2)), y0 + int(rng.integers(40, h // 2))
        fill = tuple(int(v) for v in rng.integers(0, 256, 3))
        d.rectangle(
            (x0, y0, x1, y1), fill=fill, outline=tuple(int(v) for v in rng.integers(0, 256, 3))
        )
    for _ in range(int(rng.integers(10, 40))):  # text lines and buttons
        x, y = int(rng.integers(0, w - 100)), int(rng.integers(0, h - 20))
        size = int(rng.integers(10, 22))
        text = "".join(
            rng.choice(list(string.ascii_letters + " 0123456789."), int(rng.integers(5, 40)))
        )
        if rng.random() < 0.3:
            d.rounded_rectangle(
                (x - 6, y - 4, x + size * len(text) // 2 + 6, y + size + 4),
                6,
                fill=tuple(int(v) for v in rng.integers(0, 256, 3)),
            )
        d.text((x, y), text, fill=tuple(int(v) for v in rng.integers(0, 256, 3)), font=_font(size))
    for _ in range(int(rng.integers(0, 4))):  # photos / avatars
        pw, ph = int(rng.integers(30, 300)), int(rng.integers(30, 300))
        noise = rng.integers(0, 256, (ph // 8 + 1, pw // 8 + 1, 3), dtype=np.uint8)
        img = Image.fromarray(noise).resize((pw, ph), Image.Resampling.BILINEAR)
        page.paste(img, (int(rng.integers(0, w - pw)), int(rng.integers(0, h - ph))))


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 10) -> bool:
    return not (
        a[2] + pad <= b[0] or b[2] + pad <= a[0] or a[3] + pad <= b[1] or b[3] + pad <= a[1]
    )


def render_screen(
    rng: np.random.Generator,
    lib: AssetLibrary,
    split: StyleSplit,
    size: tuple[int, int] = (1920, 1080),
    board_image: ImageU8 | None = None,
    distractors: bool = True,
) -> Screen:
    """A random screen with one main board. ``board_image`` (a tight board crop, e.g. a real
    site render) replaces the synthetic main board when given; its labels are then unknown."""
    w, h = size
    bg = tuple(int(v) for v in rng.integers(0, 256, 3))
    page = Image.new("RGB", size, bg)
    _clutter(rng, page)
    taken: list[tuple[int, int, int, int]] = []

    if board_image is not None:
        side = int(rng.uniform(0.45, 0.85) * h)
        main = Image.fromarray(board_image).resize((side, side), Image.Resampling.LANCZOS)
        labels = np.full(64, -1, np.int64)
        white_bottom, board = True, chess.Board(None)
    else:
        scene = render_scene(rng, lib, split, square_px=int(rng.integers(36, int(0.85 * h) // 8)))
        x, y, bw, bh = (int(v) for v in scene.bbox)
        main = Image.fromarray(scene.image[y : y + bh, x : x + bw])
        side = main.width
        labels, white_bottom, board = scene.labels, scene.white_bottom, scene.board
    mx, my = int(rng.integers(0, w - side)), int(rng.integers(0, h - side))
    page.paste(main, (mx, my))
    taken.append((mx, my, mx + side, my + side))

    if distractors:
        for _ in range(int(rng.integers(0, 4))):  # thumbnails, always smaller than the main board
            thumb_sq = int(rng.integers(8, max(9, min(30, side // 8 - 4))))
            t = render_scene(rng, lib, split, square_px=thumb_sq, decorate=False)
            x, y, tw, th = (int(v) for v in t.bbox)
            tx, ty = int(rng.integers(0, w - tw)), int(rng.integers(0, h - th))
            rect = (tx, ty, tx + tw, ty + th)
            if any(_overlaps(rect, r) for r in taken):
                continue
            page.paste(Image.fromarray(t.image[y : y + th, x : x + tw]), (tx, ty))
            taken.append(rect)

    img = np.asarray(page, dtype=np.uint8).copy()
    # Screen captures are neither rescaled nor JPEG'd; colour shifts, blur and noise still apply.
    s = degrade(Scene(img, (mx, my, side, side), labels, white_bottom, board), rng, False, False)
    return Screen(s.image, BoardBox(mx, my, side, side), labels, white_bottom, board)
