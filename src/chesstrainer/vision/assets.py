"""Download and prepare open-licensed piece sets and board textures for the data generator.

Piece sets come from lichess-org/lila (``public/piece/<set>/{w,b}{K,Q,R,B,N,P}.svg``). SVGs are
rasterized once with Qt's SVG renderer (already a dependency for the overlay) to RGBA PNGs, so the
generator itself only needs Pillow. Licenses for each set are listed in lila's COPYING.md; sets
marked non-free there are not used. Assets are only used locally to train the classifier and are
not redistributed with the app.
"""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from chesstrainer.vision.labels import NUM_CLASSES, piece_asset_name

log = logging.getLogger(__name__)

LILA_RAW = "https://raw.githubusercontent.com/lichess-org/lila/master/public"

ASSET_DIR = Path(__file__).resolve().parents[3] / "data" / "assets"
PIECE_DIR = ASSET_DIR / "pieces"
BOARD_DIR = ASSET_DIR / "boards"

#: Rasterized piece side length. Pieces are downscaled from this at render time.
PIECE_RASTER_SIZE = 256


@dataclass(frozen=True)
class PieceSet:
    name: str
    license: str

    @property
    def non_commercial(self) -> bool:
        return "-NC" in self.license


# Free-licensed, realistic-looking sets. Abstract or novelty sets (letter, shapes, mono,
# disguised, kiwen-suwi, anarcandy, horsey, firi) are excluded: no real site uses them.
PIECE_SETS: tuple[PieceSet, ...] = (
    PieceSet("cburnett", "GPLv2+"),
    PieceSet("merida", "GPLv2+"),
    PieceSet("chessnut", "Apache-2.0"),
    PieceSet("pirouetti", "AGPLv3+"),
    PieceSet("pixel", "AGPLv3+"),
    PieceSet("rhosgfx", "CC0-1.0"),
    PieceSet("fantasy", "MIT"),
    PieceSet("spatial", "MIT"),
    PieceSet("celtic", "MIT"),
    PieceSet("mpchess", "GPLv3+"),
    PieceSet("totoy", "CC-BY-4.0"),
    PieceSet("papercut", "CC-BY-4.0"),
    PieceSet("california", "CC-BY-NC-SA-4.0"),
    PieceSet("caliente", "CC-BY-NC-SA-4.0"),
    PieceSet("maestro", "CC-BY-NC-SA-4.0"),
    PieceSet("fresca", "CC-BY-NC-SA-4.0"),
    PieceSet("cardinal", "CC-BY-NC-SA-4.0"),
    PieceSet("icpieces", "CC-BY-NC-SA-4.0"),
    PieceSet("gioco", "CC-BY-NC-SA-4.0"),
    PieceSet("tatiana", "CC-BY-NC-SA-4.0"),
    PieceSet("staunty", "CC-BY-NC-SA-4.0"),
    PieceSet("dubrovny", "CC-BY-NC-SA-4.0"),
    PieceSet("cooke", "CC-BY-NC-SA-4.0"),
    PieceSet("monarchy", "CC-BY-NC-SA-4.0"),
    PieceSet("minimal-warmth", "CC-BY-NC-SA-4.0"),
)

#: Sets whose licence allows redistribution in the packaged app (the shipped classifier is
#: trained on these only, see ``train --free-only``).
FREE_PIECE_SETS: frozenset[str] = frozenset(s.name for s in PIECE_SETS if not s.non_commercial)

#: Full-board texture images from lila public/images/board (AGPLv3+).
BOARD_TEXTURES: tuple[str, ...] = (
    "blue-marble.jpg",
    "blue2.jpg",
    "blue3.jpg",
    "canvas2.jpg",
    "green-plastic.png",
    "grey.jpg",
    "horsey.jpg",
    "leather.jpg",
    "maple.jpg",
    "maple2.jpg",
    "marble.jpg",
    "metal.jpg",
    "olive.jpg",
    "pink-pyramid.png",
    "purple-diag.png",
    "wood.jpg",
    "wood2.jpg",
    "wood3.jpg",
    "wood4.jpg",
)

#: Styles never seen in training, used for the held-out generalization test split.
HELD_OUT_PIECE_SETS: frozenset[str] = frozenset({"merida", "staunty", "fantasy"})
HELD_OUT_TEXTURES: frozenset[str] = frozenset({"wood4.jpg", "marble.jpg"})

PIECE_FILES: tuple[str, ...] = tuple(piece_asset_name(i) for i in range(1, NUM_CLASSES))


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "chesstrainer-asset-fetch"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        dest.write_bytes(resp.read())


def _rasterize_svg(svg_path: Path, png_path: Path, size: int = PIECE_RASTER_SIZE) -> None:
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QGuiApplication, QImage, QPainter
    from PyQt6.QtSvg import QSvgRenderer

    if QGuiApplication.instance() is None:
        _rasterize_svg.app = QGuiApplication([])  # type: ignore[attr-defined]
    renderer = QSvgRenderer(str(svg_path))
    if not renderer.isValid():
        raise ValueError(f"invalid SVG: {svg_path}")
    image = QImage(size, size, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    renderer.render(painter)
    painter.end()
    if not image.save(str(png_path)):
        raise OSError(f"failed to write {png_path}")


def _fetch_piece(set_name: str, stem: str, cache: Path, png: Path, force: bool) -> None:
    """Download one piece image (SVG preferred, raster fallback) and write a square RGBA PNG."""
    from PIL import Image

    for ext in ("svg", "webp", "png"):
        src = cache / f"{stem}.{ext}"
        if not src.exists() or force:
            try:
                _download(f"{LILA_RAW}/piece/{set_name}/{stem}.{ext}", src)
            except urllib.error.HTTPError as err:
                if err.code == 404:
                    continue
                raise
        if ext == "svg":
            _rasterize_svg(src, png)
        else:
            img = Image.open(src).convert("RGBA")
            img.thumbnail((PIECE_RASTER_SIZE, PIECE_RASTER_SIZE), Image.Resampling.LANCZOS)
            square = Image.new("RGBA", (PIECE_RASTER_SIZE, PIECE_RASTER_SIZE), (0, 0, 0, 0))
            square.paste(
                img, ((PIECE_RASTER_SIZE - img.width) // 2, (PIECE_RASTER_SIZE - img.height) // 2)
            )
            square.save(png)
        return
    raise FileNotFoundError(f"no image for {set_name}/{stem}")


def prepare_assets(force: bool = False) -> None:
    """Download every piece set and board texture, rasterizing SVG pieces to PNG."""
    svg_cache = ASSET_DIR / "_svg"
    for piece_set in PIECE_SETS:
        out_dir = PIECE_DIR / piece_set.name
        for stem in PIECE_FILES:
            png = out_dir / f"{stem}.png"
            if png.exists() and not force:
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            _fetch_piece(piece_set.name, stem, svg_cache / piece_set.name, png, force)
        log.info("piece set ready: %s", piece_set.name)
    for name in BOARD_TEXTURES:
        dest = BOARD_DIR / name
        if dest.exists() and not force:
            continue
        _download(f"{LILA_RAW}/images/board/{name}", dest)
        log.info("board texture ready: %s", name)


def available_piece_sets() -> list[str]:
    """Names of piece sets whose 12 PNGs are present on disk."""
    return [
        s.name
        for s in PIECE_SETS
        if all((PIECE_DIR / s.name / f"{stem}.png").exists() for stem in PIECE_FILES)
    ]


def available_textures() -> list[str]:
    return [name for name in BOARD_TEXTURES if (BOARD_DIR / name).exists()]
