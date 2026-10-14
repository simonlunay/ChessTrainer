# Third-party notices

ChessTrainer is licensed under the GNU General Public License v3.0 (see `LICENSE`). It builds on
the components below. Each keeps its own licence; this file lists them, their authors and where
to get their source.

What ships where:

| Component | Git repository | Windows release zip | Weights release asset |
|---|---|---|---|
| ChessTrainer source | yes | compiled | |
| Stockfish 19 | no | yes, with full source | |
| python-chess, PyQt6/Qt 6, other Python libraries | no (installed by `uv sync`) | yes | |
| Lichess chess-openings dataset | yes (`src/chesstrainer/openings/data/`) | yes | |
| Piece classifier weights (`square_classifier_free.onnx`) | no | yes | yes |
| Piece set and board texture images | no (downloaded for training only) | no | no |

Source code of ChessTrainer for every release: <https://github.com/simonlunay/ChessTrainer>
(the release's git tag).

## Stockfish

- Version: Stockfish 19 (`stockfish-windows-x86-64-universal.exe`)
- Authors: the Stockfish developers (see `AUTHORS` in the bundled folder)
- Licence: GNU GPL v3.0
- Source: <https://github.com/official-stockfish/Stockfish>. The release zip contains the
  complete source, `Copying.txt` (the licence) and `AUTHORS` in `_internal/engine/stockfish/`.

Stockfish runs as a separate process and talks to ChessTrainer over the UCI protocol.

## python-chess

- Version: 1.11.x (package `chess`)
- Author: Niklas Fiekas and contributors
- Licence: GNU GPL v3.0 or later
- Source: <https://github.com/niklasf/python-chess>

## PyQt6 and Qt 6

- PyQt6 6.11 by Riverbank Computing Limited, licensed under the GNU GPL v3.0.
  Source: <https://pypi.org/project/PyQt6/#files> and <https://www.riverbankcomputing.com/software/pyqt/>
- Qt 6.11 by The Qt Company Ltd. and contributors, used under the GNU LGPL v3.0.
  Source: <https://download.qt.io/official_releases/qt/> and <https://code.qt.io/>

Because PyQt6, python-chess and Stockfish are GPL v3, ChessTrainer is distributed under the
GPL v3 as well.

## Lichess chess-openings dataset

- Files: `src/chesstrainer/openings/data/a.tsv` to `e.tsv` (unchanged)
- Authors: the Lichess contributors
- Licence: CC0 1.0 Universal (public domain dedication)
- Source: <https://github.com/lichess-org/chess-openings>, commit
  `a6189a30dc273ccb21fc2536a9a2fefd5592a67a`

## Piece classifier weights

`square_classifier_free.onnx` (and `square_classifier_free.pt`, from which it is exported) is a
small CNN trained only on synthetic board images. The images were rendered on the fly from the
piece sets and board textures below, fetched from the Lichess source repository
([lichess-org/lila](https://github.com/lichess-org/lila), licences per its `COPYING.md`).
Changes made: the SVG pieces were rasterized, resized, composited on boards, and degraded
(blur, JPEG, colour shifts). No piece or board image is included in this repository, the
release zip or the weights asset.

Piece sets used to train the shipped model (all free licences):

| Piece set | Author | Licence |
|---|---|---|
| cburnett | [Colin M.L. Burnett](https://en.wikipedia.org/wiki/User:Cburnett) | GPL v2 or later |
| chessnut | [Alexis Luengas](https://github.com/LexLuengas/chessnut-pieces) | Apache 2.0 |
| pirouetti | [pirouetti](https://lichess.org/@/pirouetti) | AGPL v3 or later |
| pixel | therealqtpi | AGPL v3 or later |
| rhosgfx | [RhosGFX](https://rhosgfx.itch.io/) | CC0 1.0 |
| spatial | [Maurizio Monge](https://github.com/maurimo/chess-art) | MIT |
| celtic | [Maurizio Monge](https://github.com/maurimo/chess-art) | MIT |
| mpchess | [Maxime Chupin](https://github.com/chupinmaxime) | GPL v3 or later |
| totoy | Kosal Sen | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| papercut | [Nikolay Anzarov](https://nikoichu.itch.io/) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

Board textures used in training: `blue-marble.jpg`, `blue2.jpg`, `blue3.jpg`, `canvas2.jpg`,
`green-plastic.png`, `grey.jpg`, `horsey.jpg`, `leather.jpg`, `maple.jpg`, `maple2.jpg`,
`metal.jpg`, `olive.jpg`, `pink-pyramid.png`, `purple-diag.png`, `wood.jpg`, `wood2.jpg`,
`wood3.jpg` from `lila/public/images/board`, by the lila authors and
[pirouetti](https://lichess.org/@/pirouetti), AGPL v3 or later.

Evaluation only (never used to train the shipped weights): the held-out piece sets merida
(Armando Hernandez Marroquin, GPL v2+), fantasy (Maurizio Monge, MIT) and staunty
(sadsnake1, CC BY-NC-SA 4.0), and the textures `wood4.jpg` and `marble.jpg` (AGPL v3+).

The development model (`square_classifier.pt`, not distributed) is also trained on CC BY-NC-SA 4.0
piece sets (california, caliente, maestro, fresca, cardinal, icpieces, gioco, tatiana, staunty,
dubrovny, cooke, monarchy, minimal-warmth). Those weights are not published anywhere.

### MIT licence (spatial, celtic, fantasy piece sets)

```
Copyright (c) Maurizio Monge

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

### Apache License 2.0 (chessnut piece set)

Chessnut pieces copyright Alexis Luengas, licensed under the Apache License, Version 2.0:
<https://www.apache.org/licenses/LICENSE-2.0>.

## Other Python libraries in the Windows build

Bundled by PyInstaller; each under its own licence.

| Library | Licence |
|---|---|
| OpenCV (`opencv-python`) | Apache 2.0 |
| NumPy | BSD 3-Clause (and bundled components: 0BSD, MIT, Zlib, CC0) |
| mss | MIT |
| Pillow | MIT-CMU (HPND) |
| PyYAML | MIT |
| python-dotenv | BSD 3-Clause |
| anthropic (Claude API SDK) and its dependencies (httpx, httpcore, pydantic, anyio, ...) | MIT / BSD 3-Clause |
| keyring and its dependencies (jaraco.*, pywin32-ctypes) | MIT / BSD 3-Clause |
| PyInstaller bootloader | GPL v2 or later with the PyInstaller bootloader exception |

## Development tools (not distributed)

PyTorch and torchvision (BSD 3-Clause, used to train the classifier), ONNX (Apache 2.0), pytest
(MIT), mypy (MIT), ruff (MIT), uv (MIT / Apache 2.0).

## Services

Explanations call the Anthropic API with the user's own API key. Game review fetches public
games from the Lichess API (<https://lichess.org/api>) and the Chess.com Published-Data API
(<https://www.chess.com/news/view/published-data-api>). ChessTrainer is not affiliated with
Anthropic, Lichess or Chess.com.
