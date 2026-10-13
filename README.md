# ChessTrainer

A desktop chess coach that works on top of **any** chess board on your screen. It reads the
position from a screen capture, draws the engine's best move as an arrow, explains it in plain
English at your skill level, and runs an opening trainer.

![demo](docs/demo.gif) <!-- placeholder: record once the overlay lands (phase 3) -->

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Project setup, synthetic data generator, CNN training + evaluation | **done** |
| 2 | Board detection + piece recognition → FEN from a live screen | **done** |
| 3 | Stockfish integration + overlay arrows | **done** |
| 4 | Skill levels + grounded LLM explanations | |
| 5 | Opening trainer (guided, practice) | |
| 6 | Review mode, hotkeys, PyInstaller packaging | |

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/) (`pip install uv`).

```powershell
uv sync                      # creates .venv; installs CUDA PyTorch on Windows/Linux
uv run pytest                # tests
```

`pyproject.toml` pulls PyTorch from the CUDA 12.6 wheel index. On a machine without an NVIDIA
GPU it still works (the CUDA wheels fall back to CPU), it is just slower to train.

### Stockfish (needed from phase 3)

Download Stockfish for your platform from <https://stockfishchess.org/download/> (Windows: the
x86-64 build), unzip it, and put the folder in `data/engine/` (or
`%APPDATA%\ChessTrainer\engine\`), or put it on your `PATH`, or set `STOCKFISH_PATH` in `.env`
to the full path of the `.exe`. Without it the app still starts and shows these instructions.

## Running the overlay

```powershell
uv run chesstrainer                 # arrows over any chess board on screen + side panel
uv run chesstrainer --demo          # overlay visible to screen recorders (OBS, Snipping Tool)
uv run chesstrainer --record-gif 20 # record a 20 s demo GIF once a board is found
```

The best move is a bold green arrow, the two alternatives lighter blue arrows. The side panel
shows the eval bar and the three lines; "Record GIF" saves a clip to `Videos\ChessTrainer\`.

### Anthropic API key (needed from phase 4)

Create `.env` in the project root (it is git-ignored):

```
ANTHROPIC_API_KEY=sk-ant-...
```

## Training the piece classifier

```powershell
uv run python scripts/download_assets.py   # piece sets + board textures -> data/assets/
uv run python scripts/preview_synth.py     # optional: eyeball samples in data/preview/
uv run python scripts/fetch_real_eval.py   # optional: real-site eval boards -> data/real/
uv run python -m chesstrainer.vision.train # ~15 min on an RTX 3060
uv run python -m chesstrainer.vision.train --eval-only   # re-run the evaluation
```

Output: `data/models/square_classifier.pt`, plus `eval_report.txt` and `metrics.json`.

### How the data is made

`vision/synth.py` renders random **legal** positions (random games with a capture bias, so
endgames are covered) using:

- **25 piece sets** from Lichess (free licences only; abstract/novelty sets excluded),
- **board themes**: lichess/chess.com palettes, random colour pairs, and 19 Lichess board
  textures (wood, marble, ...),
- **UI clutter**: last-move and selection tints, check glow, legal-move dots/rings, hover
  outlines, coordinates inside the edge squares, arrows, page background and frames,
- **degradation**: non-integer rescaling (browser zoom/HiDPI), blur, JPEG, gamma/contrast/
  white-balance/saturation shifts, noise,
- **detection error**: squares are cut with `geometry.split_board` (the same function used at
  runtime) from a deliberately imperfect bounding box (±12% of a square per edge, plus ±4%
  per square).

Training streams fresh samples from 10 worker processes, so no image is seen twice.

### Evaluation sets

| Set | What it measures |
|---|---|
| `val` | Same styles as training, unseen positions/augmentations (300 boards). |
| `heldout` | Piece sets (merida, staunty, fantasy), textures and palettes **never used in training** (300 boards). |
| `real` | Boards rendered by lichess.org and chess.com's own servers (60 boards, 16 chess.com piece sets; none are in the training data). |

Reported per set: per-square accuracy, full-board accuracy (all 64 squares right), per-class
precision/recall, and the most common confusions.

**Your own screenshots**: put tightly cropped board images in `data/real/` and add rows to
`data/real/labels.csv` (`file,fen,white_bottom`; the FEN only needs the piece-placement field).

## Reading the board from the screen (phase 2)

```powershell
uv run python -m chesstrainer.vision.live                  # print the FEN whenever it changes
uv run python -m chesstrainer.vision.live --select         # drag a box around the board (saved)
uv run python -m chesstrainer.vision.live --clear-region   # back to automatic detection
uv run python -m chesstrainer.vision.live --orientation black   # or white / auto (saved)
uv run python -m chesstrainer.vision.live --image shot.png --debug-dir out/
```

How it works:

- **Detection** (`vision/detect.py`): edge contours propose boards; a lattice fit on edge
  projections snaps to the exact squares; a checkerboard colour test (top-left must be the light
  square) verifies. The box is cached and re-checked each frame; full detection only runs when
  the board moves, changes, or disappears.
- **Orientation** (`vision/orientation.py`): coordinate labels when the site draws them (the "8"
  has two enclosed holes, the "1" none) plus piece placement (pawn direction, kings, piece mass).
  Once confident it sticks; `--orientation` or a manual flip overrides it.
- **Position** (`vision/position.py`): per-square argmax, then legality repair (two kings of one
  colour: the more confident stays and the other takes its next most likely class; a missing king
  is restored from the most king-like square; no pawns on back ranks; at most 8 pawns). Each new
  frame is then explained by the game so far (no change, any legal move, any two moves, or an
  earlier position). That fixes low-confidence squares and gives exact side to move, castling
  rights and en passant. Positions seen without history get these from check status, the
  last-move highlight, or (last resort) "the side at the bottom moves". Changes must hold for two
  frames, so animations and drags are ignored.
- **Settings** live in `%APPDATA%\ChessTrainer\settings.json`: monitor, fps, orientation, saved
  region, and `hide_overlay_from_capture` (default **on**: the overlay is invisible to screen
  capture via `SetWindowDisplayAffinity`; turn it off to record demo GIFs).

Evaluation:

```powershell
uv run python scripts/eval_phase2.py     # real boards, real boards in cluttered pages, synthetic screens
uv run python scripts/live_selftest.py   # plays a real lichess-rendered game on screen, reads it via capture
```

## Licences

Piece sets and board textures are downloaded from
[lichess-org/lila](https://github.com/lichess-org/lila) at training time and are not
redistributed. Their licences (GPL, AGPL, MIT, Apache, CC0, CC BY, CC BY-NC-SA) are listed in
`src/chesstrainer/vision/assets.py` and in lila's `COPYING.md`. The CC BY-NC-SA sets mean the
trained weights are for **non-commercial** use. Drop those sets from `PIECE_SETS` before
retraining if that matters.

## Layout

```
src/chesstrainer/
  capture/     screen capture (mss)
  vision/      assets, synthetic generator, CNN + training, detection, orientation,
               position tracking, the live reader
  engine/      Stockfish (phase 3)
  explain/     grounded facts + Claude explanations (phase 4)
  openings/    opening trainer (phase 5)
  overlay/     drag-to-select, capture exclusion; overlay and side panel (phase 3)
  settings/    user settings (JSON); skill levels, levels.yaml (phase 4)
data/          downloaded assets, eval sets, model checkpoints (mostly git-ignored)
scripts/       asset download, previews, real-eval fetch
tests/
```
