# ChessTrainer

A desktop chess coach for Windows that works on top of **any** chess board on your screen
(lichess, chess.com, a YouTube video, a PDF...). It reads the position from a screen capture,
draws Stockfish's best move as an arrow on the board, explains the move in plain English at
your skill level, trains your openings, and reviews your games to show you what to work on.

![ChessTrainer demo](docs/demo.gif)
<!-- Demo GIF goes here: record with the panel's "Record GIF" button or
     `chesstrainer --record-gif 20`, then save it as docs/demo.gif. -->

## What it does

- **Reads any board on screen.** A small CNN classifies each square; the board is found with
  classic computer vision (edges, lattice fit, checkerboard check). Works with any site and
  piece style, flipped boards, highlighted squares and arrows drawn by the site.
- **Best-move overlay.** A click-through window draws the engine's moves right on the board,
  plus a side panel with an eval bar, the top lines and the explanation.
- **Explanations that stick to the facts.** python-chess and Stockfish compute facts about the
  move (captures, threats, forks, pins, material over the line, ...); only those facts go to
  Claude, and every answer is checked against them before it is shown.
- **Five skill levels** (Beginner < 800 to Expert 2000+): how many moves are shown, whether a
  simpler "practical" move is preferred, explanation style and the review thresholds.
- **Opening trainer.** Pick any opening from the Lichess openings dataset; guided mode shows
  and explains the book move, practice mode quizzes you with spaced repetition. Deviations are
  named and graded.
- **Game review.** Paste a PGN or load your recent Lichess / Chess.com games by username. Your
  mistakes are marked using your level's thresholds, the biggest ones are explained, and a
  summary tells you which mistake types to work on (hanging pieces, missed captures, ...).
- **System-wide hotkeys** for showing/hiding the arrows, explanations on/off, pause, flipping
  the board and switching modes, even while the browser has focus.
- **Explanations are optional.** With them off (or no API key) the app makes no API calls at
  all; the arrows, engine lines, trainer and reviews keep working.

## Architecture

```
 screen ──mss──▶ BoardTracker ──▶ PieceRecognizer ──▶ PositionTracker ──▶ Snapshot
                 (vision/detect)  (CNN: PyTorch in dev,  (legality repair,      │
                                   ONNX/OpenCV shipped)   move inference)        ▼
                                                                         TrainerApp (Qt)
   ┌──────────────────────────────────────────────────────────────────────┤
   ▼                          ▼                        ▼                  ▼
 AnalysisEngine          OpeningTrainer          ReviewWindow        GlobalHotkeys
 (Stockfish, MultiPV,    (book graph, guided /   (PGN / Lichess /    (RegisterHotKey
  stale searches          practice, SM-2 cards)   Chess.com, engine   thread)
  cancelled)                    │                 pass, mistake types)
   │                            │                        │
   ▼                            ▼                        ▼
 facts.py (python-chess) ──▶ Claude Haiku ──▶ grounding check ──▶ panel / review window
 only computed facts are sent; claims not in the facts trigger one rewrite, else the facts are shown
   │
   ▼
 BoardOverlay (click-through, DPI-exact, hidden from screen capture) + SidePanel
```

| Package | Role |
|---|---|
| `vision/` | synthetic data generator, CNN + training, ONNX export, board detection, orientation, position tracking |
| `engine/` | Stockfish discovery and the background MultiPV analysis thread |
| `explain/` | facts about a move, practical-move choice, prompt, explainer + cache, grounding check |
| `openings/` | opening book (Lichess dataset), curated repertoire, trainer session, spaced repetition |
| `review/` | PGN parsing, Lichess / Chess.com API clients, engine pass, mistake types, summary |
| `overlay/` | Qt app, overlay, side panel, dialogs, review window, global hotkeys |
| `levels.yaml` | everything that depends on the skill level, in one place |

## Results

All numbers come from the eval scripts in `scripts/` (reports in `data/eval/`).

### Phase 1: piece classifier (per-square accuracy)

| Set | Development model (25 piece sets) | Shipped model (free-licence sets only) |
|---|---|---|
| Validation (training styles, unseen positions) | 99.61% | 99.03%* |
| Held-out piece sets, textures and palettes | 99.88% | 99.59%* |
| Real lichess.org / chess.com renders (71 boards) | 97.76% | 97.78% |

\* These sets include the CC BY-NC-SA piece styles that the shipped model never saw in training.

### Phase 2: board reading from the screen

| Test | Result |
|---|---|
| My own screenshots (11) | 11/11 exact positions, orientation found automatically |
| Lichess renders (20) | 20/20 |
| Chess.com renders (40), automatic / given orientation | 27/40 / 33/40 (shipped model: 25/40 / 31/40) |
| Synthetic full screens with decoys (200) | 199/200 boards found |
| Live self-test (real game on screen) | 23/23 exact positions incl. castling and en passant, ~11 ms/frame |

### Phase 3: overlay speed and alignment (150% display scaling)

| Measure | Result |
|---|---|
| New position on screen → arrows painted | median 226 ms, max 259 ms (22/22 under 500 ms) |
| Real lichess in Chrome, key press → arrows | median 386 ms (includes the site's ~250 ms animation) |
| Arrow alignment | within 2.4 px of the square centres (≤ 3.1% of a square) |
| Board window moved → arrows back | 240-400 ms |

### Phase 4: grounded explanations (150 explanations, 30 positions × 5 levels)

| Measure | Result |
|---|---|
| Explanations with ≥ 1 claim not in the facts (LLM judge) | 7-9% (effort medium), 22-25% (effort low) |
| Unsupported claims overall | ~1% of ~930 claims |
| Positions from my real screenshots | 0/25 flagged |
| Latency / cost | median 3.0 s, ~$0.0004 per explanation (Claude Haiku) |

### Phase 5: opening trainer (scripted, real Stockfish and Claude)

| Measure | Result |
|---|---|
| Scripted scenarios (main lines, transpositions, deviations, takebacks, practice) | 15/15 exact expected events |
| Explanations through the live path | 73: 71 passed the fact check first time, 2 rewritten once, 0 fell back |
| Cost per run | ~$0.02 |

### Phase 6: game review and packaging

| Measure | Result |
|---|---|
| Real games reviewed (Lichess API: Maia bots' games; Chess.com API: Hikaru) + 3 sample PGNs | 10 games, side found by username in 7/7 fetched |
| Engine pass | ~0.14 s per position (depth 14), a 40-move game in ~10 s |
| Explanations of the biggest mistakes (two runs) | 15 new: 13 passed the fact check first time, 2 rewritten once, 0 fell back; ~$0.0006 each |
| Packaged app | 379 MB folder (105 MB of it Stockfish), no PyTorch; audit found no API key, `.env`, training images or non-commercial piece sets; headless self-check of the exe passes |
| Shipped classifier, ONNX via OpenCV vs PyTorch | 0 of 42,944 eval predictions differ |

## Setup

### Option A: the packaged app

Build it once (below) and run `dist\ChessTrainer\ChessTrainer.exe`; no Python or terminal is
needed to use it. It includes Stockfish 19 and the piece classifier. For explanations, put your
Anthropic API key in `%APPDATA%\ChessTrainer\.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
```

### Option B: from source

Requires Windows 10/11, Python 3.11+ and [uv](https://docs.astral.sh/uv/) (`pip install uv`).

```powershell
python -m uv sync                                   # .venv with CUDA PyTorch (CPU works too)
python -m uv run python scripts/download_assets.py  # piece sets + board textures (training only)
python -m uv run python -m chesstrainer.vision.train            # dev model, ~13 min on an RTX 3060
python -m uv run python -m chesstrainer.vision.train --free-only # shipped model (free-licence sets)
python -m uv run python -m chesstrainer.vision.export_onnx      # -> square_classifier_free.onnx
python -m uv run chesstrainer
```

**Stockfish**: download it from <https://stockfishchess.org/download/> and unzip it into
`data/engine/` (or `%APPDATA%\ChessTrainer\engine\`), or set `STOCKFISH_PATH` in `.env`.
Without it the app still starts and explains how to install it.

**API key** (optional): `ANTHROPIC_API_KEY=sk-ant-...` in `.env` in the project folder or in
`%APPDATA%\ChessTrainer\`. Both are git-ignored and never bundled into the build.

### Building the Windows app

```powershell
python -m uv run python scripts/build_exe.py
```

PyInstaller builds `dist\ChessTrainer\` (one folder, no console window), then the script audits
it: no PyTorch, no training images, no piece sets under a non-commercial licence, no `.env`, and
nothing that looks like an API key anywhere (including inside the compressed Python archive). It
finally runs the exe's headless `--self-check`. PyTorch is not shipped: the classifier runs as
ONNX through OpenCV (identical predictions on all eval sets), which keeps the build small.

## Using it

| | |
|---|---|
| **Analysis** | Open a game in any app. The best move appears as a green arrow; the panel shows the eval bar, lines and explanation (hover it to see the facts it was written from). |
| **Opening trainer** | Mode → Opening trainer → choose an opening and side (recommended ones for your level first), guided or practice. |
| **Game review** | Mode → Game review. Paste PGN, or enter your Lichess / Chess.com username and load recent games. Select games → Review. Step with ← / →, **N** jumps to your next mistake. Red arrow = your move, green = Stockfish's. |
| **Explanations on/off** | Panel checkbox, Ctrl+E (panel focused) or the global hotkey. Off = no API calls. |

### Hotkeys (system-wide)

| Hotkey | Action |
|---|---|
| Ctrl+Alt+O | show / hide the arrows |
| Ctrl+Alt+E | explanations on / off |
| Ctrl+Alt+P | pause / resume |
| Ctrl+Alt+F | flip the board |
| Ctrl+Alt+M | switch mode (analysis → opening trainer → game review) |

Change them in `%APPDATA%\ChessTrainer\settings.json` (`"hotkeys": {"flip": "Ctrl+Shift+F9"}`;
an empty string disables one, `"global_hotkeys": false` disables all). A combination already
taken by another program is skipped and listed in the panel's hotkey tooltip.

## Tests and evals

```powershell
python -m uv run pytest                                # unit + offscreen app tests
python -m uv run python scripts/eval_phase2.py         # board reading
python -m uv run python scripts/eval_phase5.py         # opening trainer (offscreen)
python -m uv run python scripts/eval_phase6.py         # game review on real fetched games
```

The on-screen self-tests (`live_selftest.py`, `overlay_selftest.py`, `browser_check.py`) take
over part of the screen for a minute or two.

## How the classifier is trained

`vision/synth.py` renders random legal positions with Lichess piece sets, board palettes and
textures, site clutter (move highlights, check glow, legal-move dots, coordinates, arrows) and
degradation (scaling, blur, JPEG, colour shifts), then cuts the squares with the same function
used at runtime from a deliberately imperfect box. Training streams fresh samples, so no image
is seen twice. Three piece sets, two textures and two palettes are held out to measure
generalization.

## Licences

- Piece sets and board textures are downloaded from
  [lichess-org/lila](https://github.com/lichess-org/lila) for training and are **not**
  redistributed. The development model also uses CC BY-NC-SA sets; the **shipped model is
  trained only on the free-licence sets** (GPL, AGPL, MIT, Apache, CC0, CC BY), see
  `vision/assets.py`.
- The opening book is the [Lichess chess-openings](https://github.com/lichess-org/chess-openings)
  dataset (CC0).
- Stockfish (GPLv3) is bundled with its source and licence. PyQt6 and python-chess are GPLv3,
  so a redistributed build must be GPLv3-compatible. See `packaging/THIRD_PARTY_LICENSES.txt`.

## Project layout

```
src/chesstrainer/
  capture/   screen capture (mss)
  vision/    synthetic data, CNN, training, ONNX export, detection, orientation, positions
  engine/    Stockfish discovery + background analysis
  explain/   facts, practical move, prompt, explainer + cache, grounding check
  openings/  book, repertoire, trainer, spaced repetition
  review/    games (PGN, Lichess, Chess.com), engine pass, mistake types, summary, runner
  overlay/   app, overlay, panel, dialogs, review window, board view, hotkeys
  settings/  settings.json, .env loading
  levels.py / levels.yaml   skill levels
  selfcheck.py              headless installation check
packaging/   PyInstaller spec, launcher, third-party licences
scripts/     asset download, evals, self-tests, build
tests/
```
