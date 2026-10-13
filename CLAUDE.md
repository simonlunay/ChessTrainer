# ChessTrainer — project notes for Claude

Desktop app (Windows 11, Python 3.11) that reads any chess board on screen, overlays the best
move, explains it in plain English at the user's skill level, and runs an opening trainer.
Stack: mss, OpenCV, PyTorch (CUDA, RTX 3060), python-chess, Stockfish, PyQt6, Anthropic API,
uv, pytest. Package: `src/chesstrainer/` (capture, vision, engine, explain, openings, overlay,
settings); assets/eval data in `data/` (mostly git-ignored).

## Working preferences (from the user)
- Work in phases; **stop after each phase** so the user can test. Don't start the next phase
  unasked.
- **Ask before major architectural decisions** that differ from the plan below.
- Report results on **real screenshots**, not just synthetic renders.
- Clean, typed (mypy --strict), tested code; ruff (line length 100).
- The user commits themselves; don't commit unless asked.
- Concise summaries with honest numbers, including failures.

## Commands
`uv` is installed via `pip --user` and is **not on PATH**: use `python -m uv ...`
(or `.venv/Scripts/python ...` directly).
```
python -m uv sync                                   # CUDA torch from the cu126 index
python -m uv run pytest                             # 78 tests (engine test skips w/o Stockfish)
.venv/Scripts/ruff format src tests scripts && .venv/Scripts/ruff check src tests scripts
.venv/Scripts/mypy src
python -m uv run python scripts/download_assets.py  # Lichess piece sets + board textures
python -m uv run python -m chesstrainer.vision.train [--eval-only]   # ~13 min on GPU
python -m uv run python scripts/fetch_real_eval.py  # lichess/chess.com server renders -> data/real
python -m uv run python -m chesstrainer.vision.live [--select|--clear-region|--orientation X|--image F]
python -m uv run python scripts/eval_phase2.py      # -> data/eval/phase2_report.txt
python -m uv run python scripts/live_selftest.py    # shows a real game on screen, reads it via capture
python -m uv run chesstrainer [--demo|--record-gif S|--grid|--calibrate|--select|--seconds S]
python -m uv run python scripts/overlay_selftest.py # phase 3: latency + alignment -> data/eval/phase3_report.txt
python -m uv run python scripts/browser_check.py    # phase 3 on lichess in a throwaway Chrome window
```
The self-tests take over part of the screen for ~1-2 min; the user may be using the PC at the same
time (never send global keystrokes; `browser_check.py` posts keys to its own Chrome window only).
Stockfish: found via `Settings.engine_path`, `STOCKFISH_PATH` (env/`.env`), PATH, or `stockfish*`
exe in `data/engine/` or `%APPDATA%\ChessTrainer\engine\` (sf 19 "x86-64-universal" is in
`data/engine/`, git-ignored). Missing -> setup message (panel + stderr), app keeps running.
API key: `ANTHROPIC_API_KEY` in `.env` (never hardcode).
User settings: `%APPDATA%\ChessTrainer\settings.json` (`CHESSTRAINER_HOME` overrides; tests use it).

## Plan and status
1. **DONE** Setup, synthetic data generator, CNN + eval.
2. **DONE** Board detection + recognition → FEN from live capture.
3. **DONE** Stockfish (background thread, MultiPV 3, stale searches stopped) + click-through
   overlay arrows aligned to the board, side panel (eval bar, 3 lines, status, controls; the
   explanation/level areas are phase-4 placeholders), GIF recording.
4. Skill levels (first-launch prompt, saved, changeable; all level behaviour in `levels.yaml`):
   Beginner <800, Novice 800–1200, Intermediate 1200–1600, Advanced 1600–2000, Expert 2000+.
   Candidates shown 1/2/3+; practical-move preference at Beginner/Novice (simpler move within a
   small eval margin, noting the engine's top move is slightly better); level-scaled review
   thresholds; opening recommendations. Grounded explanations: compute facts with python-chess
   (eval before/after, PV, captures, checks, hanging/attacked/undefended pieces, forks, pins,
   skewers, discovered attacks), send ONLY those + move + level to Claude, instruct it to use only
   the facts. Style per level (Beginner 1–2 simple sentences defining terms; Novice 2; Intermediate
   2–3 with positional ideas; Advanced/Expert 3 with standard terminology + key line). Cache by
   FEN+move+level.
5. Opening trainer: Lichess chess-openings dataset (CC0, ECO/name/moves); searchable picker with
   side choice; level-based recommendations (Italian, London for beginners) but any opening
   allowed; line depth scales with level; guided mode (next book move arrow + explanation,
   identify variation on replies), deviation handling (flag, say if it's a mistake, show engine
   reply); practice mode (no arrows, check moves, spaced repetition in SQLite).
6. Review mode (paste PGN, step through, best moves + explanations for the user's mistakes,
   filtered by level thresholds), global hotkeys (toggle overlay, switch modes, flip
   orientation), PyInstaller packaging.

## Key design decisions
- **Classifier** (`vision/model.py`, `train.py`): 13 classes (empty, PNBRQK, pnbrqk; fixed index
  order in `vision/labels.py`), 64x64 RGB crops, VGG-style CNN ~585k params, normalization inside
  the model. Checkpoint `data/models/square_classifier.pt` (git-ignored; retrain to recreate).
- **Synthetic data** (`vision/synth.py`): random legal positions (random games, capture bias),
  25 free-licensed Lichess piece sets (novelty sets excluded), palettes + random colours + 19
  Lichess textures, UI clutter (last-move tints, check glow, dots, coords, arrows), degradation;
  crops cut with the same `geometry.split_board` used at runtime, with simulated detection error.
  Streamed on the fly (IterableDataset, 10 workers; more workers OOM'd on 16 GB RAM).
  Held-out styles (merida, staunty, fantasy; 2 textures; 2 palettes) for generalization eval.
- **Detection** (`vision/detect.py`): Canny contours (pass 1 per RGB channel with 5x5 blur, pass 2
  grayscale 9x9 blur for wood grain) → cell lattices / outlines → lattice fit on edge projections
  → checkerboard colour verification (top-left must be the light square). `BoardTracker` caches
  the box and re-verifies cheaply each frame.
- **Orientation** (`vision/orientation.py`): coordinate-label glyph hole counts ("8" two holes,
  "1" none; "6" vs "3") added to a piece-placement score; hysteresis; manual override.
- **Position** (`vision/position.py`): `repair()` — two kings of one colour: keep the more
  confident, reclassify the other to its next most likely class (user's rule); missing king
  restored; no back-rank pawns; ≤8 pawns. `PositionTracker` explains each frame by no-change,
  1-ply, 2-ply, or an earlier position (log-likelihood margin 3.0) → exact turn/castling/ep; fresh
  positions infer turn from check → last-move highlight → assume bottom side (flagged). Changes
  must hold 2 frames.
- **Capture exclusion**: `SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)`, setting
  `hide_overlay_from_capture` default on (off for recording demo GIFs). Verified working.
- Region select fallback (`overlay/region_select.py`) maps on the captured image, so it is
  DPI-independent.
- **Overlay app** (`overlay/app.py` `TrainerApp`): `VisionThread` (capture + `BoardReader`)
  emits a `Snapshot` per frame -> overlay placed on the box -> new committed position sent to
  `engine/analysis.AnalysisEngine` (one Stockfish process + worker thread; `analyze()` stops the
  running search; updates carry a request id, stale ids are dropped; first update at depth >= 8,
  then throttled to 10 Hz; stops after `engine_max_seconds`). Arrows are shown only when the
  analysis is for the board on screen AND the frame is not `pending` (move in progress,
  `PositionTracker.pending`) AND the box did not move since the last frame (window dragging).
- **Coordinates** (`overlay/screen_map.py`): Qt 6 scales each screen about its native top-left:
  logical = origin + physical / dpr. The overlay window covers just the board; its top-left is
  snapped to a whole *physical* pixel (even logical px at 150%) and the painter is translated by
  the sub-pixel remainder; everything is drawn in board units (0..8).
- **Speed**: while a board is tracked only its neighbourhood is grabbed (~12 ms vs ~45 ms for
  the full 2560x1600 monitor); a board lost in a region frame triggers a full grab. A dragged
  window is re-found by template matching a 48 px board thumbnail on a downscaled frame +
  `refine(known_size=True)` (~25 ms) before full detection. While `pending`, the next frame is
  grabbed after 35 ms instead of 100. `gpu_keep_warm` (default on) classifies every frame because
  an idle laptop GPU took ~110 ms for the first inference after a move (vs ~20 ms warm).
- **Demo GIFs**: panel "Record GIF" button or `--record-gif SECONDS`; turns capture hiding off
  for the overlay/panel while recording, writes `%USERPROFILE%\Videos\ChessTrainer\*.gif`
  (board + panel if adjacent). `--demo` turns capture hiding off for the whole session.

## Results so far
- Phase 1 classifier: per-square 99.61% val, 99.88% held-out styles, 98.1% real renders.
- Phase 2 (`data/eval/phase2_report.txt`): user's 11 screenshots 11/11 exact FEN with auto
  orientation (also when pasted into cluttered pages); Lichess renders 20/20; chess.com renders
  27/40 auto-orient, 33/40 with orientation given; synthetic full screens 199/200 detected.
  Live self-test: 23/23 exact FENs (incl. castling, en passant), ~11 ms/frame median.
- Phase 3 (`data/eval/phase3_report.txt`, `phase3_browser.txt`; 150% display, SF19 4 threads):
  self-test on lichess renders: 23/23 positions, new position -> arrows median 226 ms / max
  259 ms (22/22 under 500 ms), stale arrows hidden median 105 ms; dots on square centres within
  2.4 px (<=3.1% of a square) incl. after moves/resizes, drawing vs detected grid <=0.22 px (the
  rest is detection); window moved -> arrows back 240-400 ms. Real lichess in Chrome: 22/22 moves
  got arrows, key press -> arrows median 386 ms (includes lichess's ~250 ms animation; 17/22 under
  500 ms), alignment within 2 px vs board found by square colours.

## Known limitations
- Unusual chess.com piece sets ("8-bit", "light") misclassify; fix = more piece styles in training.
- Orientation is ambiguous on boards without coordinates in random/endgame positions (real games
  lock it in from the opening; manual flip/`--orientation` available).
- Side to move for positions seen without history may be assumed (flagged `turn_assumed`).
- Very low-contrast pastel themes can still be missed in cluttered screens (1/40); drag-select works.
- Some piece sets are CC BY-NC-SA → trained weights are non-commercial unless those are dropped.
- Phase 3 browser check: one resize measurement found no calibration dots and one took 3.8 s
  to re-find the board, both while other windows were covering/uncovering the test window;
  unconfirmed whether that is an app problem. Move -> arrows from a key press exceeds 500 ms on
  ~1 in 4 lichess moves because of the site's move animation (the app waits for 2 equal frames).
- The `--record-gif` / Record button path is unit-tested (GIF writing) but was not exercised live.
  Live self-test: 23/23 exact FENs (incl. castling, en passant), ~11 ms/frame median.
- Phase 3 (`data/eval/phase3_report.txt`, `phase3_browser.txt`; 150% display, SF19 4 threads):
  self-test on lichess renders: 23/23 positions, new position -> arrows median 226 ms / max
  259 ms (22/22 under 500 ms), stale arrows hidden median 105 ms; dots on square centres within
  2.4 px (<=3.1% of a square) incl. after moves/resizes, drawing vs detected grid <=0.22 px (the
  rest is detection); window moved -> arrows back 240-400 ms. Real lichess in Chrome: 22/22 moves
  got arrows, key press -> arrows median 386 ms (includes lichess's ~250 ms animation; 17/22 under
  500 ms), alignment within 2 px vs board found by square colours.
 + 11 user screenshots; more real in-the-wild
  screenshots (add to `data/real/labels.csv`: file,fen,white_bottom) would strengthen evals.
