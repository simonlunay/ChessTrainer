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
- **Never send keystrokes or mouse input, or open windows on the user's screen, for testing
  without asking first** (this phase and every later one). That includes the on-screen
  self-tests (`live_selftest.py`, `overlay_selftest.py`, `browser_check.py`) and launching the
  app; use `QT_QPA_PLATFORM=offscreen` for checks that don't need the real screen.

## Commands
`uv` is installed via `pip --user` and is **not on PATH**: use `python -m uv ...`
(or `.venv/Scripts/python ...` directly).
```
python -m uv sync                                   # CUDA torch from the cu126 index
python -m uv run pytest                             # 101 tests (engine test skips w/o Stockfish)
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
python -m uv run python scripts/eval_phase4.py [--effort low|medium]  # -> data/eval/phase4_report.txt
```
`eval_phase4.py` makes ~150 explanation calls (Haiku, ~$0.05) + ~150 judge calls (Opus 5.5,
~$1.80) per run: ask before running it again.
The self-tests take over part of the screen for ~1-2 min; the user may be using the PC at the same
time (never send global keystrokes; `browser_check.py` posts keys to its own Chrome window only).
Stockfish: found via `Settings.engine_path`, `STOCKFISH_PATH` (env/`.env`), PATH, or `stockfish*`
exe in `data/engine/` or `%APPDATA%\ChessTrainer\engine\` (sf 19 "x86-64-universal" is in
`data/engine/`, git-ignored). Missing -> setup message (panel + stderr), app keeps running.
API key: `ANTHROPIC_API_KEY` in the project `.env` or `%APPDATA%\ChessTrainer\.env` (git-ignored,
never hardcode; `settings/env.py`). Missing -> explanations off with a message, app keeps running.
User settings: `%APPDATA%\ChessTrainer\settings.json` (`CHESSTRAINER_HOME` overrides; tests use it).

## Plan and status
1. **DONE** Setup, synthetic data generator, CNN + eval.
2. **DONE** Board detection + recognition → FEN from live capture.
3. **DONE** Stockfish (background thread, MultiPV 3, stale searches stopped) + click-through
   overlay arrows aligned to the board, side panel (eval bar, 3 lines, status, controls), GIF
   recording.
4. **DONE** Skill levels (first-launch prompt, saved, changeable; all level behaviour in `levels.yaml`):
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
- **Levels** (`levels.py` + `levels.yaml`, the only place level behaviour lives): candidates,
  practical margin, review thresholds (`Level.classify_loss`, for phase 6), openings and opening
  depth (phase 5), explanation style (sentences, style text, PV plies, numbers or words).
  `Settings.skill_level` default "beginner" (user is ~400 Elo); `level_chosen` False -> modal
  `overlay/level_dialog.py` on first launch (Beginner preselected); panel combo box changes it.
- **Practical move** (`explain/practical.py`): Beginner (60 cp) / Novice (35 cp): a candidate
  within the margin whose simplicity score is >= 2 higher replaces the top move as rank 1 (green
  arrow); the panel notes the top move is slightly better. Simplicity: +3 capture winning by
  SEE, +2 mate, +1 castling / developing a minor piece (move <= 15), -3 moved piece left en
  prise, -1 king walk with queens on. Engine always runs MultiPV 3; display is cut to the level.
- **Facts** (`explain/facts.py`, helpers + own SEE in `board_tools.py`): move/capture/castling/
  promotion/en passant, check + legal replies, mate, eval in words (+number at Intermediate+),
  top-move note for practical choices, alternatives (Intermediate+), material with piece
  breakdown, line cut at a quiet point (`_quiet_end`: pv_plies, then extended while a capture is
  pending), captures in order, best reply + what it attacks, threats stopped/created (null-move
  SEE), attacks/forks/en prise of the moved piece, discovered attacks, pins (relative only if
  the piece behind is worth more than the pinner or undefended), skewers, development, castling
  rights, centre, king centralisation, opened lines, open files, passed pawns, doubled/isolated
  pawns and bishop pair at the end of the line, game phase. Each fact = tag + plain sentence.
- **Explanations** (`explain/prompt.py`, `explainer.py`): only facts + move + level are sent
  (no FEN). System prompt bans unsupported judgements/purposes, upgraded strength words and
  meta-talk. `Settings.explain_model` "claude-haiku-5-5", `explain_effort` "medium" (see
  results), max_tokens 2000, no temperature (Haiku 5.5 rejects it). SQLite cache
  `%APPDATA%\ChessTrainer\explanations.sqlite`, key = PROMPT_VERSION + FEN (no move counters) +
  move + level + model. `ExplainWorker` thread, latest-wins, request ids. App trigger
  (`TrainerApp._maybe_explain`, called from engine updates, snapshots and a timer): position
  unchanged for `explain_delay_s` (1.5) AND depth >= `explain_min_depth` (16) or search done AND
  not pending; once per (FEN, move, level); errors are not retried until the position or level
  changes (it runs at 10 Hz). Typed API errors -> readable panel messages (no key, bad key,
  out of credits, model not found, rate limit, offline). Tooltip shows the facts.
- **Grounding check** (`explain/grounding.py`): literal checker for squares, moves, piece-on-
  square, concepts (tag map), numbers and verdict words not in the facts; the eval pairs it with
  an Opus 5.5 judge that lists claims and marks each supported/unsupported (JSON schema output).

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
- Phase 4 (`data/eval/phase4_report.txt`; 25 curated positions + the 5 distinct positions read
  from the user's screenshots, x 5 levels = 150 explanations, Stockfish depth 18). Share of
  explanations with >= 1 claim not in the facts, judge (Opus 5.5) / either check: effort medium
  7-9% / 9-11% (two runs; unsupported claims ~1% of ~930), effort low 22-25% / 23-29% (claims
  4-5%; `phase4_report_effort_low.txt`). Final full medium run per level: Beginner 7%, Novice
  10%, Intermediate 10%, Advanced 7%, Expert 3%; real screenshots 0/25. Latency median 3.0 s /
  p90 6.5 s (medium) vs 1.2 s / 2.6 s (low); ~1000 input + ~520 output tokens, ~$0.0004 per
  explanation. Manual audit of the 13 flags in that run: 6 wrong (4 from a doubled-pawn fact
  measured mid-trade, fixed afterwards; the 83 explanations re-run after the fix had 5/83 judge
  flags, `phase4_report_after_fix_partial.txt`), 7 unstated-but-harmless inferences or wording.
  Prompt v1 (before the fact/prompt fixes) was 67% at low effort. Lengths 1.7-2.4 sentences on
  average, max 3.

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
- More real in-the-wild screenshots (add to `data/real/labels.csv`: file,fen,white_bottom) would
  strengthen evals.
- Phase 4: the app's explanation flow (level dialog, panel, stable-position trigger) is tested
  offscreen with a fake Claude client, not live on screen (rule: ask first). About 1 in 12
  explanations still adds something not in the facts, mostly causal glue ("because X, Y") or
  imprecise wording, occasionally a wrong inference ("the queen checks along the e-file" when it
  is the rank). The judge is strict and is itself an LLM; its flags were audited by hand for one
  run only. Run-to-run noise is about +-5 points on the flag rate.
- The Anthropic account ran out of credits during the last eval run (2026-10-13, after ~$13 of
  eval calls, mostly the Opus judge); explanations show "out of credits" until it is topped up.
- Positional facts are basic (no outposts, king safety beyond castling, space or plans); the
  model is told not to invent them, so quiet moves get plain "what it does + evaluation" texts.
