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
- **API budget: keep total API spend under $5 per phase.** Don't ask about it; just stay under
  it. Use cheap models (Haiku) or the rule-based checker for evals, **never Opus**. If something
  would push past $5, cut it down or skip it and say so in the report.
- **Report total API spend at the end of each phase.**
- **Never send keystrokes or mouse input, or open windows on the user's screen, for testing
  without asking first** (this phase and every later one). That includes the on-screen
  self-tests (`live_selftest.py`, `overlay_selftest.py`, `browser_check.py`) and launching the
  app; use `QT_QPA_PLATFORM=offscreen` for checks that don't need the real screen.

## Commands
`uv` is installed via `pip --user` and is **not on PATH**: use `python -m uv ...`
(or `.venv/Scripts/python ...` directly).
```
python -m uv sync                                   # CUDA torch from the cu126 index
python -m uv run pytest                             # 151 tests (engine tests skip w/o Stockfish)
.venv/Scripts/ruff format src tests scripts && .venv/Scripts/ruff check src tests scripts
.venv/Scripts/mypy src
python -m uv run python scripts/download_assets.py  # Lichess piece sets + board textures
python -m uv run python -m chesstrainer.vision.train [--eval-only] [--free-only] [--steps N]
python -m uv run python -m chesstrainer.vision.export_onnx   # free .pt -> .onnx + .json, checks vs torch
python -m uv run python scripts/fetch_real_eval.py  # lichess/chess.com server renders -> data/real
python -m uv run python -m chesstrainer.vision.live [--select|--clear-region|--orientation X|--image F]
python -m uv run python scripts/eval_phase2.py      # -> data/eval/phase2_report.txt
python -m uv run python scripts/live_selftest.py    # shows a real game on screen, reads it via capture
python -m uv run chesstrainer [--demo|--record-gif S|--grid|--calibrate|--select|--seconds S]
python -m uv run python scripts/overlay_selftest.py # phase 3: latency + alignment -> data/eval/phase3_report.txt
python -m uv run python scripts/browser_check.py    # phase 3 on lichess in a throwaway Chrome window
python -m uv run python scripts/eval_phase4.py [--effort low|medium]  # -> data/eval/phase4_report.txt
python -m uv run python scripts/eval_phase5.py [--no-explain] [--budget 1.0]  # -> data/eval/phase5_report.txt
python -m uv run python scripts/eval_phase6.py [--no-explain] [--offline]  # review on fetched games
python -m uv run python scripts/build_exe.py [--skip-build]  # PyInstaller -> dist/ChessTrainer + audit
dist/ChessTrainer/ChessTrainer.exe --self-check report.json  # headless check (no window/capture)
```
`eval_phase6.py` fetches public games (Lichess maia1/maia5, Chess.com hikaru) + 3 sample PGNs,
real Stockfish, explains the 3 biggest mistakes per game with Haiku (~$0.005/run).
`build_exe.py` takes ~90 s and never needs the screen (the self-check is headless, with an empty
CHESSTRAINER_HOME and ANTHROPIC_API_KEY removed from its env).
`eval_phase4.py` defaults to an Opus judge (~$1.80/run): under the budget rule never run it with
Opus again; use `--judge-model claude-haiku-5-5` or rely on the rule checker.
`eval_phase5.py` drives the real app offscreen (no screen use) with real Stockfish through 15
scripted opening lines incl. deviations/takebacks; with explanations ~73 Haiku calls, ~$0.02.
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
   Follow-ups (done in phase 5): explanations on/off (panel checkbox, Ctrl+E, saved; off = zero
   API calls); rule-based grounding check on every live explanation (regenerate once, else show
   the computed facts), outcomes logged.
5. **DONE** Opening trainer: Lichess chess-openings dataset (CC0, ECO/name/moves); searchable
   picker with side choice; level-based recommendations (2-3 per side at Beginner) but any opening
   allowed; line depth scales with level; guided mode (next book move arrow + grounded
   explanation with the opening's ideas, variation named on replies), deviation handling (flag,
   say if it's a mistake, show engine reply, switch to normal analysis); practice mode (no
   arrows, check moves, spaced repetition in SQLite). Works with explanations off.
6. **DONE** Review mode: paste PGN or load recent Lichess / Chess.com games by username (public
   APIs), engine pass, the user's mistakes marked by the level's `review` thresholds, mistake
   types + "what to work on" summary, grounded explanations of the biggest mistakes (respecting
   the toggle). System-wide hotkeys (arrows, explanations, pause, flip, mode). PyInstaller
   onedir build with the ONNX free-licence classifier, bundled Stockfish, build audit. README
   rewritten for the public repo.

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
- **Live grounding check** (`explainer.py`, `fallback.py`): every new explanation is checked with
  `check_grounding`; if flagged, one retry with the flagged claims listed in the prompt; if that
  is flagged too, `facts_summary` (move fact + top facts by tag priority, level length) is shown.
  Outcome (passed/regenerated/fallback) cached with the text and logged per explanation to
  `grounding_log` in `explanations.sqlite`; the app logs session counts; panel tooltip says which.
  Cache key gains `MoveFacts.context` (opening explanations are cached separately).
- **Explanations toggle**: `Settings.explanations` (saved), panel checkbox + Ctrl+E (application
  shortcut, works while the panel has focus; global hotkeys are phase 6). Off: the `Explainer`
  is not even created, `_maybe_explain` returns, late results are dropped.
- **Opening book** (`openings/book.py`): `openings/data/{a..e}.tsv` vendored (CC0, commit noted in
  `SOURCE.txt`), replayed into a position graph keyed by FEN without counters (transpositions);
  ~1.8 s to load, so the app loads it on a thread. `name_matches`: full variation names match
  themselves + sub-variations; plain names match any name part that starts or ends with them
  ("Queen's Pawn Game: London System" is a London; "Semi-Slav" is not a Slav). Accents folded.
- **Repertoire** (`openings/repertoire.yaml`, the only place curated opening knowledge lives):
  per opening + side: `ideas` (sentences given to Claude as facts), `notes` (per SAN), `lines`
  (curated main lines). Curated for Italian (W/B), London, Scotch, Queen's Gambit, Scandinavian,
  QGD, Caro-Kann; ideas only for the intermediate+ recommendations. Recommendations live in
  `levels.yaml` (Beginner W: Italian, London, Scotch; B: Italian, Scandinavian, QGD).
- **Trainer** (`openings/trainer.py`, Qt-free): `OpeningTree` = dataset lines matching the name +
  curated lines; trainer's move = curated, else the book move with most lines (gambit lines
  weigh 0.3 unless the opening is a gambit). `TrainerSession.observe(board)` matches placements
  only: same / earlier (takeback) / 1-2 legal moves / start position (new game) / a book position
  (resync mid-line) / else "lost". Events: user_correct, user_other (other book move of this
  opening), user_other_opening, user_left (graded), opp_book (variation named), opp_other_opening,
  opp_left (graded), complete (depth = `opening_depth_plies` or end of book). Out of book ->
  state `left` -> the app runs normal analysis.
- **Trainer in the app**: the session's board replaces the reading's (exact turn/castling/ep).
  In book on the player's turn Stockfish searches only the book move (`AnalysisEngine.analyze(
  board, root_moves)`), giving the eval bar and the PV for `openings/explain.opening_facts`
  (phase-4 facts + opening/variation names + note + ideas, extra prompt instructions); arrow =
  book move (guided). Opponent's turn in book: normal MultiPV search, no arrows, no explanation.
  Grading: eval before (previous search, mover's view) minus after (new position's best, at depth
  14 or final) -> `Level.classify_loss` -> "Stockfish: X is a mistake (about N pawns worse than
  the best/book move)". Practice: no arrow, no lines, no explanation of the answer; misses are
  revealed after a takeback; `PracticeStore` (`openings.sqlite`): SM-2-lite cards per (opening,
  side, position), reviews logged; `practice_reply` suggests the opponent book move leading to
  the most due/new positions. Settings: `app_mode`, `trainer_opening/side/mode`.
- **Grounding check** (`explain/grounding.py`): literal checker for squares, moves, piece-on-
  square, concepts (tag map), numbers and verdict words not in the facts; the eval pairs it with
  an Opus 5.5 judge that lists claims and marks each supported/unsupported (JSON schema output).

- **Review** (`review/`, Qt-free; `overlay/review_window.py`): `games.py` parses PGN (variants
  skipped) and fetches Lichess `api/games/user/{name}` (PGN) / Chess.com monthly archives (JSON,
  newest first, <= 6 months); usernames validated by regex; readable `FetchError`s. Player's
  side = username match (pasted PGN: auto/White/Black combo; unknown -> White, marked
  "assumed"). `analysis.py`: separate `StockfishEvaluator` (depth `review_depth` 14, cap
  `review_seconds` 0.5) searches every position once; loss = mover's eval of the best move minus
  the eval after the played move, both clamped to +-1000 cp; played == best -> 0;
  `Level.classify_loss` grades the player's moves only. `mistakes.py`: first match of allowed
  mate, missed mate, opponent's best reply wins material by SEE (moved piece / other piece =
  hung_piece; piece already attacked before the move = ignored_threat), missed winning capture,
  reply is a fork/skewer/discovered attack, reply line wins material (lost_material), reply
  pins, best move fork/skewer/discovered, best line wins material (missed_win), best pins, else
  positional; lines judged at the Expert horizon (8 plies) for every level. Labels + tips in
  `KINDS`. `report.py`: summary (counts by severity and type, average loss, top 4 types with
  tips) and `review_facts` = facts of the best move + "In the game you played X (severity, N
  pawns worse)" + mistake type + facts about the played move prefixed "About X (the move
  played)"; context `review:<uci>` (own cache entries); extra instructions. `runner.py`:
  background thread, all games first, then up to `review_explain_max` (3) biggest per game;
  checks `explanations_on()` before each call; stops at the first API error.
- **Review window**: Paste / Lichess / Chess.com tabs (usernames + count saved), multi-select
  list, progress + Stop; board (`board_view.py`: Unicode glyphs drawn as a light backing + solid
  or outline glyph, no image assets), red arrow = played, green = best; move list with ?!/?/??
  colours; Left/Right, N = next mistake; detail pane (severity, loss, best move, type + detail,
  explanation; facts in the tooltip); summary pane. Opening it (mode combo "Game review" or the
  mode hotkey) pauses live reading; closing restores the previous pause state and mode. Review
  is not persisted as `app_mode`.
- **Hotkeys** (`overlay/hotkeys.py`): `RegisterHotKey` on a dedicated thread with its own message
  loop (MOD_NOREPEAT), WM_HOTKEY -> Qt signal. Defaults Ctrl+Alt+O (arrows), E (explanations),
  P (pause), F (flip), M (mode: analysis -> trainer -> review). `Settings.hotkeys` overrides
  ("" disables one), `global_hotkeys` False = none. Taken combos are skipped and listed in the
  panel tooltip. Hiding arrows keeps the analysis running (`overlay_hidden`).
- **Packaging**: PyTorch with CUDA is 4 GB, so (user's decision) the shipped app runs the
  classifier as ONNX via `cv2.dnn` (`vision/onnx_backend.py`; `PieceRecognizer(backend="auto")`
  = torch when installed and the .pt exists, else ONNX; `Settings.classifier_backend`). Shipped
  model = `square_classifier_free.pt` (`train --free-only --steps 24000`, no CC BY-NC-SA sets),
  exported to `square_classifier_free.onnx` + `.json` sidecar (class order, source). `paths.py`:
  `FROZEN`, `bundle_dir()` (`_MEIPASS` when frozen, else `data/`); the frozen app reads only
  `%APPDATA%\ChessTrainer\.env`; engine search uses `bundle_dir()/engine`. Spec
  `packaging/chesstrainer.spec`: onedir, windowed, icon, excludes torch and training modules,
  bundles YAML/TSV data, the ONNX model, the Stockfish folder (with source + licence) and
  `THIRD_PARTY_LICENSES.txt`. Frozen app logs to `%APPDATA%\ChessTrainer\chesstrainer.log`.
  `--self-check` (`selfcheck.py`) checks classifier/engine/data/key presence headless.

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

- Phase 5 (`data/eval/phase5_report.txt`, `phase5_steps.jsonl`; offscreen, SF19, 2 s/position,
  no screen use): 15/15 scripted scenarios produced exactly the expected trainer events: main
  lines to depth (Italian W/B, London via transposition, Scandinavian, QGD), opponent leaving
  book (Qh4 graded "blunder", h6/Qf3 "about 1 pawn worse, not a real mistake at your level"),
  opponent choosing another opening (Sicilian, Ruy Lopez, London), user's other book move +
  takeback, user leaving book (Qxd4 graded "blunder vs the book move"), a dataset-only Najdorf at
  Expert, practice mode with a miss. Explanations through the live path: 73 (52 book moves with
  opening facts); grounding check 71 passed first time, 2 regenerated, 0 fell back to facts
  (earlier run: 72/1/0); latency median 2.4 s, max 4.5 s; ~$0.02 per run.
  Phase 5 API spend in total: ~$0.04 (two explanation runs; all tests use fake clients).

- Phase 6 (`data/eval/phase6_report.txt`, `phase6_build.txt`, `phase2_report_free_onnx_24k.txt`):
  free-licence model per-square val 99.03% / held-out 99.59% / real 97.78% (dev model 99.61 /
  99.88 / 97.76; val and held-out contain NC styles the free model never saw). Phase-2 eval with
  it through ONNX: user screenshots 11/11, lichess 20/20, chess.com 25/40 auto / 31/40 with
  orientation given (dev model 27 / 33), synthetic screens 163/200 auto (dev 171). ONNX vs
  PyTorch: 0 of 42,944 eval predictions differ (max prob diff 4e-6). A 12k-step free model was
  similar (backup `square_classifier_free_12k.pt`). Review eval: 10 games (7 fetched live from
  Lichess/Chess.com, 3 sample PGNs), ~0.14 s per position; explanations 15 new over two runs: 13
  passed first time, 2 regenerated, 0 fallback; latency median ~4.5 s. Build: 379 MB (Stockfish
  105, cv2 112, PyQt6 73), ~90 s, audit passed (3,613 archived modules + zips scanned for keys),
  frozen self-check passed (ONNX ~44 ms first inference, bundled SF19, no key without the user's
  .env). Phase 6 API spend: ~$0.01 (two eval_phase6 runs; tests use fake clients).

## Known limitations
- Unusual chess.com piece sets ("8-bit", "light") misclassify; fix = more piece styles in training.
- Orientation is ambiguous on boards without coordinates in random/endgame positions (real games
  lock it in from the opening; manual flip/`--orientation` available).
- Side to move for positions seen without history may be assumed (flagged `turn_assumed`).
- Very low-contrast pastel themes can still be missed in cluttered screens (1/40); drag-select works.
- The dev model is trained with CC BY-NC-SA piece sets (non-commercial weights); the shipped
  model is not (slightly weaker on chess.com renders, see the Phase 6 results).
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
- Phase 5: the trainer was tested offscreen only (scripted positions fed as board-reader
  readings, real Stockfish + Claude), not on the real screen (rule: ask first). The rule checker
  passes text it cannot judge for sense: a few Expert explanations were garbled or added loose
  causal glue but used only fact words. Explanations use the depth-16 evaluation, so the wording
  can differ from the final eval shown later. Dataset lines are shorter than the Expert depth (24
  plies) in many openings (Najdorf/English Attack ends at ply 15): the line then ends as "End of
  the book". Uncurated openings pick the most common dataset move, which is not always the most
  popular/best one. Book-move explanations are triggered only for guided mode (practice would
  give the answer away). Ctrl+E only works while the panel has focus.
- Positional facts are basic (no outposts, king safety beyond castling, space or plans); the
  model is told not to invent them, so quiet moves get plain "what it does + evaluation" texts.
- Phase 6: the review window, hotkeys and packaged exe were tested offscreen / headless only
  (window rendered offscreen to an image; hotkeys registered with an unused combo plus a posted
  WM_HOTKEY, no key presses); the exe's GUI has not been launched on the real screen (rule: ask
  first). Mistake types are heuristics over 8-ply lines: a loss that only shows later (trapped
  piece, slow attack) gets "missed ways to win material" or "positional" even when a human would
  name it differently. Explanations start only after all selected games are analysed. Lichess
  returns every speed (bullet included); the Chess.com fetch looks back at most 6 monthly
  archives. Builds are GPLv3-bound (PyQt6, python-chess, Stockfish); no LICENSE file has been
  chosen for the repo. `docs/demo.gif` is a placeholder in the README (record it with the Record
  button). Global hotkeys are Windows-only.
