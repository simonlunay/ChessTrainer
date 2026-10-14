"""The ChessTrainer overlay app: board reading -> Stockfish -> arrows + explanation.

    python -m uv run chesstrainer                 # or: python -m chesstrainer
    python -m uv run chesstrainer --demo          # overlay visible to screen recorders
    python -m uv run chesstrainer --record-gif 20 # record a 20 s demo GIF once a board is found

Data flow: :class:`VisionThread` (capture + recognition, ``capture_fps``) emits a Snapshot per
frame -> :class:`TrainerApp` places the overlay on the board and, when a new position is
committed, asks :class:`AnalysisEngine` for it (stale searches are stopped). Arrows are hidden
whenever the screen shows something other than the analysed position (a move in progress, a new
position waiting for its first result, or no board).

The skill level picks how many candidates are drawn and whether a simpler "practical" move is
shown first (``explain/practical.py``). Once a position has been on screen for
``explain_delay_s`` and the engine has reached ``explain_min_depth``, the suggested move is
explained by Claude from facts computed with python-chess (``explain/``), at most once per
position + move + level (cached on disk). Every explanation passes the rule-based grounding check
(regenerated once, else the facts are shown); explanations can be switched off (no API calls).

Opening trainer mode (``openings/``): a :class:`TrainerSession` follows the game on screen. While
the game is in the chosen opening's book, the trainer's move is drawn (guided) or checked
(practice, spaced repetition), Stockfish evaluates just that move (``root_moves``) for the eval
bar and the explanation facts, and the opponent's replies are named. A move out of book is
announced, graded with the engine, and the app falls back to normal analysis.

Game review (``review/``, ``overlay/review_window.py``) opens in its own window; live reading is
paused while it is open. System-wide hotkeys (``overlay/hotkeys.py``) toggle the arrows,
explanations and pause, flip the board and cycle the modes while another window has focus.
"""

from __future__ import annotations

import argparse
import html
import logging
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import chess
from PyQt6.QtCore import QObject, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QGuiApplication, QScreen
from PyQt6.QtWidgets import QApplication

from chesstrainer.capture.screen import MonitorInfo, ScreenCapture
from chesstrainer.engine.analysis import Analysis, AnalysisEngine
from chesstrainer.engine.locate import DOWNLOAD_URL, find_stockfish
from chesstrainer.explain.explainer import (
    Explainer,
    ExplainWorker,
    Explanation,
    ExplanationCache,
)
from chesstrainer.explain.facts import compute_facts, eval_number, mover_cp
from chesstrainer.explain.practical import Suggestion, suggest
from chesstrainer.levels import Level, get_level
from chesstrainer.openings.book import Book, default_book
from chesstrainer.openings.explain import opening_facts
from chesstrainer.openings.practice import PracticeStore
from chesstrainer.openings.trainer import TrainerEvent, TrainerSession
from chesstrainer.overlay.board_overlay import Arrow, BoardOverlay
from chesstrainer.overlay.capture_exclusion import apply_capture_setting
from chesstrainer.overlay.gif_recorder import GifRecorder
from chesstrainer.overlay.hotkeys import ACTION_TEXT, GlobalHotkeys, effective_hotkeys
from chesstrainer.overlay.panel import SidePanel
from chesstrainer.overlay.screen_map import ScreenMap
from chesstrainer.overlay.vision_thread import Snapshot, VisionThread
from chesstrainer.paths import FROZEN
from chesstrainer.settings.config import Settings, load_settings, save_settings, settings_path
from chesstrainer.vision.detect import BoardBox
from chesstrainer.vision.position import PositionReading
from chesstrainer.vision.reader import BoardReader

if TYPE_CHECKING:
    from chesstrainer.overlay.review_window import ReviewWindow

log = logging.getLogger("chesstrainer")


@dataclass
class MoveLatency:
    """One position change, timed on the app's clock (perf_counter seconds)."""

    fen: str
    seen: float  # first captured frame that differed from the previous position
    committed: float  # frame on which the new position was committed
    arrows: float = 0.0  # arrows for it painted
    engine_s: float = 0.0  # engine time to the first displayed result
    depth: int = 0

    @property
    def total_ms(self) -> float:
        return (self.arrows - self.seen) * 1000


def find_qscreen(monitor: MonitorInfo) -> QScreen | None:
    """The QScreen showing an mss monitor (Qt 6 keeps each screen's native top-left)."""
    screens = QGuiApplication.screens()
    for s in screens:
        g = s.geometry()
        if (g.x(), g.y()) == (monitor.left, monitor.top):
            return s
    return QGuiApplication.primaryScreen()


class TrainerApp(QObject):
    engine_update = pyqtSignal(object)  # Analysis, emitted from the engine thread
    engine_error = pyqtSignal(str)
    recording_done = pyqtSignal(object, str)  # Path | None, message
    explain_result = pyqtSignal(int, object, object)  # request id, Explanation | None, error
    book_ready = pyqtSignal(object)  # Book, loaded on a background thread
    hotkey_pressed = pyqtSignal(str)  # action, emitted from the hotkey thread

    #: Order of the "switch mode" hotkey.
    MODES = ("analysis", "trainer", "review")

    #: Depth at which a move that left the book is graded (or when the search ends).
    GRADE_DEPTH = 14

    def __init__(
        self,
        settings: Settings,
        *,
        hide_from_capture: bool | None = None,
        show_grid: bool = False,
        calibration: bool = False,
        record_gif_seconds: float | None = None,
        book: Book | None = None,
        practice_store: PracticeStore | None = None,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.hide_from_capture = (
            settings.hide_overlay_from_capture if hide_from_capture is None else hide_from_capture
        )
        self.monitor = ScreenCapture.monitors()[settings.monitor]
        self.overlay = BoardOverlay()
        self.overlay.show_grid = show_grid
        self.overlay.calibration = calibration
        self.panel = SidePanel()
        self.vision = VisionThread(settings)

        # Current state (GUI thread only).
        self.box: BoardBox | None = None
        self.white_bottom = True
        self.pending = False
        self.moving = False
        self.reading_fen: str | None = None  # last committed position from the vision side
        self.board: chess.Board | None = None  # position sent to the engine
        self.turn_assumed = False
        self.req_id: int | None = None
        self.analysis: Analysis | None = None
        self.paused = False
        self.recorder: GifRecorder | None = None
        self._record_on_board = record_gif_seconds
        self._panel_placed = settings.panel_pos is not None
        self._move_seen: float | None = None
        self._open_latency: MoveLatency | None = None
        self.latencies: list[MoveLatency] = []
        self.last_snapshot: Snapshot | None = None

        self.level: Level = get_level(settings.skill_level)
        self.suggestion: Suggestion | None = None
        self._position_t = 0.0  # when the analysed position appeared
        self._explained: tuple[str, str, str] | None = None  # (fen, move, level) requested
        self._facts_text = ""
        self.explain_req: int | None = None
        self.explanations: list[tuple[str, str, Explanation]] = []  # (fen, level, result)
        self.explainer: Explainer | None = None
        self.explain_worker: ExplainWorker | None = None
        self.explain_status = ""
        self._session_t0 = time.time()
        self._setup_explainer()
        self._explain_timer = QTimer(self)
        self._explain_timer.setSingleShot(True)
        self._explain_timer.timeout.connect(self._maybe_explain)

        # Opening trainer (GUI thread only).
        self.book: Book | None = book
        self._book_loading = False
        self.practice_store = practice_store
        self.session: TrainerSession | None = None
        self.trainer_message = ""
        self._root: tuple[chess.Move, ...] = ()  # moves the current search is restricted to
        self._grade: dict[str, object] | None = None  # out-of-book move waiting for its grade

        # Review window and hotkeys (GUI thread only).
        self.review_window: ReviewWindow | None = None
        self._paused_before_review = False
        self.overlay_hidden = False
        self.hotkeys: GlobalHotkeys | None = None

        self.engine: AnalysisEngine | None = None
        self.engine_status = ""
        search = find_stockfish(settings.engine_path)
        self.engine_path = search.path
        if search.path is not None:
            self.engine = AnalysisEngine(
                search.path,
                self.engine_update.emit,
                self.engine_error.emit,
                threads=settings.engine_threads,
                hash_mb=settings.engine_hash_mb,
                multipv=settings.multipv,
                max_seconds=settings.engine_max_seconds,
            )
        else:
            message = search.setup_message()
            print(message, file=sys.stderr)
            link = f"<a href='{DOWNLOAD_URL}'>{DOWNLOAD_URL}</a>"
            self.engine_status = (
                "<pre style='white-space: pre-wrap'>"
                + html.escape(message).replace(html.escape(DOWNLOAD_URL), link)
                + "</pre>"
            )

        self.engine_update.connect(self.on_engine_update)
        self.engine_error.connect(self.on_engine_error)
        self.recording_done.connect(self.on_recording_done)
        self.explain_result.connect(self.on_explain_result)
        self.panel.level_changed.connect(self.set_level)
        self.panel.explanations_toggled.connect(self.set_explanations)
        self.panel.mode_changed.connect(self.set_mode)
        self.panel.choose_opening_clicked.connect(self.choose_opening)
        self.book_ready.connect(self._on_book_ready)
        self.hotkey_pressed.connect(self.on_hotkey)
        self.panel.set_explanations_enabled(settings.explanations)
        self.panel.set_mode(settings.app_mode)
        self.vision.snapshot.connect(self.on_snapshot)
        self.vision.failed.connect(lambda msg: self.panel.set_status(msg))
        self.panel.flip_clicked.connect(self.flip)
        self.panel.select_clicked.connect(self.select_board)
        self.panel.switch_turn_clicked.connect(self.switch_turn)
        self.panel.pause_toggled.connect(self.set_paused)
        self.panel.record_toggled.connect(self.toggle_recording)
        self.panel.closed.connect(self.quit)
        self._record_timer = QTimer(self)
        self._record_timer.timeout.connect(self._update_record_button)

    # -- lifecycle ---------------------------------------------------------------------------

    def start(self) -> None:
        if not self.settings.level_chosen:
            from chesstrainer.overlay.level_dialog import ask_level

            self.settings.skill_level = ask_level(self.settings.skill_level)
            self.settings.level_chosen = True
            save_settings(self.settings)
            self.level = get_level(self.settings.skill_level)
        self.panel.set_level(self.level.key)
        if self.settings.app_mode == "trainer":
            self._start_trainer()
        if self.settings.panel_pos:
            self.panel.move(*self.settings.panel_pos)
        self.panel.show()
        apply_capture_setting(self.panel, self.hide_from_capture)
        self._status()
        self._start_hotkeys()
        if self.engine is not None:
            self.engine.start()
        self.vision.start()

    def quit(self) -> None:
        if self.hotkeys is not None:
            self.hotkeys.stop()
        if self.review_window is not None:
            self.review_window.stop()
            self.review_window.close()
        if self.recorder is not None:
            self.recorder.stop()
        pos = self.panel.pos()
        self.settings.panel_pos = [pos.x(), pos.y()]
        save_settings(self.settings)
        self.vision.stop()
        if self.engine is not None:
            self.engine.close()
        if self.explain_worker is not None:
            self.explain_worker.close()
        self.overlay.close()
        app = QApplication.instance()
        if app is not None:
            app.quit()

    # -- vision ------------------------------------------------------------------------------

    def screen_map(self) -> ScreenMap:
        screen = find_qscreen(self.monitor)
        if screen is None:
            return ScreenMap(self.monitor, 1.0)
        g = screen.geometry()
        return ScreenMap(self.monitor, screen.devicePixelRatio(), (float(g.x()), float(g.y())))

    def on_snapshot(self, snap: Snapshot) -> None:
        if self.paused:
            return
        self.last_snapshot = snap
        # A board that moved since the last frame is probably still moving (window being
        # dragged): arrows would trail behind it, so they wait for a frame where it stays put.
        self.moving = (
            snap.box is not None and self.box is not None and snap.box.corner_error(self.box) > 0.03
        )
        self.box = snap.box
        if snap.box is None:
            self._set_overlay_visible(False)
            self._status()
            return
        sm = self.screen_map()
        self.overlay.place(sm.to_logical(snap.box), sm.dpr, sm.origin or (0.0, 0.0))
        self._set_overlay_visible(True)
        if not self._panel_placed:
            self._place_panel(sm)
        if self._record_on_board is not None:
            seconds, self._record_on_board = self._record_on_board, None
            self.panel.set_recording(True)
            self.toggle_recording(True, max_seconds=seconds)

        self.white_bottom = snap.white_bottom
        self.pending = snap.pending
        if snap.pending and self._move_seen is None:
            self._move_seen = snap.t
        reading = snap.reading
        if reading is not None and reading.fen != self.reading_fen:
            self._open_latency = MoveLatency(reading.fen, self._move_seen or snap.t, snap.t)
            self._new_position(reading)
        if not snap.pending:
            self._move_seen = None
        self._refresh_arrows()
        self._maybe_explain()

    def _new_position(self, reading: PositionReading) -> None:
        self.reading_fen = reading.fen
        self.turn_assumed = reading.turn_assumed
        board = reading.board.copy(stack=False)
        if self.session is not None:
            previous = self.analysis
            events = self.session.observe(reading.board)
            if self.session.board is not None:
                # The trainer knows the game: exact side to move, castling and en passant.
                board = self.session.board.copy(stack=False)
                self.turn_assumed = False
            self._trainer_events(events, previous)
        self._analyse(board)

    def _analyse(self, board: chess.Board) -> None:
        self.board = board
        self.analysis = None
        self.suggestion = None
        self.req_id = None
        self._root = ()
        if self.session is not None and self.session.board is not None:
            expected = self.session.expected_move()
            if expected is not None and self.session.board.board_fen() == board.board_fen():
                self._root = (expected,)
        self.panel.clear_analysis()
        self._reset_explanation()
        self.panel.set_turn_assumed(self.turn_assumed)
        if self.engine is None:
            self._status()
            return
        if not board.is_valid() or board.outcome() is not None:
            self.engine.cancel()
            self._status()
            return
        self.req_id = self.engine.analyze(board, self._root)
        self._status()

    def _book_arrow(self) -> chess.Move | None | bool:
        """Trainer in book: the move to draw (None = draw nothing). False = not in book."""
        s = self.session
        if s is None or not s.in_book or self.board is None or s.board is None:
            return False
        if s.board.board_fen() != self.board.board_fen():
            return False
        return s.arrow_move()

    def _refresh_arrows(self) -> None:
        book = self._book_arrow()
        if book is not False:
            if book is None or book is True or self.pending or self.moving:
                self.overlay.clear()
            else:
                self.overlay.set_arrows([Arrow.from_move(book, 1)], self.white_bottom)
            return
        a = self.analysis
        show = (
            a is not None
            and not self.pending
            and not self.moving
            and self.board is not None
            and a.fen == self.board.fen()
        )
        if not show or a is None:
            self.overlay.clear()
            return
        lines = self.suggestion.lines if self.suggestion is not None else a.lines
        arrows = [Arrow.from_move(line.move, line.rank) for line in lines]
        first = not self.overlay.arrows
        self.overlay.set_arrows(arrows, self.white_bottom)
        lat = self._open_latency
        if first and lat is not None and lat.fen == self.reading_fen and lat.arrows == 0.0:
            self.overlay.repaint()  # paint now so the timestamp is honest
            lat.arrows = time.perf_counter()
            lat.engine_s, lat.depth = a.elapsed, a.depth
            self.latencies.append(lat)
            log.info(
                "move -> arrows %.0f ms (seen -> committed %.0f ms, engine %.0f ms, depth %d)",
                lat.total_ms,
                (lat.committed - lat.seen) * 1000,
                lat.engine_s * 1000,
                lat.depth,
            )

    # -- engine ------------------------------------------------------------------------------

    def on_engine_update(self, analysis: Analysis) -> None:
        if analysis.request_id != self.req_id:
            return  # result for a position that is no longer on screen
        self.analysis = analysis
        self._show_analysis()
        self._refresh_arrows()
        self._maybe_grade()
        self._maybe_explain()
        if analysis.final:
            self._status()

    def _show_analysis(self) -> None:
        a, board = self.analysis, self.board
        if a is None or board is None or a.fen != board.fen():
            return
        if self._root:
            # Book move only: the eval bar and the move's line; practice mode hides the move.
            self.suggestion = Suggestion(a.lines[:1], a.lines[0], False)
            practice = self.session is not None and self.session.arrow_move() is None
            shown = () if practice else self.suggestion.lines
            self.panel.show_analysis(a, self.white_bottom, shown, "")
            return
        self.suggestion = suggest(board, a.lines, self.level)
        note = ""
        if self.suggestion.practical:
            best, chosen = self.suggestion.engine_best, self.suggestion.line
            note = (
                f"Suggested: {board.san(chosen.move)} ({eval_number(chosen)}), easier to play. "
                f"Stockfish's top move {board.san(best.move)} ({eval_number(best)}) is "
                "slightly better."
            )
        self.panel.show_analysis(a, self.white_bottom, self.suggestion.lines, note)

    # -- explanations ------------------------------------------------------------------------

    def _reset_explanation(self) -> None:
        self._position_t = time.perf_counter()
        self._explained = None
        self.explain_req = None
        self._explain_timer.start(int(self.settings.explain_delay_s * 1000) + 10)
        if self.explain_status:
            self.panel.set_explanation(self.explain_status, status=True)
        else:
            self.panel.set_explanation("")

    def _setup_explainer(self) -> None:
        """Create the explainer when explanations are on (never touches the API when off)."""
        if not self.settings.explanations:
            self.explain_status = (
                f"Explanations are off ({SidePanel.EXPLAIN_SHORTCUT} or the checkbox turns them "
                "on). No API calls are made."
            )
            return
        if self.explainer is None:
            self.explainer = Explainer(
                self.settings.explain_model, self.settings.explain_effort, cache=ExplanationCache()
            )
            if self.explainer.available:
                self.explain_worker = ExplainWorker(self.explainer, self.explain_result.emit)
            else:
                print(self.explainer.unavailable_reason, file=sys.stderr)
        self.explain_status = "" if self.explainer.available else self.explainer.unavailable_reason

    def set_explanations(self, on: bool) -> None:
        self.settings.explanations = on
        save_settings(self.settings)
        self.panel.set_explanations_enabled(on)
        self._explained = None
        self.explain_req = None
        self._setup_explainer()
        log.info("explanations %s", "on" if on else "off")
        if self.explain_status:
            self.panel.set_explanation(self.explain_status, status=True)
        else:
            self.panel.set_explanation("")
        self._maybe_explain()

    def _maybe_explain(self) -> None:
        """Explain the suggested move once the position is stable and the analysis settled."""
        if not self.settings.explanations:
            return
        if self.explainer is None or not self.explainer.available:
            return
        a, board, sug = self.analysis, self.board, self.suggestion
        if a is None or board is None or sug is None or a.fen != board.fen():
            return
        if self._root and self.session is not None and self.session.arrow_move() is None:
            return  # practice: explaining the move would give the answer away
        if self.session is not None and self.session.in_book and not self._root:
            return  # opponent's turn in book: their reply is named, not explained
        if self.pending or self.paused:
            return
        waited = time.perf_counter() - self._position_t
        if waited < self.settings.explain_delay_s:
            self._explain_timer.start(int((self.settings.explain_delay_s - waited) * 1000) + 10)
            return
        if not a.final and a.depth < self.settings.explain_min_depth:
            return  # the next engine update calls this again
        key = (board.fen(), sug.move.uci(), self.level.key)
        if key == self._explained:
            return
        self._explained = key
        if self._root and self.session is not None:
            facts = opening_facts(self.session, board, sug.line, self.level)
        else:
            facts = compute_facts(
                board,
                sug.line,
                self.level,
                engine_best=sug.engine_best if sug.practical else None,
                alternatives=sug.lines[1:] if self.level.explanation.eval_numbers else (),
            )
        self._facts_text = facts.text()
        hit = self.explainer.cached(facts, self.level)
        if hit is not None:
            self.explain_req = None
            self._show_explanation(hit)
            return
        assert self.explain_worker is not None
        self.panel.set_explanation(f"Explaining {facts.san}\u2026", status=True)
        self.explain_req = self.explain_worker.submit(facts, self.level)

    def on_explain_result(
        self, request_id: int, result: Explanation | None, error: str | None
    ) -> None:
        if request_id != self.explain_req or not self.settings.explanations:
            return  # for a position or level that is no longer shown
        self.explain_req = None
        if result is None:
            # No automatic retry (this runs at 10 Hz): the next position or level change retries.
            self.panel.set_explanation(error or "Explanation failed.", status=True)
            return
        self._show_explanation(result)

    def _show_explanation(self, result: Explanation) -> None:
        source = "cached" if result.cached else f"{result.seconds:.1f}s"
        log.info(
            "explanation (%s, %s, %s): %s", self.level.key, source, result.outcome, result.text
        )
        if not result.cached and self.explainer is not None and self.explainer.cache is not None:
            stats = self.explainer.cache.grounding_stats(since=self._session_t0)
            log.info(
                "grounding check this session: %d passed, %d regenerated, %d shown as facts",
                stats["passed"],
                stats["regenerated"],
                stats["fallback"],
            )
        if self.board is not None:
            self.explanations.append((self.board.fen(), self.level.key, result))
        if result.outcome == "fallback":
            head = "Computed facts (the explanation did not pass the fact check):"
        elif result.outcome == "regenerated":
            head = f"Rewritten once by {result.model} to pass the fact check. Facts:"
        else:
            head = f"Checked against the facts given to {result.model}:"
        self.panel.set_explanation(result.text, tooltip=f"{head}\n{self._facts_text}")

    def set_level(self, key: str) -> None:
        self.level = get_level(key)
        self.settings.skill_level = self.level.key
        self.settings.level_chosen = True
        save_settings(self.settings)
        if self.review_window is not None:
            self.review_window.set_level(self.level)
        if self.session is not None:
            self.session.level = self.level  # line depth follows the level
            self._update_trainer_panel()
        self._explained = None
        self.explain_req = None
        self._show_analysis()
        self._refresh_arrows()
        self._maybe_explain()

    def on_engine_error(self, message: str) -> None:
        log.error("%s", message)
        self.engine_status = html.escape(message)
        self._status()

    # -- game review -------------------------------------------------------------------------

    def open_review(self) -> None:
        from chesstrainer.overlay.review_window import ReviewWindow

        if self.review_window is None:
            self.review_window = ReviewWindow(
                self.settings,
                self.level,
                self.engine_path,
                self.review_explainer,
                explanations_on=lambda: self.settings.explanations,
            )
            self.review_window.closed.connect(self._on_review_closed)
        if not self.review_window.isVisible():
            # Live reading pauses while reviewing (no screen capture, no live Stockfish).
            self._paused_before_review = self.paused
            if not self.paused:
                self.panel.pause.setChecked(True)
        self.panel.set_mode("review")
        self.review_window.show()
        self.review_window.raise_()
        self.review_window.activateWindow()

    def _on_review_closed(self) -> None:
        self.panel.set_mode(self.settings.app_mode)
        if not self._paused_before_review:
            self.panel.pause.setChecked(False)

    def review_explainer(self) -> tuple[Explainer | None, str]:
        """The explainer for a review, or None with the reason (off = no API calls)."""
        if not self.settings.explanations:
            return None, (
                "Explanations are off, so mistakes are not explained (no API calls); "
                f"turn them on with {SidePanel.EXPLAIN_SHORTCUT} or the panel checkbox."
            )
        self._setup_explainer()
        if self.explainer is None or not self.explainer.available:
            return None, self.explain_status
        return self.explainer, ""

    # -- hotkeys -----------------------------------------------------------------------------

    def _start_hotkeys(self) -> None:
        if not self.settings.global_hotkeys:
            return
        bindings = effective_hotkeys(self.settings.hotkeys)
        self.hotkeys = GlobalHotkeys(bindings, self.hotkey_pressed.emit)
        self.hotkeys.start()
        ok = self.hotkeys.registered
        text = " · ".join(f"{combo} {ACTION_TEXT[a].split(' (')[0]}" for a, combo in ok.items())
        tip = "System-wide hotkeys (work in any window):\n" + "\n".join(
            f"{combo}: {ACTION_TEXT[a]}" for a, combo in ok.items()
        )
        if self.hotkeys.failed:
            tip += "\n\nNot available:\n" + "\n".join(
                f"{ACTION_TEXT.get(a, a)}: {why}" for a, why in self.hotkeys.failed.items()
            )
            text += " (some hotkeys are taken; hover for details)"
        self.panel.set_hotkeys(f"Hotkeys: {text}" if ok else text, tip)

    def _hotkey_for(self, action: str) -> str:
        return self.hotkeys.registered.get(action, "") if self.hotkeys is not None else ""

    def on_hotkey(self, action: str) -> None:
        log.info("hotkey: %s", action)
        if action == "toggle_overlay":
            self.set_overlay_hidden(not self.overlay_hidden)
        elif action == "toggle_explanations":
            self.set_explanations(not self.settings.explanations)
        elif action == "pause":
            self.panel.pause.toggle()
        elif action == "flip":
            self.flip()
        elif action == "switch_mode":
            reviewing = self.review_window is not None and self.review_window.isVisible()
            current = "review" if reviewing else self.settings.app_mode
            self.set_mode(self.MODES[(self.MODES.index(current) + 1) % len(self.MODES)])

    def set_overlay_hidden(self, hidden: bool) -> None:
        """Hide the arrows without stopping the analysis (the panel keeps updating)."""
        self.overlay_hidden = hidden
        self.panel.set_overlay_hidden(hidden, self._hotkey_for("toggle_overlay"))
        if hidden:
            self._set_overlay_visible(False)
        elif self.box is not None and not self.paused:
            self._set_overlay_visible(True)

    # -- opening trainer ---------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        if mode == "review":
            self.open_review()
            return
        if self.review_window is not None and self.review_window.isVisible():
            self.review_window.close()  # back to live reading
        self.settings.app_mode = "trainer" if mode == "trainer" else "analysis"
        save_settings(self.settings)
        self.panel.set_mode(self.settings.app_mode)
        if self.settings.app_mode == "trainer":
            self._start_trainer()
        else:
            self.session = None
            self.trainer_message = ""
            self._grade = None
            self._reanalyse()

    def _start_trainer(self) -> None:
        if self.book is None:
            self.panel.set_trainer("Loading the opening book…")
            self._load_book()
            return
        if not self.settings.trainer_opening:
            self.panel.set_trainer("No opening chosen yet.", prompt="Choose an opening to train.")
            QTimer.singleShot(0, self.choose_opening)
            return
        self._new_session()

    def _load_book(self) -> None:
        if self._book_loading:
            return
        self._book_loading = True
        threading.Thread(
            target=lambda: self.book_ready.emit(default_book()), name="book", daemon=True
        ).start()

    def _on_book_ready(self, book: Book) -> None:
        self.book = book
        self._book_loading = False
        if self.settings.app_mode == "trainer":
            self._start_trainer()

    def choose_opening(self) -> None:
        from chesstrainer.overlay.opening_dialog import OpeningChoice, ask_opening

        if self.book is None:
            self.book = default_book()
        s = self.settings
        current = (
            OpeningChoice(s.trainer_opening, s.trainer_side, s.trainer_mode)
            if s.trainer_opening
            else None
        )
        choice = ask_opening(self.book, self.level, current)
        if choice is None:
            if not s.trainer_opening:
                self.panel.set_trainer("No opening chosen.", prompt="Choose an opening to train.")
            return
        self.set_opening(choice.name, choice.side, choice.mode)

    def set_opening(self, name: str, side: str, mode: str) -> None:
        s = self.settings
        s.trainer_opening = name
        s.trainer_side = "black" if side == "black" else "white"
        s.trainer_mode = "practice" if mode == "practice" else "guided"
        s.app_mode = "trainer"
        save_settings(s)
        self.panel.set_mode("trainer")
        if self.book is None:
            self.book = default_book()
        self._new_session()

    def _new_session(self) -> None:
        assert self.book is not None
        s = self.settings
        if s.trainer_mode == "practice" and self.practice_store is None:
            self.practice_store = PracticeStore()
        self.session = TrainerSession.create(
            self.book,
            s.trainer_opening,
            chess.WHITE if s.trainer_side == "white" else chess.BLACK,
            self.level,
            s.trainer_mode,
            self.practice_store if s.trainer_mode == "practice" else None,
        )
        if self.session.tree.size == 0 and self.session.tree.spec is None:
            self.trainer_message = f"No book lines found for {s.trainer_opening!r}."
        else:
            self.trainer_message = ""
        self._grade = None
        log.info(
            "trainer: %s as %s (%s), %d book lines",
            s.trainer_opening,
            s.trainer_side,
            s.trainer_mode,
            self.session.tree.size,
        )
        self._reanalyse()

    def _reanalyse(self) -> None:
        """Re-run the current screen position through the trainer (or plain analysis)."""
        snap = self.last_snapshot
        self.reading_fen = None
        if snap is not None and snap.reading is not None:
            self._new_position(snap.reading)
        elif self.session is None and self.board is not None:
            self._analyse(self.board)
        else:
            self._update_trainer_panel()
        self._refresh_arrows()

    def _trainer_events(self, events: list[TrainerEvent], previous: Analysis | None) -> None:
        if events:
            self.trainer_message = " ".join(e.message for e in events)
            for e in events:
                log.info("trainer: %s: %s", e.kind, e.message)
        self._grade = None
        for e in events:
            if not e.grade or e.move is None or e.mover is None:
                continue
            before = None
            if previous is not None and previous.fen == e.fen_before:
                before = mover_cp(previous.best, e.mover)
            board = chess.Board(e.fen_before)
            self._grade = {
                "san": board.san(e.move),
                "mover": e.mover,
                "before": before,
                "versus": "the book move" if e.kind == "user_left" else "the best move",
            }
            if before is None:
                self.trainer_message += " (No evaluation of the previous position yet.)"
        self._update_trainer_panel()

    def _maybe_grade(self) -> None:
        """Say whether the move that left the book was a mistake, once the engine knows."""
        g, a, board = self._grade, self.analysis, self.board
        if g is None or a is None or board is None or a.fen != board.fen():
            return
        if not a.final and a.depth < self.GRADE_DEPTH:
            return
        self._grade = None
        before = g["before"]
        mover = g["mover"]
        if not isinstance(before, int) or not isinstance(mover, bool):
            return
        loss = before - mover_cp(a.best, mover)
        san, versus = g["san"], g["versus"]
        if loss >= 50_000:
            verdict = f"Stockfish: {san} is a blunder: it allows a forced mate."
        elif before >= 50_000 and loss > 0:
            verdict = f"Stockfish: {san} misses a forced mate."
        else:
            severity = self.level.classify_loss(loss)
            pawns = f"about {loss / 100:.1f} pawns worse than {versus}"
            if severity is not None:
                article = "an" if severity == "inaccuracy" else "a"
                verdict = f"Stockfish: {san} is {article} {severity} ({pawns})."
            elif loss < 30:
                verdict = f"Stockfish: {san} is a reasonable move (about as good as {versus})."
            else:
                verdict = f"Stockfish: {san} is {pawns}, but not a real mistake at your level."
        self.trainer_message += " " + verdict
        log.info("trainer: grade: %s (loss %d cp)", verdict, loss)
        self._update_trainer_panel()

    def _update_trainer_panel(self) -> None:
        s = self.session
        if s is None:
            return
        mode = "practice" if s.mode == "practice" else "guided"
        title = f"{s.tree.name} - you play {s.side_text} ({mode})"
        self.panel.set_trainer(
            title,
            s.variation_name(),
            self.trainer_message,
            s.prompt(),
            s.practice_summary(),
        )

    # -- status line -------------------------------------------------------------------------

    def _status(self) -> None:
        if self.paused:
            self.panel.set_status("Paused.")
            return
        if self.engine is None or (self.engine_status and self.req_id is None):
            self.panel.set_status(self.engine_status, rich=True)
            return
        if self.box is None:
            self.panel.set_status(
                "Looking for a chess board on screen…\n"
                "If it is not found, use “Select board…” to drag a box around it."
            )
            return
        b = self.board
        if b is None:
            self.panel.set_status("Board found; reading the position…")
            return
        if not b.is_valid():
            self.panel.set_status(
                "The position does not look legal, so it is not analysed.\n"
                f"Read as: {b.fen()}\nIf the board is upside down, press “Flip board”."
            )
            return
        outcome = b.outcome()
        if outcome is not None:
            if outcome.winner is None:
                self.panel.set_status(f"Game over: draw ({outcome.termination.name.lower()}).")
            else:
                winner = "White" if outcome.winner else "Black"
                self.panel.set_status(f"Checkmate: {winner} wins.")
            return
        side = "White" if b.turn else "Black"
        note = " (guessed; no move seen yet)" if self.turn_assumed else ""
        self.panel.set_status(f"{side} to move{note}.")

    # -- controls ----------------------------------------------------------------------------

    def flip(self) -> None:
        self.vision.call(lambda reader: reader.flip())
        self._reset_position()

    def switch_turn(self) -> None:
        if self.board is None:
            return
        board = self.board.copy(stack=False)
        board.turn = not board.turn
        board.ep_square = None
        self.turn_assumed = False
        self._analyse(board)

    def select_board(self) -> None:
        from chesstrainer.overlay.region_select import select_region

        self._set_overlay_visible(False)
        region = select_region(self.settings.monitor)
        if region is None:
            return
        self.settings.locked_region = list(region)
        save_settings(self.settings)

        def lock(reader: BoardReader) -> None:
            reader.boards.locked = BoardBox(*region)
            reader.boards.reset()
            reader.positions.reset()

        self.vision.call(lock)
        self._reset_position()

    def set_paused(self, paused: bool) -> None:
        self.paused = paused
        self.vision.paused = paused
        if paused:
            self._set_overlay_visible(False)
            if self.engine is not None:
                self.engine.cancel()
            self.req_id = None
        self._reset_position()
        self._status()

    def _reset_position(self) -> None:
        self._grade = None
        self.reading_fen = None
        self.board = None
        self.analysis = None
        self.suggestion = None
        self.req_id = None
        self.overlay.clear()
        self.panel.clear_analysis()
        self._reset_explanation()

    # -- recording ---------------------------------------------------------------------------

    def toggle_recording(self, on: bool, max_seconds: float = 60.0) -> None:
        if not on:
            if self.recorder is not None:
                self.recorder.stop()
                self.panel.set_recording(True, "Saving GIF…")
            return
        region = self._record_region()
        if region is None:
            self.panel.set_recording(False)
            self.panel.set_status("No board found to record.")
            return
        # Screen recorders (and this GIF) must see the overlay while recording.
        apply_capture_setting(self.overlay, False)
        apply_capture_setting(self.panel, False)
        self.recorder = GifRecorder(
            self.settings.monitor,
            region,
            max_seconds=max_seconds,
            on_done=self.recording_done.emit,
        )
        self.recorder.start()
        self._record_timer.start(500)
        log.info("recording GIF of region %s for up to %.0f s", region, max_seconds)

    def _update_record_button(self) -> None:
        r = self.recorder
        if r is not None and not r._stop.is_set():
            self.panel.set_recording(True, f"■ Stop recording ({r.elapsed:.0f}s)")

    def on_recording_done(self, path: Path | None, message: str) -> None:
        self._record_timer.stop()
        self.recorder = None
        apply_capture_setting(self.overlay, self.hide_from_capture)
        apply_capture_setting(self.panel, self.hide_from_capture)
        self.panel.set_recording(False)
        if path is not None:
            url = QUrl.fromLocalFile(str(path.parent)).toString()
            self.panel.set_status(
                f"GIF {html.escape(message)}:<br><a href='{url}'>{html.escape(str(path))}</a>",
                rich=True,
            )
            print(f"GIF {message}: {path}")
        else:
            self.panel.set_status(message)

    def _record_region(self) -> tuple[int, int, int, int] | None:
        """Board plus a small margin, and the side panel if it sits right next to the board."""
        if self.box is None:
            return None
        m = self.monitor
        b = self.box
        pad = 0.04 * b.w
        x0, y0, x1, y1 = b.x - pad, b.y - pad, b.x + b.w + pad, b.y + b.h + pad
        sm = self.screen_map()
        g = self.panel.frameGeometry()
        px0, py0 = sm.to_physical(g.x(), g.y())
        px1, py1 = sm.to_physical(g.x() + g.width(), g.y() + g.height())
        ux0, uy0, ux1, uy1 = min(x0, px0), min(y0, py0), max(x1, px1), max(y1, py1)
        if (ux1 - ux0) * (uy1 - uy0) <= 2.0 * (x1 - x0) * (y1 - y0):
            x0, y0, x1, y1 = ux0, uy0, ux1, uy1
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(float(m.width), x1), min(float(m.height), y1)
        # Even sizes keep GIF encoders and video converters happy.
        w, h = int(x1 - x0) // 2 * 2, int(y1 - y0) // 2 * 2
        return round(x0), round(y0), w, h

    # -- windows -----------------------------------------------------------------------------

    def _set_overlay_visible(self, visible: bool) -> None:
        visible = visible and not self.overlay_hidden
        if visible and not self.overlay.isVisible():
            self.overlay.show()
            recording = self.recorder is not None
            apply_capture_setting(self.overlay, self.hide_from_capture and not recording)
        elif not visible and self.overlay.isVisible():
            self.overlay.hide()

    def _place_panel(self, sm: ScreenMap) -> None:
        """First board sighting: put the panel beside the board (right, else left)."""
        self._panel_placed = True
        x, y, w, _ = sm.to_logical(self.box) if self.box else (0.0, 0.0, 0.0, 0.0)
        screen = find_qscreen(self.monitor)
        if screen is None:
            return
        avail = screen.availableGeometry()
        pw = self.panel.frameGeometry().width()
        right = x + w + 12
        px = right if right + pw <= avail.right() else max(avail.left(), x - pw - 12)
        py = max(avail.top(), min(int(y), avail.bottom() - self.panel.frameGeometry().height()))
        self.panel.move(int(px), int(py))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="chesstrainer",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--monitor", type=int, help="mss monitor index (1 = primary; saved)")
    parser.add_argument(
        "--orientation", choices=["auto", "white", "black"], help="side at the bottom (saved)"
    )
    parser.add_argument("--select", action="store_true", help="drag-select the board (saved)")
    parser.add_argument("--clear-region", action="store_true", help="forget a selected region")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="show the overlay to screen recorders this session (capture hiding off)",
    )
    parser.add_argument(
        "--record-gif",
        type=float,
        metavar="SECONDS",
        help="record a demo GIF (capture hiding off while recording) once a board is found",
    )
    parser.add_argument("--grid", action="store_true", help="draw the detected grid (debug)")
    parser.add_argument(
        "--calibrate", action="store_true", help="draw dots on square centres (alignment check)"
    )
    parser.add_argument("--seconds", type=float, help="quit after this many seconds")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--self-check",
        type=Path,
        metavar="REPORT",
        help="headless installation check (no window, no capture); writes a JSON report",
    )
    args = parser.parse_args(argv)
    if args.self_check:
        from chesstrainer.selfcheck import run

        sys.exit(run(args.self_check))
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if FROZEN or sys.stderr is None:
        # Packaged app without a console: log to a file next to the settings.
        log_file = settings_path().parent / "chesstrainer.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers = [logging.FileHandler(log_file, mode="w", encoding="utf-8")]
        sys.excepthook = lambda *exc: log.critical("unhandled error", exc_info=exc)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )

    settings = load_settings()
    changed = False
    if args.monitor is not None:
        settings.monitor, changed = args.monitor, True
    if args.orientation:
        settings.orientation, changed = args.orientation, True
    if args.clear_region:
        settings.locked_region, changed = None, True

    app = QApplication.instance() or QApplication(sys.argv[:1])
    assert isinstance(app, QApplication)
    if args.select:
        from chesstrainer.overlay.region_select import select_region

        region = select_region(settings.monitor)
        if region is not None:
            settings.locked_region, changed = list(region), True
    if changed:
        save_settings(settings)

    trainer = TrainerApp(
        settings,
        hide_from_capture=False if args.demo else None,
        show_grid=args.grid,
        calibration=args.calibrate,
        record_gif_seconds=args.record_gif,
    )
    # Let Ctrl+C in the console close the app (Python only sees signals between Qt events).
    signal.signal(signal.SIGINT, lambda *_: trainer.quit())
    tick = QTimer()
    tick.timeout.connect(lambda: None)
    tick.start(200)
    if args.seconds:
        QTimer.singleShot(int(args.seconds * 1000), trainer.quit)
    trainer.start()
    app.exec()


if __name__ == "__main__":
    main()
