"""The ChessTrainer overlay app: live board reading -> Stockfish -> arrows over the board.

    python -m uv run chesstrainer                 # or: python -m chesstrainer
    python -m uv run chesstrainer --demo          # overlay visible to screen recorders
    python -m uv run chesstrainer --record-gif 20 # record a 20 s demo GIF once a board is found

Data flow: :class:`VisionThread` (capture + recognition, ``capture_fps``) emits a Snapshot per
frame -> :class:`TrainerApp` places the overlay on the board and, when a new position is
committed, asks :class:`AnalysisEngine` for it (stale searches are stopped). Arrows are hidden
whenever the screen shows something other than the analysed position (a move in progress, a new
position waiting for its first result, or no board).
"""

from __future__ import annotations

import argparse
import html
import logging
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import chess
from PyQt6.QtCore import QObject, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QGuiApplication, QScreen
from PyQt6.QtWidgets import QApplication

from chesstrainer.capture.screen import MonitorInfo, ScreenCapture
from chesstrainer.engine.analysis import Analysis, AnalysisEngine
from chesstrainer.engine.locate import DOWNLOAD_URL, find_stockfish
from chesstrainer.overlay.board_overlay import Arrow, BoardOverlay
from chesstrainer.overlay.capture_exclusion import apply_capture_setting
from chesstrainer.overlay.gif_recorder import GifRecorder
from chesstrainer.overlay.panel import SidePanel
from chesstrainer.overlay.screen_map import ScreenMap
from chesstrainer.overlay.vision_thread import Snapshot, VisionThread
from chesstrainer.settings.config import Settings, load_settings, save_settings
from chesstrainer.vision.detect import BoardBox
from chesstrainer.vision.position import PositionReading
from chesstrainer.vision.reader import BoardReader

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

    def __init__(
        self,
        settings: Settings,
        *,
        hide_from_capture: bool | None = None,
        show_grid: bool = False,
        calibration: bool = False,
        record_gif_seconds: float | None = None,
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

        self.engine: AnalysisEngine | None = None
        self.engine_status = ""
        search = find_stockfish(settings.engine_path)
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
        if self.settings.panel_pos:
            self.panel.move(*self.settings.panel_pos)
        self.panel.show()
        apply_capture_setting(self.panel, self.hide_from_capture)
        self._status()
        if self.engine is not None:
            self.engine.start()
        self.vision.start()

    def quit(self) -> None:
        if self.recorder is not None:
            self.recorder.stop()
        pos = self.panel.pos()
        self.settings.panel_pos = [pos.x(), pos.y()]
        save_settings(self.settings)
        self.vision.stop()
        if self.engine is not None:
            self.engine.close()
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

    def _new_position(self, reading: PositionReading) -> None:
        self.reading_fen = reading.fen
        self.turn_assumed = reading.turn_assumed
        self._analyse(reading.board.copy(stack=False))

    def _analyse(self, board: chess.Board) -> None:
        self.board = board
        self.analysis = None
        self.req_id = None
        self.panel.clear_analysis()
        self.panel.set_turn_assumed(self.turn_assumed)
        if self.engine is None:
            self._status()
            return
        if not board.is_valid() or board.outcome() is not None:
            self.engine.cancel()
            self._status()
            return
        self.req_id = self.engine.analyze(board)
        self._status()

    def _refresh_arrows(self) -> None:
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
        arrows = [Arrow.from_move(line.move, line.rank) for line in a.lines]
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
        self.panel.show_analysis(analysis, self.white_bottom)
        self._refresh_arrows()
        if analysis.final:
            self._status()

    def on_engine_error(self, message: str) -> None:
        log.error("%s", message)
        self.engine_status = html.escape(message)
        self._status()

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
        self.reading_fen = None
        self.board = None
        self.analysis = None
        self.req_id = None
        self.overlay.clear()
        self.panel.clear_analysis()

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
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
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
