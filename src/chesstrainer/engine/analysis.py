"""Background Stockfish analysis with MultiPV and cancellation of stale requests.

:class:`AnalysisEngine` owns one Stockfish process and one worker thread. ``analyze(board)``
returns a request id and immediately stops whatever is running, so only the newest position is
ever searched. Results arrive through the ``on_update`` callback (called on the worker thread)
each time a full set of MultiPV lines completes at a new depth, throttled, plus once more with
``final=True`` when the search ends without being superseded. Callers drop updates whose
``request_id`` is not their latest.
"""

from __future__ import annotations

import logging
import math
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import chess.engine

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Line:
    rank: int  # 1 = engine's best
    pv: tuple[chess.Move, ...]
    cp: int | None  # centipawns from White's point of view (None when a mate is found)
    mate: int | None  # mate in N from White's point of view (negative: Black mates)

    @property
    def move(self) -> chess.Move:
        return self.pv[0]

    def score_text(self) -> str:
        if self.mate is not None:
            return f"#{self.mate}" if self.mate > 0 else f"#-{-self.mate}"
        cp = self.cp or 0
        return f"{cp / 100:+.2f}"

    def white_win_fraction(self) -> float:
        """0..1 for the eval bar (Lichess's win-percentage curve)."""
        if self.mate is not None:
            return 1.0 if self.mate > 0 else 0.0
        return 1.0 / (1.0 + math.exp(-0.00368208 * (self.cp or 0)))


@dataclass(frozen=True)
class Analysis:
    request_id: int
    fen: str
    depth: int
    lines: tuple[Line, ...]
    elapsed: float  # seconds since the request
    final: bool = False

    @property
    def best(self) -> Line:
        return self.lines[0]


class LineCollector:
    """Turns a stream of UCI ``info`` dicts into complete MultiPV snapshots (one per depth)."""

    def __init__(self, board: chess.Board, multipv: int) -> None:
        self.expected = max(1, min(multipv, board.legal_moves.count()))
        self._lines: dict[int, tuple[int, Line]] = {}  # multipv index -> (depth, line)
        self.complete: tuple[int, tuple[Line, ...]] | None = None

    def add(self, info: Mapping[str, Any]) -> tuple[int, tuple[Line, ...]] | None:
        """Feed one info dict. Returns ``(depth, lines)`` when it completes a snapshot."""
        pv = info.get("pv")
        score = info.get("score")
        depth = info.get("depth")
        if not pv or score is None or depth is None:
            return None
        if info.get("lowerbound") or info.get("upperbound"):
            return None  # aspiration-window bound, not an exact score
        k = int(info.get("multipv", 1))
        if k > self.expected:
            return None
        white = score.white()
        line = Line(k, tuple(pv), white.score(), white.mate())
        self._lines[k] = (int(depth), line)
        if len(self._lines) < self.expected:
            return None
        depths = {d for d, _ in self._lines.values()}
        if k != self.expected or len(depths) != 1:
            return None
        lines = tuple(self._lines[i][1] for i in range(1, self.expected + 1))
        self.complete = (int(depth), lines)
        return self.complete


class AnalysisEngine:
    def __init__(
        self,
        path: Path,
        on_update: Callable[[Analysis], None],
        on_error: Callable[[str], None] | None = None,
        *,
        threads: int = 4,
        hash_mb: int = 128,
        multipv: int = 3,
        max_seconds: float = 8.0,
        min_depth: int = 8,
        throttle_s: float = 0.1,
    ) -> None:
        self.path = path
        self.on_update = on_update
        self.on_error = on_error or (lambda msg: log.error("%s", msg))
        self.threads = threads
        self.hash_mb = hash_mb
        self.multipv = multipv
        self.max_seconds = max_seconds
        self.min_depth = min_depth
        self.throttle_s = throttle_s
        self.name = ""
        self._engine: chess.engine.SimpleEngine | None = None
        self._cond = threading.Condition()
        self._pending: tuple[int, chess.Board, float, tuple[chess.Move, ...]] | None = None
        self._active: chess.engine.SimpleAnalysisResult | None = None
        self._next_id = 0
        self._closing = False
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="stockfish", daemon=True)

    # -- public API (any thread) -----------------------------------------------------------

    def start(self) -> None:
        self._thread.start()

    def wait_ready(self, timeout: float = 10.0) -> bool:
        return self._ready.wait(timeout) and self._engine is not None

    def analyze(self, board: chess.Board, root_moves: tuple[chess.Move, ...] = ()) -> int:
        """Start analysing ``board`` (stopping any running search). Returns the request id.

        ``root_moves`` restricts the search to those moves (the opening trainer uses it to
        evaluate the book move, which need not be among Stockfish's top lines)."""
        with self._cond:
            self._next_id += 1
            pending = (self._next_id, board.copy(stack=False), time.perf_counter(), root_moves)
            self._pending = pending
            if self._active is not None:
                self._active.stop()
            self._cond.notify()
            return self._next_id

    def cancel(self) -> None:
        with self._cond:
            self._pending = None
            if self._active is not None:
                self._active.stop()

    def close(self) -> None:
        with self._cond:
            self._closing = True
            self._pending = None
            if self._active is not None:
                self._active.stop()
            self._cond.notify()
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    # -- worker thread ---------------------------------------------------------------------

    def _open(self) -> None:
        kwargs: dict[str, Any] = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        engine = chess.engine.SimpleEngine.popen_uci(str(self.path), **kwargs)
        options: dict[str, chess.engine.ConfigValue] = {}
        if "Threads" in engine.options:
            options["Threads"] = self.threads
        if "Hash" in engine.options:
            options["Hash"] = self.hash_mb
        engine.configure(options)
        self.name = str(engine.id.get("name", self.path.name))
        self._engine = engine

    def _run(self) -> None:
        try:
            self._open()
        except Exception as err:  # noqa: BLE001 - any failure to launch is reported, not raised
            self.on_error(f"Could not start Stockfish ({self.path}): {err}")
            return
        finally:
            self._ready.set()
        while True:
            with self._cond:
                while self._pending is None and not self._closing:
                    self._cond.wait()
                if self._closing:
                    break
                assert self._pending is not None
                req_id, board, t_req, root_moves = self._pending
                self._pending = None
            try:
                self._search(req_id, board, t_req, root_moves)
            except (chess.engine.EngineError, chess.engine.EngineTerminatedError) as err:
                self.on_error(f"Stockfish stopped unexpectedly ({err}); restarting it.")
                with self._cond:
                    self._active = None
                try:
                    self._open()
                except Exception as err2:  # noqa: BLE001
                    self.on_error(f"Could not restart Stockfish: {err2}")
                    return
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:  # noqa: BLE001 - shutting down anyway
                self._engine.close()

    def _search(
        self,
        req_id: int,
        board: chess.Board,
        t_req: float,
        root_moves: tuple[chess.Move, ...] = (),
    ) -> None:
        assert self._engine is not None
        multipv = min(self.multipv, len(root_moves)) if root_moves else self.multipv
        collector = LineCollector(board, multipv)
        fen = board.fen()
        last_emit = -math.inf
        last_best: chess.Move | None = None
        limit = chess.engine.Limit(time=self.max_seconds)
        roots = list(root_moves) if root_moves else None
        analysis = self._engine.analysis(board, limit, multipv=collector.expected, root_moves=roots)
        with analysis as result:
            with self._cond:
                self._active = result
                if self._pending is not None or self._closing:
                    result.stop()  # superseded before it started
            for info in result:
                snap = collector.add(info)
                if snap is None or snap[0] < self.min_depth:
                    continue
                depth, lines = snap
                now = time.perf_counter()
                if lines[0].move != last_best or now - last_emit >= self.throttle_s:
                    last_emit, last_best = now, lines[0].move
                    self.on_update(Analysis(req_id, fen, depth, lines, now - t_req))
        with self._cond:
            self._active = None
            superseded = self._pending is not None or self._closing
        if not superseded and collector.complete is not None:
            depth, lines = collector.complete
            elapsed = time.perf_counter() - t_req
            self.on_update(Analysis(req_id, fen, depth, lines, elapsed, final=True))
