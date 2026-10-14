"""Engine pass over a game: evaluation before and after every move, the player's mistakes.

Each position of the game is searched once (Stockfish, fixed depth with a time cap); the search
of the position after a move doubles as the evaluation of that move. A move's loss is the
mover's evaluation of the best move minus that of the move played, in centipawns, with both
scores clamped to +-1000 so that a won position thrown from +15 to +9 does not count as a
"600 cp blunder". The player's moves are graded with ``Level.classify_loss`` (``levels.yaml``).
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import chess
import chess.engine

from chesstrainer.engine.analysis import Line
from chesstrainer.explain.facts import mover_cp
from chesstrainer.levels import Level, Severity
from chesstrainer.review.games import GameRecord
from chesstrainer.review.mistakes import Mistake, classify_mistake

CLAMP_CP = 1000


class Evaluator(Protocol):
    def evaluate(self, board: chess.Board) -> Line: ...

    def close(self) -> None: ...


class StockfishEvaluator:
    """A separate Stockfish process for reviews (the live overlay keeps its own)."""

    def __init__(
        self, path: Path, depth: int = 14, max_seconds: float = 0.5, threads: int = 4
    ) -> None:
        kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
        self.engine = chess.engine.SimpleEngine.popen_uci(str(path), **kwargs)  # type: ignore[arg-type]
        if "Threads" in self.engine.options:
            self.engine.configure({"Threads": threads})
        self.limit = chess.engine.Limit(depth=depth, time=max_seconds)

    def evaluate(self, board: chess.Board) -> Line:
        info = self.engine.analyse(board, self.limit)
        white = info["score"].white()
        return Line(1, tuple(info.get("pv", ())), white.score(), white.mate())

    def close(self) -> None:
        try:
            self.engine.quit()
        except Exception:  # noqa: BLE001 - shutting down anyway
            self.engine.close()


def score_for(board: chess.Board, line: Line | None, color: chess.Color) -> int:
    """Clamped centipawns for ``color`` in ``board`` (game over: mate or draw)."""
    if line is None:
        if board.is_checkmate():
            return CLAMP_CP if board.turn != color else -CLAMP_CP
        return 0
    return max(-CLAMP_CP, min(CLAMP_CP, mover_cp(line, color)))


@dataclass
class MoveReview:
    ply: int  # 0 = White's first move
    fen_before: str
    move: chess.Move
    san: str
    mover: chess.Color
    best: Line | None  # engine line in the position before the move (None: game over)
    after: Line | None  # engine line in the position after it (None: game over)
    loss: int  # clamped centipawns, >= 0
    by_user: bool
    severity: Severity | None = None
    mistake: Mistake | None = None

    @property
    def move_number(self) -> str:
        n = self.ply // 2 + 1
        return f"{n}." if self.mover == chess.WHITE else f"{n}..."

    @property
    def best_san(self) -> str:
        if self.best is None or not self.best.pv:
            return ""
        return chess.Board(self.fen_before).san(self.best.move)

    @property
    def is_best(self) -> bool:
        return self.best is not None and bool(self.best.pv) and self.best.move == self.move


@dataclass
class GameReview:
    game: GameRecord
    user_color: chess.Color
    level_key: str
    moves: list[MoveReview] = field(default_factory=list)
    #: Evaluation (White's view, clamped cp) of each position: index i = before ply i.
    evals: list[int] = field(default_factory=list)

    def user_moves(self) -> list[MoveReview]:
        return [m for m in self.moves if m.by_user]

    def flagged(self) -> list[MoveReview]:
        return [m for m in self.moves if m.by_user and m.severity is not None]

    def biggest(self, n: int) -> list[MoveReview]:
        return sorted(self.flagged(), key=lambda m: (-m.loss, m.ply))[:n]

    def average_loss(self) -> float:
        mine = self.user_moves()
        return sum(m.loss for m in mine) / len(mine) if mine else 0.0


ProgressFn = Callable[[int, int], None]


def review_game(
    game: GameRecord,
    user_color: chess.Color,
    level: Level,
    evaluator: Evaluator,
    progress: ProgressFn | None = None,
    should_stop: Callable[[], bool] = lambda: False,
) -> GameReview:
    board = game.board()
    boards = [board.copy(stack=False)]
    for mv in game.moves:
        board.push(mv)
        boards.append(board.copy(stack=False))
    lines: list[Line | None] = []
    for i, b in enumerate(boards):
        if should_stop():
            raise InterruptedError
        lines.append(None if b.is_game_over() else evaluator.evaluate(b))
        if progress is not None:
            progress(i + 1, len(boards))
    review = GameReview(game, user_color, level.key)
    review.evals = [score_for(b, ln, chess.WHITE) for b, ln in zip(boards, lines, strict=True)]
    for ply, mv in enumerate(game.moves):
        before, after = boards[ply], boards[ply + 1]
        mover = before.turn
        best, reply = lines[ply], lines[ply + 1]
        if best is not None and best.pv and best.move == mv:
            loss = 0
        else:
            loss = max(0, score_for(before, best, mover) - score_for(after, reply, mover))
        m = MoveReview(
            ply, before.fen(), mv, before.san(mv), mover, best, reply, loss, mover == user_color
        )
        if m.by_user:
            m.severity = level.classify_loss(loss)
            if m.severity is not None and best is not None and best.pv:
                m.mistake = classify_mistake(before, mv, best, reply)
        review.moves.append(m)
    return review
