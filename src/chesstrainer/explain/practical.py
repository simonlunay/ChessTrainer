"""Which candidate moves to show at a level, and the "practical move" preference.

At Beginner/Novice (``practical_margin_cp`` > 0 in ``levels.yaml``) a simpler candidate may be
shown first when it is at most that many centipawns worse than Stockfish's top move and clearly
simpler; the panel and the explanation then note that the engine's top move is slightly better.

Simplicity is a small, explainable score (python-chess only):
  +3 wins material outright (a capture that wins by static exchange)
  +2 gives checkmate
  +1 castles, or develops a knight/bishop from its starting square in the first 15 moves
  -3 leaves the moved piece where it can be captured with a material loss (a sacrifice)
  -1 non-castling king move while queens are on the board
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import chess

from chesstrainer.engine.analysis import Line
from chesstrainer.explain import board_tools as bt
from chesstrainer.explain.facts import mover_cp
from chesstrainer.levels import Level

_START = {
    chess.KNIGHT: (chess.B1, chess.G1, chess.B8, chess.G8),
    chess.BISHOP: (chess.C1, chess.F1, chess.C8, chess.F8),
}


@dataclass(frozen=True)
class Suggestion:
    #: Lines to draw and list, re-ranked for display (rank 1 = suggested move).
    lines: tuple[Line, ...]
    #: Stockfish's top line (``lines[0]`` unless a practical move was preferred).
    engine_best: Line
    practical: bool

    @property
    def move(self) -> chess.Move:
        return self.lines[0].move

    @property
    def line(self) -> Line:
        return self.lines[0]


def simplicity(board: chess.Board, line: Line) -> int:
    move = line.move
    piece = board.piece_at(move.from_square)
    if piece is None:
        return 0
    score = 0
    after = board.copy(stack=False)
    after.push(move)
    if after.is_checkmate():
        score += 2
    if board.is_capture(move) and bt.see(board, move) > 0:
        score += 3
    if board.is_castling(move):
        score += 1
    elif piece.piece_type in _START and move.from_square in _START[piece.piece_type]:
        score += 1 if board.fullmove_number <= 15 else 0
    if (
        piece.piece_type != chess.KING
        and not after.is_checkmate()
        and bt.en_prise(after, move.to_square)
    ):
        score -= 3
    if (
        piece.piece_type == chess.KING
        and not board.is_castling(move)
        and board.pieces(chess.QUEEN, chess.WHITE)
        and board.pieces(chess.QUEEN, chess.BLACK)
    ):
        score -= 1
    return score


def suggest(board: chess.Board, lines: tuple[Line, ...], level: Level) -> Suggestion:
    """The lines to show at ``level`` (``level.candidates`` of them), practical move first."""
    best = lines[0]
    chosen = best
    if level.practical_margin_cp > 0 and len(lines) > 1 and best.mate is None:
        me = board.turn
        best_cp = mover_cp(best, me)
        best_simple = simplicity(board, best)
        top = best_simple
        for alt in lines[1:]:
            if alt.mate is not None and mover_cp(alt, me) < 0:
                continue
            if best_cp - mover_cp(alt, me) > level.practical_margin_cp:
                continue
            s = simplicity(board, alt)
            if s >= best_simple + level.practical_min_gain and s > top:
                chosen, top = alt, s
    ordered = [chosen] + [ln for ln in lines if ln is not chosen]
    shown = tuple(replace(ln, rank=i + 1) for i, ln in enumerate(ordered[: level.candidates]))
    return Suggestion(shown, best, chosen is not best)
