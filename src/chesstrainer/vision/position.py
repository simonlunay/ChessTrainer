"""Turn per-square classifier probabilities into a full chess position.

Three layers:

* :func:`repair` enforces hard constraints on a single frame. Two kings of one colour: the more
  confident one stays and every other one is reclassified to its next most likely class. No king:
  the most king-like square becomes the king. Pawns on the back ranks and a tenth pawn of one
  colour are reclassified the same way.
* :class:`PositionTracker` explains each new frame with the game so far. It tries the current
  position, every legal move, every two-move sequence (both players moved between frames) and
  earlier positions of the game (takebacks / stepping through moves). If one of these explains
  the probabilities almost as well as the raw per-square argmax, it wins. That fixes
  low-confidence squares and gives an exact side to move, castling rights and en passant square.
  Otherwise the frame is a new position, and its state is inferred: a king in check must be on
  the move; otherwise the last-move highlight tells who just moved; otherwise the side at the
  bottom is assumed to move. Castling is granted when king and rook stand on their home squares.
* Every change must be seen on ``stable_frames`` consecutive frames, so half-animated pieces and
  drags never get committed.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Literal

import chess
import numpy as np
from numpy.typing import NDArray

from chesstrainer.vision.geometry import cell_to_square
from chesstrainer.vision.labels import (
    CLASS_NAMES,
    EMPTY,
    NUM_CLASSES,
    label_to_piece,
    piece_to_label,
)

Probs = NDArray[np.float32]
Labels = NDArray[np.int64]

_K = {chess.WHITE: CLASS_NAMES.index("K"), chess.BLACK: CLASS_NAMES.index("k")}
_P = {chess.WHITE: CLASS_NAMES.index("P"), chess.BLACK: CLASS_NAMES.index("p")}
_BACK_RANKS = [s for s in chess.SQUARES if chess.square_rank(s) in (0, 7)]


def cells_to_squares(cell_probs: Probs, white_bottom: bool) -> Probs:
    """Reorder (64, 13) probabilities from screen-cell order to square order (a1=0 ... h8=63)."""
    out = np.empty_like(cell_probs)
    for cell in range(64):
        out[cell_to_square(cell, white_bottom)] = cell_probs[cell]
    return out


def squares_to_cells(squares: list[chess.Square], white_bottom: bool) -> list[int]:
    lookup = {cell_to_square(c, white_bottom): c for c in range(64)}
    return [lookup[s] for s in squares]


def labels_of(board: chess.Board) -> Labels:
    labels = np.zeros(64, dtype=np.int64)
    for sq, piece in board.piece_map().items():
        labels[sq] = piece_to_label(piece)
    return labels


def board_from_labels(labels: Labels) -> chess.Board:
    board = chess.Board(None)
    for sq in range(64):
        piece = label_to_piece(int(labels[sq]))
        if piece is not None:
            board.set_piece_at(sq, piece)
    return board


def log_likelihood(logp: NDArray[np.float64], labels: Labels) -> float:
    return float(logp[np.arange(64), labels].sum())


def _next_best(p: Probs, exclude: set[int]) -> int:
    masked = p.astype(np.float64).copy()
    masked[list(exclude)] = -1.0
    return int(masked.argmax())


def repair(probs: Probs) -> tuple[Labels, list[str]]:
    """Argmax labels per square, then fix impossible placements. ``probs`` is in square order."""
    labels = probs.argmax(axis=1).astype(np.int64)
    notes: list[str] = []
    banned: dict[int, set[int]] = {sq: set() for sq in range(64)}

    def reassign(sq: int, why: str) -> None:
        old = labels[sq]
        banned[sq].add(int(old))
        if sq in _BACK_RANKS:
            banned[sq] |= {_P[chess.WHITE], _P[chess.BLACK]}
        labels[sq] = _next_best(probs[sq], banned[sq])
        notes.append(
            f"{why}: {chess.square_name(sq)} {CLASS_NAMES[old]}->{CLASS_NAMES[labels[sq]]} "
            f"(p={probs[sq, labels[sq]]:.2f})"
        )

    for _ in range(12):  # each pass fixes something or we stop
        changed = False
        for sq in _BACK_RANKS:
            if labels[sq] in (_P[chess.WHITE], _P[chess.BLACK]):
                reassign(sq, "pawn on back rank")
                changed = True
        for color in (chess.WHITE, chess.BLACK):
            name = "white" if color else "black"
            kings = [int(s) for s in np.flatnonzero(labels == _K[color])]
            if len(kings) > 1:
                keep = max(kings, key=lambda s: probs[s, _K[color]])
                for sq in kings:
                    if sq != keep:
                        reassign(sq, f"two {name} kings, kept {chess.square_name(keep)}")
                changed = True
            elif not kings:
                p_king = probs[:, _K[color]].astype(np.float64).copy()
                p_king[(labels == _K[not color])] = -1.0
                p_king[[s for s in range(64) if _K[color] in banned[s]]] = -1.0
                sq = int(p_king.argmax())
                if p_king[sq] > 0.01:
                    old = labels[sq]
                    labels[sq] = _K[color]
                    notes.append(
                        f"no {name} king: {chess.square_name(sq)} {CLASS_NAMES[old]}->"
                        f"{CLASS_NAMES[labels[sq]]} (p={p_king[sq]:.2f})"
                    )
                    changed = True
            pawns = [int(s) for s in np.flatnonzero(labels == _P[color])]
            if len(pawns) > 8:
                for sq in sorted(pawns, key=lambda s: probs[s, _P[color]])[: len(pawns) - 8]:
                    reassign(sq, f"more than 8 {name} pawns")
                changed = True
        if not changed:
            break
    return labels, notes


@dataclass
class PositionReading:
    board: chess.Board
    labels: Labels  # square order
    confidence: float  # lowest probability among the chosen square labels
    source: Literal["history", "fresh"]
    moves: list[chess.Move] = field(default_factory=list)  # moves inferred since previous reading
    turn_assumed: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def fen(self) -> str:
        return self.board.fen()

    @property
    def valid(self) -> bool:
        return self.board.is_valid()


def infer_state(
    labels: Labels,
    white_bottom: bool,
    highlight_squares: list[chess.Square] | None = None,
) -> tuple[chess.Board, bool, list[str]]:
    """Side to move, castling rights and en passant for a position seen without history.

    Returns (board, turn_assumed, notes).
    """
    board = board_from_labels(labels)
    notes: list[str] = []
    rights = ""
    for color, k_sq, rooks in (
        (chess.WHITE, chess.E1, ((chess.H1, "K"), (chess.A1, "Q"))),
        (chess.BLACK, chess.E8, ((chess.H8, "k"), (chess.A8, "q"))),
    ):
        if board.piece_at(k_sq) == chess.Piece(chess.KING, color):
            for r_sq, flag in rooks:
                if board.piece_at(r_sq) == chess.Piece(chess.ROOK, color):
                    rights += flag
    board.set_castling_fen(rights or "-")

    in_check = []
    for color in (chess.WHITE, chess.BLACK):
        king = board.king(color)
        if king is not None and board.is_attacked_by(not color, king):
            in_check.append(color)
    if len(in_check) == 1:
        board.turn = in_check[0]
        notes.append("side to move: the side in check")
        return board, False, notes

    if highlight_squares and len(highlight_squares) == 2:
        occupied = [s for s in highlight_squares if board.piece_at(s) is not None]
        if len(occupied) == 1:
            to_sq = occupied[0]
            from_sq = next(s for s in highlight_squares if s != to_sq)
            mover = board.piece_at(to_sq)
            assert mover is not None
            board.turn = not mover.color
            last = chess.square_name(from_sq) + chess.square_name(to_sq)
            notes.append(f"side to move: last-move highlight {last}")
            if (
                mover.piece_type == chess.PAWN
                and chess.square_file(from_sq) == chess.square_file(to_sq)
                and abs(chess.square_rank(from_sq) - chess.square_rank(to_sq)) == 2
            ):
                board.ep_square = (from_sq + to_sq) // 2
            return board, False, notes

    board.turn = chess.WHITE if white_bottom else chess.BLACK
    notes.append("side to move: assumed (the side at the bottom)")
    return board, True, notes


class PositionTracker:
    def __init__(
        self, accept_margin: float = 3.0, stable_frames: int = 2, history_size: int = 400
    ) -> None:
        self.accept_margin = accept_margin
        self.stable_frames = stable_frames
        self.history_size = history_size
        self.current: PositionReading | None = None
        self._seen: OrderedDict[bytes, chess.Board] = OrderedDict()
        self._pending_key: bytes | None = None
        self._pending_count = 0

    def reset(self) -> None:
        self.__init__(self.accept_margin, self.stable_frames, self.history_size)  # type: ignore[misc]

    # -- candidates -------------------------------------------------------------------------

    def _move_candidates(
        self, logp: NDArray[np.float64], bar: float, two_ply: bool
    ) -> tuple[float, chess.Board, list[chess.Move]] | None:
        assert self.current is not None
        roots = [self.current.board]
        if self.current.turn_assumed:  # the guess may be wrong: also try the other side moving
            flipped = self.current.board.copy(stack=False)
            flipped.turn = not flipped.turn
            flipped.ep_square = None
            roots.append(flipped)
        best: tuple[float, chess.Board, list[chess.Move]] | None = None
        for root in roots:
            for m in root.legal_moves:
                b = root.copy()
                b.push(m)
                ll = log_likelihood(logp, labels_of(b))
                if best is None or ll > best[0]:
                    best = (ll, b, [m])
        if not two_ply or (best is not None and best[0] >= bar):
            return best
        for root in roots:  # both sides moved between two frames
            for m1 in root.legal_moves:
                b1 = root.copy()
                b1.push(m1)
                for m2 in b1.legal_moves:
                    b2 = b1.copy()
                    b2.push(m2)
                    ll = log_likelihood(logp, labels_of(b2))
                    if best is None or ll > best[0]:
                        best = (ll, b2, [m1, m2])
        return best

    # -- main entry -------------------------------------------------------------------------

    def update(
        self,
        square_probs: Probs,
        white_bottom: bool,
        highlight_squares: list[chess.Square] | None = None,
    ) -> PositionReading | None:
        """Feed one frame. Returns a new reading when the committed position changes, else None."""
        logp = np.log(np.clip(square_probs.astype(np.float64), 1e-6, 1.0))
        raw, notes = repair(square_probs)
        ll_raw = log_likelihood(logp, raw)
        bar = ll_raw - self.accept_margin

        if self.current is not None and np.array_equal(raw, self.current.labels):
            self._pending_key = None
            return None

        cand: tuple[str, chess.Board | None, list[chess.Move], Labels] = ("fresh", None, [], raw)
        if self.current is not None:
            ll_cur = log_likelihood(logp, self.current.labels)
            still = ll_cur >= bar  # the committed position still explains the frame
            found = None
            if self.current.valid:
                found = self._move_candidates(logp, bar, two_ply=not still)
            if still and (found is None or found[0] <= ll_cur + 1.0):
                self._pending_key = None  # only noise changed
                return None
            if found is not None and found[0] >= bar:
                cand = ("history", found[1], found[2], labels_of(found[1]))
        if cand[0] == "fresh":
            past = self._seen.get(raw.tobytes())
            if past is not None:
                cand = ("history", past.copy(), [], raw)

        key = cand[3].tobytes() + str(cand[2]).encode()
        if key != self._pending_key:
            self._pending_key, self._pending_count = key, 0
        self._pending_count += 1
        if self._pending_count < self.stable_frames:
            return None
        self._pending_key = None

        source, board, moves, labels = cand
        turn_assumed = False
        if board is None:
            hl = highlight_squares
            board, turn_assumed, state_notes = infer_state(labels, white_bottom, hl)
            notes += state_notes
            self._seen.clear()
        chosen = square_probs[np.arange(64), labels]
        reading = PositionReading(
            board=board,
            labels=labels,
            confidence=float(chosen.min()),
            source="fresh" if source == "fresh" else "history",
            moves=moves,
            turn_assumed=turn_assumed,
            notes=notes,
        )
        self.current = reading
        self._seen[labels.tobytes()] = board.copy()
        self._seen.move_to_end(labels.tobytes())
        while len(self._seen) > self.history_size:
            self._seen.popitem(last=False)
        return reading


__all__ = [
    "EMPTY",
    "NUM_CLASSES",
    "PositionReading",
    "PositionTracker",
    "board_from_labels",
    "cells_to_squares",
    "infer_state",
    "labels_of",
    "repair",
    "squares_to_cells",
]
