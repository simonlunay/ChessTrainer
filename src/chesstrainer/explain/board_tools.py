"""Small python-chess helpers shared by the fact extractor and the practical-move chooser."""

from __future__ import annotations

import chess

VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}
PIECE_ORDER = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN)
_DIAG = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_ORTH = ((1, 0), (-1, 0), (0, 1), (0, -1))


def side(color: chess.Color) -> str:
    return "White" if color == chess.WHITE else "Black"


def piece_name(pt: chess.PieceType) -> str:
    return chess.piece_name(pt)


def describe(board: chess.Board, sq: chess.Square) -> str:
    """``"Black's knight on c6"`` (empty square: ``"the empty square c6"``)."""
    p = board.piece_at(sq)
    if p is None:
        return f"the empty square {chess.square_name(sq)}"
    return f"{side(p.color)}'s {piece_name(p.piece_type)} on {chess.square_name(sq)}"


def join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def material(board: chess.Board, color: chess.Color) -> int:
    return sum(VALUES[pt] * len(board.pieces(pt, color)) for pt in PIECE_ORDER if pt != chess.KING)


def material_diff(board: chess.Board) -> int:
    """White minus Black, in pawns."""
    return material(board, chess.WHITE) - material(board, chess.BLACK)


def with_turn(board: chess.Board, color: chess.Color) -> chess.Board | None:
    """A copy with ``color`` to move ("null move"); None when that is not a legal position."""
    b = board.copy(stack=False)
    if b.turn != color:
        if b.is_check():
            return None  # the side in check cannot pass
        b.turn = color
        b.ep_square = None
    return b if b.is_valid() else None


def _least_valuable_capture(board: chess.Board, sq: chess.Square) -> chess.Move | None:
    best: chess.Move | None = None
    best_val = 1000
    for m in board.legal_moves:
        if m.to_square != sq or not board.is_capture(m):
            continue
        p = board.piece_at(m.from_square)
        assert p is not None
        v = VALUES[p.piece_type]
        if m.promotion is not None and m.promotion != chess.QUEEN:
            continue
        if v < best_val:
            best, best_val = m, v
    return best


def _swap(board: chess.Board, sq: chess.Square, depth: int = 0) -> int:
    """Gain for the side to move from the best capture sequence on ``sq`` (0 = no capture)."""
    if depth > 12:
        return 0
    m = _least_valuable_capture(board, sq)
    if m is None:
        return 0
    target = board.piece_at(sq)
    captured = VALUES[target.piece_type] if target is not None else 0
    promo = VALUES[chess.QUEEN] - 1 if m.promotion else 0
    board.push(m)
    gain = captured + promo - _swap(board, sq, depth + 1)
    board.pop()
    return max(0, gain)


def see(board: chess.Board, move: chess.Move) -> int:
    """Static exchange evaluation of ``move`` (pawn units, mover's view), legal recaptures only."""
    target = board.piece_at(move.to_square)
    if board.is_en_passant(move):
        captured = 1
    else:
        captured = VALUES[target.piece_type] if target is not None else 0
    promo = VALUES[chess.QUEEN] - 1 if move.promotion == chess.QUEEN else 0
    b = board.copy(stack=False)
    b.push(move)
    return captured + promo - _swap(b, move.to_square)


def winning_captures(board: chess.Board) -> list[tuple[chess.Move, int]]:
    """Captures for the side to move that win material by static exchange, best first."""
    out = []
    for m in board.legal_moves:
        if board.is_capture(m):
            g = see(board, m)
            if g > 0:
                out.append((m, g))
    out.sort(key=lambda mg: -mg[1])
    return out


def en_prise(board: chess.Board, sq: chess.Square) -> list[chess.Move]:
    """Opponent captures of the piece on ``sq`` that win material (empty if it is safe)."""
    p = board.piece_at(sq)
    if p is None or p.piece_type == chess.KING:
        return []
    b = with_turn(board, not p.color)
    if b is None:
        return []
    return [m for m in b.legal_moves if m.to_square == sq and see(b, m) > 0]


def defenders(board: chess.Board, sq: chess.Square) -> chess.SquareSet:
    p = board.piece_at(sq)
    if p is None:
        return chess.SquareSet()
    return board.attackers(p.color, sq)


def rays(board: chess.Board, sq: chess.Square) -> list[list[chess.Square]]:
    """Occupied squares along each line the slider on ``sq`` moves on, nearest first."""
    p = board.piece_at(sq)
    if p is None:
        return []
    dirs: tuple[tuple[int, int], ...]
    if p.piece_type == chess.BISHOP:
        dirs = _DIAG
    elif p.piece_type == chess.ROOK:
        dirs = _ORTH
    elif p.piece_type == chess.QUEEN:
        dirs = _DIAG + _ORTH
    else:
        return []
    out = []
    for df, dr in dirs:
        f, r = chess.square_file(sq) + df, chess.square_rank(sq) + dr
        line = []
        while 0 <= f < 8 and 0 <= r < 8:
            s = chess.square(f, r)
            if board.piece_at(s) is not None:
                line.append(s)
            f, r = f + df, r + dr
        out.append(line)
    return out


def is_passed(board: chess.Board, sq: chess.Square) -> bool:
    p = board.piece_at(sq)
    if p is None or p.piece_type != chess.PAWN:
        return False
    f, r = chess.square_file(sq), chess.square_rank(sq)
    ahead = range(r + 1, 8) if p.color == chess.WHITE else range(r - 1, -1, -1)
    for ff in (f - 1, f, f + 1):
        if not 0 <= ff < 8:
            continue
        for rr in ahead:
            q = board.piece_at(chess.square(ff, rr))
            if q is not None and q.piece_type == chess.PAWN and q.color != p.color:
                return False
    return True


def pawn_files(board: chess.Board, color: chess.Color) -> list[int]:
    return [chess.square_file(s) for s in board.pieces(chess.PAWN, color)]


def doubled_files(board: chess.Board, color: chess.Color) -> set[int]:
    files = pawn_files(board, color)
    return {f for f in files if files.count(f) > 1}


def isolated_pawns(board: chess.Board, color: chess.Color) -> set[chess.Square]:
    files = set(pawn_files(board, color))
    return {
        s
        for s in board.pieces(chess.PAWN, color)
        if chess.square_file(s) - 1 not in files and chess.square_file(s) + 1 not in files
    }
