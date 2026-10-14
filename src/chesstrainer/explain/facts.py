"""Facts about a move, computed with python-chess and Stockfish's lines (no LLM involved).

These facts are the only chess content the explanation model receives, and the evaluation
checks every explanation against them. Each :class:`Fact` has a ``tag`` (the concept it
grounds: ``fork``, ``pin``, ``check``, ``material_gain``...) and a plain-English ``text``
that names pieces by colour and square ("Black's knight on c6").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import chess

from chesstrainer.engine.analysis import Line
from chesstrainer.explain import board_tools as bt
from chesstrainer.explain.board_tools import VALUES, describe, join, side
from chesstrainer.levels import Level

CENTER = (chess.D4, chess.E4, chess.D5, chess.E5)
_START_SQUARES = {
    (chess.WHITE, chess.KNIGHT): (chess.B1, chess.G1),
    (chess.WHITE, chess.BISHOP): (chess.C1, chess.F1),
    (chess.BLACK, chess.KNIGHT): (chess.B8, chess.G8),
    (chess.BLACK, chess.BISHOP): (chess.C8, chess.F8),
}
SQUARE_RE = re.compile(r"(?<![a-wyz0-9])([a-h][1-8])(?![0-9])")
SAN_RE = re.compile(
    r"(?<![\w-])(O-O-O|O-O|[KQRBN][a-h]?[1-8]?x?[a-h][1-8]|[a-h](?:x[a-h])?[1-8](?:=[QRBN])?)[+#]?"
)


@dataclass(frozen=True)
class Fact:
    tag: str
    text: str


@dataclass
class MoveFacts:
    fen: str
    move: chess.Move
    san: str
    mover: chess.Color
    level_key: str
    facts: list[Fact] = field(default_factory=list)
    #: Extra prompt lines (e.g. the opening trainer's task) and a cache-key discriminator, so
    #: an opening explanation and a plain one of the same move are cached separately.
    instructions: str = ""
    context: str = ""

    @property
    def tags(self) -> set[str]:
        return {f.tag for f in self.facts}

    def text(self) -> str:
        return "\n".join(f"- {f.text}" for f in self.facts)

    def squares(self) -> set[str]:
        return {m for f in self.facts for m in SQUARE_RE.findall(f.text)}

    def sans(self) -> set[str]:
        out = {self.san.rstrip("+#")}
        for f in self.facts:
            out.update(SAN_RE.findall(f.text))
        return out

    def add(self, tag: str, text: str) -> None:
        self.facts.append(Fact(tag, text))


# -- evaluation wording ----------------------------------------------------------------------


def eval_words(line: Line) -> str:
    if line.mate is not None:
        winner = "White" if line.mate > 0 else "Black"
        n = abs(line.mate)
        return f"{winner} can force checkmate in {n} move{'s' if n != 1 else ''}"
    cp = line.cp or 0
    a = abs(cp)
    ahead = "White" if cp > 0 else "Black"
    if a < 40:
        return "the position is about equal"
    if a < 120:
        return f"{ahead} is slightly better"
    if a < 250:
        return f"{ahead} is clearly better"
    if a < 600:
        return f"{ahead} is winning"
    return f"{ahead} is completely winning"


def eval_number(line: Line) -> str:
    if line.mate is not None:
        return f"mate in {abs(line.mate)}"
    cp = line.cp or 0
    return "0.0" if abs(cp) < 5 else f"{cp / 100:+.1f}"


def eval_text(line: Line, numbers: bool) -> str:
    words = eval_words(line)
    if numbers and line.mate is None:
        return f"{words}, {eval_number(line)} in pawns from White's point of view"
    return words


def mover_cp(line: Line, color: chess.Color) -> int:
    """Score for ``color`` in centipawns; mates map to +-(100000 - N)."""
    if line.mate is not None:
        v = 100_000 - abs(line.mate)
        white = v if line.mate > 0 else -v
    else:
        white = line.cp or 0
    return white if color == chess.WHITE else -white


# -- material over a line --------------------------------------------------------------------


_COUNT_WORDS = ("no", "an", "two", "three", "four", "five", "six", "seven", "eight")


def material_text(board: chess.Board) -> str:
    """ "White is ahead by 1 point of material (White has an extra pawn)"."""
    diff = bt.material_diff(board)
    extras: dict[chess.Color, list[str]] = {chess.WHITE: [], chess.BLACK: []}
    for pt in bt.PIECE_ORDER:
        d = len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK))
        if d:
            n = abs(d)
            word = _COUNT_WORDS[n] if n < len(_COUNT_WORDS) else str(n)
            name = bt.piece_name(pt) + ("s" if n > 1 else "")
            extras[chess.WHITE if d > 0 else chess.BLACK].append(f"{word} extra {name}")
    detail = "; ".join(
        f"{side(c)} has {join(extras[c])}" for c in (chess.WHITE, chess.BLACK) if extras[c]
    )
    if diff == 0:
        return f"material is equal ({detail})" if detail else "material is equal"
    ahead = "White" if diff > 0 else "Black"
    n = abs(diff)
    return f"{ahead} is ahead by {n} point{'s' if n != 1 else ''} of material ({detail})"


def _quiet_end(board: chess.Board, pv: tuple[chess.Move, ...], plies: int) -> int:
    """Length of the line to show: ``plies``, extended (by up to 6) until it reaches a quiet
    position - the next move is not a capture and the side to move has no winning capture -
    so trades complete and material claims are not made in the middle of a sequence."""
    n = min(plies, len(pv))
    b = board.copy(stack=False)
    for m in pv[:n]:
        b.push(m)
    while n < len(pv) and n < plies + 6:
        if not b.is_capture(pv[n]) and not bt.winning_captures(b):
            break
        b.push(pv[n])
        n += 1
    return n


def _captures_along(board: chess.Board, moves: tuple[chess.Move, ...]) -> list[str]:
    """Each capture in the line, in order: "White's knight takes the pawn on c7 (Nxc7+)"."""
    out = []
    b = board.copy(stack=False)
    for m in moves:
        if b.is_capture(m):
            sq = m.to_square
            if b.is_en_passant(m):
                sq = chess.square(chess.square_file(m.to_square), chess.square_rank(m.from_square))
            victim, taker = b.piece_at(sq), b.piece_at(m.from_square)
            assert victim is not None and taker is not None
            out.append(
                f"{side(b.turn)}'s {bt.piece_name(taker.piece_type)} takes the "
                f"{bt.piece_name(victim.piece_type)} on {chess.square_name(sq)} ({b.san(m)})"
            )
        b.push(m)
    return out


# -- tactics ---------------------------------------------------------------------------------


def _line_targets(
    board: chess.Board, slider: chess.Square
) -> list[tuple[chess.Square, chess.Square]]:
    """(front, behind) enemy pairs on the slider's lines with nothing in between."""
    p = board.piece_at(slider)
    assert p is not None
    pairs = []
    for line in bt.rays(board, slider):
        if len(line) < 2:
            continue
        a, b = board.piece_at(line[0]), board.piece_at(line[1])
        if a is None or b is None or a.color == p.color or b.color == p.color:
            continue
        pairs.append((line[0], line[1]))
    return pairs


def pins(board: chess.Board, color: chess.Color) -> set[tuple[chess.Square, chess.Square, str]]:
    """Pins by ``color``'s sliders: (pinned square, piece behind, kind) - absolute/relative.

    A relative pin only counts when capturing the piece behind would actually gain: it is worth
    more than both the pinned piece and the pinner, or it is undefended.
    """
    out = set()
    for sq in chess.SquareSet(board.occupied_co[color]):
        if not _is_slider(board, sq):
            continue
        pinner = board.piece_at(sq)
        assert pinner is not None
        for front, behind in _line_targets(board, sq):
            fp, bp = board.piece_at(front), board.piece_at(behind)
            assert fp is not None and bp is not None
            if fp.piece_type == chess.KING:
                continue
            if bp.piece_type == chess.KING:
                out.add((front, behind, "absolute"))
                continue
            worth = VALUES[bp.piece_type] > VALUES[pinner.piece_type] or not bt.defenders(
                board, behind
            )
            if VALUES[bp.piece_type] > VALUES[fp.piece_type] and worth:
                out.add((front, behind, "relative"))
    return out


def skewers(board: chess.Board, color: chess.Color) -> set[tuple[chess.Square, chess.Square]]:
    """Skewers by ``color``: a valuable enemy piece in front of a less valuable one."""
    out = set()
    for sq in chess.SquareSet(board.occupied_co[color]):
        if not _is_slider(board, sq):
            continue
        sp = board.piece_at(sq)
        assert sp is not None
        for front, behind in _line_targets(board, sq):
            fp, bp = board.piece_at(front), board.piece_at(behind)
            assert fp is not None and bp is not None
            if bp.piece_type == chess.PAWN:
                continue
            front_v = VALUES[fp.piece_type]
            if front_v > VALUES[bp.piece_type] and front_v > VALUES[sp.piece_type]:
                out.add((front, behind))
    return out


def _is_slider(board: chess.Board, sq: chess.Square) -> bool:
    p = board.piece_at(sq)
    return p is not None and p.piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN)


def _attacked_enemies(board: chess.Board, sq: chess.Square) -> list[chess.Square]:
    p = board.piece_at(sq)
    assert p is not None
    return [s for s in board.attacks(sq) if (q := board.piece_at(s)) and q.color != p.color]


# -- the extractor ---------------------------------------------------------------------------


@dataclass
class _Ctx:
    board: chess.Board
    after: chess.Board
    line: Line
    level: Level
    mf: MoveFacts

    def __post_init__(self) -> None:
        board, move = self.board, self.line.move
        piece = board.piece_at(move.from_square)
        assert piece is not None
        self.piece = piece
        self.me, self.them = board.turn, not board.turn
        self.pname = bt.piece_name(piece.piece_type)
        self.frm = chess.square_name(move.from_square)
        self.san = self.mf.san
        self.castling = board.is_castling(move)
        self.moved_to = move.to_square
        if self.castling:
            file = 6 if board.is_kingside_castling(move) else 2
            self.moved_to = chess.square(file, chess.square_rank(move.from_square))
        # Where the line is cut: after ``pv_plies``, extended until the captures stop.
        self.line_end = _quiet_end(board, self.line.pv, self.level.explanation.pv_plies)

    @property
    def move(self) -> chess.Move:
        return self.line.move

    @property
    def pv(self) -> tuple[chess.Move, ...]:
        return self.line.pv

    def moved_piece(self) -> chess.Piece | None:
        return self.after.piece_at(self.moved_to)

    def board_after(self, plies: int) -> chess.Board:
        b = self.board.copy(stack=False)
        for m in self.pv[:plies]:
            b.push(m)
        return b

    def add(self, tag: str, text: str) -> None:
        self.mf.add(tag, text)


def compute_facts(
    board: chess.Board,
    line: Line,
    level: Level,
    engine_best: Line | None = None,
    alternatives: tuple[Line, ...] = (),
) -> MoveFacts:
    """Facts about ``line.move`` in ``board``.

    ``engine_best`` is Stockfish's top line when ``line`` is a different (practical) choice;
    ``alternatives`` are other candidate lines worth mentioning (higher levels).
    """
    after = board.copy(stack=False)
    after.push(line.move)
    mf = MoveFacts(board.fen(), line.move, board.san(line.move), board.turn, level.key)
    c = _Ctx(board, after, line, level, mf)
    _what_it_is(c)
    _check_and_mate(c)
    _verdict(c, engine_best, alternatives)
    _material_and_line(c)
    _threats(c)
    _moved_piece(c)
    _discovered(c)
    _pins_and_skewers(c)
    _positional(c)
    _structure(c)
    phase = f"{_phase(board)} (move {board.fullmove_number}, {side(c.me)} to move)"
    c.add("phase", f"Game phase: {phase}.")
    return mf


def _what_it_is(c: _Ctx) -> None:
    board, move, san, me = c.board, c.move, c.san, c.me
    if c.castling:
        wing = "kingside" if board.is_kingside_castling(move) else "queenside"
        rank = "1" if me == chess.WHITE else "8"
        k, r = ("g", "f") if wing == "kingside" else ("c", "d")
        c.add(
            "castle",
            f"{side(me)} plays {san}: castling {wing} (the king goes to {k}{rank} and the rook "
            f"to {r}{rank}). Castling tucks the king away from the centre and brings the rook "
            "into play.",
        )
    elif board.is_capture(move):
        cap_sq = move.to_square
        if board.is_en_passant(move):
            cap_sq = chess.square(chess.square_file(cap_sq), chess.square_rank(move.from_square))
        victim = describe(board, cap_sq)
        how = " en passant" if board.is_en_passant(move) else ""
        c.add(
            "capture", f"{side(me)} plays {san}: the {c.pname} on {c.frm} captures {victim}{how}."
        )
        if board.is_en_passant(move):
            c.add("en_passant", f"{san} is an en passant capture.")
        defs = bt.defenders(board, cap_sq)
        if not defs:
            c.add("hanging", f"{victim} was not defended by any {side(c.them)} piece.")
        else:
            names = join([describe(board, s) for s in defs])
            only = "only " if len(defs) == 1 else ""
            c.add("recapture", f"{victim} was defended {only}by {names}.")
    else:
        to = chess.square_name(move.to_square)
        c.add("move", f"{side(me)} plays {san}: the {c.pname} moves from {c.frm} to {to}.")
    if move.promotion is not None:
        promo = bt.piece_name(move.promotion)
        c.add("promotion", f"The pawn reaches the last rank and promotes to a {promo}.")


def _check_and_mate(c: _Ctx) -> None:
    after, san, them = c.after, c.san, c.them
    if after.is_checkmate():
        c.add(
            "mate",
            f"{san} is checkmate: {side(them)}'s king cannot escape the check. {side(c.me)} wins.",
        )
    elif after.is_check():
        checkers = after.checkers()
        names = join([describe(after, s) for s in checkers])
        kind = "double check" if len(checkers) > 1 else "check"
        c.add("check", f"{san} gives {kind} to {side(them)}'s king (checking piece: {names}).")
        c.add("check", _check_replies(after))
    elif after.is_stalemate():
        c.add("stalemate", f"{san} is stalemate: the game is a draw.")
    if c.board.is_check():
        how = "by moving the king"
        if c.piece.piece_type != chess.KING:
            how = "by capturing the checking piece" if c.board.is_capture(c.move) else "by blocking"
        c.add("evasion", f"{side(c.me)} was in check; {san} gets out of check {how}.")


def _check_replies(after: chess.Board) -> str:
    """How the side in check can answer: king moves, captures of the checker, blocks."""
    kinds: dict[str, list[str]] = {"moving the king": [], "capturing": [], "blocking": []}
    checkers = after.checkers()
    for m in after.legal_moves:
        p = after.piece_at(m.from_square)
        assert p is not None
        if p.piece_type == chess.KING:
            key = "moving the king"
        elif m.to_square in checkers:
            key = "capturing"
        else:
            key = "blocking"
        kinds[key].append(after.san(m))
    parts = [
        f"{k} ({', '.join(v[:3])}{', ...' if len(v) > 3 else ''})" for k, v in kinds.items() if v
    ]
    who = side(after.turn)
    n = after.legal_moves.count()
    return f"{who} has {n} legal repl{'y' if n == 1 else 'ies'} to the check: {join(parts)}."


def _verdict(c: _Ctx, engine_best: Line | None, alternatives: tuple[Line, ...]) -> None:
    numbers = c.level.explanation.eval_numbers
    line, san = c.line, c.san
    if line.mate is not None and not c.after.is_checkmate():
        c.add("mate", f"Stockfish: after {san}, {eval_words(line)} with best play.")
    elif line.mate is None:
        c.add("eval", f"Stockfish evaluation after {san}: {eval_text(line, numbers)}.")
    if engine_best is not None and engine_best.move != c.move:
        c.add(
            "engine_best",
            f"Stockfish's top move is {c.board.san(engine_best.move)}, which is slightly better "
            f"(with it {eval_text(engine_best, numbers)}); {san} is simpler to play.",
        )
    for alt in alternatives:
        if alt.move != c.move and (engine_best is None or alt.move != engine_best.move):
            c.add(
                "alternative",
                f"Another candidate is {c.board.san(alt.move)}; with it {eval_text(alt, numbers)}.",
            )


def _material_and_line(c: _Ctx) -> None:
    board, pv, me = c.board, c.pv, c.me
    c.add("material", f"Before the move, {material_text(board)}.")
    seg = pv[: c.line_end]
    if not seg:
        return
    c.add("line", f"Expected continuation (Stockfish): {board.variation_san(list(seg))}.")
    if len(pv) >= 2:
        c.add("reply", _reply_text(c.after, pv[1]))
    parts = _captures_along(board, seg)
    if parts:
        c.add("captures", f"Captures in that line, in order: {'; '.join(parts)}.")
    sign = 1 if me == chess.WHITE else -1
    end = c.board_after(len(seg))
    gained = (bt.material_diff(end) - bt.material_diff(board)) * sign
    pts = f"{abs(gained)} point{'s' if abs(gained) != 1 else ''} of material"
    where = "At the end of that line"
    balance = f"so overall {material_text(end)}"
    if gained > 0:
        c.add("material_gain", f"{where}, {side(me)} has won {pts}, {balance}.")
    elif gained < 0:
        c.add("material_loss", f"{where}, {side(me)} has given up {pts}, {balance}.")
    elif parts:
        c.add("trade", f"{where}, material is level again (the captures cancel out).")
    else:
        c.add("material", "No material changes hands in that line.")


def _reply_text(after: chess.Board, reply: chess.Move) -> str:
    """The opponent's best reply and what its piece attacks or newly defends."""
    who = side(after.turn)
    text = f"{who}'s best reply is {after.san(reply)}"
    b = after.copy(stack=False)
    b.push(reply)
    sq = reply.to_square
    if b.is_castling(reply) or b.piece_at(sq) is None:
        return text + "."
    attacked = []
    for t in _attacked_enemies(b, sq):
        p = b.piece_at(t)
        assert p is not None
        if p.piece_type == chess.KING or (p.piece_type == chess.PAWN and bt.defenders(b, t)):
            continue  # check is in the notation; defended pawns are not worth listing
        attacked.append(describe(b, t))
    if attacked:
        text += f", which attacks {join(attacked)}"
    return text + "."


def _threats(c: _Ctx) -> None:
    """Winning captures the move takes away from the opponent, and new ones it creates."""
    null_before = bt.with_turn(c.board, c.them)
    if null_before is not None:
        after_threats = {m for m, _ in bt.winning_captures(c.after)}
        stopped = [
            f"{null_before.san(m)} ({describe(c.board, m.from_square)} capturing "
            f"{describe(c.board, m.to_square)})"
            for m, _ in bt.winning_captures(null_before)
            if m not in after_threats
        ]
        if stopped:
            c.add(
                "defends",
                f"Before the move {side(c.them)} threatened {join(stopped[:2])}; after {c.san} "
                "that threat no longer wins material.",
            )
    null_after = bt.with_turn(c.after, c.me)
    if null_after is not None:
        old = {m for m, _ in bt.winning_captures(c.board)}
        new = [m for m, _ in bt.winning_captures(null_after) if m not in old]
        if new:
            items = [
                f"{null_after.san(m)} ({describe(c.after, m.from_square)} capturing "
                f"{describe(c.after, m.to_square)})"
                for m in new[:2]
            ]
            c.add("threat", f"{c.san} threatens {join(items)} on the next move.")


def _moved_piece(c: _Ctx) -> None:
    """What the moved piece attacks from its new square (forks), and whether it is safe there."""
    after, sq = c.after, c.moved_to
    mp = c.moved_piece()
    if mp is None or after.is_checkmate():
        return
    name, where = bt.piece_name(mp.piece_type), chess.square_name(sq)
    targets, fork_targets = [], []
    for s in _attacked_enemies(after, sq):
        tp = after.piece_at(s)
        assert tp is not None
        undefended = not bt.defenders(after, s)
        if tp.piece_type == chess.KING:
            fork_targets.append(s)
            continue
        if tp.piece_type != chess.PAWN or undefended:
            targets.append(s)
        if VALUES[tp.piece_type] > VALUES[mp.piece_type] or undefended:
            fork_targets.append(s)
    if targets:
        names = join([describe(after, s) for s in targets])
        c.add("attack", f"From {where} the {name} attacks {names}.")
    if len(fork_targets) >= 2:
        names = join([describe(after, s) for s in fork_targets])
        c.add("fork", f"This is a fork: the {name} on {where} attacks {names} at the same time.")
    if mp.piece_type == chess.KING:
        return
    caps = bt.en_prise(after, sq)
    if not caps:
        return
    attackers = join([describe(after, m.from_square) for m in caps])
    text = f"After {c.san}, the {name} on {where} can be captured by {attackers}."
    if len(c.pv) >= 2 and c.pv[1].to_square == sq:
        text += f" In Stockfish's line {side(c.them)} does capture it ({after.san(c.pv[1])})."
        c.add("sacrifice", text)
    else:
        c.add("en_prise", text + " Stockfish's line does not take it.")


def _discovered(c: _Ctx) -> None:
    """Lines opened by the moving piece for another of the mover's pieces."""
    after, board = c.after, c.board
    for s in chess.SquareSet(after.occupied_co[c.me]):
        if s == c.moved_to or not _is_slider(after, s):
            continue
        for t in _attacked_enemies(after, s):
            if t in board.attacks(s):
                continue
            if c.move.from_square not in chess.SquareSet(chess.between(s, t)):
                continue
            tp = after.piece_at(t)
            assert tp is not None
            if tp.piece_type == chess.PAWN:
                continue
            kind = "discovered check" if tp.piece_type == chess.KING else "discovered attack"
            c.add(
                "discovered",
                f"Moving the {c.pname} off {c.frm} opens the line of {describe(after, s)}, which "
                f"now attacks {describe(after, t)} (a {kind}).",
            )


def _pins_and_skewers(c: _Ctx) -> None:
    board, after, me, them = c.board, c.after, c.me, c.them
    for front, behind, kind in sorted(pins(after, me) - pins(board, me)):
        pinner = next((s for s in after.attackers(me, front) if _is_slider(after, s)), None)
        by = f" by {describe(after, pinner)}" if pinner is not None else ""
        result = (
            "it cannot legally move"
            if kind == "absolute"
            else "if it moves, the piece behind it can be captured"
        )
        pinned, to = describe(after, front), describe(after, behind)
        c.add("pin", f"{pinned} is pinned{by} to {to}: {result}.")
    for front, _behind, _kind in sorted(pins(board, them) - pins(after, them)):
        p = after.piece_at(front)
        if p is not None and p.color == me:
            c.add("unpin", f"The move breaks the pin on {describe(after, front)}.")
    for front, behind in sorted(skewers(after, me) - skewers(board, me)):
        c.add(
            "skewer",
            f"This is a skewer: {describe(after, front)} is attacked and, when it moves away, "
            f"{describe(after, behind)} behind it can be captured.",
        )


def _positional(c: _Ctx) -> None:
    board, after, move, san, me = c.board, c.after, c.move, c.san, c.me
    pt = c.piece.piece_type
    mp = c.moved_piece()
    where = chess.square_name(c.moved_to)
    if move.from_square in _START_SQUARES.get((me, pt), ()) and board.fullmove_number <= 20:
        c.add(
            "development",
            f"{san} develops the {c.pname}: it leaves its starting square and joins the game.",
        )
    if not c.castling:
        for wing, has in (
            ("kingside", chess.Board.has_kingside_castling_rights),
            ("queenside", chess.Board.has_queenside_castling_rights),
        ):
            if has(board, me) and not has(after, me):
                c.add("castling_rights", f"After {san}, {side(me)} can no longer castle {wing}.")
    if mp is not None and pt != chess.KING:
        center = [chess.square_name(s) for s in CENTER if s in after.attacks(c.moved_to)]
        if c.moved_to in CENTER and mp.piece_type == chess.PAWN:
            c.add("center", f"The pawn now stands in the centre on {where}.")
        elif center:
            plural = "s" if len(center) > 1 else ""
            c.add(
                "center",
                f"From {where} the {bt.piece_name(mp.piece_type)} controls the central "
                f"square{plural} {join(center)}.",
            )
    _open_file(c)
    _opened_lines(c)
    if pt == chess.KING and not c.castling:
        before_d = min(chess.square_distance(move.from_square, s) for s in CENTER)
        after_d = min(chess.square_distance(c.moved_to, s) for s in CENTER)
        if after_d < before_d:
            c.add("center", f"The king moves closer to the centre ({c.frm} to {where}).")
    passed_before = {s for s in board.pieces(chess.PAWN, me) if bt.is_passed(board, s)}
    for s in sorted(after.pieces(chess.PAWN, me)):
        if bt.is_passed(after, s) and (s not in passed_before or s == c.moved_to):
            steps = 7 - chess.square_rank(s) if me == chess.WHITE else chess.square_rank(s)
            was = (
                "was already a passed pawn before this move"
                if s == c.moved_to and move.from_square in passed_before
                else "becomes a passed pawn with this move"
            )
            c.add(
                "passed_pawn",
                f"{side(me)}'s pawn on {chess.square_name(s)} {was} (no {side(c.them)} pawn "
                f"can block or capture it on the way); it is {steps} square"
                f"{'s' if steps != 1 else ''} from promoting.",
            )
    if pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN) and move.promotion is None:
        before_n = len(board.attacks(move.from_square))
        after_n = len(after.attacks(c.moved_to))
        if after_n - before_n >= 3:
            c.add(
                "mobility", f"The {c.pname} now controls {after_n} squares instead of {before_n}."
            )
    king = after.king(c.them)
    if king is not None and mp is not None and pt != chess.KING:
        zone = sorted(s for s in after.attacks(c.moved_to) if chess.square_distance(s, king) == 1)
        if len(zone) >= 2:
            c.add(
                "king_attack",
                f"The {bt.piece_name(mp.piece_type)} now attacks squares next to "
                f"{side(c.them)}'s king: {join([chess.square_name(s) for s in zone])}.",
            )


def _opened_lines(c: _Ctx) -> None:
    """Other pieces of the mover that control more squares because the moved piece got out of
    the way (a pawn move freeing a bishop, for example)."""
    for s in chess.SquareSet(c.after.occupied_co[c.me]):
        if s in (c.moved_to, c.move.from_square) or not _is_slider(c.after, s):
            continue
        if c.board.piece_at(s) != c.after.piece_at(s):
            continue  # not a piece that stood still (e.g. the castled rook)
        gain = len(c.after.attacks(s)) - len(c.board.attacks(s))
        if gain >= 2 and c.move.from_square in c.board.attacks(s):
            c.add(
                "opens_line",
                f"{c.san} opens a line for {describe(c.after, s)}: it now controls {gain} more "
                "squares.",
            )


def _open_file(c: _Ctx) -> None:
    if c.castling:
        rank = chess.square_rank(c.move.from_square)
        sq = chess.square(5 if c.board.is_kingside_castling(c.move) else 3, rank)
    elif c.piece.piece_type in (chess.ROOK, chess.QUEEN):
        sq = c.moved_to
    else:
        return
    p = c.after.piece_at(sq)
    if p is None or p.piece_type not in (chess.ROOK, chess.QUEEN):
        return
    f = chess.square_file(sq)
    own = any(chess.square_file(s) == f for s in c.after.pieces(chess.PAWN, c.me))
    opp = any(chess.square_file(s) == f for s in c.after.pieces(chess.PAWN, c.them))
    if own:
        return
    kind = f"a half-open file (no {side(c.me)} pawns on it)" if opp else "an open file (no pawns)"
    c.add(
        "open_file",
        f"The {bt.piece_name(p.piece_type)} on {chess.square_name(sq)} stands on the "
        f"{chess.FILE_NAMES[f]}-file, {kind}.",
    )


def _structure(c: _Ctx) -> None:
    """Pawn weaknesses and the bishop pair at the end of the shown line.

    Measured where the line is quiet (``line_end``), not half-way through a trade: after
    4. d4 exd4 Black's pawns are doubled, but 5. Nxd4 takes one of them back.
    """
    board = c.board
    end = c.board_after(c.line_end)
    for color in (c.me, c.them):
        for f in sorted(bt.doubled_files(end, color) - bt.doubled_files(board, color)):
            c.add(
                "doubled",
                f"At the end of that line {side(color)} has doubled pawns on the "
                f"{chess.FILE_NAMES[f]}-file.",
            )
        for s in sorted(bt.isolated_pawns(end, color) - bt.isolated_pawns(board, color)):
            c.add(
                "isolated",
                f"At the end of that line {side(color)}'s pawn on {chess.square_name(s)} is "
                "isolated (no pawns of the same colour on the files next to it).",
            )
    for color in (c.me, c.them):
        lost = (
            len(board.pieces(chess.BISHOP, color)) >= 2 and len(end.pieces(chess.BISHOP, color)) < 2
        )
        if lost and len(end.pieces(chess.BISHOP, not color)) >= 2:
            c.add(
                "bishop_pair",
                f"In that line {side(color)} loses the bishop pair; {side(not color)} keeps "
                "both bishops.",
            )


def _phase(board: chess.Board) -> str:
    non_pawn = sum(
        VALUES[pt] * len(board.pieces(pt, c))
        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
        for c in (chess.WHITE, chess.BLACK)
    )
    if board.fullmove_number <= 12 and non_pawn >= 56:
        return "opening"
    if non_pawn <= 26:
        return "endgame"
    return "middlegame"
