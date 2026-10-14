"""What kind of mistake a move was, computed with python-chess from Stockfish's lines.

Checked in order (the first match is the mistake's type): allowed a forced mate, missed a forced
mate, a piece the opponent can win at once (the moved piece or another one: "hanging"; a piece
that was already attacked before the move: "ignored threat"), a winning capture not taken, the
opponent's best reply is a fork / skewer / discovered attack, the opponent's line wins material,
or it pins; the best move was a fork / skewer / discovered attack, the best line wins material,
or it pins; else positional. Lines are judged 8 plies deep (``HORIZON``). Each type has a label
and a training tip for the review summary."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import chess

from chesstrainer.engine.analysis import Line
from chesstrainer.explain import board_tools as bt
from chesstrainer.explain.board_tools import describe, side
from chesstrainer.explain.facts import Fact, compute_facts, mover_cp
from chesstrainer.levels import get_level

#: kind -> (summary label, what to work on). Order = priority in the summary when counts tie.
KINDS: dict[str, tuple[str, str]] = {
    "hung_piece": (
        "Hanging pieces",
        "Before you let go of a piece, ask: can it be captured, and is it defended enough?",
    ),
    "ignored_threat": (
        "Ignored threats",
        "After every opponent move, ask what it attacks, and save attacked pieces first.",
    ),
    "missed_capture": (
        "Missed captures",
        "Each move, check whether your opponent left something you can take for free.",
    ),
    "allowed_mate": (
        "Allowed checkmate",
        "Before every move, look at all of your opponent's checks: can one of them mate?",
    ),
    "missed_mate": (
        "Missed checkmate",
        "Look at every check you have first: checks, then captures, then threats.",
    ),
    "allowed_tactic": (
        "Walked into tactics (forks, pins, skewers)",
        "Check where enemy knights and long-range pieces can go next, and avoid leaving two "
        "valuable pieces where one move can attack both.",
    ),
    "lost_material": (
        "Lost material in exchanges",
        "Count attackers and defenders before you start or allow a series of captures.",
    ),
    "missed_tactic": (
        "Missed tactics",
        "Look for forks, pins and skewers: one move that attacks two things at once.",
    ),
    "missed_win": (
        "Missed ways to win material",
        "When an enemy piece looks loose, work out captures and checks a few moves deep.",
    ),
    "positional": (
        "Positional / long-term",
        "Compare your move with Stockfish's line: development, king safety, pawn structure.",
    ),
}

#: Pins come last: they are often incidental, so a line that simply wins material is named
#: "lost material" / "missed win" before an accompanying pin.
TACTIC_TAGS = ("fork", "skewer", "discovered", "pin")

#: Lines are judged over the Expert horizon (8 plies, longer while captures continue) whatever
#: the player's level: a short Beginner line would miss e.g. a trapped piece lost 4 plies later.
HORIZON = get_level("expert")


@dataclass(frozen=True)
class Mistake:
    kind: str  # key of KINDS
    detail: str  # plain sentence(s), computed

    @property
    def label(self) -> str:
        return KINDS[self.kind][0]


def _mate_for(line: Line | None, color: chess.Color) -> int | None:
    """Mate-in-N for ``color`` in ``line`` (None: no forced mate for that side)."""
    if line is None or line.mate is None:
        return None
    return abs(line.mate) if mover_cp(line, color) > 0 else None


def _wins_now(before: chess.Board, sq: chess.Square, attacker: chess.Color) -> bool:
    """Could ``attacker`` already win the piece on ``sq`` before the move (null move)?"""
    null = bt.with_turn(before, attacker)
    if null is None:
        return False
    return any(m.to_square == sq and bt.see(null, m) > 0 for m in null.legal_moves)


def _tactic(facts: list[Fact]) -> Fact | None:
    for tag in TACTIC_TAGS:
        fact = next((f for f in facts if f.tag == tag), None)
        if fact is not None:
            return fact
    return None


def classify_mistake(
    before: chess.Board, move: chess.Move, best: Line, reply: Line | None
) -> Mistake:
    me, them = before.turn, not before.turn
    san = before.san(move)
    after = before.copy(stack=False)
    after.push(move)
    best_san = before.san(best.move)

    n = _mate_for(reply, them)
    if n is not None and _mate_for(best, them) is None:
        return Mistake(
            "allowed_mate", f"After {san}, {side(them)} can force checkmate in {n} (Stockfish)."
        )
    n = _mate_for(best, me)
    if n is not None and _mate_for(reply, me) is None:
        return Mistake("missed_mate", f"{best_san} would have forced checkmate in {n} (Stockfish).")

    if reply is not None and reply.pv:
        r = reply.move
        if after.is_capture(r) and bt.see(after, r) > 0:
            sq = move.to_square if after.is_en_passant(r) else r.to_square
            victim, by = describe(after, sq), describe(after, r.from_square)
            r_san = after.san(r)
            if sq != move.to_square and _wins_now(before, sq, them):
                return Mistake(
                    "ignored_threat",
                    f"{victim} was already under attack and {san} did not save it: "
                    f"{by} can capture it ({r_san}) and win material.",
                )
            return Mistake(
                "hung_piece",
                f"After {san}, {by} can capture {victim} ({r_san}) and win material.",
            )

    b = best.move
    if before.is_capture(b) and bt.see(before, b) > 0:
        won = bt.see(before, b)
        took = before.is_capture(move) and bt.see(before, move) >= won
        if not took:
            target = (
                f"{side(them)}'s pawn" if before.is_en_passant(b) else describe(before, b.to_square)
            )
            return Mistake(
                "missed_capture",
                f"{best_san} would have won material: {describe(before, b.from_square)} could "
                f"capture {target}.",
            )

    if reply is not None and reply.pv:
        rf = compute_facts(after, reply, HORIZON)
        tactic = _tactic(rf.facts)
        if tactic is not None and tactic.tag != "pin":
            return Mistake(
                "allowed_tactic", f"{side(them)}'s best reply is {rf.san}. {tactic.text}"
            )
        gain = next((f for f in rf.facts if f.tag == "material_gain"), None)
        if gain is not None:
            line = next((f.text for f in rf.facts if f.tag == "line"), "")
            return Mistake("lost_material", f"After {san}: {line} {gain.text}".strip())
        if tactic is not None:
            return Mistake(
                "allowed_tactic", f"{side(them)}'s best reply is {rf.san}. {tactic.text}"
            )

    bf = compute_facts(before, best, HORIZON)
    tactic = _tactic(bf.facts)
    if tactic is not None and tactic.tag != "pin":
        return Mistake("missed_tactic", f"With {best_san}: {tactic.text}")
    gain = next((f for f in bf.facts if f.tag == "material_gain"), None)
    if gain is not None:
        line = next((f.text for f in bf.facts if f.tag == "line"), "")
        return Mistake("missed_win", f"With {best_san}: {line} {gain.text}".strip())
    if tactic is not None:
        return Mistake("missed_tactic", f"With {best_san}: {tactic.text}")
    return Mistake(
        "positional",
        f"No immediate tactic or material loss: Stockfish simply prefers {best_san}.",
    )


@dataclass(frozen=True)
class KindCount:
    kind: str
    count: int
    total_loss: int

    @property
    def label(self) -> str:
        return KINDS[self.kind][0]

    @property
    def tip(self) -> str:
        return KINDS[self.kind][1]


def count_kinds(mistakes: Iterable[tuple[Mistake, int]]) -> list[KindCount]:
    """(mistake, loss) pairs -> counts per kind, most frequent (then costliest) first."""
    counts: dict[str, list[int]] = {}
    for m, loss in mistakes:
        counts.setdefault(m.kind, []).append(loss)
    order = list(KINDS)
    out = [KindCount(k, len(v), sum(v)) for k, v in counts.items()]
    out.sort(key=lambda kc: (-kc.count, -kc.total_loss, order.index(kc.kind)))
    return out
