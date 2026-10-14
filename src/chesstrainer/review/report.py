"""Review summary (mistake counts and types, what to work on) and the facts for explaining a
mistake. Explanations use the same grounded pipeline as the live app: only computed facts go to
Claude, and the rule-based check runs on every answer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import chess

from chesstrainer.engine.analysis import Line
from chesstrainer.explain.board_tools import side
from chesstrainer.explain.facts import MoveFacts, compute_facts
from chesstrainer.levels import Level
from chesstrainer.review.analysis import GameReview, MoveReview
from chesstrainer.review.mistakes import KindCount, count_kinds

#: Facts about the move actually played that are added to the best move's facts.
PLAYED_TAGS = (
    "mate",
    "eval",
    "line",
    "reply",
    "captures",
    "material_loss",
    "material_gain",
    "sacrifice",
    "en_prise",
)

INSTRUCTIONS = (
    "This is a review of the player's own game. The player played {played}, which Stockfish "
    "rates as a {severity}; the move to explain, {best}, was better. First say briefly what was "
    "wrong with {played}, using only the facts about it (for example what the opponent's best "
    "reply wins), then what {best} does instead. Address the player as 'you'."
)


@dataclass
class ReviewSummary:
    games: int = 0
    user_moves: int = 0
    severities: dict[str, int] = field(default_factory=dict)
    average_loss: float = 0.0  # centipawns per move of the player's
    kinds: list[KindCount] = field(default_factory=list)

    @property
    def flagged(self) -> int:
        return sum(self.severities.values())


def summarize(reviews: Sequence[GameReview]) -> ReviewSummary:
    s = ReviewSummary(games=len(reviews))
    pairs = []
    total = 0
    for r in reviews:
        for m in r.user_moves():
            s.user_moves += 1
            total += m.loss
            if m.severity is not None:
                s.severities[m.severity] = s.severities.get(m.severity, 0) + 1
                if m.mistake is not None:
                    pairs.append((m.mistake, m.loss))
    s.average_loss = total / s.user_moves if s.user_moves else 0.0
    s.kinds = count_kinds(pairs)
    return s


def _plural(n: int, word: str) -> str:
    if n == 1:
        return f"{n} {word}"
    return f"{n} {word[:-1]}ies" if word.endswith("y") else f"{n} {word}s"


def summary_text(s: ReviewSummary, level: Level) -> str:
    games = _plural(s.games, "game")
    head = f"{games}, {_plural(s.user_moves, 'move')} of yours."
    parts = [
        _plural(s.severities[k], k)
        for k in ("blunder", "mistake", "inaccuracy")
        if s.severities.get(k)
    ]
    if level.review.get("inaccuracy") is None:
        note = f" (inaccuracies are not flagged at {level.name} level)"
    else:
        note = ""
    lines = [head]
    if parts:
        lines.append("Flagged: " + ", ".join(parts) + note + ".")
    else:
        lines.append(f"No mistakes at the {level.name} thresholds{note}. Well played!")
    lines.append(f"Average loss per move: {s.average_loss / 100:.2f} pawns.")
    if s.kinds:
        lines.append("")
        lines.append("What to work on:")
        for i, kc in enumerate(s.kinds[:4], 1):
            lines.append(f"{i}. {kc.label} ({kc.count}): {kc.tip}")
    return "\n".join(lines)


def review_facts(m: MoveReview, level: Level) -> MoveFacts:
    """Facts for explaining why the best move beats the (flagged) move the player made."""
    assert m.best is not None and m.best.pv and m.severity is not None
    board = chess.Board(m.fen_before)
    mf = compute_facts(board, m.best, level)
    if m.after is not None:
        played = Line(1, (m.move, *m.after.pv), m.after.cp, m.after.mate)
    else:
        played = Line(1, (m.move,), 0, None)
    pf = compute_facts(board, played, level)
    best_san = mf.san
    pawns = m.loss / 100
    amount = f"{pawns:.1f}" if level.explanation.eval_numbers else f"{max(1, round(pawns))}"
    mf.add(
        "review_played",
        f"In the game, {side(m.mover)} played {m.san} instead of {best_san}. Stockfish rates "
        f"{m.san} as a {m.severity}: about {amount} pawns worse than {best_san}.",
    )
    if m.mistake is not None:
        mf.add("review_type", f"Mistake type: {m.mistake.label.lower()}. {m.mistake.detail}")
    for f in pf.facts:
        if f.tag in PLAYED_TAGS:
            mf.add(f.tag, f"About {m.san} (the move played): {f.text}")
    mf.instructions = INSTRUCTIONS.format(played=m.san, best=best_san, severity=m.severity)
    mf.context = f"review:{m.move.uci()}"
    return mf
