"""Facts for explaining a book move: the Phase-4 facts plus what the opening book knows.

The extra facts are the opening and variation names, the curated note for the move (if any)
and the opening's known ideas (``repertoire.yaml``). They are ordinary facts, so the prompt's
rules and the grounding check apply to them like to everything else.
"""

from __future__ import annotations

import chess

from chesstrainer.engine.analysis import Line
from chesstrainer.explain.facts import MoveFacts, compute_facts
from chesstrainer.levels import Level
from chesstrainer.openings.book import pos_key
from chesstrainer.openings.trainer import TrainerSession, side_name

INSTRUCTIONS = (
    "This move is the opening book move the player is training ({opening}, playing {side}). "
    "Explain what the move does. If a fact describes this move's idea in the opening, use it; "
    "mention a general opening idea only when it clearly matches this move. Do not call any "
    "other move a book move."
)


def opening_facts(
    session: TrainerSession, board: chess.Board, line: Line, level: Level
) -> MoveFacts:
    """Facts about the trainer's move ``line.move`` in ``board``."""
    mf = compute_facts(board, line, level)
    tree = session.tree
    san = mf.san
    after = board.copy(stack=False)
    after.push(line.move)
    entry = session.book.entry_at(pos_key(after))
    current = session.variation_name()
    name = tree.name + (f" (ECO {tree.eco})" if tree.eco else "")
    text = f"{san} is the book move of the {name} in this position"
    if entry is not None:
        text += f"; it leads to the {entry.name}"
    elif current:
        text += f"; the opening so far is the {current}"
    mf.add("opening_move", text + ".")
    if tree.spec is not None:
        note = tree.spec.note(san)
        if note is not None:
            mf.add("opening_move", f"In this opening: {note}")
        side = side_name(tree.side)
        for idea in tree.spec.ideas:
            mf.add("opening_idea", f"Known idea of the {tree.name} for {side}: {idea}")
    mf.instructions = INSTRUCTIONS.format(opening=tree.name, side=side_name(tree.side))
    mf.context = f"opening:{tree.name}:{side_name(tree.side)}"
    return mf
