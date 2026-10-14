"""Opening trainer logic, independent of Qt and the screen.

:class:`OpeningTree` is the book of one chosen opening for one side: every Lichess dataset line
that belongs to it (``book.name_matches``) plus the curated lines from ``repertoire.yaml``. At a
position where the trained side moves, the trainer's move is the curated one if there is one,
otherwise the book move followed by the most dataset lines (lines named "Gambit" count 0.3 unless
the chosen opening is itself a gambit).

:class:`TrainerSession` follows a game from the positions the board reader commits. Each new
position is matched to the previous one by one or two legal moves (placement only, so a guessed
side to move does not matter), to an earlier position (a takeback), to the start position (a new
game), or to a position of the opening's book (training starts mid-line). Every move produces
:class:`TrainerEvent`\\ s with a message for the panel; a move that leaves the book switches the
session to ``left`` and the app to normal analysis. ``grade`` asks the app to say whether the
move that left the book was a mistake (that needs the engine).

Modes: ``guided`` shows the trainer's move as an arrow; ``practice`` shows nothing, checks the
player's move and records it in the spaced-repetition store (a missed position shows its arrow
after a takeback).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import chess

from chesstrainer.levels import Level
from chesstrainer.openings.book import Book, Entry, fold, pos_key
from chesstrainer.openings.practice import Card, PracticeStore
from chesstrainer.openings.repertoire import OpeningSpec, find_spec

Mode = Literal["guided", "practice"]
State = Literal["waiting", "book", "left", "complete"]

GAMBIT_WEIGHT = 0.3


def side_name(color: chess.Color) -> str:
    return "White" if color == chess.WHITE else "Black"


def move_text(board: chess.Board, move: chess.Move) -> str:
    """ "3. Bc4" / "3... Bc5"."""
    dots = "." if board.turn == chess.WHITE else "..."
    return f"{board.fullmove_number}{dots} {board.san(move)}"


class OpeningTree:
    def __init__(self, book: Book, name: str, side: chess.Color, spec: OpeningSpec | None) -> None:
        self.book = book
        self.name = name
        self.side = side
        self.spec = spec
        self.eco = book.eco_of(name)
        self.children: dict[str, dict[chess.Move, str]] = {}
        self.weight: dict[tuple[str, chess.Move], float] = {}
        self.paths: dict[str, tuple[chess.Move, ...]] = {}
        self.by_placement: dict[str, list[str]] = {}
        self.curated: dict[str, chess.Move] = {}
        gambit_ok = "gambit" in fold(name)
        self.entries = [book.entries[i] for i in book.matching(name)]
        for e in self.entries:
            w = GAMBIT_WEIGHT if "gambit" in fold(e.name) and not gambit_ok else 1.0
            self._add_line(e.moves, e.keys, w)
        if spec is not None:
            for line in spec.lines:
                board = chess.Board()
                keys = [pos_key(board)]
                for ply, move in enumerate(line):
                    if board.turn == side:
                        self.curated.setdefault(keys[ply], move)
                    board.push(move)
                    keys.append(pos_key(board))
                self._add_line(line, tuple(keys), 1.0)

    def _add_line(
        self, moves: tuple[chess.Move, ...], keys: tuple[str, ...], weight: float
    ) -> None:
        for ply, key in enumerate(keys):
            if key not in self.paths or len(self.paths[key]) > ply:
                self.paths[key] = moves[:ply]
                placement = key.split()[0]
                if key not in self.by_placement.setdefault(placement, []):
                    self.by_placement[placement].append(key)
            if ply < len(moves):
                self.children.setdefault(key, {})[moves[ply]] = keys[ply + 1]
                k = (key, moves[ply])
                self.weight[k] = self.weight.get(k, 0.0) + weight

    @property
    def size(self) -> int:
        return len(self.entries)

    def __contains__(self, key: str) -> bool:
        return key in self.paths

    def book_moves(self, board: chess.Board) -> list[chess.Move]:
        """Book continuations of this opening, most common first."""
        key = pos_key(board)
        moves = list(self.children.get(key, {}))
        return sorted(moves, key=lambda m: -self.weight.get((key, m), 0.0))

    def is_book(self, board: chess.Board, move: chess.Move) -> bool:
        return move in self.children.get(pos_key(board), {})

    def user_move(self, board: chess.Board) -> chess.Move | None:
        """The trainer's move for the trained side (None: not its turn, or end of the book)."""
        if board.turn != self.side:
            return None
        key = pos_key(board)
        if key in self.curated:
            return self.curated[key]
        moves = self.book_moves(board)
        return moves[0] if moves else None


@dataclass(frozen=True)
class TrainerEvent:
    kind: str  # start resync takeback lost user_correct user_other user_left opp_book
    #            opp_other_opening opp_left complete
    message: str
    move: chess.Move | None = None
    mover: chess.Color | None = None
    expected: chess.Move | None = None
    grade: bool = False  # the app should tell whether ``move`` was a mistake
    fen_before: str = ""


@dataclass
class _Ply:
    board: chess.Board
    state: State
    variation: Entry | None


@dataclass
class TrainerSession:
    book: Book
    tree: OpeningTree
    level: Level
    mode: Mode = "guided"
    store: PracticeStore | None = None
    history: list[_Ply] = field(default_factory=list)
    reveal: set[str] = field(default_factory=set)  # practice: missed positions show the arrow
    score: dict[str, int] = field(default_factory=lambda: {"correct": 0, "missed": 0})

    @classmethod
    def create(
        cls,
        book: Book,
        name: str,
        side: chess.Color,
        level: Level,
        mode: Mode = "guided",
        store: PracticeStore | None = None,
    ) -> TrainerSession:
        tree = OpeningTree(book, name, side, find_spec(name, side))
        return cls(book, tree, level, mode, store)

    # -- state -------------------------------------------------------------------------------

    @property
    def side(self) -> chess.Color:
        return self.tree.side

    @property
    def board(self) -> chess.Board | None:
        return self.history[-1].board if self.history else None

    @property
    def state(self) -> State:
        return self.history[-1].state if self.history else "waiting"

    @property
    def in_book(self) -> bool:
        return self.state == "book"

    @property
    def variation(self) -> Entry | None:
        return self.history[-1].variation if self.history else None

    @property
    def depth(self) -> int:
        return self.level.opening_depth_plies

    @property
    def side_text(self) -> str:
        return side_name(self.side)

    def user_to_move(self) -> bool:
        b = self.board
        return b is not None and b.turn == self.side

    def expected_move(self) -> chess.Move | None:
        """The trainer's move in the current position, if training is on and it is our turn."""
        b = self.board
        if b is None or not self.in_book:
            return None
        return self.tree.user_move(b)

    def arrow_move(self) -> chess.Move | None:
        """What to draw: the trainer's move in guided mode (or after a miss in practice)."""
        move = self.expected_move()
        b = self.board
        if move is None or b is None:
            return None
        if self.mode == "guided" or pos_key(b) in self.reveal:
            return move
        return None

    def variation_name(self) -> str:
        v = self.variation
        return v.name if v is not None else ""

    # -- following the game ------------------------------------------------------------------

    def observe(self, board: chess.Board) -> list[TrainerEvent]:
        """Feed a position read from the screen; returns what happened (usually 0-2 events)."""
        placement = board.board_fen()
        if self.history:
            cur = self.history[-1].board
            if cur.board_fen() == placement:
                return []
            for i in range(len(self.history) - 2, -1, -1):
                if self.history[i].board.board_fen() == placement:
                    return [self._takeback(i)]
            path = _connect(cur, placement)
            if path is not None:
                events: list[TrainerEvent] = []
                for move in path:
                    events += self._play(move)
                return events
        if placement == chess.Board().board_fen():
            self.history = [_Ply(chess.Board(), "book", None)]
            self.reveal.clear()
            return [TrainerEvent("start", f"New game: {self.prompt()}")]
        keys = self.tree.by_placement.get(placement, [])
        keys = sorted(keys, key=lambda k: k.split()[1] != ("w" if board.turn else "b"))
        if keys:
            self._replay(self.tree.paths[keys[0]])
            return [TrainerEvent("resync", f"Position found in the book. {self.prompt()}")]
        had = bool(self.history)
        self.history = []
        if had:
            return [
                TrainerEvent(
                    "lost",
                    "This position is not from the line being trained; normal analysis is on. "
                    "Set up the starting position (or a position from the line) to train again.",
                )
            ]
        return []

    def _replay(self, moves: tuple[chess.Move, ...]) -> None:
        board = chess.Board()
        self.history = [_Ply(board.copy(), "book", None)]
        variation = None
        for m in moves:
            board.push(m)
            variation = self.book.entry_at(pos_key(board)) or variation
            self.history.append(_Ply(board.copy(), "book", variation))
        self._check_complete()

    def _takeback(self, index: int) -> TrainerEvent:
        self.history = self.history[: index + 1]
        if self.in_book:
            return TrainerEvent("takeback", f"Move taken back. {self.prompt()}")
        return TrainerEvent("takeback", "Move taken back.")

    def _play(self, move: chess.Move) -> list[TrainerEvent]:
        cur = self.history[-1].board
        state = self.state
        nxt = cur.copy()
        nxt.push(move)
        key_after = pos_key(nxt)
        variation = self.book.entry_at(key_after) or self.variation
        events: list[TrainerEvent] = []
        if state == "book":
            if cur.turn == self.side:
                event, state = self._user_move(cur, move)
            else:
                event, state = self._opponent_move(cur, move, nxt)
            events.append(event)
        self.history.append(_Ply(nxt, state, variation))
        if state == "book":
            done = self._check_complete()
            if done is not None:
                events.append(done)
        return events

    def _user_move(self, cur: chess.Board, move: chess.Move) -> tuple[TrainerEvent, State]:
        expected = self.tree.user_move(cur)
        key = pos_key(cur)
        played, fen = move_text(cur, move), cur.fen()
        exp_text = move_text(cur, expected) if expected is not None else ""
        if expected is not None and self.mode == "practice" and self.store is not None:
            card = self.store.card(self.tree.name, self.side_text.lower(), key, expected.uci())
            self.store.review(card, move.uci())
        if expected is not None and self.mode == "practice":
            self.score["correct" if move == expected else "missed"] += 1
        if move == expected:
            self.reveal.discard(key)
            word = "Correct" if self.mode == "practice" else "Book move"
            return TrainerEvent("user_correct", f"{word}: {played}.", move, self.side), "book"
        if self.mode == "practice":
            self.reveal.add(key)
        if self.tree.is_book(cur, move):
            msg = f"{played} is also a book move, but the line's move here was {exp_text}."
            if self.mode == "practice":
                msg += " Take it back to practise the line's move, or play on."
            return TrainerEvent("user_other", msg, move, self.side, expected), "book"
        after = cur.copy(stack=False)
        after.push(move)
        other = self.book.entry_at(pos_key(after))
        if pos_key(after) in self.book and other is not None:
            msg = (
                f"{played} is a book move of the {other.name}, not the {self.tree.name} "
                f"(the line's move was {exp_text})."
            )
            if self.mode == "practice":
                msg += " Take it back to practise the line's move, or play on."
            else:
                msg += " The trainer's line ends here; normal analysis is on."
            return TrainerEvent("user_other_opening", msg, move, self.side, expected), "left"
        msg = f"You left the book with {played}. The book move was {exp_text}."
        if self.mode == "practice":
            msg += " Take it back to try again, or play on with normal analysis."
        else:
            msg += " Normal analysis is on from here."
        event = TrainerEvent("user_left", msg, move, self.side, expected, True, fen)
        return event, "left"

    def _opponent_move(
        self, cur: chess.Board, move: chess.Move, nxt: chess.Board
    ) -> tuple[TrainerEvent, State]:
        opp = side_name(not self.side)
        played, fen = move_text(cur, move), cur.fen()
        if self.tree.is_book(cur, move):
            entry = self.book.entry_at(pos_key(nxt))
            named = f" ({entry.name})" if entry is not None else ""
            msg = f"{opp} played {played}, a book move{named}."
            return TrainerEvent("opp_book", msg, move, not self.side), "book"
        other = self.book.entry_at(pos_key(nxt))
        if other is None and pos_key(nxt) in self.book:
            other = self.variation
        if pos_key(nxt) in self.book and other is not None:
            msg = (
                f"{opp} played {played}, which leads to the {other.name}, not the "
                f"{self.tree.name}. The trainer's line ends here; normal analysis is on."
            )
            return TrainerEvent("opp_other_opening", msg, move, not self.side), "left"
        msg = (
            f"{opp} left the book with {played}. Normal analysis is on: the green arrow shows "
            "the best reply."
        )
        return TrainerEvent("opp_left", msg, move, not self.side, None, True, fen), "left"

    def _check_complete(self) -> TrainerEvent | None:
        ply = self.history[-1]
        board = ply.board
        plies = len(board.move_stack)
        reason = ""
        if plies >= self.depth:
            reason = f"Line complete ({plies} plies, the depth for {self.level.name} level)."
        elif not self.tree.book_moves(board):
            reason = "End of the book for this line."
        if not reason:
            return None
        ply.state = "complete"
        return TrainerEvent("complete", f"{reason} Normal analysis is on from here.")

    # -- what to show ------------------------------------------------------------------------

    def prompt(self) -> str:
        """One line for the panel: what the player should do now."""
        b = self.board
        if b is None:
            return (
                f"Set up the starting position (or a position from the line) to train the "
                f"{self.tree.name} as {self.side_text}."
            )
        if self.state == "left":
            return "Out of book: normal analysis."
        if self.state == "complete":
            return "Line complete: normal analysis."
        if b.turn == self.side:
            move = self.expected_move()
            if move is None:
                return "End of the book for this line."
            if self.arrow_move() is not None:
                return f"Your move: {move_text(b, move)} (green arrow)."
            return "Your move: play the book move."
        replies = [move_text(b, m) for m in self.tree.book_moves(b)[:4]]
        text = f"Waiting for {side_name(not self.side)}'s reply."
        if replies:
            text += f" Book replies: {', '.join(replies)}."
        suggestion = self.practice_reply()
        if suggestion is not None:
            text += f" To practise what is due, play {move_text(b, suggestion)} for the opponent."
        return text

    def practice_reply(self) -> chess.Move | None:
        """Practice: the opponent book move leading to the most due (or new) positions."""
        b = self.board
        if self.mode != "practice" or b is None or b.turn == self.side or not self.in_book:
            return None
        moves = self.tree.book_moves(b)
        if len(moves) < 2:
            return None
        cards = self.store.cards(self.tree.name, self.side_text.lower()) if self.store else {}
        best, best_n = None, -1
        for m in moves[:6]:
            after = b.copy()
            after.push(m)
            n = self._due_ahead(after, cards, budget=[200])
            if n > best_n:
                best, best_n = m, n
        return best

    def _due_ahead(self, board: chess.Board, cards: dict[str, Card], budget: list[int]) -> int:
        """Due or unseen trainer positions reachable within the line depth."""
        if budget[0] <= 0 or len(board.move_stack) >= self.depth:
            return 0
        budget[0] -= 1
        if board.turn == self.side:
            move = self.tree.user_move(board)
            if move is None:
                return 0
            card = cards.get(pos_key(board))
            due = card is None or (self.store is not None and self.store.is_due(card))
            board.push(move)
            n = int(due) + self._due_ahead(board, cards, budget)
            board.pop()
            return n
        total = 0
        for m in self.tree.book_moves(board)[:3]:
            board.push(m)
            total += self._due_ahead(board, cards, budget)
            board.pop()
        return total

    def practice_summary(self) -> str:
        if self.mode != "practice" or self.store is None:
            return ""
        cards = self.store.cards(self.tree.name, self.side_text.lower())
        due = sum(1 for c in cards.values() if self.store.is_due(c))
        learned = sum(1 for c in cards.values() if c.reps >= 2)
        s = self.score
        return (
            f"This session: {s['correct']} correct, {s['missed']} missed. "
            f"Saved: {len(cards)} positions seen, {learned} learned, {due} due now."
        )


def _connect(board: chess.Board, placement: str) -> list[chess.Move] | None:
    """One or two legal moves from ``board`` that reach ``placement``."""
    for m in board.legal_moves:
        board.push(m)
        hit = board.board_fen() == placement
        board.pop()
        if hit:
            return [m]
    for m in board.legal_moves:
        board.push(m)
        for r in board.legal_moves:
            board.push(r)
            hit = board.board_fen() == placement
            board.pop()
            if hit:
                board.pop()
                return [m, r]
        board.pop()
    return None
