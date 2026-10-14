"""The Lichess chess-openings dataset (CC0) as a position graph.

Each TSV row is one named line (ECO code, name, moves). :class:`Book` replays every line once
and indexes positions by a transposition-safe key (placement, side to move, castling, en
passant), so a variation is recognised however it was reached.

Names look like ``"Italian Game: Giuoco Piano"`` - family before the colon, variation after.
:func:`name_matches` decides which lines belong to a chosen opening (see its docstring).
"""

from __future__ import annotations

import re
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import chess

DATA_DIR = Path(__file__).with_name("data")


def pos_key(board: chess.Board) -> str:
    """Placement, side to move, castling and (legal) en passant: the FEN without counters."""
    return " ".join(board.fen().split()[:4])


def fold(text: str) -> str:
    """Lower case without accents ("Grünfeld" -> "grunfeld") for search and matching."""
    norm = unicodedata.normalize("NFKD", text)
    return "".join(c for c in norm if not unicodedata.combining(c)).lower().strip()


def name_matches(name: str, wanted: str) -> bool:
    """Does the dataset line ``name`` belong to the opening ``wanted``?

    A full variation name ("Italian Game: Giuoco Pianissimo") matches itself and its
    sub-variations ("..., Canal Variation"). A plain name ("London System", "Queen's Gambit")
    matches every line where one of the name's parts starts with it or ends with it as whole
    words, so "Queen's Pawn Game: London System" and "Indian Defense: Accelerated London System"
    are Londons and "Queen's Gambit Declined" is a Queen's Gambit, but "Semi-Slav Defense" is
    not a "Slav Defense".
    """
    n, w = fold(name), fold(wanted)
    if ":" in w:
        return n == w or n.startswith(w + ",") or n.startswith(w + ":")
    for part in re.split(r"[:,]", n):
        p = part.strip()
        if p == w or p.startswith(w + " ") or p.endswith(" " + w):
            return True
    return False


@dataclass(frozen=True)
class Entry:
    eco: str
    name: str
    moves: tuple[chess.Move, ...]
    keys: tuple[str, ...]  # position key after 0, 1, ... len(moves) plies

    @property
    def family(self) -> str:
        return self.name.split(":")[0].strip()

    @property
    def end(self) -> str:
        return self.keys[-1]

    def san_text(self) -> str:
        return chess.Board().variation_san(list(self.moves))


class Book:
    def __init__(self, entries: list[Entry]) -> None:
        self.entries = entries
        #: position key -> entries that end there (named positions)
        self.named: dict[str, list[int]] = {}
        #: position key -> {move: child key} over all lines
        self.children: dict[str, dict[chess.Move, str]] = {}
        #: position key -> moves from the start that first reached it
        self.paths: dict[str, tuple[chess.Move, ...]] = {}
        for i, e in enumerate(entries):
            self.named.setdefault(e.end, []).append(i)
            for ply, key in enumerate(e.keys):
                self.paths.setdefault(key, e.moves[:ply])
                if ply < len(e.moves):
                    self.children.setdefault(key, {})[e.moves[ply]] = e.keys[ply + 1]

    @classmethod
    def load(cls, data_dir: Path = DATA_DIR) -> Book:
        entries = []
        for path in sorted(data_dir.glob("*.tsv")):
            lines = path.read_text(encoding="utf-8").splitlines()
            for row in lines[1:]:
                if not row.strip():
                    continue
                eco, name, pgn = row.split("\t")
                entries.append(_entry(eco, name, pgn))
        return cls(entries)

    def __contains__(self, key: str) -> bool:
        return key in self.paths

    def entry_at(self, key: str) -> Entry | None:
        """The line named at exactly this position (the longest name if several)."""
        idx = self.named.get(key)
        if not idx:
            return None
        return max((self.entries[i] for i in idx), key=lambda e: (len(e.moves), len(e.name)))

    def names(self) -> list[str]:
        """Distinct opening names, families first then variations, alphabetical."""
        seen: dict[str, None] = {}
        for e in sorted(self.entries, key=lambda e: fold(e.name)):
            seen.setdefault(e.family, None)
            seen.setdefault(e.name, None)
        return sorted(seen, key=lambda n: (fold(n.split(":")[0]), ":" in n, fold(n)))

    def search(self, query: str) -> list[str]:
        words = fold(query).split()
        return [n for n in self.names() if all(w in fold(n) for w in words)]

    def matching(self, wanted: str) -> list[int]:
        return [i for i, e in enumerate(self.entries) if name_matches(e.name, wanted)]

    def eco_of(self, wanted: str) -> str:
        idx = self.matching(wanted)
        if not idx:
            return ""
        return min((self.entries[i] for i in idx), key=lambda e: len(e.moves)).eco


def _entry(eco: str, name: str, pgn: str) -> Entry:
    board = chess.Board()
    keys = [pos_key(board)]
    moves = []
    for token in pgn.split():
        if token.endswith("."):
            continue
        move = board.parse_san(token)
        board.push(move)
        moves.append(move)
        keys.append(pos_key(board))
    return Entry(eco, name, tuple(moves), tuple(keys))


_lock = threading.Lock()
_default: Book | None = None


def default_book() -> Book:
    """The bundled dataset, loaded once (~2 s; thread-safe, so it can be preloaded)."""
    global _default
    with _lock:
        if _default is None:
            _default = Book.load()
        return _default
