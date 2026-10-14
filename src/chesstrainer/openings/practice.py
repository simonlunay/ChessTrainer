"""Spaced repetition for opening practice, stored in SQLite.

One card per (opening, side, position) where the player has to find the trainer's move. A
simplified SM-2 schedule: a correct answer moves the card to 1 day, then 3 days, then the last
interval times the card's ease (2.5 to start, +0.05 per correct answer, -0.2 per miss, at least
1.3); a miss makes the card due again in 10 minutes and restarts the ladder.

File: ``%APPDATA%\\ChessTrainer\\openings.sqlite`` (next to settings.json).
"""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from chesstrainer.settings.config import settings_path

DAY = 86_400.0
RELEARN_S = 600.0
START_EASE = 2.5
MIN_EASE = 1.3


@dataclass(frozen=True)
class Card:
    opening: str
    side: str  # "white" / "black"
    position: str  # book.pos_key
    move: str  # UCI of the trainer's move
    due: float = 0.0
    interval_days: float = 0.0
    ease: float = START_EASE
    reps: int = 0
    lapses: int = 0
    last: float = 0.0

    @property
    def new(self) -> bool:
        return self.last == 0.0


class PracticeStore:
    def __init__(self, path: Path | None = None, clock: Callable[[], float] = time.time) -> None:
        self.path = path or settings_path().parent / "openings.sqlite"
        self.clock = clock
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS cards ("
                " opening TEXT, side TEXT, position TEXT, move TEXT, due REAL,"
                " interval_days REAL, ease REAL, reps INTEGER, lapses INTEGER, last REAL,"
                " PRIMARY KEY (opening, side, position))"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS reviews ("
                " created REAL, opening TEXT, side TEXT, position TEXT, expected TEXT,"
                " played TEXT, correct INTEGER)"
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=5)
        try:
            with db:
                yield db
        finally:
            db.close()

    def card(self, opening: str, side: str, position: str, move: str) -> Card:
        with self._lock, self._connect() as db:
            row = db.execute(
                "SELECT move, due, interval_days, ease, reps, lapses, last FROM cards"
                " WHERE opening = ? AND side = ? AND position = ?",
                (opening, side, position),
            ).fetchone()
        if row is None or row[0] != move:  # unseen, or the trainer's move changed
            return Card(opening, side, position, move)
        return Card(opening, side, position, move, *row[1:])

    def cards(self, opening: str, side: str) -> dict[str, Card]:
        with self._lock, self._connect() as db:
            rows = db.execute(
                "SELECT position, move, due, interval_days, ease, reps, lapses, last FROM cards"
                " WHERE opening = ? AND side = ?",
                (opening, side),
            ).fetchall()
        return {r[0]: Card(opening, side, *r) for r in rows}

    def review(self, card: Card, played: str) -> Card:
        """Record an answer and return the rescheduled card."""
        now = self.clock()
        correct = played == card.move
        if correct:
            reps = card.reps + 1
            if reps == 1:
                interval = 1.0
            elif reps == 2:
                interval = 3.0
            else:
                interval = max(card.interval_days * card.ease, 1.0)
            ease = card.ease + 0.05
            due, lapses = now + interval * DAY, card.lapses
        else:
            reps, interval = 0, 0.0
            ease = max(MIN_EASE, card.ease - 0.2)
            due, lapses = now + RELEARN_S, card.lapses + 1
        new = Card(
            card.opening,
            card.side,
            card.position,
            card.move,
            due,
            interval,
            ease,
            reps,
            lapses,
            now,
        )
        with self._lock, self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO cards VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    new.opening,
                    new.side,
                    new.position,
                    new.move,
                    new.due,
                    new.interval_days,
                    new.ease,
                    new.reps,
                    new.lapses,
                    new.last,
                ),
            )
            db.execute(
                "INSERT INTO reviews VALUES (?, ?, ?, ?, ?, ?, ?)",
                (now, card.opening, card.side, card.position, card.move, played, int(correct)),
            )
        return new

    def is_due(self, card: Card) -> bool:
        return card.new or card.due <= self.clock()
