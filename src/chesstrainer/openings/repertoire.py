"""Curated opening knowledge from ``repertoire.yaml``: main lines, known ideas, move notes."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import chess
import yaml

from chesstrainer.openings.book import fold

REPERTOIRE_FILE = Path(__file__).with_name("repertoire.yaml")


@dataclass(frozen=True)
class OpeningSpec:
    name: str
    side: chess.Color
    ideas: tuple[str, ...]
    notes: dict[str, str]  # SAN without check marks -> sentence
    lines: tuple[tuple[chess.Move, ...], ...]

    def note(self, san: str) -> str | None:
        return self.notes.get(san.rstrip("+#"))


def _parse(raw: dict[str, Any]) -> OpeningSpec:
    name = str(raw["name"])
    lines = []
    for text in raw.get("lines", []) or []:
        board = chess.Board()
        moves = []
        for san in str(text).split():
            try:
                move = board.parse_san(san)
            except ValueError as err:
                raise ValueError(f"{REPERTOIRE_FILE.name}: {name}: bad move {san!r}") from err
            board.push(move)
            moves.append(move)
        lines.append(tuple(moves))
    return OpeningSpec(
        name=name,
        side=chess.WHITE if raw["side"] == "white" else chess.BLACK,
        ideas=tuple(" ".join(str(i).split()) for i in raw.get("ideas", []) or []),
        notes={str(k): " ".join(str(v).split()) for k, v in (raw.get("notes") or {}).items()},
        lines=tuple(lines),
    )


@cache
def all_specs(path: Path = REPERTOIRE_FILE) -> tuple[OpeningSpec, ...]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return tuple(_parse(raw) for raw in data["openings"])


def find_spec(name: str, side: chess.Color) -> OpeningSpec | None:
    """The curated spec for this opening and side; a variation falls back to its family."""
    candidates = [fold(name), fold(name.split(":")[0])]
    for wanted in candidates:
        for spec in all_specs():
            if spec.side == side and fold(spec.name) == wanted:
                return spec
    return None
