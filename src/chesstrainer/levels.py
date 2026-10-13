"""Skill levels, loaded from ``levels.yaml`` (the only place level behaviour is defined)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal

import yaml

LEVELS_FILE = Path(__file__).with_name("levels.yaml")
LEVEL_KEYS = ("beginner", "novice", "intermediate", "advanced", "expert")
DEFAULT_LEVEL = "beginner"

Severity = Literal["inaccuracy", "mistake", "blunder"]


@dataclass(frozen=True)
class ExplanationStyle:
    sentences: str
    style: str
    pv_plies: int
    eval_numbers: bool


@dataclass(frozen=True)
class Level:
    key: str
    name: str
    elo_min: int
    elo_max: int | None
    candidates: int
    practical_margin_cp: int
    practical_min_gain: int
    review: dict[Severity, int | None]
    openings: dict[str, tuple[str, ...]]  # "white" / "black" -> names
    opening_depth_plies: int
    explanation: ExplanationStyle

    @property
    def elo_text(self) -> str:
        if self.elo_max is None:
            return f"{self.elo_min}+"
        if self.elo_min == 0:
            return f"under {self.elo_max}"
        return f"{self.elo_min}-{self.elo_max}"

    @property
    def label(self) -> str:
        return f"{self.name} ({self.elo_text} Elo)"

    def classify_loss(self, cp_loss: int) -> Severity | None:
        """Review severity of a move that loses ``cp_loss`` centipawns (None = not flagged)."""
        order: tuple[Severity, ...] = ("blunder", "mistake", "inaccuracy")
        for severity in order:
            threshold = self.review[severity]
            if threshold is not None and cp_loss >= threshold:
                return severity
        return None


def _parse(raw: dict[str, Any]) -> Level:
    lo, hi = raw["elo"]
    ex = raw["explanation"]
    return Level(
        key=str(raw["key"]),
        name=str(raw["name"]),
        elo_min=int(lo),
        elo_max=None if hi is None else int(hi),
        candidates=max(1, min(3, int(raw["candidates"]))),
        practical_margin_cp=int(raw["practical_margin_cp"]),
        practical_min_gain=int(raw["practical_min_gain"]),
        review={
            k: (None if raw["review"].get(k) is None else int(raw["review"][k]))
            for k in ("inaccuracy", "mistake", "blunder")
        },
        openings={side: tuple(raw["openings"].get(side, ())) for side in ("white", "black")},
        opening_depth_plies=int(raw["opening_depth_plies"]),
        explanation=ExplanationStyle(
            sentences=str(ex["sentences"]),
            style=" ".join(str(ex["style"]).split()),
            pv_plies=int(ex["pv_plies"]),
            eval_numbers=bool(ex["eval_numbers"]),
        ),
    )


@cache
def all_levels(path: Path = LEVELS_FILE) -> tuple[Level, ...]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    levels = tuple(_parse(raw) for raw in data["levels"])
    keys = tuple(lv.key for lv in levels)
    if keys != LEVEL_KEYS:
        raise ValueError(f"{path}: levels must be {LEVEL_KEYS} in order, got {keys}")
    return levels


def get_level(key: str) -> Level:
    for lv in all_levels():
        if lv.key == key:
            return lv
    return get_level(DEFAULT_LEVEL)


def level_for_elo(elo: int) -> Level:
    for lv in all_levels():
        if elo >= lv.elo_min and (lv.elo_max is None or elo < lv.elo_max):
            return lv
    return all_levels()[0]
