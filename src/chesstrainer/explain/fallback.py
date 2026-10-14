"""Plain-language explanation built from the facts alone (no LLM).

Shown when Claude's explanation fails the grounding check twice: the most important computed
facts, already written as plain sentences, in priority order and cut to the level's length.
"""

from __future__ import annotations

from chesstrainer.explain.facts import Fact, MoveFacts
from chesstrainer.levels import Level

#: Most important first. Tags not listed are never used in the summary.
PRIORITY = (
    "mate",
    "check",
    "capture",
    "material_gain",
    "fork",
    "skewer",
    "pin",
    "discovered",
    "promotion",
    "threat",
    "defends",
    "hanging",
    "sacrifice",
    "en_prise",
    "engine_best",
    "opening_move",
    "castle",
    "development",
    "center",
    "opening_idea",
    "eval",
)

#: Sentences per level (the explanation's maximum, plus the opening sentence about the move).
_MAX = {"1-2": 2, "2": 2, "2-3": 3, "1": 1}


def facts_summary(facts: MoveFacts, level: Level) -> str:
    n = _MAX.get(level.explanation.sentences, 3)
    by_tag: dict[str, list[Fact]] = {}
    for f in facts.facts:
        by_tag.setdefault(f.tag, []).append(f)
    chosen: list[Fact] = []
    first = by_tag.get("move", [])[:1]  # "White plays Bc4: the bishop moves from f1 to c4."
    for tag in PRIORITY:
        for f in by_tag.get(tag, []):
            if len(chosen) < n and f not in chosen:
                chosen.append(f)
    if not chosen and not first:
        return f"{facts.san}."
    return " ".join(f.text for f in first + chosen)
