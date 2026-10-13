"""Rule-based check that an explanation only states what its facts contain.

It flags, in the explanation text:
  square   a square (``e5``) that no fact mentions
  move     a move in notation (``Nxe5``, ``O-O``) that no fact mentions
  piece    "<piece> on/to/from <square>" where the facts never put that piece type there
  concept  a chess concept (fork, pin, passed pawn, ...) whose fact tag is absent
  number   an evaluation or material number that no fact contains
  verdict  an evaluation word (winning / better / equal) that the facts do not use

It is deliberately strict and literal; the eval pairs it with an LLM judge for paraphrases.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from chesstrainer.explain.facts import SQUARE_RE, MoveFacts

PIECES = "king|queen|rook|bishop|knight|pawn"
_LETTER = {"K": "king", "Q": "queen", "R": "rook", "B": "bishop", "N": "knight"}
# Moves with a piece letter, a capture, castling or a promotion (plain pawn pushes like "e4"
# are indistinguishable from squares and are covered by the square check).
MOVE_RE = re.compile(
    r"(?<![\w-])(O-O-O|O-O|[KQRBN][a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?|[a-h]x[a-h][1-8]"
    r"(?:=[QRBN])?|[a-h][18]=[QRBN])[+#]?"
)
PIECE_SQ_RE = re.compile(
    rf"\b({PIECES})s?\s+(?:on|from|to|at|goes to|moves to|lands on)\s+([a-h][1-8])\b", re.I
)
_FACT_PAIR_RE = re.compile(
    rf"\b({PIECES})\s+(?:moves\s+)?(?:on|from|goes to|to)\s+([a-h][1-8])"
    r"(?:\s+to\s+([a-h][1-8]))?",
    re.I,
)
NUMBER_RE = re.compile(r"(?<![\w.])[+\-−]?\d+\.\d\b")
POINTS_RE = re.compile(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine)\s+points?\b", re.I)
MATE_IN_RE = re.compile(r"\bmate in (\d+)", re.I)
_WORD_NUM = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}

#: concept -> (pattern in the explanation, fact tags that ground it; empty = never computed)
CONCEPTS: dict[str, tuple[str, frozenset[str]]] = {
    "fork": (r"\bfork", frozenset({"fork"})),
    "pin": (r"\bpin(?:s|ned|ning)?\b", frozenset({"pin", "unpin"})),
    "skewer": (r"\bskewer", frozenset({"skewer"})),
    "discovered attack": (r"\bdiscover", frozenset({"discovered"})),
    "checkmate": (r"\bcheckmat|\bmate\b|\bmates\b|\bmating\b", frozenset({"mate"})),
    "check": (r"\bchecks?\b|\bchecking\b", frozenset({"check", "evasion", "mate"})),
    "en passant": (r"\ben passant\b", frozenset({"en_passant"})),
    "promotion": (r"\bpromot", frozenset({"promotion", "passed_pawn"})),
    "castling": (r"\bcastl", frozenset({"castle", "castling_rights"})),
    "sacrifice": (r"\bsacrific", frozenset({"sacrifice"})),
    "undefended": (
        r"\bhanging\b|\bundefended\b|\bunprotected\b|\bunguarded\b|\bnot (?:defended|protected)"
        r"|\bfree (?:piece|pawn|knight|bishop|rook|queen)\b",
        frozenset({"hanging"}),
    ),
    "passed pawn": (r"\bpassed pawn|\bpasser\b", frozenset({"passed_pawn"})),
    "open file": (r"\bopen (?:[a-h]-)?file|\bhalf-open", frozenset({"open_file"})),
    "doubled pawns": (r"\bdoubled\b", frozenset({"doubled"})),
    "isolated pawn": (r"\bisolated\b", frozenset({"isolated"})),
    "bishop pair": (r"\bbishop pair\b|\btwo bishops\b", frozenset({"bishop_pair"})),
    "development": (r"\bdevelop", frozenset({"development"})),
    "centre": (r"\bcent(?:er|re|ral)\b", frozenset({"center"})),
    "threat": (r"\bthreat", frozenset({"threat", "defends"})),
    "opens a line": (
        r"\bopens? (?:up )?(?:a |the )?(?:line|diagonal|path)|\bfrees? (?:up )?(?:the|your|a|its)\b"
        r"|\bmakes? room\b",
        frozenset({"opens_line", "discovered", "open_file"}),
    ),
    "space": (r"\bspace\b", frozenset()),
    "stalemate": (r"\bstalemate", frozenset({"stalemate"})),
    "trapped piece": (r"\btrap(?:s|ped)?\b", frozenset()),
    "outpost": (r"\boutpost", frozenset()),
    "back rank": (r"\bback[- ]rank", frozenset()),
    "zwischenzug": (r"\bzwischenzug|\bin-between move", frozenset()),
    "deflection / decoy": (r"\bdeflect|\bdecoy", frozenset()),
    "overloading": (r"\boverload", frozenset()),
    "battery / x-ray": (r"\bbattery\b|\bx-ray", frozenset()),
    "zugzwang": (r"\bzugzwang", frozenset()),
    "opposition": (r"\bopposition\b", frozenset()),
    "tempo": (r"\btempo", frozenset()),
}

#: verdict phrase in the explanation -> substring a fact must contain
VERDICTS = (
    (
        re.compile(r"\b(?:completely|totally|simply|clearly|is|are|stands?|be) winning\b", re.I),
        "winning",
    ),
    (re.compile(r"\b(?:slightly|clearly|much) better\b", re.I), "better"),
    (
        re.compile(
            r"\b(?:about|roughly|nearly|dead) (?:equal|even|level)\b|\bequal position\b", re.I
        ),
        "equal",
    ),
)


@dataclass(frozen=True)
class Issue:
    kind: str
    claim: str
    detail: str


def _fact_pairs(facts: MoveFacts) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for f in facts.facts:
        for m in _FACT_PAIR_RE.finditer(f.text):
            piece = m.group(1).lower()
            pairs.add((piece, m.group(2)))
            if m.group(3):
                pairs.add((piece, m.group(3)))
        if "castling kingside" in f.text or "castling queenside" in f.text:
            for piece, sq in re.findall(r"the (king|rook) (?:goes )?to ([a-h][1-8])", f.text):
                pairs.add((piece, sq))
    for san in facts.sans():
        sm = re.match(r"([KQRBN])?[a-h]?[1-8]?x?([a-h][1-8])(?:=([QRBN]))?", san)
        if sm:
            pairs.add((_LETTER.get(sm.group(1) or "", "pawn"), sm.group(2)))
            if sm.group(3):  # the promoted piece stands there afterwards
                pairs.add((_LETTER[sm.group(3)], sm.group(2)))
    return pairs


def _notation_tags(fact_text: str) -> set[str]:
    """Concepts implied by moves written in the facts' lines ("O-O", "Qe3+", "e8=Q")."""
    tags = set()
    if re.search(r"[a-h1-8QRBN]\+", fact_text):
        tags.add("check")
    if re.search(r"[a-h1-8QRBN]#", fact_text):
        tags.add("mate")
    if "O-O" in fact_text:
        tags.add("castle")
    if re.search(r"=[QRBN]", fact_text):
        tags.add("promotion")
    return tags


def check_grounding(text: str, facts: MoveFacts) -> list[Issue]:
    issues: list[Issue] = []
    fact_text = facts.text()
    squares = facts.squares()
    for sq in sorted(set(SQUARE_RE.findall(text)) - squares):
        issues.append(Issue("square", sq, "square not in the facts"))
    sans = {s.rstrip("+#") for s in facts.sans()}
    for mv in sorted({m.rstrip("+#") for m in MOVE_RE.findall(text)} - sans):
        issues.append(Issue("move", mv, "move not in the facts"))
    pairs = _fact_pairs(facts)
    for m in PIECE_SQ_RE.finditer(text):
        piece, sq = m.group(1).lower(), m.group(2)
        if (piece, sq) not in pairs and sq in squares:
            issues.append(Issue("piece", m.group(0), f"facts never put a {piece} on {sq}"))
    tags = facts.tags | _notation_tags(fact_text)
    lowered = text.lower()
    for concept, (pattern, needed) in CONCEPTS.items():
        hit = re.search(pattern, lowered)
        if hit and not (needed & tags) and not re.search(pattern, fact_text.lower()):
            issues.append(Issue("concept", hit.group(0), f"'{concept}' is not in the facts"))
    for num in NUMBER_RE.findall(text):
        plain = num.replace("−", "-").lstrip("+")
        if plain not in fact_text and plain.lstrip("-") not in fact_text:
            issues.append(Issue("number", num, "number not in the facts"))
    for m in POINTS_RE.finditer(text):
        n = _WORD_NUM.get(m.group(1).lower(), m.group(1))
        if not re.search(rf"\b{n} points?\b", fact_text):
            issues.append(Issue("number", m.group(0), "material figure not in the facts"))
    for m in MATE_IN_RE.finditer(text):
        if f"mate in {m.group(1)}" not in fact_text:
            issues.append(Issue("number", m.group(0), "mate distance not in the facts"))
    for verdict, needed_word in VERDICTS:
        m2 = verdict.search(text)
        if m2 and needed_word not in fact_text:
            issues.append(Issue("verdict", m2.group(0), f"facts never say '{needed_word}'"))
    return issues
