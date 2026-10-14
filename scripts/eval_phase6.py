"""Phase 6 review-mode evaluation: real games through the real review pipeline, no screen use.

    python -m uv run python scripts/eval_phase6.py [--no-explain] [--budget 0.5]

1. Fetches recent games from the public Lichess and Chess.com APIs (``fetch_lichess`` /
   ``fetch_chesscom``, the same code the window uses) for a few public accounts, plus the
   bundled sample PGN with typical beginner blunders.
2. Reviews them with real Stockfish at the given levels (the player = the named account).
3. Explains the biggest mistakes with Claude Haiku through the live path (facts -> explanation
   -> rule-based grounding check, regenerate once, else facts), within ``--budget`` dollars.
Writes ``data/eval/phase6_report.txt`` and ``phase6_reviews.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chesstrainer.engine.locate import find_stockfish  # noqa: E402
from chesstrainer.explain.explainer import Explainer, ExplainError, ExplanationCache  # noqa: E402
from chesstrainer.levels import get_level  # noqa: E402
from chesstrainer.review.analysis import GameReview, StockfishEvaluator, review_game  # noqa: E402
from chesstrainer.review.games import (  # noqa: E402
    FetchError,
    GameRecord,
    fetch_chesscom,
    fetch_lichess,
    parse_pgn,  # noqa: E402
)
from chesstrainer.review.report import review_facts, summarize, summary_text  # noqa: E402

OUT = ROOT / "data" / "eval"
#: claude-haiku-5-5 price per million tokens (input, output), as in eval_phase5.py.
PRICE = (0.10, 0.50)

#: (site, account, games, level). maia1 is Lichess's human-like ~1100 bot: realistic mistakes.
SOURCES = [
    ("lichess", "maia1", 3, "beginner"),
    ("lichess", "maia5", 2, "intermediate"),
    ("chess.com", "hikaru", 2, "expert"),
]

SAMPLE = """[Event "sample: beginner blunders"]
[White "me"]
[Black "opponent"]
[Result "0-1"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Nd4 4. Nxe5 Qg5 5. Nxf7 Qxg2 6. Rf1 Qxe4+ 7. Be2 Nf3# 0-1

[Event "sample: hanging pieces"]
[White "opponent"]
[Black "me"]
[Result "1-0"]

1. e4 e5 2. Nf3 d6 3. Bc4 Bg4 4. Nc3 g6 5. Nxe5 Bxd1 6. Bxf7+ Ke7 7. Nd5# 1-0

[Event "sample: a quiet London (no beginner-level mistakes expected)"]
[White "me"]
[Black "opponent"]
[Result "*"]

1. d4 d5 2. Bf4 Nf6 3. e3 Bf5 4. Nf3 e6 5. Bd3 Bxd3 6. Qxd3 Nc6 7. O-O Nb4 8. Qe2 Bd6
9. Bxd6 Qxd6 10. c3 Nc6 11. Nbd2 O-O 12. e4 dxe4 13. Nxe4 Nxe4 14. Qxe4 Qd5 15. Qe3 Rfe8 *
"""


def load_games(
    args: argparse.Namespace,
) -> tuple[list[tuple[GameRecord, chess.Color, str]], list[str]]:
    games: list[tuple[GameRecord, chess.Color, str]] = []
    notes: list[str] = []
    for g in parse_pgn(SAMPLE):
        color = g.color_of(["me"])
        assert color is not None
        games.append((g, color, "beginner"))
    notes.append(f"sample PGN: {len(games)} games")
    if args.offline:
        return games, notes
    for site, account, n, level in SOURCES:
        fetch = fetch_lichess if site == "lichess" else fetch_chesscom
        t0 = time.perf_counter()
        try:
            fetched = fetch(account, n)
        except FetchError as err:
            notes.append(f"{site} {account}: FAILED ({err})")
            continue
        ok = [g for g in fetched if g.user_color is not None]
        notes.append(
            f"{site} {account}: {len(fetched)} games in {time.perf_counter() - t0:.1f} s, "
            f"player side found in {len(ok)}"
        )
        games += [(g, g.user_color, level) for g in ok if g.user_color is not None]
    return games, notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-explain", action="store_true")
    parser.add_argument("--offline", action="store_true", help="sample PGN only (no fetching)")
    parser.add_argument("--budget", type=float, default=0.5, help="max API dollars")
    parser.add_argument("--depth", type=int, default=14)
    args = parser.parse_args()

    path = find_stockfish(None).path
    if path is None:
        sys.exit("Stockfish not found")
    games, notes = load_games(args)
    ev = StockfishEvaluator(path, depth=args.depth, max_seconds=0.5)
    explainer = None
    if not args.no_explain:
        explainer = Explainer(
            "claude-haiku-5-5", "medium", cache=ExplanationCache(OUT / "phase6_cache.sqlite")
        )
        if not explainer.available:
            notes.append("explanations skipped: " + explainer.unavailable_reason)
            explainer = None

    lines = ["Phase 6 review-mode eval (real Stockfish, no screen use)", *notes, ""]
    rows: list[dict[str, object]] = []
    reviews: dict[str, list[GameReview]] = {}
    spent = 0.0
    outcomes: dict[str, int] = {}
    latencies: list[float] = []
    review_times: list[float] = []
    for game, color, level_key in games:
        level = get_level(level_key)
        t0 = time.perf_counter()
        r = review_game(game, color, level, ev)
        secs = time.perf_counter() - t0
        review_times.append(secs / max(1, len(game.moves) + 1))
        reviews.setdefault(level_key, []).append(r)
        side = "White" if color == chess.WHITE else "Black"
        lines.append(
            f"[{game.source}] {game.title()}  (player: {side}, {level.name}) reviewed in "
            f"{secs:.1f} s"
        )
        for m in r.flagged():
            kind = m.mistake.label if m.mistake else "?"
            lines.append(
                f"   {m.move_number} {m.san}: {m.severity}, -{m.loss / 100:.1f} (best "
                f"{m.best_san}); {kind}: {m.mistake.detail if m.mistake else ''}"
            )
        for m in r.biggest(3):
            row: dict[str, object] = {
                "game": game.title(),
                "level": level_key,
                "move": f"{m.move_number} {m.san}",
                "best": m.best_san,
                "loss": m.loss,
                "severity": m.severity,
                "kind": m.mistake.kind if m.mistake else None,
            }
            if explainer is not None and spent < args.budget:
                facts = review_facts(m, level)
                try:
                    res = explainer.explain(facts, level)
                except ExplainError as err:
                    row["error"] = str(err)
                else:
                    cost = (res.input_tokens * PRICE[0] + res.output_tokens * PRICE[1]) / 1e6
                    spent += cost
                    if not res.cached:
                        outcomes[res.outcome] = outcomes.get(res.outcome, 0) + 1
                        latencies.append(res.seconds)
                    row.update(text=res.text, outcome=res.outcome, cached=res.cached)
                    lines.append(f"   explained {m.san} [{res.outcome}]: {res.text}")
            rows.append(row)
        lines.append("")
    ev.close()

    lines.append("== Summaries per level ==")
    for level_key, rs in reviews.items():
        lines.append(f"-- {get_level(level_key).name} ({len(rs)} games)")
        lines.append(summary_text(summarize(rs), get_level(level_key)))
        lines.append("")
    lines.append(
        f"engine time per position: median {sorted(review_times)[len(review_times) // 2]:.2f} s"
        if review_times
        else "no games"
    )
    if latencies:
        lat = sorted(latencies)
        lines.append(
            f"explanations: {sum(outcomes.values())} new, grounding {outcomes}; latency median "
            f"{lat[len(lat) // 2]:.1f} s, max {lat[-1]:.1f} s"
        )
    lines.append(f"API spend this run: ${spent:.4f} (budget ${args.budget:.2f})")
    text = "\n".join(lines)
    print(text)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phase6_report.txt").write_text(text + "\n", encoding="utf-8")
    with (OUT / "phase6_reviews.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
