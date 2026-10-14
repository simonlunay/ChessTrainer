"""Phase 5 scripted test: the opening trainer on full lines, with deviations, off screen.

    python -m uv run python scripts/eval_phase5.py [--no-explain] [--budget 1.0]

Drives the real :class:`TrainerApp` with ``QT_QPA_PLATFORM=offscreen`` (no window, no input, no
screen capture): each scenario feeds the positions of a scripted game to the app exactly as the
board reader would commit them, waits for real Stockfish results (and, unless ``--no-explain``,
real Claude explanations through the live path: grounding check, one retry, facts fallback),
and records what the panel and overlay would show. A token ``<`` takes the last move back.

Every scenario lists the trainer events it must produce; the report says which matched.
Spend is counted from the API's token usage (Haiku 5.5: $0.10 / $0.50 per million tokens
in / out) and explanations stop when ``--budget`` dollars are reached.
Writes ``data/eval/phase5_report.txt`` and ``data/eval/phase5_steps.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import chess  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from chesstrainer.explain.explainer import Explanation  # noqa: E402
from chesstrainer.openings.book import default_book  # noqa: E402
from chesstrainer.openings.practice import PracticeStore  # noqa: E402
from chesstrainer.openings.trainer import TrainerEvent  # noqa: E402
from chesstrainer.settings.config import Settings  # noqa: E402
from chesstrainer.vision.position import PositionReading  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "eval"
PRICE_IN, PRICE_OUT = 0.10 / 1e6, 0.50 / 1e6  # claude-haiku-5-5, $ per token


@dataclass
class Scenario:
    name: str
    opening: str
    side: str
    level: str
    moves: str  # SAN from the start; "<" = take back one move
    expect: list[str]  # trainer event kinds, in order
    mode: str = "guided"
    expect_grade: str = ""  # substring of the grade (move that left the book)


SCENARIOS = [
    Scenario(
        "Italian (W) main line to the beginner depth",
        "Italian Game",
        "white",
        "beginner",
        "e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6",
        ["start"] + ["user_correct", "opp_book"] * 4 + ["complete"],
    ),
    Scenario(
        "Italian (W): Black blunders the queen out of book",
        "Italian Game",
        "white",
        "beginner",
        "e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Qh4",
        ["start"] + ["user_correct", "opp_book"] * 3 + ["user_correct", "opp_left"],
        expect_grade="Qh4 is a blunder",
    ),
    Scenario(
        "Italian (W): Black plays a quiet non-book move",
        "Italian Game",
        "white",
        "novice",
        "e4 e5 Nf3 Nc6 Bc4 Bc5 c3 h6",
        ["start"] + ["user_correct", "opp_book"] * 3 + ["user_correct", "opp_left"],
        expect_grade="Stockfish: h6",
    ),
    Scenario(
        "Italian (W): Black answers 1.e4 with the Sicilian",
        "Italian Game",
        "white",
        "beginner",
        "e4 c5",
        ["start", "user_correct", "opp_other_opening"],
    ),
    Scenario(
        "London (W): transposition 1.d4 Nf6 2.Bf4 d5",
        "London System",
        "white",
        "beginner",
        "d4 Nf6 Bf4 d5 e3 e6 Nf3 c5",
        ["start"] + ["user_correct", "opp_book"] * 4 + ["complete"],
    ),
    Scenario(
        "Scotch (W): user plays another book move, takes it back, plays the line (9 plies)",
        "Scotch Game",
        "white",
        "novice",
        "e4 e5 Nf3 Nc6 d4 exd4 c3 < Nxd4 Nf6 Nxc6",
        ["start"]
        + ["user_correct", "opp_book"] * 3
        + ["user_other", "takeback", "user_correct", "opp_book", "user_correct"],
    ),
    Scenario(
        "Scotch (W): user leaves the book with a bad move",
        "Scotch Game",
        "white",
        "beginner",
        "e4 e5 Nf3 Nc6 d4 exd4 Qxd4",
        ["start"] + ["user_correct", "opp_book"] * 3 + ["user_left"],
        expect_grade="Stockfish: Qxd4",
    ),
    Scenario(
        "Italian (B) main line",
        "Italian Game",
        "black",
        "beginner",
        "e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6",
        ["start"] + ["opp_book", "user_correct"] * 4 + ["complete"],
    ),
    Scenario(
        "Italian (B): White plays the Ruy Lopez",
        "Italian Game",
        "black",
        "beginner",
        "e4 e5 Nf3 Nc6 Bb5",
        ["start"] + ["opp_book", "user_correct"] * 2 + ["opp_other_opening"],
    ),
    Scenario(
        "Scandinavian (B) main line",
        "Scandinavian Defense",
        "black",
        "beginner",
        "e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6",
        ["start"] + ["opp_book", "user_correct"] * 4 + ["complete"],
    ),
    Scenario(
        "Scandinavian (B): White leaves the book early",
        "Scandinavian Defense",
        "black",
        "beginner",
        "e4 d5 exd5 Qxd5 Qf3",
        ["start", "opp_book", "user_correct", "opp_book", "user_correct", "opp_left"],
        expect_grade="Stockfish: Qf3",
    ),
    Scenario(
        "QGD (B) main line",
        "Queen's Gambit Declined",
        "black",
        "beginner",
        "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7",
        ["start"] + ["opp_book", "user_correct"] * 4 + ["complete"],
    ),
    Scenario(
        "QGD (B): White plays the London",
        "Queen's Gambit Declined",
        "black",
        "beginner",
        "d4 d5 Bf4",
        ["start", "opp_book", "user_correct", "opp_other_opening"],
    ),
    Scenario(
        "Najdorf (B, dataset only, Expert): book runs out before the depth",
        "Sicilian Defense: Najdorf Variation",
        "black",
        "expert",
        "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nb3 Be6 f3 Be7 Qd2 O-O O-O-O Nbd7 g4 b5",
        # The dataset's English Attack line ends after 8.f3 (15 plies), before the depth of 24.
        ["start"] + ["opp_book", "user_correct"] * 7 + ["opp_book", "complete"],
    ),
    Scenario(
        "Practice: Italian (W), a miss, takeback, then correct",
        "Italian Game",
        "white",
        "beginner",
        "e4 e5 Nf3 Nc6 d4 < Bc4 Bc5 c3 Nf6",
        ["start"]
        + ["user_correct", "opp_book"] * 2
        + ["user_other", "takeback", "user_correct", "opp_book", "user_correct", "opp_book"]
        + ["complete"],
        mode="practice",
    ),
]


@dataclass
class Step:
    scenario: str
    ply: int
    played: str
    fen: str
    events: list[dict[str, Any]] = field(default_factory=list)
    arrow: str = ""
    prompt: str = ""
    variation: str = ""
    message: str = ""
    engine: str = ""
    explanation: str = ""
    outcome: str = ""
    issues: list[str] = field(default_factory=list)
    explain_s: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    error: str = ""


def wait(cond: Any, seconds: float) -> bool:
    deadline = time.time() + seconds
    while not cond() and time.time() < deadline:
        QApplication.processEvents()
        time.sleep(0.005)
    QApplication.processEvents()
    return bool(cond())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-explain", action="store_true", help="no API calls")
    parser.add_argument("--budget", type=float, default=1.0, help="max API spend in dollars")
    parser.add_argument("--engine-seconds", type=float, default=2.0)
    args = parser.parse_args()

    home = Path(tempfile.mkdtemp(prefix="ct_phase5_"))
    os.environ["CHESSTRAINER_HOME"] = str(home)  # fresh caches: every explanation is a real call
    from chesstrainer.overlay.app import TrainerApp

    app = QApplication.instance() or QApplication(sys.argv[:1])
    assert app is not None
    t_load = time.perf_counter()
    book = default_book()
    load_s = time.perf_counter() - t_load
    store = PracticeStore(home / "openings.sqlite")
    settings = Settings(
        level_chosen=True,
        explain_delay_s=0.0,
        engine_max_seconds=args.engine_seconds,
        explanations=not args.no_explain,
    )
    trainer = TrainerApp(settings, book=book, practice_store=store)
    if trainer.engine is None:
        sys.exit("Stockfish not found")
    trainer.engine.start()
    trainer.engine.wait_ready()
    if trainer.explainer is not None and not trainer.explainer.available:
        print(trainer.explainer.unavailable_reason)

    captured: list[TrainerEvent] = []
    original = trainer._trainer_events

    def spy(events: list[TrainerEvent], previous: Any) -> None:
        captured.extend(events)
        original(events, previous)

    trainer._trainer_events = spy  # type: ignore[method-assign]

    steps: list[Step] = []
    results: list[tuple[Scenario, list[str], bool, str]] = []
    spend = 0.0
    explain_errors: Counter[str] = Counter()
    for sc in SCENARIOS:
        trainer.set_level(sc.level)
        trainer.set_opening(sc.opening, sc.side, sc.mode)
        board = chess.Board()
        kinds: list[str] = []
        grade_text = ""
        tokens = ["start"] + sc.moves.split()
        for i, tok in enumerate(tokens):
            if tok == "<":
                board.pop()
                played = "<"
            elif tok != "start":
                played = board.san(board.parse_san(tok))
                board.push_san(tok)
            else:
                played = "start"
            captured.clear()
            n_expl = len(trainer.explanations)
            trainer._new_position(PositionReading(board.copy(stack=False), [], 1.0, "history"))
            step = Step(sc.name, i, played, board.fen())
            step.events = [{"kind": e.kind, "message": e.message} for e in captured]
            kinds += [e.kind for e in captured]
            final = wait(
                lambda: (
                    trainer.analysis is not None
                    and trainer.board is not None
                    and trainer.analysis.fen == trainer.board.fen()
                    and trainer.analysis.final
                ),
                args.engine_seconds + 10,
            )
            a = trainer.analysis
            if a is not None and final and trainer.board is not None:
                b = trainer.board
                step.engine = (
                    f"{'book move only' if trainer._root else 'MultiPV'}: "
                    f"{b.san(a.best.move)} {a.best.score_text()} (depth {a.depth})"
                )
            if trainer.explainer is not None and trainer.explainer.available:
                over = spend >= args.budget
                if over and trainer.settings.explanations:
                    print(f"budget ${args.budget:.2f} reached: explanations off")
                    trainer.set_explanations(False)
                elif not over:
                    t0 = time.perf_counter()
                    wait(lambda: trainer.explain_req is None, 60)
                    ex: Explanation | None = (
                        trainer.explanations[-1][2] if len(trainer.explanations) > n_expl else None
                    )
                    if ex is not None:
                        step.explanation, step.outcome = ex.text, ex.outcome
                        step.issues = list(ex.issues)
                        step.explain_s = ex.seconds or (time.perf_counter() - t0)
                        step.tokens_in, step.tokens_out = ex.input_tokens, ex.output_tokens
                        spend += ex.input_tokens * PRICE_IN + ex.output_tokens * PRICE_OUT
                    else:
                        text = trainer.panel.explanation.text()
                        if text and trainer.panel.explanation.styleSheet():
                            step.error = text
                            explain_errors[text[:80]] += 1
            arrow = trainer._book_arrow()
            if isinstance(arrow, chess.Move) and trainer.board is not None:
                step.arrow = f"book {trainer.board.san(arrow)}"
            elif arrow is False and trainer.overlay.arrows:
                step.arrow = "engine " + " ".join(
                    chess.square_name(x.from_sq) + chess.square_name(x.to_sq)
                    for x in trainer.overlay.arrows
                )
            panel = trainer.panel
            step.prompt = panel.trainer_prompt.text()
            step.variation = panel.trainer_variation.text()
            step.message = panel.trainer_message.text()
            if "Stockfish:" in step.message:
                grade_text = step.message
            steps.append(step)
        ok = kinds == sc.expect and (not sc.expect_grade or sc.expect_grade in grade_text)
        results.append((sc, kinds, ok, grade_text))
        print(f"{'PASS' if ok else 'FAIL'}  {sc.name}")

    if trainer.explain_worker is not None:
        trainer.explain_worker.close()
    trainer.engine.close()
    write_report(steps, results, spend, load_s, explain_errors, store, args)


def write_report(
    steps: list[Step],
    results: list[tuple[Scenario, list[str], bool, str]],
    spend: float,
    load_s: float,
    explain_errors: Counter[str],
    store: PracticeStore,
    args: argparse.Namespace,
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "phase5_steps.jsonl").open("w", encoding="utf-8") as f:
        for s in steps:
            f.write(json.dumps(s.__dict__, ensure_ascii=False) + "\n")
    explained = [s for s in steps if s.explanation]
    outcomes = Counter(s.outcome for s in explained)
    book_expl = [s for s in explained if s.engine.startswith("book")]
    lines = [
        "Phase 5 scripted opening-trainer test (offscreen, real Stockfish"
        + ("" if args.no_explain else ", real Claude explanations")
        + ")",
        f"Book: {len(default_book().entries)} Lichess lines, loaded in {load_s:.1f} s.",
        "",
        f"Scenarios: {sum(ok for *_, ok, _ in results)}/{len(results)} produced exactly the "
        "expected trainer events (and grade, where one is expected).",
    ]
    for sc, kinds, ok, grade in results:
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {sc.name} ({sc.level}, {sc.mode})")
        if not ok:
            lines.append(f"         expected {sc.expect}")
            lines.append(f"         got      {kinds}")
        if grade:
            g = grade[grade.index("Stockfish:") :]
            lines.append(f"         grade: {g}")
    lines += ["", f"Event kinds: {dict(Counter(e['kind'] for s in steps for e in s.events))}"]
    if not args.no_explain:
        times = sorted(s.explain_s for s in explained)
        median = times[len(times) // 2] if times else 0.0
        lines += [
            "",
            f"Explanations: {len(explained)} ({len(book_expl)} of book moves with opening facts)",
            f"  grounding check: {outcomes['passed']} passed first time, "
            f"{outcomes['regenerated']} regenerated once, {outcomes['fallback']} shown as facts",
            f"  latency median {median:.1f} s, max {max(times, default=0):.1f} s",
            f"  tokens in {sum(s.tokens_in for s in steps)},"
            f" out {sum(s.tokens_out for s in steps)}",
            f"  API spend this run: ${spend:.4f} (budget ${args.budget:.2f})",
        ]
        for msg, n in explain_errors.items():
            lines.append(f"  explanation errors x{n}: {msg}")
    reviews = store.cards("Italian Game", "white")
    lines += [
        "",
        f"Practice store after the run: {len(reviews)} Italian (W) cards, "
        f"{sum(c.lapses for c in reviews.values())} lapses.",
    ]
    lines += ["", "Steps (ply, played, events, overlay arrow, engine, explanation):"]
    current = ""
    for s in steps:
        if s.scenario != current:
            current = s.scenario
            lines += ["", f"== {current}"]
        ev = " | ".join(f"{e['kind']}: {e['message']}" for e in s.events) or "-"
        lines.append(f"  {s.ply:>2} {s.played:<6} {ev}")
        lines.append(f"       arrow: {s.arrow or 'none'}; {s.engine}; prompt: {s.prompt}")
        if s.variation:
            lines.append(f"       variation: {s.variation}")
        if s.explanation:
            lines.append(f"       [{s.outcome}] {s.explanation}")
            for issue in s.issues:
                lines.append(f"         flagged: {issue}")
        if s.error:
            lines.append(f"       explanation error: {s.error}")
    (OUT / "phase5_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
