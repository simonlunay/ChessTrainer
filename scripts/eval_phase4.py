"""Phase 4 evaluation: are the explanations grounded in the computed facts?

    python -m uv run python scripts/eval_phase4.py [--judge-model M] [--workers N]

Positions: the curated ones in ``scripts/phase4_positions.yaml`` plus the 5 distinct positions
in the user's real screenshots (read from the images with the phase-2 board reader). Each
position is analysed once by Stockfish (depth 18, MultiPV 3), then for every skill level the
app's own pipeline runs: practical-move choice -> facts -> explanation (the configured
explanation model, uncached). Every explanation is checked twice against its facts:

  rules  ``explain/grounding.py``: squares, moves, piece placements, concepts, numbers and
         evaluation words that the facts do not contain (strict, literal)
  judge  a stronger Claude model lists the explanation's factual claims and marks each one
         supported or not by the facts (catches paraphrased inventions)

Hallucination rate = share of explanations with at least one unsupported claim.
Writes ``data/eval/phase4_report.txt`` and ``data/eval/phase4_explanations.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import anthropic
import chess
import chess.engine
import numpy as np
import yaml
from PIL import Image

from chesstrainer.engine.analysis import Line
from chesstrainer.engine.locate import find_stockfish
from chesstrainer.explain.explainer import Explainer
from chesstrainer.explain.facts import MoveFacts, compute_facts
from chesstrainer.explain.grounding import check_grounding
from chesstrainer.explain.practical import suggest
from chesstrainer.levels import Level, all_levels
from chesstrainer.settings.config import Settings
from chesstrainer.settings.env import anthropic_api_key
from chesstrainer.vision.evaluate import REAL_DIR
from chesstrainer.vision.reader import BoardReader, read_image

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "eval"
POSITIONS = Path(__file__).with_name("phase4_positions.yaml")
SCREENSHOTS = [  # one per distinct position in the user's screenshots
    ("Screenshot 2026-10-13 165018.png", "real: starting position"),
    ("Screenshot 2026-10-13 165030.png", "real: Italian Game"),
    ("Screenshot 2026-10-13 165040.png", "real: Ruy Lopez, castled"),
    ("Screenshot 2026-10-13 165051.png", "real: queen endgame"),
    ("Screenshot 2026-10-13 165100.png", "real: rook endgame"),
]
# $ per million tokens (input, output), for the cost line of the report.
SENTENCE_END = re.compile(r"(?<=[^\d.][.!?])\s+(?=[A-Z])")
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-opus-5-5": (4.0, 20.0)}

JUDGE_SYSTEM = """\
You are a strict fact-checker for a chess training app. An explanation of a chess move was
written from a list of FACTS (computed by a chess engine and a rules library). The writer could
not see the board, so anything not in the facts is invented.

Split the EXPLANATION into its individual factual claims about this position: moves, pieces,
squares, attacks, threats, tactics, material, evaluation, plans and consequences. For each
claim, decide:
- supported = true if a fact states it, or it follows directly and unambiguously from the facts
  (restating, combining two facts, or a plain consequence such as "wins material" from a fact
  that says material is won).
- supported = false if it adds anything the facts do not say (another move, square, piece,
  threat, plan, opening name, evaluation strength) or contradicts them, even slightly.
Do not list general definitions of chess terms ("a fork is when one piece attacks two") or
advice/encouragement that makes no claim about this position.
"""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "supported": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["claim", "supported", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}


@dataclass
class Position:
    name: str
    theme: str
    fen: str
    real: bool = False
    turn_assumed: bool = False
    lines: tuple[Line, ...] = ()


@dataclass
class Result:
    position: str
    real: bool
    level: str
    fen: str
    move: str
    practical: bool
    facts: str
    text: str = ""
    error: str = ""
    seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    rule_issues: list[str] = field(default_factory=list)
    claims: list[dict[str, Any]] = field(default_factory=list)
    judge_error: str = ""
    judge_tokens: tuple[int, int] = (0, 0)

    @property
    def unsupported(self) -> list[dict[str, Any]]:
        return [c for c in self.claims if not c["supported"]]

    @property
    def sentences(self) -> int:
        # A sentence ends with . ! or ? before a capital letter; move numbers ("45. b6",
        # "3...d6") do not end sentences.
        return len([s for s in SENTENCE_END.split(self.text.strip()) if s])


def load_positions() -> list[Position]:
    data = yaml.safe_load(POSITIONS.read_text(encoding="utf-8"))
    out = [Position(p["name"], p["theme"], p["fen"]) for p in data["positions"]]
    reader_settings = Settings()
    for file, name in SCREENSHOTS:
        image = np.asarray(Image.open(REAL_DIR / file).convert("RGB"), np.uint8)
        state = read_image(image, BoardReader(reader_settings))
        if state.reading is None:
            print(f"could not read {file}", file=sys.stderr)
            continue
        r = state.reading
        out.append(Position(name, "real screenshot", r.fen, True, r.turn_assumed))
    return out


def analyse(positions: list[Position], depth: int) -> None:
    search = find_stockfish(None)
    if search.path is None:
        sys.exit(search.setup_message())
    engine = chess.engine.SimpleEngine.popen_uci(str(search.path))
    engine.configure({"Threads": 4, "Hash": 128})
    try:
        for p in positions:
            board = chess.Board(p.fen)
            infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=3)
            p.lines = tuple(
                Line(i + 1, tuple(x["pv"]), x["score"].white().score(), x["score"].white().mate())
                for i, x in enumerate(infos)
            )
    finally:
        engine.quit()


def make_facts(p: Position, level: Level) -> tuple[MoveFacts, bool]:
    """Exactly what the app does (``TrainerApp._maybe_explain``)."""
    board = chess.Board(p.fen)
    sug = suggest(board, p.lines, level)
    facts = compute_facts(
        board,
        sug.line,
        level,
        engine_best=sug.engine_best if sug.practical else None,
        alternatives=sug.lines[1:] if level.explanation.eval_numbers else (),
    )
    return facts, sug.practical


def judge(client: anthropic.Anthropic, model: str, r: Result) -> None:
    user = f"FACTS:\n{r.facts}\n\nEXPLANATION:\n{r.text}\n"
    try:
        response = client.messages.create(
            model=model,
            max_tokens=8000,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": user}],
            output_config={
                "effort": "medium",
                "format": {"type": "json_schema", "schema": JUDGE_SCHEMA},
            },
        )
    except anthropic.APIError as err:
        r.judge_error = str(err)
        return
    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        r.claims = json.loads(text)["claims"]
    except (json.JSONDecodeError, KeyError) as err:
        r.judge_error = f"unparseable judge output ({err}): {text[:200]}"
    r.judge_tokens = (response.usage.input_tokens, response.usage.output_tokens)


def run_one(
    explainer: Explainer, client: anthropic.Anthropic, judge_model: str, p: Position, level: Level
) -> Result:
    facts, practical = make_facts(p, level)
    r = Result(p.name, p.real, level.key, p.fen, facts.san, practical, facts.text())
    try:
        ex = explainer.explain(facts, level)
    except Exception as err:  # noqa: BLE001 - recorded in the report
        r.error = str(err)
        return r
    r.text, r.seconds = ex.text, ex.seconds
    r.input_tokens, r.output_tokens = ex.input_tokens, ex.output_tokens
    r.rule_issues = [f"{i.kind}: {i.claim} ({i.detail})" for i in check_grounding(ex.text, facts)]
    judge(client, judge_model, r)
    return r


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "0/0"


def report(results: list[Result], positions: list[Position], model: str, judge_model: str) -> str:
    ok = [r for r in results if not r.error]
    lines = [
        "Phase 4 explanation eval",
        f"{time.strftime('%Y-%m-%d %H:%M')}  explainer: {model}  judge: {judge_model}",
        f"positions: {len(positions)} ({sum(p.real for p in positions)} read from the user's "
        f"screenshots), levels: {len(all_levels())}, explanations: {len(results)} "
        f"({len(results) - len(ok)} failed)",
        "",
        "Hallucination rate = explanations with >= 1 claim not supported by the facts.",
        "rules = literal checker (explain/grounding.py); judge = LLM claim-by-claim check.",
        "",
        f"{'level':13} {'n':>3}  {'rules flag':>12}  {'judge flag':>12}  {'either':>12}  "
        f"{'claims unsupp.':>15}  {'sent.':>5}  {'latency med/p90':>15}",
    ]
    groups: list[tuple[str, list[Result]]] = [
        (lv.key, [r for r in ok if r.level == lv.key]) for lv in all_levels()
    ]
    groups.append(("ALL", ok))
    for name, rs in groups:
        n = len(rs)
        rules = sum(bool(r.rule_issues) for r in rs)
        judged = [r for r in rs if not r.judge_error]
        jflag = sum(bool(r.unsupported) for r in judged)
        either = sum(bool(r.rule_issues) or bool(r.unsupported) for r in rs)
        claims = sum(len(r.claims) for r in judged)
        unsupp = sum(len(r.unsupported) for r in judged)
        secs = sorted(r.seconds for r in rs)
        p90 = secs[min(len(secs) - 1, int(0.9 * len(secs)))] if secs else 0.0
        sent = statistics.mean(r.sentences for r in rs) if rs else 0.0
        lines.append(
            f"{name:13} {n:>3}  {pct(rules, n):>12}  {pct(jflag, len(judged)):>12}  "
            f"{pct(either, n):>12}  {pct(unsupp, claims):>15}  {sent:>5.1f}  "
            f"{statistics.median(secs) if secs else 0:>6.2f}s/{p90:.2f}s"
        )
    real = [r for r in ok if r.real]
    real_judge = pct(sum(bool(r.unsupported) for r in real), len(real))
    real_rules = pct(sum(bool(r.rule_issues) for r in real), len(real))
    lines.append(f"real screenshots only: judge flag {real_judge}, rules flag {real_rules}")
    prac = [r for r in ok if r.practical]
    lines.append(f"practical-move explanations (Beginner/Novice): {len(prac)}")
    tin = sum(r.input_tokens for r in ok)
    tout = sum(r.output_tokens for r in ok)
    pin, pout = PRICES.get(model.split()[0], (0.0, 0.0))
    lines.append(
        f"explainer tokens: {tin} in / {tout} out = ${(tin * pin + tout * pout) / 1e6:.4f} total, "
        f"{tin / max(1, len(ok)):.0f}/{tout / max(1, len(ok)):.0f} per explanation"
    )
    jin = sum(r.judge_tokens[0] for r in ok)
    jout = sum(r.judge_tokens[1] for r in ok)
    jpi, jpo = PRICES.get(judge_model, (0.0, 0.0))
    lines.append(f"judge tokens: {jin} in / {jout} out = ${(jin * jpi + jout * jpo) / 1e6:.2f}")
    if len(ok) != len(results):
        lines += ["", "FAILED:"] + [
            f"  {r.position} [{r.level}]: {r.error}" for r in results if r.error
        ]

    lines += ["", "=" * 100, "Every flagged explanation", "=" * 100]
    for r in ok:
        if not (r.rule_issues or r.unsupported or r.judge_error):
            continue
        lines += ["", f"[{r.level}] {r.position}: {r.move}", f"  text: {r.text}"]
        lines += [f"  rules: {i}" for i in r.rule_issues]
        lines += [f"  judge: {c['claim']}  -- {c['reason']}" for c in r.unsupported]
        if r.judge_error:
            lines.append(f"  judge error: {r.judge_error}")

    lines += ["", "=" * 100, "Samples (3 per level)", "=" * 100]
    sample_names = [
        "knight fork with check",
        "real: Ruy Lopez, castled",
        "Queen's Gambit, Black to move",
    ]
    for lv in all_levels():
        lines += ["", f"--- {lv.label} ---"]
        for name in sample_names:
            r = next((x for x in ok if x.level == lv.key and x.position == name), None)
            if r is None:
                continue
            flag = " [flagged]" if r.rule_issues or r.unsupported else ""
            note = " (practical move)" if r.practical else ""
            lines += [f"* {r.position}: {r.move}{note}{flag}", f"  {r.text}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default=Settings().explain_model)
    parser.add_argument("--effort", default=Settings().explain_effort)
    parser.add_argument("--judge-model", default="claude-opus-5-5")
    parser.add_argument("--depth", type=int, default=18)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    key = anthropic_api_key()
    if not key:
        sys.exit("ANTHROPIC_API_KEY is not set (add it to .env).")

    positions = load_positions()
    print(f"{len(positions)} positions; analysing with Stockfish (depth {args.depth})...")
    analyse(positions, args.depth)
    explainer = Explainer(args.model, args.effort, api_key=key, cache=None)
    client = anthropic.Anthropic(api_key=key, timeout=120.0, max_retries=3)
    jobs = [(p, lv) for p in positions for lv in all_levels()]
    print(f"{len(jobs)} explanations + judge calls...")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda j: run_one(explainer, client, args.judge_model, *j), jobs))
    print(f"done in {time.perf_counter() - t0:.0f}s")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "phase4_explanations.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")
    text = report(results, positions, f"{args.model} (effort {args.effort})", args.judge_model)
    (OUT_DIR / "phase4_report.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
