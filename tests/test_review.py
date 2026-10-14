"""Game review: PGN parsing, Lichess / Chess.com fetching (fake HTTP), engine pass, mistake
types, summary, explanation facts and the background runner (fake engine and Claude client)."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import chess
import pytest

from chesstrainer.engine.analysis import Line
from chesstrainer.engine.locate import find_stockfish
from chesstrainer.explain import board_tools as bt
from chesstrainer.explain.explainer import Explainer, ExplanationCache
from chesstrainer.explain.grounding import check_grounding
from chesstrainer.levels import get_level
from chesstrainer.review.analysis import StockfishEvaluator, review_game, score_for
from chesstrainer.review.games import FetchError, fetch_chesscom, fetch_lichess, parse_pgn
from chesstrainer.review.mistakes import classify_mistake, count_kinds
from chesstrainer.review.report import review_facts, summarize, summary_text
from chesstrainer.review.runner import ReviewCallbacks, ReviewRunner

HUNG_QUEEN = '[White "opp"]\n[Black "me"]\n[Result "1-0"]\n\n1. e4 e5 2. Nf3 Qg5 3. Nxg5 1-0\n'
SCHOLARS = '[White "opp"]\n[Black "me"]\n\n1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0\n'


class MaterialEvaluator:
    """Fake engine: plays the best winning capture (else the first legal move); score = material
    after that capture, in centipawns from White's view."""

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, board: chess.Board) -> Line:
        self.calls += 1
        caps = bt.winning_captures(board)
        move = caps[0][0] if caps else next(iter(board.legal_moves))
        gain = caps[0][1] if caps else 0
        sign = 1 if board.turn == chess.WHITE else -1
        return Line(1, (move,), (bt.material_diff(board) + sign * gain) * 100, None)

    def close(self) -> None:
        pass


# -- PGN and fetching ----------------------------------------------------------------------------


def test_parse_pgn_keeps_standard_games_only() -> None:
    text = (
        HUNG_QUEEN
        + "\n"
        + '[Variant "Chess960"]\n\n1. e4 *\n\n'
        + SCHOLARS
        + '\n[White "x"]\n\n*\n'
    )
    games = parse_pgn(text)
    assert [len(g.moves) for g in games] == [5, 7]
    assert games[0].color_of(["ME"]) == chess.BLACK
    assert games[0].color_of(["nobody"]) is None
    assert "opp vs me" in games[0].title()


def test_fetch_lichess_builds_the_url_and_finds_the_player() -> None:
    seen: list[tuple[str, str]] = []

    def opener(url: str, accept: str) -> str:
        seen.append((url, accept))
        return HUNG_QUEEN + "\n" + SCHOLARS

    games = fetch_lichess("Me", 5, opener)
    assert seen == [
        (
            "https://lichess.org/api/games/user/Me?max=5&moves=true&tags=true&clocks=false",
            "application/x-chess-pgn",
        )
    ]
    assert [g.source for g in games] == ["lichess", "lichess"]
    assert all(g.user_color == chess.BLACK for g in games)


def test_fetch_chesscom_walks_archives_newest_first() -> None:
    archives = [
        "https://api.chess.com/pub/player/me/games/2026/08",
        "https://api.chess.com/pub/player/me/games/2026/09",
    ]
    months = {
        archives[0]: {"games": [{"rules": "chess", "pgn": SCHOLARS, "url": "u-old"}]},
        archives[1]: {
            "games": [
                {"rules": "chess", "pgn": HUNG_QUEEN, "url": "u1"},
                {"rules": "chess960", "pgn": HUNG_QUEEN},
                {"rules": "chess", "pgn": SCHOLARS, "url": "u2"},
            ]
        },
    }

    def opener(url: str, accept: str) -> str:
        if url.endswith("/archives"):
            assert url == "https://api.chess.com/pub/player/me/games/archives"
            return json.dumps({"archives": archives})
        return json.dumps(months[url])

    games = fetch_chesscom("ME", 2, opener)
    assert [g.headers["Link"] for g in games] == ["u2", "u1"]  # newest month, newest game first
    assert all(g.user_color == chess.BLACK and g.source == "chess.com" for g in games)
    assert [g.headers["Link"] for g in fetch_chesscom("me", 10, opener)] == ["u2", "u1", "u-old"]


def test_fetch_errors_are_readable() -> None:
    def not_found(url: str, accept: str) -> str:
        raise FetchError("not_found")

    with pytest.raises(FetchError, match="no user called 'ghost'"):
        fetch_lichess("ghost", 5, not_found)
    with pytest.raises(FetchError, match="Chess.com has no user"):
        fetch_chesscom("ghost", 5, not_found)
    with pytest.raises(FetchError, match="not a valid username"):
        fetch_lichess("../admin?x=1", 5, not_found)


# -- engine pass and mistake types ---------------------------------------------------------------


def test_review_game_flags_the_hung_queen() -> None:
    game = parse_pgn(HUNG_QUEEN)[0]
    ev = MaterialEvaluator()
    review = review_game(game, chess.BLACK, get_level("beginner"), ev)
    assert ev.calls == 6  # start position + one per move
    flagged = review.flagged()
    assert [m.san for m in flagged] == ["Qg5"]
    m = flagged[0]
    assert m.severity == "blunder" and m.loss == 900
    assert m.mistake is not None and m.mistake.kind == "hung_piece"
    assert "Black's queen on g5" in m.mistake.detail and "Nxg5" in m.mistake.detail
    assert [x.by_user for x in review.moves] == [False, True, False, True, False]


def test_score_is_clamped_and_game_over_is_scored() -> None:
    b = chess.Board()
    assert score_for(b, Line(1, (), 5000, None), chess.WHITE) == 1000
    assert score_for(b, Line(1, (), None, -3), chess.WHITE) == -1000
    mated = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert (
        score_for(mated, None, chess.BLACK) == 1000 and score_for(mated, None, chess.WHITE) == -1000
    )


def _line(board: chess.Board, sans: str, cp: int | None, mate: int | None = None) -> Line:
    b = board.copy()
    moves = []
    for san in sans.split():
        moves.append(b.parse_san(san))
        b.push(moves[-1])
    return Line(1, tuple(moves), cp, mate)


def test_missed_capture() -> None:
    before = chess.Board("rnb1kbnr/pppp1ppp/8/4p1q1/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    move = before.parse_san("d3")
    after = before.copy()
    after.push(move)
    m = classify_mistake(before, move, _line(before, "Nxg5", 900), _line(after, "Qf6", 0))
    assert m.kind == "missed_capture"
    assert "White's knight on f3 could capture Black's queen on g5" in m.detail


def test_allowed_and_missed_mate() -> None:
    before = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 3 3")
    move = before.parse_san("Nf6")
    after = before.copy()
    after.push(move)
    m = classify_mistake(before, move, _line(before, "g6", 0), _line(after, "Qxf7#", None, 1))
    assert m.kind == "allowed_mate" and "checkmate in 1" in m.detail
    # White to move misses the mate.
    move2 = after.parse_san("Qe2")
    after2 = after.copy()
    after2.push(move2)
    m2 = classify_mistake(after, move2, _line(after, "Qxf7#", None, 1), _line(after2, "Nd4", 0))
    assert m2.kind == "missed_mate" and "Qxf7#" in m2.detail


@pytest.mark.skipif(find_stockfish(None).path is None, reason="Stockfish not installed")
def test_real_engine_review_of_scholars_mate() -> None:
    path = find_stockfish(None).path
    assert path is not None
    ev = StockfishEvaluator(path, depth=10, max_seconds=0.3)
    try:
        review = review_game(parse_pgn(SCHOLARS)[0], chess.BLACK, get_level("novice"), ev)
    finally:
        ev.close()
    flagged = review.flagged()
    assert [m.san for m in flagged] == ["Nf6"]
    assert flagged[0].mistake is not None and flagged[0].mistake.kind == "allowed_mate"
    assert review.evals[-1] == 1000  # White mated Black


# -- summary and explanation facts ---------------------------------------------------------------


def test_summary_counts_types_and_gives_tips() -> None:
    review = review_game(
        parse_pgn(HUNG_QUEEN)[0], chess.BLACK, get_level("beginner"), MaterialEvaluator()
    )
    s = summarize([review, review])
    assert s.games == 2 and s.user_moves == 4 and s.severities == {"blunder": 2}
    assert [(k.kind, k.count) for k in s.kinds] == [("hung_piece", 2)]
    text = summary_text(s, get_level("beginner"))
    assert "2 blunders" in text and "inaccuracies are not flagged" in text
    assert "1. Hanging pieces (2): Before you let go of a piece" in text
    clean = summary_text(summarize([]), get_level("expert"))
    assert "No mistakes" in clean and "not flagged" not in clean


def test_count_kinds_orders_by_frequency_then_cost() -> None:
    from chesstrainer.review.mistakes import Mistake

    pairs = [(Mistake("missed_capture", ""), 300), (Mistake("hung_piece", ""), 100)]
    pairs += [(Mistake("missed_capture", ""), 300), (Mistake("positional", ""), 900)]
    assert [k.kind for k in count_kinds(pairs)] == ["missed_capture", "positional", "hung_piece"]


def test_review_facts_ground_a_good_explanation() -> None:
    review = review_game(
        parse_pgn(HUNG_QUEEN)[0], chess.BLACK, get_level("beginner"), MaterialEvaluator()
    )
    m = review.flagged()[0]
    facts = review_facts(m, get_level("beginner"))
    assert facts.context == "review:d8g5" and "Qg5" in facts.instructions
    assert {"review_played", "review_type"} <= facts.tags
    text = (
        "After Qg5, White's knight on f3 can capture your queen on g5, so it was left hanging. "
        f"{facts.san} was better."
    )
    assert check_grounding(text, facts) == []
    assert check_grounding("Qg5 walks into a fork.", facts)  # not in the facts


# -- runner ---------------------------------------------------------------------------------------


class ScriptedClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict[str, Any]] = []
        self.messages = self

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.text)],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=900, output_tokens=60),
        )


def _run(explainer: Explainer | None) -> dict[str, Any]:
    out: dict[str, Any] = {"progress": [], "games": [], "explained": [], "done": threading.Event()}
    cb = ReviewCallbacks(
        progress=lambda i, d, t: out["progress"].append((i, d, t)),
        game_done=lambda i, r: out["games"].append(r),
        explained=lambda i, ply, res, err, facts: out["explained"].append((i, ply, res, err)),
        finished=lambda err: (out.__setitem__("error", err), out["done"].set()),
    )
    games = [(g, chess.BLACK) for g in parse_pgn(HUNG_QUEEN + "\n" + HUNG_QUEEN)]
    runner = ReviewRunner(games, get_level("beginner"), MaterialEvaluator, cb, explainer)
    runner.start()
    assert out["done"].wait(20)
    return out


def test_runner_explains_biggest_mistakes_when_on(tmp_path: Path) -> None:
    client = ScriptedClient("Qg5 left your queen on g5 hanging: White's knight on f3 can take it.")
    explainer = Explainer("m", cache=ExplanationCache(tmp_path / "c.sqlite"), client=client)
    out = _run(explainer)
    assert out["error"] is None and len(out["games"]) == 2
    assert out["progress"][-1] == (1, 6, 6)
    assert [(i, ply) for i, ply, _, _ in out["explained"]] == [(0, 3), (1, 3)]
    assert len(client.calls) == 1  # the second game's identical mistake came from the cache
    assert out["explained"][1][2].cached


def test_runner_makes_no_api_calls_when_explanations_are_off() -> None:
    out = _run(None)
    assert out["error"] is None and out["explained"] == [] and len(out["games"]) == 2
