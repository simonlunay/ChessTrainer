from __future__ import annotations

import os
import threading
from pathlib import Path

import chess
import chess.engine
import pytest

from chesstrainer.engine import locate
from chesstrainer.engine.analysis import Analysis, AnalysisEngine, Line, LineCollector
from chesstrainer.engine.locate import DOWNLOAD_URL, find_stockfish


def info(multipv: int, depth: int, uci: str, cp: int, turn: chess.Color = chess.WHITE) -> dict:
    return {
        "multipv": multipv,
        "depth": depth,
        "pv": [chess.Move.from_uci(uci)],
        "score": chess.engine.PovScore(chess.engine.Cp(cp), turn),
    }


def test_collector_emits_complete_sets_per_depth() -> None:
    c = LineCollector(chess.Board(), multipv=3)
    assert c.add(info(1, 5, "e2e4", 30)) is None
    assert c.add(info(2, 5, "d2d4", 25)) is None
    snap = c.add(info(3, 5, "g1f3", 20))
    assert snap is not None and snap[0] == 5
    assert [line.move.uci() for line in snap[1]] == ["e2e4", "d2d4", "g1f3"]
    # Next depth: incomplete until its last line arrives.
    assert c.add(info(1, 6, "d2d4", 33)) is None
    assert c.add(info(2, 6, "e2e4", 31)) is None
    assert c.add(info(3, 6, "c2c4", 10)) is not None


def test_collector_skips_bounds_and_caps_multipv_at_legal_moves() -> None:
    board = chess.Board("7k/8/8/8/8/8/8/K6R b - - 0 1")  # black king: 2 legal moves
    c = LineCollector(board, multipv=3)
    assert c.expected == 2
    bound = info(1, 4, "h8g7", 0, chess.BLACK) | {"lowerbound": True}
    assert c.add(bound) is None
    assert c.add(info(1, 4, "h8g7", -500, chess.BLACK)) is None
    snap = c.add(info(2, 4, "h8g8", -600, chess.BLACK))
    assert snap is not None
    # Scores are converted to White's point of view.
    assert snap[1][0].cp == 500


def test_line_score_text_and_eval_bar() -> None:
    move = (chess.Move.from_uci("e2e4"),)
    assert Line(1, move, 123, None).score_text() == "+1.23"
    assert Line(1, move, -50, None).score_text() == "-0.50"
    assert Line(1, move, None, 3).score_text() == "#3"
    assert Line(1, move, None, -2).score_text() == "#-2"
    assert Line(1, move, 0, None).white_win_fraction() == pytest.approx(0.5)
    assert Line(1, move, 400, None).white_win_fraction() > 0.75
    assert Line(1, move, None, -1).white_win_fraction() == 0.0


def test_locate_reports_missing_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(locate, "engine_dirs", lambda: [tmp_path / "engine"])
    monkeypatch.setattr(locate.shutil, "which", lambda name: None)
    monkeypatch.delenv("STOCKFISH_PATH", raising=False)
    search = find_stockfish()
    assert not search.found
    msg = search.setup_message()
    assert DOWNLOAD_URL in msg and str(tmp_path / "engine") in msg and "STOCKFISH_PATH" in msg


def test_locate_flags_bad_env_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STOCKFISH_PATH", str(tmp_path / "nope.exe"))
    search = find_stockfish()
    assert not search.found
    assert search.problem is not None and "nope.exe" in search.problem


def test_locate_finds_engine_in_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exe = tmp_path / "engine" / "stockfish" / "stockfish-windows-x86-64.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    if os.name != "nt":
        exe = exe.rename(exe.with_suffix(""))
        exe.chmod(0o755)
    monkeypatch.setattr(locate, "engine_dirs", lambda: [tmp_path / "engine"])
    monkeypatch.setattr(locate.shutil, "which", lambda name: None)
    monkeypatch.delenv("STOCKFISH_PATH", raising=False)
    assert find_stockfish().path == exe


STOCKFISH = find_stockfish().path
needs_stockfish = pytest.mark.skipif(STOCKFISH is None, reason="Stockfish not installed")


@needs_stockfish
def test_engine_multipv_and_stale_request_cancelled() -> None:
    assert STOCKFISH is not None
    updates: list[Analysis] = []
    done = threading.Event()

    def on_update(a: Analysis) -> None:
        updates.append(a)
        if a.final:
            done.set()

    engine = AnalysisEngine(STOCKFISH, on_update, threads=1, max_seconds=0.5)
    engine.start()
    try:
        assert engine.wait_ready()
        first = engine.analyze(chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w"))
        second = engine.analyze(chess.Board())  # supersedes the first immediately
        assert done.wait(10)
    finally:
        engine.close()
    finals = [a for a in updates if a.final]
    assert [a.request_id for a in finals] == [second]
    assert all(a.request_id != first or not a.final for a in updates)
    last = finals[0]
    assert len(last.lines) == 3 and last.depth >= 8
    assert len({line.move for line in last.lines}) == 3
    assert all(line.move in chess.Board().legal_moves for line in last.lines)
