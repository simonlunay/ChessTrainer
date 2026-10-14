from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import chess
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from chesstrainer.engine.analysis import Analysis, Line  # noqa: E402
from chesstrainer.explain import board_tools as bt  # noqa: E402
from chesstrainer.explain.explainer import (  # noqa: E402
    Explainer,
    ExplainError,
    ExplainWorker,
    ExplanationCache,
)
from chesstrainer.explain.facts import compute_facts  # noqa: E402
from chesstrainer.explain.grounding import check_grounding  # noqa: E402
from chesstrainer.explain.practical import simplicity, suggest  # noqa: E402
from chesstrainer.explain.prompt import user_message  # noqa: E402
from chesstrainer.levels import LEVEL_KEYS, all_levels, get_level, level_for_elo  # noqa: E402
from chesstrainer.settings.config import Settings, load_settings  # noqa: E402


def line(board: chess.Board, sans: str, cp: int | None = 0, mate: int | None = None) -> Line:
    b = board.copy()
    moves = []
    for san in sans.split():
        moves.append(b.parse_san(san))
        b.push(moves[-1])
    return Line(1, tuple(moves), cp, mate)


def facts_for(fen: str, sans: str, level: str = "beginner", cp: int = 0, mate: int | None = None):
    board = chess.Board(fen)
    return compute_facts(board, line(board, sans, cp, mate), get_level(level))


# -- levels and settings -----------------------------------------------------------------------


def test_levels_load_in_order_with_expected_behaviour() -> None:
    levels = all_levels()
    assert tuple(lv.key for lv in levels) == LEVEL_KEYS
    assert [lv.candidates for lv in levels] == [1, 2, 3, 3, 3]
    assert get_level("beginner").practical_margin_cp > 0
    assert get_level("advanced").practical_margin_cp == 0
    assert get_level("beginner").label == "Beginner (under 800 Elo)"
    assert get_level("expert").label == "Expert (2000+ Elo)"
    assert get_level("nonsense").key == "beginner"
    assert level_for_elo(400).key == "beginner"
    assert level_for_elo(1200).key == "intermediate"
    assert level_for_elo(2400).key == "expert"
    assert "Italian Game" in get_level("beginner").openings["white"]
    assert "London System" in get_level("beginner").openings["white"]


def test_review_thresholds_scale_with_level() -> None:
    beginner, expert = get_level("beginner"), get_level("expert")
    assert beginner.classify_loss(100) is None  # beginners are not told about inaccuracies
    assert beginner.classify_loss(450) == "blunder"
    assert expert.classify_loss(50) == "inaccuracy"
    assert expert.classify_loss(250) == "blunder"


def test_settings_default_to_beginner_and_cheap_model(tmp_path: Path) -> None:
    s = Settings()
    assert s.skill_level == "beginner" and not s.level_chosen
    assert s.explain_model == "claude-haiku-5-5"
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"skill_level": "grandmaster"}))
    assert load_settings(path).skill_level == "beginner"


# -- python-chess facts ------------------------------------------------------------------------


def test_see_and_hanging_pieces() -> None:
    b = chess.Board("4k3/8/8/3p4/4N3/8/8/4K3 w - - 0 1")  # knight takes an undefended pawn?
    assert bt.see(b, chess.Move.from_uci("e4d6")) == 0  # not a capture
    b = chess.Board("4k3/8/2p5/3p4/4P3/8/8/4K3 w - - 0 1")
    assert bt.see(b, chess.Move.from_uci("e4d5")) == 0  # pawn for pawn
    b = chess.Board("4k3/8/2p5/3n4/4P3/8/8/4K3 w - - 0 1")
    assert bt.see(b, chess.Move.from_uci("e4d5")) == 2  # pawn takes knight, pawn recaptures
    assert bt.en_prise(b, chess.D5)  # the knight can be won


def test_fork_check_capture_and_material() -> None:
    mf = facts_for(
        "r3k2r/ppp2ppp/8/1N6/8/8/PPP2PPP/4K2R w Kkq - 0 18", "Nxc7+ Ke7 Nxa8 Rxa8", cp=250
    )
    assert {"capture", "hanging", "check", "fork", "material_gain", "line"} <= mf.tags
    fork = next(f.text for f in mf.facts if f.tag == "fork")
    assert "rook on a8" in fork and "king on e8" in fork
    gain = next(f.text for f in mf.facts if f.tag == "material_gain")
    assert "won 3 points" in gain and "White is ahead by 1 point" in gain


def test_skewer_and_no_fake_development() -> None:
    mf = facts_for("q7/8/2k5/8/8/8/8/1B2K3 w - - 0 50", "Be4+ Kd6 Bxa8")
    assert {"skewer", "check", "material_gain"} <= mf.tags
    assert "development" not in mf.tags  # the bishop did not start on b1


def test_pin_development_and_castling_facts() -> None:
    mf = facts_for("r1bqkbnr/ppp2ppp/2np4/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 4", "Bb5")
    assert {"pin", "development"} <= mf.tags
    pin = next(f.text for f in mf.facts if f.tag == "pin")
    assert "knight on c6" in pin and "king on e8" in pin and "cannot legally move" in pin
    mf = facts_for("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 1 5", "O-O")
    assert "castle" in mf.tags


def test_mate_promotion_en_passant_and_discovered_check() -> None:
    mf = facts_for(
        "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "Qxf7#", mate=1
    )
    assert "mate" in mf.tags and "check" not in mf.tags
    assert "promotion" in facts_for("8/4P3/8/8/8/2k5/8/4K3 w - - 0 60", "e8=Q").tags
    assert "en_passant" in facts_for("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 40", "exd6").tags
    mf = facts_for("4k3/8/8/8/4N3/1q6/6PP/4R1K1 w - - 0 30", "Nc5+")
    assert {"discovered", "check", "attack"} <= mf.tags
    assert any("discovered check" in f.text for f in mf.facts)


def test_threat_stopped_by_retreat() -> None:
    mf = facts_for("r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4", "Ba4")
    defends = next(f.text for f in mf.facts if f.tag == "defends")
    assert "axb5" in defends and "bishop on b5" in defends


def test_numbers_only_for_levels_that_use_them() -> None:
    fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    beginner = facts_for(fen, "Nf3 Nc6", "beginner", cp=35)
    expert = facts_for(fen, "Nf3 Nc6", "expert", cp=35)
    assert "+0.3" not in beginner.text() and "+0.3" in expert.text()
    assert "about equal" in beginner.text()


# -- practical move ----------------------------------------------------------------------------


def test_beginner_gets_simpler_move_within_margin() -> None:
    # Black's knight on d5 hangs to exd5; pretend the engine prefers a quiet move by 30 cp.
    board = chess.Board("r1bqkb1r/pppp1ppp/8/3np3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 0 5")
    quiet = Line(1, (chess.Move.from_uci("f1c4"),), 150, None)
    take = Line(2, (chess.Move.from_uci("e4d5"),), 120, None)
    other = Line(3, (chess.Move.from_uci("d2d4"),), 60, None)
    assert simplicity(board, take) > simplicity(board, quiet)
    s = suggest(board, (quiet, take, other), get_level("beginner"))
    assert s.practical and s.move == take.move and len(s.lines) == 1
    assert s.lines[0].rank == 1 and s.engine_best == quiet
    s = suggest(board, (quiet, take, other), get_level("novice"))
    assert s.practical and [ln.move for ln in s.lines] == [take.move, quiet.move]
    s = suggest(board, (quiet, take, other), get_level("intermediate"))
    assert not s.practical and len(s.lines) == 3 and s.move == quiet.move
    far = Line(2, (chess.Move.from_uci("e4d5"),), 50, None)  # 100 cp worse: too much
    assert not suggest(board, (quiet, far), get_level("beginner")).practical


def test_practical_facts_mention_engine_top_move() -> None:
    board = chess.Board("r1bqkb1r/pppp1ppp/8/3np3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 0 5")
    quiet = Line(1, (chess.Move.from_uci("f1c4"),), 150, None)
    take = Line(2, (chess.Move.from_uci("e4d5"),), 120, None)
    mf = compute_facts(board, take, get_level("beginner"), engine_best=quiet)
    best = next(f.text for f in mf.facts if f.tag == "engine_best")
    assert "Bc4" in best and "simpler" in best


# -- grounding check ---------------------------------------------------------------------------


def test_grounding_accepts_faithful_text_and_flags_inventions() -> None:
    mf = facts_for(
        "r3k2r/ppp2ppp/8/1N6/8/8/PPP2PPP/4K2R w Kkq - 0 18", "Nxc7+ Ke7 Nxa8 Rxa8", cp=250
    )
    good = (
        "Your knight on b5 takes the undefended pawn on c7 with check, and it also attacks the "
        "rook on a8 at the same time - a fork, which means one piece attacks two at once."
    )
    assert check_grounding(good, mf) == []
    bad = (
        "Nxc7+ is a fork that also pins the bishop on c8, and after Nd5 White is completely "
        "winning by +3.5 with mate in 2."
    )
    kinds = {i.kind for i in check_grounding(bad, mf)}
    assert {"square", "move", "concept", "number"} <= kinds


def test_grounding_flags_wrong_piece_on_square() -> None:
    mf = facts_for(
        "r3k2r/ppp2ppp/8/1N6/8/8/PPP2PPP/4K2R w Kkq - 0 18", "Nxc7+ Ke7 Nxa8 Rxa8", cp=250
    )
    issues = check_grounding("The knight forks the queen on a8.", mf)
    assert [i.kind for i in issues] == ["piece"]


# -- explainer, cache and worker ---------------------------------------------------------------


class FakeClient:
    def __init__(self, text: str = "A fine move.", stop_reason: str = "end_turn") -> None:
        self.calls: list[dict[str, Any]] = []
        self.text, self.stop_reason = text, stop_reason
        self.messages = self

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text=self.text),
            ],
            stop_reason=self.stop_reason,
            usage=SimpleNamespace(input_tokens=500, output_tokens=40),
        )


def test_explainer_sends_only_facts_and_caches(tmp_path: Path) -> None:
    mf = facts_for("r3k2r/ppp2ppp/8/1N6/8/8/PPP2PPP/4K2R w Kkq - 0 18", "Nxc7+ Ke7 Nxa8 Rxa8")
    client = FakeClient()
    ex = Explainer("claude-haiku-5-5", cache=ExplanationCache(tmp_path / "c.sqlite"), client=client)
    level = get_level("beginner")
    first = ex.explain(mf, level)
    assert first.text == "A fine move." and not first.cached
    sent = client.calls[0]
    assert sent["model"] == "claude-haiku-5-5"
    content = sent["messages"][0]["content"]
    assert content == user_message(mf, level)
    assert mf.fen not in content  # no FEN/board: only the facts, the move and the level
    assert "Beginner" in content and "Nxc7+" in content
    second = ex.explain(mf, level)
    assert second.cached and len(client.calls) == 1
    ex.explain(mf, get_level("expert"))  # other level: new call
    assert len(client.calls) == 2


def test_explainer_without_key_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("chesstrainer.explain.explainer.anthropic_api_key", lambda: None)
    ex = Explainer("claude-haiku-5-5")
    assert not ex.available and "API key" in ex.unavailable_reason
    mf = facts_for("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "Nf3")
    with pytest.raises(ExplainError):
        ex.explain(mf, get_level("beginner"))


def test_refusal_is_reported() -> None:
    ex = Explainer("m", client=FakeClient(text="", stop_reason="refusal"))
    mf = facts_for("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "Nf3")
    with pytest.raises(ExplainError, match="declined"):
        ex.explain(mf, get_level("beginner"))


def test_worker_reports_results_by_request_id() -> None:
    done = threading.Event()
    got: list[tuple[int, str | None, str | None]] = []

    def on_result(rid: int, result: Any, error: str | None) -> None:
        got.append((rid, result.text if result else None, error))
        done.set()

    worker = ExplainWorker(Explainer("m", client=FakeClient("ok")), on_result)
    mf = facts_for("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "Nf3")
    rid = worker.submit(mf, get_level("beginner"))
    assert done.wait(5)
    worker.close()
    assert got == [(rid, "ok", None)]


# -- app wiring (offscreen Qt) -----------------------------------------------------------------


def test_app_explains_only_once_position_is_stable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from PyQt6.QtWidgets import QApplication

    from chesstrainer.overlay import app as app_mod

    monkeypatch.setenv("CHESSTRAINER_HOME", str(tmp_path))
    client = FakeClient("Develop your knight.")
    real = app_mod.Explainer
    monkeypatch.setattr(
        app_mod,
        "Explainer",
        lambda model, effort, cache: real(model, effort, cache=cache, client=client),
    )
    qapp = QApplication.instance() or QApplication([])
    trainer = app_mod.TrainerApp(Settings(level_chosen=True, explain_delay_s=1.0))
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    trainer._analyse(board)  # the engine is not started: nothing is searched
    lines = (line(board, "Nf3 Nc6", 40),)
    trainer.req_id = 7
    trainer.on_engine_update(Analysis(7, board.fen(), 20, lines, 0.3))
    assert client.calls == []  # too soon: the position has not been stable for 1 s
    trainer._position_t -= 2.0
    trainer.on_engine_update(Analysis(7, board.fen(), 21, lines, 0.4))
    deadline = time.time() + 5
    while trainer.panel.explanation.text() != "Develop your knight." and time.time() < deadline:
        qapp.processEvents()
    assert trainer.panel.explanation.text() == "Develop your knight."
    trainer.on_engine_update(Analysis(7, board.fen(), 22, lines, 0.5))
    assert len(client.calls) == 1  # same position + move + level: no second call
    trainer.set_level("expert")
    deadline = time.time() + 5
    while len(client.calls) < 2 and time.time() < deadline:
        qapp.processEvents()
    assert len(client.calls) == 2  # a new level re-explains
    assert load_settings().skill_level == "expert"
    if trainer.explain_worker is not None:
        trainer.explain_worker.close()


def test_level_dialog_preselects_beginner() -> None:
    from PyQt6.QtWidgets import QApplication

    from chesstrainer.overlay.level_dialog import LevelDialog

    _ = QApplication.instance() or QApplication([])
    dialog = LevelDialog("beginner")
    assert dialog.selected() == "beginner"
    dialog.group.button(3).setChecked(True)
    assert dialog.selected() == "advanced"


def test_pawn_structure_is_judged_at_the_end_of_the_line() -> None:
    fen = "r1bqkbnr/ppp2ppp/2np4/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 4"
    mf = facts_for(fen, "d4 exd4 Nxd4 g6", "intermediate")
    assert "doubled" not in mf.tags  # exd4 doubled Black's d-pawns, but Nxd4 took one back
    mf = facts_for(fen, "d4 exd4", "beginner")
    assert "doubled" in mf.tags


def test_out_of_credits_gets_a_readable_message() -> None:
    import anthropic
    import httpx2

    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(400, request=request)
    body = {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": "Your credit balance is too low."},
    }

    class Broke(FakeClient):
        def create(self, **kwargs: Any) -> Any:
            raise anthropic.BadRequestError("Error code: 400 - {...}", response=response, body=body)

    mf = facts_for("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "Nf3")
    with pytest.raises(ExplainError, match="out of credits"):
        Explainer("m", client=Broke()).explain(mf, get_level("beginner"))


def test_app_does_not_hammer_the_api_after_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from PyQt6.QtWidgets import QApplication

    from chesstrainer.overlay import app as app_mod

    class Failing(FakeClient):
        def create(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            raise ExplainError("Explanations paused: the Anthropic account is out of credits.")

    monkeypatch.setenv("CHESSTRAINER_HOME", str(tmp_path))
    client = Failing()
    real = app_mod.Explainer
    monkeypatch.setattr(
        app_mod,
        "Explainer",
        lambda model, effort, cache: real(model, effort, cache=cache, client=client),
    )
    qapp = QApplication.instance() or QApplication([])
    trainer = app_mod.TrainerApp(Settings(level_chosen=True, explain_delay_s=0.0))
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    trainer._analyse(board)
    trainer.req_id = 3
    lines = (line(board, "Nf3 Nc6", 40),)
    trainer.on_engine_update(Analysis(3, board.fen(), 20, lines, 0.3, final=True))
    deadline = time.time() + 5
    while "out of credits" not in trainer.panel.explanation.text() and time.time() < deadline:
        qapp.processEvents()
    for _ in range(20):  # snapshots keep arriving at 10 Hz
        trainer._maybe_explain()
        qapp.processEvents()
    assert len(client.calls) == 1
    assert "out of credits" in trainer.panel.explanation.text()
    if trainer.explain_worker is not None:
        trainer.explain_worker.close()
