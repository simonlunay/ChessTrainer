"""Phase 4 follow-ups (grounding check in the live path, explanations toggle) and the opening
trainer inside the app, offscreen with fake engine results and a fake Claude client."""

from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import chess
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from chesstrainer.engine.analysis import Analysis, Line  # noqa: E402
from chesstrainer.explain.explainer import Explainer, ExplanationCache  # noqa: E402
from chesstrainer.explain.facts import compute_facts  # noqa: E402
from chesstrainer.levels import get_level  # noqa: E402
from chesstrainer.openings.book import default_book  # noqa: E402
from chesstrainer.openings.practice import PracticeStore  # noqa: E402
from chesstrainer.settings.config import Settings, load_settings  # noqa: E402
from chesstrainer.vision.position import PositionReading  # noqa: E402

_QAPP: Any = None
START_E4E5 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"


class ScriptedClient:
    """Returns the given texts in order (the last one repeats)."""

    def __init__(self, *texts: str) -> None:
        self.texts = list(texts)
        self.calls: list[dict[str, Any]] = []
        self.messages = self

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        text = self.texts[min(len(self.calls), len(self.texts)) - 1]
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=900, output_tokens=60),
        )


def nf3_facts():  # type: ignore[no-untyped-def]
    board = chess.Board(START_E4E5)
    moves = []
    b = board.copy()
    for san in ("Nf3", "Nc6"):
        moves.append(b.parse_san(san))
        b.push(moves[-1])
    return compute_facts(board, Line(1, tuple(moves), 40, None), get_level("beginner"))


# -- grounding check in the explainer ----------------------------------------------------------


def test_grounded_explanation_passes_first_time(tmp_path: Path) -> None:
    cache = ExplanationCache(tmp_path / "c.sqlite")
    client = ScriptedClient("Nf3 develops your knight and attacks the pawn on e5.")
    ex = Explainer("m", cache=cache, client=client)
    result = ex.explain(nf3_facts(), get_level("beginner"))
    assert result.outcome == "passed" and result.calls == 1 and len(client.calls) == 1
    assert cache.grounding_stats() == {"passed": 1, "regenerated": 0, "fallback": 0}


def test_flagged_explanation_is_regenerated_once(tmp_path: Path) -> None:
    cache = ExplanationCache(tmp_path / "c.sqlite")
    client = ScriptedClient(
        "Nf3 forks the rook on a8 and the queen.",  # fork + a8: not in the facts
        "Nf3 develops your knight and attacks the pawn on e5.",
    )
    ex = Explainer("m", cache=cache, client=client)
    result = ex.explain(nf3_facts(), get_level("beginner"))
    assert result.outcome == "regenerated" and result.calls == 2
    assert result.text.startswith("Nf3 develops")
    retry = client.calls[1]["messages"][0]["content"]
    assert "'a8'" in retry and "forks the rook on a8" in retry  # the flagged claims are named
    assert result.input_tokens == 1800 and result.output_tokens == 120
    assert cache.grounding_stats()["regenerated"] == 1
    hit = ex.explain(nf3_facts(), get_level("beginner"))  # cached with its outcome
    assert hit.cached and hit.outcome == "regenerated" and len(client.calls) == 2


def test_twice_flagged_explanation_falls_back_to_the_facts(tmp_path: Path) -> None:
    cache = ExplanationCache(tmp_path / "c.sqlite")
    client = ScriptedClient("Nf3 pins the bishop on b4.", "Nf3 sets up a skewer on h5.")
    ex = Explainer("m", cache=cache, client=client)
    facts = nf3_facts()
    result = ex.explain(facts, get_level("beginner"))
    assert result.outcome == "fallback" and len(client.calls) == 2
    assert result.text.startswith("White plays Nf3")
    assert all(sentence.strip() in facts.text() for sentence in result.text.split(". ")[:1])
    assert len(result.issues) >= 2
    assert cache.grounding_stats() == {"passed": 0, "regenerated": 0, "fallback": 1}


# -- app: explanations toggle ------------------------------------------------------------------


def _app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, client: Any, **settings: Any) -> Any:
    from PyQt6.QtWidgets import QApplication

    from chesstrainer.overlay import app as app_mod

    monkeypatch.setenv("CHESSTRAINER_HOME", str(tmp_path))
    real = app_mod.Explainer
    monkeypatch.setattr(
        app_mod,
        "Explainer",
        lambda model, effort, cache: real(model, effort, cache=cache, client=client),
    )
    global _QAPP
    _QAPP = QApplication.instance() or QApplication([])  # keep it alive for the test
    defaults: dict[str, Any] = {"level_chosen": True, "explain_delay_s": 0.0}
    defaults.update(settings)
    return app_mod.TrainerApp(
        Settings(**defaults),
        book=default_book(),
        practice_store=PracticeStore(tmp_path / "o.sqlite"),
    )


def _wait(cond: Any, seconds: float = 5.0) -> bool:
    from PyQt6.QtWidgets import QApplication

    deadline = time.time() + seconds
    while not cond() and time.time() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)
    return bool(cond())


def test_explanations_off_makes_no_api_calls_and_is_remembered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = ScriptedClient("Nf3 develops your knight.")
    trainer = _app(monkeypatch, tmp_path, client, explanations=False)
    assert trainer.explainer is None  # not even created
    board = chess.Board(START_E4E5)
    trainer._analyse(board)
    trainer.req_id = 1
    lines = (Line(1, (board.parse_san("Nf3"),), 40, None),)
    trainer.on_engine_update(Analysis(1, board.fen(), 22, lines, 0.5, final=True))
    for _ in range(10):
        trainer._maybe_explain()
    assert client.calls == []
    assert trainer.panel.lines[0].text()  # engine lines still shown
    assert "Explanations are off" in trainer.panel.explanation.text()
    # Ctrl+E / checkbox: on -> explains, saved; off again -> saved.
    trainer.panel.explain_box.setChecked(True)
    assert load_settings().explanations is True
    assert _wait(lambda: len(client.calls) == 1)
    trainer.panel.explain_box.setChecked(False)
    assert load_settings().explanations is False
    trainer._explained = None
    trainer._maybe_explain()
    assert len(client.calls) == 1
    if trainer.explain_worker is not None:
        trainer.explain_worker.close()


# -- app: opening trainer ----------------------------------------------------------------------


def _reading(board: chess.Board) -> PositionReading:
    return PositionReading(board.copy(stack=False), [], 1.0, "history")


def test_trainer_mode_draws_the_book_move_and_grades_a_deviation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = ScriptedClient("Nf3 develops your knight and attacks the pawn on e5.")
    trainer = _app(monkeypatch, tmp_path, client, explanations=True)
    trainer.set_opening("Italian Game", "white", "guided")
    assert load_settings().app_mode == "trainer"
    assert trainer.panel.trainer_box.isVisibleTo(trainer.panel)

    board = chess.Board()
    for san in ("e4", "e5"):
        board.push_san(san)
        trainer._new_position(_reading(board))
    # White to move in book: the engine is asked about Nf3 only, and Nf3 is the arrow.
    nf3 = board.parse_san("Nf3")
    assert trainer._root == (nf3,)
    assert trainer._book_arrow() == nf3
    assert "Italian" in trainer.panel.trainer_title.text()
    assert "Nf3" in trainer.panel.trainer_prompt.text()
    # Explanation of the book move uses the opening facts and passes the check.
    trainer.req_id = 5
    line = Line(1, (nf3,), 40, None)
    trainer._position_t -= 5
    trainer.on_engine_update(Analysis(5, board.fen(), 20, (line,), 0.4, final=True))
    assert _wait(lambda: trainer.panel.explanation.text().startswith("Nf3 develops"))
    sent = client.calls[0]["messages"][0]["content"]
    assert "book move of the Italian Game" in sent and "Known idea of the Italian Game" in sent

    # Opponent's turn: normal search, no arrows; then Black leaves the book with ...a5.
    board.push(nf3)
    trainer._new_position(_reading(board))
    assert trainer._root == () and trainer._book_arrow() is None
    trainer.req_id = 6
    best_black = Line(1, (board.parse_san("Nc6"),), 40, None)
    trainer.on_engine_update(Analysis(6, board.fen(), 20, (best_black,), 0.4, final=True))
    board.push_san("a5")
    trainer._new_position(_reading(board))
    assert "left the book" in trainer.panel.trainer_message.text()
    assert trainer._book_arrow() is False  # normal analysis from here
    trainer.req_id = 7
    after = Line(1, (board.parse_san("Nxe5"),), 300, None)
    trainer.on_engine_update(Analysis(7, board.fen(), 18, (after,), 0.6))
    msg = trainer.panel.trainer_message.text()
    assert "Stockfish: a5 is a mistake (about 2.6 pawns worse than the best move)" in msg
    if trainer.explain_worker is not None:
        trainer.explain_worker.close()


def test_trainer_works_without_explanations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = ScriptedClient("unused")
    trainer = _app(monkeypatch, tmp_path, client, explanations=False)
    trainer.set_opening("Scandinavian Defense", "black", "guided")
    board = chess.Board()
    trainer._new_position(_reading(board))
    board.push_san("e4")
    trainer._new_position(_reading(board))
    assert trainer._book_arrow() == board.parse_san("d5")
    assert "Scandinavian" in trainer.panel.trainer_title.text()
    board.push_san("d5")
    board.push_san("exd5")
    trainer._new_position(_reading(board))
    assert "Scandinavian Defense" in trainer.panel.trainer_variation.text()
    assert trainer._book_arrow() == board.parse_san("Qxd5")
    trainer.req_id = 3
    line = Line(1, (board.parse_san("Qxd5"),), -20, None)
    trainer.on_engine_update(Analysis(3, board.fen(), 22, (line,), 0.5, final=True))
    trainer._maybe_explain()
    assert client.calls == []
    # Back to plain analysis mode.
    trainer.set_mode("analysis")
    assert trainer.session is None and trainer._root == ()


def test_opening_picker_lists_recommendations_first_and_searches() -> None:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication

    from chesstrainer.overlay.opening_dialog import OpeningChoice, OpeningDialog

    global _QAPP
    _QAPP = QApplication.instance() or QApplication([])
    dialog = OpeningDialog(default_book(), get_level("beginner"))
    names = [dialog.list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(dialog.list.count())]
    assert names[1:4] == ["Italian Game", "London System", "Scotch Game"]  # after the header
    assert dialog.choice() == OpeningChoice("Italian Game", "white", "guided")
    dialog.black.setChecked(True)
    dialog.practice.setChecked(True)
    names = [dialog.list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(dialog.list.count())]
    assert names[1:4] == ["Italian Game", "Scandinavian Defense", "Queen's Gambit Declined"]
    dialog.search.setText("najdorf")
    choice = dialog.choice()
    assert choice is not None and "Najdorf" in choice.name and choice.side == "black"
    assert choice.mode == "practice"
