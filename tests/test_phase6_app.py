"""Phase 6 in the app, offscreen: the review window (fake engine, fake fetchers, fake Claude
client), the explanations toggle in reviews, and the hotkey actions."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import chess
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from chesstrainer.explain.explainer import Explainer, ExplanationCache  # noqa: E402
from chesstrainer.levels import get_level  # noqa: E402
from chesstrainer.review.games import FetchError, parse_pgn  # noqa: E402
from chesstrainer.settings.config import Settings, load_settings  # noqa: E402
from tests.test_phase5_app import ScriptedClient, _app  # noqa: E402
from tests.test_review import HUNG_QUEEN, SCHOLARS, MaterialEvaluator  # noqa: E402

_QAPP: Any = None


def _qapp() -> Any:
    from PyQt6.QtWidgets import QApplication

    global _QAPP
    _QAPP = QApplication.instance() or QApplication([])
    return _QAPP


def _wait(cond: Any, seconds: float = 10.0) -> bool:
    from PyQt6.QtWidgets import QApplication

    deadline = time.time() + seconds
    while not cond() and time.time() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)
    return bool(cond())


def _window(tmp_path: Path, explainer: Explainer | None, **kw: Any) -> Any:
    from chesstrainer.overlay.review_window import ReviewWindow

    _qapp()
    os.environ["CHESSTRAINER_HOME"] = str(tmp_path)
    note = "" if explainer is not None else "Explanations are off."
    return ReviewWindow(
        Settings(), get_level("beginner"), None, lambda: (explainer, note),
        make_evaluator=MaterialEvaluator, **kw,
    )  # fmt: skip


def test_review_window_pasted_pgn_marks_and_explains_mistakes(tmp_path: Path) -> None:
    client = ScriptedClient("Qg5 left your queen on g5 hanging: White's knight on f3 can take it.")
    explainer = Explainer("m", cache=ExplanationCache(tmp_path / "c.sqlite"), client=client)
    w = _window(tmp_path, explainer)
    w.pgn_edit.setPlainText(HUNG_QUEEN)
    w.side_combo.setCurrentIndex(2)  # I played Black
    w.load_pasted()
    assert w.game_list.count() == 1 and w.loaded[0].user_color == chess.BLACK
    w.review_selected()
    assert _wait(lambda: w.review_button.isEnabled() and len(w.explanations) == 1)
    rows = [w.move_list.item(i).text() for i in range(w.move_list.count())]
    assert rows[0] == "Start position" and "2... Qg5??   blunder · explained" in rows
    assert w.ply == 4  # opened on the first flagged move
    assert "Hanging pieces" in w.detail.text() and "Blunder" in w.detail.text()
    assert "left your queen on g5 hanging" in w.detail.text()
    assert "Checked against the facts (passed)" in w.detail.toolTip()
    assert len(w.board_view.arrows) == 2  # best move + the move played
    assert "1. Hanging pieces (1)" in w.summary.text()
    assert len(client.calls) == 1
    w.first()
    assert w.ply == 0 and w.board_view.arrows == []
    w.next_mistake()
    assert w.ply == 4


def test_review_window_explanations_off_makes_no_calls(tmp_path: Path) -> None:
    w = _window(tmp_path, None)
    w.pgn_edit.setPlainText(HUNG_QUEEN.replace('"me"', '"someone"'))
    w.load_pasted()  # auto: no username matches -> White assumed
    assert "assumed" in w.game_list.item(0).text()
    w.side_combo.setCurrentIndex(2)
    w.load_pasted()
    w.review_selected()
    assert _wait(lambda: w.review_button.isEnabled() and w.reviews)
    assert w.explanations == {}
    assert "Explanations are off." in w.status.text()
    assert "Hanging pieces (1)" in w.summary.text()


def test_review_window_fetch_saves_username_and_reports_errors(tmp_path: Path) -> None:
    calls: list[tuple[str, int]] = []

    def lichess(name: str, n: int) -> list[Any]:
        calls.append((name, n))
        games = parse_pgn(HUNG_QUEEN + "\n" + SCHOLARS, "lichess")
        for g in games:
            g.user_color = g.color_of([name])
        return games

    def chesscom(name: str, n: int) -> list[Any]:
        raise FetchError("Chess.com has no user called 'ghost'.")

    w = _window(tmp_path, None, fetchers={"lichess": lichess, "chess.com": chesscom})
    w.fetch("lichess", " me ", 7)
    assert _wait(lambda: w.game_list.count() == 2)
    assert calls == [("me", 7)]
    saved = load_settings()
    assert saved.lichess_username == "me" and saved.review_fetch_count == 7
    w.game_list.selectAll()
    w.review_selected()
    assert _wait(lambda: w.review_button.isEnabled() and len(w.reviews) == 2)
    assert w.reviewed_combo.count() == 2
    assert "2 games" in w.summary.text()
    w.fetch("chess.com", "ghost", 5)
    assert _wait(lambda: "no user called" in w.status.text())


def test_app_review_mode_and_hotkeys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = ScriptedClient("x")
    trainer = _app(monkeypatch, tmp_path, client, explanations=False)
    explainer, note = trainer.review_explainer()
    assert explainer is None and "no API calls" in note and client.calls == []

    trainer.on_hotkey("toggle_overlay")
    assert trainer.overlay_hidden and trainer.panel.hidden_note.text().startswith("Arrows hidden")
    trainer._set_overlay_visible(True)
    assert not trainer.overlay.isVisible()  # stays hidden while toggled off
    trainer.on_hotkey("toggle_overlay")
    assert not trainer.overlay_hidden and trainer.panel.hidden_note.isHidden()

    trainer.on_hotkey("pause")
    assert trainer.paused and trainer.panel.pause.isChecked()
    trainer.on_hotkey("pause")
    assert not trainer.paused

    trainer.on_hotkey("toggle_explanations")
    assert load_settings().explanations is True
    trainer.on_hotkey("toggle_explanations")
    assert load_settings().explanations is False

    flips: list[Any] = []
    monkeypatch.setattr(trainer.vision, "call", lambda fn: flips.append(fn))
    trainer.on_hotkey("flip")
    assert len(flips) == 1

    # analysis -> trainer -> review (window opens, live reading paused) -> analysis
    monkeypatch.setattr(trainer, "choose_opening", lambda: None)
    trainer.on_hotkey("switch_mode")
    assert trainer.settings.app_mode == "trainer"
    trainer.on_hotkey("switch_mode")
    assert trainer.review_window is not None and trainer.review_window.isVisible()
    assert trainer.paused and trainer.panel.mode.currentData() == "review"
    trainer.on_hotkey("switch_mode")
    assert not trainer.review_window.isVisible() and not trainer.paused
    assert trainer.settings.app_mode == "analysis"
    assert trainer.panel.mode.currentData() == "analysis"
