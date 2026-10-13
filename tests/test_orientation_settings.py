from __future__ import annotations

import json
import sys
from pathlib import Path

import chess
import numpy as np
import pytest

from chesstrainer.overlay.capture_exclusion import set_excluded_from_capture
from chesstrainer.settings.config import Settings, load_settings, save_settings
from chesstrainer.vision.geometry import cell_to_square
from chesstrainer.vision.labels import piece_to_label
from chesstrainer.vision.orientation import OrientationTracker, orientation_score


def cell_labels(board: chess.Board, white_bottom: bool) -> np.ndarray:
    return np.array(
        [piece_to_label(board.piece_at(cell_to_square(c, white_bottom))) for c in range(64)]
    )


@pytest.mark.parametrize("white_bottom", [True, False])
def test_orientation_score_sign(white_bottom: bool) -> None:
    board = chess.Board("r1bqk2r/1pppbppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 4 6")
    s = orientation_score(cell_labels(board, white_bottom))
    assert (s > 0) == white_bottom and abs(s) > 3


def test_tracker_hysteresis_and_flip() -> None:
    tr = OrientationTracker("auto", switch_after=3)
    start = chess.Board()
    assert tr.update(cell_labels(start, False)) is False
    # An ambiguous endgame that weakly suggests the other way does not flip it.
    endgame = cell_labels(chess.Board("8/8/8/3k4/8/8/3K4/8 w - - 0 1"), True)
    assert tr.update(endgame) is False
    tr.flip()
    assert tr.white_bottom is True and tr.mode == "white"
    assert tr.update(cell_labels(start, False)) is True  # manual choice sticks


def test_coordinate_evidence_breaks_ties() -> None:
    tr = OrientationTracker("auto")
    endgame = cell_labels(chess.Board("8/8/8/3k4/8/8/3K4/8 w - - 0 1"), True)
    assert tr.update(endgame, coords=-3.0) is False


def test_settings_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    s = Settings(
        monitor=2, hide_overlay_from_capture=False, orientation="black", locked_region=[1, 2, 3, 3]
    )
    save_settings(s, path)
    loaded = load_settings(path)
    assert loaded == s


def test_settings_defaults_and_tolerance(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    assert load_settings(path).hide_overlay_from_capture is True  # on by default
    path.write_text(json.dumps({"orientation": "sideways", "unknown_key": 1, "capture_fps": 3}))
    s = load_settings(path)
    assert s.orientation == "auto" and s.capture_fps == 3
    path.write_text("{not json")
    assert load_settings(path) == Settings()


def test_settings_home_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CHESSTRAINER_HOME", str(tmp_path))
    save_settings(Settings(capture_fps=7))
    assert load_settings().capture_fps == 7
    assert (tmp_path / "settings.json").exists()


def test_capture_exclusion_noop_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    assert set_excluded_from_capture(1234, True) is False
