from __future__ import annotations

import csv
from pathlib import Path

import chess
import numpy as np
from PIL import Image

from chesstrainer.vision.evaluate import EvalReport, board_labels, load_real_screenshots
from chesstrainer.vision.geometry import cell_to_square
from chesstrainer.vision.labels import NUM_CLASSES, piece_to_label


def test_board_labels_orientation() -> None:
    fen = chess.STARTING_BOARD_FEN
    white = board_labels(fen, True)
    black = board_labels(fen, False)
    board = chess.Board()
    for c in range(64):
        assert white[c] == piece_to_label(board.piece_at(cell_to_square(c, True)))
    assert white[0] == piece_to_label(chess.Piece.from_symbol("r"))
    assert black[0] == piece_to_label(chess.Piece.from_symbol("R"))


def test_report_metrics() -> None:
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    conf[0, 0] = 9
    conf[1, 0] = 1
    report = EvalReport("t", 0.9, 0.0, 1, conf)
    prec, rec, n = report.per_class()["empty"]
    assert n == 9 and rec == 1.0 and abs(prec - 0.9) < 1e-9
    assert "P->empty x1" in report.format()


def test_load_real_screenshots(tmp_path: Path) -> None:
    Image.new("RGB", (160, 160), (200, 200, 200)).save(tmp_path / "a.gif")
    with (tmp_path / "labels.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "fen", "white_bottom"])
        w.writeheader()
        w.writerow({"file": "a.gif", "fen": "8/8/8/8/8/8/8/K6k w - - 0 1", "white_bottom": "0"})
    x, y, names = load_real_screenshots(tmp_path)
    assert x.shape[:2] == (1, 64) and names == ["a.gif"]
    assert (y > 0).sum() == 2


def test_load_real_screenshots_missing(tmp_path: Path) -> None:
    x, y, names = load_real_screenshots(tmp_path)
    assert len(x) == 0 and names == []
