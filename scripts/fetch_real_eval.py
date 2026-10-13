"""Build a real-site evaluation set from Lichess and Chess.com's own board renderers.

Both sites serve server-rendered position images (lichess1.org/export/fen.gif and
chess.com/dynboard). They use each site's real piece art and themes - including chess.com's
proprietary sets, which never appear in training - so they measure generalization to real
sites. Images go to ``data/real/`` (git-ignored) with a ``labels.csv`` read by the evaluator.
Add your own tightly cropped screenshots to the same CSV for true in-the-wild numbers.
"""

from __future__ import annotations

import argparse
import csv
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from chesstrainer.vision.evaluate import REAL_DIR
from chesstrainer.vision.synth import random_position

LICHESS_THEMES = ("brown", "blue", "green", "ic", "purple", "pink")
CHESSCOM_PIECES = (
    "neo",
    "classic",
    "wood",
    "glass",
    "alpha",
    "icy_sea",
    "8_bit",
    "bases",
    "book",
    "club",
    "gothic",
    "light",
    "marble",
    "metal",
    "tournament",
    "vintage",
)
CHESSCOM_BOARDS = (
    "green",
    "brown",
    "blue",
    "dash",
    "wood",
    "marble",
    "metal",
    "tournament",
    "bubblegum",
    "icy_sea",
    "purple",
    "sky",
)


def _get(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (chesstrainer eval)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        dest.write_bytes(resp.read())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lichess", type=int, default=20)
    parser.add_argument("--chesscom", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=REAL_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, str]] = []

    for i in range(args.lichess):
        board = random_position(rng)
        params = {"fen": board.board_fen(), "theme": LICHESS_THEMES[i % len(LICHESS_THEMES)]}
        if board.move_stack:
            params["lastMove"] = board.peek().uci()
        name = f"lichess_{i:03d}.gif"
        _get(
            "https://lichess1.org/export/fen.gif?" + urllib.parse.urlencode(params), args.out / name
        )
        rows.append({"file": name, "fen": board.board_fen(), "white_bottom": "1"})
        time.sleep(0.3)

    for i in range(args.chesscom):
        board = random_position(rng)
        flip = bool(rng.random() < 0.4)
        params = {
            "fen": board.fen(),
            "board": CHESSCOM_BOARDS[int(rng.integers(len(CHESSCOM_BOARDS)))],
            "piece": CHESSCOM_PIECES[i % len(CHESSCOM_PIECES)],
            "size": "2",
        }
        if flip:
            params["flip"] = "true"
        name = f"chesscom_{params['piece']}_{i:03d}.png"
        _get("https://www.chess.com/dynboard?" + urllib.parse.urlencode(params), args.out / name)
        rows.append({"file": name, "fen": board.board_fen(), "white_bottom": "0" if flip else "1"})
        time.sleep(0.3)

    with (args.out / "labels.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "fen", "white_bottom"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} labelled boards to {args.out}")


if __name__ == "__main__":
    main()
