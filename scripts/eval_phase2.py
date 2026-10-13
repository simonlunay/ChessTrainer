"""Phase 2 evaluation: board detection + piece recognition -> FEN.

Sections:
  A. Real boards (data/real): your screenshots, lichess.org and chess.com server renders.
  B. The same real boards pasted into cluttered full-screen pages (1920x1080 / 2560x1440).
  C. Synthetic full screens with decoy thumbnail boards (exact ground-truth boxes).
The live screen-capture test (a real game played through on screen) is scripts/live_selftest.py.

Writes data/eval/phase2_report.txt.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from chesstrainer.settings.config import Settings
from chesstrainer.vision.dataset import EVAL_DIR, train_split
from chesstrainer.vision.detect import BoardBox, detect_board
from chesstrainer.vision.evaluate import REAL_DIR, board_labels
from chesstrainer.vision.position import cells_to_squares, labels_of
from chesstrainer.vision.reader import BoardReader, read_image
from chesstrainer.vision.recognizer import PieceRecognizer
from chesstrainer.vision.screens import render_screen
from chesstrainer.vision.synth import AssetLibrary


@dataclass
class Tally:
    n: int = 0
    detected: int = 0
    exact_auto: int = 0  # placement exact, orientation detected automatically
    orient_ok: int = 0
    exact_given: int = 0  # placement exact, orientation given
    exact_no_repair: int = 0  # argmax only, orientation given (ablation)
    squares_ok: int = 0
    repairs: int = 0
    kings_fixed: int = 0
    corner_err: list[float] = field(default_factory=list)
    ms: list[float] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def row(self, name: str) -> str:
        pct = lambda k: f"{k}/{self.n} ({100 * k / max(1, self.n):.1f}%)"  # noqa: E731
        lines = [
            f"{name}  [{self.n} boards]",
            f"  board detected ............ {pct(self.detected)}",
            f"  FEN placement, auto orient  {pct(self.exact_auto)}",
            f"  orientation correct ....... {pct(self.orient_ok)}",
            f"  FEN placement, orient given {pct(self.exact_given)}",
            f"    without legality repair . {pct(self.exact_no_repair)}",
            f"  squares correct ........... {self.squares_ok}/{64 * self.n}"
            f" ({100 * self.squares_ok / max(1, 64 * self.n):.2f}%)",
            f"  repairs applied ........... {self.repairs} (two-king fixes: {self.kings_fixed})",
        ]
        if self.corner_err:
            ce = np.array(self.corner_err)
            lines.append(
                f"  box error (squares) ....... median {np.median(ce):.3f}, max {ce.max():.3f}"
            )
        if self.ms:
            lines.append(f"  detection time ............ median {np.median(self.ms):.0f} ms")
        if self.failures:
            shown = ", ".join(self.failures[:14]) + (" ..." if len(self.failures) > 14 else "")
            lines.append(f"  failures: {shown}")
        return "\n".join(lines)


def group_of(name: str) -> str:
    if name.startswith("lichess_"):
        return "lichess.org renders"
    if name.startswith("chesscom_"):
        return "chess.com renders"
    return "your screenshots"


def score(
    image: np.ndarray,
    truth_fen: str,
    white_bottom: bool,
    rec: PieceRecognizer,
    t: Tally,
    truth_box: BoardBox | None = None,
    name: str = "",
) -> None:
    t.n += 1
    t0 = time.perf_counter()
    det = detect_board(image)
    t.ms.append((time.perf_counter() - t0) * 1000)
    if det is None:
        t.failures.append(f"{name}:not found")
        return
    if truth_box is not None:
        err = truth_box.corner_error(det.box)
        t.corner_err.append(err)
        if err > 0.25:
            t.failures.append(f"{name}:wrong box")
            return
    t.detected += 1
    truth = board_labels(truth_fen, True)  # square order is orientation independent
    truth_sq = cells_to_squares(np.eye(13, dtype=np.float32)[truth], True).argmax(1)

    auto = read_image(image, BoardReader(Settings(), recognizer=rec))
    if auto.reading is not None:
        t.orient_ok += auto.white_bottom == white_bottom
        t.exact_auto += auto.reading.board.board_fen() == truth_fen.split()[0]
        if auto.white_bottom != white_bottom:
            t.failures.append(f"{name}:orientation")

    given = Settings(orientation="white" if white_bottom else "black")
    st = read_image(image, BoardReader(given, recognizer=rec))
    if st.reading is None or st.cell_probs is None:
        return
    t.exact_given += st.reading.board.board_fen() == truth_fen.split()[0]
    if st.reading.board.board_fen() != truth_fen.split()[0]:
        t.failures.append(f"{name}:pieces")
    got = labels_of(st.reading.board)
    t.squares_ok += int((got == truth_sq).sum())
    t.repairs += len([n for n in st.reading.notes if not n.startswith("side to move")])
    t.kings_fixed += len([n for n in st.reading.notes if n.startswith("two ")])
    sq_probs = cells_to_squares(st.cell_probs, white_bottom)
    t.exact_no_repair += bool((sq_probs.argmax(1) == truth_sq).all())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()
    rec = PieceRecognizer()
    rng = np.random.default_rng(args.seed)
    rows = list(csv.DictReader((REAL_DIR / "labels.csv").open(newline="")))
    images = {
        r["file"]: np.asarray(Image.open(REAL_DIR / r["file"]).convert("RGB"), np.uint8)
        for r in rows
    }

    report: list[str] = []

    # A. real boards as captured
    tallies: dict[str, Tally] = defaultdict(Tally)
    for r in rows:
        img = images[r["file"]]
        wb = r["white_bottom"] == "1"
        tight = r["file"].startswith(("lichess_", "chesscom_"))  # exact board crops
        box = BoardBox(0, 0, img.shape[1], img.shape[0]) if tight else None
        score(img, r["fen"], wb, rec, tallies[group_of(r["file"])], box, r["file"])
    report.append("== A. Real boards, as captured ==")
    report += [
        tallies[g].row(g) for g in ("your screenshots", "lichess.org renders", "chess.com renders")
    ]

    # B. real boards inside cluttered full-screen pages
    lib, split = AssetLibrary(), train_split()
    tallies = defaultdict(Tally)
    for r in rows:
        size = (1920, 1080) if rng.random() < 0.6 else (2560, 1440)
        tight = r["file"].startswith(("lichess_", "chesscom_"))
        screen = render_screen(rng, lib, split, size=size, board_image=images[r["file"]])
        score(
            screen.image,
            r["fen"],
            r["white_bottom"] == "1",
            rec,
            tallies[group_of(r["file"])],
            screen.box if tight else None,
            r["file"],
        )
    report.append("\n== B. Real boards pasted into cluttered full-screen pages ==")
    report += [
        tallies[g].row(g) for g in ("your screenshots", "lichess.org renders", "chess.com renders")
    ]

    # C. synthetic full screens
    t = Tally()
    for i in range(args.synthetic):
        s = render_screen(
            rng, lib, split, size=(1920, 1080) if rng.random() < 0.6 else (2560, 1440)
        )
        score(s.image, s.board.board_fen(), s.white_bottom, rec, t, s.box, f"#{i}")
    report.append("\n== C. Synthetic full screens (decoy thumbnails, exact boxes) ==")
    report.append(t.row("synthetic screens"))

    text = "\n\n".join(report[:1]) + "\n" + "\n\n".join(report[1:])
    print(text)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / "phase2_report.txt"
    out.write_text(text + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
