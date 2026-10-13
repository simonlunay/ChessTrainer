"""Classifier evaluation: per-square accuracy, per-class metrics, confusion matrix, board accuracy.

Real screenshots live in ``data/real/``: tightly cropped board images plus ``labels.csv`` with
columns ``file,fen,white_bottom``. ``fen`` needs only the piece-placement field.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from chesstrainer.vision.dataset import DATA_DIR, to_tensor
from chesstrainer.vision.geometry import CROP_SIZE, ImageU8, cell_to_square, split_board
from chesstrainer.vision.labels import CLASS_NAMES, NUM_CLASSES, piece_to_label
from chesstrainer.vision.model import SquareClassifier

REAL_DIR = DATA_DIR / "real"


@dataclass
class EvalReport:
    name: str
    square_accuracy: float
    board_accuracy: float  # fraction of boards with all 64 squares right
    n_boards: int
    confusion: NDArray[np.int64]  # [true, pred]

    def per_class(self) -> dict[str, tuple[float, float, int]]:
        """class -> (precision, recall, support)."""
        out = {}
        for i, name in enumerate(CLASS_NAMES):
            tp = self.confusion[i, i]
            support = int(self.confusion[i].sum())
            predicted = self.confusion[:, i].sum()
            out[name] = (
                float(tp / predicted) if predicted else 0.0,
                float(tp / support) if support else 0.0,
                support,
            )
        return out

    def format(self) -> str:
        lines = [
            f"== {self.name}: {self.n_boards} boards ==",
            f"per-square accuracy: {self.square_accuracy:.4%}",
            f"full-board accuracy: {self.board_accuracy:.2%}",
            f"{'class':>6} {'prec':>7} {'recall':>7} {'n':>6}",
        ]
        for name, (p, r, n) in self.per_class().items():
            lines.append(f"{name:>6} {p:7.4f} {r:7.4f} {n:6d}")
        errors = [
            (int(self.confusion[t, p]), CLASS_NAMES[t], CLASS_NAMES[p])
            for t in range(NUM_CLASSES)
            for p in range(NUM_CLASSES)
            if t != p and self.confusion[t, p]
        ]
        if errors:
            top = ", ".join(f"{t}->{p} x{n}" for n, t, p in sorted(errors, reverse=True)[:8])
            lines.append(f"top confusions (true->pred): {top}")
        return "\n".join(lines)


@torch.no_grad()
def predict(
    model: SquareClassifier, crops: ImageU8, device: str, batch: int = 1024
) -> NDArray[np.float32]:
    """Softmax probabilities for uint8 NHWC crops, shape (N, 13)."""
    model.eval()
    out = []
    for i in range(0, len(crops), batch):
        x = to_tensor(crops[i : i + batch]).to(device)
        out.append(torch.softmax(model(x), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, NUM_CLASSES), np.float32)


def evaluate_boards(
    model: SquareClassifier, x: ImageU8, y: NDArray[np.int64], device: str, name: str
) -> EvalReport:
    """Evaluate on boards: ``x`` (B, 64, H, W, 3), ``y`` (B, 64)."""
    b = x.shape[0]
    pred = predict(model, x.reshape(-1, *x.shape[2:]), device).argmax(1).reshape(b, 64)
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    np.add.at(conf, (y.ravel(), pred.ravel()), 1)
    return EvalReport(
        name=name,
        square_accuracy=float((pred == y).mean()),
        board_accuracy=float((pred == y).all(axis=1).mean()),
        n_boards=b,
        confusion=conf,
    )


def board_labels(fen: str, white_bottom: bool) -> NDArray[np.int64]:
    board = chess.Board(fen.split()[0] + " w - - 0 1")
    return np.array(
        [piece_to_label(board.piece_at(cell_to_square(c, white_bottom))) for c in range(64)],
        dtype=np.int64,
    )


def load_real_screenshots(
    real_dir: Path = REAL_DIR,
) -> tuple[ImageU8, NDArray[np.int64], list[str]]:
    """Load labelled real board screenshots (see module docstring). Empty arrays if none."""
    labels_csv = real_dir / "labels.csv"
    if not labels_csv.exists():
        return np.zeros((0, 64, CROP_SIZE, CROP_SIZE, 3), np.uint8), np.zeros((0, 64), np.int64), []
    xs, ys, names = [], [], []
    with labels_csv.open(newline="") as fh:
        for row in csv.DictReader(fh):
            with Image.open(real_dir / row["file"]) as im:  # Pillow: handles GIF/WebP too
                img = np.asarray(im.convert("RGB"), dtype=np.uint8)
            white_bottom = row.get("white_bottom", "1").strip().lower() in {
                "1",
                "true",
                "yes",
                "white",
            }
            xs.append(split_board(img))
            ys.append(board_labels(row["fen"], white_bottom))
            names.append(row["file"])
    return np.stack(xs), np.stack(ys), names
