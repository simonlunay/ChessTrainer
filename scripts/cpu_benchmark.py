"""Piece recognition on a CPU-only machine: accuracy and timing with the GPU hidden.

    python -m uv run python scripts/cpu_benchmark.py      # -> data/eval/cpu_report.txt

CUDA is hidden (``CUDA_VISIBLE_DEVICES=-1``) before anything imports PyTorch, so ``device="auto"``
must fall back to the CPU. On the real boards in ``data/real`` (the user's screenshots and the
lichess / chess.com renders) it measures, per backend:
  - exact positions (orientation given; the classifier is what differs between backends);
  - the classifier alone on one board (64 crops), median / p90 over repeats;
  - a whole frame on a new position: board re-verification + crops + classifier + position
    logic (``BoardReader.process`` with the tracker already locked on the board);
  - the first frame on a full screenshot: board detection from scratch + the rest.
Backends: ONNX through OpenCV (what the packaged app uses), with all cores and with 1 thread
(a slow machine), and PyTorch on the CPU (source checkout without a GPU).
No screen use: images are read from disk.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import csv  # noqa: E402
import platform  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from chesstrainer.settings.config import Settings  # noqa: E402
from chesstrainer.vision.geometry import split_board  # noqa: E402
from chesstrainer.vision.reader import BoardReader, read_image  # noqa: E402
from chesstrainer.vision.recognizer import PieceRecognizer, torch_available  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data" / "real"
REPORT = ROOT / "data" / "eval" / "cpu_report.txt"
FREE_PT = ROOT / "data" / "models" / "square_classifier_free.pt"


def pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def timed(fn: Callable[[], object], repeats: int) -> list[float]:
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000)
    return out


def bench(name: str, make: Callable[[], PieceRecognizer], rows: list[dict[str, str]]) -> list[str]:
    t0 = time.perf_counter()
    rec = make()
    load_ms = (time.perf_counter() - t0) * 1000
    images = {r["file"]: np.asarray(Image.open(REAL / r["file"]).convert("RGB")) for r in rows}
    exact = {"your screenshots": [0, 0], "renders": [0, 0]}
    classify_ms: list[float] = []
    frame_ms: list[float] = []
    first_ms: list[float] = []
    first_inference = None
    for r in rows:
        img = images[r["file"]]
        group = "renders" if r["file"].startswith(("lichess_", "chesscom_")) else "your screenshots"
        wb = r["white_bottom"] == "1"
        settings = Settings(orientation="white" if wb else "black", device="auto")
        reader = BoardReader(settings, recognizer=rec)
        rec._last_probs = None
        t = time.perf_counter()
        state = read_image(img, reader)
        if group == "your screenshots":
            first_ms.append((time.perf_counter() - t) * 1000)
        exact[group][1] += 1
        if state.reading is not None:
            exact[group][0] += state.reading.board.board_fen() == r["fen"].split()[0]
        if state.detection is None:
            continue
        box = state.detection.box
        crops = split_board(img, rec.crop_size, bbox=box.as_tuple())
        if first_inference is None:
            first_inference = timed(lambda c=crops: rec.classify_crops(c), 1)[0]
        classify_ms += timed(lambda c=crops: rec.classify_crops(c), 5)

        def frame(img: np.ndarray = img, reader: BoardReader = reader) -> None:
            rec._last_probs = None  # as if the position just changed
            reader.process(img, now=time.perf_counter())

        frame_ms += timed(frame, 5)
    return [
        f"== {name} ==",
        f"  backend {rec.backend}, device {rec.device}, model {rec.model_path.name}, "
        f"load {load_ms:.0f} ms, first inference {first_inference:.1f} ms",
        f"  exact positions: your screenshots {exact['your screenshots'][0]}/"
        f"{exact['your screenshots'][1]}, lichess+chess.com renders {exact['renders'][0]}/"
        f"{exact['renders'][1]}",
        f"  classifier, 1 board (64 crops): median {statistics.median(classify_ms):.1f} ms, "
        f"p90 {pct(classify_ms, 90):.1f} ms (n={len(classify_ms)})",
        f"  frame on a new position (tracked board): median {statistics.median(frame_ms):.1f} ms,"
        f" p90 {pct(frame_ms, 90):.1f} ms",
        f"  first read of a full screenshot (detection from scratch): median "
        f"{statistics.median(first_ms):.0f} ms, max {max(first_ms):.0f} ms",
    ]


def main() -> None:
    rows = list(csv.DictReader((REAL / "labels.csv").open(newline="")))
    lines = [
        "CPU-only piece recognition (CUDA hidden with CUDA_VISIBLE_DEVICES=-1)",
        f"CPU: {platform.processor()} ({os.cpu_count()} logical cores), "
        f"OpenCV {cv2.__version__}, {len(rows)} real boards",
    ]
    if torch_available():
        import torch

        lines.append(
            f"torch {torch.__version__}, cuda.is_available() = {torch.cuda.is_available()}"
        )
        auto = PieceRecognizer(FREE_PT, device="auto", backend="torch")
        lines.append(f"PieceRecognizer(device='auto', backend='torch') -> device {auto.device}")
    threads = cv2.getNumThreads()
    onnx = lambda: PieceRecognizer(backend="onnx")  # noqa: E731
    lines += bench(f"ONNX / OpenCV DNN, {threads} threads (packaged app)", onnx, rows)
    cv2.setNumThreads(1)
    lines += bench("ONNX / OpenCV DNN, 1 thread (slow-PC stand-in)", onnx, rows)
    cv2.setNumThreads(threads)
    if torch_available() and FREE_PT.exists():
        torch_cpu = lambda: PieceRecognizer(FREE_PT, device="cpu", backend="torch")  # noqa: E731
        lines += bench("PyTorch CPU, free model (source checkout without a GPU)", torch_cpu, rows)
    text = "\n".join(lines)
    print(text)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
