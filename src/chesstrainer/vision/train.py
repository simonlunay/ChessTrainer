"""Train the square classifier on streamed synthetic data.

    uv run python -m chesstrainer.vision.train --steps 12000

Writes ``data/models/square_classifier.pt`` (best validation checkpoint) and
``data/models/eval_report.txt`` with metrics on the validation, held-out-style and
real-screenshot sets.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from chesstrainer.vision.dataset import SyntheticSquares, load_or_build_eval, train_split
from chesstrainer.vision.evaluate import EvalReport, evaluate_boards, load_real_screenshots
from chesstrainer.vision.model import (
    DEFAULT_CHECKPOINT,
    CheckpointMeta,
    SquareClassifier,
    load_checkpoint,
    save_checkpoint,
)

log = logging.getLogger(__name__)


def run_evals(model: SquareClassifier, device: str) -> list[EvalReport]:
    reports = []
    for name in ("val", "heldout"):
        x, y = load_or_build_eval(name)
        reports.append(evaluate_boards(model, x, y, device, name))
    x, y, _ = load_real_screenshots()
    if len(x):
        reports.append(evaluate_boards(model, x, y, device, "real"))
    return reports


def train(
    steps: int,
    batch_size: int,
    lr: float,
    workers: int,
    eval_every: int,
    seed: int,
    width: int,
) -> list[EvalReport]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    split = train_split()
    log.info(
        "training on %d piece sets, %d textures, device=%s",
        len(split.piece_sets),
        len(split.textures),
        device,
    )

    log.info("preparing eval sets (cached under data/eval after the first run)")
    val_x, val_y = load_or_build_eval("val")
    load_or_build_eval("heldout")

    loader = DataLoader(
        SyntheticSquares(split, seed=seed),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=device == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )
    model = SquareClassifier(width=width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    log.info("model parameters: %d", sum(p.numel() for p in model.parameters()))

    best = -1.0
    t0 = time.time()
    running = 0.0
    model.train()
    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device, enabled=device == "cuda"):
            loss = loss_fn(model(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        running = 0.98 * running + 0.02 * loss.item() if step > 1 else loss.item()

        if step % eval_every == 0 or step == steps:
            report = evaluate_boards(model, val_x, val_y, device, "val")
            rate = step * batch_size / (time.time() - t0)
            log.info(
                "step %5d/%d loss %.4f val-square %.4f%% val-board %.1f%% (%.0f samples/s)",
                step,
                steps,
                running,
                100 * report.square_accuracy,
                100 * report.board_accuracy,
                rate,
            )
            if report.square_accuracy >= best:
                best = report.square_accuracy
                save_checkpoint(model, DEFAULT_CHECKPOINT, CheckpointMeta(width=width))
            model.train()
        if step >= steps:
            break

    best_model, meta = load_checkpoint(DEFAULT_CHECKPOINT, device)
    reports = run_evals(best_model, device)
    meta.metrics = {f"{r.name}_square_accuracy": r.square_accuracy for r in reports} | {
        f"{r.name}_board_accuracy": r.board_accuracy for r in reports
    }
    save_checkpoint(best_model, DEFAULT_CHECKPOINT, meta)
    text = "\n\n".join(r.format() for r in reports)
    (DEFAULT_CHECKPOINT.parent / "eval_report.txt").write_text(text + "\n", encoding="utf-8")
    (DEFAULT_CHECKPOINT.parent / "metrics.json").write_text(
        json.dumps(meta.metrics, indent=2), encoding="utf-8"
    )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the chess square classifier.")
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--workers", type=int, default=max(1, min(10, (os.cpu_count() or 2) - 2)))
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--eval-only", action="store_true", help="evaluate the saved checkpoint")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    if args.eval_only:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = load_checkpoint(DEFAULT_CHECKPOINT, device)
        reports = run_evals(model, device)
    else:
        reports = train(
            args.steps,
            args.batch_size,
            args.lr,
            args.workers,
            args.eval_every,
            args.seed,
            args.width,
        )
    for r in reports:
        print(r.format(), end="\n\n")


if __name__ == "__main__":
    main()
