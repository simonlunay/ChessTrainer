"""Export a trained square classifier to ONNX for the packaged app (no PyTorch at runtime).

    python -m uv run python -m chesstrainer.vision.export_onnx            # the free-licence model
    python -m uv run python -m chesstrainer.vision.export_onnx --checkpoint data/models/x.pt

Writes ``<checkpoint>.onnx`` plus a ``.json`` sidecar (class order, crop size, source) and checks
that OpenCV's DNN module (the runtime backend) gives the same predictions as PyTorch on the
validation, held-out and real-screenshot sets.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from chesstrainer.vision.dataset import load_or_build_eval
from chesstrainer.vision.evaluate import load_real_screenshots
from chesstrainer.vision.model import FREE_CHECKPOINT, load_checkpoint
from chesstrainer.vision.onnx_backend import OnnxClassifier, sidecar_path

log = logging.getLogger(__name__)


def export(checkpoint: Path, out: Path | None = None) -> Path:
    model, meta = load_checkpoint(checkpoint, "cpu")
    out = out or checkpoint.with_suffix(".onnx")
    dummy = torch.zeros(1, 3, meta.crop_size, meta.crop_size)
    torch.onnx.export(
        model,
        (dummy,),
        str(out),
        input_names=["crops"],
        output_names=["logits"],
        dynamic_axes={"crops": {0: "n"}, "logits": {0: "n"}},
        opset_version=17,
        dynamo=False,
    )
    sidecar_path(out).write_text(
        json.dumps(
            {
                "class_names": list(meta.class_names),
                "crop_size": meta.crop_size,
                "source": checkpoint.name,
                "metrics": meta.metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out


def compare(checkpoint: Path, onnx_path: Path) -> dict[str, tuple[int, int, float]]:
    """Per eval set: (squares, predictions that differ, max probability difference)."""
    model, _ = load_checkpoint(checkpoint, "cpu")
    onnx = OnnxClassifier(onnx_path)
    sets = {name: load_or_build_eval(name)[0] for name in ("val", "heldout")}
    real = load_real_screenshots()[0]
    if len(real):
        sets["real"] = real
    out = {}
    for name, boards in sets.items():
        crops = boards.reshape(-1, *boards.shape[2:])
        diff, max_p = 0, 0.0
        for i in range(0, len(crops), 1024):
            batch = crops[i : i + 1024]
            x = torch.from_numpy(np.ascontiguousarray(batch)).permute(0, 3, 1, 2).float() / 255.0
            with torch.no_grad():
                p_torch = torch.softmax(model(x), dim=1).numpy()
            p_onnx = onnx.probabilities(batch)
            diff += int((p_torch.argmax(1) != p_onnx.argmax(1)).sum())
            max_p = max(max_p, float(np.abs(p_torch - p_onnx).max()))
        out[name] = (len(crops), diff, max_p)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the square classifier to ONNX.")
    parser.add_argument("--checkpoint", type=Path, default=FREE_CHECKPOINT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    out = export(args.checkpoint, args.out)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    for name, (n, diff, max_p) in compare(args.checkpoint, out).items():
        print(f"{name}: {diff}/{n} predictions differ from PyTorch, max prob diff {max_p:.2e}")


if __name__ == "__main__":
    main()
