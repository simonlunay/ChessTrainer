"""Save a mosaic of synthetic boards and square crops for eyeballing the data generator."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from chesstrainer.vision.dataset import train_split
from chesstrainer.vision.labels import CLASS_NAMES
from chesstrainer.vision.synth import AssetLibrary, degrade, render_scene, sample_crops


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/preview"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    lib = AssetLibrary()
    split = train_split()

    for i in range(6):
        scene = degrade(render_scene(rng, lib, split), rng)
        img = scene.image.copy()
        x, y, w, h = (int(round(v)) for v in scene.bbox)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 1)
        cv2.imwrite(str(args.out / f"board_{i}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    crops, labels = sample_crops(rng, lib, split)
    for _ in range(5):
        c, lab = sample_crops(rng, lib, split)
        crops, labels = np.concatenate([crops, c]), np.concatenate([labels, lab])
    n = min(len(crops), 96)
    grid = np.full((8 * 80, 12 * 64, 3), 255, np.uint8)
    for k in range(n):
        r, col = divmod(k, 12)
        grid[r * 80 : r * 80 + 64, col * 64 : col * 64 + 64] = crops[k]
        cv2.putText(
            grid,
            CLASS_NAMES[labels[k]],
            (col * 64 + 2, r * 80 + 77),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
        )
    cv2.imwrite(str(args.out / "crops.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"wrote previews to {args.out}")


if __name__ == "__main__":
    main()
