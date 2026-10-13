"""Fetch piece sets and board textures used by the synthetic data generator."""

from __future__ import annotations

import argparse
import logging

from chesstrainer.vision.assets import PIECE_DIR, available_piece_sets, prepare_assets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download and re-rasterize")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    prepare_assets(force=args.force)
    print(f"{len(available_piece_sets())} piece sets ready in {PIECE_DIR}")


if __name__ == "__main__":
    main()
