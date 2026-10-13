"""Application entry point. The overlay app arrives in phase 3; until then this points at tools."""

from __future__ import annotations


def main() -> None:
    print(
        "ChessTrainer (phase 1).\n"
        "  Train the classifier: python -m chesstrainer.vision.train\n"
        "  Evaluate it:          python -m chesstrainer.vision.train --eval-only"
    )


if __name__ == "__main__":
    main()
