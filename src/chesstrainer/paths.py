"""Where files live in a source checkout and in the packaged (PyInstaller) app.

Source checkout: models and engines under ``data/`` of the project. Packaged app: read-only
resources under the bundle folder (``sys._MEIPASS``), and nothing from the project tree (in
particular never a project ``.env``).
"""

from __future__ import annotations

import sys
from pathlib import Path

#: True inside the PyInstaller build.
FROZEN = bool(getattr(sys, "frozen", False))

#: The project folder (source checkout; meaningless when frozen).
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def bundle_dir() -> Path:
    """Read-only resources: the PyInstaller bundle, or the project's ``data`` folder."""
    if FROZEN:
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return PROJECT_ROOT / "data"


def model_dir() -> Path:
    return bundle_dir() / "models"


#: ONNX model name in both layouts (the packaged app only ships this one).
ONNX_MODEL = "square_classifier_free.onnx"
