"""Secrets and paths from ``.env`` files (never hardcoded, never committed).

Searched in order, without overriding variables already set in the environment: the project's
``.env`` (development), then ``.env`` next to the user settings (``%APPDATA%\ChessTrainer``).
"""

from __future__ import annotations

import os
from pathlib import Path

from chesstrainer.settings.config import settings_path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def env_files() -> list[Path]:
    return [PROJECT_ROOT / ".env", settings_path().parent / ".env"]


def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dependency is declared
        return
    for path in env_files():
        if path.is_file():
            load_dotenv(path, override=False)


def anthropic_api_key() -> str | None:
    """``ANTHROPIC_API_KEY`` from the environment or a ``.env`` file; None when unset/empty."""
    load_env()
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip().strip('"').strip("'")
    return key or None
