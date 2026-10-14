"""Secrets and paths from ``.env`` files (never hardcoded, never committed).

Searched in order, without overriding variables already set in the environment: the project's
``.env`` (development only; the packaged app never looks there), then ``.env`` next to the user
settings (``%APPDATA%\ChessTrainer``).
"""

from __future__ import annotations

import os
from pathlib import Path

from chesstrainer.paths import FROZEN, PROJECT_ROOT
from chesstrainer.settings.config import settings_path


def env_files() -> list[Path]:
    user = settings_path().parent / ".env"
    return [user] if FROZEN else [PROJECT_ROOT / ".env", user]


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
