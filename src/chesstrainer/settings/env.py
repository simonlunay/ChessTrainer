"""Secrets and paths: the OS credential store and ``.env`` files (never hardcoded or committed).

The Anthropic API key is looked up in this order:

1. ``ANTHROPIC_API_KEY`` set in the real environment (not one loaded from a ``.env`` file);
2. the OS credential store (Windows Credential Manager through ``keyring``), where the app's
   settings dialog saves the key the user pastes;
3. ``.env`` files: the project's ``.env`` (development only; the packaged app never looks there),
   then ``.env`` next to the user settings (``%APPDATA%\\ChessTrainer``), kept as a fallback.

Other variables (``STOCKFISH_PATH``) are loaded from the ``.env`` files into the environment
without overriding variables that are already set. The API key is never copied into the
environment, so the order above stays exact.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

from chesstrainer.paths import FROZEN, PROJECT_ROOT
from chesstrainer.settings.config import settings_path

log = logging.getLogger(__name__)

KEY_VAR = "ANTHROPIC_API_KEY"
#: Credential-store entry: service and user name (Windows: "ChessTrainer" generic credential).
KEYRING_SERVICE = "ChessTrainer"
KEYRING_USER = KEY_VAR
#: Where users create a key.
KEY_URL = "https://console.anthropic.com/settings/keys"
#: Measured in the phase 4-6 evals with Claude Haiku: $0.0003-0.0006 per new explanation.
COST_NOTE = (
    "Explanations use Claude Haiku and cost about $0.0003-0.0006 each (measured), so $1 covers "
    "roughly 2,000 of them. Repeated positions come from a local cache for free. Anthropic "
    "bills your account directly; with explanations off the app makes no API calls."
)

KeySource = Literal["environment", "credential store", ".env file"]


class KeyStoreError(RuntimeError):
    """The credential store is unavailable or refused the operation (readable message)."""


def env_files() -> list[Path]:
    user = settings_path().parent / ".env"
    return [user] if FROZEN else [PROJECT_ROOT / ".env", user]


def _env_values() -> dict[str, str]:
    """Variables from the ``.env`` files; earlier files win."""
    try:
        from dotenv import dotenv_values
    except ImportError:  # pragma: no cover - dependency is declared
        return {}
    values: dict[str, str] = {}
    for path in env_files():
        if path.is_file():
            for k, v in dotenv_values(path).items():
                if v is not None:
                    values.setdefault(k, v)
    return values


def load_env() -> None:
    """Load ``.env`` variables (except the API key) into the environment, without overriding."""
    for k, v in _env_values().items():
        if k != KEY_VAR:
            os.environ.setdefault(k, v)


def _clean(key: str | None) -> str | None:
    key = (key or "").strip().strip('"').strip("'")
    return key or None


def stored_api_key() -> str | None:
    """The key saved in the OS credential store, or None (also when no store is available)."""
    try:
        import keyring

        return _clean(keyring.get_password(KEYRING_SERVICE, KEYRING_USER))
    except Exception as err:  # noqa: BLE001 - no backend, locked store, ...: treat as unset
        log.info("credential store not readable: %s", err)
        return None


def store_api_key(key: str) -> None:
    """Save the key in the OS credential store. Raises :class:`KeyStoreError`."""
    clean = _clean(key)
    if clean is None:
        raise KeyStoreError("The key is empty.")
    try:
        import keyring

        keyring.set_password(KEYRING_SERVICE, KEYRING_USER, clean)
    except Exception as err:  # noqa: BLE001 - shown to the user
        raise KeyStoreError(f"Could not save the key in the credential store: {err}") from err


def delete_stored_api_key() -> bool:
    """Remove the saved key; True if one was removed."""
    try:
        import keyring
        from keyring.errors import PasswordDeleteError

        try:
            keyring.delete_password(KEYRING_SERVICE, KEYRING_USER)
        except PasswordDeleteError:
            return False
        return True
    except Exception as err:  # noqa: BLE001 - shown to the user
        raise KeyStoreError(f"Could not remove the key from the credential store: {err}") from err


def credential_store_name() -> str:
    """Readable name of the keyring backend ("" when none is usable)."""
    try:
        import keyring
        from keyring.backends.fail import Keyring as FailKeyring

        backend = keyring.get_keyring()
        if isinstance(backend, FailKeyring):
            return ""
        name = type(backend).__name__
        return "Windows Credential Manager" if name == "WinVaultKeyring" else name
    except Exception:  # noqa: BLE001
        return ""


def key_problem(key: str) -> str:
    """Why a pasted key cannot be right ("" if it looks plausible). Format check only."""
    key = key.strip()
    if not key:
        return "Paste a key first."
    if any(c.isspace() for c in key):
        return "The key contains spaces or line breaks; copy it again."
    if len(key) < 20:
        return "That is too short to be an API key."
    if not key.startswith("sk-"):
        return "Anthropic API keys start with 'sk-ant-'."
    return ""


def api_key_with_source() -> tuple[str | None, KeySource | None]:
    """The key and where it came from (see the module docstring for the order)."""
    key = _clean(os.environ.get(KEY_VAR))
    if key:
        return key, "environment"
    key = stored_api_key()
    if key:
        return key, "credential store"
    key = _clean(_env_values().get(KEY_VAR))
    if key:
        return key, ".env file"
    return None, None


def anthropic_api_key() -> str | None:
    """The Anthropic API key, or None when none is configured."""
    return api_key_with_source()[0]
