"""API key lookup order (environment > credential store > .env) and the key field."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from chesstrainer.settings import env

KEY_A = "sk-ant-api03-" + "a" * 40
KEY_B = "sk-ant-api03-" + "b" * 40


@pytest.fixture
def no_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """No env var, an empty CHESSTRAINER_HOME and no project .env."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CHESSTRAINER_HOME", str(tmp_path))
    monkeypatch.setattr(env, "env_files", lambda: [tmp_path / ".env"])
    return tmp_path


def test_no_key_anywhere(no_key: Path) -> None:
    assert env.api_key_with_source() == (None, None)


def test_dotenv_is_the_fallback(no_key: Path) -> None:
    (no_key / ".env").write_text(f"ANTHROPIC_API_KEY={KEY_B}\nSTOCKFISH_PATH=x\n")
    assert env.api_key_with_source() == (KEY_B, ".env file")
    env.load_env()
    assert "ANTHROPIC_API_KEY" not in os.environ  # never copied into the environment
    assert os.environ.get("STOCKFISH_PATH") == "x"
    os.environ.pop("STOCKFISH_PATH")


def test_credential_store_beats_dotenv(no_key: Path, memory_keyring: Any) -> None:
    (no_key / ".env").write_text(f"ANTHROPIC_API_KEY={KEY_B}\n")
    env.store_api_key(f"  {KEY_A}\n")
    assert memory_keyring.items[("ChessTrainer", "ANTHROPIC_API_KEY")] == KEY_A
    assert env.api_key_with_source() == (KEY_A, "credential store")
    assert env.delete_stored_api_key() and not env.delete_stored_api_key()
    assert env.api_key_with_source() == (KEY_B, ".env file")


def test_environment_beats_everything(no_key: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env.store_api_key(KEY_A)
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY_B)
    assert env.api_key_with_source() == (KEY_B, "environment")


def test_broken_store_counts_as_empty(no_key: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import keyring

    def boom(*_: object) -> None:
        raise RuntimeError("locked")

    monkeypatch.setattr(keyring, "get_password", boom)
    monkeypatch.setattr(keyring, "set_password", boom)
    assert env.stored_api_key() is None
    with pytest.raises(env.KeyStoreError, match="locked"):
        env.store_api_key(KEY_A)


def test_key_problem() -> None:
    assert env.key_problem(KEY_A) == ""
    assert "spaces" in env.key_problem("sk-ant-abc def" + "x" * 20)
    assert "short" in env.key_problem("sk-ant-1")
    assert "sk-ant-" in env.key_problem("x" * 40)
    assert env.key_problem("")


def test_key_box_saves_to_the_store(no_key: Path, memory_keyring: Any) -> None:
    from PyQt6.QtWidgets import QApplication, QLineEdit

    from chesstrainer.overlay.api_key_dialog import ApiKeyBox
    from chesstrainer.overlay.level_dialog import LevelDialog

    _ = QApplication.instance() or QApplication([])
    box = ApiKeyBox()
    assert box.edit.echoMode() == QLineEdit.EchoMode.Password
    assert "No key" in box.status.text() and box.remove.isHidden()
    box.edit.setText("not a key")
    assert not box.apply() and not box.changed and not memory_keyring.items
    box.edit.setText(KEY_A)
    assert box.apply() and box.changed and box.edit.text() == ""
    assert env.stored_api_key() == KEY_A
    assert "saved" in box.status.text() and KEY_A not in box.status.text()
    box._remove()
    assert env.stored_api_key() is None

    dialog = LevelDialog("beginner", with_api_key=True)
    assert dialog.key_box is not None
    dialog.key_box.edit.setText(KEY_B)
    dialog._ok()
    assert env.stored_api_key() == KEY_B and dialog.selected() == "beginner"


def test_app_picks_up_a_new_key(no_key: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from PyQt6.QtWidgets import QApplication

    import chesstrainer.overlay.app as app_mod
    from chesstrainer.settings.config import Settings

    _ = QApplication.instance() or QApplication([])
    monkeypatch.setattr(app_mod, "find_stockfish", _no_engine)
    trainer = app_mod.TrainerApp(Settings(level_chosen=True, api_key_prompted=True))
    try:
        assert trainer.explainer is not None and not trainer.explainer.available
        env.store_api_key(KEY_A)
        monkeypatch.setattr(
            "chesstrainer.overlay.api_key_dialog.edit_api_key", lambda parent=None: True
        )
        trainer.edit_api_key()
        assert trainer.explainer is not None and trainer.explainer.available
        assert trainer.explain_worker is not None
    finally:
        if trainer.explain_worker is not None:
            trainer.explain_worker.close()
        trainer.vision.stop()


def _no_engine(_: object) -> object:
    from chesstrainer.engine.locate import EngineSearch

    return EngineSearch(None, ["test"])
