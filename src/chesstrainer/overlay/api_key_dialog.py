"""Anthropic API key field: paste a key, saved in the OS credential store (Windows Credential
Manager). Shown in the first-launch dialog and from the panel's "API key…" button.

The ``.env`` files stay a fallback (see ``settings/env.py``); the key itself is never shown back,
logged or written to settings.json.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from chesstrainer.settings.env import (
    COST_NOTE,
    KEY_URL,
    KeyStoreError,
    api_key_with_source,
    credential_store_name,
    delete_stored_api_key,
    env_files,
    key_problem,
    store_api_key,
)


def _source_text() -> str:
    key, source = api_key_with_source()
    if key is None:
        return "No key is set yet: explanations stay off until you add one."
    if source == "credential store":
        return f"A key is saved in {credential_store_name() or 'the credential store'}."
    if source == "environment":
        return "A key is set by the ANTHROPIC_API_KEY environment variable."
    return "A key is set in a .env file (" + ", ".join(str(p) for p in env_files()) + ")."


class ApiKeyBox(QWidget):
    """The key field with its link, cost note and status. ``apply()`` saves what was typed."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel("<b>Anthropic API key</b> (optional, only for explanations)")
        layout.addWidget(title)
        row = QHBoxLayout()
        self.edit = QLineEdit()
        self.edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.edit.setPlaceholderText("sk-ant-...")
        self.edit.setClearButtonEnabled(True)
        row.addWidget(self.edit, 1)
        self.show_box = QCheckBox("Show")
        self.show_box.toggled.connect(
            lambda on: self.edit.setEchoMode(
                QLineEdit.EchoMode.Normal if on else QLineEdit.EchoMode.Password
            )
        )
        row.addWidget(self.show_box)
        layout.addLayout(row)
        link = QLabel(f"Get a key: <a href='{KEY_URL}'>console.anthropic.com</a>")
        link.setOpenExternalLinks(True)
        link.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        layout.addWidget(link)
        note = QLabel(COST_NOTE)
        note.setWordWrap(True)
        note.setStyleSheet("color: gray")
        layout.addWidget(note)
        status_row = QHBoxLayout()
        self.status = QLabel()
        self.status.setWordWrap(True)
        status_row.addWidget(self.status, 1)
        self.remove = QPushButton("Remove saved key")
        self.remove.clicked.connect(self._remove)
        status_row.addWidget(self.remove)
        layout.addLayout(status_row)
        self.error = QLabel()
        self.error.setWordWrap(True)
        self.error.setStyleSheet("color: #b00020")
        self.error.hide()
        layout.addWidget(self.error)
        self.changed = False
        self._refresh()

    def _refresh(self) -> None:
        self.status.setText(_source_text())
        self.remove.setVisible(api_key_with_source()[1] == "credential store")

    def _show_error(self, text: str) -> None:
        self.error.setText(text)
        self.error.setVisible(bool(text))

    def _remove(self) -> None:
        try:
            removed = delete_stored_api_key()
        except KeyStoreError as err:
            self._show_error(str(err))
            return
        self.changed = self.changed or removed
        self._show_error("")
        self._refresh()

    def apply(self) -> bool:
        """Save the typed key (if any). False, with the reason shown, if it cannot be saved."""
        text = self.edit.text().strip()
        if not text:
            return True
        problem = key_problem(text)
        if problem:
            self._show_error(problem)
            return False
        try:
            store_api_key(text)
        except KeyStoreError as err:
            self._show_error(f"{err}. You can put it in {env_files()[-1]} instead.")
            return False
        self.edit.clear()
        self.changed = True
        self._show_error("")
        self._refresh()
        return True


class ApiKeyDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("ChessTrainer - API key")
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        layout = QVBoxLayout(self)
        self.box = ApiKeyBox()
        layout.addWidget(self.box)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Close
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(480, self.sizeHint().height())

    def _save(self) -> None:
        if self.box.apply():
            self.accept()


def edit_api_key(parent: QWidget | None = None) -> bool:
    """Show the dialog (modal); True if the stored key changed."""
    dialog = ApiKeyDialog(parent)
    dialog.exec()
    return dialog.box.changed
