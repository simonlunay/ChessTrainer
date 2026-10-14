"""First-launch prompt: pick a skill level (saved; changeable later in the side panel) and,
optionally, paste an Anthropic API key (saved in the OS credential store)."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from chesstrainer.levels import Level, all_levels, get_level
from chesstrainer.overlay.api_key_dialog import ApiKeyBox


def _describe(lv: Level) -> str:
    shown = "the best move" if lv.candidates == 1 else f"the top {lv.candidates} moves"
    white = ", ".join(lv.openings["white"])
    black = ", ".join(lv.openings["black"])
    return (
        f"Shows {shown}; explanations of {lv.explanation.sentences} sentence(s).\n"
        f"Suggested openings - as White: {white}; as Black: {black}."
    )


class LevelDialog(QDialog):
    def __init__(
        self, current: str, parent: QWidget | None = None, *, with_api_key: bool = False
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("ChessTrainer - your level")
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        layout = QVBoxLayout(self)
        intro = QLabel(
            "How strong are you? This sets how many moves are shown and how the explanations "
            "are written. You can change it any time in the side panel."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        # Levels on the left, the optional API key field on the right: one column of both would be
        # taller than a 1080p screen at 150% scaling.
        columns = QHBoxLayout()
        levels = QVBoxLayout()
        columns.addLayout(levels, 3)
        layout.addLayout(columns)
        self.group = QButtonGroup(self)
        for i, lv in enumerate(all_levels()):
            button = QRadioButton(lv.label)
            button.setProperty("level", lv.key)
            button.setChecked(lv.key == current)
            self.group.addButton(button, i)
            levels.addWidget(button)
            detail = QLabel(_describe(lv))
            detail.setWordWrap(True)
            detail.setStyleSheet("color: gray; margin-left: 22px")
            levels.addWidget(detail)
        levels.addStretch(1)
        self.key_box: ApiKeyBox | None = None
        if with_api_key:
            self.key_box = ApiKeyBox()
            right = QVBoxLayout()
            right.addWidget(self.key_box)
            right.addStretch(1)
            columns.addSpacing(16)
            columns.addLayout(right, 2)
            self.setWindowTitle("ChessTrainer - welcome")
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self._ok)
        layout.addWidget(buttons)
        width = 820 if with_api_key else 460
        self.resize(width, self.heightForWidth(width) if self.hasHeightForWidth() else 1)

    def _ok(self) -> None:
        if self.key_box is None or self.key_box.apply():
            self.accept()

    def selected(self) -> str:
        button = self.group.checkedButton()
        key = button.property("level") if button is not None else None
        return get_level(str(key)).key if key else get_level("beginner").key


def ask_level(current: str, with_api_key: bool = False) -> str:
    """Show the prompt (modal) and return the chosen level key."""
    dialog = LevelDialog(current, with_api_key=with_api_key)
    dialog.exec()
    return dialog.selected()
