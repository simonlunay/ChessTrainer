"""First-launch prompt: pick a skill level (saved; changeable later in the side panel)."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from chesstrainer.levels import Level, all_levels, get_level


def _describe(lv: Level) -> str:
    shown = "the best move" if lv.candidates == 1 else f"the top {lv.candidates} moves"
    white = ", ".join(lv.openings["white"])
    black = ", ".join(lv.openings["black"])
    return (
        f"Shows {shown}; explanations of {lv.explanation.sentences} sentence(s).\n"
        f"Suggested openings - as White: {white}; as Black: {black}."
    )


class LevelDialog(QDialog):
    def __init__(self, current: str, parent: QWidget | None = None) -> None:
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
        self.group = QButtonGroup(self)
        for i, lv in enumerate(all_levels()):
            button = QRadioButton(lv.label)
            button.setProperty("level", lv.key)
            button.setChecked(lv.key == current)
            self.group.addButton(button, i)
            layout.addWidget(button)
            detail = QLabel(_describe(lv))
            detail.setWordWrap(True)
            detail.setStyleSheet("color: gray; margin-left: 22px")
            layout.addWidget(detail)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
        self.resize(460, self.sizeHint().height())

    def selected(self) -> str:
        button = self.group.checkedButton()
        key = button.property("level") if button is not None else None
        return get_level(str(key)).key if key else get_level("beginner").key


def ask_level(current: str) -> str:
    """Show the prompt (modal) and return the chosen level key."""
    dialog = LevelDialog(current)
    dialog.exec()
    return dialog.selected()
