"""Opening picker: searchable list of every Lichess opening name, side, and training mode.

The openings recommended for the player's level (``levels.yaml``) are listed first for the
chosen side; any other opening or variation can be found with the search box.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from chesstrainer.levels import Level
from chesstrainer.openings.book import Book, fold


@dataclass(frozen=True)
class OpeningChoice:
    name: str
    side: str  # "white" / "black"
    mode: str  # "guided" / "practice"


class OpeningDialog(QDialog):
    def __init__(
        self,
        book: Book,
        level: Level,
        current: OpeningChoice | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Choose an opening")
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.resize(460, 560)
        self.book = book
        self.level = level
        self._names = book.names()

        self.white = QRadioButton("White")
        self.black = QRadioButton("Black")
        sides = QButtonGroup(self)
        sides.addButton(self.white)
        sides.addButton(self.black)
        self.guided = QRadioButton("Guided (arrows + explanations)")
        self.practice = QRadioButton("Practice (no arrows, spaced repetition)")
        modes = QButtonGroup(self)
        modes.addButton(self.guided)
        modes.addButton(self.practice)
        side = current.side if current else "white"
        (self.white if side == "white" else self.black).setChecked(True)
        mode = current.mode if current else "guided"
        (self.guided if mode == "guided" else self.practice).setChecked(True)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search openings (e.g. caro, najdorf, london)…")
        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(lambda _item: self.accept())

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        side_row = QHBoxLayout()
        side_row.addWidget(QLabel("I play:"))
        side_row.addWidget(self.white)
        side_row.addWidget(self.black)
        side_row.addStretch(1)
        layout = QVBoxLayout(self)
        layout.addLayout(side_row)
        layout.addWidget(self.guided)
        layout.addWidget(self.practice)
        layout.addWidget(self.search)
        layout.addWidget(self.list, 1)
        layout.addWidget(buttons)

        self.search.textChanged.connect(self._fill)
        self.white.toggled.connect(lambda _on: self._fill())
        self._fill()
        if current is not None:
            self.select(current.name)

    def side(self) -> str:
        return "white" if self.white.isChecked() else "black"

    def recommended(self) -> tuple[str, ...]:
        return self.level.openings[self.side()]

    def _fill(self) -> None:
        query = self.search.text()
        self.list.clear()
        words = fold(query).split()
        rec = [n for n in self.recommended() if all(w in fold(n) for w in words)]
        if rec:
            self._header(f"Recommended for {self.level.name} ({self.side()})")
            for name in rec:
                item = QListWidgetItem(f"★ {name}")
                item.setData(Qt.ItemDataRole.UserRole, name)
                self.list.addItem(item)
            self._header("All openings")
        names = self.book.search(query) if words else self._names
        for name in names:
            item = QListWidgetItem(name if ":" not in name else f"    {name}")
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.list.addItem(item)
        for i in range(self.list.count()):
            if self.list.item(i).data(Qt.ItemDataRole.UserRole):  # type: ignore[union-attr]
                self.list.setCurrentRow(i)
                break

    def _header(self, text: str) -> None:
        item = QListWidgetItem(text)
        font = QFont(item.font())
        font.setBold(True)
        item.setFont(font)
        item.setFlags(Qt.ItemFlag.NoItemFlags)
        self.list.addItem(item)

    def select(self, name: str) -> None:
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == name:
                self.list.setCurrentRow(i)
                return

    def choice(self) -> OpeningChoice | None:
        item = self.list.currentItem()
        name = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        if not name:
            return None
        mode = "guided" if self.guided.isChecked() else "practice"
        return OpeningChoice(str(name), self.side(), mode)


def ask_opening(
    book: Book, level: Level, current: OpeningChoice | None = None
) -> OpeningChoice | None:
    dialog = OpeningDialog(book, level, current)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    return dialog.choice()
