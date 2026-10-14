"""Game review window: load games (pasted PGN, Lichess, Chess.com), review them with Stockfish,
step through the moves with the player's mistakes marked, read explanations of the biggest ones,
and get a summary of mistake types to work on."""

from __future__ import annotations

import html
import logging
import threading
from collections.abc import Callable
from pathlib import Path

import chess
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QColor, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from chesstrainer.explain.explainer import Explainer, Explanation
from chesstrainer.explain.facts import eval_number
from chesstrainer.levels import Level
from chesstrainer.overlay.board_view import BEST_COLOR, PLAYED_COLOR, BoardView
from chesstrainer.review.analysis import Evaluator, GameReview, MoveReview, StockfishEvaluator
from chesstrainer.review.games import (
    FetchError,
    GameRecord,
    fetch_chesscom,
    fetch_lichess,
    parse_pgn,
)
from chesstrainer.review.report import summarize, summary_text
from chesstrainer.review.runner import ReviewCallbacks, ReviewRunner
from chesstrainer.settings.config import Settings, save_settings

log = logging.getLogger(__name__)

SEVERITY_COLOR = {"blunder": "#c62828", "mistake": "#e67700", "inaccuracy": "#b8a000"}
SEVERITY_MARK = {"blunder": "??", "mistake": "?", "inaccuracy": "?!"}

Fetcher = Callable[[str, int], list[GameRecord]]


class ReviewWindow(QWidget):
    fetched = pyqtSignal(object, str)  # list[GameRecord], error
    progress = pyqtSignal(int, int, int)
    game_done = pyqtSignal(int, object)
    explained = pyqtSignal(int, int, object, object, str)
    finished = pyqtSignal(object)
    closed = pyqtSignal()

    def __init__(
        self,
        settings: Settings,
        level: Level,
        engine_path: Path | None,
        make_explainer: Callable[[], tuple[Explainer | None, str]],
        *,
        explanations_on: Callable[[], bool] = lambda: True,
        make_evaluator: Callable[[], Evaluator] | None = None,
        fetchers: dict[str, Fetcher] | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("ChessTrainer - Game review")
        self.resize(1200, 760)
        self.settings = settings
        self.level = level
        self.engine_path = engine_path
        self.make_explainer = make_explainer
        self.explanations_on = explanations_on
        self._make_evaluator = make_evaluator
        self.fetchers: dict[str, Fetcher] = fetchers or {
            "lichess": fetch_lichess,
            "chess.com": fetch_chesscom,
        }
        self.loaded: list[GameRecord] = []
        self.reviews: dict[int, GameReview] = {}  # index into the reviewed list
        self.reviewed_games: list[GameRecord] = []
        self.explanations: dict[tuple[int, int], tuple[str, str]] = {}  # (game, ply) -> text, tip
        self.explain_note = ""
        self.runner: ReviewRunner | None = None
        self.current = 0  # reviewed game shown
        self.ply = 0  # position shown: after `ply` moves

        self._build()
        self.fetched.connect(self._on_fetched)
        self.progress.connect(self._on_progress)
        self.game_done.connect(self._on_game_done)
        self.explained.connect(self._on_explained)
        self.finished.connect(self._on_finished)

    # -- layout ------------------------------------------------------------------------------

    def _build(self) -> None:
        s = self.settings
        self.tabs = QTabWidget()
        paste = QWidget()
        self.pgn_edit = QPlainTextEdit()
        self.pgn_edit.setPlaceholderText("Paste one or more games in PGN here…")
        self.side_combo = QComboBox()
        self.side_combo.addItem("I played: auto (by username, else White)", "auto")
        self.side_combo.addItem("I played White", "white")
        self.side_combo.addItem("I played Black", "black")
        load = QPushButton("Load PGN")
        load.clicked.connect(self.load_pasted)
        pl = QVBoxLayout(paste)
        pl.addWidget(self.pgn_edit, 1)
        pl.addWidget(self.side_combo)
        pl.addWidget(load)
        self.tabs.addTab(paste, "Paste PGN")
        self.user_edits: dict[str, QLineEdit] = {}
        for site, name in (("lichess", s.lichess_username), ("chess.com", s.chesscom_username)):
            tab = QWidget()
            edit = QLineEdit(name)
            edit.setPlaceholderText(f"Your {site} username")
            count = QSpinBox()
            count.setRange(1, 50)
            count.setValue(max(1, min(50, s.review_fetch_count)))
            button = QPushButton("Load recent games")
            button.clicked.connect(
                lambda _=False, st=site, e=edit, c=count: self.fetch(st, e.text(), c.value())
            )
            edit.returnPressed.connect(button.click)
            row = QHBoxLayout()
            row.addWidget(QLabel("Games:"))
            row.addWidget(count)
            row.addStretch(1)
            tl = QVBoxLayout(tab)
            tl.addWidget(edit)
            tl.addLayout(row)
            tl.addWidget(button)
            tl.addStretch(1)
            self.tabs.addTab(tab, "Lichess" if site == "lichess" else "Chess.com")
            self.user_edits[site] = edit

        self.game_list = QListWidget()
        self.game_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.review_button = QPushButton("Review selected games")
        self.review_button.clicked.connect(self.review_selected)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.hide()
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        self.status = QLabel("Load games to review.")
        self.status.setWordWrap(True)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.summary.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.addWidget(self.tabs, 2)
        ll.addWidget(QLabel("Games (select one or more):"))
        ll.addWidget(self.game_list, 2)
        brow = QHBoxLayout()
        brow.addWidget(self.review_button, 1)
        brow.addWidget(self.stop_button)
        ll.addLayout(brow)
        ll.addWidget(self.progress_bar)
        ll.addWidget(self.status)
        ll.addWidget(self.summary, 3)

        self.reviewed_combo = QComboBox()
        self.reviewed_combo.currentIndexChanged.connect(self._show_game)
        self.board_view = BoardView()
        nav = QHBoxLayout()
        self.nav_buttons = []
        for text, slot, tip in (
            ("⏮", self.first, "Start"),
            ("◀", self.prev, "Previous move (Left)"),
            ("▶", self.next, "Next move (Right)"),
            ("⏭", self.last, "End"),
            ("Next mistake ▸", self.next_mistake, "Jump to your next flagged move (N)"),
        ):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            nav.addWidget(b)
            self.nav_buttons.append(b)
        center = QWidget()
        cl = QVBoxLayout(center)
        cl.addWidget(self.reviewed_combo)
        cl.addWidget(self.board_view, 1)
        cl.addLayout(nav)

        self.move_list = QListWidget()
        self.move_list.currentRowChanged.connect(self._on_move_row)
        self.detail = QLabel()
        self.detail.setWordWrap(True)
        self.detail.setTextFormat(Qt.TextFormat.RichText)
        self.detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.detail.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.detail.setMinimumHeight(140)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.addWidget(self.move_list, 1)
        rl.addWidget(self.detail)

        split = QSplitter()
        split.addWidget(left)
        split.addWidget(center)
        split.addWidget(right)
        right.setMinimumWidth(280)
        left.setMinimumWidth(280)
        split.setStretchFactor(1, 1)
        split.setSizes([330, 520, 330])
        root = QVBoxLayout(self)
        root.addWidget(split)
        for key, slot in (("Left", self.prev), ("Right", self.next), ("N", self.next_mistake)):
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(slot)

    # -- loading -----------------------------------------------------------------------------

    def load_pasted(self) -> None:
        games = parse_pgn(self.pgn_edit.toPlainText())
        if not games:
            self.status.setText("No standard chess game found in the pasted text.")
            return
        choice = str(self.side_combo.currentData())
        names = [self.settings.lichess_username, self.settings.chesscom_username]
        for g in games:
            if choice == "white":
                g.user_color = chess.WHITE
            elif choice == "black":
                g.user_color = chess.BLACK
            else:
                g.user_color = g.color_of(names)
        self._set_loaded(games, f"Loaded {len(games)} game(s) from the pasted PGN.")

    def fetch(self, site: str, username: str, count: int) -> None:
        username = username.strip()
        if site == "lichess":
            self.settings.lichess_username = username
        else:
            self.settings.chesscom_username = username
        self.settings.review_fetch_count = count
        save_settings(self.settings)
        self.status.setText(f"Loading {username}'s recent games from {site}…")
        fetcher = self.fetchers[site]

        def work() -> None:
            try:
                self.fetched.emit(fetcher(username, count), "")
            except FetchError as err:
                self.fetched.emit([], str(err))
            except Exception as err:  # noqa: BLE001 - shown in the window
                log.exception("fetch failed")
                self.fetched.emit([], f"Loading failed: {err}")

        threading.Thread(target=work, name="fetch", daemon=True).start()

    def _on_fetched(self, games: list[GameRecord], error: str) -> None:
        if error:
            self.status.setText(error)
            return
        if not games:
            self.status.setText("No standard games found for that user.")
            return
        self._set_loaded(games, f"Loaded {len(games)} recent game(s). Select and review.")

    def _set_loaded(self, games: list[GameRecord], message: str) -> None:
        self.loaded = games
        self.game_list.clear()
        for g in games:
            mark = "" if g.user_color is not None else "  (your side: White, assumed)"
            self.game_list.addItem(QListWidgetItem(g.title() + mark))
        first = self.game_list.item(0)
        if first is not None:
            first.setSelected(True)
        self.status.setText(message)

    # -- running -----------------------------------------------------------------------------

    def review_selected(self) -> None:
        rows = sorted(i.row() for i in self.game_list.selectedIndexes())
        if not rows:
            self.status.setText("Select at least one game.")
            return
        if self.engine_path is None and self._make_evaluator is None:
            self.status.setText("Stockfish was not found, so games cannot be reviewed.")
            return
        self.stop()
        games = [self.loaded[r] for r in rows]
        pairs = [(g, g.user_color if g.user_color is not None else chess.WHITE) for g in games]
        explainer, note = self.make_explainer()
        self.explain_note = note
        self.reviewed_games = games
        self.reviews.clear()
        self.explanations.clear()
        self.reviewed_combo.blockSignals(True)
        self.reviewed_combo.clear()
        self.reviewed_combo.blockSignals(False)
        self.move_list.clear()
        self.summary.clear()
        self.detail.clear()
        self.progress_bar.setRange(0, 100 * len(games))
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.stop_button.show()
        self.review_button.setEnabled(False)
        self.status.setText(f"Reviewing {len(games)} game(s) at {self.level.name} level…")
        self.runner = ReviewRunner(
            pairs,
            self.level,
            self._make_evaluator or self._stockfish,
            ReviewCallbacks(
                self.progress.emit, self.game_done.emit, self.explained.emit, self.finished.emit
            ),
            explainer,
            self.settings.review_explain_max,
            self.explanations_on,
        )
        self.runner.start()

    def _stockfish(self) -> Evaluator:
        assert self.engine_path is not None
        s = self.settings
        return StockfishEvaluator(
            self.engine_path, s.review_depth, s.review_seconds, s.engine_threads
        )

    def stop(self) -> None:
        if self.runner is not None and self.runner.running:
            self.runner.stop()
            self.runner.join(5)

    def _on_progress(self, game: int, done: int, total: int) -> None:
        self.progress_bar.setValue(100 * game + int(100 * done / max(1, total)))

    def _on_game_done(self, index: int, review: GameReview) -> None:
        self.reviews[index] = review
        g = review.game
        flagged = len(review.flagged())
        self.reviewed_combo.addItem(f"{g.white} vs {g.black} ({flagged} flagged)", index)
        if self.reviewed_combo.count() == 1:
            self._show_game(0)
        self._update_summary()

    def _on_explained(
        self, game: int, ply: int, result: Explanation | None, error: str | None, facts: str
    ) -> None:
        if result is None:
            self.explanations[(game, ply)] = (error or "Explanation failed.", "")
        else:
            tip = f"Checked against the facts ({result.outcome}):\n{facts}"
            self.explanations[(game, ply)] = (result.text, tip)
        if game == self._game_index():
            self._refresh_moves()
            self._show_ply(self.ply)

    def _on_finished(self, error: str | None) -> None:
        self.progress_bar.hide()
        self.stop_button.hide()
        self.review_button.setEnabled(True)
        n = len(self.reviews)
        text = f"Reviewed {n} game(s)."
        if self.explain_note:
            text += " " + self.explain_note
        if error:
            text += " " + error
        self.status.setText(text)

    def _update_summary(self) -> None:
        reviews = [self.reviews[i] for i in sorted(self.reviews)]
        self.summary.setText(summary_text(summarize(reviews), self.level))

    # -- navigation --------------------------------------------------------------------------

    def _game_index(self) -> int:
        data = self.reviewed_combo.currentData()
        return int(data) if data is not None else -1

    def _review(self) -> GameReview | None:
        return self.reviews.get(self._game_index())

    def _show_game(self, _row: int) -> None:
        r = self._review()
        if r is None:
            return
        self.board_view.set_orientation(r.user_color == chess.WHITE)
        self._refresh_moves()
        flagged = r.flagged()
        self._show_ply(flagged[0].ply + 1 if flagged else 0)

    def _refresh_moves(self) -> None:
        r = self._review()
        if r is None:
            return
        row = self.move_list.currentRow()
        self.move_list.blockSignals(True)
        self.move_list.clear()
        self.move_list.addItem("Start position")
        for m in r.moves:
            text = f"{m.move_number} {m.san}"
            if m.severity is not None:
                text += f"{SEVERITY_MARK[m.severity]}   {m.severity}"
                if (self._game_index(), m.ply) in self.explanations:
                    text += " · explained"
            item = QListWidgetItem(text)
            if m.severity is not None:
                item.setForeground(QColor(SEVERITY_COLOR[m.severity]))
            elif not m.by_user:
                item.setForeground(QColor("gray"))
            self.move_list.addItem(item)
        self.move_list.setCurrentRow(row)
        self.move_list.blockSignals(False)

    def _on_move_row(self, row: int) -> None:
        if row >= 0:
            self._show_ply(row, from_list=True)

    def _show_ply(self, ply: int, from_list: bool = False) -> None:
        r = self._review()
        if r is None:
            return
        ply = max(0, min(ply, len(r.moves)))
        self.ply = ply
        if not from_list:
            self.move_list.blockSignals(True)
            self.move_list.setCurrentRow(ply)
            self.move_list.blockSignals(False)
        board = r.game.board()
        for m in r.game.moves[:ply]:
            board.push(m)
        last = r.moves[ply - 1] if ply > 0 else None
        arrows: list[tuple[chess.Move, QColor]] = []
        if last is not None and last.severity is not None and last.best is not None:
            if last.best.pv and last.best.move != last.move:
                arrows.append((last.best.move, BEST_COLOR))
            arrows.append((last.move, PLAYED_COLOR))
        self.board_view.set_position(board, last.move if last else None, arrows)
        self.detail.setText(self._detail_html(r, last))

    def _detail_html(self, r: GameReview, m: MoveReview | None) -> str:
        ev = r.evals[self.ply] if self.ply < len(r.evals) else 0
        evtext = f"Evaluation: {ev / 100:+.1f}" if abs(ev) < 1000 else "Evaluation: decisive"
        if m is None:
            return html.escape(evtext)
        parts = [f"<b>{html.escape(m.move_number + ' ' + m.san)}</b>"]
        if not m.by_user:
            parts[0] += " <span style='color: gray'>(opponent)</span>"
        if m.severity is not None:
            color = SEVERITY_COLOR[m.severity]
            parts.append(
                f"<span style='color: {color}; font-weight: bold'>{m.severity.capitalize()}</span>"
                f" - about {m.loss / 100:.1f} pawns worse than <b>{html.escape(m.best_san)}</b>"
                + (f" ({eval_number(m.best)})" if m.best is not None else "")
                + "."
            )
            if m.mistake is not None:
                parts.append(
                    f"<i>{html.escape(m.mistake.label)}</i>: {html.escape(m.mistake.detail)}"
                )
            exp = self.explanations.get((self._game_index(), m.ply))
            if exp is not None:
                parts.append(html.escape(exp[0]))
                self.detail.setToolTip(exp[1])
            else:
                self.detail.setToolTip("")
        elif m.by_user and m.best_san and not m.is_best and m.loss >= 30:
            parts.append(
                f"Stockfish preferred {html.escape(m.best_san)} (loss {m.loss / 100:.1f})."
            )
        elif m.by_user and m.is_best:
            parts.append("Stockfish's top move.")
        parts.append(f"<span style='color: gray'>{html.escape(evtext)}</span>")
        return "<br><br>".join(parts)

    def first(self) -> None:
        self._show_ply(0)

    def prev(self) -> None:
        self._show_ply(self.ply - 1)

    def next(self) -> None:
        self._show_ply(self.ply + 1)

    def last(self) -> None:
        r = self._review()
        self._show_ply(len(r.moves) if r else 0)

    def next_mistake(self) -> None:
        r = self._review()
        if r is None:
            return
        flagged = r.flagged()
        if not flagged:
            return
        after = [m for m in flagged if m.ply + 1 > self.ply]
        target = (after or flagged)[0]
        self._show_ply(target.ply + 1)

    def set_level(self, level: Level) -> None:
        self.level = level

    def closeEvent(self, event: QCloseEvent | None) -> None:  # noqa: N802 - Qt API
        self.stop()
        self.closed.emit()
        super().closeEvent(event)
