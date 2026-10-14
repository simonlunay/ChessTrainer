"""Runs a review off the GUI thread: engine pass per game, then explanations of the biggest
mistakes (only when an explainer is given, i.e. explanations are on and a key is set)."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass

import chess

from chesstrainer.explain.explainer import Explainer, ExplainError, Explanation
from chesstrainer.levels import Level
from chesstrainer.review.analysis import Evaluator, GameReview, review_game
from chesstrainer.review.games import GameRecord
from chesstrainer.review.report import review_facts

log = logging.getLogger(__name__)


@dataclass
class ReviewCallbacks:
    progress: Callable[[int, int, int], None]  # game index, positions done, positions total
    game_done: Callable[[int, GameReview], None]
    explained: Callable[[int, int, Explanation | None, str | None, str], None]
    # ^ game index, ply, result, error, facts text
    finished: Callable[[str | None], None]  # error message or None


class ReviewRunner:
    def __init__(
        self,
        games: list[tuple[GameRecord, chess.Color]],
        level: Level,
        make_evaluator: Callable[[], Evaluator],
        callbacks: ReviewCallbacks,
        explainer: Explainer | None = None,
        explain_max: int = 3,
        explanations_on: Callable[[], bool] = lambda: True,
    ) -> None:
        self.games = games
        self.level = level
        self.make_evaluator = make_evaluator
        self.cb = callbacks
        self.explainer = explainer
        self.explain_max = explain_max
        self.explanations_on = explanations_on  # checked before every API call
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="review", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    def _run(self) -> None:
        try:
            evaluator = self.make_evaluator()
        except Exception as err:  # noqa: BLE001 - reported to the window
            self.cb.finished(f"Could not start Stockfish: {err}")
            return
        reviews: list[GameReview] = []
        try:
            for i, (game, color) in enumerate(self.games):
                review = review_game(
                    game,
                    color,
                    self.level,
                    evaluator,
                    lambda done, total, i=i: self.cb.progress(i, done, total),  # type: ignore[misc]
                    self._stop.is_set,
                )
                reviews.append(review)
                self.cb.game_done(i, review)
        except InterruptedError:
            self.cb.finished("Review stopped.")
            return
        except Exception as err:  # noqa: BLE001
            log.exception("review failed")
            self.cb.finished(f"Review failed: {err}")
            return
        finally:
            evaluator.close()
        error = self._explain(reviews) if self.explainer is not None else None
        self.cb.finished(error)

    def _explain(self, reviews: list[GameReview]) -> str | None:
        """Biggest mistakes of each game, biggest first. Stops at the first API error."""
        assert self.explainer is not None
        for i, review in enumerate(reviews):
            for m in review.biggest(self.explain_max):
                if self._stop.is_set():
                    return "Review stopped."
                if not self.explanations_on():
                    return "Explanations were turned off; the rest are not explained."
                facts = review_facts(m, self.level)
                try:
                    result = self.explainer.explain(facts, self.level)
                except ExplainError as err:
                    self.cb.explained(i, m.ply, None, str(err), facts.text())
                    return str(err)
                self.cb.explained(i, m.ply, result, None, facts.text())
        return None
