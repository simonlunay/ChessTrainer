"""Grounded move explanations from Claude, cached, on a background thread.

:class:`Explainer` turns :class:`MoveFacts` into text with one Messages API call (only the facts,
the move and the level are sent) and caches the result in SQLite by position + move + level +
model, so a position seen again (or a replay) costs nothing. Without an API key it stays
``available = False`` with a clear ``unavailable_reason`` and the app just skips explanations.

Every new explanation goes through the rule-based grounding check (``grounding.py``). If it
flags a claim that is not in the facts, the explanation is requested once more with the flagged
claims listed; if that one fails too, the facts themselves are shown in plain language
(``fallback.py``). Each outcome (passed / regenerated / fallback) is logged to the
``grounding_log`` table next to the cache; :meth:`ExplanationCache.grounding_stats` counts them.

:class:`ExplainWorker` runs explanations off the GUI thread; like the engine, a new request
replaces a queued one, and results carry the request id so stale ones can be dropped.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic

from chesstrainer.explain.facts import MoveFacts
from chesstrainer.explain.fallback import facts_summary
from chesstrainer.explain.grounding import Issue, check_grounding
from chesstrainer.explain.prompt import PROMPT_VERSION, SYSTEM, retry_message, user_message
from chesstrainer.levels import Level
from chesstrainer.settings.config import settings_path
from chesstrainer.settings.env import anthropic_api_key, env_files

log = logging.getLogger(__name__)

#: Response cap. Explanations are 1-3 sentences; the rest is headroom for (low-effort) thinking.
MAX_TOKENS = 2000

#: Grounding outcomes: first answer passed the check / second answer passed / facts shown.
OUTCOMES = ("passed", "regenerated", "fallback")


class ExplainError(Exception):
    """A failed explanation, with a message fit for the panel."""


@dataclass(frozen=True)
class Explanation:
    text: str
    model: str
    cached: bool
    seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    outcome: str = "passed"  # one of OUTCOMES
    issues: tuple[str, ...] = ()  # what the check flagged (both attempts)
    calls: int = 1  # API calls made (0 when cached)


def position_key(fen: str) -> str:
    """FEN without the move counters (they do not change the explanation)."""
    return " ".join(fen.split()[:4])


class ExplanationCache:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or settings_path().parent / "explanations.sqlite"
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS explanations ("
                " key TEXT PRIMARY KEY, fen TEXT, move TEXT, level TEXT, model TEXT,"
                " text TEXT, facts TEXT, created REAL)"
            )
            columns = {row[1] for row in db.execute("PRAGMA table_info(explanations)")}
            if "outcome" not in columns:  # caches written before the grounding check
                db.execute("ALTER TABLE explanations ADD COLUMN outcome TEXT DEFAULT 'passed'")
            db.execute(
                "CREATE TABLE IF NOT EXISTS grounding_log ("
                " created REAL, fen TEXT, move TEXT, level TEXT, model TEXT, context TEXT,"
                " outcome TEXT, issues TEXT)"
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=5)
        try:
            with db:  # commits (or rolls back) the transaction
                yield db
        finally:
            db.close()

    @staticmethod
    def key(fen: str, move_uci: str, level: str, model: str, context: str = "") -> str:
        raw = f"v{PROMPT_VERSION}|{position_key(fen)}|{move_uci}|{level}|{model}"
        if context:
            raw += f"|{context}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(
        self, fen: str, move_uci: str, level: str, model: str, context: str = ""
    ) -> tuple[str, str] | None:
        """(text, outcome) or None."""
        k = self.key(fen, move_uci, level, model, context)
        with self._lock, self._connect() as db:
            row = db.execute(
                "SELECT text, outcome FROM explanations WHERE key = ?", (k,)
            ).fetchone()
        return None if row is None else (str(row[0]), str(row[1] or "passed"))

    def put(
        self, facts: MoveFacts, level: str, model: str, text: str, outcome: str = "passed"
    ) -> None:
        k = self.key(facts.fen, facts.move.uci(), level, model, facts.context)
        row = (k, facts.fen, facts.move.uci(), level, model, text, facts.text(), time.time())
        with self._lock, self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO explanations"
                " (key, fen, move, level, model, text, facts, created, outcome)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (*row, outcome),
            )

    def log_grounding(
        self, facts: MoveFacts, level: str, model: str, outcome: str, issues: list[str]
    ) -> None:
        row = (time.time(), facts.fen, facts.move.uci(), level, model, facts.context)
        with self._lock, self._connect() as db:
            db.execute(
                "INSERT INTO grounding_log VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (*row, outcome, json.dumps(issues)),
            )

    def grounding_stats(self, since: float = 0.0) -> dict[str, int]:
        """How often each outcome happened (new explanations only, not cache hits)."""
        counts = dict.fromkeys(OUTCOMES, 0)
        with self._lock, self._connect() as db:
            rows = db.execute(
                "SELECT outcome, COUNT(*) FROM grounding_log WHERE created >= ? GROUP BY outcome",
                (since,),
            ).fetchall()
        for outcome, n in rows:
            counts[str(outcome)] = int(n)
        return counts


def missing_key_message() -> str:
    places = " or ".join(str(p) for p in env_files())
    return (
        "Explanations are off: no ANTHROPIC_API_KEY was found. Add the line "
        f"ANTHROPIC_API_KEY=sk-ant-... to {places} and restart ChessTrainer. "
        "Arrows and engine lines work without it."
    )


class Explainer:
    def __init__(
        self,
        model: str,
        effort: str = "medium",
        *,
        api_key: str | None = None,
        cache: ExplanationCache | None = None,
        client: Any = None,
    ) -> None:
        self.model = model
        self.effort = effort
        self.cache = cache
        self.unavailable_reason = ""
        self._client = client
        if self._client is None:
            key = api_key if api_key is not None else anthropic_api_key()
            if not key:
                self.unavailable_reason = missing_key_message()
            else:
                self._client = anthropic.Anthropic(api_key=key, timeout=20.0, max_retries=2)

    @property
    def available(self) -> bool:
        return self._client is not None

    def cached(self, facts: MoveFacts, level: Level) -> Explanation | None:
        if self.cache is None:
            return None
        hit = self.cache.get(facts.fen, facts.move.uci(), level.key, self.model, facts.context)
        if hit is None:
            return None
        return Explanation(hit[0], self.model, True, 0.0, outcome=hit[1], calls=0)

    def explain(self, facts: MoveFacts, level: Level) -> Explanation:
        """Cached text, or 1-2 API calls checked for grounding (see the module docstring).
        Raises :class:`ExplainError` with a readable message."""
        hit = self.cached(facts, level)
        if hit is not None:
            return hit
        if self._client is None:
            raise ExplainError(self.unavailable_reason)
        t0 = time.perf_counter()
        prompt = user_message(facts, level)
        text, tokens_in, tokens_out = self._ask(prompt)
        flagged = [_issue_text(i) for i in check_grounding(text, facts)]
        outcome, calls = "passed", 1
        if flagged:
            log.info("explanation failed the grounding check (%s); asking again", flagged)
            text2, in2, out2 = self._ask(f"{prompt}\n{retry_message(text, flagged)}")
            tokens_in, tokens_out, calls = tokens_in + in2, tokens_out + out2, 2
            flagged2 = [_issue_text(i) for i in check_grounding(text2, facts)]
            flagged += flagged2
            if flagged2:
                outcome, text = "fallback", facts_summary(facts, level)
                log.info("second explanation failed too (%s); showing the facts", flagged2)
            else:
                outcome, text = "regenerated", text2
        if self.cache is not None:
            self.cache.put(facts, level.key, self.model, text, outcome)
            self.cache.log_grounding(facts, level.key, self.model, outcome, flagged)
        return Explanation(
            text,
            self.model,
            False,
            time.perf_counter() - t0,
            tokens_in,
            tokens_out,
            outcome,
            tuple(flagged),
            calls,
        )

    def _ask(self, content: str) -> tuple[str, int, int]:
        response = self._request(content)
        text = " ".join(
            b.text.strip() for b in response.content if b.type == "text" and b.text.strip()
        )
        if response.stop_reason == "refusal":
            raise ExplainError("Claude declined to explain this position.")
        if not text:
            raise ExplainError(f"Claude returned no text (stop reason: {response.stop_reason}).")
        usage = response.usage
        return text, int(usage.input_tokens), int(usage.output_tokens)

    def _request(self, content: str) -> Any:
        assert self._client is not None
        try:
            return self._client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=SYSTEM,
                messages=[{"role": "user", "content": content}],
                output_config={"effort": self.effort},
            )
        except anthropic.AuthenticationError as err:
            raise ExplainError(
                "The Anthropic API key was rejected; check ANTHROPIC_API_KEY in .env."
            ) from err
        except anthropic.PermissionDeniedError as err:
            raise ExplainError("This API key may not use the explanation model.") from err
        except anthropic.NotFoundError as err:
            raise ExplainError(
                f"Model {self.model!r} was not found; check explain_model in settings.json."
            ) from err
        except anthropic.BadRequestError as err:
            detail = _error_detail(err)
            if "credit balance" in detail.lower():
                raise ExplainError(
                    "Explanations paused: the Anthropic account is out of credits. Add credits "
                    "at console.anthropic.com (Plans & Billing); arrows keep working."
                ) from err
            raise ExplainError(f"The explanation request was rejected: {detail}") from err
        except anthropic.RateLimitError as err:
            raise ExplainError("Rate limited by the Anthropic API; try again shortly.") from err
        except anthropic.APIStatusError as err:
            raise ExplainError(f"Anthropic API error {err.status_code}; try again later.") from err
        except anthropic.APIConnectionError as err:
            raise ExplainError("Could not reach the Anthropic API (no internet?).") from err


def _issue_text(issue: Issue) -> str:
    return f"{issue.claim!r}: {issue.detail}"


def _error_detail(err: anthropic.APIStatusError) -> str:
    """The API's own message ("Your credit balance is too low...") without the JSON wrapper."""
    body = err.body
    if isinstance(body, dict):
        inner = body.get("error")
        if isinstance(inner, dict) and isinstance(inner.get("message"), str):
            return str(inner["message"])
    return err.message


@dataclass(frozen=True)
class ExplainJob:
    request_id: int
    facts: MoveFacts
    level: Level


ResultCallback = Callable[[int, Explanation | None, str | None], None]


class ExplainWorker:
    """One background thread; ``submit`` replaces any job that has not started yet."""

    def __init__(self, explainer: Explainer, on_result: ResultCallback) -> None:
        self.explainer = explainer
        self.on_result = on_result
        self._cond = threading.Condition()
        self._job: ExplainJob | None = None
        self._closing = False
        self._next_id = 0
        self._thread = threading.Thread(target=self._run, name="explain", daemon=True)
        self._thread.start()

    def submit(self, facts: MoveFacts, level: Level) -> int:
        with self._cond:
            self._next_id += 1
            self._job = ExplainJob(self._next_id, facts, level)
            self._cond.notify()
            return self._next_id

    def close(self) -> None:
        with self._cond:
            self._closing = True
            self._job = None
            self._cond.notify()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._job is None and not self._closing:
                    self._cond.wait()
                if self._closing:
                    return
                job, self._job = self._job, None
            assert job is not None
            try:
                result = self.explainer.explain(job.facts, job.level)
            except ExplainError as err:
                self.on_result(job.request_id, None, str(err))
            except Exception as err:  # noqa: BLE001 - never kill the worker thread
                log.exception("explanation failed")
                self.on_result(job.request_id, None, f"Explanation failed: {err}")
            else:
                self.on_result(job.request_id, result, None)
