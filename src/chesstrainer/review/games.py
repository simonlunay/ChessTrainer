"""Games to review: parsed from pasted PGN or fetched from the Lichess / Chess.com public APIs.

Lichess: ``GET https://lichess.org/api/games/user/{name}?max=N`` returns PGN (newest first).
Chess.com: ``GET https://api.chess.com/pub/player/{name}/games/archives`` lists monthly archive
URLs; each archive is JSON with a ``pgn`` per game. Both are public (no login, no API key).
Only standard chess is kept (variants are skipped).
"""

from __future__ import annotations

import io
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import chess
import chess.pgn

USER_AGENT = "ChessTrainer/0.1 (desktop chess coach; game review)"
USERNAME_RE = re.compile(r"^[A-Za-z0-9_-]{2,30}$")

#: ``opener(url, accept) -> body text``; replaced in tests.
Opener = Callable[[str, str], str]


class FetchError(Exception):
    """A failed download, with a message fit for the window."""


@dataclass
class GameRecord:
    headers: dict[str, str]
    start_fen: str
    moves: list[chess.Move]
    source: str = "pgn"  # "pgn" | "lichess" | "chess.com"
    #: Side the player had, when known (username match or chosen by the user).
    user_color: chess.Color | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def white(self) -> str:
        return self.headers.get("White", "?")

    @property
    def black(self) -> str:
        return self.headers.get("Black", "?")

    @property
    def result(self) -> str:
        return self.headers.get("Result", "*")

    @property
    def date(self) -> str:
        return self.headers.get("UTCDate") or self.headers.get("Date", "????.??.??")

    def board(self) -> chess.Board:
        return chess.Board(self.start_fen)

    def title(self) -> str:
        date = self.date.replace("????.??.??", "").strip()
        speed = self.headers.get("TimeControl", "")
        bits = [f"{self.white} vs {self.black}", self.result]
        if date:
            bits.insert(0, date)
        if speed and speed != "-":
            bits.append(speed)
        return " · ".join(bits) + f" · {(len(self.moves) + 1) // 2} moves"

    def color_of(self, usernames: Iterable[str]) -> chess.Color | None:
        names = {u.strip().lower() for u in usernames if u.strip()}
        white, black = self.white.lower() in names, self.black.lower() in names
        if white != black:
            return chess.WHITE if white else chess.BLACK
        return None


def parse_pgn(text: str, source: str = "pgn") -> list[GameRecord]:
    """Every standard-chess game in ``text`` (empty games and variants are skipped)."""
    games = []
    stream = io.StringIO(text)
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        variant = game.headers.get("Variant", "Standard").lower()
        if variant not in ("standard", "from position", "chess"):
            continue
        moves = list(game.mainline_moves())
        if not moves:
            continue
        errors = [str(e) for e in game.errors]
        games.append(
            GameRecord(dict(game.headers), game.board().fen(), moves, source, None, errors)
        )
    return games


def check_username(name: str) -> str:
    name = name.strip()
    if not USERNAME_RE.match(name):
        raise FetchError(f"{name!r} is not a valid username (letters, digits, _ and - only).")
    return name


def http_get(url: str, accept: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body: bytes = resp.read()
            return body.decode("utf-8", errors="replace")
    except urllib.error.HTTPError as err:
        if err.code == 404:
            raise FetchError("not_found") from err
        if err.code == 429:
            raise FetchError("Too many requests; wait a minute and try again.") from err
        raise FetchError(f"The server answered {err.code} {err.reason}.") from err
    except (urllib.error.URLError, TimeoutError, OSError) as err:
        raise FetchError("Could not connect (no internet?).") from err


def fetch_lichess(
    username: str, max_games: int = 10, opener: Opener = http_get
) -> list[GameRecord]:
    name = check_username(username)
    query = urllib.parse.urlencode(
        {"max": max(1, min(max_games, 50)), "moves": "true", "tags": "true", "clocks": "false"}
    )
    url = f"https://lichess.org/api/games/user/{urllib.parse.quote(name)}?{query}"
    try:
        text = opener(url, "application/x-chess-pgn")
    except FetchError as err:
        if str(err) == "not_found":
            raise FetchError(f"Lichess has no user called {name!r}.") from err
        raise
    games = parse_pgn(text, "lichess")
    for g in games:
        g.user_color = g.color_of([name])
    return games[:max_games]


def fetch_chesscom(
    username: str, max_games: int = 10, opener: Opener = http_get
) -> list[GameRecord]:
    """Newest standard games first, walking back through the monthly archives (at most 6)."""
    name = check_username(username).lower()
    base = f"https://api.chess.com/pub/player/{urllib.parse.quote(name)}/games/archives"
    try:
        archives = json.loads(opener(base, "application/json")).get("archives", [])
    except FetchError as err:
        if str(err) == "not_found":
            raise FetchError(f"Chess.com has no user called {name!r}.") from err
        raise
    except json.JSONDecodeError as err:
        raise FetchError("Chess.com sent an unexpected answer.") from err
    games: list[GameRecord] = []
    for url in list(reversed(archives))[:6]:
        if not str(url).startswith("https://api.chess.com/"):
            continue
        try:
            month: dict[str, Any] = json.loads(opener(str(url), "application/json"))
        except json.JSONDecodeError as err:
            raise FetchError("Chess.com sent an unexpected answer.") from err
        for raw in reversed(month.get("games", [])):
            if raw.get("rules", "chess") != "chess" or not raw.get("pgn"):
                continue
            for g in parse_pgn(str(raw["pgn"]), "chess.com"):
                if raw.get("url"):
                    g.headers.setdefault("Link", str(raw["url"]))
                g.user_color = g.color_of([name])
                games.append(g)
            if len(games) >= max_games:
                return games
    return games
