"""Find the Stockfish executable, or explain how to install it.

Search order: ``Settings.engine_path``, ``STOCKFISH_PATH`` (environment or ``.env``), ``PATH``,
then ``stockfish*`` executables (up to two folders deep) in ``data/engine/`` of the project and
in ``engine/`` next to the user settings (``%APPDATA%\\ChessTrainer\\engine``).
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from chesstrainer.paths import bundle_dir
from chesstrainer.settings.config import settings_path
from chesstrainer.settings.env import load_env

DOWNLOAD_URL = "https://stockfishchess.org/download/"
_EXE = ".exe" if sys.platform == "win32" else ""


def engine_dirs() -> list[Path]:
    """Folders searched for a ``stockfish*`` executable."""
    return [bundle_dir() / "engine", settings_path().parent / "engine"]


@dataclass
class EngineSearch:
    path: Path | None
    tried: list[str] = field(default_factory=list)
    problem: str | None = None  # e.g. STOCKFISH_PATH points at a missing file

    @property
    def found(self) -> bool:
        return self.path is not None

    def setup_message(self) -> str:
        lines = ["Stockfish was not found, so no moves can be analysed."]
        if self.problem:
            lines.append(self.problem)
        lines += [
            "",
            "To set it up:",
            f"  1. Download Stockfish for Windows (x86-64) from {DOWNLOAD_URL}",
            "  2. Unzip it and put the folder in one of:",
            *(f"       {d}" for d in engine_dirs()),
            "     or set STOCKFISH_PATH=C:\\path\\to\\stockfish.exe in the .env file,",
            "     or put stockfish on your PATH.",
            "  3. Restart ChessTrainer.",
        ]
        return "\n".join(lines)


def _is_exe(p: Path) -> bool:
    return p.is_file() and (sys.platform == "win32" or os.access(p, os.X_OK))


def _search_dir(d: Path) -> Path | None:
    if not d.is_dir():
        return None
    for pattern in (f"stockfish*{_EXE}", f"*/stockfish*{_EXE}", f"*/*/stockfish*{_EXE}"):
        hits = sorted(p for p in d.glob(pattern) if _is_exe(p))
        if hits:
            return hits[0]
    return None


def find_stockfish(configured: str | None = None) -> EngineSearch:
    load_env()
    tried: list[str] = []
    for source, value in (("settings engine_path", configured), ("STOCKFISH_PATH", None)):
        if source == "STOCKFISH_PATH":
            value = os.environ.get("STOCKFISH_PATH")
        if not value:
            continue
        p = Path(value.strip().strip('"')).expanduser()
        if p.is_dir():
            p = _search_dir(p) or p
        tried.append(f"{source} = {value}")
        if _is_exe(p):
            return EngineSearch(p, tried)
        return EngineSearch(None, tried, f"{source} points to {value}, which is not a file.")
    tried.append("PATH")
    on_path = shutil.which("stockfish")
    if on_path:
        return EngineSearch(Path(on_path), tried)
    for d in engine_dirs():
        tried.append(str(d))
        hit = _search_dir(d)
        if hit:
            return EngineSearch(hit, tried)
    return EngineSearch(None, tried)
