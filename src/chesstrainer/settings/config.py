"""User settings, persisted as JSON in the per-user config directory.

Location: ``%APPDATA%\\ChessTrainer\\settings.json`` on Windows,
``~/.config/chesstrainer`` elsewhere,
or ``$CHESSTRAINER_HOME/settings.json`` when that variable is set (used by tests).
Unknown keys are ignored and missing keys take their defaults, so old files keep loading.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

log = logging.getLogger(__name__)

Orientation = Literal["auto", "white", "black"]


@dataclass
class Settings:
    #: mss monitor index: 1 = primary, 2.. = others, 0 = all monitors combined.
    monitor: int = 1
    capture_fps: float = 5.0
    #: Exclude the overlay window from screen capture (Windows 10 2004+). Turn off to record GIFs;
    #: ChessTrainer then sees its own arrows, which the classifier tolerates but is not ideal.
    hide_overlay_from_capture: bool = True
    #: "auto" detects orientation from piece placement; "white"/"black" = side at the bottom.
    orientation: Orientation = "auto"
    #: Manually selected board region ``[x, y, w, h]`` in monitor pixels; None = auto-detect.
    locked_region: list[int] | None = None
    #: Torch device for the classifier: "auto", "cpu" or "cuda".
    device: str = "auto"
    extra: dict[str, Any] = field(default_factory=dict)


def settings_path() -> Path:
    home = os.environ.get("CHESSTRAINER_HOME")
    if home:
        return Path(home) / "settings.json"
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / "ChessTrainer" / "settings.json"
    return Path.home() / ".config" / "chesstrainer" / "settings.json"


def load_settings(path: Path | None = None) -> Settings:
    path = path or settings_path()
    if not path.exists():
        return Settings()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        log.warning("could not read %s (%s); using defaults", path, err)
        return Settings()
    known = {f.name for f in fields(Settings)}
    settings = Settings(**{k: v for k, v in raw.items() if k in known})
    if settings.orientation not in ("auto", "white", "black"):
        settings.orientation = "auto"
    return settings


def save_settings(settings: Settings, path: Path | None = None) -> None:
    path = path or settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
    tmp.replace(path)
