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
AppMode = Literal["analysis", "trainer"]
Side = Literal["white", "black"]
TrainerMode = Literal["guided", "practice"]


@dataclass
class Settings:
    #: mss monitor index: 1 = primary, 2.. = others, 0 = all monitors combined.
    monitor: int = 1
    #: 10 fps keeps move-to-arrow latency well under 500 ms (2 stable frames + engine).
    capture_fps: float = 10.0
    #: Exclude the overlay window from screen capture (Windows 10 2004+). Turn off to record GIFs;
    #: ChessTrainer then sees its own arrows, which the classifier tolerates but is not ideal.
    hide_overlay_from_capture: bool = True
    #: "auto" detects orientation from piece placement; "white"/"black" = side at the bottom.
    orientation: Orientation = "auto"
    #: Manually selected board region ``[x, y, w, h]`` in monitor pixels; None = auto-detect.
    locked_region: list[int] | None = None
    #: Torch device for the classifier: "auto", "cpu" or "cuda".
    device: str = "auto"
    #: Classify every frame on CUDA so the GPU stays clocked up: ~80 ms faster reaction to a
    #: move on a laptop, at the cost of ~6 ms of GPU work per frame. Off = classify on change only.
    gpu_keep_warm: bool = True
    #: Classifier backend: "auto" (PyTorch when installed, else the ONNX model), "torch", "onnx".
    #: The packaged app has no PyTorch and always uses ONNX (OpenCV DNN, CPU).
    classifier_backend: str = "auto"
    #: Stockfish executable; None = search STOCKFISH_PATH, PATH, data/engine and the app folder.
    engine_path: str | None = None
    engine_threads: int = 4
    engine_hash_mb: int = 128
    #: Candidate moves analysed and drawn (phase 4 makes this depend on the skill level).
    multipv: int = 3
    #: Analysis stops after this long per position (the arrows stay up).
    engine_max_seconds: float = 8.0
    #: Side panel top-left in Qt logical pixels; None = next to the board.
    panel_pos: list[int] | None = None
    #: Skill level key from ``levels.yaml`` (beginner, novice, intermediate, advanced, expert).
    skill_level: str = "beginner"
    #: False until the user confirmed a level in the first-launch prompt.
    level_chosen: bool = False
    #: Plain-English explanations of the suggested move (needs ANTHROPIC_API_KEY). Off = no API
    #: calls at all (arrows, engine lines and eval bar only). Panel checkbox / Ctrl+E; saved.
    explanations: bool = True
    #: True once the API key field was offered (first launch); the key itself is never stored
    #: here but in the OS credential store (panel "API key…" button changes it).
    api_key_prompted: bool = False
    #: Claude model for explanations: they run often, so a fast, cheap one by default.
    explain_model: str = "claude-haiku-5-5"
    #: Claude effort ("low" | "medium" | "high"). Phase-4 eval: medium flagged 7-9% of
    #: explanations as containing an unsupported claim vs 22-25% at low, at ~3.0 s vs ~1.2 s.
    explain_effort: str = "medium"
    #: The position must stay unchanged this long before it is explained (no API call per move
    #: while a game or a replay is moving fast).
    explain_delay_s: float = 1.5
    #: ...and the engine must have reached this depth (or finished), so the move is settled.
    explain_min_depth: int = 16
    #: "analysis" (best-move arrows) or "trainer" (opening trainer).
    app_mode: AppMode = "analysis"
    #: Opening trainer choice: a Lichess opening name (family or full variation), the side the
    #: player trains, and guided (arrows) or practice (spaced repetition, no arrows).
    trainer_opening: str = ""
    trainer_side: Side = "white"
    trainer_mode: TrainerMode = "guided"
    #: System-wide hotkeys, action -> combination (e.g. "Ctrl+Alt+O"; "" disables one). Missing
    #: actions use the defaults in ``overlay/hotkeys.py``. ``global_hotkeys`` False = none.
    global_hotkeys: bool = True
    hotkeys: dict[str, str] = field(default_factory=dict)
    #: Game review: usernames (remembered), how many recent games to list, engine depth and time
    #: cap per position, and how many of the biggest mistakes per game are explained.
    lichess_username: str = ""
    chesscom_username: str = ""
    review_fetch_count: int = 10
    review_depth: int = 14
    review_seconds: float = 0.5
    review_explain_max: int = 3
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
    if settings.classifier_backend not in ("auto", "torch", "onnx"):
        settings.classifier_backend = "auto"
    if settings.app_mode not in ("analysis", "trainer"):
        settings.app_mode = "analysis"
    if settings.trainer_side not in ("white", "black"):
        settings.trainer_side = "white"
    if settings.trainer_mode not in ("guided", "practice"):
        settings.trainer_mode = "guided"
    from chesstrainer.levels import LEVEL_KEYS  # local: levels imports nothing from here

    if settings.skill_level not in LEVEL_KEYS:
        settings.skill_level = "beginner"
    return settings


def save_settings(settings: Settings, path: Path | None = None) -> None:
    path = path or settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
    tmp.replace(path)
