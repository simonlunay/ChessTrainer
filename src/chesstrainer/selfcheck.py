"""Headless check of an installation (used on the packaged app by ``scripts/build_exe.py``).

    ChessTrainer.exe --self-check report.json

Opens no window, captures nothing and registers no hotkey: it loads the classifier and runs it
once, starts the bundled Stockfish for a short search, loads the levels, the opening book and the
repertoire, and reports whether an API key is configured (never the key itself), from where, and
which OS credential store backend is available.
"""

from __future__ import annotations

import importlib.util
import json
import os
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import chess
import chess.engine
import numpy as np

from chesstrainer.paths import FROZEN, bundle_dir


def _step(report: dict[str, Any], name: str, fn: Callable[[], dict[str, Any]]) -> None:
    t0 = time.perf_counter()
    try:
        result = fn()
        result["ok"] = True
    except Exception as err:  # noqa: BLE001 - every failure goes into the report
        result = {
            "ok": False,
            "error": f"{type(err).__name__}: {err}",
            "trace": traceback.format_exc(),
        }
    result["seconds"] = round(time.perf_counter() - t0, 3)
    report[name] = result


def _classifier() -> dict[str, Any]:
    from chesstrainer.vision.recognizer import PieceRecognizer

    rec = PieceRecognizer()
    crops = np.full((64, rec.crop_size, rec.crop_size, 3), 200, np.uint8)
    t0 = time.perf_counter()
    probs = rec.classify_crops(crops)
    ms = (time.perf_counter() - t0) * 1000
    return {
        "backend": rec.backend,
        "model": rec.model_path.name,
        "shape": list(probs.shape),
        "empty_square_prob": round(float(probs[:, 0].mean()), 4),
        "inference_ms": round(ms, 1),
    }


def _engine() -> dict[str, Any]:
    from chesstrainer.engine.locate import find_stockfish

    search = find_stockfish(None)
    if search.path is None:
        raise FileNotFoundError("Stockfish not found: " + "; ".join(search.tried))
    with chess.engine.SimpleEngine.popen_uci(str(search.path)) as engine:
        info = engine.analyse(chess.Board(), chess.engine.Limit(depth=12))
        name = engine.id.get("name", "")
    bundled = bundle_dir() in search.path.parents
    return {"engine": name, "bundled": bundled, "best": info["pv"][0].uci()}


def _data() -> dict[str, Any]:
    from chesstrainer.levels import all_levels
    from chesstrainer.openings.book import Book
    from chesstrainer.openings.repertoire import all_specs

    book = Book.load()
    return {
        "levels": [lv.key for lv in all_levels()],
        "book_lines": len(book.entries),
        "repertoire_specs": len(all_specs()),
    }


def _api_key() -> dict[str, Any]:
    from chesstrainer.settings.env import api_key_with_source, credential_store_name, env_files

    files = [str(p) for p in env_files()]
    key, source = api_key_with_source()
    return {
        "key_configured": key is not None,
        "key_source": source,
        "credential_store": credential_store_name() or "none",
        "env_files_searched": files,
        "env_var_set": bool(os.environ.get("ANTHROPIC_API_KEY")),
    }


def run(out: Path) -> int:
    report: dict[str, Any] = {
        "frozen": FROZEN,
        "bundle_dir": str(bundle_dir()),
        "torch_importable": importlib.util.find_spec("torch") is not None,
    }
    _step(report, "classifier", _classifier)
    _step(report, "engine", _engine)
    _step(report, "data", _data)
    _step(report, "api_key", _api_key)
    ok = all(v.get("ok", True) for v in report.values() if isinstance(v, dict))
    report["ok"] = ok
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if ok else 1
