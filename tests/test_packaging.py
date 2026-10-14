"""Packaging-related behaviour: the ONNX backend, frozen-mode paths, the free-licence split and
the headless self-check."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from chesstrainer import paths
from chesstrainer.settings import env
from chesstrainer.vision.assets import FREE_PIECE_SETS, PIECE_SETS
from chesstrainer.vision.dataset import train_split
from chesstrainer.vision.recognizer import PieceRecognizer, default_onnx


def test_free_split_has_no_non_commercial_sets() -> None:
    nc = {s.name for s in PIECE_SETS if "NC" in s.license}
    assert nc and not (nc & FREE_PIECE_SETS)
    assert "cburnett" in FREE_PIECE_SETS
    assert not set(train_split(free_only=True).piece_sets) & nc


def test_frozen_app_never_reads_the_project_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env, "FROZEN", False)
    assert env.PROJECT_ROOT / ".env" in env.env_files()
    monkeypatch.setattr(env, "FROZEN", True)
    files = env.env_files()
    assert files == [env.settings_path().parent / ".env"]


def test_bundle_dir_when_frozen(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert paths.bundle_dir() == paths.PROJECT_ROOT / "data"
    monkeypatch.setattr(paths, "FROZEN", True)
    monkeypatch.setattr(paths.sys, "_MEIPASS", str(tmp_path), raising=False)
    assert paths.bundle_dir() == tmp_path
    assert paths.model_dir() == tmp_path / "models"


@pytest.mark.skipif(not default_onnx().exists(), reason="ONNX model not exported")
def test_onnx_backend_matches_torch() -> None:
    from chesstrainer.vision.model import FREE_CHECKPOINT

    if not FREE_CHECKPOINT.exists():
        pytest.skip("free checkpoint missing")
    meta = json.loads(default_onnx().with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["source"] == "square_classifier_free.pt"
    onnx = PieceRecognizer(default_onnx())
    torch_rec = PieceRecognizer(FREE_CHECKPOINT, device="cpu", backend="torch")
    assert onnx.backend == "onnx" and torch_rec.backend == "torch"
    rng = np.random.default_rng(0)
    crops = rng.integers(0, 256, (64, 64, 64, 3), dtype=np.uint8)
    a, b = onnx.classify_crops(crops), torch_rec.classify_crops(crops)
    assert a.shape == (64, 13)
    assert np.abs(a - b).max() < 1e-4
    assert (a.argmax(1) == b.argmax(1)).all()


def test_auto_backend_falls_back_to_onnx(monkeypatch: pytest.MonkeyPatch) -> None:
    from chesstrainer.vision import recognizer

    if not default_onnx().exists():
        pytest.skip("ONNX model not exported")
    monkeypatch.setattr(recognizer, "torch_available", lambda: False)
    rec = PieceRecognizer()
    assert rec.backend == "onnx" and rec.device == "cpu" and not rec.keep_warm


def test_self_check_writes_a_report(tmp_path: Path) -> None:
    from chesstrainer.selfcheck import run

    out = tmp_path / "r.json"
    code = run(out)
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["data"]["ok"] and report["classifier"]["ok"]
    assert "api_key" in report and "sk-ant" not in out.read_text(encoding="utf-8")
    assert code in (0, 1)  # 1 only when Stockfish is missing on this machine
