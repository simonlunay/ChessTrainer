"""Downloading the shipped classifier from the release when it is missing (local HTTP server)."""

from __future__ import annotations

import functools
import http.server
import shutil
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from chesstrainer.vision import weights
from chesstrainer.vision.recognizer import default_onnx

MODEL = Path(__file__).resolve().parents[1] / "data" / "models" / weights.ONNX_MODEL
needs_model = pytest.mark.skipif(not MODEL.exists(), reason="ONNX model not present")


@pytest.fixture
def server(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    root = tmp_path / "release"
    root.mkdir()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    handler.log_message = lambda *a, **k: None  # type: ignore[attr-defined]
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/", root
    httpd.shutdown()


@pytest.fixture
def empty_install(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    models = tmp_path / "models"
    monkeypatch.setattr(weights, "model_dir", lambda: models)
    monkeypatch.setenv("CHESSTRAINER_HOME", str(tmp_path / "home"))
    return models


@needs_model
def test_missing_model_is_downloaded_and_verified(
    server: tuple[str, Path], empty_install: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url, root = server
    shutil.copy(MODEL, root / weights.ONNX_MODEL)
    shutil.copy(MODEL.with_suffix(".json"), root / weights.SIDECAR)
    monkeypatch.setenv("CHESSTRAINER_WEIGHTS_URL", url)
    assert weights.find_model() is None
    messages: list[str] = []
    path = weights.ensure_model(messages.append)
    assert path == empty_install / weights.ONNX_MODEL and path.is_file()
    assert (empty_install / weights.SIDECAR).is_file() and len(messages) == 2
    assert weights.find_model() == path and default_onnx() == path

    from chesstrainer.vision.recognizer import PieceRecognizer

    rec = PieceRecognizer(backend="onnx")
    assert rec.model_path == path and rec.device == "cpu"


def test_frozen_app_downloads_to_the_user_folder(
    empty_install: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(weights, "FROZEN", True)
    assert weights.download_dir() == tmp_path / "home" / "models"


def test_wrong_hash_is_rejected(
    server: tuple[str, Path], empty_install: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url, root = server
    (root / weights.ONNX_MODEL).write_bytes(b"not a model")
    monkeypatch.setenv("CHESSTRAINER_WEIGHTS_URL", url)
    with pytest.raises(weights.WeightsError, match="SHA-256"):
        weights.ensure_model()
    assert not (empty_install / weights.ONNX_MODEL).exists()


def test_missing_release_gives_a_readable_error(
    server: tuple[str, Path], empty_install: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHESSTRAINER_WEIGHTS_URL", server[0])
    with pytest.raises(weights.WeightsError, match="404.*releases/tag/v"):
        weights.ensure_model()
