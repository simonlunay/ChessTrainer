"""Find the shipped piece classifier, downloading it from the GitHub Release when it is missing.

The ONNX weights are not in the git repository. They are published as assets of the release
``RELEASE_TAG`` and checked against a pinned SHA-256 after download. Search order:

1. ``models/`` in the bundle (packaged app) or ``data/models/`` (source checkout);
2. ``%APPDATA%\\ChessTrainer\\models`` (where the packaged app downloads to).

A source checkout downloads into ``data/models``. ``CHESSTRAINER_WEIGHTS_URL`` overrides the base
URL (a mirror, or a local server in tests).

    python -m chesstrainer.vision.weights     # download now (prints where it went)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from chesstrainer.paths import FROZEN, ONNX_MODEL, model_dir
from chesstrainer.settings.config import settings_path
from chesstrainer.vision.labels import CLASS_NAMES

log = logging.getLogger(__name__)

REPO = "simonlunay/ChessTrainer"
RELEASE_TAG = "v0.1.0"
RELEASE_URL = f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}/"
SIDECAR = ONNX_MODEL.removesuffix(".onnx") + ".json"
#: SHA-256 of the released ONNX file (``square_classifier_free.pt``, 24k steps, free sets only).
ONNX_SHA256 = "d775e8ce1ce4bd9488f0b6ad05df602af31cec82628da3c36392429a33df4b56"
MAX_BYTES = 50_000_000

Progress = Callable[[str], None]


class WeightsError(RuntimeError):
    """The model is missing and could not be downloaded (readable message)."""


def user_model_dir() -> Path:
    return settings_path().parent / "models"


def search_dirs() -> list[Path]:
    return [model_dir(), user_model_dir()]


def find_model() -> Path | None:
    for d in search_dirs():
        if (d / ONNX_MODEL).is_file():
            return d / ONNX_MODEL
    return None


def download_dir() -> Path:
    return user_model_dir() if FROZEN else model_dir()


def base_url() -> str:
    url = os.environ.get("CHESSTRAINER_WEIGHTS_URL", "").strip() or RELEASE_URL
    return url if url.endswith("/") else url + "/"


def _fetch(url: str, limit: int = MAX_BYTES) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "chesstrainer-weights"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data: bytes = resp.read(limit + 1)
    except urllib.error.HTTPError as err:
        raise WeightsError(f"download failed ({err.code} {err.reason}): {url}") from err
    except (urllib.error.URLError, TimeoutError, OSError) as err:
        raise WeightsError(f"download failed (offline?): {url}: {err}") from err
    if len(data) > limit:
        raise WeightsError(f"download too large: {url}")
    return data


def _write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


def download_model(dest: Path | None = None, progress: Progress | None = None) -> Path:
    """Download the ONNX model + sidecar into ``dest``, verifying the SHA-256. Returns the path."""
    dest = dest or download_dir()
    say = progress or (lambda msg: log.info("%s", msg))
    url = base_url()
    say(f"Downloading the piece classifier (~2 MB) from {url}")
    data = _fetch(url + ONNX_MODEL)
    digest = hashlib.sha256(data).hexdigest()
    if digest != ONNX_SHA256:
        raise WeightsError(
            f"downloaded {ONNX_MODEL} has SHA-256 {digest}, expected {ONNX_SHA256}; not used"
        )
    meta = json.loads(_fetch(url + SIDECAR, 100_000).decode("utf-8"))
    if tuple(meta.get("class_names", ())) != CLASS_NAMES:
        raise WeightsError(f"downloaded {SIDECAR} has an unexpected class order")
    try:
        dest.mkdir(parents=True, exist_ok=True)
        if not (dest / SIDECAR).is_file():
            _write(dest / SIDECAR, json.dumps(meta, indent=2).encode("utf-8"))
        _write(dest / ONNX_MODEL, data)
    except OSError as err:
        raise WeightsError(f"could not save the model in {dest}: {err}") from err
    say(f"Piece classifier saved to {dest / ONNX_MODEL}")
    return dest / ONNX_MODEL


def ensure_model(progress: Progress | None = None) -> Path:
    """The ONNX model path, downloading it first if it is not installed."""
    found = find_model()
    if found is not None:
        return found
    try:
        return download_model(progress=progress)
    except WeightsError as err:
        raise WeightsError(
            f"The piece classifier ({ONNX_MODEL}) is missing and could not be downloaded: {err}. "
            f"Download it from https://github.com/{REPO}/releases/tag/{RELEASE_TAG} and put it "
            f"in {download_dir()}."
        ) from err


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    found = find_model()
    print(found if found is not None else download_model(progress=print))


if __name__ == "__main__":
    main()
