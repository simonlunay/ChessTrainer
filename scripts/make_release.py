"""Package a GitHub Release: the zipped Windows build and the shipped classifier weights.

    python -m uv run python scripts/build_exe.py           # build + audit dist/ChessTrainer first
    python -m uv run python scripts/make_release.py        # -> release/ (assets + SHA256SUMS.txt)
    python -m uv run python scripts/make_release.py --publish   # + draft release via the gh CLI

Assets (names are what ``vision/weights.py`` downloads, so keep them):
  - ``ChessTrainer-<version>-windows-x64.zip``  (dist/ChessTrainer, incl. Stockfish + source)
  - ``square_classifier_free.onnx`` + ``square_classifier_free.json``  (the shipped weights)
  - ``SHA256SUMS.txt``
The ONNX file must match ``weights.ONNX_SHA256``, otherwise the app would reject the download.
``--publish`` creates a *draft* release for tag ``weights.RELEASE_TAG`` with ``gh release create``
(needs the GitHub CLI, logged in, and the tag's commit pushed). Publish the draft on github.com.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chesstrainer.vision.weights import ONNX_MODEL, ONNX_SHA256, RELEASE_TAG, SIDECAR  # noqa: E402

DIST = ROOT / "dist" / "ChessTrainer"
MODELS = ROOT / "data" / "models"
OUT = ROOT / "release"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def zip_build(dest: Path) -> None:
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in sorted(DIST.rglob("*")):
            if f.is_file():
                z.write(f, Path("ChessTrainer") / f.relative_to(DIST))


def notes(version: str, files: list[Path]) -> str:
    sums = "\n".join(f"{sha256(f)}  {f.name}" for f in files)
    return f"""ChessTrainer {version} for Windows 10/11 (x64).

**Install:** download `{files[0].name}`, unzip it anywhere, run `ChessTrainer.exe`.
Stockfish 19 and the piece classifier are included. Explanations are optional: paste your own
Anthropic API key in the first-run dialog (stored in Windows Credential Manager).

**Weights:** `{ONNX_MODEL}` + `{SIDECAR}` are the shipped piece classifier, trained only on
free-licence piece sets (see THIRD_PARTY_NOTICES.md). A source checkout downloads them
automatically when missing.

Built for analysis boards, bots and reviewing your own games. Do not use it in live rated games.

Licence: GPL-3.0. Source: this tag. Third-party licences: THIRD_PARTY_NOTICES.md (also inside
the zip).

SHA-256:
```
{sums}
```
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=RELEASE_TAG.lstrip("v"))
    parser.add_argument("--publish", action="store_true", help="create a draft release with gh")
    args = parser.parse_args()
    exe = DIST / "ChessTrainer.exe"
    if not exe.exists():
        sys.exit(f"{exe} missing: run scripts/build_exe.py first")
    onnx = MODELS / ONNX_MODEL
    if sha256(onnx) != ONNX_SHA256:
        sys.exit(f"{onnx} does not match weights.ONNX_SHA256; update the constant or the file")
    bundled = DIST / "_internal" / "models" / ONNX_MODEL
    if sha256(bundled) != ONNX_SHA256:
        sys.exit("the build bundles a different model than the release weights; rebuild")
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    zip_path = OUT / f"ChessTrainer-{args.version}-windows-x64.zip"
    zip_build(zip_path)
    shutil.copy2(onnx, OUT / ONNX_MODEL)
    shutil.copy2(MODELS / SIDECAR, OUT / SIDECAR)
    files = [zip_path, OUT / ONNX_MODEL, OUT / SIDECAR]
    (OUT / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256(f)}  {f.name}\n" for f in files), encoding="utf-8"
    )
    (OUT / "RELEASE_NOTES.md").write_text(notes(args.version, files), encoding="utf-8")
    for f in [*files, OUT / "SHA256SUMS.txt"]:
        print(f"{f.stat().st_size / 1e6:8.1f} MB  {f.relative_to(ROOT)}")
    if args.publish:
        gh = shutil.which("gh")
        if gh is None:
            sys.exit("gh (GitHub CLI) not found: https://cli.github.com, then `gh auth login`")
        subprocess.run(
            [gh, "release", "create", RELEASE_TAG, "--draft", "--title",
             f"ChessTrainer {args.version}", "--notes-file", str(OUT / "RELEASE_NOTES.md"),
             *map(str, files), str(OUT / "SHA256SUMS.txt")],
            check=True, cwd=ROOT,
        )  # fmt: skip


if __name__ == "__main__":
    main()
