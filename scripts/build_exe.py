"""Build the Windows app with PyInstaller and audit what ships.

    python -m uv run python scripts/build_exe.py [--skip-build]

Steps: export the free-licence classifier to ONNX (if missing or older than its checkpoint),
draw the icon, run PyInstaller (``packaging/chesstrainer.spec``), then audit ``dist/ChessTrainer``:
  - no PyTorch, no training/eval data, no piece or board images, no CC BY-NC-SA set names;
  - the bundled model's sidecar says it came from ``square_classifier_free.pt``;
  - no ``.env`` file, and nothing that looks like an Anthropic key (``sk-ant-`` + 20 key
    characters) or the configured key's value in any file, in any module of the compressed
    Python archive inside the exe, or inside any .zip (the key is never printed);
  - the exe's headless ``--self-check`` passes (no window, no capture, no hotkeys), with
    CHESSTRAINER_HOME pointed at an empty temporary folder so no user .env can be read.
Writes ``data/eval/phase6_build.txt``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist" / "ChessTrainer"
SPEC = ROOT / "packaging" / "chesstrainer.spec"
ICON = ROOT / "packaging" / "chesstrainer.ico"
MODELS = ROOT / "data" / "models"
REPORT = ROOT / "data" / "eval" / "phase6_build.txt"


def ensure_onnx() -> None:
    pt = MODELS / "square_classifier_free.pt"
    onnx = MODELS / "square_classifier_free.onnx"
    if not pt.exists():
        sys.exit(f"{pt} is missing: run `python -m chesstrainer.vision.train --free-only` first")
    if onnx.exists() and onnx.stat().st_mtime >= pt.stat().st_mtime:
        return
    subprocess.run(
        [sys.executable, "-m", "chesstrainer.vision.export_onnx", "--checkpoint", str(pt)],
        check=True,
    )


def ensure_icon() -> None:
    if ICON.exists():
        return
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PIL import Image
    from PyQt6.QtCore import QRectF, Qt
    from PyQt6.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter, QPainterPath

    _app = QGuiApplication.instance() or QGuiApplication([])
    size = 256
    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor(21, 120, 27))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(QRectF(8, 8, size - 16, size - 16), 48, 48)
    font = QFont("Segoe UI Symbol")
    font.setPixelSize(200)
    path = QPainterPath()
    path.addText(0, 0, font, "♞")
    path.translate(QRectF(0, 0, size, size).center() - path.boundingRect().center())
    p.setBrush(QColor(250, 250, 250))
    p.drawPath(path)
    p.end()
    png = ICON.with_suffix(".png")
    img.save(str(png))
    Image.open(png).save(ICON, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (256, 256)])
    png.unlink()


def build() -> float:
    t0 = time.time()
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    subprocess.run(
        [sys.executable, "-m", "PyInstaller", str(SPEC), "--noconfirm", "--clean",
         "--distpath", str(ROOT / "dist"), "--workpath", str(ROOT / "build")],
        check=True,
        env=env,
        cwd=ROOT,
    )  # fmt: skip
    return time.time() - t0


def configured_key() -> str | None:
    sys.path.insert(0, str(ROOT / "src"))
    from chesstrainer.settings.env import anthropic_api_key

    key: str | None = anthropic_api_key()
    return key


def audit() -> list[str]:
    from chesstrainer.vision.assets import PIECE_SETS

    problems: list[str] = []
    files = [f for f in DIST.rglob("*") if f.is_file()]
    rel = [f.relative_to(DIST).as_posix().lower() for f in files]
    for needle in ("torch/", "torch_", "torchvision", "c10.dll", "cudnn", "cublas"):
        hits = [r for r in rel if needle in r]
        if hits:
            problems.append(f"PyTorch/CUDA file shipped: {hits[:3]}")
    for r in rel:
        name = r.rsplit("/", 1)[-1]
        if name == ".env" or name.endswith(".env"):
            problems.append(f".env file shipped: {r}")
        if r.endswith((".npz", ".pt")) or "/assets/" in f"/{r}" or "/real/" in f"/{r}":
            problems.append(f"training/eval data shipped: {r}")
        if r.endswith((".png", ".svg", ".webp")) and "/pyqt6/" not in f"/{r}":
            problems.append(f"image shipped: {r}")
    nc = [s.name for s in PIECE_SETS if s.non_commercial]
    for r in rel:
        parts = set(r.replace(".", "/").split("/"))
        if any(n in parts for n in nc):
            problems.append(f"NC piece-set name in path: {r}")
    meta_file = DIST / "_internal" / "models" / "square_classifier_free.json"
    if not meta_file.exists():
        problems.append("bundled model sidecar missing")
    elif json.loads(meta_file.read_text(encoding="utf-8")).get("source") != (
        "square_classifier_free.pt"
    ):
        problems.append("bundled model was not exported from square_classifier_free.pt")
    key = configured_key()
    blobs = [(str(f.relative_to(DIST)), f.read_bytes()) for f in files]
    blobs += python_archive(DIST / "ChessTrainer.exe")
    for f in files:
        if f.suffix == ".zip":
            with zipfile.ZipFile(f) as z:
                blobs += [(f"{f.name}:{n}", z.read(n)) for n in z.namelist()]
    for where, data in blobs:
        if KEY_RE.search(data):
            problems.append(f"something shaped like an API key found in {where}")
        if key and key.encode() in data:
            problems.append(f"the configured API key found in {where}")
    return problems


KEY_RE = re.compile(rb"sk-ant-[A-Za-z0-9_\-]{20,}")


def python_archive(exe: Path) -> list[tuple[str, bytes]]:
    """Every module in the exe's compressed PYZ archive, decompressed (code objects marshalled)."""
    import marshal

    from PyInstaller.archive.readers import CArchiveReader, ZlibArchiveReader

    out: list[tuple[str, bytes]] = []
    pkg = CArchiveReader(str(exe))
    for name in pkg.toc:
        if not name.endswith(".pyz"):
            continue
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.pyz"
            path.write_bytes(pkg.extract(name))
            pyz = ZlibArchiveReader(str(path))
            for mod in pyz.toc:
                code = pyz.extract(mod)
                if code is not None:
                    out.append((f"{name}:{mod}", marshal.dumps(code)))
    if not out:
        out.append(("(no PYZ found)", b""))
    return out


def self_check() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as home:
        out = Path(home) / "report.json"
        env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY",)}
        env["CHESSTRAINER_HOME"] = home
        env.pop("STOCKFISH_PATH", None)
        proc = subprocess.run(
            [str(DIST / "ChessTrainer.exe"), "--self-check", str(out)], env=env, timeout=180
        )
        report: dict[str, object] = (
            json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
        )
        report["exit_code"] = proc.returncode
        return report


def size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-build", action="store_true", help="audit an existing dist/")
    args = parser.parse_args()
    ensure_onnx()
    ensure_icon()
    seconds = 0.0 if args.skip_build else build()
    problems = audit()
    check = self_check()
    engine_dir = DIST / "_internal" / "engine"
    lines = [
        f"dist: {DIST}",
        f"build time: {seconds:.0f} s" if seconds else "build: skipped (existing dist audited)",
        f"size: {size_mb(DIST):.0f} MB total, Stockfish {size_mb(engine_dir):.0f} MB, "
        f"files: {sum(1 for f in DIST.rglob('*') if f.is_file())}",
        "audit: " + ("passed" if not problems else f"{len(problems)} problem(s)"),
        *(f"  - {p}" for p in problems[:40]),
        "self-check (frozen exe, headless, empty CHESSTRAINER_HOME, no ANTHROPIC_API_KEY env):",
    ]
    for k, v in check.items():
        if isinstance(v, dict):
            v = {kk: vv for kk, vv in v.items() if kk != "trace"}
        lines.append(f"  {k}: {v}")
    text = "\n".join(lines)
    print(text)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text + "\n", encoding="utf-8")
    ok = not problems and check.get("ok") is True and check.get("exit_code") == 0
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
