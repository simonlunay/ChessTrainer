# PyInstaller spec for the Windows app (onedir, no console).
#
#   python -m uv run python scripts/build_exe.py      # builds + audits; use that, not this directly
#
# Ships: the chesstrainer package with its YAML/TSV data, the ONNX classifier trained without the
# CC BY-NC-SA piece sets (no piece images at all), and Stockfish (GPLv3, with its source and
# licence). Never ships: PyTorch, training assets, eval data, .env files or API keys.
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files

ROOT = Path(SPECPATH).parent
MODEL = ROOT / "data" / "models" / "square_classifier_free.onnx"
ENGINE = ROOT / "data" / "engine" / "stockfish"

datas = collect_data_files(
    "chesstrainer", includes=["**/*.yaml", "openings/data/*.tsv", "openings/data/*.txt"]
)
datas += [(str(MODEL), "models"), (str(MODEL.with_suffix(".json")), "models")]
datas += [(str(ROOT / "packaging" / "THIRD_PARTY_LICENSES.txt"), ".")]
if ENGINE.is_dir():
    for f in ENGINE.rglob("*"):
        if f.is_file():
            datas.append((str(f), str(Path("engine/stockfish") / f.parent.relative_to(ENGINE))))

EXCLUDES = [
    "torch", "torchvision", "torchaudio", "tensorboard", "onnx", "onnxscript", "tqdm",
    "matplotlib", "pytest", "IPython", "jupyter", "notebook", "PyInstaller", "mypy", "ruff",
    "tkinter", "chesstrainer.vision.train", "chesstrainer.vision.dataset",
    "chesstrainer.vision.synth", "chesstrainer.vision.evaluate", "chesstrainer.vision.model",
    "chesstrainer.vision.export_onnx", "chesstrainer.vision.assets",
    "chesstrainer.vision.screens",
]  # fmt: skip

a = Analysis(
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT / "src")],
    datas=datas,
    excludes=EXCLUDES,
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ChessTrainer",
    console=False,
    icon=str(ROOT / "packaging" / "chesstrainer.ico"),
    version=None,
)
coll = COLLECT(exe, a.binaries, a.datas, name="ChessTrainer")
