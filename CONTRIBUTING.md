# Contributing to ChessTrainer

Thanks for helping. Bug reports, misread boards and pull requests are all welcome.

## Reporting a misread board

The most useful report is a screenshot of the board it got wrong plus the correct FEN. If you
can, add it to `data/real/` with a row in `data/real/labels.csv` (`file,fen,white_bottom`) and
say which site and piece set it is. Do not include usernames or chat you do not want public.

## Development setup

Windows 10/11, Python 3.11, [uv](https://docs.astral.sh/uv/).

```powershell
python -m uv sync                                        # dev environment (.venv)
python -m uv run python -m chesstrainer.vision.weights   # classifier weights from the release
```

Stockfish: unzip it into `data/engine/` or set `STOCKFISH_PATH` in `.env` (copy `.env.example`).
Engine tests skip without it.

## Before you open a pull request

All three must pass; CI runs the same commands:

```powershell
.venv/Scripts/ruff format src tests scripts
.venv/Scripts/ruff check src tests scripts
.venv/Scripts/mypy src                          # strict mode
$env:QT_QPA_PLATFORM = "offscreen"; python -m uv run pytest
```

Guidelines:

- **Typed, tested code.** `mypy --strict` clean, ruff with line length 100. Add a test for every
  behaviour change.
- **Tests never touch the real screen, keyboard, credential store or API.** Use
  `QT_QPA_PLATFORM=offscreen`, fake Claude clients (see `tests/test_explain.py`) and the in-memory
  keyring from `tests/conftest.py`. On-screen self-tests (`scripts/live_selftest.py`, ...) are
  run by hand only.
- **Level behaviour lives in `levels.yaml`, opening knowledge in `openings/repertoire.yaml`.**
  Do not hardcode either in Python.
- **Explanations stay grounded.** New explanation content must come from a computed fact in
  `explain/facts.py`, never from free text in the prompt.
- **No secrets, no weights, no generated data in git.** `.gitignore` covers them; keys belong in
  the credential store or `.env`.
- **Licences.** Only add training assets with a licence that allows redistribution of trained
  weights, and list them in `THIRD_PARTY_NOTICES.md`. Contributions are accepted under GPL-3.0.

## Fair play

ChessTrainer is for analysis boards, bots and reviewing your own games. Features whose main
purpose is evading cheat detection in live games will not be merged.
