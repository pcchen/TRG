# CLAUDE.md

Guidance for working in this repository.

## What this is

A research codebase for the **Tensor Renormalization Group (TRG)** algorithm applied to
2D classical lattice models — primarily the **Ising model** and the **clock model**.
It coarse-grains a 2D tensor network to compute the free energy per site, and compares
results against exact solutions.

The numerical engine is **[cytnx](https://github.com/Cytnx-dev/Cytnx)** (`UniTensor`,
`Bond`, `linalg.Svd`), with `numpy`/`scipy` for reference computations and `sympy`/
`reportlab` for utilities and diagram export.

## Environment

This is a **uv project**. The virtual environment lives in `.venv`.

```bash
# Create the env (Python >=3.11) and install dependencies
uv venv
uv pip install "cytnx>=1.1.0"          # core requirement
uv pip install scipy sympy reportlab   # numpy comes in with cytnx

# Run anything with the project interpreter
uv run python trg.py
# or activate: source .venv/bin/activate
```

> Note: a global `myvenv` may be active via `$VIRTUAL_ENV`. Always run through
> `uv run` or pass `--python .venv` to `uv pip install` so commands target the
> project's `.venv`, not the global one.

Dependencies are declared in `pyproject.toml` (`package = false` — this is an
application/script collection, not an installable library).

## Layout

- `trg.py` — current, cleanest TRG implementation (`TRG` class, `initial_TN`).
- `myTRG.py`, `TRG_Ising.py`, `TRG_Ising_Recycle.py` — earlier / experimental TRG variants.
- `exact_free_energy.py`, `Exact_Ising.py` — exact 2D Ising free energy for benchmarking.
- `Clock_model/TRG_clock.py` — TRG for the q-state clock model.
- `Network.py`, `NewContract.py`, `Combiner.py`, `extension.py`, `WM.py` — tensor-network
  contraction helpers and utilities.
- `common.py`, `util_sympy.py`, `ascii2pdf.py` — small helpers (initial tensors, symbolic
  math, ASCII-diagram → PDF export).
- `test/` — `test_svd.py`, `Test.py` (ad-hoc checks, not a formal pytest suite).
- `docs/` — notes and PDF diagrams of the tensor networks.

There is no single entry point; files are run individually as scripts.

## Conventions

- Tensor legs are labeled by direction: `"u"`, `"r"`, `"d"`, `"l"` (up/right/down/left).
- `chi` is the bond dimension cut kept after each SVD; `temp` is the temperature.
- Prefer `trg.py` as the reference implementation when extending or comparing — the
  other `*TRG*.py` files are kept for history.
- Validate changes by checking the computed free energy per site against
  `exact_free_energy.free_energy_per_site`.

## Working notes

- When adding a dependency, install it with `uv pip install <pkg>` **and** add it to
  `pyproject.toml`'s `dependencies`.
- cytnx APIs differ across versions; this repo targets **cytnx >= 1.1.0**.
