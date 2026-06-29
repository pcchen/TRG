# SKILLS.md

Reusable know-how for this TRG codebase. Each section is a self-contained skill:
when you need it, what it covers, and how to do it.

## Set up / refresh the environment

This is a uv project; the env is `.venv`.

```bash
uv venv                                 # Python >=3.11
uv pip install "cytnx>=1.1.0"           # core requirement (pulls in numpy)
uv pip install scipy sympy reportlab    # remaining deps
uv run python -c "import cytnx; print(cytnx.__version__)"   # verify
```

Run scripts through the project interpreter so they don't pick up a global env:

```bash
uv run python trg.py
```

## Run a TRG calculation

The reference implementation is `trg.py`.

```python
from trg import TRG

t = TRG(temp=2.269, chi=16)   # temperature, bond-dimension cutoff
for _ in range(20):
    t.update()                # one coarse-graining step
# free energy per site accumulates in t.log_factors / t.n_spins
```

Key pieces: `initial_TN(temp)` builds the rank-4 Boltzmann tensor with legs
`u,r,d,l`; `TRG.update()` does SVD → truncate to `chi` → recontract.

## Benchmark against the exact solution

Use `exact_free_energy.free_energy_per_site` to check correctness — the TRG free
energy should converge to it as `chi` grows.

```python
from exact_free_energy import free_energy_per_site
f_exact = free_energy_per_site(temp)
```

`Exact_Ising.py` holds related exact-Ising reference code.

## Work with cytnx UniTensors

- Build bonds with `cytnx.Bond(dim)`; build tensors with
  `cytnx.UniTensor([bd, ...], rowrank=k)`.
- Name/relabel legs: `.set_name("T").relabel(["u","r","d","l"])`.
- Set/read a single element via `.at(labels, index).value`.
- Reshape contraction structure with `.set_rowrank(k)` and `.permute([...])`.
- Decompose with `cytnx.linalg.Svd(...)`; truncate singular values to `chi`.
- Inspect structure with `.print_diagram()` while debugging.

Target **cytnx >= 1.1.0** — older versions have different signatures.

## The clock model

`Clock_model/TRG_clock.py` adapts the TRG flow to the q-state clock model. Start
from the Ising version in `trg.py` and generalize the initial tensor's local
Boltzmann weights.

## Export tensor-network diagrams

`ascii2pdf.py` renders ASCII network drawings to PDF via `reportlab`; rendered
diagrams live in `docs/`.

## Add a dependency

1. `uv pip install <pkg>`
2. Add it to `pyproject.toml` under `[project] dependencies`.
3. Verify with `uv run python -c "import <pkg>"`.
