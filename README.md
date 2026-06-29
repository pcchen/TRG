# Levin-Nave TRG

A [cytnx](https://github.com/Cytnx-dev/Cytnx)-based implementation of the
**Levin-Nave Tensor Renormalization Group (TRG)** algorithm for 2D classical
lattice models — primarily the Ising and clock models.

## The algorithm

The Levin-Nave TRG coarse-grains a 2D tensor network to evaluate its partition
function (and hence the free energy per site) approximately but efficiently. The
partition function of a 2D classical model is written as a square-lattice network
of rank-4 tensors. Each step of the algorithm:

1. **Decompose** every rank-4 tensor into a pair of rank-3 tensors via a
   singular value decomposition (SVD), keeping only the largest `chi` singular
   values to bound the bond dimension.
2. **Contract** the rank-3 tensors of four neighboring plaquettes into a new
   rank-4 tensor on a coarser, rotated lattice.

Repeating these steps halves the number of tensors each iteration, so the network
is renormalized down to a single tensor in a logarithmic number of steps. The
truncation parameter `chi` controls the trade-off between accuracy and cost.

### Reference

> M. Levin and C. P. Nave,
> *"Tensor Renormalization Group Approach to Two-Dimensional Classical Lattice
> Models,"*
> Phys. Rev. Lett. **99**, 120601 (2007).
> [doi:10.1103/PhysRevLett.99.120601](https://doi.org/10.1103/PhysRevLett.99.120601)
> · [arXiv:cond-mat/0611687](https://arxiv.org/abs/cond-mat/0611687)

## Related implementations & tutorials

- [smorita/TRG_Ising_2D](https://github.com/smorita/TRG_Ising_2D/tree/main) — a
  reference TRG implementation for the 2D Ising model.
- [ITensor TRG tutorial](https://itensor.org/docs.cgi?page=book/trg) — a
  step-by-step walkthrough of the algorithm.

## Repository

See [`CLAUDE.md`](CLAUDE.md) for the project layout and the uv environment setup,
and [`SKILLS.md`](SKILLS.md) for task-oriented recipes (running a calculation,
benchmarking against the exact free energy, working with cytnx tensors).
