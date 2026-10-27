# Changelog

All notable changes to the routines in this library. Versions follow semantic versioning.

## Unreleased

- Draft: agent instructions, skill, catalog schema and templates. No routines yet.
- Scope broadened from numerical solvers to general scientific and engineering routines: `numerics/` renamed `library/`, skill renamed `scientific-routines`, area guides added.
- Scaffold: `make verify`, crate `agent_tools` (`library/rust/`), Python package `agent_tools` (`library/python/`), pytest helpers (`rust_example`, `python_example`, `load_reference`), catalog check. Layout in README updated.
- `dual` 0.1.0 (experimental, new area `autodiff`): `Real` trait and forward-mode `Dual` numbers; nested duals give second derivatives.
- `lu_dense` 0.1.0 (experimental, `linalg`): dense LU with partial pivoting, generic over `Real`.
- `cholesky` 0.1.0 (experimental, `linalg`): Cholesky factorization and triangular solves.
- `splitmix64` 0.1.0 (experimental, `random`): SplitMix64 with uniform, normal and shuffle.
- `mulberry32` 0.1.0 (experimental, `random`): Mulberry32, bit-identical to the JavaScript version.
- `hermite_extremum` 0.1.0 (experimental, new area `interpolate`): global maximum or minimum of the piecewise cubic Hermite interpolant through `(t, y, dy/dt)`, with location, value and interval index; generic over `Real`.
- `cubic_hermite` 0.1.0 (experimental, `interpolate`): piecewise cubic Hermite evaluator.
- `rosenbrock23` 0.1.0 (experimental, `ode`): adaptive Rosenbrock 2(3) (ode23s) stiff integrator generic over `Real`, with optional `df/dt`, per-component `atol`, dual-aware step control, and stored `t, y, dy/dt`.
- `block_tridiagonal` 0.1.0 (experimental, `linalg`): block-tridiagonal LU (block Thomas), runtime block size, factor once and solve many, generic over `Real`.
