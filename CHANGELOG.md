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
- `bfgs` 0.1.0 (experimental, `optimize`): BFGS with Armijo backtracking, unconstrained.
- `bfgs` 0.1.0 (review fixes): stall stop (step leaves `x` unchanged, `success = false`, "no progress"), validation of `gtol` and `max_iter`, approximate-Wolfe acceptance in the noise regime (ill-conditioned quadratics no longer stall near max|g| = 1e-8), evaluation-cost documentation from the 8-D quadratic case. Only the Rust 5-D quadratic case changed (nit 14 to 13, nfev 31 to 26, now equal to Python).
