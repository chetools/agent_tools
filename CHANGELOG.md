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
- `porous_permeability` 0.1.0 (experimental, `properties`; area guide added): Kozeny-Carman, Ergun, Gebart (square, hexagonal), Sangani-Acrivos square-array permeability correlations; NaN outside the domain.
- `rtd_analysis` 0.1.0 (experimental, `stats`): RTD moments with exponential tail extrapolation, tanks-in-series and axial-dispersion models, moment and least-squares fits of N and Pe, two-plane dispersion analysis.
- `lhs_maximin` 0.1.0 (experimental, `stats`): maximin Latin hypercube design, best of k random Latin hypercubes.
- `bfgs` 0.1.0 (experimental, `optimize`): BFGS with Armijo backtracking, unconstrained.
- `bfgs` 0.1.0 (review fixes): stall stop (step leaves `x` unchanged, `success = false`, "no progress"), validation of `gtol` and `max_iter`, approximate-Wolfe acceptance in the noise regime (ill-conditioned quadratics no longer stall near max|g| = 1e-8), evaluation-cost documentation from the 8-D quadratic case. Only the Rust 5-D quadratic case changed (nit 14 to 13, nfev 31 to 26, now equal to Python).
- `gp_matern52` 0.1.0 (experimental, new area guide `stats`): Matern-5/2 ARD Gaussian-process regression with maximum-likelihood hyperparameters, predictions and posterior covariance.
- `imse_batch` 0.1.0 (experimental, `stats`): greedy IMSE batch selection with rank-one covariance downdates.
- `hermite_extremum` 0.1.0 (experimental, new area `interpolate`): global maximum or minimum of the piecewise cubic Hermite interpolant through `(t, y, dy/dt)`, with location, value and interval index; generic over `Real`.
- `cubic_hermite` 0.1.0 (experimental, `interpolate`): piecewise cubic Hermite evaluator.
- `rosenbrock23` 0.1.0 (experimental, `ode`): adaptive Rosenbrock 2(3) (ode23s) stiff integrator generic over `Real`, with optional `df/dt`, per-component `atol`, dual-aware step control, and stored `t, y, dy/dt`.
- `block_tridiagonal` 0.1.0 (experimental, `linalg`): block-tridiagonal LU (block Thomas), runtime block size, factor once and solve many, generic over `Real`.
