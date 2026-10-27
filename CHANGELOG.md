# Changelog

All notable changes to the routines in this library. Versions follow semantic versioning.

## Unreleased

- Draft: agent instructions, skill, catalog schema and templates. No routines yet.
- Scope broadened from numerical solvers to general scientific and engineering routines: `numerics/` renamed `library/`, skill renamed `scientific-routines`, area guides added.
- Scaffold: `make verify`, crate `agent_tools` (`library/rust/`), Python package `agent_tools` (`library/python/`), pytest helpers (`rust_example`, `python_example`, `load_reference`), catalog check. Layout in README updated.
- `dual` 0.1.0 (experimental, new area `autodiff`): `Real` trait and forward-mode `Dual` numbers; nested duals give second derivatives.
- `lu_dense` 0.1.0 (experimental, `linalg`): dense LU with partial pivoting, generic over `Real`.
