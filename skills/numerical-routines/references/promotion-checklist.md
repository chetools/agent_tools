# Promotion checklist

Follow this checklist when adding a routine to the library. Every item is required before you open the pull request. Copy the list into the pull request description and tick items off.

## Before you start

- [ ] Confirm the routine is not already in `numerics/catalog.toml` under another name.
- [ ] Confirm it is general: reusable beyond the project that needed it.
- [ ] Note the source you are following (paper, book chapter, SciPy module, Fortran code) and check its license. Keep any required notices in the source file header.

## Implementation

- [ ] **Rust** in `numerics/rust/src/<area>/<routine>.rs`: `f64`, closure-based callbacks, no global state, no new dependencies without justification in the pull request.
- [ ] **Python twin** in `numerics/python/<area>/<routine>.py`: the same algorithm, defaults and stopping tests, written for readability. It is both the teaching version and the test oracle.
- [ ] A result struct reporting success, message, and evaluation counts (function, Jacobian, LU factorizations, steps accepted and rejected, as applicable).
- [ ] Doc comments on every public item: what it solves, defaults, failure modes.

## Tests and reference data

- [ ] At least two benchmark problems from `numerics/tests/problems/` (add new ones if needed). One should be hard for the problem class. Typical choices:
  - ODE: Robertson, Van der Pol (μ = 1000), HIRES, Oregonator for stiff; Arenstorf orbit or a two-body problem for nonstiff.
  - Roots: Powell singular function, Broyden tridiagonal, problems with a bad starting point.
  - Optimization: Rosenbrock, Himmelblau, selected Moré–Garbow–Hillstrom problems.
- [ ] Reference outputs saved in `numerics/tests/reference_data/`. The file header records the generator (e.g. SciPy version and method), tolerances and date. Use tighter tolerances than the routine will be tested at, or published reference solutions.
- [ ] Rust and Python each compared against the reference data, with error norms and evaluation counts recorded in the test output.
- [ ] Rust and Python compared against each other: same accepted steps or iterates, to a stated tolerance.
- [ ] `make verify` passes locally.

## Catalog and documentation

- [ ] Entry added to `numerics/catalog.toml` with `status = "experimental"`.
- [ ] Row added or updated in `skills/numerical-routines/references/method-selection.md`.
- [ ] Known failure symptoms added to `skills/numerical-routines/references/troubleshooting.md`.
- [ ] Minimal working example in `numerics/examples/`.
- [ ] `CHANGELOG.md` entry and version bump.

## Pull request

- [ ] Branch name `routine/<routine-name>`. No direct push to `main`.
- [ ] Description includes the verification numbers, not only "tests pass".
- [ ] A maintainer merge (not the agent) moves the status to `verified`.
