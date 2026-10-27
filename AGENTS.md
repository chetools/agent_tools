# Instructions for AI agents

These rules apply to every agent working in this repository or using its routines in another project. They are written to be tool-neutral. If your agent supports Agent Skills, the same guidance is packaged in `skills/numerical-routines/`.

## 1. Check the catalog before writing a numerical routine

Before you implement an ODE integrator, root solver, optimizer, or supporting linear algebra, read `numerics/catalog.toml`.

- If a routine with `status = "verified"` fits the problem, **use it.** Do not rewrite it.
- If only an `experimental` routine fits, you may use it. Say so in your output, and treat its results as unverified.
- If nothing fits, write the routine for your task, then promote it (section 3).

Use `skills/numerical-routines/references/method-selection.md` to choose between routines.

## 2. How to use a routine in another project

- **Rust / Rust-wasm:** depend on the crate in `numerics/rust/` by git path or tag. Do not copy files in. Unused functions are removed at link time with LTO, so the dependency does not bloat the wasm binary.
- **No Rust toolchain or no wasm32 target:** use the prebuilt `.wasm` release assets with their JS bindings.
- **Python / teaching pages:** use `numerics/python/`, which mirrors the Rust code step by step.
- Never patch a routine inside a downstream project. If it needs a fix, change it here (section 4).

## 3. Promotion: adding a routine that does not exist yet

When a task needs a general routine that is missing, add it here, following `skills/numerical-routines/references/promotion-checklist.md`. In short:

1. Rust implementation and Python reference twin, with the same algorithm and the same defaults.
2. A minimal working example in `numerics/examples/`.
3. Benchmark problems and stored reference outputs in `numerics/tests/`.
4. A catalog entry with `status = "experimental"`.
5. Documentation: method-selection entry and failure symptoms.
6. `make verify` passes.
7. Open a pull request. Do not push directly to `main`.

**Promote** a routine when it is general and likely to be reused (an integrator, a bracketed root finder, a line search). **Do not promote** problem-specific code (a residual for one reactor model, a tuned initial guess). That stays in its project.

### If you cannot push here

Some sessions lack push access or network access. In that case, write the routine and a filled-in copy of `templates/library-candidate.md` into a `library-candidates/<routine-name>/` folder **in the repository you are working in**. Mention it in your final output, so a later session can finish the promotion.

## 4. Changing an existing routine

- Bug fixes and improvements go through a pull request with tests that fail before and pass after.
- Record the change in `CHANGELOG.md` and bump the version (semantic versioning). A change in default tolerances or step-control behaviour counts as a breaking change.
- If a change alters results beyond the stored tolerances, regenerate reference data only with a stated reason in the pull request.

## 5. Verification rules

- `status = "verified"` means **both** that the routine passes the benchmark suite **and** that a human maintainer merged the pull request that set it. An agent never sets `verified` on its own work.
- Reference outputs are generated once from a trusted implementation (for example SciPy at tight tolerances) or taken from published reference solutions, and are committed as data. Record the source and settings in the data file's header.
- Report results with numbers: error norms against reference, plus function, Jacobian and step counts. A claim such as "works well" is not verification.
- Anything not computed and checked is labelled as an expectation.

## 6. When a result looks wrong

Reproduce it with the Python reference twin, and with SciPy if available, before explaining or changing anything. If the reference implementation agrees, the problem is probably in the problem setup, not the routine.

## 7. Style for routines

- `f64` throughout. Right-hand sides, Jacobians and objectives are passed as closures. No global state.
- Minimal dependencies. Dense linear algebra for small systems lives in this crate.
- Every option has a documented default that matches the Python twin.
- Return a result struct with success flag, message, and evaluation counts, similar to SciPy's `OdeResult` / `OptimizeResult`.
