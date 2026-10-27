# Promotion checklist

Follow this checklist when adding a routine (or a new area) to the library. Every item is required before you open the pull request, unless marked otherwise. Copy the list into the pull request description and tick items off.

## Before you start

- [ ] Confirm the routine is not already in `library/catalog.toml` under another name.
- [ ] Confirm it is general: reusable beyond the project that needed it.
- [ ] Note the source you are following (paper, book chapter, established library module, data handbook) and check its license or terms. Keep any required notices in the source file header.
- [ ] Pick its area. If none fits, add a new area: an entry in the `README.md` area table and `skills/scientific-routines/references/areas/<area>.md` from `_template.md`.

## Implementation

- [ ] **Rust** in `library/rust/src/<area>/<routine>.rs`: closure-based callbacks, no global state, no new dependencies without justification in the pull request.
- [ ] **Python reference** in `library/python/<area>/<routine>.py`: the same algorithm, defaults and stopping tests, written for readability. It is both the teaching version and the test oracle.
- [ ] If one of the two implementations adds nothing for this routine, it is skipped and the reason is stated in the catalog `notes`.
- [ ] Units documented on every public input and output.
- [ ] Iterative routines return a result struct reporting success, message, and evaluation counts.
- [ ] Doc comments on every public item: what it does, defaults, valid ranges, failure modes.

### Area `gpu` (WebGPU kernels)

- [ ] Kernel in `library/wgsl/<routine>.wgsl` and host wrapper in `library/js/<routine>.js` built on `library/js/gpu_util.js`: own uniform struct and bind group, explicit bind-group layouts, compilation errors and device limits raise errors. No Rust implementation (AGENTS.md section 7).
- [ ] Float64 Python reference in `library/python/gpu/<routine>.py`, mirroring the kernel step by step, with `make_reference()` producing the stored test data.
- [ ] GPU test `library/tests/gpu/<routine>.test.js` (rows of error, tolerance, pass) run by `library/tests/gpu/run.mjs`; Python test `library/tests/python/test_<routine>.py` against an independent method.
- [ ] Planted-defect run: a deliberately broken kernel line makes the GPU test fail.
- [ ] `verified_rtol` holds the stated `f32` tolerance, set with margin over the measured error.

## Tests and reference data

- [ ] At least two test cases, one of them hard for the problem class. The area guide lists typical choices; for numerical solvers, for example, Robertson and Van der Pol (μ = 1000) for stiff ODEs, Rosenbrock for optimization.
- [ ] Reference outputs saved in `library/tests/reference_data/`. The file header records the generator (library and version, method, tolerances) or the published source, and the date.
- [ ] Physical data: source cited (author, title, edition or DOI), valid ranges and units recorded; tests check values against the source's own tables.
- [ ] Rust and Python each compared against the reference data, with errors (and evaluation counts, where relevant) recorded in the test output.
- [ ] Rust and Python compared against each other to a stated tolerance.
- [ ] `make verify` passes locally.

## Catalog and documentation

- [ ] Entry added to `library/catalog.toml` with `status = "experimental"`.
- [ ] Row added or updated in the area guide.
- [ ] Known failure symptoms added to `troubleshooting.md` or the area guide.
- [ ] Minimal working example in `library/examples/`.
- [ ] `CHANGELOG.md` entry and version bump.

## Pull request

- [ ] Branch name `routine/<routine-name>`. No direct push to `main`.
- [ ] Description includes the verification numbers, not only "tests pass".
- [ ] A review report with verification numbers is posted on the pull request.
- [ ] The status moves to `verified` only on a maintainer's explicit approval of this pull request. If an agent performs the merge, it first comments "Merged on instruction from <maintainer>, <date>." (see `AGENTS.md` section 5).
