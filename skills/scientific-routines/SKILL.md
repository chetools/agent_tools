---
name: scientific-routines
description: Use before writing reusable scientific or engineering code (ODE/PDE solvers, root finding, optimization, linear algebra, interpolation, quadrature, FFT/filtering, regression and statistics, units, property correlations, data formats), especially in Rust, WASM or standalone pages. Finds a verified routine in chetools/agent_tools or adds a missing one.
---

# Scientific routines library (chetools/agent_tools)

This skill points to a shared library of routines that scientists and engineers use regularly. Each routine has a Rust implementation, a matching Python reference, and stored reference results. The rules in `AGENTS.md` at the repository root are binding. This file is the working summary.

## Workflow

1. **Identify the area and problem class.** For example: stiff IVP, monotone interpolation, weighted nonlinear regression, unit conversion. Area guides are in `references/areas/`.
2. **Look it up** in `library/catalog.toml`. Match on `area`, `problem_class` and `suitable_for`, and check `status`.
3. **Use it** if it fits. Rust and wasm projects depend on the crate (`library/rust/`). Python and teaching pages use `library/python/`. Start from the routine's file in `library/examples/`.
4. **If it is missing,** write it for the task, then promote it following `references/promotion-checklist.md`. If no area fits, propose a new one in the same pull request. Without push access to this repo, use the `library-candidates/` fallback described in `AGENTS.md`.
5. **If it misbehaves,** use `references/troubleshooting.md`. Reproduce with the Python reference before changing code.

## Non-negotiables

- Do not rewrite a routine that is in the catalog with `status = "verified"`.
- Never set `status = "verified"` yourself. New routines start as `experimental`; only a maintainer's explicit approval of that pull request makes them verified. You may merge on that explicit instruction, after posting the comment required by `AGENTS.md` section 5.
- Change routines here through a pull request, never by patching a downstream copy.
- Document units on every input and output. Cite the source of any physical data.
- Report numbers (errors against reference, evaluation counts), not impressions. Label anything unchecked as an expectation.

## Files in this skill

- `references/areas/`: one selection guide per area (`_template.md` for new areas).
- `references/promotion-checklist.md`: the steps for adding a routine or an area.
- `references/troubleshooting.md`: symptoms, likely causes, fixes.
