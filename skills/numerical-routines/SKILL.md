---
name: numerical-routines
description: Use when code needs an ODE integrator, root solver, optimizer or small dense linear solver (e.g. replacing SciPy solve_ivp, root, minimize, brentq in Rust, WASM or standalone pages). Finds a verified routine in chetools/agent_tools or adds a missing one.
---

# Numerical routines library (chetools/agent_tools)

This skill points to a shared library of numerical routines. Each routine exists as a Rust implementation, a matching Python reference, and stored reference results. The rules in `AGENTS.md` at the repository root are binding. This file is the working summary.

## Workflow

1. **Identify the problem class.** Is it an initial value problem, stiff or not? A scalar root, or a system of equations? Unconstrained, bounded or constrained optimization? See `references/method-selection.md`.
2. **Look it up** in `numerics/catalog.toml`. Match on `problem_class` and `suitable_for`, and check `status`.
3. **Use it** if it fits. Rust and wasm projects depend on the crate (`numerics/rust/`). Python and teaching pages use `numerics/python/`. Start from the routine's file in `numerics/examples/`.
4. **If it is missing,** write it for the task, then promote it, following `references/promotion-checklist.md`. Without push access to this repo, use the `library-candidates/` fallback described in `AGENTS.md`.
5. **If it misbehaves,** use `references/troubleshooting.md`. Reproduce with the Python twin before changing code.

## Non-negotiables

- Do not rewrite a routine that is in the catalog with `status = "verified"`.
- Never set `status = "verified"` yourself. New routines start as `experimental`; only a maintainer's explicit approval of that pull request makes them verified. You may merge on that explicit instruction, after posting the comment required by `AGENTS.md` section 5.
- Change routines here through a pull request, never by patching a downstream copy.
- Report numbers (errors against reference, evaluation counts), not impressions. Label anything unchecked as an expectation.

## Files in this skill

- `references/method-selection.md`: which method for which problem, with SciPy equivalents.
- `references/promotion-checklist.md`: the steps for adding a routine.
- `references/troubleshooting.md`: symptoms, likely causes, fixes.
