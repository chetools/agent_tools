# Troubleshooting

General symptoms and fixes from standard practice. Add routine-specific entries as they are found, with the routine name and the test that exposed them. Area-specific pitfalls may instead live in the area guide.

**First step in every case:** reproduce the problem with the Python reference implementation (and the established library it mirrors, if available) before changing code. If the reference agrees, suspect the problem setup.

## Any routine

| Symptom | Likely cause | Fix |
|---|---|---|
| Rust and Python results differ | Different defaults, or floating-point order of operations | Compare defaults first; differences within the stated tolerance are expected. |
| Results off by a constant factor | Unit mismatch | Check the documented units of every input and output. |
| Values wrong outside some range | Correlation or fit used beyond its valid range | Check the valid range in the catalog entry; do not extrapolate silently. |
| Results change from run to run | Unseeded randomness | Pass an explicit seed. |

## ODE integrators

| Symptom | Likely cause | Fix |
|---|---|---|
| Explicit method takes huge numbers of tiny steps | Problem is stiff | Switch to Radau or BDF. |
| Implicit method: repeated Newton failures, step size collapses | Wrong or stale Jacobian; poor scaling | Check the analytic Jacobian against finite differences; refresh it more often; set `atol` per component to the variable's scale. |
| Negative concentrations or other nonphysical values | Tolerances too loose for small components | Tighten `atol` for those components. Do not clip values inside the right-hand side. |
| Solution drifts at long times | Accumulated error | Tighten `rtol`; use a higher-order method; check conservation laws as a diagnostic. |
| Results differ from SciPy beyond tolerance | Different defaults (initial step, safety factor, error norm) | Compare defaults with the Python reference first. Differences within the requested tolerance are expected. |
| Event missed | Sign change happens within a single step | Use dense-output root finding for events; limit the maximum step. |

## Root solvers

| Symptom | Likely cause | Fix |
|---|---|---|
| Newton diverges or oscillates | Starting point too far away | Add line search or trust region; use continuation in a parameter. |
| Converges to a nonphysical root | Several roots exist | Bracket the physical one; use Brent for scalar problems. |
| Stalls with a small step but a large residual | Singular or ill-conditioned Jacobian | Rescale variables and equations; try Levenberg–Marquardt. |

## Optimizers

| Symptom | Likely cause | Fix |
|---|---|---|
| Line search fails | Gradient wrong or not consistent with objective | Check the gradient against finite differences. |
| Converges slowly along a curved valley | Poor scaling or conditioning | Rescale variables; use a quasi-Newton method. |
| Different runs end at different points | Multiple local minima | Multi-start, or a global method, chosen deliberately. |
| Constraints violated at the end | Infeasible start or loose constraint tolerance | Start from a feasible point; tighten the constraint tolerance. |
