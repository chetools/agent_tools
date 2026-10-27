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

## Automatic differentiation (`dual`)

| Symptom | Likely cause | Fix |
|---|---|---|
| Derivative NaN or infinite (Rust), or `ZeroDivisionError`/`ValueError` (Python) | Evaluation at `sqrt(0)`, `ln(0)` or a zero denominator | Move the point; the function is singular there. |
| AD derivative disagrees with finite differences by about 1e-6 | Finite-difference step too large or too small | Use central differences, h about 1e-5 for second derivatives; check against an analytic form. |
| Derivative of `abs` or a branch has the wrong sign near a kink | The derivative follows the branch taken; at exactly 0 `abs` uses the right-hand derivative | Avoid evaluating at the kink. |
| Second derivative is 0 or wrong | Single `Dual<f64>` used | Nest: `Dual<Dual<f64>>` with seeds `Dual::new(Dual::new(x, 1.0), Dual::new(1.0, 0.0))`. |

## Linear algebra (`lu_dense`, `cholesky`, `block_tridiagonal`)

| Symptom | Likely cause | Fix |
|---|---|---|
| `lu_dense` reports singular | Exact zero pivot (rank deficiency, a zero row or column) | Check the matrix construction; rescale. |
| `lu_dense` succeeds but the solution is huge or the residual is large | Nearly singular matrix (not detected) | Compute the residual `A x - b`; estimate the condition number; expect a relative error up to about `eps * cond`. |
| `cholesky` fails with "not positive definite" | Matrix is not SPD, or only semidefinite, or too ill-conditioned in f64 | Check symmetry and eigenvalues; add a small diagonal jitter (for Gaussian-process covariances, a nugget) deliberately. |
| `cholesky` result differs from NumPy at the 1e-10 level | Ill-conditioned matrix (for example Hilbert n = 10): errors scale with `eps * cond` | Expected; compare the residual `L L^T - A` instead. |
| `block_tridiag_factor` fails with "diagonal block i ... singular" although the full matrix is nonsingular | No pivoting across blocks: the eliminated block `M_i = D_i - L C` is singular (for example a zero diagonal block) | Reorder or merge unknowns so the diagonal blocks dominate, or use a banded/dense solver (`scipy.linalg.solve_banded`, `lu_dense`). |
| `block_tridiag_solve` succeeds but the residual is large, or `growth` is far above 1 | Nearly singular `M_i` (not detected); the matrix is not block diagonally dominant | Compute `A x - b`; compare with a banded or dense solve; reorder unknowns. |

## Random numbers (`splitmix64`, `mulberry32`)

| Symptom | Likely cause | Fix |
|---|---|---|
| Rust and Python (or JavaScript) streams differ | Different constructor or seed (SplitMix64 `new` vs `from_seed_stream`; Mulberry32 seed masked to 32 bits), or a different number of draws consumed | See `areas/random.md`; compare the first outputs against `reference_data/`. |
| Normals differ in the last digits between languages | libm rounding in `ln` and `cos` | Expected, about 1e-15 relative; integer outputs and uniforms are exact. |
| Visible pattern or poor quality in a large Monte Carlo run with Mulberry32 | 32-bit state, modest quality | Use SplitMix64. |

## Interpolation (`cubic_hermite`, `hermite_extremum`)

| Symptom | Likely cause | Fix |
|---|---|---|
| `hermite_extremum` returns `success = false` | Fewer than 2 points, unequal lengths, non-finite entries, or `t` not strictly increasing (for example a rejected-step duplicate time from an integrator) | Drop repeated times, sort, and check for NaN in `y` or `dy`. |
| Peak lies at the first or last sample | The interpolant is monotone there, or the true peak is outside the sampled range | Extend the data; plot `y`; the endpoint is a legitimate answer for monotone data. |
| Peak value changes noticeably when samples are added | Knots too coarse for the feature | Halve the spacing; the peak-value error should drop by about 16. |
| Slopes inconsistent with values (visible kinks or overshoot) | `dy` is not the derivative of `y` (wrong units, or per step instead of per unit of `t`) | `dy` is `dy/dt` in unit of `y` per unit of `t`; check it against finite differences of `y`. |
| `cubic_hermite` gives wild values outside the data | Extrapolation with the end cubic | Clamp `x` to the data range. |

## ODE integrators (`rosenbrock23`)

| Symptom | Likely cause | Fix |
|---|---|---|
| Very many steps (hundreds of thousands) at tight rtol, or `max_steps` reached | Second-order method: steps scale like `rtol^(1/3)` | Loosen rtol, raise `max_steps`, or use a higher-order stiff solver (Radau) when no AD is needed. |
| Time-dependent `f` is integrated with a large error or needs far more steps | `df/dt` taken as zero (the autonomous default) | Pass the `dfdt` closure, or set `fd_dfdt = true`. |
| Dual sensitivities are wrong or zero | The Jacobian closure returns plain values while the state is Dual, or the parameter is not seeded in `jac` | Build `f` and `jac` from the same generic code, so the parameter's derivative part enters both. |
| Failure "W = I - h d J is singular" | `1/(h d)` is an eigenvalue of `J`, or `J` has NaN | Check the Jacobian; the step is not shrunk automatically after this failure. |
| Failure "step size underflow" | Finite-time blow-up, a discontinuity, or an inconsistent Jacobian | Check `f` and `jac` against each other by finite differences; shorten the span before a singularity. |
| Achieved error is 10 to 100 times larger than rtol | Local, not global, error control (expected) | Request an rtol 10 to 100 times smaller than the accuracy you need; check against a tighter run. |
