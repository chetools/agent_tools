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

## Linear algebra (`lu_dense`, `cholesky`)

| Symptom | Likely cause | Fix |
|---|---|---|
| `lu_dense` reports singular | Exact zero pivot (rank deficiency, a zero row or column) | Check the matrix construction; rescale. |
| `lu_dense` succeeds but the solution is huge or the residual is large | Nearly singular matrix (not detected) | Compute the residual `A x - b`; estimate the condition number; expect a relative error up to about `eps * cond`. |
| `cholesky` fails with "not positive definite" | Matrix is not SPD, or only semidefinite, or too ill-conditioned in f64 | Check symmetry and eigenvalues; add a small diagonal jitter (for Gaussian-process covariances, a nugget) deliberately. |
| `cholesky` result differs from NumPy at the 1e-10 level | Ill-conditioned matrix (for example Hilbert n = 10): errors scale with `eps * cond` | Expected; compare the residual `L L^T - A` instead. |

## Random numbers (`splitmix64`, `mulberry32`)

| Symptom | Likely cause | Fix |
|---|---|---|
| Rust and Python (or JavaScript) streams differ | Different constructor or seed (SplitMix64 `new` vs `from_seed_stream`; Mulberry32 seed masked to 32 bits), or a different number of draws consumed | See `areas/random.md`; compare the first outputs against `reference_data/`. |
| Normals differ in the last digits between languages | libm rounding in `ln` and `cos` | Expected, about 1e-15 relative; integer outputs and uniforms are exact. |
| Visible pattern or poor quality in a large Monte Carlo run with Mulberry32 | 32-bit state, modest quality | Use SplitMix64. |

## Optimization (`bfgs`)

| Symptom | Likely cause | Fix |
|---|---|---|
| `success = false`, "line search failed" | Wrong gradient (sign or scale), discontinuous or noisy objective, or `f` at the limit of rounding near the minimum | Check the gradient against central differences on both sides of the point; loosen `gtol` if `f` is already flat to rounding. |
| `success = false`, "max_iter reached" | Badly scaled variables, a narrow curved valley, or too small a limit | Rescale variables to O(1); raise `max_iter`; the identity start makes the first steps short or long by the gradient size. |
| Many more evaluations than `scipy.optimize.minimize(method='BFGS')` | Halving-only Armijo line search, no Wolfe curvature condition or interpolation | Expected (1.3 to 2.4 times SciPy's count on the four original catalog cases, 2.9 times (50 vs 17) on the 8-D quadratic with eigenvalues 1 to 1e3, and 5 times, 80 vs 16, measured by a reviewer on another such quadratic); use SciPy where evaluations are expensive. |
| Inverse-Hessian update silently skipped | Curvature `s.y <= 1e-12` (absolute threshold) | Rescale `x` and `f` so curvature is O(1). |
| Result violates a bound | `bfgs` is unconstrained | Optimize transformed unbounded variables, scaling the gradient by the chain rule. |
| `success = false`, "no progress" | `gtol` is below what rounding allows at this `x` and `f` (`f` flat to rounding, gradient at rounding level): a step left `x` unchanged, or `max|g|` reached no new low in 20 steps | The returned `x` is the last accepted point (at rounding level); raise `gtol` (`grad_norm` shows what was attainable) or rescale `x` and `f`. |
| `success = false`, "invalid option" and `nfev = 0` | `gtol` not finite or not > 0, or `max_iter` < 1 | Pass a positive finite `gtol` and `max_iter >= 1`. |

## Gaussian processes and design (`gp_matern52`, `imse_batch`)

| Symptom | Likely cause | Fix |
|---|---|---|
| `gp_fit` reports `success = false`, "line search failed", but the NLL agrees with other starts | Ill-conditioned covariance (near-duplicate inputs, tiny noise): the NLL is flat to rounding at the optimum | Expected; compare `start_nll` across starts. Fix the noise to a realistic value or remove duplicates if it matters. |
| Fitted length scale at about 0.02 or 5 (or noise at 1e-3 std(y)) | Hyperparameter on its bound: inputs not scaled to [0, 1], or the data do not determine it | Scale inputs, widen `log_length_bounds`, or fix the noise. |
| One start ends with a much larger NLL (for example 45 vs 9.75) | Local optimum of the marginal likelihood | Expected; the best start is kept. Add `n_random_starts` for more coverage. |
| `gp_build` returns an error / raises "not positive definite" | Duplicate inputs with `noise_sd = 0` | Use a positive `noise_sd`. |
| IMSE batch picks differ between Rust and Python | Two candidates with nearly equal scores (symmetric grids or data) | Compare `first_scores`; ties go to the first index. |
