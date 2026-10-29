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

## Process engineering (`lmtd`, `hx_capital_cost`, `pinch_targets`)

| Symptom | Likely cause | Fix |
|---|---|---|
| `lmtd` returns NaN | A terminal difference is <= 0 (temperature cross) or not finite | Treat as infeasible; check stream temperatures and flow arrangement. |
| LMTD derivative differs from a finite difference at the 1e-8 level near equal terminal differences | The analytic derivative is accurate to about 3e-12 relative (measured against 50-digit values for \|d\|/m from 1e-9 to 2); the finite difference is the less accurate side (cancellation in the difference quotient) | Trust the analytic derivative; use a central difference with a step of about 1e-4 of the value. |
| `hx_capital_cost` has zero gradient for small exchangers | Area below the correlation range: the cost of the lower limit is charged (`clamped_below`) | Bound the area from below at the limit, or use a different cost law. |
| Seider and IDAES/Turton costs differ by a factor | Different cost-index years (Seider CE 567 here, IDAES CE 500, Turton CEPCI 397) | Escalate with index_new / index_basis before comparing. |
| `pinch_targets` pinch is at the highest or lowest temperature | Threshold problem (only hot or only cold utility needed) | No true pinch; read the targets and `heat_flow`. |
| `pinch_targets` targets differ from a hand calculation by a shift | Hot/cold temperature shift convention (hot scale here: cold + ΔTmin) | Compare heat flows, not shifted temperatures, or use `t_shifted`. |

## Porous-media correlations (`porous_permeability`)

| Symptom | Likely cause | Fix |
|---|---|---|
| Result is NaN | An input is outside the domain: porosity not in (0,1), solid fraction above close packing (pi/4 square, 0.9069 hexagonal), negative velocity, non-positive diameter or `a_s` | Check units (diameter in m, not mm or radius) and the porosity/solid-fraction convention (eps = 1 - phi). |
| Kozeny-Carman permeability off by a factor of order 1-10 | `a_s` taken per bed volume, wrong shape factor (6/d spheres, 4/d cylinders), or K not matched to the bed | Use surface per solid volume; fit K with `fit_kozeny_constant`. |
| Gebart far from data at low solid fraction, or Sangani-Acrivos far at high | Outside the recommended range (Gebart: close packing; S-A: dilute) | Choose the correlation for the regime; compare both near phi 0.3-0.5 and treat the gap as uncertainty. |

## Residence-time analysis (`rtd_analysis`)

| Symptom | Likely cause | Fix |
|---|---|---|
| Variance or N changes strongly with the record length | Tail cut off, or no tail applied (`tail_decay_time` is None) | Record until C is below about 1 % of peak; check the decaying tail is positive; use `fit_lsq`. |
| Variance of a noisy record is too large (N too small) | Noise in the tail is clipped to positive values and adds area and spread; the tail fit is skipped when any tail sample is <= 0 | Subtract the baseline, truncate where the signal reaches the noise, or use `fit_lsq`. |
| `fit_moments` fails for closed-closed | Variance/mean^2 is >= 1 (closed-closed maximum) | Use open-open or tanks, or check the record. |
| `fit_lsq` reports "parameter at the edge of its search range" | N or Pe outside [0.05, 1e4], or tau outside 0.2-5 times the data mean, or a multi-peak signal | Fix tau (pass `tau`), rescale, or inspect the data; the search is a unimodal golden section. |
| Pe from `fit_dispersion_planes` is negative or huge | Variance or mean did not increase between the planes | Use the same pulse and tail handling at both planes; check the plane order. |

## Design of experiments (`lhs_maximin`)

| Symptom | Likely cause | Fix |
|---|---|---|
| Design points are not at slice centres | By design: each point sits at a uniformly random position inside its slice | Use `(slice + 0.5) / n` yourself if centred points are wanted. |
| Min distance is small in high dimension (d = 10, n = 50: about 0.5 to 0.6) | Best-of-k random search is weak in high dimension; gain per extra candidate shrinks | Raise `tries` (cost is linear) or use an optimizing search (exchange / annealing, Morris & Mitchell 1995). |
| `success == false`, empty points | `n`, `d` or `tries` equal to 0 | Pass values of at least 1. |
| Rust and Python designs differ | Different generator constructor or seed, or a different draw order before the call | Build both with `from_seed_stream(seed, stream)` and consume nothing from the generator beforehand. |

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

## ODE integrators (`dopri5`)

| Symptom | Likely cause | Fix |
|---|---|---|
| `nsteps` grows like the stiffness (for example 6074 steps for `y' = -1e4 (y - cos t)` at rtol 1e-6) although the solution is smooth | Stiffness: the explicit stability limit, not accuracy, sets the step | Use `rosenbrock23` or a Radau/BDF solver; loosening rtol does not help. |
| Failure "step size underflow" or "max_steps" | Finite-time blow-up, a discontinuity, `f` returning NaN, or a stiff problem with a small `max_steps` | Check `f` for NaN/inf, shorten the span before a singularity, raise `max_steps` only for a stiff or long run. |
| Failure "automatic first step is not a positive finite number" | `atol = 0` with a zero component in `y0`, or `f(t0, y0)` is not finite | Give a positive `atol`, or pass `h0`. |
| Final error is 100 to 10000 times larger than rtol on an orbit | Local, not global, error control; periodic orbits do not damp errors | Request a smaller rtol (try 1e-9 for Arenstorf-type orbits) and compare with a tighter run. |
| A component near zero is badly resolved | `atol` is too loose for it (the scale is `atol + rtol * abs(y)`) | Pass `atol_vec` with a small entry for that component. |
| Step sequence differs from SciPy `RK45` in the last digits | Summation order differs, and the controller feeds the round-off back | Expected: step counts and `nfev` are equal, times agree to about 1e-8 on the tested cases. |

## Step Jacobian of a fixed-step method (`rk4_step_jacobian`)

| Symptom | Likely cause | Fix |
|---|---|---|
| `phi` differs from finite differences of the step by far more than 1e-6 | `jac` is not the derivative of `f` (or is evaluated at different arguments) | Check `jac` against central differences of `f` first; `phi` is only as exact as `jac`. |
| `phi` differs from the flow sensitivity matrix `dy(T)/dy(0)` | Expected: `phi` is the derivative of the discrete map, which differs from the flow's by the truncation error | Use more substeps (`n_sub`); the difference falls like `(h / n_sub)^4`. |
| `y1` and `phi` are huge or NaN | Explicit RK4 is unstable when `h * lambda` is outside its stability region (about -2.78 on the real axis), e.g. a stiff or ignited reactor | Increase `n_sub`, or use a stiff integrator (`rosenbrock23`). |
| `Dual` derivatives are wrong or zero | `jac` returns plain floats while the state is `Dual` (the Jacobian is then wrong), or no input was seeded | Build `f` and `jac` from the same generic code; seed one `Dual` derivative per column with `rk4_step`. |
| Result says success = false | `n = 0`, `n_sub = 0` or non-finite `t0` or `h` | Fix the arguments. |
