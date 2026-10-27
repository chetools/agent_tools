# Area guide: ode, roots, optimize, linalg

This is general guidance from standard numerical-analysis practice. It is not a set of measured results for this library. Once routines exist, their `library/catalog.toml` entries record what each one has actually been tested on. Where the two disagree, the catalog wins.

"Planned" means the routine is not in the library yet; check `catalog.toml` for current status.

## Initial value problems (SciPy `solve_ivp`)

| Situation | Method | SciPy equivalent | Notes |
|---|---|---|---|
| Nonstiff, moderate accuracy | Dormand–Prince 5(4) | `RK45` | Good default when stiffness is not expected. |
| Nonstiff, tight tolerances (rtol ≲ 1e-8) | Dormand–Prince 8(5,3) | `DOP853` | Fewer steps than RK45 at tight tolerances. |
| Stiff, or stiffness unknown; also implicit-DAE-like systems | Radau IIA, order 5 | `Radau` | Robust (L-stable). Needs a Jacobian (analytic preferred, finite-difference fallback). Each step solves a 3·n implicit system. |
| Stiff, large n, moderate accuracy | Variable-order BDF (1–5) | `BDF` | Cheaper per step than Radau for large systems. Less robust with strongly oscillatory or highly stiff components. |
| Stiff, small n, cheap Jacobian | Rosenbrock (linearly implicit) | (none in `solve_ivp`) | No Newton iteration. Sensitive to Jacobian accuracy. |
| Stiffness switches during the run | Radau, or switching LSODA-style | `LSODA` | Prefer Radau unless the cost of the Jacobian dominates. |

Signs of stiffness: an explicit method takes very many tiny steps while the solution varies slowly; fast transients decay from widely separated time scales (e.g. chemical kinetics, reactor thermal runaway).

Event detection (zero crossings) and dense output are features of the integrator interface. Record in the catalog which routines support them.

## Scalar roots (SciPy `brentq`, `root_scalar`)

| Situation | Method | Notes |
|---|---|---|
| Sign-change bracket available | Brent | Guaranteed convergence, superlinear in practice. Default choice. |
| Good initial guess, cheap derivative | Newton with bracket safeguard | Fall back to bisection when a step leaves the bracket. |
| No derivative, no bracket | Secant | Can diverge; use only with a reasonable initial guess. |

## Systems of nonlinear equations (SciPy `root`, `fsolve`)

| Situation | Method | SciPy equivalent | Notes |
|---|---|---|---|
| General square system, default | Powell hybrid (dogleg trust region) | `root(method='hybr')` / `fsolve` | MINPACK `hybrd`/`hybrj`. Robust from moderate starting points. |
| Good initial guess, Jacobian available | Newton with line search (damped Newton) | — | Fast near the solution. Line search guards against overshoot. |
| Over-determined or near-singular Jacobian | Levenberg–Marquardt | `root(method='lm')` | Least-squares form. |
| Hard to converge | Continuation / homotopy over a parameter | — | Use the previous solution as the next initial guess. |

## Optimization (SciPy `minimize`)

| Situation | Method | SciPy equivalent | Notes |
|---|---|---|---|
| Smooth, unconstrained, gradient available | BFGS | `BFGS` | Needs a line search satisfying Wolfe conditions. |
| Smooth, many variables, or simple bounds | L-BFGS-B | `L-BFGS-B` | Limited memory; handles box bounds. |
| Noisy or non-differentiable objective, few variables | Nelder–Mead | `Nelder-Mead` | Derivative-free. Slow in high dimensions. |
| Equality and inequality constraints | SQP | `SLSQP` | Needs gradients of objective and constraints. |
| Least squares (model fitting) | Levenberg–Marquardt / trust-region reflective | `least_squares` | Exploit residual structure instead of minimizing the sum of squares directly. |
| Scalar, bracketed minimum | Brent's minimizer / golden section | `minimize_scalar` | |

## Supporting linear algebra

Implicit integrators and Newton-type solvers need dense LU with partial pivoting. Radau IIA additionally needs the complex LU used for its transformed system. Small-system linear algebra is kept in the crate rather than added as an external dependency.

| Situation | Routine | SciPy / NumPy equivalent | Notes |
|---|---|---|---|
| Small dense general system `A x = b` (Newton steps, implicit integrators); also with `Dual` entries for sensitivities | `lu_dense` | `scipy.linalg.lu_factor` / `lu_solve`, `numpy.linalg.solve` | Partial pivoting on values. Reports exact zero or non-finite pivots only; check the residual for near-singular systems. Generic over `Real`. |
