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

### Implemented ODE routines

| Situation | Routine | SciPy equivalent | Notes |
|---|---|---|---|
| Stiff IVP, small n, dense Jacobian available, rtol about 1e-3 to 1e-6; or derivatives of the result with respect to parameters through the adaptive steps (`Dual` state) | `rosenbrock23` (ode23s, 2nd order, Shampine and Reichelt 1997) | none in `solve_ivp`; compare `Radau`, `BDF` | One LU of `I - h d J` per attempted step, no Newton iteration. Measured (see catalog): at the same rtol it needs many more steps than Radau or BDF (Robertson to 1e5, rtol 1e-7: 9105 steps vs 372 and 536), so prefer Radau for tight tolerances when no AD is needed. Achieved error is 2 to 150 times rtol. Needs `df/dt` (`dfdt` closure or `fd_dfdt`) for time-dependent `f`. Stores t, y and dy/dt at accepted steps for Hermite dense output. No event detection. |

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
| Smooth, unconstrained, gradient available, small to moderate n (tens) | BFGS: `bfgs` | `BFGS` | Armijo backtracking only (no Wolfe curvature condition), identity start, dense `O(n^2)` inverse Hessian. Stops on max-norm gradient < `gtol` (default 1e-5). Tested to `gtol` = 1e-10; needed 1.25 to 1.4 times SciPy's function evaluations on the Rosenbrock cases 2.4 times on a 5-D quadratic, and 2.9 times on an 8-D quadratic with eigenvalues 1 to 1e3 (a reviewer measured 5 times on another one; see catalog). No bounds: transform to unbounded variables. |
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
| Symmetric positive definite system, Gaussian-process covariances, SPD test | `cholesky` (+ `solve_lower`, `solve_upper_t`, `cholesky_solve`) | `numpy.linalg.cholesky`, `scipy.linalg.cho_solve` | Fails (`success = false`) on non-SPD input, which doubles as an SPD test. f64 only. About half the cost of LU. |
| Block-tridiagonal system (banded Jacobians of 1D discretisations with `m` coupled unknowns per node), many right-hand sides per factorization; also with `Dual` entries | `block_tridiagonal` (`block_tridiag_factor`, `block_tridiag_solve`) | `numpy.linalg.solve` on the assembled matrix; `scipy.linalg.solve_banded` with bandwidth `2m-1` | Block Thomas, O(nb m^3) factor, O(nb m^2) per solve. Pivots only inside diagonal blocks, never across blocks: needs block diagonal dominance, SPD or M-matrix structure. Exactly singular diagonal block gives `success = false`; near-singular is not detected, watch `growth`. Generic over `Real`. |
