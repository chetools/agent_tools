"""BFGS quasi-Newton minimization with an Armijo backtracking line search (unconstrained).

Python reference of the Rust `optimize::bfgs` (same algorithm, defaults and stopping test).
Source: Nocedal & Wright, "Numerical Optimization", 2nd ed. (2006): Algorithm 6.1 (BFGS), eq. (6.17)
(inverse-Hessian update H <- (I - r s y^T) H (I - r y s^T) + r s s^T, r = 1/(y^T s)) and
Algorithm 3.1 (backtracking with the sufficient-decrease condition, eq. 3.4, c1 = 1e-4).
Ported from chetools/claude `sequential-design/python/sdl.py` (`bfgs`; same authors), with the
Gaussian-process likelihood left out. Reference for tests: scipy.optimize.minimize(method="BFGS").
Rust twin: library/rust/src/optimize/bfgs.rs.

The source has no bound handling (its Gaussian-process caller maps bounded hyperparameters to
unbounded ones with a tanh transform; that transform is problem-specific and not part of this
routine). To respect bounds, minimize over transformed unbounded variables and multiply the
gradient by the chain-rule factor.

Per iteration:
1. Stop with success if max|g| < gtol (strict).
2. Direction p = -H g; if g.p >= 0 (not a descent direction) reset H = I and use p = -g.
3. Backtrack a = 1, 1/2, 1/4, ... until f(x + a p) <= f(x) + C1 a g.p; fail if a < MIN_STEP.
4. Update H unless the curvature s.y <= SY_MIN; accept the step.

H starts as the identity (no scaling), so the first step has length |g|. Cost O(n^2) memory and
O(n^3) work per iteration: intended for small to moderate n (tens).

The inner products and matrix products are written as plain loops that accumulate left to right,
in exactly the order of the Rust code, so both languages follow the same iterates to rounding
(NumPy's `@` and `np.dot` sum in another order and the two paths would drift apart near the end).
"""
from dataclasses import dataclass

import numpy as np

C1 = 1e-4        # sufficient-decrease constant of the Armijo condition (dimensionless)
MIN_STEP = 1e-10  # smallest accepted line-search step length (dimensionless)
SY_MIN = 1e-12   # H update skipped if s.y <= SY_MIN; absolute, in (units of x)*(units of g)


@dataclass
class BfgsOptions:
    """Options of `bfgs`.

    max_iter: maximum number of accepted steps (default 200).
    gtol: stop when max|gradient| < gtol, in units of f per unit of x (default 1e-5, positive).
    """
    max_iter: int = 200
    gtol: float = 1e-5


@dataclass
class BfgsResult:
    """Result of `bfgs`.

    x: last accepted point (units of x), also when success is False.
    f: objective value at x (units of f).
    grad_norm: max-norm of the gradient at x (units of f per unit of x); compared with gtol.
    success: True if grad_norm < gtol was reached.
    message: "converged: ..." or the reason for failure (iteration limit, line-search failure,
        non-finite objective or gradient).
    nit: number of accepted steps.
    nfev: number of calls of fg (f and gradient together), including the call at x0 and the
        rejected line-search trial points.
    """
    x: np.ndarray
    f: float
    grad_norm: float
    success: bool
    message: str
    nit: int
    nfev: int


def _max_norm(g):
    m = 0.0
    for v in g:
        if not np.isfinite(v):
            return float("nan")
        m = max(m, abs(v))
    return m


def _dot(a, b):
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s


def bfgs(fg, x0, options=None):
    """Minimize f(x) over x in R^n (unconstrained) with BFGS and Armijo backtracking.

    fg(x, grad) must return f(x) and write the gradient df/dx into the array `grad` in place
    (grad[:] = ...). It may return inf or NaN where f is undefined: that trial point is rejected
    and the step halved (used for a failed Cholesky factorization in Gaussian-process fitting);
    the gradient written for a rejected point is ignored. `x` and `grad` are NumPy arrays that may
    be reused between calls: copy them if you keep them. x0 is the start (units of x),
    n = len(x0) >= 1; options is a `BfgsOptions` (defaults above).

    Failure modes (success False, best point so far returned): non-finite f or gradient at x0 or
    at an accepted point; the line search shrinks the step below MIN_STEP (typical for a wrong
    gradient sign or a discontinuous objective); max_iter steps without convergence.
    """
    opt = options if options is not None else BfgsOptions()
    z = [float(v) for v in x0]
    n = len(z)
    grad = np.zeros(n)
    f = float(fg(np.array(z), grad))
    g = grad.tolist()
    nfev, nit = 1, 0
    gn = _max_norm(g)
    if not (np.isfinite(f) and np.isfinite(gn)):
        return BfgsResult(np.array(z), f, gn, False, "non-finite objective or gradient at x0", nit, nfev)

    def eye():
        return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    H = eye()                                                  # inverse-Hessian estimate
    while True:
        if gn < opt.gtol:
            return BfgsResult(np.array(z), f, gn, True,
                              f"converged: max|grad| = {gn:e} < gtol = {opt.gtol:e}", nit, nfev)
        if nit >= opt.max_iter:
            return BfgsResult(np.array(z), f, gn, False,
                              f"max_iter = {opt.max_iter} reached, max|grad| = {gn:e}", nit, nfev)
        d = [-_dot(H[i], g) for i in range(n)]                 # p = -H g
        gp = _dot(g, d)
        if gp >= 0:                                            # not a descent direction: reset
            H = eye()
            d = [-v for v in g]
            gp = _dot(g, d)
        a = 1.0
        while True:
            z1 = [z[i] + a * d[i] for i in range(n)]
            f1 = float(fg(np.array(z1), grad))
            nfev += 1
            if f1 <= f + C1 * a * gp:
                break
            a *= 0.5
            if a < MIN_STEP:
                return BfgsResult(np.array(z), f, gn, False,
                                  f"line search failed: step below {MIN_STEP:e}", nit, nfev)
        g1 = grad.tolist()
        s = [z1[i] - z[i] for i in range(n)]
        yv = [g1[i] - g[i] for i in range(n)]
        sy = _dot(s, yv)
        if sy > SY_MIN:
            r = 1.0 / sy
            # H <- (I - r s y^T) H (I - r y s^T) + r s s^T
            A1 = [[(1.0 if i == j else 0.0) - r * s[i] * yv[j] for j in range(n)] for i in range(n)]
            T = [[_dot(A1[i], [H[k][j] for k in range(n)]) for j in range(n)] for i in range(n)]
            H = [[_dot(T[i], A1[j]) + r * s[i] * s[j] for j in range(n)] for i in range(n)]
        z, f, g = z1, f1, g1
        gn = _max_norm(g)
        nit += 1
        if not np.isfinite(gn):
            return BfgsResult(np.array(z), f, gn, False, "non-finite gradient at an accepted point", nit, nfev)
