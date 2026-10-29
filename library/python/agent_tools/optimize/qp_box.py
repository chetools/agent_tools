"""Box-constrained convex quadratic program, primal active-set method (Python reference of Rust `optimize::qp_box`).

Problem: minimize f(x) = 0.5 x^T H x + g^T x  subject to  lo <= x <= hi,  H symmetric positive definite.

Source: Nocedal & Wright, "Numerical Optimization", 2nd ed. (2006), Algorithm 16.3 (Active-Set Method for
Convex QP, Section 16.5), specialised to bound constraints: the working set is a set of variables held
at a bound, so the equality-constrained subproblem is an unconstrained solve on the free variables, the
blocking-constraint step length is the book's min(1, ...) ratio test, and the multiplier test is the
book's sign test (a bound is optimal if its multiplier is >= 0; the multiplier of a bound is the
sign-adjusted gradient component q_i = (Hx + g)_i). Equation numbers were not checked against the book.
Ported from chetools/muse `twin-cstr/solver/src/qp.rs` (same authors); the Cholesky factorization is the
library routine `linalg.cholesky`. Reference for tests: scipy.optimize.minimize(method="L-BFGS-B") and exact
enumeration of active sets. Rust twin: library/rust/src/optimize/qp_box.rs.

Algorithm (one pass of the loop = one iteration `nit`; exactly the steps of the Rust twin):
0. Factor H once (probe); fail with "not positive definite" if that fails. Start at x = clamp(0, lo, hi)
   with every variable at a bound (x_i <= lo_i or x_i >= hi_i) in the working set. A variable with
   lo_i == hi_i is fixed: it is in the working set for good.
1. q = H x + g. Free set F = variables not in the working set. Solve H_FF p_F = -q_F by Cholesky
   (one factorization per pass, none right after a full step), p = 0 elsewhere.
2. If max|p| <= tol_step: x is the minimizer on the free set. Multiplier of a variable at its lower
   bound is lam_i = q_i, at its upper bound lam_i = -q_i (a bound is optimal if lam_i >= 0). If
   no non-fixed bound has lam_i < -tol_dual stop (success); otherwise release the bound with the most
   negative multiplier and go to 1.
3. Else step x += alpha p, alpha = min(1, ratios to the bounds of the free variables); the first
   bound reached (lowest index on ties, as the first strictly smaller ratio wins) joins the working set and
   x_i is set exactly to it. After a full step (alpha = 1, nothing blocks) x is the free-set minimizer by
   construction, so the next pass takes p = 0 without a new solve and goes straight to the multiplier test of step 2.

In exact arithmetic the method terminates finitely when no blocking step has length zero; the pass count is
capped at max_iter (default 4 n) to guard against cycling in degenerate cases and against rounding (a standard
result for nondegenerate problems; no test case needed more than 85 iterations at n = 124).

Changes from the muse source (each is a test below):
- the diagonal shift 1e-10 * max(1, max H_ii) applied when H fails Cholesky is specific to its caller and is NOT
  ported: H not positive definite is a reported failure;
- after a full step the next pass takes p = 0 instead of re-solving. The source re-solves and needs
  max|p| <= tol_step, which rounding denies when cond(H) |x| eps > tol_step; it then runs into its cap and fails
  (n = 2, cond 6e6: test_ill_conditioned_free_set_does_not_stall);
- a fixed variable (lo == hi) is never released (the source releases one whose multiplier has the "wrong" sign and
  re-adds it later: 6 iterations instead of 2 on fixed_variable_n5);
- the blocking variable is set exactly to its bound and free variables are clamped to [lo, hi] after each step
  (feasibility holds exactly);
- the result carries the active set, multipliers, objective and counts instead of Option<x>; only the lower
  triangle of H is read, everywhere (as in `cholesky`); the cap and tolerances are options.

Units: x, lo, hi in units of x; H in units of f / x^2; g in f / x; f in units of f; multipliers in f / x
(>= 0). tol_step is in units of x (absolute), tol_dual in f / x (absolute): rescale the problem if
its natural size is far from 1.

Semidefinite H is rejected (success False, x = clamp(0)): the up-front factorization fails; add a small
diagonal term yourself, or use a method for convex QP that handles singular H. Infinite bounds
(+-inf, any mix) are allowed: a variable with both bounds infinite is always free, and an unbounded-below
direction cannot occur for H positive definite. NaN bounds, lo > hi and non-finite H or g are invalid input
(success False, nfact 0).
"""
from dataclasses import dataclass

import numpy as np

from agent_tools.linalg.cholesky import cholesky, cholesky_solve

TOL_STEP = 1e-10   # default tol_step (units of x, absolute)
TOL_DUAL = 1e-9    # default tol_dual (units of f / x, absolute)


@dataclass
class QpBoxOptions:
    """Options of `qp_box`.

    max_iter: cap on active-set iterations (passes); None means 4 n (the source's cap). Must be >= 1 if given.
    tol_step: the free-set subproblem counts as solved when max|p| <= tol_step, units of x, absolute (default 1e-10).
    tol_dual: a bound is released only if its multiplier is below -tol_dual, units of f / x, absolute (default 1e-9).
    Both tolerances must be finite and >= 0.
    """
    max_iter: int = None
    tol_step: float = TOL_STEP
    tol_dual: float = TOL_DUAL


@dataclass
class QpBoxResult:
    """Result of `qp_box`.

    x: solution (units of x); the last iterate (feasible) when success is False.
    f: objective 0.5 x^T H x + g^T x at x (units of f).
    active: int array (n,): -1 variable at its lower bound, +1 at its upper bound, 0 free. A fixed variable
        (lo == hi) is -1 if its multiplier is >= 0, else +1.
    multipliers: float array (n,), >= 0 for a bound-active variable, in units of f / x; 0 for a free variable.
        Multiplier of the lower bound is q_i = (Hx + g)_i, of the upper bound -q_i. A fixed variable
        reports |q_i| (side chosen by its sign). An active variable with multiplier 0 is a degenerate
        (weakly active) bound.
    success: True if the KKT conditions hold to the tolerances.
    message: "converged" or the reason for failure (invalid input, not positive definite, max_iter reached).
    nit: active-set iterations (passes of the loop, including the last, certifying one).
    nfact: Cholesky factorizations (the up-front probe of H plus one per pass with a non-empty free set, except the
        pass right after a full step).
    """
    x: np.ndarray
    f: float
    active: np.ndarray
    multipliers: np.ndarray
    success: bool
    message: str
    nit: int
    nfact: int


def _symmetric_matvec(h, x):
    """H x reading only the lower triangle of h, accumulating left to right (same order as Rust)."""
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += (h[i, j] if j <= i else h[j, i]) * x[j]
        y[i] = s
    return y


def _objective(h, g, x):
    hx = _symmetric_matvec(h, x)
    f = 0.0
    for i in range(len(x)):
        f += 0.5 * x[i] * hx[i] + g[i] * x[i]
    return f


def qp_box(h, g, lo, hi, options=None):
    """Minimize 0.5 x^T H x + g^T x subject to lo <= x <= hi by the primal active-set method.

    h: (n, n) array, symmetric positive definite, units f / x^2 (lower triangle read). g: (n,), f / x.
    lo, hi: (n,), units of x; may be -inf / +inf; lo <= hi. Returns a `QpBoxResult`.
    """
    opt = options or QpBoxOptions()
    h = np.array(h, dtype=float)
    g = np.array(g, dtype=float)
    lo = np.array(lo, dtype=float)
    hi = np.array(hi, dtype=float)
    n = len(g)
    max_iter = 4 * n if opt.max_iter is None else opt.max_iter

    def fail(msg, x=None, nit=0, nfact=0, active=None, mult=None):
        x = np.zeros(n) if x is None else x
        return QpBoxResult(x, float("nan") if nfact == 0 and nit == 0 else _objective(h, g, x),
                           np.zeros(n, dtype=int) if active is None else active,
                           np.zeros(n) if mult is None else mult, False, msg, nit, nfact)

    if n == 0 or h.shape != (n, n) or len(lo) != n or len(hi) != n:
        return fail("invalid input: need n >= 1, h n x n, g, lo, hi of length n")
    if not (np.all(np.isfinite(h)) and np.all(np.isfinite(g))):
        return fail("invalid input: h or g not finite")
    if np.any(np.isnan(lo)) or np.any(np.isnan(hi)) or np.any(lo > hi) or np.any(lo == np.inf) or np.any(hi == -np.inf):
        return fail("invalid input: bounds must satisfy lo <= hi, be finite or infinite of the right sign, not NaN")
    if not (max_iter >= 1 and np.isfinite(opt.tol_step) and opt.tol_step >= 0
            and np.isfinite(opt.tol_dual) and opt.tol_dual >= 0):
        return fail("invalid option: max_iter >= 1, tol_step and tol_dual finite and >= 0")

    x = np.minimum(np.maximum(0.0, lo), hi)
    probe = cholesky(h)
    nfact = 1
    if not probe.success:
        r = fail("h is not positive definite (" + probe.message + ")", x, 0, nfact)
        return r
    fixed = lo == hi
    at_lo = np.array([x[i] <= lo[i] for i in range(n)])
    at_hi = np.array([x[i] >= hi[i] for i in range(n)])
    nit = 0
    solved = False   # True right after a full step: x is then the free-set minimizer, p = 0 without a new solve
    while nit < max_iter:
        nit += 1
        q = _symmetric_matvec(h, x) + g
        free = [i for i in range(n) if not at_lo[i] and not at_hi[i]]
        p = np.zeros(n)
        if free and not solved:
            nf = len(free)
            hff = np.empty((nf, nf))
            for a in range(nf):
                for b in range(nf):
                    i, j = free[a], free[b]
                    hff[a, b] = h[i, j] if j <= i else h[j, i]
            c = cholesky(hff)
            nfact += 1
            if not c.success:     # cannot happen for H positive definite (principal submatrix)
                return fail("free-set submatrix not positive definite (" + c.message + ")", x, nit, nfact,
                            _active(at_lo, at_hi), np.zeros(n))
            pf = cholesky_solve(c.l, np.array([-q[i] for i in free]))
            for a, i in enumerate(free):
                p[i] = pf[a]
        solved = False
        pmax = 0.0
        for v in p:
            pmax = max(pmax, abs(v))
        if pmax <= opt.tol_step:
            drop, worst = None, opt.tol_dual
            for i in range(n):
                if fixed[i]:
                    continue
                if at_lo[i] and -q[i] > worst:
                    worst, drop = -q[i], i
                if at_hi[i] and q[i] > worst:
                    worst, drop = q[i], i
            if drop is None:
                active, mult = _active(at_lo, at_hi), np.zeros(n)
                for i in range(n):
                    if at_lo[i] or at_hi[i]:
                        if fixed[i]:
                            active[i] = -1 if q[i] >= 0 else 1
                            mult[i] = abs(q[i])
                        else:
                            mult[i] = q[i] if at_lo[i] else -q[i]
                            mult[i] = max(mult[i], 0.0)      # in [-tol_dual, 0) roundoff: report as 0
                return QpBoxResult(x, _objective(h, g, x), active, mult, True, "converged", nit, nfact)
            at_lo[drop] = False
            at_hi[drop] = False
            continue
        alpha, blocker = 1.0, None
        for i in free:
            if p[i] > 0.0:
                a = (hi[i] - x[i]) / p[i]
                if a < alpha:
                    alpha, blocker = a, (i, True)
            elif p[i] < 0.0:
                a = (lo[i] - x[i]) / p[i]
                if a < alpha:
                    alpha, blocker = a, (i, False)
        for i in free:
            x[i] = min(max(x[i] + alpha * p[i], lo[i]), hi[i])
        solved = blocker is None
        if blocker is not None:
            i, upper = blocker
            if upper:
                at_hi[i] = True
                x[i] = hi[i]
            else:
                at_lo[i] = True
                x[i] = lo[i]
    return fail("max_iter reached", x, nit, nfact, _active(at_lo, at_hi), np.zeros(n))


def _active(at_lo, at_hi):
    return np.array([-1 if at_lo[i] else (1 if at_hi[i] else 0) for i in range(len(at_lo))], dtype=int)
