"""Adaptive Rosenbrock 2(3) integrator for stiff IVPs (the scheme of MATLAB `ode23s`), generic over
floats and `agent_tools.autodiff.dual.Dual`, so dual numbers differentiate through the adaptive steps.

Source: L. F. Shampine and M. W. Reichelt, "The MATLAB ODE Suite", SIAM J. Sci. Comput. 18(1),
1-22 (1997), section 2. Ported from chetools/claude `packed-bed-runaway` (`integrate`), with a dense
caller-supplied Jacobian and the LU from `agent_tools.linalg.lu_dense`. Reference for tests:
scipy.integrate.solve_ivp(method="Radau") at rtol 1e-12, atol 1e-14.
Rust twin: library/rust/src/ode/rosenbrock23.rs (same algorithm and defaults).

One step from (t, y) with step h, W = I - h d J, J = df/dy, T = df/dt, d = 1/(2+sqrt 2),
e32 = 6+sqrt 2 (one LU of W serves all stages, no Newton iteration):

    k1 = W^-1 (f(t,y) + h d T)
    k2 = W^-1 (f(t+h/2, y + h/2 k1) - k1) + k1          y_new = y + h k2     (2nd order)
    k3 = W^-1 (f(t+h, y_new) - e32 (k2 - f1) - 2 (k1 - f0) + h d T)
    err = h/6 (k1 - 2 k2 + k3)                                               (3rd-order estimate)

Error control: err scaled componentwise by atol_i + rtol max(|y_i|, |y_new_i|), max-norm <= 1 accepts;
next step h * clamp(0.8 err**(-1/3), 0.2, 5) (5 if err == 0).

Derivative parts in the error test: step sizes are plain floats, so for Dual states the derivative
part of the result is the exact derivative of the discrete solution ("internal differentiation").
The value test does not control its accuracy, so the same test is applied to the derivative parts
of the error vector, scaled by atol_i + rtol max_j(|dy_j|, |dy_new_j|) (the largest derivative
component: tiny components carry only round-off and must not stall the step). For floats the
derivative part is 0 and this vanishes.

Pure Python: y is a list of floats or Duals, matrices are flat row-major lists (as in lu_dense).
"""
import math
from dataclasses import dataclass, field

from agent_tools.autodiff.dual import deriv, val
from agent_tools.linalg.lu_dense import lu_factor, lu_solve

_D = 1.0 / (2.0 + math.sqrt(2.0))
_E32 = 6.0 + math.sqrt(2.0)


@dataclass
class Rosenbrock23Result:
    """Result of `rosenbrock23`; the trajectory holds the initial point and every accepted step.

    t: list of times [units of t]; y: list of states (each a list of n floats/Duals) [units of y];
    dydt: list of f(t, y) at those times [units of y per unit of t] (data for Hermite dense output);
    success / message: status ("success" or the reason for stopping early);
    nfev: calls of f (including finite-difference df/dt and rejected steps); njac: Jacobian
    evaluations (one per accepted step; reused after a rejected step); nlu: LU
    factorizations of W (one per attempted step); nsteps: accepted steps; nrejected: rejected steps.
    """
    t: list
    y: list
    dydt: list
    success: bool = False
    message: str = ""
    nfev: int = 0
    njac: int = 0
    nlu: int = 0
    nsteps: int = 0
    nrejected: int = 0

    @property
    def y_final(self):
        """State at the last stored time (at t1 if success)."""
        return self.y[-1]


def rosenbrock23(f, jac, t_span, y0, dfdt=None, rtol=1e-5, atol=1e-8, atol_vec=None, h0=None,
                 hmax=None, max_steps=200_000, fd_dfdt=False):
    """Integrate dy/dt = f(t, y), y(t0) = y0 from t0 to t1 > t0 (forward in time only).

    f(t, y, dydt): writes the right-hand side into the list dydt (length n).
    jac(t, y, J): writes the dense Jacobian J[i*n + k] = d f_i / d y_k (flat row-major list, n*n).
        For Dual states the Jacobian must be Dual too (same generic code as f), else the
        derivative parts of the solution are wrong.
    dfdt: optional dfdt(t, y, out) writing df/dt (length n). None means an autonomous system
        (df/dt = 0, the source behaviour) unless fd_dfdt is True, which uses a forward difference of
        f in t (one extra f call per Jacobian evaluation, accuracy about sqrt(eps)). A
        time-dependent f with neither is integrated with a wrong df/dt = 0 (lower accuracy).
    t_span: (t0, t1) [units of t]; y0: initial state, list of n >= 1 floats or Duals [units of y].
    rtol: relative tolerance, default 1e-5, > 0. atol: absolute tolerance [units of y], default
        1e-8, >= 0; atol_vec: per-component absolute tolerances (length n) overriding atol.
    h0: first step [units of t], default 1e-4 (t1 - t0). hmax: largest step, default no limit
        (the source's 0.05 cap was specific to its problem). max_steps: attempted steps (accepted +
        rejected), default 200000.

    Failures are reported in the result (never an exception) for: t1 <= t0, empty y0, bad
    tolerances or atol_vec length, singular W, step underflow (h < 1e-15 max(|t|, 1e-3 (t1 - t0))),
    and max_steps; the partial trajectory is returned. A non-finite error estimate rejects the step.
    """
    n = len(y0)
    t0, t1 = float(t_span[0]), float(t_span[1])
    res = Rosenbrock23Result(t=[t0], y=[list(y0)], dydt=[])

    def fail(msg):
        res.message = msg
        return res

    if n == 0:
        return fail("empty initial state")
    if not (t1 > t0) or not math.isfinite(t0) or not math.isfinite(t1):
        return fail("t_span must satisfy t0 < t1 (both finite)")
    if not (rtol > 0.0) or not (atol >= 0.0):
        return fail("need rtol > 0 and atol >= 0")
    if atol_vec is not None and len(atol_vec) != n:
        return fail("atol_vec must be None or have one entry per component")
    atols = list(atol_vec) if atol_vec is not None else [atol] * n
    span = t1 - t0

    y = list(y0)
    f0, f1, f2, tv = [0.0] * n, [0.0] * n, [0.0] * n, [0.0] * n
    jm = [0.0] * (n * n)
    f(t0, y, f0)
    res.nfev = 1
    res.dydt.append(list(f0))

    t = t0
    h = 1e-4 * span if h0 is None else h0
    hmax = span if hmax is None else hmax
    have_jac = False        # J and df/dt at the current (t, y); kept across a rejected step
    attempts = 0
    while t < t1:
        if attempts >= max_steps:
            return fail(f"max_steps ({max_steps}) reached at t = {t}")
        attempts += 1
        if not (h > 0.0):
            return fail(f"step size {h} is not positive at t = {t}")
        h = min(h, hmax)
        last = h >= t1 - t
        if last:
            h = t1 - t
        hd = h * _D
        if not have_jac:
            jac(t, y, jm)
            res.njac += 1
            if dfdt is not None:
                dfdt(t, y, tv)
            elif fd_dfdt:
                dt = math.sqrt(2.220446049250313e-16) * max(abs(t), 1e-3 * span)
                f(t + dt, y, tv)
                res.nfev += 1
                tv = [(tv[i] - f0[i]) / dt for i in range(n)]
            else:
                tv = [0.0] * n
            have_jac = True
        w = [(1.0 if i == k else 0.0) - hd * jm[i * n + k] for i in range(n) for k in range(n)]
        fac = lu_factor(w, n)
        res.nlu += 1
        if not fac.success:
            return fail(f"W = I - h d J is singular at t = {t}, h = {h}: {fac.message}")

        # stage 1: W k1 = f0 + h d T
        k1 = lu_solve(fac, [f0[i] + hd * tv[i] for i in range(n)]).x
        # stage 2: W (k2 - k1) = f(t + h/2, y + h/2 k1) - k1
        yt = [y[i] + 0.5 * h * k1[i] for i in range(n)]
        f(t + 0.5 * h, yt, f1)
        s = lu_solve(fac, [f1[i] - k1[i] for i in range(n)]).x
        k2 = [s[i] + k1[i] for i in range(n)]
        yt = [y[i] + h * k2[i] for i in range(n)]       # candidate y_new (2nd order)
        t_new = t1 if last else t + h
        f(t_new, yt, f2)
        res.nfev += 2
        # stage 3, only for the error estimate
        k3 = lu_solve(fac, [f2[i] - _E32 * (k2[i] - f1[i]) - 2.0 * (k1[i] - f0[i]) + hd * tv[i]
                            for i in range(n)]).x

        # error = h/6 (k1 - 2 k2 + k3), derivative parts tested too (see module docstring)
        dmax = 0.0
        for i in range(n):
            dmax = max(dmax, abs(deriv(y[i])), abs(deriv(yt[i])))
        err = 0.0
        for i in range(n):
            e = (h / 6.0) * (val(k1[i]) - 2.0 * val(k2[i]) + val(k3[i]))
            sc = atols[i] + rtol * max(abs(val(y[i])), abs(val(yt[i])))
            es = (h / 6.0) * (deriv(k1[i]) - 2.0 * deriv(k2[i]) + deriv(k3[i]))
            ss = atols[i] + rtol * dmax
            for r in (abs(e / sc), abs(es / ss)):
                if math.isnan(r) or r > err:             # max() would drop a NaN term
                    err = r
        if not math.isfinite(err):
            err = 1e10
        if err <= 1.0:
            t = t_new
            y = yt
            f0 = list(f2)                                # first-same-as-last
            f2 = [0.0] * n
            have_jac = False
            res.nsteps += 1
            res.t.append(t)
            res.y.append(list(y))
            res.dydt.append(list(f0))
        else:
            res.nrejected += 1
        # new step: 3rd-order error estimate -> exponent 1/3, safety 0.8
        h *= 5.0 if err == 0.0 else min(5.0, max(0.2, 0.8 * err ** (-1.0 / 3.0)))
        if t < t1 and h < 1e-15 * max(abs(t), 1e-3 * span):
            return fail(f"step size underflow at t = {t}")
    res.success = True
    res.message = "success"
    return res
