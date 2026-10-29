"""Adaptive Dormand-Prince 5(4) explicit Runge-Kutta integrator for non-stiff IVPs of any dimension n
(the scheme of SciPy `solve_ivp(method="RK45")` and MATLAB `ode45`).

Sources: J. R. Dormand and P. J. Prince, "A family of embedded Runge-Kutta formulae", J. Comput. Appl.
Math. 6(1), 19-26 (1980) (7-stage FSAL tableau, pair 5(4)); E. Hairer, S. P. Norsett and G. Wanner,
"Solving Ordinary Differential Equations I", 2nd ed., Springer (1993), Sec. II.4 (starting step size,
step size control). Ported from chetools/muse `twin-cstr` (`python/cstr_model.py` `_solve_ivp` and
`solver/src/model.rs` `dopri5`), generalised with the Rust-style options and result of
`agent_tools.ode.rosenbrock23`, failures reported instead of raised. Reference for tests: SciPy
`solve_ivp(method="RK45")` (same tolerances, compared step by step) and `DOP853` at rtol 1e-13, atol 1e-14.
Rust twin: library/rust/src/ode/dopri5.rs (same algorithm and defaults).

One step from (t, y) with step h and k1 = f(t, y) (kept from the previous accepted step, "first same as last"):

    k_s = f(t + c_s h, y + h sum_{j<s} a_sj k_j),  s = 2..6      (c = 1/5, 3/10, 4/5, 8/9, 1)
    y_new = y + h (35/384 k1 + 500/1113 k3 + 125/192 k4 - 2187/6784 k5 + 11/84 k6)      (5th order)
    k7 = f(t + h, y_new)
    err = h (71/57600 k1 - 71/16695 k3 + 71/1920 k4 - 17253/339200 k5 + 22/525 k6 - 1/40 k7)  (4th order)

Step acceptance and sizing (SciPy RK45): E = rms_i(err_i / (atol_i + rtol max(|y_i|, |y_new_i|))), accepted
if E < 1. With fac = 0.9 E**(-1/5): after an accepted step h *= min(10, fac) (10 if E == 0), and fac is at
most 1 if the step before was rejected; after a rejected step h *= max(0.2, fac). The starting step is
SciPy's `select_initial_step`.

Changes from the source (`_solve_ivp` / `model.rs`), made to follow SciPy RK45 exactly: growth cap 10
(source 5), shrink floor 0.2 (source 0.1), no growth right after a rejection (source: none), acceptance
E < 1 (source E <= 1); starting step as SciPy (source: 1e-6 only if both d0 and d1 are small, and
max(d0, d1, d2) in h1, which divides by zero for f(t0, y0) = 0). Also new: failures are returned instead
of raised, non-finite error estimates reject the step, per-component atol, h0/hmax options, and
step-size underflow detection.

Pure Python: y is a list of floats.
"""
import math
from dataclasses import dataclass

_SAFETY, _MIN_FACTOR, _MAX_FACTOR = 0.9, 0.2, 10.0
_C = (1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0)
_A = (
    (1.0 / 5.0,),
    (3.0 / 40.0, 9.0 / 40.0),
    (44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0),
    (19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0),
    (9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0),
)
_B = (35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0)
_E = (71.0 / 57600.0, 0.0, -71.0 / 16695.0, 71.0 / 1920.0, -17253.0 / 339200.0, 22.0 / 525.0, -1.0 / 40.0)


@dataclass
class Dopri5Result:
    """Result of `dopri5`; the trajectory holds the initial point and every accepted step.

    t: list of times [units of t], ends exactly at t1 on success; y: list of states (each a list of n
    floats) [units of y]; success / message: status ("success" or the reason for stopping early);
    nfev: calls of f (1 at t0, 1 for the automatic first step, 6 per attempted step);
    nsteps: accepted steps; nrejected: rejected steps.
    """
    t: list
    y: list
    success: bool = False
    message: str = ""
    nfev: int = 0
    nsteps: int = 0
    nrejected: int = 0

    @property
    def y_final(self):
        """State at the last stored time (at t1 if success)."""
        return self.y[-1]


def _rms(v):
    """Root-mean-square norm; NaN if any entry is not finite."""
    s = sum(x * x for x in v)
    return math.sqrt(s / len(v)) if math.isfinite(s) else math.nan


def dopri5(f, t_span, y0, rtol=1e-6, atol=1e-9, atol_vec=None, h0=None, hmax=None, max_steps=100_000):
    """Integrate dy/dt = f(t, y), y(t0) = y0 from t0 to t1 > t0 (forward in time only).

    f(t, y, dydt): writes the right-hand side into the list dydt (length n) [y per unit of t]. It is
        called at the stage times, so time-dependent systems need nothing extra.
    t_span: (t0, t1) [units of t]; y0: initial state, list of n >= 1 floats [units of y].
    rtol: relative tolerance, default 1e-6, > 0. atol: absolute tolerance [units of y], default 1e-9,
        >= 0; atol_vec: per-component absolute tolerances (length n, each >= 0) overriding atol.
    h0: first step [units of t] (SciPy `first_step`), > 0; default None = SciPy's starting-step
        heuristic (one extra f call): with scale_i = atol_i + rtol |y0_i|, d0 = rms(y0/scale),
        d1 = rms(f0/scale), h0 = 1e-6 if d0 < 1e-5 or d1 < 1e-5 else 0.01 d0/d1; an explicit Euler
        probe gives d2 = rms((f(t0+h0, y0+h0 f0) - f0)/scale)/h0; h1 = (0.01/max(d1, d2))**(1/5)
        (max(1e-6, 1e-3 h0) if both <= 1e-15); the start is min(100 h0, h1, t1 - t0, hmax).
    hmax: largest step [units of t] (SciPy `max_step`), > 0; default None = no limit (t1 - t0).
    max_steps: attempted steps (accepted + rejected), default 100000 (the source budget).

    Failures are reported in the result (never an exception) for: t1 <= t0, empty y0, bad tolerances,
    atol_vec length, h0 or hmax; an automatic first step that is not a positive finite number (zero
    scale or non-finite f: pass h0); step underflow (h < 1e-15 max(|t|, 1e-3 (t1 - t0)), where a
    persistent NaN or a finite-time blow-up ends); and max_steps. The partial trajectory is returned.
    A non-finite error estimate rejects the step (h shrinks by the floor factor 0.2).
    Cost per attempted step: 6 calls of f. Not suited to stiff problems (nsteps then grows with the
    stiffness, not with the accuracy).
    """
    n = len(y0)
    t0, t1 = float(t_span[0]), float(t_span[1])
    res = Dopri5Result(t=[t0], y=[[float(v) for v in y0]])

    def fail(msg):
        res.message = msg
        return res

    if n == 0:
        return fail("empty initial state")
    if not (t1 > t0) or not math.isfinite(t0) or not math.isfinite(t1):
        return fail("t_span must satisfy t0 < t1 (both finite)")
    if not (rtol > 0.0) or not (atol >= 0.0) or not math.isfinite(rtol) or not math.isfinite(atol):
        return fail("need finite rtol > 0 and atol >= 0")
    if atol_vec is not None and (len(atol_vec) != n or any(not (a >= 0.0) for a in atol_vec)):
        return fail("atol_vec must be None or have one entry >= 0 per component")
    if (h0 is not None and not (h0 > 0.0)) or (hmax is not None and not (hmax > 0.0)):
        return fail("h0 and hmax must be > 0")
    atols = list(atol_vec) if atol_vec is not None else [atol] * n
    span = t1 - t0
    hmax = span if hmax is None else hmax

    y = [float(v) for v in y0]
    k = [[0.0] * n for _ in range(7)]
    f(t0, y, k[0])
    res.nfev = 1

    if h0 is None:
        sc = [atols[i] + rtol * abs(y[i]) for i in range(n)]
        with_scale = all(s > 0.0 for s in sc)
        d0 = _rms([y[i] / sc[i] for i in range(n)]) if with_scale else math.nan
        d1 = _rms([k[0][i] / sc[i] for i in range(n)]) if with_scale else math.nan
        hh = 1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1
        f1 = [0.0] * n
        f(t0 + hh, [y[i] + hh * k[0][i] for i in range(n)], f1)
        res.nfev += 1
        d2 = _rms([(f1[i] - k[0][i]) / sc[i] for i in range(n)]) / hh if with_scale else math.nan
        if d1 <= 1e-15 and d2 <= 1e-15:
            h1 = max(1e-6, hh * 1e-3)
        else:
            h1 = (0.01 / max(d1, d2)) ** 0.2
        h = min(100.0 * hh, h1)
    else:
        h = h0
    h = min(h, span, hmax)
    if not (h > 0.0) or not math.isfinite(h):
        return fail("automatic first step is not a positive finite number "
                    "(zero scale or non-finite f); pass h0")

    t = t0
    attempts = 0
    rejected = False                                      # the previous attempt was rejected
    while t < t1:
        if attempts >= max_steps:
            return fail(f"max_steps ({max_steps}) reached at t = {t}")
        attempts += 1
        h = min(h, hmax)
        last = h >= t1 - t
        if last:
            h = t1 - t
        for s in range(5):
            yt = []
            for i in range(n):
                acc = 0.0
                for j in range(s + 1):
                    acc += _A[s][j] * k[j][i]
                yt.append(y[i] + h * acc)
            f(t + _C[s] * h, yt, k[s + 1])
        ynew = []
        for i in range(n):
            acc = 0.0
            for j in range(6):
                acc += _B[j] * k[j][i]
            ynew.append(y[i] + h * acc)
        t_new = t1 if last else t + h
        f(t_new, ynew, k[6])
        res.nfev += 6
        ev = []
        for i in range(n):
            acc = 0.0
            for j in range(7):
                acc += _E[j] * k[j][i]
            ev.append(h * acc / (atols[i] + rtol * max(abs(y[i]), abs(ynew[i]))))
        en = _rms(ev)
        if not math.isfinite(en):
            en = 1e10
        if en < 1.0:
            t = t_new
            y = ynew
            k[0], k[6] = k[6], [0.0] * n                  # first-same-as-last
            res.nsteps += 1
            res.t.append(t)
            res.y.append(list(y))
            fac = _MAX_FACTOR if en == 0.0 else min(_MAX_FACTOR, _SAFETY * en ** -0.2)
            if rejected:
                fac = min(1.0, fac)                       # no growth right after a rejection
            h *= fac
            rejected = False
        else:
            res.nrejected += 1
            h *= max(_MIN_FACTOR, _SAFETY * en ** -0.2)
            rejected = True
        if t < t1 and h < 1e-15 * max(abs(t), 1e-3 * span):
            return fail(f"step size underflow at t = {t}")
    res.success = True
    res.message = "success"
    return res
