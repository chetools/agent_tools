"""Generator for reference_data/hermite_extremum.json (also used by the cubic_hermite tests).

Cases:
  cubic     an exact cubic sampled with its exact slopes: the Hermite interpolant IS the cubic, so the
            extrema are the roots of p'(t) (numpy.roots of the exact derivative coefficients).
  sine      sin(t) on 8 coarse non-uniform knots with exact slopes cos(t). Reference: scipy
            CubicHermiteSpline evaluated on a dense grid (2001 points per interval) and refined in each
            interval with scipy.optimize.minimize_scalar(method="bounded", xatol=1e-13); the global best
            over intervals. Cross-checked here against the roots of the spline's derivative PPoly.
  gaussian  narrow Gaussian exp(-(t-t0)^2/(2 sigma^2)), sigma=0.02, t0=0.5137 (off the grid), peak 1,
            sampled with exact slopes on 12*2^k uniform intervals of width 0.0375/2^k, placed so the peak
            always sits at the same fraction 0.37 of its cell (clean error constant across spacings); the "stiff-looking
            runaway-like peak". Reference: the same dense + minimize_scalar procedure on the spline, and
            the true peak (1 at t0) to measure the interpolation error as the spacing shrinks.
  evaluator CubicHermiteSpline values at knots, interior points and outside the data (extrapolation).
Run: python library/tests/generators/hermite_extremum.py
"""
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import minimize_scalar

from _common import write

NDENSE = 2001
XATOL = 1e-13


def scipy_extremum(t, y, dy, kind):
    """Global extremum of the spline: dense grid per interval, then bounded minimize_scalar."""
    sp = CubicHermiteSpline(t, y, dy)
    sg = 1.0 if kind == "max" else -1.0
    best = None
    for j in range(len(t) - 1):
        xs = np.linspace(t[j], t[j + 1], NDENSE)
        k = int(np.argmax(sg * sp(xs)))
        lo, hi = xs[max(k - 1, 0)], xs[min(k + 1, NDENSE - 1)]
        r = minimize_scalar(lambda x: -sg * float(sp(x)), bounds=(lo, hi), method="bounded",
                            options={"xatol": XATOL})
        cands = [(float(sp(r.x)), float(r.x)), (float(sp(xs[k])), float(xs[k]))]
        v, x = max(cands, key=lambda c: sg * c[0])
        if best is None or sg * v > sg * best[0]:
            best = (v, x, j)
    return best, sp


def rootcheck(sp, t, kind, val, x):
    """Second method: stationary points from the roots of the spline derivative plus the knots."""
    sg = 1.0 if kind == "max" else -1.0
    cand = [(float(sp(ti)), ti) for ti in t]
    for r in sp.derivative().roots(extrapolate=False):
        cand.append((float(sp(r)), float(r)))
    v, _ = max(cand, key=lambda c: sg * c[0])
    assert abs(v - val) < 1e-11 * max(1.0, abs(val)), (kind, v, val)


cases = {}

# --- exact cubic: p(t) = 0.5 t^3 - 2.6 t^2 + 3 t + 1 on non-uniform knots in [0.2, 3.4]; the global
# max (about 2.0 at t=0.731) and min (about -0.01 at t=2.736) are both interior stationary points
coef = np.array([0.5, -2.6, 3.0, 1.0])
p, dp = np.poly1d(coef), np.poly1d(coef).deriv()
t = np.array([0.2, 0.55, 0.95, 1.3, 2.1, 2.9, 3.4])
y, dy = p(t), dp(t)
crit = [r.real for r in np.roots(dp.coeffs) if abs(r.imag) < 1e-12 and t[0] < r.real < t[-1]]
cand = [(float(p(x)), float(x)) for x in list(t) + crit]
vmax, tmax = max(cand)
vmin, tmin = min(cand)
assert 0.2 < tmax < 3.4 and 0.2 < tmin < 3.4 and len(crit) == 2  # interior, not endpoints
cases["cubic"] = {"coef": coef.tolist(), "t": t.tolist(), "y": y.tolist(), "dy": dy.tolist(),
                  "max": {"t": tmax, "value": vmax}, "min": {"t": tmin, "value": vmin},
                  "stationary_points": sorted(crit)}

# --- coarse sine
t = np.array([0.0, 0.9, 1.7, 2.6, 3.9, 4.6, 5.9, 7.2])
y, dy = np.sin(t), np.cos(t)
sine = {"t": t.tolist(), "y": y.tolist(), "dy": dy.tolist()}
for kind in ("max", "min"):
    (v, x, j), sp = scipy_extremum(t, y, dy, kind)
    rootcheck(sp, t, kind, v, x)
    sine[kind] = {"t": x, "value": v, "interval": j}
cases["sine"] = sine

# --- narrow Gaussian
SIGMA, T0 = 0.02, 0.5137
H0, PHASE = 0.0375, 0.37
runs = []
for k in range(6):
    n = 12 * 2 ** k
    h = H0 / 2 ** k
    # the peak sits at the same relative position PHASE inside its cell for every h, so the error
    # constant is the same on all grids and the observed order is clean
    t = (T0 - (n // 2 + PHASE) * h) + h * np.arange(n + 1)
    g = np.exp(-0.5 * ((t - T0) / SIGMA) ** 2)
    dg = -(t - T0) / SIGMA ** 2 * g
    (v, x, j), sp = scipy_extremum(t, g, dg, "max")
    rootcheck(sp, t, "max", v, x)
    runs.append({"n": n, "h": float(t[1] - t[0]), "t": t.tolist(), "y": g.tolist(), "dy": dg.tolist(),
                 "ref": {"t": x, "value": v, "interval": j}})
cases["gaussian"] = {"sigma": SIGMA, "t0": T0, "phase": PHASE, "true_value": 1.0, "runs": runs}

# --- evaluator
t = np.array([0.0, 0.5, 1.4, 2.0, 3.5])
y, dy = np.array([1.0, -0.3, 2.0, 0.5, 0.0]), np.array([0.2, 1.0, -2.0, 0.7, -1.0])
sp = CubicHermiteSpline(t, y, dy)
xs = np.concatenate([t, [0.1, 0.25, 0.9, 1.4000001, 1.99, 2.75, 3.4], [-1.0, -0.001, 3.6, 5.0]])
cases["evaluator"] = {"t": t.tolist(), "y": y.tolist(), "dy": dy.tolist(), "x": xs.tolist(),
                      "ref": sp(xs).tolist()}

write("hermite_extremum", "library/tests/generators/hermite_extremum.py",
      {"dense_points_per_interval": NDENSE, "minimize_scalar": "bounded", "xatol": XATOL,
       "spline": "scipy.interpolate.CubicHermiteSpline", "gaussian_sigma": SIGMA, "gaussian_t0": T0, "gaussian_h0": H0, "gaussian_cell_phase": PHASE},
      "t and y in arbitrary consistent units, dy = dy/dt. The exact-cubic reference is analytic "
      "(numpy.roots of p'). Sine and Gaussian references are SciPy's spline refined by bounded scalar "
      "minimisation and cross-checked against the roots of the spline's derivative. The Gaussian "
      "'true' peak is 1 at t0.", cases)
