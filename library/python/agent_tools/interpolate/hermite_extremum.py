"""Global maximum or minimum of the piecewise cubic Hermite interpolant through (t_i, y_i, dy/dt_i)
(Python reference of the Rust `hermite_extremum`).

Typical use: the peak of a profile returned by an adaptive ODE integrator, from the accepted steps and
their right-hand sides. On each interval the derivative of the Hermite cubic is a quadratic
A s^2 + B s + C in the scaled coordinate s in [0, 1]; its roots in (0, 1) are the stationary points,
and the extremum is the best of all knots and stationary points. Stable quadratic root formula
(Press et al., "Numerical Recipes", 3rd ed., eq. 5.6.4); cubic: Burden & Faires, "Numerical Analysis",
9th ed., section 3.4. Developed in chetools/claude `packed-bed-runaway/python/pbr.py`
(`hermite_peak`; same authors). Rust twin: library/rust/src/interpolate/hermite_extremum.rs.
The Rust version is also generic over dual numbers (value part frozen at the stationary point, envelope
theorem); this reference works on floats only.

Units: t and the returned t share a unit; y and the returned value share a unit; dy is the unit of y
per unit of t.
"""
import math
from dataclasses import dataclass

import numpy as np

from .cubic_hermite import hermite_piece


@dataclass
class HermiteExtremum:
    """Result of `hermite_extremum`.

    t: location of the extremum (unit of t); NaN if success is False.
    value: interpolant value there (unit of y); NaN if success is False.
    interval: index j of the interval [t[j], t[j+1]] holding the extremum; for an extremum exactly at
        knot t[i] it is i, except at the last knot where it is n - 2; 0 if success is False.
    success: True when the input was valid.
    message: "success", or the reason the input was rejected.
    """
    t: float
    value: float
    interval: int
    success: bool
    message: str


def _fail(msg):
    return HermiteExtremum(math.nan, math.nan, 0, False, msg)


def hermite_extremum(t, y, dy, kind="max"):
    """Global maximum (kind="max", default) or minimum (kind="min") of the piecewise cubic Hermite
    interpolant through (t[i], y[i], dy[i]).

    Inputs: t strictly increasing (unit of t), y values, dy slopes dy/dt; equal length n >= 2, all
    finite. Otherwise success is False with a message (no exception, no search). An unknown kind
    raises ValueError.

    Special cases (identical in the Rust twin):
    - Candidates are scanned in increasing t: knot t[0], the stationary points inside interval 0 in
      increasing order, knot t[1], and so on. A candidate replaces the best only if it is strictly
      better, so on exact ties the smallest t wins, and a knot beats an interior point of equal value
      (a flat cubic or a tied plateau returns its left end).
    - Monotone pieces have no interior stationary point; the extremum is then a knot (an endpoint of
      the data if the whole interpolant is monotone).
    - Only genuine sign changes of the derivative count (discriminant strictly positive), so an
      inflection with zero slope is never reported.
    - If |A| < 1e-14 (|B| + |C|) the derivative is treated as linear (root -C/B, none if B = 0).
    - Cost O(n); no iteration.
    """
    if kind not in ("max", "min"):
        raise ValueError("kind must be 'max' or 'min'")
    t, y, dy = (np.asarray(v, dtype=float).ravel() for v in (t, y, dy))
    n = len(t)
    if n < 2 or len(y) != n or len(dy) != n:
        return _fail("t, y and dy must have equal length >= 2")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(y)) and np.all(np.isfinite(dy))):
        return _fail("t, y and dy must be finite")
    if not np.all(np.diff(t) > 0):
        return _fail("t must be strictly increasing")
    sg = 1.0 if kind == "max" else -1.0
    best = (y[0], t[0], 0)  # (value, location, interval)
    for j in range(n - 1):
        h = t[j + 1] - t[j]
        y0, y1, m0, m1 = y[j], y[j + 1], h * dy[j], h * dy[j + 1]
        # dp/ds = A s^2 + B s + C for the cubic Hermite p(s), s in [0, 1]
        a = 6 * y0 + 3 * m0 - 6 * y1 + 3 * m1
        b = -6 * y0 - 4 * m0 + 6 * y1 - 2 * m1
        c = m0
        roots = []
        if abs(a) < 1e-14 * (abs(b) + abs(c)):
            if b != 0:
                roots = [-c / b]
        else:
            disc = b * b - 4 * a * c
            if disc > 0:
                q = -0.5 * (b + math.copysign(math.sqrt(disc), b))
                roots = sorted([q / a] + ([c / q] if q != 0 else []))
        for s in roots:
            if 0.0 < s < 1.0:
                p = float(hermite_piece(y[j], y[j + 1], dy[j], dy[j + 1], h, s))
                if sg * p > sg * best[0]:
                    best = (p, t[j] + s * h, j)
        if sg * y1 > sg * best[0]:
            best = (y1, t[j + 1], j + 1 if j + 1 < n - 1 else j)
    return HermiteExtremum(float(best[1]), float(best[0]), int(best[2]), True, "success")
