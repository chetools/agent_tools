"""Piecewise cubic Hermite interpolation through samples (t_i, y_i, dy/dt_i) (Python reference).

Source: Burden & Faires, "Numerical Analysis", 9th ed., section 3.4 (cubic Hermite interpolation);
reference for tests: scipy.interpolate.CubicHermiteSpline. Developed in chetools/claude
`packed-bed-runaway/python/pbr.py` (`hermite_peak`; same authors).
Rust twin: library/rust/src/interpolate/cubic_hermite.rs.

Units: t and x share a unit; y and the result share a unit; dy is the unit of y per unit of t.
"""
import numpy as np


def hermite_piece(y0, y1, m0, m1, h, s):
    """Cubic Hermite polynomial on one interval of width h > 0 (unit of t) at the fraction
    s = (x - t0) / h (dimensionless).

    y0, y1: end values; m0, m1: end slopes dy/dt (per unit of t). s may be an array; values outside
    [0, 1] extrapolate the same cubic. No checks are made.
    """
    s2, s3 = s * s, s * s * s
    return (y0 * (2.0 * s3 - 3.0 * s2 + 1.0) + m0 * (h * (s3 - 2.0 * s2 + s))
            + y1 * (-2.0 * s3 + 3.0 * s2) + m1 * (h * (s3 - s2)))


def cubic_hermite(t, y, dy, x):
    """Evaluate the piecewise cubic Hermite interpolant through (t[i], y[i], dy[i]) at x.

    t: strictly increasing, at least 2 points (not checked); y: values; dy: slopes dy/dt.
    x: scalar or array in the unit of t. Outside [t[0], t[-1]] the first or last cubic is extrapolated
    (as SciPy does by default); x equal to a knot uses the interval to its right (the last knot uses the
    last interval). NaN x gives NaN. Raises ValueError if t, y, dy differ in length or have < 2 points.
    """
    t, y, dy = (np.asarray(v, dtype=float) for v in (t, y, dy))
    n = len(t)
    if n < 2 or len(y) != n or len(dy) != n:
        raise ValueError("cubic_hermite: t, y, dy need equal length >= 2")
    x = np.asarray(x, dtype=float)
    j = np.clip(np.searchsorted(t, x, side="right") - 1, 0, n - 2)
    h = t[j + 1] - t[j]
    return hermite_piece(y[j], y[j + 1], dy[j], dy[j + 1], h, (x - t[j]) / h)
