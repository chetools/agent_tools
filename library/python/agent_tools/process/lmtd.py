"""Numerically safe log-mean temperature difference with partial derivatives.

Source: standard definition LMTD = (dT1 - dT2) / ln(dT1 / dT2) (e.g. Kern, Process Heat
Transfer, 1950; Incropera et al., Fundamentals of Heat and Mass Transfer, counter-current
exchanger); series branch and derivative formulas derived here and checked in the tests.
Ported from chetools/claude heat-exchanger-network (hen.py `lmtd`, solver/src/lib.rs `lmtd`).

Algorithm. With m = (dT1 + dT2)/2 and d = dT1 - dT2:
  * if |d| < LMTD_SERIES_THRESHOLD * m (default 1e-4) use the series
        L ~= m - d^2/(12 m)
        dL/ddT1 ~= 1/2 - d/(6 m) + d^2/(24 m^2)
        dL/ddT2 ~= 1/2 + d/(6 m) + d^2/(24 m^2)
  * otherwise use the closed form with lr = ln(dT1/dT2):
        L = d / lr,  dL/ddT1 = (lr - d/dT1)/lr^2,  dL/ddT2 = (d/dT2 - lr)/lr^2.
Accuracy of both branches is measured by the generator tests/generators/lmtd.py against
50-digit mpmath values (see the catalog entry).
"""
import math

#: Relative switch threshold |dT1 - dT2| < threshold * mean(dT1, dT2) (dimensionless).
LMTD_SERIES_THRESHOLD = 1e-4


def lmtd(dt1, dt2):
    """Log-mean temperature difference and its partial derivatives.

    Inputs: dt1, dt2 are the two terminal temperature differences [K], both > 0
    (any consistent temperature-difference unit works; derivatives are then dimensionless).
    Returns (L, dL_ddt1, dL_ddt2): L in the unit of the inputs, derivatives dimensionless.
    Failure mode: if either input is <= 0 or not finite, returns (nan, nan, nan); the
    caller must treat a temperature cross as infeasible. Symmetric: L(a,b) = L(b,a).
    """
    if not (math.isfinite(dt1) and math.isfinite(dt2)) or dt1 <= 0.0 or dt2 <= 0.0:
        return (math.nan, math.nan, math.nan)
    m = 0.5 * (dt1 + dt2)
    d = dt1 - dt2
    if abs(d) < LMTD_SERIES_THRESHOLD * m:
        L = m - d * d / (12.0 * m)
        return (L, 0.5 - d / (6.0 * m) + d * d / (24.0 * m * m),
                0.5 + d / (6.0 * m) + d * d / (24.0 * m * m))
    # |d| < m  =>  d/dt2 in (-1, 2): log1p is exact where log(dt1/dt2) would cancel
    lr = math.log1p(d / dt2) if abs(d) < m else math.log(dt1 / dt2)
    return (d / lr, (lr - d / dt1) / (lr * lr), (d / dt2 - lr) / (lr * lr))
