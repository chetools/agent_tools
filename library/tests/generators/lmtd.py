"""Generator for reference_data/lmtd.json.

Reference: 50-digit mpmath evaluation of L = (a-b)/ln(a/b) (L = a when a == b) and its exact
partial derivatives dL/da = (ln(a/b) - (a-b)/a)/ln(a/b)^2, dL/db = ((a-b)/b - ln(a/b))/ln(a/b)^2,
evaluated at mp.dps = 50 (the cancellation near a = b is harmless at that precision; for
a == b the limits 1/2, 1/2 are used). Cases: ordinary ratios, extreme ratios (1e-6 .. 1e6),
and pairs straddling the series switch |a-b| < 1e-4 (a+b)/2 at relative differences from 1e-12 to 1e-2.
Sweeps (added after review): "sweep_rel_diff" has 240 log-spaced |a-b|/mean from 1e-9 to 1.999999 (mean 20,
alternating sign of a-b) and "sweep_ratio" has 241 log-spaced a/b from 1e-6 to 1e6 (b = 1 and a = 1 branches).
Run: python library/tests/generators/lmtd.py
"""
import mpmath as mp
import numpy as np

from _common import write

mp.mp.dps = 50


def exact(a, b):
    A, B = mp.mpf(a), mp.mpf(b)
    if A == B:
        return float(A), 0.5, 0.5
    lr = mp.log(A / B)
    d = A - B
    return float(d / lr), float((lr - d / A) / lr**2), float((d / B - lr) / lr**2)


pairs = [(30.0, 10.0), (10.0, 30.0), (5.0, 50.0), (100.0, 1.0), (1.0, 100.0), (1e6, 1.0), (1.0, 1e6),
         (1e-3, 1e3), (20.0, 20.0), (20.0, 20.00001), (45.0, 15.0), (2.0, 1.0)]
straddle = []
for rel in (1e-12, 1e-8, 1e-6, 1e-5, 5e-5, 9.9e-5, 1.01e-4, 2e-4, 1e-3, 1e-2):
    m = 20.0
    d = rel * m                           # |a-b| = rel * mean
    straddle.append((m + d / 2, m - d / 2))
sweep_rel = []
for i, rel in enumerate(np.geomspace(1e-9, 1.999999, 240)):
    d = float(rel) * 20.0 * (1 if i % 2 == 0 else -1)
    sweep_rel.append((20.0 + d / 2, 20.0 - d / 2))
sweep_ratio = [(float(r), 1.0) for r in np.geomspace(1e-6, 1e6, 241)]
cases = {"ordinary_and_extreme": [], "switch_region": [], "sweep_rel_diff": [], "sweep_ratio": []}
for key, ps in (("ordinary_and_extreme", pairs), ("switch_region", straddle), ("sweep_rel_diff", sweep_rel),
                ("sweep_ratio", sweep_ratio)):
    for a, b in ps:
        L, La, Lb = exact(a, b)
        cases[key].append({"a": a, "b": b, "L": L, "dLda": La, "dLdb": Lb,
                           "rel_diff": abs(a - b) / (0.5 * (a + b))})
write("lmtd", "library/tests/generators/lmtd.py",
      {"precision_digits": 50, "library": "mpmath", "switch_threshold": 1e-4},
      "a, b: terminal temperature differences [K]; L [K]; dLda, dLdb dimensionless.",
      cases, mpmath=mp.__version__)
