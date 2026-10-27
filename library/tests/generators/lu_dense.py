"""Generator for reference_data/lu_dense.json.

Cases: random well-conditioned matrices n = 1..50 (entries rounded to 6 decimals so the file
stays small; the rounded matrix IS the test matrix), ill-conditioned matrices (SVD
construction, cond about 1e4, 1e8, 1e12, not rounded), singular matrices, and a parametric
case A(p) x = b(p) for the Dual test. References: scipy.linalg.lu_factor/lu_solve; numpy.linalg.solve
is used as a second check inside this script.
Run: python library/tests/generators/lu_dense.py
"""
import warnings

import numpy as np
import scipy.linalg as sla

from _common import write

SEED = 20261027
rng = np.random.RandomState(SEED)


def reference(A, b):
    lu, piv = sla.lu_factor(A)
    x = sla.lu_solve((lu, piv), b)
    x2 = np.linalg.solve(A, b)
    c = np.linalg.cond(A)
    assert np.max(np.abs(x - x2)) / np.max(np.abs(x)) < 1e-12 * c + 1e-14, "scipy and numpy disagree"
    return lu, piv, x, c


well = []
for n in range(1, 51):
    A = np.round(rng.randn(n, n), 6)
    b = np.round(rng.randn(n), 6)
    lu, piv, x, c = reference(A, b)
    row = {"n": n, "A": A.ravel().tolist(), "b": b.tolist(), "x": x.tolist(), "cond": c}
    if n <= 10:
        row["lu"] = lu.ravel().tolist()
        row["piv"] = piv.tolist()
    well.append(row)

ill = []
for cond in (1e4, 1e8, 1e12):
    for n in (2, 3, 5, 8, 10, 15, 20, 30):
        U, _ = np.linalg.qr(rng.randn(n, n))
        V, _ = np.linalg.qr(rng.randn(n, n))
        s = np.logspace(0, -np.log10(cond), n)
        A = U @ np.diag(s) @ V.T
        b = rng.randn(n)
        lu, piv, x, c = reference(A, b)
        ill.append({"n": n, "target_cond": cond, "cond": c, "A": A.ravel().tolist(),
                    "b": b.tolist(), "x": x.tolist()})

singular = []
for name, A in [("rank1_2x2", [[1.0, 2.0], [2.0, 4.0]]),
                ("zero_3x3", [[0.0] * 3] * 3),
                ("zero_column", [[1.0, 0.0, 2.0], [3.0, 0.0, 4.0], [5.0, 0.0, 7.0]])]:
    A = np.array(A)
    try:
        np.linalg.solve(A, np.ones(len(A)))
        raised = False
    except np.linalg.LinAlgError:
        raised = True
    assert raised, name
    singular.append({"name": name, "n": len(A), "A": A.ravel().tolist(), "numpy_raises": True})

param = []
for n in (4, 8):
    A0 = np.round(rng.randn(n, n), 6) + n * np.eye(n)
    A1 = np.round(rng.randn(n, n), 6)
    b0 = np.round(rng.randn(n), 6)
    b1 = np.round(rng.randn(n), 6)
    p0 = 0.3
    Ap = A0 + p0 * A1
    x = np.linalg.solve(Ap, b0 + p0 * b1)
    dx = np.linalg.solve(Ap, b1 - A1 @ x)          # d/dp [A(p)^-1 b(p)] = A^-1 (b' - A' x)
    h = 1e-6                                        # independent check by central differences
    fd = (np.linalg.solve(A0 + (p0 + h) * A1, b0 + (p0 + h) * b1)
          - np.linalg.solve(A0 + (p0 - h) * A1, b0 + (p0 - h) * b1)) / (2 * h)
    assert np.max(np.abs(fd - dx)) < 1e-7 * np.max(np.abs(dx)), "analytic derivative check"
    param.append({"n": n, "p0": p0, "A0": A0.ravel().tolist(), "A1": A1.ravel().tolist(),
                  "b0": b0.tolist(), "b1": b1.tolist(), "x": x.tolist(), "dx": dx.tolist()})

write("lu_dense", "library/tests/generators/lu_dense.py",
      {"seed": SEED, "rng": "numpy.random.RandomState", "well_rounding_decimals": 6,
       "ill_cond_targets": [1e4, 1e8, 1e12], "reference_x": "scipy.linalg.lu_solve",
       "reference_lu": "scipy.linalg.lu_factor (n <= 10 only)"},
      "Dimensionless. 'piv' is the 0-based LAPACK pivot vector. 'param': x(p) = A(p)^-1 b(p) with "
      "A(p) = A0 + p A1, b(p) = b0 + p b1; dx = A^-1 (b1 - A1 x) from numpy.linalg.solve, "
      "checked against central differences (h=1e-6) in the generator.",
      {"well": well, "ill": ill, "singular": singular, "param": param})
