"""Generator for reference_data/cholesky.json.

Cases: random SPD matrices (M M^T + n I, symmetrized, rounded to 8 decimals), Hilbert matrices
(ill-conditioned SPD, cond about 5e5 to 1.6e13) and non-SPD matrices. References:
numpy.linalg.cholesky (L) and scipy.linalg.cho_solve (x).
Run: python library/tests/generators/cholesky.py
"""
import numpy as np
import scipy.linalg as sla

from _common import write

SEED = 20261028
rng = np.random.RandomState(SEED)


def ref(A, b):
    L = np.linalg.cholesky(A)
    x = sla.cho_solve(sla.cho_factor(A, lower=True), b)
    assert np.allclose(L @ L.T, A, rtol=0, atol=1e-10 * np.max(np.abs(A)))
    return L, x


spd = []
for n in (1, 2, 3, 5, 10, 20, 50):
    M = rng.randn(n, n)
    A = np.round(0.5 * (M @ M.T + (M @ M.T).T) + n * np.eye(n), 8)
    A = 0.5 * (A + A.T)
    b = np.round(rng.randn(n), 6)
    L, x = ref(A, b)
    spd.append({"n": n, "cond": np.linalg.cond(A), "A": A.ravel().tolist(), "b": b.tolist(),
                "L": L.ravel().tolist(), "x": x.tolist()})

hilbert = []
for n in (5, 8, 10):
    A = 1.0 / (np.arange(n)[:, None] + np.arange(n)[None, :] + 1.0)
    b = np.ones(n)
    L, x = ref(A, b)
    hilbert.append({"n": n, "cond": np.linalg.cond(A), "A": A.ravel().tolist(), "b": b.tolist(),
                    "L": L.ravel().tolist(), "x": x.tolist()})

not_spd = []
for name, A in [("indefinite", [[1.0, 2.0], [2.0, 1.0]]),
                ("zero_pivot", [[0.0, 0.0], [0.0, 1.0]]),
                ("negative_definite", [[-2.0, 0.0], [0.0, -3.0]]),
                ("semidefinite", [[1.0, 1.0], [1.0, 1.0]])]:
    try:
        np.linalg.cholesky(np.array(A))
        raised = False
    except np.linalg.LinAlgError:
        raised = True
    assert raised, name
    not_spd.append({"name": name, "n": 2, "A": np.array(A).ravel().tolist(), "numpy_raises": True})

write("cholesky", "library/tests/generators/cholesky.py",
      {"seed": SEED, "rng": "numpy.random.RandomState", "L": "numpy.linalg.cholesky",
       "x": "scipy.linalg.cho_solve", "spd_rounding_decimals": 8},
      "Dimensionless. Matrices row-major flat. L lower triangular.",
      {"spd": spd, "hilbert": hilbert, "not_spd": not_spd})
