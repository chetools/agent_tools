"""Generator for reference_data/bfgs.json.

Cases (all unconstrained, gradient analytic):
  quadratic5        f = x.A.x/2 - b.x, A = M M^T + 5 I rounded to 6 decimals (n = 5); exact x* = A^-1 b
                    (numpy.linalg.solve), start at 0.
  rosenbrock2       f = 100 (y - x^2)^2 + (1 - x)^2 from (-1.2, 1); x* = (1, 1), f* = 0.
  ext_rosenbrock10  separable extended Rosenbrock (More, Garbow, Hillstrom 1981, problem 21), n = 10,
                    sum over pairs of the 2-D function, from (-1.2, 1, -1.2, 1, ...); x* = 1, f* = 0.
  scaled_valley     f = 1000 (y - x^2)^2 + (1 - x)^2 from (-1.2, 1): a narrow curved valley with a
                    Hessian condition number 2.5e4 at the minimum (2.5e3 for the
                    standard a = 100), so the identity start is badly scaled; x* = (1, 1), f* = 0.
  quadratic8_cond1e3  f = x.A.x/2 - b.x, n = 8, A = Q diag(logspace(0, 3, 8)) Q^T (Q from the QR factorization of a
                    seeded Gaussian matrix, RandomState(SEED8 = SEED + 3)), rounded to 6 decimals: eigenvalues 1 to 1e3
                    (condition number 1e3). The cost case: halving-only Armijo needs several times SciPy's nfev
                    here (a Wolfe line search accepts the unit step on a quadratic far more often).
                    Seed 3 of the sweep in bfgs_sweep.json (all 8 seeds are tested there).
Reference minimizer: scipy.optimize.minimize(method="BFGS", jac=analytic, options gtol=1e-10, maxiter=10000)
(max-norm of the gradient, the same stopping test as the library); its nit, nfev (f calls) and njev
(gradient calls) are stored for comparison.
Run: python library/tests/generators/bfgs.py
"""
import numpy as np
from scipy.optimize import minimize

from _common import write

SEED = 20261029
GTOL = 1e-10


def quadratic(A, b):
    def fg(x):
        return 0.5 * x @ A @ x - b @ x, A @ x - b
    return fg


def ext_rosenbrock(a):
    def fg(x):
        u, v = x[0::2], x[1::2]
        t = v - u * u
        g = np.empty_like(x)
        g[0::2] = -4 * a * u * t - 2 * (1 - u)
        g[1::2] = 2 * a * t
        return float(np.sum(a * t * t + (1 - u) ** 2)), g
    return fg


rng = np.random.RandomState(SEED)
M = rng.randn(5, 5)
A = np.round(M @ M.T + 5 * np.eye(5), 6)
A = 0.5 * (A + A.T)
b = np.round(rng.randn(5), 6)
x_quad = np.linalg.solve(A, b)

cases = {}


def add(name, kind, params, fg, x0, x_exact, f_exact):
    r = minimize(fg, np.array(x0, float), jac=True, method="BFGS",
                 options={"gtol": GTOL, "maxiter": 10000})
    assert r.success, (name, r.message)
    cases[name] = {"kind": kind, "params": params, "x0": list(x0), "x_exact": list(map(float, x_exact)),
                   "f_exact": float(f_exact),
                   "scipy": {"x": r.x.tolist(), "fun": float(r.fun), "nit": int(r.nit), "nfev": int(r.nfev),
                             "njev": int(r.njev), "grad_max": float(np.max(np.abs(r.jac)))}}


SEED8 = SEED + 3
rng2 = np.random.RandomState(SEED8)
Q, _ = np.linalg.qr(rng2.randn(8, 8))
A8 = np.round(Q @ np.diag(np.logspace(0, 3, 8)) @ Q.T, 6)
A8 = 0.5 * (A8 + A8.T)
b8 = np.round(rng2.randn(8), 6)
x_quad8 = np.linalg.solve(A8, b8)

add("quadratic5", "quadratic", {"A": A.ravel().tolist(), "b": b.tolist()}, quadratic(A, b),
    [0.0] * 5, x_quad, 0.5 * x_quad @ A @ x_quad - b @ x_quad)
add("rosenbrock2", "ext_rosenbrock", {"a": 100.0}, ext_rosenbrock(100.0), [-1.2, 1.0], [1.0, 1.0], 0.0)
add("ext_rosenbrock10", "ext_rosenbrock", {"a": 100.0}, ext_rosenbrock(100.0), [-1.2, 1.0] * 5, [1.0] * 10, 0.0)
add("scaled_valley", "ext_rosenbrock", {"a": 1000.0}, ext_rosenbrock(1000.0), [-1.2, 1.0], [1.0, 1.0], 0.0)
add("quadratic8_cond1e3", "quadratic", {"A": A8.ravel().tolist(), "b": b8.tolist()}, quadratic(A8, b8),
    [0.0] * 8, x_quad8, 0.5 * x_quad8 @ A8 @ x_quad8 - b8 @ x_quad8)

write("bfgs", "library/tests/generators/bfgs.py",
      {"seed": SEED, "seed_quadratic8": SEED8, "rng": "numpy.random.RandomState", "reference": "scipy.optimize.minimize BFGS",
       "gtol": GTOL, "maxiter": 10000, "gtol_norm": "max-norm (scipy default norm=inf)"},
      "Dimensionless test functions. kind 'quadratic': f = x.A.x/2 - b.x (A row-major flat). kind "
      "'ext_rosenbrock': sum over pairs (x_2i, x_2i+1) of a (x_2i+1 - x_2i^2)^2 + (1 - x_2i)^2. "
      "x_exact and f_exact are analytic (quadratic: numpy.linalg.solve).",
      cases)
for k, c in cases.items():
    print(k, c["scipy"])

# Seed sweep for the curvature-threshold test: the same ill-conditioned 8-D quadratic family as
# quadratic8_cond1e3 for 8 seeds (SEED + 1 ... SEED + 8). No SciPy reference is needed: the exact
# minimizer is numpy.linalg.solve.
sweep = {}
for off in range(1, 9):
    r = np.random.RandomState(SEED + off)
    Qs, _ = np.linalg.qr(r.randn(8, 8))
    As = np.round(Qs @ np.diag(np.logspace(0, 3, 8)) @ Qs.T, 6)
    As = 0.5 * (As + As.T)
    bs = np.round(r.randn(8), 6)
    sweep[f"seed_plus_{off}"] = {"A": As.ravel().tolist(), "b": bs.tolist(),
                                 "x_exact": np.linalg.solve(As, bs).tolist()}
write("bfgs_sweep", "library/tests/generators/bfgs.py",
      {"seed": SEED, "offsets": "1..8", "rng": "numpy.random.RandomState", "gtol": GTOL},
      "Ill-conditioned 8-D quadratics f = x.A.x/2 - b.x (A row-major flat), eigenvalues logspace(0, 3, 8), "
      "start at 0; x_exact = numpy.linalg.solve(A, b).", sweep)
