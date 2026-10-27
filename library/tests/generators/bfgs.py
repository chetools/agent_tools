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


add("quadratic5", "quadratic", {"A": A.ravel().tolist(), "b": b.tolist()}, quadratic(A, b),
    [0.0] * 5, x_quad, 0.5 * x_quad @ A @ x_quad - b @ x_quad)
add("rosenbrock2", "ext_rosenbrock", {"a": 100.0}, ext_rosenbrock(100.0), [-1.2, 1.0], [1.0, 1.0], 0.0)
add("ext_rosenbrock10", "ext_rosenbrock", {"a": 100.0}, ext_rosenbrock(100.0), [-1.2, 1.0] * 5, [1.0] * 10, 0.0)
add("scaled_valley", "ext_rosenbrock", {"a": 1000.0}, ext_rosenbrock(1000.0), [-1.2, 1.0], [1.0, 1.0], 0.0)

write("bfgs", "library/tests/generators/bfgs.py",
      {"seed": SEED, "rng": "numpy.random.RandomState", "reference": "scipy.optimize.minimize BFGS",
       "gtol": GTOL, "maxiter": 10000, "gtol_norm": "max-norm (scipy default norm=inf)"},
      "Dimensionless test functions. kind 'quadratic': f = x.A.x/2 - b.x (A row-major flat). kind "
      "'ext_rosenbrock': sum over pairs (x_2i, x_2i+1) of a (x_2i+1 - x_2i^2)^2 + (1 - x_2i)^2. "
      "x_exact and f_exact are analytic (quadratic: numpy.linalg.solve).",
      cases)
for k, c in cases.items():
    print(k, c["scipy"])
