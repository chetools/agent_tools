"""Example: BFGS on the 2-D Rosenbrock function from (-1.2, 1) and on the separable extended
Rosenbrock function with n = 10 from (-1.2, 1, -1.2, 1, ...), gtol = 1e-10.
Prints one JSON object (same keys as the Rust example). "xs" holds every point at which
fg was called, in call order (flat), so the two languages can be compared iterate by iterate."""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
from agent_tools.optimize.bfgs import BfgsOptions, bfgs  # noqa: E402


def ext_rosenbrock(x, g):
    """Extended Rosenbrock (More, Garbow, Hillstrom 1981, problem 21), n even.
    Written as a loop in the same operation order as the Rust example."""
    f = 0.0
    for i in range(len(x) // 2):
        a, b = x[2 * i], x[2 * i + 1]
        t = b - a * a
        u = 1.0 - a
        f += 100.0 * (t * t) + u * u
        g[2 * i] = -400.0 * a * t - 2.0 * u
        g[2 * i + 1] = 200.0 * t
    return f


def run(n):
    x0 = np.where(np.arange(n) % 2 == 0, -1.2, 1.0)
    xs = []

    def fg(x, g):
        xs.extend(x.tolist())
        return ext_rosenbrock(x, g)

    r = bfgs(fg, x0, BfgsOptions(max_iter=200, gtol=1e-10))
    return {"x": r.x.tolist(), "f": r.f, "grad_norm": r.grad_norm, "success": r.success,
            "nit": r.nit, "nfev": r.nfev, "xs": xs}


print(json.dumps({"rosenbrock2": run(2), "ext_rosenbrock10": run(10)}))
