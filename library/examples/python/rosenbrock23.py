"""Example: Robertson's stiff kinetics to t = 40 with dual numbers: y(40) and dy(40)/dk1 (k1 = 0.04).
Prints one JSON object (same keys as the Rust example)."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
from agent_tools.autodiff.dual import Dual, val  # noqa: E402
from agent_tools.ode.rosenbrock23 import rosenbrock23  # noqa: E402

k1 = Dual(0.04, 1.0)


def f(t, y, d):
    d[0] = -(k1 * y[0]) + 1e4 * y[1] * y[2]
    d[1] = k1 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1]
    d[2] = 3e7 * y[1] * y[1]


def jac(t, y, j):
    j[:] = [-k1, 1e4 * y[2], 1e4 * y[1],
            k1, -(1e4 * y[2]) - 6e7 * y[1], -(1e4 * y[1]),
            Dual(0.0), 6e7 * y[1], Dual(0.0)]


r = rosenbrock23(f, jac, (0.0, 40.0), [Dual(1.0), Dual(0.0), Dual(0.0)], rtol=1e-5, atol=1e-9, h0=1e-6)
mid = r.nsteps // 2
print(json.dumps({"success": r.success, "message": r.message, "nsteps": r.nsteps, "nrejected": r.nrejected,
                  "nfev": r.nfev, "njac": r.njac, "nlu": r.nlu, "t_mid": r.t[mid],
                  "y_mid": [val(v) for v in r.y[mid]], "t_final": r.t[r.nsteps],
                  "y": [val(v) for v in r.y_final], "dy_dk1": [v.d for v in r.y_final]}))
