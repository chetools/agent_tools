"""Example: generic exothermic CSTR (C in mol/L, T in K, t in s), one RK4 step over dt = 10 s with 10
substeps: y1 and the step Jacobian, plus column 0 of the Jacobian by dual numbers (seed dC0).
Prints one JSON object (same keys as the Rust example)."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
from agent_tools.autodiff.dual import Dual, exp, val  # noqa: E402
from agent_tools.ode.rk4_step_jacobian import rk4_step, rk4_step_jacobian  # noqa: E402

QV, C0, T0, TC, K0, ER, DH, UA = 1.0 / 60.0, 1.0, 350.0, 300.0, 1.2e9, 8750.0, 209.0, 2.09 / 60.0


def rhs(t, y, d):
    k = K0 * exp(-(ER / y[1]))
    d[0] = QV * (C0 - y[0]) - k * y[0]
    d[1] = QV * (T0 - y[1]) + DH * k * y[0] + UA * (TC - y[1])


def jacobian(t, y, j):
    k = K0 * exp(-(ER / y[1]))
    dk = k * ER / (y[1] * y[1])
    j[0] = -QV - k
    j[1] = -(dk * y[0])
    j[2] = DH * k
    j[3] = -QV - UA + DH * dk * y[0]


r = rk4_step_jacobian(rhs, jacobian, 0.0, [0.5, 355.0], 10.0, 10)
rd = rk4_step(rhs, 0.0, [Dual(0.5, 1.0), Dual(355.0, 0.0)], 10.0, 10)
print(json.dumps({"success": r.success, "message": r.message, "nfev": r.nfev, "njac": r.njac,
                  "y1": [val(v) for v in r.y1], "phi": [val(v) for v in r.phi],
                  "dual_col0": [v.d for v in rd.y1]}))
