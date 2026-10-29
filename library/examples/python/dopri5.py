"""Example: Arenstorf's periodic orbit (restricted three-body problem, Hairer-Norsett-Wanner I, II.0)
over one period at rtol 1e-6, atol 1e-9. Prints one JSON object (same keys as the Rust example)."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
from agent_tools.ode.dopri5 import dopri5  # noqa: E402

M = 0.012277471  # moon mass fraction, dimensionless
MH = 1.0 - M


def f(t, y, d):
    d1 = ((y[0] + M) ** 2 + y[1] * y[1]) ** 1.5
    d2 = ((y[0] - MH) ** 2 + y[1] * y[1]) ** 1.5
    d[0] = y[2]
    d[1] = y[3]
    d[2] = y[0] + 2.0 * y[3] - MH * (y[0] + M) / d1 - M * (y[0] - MH) / d2
    d[3] = y[1] - 2.0 * y[2] - MH * y[1] / d1 - M * y[1] / d2


PERIOD = 17.0652165601579625588917206249
Y0 = [0.994, 0.0, 0.0, -2.00158510637908252240537862224]
r = dopri5(f, (0.0, PERIOD), Y0, rtol=1e-6, atol=1e-9)
mid = r.nsteps // 2
print(json.dumps({"success": r.success, "message": r.message, "nsteps": r.nsteps, "nrejected": r.nrejected,
                  "nfev": r.nfev, "t_mid": r.t[mid], "y_mid": r.y[mid], "t_final": r.t[r.nsteps],
                  "y": r.y_final}))
