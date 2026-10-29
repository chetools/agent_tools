"""Example: box-constrained convex QP, minimize 0.5 x.H.x + g.x subject to lo <= x <= hi, on nine cases of
library/tests/reference_data/qp_box.json (mixed active set, infinite bounds, unconstrained, degenerate zero
multiplier, a fixed variable, cond(H) = 1e8 with n = 30, a random n = 20 problem, all bounds active, and a start at a
bound with zero gradient), default options.
Prints one JSON object (same keys as the Rust example): per case x, f, active, multipliers, success, nit, nfact."""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
from agent_tools.optimize.qp_box import QpBoxOptions, qp_box  # noqa: E402

CASES = ["random_n6", "infinite_bounds_n6", "unconstrained_inf_n10", "degenerate_zero_multiplier_n5",
         "fixed_variable_n5", "cond1e8_n30", "sweep_n20_cond1e+02", "all_active_n5", "start_at_bound_zero_gradient_n4"]

doc = json.load(open(pathlib.Path(__file__).resolve().parents[2] / "tests" / "reference_data" / "qp_box.json"))
out = {}
for name in CASES:
    c = doc["cases"][name]
    n = c["n"]
    lo = np.array([-np.inf if v is None else v for v in c["lo"]])
    hi = np.array([np.inf if v is None else v for v in c["hi"]])
    r = qp_box(np.array(c["H"]).reshape(n, n), c["g"], lo, hi, QpBoxOptions())
    out[name] = {"x": r.x.tolist(), "f": r.f, "active": r.active.tolist(), "multipliers": r.multipliers.tolist(),
                 "success": r.success, "nit": r.nit, "nfact": r.nfact}
print(json.dumps(out))
