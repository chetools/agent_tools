"""Example: RTD analysis of two synthetic records truncated early (tanks N = 4, tau = 60 s; open-open
dispersion Pe = 12, L/u = 80 s): moments with tail, moment fits, least-squares fits.
Prints one JSON object (same keys as the Rust example)."""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
from agent_tools.stats.rtd_analysis import (CLOSED_CLOSED, DISPERSION, OPEN_OPEN, TANKS,  # noqa: E402
                                            e_dispersion, e_tanks, fit_lsq, fit_moments, rtd_moments)

n, dt = 481, 0.5                      # samples, s (record ends at 240 s = 4 tau)
t = np.arange(n) * dt
C = 3.0 * e_tanks(t / 60.0, 4.0) / 60.0
m = rtd_moments(t, C)
nt = fit_moments(m.mean, m.variance, TANKS)
oo = fit_moments(m.mean, m.variance, DISPERSION, OPEN_OPEN)
cc = fit_moments(m.mean, m.variance, DISPERSION, CLOSED_CLOSED)
lt = fit_lsq(t, m.e, TANKS)
t2 = np.arange(801) * 0.5             # 400 s
E2 = e_dispersion(t2 / 80.0, 12.0) / 80.0
ld = fit_lsq(t2, E2, DISPERSION)
print(json.dumps({
    "mean": m.mean, "variance": m.variance, "lam": m.tail_decay_time, "e_sum_dt": float(np.sum(m.e) * dt),
    "N_moments": nt.param, "Pe_oo": oo.param, "tau_oo": oo.tau, "Pe_cc": cc.param,
    "N_lsq": lt.param, "tau_lsq": lt.tau, "sse_lsq": lt.sse, "nfev_lsq": lt.nfev, "ok_lsq": lt.success,
    "Pe_lsq": ld.param, "tau_disp_lsq": ld.tau, "nfev_disp": ld.nfev, "ok_disp": ld.success}))
