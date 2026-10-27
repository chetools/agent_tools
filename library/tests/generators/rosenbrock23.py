"""Generator for reference_data/rosenbrock23.json.

References: scipy.integrate.solve_ivp(method="Radau") at rtol=1e-12, atol=1e-14, with the analytic
Jacobian. Each reference is cross-checked inside this script against a second integration
(LSODA at rtol 1e-12 / atol 1e-14) and, for Robertson, against the conserved sum y1+y2+y3 = 1.
Cases:
  robertson : y1' = -0.04 y1 + 1e4 y2 y3; y2' = 0.04 y1 - 1e4 y2 y3 - 3e7 y2^2; y3' = 3e7 y2^2;
              y(0) = (1, 0, 0); end times 0.4, 40, 4e3, 1e5 (Robertson/Hairer-Wanner values).
  vdp1000   : y1' = y2; y2' = mu (1 - y1^2) y2 - y1, mu = 1000, y(0) = (2, 0); end times 1000, 3000
              (the period is about (3 - 2 ln 2) mu = 1614, so 3000 spans more than one period).
  robertson_sens : dy/dk1 at t = 40 (k1 = 0.04), from the forward sensitivity equations
              solved with Radau (independent of the integrator under test); checked here against
              central differences of Radau (relative step 1e-4).
  tolerance_study : SciPy Radau and BDF at rtol in {1e-3, 1e-5, 1e-7}: nfev, njev, nlu, steps and
              the error against the reference (same error measure as the tests).
Error measure: max_i |y_i - ref_i| / (|ref_i| + atol_i), atol = 1e-6 rtol (Robertson), 1e-3 rtol (VdP).
Run: python library/tests/generators/rosenbrock23.py
"""
import numpy as np
from scipy.integrate import solve_ivp

from _common import write

RTOL_REF, ATOL_REF = 1e-12, 1e-14
RTOLS = (1e-3, 1e-5, 1e-7)
ATOL_FACTOR = {"robertson": 1e-6, "vdp1000": 1e-3}
MU = 1000.0
K1 = 0.04


def rob(t, y, k1=K1):
    return [-k1 * y[0] + 1e4 * y[1] * y[2], k1 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] ** 2, 3e7 * y[1] ** 2]


def rob_jac(t, y, k1=K1):
    return np.array([[-k1, 1e4 * y[2], 1e4 * y[1]],
                     [k1, -1e4 * y[2] - 6e7 * y[1], -1e4 * y[1]],
                     [0.0, 6e7 * y[1], 0.0]])


def vdp(t, y):
    return [y[1], MU * (1 - y[0] ** 2) * y[1] - y[0]]


def vdp_jac(t, y):
    return np.array([[0.0, 1.0], [-2 * MU * y[0] * y[1] - 1, MU * (1 - y[0] ** 2)]])


PROBLEMS = {
    "robertson": dict(f=rob, jac=rob_jac, y0=[1.0, 0.0, 0.0], t_ends=[0.4, 40.0, 4e3, 1e5]),
    "vdp1000": dict(f=vdp, jac=vdp_jac, y0=[2.0, 0.0], t_ends=[1000.0, 3000.0]),
}


def run(p, t_end, method, rtol, atol):
    s = solve_ivp(p["f"], (0.0, t_end), p["y0"], method=method, rtol=rtol, atol=atol,
                  jac=p["jac"] if method in ("Radau", "BDF") else None, dense_output=False)
    assert s.success, s.message
    return s


def err_measure(y, ref, atol):
    y, ref = np.asarray(y), np.asarray(ref)
    return float(np.max(np.abs(y - ref) / (np.abs(ref) + atol)))


cases = {}
for name, p in PROBLEMS.items():
    refs = []
    for te in p["t_ends"]:
        y = run(p, te, "Radau", RTOL_REF, ATOL_REF).y[:, -1]
        y2 = run(p, te, "LSODA", RTOL_REF, ATOL_REF).y[:, -1]
        cross = err_measure(y2, y, 1e-12)
        # tolerance of the cross check: the stiff problems amplify phase errors, so loose-ish
        assert cross < 1e-6, (name, te, cross)
        if name == "robertson":
            assert abs(y.sum() - 1.0) < 1e-10
        refs.append({"t_end": te, "y": y.tolist(), "cross_check_lsoda": cross})
    cases[name] = {"y0": p["y0"], "refs": refs}

# tolerance study at the last end time of each problem
study = {}
for name, p in PROBLEMS.items():
    te = p["t_ends"][-1] if name == "robertson" else p["t_ends"][-1]
    ref = np.array(cases[name]["refs"][-1]["y"])
    rows = []
    for rtol in RTOLS:
        atol = ATOL_FACTOR[name] * rtol
        row = {"rtol": rtol, "atol": atol, "t_end": te}
        for method in ("Radau", "BDF"):
            s = run(p, te, method, rtol, atol)
            row[method] = {"nfev": int(s.nfev), "njev": int(s.njev), "nlu": int(s.nlu),
                           "nsteps": int(s.t.size - 1), "err": err_measure(s.y[:, -1], ref, atol)}
        rows.append(row)
    study[name] = rows
cases["tolerance_study"] = study


# sensitivity dy/dk1 at t = 40 by forward sensitivity equations (3 states + 3 sensitivities)
def aug(t, z):
    y, s = z[:3], z[3:]
    dfdk = np.array([-y[0], y[0], 0.0])
    return np.concatenate([rob(t, y), rob_jac(t, y) @ s + dfdk])


def aug_jac(t, z):
    J = rob_jac(t, z[:3])
    A = np.zeros((6, 6))
    A[:3, :3] = J
    A[3:, 3:] = J
    h = 1e-7  # d(J s)/dy part by central differences (generator only; Radau tolerates a rough Jacobian)
    for k in range(3):
        e = np.zeros(3); e[k] = h
        A[3:, k] = (rob_jac(t, z[:3] + e) - rob_jac(t, z[:3] - e)) @ z[3:] / (2 * h)
    return A


te = 40.0
sz = solve_ivp(aug, (0, te), [1, 0, 0, 0, 0, 0], method="Radau", rtol=RTOL_REF, atol=ATOL_REF, jac=aug_jac)
assert sz.success
sens = sz.y[3:, -1]
rel_h = 1e-4
yp = solve_ivp(lambda t, y: rob(t, y, K1 * (1 + rel_h)), (0, te), [1, 0, 0], method="Radau",
               rtol=RTOL_REF, atol=ATOL_REF, jac=lambda t, y: rob_jac(t, y, K1 * (1 + rel_h))).y[:, -1]
ym = solve_ivp(lambda t, y: rob(t, y, K1 * (1 - rel_h)), (0, te), [1, 0, 0], method="Radau",
               rtol=RTOL_REF, atol=ATOL_REF, jac=lambda t, y: rob_jac(t, y, K1 * (1 - rel_h))).y[:, -1]
fd = (yp - ym) / (2 * rel_h * K1)
fd_check = float(np.max(np.abs(fd - sens) / (np.abs(sens) + 1e-12)))
assert fd_check < 1e-5, fd_check
cases["robertson_sens"] = {"k1": K1, "t_end": te, "dy_dk1": sens.tolist(), "fd_check": fd_check}

write("rosenbrock23", "library/tests/generators/rosenbrock23.py",
      {"reference_method": "scipy.integrate.solve_ivp Radau, analytic Jacobian", "rtol_ref": RTOL_REF,
       "atol_ref": ATOL_REF, "cross_check": "LSODA rtol 1e-12 atol 1e-14", "study_rtols": list(RTOLS),
       "study_atol_factor": ATOL_FACTOR, "mu": MU, "robertson_k1": K1},
      "Units: dimensionless (time in the problems' own units). Study entries are SciPy Radau/BDF counts "
      "(nfev excludes Jacobian calls) and errors in the measure max_i |y-ref|/(|ref|+atol). "
      "robertson_sens.dy_dk1 is d y(40)/d k1 (k1 = 0.04) from forward sensitivity equations, checked "
      "against central differences of Radau in this script.",
      cases)
