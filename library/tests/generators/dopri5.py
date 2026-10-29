"""Generator for reference_data/dopri5.json.

References: scipy.integrate.solve_ivp(method="RK45") (accepted step times, nfev, final state: the
step-by-step comparison, same rtol and atol) and solve_ivp(method="DOP853") at rtol=1e-13, atol=1e-14
(tight reference solution). The Kepler reference is the analytic solution (Kepler's equation solved by
Newton iteration to 1e-15, in this script), cross-checked here against DOP853.
Cases (all dimensionless; t and y in the problems' own units):
  arenstorf : restricted three-body problem, Hairer-Norsett-Wanner I, Sec. II.0 eq. (0.1), mu = 0.012277471,
              y(0) = (0.994, 0, 0, -2.00158510637908252240537862224), period T = 17.0652165601579625588917206249.
  lotka     : x' = a x - b x y, y' = d x y - g y, a = 1, b = 0.5, d = 0.2, g = 0.6, y(0) = (4, 2),
              t in [0, 30]; conserved H = d x - g ln x + b y - a ln y.
  vdp1      : Van der Pol mu = 1, y(0) = (2, 0), t in [0, 20].
  oscillator: damped rotation y' = [[-0.1, 2],[-2, -0.1]] y (exact: exp(-0.1 t)(cos 2t, -sin 2t) for y0 = (1, 0)), t in [0, 20].
  kepler    : 2-D Kepler orbit, eccentricity 0.6, a = 1, mu = 1, from perihelion, t in [0, 4 pi] (two periods).
  stiffness : y' = -lam (y - cos t), y(0) = 0, t in [0, 2], lam in {10, 100, 1000, 10000}; SciPy RK45 step counts at rtol 1e-6.
RK45 runs: rtol in {1e-3, 1e-6, 1e-9}, atol = 1e-3 rtol (oscillator/kepler/lotka/vdp1/arenstorf).
Run: python library/tests/generators/dopri5.py
"""
import numpy as np
from scipy.integrate import solve_ivp

from _common import write

RTOLS = (1e-3, 1e-6, 1e-9)
ATOL_FACTOR = 1e-3
REF_RTOL, REF_ATOL = 1e-13, 1e-14
MU_A = 0.012277471
T_ARENSTORF = 17.0652165601579625588917206249
Y0_ARENSTORF = [0.994, 0.0, 0.0, -2.00158510637908252240537862224]
LV = dict(a=1.0, b=0.5, d=0.2, g=0.6)
ECC = 0.6


def arenstorf(t, y):
    m, mh = MU_A, 1.0 - MU_A
    d1 = ((y[0] + m) ** 2 + y[1] ** 2) ** 1.5
    d2 = ((y[0] - mh) ** 2 + y[1] ** 2) ** 1.5
    return [y[2], y[3],
            y[0] + 2 * y[3] - mh * (y[0] + m) / d1 - m * (y[0] - mh) / d2,
            y[1] - 2 * y[2] - mh * y[1] / d1 - m * y[1] / d2]


def lotka(t, y):
    return [LV["a"] * y[0] - LV["b"] * y[0] * y[1], LV["d"] * y[0] * y[1] - LV["g"] * y[1]]


def vdp1(t, y):
    return [y[1], (1 - y[0] ** 2) * y[1] - y[0]]


def oscillator(t, y):
    return [-0.1 * y[0] + 2.0 * y[1], -2.0 * y[0] - 0.1 * y[1]]


def kepler(t, y):
    r3 = (y[0] ** 2 + y[1] ** 2) ** 1.5
    return [y[2], y[3], -y[0] / r3, -y[1] / r3]


KEPLER_Y0 = [1 - ECC, 0.0, 0.0, np.sqrt((1 + ECC) / (1 - ECC))]


def kepler_exact(t):
    """Analytic orbit from perihelion: M = E - e sin E, x = cos E - e, y = sqrt(1-e^2) sin E."""
    M = t
    E = M
    for _ in range(100):
        E = E - (E - ECC * np.sin(E) - M) / (1 - ECC * np.cos(E))
    s = np.sqrt(1 - ECC ** 2)
    r = 1 - ECC * np.cos(E)
    return [np.cos(E) - ECC, s * np.sin(E), -np.sin(E) / r, s * np.cos(E) / r]


PROBLEMS = {
    "arenstorf": dict(f=arenstorf, y0=Y0_ARENSTORF, t_end=T_ARENSTORF),
    "lotka": dict(f=lotka, y0=[4.0, 2.0], t_end=30.0),
    "vdp1": dict(f=vdp1, y0=[2.0, 0.0], t_end=20.0),
    "oscillator": dict(f=oscillator, y0=[1.0, 0.0], t_end=20.0),
    "kepler": dict(f=kepler, y0=KEPLER_Y0, t_end=4 * np.pi),
}


def hamiltonian(y):
    return LV["d"] * y[0] - LV["g"] * np.log(y[0]) + LV["b"] * y[1] - LV["a"] * np.log(y[1])


cases = {}
for name, p in PROBLEMS.items():
    ref = solve_ivp(p["f"], (0.0, p["t_end"]), p["y0"], method="DOP853", rtol=REF_RTOL, atol=REF_ATOL)
    assert ref.success
    yref = ref.y[:, -1]
    if name == "oscillator":
        ex = [np.exp(-0.1 * p["t_end"]) * np.cos(2 * p["t_end"]), -np.exp(-0.1 * p["t_end"]) * np.sin(2 * p["t_end"])]
        assert np.max(np.abs(np.array(ex) - yref)) < 1e-11
    if name == "kepler":
        ex = kepler_exact(p["t_end"])
        assert np.max(np.abs(np.array(ex) - yref)) < 1e-10, np.max(np.abs(np.array(ex) - yref))
    if name == "arenstorf":
        assert np.max(np.abs(yref - np.array(p["y0"]))) < 1e-6   # periodic orbit
    rk = {}
    for rtol in RTOLS:
        atol = ATOL_FACTOR * rtol
        s = solve_ivp(p["f"], (0.0, p["t_end"]), p["y0"], method="RK45", rtol=rtol, atol=atol)
        assert s.success
        rk[f"{rtol:g}"] = {"rtol": rtol, "atol": atol, "t": s.t.tolist(), "y_final": s.y[:, -1].tolist(),
                           "nfev": int(s.nfev), "nsteps": int(len(s.t) - 1)}
    cases[name] = {"y0": list(map(float, p["y0"])), "t_end": float(p["t_end"]), "y_ref": yref.tolist(),
                   "dop853_nfev": int(ref.nfev), "rk45": rk}

# Lotka-Volterra invariant of the reference and of RK45
cases["lotka"]["H0"] = float(hamiltonian(cases["lotka"]["y0"]))
cases["lotka"]["H_ref"] = float(hamiltonian(cases["lotka"]["y_ref"]))
assert abs(cases["lotka"]["H0"] - cases["lotka"]["H_ref"]) < 1e-11

stiff = []
for lam in (10.0, 100.0, 1000.0, 10000.0):
    f = lambda t, y, lam=lam: [-lam * (y[0] - np.cos(t))]
    ref = solve_ivp(f, (0.0, 2.0), [0.0], method="Radau", rtol=1e-12, atol=1e-14)
    s = solve_ivp(f, (0.0, 2.0), [0.0], method="RK45", rtol=1e-6, atol=1e-9)
    assert ref.success and s.success
    stiff.append({"lam": lam, "y_ref": float(ref.y[0, -1]), "rk45_nsteps": int(len(s.t) - 1),
                  "rk45_nfev": int(s.nfev), "rk45_y_final": float(s.y[0, -1])})
cases["stiffness"] = {"rtol": 1e-6, "atol": 1e-9, "t_end": 2.0, "rows": stiff,
                      "reference": "scipy Radau rtol 1e-12 atol 1e-14"}

write("dopri5", "library/tests/generators/dopri5.py",
      {"reference_methods": "scipy solve_ivp RK45 (step sequence) and DOP853 rtol 1e-13 atol 1e-14 (solution)",
       "rtols": list(RTOLS), "atol_factor": ATOL_FACTOR, "arenstorf_mu": MU_A, "lotka_volterra": LV,
       "kepler_eccentricity": ECC},
      "Units: dimensionless. rk45.t is the accepted step sequence of scipy RK45 with automatic first step "
      "(t[0] = 0); nfev is SciPy's count including the initial f and the starting-step probe. "
      "Arenstorf constants from Hairer, Norsett, Wanner, Solving ODEs I, 2nd ed., Sec. II.0.",
      cases)
