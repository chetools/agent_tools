"""Generator for reference_data/rk4_step_jacobian.json.

Independent references (none of them uses the variational recursion under test):
  * step Jacobian of the discrete RK4 step: complex-step differentiation Im f(y + i 1e-30 e_j) / 1e-30
    of a separate NumPy RK4 (exact to rounding for analytic f); the step itself is the real part.
  * linear system y' = A y: degree-4 Taylor polynomial P4(Ah) = sum_{k<=4} (Ah)^k / k! (the RK4 step
    of a linear system equals P4 exactly); for n_sub substeps P4(Ah/m)^m. Cross-check here: ||P4 - expm(Ah)||
    is O(h^5), and the Jacobian equals the matrix.
  * flow values and flow sensitivity matrices (convergence and parity cases): scipy DOP853 at
    rtol 1e-13, atol 1e-15 on the augmented system (state + n^2 sensitivities, analytic Jacobian written
    separately in this file).
Cases:
  linear     : A (3x3, non-normal, eigenvalues in the left half plane), y0, h = 0.3, n_sub in {1, 3}.
  vdp        : Van der Pol y1' = y2, y2' = mu (1 - y1^2) y2 - y1, mu = 5; states on the limit cycle
               before, at and after the maximum of |y2| (the steep fast transition), h = 0.05.
  lorenz     : sigma = 10, rho = 28, beta = 8/3; states before, at and after a sign change of x
               (lobe switch), h = 0.02, n_sub = 2.
  convergence: Van der Pol mu = 1, y(0) = (2, 0), T = 2: y(T) and the flow sensitivity matrix dy(T)/dy(0).
  parity     : first-order exothermic CSTR (generic, Arrhenius; Seborg-style constants in the file) over
               dt = 10 s with 10 substeps (h = 1 s), the setting of the source's RK4 parity test (there the
               reference was an adaptive solve_ivp at rtol 1e-6): y(dt) and flow sensitivity matrix.
Run: python library/tests/generators/rk4_step_jacobian.py
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from _common import write

# ---------------- systems (NumPy, complex-capable) and analytic Jacobians ----------------
A_LIN = np.array([[-1.0, 2.0, 0.0], [0.0, -3.0, 1.5], [0.5, 0.0, -2.0]])
Y0_LIN = np.array([1.0, -0.5, 2.0])
MU_POINTS, MU_CONV = 5.0, 1.0
SIG, RHO, BET = 10.0, 28.0, 8.0 / 3.0
# generic exothermic CSTR; time in s, C in mol/L, T in K (constants after Seborg, Edgar, Mellichamp:
# q/V = 1/60 per s, k0 = 7.2e10/min = 1.2e9/s, E/R = 8750 K, -dH/(rho cp) = 209 K L/mol, UA/(V rho cp) = 2.09/min)
CSTR = dict(qv=1.0 / 60.0, c0=1.0, t0=350.0, tc=300.0, k0=1.2e9, er=8750.0, dh=209.0, ua=2.09 / 60.0)


def f_lin(t, y): return A_LIN @ y
def j_lin(t, y): return A_LIN
def f_vdp(mu): return lambda t, y: np.array([y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]])
def j_vdp(mu): return lambda t, y: np.array([[0.0, 1.0], [-2 * mu * y[0] * y[1] - 1.0, mu * (1 - y[0] ** 2)]])
def f_lor(t, y): return np.array([SIG * (y[1] - y[0]), y[0] * (RHO - y[2]) - y[1], y[0] * y[1] - BET * y[2]])
def j_lor(t, y): return np.array([[-SIG, SIG, 0.0], [RHO - y[2], -1.0, -y[0]], [y[1], y[0], -BET]])


def f_cstr(t, y):
    p = CSTR
    k = p["k0"] * np.exp(-p["er"] / y[1])
    return np.array([p["qv"] * (p["c0"] - y[0]) - k * y[0],
                     p["qv"] * (p["t0"] - y[1]) + p["dh"] * k * y[0] + p["ua"] * (p["tc"] - y[1])])


def j_cstr(t, y):
    p = CSTR
    k = p["k0"] * np.exp(-p["er"] / y[1])
    dk = k * p["er"] / y[1] ** 2
    return np.array([[-p["qv"] - k, -dk * y[0]],
                     [p["dh"] * k, -p["qv"] - p["ua"] + p["dh"] * dk * y[0]]])


# ---------------- independent references ----------------
def rk4_np(f, t, y, h, n_sub):
    hs = h / n_sub
    for s in range(n_sub):
        ts = t + s * hs
        k1 = f(ts, y); k2 = f(ts + hs / 2, y + hs / 2 * k1)
        k3 = f(ts + hs / 2, y + hs / 2 * k2); k4 = f(ts + hs, y + hs * k3)
        y = y + hs / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def step_complex(f, t, y0, h, n_sub):
    """(y1, Phi) with Phi by complex-step differentiation of the discrete step."""
    n = len(y0)
    y1 = rk4_np(f, t, np.array(y0, dtype=complex), h, n_sub).real
    phi = np.zeros((n, n))
    for j in range(n):
        yc = np.array(y0, dtype=complex)
        yc[j] += 1e-30j
        phi[:, j] = rk4_np(f, t, yc, h, n_sub).imag / 1e-30
    return y1, phi


def flow(f, jac, y0, T):
    """Flow value and sensitivity matrix dy(T)/dy(0) by DOP853 on the augmented system."""
    n = len(y0)

    def aug(t, z):
        y, S = z[:n], z[n:].reshape(n, n)
        return np.concatenate([f(t, y), (jac(t, y) @ S).ravel()])

    s = solve_ivp(aug, (0, T), np.concatenate([y0, np.eye(n).ravel()]), method="DOP853", rtol=1e-13, atol=1e-15)
    assert s.success
    return s.y[:n, -1], s.y[n:, -1].reshape(n, n)


def fd_check(f, jac, t, y, tol=1e-6):
    """The analytic Jacobian used above agrees with central differences of f (guards the generator)."""
    n = len(y)
    J = np.zeros((n, n))
    for j in range(n):
        d = np.zeros(n); d[j] = 1e-6 * max(1.0, abs(y[j]))
        J[:, j] = (f(t, y + d) - f(t, y - d)) / (2 * d[j])
    assert np.max(np.abs(J - jac(t, y))) < tol * max(1.0, np.max(np.abs(J))), (J, jac(t, y))


def point(f, y0, h, n_sub, **extra):
    y1, phi = step_complex(f, 0.0, y0, h, n_sub)
    return {"y0": list(map(float, y0)), "h": h, "n_sub": n_sub, "y1": y1.tolist(), "phi": phi.ravel().tolist(), **extra}


cases = {}

# linear: P4(Ah)^m
def p4(M):
    n = M.shape[0]
    P, term = np.eye(n), np.eye(n)
    for k in range(1, 5):
        term = term @ M / k
        P = P + term
    return P

lin = []
for h, m in ((0.3, 1), (0.3, 3)):
    P = np.linalg.matrix_power(p4(A_LIN * h / m), m)
    y1c, phic = step_complex(f_lin, 0.0, Y0_LIN, h, m)
    assert np.max(np.abs(P - phic)) < 1e-14 and np.max(np.abs(P @ Y0_LIN - y1c)) < 1e-14
    assert np.max(np.abs(P - expm(A_LIN * h))) < 5 * np.linalg.norm(A_LIN * h / m, 2) ** 5 / 120 * m + 1e-14
    lin.append({"h": h, "n_sub": m, "y1": (P @ Y0_LIN).tolist(), "phi": P.ravel().tolist()})
cases["linear"] = {"a": A_LIN.ravel().tolist(), "y0": Y0_LIN.tolist(), "steps": lin}

# Van der Pol mu = 5: states around the steepest part of the limit cycle
fv, jv = f_vdp(MU_POINTS), j_vdp(MU_POINTS)
fd_check(fv, jv, 0.0, np.array([1.3, -2.0]))
sol = solve_ivp(fv, (0, 40), [2.0, 0.0], method="DOP853", rtol=1e-12, atol=1e-14, dense_output=True)
tt = np.linspace(20, 36, 40001)
yy = sol.sol(tt)
k_star = 20000 + int(np.argmax(np.abs(yy[1, 20000:])))   # only the later part: on the periodic orbit
t_star = tt[k_star]
pts = []
for dt_off, tag in ((-0.1, "before"), (0.0, "at_max_abs_y2"), (0.1, "after")):
    y0 = sol.sol(t_star + dt_off)
    for n_sub in (1, 4):
        pts.append(point(fv, y0, 0.05, n_sub, tag=tag, t_orbit=t_star + dt_off, speed=float(abs(y0[1]))))
cases["vdp"] = {"mu": MU_POINTS, "points": pts}

# Lorenz: around a sign change of x
fd_check(f_lor, j_lor, 0.0, np.array([1.0, 2.0, 3.0]))
sl = solve_ivp(f_lor, (0, 60), [1.0, 1.0, 1.0], method="DOP853", rtol=1e-12, atol=1e-14, dense_output=True)
tt = np.linspace(20, 60, 400001)
xx = sl.sol(tt)[0]
idx = np.where(np.sign(xx[1:]) != np.sign(xx[:-1]))[0]
t_c = tt[idx[len(idx) // 2]]
pts = []
for dt_off, tag in ((-0.3, "before"), (0.0, "at_sign_change_of_x"), (0.3, "after")):
    y0 = sl.sol(t_c + dt_off)
    pts.append(point(f_lor, y0, 0.02, 2, tag=tag, t_orbit=t_c + dt_off))
cases["lorenz"] = {"sigma": SIG, "rho": RHO, "beta": BET, "points": pts}

# convergence: VdP mu = 1, T = 2
fc, jc = f_vdp(MU_CONV), j_vdp(MU_CONV)
yT, ST = flow(fc, jc, np.array([2.0, 0.0]), 2.0)
yT2 = solve_ivp(fc, (0, 2.0), [2.0, 0.0], method="DOP853", rtol=1e-13, atol=1e-15).y[:, -1]
assert np.max(np.abs(yT - yT2)) < 1e-11
cases["convergence"] = {"mu": MU_CONV, "y0": [2.0, 0.0], "T": 2.0, "y_T": yT.tolist(), "phi_T": ST.ravel().tolist()}

# parity: CSTR over dt = 10 s, 10 substeps
fd_check(f_cstr, j_cstr, 0.0, np.array([0.5, 380.0]))
y0 = np.array([0.5, 355.0])
yT, ST = flow(f_cstr, j_cstr, y0, 10.0)
y1, phi = step_complex(f_cstr, 0.0, y0, 10.0, 10)
cases["parity"] = {"cstr": CSTR, "y0": y0.tolist(), "dt": 10.0, "n_sub": 10, "y_ref": yT.tolist(), "phi_ref": ST.ravel().tolist(),
                   "y1_np_rk4": y1.tolist(), "phi_complex_step": phi.ravel().tolist()}

write("rk4_step_jacobian", "library/tests/generators/rk4_step_jacobian.py",
      {"step_jacobian_reference": "complex-step differentiation of a NumPy RK4", "linear_reference": "degree-4 Taylor polynomial of expm(A h) (matrix_power for substeps)",
       "flow_reference": "scipy solve_ivp DOP853 rtol 1e-13 atol 1e-15 on state + sensitivity system", "vdp_mu_points": MU_POINTS,
       "vdp_mu_convergence": MU_CONV, "lorenz": [SIG, RHO, BET], "cstr": CSTR},
      "Units: dimensionless except the CSTR case (t in s, C in mol/L, T in K). Each point stores y0, h, n_sub, "
      "y1 and phi (row-major, phi[i*n+k] = d y1_i / d y0_k). 'convergence' and 'parity' store the exact flow "
      "value and sensitivity matrix (not a discrete step); y1_np_rk4 and phi_complex_step are the NumPy RK4 and its "
      "complex-step Jacobian for the parity setting.", cases)
