"""Tests for ode.rk4_step_jacobian: linear system (degree-4 Taylor polynomial of exp(A h)), Van der Pol and
Lorenz step Jacobians vs complex-step reference data, central finite differences and dual numbers, fourth-order
convergence of the state and of the product of step Jacobians vs DOP853 flow data, the CSTR setting of the
source's RK4 parity test, and the Rust example."""
import math

from conftest import load_reference, python_example, rust_example

from agent_tools.autodiff.dual import Dual
from agent_tools.ode.rk4_step_jacobian import rk4_step, rk4_step_jacobian

TOL_LINEAR = 1e-13       # step and Jacobian vs degree-4 Taylor polynomial, relative to max |entry|
TOL_REF = 1e-11          # non-linear points vs complex-step reference data, relative to max |phi|
TOL_FD = 1e-6            # variational Jacobian vs central FD of the step (relative step 1e-5)
TOL_DUAL = 1e-12         # variational Jacobian vs dual-number derivative of the step
TOL_ORDER = 0.15         # observed convergence order within this distance of 4
TOL_PARITY_Y = 1e-6      # CSTR: RK4 (h = 1 s) vs DOP853, relative to |y| (truncation error of the chosen state)
TOL_PARITY_PHI = 1e-5    # CSTR: step Jacobian vs flow sensitivity matrix (truncation, not rounding)
TOL_RUST = 1e-12         # Python vs Rust example, relative to max |entry|


def rel_max(a, b):
    s = max(abs(x) for x in b)
    return max(abs(x - y) for x, y in zip(a, b)) / s


def vdp(mu):
    def f(t, y, d):
        d[0] = y[1]
        d[1] = mu * (1.0 - y[0] * y[0]) * y[1] - y[0]

    def j(t, y, jm):
        jm[:] = [0.0 * y[0], 1.0 + 0.0 * y[0], -(2.0 * mu * y[0] * y[1]) - 1.0, mu * (1.0 - y[0] * y[0])]
    return f, j


def lorenz():
    def f(t, y, d):
        d[0] = 10.0 * (y[1] - y[0])
        d[1] = y[0] * (28.0 - y[2]) - y[1]
        d[2] = y[0] * y[1] - (8.0 / 3.0) * y[2]

    def j(t, y, jm):
        jm[:] = [-10.0 + 0.0 * y[0], 10.0 + 0.0 * y[0], 0.0 * y[0], 28.0 - y[2], -1.0 + 0.0 * y[0], -y[0],
                 y[1], y[0], -(8.0 / 3.0) + 0.0 * y[0]]
    return f, j


def dual_phi(f, y0, h, m):
    n = len(y0)
    phi = [0.0] * (n * n)
    for j in range(n):
        r = rk4_step(f, 0.0, [Dual(y0[i], 1.0 if i == j else 0.0) for i in range(n)], h, m)
        for i in range(n):
            phi[i * n + j] = r.y1[i].d
    return phi


def fd_phi(f, y0, h, m):
    n = len(y0)
    phi = [0.0] * (n * n)
    for j in range(n):
        d = 1e-5 * max(abs(y0[j]), 1.0)
        yp, ym = list(y0), list(y0)
        yp[j] += d
        ym[j] -= d
        a, b = rk4_step(f, 0.0, yp, h, m).y1, rk4_step(f, 0.0, ym, h, m).y1
        for i in range(n):
            phi[i * n + j] = (a[i] - b[i]) / (2 * d)
    return phi


def check_points(name, pts, f, jac):
    for p in pts:
        y0, h, m = p["y0"], p["h"], p["n_sub"]
        r = rk4_step_jacobian(f, jac, 0.0, y0, h, m)
        assert r.success and r.nfev == 4 * m and r.njac == 4 * m
        assert rk4_step(f, 0.0, y0, h, m).y1 == r.y1
        ey, ep = rel_max(r.y1, p["y1"]), rel_max(r.phi, p["phi"])
        efd, edu = rel_max(fd_phi(f, y0, h, m), r.phi), rel_max(dual_phi(f, y0, h, m), r.phi)
        print(f"VERIFY rk4_step_jacobian py_{name}_{p['tag']}_nsub{m} y_vs_ref={ey:e} phi_vs_ref={ep:e} "
              f"phi_vs_fd={efd:e} phi_vs_dual={edu:e}")
        assert ey < TOL_REF and ep < TOL_REF and efd < TOL_FD and edu < TOL_DUAL


def test_linear_system_is_taylor_polynomial():
    c = load_reference("rk4_step_jacobian")["cases"]["linear"]
    a, n = c["a"], 3

    def f(t, y, d):
        for i in range(n):
            d[i] = sum(a[i * n + k] * y[k] for k in range(n))

    for s in c["steps"]:
        r = rk4_step_jacobian(f, lambda t, y, j: j.__setitem__(slice(None), a), 0.0, c["y0"], s["h"], s["n_sub"])
        ey, ep = rel_max(r.y1, s["y1"]), rel_max(r.phi, s["phi"])
        print(f"VERIFY rk4_step_jacobian py_linear_nsub{s['n_sub']} y_vs_taylor={ey:e} phi_vs_taylor={ep:e}")
        assert ey < TOL_LINEAR and ep < TOL_LINEAR


def test_van_der_pol_step_jacobian():
    c = load_reference("rk4_step_jacobian")["cases"]["vdp"]
    check_points("vdp", c["points"], *vdp(c["mu"]))


def test_lorenz_step_jacobian():
    check_points("lorenz", load_reference("rk4_step_jacobian")["cases"]["lorenz"]["points"], *lorenz())


def test_global_error_is_fourth_order():
    c = load_reference("rk4_step_jacobian")["cases"]["convergence"]
    f, j = vdp(c["mu"])
    prev = None
    for m in (10, 20, 40, 80, 160, 320):
        r = rk4_step_jacobian(f, j, 0.0, c["y0"], c["T"], m)
        ey, ep = rel_max(r.y1, c["y_T"]), rel_max(r.phi, c["phi_T"])
        if prev is None:
            print(f"VERIFY rk4_step_jacobian py_convergence_nsub{m} err_y={ey:e} err_phi={ep:e}")
        else:
            oy, op = math.log2(prev[0] / ey), math.log2(prev[1] / ep)
            print(f"VERIFY rk4_step_jacobian py_convergence_nsub{m} err_y={ey:e} err_phi={ep:e} order_y={oy:.3f} order_phi={op:.3f}")
            if m >= 40:
                assert abs(oy - 4) < TOL_ORDER and abs(op - 4) < TOL_ORDER
        prev = (ey, ep)


def test_cstr_parity_case():
    c = load_reference("rk4_step_jacobian")["cases"]["parity"]
    p = c["cstr"]

    def f(t, y, d):
        k = p["k0"] * math.exp(-p["er"] / y[1])
        d[0] = p["qv"] * (p["c0"] - y[0]) - k * y[0]
        d[1] = p["qv"] * (p["t0"] - y[1]) + p["dh"] * k * y[0] + p["ua"] * (p["tc"] - y[1])

    def jac(t, y, j):
        k = p["k0"] * math.exp(-p["er"] / y[1])
        dk = k * p["er"] / (y[1] * y[1])
        j[:] = [-p["qv"] - k, -dk * y[0], p["dh"] * k, -p["qv"] - p["ua"] + p["dh"] * dk * y[0]]

    r = rk4_step_jacobian(f, jac, 0.0, c["y0"], c["dt"], c["n_sub"])
    ey = max(abs(a - b) / abs(b) for a, b in zip(r.y1, c["y_ref"]))
    ep, ec = rel_max(r.phi, c["phi_ref"]), rel_max(r.phi, c["phi_complex_step"])
    print(f"VERIFY rk4_step_jacobian py_cstr_parity y_rel_vs_dop853={ey:e} phi_vs_flow_sensitivity={ep:e} phi_vs_complex_step={ec:e}")
    assert ey < TOL_PARITY_Y and ep < TOL_PARITY_PHI and ec < TOL_REF


def test_invalid_arguments_are_reported():
    f = lambda t, y, d: d.__setitem__(0, -y[0])  # noqa: E731
    j = lambda t, y, jm: jm.__setitem__(0, -1.0)  # noqa: E731
    for args in (([], 1.0, 1, 0.0), ([1.0], 1.0, 0, 0.0), ([1.0], float("nan"), 1, 0.0), ([1.0], 1.0, 1, float("inf"))):
        y0, h, m, t0 = args
        r = rk4_step_jacobian(f, j, t0, y0, h, m)
        assert not r.success and r.message and r.nfev == 0
    assert not rk4_step(f, 0.0, [1.0], 1.0, 0).success
    fw = rk4_step(f, 0.0, [1.0], 0.1, 4)
    bw = rk4_step(f, 0.1, fw.y1, -0.1, 4)
    print(f"VERIFY rk4_step_jacobian py_backward_roundtrip err={abs(bw.y1[0] - 1.0):e}")
    assert abs(bw.y1[0] - 1.0) < 1e-8


def test_python_vs_rust_example():
    py, rs = python_example("rk4_step_jacobian"), rust_example("rk4_step_jacobian")
    assert py["success"] and rs["success"] and py["message"] == rs["message"]
    assert py["nfev"] == rs["nfev"] and py["njac"] == rs["njac"]
    worst = max(rel_max(py[k], rs[k]) for k in ("y1", "phi", "dual_col0"))
    print(f"VERIFY rk4_step_jacobian python_vs_rust max_rel_diff={worst:e} nfev={py['nfev']}")
    assert worst < TOL_RUST
