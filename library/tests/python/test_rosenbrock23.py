"""Tests for ode.rosenbrock23: exact solutions, SciPy Radau reference data (Robertson, Van der Pol
mu = 1000), dual-number sensitivities vs finite differences / independent references, and Rust."""
import math

from conftest import load_reference, python_example, rust_example

from agent_tools.autodiff.dual import Dual, val
from agent_tools.ode.rosenbrock23 import rosenbrock23

# Achieved error may exceed the requested rtol by at most this factor (the tolerance controls the
# local error of a 2nd-order method, so the global error is larger).
TOL_ERR_OVER_RTOL = 500.0
TOL_AD_EXACT = 1e-4      # AD derivative vs exact -t exp(-k t), relative, rtol 1e-8
TOL_AD_FD = 1e-3         # AD derivative vs central FD of the same integrator, relative
TOL_AD_ROB = 1e-4        # AD Robertson dy/dk1 vs forward-sensitivity reference, relative to max|dy|
TOL_RUST_Y = 1e-12       # Python vs Rust example: relative difference in y, dy (same operation order)


def measure(y, ref, atol):
    return max(abs(a - b) / (abs(b) + atol) for a, b in zip(y, ref))


def rob_f(k1):
    def f(t, y, d):
        d[0] = -(k1 * y[0]) + 1e4 * y[1] * y[2]
        d[1] = k1 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1]
        d[2] = 3e7 * y[1] * y[1]
    return f


def rob_j(k1):
    def j(t, y, jm):
        z = k1 * 0.0
        jm[:] = [-k1, 1e4 * y[2], 1e4 * y[1],
                 k1, -(1e4 * y[2]) - 6e7 * y[1], -(1e4 * y[1]),
                 z, 6e7 * y[1], z]
    return j


MU = 1000.0


def vdp_f(t, y, d):
    d[0] = y[1]
    d[1] = MU * (1.0 - y[0] * y[0]) * y[1] - y[0]


def vdp_j(t, y, jm):
    jm[:] = [0.0, 1.0, -2.0 * MU * y[0] * y[1] - 1.0, MU * (1.0 - y[0] * y[0])]


def run_rob(t_end, rtol, atol):
    return rosenbrock23(rob_f(0.04), rob_j(0.04), (0.0, t_end), [1.0, 0.0, 0.0], rtol=rtol, atol=atol, h0=1e-6)


def run_vdp(t_end, rtol, atol):
    return rosenbrock23(vdp_f, vdp_j, (0.0, t_end), [2.0, 0.0], rtol=rtol, atol=atol, h0=1e-3)


def test_exponential_decay_vs_exact():
    k = 2.5
    for rtol in (1e-4, 1e-6, 1e-8):
        r = rosenbrock23(lambda t, y, d: d.__setitem__(0, -k * y[0]),
                         lambda t, y, j: j.__setitem__(0, -k), (0.0, 1.0), [1.0], rtol=rtol, atol=1e-14)
        assert r.success
        e = abs(r.y_final[0] - math.exp(-k)) / math.exp(-k)
        print(f"VERIFY rosenbrock23 py_decay_rtol{rtol:e} rel_err_vs_exact={e:e} nsteps={r.nsteps} nrejected={r.nrejected}")
        assert r.t[-1] == 1.0 and abs(r.dydt[-1][0] + k * r.y_final[0]) < 1e-15
        assert r.nfev == 1 + 2 * (r.nsteps + r.nrejected) and r.nlu == r.nsteps + r.nrejected
        assert r.njac == r.nsteps
        assert e < TOL_ERR_OVER_RTOL * rtol


def test_nonautonomous_vs_exact():
    lam = 1000.0

    def exact(t):
        c = 1.0 - lam * lam / (1.0 + lam * lam)
        return (lam * lam * math.cos(t) + lam * math.sin(t)) / (1.0 + lam * lam) + c * math.exp(-lam * t)

    f = lambda t, y, d: d.__setitem__(0, -lam * (y[0] - math.cos(t)))  # noqa: E731
    j = lambda t, y, jm: jm.__setitem__(0, -lam)  # noqa: E731
    g = lambda t, y, o: o.__setitem__(0, -lam * math.sin(t))  # noqa: E731
    kw = dict(rtol=1e-8, atol=1e-12, h0=1e-6)
    ra = rosenbrock23(f, j, (0.0, 3.0), [1.0], dfdt=g, **kw)
    rf = rosenbrock23(f, j, (0.0, 3.0), [1.0], fd_dfdt=True, **kw)
    r0 = rosenbrock23(f, j, (0.0, 3.0), [1.0], **kw)
    assert ra.success and rf.success and r0.success
    ex = exact(3.0)
    ea, ef, e0 = (abs(r.y_final[0] - ex) / abs(ex) for r in (ra, rf, r0))
    print(f"VERIFY rosenbrock23 py_nonautonomous rel_err_analytic_dfdt={ea:e} fd_dfdt={ef:e} zero_dfdt={e0:e}")
    print(f"VERIFY rosenbrock23 py_nonautonomous nsteps analytic={ra.nsteps} fd={rf.nsteps} zero={r0.nsteps}")
    assert ea < 1e-6 and ef < 1e-6


def test_robertson_vs_radau():
    rtol, af = 1e-6, 1e-6
    for ref in load_reference("rosenbrock23")["cases"]["robertson"]["refs"]:
        r = run_rob(ref["t_end"], rtol, af * rtol)
        assert r.success, r.message
        e = measure(r.y_final, ref["y"], af * rtol)
        print(f"VERIFY rosenbrock23 py_robertson_t{ref['t_end']:e} err_vs_radau={e:e} nsteps={r.nsteps} nrejected={r.nrejected} nfev={r.nfev} nlu={r.nlu}")
        assert e < TOL_ERR_OVER_RTOL * rtol
        assert abs(sum(r.y_final) - 1.0) < 1e-8


def test_van_der_pol_and_tolerance_study():
    data = load_reference("rosenbrock23")["cases"]
    rtol = 1e-7
    for ref in data["vdp1000"]["refs"]:
        r = run_vdp(ref["t_end"], rtol, 1e-3 * rtol)
        assert r.success, r.message
        e = measure(r.y_final, ref["y"], 1e-3 * rtol)
        print(f"VERIFY rosenbrock23 py_vdp1000_t{ref['t_end']:e}_rtol1e-7 err_vs_radau={e:e} nsteps={r.nsteps}")
        assert e < TOL_ERR_OVER_RTOL * rtol
    for name, af, run in (("robertson", 1e-6, run_rob), ("vdp1000", 1e-3, run_vdp)):
        last = data[name]["refs"][-1]
        for row in data["tolerance_study"][name]:
            rt, atol = row["rtol"], af * row["rtol"]
            r = run(last["t_end"], rt, atol)
            assert r.success
            e = measure(r.y_final, last["y"], atol)
            print(f"VERIFY rosenbrock23 py_study_{name}_rtol{rt:e} err={e:e} err/rtol={e / rt:e} nfev={r.nfev} "
                  f"nsteps={r.nsteps} nrejected={r.nrejected} nlu={r.nlu} njac={r.njac} | "
                  f"Radau err={row['Radau']['err']:e} nfev={row['Radau']['nfev']} nsteps={row['Radau']['nsteps']} | "
                  f"BDF err={row['BDF']['err']:e} nfev={row['BDF']['nfev']} nsteps={row['BDF']['nsteps']}")
            assert e < TOL_ERR_OVER_RTOL * rt


def decay_dual(k, rtol, t_end):
    r = rosenbrock23(lambda t, y, d: d.__setitem__(0, -(k * y[0])), lambda t, y, j: j.__setitem__(0, -k),
                     (0.0, t_end), [Dual(1.0)], rtol=rtol, atol=1e-14)
    assert r.success
    return r.y_final[0]


def test_ad_sensitivity_decay_and_robertson():
    k, te = 2.5, 3.0
    for rtol in (1e-6, 1e-8):
        y = decay_dual(Dual(k, 1.0), rtol, te)
        exact = -te * math.exp(-k * te)
        h = 1e-3 * k
        fd = (val(decay_dual(Dual(k + h), rtol, te)) - val(decay_dual(Dual(k - h), rtol, te))) / (2 * h)
        e_ex, e_fd = abs(y.d - exact) / abs(exact), abs(y.d - fd) / abs(exact)
        plain = val(decay_dual(Dual(k), rtol, te))
        print(f"VERIFY rosenbrock23 py_ad_decay_rtol{rtol:e} rel_err_vs_exact={e_ex:e} rel_diff_vs_fd={e_fd:e} value_changed_by_ad={abs(y.v - plain):e}")
        if rtol == 1e-8:
            assert e_ex < TOL_AD_EXACT and e_fd < TOL_AD_FD
    sref = load_reference("rosenbrock23")["cases"]["robertson_sens"]["dy_dk1"]
    rtol = 1e-7

    def go(k1):
        return rosenbrock23(rob_f(k1), rob_j(k1), (0.0, 40.0), [Dual(1.0), Dual(0.0), Dual(0.0)],
                            rtol=rtol, atol=1e-15, h0=1e-6)

    run = go(Dual(0.04, 1.0))
    assert run.success, run.message
    dy = [v.d for v in run.y_final]
    smax = max(abs(s) for s in sref)
    e_ref = max(abs(a - b) for a, b in zip(dy, sref)) / smax
    h = 0.04 * 1e-3
    yp, ym = go(Dual(0.04 + h)).y_final, go(Dual(0.04 - h)).y_final
    e_fd = max(abs((val(yp[i]) - val(ym[i])) / (2 * h) - dy[i]) for i in range(3)) / smax
    print(f"VERIFY rosenbrock23 py_ad_robertson_dk1 rel_err_vs_forward_sens={e_ref:e} rel_diff_vs_fd={e_fd:e} nsteps_dual={run.nsteps} nrejected_dual={run.nrejected}")
    assert e_ref < TOL_AD_ROB and e_fd < TOL_AD_FD


def test_failures_are_reported():
    f = lambda t, y, d: d.__setitem__(0, y[0] * y[0])  # noqa: E731
    j = lambda t, y, jm: jm.__setitem__(0, 2.0 * y[0])  # noqa: E731
    r = rosenbrock23(f, j, (0.0, 2.0), [1.0])          # finite-time blow-up at t = 1
    assert not r.success and r.message
    assert not rosenbrock23(f, j, (1.0, 1.0), [1.0]).success
    assert not rosenbrock23(f, j, (0.0, 1.0), []).success
    r = rosenbrock23(lambda t, y, d: d.__setitem__(0, -y[0]), lambda t, y, jm: jm.__setitem__(0, -1.0),
                     (0.0, 100.0), [1.0], max_steps=5)
    assert not r.success and "max_steps" in r.message
    assert not rosenbrock23(f, j, (0.0, 1.0), [1.0], atol_vec=[1e-8, 1e-8]).success
    print(f"VERIFY rosenbrock23 py_blowup message={rosenbrock23(f, j, (0.0, 2.0), [1.0]).message!r}")


def test_per_component_atol_is_used():
    def f(t, y, d):
        d[0] = -y[0]
        d[1] = y[0] * 1e-9 - 10.0 * y[1]

    j = lambda t, y, jm: jm.__setitem__(slice(None), [-1.0, 0.0, 1e-9, -10.0])  # noqa: E731
    exact = 1e-9 / 9.0 * (math.exp(-1.0) - math.exp(-10.0))
    loose = rosenbrock23(f, j, (0.0, 1.0), [1.0, 0.0], rtol=1e-6, atol=1e-6)
    tight = rosenbrock23(f, j, (0.0, 1.0), [1.0, 0.0], rtol=1e-6, atol_vec=[1e-6, 1e-16])
    el, et = (abs(r.y_final[1] - exact) / exact for r in (loose, tight))
    print(f"VERIFY rosenbrock23 py_atol_vec err_loose={el:e} nsteps={loose.nsteps} err_per_component={et:e} nsteps={tight.nsteps}")
    assert et < 1e-4 and et < el


def test_python_vs_rust_example():
    py, rs = python_example("rosenbrock23"), rust_example("rosenbrock23")
    assert py["success"] and rs["success"] and py["message"] == rs["message"]
    for key in ("nsteps", "nrejected", "nfev", "njac", "nlu"):
        assert py[key] == rs[key], key
    assert py["t_final"] == rs["t_final"] == 40.0
    worst = 0.0
    for key in ("t_mid", "y_mid", "y", "dy_dk1"):
        a = py[key] if isinstance(py[key], list) else [py[key]]
        b = rs[key] if isinstance(rs[key], list) else [rs[key]]
        scale = max(abs(x) for x in b)
        worst = max(worst, max(abs(x - y) for x, y in zip(a, b)) / scale)
    print(f"VERIFY rosenbrock23 python_vs_rust max_rel_diff={worst:e} nsteps={py['nsteps']} nfev={py['nfev']}")
    assert worst < TOL_RUST_Y
