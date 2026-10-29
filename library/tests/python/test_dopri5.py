"""Tests for ode.dopri5: step-by-step comparison with SciPy RK45, exact solutions and invariants,
DOP853 reference data, Arenstorf and a stiffness study (step-size limitation), failures, and Rust."""
import math

from conftest import load_reference, python_example, rust_example

from agent_tools.ode.dopri5 import dopri5

TOL_T_SEQ = 1e-6          # accepted step times vs SciPy RK45, absolute (round-off feeds back through the controller)
TOL_Y_VS_RK45 = 1e-6      # final state vs SciPy RK45, relative to max|y|
TOL_ERR_OVER_RTOL = 1e3   # achieved final error may exceed rtol by this factor (global vs local error)
TOL_ERR_OVER_RTOL_ARENSTORF = 2e4   # close approaches amplify local errors in the undamped orbit
TOL_RUST_Y = 1e-9         # Python vs Rust example: relative difference (powf vs ** differ by an ulp; fed back by the controller)

MU = 0.012277471
ECC = 0.6


def arenstorf(t, y, d):
    mh = 1.0 - MU
    d1 = ((y[0] + MU) ** 2 + y[1] * y[1]) ** 1.5
    d2 = ((y[0] - mh) ** 2 + y[1] * y[1]) ** 1.5
    d[0], d[1] = y[2], y[3]
    d[2] = y[0] + 2.0 * y[3] - mh * (y[0] + MU) / d1 - MU * (y[0] - mh) / d2
    d[3] = y[1] - 2.0 * y[2] - mh * y[1] / d1 - MU * y[1] / d2


def lotka(t, y, d):
    d[0] = y[0] - 0.5 * y[0] * y[1]
    d[1] = 0.2 * y[0] * y[1] - 0.6 * y[1]


def ham(y):
    return 0.2 * y[0] - 0.6 * math.log(y[0]) + 0.5 * y[1] - math.log(y[1])


def vdp1(t, y, d):
    d[0], d[1] = y[1], (1.0 - y[0] * y[0]) * y[1] - y[0]


def oscillator(t, y, d):
    d[0] = -0.1 * y[0] + 2.0 * y[1]
    d[1] = -2.0 * y[0] - 0.1 * y[1]


def kepler(t, y, d):
    r3 = (y[0] * y[0] + y[1] * y[1]) ** 1.5
    d[0], d[1], d[2], d[3] = y[2], y[3], -y[0] / r3, -y[1] / r3


def kepler_exact(t):
    e = t
    for _ in range(100):
        e -= (e - ECC * math.sin(e) - t) / (1.0 - ECC * math.cos(e))
    s = math.sqrt(1.0 - ECC * ECC)
    r = 1.0 - ECC * math.cos(e)
    return [math.cos(e) - ECC, s * math.sin(e), -math.sin(e) / r, s * math.cos(e) / r]


PROBLEMS = {"arenstorf": arenstorf, "lotka": lotka, "vdp1": vdp1, "oscillator": oscillator, "kepler": kepler}


def relmax(a, b):
    s = max(abs(x) for x in b)
    return max(abs(x - y) for x, y in zip(a, b)) / s


def run(name, c, rtol, atol):
    return dopri5(PROBLEMS[name], (0.0, c["t_end"]), c["y0"], rtol=rtol, atol=atol)


def test_step_sequence_vs_scipy_rk45():
    data = load_reference("dopri5")["cases"]
    for name in PROBLEMS:
        c = data[name]
        for key, r in c["rk45"].items():
            res = run(name, c, r["rtol"], r["atol"])
            assert res.success, res.message
            m = min(len(r["t"]), len(res.t))
            et = max(abs(a - b) for a, b in zip(res.t[:m], r["t"][:m]))
            ey = relmax(res.y_final, r["y_final"])
            print(f"VERIFY dopri5 py_{name}_rtol{key} nsteps={res.nsteps} scipy_nsteps={r['nsteps']} nfev={res.nfev} "
                  f"scipy_nfev={r['nfev']} nrejected={res.nrejected} max_dt_vs_scipy={et:e} y_final_vs_scipy={ey:e}")
            assert res.nsteps == r["nsteps"] and res.nfev == r["nfev"], (name, key)
            assert et < TOL_T_SEQ and ey < TOL_Y_VS_RK45


def test_exact_solutions_and_invariant():
    data = load_reference("dopri5")["cases"]
    for rtol in (1e-3, 1e-6, 1e-9):
        atol = 1e-3 * rtol
        c = data["oscillator"]
        te = c["t_end"]
        ex = [math.exp(-0.1 * te) * math.cos(2 * te), -math.exp(-0.1 * te) * math.sin(2 * te)]
        e_osc = relmax(run("oscillator", c, rtol, atol).y_final, ex)
        c = data["kepler"]
        e_kep = relmax(run("kepler", c, rtol, atol).y_final, kepler_exact(c["t_end"]))
        c = data["lotka"]
        drift = abs(ham(run("lotka", c, rtol, atol).y_final) - c["H0"]) / abs(c["H0"])
        print(f"VERIFY dopri5 py_exact_rtol{rtol:e} oscillator_err={e_osc:e} err/rtol={e_osc / rtol:e} "
              f"kepler_err={e_kep:e} err/rtol={e_kep / rtol:e} lotka_H_drift={drift:e} drift/rtol={drift / rtol:e}")
        assert max(e_osc, e_kep, drift) < TOL_ERR_OVER_RTOL * rtol


def test_dop853_reference_and_arenstorf():
    data = load_reference("dopri5")["cases"]
    last = 0
    for rtol in (1e-3, 1e-6, 1e-9):
        for name in PROBLEMS:
            c = data[name]
            r = run(name, c, rtol, 1e-3 * rtol)
            assert r.success, r.message
            e = relmax(r.y_final, c["y_ref"])
            print(f"VERIFY dopri5 py_{name}_rtol{rtol:e} err_vs_dop853={e:e} err/rtol={e / rtol:e} nsteps={r.nsteps} nrejected={r.nrejected}")
            factor = TOL_ERR_OVER_RTOL_ARENSTORF if name == "arenstorf" else TOL_ERR_OVER_RTOL
            if not (name == "arenstorf" and rtol == 1e-3):   # too loose to follow the orbit
                assert e < factor * rtol, (name, rtol, e)
            if name == "arenstorf":
                assert r.nsteps > last
                last = r.nsteps


def test_stiffness_limits_the_step():
    s = load_reference("dopri5")["cases"]["stiffness"]
    prev = 0
    for row in s["rows"]:
        lam = row["lam"]
        r = dopri5(lambda t, y, d: d.__setitem__(0, -lam * (y[0] - math.cos(t))), (0.0, s["t_end"]), [0.0],
                   rtol=1e-6, atol=1e-9)
        assert r.success, r.message
        e = abs(r.y_final[0] - row["y_ref"])
        print(f"VERIFY dopri5 py_stiff_lam{lam:e} nsteps={r.nsteps} scipy_nsteps={row['rk45_nsteps']} "
              f"nrejected={r.nrejected} abs_err_vs_radau={e:e}")
        assert r.nsteps == row["rk45_nsteps"] and r.nsteps > prev and e < 1e-4
        prev = r.nsteps
    lam = 1e4
    r = dopri5(lambda t, y, d: d.__setitem__(0, -lam * (y[0] - math.cos(t))), (0.0, 2.0), [0.0], max_steps=500)
    print(f"VERIFY dopri5 py_stiff_budget success={r.success} message={r.message!r} t_reached={r.t[-1]:e}")
    assert not r.success and "max_steps" in r.message and r.t[-1] < 2.0


def test_options_and_time_dependence():
    f = lambda t, y, d: d.__setitem__(0, math.cos(t))  # noqa: E731
    r = dopri5(f, (0.0, 5.0), [0.0], rtol=1e-10, atol=1e-13)
    assert r.success and abs(r.y_final[0] - math.sin(5.0)) < 1e-8 and r.t[-1] == 5.0
    r = dopri5(f, (0.0, 1.0), [0.0], hmax=0.01)
    assert r.success and r.nsteps >= 100
    r = dopri5(f, (0.0, 1.0), [0.0], h0=1e-3)
    assert r.success and r.t[1] == 1e-3 and r.nfev == 1 + 6 * (r.nsteps + r.nrejected)
    r = dopri5(f, (0.0, 1.0), [0.0])
    assert r.nfev == 2 + 6 * (r.nsteps + r.nrejected)

    def g(t, y, d):
        d[0] = -y[0]
        d[1] = y[0] * 1e-9 - 10.0 * y[1]

    exact = 1e-9 / 9.0 * (math.exp(-1.0) - math.exp(-10.0))
    loose = dopri5(g, (0.0, 1.0), [1.0, 0.0], rtol=1e-6, atol=1e-6)
    tight = dopri5(g, (0.0, 1.0), [1.0, 0.0], rtol=1e-6, atol_vec=[1e-6, 1e-16])
    el, et = (abs(r.y_final[1] - exact) / exact for r in (loose, tight))
    print(f"VERIFY dopri5 py_atol_vec err_loose={el:e} nsteps={loose.nsteps} err_per_component={et:e} nsteps={tight.nsteps}")
    assert et < 1e-4 and et < el


def test_failures_are_reported():
    f = lambda t, y, d: d.__setitem__(0, y[0] * y[0])  # noqa: E731
    r = dopri5(f, (0.0, 2.0), [1.0])                    # finite-time blow-up at t = 1
    print(f"VERIFY dopri5 py_blowup success={r.success} message={r.message!r} t_reached={r.t[-1]:e}")
    assert not r.success and r.message and r.t[-1] < 2.0
    assert not dopri5(f, (1.0, 1.0), [1.0]).success
    assert not dopri5(f, (0.0, 1.0), []).success
    assert not dopri5(f, (0.0, 1.0), [1.0], rtol=0.0).success
    assert not dopri5(f, (0.0, 1.0), [1.0], atol_vec=[1e-8, 1e-8]).success
    assert not dopri5(f, (0.0, 1.0), [1.0], h0=-1.0).success
    r = dopri5(lambda t, y, d: d.__setitem__(0, -y[0]), (0.0, 100.0), [1.0], max_steps=5)
    assert not r.success and "max_steps" in r.message
    zero = lambda t, y, d: d.__setitem__(0, 0.0)  # noqa: E731
    assert not dopri5(zero, (0.0, 1.0), [0.0], atol=0.0).success
    r = dopri5(zero, (0.0, 1.0), [1.0])
    assert r.success and r.y_final[0] == 1.0, r.message


def test_nan_rhs_is_not_accepted():
    f = lambda t, y, d: d.__setitem__(0, float("nan") if t > 0.5 else -y[0])  # noqa: E731
    r = dopri5(f, (0.0, 1.0), [1.0])
    print(f"VERIFY dopri5 py_nan_rhs success={r.success} message={r.message!r}")
    assert not r.success
    assert all(math.isfinite(v) for row in r.y for v in row)


def test_python_vs_rust_example():
    py, rs = python_example("dopri5"), rust_example("dopri5")
    assert py["success"] and rs["success"] and py["message"] == rs["message"]
    for key in ("nsteps", "nrejected", "nfev"):
        assert py[key] == rs[key], key
    worst = 0.0
    for key in ("t_mid", "y_mid", "t_final", "y"):
        a = py[key] if isinstance(py[key], list) else [py[key]]
        b = rs[key] if isinstance(rs[key], list) else [rs[key]]
        worst = max(worst, max(abs(x - y) for x, y in zip(a, b)) / max(abs(x) for x in b))
    print(f"VERIFY dopri5 python_vs_rust max_rel_diff={worst:e} nsteps={py['nsteps']} nfev={py['nfev']}")
    assert worst < TOL_RUST_Y
