"""Tests for optimize.bfgs: Python vs exact minimizers and SciPy BFGS reference data, failure modes, and Rust."""
import numpy as np
from conftest import load_reference, python_example, rust_example

from agent_tools.optimize.bfgs import BfgsOptions, bfgs

TOL_X_EXACT = 1e-8      # max |x - x*| vs the exact minimizer
TOL_F_EXACT = 1e-15     # |f - f*| vs the exact value (absolute)
TOL_X_SCIPY = 1e-8      # max |x - x_scipy|
NFEV_FACTOR = 3.0       # cost guard: nfev at most this multiple of SciPy's nfev
TOL_GRAD_FD = 1e-6      # analytic gradient vs central differences, relative to max|g|
TOL_ITERATES = 1e-12    # Python vs Rust: every evaluated point, max abs difference
GTOL = 1e-10


def objective(c):
    p = c["params"]
    if c["kind"] == "quadratic":
        n = len(p["b"])
        A, b = np.array(p["A"]).reshape(n, n), np.array(p["b"])

        def fg(x, g):
            g[:] = A @ x - b
            return float(0.5 * x @ A @ x - b @ x)
    else:
        a = p["a"]

        def fg(x, g):
            u, v = x[0::2], x[1::2]
            t = v - u * u
            g[0::2] = -4 * a * u * t - 2 * (1 - u)
            g[1::2] = 2 * a * t
            return float(np.sum(a * t * t + (1 - u) ** 2))
    return fg


def test_against_exact_and_scipy():
    for name, c in load_reference("bfgs")["cases"].items():
        s = c["scipy"]
        res = bfgs(objective(c), c["x0"], BfgsOptions(max_iter=1000, gtol=GTOL))
        assert res.success, (name, res.message)
        ex = np.max(np.abs(res.x - c["x_exact"]))
        ef = abs(res.f - c["f_exact"])
        es = np.max(np.abs(res.x - s["x"]))
        print(f"VERIFY bfgs py_{name} max_err_x_vs_exact={ex:e} abs_err_f_vs_exact={ef:e} "
              f"max_diff_x_vs_scipy={es:e} grad_norm={res.grad_norm:e} nit={res.nit} nfev={res.nfev} "
              f"scipy_nit={s['nit']} scipy_nfev={s['nfev']}")
        assert res.grad_norm < GTOL
        assert ex < TOL_X_EXACT and ef < TOL_F_EXACT and es < TOL_X_SCIPY, name
        assert res.nfev <= NFEV_FACTOR * s["nfev"], name


def test_test_gradients_match_finite_differences():
    # the analytic gradients of the test functions, on both sides of the valley (x < 0 and x > 0)
    rng = np.random.RandomState(1)
    for name, c in load_reference("bfgs")["cases"].items():
        fg = objective(c)
        n = len(c["x0"])
        for x in (np.array(c["x0"]), rng.randn(n), np.ones(n) + 0.1 * rng.randn(n)):
            g = np.zeros(n)
            fg(x, g)
            fd = np.empty(n)
            for i in range(n):
                h = 1e-6 * max(1.0, abs(x[i]))
                xp, xm = x.copy(), x.copy()
                xp[i] += h
                xm[i] -= h
                fd[i] = (fg(xp, np.zeros(n)) - fg(xm, np.zeros(n))) / (2 * h)
            err = np.max(np.abs(g - fd)) / max(1.0, np.max(np.abs(g)))
            assert err < TOL_GRAD_FD, (name, err)
    print("VERIFY bfgs py_gradient_check central_difference_ok=1")


def test_failures_are_reported():
    o = BfgsOptions()
    assert (o.max_iter, o.gtol) == (200, 1e-5)

    def ros(x, g):
        g[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
        g[1] = 200 * (x[1] - x[0] ** 2)
        return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    r = bfgs(ros, [-1.2, 1.0], BfgsOptions(max_iter=5, gtol=1e-10))
    assert not r.success and r.nit == 5 and "max_iter" in r.message

    def wrong_sign(x, g):
        g[0] = -2 * x[0]
        return x[0] ** 2

    r = bfgs(wrong_sign, [1.0])
    assert not r.success and "line search failed" in r.message

    def nan_f(x, g):
        g[0] = 0.0
        return float("nan")

    r = bfgs(nan_f, [1.0])
    assert not r.success and "non-finite" in r.message

    def x_minus_log(x, g):                      # +inf outside x > 0: infeasible trial points rejected
        if x[0] <= 0:
            return float("inf")
        g[0] = 1 - 1 / x[0]
        return x[0] - np.log(x[0])

    r = bfgs(x_minus_log, [0.1], BfgsOptions(max_iter=200, gtol=1e-8))
    assert r.success and abs(r.x[0] - 1) < 1e-7
    print("VERIFY bfgs py_failure_modes max_iter,line_search,nonfinite_x0,infinite_f_rejected=ok")


def test_python_vs_rust_example():
    py, rs = python_example("bfgs"), rust_example("bfgs")
    for k in ("rosenbrock2", "ext_rosenbrock10"):
        a, b = py[k], rs[k]
        assert a["success"] and b["success"]
        assert (a["nit"], a["nfev"]) == (b["nit"], b["nfev"]) and len(a["xs"]) == len(b["xs"])
        dx = np.max(np.abs(np.array(a["xs"]) - np.array(b["xs"])))
        dxf = np.max(np.abs(np.array(a["x"]) - np.array(b["x"])))
        print(f"VERIFY bfgs python_vs_rust_{k} max_diff_all_iterates={dx:e} max_diff_x={dxf:e} "
              f"diff_f={abs(a['f'] - b['f']):e} nit={a['nit']} nfev={a['nfev']}")
        assert dx < TOL_ITERATES and dxf < TOL_ITERATES
