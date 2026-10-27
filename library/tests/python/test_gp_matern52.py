"""Tests for stats.gp_matern52: kernel vs the Bessel-function Matern formula, likelihood and gradient vs an
independent dense implementation and finite differences, fits vs SciPy L-BFGS-B, predictions vs a dense
solve, interpolation, and Python vs Rust."""
import math

import numpy as np
import pytest
from conftest import load_reference, python_example, rust_example

from agent_tools.stats.gp_matern52 import (GpOptions, gp_build, gp_fit, matern52, nll_grad, to_theta,
                                           to_z)

TOL_KERNEL = 1e-13         # relative to sf2: closed form vs Bessel-function Matern
TOL_NLL = 1e-9             # |nll - nll_ref| (dimensionless), well-conditioned case
TOL_NLL_HARD = 1e-8        # same, hard case (cond(K) ~ 1e8)
TOL_GRAD = 1e-7            # max |grad - grad_fd_ref| relative to max(1, max|grad|)
TOL_GRAD_HARD = 1e-5       # same, hard case (the 4th-order FD reference is noisy there)
TOL_FIT_NLL = 1e-7         # library NLL may exceed the best L-BFGS-B NLL by at most this
TOL_FIT_THETA = 1e-3       # max |theta - theta_ref| at the optimum (well-determined case only)
TOL_PRED = 1e-9            # predictions vs dense solve, relative to max(1, max|value|) (y units)
TOL_INTERP = 1e-5          # noise-free interpolation: |mean - y| (y units)
TOL_INTERP_VAR = 1e-6      # noise-free interpolation: variance at the data, relative to signal_sd^2
TOL_PY_RS = 1e-9           # Python vs Rust example, relative
TOL_LML = 1e-8             # -lml - n ln ys vs the optimizer's nll_std (two factorizations; cond(K) ~ 1e8 in hard5d)


def data(c):
    X = np.array(c["x"]).reshape(-1, c["d"])
    return X, np.array(c["y"])


def test_kernel_vs_bessel():
    c = load_reference("gp_matern52")["cases"]["kernel"]
    a, b = np.array(c["a"]), np.array(c["b"])
    k = np.array([matern52(a[i:i + 1], b[i:i + 1], c["ell"], c["sf2"])[0, 0] for i in range(len(a))])
    err = np.max(np.abs(k - c["k"])) / c["sf2"]
    e0 = abs(matern52(a[:1], a[:1], c["ell"], c["sf2"])[0, 0] - c["k_zero_distance"])
    print(f"VERIFY gp_matern52 kernel max_rel_err_vs_bessel={err:e} zero_distance_err={e0:e}")
    assert err < TOL_KERNEL and e0 == 0.0


@pytest.mark.parametrize("name", ["smooth2d", "hard5d"])
def test_nll_and_gradient_vs_reference(name):
    c = load_reference("gp_matern52")["cases"][name]
    X, y = data(c)
    yz = (y - c["y_mean"]) / c["y_scale"]
    tn, tg = (TOL_NLL, TOL_GRAD) if name == "smooth2d" else (TOL_NLL_HARD, TOL_GRAD_HARD)
    en = eg = 0.0
    for t in c["theta_checks"]:
        f, g = nll_grad(X, yz, t["theta"], t["fixed_noise_var"])
        en = max(en, abs(f - t["nll_std"]))
        eg = max(eg, np.max(np.abs(g - t["grad_fd"])) / max(1.0, np.max(np.abs(g))))
    print(f"VERIFY gp_matern52 py_{name} max_abs_err_nll={en:e} max_rel_err_grad_vs_fd={eg:e} "
          f"n_thetas={len(c['theta_checks'])}")
    assert en < tn and eg < tg


def test_gradient_central_differences_both_sides():
    # own gradient vs central differences of own NLL, at short and long length scales (both sides of the
    # optimum), fitted and fixed noise; and the tanh transform derivative
    c = load_reference("gp_matern52")["cases"]["smooth2d"]
    X, y = data(c)
    yz = (y - y.mean()) / y.std()
    worst = 0.0
    for theta, fv in (([-2.5, -2.0, 0.1, -3.0], None), ([1.0, 1.2, 0.7, -0.5], None), ([-1.0, -0.3, 0.5], 1e-3)):
        theta = np.array(theta)
        _, g = nll_grad(X, yz, theta, fv)
        h = 1e-6
        fd = np.array([(nll_grad(X, yz, theta + h * e, fv)[0] - nll_grad(X, yz, theta - h * e, fv)[0]) / (2 * h)
                       for e in np.eye(len(theta))])
        worst = max(worst, np.max(np.abs(g - fd)) / max(1.0, np.max(np.abs(fd))))
    lo, hi = np.array([-4.0, -1.0]), np.array([2.0, 3.0])
    z = np.array([-3.0, 2.5])
    _, dth = to_theta(z, lo, hi)
    fdt = (to_theta(z + 1e-6, lo, hi)[0] - to_theta(z - 1e-6, lo, hi)[0]) / 2e-6
    et = np.max(np.abs(dth - fdt))
    rt = np.max(np.abs(to_z(to_theta(z, lo, hi)[0], lo, hi) - z))
    print(f"VERIFY gp_matern52 py_gradient_fd max_rel_err={worst:e} transform_deriv_err={et:e} roundtrip_err={rt:e}")
    assert worst < 1e-6 and et < 1e-9 and rt < 1e-12


@pytest.mark.parametrize("name", ["smooth2d", "smooth2d_fixed_noise", "hard5d"])
def test_fit_vs_scipy_lbfgsb(name):
    cases = load_reference("gp_matern52")["cases"]
    c = cases["smooth2d" if name.startswith("smooth2d") else name]
    X, y = data(c)
    if name == "smooth2d":
        ref, noise = c, None
    elif name == "smooth2d_fixed_noise":
        ref, noise = c["fixed_noise"], c["fixed_noise"]["noise_sd"]
    else:
        ref, noise = c, c["noise_sd"]
    r = gp_fit(X, y, GpOptions(noise_sd=noise))
    best = ref["best"]["nll_std"]
    diff = r.nll_std - best
    ref_starts = [f["nll_std"] for f in ref["lbfgsb_fits"]]
    m = r.model
    theta = list(np.log(m.length_scales)) + [math.log(m.signal_sd / m.y_scale)]
    if noise is None:
        theta.append(math.log(m.noise_sd / m.y_scale))
    dth = np.max(np.abs(np.array(theta) - ref["best"]["theta"]))
    lml_ok = abs(-m.log_marginal_likelihood - len(y) * math.log(m.y_scale) - r.nll_std)
    print(f"VERIFY gp_matern52 py_fit_{name} nll_minus_lbfgsb_best={diff:e} max_abs_diff_theta={dth:e} "
          f"nfev={r.nfev} nit={r.nit} success={r.success} lbfgsb_nfev={sum(f['nfev'] for f in ref['lbfgsb_fits'])} "
          f"lml_consistency={lml_ok:e} start_nll={[round(v, 9) for v in r.start_nll]} "
          f"lbfgsb_start_nll={[round(v, 9) for v in ref_starts]}")
    assert diff < TOL_FIT_NLL and lml_ok < TOL_LML
    if name != "hard5d":
        assert r.success and dth < TOL_FIT_THETA


def test_predict_vs_dense_solve():
    cases = load_reference("gp_matern52")["cases"]
    c = cases["smooth2d"]
    X, y = data(c)
    f = c["fitted"]
    gp = gp_build(X, y, f["ell"], f["signal_sd"], f["noise_sd"])
    G = np.array(c["test_points"]).reshape(-1, 2)
    mean, var = gp.predict(G)
    cov = gp.posterior_cov(G[:4], G[3:])
    em = np.max(np.abs(mean - c["pred_mean"])) / max(1.0, np.max(np.abs(c["pred_mean"])))
    ev = np.max(np.abs(var - c["pred_var"])) / max(1.0, np.max(np.abs(c["pred_var"])))
    ec = np.max(np.abs(cov.ravel() - c["cov_ab"])) / max(1.0, np.max(np.abs(c["cov_ab"])))
    el = abs(gp.log_marginal_likelihood - f["log_marginal_likelihood"])
    c1 = cases["interp1d"]
    X1, y1 = data(c1)
    gp1 = gp_build(X1, y1, c1["ell"], c1["signal_sd"], c1["noise_sd"])
    m1, v1 = gp1.predict(np.array(c1["test_points"]).reshape(-1, 1))
    e1 = max(np.max(np.abs(m1 - c1["pred_mean"])) / max(1.0, np.max(np.abs(c1["pred_mean"]))),
             np.max(np.abs(v1 - c1["pred_var"])) / max(1.0, np.max(np.abs(c1["pred_var"]))))
    print(f"VERIFY gp_matern52 py_predict_smooth2d rel_err_mean={em:e} rel_err_var={ev:e} rel_err_cov={ec:e} "
          f"abs_err_lml={el:e}")
    print(f"VERIFY gp_matern52 py_predict_interp1d rel_err_mean_var={e1:e}")
    assert max(em, ev, ec, e1) < TOL_PRED and el < 1e-6


def test_noise_free_interpolation():
    c = load_reference("gp_matern52")["cases"]["interp1d"]
    X, y = data(c)
    gp = gp_build(X, y, c["ell"], c["signal_sd"], 0.0)
    mean, var = gp.predict(X)
    em = np.max(np.abs(mean - y))
    ev = np.max(var) / c["signal_sd"] ** 2
    print(f"VERIFY gp_matern52 py_interpolation max_abs_err_mean_at_data={em:e} max_rel_var_at_data={ev:e}")
    assert em < TOL_INTERP and ev < TOL_INTERP_VAR


def test_options_and_failures():
    o = GpOptions()
    assert o.noise_sd is None and o.length_starts == (0.1, 0.3, 1.0) and o.n_random_starts == 0
    assert abs(o.log_length_bounds[0] - math.log(0.02)) < 1e-15 and o.bfgs.gtol == 1e-5
    c = load_reference("gp_matern52")["cases"]["smooth2d"]
    X, y = data(c)
    a = gp_fit(X, y, GpOptions(length_starts=(), n_random_starts=3, seed=9))
    b = gp_fit(X, y, GpOptions(length_starts=(), n_random_starts=3, seed=9))
    assert a.start_nll == b.start_nll                         # deterministic random starts
    with pytest.raises(ValueError):
        gp_fit(X, y, GpOptions(length_starts=(), n_random_starts=0))
    with pytest.raises(RuntimeError):                         # duplicate point, no noise: K singular
        gp_build(np.vstack([X, X[:1]]), np.r_[y, y[0]], [1.0, 1.0], 1.0, 0.0, standardization=(0.0, 1e-6))
    print("VERIFY gp_matern52 py_options_failures defaults,seeded_restarts,no_start,singular_build=ok")


def test_python_vs_rust_example():
    py, rs = python_example("gp_matern52"), rust_example("gp_matern52")
    assert (py["best_start"], py["nfev"], py["nit"], py["success"]) == (rs["best_start"], rs["nfev"], rs["nit"], rs["success"])
    worst = 0.0
    for k in ("length_scales", "signal_sd", "noise_sd", "nll_std", "log_marginal_likelihood", "start_nll", "mean",
              "var", "cov"):
        a, b = np.atleast_1d(py[k]), np.atleast_1d(rs[k])
        worst = max(worst, np.max(np.abs(a - b) / np.maximum(1e-3, np.abs(b))))
    print(f"VERIFY gp_matern52 python_vs_rust max_rel_diff={worst:e} nfev={py['nfev']} nit={py['nit']}")
    assert worst < TOL_PY_RS
