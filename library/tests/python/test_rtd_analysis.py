"""Tests for stats.rtd_analysis: Python vs SciPy/exact reference data, noisy data, failure modes, and Rust."""
import math

import numpy as np
from conftest import load_reference, python_example, rust_example

from agent_tools.stats.rtd_analysis import (CLOSED_CLOSED, DISPERSION, OPEN_OPEN, TANKS, e_dispersion, e_tanks,
                                            fit_dispersion_planes, fit_lsq, fit_moments, ln_gamma, rtd_moments,
                                            sigma2_dispersion, tail_fit)

TOL_MOMENTS_VS_SCIPY = 1e-9    # moments of a truncated record vs independent SciPy computation, relative
TOL_EXP_TAIL = 1e-5            # exponential record: tail extrapolation exact up to trapezoid error
TOL_TANKS_VAR_TAIL = 5e-2      # early-truncated tanks record with tail vs exact variance (tail only ~exponential)
TOL_DISP_VAR_TAIL = 1e-2       # same for dispersion records
TOL_MOMENT_INVERSION = 1e-9    # moment fit on exact moments
TOL_CC_SERIES = 1e-5           # closed-closed reference moments come from a series accurate to ~1e-6
TOL_MOMENT_FIT_TRUNC = 0.06    # moment-fit parameter from a truncated record vs truth
TOL_LSQ_VS_CURVEFIT = 1e-5
TOL_LSQ_EXACT = 1e-6
TOL_LSQ_NOISY = 5e-2
TOL_LNGAMMA = 1e-13
TOL_RUST = 1e-9                # Python vs Rust example, relative (float64 port; nfev must match exactly)


def rel(a, b):
    return abs(a - b) / abs(b)


def datasets():
    return load_reference("rtd_analysis")["cases"]["datasets"]


def test_ln_gamma_vs_scipy():
    g = load_reference("rtd_analysis")["cases"]["ln_gamma"]
    worst = max(abs(ln_gamma(x) - y) / max(abs(y), 1.0) for x, y in zip(g["x"], g["ref"]))
    print(f"VERIFY rtd_analysis py_ln_gamma max_rel_err_vs_scipy_gammaln={worst:e}")
    assert worst < TOL_LNGAMMA


def test_moments_of_truncated_records():
    for d in datasets():
        name, rm = d["name"], d["ref_moments"]
        t, C = np.array(d["t"]), np.array(d["C"])
        w, wo = rtd_moments(t, C), rtd_moments(t, C, tail=False)
        em, ev, ev0 = rel(w.mean, rm["mean_tail"]), rel(w.variance, rm["var_tail"]), rel(wo.variance, rm["var_notail"])
        print(f"VERIFY rtd_analysis py_{name} moments_rel_err_vs_scipy mean={em:e} var_tail={ev:e} var_notail={ev0:e}")
        assert max(em, ev, ev0) < TOL_MOMENTS_VS_SCIPY
        assert (rm["lam"] is None) == (w.tail_decay_time is None)
        if rm["lam"] is not None:
            assert rel(w.tail_decay_time, rm["lam"]) < 1e-9
        xm, xv = d["exact_mean"], d["exact_variance"]
        et, en = rel(w.variance, xv), rel(wo.variance, xv)
        print(f"VERIFY rtd_analysis py_{name} var_rel_err_vs_exact with_tail={et:e} without_tail={en:e} "
              f"mean_with_tail={rel(w.mean, xm):e}")
        if d["noise_rel_peak"] != 0:
            assert w.tail_decay_time is None   # negative samples in the tail disable the extrapolation
        elif name.startswith("cstr"):
            assert et < TOL_EXP_TAIL and rel(w.mean, xm) < TOL_EXP_TAIL
        else:
            assert et < (TOL_TANKS_VAR_TAIL if d["kind"] == "tanks" else TOL_DISP_VAR_TAIL) and et < en
        area = np.sum(0.5 * (w.e[1:] + w.e[:-1]) * np.diff(t))
        assert 0.5 < area < 1 + 1e-6


def test_exponential_with_pulse_correction():
    lam, w = 40.0, 10.0
    t = np.arange(40001) * 0.01
    c = np.where(t < w, (1 - np.exp(-t / lam)) / w, np.exp(-t / lam) * (math.exp(w / lam) - 1) / w)
    m = rtd_moments(t, c, pulse=w)
    em, ev = rel(m.mean, lam), rel(m.variance, lam * lam)
    print(f"VERIFY rtd_analysis py_pulse_cstr mean_rel_err={em:e} var_rel_err={ev:e}")
    assert em < 1e-4 and ev < 1e-4
    assert abs(rtd_moments(t, c).mean - lam - w / 2) < 1e-2 * lam


def test_two_plane_moment_difference_is_exact():
    u, D, x1, x2 = 0.3, 0.05, 20.0, 60.0
    t = np.linspace(1e-3, 2000, 400001)

    def c(x):
        return np.exp(-(x - u * t) ** 2 / (4 * D * t)) / np.sqrt(4 * np.pi * D * t)

    m1, m2 = rtd_moments(t, c(x1), tail=False), rtd_moments(t, c(x2), tail=False)
    p = fit_dispersion_planes(m1.mean, m1.variance, m2.mean, m2.variance, x2 - x1)
    eu, ed = rel(p["u"], u), rel(p["D_L"], D)
    print(f"VERIFY rtd_analysis py_two_plane u_rel_err={eu:e} D_L_rel_err={ed:e}")
    assert eu < 1e-3 and ed < 1e-3


def test_moment_inversions_on_exact_moments():
    for d in datasets():
        if d["noise_rel_peak"] != 0:
            continue
        name, kind, truth, tau = d["name"], d["kind"], d["param"], d["tau"]
        xm, xv = d["exact_mean"], d["exact_variance"]
        if kind == "tanks":
            fit, tol = fit_moments(xm, xv, TANKS), TOL_MOMENT_INVERSION
        elif kind == "oo":
            fit, tol = fit_moments(xm, xv, DISPERSION, OPEN_OPEN), TOL_MOMENT_INVERSION
        else:
            fit, tol = fit_moments(xm, xv, DISPERSION, CLOSED_CLOSED), TOL_CC_SERIES
        e = rel(fit.param, truth)
        print(f"VERIFY rtd_analysis py_{name} moment_inversion_exact param_rel_err={e:e}")
        assert fit.success and e < tol
        if kind == "oo":
            assert rel(fit.tau, tau) < TOL_MOMENT_INVERSION
        if kind != "tanks":
            e = rel(sigma2_dispersion(truth, OPEN_OPEN if kind == "oo" else CLOSED_CLOSED) * tau * tau, xv)
            print(f"VERIFY rtd_analysis py_{name} sigma2_formula_vs_quad rel_err={e:e}")
            assert e < (TOL_MOMENT_INVERSION if kind == "oo" else TOL_CC_SERIES)


def _moment_fit(d, m):
    if d["kind"] == "tanks":
        return fit_moments(m.mean, m.variance, TANKS)
    return fit_moments(m.mean, m.variance, DISPERSION, OPEN_OPEN if d["kind"] == "oo" else CLOSED_CLOSED)


def test_moment_fits_on_truncated_records():
    for d in datasets():
        if d["noise_rel_peak"] != 0:
            continue
        fit = _moment_fit(d, rtd_moments(np.array(d["t"]), np.array(d["C"])))
        e = rel(fit.param, d["param"])
        print(f"VERIFY rtd_analysis py_{d['name']} moment_fit_param_rel_err_vs_truth={e:e}")
        assert fit.success and e < TOL_MOMENT_FIT_TRUNC


def test_least_squares_vs_curve_fit_and_truth():
    for d in datasets():
        cf = d.get("curve_fit")
        if cf is None:
            continue
        noisy = d["noise_rel_peak"] != 0
        model = TANKS if d["kind"] == "tanks" else DISPERSION
        t, E = np.array(d["t"]), np.array(d["E"])
        free, fixed = fit_lsq(t, E, model), fit_lsq(t, E, model, tau=d["tau"])
        ep, et, ef = rel(free.param, cf["p"]), rel(free.tau, cf["tau"]), rel(fixed.param, cf["p_fixed_tau"])
        tr = max(rel(free.param, d["param"]), rel(free.tau, d["tau"]))
        print(f"VERIFY rtd_analysis py_{d['name']} lsq_vs_curve_fit param_rel={ep:e} tau_rel={et:e} "
              f"fixed_tau_param_rel={ef:e} vs_truth_rel={tr:e} nfev_free={free.nfev} nfev_fixed={fixed.nfev}")
        assert free.success and fixed.success
        if noisy:
            assert max(ep, et, ef) < TOL_LSQ_VS_CURVEFIT and tr < TOL_LSQ_NOISY
            assert abs(free.sse - cf["sse"]) < 1e-6 * cf["sse"]
        else:
            assert tr < TOL_LSQ_EXACT


def test_noisy_records_moments_fail_but_lsq_works():
    for d in datasets():
        if d["noise_rel_peak"] == 0:
            continue
        t = np.array(d["t"])
        mf = _moment_fit(d, rtd_moments(t, np.array(d["C"])))
        lf = fit_lsq(t, np.array(d["E"]), TANKS if d["kind"] == "tanks" else DISPERSION)
        em, el = rel(mf.param, d["param"]), rel(lf.param, d["param"])
        print(f"VERIFY rtd_analysis py_{d['name']} param_rel_err_vs_truth moments={em:e} lsq={el:e}")
        assert el < em


def test_failure_modes():
    assert tail_fit([0.0, 1.0, 2.0], [1.0, 0.5, 0.2]) is None
    t = np.arange(20.0)
    c = np.exp(-t / 3)
    assert abs(tail_fit(t, c) - 3.0) < 1e-9
    c[19] = 0.0
    assert tail_fit(t, c) is None
    assert tail_fit(t, np.exp(t / 3)) is None
    assert not fit_moments(10.0, 0.0, TANKS).success
    assert not fit_moments(10.0, float("nan"), TANKS).success
    assert not fit_moments(10.0, 100.0, DISPERSION, CLOSED_CLOSED).success
    tt = 0.8 + 0.4 * np.arange(4001) / 4000
    r = fit_lsq(tt, e_dispersion(tt, 2e4), DISPERSION, tau=1.0)   # Pe above the search range
    assert not r.success and "edge" in r.message
    assert not fit_lsq([0.0, 1.0], [0.0, 0.0], TANKS).success
    print("VERIFY rtd_analysis py_failure_modes all_ok=1")


def test_python_vs_rust_example():
    py, rs = python_example("rtd_analysis"), rust_example("rtd_analysis")
    assert py.keys() == rs.keys()
    worst = 0.0
    for k in py:
        if isinstance(py[k], bool):
            assert py[k] == rs[k], k
        elif k.startswith("nfev"):
            assert py[k] == rs[k], k
        else:
            worst = max(worst, rel(py[k], rs[k]) if rs[k] != 0 else abs(py[k]))
    print(f"VERIFY rtd_analysis python_vs_rust max_rel_diff={worst:e} nfev_lsq={py['nfev_lsq']}")
    assert worst < TOL_RUST
    # the example's physics: tanks N=4 recovered by both fits, Pe = 12 by lsq
    assert abs(py["N_lsq"] - 4) < 1e-4 and abs(py["Pe_lsq"] - 12) < 1e-6 and abs(py["tau_disp_lsq"] - 80) < 1e-6
