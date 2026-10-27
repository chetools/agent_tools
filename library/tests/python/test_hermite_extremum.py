"""Tests for interpolate.hermite_extremum and interpolate.cubic_hermite: Python vs reference data
(exact cubic roots; SciPy CubicHermiteSpline + bounded minimize_scalar), edge cases, and Rust."""
import math

import numpy as np
import pytest
from conftest import load_reference, python_example, rust_example

from agent_tools.interpolate.cubic_hermite import cubic_hermite
from agent_tools.interpolate.hermite_extremum import hermite_extremum

TOL_CUBIC_VALUE = 1e-13   # exact cubic: extremum value (abs)
TOL_CUBIC_T = 1e-12       # exact cubic: extremum location (abs)
TOL_REF_VALUE = 1e-12     # sine / Gaussian vs SciPy dense+refine, value (abs)
TOL_REF_T = 1e-7          # same, location (limited by minimize_scalar)
TOL_EVAL = 1e-13          # evaluator vs CubicHermiteSpline (abs)
ORDER_RANGE = (3.7, 4.3)  # observed order of the Gaussian peak-value error, finest spacings
TOL_RUST = 1e-13          # Python vs Rust example (abs; values are O(1))


def arr(c):
    return np.array(c["t"]), np.array(c["y"]), np.array(c["dy"])


def test_exact_cubic():
    c = load_reference("hermite_extremum")["cases"]["cubic"]
    t, y, dy = arr(c)
    for kind in ("max", "min"):
        e = hermite_extremum(t, y, dy, kind)
        et, ev = abs(e.t - c[kind]["t"]), abs(e.value - c[kind]["value"])
        print(f"VERIFY hermite_extremum py_cubic_{kind} err_t={et:e} err_value={ev:e} interval={e.interval}")
        assert e.success and et < TOL_CUBIC_T and ev < TOL_CUBIC_VALUE
    x = np.linspace(0.0, 3.6, 401)
    worst = np.max(np.abs(cubic_hermite(t, y, dy, x) - np.polyval(c["coef"], x)))
    print(f"VERIFY hermite_extremum py_cubic_reproduce max_abs_err={worst:e}")
    assert worst < TOL_CUBIC_VALUE


@pytest.mark.parametrize("kind", ["max", "min"])
def test_sine_vs_scipy(kind):
    c = load_reference("hermite_extremum")["cases"]["sine"]
    e = hermite_extremum(*arr(c), kind)
    et, ev = abs(e.t - c[kind]["t"]), abs(e.value - c[kind]["value"])
    print(f"VERIFY hermite_extremum py_sine_{kind} err_t={et:e} err_value={ev:e} interval={e.interval}")
    assert et < TOL_REF_T and ev < TOL_REF_VALUE and e.interval == c[kind]["interval"]


def test_gaussian_convergence():
    c = load_reference("hermite_extremum")["cases"]["gaussian"]
    errs = []
    for run in c["runs"]:
        e = hermite_extremum(*arr(run), "max")
        dt, dv = abs(e.t - run["ref"]["t"]), abs(e.value - run["ref"]["value"])
        err_v, err_t = abs(c["true_value"] - e.value), abs(e.t - c["t0"])
        print(f"VERIFY hermite_extremum py_gaussian_n{run['n']} h={run['h']:.5e} err_value_vs_true={err_v:e} "
              f"err_t_vs_true={err_t:e} diff_value_vs_scipy={dv:e} diff_t_vs_scipy={dt:e}")
        assert dt < TOL_REF_T and dv < TOL_REF_VALUE
        errs.append((run["h"], err_v))
    for (h0, e0), (h1, e1) in zip(errs[-3:-1], errs[-2:]):
        order = math.log(e0 / e1) / math.log(h0 / h1)
        print(f"VERIFY hermite_extremum py_gaussian_order h={h0:.5e}->{h1:.5e} observed_order={order:.3f}")
        assert ORDER_RANGE[0] < order < ORDER_RANGE[1]


def test_evaluator_vs_scipy():
    c = load_reference("hermite_extremum")["cases"]["evaluator"]
    t, y, dy = arr(c)
    worst = np.max(np.abs(cubic_hermite(t, y, dy, np.array(c["x"])) - np.array(c["ref"])))
    print(f"VERIFY cubic_hermite py_evaluator max_abs_err_vs_scipy={worst:e} npoints={len(c['x'])}")
    assert worst < TOL_EVAL
    assert np.isscalar(float(cubic_hermite(t, y, dy, 1.0)))


def test_edge_cases():
    mx = lambda t, y, dy: hermite_extremum(t, y, dy, "max")  # noqa: E731
    mn = lambda t, y, dy: hermite_extremum(t, y, dy, "min")  # noqa: E731
    t, y, dy = [0.0, 1.0, 3.0], [0.0, 1.0, 4.0], [1.0, 1.0, 2.0]
    a, b = mx(t, y, dy), mn(t, y, dy)
    assert a.success and a.t == 3.0 and a.value == 4.0 and a.interval == 1
    assert b.t == 0.0 and b.value == 0.0 and b.interval == 0
    a = mx([0, 1, 2], [0, 1, 0], [1, 0, -1])
    assert a.t == 1.0 and a.value == 1.0 and a.interval == 1
    a = mx([0, 1, 2], [1, 1, 1], [0, 0, 0])
    assert a.t == 0.0 and a.interval == 0
    a = mx([0, 1, 2, 3, 4], [0, 1, 0, 1, 0], [1, 0, -1, 0, -1])
    assert a.t == 1.0 and a.interval == 1
    a = mx([-1, 1], [-1, 1], [3, 3])  # inflection with zero slope is not an extremum
    assert a.t == 1.0 and a.value == 1.0
    a = mx([0, 2], [1, 5], [2, 2])    # linear data
    assert a.t == 2.0 and a.value == 5.0
    a = mx([0, 1], [-0.09, -0.49], [0.6, -1.4])  # quadratic piece, max at 0.3
    assert abs(a.t - 0.3) < 1e-15 and abs(a.value) < 1e-15 and a.interval == 0
    a = mn([0, 1], [0.09, 0.49], [-0.6, 1.4])
    assert abs(a.t - 0.3) < 1e-15 and abs(a.value) < 1e-15
    for t, y, dy in [([0.0], [0.0], [0.0]), ([0, 1], [0], [0, 0]), ([0, 1, 1], [0] * 3, [0] * 3),
                     ([1, 0], [0] * 2, [0] * 2), ([0, 1], [np.nan, 0], [0, 0]), ([0, np.inf], [0, 0], [0, 0])]:
        e = mx(t, y, dy)
        assert not e.success and math.isnan(e.t) and math.isnan(e.value) and e.message != "success"
    with pytest.raises(ValueError):
        hermite_extremum([0, 1], [0, 1], [0, 0], "median")
    print("VERIFY hermite_extremum py_edge_cases checks_passed=1")


def test_python_vs_rust_example():
    py, rs = python_example("hermite_extremum"), rust_example("hermite_extremum")
    diffs = {k: abs(py[k] - rs[k]) for k in ("max_t", "max_value", "min_t", "min_value", "tie_t")}
    diffs["eval"] = float(np.max(np.abs(np.array(py["eval"]) - np.array(rs["eval"]))))
    print("VERIFY hermite_extremum python_vs_rust " + " ".join(f"diff_{k}={v:e}" for k, v in diffs.items()))
    assert all(v < TOL_RUST for v in diffs.values())
    for k in ("max_interval", "min_interval", "tie_interval", "bad_success"):
        assert py[k] == rs[k]
    assert py["bad_success"] is False
