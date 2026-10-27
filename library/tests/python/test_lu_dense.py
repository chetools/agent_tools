"""Tests for linalg.lu_dense: Python vs SciPy reference data, Dual derivative, and Rust."""
import numpy as np
from conftest import load_reference, python_example, rust_example

from agent_tools.autodiff.dual import Dual
from agent_tools.linalg.lu_dense import lu_factor, lu_solve, solve

TOL_WELL = 1e-12          # n = 1..50 well-conditioned, solution vs SciPy, relative to max|x|
TOL_FACTORS = 1e-12       # LU factors vs SciPy (n <= 10)
TOL_ILL_FACTOR = 10.0     # ill-conditioned: error <= factor * eps * cond
TOL_BACKWARD = 1e-14      # ||Ax-b|| / (||A|| ||x|| + ||b||), inf norms
TOL_DUAL = 1e-11          # dx vs analytic, relative to max|dx|
TOL_RUST = 1e-13          # Python vs Rust example, relative (same op order; expect ~0)
EPS = np.finfo(float).eps


def rel(x, ref):
    return np.max(np.abs(np.asarray(x) - ref)) / np.max(np.abs(ref))


def backward(A, x, b):
    A, x, b = np.asarray(A), np.asarray(x), np.asarray(b)
    return np.max(np.abs(A @ x - b)) / (np.linalg.norm(A, np.inf) * np.max(np.abs(x)) + np.max(np.abs(b)))


def test_well_conditioned_n1_to_50():
    ex = ef = eb = 0.0
    for c in load_reference("lu_dense")["cases"]["well"]:
        n = c["n"]
        A, b = np.array(c["A"]).reshape(n, n), np.array(c["b"])
        f = lu_factor(c["A"], n)
        assert f.success
        x = lu_solve(f, c["b"]).x
        ex = max(ex, rel(x, np.array(c["x"])))
        eb = max(eb, backward(A, x, b))
        if "lu" in c:
            assert f.piv == c["piv"]
            ef = max(ef, rel(f.lu, np.array(c["lu"])))
    print(f"VERIFY lu_dense py_well max_rel_err_x_vs_scipy={ex:e}")
    print(f"VERIFY lu_dense py_well max_rel_err_lu_vs_scipy_n<=10={ef:e}")
    print(f"VERIFY lu_dense py_well max_backward_error={eb:e}")
    assert ex < TOL_WELL and ef < TOL_FACTORS and eb < TOL_BACKWARD


def test_ill_conditioned():
    for target in (1e4, 1e8, 1e12):
        worst = eb = 0.0
        for c in [c for c in load_reference("lu_dense")["cases"]["ill"] if c["target_cond"] == target]:
            n = c["n"]
            s = solve(c["A"], c["b"], n)
            assert s.success
            e = rel(s.x, np.array(c["x"]))
            worst = max(worst, e)
            eb = max(eb, backward(np.array(c["A"]).reshape(n, n), s.x, np.array(c["b"])))
            assert e < TOL_ILL_FACTOR * EPS * c["cond"]
        print(f"VERIFY lu_dense py_ill_cond_{target:e} max_rel_err_x_vs_scipy={worst:e}")
        print(f"VERIFY lu_dense py_ill_cond_{target:e} max_backward_error={eb:e}")
        assert eb < TOL_BACKWARD


def test_singular_is_reported():
    cases = load_reference("lu_dense")["cases"]["singular"]
    for c in cases:
        f = lu_factor(c["A"], c["n"])
        assert not f.success and "singular" in f.message
        s = lu_solve(f, [1.0] * c["n"])
        assert not s.success and s.x == []
    assert not lu_factor([1.0, float("nan"), 0.0, 1.0], 2).success
    print(f"VERIFY lu_dense py_singular cases_reported={len(cases) + 1}")


def test_dual_solution_derivative():
    for c in load_reference("lu_dense")["cases"]["param"]:
        n, p = c["n"], Dual(c["p0"], 1.0)
        A = [a0 + a1 * p for a0, a1 in zip(c["A0"], c["A1"])]
        b = [b0 + b1 * p for b0, b1 in zip(c["b0"], c["b1"])]
        s = solve(A, b, n)
        assert s.success
        ex = rel([d.v for d in s.x], np.array(c["x"]))
        edx = rel([d.d for d in s.x], np.array(c["dx"]))
        print(f"VERIFY lu_dense py_dual_n{n} rel_err_x={ex:e}")
        print(f"VERIFY lu_dense py_dual_n{n} rel_err_dx_vs_analytic={edx:e}")
        assert ex < TOL_WELL and edx < TOL_DUAL


def test_python_vs_rust_example():
    py, rs = python_example("lu_dense"), rust_example("lu_dense")
    assert py["piv"] == rs["piv"] and py["success"] == rs["success"]
    assert py["singular_success"] is False and rs["singular_success"] is False
    assert py["singular_message"] == rs["singular_message"]
    e = rel(py["x"], np.array(rs["x"]))
    print(f"VERIFY lu_dense python_vs_rust rel_diff_x={e:e}")
    assert e < TOL_RUST
