"""Tests for linalg.cholesky: Python vs NumPy/SciPy reference data, non-SPD failure, and Rust."""
import numpy as np
import pytest
from conftest import load_reference, python_example, rust_example

from agent_tools.linalg.cholesky import cholesky, cholesky_solve, solve_lower, solve_upper_t

TOL_L = 1e-12            # L vs NumPy, relative to max|L| (widened to TOL_X_FACTOR*eps*cond if larger)
TOL_X_FACTOR = 10.0      # x vs SciPy: factor * eps * cond
TOL_RUST = 1e-13         # Python vs Rust example, relative
EPS = np.finfo(float).eps


def rel(x, ref):
    return np.max(np.abs(np.asarray(x) - ref)) / np.max(np.abs(ref))


@pytest.mark.parametrize("group", ["spd", "hilbert"])
def test_against_numpy_scipy(group):
    for c in load_reference("cholesky")["cases"][group]:
        n = c["n"]
        A = np.array(c["A"]).reshape(n, n)
        r = cholesky(A)
        assert r.success, r.message
        el = rel(r.l, np.array(c["L"]).reshape(n, n))
        x = cholesky_solve(r.l, np.array(c["b"]))
        ex = rel(x, np.array(c["x"]))
        ea = np.max(np.abs(r.l @ r.l.T - A)) / np.max(np.abs(A))
        print(f"VERIFY cholesky py_{group}_n{n} cond={c['cond']:e} rel_err_L_vs_numpy={el:e} "
              f"rel_err_x_vs_scipy={ex:e} rel_err_LLt_vs_A={ea:e}")
        assert el < max(TOL_L, TOL_X_FACTOR * EPS * c["cond"])
        assert ex < TOL_X_FACTOR * EPS * max(c["cond"], 1.0)


def test_non_spd_reported():
    cases = load_reference("cholesky")["cases"]["not_spd"]
    for c in cases:
        r = cholesky(np.array(c["A"]).reshape(c["n"], c["n"]))
        assert not r.success and "not positive definite" in r.message
    assert not cholesky(np.array([[np.nan]])).success
    assert not cholesky(np.array([[np.inf]])).success
    with pytest.raises(ValueError):
        cholesky(np.ones((2, 3)))
    print(f"VERIFY cholesky py_not_spd cases_reported={len(cases) + 1}")


def test_triangular_solves_multiple_rhs():
    l = np.array([[2.0, 0.0], [1.0, 3.0]])
    assert np.array_equal(solve_lower(l, np.array([[2.0, 4.0], [7.0, 5.0]])), [[1.0, 2.0], [2.0, 1.0]])
    assert np.array_equal(solve_upper_t(l, np.array([[3.0, 5.0], [3.0, 3.0]])), [[1.0, 2.0], [1.0, 1.0]])
    print("VERIFY cholesky py_triangular_multi_rhs exact_match=1")


def test_python_vs_rust_example():
    py, rs = python_example("cholesky"), rust_example("cholesky")
    assert py["success"] and rs["success"] and not py["bad_success"] and not rs["bad_success"]
    el, ex = rel(py["l"], np.array(rs["l"])), rel(py["x"], np.array(rs["x"]))
    print(f"VERIFY cholesky python_vs_rust rel_diff_L={el:e} rel_diff_x={ex:e}")
    assert el < TOL_RUST and ex < TOL_RUST
